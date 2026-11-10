"""Offline tests for the H5 submission notebook (CPU, no network). Tests that need the competition train.csv
(ENGS108_DATA_DIR) skip when it is absent; the notebook execution tests stub torch/transformers/peft and the Qwen
forward and need no train.csv."""
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "exp03" / "stylo"))
sys.path.insert(0, str(REPO / "exp03" / "probe"))
import build_notebook as B  # noqa: E402
import h5_core as C  # noqa: E402
import protocol as P  # noqa: E402
from probe import ProbeModel  # noqa: E402

CFG = json.loads((HERE / "config.json").read_text())
ARMH = REPO / "exp03" / "qlora_armh" / "artifacts"
REF = HERE / "artifacts" / "h5_dev_reference.csv"
HAS_DATA = (Path(os.environ.get("ENGS108_DATA_DIR", "/kaggle/input/competitions/name-that-llm")) / "train.csv").exists()


def nb_code():
    nb = json.loads((HERE / B.NB_NAME).read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_in_sync_with_generator():
    assert (HERE / B.NB_NAME).read_text() == B.build()
    assert (HERE / "kernel-metadata.json").read_text() == B.kernel_metadata()


def test_notebook_inlines_repo_sources_verbatim():
    code = "\n".join(nb_code())
    assert B.extract(B.SRC["armh_core"], ["pooled_forward"]) in code
    assert B.extract(B.SRC["probe"], ["ProbeModel"]) in code
    assert B.extract(B.SRC["protocol"], ["log_proba", "logprob_blend"]).split("\n\n\n")[0] in code
    assert B.SRC["features"].read_text() in code
    assert B.h5_core_source() in code and "<repo-imports>" not in code
    for cell in nb_code():
        compile(cell, "cell", "exec")


def test_inference_only_and_frozen_values():
    code = "\n".join(nb_code())
    for banned in ("Trainer(", ".train()", "backward(", "optimizer", "fit_probe", ".fit("):
        assert banned not in code, banned
    assert "prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)" in code
    assert "is_trainable=False" in code
    m = CFG["model"]
    assert (m["max_length"], m["batch_size"], len(CFG["class_names"])) == (512, 8, 18)
    assert m["quantization"] == {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "float16",
                                 "bnb_4bit_use_double_quant": True}
    assert CFG["blend"]["weights"] == [1.0, 1.0, 1.0] and tuple(CFG["blend"]["components"]) == C.H5_COMPONENTS
    assert CFG["class_names"] == P.CLASS_NAMES == sorted(P.CLASS_NAMES)
    assert "submission.parquet" in code and "test.csv" in code
    k = json.loads(B.kernel_metadata())
    assert k["enable_internet"] is False and k["enable_gpu"] is True
    assert k["competition_sources"] == ["name-that-llm"] and k["model_sources"] == ["qwen-lm/qwen-3/Transformers/8b-base/1"]
    assert {"ritisingh123/engs108-exp03-armh-artifacts", "ritisingh123/engs108-submission-h5-artifacts"} <= set(k["dataset_sources"])


def test_artifact_hashes_and_class_order():
    for fn, path in [("probe_armh.npz", ARMH / "probe_armh.npz"), ("blend_H5_dev.csv", ARMH / "blend_H5_dev.csv"),
                     ("stylo_exp03_model.joblib", REPO / "exp03" / "stylo" / "artifacts" / "stylo_exp03_model.joblib"),
                     ("h5_dev_reference.csv", REF)]:
        assert P.sha256_file(path) == CFG["sha256"][fn], fn
    sums = dict(l.split(maxsplit=1)[::-1] for l in (ARMH / "kernel_output_SHA256SUMS").read_text().splitlines())
    for k in ("adapter/adapter_model.safetensors", "adapter/adapter_config.json", "adapter/label_map.json"):
        assert CFG["sha256"][k] == sums[k]
    probe = ProbeModel(ARMH / "probe_armh.npz")
    assert probe.meta["C"] == 0.03
    C.check_class_order(P.CLASS_NAMES, P.CLASS_NAMES, probe.class_names, range(18), P.CLASS_NAMES)
    C.check_class_order(P.CLASS_NAMES, P.CLASS_NAMES, probe.class_names, range(18))  # official path: no train.csv
    with pytest.raises(AssertionError):
        C.check_class_order(P.CLASS_NAMES, P.CLASS_NAMES, probe.class_names[::-1], range(18), P.CLASS_NAMES)
    with pytest.raises(AssertionError):
        C.check_class_order(P.CLASS_NAMES, P.CLASS_NAMES, probe.class_names[::-1], range(18))
    with pytest.raises(AssertionError):
        C.check_class_order(P.CLASS_NAMES, P.CLASS_NAMES, probe.class_names, range(18), P.CLASS_NAMES[:-1])


def _ref():
    r = pd.read_csv(REF)
    comps = {k: r[[f"{k}:{c}" for c in P.PROBA_COLS]].to_numpy(np.float64) for k in C.H5_COMPONENTS}
    return r, comps, r[[f"h5:{c}" for c in P.PROBA_COLS]].to_numpy(np.float64)


def test_h5_blend_reproduces_reported_dev():
    r, comps, ref_blend = _ref()
    blend = C.h5_blend(comps)
    h5 = pd.read_csv(ARMH / "blend_H5_dev.csv").set_index("id").loc[r["id"]][P.PROBA_COLS].to_numpy(np.float64)
    rep, fails = C.dev_reproduction(r["labels"].to_numpy(), blend, comps, h5, ref_blend, comps,
                                    CFG["dev_check"]["expected_balanced_accuracy"], CFG["dev_check"]["tolerance"],
                                    CFG["dev_check"]["max_abs_proba_diff_vs_shared_pass_reference"])
    assert not fails, fails
    assert rep["argmax_agreement_vs_blend_H5_dev"] == 1.0 and rep["max_abs_proba_diff_vs_shared_pass_reference"] < 1e-8
    # a perturbed (e.g. H6-like re-weighted) blend must be caught
    bad = P.logprob_blend([P.log_proba(comps[k]) for k in C.H5_COMPONENTS], [0.221, 0.114, 0.161])
    _, fails = C.dev_reproduction(r["labels"].to_numpy(), bad, comps, h5, ref_blend, comps, 0.4323671497584541, 1e-3, 0.01)
    assert fails


def test_validate_submission():
    test = pd.DataFrame({"id": ["a", "b", "c"], "text": ["x", "y", "z"]})
    ok = pd.DataFrame({"id": ["a", "b", "c"], "model": P.CLASS_NAMES[:3]})
    assert C.validate_submission(ok, test, P.CLASS_NAMES)
    for bad in (ok.assign(model=["a", "b", "not-a-model"]), ok.assign(id=["a", "a", "c"]), ok.iloc[:2],
                ok.assign(extra=1), ok.assign(id=["a", None, "c"])):
        with pytest.raises(AssertionError):
            C.validate_submission(bad, test, P.CLASS_NAMES)


def test_extrapolate():
    e = C.extrapolate([1.0, 1.0, 0.5], [8, 8, 4], n_rows=4000)
    assert e["batches"] == 500 and abs(e["seconds_per_row"] - 0.125) < 1e-12 and abs(e["estimated_forward_minutes"] - 500 / 60) < 1e-9


@pytest.mark.skipif(not HAS_DATA, reason="needs ENGS108_DATA_DIR with train.csv")
def test_inlined_cpu_cells_rebuild_stylo_and_blend(tmp_path):
    """Executes the notebook's inlined-source cell in a fresh process (no repo modules on sys.path), unpickles the
    stylo joblib through the written features.py, and recomputes H5 from the reference head/probe components."""
    script = tmp_path / "run.py"
    inline = nb_code()[2]
    script.write_text(
        "import json, os, sys, numpy as np, pandas as pd, joblib\n"
        f"CONFIG = json.loads({json.dumps((HERE / 'config.json').read_text())})\nOUT = {str(tmp_path)!r}\n"
        + inline + "\n"
        f"r = pd.read_csv({str(REF)!r})\n"
        "cols = [f'p_{n}' for n in CLASS_NAMES]\n"
        "tr = pd.read_csv(os.environ['ENGS108_DATA_DIR'] + '/train.csv').set_index('id').loc[r['id']]\n"
        f"stylo = joblib.load({str(REPO / 'exp03/stylo/artifacts/stylo_exp03_model.joblib')!r})\n"
        "ps = stylo.predict_proba(pd.DataFrame({'text': tr['text'].tolist()}))\n"
        "comps = {k: r[[f'{k}:{c}' for c in cols]].to_numpy(np.float64) for k in H5_COMPONENTS}\n"
        "assert np.abs(ps - comps['stylo_exp03']).max() < 1e-6\n"
        "comps['stylo_exp03'] = ps\n"
        "b = h5_blend(comps)\n"
        "assert np.abs(b - r[[f'h5:{c}' for c in cols]].to_numpy()).max() < 1e-6\n"
        "print('ok', features.__file__)\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    res = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-3000:]
    assert str(tmp_path) in res.stdout


@pytest.mark.skipif(not (HAS_DATA and (Path.home() / "data" / "armh_ds" / "pooled_hidden_armh.npy").exists()),
                    reason="needs train.csv and the recorded pooled_hidden_armh.npy")
def test_component_probas_from_recorded_shared_pass():
    _, dev_df = P.load_data()
    pm = pd.read_parquet(ARMH / "armh_pooled_pass_logits.parquet")
    pos = {k: i for i, k in enumerate(pm["id"])}
    rows = [pos[k] for k in dev_df["id"]]
    H = np.load(Path.home() / "data" / "armh_ds" / "pooled_hidden_armh.npy")[rows].astype(np.float32)
    lg = pm[[f"logit_{n}" for n in P.CLASS_NAMES]].to_numpy(np.float32)[rows]
    import joblib
    comps = C.component_probas(H, lg, dev_df["text"].tolist(), ProbeModel(ARMH / "probe_armh.npz"),
                               joblib.load(REPO / "exp03" / "stylo" / "artifacts" / "stylo_exp03_model.joblib"))
    _, ref_comps, ref_blend = _ref()
    assert np.abs(C.h5_blend(comps) - ref_blend).max() < 1e-8
    assert P.metrics(dev_df["labels"].to_numpy(), C.h5_blend(comps))["balanced_accuracy"] == pytest.approx(0.4323671497584541, abs=1e-12)


# ---------------------------------------------------------------- optional dev check / submission-first ordering
CELL = {name: i for i, name in enumerate(["install", "setup", "inline", "artifacts", "model", "forward", "dev", "test",
                                          "summary"])}
TRAIN_NAMES = {"train", "dev_df", "df", "TRAIN_CSV", "va_idx"}


def _unguarded_train_refs(src):
    """train.csv / train-frame references that are not inside the body of an `if RUN_DEV_CHECK:` block."""
    bad = []

    def visit(node, guarded):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "RUN_DEV_CHECK":
            for n in node.body:
                visit(n, True)
            for n in node.orelse:
                visit(n, guarded)
            return
        if not guarded:
            if isinstance(node, ast.Name) and node.id in TRAIN_NAMES:
                bad.append((node.lineno, node.id))
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and "train.csv" in node.value:
                bad.append((node.lineno, node.value))
        for child in ast.iter_child_nodes(node):
            visit(child, guarded)

    visit(ast.parse(src), False)
    return bad


def test_dev_check_off_by_default_and_train_csv_only_inside_gate():
    assert CFG["run_dev_check"] is False
    code = nb_code()
    assert 'RUN_DEV_CHECK = CONFIG.get("run_dev_check", False)' in code[CELL["setup"]]
    # the embedded config JSON names the train.csv hash (used only by the dev check); it is data, not a read
    code[CELL["setup"]] = code[CELL["setup"]].replace((HERE / "config.json").read_text(), "{}")
    for name in ("setup", "artifacts", "model", "forward", "dev", "test", "summary"):
        assert not _unguarded_train_refs(code[CELL[name]]), (name, _unguarded_train_refs(code[CELL[name]]))
    # the guard itself is detected: an unconditional train.csv read would be flagged
    assert _unguarded_train_refs('train = pd.read_csv(f"{DATA_DIR}/train.csv")\nif RUN_DEV_CHECK:\n    x = train')
    art = code[CELL["artifacts"]]
    checks_src = art[art.index("checks = {"):art.index("for k, path in checks.items()")]
    assert "train.csv" not in checks_src and "adapter_model.safetensors" in checks_src and "probe_armh.npz" in checks_src
    assert "os.path.exists(TRAIN_CSV)" in art
    # dev gate code and tolerances are unchanged, only behind the flag
    dev = code[CELL["dev"]]
    assert dev.index("if RUN_DEV_CHECK:") < dev.index("dev_reproduction(") < dev.index("raise RuntimeError")
    assert 'dev_report = {"skipped": True}' in dev and "dev_minutes = 0" in dev
    assert CFG["dev_check"] == {"rows": 414, "prompts": 23, "expected_balanced_accuracy": 0.4323671497584541,
                                "tolerance": 0.001, "require_argmax_agreement_vs_blend_H5_dev": 1.0,
                                "max_abs_proba_diff_vs_shared_pass_reference": 0.01}


def test_submission_written_before_diagnostics():
    test, summary = nb_code()[CELL["test"]], nb_code()[CELL["summary"]]
    i_coerce, i_pred = test.index("coerced to empty string"), test.index("predict_h5(")
    i_write, i_check = test.index("pq.write_table("), test.index("validate_submission(written")
    i_try = test.index("try:")
    assert i_coerce < i_pred < i_write < i_check < i_try
    assert 'assert sample_schema.names == ["id", "model"]' in test and ".cast(sample_schema)" in test
    assert "rename" not in test
    for diag in ("batch_log", "extrapolate", "to_csv", "head(", "value_counts"):
        assert diag not in test[:i_try], diag
    # every statement of the summary cell is inside one try/except that only prints
    tree = ast.parse(summary).body
    assert [type(n) for n in tree] == [ast.Import, ast.Try]
    assert all(isinstance(h.type, ast.Name) and h.type.id == "Exception" and
               not any(isinstance(n, ast.Raise) for n in ast.walk(h)) for h in tree[1].handlers)


HARNESS = r"""
import hashlib, importlib.metadata, json, os, sys, types, urllib.request
import numpy as np
import pandas as pd

args = json.loads(sys.argv[1])
cells = json.load(open(args["nb"]))["cells"]
cells = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]

# ---- stubs: no GPU, no Qwen weights, no network
_version = importlib.metadata.version
def version(p):
    try:
        return _version(p)
    except importlib.metadata.PackageNotFoundError:
        return "absent"
importlib.metadata.version = version
def no_net(*a, **k):
    raise OSError("network disabled in tests")
urllib.request.urlopen = no_net
read_paths = []
_read_csv = pd.read_csv
def read_csv(path, *a, **k):
    read_paths.append(str(path))
    return _read_csv(path, *a, **k)
pd.read_csv = read_csv

class _D:
    def __init__(self, **kw): self.__dict__.update(kw)
torch = types.ModuleType("torch")
torch.float16, torch.float32 = "float16", "float32"
torch.Tensor = type("Tensor", (), {})  # scipy array-api checks probe sys.modules["torch"]
torch.cuda = _D(is_available=lambda: args.get("cuda", True), device_count=lambda: 1, get_device_name=lambda i: "stub",
                memory_allocated=lambda: 0, max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0,
                reset_peak_memory_stats=lambda: None, synchronize=lambda: None)
class Qwen3ForSequenceClassification:
    def __init__(self, id2label):
        self.config = _D(id2label=id2label, pad_token_id=None)
        self.model = _D(norm=_D(weight=_D(dtype="float16")))
        self.score = _D(weight=_D(dtype="float16"))
class Peft:
    def __init__(self, base):
        self.config = base.config
        self.base_model = _D(model=_D(score=_D(modules_to_save={"default": None})))
    def parameters(self): return iter(())
    def eval(self): return self
def kbit(base, use_gradient_checkpointing):
    base.model.norm.weight.dtype = base.score.weight.dtype = "float32"
    return base
transformers = types.ModuleType("transformers")
transformers.AutoTokenizer = _D(from_pretrained=lambda *a, **k: _D(pad_token_id=0))
transformers.AutoModelForSequenceClassification = _D(
    from_pretrained=lambda *a, id2label, **k: Qwen3ForSequenceClassification(id2label))
transformers.BitsAndBytesConfig = lambda **k: k
transformers.set_seed = lambda s: None
peft = types.ModuleType("peft")
peft.PeftModel = _D(from_pretrained=lambda base, *a, **k: Peft(base))
peft.prepare_model_for_kbit_training = kbit
sys.modules.update(torch=torch, transformers=transformers, peft=peft)

seen_texts = []
def stub_pooled_forward(model, score_wrapper, tokenizer, texts, batch_size, max_length, n_classes):
    assert all(isinstance(t, str) for t in texts), texts  # the real tokenizer raises on non-str input
    seen_texts.extend(texts)
    rng = np.random.default_rng(len(seen_texts))
    return (rng.normal(size=(len(texts), ns["probe"].mean.shape[0])).astype(np.float32),
            rng.normal(size=(len(texts), n_classes)).astype(np.float32))

cfg = json.loads(open(args["config"]).read())
cfg["run_dev_check"] = args["run_dev_check"]
cfg["sha256"].update(args["sha_overrides"])
setup = cells[1]
for old, new in [(open(args["config"]).read(), json.dumps(cfg)),
                 ('INPUT_ROOT = "/kaggle/input"', f"INPUT_ROOT = {args['input']!r}"),
                 ('OUT = "/kaggle/working"', f"OUT = {args['out']!r}"),
                 ('"/kaggle/input/**/', json.dumps(args["input"] + "/**/")[:-1])]:
    assert old in setup, old
    setup = setup.replace(old, new)
cells[1] = setup

ns = {"__name__": "__main__"}
for i, src in enumerate(cells[1:], start=1):
    exec(compile(src, f"cell-{i}", "exec"), ns)
    if i == 2:  # inlined sources: replace the GPU forward with a stub
        ns["pooled_forward"] = stub_pooled_forward
        if args.get("break_extrapolate"):
            ns["extrapolate"] = lambda *a, **k: 1 / 0
json.dump({"read_paths": read_paths, "seen_texts": seen_texts, "run_dev_check": ns["RUN_DEV_CHECK"],
           "data_dir": ns["DATA_DIR"],
           "dev_report": ns["dev_report"]}, open(os.path.join(args["out"], "harness.json"), "w"))
"""


def _write_data(d, ids, texts, sample_names=("id", "model")):
    import pyarrow as pa
    import pyarrow.parquet as pq
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "text": texts}).to_csv(d / "test.csv", index=False)
    pq.write_table(pa.table({sample_names[0]: pa.array(ids[:3], pa.large_string()),
                             sample_names[1]: pa.array(["gpt-5.5"] * len(ids[:3]), pa.large_string())}),
                   d / "sample_submission.parquet")


def _run_notebook(tmp_path, run_dev_check=False, break_extrapolate=False, cuda=True, decoy=True):
    """Executes every code cell after the bitsandbytes install, with torch/transformers/peft and the Qwen forward
    stubbed. The fake /kaggle/input root has the competition mount competitions/name-that-llm (test.csv +
    sample_submission.parquet, no train.csv) found by the notebook's own discovery, plus (decoy=True) an attached
    dataset datasets/<owner>/name-that-llm with the same file names that must not be picked."""
    out, inp = tmp_path / "working", tmp_path / "input"
    data = inp / "competitions" / "name-that-llm"
    adapter = inp / "datasets" / "ritisingh123" / "engs108-exp03-armh-artifacts" / "adapter"
    h5 = inp / "datasets" / "ritisingh123" / "engs108-submission-h5-artifacts"
    for d in (out, adapter, h5):
        d.mkdir(parents=True)
    texts = ["Plain answer.", "  leading/trailing whitespace \n", "unicode — “quotes” … 漢字", None, "NA", "12345",
             "Another\r\nline", "x" * 3000, "short", "tenth row"]
    ids = [f"a{i:011x}" for i in range(len(texts))]  # hex ids as in test.csv (all-digit ids would parse as int)
    _write_data(data, ids, texts)
    if decoy:
        _write_data(inp / "datasets" / "someone" / "name-that-llm", ["d0", "d1", "d2"], ["decoy"] * 3)
    (adapter / "adapter_model.safetensors").write_bytes(b"stub adapter weights")
    (adapter / "adapter_config.json").write_text(json.dumps({"modules_to_save": ["score", "classifier", "score"],
                                                             "task_type": "SEQ_CLS"}))
    (adapter / "label_map.json").write_text(json.dumps({"id2label": {str(i): n for i, n in enumerate(P.CLASS_NAMES)}}))
    for fn, src in [("probe_armh.npz", ARMH / "probe_armh.npz"), ("blend_H5_dev.csv", ARMH / "blend_H5_dev.csv"),
                    ("stylo_exp03_model.joblib", REPO / "exp03" / "stylo" / "artifacts" / "stylo_exp03_model.joblib"),
                    ("h5_dev_reference.csv", REF)]:
        (h5 / fn).write_bytes(src.read_bytes())
    sha = {f"adapter/{fn}": hashlib.sha256((adapter / fn).read_bytes()).hexdigest()
           for fn in ("adapter_model.safetensors", "adapter_config.json", "label_map.json")}
    (tmp_path / "harness.py").write_text(HARNESS)
    args = {"nb": str(HERE / B.NB_NAME), "config": str(HERE / "config.json"), "data": str(data), "out": str(out),
            "input": str(inp), "run_dev_check": run_dev_check, "sha_overrides": sha, "break_extrapolate": break_extrapolate,
            "cuda": cuda}
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    res = subprocess.run([sys.executable, str(tmp_path / "harness.py"), json.dumps(args)], cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=600)
    if not cuda:
        return res
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    return res.stdout, out, texts, ids


def _check_submission(out, ids):
    import pyarrow.parquet as pq
    sub = pd.read_parquet(out / "submission.parquet")
    assert list(sub.columns) == ["id", "model"] and sub["id"].tolist() == ids
    assert set(sub["model"]) <= set(P.CLASS_NAMES)
    assert pq.read_schema(out / "submission.parquet").names == ["id", "model"]
    assert str(pq.read_schema(out / "submission.parquet").field("id").type) == "large_string"


def test_notebook_submission_path_runs_without_train_csv(tmp_path):
    stdout, out, texts, ids = _run_notebook(tmp_path)
    _check_submission(out, ids)
    h = json.loads((out / "harness.json").read_text())
    assert not any("train" in p for p in h["read_paths"]), h["read_paths"]
    assert h["data_dir"] == str(tmp_path / "input" / "competitions" / "name-that-llm")
    assert f"DATA_DIR: {h['data_dir']} |" in stdout
    assert not any("datasets" in p for p in h["read_paths"]), h["read_paths"]
    assert h["run_dev_check"] is False and h["dev_report"] == {"skipped": True}
    assert "dev check skipped (run_dev_check=false)" in stdout
    # None and the literal "NA" (parsed as NaN by read_csv) become ""; every other string reaches the model unchanged
    assert "test.csv: 2 null/non-string text rows coerced to empty string" in stdout
    assert h["seen_texts"] == [t if t not in (None, "NA") else "" for t in texts]
    assert stdout.index("submission.parquet written and validated") < stdout.index("internet:")
    summary = json.loads((out / "h5_run_summary.json").read_text())
    assert summary["status"] == "completed" and summary["dev_check"] == {"skipped": True}
    assert summary["dev_minutes"] == 0 and summary["test_rows"] == len(ids)
    assert (out / "h5_batch_log.csv").exists() and not (out / "h5_dev_check.json").exists()


def test_post_submission_diagnostics_cannot_fail_the_run(tmp_path):
    stdout, out, _, ids = _run_notebook(tmp_path, break_extrapolate=True)
    _check_submission(out, ids)
    assert "WARNING: run summary failed (submission.parquet already written): ZeroDivisionError" in stdout
    assert not (out / "h5_run_summary.json").exists()


def test_dev_check_requested_but_train_csv_missing_is_skipped(tmp_path):
    stdout, out, _, ids = _run_notebook(tmp_path, run_dev_check=True)
    _check_submission(out, ids)
    assert "WARNING: run_dev_check=true but" in stdout and "dev check skipped" in stdout
    assert json.loads((out / "harness.json").read_text())["run_dev_check"] is False


# ---------------------------------------------------------------- DATA_DIR discovery
def _discovery():
    """find_data_dir exactly as generated into the setup cell."""
    setup = nb_code()[CELL["setup"]]
    fn = next(n for n in ast.parse(setup).body if isinstance(n, ast.FunctionDef) and n.name == "find_data_dir")
    import glob
    import pyarrow.parquet as pq
    ns = {"glob": glob, "os": os, "pd": pd, "pq": pq}
    exec(compile(ast.get_source_segment(setup, fn), "find_data_dir", "exec"), ns)
    return ns["find_data_dir"]


def test_setup_cell_discovers_data_dir_instead_of_hardcoding():
    setup = nb_code()[CELL["setup"]]
    assert 'INPUT_ROOT = "/kaggle/input"' in setup
    assert "DATA_DIR, data_dir_rejected = find_data_dir(INPUT_ROOT)" in setup and 'print("DATA_DIR:", DATA_DIR' in setup
    assert not any(l.lstrip().startswith("DATA_DIR = ") for l in setup.splitlines())
    # adapter / H5 artifact lookup is unchanged
    assert 'find_one("/kaggle/input/**/engs108-exp03-armh-artifacts/adapter/adapter_config.json")' in setup
    assert 'find_one("/kaggle/input/**/engs108-submission-h5-artifacts/probe_armh.npz")' in setup
    code = "\n".join(nb_code())
    for read in ('pd.read_csv(f"{DATA_DIR}/test.csv")', 'pq.read_schema(f"{DATA_DIR}/sample_submission.parquet")',
                 'TRAIN_CSV = f"{DATA_DIR}/train.csv"'):
        assert read in code, read


def test_discovery_order_and_competitions_preferred_over_datasets(tmp_path):
    find = _discovery()
    ids, texts = ["a", "b", "c"], ["x", "y", "z"]
    root = tmp_path / "input"
    # datasets/<owner>/<slug> mounts are never candidates, even with the exact slug and both files
    _write_data(root / "datasets" / "someone" / "name-that-llm", ids, texts)
    with pytest.raises(AssertionError):
        find(str(root))
    # an old-style top-level dataset mount vs a competitions/* mount: competitions/* wins
    _write_data(root / "aaa-decoy-dataset", ids, texts)
    _write_data(root / "competitions" / "other-slug", ids, texts)
    assert find(str(root))[0] == str(root / "competitions" / "other-slug")
    # the two known mounts come before the globs: competitions/<slug> first, then /kaggle/input/<slug>
    _write_data(root / "name-that-llm", ids, texts)
    assert find(str(root))[0] == str(root / "name-that-llm")
    _write_data(root / "competitions" / "name-that-llm", ids, texts)
    assert find(str(root)) == (str(root / "competitions" / "name-that-llm"), [])
    # without competitions/, the last-resort top-level glob still finds an old-style mount
    root2 = tmp_path / "old"
    _write_data(root2 / "some-mount", ids, texts)
    assert find(str(root2))[0] == str(root2 / "some-mount")


def test_discovery_rejects_wrong_files_and_reports_every_candidate(tmp_path):
    find = _discovery()
    root = tmp_path / "input"
    (root / "competitions" / "name-that-llm").mkdir(parents=True)
    (root / "competitions" / "name-that-llm" / "test.csv").write_text("id,text\na,x\n")  # no sample_submission
    _write_data(root / "competitions" / "bad-schema", ["a"], ["x"], sample_names=("row_id", "label"))
    _write_data(root / "bad-cols", ["a"], ["x"])
    pd.DataFrame({"id": ["a"], "prompt": ["x"]}).to_csv(root / "bad-cols" / "test.csv", index=False)
    _write_data(root / "datasets" / "someone" / "name-that-llm", ["a"], ["x"])
    with pytest.raises(AssertionError) as e:
        find(str(root))
    msg = str(e.value)
    for d, why in [("competitions/name-that-llm", "missing ['sample_submission.parquet']"), ("name-that-llm", "missing"),
                   ("competitions/bad-schema", "test.csv columns ['id', 'text'], sample_submission.parquet names ['row_id', 'label']"),
                   ("bad-cols", "test.csv columns ['id', 'prompt']"),
                   ("competitions", "missing"), ("datasets", "missing")]:
        assert f"{root}/{d}: {why}" in msg, (d, msg)
    assert "datasets/someone" not in msg
    # a valid competitions/* dir after the rejected ones is picked, and the rejections are returned for the log
    _write_data(root / "competitions" / "zz-good", ["a"], ["x"])
    d, rejected = find(str(root))
    assert d == str(root / "competitions" / "zz-good") and len(rejected) == 3


def test_missing_data_dir_fails_with_candidate_list(tmp_path):
    with pytest.raises(AssertionError) as e:
        _discovery()(str(tmp_path / "input"))
    assert f"{tmp_path}/input/competitions/name-that-llm: missing ['test.csv', 'sample_submission.parquet']" in str(e.value)


def test_notebook_metadata_gpu_consistent():
    md = json.loads((HERE / B.NB_NAME).read_text())["metadata"]
    assert md["kaggle"] == {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": False}
    k = json.loads(B.kernel_metadata())
    assert k["enable_gpu"] is True and k["machine_shape"] == "NvidiaTeslaT4" and k["enable_internet"] is False


def test_no_cuda_prints_loud_warning_and_stops(tmp_path):
    res = _run_notebook(tmp_path, cuda=False)
    assert res.returncode != 0
    assert "WARNING: torch.cuda.is_available() is False" in res.stdout
    assert "AssertionError: GPU required" in res.stderr
    assert not (tmp_path / "working" / "submission.parquet").exists()


def test_sim4k_kernel_runs_the_submission_cells_unchanged():
    sys.path.insert(0, str(HERE / "sim4k"))
    import build_sim as S
    assert (HERE / "sim4k" / S.NB_NAME).read_text() == S.build()
    assert (HERE / "sim4k" / "kernel-metadata.json").read_text() == S.kernel_metadata()
    sim = json.loads(S.build())
    runner = "".join(sim["cells"][3]["source"])
    cells = json.loads(runner[runner.index('r"""') + 4:runner.index('""")')])
    sub = nb_code()[1:]
    assert cells[0] == sub[0].replace('INPUT_ROOT = "/kaggle/input"', f'INPUT_ROOT = "{S.SIM_ROOT}"') != sub[0]
    assert cells[1:] == sub[1:]
    k = json.loads(S.kernel_metadata())
    assert k["id"] != json.loads(B.kernel_metadata())["id"] and k["enable_gpu"] is True
