"""Offline tests for the H5 submission notebook (CPU, no network). Tests that need the competition train.csv
(ENGS108_DATA_DIR) skip when it is absent."""
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
    with pytest.raises(AssertionError):
        C.check_class_order(P.CLASS_NAMES, P.CLASS_NAMES, probe.class_names[::-1], range(18), P.CLASS_NAMES)


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
