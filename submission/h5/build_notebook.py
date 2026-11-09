"""Builds the inference-only H5 submission notebook (mirrors exp03/qlora_armh/build_notebook.py).

  python submission/h5/build_notebook.py     # writes rs2027-engs108-submit-h5.ipynb + kernel-metadata.json

Code is inlined from the repo so the kernel runs exactly the exp03 implementations:
  exp03/qlora_armh/armh_core.py  pooled_forward (shared pass: head logits + pooled hidden state)
  exp03/probe/probe.py           ProbeModel (numpy probe from probe_armh.npz)
  exp03/stylo/protocol.py        log_proba, logprob_blend (the H5 log-linear pool)
  exp03/stylo/features.py        written to a module file so stylo_exp03_model.joblib can be unpickled
  submission/h5/h5_core.py       component probabilities, H5 blend, class-order / dev / submission checks
"""
import ast
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NB_NAME = "rs2027-engs108-submit-h5.ipynb"
SRC = {"armh_core": REPO / "exp03" / "qlora_armh" / "armh_core.py", "probe": REPO / "exp03" / "probe" / "probe.py",
       "protocol": REPO / "exp03" / "stylo" / "protocol.py", "features": REPO / "exp03" / "stylo" / "features.py",
       "h5_core": HERE / "h5_core.py"}


def extract(path, names):
    """Source of the named top-level functions/classes/assignments, verbatim, in file order."""
    text = path.read_text()
    out = []
    for node in ast.parse(text).body:
        name = getattr(node, "name", None) or (node.targets[0].id if isinstance(node, ast.Assign) else None)
        if name in names:
            out.append(ast.get_source_segment(text, node))
    assert len(out) == len(names), (path, names)
    return "\n\n\n".join(out) + "\n"


def h5_core_source():
    text = SRC["h5_core"].read_text()
    stripped, n = re.subn(r"# <repo-imports>.*?# </repo-imports>\n", "", text, flags=re.S)
    assert n == 1
    return stripped


SETUP = r'''import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # one T4 (as the Arm H kernel)

import glob, hashlib, json, sys, time
import importlib.metadata as metadata

import numpy as np
import pandas as pd
import torch

RUN_START = time.time()
CONFIG = json.loads(r"""__CONFIG__""")
MC, DC, SHA = CONFIG["model"], CONFIG["dev_check"], CONFIG["sha256"]
SEED, MAX_LENGTH, BATCH, N_CLASSES = CONFIG["seed"], MC["max_length"], MC["batch_size"], len(CONFIG["class_names"])
DATA_DIR = "/kaggle/input/competitions/name-that-llm"
MODEL_PATH = MC["base_path"]
OUT = "/kaggle/working"

versions = {p: metadata.version(p) for p in
            ["torch", "transformers", "peft", "accelerate", "bitsandbytes", "scikit-learn", "numpy", "pandas", "scipy"]}
print(versions)
print("CUDA:", torch.cuda.is_available(), "| visible GPUs:", torch.cuda.device_count(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
assert torch.cuda.is_available(), "GPU required"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_one(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    assert len(hits) == 1, (pattern, hits)
    return hits[0]


ADAPTER_DIR = os.path.dirname(find_one("/kaggle/input/**/engs108-exp03-armh-artifacts/adapter/adapter_config.json"))
H5_DIR = os.path.dirname(find_one("/kaggle/input/**/engs108-submission-h5-artifacts/probe_armh.npz"))
print("adapter:", ADAPTER_DIR, sorted(os.listdir(ADAPTER_DIR)))
print("h5 artifacts:", H5_DIR, sorted(os.listdir(H5_DIR)))
'''

INLINE = r'''from scipy.special import log_softmax, softmax

# ---- exp03/stylo/features.py (written as a module: stylo_exp03_model.joblib pickles references to `features`) ----
SRC_DIR = f"{OUT}/h5_src"
os.makedirs(SRC_DIR, exist_ok=True)
with open(f"{SRC_DIR}/features.py", "w") as f:
    f.write(FEATURES_SRC)
sys.path.insert(0, SRC_DIR)
import features  # noqa: F401

# ---- exp03/stylo/protocol.py (inlined: log_proba, logprob_blend) ----
CLASS_NAMES = CONFIG["class_names"]
__PROTOCOL__
# ---- exp03/probe/probe.py (inlined: ProbeModel) ----
__PROBE__
# ---- exp03/qlora_armh/armh_core.py (inlined: pooled_forward) ----
__POOLED__
# ---- submission/h5/h5_core.py (inlined) ----
__H5CORE__'''

ARTIFACTS = r'''import joblib
from sklearn.model_selection import GroupShuffleSplit

checks = {"train.csv": f"{DATA_DIR}/train.csv",
          "adapter/adapter_model.safetensors": f"{ADAPTER_DIR}/adapter_model.safetensors",
          "adapter/adapter_config.json": f"{ADAPTER_DIR}/adapter_config.json",
          "adapter/label_map.json": f"{ADAPTER_DIR}/label_map.json",
          **{k: f"{H5_DIR}/{k}" for k in ["probe_armh.npz", "stylo_exp03_model.joblib", "blend_H5_dev.csv", "h5_dev_reference.csv"]}}
for k, path in checks.items():
    got = sha256_file(path)
    assert got == SHA[k], f"{k}: sha256 {got} != expected {SHA[k]}"
print("all artifact sha256 verified:", list(checks))

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")  # row count not assumed: the hidden rerun has ~4,000 rows
assert {"id", "text"} <= set(test.columns) and len(test) > 0
print("test:", test.shape)

label2id = {n: i for i, n in enumerate(CLASS_NAMES)}
id2label = {i: n for n, i in label2id.items()}
df = train[["id", "prompt_id", "text", "model"]].copy()
df["labels"] = df["model"].map(label2id).astype(int)
_, va_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED).split(df, groups=df["prompt_id"]))
dev_df = df.iloc[va_idx].reset_index(drop=True)
assert (len(dev_df), dev_df["prompt_id"].nunique()) == (DC["rows"], DC["prompts"])
assert (dev_df["model"].value_counts() == 23).all() and dev_df["model"].nunique() == N_CLASSES
print("dev split verified: 414 rows / 23 prompts (exp01 split)")

with open(f"{ADAPTER_DIR}/label_map.json") as f:
    label_map_names = [n for _, n in sorted((int(k), v) for k, v in json.load(f)["id2label"].items())]
with open(f"{ADAPTER_DIR}/adapter_config.json") as f:
    adapter_cfg = json.load(f)
# PEFT appends its SEQ_CLS defaults ("classifier", "score") to the trained modules_to_save=["score"].
assert "score" in adapter_cfg["modules_to_save"], adapter_cfg["modules_to_save"]
assert set(adapter_cfg["modules_to_save"]) <= {"score", "classifier"}, adapter_cfg["modules_to_save"]
assert adapter_cfg["task_type"] == "SEQ_CLS", adapter_cfg["task_type"]
probe = ProbeModel(f"{H5_DIR}/probe_armh.npz")
assert probe.meta["C"] == 0.03 and probe.meta["model"] == "probe_armh", probe.meta.get("C")
stylo = joblib.load(f"{H5_DIR}/stylo_exp03_model.joblib")
print("probe C:", probe.meta["C"], "| coef", probe.coef.shape, "| stylo:", [s for s, _ in stylo.steps])
'''

MODEL = r'''from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, set_seed

t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                          bnb_4bit_use_double_quant=True)
set_seed(SEED)
base = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, num_labels=N_CLASSES, id2label=id2label, label2id=label2id, quantization_config=qcfg,
    device_map={"": 0}, torch_dtype=torch.float16, local_files_only=True)
base.config.pad_token_id = tokenizer.pad_token_id
assert type(base).__name__ == "Qwen3ForSequenceClassification", type(base).__name__
# Same dtype state as the Arm H kernel that produced the probe features and head logits: prepare_model_for_kbit_training
# upcasts the non-quantized params (embeddings, RMSNorms, score) to fp32. Without it the dev check fails
# (kernel v2: bal-acc 0.4300, max |dp| 0.017). No gradient checkpointing, all params frozen.
base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
assert base.model.norm.weight.dtype == torch.float32 and base.score.weight.dtype == torch.float32
model = PeftModel.from_pretrained(base, ADAPTER_DIR, is_trainable=False)
assert not any(p.requires_grad for p in model.parameters())
model.eval()
score_wrapper = model.base_model.model.score  # ModulesToSaveWrapper; the hook captures its input
assert "default" in score_wrapper.modules_to_save
load_minutes = (time.time() - t0) / 60
print(f"model + adapter loaded in {load_minutes:.1f} min | GPU allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB")

model_config_names = [model.config.id2label[i] for i in range(N_CLASSES)]
check_class_order(label_map_names, model_config_names, probe.class_names, stylo.classes_, train["model"])
print("class order identical across head config, label_map.json, probe npz and stylo classes_:", CLASS_NAMES)
'''

FORWARD = r'''batch_log = []


def shared_forward(texts, split):
    """One forward pass per batch of 8 (file order): pooled_forward returns head logits + pooled hidden states."""
    hidden = np.zeros((len(texts), probe.mean.shape[0]), dtype=np.float32)
    logits = np.zeros((len(texts), N_CLASSES), dtype=np.float32)
    for start in range(0, len(texts), BATCH):
        chunk = texts[start:start + BATCH]
        t = time.perf_counter()
        h, lg = pooled_forward(model, score_wrapper, tokenizer, chunk, BATCH, MAX_LENGTH, N_CLASSES)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        hidden[start:start + len(chunk)], logits[start:start + len(chunk)] = h, lg
        batch_log.append({"split": split, "start": start, "rows": len(chunk), "seconds": dt,
                          "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9})
        if (start // BATCH) % 25 == 0:
            print(f"[{split}] batch {start // BATCH + 1}/{-(-len(texts) // BATCH)}: {dt:.2f}s")
    return hidden, logits


def predict_h5(texts, split):
    hidden, logits = shared_forward(list(texts), split)
    t = time.perf_counter()
    comps = component_probas(hidden, logits, texts, probe, stylo)
    blend = h5_blend(comps)
    print(f"[{split}] probe + stylo + blend on CPU: {time.perf_counter() - t:.1f}s")
    return comps, blend, logits


torch.cuda.reset_peak_memory_stats()
'''

DEV = r'''# Dev reproduction (414 rows): same inference path, compared with the reported H5 before any test prediction.
t0 = time.time()
dev_comps, dev_blend, dev_logits = predict_h5(dev_df["text"].tolist(), "dev")
dev_minutes = (time.time() - t0) / 60


def aligned(path):
    r = pd.read_csv(path).set_index("id").loc[dev_df["id"]].reset_index()
    assert (r["labels"].to_numpy() == dev_df["labels"].to_numpy()).all()
    return r


PCOLS = [f"p_{n}" for n in CLASS_NAMES]
h5_rep = aligned(f"{H5_DIR}/blend_H5_dev.csv")[PCOLS].to_numpy(np.float64)
h5_rep = h5_rep / h5_rep.sum(1, keepdims=True)
ref = aligned(f"{H5_DIR}/h5_dev_reference.csv")
ref_shared = ref[[f"h5:{c}" for c in PCOLS]].to_numpy(np.float64)
ref_comps = {k: ref[[f"{k}:{c}" for c in PCOLS]].to_numpy(np.float64) for k in H5_COMPONENTS}
dev_report, dev_failures = dev_reproduction(
    dev_df["labels"].to_numpy(), dev_blend, dev_comps, h5_rep, ref_shared, ref_comps,
    DC["expected_balanced_accuracy"], DC["tolerance"], DC["max_abs_proba_diff_vs_shared_pass_reference"])
dev_report["minutes"] = dev_minutes
print(json.dumps(dev_report, indent=1))
out = dev_df[["id", "prompt_id", "labels"]].copy()
out = pd.concat([out, pd.DataFrame(dev_blend, columns=PCOLS)], axis=1)
out.to_csv(f"{OUT}/h5_dev_predictions.csv", index=False, float_format="%.10g")
with open(f"{OUT}/h5_dev_check.json", "w") as f:
    json.dump(dev_report, f, indent=1)
if dev_failures:
    raise RuntimeError("dev reproduction failed; investigate before writing a submission: " + "; ".join(dev_failures))
print(f"dev reproduction OK: balanced accuracy {dev_report['balanced_accuracy']:.4f} "
      f"(expected {DC['expected_balanced_accuracy']:.4f} +/- {DC['tolerance']}), argmax identical to blend_H5_dev.csv")
'''

TEST = r'''import pyarrow as pa
import pyarrow.parquet as pq

t0 = time.time()
test_comps, test_blend, _ = predict_h5(test["text"].tolist(), "test")
test_minutes = (time.time() - t0) / 60
submission = pd.DataFrame({"id": test["id"].to_numpy(), "model": [CLASS_NAMES[i] for i in test_blend.argmax(1)]})
validate_submission(submission, test, CLASS_NAMES)
sample_schema = pq.read_schema(f"{DATA_DIR}/sample_submission.parquet")
assert sample_schema.names == ["id", "model"], sample_schema
pq.write_table(pa.Table.from_pandas(submission, preserve_index=False).cast(sample_schema), f"{OUT}/submission.parquet")

written = pd.read_parquet(f"{OUT}/submission.parquet")
validate_submission(written, test, CLASS_NAMES)
assert pq.read_schema(f"{OUT}/submission.parquet").equals(sample_schema, check_metadata=False)
print(f"submission.parquet: {len(written)} rows, columns {list(written.columns)}, {test_minutes:.2f} min")
print(written.head(10))
print(written["model"].value_counts().to_string())
'''

SUMMARY = r'''import urllib.request

try:
    urllib.request.urlopen("https://pypi.org", timeout=5)
    internet = "REACHABLE"
except Exception as e:
    internet = f"unreachable ({type(e).__name__})"
print("internet:", internet)

bl = pd.DataFrame(batch_log)
bl.to_csv(f"{OUT}/h5_batch_log.csv", index=False)
peak = {"allocated_gb": torch.cuda.max_memory_allocated() / 1e9, "reserved_gb": torch.cuda.max_memory_reserved() / 1e9}
est = extrapolate(bl["seconds"], bl["rows"], n_rows=4000, batch_size=BATCH)
total_minutes = (time.time() - RUN_START) / 60
est["estimated_total_minutes_4000_rows"] = (total_minutes - test_minutes) + est["estimated_forward_minutes"]
summary = {"status": "completed", "test_rows": len(test), "dev_balanced_accuracy": dev_report["balanced_accuracy"],
           "dev_check": dev_report, "peak_gpu_memory": peak, "model_load_minutes": load_minutes,
           "dev_minutes": dev_minutes, "test_minutes": test_minutes, "total_minutes": total_minutes,
           "per_batch_seconds": {s: g["seconds"].describe().to_dict() for s, g in bl.groupby("split")},
           "extrapolation": est, "internet": internet, "versions": versions, "config": CONFIG}
with open(f"{OUT}/h5_run_summary.json", "w") as f:
    json.dump(summary, f, indent=1, default=float)
print(json.dumps({k: v for k, v in summary.items() if k not in ("config", "dev_check")}, indent=1, default=float))
'''


def build():
    config = (HERE / "config.json").read_text()
    assert '"""' not in config
    features = SRC["features"].read_text()
    assert "'''" not in features
    inline = (INLINE.replace("__PROTOCOL__", extract(SRC["protocol"], ["PROBA_FLOOR", "log_proba", "logprob_blend"]))
              .replace("__PROBE__", extract(SRC["probe"], ["ProbeModel"]))
              .replace("__POOLED__", extract(SRC["armh_core"], ["pooled_forward"]))
              .replace("__H5CORE__", h5_core_source()))
    inline = "FEATURES_SRC = r'''" + features + "'''\n\n" + inline
    md = ("# ENGS108 Name That LLM — H5 submission (inference only)\n\n"
          "Generated by `submission/h5/build_notebook.py`. No training. Qwen3-8B-Base 4-bit NF4 + the exp03 Arm H "
          "epoch-6 adapter; one shared forward pass per batch gives the head logits and the pooled hidden state. "
          "H5 = equal-weight log-linear blend of the C=0.03 probe (`probe_armh.npz`), the head, and the stylometric "
          "model (`stylo_exp03_model.joblib`). The 414-row dev split is re-scored and checked against the reported "
          "H5 (bal-acc 0.4324) before `submission.parquet` (`id`, `model`) is written. Internet off.")
    install = ('import glob, subprocess, sys\n'
               'wheels = sorted(glob.glob("/kaggle/input/**/bitsandbytes-*.whl", recursive=True))\n'
               'assert wheels, "attach the engs108-offline-wheels dataset"\n'
               'subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", wheels[-1]], check=True)\n')
    cells = [("markdown", md), ("code", install), ("code", SETUP.replace("__CONFIG__", config)), ("code", inline),
             ("code", ARTIFACTS), ("code", MODEL), ("code", FORWARD), ("code", DEV), ("code", TEST), ("code", SUMMARY)]
    nb = {"cells": [{"cell_type": t, "id": f"cell-{i:02d}", "metadata": {}, "source": s.splitlines(keepends=True),
                     **({"outputs": [], "execution_count": None} if t == "code" else {})} for i, (t, s) in enumerate(cells)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    return json.dumps(nb, indent=1) + "\n"


def kernel_metadata():
    k = json.loads((HERE / "config.json").read_text())["kaggle"]
    return json.dumps({
        "id": k["kernel"], "title": k["kernel"].split("/")[1].replace("rs2027-engs108", "RS2027-ENGS108"),
        "code_file": NB_NAME, "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": k["gpu"], "enable_tpu": False, "enable_internet": k["internet"], "keywords": ["gpu"],
        "dataset_sources": k["datasets"], "kernel_sources": [], "competition_sources": [k["competition"]],
        "model_sources": k["model_sources"], "docker_image": k["docker_image"], "machine_shape": k["machine_shape"]},
        indent=2) + "\n"


if __name__ == "__main__":
    (HERE / NB_NAME).write_text(build())
    (HERE / "kernel-metadata.json").write_text(kernel_metadata())
    print("wrote", HERE / NB_NAME)
