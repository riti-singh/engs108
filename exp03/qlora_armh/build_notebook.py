"""Builds the Arm H Kaggle notebook from config.json + armh_core.py (so the kernel runs exactly the frozen config).

  python exp03/qlora_armh/build_notebook.py            # writes rs2027-engs108-exp03-armh.ipynb (SMOKE_TEST=False)
  python exp03/qlora_armh/build_notebook.py --smoke DIR # writes a SMOKE_TEST=True copy + smoke kernel-metadata.json to DIR
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB_NAME = "rs2027-engs108-exp03-armh.ipynb"

SETUP = r'''import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # one T4: with 2 visible GPUs the Trainer uses n_gpu=2 for batch/loss scaling

import glob, hashlib, json, math, shutil, time, traceback
import importlib.metadata as metadata

import numpy as np
import pandas as pd
import torch

RUN_START = time.time()
SMOKE_TEST = __SMOKE__  # True only for the tiny pipeline smoke kernel

CONFIG = json.loads(r"""__CONFIG__""")
TC, LC, SC = CONFIG["training"], CONFIG["lora"], CONFIG["stop_criteria"]["params"]
SEED = CONFIG["seed"]
DATA_DIR = "/kaggle/input/competitions/name-that-llm"
MODEL_PATH = "/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-base/1"
OUT = "/kaggle/working"
CKPT_DIR = f"{OUT}/checkpoints"
ADAPTER_DIR = f"{OUT}/adapter"
N_CLASSES = 18

versions = {p: metadata.version(p) for p in
            ["torch", "transformers", "peft", "accelerate", "bitsandbytes", "datasets", "scikit-learn", "numpy"]}
print(versions)
print("CUDA:", torch.cuda.is_available(), "| visible GPUs:", torch.cuda.device_count(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
'''

DATA = r'''from sklearn.model_selection import GroupShuffleSplit

assert sha256_file(f"{DATA_DIR}/train.csv") == CONFIG["data"]["train_csv_sha256"]
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
model_names = sorted(train["model"].unique())
label2id = {n: i for i, n in enumerate(model_names)}
id2label = {i: n for n, i in label2id.items()}

df = train[["id", "prompt_id", "text", "model"]].copy()
df["labels"] = df["model"].map(label2id).astype(int)
tr_idx, va_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED).split(df, groups=df["prompt_id"]))
train_df = df.iloc[tr_idx].reset_index(drop=True)
val_df = df.iloc[va_idx].reset_index(drop=True)
DEV_PROMPTS = ['p009', 'p012', 'p018', 'p019', 'p026', 'p029', 'p031', 'p036', 'p055', 'p056', 'p064', 'p068',
               'p069', 'p073', 'p076', 'p078', 'p082', 'p104', 'p110', 'p118', 'p127', 'p132', 'p141']
assert (len(train_df), train_df["prompt_id"].nunique()) == (2285, 127)
assert (len(val_df), val_df["prompt_id"].nunique()) == (414, 23)
assert not set(train_df["prompt_id"]) & set(val_df["prompt_id"])
assert sorted(val_df["prompt_id"].unique()) == DEV_PROMPTS
assert (val_df["model"].value_counts() == 23).all() and val_df["model"].nunique() == N_CLASSES
print("Split verified: 2285/127 train, 414/23 dev (exp01 split)")
full_train_df, full_val_df = train_df, val_df

NUM_EPOCHS, EXPECTED_STEPS = TC["num_train_epochs"], TC["expected_optimizer_steps"]
if SMOKE_TEST:
    train_df = (train_df.assign(n=train_df["text"].str.len()).sort_values("n", ascending=False)
                .head(64).drop(columns="n").reset_index(drop=True))
    val_df = val_df.groupby("labels").head(2).reset_index(drop=True)
    NUM_EPOCHS, EXPECTED_STEPS = 2, 8
    print(f"SMOKE_TEST: train={len(train_df)} val={len(val_df)} epochs={NUM_EPOCHS}")
'''

MODEL = r'''from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, set_seed
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                          bnb_4bit_use_double_quant=True)
set_seed(SEED)  # exp01: set_seed immediately before from_pretrained -> same score-head init
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, num_labels=N_CLASSES, id2label=id2label, label2id=label2id, quantization_config=qcfg,
    device_map={"": 0}, torch_dtype=torch.float16, local_files_only=True)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, LoraConfig(
    task_type=TaskType.SEQ_CLS, r=LC["r"], lora_alpha=LC["lora_alpha"], lora_dropout=LC["lora_dropout"],
    target_modules=LC["target_modules"], modules_to_save=LC["modules_to_save"], bias=LC["bias"]))
model.print_trainable_parameters()
head_named, lora_named = split_param_groups(model)
n_head, n_lora = sum(p.numel() for _, p in head_named), sum(p.numel() for _, p in lora_named)
print("head tensors:", [n for n, _ in head_named], "| head params", n_head, "| lora params", n_lora)
exp = LC["expected_trainable_params"]
assert (n_head + n_lora, n_head, n_lora) == (exp["total"], exp["head"], exp["lora"])
score_wrapper = model.base_model.model.score
SCORE_IN = head_named[0][1].shape[1]
np.save(f"{OUT}/score_head_init.npy", head_named[0][1].detach().float().cpu().numpy())
'''

TRAINER = r'''from datasets import Dataset
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from transformers import DataCollatorWithPadding, TrainingArguments


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=TC["max_length"])


train_tok = Dataset.from_pandas(train_df[["text", "labels"]], preserve_index=False).map(tokenize, batched=True, remove_columns=["text"])
val_tok = Dataset.from_pandas(val_df[["text", "labels"]], preserve_index=False).map(tokenize, batched=True, remove_columns=["text"])
epoch_val_logits = []


def softmax_np(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = logits[0] if isinstance(logits, tuple) else logits
    epoch_val_logits.append(np.asarray(logits, dtype=np.float32))
    pred = logits.argmax(-1)
    return {"balanced_accuracy": balanced_accuracy_score(labels, pred),
            "macro_f1": f1_score(labels, pred, average="macro", labels=range(N_CLASSES), zero_division=0),
            "log_loss_proba": log_loss(labels, np.clip(softmax_np(logits.astype(np.float64)), 1e-12, 1), labels=range(N_CLASSES))}


args = TrainingArguments(
    output_dir=CKPT_DIR, per_device_train_batch_size=TC["per_device_train_batch_size"],
    per_device_eval_batch_size=TC["per_device_eval_batch_size"], gradient_accumulation_steps=TC["gradient_accumulation_steps"],
    learning_rate=TC["param_groups"]["lora"]["lr"], num_train_epochs=NUM_EPOCHS, lr_scheduler_type=TC["lr_scheduler_type"],
    warmup_steps=TC["warmup_steps"], max_grad_norm=TC["max_grad_norm"], fp16=True, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}, eval_strategy="epoch", save_strategy="epoch",
    save_total_limit=None, save_only_model=True, load_best_model_at_end=False, logging_steps=TC["logging_steps"],
    label_names=["labels"], report_to="none", seed=SEED)
assert args.weight_decay == 0.0
optimizer, opt_name, opt_kwargs = build_optimizer(model, args, TC["param_groups"]["head"]["lr"], TC["param_groups"]["lora"]["lr"])
print("optimizer:", opt_name, opt_kwargs, [(g["name"], g["lr"], len(g["params"])) for g in optimizer.param_groups])

trainer = ArmHTrainer(model=model, args=args, train_dataset=train_tok, eval_dataset=val_tok,
                      data_collator=DataCollatorWithPadding(tokenizer=tokenizer), compute_metrics=compute_metrics,
                      optimizers=(optimizer, None),
                      head_params=[p for _, p in head_named], lora_params=[p for _, p in lora_named])
trainer.model_accepts_loss_kwargs = False  # GA fix: Trainer divides each micro-batch loss by GA
monitor = ArmHMonitor(trainer, SC, log_every=10)
trainer.add_callback(monitor)
print("n_gpu:", args.n_gpu, "| model_accepts_loss_kwargs:", trainer.model_accepts_loss_kwargs)
assert args.n_gpu == 1
'''

TRAIN = r'''status, error = "running", None
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
try:
    train_result = trainer.train()
    status = "stopped: " + monitor.stop_reason if monitor.stop_reason else "completed"
except Exception as e:  # S4 operational failure: keep the logs, re-raise after saving
    status, error = f"failed: {type(e).__name__}", traceback.format_exc()
    print(error)
train_minutes = (time.time() - t0) / 60
print("status:", status, "| global_step:", trainer.state.global_step, f"| {train_minutes:.1f} min")
peak = {"allocated_gb": torch.cuda.max_memory_allocated() / 1e9, "reserved_gb": torch.cuda.max_memory_reserved() / 1e9}
print(peak)

step_log = pd.DataFrame(monitor.rows)
step_log.to_csv(f"{OUT}/step_log.csv", index=False)
with open(f"{OUT}/trainer_log_history.json", "w") as f:
    json.dump(trainer.state.log_history, f, indent=1)
ga_check = ga_fix_check(monitor.rows, trainer.state.log_history)
pd.DataFrame(ga_check).to_csv(f"{OUT}/ga_fix_check.csv", index=False)
epoch_hist = [h for h in trainer.state.log_history if "eval_balanced_accuracy" in h]
for h in epoch_hist:
    print(f"epoch {h['epoch']:.2f}: eval_loss={h['eval_loss']:.4f} bal_acc={h['eval_balanced_accuracy']:.4f} "
          f"macro_f1={h['eval_macro_f1']:.4f}")
if epoch_val_logits:
    np.save(f"{OUT}/epoch_dev_logits.npy", np.stack(epoch_val_logits))  # (n_epochs, n_dev, 18), val_df order
print("GA-fix check max rel diff:", max([r["rel_diff"] for r in ga_check], default=float("nan")))
print("skipped steps:", int(step_log["skipped"].sum()) if len(step_log) else None)
'''

OUTPUTS = r'''completed = status == "completed" and trainer.state.global_step == EXPECTED_STEPS
final_ckpt = f"{CKPT_DIR}/checkpoint-{trainer.state.global_step}"
print("epoch-6 primary checkpoint:", final_ckpt, os.path.isdir(final_ckpt), "| completed:", completed)
model.save_pretrained(ADAPTER_DIR)
with open(f"{ADAPTER_DIR}/label_map.json", "w") as f:
    json.dump({"id2label": {str(i): n for i, n in id2label.items()}}, f, indent=2)

model.eval()
EB, ML = TC["per_device_eval_batch_size"], TC["max_length"]
ev_train, ev_dev = (train_df, val_df) if SMOKE_TEST else (full_train_df, full_val_df)
if SMOKE_TEST:
    ev_train = ev_train.head(40)
t1 = time.time()
logit_cols = [f"logit_{n}" for n in model_names]
parts = []
for name, d in [("train", ev_train), ("dev", ev_dev), ("test", test)]:
    lg = sorted_predict_logits(model, tokenizer, d["text"].tolist(), EB, ML)
    meta = d[["id"]].copy()
    meta["prompt_id"] = d["prompt_id"].values if "prompt_id" in d else None
    meta["labels"] = d["labels"].values if "labels" in d else -1
    meta["split"] = name
    parts.append(pd.concat([meta.reset_index(drop=True), pd.DataFrame(lg, columns=logit_cols)], axis=1))
sorted_logits = pd.concat(parts, ignore_index=True)
sorted_logits.to_parquet(f"{OUT}/armh_sorted_logits.parquet", index=False)
dv = sorted_logits[sorted_logits.split == "dev"]
dev_bal_acc = balanced_accuracy_score(dv["labels"], dv[logit_cols].to_numpy().argmax(1))
print(f"length-sorted predict: dev bal-acc {dev_bal_acc:.4f} [{(time.time() - t1) / 60:.1f} min]")

t2 = time.time()
h_tr, l_tr = pooled_forward(model, score_wrapper, tokenizer, ev_train["text"].tolist(), EB, ML, N_CLASSES)
h_dv, l_dv = pooled_forward(model, score_wrapper, tokenizer, ev_dev["text"].tolist(), EB, ML, N_CLASSES)
h_te, l_te = pooled_forward(model, score_wrapper, tokenizer, test["text"].tolist(), EB, ML, N_CLASSES)
H = np.concatenate([h_tr, h_dv]).astype(np.float16)
np.save(f"{OUT}/pooled_hidden_armh.npy", H)
np.save(f"{OUT}/pooled_hidden_armh_test.npy", h_te.astype(np.float16))
pmeta = pd.concat([ev_train[["id", "prompt_id", "labels"]].assign(split="train"),
                   ev_dev[["id", "prompt_id", "labels"]].assign(split="dev")], ignore_index=True)
pd.concat([pmeta, pd.DataFrame(np.concatenate([l_tr, l_dv]), columns=logit_cols)], axis=1).to_parquet(
    f"{OUT}/armh_pooled_pass_logits.parquet", index=False)
pd.concat([test[["id"]].reset_index(drop=True), pd.DataFrame(l_te, columns=logit_cols)], axis=1).to_parquet(
    f"{OUT}/armh_pooled_pass_test_logits.parquet", index=False)
W = head_named[0][1].detach().float().cpu().numpy()
recon = float(np.abs(np.concatenate([l_tr, l_dv]) - np.concatenate([h_tr, h_dv]) @ W.T).max())
print(f"pooled pass: {(time.time() - t2) / 60:.1f} min | max |logits - H W^T| = {recon:.4f}")
'''

SUMMARY = r'''files = {}
for root, _, fns in os.walk(OUT):
    for fn in fns:
        p = os.path.join(root, fn)
        files[os.path.relpath(p, OUT)] = {"bytes": os.path.getsize(p), "sha256": sha256_file(p)}
with open(f"{OUT}/SHA256SUMS", "w") as f:
    for k in sorted(files):
        f.write(f"{files[k]['sha256']}  {k}\n")
summary = {"status": status, "completed_epoch6": completed, "error": error, "smoke_test": SMOKE_TEST,
           "global_step": trainer.state.global_step, "expected_steps": EXPECTED_STEPS, "stop_reason": monitor.stop_reason,
           "train_minutes": train_minutes, "total_minutes": (time.time() - RUN_START) / 60, "peak_gpu_memory": peak,
           "versions": versions, "optimizer": {"class": opt_name, "kwargs": {k: str(v) for k, v in opt_kwargs.items()}},
           "n_gpu": args.n_gpu, "model_accepts_loss_kwargs": trainer.model_accepts_loss_kwargs,
           "skipped_steps": int(step_log["skipped"].sum()) if len(step_log) else None,
           "ga_fix_check_max_rel_diff": max([r["rel_diff"] for r in ga_check], default=None),
           "epoch_dev_history": epoch_hist, "dev_bal_acc_sorted_predict": dev_bal_acc,
           "pooled_logit_reconstruction_max_abs": recon, "final_checkpoint": final_ckpt,
           "checkpoints": sorted(os.listdir(CKPT_DIR)) if os.path.isdir(CKPT_DIR) else [],
           "config": CONFIG, "files": files}
with open(f"{OUT}/run_summary.json", "w") as f:
    json.dump(summary, f, indent=1, default=str)
print(json.dumps({k: v for k, v in summary.items() if k not in ("config", "files")}, indent=1, default=str))
if error is not None:
    raise RuntimeError(f"training failed operationally: {status}")
'''


def build(smoke=False):
    config = (HERE / "config.json").read_text()
    core = (HERE / "armh_core.py").read_text()
    md = ("# exp03 Arm H — Qwen3-8B-Base QLoRA, head LR 1e-3 / LoRA LR 1e-4, 6 fixed epochs\n\n"
          "Generated by `exp03/qlora_armh/build_notebook.py` from the frozen `config.json` and `armh_core.py`. "
          "Original competition train rows only (2,285), exp01 split, epoch 6 is the primary checkpoint (fixed in advance). "
          "Internet off; bitsandbytes from the `engs108-offline-wheels` dataset. No submission is written.")
    install = ('import glob, subprocess, sys\n'
               'wheels = sorted(glob.glob("/kaggle/input/**/bitsandbytes-*.whl", recursive=True))\n'
               'assert wheels, "attach the engs108-offline-wheels dataset"\n'
               'subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", wheels[-1]], check=True)\n')
    assert '"""' not in config
    cells = [("markdown", md), ("code", install),
             ("code", SETUP.replace("__SMOKE__", str(bool(smoke))).replace("__CONFIG__", config)),
             ("code", "# ---- armh_core.py (inlined) ----\n" + core),
             ("code", DATA), ("code", MODEL), ("code", TRAINER), ("code", TRAIN), ("code", OUTPUTS), ("code", SUMMARY)]
    nb = {"cells": [{"cell_type": t, "id": f"cell-{i:02d}", "metadata": {}, "source": s.splitlines(keepends=True),
                     **({"outputs": [], "execution_count": None} if t == "code" else {})} for i, (t, s) in enumerate(cells)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    return json.dumps(nb, indent=1) + "\n"


def kernel_metadata(smoke=False):
    cfg = json.loads((HERE / "config.json").read_text())["kaggle"]
    kid = cfg["smoke_kernel"] if smoke else cfg["kernel"]
    return json.dumps({
        "id": kid, "title": kid.split("/")[1].replace("rs2027-engs108", "RS2027-ENGS108"),
        "code_file": NB_NAME, "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": ["gpu"],
        "dataset_sources": ["ritisingh123/engs108-offline-wheels"], "kernel_sources": [],
        "competition_sources": ["name-that-llm"], "model_sources": ["qwen-lm/qwen-3/Transformers/8b-base/1"],
        "docker_image": cfg["docker_image"].split(" ")[0], "machine_shape": "NvidiaTeslaT4"}, indent=2) + "\n"


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--smoke":
        out = Path(sys.argv[2])
        out.mkdir(parents=True, exist_ok=True)
        (out / NB_NAME).write_text(build(smoke=True))
        (out / "kernel-metadata.json").write_text(kernel_metadata(smoke=True))
        print("smoke kernel written to", out)
    else:
        (HERE / NB_NAME).write_text(build())
        (HERE / "kernel-metadata.json").write_text(kernel_metadata())
        print("wrote", HERE / NB_NAME)
