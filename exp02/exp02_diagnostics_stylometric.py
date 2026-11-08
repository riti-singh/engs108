# %% [markdown]
# # exp02 — Exp01 diagnostics, stylometric baseline, leakage-safe ensemble
#
# CPU-only analysis. **No Qwen training and no Qwen inference happens here.** Exp01 QLoRA logits
# come from the inference-only Kaggle kernel `ritisingh123/rs2027-engs108-exp02-val-inference`
# (`exp02/qwen-val-inference/`), copied into `exp02/artifacts/`.
#
# Everything uses the exp01 split: `GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)`
# on `prompt_id`.

# %%
import json
import os
import re
import string
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, recall_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, LeaveOneGroupOut, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC
import sklearn

from blend import equal_logprob_blend

warnings.filterwarnings("ignore", message="Unknown solver options")  # scipy/sklearn version notice
os.environ["PYTHONWARNINGS"] = "ignore:Unknown solver options"  # same, for joblib workers

SEED = 42
VAL_SIZE = 0.15
EXP02_DIR = Path(os.environ.get("EXP02_DIR", "."))
ART = EXP02_DIR / "artifacts"
FIG = EXP02_DIR / "figures"
FIG.mkdir(exist_ok=True)
DATA_DIR = Path(os.environ.get("ENGS108_DATA_DIR", "/kaggle/input/competitions/name-that-llm"))
print("scikit-learn", sklearn.__version__, "| data:", DATA_DIR)
RESULTS = {"seed": SEED, "val_size": VAL_SIZE, "sklearn": sklearn.__version__}

# %% [markdown]
# ## 1. Recreate and verify the exp01 split

# %%
train = pd.read_csv(DATA_DIR / "train.csv")
df = train[["id", "prompt_id", "text", "model"]].copy()
model_names = sorted(df["model"].unique())
label2id = {n: i for i, n in enumerate(model_names)}
df["labels"] = df["model"].map(label2id).astype(int)

splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED)
train_idx, val_idx = next(splitter.split(df, groups=df["prompt_id"]))
train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)

EXP01_VAL_PROMPTS = ['p009', 'p012', 'p018', 'p019', 'p026', 'p029', 'p031', 'p036', 'p055',
                     'p056', 'p064', 'p068', 'p069', 'p073', 'p076', 'p078', 'p082', 'p104',
                     'p110', 'p118', 'p127', 'p132', 'p141']  # printed by the exp01 training log
split_check = {
    "train_rows": len(train_df), "train_prompts": int(train_df["prompt_id"].nunique()),
    "val_rows": len(val_df), "val_prompts": int(val_df["prompt_id"].nunique()),
    "prompt_overlap": len(set(train_df["prompt_id"]) & set(val_df["prompt_id"])),
    "val_prompts_match_exp01_log": sorted(val_df["prompt_id"].unique()) == EXP01_VAL_PROMPTS,
    "val_rows_per_class": sorted(set(val_df["model"].value_counts())),
    "train_rows_per_class": {k: int(v) for k, v in train_df["model"].value_counts().sort_index().items()},
}
print(json.dumps(split_check, indent=1))
assert (split_check["train_rows"], split_check["train_prompts"]) == (2285, 127)
assert (split_check["val_rows"], split_check["val_prompts"]) == (414, 23)
assert split_check["prompt_overlap"] == 0 and split_check["val_prompts_match_exp01_log"]
assert split_check["val_rows_per_class"] == [23]
RESULTS["split"] = split_check

# Each prompt has one text per model (17-18 rows per prompt). Nothing below uses that
# structure (e.g. per-prompt assignment), because test.csv has no prompt_id.
print("rows per prompt:", df.groupby("prompt_id").size().value_counts().to_dict())

# %% [markdown]
# ## 2. Load exp01 QLoRA artifacts (inference-only kernel outputs)

# %%
diag = json.loads((ART / "qwen_diagnostics.json").read_text())
q_all = pd.read_parquet(ART / "exp01_logits.parquet")
s0_all = pd.read_parquet(ART / "step0_logits.parquet")
logit_cols = [f"logit_{n}" for n in model_names]
assert diag["inference_only"] is True

# Primary val logits: exp02-val-logits kernel, same predict code path as the offline inference kernel
# that reproduced exp01's 0.2464. exp01_logits.parquet (exp02-val-inference kernel) holds logits for all
# rows in split order and is used for train-split CE, linear probes and the batch-sensitivity check.
q_val = pd.read_parquet(ART / "exp01_val_logits.parquet").set_index("id").loc[val_df["id"]].reset_index()
assert (q_val["prompt_id"].values == val_df["prompt_id"].values).all()
assert (q_val["labels"].values == val_df["labels"].values).all()
assert set(q_all.loc[q_all["split"] == "train", "id"]) == set(train_df["id"])

y_val = val_df["labels"].to_numpy()
g_val = val_df["prompt_id"].to_numpy()
logit_q = q_val[logit_cols].to_numpy(np.float64)
logp_q = log_softmax(logit_q, axis=1)
pred_q = logit_q.argmax(1)

EXP01_RECALL = [0.174, 0.391, 0.043, 0.652, 0.217, 0.783, 0.217, 0.174, 0.304, 0.000,
                0.087, 0.304, 0.174, 0.174, 0.217, 0.261, 0.043, 0.217]  # experiments.md exp01
bal_q = balanced_accuracy_score(y_val, pred_q)
rec_q = recall_score(y_val, pred_q, average=None)
print(f"exp01 val balanced accuracy recomputed from saved logits: {bal_q:.4f} (exp01 reported 0.2464)")
print("max |recall - exp01 table|:", np.abs(rec_q.round(3) - EXP01_RECALL).max())
assert abs(bal_q - 0.2464) < 5e-4

# %% [markdown]
# ## 3. Exp01 classification diagnostics

# %%
FAMILY = {n: ("claude" if n.startswith("claude") else "gemini" if n.startswith("gemini")
              else "gpt" if n.startswith("gpt") else "glm" if n.startswith("glm")
              else n.split("-")[0]) for n in model_names}


def per_class_table(y, pred):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(18), zero_division=0)
    return pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s,
                         "n_predicted": np.bincount(pred, minlength=18)}, index=model_names).round(3)


def summary(y, pred):
    fam = np.array([FAMILY[model_names[i]] for i in range(18)])
    return {"balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "accuracy": float((y == pred).mean()),
            "family_accuracy": float((fam[y] == fam[pred]).mean())}


def top_confusions(y, pred, k=12):
    cm = confusion_matrix(y, pred, labels=range(18))
    np.fill_diagonal(cm, 0)
    pairs = [(model_names[i], model_names[j], int(cm[i, j])) for i in range(18) for j in range(18) if cm[i, j]]
    out = pd.DataFrame(pairs, columns=["true", "predicted", "count"]).sort_values("count", ascending=False)
    out["same_family"] = [FAMILY[a] == FAMILY[b] for a, b in zip(out["true"], out["predicted"])]
    return out.head(k).reset_index(drop=True)


def plot_confusion(y, pred, title, path):
    cm = confusion_matrix(y, pred, labels=range(18))
    cmn = cm / cm.sum(1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(22, 9.5))
    for ax, mat, fmt, ttl in [(axes[0], cm, "d", "counts"), (axes[1], cmn, ".2f", "row-normalised (recall)")]:
        ax.imshow(mat, cmap="Blues")
        for i in range(18):
            for j in range(18):
                v = mat[i, j]
                if v:
                    ax.text(j, i, format(v, fmt).lstrip("0") if fmt == ".2f" else v, ha="center", va="center",
                            fontsize=7, color="white" if v > mat.max() * 0.6 else "black")
        ax.set_xticks(range(18), model_names, rotation=90, fontsize=8)
        ax.set_yticks(range(18), model_names, fontsize=8)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{title} — {ttl}")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return cm


cm_q = plot_confusion(y_val, pred_q, "exp01 QLoRA (val, 414 rows)", FIG / "confusion_exp01_qlora.png")
pc_q = per_class_table(y_val, pred_q)
print(summary(y_val, pred_q))
print(pc_q.to_string())
tc_q = top_confusions(y_val, pred_q)
print(tc_q.to_string())
RESULTS["exp01"] = {**summary(y_val, pred_q), "per_class": pc_q.to_dict(orient="index"),
                    "confusion_matrix": cm_q.tolist(), "top_confusions": tc_q.to_dict(orient="records")}

# %% [markdown]
# ## 4. Loss diagnostics: why was the mean training loss 24.6?
#
# exp01 did **not** persist `trainer.state.log_history`. The per-10-step losses were only rendered
# by the notebook progress widget, which is not in the Kaggle stream log, and the Kaggle CLI does not
# return the rendered notebook. Recovering them would need a re-run of training, which is out of scope.
# Available evidence:
#
# * exp01 log: `train_loss` = 24.645 (mean of the 429 logged step losses), epoch eval losses
#   2.611 / 2.317 / 2.210; smoke run (4 steps on the 64 longest train texts) mean train loss 147;
# * inference-only kernel: step-0 loss of the exact starting function (base model + seed-42 random
#   `score` head; LoRA B=0 makes the adapter a no-op at step 0) and end-of-training train/val loss;
# * `check_grad_accum_loss_logging.py`: a CPU Trainer run in transformers 5.20.2 on a tiny random
#   `Qwen3ForSequenceClassification` with `learning_rate=0`, so the true loss is known.

# %%
ga_check = json.loads((ART / "grad_accum_loss_check.json").read_text())
ga_tbl = pd.DataFrame(ga_check["results"]).drop(columns="step_losses")
print("transformers", ga_check["versions"]["transformers"])
print(ga_tbl.round(4).to_string(index=False))

s0 = diag["step0"]
e1 = diag["exp01"]
GA = 8
STEPS, STEPS_PER_EPOCH, TRAIN_LOSS_LOGGED, SMOKE_LOSS_LOGGED = 429, 143, 24.64483656805434, 147.0

texts_by_id = df.set_index("id")["text"].str.len()
s0_tr = s0_all[s0_all["split"] == "train"].copy()
s0_tr["ce"] = -log_softmax(s0_tr[logit_cols].to_numpy(np.float64), axis=1)[np.arange(len(s0_tr)), s0_tr["labels"].to_numpy()]
s0_tr["n_chars"] = s0_tr["id"].map(texts_by_id)
smoke_ce = s0_tr.sort_values("n_chars", ascending=False).head(64)["ce"].mean()  # smoke run's 64 longest train texts

loss_points = pd.DataFrame([
    {"quantity": "smoke run mean train loss (4 steps, 64 longest train texts)", "logged": SMOKE_LOSS_LOGGED,
     "logged / GA": SMOKE_LOSS_LOGGED / GA, "measured (inference)": smoke_ce},
    {"quantity": "exp01 mean train loss over 429 steps", "logged": TRAIN_LOSS_LOGGED,
     "logged / GA": TRAIN_LOSS_LOGGED / GA, "measured (inference)": np.nan},
    {"quantity": "step-0 train-split CE (random score head)", "logged": np.nan, "logged / GA": np.nan,
     "measured (inference)": s0["train_mean_ce"]},
    {"quantity": "step-0 val CE", "logged": np.nan, "logged / GA": np.nan, "measured (inference)": s0["val_mean_ce"]},
    {"quantity": "epoch 1 / 2 / 3 eval loss", "logged": "2.611 / 2.317 / 2.210", "logged / GA": "n/a (eval has no GA)",
     "measured (inference)": np.nan},
    {"quantity": "end-of-training train-split CE (eval mode)", "logged": np.nan, "logged / GA": np.nan,
     "measured (inference)": e1["train_mean_ce"]},
    {"quantity": "end-of-training val CE", "logged": 2.2104, "logged / GA": np.nan, "measured (inference)": e1["val_mean_ce"]},
])
print(loss_points.to_string(index=False))
print(f"\nuniform-prediction CE ln(18) = {np.log(18):.3f}; step-0 val balanced accuracy = {s0['val_balanced_accuracy']:.4f}")
print(f"step-0 score head: std={diag['W0']['std']:.4f} frob={diag['W0']['frob']:.2f}; trained head: "
      f"std={diag['W1']['std']:.4f} frob={diag['W1']['frob']:.2f}; cosine(W0, W1)={diag['W1']['cosine_with_W0']:.3f}")
print(f"pooled hidden-state L2 norm: base median {diag['pooled_hidden_norm']['median']:.1f} -> "
      f"with exp01 LoRA {diag['pooled_hidden_norm_exp01_median']:.1f}")
print(f"median per-row logit std: step 0 {s0['median_row_logit_std']:.2f} -> trained {e1['median_row_logit_std']:.2f}")
sim = pd.DataFrame(s0["sim_heads"])
print(f"{len(sim)} other random heads with the same std: train CE {sim.train_mean_ce.min():.1f}-{sim.train_mean_ce.max():.1f}")

# Inference (not measured): bound on the true epoch-1 mean train loss, assuming epochs 2-3 train loss
# never exceeded the epoch-1 eval loss.
true_mean = TRAIN_LOSS_LOGGED / GA
epoch1_lb = (true_mean * STEPS - 2.6108 * (STEPS - STEPS_PER_EPOCH)) / STEPS_PER_EPOCH
print(f"true mean train loss = {true_mean:.3f}; implied epoch-1 mean >= {epoch1_lb:.2f} (inference)")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ax = axes[0]
ax.semilogy([0, 1, 2, 3], [s0["val_mean_ce"], 2.6108, 2.3173, 2.2104], "o-", label="val CE (step 0 measured; epochs 1-3 exp01 log)")
ax.semilogy([0, 3], [s0["train_mean_ce"], e1["train_mean_ce"]], "s", ms=8, label="train-split CE (inference: step 0, end)")
ax.hlines(TRAIN_LOSS_LOGGED, 0, 3, colors="C3", linestyles="--", label=f"logged mean train loss {TRAIN_LOSS_LOGGED:.1f} (8x GA artifact)")
ax.hlines(true_mean, 0, 3, colors="C3", linestyles="-", label=f"logged / 8 = {true_mean:.2f}")
ax.hlines(np.log(18), 0, 3, colors="gray", linestyles=":", label="uniform ln(18) = 2.89")
ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy (log scale)"); ax.set_title("exp01 loss: available points")
ax.legend(fontsize=7)
ax = axes[1]
ax.hist(s0_tr["ce"], bins=60, color="C1")
ax.axvline(s0_tr["ce"].mean(), color="k", ls="--", label=f"mean {s0_tr['ce'].mean():.1f}")
ax.set_xlabel("per-row CE at step 0 (train split)"); ax.set_ylabel("rows"); ax.set_title("step-0 loss distribution")
ax.legend()
fig.tight_layout(); fig.savefig(FIG / "loss_diagnostics.png", dpi=110); plt.close(fig)
RESULTS["loss_diagnostics"] = {"log_history_persisted": False, "grad_accum_check": ga_tbl.to_dict(orient="records"),
                               "smoke_logged": SMOKE_LOSS_LOGGED, "smoke_step0_measured": float(smoke_ce),
                               "train_loss_logged": TRAIN_LOSS_LOGGED, "train_loss_true_mean": true_mean,
                               "epoch1_true_mean_lower_bound_inferred": float(epoch1_lb),
                               "step0": s0, "W0": diag["W0"], "W1": diag["W1"],
                               "pooled_hidden_norm": diag["pooled_hidden_norm"],
                               "pooled_hidden_norm_exp01_median": diag["pooled_hidden_norm_exp01_median"],
                               "end_of_training": e1}

# %% [markdown]
# ### 4a'. Does the 8x GA loss scaling change optimisation? (Qwen, learning_rate=0, no weight updates)
#
# `qwen-grad-norm/` kernel: exp01 model setup and first train batches, lr=0, Trainer defaults ("default",
# = exp01) vs `trainer.model_accepts_loss_kwargs=False` ("fixed"), at exp01's step-0 weights and at its
# trained weights. Non-finite steps are fp16 GradScaler overflow steps (skipped by the optimizer).

# %%
gn = json.loads((ART / "qwen_grad_norm_check.json").read_text())
assert gn["lr"] == 0.0
qgn = {}
for init, (d_run, f_run) in gn["runs"].items():
    gd, gf = np.array(d_run["grad_norm_preclip"], float), np.array(f_run["grad_norm_preclip"], float)
    ok = np.isfinite(gd) & np.isfinite(gf)
    grp = [(g["score"], g["lora"]) for g in f_run["post_clip_group_norms"] if np.isfinite(g["score"]) and np.isfinite(g["lora"]) and g["lora"] > 0]
    qgn[init] = {"n_steps": len(gd), "nonfinite_steps_default": int((~np.isfinite(gd)).sum()),
                 "nonfinite_steps_fixed": int((~np.isfinite(gf)).sum()),
                 "default_over_fixed_ratio": float(np.median(gd[ok] / gf[ok])),
                 "true_grad_norm_min": float(np.nanmin(np.where(np.isfinite(gf), gf, np.nan))),
                 "true_grad_norm_max": float(np.nanmax(np.where(np.isfinite(gf), gf, np.nan))),
                 "frac_finite_steps_clipped_default": float((gd[np.isfinite(gd)] > gn["max_grad_norm"]).mean()),
                 "frac_finite_steps_clipped_fixed": float((gf[np.isfinite(gf)] > gn["max_grad_norm"]).mean()),
                 "head_over_lora_grad_norm_median": float(np.median([a / b for a, b in grp]))}
print(json.dumps(qgn, indent=1))
cpu = json.loads((ART / "grad_accum_optimization_check.json").read_text())["results"]
print(pd.DataFrame(cpu).drop(columns="grad_norm").round(6).to_string(index=False))
RESULTS["qwen_grad_norm_lr0"] = qgn
RESULTS["cpu_grad_accum_optimization"] = [{k: v for k, v in r.items() if k != "grad_norm"} for r in cpu]

# %% [markdown]
# ### 4b. Batch-padding sensitivity of the exp01 logits
#
# Same weights, same texts, eval mode; only the batch composition differs. 4-bit NF4 + fp16 logits
# shift slightly with the amount of right-padding in a batch.
#
# * primary: length-sorted batches of 8 (offline inference kernel code path);
# * `split order`: val rows in `val_df` order, batches of 8 (exp02-val-inference kernel v2);
# * `mixed`: val rows interleaved with train rows (exp02-val-inference kernel v1).

# %%
batch_sens = {"primary_length_sorted": {"balanced_accuracy": float(balanced_accuracy_score(y_val, pred_q))}}
alts = {"split_order": q_all[q_all["split"] == "val"],
        "mixed": pd.read_parquet(ART / "exp01_val_logits_mixed_batches.parquet")}
for name, a in alts.items():
    a_logits = a.set_index("id").loc[val_df["id"]][logit_cols].to_numpy(np.float64)
    batch_sens[name] = {"balanced_accuracy": float(balanced_accuracy_score(y_val, a_logits.argmax(1))),
                        "rows_with_argmax_flip_vs_primary": int((a_logits.argmax(1) != pred_q).sum()),
                        "max_abs_logit_diff_vs_primary": float(np.abs(a_logits - logit_q).max())}
print(json.dumps(batch_sens, indent=1))
RESULTS["batch_padding_sensitivity"] = batch_sens

# %% [markdown]
# ### 4c. Linear probes on saved pooled hidden states (CPU, no Qwen compute)
#
# Logistic regression on the 4096-d last-token hidden state that feeds `score`, for the base model
# (step 0, no adapter) and for the exp01 LoRA model. Fit on train rows, C chosen by GroupKFold(5) on
# train prompts, scored once on val. Shows how much signal is linearly available before and after LoRA.
# Needs `pooled_hidden_*.npy` (22 MB each, not committed). Get them with `./fetch_pooled_hidden.sh`
# (private Kaggle dataset ritisingh123/engs108-exp02-pooled-hidden-states) or regenerate them with the
# exp02-val-inference kernel. Their sha256 must match artifacts/pooled_hidden_SHA256SUMS unless
# EXP02_ALLOW_HIDDEN_MISMATCH=1 (regenerated 4-bit/fp16 states need not be bit-identical).

# %%
import hashlib

HIDDEN_DIR = Path(os.environ.get("EXP02_HIDDEN_DIR", ART))
y_tr, g_tr = train_df["labels"].to_numpy(), train_df["prompt_id"].to_numpy()
probe_rows = []
if (HIDDEN_DIR / "pooled_hidden_step0.npy").exists():
    expected = dict(l.split()[::-1] for l in (ART / "pooled_hidden_SHA256SUMS").read_text().splitlines())
    sha_ok = {fn: hashlib.sha256((HIDDEN_DIR / fn).read_bytes()).hexdigest() == h for fn, h in expected.items()}
    print("pooled hidden sha256 match:", sha_ok)
    if not all(sha_ok.values()) and os.environ.get("EXP02_ALLOW_HIDDEN_MISMATCH") != "1":
        raise SystemExit("pooled_hidden_*.npy differ from the exp02 originals; set EXP02_ALLOW_HIDDEN_MISMATCH=1 to proceed")
    RESULTS["pooled_hidden_sha256_match"] = sha_ok
    order = q_all["id"].to_numpy()
    pos = {k: i for i, k in enumerate(order)}
    tr_pos = np.array([pos[k] for k in train_df["id"]]); va_pos = np.array([pos[k] for k in val_df["id"]])
    for name, fn in [("base Qwen3-8B (step 0, no adapter)", "pooled_hidden_step0.npy"),
                     ("exp01 LoRA model", "pooled_hidden_exp01.npy")]:
        H = np.load(HIDDEN_DIR / fn).astype(np.float64)
        best = None
        for C in (0.001, 0.01, 0.1):
            pipe = Pipeline([("z", StandardScaler()), ("lr", LogisticRegression(C=C, max_iter=3000))])
            cvs = cross_validate(pipe, H[tr_pos], y_tr, groups=g_tr, cv=GroupKFold(5),
                                 scoring="balanced_accuracy", n_jobs=5)["test_score"].mean()
            if best is None or cvs > best[1]:
                best = (C, cvs)
        pipe = Pipeline([("z", StandardScaler()), ("lr", LogisticRegression(C=best[0], max_iter=3000))]).fit(H[tr_pos], y_tr)
        pp = pipe.predict_proba(H[va_pos]); pv = pp.argmax(1)
        assert list(pipe.classes_) == list(range(18))
        tag = "step0" if "step 0" in name else "exp01"
        pd.DataFrame({"id": val_df["id"], "prompt_id": g_val, "labels": y_val,
                      **{f"p_{n}": pp[:, i] for i, n in enumerate(model_names)}}).to_csv(
            ART / f"linear_probe_{tag}_val_proba.csv", index=False, float_format="%.6g")
        probe_rows.append({"features": name, "C": best[0], "train_cv_bal_acc": best[1],
                           "val_bal_acc": balanced_accuracy_score(y_val, pv), "val_macro_f1": f1_score(y_val, pv, average="macro")})
    probes = pd.DataFrame(probe_rows)
    print(probes.round(4).to_string(index=False))
    RESULTS["linear_probes"] = probes.to_dict(orient="records")
else:
    print("pooled hidden states not found; skipping probes")

# %% [markdown]
# ## 5. Classical stylometric baseline (same split)
#
# Model selection uses **GroupKFold(5) on the train split only** (grouped by `prompt_id`); the
# validation split is scored once per variant after selection. The primary stylometric model is the
# variant with the best train-CV balanced accuracy, not the best validation score.

# %%
PUNCT = set(string.punctuation) | set("—–‘’“”…«»·•")
STYLE_NAMES = ["log_chars", "log_words", "mean_word_len", "n_sentences", "mean_sentence_words", "type_token_ratio",
               "punct_rate", "upper_rate", "digit_rate", "whitespace_rate", "newline_rate",
               "comma_rate", "semicolon_rate", "colon_rate", "em_dash_rate", "en_dash_rate", "hyphen_rate",
               "curly_apos_rate", "straight_apos_rate", "curly_quote_rate", "straight_quote_rate",
               "paren_rate", "exclaim_rate", "question_rate", "ellipsis_rate", "asterisk_rate", "hash_rate"]


def style_features(texts):
    rows = []
    for t in texts:
        n = max(len(t), 1)
        words = re.findall(r"\w+", t)
        sents = [s for s in re.split(r"[.!?]+\s", t) if s.strip()]
        rows.append([
            np.log1p(len(t)), np.log1p(len(words)), np.mean([len(w) for w in words]) if words else 0.0,
            len(sents), len(words) / max(len(sents), 1), len({w.lower() for w in words}) / max(len(words), 1),
            sum(c in PUNCT for c in t) / n, sum(c.isupper() for c in t) / n, sum(c.isdigit() for c in t) / n,
            sum(c.isspace() for c in t) / n, t.count("\n") / n,
            t.count(",") / n, t.count(";") / n, t.count(":") / n, t.count("—") / n, t.count("–") / n, t.count("-") / n,
            t.count("’") / n, t.count("'") / n, (t.count("“") + t.count("”")) / n, t.count('"') / n,
            (t.count("(") + t.count(")")) / n, t.count("!") / n, t.count("?") / n,
            (t.count("…") + t.count("...")) / n, t.count("*") / n, t.count("#") / n,
        ])
    return np.asarray(rows, dtype=np.float64)


def char_tfidf():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)


def word_tfidf():
    return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def features(kind):
    blocks = [("char", char_tfidf(), "text")]
    if kind == "char+word+style":
        blocks += [("word", word_tfidf(), "text"),
                   ("style", Pipeline([("f", FunctionTransformer(style_features)), ("z", StandardScaler())]), "text")]
    return ColumnTransformer(blocks)


def lr(C):
    return LogisticRegression(C=C, max_iter=5000)


def svc(C):
    # Sigmoid calibration on 3 internal folds of the training data (never validation rows).
    return CalibratedClassifierCV(LinearSVC(C=C, max_iter=20000), method="sigmoid", cv=3)


CANDIDATES = (
    [("char", "lr", C) for C in (1, 10, 100)]
    + [("char+word+style", "lr", C) for C in (1, 10, 100)]
    + [("char+word+style", "svc", C) for C in (0.1, 0.5)]
)
X_tr, y_tr, g_tr = train_df[["text"]], train_df["labels"].to_numpy(), train_df["prompt_id"].to_numpy()
X_va = val_df[["text"]]
inner_cv = GroupKFold(n_splits=5)

rows, fitted = [], {}
for feat, clf_name, C in CANDIDATES:
    pipe = Pipeline([("features", features(feat)), ("clf", lr(C) if clf_name == "lr" else svc(C))])
    t0 = time.time()
    cv = cross_validate(pipe, X_tr, y_tr, groups=g_tr, cv=inner_cv, scoring="balanced_accuracy", n_jobs=5)
    pipe.fit(X_tr, y_tr)
    pred_v = pipe.predict(X_va)
    key = f"{feat} | {clf_name} | C={C}"
    fitted[key] = pipe
    rows.append({"variant": key, "train_cv_bal_acc": cv["test_score"].mean(), "train_cv_std": cv["test_score"].std(),
                 "val_bal_acc": balanced_accuracy_score(y_val, pred_v),
                 "val_macro_f1": f1_score(y_val, pred_v, average="macro"), "seconds": time.time() - t0})
    print(rows[-1])
sweep = pd.DataFrame(rows).sort_values("train_cv_bal_acc", ascending=False).reset_index(drop=True)
print(sweep.round(4).to_string())
PRIMARY_STYLO = sweep.loc[0, "variant"]
print("\nprimary stylometric model (best grouped train-CV):", PRIMARY_STYLO)
best_char_lr = sweep[sweep.variant.str.startswith("char | lr")].iloc[0]["variant"]
print("best char-only LR (spec baseline):", best_char_lr)
RESULTS["stylometric_sweep"] = sweep.to_dict(orient="records")
RESULTS["primary_stylometric"] = PRIMARY_STYLO

# %%
stylo = fitted[PRIMARY_STYLO]
p_s = stylo.predict_proba(X_va)
assert list(stylo.classes_) == list(range(18))
logp_s = np.log(np.clip(p_s, 1e-12, 1))
pred_s = p_s.argmax(1)
cm_s = plot_confusion(y_val, pred_s, f"stylometric ({PRIMARY_STYLO})", FIG / "confusion_stylometric.png")
pc_s = per_class_table(y_val, pred_s)
print(summary(y_val, pred_s))
print(pc_s.to_string())
tc_s = top_confusions(y_val, pred_s)
print(tc_s.to_string())
RESULTS["stylometric"] = {**summary(y_val, pred_s), "per_class": pc_s.to_dict(orient="index"),
                          "confusion_matrix": cm_s.tolist(), "top_confusions": tc_s.to_dict(orient="records")}
pd.DataFrame({"id": val_df["id"], "prompt_id": g_val, "labels": y_val,
              **{f"p_{n}": p_s[:, i] for i, n in enumerate(model_names)}}).to_csv(
    ART / "stylometric_val_proba.csv", index=False, float_format="%.6g")

# %% [markdown]
# ## 6. Complementarity and leakage-safe ensembling
#
# Inputs per validation row: QLoRA log-probabilities (softmax of saved exp01 logits) and
# stylometric log-probabilities (model fit on the train split only). Neither base model has seen
# any validation row. Ensemble parameters are only ever fit on validation rows from prompts
# *other* than the ones being scored.
#
# * **equal** — `log p_q + log p_s`, zero fitted parameters (fixed a priori).
# * **blend2** (primary, chosen a priori) — `a·log p_q + b·log p_s`, `a,b ≥ 0`, fit by minimising NLL.
# * **stacker** — multinomial logistic regression on the 36 concatenated log-probs, `C=1.0` fixed a priori.
#
# Protocols: (A) val-A / val-B prompt-group split, fit on A, report on B (and swapped, as a
# sensitivity check); (B) leave-one-prompt-out cross-fitting over the 23 val prompts: each prompt is
# scored by an ensemble fit on the other 22, giving honest out-of-fold predictions for all 414 rows.

# %%
agree = (pred_q == pred_s).mean()
both = ((pred_q == y_val) & (pred_s == y_val)).mean()
either = ((pred_q == y_val) | (pred_s == y_val)).mean()
err_corr = np.corrcoef(pred_q != y_val, pred_s != y_val)[0, 1]
comp = {"prediction_agreement": float(agree), "both_correct": float(both), "oracle_either_correct": float(either),
        "qlora_only_correct": float(((pred_q == y_val) & (pred_s != y_val)).mean()),
        "stylo_only_correct": float(((pred_q != y_val) & (pred_s == y_val)).mean()),
        "error_correlation": float(err_corr)}
print(json.dumps(comp, indent=1))
RESULTS["complementarity"] = comp


def fit_blend2(lq, ls, y):
    def nll(w):
        z = w[0] * lq + w[1] * ls
        return -log_softmax(z, axis=1)[np.arange(len(y)), y].mean()
    res = minimize(nll, x0=[0.5, 0.5], bounds=[(0, None), (0, None)], method="L-BFGS-B")
    return res.x


def ensemble_predict(method, fit_idx, score_idx):
    lq, ls = logp_q, logp_s
    if method == "equal":
        return equal_logprob_blend(logit_q, p_s)[score_idx], None
    if method == "blend2":
        w = fit_blend2(lq[fit_idx], ls[fit_idx], y_val[fit_idx])
        return (w[0] * lq + w[1] * ls)[score_idx], w
    if method == "stacker":
        X = np.hstack([lq, ls])
        m = LogisticRegression(C=1.0, max_iter=5000).fit(X[fit_idx], y_val[fit_idx])
        return m.predict_log_proba(X[score_idx]), None
    raise ValueError(method)


METHODS = ["equal", "blend2", "stacker"]

# Protocol A: val-A / val-B by prompt group
ab = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
idx_a, idx_b = next(ab.split(val_df, groups=g_val))
assert not set(g_val[idx_a]) & set(g_val[idx_b])
print(f"val-A: {len(idx_a)} rows / {len(set(g_val[idx_a]))} prompts; val-B: {len(idx_b)} rows / {len(set(g_val[idx_b]))} prompts")
protoA = []
for fit_idx, score_idx, name in [(idx_a, idx_b, "fit A -> report B"), (idx_b, idx_a, "fit B -> report A (sensitivity)")]:
    row = {"protocol": name, "report_rows": len(score_idx),
           "qlora": balanced_accuracy_score(y_val[score_idx], pred_q[score_idx]),
           "stylometric": balanced_accuracy_score(y_val[score_idx], pred_s[score_idx])}
    for m in METHODS:
        z, w = ensemble_predict(m, fit_idx, score_idx)
        row[m] = balanced_accuracy_score(y_val[score_idx], z.argmax(1))
        if w is not None:
            row["blend2_weights(a_qlora,b_stylo)"] = np.round(w, 3).tolist()
    protoA.append(row)
protoA = pd.DataFrame(protoA)
print(protoA.round(4).to_string())
RESULTS["protocol_A_val_split"] = {"val_A_prompts": sorted(set(g_val[idx_a])), "val_B_prompts": sorted(set(g_val[idx_b])),
                                   "rows": protoA.to_dict(orient="records")}

# Protocol B: leave-one-prompt-out cross-fitting
oof = {m: np.zeros((len(y_val), 18)) for m in METHODS}
weights = []
for fit_idx, score_idx in LeaveOneGroupOut().split(val_df, groups=g_val):
    assert not set(g_val[fit_idx]) & set(g_val[score_idx])
    for m in METHODS:
        z, w = ensemble_predict(m, fit_idx, score_idx)
        oof[m][score_idx] = z
        if w is not None:
            weights.append(w)
weights = np.array(weights)
pred_oof = {m: oof[m].argmax(1) for m in METHODS}
print(f"blend2 weights across 23 folds: a_qlora {weights[:, 0].mean():.3f}±{weights[:, 0].std():.3f}, "
      f"b_stylo {weights[:, 1].mean():.3f}±{weights[:, 1].std():.3f}")

# %%
rng = np.random.default_rng(SEED)
prompts = np.unique(g_val)
rows_of = {p: np.where(g_val == p)[0] for p in prompts}


def grouped_bootstrap_delta(pa, pb, n=2000):
    deltas = []
    for _ in range(n):
        idx = np.concatenate([rows_of[p] for p in rng.choice(prompts, size=len(prompts), replace=True)])
        deltas.append(balanced_accuracy_score(y_val[idx], pa[idx]) - balanced_accuracy_score(y_val[idx], pb[idx]))
    d = np.array(deltas)
    return {"delta": float(balanced_accuracy_score(y_val, pa) - balanced_accuracy_score(y_val, pb)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_le_0": float((d <= 0).mean())}


preds = {"exp01 QLoRA": pred_q, "stylometric": pred_s, **{f"{m} (LOPO OOF)": pred_oof[m] for m in METHODS}}
comparison = pd.DataFrame({k: summary(y_val, v) for k, v in preds.items()}).T
recall_tbl = pd.DataFrame({k: recall_score(y_val, v, average=None) for k, v in preds.items()}, index=model_names).round(3)
print(comparison.round(4).to_string())
print(recall_tbl.to_string())
boot = {"blend2 - qlora": grouped_bootstrap_delta(pred_oof["blend2"], pred_q),
        "blend2 - stylometric": grouped_bootstrap_delta(pred_oof["blend2"], pred_s),
        "stylometric - qlora": grouped_bootstrap_delta(pred_s, pred_q),
        "stacker - qlora": grouped_bootstrap_delta(pred_oof["stacker"], pred_q),
        "equal - qlora": grouped_bootstrap_delta(pred_oof["equal"], pred_q)}
print(json.dumps(boot, indent=1))

# %%
def class_changes(base_pred, new_pred):
    out = []
    for i, n in enumerate(model_names):
        m = y_val == i
        fixed = int(((base_pred != i) & (new_pred == i) & m).sum())
        broke = int(((base_pred == i) & (new_pred != i) & m).sum())
        out.append({"class": n, "recall_base": (base_pred[m] == i).mean(), "recall_blend": (new_pred[m] == i).mean(),
                    "corrected": fixed, "worsened": broke, "net": fixed - broke})
    return pd.DataFrame(out).round(3)


chg_q = class_changes(pred_q, pred_oof["blend2"])
chg_s = class_changes(pred_s, pred_oof["blend2"])
print("blend2 (LOPO OOF) vs exp01 QLoRA:\n", chg_q.to_string(index=False))
print("\nblend2 (LOPO OOF) vs stylometric:\n", chg_s.to_string(index=False))
cm_b = plot_confusion(y_val, pred_oof["blend2"], "blend2 QLoRA+stylometric (LOPO out-of-fold)", FIG / "confusion_blend2_oof.png")
tc_b = top_confusions(y_val, pred_oof["blend2"])
print(tc_b.to_string())

fig, ax = plt.subplots(figsize=(12, 4.5))
x = np.arange(18)
for k, (name, col) in enumerate([("exp01 QLoRA", "C0"), ("stylometric", "C1"), ("blend2 (LOPO OOF)", "C2")]):
    ax.bar(x + (k - 1) * 0.27, recall_tbl[name], 0.27, label=name, color=col)
ax.set_xticks(x, model_names, rotation=70, ha="right", fontsize=8)
ax.set_ylabel("recall (23 rows/class)"); ax.legend(); ax.set_title("per-class recall on the exp01 validation split")
fig.tight_layout(); fig.savefig(FIG / "per_class_recall.png", dpi=110); plt.close(fig)

RESULTS["comparison"] = comparison.to_dict(orient="index")
RESULTS["per_class_recall"] = recall_tbl.to_dict(orient="index")
RESULTS["bootstrap_by_prompt"] = boot
RESULTS["blend2_lopo_weights"] = {"a_qlora_mean": float(weights[:, 0].mean()), "a_qlora_std": float(weights[:, 0].std()),
                                  "b_stylo_mean": float(weights[:, 1].mean()), "b_stylo_std": float(weights[:, 1].std())}
RESULTS["blend2_vs_qlora_class_changes"] = chg_q.to_dict(orient="records")
RESULTS["blend2_vs_stylo_class_changes"] = chg_s.to_dict(orient="records")
RESULTS["blend2"] = {"confusion_matrix": cm_b.tolist(), "top_confusions": tc_b.to_dict(orient="records")}
(ART / "exp02_results.json").write_text(json.dumps(RESULTS, indent=1, default=float))
print("wrote", ART / "exp02_results.json")
