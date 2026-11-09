"""exp03 Arm H analysis: protocol prediction files, the new probe on adapted states, the pre-registered
dev reports H1-H6 (config.json) and the diagnostics. CPU only, no network.

  ARMH_OUT_DIR=~/data/armh_out ENGS108_DATA_DIR=~/data/nthl ~/venvs/engs108/bin/python exp03/qlora_armh/analyze_armh.py

ARMH_OUT_DIR is the downloaded output of Kaggle kernel ritisingh123/rs2027-engs108-exp03-armh; every file used
is checked against artifacts/kernel_output_SHA256SUMS.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import softmax
from sklearn.metrics import recall_score

HERE = Path(__file__).resolve().parent
EXP03 = HERE.parent
sys.path.insert(0, str(EXP03 / "stylo"))
sys.path.insert(0, str(EXP03 / "probe"))
import protocol as P  # noqa: E402  (read-only import)
from probe import ProbeModel, export_npz, fit_probe  # noqa: E402  (read-only import)

ART = Path(os.environ.get("ARMH_ART_DIR", HERE / "artifacts"))
OUT_DIR = Path(os.environ.get("ARMH_OUT_DIR", Path.home() / "data" / "armh_out")).expanduser()
CFG = json.loads((HERE / "config.json").read_text())
C_GRID = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)
assert len(C_GRID) == 6
LOGIT_COLS = [f"logit_{n}" for n in P.CLASS_NAMES]
T0 = time.time()


def checked(fn):
    expected = dict(l.split(maxsplit=1)[::-1] for l in (ART / "kernel_output_SHA256SUMS").read_text().splitlines())
    path = OUT_DIR / fn
    assert P.sha256_file(path) == expected[fn], f"{fn} differs from the recorded kernel output"
    return path


def aligned(frame, df):
    f = frame.set_index("id").loc[df["id"]]
    assert (f["prompt_id"].to_numpy() == df["prompt_id"].to_numpy()).all()
    assert (f["labels"].to_numpy() == df["labels"].to_numpy()).all()
    return f


summary = json.loads(checked("run_summary.json").read_text())
assert summary["status"] == "completed" and summary["completed_epoch6"] and not summary["smoke_test"], summary["status"]
assert summary["global_step"] == CFG["training"]["expected_optimizer_steps"]

train_df, dev_df = P.load_data()
y, y_dev, g_dev = train_df["labels"].to_numpy(), dev_df["labels"].to_numpy(), dev_df["prompt_id"].to_numpy()
folds = list(P.iter_folds(train_df))
base_meta = {"protocol": "exp03/eval_protocol.md", "seed": P.SEED, "train_csv_sha256": P.TRAIN_CSV_SHA256,
             "versions": P.library_versions(), "kaggle_kernel": CFG["kaggle"]["kernel"],
             "epoch6_checkpoint": summary["final_checkpoint"], "global_step": summary["global_step"]}

# ---- H1: trained head, length-sorted predict loop ----
sl = pd.read_parquet(checked("armh_sorted_logits.parquet"))
head_dev = softmax(aligned(sl[sl.split == "dev"], dev_df)[LOGIT_COLS].to_numpy(np.float64), axis=1)
head_tr = softmax(aligned(sl[sl.split == "train"], train_df)[LOGIT_COLS].to_numpy(np.float64), axis=1)
P.write_predictions(ART / "armh_head_dev.csv", dev_df, head_dev)
P.write_predictions(ART / "armh_head_train_insample.csv", train_df, head_tr, fold=train_df["fold"])
te = sl[sl.split == "test"]
pd.concat([te[["id"]].reset_index(drop=True),
           pd.DataFrame(softmax(te[LOGIT_COLS].to_numpy(np.float64), axis=1), columns=P.PROBA_COLS)], axis=1).to_csv(
    ART / "armh_head_test.csv", index=False, float_format="%.8g")
head_meta = {**base_meta, "model": "armh_head", "report_id": "H1",
             "source": "armh_sorted_logits.parquet (length-sorted predict loop, batch 8, fp16 autocast)",
             "oof_honest": False, "train_file": "armh_head_train_insample.csv",
             "oof_note": "no train OOF exists: one adapter+head trained on all 2285 train rows; train predictions are in-sample",
             "train_insample": P.metrics(y, head_tr), "dev": P.metrics(y_dev, head_dev)}
P.write_json(ART / "armh_head_meta.json", head_meta)

# ---- H2: new probe on adapted pooled states ----
H = np.load(checked("pooled_hidden_armh.npy")).astype(np.float64)
pm = pd.read_parquet(checked("armh_pooled_pass_logits.parquet"))
assert len(H) == len(pm) == 2699 and H.shape[1] == 4096
pos = {k: i for i, k in enumerate(pm["id"])}
H_tr, H_dev = H[[pos[k] for k in train_df["id"]]], H[[pos[k] for k in dev_df["id"]]]
pool_dev_logits = aligned(pm[pm.split == "dev"], dev_df)[LOGIT_COLS].to_numpy(np.float64)


def oof_for(C):
    def one(k):
        fi, si = folds[k]
        return si, fit_probe(H_tr[fi], y[fi], C).predict_proba(H_tr[si])
    out = np.zeros((len(y), P.N_CLASSES))
    for si, p in Parallel(n_jobs=P.N_FOLDS)(delayed(one)(k) for k in range(P.N_FOLDS)):
        out[si] = p
    return out


sweep, oofs = [], {}
for C in C_GRID:
    oofs[C] = oof_for(C)
    fold_scores = [P.metrics(y[si], oofs[C][si])["balanced_accuracy"] for _, si in folds]
    sweep.append({"C": C, **P.metrics(y, oofs[C]), "mean_fold_bal_acc": float(np.mean(fold_scores))})
    print("probe_armh", sweep[-1], f"{time.time() - T0:.0f}s", flush=True)
sweep = pd.DataFrame(sweep)
best = sweep["balanced_accuracy"].max()
C = float(sweep[sweep["balanced_accuracy"] >= best - 0.003]["C"].min())  # protocol §3 tie rule
pipe = fit_probe(H_tr, y, C)
probe_dev = pipe.predict_proba(H_dev)
npz = ART / "probe_armh.npz"
expected = dict(l.split(maxsplit=1)[::-1] for l in (ART / "kernel_output_SHA256SUMS").read_text().splitlines())
probe_meta = {**base_meta, "model": "probe_armh", "report_id": "H2", "C": C, "C_source": "protocol pooled-OOF selection",
              "features": "pooled_hidden_armh.npy (2699 x 4096 float16 -> float64), epoch-6 adapted score input",
              "hidden_sha256": expected["pooled_hidden_armh.npy"],
              "estimator": "StandardScaler + LogisticRegression(lbfgs, multinomial, max_iter=3000)", "oof_honest": False,
              "oof_note": "Arm H adapter was trained on all train rows: OOF (and the C choice) is optimistic"}
export_npz(pipe, npz, P.CLASS_NAMES, probe_meta)
p_np = ProbeModel(npz).predict_proba(H_dev)
assert np.allclose(p_np, probe_dev, atol=1e-9)
P.write_predictions(ART / "probe_armh_oof_train.csv", train_df, oofs[C], fold=train_df["fold"])
P.write_predictions(ART / "probe_armh_dev.csv", dev_df, probe_dev)
probe_meta.update({"npz_sha256": P.sha256_file(npz), "oof": P.metrics(y, oofs[C]), "dev": P.metrics(y_dev, probe_dev),
                   "c_sweep": sweep.to_dict(orient="records")})
P.write_json(ART / "probe_armh_meta.json", probe_meta)

# ---- pre-registered dev reports ----
sty = P.read_predictions(EXP03 / "stylo" / "artifacts" / "stylo_exp03_dev.csv", dev_df)
b1 = P.read_predictions(EXP03 / "probe" / "artifacts" / "blend_B1_dev.csv", dev_df)
r2 = P.read_predictions(EXP03 / "probe" / "artifacts" / "probe_exp01_dev.csv", dev_df)
L = {"probe_armh": P.log_proba(probe_dev), "armh_head": P.log_proba(head_dev), "stylo_exp03": P.log_proba(sty)}
reports = {"H1 armh_head": head_dev, "H2 probe_armh": probe_dev,
           "H3 equal: probe_armh + stylo_exp03": P.logprob_blend([L["probe_armh"], L["stylo_exp03"]]),
           "H4 equal: armh_head + stylo_exp03": P.logprob_blend([L["armh_head"], L["stylo_exp03"]]),
           "H5 equal: probe_armh + armh_head + stylo_exp03": P.logprob_blend([L["probe_armh"], L["armh_head"], L["stylo_exp03"]])}
h6, w6 = P.lopo_blend([L["probe_armh"], L["armh_head"], L["stylo_exp03"]], y_dev, g_dev)
reports["H6 dev-LOPO: probe_armh + armh_head + stylo_exp03"] = h6
refs = {"B1 (probe_exp01 + stylo_exp03 equal)": b1.argmax(1), "R2 probe_exp01": r2.argmax(1)}
ref_metrics = {"B1": P.metrics(y_dev, b1), "R2": P.metrics(y_dev, r2)}
rows = []
for name, p in reports.items():
    row = {"report": name, **P.metrics(y_dev, p),
           "bal_acc_ci95": P.prompt_bootstrap_ci(y_dev, p.argmax(1), g_dev)["ci95"],
           "macro_f1_ci95": P.prompt_bootstrap_ci(y_dev, p.argmax(1), g_dev, metric="macro_f1")["ci95"]}
    for rn, rp in refs.items():
        d = P.paired_bootstrap_delta(y_dev, p.argmax(1), rp, g_dev)
        row[f"delta_vs[{rn}]"], row[f"delta_ci95_vs[{rn}]"], row[f"p_le_0_vs[{rn}]"] = d["delta"], d["ci95"], d["p_le_0"]
    rows.append(row)
    if name[:2] in ("H3", "H4", "H5", "H6"):
        P.write_predictions(ART / f"blend_{name.split()[0]}_dev.csv", dev_df, p)
tbl = pd.DataFrame(rows)
tbl.to_csv(ART / "armh_report_table.csv", index=False, float_format="%.5g")
print(tbl[["report", "balanced_accuracy", "bal_acc_ci95", "macro_f1", "log_loss",
           "delta_vs[B1 (probe_exp01 + stylo_exp03 equal)]", "delta_ci95_vs[B1 (probe_exp01 + stylo_exp03 equal)]",
           "delta_vs[R2 probe_exp01]", "delta_ci95_vs[R2 probe_exp01]"]].round(4).to_string())

# ---- diagnostics (not reports) ----
diag = {}
diag["H1_vs_H2_head_minus_probe"] = P.paired_bootstrap_delta(y_dev, head_dev.argmax(1), probe_dev.argmax(1), g_dev)
diag["head_probe_argmax_agreement_dev"] = float((head_dev.argmax(1) == probe_dev.argmax(1)).mean())
diag["head_train_insample"] = head_meta["train_insample"]
diag["pooled_vs_sorted_argmax_agreement_dev"] = float((pool_dev_logits.argmax(1) == head_dev.argmax(1)).mean())
diag["pooled_pass_dev_bal_acc"] = P.metrics(y_dev, softmax(pool_dev_logits, axis=1))["balanced_accuracy"]
q = pd.read_parquet(P.EXP02_ART / "exp01_val_logits.parquet").set_index("id").loc[dev_df["id"]]
exp01_head = softmax(q[LOGIT_COLS].to_numpy(np.float64), axis=1)
diag["exp01_head_dev"] = P.metrics(y_dev, exp01_head)
diag["H1_minus_exp01_head"] = P.paired_bootstrap_delta(y_dev, head_dev.argmax(1), exp01_head.argmax(1), g_dev)
ep = np.load(checked("epoch_dev_logits.npy"))  # Trainer evaluate, val_df order == dev_df order
assert ep.shape == (6, len(dev_df), P.N_CLASSES)
diag["per_epoch_dev_trainer_eval"] = [{"epoch": i + 1, **P.metrics(y_dev, softmax(ep[i].astype(np.float64), axis=1))}
                                      for i in range(6)]
st = pd.read_csv(checked("step_log.csv"))
diag["training"] = {
    "steps": int(len(st)), "skipped_steps": int(st["skipped"].sum()),
    "skipped_step_ids": st.loc[st["skipped"], "step"].tolist(),
    "first_step_loss": float(st["train_loss"].iloc[0]), "last_50_steps_mean_loss": float(st["train_loss"].tail(50).mean()),
    "per_epoch_mean_train_loss": st.assign(e=np.ceil(st["epoch_float"] - 1e-9)).groupby("e")["train_loss"].mean().to_dict(),
    "median_preclip_grad_norm": {k: float(st.loc[~st["skipped"], f"grad_norm_{k}_preclip"].median()) for k in ("head", "lora")},
    "frac_steps_clipped": float((st.loc[~st["skipped"], "grad_norm_total_preclip"] > 1.0).mean()),
    "median_update_norm": {k: float(st.loc[~st["skipped"], f"update_norm_{k}"].median()) for k in ("head", "lora")},
    "final_head_cos_init": float(st["head_cos_init"].iloc[-1]), "final_head_delta_frob": float(st["head_delta_frob"].iloc[-1]),
    "ga_fix_check_max_rel_diff": summary["ga_fix_check_max_rel_diff"]}
diag["per_class_recall"] = pd.DataFrame({k.split()[0]: recall_score(y_dev, v.argmax(1), average=None)
                                         for k, v in {**reports, "B1": b1, "R2": r2}.items()}, index=P.CLASS_NAMES).round(3).to_dict(orient="index")
P.write_json(ART / "armh_results.json", {
    "pre_registered": "config.json pre_registered_dev_reports (committed before training)", "references": ref_metrics,
    "reports": rows, "H6_lopo_weights": {"components": ["probe_armh", "armh_head", "stylo_exp03"], "mean": w6.mean(0).tolist(),
                                          "std": w6.std(0).tolist(), "min": w6.min(0).tolist(), "max": w6.max(0).tolist()},
    "diagnostics": diag, "probe_C": C, "versions": P.library_versions(),
    "bootstrap": {"n": P.N_BOOT, "seed": P.SEED, "unit": "dev prompt (23)"}, "runtime_s": time.time() - T0})
print(json.dumps({k: v for k, v in diag.items() if k != "per_class_recall"}, indent=1, default=float))
