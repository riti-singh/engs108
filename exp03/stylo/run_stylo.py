"""exp03 stylometric feature-group ablations (prompt-grouped CV on train only) and final model.

Usage (from repo root):
  ENGS108_DATA_DIR=<dir with train.csv> python exp03/stylo/run_stylo.py
Writes exp03/stylo/artifacts/: ablation.csv, stylo_results.json, stylo_{exp03,exp02ref}_{oof_train,dev}.csv,
stylo_exp03_meta.json, stylo_exp03_model.joblib (fitted sklearn Pipeline on all 2,285 train rows).
Selection uses only pooled OOF balanced accuracy on the frozen folds; dev is scored once for R0/R1.
"""
import sys
import time
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
import protocol as P  # noqa: E402

ART = P.PROTOCOL_DIR
N_JOBS = 8
TIE = 0.003
MAX_ITER = 5000
T0 = time.time()

train_df, dev_df = P.load_data()
y = train_df["labels"].to_numpy()
y_dev = dev_df["labels"].to_numpy()
g = train_df["prompt_id"].to_numpy()
folds = list(P.iter_folds(train_df))
X_all, X_dev = train_df[["text"]], dev_df[["text"]]


def fit_block(group, fit_idx, score_frames):
    tr = F.make_group(group)
    Xf = tr.fit_transform(X_all.iloc[fit_idx]["text"] if group in ("char", "word") else X_all.iloc[fit_idx]["text"].tolist())
    outs = [tr.transform(s["text"] if group in ("char", "word") else s["text"].tolist()) for s in score_frames]
    return [sp.csr_matrix(Xf)] + [sp.csr_matrix(o) for o in outs]


# blocks[group][k] = (X_fit, X_score) for fold k; blocks[group]["full"] = (X_train, X_dev)
jobs = [(gr, k) for gr in F.GROUPS for k in list(range(P.N_FOLDS)) + ["full"]]
res = Parallel(n_jobs=N_JOBS)(
    delayed(fit_block)(gr, np.arange(len(train_df)) if k == "full" else folds[k][0],
                       [X_dev] if k == "full" else [X_all.iloc[folds[k][1]]]) for gr, k in jobs)
blocks = {gr: {} for gr in F.GROUPS}
for (gr, k), r in zip(jobs, res):
    blocks[gr][k] = tuple(r)
print(f"feature blocks built in {time.time() - T0:.0f}s:",
      {gr: blocks[gr]["full"][0].shape[1] for gr in F.GROUPS}, flush=True)


def design(groups, k):
    return [sp.hstack([blocks[gr][k][i] for gr in groups]).tocsr() for i in (0, 1)]


def fit_fold(groups, C, k):
    Xf, Xs = design(groups, k)
    fit_idx = folds[k][0] if k != "full" else np.arange(len(train_df))
    m = LogisticRegression(C=C, max_iter=MAX_ITER).fit(Xf, y[fit_idx])
    assert list(m.classes_) == list(range(P.N_CLASSES))
    return m.predict_proba(Xs), int(m.n_iter_.max())


def run_configs(configs):
    """configs: list of (name, groups, C). Returns dict name -> (OOF proba, max n_iter)."""
    tasks = [(c, k) for c in configs for k in range(P.N_FOLDS)]
    out = Parallel(n_jobs=N_JOBS)(delayed(fit_fold)(c[1], c[2], k) for c, k in tasks)
    oof = {c[0]: [np.zeros((len(train_df), P.N_CLASSES)), 0] for c in configs}
    for (c, k), (p, it) in zip(tasks, out):
        oof[c[0]][0][folds[k][1]] = p
        oof[c[0]][1] = max(oof[c[0]][1], it)
    return oof


def name_of(groups, C):
    return f"{'+'.join(groups)} | C={C}"


base = F.EXP02_GROUPS
stage1 = [(base, 10), (base, 100)]
stage1 += [(base + [n], 100) for n in F.NEW_GROUPS]
stage1 += [(F.GROUPS, 100)]
stage1 += [([x for x in F.GROUPS if x != d], 100) for d in F.GROUPS]
stage1 += [([n], C) for n in F.NEW_GROUPS for C in (1, 100)]
stage1 += [(F.NEW_GROUPS, 100), (["char"] + F.NEW_GROUPS, 100)]
configs, seen = [], set()
for grs, C in stage1:
    nm = name_of(grs, C)
    if nm not in seen:
        seen.add(nm)
        configs.append((nm, list(grs), C))
oof = run_configs(configs)
print(f"stage 1 ({len(configs)} configs) done at {time.time() - T0:.0f}s", flush=True)


def table(cfgs, stage):
    rows = []
    ref = oof[name_of(base, 100)][0].argmax(1)
    for nm, grs, C in cfgs:
        p, it = oof[nm]
        m = P.metrics(y, p)
        ci = P.prompt_bootstrap_ci(y, p.argmax(1), g)
        d = P.paired_bootstrap_delta(y, p.argmax(1), ref, g)
        rows.append({"stage": stage, "config": nm, "groups": "+".join(grs), "n_groups": len(grs), "C": C,
                     "n_features": sum(blocks[gr]["full"][0].shape[1] for gr in grs),
                     "oof_bal_acc": m["balanced_accuracy"], "oof_bal_acc_ci95": ci["ci95"],
                     "oof_macro_f1": m["macro_f1"], "oof_log_loss": m["log_loss"],
                     "delta_vs_exp02_base": d["delta"], "delta_ci95": d["ci95"], "p_delta_le_0": d["p_le_0"],
                     "max_n_iter": it})
    return rows


rows = table(configs, 1)
# Stage 2: C sweep for the 3 best stage-1 multi-group configurations.
s1 = pd.DataFrame(rows)
top = s1[s1.n_groups > 1].sort_values("oof_bal_acc", ascending=False).head(3)
cfg2 = []
for grs in top["groups"]:
    for C in (10, 30, 300):
        nm = name_of(grs.split("+"), C)
        if nm not in seen:
            seen.add(nm)
            cfg2.append((nm, grs.split("+"), C))
oof.update(run_configs(cfg2))
rows += table(cfg2, 2)
abl = pd.DataFrame(rows).sort_values("oof_bal_acc", ascending=False).reset_index(drop=True)
print(f"stage 2 ({len(cfg2)} configs) done at {time.time() - T0:.0f}s", flush=True)

# Selection: best pooled OOF bal-acc; within TIE of the best, prefer fewer groups, then smaller C.
best = abl.loc[0, "oof_bal_acc"]
cands = abl[abl.oof_bal_acc >= best - TIE].sort_values(["n_groups", "C", "oof_bal_acc"],
                                                         ascending=[True, True, False])
sel = cands.iloc[0]
SEL_GROUPS, SEL_C = sel["groups"].split("+"), float(sel["C"])
SEL_C = int(SEL_C) if SEL_C.is_integer() else SEL_C
print("selected:", sel["config"], "OOF", round(sel["oof_bal_acc"], 4), "| best", abl.loc[0, "config"], round(best, 4))
abl.to_csv(ART / "ablation.csv", index=False, float_format="%.5g")
print(abl[["stage", "config", "n_features", "oof_bal_acc", "oof_macro_f1", "oof_log_loss",
           "delta_vs_exp02_base", "delta_ci95", "max_n_iter"]].round(4).to_string(), flush=True)

# Final refits on all train rows (sklearn Pipeline = packaged model) and the single dev scoring.
results = {"protocol": "exp03/eval_protocol.md", "seed": P.SEED, "versions": P.library_versions(),
           "train_csv_sha256": P.TRAIN_CSV_SHA256, "folds_sha256": P.sha256_file(P.FOLDS_FILE),
           "tie_rule": f"within {TIE} of best OOF bal-acc -> fewest groups, then smallest C",
           "selected": {"config": sel["config"], "groups": SEL_GROUPS, "C": SEL_C},
           "best_oof_config": abl.loc[0, "config"], "dev_evaluations": []}
for tag, grs, C, rid in [("exp03", SEL_GROUPS, SEL_C, "R1"), ("exp02ref", base, 100, "R0")]:
    pipe = Pipeline([("features", F.make_features(grs)), ("clf", LogisticRegression(C=C, max_iter=MAX_ITER))])
    pipe.fit(X_all, y)
    assert list(pipe.classes_) == list(range(P.N_CLASSES))
    p_dev = pipe.predict_proba(X_dev)
    # Same design matrix (checked to 1e-15) via the cached blocks; lbfgs at large C is sensitive to
    # float summation order, so record the refit-to-refit difference as a numerical-noise diagnostic.
    p_dev_cached, _ = fit_fold(grs, C, "full")
    refit_noise = {"max_abs_proba_diff": float(np.abs(p_dev - p_dev_cached).max()),
                   "argmax_agreement": float((p_dev.argmax(1) == p_dev_cached.argmax(1)).mean()),
                   "bal_acc_cached_path": P.metrics(y_dev, p_dev_cached)["balanced_accuracy"]}
    p_oof = oof[name_of(grs, C)][0]
    P.write_predictions(ART / f"stylo_{tag}_oof_train.csv", train_df, p_oof, fold=train_df["fold"])
    P.write_predictions(ART / f"stylo_{tag}_dev.csv", dev_df, p_dev)
    m_oof, m_dev = P.metrics(y, p_oof), P.metrics(y_dev, p_dev)
    ci_dev = P.prompt_bootstrap_ci(y_dev, p_dev.argmax(1), dev_df["prompt_id"])
    ci_dev_f1 = P.prompt_bootstrap_ci(y_dev, p_dev.argmax(1), dev_df["prompt_id"], metric="macro_f1")
    meta = {"model": f"stylo_{tag}", "report_id": rid, "config": name_of(grs, C), "groups": grs, "C": C,
            "classifier": f"LogisticRegression(C={C}, max_iter={MAX_ITER}, lbfgs, multinomial)",
            "oof_honest": True, "oof": m_oof, "oof_bal_acc_ci95": P.prompt_bootstrap_ci(y, p_oof.argmax(1), g)["ci95"],
            "dev": m_dev, "dev_bal_acc_ci95": ci_dev["ci95"], "dev_macro_f1_ci95": ci_dev_f1["ci95"],
            "refit_numerical_noise": refit_noise,
            "n_features": sum(blocks[gr]["full"][0].shape[1] for gr in grs),
            "versions": results["versions"], "train_csv_sha256": P.TRAIN_CSV_SHA256, "seed": P.SEED}
    P.write_json(ART / f"stylo_{tag}_meta.json", meta)
    results[f"stylo_{tag}"] = meta
    results["dev_evaluations"].append({"id": rid, "model": f"stylo_{tag}", **m_dev})
    if tag == "exp03":
        joblib.dump(pipe, ART / "stylo_exp03_model.joblib", compress=3)
        results["model_artifact"] = {"path": "exp03/stylo/artifacts/stylo_exp03_model.joblib",
                                     "sha256": P.sha256_file(ART / "stylo_exp03_model.joblib")}
    else:
        ref = pd.read_csv(P.EXP02_ART / "stylometric_val_proba.csv").set_index("id").loc[dev_df["id"]]
        results["exp02ref_matches_exp02_val_proba_max_abs_diff"] = float(
            np.abs(ref[P.PROBA_COLS].to_numpy() - p_dev).max())
    print(tag, "OOF", m_oof, "| dev", m_dev, ci_dev, flush=True)

results["ablation_csv"] = "exp03/stylo/artifacts/ablation.csv"
results["runtime_s"] = time.time() - T0
P.write_json(ART / "stylo_results.json", results)
print(f"done in {time.time() - T0:.0f}s")
