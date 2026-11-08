"""Fit, select and package the linear probes on exp02's pooled hidden states (CPU only, no Qwen compute).

Usage (from repo root):
  exp02/fetch_pooled_hidden.sh <hidden_dir>    # private Kaggle dataset, sha256-verified
  ENGS108_DATA_DIR=<dir with train.csv> EXP03_HIDDEN_DIR=<hidden_dir> python exp03/probe/fit_probe.py
Writes exp03/probe/artifacts/: probe_{exp01,step0}.npz (packaged models), probe_*_{oof_train,dev}.csv,
probe_*_meta.json, probe_results.json.

exp01 probe: C=0.001 fixed (chosen in exp02 by GroupKFold(5) on train; kept so it reproduces 0.377).
step0 probe: C chosen here by pooled OOF bal-acc on the frozen folds (protocol §3).
The exp01 OOF is not honest (the exp01 adapter was trained on every train row), see protocol §5.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stylo"))
sys.path.insert(0, str(HERE))
import protocol as P  # noqa: E402
from probe import ProbeModel, export_npz, fit_probe  # noqa: E402

ART = HERE / "artifacts"
C_GRID = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)
FIXED_C = {"exp01": 0.001}
T0 = time.time()

train_df, dev_df = P.load_data()
y, y_dev = train_df["labels"].to_numpy(), dev_df["labels"].to_numpy()
g = train_df["prompt_id"].to_numpy()
folds = list(P.iter_folds(train_df))
expected = dict(l.split()[::-1] for l in (P.EXP02_ART / "pooled_hidden_SHA256SUMS").read_text().splitlines())
results = {"protocol": "exp03/eval_protocol.md", "seed": P.SEED, "versions": P.library_versions(),
           "train_csv_sha256": P.TRAIN_CSV_SHA256, "folds_sha256": P.sha256_file(P.FOLDS_FILE),
           "hidden_states": {"kaggle_dataset": "ritisingh123/engs108-exp02-pooled-hidden-states",
                             "producer": "Kaggle kernel ritisingh123/rs2027-engs108-exp02-val-inference v2",
                             "sha256": expected},
           "c_grid": C_GRID, "dev_evaluations": []}


def oof_for(H, C):
    def one(k):
        fi, si = folds[k]
        return si, fit_probe(H[fi], y[fi], C).predict_proba(H[si])
    out = np.zeros((len(y), P.N_CLASSES))
    for si, p in Parallel(n_jobs=P.N_FOLDS)(delayed(one)(k) for k in range(P.N_FOLDS)):
        out[si] = p
    return out


for tag, rid in [("exp01", "R2"), ("step0", "R3")]:
    H_tr, H_dev = P.load_hidden(tag, train_df, dev_df)
    sweep = []
    oofs = {}
    for C in C_GRID:
        oofs[C] = oof_for(H_tr, C)
        fold_scores = [P.metrics(y[si], oofs[C][si])["balanced_accuracy"] for _, si in folds]
        sweep.append({"C": C, **P.metrics(y, oofs[C]), "mean_fold_bal_acc": float(np.mean(fold_scores))})
        print(tag, sweep[-1], f"{time.time() - T0:.0f}s", flush=True)
    sweep = pd.DataFrame(sweep)
    best = sweep["balanced_accuracy"].max()
    # tie rule (protocol §3): within 0.003 of the best -> the most regularised (smallest C)
    c_sel = float(sweep[sweep["balanced_accuracy"] >= best - 0.003]["C"].min())
    C = FIXED_C.get(tag, c_sel)
    pipe = fit_probe(H_tr, y, C)
    p_dev = pipe.predict_proba(H_dev)
    npz = ART / f"probe_{tag}.npz"
    meta = {"model": f"probe_{tag}", "report_id": rid, "C": C, "C_selected_by_protocol_oof": c_sel,
            "C_source": "fixed from exp02 (GroupKFold(5) mean-fold CV, grid 0.001/0.01/0.1)" if tag in FIXED_C
            else "protocol pooled-OOF selection",
            "features": f"pooled_hidden_{tag}.npy (2699 x 4096 float16 -> float64)", "hidden_sha256": expected[f"pooled_hidden_{tag}.npy"],
            "estimator": "StandardScaler + LogisticRegression(lbfgs, multinomial, max_iter=3000)",
            "oof_honest": tag == "step0",
            "oof_note": None if tag == "step0" else "exp01 adapter was trained on all train rows: OOF is optimistic",
            "versions": results["versions"], "seed": P.SEED, "train_csv_sha256": P.TRAIN_CSV_SHA256}
    export_npz(pipe, npz, P.CLASS_NAMES, meta)
    p_np = ProbeModel(npz).predict_proba(H_dev)
    assert np.allclose(p_np, p_dev, atol=1e-9), np.abs(p_np - p_dev).max()
    p_oof = oofs[C] if C in oofs else oof_for(H_tr, C)
    P.write_predictions(ART / f"probe_{tag}_oof_train.csv", train_df, p_oof, fold=train_df["fold"])
    P.write_predictions(ART / f"probe_{tag}_dev.csv", dev_df, p_dev)
    m_dev = P.metrics(y_dev, p_dev)
    meta.update({"npz_sha256": P.sha256_file(npz), "oof": P.metrics(y, p_oof), "dev": m_dev,
                 "dev_bal_acc_ci95": P.prompt_bootstrap_ci(y_dev, p_dev.argmax(1), dev_df["prompt_id"])["ci95"],
                 "dev_macro_f1_ci95": P.prompt_bootstrap_ci(y_dev, p_dev.argmax(1), dev_df["prompt_id"],
                                                            metric="macro_f1")["ci95"],
                 "npz_vs_sklearn_max_abs_diff": float(np.abs(p_np - p_dev).max()),
                 "c_sweep": sweep.to_dict(orient="records")})
    ref = pd.read_csv(P.EXP02_ART / f"linear_probe_{tag}_val_proba.csv").set_index("id").loc[dev_df["id"]]
    meta["matches_exp02_val_proba_max_abs_diff"] = float(np.abs(ref[P.PROBA_COLS].to_numpy() - p_dev).max())
    meta["exp02_val_argmax_agreement"] = float((ref[P.PROBA_COLS].to_numpy().argmax(1) == p_dev.argmax(1)).mean())
    P.write_json(ART / f"probe_{tag}_meta.json", meta)
    results[f"probe_{tag}"] = meta
    results["dev_evaluations"].append({"id": rid, "model": f"probe_{tag}", **m_dev})
    print(tag, "C", C, "dev", m_dev, "exp02 diff", meta["matches_exp02_val_proba_max_abs_diff"], flush=True)

results["runtime_s"] = time.time() - T0
P.write_json(ART / "probe_results.json", results)
print(f"done in {time.time() - T0:.0f}s")
