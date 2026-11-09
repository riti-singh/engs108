"""Exp03 follow-up: 4-view stylegram LinearSVC vs R1 (stylo_exp03 LR) and exp02's calibrated LinearSVC.

Frozen protocol (exp03/eval_protocol.md): GroupKFold(5) pooled OOF on the 2,285 train rows; selection by
pooled OOF balanced accuracy with the 0.003 tie rule (-> smallest C); dev scored only for the selected model.
Calibration: temperature-scaled softmax of the SVC decision values, temperature fit on train OOF decisions
only (outer fold k uses a temperature fit on the other 4 folds' OOF decisions; dev uses all 5 folds').
Writes artifacts/svc_preregistration.json (before any fit), stylo_svc_{oof_train,dev}.csv, stylo_svc_meta.json,
stylo_svc_exp02ref_oof_train.csv, blend_S2_dev.csv, svc_results.json.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
import protocol as P  # noqa: E402
import stylegram as S  # noqa: E402

ART = P.PROTOCOL_DIR
PROBE_ART = P.REPO / "exp03" / "probe" / "artifacts"
TIE = 0.003
PREREG = {
    "registered_before_fitting": True,
    "candidate": "stylegram 4-view LinearSVC (exp03/stylo/stylegram.py, SPEC from lucasboesen/lora-x-adaptive-centering v5)",
    "svc": {"random_state": P.SEED, "max_iter": 4000},
    "C_grid": [0.1, 0.25, 0.5, 1.0],
    "comparators_oof": {
        "R1": "stylo_exp03 LR (char+word+style, C=300), existing artifacts/stylo_exp03_oof_train.csv",
        "exp02_svc": "exp02 calibrated LinearSVC: char+word+style, CalibratedClassifierCV(LinearSVC(C=0.5, "
                     "max_iter=20000), method='sigmoid', cv=3), refit on the frozen folds",
    },
    "selection": "pooled train OOF balanced accuracy; within 0.003 of best -> smallest C",
    "calibration": "softmax(a * decision); a fit by NLL on train OOF decisions only (cross-fitted for OOF rows)",
    "dev_reports": {
        "S1": "selected stylegram SVC alone",
        "S2": "equal log-prob blend: probe_exp01 + S1",
        "paired_deltas": "each of S1, S2 vs R1 (stylo_exp03 dev) and vs B1 (probe_exp01 + stylo_exp03, 0.3986)",
    },
    "data": "official train.csv only",
}
PREREG_FILE = ART / "svc_preregistration.json"
if PREREG_FILE.exists():
    assert json.loads(PREREG_FILE.read_text()) == PREREG, "pre-registration changed after it was written"
else:
    P.write_json(PREREG_FILE, PREREG)

train_df, dev_df = P.load_data()
y, g = train_df["labels"].to_numpy(), train_df["prompt_id"].to_numpy()
y_dev, g_dev = dev_df["labels"].to_numpy(), dev_df["prompt_id"].to_numpy()
texts, texts_dev = train_df["text"].tolist(), dev_df["text"].tolist()
folds = list(P.iter_folds(train_df))


def fit_svcs(fit_idx, score_texts, Cs):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        vec = S.StylegramVectorizer().fit([texts[i] for i in fit_idx])
        X, Xs = vec.transform([texts[i] for i in fit_idx]), vec.transform(score_texts)
        out = {}
        for C in Cs:
            svc = LinearSVC(C=C, random_state=P.SEED, max_iter=4000).fit(X, y[fit_idx])
            assert list(svc.classes_) == list(range(P.N_CLASSES))
            out[C] = svc.decision_function(Xs)
    n_conv = sum(issubclass(x.category, ConvergenceWarning) for x in w)
    return out, n_conv, X.shape[1]


def fit_exp02_svc(fit_idx, score_idx):
    pipe = Pipeline([("features", F.make_features(["char", "word", "style"])),
                     ("clf", CalibratedClassifierCV(LinearSVC(C=0.5, max_iter=20000), method="sigmoid", cv=3))])
    pipe.fit(train_df.iloc[fit_idx], y[fit_idx])
    return pipe.predict_proba(train_df.iloc[score_idx])


def nll(d, yy, a):
    return -log_softmax(a * d, axis=1)[np.arange(len(yy)), yy].mean()


def fit_temp(d, yy):
    r = minimize_scalar(lambda la: nll(d, yy, np.exp(la)), bounds=(-6, 6), method="bounded")
    return float(np.exp(r.x))


def calib(d, a):
    return np.exp(log_softmax(a * d, axis=1))


t0 = time.time()
Cs = PREREG["C_grid"]
jobs = [delayed(fit_svcs)(fi, [texts[i] for i in si], Cs) for fi, si in folds]
jobs += [delayed(fit_exp02_svc)(fi, si) for fi, si in folds]
res = Parallel(n_jobs=8)(jobs)
sg, ex = res[:P.N_FOLDS], res[P.N_FOLDS:]
fold = train_df["fold"].to_numpy()
D = {C: np.zeros((len(y), P.N_CLASSES)) for C in Cs}
for (fi, si), (out, _, _) in zip(folds, sg):
    for C in Cs:
        D[C][si] = out[C]
p_ex = np.zeros((len(y), P.N_CLASSES))
for (fi, si), p in zip(folds, ex):
    p_ex[si] = p
p_ex /= p_ex.sum(1, keepdims=True)
P.write_predictions(ART / "stylo_svc_exp02ref_oof_train.csv", train_df, p_ex, fold=fold)


def oof_calibrated(d):
    p = np.zeros_like(d)
    temps = []
    for k in range(P.N_FOLDS):
        a = fit_temp(d[fold != k], y[fold != k])
        temps.append(a)
        p[fold == k] = calib(d[fold == k], a)
    return p, temps


def fold_scores(pred):
    return [float(balanced_accuracy_score(y[fold == k], pred[fold == k])) for k in range(P.N_FOLDS)]


r1_oof = P.read_predictions(ART / "stylo_exp03_oof_train.csv", train_df)
grid = []
for C in Cs:
    p, temps = oof_calibrated(D[C])
    m = P.metrics(y, p)
    grid.append({"model": f"stylegram_svc C={C}", "C": C, **m, "fold_bal_acc": fold_scores(D[C].argmax(1)),
                 "oof_temps": temps, "convergence_warnings_folds": [s[1] for s in sg],
                 "uncalibrated_softmax_log_loss": float(-log_softmax(D[C], axis=1)[np.arange(len(y)), y].mean())})
comparators = [{"model": "R1 stylo_exp03 LR C=300", **P.metrics(y, r1_oof), "fold_bal_acc": fold_scores(r1_oof.argmax(1))},
               {"model": "exp02 calibrated LinearSVC C=0.5 (char+word+style)", **P.metrics(y, p_ex),
                "fold_bal_acc": fold_scores(p_ex.argmax(1))}]
key = "balanced_accuracy"
best = max(r[key] for r in grid)
sel = min((r for r in grid if r[key] >= best - TIE), key=lambda r: r["C"])
C_sel = sel["C"]
print(json.dumps([{k: r[k] for k in ("model", key, "macro_f1", "log_loss")} for r in grid + comparators], indent=1))
print("selected", C_sel)
for r in comparators:
    r["paired_oof_delta_selected_minus_this"] = P.paired_bootstrap_delta(
        y, D[C_sel].argmax(1), (r1_oof if r["model"].startswith("R1") else p_ex).argmax(1), g)

# full refit on all 2,285 train rows, selected C only; temperature from all OOF decisions
a_dev = fit_temp(D[C_sel], y)
out, n_conv_full, n_feat = fit_svcs(np.arange(len(y)), texts_dev, [C_sel])
p_dev = calib(out[C_sel], a_dev)
p_oof_sel, _ = oof_calibrated(D[C_sel])
P.write_predictions(ART / "stylo_svc_oof_train.csv", train_df, p_oof_sel, fold=fold)
P.write_predictions(ART / "stylo_svc_dev.csv", dev_df, p_dev)

r1_dev = P.read_predictions(ART / "stylo_exp03_dev.csv", dev_df)
probe_dev = P.read_predictions(PROBE_ART / "probe_exp01_dev.csv", dev_df)
b1_dev = P.read_predictions(PROBE_ART / "blend_B1_dev.csv", dev_df)
s2 = P.logprob_blend([P.log_proba(probe_dev), P.log_proba(p_dev)])
P.write_predictions(ART / "blend_S2_dev.csv", dev_df, s2)
dev = {}
for name, p in [("S1 stylegram_svc", p_dev), ("S2 equal: probe_exp01 + stylegram_svc", s2)]:
    dev[name] = {**P.metrics(y_dev, p), "ci95_bal_acc": P.prompt_bootstrap_ci(y_dev, p.argmax(1), g_dev)["ci95"],
                 "delta_vs_R1": P.paired_bootstrap_delta(y_dev, p.argmax(1), r1_dev.argmax(1), g_dev),
                 "delta_vs_B1": P.paired_bootstrap_delta(y_dev, p.argmax(1), b1_dev.argmax(1), g_dev)}
dev["reference R1 stylo_exp03"] = P.metrics(y_dev, r1_dev)
dev["reference B1"] = P.metrics(y_dev, b1_dev)
print(json.dumps(dev, indent=1))

meta = {"model": "stylo_svc", "selected_C": C_sel, "dev_temperature": a_dev, "n_features_full_fit": int(n_feat),
        "convergence_warnings_full_fit": n_conv_full, "spec": S.SPEC, "prereg": "artifacts/svc_preregistration.json",
        "protocol": "exp03/eval_protocol.md §8", "seed": P.SEED, "train_csv_sha256": P.TRAIN_CSV_SHA256,
        "folds_sha256": P.sha256_file(P.FOLDS_FILE), "versions": P.library_versions(),
        "files": ["stylo_svc_oof_train.csv", "stylo_svc_dev.csv"]}
P.write_json(ART / "stylo_svc_meta.json", meta)
P.write_json(ART / "svc_results.json", {"oof_grid": grid, "oof_comparators": comparators, "selected": sel,
                                        "dev": dev, "meta": meta, "seconds": time.time() - t0})
print("done", time.time() - t0)
