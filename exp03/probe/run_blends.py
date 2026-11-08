"""Pre-registered dev reports (exp03/eval_protocol.md §6): probe x stylometric combinations.

Usage (from repo root, after run_stylo.py and fit_probe.py):
  ENGS108_DATA_DIR=<dir with train.csv> python exp03/probe/run_blends.py
Reads only committed prediction files (no refits, no hidden states needed); writes
exp03/probe/artifacts/blend_results.json, blend_table.csv and blend_*_dev.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stylo"))
import protocol as P  # noqa: E402

ART = HERE / "artifacts"
STY = P.PROTOCOL_DIR

train_df, dev_df = P.load_data()
y, y_dev, g_dev = train_df["labels"].to_numpy(), dev_df["labels"].to_numpy(), dev_df["prompt_id"].to_numpy()

dev = {"stylo_exp03": P.read_predictions(STY / "stylo_exp03_dev.csv", dev_df),
       "stylo_exp02ref": P.read_predictions(STY / "stylo_exp02ref_dev.csv", dev_df),
       "probe_exp01": P.read_predictions(ART / "probe_exp01_dev.csv", dev_df),
       "probe_step0": P.read_predictions(ART / "probe_step0_dev.csv", dev_df)}
oof = {k: P.read_predictions(d / f"{k}_oof_train.csv", train_df)
       for k, d in [("stylo_exp03", STY), ("probe_exp01", ART), ("probe_step0", ART)]}
q = pd.read_parquet(P.EXP02_ART / "exp01_val_logits.parquet").set_index("id").loc[dev_df["id"]]
dev["qlora_exp01"] = softmax(q[[f"logit_{n}" for n in P.CLASS_NAMES]].to_numpy(np.float64), axis=1)
exp02_stylo = pd.read_csv(P.EXP02_ART / "stylometric_val_proba.csv").set_index("id").loc[dev_df["id"]][P.PROBA_COLS].to_numpy()
L = {k: P.log_proba(v) for k, v in dev.items()}

reports, weights = {}, {}
reports["R0 stylo_exp02ref"] = dev["stylo_exp02ref"]
reports["R1 stylo_exp03"] = dev["stylo_exp03"]
reports["R2 probe_exp01"] = dev["probe_exp01"]
reports["R3 probe_step0"] = dev["probe_step0"]
reports["B1 equal: probe_exp01 + stylo_exp03"] = P.logprob_blend([L["probe_exp01"], L["stylo_exp03"]])
reports["B2 equal: probe_exp01 + stylo_exp03 + qlora"] = P.logprob_blend([L["probe_exp01"], L["stylo_exp03"], L["qlora_exp01"]])
for rid, comps in [("B3", ["probe_exp01", "stylo_exp03"]), ("B4", ["probe_exp01", "stylo_exp03", "qlora_exp01"])]:
    p, w = P.lopo_blend([L[c] for c in comps], y_dev, g_dev)
    name = f"{rid} dev-LOPO: {' + '.join(comps)}"
    reports[name] = p
    weights[name] = {"components": comps, "mean": w.mean(0).tolist(), "std": w.std(0).tolist(),
                     "min": w.min(0).tolist(), "max": w.max(0).tolist()}
for rid, comps in [("B5", ["probe_step0", "stylo_exp03"]), ("B6", ["probe_exp01", "stylo_exp03"])]:
    w = P.fit_blend_weights([P.log_proba(oof[c]) for c in comps], y)
    name = f"{rid} train-OOF weights: {' + '.join(comps)}" + (" (biased, sensitivity)" if rid == "B6" else "")
    reports[name] = P.logprob_blend([L[c] for c in comps], w)
    weights[name] = {"components": comps, "weights": w.tolist(),
                     "oof_bal_acc": P.metrics(y, P.logprob_blend([P.log_proba(oof[c]) for c in comps], w))["balanced_accuracy"]}

refs = {
        "R2 probe_exp01": dev["probe_exp01"]}
refs["exp02 equal blend (qlora + stylo_exp02)"] = P.logprob_blend([L["qlora_exp01"], P.log_proba(exp02_stylo)])
ref_pred = {k: v.argmax(1) for k, v in refs.items()}

rows = []
for name, p in reports.items():
    m = P.metrics(y_dev, p)
    row = {"report": name, **m,
           "bal_acc_ci95": P.prompt_bootstrap_ci(y_dev, p.argmax(1), g_dev)["ci95"],
           "macro_f1_ci95": P.prompt_bootstrap_ci(y_dev, p.argmax(1), g_dev, metric="macro_f1")["ci95"]}
    for rn, rp in ref_pred.items():
        d = P.paired_bootstrap_delta(y_dev, p.argmax(1), rp, g_dev)
        row[f"delta_vs[{rn}]"] = d["delta"]
        row[f"delta_ci95_vs[{rn}]"] = d["ci95"]
        row[f"p_le_0_vs[{rn}]"] = d["p_le_0"]
    rows.append(row)
    if name.startswith("B"):
        P.write_predictions(ART / f"blend_{name.split()[0]}_dev.csv", dev_df, p)
tbl = pd.DataFrame(rows)
ref_metrics = {k: P.metrics(y_dev, v) for k, v in refs.items()}
print("references:", ref_metrics)
print(tbl[["report", "balanced_accuracy", "bal_acc_ci95", "macro_f1", "log_loss",
           "delta_vs[exp02 equal blend (qlora + stylo_exp02)]", "delta_ci95_vs[exp02 equal blend (qlora + stylo_exp02)]",
           "delta_vs[R2 probe_exp01]", "delta_ci95_vs[R2 probe_exp01]"]].round(4).to_string())

# diagnostics on the primary blend: complementarity of the two components
pa, ps = dev["probe_exp01"].argmax(1), dev["stylo_exp03"].argmax(1)
comp = {"agreement": float((pa == ps).mean()), "both_correct": float(((pa == y_dev) & (ps == y_dev)).mean()),
        "probe_only_correct": float(((pa == y_dev) & (ps != y_dev)).mean()),
        "stylo_only_correct": float(((pa != y_dev) & (ps == y_dev)).mean()),
        "oracle_either": float(((pa == y_dev) | (ps == y_dev)).mean()),
        "error_correlation": float(np.corrcoef(pa != y_dev, ps != y_dev)[0, 1])}
fam = np.array([n.split("-")[0].rstrip("0123456789.") for n in P.CLASS_NAMES])
b1 = reports["B1 equal: probe_exp01 + stylo_exp03"].argmax(1)
comp["B1_family_accuracy"] = float((fam[b1] == fam[y_dev]).mean())
from sklearn.metrics import recall_score  # noqa: E402
per_class = pd.DataFrame({k.split()[0]: recall_score(y_dev, v.argmax(1), average=None) for k, v in reports.items()},
                         index=P.CLASS_NAMES).round(3)
print(per_class.to_string())
print(comp)
tbl.to_csv(ART / "blend_table.csv", index=False, float_format="%.5g")
P.write_json(ART / "blend_results.json", {
    "protocol": "exp03/eval_protocol.md §5-§6", "primary": "B1 equal: probe_exp01 + stylo_exp03",
    "references": ref_metrics, "reports": rows, "weights": weights, "complementarity_probe_vs_stylo": comp,
    "per_class_recall": per_class.to_dict(orient="index"), "versions": P.library_versions(),
    "bootstrap": {"n": P.N_BOOT, "seed": P.SEED, "unit": "dev prompt (23)"},
    "dev_evaluations": [{"id": r["report"].split()[0], "balanced_accuracy": r["balanced_accuracy"]} for r in rows]})
