"""H5 submission logic that runs after the shared Qwen forward pass (CPU; inlined into the Kaggle notebook by
build_notebook.py, tested offline in tests/).

H5 = equal-weight log-linear pool of three components, exactly as in exp03/qlora_armh/analyze_armh.py:
  probe_armh   C=0.03 StandardScaler+LR probe (probe_armh.npz) on the pooled score-input hidden state
  armh_head    softmax of the epoch-6 Arm H classification head logits
  stylo_exp03  char+word+style TF-IDF/LR pipeline (stylo_exp03_model.joblib)
blend = protocol.logprob_blend([log p_probe, log p_head, log p_stylo]) (weights all 1, nothing fitted).
"""
# <repo-imports> (replaced by the inlined protocol.py functions in the notebook)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "exp03" / "stylo"))
from protocol import CLASS_NAMES, log_proba, logprob_blend  # noqa: E402
# </repo-imports>
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score

H5_COMPONENTS = ("probe_armh", "armh_head", "stylo_exp03")


def component_probas(hidden, logits, texts, probe, stylo):
    """Per-component probabilities (n, 18) in CLASS_NAMES order from one shared forward pass.

    The probe was fit and scored on pooled_hidden_armh.npy, which the Arm H kernel stored as float16, so the
    hidden states take the same float32 -> float16 -> float64 path here."""
    H = np.asarray(hidden, dtype=np.float32).astype(np.float16).astype(np.float64)
    return {"probe_armh": probe.predict_proba(H),
            "armh_head": softmax(np.asarray(logits, dtype=np.float32).astype(np.float64), axis=1),
            "stylo_exp03": stylo.predict_proba(pd.DataFrame({"text": list(texts)}))}


def h5_blend(comps):
    """Fixed equal-weight H5 blend (not the H6 dev-LOPO weights)."""
    return logprob_blend([log_proba(comps[k]) for k in H5_COMPONENTS])


def check_class_order(label_map_names, model_config_names, probe_names, stylo_classes, train_models=None):
    """Every component must use the alphabetical 18-class order (class id c = c-th sorted model name).

    train_models (train.csv "model") is only checked when given, i.e. when the optional dev check reads train.csv."""
    names = list(CLASS_NAMES)
    assert len(names) == 18 and names == sorted(names)
    assert list(label_map_names) == names, "adapter label_map.json order differs"
    assert list(model_config_names) == names, "model.config.id2label order differs"
    assert list(probe_names) == names, "probe_armh.npz class_names order differs"
    assert [int(c) for c in stylo_classes] == list(range(18)), "stylo classes_ are not label ids 0..17"
    if train_models is not None:
        assert sorted(set(train_models)) == names, "train.csv model names differ from CLASS_NAMES"
    return names


def dev_reproduction(y, blend, comps, ref_h5, ref_shared, ref_comps, expected_bal_acc, tol, max_dp_shared):
    """Compare the dev run against blend_H5_dev.csv (the reported H5) and the shared-pass reference.

    Returns (report, failures). failures is empty when the run reproduces H5."""
    pred = blend.argmax(1)
    bal = float(balanced_accuracy_score(y, pred))
    rep = {"balanced_accuracy": bal, "expected_balanced_accuracy": expected_bal_acc, "tolerance": tol,
           "argmax_agreement_vs_blend_H5_dev": float((pred == ref_h5.argmax(1)).mean()),
           "rows_differing_argmax_vs_blend_H5_dev": np.where(pred != ref_h5.argmax(1))[0].tolist(),
           "max_abs_proba_diff_vs_blend_H5_dev": float(np.abs(blend - ref_h5).max()),
           "max_abs_proba_diff_vs_shared_pass_reference": float(np.abs(blend - ref_shared).max()),
           "components": {k: {"balanced_accuracy": float(balanced_accuracy_score(y, comps[k].argmax(1))),
                              "max_abs_proba_diff_vs_reference": float(np.abs(comps[k] - ref_comps[k]).max()),
                              "argmax_agreement_vs_reference": float((comps[k].argmax(1) == ref_comps[k].argmax(1)).mean())}
                          for k in H5_COMPONENTS}}
    fails = []
    if abs(bal - expected_bal_acc) > tol:
        fails.append(f"dev balanced accuracy {bal:.6f} differs from {expected_bal_acc:.6f} by more than {tol}")
    if rep["argmax_agreement_vs_blend_H5_dev"] < 1.0:
        fails.append(f"argmax differs from blend_H5_dev.csv on rows {rep['rows_differing_argmax_vs_blend_H5_dev']}")
    if rep["max_abs_proba_diff_vs_shared_pass_reference"] > max_dp_shared:
        fails.append(f"max |p - shared-pass reference| {rep['max_abs_proba_diff_vs_shared_pass_reference']:.3g} > {max_dp_shared}")
    return rep, fails


def validate_submission(sub, test, class_names):
    assert list(sub.columns) == ["id", "model"], list(sub.columns)
    assert len(sub) == len(test), (len(sub), len(test))
    assert sub["id"].notna().all() and sub["id"].is_unique, "ids must be unique and non-null"
    assert sub["id"].tolist() == test["id"].tolist(), "ids must match test.csv (same order)"
    assert sub["model"].notna().all() and set(sub["model"]) <= set(class_names), "labels outside the 18 classes"
    return True


def extrapolate(batch_seconds, batch_rows, n_rows=4000, batch_size=8):
    """Linear extrapolation of the shared forward pass from measured per-batch wall-clock."""
    secs, rows = np.asarray(batch_seconds, dtype=float), np.asarray(batch_rows, dtype=float)
    per_row = secs.sum() / rows.sum()
    return {"measured_batches": int(len(secs)), "measured_rows": int(rows.sum()), "measured_seconds": float(secs.sum()),
            "mean_seconds_per_batch": float(secs.mean()), "p95_seconds_per_batch": float(np.percentile(secs, 95)),
            "seconds_per_row": float(per_row), "rows": n_rows, "batches": int(np.ceil(n_rows / batch_size)),
            "estimated_forward_minutes": float(per_row * n_rows / 60)}
