"""Offline inference smoke test on the 3-row Kaggle test stub (ENGS108_DATA_DIR/test.csv; skipped if absent).

Loads the packaged probe (.npz) and the stylometric model (joblib), feeds them only test-time inputs
(id/text frame; synthetic pooled hidden states, since test hidden states are not built), and asserts that the
inference path opens no train/dev/label file and sees no `labels`/`model` column.
"""
import builtins
import io
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
STYLO = HERE.parent / "stylo"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STYLO))
import protocol as P  # noqa: E402
from probe import load_probe  # noqa: E402

STUB = Path(os.environ.get("ENGS108_DATA_DIR", "/kaggle/input/competitions/name-that-llm")) / "test.csv"
STYLO_MODEL = STYLO / "artifacts" / "stylo_exp03_model.joblib"
PROBE_NPZ = HERE / "artifacts" / "probe_exp01.npz"
FORBIDDEN = ("train.csv", "label", "_dev", "oof", "folds")


@pytest.fixture
def opened(monkeypatch):
    """Record every path opened via open/pandas/numpy during the test body."""
    seen = []

    def wrap(fn):
        def inner(path, *a, **k):
            if not isinstance(path, io.IOBase):
                seen.append(str(path))
            return fn(path, *a, **k)
        return inner

    for mod, name in [(builtins, "open"), (pd, "read_csv"), (pd, "read_parquet"), (np, "load")]:
        monkeypatch.setattr(mod, name, wrap(getattr(mod, name)))
    return seen


def _check_proba(p):
    assert p.shape == (3, P.N_CLASSES)
    assert np.all(p >= 0) and np.allclose(p.sum(1), 1.0, atol=1e-6)


def test_class_order_is_label_map_order():
    assert P.CLASS_NAMES == sorted(P.CLASS_NAMES) and len(set(P.CLASS_NAMES)) == P.N_CLASSES
    assert P.PROBA_COLS == [f"p_{c}" for c in P.CLASS_NAMES]


@pytest.mark.skipif(not (STUB.exists() and STYLO_MODEL.exists() and PROBE_NPZ.exists()),
                    reason="test stub or committed model artifacts not available")
def test_inference_on_test_stub(opened):
    stub = pd.read_csv(STUB)
    assert len(stub) == 3 and {"labels", "model"}.isdisjoint(stub.columns)
    X = stub[["id", "text"]].copy()

    stylo = joblib.load(STYLO_MODEL)
    probe = load_probe(PROBE_NPZ)
    assert list(stylo.classes_) == list(range(P.N_CLASSES))
    assert probe.class_names == P.CLASS_NAMES

    n_before = len(opened)
    p_stylo = stylo.predict_proba(X)
    H = np.random.default_rng(P.SEED).standard_normal((3, probe.mean.shape[0]))
    assert H.shape == (3, 4096)
    p_probe = probe.predict_proba(H)
    p_blend = P.logprob_blend([P.log_proba(p_probe), P.log_proba(p_stylo)])
    inference_reads = opened[n_before:]

    for p in (p_stylo, p_probe, p_blend):
        _check_proba(p)
    assert inference_reads == [], inference_reads
    loads = [s.lower() for s in opened if s != str(STUB)]
    assert not [s for s in loads if any(f in Path(s).name for f in FORBIDDEN)], loads
    sub = pd.DataFrame({"id": stub["id"], "model": [P.CLASS_NAMES[i] for i in p_blend.argmax(1)]})
    assert set(sub["model"]) <= set(P.CLASS_NAMES)
