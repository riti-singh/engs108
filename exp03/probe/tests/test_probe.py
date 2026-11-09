"""Offline tests for the packaged probe (synthetic data; committed artifacts if present)."""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "stylo"))
import protocol as P  # noqa: E402
from probe import ProbeModel, export_npz, fit_probe, load_probe  # noqa: E402


def test_npz_export_matches_sklearn(tmp_path):
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(P.N_CLASSES), 5)
    H = rng.normal(size=(len(y), 32)) + y[:, None] * 0.1
    pipe = fit_probe(H, y, C=0.01)
    export_npz(pipe, tmp_path / "p.npz", P.CLASS_NAMES, {"C": 0.01})
    m = ProbeModel(tmp_path / "p.npz")
    Hn = rng.normal(size=(7, 32))
    assert np.allclose(m.predict_proba(Hn), pipe.predict_proba(Hn), atol=1e-9)
    assert m.class_names == P.CLASS_NAMES and m.meta == {"C": 0.01}
    with pytest.raises(AssertionError):
        m.predict_proba(np.zeros((2, 31)))


@pytest.mark.parametrize("tag", ["exp01", "step0"])
def test_committed_probe_artifact(tag):
    path = HERE / "artifacts" / f"probe_{tag}.npz"
    if not path.exists():
        pytest.skip("artifact not built")
    m = load_probe(path)
    assert m.coef.shape == (P.N_CLASSES, 4096) and m.class_names == P.CLASS_NAMES
    p = m.predict_proba(np.zeros((3, 4096), dtype=np.float16))
    assert p.shape == (3, P.N_CLASSES) and np.allclose(p.sum(1), 1)
