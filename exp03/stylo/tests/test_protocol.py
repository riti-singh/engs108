"""Offline tests for the frozen protocol and feature groups (no network). Data-dependent tests
skip when ENGS108_DATA_DIR has no train.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import features as F  # noqa: E402
import protocol as P  # noqa: E402

HAVE_DATA = (P.DATA_DIR / "train.csv").exists()
TEXTS = ["Hello, world! It's a “test” — isn't it?\n\n- item one\n- item 2: 3.5%",
         "# Heading\n\nShort. Another sentence; with a semicolon... And **bold** text.",
         ""]


def test_class_order_matches_label_map():
    import json
    lm = json.loads((P.REPO / "label_map.json").read_text()) if (P.REPO / "label_map.json").exists() else None
    assert P.CLASS_NAMES == sorted(P.CLASS_NAMES) and len(P.CLASS_NAMES) == P.N_CLASSES
    if lm is not None:
        names = lm if isinstance(lm, list) else [k for k, _ in sorted(lm.items(), key=lambda kv: kv[1])]
        assert names == P.CLASS_NAMES


def test_frozen_folds_file():
    f = pd.read_csv(P.FOLDS_FILE)
    assert len(f) == 2285 and f["id"].is_unique
    assert f.groupby("prompt_id")["fold"].nunique().max() == 1
    assert sorted(f["fold"].value_counts().tolist()) == [450, 450, 450, 467, 468]
    assert not set(f["prompt_id"]) & set(P.DEV_PROMPTS)


@pytest.mark.skipif(not HAVE_DATA, reason="competition train.csv not available")
def test_split_and_folds_reproduce():
    train_df, dev_df = P.load_data()
    assert len(train_df) == 2285 and len(dev_df) == 414
    assert sorted(dev_df["prompt_id"].unique()) == P.DEV_PROMPTS
    assert (dev_df["labels"].value_counts() == 23).all()
    for fi, si in P.iter_folds(train_df):
        assert not set(train_df["prompt_id"].iloc[fi]) & set(train_df["prompt_id"].iloc[si])


def test_metrics_and_bootstrap_deterministic():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(P.N_CLASSES), 6)
    g = np.repeat([f"p{i}" for i in range(18)], 6)
    pred = np.where(rng.random(len(y)) < 0.5, y, rng.integers(0, P.N_CLASSES, len(y)))
    a = P.prompt_bootstrap_ci(y, pred, g, n_boot=200)
    b = P.prompt_bootstrap_ci(y, pred, g, n_boot=200)
    assert a == b and a["ci95"][0] <= a["value"] <= a["ci95"][1]
    d = P.paired_bootstrap_delta(y, pred, pred, g, n_boot=100)
    assert d["delta"] == 0
    assert P.metrics(y, np.eye(P.N_CLASSES)[y])["balanced_accuracy"] == 1.0


def test_logprob_blend_and_lopo():
    rng = np.random.default_rng(1)
    y = np.repeat(np.arange(P.N_CLASSES), 4)
    g = np.repeat([f"p{i}" for i in range(12)], 6)
    ps = [rng.dirichlet(np.ones(P.N_CLASSES), len(y)) for _ in range(2)]
    eq = P.logprob_blend([P.log_proba(p) for p in ps])
    assert np.allclose(eq.sum(1), 1)
    w = P.fit_blend_weights([P.log_proba(p) for p in ps], y)
    assert w.shape == (2,) and (w >= 0).all()
    pl, W = P.lopo_blend([P.log_proba(p) for p in ps], y, g)
    assert pl.shape == ps[0].shape and np.allclose(pl.sum(1), 1) and W.shape == (12, 2)


def test_prediction_artifact_roundtrip(tmp_path):
    meta = pd.DataFrame({"id": ["b", "a", "c"], "prompt_id": ["p1", "p1", "p2"], "labels": [0, 5, 17]})
    p = np.random.default_rng(2).dirichlet(np.ones(P.N_CLASSES), 3)
    P.write_predictions(tmp_path / "x.csv", meta, p, fold=[0, 1, 2])
    df = pd.read_csv(tmp_path / "x.csv")
    assert list(df.columns) == P.META_COLS + ["fold"] + P.PROBA_COLS
    back = P.read_predictions(tmp_path / "x.csv", meta.iloc[[2, 0, 1]])
    assert np.allclose(back, p[[2, 0, 1]], atol=1e-6)


@pytest.mark.parametrize("fn", [F.style_features, F.function_word_features, F.typography_features,
                                F.sentence_features])
def test_dense_features_finite(fn):
    X = np.asarray(fn(TEXTS))
    assert X.shape[0] == len(TEXTS) and np.isfinite(X).all()


def test_all_groups_build():
    for name in F.GROUPS:
        X = F.make_group(name).fit_transform(TEXTS * 3)
        assert X.shape[0] == 9
    assert F.shape_docs(["Hello 42!"])[0] and F.fw_masked_docs(["the cat"])[0]
