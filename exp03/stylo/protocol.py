"""Frozen exp03 evaluation protocol (see exp03/eval_protocol.md). Shared by exp03/stylo and exp03/probe.

Everything that decides which rows a model may see, how folds are built, how metrics and CIs are
computed and what a blendable prediction file looks like lives here, so that every exp03 component
(and later GPU models) is scored the same way.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

SEED = 42
VAL_SIZE = 0.15
N_FOLDS = 5
N_CLASSES = 18
N_BOOT = 2000
PROBA_FLOOR = 1e-12

REPO = Path(__file__).resolve().parents[2]
PROTOCOL_DIR = Path(__file__).resolve().parent / "artifacts"
FOLDS_FILE = PROTOCOL_DIR / "train_folds.csv"
DATA_DIR = Path(os.environ.get("ENGS108_DATA_DIR", "/kaggle/input/competitions/name-that-llm"))
HIDDEN_DIR = Path(os.environ.get("EXP03_HIDDEN_DIR", REPO / "exp02" / "hidden"))
EXP02_ART = REPO / "exp02" / "artifacts"

TRAIN_CSV_SHA256 = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
DEV_PROMPTS = ['p009', 'p012', 'p018', 'p019', 'p026', 'p029', 'p031', 'p036', 'p055',
               'p056', 'p064', 'p068', 'p069', 'p073', 'p076', 'p078', 'p082', 'p104',
               'p110', 'p118', 'p127', 'p132', 'p141']  # exp01 training log

# label_map.json of the exp01 adapter: class id c = c-th model name in sorted order.
CLASS_NAMES = ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
               'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'glm-5.3',
               'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
               'kimi-k3', 'qwen3.8-max']
PROBA_COLS = [f"p_{n}" for n in CLASS_NAMES]
META_COLS = ["id", "prompt_id", "labels"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_split(df):
    """Exp01 split: GroupShuffleSplit(1, test_size=0.15, random_state=42) on prompt_id over train.csv order."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED)
    tr, va = next(splitter.split(df, groups=df["prompt_id"]))
    return df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)


def assert_split(train_df, dev_df):
    assert (len(train_df), train_df["prompt_id"].nunique()) == (2285, 127)
    assert (len(dev_df), dev_df["prompt_id"].nunique()) == (414, 23)
    assert not set(train_df["prompt_id"]) & set(dev_df["prompt_id"])
    assert sorted(dev_df["prompt_id"].unique()) == DEV_PROMPTS
    assert (dev_df["model"].value_counts() == 23).all() and dev_df["model"].nunique() == N_CLASSES


def make_folds(train_df):
    """GroupKFold(5) on prompt_id over the exp01 train rows in split order (deterministic, as in exp02)."""
    fold = np.full(len(train_df), -1)
    for k, (_, idx) in enumerate(GroupKFold(n_splits=N_FOLDS).split(train_df, groups=train_df["prompt_id"])):
        fold[idx] = k
    return fold


def load_data(data_dir=None, check_sha=True):
    """Return (train_df, dev_df) with columns id, prompt_id, text, model, labels, and train_df['fold']."""
    data_dir = Path(data_dir or DATA_DIR)
    if check_sha:
        assert sha256_file(data_dir / "train.csv") == TRAIN_CSV_SHA256, "train.csv differs from the frozen version"
    df = pd.read_csv(data_dir / "train.csv")[["id", "prompt_id", "text", "model"]].copy()
    assert sorted(df["model"].unique()) == CLASS_NAMES
    df["labels"] = df["model"].map({n: i for i, n in enumerate(CLASS_NAMES)}).astype(int)
    train_df, dev_df = make_split(df)
    assert_split(train_df, dev_df)
    train_df["fold"] = make_folds(train_df)
    if FOLDS_FILE.exists():
        frozen = pd.read_csv(FOLDS_FILE)
        assert frozen["id"].tolist() == train_df["id"].tolist(), "train row order differs from frozen folds"
        assert (frozen["fold"].to_numpy() == train_df["fold"].to_numpy()).all(), "folds differ from frozen folds"
    for k in range(N_FOLDS):
        assert not set(train_df.loc[train_df.fold == k, "prompt_id"]) & set(train_df.loc[train_df.fold != k, "prompt_id"])
    return train_df, dev_df


def iter_folds(train_df):
    """Yield (fit_idx, score_idx) positional indices into train_df for the 5 frozen folds."""
    fold = train_df["fold"].to_numpy()
    for k in range(N_FOLDS):
        yield np.where(fold != k)[0], np.where(fold == k)[0]


def metrics(y, proba_or_pred):
    a = np.asarray(proba_or_pred)
    pred = a.argmax(1) if a.ndim == 2 else a
    out = {"balanced_accuracy": float(balanced_accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, average="macro", labels=range(N_CLASSES), zero_division=0))}
    if a.ndim == 2:
        out["log_loss"] = float(log_loss(y, np.clip(a, PROBA_FLOOR, 1), labels=range(N_CLASSES)))
    return out


def _boot_indices(groups, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    prompts = np.unique(groups)
    rows_of = {p: np.where(groups == p)[0] for p in prompts}
    for _ in range(n_boot):
        yield np.concatenate([rows_of[p] for p in rng.choice(prompts, size=len(prompts), replace=True)])


def prompt_bootstrap_ci(y, pred, groups, n_boot=N_BOOT, seed=SEED, metric="balanced_accuracy"):
    """Point estimate and 95% percentile CI, resampling whole prompts with replacement."""
    fn = (lambda yy, pp: balanced_accuracy_score(yy, pp)) if metric == "balanced_accuracy" else \
        (lambda yy, pp: f1_score(yy, pp, average="macro", labels=range(N_CLASSES), zero_division=0))
    y, pred, groups = np.asarray(y), np.asarray(pred), np.asarray(groups)
    vals = np.array([fn(y[i], pred[i]) for i in _boot_indices(groups, n_boot, seed)])
    return {"value": float(fn(y, pred)), "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}


def paired_bootstrap_delta(y, pred_a, pred_b, groups, n_boot=N_BOOT, seed=SEED):
    """bal-acc(a) - bal-acc(b) with a paired prompt-level bootstrap CI and P(delta <= 0)."""
    y, pa, pb, groups = map(np.asarray, (y, pred_a, pred_b, groups))
    d = np.array([balanced_accuracy_score(y[i], pa[i]) - balanced_accuracy_score(y[i], pb[i])
                  for i in _boot_indices(groups, n_boot, seed)])
    return {"delta": float(balanced_accuracy_score(y, pa) - balanced_accuracy_score(y, pb)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_le_0": float((d <= 0).mean())}


def log_proba(p):
    return np.log(np.clip(np.asarray(p, dtype=np.float64), PROBA_FLOOR, 1.0))


def logprob_blend(log_probs, weights=None):
    """Weighted log-linear pool: softmax(sum_k w_k log p_k). weights=None -> all 1 (equal blend, nothing fitted)."""
    weights = np.ones(len(log_probs)) if weights is None else np.asarray(weights, dtype=np.float64)
    z = sum(w * lp for w, lp in zip(weights, log_probs))
    return np.exp(log_softmax(z, axis=1))


def fit_blend_weights(log_probs, y):
    """Non-negative log-linear pool weights minimising NLL on (log_probs, y). Only call on rows the
    component models did not train on, and never on the rows the blend is reported on."""
    from scipy.optimize import minimize
    y = np.asarray(y)

    def nll(w):
        z = sum(wk * lp for wk, lp in zip(w, log_probs))
        return -log_softmax(z, axis=1)[np.arange(len(y)), y].mean()

    k = len(log_probs)
    res = minimize(nll, x0=np.full(k, 1.0 / k), bounds=[(0, None)] * k, method="L-BFGS-B")
    return res.x


def lopo_blend(log_probs, y, groups):
    """Leave-one-prompt-out cross-fitted blend on rows the components never trained on (the dev set):
    each prompt is scored with weights fit on the other prompts. Returns (proba, per-fold weights)."""
    groups = np.asarray(groups)
    out = np.zeros_like(log_probs[0])
    ws = []
    for p in np.unique(groups):
        s, f = groups == p, groups != p
        w = fit_blend_weights([lp[f] for lp in log_probs], np.asarray(y)[f])
        out[s] = logprob_blend([lp[s] for lp in log_probs], w)
        ws.append(w)
    return out, np.array(ws)


def write_predictions(path, meta_df, proba, fold=None):
    """Blendable prediction file: id, prompt_id, labels, [fold], p_<class> in CLASS_NAMES order."""
    proba = np.asarray(proba, dtype=np.float64)
    assert proba.shape == (len(meta_df), N_CLASSES)
    assert np.allclose(proba.sum(1), 1.0, atol=1e-4) and (proba >= 0).all()
    out = meta_df[META_COLS].reset_index(drop=True).copy()
    if fold is not None:
        out["fold"] = np.asarray(fold)
    out = pd.concat([out, pd.DataFrame(proba, columns=PROBA_COLS)], axis=1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format="%.8g")
    return out


def read_predictions(path, meta_df):
    """Read a prediction file and align it to meta_df rows by id; asserts labels and prompt_ids agree."""
    p = pd.read_csv(path)
    assert list(p.columns[-N_CLASSES:]) == PROBA_COLS, "probability columns must be p_<class> in label_map order"
    p = p.set_index("id").loc[meta_df["id"]].reset_index()
    assert (p["prompt_id"].to_numpy() == meta_df["prompt_id"].to_numpy()).all()
    assert (p["labels"].to_numpy() == meta_df["labels"].to_numpy()).all()
    proba = p[PROBA_COLS].to_numpy(np.float64)
    return proba / proba.sum(1, keepdims=True)


def load_hidden(tag, train_df, dev_df, hidden_dir=None):
    """Pooled Qwen hidden states (exp02 kernel output) aligned to train_df / dev_df. tag in {'step0', 'exp01'}."""
    hidden_dir = Path(hidden_dir or HIDDEN_DIR)
    fn = f"pooled_hidden_{tag}.npy"
    expected = dict(l.split()[::-1] for l in (EXP02_ART / "pooled_hidden_SHA256SUMS").read_text().splitlines())
    assert sha256_file(hidden_dir / fn) == expected[fn], f"{fn} differs from the exp02 dataset version"
    order = pd.read_parquet(EXP02_ART / "exp01_logits.parquet", columns=["id"])["id"].to_numpy()
    pos = {k: i for i, k in enumerate(order)}
    H = np.load(hidden_dir / fn).astype(np.float64)
    return H[[pos[k] for k in train_df["id"]]], H[[pos[k] for k in dev_df["id"]]]


def library_versions():
    import platform
    import scipy
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=float))
