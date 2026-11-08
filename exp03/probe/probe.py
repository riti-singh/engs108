"""Linear probe on pooled Qwen3-8B hidden states: a reproducible inference component.

The probe is StandardScaler + multinomial LogisticRegression (lbfgs), fit with sklearn. It is
exported to a plain .npz (mean, scale, coef, intercept, class names), so prediction needs only
numpy:
    proba = softmax(((H - mean) / scale) @ coef.T + intercept)
`H` is (n, 4096): the last-non-pad-token final hidden state that feeds the `score` head, i.e. the input to
`score` in exp01's PeftModel (see exp03/probe/README.md for how to produce it at test time).
Column c of the output = class id c from label_map.json (CLASS_NAMES order).
"""
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MAX_ITER = 3000


def make_probe(C):
    return Pipeline([("z", StandardScaler()), ("lr", LogisticRegression(C=C, max_iter=MAX_ITER))])


def fit_probe(H, y, C):
    pipe = make_probe(C).fit(np.asarray(H, dtype=np.float64), y)
    assert list(pipe.classes_) == list(range(pipe.named_steps["lr"].coef_.shape[0]))
    return pipe


def export_npz(pipe, path, class_names, meta=None):
    z, lr = pipe.named_steps["z"], pipe.named_steps["lr"]
    np.savez(path, mean=z.mean_, scale=z.scale_, coef=lr.coef_, intercept=lr.intercept_,
             class_names=np.array(class_names), meta=np.array(json.dumps(meta or {})))


class ProbeModel:
    """Numpy-only probe loaded from the exported .npz."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as d:
            self.mean, self.scale = d["mean"], d["scale"]
            self.coef, self.intercept = d["coef"], d["intercept"]
            self.class_names = [str(c) for c in d["class_names"]]
            self.meta = json.loads(str(d["meta"]))

    def decision_function(self, H):
        H = np.asarray(H, dtype=np.float64)
        assert H.ndim == 2 and H.shape[1] == self.mean.shape[0], H.shape
        return ((H - self.mean) / self.scale) @ self.coef.T + self.intercept

    def predict_proba(self, H):
        return softmax(self.decision_function(H), axis=1)


def load_probe(path=None):
    return ProbeModel(path or Path(__file__).resolve().parent / "artifacts" / "probe_exp01.npz")
