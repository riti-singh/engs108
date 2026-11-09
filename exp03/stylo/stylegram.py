"""4-view "stylegram" n-gram LinearSVC features, after lucasboesen/lora-x-adaptive-centering (v5,
section "1) style-aware n-grams + LinearSVC"). Views, vectorizer options and block weights are copied from
that notebook's SPEC; only the official train rows are used here (no external data, no gates, no centering).
"""
import re

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
# (view, ngram lo, hi, lowercase, max_features, block weight)
SPEC = [('char', 3, 5, False, 250000, 1.0), ('word', 1, 3, False, 180000, 0.7),
        ('shape', 2, 6, False, 60000, 0.5), ('function', 1, 3, False, 50000, 0.5)]
TOKEN_PATTERN = r"(?u)\b\w+\b|[^\w\s]"


def shape_text(text):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", text)


def function_text(text):
    return " ".join(t.lower() if t.lower() in FUNCTION_WORDS or not t.isalnum() else "CONTENT"
                    for t in re.findall(r"\w+|[^\w\s]", text))


def _view_text(view, texts):
    tf = shape_text if view == 'shape' else function_text if view == 'function' else str
    return [tf(t) for t in texts]


class StylegramVectorizer(BaseEstimator, TransformerMixin):
    """hstack of the 4 weighted sublinear TF-IDF views; input: iterable of texts."""

    def fit(self, texts, y=None):
        texts = list(texts)
        self.vecs_ = []
        for view, lo, hi, lower, maxf, w in SPEC:
            actual = 'char' if view == 'shape' else 'word' if view == 'function' else view
            opts = dict(analyzer=actual, ngram_range=(lo, hi), lowercase=lower, sublinear_tf=True, min_df=2,
                        max_features=maxf, dtype=np.float64)
            if actual == 'word':
                opts['token_pattern'] = TOKEN_PATTERN
            self.vecs_.append(TfidfVectorizer(**opts).fit(_view_text(view, texts)))
        return self

    def transform(self, texts):
        texts = list(texts)
        return sp.hstack([v.transform(_view_text(s[0], texts)) * s[5] for v, s in zip(self.vecs_, SPEC)],
                         format='csr')
