"""Equal-weight log-probability blend of exp01 QLoRA and the exp02 stylometric model.

    z_c   = log_softmax(qlora_logits)_c + log(clip(stylo_proba_c, 1e-12, 1))
    p_c   = softmax(z)_c            (normalised product of the two class distributions)
    label = argmax_c z_c            (ties -> lowest class id, i.e. first in label_map order)

Weights are fixed at (1, 1) and nothing is fitted. Column c of both inputs must be class id c from the
exp01 adapter's `label_map.json` (= sorted model names); sklearn's `classes_` must equal range(18).
"""
import numpy as np
from scipy.special import log_softmax

STYLO_PROBA_FLOOR = 1e-12


def equal_logprob_blend(qlora_logits, stylo_proba):
    """Return blended class log-probabilities, shape (n, n_classes); argmax gives the label."""
    lq = log_softmax(np.asarray(qlora_logits, dtype=np.float64), axis=1)
    ls = np.log(np.clip(np.asarray(stylo_proba, dtype=np.float64), STYLO_PROBA_FLOOR, 1.0))
    assert lq.shape == ls.shape, (lq.shape, ls.shape)
    return log_softmax(lq + ls, axis=1)
