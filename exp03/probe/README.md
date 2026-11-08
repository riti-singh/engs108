# exp03/probe — packaged linear probe on pooled Qwen3-8B hidden states

CPU-only. No Qwen inference, no QLoRA training: the probes read the pooled hidden states that exp02's
Kaggle kernel already produced (`ritisingh123/engs108-exp02-pooled-hidden-states`, sha256 in
`exp02/artifacts/pooled_hidden_SHA256SUMS`). Only the original competition `train.csv` is used.

| file | what |
|---|---|
| `probe.py` | `fit_probe(H, y, C)` (StandardScaler + multinomial LogisticRegression), `export_npz`, `ProbeModel` / `load_probe(path)` -> `.predict_proba(H)` (numpy + scipy only) |
| `fit_probe.py` | fits `probe_exp01` (C=0.001, the exp02 0.377 probe) and `probe_step0` (C chosen by the protocol), writes artifacts |
| `run_blends.py` | pre-registered probe x stylometry reports R0–B6 (`exp03/eval_protocol.md` §6) with prompt CIs and paired deltas |
| `artifacts/probe_{exp01,step0}.npz` | packaged models: `mean`, `scale`, `coef` (18 x 4096), `intercept`, `class_names`, `meta` (JSON) |
| `artifacts/probe_*_{oof_train,dev}.csv` | per-row probabilities in the protocol format (§7) |
| `artifacts/probe_*_meta.json`, `probe_results.json` | C sweep, metrics, CIs, hashes, versions |
| `artifacts/blend_*` | blend reports and per-row dev probabilities |
| `tests/test_probe.py` | offline tests (synthetic + committed artifacts) |

## Reproduce

```bash
exp02/fetch_pooled_hidden.sh ~/data/hidden            # needs KAGGLE_API_TOKEN
ENGS108_DATA_DIR=<dir with train.csv> EXP03_HIDDEN_DIR=~/data/hidden python exp03/probe/fit_probe.py
ENGS108_DATA_DIR=<dir with train.csv> python exp03/probe/run_blends.py   # after exp03/stylo/run_stylo.py
```

## Use at inference

```python
from probe import load_probe
p = load_probe("exp03/probe/artifacts/probe_exp01.npz").predict_proba(H)   # H: (n, 4096) -> (n, 18)
```
Column `c` is class id `c` of the exp01 `label_map.json` (sorted model names, `ProbeModel.class_names`).

## Producing `H` for test rows (not built here)

`H` must be produced exactly as in `exp02/qwen-val-inference/rs2027-engs108-exp02-val-inference.ipynb`
(`pooled_forward`), otherwise the probe sees a shifted input distribution:

- Model: Qwen3-8B-Base (`qwen-lm/qwen-3/transformers/8b-base/1`), 4-bit NF4, double quant, fp16 compute,
  `AutoModelForSequenceClassification(num_labels=18)`; for `probe_exp01` also the exp01 adapter
  (`ritisingh123/engs108-qwen3-8b-lora-adapter`, `is_trainable=False`, `eval()`). `probe_step0` needs the base model only.
- Tokenizer: same model path, `truncation=True, max_length=512`, `padding=True`, `pad_token_id` set on the config.
- Pooling: forward hook on the `score` module input; take the hidden state at the last non-pad token of each row
  (same rule as `*ForSequenceClassification`), cast to float32 (stored as float16; the probe upcasts).
- Cost: one forward pass per test row on a Kaggle GPU (the exp02 kernel did 2,699 rows per pass). This is
  the same pass the exp01 inference notebook already runs to get logits, so the hidden states can be captured
  from it with the hook at no extra forward cost.
- fp16 4-bit outputs move slightly with batch padding composition (exp02 §4b: up to 3 argmax flips on 414 rows
  for logits), so expect small test-time differences; batch size 8 matches the artifacts.

## Caveat

The exp01 adapter was trained on all 2,285 train rows, so `probe_exp01` OOF scores are optimistic
(`oof_honest: false` in its meta) and must not be used to select or weight it; see protocol §5.
`probe_step0` OOF is honest.
