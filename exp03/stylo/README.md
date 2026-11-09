# exp03/stylo — frozen protocol code + extended stylometric features

CPU-only, original competition `train.csv` only. Protocol: `exp03/eval_protocol.md`.

| file | what |
|---|---|
| `protocol.py` | split (`GroupShuffleSplit(0.15, seed 42)` on `prompt_id`) + assertions, frozen `GroupKFold(5)` folds, metrics, prompt bootstrap CIs, paired deltas, log-linear blends (equal / train-OOF / dev-LOPO), prediction-artifact I/O, hidden-state loader |
| `features.py` | feature groups: exp02 `char` (char_wb 3–5 TF-IDF), `word` (1–2 TF-IDF), `style` (27 exp02 stats, verbatim); new `fw` (223 function words, sublinear TF-IDF), `fwng` (function-word/content-masked 1–3 grams), `typo` (32 typography stats), `sent` (46 sentence/paragraph stats), `shape` (token-shape 1–3 grams) |
| `run_stylo.py` | 36-config ablation on train-CV, selection, full-train refit, single dev scoring |
| `artifacts/train_folds.csv` | frozen folds (sha256 in protocol §3) |
| `artifacts/ablation.csv` | every config: pooled OOF bal-acc + prompt CI, macro F1, log loss, paired delta vs exp02 base |
| `artifacts/stylo_{exp03,exp02ref}_{oof_train,dev}.csv`, `*_meta.json`, `stylo_results.json` | predictions in protocol format + metadata |
| `artifacts/stylo_exp03_model.joblib` | fitted sklearn Pipeline (`ColumnTransformer` features + LR), `predict_proba(DataFrame({"text": ...}))`; needs `features.py` importable |
| `tests/test_protocol.py` | offline tests |

```bash
ENGS108_DATA_DIR=<dir with train.csv> python exp03/stylo/run_stylo.py   # ~15 min on 8 CPUs
python -m pytest exp03                                                  # no network
```

## Result (train-CV selects, dev reported once)

- Best pooled OOF: `char+word+style+fwng, C=100` 0.3552; tie rule (within 0.003 -> fewer groups, smaller C)
  selected `char+word+style, C=300` (OOF 0.3552, CI [0.337, 0.376]). Dev: **0.341** [0.287, 0.389], vs exp02
  refit 0.336 (reproduces exp02 exactly, max |Δp| 5e-7).
- None of the new groups improves OOF beyond noise: `+fwng` +0.007 (CI [-0.005, 0.020]); `+typo`, `+sent`, `+shape`,
  `+fw` and all-groups are −0.000 to −0.011. Alone, the new groups score 0.18–0.32 OOF (all new groups together 0.319).
- The first ablation run standardised per-word function-word rates; rare words got huge z-scores on held-out rows
  (OOF log loss ~12). `fw` is now sublinear TF-IDF over the fixed vocabulary, and `typo`/`sent` z-scores are clipped at ±5.
  The table above is from the rerun; dev was not scored in the first run.
