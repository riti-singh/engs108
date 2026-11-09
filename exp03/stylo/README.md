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

## Follow-up: 4-view stylegram LinearSVC (`stylegram.py`, `run_svc.py`; protocol §8)

`ENGS108_DATA_DIR=<dir> python exp03/stylo/run_svc.py` takes about 2 minutes on 8 CPUs. It writes `artifacts/svc_results.json`, `stylo_svc_{oof_train,dev}.csv`, `stylo_svc_meta.json`, `stylo_svc_exp02ref_oof_train.csv` and `blend_S2_dev.csv`.

| model | train OOF bal-acc | OOF log loss | dev bal-acc [95% prompt CI] |
|---|---|---|---|
| stylegram SVC C=0.1 (**selected**) | 0.3570 | 1.856 | **0.3527** [0.300, 0.406] |
| stylegram SVC C=0.25 / 0.5 / 1.0 | 0.3566 / 0.3531 / 0.3557 | 1.88 / 1.91 / 1.93 | not scored |
| R1 stylo_exp03 LR C=300 | 0.3552 | 2.528 | 0.3406 |
| exp02 calibrated LinearSVC C=0.5 (refit on frozen folds) | 0.3303 | 2.130 | not scored |
| S2 = equal blend probe_exp01 + stylegram SVC | – | – | **0.4155** [0.357, 0.469] |

The table below gives paired prompt-bootstrap deltas (2,000 resamples) of the selected SVC or its blend against each reference:

| delta | value | 95% CI |
|---|---|---|
| OOF: SVC − R1 | +0.0018 | [−0.016, +0.018] |
| OOF: SVC − exp02 SVC | +0.0267 | [+0.008, +0.044] |
| dev: S1 − R1 | +0.0121 | [−0.041, +0.065] |
| dev: S1 − B1 | −0.0459 | [−0.106, +0.017] |
| dev: S2 − R1 | +0.0749 | [+0.031, +0.118] |
| dev: S2 − B1 (0.3986) | +0.0169 | [−0.022, +0.056] |

Reading:
- **Against R1 the stylegram SVC is a tie.** It is clearly better than exp02's calibrated SVC.
- **Its main gain is calibration.** Its OOF log loss is 1.86 vs R1's 2.53, which likely helps the log-prob blend. S2 beats B1 by +0.017, but the CI includes 0.
- **The selected C is the lowest grid value**, and OOF is flat across the grid (range 0.004). I did not extend the grid after seeing results.
- **Fit diagnostics.** None of the fits raised ConvergenceWarning. The temperature fit on OOF decisions is a≈13.5.
- **Model size.** No joblib is saved for the SVC. With 207k features, the four vectorizer vocabularies would be large. The predictions are the blendable artifact, and `run_svc.py` refits it deterministically.
