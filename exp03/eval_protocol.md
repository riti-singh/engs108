# exp03 evaluation protocol (frozen)

This file fixes how every exp03 model is trained, selected, blended and reported. It was committed
before any exp03 model was fitted. Code: `exp03/stylo/protocol.py`, which every exp03 script imports.
Later agents (including the GPU/QLoRA agent) must follow it so their outputs can be blended and compared.

## 1. Data and split

- **Data:** competition `name-that-llm`, `train.csv` only, sha256
  `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707` (2,699 rows, 150 prompts,
  18 classes). `load_data()` asserts this hash. No external or additional-generation data is used
  for any fit or evaluation (the manager confirmed this on 2026-11-08).
- **Classes:** class id `c` is the `c`-th name in sorted order. This is the exp01 adapter's
  `label_map.json`; the order is hard-coded as `protocol.CLASS_NAMES` and asserted against the data.
- **Split:** the exp01 split, unchanged:
  `GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42).split(df, groups=df.prompt_id)`.
  Here `df = train[["id","prompt_id","text","model"]]` keeps the file order.
  - **train:** 2,285 rows, 127 prompts.
  - **dev:** 414 rows, 23 prompts, 23 rows per class. The prompt list equals the exp01 log, `protocol.DEV_PROMPTS`.
- **Assertions:** `assert_split()` checks the row and prompt counts, zero prompt overlap, the
  dev prompt list, and 23 dev rows per class. Every script runs these assertions.

## 2. Roles of the two sets

| Set | Used for | Never used for |
|---|---|---|
| train (2,285) | Fitting, and **all model selection** (feature groups, hyper-parameters, blend weights for components with honest OOF) via the frozen 5-fold prompt-grouped CV | — |
| dev (414) | Reporting a short **pre-registered** list of final configurations (§6). Also cross-fitted blend weights for components whose train OOF is not honest (§5b) | Choosing features, hyper-parameters or which configuration is "best" |

The dev set is a *development comparison set*, not an untouched test set. exp01 and exp02 already
reported on it. Every dev evaluation run in exp03 is listed in the results JSON
(`dev_evaluations`), so the number of looks is auditable. Final scoring is the user's Kaggle submission.

## 3. Cross-validation folds (model selection)

- **Construction:** `GroupKFold(n_splits=5)` on `prompt_id` over the 2,285 train rows in split
  order (sklearn 1.6.1, deterministic, no shuffle; the same folds as exp02's model selection).
- **Fold sizes:** 468/26, 467/26, 450/25, 450/25, 450/25 rows/prompts.
  - Every class has 25–26 rows in every fold, because each prompt has one text per model.
- **Frozen copy:** `exp03/stylo/artifacts/train_folds.csv` (`id, prompt_id, fold`), sha256
  `ea6a145239efac7fab99948f443d16ba79c087ecbc7f13c46f196753d073e4fa`. `load_data()` asserts the
  recomputed folds equal this file, and that no prompt crosses folds.
- **Out-of-fold (OOF) predictions:** for fold k, fit every learned step (vectorizers, scalers,
  classifier) on folds ≠ k and predict fold k. Concatenating the 5 parts gives one honest
  prediction per train row.
- **Selection:** the CV score is the bal-acc of the concatenated OOF predictions (pooled over
  folds; exp02 averaged per-fold scores, which differs slightly). It drives all choices. Ties within
  0.003 go to the simpler or more regularised option.
- **Dev predictions** come from a single refit on all 2,285 train rows.
- **Probe C:** `probe_step0` picks C from {0.0003, 0.001, 0.003, 0.01, 0.03, 0.1} by pooled OOF bal-acc,
  with ties within 0.003 going to the smallest C. exp02 chose C=0.01 from a coarser grid by mean-fold
  score, so the step-0 probe here differs from exp02's (dev 0.321 vs 0.329). `probe_exp01` keeps exp02's
  C=0.001.

## 4. Metrics and confidence intervals

- **Primary metric:** balanced accuracy (= macro recall; the competition metric).
- **Secondary:** macro F1 (18 classes, `zero_division=0`) and multi-class log loss (probabilities
  clipped at 1e-12) for probabilistic outputs.
- **Prompt-level CIs:** a cluster bootstrap over prompts. Resample the dev prompts (23) or
  train prompts (127, for OOF) with replacement, 2,000 times, using `numpy.random.default_rng(42)`.
  Take all rows of each drawn prompt, recompute the metric, and report the 2.5/97.5 percentiles
  (`prompt_bootstrap_ci`).
- **Model comparisons:** paired prompt bootstrap of the bal-acc difference, giving the delta, a 95% CI and
  P(delta ≤ 0) (`paired_bootstrap_delta`).
- **Noise floor:** with 23 dev prompts, a dev CI is roughly ±0.05. Dev differences below about 0.005
  (2 rows) are noise even before sampling error (exp02 batch-padding result).
- **Optimizer noise:** lbfgs logistic regression at large C (≥100) on the ~10^5-column TF-IDF designs
  is sensitive to float summation order. Two fits on the identical matrix (equal to 1e-15) agreed on
  95–100% of dev argmaxes (measured: 96.4% at C=100 and 100% at C=300 in one check; 95.2% at C=300
  and 99.0% at C=100 in `run_stylo.py`'s refit check, dev bal-acc 0.341 vs 0.345). Train-CV differences below the
  0.003 tie margin, and dev differences of about ±0.01 between stylometric configurations, are within
  this noise. `run_stylo.py` records the refit-to-refit difference as `refit_numerical_noise`.

## 5. Ensembles: leakage-safe weight fitting

The pool is log-linear: `p = softmax(Σ_k w_k · log clip(p_k, 1e-12, 1))`, with `w_k ≥ 0`
(`logprob_blend`). With all `w_k = 1` this is the exp02 equal blend; nothing is fitted.
There are two ways to fit weights, depending on whether a component's train OOF is honest.

- **(a) Train-OOF fitting.** Use this when every component has honest OOF on train, i.e. the
  component never saw a train row's label except through the frozen CV fold fit.
  - Fit the weights by NLL on the 2,285 OOF rows, then apply them unchanged to the dev
    predictions of the full-train refits.
  - The dev rows are touched once, at reporting time.
  - Applies to: stylometric models, and probes on the **step-0 (base-model)** hidden states.
- **(b) Dev cross-fitting.** Use this when a component's train OOF is not honest. These rows are
  scored with weights fit on the other 22 dev prompts, so each prompt's prediction is out-of-fold
  (`lopo_blend`, leave-one-prompt-out over the 23 dev prompts, as in exp02).
  - **The exp01 adapter was trained on all 2,285 train rows.** So the exp01 QLoRA logits, and
    any probe on `pooled_hidden_exp01.npy`, are optimistic on train OOF: the encoder has seen the
    held-out fold's labels. The exp01 probe's CV bal-acc is 0.417 against 0.377 on dev.
  - Weights fit on such OOF would over-weight these components.
  - For blends that include exp01-derived components, use either fixed equal weights or (b).
  - (a) with exp01 components is reported only as a labelled sensitivity check.

**Caveat for train-CV selection with exp01 representations.** Hyper-parameters of probes on exp01
hidden states can still be chosen by train-CV, but the CV *level* is inflated. The step-0 states
are the leakage-free comparison. The exp01 probe C (0.001) was chosen this way in exp02 and is
kept unchanged here, so the packaged probe reproduces 0.377.

Removing this limitation needs fold-wise adapters: one QLoRA run per fold, 5× training cost. That
is the GPU agent's call.

## 6. Pre-registered dev reports (exp03 stylometry + probe)

Selection happens only on train-CV. Dev is then reported for exactly these rows:

| id | Model | Weights |
|---|---|---|
| R0 | exp02 stylometric `char+word+style, LR C=100` (reference, refit) | — |
| R1 | exp03 stylometric (best train-CV configuration from the ablation) | — |
| R2 | exp01 linear probe (packaged; reference 0.377) | — |
| R3 | step-0 linear probe | — |
| B1 | **probe_exp01 + stylo_exp03** (primary) | equal |
| B2 | probe_exp01 + stylo_exp03 + exp01 QLoRA | equal |
| B3 | probe_exp01 + stylo_exp03 | dev LOPO (5b) |
| B4 | probe_exp01 + stylo_exp03 + exp01 QLoRA | dev LOPO (5b) |
| B5 | probe_step0 + stylo_exp03 (fully train-only pipeline) | train OOF (5a) |
| B6 | probe_exp01 + stylo_exp03 (sensitivity, biased weights) | train OOF (5a) |

- **References:** the exp02 equal blend (QLoRA + stylo_exp02, 0.372) and R2 (0.377) are the
  comparisons for paired bootstrap deltas.
- **Primary model:** B1 is the primary exp03 result for this workstream, chosen before seeing dev.
  It has no fitted blend parameters.

## 7. Artifact format for blendable predictions

Every model that may be blended must write CSV files with this layout, via `protocol.write_predictions`.

| File | Rows | Columns |
|---|---|---|
| `<model>_oof_train.csv` | 2,285 train rows, in `train_folds.csv` order | `id, prompt_id, labels, fold, p_<class_0> … p_<class_17>` |
| `<model>_dev.csv` | 414 dev rows, in split order | `id, prompt_id, labels, p_<class_0> … p_<class_17>` |
| `<model>_test.csv` (inference only) | test rows | `id, p_<class_0> … p_<class_17>` |

**Column and value rules.**
- `p_<class>` columns are in `CLASS_NAMES` (= `label_map.json`) order, e.g. `p_claude-fable-5` first and `p_qwen3.8-max` last.
- Values are probabilities that sum to 1 (±1e-4). Logit-only models should write `softmax(logits)`.
- `labels` is the integer class id.
- `read_predictions()` aligns files by `id` and asserts that `prompt_id` and `labels` agree.

**Requirements on the producing model.**
- **OOF:** the train OOF file must come from 5 models, each fit on the frozen folds ≠ k. Any
  learned upstream representation must also be fit without fold k; otherwise mark the OOF
  `"oof_honest": false` in the metadata, and it may only be blended via §5b.
- **Dev:** the dev file must come from a model that never saw dev rows.

**Metadata.** Each pair of files has a `<model>_meta.json` alongside, recording:
- config and seed(s);
- library versions;
- input hashes (train.csv, hidden states, adapters with their Kaggle dataset version);
- `oof_honest`;
- train-OOF and dev metrics.

A model that can only produce dev predictions (e.g. a single QLoRA run) can still be blended with
equal weights or via dev LOPO (§5b), but not via train-OOF weights.

## 8. Follow-up: stylegram LinearSVC (pre-registered in `stylo/artifacts/svc_preregistration.json`)

- **Model.** `stylo/stylegram.py` implements four views, with view definitions copied from `lucasboesen/lora-x-adaptive-centering` v5 (`SPEC`, `shape_text`, `function_text`, `FUNCTION_WORDS`). The views are char 3–5, word 1–3, token shape 2–6 (char), and function-word/`CONTENT` 1–3 (word). Each is sublinear TF-IDF with `min_df=2`, the notebook's `max_features`, and block weights 1.0/0.7/0.5/0.5. They feed `LinearSVC(random_state=42, max_iter=4000)`. Only official train rows are used: no external data, gates, or centering.
- **Grid and selection.** C ∈ {0.1, 0.25, 0.5, 1.0} was committed before any fit (commit "pre-register stylegram …"). The model is selected by pooled GroupKFold(5) OOF balanced accuracy using the §3 tie rule (within 0.003 of the best, pick the smallest C).
- **OOF comparators.** R1 is `stylo_exp03` LR. Exp02's `CalibratedClassifierCV(LinearSVC(C=0.5, max_iter=20000), sigmoid, cv=3)` on char+word+style is refit on the frozen folds. Its inner cv=3 is internal to each training fold.
- **Calibration.** p = softmax(a · decision). The scalar a is fit by NLL on train OOF decisions only: OOF rows in fold k use an a fit on the other four folds, and dev uses an a fit on all train OOF. Argmax, and therefore balanced accuracy and selection, does not depend on a.
- **Dev reports.** Only S1 (the selected SVC) and S2 (an equal log-prob blend of probe_exp01 + S1) are reported. Each gets a prompt CI and paired deltas against R1 and B1. Artifacts follow §7: `stylo_svc_{oof_train,dev}.csv` and `stylo_svc_meta.json`.
