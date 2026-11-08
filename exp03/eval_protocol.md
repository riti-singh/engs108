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
is the GPU agent's call (§7).

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
