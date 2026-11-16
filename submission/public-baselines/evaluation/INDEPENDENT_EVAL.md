# Independent evaluation of the frozen V1 pipeline: audit, protocol, and result

**Result: 1,179 / 2,178 correct (54.1%, 95% Wilson interval 52.0-56.2%), macro F1 0.518**, from one run of the
unchanged V1 notebook on 2,178 externally labeled passages that are not in `train.csv` (121 per label, 121 complete
18-model prompt panels; section 4). This is accuracy on a **shifted distribution** (different prompts, uploader-claimed
labels, mostly short passages), not an estimate of the hidden-test score. The official public leaderboard score
(0.680, see `EVALUATION.md`) remains the only measurement on the competition's own test distribution.

The 533/540 matrix in `EVALUATION.md` stays a training-derived, in-sample diagnostic. No split of the competition
`train.csv` is used or reported as held out anywhere here.

## 1. Where the V1 inference path uses training labels, reference statistics, calibration, or test-batch information

Source: the runtime modules the V1 notebook writes (`build/V1/validate/engs108-v1-a0-nativeshare040.ipynb`;
`ensemble.py`, `core_math.py`, `core_readout.py`, `core_worker.py`, `comp_cached.py`, `native_fusion.py`, `gauss.py`),
read for this audit. V1 differs from Ref A only in `CONFIG['native_embedding']['share']` = 0.4.

| Step (file) | Training labels / train.csv | Reference statistics | Calibration | Test-batch information |
|---|---|---|---|---|
| Topic groups (`core_math.topic_groups`, called in `ensemble.py`) | no | no | no | **yes**: TF-IDF fit on the test batch (`min_df=2, max_df=0.05`), agglomerative groups capped at 18 rows. Every group-aware step below depends on it |
| Core decoder/LoRA members (`core_worker.py`, `core_readout.py`) | prefit by upstream authors on competition train + public pools | twin heads subtract the **per-test-group mean** of features (groups of size >= 2) | fixed prefit scales | **yes** (group centering) |
| Core Gaussian shared-prompt head (`core_readout.py`) | **yes, at inference**: when a member's `gaussian_branch == 'source-test-conditioned'`, `gauss.fit` is refit on the 2,699 `train.csv` features, labels and prompt groups (asserts 2,699 rows) | class means, prompt effects and Ledoit-Wolf-shrunk covariances estimated from that refit; temperatures fixed at 3.0 | fixed | **yes**: scored only for rows in test groups of size 2..18, with injective (one-label-per-row) group marginals |
| Complementary Gaussian heads (`comp_cached.py`) | prefit (frozen `.npz` per encoder layer) | frozen reference statistics | fixed T1/Tg per head | **yes** (same group-size gate and group marginals) |
| Core + complementary mix (`ensemble.py`) | no | no | fixed 0.85/0.15 | no |
| ATLAS re-ranker (`ensemble.py:101`) | prefit adapter | no | no | **yes**: applied only to rows with top-2 margin < 0.3 when the batch has >= 50 rows, to all rows otherwise |
| Native Qwen3-Embedding-8B expert (`native_fusion.calibrate_native`) | prefit 5-fold heads (`outer_labels_used` asserted False) | **per-test-group mean** of centered logits subtracted (groups of size 2..18) | fold temperatures `plain_scale` / `centered_scale` and `selected_centered_weight` <= 0.5, fixed from the asset's selections | **yes** (group centering) |
| Native mixture | no | no | fixed share s = 0.4 (the V1 change) | no |
| Sinkhorn balancing (`core_math.balance_logits`, strength 0.30) | **yes**: label prior = `bincount(train labels)` | no | yes | **yes**: target marginal = 0.65 x the batch's mean predicted distribution + 0.35 x train label prior; only runs when the batch has >= 18 rows |
| Typography guard (`core_math.typography_evidence` / `finalize`) | **yes, at inference**: per-label mark rates estimated from `train.csv` texts and labels | yes (smoothed rates, feature weights) | clipped to +-0.30, gated by margin, length and mark coverage | per-row only |
| Fold averaging + argmax | no | no | no | no |

Consequences for any evaluation of the frozen pipeline:

* **Labels are used, but only `train.csv` labels.** Gaussian refit, Sinkhorn prior and typography rates read
  competition train labels at inference. That is legitimate for text outside `train.csv` and is exactly why any
  `train.csv` row, however split, is contaminated.
* **Predictions are not per-row.** Topic grouping, group centering, Gaussian group marginals, Sinkhorn and the ATLAS
  gate all depend on the other rows in the batch. A score is therefore a property of (model, batch). The
  evaluation batch must look like the hidden test in structure: prompt panels (one prompt answered by several of
  the 18 models), a roughly balanced label mix, and a size well above the 50-row ATLAS threshold and the 18-row
  Sinkhorn threshold.
* **Upstream prefit members** were trained by their authors on the competition train set plus public pools
  (V1 mounts, among others, `prvsiyan/prvsiyans-extended-name-that-llm-dataset`, recorded at v12, and prefit heads fitted on
  `crtproduction/name-that-llm-open-pool`). Any passage published before the last of those asset versions may
  have been training data, whatever its labels.

## 2. Protocol required for a valid independent evaluation

1. Labeled passages from all 18 labels that are not in `train.csv` and were not public before the last version of
   every dataset V1 mounts (latest: `crtproduction/name-that-llm-open-pool` v16, 2026-11-09 04:41 UTC; V1 reads
   `lightningv08/ntl-olmo2-native24-final-inference-assets` v22, 2026-11-07), and are not exact copies of
   anything in those pools.
2. Prompts not drawn from or reconstructed from competition train prompts (otherwise topic leaks into the
   prompt-group heads).
3. Feed the pipeline `id,text` only, ids re-hashed; keep labels in a separate file that the run never reads.
4. Score the whole set as one batch with balanced, prompt-paneled composition (because of the batch-dependent
   steps above); report the batch composition with the result.
5. Run the unchanged V1 notebook (same code, same mounted asset versions); the only edit is the input cell
   pointing at the new `test.csv`, as in the existing validate mode (`build_notebooks.py --mode independent`).
6. Decide nothing (no thresholds, weights, or subsets) after seeing the labels.

## 3. Does a suitable dataset already exist?

Surveyed: every Kaggle dataset returned for "name-that-llm", "name that llm", "ntl" searches (186 refs, 2026-11-15).

| Dataset | Labels | Published | Verdict |
|---|---|---|---|
| `nischaydnk/ntl-topic-synthetic-18llms` | **18/18** (137-294 rows each) | 2026-11-11 00:31 UTC | **Best candidate; passes all checks we can run** (below) |
| `prvsiyan/prvsiyans-extended-name-that-llm-dataset` | 18/18 | from 2026-10 | **Not independent**: mounted by V1 (v12) and merged into the public pools prefit heads were fitted on |
| `prvsiyan/the-paragraph-without-a-signature` | 17/18 (gemini-3.7-flash unverified, side file) | 2026-11-11 | Answers reconstructions of competition train prompts and reuses train `prompt_id`s: topic overlap with train |
| `lourenswalters/ntl-community-vetted-05` | 17/18 (no gemini-3.7-flash) | 2026-11-12 | Merge of 28 community datasets, many pre-date V1's assets; labels are uploaders' claims |
| `kilupy/ntl-18-model-corpus` | 18/18 | 2026-11-02 | Published before V1's asset versions; may be in the pools |
| `dhananjayk2609/ntl-panel-gap-passages-kaggle-benchmarks`, `aitahmednasser/ntl-kaggle-ai-gen` | 12/18 | 2026-11-11..14 | Missing 6 labels; prompts taken from the Open Pool |
| `tamerlanomralinov/ntl-matched-bench-gen-r1` | 2/18 | 2026-11-12 | Missing 16 labels |
| `zakirkhanaleemi/name-that-llm-benchmarks-proxy-generations` | 8/18 | 2026-10-19 | Missing 10 labels (see `exp03/research_notes.md`) |

### Checks run on `nischaydnk/ntl-topic-synthetic-18llms`

`prepare_independent_eval.py prepare` verifies these and refuses to write output otherwise:

| Check | Result |
|---|---|
| `generations.parquet` sha256 | `265293398f3231032bc8afef4f0bf0b445a0f2b89c7adcc101e63ea568114753` (4,137 rows) |
| Labels present | all 18 |
| First published | 2026-11-11 00:31 UTC, after every V1-mounted dataset version |
| Exact text matches in competition `train.csv` | 0 |
| Largest share of a passage's 8-grams seen anywhere in `train.csv` | 0.06 |
| Exact text matches in `crtproduction/name-that-llm-open-pool` (current v16, 226,191 rows) | 0 |
| Duplicate texts within the set | 0 |
| Prompts | 100 uploader-written topics x 3 instructions, not competition prompts |

Evaluation input built from it (not committed; regenerate with the script): only complete panels, i.e.
(topic, instruction) pairs answered by all 18 labels, so the batch is exactly balanced and prompt-paneled.

| | |
|---|---|
| Panels | 121 (short 89, median 22, long 10) |
| Rows | 2,178 = 121 per label |
| `test.csv` (`id,text`) | 1,059,119 bytes, sha256 `711ed56b79e07b5b14ec0af3b26a25b2a4266d4437307a23e7204d49262dbe4d` |
| `truth.csv` (`id,topic_id,version,model`) | 78,488 bytes, sha256 `5d5b10058b493a2e0e95bff0b09f4b13e2b3f1837693f931486866b16a5ee245` |

### What this dataset does not establish

* **Label provenance is the uploader's claim.** It records no API route, served model id, timestamps, or
  generation settings. For several labels, public APIs have been reported to serve a different model under the
  same name (notably `gemini-3.7-flash`, per `lourenswalters/ntl-community-vetted-05`). A wrong label would show
  up as model error.
* **Different generation protocol from the competition.** Instructions carry explicit style and length cues
  ("with an awe-struck and vivid tone", "in about N words"); complete panels skew short (89 of 121 panels are the
  ~55-word instruction, versus a train mean of ~115 words). Accuracy here measures transfer to a shifted
  distribution, not the hidden test distribution, and is expected to sit below the 0.680 leaderboard number.
* **Size.** 121 rows per label gives a 95% interval of up to about +-9 percentage points on each per-class recall.
* **The uploader's own models were trained on it**; that does not affect V1 (V1's assets pre-date it) but it does
  mean it is not secret, only unseen by V1.

## 4. Independent run and result

### Run

| | |
|---|---|
| Notebook | `ritisingh123/engs108-v1-a0-nativeshare040-indep` v1, built by `build_notebooks.py V1 --mode independent` (`build/V1/independent/`) |
| Difference from the V1 validate notebook | 3 cells: the input cell (copies `test.csv` from the private dataset `ritisingh123/engs108-v1-independent-input` after asserting its sha256), and the mode label in the run-monitor and run-report cells. All model code, config and mounted model assets are identical; rebuilding A0/B0/V1-V3 after the change left every existing notebook byte-identical |
| Input mounted on Kaggle | `test.csv` (`id,text` only), sha256 `711ed56b...4dbe4d` (asserted in the notebook and re-checked on the downloaded copy). `truth.csv` never left the local machine |
| Hardware / wall time | 2x T4, 12,036 s (3h21m); run-report contract checks all passed (ids and order match `test.csv`, 18 valid labels) |
| Execution contract | `native_embedding_share` 0.4, Open Pool v13, 11 stages, Sinkhorn 0.30 / 0.35, 37 typography markers (`docs/raw/runs/V1_independent_v1/engs108_run/execution-contract.json`) |
| Output | `docs/raw/runs/V1_independent_v1/submission.parquet`, sha256 `df2cc0fc9572886d7808ff1c01c36cb9074a4ffba7bed564ca6e140f6a0306f0` |
| Scoring | `prepare_independent_eval.py score --truth truth.csv --submission submission.parquet --out independent` (locally, after the run; no thresholds or weights touched) |

**Asset-version caveat.** Kaggle mounts the latest version of each dataset listed by slug. All V1 datasets were
still at the versions recorded in `docs/raw/dataset_versions.json` except
`prvsiyan/prvsiyans-extended-name-that-llm-dataset` (recorded v12, 2026-11-08; a newer version was published
2026-11-11). V1 references that dataset only as `EXPLICIT_PATHS['lists']['extra_clean']` (`passages_clean.csv`); the only
reader of that ledger's lists is `asset_paths('ultrabert_folds')`, so nothing in the notebook loads the file and we
do not expect it to change predictions, but the original V1 runs and this run
did not necessarily mount the same version of it.

### Result

Files in `independent/`: `confusion_matrix_normalized.png`, `confusion_matrix_counts.png`, `class_metrics.csv`,
`top_confusions.csv` (same label order and plotting code as the in-sample matrix).

| | |
|---|---|
| Accuracy | 1,179 / 2,178 = 54.1% (Wilson 95%: 52.0-56.2%; rows within a panel are not independent, so the true interval is wider) |
| Macro F1 | 0.518 |
| Correct vendor family (claude / gemini / glm / gpt / kimi / qwen) | 80.2%; 568 of the 999 errors stay inside the true family |
| By instruction length | short 779/1,602 (48.6%), median 262/396 (66.2%), long 138/180 (76.7%) |
| Best recall | gpt-6-astra 0.88, claude-opus-5 0.87, gemini-3.1-pro-preview 0.87, claude-sonnet-5 0.79 |
| Worst recall | kimi-k3 0.12, gemini-3.7-flash 0.13, glm-5.3-flash 0.24, glm-5.3 0.30 |

Top confusions (true -> predicted, of 121 per row): gemini-3.7-flash -> gemini-3.8-flash 62;
gemini-3.6-flash -> gemini-3.1-pro-preview 35; glm-5.3-flash -> glm-5.3 32; glm-5.3-flash -> claude-opus-5 28;
glm-5.3 -> claude-opus-5 27.

Reading it:

* Predictions are not balanced although the truth is (121 each): kimi-k3 is predicted 38 times and gemini-3.7-flash
  41 times, while gemini-3.1-pro-preview, gpt-5.5, claude-opus-5 and gpt-6-astra are each predicted 170-180 times.
  The Sinkhorn target mixes in the `train.csv` label prior, and the batch-dependent steps were tuned on
  competition-style batches, so part of this skew is the distribution shift itself.
* Most errors are between sibling models (Gemini Flash versions, GLM vs GLM-Flash, GPT-5.6 Luna/Sol/Terra), the same
  pattern as the 7 in-sample errors, but far more often.
* Accuracy rises steeply with length; 74% of this set is the ~55-word instruction, shorter than the competition's
  passages, which is one reason this number sits below 0.680.
* The low gemini-3.7-flash and kimi-k3 recalls may partly be label noise: the dataset does not record served model
  ids, and gemini-3.7-flash is the label other uploaders flag as served inconsistently (section 3).

To meet a stricter standard (verified served-model provenance and the competition's own prompt protocol), no
existing public dataset qualifies; new passages would have to be generated: fresh prompts written in the
competition style, each answered by all 18 models through logged routes that record the served model id, at
least ~100 panels. The uploader above reports ~$435 for 5,400 requests across the 18 labels.
