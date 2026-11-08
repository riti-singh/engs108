# Exp03 Agent 1: research notes and data audit

Branch `exp03-research`, stacked on PR #2 head `550063c36b5f58f13f6c1672ebe4cd829a2ea689` (re-verified with `git fetch` on 2026-11-08; unchanged).
Scope: research and data audit only. No training, no Kaggle kernels, no leaderboard submissions.

**Status of the two candidate datasets: both are candidate augmentation data and are not approved for training. Nothing from either dataset was merged into any training set, and no rows were selected using the 414-row dev set.**

Machine-readable results: `exp03/artifacts/data_audit.json`, produced by `exp03/research/audit_datasets.py`. The script reads local files only and is deterministic; two runs gave identical output.

## 1. Datasets audited (the exact refs Riti supplied)

| ref | status |
|---|---|
| `maniceet/name-that-llm-openrouter-openai-generations` | **Not audited. The dataset is inaccessible to us** (see 1.1) |
| `zakirkhanaleemi/name-that-llm-benchmarks-proxy-generations` | Fully audited, version 1 (see 1.2) |

These two refs replace the manager's earlier candidate list. Other public datasets get brief notes only (section 3).

### 1.1 `maniceet/name-that-llm-openrouter-openai-generations`: blocker

Every access attempt on 2026-11-08 used Kaggle CLI 2.2.5 with account `ritisingh123`:

- `kaggle datasets files | metadata | status | download`: **HTTP 403** "You don't have access to this resource."
- An unauthenticated GET of `https://www.kaggle.com/datasets/maniceet/name-that-llm-openrouter-openai-generations` returned **HTTP 404**.
- `kaggle datasets list --user maniceet` shows only four datasets, all from 2018 (histograms, histogram, keyalldays, teamscsv).
- Searches for `openrouter-openai-generations` and "name that llm" return no dataset with this ref.
- None of the 6 public notebooks I pulled list it in `dataset_sources`.

The dataset is therefore private, deleted, renamed, or the ref is mistyped. Its contents are unknown. I have recorded **no** version, hash, class count, provenance, duplicate, or shift result for it, and I have inferred none. Two consequences:

- The cross-dataset duplicate check (maniceet vs zakirkhanaleemi) could not be run.
- On rule compliance, a dataset that other competitors cannot access currently fails the "publicly and equally accessible" condition.

**I need the corrected ref, or a confirmation that the dataset is public, to finish this half of the audit.**

### 1.2 `zakirkhanaleemi/name-that-llm-benchmarks-proxy-generations` (Kaggle id 12703965, **version 1**)

Files and SHA-256 hashes:

| file | sha256 |
|---|---|
| `ntl_proxy_generations.csv` (928,623 B) | `a8f61b647b5f2b2f99b73f083b7358445d93c3ebbff8607b44423850e9fb5583` |
| `README.md` | `f3fcb7159573d80b144d3e48aaa9532c121ff9b8b9774300eaed25ca12b0bc1b` |
| `dataset-metadata.json` (from `kaggle datasets metadata`) | `8298bf6edeac543492644f9d57ec621ae9c0c4aa0ac3775ca3945101f3d8ba5e` |

- Size: 834 rows × 13 columns.
- License: CC BY 4.0.
- Published 2026-10-19 10:03 UTC. The rows were generated between 03:47 and 04:09 UTC the same day.

**Class coverage: 8 of 18 classes.** Only GPT and Gemini models appear, with no labels outside the 18 target classes.

| label | rows | label | rows |
|---|---|---|---|
| gemini-3.1-pro-preview | 186 | gpt-5.5 | 81 |
| gpt-6-astra | 133 | gpt-5.6-luna | 79 |
| gemini-3.6-flash | 111 | gpt-5.6-sol | 73 |
| gemini-3.8-flash | 107 | gpt-5.6-terra | 64 |

Absent, 0 rows each: all five Claude labels, gemini-3.7-flash, glm-5.3, glm-5.3-flash, kimi-k3, and qwen3.8-max. The classes are imbalanced, with up to a 2.9× ratio between the largest and smallest.

**Provenance.** "Verified" means the fact can be checked against the data itself; "claimed" means it rests only on the uploader's README or metadata.

| fact | value | status |
|---|---|---|
| Source model | Kaggle Benchmarks Model Proxy slugs, one per label, e.g. `openai/gpt-5.5-2026-04-23`, `openai/gpt-6-astra`, `google/gemini-3.6-flash` (full map in the JSON) | **Claimed.** The `model_slug` column and README are consistent, but no generation code or proxy logs are published, and the mapping from proxy slug to competition label is the uploader's. |
| Indirect identity check | An 8-way char TF-IDF logistic regression fit on the Exp01 training rows of these 8 classes scores 0.317 balanced accuracy and 87.3% family accuracy on this dataset. The same model scores 0.403 and 96.4% under 5-fold prompt-grouped CV on train. Chance is 0.125. | Verified (computed). The labels mostly agree with the train styles, but there is a clear shift. |
| Settings | `rlow_t1` (reasoning=low, temperature=1.0, max_tokens=4096, daily runs): 747 rows. `rlow_t0` (reasoning=low, temperature=0, `ab2`): 80 rows, 10 per label. `default` / `t1` / `t1_rlow` (`ab_settings`): 7 rows. | **Claimed.** The setting labels are per-row metadata, but there is no code. |
| Request template | `"<prompt>\n\nWrite a single paragraph of about N words, with no title or headings."` | Verified from the `request` column. N tracks the output length (corr 0.957, median n_words/N = 1.00). |
| Prompt source | 213 prompts: bank `sy` (prompts.csv of seyitkaangunes/ntl-extra-api-passages) for 821 rows, bank `pv` (prvsiyan extended dataset) for 13 rows | **Claimed** source. The README's claim that no prompt was derived from competition data is consistent with the topic check below. |
| Output | All 834 rows have `finish_reason=stop`; 833 are single-paragraph; text is "unedited apart from whitespace strip" | The fields are verified; the "unedited" part is claimed. |

**Duplicates.** Near-duplicate detection used two tests: word 5-gram Jaccard ≥ 0.5, or char 3–5 gram TF-IDF cosine ≥ 0.9. Exact duplicates were matched on lowercased text with punctuation and whitespace normalized.

| comparison | exact | near | max Jaccard5 | max char cosine |
|---|---|---|---|---|
| within the dataset (347,361 pairs) | 0 | 0 | 0.352 | 0.799 |
| vs competition train (2,699 texts) | 0 | 0 | 0.015 | 0.246 |
| vs competition test (3 local texts) | 0 | 0 | 0.004 | 0.193 |
| vs maniceet | not run (dataset inaccessible) | | | |
| reference: within train | 0 | 0 | 0.134 | 0.624 |

The `rlow_t0` (temperature 0) runs did not produce duplicate text.

**Prompt and topic overlap with train.** Train has no prompt text, so I represented each train prompt by the content-word TF-IDF centroid of its passages.

- Calibration within train: half-prompt centroids of the same prompt have a 1st-percentile cosine of 0.495. A half-prompt's best match to a different prompt has a 99th-percentile cosine of 0.535.
- This dataset's 213 prompt centroids have a maximum cosine of **0.334** to any train prompt, and the 99th percentile is 0.257.
- **0 of 213 prompts exceed the 0.535 threshold.** No topic overlap was detected.
- The prompt-id namespaces (`g0270`-style vs `p081`) do not intersect.
- Against the hidden test, only the 3 local test rows could be checked.

**Distribution shift against train, same 8 classes.**

- A domain classifier (char_wb 2–4 TF-IDF + logistic regression, grouped by prompt, 5-fold) separates train from this dataset with **out-of-fold AUC 0.832**. That is a strong shift.
- Length matches overall: mean 111 vs 113 words, KS 0.034, p=0.60.
- Length differs by class: median words are 87 vs 111.5 for gpt-5.6-sol, 88.5 vs 100.5 for gpt-5.6-terra, and 101 vs 84 for gemini-3.8-flash.
- The largest style shifts, pooled over the 8 classes (KS statistic; the per-class tables are in the JSON):

| feature | train | this dataset | KS |
|---|---|---|---|
| uppercase chars /1k | 9.26 | 12.29 | 0.27 |
| words per sentence | 23.7 | 20.7 | 0.22 |
| digits /1k | 0.48 | 2.73 | 0.22 |
| hyphens /1k | 0.61 | 1.18 | 0.16 |
| semicolons /1k | 0.69 | 0.46 | 0.14 |
| any curly quote | 38% | 51% | 0.13 |

The extra digits and capitals most likely come from the prompt bank: the seyit prompts are fact-heavy science and history questions, such as the Chien-Shiung Wu / cobalt-60 prompt. This is my interpretation, not a verified cause. Dev-set hygiene for these checks:

- The style statistics describe all 2,699 train rows. They are descriptive only and select nothing.
- The domain and label-consistency classifiers were fit on the 2,285 Exp01 training rows only.

**Coverage for within-family use** (the way the public "gates" use extra data):

- GPT: 430 rows on 146 prompts. **Only 44 prompts carry all 5 GPT labels**; 72 carry only one.
- Gemini: 404 rows on 182 prompts. 94 prompts carry all 3 *present* labels, but gemini-3.7-flash is missing entirely.

Prompts that carry only one label let topic act as a label shortcut. Restricting to family-complete prompts gives 270 GPT rows (44 prompts).

**Rule compliance.**

- The competition rules require external data to be publicly and equally accessible at no cost, or to meet the reasonableness criteria. Code Requirements allow external data, offline, within 9 h.
- Host posts in discussion 752084:
  - 2026-10-28: "Existing public community datasets on Kaggle are fine to train on."
  - 2026-10-29: "It is not the intent of the Benchmarks proxy quota to be used for non-benchmarking work; however, we will not penalize participants who have already used it … Public community datasets on Kaggle (including those generated via the proxy) are fine to train on."
- **zakirkhanaleemi: likely compliant.** It is public, free, CC BY 4.0, and was published before the clarification. The host explicitly allows public datasets generated via the proxy.
- Obligations if it is used: attribute the dataset and both prompt banks (CC BY 4.0), and do not generate more proxy data ourselves.
- This is my reading of public posts, not a ruling.
- **maniceet: undetermined**, because we cannot access it.

**Verdict, for the user to decide.** The data is clean: no duplicates and no leakage into train or test. It is usable only as within-family data for GPT and Gemini, and gemini-3.7-flash is missing. The shift is strong (AUC 0.83), and the provenance of model identity and settings rests on the uploader's claim. Expected value is small: the public GPT gate built on about 1,342 GPT passages reported +0.013 OOF (self-reported, section 2), and this dataset has 270 family-complete GPT rows. I do not recommend adding it to the 18-way QLoRA training set.

## 2. Public notebooks and discussions

Notebooks were pulled with `kaggle kernels pull` on 2026-11-08. All leaderboard (LB) numbers are **self-reported** by the authors; I did not run any of these notebooks.

**Author identity.**

- **PiKa919**: no Kaggle user or kernel by this name exists (`kaggle kernels list --user PiKa919` / `-s PiKa919` returned "Not found"). The closest match is **`pikaaa` (display name "PiKaaa")**, which has 2 competition notebooks. I treat these as the likely intended author, but **please confirm**.
- **jerrythomas**: the user `jerrythomas` exists but has only 2016–2018 kernels. None of the 1,034 competition kernels I listed is by a "jerry" author. **This author was not found.** I did not substitute anyone for them.

| notebook | technique | status |
|---|---|---|
| `lucasboesen/lora-x-adaptive-centering` (v5; v4 LB 0.550, v3 0.549, self-reported) | **Capped adaptive centering**: average-linkage clustering of test passages. The distance is either content-word TF-IDF cosine (cut 0.85) or the mean of TF-IDF and GLM-4-9B layer-24 cosine (cut 0.60). Only clusters of size 2–18 are centered (`ADAPT_MIN/MAX`), never singletons. A "twin" head trained on prompt-centered train features (`group_center`: subtract the group mean) is **added** as extra evidence (weight 1.5×w, `ADAPT_MODE='add'`); the plain head is kept. | Verified in code |
| same | Sinkhorn marginal balancing of the final posterior (`BALANCE_ALPHA=0.5`) | Verified in code (constant and comment). I did not trace the full function. |
| same | Blend weights: `stylegram` (4-view n-gram SVC) 1.516 is by far the largest, with frozen-LM heads, encoders, LoRA, and token-attention heads | Verified in code (`W` dict) |
| same | Claude gate: a 5-way stylegram SVC adds 1.5× log P inside the Claude family, trained on train Claude rows plus 4,998 seyit API passages. GPT gate: 5-way SVC + DeBERTa head trained on GPT rows plus about 1,342 free-proxy GPT passages (lucasboesen / abdullah1afify / dophuchung datasets, not zakirkhanaleemi). The author reports OOF +0.0018 (v4) and +0.013 (v5). | The gates are described in markdown, and their constants appear in code. The gains are self-reported. |
| `zbigniewkirchner/0-552-adaptive-centering-lora-claude-gates` (0.552, self-reported) | Same lineage (reuses lucasboesen kernels); origin of the Claude-gate idea | Code present, same functions. I did not diff it in detail. |
| `pikaaa/name-that-llm-v20-pool-centred` | alexchilton's 0.660 pipeline plus three changes: (1) pooled external pool (nlm_pool_v5 + crtproduction open pool, deduplicated, capped at 11k/model); (2) new `svc4c`, a 4-view LinearSVC on prompt-centered official + pool rows, scored on inferred clusters of size 2–18; (3) ATLAS weight 0.2 | Verified in code. Header claim: grouped-CV member accuracy 0.500 vs 0.440 plain (2 folds, self-reported). |
| same (markdown: "What we tried that did not work") | On the LB: the twin rule went 0.560 → 0.588. Hard Hungarian assignment inside clusters **lost every time** (−0.009 to −0.017). Nested CV of 0.6865 became LB 0.56, because "the offline folds used true 18-passage prompt groups". | Described only (self-reported LB) |
| `pikaaa/name-that-llm-sota-master-ensemble` | Docstring lists OT Sinkhorn, cavity calibration, prompt-subspace deconfounding, and a "0.70+ target" | **Described only.** The docstring is aspirational, and I found no reported score. |
| `alexchilton/name-that-llm-0-680-all-public-what-failed` | Assembles about 21 public datasets and adapters into a 0.680 ensemble (self-reported) | Described. I did not audit its 4.6k lines. |
| `lucasboesen/0-499-llm-heads-adaptive-prompt-centering` | Earlier version: frozen LLM heads + centering | Pulled, not reviewed in depth |

**Prompt-grouping discussions** (topics 750213, 750390, 750706, 750768, 752084):

- **The test layout is unknown.** In train, each prompt has exactly 18 passages (one per model). The hidden test has about 4,000 passages on unseen prompts with unequal counts per model. In topic 750768, participants infer "at most once per model per prompt", but no host answer confirms this.
- **Simulation (750213).** With the true groups, centering gains +0.101 at 18 answers per prompt but −0.067 at 2. With groups recovered by clustering, it gains +0.098 in train-like CV but −0.102 at 1 answer per prompt.
  - Hungarian assignment is about 0 at 1–2 answers per prompt.
  - Using real n-gram SVC OOF scores (0.370), centering of log-probs gives −0.062 at 2 per prompt and +0.045 at 18.
  - Mechanism: centering a small group removes the passage's own author signal. A **cavity (leave-self-out) estimate** avoids this. Wrong cluster merges are what hurt.
- **LB evidence (750390, self-reported).** Clustering + Hungarian took a run from 0.353 to 0.344 on the LB, and adding centering brought it to 0.329, even though CV showed +0.10.
- **Conclusion.** Any test-time grouping must be validated with a **pooled simulation at 1–5 passages per prompt**, never with train-layout grouped CV. That is consistent with "prompt-centering has mixed evidence".

## 3. Other datasets (brief notes only, not audited)

These datasets appear in the pulled notebooks' `dataset_sources`, but I did not download them and record no counts:

- seyitkaangunes/ntl-extra-api-passages: the Claude-gate data, and the prompt bank behind zakirkhanaleemi.
- lucasboesen/ntl-free-proxy-generations, abdullah1afify/ntl-generations, dophuchung/ntl-proxy-generations: the GPT-gate data.
- crtproduction/name-that-llm-open-pool: about 213k passages, the most voted.
- prvsiyan/prvsiyans-extended-name-that-llm-dataset.
- dhruvtibarewal/nlm-matched-clean.

Any of them would need the same audit before use.

## 4. Recommendations

Ranked by expected value relative to cost. All are proposals; none is implemented here. All selection should use prompt-grouped CV on the 2,285 Exp01 training rows.

1. **4-view n-gram "stylegram" LinearSVC** (char, word, shape, and function-word views). It is the dominant blend member in the verified lucasboesen code. Our current stylometric model scores 0.336, and an n-gram SVC is reported at about 0.37 grouped OOF (750213). Value: high. Cost: CPU, minutes.
2. **Probability-level blend of the stylegram with the Exp02 probe/QLoRA log-probs**, with weights fit on grouped OOF. Value: high. Cost: CPU only. Prerequisite: recommendation 1.
3. **Capped, additive centering ("twin" rule)** on probe/stylegram features. Use clusters of size 2–18 only, never singletons. Keep the plain head and add the centered evidence, preferably with a cavity estimate. Evaluate it **only** with a pooled simulation that subsamples 1–5 passages per prompt from held-out folds. Value: medium (+0.028 on the LB is self-reported). Cost: CPU. Risk: train-layout CV overstates the gain by about 0.1.
4. **Sinkhorn marginal balancing (α≈0.5)**. Value: low–medium. Cost: trivial. Evaluate it under unequal class counts in the same simulation.
5. **Do not implement Hungarian or hard one-author-per-cluster assignment.** It lost every time in self-reported LB results.
6. **Within-family gates with external data.** Use only after the user approves a dataset, and only on prompts that carry all labels in the family. zakirkhanaleemi can contribute 270 family-complete GPT rows (44 prompts) and no data for Claude, GLM, Kimi, Qwen, or gemini-3.7-flash. A Claude gate would need seyit (not audited). Value: low–medium. Cost: CPU. Fit the gates with train-grouped CV, and never choose rows using the dev set.
7. **Multi-layer pooled hidden-state readouts and a second LM backbone** (the two-stage LoRA lineage). Value: medium–high, but the cost is GPU: it needs Kaggle GPU, which this worker did not launch.

## 5. Reproducibility

- Command: `python exp03/research/audit_datasets.py --comp <dir with train.csv,test.csv,sample_submission.parquet> --zak <unzipped zakirkhanaleemi v1> --out exp03/artifacts/data_audit.json` (about 90 s on CPU).
- Seed: 42. Environment: Python 3.12.16, numpy 2.5.4, pandas 3.1.1, scipy 1.18.1, scikit-learn 1.6.1.
- Competition file SHA-256:
  - train `40a22ce8…7707`
  - test `4545aa3a…0c57`
  - sample_submission `ccf86796…6188`
  - Full hashes are in the JSON.
- The local `test.csv` is the 3-row public stub.
- The Exp01 split is recreated with `GroupShuffleSplit(test_size=0.15, random_state=42)`, asserting 2,285 training rows.
- No data files are committed.
