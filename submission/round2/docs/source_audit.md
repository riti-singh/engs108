# Source audit: public baselines A and B

Pulled 2026-11-10 with `kaggle kernels pull` (CLI 2.2.5) into `refA/` and `refB/`. Everything below was read from the
pulled notebook source and the Kaggle API, not inferred from titles. Raw API output is under `docs/raw/`
(`dataset_versions.json`, `dataset_access.log`, `dataset_meta/`, `asset_tables.md`).

## Summary

| | Ref A | Ref B |
|---|---|---|
| Kernel | [`lightningv08/0-681-qwen3-embedding-lora-qwen3-8-27b`](https://www.kaggle.com/code/lightningv08/0-681-qwen3-embedding-lora-qwen3-8-27b) | [`mathleteutd/0-678-embedding-expert-lora-pool`](https://www.kaggle.com/code/mathleteutd/0-678-embedding-expert-lora-pool) |
| Title | 0.681 \| Qwen3-Embedding LoRA + Qwen3.8-27B | 0.678 \| Embedding Expert, LoRA, Pool |
| Author (API display name) | Vladislav Lavrentev (`lightningv08`) | Rohan (`mathleteutd`) |
| Version pulled (API `current_version_number`) | 16 (last run 2026-11-07 01:43 UTC) | 2 (last run 2026-11-07 01:43 UTC) |
| Kaggle kernel id | 138930242 | 143367879 |
| Notebook license | Not exposed by the Kaggle API; see the notebook page footer. No license file in the pull. | same |
| Accelerator | `NvidiaTeslaT4` (2× T4; both GPUs used), internet off | `NvidiaTeslaT4` (2× T4), internet off |
| Docker image | `gcr.io/kaggle-private-byod/python@sha256:2757e0c7…ad2b2f` | `gcr.io/kaggle-private-byod/python@sha256:de153e6d…eea01` |
| Datasets / kernels / models attached | 21 / 8 / 6 | 22 / 8 / 9 |
| All attached assets public? | Yes (all 26 datasets are `is_private: false` and list files from `ritisingh123`) | Yes |
| Training at inference time | Prefit heads; train.csv used for fusion/balancing and gauss heads | 4-view SVM, TF-IDF LR, LightGBM and Gaussian heads are **fitted on train.csv (+pool) inside `infer.py`/`comp_stack.py`** |
| 3-row shortcut | `emit_known_preview`: if test.csv is exactly the 3 public rows (ids + blake2s fingerprints), write constant `claude-fable-5` and skip every model | `FAST_COMMIT = len(test) < 50`: skip comp stack + `infer.py`, write constant `gpt-5.5` |

## Ref A: inference and ensemble logic (from source)

Notebook writes ~33 modules to `/kaggle/temp/ntl-gsq-v27-runtime/` via `%%writefile`, then runs `ensemble.py` in a subprocess
(cell 49). `ensemble.py main()`:

1. Reads `test.csv` (`id,text`), builds lexical `topic_groups` (prompt groups capped at 18).
2. **Core members** (`config['models']`, each its own GPU child process `core_worker.py` on GPUs 0,1): truncated-decoder LoRA24
   members `q14v5`, `q8v5`, … (Qwen3-14B-Base, Qwen3-8B-Base, Qwen2.5, Llama-free) with Dhruv Tibarewal's v5 adapters plus
   two-stage members; then `core_readout.py` (CPU) applies prefit sparse heads (`ntl666-prefit`: SVC4 / TF-IDF / distort,
   fitted on the Open Pool v13 = 190,588 rows; `primary_svc_weight = 12.0`) and Gaussian shared-prompt heads, giving `core` logits.
3. **Complementary stack** `legacy/comp_cached.py` (style n-grams, family gates, Claude LoRA gate, UltraBERT 5-fold, token-attention
   heads) from the 12-part `ntl660-prefit` numeric cache → `comp` logits.
4. `prior = 0.85·softmax(core) + 0.15·softmax(comp)`.
5. **ATLAS** (`legacy/atlas_worker.py`, Joseph Adamski's source-conditioned likelihood on Qwen3-8B) on rows whose top-2 margin < 0.3
   (all rows if fewer than 50); `prior[atlas_rows] = 0.9·prior + 0.1·atlas`.
6. **GSQ**: Qwen3.8-27B 3-bit task classifier (`gsq/infer.py`, needs transformers 5.14.1 wheels installed with `--target` from
   the artifact dataset); `mixed = (1−0.03)·prior + 0.03·gsq` (`CONFIG['gsq_share']`: 0.07 in cell 4, 0.05 in cell 45, **0.03** in cell 49 = effective).
7. **Native expert** `native_embedding_worker.py`: full Qwen3-Embedding-8B (36 layers, FP16, split over both T4s), five
   prompt-fold rank-16 LoRA states (CHEN Han), 8-depth readout, plain + prompt-centred heads, 512-token covering windows.
8. `native_fusion.finish_native`: per fold, fold-temperature calibration, then geometric mixture
   `softmax((1−s)·log mixed + s·log p_native)` with **s = 0.5** (`CONFIG['native_embedding']['share']`), then marginal
   (Sinkhorn, strength 0.30, target mix 0.35) balancing and 37 typography-marker guards **using `train.csv`** (asserts 2,699 rows);
   average of the five corrected folds → argmax → `submission.parquet`; writer re-reads and asserts ids/labels.

## Ref B: inference and ensemble logic (from source)

Cells write `gauss.py`, `sibling_style.py`, `v34_pairwise.py`, `infer.py`, `comp_stack.py`, `native_embedding_worker.py`,
`native_mix.py`, then run (in order):

1. Cell 2: discovers inputs by glob; **`DATA` = the competition dir that contains `train.csv`**; builds the SVM pool from
   `mathleteutd/ntl-public-pool-v2` filtered to the **labels present in train.csv**; draws "extra" rows from crtproduction's open
   pool (pinned in code to ids < 279271 ≈ release v11 / 190,588 rows) by `match_weight`; copies Llama-3.1-8B and DeBERTa to local disk.
2. Cell 10: pairwise A/B verifier `v34_pairwise.py` if a revision-03 A/B dataset with `inputs/train.csv` is attached
   (checks it equals the competition train.csv).
3. Cell 46: `comp_stack.py` — complementary public stack (style n-grams, LightGBM stylometrics, frozen encoders/decoders,
   LoRA classifiers of Lucas Boesen / vnton37nguyn, token-attention heads, UltraBERT) → `comp_logit.npy`. Skipped when `FAST_COMMIT`.
4. Cell 52: native Qwen3-Embedding-8B expert (same worker/assets as Ref A), capped at 9000 s.
5. Cell 54: `infer.py` with `MODELS` = `q14v5` (w 3.0, gauss 0.70), `q8v5` (2.0, gauss 0.44), `ll8v5` (1.0), `q8v7` (2.0),
   `glm9v3` (1.0), `n14b_proven` (2.8), `q25_14` (1.0), `deb` (1.0); all `max_len 384`, `twin 0.35`. `infer.py` fits TF-IDF LR
   (`--tfidf_w 0.25`), distort (`0.10`), 4-view LinearSVC on train + pool (`--svc_w 10.0`, pool rows ×0.5), nuridudak prefit SVC
   (`--ours_svc_w 1.2`), Top-34 LightGBM (`0.20`); sums weighted log-probs; adaptive centred twins (`--adapt_w 1.0`, `--adapt_thr 0.15`);
   ATLAS on rows with margin < 0.36 at weight 0.22 within the remaining 8.3 h budget; complementary stack at `--comp_w 0.15`;
   native mixture `--native_share 0.4` (per fold `softmax((1−s)·log P + s·log p_native)`, fold average); Sinkhorn
   (`--sinkhorn 0.30`, `--sinkhorn_target_mix 0.35`) → `submission.parquet`. When `FAST_COMMIT` it writes a `gpt-5.5` placeholder instead.

## train.csv dependency (both sources)

Both methods read the competition `train.csv` as part of inference (Ref A: `ensemble.py` native fusion/balancing and gauss
heads; Ref B: data discovery, label filter, and every fitted member in `infer.py`/`comp_stack.py`). Removing it would require
new prefit artifacts (= retraining). **Decision (user, 2026-11-10): keep the sources' own `train.csv` reads, add none, remove the
3-row stubs**, and validate on a full-size synthetic test. Kaggle mounts `train.csv` on the hidden rerun (both sources were scored).

## Blockers and risks

Ref B v2 has two separate problems, both observed on Kaggle in B0 validate v1 (540 rows, `ERROR` after ~2h46m). Neither is
patched: each fix would be a code change, not an environment fix. On the public 3-row sample the source never reaches either
(`FAST_COMMIT` placeholder). B0 stays **blocked**; no B0 submit version exists.

* **B0 blocker 1 — missing `pathlib` import (crash).** `infer.py` references `pathlib` (line 753,
  `_pairwise_path = pathlib.Path('/kaggle/working/v34_pairwise_probs.npy')`) but only imports
  `os, sys, time, gc, argparse, json, re`. The `NameError` makes `infer.py` exit 1 after it has written the parquet, so the
  notebook's `subprocess.run(cmd, check=True)` raises and the run ends `ERROR`. The same error earlier makes `infer.py` skip its
  `ours_svc`/`top34` members (`WARNING: ours_svc / top34 skipped`).
* **B0 blocker 2 — complementary-stack parity mismatch (silent component loss).** Independently of blocker 1, `comp_stack.py`'s
  self-check `lora24_public: our features do not match the checkpoint` fails (max abs diff 6.0615, mean rel diff 0.54065), so the
  complementary stack produces no output and is dropped (`COMP SKIPPED: no stack output`). Likely asset or library drift between
  the attached checkpoint and today's datasets/image; adding `import pathlib` would not fix this. Compact error record:
  `docs/raw/runs/runs.json` (`B0`); the raw kernel log was removed from git (sha256 in `docs/raw/runs/artifact_hashes.json`).
* **No private/disallowed assets**: all 26 datasets, 8 kernel outputs and 9 model instances resolve and are public.
* **Version drift (not a hard blocker):** `kaggle kernels push` attaches the *latest* dataset/kernel versions. Several datasets were
  updated after the sources' last run (2026-11-07 01:43 UTC): `lightningv08/ntl-olmo2-native24-final-inference-assets` (v22,
  2026-11-07 16:24; Ref A pins `artifact_hub_version = 22` and asserts provenance inside), `mathleteutd/ntl-public-pool-v2` (v6,
  2026-11-07 13:05), `crtproduction/name-that-llm-open-pool` (v16, 2026-11-09; Ref B filters ids in code), `prvsiyan/…` (v12,
  2026-11-08), `mathleteutd/ntl-two-stage-lora24-members-q8-glm9` (v1, 2026-11-08). Exact version used per run is recorded in `manifest.csv`.
* Licenses: several assets declare `other` or `unknown` (see table); they are used unchanged and only attached, not redistributed.
* Runtime: Ref B budgets ~8.3 h of the 9 h limit on the hidden test (ATLAS fills the remainder).

## Attached assets

| Dataset | Used by | Current version (2026-11-10) | Last updated | Size (GB) | Visibility | License (dataset metadata) | Files listed |
|---|---|---|---|---|---|---|---|
| `abdullah1afify/ntl-generations` | A/B | 11 | 2026-11-04 | 0.01 | public | Attribution 4.0 International (CC BY 4.0) | 6 |
| `bbucxi/qwen35-9b-safetensors` | A | 1 | 2026-08-16 | 19.33 | public | unknown | 10 |
| `crtproduction/name-that-llm-open-pool` | B | 16 | 2026-11-09 | 0.12 | public | CC-BY-SA-4.0 | 28 |
| `dhruvtibarewal/nlm-enc-v5` | A | 3 | 2026-10-25 | 0.79 | public | other | 6 |
| `dhruvtibarewal/nlm-l24-adapters` | A/B | 7 | 2026-10-21 | 3.73 | public | MIT | 88 |
| `dhruvtibarewal/nlm-l24-adapters-v5` | A/B | 1 | 2026-10-25 | 2.59 | public | other | 54 |
| `dhruvtibarewal/nlm-matched-clean` | B | 3 | 2026-10-23 | 0.08 | public | MIT | 3 |
| `dhruvtibarewal/nlm-offline-wheels` | B | 2 | 2026-10-17 | 0.05 | public | other | 4 |
| `dophuchung/ntl-proxy-generations` | A/B | 1 | 2026-10-17 | 0.00 | public | Attribution 4.0 International (CC BY 4.0) | 2 |
| `josephadamski91/ntl-atlas-qwen3-8b-v1` | A/B | 3 | 2026-10-20 | 0.46 | public | other | 15 |
| `kirilltym/ntl-spectral-style-heads-paired-4404` | A | 1 | 2026-10-24 | 0.03 | public | unknown | 4 |
| `lightningv08/ntl-olmo2-native24-final-inference-assets` | A/B | 22 | 2026-11-07 | 104.72 | public | other | 201 |
| `lucasboesen/ntl-free-proxy-generations` | A/B | 12 | 2026-11-01 | 0.01 | public | Attribution 4.0 International (CC BY 4.0) | 2 |
| `mathleteutd/ntl-public-pool-v2` | B | 6 | 2026-11-07 | 0.11 | public | CC-BY-SA-4.0 | 2 |
| `mathleteutd/ntl-two-stage-lora24-members-q8-glm9` | B | 1 | 2026-11-08 | 1.33 | public | other | 48 |
| `mercurydust/ntllm-chatgpt-a-b` | A/B | 2 | 2026-11-06 | 0.20 | public | CC-BY-SA-4.0 | 191 |
| `nishantdahal/ntl-handwriting-blend` | A/B | 1 | 2026-10-23 | 1.28 | public | CC-BY-SA-4.0 | 15 |
| `nishantdahal/ntl-two-stage-lora` | A/B | 1 | 2026-10-19 | 0.32 | public | CC-BY-SA-4.0 | 11 |
| `nuridudak/ntl-q14-lora` | A/B | 1 | 2026-10-23 | 0.18 | public | other | 4 |
| `nuridudak/ntl-q14-lora-v4` | A/B | 1 | 2026-10-26 | 0.18 | public | other | 4 |
| `nuridudak/ntl-svc-prefit-pool-v4n` | A/B | 1 | 2026-10-28 | 0.21 | public | CC-BY-SA-4.0 | 38 |
| `nuridudak/ntl-two-stage-encoders` | A | 1 | 2026-10-23 | 1.81 | public | apache-2.0 | 14 |
| `pjmathematician/ultrabert-380m-ntl` | A/B | 1 | 2026-10-14 | 3.82 | public | apache-2.0 | 56 |
| `prvsiyan/prvsiyans-extended-name-that-llm-dataset` | A/B | 12 | 2026-11-08 | 0.13 | public | Attribution 4.0 International (CC BY 4.0) | 14 |
| `seyitkaangunes/ntl-extra-api-passages` | A/B | 2 | 2026-10-16 | 0.00 | public | Attribution 4.0 International (CC BY 4.0) | 3 |
| `zbigniewkirchner/ntl-qwen3-8b-lora-claude-head` | A/B | 1 | 2026-10-16 | 0.32 | public | apache-2.0 | 17 |

| Kernel output | Used by |
|---|---|
| `lucasboesen/name-that-llm-desklib-weights` | A/B |
| `lucasboesen/name-that-llm-electra-large-weights` | A/B |
| `lucasboesen/name-that-llm-extract-encoders2` | A/B |
| `lucasboesen/name-that-llm-glm4-9b-trunc25` | A/B |
| `lucasboesen/name-that-llm-train-lora24-s7` | A/B |
| `lucasboesen/name-that-llm-train-tokattn` | A/B |
| `lucasboesen/name-that-llm-train-tokattn2` | A/B |
| `vnton37nguyn/ntl-train-lora-qwen25-instyle` | A/B |

| Model | Used by |
|---|---|
| `alexchilton/llama-3-1-8b-base/Transformers/default/1` | B |
| `qwen-lm/qwen-3-embedding/Transformers/8b/1` | A/B |
| `qwen-lm/qwen-3/Transformers/14b-base/1` | A/B |
| `qwen-lm/qwen-3/Transformers/14b/1` | B |
| `qwen-lm/qwen-3/Transformers/4b-base/1` | A/B |
| `qwen-lm/qwen-3/Transformers/8b-base/1` | A/B |
| `qwen-lm/qwen-3/Transformers/8b/1` | B |
| `qwen-lm/qwen2.5/Transformers/14b/1` | A/B |
| `qwen-lm/qwen2.5/Transformers/7b/1` | A/B |
