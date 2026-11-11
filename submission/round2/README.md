# Round 2: combined A0 parameter variants (S1–S4) for "Name That LLM"

Four A0-derived Kaggle notebooks, each = A0 (Ref A + the A0 compat edits) + the parameter edits listed below. They are
saved as private Kaggle versions under `ritisingh123`. **Nothing was submitted to the competition**; submitting is left to the user.

Copied unchanged from `submission/public-baselines/`: `refA/` (Ref A v16 pull, incl. its metadata), `configs/A0.json` (and
`V1`–`V3` for reference), `check_notebooks.py`, `docs/source_audit.md`, `docs/raw/dataset_versions.json`.
`build_notebooks.py` has one change: a config may hold either the existing single `"change"` object or a `"changes"` list.
Every entry in `"changes"` goes through the same exact-one-match `apply_edit`, and each entry's optional `provenance_edits`
goes through the existing `PROVENANCE_ONLY` guard. Everything else is unchanged: submit/validate modes, monitor cell,
layout mapper, report cell, metadata, GPU/internet settings and output-contract checks.

| Path | What |
|---|---|
| `configs/S1.json` … `S4.json` | frozen configs, `"base": "A0"`, `"changes"` = exact `old`/`new` source text |
| `build_notebooks.py` | `python build_notebooks.py S1 S2 S3 S4 --mode submit` → `build/<ID>/submit/` |
| `check_notebooks.py` | compiles every generated code cell and `%%writefile` module |
| `verify_static.py` | checks the built notebooks against Ref A and the submit contract (below) |
| `run_queue.py` | pushes `build/<ID>/submit` (Save & Run All), retries while the 2-GPU-session cap is reached, polls each kernel independently, downloads outputs |
| `verify_outputs.py` | checks a downloaded `submission.parquet` against the real competition `test.csv` |
| `verify_kaggle_versions.py` | checks notebooks pulled back from Kaggle against `build/<ID>/submit` |
| `make_manifest.py` → `manifest.csv` | one row per experiment (+ A0/V1–V3 reference rows) |
| `docs/prior_public_scores.json` | public scores of A0/V1–V3 as confirmed by the user (not measured here) |
| `docs/raw/runs/` | `runs.json` (versions, output checks), per-run `submission.parquet` + `engs108_run/run_report.json`, push/status logs |

## The four configurations

All edits hit the same Ref A cells as the round-1 variants: native share is cell 49 (V1), `balance_logits` is cell 6
`core_math.py` (V2), and the ATLAS gate and the execution-contract record are cell 41 `ensemble.py` (V3 / V2 provenance edit).
Each config also includes the A0 compat edit `remove_public_sample_preview` and the A0 layout mapper.

| ID | Kaggle slug | Native share (cell 49) | `balance_logits` strength (cell 6) | ATLAS margin (cell 41) | contract `sinkhorn_strength` record (cell 41, provenance only) |
|---|---|---|---|---|---|
| A0 | `engs108-a0-qwen3-embed-lora` | 0.5 | .30 | `< .3)` | .30 |
| S1 | `engs108-s1-a0-combined-r2` | `'share': 0.5}` → `'share': 0.4}` | `strength=.30` → `strength=.25` | `< .3)` (unchanged) | `.30,` → `.25,` |
| S2 | `engs108-s2-a0-nativeshare060` | `'share': 0.5}` → `'share': 0.6}` | `strength=.30` → `strength=.25` | `< .3)` (unchanged) | `.30,` → `.25,` |
| S3 | `engs108-s3-a0-balance015` | `'share': 0.5}` → `'share': 0.4}` | `strength=.30` → `strength=.15` | `< .3)` (unchanged) | `.30,` → `.15,` |
| S4 | `engs108-s4-a0-atlasmargin035` | `'share': 0.5}` → `'share': 0.4}` | `strength=.30` → `strength=.25` | `< .3)` → `< .35)` | `.30,` → `.25,` |

Not changed: feature extraction, prompt grouping, preprocessing, model lists, complementary/GSQ shares, datasets, kernels,
model assets, source attribution/header cells, training behaviour.

## Prior public scores (round 1, confirmed by the user)

| ID | Saved version | Change vs A0 | Public score |
|---|---|---|---|
| A0 | v4 | none | 0.679 |
| V1 | v2 | native share 0.5 → 0.4 | 0.680 |
| V2 | v3 | balance strength .30 → .25 | 0.680 |
| V3 | v2 | ATLAS margin .3 → .25 | 0.679 |

## Static verification (`verify_static.py`, all PASS)

For each `build/<ID>/submit/`: unique `kernel-metadata.json` `id`/`title`/`code_file`; private; `enable_gpu=true`,
`enable_internet=false`; `machine_shape` (`NvidiaTeslaT4`) and Docker image equal Ref A's; `dataset_sources`,
`kernel_sources`, `model_sources`, `competition_sources` equal Ref A's; `test.csv` is read through
`/kaggle/input/competitions/name-that-llm` (`config['paths']['test_csv']`, remapped by the layout mapper if the session
mounts the flat layout); no `/kaggle/working/ntl_val_data`; `preview = False` and no `emit_known_preview(DATA,None,OUTPUT)`;
writes `/kaggle/working/submission.parquet`; the report cell asserts columns exactly `['id', 'model']`; markdown/header
cells are identical to Ref A; and every changed code line vs Ref A is one of the configured edits, each applied once.
`check_notebooks.py`: all 4 notebooks and their 33 `%%writefile` modules compile.

## Kaggle runs (Save & Run All, private, GPU T4 ×2, internet off)

| ID | Notebook | Saved version | Runtime | Peak GPU memory | Output checks (real `test.csv`) | Contract record | Visible-test predictions vs A0 v4 |
|---|---|---|---|---|---|---|---|
| S1 | [engs108-s1-a0-combined-r2](https://www.kaggle.com/code/ritisingh123/engs108-s1-a0-combined-r2) | v1 | 45m (2724 s) | GPU0 9.9 / GPU1 10.0 GiB | 3 rows, PASS | share 0.4, sinkhorn 0.25 | identical |
| S2 | [engs108-s2-a0-nativeshare060](https://www.kaggle.com/code/ritisingh123/engs108-s2-a0-nativeshare060) | v1 | 52m (3173 s) | GPU0 9.9 / GPU1 10.0 GiB | 3 rows, PASS | share 0.6, sinkhorn 0.25 | 1 of 3 differs (`005f5a9a6139`: `gpt-5.6-sol` → `gpt-5.5`) |
| S3 | [engs108-s3-a0-balance015](https://www.kaggle.com/code/ritisingh123/engs108-s3-a0-balance015) | v1 | 51m (3085 s) | GPU0 9.7 / GPU1 10.0 GiB | 3 rows, PASS | share 0.4, sinkhorn 0.15 | identical |
| S4 | [engs108-s4-a0-atlasmargin035](https://www.kaggle.com/code/ritisingh123/engs108-s4-a0-atlasmargin035) | v1 | 47m (2854 s) | GPU0 9.9 / GPU1 10.0 GiB | 3 rows, PASS | share 0.4, sinkhorn 0.25 | identical |

Each run finished `COMPLETE` on its first version, so nothing had to be repaired. Runtime and GPU memory come from each
run's `engs108_run/run_report.json` (nvidia-smi sampled every 10 s). Output checks (`verify_outputs.py` on the downloaded
outputs against the real competition `test.csv`) cover: `submission.parquet` exists, has exactly `id, model`, matches
`test.csv`'s row count and id order, has unique ids and no null labels, and every label is one of the 18 valid labels.
All 11 stages ran in every run (`execution-contract.json`). `verify_kaggle_versions.py` on the notebooks pulled back
from Kaggle shows their code cells are identical to `build/<ID>/submit` and the four sources are distinct. The pulled
metadata differs only in model-source order, `keywords: ["gpu"]` and no `docker_image_pinning_type`, which Kaggle sets
on its side.

Limitations:
* The visible `test.csv` has 3 rows. Ref A rescores every row with ATLAS when the test has < 50 rows, so S4's
  margin edit (`< .35`) has no effect on these runs. It only acts on the hidden test.
* Three rows say nothing about leaderboard performance. Kaggle also attaches the latest dataset versions, so author
  updates can change results.

**No competition submission was made.** No notebook was submitted with "Submit to Competition". To score a variant,
submit its saved version (v1) from the Kaggle UI.
