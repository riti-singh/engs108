# Iteration plan

## Runs per experiment
Each experiment (A0, B0, V1–V3) is built by `build_notebooks.py <ID>` from `configs/<ID>.json` in two modes:

| Mode | Kaggle notebook | test.csv | Purpose |
|---|---|---|---|
| `validate` | same slug, earlier saved version | 540-row synthetic test = all rows of 30 train prompts (seed 108), ids replaced by blake2s hashes | proves the full-size path runs end-to-end and writes a valid `submission.parquet`; accuracy is in-sample (train rows) and **not** a leaderboard estimate |
| `submit` | same slug, latest saved version | real competition `test.csv` | the version to submit manually; with the 3-row public sample it runs the full pipeline (stubs removed) |

Both modes record `engs108_run/run_report.json` (contract checks, wall time, peak GPU memory from `nvidia-smi` every 10 s).

## Baseline selection rule (decided before any leaderboard feedback)
1. A baseline is feasible only if its `validate` run completes with a valid parquet.
2. Among feasible baselines prefer the one whose validated runtime leaves more headroom under Kaggle's 9 h limit for the hidden test
   (Ref B budgets up to ~8.3 h for ATLAS alone; Ref A has no such budget-filling stage).
3. Tie-break: higher public score of the source (A 0.681 > B 0.678).

## Variant candidates (one existing parameter each, from the unchanged baseline)

| Variant | Ref A candidate | Ref B candidate |
|---|---|---|
| V1 — ensemble weight ±10–20 % | complementary-stack share in `blend_all_rows` 0.15 → 0.18 (core 0.85 → 0.82) | `--comp_w` 0.15 → 0.18 |
| V2 — temperature | ATLAS `prod_temperatures` (1, 2, 4, 8) → (1.2, 2.4, 4.8, 9.6) (×1.2) | no explicit temperature argument exists in `infer.py`; substitute `--sinkhorn` balancing strength 0.30 → 0.25 (justified in README) |
| V3 — existing threshold | ATLAS margin gate `< .3` → `< .25` (fewer rows re-scored) | `--atlas_margin` 0.36 → 0.30 |

The final three are frozen in `configs/V1.json`–`V3.json` (`"frozen": true`) once the baseline is selected and before any
submission. No variant is derived from another; each config lists exactly one `change` edit applied to the baseline notebook.

## Out of scope
No retraining/fine-tuning, no new train.csv reads, no stacking of changes, no leaderboard submissions (user submits manually).
