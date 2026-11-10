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

## Selection outcome (from executed runs)
* **A0 feasible.** validate v3 completed (540 rows, all contract checks pass, 1h33m50s, peak 10.4/10.2 GiB on the two T4s);
  submit v4 completed on the real 3-row `test.csv` (50m39s, all checks pass).
* **B0 not feasible as published.** validate v1 ended `ERROR` after ~2h46m: Ref B v2's `infer.py` uses `pathlib` without importing it
  (`NameError` at `infer.py` line 753, after the parquet write), so `subprocess.run(cmd, check=True)` raises. The same `NameError`
  earlier made `infer.py` skip `ours_svc`/`top34`, and `comp_stack.py` failed its own parity assert
  (`lora24_public: our features do not match the checkpoint`, mean rel 0.54) so the complementary stack was skipped.
  Fixing this is a code change, not an environment fix, so B0 is recorded as a blocker and not patched.
* **Selected baseline: A0** (rule 1). The B-based drafts (`--comp_w`, `--sinkhorn`, `--atlas_margin`) were discarded unrun.

## Variant candidates (one existing parameter each, from the unchanged baseline)

| Variant | Ref A candidate | Ref B candidate |
|---|---|---|
| V1 — ensemble weight ±10–20 % | complementary-stack share in `blend_all_rows` 0.15 → 0.18 (core 0.85 → 0.82) | `--comp_w` 0.15 → 0.18 |
| V2 — temperature | ATLAS `prod_temperatures` (1, 2, 4, 8) → (1.2, 2.4, 4.8, 9.6) (×1.2) | no explicit temperature argument exists in `infer.py`; substitute `--sinkhorn` balancing strength 0.30 → 0.25 (justified in README) |
| V3 — existing threshold | ATLAS margin gate `< .3` → `< .25` (fewer rows re-scored) | `--atlas_margin` 0.36 → 0.30 |

**Final (frozen, A0-based):** V1 `CONFIG['native_embedding']['share']` 0.5 → 0.4; V2 `core_math.balance_logits` default
`strength` .30 → .25; V3 ATLAS margin `< .3` → `< .25`. Rationale per variant is in `configs/V*.json` (`change.why`) and the README.
The complementary-share V1 candidate was replaced by the native share because the latter is one literal with no pinning assert;
`prod_temperatures` was replaced by the balancing strength because it lives in a dataset file.

The final three are frozen in `configs/V1.json`–`V3.json` (`"frozen": true`) once the baseline is selected and before any
submission. No variant is derived from another; each config lists exactly one `change` edit applied to the baseline notebook.

## Out of scope
No retraining/fine-tuning, no new train.csv reads, no stacking of changes, no leaderboard submissions (user submits manually).

## Notes from inspecting candidate parameters (pre-run)
* Ref A's complementary share is hard-coded twice in `ensemble.py` (`prior = .85 * softmax(core) + .15 * softmax(...)`, cell 41)
  and pinned by `assert config['complementary_share'] == .15` plus a provenance record; a V1 on A means 4 coordinated literal edits for one parameter.
* Ref A's ATLAS `prod_temperatures` is read from the attached ATLAS bundle config (`self.config['prod_temperatures']`), not from
  notebook code; changing it would require overriding a dataset file.
* Ref B exposes every candidate as a single `infer.py` CLI argument in cell 54 (`--comp_w`, `--sinkhorn`, `--atlas_margin`,
  `--native_share`, …), so each variant is one literal edit. This favours B for clean single-variable variants if both run.
