# Readiness report (A0, V1, V2, V3; B0 blocked)

Nothing here has been submitted to the competition; leaderboard scores are blank in `manifest.csv`.

## Production (submit) versions

| ID | Kaggle notebook | Saved version | Frozen config | Behavioural change vs A0 |
|---|---|---|---|---|
| A0 | [ritisingh123/engs108-a0-qwen3-embed-lora](https://www.kaggle.com/code/ritisingh123/engs108-a0-qwen3-embed-lora) | v4 | `configs/A0.json` v1 | none (compat edits only) |
| V1 | [ritisingh123/engs108-v1-a0-nativeshare040](https://www.kaggle.com/code/ritisingh123/engs108-v1-a0-nativeshare040) | v2 | `configs/V1.json` | `native_embedding.share` 0.5 → 0.4 |
| V2 | [ritisingh123/engs108-v2-a0-balance025](https://www.kaggle.com/code/ritisingh123/engs108-v2-a0-balance025) | v3 | `configs/V2.json` v3 | `balance_logits` strength .30 → .25 |
| V3 | [ritisingh123/engs108-v3-a0-atlasmargin025](https://www.kaggle.com/code/ritisingh123/engs108-v3-a0-atlasmargin025) | v2 | `configs/V3.json` | ATLAS gate `< .3` → `< .25` |
| B0 | [ritisingh123/engs108-b0-embed-expert-pool](https://www.kaggle.com/code/ritisingh123/engs108-b0-embed-expert-pool) | none | `configs/B0.json` | **blocked** (see below) |

V2 v3 differs from V2 v2 only in the `'sinkhorn_strength'` value written to `inference-logs/execution-contract.json`
(.30 → .25) and the `config_version` in the run report; that record is never read, so predictions are unaffected. V2 v3
completed (58m, peak ~9.7 / 10.0 GiB), its contract records `sinkhorn_strength: 0.25`, and its `submission.parquet` is identical
to v2's. V2's 540-row validation run (v1) used config v2, which has the same behaviour.

Production checks (`verify_kaggle_versions.py` on notebooks pulled from Kaggle after cleanup): for each of A0/V1/V2/V3 the
latest saved version's code cells are identical to `build/<ID>/submit`; all four sources are distinct; each reads
`/kaggle/input/competitions/name-that-llm/test.csv` via `config['paths']['test_csv']`, contains no reference to the validation
directory, and has Ref A's fingerprint preview disabled (`preview = False`, no `emit_known_preview(DATA,None,OUTPUT)` call).
At runtime every submit run processed the 3 real test rows through the full pipeline (all 11 stages, ATLAS on all 3 rows
because the source rescores every row when the test has < 50 rows); none took the constant-output preview.

## Validation comparison vs A0 — IN-SAMPLE, not a generalization benchmark

All four ran on the same 540-row synthetic test (30 train prompts, seed 108). Those rows come from `train.csv`, which the
pipeline also reads at inference time, so accuracy and agreement here say nothing about leaderboard performance.
Counts from `compare_validation.py`:

| Variant | Changed predictions vs A0 (of 540) | Note |
|---|---|---|
| V1 | 3 | 1 of the 3 moved to the correct label, 2 changed between wrong labels; in-sample correct 533 vs A0 532 |
| V2 | 0 | |
| V3 | 0 | ATLAS rescored 9 rows instead of 12; the 3 dropped rows kept the same final label |

Code branches exercised (from each run's `inference-logs`, extracted to `docs/raw/runs/inference_summary.json`):
all runs executed the 6 core workers (`q14v5, q8v5, ll8v5, n14b, q25_14, deb`), `core-readouts-cpu`, `complementary`, `atlas`,
`gsq-two-t4` and `native-q8-fivefold`, then `balance_logits` and the typography guards. On 540 rows the ATLAS margin branch
(`len(ids) >= 50`) was used: 12 rows rescored for A0/V1/V2, 9 for V3. V1's change acts in `native_mixture` (share 0.4 recorded
in the contract); V2's in `balance_logits`; V3's in the ATLAS row selection.

## B0 — blocked (two separate problems, neither patched)

1. **Missing `pathlib` import** in Ref B's `infer.py` (line 753): `NameError`, `infer.py` exits 1, notebook ends `ERROR`;
   the same error also skips the `ours_svc`/`top34` members.
2. **Complementary-stack parity mismatch**: `comp_stack.py` asserts `lora24_public: our features do not match the checkpoint`
   (max abs 6.0615, mean rel 0.54065) and the stack is dropped. Independent of (1).

Details: `docs/source_audit.md`.

## Limitations
* Kaggle attaches the latest version of each dataset; author updates can change results.
* V2 and V3 produced identical predictions to A0 on all 540 validation rows and on the 3 visible test rows, so they may score
  the same as A0.
