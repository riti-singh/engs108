# V1 validation confusion matrix: training-derived, in-sample diagnostic

**This is not a held-out evaluation and not a leaderboard estimate.** The 540 rows are every passage of
30 prompts drawn from the competition `train.csv` (seed 108, ids re-hashed), and the V1 pipeline reads
`train.csv` at inference time (Ref A's native fusion, Sinkhorn balancing, typography guards and Gaussian heads;
see `docs/source_audit.md`; kept as-is by decision). So the pipeline has seen these exact passages and labels. Treat everything below as an in-sample diagnostic.

* No independent labeled data was used for this matrix: the only labeled data the competition provides is
  `train.csv`, and the hidden test labels are not public. A separate evaluation of the unchanged V1 notebook on an
  independent, externally labeled set (2,178 passages not in `train.csv`) is in `INDEPENDENT_EVAL.md` and
  `independent/`; it is not mixed into the numbers on this page.
* A fresh random split of `train.csv` would still be contaminated: the pipeline reads the whole `train.csv` at
  inference regardless of how we split it, and its prefit public members were trained by their authors on the
  competition train set, which we cannot undo without retraining. No such split is reported anywhere here.

**Do not quote 98.7% as validation or generalization performance.**

## Result (in-sample)

| | |
|---|---|
| Run | `V1_validate_v1` (`ritisingh123/engs108-v1-a0-nativeshare040`, validate mode) |
| Rows | 540 (30 train prompts x 18 models) |
| In-sample accuracy | **533/540** (0.9870) |
| In-sample macro F1 / balanced accuracy | 0.9869 / 0.9870 |
| Off-diagonal (misclassified) rows | 7, spread over 5 distinct (true, predicted) cells |

The 533/540 figure matches `in_sample_accuracy_not_a_score` = 0.987037037037037 in
`docs/raw/runs/V1_validate_v1/engs108_run/run_report.json`.

## Official Kaggle public leaderboard (separate from the diagnostic above)

V1 was submitted to the competition and scored **0.680** on the public leaderboard. This is the only number in this
document measured on text the pipeline did not have labels for. Source: the Kaggle submissions API for account
`ritisingh123`, queried 2026-11-15:

| | |
|---|---|
| Submission ref | `58222435` |
| Notebook | `ritisingh123/engs108-v1-a0-nativeshare040`, scriptVersionId `367976120` |
| Submitted (UTC) | 2026-11-11 02:43:18 |
| Status | COMPLETE |
| Public score | **0.680** |

The public leaderboard is scored on a subset of the hidden test set; the private score is not yet available.
`docs/readiness.md` and the leaderboard column of `manifest.csv` predate this submission and do not record it.
The 0.680 and the in-sample 533/540 measure different things and must not be combined or compared as if
they were the same kind of number.

## Top confusions (true -> predicted)

| # | true | predicted | count | share of true row |
|---|---|---|---|---|
| 1 | gemini-3.8-flash | gemini-3.7-flash | 2 | 0.0667 |
| 2 | kimi-k3 | glm-5.3 | 2 | 0.0667 |
| 3 | gemini-3.8-flash | gemini-3.6-flash | 1 | 0.0333 |
| 4 | glm-5.3 | claude-sonnet-5 | 1 | 0.0333 |
| 5 | kimi-k3 | claude-fable-5 | 1 | 0.0333 |

Ties are broken by the 18-label order (true, then predicted).

## Files

* `make_confusion_matrix.py`: rebuilds ground truth from `train.csv`, verifies alignment, writes all outputs.
* `confusion_matrix_normalized.png`: row-normalized (each true-label row sums to 1), 300 DPI.
* `confusion_matrix_counts.png`: raw counts, 300 DPI.
* `class_metrics.csv`: per-class precision, recall, F1, support (plus predicted count and correct count).
* `top_confusions.csv`: the five largest off-diagonal cells.
* `INDEPENDENT_EVAL.md`: pipeline leakage audit, independent-data protocol, and the independent-set result
  (hand-written); its matrices and metrics are in `independent/`.

Label order on both axes is the 18-label list in the run-report cell of
`build/V1/validate/engs108-v1-a0-nativeshare040.ipynb`.

## Ground-truth reconstruction and alignment

Ground truth (`val_truth.csv`) and the validation `test.csv` are train-derived and were removed from the repo;
only their hashes are kept in `docs/raw/runs/artifact_hashes.json`. The script re-runs the notebook's selection cell
(parameters parsed from the notebook: `RandomState(108).choice(sorted(prompt_id.unique()), 30)`, row shuffle
`random_state=108`, ids `blake2s('engs108-val:' + id, digest_size=6)`) on the competition `train.csv`
and refuses to continue unless every check passes:

* `val_truth_sha256_matches_record`: True
* `test_csv_sha256_matches_record`: True
* `parquet_sha256_matches_record`: True
* `row_count_540`: True
* `ids_same_order`: True
* `ids_same_set`: True

The hashed-id fallback was not needed: reconstruction reproduced the recorded files byte-for-byte.

Reproduce (no GPU, no retraining):

```
kaggle competitions download -c name-that-llm -f train.csv -p /tmp/ntl
python make_confusion_matrix.py --train /tmp/ntl/train.csv
```

Run with pandas 3.1.1, numpy 2.5.4, matplotlib 3.11.2.
