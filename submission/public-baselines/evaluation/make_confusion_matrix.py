"""Confusion matrix and per-class metrics for the V1 validation run (TRAINING-DERIVED, IN-SAMPLE diagnostic).

The 540 validation rows are 30 whole prompts of the competition train.csv, which the V1 pipeline also reads at
inference time, so these numbers are not a generalization estimate and not a leaderboard score.

Ground truth is not committed (train-derived). It is rebuilt here by re-running the selection cell of the V1
validate notebook against the competition train.csv, then checked three ways before anything is computed:
  1. sha256 of the rebuilt val_truth.csv / test.csv == the hashes recorded in docs/raw/runs/artifact_hashes.json;
  2. sha256 of the V1 submission.parquet == the recorded hash;
  3. parquet ids == rebuilt ids, in order and as a set.

usage: python make_confusion_matrix.py --train /path/to/name-that-llm/train.csv
       (or set NTL_TRAIN_CSV; download with `kaggle competitions download -c name-that-llm -f train.csv`)"""
import argparse, ast, csv, hashlib, json, os, pathlib, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT / 'docs' / 'raw' / 'runs'
RUN = 'V1_validate_v1'
NOTEBOOK = ROOT / 'build' / 'V1' / 'validate' / 'engs108-v1-a0-nativeshare040.ipynb'
PARQUET = RUNS / RUN / 'submission.parquet'
RUN_REPORT = RUNS / RUN / 'engs108_run' / 'run_report.json'
HASHES = RUNS / 'artifact_hashes.json'
DPI = 300


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def notebook_cells():
    return [''.join(c['source']) for c in json.loads(NOTEBOOK.read_text())['cells'] if c['cell_type'] == 'code']


def run_report_labels(cells):
    cell = next(s for s in cells if s.startswith('# engs108: submission contract check and run report'))
    labels = ast.literal_eval(re.search(r"^_e_labels = (\[.*\])$", cell, re.M).group(1))
    assert len(labels) == len(set(labels)) == 18, labels
    return labels


def selection_params(cells):
    cell = next(s for s in cells if s.startswith('# engs108: validation-only input'))
    pick = re.search(r"RandomState\((\d+)\)\.choice\(sorted\(_e_tr\.prompt_id\.unique\(\)\), (\d+), replace=False\)", cell)
    shuffle = re.search(r"\.sample\(frac=1\.0, random_state=(\d+)\)", cell)
    salt = re.search(r"blake2s\(\('([^']+)' \+ i\)\.encode\(\), digest_size=(\d+)\)", cell)
    assert pick and shuffle and salt, 'validation selection cell changed; update the reconstruction'
    return {'seed': int(pick.group(1)), 'n_prompts': int(pick.group(2)), 'shuffle_seed': int(shuffle.group(1)),
            'salt': salt.group(1), 'digest_size': int(salt.group(2))}


def rebuild_validation(train_csv, p):
    """Same operations, in the same order, as the validate notebook's selection cell."""
    tr = pd.read_csv(train_csv, dtype={'id': str})
    pick = np.random.RandomState(p['seed']).choice(sorted(tr.prompt_id.unique()), p['n_prompts'], replace=False)
    te = tr[tr.prompt_id.isin(pick)].sample(frac=1.0, random_state=p['shuffle_seed']).reset_index(drop=True)
    te['id'] = [hashlib.blake2s((p['salt'] + i).encode(), digest_size=p['digest_size']).hexdigest() for i in te.id]
    assert te.id.is_unique
    return te


def verify(te, sub):
    hashes = json.loads(HASHES.read_text())
    truth_b = te[['id', 'prompt_id', 'model']].to_csv(index=False).encode()
    test_b = te[['id', 'text']].to_csv(index=False).encode()
    rec_truth = hashes[f'{RUN}/engs108_run/val_truth.csv']
    rec_test = hashes['A0_v2/ntl_val_data/test.csv']  # only recorded copy of the shared validation test.csv
    checks = {
        'val_truth_sha256_matches_record': sha256(truth_b) == rec_truth['sha256'] and len(truth_b) == rec_truth['bytes'],
        'test_csv_sha256_matches_record': sha256(test_b) == rec_test['sha256'] and len(test_b) == rec_test['bytes'],
        'parquet_sha256_matches_record': sha256(PARQUET.read_bytes()) == hashes['_kept'][f'{RUN}/submission.parquet'],
        'row_count_540': len(te) == len(sub) == 540,
        'ids_same_order': sub['id'].astype(str).tolist() == te['id'].tolist(),
        'ids_same_set': set(sub['id'].astype(str)) == set(te['id']),
    }
    for k, v in checks.items():
        print(f'  {k}: {v}')
    assert all(checks.values()), checks
    return checks


def confusion(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


def class_metrics(cm, labels):
    tp = np.diag(cm).astype(float)
    pred, support = cm.sum(0).astype(float), cm.sum(1).astype(float)
    precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    return pd.DataFrame({'label': labels, 'precision': precision.round(4), 'recall': recall.round(4),
                         'f1': f1.round(4), 'support': support.astype(int), 'predicted': pred.astype(int),
                         'correct': tp.astype(int)})


def top_confusions(cm, labels, k=5):
    """Largest off-diagonal cells; ties broken by label order (true, then predicted)."""
    cells = [(int(cm[i, j]), i, j) for i in range(len(labels)) for j in range(len(labels)) if i != j and cm[i, j] > 0]
    cells.sort(key=lambda c: (-c[0], c[1], c[2]))
    support = cm.sum(1)
    return pd.DataFrame([{'rank': r + 1, 'true': labels[i], 'predicted': labels[j], 'count': n,
                          'share_of_true_row': round(n / support[i], 4)} for r, (n, i, j) in enumerate(cells[:k])]), len(cells)


def plot(matrix, labels, title, path, fmt, vmax):
    fig, ax = plt.subplots(figsize=(11, 9.5), facecolor='white')
    ax.set_facecolor('white')
    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=vmax)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = matrix[i, j]
            if v:
                ax.text(j, i, fmt(v), ha='center', va='center', fontsize=6.5,
                        color='white' if v > 0.6 * vmax else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor='white')
    plt.close(fig)


EVALUATION_MD = """# V1 validation confusion matrix: training-derived, in-sample diagnostic

**This is not a held-out evaluation and not a leaderboard estimate.** The 540 rows are every passage of
{n_prompts} prompts drawn from the competition `train.csv` (seed {seed}, ids re-hashed), and the V1 pipeline reads
`train.csv` at inference time (Ref A's native fusion, Sinkhorn balancing, typography guards and Gaussian heads;
see `docs/source_audit.md`; kept as-is by decision). So the pipeline has seen these exact passages and labels. Treat everything below as an in-sample diagnostic.

* No independent labeled data was used for this matrix: the only labeled data the competition provides is
  `train.csv`, and the hidden test labels are not public. A candidate independent set, the protocol needed to use it,
  and what is still missing are in `INDEPENDENT_EVAL.md`.
* A fresh random split of `train.csv` would still be contaminated: the pipeline reads the whole `train.csv` at
  inference regardless of how we split it, and its prefit public members were trained by their authors on the
  competition train set, which we cannot undo without retraining. No such split is reported anywhere here.

**Do not quote {acc:.1%} as validation or generalization performance.**

## Result (in-sample)

| | |
|---|---|
| Run | `{run}` (`ritisingh123/engs108-v1-a0-nativeshare040`, validate mode) |
| Rows | {rows} ({n_prompts} train prompts x 18 models) |
| In-sample accuracy | **{correct}/{rows}** ({acc:.4f}) |
| In-sample macro F1 / balanced accuracy | {macro_f1:.4f} / {bal_acc:.4f} |
| Off-diagonal (misclassified) rows | {errors}, spread over {n_offdiag} distinct (true, predicted) cells |

The {correct}/{rows} figure matches `in_sample_accuracy_not_a_score` = {report_acc} in
`docs/raw/runs/{run}/engs108_run/run_report.json`.

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
The 0.680 and the in-sample {correct}/{rows} measure different things and must not be combined or compared as if
they were the same kind of number.

## Top confusions (true -> predicted)

{top_table}

Ties are broken by the 18-label order (true, then predicted).{tie_note}

## Files

* `make_confusion_matrix.py`: rebuilds ground truth from `train.csv`, verifies alignment, writes all outputs.
* `confusion_matrix_normalized.png`: row-normalized (each true-label row sums to 1), {dpi} DPI.
* `confusion_matrix_counts.png`: raw counts, {dpi} DPI.
* `class_metrics.csv`: per-class precision, recall, F1, support (plus predicted count and correct count).
* `top_confusions.csv`: the five largest off-diagonal cells.
* `INDEPENDENT_EVAL.md`: pipeline leakage audit, independent-data feasibility, and what is missing (hand-written).

Label order on both axes is the 18-label list in the run-report cell of
`build/V1/validate/engs108-v1-a0-nativeshare040.ipynb`.

## Ground-truth reconstruction and alignment

Ground truth (`val_truth.csv`) and the validation `test.csv` are train-derived and were removed from the repo;
only their hashes are kept in `docs/raw/runs/artifact_hashes.json`. The script re-runs the notebook's selection cell
(parameters parsed from the notebook: `RandomState({seed}).choice(sorted(prompt_id.unique()), {n_prompts})`, row shuffle
`random_state={shuffle_seed}`, ids `blake2s('{salt}' + id, digest_size={digest_size})`) on the competition `train.csv`
and refuses to continue unless every check passes:

{checks}

The hashed-id fallback was not needed: reconstruction reproduced the recorded files byte-for-byte.

Reproduce (no GPU, no retraining):

```
kaggle competitions download -c name-that-llm -f train.csv -p /tmp/ntl
python make_confusion_matrix.py --train /tmp/ntl/train.csv
```

Run with pandas {pandas}, numpy {numpy}, matplotlib {matplotlib}.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default=os.environ.get('NTL_TRAIN_CSV'), help='competition train.csv')
    ap.add_argument('--out', default=str(HERE))
    args = ap.parse_args()
    assert args.train, 'pass --train or set NTL_TRAIN_CSV'
    out = pathlib.Path(args.out)

    cells = notebook_cells()
    labels = run_report_labels(cells)
    params = selection_params(cells)
    te = rebuild_validation(args.train, params)
    sub = pd.read_parquet(PARQUET)
    print('alignment checks:')
    checks = verify(te, sub)

    y_true, y_pred = te['model'].tolist(), sub['model'].astype(str).tolist()
    assert set(y_true) <= set(labels) and set(y_pred) <= set(labels)
    cm = confusion(y_true, y_pred, labels)
    rows, correct = int(cm.sum()), int(np.trace(cm))
    report_acc = json.loads(RUN_REPORT.read_text())['in_sample_accuracy_not_a_score']
    assert abs(correct / rows - report_acc) < 1e-12, (correct, report_acc)

    norm = cm / cm.sum(1, keepdims=True)
    plot(norm, labels, f'V1 validation, row-normalized (IN-SAMPLE: rows from train.csv), {correct}/{rows} correct',
         out / 'confusion_matrix_normalized.png', lambda v: f'{v:.2f}', 1.0)
    plot(cm, labels, f'V1 validation, counts (IN-SAMPLE: rows from train.csv), {correct}/{rows} correct',
         out / 'confusion_matrix_counts.png', lambda v: str(int(v)), int(cm.max()))

    metrics = class_metrics(cm, labels)
    metrics.to_csv(out / 'class_metrics.csv', index=False)
    top, n_offdiag = top_confusions(cm, labels)
    top.to_csv(out / 'top_confusions.csv', index=False)

    print(f'in-sample accuracy: {correct}/{rows} = {correct / rows:.4f} (training-derived; not a score)')
    print('top 5 off-diagonal (true -> predicted):')
    for r in top.itertuples():
        print(f'  {r.rank}. {r.true} -> {r.predicted}: {r.count}')

    tied = int((cm[~np.eye(len(labels), dtype=bool)] == top['count'].iloc[-1]).sum()) if len(top) else 0
    in_top = int((top['count'] == top['count'].iloc[-1]).sum()) if len(top) else 0
    tie_note = (f' {tied} off-diagonal cells share the fifth-place count of {top["count"].iloc[-1]}, so only {in_top} of'
                f' them are listed; the ranking among tied cells carries no information.') if tied > in_top else ''
    top_table = '| # | true | predicted | count | share of true row |\n|---|---|---|---|---|\n' + '\n'.join(
        f'| {r.rank} | {r.true} | {r.predicted} | {r.count} | {r.share_of_true_row:.4f} |' for r in top.itertuples())
    (out / 'EVALUATION.md').write_text(EVALUATION_MD.format(
        run=RUN, rows=rows, correct=correct, acc=correct / rows, report_acc=report_acc, errors=rows - correct,
        n_offdiag=n_offdiag, macro_f1=metrics.f1.mean(), bal_acc=metrics.recall.mean(), top_table=top_table,
        tie_note=tie_note, dpi=DPI, checks='\n'.join(f'* `{k}`: {v}' for k, v in checks.items()),
        pandas=pd.__version__, numpy=np.__version__, matplotlib=matplotlib.__version__, **params))


if __name__ == '__main__':
    main()
