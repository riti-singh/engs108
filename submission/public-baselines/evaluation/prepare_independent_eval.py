"""Build (and later score) an independent evaluation set for the frozen V1 pipeline. No GPU, no retraining.

Source: nischaydnk/ntl-topic-synthetic-18llms (Kaggle, CC0), first published 2026-11-11 00:31 UTC, after the last
version of every dataset V1 mounts (latest: crtproduction/name-that-llm-open-pool v16, 2026-11-09). See
INDEPENDENT_EVAL.md for why it qualifies, what it does not establish, and what is still missing.

  prepare: verify the source file hash, check overlap against the competition train.csv (and optionally the Open
           Pool), keep only complete 18-model panels (one topic x one instruction answered by all 18 labels), and
           write test.csv (id,text: the only file the pipeline sees) and truth.csv (labels, kept apart).
  score:   given V1's submission.parquet for that test.csv, write the confusion matrices and per-class metrics.

usage:
  kaggle datasets download nischaydnk/ntl-topic-synthetic-18llms -p /tmp/indep --unzip
  python prepare_independent_eval.py prepare --generations /tmp/indep/generations.parquet \
      --train /tmp/ntl/train.csv [--pool ntl_open_pool.parquet] --out /tmp/indep_eval
  python prepare_independent_eval.py score --truth /tmp/indep_eval/truth.csv --submission submission.parquet"""
import argparse, collections, hashlib, pathlib, re
import numpy as np
import pandas as pd
from make_confusion_matrix import HERE, class_metrics, confusion, notebook_cells, plot, run_report_labels, sha256, \
    top_confusions

GENERATIONS_SHA256 = '265293398f3231032bc8afef4f0bf0b445a0f2b89c7adcc101e63ea568114753'
SALT, SEED = 'engs108-indep:', 108


def norm(s):
    return re.sub(r'\s+', ' ', str(s).lower()).strip()


def grams(s, n=8):
    w = re.findall(r"[a-z0-9']+", str(s).lower())
    return {' '.join(w[i:i + n]) for i in range(len(w) - n + 1)}


def prepare(args, labels):
    raw = pathlib.Path(args.generations).read_bytes()
    assert sha256(raw) == GENERATIONS_SHA256, 'source file differs from the audited version'
    g = pd.read_parquet(args.generations)
    assert g.key.is_unique and not g.text.map(norm).duplicated().any()
    assert set(g.model) == set(labels), sorted(set(labels) ^ set(g.model))
    train = pd.read_csv(args.train)
    checks = {'exact_text_matches_in_train_csv': int(g.text.map(norm).isin(set(train.text.map(norm))).sum())}
    train_grams = set().union(*map(grams, train.text))
    share = np.array([len(x & train_grams) / max(len(x), 1) for x in map(grams, g.text)])
    checks['max_share_of_8grams_seen_in_train_csv'] = round(float(share.max()), 4)
    if args.pool:
        pool = pd.read_parquet(args.pool, columns=['text'])
        checks['pool_rows_checked'] = len(pool)
        checks['exact_text_matches_in_pool'] = int(g.text.map(norm).isin(set(pool.text.map(norm))).sum())
    assert checks['exact_text_matches_in_train_csv'] == 0 and checks.get('exact_text_matches_in_pool', 0) == 0, checks

    width = g.groupby(['topic_id', 'version']).model.transform('nunique')
    te = g[width == len(labels)].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    te['id'] = [hashlib.blake2s((SALT + k).encode(), digest_size=6).hexdigest() for k in te.key]
    assert te.id.is_unique and (te.model.value_counts() == te.model.value_counts().iloc[0]).all()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    te[['id', 'text']].to_csv(out / 'test.csv', index=False)
    te[['id', 'topic_id', 'version', 'model']].to_csv(out / 'truth.csv', index=False)
    checks.update(source_rows=len(g), panels=int(te.groupby(['topic_id', 'version']).ngroups), rows=len(te),
                  rows_per_label=int(te.model.value_counts().iloc[0]),
                  panels_by_version=te.groupby('version').topic_id.nunique().to_dict())
    for k, v in checks.items():
        print(f'{k}: {v}')
    for name in ('test.csv', 'truth.csv'):
        b = (out / name).read_bytes()
        print(f'{name}: {len(b)} bytes, sha256 {sha256(b)}')


def score(args, labels):
    truth = pd.read_csv(args.truth, dtype={'id': str})
    sub = pd.read_parquet(args.submission)
    assert sub['id'].astype(str).tolist() == truth['id'].tolist(), 'submission ids do not match truth ids'
    y_true, y_pred = truth.model.tolist(), sub.model.astype(str).tolist()
    assert set(y_pred) <= set(labels)
    cm = confusion(y_true, y_pred, labels)
    rows, correct = int(cm.sum()), int(np.trace(cm))
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f'V1 on independent set (nischaydnk 18-model panels), {correct}/{rows} correct'
    plot(cm / cm.sum(1, keepdims=True), labels, tag + ', row-normalized', out / 'confusion_matrix_normalized.png',
         lambda v: f'{v:.2f}', 1.0)
    plot(cm, labels, tag + ', counts', out / 'confusion_matrix_counts.png', lambda v: str(int(v)), int(cm.max()))
    metrics = class_metrics(cm, labels)
    metrics.to_csv(out / 'class_metrics.csv', index=False)
    top_confusions(cm, labels)[0].to_csv(out / 'top_confusions.csv', index=False)
    by_version = collections.Counter(v for v, t, p in zip(truth.version, y_true, y_pred) if t == p)
    print(f'accuracy {correct}/{rows} = {correct / rows:.4f}; macro F1 {metrics.f1.mean():.4f}')
    print('correct by instruction length:', {v: f'{by_version[v]}/{n}' for v, n in truth.version.value_counts().items()})


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('prepare')
    p.add_argument('--generations', required=True)
    p.add_argument('--train', required=True)
    p.add_argument('--pool')
    p.add_argument('--out', required=True)
    s = sp.add_parser('score')
    s.add_argument('--truth', required=True)
    s.add_argument('--submission', required=True)
    s.add_argument('--out', default=str(HERE / 'independent'))
    args = ap.parse_args()
    labels = run_report_labels(notebook_cells())
    (prepare if args.cmd == 'prepare' else score)(args, labels)


if __name__ == '__main__':
    main()
