"""Check a downloaded Kaggle output dir against the real competition test.csv.

usage: python verify_outputs.py <test.csv> <ID>=<output_dir> ...   (prints one JSON line per ID)"""
import json, pathlib, sys
import pandas as pd
import build_notebooks as bn


def check(test_csv, out_dir):
    test = pd.read_csv(test_csv, dtype={'id': str})
    sub = pd.read_parquet(pathlib.Path(out_dir) / 'submission.parquet')
    rep = json.loads((pathlib.Path(out_dir) / 'engs108_run' / 'run_report.json').read_text())
    checks = {
        'submission_parquet_exists': True,
        'columns_exactly_id_model': sub.columns.tolist() == ['id', 'model'],
        'row_count_matches_test': len(sub) == len(test),
        'ids_and_order_match_test': sub['id'].astype(str).tolist() == test['id'].tolist(),
        'ids_unique': bool(sub['id'].is_unique),
        'no_null_labels': bool(sub['model'].notna().all()),
        'labels_valid': bool(sub['model'].isin(bn.LABELS).all()),
        'kaggle_run_report_passed': rep['all_checks_passed'] and rep['mode'] == 'submit',
    }
    return {'rows': len(sub), 'checks': checks, 'all_passed': all(checks.values()),
            'wall_seconds': rep['wall_seconds'], 'peak_gpu_mem_mib': rep['peak_gpu_mem_mib'],
            'predictions': dict(zip(sub['id'].astype(str), sub['model']))}


if __name__ == '__main__':
    ok = True
    for arg in sys.argv[2:]:
        exp, d = arg.split('=', 1)
        r = check(sys.argv[1], d)
        ok &= r['all_passed']
        print(exp, json.dumps(r))
    sys.exit(0 if ok else 1)
