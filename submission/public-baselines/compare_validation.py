"""Exact changed-prediction counts of V1/V2/V3 vs A0 on the shared 540-row synthetic validation test.
IN-SAMPLE ONLY: the validation rows come from train.csv, which the pipeline also reads; this is not a generalization benchmark.

usage: python compare_validation.py"""
import json, pathlib
import pandas as pd

RUNS = pathlib.Path(__file__).resolve().parent / 'docs' / 'raw' / 'runs'
RUN_DIRS = {'A0': 'A0_validate_v3', 'V1': 'V1_validate_v1', 'V2': 'V2_validate_v1', 'V3': 'V3_validate_v1'}


def main():
    base = pd.read_parquet(RUNS / RUN_DIRS['A0'] / 'submission.parquet')
    out = {}
    for exp in ['V1', 'V2', 'V3']:
        var = pd.read_parquet(RUNS / RUN_DIRS[exp] / 'submission.parquet')
        assert len(var) == len(base) == 540 and (var.id.values == base.id.values).all()
        diff = base.model.values != var.model.values
        out[exp] = {'rows': len(base), 'changed': int(diff.sum()),
                    'changed_rows': [{'id': i, 'A0': a, exp: b} for i, a, b in
                                     zip(base.id[diff], base.model[diff], var.model[diff])]}
        print(exp, 'vs A0:', out[exp]['changed'], '/', len(base), 'predictions changed')
    return out


if __name__ == '__main__':
    print(json.dumps(main(), indent=1))
