"""Check that notebooks pulled from Kaggle (`kaggle kernels pull ritisingh123/<slug> -p <dir>/<ID>`) match build/<ID>/submit
and that each production notebook reads the real competition test.csv with the source's small-input shortcuts disabled.

usage: python verify_kaggle_versions.py <pulled_dir> [IDs...]"""
import hashlib, json, pathlib, sys
import build_notebooks as bn

HERE = pathlib.Path(__file__).resolve().parent


def code(nb):
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']


def main(pulled, ids):
    ok, digests = True, {}
    for exp in ids:
        cfg = bn.load_config(exp)
        built = json.loads((HERE / 'build' / exp / 'submit' / f"{cfg['slug']}.ipynb").read_text())
        live = json.loads((pathlib.Path(pulled) / exp / f"{cfg['slug']}.ipynb").read_text())
        src = '\n'.join(code(live))
        checks = {
            'matches_build_submit': code(live) == code(built),
            'reads_competition_test_csv': bn.COMP_DIR in src and "pd.read_csv(config['paths']['test_csv']" in src
                                          if exp != 'B0' else bn.COMP_DIR in src,
            'no_validation_data_dir': bn.VAL_DIR not in src,
            'shortcut_disabled': ('preview = False' in src and 'emit_known_preview(DATA,None,OUTPUT)' not in src)
                                 if exp != 'B0' else ('FAST_COMMIT = False' in src and 'FAST_COMMIT = len(' not in src),
        }
        for ch in cfg.get('changes') or ([cfg['change']] if cfg.get('change') else []):
            checks[f"change_present_{ch['id']}"] = ch['new'] in src and ch['old'] not in src.replace(ch['new'], '')
        digests[exp] = hashlib.sha256(src.encode()).hexdigest()
        ok &= all(checks.values())
        print(exp, cfg['slug'], digests[exp][:12], json.dumps(checks))
    distinct = len(set(digests.values())) == len(digests)
    print('distinct_sources', distinct)
    return ok and distinct


if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1], sys.argv[2:] or ['S1', 'S2', 'S3', 'S4']) else 1)
