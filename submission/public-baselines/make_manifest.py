"""Write manifest.csv from configs/*.json, docs/raw/runs/runs.json and each run's engs108_run/run_report.json."""
import csv, json, pathlib

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / 'docs' / 'raw' / 'runs'
FIELDS = ['Experiment ID', 'Original source notebook', 'Original source version', 'Kaggle notebook slug', 'Saved version',
          'Exact change from baseline', 'Original parameter value', 'Modified parameter value', 'Attached asset versions',
          'Local validation result', 'Official leaderboard score', 'Runtime', 'Peak GPU memory', 'Status', 'Submission link']


def report(run):
    p = RUNS / run['dir'] / 'engs108_run' / 'run_report.json' if run and run.get('dir') else None
    return json.loads(p.read_text()) if p and p.exists() else None


def hms(s):
    return f'{int(s // 3600)}h{int(s % 3600 // 60):02d}m'


def gpu(r):
    return ' / '.join(f'GPU{k} {v / 1024:.1f} GiB' for k, v in sorted(r['peak_gpu_mem_mib'].items()))


def main():
    import build_notebooks as bn
    runs = json.loads((RUNS / 'runs.json').read_text())
    versions = json.loads((HERE / 'docs' / 'raw' / 'dataset_versions.json').read_text())
    rows = []
    for exp in ['A0', 'B0', 'V1', 'V2', 'V3']:
        if not (HERE / 'configs' / f'{exp}.json').exists():
            continue
        cfg = bn.load_config(exp)
        meta = json.loads((HERE / cfg['source']['metadata']).read_text())
        assets = '; '.join(f"{d} v{versions[d]['version']}" for d in meta['dataset_sources'])
        assets += '; kernels: ' + ', '.join(meta['kernel_sources']) + '; models: ' + ', '.join(meta['model_sources'])
        run = runs.get(exp, {})
        rv, rs = report(run.get('validate')), report(run.get('submit'))
        ch = cfg.get('change')
        val = 'not run'
        if rv:
            val = (f"validate v{run['validate']['version']}: {rv['rows']}-row synthetic test, contract checks "
                   f"{'PASS' if rv['all_checks_passed'] else 'FAIL'}, in-sample acc {rv['in_sample_accuracy_not_a_score']:.3f} (not a score)")
        if rs:
            val += f"; submit v{run['submit']['version']}: {rs['rows']}-row test.csv, checks {'PASS' if rs['all_checks_passed'] else 'FAIL'}"
        v = run.get('validate') or {}
        if not rv and v.get('error'):
            val = f"validate v{v['version']}: {v['status']} - {v['error']}"
        status = (run.get('submit') or {}).get('status') or (run.get('validate') or {}).get('status') or 'not run'
        rows.append({
            'Experiment ID': exp,
            'Original source notebook': f"https://www.kaggle.com/code/{cfg['source']['ref']}",
            'Original source version': cfg['source']['version'],
            'Kaggle notebook slug': f"{bn.OWNER}/{cfg['slug']}",
            'Saved version': ((f"submit v{run['submit']['version']}" if run.get('submit') else '') +
                              (f" (validate v{run['validate']['version']})" if run.get('validate') else '')).strip(),
            'Exact change from baseline': (f"{ch['parameter']}: {ch['what']}" if ch else
                                           'none (baseline); compat edits: ' + ', '.join(e['id'] for e in cfg['compat_edits'])
                                           + (', map_input_layout' if cfg.get('map_input_layout') else '')),
            'Original parameter value': ch['original_value'] if ch else '',
            'Modified parameter value': ch['modified_value'] if ch else '',
            'Attached asset versions': assets,
            'Local validation result': val,
            'Official leaderboard score': '',
            'Runtime': '; '.join(f"{m} {hms(r['wall_seconds'])}" for m, r in (('validate', rv), ('submit', rs)) if r)
                       or (f"validate ~{hms(v['wall_seconds_approx'])} until {v['status']}" if v.get('wall_seconds_approx') else ''),
            'Peak GPU memory': '; '.join(f"{m} {gpu(r)}" for m, r in (('validate', rv), ('submit', rs)) if r)
                               or (f"validate {gpu(v)}" if v.get('peak_gpu_mem_mib') else ''),
            'Status': status,
            'Submission link': f"https://www.kaggle.com/code/{bn.OWNER}/{cfg['slug']}",
        })
    with open(HERE / 'manifest.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
