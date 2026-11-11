"""Write manifest.csv from configs/*.json, docs/raw/runs/runs.json (saved versions + output checks of each S run) and each
run's engs108_run/run_report.json. Reference rows (A0, V1-V3) come from docs/prior_public_scores.json."""
import csv, json, pathlib
import build_notebooks as bn

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / 'docs' / 'raw' / 'runs'
IDS = ['S1', 'S2', 'S3', 'S4']
FIELDS = ['Experiment ID', 'Kaggle notebook', 'Saved version', 'Notebook URL', 'Base', 'Exact changes from base',
          'Native share', 'balance_logits strength', 'ATLAS margin', 'execution-contract sinkhorn_strength record',
          'Output checks', 'Runtime', 'Peak GPU memory', 'Prior public score (user-confirmed)', 'Competition submission',
          'Status']


def hms(s):
    return f'{int(s // 3600)}h{int(s % 3600 // 60):02d}m'


def gpu(r):
    return ' / '.join(f'GPU{k} {v / 1024:.1f} GiB' for k, v in sorted(r['peak_gpu_mem_mib'].items()))


def value(cfg, param, default):
    return next((c['modified_value'] for c in cfg['changes'] if c['parameter'] == param), default)


def main():
    runs = json.loads((RUNS / 'runs.json').read_text()) if (RUNS / 'runs.json').exists() else {}
    prior = json.loads((HERE / 'docs' / 'prior_public_scores.json').read_text())
    rows = []
    for exp in [k for k in prior if not k.startswith('_')]:
        p = prior[exp]
        rows.append({'Experiment ID': exp, 'Kaggle notebook': f"{bn.OWNER}/{p['slug']}", 'Saved version': f"v{p['saved_version']}",
                     'Notebook URL': f"https://www.kaggle.com/code/{bn.OWNER}/{p['slug']}", 'Base': 'Ref A' if exp == 'A0' else 'A0',
                     'Exact changes from base': p['change'], 'Prior public score (user-confirmed)': p['public_score'],
                     'Competition submission': 'not made by this work', 'Status': 'reference (submission/public-baselines)'})
    for exp in IDS:
        cfg = bn.load_config(exp)
        run = runs.get(exp, {})
        rep = None
        if run.get('dir'):
            rp = RUNS / run['dir'] / 'engs108_run' / 'run_report.json'
            rep = json.loads(rp.read_text()) if rp.exists() else None
        prov = [e for c in cfg['changes'] for e in c.get('provenance_edits', [])]
        rows.append({
            'Experiment ID': exp, 'Kaggle notebook': f"{bn.OWNER}/{cfg['slug']}",
            'Saved version': f"v{run['version']}" if run.get('version') else '',
            'Notebook URL': f"https://www.kaggle.com/code/{bn.OWNER}/{cfg['slug']}", 'Base': 'A0',
            'Exact changes from base': '; '.join(f"cell {c['cell']}: {c['old']} -> {c['new']}" for c in cfg['changes'] + prov),
            'Native share': value(cfg, "CONFIG['native_embedding']['share']", '0.5'),
            'balance_logits strength': value(cfg, 'core_math.balance_logits(strength=...)', '.30'),
            'ATLAS margin': value(cfg, 'ensemble.py ATLAS top-2 margin threshold', '.3'),
            'execution-contract sinkhorn_strength record': prov[0]['new'].split(':')[1].strip(' ,') if prov else '.30',
            'Output checks': (f"{run['output_checks']['rows']}-row test.csv: " +
                              ('PASS' if run['output_checks']['all_passed'] else 'FAIL')) if run.get('output_checks') else '',
            'Runtime': hms(rep['wall_seconds']) if rep else '',
            'Peak GPU memory': gpu(rep) if rep else '',
            'Prior public score (user-confirmed)': '',
            'Competition submission': 'none (not submitted)',
            'Status': run.get('status', 'not run'),
        })
    with open(HERE / 'manifest.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
