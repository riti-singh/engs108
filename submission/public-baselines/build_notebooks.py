"""Build the Kaggle push folders for the public-baseline experiments.

Each experiment config (configs/<ID>.json) names a pulled source notebook, the
compatibility edits of its baseline and (for variants) exactly one parameter
change.  Every edit is an exact string replacement that must match once.

Two modes are generated per experiment:
  submit    - the notebook to submit; reads the real competition test.csv.
  validate  - identical code, except the competition directory literal is
              replaced by a synthetic data dir (real train.csv, a held-out-style
              test.csv of whole train prompts) built by a prepended cell.
"""
import argparse, copy, json, pathlib, re, sys

HERE = pathlib.Path(__file__).resolve().parent
COMP_DIR = '/kaggle/input/competitions/name-that-llm'
VAL_DIR = '/kaggle/working/ntl_val_data'
RUN_DIR = '/kaggle/working/engs108_run'
OWNER = 'ritisingh123'
LABELS = ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'glm-5.3',
          'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'kimi-k3',
          'qwen3.8-max']


# Literals that are only written to /kaggle/working/inference-logs/execution-contract.json and never read back.
PROVENANCE_ONLY = {"'sinkhorn_strength': .30,"}


def load_config(exp_id):
    cfg = json.loads((HERE / 'configs' / f'{exp_id}.json').read_text())
    if cfg.get('base'):
        base = load_config(cfg['base'])
        assert not base.get('base'), 'variants must derive from an unchanged baseline'
        merged = copy.deepcopy(base)
        merged.update({k: v for k, v in cfg.items() if k != 'base'})
        merged['base'] = cfg['base']
        merged['compat_edits'] = base['compat_edits']
        return merged
    return cfg


def apply_edit(cells, edit):
    src = cells[edit['cell']]['source']
    n = src.count(edit['old'])
    assert n == 1, f"edit {edit.get('id', edit['old'][:60])!r} matched {n} times in cell {edit['cell']}"
    cells[edit['cell']]['source'] = src.replace(edit['old'], edit['new'])


def layout_cell(src_meta):
    """Kaggle mounts inputs either flat (/kaggle/input/<slug>) or nested (/kaggle/input/{competitions,datasets,notebooks}/...),
    and the layout differed between our GPU and CPU sessions. Rewrite input paths in every later cell (incl. %%writefile bodies)
    to whichever layout is mounted."""
    refs = [('datasets', r) for r in src_meta['dataset_sources']] + [('notebooks', r) for r in src_meta['kernel_sources']]
    return f'''# engs108 compat: map /kaggle/input paths to the mounted layout (environment fix only)
import os as _l_os, re as _l_re
_L_REFS = {refs!r}
_L_B = r"(?![A-Za-z0-9_.-])"
if _l_os.path.isdir('/kaggle/input/competitions/name-that-llm'):
    _L_RULES = [(_l_re.compile('/kaggle/input/' + _l_re.escape(r.split('/')[1]) + _L_B), '/kaggle/input/%s/%s' % (k, r)) for k, r in _L_REFS]
    _L_RULES.append((_l_re.compile('/kaggle/input/name-that-llm' + _L_B), '/kaggle/input/competitions/name-that-llm'))
    _L_LAYOUT = 'nested'
else:
    _L_RULES = [(_l_re.compile('/kaggle/input/%s/%s' % (k, _l_re.escape(r)) + _L_B), '/kaggle/input/' + r.split('/')[1]) for k, r in _L_REFS]
    _L_RULES.append((_l_re.compile('/kaggle/input/competitions/name-that-llm' + _L_B), '/kaggle/input/name-that-llm'))
    _L_LAYOUT = 'flat'
def _l_map(lines):
    src = ''.join(lines)
    for pat, rep in _L_RULES:
        src = pat.sub(rep, src)
    return src.splitlines(keepends=True)
get_ipython().input_transformers_cleanup.append(_l_map)
print('engs108 input layout:', _L_LAYOUT, flush=True)
'''


def monitor_cell(exp_id, mode):
    return f'''# engs108: run monitor (not part of the source). Samples GPU memory every 10 s with nvidia-smi
# so that peak usage of the source's subprocesses is captured.
import os as _e_os, subprocess as _e_sp, time as _e_time, json as _e_json
_e_os.makedirs({RUN_DIR!r}, exist_ok=True)
ENGS108_T0 = _e_time.time()
_e_log = open({RUN_DIR!r} + '/gpu_mem.csv', 'w')
ENGS108_SMI = _e_sp.Popen(['nvidia-smi', '--query-gpu=timestamp,index,memory.used,memory.total',
                           '--format=csv,noheader,nounits', '-l', '10'], stdout=_e_log, stderr=_e_sp.DEVNULL)
print('engs108 experiment {exp_id} mode {mode}', flush=True)
import glob as _e_glob
print('engs108 mounts:', sorted(_e_glob.glob('/kaggle/input/*')), sorted(_e_glob.glob('/kaggle/input/competitions/*')),
      sorted(_e_glob.glob('/kaggle/input/*/name-that-llm*')), flush=True)
'''


def validation_cell(cfg):
    v = cfg['validation']
    return f'''# engs108: validation-only input (not part of the source; absent from the submit notebook).
# Builds a full-size synthetic test.csv from {v['n_prompts']} whole train prompts (all of their passages, shuffled,
# new ids) so every size-dependent branch of the pipeline runs as on the hidden test. Rows come from train.csv,
# so any accuracy printed below is in-sample and is NOT an estimate of the leaderboard score.
import hashlib as _e_h, os as _e_os, numpy as _e_np, pandas as _e_pd
_e_src = next(p for p in ['/kaggle/input/competitions/' + 'name-that-llm', '/kaggle/input/' + 'name-that-llm'] if _e_os.path.isfile(p + '/train.csv'))
_e_os.makedirs({VAL_DIR!r}, exist_ok=True)
for _e_f in ['train.csv', 'sample_submission.parquet']:
    _e_dst = {VAL_DIR!r} + '/' + _e_f
    if not _e_os.path.lexists(_e_dst): _e_os.symlink(_e_src + '/' + _e_f, _e_dst)
_e_tr = _e_pd.read_csv(_e_src + '/train.csv', dtype={{'id': str}})
_e_pick = _e_np.random.RandomState({v['seed']}).choice(sorted(_e_tr.prompt_id.unique()), {v['n_prompts']}, replace=False)
_e_te = _e_tr[_e_tr.prompt_id.isin(_e_pick)].sample(frac=1.0, random_state={v['seed']}).reset_index(drop=True)
_e_te['id'] = [_e_h.blake2s(('engs108-val:' + i).encode(), digest_size=6).hexdigest() for i in _e_te.id]
assert _e_te.id.is_unique
_e_te[['id', 'text']].to_csv({VAL_DIR!r} + '/test.csv', index=False)
_e_te[['id', 'prompt_id', 'model']].to_csv({RUN_DIR!r} + '/val_truth.csv', index=False)
print('engs108 validation test.csv:', len(_e_te), 'rows from', len(_e_pick), 'train prompts', flush=True)
'''


def report_cell(cfg, mode):
    data_dir = VAL_DIR if mode == 'validate' else COMP_DIR
    return f'''# engs108: submission contract check and run report (not part of the source).
import json as _e_json, os as _e_os, time as _e_time, pandas as _e_pd
ENGS108_SMI.terminate(); _e_log.close()
_e_labels = {LABELS!r}
_e_test = _e_pd.read_csv({data_dir!r} + '/test.csv', dtype={{'id': str}})
_e_sub = _e_pd.read_parquet('/kaggle/working/submission.parquet')
_e_checks = {{
    'columns_exactly_id_model': _e_sub.columns.tolist() == ['id', 'model'],
    'row_count_matches_test': len(_e_sub) == len(_e_test),
    'ids_and_order_match_test': _e_sub['id'].astype(str).tolist() == _e_test['id'].tolist(),
    'ids_unique': bool(_e_sub['id'].is_unique),
    'no_null_labels': bool(_e_sub['model'].notna().all()),
    'labels_in_18_valid': bool(_e_sub['model'].isin(_e_labels).all()),
}}
_e_peak = {{}}
for _e_line in open({RUN_DIR!r} + '/gpu_mem.csv'):
    _e_p = [x.strip() for x in _e_line.split(',')]
    if len(_e_p) == 4 and _e_p[2].isdigit(): _e_peak[_e_p[1]] = max(_e_peak.get(_e_p[1], 0), int(_e_p[2]))
_e_report = {{'experiment': {cfg['experiment_id']!r}, 'mode': {mode!r}, 'config_version': {cfg['config_version']!r},
             'rows': len(_e_test), 'checks': _e_checks, 'all_checks_passed': all(_e_checks.values()),
             'wall_seconds': round(_e_time.time() - ENGS108_T0, 1), 'peak_gpu_mem_mib': _e_peak,
             'label_counts': _e_sub['model'].value_counts().to_dict()}}
if {mode!r} == 'validate':
    _e_truth = _e_pd.read_csv({RUN_DIR!r} + '/val_truth.csv', dtype={{'id': str}})
    _e_report['in_sample_accuracy_not_a_score'] = float((_e_sub['model'].values == _e_truth['model'].values).mean())
_e_json.dump(_e_report, open({RUN_DIR!r} + '/run_report.json', 'w'), indent=1)
print('ENGS108_RUN_REPORT', _e_json.dumps(_e_report), flush=True)
assert _e_report['all_checks_passed'], _e_checks
'''


def code_cell(src):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': src}


def build(exp_id, mode):
    cfg = load_config(exp_id)
    nb = json.loads((HERE / cfg['source']['file']).read_text())
    cells = nb['cells']
    for c in cells:
        c['source'] = ''.join(c['source'])
        if c['cell_type'] == 'code':
            c['outputs'], c['execution_count'] = [], None
    for edit in cfg['compat_edits']:
        apply_edit(cells, edit)
    if cfg.get('change'):
        apply_edit(cells, cfg['change'])
        for edit in cfg['change'].get('provenance_edits', []):
            assert edit['old'] in PROVENANCE_ONLY, 'provenance edits may only touch execution-contract.json records'
            apply_edit(cells, edit)
    src_meta = json.loads((HERE / cfg['source']['metadata']).read_text())

    if mode == 'validate':
        for c in cells:
            if c['cell_type'] == 'code':
                c['source'] = c['source'].replace(COMP_DIR, VAL_DIR)
    head = [code_cell(monitor_cell(exp_id, mode))]
    if cfg.get('map_input_layout'):
        head.append(code_cell(layout_cell(src_meta)))
    if mode == 'validate':
        head.append(code_cell(validation_cell(cfg)))
    nb['cells'] = head + cells + [code_cell(report_cell(cfg, mode))]
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            c['source'] = c['source']
    slug = cfg['slug']
    meta = {
        'id': f'{OWNER}/{slug}', 'title': slug, 'code_file': f'{slug}.ipynb', 'language': 'python',
        'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
        'enable_internet': False, 'competition_sources': src_meta['competition_sources'],
        'dataset_sources': src_meta['dataset_sources'], 'kernel_sources': src_meta['kernel_sources'],
        'model_sources': src_meta['model_sources'], 'machine_shape': src_meta['machine_shape'],
        'docker_image_pinning_type': 'original', 'docker_image': src_meta['docker_image'], 'keywords': [],
    }
    out = HERE / 'build' / exp_id / mode
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{slug}.ipynb').write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=1) + '\n')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('experiments', nargs='+')
    ap.add_argument('--mode', choices=['submit', 'validate', 'both'], default='both')
    a = ap.parse_args()
    for e in a.experiments:
        for m in (['submit', 'validate'] if a.mode == 'both' else [a.mode]):
            print(build(e, m))
