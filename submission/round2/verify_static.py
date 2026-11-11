"""Static checks of build/<ID>/submit against Ref A and the submit contract (no Kaggle access needed).

usage: python verify_static.py [IDs...]   (default S1 S2 S3 S4)"""
import difflib, json, pathlib, sys
import build_notebooks as bn

HERE = pathlib.Path(__file__).resolve().parent
SOURCE_KEYS = ['dataset_sources', 'kernel_sources', 'model_sources', 'competition_sources']


def code(nb):
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']


def changes(cfg):
    return cfg.get('changes') or ([cfg['change']] if cfg.get('change') else [])


def main(ids):
    ok, seen = True, {'id': set(), 'title': set(), 'code_file': set()}
    for exp in ids:
        cfg = bn.load_config(exp)
        ref = json.loads((HERE / cfg['source']['metadata']).read_text())
        refnb = json.loads((HERE / cfg['source']['file']).read_text())
        d = HERE / 'build' / exp / 'submit'
        meta = json.loads((d / 'kernel-metadata.json').read_text())
        nb = json.loads((d / meta['code_file']).read_text())
        cells = code(nb)
        src = '\n'.join(cells)
        report = cells[-1]
        checks = {f'unique_{k}': meta[k] not in seen[k] for k in seen}
        for k in seen:
            seen[k].add(meta[k])
        checks.update({
            'id_is_slug': meta['id'] == f"{bn.OWNER}/{cfg['slug']}" and (d / f"{cfg['slug']}.ipynb").exists(),
            'private': meta['is_private'] is True,
            'gpu_on_internet_off': meta['enable_gpu'] is True and meta['enable_internet'] is False,
            'machine_shape_and_docker_match_ref': meta['machine_shape'] == ref['machine_shape'] and meta['docker_image'] == ref['docker_image'],
            **{f'{k}_preserved': meta[k] == ref[k] for k in SOURCE_KEYS},
            'reads_competition_test_csv': bn.COMP_DIR in src and "pd.read_csv(config['paths']['test_csv']" in src,
            'layout_mapper_present': 'input_transformers_cleanup.append(_l_map)' in src,
            'no_validation_data_dir': bn.VAL_DIR not in src,
            'preview_shortcut_disabled': 'preview = False' in src and 'emit_known_preview(DATA,None,OUTPUT)' not in src,
            'writes_submission_parquet': "/kaggle/working/submission.parquet" in src,
            'report_requires_exactly_id_model': "_e_sub.columns.tolist() == ['id', 'model']" in report
                                                and "assert _e_report['all_checks_passed']" in report,
            'ref_markdown_and_header_cells_verbatim': [''.join(c['source']) for c in refnb['cells'] if c['cell_type'] == 'markdown']
                                                      == [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'markdown'],
        })
        # Every changed line vs Ref A must be one of the configured edits (compat, changes, provenance) -- nothing else.
        ref_cells = [''.join(c['source']) for c in refnb['cells'] if c['cell_type'] == 'code']
        body = cells[2:-1] if cfg.get('map_input_layout') else cells[1:-1]
        edits = cfg['compat_edits'] + [e for ch in changes(cfg) for e in [ch] + ch.get('provenance_edits', [])]
        expected = 0
        unexplained = []
        for a, b in zip(ref_cells, body):
            if a == b:
                continue
            for line in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm='', n=0):
                if line.startswith(('---', '+++', '@@')):
                    continue
                if line.startswith('-'):
                    expected += 1
                    if not any(e['old'] in line for e in edits): unexplained.append(line)
                elif line.startswith('+') and not any(e['new'] in line for e in edits):
                    unexplained.append(line)
        checks['same_code_cell_count'] = len(ref_cells) == len(body)
        checks['only_configured_edits'] = not unexplained
        checks['each_edit_applied_once'] = all(src.count(e['new']) >= 1 for e in edits) and expected == len(edits)
        ok &= all(checks.values())
        bad = [k for k, v in checks.items() if not v]
        print(exp, meta['id'], 'PASS' if not bad else f'FAIL {bad}', unexplained[:5])
    return ok


if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1:] or ['S1', 'S2', 'S3', 'S4']) else 1)
