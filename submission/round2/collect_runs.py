"""Copy the committed evidence of each finished run from run_queue.py's download dir into docs/raw/runs/ and write runs.json.
Kept per run: submission.parquet, engs108_run/run_report.json, inference-logs/execution-contract.json. Kernel logs and GPU
telemetry are not committed; their sha256/size go into artifact_hashes.json.

usage: python collect_runs.py <real test.csv>"""
import hashlib, json, pathlib, shutil, sys
import verify_outputs

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / 'docs' / 'raw' / 'runs'
KEEP = ['submission.parquet', 'engs108_run/run_report.json', 'inference-logs/execution-contract.json']


def main(test_csv):
    state = json.loads((RUNS / 'queue_state.json').read_text())
    runs, hashes = {}, {}
    for exp, s in state.items():
        if s['status'] != 'COMPLETE':
            runs[exp] = {k: v for k, v in s.items() if k != 'output_dir'}
            continue
        src, name = pathlib.Path(s['output_dir']), f"{exp}_submit_v{s['version']}"
        for rel in KEEP:
            (RUNS / name / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / rel, RUNS / name / rel)
        hashes[name] = {str(p.relative_to(src)): {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
                        for p in sorted(src.rglob('*')) if p.is_file()}
        chk = verify_outputs.check(test_csv, src)
        contract = json.loads((src / 'inference-logs' / 'execution-contract.json').read_text())
        runs[exp] = {'version': s['version'], 'dir': name, 'status': 'COMPLETE', 'pushed_at': s['pushed_at'],
                     'finished_at': s['finished_at'], 'output_checks': chk,
                     'contract': {k: contract.get(k) for k in ['stages', 'native_embedding_share', 'sinkhorn_strength',
                                                               'complementary_share', 'gsq_share']}}
    (RUNS / 'runs.json').write_text(json.dumps(runs, indent=1) + '\n')
    (RUNS / 'artifact_hashes.json').write_text(json.dumps(hashes, indent=1) + '\n')
    for exp, r in runs.items():
        print(exp, r['status'], r.get('output_checks', {}).get('all_passed'))


if __name__ == '__main__':
    main(sys.argv[1])
