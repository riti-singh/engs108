"""Push build/<ID>/submit to Kaggle (Save & Run All, private, settings from kernel-metadata.json), retrying while the
account's 2-batch-GPU-session cap is hit, then poll each kernel independently and download its outputs when COMPLETE.
Never submits to the competition.

usage: KAGGLE_API_TOKEN=... python run_queue.py S1 S2 S3 S4 [--out DIR]
Logs: docs/raw/runs/push_queue.log, docs/raw/runs/status.log; state: docs/raw/runs/queue_state.json."""
import argparse, json, pathlib, re, subprocess, time
import build_notebooks as bn

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / 'docs' / 'raw' / 'runs'
STATE = RUNS / 'queue_state.json'


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def log(name, msg):
    with open(RUNS / name, 'a') as f:
        f.write(f'{now()} {msg}\n')
    print(now(), msg, flush=True)


def kaggle(*args):
    r = subprocess.run(['kaggle', *args], capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def main(ids, out):
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    for e in ids:
        state.setdefault(e, {'status': 'pending'})
    while any(state[e]['status'] not in ('COMPLETE', 'ERROR', 'CANCEL') for e in ids):
        for e in ids:
            s, slug = state[e], bn.load_config(e)['slug']
            if s['status'] == 'pending':
                msg = kaggle('kernels', 'push', '-p', str(HERE / 'build' / e / 'submit'))
                log('push_queue.log', f'build/{e}/submit :: {msg.splitlines()[-1] if msg else ""}')
                m = re.search(r'Kernel version (\d+) successfully pushed', msg)
                if m:
                    s.update(status='RUNNING', version=int(m.group(1)), pushed_at=now())
            elif s['status'] == 'RUNNING':
                msg = kaggle('kernels', 'status', f'{bn.OWNER}/{slug}')
                log('status.log', f'{slug} v{s["version"]} :: {msg}')
                m = re.search(r'KernelWorkerStatus\.(\w+)', msg)
                st = m.group(1) if m else ''
                if st in ('COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED', 'CANCEL_REQUESTED'):
                    d = pathlib.Path(out) / f'{e}_submit_v{s["version"]}'
                    d.mkdir(parents=True, exist_ok=True)
                    log('status.log', f'{slug} v{s["version"]} output :: ' +
                        kaggle('kernels', 'output', f'{bn.OWNER}/{slug}', '-p', str(d)).replace('\n', ' | ')[-300:])
                    s.update(status='COMPLETE' if st == 'COMPLETE' else ('ERROR' if st == 'ERROR' else 'CANCEL'),
                             finished_at=now(), output_dir=str(d))
            STATE.write_text(json.dumps(state, indent=1) + '\n')
            time.sleep(5)
        time.sleep(300)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('ids', nargs='+')
    ap.add_argument('--out', default=str(pathlib.Path.home() / 'engs108_round2_outputs'))
    a = ap.parse_args()
    main(a.ids, a.out)
