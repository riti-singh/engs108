"""Builds the 4,000-row inference simulation kernel from the generated H5 submission notebook.

  python submission/h5/sim4k/build_sim.py     # writes rs2027-engs108-h5-sim4k.ipynb + kernel-metadata.json

The simulation runs the submission notebook's own code cells unchanged (real Qwen3-8B 4-bit, Arm H adapter, probe, stylo;
no stubs). The only patch is INPUT_ROOT, so DATA_DIR discovery resolves a synthetic competition mount holding a 4,000-row
test.csv built from train.csv passages, including contiguous full batches of texts longer than MAX_LENGTH tokens.
Any exception is caught and recorded in h5_run_summary.json together with the per-batch log and peak GPU memory.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
H5 = HERE.parent
sys.path.insert(0, str(H5))
import build_notebook as B  # noqa: E402

NB_NAME = "rs2027-engs108-h5-sim4k.ipynb"
KERNEL = "ritisingh123/rs2027-engs108-h5-sim4k"
SIM_ROOT = "/kaggle/working/sim_input"

SIM_DATA = r'''import glob, os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

N_ROWS, N_MAXLEN_BLOCK, N_MAXLEN_SPREAD, N_EMPTY = 4000, 64, 136, 8
train_csv = sorted(glob.glob("/kaggle/input/competitions/*/train.csv") + glob.glob("/kaggle/input/*/train.csv"))[0]
real_sample = os.path.join(os.path.dirname(train_csv), "sample_submission.parquet")
train = pd.read_csv(train_csv)
rng = np.random.default_rng(0)
pool = train["text"].astype(str).to_numpy()
# max-length passages: 6 concatenated train texts (about 4,500 chars, far above 512 tokens)
maxlen = ["\n\n".join(rng.choice(pool, 6, replace=False)) for _ in range(N_MAXLEN_BLOCK + N_MAXLEN_SPREAD)]
regular = list(rng.choice(pool, N_ROWS - len(maxlen) - N_EMPTY))
spread = regular + maxlen[N_MAXLEN_BLOCK:] + [""] * (N_EMPTY // 2) + [None] * (N_EMPTY // 2)
spread = [spread[i] for i in rng.permutation(len(spread))]
mid = len(spread) // 2
texts = spread[:mid] + maxlen[:N_MAXLEN_BLOCK] + spread[mid:]  # 8 contiguous batches of 8 max-length texts
ids = [f"{i:012x}" for i in rng.choice(16 ** 10, N_ROWS, replace=False)]
SIM_DIR = "__SIM_ROOT__/competitions/name-that-llm"
os.makedirs(SIM_DIR, exist_ok=True)
pd.DataFrame({"id": ids, "text": texts}).to_csv(f"{SIM_DIR}/test.csv", index=False)
schema = pq.read_schema(real_sample)
pq.write_table(pa.table({"id": ids, "model": ["gpt-5.5"] * N_ROWS}).cast(schema), f"{SIM_DIR}/sample_submission.parquet")
SIM_INFO = {"rows": N_ROWS, "maxlen_rows": len(maxlen), "maxlen_contiguous_block_rows": N_MAXLEN_BLOCK,
            "maxlen_block_start_row": mid, "empty_or_null_rows": N_EMPTY, "source": train_csv,
            "text_chars": pd.Series([len(t or "") for t in texts]).describe().to_dict()}
print(SIM_INFO)
'''

RUNNER = r'''import json, time, traceback

CELLS = json.loads(r"""__CELLS__""")
sim_ns = {"__name__": "__main__"}
sim_error, sim_failed_cell = None, None
for i, src in enumerate(CELLS):
    try:
        exec(compile(src, f"h5-cell-{i + 2}", "exec"), sim_ns)
    except Exception:
        sim_error, sim_failed_cell = traceback.format_exc(), i + 2
        print(sim_error)
        break

import pandas as pd
import torch
summary_path = "/kaggle/working/h5_run_summary.json"
try:
    summary = json.load(open(summary_path)) if sim_error is None else {}
except Exception:
    summary = {}
bl = pd.DataFrame(sim_ns.get("batch_log", []))
if len(bl) and "tokenizer" in sim_ns and "test" in sim_ns:
    tok = sim_ns["tokenizer"]
    texts = sim_ns["test"]["text"].tolist()
    lens = [len(x) for x in tok(texts, truncation=True, max_length=sim_ns["MAX_LENGTH"])["input_ids"]]
    tb = bl[bl["split"] == "test"].copy()
    tb["max_tokens"] = [max(lens[s:s + r]) for s, r in zip(tb["start"], tb["rows"])]
    tb["full_maxlen_batch"] = [min(lens[s:s + r]) == sim_ns["MAX_LENGTH"] for s, r in zip(tb["start"], tb["rows"])]
    tb.to_csv("/kaggle/working/h5_batch_log.csv", index=False)
    summary["sim_batches"] = {
        "test_batches": int(len(tb)), "truncated_rows": int(sum(l == sim_ns["MAX_LENGTH"] for l in lens)),
        "seconds_all": tb["seconds"].describe().to_dict(),
        "seconds_full_512_token_batches": tb.loc[tb["full_maxlen_batch"], "seconds"].describe().to_dict(),
        "seconds_by_max_tokens_bucket": {str(k): v for k, v in tb.groupby(
            pd.cut(tb["max_tokens"], [0, 128, 256, 384, 511, 512]), observed=True)["seconds"]
            .describe()[["count", "mean", "max"]].to_dict("index").items()}}
summary.update({"simulation": SIM_INFO, "exception": sim_error, "failed_cell": sim_failed_cell,
                "status": summary.get("status", "exception" if sim_error else "incomplete"),
                "peak_gpu_memory": summary.get("peak_gpu_memory") or (
                    {"allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                     "reserved_gb": torch.cuda.max_memory_reserved() / 1e9} if torch.cuda.is_available() else None),
                "submission_parquet_written": __import__("os").path.exists("/kaggle/working/submission.parquet")})
text = json.dumps(summary, indent=1, default=str)  # serialise fully before touching the file
with open(summary_path, "w") as f:
    f.write(text)
print(json.dumps({k: v for k, v in summary.items() if k not in ("config", "versions", "dev_check")}, indent=1, default=str))
'''


def build():
    nb = json.loads(B.build())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    install, cells = code[0], code[1:]
    old = 'INPUT_ROOT = "/kaggle/input"'
    assert cells[0].count(old) == 1
    cells[0] = cells[0].replace(old, f'INPUT_ROOT = "{SIM_ROOT}"')
    cells_json = json.dumps(cells)
    assert '"""' not in cells_json
    md = ("# ENGS108 H5: 4,000-row inference simulation\n\nGenerated by `submission/h5/sim4k/build_sim.py`. Runs the code "
          "cells of `rs2027-engs108-submit-h5.ipynb` unchanged (real model, no stubs) on a synthetic 4,000-row test.csv; "
          "only `INPUT_ROOT` points at the synthetic mount. Not a competition submission.")
    src = [("markdown", md), ("code", install), ("code", SIM_DATA.replace("__SIM_ROOT__", SIM_ROOT)),
           ("code", RUNNER.replace("__CELLS__", cells_json))]
    nb["cells"] = [{"cell_type": t, "id": f"cell-{i:02d}", "metadata": {}, "source": s.splitlines(keepends=True),
                    **({"outputs": [], "execution_count": None} if t == "code" else {})} for i, (t, s) in enumerate(src)]
    return json.dumps(nb, indent=1) + "\n"


def kernel_metadata():
    k = json.loads(B.kernel_metadata())
    k.update(id=KERNEL, title="RS2027-ENGS108-h5-sim4k", code_file=NB_NAME)
    return json.dumps(k, indent=2) + "\n"


if __name__ == "__main__":
    (HERE / NB_NAME).write_text(build())
    (HERE / "kernel-metadata.json").write_text(kernel_metadata())
    print("wrote", HERE / NB_NAME)
