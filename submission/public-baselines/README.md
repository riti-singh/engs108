# Public baselines for "Name That LLM"

Faithful reproductions of two public Kaggle notebooks (A0, B0) plus three single-change variants (V1–V3) of the selected
baseline. Notebooks are saved as Kaggle versions under `ritisingh123`; **nothing is submitted to the competition** — submit manually.

| Path | What |
|---|---|
| `refA/`, `refB/` | unmodified `kaggle kernels pull` of the sources (+ extracted `.py.txt` for review) |
| `configs/<ID>.json` | frozen, versioned config per experiment: compat edits and the single variant `change` (exact `old`/`new` source text) |
| `build_notebooks.py` | rebuilds `build/<ID>/{submit,validate}/` (notebook + `kernel-metadata.json`) from the source + config; every edit must match exactly once |
| `check_notebooks.py` | static syntax check of all built notebooks and their `%%writefile` modules |
| `build/` | the generated notebooks that were pushed |
| `manifest.csv` | one row per experiment: versions, change, assets, validation, runtime, GPU memory, status, link |
| `docs/source_audit.md` | versions, authors, assets, licenses, and the inference/ensemble logic read from source |
| `docs/iteration_plan.md` | validate/submit protocol, baseline selection rule, variant candidates |
| `docs/raw/` | raw Kaggle API output, run logs, `run_report.json` per run |

Rebuild: `python build_notebooks.py A0 B0 V1 V2 V3 && python check_notebooks.py`, then `kaggle kernels push -p build/<ID>/<mode>`.

## How the notebooks differ from the sources

Only these edits are applied (all in `configs/*.json`):

* **Prepended telemetry cell** — starts `nvidia-smi` logging (10 s) for peak GPU memory; prints mounted input layout.
* **Appended report cell** — re-reads `/kaggle/working/submission.parquet` and asserts columns exactly `id, model`, same row
  count and id order as `test.csv`, unique ids, no nulls, labels ⊂ the 18 valid labels; writes `engs108_run/run_report.json`.
* **A0 `remove_public_sample_preview`** — Ref A writes a constant `claude-fable-5` 3-row file and skips all models when
  `test.csv` fingerprint-matches the public sample. Disabled: every input runs the full pipeline.
* **A0 `map_input_layout`** — Ref A hardcodes both flat (`/kaggle/input/<slug>`) and nested
  (`/kaggle/input/datasets/<owner>/<slug>`, `/kaggle/input/competitions/name-that-llm`) paths; our GPU session mounted the flat
  layout (first two A0 runs failed on this) and a CPU session the nested one. An IPython input transformer rewrites input paths
  (including `%%writefile` bodies) to the mounted layout. Ref B discovers inputs with recursive globs and needs no change.
* **B0 `remove_fast_commit_placeholder`** — Ref B sets `FAST_COMMIT = len(test) < 50` and then skips the comp stack and
  `infer.py`, writing a constant `gpt-5.5` file. Set to `False`. Ref B's other `len(test) < 50` branches inside `infer.py` are
  source behaviour for tiny inputs and are not reached by the 540-row validation or the hidden test.
* **`validate` mode only** — a prepended cell builds `/kaggle/working/ntl_val_data/` (real `train.csv`/`sample_submission`
  symlinked; `test.csv` = 540 rows of 30 train prompts, hashed ids) and the competition directory literal is pointed there.

**train.csv:** both sources read the competition `train.csv` as part of their method (fusion/balancing in A, fitted heads in B).
Per the user's decision these reads are kept as-is; no new train.csv reads are added.
