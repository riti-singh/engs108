# submission/h5 — inference-only H5 ensemble for Name That LLM

Inference only, no training. The notebook predicts with the exp03 pre-registered **H5** blend: an equal-weight log-linear blend
(`exp03/stylo/protocol.py::logprob_blend`, weights `[1, 1, 1]`) of

| component | source |
|---|---|
| `armh_head` | Arm H epoch-6 QLoRA score head (`ritisingh123/engs108-exp03-armh-artifacts/adapter`) |
| `probe_armh` | C=0.03 logistic probe on the Arm H pooled hidden state (`probe_armh.npz`) |
| `stylo_exp03` | exp03 stylometric model (`stylo_exp03_model.joblib`) |

The H6 dev-LOPO weights are **not** used.

## Branch dependency

Branched from the PR #5 head `76392e05`, not from `main`, because `main` has no exp03 code. That commit already contains
`exp03/stylo/artifacts/stylo_exp03_model.joblib`, `exp03/stylo/protocol.py` (frozen `CLASS_NAMES`/split),
`exp03/probe/probe.py`, `exp03/eval_protocol.md` and `exp03/qlora_armh/`. Nothing had to be merged or cherry-picked:
the later commits on `exp03-stylo-probe` add stylegram/SVC models that H5 does not use.

## Files

| file | purpose |
|---|---|
| `config.json` | frozen settings: classes, model/quantization, batch 8, MAX_LENGTH 512, H5 blend, dev expectations, SHA-256 hashes, Kaggle settings |
| `h5_core.py` | component probabilities, H5 blend, class-order, dev-reproduction and submission checks, runtime extrapolation |
| `build_notebook.py` | generates `rs2027-engs108-submit-h5.ipynb` + `kernel-metadata.json` (inlines `armh_core.pooled_forward`, `probe.py`, `protocol.py`, `stylo/features.py`, `h5_core.py`) |
| `make_artifacts_dataset.py` | builds the Kaggle dataset `ritisingh123/engs108-submission-h5-artifacts` (CPU only) |
| `artifacts/h5_dev_reference.csv` | per-row dev component + H5 probabilities from the Arm H shared pooled pass (pooled logits + `pooled_hidden_armh.npy`) |
| `tests/test_h5.py` | offline tests (notebook in sync with config, inference-only, hashes, class order, H5 reproduction, submission checks) |
| `kaggle_run_v3/` | output from the saved Kaggle run (summary, dev check, per-batch log, visible-stub predictions) |

## Kernel

`ritisingh123/rs2027-engs108-submit-h5` (private, GPU T4, internet off). Data sources: the `name-that-llm` competition, the model
`qwen-lm/qwen-3/transformers/8b-base/1`, `ritisingh123/engs108-exp03-armh-artifacts`,
`ritisingh123/engs108-submission-h5-artifacts` (`probe_armh.npz`, `stylo_exp03_model.joblib`, `blend_H5_dev.csv`,
`h5_dev_reference.csv`, `SHA256SUMS`) and `ritisingh123/engs108-offline-wheels` (bitsandbytes).

What the notebook does:
1. Checks SHA-256 hashes for all artifacts, loads `test.csv` (does not assume a row count) and rebuilds the exp01 dev split (414 rows / 23 prompts).
2. Loads `Qwen3ForSequenceClassification` in 4-bit NF4 (fp16 compute, double quant), `num_labels=18`, alphabetical classes, then
   `prepare_model_for_kbit_training(use_gradient_checkpointing=False)` and the frozen epoch-6 adapter (`PeftModel`, `is_trainable=False`).
   Asserts identical class order across the head config, `label_map.json`, the probe npz and stylo `classes_`.
3. One forward pass per batch of 8 (`armh_core.pooled_forward`, fp16 autocast, MAX_LENGTH 512) returns the head logits and
   the pooled hidden state together. The probe uses the hidden state, cast to fp16 as in its training features.
4. Dev reproduction. Raises an error before any test prediction unless all of these hold: balanced accuracy within 1e-3 of
   0.43237, argmax identical to `blend_H5_dev.csv` on every row, and max |Δp| ≤ 0.01 against `h5_dev_reference.csv`.
5. Takes the argmax of the test H5 blend and writes `/kaggle/working/submission.parquet` with exactly `id`, `model` (schema cast to
   `sample_submission.parquet`). Then checks unique non-null ids that match `test.csv` and labels within the 18 classes.
6. Logs time and peak GPU memory for each batch and extrapolates the runtime to 4,000 rows.

Why `prepare_model_for_kbit_training` is needed: the Arm H kernel computed `pooled_hidden_armh.npy` and the head logits after
that call, which upcasts the embeddings, RMSNorms and score layer to fp32. Kernel v2 loaded those layers in fp16 and the dev check
failed (bal-acc 0.4300, 1 argmax flip, max |Δp| 0.017). With the same dtype state (v3), the probabilities match the reference
to about 5e-11. `blend_H5_dev.csv` was blended from the length-sorted head pass. Its probabilities differ from the shared-pass
blend by up to 0.0066, but the argmax is identical on all 414 rows.

## Saved run (version 3)

https://www.kaggle.com/code/ritisingh123/rs2027-engs108-submit-h5, saved version 3, status `COMPLETE`, internet unreachable
from inside the kernel. Run on the 3-row visible test stub.

| metric | value |
|---|---|
| dev balanced accuracy | 0.432367 (expected 0.432367), argmax agreement with `blend_H5_dev.csv` 414/414 |
| max \|Δp\| vs shared-pass reference | 5.0e-11 |
| notebook runtime | 4.3 min (model load 1.2, dev 2.0, test 0.01) |
| peak GPU memory | 7.44 GB allocated / 7.70 GB reserved (T4, 15 GB) |
| batch time (dev, batch 8) | mean 2.28 s, p95 2.91 s |
| estimate for 4,000 test rows | about 19 min forward pass, about 23 min total including dev check and load |

The competition Submit button was not clicked. Submitting is a manual step.

## Commands

```bash
ENGS108_DATA_DIR=~/data/nthl ~/venvs/engs108/bin/python -m pytest -q submission/h5/tests
python submission/h5/build_notebook.py
ENGS108_DATA_DIR=~/data/nthl ARMH_OUT_DIR=~/data/armh_ds python submission/h5/make_artifacts_dataset.py ~/kaggle_h5_artifacts  # rebuild dataset only
KAGGLE_API_TOKEN=... kaggle kernels push -p submission/h5
KAGGLE_API_TOKEN=... kaggle kernels output ritisingh123/rs2027-engs108-submit-h5 -p ~/data/h5_run
```
