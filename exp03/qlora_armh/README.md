# exp03 Arm H — QLoRA with a 10x head learning rate (6 fixed epochs)

Owner: Agent 3. Only `exp03/qlora_armh/` is written; `exp03/stylo/protocol.py` and `exp03/probe/probe.py` are imported read-only.

## What was run

One approved GPU arm: Qwen3-8B-Base, 4-bit NF4 QLoRA, LoRA unchanged from exp01 (r=8, alpha=16, dropout 0.05,
q_proj/v_proj, `modules_to_save=["score"]`), MAX_LENGTH 512, batch 2 x GA 8, max_grad_norm 1.0, fp16, linear schedule,
0 warmup, seed 42 — and two changes:

* **GA fix**: `trainer.model_accepts_loss_kwargs = False`, so each micro-batch loss is divided by GA (logged loss = true mean CE).
* **Two optimizer groups**: head (`score.modules_to_save.default.weight`, 73,728 params) at lr 1e-3; LoRA (144 tensors, 3,833,856 params) at lr 1e-4. Same fused AdamW as the Trainer default.

Training data: the 2,285 official train rows of the exp01 split (dev 414 rows / 23 prompts never trained on).
Six fixed epochs (858 optimizer steps); **epoch 6 is the primary checkpoint, fixed before training**. Per-epoch dev
metrics are logged as diagnostics and never used to choose anything. Everything is frozen in `config.json`
(committed before the kernel was pushed), including the stop criteria S1-S4 and the pre-registered dev reports H1-H6.

## Files

| file | purpose |
|---|---|
| `config.json` | frozen config: data/split, model, LoRA, training, stop criteria, logging, pre-registered reports, Kaggle |
| `armh_core.py` | param groups, `ArmHTrainer` (raw micro-batch losses, per-group pre-clip grad norms), `ArmHMonitor` (update norms, lrs, GradScaler state/skips, head-vs-init cosine, stop criteria), `pooled_forward`, length-sorted predict loop |
| `build_notebook.py` | generates the Kaggle notebook from `config.json` + `armh_core.py` (and the smoke variant) |
| `rs2027-engs108-exp03-armh.ipynb`, `kernel-metadata.json` | the Kaggle kernel `ritisingh123/rs2027-engs108-exp03-armh` |
| `analyze_armh.py` | CPU analysis: protocol prediction files, new probe, H1-H6, diagnostics |
| `tests/test_armh.py` | offline tests (notebook in sync with config/core, frozen values, stop criteria, tiny CPU Qwen3+LoRA run checking param-group lrs, GA fix, grad-norm bookkeeping, skip detection) |
| `artifacts/` | see "Artifacts" |

## Commands

```bash
~/venvs/torchcpu/bin/python -m pytest -q exp03/qlora_armh/tests              # offline tests
python exp03/qlora_armh/build_notebook.py                                    # regenerate the notebook
python exp03/qlora_armh/build_notebook.py --smoke ~/kaggle_armh_smoke        # smoke variant
KAGGLE_API_TOKEN=... kaggle kernels push -p ~/kaggle_armh_smoke              # smoke kernel (v1, v2)
KAGGLE_API_TOKEN=... kaggle kernels push -p exp03/qlora_armh                 # full run (v1)
KAGGLE_API_TOKEN=... kaggle kernels output ritisingh123/rs2027-engs108-exp03-armh -p ~/data/armh_out
ARMH_OUT_DIR=~/data/armh_out ENGS108_DATA_DIR=~/data/nthl ~/venvs/engs108/bin/python exp03/qlora_armh/analyze_armh.py
```

## Smoke tests (kernel `ritisingh123/rs2027-engs108-exp03-armh-smoke`)

64 longest train texts, 36 dev rows, 2 epochs = 8 optimizer steps, then the full epoch-6 output path on 40 train rows.

* v1 (commit e2e52d0): ran end to end in 5.2 min, peak 7.5 GB reserved. Param groups 0.001/0.0001, n_gpu 1,
  `model_accepts_loss_kwargs` False, trainer `train_loss` 17.498 = mean of the 8 recorded step losses (not 8x).
  Exposed one instrumentation bug: steps 1-5 overflowed (scale 65536 -> 2048, pre-clip grad norm inf, update norm 0)
  but `accelerator.optimizer_step_was_skipped` stayed False: with fused AdamW the GradScaler still calls `step()`
  (passing `found_inf`), so accelerate does not see the skip. S3 would never have fired. Fixed in 7244343: a step is
  "skipped" when the GradScaler scale decreased during it. The same commit corrects the config's schedule note
  (the scheduler *does* advance on those skipped steps — exactly as in exp01, which used the same optimizer) and adds
  notebook cell ids. No hyper-parameter or stop-criterion changed (`config.json` `amendments_before_full_run`).
* v2 (commit 7244343): identical numbers, 5/8 steps now flagged skipped.
* Full run: kernel `ritisingh123/rs2027-engs108-exp03-armh` **v1** (pushed from commit 79ce6fb; notebook identical to
  7244343). Completed on the first launch, no relaunch: 858/858 steps, status `completed`, no stop criterion fired,
  177.7 min training, 198.4 min total, peak 7.5 GB reserved on one T4.

## Training diagnostics (full run, `artifacts/step_log.csv`)

* GradScaler: 5 skipped steps (1-5, scale calibration 65536 -> 2048), none afterwards. S3 rate at step 100 = 5.0% (not above 5%).
* GA fix: logged loss vs mean of the recorded micro-batch CE, max relative difference 2.2e-7 over 85 logging steps (`ga_fix_check.csv`).
* Train CE by epoch (mean): 4.31, 2.09, 1.28, 0.46, 0.074, 0.0081. The head **memorises the train set**: in-sample bal-acc 1.000, log loss 0.0026.
* Pre-clip grad norm (median over non-skipped steps, per epoch) head / LoRA: 103 / 2.5, 22 / 6.5, 12.7 / 8.1, 9.6 / 8.9, 3.9 / 4.1, 0.46 / 0.46.
  88.7% of steps were clipped to 1.0 (100% in epochs 1-4), so the head dominated the clipped direction early in training.
* Update norm ||Δθ|| (median) head / LoRA: 0.039 / 0.038. Head cosine to its init: 0.933 (epoch 1), 0.828, 0.758, 0.723, 0.711, 0.708 (epoch 6); ||W - W0||_F = 5.26 (||W0||_F = 5.42).
* Pooled pass vs length-sorted predict: argmax agreement on dev 99.76%; max |logits - H W^T| = 0.024.

Per-epoch dev (Trainer evaluate; **diagnostic only, not used for any choice**): bal-acc 0.200, 0.430, 0.408, 0.370, 0.401, 0.406;
log loss 2.55, 1.80, 1.88, 2.73, 3.64, 3.54. Epoch 2 is the best epoch on dev, but the primary checkpoint stays epoch 6, as pre-registered.

## Pre-registered dev results (epoch 6; `artifacts/armh_report_table.csv`, `armh_results.json`)

414 dev rows, 23 prompts; CIs are 23-prompt bootstrap (2,000 draws, seed 42); deltas are paired prompt-bootstrap.
B1 = 0.3986 (probe_exp01 + stylo_exp03 equal), R2 = probe_exp01 0.3768.

| id | report | bal-acc [95% CI] | macro F1 | log loss | Δ vs B1 [95% CI] (P≤0) | Δ vs R2 [95% CI] (P≤0) |
|---|---|---|---|---|---|---|
| H1 | armh_head | 0.403 [0.350, 0.457] | 0.394 | 3.540 | +0.005 [-0.043, 0.051] (0.43) | +0.027 [-0.017, 0.065] (0.11) |
| H2 | probe_armh (C=0.03) | 0.415 [0.338, 0.483] | 0.403 | 1.962 | +0.017 [-0.034, 0.068] (0.26) | +0.039 [-0.005, 0.080] (0.04) |
| H3 | equal: probe_armh + stylo_exp03 | 0.423 [0.360, 0.481] | 0.408 | 2.805 | +0.024 [-0.012, 0.060] (0.10) | +0.046 [0.000, 0.087] (0.02) |
| H4 | equal: armh_head + stylo_exp03 | 0.428 [0.367, 0.486] | 0.423 | 4.038 | +0.029 [-0.019, 0.080] (0.13) | +0.051 [0.002, 0.097] (0.02) |
| H5 | equal: probe_armh + armh_head + stylo_exp03 | 0.432 [0.370, 0.488] | 0.426 | 5.056 | +0.034 [-0.017, 0.089] (0.11) | +0.056 [0.005, 0.104] (0.01) |
| H6 | dev-LOPO: probe_armh + armh_head + stylo_exp03 | 0.428 [0.365, 0.483] | 0.419 | 1.655 | +0.029 [-0.022, 0.082] (0.15) | +0.051 [0.005, 0.097] (0.02) |

H6 mean LOPO weights (probe_armh, armh_head, stylo_exp03) = (0.221, 0.114, 0.161).

**Does the head catch up to the probe?** Mostly: H1 − H2 = −0.012 [−0.046, +0.029] (P≤0 = 0.76), argmax agreement 71.7%.
In exp01 the head was 0.130 behind its probe (0.246 vs 0.377). Now the gap is within noise. The Arm H head beats the exp01
head by +0.157 [0.114, 0.203]. But it is badly calibrated (log loss 3.54: train CE went to ~0.008), so the equal blends
that include it have high log loss; H6 (LOPO weights shrink every component) is the calibrated version.

Probe C sweep (pooled train OOF bal-acc on the frozen folds; **optimistic**, the adapter saw all train labels, so
`oof_honest: false`): 0.634, 0.677, 0.691, 0.694, 0.698, 0.697 for C = 0.0003 ... 0.1 → C = 0.03. The OOF level (0.70)
is far above dev (0.415), which shows how optimistic it is.

**Reading.** No report clears B1 by a margin the 23-prompt bootstrap can resolve: every Δ vs B1 CI contains 0. H3-H6 are
reliably above R2 = 0.377. H5 = 0.432 is the highest pre-registered number, but it should be read as "≈ B1 + 0.03 ± 0.05"
rather than as a confirmed gain.

## Artifacts

In the repo (`artifacts/`): `step_log.csv`, `trainer_log_history.json`, `ga_fix_check.csv`, `run_summary.json`,
`kernel_log.json`, `epoch_dev_logits.npy` (6 x 414 x 18, dev order), `armh_sorted_logits.parquet` (train/dev/test
epoch-6 logits, length-sorted loop), `armh_pooled_pass_logits.parquet` / `_test_`, `score_head_init.npy`,
`kernel_output_SHA256SUMS`; protocol §7 files `armh_head_dev.csv`, `armh_head_train_insample.csv`,
`armh_head_test.csv`, `armh_head_meta.json`, `probe_armh_dev.csv`, `probe_armh_oof_train.csv`, `probe_armh.npz`,
`probe_armh_meta.json`, `blend_H{3,4,5,6}_dev.csv`, `armh_report_table.csv`, `armh_results.json`.

Private Kaggle dataset `ritisingh123/engs108-exp03-armh-artifacts` (193 MB, `SHA256SUMS` inside; the same hashes are in
`artifacts/kernel_output_SHA256SUMS`): `adapter/` (epoch 6, primary), `checkpoints/checkpoint-{143,286,429,572,715,858}`
(one per epoch), `pooled_hidden_armh.npy` (2699 x 4096 float16, sha256 in `kernel_output_SHA256SUMS`),
`pooled_hidden_armh_test.npy`, and the logits/logs above. The kernel wrote `run_summary.json` after `SHA256SUMS`, so
its hash was added from the downloaded copy.

## Deviations, assumptions, limitations

* The amendments after smoke v1 are listed above (instrumentation and documentation only).
* The scheduler advances on GradScaler-skipped steps (5 of 858), as it did in exp01.
* Head and LoRA share one global clip at 1.0 (as frozen). The head's large early gradients dominate the clip, so the
  LoRA's effective step early in training is much smaller than its lr suggests.
* Dev is a development comparison set that has now been looked at many times across exp02/exp03; a 23-prompt bootstrap
  cannot resolve differences of ~0.03.
* The epoch-6 head is overfit/overconfident (train CE 0.008, dev log loss 3.54); dev bal-acc peaked at epoch 2 (diagnostic only).
* No test-set scoring or leaderboard submission. `armh_head_test.csv` is written for later integration only.
* Not run (needs approval): intermediate-layer extraction, Arm P, augmented data, a shorter schedule.

## Proposal (not run)

Intermediate-layer extraction is cheap relative to training: one pooled forward pass over the 2,699 rows plus test took
11.5 min at epoch 6 on one T4 (hooking several layers in the same pass adds little). It is justified if the integration
review wants to test whether a mid-layer probe is less overfit than the score-input probe, given the gap between the
train-OOF (0.70) and dev (0.415). It needs the `adapter/` from the dataset and approval for a GPU kernel.
