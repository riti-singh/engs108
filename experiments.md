# Experiments

## 2026-11-06 — exp01: Qwen3-8B-Base QLoRA sequence classifier (baseline)

- **Kaggle training run:** [`ritisingh123/rs2027-engs108`](https://www.kaggle.com/code/ritisingh123/rs2027-engs108), version 2. Ran on a single T4 (`CUDA_VISIBLE_DEVICES=0`, `device_map={"": 0}`); the second T4 stayed at 0 MiB used.
- **Environment:** python 3.13, torch 2.11.0+cu130, transformers 5.20.2, peft 0.21.2, accelerate 1.15.0, bitsandbytes 0.50.2.
- **Validation:** `GroupShuffleSplit` on `prompt_id` with `test_size=0.15` and `random_state=42`. This gives 2285 train rows from 127 prompts and 414 validation rows from 23 held-out prompts, i.e. 23 rows per class.
- **Artifacts:**
  - Adapter: Kaggle dataset [`ritisingh123/engs108-qwen3-8b-lora-adapter`](https://www.kaggle.com/datasets/ritisingh123/engs108-qwen3-8b-lora-adapter). It holds the LoRA adapter, the `score` head and `label_map.json`, 15.65 MB in total.
  - bitsandbytes wheel for offline use: [`ritisingh123/engs108-offline-wheels`](https://www.kaggle.com/datasets/ritisingh123/engs108-offline-wheels).

### Hyperparameters

| | |
|---|---|
| Base model | Qwen3-8B-Base, 4-bit NF4, double quant, fp16 compute |
| LoRA | r=8, alpha=16, dropout=0.05, target `q_proj`/`v_proj`, `modules_to_save=["score"]` (3,907,584 trainable params, 0.05%) |
| max_length | 512. Token lengths: p99=316, max=381, so 0 of 2699 texts were truncated |
| Batch | per-device 2 × grad accum 8 = effective 16. A smoke run on the 64 longest texts peaked at 7.5 GB reserved |
| Optimizer | AdamW (Trainer default), lr 1e-4, 3 epochs, 429 steps, fp16, gradient checkpointing (non-reentrant) |
| Selection | eval/save every epoch, `load_best_model_at_end` on `balanced_accuracy`, seed 42 |

### Results (held-out prompt groups)

| Epoch | eval_loss | balanced_accuracy |
|---|---|---|
| 1 | 2.6108 | 0.1304 |
| 2 | 2.3173 | 0.2005 |
| 3 | 2.2104 | **0.2464** (best, loaded) |

Validation balanced accuracy is **0.2464**. Chance level is 1/18 = 0.0556. Plain accuracy is the same (0.2464) because the classes are balanced.

Per-class recall (support 23 each):

| model | recall |
|---|---|
| claude-fable-5 | 0.174 |
| claude-opus-4-6 | 0.391 |
| claude-opus-4-8 | 0.043 |
| claude-opus-5 | 0.652 |
| claude-sonnet-5 | 0.217 |
| gemini-3.1-pro-preview | 0.783 |
| gemini-3.6-flash | 0.217 |
| gemini-3.7-flash | 0.174 |
| gemini-3.8-flash | 0.304 |
| glm-5.3 | 0.000 |
| glm-5.3-flash | 0.087 |
| gpt-5.5 | 0.304 |
| gpt-5.6-luna | 0.174 |
| gpt-5.6-sol | 0.174 |
| gpt-5.6-terra | 0.217 |
| gpt-6-astra | 0.261 |
| kimi-k3 | 0.043 |
| qwen3.8-max | 0.217 |

### Runtime

- Training took 100.1 min (`train_runtime` 6004 s, 1.14 samples/s). The whole notebook took 104.1 min on a T4.
- Peak GPU memory was 6.96 GB allocated and 7.50 GB reserved.

### Observations

- Validation balanced accuracy was still rising at epoch 3 (0.130 → 0.200 → 0.246), so the model probably had not converged.
- The mean training loss over the run is 24.6, far above the validation loss of about 2.2. The smoke run (4 optimizer steps) averaged 147 training loss. By the end of epoch 1, eval loss was 2.61.
  - *Inference, not measured:* the randomly initialised `score` head probably produces very large logits on Qwen3's final hidden states, so the early steps go mostly to rescaling the head.
- No confusion matrix was computed in this run. Which classes get confused with which is not yet measured.

### Offline inference check

- **Kernel:** [`ritisingh123/rs2027-engs108-inference`](https://www.kaggle.com/code/ritisingh123/rs2027-engs108-inference), version 2, with `enable_internet: false`. A request to `https://pypi.org` from inside the session failed with `URLError`.
- **Setup:** bitsandbytes 0.50.2 was installed from the attached wheel dataset using `--no-index`. The adapter was loaded from the attached adapter dataset.
- **Result:** on the same held-out prompt groups, balanced accuracy was 0.2464, identical to the training run. The written `submission.parquet` matches `sample_submission.parquet` (`id: large_string`, `model: large_string`). Its 3 test predictions are identical to the ones the training kernel wrote.
- **Runtime:** 4.7 min total.
