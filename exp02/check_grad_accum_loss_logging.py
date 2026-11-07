"""CPU check: does the Trainer's logged training loss scale with gradient_accumulation_steps
for a *ForSequenceClassification model in the exp01 library versions?

Uses a tiny randomly initialised Qwen3ForSequenceClassification (not Qwen3-8B) with
learning_rate=0, so the weights never change and the true loss of every batch is known.
"""
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import Dataset
from transformers import (DataCollatorWithPadding, PreTrainedTokenizerFast, Qwen3Config,
                          Qwen3ForSequenceClassification, Trainer, TrainingArguments, set_seed)
from transformers.loss import loss_utils

N, NUM_LABELS, VOCAB, PAD = 64, 18, 512, 0
rng = np.random.default_rng(0)
rows = [{"input_ids": rng.integers(1, VOCAB, size=int(rng.integers(8, 40))).tolist(),
         "labels": int(rng.integers(0, NUM_LABELS))} for _ in range(N)]
ds = Dataset.from_list(rows)


class Collator:
    def __call__(self, feats):
        L = max(len(f["input_ids"]) for f in feats)
        ids = torch.full((len(feats), L), PAD)
        mask = torch.zeros((len(feats), L), dtype=torch.long)
        for i, f in enumerate(feats):
            ids[i, :len(f["input_ids"])] = torch.tensor(f["input_ids"])
            mask[i, :len(f["input_ids"])] = 1
        return {"input_ids": ids, "attention_mask": mask, "labels": torch.tensor([f["labels"] for f in feats])}


def make_model():
    set_seed(42)
    cfg = Qwen3Config(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_labels=NUM_LABELS,
                      pad_token_id=PAD)
    return Qwen3ForSequenceClassification(cfg)


@torch.no_grad()
def true_mean_ce(model):
    model.eval()
    batch = Collator()(rows)
    return float(model(**batch).loss)


def logged_mean_loss(per_device_bs, grad_accum):
    model = make_model()
    ref = true_mean_ce(model)
    args = TrainingArguments(output_dir="/tmp/ga_check", per_device_train_batch_size=per_device_bs,
                             gradient_accumulation_steps=grad_accum, learning_rate=0.0, num_train_epochs=1,
                             logging_steps=1, report_to="none", use_cpu=True, seed=42, save_strategy="no",
                             max_grad_norm=1.0)
    tr = Trainer(model=model, args=args, train_dataset=ds, data_collator=Collator())
    out = tr.train()
    step_losses = [h["loss"] for h in tr.state.log_history if "loss" in h]
    return {"per_device_bs": per_device_bs, "grad_accum": grad_accum, "steps": out.global_step,
            "model_accepts_loss_kwargs": bool(getattr(tr, "model_accepts_loss_kwargs", None)),
            "true_mean_ce": ref, "logged_train_loss": out.training_loss,
            "ratio_logged_over_true": out.training_loss / ref, "step_losses": step_losses}


if __name__ == "__main__":
    print("python", sys.version.split()[0], "| torch", torch.__version__, "| transformers", transformers.__version__)
    print("ForSequenceClassificationLoss signature:", inspect.signature(loss_utils.ForSequenceClassificationLoss))
    results = [logged_mean_loss(16, 1), logged_mean_loss(2, 8), logged_mean_loss(4, 4)]
    for r in results:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "step_losses"}))
    out = {"versions": {"torch": torch.__version__, "transformers": transformers.__version__},
           "results": results}
    with open(Path(__file__).parent / "artifacts" / "grad_accum_loss_check.json", "w") as f:
        json.dump(out, f, indent=1)
