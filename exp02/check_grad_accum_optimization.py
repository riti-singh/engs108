"""CPU check: does the exp01 gradient-accumulation loss scaling change *optimization*, not just logging?

Tiny randomly initialised Qwen3ForSequenceClassification, transformers 5.20.2 Trainer, AdamW, lr>0.
Every config sees the same 16-row optimizer batches (same seed and sampler), so a correct GA=8 run
should match the GA=1 reference up to float error. Compared configs:

* ref        : per-device 16 x GA 1 (no accumulation, correct by construction)
* exp01      : per-device 2 x GA 8, Trainer defaults (what exp01 ran)
* fix_kwargs : per-device 2 x GA 8, `trainer.model_accepts_loss_kwargs = False` (Trainer divides by GA)
* fix_lossfn : per-device 2 x GA 8, `compute_loss_func` = summed CE / num_items_in_batch

each with `max_grad_norm` in {1.0 (exp01 default), 0 (clipping off)}.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from datasets import Dataset
from transformers import Qwen3Config, Qwen3ForSequenceClassification, Trainer, TrainingArguments, set_seed

N, NUM_LABELS, VOCAB, PAD, LR, EPOCHS = 128, 18, 512, 0, 1e-3, 2
rng = np.random.default_rng(0)
rows = [{"input_ids": rng.integers(1, VOCAB, size=int(rng.integers(8, 40))).tolist(),
         "labels": int(rng.integers(0, NUM_LABELS))} for _ in range(N)]
ds = Dataset.from_list(rows)


def collate(feats):
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


def summed_ce_over_items(outputs, labels, num_items_in_batch=None):
    loss = F.cross_entropy(outputs.logits.float(), labels, reduction="sum")
    return loss / num_items_in_batch


@torch.no_grad()
def full_ce(model):
    model.eval()
    return float(model(**collate(rows)).loss)


def run(name, bs, ga, max_grad_norm):
    model = make_model()
    args = TrainingArguments(output_dir=f"/tmp/ga_opt_{name}", per_device_train_batch_size=bs,
                             gradient_accumulation_steps=ga, learning_rate=LR, num_train_epochs=EPOCHS,
                             logging_steps=1, report_to="none", use_cpu=True, seed=42, save_strategy="no",
                             max_grad_norm=max_grad_norm, lr_scheduler_type="constant", warmup_steps=0)
    kw = {"compute_loss_func": summed_ce_over_items} if name == "fix_lossfn" else {}
    tr = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate, **kw)
    if name == "fix_kwargs":
        tr.model_accepts_loss_kwargs = False
    tr.train()
    logs = [h for h in tr.state.log_history if "loss" in h and "grad_norm" in h]
    params = torch.cat([p.detach().flatten() for p in model.parameters()])
    return {"name": name, "per_device_bs": bs, "grad_accum": ga, "max_grad_norm": max_grad_norm,
            "steps": len(logs), "logged_loss": [h["loss"] for h in logs],
            "grad_norm": [h["grad_norm"] for h in logs], "final_full_ce": full_ce(model), "params": params}


if __name__ == "__main__":
    out = {"versions": {"torch": torch.__version__, "transformers": transformers.__version__},
           "lr": LR, "epochs": EPOCHS, "rows": N, "results": []}
    for mgn in (1.0, 0.0):
        runs = [run("ref", 16, 1, mgn), run("exp01", 2, 8, mgn), run("fix_kwargs", 2, 8, mgn), run("fix_lossfn", 2, 8, mgn)]
        ref = runs[0]
        for r in runs:
            gn = np.array(r["grad_norm"])
            row = {k: r[k] for k in ("name", "per_device_bs", "grad_accum", "max_grad_norm", "steps", "final_full_ce")}
            row["grad_norm_ratio_vs_ref_step1"] = float(gn[0] / ref["grad_norm"][0])
            row["grad_norm_ratio_vs_ref_median"] = float(np.median(gn / np.array(ref["grad_norm"])))
            row["logged_loss_ratio_vs_ref_step1"] = float(r["logged_loss"][0] / ref["logged_loss"][0])
            row["frac_steps_clipped"] = float((gn > mgn).mean()) if mgn > 0 else 0.0
            row["param_rel_l2_vs_ref"] = float((r["params"] - ref["params"]).norm() / ref["params"].norm())
            row["grad_norm"] = [round(x, 4) for x in r["grad_norm"]]
            out["results"].append(row)
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items() if k != "grad_norm"}))
    with open(Path(__file__).parent / "artifacts" / "grad_accum_optimization_check.json", "w") as f:
        json.dump(out, f, indent=1)
