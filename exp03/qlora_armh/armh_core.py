"""Arm H training utilities (inlined into the Kaggle notebook by build_notebook.py; tested on CPU in tests/).

- `split_param_groups`: head (`score` modules_to_save copy) vs LoRA params, with their own learning rates.
- `ArmHTrainer`: Trainer that records raw micro-batch training losses and per-group pre-clip grad norms.
- `ArmHMonitor`: callback logging, per optimizer step, loss, grad norms, update norms ||delta theta||, lrs,
  GradScaler state, head-vs-init cosine; applies the frozen stop criteria (config.json `stop_criteria`).
- `pooled_forward` (exp02) and `sorted_predict_logits` (exp01 inference notebook) for epoch-6 outputs.
"""
import math

import numpy as np
import torch
from transformers import Trainer, TrainerCallback


def split_param_groups(model):
    head, lora, other = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "score" in n:
            head.append((n, p))
        elif "lora_" in n:
            lora.append((n, p))
        else:
            other.append(n)
    assert not other, f"unexpected trainable params: {other[:5]}"
    assert head and lora
    return head, lora


def build_optimizer(model, training_args, head_lr, lora_lr):
    """Trainer's default optimizer class/kwargs (as exp01), with one param group per block."""
    head, lora = split_param_groups(model)
    opt_cls, opt_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args, model)
    groups = [{"params": [p for _, p in head], "lr": head_lr, "weight_decay": training_args.weight_decay, "name": "head"},
              {"params": [p for _, p in lora], "lr": lora_lr, "weight_decay": training_args.weight_decay, "name": "lora"}]
    return opt_cls(groups, **opt_kwargs), opt_cls.__name__, {k: v for k, v in opt_kwargs.items() if k != "lr"}


def _norm(params, scale=1.0):
    sq = 0.0
    for p in params:
        if p.grad is not None:
            sq += float(p.grad.detach().float().pow(2).sum())
    return math.sqrt(sq) / scale


class ArmHTrainer(Trainer):
    """Records raw (pre-GA-division) micro-batch losses and per-group pre-clip grad norms."""

    def __init__(self, *a, head_params=None, lora_params=None, **kw):
        super().__init__(*a, **kw)
        self.head_params, self.lora_params = head_params, lora_params
        self.micro_losses, self.last_clip = [], {}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        out = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        if model.training:
            loss = out[0] if return_outputs else out
            self.micro_losses.append(float(loss.detach().float()))
        return out

    def _clip_grad_norm(self, model):
        scale = scaler_scale(self)
        head = _norm(self.head_params, scale)
        lora = _norm(self.lora_params, scale)
        total = super()._clip_grad_norm(model)
        self.last_clip = {"grad_norm_head_preclip": head, "grad_norm_lora_preclip": lora,
                          "grad_norm_total_preclip": float(total), "scaler_scale": scale}
        return total


def scaler_scale(trainer):
    scaler = getattr(trainer.accelerator, "scaler", None)
    return float(scaler.get_scale()) if scaler is not None and scaler.is_enabled() else 1.0


def check_stop(rows, params):
    """Frozen stop criteria on the per-step log rows. Returns a reason string or None."""
    k1, k2 = params["S1_k"], params["S2_k"]
    if len(rows) >= k1 and all(not math.isfinite(r["train_loss"]) for r in rows[-k1:]):
        return f"S1: non-finite train loss on {k1} consecutive optimizer steps"
    if len(rows) >= k2 and all(not math.isfinite(r["grad_norm_total_preclip"]) for r in rows[-k2:]):
        return f"S2: non-finite pre-clip grad norm on {k2} consecutive optimizer steps"
    if len(rows) >= params["S3_min_step"]:
        rate = sum(r["skipped"] for r in rows) / len(rows)
        if rate > params["S3_max_rate"]:
            return f"S3: GradScaler skip rate {rate:.3f} > {params['S3_max_rate']}"
    return None


class ArmHMonitor(TrainerCallback):
    def __init__(self, trainer, stop_params, log_every=10, printer=print):
        self.tr, self.stop_params, self.log_every, self.printer = trainer, stop_params, log_every, printer
        self.head0 = [p.detach().float().clone() for p in trainer.head_params]
        self.rows, self.stop_reason, self._before = [], None, None

    def on_pre_optimizer_step(self, args, state, control, **kw):
        self._before = {"head": [p.detach().float().clone() for p in self.tr.head_params],
                        "lora": [p.detach().float().clone() for p in self.tr.lora_params]}
        self._lrs = {g.get("name", str(i)): g["lr"] for i, g in enumerate(self.tr.optimizer.param_groups)}
        self._post = (_norm(self.tr.head_params), _norm(self.tr.lora_params))

    def on_optimizer_step(self, args, state, control, **kw):
        upd = {}
        for k, params in (("head", self.tr.head_params), ("lora", self.tr.lora_params)):
            sq = sum(float((p.detach().float() - b).pow(2).sum()) for p, b in zip(params, self._before[k]))
            upd[k] = math.sqrt(sq)
        w = torch.cat([p.detach().float().flatten() for p in self.tr.head_params])
        w0 = torch.cat([p.flatten() for p in self.head0])
        losses = self.tr.micro_losses
        self.tr.micro_losses = []
        clip = self.tr.last_clip
        row = {"step": state.global_step + 1, "epoch_float": None,
               "train_loss": float(np.mean(losses)) if losses else float("nan"), "n_micro": len(losses),
               **clip,
               "grad_norm_head_postclip": self._post[0], "grad_norm_lora_postclip": self._post[1],
               "update_norm_head": upd["head"], "update_norm_lora": upd["lora"],
               "lr_head": self._lrs.get("head"), "lr_lora": self._lrs.get("lora"),
               # GradScaler halves its scale exactly when it skips a step (found inf/nan). accelerate's
               # optimizer_step_was_skipped misses skips with fused AdamW (step() is still called with found_inf).
               "scaler_scale_after": scaler_scale(self.tr),
               "skipped": scaler_scale(self.tr) < clip.get("scaler_scale", 1.0),
               "skipped_accelerate_flag": bool(self.tr.accelerator.optimizer_step_was_skipped),
               "head_cos_init": float(torch.nn.functional.cosine_similarity(w, w0, dim=0)),
               "head_frob": float(w.norm()), "head_delta_frob": float((w - w0).norm())}
        self.rows.append(row)
        reason = check_stop(self.rows, self.stop_params)
        if reason and self.stop_reason is None:
            self.stop_reason = reason
            control.should_training_stop = True
            self.printer(f"STOP at step {row['step']}: {reason}")
        if row["step"] % self.log_every == 0 or row["step"] == 1:
            self.printer({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
        return control

    def on_step_end(self, args, state, control, **kw):
        if self.rows:
            self.rows[-1]["epoch_float"] = state.epoch


def ga_fix_check(rows, log_history):
    """Trainer-logged loss vs mean of recorded raw micro-batch losses over the same optimizer steps."""
    by_step = {r["step"]: r for r in rows}
    out, prev = [], 0
    for h in log_history:
        if "loss" in h and "step" in h and "eval_loss" not in h:
            s = h["step"]
            seg = [by_step[i] for i in range(prev + 1, s + 1) if i in by_step]
            # Trainer weights each step's loss by n_micro/GA; logged = sum(step losses) / n_steps
            ref = float(np.mean([r["train_loss"] for r in seg])) if seg else float("nan")
            out.append({"step": s, "logged": h["loss"], "recorded_mean": ref,
                        "rel_diff": abs(h["loss"] - ref) / max(abs(ref), 1e-12)})
            prev = s
    return out


def pooled_forward(model, score_module, tokenizer, texts, batch_size, max_length, n_classes, device="cuda:0"):
    """exp02 pooled_forward: hidden state at the last non-pad token feeding `score`, plus the model logits."""
    captured = {}
    handle = score_module.register_forward_hook(lambda mod, inp, out: captured.__setitem__("h", inp[0].detach()))
    hidden = np.zeros((len(texts), score_module.in_features if hasattr(score_module, "in_features")
                       else score_module.original_module.in_features), dtype=np.float32)
    logits = np.zeros((len(texts), n_classes), dtype=np.float32)
    try:
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                enc = tokenizer(batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
                with torch.autocast(device.split(":")[0], dtype=torch.float16, enabled=device.startswith("cuda")):
                    out = model(**enc).logits
                ids = enc["input_ids"]
                non_pad = (ids != tokenizer.pad_token_id).int()
                last = (torch.arange(ids.shape[-1], device=ids.device) * non_pad).argmax(-1)
                h = captured["h"][torch.arange(ids.shape[0], device=ids.device), last]
                hidden[start:start + len(batch)] = h.float().cpu().numpy()
                logits[start:start + len(batch)] = out.float().cpu().numpy()
    finally:
        handle.remove()
    return hidden, logits


@torch.no_grad()
def sorted_predict_logits(model, tokenizer, texts, batch_size, max_length, device="cuda:0"):
    """Length-sorted (longest first) predict loop of inference/rs2027-engs108-inference.ipynb, returning logits."""
    order = np.argsort([len(t) for t in texts])[::-1]
    out = None
    for start in range(0, len(texts), batch_size):
        idx = order[start:start + batch_size]
        enc = tokenizer([texts[i] for i in idx], truncation=True, max_length=max_length, padding=True,
                        return_tensors="pt").to(device)
        with torch.autocast(device.split(":")[0], dtype=torch.float16, enabled=device.startswith("cuda")):
            lg = model(**enc).logits.float().cpu().numpy()
        if out is None:
            out = np.zeros((len(texts), lg.shape[1]), dtype=np.float32)
        out[idx] = lg
    return out
