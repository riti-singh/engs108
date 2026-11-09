"""Offline tests for exp03 Arm H (no network). The tiny-model training test needs torch/transformers/peft
(CPU) and skips otherwise; it checks the param groups, the GA fix, the grad-norm bookkeeping and the monitor."""
import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def test_notebook_in_sync_with_config_and_core():
    import build_notebook as B
    assert (HERE / B.NB_NAME).read_text() == B.build(smoke=False)
    assert (HERE / "kernel-metadata.json").read_text() == B.kernel_metadata(smoke=False)
    assert "SMOKE_TEST = False" in B.build(smoke=False) and "SMOKE_TEST = True" in B.build(smoke=True)


def test_frozen_config_values():
    c = json.loads((HERE / "config.json").read_text())
    t, l = c["training"], c["lora"]
    assert c["seed"] == 42 and t["num_train_epochs"] == 6 and t["expected_optimizer_steps"] == 6 * 143
    assert t["param_groups"]["head"]["lr"] == 1e-3 and t["param_groups"]["lora"]["lr"] == 1e-4
    assert (l["r"], l["lora_alpha"], l["lora_dropout"], l["target_modules"], l["modules_to_save"]) == \
        (8, 16, 0.05, ["q_proj", "v_proj"], ["score"])
    assert (t["max_length"], t["per_device_train_batch_size"], t["gradient_accumulation_steps"], t["max_grad_norm"]) == (512, 2, 8, 1.0)
    assert (t["lr_scheduler_type"], t["warmup_steps"], t["load_best_model_at_end"]) == ("linear", 0, False)
    assert set(c["pre_registered_dev_reports"]) >= {"H1", "H2", "H3", "H4", "H5", "H6"}
    assert c["kaggle"]["leaderboard_submission"] is False and c["data"]["external_data"] is False


torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
import armh_core as A  # noqa: E402

P = {"S1_k": 3, "S2_k": 10, "S3_max_rate": 0.05, "S3_min_step": 100}


def _row(loss=1.0, gn=1.0, skipped=False):
    return {"train_loss": loss, "grad_norm_total_preclip": gn, "skipped": skipped}


def test_stop_criteria():
    assert A.check_stop([_row()] * 200, P) is None
    assert A.check_stop([_row()] * 5 + [_row(loss=float("nan"))] * 2, P) is None
    assert A.check_stop([_row()] * 5 + [_row(loss=float("nan"))] * 3, P).startswith("S1")
    assert A.check_stop([_row(gn=float("inf"), skipped=True)] * 9, P) is None   # GradScaler calibration
    assert A.check_stop([_row(gn=float("inf"), skipped=True)] * 10, P).startswith("S2")
    rows = [_row(gn=float("inf"), skipped=True)] * 5 + [_row()] * 94
    assert A.check_stop(rows, P) is None and A.check_stop(rows + [_row()], P) is None   # 5/100 = 5%: not above
    rows = ([_row(gn=float("inf"), skipped=True)] * 3 + [_row()] * 30) * 4
    assert A.check_stop(rows, P).startswith("S3")


def _tiny_run(fix, epochs=2, n=40, logging_steps=1):
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Qwen3Config, Qwen3ForSequenceClassification, TrainingArguments, set_seed
    import numpy as np
    rng = np.random.default_rng(0)
    rows = [{"input_ids": rng.integers(1, 256, size=int(rng.integers(6, 30))).tolist(), "labels": int(rng.integers(0, 18))}
            for _ in range(n)]
    ds = Dataset.from_list(rows)

    def collate(feats):
        L = max(len(f["input_ids"]) for f in feats)
        ids = torch.zeros((len(feats), L), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, f in enumerate(feats):
            ids[i, :len(f["input_ids"])] = torch.tensor(f["input_ids"])
            mask[i, :len(f["input_ids"])] = 1
        return {"input_ids": ids, "attention_mask": mask, "labels": torch.tensor([f["labels"] for f in feats])}

    set_seed(42)
    m = Qwen3ForSequenceClassification(Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                                   num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                                   num_labels=18, pad_token_id=0))
    m = get_peft_model(m, LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.05,
                                     target_modules=["q_proj", "v_proj"], modules_to_save=["score"], bias="none"))
    head, lora = A.split_param_groups(m)
    args = TrainingArguments(output_dir="/tmp/armh_test", per_device_train_batch_size=2, gradient_accumulation_steps=8,
                             learning_rate=1e-4, num_train_epochs=epochs, max_grad_norm=1.0, logging_steps=logging_steps,
                             report_to="none", use_cpu=True, seed=42, save_strategy="no", lr_scheduler_type="linear",
                             warmup_steps=0)
    opt, name, kw = A.build_optimizer(m, args, 1e-3, 1e-4)
    tr = A.ArmHTrainer(model=m, args=args, train_dataset=ds, data_collator=collate, optimizers=(opt, None),
                       head_params=[p for _, p in head], lora_params=[p for _, p in lora])
    if fix:
        tr.model_accepts_loss_kwargs = False
    mon = A.ArmHMonitor(tr, P, printer=lambda *a: None)
    tr.add_callback(mon)
    tr.train()
    return tr, mon


def test_tiny_training_groups_gafix_and_norms():
    tr, mon = _tiny_run(fix=True)
    rows = mon.rows
    assert len(rows) == tr.state.global_step == 6          # 20 micro-batches/epoch -> 8, 8, 4
    assert [r["n_micro"] for r in rows] == [8, 8, 4, 8, 8, 4]
    assert math.isclose(rows[0]["lr_head"], 1e-3) and math.isclose(rows[0]["lr_lora"], 1e-4)
    for r in rows:
        assert math.isclose(r["lr_head"] / r["lr_lora"], 10.0, rel_tol=1e-9)
        tot = math.hypot(r["grad_norm_head_preclip"], r["grad_norm_lora_preclip"])
        assert math.isclose(tot, r["grad_norm_total_preclip"], rel_tol=1e-4)
        coef = min(1.0, 1.0 / (r["grad_norm_total_preclip"] + 1e-6))
        assert math.isclose(r["grad_norm_head_postclip"], r["grad_norm_head_preclip"] * coef, rel_tol=1e-3)
        assert r["update_norm_head"] > 0 and r["update_norm_lora"] > 0 and not r["skipped"]
    assert rows[0]["head_cos_init"] < 1.0 and rows[-1]["head_cos_init"] < rows[0]["head_cos_init"] + 1e-6
    chk = A.ga_fix_check(rows, tr.state.log_history)
    assert len(chk) == 6 and max(c["rel_diff"] for c in chk) < 1e-3   # logged loss = true mean micro-batch CE


def test_tiny_training_without_fix_logs_ga_times_loss():
    tr, mon = _tiny_run(fix=False, epochs=1)
    chk = A.ga_fix_check(mon.rows, tr.state.log_history)
    assert math.isclose(chk[0]["logged"] / chk[0]["recorded_mean"], 8.0, rel_tol=1e-3)


def test_skip_detection_uses_scaler_scale():
    import types
    scale = {"v": 4096.0}
    scaler = types.SimpleNamespace(get_scale=lambda: scale["v"], is_enabled=lambda: True)
    tr = types.SimpleNamespace(accelerator=types.SimpleNamespace(scaler=scaler, optimizer_step_was_skipped=False))
    assert A.scaler_scale(tr) == 4096.0
    tr.accelerator.scaler = None
    assert A.scaler_scale(tr) == 1.0
