#!/usr/bin/env python3
"""Route A Step 2: per-task LoRA/head-only adaptation of the 280K-pre-finetuned
mRNABERT scorer on the project MRL task (GSE114002 TRAIN, 2,443 rows,
source-relative delta ranking), reusing the W1' mechanism.

Spec (docs/paper/route2_route_a_necessity_certainty_analysis.md §1):
- Init: Step 1 checkpoint (mRNABERT all-12 + LoRA r16 a32 + masked-mean-pool
  + linear head, 280K library supervised pre-finetuning).
- Step 1 LoRA is MERGED into the base weights (the 280K knowledge becomes the
  frozen starting point), then a FRESH LoRA (r16 a32, zero-init B) is wrapped
  for task adaptation. Trainable = fresh LoRA + head only (W1' no-full-FT
  clause).
- Loss: the Critic V4 screen objective mirrored without the critic machinery -
  huber(delta=1.0) on direction_normalized_delta + different-source-group
  pairwise softplus + soft-Spearman + within-source ranking, with the
  immutable pass-1..8 weight schedule (pass 1-2: huber 1.0 / pairwise 0.25 /
  soft_spearman 0.0; pass 3-8: huber 1.0 / pairwise 0.5 / soft_spearman 0.25;
  within_source 0.5 throughout). Router balance is not applicable.
- Budget: 8 passes x 77 updates (batch 32 records), cosine to 10% of initial
  LR after 5% warmup, grad clip 1.0, BF16 autocast, seed 20260903.
- Selection: FINAL_PASS_8_FIXED (no validation-peak reselection); per-pass
  validation Spearman is logged for the record only.

Eval: frozen-delta protocol on GSE114002 VALIDATION (730 records, K=10,
frozen Task-1 evaluator) - directly comparable to Step 1 0.2470 /
frozen-Optimus 0.3132 / W0 0.1987. Preregistered Step-2 gate: CI not crossing
zero above W0 0.1987 (adjudicated separately via paired bootstrap).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_mrnabert_lora_v3 import LoRALinearV3  # noqa: E402

EVAL_REPO = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
_ev_spec = importlib.util.spec_from_file_location(
    "ev", EVAL_REPO + "/scripts/route_a_v3/evaluate_route2_prediction_v1.py"
)
ev = importlib.util.module_from_spec(_ev_spec)
sys.modules["ev"] = ev
_ev_spec.loader.exec_module(ev)

MRNABERT_PATH = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/mrnabert_a1eb7df25804d23f08646e1cb996b234d7208a40")
STEP1_CHECKPOINT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_route_a/280k_prefinetune_20260903/lora_scorer_checkpoint.pt")
PROJECTION_TRAIN = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/projections/xedit_v3/development_train_validation_v1/train.jsonl")
MANIFEST = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/manifests/route2_development_frozen_v1/development_manifest.jsonl")
CANONICAL_GSE114002 = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/canonical/GSE114002/v1/canonical_records.private.jsonl")
OUT_DIR = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_route_a/step2_task_adapt_20260903")

LORA_RANK = 16
LORA_ALPHA = 32.0
LORA_DROPOUT = 0.05
BATCH = 32
PASSES = 8
LORA_LR = 1e-4
HEAD_LR = 2e-4
WEIGHT_DECAY = 1e-4
SEED = 20260903
HUBER_DELTA = 1.0
SOFT_RANK_TEMPERATURE = 0.2
WITHIN_SOURCE_WEIGHT = 0.5
GRAD_CLIP = 1.0


def format_sequence(sequence: str) -> str:
    return " ".join(str(sequence).upper().replace("U", "T"))


def wrap_lora(model: nn.Module) -> int:
    wrapped = 0
    for layer in model.encoder.layer:
        for attr_path in (
            ("attention", "self", "Wqkv"),
            ("attention", "output", "dense"),
            ("mlp", "gated_layers"),
            ("mlp", "wo"),
        ):
            parent = layer
            for step in attr_path[:-1]:
                parent = getattr(parent, step)
            base = getattr(parent, attr_path[-1])
            if isinstance(base, nn.Linear):
                setattr(parent, attr_path[-1], LoRALinearV3(base, rank=LORA_RANK, alpha=LORA_ALPHA, dropout=LORA_DROPOUT))
                wrapped += 1
    return wrapped


class MeanPoolRegressor(nn.Module):
    def __init__(self, base_model: nn.Module, width: int):
        super().__init__()
        self.base = base_model
        self.head = nn.Linear(width, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.base(input_ids=input_ids, attention_mask=attention_mask)[0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


def loss_weights(pass_number: int) -> dict[str, float]:
    if pass_number <= 2:
        return {"huber": 1.0, "pairwise": 0.25, "soft_spearman": 0.0}
    return {"huber": 1.0, "pairwise": 0.5, "soft_spearman": 0.25}


def soft_ranks(values: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.sigmoid((values.unsqueeze(0) - values.unsqueeze(1)) / temperature).sum(dim=1)


def target_midranks(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort()
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(1, values.numel() + 1, device=values.device, dtype=torch.float32)
    return ranks


def pairwise_softplus(predictions, targets, pairs):
    left = torch.tensor([p[0] for p in pairs], device=predictions.device)
    right = torch.tensor([p[1] for p in pairs], device=predictions.device)
    target_delta = targets[left] - targets[right]
    prediction_delta = predictions[left] - predictions[right]
    return F.softplus(-target_delta.sign() * prediction_delta).mean()


def soft_spearman(predictions, targets, temperature):
    soft = soft_ranks(predictions, temperature)
    target = target_midranks(targets)
    soft_c = soft - soft.mean()
    target_c = target - target.mean()
    denom = torch.linalg.vector_norm(soft_c) * torch.linalg.vector_norm(target_c)
    if not bool((denom > 0).item()):
        return predictions.new_zeros(())
    return 1.0 - (soft_c * target_c).sum() / denom


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU required")
    device = torch.device(f"cuda:{args.physical_gpu_index}")
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- data: GSE114002 TRAIN projection rows ----
    rows = []
    with PROJECTION_TRAIN.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("study_unit_id") == "GSE114002" and row.get("split") == "TRAIN":
                rows.append(row)
    if len(rows) != 2443:
        raise SystemExit(f"expected 2443 GSE114002 TRAIN rows, got {len(rows)}")

    # ---- model: Step-1 structure, merged init, fresh LoRA ----
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    import os as _os
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(MRNABERT_PATH, local_files_only=True)
    model_config = AutoConfig.from_pretrained(MRNABERT_PATH, local_files_only=True, trust_remote_code=True)
    base = AutoModel.from_config(model_config, trust_remote_code=True, add_pooling_layer=False)
    modeling_module = sys.modules[base.__class__.__module__]
    modeling_module.flash_attn_qkvpacked_func = None  # official fallback backend
    wrapped = wrap_lora(base)
    model = MeanPoolRegressor(base, base.config.hidden_size)
    payload = torch.load(STEP1_CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    step1_meta = {k: payload[k] for k in ("lora_rank", "lora_alpha", "target_mean", "target_std", "seed", "epochs", "library_clean_count")}
    del payload

    merged = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinearV3):
                module.base.weight.add_(module.scaling * (module.lora_b @ module.lora_a))
                nn.init.kaiming_uniform_(module.lora_a, a=math.sqrt(5))
                module.lora_b.zero_()
                merged += 1
    print(f"Step-1 LoRA merged into {merged} Linears; fresh LoRA (rank {LORA_RANK}) reset")

    # freeze everything except fresh LoRA + head
    model.requires_grad_(False)
    lora_params = []
    for name, parameter in model.named_parameters():
        if ".lora_a" in name or ".lora_b" in name:
            parameter.requires_grad_(True)
            lora_params.append(parameter)
    model.head.requires_grad_(True)
    head_params = list(model.head.parameters())
    trainable = lora_params + head_params
    trainable_count = sum(p.numel() for p in trainable)
    print(f"trainable parameters: {trainable_count:,} (LoRA {sum(p.numel() for p in lora_params):,} + head {sum(p.numel() for p in head_params):,})")
    model.to(device)

    # ---- tokenize task data ----
    sources = [row["source_sequence"] for row in rows]
    candidates = [row["candidate_sequence"] for row in rows]
    targets = torch.tensor([float(row["direction_normalized_delta"]) for row in rows], dtype=torch.float32)
    groups = [str(row["source_group_id"]) for row in rows]
    enc_src = tokenizer([format_sequence(s) for s in sources], add_special_tokens=True, padding=True, truncation=False, return_tensors="pt")
    enc_cnd = tokenizer([format_sequence(s) for s in candidates], add_special_tokens=True, padding=True, truncation=False, return_tensors="pt")

    # ---- validation records (per-pass monitoring + final eval) ----
    validation_ids = set()
    with MANIFEST.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["study_unit_id"] == "GSE114002" and row["split"] == "VALIDATION":
                validation_ids.add(str(row["canonical_record_id"]))
    vrecords = {}
    with CANONICAL_GSE114002.open() as handle:
        for line in handle:
            row = json.loads(line)
            rid = str(row.get("canonical_record_id"))
            if rid in validation_ids:
                vrecords[rid] = row
    vids = sorted(vrecords)
    v_enc_src = tokenizer([format_sequence(vrecords[rid]["source_sequence"]) for rid in vids], add_special_tokens=True, padding=True, truncation=False, return_tensors="pt")
    v_enc_cnd = tokenizer([format_sequence(vrecords[rid]["candidate_sequence"]) for rid in vids], add_special_tokens=True, padding=True, truncation=False, return_tensors="pt")

    def validate_spearman() -> float:
        model.eval()
        values = []
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for start in range(0, len(vids), 256):
                s = model(v_enc_src["input_ids"][start:start + 256].to(device), v_enc_src["attention_mask"][start:start + 256].to(device)).float()
                c = model(v_enc_cnd["input_ids"][start:start + 256].to(device), v_enc_cnd["attention_mask"][start:start + 256].to(device)).float()
                values.append((c - s).cpu().numpy())
        delta = np.concatenate(values)
        preds = {rid: float(delta[i]) for i, rid in enumerate(vids)}
        observations = ev.load_observations([CANONICAL_GSE114002], validation_ids)
        metrics = ev.evaluate(observations, preds, 10)
        model.train()
        return float(metrics.get("task_macro_spearman"))

    # ---- optimizer ----
    optimizer = torch.optim.AdamW(
        [{"params": lora_params, "lr": LORA_LR}, {"params": head_params, "lr": HEAD_LR}],
        weight_decay=WEIGHT_DECAY,
    )
    updates_per_pass = (len(rows) + BATCH - 1) // BATCH
    total_steps = updates_per_pass * PASSES
    warmup_steps = max(1, int(total_steps * 0.05))

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)

    print(f"pass 0 (pre-training) validation Spearman: {validate_spearman():.4f}")
    order = torch.randperm(len(rows))
    step = 0
    history = []
    for pass_number in range(1, PASSES + 1):
        weights = loss_weights(pass_number)
        losses, hubers, pairs_n, within_n = [], [], 0, 0
        for start in range(0, len(order), BATCH):
            idx = order[start : start + BATCH]
            batch_src_ids = enc_src["input_ids"][idx].to(device)
            batch_src_mask = enc_src["attention_mask"][idx].to(device)
            batch_cnd_ids = enc_cnd["input_ids"][idx].to(device)
            batch_cnd_mask = enc_cnd["attention_mask"][idx].to(device)
            batch_targets = targets[idx].to(device)
            batch_groups = [groups[i] for i in idx.tolist()]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                src_pred = model(batch_src_ids, batch_src_mask).float()
                cnd_pred = model(batch_cnd_ids, batch_cnd_mask).float()
                prediction = cnd_pred - src_pred

            huber = F.huber_loss(prediction, batch_targets, delta=HUBER_DELTA)
            n = len(idx)
            different_pairs, same_pairs = [], []
            for i in range(n):
                for j in range(i + 1, n):
                    (same_pairs if batch_groups[i] == batch_groups[j] else different_pairs).append((i, j))
            pairwise = (
                pairwise_softplus(prediction, batch_targets, different_pairs)
                if different_pairs else prediction.new_zeros(())
            )
            within = (
                pairwise_softplus(prediction, batch_targets, same_pairs)
                if same_pairs else prediction.new_zeros(())
            )
            soft = (
                soft_spearman(prediction, batch_targets, SOFT_RANK_TEMPERATURE)
                if weights["soft_spearman"] > 0 else prediction.new_zeros(())
            )
            loss = (
                weights["huber"] * huber
                + weights["pairwise"] * pairwise
                + weights["soft_spearman"] * soft
                + WITHIN_SOURCE_WEIGHT * within
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss))
            hubers.append(float(huber))
            pairs_n += len(different_pairs)
            within_n += len(same_pairs)
            step += 1
        val = validate_spearman()
        history.append({
            "pass": pass_number,
            "mean_loss": float(np.mean(losses)),
            "mean_huber": float(np.mean(hubers)),
            "different_source_pairs": pairs_n,
            "within_source_pairs": within_n,
            "validation_spearman": val,
        })
        print(f"pass {pass_number}/{PASSES}: loss {np.mean(losses):.4f} huber {np.mean(hubers):.4f} val_spearman {val:.4f} lr {scheduler.get_last_lr()[0]:.2e}", flush=True)
        order = torch.randperm(len(rows))

    # ---- final frozen-delta evaluation (final pass 8 fixed) ----
    model.eval()
    values = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for start in range(0, len(vids), 256):
            s = model(v_enc_src["input_ids"][start:start + 256].to(device), v_enc_src["attention_mask"][start:start + 256].to(device)).float()
            c = model(v_enc_cnd["input_ids"][start:start + 256].to(device), v_enc_cnd["attention_mask"][start:start + 256].to(device)).float()
            values.append((c - s).cpu().numpy())
    delta = np.concatenate(values)
    predictions = {rid: float(delta[i]) for i, rid in enumerate(vids)}
    observations = ev.load_observations([CANONICAL_GSE114002], validation_ids)
    metrics = ev.evaluate(observations, predictions, 10)

    torch.save({
        "schema_version": "route_a_v3_route2_mrnabert_step2_task_adapt.v1",
        "model_state_dict": {k: v for k, v in model.state_dict().items()},
        "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
        "seed": SEED, "passes": PASSES,
        "step1_meta": step1_meta,
    }, OUT_DIR / "step2_checkpoint.pt")

    report = {
        "schema_version": "route_a_v3_route2_mrnabert_step2_task_adapt.v1",
        "mode": "ROUTE_A_STEP2_TASK_ADAPT_FROZEN_DELTA",
        "step1_init": str(STEP1_CHECKPOINT),
        "trainable_parameter_count": trainable_count,
        "lora_merged_linears": merged,
        "budget": {"passes": PASSES, "updates_per_pass": updates_per_pass, "total_updates": total_steps, "batch": BATCH},
        "pass_history": history,
        "metrics": {
            "task_macro_spearman": metrics.get("task_macro_spearman"),
            "top_1": metrics.get("source_macro_top_1_accuracy"),
            "ndcg_at_10": metrics.get("source_macro_ndcg_at_k"),
        },
        "reference": {
            "route_a_step1_280k": 0.2470,
            "frozen_optimus_280k": 0.3132,
            "frozen_framepool_280k": 0.2956,
            "w0_critic_from_scratch": 0.1987,
            "w1_lora_v5_init": 0.1486,
            "critic_v5": 0.1354,
            "optimus_arch_from_scratch": 0.0984,
        },
        "preregistered_gate": "paired bootstrap CI vs W0 0.1987 must exclude zero (adjudicated separately)",
    }
    (OUT_DIR / "task_adapt_results.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    with (OUT_DIR / "predictions.jsonl").open("w") as handle:
        for rid in vids:
            handle.write(json.dumps({"canonical_record_id": rid, "predicted_direction_normalized_delta": predictions[rid]}) + "\n")
    print(json.dumps(report["metrics"], indent=1))
    print("wrote", OUT_DIR / "task_adapt_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
