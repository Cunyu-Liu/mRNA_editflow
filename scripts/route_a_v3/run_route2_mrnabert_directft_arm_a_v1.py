#!/usr/bin/env python3
"""Route 2 diagnosis: mRNABERT raw-pretrained-init direct task fine-tuning (Arm A).

Phase 0 Task 3 diagnostic experiment (docs/paper/route2_mrnabert_directft_diag_prereg_v1.md).
Fills the only untested evidence-matrix cell: mRNABERT ORIGINAL pretrained
weights init + task-data LoRA fine-tuning (no 280K pre-finetuning stage).

Per task (mrl_gse114002 / polya_gse269595 / mprau_encsr854ruf), single-task:
- Init: raw mRNABERT pytorch_model.bin via from_config + manual strip of the
  "bert." prefix + flash_attn_qkvpacked_func=None (first-launch lesson:
  AutoModel.from_pretrained is incompatible with the custom ALiBi).
- Fresh LoRA r16 a32 dropout 0.05 on all 12 layers x 4 targets (Wqkv /
  attn-out.dense / mlp.gated_layers / mlp.wo = 48 Linears) + masked-mean-pool
  + fresh linear head. Trainable = LoRA + head only.
- Objective: direction_normalized_delta regression/ranking, frozen-delta
  structure (prediction = f(candidate) - f(source)); loss fully mirrors the
  Step-2 recipe: huber(delta=1.0) + cross-source pairwise softplus +
  soft-Spearman (temp 0.2) + within-source 0.5; pass 1-2 {1.0, 0.25, 0.0} /
  pass 3-8 {1.0, 0.5, 0.25}.
- Budget: 8 passes, batch 32, LoRA lr 1e-4 / head lr 2e-4, AdamW wd 1e-4,
  cosine to 10% after 5% warmup, grad clip 1.0, BF16 autocast, seed 20260907
  (W series). Selection = FINAL-PASS-8-FIXED (no peak-picking); per-pass
  VALIDATION metrics are logged for the record only.

Calibers (identical to historical rows):
- MRL/polyA: frozen Task-1 evaluator K=10 (task_macro_spearman +
  source_macro_top_1_accuracy + source_macro_ndcg_at_k).
- MPRAU MAIN criterion: variant pair-mean rho (record_id minus ":context:"
  suffix, >=2 contexts, per-variant means), plus paired bootstrap vs V5
  multi-task 0.1025 (2,000 iters, seed 20260816). The pooled run_summary
  pair_mean_spearman is NEVER used as the criterion.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import sys
from collections import defaultdict
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

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
MRNABERT_PATH = MNT / "external_model_assets/mrnabert_a1eb7df25804d23f08646e1cb996b234d7208a40"
PROJECTION_DIR = MNT / "projections/xedit_v3/development_train_validation_v1"
MANIFEST = MNT / "manifests/route2_development_frozen_v1/development_manifest.jsonl"
V5_PRED_GLOB = str(MNT / "experiments/xeditcritic_v5/*/v5_full/final_validation_predictions.jsonl")
OUT_ROOT = MNT / "experiments/xeditcritic_route_a/directft_diag_20260904"

LORA_RANK = 16
LORA_ALPHA = 32.0
LORA_DROPOUT = 0.05
BATCH = 32
PASSES = 8
LORA_LR = 1e-4
HEAD_LR = 2e-4
WEIGHT_DECAY = 1e-4
SEED = 20260907
HUBER_DELTA = 1.0
SOFT_RANK_TEMPERATURE = 0.2
WITHIN_SOURCE_WEIGHT = 0.5
GRAD_CLIP = 1.0
K = 10
BOOT_ITERS = 2000
BOOT_SEED = 20260816
MPRAU_ROWS_PER_PASS = 12048  # registered budget: ceil(12048/32)=377 updates/pass

TASKS = {
    "mrl_gse114002": {
        "study": "GSE114002",
        "canonical": MNT / "canonical/GSE114002/v1/canonical_records.private.jsonl",
        "expected_train": 2443,
        "expected_validation": 730,
        "reference": {
            "w0_from_scratch": 0.1987,
            "w1_lora_v5_init": 0.1486,
            "route_a_step2_280k": 0.2159,
            "route_a_zeroshot_280k": 0.2470,
            "frozen_framepool_280k": 0.2956,
            "frozen_optimus_280k": 0.3132,
        },
    },
    "polya_gse269595": {
        "study": "GSE269595",
        "canonical": MNT / "canonical/GSE269595/v1/canonical_records.private.jsonl",
        "expected_train": 25710,
        "expected_validation": 2628,
        "reference": {
            "w0_polya_spearman": 0.8142,
            "aparent_adapter_top_1": 0.6011,
            "aparent_adapter_ndcg10": 0.8906,
        },
    },
    "mprau_encsr854ruf": {
        "study": "ENCSR854RUF",
        "canonical": MNT / "canonical/ENCSR854RUF/v1/canonical_records.private.jsonl",
        "expected_train": 55704,  # 9,284 variants x 6 contexts (registered budget subsamples per pass)
        "expected_validation": 12048,  # 2,008 variants x 6 contexts
        "reference": {
            "v5_multitask_pair_mean": 0.1025,
            "w_ladder_arms_pair_mean": "0.0510-0.0883",
            "ceiling": 0.683,
        },
    },
}


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

    def pooled(self, input_ids, attention_mask):
        hidden = self.base(input_ids=input_ids, attention_mask=attention_mask)[0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


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


def load_projection_rows(study: str, split: str, limit: int | None = None) -> list[dict]:
    path = PROJECTION_DIR / f"{split.lower()}.jsonl"
    rows = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("study_unit_id") == study and row.get("split") == split:
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def manifest_validation_ids(study: str) -> set[str]:
    ids = set()
    with MANIFEST.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["study_unit_id"] == study and row["split"] == "VALIDATION":
                ids.add(str(row["canonical_record_id"]))
    return ids


def mprau_variant_table(predictions: dict[str, float], targets: dict[str, float]) -> dict[str, tuple[float, float]]:
    by_variant: dict[str, list[str]] = defaultdict(list)
    for rid in predictions:
        by_variant[rid.split(":context:")[0]].append(rid)
    variants = {}
    for variant, rids in by_variant.items():
        if len(rids) >= 2:
            variants[variant] = (
                float(np.mean([targets[r] for r in rids])),
                float(np.mean([predictions[r] for r in rids])),
            )
    return variants


def pair_mean_rho(variants: dict[str, tuple[float, float]]) -> float:
    from scipy.stats import spearmanr

    t = np.asarray([v[0] for v in variants.values()], dtype=np.float64)
    p = np.asarray([v[1] for v in variants.values()], dtype=np.float64)
    value = spearmanr(t, p).statistic
    return float(value)


def paired_bootstrap_vs_v5(v5_variants, arm_variants, iters=BOOT_ITERS, seed=BOOT_SEED):
    from scipy.stats import spearmanr

    shared = sorted(set(v5_variants) & set(arm_variants))
    if len(shared) < 10:
        return {"shared_variant_count": len(shared), "skipped": "too few shared variants"}
    t = np.array([v5_variants[v][0] for v in shared])
    p5 = np.array([v5_variants[v][1] for v in shared])
    pa = np.array([arm_variants[v][1] for v in shared])
    rho5 = float(spearmanr(t, p5).statistic)
    rhoa = float(spearmanr(t, pa).statistic)
    rng = np.random.default_rng(seed)
    n = len(shared)
    deltas = []
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        try:
            r5 = spearmanr(t[idx], p5[idx]).statistic
            ra = spearmanr(t[idx], pa[idx]).statistic
            if np.isfinite(r5) and np.isfinite(ra):
                deltas.append(float(ra) - float(r5))
        except Exception:
            continue
    deltas = np.asarray(deltas)
    ci = [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]
    return {
        "shared_variant_count": n,
        "v5_pair_mean_spearman": rho5,
        "arm_pair_mean_spearman": rhoa,
        "delta_pair_mean_spearman": rhoa - rho5,
        "bootstrap_ci_95": ci,
        "bootstrap_iterations": int(len(deltas)),
        "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
        "ceiling_ratio_0_683": rhoa / 0.683,
    }


def load_v5_variants():
    paths = sorted(glob.glob(V5_PRED_GLOB))
    if not paths:
        return None
    rows = {}
    with open(paths[0]) as handle:
        for line in handle:
            r = json.loads(line)
            rows[str(r["record_id"])] = r
    by_variant = defaultdict(list)
    for rid, row in rows.items():
        if rid.startswith("ENCSR854RUF:"):
            by_variant[rid.split(":context:")[0]].append(row)
    variants = {}
    for variant, rs in by_variant.items():
        if len(rs) >= 2:
            variants[variant] = (
                float(np.mean([r["target"] for r in rs])),
                float(np.mean([r["prediction"] for r in rs])),
            )
    return variants


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--smoke", action="store_true", help="tiny dry-run; writes /tmp only")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU required (BF16-only discipline)")
    device = torch.device(f"cuda:{args.physical_gpu_index}")
    torch.manual_seed(SEED)

    task_cfg = TASKS[args.task]
    study = task_cfg["study"]
    if args.smoke:
        out_dir = Path(f"/tmp/directft_smoke/{args.task}_arm_a")
        passes = 1
        train_limit, val_limit = 64, 48
        batch = 8  # smaller smoke footprint on contended GPUs; same code path
    else:
        out_dir = OUT_ROOT / f"{args.task}_arm_a"
        passes = PASSES
        train_limit, val_limit = None, None
        batch = BATCH
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    rows = load_projection_rows(study, "TRAIN", limit=train_limit)
    if not args.smoke and len(rows) != task_cfg["expected_train"]:
        raise SystemExit(f"expected {task_cfg['expected_train']} TRAIN rows for {study}, got {len(rows)}")
    vrows = load_projection_rows(study, "VALIDATION", limit=val_limit)
    if not args.smoke and len(vrows) != task_cfg["expected_validation"]:
        raise SystemExit(f"expected {task_cfg['expected_validation']} VALIDATION rows for {study}, got {len(vrows)}")
    budget_rows = MPRAU_ROWS_PER_PASS if study == "ENCSR854RUF" else len(rows)
    if args.smoke:
        budget_rows = min(budget_rows, len(rows))
    print(f"task={args.task} train_rows={len(rows)} validation_rows={len(vrows)} "
          f"budget_rows_per_pass={budget_rows} passes={passes} batch={batch} seed={SEED}")

    # ---- model: raw pretrained init (verified loading pattern) ----
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    import os as _os
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(MRNABERT_PATH, local_files_only=True)
    model_config = AutoConfig.from_pretrained(MRNABERT_PATH, local_files_only=True, trust_remote_code=True)
    base = AutoModel.from_config(model_config, trust_remote_code=True, add_pooling_layer=False)
    modeling_module = sys.modules[base.__class__.__module__]
    modeling_module.flash_attn_qkvpacked_func = None  # official fallback backend
    checkpoint = torch.load(MRNABERT_PATH / "pytorch_model.bin", map_location="cpu", weights_only=False)
    base_state = {
        key.removeprefix("bert."): value
        for key, value in checkpoint.items()
        if key.startswith("bert.")
    }
    base.load_state_dict(base_state, strict=True)
    del checkpoint, base_state
    wrapped = wrap_lora(base)
    if wrapped != 48:
        raise SystemExit(f"expected 48 LoRA-wrapped Linears, got {wrapped}")
    model = MeanPoolRegressor(base, base.config.hidden_size)

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
    print(f"LoRA wrapped {wrapped} Linears (rank {LORA_RANK}); trainable parameters: {trainable_count:,}")
    model.to(device)

    # ---- tokenize ----
    sources = [row["source_sequence"] for row in rows]
    candidates = [row["candidate_sequence"] for row in rows]
    targets = torch.tensor([float(row["direction_normalized_delta"]) for row in rows], dtype=torch.float32)
    groups = [str(row["source_group_id"]) for row in rows]
    enc_src = tokenizer([format_sequence(s) for s in sources], add_special_tokens=True, padding=True, truncation=False, return_tensors="pt")
    enc_cnd = tokenizer([format_sequence(s) for s in candidates], add_special_tokens=True, padding=True, truncation=False, return_tensors="pt")

    vrecords = {str(row["canonical_record_id"]): row for row in vrows}
    vids = sorted(vrecords)
    v_enc_src = tokenizer([format_sequence(vrecords[rid]["source_sequence"]) for rid in vids], add_special_tokens=True, padding=True, truncation=False, return_tensors="pt")
    v_enc_cnd = tokenizer([format_sequence(vrecords[rid]["candidate_sequence"]) for rid in vids], add_special_tokens=True, padding=True, truncation=False, return_tensors="pt")
    v_targets = {rid: float(vrecords[rid]["direction_normalized_delta"]) for rid in vids}

    # ---- validation calibers ----
    validation_ids = manifest_validation_ids(study) if not args.smoke else set(vids)
    observations = ev.load_observations([Path(task_cfg["canonical"])], validation_ids)

    def predict_validation() -> dict[str, float]:
        model.eval()
        values = []
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for start in range(0, len(vids), 256):
                s = model(v_enc_src["input_ids"][start:start + 256].to(device), v_enc_src["attention_mask"][start:start + 256].to(device)).float()
                c = model(v_enc_cnd["input_ids"][start:start + 256].to(device), v_enc_cnd["attention_mask"][start:start + 256].to(device)).float()
                values.append((c - s).cpu().numpy())
        model.train()
        delta = np.concatenate(values)
        return {rid: float(delta[i]) for i, rid in enumerate(vids)}

    def validation_report(preds: dict[str, float]) -> dict:
        report: dict = {}
        if study == "ENCSR854RUF":
            variants = mprau_variant_table(preds, v_targets)
            report["variant_pair_mean_spearman"] = pair_mean_rho(variants) if variants else None
            report["variant_count"] = len(variants)
        if observations is not None:
            sub = {k: v for k, v in preds.items() if k in validation_ids}
            if len(sub) == len(validation_ids):
                metrics = ev.evaluate(observations, sub, K)
                report["task_macro_spearman"] = metrics.get("task_macro_spearman")
                report["top_1"] = metrics.get("source_macro_top_1_accuracy")
                report["ndcg_at_10"] = metrics.get("source_macro_ndcg_at_k")
            else:
                report["aligned_eval"] = f"skipped: {len(validation_ids) - len(sub)} predictions missing"
        return report

    # ---- optimizer ----
    optimizer = torch.optim.AdamW(
        [{"params": lora_params, "lr": LORA_LR}, {"params": head_params, "lr": HEAD_LR}],
        weight_decay=WEIGHT_DECAY,
    )
    updates_per_pass = (budget_rows + BATCH - 1) // BATCH
    total_steps = updates_per_pass * passes
    warmup_steps = max(1, int(total_steps * 0.05))

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)

    print(f"pass 0 (pre-training) validation: {json.dumps(validation_report(predict_validation()))}")
    history = []
    step = 0
    for pass_number in range(1, passes + 1):
        weights = loss_weights(pass_number)
        pool = torch.randperm(len(rows))
        order = pool[:budget_rows]
        losses, hubers, pairs_n, within_n = [], [], 0, 0
        for start in range(0, len(order), batch):
            idx = order[start : start + batch]
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
        val = validation_report(predict_validation())
        history.append({
            "pass": pass_number,
            "mean_loss": float(np.mean(losses)),
            "mean_huber": float(np.mean(hubers)),
            "different_source_pairs": pairs_n,
            "within_source_pairs": within_n,
            "validation": val,
        })
        print(f"pass {pass_number}/{passes}: loss {np.mean(losses):.4f} huber {np.mean(hubers):.4f} "
              f"val {json.dumps(val)} lr {scheduler.get_last_lr()[0]:.2e}", flush=True)

    # ---- final frozen-delta evaluation (FINAL-PASS-8-FIXED) ----
    model.eval()
    preds = predict_validation()
    final_metrics = validation_report(preds)

    torch.save({
        "schema_version": "route_a_v3_route2_mrnabert_directft_arm_a.v1",
        "model_state_dict": {k: v for k, v in model.state_dict().items()},
        "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
        "seed": SEED, "passes": passes, "task": args.task,
        "train_rows": len(rows), "budget_rows_per_pass": budget_rows,
    }, out_dir / "arm_a_checkpoint.pt")

    mprau_bootstrap = None
    if study == "ENCSR854RUF":
        variants = mprau_variant_table(preds, v_targets)
        mprau_bootstrap = paired_bootstrap_vs_v5(load_v5_variants(), variants)

    report = {
        "schema_version": "route_a_v3_route2_mrnabert_directft_arm_a.v1",
        "mode": "ROUTE_A_DIRECTFT_ARM_A_FROZEN_DELTA",
        "task": args.task,
        "init": "mRNABERT raw pretrained weights (no 280K pre-finetune, no merge)",
        "smoke": bool(args.smoke),
        "trainable_parameter_count": trainable_count,
        "lora_wrapped_linears": wrapped,
        "budget": {
            "passes": passes,
            "updates_per_pass": updates_per_pass,
            "total_updates": total_steps,
            "batch": batch,
            "train_pool_rows": len(rows),
            "budget_rows_per_pass": budget_rows,
            "mprau_subsampling": "uniform without replacement per pass" if budget_rows < len(rows) else None,
        },
        "seed": SEED,
        "selection_rule": "FINAL-PASS-8-FIXED_NO_VALIDATION_PEAK_RESELECTION",
        "pass_history": history,
        "final_metrics": final_metrics,
        "reference": task_cfg["reference"],
        "mprau_paired_bootstrap_vs_v5": mprau_bootstrap,
        "mpooled_pair_mean_warning": "run_summary pooled pair_mean_spearman is never the criterion" if study == "ENCSR854RUF" else None,
    }
    (out_dir / "arm_a_results.json").write_text(json.dumps(report, indent=1, sort_keys=True))

    with (out_dir / "predictions.jsonl").open("w") as handle:
        for rid in vids:
            if study == "ENCSR854RUF":
                handle.write(json.dumps({
                    "record_id": rid, "prediction": preds[rid], "target": v_targets[rid],
                    "source_group_id": vrecords[rid]["source_group_id"],
                    "task_id": vrecords[rid]["task_id"],
                }) + "\n")
            else:
                handle.write(json.dumps({"canonical_record_id": rid, "predicted_direction_normalized_delta": preds[rid]}) + "\n")
    print(json.dumps(final_metrics, indent=1))
    print("wrote", out_dir / "arm_a_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
