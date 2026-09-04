#!/usr/bin/env python3
"""Route 2 diagnosis: two-stage frozen-embedding arm (Arm B), per task.

Phase 0 Task 3 (docs/paper/route2_mrnabert_directft_diag_prereg_v1.md §3).
Loads the FINAL-PASS-8-FIXED Arm A checkpoint of the task, freezes it, and
fits a light predictor on pooled embeddings of TRAIN (+ evaluates on
VALIDATION):

- Poolings (both computed in one forward pass):
  1. masked-mean-pool (same structure as the Arm A readout);
  2. edit-centered pooling: mean of hidden states over the union of +/-16
     token windows around source_relative_edits positions (per-nucleotide
     word-level tokenizer verified: token index = nucleotide index + 1 for
     [CLS]). Rows lacking edits fall back to masked-mean and are counted.
- Features: [e_source; e_candidate; e_source - e_candidate] (2304-d),
  StandardScaler fitted on TRAIN (fixed).
- Predictors (sklearn, fixed hyperparameters): ridge / mlp / gbdt.
- Grid = {pooling} x {predictor} = 6 configs. Selection rule (preregistered):
  MRL/polyA -> highest VALIDATION task_macro_spearman; MPRAU -> highest
  VALIDATION variant pair-mean rho (same caliber as the Arm A main criterion).
  The reported VALIDATION numbers of the SELECTED config are grid maxima and
  therefore selection-biased (declared in the report); Arm A has no such bias.

MPRAU caliber: variant pair-mean rho (record_id minus ":context:", >=2
contexts); paired bootstrap vs V5 multi-task 0.1025 (2,000 iters, seed
20260816). Pooled pair_mean_spearman is never used.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

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
SEED = 20260907
EDIT_WINDOW_RADIUS = 16
K = 10
BOOT_ITERS = 2000
BOOT_SEED = 20260816

TASKS = {
    "mrl_gse114002": {"study": "GSE114002", "canonical": MNT / "canonical/GSE114002/v1/canonical_records.private.jsonl"},
    "polya_gse269595": {"study": "GSE269595", "canonical": MNT / "canonical/GSE269595/v1/canonical_records.private.jsonl"},
    "mprau_encsr854ruf": {"study": "ENCSR854RUF", "canonical": MNT / "canonical/ENCSR854RUF/v1/canonical_records.private.jsonl"},
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
        raise NotImplementedError("Arm B extracts pooled embeddings, not head outputs")


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
    return float(spearmanr(t, p).statistic)


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


def edit_pool_mask(row: dict, seq_len: int, token_count: int) -> torch.Tensor | None:
    """Boolean token mask over the +/-radius window union around edit positions."""
    edits = row.get("source_relative_edits") or []
    positions = []
    for edit in edits:
        try:
            positions.append(int(edit["position"]))
        except (KeyError, TypeError, ValueError):
            return None
    if not positions:
        return None
    nuc_count = token_count - 2  # [CLS] ... [SEP]
    window = set()
    for position in positions:
        for offset in range(-EDIT_WINDOW_RADIUS, EDIT_WINDOW_RADIUS + 1):
            index = position + offset
            if 0 <= index < nuc_count:
                window.add(index)
    if not window:
        return None
    mask = torch.zeros(token_count, dtype=torch.bool)
    for index in window:
        mask[index + 1] = True  # [CLS] offset
    return mask


def extract_embeddings(model, tokenizer, rows, device, tag: str):
    """Return (masked_mean, edit_centered, edit_fallback_count) arrays [N, 768] x2."""
    sequences_src = [row["source_sequence"] for row in rows]
    sequences_cnd = [row["candidate_sequence"] for row in rows]
    pooled = {"masked_mean": {"src": [], "cnd": []}, "edit_centered": {"src": [], "cnd": []}}
    fallback_counts = {"src": 0, "cnd": 0}
    verified_alignment = False
    for side, sequences in (("src", sequences_src), ("cnd", sequences_cnd)):
        for start in range(0, len(sequences), 256):
            chunk = sequences[start : start + 256]
            encoded = tokenizer([format_sequence(s) for s in chunk], add_special_tokens=True, padding=True, truncation=False, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            if not verified_alignment:
                tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].tolist())
                expected = format_sequence(chunk[0]).split(" ")
                body = tokens[1 : 1 + len(expected)]
                if body != expected:
                    raise SystemExit(f"tokenizer alignment check failed ({tag}/{side}): {body[:8]} vs {expected[:8]}")
                verified_alignment = True
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden = model.base(input_ids=input_ids, attention_mask=attention_mask)[0].float()
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled["masked_mean"][side].append(mean.cpu())
            edit_rows = []
            for offset, row in enumerate(rows[start : start + 256]):
                token_count = int(attention_mask[offset].sum().item())
                edit_rows.append(edit_pool_mask(row, len(row["source_sequence"]), token_count))
            edit_stack = []
            for offset, emask in enumerate(edit_rows):
                if emask is None:
                    edit_stack.append(mean[offset])
                    fallback_counts[side] += 1
                else:
                    emask = emask.to(hidden.device)
                    pooled_vec = (hidden[offset] * emask.unsqueeze(-1)).sum(dim=0) / emask.sum().clamp(min=1)
                    edit_stack.append(pooled_vec.float().cpu() if pooled_vec.is_cuda else pooled_vec.float())
            pooled["edit_centered"][side].append(torch.stack([t.detach().cpu().float() for t in edit_stack]))
            if start % 2560 == 0:
                print(f"  embedding {tag}/{side}: {start + len(chunk)}/{len(sequences)}", flush=True)
    return (
        torch.cat(pooled["masked_mean"]["src"]).numpy(),
        torch.cat(pooled["masked_mean"]["cnd"]).numpy(),
        torch.cat(pooled["edit_centered"]["src"]).numpy(),
        torch.cat(pooled["edit_centered"]["cnd"]).numpy(),
        fallback_counts,
    )


def build_features(src: np.ndarray, cnd: np.ndarray) -> np.ndarray:
    return np.concatenate([src, cnd, src - cnd], axis=1)


def make_predictor(kind: str):
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0)
    if kind == "mlp":
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(hidden_layer_sizes=(256, 64), early_stopping=True, max_iter=500, random_state=SEED)
    if kind == "gbdt":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=SEED)
    raise ValueError(kind)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--arm-a-checkpoint", type=Path, default=None,
                        help="Arm A final checkpoint (default: formal OUT_ROOT path)")
    parser.add_argument("--smoke", action="store_true", help="tiny dry-run; writes /tmp only")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU required (BF16-only discipline)")
    device = torch.device(f"cuda:{args.physical_gpu_index}")
    torch.manual_seed(SEED)

    task_cfg = TASKS[args.task]
    study = task_cfg["study"]
    if args.smoke:
        out_dir = Path(f"/tmp/directft_smoke/{args.task}_arm_b")
        train_limit, val_limit = 64, 48
    else:
        out_dir = OUT_ROOT / f"{args.task}_arm_b"
        train_limit, val_limit = None, None
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.arm_a_checkpoint or (Path(f"/tmp/directft_smoke/{args.task}_arm_a/arm_a_checkpoint.pt") if args.smoke else OUT_ROOT / f"{args.task}_arm_a" / "arm_a_checkpoint.pt")
    if not checkpoint.exists():
        raise SystemExit(f"Arm A checkpoint not found: {checkpoint}")

    # ---- data ----
    train_rows = load_projection_rows(study, "TRAIN", limit=train_limit)
    val_rows = load_projection_rows(study, "VALIDATION", limit=val_limit)
    print(f"task={args.task} train_rows={len(train_rows)} validation_rows={len(val_rows)} checkpoint={checkpoint}")

    # ---- model: load frozen Arm A final state ----
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    import os as _os
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(MRNABERT_PATH, local_files_only=True)
    model_config = AutoConfig.from_pretrained(MRNABERT_PATH, local_files_only=True, trust_remote_code=True)
    base = AutoModel.from_config(model_config, trust_remote_code=True, add_pooling_layer=False)
    modeling_module = sys.modules[base.__class__.__module__]
    modeling_module.flash_attn_qkvpacked_func = None
    wrapped = wrap_lora(base)
    if wrapped != 48:
        raise SystemExit(f"expected 48 LoRA-wrapped Linears, got {wrapped}")
    model = MeanPoolRegressor(base, base.config.hidden_size)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    print(f"loaded Arm A checkpoint (seed {payload.get('seed')}, passes {payload.get('passes')})")

    # ---- embeddings ----
    tr_mean_src, tr_mean_cnd, tr_edit_src, tr_edit_cnd, tr_fallback = extract_embeddings(model, tokenizer, train_rows, device, "train")
    va_mean_src, va_mean_cnd, va_edit_src, va_edit_cnd, va_fallback = extract_embeddings(model, tokenizer, val_rows, device, "validation")

    train_targets = np.asarray([float(r["direction_normalized_delta"]) for r in train_rows], dtype=np.float64)
    val_targets = np.asarray([float(r["direction_normalized_delta"]) for r in val_rows], dtype=np.float64)
    val_ids = [str(r["canonical_record_id"]) for r in val_rows]
    val_target_map = {rid: float(t) for rid, t in zip(val_ids, val_targets)}

    validation_ids = manifest_validation_ids(study) if not args.smoke else set(val_ids)
    observations = ev.load_observations([Path(task_cfg["canonical"])], validation_ids)

    # ---- grid ----
    from sklearn.preprocessing import StandardScaler

    grid = []
    for pooling, (tr_src, tr_cnd, va_src, va_cnd) in {
        "masked_mean": (tr_mean_src, tr_mean_cnd, va_mean_src, va_mean_cnd),
        "edit_centered": (tr_edit_src, tr_edit_cnd, va_edit_src, va_edit_cnd),
    }.items():
        x_train = build_features(tr_src, tr_cnd)
        x_val = build_features(va_src, va_cnd)
        scaler = StandardScaler().fit(x_train)
        x_train_s = scaler.transform(x_train)
        x_val_s = scaler.transform(x_val)
        for predictor_kind in ("ridge", "mlp", "gbdt"):
            predictor = make_predictor(predictor_kind)
            predictor.fit(x_train_s, train_targets)
            val_pred = predictor.predict(x_val_s)
            preds = {rid: float(v) for rid, v in zip(val_ids, val_pred)}
            entry: dict = {"pooling": pooling, "predictor": predictor_kind}
            if study == "ENCSR854RUF":
                variants = mprau_variant_table(preds, val_target_map)
                entry["variant_pair_mean_spearman"] = pair_mean_rho(variants) if variants else None
                entry["variant_count"] = len(variants)
            if observations is not None:
                sub = {k: v for k, v in preds.items() if k in validation_ids}
                if len(sub) == len(validation_ids):
                    metrics = ev.evaluate(observations, sub, K)
                    entry["task_macro_spearman"] = metrics.get("task_macro_spearman")
                    entry["top_1"] = metrics.get("source_macro_top_1_accuracy")
                    entry["ndcg_at_10"] = metrics.get("source_macro_ndcg_at_k")
            grid.append(entry)
            print(f"config {pooling}/{predictor_kind}: {json.dumps({k: v for k, v in entry.items() if k not in ('pooling', 'predictor')})}", flush=True)

    selection_key = "variant_pair_mean_spearman" if study == "ENCSR854RUF" else "task_macro_spearman"
    eligible = [e for e in grid if e.get(selection_key) is not None]
    selected = max(eligible, key=lambda e: e[selection_key]) if eligible else None

    # ---- re-run selected config to persist predictions + MPRAU bootstrap ----
    predictions_out = None
    mprau_bootstrap = None
    if selected is not None:
        tr_src, tr_cnd, va_src, va_cnd = {
            "masked_mean": (tr_mean_src, tr_mean_cnd, va_mean_src, va_mean_cnd),
            "edit_centered": (tr_edit_src, tr_edit_cnd, va_edit_src, va_edit_cnd),
        }[selected["pooling"]]
        x_train = build_features(tr_src, tr_cnd)
        x_val = build_features(va_src, va_cnd)
        scaler = StandardScaler().fit(x_train)
        predictor = make_predictor(selected["predictor"])
        predictor.fit(scaler.transform(x_train), train_targets)
        val_pred = predictor.predict(scaler.transform(x_val))
        predictions_out = {rid: float(v) for rid, v in zip(val_ids, val_pred)}
        if study == "ENCSR854RUF":
            variants = mprau_variant_table(predictions_out, val_target_map)
            mprau_bootstrap = paired_bootstrap_vs_v5(load_v5_variants(), variants)

    report = {
        "schema_version": "route_a_v3_route2_mrnabert_directft_arm_b.v1",
        "mode": "ROUTE_A_DIRECTFT_ARM_B_FROZEN_EMBEDDING",
        "task": args.task,
        "arm_a_checkpoint": str(checkpoint),
        "smoke": bool(args.smoke),
        "feature_layout": "[e_source; e_candidate; e_source - e_candidate] (2304-d), StandardScaler fit on TRAIN",
        "edit_window_radius": EDIT_WINDOW_RADIUS,
        "edit_fallback_counts": {"train": tr_fallback, "validation": va_fallback},
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "selection_rule": (f"max VALIDATION {selection_key} over 6 configs" if selection_key != "variant_pair_mean_spearman"
                           else "max VALIDATION variant pair-mean rho over 6 configs"),
        "selection_bias_warning": "Arm B VALIDATION numbers are grid maxima over 6 configs (selection-biased); Arm A is not",
        "grid": grid,
        "selected": selected,
        "mprau_paired_bootstrap_vs_v5_selected": mprau_bootstrap,
    }
    (out_dir / "arm_b_results.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    if predictions_out is not None:
        val_row_by_id = {str(r["canonical_record_id"]): r for r in val_rows}
        with (out_dir / "predictions.jsonl").open("w") as handle:
            for rid in val_ids:
                if study == "ENCSR854RUF":
                    handle.write(json.dumps({
                        "record_id": rid, "prediction": predictions_out[rid], "target": val_target_map[rid],
                        "source_group_id": val_row_by_id[rid]["source_group_id"],
                        "task_id": val_row_by_id[rid]["task_id"],
                    }) + "\n")
                else:
                    handle.write(json.dumps({"canonical_record_id": rid, "predicted_direction_normalized_delta": predictions_out[rid]}) + "\n")
    print(json.dumps({"selected": selected}, indent=1))
    print("wrote", out_dir / "arm_b_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
