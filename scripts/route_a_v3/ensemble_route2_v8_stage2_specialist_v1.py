#!/usr/bin/env python3
"""V8 Stage 2 specialist 3-seed ensemble + paired bootstrap vs V5 (MPRAU).

Loads the FINAL-EPOCH checkpoint of each seed of a specialist arm (e.g.
s_mprau_in, h_mprau_lora), scores MPRAU VALIDATION (2,008 variants, pair-mean
calibre), averages per-variant predictions across seeds, and computes the
paired bootstrap delta vs V5 (2,000 iters, seed 20260816).

Prereg: route2_v8_stage2_prereg_v1.md (MPRAU primary > 0.1025, CI not crossing
zero). Selection = FINAL-EPOCH-FIXED only; smoke/proxy rejected.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_v8_hybrid_backbone_v1 import (  # noqa: E402
    CELL_IDS,
    DOMAIN_IDS,
    NUM_DOMAINS,
    build_v8_regressor,
)
from core.route2_v8_joint_library_v1 import (  # noqa: E402
    MNT,
    format_sequence,
)

EVAL_REPO = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
_ev_spec = importlib.util.spec_from_file_location(
    "ev", EVAL_REPO + "/scripts/route_a_v3/evaluate_route2_prediction_v1.py"
)
ev = importlib.util.module_from_spec(_ev_spec)
sys.modules["ev"] = ev
_ev_spec.loader.exec_module(ev)

MRNABERT_PATH = MNT / "external_model_assets/mrnabert_a1eb7df25804d23f08646e1cb996b234d7208a40"
MANIFEST = MNT / "manifests/route2_development_frozen_v1/development_manifest.jsonl"
MPRAU_CANON = MNT / "canonical/ENCSR854RUF/v1/canonical_records.private.jsonl"
V5_PRED_GLOB = str(MNT / "experiments/xeditcritic_v5/*/v5_full/final_validation_predictions.jsonl")
OUT_ROOT = MNT / "experiments/xeditcritic_route_a/v8_stage2_adapt_20260907"
BOOT_ITERS = 2000
BOOT_SEED = 20260816
EVAL_BATCH = 256
V5_MPRAU_REFERENCE = 0.1025

# arm -> seed list (dir suffix naming)
ARM_SEEDS = {
    "s_mprau_in": ["", "_s2", "_s3", "_s4", "_s5"],
    "h_mprau_in": ["", "_s2"],
    "h_mprau_lora": ["", "_s2"],
}


def load_final_checkpoint(arm_dir: Path, arch: str) -> Path:
    ckpt = arm_dir / f"stage2_{arch}_benchmark_full_epoch6.pt"
    if not ckpt.is_file():
        ckpt = arm_dir / f"stage2_{arch}_benchmark_lora_epoch6.pt"
    if not ckpt.is_file():
        cands = sorted(arm_dir.glob("stage2_*.pt"))
        if not cands:
            raise SystemExit(f"no checkpoint in {arm_dir}")
        # prefer the highest-epoch FINAL (epoch6)
        ckpt = sorted(cands, key=lambda p: int(str(p).rsplit("epoch", 1)[1].split(".")[0]))[-1]
    return ckpt


def score_validation(model, tokenizer, device) -> dict[str, float]:
    validation_ids = set()
    with MANIFEST.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["study_unit_id"] == "ENCSR854RUF" and row["split"] == "VALIDATION":
                validation_ids.add(str(row["canonical_record_id"]))
    records = {}
    with MPRAU_CANON.open() as handle:
        for line in handle:
            row = json.loads(line)
            rid = str(row["canonical_record_id"])
            if rid in validation_ids:
                records[rid] = row
    ids = sorted(records)
    cells = torch.tensor(
        [CELL_IDS.get(str(records[rid].get("biological_context_id")), 0) for rid in ids], dtype=torch.long)
    domain_id = DOMAIN_IDS["mprau"]
    model.eval()
    values = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for start in range(0, len(ids), EVAL_BATCH):
            chunk = ids[start:start + EVAL_BATCH]
            enc = tokenizer(
                [format_sequence(str(records[r]["source_sequence"])) for r in chunk],
                add_special_tokens=True, padding=True, truncation=True, max_length=512, return_tensors="pt",
            )
            dom = torch.full((len(chunk),), domain_id, dtype=torch.long, device=device)
            c = cells[start:start + len(chunk)].to(device)
            values.append(model(enc["input_ids"].to(device), enc["attention_mask"].to(device), dom, c).float().cpu().numpy())
    source = np.concatenate(values)
    values = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for start in range(0, len(ids), EVAL_BATCH):
            chunk = ids[start:start + EVAL_BATCH]
            enc = tokenizer(
                [format_sequence(str(records[r]["candidate_sequence"])) for r in chunk],
                add_special_tokens=True, padding=True, truncation=True, max_length=512, return_tensors="pt",
            )
            dom = torch.full((len(chunk),), domain_id, dtype=torch.long, device=device)
            c = cells[start:start + len(chunk)].to(device)
            values.append(model(enc["input_ids"].to(device), enc["attention_mask"].to(device), dom, c).float().cpu().numpy())
    candidate = np.concatenate(values)
    delta = candidate - source
    return {rid: float(delta[i]) for i, rid in enumerate(ids)}


def variant_table(predictions: dict[str, float], targets: dict[str, float]) -> dict[str, tuple[float, float]]:
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


def paired_bootstrap_vs_reference(ref_variants, arm_variants, label: str) -> dict:
    shared = sorted(set(ref_variants) & set(arm_variants))
    if len(shared) < 50:
        return {"skipped": "too few shared variants", "shared_variant_count": len(shared)}
    t = np.asarray([ref_variants[v][0] for v in shared], dtype=np.float64)
    pr = np.asarray([ref_variants[v][1] for v in shared], dtype=np.float64)
    pa = np.asarray([arm_variants[v][1] for v in shared], dtype=np.float64)

    def rho(x, y):
        return float(ev.numeric_metrics(x, y)["spearman"])

    rng = np.random.default_rng(BOOT_SEED)
    base_r, base_a = rho(t, pr), rho(t, pa)
    deltas = []
    n = len(shared)
    for _ in range(BOOT_ITERS):
        idx = rng.integers(0, n, size=n)
        deltas.append(rho(t[idx], pa[idx]) - rho(t[idx], pr[idx]))
    ci = np.percentile(deltas, [2.5, 97.5])
    return {
        "reference": label, "shared_variant_count": len(shared),
        "reference_pair_mean_spearman": base_r, "arm_pair_mean_spearman": base_a,
        "delta_pair_mean_spearman": base_a - base_r,
        "delta_ci95": [float(ci[0]), float(ci[1])], "crosses_zero": bool(ci[0] < 0 < ci[1]),
        "bootstrap_iters": BOOT_ITERS, "bootstrap_seed": BOOT_SEED,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, choices=sorted(ARM_SEEDS))
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU required")
    device = torch.device(f"cuda:{args.physical_gpu_index}")
    root = Path(args.out_root)
    arch = args.group.split("_")[0]  # s_mprau_in -> s ; h_mprau_* -> h

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MRNABERT_PATH, local_files_only=True)

    # targets for MPRAU VALIDATION
    targets = {}
    with MPRAU_CANON.open() as handle:
        for line in handle:
            r = json.loads(line)
            targets[str(r["canonical_record_id"])] = float(r["direction_normalized_delta"])

    # V5 reference variants
    v5_paths = sorted(__import__("glob").glob(V5_PRED_GLOB))
    v5_preds = {}
    if v5_paths:
        with open(v5_paths[0]) as handle:
            for line in handle:
                r = json.loads(line)
                rid = str(r.get("canonical_record_id") or r.get("record_id"))
                if rid in targets:
                    v5_preds[rid] = float(r["prediction"])
    ref_variants = variant_table(v5_preds, targets)

    seeds = []
    per_variant = None
    for suffix in ARM_SEEDS[args.group]:
        arm_dir = root / f"{args.group}{suffix}"
        ckpt = load_final_checkpoint(arm_dir, arch)
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = raw["model_state_dict"]
        model = build_v8_regressor(MRNABERT_PATH, arch, num_domains=len(DOMAIN_IDS), num_cells=max(CELL_IDS.values()) + 1)
        model_state = model.state_dict()
        compatible = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
        model.load_state_dict(compatible, strict=False)
        dom = state.get("domain_embeddings.weight")
        if dom is not None:
            n = min(dom.shape[0], model.domain_embeddings.weight.shape[0])
            model.domain_embeddings.weight.data[:n] = dom[:n]
        model.to(device)
        preds = score_validation(model, tokenizer, device)
        table = variant_table(preds, targets)
        rho = float(ev.numeric_metrics(
            [table[v][0] for v in sorted(table)], [table[v][1] for v in sorted(table)])["spearman"])
        seeds.append({"seed_dir": arm_dir.name, "checkpoint": ckpt.name,
                      "mprau_pair_mean": rho, "n_variants": len(table)})
        print(f"seed {suffix or '(seed1)'}: rho={rho:.4f} variants={len(table)} ckpt={ckpt.name}", flush=True)
        # accumulate per-variant predictions (mean across seeds of per-cell means)
        if per_variant is None:
            per_variant = {v: [p[1]] for v, p in table.items()}
        else:
            for v, p in table.items():
                if v in per_variant:
                    per_variant[v].append(p[1])
    # variant-level target (mean of per-cell observed deltas) for the ensemble table
    variant_targets = {v: t[0] for v, t in ref_variants.items()}
    # ensemble = mean of per-variant predictions across seeds
    ens_table = {v: (variant_targets[v], float(np.mean(ps))) for v, ps in per_variant.items() if v in variant_targets}
    ens_rho = float(ev.numeric_metrics([ens_table[v][0] for v in sorted(ens_table)],
                                       [ens_table[v][1] for v in sorted(ens_table)])["spearman"])
    vs_v5 = paired_bootstrap_vs_reference(ref_variants, ens_table, "v5_multitask")
    report = {
        "schema_version": "route_a_v3_route2_v8_stage2_specialist_ensemble.v1",
        "group": args.group, "seeds": seeds, "n_seeds": len(seeds),
        "ensemble_mprau_pair_mean": ens_rho, "ensemble_n_variants": len(ens_table),
        "vs_v5": vs_v5,
        "gate": {"mprau_ref": V5_MPRAU_REFERENCE,
                 "pass": bool(ens_rho > V5_MPRAU_REFERENCE and not vs_v5.get("crosses_zero", True))},
    }
    out = root / f"ensemble_{args.group}.json"
    out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
