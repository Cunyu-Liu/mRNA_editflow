#!/usr/bin/env python3
"""Saluki frozen zero-shot delta evaluation on ENCSR854RUF / MPRAU (weak-control row, protocol v1 §4-5).

Protocol scope: Saluki-delta is a WEAK CONTROL for MPRAU (3'UTR degradation
caliber != allelic-skew caliber; preregistered caliber-difference annotation).
Frozen mode: official Zenodo Saluki checkpoints, no fine-tuning. For each
VALIDATION record of ENCSR854RUF, score source and candidate sequences with
the full official fold ensemble (mean over all model{0,1}_best.h5), delta =
candidate - source, then:

1. Frozen Task-1 evaluator metrics (K=10) - leaderboard row caliber.
2. MPRAU variant pair-mean rho (same caliber as the W-ladder adjudication:
   variants = rid before ":context:", >=2 contexts, per-variant means of
   direction_normalized_delta vs predicted delta, Spearman across variants)
   - directly comparable to V5 reference 0.1025.
3. Unpaired bootstrap CI (2,000 iters, seed 20260816) for the pair-mean rho.

UTR-window caveat (native-truncation clause): Saluki consumes full spliced
mRNA transcripts (12,288 right-padded); ENCSR854RUF supplies 133bp MPRA
oligo windows. Reported as such.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from scipy.stats import spearmanr

EVAL_REPO = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
sys.path.insert(0, EVAL_REPO + "/scripts/route_a_v3")

import importlib.util

_ev_spec = importlib.util.spec_from_file_location(
    "ev", EVAL_REPO + "/scripts/route_a_v3/evaluate_route2_prediction_v1.py"
)
ev = importlib.util.module_from_spec(_ev_spec)
_ev_spec.loader.exec_module(ev)

from core.route2_saluki_port_v1 import SalukiGRUV1, encode_saluki_six_channel_v1  # noqa: E402

SALUKI_ROOT = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/saluki/datasets/deeplearning/train_gru"
)
MANIFEST = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/manifests/route2_development_frozen_v1/development_manifest.jsonl"
)
CANONICAL = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/canonical/ENCSR854RUF/v1/canonical_records.private.jsonl"
)
SALUKI_FULL_LENGTH = 12288
BATCH = 512
K = 10
BOOT_ITERS = 2000
BOOT_SEED = 20260816


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--limit-models", type=int, default=0, help="0 = all checkpoints")
    parser.add_argument("--output-dir", type=Path, default=Path(
        "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_saluki_frozen_mprau_20260903"
    ))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU required")
    device = torch.device(f"cuda:{args.physical_gpu_index}")

    validation_ids = set()
    with MANIFEST.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["study_unit_id"] == "ENCSR854RUF" and row["split"] == "VALIDATION":
                validation_ids.add(str(row["canonical_record_id"]))

    records = {}
    with CANONICAL.open() as handle:
        for line in handle:
            row = json.loads(line)
            rid = str(row.get("canonical_record_id"))
            if rid in validation_ids:
                records[rid] = row
    if set(records) != validation_ids:
        raise SystemExit(
            f"coverage mismatch: {len(records)} loaded vs {len(validation_ids)} manifest ids"
        )

    ids = sorted(records)
    sources = [records[rid]["source_sequence"] for rid in ids]
    candidates = [records[rid]["candidate_sequence"] for rid in ids]
    encoded = np.stack(
        [encode_saluki_six_channel_v1(seq, SALUKI_FULL_LENGTH) for seq in sources + candidates]
    )
    tensor_all = torch.from_numpy(encoded).to(device)
    del encoded

    checkpoint_paths = sorted(SALUKI_ROOT.glob("f*_c*/train/model*_best.h5"))
    if args.limit_models:
        checkpoint_paths = checkpoint_paths[: args.limit_models]

    source_scores = np.zeros(len(ids), dtype=np.float64)
    candidate_scores = np.zeros(len(ids), dtype=np.float64)
    for index, checkpoint in enumerate(checkpoint_paths):
        model = SalukiGRUV1(checkpoint).to(device).eval()
        scores = []
        with torch.no_grad():
            for start in range(0, tensor_all.shape[0], BATCH):
                scores.append(model(tensor_all[start : start + BATCH]).double().cpu().numpy())
        scores = np.concatenate(scores)
        source_scores += scores[: len(ids)]
        candidate_scores += scores[len(ids) :]
        del model
        torch.cuda.empty_cache()
        if (index + 1) % 5 == 0 or index + 1 == len(checkpoint_paths):
            print(f"[{index + 1}/{len(checkpoint_paths)}] checkpoints scored", flush=True)

    source_scores /= len(checkpoint_paths)
    candidate_scores /= len(checkpoint_paths)
    delta = candidate_scores - source_scores

    predictions = {rid: float(delta[i]) for i, rid in enumerate(ids)}

    # 1. Frozen evaluator (Task-1 caliber, K=10).
    observations = ev.load_observations([CANONICAL], validation_ids)
    metrics = ev.evaluate(observations, predictions, K)

    # 2. MPRAU variant pair-mean rho (W-ladder adjudication caliber).
    by_variant = defaultdict(list)
    for rid in ids:
        if rid.startswith("ENCSR854RUF:"):
            by_variant[rid.split(":context:")[0]].append(rid)
    variant_targets, variant_preds = [], []
    for variant, rids in sorted(by_variant.items()):
        if len(rids) >= 2:
            variant_targets.append(float(np.mean([records[r]["direction_normalized_delta"] for r in rids])))
            variant_preds.append(float(np.mean([predictions[r] for r in rids])))
    t = np.asarray(variant_targets)
    p = np.asarray(variant_preds)
    pair_mean_rho = float(spearmanr(t, p).statistic)

    # 3. Unpaired bootstrap CI for the pair-mean rho.
    rng = np.random.default_rng(BOOT_SEED)
    n = len(t)
    boots = []
    for _ in range(BOOT_ITERS):
        idx = rng.integers(0, n, n)
        r = spearmanr(t[idx], p[idx]).statistic
        if np.isfinite(r):
            boots.append(r)
    boots = np.asarray(boots)
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    report = {
        "schema_version": "route_a_v3_route2_saluki_frozen_delta_mprau.v1",
        "mode": "FROZEN_ZERO_SHOT_DELTA_WEAK_CONTROL",
        "device": torch.cuda.get_device_name(device),
        "checkpoint_count": len(checkpoint_paths),
        "record_count": len(ids),
        "metrics": metrics,
        "mprau_pair_mean": {
            "variant_count": n,
            "pair_mean_spearman": pair_mean_rho,
            "bootstrap_ci_95": ci,
            "bootstrap_iterations": int(boots.shape[0]),
            "bootstrap_seed": BOOT_SEED,
            "v5_reference_pair_mean_spearman": 0.1025449348211772,
            "note": (
                "Variant pair-mean caliber identical to W-ladder adjudication "
                "(variants = rid before ':context:', >=2 contexts, per-variant means). "
                "Weak control: 3'UTR degradation model on allelic-skew endpoint."
            ),
        },
        "note": (
            "Official Saluki fold-ensemble mean; delta = candidate - source; "
            "133bp MPRA oligo windows right-padded to 12288 (native-truncation clause). "
            "Preregistered weak control: caliber mismatch vs allelic skew."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "frozen_delta_results.json").open("w") as handle:
        json.dump(report, handle, indent=1, sort_keys=True)
    with (args.output_dir / "predictions.jsonl").open("w") as handle:
        for rid in ids:
            handle.write(json.dumps({
                "canonical_record_id": rid,
                "predicted_direction_normalized_delta": predictions[rid],
            }) + "\n")

    print(json.dumps({
        "checkpoint_count": len(checkpoint_paths),
        "record_count": len(ids),
        "task_macro_spearman": metrics.get("task_macro_spearman"),
        "top_1": metrics.get("source_macro_top_1_accuracy"),
        "ndcg_at_10": metrics.get("source_macro_ndcg_at_k"),
        "mprau_pair_mean_spearman": pair_mean_rho,
        "mprau_variant_count": n,
        "bootstrap_ci_95": ci,
        "v5_reference": 0.1025449348211772,
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
