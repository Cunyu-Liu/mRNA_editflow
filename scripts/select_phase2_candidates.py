#!/usr/bin/env python3
"""Score registered final-role inputs without reading final labels.

The resulting artifact is the only valid input to
``freeze_phase2_candidate_manifest.py``.  All eligible registered candidates
are retained so the acceptance metrics are not changed by a post-hoc score
cutoff; model scores are recorded for prospective screening only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mrna_editflow.models.paired_delta_former import PairedDeltaFormer
from mrna_editflow.train.train_paired_delta import file_sha256
from scripts.evaluate_phase2_oracle import (
    EXPECTED_ROLES,
    INDEPENDENT_ASSAY_NAME,
    candidate_digest,
    load_selection_rows,
    predict,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", default="data/nmi_benchmark_v2")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--role", choices=["test_id", "test_ood"], required=True)
    parser.add_argument("--alias", choices=["test_v2_untouched", "independent_assay"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--max-edits", type=int, default=10)
    args = parser.parse_args()
    if EXPECTED_ROLES[args.alias] != args.role:
        raise SystemExit("role/alias combination is not registered for Phase 2")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"checkpoint does not exist: {checkpoint}")
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite selection artifact: {out}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    root = Path(args.benchmark_root)
    rows = load_selection_rows(root, args.role, args.alias)
    if not rows:
        raise SystemExit("no eligible pre-unblinding inputs found")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = payload.get("config", {})
    model = PairedDeltaFormer(
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        layers=int(cfg.get("layers", 2)),
        max_len=int(cfg.get("max_len", args.max_len)),
        backbone=str(payload.get("backbone", "small")),
        foundation_path=cfg.get("foundation_path"),
        allow_foundation_stub=False,
        foundation_name=str(cfg.get("foundation_name", "rna_foundation")),
        unfreeze_last_n=int(cfg.get("unfreeze_last_n", 1)),
    ).to(device)
    model.load_state_dict(payload["model"], strict=False)

    # DeltaDataset requires a target-shaped field, but this is a synthetic zero
    # used only to satisfy collation; no final label is read or used.
    model_rows = []
    for row in rows:
        safe = dict(row)
        safe["delta"] = 0.0
        model_rows.append(safe)
    prediction_args = argparse.Namespace(
        max_len=int(args.max_len), max_edits=int(args.max_edits), batch_size=int(args.batch_size),
    )
    predictions = predict(model, model_rows, prediction_args, device)
    by_id = {}
    for row, mean, prob, variance in zip(
        rows, predictions["mean"], predictions["prob"], predictions["variance"]
    ):
        by_id[str(row.get("record_id"))] = {
            "record_id": str(row.get("record_id")),
            "predicted_delta": float(mean),
            "beneficial_probability": float(prob),
            "predicted_variance": float(variance),
        }
    report = {
        "schema_version": "phase2_candidate_selection_v1",
        "role": args.role,
        "alias": args.alias,
        "selection_filter": INDEPENDENT_ASSAY_NAME if args.alias == "independent_assay" else None,
        "selection_rule": "retain_all_registered_eligible_candidates; record_model_scores_for_screening",
        "labels_accessed": False,
        "candidate_digest": candidate_digest(rows),
        "candidate_count": len(rows),
        "candidate_ids": sorted(by_id),
        "scores": [by_id[key] for key in sorted(by_id)],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(str(checkpoint)),
        "checkpoint_final_test_used": bool(payload.get("final_test_used", False)),
        "claim_policy": "pre-unblinding selection evidence only; no final metric or biological claim",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
