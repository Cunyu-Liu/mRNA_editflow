#!/usr/bin/env python3
"""Explicit, post-freeze evaluator for Phase 2 final and independent roles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mrna_editflow.data.nmi_benchmark_v2 import iter_role_records, manifest_sha256
from mrna_editflow.models.paired_delta_former import PairedDeltaFormer
from mrna_editflow.train.train_paired_delta import DeltaDataset, collate, file_sha256, metrics_from_predictions, move_batch
from torch.utils.data import DataLoader


THRESHOLDS = {
    "test_v2_untouched": {"spearman": 0.35, "sign_accuracy": 0.68, "top10_enrichment": 1.75, "beneficial_precision": 0.75, "ece": 0.10},
    "independent_assay": {"spearman": 0.25, "top10_enrichment": 1.40, "beneficial_precision": 0.65},
}


def load_final_rows(root: Path, role: str) -> list[dict]:
    rows = []
    for row in iter_role_records(root / "manifests" / f"{role}.json", allow_final_labels=True):
        if row.get("delta") is None:
            continue
        if row.get("task_kind") != "local_delta" or row.get("data_layer") != "C_source_matched_intervention":
            continue
        if not bool(row.get("local_delta_eligible")):
            continue
        rows.append(row)
    return rows


@torch.no_grad()
def predict(model, rows, args, device):
    model.eval()
    ys, means, probs, variances = [], [], [], []
    loader = DataLoader(DeltaDataset(rows, args.max_len, args.max_edits), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    for batch in loader:
        batch = move_batch(batch, device)
        target = batch.pop("delta")
        out = model(**batch)
        ys.extend(target.cpu().tolist())
        means.extend(out["mean"].cpu().tolist())
        probs.extend(torch.sigmoid(out["beneficial_logit"]).cpu().tolist())
        variances.extend(out["variance"].cpu().tolist())
    import numpy as np
    return {"y": np.asarray(ys), "mean": np.asarray(means), "prob": np.asarray(probs), "variance": np.asarray(variances)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", default="data/nmi_benchmark_v2")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--role", choices=["test_id", "test_assay", "test_context", "test_family", "test_ood"], required=True)
    parser.add_argument("--alias", choices=["test_v2_untouched", "independent_assay"], required=True)
    parser.add_argument("--candidate-freeze-manifest", required=True)
    parser.add_argument("--allow-final-labels", action="store_true")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--max-edits", type=int, default=10)
    parser.add_argument("--foundation-leakage-audit", default=None)
    args = parser.parse_args()
    if not args.allow_final_labels:
        raise SystemExit("refusing final evaluation: pass --allow-final-labels after model/candidate freeze")
    freeze_path = Path(args.candidate_freeze_manifest)
    if not freeze_path.exists():
        raise SystemExit("refusing final evaluation: candidate freeze manifest does not exist")
    freeze = json.loads(freeze_path.read_text())
    if not freeze.get("candidate_selection_frozen_before_unblinding", False):
        raise SystemExit("refusing final evaluation: freeze manifest does not attest pre-unblinding candidate freeze")
    root = Path(args.benchmark_root)
    rows = load_final_rows(root, args.role)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = payload.get("config", {})
    if payload.get("backbone") != "small":
        if not payload.get("is_real_foundation", False):
            raise SystemExit("refusing scientific final evaluation: checkpoint uses a foundation adapter stub")
        expected_sha = payload.get("foundation_sha256")
        actual_sha = file_sha256(cfg.get("foundation_path"))
        if not expected_sha or actual_sha != expected_sha:
            raise SystemExit("refusing final evaluation: foundation checkpoint SHA256 provenance is missing or changed")
        audit_name = args.foundation_leakage_audit or cfg.get("foundation_leakage_audit")
        if not audit_name:
            raise SystemExit("refusing final evaluation: foundation leakage audit is missing")
        audit_path = Path(audit_name)
        if not audit_path.exists():
            raise SystemExit("refusing final evaluation: foundation leakage audit does not exist")
        audit = json.loads(audit_path.read_text())
        if audit.get("status") != "pass" or int(audit.get("exact_overlap_count", -1)) != 0:
            raise SystemExit("refusing final evaluation: foundation leakage audit is not clean")
        if audit.get("foundation_provenance", {}).get("foundation_sha256") != actual_sha:
            raise SystemExit("refusing final evaluation: foundation leakage audit SHA256 mismatch")
    model = PairedDeltaFormer(
        hidden_dim=int(cfg.get("hidden_dim", 128)), layers=int(cfg.get("layers", 2)),
        max_len=int(cfg.get("max_len", args.max_len)), backbone=str(payload.get("backbone", "small")),
        foundation_path=cfg.get("foundation_path"), allow_foundation_stub=False,
        foundation_name=str(cfg.get("foundation_name", "rna_foundation")),
        unfreeze_last_n=int(cfg.get("unfreeze_last_n", 1)),
    )
    model.load_state_dict(payload["model"], strict=False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    model.to(device)
    metrics = metrics_from_predictions(predict(model, rows, args, device)) if rows else {"n": 0}
    threshold = THRESHOLDS[args.alias]
    gate = bool(rows) and all(metrics.get(k, float("nan")) >= v if k != "ece" else metrics.get(k, float("inf")) <= v for k, v in threshold.items())
    report = {
        "schema_version": "phase2_final_evaluation_v1",
        "alias": args.alias, "role": args.role,
        "n_eligible_local_delta": len(rows), "metrics": metrics,
        "thresholds": threshold, "gate_passed": gate,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "candidate_freeze_manifest": str(freeze_path.resolve()),
        "candidate_freeze_manifest_sha256": manifest_sha256(freeze_path),
        "foundation_leakage_audit": str(Path(args.foundation_leakage_audit or cfg.get("foundation_leakage_audit")).resolve()) if payload.get("backbone") != "small" else None,
        "final_test_used": True,
        "claim_policy": "only prospective_measured or explicitly eligible measured local-delta records support biological claims",
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
