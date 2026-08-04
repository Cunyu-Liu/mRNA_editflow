#!/usr/bin/env python3
"""Write an explicit fail-closed D1 status from recorded evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--assembly-summary", type=Path, required=True)
    ap.add_argument("--baseline-log", type=Path, required=True)
    ap.add_argument("--authority-contract-sha256", required=True)
    ap.add_argument("--source-head", required=True)
    ap.add_argument("--baseline-root", required=True)
    ap.add_argument("--old-validator-log", type=Path)
    ap.add_argument("--old-validator-status", type=Path)
    args = ap.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    assembly = json.loads(args.assembly_summary.read_text(encoding="utf-8"))
    counters = {}
    for key, value in assembly.get("counts", {}).items():
        counters[key] = value
    baseline_log_sha = sha256(args.baseline_log)
    old_validator = None
    if args.old_validator_log and args.old_validator_log.exists():
        old_validator = {
            "path": str(args.old_validator_log),
            "sha256": sha256(args.old_validator_log),
            "size_bytes": args.old_validator_log.stat().st_size,
            "status": "RECORDED",
        }
        if args.old_validator_status and args.old_validator_status.exists():
            old_validator["status_marker_path"] = str(args.old_validator_status)
            old_validator["status_marker_sha256"] = sha256(args.old_validator_status)
            old_validator["status_marker"] = args.old_validator_status.read_text(encoding="utf-8").strip()
            old_validator["status"] = "NOT_COMPLETED_AFTER_TIMEOUT"
    report = {
        "artifact_kind": "D1_R_STATUS",
        "phase": "D1-R",
        "attempt_id": Path(args.baseline_root).name,
        "status": "FAIL",
        "d1_acceptance_asserted": False,
        "next_phase_unlocked": False,
        "model_access": False,
        "training": False,
        "final_evaluator_access": False,
        "source_head": args.source_head,
        "authority_contract_sha256": args.authority_contract_sha256,
        "baseline_root": args.baseline_root,
        "baseline_kind": "DEVELOPMENT_ONLY_TECHNICAL_BASELINE",
        "raw_assembly": {
            "summary_path": str(args.assembly_summary),
            "summary_sha256": sha256(args.assembly_summary),
            "duplicate_record_id_count": assembly.get("duplicate_record_id_count"),
            "counts": counters,
        },
        "strict_contract_surface_audit": {
            "path": str(args.audit),
            "sha256": sha256(args.audit),
            "status": audit.get("status"),
            "total_errors": audit.get("total_errors"),
            "full_acceptance_asserted": audit.get("full_acceptance_asserted"),
            "validation_depth": audit.get("validation_depth"),
        },
        "baseline_log": {
            "path": str(args.baseline_log),
            "sha256": baseline_log_sha,
            "size_bytes": args.baseline_log.stat().st_size,
        },
        "old_validator": old_validator,
        "blockers": [
            "strict_contract_surface_audit_failed",
            "required_d1_artifacts_missing_in_ordinary_and_restricted_namespaces",
            "sampled_contract_rows_missing_lineage_candidate_context_endpoint_and_delta_fields",
            "eight_acquired_d0_assets_have_no_ordinary_or_restricted_canonical_output",
            "full_acceptance_and_row_level_negative_lineage_validation_not_asserted",
        ],
        "phase_boundary": "D1-R failed; FM0-A, B0-R, G7, training, and final evaluator remain locked.",
        "quota_note": "The earlier /home technical build stopped at user quota; the complete development-only baseline was rerun under /mnt without deleting prior artifacts.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
