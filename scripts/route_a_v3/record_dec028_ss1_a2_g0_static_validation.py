#!/usr/bin/env python3
"""Write an aggregate-only audit record for the permitted DEC028 A2 G0 check."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


class StaticAuditError(RuntimeError):
    """The non-authoritative static A2 result does not meet its record contract."""


def _load_candidate(repo_root: Path) -> Any:
    path = repo_root / "scripts/route_a_v3/a2_g0_synthetic_evaluator_candidate.py"
    spec = importlib.util.spec_from_file_location("dec028_a2_g0_candidate_for_static_record", path)
    if spec is None or spec.loader is None:
        raise StaticAuditError(f"cannot load A2 G0 candidate: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validated_result(repo_root: Path) -> dict[str, Any]:
    candidate = _load_candidate(repo_root)
    result = candidate.validate_only(
        repo_root / "configs/route_a_v3_a2_g0_evaluator_candidate_v1.json"
    )
    required = {
        "status": "DRAFT_FOR_REVIEW_NOT_ACTIVE_PROTOCOL",
        "authority_status": "NON_AUTHORITATIVE",
        "mode": "VALIDATE_ONLY_SYNTHETIC_G0",
        "synthetic_fixture_only": True,
        "component_disjoint": True,
        "outcome_values_consumed_for_split": False,
        "final_assignment_materialized": False,
        "candidate_minus_source_valid": True,
        "missing_or_nonfinite_coerced_to_zero": False,
        "real_evaluation": False,
        "a2_pass_asserted": False,
        "scientific_claim_established": False,
    }
    for key, expected in required.items():
        if result.get(key) != expected:
            raise StaticAuditError(f"A2 G0 static result differs at {key}")
    if any(value not in (0, False) for value in result["runtime_truth"].values()):
        raise StaticAuditError("A2 G0 runtime truth is nonzero")
    return result


def record(repo_root: Path, run_root: Path) -> dict[str, Any]:
    result = _validated_result(repo_root)
    run_root.mkdir(parents=True, exist_ok=False)
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "record_type": "DEC028_SS1_A2_G0_STATIC_VALIDATION",
        "generated_at": generated_at,
        "scope": "STATIC_SYNTHETIC_G0_ONLY",
        "scientific_claim_status": "NOT_ESTABLISHED",
        "result": result,
        "project_rows_accessed": False,
        "private_rows_accessed": False,
        "sealed_rows_accessed": False,
        "cuda_probed": False,
        "model_or_optimizer_constructed": False,
        "checkpoint_accessed": False,
        "parameter_updates": 0,
        "real_evaluation": False,
        "final_membership_or_split_assignment_materialized": False,
    }
    acceptance = {
        "record_type": "DEC028_SS1_A2_G0_STATIC_ACCEPTANCE",
        "status": "PASS_STATIC_SYNTHETIC_G0_ONLY",
        "scientific_claim_status": "NOT_ESTABLISHED",
        "next_phase": "INDEPENDENT_IMPLEMENTATION_REVIEW_REQUIRED",
        "a2_pass_asserted": False,
        "ss1_complete": False,
        "g1_launched": False,
    }
    manifest = {
        "record_type": "DEC028_SS1_A2_G0_STATIC_RUN_MANIFEST",
        "run_id": run_root.name,
        "scope": "STATIC_SYNTHETIC_G0_ONLY",
        "outputs": [
            "A2_G0_STATIC_VALIDATION.json",
            "RUN_MANIFEST.json",
            "ACCEPTANCE.json",
            "RUN_LOG.md",
        ],
        "project_rows_accessed": False,
        "cuda_probed": False,
        "parameter_updates": 0,
        "real_evaluation": False,
    }
    (run_root / "A2_G0_STATIC_VALIDATION.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (run_root / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (run_root / "ACCEPTANCE.json").write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n")
    (run_root / "RUN_LOG.md").write_text(
        "generated_at: " + generated_at + "\n"
        "scope: STATIC_SYNTHETIC_G0_ONLY\n"
        "next_phase: INDEPENDENT_IMPLEMENTATION_REVIEW_REQUIRED\n"
    )
    return acceptance


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(record(args.repo_root, args.run_root), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
