#!/usr/bin/env python3
"""Write an aggregate-only audit record for the permitted DEC028 A6 G0 check."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


class StaticAuditError(RuntimeError):
    """The non-authoritative static A6 result does not meet its record contract."""


def _load_candidate(repo_root: Path) -> Any:
    path = repo_root / "scripts/route_a_v3/a6_learned_base_value_g0_candidate.py"
    spec = importlib.util.spec_from_file_location("dec028_a6_g0_candidate_for_static_record", path)
    if spec is None or spec.loader is None:
        raise StaticAuditError(f"cannot load A6 G0 candidate: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validated_result(repo_root: Path) -> dict[str, Any]:
    candidate = _load_candidate(repo_root)
    result = candidate.validate_only(
        protocol_path=repo_root / "configs/route_a_v3_a6_learned_base_value_gpu_protocol_draft_v1.json",
        candidate_path=repo_root / "configs/route_a_v3_a6_learned_base_value_g0_implementation_candidate_v1.json",
    )
    required = {
        "status": "DRAFT_FOR_REVIEW_NOT_ACTIVE_PROTOCOL",
        "authority_status": "NON_AUTHORITATIVE",
        "mode": "VALIDATE_ONLY_ZERO_UPDATE_DRY_RUN",
        "parent_protocol_valid": True,
        "candidate_contract_valid": True,
        "kernel_loaded": False,
        "torch_imported": False,
        "cuda_probe_invoked": False,
        "training_allowed": False,
        "a6_pass_asserted": False,
        "l3_claim_established": False,
        "a7_unlocked": False,
        "qualification_or_credit_delta": 0,
        "canonical_delta": 0,
    }
    for key, expected in required.items():
        if result.get(key) != expected:
            raise StaticAuditError(f"A6 G0 static result differs at {key}")
    if any(value != 0 for value in result["audit"].values()):
        raise StaticAuditError("A6 G0 audit is nonzero")
    if result["architecture_plan"]["parameter_tensors_constructed"] != 0:
        raise StaticAuditError("A6 G0 constructed a parameter tensor")
    if result["future_manifest_plan"]["files_created"] != 0:
        raise StaticAuditError("A6 G0 created a future runtime manifest")
    return result


def record(repo_root: Path, run_root: Path) -> dict[str, Any]:
    result = _validated_result(repo_root)
    run_root.mkdir(parents=True, exist_ok=False)
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "record_type": "DEC028_SS1_A6_G0_STATIC_VALIDATION",
        "generated_at": generated_at,
        "scope": "STATIC_SYNTHETIC_ZERO_UPDATE_G0_ONLY",
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
    }
    acceptance = {
        "record_type": "DEC028_SS1_A6_G0_STATIC_ACCEPTANCE",
        "status": "PASS_STATIC_ZERO_UPDATE_G0_ONLY",
        "scientific_claim_status": "NOT_ESTABLISHED",
        "next_phase": "INDEPENDENT_IMPLEMENTATION_REVIEW_REQUIRED",
        "a6_pass_asserted": False,
        "ss1_complete": False,
        "g1_launched": False,
    }
    manifest = {
        "record_type": "DEC028_SS1_A6_G0_STATIC_RUN_MANIFEST",
        "run_id": run_root.name,
        "scope": "STATIC_SYNTHETIC_ZERO_UPDATE_G0_ONLY",
        "outputs": [
            "A6_G0_STATIC_VALIDATION.json",
            "RUN_MANIFEST.json",
            "ACCEPTANCE.json",
            "RUN_LOG.md",
        ],
        "project_rows_accessed": False,
        "cuda_probed": False,
        "parameter_updates": 0,
        "real_evaluation": False,
    }
    (run_root / "A6_G0_STATIC_VALIDATION.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (run_root / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (run_root / "ACCEPTANCE.json").write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n")
    (run_root / "RUN_LOG.md").write_text(
        "generated_at: " + generated_at + "\n"
        "scope: STATIC_SYNTHETIC_ZERO_UPDATE_G0_ONLY\n"
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
