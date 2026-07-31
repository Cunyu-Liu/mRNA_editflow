#!/usr/bin/env python3
"""Create a CPU-only, no-overwrite D1-to-B0 input preflight record."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.execution.acceptance_semantics import validate_phase_acceptance


RUN_ID_RE = re.compile(r"^D1_B0_PREFLIGHT_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{7,40}$")
HISTORICAL_D1_CONTRACT = {
    "id": "utr_editflow_goal_v2",
    "sha256": "c3dc5875868d847b8519fee40b14c43b65e4c5948dc5c3b98101ca61a5671dd5",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def build_preflight(
    *,
    run_id: str,
    contract: Path,
    d1_snapshot: Path,
    d1_acceptance: Path,
    candidate_store: Path,
) -> dict[str, Any]:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id does not match the D1_B0 preflight pattern")
    snapshot = _load_json(d1_snapshot)
    acceptance = _load_json(d1_acceptance)
    semantic_errors = validate_phase_acceptance("D1", acceptance, require_pass=True)
    if semantic_errors:
        raise ValueError("D1 acceptance is not a semantic PASS: " + "; ".join(semantic_errors))
    if snapshot.get("goal_contract") != HISTORICAL_D1_CONTRACT:
        raise ValueError("D1 snapshot does not declare the exact historical source contract")
    expected_acceptance = snapshot.get("acceptance")
    observed_acceptance = _ref(d1_acceptance)
    if not isinstance(expected_acceptance, Mapping):
        raise ValueError("D1 snapshot acceptance reference is missing")
    for key in ("bytes", "sha256"):
        if expected_acceptance.get(key) != observed_acceptance[key]:
            raise ValueError(f"D1 acceptance {key} differs from frozen snapshot")
    if snapshot.get("stage_id") != "D1_B0_20260728T160012Z_8862125":
        raise ValueError("D1 snapshot stage_id is not the registered frozen source")
    contract_ref = _ref(contract)
    candidate_ref = _ref(candidate_store)
    return {
        "artifact_type": "d1_to_b0_preflight_input_manifest.v1",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PREFLIGHT_PASSED",
        "scope": "frozen_d1_canonical_edit_script_prefixes_and_declared_intermediates",
        "contract": {
            "path": contract_ref["path"],
            "sha256": contract_ref["sha256"],
            "authority": "single_active_contract",
        },
        "historical_d1_source": {
            "snapshot": _ref(d1_snapshot),
            "snapshot_contract": HISTORICAL_D1_CONTRACT,
            "acceptance": observed_acceptance,
            "semantic_acceptance_passed": True,
        },
        "label_isolation": {
            "candidate_store": candidate_ref,
            "final_label_store_opened": False,
            "final_labels_used": False,
            "allowed_next_action": "independent_B0_acceptance_only_after_authorized_label_isolated_evaluator",
        },
        "resources": {
            "gpu_requested": False,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "mode": "CPU_IO_ONLY",
        },
        "scientific_result_claimed": False,
    }


def write_exclusive(output_root: Path, payload: Mapping[str, Any]) -> None:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite preflight root: {output_root}")
    output_root.mkdir(parents=True)
    manifest = output_root / "input_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    terminal = output_root / "terminal.json"
    terminal.write_text(
        json.dumps({"status": "PREFLIGHT_PASSED", "scientific_result_claimed": False}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--d1-snapshot", type=Path, required=True)
    parser.add_argument("--d1-acceptance", type=Path, required=True)
    parser.add_argument("--candidate-store", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        payload = build_preflight(
            run_id=args.run_id,
            contract=args.contract,
            d1_snapshot=args.d1_snapshot,
            d1_acceptance=args.d1_acceptance,
            candidate_store=args.candidate_store,
        )
        write_exclusive(args.output_root, payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAILED_WITH_EVIDENCE", "error": str(exc), "scientific_result_claimed": False}, sort_keys=True))
        return 2
    print(json.dumps({"status": "PREFLIGHT_PASSED", "output_root": str(args.output_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
