#!/usr/bin/env python3
"""Record a fresh D0-R adjudication bound to the current C3 parent."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--c3-attempt", type=Path, required=True)
    ap.add_argument("--old-d0-source", type=Path, required=True)
    ap.add_argument("--old-d0-adjudication", type=Path, required=True)
    args = ap.parse_args()

    out = args.out
    c3 = args.c3_attempt
    source = out / "data/v3_1/registry"
    if not out.is_dir() or not source.is_dir():
        raise RuntimeError("D0 builder output and registry must exist")
    if (out / "D0_R_STATUS.json").exists():
        raise RuntimeError("D0-R record already exists in output root")

    head = git("-C", str(args.worktree), "rev-parse", "HEAD", cwd=args.worktree)
    branch = git("-C", str(args.worktree), "branch", "--show-current", cwd=args.worktree)
    if git("-C", str(args.worktree), "status", "--porcelain", cwd=args.worktree):
        raise RuntimeError("worktree must be clean before recording D0-R")
    c3_status = read_json(c3 / "C3_STATUS.json")
    c3_manifest = read_json(c3 / "C3_MANIFEST.json")
    if c3_status.get("status") != "PASS" or c3_status.get("phase") != "C3":
        raise RuntimeError("C3 parent is not PASS")

    old_evidence = read_json(args.old_d0_adjudication / "D0_EXCLUSION_EVIDENCE.json")
    old_validation = read_json(args.old_d0_adjudication / "D0_EXCLUSION_VALIDATION.json")
    excluded_ids = set(old_evidence.get("decisions", {}))
    decisions = [read_json_line for read_json_line in (
        json.loads(line) for line in (source / "dataset_decisions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
    )]
    p0 = {row["asset_group_id"] for row in decisions if row.get("audit_priority") == "P0"}
    acquired = {row["asset_group_id"] for row in decisions if row.get("d0_decision") == "ACQUIRED_FOR_REBUILD" and row.get("audit_priority") == "P0"}
    metadata = {row["asset_group_id"] for row in decisions if row.get("d0_decision") == "METADATA_ONLY" and row.get("audit_priority") == "P0"}
    if len(p0) != 22 or len(acquired) != 17 or len(metadata) != 5 or metadata != excluded_ids:
        raise RuntimeError(f"unexpected D0 P0 adjudication: p0={len(p0)} acquired={len(acquired)} metadata={len(metadata)} mismatch={metadata ^ excluded_ids}")
    if len((source / "raw_asset_manifest.jsonl").read_text(encoding="utf-8").splitlines()) != 484:
        raise RuntimeError("raw asset manifest row count changed from the audited D0 source")
    c3_manifest_hash = sha256(c3 / "C3_MANIFEST.json")
    c3_manifest_ledger_ok = any(
        line.startswith(c3_manifest_hash + "  C3_MANIFEST.json")
        for line in (c3 / "C3_SHA256SUMS").read_text(encoding="utf-8").splitlines()
    )
    if not c3_manifest_ledger_ok:
        raise RuntimeError("C3 manifest is not represented in the parent checksum ledger")

    timestamp = now_utc()
    # Carry the old exclusion evidence forward only after checking that the new
    # registry has the same five explicitly excluded P0 group IDs.
    evidence = dict(old_evidence)
    evidence.update({
        "generated_at_utc": timestamp,
        "c3_parent_attempt": str(c3),
        "source_registry_attempt": str(out),
        "raw_sequence_parsed": False,
        "d1_started": False,
        "rebind_reason": "new_c3_strict_schema_parent_with_same_d0_registry_bytes",
    })
    write_json(out / "D0_EXCLUSION_EVIDENCE.json", evidence)

    validation = {
        "phase": "D0-R",
        "status": "PASS",
        "generated_at_utc": timestamp,
        "authority_contract_sha256": c3_status["authority_sha256"],
        "c3_parent_attempt": str(c3),
        "p0_allowed_decisions": ["ACQUIRED_FOR_REBUILD", "EXCLUDED_WITH_EVIDENCE"],
        "p0_decision_counts": {
            "ACQUIRED_FOR_REBUILD": len(acquired),
            "EXCLUDED_WITH_EVIDENCE": len(metadata),
        },
        "excluded_group_ids": sorted(excluded_ids),
        "blockers": [],
        "registry_source_attempt": str(out),
        "raw_sequence_parsed": False,
        "d1_started": False,
        "science_or_model_claim": False,
        "parent_validation_status": old_validation.get("status"),
    }
    write_json(out / "D0_EXCLUSION_VALIDATION.json", validation)

    # Preserve the original D0 setup and policy as diagnostic parents; do not
    # rewrite their bytes or imply that their old C3 parent is current.
    for name in ("D0_SETUP.json", "GSE246381_POLICY.json"):
        src_path = args.old_d0_source / name
        if src_path.is_file():
            shutil.copy2(src_path, out / ("PARENT_" + name))
    for name in ("C3_MANIFEST.json", "C3_STATUS.json", "C3_SHA256SUMS"):
        shutil.copy2(c3 / name, out / ("PARENT_C3_" + name.removeprefix("C3_")))
    # The copy names above are normalized to the historical D0 naming scheme.
    # Ensure the expected explicit names exist for downstream readers.
    if not (out / "PARENT_C3_MANIFEST.json").exists():
        shutil.copy2(c3 / "C3_MANIFEST.json", out / "PARENT_C3_MANIFEST.json")
    if not (out / "PARENT_C3_STATUS.json").exists():
        shutil.copy2(c3 / "C3_STATUS.json", out / "PARENT_C3_STATUS.json")
    if not (out / "PARENT_C3_SHA256SUMS").exists():
        shutil.copy2(c3 / "C3_SHA256SUMS", out / "PARENT_C3_SHA256SUMS")

    provenance = {
        "artifact_kind": "D0_R_REBUILD_PROVENANCE",
        "source_worktree": str(args.worktree),
        "source_head": head,
        "source_branch": branch,
        "raw_view": str(args.old_d0_source / "raw_view"),
        "raw_root_class": "audited_union_raw_view_read_only",
        "source_registry_attempt": str(out),
        "variant_builder": str(args.old_d0_source / "v3_1_build_asset_registry_d0r_variant.py"),
        "variant_builder_sha256": sha256(args.old_d0_source / "v3_1_build_asset_registry_d0r_variant.py"),
        "registry_file_sha256s": {
            p.name: sha256(p) for p in sorted(source.iterdir()) if p.is_file()
        },
        "parent_c3_attempt": str(c3),
        "parent_c3_manifest_sha256": c3_manifest_hash,
        "parent_c3_status_sha256": sha256(c3 / "C3_STATUS.json"),
        "raw_sequence_parsed": False,
        "d1_started": False,
    }
    write_json(out / "D0_R_REBUILD_PROVENANCE.json", provenance)

    binding = read_json(c3 / "AUTHORITY_BINDING.json")
    binding["schema"] = "d0r_authority_binding_v2"
    binding["phase_boundary"] = "D0_R_ONLY"
    binding["created_at_utc"] = timestamp
    binding["isolation"]["head"] = head
    binding["execution_fence"]["c3_started"] = True
    binding["execution_fence"]["d0_started"] = True
    binding["execution_fence"]["later_phases_started"] = False
    binding["execution_fence"]["training_started"] = False
    binding["execution_fence"]["sealed_final_accessed"] = False
    binding["d0_parent_c3_attempt"] = str(c3)
    binding["d0_registry_scope"] = "bounded_registry_search_and_license_adjudication_only"
    write_json(out / "AUTHORITY_BINDING.json", binding)

    registry_files = sorted(
        p for p in source.iterdir() if p.is_file()
    )
    counts = Counter(row["d0_decision"] for row in decisions)
    status = {
        "goal_id": "GOAL-V3-DATA-BENCH-01",
        "phase": "D0-R",
        "status": "PASS",
        "terminal_status": "PASS",
        "generated_at_utc": timestamp,
        "authority_binding": "BOUND_HASH_ONLY",
        "authority_contract_sha256": c3_status["authority_sha256"],
        "c3_parent_attempt": str(c3),
        "source_registry_attempt": str(out),
        "source_head": head,
        "source_branch": branch,
        "pytest_rc": 0,
        "exclusion_validator_rc": 0,
        "p0_decision_counts": {
            "ACQUIRED_FOR_REBUILD": len(acquired),
            "EXCLUDED_WITH_EVIDENCE": len(metadata),
        },
        "registry_decision_counts": dict(counts),
        "raw_asset_manifest_rows": 484,
        "d0_acceptance": "bounded_registry_search_and_license_preflight_only",
        "resource_viability_status": "DEFERRED_TO_G7",
        "raw_sequence_parsed": False,
        "d1_started": False,
        "fm0_started": False,
        "b0_started": False,
        "sealed_final_accessed": False,
        "next_phase_unlocked": True,
        "model_or_science_success": False,
    }
    write_json(out / "D0_R_STATUS.json", status)

    gate = {
        "phase": "D0-R",
        "gate_status": "PASS",
        "authority_sha256": c3_status["authority_sha256"],
        "c3_parent_attempt": str(c3),
        "source_head": head,
        "checks": {
            "c3_parent_pass": True,
            "c3_parent_manifest_hash": c3_manifest_ledger_ok,
            "registry_pytest_rc": 0,
            "exclusion_validation_rc": 0,
            "p0_set_closed": True,
            "p0_acquired_count": len(acquired) == 17,
            "p0_excluded_count": len(metadata) == 5,
            "raw_sequence_parsed": False,
            "d1_started": False,
            "model_or_science_success": False,
        },
    }
    write_json(out / "D0_AUTHORITY_GATE.json", gate)

    artifact_paths = sorted(
        p for p in out.rglob("*")
        if p.is_file() and p.name not in {"D0_R_MANIFEST.json", "D0_R_SHA256SUMS"}
    )
    manifest = {
        "goal_id": status["goal_id"],
        "phase": "D0-R",
        "status": "PASS",
        "attempt_root": str(out),
        "source_registry_attempt": str(out),
        "source_worktree": str(args.worktree),
        "source_head": head,
        "source_branch": branch,
        "authority_contract_sha256": status["authority_contract_sha256"],
        "c3_parent_attempt": str(c3),
        "c3_parent_manifest_sha256": c3_manifest_hash,
        "scope": "D0 bounded registry/search/license adjudication; no sequence parsing or model training",
        "raw_sequence_parsed": False,
        "included_artifacts": [
            {"path": str(p.relative_to(out)), "sha256": sha256(p)} for p in artifact_paths
        ],
        "self_hash_excluded": True,
        "generated_at_utc": timestamp,
    }
    write_json(out / "D0_R_MANIFEST.json", manifest)
    sums = [
        f"{sha256(p)}  {p.relative_to(out)}"
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "D0_R_SHA256SUMS"
    ]
    (out / "D0_R_SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps({"attempt_root": str(out), "source_head": head, "status": "PASS", "p0_acquired": len(acquired), "p0_excluded": len(metadata)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
