#!/usr/bin/env python3
"""Fail-closed surface audit for the v3.1 D1-R contract.

This intentionally audits the generated artifact namespace and a bounded sample
of every row-level JSONL file.  A PASS requires exact required artifact names
and all sampled rows to satisfy the contract-level field sets; a sampled
failure is sufficient to reject a D1 attempt, while a PASS is never claimed
without a separately recorded full validator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


HEX64 = re.compile(r"^[0-9a-f]{64}$")

CONTRACT_REQUIRED = {
    "sequence_entities.jsonl": [
        "sequence_id", "primary_asset_id", "contributing_asset_ids",
        "contributing_source_file_sha256s", "contributor_set_sha256",
        "sequence_reconstruction_rule_id", "sequence_reconstruction_rule_sha256",
        "source_record_id", "source_row_locator", "raw_sequence_sha256",
        "normalized_sequence_sha256", "normalization_steps", "alphabet_status",
        "model_sequence_eligible", "invalid_symbol_status", "region",
        "sequence_scope", "species", "strand", "original_length",
        "full_sequence_sha256",
    ],
    "functional_observation_candidates.jsonl": [
        "observation_candidate_id", "asset_ids", "contributing_source_file_sha256s",
        "contributor_set_sha256", "source_unit_ids", "sequence_id", "context_id",
        "endpoint_id", "join_method_id", "join_method_sha256",
        "observation_acceptance_status", "accepted_observation_id",
        "terminal_disposition_reason", "source_row_locators", "evidence_ids",
        "parent_candidate_id",
    ],
    "functional_observations.jsonl": [
        "observation_id", "observation_candidate_id", "canonical_status", "sequence_id",
        "primary_label_asset_id", "contributing_asset_ids", "contributing_source_file_sha256s",
        "contributor_set_sha256", "source_file_sha256", "source_record_id",
        "source_row_locator", "scientific_track", "observation_role", "context_id",
        "endpoint_id", "raw_value", "normalized_value", "label_status", "label_unit",
        "label_transform", "source_replicate_label", "sample_id",
        "biological_replicate_id", "technical_replicate_id", "barcode_id",
    ],
    "ENDPOINT_REGISTRY.jsonl": [
        "endpoint_id", "biological_quantity", "raw_field_mappings", "label_unit",
        "directionality", "label_transform", "comparability_scope",
        "aggregation_rule_id", "aggregation_rule_sha256", "delta_rule_id",
        "delta_rule_sha256", "unknown_or_ambiguous_policy", "record_sha256",
    ],
    "utr_edit_relation_candidates.jsonl": [
        "relation_candidate_id", "design_relation_group_id", "contributing_asset_ids",
        "contributing_source_file_sha256s", "contributor_set_sha256", "relation_context_key",
        "context_id", "endpoint_id", "label_unit", "label_transform", "delta_rule_id",
        "delta_rule_sha256", "scientific_track", "relation_acceptance_status",
        "relation_type", "future_use_role", "source_sequence_id", "candidate_sequence_id",
        "pair_evidence_id", "accepted_pair_id",
    ],
    "utr_edit_pairs.jsonl": [
        "pair_id", "relation_candidate_id", "design_relation_group_id",
        "contributing_asset_ids", "contributing_source_file_sha256s",
        "contributor_set_sha256", "context_id", "endpoint_id", "label_unit",
        "label_transform", "delta_rule_id", "delta_rule_sha256", "scientific_track",
        "relation_acceptance_status", "relation_type", "future_use_role",
        "source_sequence_id", "candidate_sequence_id", "same_assay_context",
        "true_length_change", "minimum_edit_distance", "path_ambiguity_count_or_bound",
        "pair_direction_verified", "pairing_method", "pair_evidence_id",
    ],
    "EXPOSURE_RECORDS.jsonl": [
        "exposure_record_id", "object_id", "object_type",
        "project_sequence_analytic_exposure", "project_sequence_analytic_use_types",
        "project_label_analytic_exposure", "project_label_analytic_use_types",
        "pipeline_sequence_materialization", "pipeline_label_materialization",
        "foundation_overlap_requirement", "foundation_audit_scope_id",
        "foundation_audit_status_at_baseline", "contributing_asset_ids",
        "contributing_file_sha256s", "rights_evidence_ids", "rights_projection_rule_id",
        "rights_projection_rule_sha256", "permitted_model_training",
        "permitted_evaluation", "permitted_derived_release", "permitted_raw_redistribution",
        "canonical_object_sha256", "record_sha256",
    ],
    "USE_ROLES.jsonl": [
        "use_role_record_id", "relation_candidate_id", "pair_id", "base_future_use_role",
        "candidate_base_payload_sha256", "pair_base_payload_sha256",
        "canonical_manifest_sha256", "record_sha256",
    ],
    "CURRENT_CANONICAL_OBJECT_PROJECTION.jsonl": [
        "projection_record_id", "run_id", "canonical_snapshot_id", "object_type",
        "chain_root_object_id", "chain_root_object_sha256", "current_leaf_object_id",
        "current_leaf_object_sha256", "generation_index", "chain_length",
        "last_supersession_edge_id", "last_supersession_edge_sha256",
        "supersession_manifest_sha256", "is_current_leaf_accepted", "projection_sha256",
    ],
    "SUPERSESSION_EDGES.jsonl": [
        "supersession_edge_id", "object_type", "old_object_id", "new_object_id",
        "old_object_sha256", "new_object_sha256", "reason", "run_id",
        "code_commit", "config_hash", "edge_sha256",
    ],
    "group_registry.jsonl": [
        "group_id", "group_type", "grouping_method", "method_version", "thresholds",
        "source_evidence", "member_count", "ambiguous_membership", "parent_group_id",
    ],
    "group_assignments.jsonl": [
        "assignment_id", "object_id", "object_type", "group_id", "grouping_atom",
        "assignment_algorithm_id",
    ],
}

REQUIRED_PATHS = [
    "sequence_entities.jsonl", "functional_observation_candidates.jsonl",
    "functional_observations.jsonl", "ENDPOINT_REGISTRY.jsonl",
    "utr_edit_relation_candidates.jsonl", "utr_edit_pairs.jsonl", "rejections.jsonl",
    "transformation_edges.jsonl", "SUPERSESSION_EDGES.jsonl",
    "CURRENT_CANONICAL_OBJECT_PROJECTION.jsonl", "EXPOSURE_RECORDS.jsonl", "USE_ROLES.jsonl",
    "group_registry.jsonl", "group_assignments.jsonl", "dataset_reconciliation.json",
    "data_units_report.json", "reporter_artifact_assessments.jsonl", "EXPOSURE_USE_MANIFEST.json",
    "EXPOSURE_USE_SHA256SUMS", "CANONICAL_MANIFEST.json", "CANONICAL_SHA256SUMS",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_rows(path: Path, limit: int = 100):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            rows.append((line_no, json.loads(line)))
            if len(rows) >= limit:
                break
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ordinary", type=Path, required=True)
    ap.add_argument("--restricted", type=Path, required=True)
    ap.add_argument("--assembly-summary", type=Path, required=True)
    ap.add_argument("--d0-decisions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--skip-ordinary-marker-scan",
        action="store_true",
        help="Skip the full ordinary JSONL sealed-lineage scan and record that limitation.",
    )
    ap.add_argument(
        "--skip-file-hashes",
        action="store_true",
        help="Skip full-file SHA-256 calculation for sampled artifacts and record that limitation.",
    )
    args = ap.parse_args()
    errors: Counter[str] = Counter()
    evidence = {"sample_limit_per_jsonl": 100, "sampled": {}, "missing_paths": [], "errors": {}}

    for root_name, root in (("ordinary", args.ordinary), ("restricted", args.restricted)):
        missing = [name for name in REQUIRED_PATHS if not (root / name).exists()]
        for name in missing:
            errors[f"missing_required_artifact:{root_name}:{name}"] += 1
        evidence["missing_paths"].extend({"root": root_name, "path": name} for name in missing)
        for name, required in CONTRACT_REQUIRED.items():
            path = root / name
            if not path.exists():
                continue
            rows = sample_rows(path)
            evidence["sampled"][f"{root_name}/{name}"] = {
                "rows": len(rows),
                "sha256": None if args.skip_file_hashes else sha256_file(path),
                "sha256_status": "SKIPPED" if args.skip_file_hashes else "FULL_FILE",
                "size_bytes": path.stat().st_size,
            }
            for line_no, row in rows:
                missing_fields = [key for key in required if key not in row]
                if missing_fields:
                    errors[f"missing_contract_fields:{root_name}:{name}"] += 1
                    evidence["errors"].setdefault(f"missing_contract_fields:{root_name}:{name}", []).append({"line": line_no, "missing": missing_fields})
                    break

    # The ordinary namespace must not contain a sealed cohort marker anywhere.
    # This full scan is intentionally optional because a partial development
    # attempt can contain multi-gigabyte JSONL files and should first receive a
    # bounded contract-surface result without pretending full acceptance.
    evidence["ordinary_sealed_marker_scan"] = "SKIPPED" if args.skip_ordinary_marker_scan else "FULL"
    evidence["file_hashes"] = "SKIPPED" if args.skip_file_hashes else "FULL"
    if not args.skip_ordinary_marker_scan:
        for path in args.ordinary.glob("*.jsonl"):
            with path.open("r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, 1):
                    if "gse246381" in line.lower():
                        errors[f"ordinary_sealed_lineage_leak:{path.name}"] += 1
                        evidence["errors"].setdefault(f"ordinary_sealed_lineage_leak:{path.name}", []).append({"line": line_no})
                        break

    assembly = json.loads(args.assembly_summary.read_text(encoding="utf-8"))
    ordinary_datasets = {key.split(":")[1] for key in assembly.get("counts", {}) if key.startswith("ordinary:") and key.endswith(":records")}
    accepted = []
    for line in args.d0_decisions.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("d0_decision") == "ACQUIRED_FOR_REBUILD":
                accepted.append(row.get("asset_group_id"))
    missing_accepted = sorted(set(accepted) - ordinary_datasets - {"GSE246381"})
    if missing_accepted:
        errors["accepted_d0_assets_without_ordinary_or_restricted_canonical"] += len(missing_accepted)
        evidence["errors"]["accepted_d0_assets_without_ordinary_or_restricted_canonical"] = missing_accepted

    # Explicitly record legacy status instead of silently counting it as E/F.
    evidence["legacy_quarantine_counts"] = {
        key: value for key, value in assembly.get("counts", {}).items() if key.startswith("legacy_quarantine:")
    }
    evidence["assembly_duplicate_record_id_count"] = assembly.get("duplicate_record_id_count")
    if assembly.get("duplicate_record_id_count"):
        errors["duplicate_assembled_record_id"] += assembly["duplicate_record_id_count"]

    total = sum(errors.values())
    report = {
        "artifact_kind": "D1_CONTRACT_SURFACE_AUDIT",
        "phase": "D1-R",
        "status": "PASS" if total == 0 else "FAIL",
        "validation_depth": "+".join(
            [
                "bounded_first_100_rows_per_artifact",
                "skipped_ordinary_sealed_marker_scan"
                if args.skip_ordinary_marker_scan
                else "full_ordinary_sealed_marker_scan",
                "skipped_file_hashes" if args.skip_file_hashes else "full_file_hashes",
            ]
        ),
        "full_acceptance_asserted": False,
        "total_errors": total,
        "errors": dict(sorted(errors.items())),
        "evidence": evidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
