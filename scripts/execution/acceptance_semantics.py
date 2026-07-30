#!/usr/bin/env python3
"""Fail-closed document semantics for D1 and B0 acceptance artifacts.

These checks deliberately validate the full acceptance *document shape* and
the gate-bearing nested summaries.  They are defense in depth for the release
validators; the phase-specific production validators remain the source that
must recompute these documents from frozen inputs.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any


D1_SCOPE_DATASETS = frozenset(
    {
        "ENCSR854RUF_raw62",
        "GSE114002",
        "GSE145046",
        "GSE149487",
        "GSE173083",
        "GSE200304",
        "GSE207584",
        "GSE217518",
        "GSE246381",
        "GSE291719",
        "GSE330741",
        "MPRAu_processed_ENCSR854RUF",
    }
)
D1_REQUIRED_SUPPORTED_DATASETS = frozenset(
    {"GSE114002", "GSE200304", "GSE217518", "GSE246381"}
)
D1_ACCEPTED_DATASETS = D1_REQUIRED_SUPPORTED_DATASETS
D1_BLOCKED_DATASETS = D1_SCOPE_DATASETS - D1_ACCEPTED_DATASETS
D1_PROVENANCE_RAW_FILE_KEYS = frozenset(
    {
        "path",
        "bytes",
        "sha256",
        "role",
        "format",
        "delimiter",
        "sheet_name",
        "defaults",
    }
)
D1_REQUIRED_ARTIFACT_PATHS = frozenset(
    {
        "data/data_exposure_ledger.jsonl",
        "data/library_ascertainment_report.json",
        "data/edit_script_ambiguity_report.json",
        "data/measured_action_coverage_report.json",
        "reports/data_reproduction/summary.csv",
    }
)

D1_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "generated_at_utc",
        "stage_d1_root",
        "fixture_mode",
        "evidence_level",
        "dataset_results",
        "required_supported_datasets",
        "missing_required_datasets",
        "expected_d1_scope_datasets",
        "missing_d1_scope_datasets",
        "structural_validation_passed",
        "required_artifact_validation",
        "global_store_validation",
        "config_binding_validation",
        "dataset_manifest_binding_validation",
        "builder_audit_validation",
        "phase_gate_passed",
        "scientific_result_claimed",
        "note",
    }
)

B0_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "b0_gate_passed",
        "failed_gates",
        "observed",
        "allowed_claim",
        "requires_fm0_reaudit",
        "re_audit_required_before_foundation_use",
        "claim_boundary",
        "exposure_ledger",
        "track_role_audit",
        "track_a_label_seal_audit",
        "required_artifact_audit",
        "d1_exposure_ledger_binding",
        "supplied_leakage_reports_match_recomputation",
        "recomputed_leakage_reports",
        "supplied_leakage_report_files",
    }
)

B0_BOUND_ARTIFACTS = frozenset(
    {
        "exposure_ledger",
        "track_role_matrix",
        "data_card",
        "claims",
    }
)
B0_SPLIT_IDENTITIES = frozenset(
    {
        ("source_disjoint", "five_utr", None),
        ("study_disjoint", "five_utr", None),
        ("source_disjoint", "three_utr", None),
        ("study_disjoint", "three_utr", None),
        ("cross_region_transfer", "five_utr", "three_utr"),
    }
)
B0_OBSERVED_KEYS = frozenset(
    {
        "leakage_report_count",
        "exposure_ledger_coverage",
        "exposure_identity_level",
        "track_role_ambiguity_count",
        "track_identity_universe_complete",
        "track_eligible_identity_universe_complete",
        "split_identities",
        "split_universe",
        "eligible_track_role_universe",
        "eligible_track_role_universe_failures",
        "foundation_states",
    }
)
B0_CLAIM_BOUNDARY = "B0 structural split acceptance is not an efficacy or SOTA result"
B0_SPLIT_UNIVERSE_KEYS = frozenset(
    {
        "canonical_records_sha256",
        "structural_records_sha256",
        "structural_records_bytes",
        "canonical_record_ids_sha256",
        "canonical_record_count",
        "structural_record_ids_sha256",
        "structural_record_count",
        "structural_content_sha256",
    }
)
B0_ELIGIBLE_UNIVERSE_KEYS = frozenset(
    {
        "record_ids_sha256",
        "record_count",
        "excluded_record_ids_sha256",
        "excluded_record_count",
    }
)
B0_D1_EXPOSURE_BINDING_KEYS = frozenset(
    {
        "schema_version",
        "gate_passed",
        "failures",
        "d1_acceptance_path",
        "d1_acceptance_sha256",
        "d1_build_manifest_path",
        "d1_build_manifest_sha256",
        "exposure_ledger_path",
        "exposure_ledger_bytes",
        "exposure_ledger_sha256",
        "ledger_semantics_valid",
    }
)
B0_FOUNDATION_STATE = {
    "status": "UNKNOWN_PENDING_FM0",
    "foundation_selected": False,
    "checkpoint_sha256": None,
    "corpus_manifest_sha256": None,
    "audit_report_sha256": None,
    "clearance_evidence_complete": False,
    "allowed_claim": "NONE",
    "re_audit_required": True,
    "gate_applicable": False,
    "gate_passed": True,
}
B0_LEAKAGE_ZERO_COUNTS = {
    "unexplained_overlap_zero": "unexplained_overlap_count",
    "exact_source_overlap_zero": "exact_source_leakage_count",
    "exact_candidate_overlap_zero": "exact_candidate_leakage_count",
    "reverse_edge_leakage_zero": "reverse_edge_leakage_count",
    "path_leakage_zero": "path_leakage_count",
    "near_neighbor_leakage_zero": "near_neighbor_leakage_count",
    "final_endpoint_as_train_intermediate_zero": (
        "final_endpoint_as_train_intermediate_count"
    ),
    "required_axis_overlap_zero": "required_axis_overlap_count",
}
B0_LEAKAGE_GATE_KEYS = frozenset({*B0_LEAKAGE_ZERO_COUNTS, "foundation_overlap_gate"})
B0_LEAKAGE_COUNT_KEYS = frozenset(
    {
        "exact_source_leakage_count",
        "exact_candidate_leakage_count",
        "reverse_edge_leakage_count",
        "path_leakage_count",
        "near_neighbor_leakage_count",
        "final_endpoint_as_train_intermediate_count",
        "metadata_overlap_count",
        "explained_metadata_overlap_count",
        "unexplained_metadata_overlap_count",
        "record_role_overlap_count",
        "component_role_overlap_count",
        "frozen_universe_issue_count",
        "required_axis_overlap_count",
        "unexplained_overlap_count",
    }
)
B0_LEAKAGE_REQUIRED_ZERO_COUNTS = frozenset(
    {
        *B0_LEAKAGE_ZERO_COUNTS.values(),
        "unexplained_metadata_overlap_count",
        "record_role_overlap_count",
        "component_role_overlap_count",
        "frozen_universe_issue_count",
    }
)
B0_REQUIRED_PARTITION_IDS = {
    ("source_disjoint", "five_utr", None): frozenset(
        {
            "source_disjoint:five_utr",
            "sequence_cluster_disjoint:five_utr",
            "scaffold_disjoint:five_utr",
            "gene_disjoint:five_utr",
            "context_disjoint:five_utr",
            "barcode_batch_disjoint:five_utr",
            "library_batch_disjoint:five_utr",
        }
    ),
    ("study_disjoint", "five_utr", None): frozenset(
        {"loso:GSE114002", "loso:GSE217518"}
    ),
    ("source_disjoint", "three_utr", None): frozenset(
        {
            "source_disjoint:three_utr",
            "sequence_cluster_disjoint:three_utr",
            "scaffold_disjoint:three_utr",
            "gene_disjoint:three_utr",
            "context_disjoint:three_utr",
            "barcode_batch_disjoint:three_utr",
            "library_batch_disjoint:three_utr",
        }
    ),
    ("study_disjoint", "three_utr", None): frozenset(
        {"loso:GSE217518", "loso:GSE200304"}
    ),
    ("cross_region_transfer", "five_utr", "three_utr"): frozenset(
        {
            "within_study:GSE217518:five_to_three",
            "within_study:GSE217518:three_to_five",
            "cross_study:GSE114002_five_to_GSE200304_three",
        }
    ),
}
B0_RECOMPUTED_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "split_kind",
        "region",
        "source_region",
        "target_region",
        "required_partition_ids",
        "partition_count",
        "partitions",
        "counts",
        "acceptance_gates",
        "gate_passed",
        "common_universe_binding",
        "foundation_pretraining_overlap",
        "structural_issues",
        "structural_records_path",
        "structural_records_sha256",
        "structural_records_bytes",
        "canonical_records_path",
        "canonical_records_sha256",
        "canonical_record_count",
        "canonical_record_ids_sha256",
        "structural_record_count",
        "structural_record_ids_sha256",
        "structural_content_sha256",
        "split_manifest_path",
        "split_manifest_sha256",
        "split_manifest_bytes",
        "foundation_exposure_path",
        "foundation_exposure_sha256",
        "recomputed_from_bound_structural_records",
        "canonical_manifest_exact_recomputation",
        "canonical_manifest_core_sha256",
        "auditor_binding",
    }
)
B0_PARTITION_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "partition_id",
        "split_partition_sha256",
        "split_kind",
        "region",
        "source_region",
        "target_region",
        "heldout_study",
        "counts",
        "acceptance_gates",
        "gate_passed",
        "foundation_pretraining_overlap",
        "metadata_axis_status",
        "required_axis_status",
        "examples",
    }
)
B0_AUDITOR_BINDING_KEYS = frozenset(
    {
        "schema_version",
        "entrypoint_path",
        "entrypoint_sha256",
        "canonical_auditor_path",
        "canonical_auditor_sha256",
    }
)

_TRUE_GATE_KEYS = frozenset(
    {
        "passed",
        "gate_passed",
        "schema_valid",
        "exists",
        "identity_universe_complete",
        "eligible_identity_binding_complete",
    }
)
_GATE_PREDICATE_CONTAINER_KEYS = frozenset(
    {"binding_checks", "checks", "content_checks", "semantic_checks"}
)
_D1_PROVENANCE_CHECK_NAME = "production_input_provenance_complete"
_D1_PROVENANCE_MUTUALLY_EXCLUSIVE_KEYS = frozenset(
    {"accepted_provenance_passed", "metadata_only_provenance_passed"}
)


def _valid_time(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _require_mapping(
    value: Any, label: str, errors: list[str]
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object")
        return None
    return value


def _has_fields(value: Mapping[str, Any], required: set[str]) -> bool:
    return required <= set(value)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_exact_int(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


def _is_exact_float(value: Any, expected: float) -> bool:
    return type(value) is float and value == expected


def _require_keys(
    payload: Mapping[str, Any],
    required: frozenset[str],
    label: str,
    errors: list[str],
) -> None:
    missing = required - set(payload)
    if missing:
        errors.append(f"{label} missing required keys: {sorted(missing)}")
    extra = set(payload) - required
    if extra:
        errors.append(f"{label} has unexpected keys: {sorted(extra)}")


def _nested_false_gate_paths(
    value: Any,
    prefix: str = "",
    *,
    ignored_false_paths: frozenset[str] = frozenset(),
    predicate_values_are_gates: bool = False,
) -> list[str]:
    failures: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            key_name = str(key)
            is_gate_key = (
                predicate_values_are_gates
                or key_name in _TRUE_GATE_KEYS
                or key_name.endswith("_passed")
                or key_name.endswith("_schema_valid")
                or key_name.endswith("_binding_complete")
            )
            if (
                is_gate_key
                and isinstance(item, bool)
                and item is not True
                and path not in ignored_false_paths
            ):
                failures.append(path)
            failures.extend(
                _nested_false_gate_paths(
                    item,
                    path,
                    ignored_false_paths=ignored_false_paths,
                    predicate_values_are_gates=(
                        key_name in _GATE_PREDICATE_CONTAINER_KEYS
                    ),
                )
            )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            failures.extend(
                _nested_false_gate_paths(
                    item,
                    f"{prefix}[{index}]",
                    ignored_false_paths=ignored_false_paths,
                    predicate_values_are_gates=predicate_values_are_gates,
                )
            )
    return failures


def _validate_d1_provenance_check(
    *,
    dataset_id: str,
    status: Any,
    check: Mapping[str, Any],
    check_path: str,
    errors: list[str],
) -> frozenset[str]:
    """Validate the accepted/blocked provenance branches before exclusions."""
    label = f"D1 acceptance dataset {dataset_id} production provenance"
    detail = _require_mapping(check.get("detail"), f"{label} detail", errors)
    if detail is None:
        return frozenset()

    ignored_false_paths = frozenset(
        f"{check_path}.detail.{key}" for key in _D1_PROVENANCE_MUTUALLY_EXCLUSIVE_KEYS
    )
    if status not in {"accepted", "blocked"}:
        errors.append(f"{label} cannot be validated for status {status!r}")
        return ignored_false_paths

    is_accepted = status == "accepted"
    expected_detail = {
        "fixture_exemption": False,
        "blocked_or_excluded": not is_accepted,
        "accepted_provenance_passed": is_accepted,
        "metadata_only_provenance_passed": not is_accepted,
    }
    for key, expected in expected_detail.items():
        if detail.get(key) is not expected:
            errors.append(f"{label} {key} is inconsistent with {status} status")

    integrity_failures = detail.get("integrity_failures")
    if integrity_failures != []:
        errors.append(f"{label} integrity_failures must be empty")

    audit = _require_mapping(detail.get("audit"), f"{label} audit", errors)
    if audit is None:
        return ignored_false_paths
    expected_audit = {
        "complete": is_accepted,
        "raw_files_complete": is_accepted,
        "download_manifest_complete": True,
        "license_complete": True,
    }
    for key, expected in expected_audit.items():
        if audit.get(key) is not expected:
            errors.append(f"{label} audit.{key} is inconsistent with {status} status")

    raw_files = audit.get("raw_files")
    if not isinstance(raw_files, list):
        errors.append(f"{label} audit.raw_files must be a list")
    elif is_accepted and not raw_files:
        errors.append(f"{label} accepted status requires bound raw files")
    elif not is_accepted and raw_files != []:
        errors.append(f"{label} blocked status may not bind opened raw files")
    elif is_accepted:
        for index, reference in enumerate(raw_files):
            reference_label = f"{label} audit.raw_files[{index}]"
            if not isinstance(reference, Mapping):
                errors.append(f"{reference_label} must be an object")
                continue
            if set(reference) != D1_PROVENANCE_RAW_FILE_KEYS:
                errors.append(f"{reference_label} field inventory is not exact")
            path = reference.get("path")
            if not isinstance(path, str) or not Path(path).is_absolute():
                errors.append(f"{reference_label} path must be absolute")
            bytes_value = reference.get("bytes")
            if (
                not isinstance(bytes_value, int)
                or isinstance(bytes_value, bool)
                or bytes_value <= 0
            ):
                errors.append(f"{reference_label} bytes must be a positive integer")
            if not _is_sha256(reference.get("sha256")):
                errors.append(f"{reference_label} sha256 is invalid")
            for field in ("role", "format"):
                value = reference.get(field)
                if not isinstance(value, str) or not value:
                    errors.append(f"{reference_label} {field} must be non-empty")
            for field in ("delimiter", "sheet_name"):
                value = reference.get(field)
                if value is not None and (not isinstance(value, str) or not value):
                    errors.append(
                        f"{reference_label} {field} must be null or non-empty"
                    )
            if not isinstance(reference.get("defaults"), Mapping):
                errors.append(f"{reference_label} defaults must be an object")

    derived_complete = (
        audit.get("raw_files_complete") is True
        and audit.get("download_manifest_complete") is True
        and audit.get("license_complete") is True
    )
    if audit.get("complete") is not derived_complete:
        errors.append(f"{label} audit.complete is not internally consistent")
    return ignored_false_paths


def _nested_false_paths(value: Any, prefix: str = "") -> list[str]:
    """Return every false boolean below a structure known to contain gates."""
    failures: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if item is False:
                failures.append(path)
            failures.extend(_nested_false_paths(item, path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            failures.extend(_nested_false_paths(item, f"{prefix}[{index}]"))
    return failures


def _validate_d1(payload: Mapping[str, Any], require_pass: bool) -> list[str]:
    errors: list[str] = []
    _require_keys(payload, D1_REQUIRED_KEYS, "D1 acceptance", errors)
    if payload.get("schema_version") != "d1_acceptance_v2":
        errors.append("D1 acceptance schema_version is invalid")
    if not _valid_time(payload.get("generated_at_utc")):
        errors.append("D1 acceptance generated_at_utc must be timezone-aware")
    stage_root = payload.get("stage_d1_root")
    if not isinstance(stage_root, str) or not Path(stage_root).is_absolute():
        errors.append("D1 acceptance stage_d1_root must be an absolute path")
    if payload.get("fixture_mode") is not False:
        errors.append("D1 acceptance fixture_mode must be false")
    if payload.get("evidence_level") != "production_reconstruction":
        errors.append("D1 acceptance evidence_level must be production_reconstruction")
    if payload.get("scientific_result_claimed") is not False:
        errors.append("D1 acceptance may not claim a scientific result")

    dataset_results = payload.get("dataset_results")
    if not isinstance(dataset_results, list):
        errors.append("D1 acceptance dataset_results must be a list")
        dataset_results = []
    elif not all(isinstance(item, Mapping) for item in dataset_results):
        errors.append("D1 acceptance dataset_results entries must be objects")

    gate_objects: dict[str, Mapping[str, Any]] = {}
    for key in (
        "required_artifact_validation",
        "global_store_validation",
        "config_binding_validation",
        "dataset_manifest_binding_validation",
        "builder_audit_validation",
    ):
        value = _require_mapping(payload.get(key), f"D1 acceptance {key}", errors)
        if value is not None:
            gate_objects[key] = value

    gate = payload.get("phase_gate_passed")
    if not isinstance(gate, bool):
        errors.append("D1 acceptance phase_gate_passed must be boolean")
    if not isinstance(payload.get("structural_validation_passed"), bool):
        errors.append("D1 acceptance structural_validation_passed must be boolean")
    for key in (
        "required_supported_datasets",
        "missing_required_datasets",
        "expected_d1_scope_datasets",
        "missing_d1_scope_datasets",
    ):
        if not isinstance(payload.get(key), list) or not all(
            isinstance(item, str) for item in payload.get(key, [])
        ):
            errors.append(f"D1 acceptance {key} must be a list of strings")

    if require_pass:
        if gate is not True:
            errors.append("D1 acceptance phase_gate_passed is not true")
        if payload.get("structural_validation_passed") is not True:
            errors.append("D1 acceptance structural validation did not pass")
        required_supported = payload.get("required_supported_datasets", [])
        if (
            len(required_supported) != len(D1_REQUIRED_SUPPORTED_DATASETS)
            or set(required_supported) != D1_REQUIRED_SUPPORTED_DATASETS
        ):
            errors.append("D1 acceptance required dataset scope is incomplete")
        expected_scope = payload.get("expected_d1_scope_datasets", [])
        if (
            len(expected_scope) != len(D1_SCOPE_DATASETS)
            or set(expected_scope) != D1_SCOPE_DATASETS
        ):
            errors.append("D1 acceptance expected dataset scope is not frozen")
        if payload.get("missing_required_datasets") != []:
            errors.append("D1 acceptance has missing required datasets")
        if payload.get("missing_d1_scope_datasets") != []:
            errors.append("D1 acceptance has missing scope datasets")
        observed_ids = {
            str(item.get("dataset_id") or "")
            for item in dataset_results
            if isinstance(item, Mapping)
        }
        if observed_ids != D1_SCOPE_DATASETS:
            errors.append("D1 acceptance dataset_results do not cover exact D1 scope")
        if len(dataset_results) != len(D1_SCOPE_DATASETS):
            errors.append("D1 acceptance dataset_results contain duplicate scope rows")
        ignored_dataset_false_paths: set[str] = set()
        for item_index, item in enumerate(dataset_results):
            if not isinstance(item, Mapping):
                continue
            dataset_id = str(item.get("dataset_id") or "<missing>")
            if item.get("passed") is not True:
                errors.append(f"D1 acceptance dataset {dataset_id} did not pass")
            if item.get("status") not in {"accepted", "blocked"}:
                errors.append(f"D1 acceptance dataset {dataset_id} status is invalid")
            if item.get("fixture_mode") is not False:
                errors.append(f"D1 acceptance dataset {dataset_id} is fixture evidence")
            if dataset_id in D1_ACCEPTED_DATASETS:
                if item.get("status") != "accepted":
                    errors.append(
                        f"D1 acceptance required dataset {dataset_id} "
                        "is not accepted"
                    )
                if item.get("paper_eligible") is not True:
                    errors.append(
                        f"D1 acceptance required dataset {dataset_id} "
                        "is not paper eligible"
                    )
            elif dataset_id in D1_BLOCKED_DATASETS:
                if item.get("status") != "blocked":
                    errors.append(
                        f"D1 acceptance blocked dataset {dataset_id} "
                        "was promoted from the frozen disposition"
                    )
                if item.get("paper_eligible") is not False:
                    errors.append(
                        f"D1 acceptance blocked dataset {dataset_id} "
                        "must not be paper eligible"
                    )
            checks = item.get("checks")
            if not isinstance(checks, list) or not checks:
                errors.append(
                    f"D1 acceptance dataset {dataset_id} has no validation checks"
                )
            elif not all(
                isinstance(check, Mapping) and check.get("passed") is True
                for check in checks
            ):
                errors.append(f"D1 acceptance dataset {dataset_id} has a failed check")
            if isinstance(checks, list):
                provenance_check_indexes = [
                    index
                    for index, check in enumerate(checks)
                    if isinstance(check, Mapping)
                    and check.get("name") == _D1_PROVENANCE_CHECK_NAME
                ]
                if not provenance_check_indexes:
                    errors.append(
                        f"D1 acceptance dataset {dataset_id} is missing the "
                        "required production provenance check"
                    )
                elif len(provenance_check_indexes) > 1:
                    errors.append(
                        f"D1 acceptance dataset {dataset_id} has duplicate "
                        "production provenance checks"
                    )
                else:
                    check_index = provenance_check_indexes[0]
                    check = checks[check_index]
                    ignored_dataset_false_paths.update(
                        _validate_d1_provenance_check(
                            dataset_id=dataset_id,
                            status=item.get("status"),
                            check=check,
                            check_path=(
                                f"dataset_results[{item_index}]"
                                f".checks[{check_index}]"
                            ),
                            errors=errors,
                        )
                    )
            counts = item.get("counts")
            if not isinstance(counts, Mapping) or not counts:
                errors.append(
                    f"D1 acceptance dataset {dataset_id} has no record counts"
                )
            elif not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in counts.values()
            ):
                errors.append(
                    f"D1 acceptance dataset {dataset_id} has invalid record counts"
                )
        for key, value in gate_objects.items():
            if value.get("passed") is not True:
                errors.append(f"D1 acceptance {key} did not pass")
            checks = value.get("semantic_checks", value.get("checks"))
            if not isinstance(checks, Mapping) or not checks:
                errors.append(f"D1 acceptance {key} has no detailed checks")
        artifact_gate = gate_objects.get("required_artifact_validation")
        if artifact_gate is not None:
            for field in ("semantic_checks", "content_checks", "binding_checks"):
                value = artifact_gate.get(field)
                invalid = not isinstance(value, Mapping) or not value
                if field in {"content_checks", "binding_checks"}:
                    invalid = invalid or set(value or {}) != (
                        D1_REQUIRED_ARTIFACT_PATHS
                    )
                if isinstance(value, Mapping):
                    invalid = invalid or any(
                        predicate is not True for predicate in value.values()
                    )
                if invalid:
                    errors.append(
                        f"D1 acceptance required artifact {field} is incomplete"
                    )
            artifacts = artifact_gate.get("artifacts")
            if (
                not isinstance(artifacts, Mapping)
                or set(artifacts) != D1_REQUIRED_ARTIFACT_PATHS
            ):
                errors.append("D1 acceptance required artifact inventory is incomplete")
            else:
                for relative, reference in artifacts.items():
                    if not (
                        isinstance(reference, Mapping)
                        and reference.get("exists") is True
                        and isinstance(reference.get("bytes"), int)
                        and not isinstance(reference.get("bytes"), bool)
                        and reference.get("bytes", 0) > 0
                        and _is_sha256(reference.get("sha256"))
                    ):
                        errors.append(f"D1 acceptance artifact {relative} is not bound")
        nested_failures = _nested_false_gate_paths(
            {"dataset_results": dataset_results},
            ignored_false_paths=frozenset(ignored_dataset_false_paths),
        ) + _nested_false_gate_paths(gate_objects)
        if nested_failures:
            errors.append(
                "D1 acceptance contains false nested gate predicates: "
                + ", ".join(sorted(nested_failures))
            )
    else:
        if gate is not False:
            errors.append("non-passing D1 acceptance must set phase_gate_passed=false")
        failure_signals = (
            payload.get("structural_validation_passed") is False
            or bool(payload.get("missing_required_datasets"))
            or bool(payload.get("missing_d1_scope_datasets"))
            or bool(
                _nested_false_gate_paths({"dataset_results": dataset_results})
                or _nested_false_gate_paths(gate_objects)
            )
        )
        if not failure_signals:
            errors.append("non-passing D1 acceptance lacks a concrete failure signal")
    return errors


def _b0_split_identity(value: Mapping[str, Any]) -> tuple[str, str, Any]:
    split_kind = str(value.get("split_kind") or "")
    if split_kind == "cross_region_transfer":
        return (
            split_kind,
            str(value.get("source_region") or ""),
            str(value.get("target_region") or ""),
        )
    return (split_kind, str(value.get("region") or ""), None)


def _validate_b0_foundation_state(
    value: Any,
    *,
    label: str,
    errors: list[str],
) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object")
    elif dict(value) != B0_FOUNDATION_STATE:
        errors.append(f"{label} is not exact UNKNOWN_PENDING_FM0 evidence")


def _validate_b0_leakage_gate_summary(
    value: Mapping[str, Any],
    *,
    label: str,
    errors: list[str],
) -> None:
    counts = value.get("counts")
    if not isinstance(counts, Mapping):
        errors.append(f"{label} counts must be an object")
    else:
        if set(counts) != B0_LEAKAGE_COUNT_KEYS:
            errors.append(f"{label} counts field inventory is not exact")
        invalid_count_names = {
            count_name
            for count_name in B0_LEAKAGE_COUNT_KEYS
            if (
                not isinstance(counts.get(count_name), int)
                or isinstance(counts.get(count_name), bool)
                or counts.get(count_name, -1) < 0
            )
        }
        for count_name in sorted(invalid_count_names):
            errors.append(f"{label} {count_name} is not a non-negative integer")
        for count_name in B0_LEAKAGE_REQUIRED_ZERO_COUNTS:
            count = counts.get(count_name)
            if not isinstance(count, int) or isinstance(count, bool) or count != 0:
                errors.append(f"{label} {count_name} is not zero")
        if not invalid_count_names:
            if counts.get("metadata_overlap_count") != (
                counts.get("explained_metadata_overlap_count")
                + counts.get("unexplained_metadata_overlap_count")
            ):
                errors.append(f"{label} metadata overlap counts are inconsistent")
            if counts.get("unexplained_overlap_count") != (
                counts.get("unexplained_metadata_overlap_count")
                + counts.get("record_role_overlap_count")
                + counts.get("component_role_overlap_count")
                + counts.get("frozen_universe_issue_count")
            ):
                errors.append(f"{label} unexplained overlap counts are inconsistent")

    gates = value.get("acceptance_gates")
    if (
        not isinstance(gates, Mapping)
        or set(gates) != B0_LEAKAGE_GATE_KEYS
        or any(gates.get(key) is not True for key in B0_LEAKAGE_GATE_KEYS)
    ):
        errors.append(f"{label} acceptance gates are not exact and all true")
    if value.get("gate_passed") is not True:
        errors.append(f"{label} gate_passed is not true")
    _validate_b0_foundation_state(
        value.get("foundation_pretraining_overlap"),
        label=f"{label} foundation state",
        errors=errors,
    )


def _validate_b0_recomputed_reports(
    value: Any,
    *,
    observed: Mapping[str, Any] | None,
    errors: list[str],
) -> None:
    label = "B0 acceptance recomputed_leakage_reports"
    if not isinstance(value, list):
        errors.append(f"{label} must be a list")
        return
    if len(value) != len(B0_SPLIT_IDENTITIES):
        errors.append(f"{label} must contain exactly five reports")

    split_universe = (
        observed.get("split_universe") if isinstance(observed, Mapping) else None
    )
    identities: list[tuple[str, str, Any]] = []
    for report_index, report in enumerate(value):
        report_label = f"{label}[{report_index}]"
        if not isinstance(report, Mapping):
            errors.append(f"{report_label} must be an object")
            continue
        if set(report) != B0_RECOMPUTED_REPORT_KEYS:
            errors.append(f"{report_label} field inventory is not exact")
        if report.get("schema_version") != "utr_b0_leakage_report.v2":
            errors.append(f"{report_label} schema_version is invalid")

        identity = _b0_split_identity(report)
        identities.append(identity)
        expected_partition_ids = B0_REQUIRED_PARTITION_IDS.get(identity)
        if expected_partition_ids is None:
            errors.append(f"{report_label} split identity is invalid")

        _validate_b0_leakage_gate_summary(
            report,
            label=report_label,
            errors=errors,
        )
        if report.get("structural_issues") != []:
            errors.append(f"{report_label} has structural issues")
        if report.get("recomputed_from_bound_structural_records") is not True:
            errors.append(
                f"{report_label} was not recomputed from bound structural records"
            )
        if report.get("canonical_manifest_exact_recomputation") is not True:
            errors.append(
                f"{report_label} canonical manifest was not exactly recomputed"
            )
        if not _is_sha256(report.get("canonical_manifest_core_sha256")):
            errors.append(f"{report_label} canonical manifest core SHA is invalid")

        for path_field in (
            "structural_records_path",
            "canonical_records_path",
            "split_manifest_path",
        ):
            path_value = report.get(path_field)
            if not isinstance(path_value, str) or not Path(path_value).is_absolute():
                errors.append(f"{report_label} {path_field} is not absolute")
        for sha_field in (
            "structural_records_sha256",
            "canonical_records_sha256",
            "canonical_record_ids_sha256",
            "structural_record_ids_sha256",
            "structural_content_sha256",
            "split_manifest_sha256",
        ):
            if not _is_sha256(report.get(sha_field)):
                errors.append(f"{report_label} {sha_field} is invalid")
        for bytes_field in ("structural_records_bytes", "split_manifest_bytes"):
            bytes_value = report.get(bytes_field)
            if (
                not isinstance(bytes_value, int)
                or isinstance(bytes_value, bool)
                or bytes_value <= 0
            ):
                errors.append(f"{report_label} {bytes_field} is invalid")
        for count_field in ("canonical_record_count", "structural_record_count"):
            count_value = report.get(count_field)
            if (
                not isinstance(count_value, int)
                or isinstance(count_value, bool)
                or count_value <= 0
            ):
                errors.append(f"{report_label} {count_field} is invalid")

        if isinstance(split_universe, Mapping):
            for field in B0_SPLIT_UNIVERSE_KEYS:
                if report.get(field) != split_universe.get(field):
                    errors.append(
                        f"{report_label} {field} differs from observed split universe"
                    )

        common_universe = report.get("common_universe_binding")
        expected_common = (
            {
                "full_record_count": split_universe.get("canonical_record_count"),
                "full_record_ids_sha256": split_universe.get(
                    "canonical_record_ids_sha256"
                ),
                "full_record_universe_sha256": split_universe.get(
                    "structural_content_sha256"
                ),
            }
            if isinstance(split_universe, Mapping)
            else None
        )
        if (
            not isinstance(common_universe, Mapping)
            or dict(common_universe) != expected_common
        ):
            errors.append(f"{report_label} common universe binding differs")

        foundation_path = report.get("foundation_exposure_path")
        foundation_sha = report.get("foundation_exposure_sha256")
        if foundation_path is None:
            if foundation_sha is not None:
                errors.append(
                    f"{report_label} foundation exposure path/SHA binding differs"
                )
        elif (
            not isinstance(foundation_path, str)
            or not Path(foundation_path).is_absolute()
            or not _is_sha256(foundation_sha)
        ):
            errors.append(
                f"{report_label} foundation exposure path/SHA binding is invalid"
            )

        auditor = report.get("auditor_binding")
        if not isinstance(auditor, Mapping) or set(auditor) != B0_AUDITOR_BINDING_KEYS:
            errors.append(f"{report_label} auditor binding is incomplete")
        else:
            if auditor.get("schema_version") != "utr_b0_leakage_auditor.v2":
                errors.append(f"{report_label} auditor schema is invalid")
            for path_field in ("entrypoint_path", "canonical_auditor_path"):
                path_value = auditor.get(path_field)
                if (
                    not isinstance(path_value, str)
                    or not Path(path_value).is_absolute()
                ):
                    errors.append(
                        f"{report_label} auditor {path_field} is not absolute"
                    )
            for sha_field in (
                "entrypoint_sha256",
                "canonical_auditor_sha256",
            ):
                if not _is_sha256(auditor.get(sha_field)):
                    errors.append(f"{report_label} auditor {sha_field} is invalid")

        required_ids = report.get("required_partition_ids")
        partitions = report.get("partitions")
        if (
            not isinstance(required_ids, list)
            or not all(isinstance(item, str) and item for item in required_ids)
            or len(required_ids) != len(set(required_ids))
            or (
                expected_partition_ids is not None
                and set(required_ids) != expected_partition_ids
            )
        ):
            errors.append(f"{report_label} required partition inventory is invalid")
            required_ids = []
        if not isinstance(partitions, list) or not partitions:
            errors.append(f"{report_label} partition count is invalid")
            partitions = []
        partition_count = report.get("partition_count")
        if (
            not isinstance(partition_count, int)
            or isinstance(partition_count, bool)
            or partition_count != len(partitions)
        ):
            errors.append(f"{report_label} partition count is invalid")
        observed_partition_ids: list[str] = []
        partition_counts: list[Mapping[str, Any]] = []
        for partition_index, partition in enumerate(partitions):
            partition_label = f"{report_label}.partitions[{partition_index}]"
            if not isinstance(partition, Mapping):
                errors.append(f"{partition_label} must be an object")
                continue
            if set(partition) != B0_PARTITION_REPORT_KEYS:
                errors.append(f"{partition_label} field inventory is not exact")
            if partition.get("schema_version") != "utr_b0_partition_leakage_report.v2":
                errors.append(f"{partition_label} schema_version is invalid")
            partition_id = partition.get("partition_id")
            if isinstance(partition_id, str) and partition_id:
                observed_partition_ids.append(partition_id)
            else:
                errors.append(f"{partition_label} partition_id is invalid")
            expected_heldout_study = (
                partition_id.split(":", 1)[1]
                if (
                    identity[0] == "study_disjoint"
                    and isinstance(partition_id, str)
                    and partition_id.startswith("loso:")
                )
                else None
            )
            if partition.get("heldout_study") != expected_heldout_study:
                errors.append(
                    f"{partition_label} heldout_study differs from its "
                    "partition identity"
                )
            if not _is_sha256(partition.get("split_partition_sha256")):
                errors.append(f"{partition_label} split partition SHA is invalid")
            for field in (
                "split_kind",
                "region",
                "source_region",
                "target_region",
            ):
                if partition.get(field) != report.get(field):
                    errors.append(f"{partition_label} {field} differs from its report")
            for field in ("metadata_axis_status", "required_axis_status", "examples"):
                if not isinstance(partition.get(field), Mapping):
                    errors.append(f"{partition_label} {field} must be an object")
            _validate_b0_leakage_gate_summary(
                partition,
                label=partition_label,
                errors=errors,
            )
            if isinstance(partition.get("counts"), Mapping):
                partition_counts.append(partition["counts"])
        if len(observed_partition_ids) != len(set(observed_partition_ids)) or set(
            observed_partition_ids
        ) != set(required_ids):
            errors.append(f"{report_label} partition identities differ")
        report_counts = report.get("counts")
        if (
            isinstance(report_counts, Mapping)
            and len(partition_counts) == len(partitions)
            and all(set(counts) == B0_LEAKAGE_COUNT_KEYS for counts in partition_counts)
            and all(
                isinstance(counts.get(name), int)
                and not isinstance(counts.get(name), bool)
                and counts.get(name, -1) >= 0
                for counts in partition_counts
                for name in B0_LEAKAGE_COUNT_KEYS
            )
        ):
            expected_report_counts = {
                name: sum(counts[name] for counts in partition_counts)
                for name in B0_LEAKAGE_COUNT_KEYS
            }
            if dict(report_counts) != expected_report_counts:
                errors.append(
                    f"{report_label} counts do not equal its partition aggregate"
                )

    if (
        len(identities) != len(set(identities))
        or set(identities) != B0_SPLIT_IDENTITIES
    ):
        errors.append(f"{label} split identities are not exact")


def _validate_b0_supplied_report_files(
    value: Any,
    *,
    errors: list[str],
) -> None:
    label = "B0 acceptance supplied_leakage_report_files"
    if not isinstance(value, list):
        errors.append(f"{label} must be a list")
        return
    if len(value) != len(B0_SPLIT_IDENTITIES):
        errors.append(f"{label} must contain exactly five references")
    paths: list[str] = []
    hashes: list[str] = []
    for index, reference in enumerate(value):
        item_label = f"{label}[{index}]"
        if not isinstance(reference, Mapping) or set(reference) != {
            "path",
            "bytes",
            "sha256",
        }:
            errors.append(f"{item_label} is not an exact file reference")
            continue
        path_value = reference.get("path")
        if not isinstance(path_value, str) or not Path(path_value).is_absolute():
            errors.append(f"{item_label} path is not absolute")
        else:
            paths.append(path_value)
        bytes_value = reference.get("bytes")
        if (
            not isinstance(bytes_value, int)
            or isinstance(bytes_value, bool)
            or bytes_value <= 0
        ):
            errors.append(f"{item_label} bytes is invalid")
        sha_value = reference.get("sha256")
        if not _is_sha256(sha_value):
            errors.append(f"{item_label} sha256 is invalid")
        else:
            hashes.append(sha_value)
    if len(paths) != len(set(paths)) or len(hashes) != len(set(hashes)):
        errors.append(f"{label} references must have unique paths and SHA values")


def _validate_b0(payload: Mapping[str, Any], require_pass: bool) -> list[str]:
    errors: list[str] = []
    _require_keys(payload, B0_REQUIRED_KEYS, "B0 acceptance", errors)
    if payload.get("schema_version") != "utr_b0_acceptance.v2":
        errors.append("B0 acceptance schema_version is invalid")
    gate = payload.get("b0_gate_passed")
    if not isinstance(gate, bool):
        errors.append("B0 acceptance b0_gate_passed must be boolean")
    failed_gates = payload.get("failed_gates")
    if not isinstance(failed_gates, list) or not all(
        isinstance(item, str) and item for item in (failed_gates or [])
    ):
        errors.append("B0 acceptance failed_gates must be a list of non-empty strings")
        failed_gates = []
    observed = _require_mapping(
        payload.get("observed"), "B0 acceptance observed", errors
    )
    exposure = _require_mapping(
        payload.get("exposure_ledger"), "B0 acceptance exposure_ledger", errors
    )
    track = _require_mapping(
        payload.get("track_role_audit"), "B0 acceptance track_role_audit", errors
    )
    seal = _require_mapping(
        payload.get("track_a_label_seal_audit"),
        "B0 acceptance track_a_label_seal_audit",
        errors,
    )
    required = _require_mapping(
        payload.get("required_artifact_audit"),
        "B0 acceptance required_artifact_audit",
        errors,
    )
    d1_binding = _require_mapping(
        payload.get("d1_exposure_ledger_binding"),
        "B0 acceptance d1_exposure_ledger_binding",
        errors,
    )
    supplied_match = payload.get("supplied_leakage_reports_match_recomputation")
    if not isinstance(supplied_match, bool):
        errors.append(
            "B0 acceptance supplied_leakage_reports_match_recomputation "
            "must be boolean"
        )
    if payload.get("allowed_claim") != "NONE":
        errors.append("B0 acceptance allowed_claim must be NONE")
    if payload.get("requires_fm0_reaudit") is not True:
        errors.append("B0 acceptance must require FM0 re-audit")
    if payload.get("re_audit_required_before_foundation_use") is not True:
        errors.append("B0 acceptance must block foundation use before re-audit")
    if payload.get("claim_boundary") != B0_CLAIM_BOUNDARY:
        errors.append("B0 acceptance claim_boundary is invalid")

    gate_objects = {
        key: value
        for key, value in {
            "exposure_ledger": exposure,
            "track_role_audit": track,
            "track_a_label_seal_audit": seal,
            "required_artifact_audit": required,
            "d1_exposure_ledger_binding": d1_binding,
        }.items()
        if value is not None
    }
    if required is not None and require_pass:
        artifacts = required.get("artifacts")
        if not isinstance(artifacts, Mapping):
            errors.append("B0 required_artifact_audit.artifacts must be an object")
        elif set(artifacts) != B0_BOUND_ARTIFACTS:
            errors.append("B0 required artifact binding set is incomplete")
        claims = _require_mapping(
            required.get("claims"),
            "B0 acceptance required_artifact_audit.claims",
            errors,
        )
        if claims is not None:
            expected_claims = {
                "schema_valid": True,
                "allowed_claims_present": True,
                "unsupported_capabilities_present": True,
                "foundation_status": "UNKNOWN_PENDING_FM0",
                "allowed_claim": "NONE",
                "requires_fm0_reaudit": True,
                "gse246381_role": (
                    "historically_exposed_retrospective_external_stress_test"
                ),
            }
            if set(claims) != set(expected_claims):
                errors.append("B0 required artifact claims field inventory is invalid")
            for key, expected in expected_claims.items():
                if claims.get(key) != expected:
                    errors.append(f"B0 required artifact claims field {key} is invalid")

    if require_pass:
        if gate is not True:
            errors.append("B0 acceptance b0_gate_passed is not true")
        if failed_gates != []:
            errors.append("B0 acceptance has failed gates")
        if supplied_match is not True:
            errors.append(
                "B0 acceptance supplied leakage reports differ " "from recomputation"
            )
        _validate_b0_recomputed_reports(
            payload.get("recomputed_leakage_reports"),
            observed=observed,
            errors=errors,
        )
        _validate_b0_supplied_report_files(
            payload.get("supplied_leakage_report_files"),
            errors=errors,
        )
        if observed is not None:
            if set(observed) != B0_OBSERVED_KEYS:
                errors.append("B0 acceptance observed inventory is not sealed")
            if not _is_exact_int(
                observed.get("leakage_report_count"),
                len(B0_SPLIT_IDENTITIES),
            ):
                errors.append(
                    "B0 acceptance leakage_report_count must be exact integer 5"
                )
            if not _is_exact_float(
                observed.get("exposure_ledger_coverage"),
                1.0,
            ):
                errors.append(
                    "B0 acceptance exposure_ledger_coverage " "must be exact float 1.0"
                )
            if not _is_exact_int(
                observed.get("track_role_ambiguity_count"),
                0,
            ):
                errors.append(
                    "B0 acceptance track_role_ambiguity_count "
                    "must be exact integer 0"
                )
            if observed.get("track_identity_universe_complete") is not True:
                errors.append("B0 acceptance track identity universe is incomplete")
            if observed.get("track_eligible_identity_universe_complete") is not True:
                errors.append(
                    "B0 acceptance eligible track identity universe is incomplete"
                )
            if not isinstance(
                observed.get("split_identities"), list
            ) or not observed.get("split_identities"):
                errors.append("B0 acceptance split identity inventory is empty")
            else:
                try:
                    split_identities = {
                        tuple(identity)
                        for identity in observed["split_identities"]
                        if isinstance(identity, list) and len(identity) == 3
                    }
                except TypeError:
                    split_identities = set()
                if (
                    len(observed["split_identities"]) != len(B0_SPLIT_IDENTITIES)
                    or split_identities != B0_SPLIT_IDENTITIES
                ):
                    errors.append("B0 acceptance split identity inventory is not exact")
            split_universe = observed.get("split_universe")
            if (
                not isinstance(split_universe, Mapping)
                or set(split_universe) != B0_SPLIT_UNIVERSE_KEYS
                or not isinstance(split_universe.get("canonical_record_count"), int)
                or isinstance(split_universe.get("canonical_record_count"), bool)
                or split_universe.get("canonical_record_count", 0) <= 0
                or split_universe.get("structural_record_count")
                != split_universe.get("canonical_record_count")
                or split_universe.get("structural_record_ids_sha256")
                != split_universe.get("canonical_record_ids_sha256")
                or not isinstance(split_universe.get("structural_records_bytes"), int)
                or isinstance(split_universe.get("structural_records_bytes"), bool)
                or split_universe.get("structural_records_bytes", 0) <= 0
                or any(
                    not _is_sha256(split_universe.get(field))
                    for field in (
                        "canonical_records_sha256",
                        "structural_records_sha256",
                        "canonical_record_ids_sha256",
                        "structural_record_ids_sha256",
                        "structural_content_sha256",
                    )
                )
            ):
                errors.append("B0 acceptance split universe is invalid")
            eligible_universe = observed.get("eligible_track_role_universe")
            if (
                not isinstance(eligible_universe, Mapping)
                or set(eligible_universe) != B0_ELIGIBLE_UNIVERSE_KEYS
                or any(
                    not _is_sha256(eligible_universe.get(field))
                    for field in (
                        "record_ids_sha256",
                        "excluded_record_ids_sha256",
                    )
                )
                or any(
                    not isinstance(eligible_universe.get(field), int)
                    or isinstance(eligible_universe.get(field), bool)
                    or eligible_universe.get(field, -1) < 0
                    for field in ("record_count", "excluded_record_count")
                )
                or eligible_universe.get("record_count", 0) <= 0
            ):
                errors.append("B0 acceptance eligible track universe is empty")
            if observed.get("eligible_track_role_universe_failures") != []:
                errors.append("B0 acceptance has eligible track-role failures")
            foundation_states = observed.get("foundation_states")
            if not isinstance(foundation_states, list) or len(foundation_states) != len(
                B0_SPLIT_IDENTITIES
            ):
                errors.append("B0 foundation overlap state is not frozen pending FM0")
            else:
                for index, state in enumerate(foundation_states):
                    _validate_b0_foundation_state(
                        state,
                        label=("B0 acceptance observed foundation_states" f"[{index}]"),
                        errors=errors,
                    )
        if exposure is not None:
            required_exposure = {
                "coverage",
                "covered",
                "expected",
                "identity_level",
                "missing",
                "extra",
                "required_ledger_identity_count",
                "missing_from_required_ledger_scope",
                "outside_required_ledger_scope",
                "ledger_scope_gate_passed",
                "gate_passed",
            }
            if set(exposure) != required_exposure:
                errors.append("B0 exposure ledger audit is incomplete")
            covered = exposure.get("covered")
            expected = exposure.get("expected")
            identity_level = exposure.get("identity_level")
            if (
                not _is_exact_float(exposure.get("coverage"), 1.0)
                or type(covered) is not int
                or covered <= 0
                or type(expected) is not int
                or expected != covered
                or identity_level not in {"record_id", "dataset_id"}
                or exposure.get("missing") != []
                or not isinstance(exposure.get("extra"), list)
                or not all(
                    isinstance(item, str) and item for item in exposure.get("extra", [])
                )
                or len(exposure.get("extra", [])) != len(set(exposure.get("extra", [])))
                or exposure.get("missing_from_required_ledger_scope") != []
                or exposure.get("outside_required_ledger_scope") != []
                or exposure.get("ledger_scope_gate_passed") is not True
                or exposure.get("gate_passed") is not True
            ):
                errors.append("B0 exposure ledger audit did not pass")
            if identity_level == "dataset_id":
                if not _is_exact_int(
                    exposure.get("required_ledger_identity_count"),
                    len(D1_SCOPE_DATASETS),
                ):
                    errors.append("B0 dataset exposure ledger scope is not exact")
            elif exposure.get("required_ledger_identity_count") is not None:
                errors.append("B0 record exposure ledger has a dataset scope")
        if track is not None:
            required_track = {
                "schema_version",
                "track_count",
                "track_role_ambiguity_count",
                "identity_universe_complete",
                "eligible_identity_binding_checked",
                "eligible_identity_binding_complete",
                "task_structural_binding_checked",
                "task_structural_binding_complete",
                "gate_passed",
                "issues",
                "universe_binding",
                "identity_universes",
                "data_card_counts",
                "tracks",
                "gse246381_role",
            }
            if not _has_fields(track, required_track):
                errors.append("B0 track-role audit is incomplete")
            track_count = track.get("track_count")
            track_role_ambiguity_count = track.get("track_role_ambiguity_count")
            track_types = {
                item.get("track_type")
                for item in track.get("tracks", [])
                if isinstance(item, Mapping)
            }
            if not (
                track.get("schema_version") == "utr_track_role_audit.v2"
                and _is_exact_int(track_count, 3)
                and _is_exact_int(track_role_ambiguity_count, 0)
                and track.get("identity_universe_complete") is True
                and track.get("eligible_identity_binding_checked") is True
                and track.get("eligible_identity_binding_complete") is True
                and track.get("task_structural_binding_checked") is True
                and track.get("task_structural_binding_complete") is True
                and track.get("gate_passed") is True
                and track.get("issues") == []
                and len(track.get("tracks", [])) == 3
                and track_types
                == {
                    "closed_measured_pool",
                    "heldout_generative",
                    "open_legal_generation",
                }
                and track.get("gse246381_role")
                == ("historically_exposed_retrospective_external_stress_test")
            ):
                errors.append("B0 track-role audit did not pass")
        if observed is not None and exposure is not None:
            if observed.get("exposure_ledger_coverage") != exposure.get(
                "coverage"
            ) or observed.get("exposure_identity_level") != exposure.get(
                "identity_level"
            ):
                errors.append("B0 observed exposure summary differs from its audit")
        if observed is not None and track is not None:
            if (
                observed.get("track_role_ambiguity_count")
                != track.get("track_role_ambiguity_count")
                or observed.get("track_identity_universe_complete")
                is not track.get("identity_universe_complete")
                or observed.get("track_eligible_identity_universe_complete")
                is not track.get("eligible_identity_binding_complete")
            ):
                errors.append("B0 observed track summary differs from its audit")
            split_universe = observed.get("split_universe")
            track_universe = track.get("universe_binding")
            if isinstance(split_universe, Mapping) and isinstance(
                track_universe, Mapping
            ):
                for field in (
                    "canonical_records_sha256",
                    "structural_records_sha256",
                ):
                    if split_universe.get(field) != track_universe.get(field):
                        errors.append(f"B0 split/track universe {field} differs")
                eligible_universe = observed.get("eligible_track_role_universe")
                if not (
                    isinstance(eligible_universe, Mapping)
                    and eligible_universe.get("record_count")
                    == track_universe.get("record_count")
                    and eligible_universe.get("record_ids_sha256")
                    == track_universe.get("record_ids_sha256")
                ):
                    errors.append(
                        "B0 eligible track identities differ " "from the track universe"
                    )
        if seal is not None:
            required_seal = {
                "schema_version",
                "track_id",
                "gate_passed",
                "candidate_label_bijection",
                "record_label_bijection",
                "strict_hidden_label_schema_passed",
                "paired_finite_measured_labels",
                "canonical_identity_binding_passed",
                "d1_acceptance_binding_passed",
                "current_d1_chain_binding_passed",
                "role_policy_exact_binding_passed",
                "label_store_sha256",
                "label_store_bytes",
                "freeze_proof_sha256",
                "selection_freeze_sha256",
                "role_policy_sha256",
                "hidden_label_schema_sha256",
                "d1_acceptance_sha256",
                "d1_build_manifest_sha256",
                "candidate_ids_sha256",
                "candidate_count",
                "label_record_ids_sha256",
                "label_record_count",
                "canonical_records_sha256",
                "structural_records_sha256",
                "record_ids_sha256",
            }
            if not _has_fields(seal, required_seal):
                errors.append("B0 Track A privileged seal is incomplete")
            if seal.get("schema_version") != "utr_track_a_label_seal_audit.v2":
                errors.append("B0 Track A privileged seal schema is invalid")
            for field in (
                "gate_passed",
                "candidate_label_bijection",
                "record_label_bijection",
                "strict_hidden_label_schema_passed",
                "paired_finite_measured_labels",
                "canonical_identity_binding_passed",
                "d1_acceptance_binding_passed",
                "current_d1_chain_binding_passed",
                "role_policy_exact_binding_passed",
            ):
                if seal.get(field) is not True:
                    errors.append(f"B0 Track A privileged seal {field} did not pass")
            for field in required_seal - {"schema_version", "track_id"}:
                if field.endswith("_sha256") and not _is_sha256(seal.get(field)):
                    errors.append(f"B0 Track A privileged seal {field} is invalid")
            for field in (
                "label_store_bytes",
                "candidate_count",
                "label_record_count",
            ):
                value = seal.get(field)
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    errors.append(f"B0 Track A privileged seal {field} is invalid")
        if required is not None:
            required_audit_fields = {
                "schema_version",
                "binding_manifest_path",
                "binding_manifest_sha256",
                "gate_passed",
                "failures",
                "universe_binding",
                "artifacts",
                "claims",
            }
            if set(required) != required_audit_fields:
                errors.append("B0 required artifact audit is incomplete")
            if (
                required.get("schema_version") != "utr_b0_required_artifact_audit.v2"
                or required.get("gate_passed") is not True
                or required.get("failures", []) != []
                or not _is_sha256(required.get("binding_manifest_sha256"))
                or not isinstance(required.get("binding_manifest_path"), str)
                or not Path(required.get("binding_manifest_path", "")).is_absolute()
            ):
                errors.append("B0 required artifact audit did not pass")
            artifacts = required.get("artifacts")
            if not isinstance(artifacts, Mapping):
                errors.append("B0 required artifact binding set is invalid")
            else:
                for name, artifact in artifacts.items():
                    if not (
                        isinstance(artifact, Mapping)
                        and set(artifact)
                        == {"path", "exists", "bytes", "sha256", "schema_valid"}
                        and isinstance(artifact.get("path"), str)
                        and Path(artifact.get("path", "")).is_absolute()
                        and artifact.get("exists") is True
                        and artifact.get("schema_valid") is True
                        and _is_sha256(artifact.get("sha256"))
                        and isinstance(artifact.get("bytes"), int)
                        and not isinstance(artifact.get("bytes"), bool)
                        and artifact.get("bytes", 0) > 0
                    ):
                        errors.append(
                            f"B0 required artifact {name} is not exactly bound"
                        )
            track_universe = (
                track.get("universe_binding") if isinstance(track, Mapping) else None
            )
            if required.get("universe_binding") != track_universe:
                errors.append("B0 required artifact audit universe differs from tracks")
        if d1_binding is not None:
            if set(d1_binding) != B0_D1_EXPOSURE_BINDING_KEYS:
                errors.append("B0 D1 exposure-ledger binding inventory is invalid")
            if (
                d1_binding.get("schema_version") != "utr_b0_d1_exposure_binding.v2"
                or d1_binding.get("gate_passed") is not True
                or d1_binding.get("failures") != []
                or d1_binding.get("ledger_semantics_valid") is not True
            ):
                errors.append("B0 D1 exposure-ledger binding did not pass")
            for path_field in (
                "d1_acceptance_path",
                "d1_build_manifest_path",
                "exposure_ledger_path",
            ):
                path_value = d1_binding.get(path_field)
                if (
                    not isinstance(path_value, str)
                    or not Path(path_value).is_absolute()
                ):
                    errors.append(
                        f"B0 D1 exposure-ledger binding {path_field} is invalid"
                    )
            for sha_field in (
                "d1_acceptance_sha256",
                "d1_build_manifest_sha256",
                "exposure_ledger_sha256",
            ):
                if not _is_sha256(d1_binding.get(sha_field)):
                    errors.append(
                        f"B0 D1 exposure-ledger binding {sha_field} is invalid"
                    )
            ledger_bytes = d1_binding.get("exposure_ledger_bytes")
            if (
                not isinstance(ledger_bytes, int)
                or isinstance(ledger_bytes, bool)
                or ledger_bytes <= 0
            ):
                errors.append("B0 D1 exposure-ledger binding bytes is invalid")
        if seal is not None and track is not None:
            universe = track.get("universe_binding")
            if isinstance(universe, Mapping):
                for seal_field, universe_field in (
                    ("candidate_ids_sha256", "candidate_ids_sha256"),
                    ("canonical_records_sha256", "canonical_records_sha256"),
                    ("structural_records_sha256", "structural_records_sha256"),
                    ("record_ids_sha256", "record_ids_sha256"),
                ):
                    if seal.get(seal_field) != universe.get(universe_field):
                        errors.append(
                            f"B0 Track A privileged seal {seal_field} "
                            "differs from the track universe"
                        )
        if seal is not None and d1_binding is not None:
            for field in (
                "d1_acceptance_sha256",
                "d1_build_manifest_sha256",
            ):
                if seal.get(field) != d1_binding.get(field):
                    errors.append(
                        f"B0 Track A privileged seal {field} "
                        "differs from the D1 exposure binding"
                    )
        if required is not None and d1_binding is not None:
            artifacts = required.get("artifacts")
            exposure_artifact = (
                artifacts.get("exposure_ledger")
                if isinstance(artifacts, Mapping)
                else None
            )
            if not (
                isinstance(exposure_artifact, Mapping)
                and exposure_artifact.get("path")
                == d1_binding.get("exposure_ledger_path")
                and exposure_artifact.get("bytes")
                == d1_binding.get("exposure_ledger_bytes")
                and exposure_artifact.get("sha256")
                == d1_binding.get("exposure_ledger_sha256")
            ):
                errors.append(
                    "B0 required exposure-ledger artifact differs "
                    "from its D1 binding"
                )
        nested_failures = _nested_false_gate_paths(gate_objects)
        if nested_failures:
            errors.append(
                "B0 acceptance contains false nested gate predicates: "
                + ", ".join(sorted(nested_failures))
            )
    else:
        if gate is not False:
            errors.append("non-passing B0 acceptance must set b0_gate_passed=false")
        if not failed_gates and not _nested_false_gate_paths(gate_objects):
            errors.append("non-passing B0 acceptance lacks a concrete failure signal")
    return errors


def validate_phase_acceptance(
    phase: str, payload: Any, *, require_pass: bool
) -> list[str]:
    """Return fail-closed document-semantic errors for one phase."""
    if not isinstance(payload, Mapping):
        return [f"{phase} acceptance document must be an object"]
    if phase == "D1":
        return _validate_d1(payload, require_pass)
    if phase == "B0":
        return _validate_b0(payload, require_pass)
    return [f"unsupported phase acceptance: {phase}"]
