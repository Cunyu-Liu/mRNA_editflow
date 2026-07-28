#!/usr/bin/env python3
"""Fail-closed document semantics for D1 and B0 acceptance artifacts.

These checks deliberately validate the full acceptance *document shape* and
the gate-bearing nested summaries.  They are defense in depth for the release
validators; the phase-specific production validators remain the source that
must recompute these documents from frozen inputs.
"""
from __future__ import annotations

import hashlib
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


def _nested_false_gate_paths(value: Any, prefix: str = "") -> list[str]:
    failures: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            is_gate_key = (
                key in _TRUE_GATE_KEYS
                or key.endswith("_passed")
                or key.endswith("_schema_valid")
                or key.endswith("_binding_complete")
            )
            if is_gate_key and isinstance(item, bool) and item is not True:
                failures.append(path)
            failures.extend(_nested_false_gate_paths(item, path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            failures.extend(_nested_false_gate_paths(item, f"{prefix}[{index}]"))
    return failures


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
        for item in dataset_results:
            if not isinstance(item, Mapping):
                continue
            dataset_id = str(item.get("dataset_id") or "<missing>")
            if item.get("passed") is not True:
                errors.append(f"D1 acceptance dataset {dataset_id} did not pass")
            if item.get("status") not in {"accepted", "blocked"}:
                errors.append(f"D1 acceptance dataset {dataset_id} status is invalid")
            if item.get("fixture_mode") is not False:
                errors.append(f"D1 acceptance dataset {dataset_id} is fixture evidence")
            if dataset_id in D1_REQUIRED_SUPPORTED_DATASETS:
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
            {"dataset_results": dataset_results}
        ) + _nested_false_paths(gate_objects)
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
                or _nested_false_paths(gate_objects)
            )
        )
        if not failure_signals:
            errors.append("non-passing D1 acceptance lacks a concrete failure signal")
    return errors


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
    if payload.get("allowed_claim") != "NONE":
        errors.append("B0 acceptance allowed_claim must be NONE")
    if payload.get("requires_fm0_reaudit") is not True:
        errors.append("B0 acceptance must require FM0 re-audit")
    if payload.get("re_audit_required_before_foundation_use") is not True:
        errors.append("B0 acceptance must block foundation use before re-audit")

    gate_objects = {
        key: value
        for key, value in {
            "exposure_ledger": exposure,
            "track_role_audit": track,
            "track_a_label_seal_audit": seal,
            "required_artifact_audit": required,
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
                "allowed_claims_present": True,
                "unsupported_capabilities_present": True,
                "foundation_status": "UNKNOWN_PENDING_FM0",
                "allowed_claim": "NONE",
                "requires_fm0_reaudit": True,
                "gse246381_role": (
                    "historically_exposed_retrospective_external_stress_test"
                ),
            }
            for key, expected in expected_claims.items():
                if claims.get(key) != expected:
                    errors.append(f"B0 required artifact claims field {key} is invalid")

    if require_pass:
        if gate is not True:
            errors.append("B0 acceptance b0_gate_passed is not true")
        if failed_gates != []:
            errors.append("B0 acceptance has failed gates")
        if observed is not None:
            if set(observed) != B0_OBSERVED_KEYS:
                errors.append("B0 acceptance observed inventory is not sealed")
            if observed.get("leakage_report_count") != len(B0_SPLIT_IDENTITIES):
                errors.append("B0 acceptance leakage report inventory is incomplete")
            if observed.get("exposure_ledger_coverage") != 1.0:
                errors.append("B0 acceptance exposure coverage is not 100%")
            if observed.get("track_role_ambiguity_count") != 0:
                errors.append("B0 acceptance has track-role ambiguity")
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
                or not isinstance(split_universe.get("canonical_record_count"), int)
                or isinstance(split_universe.get("canonical_record_count"), bool)
                or split_universe.get("canonical_record_count", 0) <= 0
            ):
                errors.append("B0 acceptance split universe is invalid")
            if not isinstance(
                observed.get("eligible_track_role_universe"), list
            ) or not observed.get("eligible_track_role_universe"):
                errors.append("B0 acceptance eligible track universe is empty")
            if observed.get("eligible_track_role_universe_failures") != []:
                errors.append("B0 acceptance has eligible track-role failures")
            foundation_states = observed.get("foundation_states")
            if (
                not isinstance(foundation_states, list)
                or len(foundation_states) != len(B0_SPLIT_IDENTITIES)
                or any(
                    not isinstance(item, Mapping)
                    or item.get("status") != "UNKNOWN_PENDING_FM0"
                    or item.get("foundation_selected") is not False
                    or item.get("allowed_claim") != "NONE"
                    or item.get("re_audit_required") is not True
                    or item.get("gate_passed") is not True
                    for item in foundation_states
                )
            ):
                errors.append("B0 foundation overlap state is not frozen pending FM0")
        if exposure is not None:
            required_exposure = {
                "coverage",
                "identity_level",
                "missing",
                "extra",
                "ledger_scope_gate_passed",
                "gate_passed",
            }
            if not _has_fields(exposure, required_exposure):
                errors.append("B0 exposure ledger audit is incomplete")
            if (
                exposure.get("coverage") != 1.0
                or exposure.get("identity_level") not in {"record_id", "dataset_id"}
                or exposure.get("missing") != []
                or exposure.get("extra") != []
                or exposure.get("ledger_scope_gate_passed") is not True
                or exposure.get("gate_passed") is not True
            ):
                errors.append("B0 exposure ledger audit did not pass")
        if track is not None:
            required_track = {
                "schema_version",
                "track_count",
                "track_role_ambiguity_count",
                "identity_universe_complete",
                "eligible_identity_binding_checked",
                "eligible_identity_binding_complete",
                "gate_passed",
                "issues",
                "universe_binding",
                "tracks",
                "gse246381_role",
            }
            if not _has_fields(track, required_track):
                errors.append("B0 track-role audit is incomplete")
            track_types = {
                item.get("track_type")
                for item in track.get("tracks", [])
                if isinstance(item, Mapping)
            }
            if not (
                track.get("schema_version") == "utr_track_role_audit.v2"
                and track.get("track_count") == 3
                and track.get("track_role_ambiguity_count") == 0
                and track.get("identity_universe_complete") is True
                and track.get("eligible_identity_binding_checked") is True
                and track.get("eligible_identity_binding_complete") is True
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
                eligible_ids = observed.get("eligible_track_role_universe")
                if isinstance(eligible_ids, list) and all(
                    isinstance(item, str) for item in eligible_ids
                ):
                    normalized = sorted(eligible_ids)
                    body = "\n".join(normalized) + "\n" if normalized else ""
                    if (
                        len(normalized) != len(set(normalized))
                        or len(normalized) != track_universe.get("record_count")
                        or hashlib.sha256(body.encode("utf-8")).hexdigest()
                        != track_universe.get("record_ids_sha256")
                    ):
                        errors.append(
                            "B0 eligible track identities differ "
                            "from the track universe"
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
                "label_store_sha256",
                "freeze_proof_sha256",
                "selection_freeze_sha256",
                "role_policy_sha256",
                "hidden_label_schema_sha256",
                "d1_acceptance_sha256",
                "d1_build_manifest_sha256",
                "candidate_ids_sha256",
                "canonical_records_sha256",
                "structural_records_sha256",
                "record_ids_sha256",
            }
            if not _has_fields(seal, required_seal):
                errors.append("B0 Track A privileged seal is incomplete")
            if seal.get("schema_version") != "utr_track_a_label_seal_audit.v2":
                errors.append("B0 Track A privileged seal schema is invalid")
            for field in required_seal - {"schema_version", "track_id"}:
                if field.endswith("_sha256") and not _is_sha256(seal.get(field)):
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
            if not _has_fields(required, required_audit_fields):
                errors.append("B0 required artifact audit is incomplete")
            if (
                required.get("schema_version") != "utr_b0_required_artifact_audit.v2"
                or required.get("gate_passed") is not True
                or required.get("failures", []) != []
                or not _is_sha256(required.get("binding_manifest_sha256"))
                or not isinstance(required.get("binding_manifest_path"), str)
                or not required.get("binding_manifest_path")
            ):
                errors.append("B0 required artifact audit did not pass")
            artifacts = required.get("artifacts")
            if not isinstance(artifacts, Mapping):
                errors.append("B0 required artifact binding set is invalid")
            else:
                for name, artifact in artifacts.items():
                    if not (
                        isinstance(artifact, Mapping)
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
        nested_failures = _nested_false_paths(gate_objects)
        if nested_failures:
            errors.append(
                "B0 acceptance contains false nested gate predicates: "
                + ", ".join(sorted(nested_failures))
            )
    else:
        if gate is not False:
            errors.append("non-passing B0 acceptance must set b0_gate_passed=false")
        if not failed_gates and not _nested_false_paths(gate_objects):
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
