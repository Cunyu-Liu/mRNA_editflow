#!/usr/bin/env python3
"""Prepare, publish, and validate the authority-only A1-EVT-058 sync.

This transaction registers only V3-DEC-024 repository authority.  It appends
three immutable EVT057 snapshots and one immutable authority-sync record, then
updates STATUS, RUN_MANIFEST, and EVENT_LOG in that order.  It registers no
evidence artifact and does not execute any GSE261709, GSE269595, or N-zip
preflight.  It preserves the
existing 1/1/0 qualified-study counts, 6547 canonical-record count, incomplete
A1 state, and every training/model/GPU/next-phase/scientific-claim lock.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any


UNKNOWN = "UNKNOWN_NOT_ASSERTED"
BOUND = "BOUND"
MUTABLE_NAMES = ("STATUS.json", "RUN_MANIFEST.json", "EVENT_LOG.jsonl")
CONFIG_REPO_PATH = "configs/route_a_v3_dec024_authority_runtime_sync_v1.json"
SCRIPT_REPO_PATH = "scripts/route_a_v3/dec024_authority_runtime_sync.py"
TEST_REPO_PATH = "tests/route_a_v3/test_dec024_authority_runtime_sync.py"
IMPLEMENTATION_PATHS = [CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH]
PRODUCTION_CONFIG_PATH = Path(__file__).resolve().parents[2] / CONFIG_REPO_PATH
PRODUCTION_REPO_ROOT = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810"
)
BRANCH = "routea-v3-a1-20260810"
AUTHORITY_PARENT = "e5d089a43d194caf59369fd12c203c0694ba40c6"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FaultInjector = Callable[[str], None]

AUTHORITY_PATHS = [
    "configs/route_a_v3.yaml",
    "configs/route_a_v3_a1_qualification.json",
    "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec024.yaml",
    "docs/contracts/supersession_mrna_xeditflow_v1_1_to_route_a_v3.yaml",
    "docs/execution/route_a_v3_a1_interim.yaml",
    "docs/execution/route_a_v3_a6_interim.yaml",
    "docs/execution/route_a_v3_data_role_registry.yaml",
    "docs/execution/route_a_v3_decision_log.yaml",
    "docs/execution/route_a_v3_registry_manifest.json",
    "docs/execution/route_a_v3_task_registry.yaml",
    "scripts/route_a_v3/validate_a0_bundle.py",
    "tests/route_a_v3/test_a0_integrity_guards.py"
]
FROZEN_G_COMMIT = "8fde46ca7daa765fa3a8ad8ce24a3da82ce1a8d0"
FROZEN_G_PATHS = [
    "configs/route_a_v3_a6_learned_base_value_g0_implementation_candidate_v1.json",
    "docs/plans/2026-08-14-route-a-v3-a6-learned-g0-implementation-candidate-v1.md",
    "scripts/route_a_v3/a6_learned_base_value_g0_candidate.py",
    "tests/route_a_v3/test_a6_learned_base_value_g0_candidate.py",
]
FROZEN_G_FILES = [
    {
        "path": FROZEN_G_PATHS[0],
        "bytes": 3282,
        "sha256": "f26ab89d8030f1c7ca91f1f60933475181b4270591532248daa4c8e1de8510f1",
    },
    {
        "path": FROZEN_G_PATHS[1],
        "bytes": 2953,
        "sha256": "371e9f2c581ec83120d0300f121d5354f4fc5d388b5afa72e4a1a0f9514595b9",
    },
    {
        "path": FROZEN_G_PATHS[2],
        "bytes": 29192,
        "sha256": "9a09df25b89ee8e08ffbb2c84d955fddffa2b93b3a5216dc3d8ee1af688fc891",
    },
    {
        "path": FROZEN_G_PATHS[3],
        "bytes": 9805,
        "sha256": "4c6ab5908f719989b42854aa73b915d7cc1d864879fa5ffc9652dcf1efb6becf",
    },
]
FROZEN_G = {
    "status": "FROZEN_BOUND_NONAUTHORITATIVE_EXACT4",
    "commit": FROZEN_G_COMMIT,
    "expected_parent": "0bb84dffb1389b9eced7e92e36ef80b8a97ed0be",
    "authority_runtime_or_science_state_change": False,
    "exact_changed_paths": FROZEN_G_PATHS,
    "files": FROZEN_G_FILES,
}
ACTIVE_DECISION_IDS = [
    "V3-DEC-017",
    "V3-DEC-018",
    "V3-DEC-019",
    "V3-DEC-020",
    "V3-DEC-021",
    "V3-DEC-022",
    "V3-DEC-023",
    "V3-DEC-024"
]
DEC024_AUTHORITY = json.loads(
    r'''{
  "decision_id": "V3-DEC-024",
  "status": "FROZEN_USER_AUTHORIZED_THREE_REPLACEMENT_AGGREGATE_ONLY_PREFLIGHTS_NO_PROMOTION",
  "authority_sync_executes_preflight": false,
  "preflight_status_after_sync": "AUTHORIZED_NOT_RUN",
  "authority_sync_assigns_dataset_role": false,
  "authority_sync_qualifies_study": false,
  "authority_sync_changes_counts": false,
  "scientific_state_changed": false,
  "gse261709": {
    "project_id": "PRJNA1088465",
    "replacement_candidate_role": "REPLACEMENT_A1_CANDIDATE_PREFLIGHT_ONLY",
    "role": "AGGREGATE_ROW_LEVEL_A1_QUALIFICATION_PREFLIGHT_ONLY",
    "authority_surface": "ORDINARY_PUBLIC_PROCESSED_ASSET_ONLY",
    "required_fail_closed_gate_ids_exactly": [
      "PUBLIC_PROCESSED_ASSET_IDENTITY_ROLE_PROVENANCE_AND_PRIMARY_MEASUREMENT_ROUTE_CLOSED",
      "BARCODE_ALLELE_TRANSCRIPT_SOURCE_AND_FULL_CONSTRUCT_JOIN_CLOSED",
      "SOURCE_CANDIDATE_IDENTITY_AND_DENSE_FAMILY_MINIMUM_THREE_CANDIDATES_CLOSED",
      "SOURCE_TO_CANDIDATE_LEGAL_EDIT_REPLAY_CLOSED",
      "ENDPOINT_DIRECTION_SCALE_EFFECT_AND_STANDARD_ERROR_SEMANTICS_CLOSED",
      "THREE_INDEPENDENT_BIOLOGICAL_REPLICATE_RNA_DNA_COUNTS_AND_VALID_STANDARD_ERROR_CLOSED",
      "MISSING_CENSORING_QC_AND_SELECTION_CLOSED",
      "LICENSE_AND_REUSE_RIGHTS_CLOSED",
      "HISTORICAL_ANALYTIC_OR_CHECKPOINT_EXPOSURE_CLOSED",
      "OUTCOME_BLIND_SOURCE_GROUP_NEAR_DUPLICATE_SPLIT_AND_ZERO_LEAKAGE_READINESS_CLOSED",
      "POST_DEDUP_INDEPENDENT_SOURCE_GROUP_EFFECTIVE_N_CLOSED",
      "PREFROZEN_SOURCE_GROUP_POWER_AND_FULL_CI_WIDTH_CLOSED"
    ],
    "minimum_distinct_candidates_per_source_family": 3,
    "full_construct_and_source_join_required": true,
    "three_biological_replicates_required": true,
    "standard_error_required": true,
    "historical_analytic_or_checkpoint_exposure": "UNKNOWN_NOT_ASSERTED",
    "unknown_historical_exposure_is_gate_blocker": true,
    "processed_public_asset_body_read_allowed": true,
    "raw_fastq_or_sra_member_payload_read_allowed": false,
    "split_assignment_execution_allowed": false,
    "formal_qualification_power_gate_execution_allowed": false,
    "member_identifier_output_allowed": false,
    "barcode_output_allowed": false,
    "transcript_output_allowed": false,
    "sequence_output_allowed": false,
    "row_endpoint_output_allowed": false,
    "row_effect_output_allowed": false,
    "row_standard_error_output_allowed": false,
    "replicate_identifier_output_allowed": false,
    "split_assignment_output_allowed": false,
    "all_required_gates_passing_automatically_qualifies_dataset": false,
    "separate_user_authority_required_for_qualification_or_counting": true
  },
  "gse269595": {
    "project_id": "PRJNA1122592",
    "replacement_candidate_role": "REPLACEMENT_A1_OR_TRUE_A2_ROLE_ADJUDICATION_PREFLIGHT_ONLY",
    "role": "AGGREGATE_SOURCE_FAMILY_ASSET_SCHEMA_GEOMETRY_AND_ROLE_ADJUDICATION_PREFLIGHT_ONLY",
    "required_fail_closed_gate_ids_exactly": [
      "A1_VERSUS_TRUE_A2_ROLE_ELIGIBILITY_AND_MUTUAL_EXCLUSIVITY_CLOSED",
      "ORDINARY_PUBLIC_ASSET_IDENTITY_ROLE_AND_PROVENANCE_CLOSED",
      "SOURCE_FAMILY_DISTRIBUTION_AND_UNIQUE_SOURCE_ANCHOR_CLOSED",
      "INTRONIC_APA_EXCLUSION_CLOSED",
      "SOURCE_TO_CANDIDATE_LEGAL_SUBSTITUTION_REPLAY_CLOSED",
      "ASSAY_CONTEXT_GUIDE_ENDPOINT_DIRECTION_SCALE_AND_SEMANTICS_CLOSED",
      "INDEPENDENT_BIOLOGICAL_REPLICATE_AND_VALID_STANDARD_ERROR_CLOSED",
      "ASSET_SCHEMA_DIMENSION_AND_COVERAGE_CLOSED",
      "MISSING_CENSORING_AND_SELECTION_CLOSED",
      "APARENT_PRIOR_EXPOSURE_AND_MODEL_INPUT_ROUTE_CLOSED",
      "LICENSE_AND_REUSE_RIGHTS_CLOSED",
      "OUTCOME_BLIND_SOURCE_GROUP_NEAR_DUPLICATE_SPLIT_AND_ZERO_LEAKAGE_READINESS_CLOSED",
      "POST_DEDUP_SOURCE_GROUP_EFFECTIVE_N_AND_PREFROZEN_POWER_FULL_CI_WIDTH_CLOSED"
    ],
    "minimum_distinct_candidates_per_source_family": 3,
    "intronic_apa_exclusion_required": true,
    "source_to_candidate_legal_substitution_replay_required": true,
    "assay_context_guide_endpoint_replicate_and_standard_error_required": true,
    "aparent_exposure_rights_split_leakage_effective_n_and_power_required": true,
    "maximum_roles_if_later_qualified": 1,
    "a1_and_true_a2_double_credit_allowed": false,
    "a1_role_may_be_presumed": false,
    "true_a2_role_may_be_presumed": false,
    "source_to_candidate_edit_relation_may_be_presumed": false,
    "historical_analytic_or_checkpoint_exposure": "UNKNOWN_NOT_ASSERTED",
    "unknown_historical_exposure_is_gate_blocker": true,
    "split_assignment_execution_allowed": false,
    "formal_qualification_power_gate_execution_allowed": false,
    "member_identifier_output_allowed": false,
    "actual_header_names_output_allowed": false,
    "sequence_output_allowed": false,
    "row_endpoint_output_allowed": false,
    "row_effect_output_allowed": false,
    "row_standard_error_output_allowed": false,
    "replicate_identifier_output_allowed": false,
    "split_assignment_output_allowed": false,
    "all_required_gates_passing_automatically_qualifies_dataset": false,
    "separate_user_authority_required_for_role_assignment_qualification_or_counting": true
  },
  "emtab10902": {
    "public_alias": "N_ZIP",
    "replacement_candidate_role": "REPLACEMENT_TRUE_A2_CANDIDATE_PREFLIGHT_ONLY",
    "role": "AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_ONLY",
    "required_fail_closed_gate_ids_exactly": [
      "INTENDED_UNIVERSE_MEMBERSHIP_CLOSED",
      "SOURCE_ANCHOR_IDENTITY_AND_FULL_REPORTER_CONTEXT_CLOSED",
      "SOURCE_TO_CANDIDATE_EDIT_REPLAY_CLOSED",
      "DENSE_SOURCE_FAMILY_MINIMUM_THREE_CANDIDATES_CLOSED",
      "ENDPOINT_DIRECTION_SCALE_UNIT_AND_SEMANTICS_CLOSED",
      "INDEPENDENT_BIOLOGICAL_REPLICATE_AND_VALID_STANDARD_ERROR_CLOSED",
      "MISSING_CENSORING_QC_AND_SELECTION_CLOSED",
      "LICENSE_REUSE_RIGHTS_AND_EXPOSURE_CLOSED",
      "OUTCOME_BLIND_SOURCE_GROUP_NEAR_DUPLICATE_SPLIT_AND_ZERO_LEAKAGE_READINESS_CLOSED",
      "POST_DEDUP_INDEPENDENT_SOURCE_GROUP_EFFECTIVE_N_CLOSED",
      "PREFROZEN_SOURCE_GROUP_POWER_AND_FULL_CI_WIDTH_CLOSED"
    ],
    "source_anchor_and_edit_replay_required": true,
    "source_anchor_may_be_inferred_from_row_order": false,
    "source_to_candidate_edit_relation_may_be_presumed": false,
    "full_reporter_context_required": true,
    "minimum_distinct_candidates_per_source_family": 3,
    "biological_replicate_and_valid_standard_error_required": true,
    "endpoint_missingness_rights_exposure_split_and_leakage_required": true,
    "reported_source_group_count_approximate": 16,
    "reported_source_group_count_is_qualification_fact": false,
    "reported_qc_design_row_count_reference_only": 5679,
    "row_count_may_substitute_for_independent_source_group_n": false,
    "prefrozen_required_effective_n_for_power_and_full_ci_width": 156,
    "power_infeasible_status_allowed": true,
    "power_infeasible_status_is_qualification_or_credit": false,
    "historical_analytic_or_checkpoint_exposure": "UNKNOWN_NOT_ASSERTED",
    "unknown_historical_exposure_is_gate_blocker": true,
    "split_assignment_execution_allowed": false,
    "formal_qualification_power_gate_execution_allowed": false,
    "member_identifier_output_allowed": false,
    "sequence_output_allowed": false,
    "row_endpoint_output_allowed": false,
    "row_effect_output_allowed": false,
    "row_standard_error_output_allowed": false,
    "replicate_identifier_output_allowed": false,
    "split_assignment_output_allowed": false,
    "true_a2_status_may_be_presumed": false,
    "all_required_gates_passing_automatically_qualifies_dataset": false,
    "separate_user_authority_required_for_qualification_or_counting": true
  },
  "qualification_allowed": false,
  "dataset_role_assignment_allowed": false,
  "canonical_materialization_allowed": false,
  "split_execution_allowed": false,
  "formal_qualification_power_gate_execution_allowed": false,
  "training_allowed": false,
  "gpu_work_allowed": false,
  "model_selection_allowed": false,
  "a7_allowed": false,
  "next_phase_authorized": false,
  "scientific_claim_status": "NOT_ESTABLISHED"
}'''
)
RUNTIME_AUTHORITY = json.loads(
    r'''{
  "historical_active_authority_commit_policy": "PRESERVE_PREDECESSOR_VALUE_UNCHANGED",
  "active_amendment_decision_ids": [
    "V3-DEC-017",
    "V3-DEC-018",
    "V3-DEC-019",
    "V3-DEC-020",
    "V3-DEC-021",
    "V3-DEC-022",
    "V3-DEC-023",
    "V3-DEC-024"
  ],
  "current_contract_authority_scope": "DEC024_GSE261709_GSE269595_EMTAB10902_AGGREGATE_ONLY_REPLACEMENT_PREFLIGHT_AUTHORITIES_NO_EXECUTION_NO_QUALIFICATION"
}'''
)
FROZEN_OUTER_TRUTH = json.loads(
    r'''{
  "current_qualified_counts": {
    "ordinary": 1,
    "a1": 1,
    "true_a2": 0,
    "canonical_records": 6547
  },
  "gse261709_contribution": {
    "ordinary": 0,
    "a1": 0,
    "true_a2": 0,
    "canonical_records": 0
  },
  "gse269595_contribution": {
    "ordinary": 0,
    "a1": 0,
    "true_a2": 0,
    "canonical_records": 0
  },
  "emtab10902_contribution": {
    "ordinary": 0,
    "a1": 0,
    "true_a2": 0,
    "canonical_records": 0
  },
  "run_status": "IN_PROGRESS",
  "evidence_status": "SCRATCH_ROUTE_QUALIFIED_GLOBAL_PHASE_INCOMPLETE",
  "gate_status": "A1_PHASE_INCOMPLETE_GLOBAL_REQUIREMENTS",
  "a1_complete": false,
  "qualified": false,
  "training_started": false,
  "training_allowed": false,
  "training_authorized": false,
  "gpu_work_started": false,
  "gpu_work_allowed": false,
  "model_selection_allowed": false,
  "a7_allowed": false,
  "next_phase_authorized": false,
  "scientific_claim_status": "NOT_ESTABLISHED"
}'''
)
ACCESS_BOUNDARY = json.loads(
    r'''{
  "new_registered_artifact_count": 0,
  "new_registered_artifact_read_count": 0,
  "registered_artifact_body_parse_count": 0,
  "registered_artifact_payload_field_read_count": 0,
  "public_asset_read_count": 0,
  "private_payload_read_count": 0,
  "private_payload_write_count": 0,
  "sealed_payload_read_count": 0,
  "member_payload_read_count": 0,
  "row_payload_read_count": 0,
  "sequence_payload_read_count": 0,
  "endpoint_payload_read_count": 0,
  "effect_payload_read_count": 0,
  "standard_error_payload_read_count": 0,
  "split_assignment_read_count": 0,
  "canonical_materialization_count": 0,
  "qualification_run_count": 0,
  "formal_power_gate_run_count": 0,
  "training_run_count": 0,
  "gpu_work_count": 0,
  "model_selection_run_count": 0,
  "restricted_or_sealed_path_accessed": false,
  "gse246381_contact": false
}'''
)
PUBLICATION_POLICY = json.loads(
    r'''{
  "registered_artifacts_remain_empty_for_this_sync": true,
  "predecessor_snapshots_are_immutable_runtime_outputs": true,
  "sync_record_is_immutable_runtime_output": true,
  "mutables_commit_after_all_immutables": true,
  "mutable_commit_order": [
    "STATUS.json",
    "RUN_MANIFEST.json",
    "EVENT_LOG.jsonl"
  ],
  "event_is_last_commit": true,
  "supported_recovery": "EXACT_PUBLICATION_PREFIX_ONLY"
}'''
)
PREDECESSOR_IDENTITIES = json.loads(
    r'''{
  "STATUS.json": {
    "bytes": 29902,
    "sha256": "d766a51e6717eefb185f683bdb7eceff27eccc7f92e2a004e1c0c09b5a4ff4f6",
    "snapshot_name": "STATUS_PRE_DEC024_AUTHORITY_RUNTIME_SYNC_V1.json"
  },
  "RUN_MANIFEST.json": {
    "bytes": 106682,
    "sha256": "0e94be47392058f8d537faf1bf79eecbe5d166fb04935f5c1099bc5e33e5d83f",
    "snapshot_name": "RUN_MANIFEST_PRE_DEC024_AUTHORITY_RUNTIME_SYNC_V1.json"
  },
  "EVENT_LOG.jsonl": {
    "bytes": 133091,
    "sha256": "ea51f1ef16f6fdcab0ec6ba6b6c43e35f21a1b9fbc54127e42d69470e1fdeb9b",
    "snapshot_name": "EVENT_LOG_PRE_DEC024_AUTHORITY_RUNTIME_SYNC_V1.jsonl"
  }
}'''
)
PREDECESSOR_TAIL = json.loads(
    r'''{
  "event_id": "A1-EVT-057",
  "decision_id": "V3-DEC-023",
  "bytes": 5043,
  "sha256": "52865f10d778fe25e27b8cd53f2afbb253f100c510d734f5da6aa23ac18420e8"
}'''
)
UNKNOWN_BINDING_FIELDS = (
    "status",
    "implementation_commit",
    "implementation_script_sha256",
    "implementation_test_sha256",
)


class RuntimeSyncError(RuntimeError):
    """The DEC024 authority/runtime contract is not satisfied."""


class BindingError(RuntimeSyncError):
    """The authority or exact3-I/config-only-B binding is incomplete."""


class AuthorityError(RuntimeSyncError):
    """The production repository authority chain is not exact."""


class PredecessorError(RuntimeSyncError):
    """The frozen EVT057 candidate is not the current runtime predecessor."""


class PublicationError(RuntimeSyncError):
    """Preparation or append-only publication failed."""


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    ).encode("utf-8")


def compact_json_line(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeSyncError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite token {token}")
            ),
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeSyncError(f"invalid JSON: {label}") from exc
    if not isinstance(value, dict):
        raise RuntimeSyncError(f"JSON root is not an object: {label}")
    return value


def load_events(payload: bytes, *, label: str) -> list[dict[str, Any]]:
    if payload and not payload.endswith(b"\n"):
        raise RuntimeSyncError(f"JSONL is not newline terminated: {label}")
    values: list[dict[str, Any]] = []
    for line in payload.splitlines():
        values.append(load_json(line, label=label))
    return values


def _typed_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _typed_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _typed_equal(item, value) for item, value in zip(actual, expected)
        )
    return actual == expected


def _expect(actual: Any, expected: Any, *, label: str) -> None:
    if not _typed_equal(actual, expected):
        raise RuntimeSyncError(f"{label} drift")


def _expect_keys(value: Any, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise RuntimeSyncError(f"{label} key closure drift")
    return value


def _expect_hex(value: Any, pattern: re.Pattern[str], *, label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise RuntimeSyncError(f"{label} is not lowercase hexadecimal")
    return value


def _implementation_binding_state(binding: Mapping[str, Any]) -> str:
    values = [binding.get(field) for field in UNKNOWN_BINDING_FIELDS]
    if values == [UNKNOWN] * len(UNKNOWN_BINDING_FIELDS):
        return "UNKNOWN"
    if binding.get("status") == BOUND:
        if not (
            isinstance(binding.get("implementation_commit"), str)
            and HEX40.fullmatch(str(binding["implementation_commit"]))
        ):
            raise BindingError("BOUND implementation commit is invalid")
        for field in (
            "implementation_script_sha256",
            "implementation_test_sha256",
        ):
            if not (
                isinstance(binding.get(field), str)
                and HEX64.fullmatch(str(binding[field]))
            ):
                raise BindingError(f"BOUND {field} is invalid")
        return "BOUND"
    raise BindingError("implementation binding is partially known")


def _authority_binding_state(authority: Mapping[str, Any]) -> str:
    files = authority.get("authority_files")
    if not isinstance(files, list) or len(files) != 12:
        raise BindingError("authority exact12 file identity closure differs")
    identity_values = [authority.get("authority_binding_status"), authority.get("authority_commit")]
    for item in files:
        if not isinstance(item, dict):
            raise BindingError("authority file identity is not an object")
        identity_values.extend([item.get("bytes"), item.get("sha256")])
    if identity_values == [UNKNOWN] * len(identity_values):
        return "UNKNOWN"
    if authority.get("authority_binding_status") != "FROZEN_BOUND_EXACT12":
        raise BindingError("authority exact12 binding is partially known")
    _expect_hex(authority.get("authority_commit"), HEX40, label="authority commit")
    for item in files:
        if not isinstance(item.get("bytes"), int) or item["bytes"] <= 0:
            raise BindingError("authority exact12 byte count is invalid")
        _expect_hex(item.get("sha256"), HEX64, label="authority file SHA-256")
    return "BOUND"


def _validate_runtime_shape(config: Mapping[str, Any]) -> None:
    runtime = _expect_keys(
        config.get("runtime"),
        {
            "run_root",
            "allowed_prepared_root",
            "predecessor_binding_status",
            "fresh_production_validation_required",
            "predecessor_event_id",
            "predecessor_event_count",
            "successor_event_id",
            "successor_event_count",
            "predecessor_manifest_output_count",
            "successor_manifest_output_count",
            "predecessor_manifest_registered_artifact_count",
            "successor_manifest_registered_artifact_count",
            "predecessor_mutables",
            "predecessor_tail",
            "sync_name",
            "output_delta_count",
            "immutable_publish_order",
            "mutable_publish_order",
        },
        label="runtime",
    )
    fixed = {
        "predecessor_binding_status": "FROZEN_BOUND_EVT057",
        "fresh_production_validation_required": True,
        "predecessor_event_id": "A1-EVT-057",
        "predecessor_event_count": 57,
        "successor_event_id": "A1-EVT-058",
        "successor_event_count": 58,
        "predecessor_manifest_output_count": 248,
        "successor_manifest_output_count": 252,
        "predecessor_manifest_registered_artifact_count": 8,
        "successor_manifest_registered_artifact_count": 8,
        "sync_name": "A1_DEC024_AUTHORITY_RUNTIME_SYNC_V1.json",
        "output_delta_count": 4,
        "mutable_publish_order": list(MUTABLE_NAMES),
    }
    for key, expected in fixed.items():
        _expect(runtime.get(key), expected, label=f"runtime {key}")
    _expect(
        runtime.get("predecessor_mutables"),
        PREDECESSOR_IDENTITIES,
        label="predecessor mutable identities",
    )
    _expect(runtime.get("predecessor_tail"), PREDECESSOR_TAIL, label="tail identity")
    snapshots = runtime["predecessor_mutables"]
    expected_immutables = [
        snapshots[name]["snapshot_name"] for name in MUTABLE_NAMES
    ] + [runtime["sync_name"]]
    _expect(
        runtime.get("immutable_publish_order"),
        expected_immutables,
        label="immutable publish order",
    )


def validate_static_config(config: dict[str, Any]) -> None:
    """Validate the closed config without Git, runtime, or prepared I/O."""

    _expect_keys(
        config,
        {
            "schema_version",
            "protocol_id",
            "contract_id",
            "phase_id",
            "dataset_ids",
            "decision_id",
            "event_id",
            "event_name",
            "sync_type",
            "implementation_binding",
            "repository_authority",
            "dec024_authority",
            "runtime_authority",
            "registered_artifacts",
            "runtime",
            "frozen_outer_truth",
            "access_boundary",
            "publication_policy",
        },
        label="config root",
    )
    identity = {
        "schema_version": "route_a_v3_dec024_authority_runtime_sync.v1",
        "protocol_id": "ROUTE_A_V3_DEC024_AUTHORITY_RUNTIME_SYNC_V1",
        "contract_id": "mrna_xeditflow_route_a_v3",
        "phase_id": "A1",
        "dataset_ids": ["GSE261709", "GSE269595", "E-MTAB-10902"],
        "decision_id": "V3-DEC-024",
        "event_id": "A1-EVT-058",
        "event_name": (
            "DEC024_THREE_REPLACEMENT_AGGREGATE_ONLY_PREFLIGHT_AUTHORITIES_"
            "REGISTERED_RUNTIME_GATES_UNCHANGED"
        ),
        "sync_type": (
            "APPEND_ONLY_AUTHORITY_ONLY_REGISTRATION_NO_SCIENTIFIC_STATE_CHANGE"
        ),
    }
    _expect(
        {key: config[key] for key in identity}, identity, label="config identity"
    )

    binding = _expect_keys(
        config["implementation_binding"],
        {
            "binding_scheme",
            "status",
            "implementation_commit",
            "implementation_script_path",
            "implementation_script_sha256",
            "implementation_test_path",
            "implementation_test_sha256",
            "unknown_to_bound_scalar_paths",
            "implementation_commit_exact_changed_paths",
            "binding_commit_exact_changed_paths",
            "activation_rule",
        },
        label="implementation binding",
    )
    _expect(
        binding["binding_scheme"],
        (
            "APPEND_ONLY_A_EXACT12_THEN_NONAUTHORITATIVE_A6_G0_EXACT4_"
            "THEN_I_EXACT3_THEN_B_CONFIG_ONLY_V1"
        ),
        label="binding scheme",
    )
    _expect(binding["implementation_script_path"], SCRIPT_REPO_PATH, label="script path")
    _expect(binding["implementation_test_path"], TEST_REPO_PATH, label="test path")
    _expect(
        binding["unknown_to_bound_scalar_paths"],
        [f"implementation_binding.{field}" for field in UNKNOWN_BINDING_FIELDS],
        label="four binding scalar paths",
    )
    _expect(
        binding["implementation_commit_exact_changed_paths"],
        IMPLEMENTATION_PATHS,
        label="implementation exact3",
    )
    _expect(
        binding["binding_commit_exact_changed_paths"],
        [CONFIG_REPO_PATH],
        label="binding config-only",
    )
    implementation_state = _implementation_binding_state(binding)

    authority = _expect_keys(
        config["repository_authority"],
        {
            "production_repo_root",
            "branch",
            "authority_expected_parent",
            "authority_binding_status",
            "authority_commit",
            "authority_exact_changed_paths",
            "authority_files",
            "predecessor_nonauthoritative_a6_g0",
            "implementation_exact_changed_paths",
            "binding_exact_changed_paths",
        },
        label="repository authority",
    )
    _expect(
        authority["production_repo_root"],
        str(PRODUCTION_REPO_ROOT),
        label="production repo root",
    )
    _expect(authority["branch"], BRANCH, label="branch")
    _expect(authority["authority_expected_parent"], AUTHORITY_PARENT, label="A parent")
    _expect(authority["authority_exact_changed_paths"], AUTHORITY_PATHS, label="A exact12")
    _expect(
        [item.get("path") for item in authority["authority_files"]],
        AUTHORITY_PATHS,
        label="A file order",
    )
    _expect(
        authority["predecessor_nonauthoritative_a6_g0"],
        FROZEN_G,
        label="frozen non-authoritative A6 G0 exact4",
    )
    _expect(
        authority["implementation_exact_changed_paths"],
        IMPLEMENTATION_PATHS,
        label="I exact3",
    )
    _expect(
        authority["binding_exact_changed_paths"],
        [CONFIG_REPO_PATH],
        label="B config-only",
    )
    authority_state = _authority_binding_state(authority)
    if implementation_state == "BOUND" and authority_state != "BOUND":
        raise BindingError("BOUND implementation requires frozen exact12 authority")

    _expect(config["dec024_authority"], DEC024_AUTHORITY, label="DEC024 authority")
    _expect(config["runtime_authority"], RUNTIME_AUTHORITY, label="runtime authority")
    _expect(config["registered_artifacts"], [], label="new registered artifacts")
    _validate_runtime_shape(config)
    _expect(config["frozen_outer_truth"], FROZEN_OUTER_TRUTH, label="outer truth")
    _expect(config["access_boundary"], ACCESS_BOUNDARY, label="access boundary")
    _expect(config["publication_policy"], PUBLICATION_POLICY, label="publication policy")


def _load_config_payload(
    config_path: Path, *, require_bound: bool
) -> tuple[dict[str, Any], bytes]:
    try:
        payload = config_path.read_bytes()
    except OSError as exc:
        raise BindingError("cannot read runtime-sync config") from exc
    config = load_json(payload, label="runtime-sync config")
    validate_static_config(config)
    if require_bound:
        if _implementation_binding_state(config["implementation_binding"]) != "BOUND":
            raise BindingError("runtime-sync implementation is not BOUND")
        if _authority_binding_state(config["repository_authority"]) != "BOUND":
            raise BindingError("DEC024 exact12 authority is not BOUND")
    return config, payload


def load_config(
    config_path: Path = PRODUCTION_CONFIG_PATH, *, require_bound: bool = False
) -> dict[str, Any]:
    return _load_config_payload(config_path, require_bound=require_bound)[0]


def load_bound_config(config_path: Path = PRODUCTION_CONFIG_PATH) -> dict[str, Any]:
    return load_config(config_path, require_bound=True)


def expected_unknown_i_config(bound_config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(bound_config))
    for field in UNKNOWN_BINDING_FIELDS:
        result["implementation_binding"][field] = UNKNOWN
    return result


def _run_git(repo_root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise AuthorityError("git is unavailable") from exc
    if completed.returncode != 0:
        raise AuthorityError("git authority command failed")
    return completed.stdout


def _git_blob(repo_root: Path, commit: str, relative_path: str) -> bytes:
    return _run_git(repo_root, "show", f"{commit}:{relative_path}")


def _changed_paths(repo_root: Path, commit: str) -> list[str]:
    output = _run_git(
        repo_root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        commit,
    ).decode("utf-8")
    return sorted(line for line in output.splitlines() if line)


def _read_repo_file(repo_root: Path, relative_path: str) -> bytes:
    try:
        return (repo_root / relative_path).read_bytes()
    except OSError as exc:
        raise AuthorityError("cannot read a bound repository file") from exc


def audit_production_repository_authority(
    config: dict[str, Any], config_payload: bytes
) -> dict[str, Any]:
    """Prove exact12 A -> non-authoritative exact4 G -> exact3 I -> B."""

    validate_static_config(config)
    if _implementation_binding_state(config["implementation_binding"]) != "BOUND":
        raise BindingError("runtime-sync implementation is not BOUND")
    if _authority_binding_state(config["repository_authority"]) != "BOUND":
        raise BindingError("DEC024 exact12 authority is not BOUND")

    authority = config["repository_authority"]
    binding = config["implementation_binding"]
    repo_root = Path(authority["production_repo_root"])
    expected_script = (repo_root / SCRIPT_REPO_PATH).resolve()
    if Path(__file__).resolve() != expected_script:
        raise AuthorityError("executing producer is not the bound repository script")
    branch = authority["branch"]
    authority_commit = authority["authority_commit"]
    frozen_g_commit = authority["predecessor_nonauthoritative_a6_g0"]["commit"]
    implementation_commit = binding["implementation_commit"]

    head = _run_git(repo_root, "rev-parse", "HEAD").decode().strip()
    upstream = _run_git(repo_root, "rev-parse", "@{upstream}").decode().strip()
    origin = _run_git(
        repo_root,
        "rev-parse",
        "--verify",
        f"refs/remotes/origin/{branch}",
    ).decode().strip()
    _expect(head, upstream, label="HEAD/upstream")
    _expect(head, origin, label="HEAD/origin")
    _expect(
        _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD").decode().strip(),
        branch,
        label="branch",
    )
    _expect(
        _run_git(repo_root, "rev-parse", "--abbrev-ref", "@{upstream}")
        .decode()
        .strip(),
        f"origin/{branch}",
        label="upstream branch",
    )
    if _run_git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise AuthorityError("production worktree or index is dirty")

    _expect(
        _run_git(repo_root, "rev-parse", f"{head}^").decode().strip(),
        implementation_commit,
        label="B parent/I",
    )
    _expect(
        _run_git(repo_root, "rev-parse", f"{implementation_commit}^")
        .decode().strip(),
        frozen_g_commit,
        label="I parent/G",
    )
    _expect(
        _run_git(repo_root, "rev-parse", f"{frozen_g_commit}^").decode().strip(),
        authority_commit,
        label="G parent/A",
    )
    _expect(
        _run_git(repo_root, "rev-parse", f"{authority_commit}^").decode().strip(),
        AUTHORITY_PARENT,
        label="A parent",
    )
    _expect(_changed_paths(repo_root, authority_commit), sorted(AUTHORITY_PATHS), label="A exact12")
    _expect(
        _changed_paths(repo_root, frozen_g_commit),
        sorted(FROZEN_G_PATHS),
        label="G exact4",
    )
    _expect(
        _changed_paths(repo_root, implementation_commit),
        sorted(IMPLEMENTATION_PATHS),
        label="I exact3",
    )
    _expect(_changed_paths(repo_root, head), [CONFIG_REPO_PATH], label="B config-only")

    for item in authority["authority_files"]:
        relative = item["path"]
        a_blob = _git_blob(repo_root, authority_commit, relative)
        if len(a_blob) != item["bytes"] or sha256(a_blob) != item["sha256"]:
            raise AuthorityError("authority exact12 blob identity differs")
        if _git_blob(repo_root, frozen_g_commit, relative) != a_blob:
            raise AuthorityError("authority exact12 blob did not persist through G")
        if _git_blob(repo_root, head, relative) != a_blob:
            raise AuthorityError("authority exact12 blob did not persist through I/B")
        if _read_repo_file(repo_root, relative) != a_blob:
            raise AuthorityError("working authority file differs from bound A")

    for item in FROZEN_G_FILES:
        relative = item["path"]
        g_blob = _git_blob(repo_root, frozen_g_commit, relative)
        if len(g_blob) != item["bytes"] or sha256(g_blob) != item["sha256"]:
            raise AuthorityError("frozen non-authoritative G exact4 blob differs")
        if _git_blob(repo_root, implementation_commit, relative) != g_blob:
            raise AuthorityError("frozen G blob did not persist through I")
        if _git_blob(repo_root, head, relative) != g_blob:
            raise AuthorityError("frozen G blob did not persist through B")
        if _read_repo_file(repo_root, relative) != g_blob:
            raise AuthorityError("working frozen G file differs")

    i_config = load_json(
        _git_blob(repo_root, implementation_commit, CONFIG_REPO_PATH),
        label="I config",
    )
    _expect(i_config, expected_unknown_i_config(config), label="I unknown config")
    script_blob = _git_blob(repo_root, implementation_commit, SCRIPT_REPO_PATH)
    test_blob = _git_blob(repo_root, implementation_commit, TEST_REPO_PATH)
    _expect(
        sha256(script_blob),
        binding["implementation_script_sha256"],
        label="I script SHA-256",
    )
    _expect(
        sha256(test_blob),
        binding["implementation_test_sha256"],
        label="I test SHA-256",
    )
    _expect(
        _git_blob(repo_root, head, CONFIG_REPO_PATH),
        config_payload,
        label="B config blob",
    )
    _expect(
        _git_blob(repo_root, head, SCRIPT_REPO_PATH),
        script_blob,
        label="B script unchanged",
    )
    _expect(
        _git_blob(repo_root, head, TEST_REPO_PATH),
        test_blob,
        label="B test unchanged",
    )
    _expect(_read_repo_file(repo_root, CONFIG_REPO_PATH), config_payload, label="working config")
    _expect(_read_repo_file(repo_root, SCRIPT_REPO_PATH), script_blob, label="working script")
    _expect(_read_repo_file(repo_root, TEST_REPO_PATH), test_blob, label="working test")

    return {
        "status": "PASS_EXACT12_A_NONAUTHORITATIVE_EXACT4_G_EXACT3_I_CONFIG_ONLY_B",
        "authority_commit": authority_commit,
        "nonauthoritative_a6_g0_commit": frozen_g_commit,
        "implementation_commit": implementation_commit,
        "binding_commit": head,
        "head_commit": head,
        "upstream_head_commit": upstream,
        "origin_branch_head_commit": origin,
        "authority_blob_count": 12,
        "nonauthoritative_a6_g0_blob_count": 4,
        "worktree_and_index_clean": True,
    }


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _prepared_path(prepared_directory: Path | str, config: Mapping[str, Any]) -> Path:
    prepared = _absolute(prepared_directory)
    allowed = _absolute(config["runtime"]["allowed_prepared_root"])
    try:
        common = Path(os.path.commonpath((str(prepared), str(allowed))))
    except ValueError as exc:
        raise PublicationError("prepared directory is outside allowed root") from exc
    if common != allowed or prepared == allowed:
        raise PublicationError("prepared directory must be a strict child of allowed root")
    return prepared


@contextmanager
def _locked_run(run_root: Path) -> Iterator[None]:
    try:
        descriptor = os.open(run_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError as exc:
        raise PublicationError("cannot open runtime root") from exc
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _read_runtime(run_root: Path) -> dict[str, bytes]:
    try:
        return {name: (run_root / name).read_bytes() for name in MUTABLE_NAMES}
    except OSError as exc:
        raise PublicationError("cannot read runtime mutables") from exc


def _snapshot_names(config: Mapping[str, Any]) -> dict[str, str]:
    return {
        name: config["runtime"]["predecessor_mutables"][name]["snapshot_name"]
        for name in MUTABLE_NAMES
    }


def _parse_runtime(
    payloads: Mapping[str, bytes],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    return (
        load_json(payloads["STATUS.json"], label="STATUS.json"),
        load_json(payloads["RUN_MANIFEST.json"], label="RUN_MANIFEST.json"),
        load_events(payloads["EVENT_LOG.jsonl"], label="EVENT_LOG.jsonl"),
    )


def _check_payload_identity(
    payload: bytes, spec: Mapping[str, Any], *, label: str
) -> None:
    if len(payload) != spec["bytes"] or sha256(payload) != spec["sha256"]:
        raise PredecessorError(f"{label} predecessor identity drift")


def _validate_outer_document(document: Mapping[str, Any], *, label: str) -> None:
    counts = FROZEN_OUTER_TRUTH["current_qualified_counts"]
    expected = {
        "qualified_ordinary_studies": counts["ordinary"],
        "qualified_a1_studies": counts["a1"],
        "qualified_a2_dense_studies": counts["true_a2"],
        "canonical_intervention_record_count": counts["canonical_records"],
        "canonical_record_count": counts["canonical_records"],
        "run_status": FROZEN_OUTER_TRUTH["run_status"],
        "evidence_status": FROZEN_OUTER_TRUTH["evidence_status"],
        "gate_status": FROZEN_OUTER_TRUTH["gate_status"],
        "qualified": False,
        "training_started": False,
        "training_allowed": False,
        "training_authorized": False,
        "gpu_work_started": False,
        "gpu_work_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }
    for key, value in expected.items():
        _expect(document.get(key), value, label=f"{label}.{key}")


def validate_predecessor(
    config: dict[str, Any], payloads: Mapping[str, bytes]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Freshly prove exact live EVT057/57/248/8 before any prepared write."""

    if set(payloads) != set(MUTABLE_NAMES):
        raise PredecessorError("runtime mutable member closure differs")
    runtime = config["runtime"]
    for name in MUTABLE_NAMES:
        _check_payload_identity(
            payloads[name], runtime["predecessor_mutables"][name], label=name
        )
    status, manifest, events = _parse_runtime(payloads)
    if len(events) != 57:
        raise PredecessorError("predecessor event count is not 57")
    expected_ids = [f"A1-EVT-{index:03d}" for index in range(1, 58)]
    if [event.get("event_id") for event in events] != expected_ids:
        raise PredecessorError("predecessor event identifiers are not exact 1..57")
    _expect(events[-1].get("decision_id"), "V3-DEC-023", label="tail decision")
    tail_payload = payloads["EVENT_LOG.jsonl"].splitlines(keepends=True)[-1]
    _check_payload_identity(tail_payload, runtime["predecessor_tail"], label="EVT057 tail")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 248:
        raise PredecessorError("predecessor manifest output count is not 248")
    if len({item.get("absolute_path") for item in outputs if isinstance(item, dict)}) != 248:
        raise PredecessorError("predecessor manifest output paths are not unique")
    _expect(
        manifest.get("registered_artifact_count"),
        8,
        label="predecessor registered artifact count",
    )
    _validate_outer_document(status, label="predecessor STATUS")
    _validate_outer_document(manifest, label="predecessor RUN_MANIFEST")
    _expect_hex(
        manifest.get("active_authority_commit"),
        HEX40,
        label="historical active_authority_commit",
    )
    return status, manifest, events


def _validate_recorded_at(recorded_at: str, predecessor_at: Any) -> None:
    if not isinstance(recorded_at, str) or not isinstance(predecessor_at, str):
        raise PredecessorError("timestamps must be explicit ISO-8601 strings")
    try:
        current = datetime.fromisoformat(recorded_at)
        predecessor = datetime.fromisoformat(predecessor_at)
    except ValueError as exc:
        raise PredecessorError("timestamp is not ISO-8601") from exc
    if current.tzinfo is None or predecessor.tzinfo is None or current <= predecessor:
        raise PredecessorError("EVT058 timestamp must follow EVT057 with an offset")


def _current_contract_authority(config: Mapping[str, Any]) -> dict[str, Any]:
    authority = config["repository_authority"]
    return {
        "decision_id": "V3-DEC-024",
        "authority_commit": authority["authority_commit"],
        "authority_expected_parent": AUTHORITY_PARENT,
        "scope": RUNTIME_AUTHORITY["current_contract_authority_scope"],
        "authority_file_count": 12,
    }


def _output_record(artifact_type: str, path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "absolute_path": str(path),
        "artifact_type": artifact_type,
        "bytes": len(payload),
        "sha256": sha256(payload),
        "status": "COMPLETE",
    }


def _build_sync_record(
    config: dict[str, Any],
    *,
    recorded_at: str,
    snapshots: Mapping[str, bytes],
    historical_active_authority_commit: str,
    authority_audit: Mapping[str, Any],
) -> bytes:
    return json_bytes(
        {
            "schema_version": "1.0.0",
            "record_type": "ROUTE_A_V3_A1_DEC024_AUTHORITY_RUNTIME_SYNC",
            "sync_type": config["sync_type"],
            "contract_id": config["contract_id"],
            "phase_id": "A1",
            "decision_id": "V3-DEC-024",
            "event_id": "A1-EVT-058",
            "recorded_at": recorded_at,
            "predecessor_event_id": "A1-EVT-057",
            "registered_artifacts": [],
            "registered_artifact_count": 0,
            "new_registered_artifact_count": 0,
            "predecessor_snapshot_count": 3,
            "predecessor_snapshot_names": list(snapshots),
            "snapshot_sha256": {
                name: sha256(payload) for name, payload in snapshots.items()
            },
            "output_delta_count": 4,
            "manifest_output_count_before": 248,
            "manifest_output_count_after": 252,
            "manifest_registered_artifact_count_before": 8,
            "manifest_registered_artifact_count_after": 8,
            "dec024_authority": copy.deepcopy(config["dec024_authority"]),
            "active_amendment_decision_ids": copy.deepcopy(ACTIVE_DECISION_IDS),
            "current_contract_authority": _current_contract_authority(config),
            "historical_outer_runtime_authority": {
                "active_authority_commit": historical_active_authority_commit,
                "active_authority_commit_rewritten": False,
                "meaning": "HISTORICAL_RUNTIME_AUTHORITY_IDENTITY",
            },
            "runtime_sync_publisher_authority": copy.deepcopy(dict(authority_audit)),
            "frozen_outer_truth": copy.deepcopy(config["frozen_outer_truth"]),
            "access_boundary": copy.deepcopy(config["access_boundary"]),
            "preflight_executed": False,
            "scientific_state_changed": False,
            "evidence_gate_statuses_changed": False,
            "overall_qualification_gate_changed": False,
            "qualification_changed": False,
        }
    )


def _event_document(
    config: Mapping[str, Any], *, recorded_at: str, sync_digest: str
) -> dict[str, Any]:
    return {
        "event_id": "A1-EVT-058",
        "at": recorded_at,
        "phase_id": "A1",
        "event": config["event_name"],
        "sync_type": config["sync_type"],
        "decision_id": "V3-DEC-024",
        "predecessor_event_id": "A1-EVT-057",
        "registered_artifacts": [],
        "registered_artifact_count": 0,
        "new_registered_artifact_count": 0,
        "predecessor_snapshot_count": 3,
        "predecessor_snapshot_names": list(_snapshot_names(config).values()),
        "sync_name": config["runtime"]["sync_name"],
        "sync_record_sha256": sync_digest,
        "output_delta_count": 4,
        "manifest_output_count_before": 248,
        "manifest_output_count_after": 252,
        "manifest_registered_artifact_count_before": 8,
        "manifest_registered_artifact_count_after": 8,
        "active_amendment_decision_ids": copy.deepcopy(ACTIVE_DECISION_IDS),
        "current_contract_authority": _current_contract_authority(config),
        "dec024_authority": copy.deepcopy(config["dec024_authority"]),
        "frozen_outer_truth": copy.deepcopy(config["frozen_outer_truth"]),
        "access_boundary": copy.deepcopy(config["access_boundary"]),
        "preflight_executed": False,
        "scientific_state_changed": False,
        "evidence_gate_statuses_changed": False,
        "overall_qualification_gate_changed": False,
        "qualification_changed": False,
        "qualified": False,
        "qualification_allowed": False,
        "canonical_materialization_allowed": False,
        "split_execution_allowed": False,
        "formal_qualification_power_gate_execution_allowed": False,
        "training_started": False,
        "training_allowed": False,
        "training_authorized": False,
        "gpu_work_started": False,
        "gpu_work_allowed": False,
        "model_selection_allowed": False,
        "a7_allowed": False,
        "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "detail": (
            "Registered only V3-DEC-024 repository authority for three ordinary-"
            "public aggregate-only replacement preflights: processed-asset "
            "GSE261709 A1 qualification preflight, mutually exclusive GSE269595 "
            "A1-or-true-A2 role-adjudication preflight, and N-zip/E-MTAB-10902 "
            "dense-family true-A2 candidate preflight. No preflight was executed; "
            "no public asset, member, row, identifier, barcode, transcript, "
            "sequence, endpoint, effect, standard-error, replicate, split, private, "
            "or sealed payload was read. N-zip's approximate 16 source groups were "
            "not replaced by 5679 rows and do not satisfy the prefrozen effective-N "
            "requirement of 156. Existing 1/1/0 qualified-study counts, 6547 "
            "canonical records, incomplete A1 state, and every qualification, "
            "training, GPU, model-selection, A7, next-phase, and scientific-claim "
            "lock remain unchanged."
        ),
    }


def _successor_updates(
    config: Mapping[str, Any], recorded_at: str, sync_digest: str
) -> dict[str, Any]:
    return {
        "active_amendment_decision_ids": copy.deepcopy(ACTIVE_DECISION_IDS),
        "current_contract_authority": _current_contract_authority(config),
        "dec024_authority_runtime_sync_status": "SYNCED_EVT_058",
        "dec024_authority_runtime_sync_recorded_at": recorded_at,
        "dec024_authority_runtime_sync_record_sha256": sync_digest,
        "dec024_authority_runtime_sync_scientific_state_changed": False,
        "dec024_authority_runtime_sync_gate_changed": False,
        "dec024_authority_runtime_sync_qualification_changed": False,
        "gse261709_dec024_aggregate_row_level_a1_preflight_status": "AUTHORIZED_NOT_RUN",
        "gse269595_dec024_role_adjudication_preflight_status": "AUTHORIZED_NOT_RUN",
        "emtab10902_dec024_dense_family_preflight_status": "AUTHORIZED_NOT_RUN",
        "gse261709_contribution": copy.deepcopy(
            FROZEN_OUTER_TRUTH["gse261709_contribution"]
        ),
        "gse269595_contribution": copy.deepcopy(
            FROZEN_OUTER_TRUTH["gse269595_contribution"]
        ),
        "emtab10902_contribution": copy.deepcopy(
            FROZEN_OUTER_TRUTH["emtab10902_contribution"]
        ),
    }


def _immutable_output_delta(
    config: Mapping[str, Any],
    predecessor_payloads: Mapping[str, bytes],
    sync_payload: bytes,
) -> list[dict[str, Any]]:
    run_root = Path(config["runtime"]["run_root"])
    snapshots = _snapshot_names(config)
    records = [
        _output_record(
            f"A1_{name.replace('.', '_').upper()}_PRE_DEC024_AUTHORITY_RUNTIME_SYNC_SNAPSHOT",
            run_root / snapshots[name],
            predecessor_payloads[name],
        )
        for name in MUTABLE_NAMES
    ]
    records.append(
        _output_record(
            "A1_DEC024_AUTHORITY_RUNTIME_SYNC_V1",
            run_root / config["runtime"]["sync_name"],
            sync_payload,
        )
    )
    return records


def build_successors(
    config: dict[str, Any],
    predecessor_payloads: Mapping[str, bytes],
    recorded_at: str,
    authority_audit: Mapping[str, Any] | None = None,
) -> dict[str, bytes]:
    status, manifest, events = validate_predecessor(config, predecessor_payloads)
    _validate_recorded_at(recorded_at, events[-1].get("at"))
    snapshots = _snapshot_names(config)
    snapshot_payloads = {
        snapshots[name]: predecessor_payloads[name] for name in MUTABLE_NAMES
    }
    audit = authority_audit or {
        "status": "SYNTHETIC_FIXTURE_NOT_PRODUCTION",
        "authority_commit": config["repository_authority"]["authority_commit"],
        "implementation_commit": config["implementation_binding"][
            "implementation_commit"
        ],
        "binding_commit": "SYNTHETIC_FIXTURE_NOT_PRODUCTION",
        "authority_blob_count": 12,
        "nonauthoritative_a6_g0_commit": FROZEN_G_COMMIT,
        "nonauthoritative_a6_g0_blob_count": 4,
        "worktree_and_index_clean": False,
    }
    historical_active = manifest["active_authority_commit"]
    sync_payload = _build_sync_record(
        config,
        recorded_at=recorded_at,
        snapshots=snapshot_payloads,
        historical_active_authority_commit=historical_active,
        authority_audit=audit,
    )
    sync_digest = sha256(sync_payload)
    updates = _successor_updates(config, recorded_at, sync_digest)
    successor_status = copy.deepcopy(status)
    successor_status["updated_at"] = recorded_at
    successor_status.update(updates)
    successor_manifest = copy.deepcopy(manifest)
    successor_manifest.update(updates)
    successor_manifest["outputs"] = list(manifest["outputs"]) + _immutable_output_delta(
        config, predecessor_payloads, sync_payload
    )
    event = _event_document(config, recorded_at=recorded_at, sync_digest=sync_digest)
    successors = {
        **snapshot_payloads,
        config["runtime"]["sync_name"]: sync_payload,
        "STATUS.json": json_bytes(successor_status),
        "RUN_MANIFEST.json": json_bytes(successor_manifest),
        "EVENT_LOG.jsonl": predecessor_payloads["EVENT_LOG.jsonl"]
        + compact_json_line(event),
    }
    validate_successors(config, predecessor_payloads, successors)
    return successors


def validate_successors(
    config: dict[str, Any],
    predecessor_payloads: Mapping[str, bytes],
    successors: Mapping[str, bytes],
) -> None:
    old_status, old_manifest, old_events = validate_predecessor(
        config, predecessor_payloads
    )
    snapshots = _snapshot_names(config)
    expected_names = set(MUTABLE_NAMES) | set(snapshots.values()) | {
        config["runtime"]["sync_name"]
    }
    if set(successors) != expected_names or len(successors) != 7:
        raise RuntimeSyncError("prepared member closure is not exact seven")
    for mutable, snapshot in snapshots.items():
        if successors[snapshot] != predecessor_payloads[mutable]:
            raise RuntimeSyncError("predecessor snapshot bytes differ")

    status, manifest, events = _parse_runtime(
        {name: successors[name] for name in MUTABLE_NAMES}
    )
    if (
        len(events) != 58
        or events[:-1] != old_events
        or not successors["EVENT_LOG.jsonl"].startswith(
            predecessor_payloads["EVENT_LOG.jsonl"]
        )
    ):
        raise RuntimeSyncError("EVENT_LOG is not one exact EVT058 append")
    event = events[-1]
    _expect(event.get("event_id"), "A1-EVT-058", label="successor event")
    _expect(event.get("decision_id"), "V3-DEC-024", label="successor decision")
    _expect(event.get("registered_artifacts"), [], label="event registered artifacts")
    _expect(event.get("registered_artifact_count"), 0, label="event artifact count")
    _expect(event.get("new_registered_artifact_count"), 0, label="event new artifacts")
    _expect(event.get("preflight_executed"), False, label="event preflight")
    _expect(event.get("scientific_state_changed"), False, label="event science")
    _expect(event.get("qualification_changed"), False, label="event qualification")

    sync_payload = successors[config["runtime"]["sync_name"]]
    sync_digest = sha256(sync_payload)
    sync = load_json(sync_payload, label="DEC024 authority runtime sync")
    _expect(sync.get("event_id"), "A1-EVT-058", label="sync event")
    _expect(sync.get("decision_id"), "V3-DEC-024", label="sync decision")
    _expect(sync.get("registered_artifacts"), [], label="sync registered artifacts")
    _expect(sync.get("registered_artifact_count"), 0, label="sync artifact count")
    _expect(sync.get("new_registered_artifact_count"), 0, label="sync new artifacts")
    _expect(sync.get("output_delta_count"), 4, label="sync output delta")
    _expect(sync.get("preflight_executed"), False, label="sync preflight")
    _expect(sync.get("scientific_state_changed"), False, label="sync science")
    _expect(sync.get("qualification_changed"), False, label="sync qualification")
    _expect(
        sync.get("current_contract_authority"),
        _current_contract_authority(config),
        label="sync authority",
    )
    _expect(sync.get("frozen_outer_truth"), FROZEN_OUTER_TRUTH, label="sync outer truth")
    _expect(sync.get("access_boundary"), ACCESS_BOUNDARY, label="sync access")

    updates = _successor_updates(config, event["at"], sync_digest)
    expected_status = copy.deepcopy(old_status)
    expected_status["updated_at"] = event["at"]
    expected_status.update(updates)
    _expect(status, expected_status, label="successor STATUS closure")
    expected_manifest = copy.deepcopy(old_manifest)
    expected_manifest.update(updates)
    output_delta = _immutable_output_delta(config, predecessor_payloads, sync_payload)
    expected_manifest["outputs"] = list(old_manifest["outputs"]) + output_delta
    _expect(manifest, expected_manifest, label="successor manifest closure")
    _expect(
        manifest.get("active_authority_commit"),
        old_manifest.get("active_authority_commit"),
        label="historical active authority preservation",
    )
    _expect(
        manifest.get("registered_artifact_count"),
        8,
        label="registered artifact count preservation",
    )
    outputs = manifest["outputs"]
    if len(outputs) != 252 or outputs[:248] != old_manifest["outputs"]:
        raise RuntimeSyncError("manifest ordered 248-to-252 append differs")
    if outputs[248:] != output_delta:
        raise RuntimeSyncError("manifest exact4 output delta differs")
    if len({item.get("absolute_path") for item in outputs}) != 252:
        raise RuntimeSyncError("successor output paths are not unique")
    _expect(
        [Path(item["absolute_path"]).name for item in outputs[-4:]],
        config["runtime"]["immutable_publish_order"],
        label="manifest exact4 output names",
    )
    _expect(event.get("sync_record_sha256"), sync_digest, label="event sync digest")
    _validate_outer_document(status, label="successor STATUS")
    _validate_outer_document(manifest, label="successor RUN_MANIFEST")


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_temp_payload(temp_path: Path, payload: bytes) -> None:
    with temp_path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _existing_immutable_result(path: Path, payload: bytes) -> str:
    try:
        existing = path.read_bytes()
    except OSError as exc:
        raise PublicationError("cannot read existing immutable output") from exc
    if existing != payload:
        raise PublicationError("existing immutable output differs")
    return "EXISTING_EXACT"


def _write_immutable_once(path: Path, payload: bytes) -> str:
    if path.exists():
        return _existing_immutable_result(path, payload)
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary)
        _write_temp_payload(temporary_path, payload)
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            return _existing_immutable_result(path, payload)
    except FileExistsError:
        return _existing_immutable_result(path, payload)
    except PublicationError:
        raise
    except OSError as exc:
        raise PublicationError("cannot create immutable output") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return "CREATED"


def _prepared_members(
    config: Mapping[str, Any],
    predecessor_payloads: Mapping[str, bytes],
    successors: Mapping[str, bytes],
) -> dict[str, bytes]:
    snapshots = _snapshot_names(config)
    members = {
        snapshots[name]: predecessor_payloads[name] for name in MUTABLE_NAMES
    }
    members[config["runtime"]["sync_name"]] = successors[
        config["runtime"]["sync_name"]
    ]
    members.update({name: successors[name] for name in MUTABLE_NAMES})
    return members


def _write_prepared(prepared: Path, members: Mapping[str, bytes]) -> None:
    prepared.mkdir(parents=True, exist_ok=True)
    observed = {item.name for item in prepared.iterdir()}
    if observed - set(members):
        raise PublicationError("prepared directory contains unexpected members")
    for name, payload in members.items():
        target = prepared / name
        if target.exists():
            if target.read_bytes() != payload:
                raise PublicationError("prepared member differs")
            continue
        _write_atomic(target, payload)
    if {item.name for item in prepared.iterdir()} != set(members):
        raise PublicationError("prepared member closure is incomplete")


def _read_prepared(
    config: Mapping[str, Any], prepared: Path
) -> dict[str, bytes]:
    expected = set(_snapshot_names(config).values()) | {
        config["runtime"]["sync_name"],
        *MUTABLE_NAMES,
    }
    try:
        observed = {item.name for item in prepared.iterdir()}
    except OSError as exc:
        raise PublicationError("prepared directory is absent") from exc
    if observed != expected:
        raise PublicationError("prepared member set is incomplete or has extras")
    try:
        return {name: (prepared / name).read_bytes() for name in expected}
    except OSError as exc:
        raise PublicationError("cannot read prepared members") from exc


def _split_prepared(
    config: Mapping[str, Any], prepared: Mapping[str, bytes]
) -> tuple[dict[str, bytes], dict[str, bytes]]:
    snapshots = _snapshot_names(config)
    predecessor = {name: prepared[snapshots[name]] for name in MUTABLE_NAMES}
    successor = {name: prepared[name] for name in MUTABLE_NAMES}
    return predecessor, successor


def _context(
    config_path: Path,
    *,
    production: bool,
    config_override: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if production and config_override is not None:
        raise BindingError("config override is forbidden in production")
    if production and Path(config_path).resolve() != PRODUCTION_CONFIG_PATH.resolve():
        raise BindingError("production config is not the executing repository config")
    if config_override is None:
        config, payload = _load_config_payload(config_path, require_bound=True)
        audit = (
            audit_production_repository_authority(config, payload)
            if production
            else None
        )
        return config, audit
    config = copy.deepcopy(config_override)
    validate_static_config(config)
    if _implementation_binding_state(config["implementation_binding"]) != "BOUND":
        raise BindingError("runtime-sync implementation is not BOUND")
    if _authority_binding_state(config["repository_authority"]) != "BOUND":
        raise BindingError("DEC024 exact12 authority is not BOUND")
    return config, None


def prepare_runtime_sync(
    *,
    prepared_directory: Path | str,
    recorded_at: str,
    config_path: Path = PRODUCTION_CONFIG_PATH,
    production: bool = True,
    config_override: dict[str, Any] | None = None,
    run_root_override: Path | None = None,
) -> dict[str, Any]:
    config, authority_audit = _context(
        config_path, production=production, config_override=config_override
    )
    prepared = _prepared_path(prepared_directory, config)
    run_root = run_root_override or Path(config["runtime"]["run_root"])
    if production and run_root_override is not None:
        raise PublicationError("run-root override is forbidden in production")
    with _locked_run(run_root):
        predecessor = _read_runtime(run_root)
        validate_predecessor(config, predecessor)
        successors = build_successors(
            config,
            predecessor,
            recorded_at,
            authority_audit=authority_audit,
        )
    _write_prepared(prepared, _prepared_members(config, predecessor, successors))
    return {
        "status": "PREPARED_NOT_PUBLISHED",
        "event_id": "A1-EVT-058",
        "prepared_directory": str(prepared),
        "prepared_member_count": 7,
        "manifest_output_transition": "248_TO_252",
        "manifest_registered_artifact_transition": "8_TO_8",
        "new_runtime_output_count": 4,
        "new_registered_artifact_count": 0,
    }


def publish_prepared(
    *,
    prepared_directory: Path | str,
    config_path: Path = PRODUCTION_CONFIG_PATH,
    production: bool = True,
    config_override: dict[str, Any] | None = None,
    run_root_override: Path | None = None,
    fault_injector: FaultInjector | None = None,
) -> dict[str, Any]:
    config, _authority_audit = _context(
        config_path, production=production, config_override=config_override
    )
    prepared_path = _prepared_path(prepared_directory, config)
    run_root = run_root_override or Path(config["runtime"]["run_root"])
    if production and run_root_override is not None:
        raise PublicationError("run-root override is forbidden in production")
    prepared = _read_prepared(config, prepared_path)
    predecessor, successor = _split_prepared(config, prepared)
    validate_successors(config, predecessor, prepared)
    snapshots = _snapshot_names(config)
    immutable_payloads = {
        **{snapshots[name]: predecessor[name] for name in MUTABLE_NAMES},
        config["runtime"]["sync_name"]: prepared[config["runtime"]["sync_name"]],
    }
    with _locked_run(run_root):
        current = _read_runtime(run_root)
        states: list[str] = []
        for name in MUTABLE_NAMES:
            if current[name] == predecessor[name]:
                states.append("OLD")
            elif current[name] == successor[name]:
                states.append("NEW")
            else:
                raise PredecessorError("runtime mutable is neither old nor EVT058")
        allowed_states = (
            ["OLD", "OLD", "OLD"],
            ["NEW", "OLD", "OLD"],
            ["NEW", "NEW", "OLD"],
            ["NEW", "NEW", "NEW"],
        )
        if states not in allowed_states:
            raise PredecessorError("runtime mutable prefix is not recoverable")
        immutable_results: dict[str, str] = {}
        for name in config["runtime"]["immutable_publish_order"]:
            if fault_injector is not None:
                fault_injector(f"before_immutable:{name}")
            immutable_results[name] = _write_immutable_once(
                run_root / name, immutable_payloads[name]
            )
        if states == ["NEW", "NEW", "NEW"]:
            return {
                "status": "PUBLISHED_VERIFIED",
                "event_id": "A1-EVT-058",
                "reused": True,
                "immutable_results": immutable_results,
            }
        try:
            for index, name in enumerate(MUTABLE_NAMES):
                if states[index] == "NEW":
                    continue
                if fault_injector is not None:
                    fault_injector(f"before_replace:{name}")
                _write_atomic(run_root / name, successor[name])
                states[index] = "NEW"
        except Exception as exc:
            after = _read_runtime(run_root)
            if all(after[name] == successor[name] for name in MUTABLE_NAMES):
                return {
                    "status": "PUBLISHED_VERIFIED_AFTER_RECHECK",
                    "event_id": "A1-EVT-058",
                    "immutable_results": immutable_results,
                }
            raise PublicationError(
                "EVT058 was not committed; retry the same prepared directory"
            ) from exc
        final = _read_runtime(run_root)
        if final != successor:
            raise PublicationError("EVT058 publication finished non-exactly")
        return {
            "status": "PUBLISHED_VERIFIED",
            "event_id": "A1-EVT-058",
            "reused": False,
            "immutable_results": immutable_results,
        }


def validate_published(
    *,
    prepared_directory: Path | str,
    config_path: Path = PRODUCTION_CONFIG_PATH,
    production: bool = True,
    config_override: dict[str, Any] | None = None,
    run_root_override: Path | None = None,
) -> dict[str, Any]:
    config, _authority_audit = _context(
        config_path, production=production, config_override=config_override
    )
    prepared_path = _prepared_path(prepared_directory, config)
    run_root = run_root_override or Path(config["runtime"]["run_root"])
    if production and run_root_override is not None:
        raise PublicationError("run-root override is forbidden in production")
    prepared = _read_prepared(config, prepared_path)
    predecessor, successor = _split_prepared(config, prepared)
    validate_successors(config, predecessor, prepared)
    with _locked_run(run_root):
        if _read_runtime(run_root) != successor:
            raise PublicationError("runtime does not match prepared EVT058")
        for name in config["runtime"]["immutable_publish_order"]:
            if (run_root / name).read_bytes() != prepared[name]:
                raise PublicationError("immutable output does not match prepared")
    return {"status": "PUBLISHED_VERIFIED", "event_id": "A1-EVT-058"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--prepared-directory", type=Path, required=True)
    prepare.add_argument("--recorded-at", required=True)
    publish = subparsers.add_parser("publish")
    publish.add_argument("--prepared-directory", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--prepared-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_runtime_sync(
            prepared_directory=args.prepared_directory,
            recorded_at=args.recorded_at,
        )
    elif args.command == "publish":
        result = publish_prepared(prepared_directory=args.prepared_directory)
    else:
        result = validate_published(prepared_directory=args.prepared_directory)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
