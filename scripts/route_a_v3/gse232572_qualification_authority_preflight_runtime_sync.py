#!/usr/bin/env python3
"""Prepare, validate, and publish the A1-EVT-049 runtime sync.

EVT049 registers the existing public aggregate-only GSE232572 A1
qualification-authority preflight. A production prepare verifies the report's
exact bytes and SHA-256 identity without parsing its JSON body.
Private JSONL is never registered, opened, copied, or listed. The remaining
immutable outputs are three predecessor snapshots and one sync record.

Mutable runtime files commit in the order STATUS, RUN_MANIFEST, EVENT_LOG, with
EVENT_LOG as the commit point. Frozen I1 is the direct exact3 child of fully
bound ledger L and leaves the four implementation-binding scalars
UNKNOWN_NOT_ASSERTED. I2 changes only this publisher and its focused test while
leaving the I1 config byte-identical. Config-only B2 then binds I2. Production
proves the clean, pushed L -> frozen exact3 I1 -> exact2 I2 -> config-only B2
history before report, prepared-output, or runtime I/O.
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
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping


UNKNOWN = "UNKNOWN_NOT_ASSERTED"
CORE_PLACEHOLDER = "CORE_SHA256_TO_REFRESH"
MUTABLE_NAMES = ("STATUS.json", "RUN_MANIFEST.json", "EVENT_LOG.jsonl")
SCRIPT_REPO_PATH = "scripts/route_a_v3/gse232572_qualification_authority_preflight_runtime_sync.py"
TEST_REPO_PATH = "tests/route_a_v3/test_gse232572_qualification_authority_preflight_runtime_sync.py"
CONFIG_REPO_PATH = "configs/route_a_v3_gse232572_qualification_authority_preflight_runtime_sync_v1.json"
FROZEN_I1_COMMIT = "f6b7f90e1eab219a1d168964f6a9bf3efa8d19a1"
FROZEN_I1_BLOB_SHA256 = {
    CONFIG_REPO_PATH: "2cc9ba117ad7c614a34c8ef20a6dcc3f4a172f8776c110dce5fec9db146f8f10",
    SCRIPT_REPO_PATH: "189523096e8ea0c2471bb389e428416b700688dd0e495bb1e1d9e1363c59a490",
    TEST_REPO_PATH: "0d8af3a978680a408a36c53f55eb20d8dc4e3ed3bb78e59b6f5dfc43fd2018d6",
}
I2_EXACT_CHANGED_PATHS = [SCRIPT_REPO_PATH, TEST_REPO_PATH]
PRODUCTION_CONFIG_PATH = Path(__file__).resolve().parents[2] / CONFIG_REPO_PATH
PRODUCTION_REPO_ROOT = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810"
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
LEDGER_LINEAGE_IDS = [
    "gse232572_a1_qualification_authority_preflight_v1",
]
LEDGER_PATHS = [
    "docs/execution/route_a_v3_a1_interim.yaml",
    "docs/execution/route_a_v3_registry_manifest.json",
    "scripts/route_a_v3/validate_a0_bundle.py",
    "tests/route_a_v3/test_a0_integrity_guards.py",
]
REGISTERED_ARTIFACTS = [
    {
        "lineage_id": LEDGER_LINEAGE_IDS[0],
        "name": "GSE232572_A1_QUALIFICATION_AUTHORITY_PREFLIGHT.json",
        "artifact_type": "GSE232572_A1_QUALIFICATION_AUTHORITY_PREFLIGHT_AGGREGATE_ONLY",
        "absolute_path": (
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/data/A1/GSE232572/"
            "GSE232572_A1_QUALIFICATION_AUTHORITY_PREFLIGHT_20260813T010116P0800/"
            "GSE232572_A1_QUALIFICATION_AUTHORITY_PREFLIGHT.json"
        ),
        "bytes": 9586,
        "sha256": "00776c808cfa3e9ba2cfdb92b866c5f7c1bc92ea3818d17687cb9a8521b30d71",
    },
]
SUCCESSOR_SCIENTIFIC_STATE = {
    "input_status_counts": {
        "PASS": 7,
        "BLOCKED": 0,
        "UNKNOWN_NOT_ASSERTED": 1,
        "NOT_RUN": 0,
    },
    "unresolved_blockers": ["CHECKPOINT_SPECIFIC_EXPOSURE_NOT_PASS"],
    "ordinary_study_contribution": 0,
    "a1_study_contribution": 0,
    "true_a2_study_contribution": 0,
    "canonical_intervention_record_count": 0,
    "canonical_materialization_allowed": False,
    "qualified": False,
    "training_started": False,
    "training_allowed": False,
    "training_authorized": False,
    "model_selection_allowed": False,
    "next_phase_authorized": False,
    "scientific_claim_status": "NOT_ESTABLISHED",
}

FaultInjector = Callable[[str], None]


class RuntimeSyncError(RuntimeError):
    """Base class for an EVT049 validation/publication error."""


class BindingError(RuntimeSyncError):
    """The implementation or predecessor ledger is not fully bound."""


class AuthorityError(RuntimeSyncError):
    """The production Git history does not prove the bound publisher lineage."""


class PredecessorError(RuntimeSyncError):
    """The runtime is not the exact EVT048 predecessor or a supported prefix."""


class PublicationError(RuntimeSyncError):
    """Prepared or immutable/mutable publication failed."""


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def compact_json_line(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeSyncError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeSyncError(f"invalid JSON: {label}") from exc
    if not isinstance(value, dict):
        raise RuntimeSyncError(f"JSON root is not an object: {label}")
    return value


def load_events(payload: bytes, *, label: str) -> list[dict[str, Any]]:
    if payload and not payload.endswith(b"\n"):
        raise RuntimeSyncError(f"JSONL is not newline terminated: {label}")
    try:
        values = [
            json.loads(line, object_pairs_hook=_reject_duplicate_keys)
            for line in payload.splitlines()
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeSyncError(f"invalid JSONL: {label}") from exc
    if any(not isinstance(value, dict) for value in values):
        raise RuntimeSyncError(f"JSONL contains a non-object: {label}")
    return values


def _expect(actual: Any, expected: Any, *, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise RuntimeSyncError(f"{label} drift: expected {expected!r}, observed {actual!r}")


def _expect_keys(value: Any, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise RuntimeSyncError(f"{label} key closure drift")
    return value


def _expect_hex(value: Any, pattern: re.Pattern[str], *, label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise RuntimeSyncError(f"{label} is not lowercase hexadecimal")
    return value


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


def compiled_core_projection(config: dict[str, Any]) -> dict[str, Any]:
    """Return the config core that must remain identical between I and B."""

    return {
        key: copy.deepcopy(value)
        for key, value in config.items()
        if key != "implementation_binding"
    }


def compiled_core_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(
        compiled_core_projection(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload)


def _binding_values_are_unknown(binding: Mapping[str, Any]) -> bool:
    return all(
        binding.get(key) == UNKNOWN
        for key in (
            "status",
            "implementation_commit",
            "implementation_script_sha256",
            "implementation_test_sha256",
        )
    )


def _ledger_values(ledger: Mapping[str, Any]) -> list[Any]:
    blobs = ledger.get("frozen_blobs")
    digests = (
        [item.get("sha256") for item in blobs if isinstance(item, Mapping)]
        if isinstance(blobs, list)
        else []
    )
    lineage_ids = ledger.get("registered_lineage_ids")
    ordered_lineage_ids = list(lineage_ids) if isinstance(lineage_ids, list) else []
    return [
        ledger.get("status"),
        ledger.get("commit"),
        ledger.get("integration_id"),
        ledger.get("manifest_status"),
        *ordered_lineage_ids,
        *digests,
    ]


def _ledger_values_are_unknown(ledger: Mapping[str, Any]) -> bool:
    values = _ledger_values(ledger)
    return len(values) == 9 and all(value == UNKNOWN for value in values)


def _validate_ledger_binding(ledger: Mapping[str, Any]) -> None:
    """Allow only the grouped exact1 UNKNOWN candidate or one fully bound L."""

    values = _ledger_values(ledger)
    if any(value == UNKNOWN for value in values):
        if not _ledger_values_are_unknown(ledger):
            raise BindingError("predecessor ledger authority is partially known")
        return
    _expect(ledger.get("status"), "BOUND", label="predecessor ledger status")
    _expect_hex(ledger.get("commit"), HEX40, label="predecessor ledger commit")
    _expect(
        ledger.get("integration_id"),
        "GSE232572_A1_QUALIFICATION_AUTHORITY_PREFLIGHT_V1",
        label="predecessor ledger integration id",
    )
    _expect(
        ledger.get("manifest_status"),
        (
            "A1_GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_"
            "LEDGER_REGISTERED_PENDING_EVT049"
        ),
        label="predecessor ledger manifest status",
    )
    _expect(
        ledger.get("registered_lineage_ids"),
        LEDGER_LINEAGE_IDS,
        label="predecessor ledger lineage ids",
    )
    for index, value in enumerate(values[5:]):
        _expect_hex(value, HEX64, label=f"predecessor ledger blob {index} SHA")


def _validate_runtime_shape(config: dict[str, Any]) -> None:
    runtime = _expect_keys(
        config.get("runtime"),
        {
            "run_root",
            "allowed_prepared_root",
            "predecessor_event_id",
            "predecessor_event_count",
            "successor_event_id",
            "successor_event_count",
            "predecessor_manifest_output_count",
            "successor_manifest_output_count",
            "predecessor_mutables",
            "predecessor_tail",
            "sync_name",
            "output_delta_count",
            "immutable_publish_order",
            "mutable_publish_order",
        },
        label="runtime",
    )
    _expect(runtime["predecessor_event_id"], "A1-EVT-048", label="predecessor event id")
    _expect(runtime["successor_event_id"], "A1-EVT-049", label="successor event id")
    for key, expected in (
        ("predecessor_event_count", 48),
        ("successor_event_count", 49),
        ("predecessor_manifest_output_count", 203),
        ("successor_manifest_output_count", 208),
        ("output_delta_count", 5),
    ):
        if type(runtime.get(key)) is not int or runtime[key] != expected:
            raise RuntimeSyncError(f"runtime {key} drift")
    _expect(runtime["mutable_publish_order"], list(MUTABLE_NAMES), label="mutable publish order")
    snapshots = runtime.get("predecessor_mutables")
    if not isinstance(snapshots, dict) or set(snapshots) != set(MUTABLE_NAMES):
        raise RuntimeSyncError("predecessor mutable metadata closure drift")
    for mutable in MUTABLE_NAMES:
        spec = _expect_keys(
            snapshots[mutable],
            {"bytes", "sha256", "snapshot_name"},
            label=f"{mutable} predecessor metadata",
        )
        if isinstance(spec["bytes"], bool) or not isinstance(spec["bytes"], int):
            raise RuntimeSyncError(f"{mutable} predecessor bytes drift")
        _expect_hex(spec["sha256"], HEX64, label=f"{mutable} predecessor SHA")
        if not isinstance(spec["snapshot_name"], str) or not spec["snapshot_name"]:
            raise RuntimeSyncError(f"{mutable} predecessor snapshot name drift")
    tail = _expect_keys(
        runtime["predecessor_tail"],
        {"event_id", "decision_id", "bytes", "sha256"},
        label="predecessor tail metadata",
    )
    _expect(tail["event_id"], "A1-EVT-048", label="predecessor tail event")
    _expect(tail["decision_id"], "V3-DEC-019", label="predecessor tail decision")
    if isinstance(tail["bytes"], bool) or not isinstance(tail["bytes"], int):
        raise RuntimeSyncError("predecessor tail bytes drift")
    _expect_hex(tail["sha256"], HEX64, label="predecessor tail SHA")
    _expect(
        runtime["sync_name"],
        "A1_GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_RUNTIME_SYNC_V1.json",
        label="sync name",
    )
    immutable_expected = [
        snapshots[name]["snapshot_name"] for name in MUTABLE_NAMES
    ] + [runtime["sync_name"]]
    _expect(runtime["immutable_publish_order"], immutable_expected, label="immutable publish order")


def validate_static_config(config: dict[str, Any]) -> None:
    """Validate the candidate core without opening repository/report/runtime paths."""

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
            "implementation_binding",
            "repository_authority",
            "registered_artifacts",
            "runtime",
            "successor_scientific_state",
            "registered_evidence_truth",
            "outer_a1_state",
            "access_boundary",
            "publication_policy",
        },
        label="config root",
    )
    _expect(
        {
            key: config.get(key)
            for key in (
                "schema_version",
                "protocol_id",
                "contract_id",
                "phase_id",
                "dataset_ids",
                "decision_id",
                "event_id",
                "event_name",
            )
        },
        {
            "schema_version": (
                "route_a_v3_gse232572_qualification_authority_"
                "preflight_runtime_sync.v1"
            ),
            "protocol_id": (
                "ROUTE_A_V3_GSE232572_A1_QUALIFICATION_AUTHORITY_"
                "PREFLIGHT_RUNTIME_SYNC_V1"
            ),
            "contract_id": "mrna_xeditflow_route_a_v3",
            "phase_id": "A1",
            "dataset_ids": ["GSE232572"],
            "decision_id": "V3-DEC-019",
            "event_id": "A1-EVT-049",
            "event_name": (
                "GSE232572_A1_QUALIFICATION_AUTHORITY_PREFLIGHT_"
                "REGISTERED_QUALIFICATION_GATE_UNCHANGED"
            ),
        },
        label="config identity",
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
            "compiled_core_sha256",
            "unknown_to_bound_scalar_paths",
            "activation_rule",
        },
        label="implementation binding",
    )
    _expect(
        binding["binding_scheme"],
        "CONFIG_ONLY_POST_IMPLEMENTATION_BINDING_V1",
        label="binding scheme",
    )
    _expect(binding["implementation_script_path"], SCRIPT_REPO_PATH, label="script path")
    _expect(binding["implementation_test_path"], TEST_REPO_PATH, label="test path")
    _expect(
        binding["unknown_to_bound_scalar_paths"],
        [
            "implementation_binding.status",
            "implementation_binding.implementation_commit",
            "implementation_binding.implementation_script_sha256",
            "implementation_binding.implementation_test_sha256",
        ],
        label="four-scalar binding paths",
    )
    if not isinstance(binding["activation_rule"], str) or not binding["activation_rule"]:
        raise BindingError("binding activation rule is absent")
    values = [
        binding[key]
        for key in (
            "status",
            "implementation_commit",
            "implementation_script_sha256",
            "implementation_test_sha256",
        )
    ]
    if any(value == UNKNOWN for value in values):
        if not _binding_values_are_unknown(binding):
            raise BindingError("implementation binding is partially known")
    else:
        _expect(binding["status"], "BOUND", label="implementation status")
        _expect_hex(binding["implementation_commit"], HEX40, label="implementation commit")
        _expect_hex(binding["implementation_script_sha256"], HEX64, label="implementation script SHA")
        _expect_hex(binding["implementation_test_sha256"], HEX64, label="implementation test SHA")
    core = binding["compiled_core_sha256"]
    if core != CORE_PLACEHOLDER:
        _expect_hex(core, HEX64, label="compiled core SHA")
        _expect(core, compiled_core_sha256(config), label="compiled core")

    authority = _expect_keys(
        config["repository_authority"],
        {
            "production_repo_root",
            "branch",
            "implementation_exact_changed_paths",
            "binding_exact_changed_paths",
            "predecessor_ledger",
        },
        label="repository authority",
    )
    _expect(
        authority["production_repo_root"],
        str(PRODUCTION_REPO_ROOT),
        label="production repository root",
    )
    _expect(authority["branch"], "routea-v3-a1-20260810", label="repository branch")
    _expect(
        authority["implementation_exact_changed_paths"],
        [CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH],
        label="implementation exact3 paths",
    )
    _expect(authority["binding_exact_changed_paths"], [CONFIG_REPO_PATH], label="binding path")
    ledger = _expect_keys(
        authority["predecessor_ledger"],
        {
            "status",
            "commit",
            "integration_id",
            "manifest_status",
            "registered_lineage_ids",
            "exact_changed_paths",
            "frozen_blobs",
        },
        label="predecessor ledger",
    )
    _expect(ledger["exact_changed_paths"], LEDGER_PATHS, label="predecessor ledger exact4 paths")
    blobs = ledger["frozen_blobs"]
    if not isinstance(blobs, list) or len(blobs) != 4:
        raise BindingError("predecessor ledger frozen blob closure drift")
    for item, path in zip(blobs, LEDGER_PATHS):
        _expect_keys(item, {"path", "sha256"}, label=f"ledger blob {path}")
        _expect(item["path"], path, label="predecessor ledger blob path")
    _validate_ledger_binding(ledger)

    registered = config["registered_artifacts"]
    _expect(registered, REGISTERED_ARTIFACTS, label="ledger-ordered exact1 artifact")
    if not _ledger_values_are_unknown(ledger) and [
        item["lineage_id"] for item in registered
    ] != ledger["registered_lineage_ids"]:
        raise RuntimeSyncError("registered artifact order is not the ledger lineage order")
    if any(Path(item["absolute_path"]).name != item["name"] for item in registered):
        raise RuntimeSyncError("registered artifact name/path identity drift")
    if any(item["absolute_path"].endswith(".jsonl") for item in registered):
        raise RuntimeSyncError("private JSONL cannot be a registered artifact")

    _validate_runtime_shape(config)
    _expect(
        config["successor_scientific_state"],
        SUCCESSOR_SCIENTIFIC_STATE,
        label="successor scientific truth",
    )
    evidence = config["registered_evidence_truth"]
    if not isinstance(evidence, dict):
        raise RuntimeSyncError("registered evidence truth must be an object")
    materialization = _expect_keys(
        evidence.get("gse232572_development_v3_materialization"),
        {"failed_attempt", "current_materialization"},
        label="GSE232572 materialization truth",
    )
    _expect(
        materialization["failed_attempt"],
        {
            "lineage_id": "gse232572_development_v3_materialization_attempt_001_failure",
            "artifact_type": (
                "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_"
                "ATTEMPT_001_FAIL_CLOSED_EVIDENCE"
            ),
            "status": "STOP_BEFORE_DEVELOPMENT_V3_ROW_PRODUCTION",
            "scientific_disposition": "NOT_QUALIFIED",
            "failure_gate": "RECOVERY_AUTHORITY",
            "failure_code": "MATERIALIZER_INPUTS_DIVERGE_FROM_RECOVERY_CONFIG",
            "failed_attempt_preserved": True,
            "historical_attempt_rewritten": False,
            "superseded_for_current_execution_by_lineage_id": (
                "gse232572_development_v3_materialization_v1"
            ),
            "produced_record_count": 0,
            "canonical_record_count": 0,
            "ordinary_study_contribution": 0,
            "a1_study_contribution": 0,
            "true_a2_study_contribution": 0,
            "qualified": False,
            "training_allowed": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
        },
        label="preserved failed materialization attempt",
    )
    _expect(
        materialization["current_materialization"],
        {
            "lineage_id": "gse232572_development_v3_materialization_v1",
            "artifact_type": (
                "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_REPORT_AGGREGATE_ONLY"
            ),
            "status": "DEVELOPMENT_V3_MATERIALIZED_NOT_QUALIFIED",
            "scientific_disposition": (
                "SCHEMA_VALID_DEVELOPMENT_ONLY_NOT_CANONICALLY_QUALIFIED"
            ),
            "registry_role": "AUDIT_ONLY",
            "qualification_status": "AUDIT_PENDING",
            "published_universe_row_count": 11929,
            "schema_valid_development_record_count": 8068,
            "development_only_record_count": 8068,
            "accepted_pair_complete_raw_endpoint_count": 8068,
            "accepted_pair_incomplete_raw_endpoint_count": 0,
            "rejected_published_row_count": 3861,
            "rejected_no_unique_sequence_pair_count": 3404,
            "rejected_ambiguous_distinct_sequence_pairs_count": 457,
            "canonical_record_count": 0,
            "ordinary_study_contribution": 0,
            "a1_study_contribution": 0,
            "true_a2_study_contribution": 0,
            "qualified": False,
            "canonical_materialization_allowed": False,
            "public_redistribution_status": (
                "UNKNOWN_NOT_ASSERTED_SUBMITTER_IP_CAVEAT"
            ),
            "redistribution_allowed": False,
            "row_license_status": "UNKNOWN_BLOCKED",
            "training_allowed": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
        },
        label="current development-only materialization",
    )
    _expect(
        evidence.get("gse232572_a1_qualification_authority_preflight"),
        {
            "schema_version": (
                "route_a_v3_gse232572_a1_qualification_authority_preflight.v1"
            ),
            "record_type": REGISTERED_ARTIFACTS[0]["artifact_type"],
            "overall_decision": "BLOCKED_MISSING_EXTERNAL_AUTHORITY",
            "terminal_status": (
                "STOP_BEFORE_PRIVATE_ROW_ACCESS_AND_CANONICAL_MATERIALIZATION"
            ),
            "registered_aggregate_pass_count": 3,
            "open_qualification_blocker_count": 12,
            "schema_valid_development_record_count": 8068,
            "canonical_record_count": 0,
            "ordinary_study_contribution": 0,
            "a1_study_contribution": 0,
            "true_a2_study_contribution": 0,
            "qualified": False,
            "training_allowed": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
            "scientific_claim_status": "NOT_ESTABLISHED",
            "private_row_artifact_read_count": 0,
        },
        label="GSE232572 qualification-authority preflight truth",
    )
    _expect(
        config["outer_a1_state"],
        {
            "run_status": "IN_PROGRESS",
            "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
            "gate_status": "A1_BLOCKED_PENDING_PUBLIC_EVIDENCE",
            "qualified_ordinary_studies": 0,
            "qualified_a1_studies": 0,
            "qualified_a2_dense_studies": 0,
            "metadata_only_qualification_count": 0,
            "qualified": False,
            "training_started": False,
            "training_allowed": False,
            "training_authorized": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
        },
        label="outer A1 state",
    )
    _expect(
        config["access_boundary"],
        {
            "registered_artifact_count": 1,
            "registered_artifact_exact_byte_validation_count": 1,
            "registered_artifact_body_parse_count": 0,
            "registered_artifact_payload_field_read_count": 0,
            "registered_artifacts_copied": False,
            "private_jsonl_read_count": 0,
            "private_jsonl_registered_artifact_count": 0,
            "private_jsonl_copied": False,
            "private_jsonl_listed": False,
            "canonical_materialization_count": 0,
            "training_run_count": 0,
            "model_selection_run_count": 0,
            "restricted_or_sealed_path_accessed": False,
            "gse246381_contact": False,
        },
        label="access boundary",
    )
    _expect(
        config["publication_policy"],
        {
            "registered_artifacts_remain_in_place": True,
            "registered_artifact_order_is_ledger_lineage_order": True,
            "predecessor_snapshots_are_immutable_runtime_outputs": True,
            "sync_record_is_immutable_runtime_output": True,
            "mutables_commit_after_all_immutables": True,
            "event_is_last_commit": True,
            "supported_recovery": "EXACT_PUBLICATION_PREFIX_ONLY",
        },
        label="publication policy",
    )


def validate_bound_config(config: dict[str, Any]) -> None:
    """Require fully bound L and implementation bundles before external I/O."""

    validate_static_config(config)
    ledger = config["repository_authority"]["predecessor_ledger"]
    if _ledger_values_are_unknown(ledger):
        raise BindingError("predecessor ledger authority is not BOUND")
    binding = config["implementation_binding"]
    if _binding_values_are_unknown(binding) or any(
        binding.get(key) == UNKNOWN
        for key in (
            "status",
            "implementation_commit",
            "implementation_script_sha256",
            "implementation_test_sha256",
        )
    ):
        raise BindingError("runtime-sync implementation is not BOUND")


def _load_config_payload(
    config_path: Path, *, require_bound: bool
) -> tuple[dict[str, Any], bytes]:
    payload = Path(config_path).read_bytes()
    config = load_json(payload, label=str(config_path))
    if require_bound:
        validate_bound_config(config)
    else:
        validate_static_config(config)
    return config, payload


def load_config(
    config_path: Path = PRODUCTION_CONFIG_PATH, *, require_bound: bool = True
) -> dict[str, Any]:
    return _load_config_payload(config_path, require_bound=require_bound)[0]


def load_bound_config(config_path: Path = PRODUCTION_CONFIG_PATH) -> dict[str, Any]:
    return load_config(config_path, require_bound=True)


def expected_unknown_i2_config(bound_config: dict[str, Any]) -> dict[str, Any]:
    expected = copy.deepcopy(bound_config)
    for key in (
        "status",
        "implementation_commit",
        "implementation_script_sha256",
        "implementation_test_sha256",
    ):
        expected["implementation_binding"][key] = UNKNOWN
    return expected


def _run_git(repo_root: Path, *arguments: str) -> bytes:
    environment = os.environ.copy()
    environment.update({"GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C", "LANG": "C"})
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AuthorityError("read-only Git command failed to start") from exc
    if result.returncode != 0:
        raise AuthorityError(f"read-only Git command failed: {arguments!r}")
    return result.stdout


def _git_blob(repo_root: Path, commit: str, path: str) -> bytes:
    return _run_git(repo_root, "show", f"{commit}:{path}")


def _changed_paths(repo_root: Path, commit: str) -> list[str]:
    return sorted(
        _run_git(
            repo_root,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            commit,
        )
        .decode("utf-8")
        .splitlines()
    )


def _read_repo_file(repo_root: Path, path: str) -> bytes:
    try:
        return (repo_root / path).read_bytes()
    except OSError as exc:
        raise AuthorityError(f"cannot read production repository file: {path}") from exc


def audit_production_repository_authority(
    config: dict[str, Any], config_payload: bytes
) -> dict[str, Any]:
    """Prove clean L -> frozen exact3 I1 -> exact2 I2 -> config-only B2."""

    validate_bound_config(config)
    authority = config["repository_authority"]
    binding = config["implementation_binding"]
    ledger = authority["predecessor_ledger"]
    repo_root = Path(authority["production_repo_root"])
    branch = authority["branch"]
    implementation_i2_commit = binding["implementation_commit"]
    ledger_commit = ledger["commit"]

    head = _run_git(repo_root, "rev-parse", "HEAD").decode("utf-8").strip()
    current_branch = _run_git(
        repo_root, "rev-parse", "--abbrev-ref", "HEAD"
    ).decode("utf-8").strip()
    upstream_branch = _run_git(
        repo_root, "rev-parse", "--abbrev-ref", "@{upstream}"
    ).decode("utf-8").strip()
    upstream_head = _run_git(repo_root, "rev-parse", "@{upstream}").decode(
        "utf-8"
    ).strip()
    origin_head = _run_git(
        repo_root, "rev-parse", "--verify", f"refs/remotes/origin/{branch}"
    ).decode("utf-8").strip()
    _expect(current_branch, branch, label="production branch")
    _expect(upstream_branch, f"origin/{branch}", label="production upstream branch")
    _expect(head, upstream_head, label="production HEAD/upstream")
    _expect(head, origin_head, label="production HEAD/origin")
    if _run_git(
        repo_root, "status", "--porcelain=v1", "--untracked-files=all"
    ) != b"":
        raise AuthorityError("production worktree or index is dirty")

    binding_b2_parent = _run_git(repo_root, "rev-parse", f"{head}^").decode(
        "utf-8"
    ).strip()
    implementation_i2_parent = _run_git(
        repo_root, "rev-parse", f"{implementation_i2_commit}^"
    ).decode("utf-8").strip()
    frozen_i1_parent = _run_git(
        repo_root, "rev-parse", f"{FROZEN_I1_COMMIT}^"
    ).decode("utf-8").strip()
    _expect(binding_b2_parent, implementation_i2_commit, label="B2 parent/I2")
    _expect(implementation_i2_parent, FROZEN_I1_COMMIT, label="I2 parent/I1")
    _expect(frozen_i1_parent, ledger_commit, label="I1 parent/L")
    _expect(
        _changed_paths(repo_root, head),
        sorted(authority["binding_exact_changed_paths"]),
        label="B2 exact config-only paths",
    )
    _expect(
        _changed_paths(repo_root, implementation_i2_commit),
        sorted(I2_EXACT_CHANGED_PATHS),
        label="I2 exact script/test paths",
    )
    _expect(
        _changed_paths(repo_root, FROZEN_I1_COMMIT),
        sorted(authority["implementation_exact_changed_paths"]),
        label="I1 exact config/script/test paths",
    )
    _expect(
        _changed_paths(repo_root, ledger_commit),
        sorted(ledger["exact_changed_paths"]),
        label="L exact frozen ledger paths",
    )

    for item in ledger["frozen_blobs"]:
        path = item["path"]
        digest = item["sha256"]
        if sha256(_git_blob(repo_root, ledger_commit, path)) != digest:
            raise AuthorityError(f"L frozen ledger blob drift: {path}")
        if sha256(_git_blob(repo_root, head, path)) != digest:
            raise AuthorityError(f"current frozen ledger blob drift: {path}")
        if sha256(_read_repo_file(repo_root, path)) != digest:
            raise AuthorityError(f"worktree frozen ledger blob drift: {path}")

    frozen_i1_payloads = {
        path: _git_blob(repo_root, FROZEN_I1_COMMIT, path)
        for path in (CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH)
    }
    for path, digest in FROZEN_I1_BLOB_SHA256.items():
        if sha256(frozen_i1_payloads[path]) != digest:
            raise AuthorityError(f"frozen I1 blob drift: {path}")

    expected_unknown_i2_payload = json_bytes(expected_unknown_i2_config(config))
    if frozen_i1_payloads[CONFIG_REPO_PATH] != expected_unknown_i2_payload:
        raise AuthorityError("frozen I1 config is not the exact four-scalar UNKNOWN form")
    if (
        _git_blob(repo_root, implementation_i2_commit, CONFIG_REPO_PATH)
        != frozen_i1_payloads[CONFIG_REPO_PATH]
    ):
        raise AuthorityError("I2 config differs from the frozen I1 config")

    if _git_blob(repo_root, head, CONFIG_REPO_PATH) != config_payload:
        raise AuthorityError("B config Git blob differs from the supplied production config")
    if _read_repo_file(repo_root, CONFIG_REPO_PATH) != config_payload:
        raise AuthorityError("worktree config differs from the supplied production config")

    for path, digest in (
        (SCRIPT_REPO_PATH, binding["implementation_script_sha256"]),
        (TEST_REPO_PATH, binding["implementation_test_sha256"]),
    ):
        for commit, label in (
            (implementation_i2_commit, "I2"),
            (head, "B2"),
        ):
            if sha256(_git_blob(repo_root, commit, path)) != digest:
                raise AuthorityError(f"{label} implementation blob drift: {path}")
        if sha256(_read_repo_file(repo_root, path)) != digest:
            raise AuthorityError(f"worktree implementation blob drift: {path}")

    core = binding["compiled_core_sha256"]
    if core != CORE_PLACEHOLDER:
        _expect(core, compiled_core_sha256(config), label="production compiled core")

    return {
        "status": "PASS_EXACT_L_TO_FROZEN_I1_TO_EXACT2_I2_TO_CONFIG_ONLY_B2",
        "ledger_commit": ledger_commit,
        "frozen_i1_commit": FROZEN_I1_COMMIT,
        "implementation_i2_commit": implementation_i2_commit,
        "binding_b2_commit": head,
        "head_commit": head,
        "upstream_head_commit": upstream_head,
        "origin_branch_head_commit": origin_head,
        "worktree_and_index_clean": True,
    }


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _prepared_path(prepared_directory: Path | str, config: dict[str, Any]) -> Path:
    prepared = _absolute(prepared_directory)
    allowed = _absolute(config["runtime"]["allowed_prepared_root"])
    try:
        common = Path(os.path.commonpath((str(prepared), str(allowed))))
    except ValueError as exc:
        raise PublicationError("prepared directory is outside allowed root") from exc
    if common != allowed or prepared == allowed:
        raise PublicationError("prepared directory must be a strict descendant of allowed root")
    return prepared


@contextmanager
def _locked_run(run_root: Path) -> Iterator[None]:
    try:
        descriptor = os.open(run_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError as exc:
        raise PublicationError(f"cannot open runtime root: {run_root}") from exc
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


def _predecessor_snapshot_names(config: dict[str, Any]) -> dict[str, str]:
    return {
        name: config["runtime"]["predecessor_mutables"][name]["snapshot_name"]
        for name in MUTABLE_NAMES
    }


def validate_registered_artifacts(
    config: dict[str, Any], *, verify_exact_bytes: bool
) -> dict[str, Any]:
    """Validate exact1 metadata and, in production, exact bytes without parsing."""

    if verify_exact_bytes:
        validate_bound_config(config)
    else:
        validate_static_config(config)
    artifacts = config["registered_artifacts"]
    if verify_exact_bytes:
        for item in artifacts:
            path = Path(item["absolute_path"])
            try:
                payload = path.read_bytes()
            except OSError as exc:
                raise PublicationError(
                    f"cannot read registered artifact bytes: {path}"
                ) from exc
            if len(payload) != item["bytes"] or sha256(payload) != item["sha256"]:
                raise PublicationError(f"registered artifact identity drift: {path}")
    return {
        "status": (
            "EXACT1_BYTES_AND_SHA256_VALIDATED"
            if verify_exact_bytes
            else "EXACT1_METADATA_VALIDATED"
        ),
        "artifact_count": len(artifacts),
        "exact_byte_validation_count": len(artifacts) if verify_exact_bytes else 0,
        "body_parse_count": 0,
        "payload_field_read_count": 0,
        "registered_artifacts": copy.deepcopy(artifacts),
    }


def _validate_recorded_at(recorded_at: str, predecessor_at: Any) -> None:
    if not isinstance(recorded_at, str) or not isinstance(predecessor_at, str):
        raise PredecessorError("timestamps must be explicit ISO-8601 strings")
    try:
        current = datetime.fromisoformat(recorded_at)
        predecessor = datetime.fromisoformat(predecessor_at)
    except ValueError as exc:
        raise PredecessorError("timestamp is not ISO-8601") from exc
    if current.tzinfo is None or predecessor.tzinfo is None or current <= predecessor:
        raise PredecessorError("EVT049 timestamp must follow EVT048 with an explicit offset")


def _parse_runtime(payloads: Mapping[str, bytes]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    return (
        load_json(payloads["STATUS.json"], label="STATUS.json"),
        load_json(payloads["RUN_MANIFEST.json"], label="RUN_MANIFEST.json"),
        load_events(payloads["EVENT_LOG.jsonl"], label="EVENT_LOG.jsonl"),
    )


def _check_payload_identity(payload: bytes, spec: Mapping[str, Any], *, label: str) -> None:
    if len(payload) != spec["bytes"] or sha256(payload) != spec["sha256"]:
        raise PredecessorError(f"{label} predecessor identity drift")


def validate_predecessor(
    config: dict[str, Any], payloads: Mapping[str, bytes]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Validate EVT048 before creating any prepared member."""

    runtime = config["runtime"]
    for name in MUTABLE_NAMES:
        _check_payload_identity(
            payloads[name], runtime["predecessor_mutables"][name], label=name
        )
    status, manifest, events = _parse_runtime(payloads)
    if len(events) != runtime["predecessor_event_count"]:
        raise PredecessorError("predecessor event count is not 48")
    tail = events[-1] if events else {}
    _expect(tail.get("event_id"), "A1-EVT-048", label="predecessor tail event")
    _expect(tail.get("decision_id"), "V3-DEC-019", label="predecessor decision")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != runtime["predecessor_manifest_output_count"]:
        raise PredecessorError("predecessor manifest output count is not 203")
    expected_scientific = config["successor_scientific_state"]
    for document, label in ((status, "STATUS"), (manifest, "RUN_MANIFEST")):
        for key, expected in expected_scientific.items():
            _expect(document.get(key), expected, label=f"predecessor {label}.{key}")
        _expect(
            document.get("claim_status"),
            "NOT_ESTABLISHED",
            label=f"predecessor {label}.claim_status",
        )
        _expect(
            document.get("canonical_record_count"),
            0,
            label=f"predecessor {label}.canonical_record_count",
        )
    for key, expected in config["outer_a1_state"].items():
        _expect(status.get(key), expected, label=f"predecessor outer STATUS.{key}")
    tail_payload = payloads["EVENT_LOG.jsonl"].splitlines(keepends=True)[-1]
    _check_payload_identity(tail_payload, runtime["predecessor_tail"], label="event tail")
    return status, manifest, events


def _event_document(
    config: dict[str, Any],
    *,
    recorded_at: str,
    predecessor: Mapping[str, Any],
    sync_digest: str,
) -> dict[str, Any]:
    runtime = config["runtime"]
    scientific = config["successor_scientific_state"]
    return {
        "event_id": "A1-EVT-049",
        "at": recorded_at,
        "phase_id": "A1",
        "event": config["event_name"],
        "decision_id": "V3-DEC-019",
        "predecessor_event_id": "A1-EVT-048",
        "registered_artifact_count": 1,
        "registered_lineage_ids": copy.deepcopy(
            config["repository_authority"]["predecessor_ledger"][
                "registered_lineage_ids"
            ]
        ),
        "registered_artifacts_copied": False,
        "registered_artifact_exact_byte_validation_count": 1,
        "registered_artifact_body_parse_count": 0,
        "registered_artifact_payload_field_read_count": 0,
        "private_jsonl_read_count": 0,
        "private_jsonl_registered_artifact_count": 0,
        "private_jsonl_copied": False,
        "private_jsonl_listed": False,
        "predecessor_snapshot_count": 3,
        "predecessor_snapshot_names": list(_predecessor_snapshot_names(config).values()),
        "sync_name": runtime["sync_name"],
        "output_delta_count": 5,
        "manifest_output_count_before": 203,
        "manifest_output_count_after": 208,
        "new_runtime_output_count": 5,
        "scientific_state_changed": True,
        "evidence_surface_changed_since_evt048": True,
        "evidence_gate_statuses_changed_since_evt048": False,
        "overall_qualification_gate_changed": False,
        "successor_scientific_state": copy.deepcopy(scientific),
        "registered_evidence_truth": copy.deepcopy(
            config["registered_evidence_truth"]
        ),
        "outer_a1_state": copy.deepcopy(config["outer_a1_state"]),
        "ledger_authority_status": config["repository_authority"]["predecessor_ledger"]["status"],
        "sync_record_sha256": sync_digest,
        "predecessor_event_count": (
            len(predecessor.get("events", []))
            if isinstance(predecessor, Mapping)
            else 48
        ),
        "training_started": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "detail": (
            "Registered the aggregate-only GSE232572 A1 qualification-authority "
            "preflight by exact bytes in place, then added three immutable EVT048 "
            "snapshots and one sync record. The evidence surface and scientific state "
            "changed, but gate statuses and overall qualification did not. The preflight "
            "decision is BLOCKED_MISSING_EXTERNAL_AUTHORITY with 3 registered-aggregate "
            "passes, 12 open blockers, zero canonical records, and zero ordinary/A1/"
            "true-A2 study credit. No private JSONL was registered, read, copied, or "
            "listed. Qualification, training, model selection, next phase, and the "
            "scientific claim remain locked."
        ),
    }


def _build_sync_record(
    config: dict[str, Any], *, recorded_at: str, snapshot_payloads: Mapping[str, bytes]
) -> bytes:
    runtime = config["runtime"]
    return json_bytes(
        {
            "record_type": "ROUTE_A_V3_A1_GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_RUNTIME_SYNC",
            "event_id": "A1-EVT-049",
            "decision_id": "V3-DEC-019",
            "recorded_at": recorded_at,
            "predecessor_event_id": "A1-EVT-048",
            "registered_artifact_count": 1,
            "registered_artifacts": copy.deepcopy(config["registered_artifacts"]),
            "registered_lineage_ids": copy.deepcopy(
                config["repository_authority"]["predecessor_ledger"][
                    "registered_lineage_ids"
                ]
            ),
            "registered_artifacts_copied": False,
            "registered_artifact_exact_byte_validation_count": 1,
            "registered_artifact_body_parse_count": 0,
            "registered_artifact_payload_field_read_count": 0,
            "private_jsonl_read_count": 0,
            "private_jsonl_registered_artifact_count": 0,
            "private_jsonl_copied": False,
            "private_jsonl_listed": False,
            "predecessor_snapshot_count": 3,
            "predecessor_snapshot_names": list(_predecessor_snapshot_names(config).values()),
            "snapshot_sha256": {
                name: sha256(payload) for name, payload in snapshot_payloads.items()
            },
            "output_delta_count": 5,
            "successor_manifest_output_count": 208,
            "successor_scientific_state": copy.deepcopy(config["successor_scientific_state"]),
            "registered_evidence_truth": copy.deepcopy(
                config["registered_evidence_truth"]
            ),
            "outer_a1_state": copy.deepcopy(config["outer_a1_state"]),
            "ledger_authority_status": config["repository_authority"]["predecessor_ledger"]["status"],
            "implementation_binding_status": config["implementation_binding"]["status"],
            "scientific_state_changed": True,
            "evidence_surface_changed_since_evt048": True,
            "evidence_gate_statuses_changed_since_evt048": False,
            "overall_qualification_gate_changed": False,
        }
    )


def _immutable_outputs(
    config: dict[str, Any],
    predecessor_outputs: list[dict[str, Any]],
    predecessor_payloads: Mapping[str, bytes],
    sync_payload: bytes,
) -> list[dict[str, Any]]:
    run_root = Path(config["runtime"]["run_root"])
    # EVT048 has 203 outputs. EVT049 appends one public aggregate report,
    # then three snapshots and one sync record, for 208.
    outputs = copy.deepcopy(predecessor_outputs)
    outputs.extend(copy.deepcopy(config["registered_artifacts"]))
    snapshots = _predecessor_snapshot_names(config)
    for mutable in MUTABLE_NAMES:
        payload = predecessor_payloads[mutable]
        outputs.append(
            {
                "absolute_path": str(run_root / snapshots[mutable]),
                "artifact_type": "PREDECESSOR_RUNTIME_SNAPSHOT",
                "bytes": len(payload),
                "sha256": sha256(payload),
                "status": "COMPLETE",
            }
        )
    outputs.append(
        {
            "absolute_path": str(run_root / config["runtime"]["sync_name"]),
            "artifact_type": "RUNTIME_SYNC_RECORD",
            "bytes": len(sync_payload),
            "sha256": sha256(sync_payload),
            "status": "COMPLETE",
        }
    )
    return outputs


def build_successors(
    config: dict[str, Any], predecessor_payloads: Mapping[str, bytes], recorded_at: str
) -> dict[str, bytes]:
    status, manifest, events = validate_predecessor(config, predecessor_payloads)
    _validate_recorded_at(recorded_at, events[-1].get("at"))
    snapshots = _predecessor_snapshot_names(config)
    snapshot_payloads = {snapshots[name]: predecessor_payloads[name] for name in MUTABLE_NAMES}
    sync_payload = _build_sync_record(
        config, recorded_at=recorded_at, snapshot_payloads=snapshot_payloads
    )
    successor_status = copy.deepcopy(status)
    successor_manifest = copy.deepcopy(manifest)
    successor_status["updated_at"] = recorded_at
    successor_manifest["outputs"] = _immutable_outputs(
        config, manifest["outputs"], predecessor_payloads, sync_payload
    )
    # Keep an explicit registration count for consumers that do not inspect the
    # output list itself.
    successor_manifest["registered_artifact_count"] = 1
    event_without_digest = _event_document(
        config,
        recorded_at=recorded_at,
        predecessor={"events": events},
        sync_digest=sha256(sync_payload),
    )
    successor_event_payload = predecessor_payloads["EVENT_LOG.jsonl"] + compact_json_line(
        event_without_digest
    )
    artifacts = {
        **{snapshots[name]: predecessor_payloads[name] for name in MUTABLE_NAMES},
        config["runtime"]["sync_name"]: sync_payload,
        "STATUS.json": json_bytes(successor_status),
        "RUN_MANIFEST.json": json_bytes(successor_manifest),
        "EVENT_LOG.jsonl": successor_event_payload,
    }
    validate_successors(config, predecessor_payloads, artifacts)
    return artifacts


def validate_successors(
    config: dict[str, Any],
    predecessor_payloads: Mapping[str, bytes],
    successors: Mapping[str, bytes],
) -> None:
    """Check exact3 mutable prefix, exact1+4 output closure, and EVT049 last."""

    old_status, old_manifest, old_events = validate_predecessor(config, predecessor_payloads)
    status, manifest, events = _parse_runtime(
        {name: successors[name] for name in MUTABLE_NAMES}
    )
    runtime = config["runtime"]
    if len(events) != runtime["successor_event_count"] or events[:-1] != old_events:
        raise RuntimeSyncError("EVT049 is not one append-only event")
    event = events[-1]
    _expect(event.get("event_id"), "A1-EVT-049", label="successor event")
    _expect(event.get("decision_id"), "V3-DEC-019", label="successor event decision")
    _expect(event.get("scientific_state_changed"), True, label="event scientific state change")
    _expect(
        event.get("evidence_surface_changed_since_evt048"),
        True,
        label="event evidence surface change",
    )
    _expect(
        event.get("evidence_gate_statuses_changed_since_evt048"),
        False,
        label="event evidence gate change",
    )
    _expect(
        event.get("overall_qualification_gate_changed"),
        False,
        label="event overall qualification gate change",
    )
    if not isinstance(event.get("at"), str):
        raise RuntimeSyncError("EVT049 timestamp is absent")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != runtime["successor_manifest_output_count"]:
        raise RuntimeSyncError("successor manifest output count is not 208")
    expected_registered = config["registered_artifacts"]
    if outputs[: runtime["predecessor_manifest_output_count"]] != old_manifest["outputs"]:
        raise RuntimeSyncError("predecessor manifest output prefix drift")
    registered_start = runtime["predecessor_manifest_output_count"]
    if outputs[registered_start : registered_start + 1] != expected_registered:
        raise RuntimeSyncError("registered exact1 output metadata drift")
    expected_tail_names = list(_predecessor_snapshot_names(config).values()) + [runtime["sync_name"]]
    actual_tail_names = [Path(item.get("absolute_path", "")).name for item in outputs[-4:]]
    if actual_tail_names != expected_tail_names:
        raise RuntimeSyncError("immutable output order drift")
    if len({item.get("absolute_path") for item in outputs}) != 208:
        raise RuntimeSyncError("successor output paths are not unique")
    expected_status = copy.deepcopy(old_status)
    expected_status["updated_at"] = event["at"]
    _expect(status, expected_status, label="non-timestamp STATUS preservation")
    expected_manifest = copy.deepcopy(old_manifest)
    expected_manifest["outputs"] = outputs
    expected_manifest["registered_artifact_count"] = 1
    _expect(manifest, expected_manifest, label="non-output manifest preservation")
    if successors.get(runtime["sync_name"]) is None:
        raise RuntimeSyncError("sync record is missing from prepared successors")
    sync = load_json(successors[runtime["sync_name"]], label=runtime["sync_name"])
    _expect(sync.get("event_id"), "A1-EVT-049", label="sync event id")
    _expect(sync.get("registered_artifact_count"), 1, label="sync exact1 count")
    _expect(sync.get("registered_artifacts"), expected_registered, label="sync exact1 metadata")
    _expect(sync.get("output_delta_count"), 5, label="sync output delta")
    _expect(sync.get("scientific_state_changed"), True, label="sync scientific state change")
    _expect(
        sync.get("evidence_surface_changed_since_evt048"),
        True,
        label="sync evidence surface change",
    )
    _expect(
        sync.get("evidence_gate_statuses_changed_since_evt048"),
        False,
        label="sync evidence gate change",
    )
    _expect(
        sync.get("overall_qualification_gate_changed"),
        False,
        label="sync overall qualification gate change",
    )


def _published_state_is_exact(config: dict[str, Any], payloads: Mapping[str, bytes]) -> bool:
    try:
        status, manifest, events = _parse_runtime(payloads)
    except RuntimeSyncError:
        return False
    runtime = config["runtime"]
    if len(events) != runtime["successor_event_count"]:
        return False
    outputs = manifest.get("outputs")
    return (
        events[-1].get("event_id") == "A1-EVT-049"
        and events[-1].get("decision_id") == "V3-DEC-019"
        and isinstance(outputs, list)
        and len(outputs) == runtime["successor_manifest_output_count"]
        and status.get("qualified") is False
        and status.get("training_started") is False
        and status.get("model_selection_allowed") is False
    )


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


def _write_immutable_once(path: Path, payload: bytes) -> str:
    if path.exists():
        try:
            observed = path.read_bytes()
        except OSError as exc:
            raise PublicationError(f"cannot read existing immutable output: {path}") from exc
        if observed != payload:
            raise PublicationError(f"immutable output differs: {path.name}")
        return "EXISTING_EXACT"
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_bytes() != payload:
            raise PublicationError(f"immutable output differs: {path.name}")
        return "EXISTING_EXACT"
    except OSError as exc:
        raise PublicationError(f"cannot create immutable output: {path.name}") from exc
    return "CREATED"


def _prepared_members(
    config: dict[str, Any], predecessor_payloads: Mapping[str, bytes], successors: Mapping[str, bytes]
) -> dict[str, bytes]:
    snapshots = _predecessor_snapshot_names(config)
    members = {snapshots[name]: predecessor_payloads[name] for name in MUTABLE_NAMES}
    members[config["runtime"]["sync_name"]] = successors[config["runtime"]["sync_name"]]
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
                raise PublicationError(f"prepared member differs: {name}")
            continue
        _write_atomic(target, payload)
    if {item.name for item in prepared.iterdir()} != set(members):
        raise PublicationError("prepared member closure is incomplete")


def _read_prepared(config: dict[str, Any], prepared: Path) -> dict[str, bytes]:
    expected = set(_predecessor_snapshot_names(config).values()) | {
        config["runtime"]["sync_name"],
        *MUTABLE_NAMES,
    }
    try:
        observed = {item.name for item in prepared.iterdir()}
    except OSError as exc:
        raise PublicationError("prepared directory is absent") from exc
    if observed != expected:
        raise PublicationError("prepared member set is incomplete or contains extras")
    try:
        return {name: (prepared / name).read_bytes() for name in expected}
    except OSError as exc:
        raise PublicationError("cannot read prepared members") from exc


def _split_prepared(
    config: dict[str, Any], prepared: Mapping[str, bytes]
) -> tuple[dict[str, bytes], dict[str, bytes]]:
    snapshots = _predecessor_snapshot_names(config)
    predecessor = {name: prepared[snapshots[name]] for name in MUTABLE_NAMES}
    successor = {name: prepared[name] for name in MUTABLE_NAMES}
    return predecessor, successor


def _context(
    config_path: Path,
    *,
    production: bool,
    config_override: dict[str, Any] | None,
) -> dict[str, Any]:
    if production and config_override is not None:
        raise BindingError("config override is forbidden in production")
    if config_override is None:
        config, payload = _load_config_payload(config_path, require_bound=True)
        if production:
            audit_production_repository_authority(config, payload)
        return config
    config = copy.deepcopy(config_override)
    validate_bound_config(config)
    return config


def prepare_runtime_sync(
    *,
    prepared_directory: Path | str,
    recorded_at: str,
    config_path: Path = PRODUCTION_CONFIG_PATH,
    production: bool = True,
    config_override: dict[str, Any] | None = None,
    run_root_override: Path | None = None,
    fault_injector: FaultInjector | None = None,
) -> dict[str, Any]:
    del fault_injector  # Preparation has no supported injected write point.
    config = _context(config_path, production=production, config_override=config_override)
    prepared = _prepared_path(prepared_directory, config)
    run_root = run_root_override or Path(config["runtime"]["run_root"])
    if production and run_root_override is not None:
        raise PublicationError("run-root override is forbidden in production")
    # Production verifies exact bytes/SHA only after repository authority. Tests
    # use metadata-only mode and never need the live /mnt artifacts.
    validate_registered_artifacts(config, verify_exact_bytes=production)
    with _locked_run(run_root):
        predecessor = _read_runtime(run_root)
        if _published_state_is_exact(config, predecessor):
            return {"status": "ALREADY_PUBLISHED_VERIFIED", "event_id": "A1-EVT-049"}
        successors = build_successors(config, predecessor, recorded_at)
    _write_prepared(prepared, _prepared_members(config, predecessor, successors))
    return {
        "status": "PREPARED_NOT_PUBLISHED",
        "event_id": "A1-EVT-049",
        "prepared_directory": str(prepared),
        "prepared_member_count": 7,
        "manifest_output_transition": "203_TO_208",
        "new_runtime_output_count": 5,
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
    config = _context(config_path, production=production, config_override=config_override)
    prepared_path = _prepared_path(prepared_directory, config)
    run_root = run_root_override or Path(config["runtime"]["run_root"])
    if production and run_root_override is not None:
        raise PublicationError("run-root override is forbidden in production")
    validate_registered_artifacts(config, verify_exact_bytes=production)
    prepared = _read_prepared(config, prepared_path)
    predecessor, successor = _split_prepared(config, prepared)
    validate_successors(config, predecessor, {**prepared, **successor})
    snapshots = _predecessor_snapshot_names(config)
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
                raise PredecessorError(
                    f"runtime mutable is neither predecessor nor successor: {name}"
                )
        allowed = (
            ["OLD", "OLD", "OLD"],
            ["NEW", "OLD", "OLD"],
            ["NEW", "NEW", "OLD"],
            ["NEW", "NEW", "NEW"],
        )
        if states not in allowed:
            raise PredecessorError(f"runtime mutable order is not recoverable: {states!r}")
        immutable_results: dict[str, str] = {}
        for name in config["runtime"]["immutable_publish_order"]:
            if fault_injector is not None:
                fault_injector(f"before_immutable:{name}")
            immutable_results[name] = _write_immutable_once(run_root / name, immutable_payloads[name])
        if states == ["NEW", "NEW", "NEW"]:
            return {
                "status": "PUBLISHED_VERIFIED",
                "event_id": "A1-EVT-049",
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
                    "event_id": "A1-EVT-049",
                    "immutable_results": immutable_results,
                }
            raise PublicationError(
                "EVT049 was not committed; retry with the same prepared directory"
            ) from exc
        final = _read_runtime(run_root)
        if final != successor:
            raise PublicationError("EVT049 publication finished with non-exact mutables")
        return {
            "status": "PUBLISHED_VERIFIED",
            "event_id": "A1-EVT-049",
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
    config = _context(config_path, production=production, config_override=config_override)
    prepared_path = _prepared_path(prepared_directory, config)
    run_root = run_root_override or Path(config["runtime"]["run_root"])
    if production and run_root_override is not None:
        raise PublicationError("run-root override is forbidden in production")
    validate_registered_artifacts(config, verify_exact_bytes=production)
    prepared = _read_prepared(config, prepared_path)
    predecessor, successor = _split_prepared(config, prepared)
    validate_successors(config, predecessor, {**prepared, **successor})
    with _locked_run(run_root):
        current = _read_runtime(run_root)
        if current != successor:
            raise PublicationError("runtime does not exactly match prepared EVT049 successor")
        for name in config["runtime"]["immutable_publish_order"]:
            expected = prepared[name]
            if (run_root / name).read_bytes() != expected:
                raise PublicationError(f"immutable output does not match prepared {name}")
    return {"status": "PUBLISHED_VERIFIED", "event_id": "A1-EVT-049"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
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
            prepared_directory=args.prepared_directory, recorded_at=args.recorded_at
        )
    elif args.command == "publish":
        result = publish_prepared(prepared_directory=args.prepared_directory)
    else:
        result = validate_published(prepared_directory=args.prepared_directory)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
