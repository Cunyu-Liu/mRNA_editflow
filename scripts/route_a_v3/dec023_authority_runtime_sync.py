#!/usr/bin/env python3
"""Prepare, publish, and validate the authority-only DEC023 A1-EVT-056 sync.

The transaction registers the two DEC023 aggregate-only preflight authorities.
It does not execute either preflight and does not add a registered artifact.  A
successful publication appends three exact EVT055 snapshots and one authority
sync record to the manifest, then commits STATUS, RUN_MANIFEST, and EVENT_LOG
in that order.  EVT056 is therefore the commit point.

Production is fail-closed.  The exact10 authority group, fresh EVT055 runtime
identity group, and exact3 implementation group must each be wholly BOUND.  The
repository chain A -> frozen I1 -> dynamic I2 -> config-only B2 is proven before
any prepared-directory or runtime I/O.  This exact3 I2 candidate freezes the
final DEC023 authority, live EVT055 predecessor, and immutable I1 lineage while
deliberately retaining only the four implementation binding scalars as UNKNOWN
for a later config-only B2 commit.
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
BOUND = "BOUND"
MUTABLE_NAMES = ("STATUS.json", "RUN_MANIFEST.json", "EVENT_LOG.jsonl")
CONFIG_REPO_PATH = "configs/route_a_v3_dec023_authority_runtime_sync_v1.json"
SCRIPT_REPO_PATH = "scripts/route_a_v3/dec023_authority_runtime_sync.py"
TEST_REPO_PATH = "tests/route_a_v3/test_dec023_authority_runtime_sync.py"
IMPLEMENTATION_PATHS = [CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH]
AUTHORITY_PATHS = [
    "configs/route_a_v3.yaml",
    "configs/route_a_v3_a1_qualification.json",
    "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec023.yaml",
    "docs/contracts/supersession_mrna_xeditflow_v1_1_to_route_a_v3.yaml",
    "docs/execution/route_a_v3_a1_interim.yaml",
    "docs/execution/route_a_v3_data_role_registry.yaml",
    "docs/execution/route_a_v3_decision_log.yaml",
    "docs/execution/route_a_v3_registry_manifest.json",
    "scripts/route_a_v3/validate_a0_bundle.py",
    "tests/route_a_v3/test_a0_integrity_guards.py",
]
FROZEN_I1_COMMIT = "b0afa92eea9718c15a5989cfa67bac57036617d9"
FROZEN_I1_PARENT = "f7cfff896a1a30d25a3b73ea7f89957d70d95d39"
FROZEN_I1_BLOBS = {
    CONFIG_REPO_PATH: "330a5fceaa97a1c1f16fcb20f1c6e4e35329923a293bcb463f35eaa666cb4701",
    SCRIPT_REPO_PATH: "3082aa44b70356d0e512fa8d1c92daadd08084a97594ba20976d0b93ca4706bd",
    TEST_REPO_PATH: "0ad48f947eee136e9104ba1dcdd5c921281ad0379ce2a17dcd4c411861115e65",
}
PRODUCTION_REPO_ROOT = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810"
)
BRANCH = "routea-v3-a1-20260810"
PRODUCTION_CONFIG_PATH = Path(__file__).resolve().parents[2] / CONFIG_REPO_PATH
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
UNKNOWN_BINDING_FIELDS = (
    "status",
    "implementation_commit",
    "implementation_script_sha256",
    "implementation_test_sha256",
)
ACTIVE_DECISION_IDS = [
    "V3-DEC-017",
    "V3-DEC-018",
    "V3-DEC-019",
    "V3-DEC-020",
    "V3-DEC-021",
    "V3-DEC-022",
    "V3-DEC-023",
]
ZERO_CONTRIBUTION = {
    "ordinary": 0,
    "a1": 0,
    "true_a2": 0,
    "canonical_records": 0,
}
FaultInjector = Callable[[str], None]


class RuntimeSyncError(RuntimeError):
    """The DEC023 authority/runtime contract is not satisfied."""


class BindingError(RuntimeSyncError):
    """An authority, predecessor, or implementation group is incomplete."""


class AuthorityError(RuntimeSyncError):
    """The production repository is not exact A -> I1 -> I2 -> B2."""


class PredecessorError(RuntimeSyncError):
    """The live runtime is not the frozen EVT055 predecessor."""


class PublicationError(RuntimeSyncError):
    """Prepared or runtime publication is not an exact recoverable prefix."""


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def compact_json_line(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
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
            payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
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
        raise RuntimeSyncError(
            f"{label} drift: expected {expected!r}, observed {actual!r}"
        )


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
    if values == [UNKNOWN] * 4:
        return "UNKNOWN"
    if any(value == UNKNOWN for value in values):
        raise BindingError("implementation binding is partially known")
    _expect(binding.get("status"), BOUND, label="implementation status")
    _expect_hex(binding.get("implementation_commit"), HEX40, label="I commit")
    _expect_hex(
        binding.get("implementation_script_sha256"), HEX64, label="script SHA-256"
    )
    _expect_hex(
        binding.get("implementation_test_sha256"), HEX64, label="test SHA-256"
    )
    return BOUND


def _authority_binding_state(authority: Mapping[str, Any]) -> str:
    files = authority.get("authority_files")
    if not isinstance(files, list) or len(files) != 10:
        raise BindingError("authority exact10 file identity closure differs")
    values = [
        authority.get("authority_binding_status"),
        authority.get("authority_commit"),
        authority.get("authority_expected_parent"),
    ]
    for item in files:
        if not isinstance(item, Mapping):
            raise BindingError("authority file identity is not an object")
        values.extend([item.get("bytes"), item.get("sha256")])
    if values == [UNKNOWN] * len(values):
        return "UNKNOWN"
    if any(value == UNKNOWN for value in values):
        raise BindingError("authority exact10 binding is partially known")
    _expect(
        authority.get("authority_binding_status"),
        "FROZEN_BOUND_EXACT10",
        label="authority status",
    )
    _expect_hex(authority.get("authority_commit"), HEX40, label="authority commit")
    _expect_hex(
        authority.get("authority_expected_parent"), HEX40, label="authority parent"
    )
    for item in files:
        if (
            isinstance(item.get("bytes"), bool)
            or not isinstance(item.get("bytes"), int)
            or item["bytes"] < 0
        ):
            raise BindingError("authority file byte count is invalid")
        _expect_hex(item.get("sha256"), HEX64, label="authority file SHA-256")
    return BOUND


def _predecessor_binding_state(runtime: Mapping[str, Any]) -> str:
    mutables = runtime.get("predecessor_mutables")
    tail = runtime.get("predecessor_tail")
    if not isinstance(mutables, Mapping) or set(mutables) != set(MUTABLE_NAMES):
        raise BindingError("predecessor mutable identity closure differs")
    if not isinstance(tail, Mapping):
        raise BindingError("predecessor tail identity is absent")
    values: list[Any] = [runtime.get("predecessor_binding_status")]
    for name in MUTABLE_NAMES:
        item = mutables[name]
        if not isinstance(item, Mapping):
            raise BindingError("predecessor mutable identity is not an object")
        values.extend([item.get("bytes"), item.get("sha256")])
    values.extend([tail.get("bytes"), tail.get("sha256")])
    if values == [UNKNOWN] * len(values):
        return "UNKNOWN"
    if any(value == UNKNOWN for value in values):
        raise BindingError("EVT055 predecessor binding is partially known")
    _expect(
        runtime.get("predecessor_binding_status"),
        "FROZEN_BOUND_EVT055",
        label="predecessor status",
    )
    for label, item in [
        *((name, mutables[name]) for name in MUTABLE_NAMES),
        ("EVT055 tail", tail),
    ]:
        if (
            isinstance(item.get("bytes"), bool)
            or not isinstance(item.get("bytes"), int)
            or item["bytes"] < 0
        ):
            raise BindingError(f"{label} predecessor byte count is invalid")
        _expect_hex(item.get("sha256"), HEX64, label=f"{label} predecessor SHA")
    return BOUND


def _validate_runtime_shape(config: Mapping[str, Any]) -> None:
    runtime = _expect_keys(
        config["runtime"],
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
    for key in ("run_root", "allowed_prepared_root"):
        if not isinstance(runtime[key], str) or not Path(runtime[key]).is_absolute():
            raise RuntimeSyncError(f"runtime {key} must be absolute")
    expected_scalars = {
        "fresh_production_validation_required": True,
        "predecessor_event_id": "A1-EVT-055",
        "predecessor_event_count": 55,
        "successor_event_id": "A1-EVT-056",
        "successor_event_count": 56,
        "predecessor_manifest_output_count": 238,
        "successor_manifest_output_count": 242,
        "predecessor_manifest_registered_artifact_count": 6,
        "successor_manifest_registered_artifact_count": 6,
        "sync_name": "A1_DEC023_AUTHORITY_RUNTIME_SYNC_V1.json",
        "output_delta_count": 4,
        "mutable_publish_order": list(MUTABLE_NAMES),
    }
    for key, expected in expected_scalars.items():
        _expect(runtime[key], expected, label=f"runtime.{key}")
    mutables = runtime["predecessor_mutables"]
    if not isinstance(mutables, Mapping) or set(mutables) != set(MUTABLE_NAMES):
        raise RuntimeSyncError("predecessor mutable identity closure differs")
    expected_snapshots = {
        "STATUS.json": "STATUS_PRE_DEC023_AUTHORITY_RUNTIME_SYNC_V1.json",
        "RUN_MANIFEST.json": "RUN_MANIFEST_PRE_DEC023_AUTHORITY_RUNTIME_SYNC_V1.json",
        "EVENT_LOG.jsonl": "EVENT_LOG_PRE_DEC023_AUTHORITY_RUNTIME_SYNC_V1.jsonl",
    }
    for name in MUTABLE_NAMES:
        _expect_keys(
            mutables[name], {"bytes", "sha256", "snapshot_name"}, label=name
        )
        _expect(
            mutables[name]["snapshot_name"],
            expected_snapshots[name],
            label=f"{name} snapshot",
        )
    tail = _expect_keys(
        runtime["predecessor_tail"],
        {"event_id", "decision_id", "bytes", "sha256"},
        label="predecessor tail",
    )
    _expect(tail["event_id"], "A1-EVT-055", label="tail event")
    _expect(tail["decision_id"], "V3-DEC-022", label="tail decision")
    _expect(
        runtime["immutable_publish_order"],
        [expected_snapshots[name] for name in MUTABLE_NAMES]
        + ["A1_DEC023_AUTHORITY_RUNTIME_SYNC_V1.json"],
        label="immutable publish order",
    )
    _predecessor_binding_state(runtime)


def validate_static_config(config: dict[str, Any]) -> None:
    """Validate the staged/I/B config without opening repository or runtime paths."""

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
            "dec023_authority",
            "runtime_authority",
            "registered_artifacts",
            "runtime",
            "frozen_outer_truth",
            "access_boundary",
            "publication_policy",
        },
        label="config root",
    )
    _expect(
        {
            key: config[key]
            for key in (
                "schema_version",
                "protocol_id",
                "contract_id",
                "phase_id",
                "dataset_ids",
                "decision_id",
                "event_id",
                "event_name",
                "sync_type",
            )
        },
        {
            "schema_version": "route_a_v3_dec023_authority_runtime_sync.v1",
            "protocol_id": "ROUTE_A_V3_DEC023_AUTHORITY_RUNTIME_SYNC_V1",
            "contract_id": "mrna_xeditflow_route_a_v3",
            "phase_id": "A1",
            "dataset_ids": ["GSE261709", "GSE207584"],
            "decision_id": "V3-DEC-023",
            "event_id": "A1-EVT-056",
            "event_name": (
                "DEC023_DUAL_AGGREGATE_ONLY_PREFLIGHT_AUTHORITIES_REGISTERED_"
                "RUNTIME_GATES_UNCHANGED"
            ),
            "sync_type": (
                "APPEND_ONLY_AUTHORITY_ONLY_REGISTRATION_"
                "NO_SCIENTIFIC_STATE_CHANGE"
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
            "APPEND_ONLY_A_EXACT10_THEN_FROZEN_I1_EXACT3_THEN_DYNAMIC_I2_"
            "EXACT3_THEN_B2_CONFIG_ONLY_V1"
        ),
        label="binding scheme",
    )
    _expect(binding["implementation_script_path"], SCRIPT_REPO_PATH, label="script")
    _expect(binding["implementation_test_path"], TEST_REPO_PATH, label="test")
    _expect(
        binding["unknown_to_bound_scalar_paths"],
        [f"implementation_binding.{field}" for field in UNKNOWN_BINDING_FIELDS],
        label="implementation four-scalar paths",
    )
    _expect(
        binding["implementation_commit_exact_changed_paths"],
        IMPLEMENTATION_PATHS,
        label="I2 exact3 paths",
    )
    _expect(
        binding["binding_commit_exact_changed_paths"],
        [CONFIG_REPO_PATH],
        label="B2 config-only path",
    )
    if not isinstance(binding["activation_rule"], str) or not binding["activation_rule"]:
        raise BindingError("activation rule is absent")
    implementation_state = _implementation_binding_state(binding)

    authority = _expect_keys(
        config["repository_authority"],
        {
            "production_repo_root",
            "branch",
            "authority_binding_status",
            "authority_commit",
            "authority_expected_parent",
            "authority_exact_changed_paths",
            "authority_files",
            "predecessor_implementation_i1",
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
    _expect(authority["authority_exact_changed_paths"], AUTHORITY_PATHS, label="A exact10")
    _expect(
        [item.get("path") for item in authority["authority_files"]],
        AUTHORITY_PATHS,
        label="A file identities",
    )
    for item in authority["authority_files"]:
        _expect_keys(item, {"path", "bytes", "sha256"}, label="authority file")
    _expect(
        authority["predecessor_implementation_i1"],
        {
            "status": "FROZEN_BOUND_EXACT3",
            "commit": FROZEN_I1_COMMIT,
            "expected_parent": FROZEN_I1_PARENT,
            "exact_changed_paths": IMPLEMENTATION_PATHS,
            "blob_sha256_by_path": FROZEN_I1_BLOBS,
        },
        label="frozen I1 lifecycle",
    )
    _expect(
        authority["implementation_exact_changed_paths"],
        IMPLEMENTATION_PATHS,
        label="authority I2 exact3",
    )
    _expect(
        authority["binding_exact_changed_paths"],
        [CONFIG_REPO_PATH],
        label="authority B2 config-only",
    )
    authority_state = _authority_binding_state(authority)
    _validate_runtime_shape(config)
    predecessor_state = _predecessor_binding_state(config["runtime"])
    if implementation_state == BOUND and (
        authority_state != BOUND or predecessor_state != BOUND
    ):
        raise BindingError(
            "BOUND implementation requires BOUND authority and predecessor"
        )

    dec023 = config["dec023_authority"]
    if not isinstance(dec023, dict):
        raise RuntimeSyncError("DEC023 authority is not an object")
    _expect(dec023.get("decision_id"), "V3-DEC-023", label="DEC023 decision")
    _expect(
        dec023.get("status"),
        "FROZEN_USER_AUTHORIZED_DUAL_AGGREGATE_ONLY_PREFLIGHTS",
        label="DEC023 status",
    )
    for key in (
        "authority_sync_executes_preflight",
        "authority_sync_qualifies_study",
        "authority_sync_changes_counts",
        "scientific_state_changed",
        "qualification_allowed",
        "canonical_materialization_allowed",
        "split_execution_allowed",
        "formal_qualification_power_gate_execution_allowed",
        "training_allowed",
        "gpu_work_allowed",
        "model_selection_allowed",
        "a7_allowed",
        "next_phase_authorized",
    ):
        _expect(dec023.get(key), False, label=f"DEC023.{key}")
    _expect(dec023.get("preflight_status_after_sync"), "AUTHORIZED_NOT_RUN", label="preflight status")
    gse261709 = dec023.get("gse261709")
    gse207584 = dec023.get("gse207584")
    if not isinstance(gse261709, dict) or not isinstance(gse207584, dict):
        raise RuntimeSyncError("DEC023 dataset authority is absent")
    _expect(gse261709.get("project_id"), "PRJNA1088465", label="GSE261709 project")
    _expect(
        gse261709.get("role"),
        "PUBLIC_IDENTIFIER_ASSET_SCHEMA_AND_AGGREGATE_GEOMETRY_PREFLIGHT_ONLY",
        label="GSE261709 role",
    )
    for key in (
        "actual_header_names_output_allowed",
        "asset_body_read_allowed",
        "member_payload_read_allowed",
        "row_level_access_allowed",
    ):
        _expect(gse261709.get(key), False, label=f"GSE261709.{key}")
    _expect(
        gse261709.get("all_pass_result"),
        "REQUEST_SEPARATE_ROW_LEVEL_AUTHORITY_ONLY",
        label="GSE261709 all-pass result",
    )
    _expect(gse207584.get("project_id"), "PRJNA856272", label="GSE207584 project")
    _expect(
        gse207584.get("role"),
        "AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_ONLY",
        label="GSE207584 role",
    )
    _expect(gse207584.get("registry_role_must_remain"), "AUDIT_ONLY", label="GSE207584 registry role")
    _expect(
        gse207584.get("allowed_output_class"),
        "AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_ONLY",
        label="GSE207584 output class",
    )
    _expect(
        gse207584.get("allowed_output_payload_classes_exactly"),
        ["AGGREGATE_COUNTS", "AGGREGATE_HISTOGRAMS", "GATE_STATUSES"],
        label="GSE207584 aggregate-only payload classes",
    )
    _expect(len(gse207584.get("required_fail_closed_gate_ids_exactly", [])), 11, label="GSE207584 gate count")
    for key in (
        "member_identifier_output_allowed",
        "sequence_output_allowed",
        "row_abundance_effect_slope_or_se_output_allowed",
        "split_assignment_output_allowed",
        "nine_observations_may_substitute_for_three_biological_replicates",
    ):
        _expect(gse207584.get(key), False, label=f"GSE207584.{key}")
    _expect(gse207584.get("three_biological_replicates_required"), True, label="GSE207584 biological replicate rule")
    _expect(gse207584.get("analysis_and_bootstrap_unit"), "BIOLOGICAL_SOURCE_GROUP", label="GSE207584 analysis unit")

    runtime_authority = config["runtime_authority"]
    _expect(
        runtime_authority,
        {
            "historical_active_authority_commit_policy": (
                "PRESERVE_PREDECESSOR_VALUE_UNCHANGED"
            ),
            "active_amendment_decision_ids": ACTIVE_DECISION_IDS,
            "current_contract_authority_scope": (
                "DEC023_GSE261709_AND_GSE207584_DUAL_AGGREGATE_ONLY_PREFLIGHT_"
                "AUTHORITIES_NO_EXECUTION_NO_QUALIFICATION"
            ),
        },
        label="runtime authority",
    )
    _expect(config["registered_artifacts"], [], label="new registered artifacts")
    frozen = config["frozen_outer_truth"]
    _expect(
        frozen.get("current_qualified_counts"),
        {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547},
        label="frozen counts",
    )
    _expect(frozen.get("gse261709_contribution"), ZERO_CONTRIBUTION, label="GSE261709 contribution")
    _expect(frozen.get("gse207584_contribution"), ZERO_CONTRIBUTION, label="GSE207584 contribution")
    for key in (
        "a1_complete",
        "qualified",
        "training_started",
        "training_allowed",
        "training_authorized",
        "gpu_work_started",
        "gpu_work_allowed",
        "model_selection_allowed",
        "a7_allowed",
        "next_phase_authorized",
    ):
        _expect(frozen.get(key), False, label=f"frozen.{key}")
    _expect(frozen.get("run_status"), "IN_PROGRESS", label="run status")
    _expect(
        frozen.get("evidence_status"),
        "SCRATCH_ROUTE_QUALIFIED_GLOBAL_PHASE_INCOMPLETE",
        label="evidence status",
    )
    _expect(
        frozen.get("gate_status"),
        "A1_PHASE_INCOMPLETE_GLOBAL_REQUIREMENTS",
        label="gate status",
    )
    _expect(frozen.get("scientific_claim_status"), "NOT_ESTABLISHED", label="claim")
    access = config["access_boundary"]
    if not isinstance(access, dict):
        raise RuntimeSyncError("access boundary is not an object")
    for key, value in access.items():
        if key in {"restricted_or_sealed_path_accessed", "gse246381_contact"}:
            _expect(value, False, label=f"access.{key}")
        else:
            _expect(value, 0, label=f"access.{key}")
    _expect(
        config["publication_policy"],
        {
            "registered_artifacts_remain_empty_for_this_sync": True,
            "predecessor_snapshots_are_immutable_runtime_outputs": True,
            "sync_record_is_immutable_runtime_output": True,
            "mutables_commit_after_all_immutables": True,
            "mutable_commit_order": list(MUTABLE_NAMES),
            "event_is_last_commit": True,
            "supported_recovery": "EXACT_PUBLICATION_PREFIX_ONLY",
        },
        label="publication policy",
    )


def validate_bound_config(config: dict[str, Any]) -> None:
    validate_static_config(config)
    if _authority_binding_state(config["repository_authority"]) != BOUND:
        raise BindingError("DEC023 exact10 authority remains UNKNOWN_NOT_ASSERTED")
    if _predecessor_binding_state(config["runtime"]) != BOUND:
        raise BindingError("EVT055 predecessor remains UNKNOWN_NOT_ASSERTED")
    if _implementation_binding_state(config["implementation_binding"]) != BOUND:
        raise BindingError("runtime-sync implementation remains UNKNOWN_NOT_ASSERTED")


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


def expected_unknown_i_config(bound_config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(bound_config)
    for field in UNKNOWN_BINDING_FIELDS:
        result["implementation_binding"][field] = UNKNOWN
    return result


def _run_git(repo_root: Path, *arguments: str) -> bytes:
    environment = os.environ.copy()
    environment.update({"GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C", "LANG": "C"})
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AuthorityError("git authority command failed") from exc
    return completed.stdout


def _git_blob(repo_root: Path, commit: str, relative_path: str) -> bytes:
    return _run_git(repo_root, "show", f"{commit}:{relative_path}")


def _changed_paths(repo_root: Path, commit: str) -> list[str]:
    payload = _run_git(
        repo_root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        commit,
    )
    return sorted(line for line in payload.decode().splitlines() if line)


def _read_repo_file(repo_root: Path, relative_path: str) -> bytes:
    try:
        return (repo_root / relative_path).read_bytes()
    except OSError as exc:
        raise AuthorityError(f"cannot read repository file: {relative_path}") from exc


def audit_production_repository_authority(
    config: dict[str, Any], config_payload: bytes
) -> dict[str, Any]:
    """Prove exact10 A -> frozen exact3 I1 -> exact3 I2 -> B2 before I/O."""

    validate_bound_config(config)
    authority = config["repository_authority"]
    binding = config["implementation_binding"]
    repo_root = Path(authority["production_repo_root"])
    expected_script = (repo_root / SCRIPT_REPO_PATH).resolve()
    if Path(__file__).resolve() != expected_script:
        raise AuthorityError("executing producer is not the bound repository script")
    head = _run_git(repo_root, "rev-parse", "HEAD").decode().strip()
    _expect(_run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD").decode().strip(), BRANCH, label="branch")
    upstream_name = _run_git(repo_root, "rev-parse", "--abbrev-ref", "@{upstream}").decode().strip()
    _expect(upstream_name, f"origin/{BRANCH}", label="upstream")
    upstream_head = _run_git(repo_root, "rev-parse", "@{upstream}").decode().strip()
    origin_head = _run_git(repo_root, "rev-parse", "--verify", f"refs/remotes/origin/{BRANCH}").decode().strip()
    _expect(head, upstream_head, label="HEAD/upstream")
    _expect(head, origin_head, label="HEAD/origin")
    if _run_git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise AuthorityError("production worktree or index is dirty")

    implementation_i2_commit = binding["implementation_commit"]
    frozen_i1 = authority["predecessor_implementation_i1"]
    frozen_i1_commit = frozen_i1["commit"]
    authority_commit = authority["authority_commit"]
    _expect(_run_git(repo_root, "rev-parse", f"{head}^").decode().strip(), implementation_i2_commit, label="B2 parent/I2")
    _expect(_run_git(repo_root, "rev-parse", f"{implementation_i2_commit}^").decode().strip(), frozen_i1_commit, label="I2 parent/I1")
    _expect(_run_git(repo_root, "rev-parse", f"{frozen_i1_commit}^").decode().strip(), authority_commit, label="I1 parent/A")
    _expect(_run_git(repo_root, "rev-parse", f"{authority_commit}^").decode().strip(), authority["authority_expected_parent"], label="A parent")
    _expect(_changed_paths(repo_root, authority_commit), sorted(AUTHORITY_PATHS), label="A exact10")
    _expect(_changed_paths(repo_root, frozen_i1_commit), sorted(IMPLEMENTATION_PATHS), label="I1 exact3")
    _expect(_changed_paths(repo_root, implementation_i2_commit), sorted(IMPLEMENTATION_PATHS), label="I2 exact3")
    _expect(_changed_paths(repo_root, head), [CONFIG_REPO_PATH], label="B2 config-only")

    _expect(_git_blob(repo_root, head, CONFIG_REPO_PATH), config_payload, label="B2 config blob")
    for relative, expected_digest in frozen_i1["blob_sha256_by_path"].items():
        if sha256(_git_blob(repo_root, frozen_i1_commit, relative)) != expected_digest:
            raise AuthorityError("frozen I1 blob identity differs")
    frozen_i1_config = load_json(
        _git_blob(repo_root, frozen_i1_commit, CONFIG_REPO_PATH),
        label="frozen I1 config",
    )
    if [
        frozen_i1_config.get("implementation_binding", {}).get(field)
        for field in UNKNOWN_BINDING_FIELDS
    ] != [UNKNOWN] * 4:
        raise AuthorityError("frozen I1 config did not retain four UNKNOWN scalars")

    i2_config_payload = _git_blob(
        repo_root, implementation_i2_commit, CONFIG_REPO_PATH
    )
    i2_config = load_json(i2_config_payload, label="I2 config")
    expected_i2 = expected_unknown_i_config(config)
    if not _typed_equal(i2_config, expected_i2):
        raise AuthorityError("I2 config is not the exact four-scalar UNKNOWN form")
    validate_static_config(i2_config)

    for item in authority["authority_files"]:
        relative = item["path"]
        a_blob = _git_blob(repo_root, authority_commit, relative)
        if len(a_blob) != item["bytes"] or sha256(a_blob) != item["sha256"]:
            raise AuthorityError("authority exact10 blob identity differs")
        if _git_blob(repo_root, frozen_i1_commit, relative) != a_blob:
            raise AuthorityError("authority blob did not persist through I1")
        if _git_blob(repo_root, implementation_i2_commit, relative) != a_blob:
            raise AuthorityError("authority blob did not persist through I2")
        if _git_blob(repo_root, head, relative) != a_blob:
            raise AuthorityError("authority blob did not persist through B2")
        if _read_repo_file(repo_root, relative) != a_blob:
            raise AuthorityError("working authority file differs from bound A")

    script_blob = _git_blob(repo_root, implementation_i2_commit, SCRIPT_REPO_PATH)
    test_blob = _git_blob(repo_root, implementation_i2_commit, TEST_REPO_PATH)
    if sha256(script_blob) != binding["implementation_script_sha256"]:
        raise AuthorityError("I2 implementation script identity differs")
    if sha256(test_blob) != binding["implementation_test_sha256"]:
        raise AuthorityError("I2 implementation test identity differs")
    for commit in (head,):
        if _git_blob(repo_root, commit, SCRIPT_REPO_PATH) != script_blob:
            raise AuthorityError("script changed in config-only B2")
        if _git_blob(repo_root, commit, TEST_REPO_PATH) != test_blob:
            raise AuthorityError("test changed in config-only B2")
    if _read_repo_file(repo_root, SCRIPT_REPO_PATH) != script_blob:
        raise AuthorityError("working script differs from bound I2")
    if _read_repo_file(repo_root, TEST_REPO_PATH) != test_blob:
        raise AuthorityError("working test differs from bound I2")
    return {
        "status": "PASS_EXACT10_A_TO_FROZEN_EXACT3_I1_TO_EXACT3_I2_TO_CONFIG_ONLY_B2",
        "authority_commit": authority_commit,
        "frozen_i1_commit": frozen_i1_commit,
        "implementation_i2_commit": implementation_i2_commit,
        "binding_b2_commit": head,
        "authority_blob_count": 10,
        "worktree_and_index_clean": True,
    }


def _absolute(path: Path | str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise PublicationError("path must be absolute")
    return candidate


def _prepared_path(
    prepared_directory: Path | str, config: Mapping[str, Any]
) -> Path:
    prepared = _absolute(prepared_directory)
    allowed = _absolute(config["runtime"]["allowed_prepared_root"])
    try:
        prepared.relative_to(allowed)
    except ValueError as exc:
        raise PublicationError("prepared directory is outside the allowed root") from exc
    if prepared == allowed:
        raise PublicationError("prepared directory must be a child of the allowed root")
    return prepared


@contextmanager
def _locked_run(run_root: Path) -> Iterator[None]:
    lock_path = run_root / ".DEC023_AUTHORITY_RUNTIME_SYNC.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    except OSError as exc:
        raise PublicationError("cannot open runtime lock") from exc
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _read_runtime(run_root: Path) -> dict[str, bytes]:
    try:
        return {name: (run_root / name).read_bytes() for name in MUTABLE_NAMES}
    except OSError as exc:
        raise PredecessorError("cannot read the three runtime mutables") from exc


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


def _outer_runtime_fields(config: Mapping[str, Any]) -> dict[str, Any]:
    frozen = config["frozen_outer_truth"]
    counts = frozen["current_qualified_counts"]
    return {
        "qualified_ordinary_studies": counts["ordinary"],
        "qualified_a1_studies": counts["a1"],
        "qualified_a2_dense_studies": counts["true_a2"],
        "canonical_intervention_record_count": counts["canonical_records"],
        "canonical_record_count": counts["canonical_records"],
        "run_status": frozen["run_status"],
        "evidence_status": frozen["evidence_status"],
        "gate_status": frozen["gate_status"],
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


def _validate_outer_document(
    config: Mapping[str, Any], document: Mapping[str, Any], *, label: str
) -> None:
    for key, expected in _outer_runtime_fields(config).items():
        _expect(document.get(key), expected, label=f"{label}.{key}")


def validate_predecessor(
    config: dict[str, Any], payloads: Mapping[str, bytes]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Freshly prove exact EVT055/55/238/6 before any prepared write."""

    if _predecessor_binding_state(config["runtime"]) != BOUND:
        raise BindingError("EVT055 predecessor remains UNKNOWN_NOT_ASSERTED")
    if set(payloads) != set(MUTABLE_NAMES):
        raise PredecessorError("runtime mutable member closure differs")
    runtime = config["runtime"]
    for name in MUTABLE_NAMES:
        _check_payload_identity(
            payloads[name], runtime["predecessor_mutables"][name], label=name
        )
    status, manifest, events = _parse_runtime(payloads)
    if len(events) != 55:
        raise PredecessorError("predecessor event count is not 55")
    expected_ids = [f"A1-EVT-{index:03d}" for index in range(1, 56)]
    if [event.get("event_id") for event in events] != expected_ids:
        raise PredecessorError("predecessor event identifiers are not exact 1..55")
    _expect(events[-1].get("decision_id"), "V3-DEC-022", label="tail decision")
    tail_payload = payloads["EVENT_LOG.jsonl"].splitlines(keepends=True)[-1]
    _check_payload_identity(
        tail_payload, runtime["predecessor_tail"], label="EVT055 tail"
    )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 238:
        raise PredecessorError("predecessor manifest output count is not 238")
    paths = [item.get("absolute_path") for item in outputs if isinstance(item, dict)]
    if len(paths) != 238 or len(set(paths)) != 238:
        raise PredecessorError("predecessor manifest output paths are not unique")
    _expect(
        manifest.get("registered_artifact_count"),
        6,
        label="predecessor registered artifact count",
    )
    _validate_outer_document(config, status, label="predecessor STATUS")
    _validate_outer_document(config, manifest, label="predecessor RUN_MANIFEST")
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
        raise PredecessorError("EVT056 timestamp must follow EVT055 with an offset")


def _current_contract_authority(config: Mapping[str, Any]) -> dict[str, Any]:
    authority = config["repository_authority"]
    return {
        "decision_id": "V3-DEC-023",
        "authority_commit": authority["authority_commit"],
        "authority_expected_parent": authority["authority_expected_parent"],
        "scope": config["runtime_authority"]["current_contract_authority_scope"],
        "authority_file_count": 10,
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
            "record_type": "ROUTE_A_V3_A1_DEC023_AUTHORITY_RUNTIME_SYNC",
            "sync_type": config["sync_type"],
            "contract_id": config["contract_id"],
            "phase_id": "A1",
            "decision_id": "V3-DEC-023",
            "event_id": "A1-EVT-056",
            "recorded_at": recorded_at,
            "predecessor_event_id": "A1-EVT-055",
            "registered_artifacts": [],
            "registered_artifact_count": 0,
            "new_registered_artifact_count": 0,
            "predecessor_snapshot_count": 3,
            "predecessor_snapshot_names": list(snapshots),
            "snapshot_sha256": {
                name: sha256(payload) for name, payload in snapshots.items()
            },
            "output_delta_count": 4,
            "manifest_output_count_before": 238,
            "manifest_output_count_after": 242,
            "manifest_registered_artifact_count_before": 6,
            "manifest_registered_artifact_count_after": 6,
            "dec023_authority": copy.deepcopy(config["dec023_authority"]),
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
        "event_id": "A1-EVT-056",
        "at": recorded_at,
        "phase_id": "A1",
        "event": config["event_name"],
        "sync_type": config["sync_type"],
        "decision_id": "V3-DEC-023",
        "predecessor_event_id": "A1-EVT-055",
        "registered_artifacts": [],
        "registered_artifact_count": 0,
        "new_registered_artifact_count": 0,
        "predecessor_snapshot_count": 3,
        "predecessor_snapshot_names": list(_snapshot_names(config).values()),
        "sync_name": config["runtime"]["sync_name"],
        "sync_record_sha256": sync_digest,
        "output_delta_count": 4,
        "manifest_output_count_before": 238,
        "manifest_output_count_after": 242,
        "manifest_registered_artifact_count_before": 6,
        "manifest_registered_artifact_count_after": 6,
        "active_amendment_decision_ids": copy.deepcopy(ACTIVE_DECISION_IDS),
        "current_contract_authority": _current_contract_authority(config),
        "dec023_authority": copy.deepcopy(config["dec023_authority"]),
        "frozen_outer_truth": copy.deepcopy(config["frozen_outer_truth"]),
        "access_boundary": copy.deepcopy(config["access_boundary"]),
        "preflight_executed": False,
        "scientific_state_changed": False,
        "evidence_gate_statuses_changed": False,
        "overall_qualification_gate_changed": False,
        "qualification_changed": False,
        "training_started": False,
        "training_allowed": False,
        "gpu_work_started": False,
        "gpu_work_allowed": False,
        "model_selection_allowed": False,
        "a7_allowed": False,
        "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "detail": (
            "Registered only V3-DEC-023 repository authority for the ordinary-"
            "public GSE261709 identifier/schema/aggregate-geometry preflight and "
            "GSE207584 aggregate dense-family qualification preflight. Neither "
            "preflight was executed; no public asset, member, row, sequence, "
            "abundance, effect, slope, SE, split, private, or sealed payload was "
            "read. Existing 1/1/0 qualified-study counts, 6547 canonical records, "
            "incomplete A1 state, and every training, GPU, model-selection, A7, "
            "next-phase, and scientific-claim lock remain unchanged."
        ),
    }


def _successor_updates(
    config: Mapping[str, Any], recorded_at: str, sync_digest: str
) -> dict[str, Any]:
    return {
        "active_amendment_decision_ids": copy.deepcopy(ACTIVE_DECISION_IDS),
        "current_contract_authority": _current_contract_authority(config),
        "dec023_authority_runtime_sync_status": "SYNCED_EVT_056",
        "dec023_authority_runtime_sync_recorded_at": recorded_at,
        "dec023_authority_runtime_sync_record_sha256": sync_digest,
        "dec023_authority_runtime_sync_scientific_state_changed": False,
        "dec023_authority_runtime_sync_gate_changed": False,
        "dec023_authority_runtime_sync_qualification_changed": False,
        "gse261709_dec023_aggregate_geometry_preflight_status": "AUTHORIZED_NOT_RUN",
        "gse207584_dec023_dense_family_preflight_status": "AUTHORIZED_NOT_RUN",
        "gse261709_contribution": copy.deepcopy(ZERO_CONTRIBUTION),
        "gse207584_contribution": copy.deepcopy(ZERO_CONTRIBUTION),
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
            f"A1_{name.replace('.', '_').upper()}_PRE_DEC023_AUTHORITY_RUNTIME_SYNC_SNAPSHOT",
            run_root / snapshots[name],
            predecessor_payloads[name],
        )
        for name in MUTABLE_NAMES
    ]
    records.append(
        _output_record(
            "A1_DEC023_AUTHORITY_RUNTIME_SYNC_V1",
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
    validate_bound_config(config)
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
        "authority_blob_count": 10,
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
        len(events) != 56
        or events[:-1] != old_events
        or not successors["EVENT_LOG.jsonl"].startswith(
            predecessor_payloads["EVENT_LOG.jsonl"]
        )
    ):
        raise RuntimeSyncError("EVENT_LOG is not one exact EVT056 append")
    event = events[-1]
    _expect(event.get("event_id"), "A1-EVT-056", label="successor event")
    _expect(event.get("decision_id"), "V3-DEC-023", label="successor decision")
    _expect(event.get("registered_artifacts"), [], label="event artifacts")
    _expect(event.get("new_registered_artifact_count"), 0, label="event new artifacts")
    _expect(event.get("preflight_executed"), False, label="event preflight")
    _expect(event.get("scientific_state_changed"), False, label="event science")
    _expect(event.get("qualification_changed"), False, label="event qualification")
    _expect(event.get("a7_allowed"), False, label="event A7")

    sync_payload = successors[config["runtime"]["sync_name"]]
    sync_digest = sha256(sync_payload)
    sync = load_json(sync_payload, label="DEC023 authority runtime sync")
    _expect(sync.get("event_id"), "A1-EVT-056", label="sync event")
    _expect(sync.get("decision_id"), "V3-DEC-023", label="sync decision")
    _expect(sync.get("registered_artifacts"), [], label="sync artifacts")
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
    _expect(
        sync.get("frozen_outer_truth"),
        config["frozen_outer_truth"],
        label="sync outer truth",
    )
    _expect(
        sync.get("access_boundary"),
        config["access_boundary"],
        label="sync access",
    )

    updates = _successor_updates(config, event["at"], sync_digest)
    expected_status = copy.deepcopy(old_status)
    expected_status["updated_at"] = event["at"]
    expected_status.update(updates)
    _expect(status, expected_status, label="successor STATUS closure")
    expected_manifest = copy.deepcopy(old_manifest)
    expected_manifest.update(updates)
    output_delta = _immutable_output_delta(
        config, predecessor_payloads, sync_payload
    )
    expected_manifest["outputs"] = list(old_manifest["outputs"]) + output_delta
    _expect(manifest, expected_manifest, label="successor manifest closure")
    _expect(
        manifest.get("active_authority_commit"),
        old_manifest.get("active_authority_commit"),
        label="historical active authority preservation",
    )
    _expect(
        manifest.get("registered_artifact_count"),
        6,
        label="registered artifact count preservation",
    )
    outputs = manifest["outputs"]
    if len(outputs) != 242 or outputs[:238] != old_manifest["outputs"]:
        raise RuntimeSyncError("manifest ordered 238-to-242 append differs")
    if outputs[238:] != output_delta:
        raise RuntimeSyncError("manifest exact4 output delta differs")
    if len({item.get("absolute_path") for item in outputs}) != 242:
        raise RuntimeSyncError("successor output paths are not unique")
    _expect(
        [Path(item["absolute_path"]).name for item in outputs[-4:]],
        config["runtime"]["immutable_publish_order"],
        label="manifest exact4 output names",
    )
    _expect(event.get("sync_record_sha256"), sync_digest, label="event sync digest")
    _validate_outer_document(config, status, label="successor STATUS")
    _validate_outer_document(config, manifest, label="successor RUN_MANIFEST")


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
        if path.read_bytes() != payload:
            raise PublicationError("existing immutable output differs")
        return "EXISTING_EXACT"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise PublicationError("existing immutable output differs")
            return "EXISTING_EXACT"
    except PublicationError:
        raise
    except OSError as exc:
        raise PublicationError("cannot create immutable output") from exc
    finally:
        if temporary_path.exists():
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
    if config_override is None:
        config, payload = _load_config_payload(config_path, require_bound=True)
        audit = (
            audit_production_repository_authority(config, payload)
            if production
            else None
        )
        return config, audit
    config = copy.deepcopy(config_override)
    validate_bound_config(config)
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
        "event_id": "A1-EVT-056",
        "prepared_directory": str(prepared),
        "prepared_member_count": 7,
        "manifest_output_transition": "238_TO_242",
        "manifest_registered_artifact_transition": "6_TO_6",
        "new_runtime_output_count": 4,
        "new_registered_artifact_count": 0,
        "scientific_state_changed": False,
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
                raise PredecessorError("runtime mutable is neither EVT055 nor EVT056")
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
                "event_id": "A1-EVT-056",
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
                    "event_id": "A1-EVT-056",
                    "immutable_results": immutable_results,
                }
            raise PublicationError(
                "EVT056 was not committed; retry the same prepared directory"
            ) from exc
        final = _read_runtime(run_root)
        if final != successor:
            raise PublicationError("EVT056 publication finished non-exactly")
        return {
            "status": "PUBLISHED_VERIFIED",
            "event_id": "A1-EVT-056",
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
            raise PublicationError("runtime does not match prepared EVT056")
        for name in config["runtime"]["immutable_publish_order"]:
            if (run_root / name).read_bytes() != prepared[name]:
                raise PublicationError("immutable output does not match prepared")
    return {
        "status": "PUBLISHED_VERIFIED",
        "event_id": "A1-EVT-056",
        "scientific_state_changed": False,
    }


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
