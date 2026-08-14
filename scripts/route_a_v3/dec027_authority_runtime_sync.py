#!/usr/bin/env python3
"""Prepare, publish, and validate the authority-only A1-EVT-059 sync.

The transaction registers only the frozen V3-DEC-027 repository authority. It
does not execute any rescue preflight, register evidence, evaluate the stop
rule, promote a dataset, or change scientific counts and locks. Publication is
immutable-first, followed by STATUS, RUN_MANIFEST, and EVENT_LOG as the sole
recoverable mutable prefix.
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
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any


UNKNOWN = "UNKNOWN_NOT_ASSERTED"
BOUND = "BOUND"
MUTABLE_NAMES = ("STATUS.json", "RUN_MANIFEST.json", "EVENT_LOG.jsonl")
CONFIG_REPO_PATH = "configs/route_a_v3_dec027_authority_runtime_sync_v1.json"
SCRIPT_REPO_PATH = "scripts/route_a_v3/dec027_authority_runtime_sync.py"
TEST_REPO_PATH = "tests/route_a_v3/test_dec027_authority_runtime_sync.py"
IMPLEMENTATION_PATHS = [CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH]
I1_COMMIT = "de40c58ab81fc06196be3bb9ffb5aa35d39c9d03"
I1_FILES = [
    {
        "path": CONFIG_REPO_PATH,
        "bytes": 13238,
        "sha256": "cbb0c8c4fb2b47a1e1c1dd629d46c149b1c35299e8127dfd431a22666e40dfd5",
    },
    {
        "path": SCRIPT_REPO_PATH,
        "bytes": 61354,
        "sha256": "ae13a64ecbe10edd47eb403a328bdb2563f7d30e185d11e73d1b10f1955da4a1",
    },
    {
        "path": TEST_REPO_PATH,
        "bytes": 25181,
        "sha256": "de253087afd14136f188d3c525be76788cd214509d96971c4f11d2799c215117",
    },
]
PRODUCTION_CONFIG_PATH = Path(__file__).resolve().parents[2] / CONFIG_REPO_PATH
PRODUCTION_REPO_ROOT = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810"
)
BRANCH = "routea-v3-a1-20260810"
AUTHORITY_COMMIT = "3e0ad158a0b45b2f26ed82da3afe60667c712cd6"
AUTHORITY_PARENT = "b1ca33d852bad111ff31b4f60493d8c43c63d1a3"
FROZEN_I1_BINDING = {
    "status": "FROZEN_BOUND_EXACT3",
    "implementation_commit": I1_COMMIT,
    "implementation_expected_parent": AUTHORITY_COMMIT,
    "implementation_exact_changed_paths": IMPLEMENTATION_PATHS,
    "implementation_files": I1_FILES,
}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FaultInjector = Callable[[str], None]

AUTHORITY_PATHS = [
    "configs/route_a_v3.yaml",
    "configs/route_a_v3_a1_qualification.json",
    "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec027.yaml",
    "docs/contracts/supersession_mrna_xeditflow_v1_1_to_route_a_v3.yaml",
    "docs/execution/route_a_v3_a1_interim.yaml",
    "docs/execution/route_a_v3_a6_interim.yaml",
    "docs/execution/route_a_v3_data_role_registry.yaml",
    "docs/execution/route_a_v3_decision_log.yaml",
    "docs/execution/route_a_v3_registry_manifest.json",
    "docs/execution/route_a_v3_task_registry.yaml",
    "scripts/route_a_v3/validate_a0_bundle.py",
    "tests/route_a_v3/test_a0_integrity_guards.py",
]
AUTHORITY_FILES = [
    {"path": AUTHORITY_PATHS[0], "bytes": 49840, "sha256": "c5ec7d236443b506c09fd3f09e149ce5d082daff618887989af6e59472727a27"},
    {"path": AUTHORITY_PATHS[1], "bytes": 40039, "sha256": "261339c38f4b8bbd48bf8f63f6a588be57af9f6229119e84bf661d7ee8f855db"},
    {"path": AUTHORITY_PATHS[2], "bytes": 16968, "sha256": "2a27c296539e8e665873363778d91cc223f56f933a815e9509b10a7267f6b5c4"},
    {"path": AUTHORITY_PATHS[3], "bytes": 8065, "sha256": "1c4b6e29c09eb24798207138047b68909d7bda8bacc3a2eab8e17a7ca789b44b"},
    {"path": AUTHORITY_PATHS[4], "bytes": 217344, "sha256": "fb50929ae2dfa0bdd1c50b003fc43c0e13b89baade203e8384c8b6ea3eba7b1e"},
    {"path": AUTHORITY_PATHS[5], "bytes": 7723, "sha256": "1d44bcfe8669a55dc42f619ed43178f0637e20a4297ca996ecab3f7165612769"},
    {"path": AUTHORITY_PATHS[6], "bytes": 43722, "sha256": "80217a8114286f84960237819ac5b2d5828afbc23af118576541cc7cee64ae4e"},
    {"path": AUTHORITY_PATHS[7], "bytes": 48841, "sha256": "e0dc2a7fb186c5c8d00c1c5604602b1b2f87b26241191d4da55a405a02387e05"},
    {"path": AUTHORITY_PATHS[8], "bytes": 27179, "sha256": "73a39a566aa0310a80cc83f4eb17ddb95cabc87e5b070ef6484c05178ba32b75"},
    {"path": AUTHORITY_PATHS[9], "bytes": 17676, "sha256": "a64d0b8bb5eb466b06daa46ed109bd19901ee775910bc5cc9221c39ead63a4bc"},
    {"path": AUTHORITY_PATHS[10], "bytes": 830950, "sha256": "81d1a8dc49375f53a1edd5c3f41625e734eacc31a4c328c8137340a041d77e65"},
    {"path": AUTHORITY_PATHS[11], "bytes": 209643, "sha256": "106f847e957a40fffec4c1b57f8f572325ef96e8a3dc5729db68499e320f380b"},
]
ACTIVE_DECISION_IDS = [
    "V3-DEC-017",
    "V3-DEC-018",
    "V3-DEC-019",
    "V3-DEC-020",
    "V3-DEC-021",
    "V3-DEC-022",
    "V3-DEC-023",
    "V3-DEC-024",
    "V3-DEC-027",
]
ROUTES = [
    "GSE217518_CORRECTED_A1_SUCCESSOR",
    "ENCSR854RUF_DATASET_SPECIFIC_A1_PREFLIGHT",
    "GSE232572_CORRECTED_A1_REPLAY",
    "GSE113849_DESIGNED_SNV_TRUE_A2_AGGREGATE_PREFLIGHT",
    "GSE269595_CORRECTED_ROLE_ADJUDICATION_SUCCESSOR",
    "GSE295080_INDEPENDENCE_OVERLAP_ADJUDICATION_ONLY",
]
DATASET_IDS = [
    "GSE217518",
    "ENCSR854RUF",
    "GSE232572",
    "GSE113849",
    "GSE269595",
    "GSE295080",
]
ZERO_CONTRIBUTION = {
    "ordinary": 0,
    "a1": 0,
    "true_a2": 0,
    "canonical_records": 0,
}
DEC027_AUTHORITY = {
    "decision_id": "V3-DEC-027",
    "status": "FROZEN_USER_AUTHORIZED_RESCUE_PREFLIGHT_ONLY_NO_AUTOMATIC_PROMOTION",
    "authority_sync_executes_preflight": False,
    "preflight_status_after_sync": "AUTHORIZED_NOT_RUN_IN_RUNTIME",
    "ordered_routes_exactly": ROUTES,
    "all_six_terminal_reports_required_before_stop_rule_evaluation": True,
    "ordinary_public_data_only": True,
    "aggregate_output_only": True,
    "private_or_sealed_access_allowed": False,
    "row_member_sequence_effect_standard_error_or_split_output_allowed": False,
    "rows_or_replicates_may_count_as_independent_studies": False,
    "post_dedup_power_analysis_unit": "INDEPENDENT_SOURCE_GROUP",
    "prefrozen_power_rule": {
        "alternative_spearman_rho": 0.25,
        "alpha_two_sided": 0.05,
        "target_power_minimum": 0.8,
        "confidence_level": 0.95,
        "maximum_full_ci_width": 0.3,
        "required_effective_n_reference": 156,
    },
    "stop_rule_evaluated_by_sync": False,
    "conditional_successor_activated_by_sync": False,
    "full_route_a_target_retained": {"ordinary": 3, "a1": 2, "true_a2": 1},
    "gate_threshold_relaxation_authorized": False,
    "credit_may_be_inferred_from_preflight_status": False,
    "separate_evidence_based_promotion_authority_required": True,
    "qualification_allowed": False,
    "dataset_role_assignment_allowed": False,
    "canonical_materialization_allowed": False,
    "training_allowed": False,
    "gpu_work_allowed": False,
    "model_selection_allowed": False,
    "a7_allowed": False,
    "next_phase_authorized": False,
    "scientific_claim_status": "NOT_ESTABLISHED",
}
RUNTIME_AUTHORITY = {
    "historical_active_authority_commit_policy": "PRESERVE_PREDECESSOR_VALUE_UNCHANGED",
    "active_amendment_decision_ids": ACTIVE_DECISION_IDS,
    "current_contract_authority_scope": (
        "DEC027_BOUNDED_ORDINARY_PUBLIC_AGGREGATE_ONLY_DATA_RESCUE_"
        "PREFLIGHT_AUTHORITY_NO_EXECUTION_NO_PROMOTION"
    ),
}
FROZEN_OUTER_TRUTH = {
    "current_qualified_counts": {
        "ordinary": 1,
        "a1": 1,
        "true_a2": 0,
        "canonical_records": 6547,
    },
    "candidate_contributions": {
        dataset: copy.deepcopy(ZERO_CONTRIBUTION) for dataset in DATASET_IDS
    },
    "run_status": "IN_PROGRESS",
    "evidence_status": "SCRATCH_ROUTE_QUALIFIED_GLOBAL_PHASE_INCOMPLETE",
    "gate_status": "A1_PHASE_INCOMPLETE_GLOBAL_REQUIREMENTS",
    "a1_complete": False,
    "qualified": False,
    "training_started": False,
    "training_allowed": False,
    "training_authorized": False,
    "gpu_work_started": False,
    "gpu_work_allowed": False,
    "model_selection_allowed": False,
    "a7_allowed": False,
    "next_phase_authorized": False,
    "scientific_claim_status": "NOT_ESTABLISHED",
}
ACCESS_BOUNDARY = {
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
    "restricted_or_sealed_path_accessed": False,
    "gse246381_contact": False,
}
PUBLICATION_POLICY = {
    "registered_artifacts_remain_empty_for_this_sync": True,
    "predecessor_snapshots_are_immutable_runtime_outputs": True,
    "sync_record_is_immutable_runtime_output": True,
    "mutables_commit_after_all_immutables": True,
    "mutable_commit_order": list(MUTABLE_NAMES),
    "event_is_last_commit": True,
    "supported_recovery": "EXACT_PUBLICATION_PREFIX_ONLY",
}
PREDECESSOR_IDENTITIES = {
    "STATUS.json": {
        "bytes": 30836,
        "sha256": "1e8417d67a7c9c21d97e8c24ba3a5f5de2a4c55433478d16533bef04ffe080f3",
        "snapshot_name": "STATUS_PRE_DEC027_AUTHORITY_RUNTIME_SYNC_V1.json",
    },
    "RUN_MANIFEST.json": {
        "bytes": 109220,
        "sha256": "de551922c402ea83958a0af83ef70cc44fa25c99567a1b39ee98473b471b0cbe",
        "snapshot_name": "RUN_MANIFEST_PRE_DEC027_AUTHORITY_RUNTIME_SYNC_V1.json",
    },
    "EVENT_LOG.jsonl": {
        "bytes": 144748,
        "sha256": "d886718913f4b153ddd13d2ea0e8661c3a9ef5e551d0d8407e7abb90317312d9",
        "snapshot_name": "EVENT_LOG_PRE_DEC027_AUTHORITY_RUNTIME_SYNC_V1.jsonl",
    },
}
PREDECESSOR_TAIL = {
    "event_id": "A1-EVT-058",
    "decision_id": "V3-DEC-024",
    "bytes": 11657,
    "sha256": "9a7533a27157f6764381a8370c37c635073fa143861a07be1e1ebde1d62f23c4",
}
UNKNOWN_BINDING_FIELDS = (
    "status",
    "implementation_commit",
    "implementation_script_sha256",
    "implementation_test_sha256",
)


class RuntimeSyncError(RuntimeError):
    """The DEC027 authority/runtime contract is not satisfied."""


class BindingError(RuntimeSyncError):
    """The authority/runtime/implementation binding is incomplete."""


class AuthorityError(RuntimeSyncError):
    """The repository authority chain is not exact."""


class PredecessorError(RuntimeSyncError):
    """The frozen EVT058 candidate is not the current runtime predecessor."""


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
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
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
    return [load_json(line, label=label) for line in payload.splitlines()]


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
    if UNKNOWN in values:
        raise BindingError("implementation binding is partially known")
    if binding.get("status") == BOUND:
        _expect_hex(binding.get("implementation_commit"), HEX40, label="I commit")
        _expect_hex(
            binding.get("implementation_script_sha256"), HEX64, label="I script SHA"
        )
        _expect_hex(
            binding.get("implementation_test_sha256"), HEX64, label="I test SHA"
        )
        return "BOUND"
    raise BindingError("implementation binding is partially known")


def _authority_binding_state(authority: Mapping[str, Any]) -> str:
    files = authority.get("authority_files")
    if not isinstance(files, list) or len(files) != 12:
        raise BindingError("authority exact12 file identity closure differs")
    if authority.get("authority_binding_status") != "FROZEN_BOUND_EXACT12":
        raise BindingError("authority exact12 is not BOUND")
    _expect(authority.get("authority_commit"), AUTHORITY_COMMIT, label="A commit")
    _expect(authority.get("authority_expected_parent"), AUTHORITY_PARENT, label="A parent")
    _expect(files, AUTHORITY_FILES, label="A exact12 identities")
    return "BOUND"


def _predecessor_implementation_binding_state(
    predecessor: Mapping[str, Any],
) -> str:
    _expect_keys(
        predecessor,
        {
            "status",
            "implementation_commit",
            "implementation_expected_parent",
            "implementation_exact_changed_paths",
            "implementation_files",
        },
        label="frozen I1 binding",
    )
    _expect(predecessor, FROZEN_I1_BINDING, label="frozen I1 binding")
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
        "predecessor_binding_status": "FROZEN_BOUND_EVT058",
        "fresh_production_validation_required": True,
        "predecessor_event_id": "A1-EVT-058",
        "predecessor_event_count": 58,
        "successor_event_id": "A1-EVT-059",
        "successor_event_count": 59,
        "predecessor_manifest_output_count": 252,
        "successor_manifest_output_count": 256,
        "predecessor_manifest_registered_artifact_count": 8,
        "successor_manifest_registered_artifact_count": 8,
        "sync_name": "A1_DEC027_AUTHORITY_RUNTIME_SYNC_V1.json",
        "output_delta_count": 4,
        "mutable_publish_order": list(MUTABLE_NAMES),
    }
    for key, expected in fixed.items():
        _expect(runtime.get(key), expected, label=f"runtime {key}")
    _expect(runtime.get("predecessor_mutables"), PREDECESSOR_IDENTITIES, label="CAS")
    _expect(runtime.get("predecessor_tail"), PREDECESSOR_TAIL, label="tail CAS")
    snapshots = runtime["predecessor_mutables"]
    expected_immutables = [
        snapshots[name]["snapshot_name"] for name in MUTABLE_NAMES
    ] + [runtime["sync_name"]]
    _expect(runtime.get("immutable_publish_order"), expected_immutables, label="immutables")


def validate_static_config(config: dict[str, Any]) -> None:
    """Validate the closed config without Git, prepared, or runtime I/O."""

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
            "dec027_authority",
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
        "schema_version": "route_a_v3_dec027_authority_runtime_sync.v1",
        "protocol_id": "ROUTE_A_V3_DEC027_AUTHORITY_RUNTIME_SYNC_V1",
        "contract_id": "mrna_xeditflow_route_a_v3",
        "phase_id": "A1",
        "dataset_ids": DATASET_IDS,
        "decision_id": "V3-DEC-027",
        "event_id": "A1-EVT-059",
        "event_name": (
            "DEC027_BOUNDED_DATA_RESCUE_PREFLIGHT_AUTHORITY_REGISTERED_"
            "RUNTIME_GATES_UNCHANGED"
        ),
        "sync_type": "APPEND_ONLY_AUTHORITY_ONLY_REGISTRATION_NO_SCIENTIFIC_STATE_CHANGE",
    }
    _expect({key: config[key] for key in identity}, identity, label="config identity")

    binding = _expect_keys(
        config["implementation_binding"],
        {
            "binding_scheme",
            "frozen_predecessor_implementation",
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
        "APPEND_ONLY_A_EXACT12_THEN_I1_EXACT3_THEN_I2_EXACT3_THEN_B2_CONFIG_ONLY_V2",
        label="binding scheme",
    )
    predecessor_implementation_state = _predecessor_implementation_binding_state(
        binding["frozen_predecessor_implementation"]
    )
    _expect(binding["implementation_script_path"], SCRIPT_REPO_PATH, label="script")
    _expect(binding["implementation_test_path"], TEST_REPO_PATH, label="test")
    _expect(
        binding["unknown_to_bound_scalar_paths"],
        [f"implementation_binding.{field}" for field in UNKNOWN_BINDING_FIELDS],
        label="four binding scalar paths",
    )
    _expect(
        binding["implementation_commit_exact_changed_paths"],
        IMPLEMENTATION_PATHS,
        label="I2 exact3",
    )
    _expect(
        binding["binding_commit_exact_changed_paths"],
        [CONFIG_REPO_PATH],
        label="B2",
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
            "implementation_exact_changed_paths",
            "binding_exact_changed_paths",
        },
        label="repository authority",
    )
    _expect(authority["production_repo_root"], str(PRODUCTION_REPO_ROOT), label="repo")
    _expect(authority["branch"], BRANCH, label="branch")
    _expect(authority["authority_exact_changed_paths"], AUTHORITY_PATHS, label="A exact12")
    _expect(authority["implementation_exact_changed_paths"], IMPLEMENTATION_PATHS, label="I")
    _expect(authority["binding_exact_changed_paths"], [CONFIG_REPO_PATH], label="B")
    authority_state = _authority_binding_state(authority)
    if predecessor_implementation_state != "BOUND":
        raise BindingError("frozen predecessor implementation is not BOUND")
    if implementation_state == "BOUND" and authority_state != "BOUND":
        raise BindingError("BOUND implementation requires BOUND authority")

    _expect(config["dec027_authority"], DEC027_AUTHORITY, label="DEC027 authority")
    _expect(config["runtime_authority"], RUNTIME_AUTHORITY, label="runtime authority")
    _expect(config["registered_artifacts"], [], label="registered artifacts")
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
    if require_bound and _implementation_binding_state(config["implementation_binding"]) != "BOUND":
        raise BindingError("runtime-sync implementation is not BOUND")
    return config, payload


def load_config(
    config_path: Path = PRODUCTION_CONFIG_PATH, *, require_bound: bool = False
) -> dict[str, Any]:
    return _load_config_payload(config_path, require_bound=require_bound)[0]


def normalized_unknown_i2_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the one valid unbound I2 config, independent of disk I2/B2 state."""

    result = copy.deepcopy(dict(config))
    result["implementation_binding"]["frozen_predecessor_implementation"] = (
        copy.deepcopy(FROZEN_I1_BINDING)
    )
    for field in UNKNOWN_BINDING_FIELDS:
        result["implementation_binding"][field] = UNKNOWN
    validate_static_config(result)
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
        repo_root, "diff-tree", "--no-commit-id", "--name-only", "-r", commit
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
    """Prove exact12 A -> exact3 I1 -> exact3 I2 -> config-only B2."""

    validate_static_config(config)
    if _implementation_binding_state(config["implementation_binding"]) != "BOUND":
        raise BindingError("runtime-sync implementation is not BOUND")
    authority = config["repository_authority"]
    binding = config["implementation_binding"]
    repo_root = Path(authority["production_repo_root"])
    if Path(__file__).resolve() != (repo_root / SCRIPT_REPO_PATH).resolve():
        raise AuthorityError("executing producer is not the bound repository script")

    implementation_commit = binding["implementation_commit"]
    head = _run_git(repo_root, "rev-parse", "HEAD").decode().strip()
    upstream = _run_git(repo_root, "rev-parse", "@{upstream}").decode().strip()
    origin = _run_git(
        repo_root, "rev-parse", "--verify", f"refs/remotes/origin/{BRANCH}"
    ).decode().strip()
    _expect(head, upstream, label="HEAD/upstream")
    _expect(head, origin, label="HEAD/origin")
    _expect(
        _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD").decode().strip(),
        BRANCH,
        label="branch",
    )
    _expect(
        _run_git(repo_root, "rev-parse", "--abbrev-ref", "@{upstream}")
        .decode()
        .strip(),
        f"origin/{BRANCH}",
        label="upstream branch",
    )
    if _run_git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise AuthorityError("production worktree or index is dirty")

    _expect(
        _run_git(repo_root, "rev-parse", f"{head}^").decode().strip(),
        implementation_commit,
        label="B2 parent/I2",
    )
    _expect(
        _run_git(repo_root, "rev-parse", f"{implementation_commit}^")
        .decode()
        .strip(),
        I1_COMMIT,
        label="I2 parent/I1",
    )
    _expect(
        _run_git(repo_root, "rev-parse", f"{I1_COMMIT}^").decode().strip(),
        AUTHORITY_COMMIT,
        label="I1 parent/A",
    )
    _expect(
        _run_git(repo_root, "rev-parse", f"{AUTHORITY_COMMIT}^").decode().strip(),
        AUTHORITY_PARENT,
        label="A parent",
    )
    _expect(
        _changed_paths(repo_root, AUTHORITY_COMMIT),
        sorted(AUTHORITY_PATHS),
        label="A exact12",
    )
    _expect(
        _changed_paths(repo_root, I1_COMMIT),
        sorted(IMPLEMENTATION_PATHS),
        label="I1 exact3",
    )
    _expect(
        _changed_paths(repo_root, implementation_commit),
        sorted(IMPLEMENTATION_PATHS),
        label="I2 exact3",
    )
    _expect(_changed_paths(repo_root, head), [CONFIG_REPO_PATH], label="B2 config-only")

    for item in AUTHORITY_FILES:
        relative = item["path"]
        blob = _git_blob(repo_root, AUTHORITY_COMMIT, relative)
        if len(blob) != item["bytes"] or sha256(blob) != item["sha256"]:
            raise AuthorityError("authority exact12 blob identity differs")
        if _git_blob(repo_root, I1_COMMIT, relative) != blob:
            raise AuthorityError("authority blob did not persist through I1")
        if _git_blob(repo_root, implementation_commit, relative) != blob:
            raise AuthorityError("authority blob did not persist through I2")
        if _git_blob(repo_root, head, relative) != blob:
            raise AuthorityError("authority blob did not persist through B2")
        if _read_repo_file(repo_root, relative) != blob:
            raise AuthorityError("working authority file differs from A")

    for item in I1_FILES:
        blob = _git_blob(repo_root, I1_COMMIT, item["path"])
        if len(blob) != item["bytes"] or sha256(blob) != item["sha256"]:
            raise AuthorityError("frozen I1 exact3 blob identity differs")

    i2_config = load_json(
        _git_blob(repo_root, implementation_commit, CONFIG_REPO_PATH),
        label="I2 config",
    )
    _expect(
        i2_config,
        normalized_unknown_i2_config(config),
        label="I2 unknown config",
    )
    script_blob = _git_blob(repo_root, implementation_commit, SCRIPT_REPO_PATH)
    test_blob = _git_blob(repo_root, implementation_commit, TEST_REPO_PATH)
    _expect(
        sha256(script_blob),
        binding["implementation_script_sha256"],
        label="I2 script SHA",
    )
    _expect(
        sha256(test_blob),
        binding["implementation_test_sha256"],
        label="I2 test SHA",
    )
    _expect(
        _git_blob(repo_root, head, CONFIG_REPO_PATH),
        config_payload,
        label="B2 config",
    )
    _expect(
        _git_blob(repo_root, head, SCRIPT_REPO_PATH),
        script_blob,
        label="B2 script",
    )
    _expect(
        _git_blob(repo_root, head, TEST_REPO_PATH),
        test_blob,
        label="B2 test",
    )
    _expect(_read_repo_file(repo_root, CONFIG_REPO_PATH), config_payload, label="working config")
    _expect(_read_repo_file(repo_root, SCRIPT_REPO_PATH), script_blob, label="working script")
    _expect(_read_repo_file(repo_root, TEST_REPO_PATH), test_blob, label="working test")
    return {
        "status": "PASS_EXACT12_A_EXACT3_I1_EXACT3_I2_CONFIG_ONLY_B2",
        "authority_commit": AUTHORITY_COMMIT,
        "predecessor_implementation_commit": I1_COMMIT,
        "implementation_commit": implementation_commit,
        "binding_commit": head,
        "head_commit": head,
        "upstream_head_commit": upstream,
        "origin_branch_head_commit": origin,
        "authority_blob_count": 12,
        "predecessor_implementation_blob_count": 3,
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


def _check_payload_identity(payload: bytes, spec: Mapping[str, Any], *, label: str) -> None:
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
    """Freshly prove exact live EVT058/58/252/8 before prepared writes."""

    if set(payloads) != set(MUTABLE_NAMES):
        raise PredecessorError("runtime mutable member closure differs")
    for name in MUTABLE_NAMES:
        _check_payload_identity(
            payloads[name], config["runtime"]["predecessor_mutables"][name], label=name
        )
    status, manifest, events = _parse_runtime(payloads)
    if len(events) != 58:
        raise PredecessorError("predecessor event count is not 58")
    expected_ids = [f"A1-EVT-{index:03d}" for index in range(1, 59)]
    if [event.get("event_id") for event in events] != expected_ids:
        raise PredecessorError("predecessor event identifiers are not exact 1..58")
    _expect(events[-1].get("decision_id"), "V3-DEC-024", label="tail decision")
    tail = payloads["EVENT_LOG.jsonl"].splitlines(keepends=True)[-1]
    _check_payload_identity(tail, config["runtime"]["predecessor_tail"], label="EVT058 tail")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 252:
        raise PredecessorError("predecessor manifest output count is not 252")
    paths = [item.get("absolute_path") for item in outputs if isinstance(item, dict)]
    if len(paths) != 252 or len(set(paths)) != 252:
        raise PredecessorError("predecessor manifest output paths are not unique")
    _expect(manifest.get("registered_artifact_count"), 8, label="registered count")
    _validate_outer_document(status, label="predecessor STATUS")
    _validate_outer_document(manifest, label="predecessor RUN_MANIFEST")
    _expect_hex(manifest.get("active_authority_commit"), HEX40, label="historical active authority")
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
        raise PredecessorError("EVT059 timestamp must follow EVT058 with an offset")


def _current_contract_authority(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "decision_id": "V3-DEC-027",
        "authority_commit": config["repository_authority"]["authority_commit"],
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
            "record_type": "ROUTE_A_V3_A1_DEC027_AUTHORITY_RUNTIME_SYNC",
            "sync_type": config["sync_type"],
            "contract_id": config["contract_id"],
            "phase_id": "A1",
            "decision_id": "V3-DEC-027",
            "event_id": "A1-EVT-059",
            "recorded_at": recorded_at,
            "predecessor_event_id": "A1-EVT-058",
            "registered_artifacts": [],
            "registered_artifact_count": 0,
            "new_registered_artifact_count": 0,
            "predecessor_snapshot_count": 3,
            "predecessor_snapshot_names": list(snapshots),
            "snapshot_sha256": {name: sha256(payload) for name, payload in snapshots.items()},
            "output_delta_count": 4,
            "manifest_output_count_before": 252,
            "manifest_output_count_after": 256,
            "manifest_registered_artifact_count_before": 8,
            "manifest_registered_artifact_count_after": 8,
            "dec027_authority": copy.deepcopy(config["dec027_authority"]),
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
            "stop_rule_evaluated": False,
            "conditional_successor_activated": False,
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
        "event_id": "A1-EVT-059",
        "at": recorded_at,
        "phase_id": "A1",
        "event": config["event_name"],
        "sync_type": config["sync_type"],
        "decision_id": "V3-DEC-027",
        "predecessor_event_id": "A1-EVT-058",
        "registered_artifacts": [],
        "registered_artifact_count": 0,
        "new_registered_artifact_count": 0,
        "predecessor_snapshot_count": 3,
        "predecessor_snapshot_names": list(_snapshot_names(config).values()),
        "sync_name": config["runtime"]["sync_name"],
        "sync_record_sha256": sync_digest,
        "output_delta_count": 4,
        "manifest_output_count_before": 252,
        "manifest_output_count_after": 256,
        "manifest_registered_artifact_count_before": 8,
        "manifest_registered_artifact_count_after": 8,
        "active_amendment_decision_ids": copy.deepcopy(ACTIVE_DECISION_IDS),
        "current_contract_authority": _current_contract_authority(config),
        "dec027_authority": copy.deepcopy(config["dec027_authority"]),
        "frozen_outer_truth": copy.deepcopy(config["frozen_outer_truth"]),
        "access_boundary": copy.deepcopy(config["access_boundary"]),
        "preflight_executed": False,
        "stop_rule_evaluated": False,
        "conditional_successor_activated": False,
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
            "Registered only V3-DEC-027 bounded ordinary-public aggregate-only "
            "data-rescue preflight authority. No rescue preflight, evidence "
            "registration, stop-rule evaluation, conditional claim-ladder "
            "activation, qualification, promotion, canonical materialization, "
            "training, GPU work, model selection, A7, or next phase occurred. "
            "Qualified counts remain 1/1/0 and canonical records remain 6547."
        ),
    }


def _successor_updates(
    config: Mapping[str, Any], recorded_at: str, sync_digest: str
) -> dict[str, Any]:
    return {
        "active_amendment_decision_ids": copy.deepcopy(ACTIVE_DECISION_IDS),
        "current_contract_authority": _current_contract_authority(config),
        "dec027_authority_runtime_sync_status": "SYNCED_EVT_059",
        "dec027_authority_runtime_sync_recorded_at": recorded_at,
        "dec027_authority_runtime_sync_record_sha256": sync_digest,
        "dec027_authority_runtime_sync_scientific_state_changed": False,
        "dec027_authority_runtime_sync_gate_changed": False,
        "dec027_authority_runtime_sync_qualification_changed": False,
        "dec027_stop_rule_evaluated": False,
        "dec027_conditional_successor_activated": False,
        "dec027_rescue_route_statuses": {
            route: "AUTHORIZED_NOT_RUN_IN_RUNTIME" for route in ROUTES
        },
        "dec027_rescue_candidate_contributions": copy.deepcopy(
            FROZEN_OUTER_TRUTH["candidate_contributions"]
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
            f"A1_{name.replace('.', '_').upper()}_PRE_DEC027_AUTHORITY_RUNTIME_SYNC_SNAPSHOT",
            run_root / snapshots[name],
            predecessor_payloads[name],
        )
        for name in MUTABLE_NAMES
    ]
    records.append(
        _output_record(
            "A1_DEC027_AUTHORITY_RUNTIME_SYNC_V1",
            run_root / config["runtime"]["sync_name"],
            sync_payload,
        )
    )
    return records


def _synthetic_publisher_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "SYNTHETIC_FIXTURE_NOT_PRODUCTION",
        "authority_commit": AUTHORITY_COMMIT,
        "predecessor_implementation_commit": I1_COMMIT,
        "implementation_commit": config["implementation_binding"][
            "implementation_commit"
        ],
        "binding_commit": "SYNTHETIC_FIXTURE_NOT_PRODUCTION",
        "authority_blob_count": 12,
        "predecessor_implementation_blob_count": 3,
        "worktree_and_index_clean": False,
    }


def build_successors(
    config: dict[str, Any],
    predecessor_payloads: Mapping[str, bytes],
    recorded_at: str,
    authority_audit: Mapping[str, Any] | None = None,
    *,
    production: bool = False,
) -> dict[str, bytes]:
    status, manifest, events = validate_predecessor(config, predecessor_payloads)
    _validate_recorded_at(recorded_at, events[-1].get("at"))
    snapshots = _snapshot_names(config)
    snapshot_payloads = {snapshots[name]: predecessor_payloads[name] for name in MUTABLE_NAMES}
    if production and authority_audit is None:
        raise AuthorityError("fresh production publisher authority audit is required")
    audit = (
        dict(authority_audit)
        if authority_audit is not None
        else _synthetic_publisher_audit(config)
    )
    sync_payload = _build_sync_record(
        config,
        recorded_at=recorded_at,
        snapshots=snapshot_payloads,
        historical_active_authority_commit=manifest["active_authority_commit"],
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
        "EVENT_LOG.jsonl": predecessor_payloads["EVENT_LOG.jsonl"] + compact_json_line(event),
    }
    validate_successors(
        config,
        predecessor_payloads,
        successors,
        authority_audit=audit,
        production=production,
    )
    return successors


def validate_successors(
    config: dict[str, Any],
    predecessor_payloads: Mapping[str, bytes],
    successors: Mapping[str, bytes],
    authority_audit: Mapping[str, Any] | None = None,
    *,
    production: bool = False,
) -> None:
    if production and authority_audit is None:
        raise AuthorityError("fresh production publisher authority audit is required")
    audit = (
        dict(authority_audit)
        if authority_audit is not None
        else _synthetic_publisher_audit(config)
    )
    old_status, old_manifest, old_events = validate_predecessor(config, predecessor_payloads)
    snapshots = _snapshot_names(config)
    expected_names = set(MUTABLE_NAMES) | set(snapshots.values()) | {config["runtime"]["sync_name"]}
    if set(successors) != expected_names or len(successors) != 7:
        raise RuntimeSyncError("prepared member closure is not exact seven")
    for mutable, snapshot in snapshots.items():
        if successors[snapshot] != predecessor_payloads[mutable]:
            raise RuntimeSyncError("predecessor snapshot bytes differ")
    status, manifest, events = _parse_runtime({name: successors[name] for name in MUTABLE_NAMES})
    if len(events) != 59 or events[:-1] != old_events:
        raise RuntimeSyncError("EVENT_LOG is not one exact EVT059 append")
    if not successors["EVENT_LOG.jsonl"].startswith(predecessor_payloads["EVENT_LOG.jsonl"]):
        raise RuntimeSyncError("EVENT_LOG predecessor prefix differs")
    event = events[-1]
    _validate_recorded_at(event.get("at"), old_events[-1].get("at"))
    _expect(event.get("event_id"), "A1-EVT-059", label="successor event")
    _expect(event.get("decision_id"), "V3-DEC-027", label="successor decision")
    _expect(event.get("registered_artifacts"), [], label="event artifacts")
    _expect(event.get("new_registered_artifact_count"), 0, label="new artifacts")
    _expect(event.get("preflight_executed"), False, label="preflight")
    _expect(event.get("stop_rule_evaluated"), False, label="stop rule")
    _expect(event.get("conditional_successor_activated"), False, label="successor")
    _expect(event.get("scientific_state_changed"), False, label="science")
    _expect(event.get("qualification_changed"), False, label="qualification")

    sync_payload = successors[config["runtime"]["sync_name"]]
    sync = load_json(sync_payload, label="DEC027 authority runtime sync")
    snapshot_payloads = {
        snapshots[name]: predecessor_payloads[name] for name in MUTABLE_NAMES
    }
    expected_sync_payload = _build_sync_record(
        config,
        recorded_at=event["at"],
        snapshots=snapshot_payloads,
        historical_active_authority_commit=old_manifest["active_authority_commit"],
        authority_audit=audit,
    )
    expected_sync = load_json(
        expected_sync_payload, label="expected DEC027 authority runtime sync"
    )
    _expect(sync, expected_sync, label="sync record structural closure")
    if sync_payload != expected_sync_payload:
        raise RuntimeSyncError("sync record byte closure drift")
    sync_digest = sha256(expected_sync_payload)
    _expect(sync.get("event_id"), "A1-EVT-059", label="sync event")
    _expect(sync.get("decision_id"), "V3-DEC-027", label="sync decision")
    _expect(sync.get("registered_artifacts"), [], label="sync artifacts")
    _expect(sync.get("output_delta_count"), 4, label="sync output delta")
    _expect(sync.get("preflight_executed"), False, label="sync preflight")
    _expect(sync.get("stop_rule_evaluated"), False, label="sync stop rule")
    _expect(sync.get("scientific_state_changed"), False, label="sync science")
    _expect(sync.get("current_contract_authority"), _current_contract_authority(config), label="sync authority")
    _expect(sync.get("frozen_outer_truth"), FROZEN_OUTER_TRUTH, label="sync outer truth")
    _expect(sync.get("access_boundary"), ACCESS_BOUNDARY, label="sync access")

    expected_event = _event_document(
        config, recorded_at=event["at"], sync_digest=sync_digest
    )
    _expect(event, expected_event, label="EVT059 whole-document closure")
    expected_event_log = predecessor_payloads["EVENT_LOG.jsonl"] + compact_json_line(
        expected_event
    )
    if successors["EVENT_LOG.jsonl"] != expected_event_log:
        raise RuntimeSyncError("EVT059 byte closure drift")

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
    _expect(manifest.get("active_authority_commit"), old_manifest.get("active_authority_commit"), label="historical active authority")
    _expect(manifest.get("registered_artifact_count"), 8, label="registered count")
    outputs = manifest["outputs"]
    if len(outputs) != 256 or outputs[:252] != old_manifest["outputs"]:
        raise RuntimeSyncError("manifest ordered 252-to-256 append differs")
    if outputs[252:] != output_delta:
        raise RuntimeSyncError("manifest exact4 output delta differs")
    if len({item.get("absolute_path") for item in outputs}) != 256:
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


def _write_immutable_once(path: Path, payload: bytes) -> str:
    if path.exists():
        if path.read_bytes() != payload:
            raise PublicationError("existing immutable output differs")
        return "EXISTING_EXACT"
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.close(descriptor)
        temporary_path = Path(temporary)
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
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return "CREATED"


def _prepared_members(
    config: Mapping[str, Any],
    predecessor_payloads: Mapping[str, bytes],
    successors: Mapping[str, bytes],
) -> dict[str, bytes]:
    snapshots = _snapshot_names(config)
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
                raise PublicationError("prepared member differs")
        else:
            _write_atomic(target, payload)
    if {item.name for item in prepared.iterdir()} != set(members):
        raise PublicationError("prepared member closure is incomplete")


def _read_prepared(config: Mapping[str, Any], prepared: Path) -> dict[str, bytes]:
    expected = set(_snapshot_names(config).values()) | {config["runtime"]["sync_name"], *MUTABLE_NAMES}
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
    return (
        {name: prepared[snapshots[name]] for name in MUTABLE_NAMES},
        {name: prepared[name] for name in MUTABLE_NAMES},
    )


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
        audit = audit_production_repository_authority(config, payload) if production else None
        return config, audit
    config = copy.deepcopy(config_override)
    validate_static_config(config)
    if _implementation_binding_state(config["implementation_binding"]) != "BOUND":
        raise BindingError("runtime-sync implementation is not BOUND")
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
    config, authority_audit = _context(config_path, production=production, config_override=config_override)
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
            production=production,
        )
    _write_prepared(prepared, _prepared_members(config, predecessor, successors))
    return {
        "status": "PREPARED_NOT_PUBLISHED",
        "event_id": "A1-EVT-059",
        "prepared_directory": str(prepared),
        "prepared_member_count": 7,
        "manifest_output_transition": "252_TO_256",
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
    config, authority_audit = _context(
        config_path, production=production, config_override=config_override
    )
    prepared_path = _prepared_path(prepared_directory, config)
    run_root = run_root_override or Path(config["runtime"]["run_root"])
    if production and run_root_override is not None:
        raise PublicationError("run-root override is forbidden in production")
    prepared = _read_prepared(config, prepared_path)
    predecessor, successor = _split_prepared(config, prepared)
    validate_successors(
        config,
        predecessor,
        prepared,
        authority_audit=authority_audit,
        production=production,
    )
    snapshots = _snapshot_names(config)
    immutable_payloads = {
        **{snapshots[name]: predecessor[name] for name in MUTABLE_NAMES},
        config["runtime"]["sync_name"]: prepared[config["runtime"]["sync_name"]],
    }
    with _locked_run(run_root):
        current = _read_runtime(run_root)
        states = [
            "OLD" if current[name] == predecessor[name] else "NEW" if current[name] == successor[name] else "INVALID"
            for name in MUTABLE_NAMES
        ]
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
            immutable_results[name] = _write_immutable_once(run_root / name, immutable_payloads[name])
        if states == ["NEW", "NEW", "NEW"]:
            return {"status": "PUBLISHED_VERIFIED", "event_id": "A1-EVT-059", "reused": True, "immutable_results": immutable_results}
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
                return {"status": "PUBLISHED_VERIFIED_AFTER_RECHECK", "event_id": "A1-EVT-059", "immutable_results": immutable_results}
            raise PublicationError("EVT059 was not committed; retry the same prepared directory") from exc
        if _read_runtime(run_root) != successor:
            raise PublicationError("EVT059 publication finished non-exactly")
        return {"status": "PUBLISHED_VERIFIED", "event_id": "A1-EVT-059", "reused": False, "immutable_results": immutable_results}


def validate_published(
    *,
    prepared_directory: Path | str,
    config_path: Path = PRODUCTION_CONFIG_PATH,
    production: bool = True,
    config_override: dict[str, Any] | None = None,
    run_root_override: Path | None = None,
) -> dict[str, Any]:
    config, authority_audit = _context(
        config_path, production=production, config_override=config_override
    )
    prepared_path = _prepared_path(prepared_directory, config)
    run_root = run_root_override or Path(config["runtime"]["run_root"])
    if production and run_root_override is not None:
        raise PublicationError("run-root override is forbidden in production")
    prepared = _read_prepared(config, prepared_path)
    predecessor, successor = _split_prepared(config, prepared)
    validate_successors(
        config,
        predecessor,
        prepared,
        authority_audit=authority_audit,
        production=production,
    )
    with _locked_run(run_root):
        if _read_runtime(run_root) != successor:
            raise PublicationError("runtime does not match prepared EVT059")
        for name in config["runtime"]["immutable_publish_order"]:
            if (run_root / name).read_bytes() != prepared[name]:
                raise PublicationError("immutable output does not match prepared")
    return {"status": "PUBLISHED_VERIFIED", "event_id": "A1-EVT-059"}


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
    try:
        if args.command == "prepare":
            result = prepare_runtime_sync(
                prepared_directory=args.prepared_directory, recorded_at=args.recorded_at
            )
        elif args.command == "publish":
            result = publish_prepared(prepared_directory=args.prepared_directory)
        else:
            result = validate_published(prepared_directory=args.prepared_directory)
    except RuntimeSyncError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
