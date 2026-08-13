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
CORE_PLACEHOLDER = "CORE_SHA256_TO_REFRESH"
MUTABLE_NAMES = ("STATUS.json", "RUN_MANIFEST.json", "EVENT_LOG.jsonl")
CONFIG_REPO_PATH = "configs/route_a_v3_dec020_authority_runtime_sync_v1.json"
SCRIPT_REPO_PATH = "scripts/route_a_v3/dec020_authority_runtime_sync.py"
TEST_REPO_PATH = "tests/route_a_v3/test_dec020_authority_runtime_sync.py"
PRODUCTION_CONFIG_PATH = Path(__file__).resolve().parents[2] / CONFIG_REPO_PATH
PRODUCTION_REPO_ROOT = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810"
)
AUTHORITY_COMMIT = "d0611622f304d2d621a35b190922ac593a3b8788"
AUTHORITY_PARENT = "ba6746adeb5dc9b2b41a69d139912006e0f5ad07"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FaultInjector = Callable[[str], None]

AUTHORITY_FILES = [
    {
        "path": "configs/route_a_v3.yaml",
        "bytes": 25075,
        "sha256": "c908ac57b7c9667398f616a0ccf7101b41451b80bf169e768131844d3b63a678",
    },
    {
        "path": "configs/route_a_v3_a1_qualification.json",
        "bytes": 13193,
        "sha256": "ac1ed9e78bf88d916f5599e3a2e75e79df1504c16ba108a12f7e28cfd3da2e20",
    },
    {
        "path": "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec020.yaml",
        "bytes": 6951,
        "sha256": "0cfbe6e35c2c7f3b19756b8aee41dc91b2a8f05b249a5b6e9cacf90185c56026",
    },
    {
        "path": "docs/contracts/supersession_mrna_xeditflow_v1_1_to_route_a_v3.yaml",
        "bytes": 7155,
        "sha256": "d7c0559742a44b4f0b6f8c941e734da52359c2733ba759ec2acd8ca40b07e62d",
    },
    {
        "path": "docs/execution/route_a_v3_a1_interim.yaml",
        "bytes": 157782,
        "sha256": "615cbb768d819f8acddfb6a5e86a59f9da21c342598caea267bf6ef101efe683",
    },
    {
        "path": "docs/execution/route_a_v3_claim_evidence_matrix.yaml",
        "bytes": 14145,
        "sha256": "9f5226ac78dd6c3848ba5ceb42742918de66ec459f951bb845ccaf21958a88f9",
    },
    {
        "path": "docs/execution/route_a_v3_data_role_registry.yaml",
        "bytes": 21122,
        "sha256": "d06bfcfb8d265153a44d270c7bc40e5dd462a5e3bdde631d91519c7d7e394852",
    },
    {
        "path": "docs/execution/route_a_v3_decision_log.yaml",
        "bytes": 23991,
        "sha256": "1332e789758a11687d3bcbbe95e0a5c7e852694e25ed90563d280006d94caced",
    },
    {
        "path": "docs/execution/route_a_v3_registry_manifest.json",
        "bytes": 20125,
        "sha256": "2d6f7166ad60fa7486659069a0e6694a4ea42f6391bd08f6f5e0f5848dd5ea6b",
    },
    {
        "path": "docs/execution/route_a_v3_split_registry.yaml",
        "bytes": 6261,
        "sha256": "52e1146027956e024dd6194ff18862e542e27fff81e8fc6b6d8aeaa972b8259c",
    },
    {
        "path": "docs/execution/route_a_v3_task_registry.yaml",
        "bytes": 13631,
        "sha256": "bf3066a7534041374685e9ebe9ac8c840e53ceec1acbb076a72a758d397c63f2",
    },
    {
        "path": "docs/execution/route_a_v3_task_split_matrix.yaml",
        "bytes": 6169,
        "sha256": "db23e96b6977339237956de57309d04a9e692bf937a8d34427d2e1b6cc150db8",
    },
    {
        "path": "scripts/route_a_v3/validate_a0_bundle.py",
        "bytes": 565642,
        "sha256": "e75ca6fb98b45122e9ee88e028a3ed34d5f40d70f5cd4cd25c9b4e446c26e2cc",
    },
    {
        "path": "tests/route_a_v3/test_a0_integrity_guards.py",
        "bytes": 148631,
        "sha256": "b3a1ea125a2264c88db422f5796aca0fcbafa85b5c5be5a3ed08c53bb9d5846c",
    },
]
AUTHORITY_PATHS = [item["path"] for item in AUTHORITY_FILES]
IMPLEMENTATION_PATHS = [CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH]
ACTIVE_DECISION_IDS = ["V3-DEC-017", "V3-DEC-018", "V3-DEC-019", "V3-DEC-020"]

DEC020_AUTHORITY = {
    "decision_id": "V3-DEC-020",
    "status": "FROZEN_USER_AUTHORIZED_GSE200304_MODEL_INPUT_ROUTE_POLICY",
    "authority_only_not_study_qualification": True,
    "dataset_id": "GSE200304",
    "selected_route": "SCRATCH_ONLY_NO_FOUNDATION_NO_EXTERNAL_LEARNED_INPUTS",
    "selected_route_status": "FROZEN_AUTHORIZED_NOT_YET_ADJUDICATED",
    "retained_foundation_route_status": "RETAINED_FAIL_CURRENT_PROTOCOL",
    "scratch_route_checkpoint_exposure_status": (
        "NOT_APPLICABLE_BY_FROZEN_NO_EXTERNAL_LEARNED_INPUT_ROUTE"
    ),
    "scratch_route_checkpoint_exposure_pass_claimed": False,
    "scratch_route_external_checkpoint_count_allowed": 0,
    "scratch_route_external_learned_input_count_allowed": 0,
    "scratch_route_parameter_initialization": "RANDOM_INITIALIZATION_ONLY",
    "full_prior_analytic_use_attestation_completed": False,
    "decision_or_policy_alone_is_training_evidence": False,
    "authority_sync_qualifies_study": False,
}
CURRENT_CONTRACT_AUTHORITY = {
    "decision_id": "V3-DEC-020",
    "authority_commit": AUTHORITY_COMMIT,
    "authority_expected_parent": AUTHORITY_PARENT,
    "scope": "DEC020_AUTHORITY_ONLY_NO_STUDY_QUALIFICATION",
    "authority_file_count": 14,
}
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
    "gpu_work_started": False,
    "gpu_work_allowed": False,
    "model_selection_allowed": False,
    "next_phase_authorized": False,
    "private_payload_access_allowed": False,
    "sealed_contact_allowed": False,
    "scientific_claim_status": "NOT_ESTABLISHED",
}
RUNTIME_SCIENTIFIC_KEYS = (
    "input_status_counts",
    "unresolved_blockers",
    "ordinary_study_contribution",
    "a1_study_contribution",
    "true_a2_study_contribution",
    "canonical_intervention_record_count",
    "canonical_materialization_allowed",
    "qualified",
    "training_started",
    "training_allowed",
    "training_authorized",
    "model_selection_allowed",
    "next_phase_authorized",
    "scientific_claim_status",
)
OUTER_A1_STATE = {
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
}
ACCESS_BOUNDARY = {
    "registered_artifact_count": 0,
    "registered_artifact_read_count": 0,
    "registered_artifact_body_parse_count": 0,
    "registered_artifact_payload_field_read_count": 0,
    "private_payload_read_count": 0,
    "private_payload_write_count": 0,
    "row_payload_read_count": 0,
    "sequence_payload_read_count": 0,
    "effect_payload_read_count": 0,
    "canonical_materialization_count": 0,
    "training_run_count": 0,
    "gpu_work_count": 0,
    "model_selection_run_count": 0,
    "restricted_or_sealed_path_accessed": False,
    "gse246381_contact": False,
}


class RuntimeSyncError(RuntimeError):
    """The exact authority/runtime contract is not satisfied."""


class BindingError(RuntimeSyncError):
    """The I-to-B implementation binding is incomplete or inconsistent."""


class AuthorityError(RuntimeSyncError):
    """The production A-to-I-to-B repository authority is not exact."""


class PredecessorError(RuntimeSyncError):
    """The frozen EVT049 predecessor candidate is not the live predecessor."""


class PublicationError(RuntimeSyncError):
    """Preparation or append-only publication failed."""


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
        raise RuntimeSyncError(f"{label} drift: expected {expected!r}, observed {actual!r}")


def _expect_keys(value: Any, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise RuntimeSyncError(f"{label} key closure drift")
    return value


def _expect_hex(value: Any, pattern: re.Pattern[str], *, label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise RuntimeSyncError(f"{label} is not lowercase hexadecimal")
    return value


def compiled_core_projection(config: dict[str, Any]) -> dict[str, Any]:
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


def _validate_runtime_shape(config: dict[str, Any]) -> None:
    runtime = _expect_keys(
        config.get("runtime"),
        {
            "run_root",
            "allowed_prepared_root",
            "predecessor_candidate_status",
            "fresh_production_validation_required",
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
    _expect(
        runtime["predecessor_candidate_status"],
        "FROZEN_PREVIOUSLY_VERIFIED_REQUIRES_FRESH_PRODUCTION_VALIDATION_BEFORE_IO",
        label="predecessor candidate status",
    )
    _expect(runtime["fresh_production_validation_required"], True, label="fresh validation")
    for key, expected in (
        ("predecessor_event_id", "A1-EVT-049"),
        ("predecessor_event_count", 49),
        ("successor_event_id", "A1-EVT-050"),
        ("successor_event_count", 50),
        ("predecessor_manifest_output_count", 208),
        ("successor_manifest_output_count", 212),
        ("sync_name", "A1_DEC020_AUTHORITY_RUNTIME_SYNC_V1.json"),
        ("output_delta_count", 4),
    ):
        _expect(runtime.get(key), expected, label=f"runtime {key}")
    _expect(runtime["mutable_publish_order"], list(MUTABLE_NAMES), label="mutable order")
    snapshots = runtime.get("predecessor_mutables")
    if not isinstance(snapshots, dict) or set(snapshots) != set(MUTABLE_NAMES):
        raise RuntimeSyncError("predecessor mutable metadata closure drift")
    expected_identities = {
        "STATUS.json": (
            26346,
            "f14476ebe60e7c70f93b335d675968ed71271184078207eab8a101ddbc818140",
            "STATUS_PRE_DEC020_AUTHORITY_RUNTIME_SYNC_V1.json",
        ),
        "RUN_MANIFEST.json": (
            85950,
            "2eac508243cccbf28f82e35c79a055edf4de5545d96176335d780887b08f34f0",
            "RUN_MANIFEST_PRE_DEC020_AUTHORITY_RUNTIME_SYNC_V1.json",
        ),
        "EVENT_LOG.jsonl": (
            94615,
            "bc3f0973adeb43054f41971931dde6fe87eb22d69dc7273ea2286c847b364ac0",
            "EVENT_LOG_PRE_DEC020_AUTHORITY_RUNTIME_SYNC_V1.jsonl",
        ),
    }
    for name, expected in expected_identities.items():
        spec = _expect_keys(
            snapshots[name], {"bytes", "sha256", "snapshot_name"}, label=f"{name} identity"
        )
        _expect((spec["bytes"], spec["sha256"], spec["snapshot_name"]), expected, label=name)
    _expect(
        runtime["predecessor_tail"],
        {
            "event_id": "A1-EVT-049",
            "decision_id": "V3-DEC-019",
            "bytes": 8030,
            "sha256": "5f251fe3a67a729b89ee621e67899ebf12b56ef553fd49ab9bbaccdb768646fc",
        },
        label="predecessor tail candidate",
    )
    expected_immutable = [snapshots[name]["snapshot_name"] for name in MUTABLE_NAMES] + [
        runtime["sync_name"]
    ]
    _expect(runtime["immutable_publish_order"], expected_immutable, label="immutable order")


def validate_static_config(config: dict[str, Any]) -> None:
    """Validate the closed local core without touching Git, runtime, or prepared paths."""

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
            "dec020_authority",
            "runtime_authority",
            "registered_artifacts",
            "runtime",
            "successor_scientific_state",
            "outer_a1_state",
            "access_boundary",
            "publication_policy",
        },
        label="config root",
    )
    _expect(
        {key: config[key] for key in (
            "schema_version",
            "protocol_id",
            "contract_id",
            "phase_id",
            "dataset_ids",
            "decision_id",
            "event_id",
            "event_name",
            "sync_type",
        )},
        {
            "schema_version": "route_a_v3_dec020_authority_runtime_sync.v1",
            "protocol_id": "ROUTE_A_V3_DEC020_AUTHORITY_RUNTIME_SYNC_V1",
            "contract_id": "mrna_xeditflow_route_a_v3",
            "phase_id": "A1",
            "dataset_ids": ["GSE200304"],
            "decision_id": "V3-DEC-020",
            "event_id": "A1-EVT-050",
            "event_name": "DEC020_AUTHORITY_REGISTERED_RUNTIME_AND_SCIENTIFIC_GATES_UNCHANGED",
            "sync_type": "APPEND_ONLY_AUTHORITY_ONLY_REGISTRATION_NO_SCIENTIFIC_STATE_CHANGE",
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
        label="four scalar paths",
    )
    if not isinstance(binding["activation_rule"], str) or not binding["activation_rule"]:
        raise BindingError("binding activation rule is absent")
    if any(
        binding.get(key) == UNKNOWN
        for key in (
            "status",
            "implementation_commit",
            "implementation_script_sha256",
            "implementation_test_sha256",
        )
    ):
        if not _binding_values_are_unknown(binding):
            raise BindingError("implementation binding is partially known")
    else:
        _expect(binding["status"], "BOUND", label="implementation status")
        _expect_hex(binding["implementation_commit"], HEX40, label="implementation commit")
        _expect_hex(binding["implementation_script_sha256"], HEX64, label="script SHA")
        _expect_hex(binding["implementation_test_sha256"], HEX64, label="test SHA")
    core = binding["compiled_core_sha256"]
    if core != CORE_PLACEHOLDER:
        _expect_hex(core, HEX64, label="compiled core SHA")
        _expect(core, compiled_core_sha256(config), label="compiled core")

    authority = _expect_keys(
        config["repository_authority"],
        {
            "production_repo_root",
            "branch",
            "authority_commit",
            "authority_expected_parent",
            "authority_snapshot_status",
            "authority_exact_changed_paths",
            "authority_files",
            "implementation_exact_changed_paths",
            "binding_exact_changed_paths",
        },
        label="repository authority",
    )
    _expect(authority["production_repo_root"], str(PRODUCTION_REPO_ROOT), label="repo root")
    _expect(authority["branch"], "routea-v3-a1-20260810", label="branch")
    _expect(authority["authority_commit"], AUTHORITY_COMMIT, label="A commit")
    _expect(authority["authority_expected_parent"], AUTHORITY_PARENT, label="A parent")
    _expect(
        authority["authority_snapshot_status"],
        "FROZEN_CONFIRMED_A_GIT_BLOB_IDENTITIES_REVALIDATE_IN_PRODUCTION",
        label="A snapshot status",
    )
    _expect(authority["authority_exact_changed_paths"], AUTHORITY_PATHS, label="A exact14")
    _expect(authority["authority_files"], AUTHORITY_FILES, label="A exact14 identities")
    _expect(authority["implementation_exact_changed_paths"], IMPLEMENTATION_PATHS, label="I exact3")
    _expect(authority["binding_exact_changed_paths"], [CONFIG_REPO_PATH], label="B exact1")

    _expect(config["dec020_authority"], DEC020_AUTHORITY, label="DEC020 authority")
    _expect(
        config["runtime_authority"],
        {
            "historical_active_authority_commit_policy": (
                "PRESERVE_PREDECESSOR_VALUE_UNCHANGED"
            ),
            "active_amendment_decision_ids": ACTIVE_DECISION_IDS,
            "current_contract_authority": CURRENT_CONTRACT_AUTHORITY,
        },
        label="runtime authority",
    )
    _expect(config["registered_artifacts"], [], label="registered artifacts")
    _validate_runtime_shape(config)
    _expect(
        config["successor_scientific_state"],
        SUCCESSOR_SCIENTIFIC_STATE,
        label="scientific state",
    )
    _expect(config["outer_a1_state"], OUTER_A1_STATE, label="outer A1 state")
    _expect(config["access_boundary"], ACCESS_BOUNDARY, label="access boundary")
    _expect(
        config["publication_policy"],
        {
            "registered_artifacts_remain_empty": True,
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
    if _binding_values_are_unknown(config["implementation_binding"]):
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


def expected_unknown_i_config(bound_config: dict[str, Any]) -> dict[str, Any]:
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
        _run_git(repo_root, "diff-tree", "--no-commit-id", "--name-only", "-r", commit)
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
    """Prove clean exact14 A -> exact3 UNKNOWN I -> config-only B before runtime I/O."""

    validate_bound_config(config)
    authority = config["repository_authority"]
    binding = config["implementation_binding"]
    repo_root = Path(authority["production_repo_root"])
    branch = authority["branch"]
    authority_commit = authority["authority_commit"]
    implementation_commit = binding["implementation_commit"]

    head = _run_git(repo_root, "rev-parse", "HEAD").decode().strip()
    current_branch = _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD").decode().strip()
    upstream_branch = _run_git(
        repo_root, "rev-parse", "--abbrev-ref", "@{upstream}"
    ).decode().strip()
    upstream_head = _run_git(repo_root, "rev-parse", "@{upstream}").decode().strip()
    origin_head = _run_git(
        repo_root, "rev-parse", "--verify", f"refs/remotes/origin/{branch}"
    ).decode().strip()
    _expect(current_branch, branch, label="production branch")
    _expect(upstream_branch, f"origin/{branch}", label="production upstream branch")
    _expect(head, upstream_head, label="HEAD/upstream")
    _expect(head, origin_head, label="HEAD/origin")
    if _run_git(repo_root, "status", "--porcelain=v1", "--untracked-files=all") != b"":
        raise AuthorityError("production worktree or index is dirty")

    binding_parent = _run_git(repo_root, "rev-parse", f"{head}^").decode().strip()
    implementation_parent = _run_git(
        repo_root, "rev-parse", f"{implementation_commit}^"
    ).decode().strip()
    authority_parent = _run_git(repo_root, "rev-parse", f"{authority_commit}^").decode().strip()
    _expect(binding_parent, implementation_commit, label="B parent/I")
    _expect(implementation_parent, authority_commit, label="I parent/A")
    _expect(authority_parent, AUTHORITY_PARENT, label="A parent")
    _expect(
        _changed_paths(repo_root, authority_commit),
        sorted(AUTHORITY_PATHS),
        label="A exact14 changed paths",
    )
    _expect(
        _changed_paths(repo_root, implementation_commit),
        sorted(IMPLEMENTATION_PATHS),
        label="I exact3 changed paths",
    )
    _expect(
        _changed_paths(repo_root, head),
        [CONFIG_REPO_PATH],
        label="B config-only changed paths",
    )

    for item in AUTHORITY_FILES:
        path = item["path"]
        for payload, label in (
            (_git_blob(repo_root, authority_commit, path), "A"),
            (_git_blob(repo_root, head, path), "B"),
            (_read_repo_file(repo_root, path), "worktree"),
        ):
            if len(payload) != item["bytes"] or sha256(payload) != item["sha256"]:
                raise AuthorityError(f"{label} DEC020 authority blob drift: {path}")

    expected_i_payload = json_bytes(expected_unknown_i_config(config))
    if _git_blob(repo_root, implementation_commit, CONFIG_REPO_PATH) != expected_i_payload:
        raise AuthorityError("I config is not the exact four-scalar UNKNOWN form")
    if _git_blob(repo_root, head, CONFIG_REPO_PATH) != config_payload:
        raise AuthorityError("B config Git blob differs from supplied production config")
    if _read_repo_file(repo_root, CONFIG_REPO_PATH) != config_payload:
        raise AuthorityError("worktree config differs from supplied production config")

    for path, digest in (
        (SCRIPT_REPO_PATH, binding["implementation_script_sha256"]),
        (TEST_REPO_PATH, binding["implementation_test_sha256"]),
    ):
        for payload, label in (
            (_git_blob(repo_root, implementation_commit, path), "I"),
            (_git_blob(repo_root, head, path), "B"),
            (_read_repo_file(repo_root, path), "worktree"),
        ):
            if sha256(payload) != digest:
                raise AuthorityError(f"{label} implementation blob drift: {path}")

    _expect(
        binding["compiled_core_sha256"],
        compiled_core_sha256(config),
        label="production compiled core",
    )
    return {
        "status": "PASS_EXACT_A_TO_I_TO_CONFIG_ONLY_B",
        "authority_commit": authority_commit,
        "implementation_commit": implementation_commit,
        "binding_commit": head,
        "head_commit": head,
        "upstream_head_commit": upstream_head,
        "origin_branch_head_commit": origin_head,
        "authority_blob_count": 14,
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


def _snapshot_names(config: dict[str, Any]) -> dict[str, str]:
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


def validate_predecessor(
    config: dict[str, Any], payloads: Mapping[str, bytes]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Freshly prove the frozen EVT049 candidate before any prepared write."""

    runtime = config["runtime"]
    for name in MUTABLE_NAMES:
        _check_payload_identity(payloads[name], runtime["predecessor_mutables"][name], label=name)
    status, manifest, events = _parse_runtime(payloads)
    if len(events) != 49 or not events:
        raise PredecessorError("predecessor event count is not 49")
    _expect(events[-1].get("event_id"), "A1-EVT-049", label="predecessor tail event")
    _expect(events[-1].get("decision_id"), "V3-DEC-019", label="predecessor tail decision")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 208:
        raise PredecessorError("predecessor manifest output count is not 208")
    scientific = config["successor_scientific_state"]
    for document, label in ((status, "STATUS"), (manifest, "RUN_MANIFEST")):
        for key in RUNTIME_SCIENTIFIC_KEYS:
            _expect(document.get(key), scientific[key], label=f"predecessor {label}.{key}")
        _expect(document.get("claim_status"), "NOT_ESTABLISHED", label=f"{label}.claim_status")
        _expect(document.get("canonical_record_count"), 0, label=f"{label}.canonical_record_count")
    for key, expected in config["outer_a1_state"].items():
        _expect(status.get(key), expected, label=f"predecessor STATUS.{key}")
    historical_active = manifest.get("active_authority_commit")
    _expect_hex(historical_active, HEX40, label="historical active_authority_commit")
    tail_payload = payloads["EVENT_LOG.jsonl"].splitlines(keepends=True)[-1]
    _check_payload_identity(tail_payload, runtime["predecessor_tail"], label="EVT049 tail")
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
        raise PredecessorError("EVT050 timestamp must follow EVT049 with an explicit offset")


def _output_record(
    artifact_type: str, path: Path, payload: bytes
) -> dict[str, Any]:
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
            "record_type": "ROUTE_A_V3_A1_DEC020_AUTHORITY_RUNTIME_SYNC",
            "sync_type": config["sync_type"],
            "contract_id": config["contract_id"],
            "phase_id": "A1",
            "decision_id": "V3-DEC-020",
            "event_id": "A1-EVT-050",
            "recorded_at": recorded_at,
            "predecessor_event_id": "A1-EVT-049",
            "registered_artifacts": [],
            "registered_artifact_count": 0,
            "predecessor_snapshot_count": 3,
            "predecessor_snapshot_names": list(snapshots),
            "snapshot_sha256": {name: sha256(payload) for name, payload in snapshots.items()},
            "output_delta_count": 4,
            "manifest_output_count_before": 208,
            "manifest_output_count_after": 212,
            "dec020_authority": copy.deepcopy(config["dec020_authority"]),
            "active_amendment_decision_ids": copy.deepcopy(ACTIVE_DECISION_IDS),
            "current_contract_authority": copy.deepcopy(CURRENT_CONTRACT_AUTHORITY),
            "historical_outer_runtime_authority": {
                "active_authority_commit": historical_active_authority_commit,
                "active_authority_commit_rewritten": False,
                "meaning": "HISTORICAL_RUNTIME_AUTHORITY_IDENTITY",
            },
            "runtime_sync_publisher_authority": copy.deepcopy(dict(authority_audit)),
            "successor_scientific_state": copy.deepcopy(config["successor_scientific_state"]),
            "outer_a1_state": copy.deepcopy(config["outer_a1_state"]),
            "access_boundary": copy.deepcopy(config["access_boundary"]),
            "scientific_state_changed": False,
            "evidence_gate_statuses_changed": False,
            "overall_qualification_gate_changed": False,
            "qualification_changed": False,
        }
    )


def _event_document(
    config: dict[str, Any], *, recorded_at: str, sync_digest: str
) -> dict[str, Any]:
    return {
        "event_id": "A1-EVT-050",
        "at": recorded_at,
        "phase_id": "A1",
        "event": config["event_name"],
        "sync_type": config["sync_type"],
        "decision_id": "V3-DEC-020",
        "predecessor_event_id": "A1-EVT-049",
        "registered_artifacts": [],
        "registered_artifact_count": 0,
        "predecessor_snapshot_count": 3,
        "predecessor_snapshot_names": list(_snapshot_names(config).values()),
        "sync_name": config["runtime"]["sync_name"],
        "sync_record_sha256": sync_digest,
        "output_delta_count": 4,
        "manifest_output_count_before": 208,
        "manifest_output_count_after": 212,
        "active_amendment_decision_ids": copy.deepcopy(ACTIVE_DECISION_IDS),
        "current_contract_authority": copy.deepcopy(CURRENT_CONTRACT_AUTHORITY),
        "scientific_state_changed": False,
        "evidence_gate_statuses_changed": False,
        "overall_qualification_gate_changed": False,
        "qualification_changed": False,
        "successor_scientific_state": copy.deepcopy(config["successor_scientific_state"]),
        "outer_a1_state": copy.deepcopy(config["outer_a1_state"]),
        "access_boundary": copy.deepcopy(config["access_boundary"]),
        "training_started": False,
        "training_allowed": False,
        "gpu_work_started": False,
        "gpu_work_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "private_payload_access_allowed": False,
        "sealed_contact_allowed": False,
        "detail": (
            "Registered only the exact DEC020 repository authority. Added three immutable "
            "EVT049 mutable snapshots and one sync record; registered no evidence artifact "
            "and read no private, sealed, row, sequence, or effect payload. GSE200304 remains "
            "7 PASS / 1 UNKNOWN_NOT_ASSERTED with CHECKPOINT_SPECIFIC_EXPOSURE_NOT_PASS, zero "
            "ordinary/A1/true-A2 contribution, zero canonical records, qualified=false, and "
            "training, GPU work, model selection, next phase, private access, sealed contact, "
            "and the scientific claim remain locked."
        ),
    }


def _successor_updates(recorded_at: str, sync_digest: str) -> dict[str, Any]:
    return {
        "active_amendment_decision_ids": copy.deepcopy(ACTIVE_DECISION_IDS),
        "current_contract_authority": copy.deepcopy(CURRENT_CONTRACT_AUTHORITY),
        "dec020_authority_runtime_sync_status": "SYNCED_EVT_050",
        "dec020_authority_runtime_sync_record_sha256": sync_digest,
        "dec020_authority_runtime_sync_scientific_state_changed": False,
        "dec020_authority_runtime_sync_gate_changed": False,
        "dec020_authority_runtime_sync_qualification_changed": False,
    }


def _immutable_output_delta(
    config: dict[str, Any],
    predecessor_payloads: Mapping[str, bytes],
    sync_payload: bytes,
) -> list[dict[str, Any]]:
    run_root = Path(config["runtime"]["run_root"])
    snapshots = _snapshot_names(config)
    records = [
        _output_record(
            f"A1_{name.replace('.', '_').upper()}_PRE_DEC020_AUTHORITY_RUNTIME_SYNC_SNAPSHOT",
            run_root / snapshots[name],
            predecessor_payloads[name],
        )
        for name in MUTABLE_NAMES
    ]
    records.append(
        _output_record(
            "A1_DEC020_AUTHORITY_RUNTIME_SYNC_V1",
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
    snapshot_payloads = {snapshots[name]: predecessor_payloads[name] for name in MUTABLE_NAMES}
    audit = authority_audit or {
        "status": "SYNTHETIC_FIXTURE_NOT_PRODUCTION",
        "authority_commit": AUTHORITY_COMMIT,
        "implementation_commit": config["implementation_binding"]["implementation_commit"],
        "binding_commit": "SYNTHETIC_FIXTURE_NOT_PRODUCTION",
        "authority_blob_count": 14,
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
    updates = _successor_updates(recorded_at, sync_digest)
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
    validate_successors(config, predecessor_payloads, successors)
    return successors


def validate_successors(
    config: dict[str, Any],
    predecessor_payloads: Mapping[str, bytes],
    successors: Mapping[str, bytes],
) -> None:
    old_status, old_manifest, old_events = validate_predecessor(config, predecessor_payloads)
    snapshots = _snapshot_names(config)
    expected_names = set(MUTABLE_NAMES) | set(snapshots.values()) | {
        config["runtime"]["sync_name"]
    }
    if set(successors) != expected_names or len(successors) != 7:
        raise RuntimeSyncError("prepared artifact schema is not exact seven-member closure")
    for mutable, snapshot in snapshots.items():
        if successors[snapshot] != predecessor_payloads[mutable]:
            raise RuntimeSyncError(f"predecessor snapshot byte drift: {mutable}")

    status, manifest, events = _parse_runtime(
        {name: successors[name] for name in MUTABLE_NAMES}
    )
    if (
        len(events) != 50
        or events[:-1] != old_events
        or not successors["EVENT_LOG.jsonl"].startswith(predecessor_payloads["EVENT_LOG.jsonl"])
    ):
        raise RuntimeSyncError("EVENT_LOG is not one exact EVT050 append")
    event = events[-1]
    _expect(event.get("event_id"), "A1-EVT-050", label="successor event")
    _expect(event.get("decision_id"), "V3-DEC-020", label="successor decision")
    _expect(event.get("scientific_state_changed"), False, label="event science change")
    _expect(event.get("registered_artifacts"), [], label="event registered artifacts")
    _expect(event.get("registered_artifact_count"), 0, label="event registered count")

    sync_payload = successors[config["runtime"]["sync_name"]]
    sync_digest = sha256(sync_payload)
    sync = load_json(sync_payload, label="DEC020 authority runtime sync")
    _expect(sync.get("event_id"), "A1-EVT-050", label="sync event")
    _expect(sync.get("decision_id"), "V3-DEC-020", label="sync decision")
    _expect(sync.get("registered_artifacts"), [], label="sync registered artifacts")
    _expect(sync.get("registered_artifact_count"), 0, label="sync registered count")
    _expect(sync.get("scientific_state_changed"), False, label="sync science change")
    _expect(sync.get("output_delta_count"), 4, label="sync output delta")
    _expect(sync.get("current_contract_authority"), CURRENT_CONTRACT_AUTHORITY, label="sync authority")
    _expect(
        sync.get("active_amendment_decision_ids"),
        ACTIVE_DECISION_IDS,
        label="sync active decisions",
    )
    _expect(sync.get("access_boundary"), ACCESS_BOUNDARY, label="sync access boundary")

    updates = _successor_updates(event["at"], sync_digest)
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
        label="historical active_authority_commit preservation",
    )
    _expect(
        sync["historical_outer_runtime_authority"]["active_authority_commit"],
        old_manifest.get("active_authority_commit"),
        label="sync historical runtime authority",
    )
    outputs = manifest["outputs"]
    if len(outputs) != 212 or outputs[:208] != old_manifest["outputs"]:
        raise RuntimeSyncError("manifest ordered 208 to 212 append drift")
    if outputs[208:] != output_delta:
        raise RuntimeSyncError("manifest exact4 output delta drift")
    if len({item.get("absolute_path") for item in outputs}) != 212:
        raise RuntimeSyncError("successor output paths are not unique")
    _expect(
        [Path(item["absolute_path"]).name for item in outputs[-4:]],
        config["runtime"]["immutable_publish_order"],
        label="manifest exact4 names",
    )
    _expect(event.get("sync_record_sha256"), sync_digest, label="event sync digest")
    for document, label in ((status, "STATUS"), (manifest, "manifest")):
        _expect(
            document.get("dec020_authority_runtime_sync_record_sha256"),
            sync_digest,
            label=f"{label} sync digest",
        )
        for key in RUNTIME_SCIENTIFIC_KEYS:
            _expect(
                document.get(key),
                old_status.get(key) if label == "STATUS" else old_manifest.get(key),
                label=f"{label} unchanged {key}",
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
    config: dict[str, Any],
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
                raise PublicationError(f"prepared member differs: {name}")
            continue
        _write_atomic(target, payload)
    if {item.name for item in prepared.iterdir()} != set(members):
        raise PublicationError("prepared member closure is incomplete")


def _read_prepared(config: dict[str, Any], prepared: Path) -> dict[str, bytes]:
    expected = set(_snapshot_names(config).values()) | {
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
        audit = audit_production_repository_authority(config, payload) if production else None
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
    # Production repository authority is audited by _context before either path
    # below is read. The frozen predecessor bytes are then validated before the
    # prepared directory is created or written.
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
            config, predecessor, recorded_at, authority_audit=authority_audit
        )
    _write_prepared(prepared, _prepared_members(config, predecessor, successors))
    return {
        "status": "PREPARED_NOT_PUBLISHED",
        "event_id": "A1-EVT-050",
        "prepared_directory": str(prepared),
        "prepared_member_count": 7,
        "manifest_output_transition": "208_TO_212",
        "new_runtime_output_count": 4,
        "registered_artifact_count": 0,
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
    # As in prepare, production A-I-B authority is proven before prepared or
    # runtime bytes are read.
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
            immutable_results[name] = _write_immutable_once(
                run_root / name, immutable_payloads[name]
            )
        if states == ["NEW", "NEW", "NEW"]:
            return {
                "status": "PUBLISHED_VERIFIED",
                "event_id": "A1-EVT-050",
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
                    "event_id": "A1-EVT-050",
                    "immutable_results": immutable_results,
                }
            raise PublicationError(
                "EVT050 was not committed; retry with the same prepared directory"
            ) from exc
        final = _read_runtime(run_root)
        if final != successor:
            raise PublicationError("EVT050 publication finished with non-exact mutables")
        return {
            "status": "PUBLISHED_VERIFIED",
            "event_id": "A1-EVT-050",
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
        current = _read_runtime(run_root)
        if current != successor:
            raise PublicationError("runtime does not exactly match prepared EVT050 successor")
        for name in config["runtime"]["immutable_publish_order"]:
            if (run_root / name).read_bytes() != prepared[name]:
                raise PublicationError(f"immutable output does not match prepared {name}")
    return {"status": "PUBLISHED_VERIFIED", "event_id": "A1-EVT-050"}


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
