"""Pre-outcome code trust and replayed execution qualification for R3X.

This module deliberately separates three different claims:

* a *preconfigured producer policy* is an external trust anchor whose exact
  SHA-256 must already be known to the caller (and, before activation, bound by
  an authority outside this module);
* a *trusted code snapshot authority* is deterministically derived from that
  policy and from exact source-file bytes, and is published without overwrite
  while the freeze/outcome namespace is still absent; and
* an *execution qualification certificate* is a measurement that is replayed
  by a verifier selected from this module's closed registry.

No public runtime API accepts a caller-provided path/hash mapping as a trust
root or a caller-provided verifier registry.  A timestamp is never used as
ordering evidence.  The pre-outcome witness relies on two explicit external
assumptions: the policy digest was prebound independently, and the commit
namespace is serialized and append-only.  Exact source hashing cannot prove
that reviewed code is free of semantic or covert channels, so every artifact
states that limitation and remains ``training_ready=false`` until integrated
into the immutable activation DAG.

The only execution verifier currently registered is a CPU-only, synthetic
``HistGradientBoostingRegressor`` qualification.  It reads no project data and
does not initialize a GPU.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import re
import secrets
import stat
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import AbstractSet, Any, Callable, Mapping, Sequence

from .artifact_io import ArtifactIOError, canonical_json_bytes
from .protocol import ProtocolHoldError, canonical_sha256


SIGN_ACCURACY_GATE = 0.60

PRODUCER_POLICY_SCHEMA = "r3x_preconfigured_preoutcome_producer_policy_v1"
TRUSTED_CODE_SNAPSHOT_SCHEMA = "r3x_preoutcome_trusted_code_snapshot_authority_v1"
EXECUTION_CERTIFICATE_SCHEMA = "r3x_execution_qualification_certificate_v1"

PRODUCER_POLICY_ROLE = "PRECONFIGURED_PREOUTCOME_PRODUCER_POLICY"
TRUSTED_CODE_SNAPSHOT_ROLE = "PREOUTCOME_TRUSTED_CODE_SNAPSHOT_AUTHORITY"
EXECUTION_CERTIFICATE_ROLE = "EXECUTION_QUALIFICATION_CERTIFICATE"

PRODUCER_POLICY_STATUS = "EXTERNALLY_PREBOUND_PRODUCER_POLICY"
PRODUCER_CAPABILITY_STATUS = (
    "INDEPENDENT_PREOUTCOME_PRODUCER_EVIDENCE_PRECONFIGURED"
)
SNAPSHOT_STATUS = (
    "EXACT_CODE_REPLAYED_PREOUTCOME_UNDER_EXTERNAL_SERIALIZATION_ASSUMPTION"
)
EXECUTION_STATUS = "VERIFIER_REGISTRY_REPLAY_REQUIRED"
EXECUTION_REPLAY_PASSED = "VERIFIER_REGISTRY_REPLAY_PASSED"

HGB_VERIFIER_ID = "SKLEARN_HISTOGRAM_GRADIENT_BOOSTING_CPU_V1"
VERIFIER_REGISTRY_SCHEMA = "r3x_execution_verifier_registry_v1"
HGB_CHILD_ENVIRONMENT_POLICY_SCHEMA = "r3x_hgb_child_environment_sanitization_v1"
HGB_CHILD_ENVIRONMENT_POLICY_ID = "HGB_CPU_MINIMAL_EXPLICIT_ENVIRONMENT_V1"
HGB_CHILD_FIXED_ENVIRONMENT = MappingProxyType({
    "CUDA_VISIBLE_DEVICES": "",
    "LANG": "C",
    "LC_ALL": "C",
    "LOKY_MAX_CPU_COUNT": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
})
# POSIX needs no inherited parent variable because ``sys.executable`` is
# absolute and the child receives an explicit cwd.  Windows may require these
# two OS roots for DLL/system discovery; no other path-like key is inherited.
HGB_CHILD_PLATFORM_INHERIT_ALLOWLIST = MappingProxyType({
    "Windows": ("SYSTEMROOT", "WINDIR"),
})
HGB_CHILD_BLOCKED_PARENT_NAME_TOKENS = (
    "CREDENTIAL",
    "DATA",
    "HOME",
    "KEY",
    "PASSWORD",
    "PATH",
    "PROJECT",
    "PWD",
    "SECRET",
    "TOKEN",
)

TRUST_DIRECTORY_PARTS = ("development", "trust")
PRODUCER_POLICY_RELATIVE_PATH = (
    "development/trust/preconfigured_producer_policy.json"
)
TRUSTED_SNAPSHOT_RELATIVE_PATH = (
    "development/trust/trusted_code_snapshot_authority.json"
)
EXECUTION_CERTIFICATE_RELATIVE_PATH = (
    "development/trust/execution_qualification_certificate.json"
)

MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_CODE_FILE_BYTES = 2 * 1024 * 1024
MAX_CODE_FILES = 4096

HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
SAFE_JSON_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.json$")

ALLOWED_CODE_TOP_LEVEL = frozenset({"scripts", "src", "models"})
ALLOWED_CODE_EXTENSIONS = frozenset({".py", ".pyi"})
FORBIDDEN_CODE_PARTS = frozenset({
    ".git",
    "__pycache__",
    "artifact",
    "artifacts",
    "cache",
    "checkpoint",
    "checkpoints",
    "data",
    "dataset",
    "datasets",
    "output",
    "outputs",
    "prediction",
    "predictions",
    "result",
    "results",
    "run",
    "runs",
    "sealed",
    "restricted",
    "test",
    "tests",
})

PROTECTED_TOKEN = re.compile(r"(?:sealed|restricted|gse246381)", re.IGNORECASE)

PRODUCER_CAPABILITY_KEYS = {
    "source_code_read",
    "source_code_write",
    "trust_authority_write",
    "training_process_control",
    "train_development_outcome_read",
    "internal_test_feature_read",
    "internal_test_outcome_read",
    "outcome_release_artifact_read",
    "sealed_data_read",
}
REQUIRED_PRODUCER_CAPABILITIES = MappingProxyType({
    "source_code_read": True,
    "source_code_write": False,
    "trust_authority_write": True,
    "training_process_control": False,
    "train_development_outcome_read": False,
    "internal_test_feature_read": False,
    "internal_test_outcome_read": False,
    "outcome_release_artifact_read": False,
    "sealed_data_read": False,
})

BOUNDARY_KEYS = {
    "sign_accuracy_gate",
    "sign_gate_change_authorized",
    "ordinary_only",
    "sealed_access",
    "downstream_unlock",
    "training_ready",
}
FROZEN_BOUNDARIES = MappingProxyType({
    "sign_accuracy_gate": SIGN_ACCURACY_GATE,
    "sign_gate_change_authorized": False,
    "ordinary_only": True,
    "sealed_access": False,
    "downstream_unlock": False,
    "training_ready": False,
})

LINEAGE_KEYS = {
    "run_start_sha256",
    "activation_sha256",
    "development_commit_sha256",
    "pre_test_freeze_state",
    "test_prediction_state",
    "ordinary_outcome_release_state",
    "outcome_release_claim_state",
    "commit_namespace_policy",
}
LINEAGE_ABSENCE_LITERALS = MappingProxyType({
    "pre_test_freeze_state": "ABSENT_REQUIRED_AT_SNAPSHOT_PUBLICATION",
    "test_prediction_state": "ABSENT_REQUIRED_AT_SNAPSHOT_PUBLICATION",
    "ordinary_outcome_release_state": "ABSENT_REQUIRED_AT_SNAPSHOT_PUBLICATION",
    "outcome_release_claim_state": "ABSENT_REQUIRED_AT_SNAPSHOT_PUBLICATION",
    "commit_namespace_policy": "EXTERNALLY_SERIALIZED_APPEND_ONLY_NO_DELETE",
})

CODE_ENTRY_KEYS = {"path", "sha256"}
POLICY_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "policy_id",
    "trusted_producer_id",
    "producer_capability_status",
    "producer_capability_evidence_sha256",
    "producer_capabilities",
    "lineage",
    "approved_code_files",
    "approved_code_tree_sha256",
    "trusted_snapshot_relative_path",
    "allowed_execution_verifier_ids",
    "trust_assumptions",
    "trust_limitations",
    "boundaries",
}

SNAPSHOT_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "policy_id",
    "trusted_producer_id",
    "producer_capability_status",
    "producer_capability_evidence_sha256",
    "producer_capabilities",
    "lineage",
    "files",
    "tree_sha256",
    "preoutcome_publication_witness",
    "trust_assumptions",
    "trust_limitations",
    "boundaries",
}
PREOUTCOME_WITNESS_KEYS = {
    "ordering_basis",
    "freeze_and_outcome_namespace_absence_replayed",
    "timestamp_used_as_ordering_evidence",
}

REFERENCE_KEYS = {"relative_path", "sha256"}
QUALIFICATION_SCOPE_KEYS = {
    "method_id",
    "device_scope",
    "data_scope",
    "project_data_accessed",
    "gpu_initialized",
}
EXECUTION_CAPABILITY_BOUNDARY_KEYS = {
    "status",
    "internal_test_feature_authority_mount_status",
    "internal_test_outcome_authority_mount_status",
    "sealed_authority_mount_status",
    "training_capability_status",
}
BLOCKED_EXECUTION_CAPABILITY_BOUNDARY = MappingProxyType({
    "status": "BLOCKED_PENDING_OS_LEVEL_DEVELOPMENT_PROCESS_MOUNT_EVIDENCE",
    "internal_test_feature_authority_mount_status": "NOT_VERIFIED_UNMOUNTED",
    "internal_test_outcome_authority_mount_status": "NOT_VERIFIED_UNMOUNTED",
    "sealed_authority_mount_status": "NOT_VERIFIED_UNMOUNTED",
    "training_capability_status": "SYNTHETIC_EXECUTABILITY_ONLY_NOT_TRAINING_CAPABILITY",
})
EXECUTION_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "certificate_id",
    "policy_id",
    "trusted_producer_id",
    "trusted_code_snapshot_ref",
    "verifier_registry_schema",
    "verifier_id",
    "verifier_implementation_sha256",
    "verifier_module_sha256",
    "qualification_scope",
    "execution_capability_boundary",
    "environment_measurement",
    "environment_measurement_sha256",
    "lineage",
    "replay_status",
    "trust_assumptions",
    "trust_limitations",
    "boundaries",
}

TRUST_ASSUMPTIONS = (
    "ACTIVATION_EXACTLY_PREBINDS_PRODUCER_POLICY_SHA256",
    "CAPABILITY_EVIDENCE_WAS_ISSUED_BY_AN_INDEPENDENT_PREOUTCOME_ACTOR",
    "COMMIT_NAMESPACE_IS_EXTERNALLY_SERIALIZED_AND_APPEND_ONLY",
    "PROCESS_INTEGRITY_PREVENTS_RUNTIME_REGISTRY_REPLACEMENT",
)
TRUST_LIMITATIONS = (
    "CODE_HASH_REPLAY_DOES_NOT_PROVE_ABSENCE_OF_COVERT_OR_SEMANTIC_CHANNELS",
    "FILESYSTEM_ABSENCE_DOES_NOT_PROVE_HISTORICAL_ABSENCE_IF_DELETION_IS_ALLOWED",
    "THIS_MODULE_DOES_NOT_SELF_PROVE_EXTERNAL_PRODUCER_INDEPENDENCE",
    "TRAINING_REMAINS_BLOCKED_UNTIL_ACTIVATION_BINDS_POLICY_SNAPSHOT_AND_CERTIFICATE",
)

COMMIT_STATE_PATHS = (
    "commits/02_PRE_TEST_FREEZE_COMMIT.json",
    "commits/02_TYPED_PRE_TEST_FREEZE_COMMIT.json",
    "commits/03_TEST_PREDICTION_COMMIT.json",
    "commits/03_SEMANTIC_TEST_PREDICTION_COMMIT.json",
    "commits/04_ORDINARY_TEST_OUTCOME_RELEASE_RECORD.json",
    "commits/.ORDINARY_TEST_OUTCOME_RELEASE_CLAIM.json",
    "commits/.ORDINARY_OUTCOME_RELEASE_CAPABILITY_CLAIM.json",
)


class TrustedExecutionError(ProtocolHoldError):
    """Structured fail-closed error for the pre-outcome trust kernel."""


def _hold(code: str, message: str, **details: Any) -> None:
    raise TrustedExecutionError(code, message, details=details or None)


def _strict_keys(value: object, expected: AbstractSet[str], *, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        _hold("HOLD_TRUSTED_EXECUTION_SCHEMA", f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        _hold(
            "HOLD_TRUSTED_EXECUTION_SCHEMA",
            f"{label} keys differ from the exact schema",
            missing=sorted(expected - actual),
            unexpected=sorted(actual - expected),
        )
    return value


def _exact(value: object, expected: object, *, label: str) -> None:
    if isinstance(expected, bool):
        matched = value is expected
    elif isinstance(expected, float):
        matched = (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and float(value) == expected
        )
    else:
        matched = type(value) is type(expected) and value == expected
    if not matched:
        _hold("HOLD_TRUSTED_EXECUTION_SCHEMA", f"{label} differs from frozen policy")


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        _hold("HOLD_TRUSTED_EXECUTION_SCHEMA", f"{label} must be lowercase SHA-256")
    return value


def _safe_id(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 160
        or not SAFE_ID.fullmatch(value)
        or HEX64.fullmatch(value)
        or PROTECTED_TOKEN.search(value)
    ):
        _hold("HOLD_TRUSTED_EXECUTION_SCHEMA", f"{label} is not a safe identifier")
    return value


def _validate_boundaries(value: object) -> dict:
    boundaries = _strict_keys(value, BOUNDARY_KEYS, label="boundaries")
    for key, expected in FROZEN_BOUNDARIES.items():
        _exact(boundaries.get(key), expected, label=f"boundaries.{key}")
    return dict(boundaries)


def _validate_capabilities(value: object) -> dict:
    capabilities = _strict_keys(
        value,
        PRODUCER_CAPABILITY_KEYS,
        label="producer_capabilities",
    )
    for key, expected in REQUIRED_PRODUCER_CAPABILITIES.items():
        _exact(capabilities.get(key), expected, label=f"producer_capabilities.{key}")
    return dict(capabilities)


def _validate_lineage(value: object) -> dict:
    lineage = _strict_keys(value, LINEAGE_KEYS, label="lineage")
    _digest(lineage.get("run_start_sha256"), label="lineage.run_start_sha256")
    _digest(lineage.get("activation_sha256"), label="lineage.activation_sha256")
    development = lineage.get("development_commit_sha256")
    if development is not None:
        _digest(development, label="lineage.development_commit_sha256")
    for key, expected in LINEAGE_ABSENCE_LITERALS.items():
        _exact(lineage.get(key), expected, label=f"lineage.{key}")
    return dict(lineage)


def _validate_code_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        _hold("HOLD_TRUSTED_CODE_PATH", "code path must be non-empty POSIX text")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        _hold("HOLD_TRUSTED_CODE_PATH", "code path must be normalized and relative")
    lowered = tuple(part.lower() for part in relative.parts)
    if not lowered or lowered[0] not in ALLOWED_CODE_TOP_LEVEL:
        _hold("HOLD_TRUSTED_CODE_PATH", "code path has an unsupported top-level directory")
    if any(part in FORBIDDEN_CODE_PARTS for part in lowered):
        _hold("HOLD_TRUSTED_CODE_PATH", "code path enters a prohibited namespace")
    if relative.suffix.lower() not in ALLOWED_CODE_EXTENSIONS:
        _hold("HOLD_TRUSTED_CODE_PATH", "code path has an unsupported extension")
    return relative.as_posix()


def _validate_code_entries(value: object, *, label: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value or len(value) > MAX_CODE_FILES:
        _hold("HOLD_TRUSTED_EXECUTION_SCHEMA", f"{label} must be a bounded non-empty list")
    normalized: list[dict[str, str]] = []
    for index, raw in enumerate(value):
        entry = _strict_keys(raw, CODE_ENTRY_KEYS, label=f"{label}[{index}]")
        normalized.append({
            "path": _validate_code_path(entry.get("path")),
            "sha256": _digest(entry.get("sha256"), label=f"{label}[{index}].sha256"),
        })
    if normalized != sorted(normalized, key=lambda item: item["path"]):
        _hold("HOLD_TRUSTED_EXECUTION_SCHEMA", f"{label} must be path-sorted")
    paths = [entry["path"] for entry in normalized]
    if len(paths) != len(set(paths)):
        _hold("HOLD_TRUSTED_EXECUTION_SCHEMA", f"{label} paths must be unique")
    return normalized


def validate_preconfigured_producer_policy(payload: Mapping) -> dict:
    """Validate one exact external trust anchor without treating it as proof.

    Qualification comes from the caller's independently preconfigured expected
    SHA-256 and producer identity, not from the status strings in ``payload``.
    """

    policy = _strict_keys(payload, POLICY_KEYS, label="producer policy")
    _exact(policy.get("schema_version"), PRODUCER_POLICY_SCHEMA, label="schema_version")
    _exact(policy.get("artifact_role"), PRODUCER_POLICY_ROLE, label="artifact_role")
    _exact(policy.get("status"), PRODUCER_POLICY_STATUS, label="status")
    _safe_id(policy.get("policy_id"), label="policy_id")
    _safe_id(policy.get("trusted_producer_id"), label="trusted_producer_id")
    _exact(
        policy.get("producer_capability_status"),
        PRODUCER_CAPABILITY_STATUS,
        label="producer_capability_status",
    )
    _digest(
        policy.get("producer_capability_evidence_sha256"),
        label="producer_capability_evidence_sha256",
    )
    _validate_capabilities(policy.get("producer_capabilities"))
    _validate_lineage(policy.get("lineage"))
    files = _validate_code_entries(policy.get("approved_code_files"), label="approved_code_files")
    if policy.get("approved_code_tree_sha256") != canonical_sha256(files):
        _hold("HOLD_TRUSTED_CODE_HASH_MISMATCH", "approved code tree digest is stale")
    _exact(
        policy.get("trusted_snapshot_relative_path"),
        TRUSTED_SNAPSHOT_RELATIVE_PATH,
        label="trusted_snapshot_relative_path",
    )
    verifiers = policy.get("allowed_execution_verifier_ids")
    if (
        not isinstance(verifiers, list)
        or verifiers != sorted(set(verifiers))
        or not verifiers
    ):
        _hold(
            "HOLD_TRUSTED_EXECUTION_SCHEMA",
            "allowed_execution_verifier_ids must be sorted and unique",
        )
    for verifier_id in verifiers:
        _safe_id(verifier_id, label="allowed execution verifier")
    if HGB_VERIFIER_ID not in verifiers:
        _hold(
            "HOLD_EXECUTION_VERIFIER_NOT_PRECONFIGURED",
            "paper-grade HGB verifier must be preconfigured",
        )
    if policy.get("trust_assumptions") != list(TRUST_ASSUMPTIONS):
        _hold("HOLD_TRUSTED_EXECUTION_SCHEMA", "trust assumptions must remain explicit")
    if policy.get("trust_limitations") != list(TRUST_LIMITATIONS):
        _hold("HOLD_TRUSTED_EXECUTION_SCHEMA", "trust limitations must remain explicit")
    _validate_boundaries(policy.get("boundaries"))
    return dict(policy)


def _snapshot_payload_from_policy(policy: Mapping) -> dict:
    return {
        "schema_version": TRUSTED_CODE_SNAPSHOT_SCHEMA,
        "artifact_role": TRUSTED_CODE_SNAPSHOT_ROLE,
        "status": SNAPSHOT_STATUS,
        "policy_id": policy["policy_id"],
        "trusted_producer_id": policy["trusted_producer_id"],
        "producer_capability_status": policy["producer_capability_status"],
        "producer_capability_evidence_sha256": policy[
            "producer_capability_evidence_sha256"
        ],
        "producer_capabilities": dict(policy["producer_capabilities"]),
        "lineage": dict(policy["lineage"]),
        "files": [dict(entry) for entry in policy["approved_code_files"]],
        "tree_sha256": policy["approved_code_tree_sha256"],
        "preoutcome_publication_witness": {
            "ordering_basis": (
                "EXCLUSIVE_PUBLICATION_AFTER_NAMESPACE_ABSENCE_REPLAY_"
                "UNDER_EXTERNAL_SERIALIZATION"
            ),
            "freeze_and_outcome_namespace_absence_replayed": True,
            "timestamp_used_as_ordering_evidence": False,
        },
        "trust_assumptions": list(TRUST_ASSUMPTIONS),
        "trust_limitations": list(TRUST_LIMITATIONS),
        "boundaries": dict(FROZEN_BOUNDARIES),
    }


def validate_trusted_code_snapshot_authority(
    payload: Mapping,
    *,
    preconfigured_policy: Mapping,
) -> dict:
    snapshot = _strict_keys(payload, SNAPSHOT_KEYS, label="trusted code snapshot")
    policy = validate_preconfigured_producer_policy(preconfigured_policy)
    expected = _snapshot_payload_from_policy(policy)
    if dict(snapshot) != expected:
        _hold(
            "HOLD_TRUSTED_CODE_POLICY_MISMATCH",
            "snapshot is not the deterministic projection of the preconfigured policy",
        )
    witness = _strict_keys(
        snapshot.get("preoutcome_publication_witness"),
        PREOUTCOME_WITNESS_KEYS,
        label="preoutcome_publication_witness",
    )
    _exact(
        witness.get("freeze_and_outcome_namespace_absence_replayed"),
        True,
        label="preoutcome witness absence replay",
    )
    _exact(
        witness.get("timestamp_used_as_ordering_evidence"),
        False,
        label="preoutcome witness timestamp rule",
    )
    return dict(snapshot)


def _checked_run_root(run_root: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(run_root)))
    try:
        metadata = os.lstat(root)
    except OSError as exc:
        _hold("HOLD_TRUST_NAMESPACE", f"run root is unavailable: {exc}")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        _hold("HOLD_TRUST_NAMESPACE", "run root must be an existing real directory")
    if any(PROTECTED_TOKEN.search(part) for part in root.parts):
        _hold("HOLD_PROTECTED_INPUT", "ordinary-only trust root refuses protected path")
    return root


def _open_absolute_directory_nofollow(path: Path) -> int:
    path = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(os.sep, flags)
    try:
        for part in path.parts[1:]:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            metadata = os.fstat(next_descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(next_descriptor)
                _hold("HOLD_TRUST_NAMESPACE", "path component is not a directory")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _open_directory_at(root_descriptor: int, parts: Sequence[str]) -> int:
    descriptor = os.dup(root_descriptor)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        for part in parts:
            if not part or part in {".", ".."} or os.sep in part:
                _hold("HOLD_TRUST_NAMESPACE", "unsafe directory component")
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            metadata = os.fstat(next_descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(next_descriptor)
                _hold("HOLD_TRUST_NAMESPACE", "namespace component is not a directory")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _verify_pinned_directory_binding(
    root: Path,
    pinned_root_descriptor: int,
    parts: Sequence[str],
    pinned_directory_descriptor: int,
    *,
    label: str,
) -> None:
    """Detect replacement of the absolute root or a traversed directory."""

    fresh_root = -1
    fresh_directory = -1
    try:
        fresh_root = _open_absolute_directory_nofollow(root)
        pinned_root = os.fstat(pinned_root_descriptor)
        replay_root = os.fstat(fresh_root)
        if (pinned_root.st_dev, pinned_root.st_ino) != (
            replay_root.st_dev,
            replay_root.st_ino,
        ):
            _hold(
                "HOLD_UNSUPPORTED_ON_MUTABLE_NAMESPACE",
                f"{label} run-root identity changed during operation",
            )
        fresh_directory = _open_directory_at(fresh_root, parts)
        pinned = os.fstat(pinned_directory_descriptor)
        replayed = os.fstat(fresh_directory)
        if (pinned.st_dev, pinned.st_ino) != (replayed.st_dev, replayed.st_ino):
            _hold(
                "HOLD_UNSUPPORTED_ON_MUTABLE_NAMESPACE",
                f"{label} directory identity changed during operation",
            )
    finally:
        if fresh_directory >= 0:
            os.close(fresh_directory)
        if fresh_root >= 0:
            os.close(fresh_root)


def _relative_trust_artifact(run_root: Path, path: Path) -> tuple[Path, str]:
    root = _checked_run_root(run_root)
    candidate = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        _hold("HOLD_TRUST_NAMESPACE", "trust artifact must be beneath run root")
    if (
        tuple(relative.parts[:-1]) != TRUST_DIRECTORY_PARTS
        or not SAFE_JSON_NAME.fullmatch(relative.name)
        or ".." in relative.name
    ):
        _hold(
            "HOLD_TRUST_NAMESPACE",
            "trust artifact must be one JSON file directly under development/trust",
        )
    return root, relative.name


def _require_fixed_trust_path(
    run_root: Path,
    path: Path,
    expected_relative_path: str,
    *,
    label: str,
) -> None:
    root = _checked_run_root(run_root)
    candidate = Path(os.path.abspath(os.fspath(path)))
    if candidate != root / expected_relative_path:
        _hold(
            "HOLD_TRUST_NAMESPACE",
            f"{label} must use its single preconfigured relative path",
            expected_relative_path=expected_relative_path,
        )


def _read_regular_at(
    directory_descriptor: int,
    name: str,
    expected_sha256: str,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    expected = _digest(expected_sha256, label=f"{label} expected SHA-256")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = -1
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            _hold("HOLD_TRUST_NAMESPACE", f"{label} must be a regular single-link file")
        if before.st_size > maximum_bytes:
            _hold("HOLD_TRUSTED_EXECUTION_SCHEMA", f"{label} is too large")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1 << 20, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                _hold("HOLD_TRUSTED_EXECUTION_SCHEMA", f"{label} is too large")
        after = os.fstat(descriptor)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or after.st_nlink != 1
        ):
            _hold("HOLD_TRUST_NAMESPACE", f"{label} changed during exact read")
        raw = b"".join(chunks)
    except TrustedExecutionError:
        raise
    except OSError as exc:
        _hold("HOLD_TRUST_NAMESPACE", f"{label} cannot be opened safely: {exc}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        _hold(
            "HOLD_TRUSTED_EXECUTION_HASH_MISMATCH",
            f"{label} SHA-256 mismatch",
            expected=expected,
            actual=actual,
        )
    return raw


def _parse_canonical_json(raw: bytes, *, label: str) -> dict:
    def unique_object(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        _hold("HOLD_TRUSTED_EXECUTION_JSON", f"{label} is not strict JSON: {exc}")
    if not isinstance(payload, dict):
        _hold("HOLD_TRUSTED_EXECUTION_JSON", f"{label} must be a JSON object")
    try:
        canonical = canonical_json_bytes(payload)
    except ArtifactIOError as exc:
        _hold("HOLD_TRUSTED_EXECUTION_JSON", str(exc))
    if raw != canonical:
        _hold(
            "HOLD_TRUSTED_EXECUTION_NONCANONICAL",
            f"{label} bytes are not the canonical representation",
        )
    return payload


def _load_trust_json_exact(
    run_root: Path,
    path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> dict:
    root, name = _relative_trust_artifact(run_root, path)
    root_descriptor = _open_absolute_directory_nofollow(root)
    trust_descriptor = -1
    try:
        trust_descriptor = _open_directory_at(root_descriptor, TRUST_DIRECTORY_PARTS)
        raw = _read_regular_at(
            trust_descriptor,
            name,
            expected_sha256,
            maximum_bytes=MAX_JSON_BYTES,
            label=label,
        )
        _verify_pinned_directory_binding(
            root,
            root_descriptor,
            TRUST_DIRECTORY_PARTS,
            trust_descriptor,
            label="trust artifact",
        )
        return _parse_canonical_json(raw, label=label)
    finally:
        if trust_descriptor >= 0:
            os.close(trust_descriptor)
        os.close(root_descriptor)


def load_preconfigured_producer_policy_exact(
    run_root: Path,
    path: Path,
    expected_sha256: str,
    *,
    expected_producer_id: str,
) -> dict:
    """Load the external trust anchor by preconfigured digest and identity."""

    expected_identity = _safe_id(expected_producer_id, label="expected producer id")
    _require_fixed_trust_path(
        run_root,
        path,
        PRODUCER_POLICY_RELATIVE_PATH,
        label="producer policy",
    )
    payload = _load_trust_json_exact(
        run_root,
        path,
        expected_sha256,
        label="preconfigured producer policy",
    )
    policy = validate_preconfigured_producer_policy(payload)
    if policy["trusted_producer_id"] != expected_identity:
        _hold(
            "HOLD_PRODUCER_IDENTITY_MISMATCH",
            "producer identity differs from the preconfigured trust root",
        )
    return policy


def _read_code_exact(run_root: Path, entry: Mapping) -> bytes:
    root = _checked_run_root(run_root)
    relative = PurePosixPath(_validate_code_path(entry.get("path")))
    expected = _digest(entry.get("sha256"), label=f"code {relative}.sha256")
    root_descriptor = _open_absolute_directory_nofollow(root)
    directory_descriptor = -1
    try:
        directory_descriptor = _open_directory_at(root_descriptor, relative.parts[:-1])
        raw = _read_regular_at(
            directory_descriptor,
            relative.parts[-1],
            expected,
            maximum_bytes=MAX_CODE_FILE_BYTES,
            label=f"source code {relative.as_posix()}",
        )
        _verify_pinned_directory_binding(
            root,
            root_descriptor,
            relative.parts[:-1],
            directory_descriptor,
            label=f"source code {relative.as_posix()}",
        )
        return raw
    finally:
        if directory_descriptor >= 0:
            os.close(directory_descriptor)
        os.close(root_descriptor)


def _assert_preoutcome_namespace_open(run_root: Path) -> None:
    root = _checked_run_root(run_root)
    root_descriptor = _open_absolute_directory_nofollow(root)
    try:
        for relative_text in COMMIT_STATE_PATHS:
            relative = PurePosixPath(relative_text)
            try:
                directory = _open_directory_at(root_descriptor, relative.parts[:-1])
            except FileNotFoundError:
                continue
            except OSError as exc:
                if exc.errno == getattr(os, "ENOENT", 2):
                    continue
                _hold("HOLD_PREOUTCOME_LINEAGE", f"commit namespace is unsafe: {exc}")
            try:
                try:
                    os.stat(relative.parts[-1], dir_fd=directory, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                _hold(
                    "HOLD_PREOUTCOME_LINEAGE_ALREADY_ADVANCED",
                    "freeze/prediction/outcome state exists before trust publication",
                    relative_path=relative_text,
                )
            finally:
                os.close(directory)
    finally:
        os.close(root_descriptor)


def _publish_trust_json_exclusive(run_root: Path, path: Path, payload: Mapping) -> str:
    root, name = _relative_trust_artifact(run_root, path)
    try:
        raw = canonical_json_bytes(payload)
    except ArtifactIOError as exc:
        _hold("HOLD_TRUSTED_EXECUTION_JSON", str(exc))
    if len(raw) > MAX_JSON_BYTES:
        _hold("HOLD_TRUSTED_EXECUTION_SCHEMA", "trust artifact is too large")
    expected = hashlib.sha256(raw).hexdigest()
    root_descriptor = _open_absolute_directory_nofollow(root)
    trust_descriptor = -1
    staging_descriptor = -1
    staging_name: str | None = None
    try:
        trust_descriptor = _open_directory_at(root_descriptor, TRUST_DIRECTORY_PARTS)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        for _ in range(128):
            candidate = f".{name}.{secrets.token_hex(16)}.staging"
            try:
                staging_descriptor = os.open(
                    candidate,
                    flags,
                    0o600,
                    dir_fd=trust_descriptor,
                )
                staging_name = candidate
                break
            except FileExistsError:
                continue
        if staging_descriptor < 0 or staging_name is None:
            _hold("HOLD_TRUSTED_EXECUTION_WRITE", "cannot allocate staging artifact")
        view = memoryview(raw)
        written = 0
        while written < len(view):
            count = os.write(staging_descriptor, view[written:])
            if count <= 0:
                _hold("HOLD_TRUSTED_EXECUTION_WRITE", "short artifact write")
            written += count
        os.fsync(staging_descriptor)
        metadata = os.fstat(staging_descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != len(raw)
        ):
            _hold("HOLD_TRUSTED_EXECUTION_WRITE", "staging metadata is invalid")
        os.close(staging_descriptor)
        staging_descriptor = -1
        try:
            os.link(
                staging_name,
                name,
                src_dir_fd=trust_descriptor,
                dst_dir_fd=trust_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            _hold("HOLD_TRUSTED_EXECUTION_WRITE", f"artifact already exists: {path}")
        os.fsync(trust_descriptor)
        os.unlink(staging_name, dir_fd=trust_descriptor)
        staging_name = None
        os.fsync(trust_descriptor)
        replay = _read_regular_at(
            trust_descriptor,
            name,
            expected,
            maximum_bytes=MAX_JSON_BYTES,
            label="published trust artifact",
        )
        if replay != raw:
            _hold("HOLD_TRUSTED_EXECUTION_WRITE", "published bytes changed during replay")
        _verify_pinned_directory_binding(
            root,
            root_descriptor,
            TRUST_DIRECTORY_PARTS,
            trust_descriptor,
            label="published trust artifact",
        )
        return expected
    except TrustedExecutionError:
        raise
    except OSError as exc:
        _hold("HOLD_TRUSTED_EXECUTION_WRITE", f"exclusive publication failed: {exc}")
    finally:
        if staging_descriptor >= 0:
            os.close(staging_descriptor)
        if staging_name is not None and trust_descriptor >= 0:
            try:
                os.unlink(staging_name, dir_fd=trust_descriptor)
                os.fsync(trust_descriptor)
            except OSError:
                pass
        if trust_descriptor >= 0:
            os.close(trust_descriptor)
        os.close(root_descriptor)


def write_trusted_code_snapshot_authority_exclusive(
    run_root: Path,
    *,
    producer_policy_path: Path,
    producer_policy_sha256: str,
    expected_producer_id: str,
) -> tuple[Path, str]:
    """Replay policy-approved code and publish its deterministic snapshot.

    There is intentionally no ``payload`` parameter.  Every snapshot field is
    derived from the exact, preconfigured policy and exact source bytes.
    """

    policy = load_preconfigured_producer_policy_exact(
        run_root,
        producer_policy_path,
        producer_policy_sha256,
        expected_producer_id=expected_producer_id,
    )
    _assert_preoutcome_namespace_open(run_root)
    for entry in policy["approved_code_files"]:
        _read_code_exact(run_root, entry)
    payload = _snapshot_payload_from_policy(policy)
    validate_trusted_code_snapshot_authority(payload, preconfigured_policy=policy)
    output_path = _checked_run_root(run_root) / TRUSTED_SNAPSHOT_RELATIVE_PATH
    digest = _publish_trust_json_exclusive(run_root, output_path, payload)
    # A freeze/outcome transition racing publication fails closed.  Under the
    # externally required serialized append-only namespace, a later transition
    # necessarily follows this successful final replay.
    _assert_preoutcome_namespace_open(run_root)
    return output_path, digest


def load_trusted_code_snapshot_authority_exact(
    run_root: Path,
    path: Path,
    expected_sha256: str,
    *,
    producer_policy_path: Path,
    producer_policy_sha256: str,
    expected_producer_id: str,
) -> dict:
    """Load and replay a code snapshot against a preconfigured trust root."""

    _require_fixed_trust_path(
        run_root,
        path,
        TRUSTED_SNAPSHOT_RELATIVE_PATH,
        label="trusted code snapshot authority",
    )
    policy = load_preconfigured_producer_policy_exact(
        run_root,
        producer_policy_path,
        producer_policy_sha256,
        expected_producer_id=expected_producer_id,
    )
    payload = _load_trust_json_exact(
        run_root,
        path,
        expected_sha256,
        label="trusted code snapshot authority",
    )
    snapshot = validate_trusted_code_snapshot_authority(
        payload,
        preconfigured_policy=policy,
    )
    for entry in snapshot["files"]:
        _read_code_exact(run_root, entry)
    return snapshot


def _hash_text(value: str | None) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def _read_executable_hash() -> tuple[str, str]:
    path = Path(os.path.realpath(sys.executable))
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            _hold("HOLD_EXECUTION_ENVIRONMENT", "Python executable is not a regular single-link file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or after.st_nlink != 1
        ):
            _hold("HOLD_EXECUTION_ENVIRONMENT", "Python executable changed during hashing")
        return str(path), digest.hexdigest()
    except TrustedExecutionError:
        raise
    except OSError as exc:
        _hold("HOLD_EXECUTION_ENVIRONMENT", f"Python executable cannot be hashed: {exc}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _package_fingerprint(distribution_name: str) -> dict:
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        _hold(
            "HOLD_EXECUTION_PACKAGE_MISSING",
            f"required package is unavailable: {distribution_name}",
        )
    metadata_text = distribution.read_text("METADATA")
    record_text = distribution.read_text("RECORD")
    files = sorted(str(item) for item in (distribution.files or ()))
    entry = {
        "distribution": distribution_name,
        "version": distribution.version,
        "metadata_sha256": _hash_text(metadata_text),
        "record_sha256": _hash_text(record_text),
        "installed_file_list_sha256": canonical_sha256(files),
    }
    entry["package_fingerprint_sha256"] = canonical_sha256(entry)
    return entry


def _runtime_measurement() -> dict:
    executable_path, executable_sha256 = _read_executable_hash()
    runtime = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_cache_tag": str(sys.implementation.cache_tag),
        "python_executable_realpath": executable_path,
        "python_executable_sha256": executable_sha256,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
    }
    runtime["runtime_fingerprint_sha256"] = canonical_sha256(runtime)
    return runtime


def _minimal_hgb_child_environment() -> tuple[dict[str, str], dict]:
    """Construct a closed child environment without copying ``os.environ``."""

    system = platform.system()
    inherited_allowlist = list(
        HGB_CHILD_PLATFORM_INHERIT_ALLOWLIST.get(system, ())
    )
    inherited: dict[str, str] = {}
    for name in inherited_allowlist:
        value = os.environ.get(name)
        if value is None:
            continue
        if "\x00" in value:
            _hold(
                "HOLD_EXECUTION_ENVIRONMENT_SANITIZATION",
                f"platform-required child environment value is invalid: {name}",
            )
        inherited[name] = value

    child_environment = dict(inherited)
    child_environment.update(HGB_CHILD_FIXED_ENVIRONMENT)
    passed_key_names = sorted(child_environment)
    unexpected_sensitive_names = sorted(
        name
        for name in passed_key_names
        if any(token in name.upper() for token in HGB_CHILD_BLOCKED_PARENT_NAME_TOKENS)
    )
    if unexpected_sensitive_names:
        _hold(
            "HOLD_EXECUTION_ENVIRONMENT_SANITIZATION",
            "minimal child environment contains a prohibited sensitive/path-like key",
            unexpected=unexpected_sensitive_names,
        )
    policy = {
        "schema_version": HGB_CHILD_ENVIRONMENT_POLICY_SCHEMA,
        "policy_id": HGB_CHILD_ENVIRONMENT_POLICY_ID,
        "parent_environment_copy_mode": "EXPLICIT_KEY_LOOKUP_ONLY_NO_FULL_COPY",
        "child_capture_point": "BEFORE_NUMPY_AND_SKLEARN_IMPORT",
        "platform_system": system,
        "platform_inherit_allowlist": inherited_allowlist,
        "platform_inherited_keys_present": sorted(inherited),
        "fixed_environment": dict(HGB_CHILD_FIXED_ENVIRONMENT),
        "blocked_parent_name_tokens": list(HGB_CHILD_BLOCKED_PARENT_NAME_TOKENS),
        "passed_key_names": passed_key_names,
        "passed_key_names_sha256": canonical_sha256(passed_key_names),
        "unlisted_parent_environment_keys_passed": False,
        "pythonpath_passed": False,
        "sensitive_name_class_passed": False,
    }
    return child_environment, policy


def _measure_hgb_cpu_v1() -> dict:
    """Measure exact CPU environment and run a deterministic synthetic HGB smoke."""

    package_names = ["joblib", "numpy", "scikit-learn", "scipy", "threadpoolctl"]
    packages = [_package_fingerprint(name) for name in package_names]
    # Execute compiled numeric code in a fresh, single-threaded child.  A
    # broken binary stack then produces a structured HOLD instead of crashing
    # the authority-preparation process.  The child receives only the measured
    # explicit environment policy below: no copied PYTHONPATH, secret, project,
    # data, key, or other path-like parent variable.  This does not claim OS
    # mount/network isolation, which remains blocked by the certificate.
    child_source = r'''
import hashlib
import json
import os

received_environment_key_names = sorted(os.environ)
received_environment_key_names_sha256 = hashlib.sha256(
    json.dumps(
        received_environment_key_names,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

x = np.asarray(
    [[float(i), float(i % 5), float((i * i) % 7)] for i in range(64)],
    dtype="<f8",
)
y = np.asarray(
    [0.13 * i - 0.31 * (i % 5) + 0.07 * ((i * i) % 7) for i in range(64)],
    dtype="<f8",
)
kwargs = {
    "learning_rate": 0.1,
    "max_iter": 12,
    "max_leaf_nodes": 7,
    "l2_regularization": 0.0,
    "random_state": 1729,
    "early_stopping": False,
}
predictions = []
for _ in range(2):
    model = HistGradientBoostingRegressor(**kwargs)
    model.fit(x, y)
    prediction = np.asarray(model.predict(x), dtype="<f8")
    predictions.append(prediction.tobytes(order="C"))
output = {
    "method_id": "HISTOGRAM_GRADIENT_BOOSTING",
    "estimator_class": "sklearn.ensemble.HistGradientBoostingRegressor",
    "parameters_sha256": hashlib.sha256(
        json.dumps(kwargs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest(),
    "synthetic_input_sha256": hashlib.sha256(
        x.tobytes(order="C") + y.tobytes(order="C")
    ).hexdigest(),
    "prediction_sha256": hashlib.sha256(predictions[0]).hexdigest(),
    "n_rows": 64,
    "repeated_fit_exact_match": predictions[0] == predictions[1],
    "finite_predictions": bool(np.isfinite(np.frombuffer(predictions[0], dtype="<f8")).all()),
    "received_environment_key_names_before_third_party_import": (
        received_environment_key_names
    ),
    "received_environment_key_names_sha256": (
        received_environment_key_names_sha256
    ),
}
print(json.dumps(output, sort_keys=True, separators=(",", ":")))
'''
    child_environment, child_environment_policy = _minimal_hgb_child_environment()
    try:
        with tempfile.TemporaryDirectory(prefix="r3x_hgb_qualification_") as child_cwd:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", child_source],
                cwd=child_cwd,
                env=child_environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=60,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        _hold("HOLD_EXECUTION_SMOKE", f"isolated HGB smoke could not run: {exc}")
    if completed.returncode != 0:
        _hold(
            "HOLD_EXECUTION_SMOKE",
            "isolated HGB smoke failed",
            returncode=completed.returncode,
        )
    try:
        smoke = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        _hold("HOLD_EXECUTION_SMOKE", f"isolated HGB smoke returned invalid JSON: {exc}")
    expected_smoke_keys = {
        "method_id",
        "estimator_class",
        "parameters_sha256",
        "synthetic_input_sha256",
        "prediction_sha256",
        "n_rows",
        "repeated_fit_exact_match",
        "finite_predictions",
        "received_environment_key_names_before_third_party_import",
        "received_environment_key_names_sha256",
    }
    _strict_keys(smoke, expected_smoke_keys, label="synthetic HGB smoke")
    if (
        smoke.get("method_id") != "HISTOGRAM_GRADIENT_BOOSTING"
        or smoke.get("estimator_class")
        != "sklearn.ensemble.HistGradientBoostingRegressor"
        or smoke.get("n_rows") != 64
        or smoke.get("repeated_fit_exact_match") is not True
        or smoke.get("finite_predictions") is not True
    ):
        _hold("HOLD_EXECUTION_SMOKE", "isolated HGB smoke did not satisfy exact invariants")
    for field in ("parameters_sha256", "synthetic_input_sha256", "prediction_sha256"):
        _digest(smoke.get(field), label=f"synthetic HGB smoke {field}")
    observed_environment_keys = smoke.get(
        "received_environment_key_names_before_third_party_import"
    )
    if (
        not isinstance(observed_environment_keys, list)
        or observed_environment_keys != sorted(set(observed_environment_keys))
        or not all(isinstance(name, str) for name in observed_environment_keys)
    ):
        _hold(
            "HOLD_EXECUTION_ENVIRONMENT_SANITIZATION",
            "isolated HGB child returned an invalid environment-key list",
        )
    observed_environment_sha = _digest(
        smoke.get("received_environment_key_names_sha256"),
        label="synthetic HGB child environment-key list",
    )
    if (
        observed_environment_keys != child_environment_policy["passed_key_names"]
        or observed_environment_sha
        != child_environment_policy["passed_key_names_sha256"]
        or observed_environment_sha != canonical_sha256(observed_environment_keys)
    ):
        _hold(
            "HOLD_EXECUTION_ENVIRONMENT_SANITIZATION",
            "isolated HGB child received keys outside the measured minimal policy",
        )
    measurement = {
        "runtime": _runtime_measurement(),
        "packages": packages,
        "package_set_sha256": canonical_sha256(packages),
        "child_environment_sanitization": child_environment_policy,
        "synthetic_hgb_smoke": smoke,
    }
    measurement["measurement_sha256"] = canonical_sha256(measurement)
    return measurement


def _verifier_implementation_sha256(verifier: Callable[[], dict]) -> str:
    try:
        source = inspect.getsource(verifier).encode("utf-8")
    except (OSError, TypeError) as exc:
        _hold(
            "HOLD_EXECUTION_VERIFIER_INTEGRITY",
            f"registered verifier source cannot be hashed: {exc}",
        )
    # CPython may specialize a live code object after execution.  Hashing a
    # marshalled live code object would therefore make a correct replay appear
    # to drift.  Exact module-byte SHA below plus the stable function source
    # hash binds the implementation without relying on mutable quickened code.
    return hashlib.sha256(source).hexdigest()


def _module_sha256() -> str:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError as exc:
        _hold("HOLD_EXECUTION_VERIFIER_INTEGRITY", f"verifier module cannot be hashed: {exc}")


_EXECUTION_VERIFIER_REGISTRY: Mapping[str, Callable[[], dict]] = MappingProxyType({
    HGB_VERIFIER_ID: _measure_hgb_cpu_v1,
})


def registered_execution_verifier_ids() -> tuple[str, ...]:
    """Return the closed verifier registry identities (never verifier callables)."""

    return tuple(sorted(_EXECUTION_VERIFIER_REGISTRY))


def _registered_verifier(verifier_id: object) -> Callable[[], dict]:
    if not isinstance(verifier_id, str) or verifier_id not in _EXECUTION_VERIFIER_REGISTRY:
        _hold(
            "HOLD_EXECUTION_VERIFIER_NOT_REGISTERED",
            "execution certificate names an unknown verifier",
        )
    return _EXECUTION_VERIFIER_REGISTRY[verifier_id]


def _execution_payload(
    *,
    policy: Mapping,
    snapshot_sha256: str,
    verifier_id: str,
    measurement: Mapping,
) -> dict:
    verifier = _registered_verifier(verifier_id)
    return {
        "schema_version": EXECUTION_CERTIFICATE_SCHEMA,
        "artifact_role": EXECUTION_CERTIFICATE_ROLE,
        "status": EXECUTION_STATUS,
        "certificate_id": "R3X_HGB_CPU_EXECUTION_QUALIFICATION_V1",
        "policy_id": policy["policy_id"],
        "trusted_producer_id": policy["trusted_producer_id"],
        "trusted_code_snapshot_ref": {
            "relative_path": TRUSTED_SNAPSHOT_RELATIVE_PATH,
            "sha256": snapshot_sha256,
        },
        "verifier_registry_schema": VERIFIER_REGISTRY_SCHEMA,
        "verifier_id": verifier_id,
        "verifier_implementation_sha256": _verifier_implementation_sha256(verifier),
        "verifier_module_sha256": _module_sha256(),
        "qualification_scope": {
            "method_id": "HISTOGRAM_GRADIENT_BOOSTING",
            "device_scope": "CPU_ONLY",
            "data_scope": "FIXED_SYNTHETIC_SMOKE_ONLY",
            "project_data_accessed": False,
            "gpu_initialized": False,
        },
        "execution_capability_boundary": dict(BLOCKED_EXECUTION_CAPABILITY_BOUNDARY),
        "environment_measurement": dict(measurement),
        "environment_measurement_sha256": canonical_sha256(measurement),
        "lineage": dict(policy["lineage"]),
        "replay_status": EXECUTION_REPLAY_PASSED,
        "trust_assumptions": list(TRUST_ASSUMPTIONS),
        "trust_limitations": list(TRUST_LIMITATIONS),
        "boundaries": dict(FROZEN_BOUNDARIES),
    }


def validate_execution_qualification_certificate(
    payload: Mapping,
    *,
    preconfigured_policy: Mapping,
    trusted_snapshot_sha256: str,
) -> dict:
    certificate = _strict_keys(payload, EXECUTION_KEYS, label="execution certificate")
    policy = validate_preconfigured_producer_policy(preconfigured_policy)
    expected_literals = {
        "schema_version": EXECUTION_CERTIFICATE_SCHEMA,
        "artifact_role": EXECUTION_CERTIFICATE_ROLE,
        "status": EXECUTION_STATUS,
        "certificate_id": "R3X_HGB_CPU_EXECUTION_QUALIFICATION_V1",
        "policy_id": policy["policy_id"],
        "trusted_producer_id": policy["trusted_producer_id"],
        "verifier_registry_schema": VERIFIER_REGISTRY_SCHEMA,
        "replay_status": EXECUTION_REPLAY_PASSED,
    }
    for key, expected in expected_literals.items():
        _exact(certificate.get(key), expected, label=key)
    verifier_id = certificate.get("verifier_id")
    verifier = _registered_verifier(verifier_id)
    if verifier_id not in policy["allowed_execution_verifier_ids"]:
        _hold(
            "HOLD_EXECUTION_VERIFIER_NOT_PRECONFIGURED",
            "registered verifier is not allowed by the preconfigured producer policy",
        )
    if certificate.get("verifier_implementation_sha256") != _verifier_implementation_sha256(verifier):
        _hold("HOLD_EXECUTION_VERIFIER_INTEGRITY", "verifier implementation changed")
    if certificate.get("verifier_module_sha256") != _module_sha256():
        _hold("HOLD_EXECUTION_VERIFIER_INTEGRITY", "verifier module bytes changed")
    reference = _strict_keys(
        certificate.get("trusted_code_snapshot_ref"),
        REFERENCE_KEYS,
        label="trusted_code_snapshot_ref",
    )
    _exact(
        reference.get("relative_path"),
        TRUSTED_SNAPSHOT_RELATIVE_PATH,
        label="trusted snapshot relative path",
    )
    if reference.get("sha256") != _digest(
        trusted_snapshot_sha256,
        label="trusted snapshot expected SHA-256",
    ):
        _hold("HOLD_EXECUTION_SNAPSHOT_BINDING", "certificate binds a different code snapshot")
    scope = _strict_keys(
        certificate.get("qualification_scope"),
        QUALIFICATION_SCOPE_KEYS,
        label="qualification_scope",
    )
    expected_scope = {
        "method_id": "HISTOGRAM_GRADIENT_BOOSTING",
        "device_scope": "CPU_ONLY",
        "data_scope": "FIXED_SYNTHETIC_SMOKE_ONLY",
        "project_data_accessed": False,
        "gpu_initialized": False,
    }
    if dict(scope) != expected_scope:
        _hold("HOLD_EXECUTION_SCOPE", "execution qualification scope changed")
    capability_boundary = _strict_keys(
        certificate.get("execution_capability_boundary"),
        EXECUTION_CAPABILITY_BOUNDARY_KEYS,
        label="execution_capability_boundary",
    )
    if dict(capability_boundary) != dict(BLOCKED_EXECUTION_CAPABILITY_BOUNDARY):
        _hold(
            "HOLD_EXECUTION_CAPABILITY_ISOLATION",
            "OS-level development-process mount isolation is not independently qualified",
        )
    measurement = certificate.get("environment_measurement")
    if not isinstance(measurement, Mapping):
        _hold("HOLD_TRUSTED_EXECUTION_SCHEMA", "environment_measurement must be an object")
    if certificate.get("environment_measurement_sha256") != canonical_sha256(measurement):
        _hold("HOLD_EXECUTION_ENVIRONMENT", "environment measurement digest is stale")
    if measurement.get("measurement_sha256") != canonical_sha256(
        {key: value for key, value in measurement.items() if key != "measurement_sha256"}
    ):
        _hold("HOLD_EXECUTION_ENVIRONMENT", "nested environment measurement digest is stale")
    if certificate.get("lineage") != policy["lineage"]:
        _hold("HOLD_PREOUTCOME_LINEAGE", "certificate lineage differs from trust policy")
    if certificate.get("trust_assumptions") != list(TRUST_ASSUMPTIONS):
        _hold("HOLD_TRUSTED_EXECUTION_SCHEMA", "certificate trust assumptions changed")
    if certificate.get("trust_limitations") != list(TRUST_LIMITATIONS):
        _hold("HOLD_TRUSTED_EXECUTION_SCHEMA", "certificate trust limitations changed")
    _validate_boundaries(certificate.get("boundaries"))
    return dict(certificate)


def write_execution_qualification_certificate_exclusive(
    run_root: Path,
    *,
    producer_policy_path: Path,
    producer_policy_sha256: str,
    expected_producer_id: str,
    trusted_snapshot_path: Path,
    trusted_snapshot_sha256: str,
    verifier_id: str = HGB_VERIFIER_ID,
) -> tuple[Path, str]:
    """Measure via the closed registry and exclusively publish a certificate."""

    policy = load_preconfigured_producer_policy_exact(
        run_root,
        producer_policy_path,
        producer_policy_sha256,
        expected_producer_id=expected_producer_id,
    )
    load_trusted_code_snapshot_authority_exact(
        run_root,
        trusted_snapshot_path,
        trusted_snapshot_sha256,
        producer_policy_path=producer_policy_path,
        producer_policy_sha256=producer_policy_sha256,
        expected_producer_id=expected_producer_id,
    )
    _assert_preoutcome_namespace_open(run_root)
    if verifier_id not in policy["allowed_execution_verifier_ids"]:
        _hold(
            "HOLD_EXECUTION_VERIFIER_NOT_PRECONFIGURED",
            "requested verifier is absent from the preconfigured policy",
        )
    verifier = _registered_verifier(verifier_id)
    measurement = verifier()
    payload = _execution_payload(
        policy=policy,
        snapshot_sha256=trusted_snapshot_sha256,
        verifier_id=verifier_id,
        measurement=measurement,
    )
    validate_execution_qualification_certificate(
        payload,
        preconfigured_policy=policy,
        trusted_snapshot_sha256=trusted_snapshot_sha256,
    )
    output_path = _checked_run_root(run_root) / EXECUTION_CERTIFICATE_RELATIVE_PATH
    digest = _publish_trust_json_exclusive(run_root, output_path, payload)
    # Publication is not qualification unless the same fixed registry can
    # immediately replay the frozen environment measurement.
    replay_execution_qualification_certificate_exact(
        run_root,
        output_path,
        digest,
        producer_policy_path=producer_policy_path,
        producer_policy_sha256=producer_policy_sha256,
        expected_producer_id=expected_producer_id,
        trusted_snapshot_path=trusted_snapshot_path,
        trusted_snapshot_sha256=trusted_snapshot_sha256,
    )
    _assert_preoutcome_namespace_open(run_root)
    return output_path, digest


def replay_execution_qualification_certificate_exact(
    run_root: Path,
    path: Path,
    expected_sha256: str,
    *,
    producer_policy_path: Path,
    producer_policy_sha256: str,
    expected_producer_id: str,
    trusted_snapshot_path: Path,
    trusted_snapshot_sha256: str,
) -> dict:
    """Exact-load a certificate and re-run its registered verifier."""

    _require_fixed_trust_path(
        run_root,
        path,
        EXECUTION_CERTIFICATE_RELATIVE_PATH,
        label="execution qualification certificate",
    )
    policy = load_preconfigured_producer_policy_exact(
        run_root,
        producer_policy_path,
        producer_policy_sha256,
        expected_producer_id=expected_producer_id,
    )
    load_trusted_code_snapshot_authority_exact(
        run_root,
        trusted_snapshot_path,
        trusted_snapshot_sha256,
        producer_policy_path=producer_policy_path,
        producer_policy_sha256=producer_policy_sha256,
        expected_producer_id=expected_producer_id,
    )
    payload = _load_trust_json_exact(
        run_root,
        path,
        expected_sha256,
        label="execution qualification certificate",
    )
    certificate = validate_execution_qualification_certificate(
        payload,
        preconfigured_policy=policy,
        trusted_snapshot_sha256=trusted_snapshot_sha256,
    )
    verifier = _registered_verifier(certificate["verifier_id"])
    replayed = verifier()
    if replayed != certificate["environment_measurement"]:
        _hold(
            "HOLD_EXECUTION_ENVIRONMENT_DRIFT",
            "registered verifier replay differs from the frozen environment measurement",
            expected_sha256=certificate["environment_measurement_sha256"],
            actual_sha256=canonical_sha256(replayed),
        )
    return certificate


def canonical_preconfigured_producer_policy_bytes(payload: Mapping) -> bytes:
    """Preparation-only canonicalizer; this does not confer trust."""

    try:
        return canonical_json_bytes(validate_preconfigured_producer_policy(payload))
    except ArtifactIOError as exc:
        _hold("HOLD_TRUSTED_EXECUTION_JSON", str(exc))
