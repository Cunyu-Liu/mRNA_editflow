"""Fail-closed capabilities for the two ordinary internal-test stages.

The feature stage receives only a semantically verified typed pre-test-freeze
commit, the feature authority, and the three feature-only row projections.  The
outcome stage receives only a semantically verified test-prediction commit, the
ordinary outcome authority/row set, and a durable claim policy.  Neither
manifest is a substitute for the semantic commit it references.

Manifest validation and exact hashing are preparation properties only.  The
shared artifact I/O helper protects the final file with no-follow, single-link,
exact-hash, and no-replace checks, but it does not descriptor-pin the parent
directory across the complete operation.  Therefore parent-directory swap is
an explicit blocker rather than a property claimed by this module.

At runtime a closed semantic-verifier registry, a closed OS mount/capability
registry, descriptor-pinned namespace I/O, and a strict aggregate evaluator
would all have to replay.  They are not connected here.  The shipped state is
``capability_ready=false`` and every public runtime/data-opening API
unconditionally HOLDs even if a private registry is monkeypatched.  A
test-private seam exercises claim/read crash ordering with synthetic fixtures;
it returns only a byte hash/count and never exports row-level outcome bytes.

No timestamp is used as ordering evidence.  An outcome-release claim is
published durably and exclusively before the first outcome byte is opened.  A
crash after that claim consumes the attempt and cannot be retried.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import AbstractSet, Any, Callable, Mapping, NoReturn

from .artifact_io import (
    ArtifactIOError,
    atomic_write_json_exclusive,
    canonical_json_bytes,
    read_exact_regular_file,
)
from .protocol import ProtocolHoldError
from .stage_row_projections import (
    CONTEXT_SOURCE_CROSSWALK,
    INCLUSION_MANIFEST,
    SOURCE_COMPONENT_SPLIT_MANIFEST,
)


SIGN_ACCURACY_GATE = 0.60

FEATURE_STAGE_SCHEMA = "r3x_internal_test_feature_stage_capability_manifest_v1"
OUTCOME_STAGE_SCHEMA = "r3x_ordinary_outcome_release_stage_capability_manifest_v1"
OUTCOME_CLAIM_SCHEMA = "r3x_ordinary_outcome_release_capability_claim_v1"

FEATURE_STAGE_ROLE = "INTERNAL_TEST_FEATURE_ONLY_STAGE_CAPABILITY"
OUTCOME_STAGE_ROLE = "ORDINARY_OUTCOME_RELEASE_ONLY_STAGE_CAPABILITY"
OUTCOME_CLAIM_ROLE = "ORDINARY_OUTCOME_RELEASE_CLAIM_BEFORE_READ"

PREPARATION_HOLD_STATUS = "EXACT_MANIFEST_PREPARED_CAPABILITY_NOT_READY"
SEMANTIC_VERIFIER_STATUS = "STRICT_SEMANTIC_VERIFIER_REGISTRY_REPLAY_REQUIRED"
OS_CAPABILITY_STATUS = "OS_MOUNT_CAPABILITY_ALLOWLIST_REPLAY_REQUIRED"
NAMESPACE_INTEGRITY_STATUS = (
    "BLOCKED_PARENT_DIRECTORY_NOT_DESCRIPTOR_PINNED_PRODUCTION_RUNTIME_DISABLED"
)

FEATURE_SEMANTIC_VERIFIER_ID = "R3X_STRICT_TYPED_PRE_TEST_FREEZE_V1"
OUTCOME_SEMANTIC_VERIFIER_ID = "R3X_STRICT_SEMANTIC_TEST_PREDICTION_V1"
FEATURE_OS_VERIFIER_ID = "R3X_INTERNAL_TEST_FEATURE_PROCESS_MOUNT_ALLOWLIST_V1"
OUTCOME_OS_VERIFIER_ID = "R3X_OUTCOME_RELEASE_PROCESS_MOUNT_ALLOWLIST_V1"

FEATURE_MANIFEST_RELATIVE_PATH = Path(
    "authority_preparation/internal_test_features/feature_stage_capability_manifest.json"
)
OUTCOME_MANIFEST_RELATIVE_PATH = Path(
    "authority_preparation/internal_test_outcomes/outcome_stage_capability_manifest.json"
)
TYPED_PRE_TEST_FREEZE_RELATIVE_PATH = (
    "commits/02_TYPED_PRE_TEST_FREEZE_COMMIT.json"
)
SEMANTIC_TEST_PREDICTION_RELATIVE_PATH = (
    "commits/03_SEMANTIC_TEST_PREDICTION_COMMIT.json"
)
FEATURE_AUTHORITY_RELATIVE_PATH = (
    "authority_preparation/internal_test_features/internal_test_features.jsonl"
)
OUTCOME_AUTHORITY_RELATIVE_PATH = (
    "authority_preparation/internal_test_outcomes/ordinary_outcomes.jsonl"
)
OUTCOME_CLAIM_RELATIVE_PATH = (
    "commits/.ORDINARY_OUTCOME_RELEASE_CAPABILITY_CLAIM.json"
)
OUTCOME_CLAIM_POLICY = (
    "DURABLE_EXCLUSIVE_CLAIM_BEFORE_FIRST_OUTCOME_BYTE_NO_RETRY_AFTER_CRASH"
)

ROW_AUTHORITIES = (
    INCLUSION_MANIFEST,
    CONTEXT_SOURCE_CROSSWALK,
    SOURCE_COMPONENT_SPLIT_MANIFEST,
)
FEATURE_PROJECTION_RELATIVE_PATHS = MappingProxyType({
    INCLUSION_MANIFEST: (
        "authority_preparation/internal_test_features/"
        "inclusion_manifest_internal_test_feature_projection.json"
    ),
    CONTEXT_SOURCE_CROSSWALK: (
        "authority_preparation/internal_test_features/"
        "context_source_crosswalk_internal_test_feature_projection.json"
    ),
    SOURCE_COMPONENT_SPLIT_MANIFEST: (
        "authority_preparation/internal_test_features/"
        "source_component_split_manifest_internal_test_feature_projection.json"
    ),
})

HEX64 = re.compile(r"^[0-9a-f]{64}$")
PROTECTED_TOKEN = re.compile(r"(?:sealed|restricted|gse246381)", re.IGNORECASE)
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
REF_KEYS = {"relative_path", "sha256"}
BOUNDARY_KEYS = {
    "sign_accuracy_gate",
    "sign_gate_change_authorized",
    "ordinary_only",
    "sealed_access",
    "downstream_unlock",
}
FROZEN_BOUNDARIES = MappingProxyType({
    "sign_accuracy_gate": SIGN_ACCURACY_GATE,
    "sign_gate_change_authorized": False,
    "ordinary_only": True,
    "sealed_access": False,
    "downstream_unlock": False,
})

COMMON_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "semantic_verifier_id",
    "semantic_verifier_status",
    "os_capability_verifier_id",
    "os_capability_status",
    "namespace_integrity_status",
    "capability_ready",
    "boundaries",
}
FEATURE_KEYS = COMMON_KEYS | {
    "typed_pre_test_freeze_commit_ref",
    "feature_authority_ref",
    "feature_row_projection_refs",
    "feature_row_set_sha256",
}
OUTCOME_KEYS = COMMON_KEYS | {
    "semantic_test_prediction_commit_ref",
    "outcome_authority_ref",
    "outcome_row_set_sha256",
    "claim_relative_path",
    "claim_policy",
}

FEATURE_SEMANTIC_EVIDENCE_KEYS = {
    "schema_version",
    "verified_commit_sha256",
    "authorized_feature_authority_sha256",
    "authorized_projection_sha256",
    "authorized_row_set_sha256",
    "semantic_replay_passed",
    "outcome_refs_absent",
    "train_refs_absent",
    "mutable_selector_checkpoint_fields_absent",
}
OUTCOME_SEMANTIC_EVIDENCE_KEYS = {
    "schema_version",
    "verified_commit_sha256",
    "authorized_outcome_authority_sha256",
    "authorized_row_set_sha256",
    "semantic_replay_passed",
    "predictions_frozen_semantically",
    "selector_checkpoint_identity_bound",
    "outcome_not_previously_accessed",
}
FEATURE_OS_EVIDENCE_KEYS = {
    "schema_version",
    "feature_authority_mounted",
    "train_authority_mounted",
    "outcome_authority_mounted",
    "sealed_authority_mounted",
    "mount_allowlist_replay_passed",
}
OUTCOME_OS_EVIDENCE_KEYS = {
    "schema_version",
    "outcome_authority_mounted",
    "train_authority_mounted",
    "feature_authority_mounted",
    "model_config_checkpoint_mounted",
    "sealed_authority_mounted",
    "mount_allowlist_replay_passed",
}

CLAIM_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "outcome_stage_manifest_ref",
    "semantic_test_prediction_commit_ref",
    "outcome_authority_ref",
    "outcome_row_set_sha256",
    "claim_policy",
    "consumed_once",
    "timestamp_used_as_ordering_evidence",
    "boundaries",
}


class TestStageCapabilityError(ProtocolHoldError):
    """Structured HOLD raised before any unqualified stage data access."""

    __test__ = False


def _hold(code: str, message: str, **details: Any) -> NoReturn:
    raise TestStageCapabilityError(code, message, details=details or None)


def _strict_keys(value: object, expected: AbstractSet[str], *, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        _hold("HOLD_TEST_STAGE_SCHEMA", f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        _hold(
            "HOLD_TEST_STAGE_SCHEMA",
            f"{label} keys differ from the exact schema",
            missing=sorted(expected - actual),
            unexpected=sorted(actual - expected),
        )
    return value


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        _hold("HOLD_TEST_STAGE_SCHEMA", f"{label} must be lowercase SHA-256")
    return value


def _exact(value: object, expected: object, *, label: str) -> None:
    if isinstance(expected, bool):
        matched = value is expected
    elif isinstance(expected, float):
        matched = (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) == expected
        )
    else:
        matched = type(value) is type(expected) and value == expected
    if not matched:
        _hold("HOLD_TEST_STAGE_SCHEMA", f"{label} differs from frozen policy")


def _canonical_relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        _hold("HOLD_TEST_STAGE_PATH", f"{label} must be relative POSIX text")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        _hold("HOLD_TEST_STAGE_PATH", f"{label} must be normalized and relative")
    if PROTECTED_TOKEN.search(value):
        _hold("HOLD_PROTECTED_INPUT", f"{label} names a protected namespace")
    return path.as_posix()


def _reference(value: object, *, label: str, expected_path: str) -> dict[str, str]:
    ref = _strict_keys(value, REF_KEYS, label=label)
    relative = _canonical_relative_path(ref.get("relative_path"), label=f"{label}.relative_path")
    if relative != expected_path:
        _hold(
            "HOLD_TEST_STAGE_CROSS_STAGE_REF",
            f"{label} does not use its single stage-specific path",
        )
    return {
        "relative_path": relative,
        "sha256": _digest(ref.get("sha256"), label=f"{label}.sha256"),
    }


def _validate_boundaries(value: object) -> dict:
    boundaries = _strict_keys(value, BOUNDARY_KEYS, label="boundaries")
    for field, expected in FROZEN_BOUNDARIES.items():
        _exact(boundaries.get(field), expected, label=f"boundaries.{field}")
    return dict(boundaries)


def _validate_common(
    payload: Mapping,
    *,
    expected_keys: set[str],
    schema: str,
    role: str,
    semantic_verifier_id: str,
    os_verifier_id: str,
) -> None:
    raw = _strict_keys(payload, expected_keys, label=role)
    expected = {
        "schema_version": schema,
        "artifact_role": role,
        "status": PREPARATION_HOLD_STATUS,
        "semantic_verifier_id": semantic_verifier_id,
        "semantic_verifier_status": SEMANTIC_VERIFIER_STATUS,
        "os_capability_verifier_id": os_verifier_id,
        "os_capability_status": OS_CAPABILITY_STATUS,
        "namespace_integrity_status": NAMESPACE_INTEGRITY_STATUS,
        "capability_ready": False,
    }
    for field, value in expected.items():
        _exact(raw.get(field), value, label=field)
    _validate_boundaries(raw.get("boundaries"))


def validate_feature_stage_manifest(payload: Mapping) -> dict:
    """Validate the feature-only shape; this never opens feature bytes."""

    _validate_common(
        payload,
        expected_keys=FEATURE_KEYS,
        schema=FEATURE_STAGE_SCHEMA,
        role=FEATURE_STAGE_ROLE,
        semantic_verifier_id=FEATURE_SEMANTIC_VERIFIER_ID,
        os_verifier_id=FEATURE_OS_VERIFIER_ID,
    )
    freeze_ref = _reference(
        payload.get("typed_pre_test_freeze_commit_ref"),
        label="typed_pre_test_freeze_commit_ref",
        expected_path=TYPED_PRE_TEST_FREEZE_RELATIVE_PATH,
    )
    feature_ref = _reference(
        payload.get("feature_authority_ref"),
        label="feature_authority_ref",
        expected_path=FEATURE_AUTHORITY_RELATIVE_PATH,
    )
    projection_refs = _strict_keys(
        payload.get("feature_row_projection_refs"),
        set(ROW_AUTHORITIES),
        label="feature_row_projection_refs",
    )
    normalized_projections = {
        name: _reference(
            projection_refs[name],
            label=f"feature_row_projection_refs.{name}",
            expected_path=FEATURE_PROJECTION_RELATIVE_PATHS[name],
        )
        for name in ROW_AUTHORITIES
    }
    row_set = _digest(payload.get("feature_row_set_sha256"), label="feature row set")
    all_hashes = [
        freeze_ref["sha256"],
        feature_ref["sha256"],
        row_set,
        *(normalized_projections[name]["sha256"] for name in ROW_AUTHORITIES),
    ]
    if len(all_hashes) != len(set(all_hashes)):
        _hold("HOLD_TEST_STAGE_ALIAS", "feature-stage commitments must be distinct")
    return dict(payload)


def validate_outcome_stage_manifest(payload: Mapping) -> dict:
    """Validate the outcome-only shape; this never opens outcome bytes."""

    _validate_common(
        payload,
        expected_keys=OUTCOME_KEYS,
        schema=OUTCOME_STAGE_SCHEMA,
        role=OUTCOME_STAGE_ROLE,
        semantic_verifier_id=OUTCOME_SEMANTIC_VERIFIER_ID,
        os_verifier_id=OUTCOME_OS_VERIFIER_ID,
    )
    prediction_ref = _reference(
        payload.get("semantic_test_prediction_commit_ref"),
        label="semantic_test_prediction_commit_ref",
        expected_path=SEMANTIC_TEST_PREDICTION_RELATIVE_PATH,
    )
    outcome_ref = _reference(
        payload.get("outcome_authority_ref"),
        label="outcome_authority_ref",
        expected_path=OUTCOME_AUTHORITY_RELATIVE_PATH,
    )
    row_set = _digest(payload.get("outcome_row_set_sha256"), label="outcome row set")
    if len({prediction_ref["sha256"], outcome_ref["sha256"], row_set}) != 3:
        _hold("HOLD_TEST_STAGE_ALIAS", "outcome-stage commitments must be distinct")
    claim_path = _canonical_relative_path(
        payload.get("claim_relative_path"),
        label="claim_relative_path",
    )
    if claim_path != OUTCOME_CLAIM_RELATIVE_PATH:
        _hold("HOLD_TEST_STAGE_PATH", "outcome claim path is not canonical")
    _exact(payload.get("claim_policy"), OUTCOME_CLAIM_POLICY, label="claim_policy")
    return dict(payload)


def canonical_feature_stage_manifest_bytes(payload: Mapping) -> bytes:
    try:
        return canonical_json_bytes(validate_feature_stage_manifest(payload))
    except ArtifactIOError as exc:
        _hold("HOLD_TEST_STAGE_JSON", str(exc))


def canonical_outcome_stage_manifest_bytes(payload: Mapping) -> bytes:
    try:
        return canonical_json_bytes(validate_outcome_stage_manifest(payload))
    except ArtifactIOError as exc:
        _hold("HOLD_TEST_STAGE_JSON", str(exc))


def _checked_root(run_root: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(run_root)))
    if root.is_symlink() or not root.is_dir():
        _hold("HOLD_TEST_STAGE_PATH", "run root must be an existing real directory")
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current = current / part
        if current.is_symlink():
            _hold("HOLD_TEST_STAGE_PATH", "run root cannot traverse symlinks")
    if any(PROTECTED_TOKEN.search(part) for part in root.parts):
        _hold("HOLD_PROTECTED_INPUT", "run root names a protected namespace")
    return root


def _fixed_artifact_path(run_root: Path, path: Path, expected_relative: Path) -> Path:
    root = _checked_root(run_root)
    candidate = Path(os.path.abspath(os.fspath(path)))
    if candidate != root / expected_relative:
        _hold("HOLD_TEST_STAGE_PATH", "artifact does not use its fixed stage path")
    current = root
    for part in expected_relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            _hold("HOLD_TEST_STAGE_PATH", "artifact path traverses a symlink")
    if candidate.is_symlink():
        _hold("HOLD_TEST_STAGE_PATH", "artifact cannot be a symlink")
    return candidate


def _publish_manifest(
    run_root: Path,
    path: Path,
    expected_relative: Path,
    payload: Mapping,
    validator: Callable[[Mapping], dict],
) -> str:
    candidate = _fixed_artifact_path(run_root, path, expected_relative)
    if candidate.parent.is_symlink() or not candidate.parent.is_dir():
        _hold("HOLD_TEST_STAGE_PATH", "manifest parent must exist as a real directory")
    try:
        return atomic_write_json_exclusive(candidate, payload, validator=validator)
    except ArtifactIOError as exc:
        _hold("HOLD_TEST_STAGE_WRITE", str(exc))


def write_feature_stage_manifest_exclusive(
    run_root: Path,
    path: Path,
    payload: Mapping,
) -> str:
    return _publish_manifest(
        run_root,
        path,
        FEATURE_MANIFEST_RELATIVE_PATH,
        payload,
        validate_feature_stage_manifest,
    )


def write_outcome_stage_manifest_exclusive(
    run_root: Path,
    path: Path,
    payload: Mapping,
) -> str:
    return _publish_manifest(
        run_root,
        path,
        OUTCOME_MANIFEST_RELATIVE_PATH,
        payload,
        validate_outcome_stage_manifest,
    )


def _parse_canonical(raw: bytes, *, label: str) -> dict:
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
        _hold("HOLD_TEST_STAGE_JSON", f"{label} is not strict JSON: {exc}")
    if not isinstance(payload, dict):
        _hold("HOLD_TEST_STAGE_JSON", f"{label} must be a JSON object")
    return payload


def _load_manifest(
    run_root: Path,
    path: Path,
    expected_sha256: str,
    expected_relative: Path,
    validator: Callable[[Mapping], dict],
    canonicalizer: Callable[[Mapping], bytes],
) -> dict:
    candidate = _fixed_artifact_path(run_root, path, expected_relative)
    expected = _digest(expected_sha256, label="manifest expected SHA-256")
    try:
        raw = read_exact_regular_file(candidate, expected)
    except ArtifactIOError as exc:
        _hold("HOLD_TEST_STAGE_HASH_MISMATCH", str(exc))
    if len(raw) > MAX_MANIFEST_BYTES:
        _hold("HOLD_TEST_STAGE_SCHEMA", "manifest is too large")
    payload = validator(_parse_canonical(raw, label="stage capability manifest"))
    if raw != canonicalizer(payload):
        _hold("HOLD_TEST_STAGE_NONCANONICAL", "manifest bytes are not canonical")
    return payload


def load_feature_stage_manifest_exact(
    run_root: Path,
    path: Path,
    expected_sha256: str,
) -> dict:
    return _load_manifest(
        run_root,
        path,
        expected_sha256,
        FEATURE_MANIFEST_RELATIVE_PATH,
        validate_feature_stage_manifest,
        canonical_feature_stage_manifest_bytes,
    )


def load_outcome_stage_manifest_exact(
    run_root: Path,
    path: Path,
    expected_sha256: str,
) -> dict:
    return _load_manifest(
        run_root,
        path,
        expected_sha256,
        OUTCOME_MANIFEST_RELATIVE_PATH,
        validate_outcome_stage_manifest,
        canonical_outcome_stage_manifest_bytes,
    )


# Fixture-only registries.  They intentionally ship empty and are unreachable
# from every public runtime API.  Future production integration must replace
# this entire disabled layer with versioned, code-frozen semantic, OS, pinned-
# namespace, and aggregate-evaluator implementations; monkeypatching these
# private maps cannot unlock production access.
_FEATURE_SEMANTIC_VERIFIER_REGISTRY: Mapping[str, Callable[[Path, Mapping], Mapping]] = (
    MappingProxyType({})
)
_OUTCOME_SEMANTIC_VERIFIER_REGISTRY: Mapping[str, Callable[[Path, Mapping], Mapping]] = (
    MappingProxyType({})
)
_FEATURE_OS_VERIFIER_REGISTRY: Mapping[str, Callable[[Path], Mapping]] = MappingProxyType({})
_OUTCOME_OS_VERIFIER_REGISTRY: Mapping[str, Callable[[Path], Mapping]] = MappingProxyType({})


@dataclass(frozen=True)
class _RuntimeCapability:
    stage: str
    manifest_sha256: str
    data_authority_sha256: str
    row_set_sha256: str
    _integrity_token: object


_RUNTIME_CAPABILITY_TOKEN = object()
_FIXTURE_ONLY_TOKEN = object()


def _require_fixture_token(value: object) -> None:
    if value is not _FIXTURE_ONLY_TOKEN:
        _hold(
            "HOLD_TEST_STAGE_FIXTURE_SEAM",
            "synthetic fixture seam is unavailable to production callers",
        )


def _registered(
    registry: Mapping[str, Callable],
    verifier_id: str,
    *,
    hold_code: str,
) -> Callable:
    verifier = registry.get(verifier_id)
    if verifier is None or not callable(verifier):
        _hold(hold_code, "required strict runtime verifier is not connected")
    return verifier


def _verify_feature_semantics(run_root: Path, manifest: Mapping) -> None:
    verifier = _registered(
        _FEATURE_SEMANTIC_VERIFIER_REGISTRY,
        manifest["semantic_verifier_id"],
        hold_code="HOLD_FEATURE_STAGE_SEMANTIC_FREEZE_VERIFIER_NOT_CONNECTED",
    )
    evidence = _strict_keys(
        verifier(_checked_root(run_root), manifest["typed_pre_test_freeze_commit_ref"]),
        FEATURE_SEMANTIC_EVIDENCE_KEYS,
        label="feature semantic verifier evidence",
    )
    expected = {
        "schema_version": "r3x_strict_typed_pre_test_freeze_replay_evidence_v1",
        "verified_commit_sha256": manifest["typed_pre_test_freeze_commit_ref"]["sha256"],
        "authorized_feature_authority_sha256": manifest["feature_authority_ref"]["sha256"],
        "authorized_projection_sha256": {
            name: manifest["feature_row_projection_refs"][name]["sha256"]
            for name in ROW_AUTHORITIES
        },
        "authorized_row_set_sha256": manifest["feature_row_set_sha256"],
        "semantic_replay_passed": True,
        "outcome_refs_absent": True,
        "train_refs_absent": True,
        "mutable_selector_checkpoint_fields_absent": True,
    }
    if dict(evidence) != expected:
        _hold("HOLD_FEATURE_STAGE_SEMANTIC_REPLAY", "freeze semantic evidence mismatches manifest")


def _verify_feature_os_capability(run_root: Path, manifest: Mapping) -> None:
    verifier = _registered(
        _FEATURE_OS_VERIFIER_REGISTRY,
        manifest["os_capability_verifier_id"],
        hold_code="HOLD_FEATURE_STAGE_OS_CAPABILITY_NOT_VERIFIED",
    )
    evidence = _strict_keys(
        verifier(_checked_root(run_root)),
        FEATURE_OS_EVIDENCE_KEYS,
        label="feature OS capability evidence",
    )
    expected = {
        "schema_version": "r3x_feature_process_mount_allowlist_evidence_v1",
        "feature_authority_mounted": True,
        "train_authority_mounted": False,
        "outcome_authority_mounted": False,
        "sealed_authority_mounted": False,
        "mount_allowlist_replay_passed": True,
    }
    if dict(evidence) != expected:
        _hold("HOLD_FEATURE_STAGE_OS_CAPABILITY_NOT_VERIFIED", "feature mount allowlist failed")


def _verify_feature_stage_capability_fixture(
    run_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    *,
    _fixture_token: object,
) -> _RuntimeCapability:
    _require_fixture_token(_fixture_token)
    manifest = load_feature_stage_manifest_exact(run_root, manifest_path, manifest_sha256)
    _verify_feature_semantics(run_root, manifest)
    _verify_feature_os_capability(run_root, manifest)
    return _RuntimeCapability(
        stage="internal_test_features",
        manifest_sha256=manifest_sha256,
        data_authority_sha256=manifest["feature_authority_ref"]["sha256"],
        row_set_sha256=manifest["feature_row_set_sha256"],
        _integrity_token=_RUNTIME_CAPABILITY_TOKEN,
    )


def _verify_outcome_semantics(run_root: Path, manifest: Mapping) -> None:
    verifier = _registered(
        _OUTCOME_SEMANTIC_VERIFIER_REGISTRY,
        manifest["semantic_verifier_id"],
        hold_code="HOLD_OUTCOME_STAGE_SEMANTIC_PREDICTION_VERIFIER_NOT_CONNECTED",
    )
    evidence = _strict_keys(
        verifier(_checked_root(run_root), manifest["semantic_test_prediction_commit_ref"]),
        OUTCOME_SEMANTIC_EVIDENCE_KEYS,
        label="outcome semantic verifier evidence",
    )
    expected = {
        "schema_version": "r3x_strict_semantic_test_prediction_replay_evidence_v1",
        "verified_commit_sha256": manifest["semantic_test_prediction_commit_ref"]["sha256"],
        "authorized_outcome_authority_sha256": manifest["outcome_authority_ref"]["sha256"],
        "authorized_row_set_sha256": manifest["outcome_row_set_sha256"],
        "semantic_replay_passed": True,
        "predictions_frozen_semantically": True,
        "selector_checkpoint_identity_bound": True,
        "outcome_not_previously_accessed": True,
    }
    if dict(evidence) != expected:
        _hold("HOLD_OUTCOME_STAGE_SEMANTIC_REPLAY", "prediction semantic evidence mismatches manifest")


def _verify_outcome_os_capability(run_root: Path, manifest: Mapping) -> None:
    verifier = _registered(
        _OUTCOME_OS_VERIFIER_REGISTRY,
        manifest["os_capability_verifier_id"],
        hold_code="HOLD_OUTCOME_STAGE_OS_CAPABILITY_NOT_VERIFIED",
    )
    evidence = _strict_keys(
        verifier(_checked_root(run_root)),
        OUTCOME_OS_EVIDENCE_KEYS,
        label="outcome OS capability evidence",
    )
    expected = {
        "schema_version": "r3x_outcome_process_mount_allowlist_evidence_v1",
        "outcome_authority_mounted": True,
        "train_authority_mounted": False,
        "feature_authority_mounted": False,
        "model_config_checkpoint_mounted": False,
        "sealed_authority_mounted": False,
        "mount_allowlist_replay_passed": True,
    }
    if dict(evidence) != expected:
        _hold("HOLD_OUTCOME_STAGE_OS_CAPABILITY_NOT_VERIFIED", "outcome mount allowlist failed")


def _verify_outcome_stage_capability_fixture(
    run_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    *,
    _fixture_token: object,
) -> _RuntimeCapability:
    _require_fixture_token(_fixture_token)
    manifest = load_outcome_stage_manifest_exact(run_root, manifest_path, manifest_sha256)
    _verify_outcome_semantics(run_root, manifest)
    _verify_outcome_os_capability(run_root, manifest)
    return _RuntimeCapability(
        stage="ordinary_outcome_release",
        manifest_sha256=manifest_sha256,
        data_authority_sha256=manifest["outcome_authority_ref"]["sha256"],
        row_set_sha256=manifest["outcome_row_set_sha256"],
        _integrity_token=_RUNTIME_CAPABILITY_TOKEN,
    )


def _read_internal_test_features_fixture(
    run_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    *,
    _fixture_token: object,
) -> bytes:
    _require_fixture_token(_fixture_token)
    capability = _verify_feature_stage_capability_fixture(
        run_root,
        manifest_path,
        manifest_sha256,
        _fixture_token=_fixture_token,
    )
    if capability._integrity_token is not _RUNTIME_CAPABILITY_TOKEN:
        _hold("HOLD_FEATURE_STAGE_CAPABILITY_TOKEN", "feature capability token is invalid")
    manifest = load_feature_stage_manifest_exact(run_root, manifest_path, manifest_sha256)
    authority_path = _checked_root(run_root) / manifest["feature_authority_ref"]["relative_path"]
    try:
        return read_exact_regular_file(authority_path, capability.data_authority_sha256)
    except ArtifactIOError as exc:
        _hold("HOLD_FEATURE_AUTHORITY_READ", str(exc))


def verify_feature_stage_capability_runtime(
    run_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
) -> NoReturn:
    """Production sentinel: exact replay cannot unlock feature access."""

    load_feature_stage_manifest_exact(run_root, manifest_path, manifest_sha256)
    _hold(
        "HOLD_FEATURE_STAGE_PRODUCTION_RUNTIME_DISABLED",
        "semantic, OS, and descriptor-pinned namespace verifiers are not integrated",
    )


def verify_outcome_stage_capability_runtime(
    run_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
) -> NoReturn:
    """Production sentinel: exact replay cannot unlock outcome access."""

    load_outcome_stage_manifest_exact(run_root, manifest_path, manifest_sha256)
    _hold(
        "HOLD_OUTCOME_STAGE_PRODUCTION_RUNTIME_DISABLED",
        "semantic, OS, pinned-namespace, and aggregate verifiers are not integrated",
    )


def read_internal_test_features_with_capability(
    run_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
) -> NoReturn:
    """Public API never exports feature bytes in this development-only build."""

    verify_feature_stage_capability_runtime(run_root, manifest_path, manifest_sha256)


def _claim_payload(manifest: Mapping, manifest_sha256: str) -> dict:
    return {
        "schema_version": OUTCOME_CLAIM_SCHEMA,
        "artifact_role": OUTCOME_CLAIM_ROLE,
        "status": "CLAIM_PUBLISHED_BEFORE_FIRST_OUTCOME_BYTE",
        "outcome_stage_manifest_ref": {
            "relative_path": OUTCOME_MANIFEST_RELATIVE_PATH.as_posix(),
            "sha256": manifest_sha256,
        },
        "semantic_test_prediction_commit_ref": dict(
            manifest["semantic_test_prediction_commit_ref"]
        ),
        "outcome_authority_ref": dict(manifest["outcome_authority_ref"]),
        "outcome_row_set_sha256": manifest["outcome_row_set_sha256"],
        "claim_policy": OUTCOME_CLAIM_POLICY,
        "consumed_once": True,
        "timestamp_used_as_ordering_evidence": False,
        "boundaries": dict(FROZEN_BOUNDARIES),
    }


def _validate_claim(payload: Mapping) -> dict:
    claim = _strict_keys(payload, CLAIM_KEYS, label="outcome release claim")
    expected_literals = {
        "schema_version": OUTCOME_CLAIM_SCHEMA,
        "artifact_role": OUTCOME_CLAIM_ROLE,
        "status": "CLAIM_PUBLISHED_BEFORE_FIRST_OUTCOME_BYTE",
        "claim_policy": OUTCOME_CLAIM_POLICY,
        "consumed_once": True,
        "timestamp_used_as_ordering_evidence": False,
    }
    for field, expected in expected_literals.items():
        _exact(claim.get(field), expected, label=f"claim.{field}")
    _reference(
        claim.get("outcome_stage_manifest_ref"),
        label="claim.outcome_stage_manifest_ref",
        expected_path=OUTCOME_MANIFEST_RELATIVE_PATH.as_posix(),
    )
    _reference(
        claim.get("semantic_test_prediction_commit_ref"),
        label="claim.semantic_test_prediction_commit_ref",
        expected_path=SEMANTIC_TEST_PREDICTION_RELATIVE_PATH,
    )
    _reference(
        claim.get("outcome_authority_ref"),
        label="claim.outcome_authority_ref",
        expected_path=OUTCOME_AUTHORITY_RELATIVE_PATH,
    )
    _digest(claim.get("outcome_row_set_sha256"), label="claim outcome row set")
    _validate_boundaries(claim.get("boundaries"))
    return dict(claim)


def _release_ordinary_outcome_fixture_once(
    run_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    *,
    _fixture_token: object,
    _after_claim_hook: Callable[[], None] | None = None,
    _read_trace_hook: Callable[[Mapping], None] | None = None,
) -> dict:
    """Synthetic-only claim/read tracer; never returns row-level bytes."""

    _require_fixture_token(_fixture_token)
    capability = _verify_outcome_stage_capability_fixture(
        run_root,
        manifest_path,
        manifest_sha256,
        _fixture_token=_fixture_token,
    )
    if capability._integrity_token is not _RUNTIME_CAPABILITY_TOKEN:
        _hold("HOLD_OUTCOME_STAGE_CAPABILITY_TOKEN", "outcome capability token is invalid")
    manifest = load_outcome_stage_manifest_exact(run_root, manifest_path, manifest_sha256)
    claim_path = _checked_root(run_root) / OUTCOME_CLAIM_RELATIVE_PATH
    claim_payload = _claim_payload(manifest, manifest_sha256)
    try:
        claim_sha256 = atomic_write_json_exclusive(
            claim_path,
            claim_payload,
            validator=_validate_claim,
        )
    except ArtifactIOError as exc:
        _hold(
            "HOLD_OUTCOME_RELEASE_ALREADY_CLAIMED_NO_RETRY",
            str(exc),
        )
    if _after_claim_hook is not None:
        _after_claim_hook()
    outcome_path = _checked_root(run_root) / manifest["outcome_authority_ref"]["relative_path"]
    try:
        outcome_bytes = read_exact_regular_file(outcome_path, capability.data_authority_sha256)
    except ArtifactIOError as exc:
        _hold("HOLD_OUTCOME_AUTHORITY_READ_AFTER_CLAIM", str(exc))
    outcome_sha256 = hashlib.sha256(outcome_bytes).hexdigest()
    read_trace = {
        "claim_exists_before_read_completed": claim_path.exists(),
        "outcome_bytes_sha256": outcome_sha256,
        "outcome_byte_count": len(outcome_bytes),
    }
    if _read_trace_hook is not None:
        _read_trace_hook(read_trace)
    return {
        "status": "SYNTHETIC_FIXTURE_OUTCOME_READ_TRACED_AFTER_DURABLE_CLAIM",
        "capability_ready": True,
        "claim_ref": {
            "relative_path": OUTCOME_CLAIM_RELATIVE_PATH,
            "sha256": claim_sha256,
        },
        "outcome_bytes_sha256": outcome_sha256,
        "outcome_byte_count": len(outcome_bytes),
        "outcome_row_set_sha256": capability.row_set_sha256,
        "ordinary_only": True,
        "sealed_access": False,
        "downstream_unlock": False,
    }


def release_ordinary_outcome_once(
    run_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
) -> NoReturn:
    """Production sentinel: no row-level outcome read or export is available."""

    load_outcome_stage_manifest_exact(run_root, manifest_path, manifest_sha256)
    _hold(
        "HOLD_OUTCOME_AGGREGATE_EVALUATOR_NOT_CONNECTED",
        "strict aggregate evaluator and descriptor-pinned claim namespace are absent",
    )


def require_feature_stage_capability(
    run_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
) -> NoReturn:
    """Current shipped behavior: exact manifest replay then structured HOLD."""

    verify_feature_stage_capability_runtime(run_root, manifest_path, manifest_sha256)


def require_outcome_stage_capability(
    run_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
) -> NoReturn:
    """Current shipped behavior: exact manifest replay then structured HOLD."""

    verify_outcome_stage_capability_runtime(run_root, manifest_path, manifest_sha256)


__all__ = [
    "FEATURE_AUTHORITY_RELATIVE_PATH",
    "FEATURE_MANIFEST_RELATIVE_PATH",
    "FEATURE_PROJECTION_RELATIVE_PATHS",
    "FEATURE_STAGE_ROLE",
    "FEATURE_STAGE_SCHEMA",
    "FEATURE_SEMANTIC_VERIFIER_ID",
    "FEATURE_OS_VERIFIER_ID",
    "OUTCOME_AUTHORITY_RELATIVE_PATH",
    "OUTCOME_CLAIM_POLICY",
    "OUTCOME_CLAIM_RELATIVE_PATH",
    "OUTCOME_MANIFEST_RELATIVE_PATH",
    "OUTCOME_OS_VERIFIER_ID",
    "OUTCOME_SEMANTIC_VERIFIER_ID",
    "OUTCOME_STAGE_ROLE",
    "OUTCOME_STAGE_SCHEMA",
    "SEMANTIC_TEST_PREDICTION_RELATIVE_PATH",
    "TYPED_PRE_TEST_FREEZE_RELATIVE_PATH",
    "TestStageCapabilityError",
    "canonical_feature_stage_manifest_bytes",
    "canonical_outcome_stage_manifest_bytes",
    "load_feature_stage_manifest_exact",
    "load_outcome_stage_manifest_exact",
    "read_internal_test_features_with_capability",
    "release_ordinary_outcome_once",
    "require_feature_stage_capability",
    "require_outcome_stage_capability",
    "validate_feature_stage_manifest",
    "validate_outcome_stage_manifest",
    "verify_feature_stage_capability_runtime",
    "verify_outcome_stage_capability_runtime",
    "write_feature_stage_manifest_exclusive",
    "write_outcome_stage_manifest_exclusive",
]
