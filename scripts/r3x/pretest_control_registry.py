"""Train/development-only negative-control registry for pre-test freeze.

The historical control summary is post-outcome and therefore cannot serve as a
pre-test authority.  This module defines a separate registry containing only
immutable train/development artifact references.  It carries no internal-test
path, digest role, row support, result, or metric.

``CONTROL_NOT_RUN`` is a typed blocking disposition, never PASS.  A completed
entry must instead be ``CONTROL_RUN_DEVELOPMENT_REPLAYED`` and bind its train
control manifest, development state, development predictions, and exact
verifier artifact.  Runtime replay uses a closed verifier registry; callers
cannot supply a verifier mapping.

The currently required broken-pair control remains ``PROPOSED_NOT_ACTIVE`` and
must be ``CONTROL_NOT_RUN``.  In addition, the shared artifact I/O layer does
not descriptor-pin the parent directory across the entire operation.  Both
facts make the current registry an honest HOLD with
``can_satisfy_pretest_freeze=false`` even when every other development control
has evidence.  This module never opens evaluation-stage data.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import AbstractSet, Any, Callable, Mapping, NoReturn

from .artifact_io import (
    ArtifactIOError,
    atomic_write_json_exclusive,
    canonical_json_bytes,
    read_exact_regular_file,
)
from .protocol import ProtocolHoldError, REQUIRED_NEGATIVE_CONTROLS, canonical_sha256


SCHEMA = "r3x_pretest_train_development_control_registry_v1"
ARTIFACT_ROLE = "PRETEST_TRAIN_DEVELOPMENT_CONTROL_EVIDENCE_REGISTRY"
OVERALL_STATUS = "HOLD_PRETEST_CONTROLS_INCOMPLETE"
REGISTRY_RELATIVE_PATH = Path(
    "development/pretest_controls/pretest_control_registry.json"
)

CONTROL_RUN = "CONTROL_RUN_DEVELOPMENT_REPLAYED"
CONTROL_NOT_RUN = "CONTROL_NOT_RUN"
DISPOSITIONS = frozenset({CONTROL_RUN, CONTROL_NOT_RUN})

ACTIVE = "ACTIVE"
PROPOSED_NOT_ACTIVE = "PROPOSED_NOT_ACTIVE"
BROKEN_PAIR_CONTROL_ID = "SOURCE_CANDIDATE_PAIRING_SHUFFLE"
BROKEN_PAIR_REASON = "BROKEN_PAIR_POLICY_PROPOSED_NOT_ACTIVE"

SIGN_ACCURACY_GATE = 0.60
SEMANTIC_VERIFIER_STATUS = "EXACT_DEVELOPMENT_VERIFIER_REGISTRY_REPLAY_REQUIRED"
NAMESPACE_INTEGRITY_STATUS = (
    "BLOCKED_PARENT_DIRECTORY_NOT_DESCRIPTOR_PINNED_PRETEST_FREEZE_DISABLED"
)

METHOD_PROTOCOL_RELATIVE_PATH = "development/authorities/method_protocol.json"
TRAIN_DEVELOPMENT_MANIFEST_RELATIVE_PATH = (
    "development/pretest_controls/train_development_control_manifest.json"
)
TRAIN_DEVELOPMENT_STATE_RELATIVE_PATH = (
    "development/pretest_controls/train_development_control_state.json"
)

REQUIRED_CONTROL_IDS = tuple(REQUIRED_NEGATIVE_CONTROLS)
if len(REQUIRED_CONTROL_IDS) != len(set(REQUIRED_CONTROL_IDS)):
    raise RuntimeError("required negative-control protocol contains duplicates")
if not REQUIRED_CONTROL_IDS or REQUIRED_CONTROL_IDS[0] != BROKEN_PAIR_CONTROL_ID:
    raise RuntimeError("broken-pair control ordering drifted from method protocol")
REQUIRED_CONTROL_IDS_SHA256 = canonical_sha256(list(REQUIRED_CONTROL_IDS))

HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_REASON = re.compile(r"^[A-Z][A-Z0-9_]{2,127}$")
PROTECTED_TOKEN = re.compile(
    r"(?:internal[_-]?test|test[_-]?outcome|sealed|restricted|gse246381)",
    re.IGNORECASE,
)

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

REGISTRY_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "method_protocol_ref",
    "required_control_ids",
    "required_control_ids_sha256",
    "train_development_manifest_ref",
    "train_development_state_ref",
    "controls",
    "semantic_verifier_status",
    "namespace_integrity_status",
    "broken_pair_activation_status",
    "all_required_controls_development_replayed",
    "can_satisfy_pretest_freeze",
    "boundaries",
}

ENTRY_KEYS = {
    "control_id",
    "disposition",
    "activation_status",
    "train_control_manifest_ref",
    "development_state_ref",
    "development_prediction_ref",
    "development_verifier_ref",
    "verifier_id",
    "evidence_bundle_sha256",
    "reason_code",
}

VERIFIER_EVIDENCE_KEYS = {
    "schema_version",
    "control_id",
    "verifier_id",
    "train_development_manifest_sha256",
    "train_development_state_sha256",
    "train_control_manifest_sha256",
    "development_state_sha256",
    "development_prediction_sha256",
    "development_verifier_artifact_sha256",
    "development_only_replay_passed",
    "train_only_transform_verified",
    "evaluation_stage_accessed",
    "metrics_or_row_support_returned",
}

ALLOWED_NOT_RUN_REASONS = frozenset({
    BROKEN_PAIR_REASON,
    "CONTROL_EVIDENCE_NOT_AVAILABLE",
    "CONTROL_PROTOCOL_NOT_ACTIVE",
    "DEVELOPMENT_REPLAY_NOT_RUN",
})

MAX_REGISTRY_BYTES = 2 * 1024 * 1024


class PretestControlRegistryError(ProtocolHoldError):
    """Structured pre-test HOLD; no NOT_RUN state is interpreted as PASS."""


def _hold(code: str, message: str, **details: Any) -> NoReturn:
    raise PretestControlRegistryError(code, message, details=details or None)


def _strict_keys(value: object, expected: AbstractSet[str], *, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        _hold("HOLD_PRETEST_CONTROL_SCHEMA", f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        _hold(
            "HOLD_PRETEST_CONTROL_SCHEMA",
            f"{label} keys differ from the exact schema",
            missing=sorted(expected - actual),
            unexpected=sorted(actual - expected),
        )
    return value


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        _hold("HOLD_PRETEST_CONTROL_SCHEMA", f"{label} must be lowercase SHA-256")
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
        _hold("HOLD_PRETEST_CONTROL_SCHEMA", f"{label} differs from frozen policy")


def _canonical_relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        _hold("HOLD_PRETEST_CONTROL_PATH", f"{label} must be relative POSIX text")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        _hold("HOLD_PRETEST_CONTROL_PATH", f"{label} must be normalized and relative")
    if PROTECTED_TOKEN.search(value):
        _hold("HOLD_PROTECTED_INPUT", f"{label} names evaluation/protected support")
    return path.as_posix()


def _reference(value: object, *, label: str, expected_path: str) -> dict[str, str]:
    ref = _strict_keys(value, REF_KEYS, label=label)
    path = _canonical_relative_path(ref.get("relative_path"), label=f"{label}.relative_path")
    if path != expected_path:
        _hold("HOLD_PRETEST_CONTROL_PATH", f"{label} path differs from exact policy")
    return {
        "relative_path": path,
        "sha256": _digest(ref.get("sha256"), label=f"{label}.sha256"),
    }


def _control_slug(control_id: str) -> str:
    return control_id.lower()


def _entry_paths(control_id: str) -> dict[str, str]:
    root = f"development/pretest_controls/{_control_slug(control_id)}"
    return {
        "train_control_manifest_ref": f"{root}/train_control_manifest.json",
        "development_state_ref": f"{root}/development_state.json",
        "development_prediction_ref": f"{root}/development_predictions.jsonl",
        "development_verifier_ref": f"{root}/development_verifier.json",
    }


def expected_verifier_id(control_id: str) -> str:
    if control_id not in REQUIRED_CONTROL_IDS:
        _hold("HOLD_PRETEST_CONTROL_ID", "unsupported negative control")
    return f"R3X_{control_id}_TRAIN_DEVELOPMENT_REPLAY_V1"


def _evidence_bundle_payload(
    control_id: str,
    train_control_manifest_ref: Mapping,
    development_state_ref: Mapping,
    development_prediction_ref: Mapping,
    development_verifier_ref: Mapping,
    verifier_id: str,
) -> dict:
    return {
        "schema_version": "r3x_pretest_control_evidence_bundle_binding_v1",
        "control_id": control_id,
        "train_control_manifest_ref": dict(train_control_manifest_ref),
        "development_state_ref": dict(development_state_ref),
        "development_prediction_ref": dict(development_prediction_ref),
        "development_verifier_ref": dict(development_verifier_ref),
        "verifier_id": verifier_id,
    }


def control_evidence_bundle_sha256(entry: Mapping) -> str:
    """Compute the exact development-only bundle commitment for a RUN entry."""

    control_id = entry.get("control_id")
    if control_id not in REQUIRED_CONTROL_IDS:
        _hold("HOLD_PRETEST_CONTROL_ID", "unsupported negative control")
    paths = _entry_paths(control_id)
    refs = {
        field: _reference(entry.get(field), label=field, expected_path=paths[field])
        for field in paths
    }
    verifier_id = entry.get("verifier_id")
    _exact(verifier_id, expected_verifier_id(control_id), label="verifier_id")
    return canonical_sha256(
        _evidence_bundle_payload(
            control_id,
            refs["train_control_manifest_ref"],
            refs["development_state_ref"],
            refs["development_prediction_ref"],
            refs["development_verifier_ref"],
            verifier_id,
        )
    )


def _validate_entry(value: object, *, expected_control_id: str) -> dict:
    entry = _strict_keys(value, ENTRY_KEYS, label=f"control {expected_control_id}")
    _exact(entry.get("control_id"), expected_control_id, label="control_id")
    disposition = entry.get("disposition")
    if disposition not in DISPOSITIONS:
        _hold("HOLD_PRETEST_CONTROL_DISPOSITION", "unsupported control disposition")
    paths = _entry_paths(expected_control_id)
    manifest_ref = _reference(
        entry.get("train_control_manifest_ref"),
        label="train_control_manifest_ref",
        expected_path=paths["train_control_manifest_ref"],
    )

    if expected_control_id == BROKEN_PAIR_CONTROL_ID:
        if (
            disposition != CONTROL_NOT_RUN
            or entry.get("activation_status") != PROPOSED_NOT_ACTIVE
            or entry.get("reason_code") != BROKEN_PAIR_REASON
        ):
            _hold(
                "HOLD_BROKEN_PAIR_CONTROL_PROPOSED_NOT_ACTIVE",
                "broken-pair control must remain typed NOT_RUN while policy is inactive",
            )

    if disposition == CONTROL_NOT_RUN:
        if entry.get("activation_status") not in {ACTIVE, PROPOSED_NOT_ACTIVE}:
            _hold("HOLD_PRETEST_CONTROL_SCHEMA", "NOT_RUN activation status is invalid")
        reason = entry.get("reason_code")
        if (
            not isinstance(reason, str)
            or not SAFE_REASON.fullmatch(reason)
            or reason not in ALLOWED_NOT_RUN_REASONS
        ):
            _hold("HOLD_PRETEST_CONTROL_NOT_RUN_REASON", "NOT_RUN reason is not exact")
        for field in (
            "development_state_ref",
            "development_prediction_ref",
            "development_verifier_ref",
            "verifier_id",
            "evidence_bundle_sha256",
        ):
            if entry.get(field) is not None:
                _hold(
                    "HOLD_PRETEST_CONTROL_NOT_RUN_EVIDENCE",
                    "NOT_RUN cannot pretend to have completed development evidence",
                )
        return dict(entry)

    _exact(entry.get("activation_status"), ACTIVE, label="RUN activation_status")
    if entry.get("reason_code") is not None:
        _hold("HOLD_PRETEST_CONTROL_SCHEMA", "completed control cannot carry NOT_RUN reason")
    refs = {"train_control_manifest_ref": manifest_ref}
    for field in (
        "development_state_ref",
        "development_prediction_ref",
        "development_verifier_ref",
    ):
        refs[field] = _reference(
            entry.get(field),
            label=field,
            expected_path=paths[field],
        )
    verifier_id = expected_verifier_id(expected_control_id)
    _exact(entry.get("verifier_id"), verifier_id, label="verifier_id")
    expected_bundle = canonical_sha256(
        _evidence_bundle_payload(
            expected_control_id,
            refs["train_control_manifest_ref"],
            refs["development_state_ref"],
            refs["development_prediction_ref"],
            refs["development_verifier_ref"],
            verifier_id,
        )
    )
    if entry.get("evidence_bundle_sha256") != expected_bundle:
        _hold("HOLD_PRETEST_CONTROL_BINDING", "development evidence bundle digest is stale")
    return dict(entry)


def _validate_boundaries(value: object) -> dict:
    boundaries = _strict_keys(value, BOUNDARY_KEYS, label="boundaries")
    for field, expected in FROZEN_BOUNDARIES.items():
        _exact(boundaries.get(field), expected, label=f"boundaries.{field}")
    return dict(boundaries)


def validate_pretest_control_registry(payload: Mapping) -> dict:
    """Validate development-only evidence bindings without claiming PASS."""

    registry = _strict_keys(payload, REGISTRY_KEYS, label="pretest control registry")
    expected_literals = {
        "schema_version": SCHEMA,
        "artifact_role": ARTIFACT_ROLE,
        "status": OVERALL_STATUS,
        "required_control_ids": list(REQUIRED_CONTROL_IDS),
        "required_control_ids_sha256": REQUIRED_CONTROL_IDS_SHA256,
        "semantic_verifier_status": SEMANTIC_VERIFIER_STATUS,
        "namespace_integrity_status": NAMESPACE_INTEGRITY_STATUS,
        "broken_pair_activation_status": PROPOSED_NOT_ACTIVE,
        "all_required_controls_development_replayed": False,
        "can_satisfy_pretest_freeze": False,
    }
    for field, expected in expected_literals.items():
        _exact(registry.get(field), expected, label=field)
    method_ref = _reference(
        registry.get("method_protocol_ref"),
        label="method_protocol_ref",
        expected_path=METHOD_PROTOCOL_RELATIVE_PATH,
    )
    train_manifest_ref = _reference(
        registry.get("train_development_manifest_ref"),
        label="train_development_manifest_ref",
        expected_path=TRAIN_DEVELOPMENT_MANIFEST_RELATIVE_PATH,
    )
    train_state_ref = _reference(
        registry.get("train_development_state_ref"),
        label="train_development_state_ref",
        expected_path=TRAIN_DEVELOPMENT_STATE_RELATIVE_PATH,
    )
    controls = registry.get("controls")
    if not isinstance(controls, list) or len(controls) != len(REQUIRED_CONTROL_IDS):
        _hold("HOLD_PRETEST_CONTROL_SET", "controls must contain every required entry")
    normalized = [
        _validate_entry(entry, expected_control_id=control_id)
        for entry, control_id in zip(controls, REQUIRED_CONTROL_IDS)
    ]
    if [entry["control_id"] for entry in normalized] != list(REQUIRED_CONTROL_IDS):
        _hold("HOLD_PRETEST_CONTROL_SET", "control ordering differs from method protocol")
    digests = [
        method_ref["sha256"],
        train_manifest_ref["sha256"],
        train_state_ref["sha256"],
    ]
    for entry in normalized:
        for field in (
            "train_control_manifest_ref",
            "development_state_ref",
            "development_prediction_ref",
            "development_verifier_ref",
        ):
            ref = entry.get(field)
            if ref is not None:
                digests.append(ref["sha256"])
        if entry.get("evidence_bundle_sha256") is not None:
            digests.append(entry["evidence_bundle_sha256"])
    if len(digests) != len(set(digests)):
        _hold("HOLD_PRETEST_CONTROL_ALIAS", "control evidence commitments must be distinct")
    _validate_boundaries(registry.get("boundaries"))
    return dict(registry)


def canonical_pretest_control_registry_bytes(payload: Mapping) -> bytes:
    try:
        raw = canonical_json_bytes(validate_pretest_control_registry(payload))
    except ArtifactIOError as exc:
        _hold("HOLD_PRETEST_CONTROL_JSON", str(exc))
    if len(raw) > MAX_REGISTRY_BYTES:
        _hold("HOLD_PRETEST_CONTROL_SCHEMA", "pretest control registry is too large")
    return raw


def _checked_root(run_root: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(run_root)))
    if root.is_symlink() or not root.is_dir():
        _hold("HOLD_PRETEST_CONTROL_PATH", "run root must be an existing real directory")
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current = current / part
        if current.is_symlink():
            _hold("HOLD_PRETEST_CONTROL_PATH", "run root cannot traverse symlinks")
    if any(PROTECTED_TOKEN.search(part) for part in root.parts):
        _hold("HOLD_PROTECTED_INPUT", "run root names evaluation/protected support")
    return root


def _fixed_registry_path(run_root: Path, path: Path) -> Path:
    root = _checked_root(run_root)
    candidate = Path(os.path.abspath(os.fspath(path)))
    if candidate != root / REGISTRY_RELATIVE_PATH:
        _hold("HOLD_PRETEST_CONTROL_PATH", "registry must use its fixed development path")
    current = root
    for part in REGISTRY_RELATIVE_PATH.parts[:-1]:
        current = current / part
        if current.is_symlink():
            _hold("HOLD_PRETEST_CONTROL_PATH", "registry path traverses a symlink")
    if candidate.is_symlink():
        _hold("HOLD_PRETEST_CONTROL_PATH", "registry cannot be a symlink")
    return candidate


def write_pretest_control_registry_exclusive(
    run_root: Path,
    path: Path,
    payload: Mapping,
) -> str:
    candidate = _fixed_registry_path(run_root, path)
    if candidate.parent.is_symlink() or not candidate.parent.is_dir():
        _hold("HOLD_PRETEST_CONTROL_PATH", "registry parent must be an existing real directory")
    try:
        return atomic_write_json_exclusive(
            candidate,
            payload,
            validator=validate_pretest_control_registry,
        )
    except ArtifactIOError as exc:
        _hold("HOLD_PRETEST_CONTROL_WRITE", str(exc))


def _parse_canonical(raw: bytes) -> dict:
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
        _hold("HOLD_PRETEST_CONTROL_JSON", f"registry is not strict JSON: {exc}")
    if not isinstance(payload, dict):
        _hold("HOLD_PRETEST_CONTROL_JSON", "registry must be a JSON object")
    return payload


def load_pretest_control_registry_exact(
    run_root: Path,
    path: Path,
    expected_sha256: str,
) -> dict:
    candidate = _fixed_registry_path(run_root, path)
    expected = _digest(expected_sha256, label="registry expected SHA-256")
    try:
        raw = read_exact_regular_file(candidate, expected)
    except ArtifactIOError as exc:
        _hold("HOLD_PRETEST_CONTROL_HASH_MISMATCH", str(exc))
    if len(raw) > MAX_REGISTRY_BYTES:
        _hold("HOLD_PRETEST_CONTROL_SCHEMA", "pretest control registry is too large")
    payload = validate_pretest_control_registry(_parse_canonical(raw))
    if raw != canonical_pretest_control_registry_bytes(payload):
        _hold("HOLD_PRETEST_CONTROL_NONCANONICAL", "registry bytes are not canonical")
    return payload


# Production closed registry: no development evidence verifier is connected in
# this local preparation module.  A future code change must register exact
# implementations and re-freeze the analysis code snapshot.
_DEVELOPMENT_CONTROL_VERIFIER_REGISTRY: Mapping[
    str,
    Callable[[Path, Mapping], Mapping],
] = MappingProxyType({})


def _expected_verifier_evidence(
    registry: Mapping,
    entry: Mapping,
) -> dict:
    return {
        "schema_version": "r3x_pretest_control_development_replay_evidence_v1",
        "control_id": entry["control_id"],
        "verifier_id": entry["verifier_id"],
        "train_development_manifest_sha256": registry[
            "train_development_manifest_ref"
        ]["sha256"],
        "train_development_state_sha256": registry[
            "train_development_state_ref"
        ]["sha256"],
        "train_control_manifest_sha256": entry["train_control_manifest_ref"]["sha256"],
        "development_state_sha256": entry["development_state_ref"]["sha256"],
        "development_prediction_sha256": entry["development_prediction_ref"]["sha256"],
        "development_verifier_artifact_sha256": entry[
            "development_verifier_ref"
        ]["sha256"],
        "development_only_replay_passed": True,
        "train_only_transform_verified": True,
        "evaluation_stage_accessed": False,
        "metrics_or_row_support_returned": False,
    }


def require_pretest_control_registry(
    run_root: Path,
    path: Path,
    expected_sha256: str,
) -> NoReturn:
    """Replay exact development evidence then HOLD on every current blocker."""

    registry = load_pretest_control_registry_exact(run_root, path, expected_sha256)
    for entry in registry["controls"]:
        if entry["disposition"] != CONTROL_RUN:
            continue
        verifier = _DEVELOPMENT_CONTROL_VERIFIER_REGISTRY.get(entry["verifier_id"])
        if verifier is None or not callable(verifier):
            _hold(
                "HOLD_PRETEST_CONTROL_VERIFIER_NOT_CONNECTED",
                "completed control lacks an exact registered development verifier",
                control_id=entry["control_id"],
            )
        evidence = _strict_keys(
            verifier(_checked_root(run_root), entry),
            VERIFIER_EVIDENCE_KEYS,
            label=f"verifier evidence {entry['control_id']}",
        )
        if dict(evidence) != _expected_verifier_evidence(registry, entry):
            _hold(
                "HOLD_PRETEST_CONTROL_VERIFIER_REPLAY",
                "development verifier evidence mismatches frozen refs",
                control_id=entry["control_id"],
            )

    not_run = [
        entry["control_id"]
        for entry in registry["controls"]
        if entry["disposition"] == CONTROL_NOT_RUN
    ]
    if BROKEN_PAIR_CONTROL_ID in not_run:
        _hold(
            "HOLD_BROKEN_PAIR_CONTROL_PROPOSED_NOT_ACTIVE",
            "broken-pair control is required but remains PROPOSED_NOT_ACTIVE",
        )
    if not_run:
        _hold(
            "HOLD_PRETEST_CONTROL_NOT_RUN",
            "required CONTROL_NOT_RUN entries cannot satisfy pre-test freeze",
            controls=not_run,
        )
    _hold(
        "HOLD_PRETEST_CONTROL_PARENT_NAMESPACE_NOT_PINNED",
        "parent-directory swap protection is not descriptor-pinned",
    )


__all__ = [
    "ACTIVE",
    "ARTIFACT_ROLE",
    "BROKEN_PAIR_CONTROL_ID",
    "BROKEN_PAIR_REASON",
    "CONTROL_NOT_RUN",
    "CONTROL_RUN",
    "NAMESPACE_INTEGRITY_STATUS",
    "OVERALL_STATUS",
    "PROPOSED_NOT_ACTIVE",
    "PretestControlRegistryError",
    "REGISTRY_RELATIVE_PATH",
    "REQUIRED_CONTROL_IDS",
    "REQUIRED_CONTROL_IDS_SHA256",
    "SCHEMA",
    "canonical_pretest_control_registry_bytes",
    "control_evidence_bundle_sha256",
    "expected_verifier_id",
    "load_pretest_control_registry_exact",
    "require_pretest_control_registry",
    "validate_pretest_control_registry",
    "write_pretest_control_registry_exclusive",
]
