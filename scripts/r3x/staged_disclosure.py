"""Strict three-stage disclosure authority for ordinary-only R3X evaluation.

This module is deliberately independent from the legacy monolithic activation
validator.  A v2 activation binds three physically distinct data authorities:

* train and development features plus outcomes;
* internal-test features without outcomes; and
* internal-test outcomes without features.

It also binds every non-data scientific authority by source digest, but never
opens those full bytes in the evaluation process.  Each authority must instead
provide separate development and internal-test-feature summary projections
with exact schemas and independent semantic-certificate digests.  Those
summaries establish the disclosure boundary only; they deliberately do not
claim typed training readiness.

The public stage functions enforce a one-way disclosure sequence.  The
development entrypoint reads only the first data authority.  The feature
entrypoint reads the feature authority only after exact replay of a pre-test
freeze commit that binds the development selector and every predictor.  The
outcome entrypoint reads the outcome authority once, after exact replay of a
test-prediction commit, its complete prediction row set, and durable
publication of the DAG release claim.

The module does not grant publication status, consume sealed data, alter the
0.60 sign gate, or unlock any downstream phase.  It supplies the physical and
cryptographic boundary required before a runner can make those separate
scientific decisions.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .protocol import (
    AUTHORIZED_ROUTE,
    ProtocolHoldError,
    assert_ordinary_path,
    assert_ordinary_record,
    canonical_sha256,
    read_exact_bytes,
)
from .staged_commits import (
    INTERNAL_TEST_FEATURE_AUTHORITY,
    INTERNAL_TEST_OUTCOME_AUTHORITY,
    PRE_TEST_FREEZE_COMMIT,
    REQUIRED_NONDATA_AUTHORITY_HASHES,
    STATE_RELATIVE_PATHS,
    STATE_SCHEMAS,
    STAGE_PROJECTED_NONDATA_AUTHORITIES,
    TEST_PREDICTION_COMMIT,
    TRAIN_DEVELOPMENT_AUTHORITY,
    StagedCommitError,
    extract_staged_bindings,
    replay_pre_test_freeze_commit,
    replay_test_prediction_commit,
    write_ordinary_test_outcome_release_record_exclusive,
)


ACTIVATION_SCHEMA = "r3_activation_manifest_v2_staged_disclosure"
PRE_TEST_FREEZE_SCHEMA = STATE_SCHEMAS[PRE_TEST_FREEZE_COMMIT]
TEST_PREDICTION_COMMIT_SCHEMA = STATE_SCHEMAS[TEST_PREDICTION_COMMIT]
AUTHORITY_ORDER = (
    TRAIN_DEVELOPMENT_AUTHORITY,
    INTERNAL_TEST_FEATURE_AUTHORITY,
    INTERNAL_TEST_OUTCOME_AUTHORITY,
)
AUTHORITY_ROLES = {
    TRAIN_DEVELOPMENT_AUTHORITY: "ORDINARY_TRAIN_DEVELOPMENT_OUTCOMES",
    INTERNAL_TEST_FEATURE_AUTHORITY: "ORDINARY_INTERNAL_TEST_FEATURES_ONLY",
    INTERNAL_TEST_OUTCOME_AUTHORITY: "ORDINARY_INTERNAL_TEST_OUTCOMES_ONLY",
}
# The only data-authority digest permitted in development checkpoint/selector
# provenance.  In particular, the feature-only and outcome-only internal-test
# authorities must not occur in a development-stage hash set.
DEVELOPMENT_STAGE_AUTHORITY_NAMES = (TRAIN_DEVELOPMENT_AUTHORITY,)
REQUIRED_DEVELOPMENT_AUTHORITY_DIGEST_KEYS = DEVELOPMENT_STAGE_AUTHORITY_NAMES

SIGN_ACCURACY_GATE = 0.60
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_METHOD_ID = re.compile(r"^[A-Z][A-Z0-9_]*$")
SAFE_FIELD_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*$")

_MANIFEST_KEYS = {
    "schema_version",
    "status",
    "mode",
    "selected_route",
    "sign_accuracy_gate",
    "sign_gate_change_authorized",
    "ordinary_only",
    "sealed_access",
    "downstream_unlock",
    "overwrite_authorized",
    "model_feature_fields",
    "outcome_fields",
    "partition_row_set_sha256",
    "authorities",
    "nondata_authorities",
}
_AUTHORITY_REF_KEYS = {"role", "path", "sha256", "status"}
_PROJECTED_NONDATA_AUTHORITY_KEYS = {
    "status",
    "source_sha256",
    "development_projection",
    "internal_test_feature_projection",
}
_PROJECTION_REF_KEYS = {"path", "sha256", "status"}
_PROJECTION_PAYLOAD_KEYS = {
    "schema_version",
    "status",
    "authority_name",
    "source_authority_sha256",
    "disclosure_stage",
    "contains_outcomes",
    "partition_scope",
    "record_count",
    "content_commitment_sha256",
    "semantic_validation_certificate_sha256",
}
_PARTITION_HASH_KEYS = {"train", "development", "internal_test"}
_TRAIN_DEVELOPMENT_ROW_KEYS = {"row_id", "partition", "features", "outcomes"}
_TEST_FEATURE_ROW_KEYS = {"row_id", "partition", "features"}
_TEST_OUTCOME_ROW_KEYS = {"row_id", "partition", "outcomes"}

_PREDICTION_ROW_KEYS = {"row_id", "partition", "predictions"}

_FORBIDDEN_FEATURE_KEYS = {
    "outcome",
    "outcomes",
    "label",
    "labels",
    "observed_outcome",
    "ground_truth",
    "y_true",
}


class StagedDisclosureError(ProtocolHoldError):
    """A fail-closed staged-disclosure protocol violation."""


@dataclass(frozen=True)
class AuthorityReference:
    """One exact, role-bound data authority reference."""

    name: str
    role: str
    path: Path
    sha256: str
    device: int
    inode: int


@dataclass(frozen=True)
class StagedActivation:
    """Validated activation metadata without any parsed data rows."""

    path: Path
    sha256: str
    feature_fields: tuple[str, ...]
    outcome_fields: tuple[str, ...]
    partition_row_set_sha256: Mapping[str, str]
    authorities: Mapping[str, AuthorityReference]
    nondata_authority_sha256: Mapping[str, str]
    nondata_projections: Mapping[str, Mapping[str, AuthorityReference]]


@dataclass(frozen=True)
class DevelopmentDisclosure:
    activation: StagedActivation
    train_rows: tuple[Mapping[str, Any], ...]
    development_rows: tuple[Mapping[str, Any], ...]
    capability_status: str
    training_ready: bool


@dataclass(frozen=True)
class FeatureDisclosure:
    activation: StagedActivation
    pre_test_freeze_sha256: str
    selected_method_id: str
    predictor_method_ids: tuple[str, ...]
    internal_test_rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class OutcomeDisclosure:
    activation: StagedActivation
    pre_test_freeze_sha256: str
    test_prediction_commit_sha256: str
    selected_method_id: str
    method_ids: tuple[str, ...]
    prediction_rows: tuple[Mapping[str, Any], ...]
    outcome_rows: tuple[Mapping[str, Any], ...]


def _hold(code: str, message: str, **details: Any) -> None:
    raise StagedDisclosureError(code, message, details=details or None)


def _strict_keys(value: object, expected: set[str], *, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        _hold("HOLD_STAGED_SCHEMA", f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        _hold(
            "HOLD_STAGED_SCHEMA",
            f"{label} keys differ from the strict schema",
            missing=sorted(expected - actual),
            unexpected=sorted(actual - expected),
        )
    return value


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        _hold(
            "HOLD_STAGED_SCHEMA",
            f"{label} must be 64 lowercase hexadecimal characters",
        )
    return value


def _exact_sign_gate(value: object, *, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) != SIGN_ACCURACY_GATE
    ):
        _hold(
            "HOLD_SIGN_GATE_MUTATION",
            f"{label} must remain exactly {SIGN_ACCURACY_GATE:.2f}",
            actual=value,
        )


def _unique_json_object(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _parse_json_object(raw: bytes, *, path: Path, label: str) -> dict:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        _hold(
            "HOLD_STAGED_JSON",
            f"{label} is not strict JSON: {exc}",
            path=str(path),
            error=str(exc),
        )
    if not isinstance(payload, dict):
        _hold("HOLD_STAGED_JSON", f"{label} must be a JSON object", path=str(path))
    return payload


def _parse_jsonl(raw: bytes, *, path: Path, label: str) -> list[dict]:
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        _hold(
            "HOLD_STAGED_JSON",
            f"{label} is not UTF-8 JSONL",
            path=str(path),
            error=str(exc),
        )
    lines = text.splitlines()
    if not lines:
        _hold("HOLD_STAGED_JSON", f"{label} contains no rows", path=str(path))
    rows: list[dict] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            _hold(
                "HOLD_STAGED_JSON",
                f"{label} contains a blank JSONL row",
                path=str(path),
                line_number=line_number,
            )
        try:
            row = json.loads(
                line,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_nonfinite_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            _hold(
                "HOLD_STAGED_JSON",
                f"{label} contains invalid strict JSONL: {exc}",
                path=str(path),
                line_number=line_number,
                error=str(exc),
            )
        if not isinstance(row, dict):
            _hold(
                "HOLD_STAGED_JSON",
                f"{label} row must be an object",
                path=str(path),
                line_number=line_number,
            )
        assert_ordinary_record(row, purpose=f"{label} line {line_number}")
        rows.append(row)
    return rows


def _resolve_reference(owner_path: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        _hold("HOLD_STAGED_SCHEMA", f"{label} path must be a non-empty string")
    raw = Path(value)
    candidate = raw if raw.is_absolute() else owner_path.parent / raw
    # ``abspath`` normalizes ``..`` without following symlinks.  Symlink
    # components are rejected independently below.
    return Path(os.path.abspath(os.fspath(candidate)))


def _assert_no_symlink_component(path: Path, *, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            _hold(
                "HOLD_STAGED_AUTHORITY_UNAVAILABLE",
                f"{label} path component does not exist",
                path=str(current),
            )
        except OSError as exc:
            _hold(
                "HOLD_STAGED_AUTHORITY_UNAVAILABLE",
                f"{label} path metadata is unavailable",
                path=str(current),
                error=str(exc),
            )
        if stat.S_ISLNK(metadata.st_mode):
            _hold(
                "HOLD_STAGED_PATH_ALIAS",
                f"{label} refuses a symlink path component",
                path=str(current),
            )


def _single_link_metadata(path: Path, *, label: str) -> os.stat_result:
    assert_ordinary_path(path, purpose=label)
    _assert_no_symlink_component(path, label=label)
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        _hold(
            "HOLD_STAGED_AUTHORITY_UNAVAILABLE",
            f"{label} metadata is unavailable",
            path=str(path),
            error=str(exc),
        )
    if not stat.S_ISREG(metadata.st_mode):
        _hold(
            "HOLD_STAGED_PATH_ALIAS",
            f"{label} must be a regular file",
            path=str(path),
        )
    if metadata.st_nlink != 1:
        _hold(
            "HOLD_STAGED_PATH_ALIAS",
            f"{label} must have exactly one hard link",
            path=str(path),
            link_count=metadata.st_nlink,
        )
    return metadata


def _assert_output_absent(path: Path | None, *, label: str) -> None:
    if path is None:
        return
    output = Path(os.path.abspath(os.fspath(path)))
    assert_ordinary_path(output, purpose=label)
    if output.exists() or output.is_symlink():
        _hold(
            "HOLD_STAGED_OUTPUT_EXISTS",
            f"{label} must not overwrite an existing path",
            path=str(output),
        )


def _field_list(value: object, *, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(item, str)
            or not SAFE_FIELD_ID.fullmatch(item)
            for item in value
        )
        or value != sorted(value)
        or len(value) != len(set(value))
    ):
        _hold(
            "HOLD_STAGED_SCHEMA",
            f"{label} must be a non-empty sorted unique list of safe field names",
        )
    return tuple(value)


def _method_list(value: object, *, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(item, str) or not SAFE_METHOD_ID.fullmatch(item)
            for item in value
        )
        or value != sorted(value)
        or len(value) != len(set(value))
    ):
        _hold(
            "HOLD_STAGED_SCHEMA",
            f"{label} must be a non-empty sorted unique method list",
        )
    return tuple(value)


def _partition_hashes(value: object, *, label: str) -> dict[str, str]:
    mapping = _strict_keys(value, _PARTITION_HASH_KEYS, label=label)
    return {
        partition: _digest(mapping[partition], label=f"{label}.{partition}")
        for partition in ("train", "development", "internal_test")
    }


def _validate_manifest_payload(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_sha256: str,
) -> StagedActivation:
    _strict_keys(payload, _MANIFEST_KEYS, label="staged activation manifest")
    assert_ordinary_record(payload, purpose="staged activation manifest")
    expected_scalars = {
        "schema_version": ACTIVATION_SCHEMA,
        "status": "FROZEN",
        "mode": "ORDINARY_ONLY",
        "selected_route": AUTHORIZED_ROUTE,
        "sign_gate_change_authorized": False,
        "ordinary_only": True,
        "sealed_access": False,
        "downstream_unlock": False,
        "overwrite_authorized": False,
    }
    for field, expected in expected_scalars.items():
        if payload.get(field) != expected:
            _hold(
                "HOLD_STAGED_SCHEMA",
                f"staged activation {field} must be {expected!r}",
                actual=payload.get(field),
            )
    _exact_sign_gate(payload.get("sign_accuracy_gate"), label="activation sign gate")

    feature_fields = _field_list(
        payload.get("model_feature_fields"), label="model_feature_fields"
    )
    outcome_fields = _field_list(
        payload.get("outcome_fields"), label="outcome_fields"
    )
    forbidden = _FORBIDDEN_FEATURE_KEYS | set(outcome_fields)
    collisions = sorted(set(feature_fields) & forbidden)
    if collisions:
        _hold(
            "HOLD_TEST_FEATURE_OUTCOME_LEAKAGE",
            "feature schema contains outcome-bearing field names",
            fields=collisions,
        )
    row_hashes = _partition_hashes(
        payload.get("partition_row_set_sha256"),
        label="partition_row_set_sha256",
    )
    if len(set(row_hashes.values())) != len(row_hashes):
        _hold(
            "HOLD_STAGED_ROW_SET_MISMATCH",
            "train, development, and internal-test row sets must be distinct",
        )

    authorities_payload = _strict_keys(
        payload.get("authorities"), set(AUTHORITY_ORDER), label="authorities"
    )
    authorities: dict[str, AuthorityReference] = {}
    paths: dict[str, str] = {}
    identities: dict[str, tuple[int, int]] = {}
    digests: dict[str, str] = {}
    for name in AUTHORITY_ORDER:
        reference = _strict_keys(
            authorities_payload[name],
            _AUTHORITY_REF_KEYS,
            label=f"authority {name}",
        )
        if reference.get("role") != AUTHORITY_ROLES[name]:
            _hold(
                "HOLD_STAGED_ROLE_BINDING",
                f"authority {name} has the wrong role",
                expected=AUTHORITY_ROLES[name],
                actual=reference.get("role"),
            )
        if reference.get("status") != "FROZEN":
            _hold(
                "HOLD_STAGED_ROLE_BINDING",
                f"authority {name} must declare status=FROZEN",
            )
        authority_path = _resolve_reference(
            manifest_path, reference.get("path"), label=f"authority {name}"
        )
        metadata = _single_link_metadata(authority_path, label=f"authority {name}")
        expected_sha256 = _digest(
            reference.get("sha256"), label=f"authority {name} SHA-256"
        )
        authorities[name] = AuthorityReference(
            name=name,
            role=AUTHORITY_ROLES[name],
            path=authority_path,
            sha256=expected_sha256,
            device=metadata.st_dev,
            inode=metadata.st_ino,
        )
        paths[name] = os.path.normcase(str(authority_path))
        identities[name] = (metadata.st_dev, metadata.st_ino)
        digests[name] = expected_sha256

    if len(set(paths.values())) != len(paths):
        _hold(
            "HOLD_STAGED_PATH_ALIAS",
            "data authorities must use three distinct paths",
            paths=paths,
        )
    if len(set(identities.values())) != len(identities):
        _hold(
            "HOLD_STAGED_PATH_ALIAS",
            "data authorities resolve to an inode alias",
            identities={key: list(value) for key, value in identities.items()},
        )
    if len(set(digests.values())) != len(digests):
        _hold(
            "HOLD_STAGED_ROLE_BINDING",
            "data authority roles must bind distinct byte digests",
            digests=digests,
        )
    manifest_metadata = _single_link_metadata(
        manifest_path, label="staged activation manifest"
    )
    manifest_identity = (manifest_metadata.st_dev, manifest_metadata.st_ino)
    if manifest_identity in set(identities.values()) or manifest_sha256 in set(
        digests.values()
    ):
        _hold(
            "HOLD_STAGED_PATH_ALIAS",
            "activation manifest cannot alias a data authority",
        )

    nondata_payload = _strict_keys(
        payload.get("nondata_authorities"),
        set(REQUIRED_NONDATA_AUTHORITY_HASHES),
        label="nondata_authorities",
    )
    nondata_authority_sha256: dict[str, str] = {}
    nondata_projections: dict[str, dict[str, AuthorityReference]] = {}
    all_paths = set(paths.values()) | {os.path.normcase(str(manifest_path))}
    all_identities = set(identities.values()) | {manifest_identity}
    all_digests = set(digests.values()) | {manifest_sha256}
    for name in REQUIRED_NONDATA_AUTHORITY_HASHES:
        candidate = nondata_payload[name]
        if not isinstance(candidate, Mapping) or set(candidate) != (
            _PROJECTED_NONDATA_AUTHORITY_KEYS
        ):
            _hold(
                "HOLD_REQUIRED_STAGE_SPECIFIC_NONDATA_PROJECTIONS",
                f"{name} must bind the exact development and feature projections",
            )
        projected = _strict_keys(
            candidate,
            _PROJECTED_NONDATA_AUTHORITY_KEYS,
            label=f"stage-projected nondata authority {name}",
        )
        if projected.get("status") != "FROZEN_WITH_STAGE_PROJECTIONS":
            _hold(
                "HOLD_REQUIRED_STAGE_SPECIFIC_NONDATA_PROJECTIONS",
                f"{name} lacks frozen stage-specific projections",
            )
        source_digest = _digest(
            projected.get("source_sha256"),
            label=f"nondata source authority {name} SHA-256",
        )
        if source_digest in all_digests:
            _hold(
                "HOLD_STAGED_ROLE_BINDING",
                f"nondata source authority {name} reuses another digest",
            )
        nondata_authority_sha256[name] = source_digest
        all_digests.add(source_digest)
        stage_refs: dict[str, AuthorityReference] = {}
        for stage, field in (
            ("development", "development_projection"),
            ("internal_test_features", "internal_test_feature_projection"),
        ):
            projection_candidate = projected.get(field)
            if not isinstance(projection_candidate, Mapping) or set(
                projection_candidate
            ) != _PROJECTION_REF_KEYS:
                _hold(
                    "HOLD_REQUIRED_STAGE_SPECIFIC_NONDATA_PROJECTIONS",
                    f"{name} lacks an exact {stage} projection reference",
                )
            projection = _strict_keys(
                projection_candidate,
                _PROJECTION_REF_KEYS,
                label=f"{name} {stage} projection",
            )
            if projection.get("status") != "FROZEN":
                _hold(
                    "HOLD_REQUIRED_STAGE_SPECIFIC_NONDATA_PROJECTIONS",
                    f"{name} {stage} projection is not frozen",
                )
            projection_path = _resolve_reference(
                manifest_path,
                projection.get("path"),
                label=f"{name} {stage} projection",
            )
            metadata = _single_link_metadata(
                projection_path, label=f"{name} {stage} projection"
            )
            projection_digest = _digest(
                projection.get("sha256"),
                label=f"{name} {stage} projection SHA-256",
            )
            normalized_path = os.path.normcase(str(projection_path))
            identity = (metadata.st_dev, metadata.st_ino)
            if (
                normalized_path in all_paths
                or identity in all_identities
                or projection_digest in all_digests
            ):
                _hold(
                    "HOLD_STAGED_PATH_ALIAS",
                    f"{name} {stage} projection aliases another authority",
                )
            stage_refs[stage] = AuthorityReference(
                name=f"{name}_{stage}_projection",
                role=stage,
                path=projection_path,
                sha256=projection_digest,
                device=metadata.st_dev,
                inode=metadata.st_ino,
            )
            all_paths.add(normalized_path)
            all_identities.add(identity)
            all_digests.add(projection_digest)
        nondata_projections[name] = stage_refs
    return StagedActivation(
        path=manifest_path,
        sha256=manifest_sha256,
        feature_fields=feature_fields,
        outcome_fields=outcome_fields,
        partition_row_set_sha256=row_hashes,
        authorities=authorities,
        nondata_authority_sha256=nondata_authority_sha256,
        nondata_projections=nondata_projections,
    )


def _load_activation(path: Path, expected_sha256: str) -> StagedActivation:
    manifest_path = Path(os.path.abspath(os.fspath(path)))
    expected = _digest(expected_sha256, label="activation manifest SHA-256")
    raw = read_exact_bytes(
        manifest_path,
        expected,
        purpose="staged activation manifest",
    )
    payload = _parse_json_object(
        raw, path=manifest_path, label="staged activation manifest"
    )
    return _validate_manifest_payload(
        payload,
        manifest_path=manifest_path,
        manifest_sha256=expected,
    )


def _projection_hashes(activation: StagedActivation) -> dict[str, str]:
    return {
        f"{name}_{stage}_projection": reference.sha256
        for name, stages in activation.nondata_projections.items()
        for stage, reference in stages.items()
    }


def _validate_projection_payload(
    payload: Mapping[str, Any],
    *,
    authority_name: str,
    source_sha256: str,
    disclosure_stage: str,
) -> None:
    _strict_keys(
        payload,
        _PROJECTION_PAYLOAD_KEYS,
        label=f"{authority_name} {disclosure_stage} projection payload",
    )
    expected = {
        "schema_version": (
            f"r3x_{authority_name}_{disclosure_stage}_summary_projection_v1"
        ),
        "status": "FROZEN",
        "authority_name": authority_name,
        "source_authority_sha256": source_sha256,
        "disclosure_stage": disclosure_stage,
        "contains_outcomes": False,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            _hold(
                "HOLD_REQUIRED_STAGE_SPECIFIC_NONDATA_PROJECTIONS",
                f"{authority_name} projection {field} binding mismatch",
            )
    expected_scope = (
        ["development", "train"]
        if disclosure_stage == "development"
        else ["internal_test_features"]
    )
    if payload.get("partition_scope") != expected_scope:
        _hold(
            "HOLD_REQUIRED_STAGE_SPECIFIC_NONDATA_PROJECTIONS",
            f"{authority_name} projection partition scope is not stage-safe",
        )
    count = payload.get("record_count")
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 10**9:
        _hold(
            "HOLD_REQUIRED_STAGE_SPECIFIC_NONDATA_PROJECTIONS",
            f"{authority_name} projection record_count is invalid",
        )
    content_digest = _digest(
        payload.get("content_commitment_sha256"),
        label=f"{authority_name} projection content commitment",
    )
    certificate_digest = _digest(
        payload.get("semantic_validation_certificate_sha256"),
        label=f"{authority_name} projection semantic certificate",
    )
    if len({source_sha256, content_digest, certificate_digest}) != 3:
        _hold(
            "HOLD_REQUIRED_STAGE_SPECIFIC_NONDATA_PROJECTIONS",
            f"{authority_name} projection commitments must be independent",
        )


def _validate_nondata_stage(
    activation: StagedActivation,
    *,
    disclosure_stage: str,
) -> None:
    if disclosure_stage not in {
        "development",
        "internal_test_features",
        "aggregate_only",
    }:
        _hold("HOLD_STAGED_NONDATA_AUTHORITY", "unknown nondata disclosure stage")

    if disclosure_stage == "aggregate_only":
        return
    for name in STAGE_PROJECTED_NONDATA_AUTHORITIES:
        try:
            reference = activation.nondata_projections[name][disclosure_stage]
        except KeyError:
            _hold(
                "HOLD_REQUIRED_STAGE_SPECIFIC_NONDATA_PROJECTIONS",
                f"{name} lacks the {disclosure_stage} projection",
            )
        raw = read_exact_bytes(
            reference.path,
            reference.sha256,
            purpose=f"{name} {disclosure_stage} projection",
        )
        payload = _parse_json_object(
            raw,
            path=reference.path,
            label=f"{name} {disclosure_stage} projection",
        )
        _validate_projection_payload(
            payload,
            authority_name=name,
            source_sha256=activation.nondata_authority_sha256[name],
            disclosure_stage=disclosure_stage,
        )


def _row_id(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        _hold("HOLD_STAGED_ROW_ID", f"{label} must be a SHA-256 row identity")
    return value


def _validate_finite_json(value: object, *, label: str) -> None:
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        _hold(
            "HOLD_STAGED_SCHEMA",
            f"{label} is not finite JSON",
            error=str(exc),
        )


def _validate_feature_mapping(
    value: object,
    *,
    feature_fields: Sequence[str],
    outcome_fields: Sequence[str],
    label: str,
) -> dict:
    mapping = _strict_keys(value, set(feature_fields), label=label)
    forbidden = _FORBIDDEN_FEATURE_KEYS | set(outcome_fields)

    def walk(current: object, path: str) -> None:
        if isinstance(current, Mapping):
            for key, nested in current.items():
                if str(key) in forbidden:
                    _hold(
                        "HOLD_TEST_FEATURE_OUTCOME_LEAKAGE",
                        "feature authority contains an outcome-bearing field",
                        field_path=f"{path}.{key}",
                    )
                walk(nested, f"{path}.{key}")
        elif isinstance(current, list):
            for index, nested in enumerate(current):
                walk(nested, f"{path}[{index}]")

    walk(mapping, label)
    _validate_finite_json(mapping, label=label)
    return dict(mapping)


def _validate_outcomes(
    value: object, *, outcome_fields: Sequence[str], label: str
) -> dict[str, float]:
    mapping = _strict_keys(value, set(outcome_fields), label=label)
    normalized: dict[str, float] = {}
    for field in outcome_fields:
        raw_value = mapping[field]
        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, (int, float))
            or not math.isfinite(float(raw_value))
        ):
            _hold(
                "HOLD_STAGED_SCHEMA",
                f"{label}.{field} must be a finite numeric outcome",
            )
        normalized[field] = float(raw_value)
    return normalized


def _row_set_sha256(row_ids: Sequence[str]) -> str:
    return canonical_sha256(sorted(row_ids))


def _assert_unique_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_sha256: str,
    label: str,
) -> tuple[str, ...]:
    ids = tuple(str(row["row_id"]) for row in rows)
    if len(ids) != len(set(ids)):
        _hold("HOLD_STAGED_DUPLICATE_ROW_ID", f"{label} row IDs are not unique")
    actual = _row_set_sha256(ids)
    if actual != expected_sha256:
        _hold(
            "HOLD_STAGED_ROW_SET_MISMATCH",
            f"{label} row set differs from the activation binding",
            expected=expected_sha256,
            actual=actual,
        )
    return ids


def _parse_train_development_rows(
    raw: bytes, *, path: Path, activation: StagedActivation
) -> tuple[tuple[dict, ...], tuple[dict, ...]]:
    rows = _parse_jsonl(raw, path=path, label="train/development authority")
    normalized = {"train": [], "development": []}
    for index, row in enumerate(rows):
        _strict_keys(
            row,
            _TRAIN_DEVELOPMENT_ROW_KEYS,
            label=f"train/development row {index}",
        )
        row_id = _row_id(row.get("row_id"), label=f"row {index} identity")
        partition = row.get("partition")
        if partition not in normalized:
            _hold(
                "HOLD_STAGED_PARTITION",
                "train/development authority contains an unknown partition",
                row_id=row_id,
                partition=partition,
            )
        features = _validate_feature_mapping(
            row.get("features"),
            feature_fields=activation.feature_fields,
            outcome_fields=activation.outcome_fields,
            label=f"train/development row {index} features",
        )
        outcomes = _validate_outcomes(
            row.get("outcomes"),
            outcome_fields=activation.outcome_fields,
            label=f"train/development row {index} outcomes",
        )
        normalized[partition].append(
            {
                "row_id": row_id,
                "partition": partition,
                "features": features,
                "outcomes": outcomes,
            }
        )
    all_ids = [str(row["row_id"]) for row in rows]
    if len(all_ids) != len(set(all_ids)):
        _hold(
            "HOLD_STAGED_DUPLICATE_ROW_ID",
            "train and development row identities must be globally unique",
        )
    for partition in ("train", "development"):
        if not normalized[partition]:
            _hold(
                "HOLD_STAGED_PARTITION",
                f"{partition} partition must contain at least one row",
            )
        _assert_unique_rows(
            normalized[partition],
            expected_sha256=activation.partition_row_set_sha256[partition],
            label=partition,
        )
    return tuple(normalized["train"]), tuple(normalized["development"])


def _parse_test_feature_rows(
    raw: bytes, *, path: Path, activation: StagedActivation
) -> tuple[dict, ...]:
    rows = _parse_jsonl(raw, path=path, label="internal-test feature authority")
    normalized: list[dict] = []
    for index, row in enumerate(rows):
        _strict_keys(row, _TEST_FEATURE_ROW_KEYS, label=f"test feature row {index}")
        row_id = _row_id(row.get("row_id"), label=f"test feature row {index} identity")
        if row.get("partition") != "internal_test":
            _hold(
                "HOLD_STAGED_PARTITION",
                "feature authority may contain only internal_test rows",
                row_id=row_id,
                partition=row.get("partition"),
            )
        features = _validate_feature_mapping(
            row.get("features"),
            feature_fields=activation.feature_fields,
            outcome_fields=activation.outcome_fields,
            label=f"test feature row {index} features",
        )
        normalized.append(
            {"row_id": row_id, "partition": "internal_test", "features": features}
        )
    _assert_unique_rows(
        normalized,
        expected_sha256=activation.partition_row_set_sha256["internal_test"],
        label="internal-test features",
    )
    return tuple(normalized)


def _parse_test_outcome_rows(
    raw: bytes, *, path: Path, activation: StagedActivation
) -> tuple[dict, ...]:
    rows = _parse_jsonl(raw, path=path, label="internal-test outcome authority")
    normalized: list[dict] = []
    for index, row in enumerate(rows):
        _strict_keys(row, _TEST_OUTCOME_ROW_KEYS, label=f"test outcome row {index}")
        row_id = _row_id(row.get("row_id"), label=f"test outcome row {index} identity")
        if row.get("partition") != "internal_test":
            _hold(
                "HOLD_STAGED_PARTITION",
                "outcome authority may contain only internal_test rows",
                row_id=row_id,
                partition=row.get("partition"),
            )
        outcomes = _validate_outcomes(
            row.get("outcomes"),
            outcome_fields=activation.outcome_fields,
            label=f"test outcome row {index} outcomes",
        )
        normalized.append(
            {"row_id": row_id, "partition": "internal_test", "outcomes": outcomes}
        )
    _assert_unique_rows(
        normalized,
        expected_sha256=activation.partition_row_set_sha256["internal_test"],
        label="internal-test outcomes",
    )
    return tuple(normalized)


def _authority_identities(activation: StagedActivation) -> set[tuple[int, int]]:
    return {
        (reference.device, reference.inode)
        for reference in (
            *activation.authorities.values(),
            *(
                reference
                for stages in activation.nondata_projections.values()
                for reference in stages.values()
            ),
        )
    }


def _assert_auxiliary_ref_is_separate(
    path: Path,
    sha256: str,
    *,
    activation: StagedActivation,
    label: str,
) -> None:
    metadata = _single_link_metadata(path, label=label)
    if (metadata.st_dev, metadata.st_ino) in _authority_identities(activation):
        _hold(
            "HOLD_STAGED_PATH_ALIAS",
            f"{label} aliases a staged authority",
            path=str(path),
        )
    authority_digests = {
        reference.sha256
        for reference in (
            *activation.authorities.values(),
            *(
                reference
                for stages in activation.nondata_projections.values()
                for reference in stages.values()
            ),
        )
    }
    if sha256 in authority_digests:
        _hold(
            "HOLD_STAGED_ROLE_BINDING",
            f"{label} reuses staged authority bytes",
            path=str(path),
        )


def _load_pre_test_freeze(
    path: Path,
    expected_sha256: str,
    *,
    activation: StagedActivation,
) -> tuple[str, tuple[str, ...]]:
    replayed = _replay_canonical_dag_commit(
        path,
        expected_sha256,
        state=PRE_TEST_FREEZE_COMMIT,
        activation=activation,
    )
    payload = replayed["payload"]
    methods = _method_list(
        payload.get("required_method_ids"),
        label="pre-test freeze required_method_ids",
    )
    selected = payload.get("selected_method_id")
    if selected not in methods:
        _hold(
            "HOLD_PRE_TEST_FREEZE",
            "selected_method_id is not in the frozen predictor method set",
        )
    return selected, methods


def _replay_canonical_dag_commit(
    path: Path,
    expected_sha256: str,
    *,
    state: str,
    activation: StagedActivation,
) -> dict:
    commit_path = Path(os.path.abspath(os.fspath(path)))
    digest = _digest(expected_sha256, label=f"{state} SHA-256")
    _assert_auxiliary_ref_is_separate(
        commit_path,
        digest,
        activation=activation,
        label=state,
    )
    relative = Path(STATE_RELATIVE_PATHS[state])
    run_root = commit_path
    for _ in relative.parts:
        run_root = run_root.parent
    if run_root.joinpath(*relative.parts) != commit_path:
        _hold(
            "HOLD_STAGED_DAG_PATH",
            f"{state} must occur at its canonical immutable DAG path",
            expected_relative_path=str(relative),
            actual_path=str(commit_path),
        )
    reference = {"relative_path": str(relative), "sha256": digest}
    try:
        if state == PRE_TEST_FREEZE_COMMIT:
            replayed = replay_pre_test_freeze_commit(
                run_root,
                reference,
                replay_data_authorities=False,
            )
        elif state == TEST_PREDICTION_COMMIT:
            replayed = replay_test_prediction_commit(
                run_root,
                reference,
                replay_data_authorities=False,
            )
        else:  # pragma: no cover - private helper has only two legal callers.
            _hold("HOLD_STAGED_DAG_PATH", "unsupported disclosure DAG state")
    except StagedCommitError as exc:
        _hold(
            "HOLD_STAGED_DAG_REPLAY",
            f"{state} immutable DAG replay failed: {exc}",
        )

    payload = replayed["payload"]
    expected_data_hashes = {
        name: activation.authorities[name].sha256 for name in AUTHORITY_ORDER
    }
    expected_nondata_hashes = dict(activation.nondata_authority_sha256)
    expected_projection_hashes = _projection_hashes(activation)
    if payload.get("activation_sha256") != activation.sha256:
        _hold(
            "HOLD_STAGED_DAG_BINDING",
            f"{state} activation SHA-256 binding mismatch",
        )
    if payload.get("authority_hashes") != dict(sorted(expected_data_hashes.items())):
        _hold(
            "HOLD_STAGED_DAG_BINDING",
            f"{state} data-authority hash binding mismatch",
        )
    if payload.get("nondata_authority_hashes") != dict(
        sorted(expected_nondata_hashes.items())
    ):
        _hold(
            "HOLD_STAGED_DAG_BINDING",
            f"{state} nondata-authority hash binding mismatch",
        )
    if payload.get("nondata_projection_hashes") != dict(
        sorted(expected_projection_hashes.items())
    ):
        _hold(
            "HOLD_STAGED_DAG_BINDING",
            f"{state} nondata projection hash binding mismatch",
        )
    _exact_sign_gate(payload.get("sign_accuracy_gate"), label=f"{state} sign gate")
    if payload.get("sign_gate_change_authorized") is not False:
        _hold(
            "HOLD_SIGN_GATE_MUTATION",
            f"{state} must keep sign_gate_change_authorized=false",
        )
    if payload.get("row_set_sha256") != activation.partition_row_set_sha256[
        "internal_test"
    ]:
        _hold(
            "HOLD_STAGED_ROW_SET_MISMATCH",
            f"{state} internal-test row-set binding mismatch",
        )
    return {**replayed, "run_root": run_root}


def _parse_prediction_rows(
    raw: bytes,
    *,
    path: Path,
    activation: StagedActivation,
    method_ids: Sequence[str],
) -> tuple[dict, ...]:
    rows = _parse_jsonl(raw, path=path, label="internal-test prediction artifact")
    normalized: list[dict] = []
    for index, row in enumerate(rows):
        _strict_keys(row, _PREDICTION_ROW_KEYS, label=f"prediction row {index}")
        row_id = _row_id(row.get("row_id"), label=f"prediction row {index} identity")
        if row.get("partition") != "internal_test":
            _hold(
                "HOLD_STAGED_PARTITION",
                "prediction artifact may contain only internal_test rows",
                row_id=row_id,
                partition=row.get("partition"),
            )
        predictions = _strict_keys(
            row.get("predictions"), set(method_ids), label=f"prediction row {index} values"
        )
        normalized_predictions: dict[str, float] = {}
        for method_id in method_ids:
            value = predictions[method_id]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                _hold(
                    "HOLD_TEST_PREDICTION_COMMIT",
                    "test predictions must be finite numbers",
                    row_id=row_id,
                    method_id=method_id,
                )
            normalized_predictions[method_id] = float(value)
        normalized.append(
            {
                "row_id": row_id,
                "partition": "internal_test",
                "predictions": normalized_predictions,
            }
        )
    _assert_unique_rows(
        normalized,
        expected_sha256=activation.partition_row_set_sha256["internal_test"],
        label="internal-test predictions",
    )
    return tuple(normalized)


def _load_test_prediction_commit(
    path: Path,
    expected_sha256: str,
    *,
    activation: StagedActivation,
) -> tuple[str, str, tuple[str, ...], tuple[dict, ...], dict]:
    replayed = _replay_canonical_dag_commit(
        path,
        expected_sha256,
        state=TEST_PREDICTION_COMMIT,
        activation=activation,
    )
    payload = replayed["payload"]
    method_ids = _method_list(
        payload.get("required_method_ids"), label="prediction required_method_ids"
    )
    selected = payload.get("selected_method_id")
    if selected not in method_ids:
        _hold(
            "HOLD_TEST_PREDICTION_COMMIT",
            "selected_method_id must occur in prediction method_ids",
        )
    freeze_ref = payload["prior_commit_refs"][PRE_TEST_FREEZE_COMMIT]
    freeze_sha = freeze_ref["sha256"]
    prediction_ref = payload.get("evaluation_predictions_ref")
    if not isinstance(prediction_ref, Mapping):
        _hold(
            "HOLD_TEST_PREDICTION_COMMIT",
            "evaluation_predictions_ref must be an immutable artifact reference",
        )
    prediction_sha = _digest(
        prediction_ref.get("sha256"), label="evaluation prediction SHA-256"
    )
    prediction_path = replayed["run_root"].joinpath(
        *Path(str(prediction_ref.get("relative_path") or "")).parts
    )
    _assert_auxiliary_ref_is_separate(
        prediction_path,
        prediction_sha,
        activation=activation,
        label="test prediction artifact",
    )
    commit_sha = _digest(expected_sha256, label="test prediction commit SHA-256")
    if prediction_sha in {commit_sha, freeze_sha}:
        _hold(
            "HOLD_STAGED_PATH_ALIAS",
            "prediction bytes must be distinct from protocol commit bytes",
        )
    prediction_raw = read_exact_bytes(
        prediction_path, prediction_sha, purpose="test prediction artifact"
    )
    prediction_rows = _parse_prediction_rows(
        prediction_raw,
        path=prediction_path,
        activation=activation,
        method_ids=method_ids,
    )
    return freeze_sha, selected, method_ids, prediction_rows, replayed


def validate_development_stage(
    activation_manifest_path: Path,
    activation_manifest_sha256: str,
    *,
    prospective_output_path: Path | None = None,
) -> DevelopmentDisclosure:
    """Read only activation metadata and train/development outcome bytes."""

    _assert_output_absent(
        prospective_output_path, label="development-stage prospective output"
    )
    activation = _load_activation(
        activation_manifest_path, activation_manifest_sha256
    )
    _validate_nondata_stage(activation, disclosure_stage="development")
    authority = activation.authorities[TRAIN_DEVELOPMENT_AUTHORITY]
    raw = read_exact_bytes(
        authority.path,
        authority.sha256,
        purpose="train/development outcome authority",
    )
    train_rows, development_rows = _parse_train_development_rows(
        raw, path=authority.path, activation=activation
    )
    return DevelopmentDisclosure(
        activation=activation,
        train_rows=train_rows,
        development_rows=development_rows,
        capability_status=(
            "STAGED_DISCLOSURE_SAFETY_VALIDATED_NOT_TRAINING_READY"
        ),
        training_ready=False,
    )


def disclose_internal_test_features(
    activation_manifest_path: Path,
    activation_manifest_sha256: str,
    *,
    pre_test_freeze_path: Path,
    pre_test_freeze_sha256: str,
    prospective_output_path: Path | None = None,
) -> FeatureDisclosure:
    """Read feature-only test bytes after predictor/selector freeze replay.

    Neither train/development bytes nor internal-test outcome bytes are opened
    by this function.
    """

    _assert_output_absent(
        prospective_output_path, label="feature-stage prospective output"
    )
    activation = _load_activation(
        activation_manifest_path, activation_manifest_sha256
    )
    selected, predictor_methods = _load_pre_test_freeze(
        pre_test_freeze_path,
        pre_test_freeze_sha256,
        activation=activation,
    )
    _validate_nondata_stage(
        activation, disclosure_stage="internal_test_features"
    )
    authority = activation.authorities[INTERNAL_TEST_FEATURE_AUTHORITY]
    raw = read_exact_bytes(
        authority.path,
        authority.sha256,
        purpose="internal-test feature-only authority",
    )
    rows = _parse_test_feature_rows(raw, path=authority.path, activation=activation)
    return FeatureDisclosure(
        activation=activation,
        pre_test_freeze_sha256=_digest(
            pre_test_freeze_sha256, label="pre-test freeze SHA-256"
        ),
        selected_method_id=selected,
        predictor_method_ids=predictor_methods,
        internal_test_rows=rows,
    )


def disclose_internal_test_outcomes(
    activation_manifest_path: Path,
    activation_manifest_sha256: str,
    *,
    test_prediction_commit_path: Path,
    test_prediction_commit_sha256: str,
    prospective_output_path: Path | None = None,
) -> OutcomeDisclosure:
    """Read outcome-only test bytes after exact prediction-commit replay.

    This function does not open train/development or feature-authority bytes.
    It first validates the complete pre-test freeze chain and prediction row
    set; only then is the outcome authority opened.
    """

    _assert_output_absent(
        prospective_output_path, label="outcome-stage prospective output"
    )
    activation = _load_activation(
        activation_manifest_path, activation_manifest_sha256
    )
    freeze_sha, selected, method_ids, prediction_rows, prediction_replay = (
        _load_test_prediction_commit(
            test_prediction_commit_path,
            test_prediction_commit_sha256,
            activation=activation,
        )
    )
    _validate_nondata_stage(activation, disclosure_stage="aggregate_only")
    authority = activation.authorities[INTERNAL_TEST_OUTCOME_AUTHORITY]
    run_root = prediction_replay["run_root"]
    try:
        outcome_relative_path = authority.path.relative_to(run_root)
    except ValueError:
        _hold(
            "HOLD_STAGED_DAG_PATH",
            "outcome authority must be beneath the immutable DAG run root",
            outcome_path=str(authority.path),
            run_root=str(run_root),
        )
    prediction_payload = prediction_replay["payload"]
    try:
        release = write_ordinary_test_outcome_release_record_exclusive(
            run_root,
            bindings=extract_staged_bindings(prediction_payload),
            development_commit_ref=prediction_payload["prior_commit_refs"][
                "DEVELOPMENT_COMMIT"
            ],
            pre_test_freeze_commit_ref=prediction_payload["prior_commit_refs"][
                PRE_TEST_FREEZE_COMMIT
            ],
            test_prediction_commit_ref=prediction_replay["ref"],
            outcome_authority_ref={
                "relative_path": outcome_relative_path.as_posix(),
                "sha256": authority.sha256,
            },
            access_assertions={
                "internal_test_feature_accessed": True,
                "internal_test_outcome_accessed": True,
            },
        )
    except StagedCommitError as exc:
        _hold(
            "HOLD_STAGED_DAG_REPLAY",
            f"ordinary outcome release DAG transition failed: {exc}",
        )
    raw = release["outcome_bytes"]
    outcome_rows = _parse_test_outcome_rows(
        raw, path=authority.path, activation=activation
    )
    prediction_ids = tuple(row["row_id"] for row in prediction_rows)
    outcome_ids = tuple(row["row_id"] for row in outcome_rows)
    if set(prediction_ids) != set(outcome_ids):
        # Both row sets are independently bound to the activation, but retain
        # the direct comparison as defense in depth if the digest algorithm is
        # ever changed in a later schema.
        _hold(
            "HOLD_STAGED_ROW_SET_MISMATCH",
            "prediction and outcome row identities differ",
        )
    return OutcomeDisclosure(
        activation=activation,
        pre_test_freeze_sha256=freeze_sha,
        test_prediction_commit_sha256=_digest(
            test_prediction_commit_sha256,
            label="test prediction commit SHA-256",
        ),
        selected_method_id=selected,
        method_ids=method_ids,
        prediction_rows=prediction_rows,
        outcome_rows=outcome_rows,
    )


__all__ = [
    "ACTIVATION_SCHEMA",
    "AUTHORITY_ORDER",
    "AUTHORITY_ROLES",
    "DevelopmentDisclosure",
    "DEVELOPMENT_STAGE_AUTHORITY_NAMES",
    "FeatureDisclosure",
    "INTERNAL_TEST_FEATURE_AUTHORITY",
    "INTERNAL_TEST_OUTCOME_AUTHORITY",
    "OutcomeDisclosure",
    "PRE_TEST_FREEZE_SCHEMA",
    "REQUIRED_DEVELOPMENT_AUTHORITY_DIGEST_KEYS",
    "SIGN_ACCURACY_GATE",
    "StagedActivation",
    "StagedDisclosureError",
    "TEST_PREDICTION_COMMIT_SCHEMA",
    "TRAIN_DEVELOPMENT_AUTHORITY",
    "disclose_internal_test_features",
    "disclose_internal_test_outcomes",
    "validate_development_stage",
]
