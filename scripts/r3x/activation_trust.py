"""Development-capability-safe v3 activation trust for ordinary R3X.

The staged-disclosure activation contains generic, disclosure-safe summary
projections.  Those summaries are *not* typed row authorities and their
producer-reported content or semantic commitments cannot be used as a
substitute for the rich projections in :mod:`stage_row_projections`.

An authority-preparation process creates one canonical v3 development
activation that independently binds and replays:

* the existing full staged activation and its train/development data authority
  during preparation only;
* the pre-training row-firewall certificate during preparation only;
* all seven typed-safe development authority artifacts; and
* all three rich typed development row projections.

Writing is construction-only: callers provide immutable artifact references,
not a proposed trust payload.  Every preparation reference is exact-loaded,
the rich row projections are rebuilt from the actually disclosed
train/development rows, and the resulting binding is published exclusively.

The published development activation deliberately omits the old activation
reference, every future-stage path, every internal-test data/projection hash,
the firewall path/hash, and every commitment-key path/hash/byte.  It carries
only a domain-separated HMAC commitment produced with a privileged, externally
pre-bound 32-byte key.  A bare SHA-256 of the firewall certificate would be
binding but not hiding against a low-entropy candidate set, so it is expressly
not accepted here.  The runtime loader never reads the HMAC key and can replay
only train/development-safe inputs; it does not call any old staged
feature/outcome disclosure function.  A full old ``StagedActivation`` is never
returned as a development capability.

The current upstream authorities still expose honest external trust
gaps: the code snapshot is a caller mapping rather than an independently
exact-loaded authority, the execution-qualification certificate is only hash
bound, and the firewall producer's internal-test-feature replay is not
re-executed in this development-only process.  Therefore this v3 artifact
closes the activation/certificate and rich-projection binding, but remains
``training_ready=False``.  The preparation writer does not open internal-test
outcomes, and the runtime loader cannot even name them.  Neither consumes
sealed data, changes the 0.60 sign gate, or unlocks a downstream phase.
"""

from __future__ import annotations

import errno
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import stat
from pathlib import Path
from typing import Any, Mapping

from .artifact_io import ArtifactIOError, canonical_json_bytes
from .protocol import AUTHORIZED_ROUTE, ProtocolHoldError, read_exact_bytes
from .stage_row_projections import (
    DEVELOPMENT_STAGE,
    PROJECTION_HASH_KEYS,
    ROW_AUTHORITIES,
    SOURCE_COMPONENT_SPLIT_MANIFEST,
    build_development_row_projections,
    canonical_stage_row_projection_bytes,
    load_pretraining_firewall_certificate_exact,
    load_stage_row_projection_exact,
    stage_projection_hashes,
)
from .staged_commits import (
    REQUIRED_NONDATA_AUTHORITY_HASHES,
    TRAIN_DEVELOPMENT_AUTHORITY,
)
from .staged_disclosure import (
    AuthorityReference,
    DevelopmentDisclosure,
    StagedActivation,
    _parse_train_development_rows,
    validate_development_stage,
)
from .typed_safe_authorities import (
    DEVELOPMENT_PROJECTION_HASH_KEYS,
    SUPPORTED_AUTHORITIES,
    load_typed_safe_bundle_exact,
)
from .typed_staged_rows import (
    EXACT_FEATURE_FIELDS,
    EXACT_OUTCOME_FIELDS,
    normalize_development_disclosure,
)


DEVELOPMENT_ACTIVATION_TRUST_SCHEMA = "r3x_development_activation_trust_v3"
DEVELOPMENT_ACTIVATION_TRUST_ROLE = "ORDINARY_DEVELOPMENT_ONLY_ACTIVATION_TRUST_BINDING"
DEVELOPMENT_ACTIVATION_TRUST_STATUS = (
    "DEVELOPMENT_ACTIVATION_TRUST_V3_BOUND_HOLD_PENDING_EXTERNAL_TRUST_ROOTS"
)
DEVELOPMENT_ACTIVATION_TRUST_RELATIVE_PATH = Path(
    "development/activation_trust/development_activation_trust_v3.json"
)
FUTURE_STAGE_COMMITMENT_KEY_RELATIVE_PATH = Path(
    "authority_preparation/activation_trust/future_stage_commitment_hmac_key.bin"
)
FUTURE_STAGE_COMMITMENT_SCHEME = (
    "HMAC-SHA256-DOMAIN-SEPARATED-FIREWALL-CERTIFICATE-SHA256-V1"
)
FUTURE_STAGE_COMMITMENT_DOMAIN = (
    b"r3x-development-activation-trust-v3\x00"
    b"future-stage-firewall-certificate-sha256\x00"
)
FUTURE_STAGE_COMMITMENT_KEY_BYTES = 32

SIGN_ACCURACY_GATE = 0.60
MAX_TRUST_ARTIFACT_BYTES = 2 * 1024 * 1024
HEX64 = re.compile(r"^[0-9a-f]{64}$")

REF_KEYS = {"relative_path", "sha256"}
TRUST_BLOCKERS = (
    "DEVELOPMENT_FILESYSTEM_CAPABILITY_SEPARATION_NOT_ENFORCED",
    "EXECUTION_QUALIFICATION_CERTIFICATE_NOT_EXACT_REPLAYED",
    "FIREWALL_PRODUCER_CAPABILITY_NOT_RUNTIME_REEXECUTED",
    "INDEPENDENT_CODE_SNAPSHOT_AUTHORITY_NOT_EXACT_LOADED",
    "RUNNER_NOT_YET_BOUND_TO_DEVELOPMENT_ACTIVATION_TRUST_V3_SHA256",
    "SOURCE_AUTHORITY_ROOTS_NOT_INDEPENDENTLY_EXACT_LOADED",
    "FUTURE_STAGE_HMAC_KEY_PREBINDING_NOT_INDEPENDENTLY_ATTESTED",
    "FUTURE_STAGE_HMAC_KEY_SECRECY_NOT_OS_ENFORCED",
)

TRUST_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "capability_scope",
    "selected_route",
    "train_development_authority_ref",
    "typed_authority_refs",
    "development_row_projection_refs",
    "partition_row_set_sha256",
    "train_development_outcomes_sha256",
    "source_authority_sha256",
    "development_projection_hashes",
    "opaque_future_stage_commitment",
    "activation_firewall_prebinding_status",
    "staged_generic_projection_role",
    "rich_row_projection_role",
    "generic_summary_used_as_rich_projection",
    "legacy_full_staged_activation_training_ready",
    "legacy_full_staged_activation_ref_exposed",
    "declared_artifact_bytes_exact_replayed",
    "development_rows_rebuilt_from_train_authority",
    "firewall_certificate_exact_bytes_cross_bound",
    "firewall_full_replay_reexecuted_at_binding",
    "independent_code_snapshot_authority_exact_replayed",
    "execution_qualification_certificate_exact_replayed",
    "semantic_dependency_chain_complete",
    "future_stage_access_authorized",
    "internal_test_feature_access",
    "internal_test_outcome_access",
    "sign_accuracy_gate",
    "sign_gate_change_authorized",
    "ordinary_only",
    "training_ready",
    "sealed_access",
    "downstream_unlock",
    "overwrite_authorized",
    "blockers",
}
FUTURE_STAGE_COMMITMENT_KEYS = {"scheme", "commitment"}


class ActivationTrustError(ProtocolHoldError):
    """A structured activation-trust HOLD."""


def _hold(code: str, message: str, **details: Any) -> None:
    raise ActivationTrustError(code, message, details=details or None)


def _strict_keys(value: object, expected: set[str], *, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        _hold("HOLD_ACTIVATION_TRUST_SCHEMA", f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        _hold(
            "HOLD_ACTIVATION_TRUST_SCHEMA",
            f"{label} keys differ from the exact schema",
            missing=sorted(expected - actual),
            unexpected=sorted(actual - expected),
        )
    return value


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        _hold(
            "HOLD_ACTIVATION_TRUST_SCHEMA",
            f"{label} must be lowercase SHA-256",
        )
    return value


def _exact_gate(value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) != SIGN_ACCURACY_GATE
    ):
        _hold(
            "HOLD_SIGN_GATE_MUTATION",
            "activation trust sign gate must remain exactly 0.60",
        )


def _parse_json_object(raw: bytes, *, label: str) -> dict:
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
        _hold("HOLD_ACTIVATION_TRUST_JSON", f"{label} is not strict JSON: {exc}")
    if not isinstance(payload, dict):
        _hold("HOLD_ACTIVATION_TRUST_JSON", f"{label} must be a JSON object")
    return payload


def _checked_run_root(run_root: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(run_root)))
    if root.is_symlink() or not root.is_dir():
        _hold(
            "HOLD_ACTIVATION_TRUST_PATH",
            "run root must be an existing real directory",
        )
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        _hold("HOLD_ACTIVATION_TRUST_PATH", f"run root is unavailable: {exc}")
    if resolved != root:
        _hold("HOLD_ACTIVATION_TRUST_PATH", "run root cannot traverse symlinks")
    return root


def _normalize_ref(
    run_root: Path,
    value: object,
    *,
    label: str,
    required_prefix: tuple[str, ...] | None = None,
) -> dict[str, str]:
    ref = _strict_keys(value, REF_KEYS, label=label)
    relative_value = ref.get("relative_path")
    if not isinstance(relative_value, str) or not relative_value:
        _hold("HOLD_ACTIVATION_TRUST_PATH", f"{label} path is invalid")
    relative = Path(relative_value)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or relative.as_posix() != relative_value
        or any(part in {"", "."} for part in relative.parts)
    ):
        _hold(
            "HOLD_ACTIVATION_TRUST_PATH",
            f"{label} must use a normalized relative path",
        )
    lowered = {part.lower() for part in relative.parts}
    if "sealed" in lowered or "restricted" in lowered:
        _hold("HOLD_ACTIVATION_TRUST_PROTECTED", f"{label} names protected support")
    if any("internal_test_outcome" in part.lower() for part in relative.parts):
        _hold(
            "HOLD_ACTIVATION_TRUST_PROTECTED",
            f"{label} cannot reference internal-test outcomes",
        )
    if (
        required_prefix is not None
        and relative.parts[: len(required_prefix)] != required_prefix
    ):
        _hold(
            "HOLD_ACTIVATION_TRUST_PATH",
            f"{label} is outside its required namespace",
            required_prefix=list(required_prefix),
        )
    candidate = run_root.joinpath(*relative.parts)
    try:
        candidate.relative_to(run_root)
    except ValueError:  # pragma: no cover - normalized relative path is defensive.
        _hold("HOLD_ACTIVATION_TRUST_PATH", f"{label} escapes the run root")
    return {
        "relative_path": relative.as_posix(),
        "sha256": _digest(ref.get("sha256"), label=f"{label} SHA-256"),
    }


def _normalize_ref_map(
    run_root: Path,
    value: object,
    *,
    expected_names: tuple[str, ...],
    label: str,
    required_prefix: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    refs = _strict_keys(value, set(expected_names), label=label)
    normalized = {
        name: _normalize_ref(
            run_root,
            refs[name],
            label=f"{label}.{name}",
            required_prefix=required_prefix,
        )
        for name in expected_names
    }
    paths = [ref["relative_path"] for ref in normalized.values()]
    hashes = [ref["sha256"] for ref in normalized.values()]
    if len(paths) != len(set(paths)) or len(hashes) != len(set(hashes)):
        _hold(
            "HOLD_ACTIVATION_TRUST_ALIAS",
            f"{label} refs must bind distinct paths and bytes",
        )
    return normalized


def _all_dependency_refs(payload: Mapping) -> list[Mapping]:
    return [
        payload["train_development_authority_ref"],
        *payload["typed_authority_refs"].values(),
        *payload["development_row_projection_refs"].values(),
    ]


def validate_development_activation_trust_payload(payload: Mapping) -> dict:
    """Validate the development-safe shape; this alone never grants trust."""

    raw = _strict_keys(
        payload,
        TRUST_KEYS,
        label="development activation trust binding",
    )
    exact = {
        "schema_version": DEVELOPMENT_ACTIVATION_TRUST_SCHEMA,
        "artifact_role": DEVELOPMENT_ACTIVATION_TRUST_ROLE,
        "status": DEVELOPMENT_ACTIVATION_TRUST_STATUS,
        "capability_scope": "TRAIN_DEVELOPMENT_ONLY",
        "selected_route": AUTHORIZED_ROUTE,
        "activation_firewall_prebinding_status": (
            "V3_EXTENSION_HMAC_BINDS_FIREWALL_CERTIFICATE_SHA256_"
            "WITH_PRIVILEGED_SECRET_KEY"
        ),
        "staged_generic_projection_role": ("DISCLOSURE_ONLY_NOT_RICH_ROW_AUTHORITY"),
        "rich_row_projection_role": "EXACT_TYPED_DEVELOPMENT_ROW_AUTHORITY",
        "generic_summary_used_as_rich_projection": False,
        "legacy_full_staged_activation_training_ready": False,
        "legacy_full_staged_activation_ref_exposed": False,
        "declared_artifact_bytes_exact_replayed": True,
        "development_rows_rebuilt_from_train_authority": True,
        "firewall_certificate_exact_bytes_cross_bound": True,
        "firewall_full_replay_reexecuted_at_binding": False,
        "independent_code_snapshot_authority_exact_replayed": False,
        "execution_qualification_certificate_exact_replayed": False,
        "semantic_dependency_chain_complete": False,
        "future_stage_access_authorized": False,
        "internal_test_feature_access": False,
        "internal_test_outcome_access": False,
        "sign_gate_change_authorized": False,
        "ordinary_only": True,
        "training_ready": False,
        "sealed_access": False,
        "downstream_unlock": False,
        "overwrite_authorized": False,
        "blockers": list(TRUST_BLOCKERS),
    }
    for field, expected in exact.items():
        value = raw.get(field)
        matches = (
            value is expected
            if isinstance(expected, bool)
            else (type(value) is type(expected) and value == expected)
        )
        if not matches:
            _hold(
                "HOLD_ACTIVATION_TRUST_SCHEMA",
                f"activation trust {field} differs from the exact policy",
            )
    _exact_gate(raw.get("sign_accuracy_gate"))

    # The only data path a development process may receive is the ordinary
    # train/development authority.  The old full activation and firewall path
    # are intentionally absent from TRUST_KEYS.
    train_ref = _strict_keys(
        raw["train_development_authority_ref"],
        REF_KEYS,
        label="train_development_authority_ref",
    )
    _digest(train_ref.get("sha256"), label="train/development authority ref")
    if not isinstance(train_ref.get("relative_path"), str):
        _hold(
            "HOLD_ACTIVATION_TRUST_PATH",
            "train/development authority ref path is invalid",
        )

    typed_refs = _strict_keys(
        raw["typed_authority_refs"],
        set(SUPPORTED_AUTHORITIES),
        label="typed_authority_refs",
    )
    row_refs = _strict_keys(
        raw["development_row_projection_refs"],
        set(ROW_AUTHORITIES),
        label="development_row_projection_refs",
    )
    for name, ref in (*typed_refs.items(), *row_refs.items()):
        checked = _strict_keys(ref, REF_KEYS, label=f"artifact ref {name}")
        _digest(checked.get("sha256"), label=f"artifact ref {name}")
        if not isinstance(checked.get("relative_path"), str):
            _hold("HOLD_ACTIVATION_TRUST_PATH", f"artifact ref {name} path is invalid")

    train_sha = _digest(
        raw["train_development_outcomes_sha256"],
        label="train/development outcomes authority",
    )
    if train_ref["sha256"] != train_sha:
        _hold(
            "HOLD_ACTIVATION_TRUST_BINDING",
            "train/development ref and digest binding disagree",
        )
    partition_hashes = _strict_keys(
        raw["partition_row_set_sha256"],
        {"train", "development"},
        label="partition_row_set_sha256",
    )
    normalized_partition_hashes = [
        _digest(partition_hashes[name], label=f"{name} row set")
        for name in ("train", "development")
    ]
    if len(set(normalized_partition_hashes)) != 2:
        _hold(
            "HOLD_ACTIVATION_TRUST_ALIAS",
            "train and development row-set commitments must be distinct",
        )
    source_hashes = _strict_keys(
        raw["source_authority_sha256"],
        set(REQUIRED_NONDATA_AUTHORITY_HASHES),
        label="source_authority_sha256",
    )
    normalized_sources = [
        _digest(source_hashes[name], label=f"source authority {name}")
        for name in REQUIRED_NONDATA_AUTHORITY_HASHES
    ]
    if len(normalized_sources) != len(set(normalized_sources)):
        _hold(
            "HOLD_ACTIVATION_TRUST_ALIAS",
            "source authorities must bind distinct bytes",
        )

    development_hashes = _strict_keys(
        raw["development_projection_hashes"],
        set(DEVELOPMENT_PROJECTION_HASH_KEYS),
        label="development_projection_hashes",
    )
    projection_values = [
        _digest(value, label=name) for name, value in development_hashes.items()
    ]
    if len(projection_values) != len(set(projection_values)):
        _hold(
            "HOLD_ACTIVATION_TRUST_ALIAS",
            "rich development projection hashes must be distinct",
        )
    expected_development_refs = {
        PROJECTION_HASH_KEYS[DEVELOPMENT_STAGE][name]: row_refs[name]["sha256"]
        for name in ROW_AUTHORITIES
    }
    if dict(development_hashes) != expected_development_refs:
        _hold(
            "HOLD_ACTIVATION_TRUST_BINDING",
            "rich development projection refs disagree with their hash binding",
        )

    dependency_refs = _all_dependency_refs(raw)
    dependency_paths = [ref["relative_path"] for ref in dependency_refs]
    dependency_hashes = [ref["sha256"] for ref in dependency_refs]
    if len(dependency_paths) != len(set(dependency_paths)) or len(
        dependency_hashes
    ) != len(set(dependency_hashes)):
        _hold(
            "HOLD_ACTIVATION_TRUST_ALIAS",
            "dependency refs must bind distinct paths and artifact bytes",
        )
    future_commitment = _strict_keys(
        raw["opaque_future_stage_commitment"],
        FUTURE_STAGE_COMMITMENT_KEYS,
        label="opaque future-stage commitment",
    )
    if future_commitment.get("scheme") != FUTURE_STAGE_COMMITMENT_SCHEME:
        _hold(
            "HOLD_ACTIVATION_TRUST_SCHEMA",
            "future-stage commitment scheme differs from the exact policy",
        )
    commitment = _digest(
        future_commitment.get("commitment"),
        label="opaque future-stage commitment",
    )
    if train_sha in dependency_hashes[1:] or set(projection_values) & set(
        normalized_sources
    ):
        _hold(
            "HOLD_ACTIVATION_TRUST_ALIAS",
            "data, source, projection, and manifest roles cannot alias bytes",
        )
    if commitment in {
        train_sha,
        *dependency_hashes,
        *normalized_sources,
        *projection_values,
    }:
        _hold(
            "HOLD_ACTIVATION_TRUST_ALIAS",
            "future-stage commitment cannot alias a disclosed artifact digest",
        )
    return dict(raw)


def canonical_development_activation_trust_bytes(payload: Mapping) -> bytes:
    try:
        raw = canonical_json_bytes(
            validate_development_activation_trust_payload(payload)
        )
    except ArtifactIOError as exc:
        _hold("HOLD_ACTIVATION_TRUST_JSON", str(exc))
    if len(raw) > MAX_TRUST_ARTIFACT_BYTES:
        _hold("HOLD_ACTIVATION_TRUST_SCHEMA", "activation trust artifact is too large")
    return raw


def _dependency_error(label: str, exc: Exception) -> None:
    if isinstance(exc, ActivationTrustError):
        raise exc
    _hold(
        "HOLD_ACTIVATION_TRUST_DEPENDENCY",
        f"{label} exact replay failed: {exc}",
        dependency=label,
    )


def _build_future_stage_hmac_commitment(
    run_root: Path,
    *,
    firewall_certificate_sha256: str,
    commitment_key_sha256: str,
) -> dict[str, str]:
    """Bind a privileged certificate without exposing a guess-checking hash.

    The key lives at one fixed authority-preparation path and is exact-read by
    its externally pre-bound SHA-256.  Neither this path nor the key digest is
    represented in the development artifact.  Runtime replay intentionally
    validates only the HMAC shape; it never mounts or reads this key.
    """

    root = _checked_run_root(run_root)
    certificate_sha = _digest(
        firewall_certificate_sha256,
        label="future-stage firewall certificate",
    )
    expected_key_sha = _digest(
        commitment_key_sha256,
        label="future-stage commitment key",
    )
    parent_parts = FUTURE_STAGE_COMMITMENT_KEY_RELATIVE_PATH.parts[:-1]
    parent_descriptor = _open_directory_beneath_nofollow(root, parent_parts)
    try:
        raw_key = _read_regular_file_at_exact(
            parent_descriptor,
            FUTURE_STAGE_COMMITMENT_KEY_RELATIVE_PATH.name,
            expected_key_sha,
        )
        _verify_namespace_binding(root, parent_parts, parent_descriptor)
    finally:
        os.close(parent_descriptor)
    if len(raw_key) != FUTURE_STAGE_COMMITMENT_KEY_BYTES:
        _hold(
            "HOLD_ACTIVATION_TRUST_COMMITMENT_KEY",
            "future-stage HMAC key must be exactly 32 bytes",
        )

    # A mutable buffer limits the lifetime of the live Python key material.
    # This is best-effort process hygiene, not a claim of OS memory erasure.
    key = bytearray(raw_key)
    del raw_key
    try:
        commitment = hmac.new(
            key,
            FUTURE_STAGE_COMMITMENT_DOMAIN + bytes.fromhex(certificate_sha),
            digestmod=hashlib.sha256,
        ).hexdigest()
    finally:
        key[:] = b"\x00" * len(key)
    return {
        "scheme": FUTURE_STAGE_COMMITMENT_SCHEME,
        "commitment": commitment,
    }


def _prepare_development_activation_payload(
    run_root: Path,
    *,
    activation_manifest_ref: Mapping,
    firewall_certificate_ref: Mapping,
    typed_authority_refs: Mapping,
    development_row_projection_refs: Mapping,
    trusted_code_snapshot: Mapping | None,
    future_stage_commitment_key_sha256: str,
) -> dict:
    root = _checked_run_root(run_root)
    activation_ref = _normalize_ref(
        root,
        activation_manifest_ref,
        label="activation manifest ref",
    )
    firewall_ref = _normalize_ref(
        root,
        firewall_certificate_ref,
        label="firewall certificate ref",
        required_prefix=("authority_preparation", "firewall"),
    )
    typed_refs = _normalize_ref_map(
        root,
        typed_authority_refs,
        expected_names=SUPPORTED_AUTHORITIES,
        label="typed authority refs",
        required_prefix=("development", "authorities"),
    )
    row_refs = _normalize_ref_map(
        root,
        development_row_projection_refs,
        expected_names=ROW_AUTHORITIES,
        label="rich development row projection refs",
        required_prefix=("development", "row_authorities"),
    )
    all_refs = [activation_ref, firewall_ref, *typed_refs.values(), *row_refs.values()]
    paths = [ref["relative_path"] for ref in all_refs]
    hashes = [ref["sha256"] for ref in all_refs]
    if len(paths) != len(set(paths)) or len(hashes) != len(set(hashes)):
        _hold(
            "HOLD_ACTIVATION_TRUST_ALIAS",
            "all activation trust dependencies must use distinct paths and bytes",
        )

    try:
        disclosure = validate_development_stage(
            root / activation_ref["relative_path"],
            activation_ref["sha256"],
        )
    except Exception as exc:  # exact upstream error is wrapped with dependency role.
        _dependency_error("staged activation", exc)
    activation = disclosure.activation
    train_sha = activation.authorities[TRAIN_DEVELOPMENT_AUTHORITY].sha256
    try:
        typed_development = normalize_development_disclosure(disclosure)
    except Exception as exc:
        _dependency_error("typed train/development rows", exc)

    row_source_hashes = {
        name: activation.nondata_authority_sha256[name] for name in ROW_AUTHORITIES
    }
    rich_projections: dict[str, dict] = {}
    for name in ROW_AUTHORITIES:
        ref = row_refs[name]
        try:
            rich_projections[name] = load_stage_row_projection_exact(
                root,
                root / ref["relative_path"],
                ref["sha256"],
                expected_authority_name=name,
                expected_stage=DEVELOPMENT_STAGE,
                expected_source_authority_sha256=row_source_hashes[name],
                expected_data_authority_sha256=train_sha,
            )
        except Exception as exc:
            _dependency_error(f"rich development projection {name}", exc)
    actual_development_hashes = stage_projection_hashes(
        rich_projections,
        stage=DEVELOPMENT_STAGE,
    )
    expected_ref_hashes = {
        PROJECTION_HASH_KEYS[DEVELOPMENT_STAGE][name]: row_refs[name]["sha256"]
        for name in ROW_AUTHORITIES
    }
    if actual_development_hashes != expected_ref_hashes:
        _hold(
            "HOLD_ACTIVATION_TRUST_BINDING",
            "rich development projection canonical hashes disagree with refs",
        )

    split_seed = rich_projections[SOURCE_COMPONENT_SPLIT_MANIFEST]["split_policy"][
        "seed"
    ]
    try:
        rebuilt_projections = build_development_row_projections(
            typed_development,
            source_authority_sha256=row_source_hashes,
            train_development_outcomes_sha256=train_sha,
            split_seed=split_seed,
        )
    except Exception as exc:
        _dependency_error("rich projections rebuilt from train/development rows", exc)
    for name in ROW_AUTHORITIES:
        if canonical_stage_row_projection_bytes(rich_projections[name]) != (
            canonical_stage_row_projection_bytes(rebuilt_projections[name])
        ):
            _hold(
                "HOLD_ACTIVATION_TRUST_RICH_REPLAY",
                "rich projection differs from the typed train/development replay",
                authority_name=name,
            )

    try:
        firewall = load_pretraining_firewall_certificate_exact(
            root,
            root / firewall_ref["relative_path"],
            firewall_ref["sha256"],
        )
    except Exception as exc:
        _dependency_error("row-firewall certificate", exc)
    if firewall["development_projection_hashes"] != actual_development_hashes:
        _hold(
            "HOLD_ACTIVATION_TRUST_FIREWALL_BINDING",
            "firewall certificate binds different rich development projections",
        )
    if firewall["source_authority_sha256"] != row_source_hashes:
        _hold(
            "HOLD_ACTIVATION_TRUST_FIREWALL_BINDING",
            "firewall source authorities disagree with the staged activation",
        )
    expected_firewall_data = {
        "train_development_outcomes": train_sha,
        "internal_test_features": activation.authorities[
            "internal_test_features"
        ].sha256,
    }
    if firewall["data_authority_sha256"] != expected_firewall_data:
        _hold(
            "HOLD_ACTIVATION_TRUST_FIREWALL_BINDING",
            "firewall data authorities disagree with the staged activation",
        )
    try:
        future_stage_commitment = _build_future_stage_hmac_commitment(
            root,
            firewall_certificate_sha256=firewall_ref["sha256"],
            commitment_key_sha256=future_stage_commitment_key_sha256,
        )
    except Exception as exc:
        _dependency_error("future-stage commitment HMAC key", exc)

    converted_typed_refs = {
        name: {
            "path": typed_refs[name]["relative_path"],
            "sha256": typed_refs[name]["sha256"],
        }
        for name in SUPPORTED_AUTHORITIES
    }
    try:
        typed_bundle = load_typed_safe_bundle_exact(
            root,
            converted_typed_refs,
            trusted_code_snapshot=trusted_code_snapshot,
        )
    except Exception as exc:
        _dependency_error("seven-authority typed-safe bundle", exc)
    typed_authorities = typed_bundle["authorities"]
    for name in SUPPORTED_AUTHORITIES:
        authority = typed_authorities[name]
        if (
            authority["source_authority_sha256"]
            != activation.nondata_authority_sha256[name]
        ):
            _hold(
                "HOLD_ACTIVATION_TRUST_TYPED_BINDING",
                "typed-safe source authority disagrees with staged activation",
                authority_name=name,
            )
        if authority["train_development_outcomes_sha256"] != train_sha:
            _hold(
                "HOLD_ACTIVATION_TRUST_TYPED_BINDING",
                "typed-safe authority binds a different train/development authority",
                authority_name=name,
            )
        if authority["development_projection_hashes"] != actual_development_hashes:
            _hold(
                "HOLD_ACTIVATION_TRUST_TYPED_BINDING",
                "typed-safe authority binds different rich development projections",
                authority_name=name,
            )

    source_hashes = dict(activation.nondata_authority_sha256)
    if set(source_hashes) != set(REQUIRED_NONDATA_AUTHORITY_HASHES):
        _hold(
            "HOLD_ACTIVATION_TRUST_BINDING",
            "staged activation does not bind the exact ten-authority source set",
        )
    # The union is exact: three rich row authorities plus seven typed-safe
    # authorities.  Generic disclosure summaries are never inserted here.
    if set(ROW_AUTHORITIES) | set(SUPPORTED_AUTHORITIES) != set(source_hashes):
        _hold(
            "HOLD_ACTIVATION_TRUST_BINDING",
            "rich and typed-safe authority families do not cover the activation",
        )

    try:
        train_relative_path = (
            activation.authorities[TRAIN_DEVELOPMENT_AUTHORITY]
            .path.relative_to(root)
            .as_posix()
        )
    except ValueError:
        _hold(
            "HOLD_ACTIVATION_TRUST_PATH",
            "train/development authority must remain beneath the run root",
        )
    train_ref = _normalize_ref(
        root,
        {
            "relative_path": train_relative_path,
            "sha256": train_sha,
        },
        label="train/development authority ref",
    )
    if any(
        "internal_test" in part.lower()
        for part in Path(train_ref["relative_path"]).parts
    ):
        _hold(
            "HOLD_ACTIVATION_TRUST_PROTECTED",
            "development activation cannot expose a future-stage data path",
        )

    payload = {
        "schema_version": DEVELOPMENT_ACTIVATION_TRUST_SCHEMA,
        "artifact_role": DEVELOPMENT_ACTIVATION_TRUST_ROLE,
        "status": DEVELOPMENT_ACTIVATION_TRUST_STATUS,
        "capability_scope": "TRAIN_DEVELOPMENT_ONLY",
        "selected_route": AUTHORIZED_ROUTE,
        "train_development_authority_ref": train_ref,
        "typed_authority_refs": typed_refs,
        "development_row_projection_refs": row_refs,
        "partition_row_set_sha256": {
            partition: activation.partition_row_set_sha256[partition]
            for partition in ("train", "development")
        },
        "train_development_outcomes_sha256": train_sha,
        "source_authority_sha256": {
            name: source_hashes[name] for name in REQUIRED_NONDATA_AUTHORITY_HASHES
        },
        "development_projection_hashes": actual_development_hashes,
        # HMAC is both binding and hiding while its privileged 32-byte key is
        # absent from the development capability.  A bare certificate SHA is
        # deliberately not disclosed because low-entropy candidate
        # certificates would otherwise be testable offline.
        "opaque_future_stage_commitment": future_stage_commitment,
        "activation_firewall_prebinding_status": (
            "V3_EXTENSION_HMAC_BINDS_FIREWALL_CERTIFICATE_SHA256_"
            "WITH_PRIVILEGED_SECRET_KEY"
        ),
        "staged_generic_projection_role": ("DISCLOSURE_ONLY_NOT_RICH_ROW_AUTHORITY"),
        "rich_row_projection_role": "EXACT_TYPED_DEVELOPMENT_ROW_AUTHORITY",
        "generic_summary_used_as_rich_projection": False,
        "legacy_full_staged_activation_training_ready": False,
        "legacy_full_staged_activation_ref_exposed": False,
        "declared_artifact_bytes_exact_replayed": True,
        "development_rows_rebuilt_from_train_authority": True,
        "firewall_certificate_exact_bytes_cross_bound": True,
        "firewall_full_replay_reexecuted_at_binding": False,
        "independent_code_snapshot_authority_exact_replayed": False,
        "execution_qualification_certificate_exact_replayed": False,
        "semantic_dependency_chain_complete": False,
        "future_stage_access_authorized": False,
        "internal_test_feature_access": False,
        "internal_test_outcome_access": False,
        "sign_accuracy_gate": SIGN_ACCURACY_GATE,
        "sign_gate_change_authorized": False,
        "ordinary_only": True,
        "training_ready": False,
        "sealed_access": False,
        "downstream_unlock": False,
        "overwrite_authorized": False,
        "blockers": list(TRUST_BLOCKERS),
    }
    return validate_development_activation_trust_payload(payload)


def _runtime_development_disclosure(
    root: Path,
    payload: Mapping,
) -> DevelopmentDisclosure:
    """Open only the train/development authority named by safe v3."""

    train_ref = _normalize_ref(
        root,
        payload["train_development_authority_ref"],
        label="train/development authority ref",
    )
    if any(
        "internal_test" in part.lower()
        for part in Path(train_ref["relative_path"]).parts
    ):
        _hold(
            "HOLD_ACTIVATION_TRUST_PROTECTED",
            "development runtime cannot receive a future-stage data path",
        )
    train_path = root / train_ref["relative_path"]
    try:
        raw = read_exact_bytes(
            train_path,
            train_ref["sha256"],
            purpose="v3 development-only train/development authority",
        )
    except Exception as exc:
        _dependency_error("train/development authority", exc)
    try:
        metadata = os.stat(train_path, follow_symlinks=False)
    except OSError as exc:
        _dependency_error("train/development authority metadata", exc)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        _hold(
            "HOLD_ACTIVATION_TRUST_PATH",
            "train/development authority must be a regular single-link file",
        )
    safe_activation = StagedActivation(
        path=root / DEVELOPMENT_ACTIVATION_TRUST_RELATIVE_PATH,
        sha256="0" * 64,
        feature_fields=EXACT_FEATURE_FIELDS,
        outcome_fields=EXACT_OUTCOME_FIELDS,
        partition_row_set_sha256=dict(payload["partition_row_set_sha256"]),
        authorities={
            TRAIN_DEVELOPMENT_AUTHORITY: AuthorityReference(
                name=TRAIN_DEVELOPMENT_AUTHORITY,
                role="ORDINARY_TRAIN_DEVELOPMENT_OUTCOMES",
                path=train_path,
                sha256=train_ref["sha256"],
                device=metadata.st_dev,
                inode=metadata.st_ino,
            )
        },
        nondata_authority_sha256=dict(payload["source_authority_sha256"]),
        nondata_projections={},
    )
    try:
        train_rows, development_rows = _parse_train_development_rows(
            raw,
            path=train_path,
            activation=safe_activation,
        )
    except Exception as exc:
        _dependency_error("typed train/development authority", exc)
    return DevelopmentDisclosure(
        activation=safe_activation,
        train_rows=train_rows,
        development_rows=development_rows,
        capability_status="DEVELOPMENT_ACTIVATION_TRUST_V3_INPUT_ONLY",
        training_ready=False,
    )


def _replay_development_runtime_dependencies(
    run_root: Path,
    payload: Mapping,
    *,
    trusted_code_snapshot: Mapping | None,
) -> dict:
    """Replay only dependencies that a development capability may name."""

    root = _checked_run_root(run_root)
    normalized = validate_development_activation_trust_payload(payload)
    train_ref = _normalize_ref(
        root,
        normalized["train_development_authority_ref"],
        label="train/development authority ref",
    )
    typed_refs = _normalize_ref_map(
        root,
        normalized["typed_authority_refs"],
        expected_names=SUPPORTED_AUTHORITIES,
        label="typed authority refs",
        required_prefix=("development", "authorities"),
    )
    row_refs = _normalize_ref_map(
        root,
        normalized["development_row_projection_refs"],
        expected_names=ROW_AUTHORITIES,
        label="rich development row projection refs",
        required_prefix=("development", "row_authorities"),
    )
    refs = [train_ref, *typed_refs.values(), *row_refs.values()]
    if len({ref["relative_path"] for ref in refs}) != len(refs) or len(
        {ref["sha256"] for ref in refs}
    ) != len(refs):
        _hold(
            "HOLD_ACTIVATION_TRUST_ALIAS",
            "development runtime dependencies must be path/hash distinct",
        )
    if train_ref["sha256"] != normalized["train_development_outcomes_sha256"]:
        _hold(
            "HOLD_ACTIVATION_TRUST_BINDING",
            "train/development authority digest drifted",
        )

    disclosure = _runtime_development_disclosure(root, normalized)
    try:
        typed_development = normalize_development_disclosure(disclosure)
    except Exception as exc:
        _dependency_error("typed train/development rows", exc)

    source_hashes = normalized["source_authority_sha256"]
    row_source_hashes = {name: source_hashes[name] for name in ROW_AUTHORITIES}
    rich_projections: dict[str, dict] = {}
    for name in ROW_AUTHORITIES:
        ref = row_refs[name]
        try:
            rich_projections[name] = load_stage_row_projection_exact(
                root,
                root / ref["relative_path"],
                ref["sha256"],
                expected_authority_name=name,
                expected_stage=DEVELOPMENT_STAGE,
                expected_source_authority_sha256=row_source_hashes[name],
                expected_data_authority_sha256=train_ref["sha256"],
            )
        except Exception as exc:
            _dependency_error(f"rich development projection {name}", exc)
    actual_projection_hashes = stage_projection_hashes(
        rich_projections,
        stage=DEVELOPMENT_STAGE,
    )
    if actual_projection_hashes != normalized["development_projection_hashes"]:
        _hold(
            "HOLD_ACTIVATION_TRUST_BINDING",
            "rich development projection hashes drifted",
        )
    split_seed = rich_projections[SOURCE_COMPONENT_SPLIT_MANIFEST]["split_policy"][
        "seed"
    ]
    try:
        rebuilt = build_development_row_projections(
            typed_development,
            source_authority_sha256=row_source_hashes,
            train_development_outcomes_sha256=train_ref["sha256"],
            split_seed=split_seed,
        )
    except Exception as exc:
        _dependency_error("runtime rich development projection replay", exc)
    for name in ROW_AUTHORITIES:
        if canonical_stage_row_projection_bytes(rich_projections[name]) != (
            canonical_stage_row_projection_bytes(rebuilt[name])
        ):
            _hold(
                "HOLD_ACTIVATION_TRUST_RICH_REPLAY",
                "rich projection differs from runtime typed-row replay",
                authority_name=name,
            )

    converted_typed_refs = {
        name: {
            "path": typed_refs[name]["relative_path"],
            "sha256": typed_refs[name]["sha256"],
        }
        for name in SUPPORTED_AUTHORITIES
    }
    try:
        typed_bundle = load_typed_safe_bundle_exact(
            root,
            converted_typed_refs,
            trusted_code_snapshot=trusted_code_snapshot,
        )
    except Exception as exc:
        _dependency_error("runtime seven-authority typed-safe bundle", exc)
    for name in SUPPORTED_AUTHORITIES:
        authority = typed_bundle["authorities"][name]
        if authority["source_authority_sha256"] != source_hashes[name]:
            _hold(
                "HOLD_ACTIVATION_TRUST_TYPED_BINDING",
                "typed-safe source authority drifted",
                authority_name=name,
            )
        if authority["train_development_outcomes_sha256"] != train_ref["sha256"]:
            _hold(
                "HOLD_ACTIVATION_TRUST_TYPED_BINDING",
                "typed-safe train/development binding drifted",
                authority_name=name,
            )
        if authority["development_projection_hashes"] != actual_projection_hashes:
            _hold(
                "HOLD_ACTIVATION_TRUST_TYPED_BINDING",
                "typed-safe rich projection binding drifted",
                authority_name=name,
            )
    # Return only the normalized development-safe payload.  In particular,
    # neither a full StagedActivation nor any preparation reference escapes.
    return normalized


def _open_absolute_directory_nofollow(path: Path) -> int:
    if (
        not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
        or os.open not in os.supports_dir_fd
    ):
        _hold(
            "HOLD_UNSUPPORTED_ON_MUTABLE_NAMESPACE",
            "platform lacks descriptor-relative no-follow directory replay",
        )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = -1
    try:
        descriptor = os.open(os.sep, flags)
        for part in path.parts[1:]:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            metadata = os.fstat(next_descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(next_descriptor)
                _hold(
                    "HOLD_ACTIVATION_TRUST_PATH",
                    "namespace component is not a directory",
                )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except ActivationTrustError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        _hold("HOLD_ACTIVATION_TRUST_PATH", f"namespace cannot be opened safely: {exc}")


def _open_directory_beneath_nofollow(root: Path, parts: tuple[str, ...]) -> int:
    descriptor = _open_absolute_directory_nofollow(root)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        for part in parts:
            if not part or part in {".", ".."} or os.sep in part:
                _hold("HOLD_ACTIVATION_TRUST_PATH", "namespace component is unsafe")
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            metadata = os.fstat(next_descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(next_descriptor)
                _hold(
                    "HOLD_ACTIVATION_TRUST_PATH",
                    "namespace component is not a directory",
                )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except ActivationTrustError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        _hold("HOLD_ACTIVATION_TRUST_PATH", f"namespace cannot be opened safely: {exc}")


def _verify_namespace_binding(
    root: Path,
    parts: tuple[str, ...],
    pinned_descriptor: int,
) -> None:
    try:
        replay_descriptor = _open_directory_beneath_nofollow(root, parts)
    except ActivationTrustError as exc:
        _hold(
            "HOLD_UNSUPPORTED_ON_MUTABLE_NAMESPACE",
            f"activation trust namespace changed during operation: {exc}",
        )
    try:
        pinned = os.fstat(pinned_descriptor)
        replayed = os.fstat(replay_descriptor)
        if pinned.st_dev != replayed.st_dev or pinned.st_ino != replayed.st_ino:
            _hold(
                "HOLD_UNSUPPORTED_ON_MUTABLE_NAMESPACE",
                "activation trust namespace identity changed during operation",
            )
    finally:
        os.close(replay_descriptor)


def _read_regular_file_at_exact(
    parent_descriptor: int,
    name: str,
    expected_sha256: str,
) -> bytes:
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = -1
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            _hold(
                "HOLD_ACTIVATION_TRUST_PATH",
                "activation trust artifact must be a regular single-link file",
            )
        if before.st_size > MAX_TRUST_ARTIFACT_BYTES:
            _hold(
                "HOLD_ACTIVATION_TRUST_SCHEMA", "activation trust artifact is too large"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                descriptor,
                min(1 << 20, MAX_TRUST_ARTIFACT_BYTES + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_TRUST_ARTIFACT_BYTES:
                _hold(
                    "HOLD_ACTIVATION_TRUST_SCHEMA",
                    "activation trust artifact is too large",
                )
        after = os.fstat(descriptor)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or after.st_nlink != 1
        ):
            _hold(
                "HOLD_ACTIVATION_TRUST_PATH",
                "activation trust artifact changed during read",
            )
        raw = b"".join(chunks)
    except ActivationTrustError:
        raise
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.EMLINK}:
            _hold(
                "HOLD_ACTIVATION_TRUST_PATH",
                "activation trust artifact cannot be a symlink",
            )
        _hold(
            "HOLD_ACTIVATION_TRUST_PATH",
            f"activation trust artifact read failed: {exc}",
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        _hold(
            "HOLD_ACTIVATION_TRUST_HASH_MISMATCH", "activation trust SHA-256 mismatch"
        )
    return raw


def _secure_read_trust_artifact(root: Path, expected_sha256: str) -> bytes:
    parent_parts = DEVELOPMENT_ACTIVATION_TRUST_RELATIVE_PATH.parts[:-1]
    descriptor = _open_directory_beneath_nofollow(root, parent_parts)
    try:
        raw = _read_regular_file_at_exact(
            descriptor,
            DEVELOPMENT_ACTIVATION_TRUST_RELATIVE_PATH.name,
            expected_sha256,
        )
        _verify_namespace_binding(root, parent_parts, descriptor)
        return raw
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise OSError("short write while publishing activation trust artifact")
        written += count


def _secure_publish_trust_artifact(root: Path, raw: bytes) -> str:
    if os.link not in os.supports_dir_fd or os.unlink not in os.supports_dir_fd:
        _hold(
            "HOLD_UNSUPPORTED_ON_MUTABLE_NAMESPACE",
            "platform lacks descriptor-relative no-replace publication",
        )
    parent_parts = DEVELOPMENT_ACTIVATION_TRUST_RELATIVE_PATH.parts[:-1]
    parent_descriptor = _open_directory_beneath_nofollow(root, parent_parts)
    expected_sha256 = hashlib.sha256(raw).hexdigest()
    staging_descriptor = -1
    staging_name: str | None = None
    published = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        for _ in range(128):
            proposed = (
                f".{DEVELOPMENT_ACTIVATION_TRUST_RELATIVE_PATH.name}."
                f"{secrets.token_hex(16)}.staging"
            )
            try:
                staging_descriptor = os.open(
                    proposed,
                    flags,
                    0o600,
                    dir_fd=parent_descriptor,
                )
                staging_name = proposed
                break
            except FileExistsError:
                continue
        if staging_descriptor < 0 or staging_name is None:
            _hold(
                "HOLD_ACTIVATION_TRUST_WRITE",
                "cannot allocate exclusive staging artifact",
            )
        _write_all(staging_descriptor, raw)
        os.fsync(staging_descriptor)
        metadata = os.fstat(staging_descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != len(raw)
        ):
            _hold("HOLD_ACTIVATION_TRUST_WRITE", "staging artifact metadata is invalid")
        os.close(staging_descriptor)
        staging_descriptor = -1
        try:
            os.link(
                staging_name,
                DEVELOPMENT_ACTIVATION_TRUST_RELATIVE_PATH.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            _hold(
                "HOLD_ACTIVATION_TRUST_WRITE", "activation trust output already exists"
            )
        published = True
        os.fsync(parent_descriptor)
        os.unlink(staging_name, dir_fd=parent_descriptor)
        staging_name = None
        os.fsync(parent_descriptor)
        replayed = _read_regular_file_at_exact(
            parent_descriptor,
            DEVELOPMENT_ACTIVATION_TRUST_RELATIVE_PATH.name,
            expected_sha256,
        )
        if replayed != raw:
            _hold(
                "HOLD_ACTIVATION_TRUST_WRITE", "published bytes changed during replay"
            )
        _verify_namespace_binding(root, parent_parts, parent_descriptor)
        return expected_sha256
    except ActivationTrustError:
        raise
    except OSError as exc:
        _hold("HOLD_ACTIVATION_TRUST_WRITE", f"publication failed: {exc}")
    finally:
        if staging_descriptor >= 0:
            os.close(staging_descriptor)
        if staging_name is not None:
            try:
                os.unlink(staging_name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
            except OSError:
                pass
        # A visible final name is never removed or overwritten after failure.
        if published:
            pass
        os.close(parent_descriptor)


def prepare_development_activation_trust_exclusive(
    run_root: Path,
    *,
    activation_manifest_ref: Mapping,
    firewall_certificate_ref: Mapping,
    typed_authority_refs: Mapping,
    development_row_projection_refs: Mapping,
    trusted_code_snapshot: Mapping | None,
    future_stage_commitment_key_sha256: str,
) -> dict:
    """Offline-preparation entrypoint; never hand its input refs/key to training.

    ``future_stage_commitment_key_sha256`` must be independently pre-bound to
    the fixed 32-byte key file in the authority-preparation namespace.  The
    returned artifact never contains that digest, its path, or its bytes.
    """

    root = _checked_run_root(run_root)
    payload = _prepare_development_activation_payload(
        root,
        activation_manifest_ref=activation_manifest_ref,
        firewall_certificate_ref=firewall_certificate_ref,
        typed_authority_refs=typed_authority_refs,
        development_row_projection_refs=development_row_projection_refs,
        trusted_code_snapshot=trusted_code_snapshot,
        future_stage_commitment_key_sha256=future_stage_commitment_key_sha256,
    )
    raw = canonical_development_activation_trust_bytes(payload)
    digest = _secure_publish_trust_artifact(root, raw)
    replayed = load_development_activation_trust_exact(
        root,
        digest,
        trusted_code_snapshot=trusted_code_snapshot,
    )
    if canonical_development_activation_trust_bytes(replayed) != raw:
        _hold(
            "HOLD_ACTIVATION_TRUST_WRITE", "post-publication dependency replay drifted"
        )
    return {
        "path": str(root / DEVELOPMENT_ACTIVATION_TRUST_RELATIVE_PATH),
        "relative_path": DEVELOPMENT_ACTIVATION_TRUST_RELATIVE_PATH.as_posix(),
        "sha256": digest,
        "payload": replayed,
        "development_activation_trust_binding_complete": True,
        "training_ready": False,
    }


def load_development_activation_trust_exact(
    run_root: Path,
    expected_sha256: str,
    *,
    trusted_code_snapshot: Mapping | None,
) -> dict:
    """Runtime entrypoint; replay only development-safe v3 dependencies."""

    root = _checked_run_root(run_root)
    digest = _digest(expected_sha256, label="activation trust artifact")
    raw = _secure_read_trust_artifact(root, digest)
    payload = validate_development_activation_trust_payload(
        _parse_json_object(raw, label="development activation trust binding")
    )
    if raw != canonical_development_activation_trust_bytes(payload):
        _hold("HOLD_ACTIVATION_TRUST_JSON", "activation trust bytes are not canonical")
    rebuilt = _replay_development_runtime_dependencies(
        root,
        payload,
        trusted_code_snapshot=trusted_code_snapshot,
    )
    if canonical_development_activation_trust_bytes(rebuilt) != raw:
        _hold(
            "HOLD_ACTIVATION_TRUST_REPLAY",
            "activation trust payload differs from exact dependency replay",
        )
    return rebuilt


__all__ = [
    "DEVELOPMENT_ACTIVATION_TRUST_RELATIVE_PATH",
    "DEVELOPMENT_ACTIVATION_TRUST_ROLE",
    "DEVELOPMENT_ACTIVATION_TRUST_SCHEMA",
    "DEVELOPMENT_ACTIVATION_TRUST_STATUS",
    "ActivationTrustError",
    "TRUST_BLOCKERS",
    "canonical_development_activation_trust_bytes",
    "load_development_activation_trust_exact",
    "prepare_development_activation_trust_exclusive",
    "validate_development_activation_trust_payload",
]
