"""Commit-last, fail-closed closure for one R3X ordinary-only run.

``RUN_COMPLETE`` is an integrity commit, not a scientific gate and not an
authorization token.  It binds every byte stream needed to audit an ordinary
run, replays those byte streams through safe run-root-relative references, and
always keeps sealed access and downstream phase unlock disabled.

The core replay deliberately does not claim to understand every scientific
artifact.  Callers that want to retain an ``ORDINARY_GO_CANDIDATE`` declaration
must supply the complete verifier registry, an external expected-method set,
and verifier context.  Without all three, replay deterministically degrades the
effective result to ``SCIENTIFIC_HOLD``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .artifact_graph import (
    ArtifactGraphError,
    make_artifact_ref,
    read_artifact_ref,
    validate_artifact_ref,
)
from .artifact_io import (
    ArtifactIOError,
    atomic_write_json_exclusive,
    canonical_json_bytes,
)
from .protocol import (
    ProtocolHoldError,
    REQUIRED_ACTIVATION_AUTHORITIES,
    assert_ordinary_path,
)


SCHEMA = "r3x_run_complete_v1"
ARTIFACT_ROLE = "ORDINARY_RUN_COMPLETE_INTEGRITY_COMMIT"
COMMIT_SEMANTICS = "PUBLISHED_LAST_AFTER_EXACT_REFERENCE_REPLAY"
REPLAY_SCHEMA = "r3x_run_complete_replay_v1"

SCIENTIFIC_HOLD = "SCIENTIFIC_HOLD"
ORDINARY_GO_CANDIDATE = "ORDINARY_GO_CANDIDATE"
ALLOWED_STATUSES = {SCIENTIFIC_HOLD, ORDINARY_GO_CANDIDATE}

REPORT_STATUS_BY_COMPLETE_STATUS = {
    SCIENTIFIC_HOLD: "ORDINARY_RUN_COMPLETE_SCIENTIFIC_HOLD",
    ORDINARY_GO_CANDIDATE: "ORDINARY_GO_EFFECT_CANDIDATE_EXISTS",
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_METHOD_ID = re.compile(r"^[A-Z][A-Z0-9_]*$")
_ESTIMAND_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

RUN_COMPLETE_KEYS = {
    "schema_version",
    "artifact_role",
    "commit_semantics",
    "status",
    "run_plan_ref",
    "run_start_ref",
    "activation_manifest_ref",
    "authority_refs",
    "required_method_ids",
    "method_artifacts",
    "paired_artifact_ref",
    "report_ref",
    "negative_control_summary_ref",
    "ordinary_only",
    "sealed_access",
    "downstream_unlock",
}
METHOD_ARTIFACT_KEYS = {
    "method_id",
    "global_prediction_ref",
    "metrics_ref",
    "checkpoint_ensemble_refs",
}

# A single authority callback is invoked once per frozen authority; likewise,
# the checkpoint/ensemble callback is invoked once per method and estimand.
VERIFIER_ROLES = {
    "run_plan",
    "run_start",
    "activation",
    "authority",
    "global_prediction",
    "metrics",
    "checkpoint_ensemble",
    "paired",
    "report",
    "negative_control_summary",
}

# Verifiers receive the exact referenced bytes and immutable call metadata.
# They should raise on failure or return anything except the literal False.
Verifier = Callable[[bytes, Mapping[str, object]], object]


class RunCompleteProtocolError(RuntimeError):
    """Fail-closed RUN_COMPLETE schema, reference, or replay error."""


def _strict_keys(value: Mapping, expected: set[str], *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise RunCompleteProtocolError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise RunCompleteProtocolError(
            f"{label} keys differ from strict schema: {sorted(actual ^ expected)}"
        )


def _method_id(value: object) -> str:
    if not isinstance(value, str) or not _METHOD_ID.fullmatch(value):
        raise RunCompleteProtocolError(
            "method_id must be a safe uppercase frozen identifier"
        )
    return value


def _estimand_id(value: object) -> str:
    if not isinstance(value, str) or not _ESTIMAND_ID.fullmatch(value):
        raise RunCompleteProtocolError("estimand_id is not a safe identifier")
    return value


def _normalize_ref(value: Mapping, *, label: str) -> dict:
    try:
        return validate_artifact_ref(value)
    except ArtifactGraphError as exc:
        raise RunCompleteProtocolError(f"{label}: {exc}") from exc


def _normalize_required_methods(values: Sequence[str]) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise RunCompleteProtocolError("required_method_ids must be a sequence")
    normalized = [_method_id(value) for value in values]
    if not normalized or len(normalized) != len(set(normalized)):
        raise RunCompleteProtocolError(
            "required_method_ids must be non-empty and unique"
        )
    if normalized != sorted(normalized):
        raise RunCompleteProtocolError(
            "required_method_ids must be in canonical sorted order"
        )
    return normalized


def _normalize_authority_refs(value: Mapping) -> dict[str, dict]:
    if not isinstance(value, Mapping):
        raise RunCompleteProtocolError("authority_refs must be an object")
    expected = set(REQUIRED_ACTIVATION_AUTHORITIES)
    if set(value) != expected:
        raise RunCompleteProtocolError(
            "authority_refs must contain exactly the eleven activation authorities: "
            + str(sorted(set(value) ^ expected))
        )
    return {
        name: _normalize_ref(value[name], label=f"authority_refs.{name}")
        for name in REQUIRED_ACTIVATION_AUTHORITIES
    }


def _normalize_method_artifacts(
    value: Mapping, required_method_ids: Sequence[str]
) -> dict[str, dict]:
    if not isinstance(value, Mapping):
        raise RunCompleteProtocolError("method_artifacts must be an object")
    if set(value) != set(required_method_ids):
        raise RunCompleteProtocolError(
            "method_artifacts must exactly cover required_method_ids"
        )
    normalized: dict[str, dict] = {}
    for method_id in required_method_ids:
        entry = value[method_id]
        _strict_keys(entry, METHOD_ARTIFACT_KEYS, label=f"method {method_id}")
        if _method_id(entry.get("method_id")) != method_id:
            raise RunCompleteProtocolError(
                f"method artifact key/embedded ID mismatch for {method_id}"
            )
        raw_ensembles = entry.get("checkpoint_ensemble_refs")
        if not isinstance(raw_ensembles, Mapping) or not raw_ensembles:
            raise RunCompleteProtocolError(
                f"method {method_id} requires checkpoint_ensemble_refs"
            )
        ensembles: dict[str, dict] = {}
        for raw_estimand_id, reference in raw_ensembles.items():
            estimand_id = _estimand_id(raw_estimand_id)
            if estimand_id in ensembles:
                raise RunCompleteProtocolError(
                    f"method {method_id} has duplicate estimand references"
                )
            ensembles[estimand_id] = _normalize_ref(
                reference,
                label=(
                    f"method_artifacts.{method_id}."
                    f"checkpoint_ensemble_refs.{estimand_id}"
                ),
            )
        normalized[method_id] = {
            "method_id": method_id,
            "global_prediction_ref": _normalize_ref(
                entry.get("global_prediction_ref"),
                label=f"method_artifacts.{method_id}.global_prediction_ref",
            ),
            "metrics_ref": _normalize_ref(
                entry.get("metrics_ref"),
                label=f"method_artifacts.{method_id}.metrics_ref",
            ),
            "checkpoint_ensemble_refs": {
                estimand_id: ensembles[estimand_id]
                for estimand_id in sorted(ensembles)
            },
        }
    return normalized


def validate_run_complete(payload: Mapping) -> dict:
    """Validate the strict integrity-commit schema without reading files."""

    _strict_keys(payload, RUN_COMPLETE_KEYS, label="RUN_COMPLETE")
    if payload.get("schema_version") != SCHEMA:
        raise RunCompleteProtocolError("unsupported RUN_COMPLETE schema")
    if payload.get("artifact_role") != ARTIFACT_ROLE:
        raise RunCompleteProtocolError("RUN_COMPLETE artifact role is invalid")
    if payload.get("commit_semantics") != COMMIT_SEMANTICS:
        raise RunCompleteProtocolError("RUN_COMPLETE must declare commit-last semantics")
    status = payload.get("status")
    if status not in ALLOWED_STATUSES:
        raise RunCompleteProtocolError("RUN_COMPLETE status is not allowed")
    if payload.get("ordinary_only") is not True:
        raise RunCompleteProtocolError("RUN_COMPLETE must remain ordinary-only")
    if payload.get("sealed_access") is not False:
        raise RunCompleteProtocolError("RUN_COMPLETE cannot carry sealed access")
    if payload.get("downstream_unlock") is not False:
        raise RunCompleteProtocolError("RUN_COMPLETE cannot unlock downstream phases")

    required = _normalize_required_methods(payload.get("required_method_ids"))
    normalized = {
        "schema_version": SCHEMA,
        "artifact_role": ARTIFACT_ROLE,
        "commit_semantics": COMMIT_SEMANTICS,
        "status": status,
        "run_plan_ref": _normalize_ref(
            payload.get("run_plan_ref"), label="run_plan_ref"
        ),
        "run_start_ref": _normalize_ref(
            payload.get("run_start_ref"), label="run_start_ref"
        ),
        "activation_manifest_ref": _normalize_ref(
            payload.get("activation_manifest_ref"),
            label="activation_manifest_ref",
        ),
        "authority_refs": _normalize_authority_refs(payload.get("authority_refs")),
        "required_method_ids": required,
        "method_artifacts": _normalize_method_artifacts(
            payload.get("method_artifacts"), required
        ),
        "paired_artifact_ref": _normalize_ref(
            payload.get("paired_artifact_ref"), label="paired_artifact_ref"
        ),
        "report_ref": _normalize_ref(payload.get("report_ref"), label="report_ref"),
        "negative_control_summary_ref": _normalize_ref(
            payload.get("negative_control_summary_ref"),
            label="negative_control_summary_ref",
        ),
        "ordinary_only": True,
        "sealed_access": False,
        "downstream_unlock": False,
    }
    return normalized


def build_run_complete(
    *,
    status: str,
    run_plan_ref: Mapping,
    run_start_ref: Mapping,
    activation_manifest_ref: Mapping,
    authority_refs: Mapping,
    required_method_ids: Sequence[str],
    method_artifacts: Mapping,
    paired_artifact_ref: Mapping,
    report_ref: Mapping,
    negative_control_summary_ref: Mapping,
) -> dict:
    """Build one strict closure payload; exact replay is performed separately."""

    payload = {
        "schema_version": SCHEMA,
        "artifact_role": ARTIFACT_ROLE,
        "commit_semantics": COMMIT_SEMANTICS,
        "status": status,
        "run_plan_ref": dict(run_plan_ref),
        "run_start_ref": dict(run_start_ref),
        "activation_manifest_ref": dict(activation_manifest_ref),
        "authority_refs": dict(authority_refs),
        "required_method_ids": list(required_method_ids),
        "method_artifacts": dict(method_artifacts),
        "paired_artifact_ref": dict(paired_artifact_ref),
        "report_ref": dict(report_ref),
        "negative_control_summary_ref": dict(negative_control_summary_ref),
        "ordinary_only": True,
        "sealed_access": False,
        "downstream_unlock": False,
    }
    return validate_run_complete(payload)


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json_object(raw: bytes, *, label: str) -> dict:
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_json_object
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RunCompleteProtocolError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RunCompleteProtocolError(f"{label} must be a JSON object")
    return value


def _authority_hash_map(payload: Mapping) -> dict[str, str]:
    return {
        f"{name}_sha256": payload["authority_refs"][name]["sha256"]
        for name in REQUIRED_ACTIVATION_AUTHORITIES
    }


def _all_references(payload: Mapping):
    yield "run_plan", payload["run_plan_ref"], {}
    yield "run_start", payload["run_start_ref"], {}
    yield "activation", payload["activation_manifest_ref"], {}
    for name in REQUIRED_ACTIVATION_AUTHORITIES:
        yield "authority", payload["authority_refs"][name], {
            "authority_name": name
        }
    for method_id in payload["required_method_ids"]:
        method = payload["method_artifacts"][method_id]
        yield "global_prediction", method["global_prediction_ref"], {
            "method_id": method_id
        }
        yield "metrics", method["metrics_ref"], {"method_id": method_id}
        for estimand_id, reference in method["checkpoint_ensemble_refs"].items():
            yield "checkpoint_ensemble", reference, {
                "method_id": method_id,
                "estimand_id": estimand_id,
            }
    yield "paired", payload["paired_artifact_ref"], {}
    yield "report", payload["report_ref"], {}
    yield "negative_control_summary", payload["negative_control_summary_ref"], {}


def _validate_unique_reference_paths(payload: Mapping) -> None:
    owners: dict[str, str] = {}
    for role, reference, details in _all_references(payload):
        label = role
        if details:
            label += ":" + ":".join(str(value) for value in details.values())
        relative = reference["relative_path"]
        if relative in owners:
            raise RunCompleteProtocolError(
                f"RUN_COMPLETE reference path is reused by {owners[relative]} and {label}"
            )
        owners[relative] = label


def _normalize_callbacks(value: Mapping[str, Verifier] | None) -> dict[str, Verifier]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RunCompleteProtocolError("verifier_callbacks must be an object")
    unknown = set(value) - VERIFIER_ROLES
    if unknown:
        raise RunCompleteProtocolError(
            f"unknown verifier callback roles: {sorted(unknown)}"
        )
    for role, callback in value.items():
        if not callable(callback):
            raise RunCompleteProtocolError(
                f"verifier callback {role} is not callable"
            )
    return dict(value)


def _invoke_verifier(
    callbacks: Mapping[str, Verifier],
    role: str,
    raw: bytes,
    *,
    reference: Mapping,
    details: Mapping,
    verifier_context: Mapping | None,
) -> None:
    callback = callbacks.get(role)
    if callback is None:
        return
    metadata = {
        "role": role,
        "reference": dict(reference),
        **dict(details),
        "verifier_context": (
            None if verifier_context is None else dict(verifier_context)
        ),
    }
    try:
        result = callback(raw, metadata)
    except Exception as exc:
        raise RunCompleteProtocolError(
            f"external verifier failed for {role}: {exc}"
        ) from exc
    if result is False:
        raise RunCompleteProtocolError(
            f"external verifier rejected {role}"
        )


def _check_core_bindings(
    payload: Mapping,
    *,
    raw_by_role_path: Mapping[tuple[str, str], bytes],
) -> None:
    authority_hashes = _authority_hash_map(payload)
    run_plan = _parse_json_object(
        raw_by_role_path[("run_plan", payload["run_plan_ref"]["relative_path"])],
        label="RUN_PLAN",
    )
    if run_plan.get("schema_version") != "r3x_run_plan_v1":
        raise RunCompleteProtocolError("RUN_PLAN schema is unsupported")
    if run_plan.get("ordinary_only") is not True:
        raise RunCompleteProtocolError("RUN_PLAN is not ordinary-only")
    if run_plan.get("sealed_access") is not False:
        raise RunCompleteProtocolError("RUN_PLAN carries sealed access")
    if run_plan.get("downstream_unlock") is not False:
        raise RunCompleteProtocolError("RUN_PLAN carries downstream unlock")
    if (
        run_plan.get("activation_manifest_sha256")
        != payload["activation_manifest_ref"]["sha256"]
    ):
        raise RunCompleteProtocolError("RUN_PLAN/activation binding mismatch")
    if run_plan.get("authority_hashes") != authority_hashes:
        raise RunCompleteProtocolError("RUN_PLAN authority bindings are incomplete")

    run_start = _parse_json_object(
        raw_by_role_path[("run_start", payload["run_start_ref"]["relative_path"])],
        label="RUN_START",
    )
    if run_start.get("schema_version") != "r3x_run_start_v1":
        raise RunCompleteProtocolError("RUN_START schema is unsupported")
    if run_start.get("sealed_access") is not False:
        raise RunCompleteProtocolError("RUN_START carries sealed access")
    if run_start.get("downstream_unlock") is not False:
        raise RunCompleteProtocolError("RUN_START carries downstream unlock")
    if run_start.get("run_plan_sha256") != payload["run_plan_ref"]["sha256"]:
        raise RunCompleteProtocolError("RUN_START/RUN_PLAN binding mismatch")
    if (
        run_start.get("activation_manifest_sha256")
        != payload["activation_manifest_ref"]["sha256"]
    ):
        raise RunCompleteProtocolError("RUN_START/activation binding mismatch")
    if run_start.get("authority_hashes") != authority_hashes:
        raise RunCompleteProtocolError("RUN_START authority bindings are incomplete")

    activation = _parse_json_object(
        raw_by_role_path[
            ("activation", payload["activation_manifest_ref"]["relative_path"])
        ],
        label="activation manifest",
    )
    if activation.get("schema_version") != "r3_activation_manifest_v1":
        raise RunCompleteProtocolError("activation manifest schema is unsupported")
    if activation.get("status") != "FROZEN" or activation.get("mode") != "ORDINARY_ONLY":
        raise RunCompleteProtocolError(
            "activation manifest must be frozen ordinary-only"
        )
    activation_authorities = activation.get("authorities")
    if (
        not isinstance(activation_authorities, Mapping)
        or set(activation_authorities) != set(REQUIRED_ACTIVATION_AUTHORITIES)
    ):
        raise RunCompleteProtocolError(
            "activation manifest does not contain exactly eleven authorities"
        )
    for name in REQUIRED_ACTIVATION_AUTHORITIES:
        entry = activation_authorities[name]
        if (
            not isinstance(entry, Mapping)
            or entry.get("sha256") != payload["authority_refs"][name]["sha256"]
        ):
            raise RunCompleteProtocolError(
                f"activation authority SHA-256 mismatch for {name}"
            )

    report = _parse_json_object(
        raw_by_role_path[("report", payload["report_ref"]["relative_path"])],
        label="run report",
    )
    if report.get("schema_version") != "r3x_ordinary_run_report_v1":
        raise RunCompleteProtocolError("run report schema is unsupported")
    if report.get("sealed_access") is not False:
        raise RunCompleteProtocolError("run report carries sealed access")
    if report.get("downstream_unlock") is not False:
        raise RunCompleteProtocolError("run report carries downstream unlock")
    if report.get("run_plan_sha256") != payload["run_plan_ref"]["sha256"]:
        raise RunCompleteProtocolError("run report/RUN_PLAN binding mismatch")
    if report.get("run_start_sha256") != payload["run_start_ref"]["sha256"]:
        raise RunCompleteProtocolError("run report/RUN_START binding mismatch")
    if (
        report.get("activation_manifest_sha256")
        != payload["activation_manifest_ref"]["sha256"]
    ):
        raise RunCompleteProtocolError("run report/activation binding mismatch")
    expected_report_status = REPORT_STATUS_BY_COMPLETE_STATUS[payload["status"]]
    expected_go = payload["status"] == ORDINARY_GO_CANDIDATE
    if report.get("status") != expected_report_status:
        raise RunCompleteProtocolError(
            "run report status is inconsistent with RUN_COMPLETE"
        )
    if report.get("GO_EFFECT") is not expected_go:
        raise RunCompleteProtocolError(
            "run report GO_EFFECT is inconsistent with RUN_COMPLETE"
        )
    methods = report.get("methods")
    if not isinstance(methods, Mapping) or not set(payload["required_method_ids"]).issubset(methods):
        raise RunCompleteProtocolError(
            "run report does not contain every required method"
        )


def replay_run_complete_payload(
    run_root: Path,
    payload: Mapping,
    *,
    expected_method_ids: Sequence[str] | None = None,
    verifier_callbacks: Mapping[str, Verifier] | None = None,
    verifier_context: Mapping | None = None,
) -> dict:
    """Replay all refs and return an effective, never-unlocking disposition.

    A declared GO candidate is degraded to ``SCIENTIFIC_HOLD`` unless the
    caller supplies an exact expected-method set, all verifier roles, and a
    non-empty external verifier context.  Structural or hash inconsistency is
    always an error rather than a downgraded success.
    """

    normalized = validate_run_complete(payload)
    _validate_unique_reference_paths(normalized)
    callbacks = _normalize_callbacks(verifier_callbacks)
    root = Path(run_root)
    try:
        assert_ordinary_path(root, purpose="RUN_COMPLETE replay root")
    except ProtocolHoldError as exc:
        raise RunCompleteProtocolError(str(exc)) from exc
    if not root.is_dir() or root.is_symlink():
        raise RunCompleteProtocolError(
            "RUN_COMPLETE replay root must be an existing real directory"
        )

    if expected_method_ids is not None:
        expected = _normalize_required_methods(expected_method_ids)
        if expected != normalized["required_method_ids"]:
            raise RunCompleteProtocolError(
                "external expected method set differs from RUN_COMPLETE"
            )

    raw_by_role_path: dict[tuple[str, str], bytes] = {}
    for role, reference, details in _all_references(normalized):
        try:
            raw = read_artifact_ref(root, reference)
        except ArtifactGraphError as exc:
            raise RunCompleteProtocolError(
                f"exact replay failed for {role}: {exc}"
            ) from exc
        raw_by_role_path[(role, reference["relative_path"])] = raw
        _invoke_verifier(
            callbacks,
            role,
            raw,
            reference=reference,
            details=details,
            verifier_context=verifier_context,
        )

    _check_core_bindings(normalized, raw_by_role_path=raw_by_role_path)

    hold_reasons: list[str] = []
    go_verified = False
    if normalized["status"] == ORDINARY_GO_CANDIDATE:
        if expected_method_ids is None:
            hold_reasons.append("EXPECTED_METHOD_SET_REQUIRED_FOR_GO")
        missing_callbacks = sorted(VERIFIER_ROLES - set(callbacks))
        if missing_callbacks:
            hold_reasons.append(
                "COMPLETE_EXTERNAL_VERIFIER_REGISTRY_REQUIRED_FOR_GO:"
                + ",".join(missing_callbacks)
            )
        if not isinstance(verifier_context, Mapping) or not verifier_context:
            hold_reasons.append("NONEMPTY_EXTERNAL_VERIFIER_CONTEXT_REQUIRED_FOR_GO")
        go_verified = not hold_reasons

    effective_status = (
        ORDINARY_GO_CANDIDATE
        if go_verified
        else SCIENTIFIC_HOLD
    )
    return {
        "schema_version": REPLAY_SCHEMA,
        "declared_status": normalized["status"],
        "effective_status": effective_status,
        "go_candidate_verified": go_verified,
        "hold_reasons": hold_reasons,
        "required_method_ids": list(normalized["required_method_ids"]),
        "verified_reference_count": len(raw_by_role_path),
        "ordinary_only": True,
        "sealed_access": False,
        "downstream_unlock": False,
    }


def replay_run_complete(
    run_root: Path,
    run_complete_ref: Mapping,
    *,
    expected_method_ids: Sequence[str] | None = None,
    verifier_callbacks: Mapping[str, Verifier] | None = None,
    verifier_context: Mapping | None = None,
) -> dict:
    """Read an exact RUN_COMPLETE ref, validate canonical bytes, and replay it."""

    try:
        raw = read_artifact_ref(run_root, run_complete_ref)
    except ArtifactGraphError as exc:
        raise RunCompleteProtocolError(f"RUN_COMPLETE exact read failed: {exc}") from exc
    payload = _parse_json_object(raw, label="RUN_COMPLETE")
    normalized = validate_run_complete(payload)
    try:
        canonical = canonical_json_bytes(normalized)
    except ArtifactIOError as exc:
        raise RunCompleteProtocolError(str(exc)) from exc
    if raw != canonical:
        raise RunCompleteProtocolError("RUN_COMPLETE bytes are not canonical")
    return replay_run_complete_payload(
        run_root,
        normalized,
        expected_method_ids=expected_method_ids,
        verifier_callbacks=verifier_callbacks,
        verifier_context=verifier_context,
    )


def write_run_complete_exclusive(
    path: Path,
    payload: Mapping,
    *,
    run_root: Path,
    expected_method_ids: Sequence[str] | None = None,
    verifier_callbacks: Mapping[str, Verifier] | None = None,
    verifier_context: Mapping | None = None,
) -> str:
    """Replay every dependency, then atomically publish the final commit name.

    A declared GO candidate is refused unless full external replay preserves
    that disposition.  A scientific HOLD needs no scientific callbacks, but
    still requires exact byte replay and all core cross-bindings.
    """

    normalized = validate_run_complete(payload)
    root = Path(run_root)
    target = Path(path)
    try:
        assert_ordinary_path(target, purpose="RUN_COMPLETE output")
        root_resolved = root.resolve(strict=True)
        parent_resolved = target.parent.resolve(strict=True)
    except (ProtocolHoldError, FileNotFoundError, OSError) as exc:
        raise RunCompleteProtocolError(str(exc)) from exc
    if root.is_symlink() or not root_resolved.is_dir():
        raise RunCompleteProtocolError("RUN_COMPLETE run root must be a real directory")
    if not parent_resolved.is_relative_to(root_resolved):
        raise RunCompleteProtocolError("RUN_COMPLETE output escapes the run root")
    target_relative = parent_resolved.joinpath(target.name).relative_to(
        root_resolved
    ).as_posix()
    if target_relative in {
        reference["relative_path"]
        for _, reference, _ in _all_references(normalized)
    }:
        raise RunCompleteProtocolError(
            "RUN_COMPLETE cannot reference its own output path"
        )

    replay = replay_run_complete_payload(
        root_resolved,
        normalized,
        expected_method_ids=expected_method_ids,
        verifier_callbacks=verifier_callbacks,
        verifier_context=verifier_context,
    )
    if (
        normalized["status"] == ORDINARY_GO_CANDIDATE
        and replay["effective_status"] != ORDINARY_GO_CANDIDATE
    ):
        raise RunCompleteProtocolError(
            "GO-candidate RUN_COMPLETE failed closed to SCIENTIFIC_HOLD: "
            + "; ".join(replay["hold_reasons"])
        )
    try:
        digest = atomic_write_json_exclusive(
            target, normalized, validator=validate_run_complete
        )
        # Constructing the safe ref immediately proves that the final name is a
        # single-link child of the same ordinary run root under the returned hash.
        make_artifact_ref(root_resolved, target, digest)
        return digest
    except (ArtifactIOError, ArtifactGraphError) as exc:
        raise RunCompleteProtocolError(str(exc)) from exc
