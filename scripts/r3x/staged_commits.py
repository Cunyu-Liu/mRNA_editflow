"""Immutable staged-disclosure commit DAG for ordinary-only R3X runs.

The four public state artifacts in this module form a strict linear DAG::

    DEVELOPMENT_COMMIT
      -> PRE_TEST_FREEZE_COMMIT
      -> TEST_PREDICTION_COMMIT
      -> ORDINARY_TEST_OUTCOME_RELEASE_RECORD

The implementation deliberately does not load a monolithic ready dataset and
does not know how to train or evaluate a model.  It only commits/replays the
state needed to prove that development, pre-test freeze, feature-only
prediction, and one-time ordinary outcome release happened in that order.

Outcome release uses a durable, no-overwrite claim written *before* the first
outcome byte is read.  A crash after that claim is therefore a fail-closed
partial release: the run cannot silently read the outcome authority again.
Completed release records can be replayed without reopening outcome bytes.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .artifact_graph import (
    ArtifactGraphError,
    make_artifact_ref,
    read_artifact_ref,
    validate_artifact_ref,
)
from .artifact_io import ArtifactIOError, atomic_write_json_exclusive, canonical_json_bytes
from .checkpoint import FROZEN_SEEDS
from .protocol import (
    ProtocolHoldError,
    REQUIRED_NEGATIVE_CONTROLS,
    SIGN_ACCURACY_GATE,
    assert_ordinary_path,
)


DEVELOPMENT_COMMIT = "DEVELOPMENT_COMMIT"
PRE_TEST_FREEZE_COMMIT = "PRE_TEST_FREEZE_COMMIT"
TEST_PREDICTION_COMMIT = "TEST_PREDICTION_COMMIT"
ORDINARY_TEST_OUTCOME_RELEASE_RECORD = "ORDINARY_TEST_OUTCOME_RELEASE_RECORD"

STATE_SEQUENCE = (
    DEVELOPMENT_COMMIT,
    PRE_TEST_FREEZE_COMMIT,
    TEST_PREDICTION_COMMIT,
    ORDINARY_TEST_OUTCOME_RELEASE_RECORD,
)

STATE_SCHEMAS = {
    DEVELOPMENT_COMMIT: "r3x_development_commit_v1",
    PRE_TEST_FREEZE_COMMIT: "r3x_pre_test_freeze_commit_v2",
    TEST_PREDICTION_COMMIT: "r3x_test_prediction_commit_v2",
    ORDINARY_TEST_OUTCOME_RELEASE_RECORD: (
        "r3x_ordinary_test_outcome_release_record_v1"
    ),
}

STATE_RELATIVE_PATHS = {
    DEVELOPMENT_COMMIT: "commits/01_DEVELOPMENT_COMMIT.json",
    PRE_TEST_FREEZE_COMMIT: "commits/02_PRE_TEST_FREEZE_COMMIT.json",
    TEST_PREDICTION_COMMIT: "commits/03_TEST_PREDICTION_COMMIT.json",
    ORDINARY_TEST_OUTCOME_RELEASE_RECORD: (
        "commits/04_ORDINARY_TEST_OUTCOME_RELEASE_RECORD.json"
    ),
}

OUTCOME_RELEASE_CLAIM_SCHEMA = "r3x_ordinary_test_outcome_release_claim_v1"
OUTCOME_RELEASE_CLAIM_RELATIVE_PATH = (
    "commits/.ORDINARY_TEST_OUTCOME_RELEASE_CLAIM.json"
)

TRAIN_DEVELOPMENT_AUTHORITY = "train_development_outcomes"
INTERNAL_TEST_FEATURE_AUTHORITY = "internal_test_features"
INTERNAL_TEST_OUTCOME_AUTHORITY = "internal_test_outcomes"
REQUIRED_STAGED_AUTHORITY_HASHES = (
    TRAIN_DEVELOPMENT_AUTHORITY,
    INTERNAL_TEST_FEATURE_AUTHORITY,
    INTERNAL_TEST_OUTCOME_AUTHORITY,
)
REQUIRED_NONDATA_AUTHORITY_HASHES = (
    "route_authorization",
    "inclusion_manifest",
    "context_source_crosswalk",
    "endpoint_estimand_registry",
    "source_component_split_manifest",
    "metric_zero_policy_registry",
    "model_optimizer_seed_config",
    "analysis_code_tree_manifest",
    "method_protocol",
    "paired_strongest_baseline_comparison_policy",
)
STAGE_PROJECTED_NONDATA_AUTHORITIES = REQUIRED_NONDATA_AUTHORITY_HASHES
REQUIRED_NONDATA_PROJECTION_HASHES = tuple(
    sorted(
        f"{name}_{stage}_projection"
        for name in STAGE_PROJECTED_NONDATA_AUTHORITIES
        for stage in ("development", "internal_test_features")
    )
)

COMMON_KEYS = {
    "schema_version",
    "artifact_role",
    "state",
    "transition_index",
    "status",
    "prior_commit_refs",
    "run_start_sha256",
    "activation_sha256",
    "authority_hashes",
    "nondata_authority_hashes",
    "nondata_projection_hashes",
    "n_rows",
    "ordered_row_ids_sha256",
    "row_set_sha256",
    "required_method_ids",
    "required_estimand_ids",
    "required_seeds",
    "required_control_ids",
    "code_sha256",
    "config_sha256",
    "sign_accuracy_gate",
    "sign_gate_change_authorized",
    "ordinary_only",
    "sealed",
}

DEVELOPMENT_KEYS = COMMON_KEYS | {
    "development_authority_ref",
    "development_artifact_refs",
    "access_audit_ref",
    "access_assertions",
}
PRE_TEST_FREEZE_KEYS = COMMON_KEYS | {
    "selector_ref",
    "predictor_provenance_refs",
    "selected_method_id",
    "frozen_checkpoint_refs",
    "access_audit_ref",
    "access_assertions",
}
TEST_PREDICTION_KEYS = COMMON_KEYS | {
    "feature_authority_ref",
    "evaluation_predictions_ref",
    "selected_method_id",
    "frozen_prediction_refs",
    "access_assertions",
}
OUTCOME_RELEASE_KEYS = COMMON_KEYS | {
    "outcome_authority_ref",
    "release_claim_ref",
    "access_assertions",
    "consumed_once",
}

ACCESS_ASSERTION_KEYS = {
    "internal_test_feature_accessed",
    "internal_test_outcome_accessed",
}
PREDICTION_BINDING_KEYS = {"checkpoint_ref", "prediction_ref"}
ARTIFACT_REF_KEYS = {"relative_path", "sha256"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SAFE_AUTHORITY_ID = re.compile(r"^[a-z][a-z0-9_]*$")
PROTECTED_TOKEN = re.compile(r"(?:sealed|restricted|gse246381)", re.IGNORECASE)


class StagedCommitError(RuntimeError):
    """Fail-closed schema, transition, reference, or disclosure error."""


def _fail(message: str) -> None:
    raise StagedCommitError(message)


def _strict_keys(value: Mapping, expected: set[str], *, label: str) -> None:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        _fail(f"{label} keys differ from strict schema: {sorted(actual ^ expected)}")


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        _fail(f"{label} must be 64 lowercase hex")
    return value


def _safe_id(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        _fail(f"{label} is not a safe frozen identifier")
    if PROTECTED_TOKEN.search(value):
        _fail(f"{label} cannot name a protected authority")
    return value


def _canonical_ids(value: object, *, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        _fail(f"{label} must be a non-empty canonical list")
    normalized = [_safe_id(item, label=f"{label} item") for item in value]
    if normalized != sorted(set(normalized)):
        _fail(f"{label} must be sorted and unique")
    return normalized


def _canonical_seeds(value: object) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in value
    ):
        _fail("required_seeds must be a canonical integer list")
    expected = sorted(FROZEN_SEEDS)
    if value != expected:
        _fail(f"required_seeds must be the frozen sequence {expected}")
    return list(value)


def _authority_hashes(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        _fail("authority_hashes must be an object")
    normalized: dict[str, str] = {}
    for name, digest in value.items():
        if not isinstance(name, str) or not SAFE_AUTHORITY_ID.fullmatch(name):
            _fail("authority_hashes contains an unsafe authority identifier")
        if PROTECTED_TOKEN.search(name):
            _fail("authority_hashes cannot name a protected authority")
        normalized[name] = _digest(digest, label=f"authority_hashes[{name}]")
    expected = set(REQUIRED_STAGED_AUTHORITY_HASHES)
    if set(normalized) != expected:
        _fail(
            "authority_hashes must contain exactly the three canonical staged "
            f"authorities: {sorted(set(normalized) ^ expected)}"
        )
    if list(value) != sorted(value):
        _fail("authority_hashes keys must be in canonical sorted order")
    return normalized


def _nondata_authority_hashes(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        _fail("nondata_authority_hashes must be an object")
    expected = set(REQUIRED_NONDATA_AUTHORITY_HASHES)
    if set(value) != expected:
        _fail(
            "nondata_authority_hashes must contain the exact independent "
            f"scientific authority set: {sorted(set(value) ^ expected)}"
        )
    if list(value) != sorted(value):
        _fail("nondata_authority_hashes keys must be in canonical sorted order")
    return {
        name: _digest(value[name], label=f"nondata_authority_hashes[{name}]")
        for name in sorted(value)
    }


def _nondata_projection_hashes(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        _fail("nondata_projection_hashes must be an object")
    expected = set(REQUIRED_NONDATA_PROJECTION_HASHES)
    if set(value) != expected:
        _fail(
            "nondata_projection_hashes must contain the exact stage-specific "
            f"projection set: {sorted(set(value) ^ expected)}"
        )
    if list(value) != sorted(value):
        _fail("nondata_projection_hashes keys must be in canonical sorted order")
    return {
        name: _digest(value[name], label=f"nondata_projection_hashes[{name}]")
        for name in sorted(value)
    }


def _artifact_ref(value: object, *, label: str) -> dict[str, str]:
    try:
        normalized = validate_artifact_ref(value)
    except (ArtifactGraphError, TypeError) as exc:
        raise StagedCommitError(f"invalid {label}: {exc}") from exc
    if set(normalized) != ARTIFACT_REF_KEYS:
        _fail(f"{label} is not an exact artifact reference")
    return normalized


def _artifact_ref_map(
    value: object,
    *,
    label: str,
    exact_keys: Sequence[str] | None = None,
) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or not value:
        _fail(f"{label} must be a non-empty object")
    if exact_keys is not None and set(value) != set(exact_keys):
        _fail(
            f"{label} does not cover the exact frozen units: "
            f"{sorted(set(value) ^ set(exact_keys))}"
        )
    normalized: dict[str, dict[str, str]] = {}
    seen_refs: set[tuple[str, str]] = set()
    for key in sorted(value):
        if exact_keys is None:
            _safe_id(key, label=f"{label} key")
        ref = _artifact_ref(value[key], label=f"{label}[{key}]")
        identity = (ref["relative_path"], ref["sha256"])
        if identity in seen_refs:
            _fail(f"{label} cannot reuse one artifact for multiple bindings")
        seen_refs.add(identity)
        normalized[key] = ref
    return normalized


def _access_assertions(value: object, *, state: str) -> dict[str, bool]:
    _strict_keys(value, ACCESS_ASSERTION_KEYS, label="access_assertions")
    normalized = {
        "internal_test_feature_accessed": value.get(
            "internal_test_feature_accessed"
        ),
        "internal_test_outcome_accessed": value.get(
            "internal_test_outcome_accessed"
        ),
    }
    if any(type(item) is not bool for item in normalized.values()):
        _fail("access_assertions values must be JSON booleans")
    expected = {
        DEVELOPMENT_COMMIT: (False, False),
        PRE_TEST_FREEZE_COMMIT: (False, False),
        TEST_PREDICTION_COMMIT: (True, False),
        ORDINARY_TEST_OUTCOME_RELEASE_RECORD: (True, True),
    }[state]
    actual = (
        normalized["internal_test_feature_accessed"],
        normalized["internal_test_outcome_accessed"],
    )
    if actual != expected:
        _fail(
            f"{state} requires feature/outcome access assertions {expected}, got {actual}"
        )
    return normalized


def build_staged_bindings(
    *,
    run_start_sha256: str,
    activation_sha256: str,
    authority_hashes: Mapping[str, str],
    nondata_authority_hashes: Mapping[str, str],
    nondata_projection_hashes: Mapping[str, str],
    n_rows: int,
    ordered_row_ids_sha256: str,
    row_set_sha256: str,
    required_method_ids: Sequence[str],
    required_estimand_ids: Sequence[str],
    required_seeds: Sequence[int],
    required_control_ids: Sequence[str],
    code_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    """Build the canonical cross-stage binding shared by all four commits."""

    payload = {
        "run_start_sha256": run_start_sha256,
        "activation_sha256": activation_sha256,
        "authority_hashes": dict(sorted(authority_hashes.items())),
        "nondata_authority_hashes": dict(sorted(nondata_authority_hashes.items())),
        "nondata_projection_hashes": dict(sorted(nondata_projection_hashes.items())),
        "n_rows": n_rows,
        "ordered_row_ids_sha256": ordered_row_ids_sha256,
        "row_set_sha256": row_set_sha256,
        "required_method_ids": sorted(set(required_method_ids)),
        "required_estimand_ids": sorted(set(required_estimand_ids)),
        "required_seeds": sorted(set(required_seeds)),
        "required_control_ids": sorted(set(required_control_ids)),
        "code_sha256": code_sha256,
        "config_sha256": config_sha256,
        "sign_accuracy_gate": SIGN_ACCURACY_GATE,
        "sign_gate_change_authorized": False,
        "ordinary_only": True,
        "sealed": False,
    }
    _validate_binding_fields(payload)
    return payload


def _validate_binding_fields(payload: Mapping) -> dict[str, Any]:
    _digest(payload.get("run_start_sha256"), label="run_start_sha256")
    _digest(payload.get("activation_sha256"), label="activation_sha256")
    _authority_hashes(payload.get("authority_hashes"))
    _nondata_authority_hashes(payload.get("nondata_authority_hashes"))
    _nondata_projection_hashes(payload.get("nondata_projection_hashes"))
    n_rows = payload.get("n_rows")
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows < 1:
        _fail("n_rows must be a positive integer")
    _digest(payload.get("ordered_row_ids_sha256"), label="ordered_row_ids_sha256")
    _digest(payload.get("row_set_sha256"), label="row_set_sha256")
    _canonical_ids(payload.get("required_method_ids"), label="required_method_ids")
    _canonical_ids(payload.get("required_estimand_ids"), label="required_estimand_ids")
    _canonical_seeds(payload.get("required_seeds"))
    controls = _canonical_ids(
        payload.get("required_control_ids"), label="required_control_ids"
    )
    if controls != sorted(REQUIRED_NEGATIVE_CONTROLS):
        _fail("required_control_ids must bind the seven frozen negative controls")
    _digest(payload.get("code_sha256"), label="code_sha256")
    _digest(payload.get("config_sha256"), label="config_sha256")
    gate = payload.get("sign_accuracy_gate")
    if (
        isinstance(gate, bool)
        or not isinstance(gate, (int, float))
        or float(gate) != SIGN_ACCURACY_GATE
    ):
        _fail(f"sign_accuracy_gate must remain exactly {SIGN_ACCURACY_GATE:.2f}")
    if payload.get("sign_gate_change_authorized") is not False:
        _fail("sign_gate_change_authorized must remain false")
    if payload.get("ordinary_only") is not True:
        _fail("staged commits must remain ordinary_only=true")
    if payload.get("sealed") is not False:
        _fail("staged commits must remain sealed=false")
    return dict(payload)


def required_predictor_unit_ids(bindings: Mapping) -> tuple[str, ...]:
    """Return the exact method/control x estimand x seed coverage."""

    _validate_binding_fields(bindings)
    units = []
    for estimand in bindings["required_estimand_ids"]:
        for method in bindings["required_method_ids"]:
            for seed in bindings["required_seeds"]:
                units.append(f"METHOD::{method}::ESTIMAND::{estimand}::SEED::{seed}")
        for control in bindings["required_control_ids"]:
            for seed in bindings["required_seeds"]:
                units.append(f"CONTROL::{control}::ESTIMAND::{estimand}::SEED::{seed}")
    return tuple(sorted(units))


def _prior_refs(value: object, *, state: str) -> dict[str, dict[str, str]]:
    expected_states = STATE_SEQUENCE[: STATE_SEQUENCE.index(state)]
    if not isinstance(value, Mapping) or set(value) != set(expected_states):
        actual = set(value) if isinstance(value, Mapping) else set()
        _fail(
            f"{state} prior_commit_refs differ from exact transition prefix: "
            f"{sorted(actual ^ set(expected_states))}"
        )
    normalized = {}
    for prior_state in expected_states:
        ref = _artifact_ref(value[prior_state], label=f"{prior_state} reference")
        if ref["relative_path"] != STATE_RELATIVE_PATHS[prior_state]:
            _fail(f"{prior_state} reference is swapped or not at its canonical path")
        normalized[prior_state] = ref
    return normalized


def _validate_common(payload: Mapping, *, state: str) -> dict:
    if payload.get("schema_version") != STATE_SCHEMAS[state]:
        _fail(f"unsupported {state} schema")
    if payload.get("artifact_role") != state:
        _fail(f"{state} artifact_role is invalid")
    if payload.get("state") != state:
        _fail(f"{state} state value is invalid")
    if payload.get("transition_index") != STATE_SEQUENCE.index(state) + 1:
        _fail(f"{state} transition_index is invalid")
    if payload.get("status") != "COMMITTED":
        _fail(f"{state} status must be COMMITTED")
    _prior_refs(payload.get("prior_commit_refs"), state=state)
    _validate_binding_fields(payload)
    return dict(payload)


def validate_development_commit(payload: Mapping) -> dict:
    _strict_keys(payload, DEVELOPMENT_KEYS, label=DEVELOPMENT_COMMIT)
    _validate_common(payload, state=DEVELOPMENT_COMMIT)
    authority_ref = _artifact_ref(
        payload.get("development_authority_ref"), label="development_authority_ref"
    )
    if authority_ref["sha256"] != payload["authority_hashes"][
        TRAIN_DEVELOPMENT_AUTHORITY
    ]:
        _fail("development authority ref disagrees with frozen authority hash")
    development_refs = _artifact_ref_map(
        payload.get("development_artifact_refs"), label="development_artifact_refs"
    )
    forbidden_hashes = {
        payload["authority_hashes"][INTERNAL_TEST_FEATURE_AUTHORITY],
        payload["authority_hashes"][INTERNAL_TEST_OUTCOME_AUTHORITY],
    }
    if authority_ref["sha256"] in forbidden_hashes or any(
        ref["sha256"] in forbidden_hashes for ref in development_refs.values()
    ):
        _fail("development commit cannot bind test feature or outcome authority bytes")
    _artifact_ref(payload.get("access_audit_ref"), label="access_audit_ref")
    _access_assertions(payload.get("access_assertions"), state=DEVELOPMENT_COMMIT)
    return dict(payload)


def validate_pre_test_freeze_commit(payload: Mapping) -> dict:
    _strict_keys(payload, PRE_TEST_FREEZE_KEYS, label=PRE_TEST_FREEZE_COMMIT)
    _validate_common(payload, state=PRE_TEST_FREEZE_COMMIT)
    selector_ref = _artifact_ref(payload.get("selector_ref"), label="selector_ref")
    provenance_refs = _artifact_ref_map(
        payload.get("predictor_provenance_refs"),
        label="predictor_provenance_refs",
        exact_keys=payload["required_method_ids"],
    )
    selected = _safe_id(payload.get("selected_method_id"), label="selected_method_id")
    if selected not in payload["required_method_ids"]:
        _fail("selected_method_id must be one of the frozen required methods")
    expected_units = required_predictor_unit_ids(payload)
    checkpoints = _artifact_ref_map(
        payload.get("frozen_checkpoint_refs"),
        label="frozen_checkpoint_refs",
        exact_keys=expected_units,
    )
    forbidden_hashes = {
        payload["authority_hashes"][INTERNAL_TEST_FEATURE_AUTHORITY],
        payload["authority_hashes"][INTERNAL_TEST_OUTCOME_AUTHORITY],
    }
    frozen_refs = [selector_ref, *provenance_refs.values(), *checkpoints.values()]
    if any(ref["sha256"] in forbidden_hashes for ref in frozen_refs):
        _fail("pre-test freeze cannot bind test feature or outcome authority bytes")
    identities = {(ref["relative_path"], ref["sha256"]) for ref in frozen_refs}
    if len(identities) != len(frozen_refs):
        _fail("selector, predictor provenance, and checkpoints must be distinct refs")
    audit_ref = _artifact_ref(payload.get("access_audit_ref"), label="access_audit_ref")
    if audit_ref["sha256"] in forbidden_hashes:
        _fail("pre-test access audit cannot be a test authority")
    _access_assertions(payload.get("access_assertions"), state=PRE_TEST_FREEZE_COMMIT)
    return dict(payload)


def _prediction_bindings(value: object, *, expected_units: Sequence[str]) -> dict:
    if not isinstance(value, Mapping) or set(value) != set(expected_units):
        actual = set(value) if isinstance(value, Mapping) else set()
        _fail(
            "frozen_prediction_refs does not cover the exact frozen units: "
            f"{sorted(actual ^ set(expected_units))}"
        )
    normalized = {}
    seen_predictions: set[tuple[str, str]] = set()
    for unit in sorted(value):
        entry = value[unit]
        _strict_keys(entry, PREDICTION_BINDING_KEYS, label=f"prediction binding {unit}")
        checkpoint_ref = _artifact_ref(
            entry.get("checkpoint_ref"), label=f"prediction checkpoint ref {unit}"
        )
        prediction_ref = _artifact_ref(
            entry.get("prediction_ref"), label=f"prediction artifact ref {unit}"
        )
        identity = (prediction_ref["relative_path"], prediction_ref["sha256"])
        if identity in seen_predictions:
            _fail("frozen_prediction_refs cannot reuse one prediction for multiple units")
        seen_predictions.add(identity)
        normalized[unit] = {
            "checkpoint_ref": checkpoint_ref,
            "prediction_ref": prediction_ref,
        }
    return normalized


def validate_test_prediction_commit(payload: Mapping) -> dict:
    _strict_keys(payload, TEST_PREDICTION_KEYS, label=TEST_PREDICTION_COMMIT)
    _validate_common(payload, state=TEST_PREDICTION_COMMIT)
    feature_ref = _artifact_ref(
        payload.get("feature_authority_ref"), label="feature_authority_ref"
    )
    if feature_ref["sha256"] != payload["authority_hashes"][
        INTERNAL_TEST_FEATURE_AUTHORITY
    ]:
        _fail("feature authority ref disagrees with frozen authority hash")
    predictions = _prediction_bindings(
        payload.get("frozen_prediction_refs"),
        expected_units=required_predictor_unit_ids(payload),
    )
    selected = _safe_id(payload.get("selected_method_id"), label="selected_method_id")
    if selected not in payload["required_method_ids"]:
        _fail("selected_method_id must be one of the frozen required methods")
    evaluation_ref = _artifact_ref(
        payload.get("evaluation_predictions_ref"),
        label="evaluation_predictions_ref",
    )
    forbidden_prediction_hashes = set(payload["authority_hashes"].values())
    if any(
        entry["prediction_ref"]["sha256"] in forbidden_prediction_hashes
        or entry["checkpoint_ref"]["sha256"] in forbidden_prediction_hashes
        for entry in predictions.values()
    ):
        _fail("test prediction artifacts cannot bind staged data-authority bytes")
    if evaluation_ref["sha256"] in forbidden_prediction_hashes:
        _fail("evaluation predictions cannot bind staged data-authority bytes")
    _access_assertions(payload.get("access_assertions"), state=TEST_PREDICTION_COMMIT)
    return dict(payload)


def validate_outcome_release_record(payload: Mapping) -> dict:
    _strict_keys(
        payload, OUTCOME_RELEASE_KEYS, label=ORDINARY_TEST_OUTCOME_RELEASE_RECORD
    )
    _validate_common(payload, state=ORDINARY_TEST_OUTCOME_RELEASE_RECORD)
    outcome_ref = _artifact_ref(
        payload.get("outcome_authority_ref"), label="outcome_authority_ref"
    )
    if outcome_ref["sha256"] != payload["authority_hashes"][
        INTERNAL_TEST_OUTCOME_AUTHORITY
    ]:
        _fail("outcome authority ref disagrees with frozen authority hash")
    claim_ref = _artifact_ref(payload.get("release_claim_ref"), label="release_claim_ref")
    if claim_ref["relative_path"] != OUTCOME_RELEASE_CLAIM_RELATIVE_PATH:
        _fail("release_claim_ref is not at the canonical one-time claim path")
    _access_assertions(
        payload.get("access_assertions"),
        state=ORDINARY_TEST_OUTCOME_RELEASE_RECORD,
    )
    if payload.get("consumed_once") is not True:
        _fail("outcome release record must assert consumed_once=true")
    return dict(payload)


VALIDATORS: dict[str, Callable[[Mapping], dict]] = {
    DEVELOPMENT_COMMIT: validate_development_commit,
    PRE_TEST_FREEZE_COMMIT: validate_pre_test_freeze_commit,
    TEST_PREDICTION_COMMIT: validate_test_prediction_commit,
    ORDINARY_TEST_OUTCOME_RELEASE_RECORD: validate_outcome_release_record,
}


def _unique_json_object(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _parse_canonical_json(raw: bytes, *, label: str) -> dict:
    try:
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_json_object
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise StagedCommitError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(payload, dict):
        _fail(f"{label} must be a JSON object")
    try:
        canonical = canonical_json_bytes(payload)
    except ArtifactIOError as exc:
        raise StagedCommitError(str(exc)) from exc
    if raw != canonical:
        _fail(f"{label} bytes are not exact canonical JSON")
    return payload


def _checked_run_root(run_root: Path) -> Path:
    root = Path(run_root)
    try:
        assert_ordinary_path(root, purpose="R3X staged commit run root")
    except ProtocolHoldError as exc:
        raise StagedCommitError(str(exc)) from exc
    absolute = Path(os.path.abspath(root))
    if absolute.is_symlink():
        _fail("staged commit run root cannot be a symlink")
    if not absolute.is_dir():
        _fail("staged commit run root must be an existing real directory")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise StagedCommitError(f"staged commit run root cannot be resolved: {exc}") from exc
    if absolute != resolved:
        _fail("staged commit run root cannot traverse a symlink")
    return resolved


def _commits_dir(root: Path, *, create: bool) -> Path:
    path = root / "commits"
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_dir():
            _fail("commits path must be a real directory")
    elif create:
        try:
            path.mkdir(mode=0o755)
        except FileExistsError:
            if path.is_symlink() or not path.is_dir():
                _fail("commits path was replaced during creation")
    return path


def _state_path(root: Path, state: str) -> Path:
    return root.joinpath(*Path(STATE_RELATIVE_PATHS[state]).parts)


def _path_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _assert_transition_prefix(root: Path, *, target_state: str) -> None:
    target_index = STATE_SEQUENCE.index(target_state)
    _commits_dir(root, create=True)
    for index, state in enumerate(STATE_SEQUENCE):
        path = _state_path(root, state)
        present = _path_present(path)
        if index < target_index and not present:
            _fail(f"cannot create {target_state}: missing prior {state}")
        if index >= target_index and present:
            if index == target_index:
                _fail(f"{state} already exists; immutable state cannot be overwritten")
            _fail(f"cannot create {target_state}: later state {state} already exists")
    claim = root.joinpath(*Path(OUTCOME_RELEASE_CLAIM_RELATIVE_PATH).parts)
    if _path_present(claim):
        if target_state == ORDINARY_TEST_OUTCOME_RELEASE_RECORD:
            _fail(
                "partial outcome release claim already exists; outcome access is "
                "fail-closed and cannot be retried"
            )
        _fail("outcome release claim exists before its legal transition")


def _common_payload(*, state: str, bindings: Mapping, prior_refs: Mapping) -> dict:
    binding = dict(bindings)
    _validate_binding_fields(binding)
    payload = {
        "schema_version": STATE_SCHEMAS[state],
        "artifact_role": state,
        "state": state,
        "transition_index": STATE_SEQUENCE.index(state) + 1,
        "status": "COMMITTED",
        "prior_commit_refs": dict(prior_refs),
        **binding,
    }
    return payload


def _publish_state(root: Path, state: str, payload: Mapping) -> dict:
    path = _state_path(root, state)
    validator = VALIDATORS[state]
    try:
        digest = atomic_write_json_exclusive(path, payload, validator=validator)
        reference = make_artifact_ref(root, path, digest)
    except (ArtifactIOError, ArtifactGraphError) as exc:
        raise StagedCommitError(str(exc)) from exc
    return {"payload": dict(payload), "sha256": digest, "ref": reference, "path": path}


def _read_state(root: Path, state: str, reference: Mapping) -> dict:
    ref = _artifact_ref(reference, label=f"{state} reference")
    if ref["relative_path"] != STATE_RELATIVE_PATHS[state]:
        _fail(f"{state} reference is not at its canonical path")
    try:
        raw = read_artifact_ref(root, ref)
    except ArtifactGraphError as exc:
        raise StagedCommitError(str(exc)) from exc
    payload = _parse_canonical_json(raw, label=state)
    VALIDATORS[state](payload)
    return {"payload": payload, "sha256": ref["sha256"], "ref": ref}


def _replay_ref(root: Path, reference: Mapping, *, label: str) -> bytes:
    try:
        return read_artifact_ref(root, reference)
    except ArtifactGraphError as exc:
        raise StagedCommitError(f"{label} cannot be replayed: {exc}") from exc


def _replay_ref_map(root: Path, references: Mapping, *, label: str) -> None:
    for key, reference in references.items():
        _replay_ref(root, reference, label=f"{label}[{key}]")


def _binding_snapshot(payload: Mapping) -> dict:
    return {
        key: payload[key]
        for key in (
            "run_start_sha256",
            "activation_sha256",
            "authority_hashes",
            "nondata_authority_hashes",
            "nondata_projection_hashes",
            "n_rows",
            "ordered_row_ids_sha256",
            "row_set_sha256",
            "required_method_ids",
            "required_estimand_ids",
            "required_seeds",
            "required_control_ids",
            "code_sha256",
            "config_sha256",
            "sign_accuracy_gate",
            "sign_gate_change_authorized",
            "ordinary_only",
            "sealed",
        )
    }


def extract_staged_bindings(payload: Mapping) -> dict:
    """Return the exact immutable binding carried by a validated DAG state."""

    state = payload.get("state") if isinstance(payload, Mapping) else None
    if state not in VALIDATORS:
        _fail("cannot extract bindings from an unknown staged state")
    VALIDATORS[state](payload)
    return _binding_snapshot(payload)


def _require_same_bindings(left: Mapping, right: Mapping, *, label: str) -> None:
    if _binding_snapshot(left) != _binding_snapshot(right):
        _fail(f"{label} changed a frozen cross-stage binding")


def write_development_commit_exclusive(
    run_root: Path,
    *,
    bindings: Mapping,
    development_authority_ref: Mapping,
    development_artifact_refs: Mapping[str, Mapping],
    access_audit_ref: Mapping,
    access_assertions: Mapping[str, bool],
) -> dict:
    root = _checked_run_root(run_root)
    _assert_transition_prefix(root, target_state=DEVELOPMENT_COMMIT)
    payload = {
        **_common_payload(state=DEVELOPMENT_COMMIT, bindings=bindings, prior_refs={}),
        "development_authority_ref": dict(development_authority_ref),
        "development_artifact_refs": {
            key: dict(value) for key, value in development_artifact_refs.items()
        },
        "access_audit_ref": dict(access_audit_ref),
        "access_assertions": dict(access_assertions),
    }
    validate_development_commit(payload)
    _replay_ref(root, development_authority_ref, label="development authority")
    _replay_ref_map(root, development_artifact_refs, label="development artifact")
    _replay_ref(root, access_audit_ref, label="development access audit")
    return _publish_state(root, DEVELOPMENT_COMMIT, payload)


def replay_development_commit(
    run_root: Path,
    reference: Mapping,
    *,
    replay_data_authority: bool = True,
) -> dict:
    """Replay a development commit.

    Physical disclosure callers set ``replay_data_authority=False`` so that
    verifying the immutable metadata DAG cannot reopen a data partition.  All
    non-data development artifacts and the access audit are still replayed.
    """

    root = _checked_run_root(run_root)
    result = _read_state(root, DEVELOPMENT_COMMIT, reference)
    payload = result["payload"]
    if replay_data_authority:
        _replay_ref(
            root,
            payload["development_authority_ref"],
            label="development authority",
        )
    _replay_ref_map(
        root, payload["development_artifact_refs"], label="development artifact"
    )
    _replay_ref(root, payload["access_audit_ref"], label="development access audit")
    return result


def write_pre_test_freeze_commit_exclusive(
    run_root: Path,
    *,
    bindings: Mapping,
    development_commit_ref: Mapping,
    selector_ref: Mapping,
    predictor_provenance_refs: Mapping[str, Mapping],
    selected_method_id: str,
    frozen_checkpoint_refs: Mapping[str, Mapping],
    access_audit_ref: Mapping,
    access_assertions: Mapping[str, bool],
) -> dict:
    root = _checked_run_root(run_root)
    _assert_transition_prefix(root, target_state=PRE_TEST_FREEZE_COMMIT)
    development = replay_development_commit(root, development_commit_ref)
    _require_same_bindings(bindings, development["payload"], label=PRE_TEST_FREEZE_COMMIT)
    prior_refs = {DEVELOPMENT_COMMIT: dict(development_commit_ref)}
    payload = {
        **_common_payload(
            state=PRE_TEST_FREEZE_COMMIT, bindings=bindings, prior_refs=prior_refs
        ),
        "selector_ref": dict(selector_ref),
        "predictor_provenance_refs": {
            key: dict(value) for key, value in predictor_provenance_refs.items()
        },
        "selected_method_id": selected_method_id,
        "frozen_checkpoint_refs": {
            key: dict(value) for key, value in frozen_checkpoint_refs.items()
        },
        "access_audit_ref": dict(access_audit_ref),
        "access_assertions": dict(access_assertions),
    }
    validate_pre_test_freeze_commit(payload)
    if payload["access_audit_ref"] == development["payload"]["access_audit_ref"]:
        _fail("pre-test freeze requires a distinct post-freeze access audit artifact")
    _replay_ref(root, selector_ref, label="development selector")
    _replay_ref_map(
        root,
        predictor_provenance_refs,
        label="development predictor provenance",
    )
    _replay_ref_map(root, frozen_checkpoint_refs, label="frozen checkpoint")
    _replay_ref(root, access_audit_ref, label="pre-test access audit")
    return _publish_state(root, PRE_TEST_FREEZE_COMMIT, payload)


def replay_pre_test_freeze_commit(
    run_root: Path,
    reference: Mapping,
    *,
    replay_data_authorities: bool = True,
) -> dict:
    root = _checked_run_root(run_root)
    result = _read_state(root, PRE_TEST_FREEZE_COMMIT, reference)
    payload = result["payload"]
    development = replay_development_commit(
        root,
        payload["prior_commit_refs"][DEVELOPMENT_COMMIT],
        replay_data_authority=replay_data_authorities,
    )
    _require_same_bindings(payload, development["payload"], label=PRE_TEST_FREEZE_COMMIT)
    if payload["access_audit_ref"] == development["payload"]["access_audit_ref"]:
        _fail("pre-test freeze reused the development access audit artifact")
    _replay_ref(root, payload["selector_ref"], label="development selector")
    _replay_ref_map(
        root,
        payload["predictor_provenance_refs"],
        label="development predictor provenance",
    )
    _replay_ref_map(root, payload["frozen_checkpoint_refs"], label="frozen checkpoint")
    _replay_ref(root, payload["access_audit_ref"], label="pre-test access audit")
    return {**result, "development": development}


def write_test_prediction_commit_exclusive(
    run_root: Path,
    *,
    bindings: Mapping,
    development_commit_ref: Mapping,
    pre_test_freeze_commit_ref: Mapping,
    feature_authority_ref: Mapping,
    evaluation_predictions_ref: Mapping,
    frozen_prediction_refs: Mapping[str, Mapping],
    access_assertions: Mapping[str, bool],
    outcome_authority_ref: Mapping | None = None,
) -> dict:
    """Commit feature-only predictions; any outcome reference is premature."""

    if outcome_authority_ref is not None:
        _fail("outcome authority cannot be supplied before outcome release")
    root = _checked_run_root(run_root)
    _assert_transition_prefix(root, target_state=TEST_PREDICTION_COMMIT)
    freeze = replay_pre_test_freeze_commit(
        root,
        pre_test_freeze_commit_ref,
        replay_data_authorities=False,
    )
    if freeze["payload"]["prior_commit_refs"][DEVELOPMENT_COMMIT] != dict(
        development_commit_ref
    ):
        _fail("TEST_PREDICTION_COMMIT received a swapped development reference")
    _require_same_bindings(bindings, freeze["payload"], label=TEST_PREDICTION_COMMIT)
    expected_units = required_predictor_unit_ids(bindings)
    if set(frozen_prediction_refs) != set(expected_units):
        _fail("frozen_prediction_refs does not cover every frozen method/control unit")
    prediction_bindings = {
        unit: {
            "checkpoint_ref": dict(freeze["payload"]["frozen_checkpoint_refs"][unit]),
            "prediction_ref": dict(frozen_prediction_refs[unit]),
        }
        for unit in expected_units
    }
    prior_refs = {
        DEVELOPMENT_COMMIT: dict(development_commit_ref),
        PRE_TEST_FREEZE_COMMIT: dict(pre_test_freeze_commit_ref),
    }
    payload = {
        **_common_payload(
            state=TEST_PREDICTION_COMMIT, bindings=bindings, prior_refs=prior_refs
        ),
        "feature_authority_ref": dict(feature_authority_ref),
        "evaluation_predictions_ref": dict(evaluation_predictions_ref),
        "selected_method_id": freeze["payload"]["selected_method_id"],
        "frozen_prediction_refs": prediction_bindings,
        "access_assertions": dict(access_assertions),
    }
    validate_test_prediction_commit(payload)
    _replay_ref(root, feature_authority_ref, label="internal-test feature authority")
    _replay_ref(root, evaluation_predictions_ref, label="evaluation predictions")
    _replay_ref_map(root, frozen_prediction_refs, label="frozen prediction")
    return _publish_state(root, TEST_PREDICTION_COMMIT, payload)


def replay_test_prediction_commit(
    run_root: Path,
    reference: Mapping,
    *,
    replay_data_authorities: bool = True,
) -> dict:
    root = _checked_run_root(run_root)
    result = _read_state(root, TEST_PREDICTION_COMMIT, reference)
    payload = result["payload"]
    freeze = replay_pre_test_freeze_commit(
        root,
        payload["prior_commit_refs"][PRE_TEST_FREEZE_COMMIT],
        replay_data_authorities=replay_data_authorities,
    )
    if payload["prior_commit_refs"][DEVELOPMENT_COMMIT] != freeze["payload"][
        "prior_commit_refs"
    ][DEVELOPMENT_COMMIT]:
        _fail("test prediction development reference is swapped")
    _require_same_bindings(payload, freeze["payload"], label=TEST_PREDICTION_COMMIT)
    if payload["selected_method_id"] != freeze["payload"]["selected_method_id"]:
        _fail("test prediction selected method changed after pre-test freeze")
    for unit, entry in payload["frozen_prediction_refs"].items():
        if entry["checkpoint_ref"] != freeze["payload"]["frozen_checkpoint_refs"][unit]:
            _fail(f"prediction binding {unit} does not point to its frozen checkpoint")
        _replay_ref(root, entry["prediction_ref"], label=f"frozen prediction[{unit}]")
    _replay_ref(
        root,
        payload["evaluation_predictions_ref"],
        label="evaluation predictions",
    )
    if replay_data_authorities:
        _replay_ref(root, payload["feature_authority_ref"], label="feature authority")
    return {**result, "freeze": freeze}


CLAIM_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "prediction_commit_ref",
    "outcome_authority_ref",
    "run_start_sha256",
    "activation_sha256",
    "authority_hashes",
    "nondata_authority_hashes",
    "nondata_projection_hashes",
    "ordinary_only",
    "sealed",
}


def _validate_release_claim(payload: Mapping) -> dict:
    _strict_keys(payload, CLAIM_KEYS, label="outcome release claim")
    if payload.get("schema_version") != OUTCOME_RELEASE_CLAIM_SCHEMA:
        _fail("unsupported outcome release claim schema")
    if payload.get("artifact_role") != "ORDINARY_TEST_OUTCOME_RELEASE_CLAIM":
        _fail("outcome release claim artifact_role is invalid")
    if payload.get("status") != "CLAIMED_BEFORE_OUTCOME_READ":
        _fail("outcome release claim status is invalid")
    prediction_ref = _artifact_ref(
        payload.get("prediction_commit_ref"), label="claim prediction_commit_ref"
    )
    if prediction_ref["relative_path"] != STATE_RELATIVE_PATHS[TEST_PREDICTION_COMMIT]:
        _fail("claim prediction ref is not canonical")
    outcome_ref = _artifact_ref(
        payload.get("outcome_authority_ref"), label="claim outcome_authority_ref"
    )
    _digest(payload.get("run_start_sha256"), label="claim run_start_sha256")
    _digest(payload.get("activation_sha256"), label="claim activation_sha256")
    authority_hashes = _authority_hashes(payload.get("authority_hashes"))
    _nondata_authority_hashes(payload.get("nondata_authority_hashes"))
    _nondata_projection_hashes(payload.get("nondata_projection_hashes"))
    if outcome_ref["sha256"] != authority_hashes[INTERNAL_TEST_OUTCOME_AUTHORITY]:
        _fail("claim outcome ref disagrees with frozen authority hash")
    if payload.get("ordinary_only") is not True or payload.get("sealed") is not False:
        _fail("outcome release claim must remain ordinary-only and unsealed")
    return dict(payload)


def _publish_release_claim(root: Path, payload: Mapping) -> dict:
    path = root.joinpath(*Path(OUTCOME_RELEASE_CLAIM_RELATIVE_PATH).parts)
    try:
        digest = atomic_write_json_exclusive(
            path, payload, validator=_validate_release_claim
        )
        reference = make_artifact_ref(root, path, digest)
    except (ArtifactIOError, ArtifactGraphError) as exc:
        raise StagedCommitError(str(exc)) from exc
    return {"payload": dict(payload), "sha256": digest, "ref": reference}


def _replay_release_claim(root: Path, reference: Mapping) -> dict:
    ref = _artifact_ref(reference, label="release claim reference")
    if ref["relative_path"] != OUTCOME_RELEASE_CLAIM_RELATIVE_PATH:
        _fail("release claim reference is not canonical")
    try:
        raw = read_artifact_ref(root, ref)
    except ArtifactGraphError as exc:
        raise StagedCommitError(str(exc)) from exc
    payload = _parse_canonical_json(raw, label="outcome release claim")
    _validate_release_claim(payload)
    return {"payload": payload, "sha256": ref["sha256"], "ref": ref}


def write_ordinary_test_outcome_release_record_exclusive(
    run_root: Path,
    *,
    bindings: Mapping,
    development_commit_ref: Mapping,
    pre_test_freeze_commit_ref: Mapping,
    test_prediction_commit_ref: Mapping,
    outcome_authority_ref: Mapping,
    access_assertions: Mapping[str, bool],
    _after_claim_hook: Callable[[], None] | None = None,
) -> dict:
    """Read ordinary outcome bytes once and publish their durable release record.

    ``_after_claim_hook`` exists only as a deterministic crash-injection seam
    for recovery tests.  Production callers should leave it unset.
    """

    root = _checked_run_root(run_root)
    _assert_transition_prefix(
        root, target_state=ORDINARY_TEST_OUTCOME_RELEASE_RECORD
    )
    prediction = replay_test_prediction_commit(
        root,
        test_prediction_commit_ref,
        replay_data_authorities=False,
    )
    expected_dev = prediction["payload"]["prior_commit_refs"][DEVELOPMENT_COMMIT]
    expected_freeze = prediction["payload"]["prior_commit_refs"][
        PRE_TEST_FREEZE_COMMIT
    ]
    if dict(development_commit_ref) != expected_dev:
        _fail("outcome release received a swapped development reference")
    if dict(pre_test_freeze_commit_ref) != expected_freeze:
        _fail("outcome release received a swapped pre-test freeze reference")
    _require_same_bindings(
        bindings, prediction["payload"], label=ORDINARY_TEST_OUTCOME_RELEASE_RECORD
    )
    _access_assertions(
        access_assertions, state=ORDINARY_TEST_OUTCOME_RELEASE_RECORD
    )
    outcome_ref = _artifact_ref(outcome_authority_ref, label="outcome_authority_ref")
    if outcome_ref["sha256"] != bindings["authority_hashes"][
        INTERNAL_TEST_OUTCOME_AUTHORITY
    ]:
        _fail("outcome authority ref disagrees with frozen authority hash")

    claim_payload = {
        "schema_version": OUTCOME_RELEASE_CLAIM_SCHEMA,
        "artifact_role": "ORDINARY_TEST_OUTCOME_RELEASE_CLAIM",
        "status": "CLAIMED_BEFORE_OUTCOME_READ",
        "prediction_commit_ref": dict(test_prediction_commit_ref),
        "outcome_authority_ref": dict(outcome_authority_ref),
        "run_start_sha256": bindings["run_start_sha256"],
        "activation_sha256": bindings["activation_sha256"],
        "authority_hashes": dict(bindings["authority_hashes"]),
        "nondata_authority_hashes": dict(bindings["nondata_authority_hashes"]),
        "nondata_projection_hashes": dict(bindings["nondata_projection_hashes"]),
        "ordinary_only": True,
        "sealed": False,
    }
    claim = _publish_release_claim(root, claim_payload)

    # From this point onward every failure is intentionally non-retryable.
    if _after_claim_hook is not None:
        _after_claim_hook()
    outcome_bytes = _replay_ref(
        root, outcome_authority_ref, label="ordinary internal-test outcome authority"
    )

    prior_refs = {
        DEVELOPMENT_COMMIT: dict(development_commit_ref),
        PRE_TEST_FREEZE_COMMIT: dict(pre_test_freeze_commit_ref),
        TEST_PREDICTION_COMMIT: dict(test_prediction_commit_ref),
    }
    payload = {
        **_common_payload(
            state=ORDINARY_TEST_OUTCOME_RELEASE_RECORD,
            bindings=bindings,
            prior_refs=prior_refs,
        ),
        "outcome_authority_ref": dict(outcome_authority_ref),
        "release_claim_ref": claim["ref"],
        "access_assertions": dict(access_assertions),
        "consumed_once": True,
    }
    validate_outcome_release_record(payload)
    result = _publish_state(root, ORDINARY_TEST_OUTCOME_RELEASE_RECORD, payload)
    return {**result, "outcome_bytes": outcome_bytes, "claim": claim}


def replay_ordinary_test_outcome_release_record(
    run_root: Path, reference: Mapping
) -> dict:
    """Replay a completed release without reopening outcome authority bytes."""

    root = _checked_run_root(run_root)
    result = _read_state(root, ORDINARY_TEST_OUTCOME_RELEASE_RECORD, reference)
    payload = result["payload"]
    prediction = replay_test_prediction_commit(
        root,
        payload["prior_commit_refs"][TEST_PREDICTION_COMMIT],
        replay_data_authorities=False,
    )
    expected_prior = {
        DEVELOPMENT_COMMIT: prediction["payload"]["prior_commit_refs"][
            DEVELOPMENT_COMMIT
        ],
        PRE_TEST_FREEZE_COMMIT: prediction["payload"]["prior_commit_refs"][
            PRE_TEST_FREEZE_COMMIT
        ],
        TEST_PREDICTION_COMMIT: payload["prior_commit_refs"][
            TEST_PREDICTION_COMMIT
        ],
    }
    if payload["prior_commit_refs"] != expected_prior:
        _fail("outcome release prior refs are swapped or inconsistent")
    _require_same_bindings(
        payload, prediction["payload"], label=ORDINARY_TEST_OUTCOME_RELEASE_RECORD
    )
    claim = _replay_release_claim(root, payload["release_claim_ref"])
    expected_claim = {
        "prediction_commit_ref": payload["prior_commit_refs"][
            TEST_PREDICTION_COMMIT
        ],
        "outcome_authority_ref": payload["outcome_authority_ref"],
        "run_start_sha256": payload["run_start_sha256"],
        "activation_sha256": payload["activation_sha256"],
        "authority_hashes": payload["authority_hashes"],
        "nondata_authority_hashes": payload["nondata_authority_hashes"],
        "nondata_projection_hashes": payload["nondata_projection_hashes"],
        "ordinary_only": True,
        "sealed": False,
    }
    for key, value in expected_claim.items():
        if claim["payload"].get(key) != value:
            _fail(f"release claim disagrees with outcome release record at {key}")
    # Deliberately do not call read_artifact_ref on outcome_authority_ref here.
    return {**result, "prediction": prediction, "claim": claim}
