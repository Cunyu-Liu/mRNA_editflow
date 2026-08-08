"""Exact-replayable pure-NumPy fitted states for R3X scalar baselines.

This module deliberately covers only the six deterministic scalar methods.  It
does not implement or certify HistogramGradientBoosting and cannot be used to
claim completion of the contract's exact-eight suite.  Every fitted state is
rebuilt from canonical ORDINARY_TRAIN predictor-input rows before publication
and after exact loading.  Development predictions are then recomputed from the
loaded state and target-free ORDINARY_DEVELOPMENT rows; caller-supplied
predictions are never accepted.

The typed predictor-input schema intentionally omits source-component labels.
Those labels enter only through a separate, canonical, hash-bound train-only
assignment artifact.  A fitter never accepts a raw component mapping.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from numbers import Real
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .artifact_io import (
    ArtifactIOError,
    atomic_write_json_exclusive,
    canonical_json_bytes,
    read_exact_regular_file,
)
from .features import (
    candidate_kmer_features,
    edit_only_features,
    matrix,
    paired_kmer_features,
)
from .information_sets import (
    InformationSetError,
    information_set_fingerprint,
)
from .predictor_provenance import (
    DEVELOPMENT_PARTITION,
    NON_DATA_AUTHORITY_HASH_KEYS,
    TRAIN_PARTITION,
    PredictorProvenanceError,
    normalize_predictor_input_rows,
    predictor_input_jsonl_bytes,
)


SCALAR_FITTED_STATE_SCHEMA = "r3x_scalar_fitted_state_v2"
SCALAR_FIT_CONFIG_SCHEMA = "r3x_scalar_fit_config_v2"
SCALAR_FEATURE_SCHEMA = "r3x_scalar_feature_maps_v2"
SCALAR_TRAINING_DESIGN_SCHEMA = "r3x_scalar_training_design_binding_v1"
TRAIN_COMPONENT_ASSIGNMENT_SCHEMA = (
    "r3x_train_source_component_assignment_v2"
)
V1_TRAIN_COMPONENT_ASSIGNMENT_SCHEMA = (
    "r3x_train_source_component_assignment_v1"
)

SCALAR_FITTED_STATE_ROLE = "ORDINARY_DEVELOPMENT_EXACT_SCALAR_FITTED_STATE"
SCALAR_FITTED_STATE_STATUS = "COMPLETE_EXACT_REPLAYABLE_SCALAR_FIT"
TRAIN_COMPONENT_ASSIGNMENT_ROLE = "ORDINARY_TRAIN_SOURCE_COMPONENT_ASSIGNMENT"
TRAIN_COMPONENT_ASSIGNMENT_STATUS = (
    "COMPLETE_EXACT_SHARED_TRAIN_SOURCE_COMPONENT_ASSIGNMENT"
)
TRAIN_COMPONENT_ASSIGNMENT_SCOPE = (
    "INFORMATION_SET_INDEPENDENT_SHARED_GLOBAL_COMPONENT_GROUPING_V1"
)

SUPPORTED_SCALAR_METHOD_IDS = frozenset({
    "ZERO_DELTA",
    "SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR",
    "EDIT_ONLY_RIDGE",
    "CANDIDATE_KMER_RIDGE",
    "PAIRED_LOCAL_GLOBAL_KMER_RIDGE",
    "PAIRED_FEATURE_NEAREST_NEIGHBOR",
})
HGB_METHOD_ID = "HISTOGRAM_GRADIENT_BOOSTING"
HGB_BLOCKER = (
    "HISTOGRAM_GRADIENT_BOOSTING requires a separately trusted sklearn "
    "fitted-state schema and independent environment-qualified replay"
)
EXACT_EIGHT_COMPLETE = False

METHOD_INFORMATION_SETS = {
    "ZERO_DELTA": "NO_ANCHOR_ZERO_PRIOR",
    "SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR": "NO_ANCHOR_ZERO_PRIOR",
    "EDIT_ONLY_RIDGE": "EDIT_ONLY",
    "CANDIDATE_KMER_RIDGE": "CANDIDATE_ONLY",
    "PAIRED_LOCAL_GLOBAL_KMER_RIDGE": (
        "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR"
    ),
    "PAIRED_FEATURE_NEAREST_NEIGHBOR": (
        "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR"
    ),
}


def _canonical_method_information_set_fingerprint(method_id: str) -> str:
    try:
        return information_set_fingerprint(METHOD_INFORMATION_SETS[method_id])
    except (KeyError, InformationSetError) as exc:
        raise ScalarFittedStateError(
            f"{method_id} has no canonical frozen information-set fingerprint"
        ) from exc


def _method_information_set_fingerprint(
    method_id: str, claimed: object, *, label: str
) -> str:
    parsed = _digest(claimed, label=label)
    canonical = _canonical_method_information_set_fingerprint(method_id)
    if parsed != canonical:
        raise ScalarFittedStateError(
            f"{label} differs from the canonical fingerprint for "
            f"{METHOD_INFORMATION_SETS[method_id]}"
        )
    return canonical

RIDGE_METHOD_IDS = frozenset({
    "EDIT_ONLY_RIDGE",
    "CANDIDATE_KMER_RIDGE",
    "PAIRED_LOCAL_GLOBAL_KMER_RIDGE",
})
COMPONENT_ASSIGNMENT_REQUIRED_METHOD_IDS = frozenset(
    SUPPORTED_SCALAR_METHOD_IDS - {"ZERO_DELTA"}
)
DEFAULT_RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
DEFAULT_NN_TRAIN_CHUNK_SIZE = 4096
NN_TIE_ATOL = 1e-12
ROW_ORDER_RULE = "CANONICAL_ASCENDING_ROW_ID"
FORMAL_WEIGHT_POLICY = (
    "CELL_THEN_GLOBAL_SOURCE_COMPONENT_THEN_ROW_"
    "SUM_TO_ONE_WEIGHTED_RISK_V1"
)
RIDGE_TRAINING_POLICY = (
    "FORMAL_WEIGHTED_SCALER_INTERCEPT_AND_SQUARED_ERROR_PLUS_ALPHA_L2_V2"
)
RIDGE_CV_RULE = (
    "TRAIN_ONLY_GLOBAL_SOURCE_COMPONENT_GROUPED_MAX5_FOLDS_"
    "SINGLE_FORMAL_OOF_WEIGHTED_RISK_V2"
)
RIDGE_CV_SCORING_POLICY = (
    "ONE_OOF_PREDICTION_PER_ROW_THEN_CELL_COMPONENT_ROW_WEIGHTED_MSE_V1"
)
PRIOR_TRAINING_POLICY = (
    "CONDITION_LOCAL_AND_GLOBAL_FALLBACK_CELL_COMPONENT_ROW_MEAN_V2"
)
NN_SCALING_POLICY = "TRAIN_ONLY_FORMAL_WEIGHTED_MEAN_VARIANCE_V2"
NN_TIE_RULE = (
    "EXACT_TWO_PASS_SQUARED_EUCLIDEAN_RTL0_ATOL1E-12_"
    "CANONICAL_ROW_ID_CELL_COMPONENT_ROW_MATH_FSUM_MEAN_V2"
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_METHOD_ID = re.compile(r"^[A-Z][A-Z0-9_]*$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_PROTECTED_TOKENS = ("internal_test", "gse246381")
_PROTECTED_PATH_PARTS = {"test", "internal_test", "sealed", "restricted"}

_ROW_BINDING_KEYS = {
    "n_rows",
    "ordered_row_ids_sha256",
    "row_set_sha256",
    "content_sha256",
}
_BOUNDARY_KEYS = {
    "ordinary_only",
    "internal_test_access",
    "sealed_access",
    "downstream_unlock",
}
_ASSIGNMENT_INPUT_ROW_KEYS = {"row_id", "source_component_id"}
_ASSIGNMENT_ROW_KEYS = {
    "row_id",
    "partition",
    "cell_id",
    "source_component_id",
}
_ASSIGNMENT_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "assignment_scope",
    "train_development_outcomes_sha256",
    "non_data_authority_hashes",
    "source_component_split_manifest_source_sha256",
    "source_component_split_manifest_projection_sha256",
    "train_row_binding",
    "assignments",
} | _BOUNDARY_KEYS
_CONDITION_VOCAB_KEYS = {
    "endpoint",
    "publication_context_id",
    "region",
}
_FIT_CONFIG_KEYS = {
    "schema_version",
    "method_id",
    "row_order_rule",
    "target_field",
    "ridge_alpha_grid",
    "ridge_cv_rule",
    "ridge_cv_scoring_policy",
    "training_weight_policy",
    "weighted_risk_normalization",
    "ridge_training_policy",
    "prior_training_policy",
    "nearest_neighbor_train_chunk_size",
    "nearest_neighbor_scaling_policy",
    "nearest_neighbor_tie_rule",
}
_FEATURE_SCHEMA_KEYS = {
    "schema_version",
    "feature_map_id",
    "base_feature_dimension",
    "condition_encoding",
    "condition_feature_dimension",
    "total_feature_dimension",
    "max_k",
    "local_radius",
}
_COMPONENT_BINDING_KEYS = {
    "schema_version",
    "sha256",
    "source_component_split_manifest_source_sha256",
    "source_component_split_manifest_projection_sha256",
}
_TRAINING_DESIGN_KEYS = {
    "schema_version",
    "design_sha256",
    "formal_weights_sha256",
    "training_weight_policy",
    "ordered_cell_ids_sha256",
    "ordered_source_component_ids_sha256",
    "cell_ids",
    "source_component_ids",
}
_STATE_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "method_id",
    "information_set",
    "information_set_fingerprint",
    "estimand_id",
    "train_development_outcomes_sha256",
    "non_data_authority_hashes",
    "train_row_binding",
    "source_component_assignment_binding",
    "training_design_binding",
    "fit_config",
    "fit_config_sha256",
    "feature_schema",
    "condition_vocab",
    "state",
} | _BOUNDARY_KEYS

_ZERO_STATE_KEYS = {"kind", "constant"}
_PRIOR_STATE_KEYS = {
    "kind",
    "condition_formal_values",
    "fallback_cell_component_values",
    "fallback_value",
}
_PRIOR_CONDITION_KEYS = {
    "endpoint",
    "publication_context_id",
    "region",
    "cell_ids",
    "source_component_ids",
    "cell_component_values",
    "formal_value",
}
_CELL_COMPONENT_VALUE_KEYS = {
    "cell_id",
    "source_component_id",
    "n_rows",
    "row_mean",
}
_RIDGE_STATE_KEYS = {
    "kind",
    "scaler_mean",
    "scaler_scale",
    "y_mean",
    "coef",
    "alpha",
    "solver",
    "square_system_dimension",
    "n_train_rows",
    "n_features",
    "cv_status",
    "cv_n_global_source_components",
    "cv_n_folds",
    "cv_fold_assignment_sha256",
    "cv_formal_oof_weighted_mse",
}
_CV_SCORE_KEYS = {"alpha", "formal_oof_weighted_mse"}
_NN_STATE_KEYS = {
    "kind",
    "scaler_mean",
    "scaler_scale",
    "train_row_ids",
    "train_cell_ids",
    "train_source_component_ids",
    "train_standardized_features",
    "train_targets",
    "tie_atol",
    "tie_rule",
}


class ScalarFittedStateError(RuntimeError):
    """Fail-closed scalar fitted-state or component-authority error."""


def _strict_keys(value: Mapping, expected: set[str], *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ScalarFittedStateError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ScalarFittedStateError(
            f"{label} keys differ from strict schema: {sorted(actual ^ expected)}"
        )


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise ScalarFittedStateError(f"{label} must be 64 lowercase hex")
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ScalarFittedStateError(f"{label} must be a canonical identifier")
    lowered = value.lower()
    if any(token in lowered for token in _PROTECTED_TOKENS):
        raise ScalarFittedStateError(f"{label} enters a protected namespace")
    if lowered.startswith("sealed") or lowered.startswith("restricted"):
        raise ScalarFittedStateError(f"{label} enters a protected namespace")
    return value


def _feature_text(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise ScalarFittedStateError(f"{label} must be canonical non-empty text")
    lowered = value.lower()
    if any(token in lowered for token in _PROTECTED_TOKENS):
        raise ScalarFittedStateError(f"{label} enters a protected namespace")
    if lowered.startswith("sealed") or lowered.startswith("restricted"):
        raise ScalarFittedStateError(f"{label} enters a protected namespace")
    return value


def _method_id(value: object) -> str:
    if not isinstance(value, str) or not _METHOD_ID.fullmatch(value):
        raise ScalarFittedStateError("method_id must be a canonical method ID")
    if value == HGB_METHOD_ID:
        raise ScalarFittedStateError(HGB_BLOCKER)
    if value not in SUPPORTED_SCALAR_METHOD_IDS:
        raise ScalarFittedStateError(
            f"unsupported scalar fitted-state method: {value}"
        )
    return value


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ScalarFittedStateError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ScalarFittedStateError(f"{label} must be a finite number")
    return result


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ScalarFittedStateError(f"{label} must be a positive integer")
    return value


def _canonical_sha256(value: object) -> str:
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ScalarFittedStateError(
            f"value is not canonical finite JSON: {exc}"
        ) from exc
    return hashlib.sha256(raw).hexdigest()


def _non_data_authority_hashes(value: Mapping) -> dict[str, str]:
    _strict_keys(
        value,
        set(NON_DATA_AUTHORITY_HASH_KEYS),
        label="non_data_authority_hashes",
    )
    return {
        key: _digest(value[key], label=f"non_data_authority_hashes.{key}")
        for key in NON_DATA_AUTHORITY_HASH_KEYS
    }


def _boundaries(value: Mapping, *, label: str) -> None:
    expected = {
        "ordinary_only": True,
        "internal_test_access": False,
        "sealed_access": False,
        "downstream_unlock": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) is not expected_value:
            raise ScalarFittedStateError(
                f"{label}.{key} violates ordinary-only boundaries"
            )


def _assert_development_path(path: Path, *, label: str) -> Path:
    candidate = Path(path)
    parts = [part.lower() for part in candidate.parts]
    if "development" not in parts:
        raise ScalarFittedStateError(
            f"{label} must be beneath a literal development namespace"
        )
    for part in parts:
        if (
            part in _PROTECTED_PATH_PARTS
            or "internal_test" in part
            or "gse246381" in part
            or part.startswith("sealed")
            or part.startswith("restricted")
        ):
            raise ScalarFittedStateError(
                f"{label} enters a protected path namespace"
            )
    return candidate


def _unique_json_object(pairs: Sequence[tuple[str, object]]) -> dict:
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
        raise ScalarFittedStateError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ScalarFittedStateError(f"{label} must be a JSON object")
    return value


def _normalized_typed_rows(
    rows: Sequence[Mapping],
    *,
    partition: str,
    label: str,
    require_single_estimand: bool = True,
) -> list[dict]:
    try:
        normalized = normalize_predictor_input_rows(rows)
    except PredictorProvenanceError as exc:
        raise ScalarFittedStateError(str(exc)) from exc
    if any(row["partition"] != partition for row in normalized):
        raise ScalarFittedStateError(
            f"{label} must contain only {partition} typed predictor rows"
        )
    estimands = {row["estimand_id"] for row in normalized}
    if require_single_estimand and len(estimands) != 1:
        raise ScalarFittedStateError(
            f"{label} must preserve exactly one training estimand"
        )
    for row in normalized:
        _identifier(row["estimand_id"], label=f"{label}.estimand_id")
        _identifier(row["cell_id"], label=f"{label}.cell_id")
        inputs = row["predictor_input"]
        for field in ("endpoint", "publication_context_id", "region"):
            _feature_text(inputs[field], label=f"{label}.predictor_input.{field}")
    return normalized


def _train_row_binding(rows: Sequence[Mapping]) -> dict:
    row_ids = [row["row_id"] for row in rows]
    ordered_raw = json.dumps(
        row_ids, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    set_raw = json.dumps(
        sorted(row_ids), separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    try:
        content = predictor_input_jsonl_bytes(rows)
    except PredictorProvenanceError as exc:
        raise ScalarFittedStateError(str(exc)) from exc
    return {
        "n_rows": len(rows),
        "ordered_row_ids_sha256": hashlib.sha256(ordered_raw).hexdigest(),
        "row_set_sha256": hashlib.sha256(set_raw).hexdigest(),
        "content_sha256": hashlib.sha256(content).hexdigest(),
    }


def _validate_row_binding(value: Mapping, *, label: str) -> dict:
    _strict_keys(value, _ROW_BINDING_KEYS, label=label)
    return {
        "n_rows": _positive_int(value.get("n_rows"), label=f"{label}.n_rows"),
        "ordered_row_ids_sha256": _digest(
            value.get("ordered_row_ids_sha256"),
            label=f"{label}.ordered_row_ids_sha256",
        ),
        "row_set_sha256": _digest(
            value.get("row_set_sha256"), label=f"{label}.row_set_sha256"
        ),
        "content_sha256": _digest(
            value.get("content_sha256"), label=f"{label}.content_sha256"
        ),
    }


def _normalize_assignment_rows(values: Sequence[Mapping]) -> list[dict]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ScalarFittedStateError("assignments must be a sequence")
    normalized = []
    for index, value in enumerate(values):
        _strict_keys(value, _ASSIGNMENT_ROW_KEYS, label=f"assignment {index}")
        normalized.append({
            "row_id": _digest(value.get("row_id"), label=f"assignment {index}.row_id"),
            "partition": value.get("partition"),
            "cell_id": _identifier(
                value.get("cell_id"), label=f"assignment {index}.cell_id"
            ),
            "source_component_id": _feature_text(
                value.get("source_component_id"),
                label=f"assignment {index}.source_component_id",
            ),
        })
        if normalized[-1]["partition"] != TRAIN_PARTITION:
            raise ScalarFittedStateError(
                "shared component assignments must contain only ORDINARY_TRAIN rows"
            )
    normalized.sort(key=lambda item: item["row_id"])
    row_ids = [item["row_id"] for item in normalized]
    if len(row_ids) != len(set(row_ids)):
        raise ScalarFittedStateError("component assignment row IDs must be unique")
    return normalized


def _build_assignment_rows(
    values: Sequence[Mapping], *, expected_train_rows: Sequence[Mapping]
) -> list[dict]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ScalarFittedStateError("assignments must be a sequence")
    components: dict[str, str] = {}
    for index, value in enumerate(values):
        _strict_keys(
            value, _ASSIGNMENT_INPUT_ROW_KEYS, label=f"assignment input {index}"
        )
        row_id = _digest(
            value.get("row_id"), label=f"assignment input {index}.row_id"
        )
        if row_id in components:
            raise ScalarFittedStateError(
                "component assignment row IDs must be unique"
            )
        components[row_id] = _feature_text(
            value.get("source_component_id"),
            label=f"assignment input {index}.source_component_id",
        )
    expected_ids = [str(row["row_id"]) for row in expected_train_rows]
    if set(components) != set(expected_ids):
        raise ScalarFittedStateError(
            "component assignments must cover exactly the bound train row IDs"
        )
    return _normalize_assignment_rows([
        {
            "row_id": row["row_id"],
            "partition": row["partition"],
            "cell_id": row["cell_id"],
            "source_component_id": components[str(row["row_id"])],
        }
        for row in expected_train_rows
    ])


def build_train_source_component_assignment(
    train_rows: Sequence[Mapping],
    assignments: Sequence[Mapping],
    *,
    train_development_outcomes_sha256: str,
    non_data_authority_hashes: Mapping,
    source_component_split_manifest_source_sha256: str,
    source_component_split_manifest_projection_sha256: str,
) -> dict:
    """Build a train-only assignment artifact from an external safe projection."""

    rows = _normalized_typed_rows(
        train_rows,
        partition=TRAIN_PARTITION,
        label="train_rows",
        require_single_estimand=False,
    )
    authorities = _non_data_authority_hashes(non_data_authority_hashes)
    source_sha = _digest(
        source_component_split_manifest_source_sha256,
        label="source_component_split_manifest_source_sha256",
    )
    if source_sha != authorities["source_component_split_manifest_sha256"]:
        raise ScalarFittedStateError(
            "component-assignment source SHA differs from the frozen non-data authority"
        )
    payload = {
        "schema_version": TRAIN_COMPONENT_ASSIGNMENT_SCHEMA,
        "artifact_role": TRAIN_COMPONENT_ASSIGNMENT_ROLE,
        "status": TRAIN_COMPONENT_ASSIGNMENT_STATUS,
        "assignment_scope": TRAIN_COMPONENT_ASSIGNMENT_SCOPE,
        "train_development_outcomes_sha256": _digest(
            train_development_outcomes_sha256,
            label="train_development_outcomes_sha256",
        ),
        "non_data_authority_hashes": authorities,
        "source_component_split_manifest_source_sha256": source_sha,
        "source_component_split_manifest_projection_sha256": _digest(
            source_component_split_manifest_projection_sha256,
            label="source_component_split_manifest_projection_sha256",
        ),
        "train_row_binding": _train_row_binding(rows),
        "assignments": _build_assignment_rows(
            assignments, expected_train_rows=rows
        ),
        "ordinary_only": True,
        "internal_test_access": False,
        "sealed_access": False,
        "downstream_unlock": False,
    }
    return validate_train_source_component_assignment(payload)


def validate_train_source_component_assignment(payload: Mapping) -> dict:
    if not isinstance(payload, Mapping):
        raise ScalarFittedStateError("component assignment must be an object")
    if payload.get("schema_version") == V1_TRAIN_COMPONENT_ASSIGNMENT_SCHEMA:
        raise ScalarFittedStateError(
            "component-assignment v1 is method-information-set-bound and is rejected"
        )
    if payload.get("schema_version") != TRAIN_COMPONENT_ASSIGNMENT_SCHEMA:
        raise ScalarFittedStateError("unsupported component-assignment schema")
    _strict_keys(payload, _ASSIGNMENT_KEYS, label="component assignment")
    if payload.get("artifact_role") != TRAIN_COMPONENT_ASSIGNMENT_ROLE:
        raise ScalarFittedStateError("component-assignment artifact role is invalid")
    if payload.get("status") != TRAIN_COMPONENT_ASSIGNMENT_STATUS:
        raise ScalarFittedStateError("component-assignment status is invalid")
    if payload.get("assignment_scope") != TRAIN_COMPONENT_ASSIGNMENT_SCOPE:
        raise ScalarFittedStateError(
            "component-assignment scope is not shared and information-set independent"
        )
    outcomes_sha = _digest(
        payload.get("train_development_outcomes_sha256"),
        label="train_development_outcomes_sha256",
    )
    authorities = _non_data_authority_hashes(
        payload.get("non_data_authority_hashes")
    )
    source_sha = _digest(
        payload.get("source_component_split_manifest_source_sha256"),
        label="source_component_split_manifest_source_sha256",
    )
    if source_sha != authorities["source_component_split_manifest_sha256"]:
        raise ScalarFittedStateError(
            "component-assignment source SHA differs from non-data authorities"
        )
    projection_sha = _digest(
        payload.get("source_component_split_manifest_projection_sha256"),
        label="source_component_split_manifest_projection_sha256",
    )
    binding = _validate_row_binding(
        payload.get("train_row_binding"), label="train_row_binding"
    )
    assignments = _normalize_assignment_rows(payload.get("assignments"))
    if payload.get("assignments") != assignments:
        raise ScalarFittedStateError(
            "component assignments are not in canonical row_id order"
        )
    if len(assignments) != binding["n_rows"]:
        raise ScalarFittedStateError(
            "component assignments do not match bound train row count"
        )
    assignment_ids = [item["row_id"] for item in assignments]
    ordered_raw = json.dumps(
        assignment_ids, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    set_raw = json.dumps(
        sorted(assignment_ids), separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if hashlib.sha256(ordered_raw).hexdigest() != binding[
        "ordered_row_ids_sha256"
    ]:
        raise ScalarFittedStateError(
            "component assignments do not match bound train row order"
        )
    if hashlib.sha256(set_raw).hexdigest() != binding["row_set_sha256"]:
        raise ScalarFittedStateError(
            "component assignments do not match bound train row set"
        )
    _boundaries(payload, label="component assignment")
    return {
        **dict(payload),
        "train_development_outcomes_sha256": outcomes_sha,
        "non_data_authority_hashes": authorities,
        "source_component_split_manifest_source_sha256": source_sha,
        "source_component_split_manifest_projection_sha256": projection_sha,
        "train_row_binding": binding,
        "assignments": assignments,
    }


def canonical_train_source_component_assignment_bytes(payload: Mapping) -> bytes:
    try:
        return canonical_json_bytes(validate_train_source_component_assignment(payload))
    except ArtifactIOError as exc:
        raise ScalarFittedStateError(str(exc)) from exc


def _verify_assignment_expected_bindings(
    payload: Mapping,
    *,
    expected_train_rows: Sequence[Mapping],
    expected_train_development_outcomes_sha256: str,
    expected_non_data_authority_hashes: Mapping,
    expected_source_component_split_manifest_source_sha256: str,
    expected_source_component_split_manifest_projection_sha256: str,
) -> list[dict]:
    rows = _normalized_typed_rows(
        expected_train_rows,
        partition=TRAIN_PARTITION,
        label="expected_train_rows",
        require_single_estimand=False,
    )
    expected = {
        "train_development_outcomes_sha256": _digest(
            expected_train_development_outcomes_sha256,
            label="expected_train_development_outcomes_sha256",
        ),
        "non_data_authority_hashes": _non_data_authority_hashes(
            expected_non_data_authority_hashes
        ),
        "source_component_split_manifest_source_sha256": _digest(
            expected_source_component_split_manifest_source_sha256,
            label="expected_source_component_split_manifest_source_sha256",
        ),
        "source_component_split_manifest_projection_sha256": _digest(
            expected_source_component_split_manifest_projection_sha256,
            label="expected_source_component_split_manifest_projection_sha256",
        ),
        "train_row_binding": _train_row_binding(rows),
    }
    for field, value in expected.items():
        if payload[field] != value:
            raise ScalarFittedStateError(
                f"component assignment {field} binding mismatch"
            )
    expected_rows = [
        {
            "row_id": row["row_id"],
            "partition": row["partition"],
            "cell_id": row["cell_id"],
            "source_component_id": item["source_component_id"],
        }
        for row, item in zip(rows, payload["assignments"], strict=True)
    ]
    if payload["assignments"] != expected_rows:
        raise ScalarFittedStateError(
            "component assignment row/cell/partition binding differs from canonical train rows"
        )
    return rows


def write_train_source_component_assignment_exclusive(
    path: Path,
    payload: Mapping,
    *,
    expected_train_rows: Sequence[Mapping],
    expected_train_development_outcomes_sha256: str,
    expected_non_data_authority_hashes: Mapping,
    expected_source_component_split_manifest_source_sha256: str,
    expected_source_component_split_manifest_projection_sha256: str,
) -> str:
    output = _assert_development_path(path, label="component-assignment output")
    validated = validate_train_source_component_assignment(payload)
    _verify_assignment_expected_bindings(
        validated,
        expected_train_rows=expected_train_rows,
        expected_train_development_outcomes_sha256=(
            expected_train_development_outcomes_sha256
        ),
        expected_non_data_authority_hashes=expected_non_data_authority_hashes,
        expected_source_component_split_manifest_source_sha256=(
            expected_source_component_split_manifest_source_sha256
        ),
        expected_source_component_split_manifest_projection_sha256=(
            expected_source_component_split_manifest_projection_sha256
        ),
    )
    try:
        return atomic_write_json_exclusive(
            output,
            validated,
            validator=validate_train_source_component_assignment,
        )
    except ArtifactIOError as exc:
        raise ScalarFittedStateError(str(exc)) from exc


def load_train_source_component_assignment_exact(
    path: Path,
    expected_sha256: str,
    *,
    expected_train_rows: Sequence[Mapping],
    expected_train_development_outcomes_sha256: str,
    expected_non_data_authority_hashes: Mapping,
    expected_source_component_split_manifest_source_sha256: str,
    expected_source_component_split_manifest_projection_sha256: str,
) -> dict:
    input_path = _assert_development_path(path, label="component-assignment input")
    digest = _digest(expected_sha256, label="expected_sha256")
    try:
        raw = read_exact_regular_file(input_path, digest)
    except ArtifactIOError as exc:
        raise ScalarFittedStateError(str(exc)) from exc
    payload = validate_train_source_component_assignment(
        _parse_json_object(raw, label="component assignment")
    )
    if canonical_train_source_component_assignment_bytes(payload) != raw:
        raise ScalarFittedStateError(
            "component-assignment bytes are not canonical"
        )
    _verify_assignment_expected_bindings(
        payload,
        expected_train_rows=expected_train_rows,
        expected_train_development_outcomes_sha256=(
            expected_train_development_outcomes_sha256
        ),
        expected_non_data_authority_hashes=expected_non_data_authority_hashes,
        expected_source_component_split_manifest_source_sha256=(
            expected_source_component_split_manifest_source_sha256
        ),
        expected_source_component_split_manifest_projection_sha256=(
            expected_source_component_split_manifest_projection_sha256
        ),
    )
    return payload


def _condition(row: Mapping) -> tuple[str, str, str]:
    inputs = row["predictor_input"]
    return (
        str(inputs["endpoint"]),
        str(inputs["publication_context_id"]),
        str(inputs["region"]),
    )


def _condition_vocab(rows: Sequence[Mapping]) -> dict[str, list[str]]:
    conditions = [_condition(row) for row in rows]
    return {
        "endpoint": sorted({condition[0] for condition in conditions}),
        "publication_context_id": sorted({
            condition[1] for condition in conditions
        }),
        "region": sorted({condition[2] for condition in conditions}),
    }


def _validate_condition_vocab(value: Mapping) -> dict[str, list[str]]:
    _strict_keys(value, _CONDITION_VOCAB_KEYS, label="condition_vocab")
    normalized = {}
    for field in ("endpoint", "publication_context_id", "region"):
        levels = value.get(field)
        if (
            isinstance(levels, (str, bytes))
            or not isinstance(levels, Sequence)
            or not levels
        ):
            raise ScalarFittedStateError(
                f"condition_vocab.{field} must be a non-empty sequence"
            )
        parsed = [
            _feature_text(level, label=f"condition_vocab.{field}[{index}]")
            for index, level in enumerate(levels)
        ]
        if parsed != sorted(set(parsed)):
            raise ScalarFittedStateError(
                f"condition_vocab.{field} must be sorted and unique"
            )
        normalized[field] = parsed
    return normalized


def _condition_matrix(
    rows: Sequence[Mapping], vocab: Mapping[str, Sequence[str]]
) -> np.ndarray:
    fields = ("endpoint", "publication_context_id", "region")
    conditions = [_condition(row) for row in rows]
    blocks = []
    for condition_index, field in enumerate(fields):
        levels = list(vocab[field])
        level_index = {
            level: position + 1 for position, level in enumerate(levels)
        }
        block = np.zeros((len(rows), len(levels) + 1), dtype=np.float64)
        for row_index, condition in enumerate(conditions):
            block[row_index, level_index.get(condition[condition_index], 0)] = 1.0
        blocks.append(block)
    return np.concatenate(blocks, axis=1)


def _flat_feature_row(row: Mapping) -> dict:
    inputs = row["predictor_input"]
    return {
        "endpoint": inputs["endpoint"],
        "publication_context_id": inputs["publication_context_id"],
        "region": inputs["region"],
        "normalized_source_sequence": inputs["normalized_source_sequence"],
        "normalized_candidate_sequence": inputs[
            "normalized_candidate_sequence"
        ],
        "edit_list": [dict(edit) for edit in inputs["edit_list"]],
        "edit_count": inputs["edit_count"],
    }


def _feature_fn(method_id: str):
    if method_id == "EDIT_ONLY_RIDGE":
        return edit_only_features
    if method_id == "CANDIDATE_KMER_RIDGE":
        return candidate_kmer_features
    if method_id in {
        "PAIRED_LOCAL_GLOBAL_KMER_RIDGE",
        "PAIRED_FEATURE_NEAREST_NEIGHBOR",
    }:
        return paired_kmer_features
    return None


def _base_feature_matrix(rows: Sequence[Mapping], method_id: str) -> np.ndarray:
    feature_fn = _feature_fn(method_id)
    if feature_fn is None:
        return np.empty((len(rows), 0), dtype=np.float64)
    try:
        return matrix([_flat_feature_row(row) for row in rows], feature_fn)
    except (TypeError, ValueError) as exc:
        raise ScalarFittedStateError(
            f"{method_id} feature replay failed: {exc}"
        ) from exc


def _feature_matrix(
    rows: Sequence[Mapping], method_id: str, vocab: Mapping[str, Sequence[str]]
) -> np.ndarray:
    base = _base_feature_matrix(rows, method_id)
    conditions = _condition_matrix(rows, vocab)
    result = np.concatenate([base, conditions], axis=1)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ScalarFittedStateError("scalar feature matrix must be finite and 2-D")
    return result


def _feature_schema(
    method_id: str,
    vocab: Mapping[str, Sequence[str]],
    *,
    observed_base_dimension: int | None = None,
) -> dict:
    condition_dimension = sum(len(vocab[field]) + 1 for field in (
        "endpoint",
        "publication_context_id",
        "region",
    ))
    if method_id == "ZERO_DELTA":
        feature_map_id = "CONSTANT_ZERO_V1"
        base_dimension = 0
        condition_encoding = "BOUND_NOT_USED"
        encoded_condition_dimension = 0
        max_k = None
        radius = None
    elif method_id == "SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR":
        feature_map_id = "CONDITION_CELL_COMPONENT_ROW_BALANCED_LOOKUP_V2"
        base_dimension = 0
        condition_encoding = (
            "EXACT_CONDITION_FORMAL_WEIGHT_LOOKUP_WITH_GLOBAL_FALLBACK_V2"
        )
        encoded_condition_dimension = 0
        max_k = None
        radius = None
    elif method_id == "EDIT_ONLY_RIDGE":
        feature_map_id = "EDIT_ONLY_FEATURES_V1"
        base_dimension = 22
        condition_encoding = (
            "TRAIN_ONLY_ENDPOINT_CONTEXT_REGION_ONE_HOT_WITH_UNK_V1"
        )
        encoded_condition_dimension = condition_dimension
        max_k = None
        radius = None
    elif method_id == "CANDIDATE_KMER_RIDGE":
        feature_map_id = "CANDIDATE_KMER_FREQUENCY_MAX3_V1"
        base_dimension = 86
        condition_encoding = (
            "TRAIN_ONLY_ENDPOINT_CONTEXT_REGION_ONE_HOT_WITH_UNK_V1"
        )
        encoded_condition_dimension = condition_dimension
        max_k = 3
        radius = None
    else:
        feature_map_id = (
            "PAIRED_LOCAL_GLOBAL_KMER_MAX3_RADIUS12_PER_EDIT_MEAN_V2"
        )
        base_dimension = 366
        condition_encoding = (
            "TRAIN_ONLY_ENDPOINT_CONTEXT_REGION_ONE_HOT_WITH_UNK_V1"
        )
        encoded_condition_dimension = condition_dimension
        max_k = 3
        radius = 12
    if (
        observed_base_dimension is not None
        and observed_base_dimension != base_dimension
    ):
        raise ScalarFittedStateError(
            f"{method_id} feature dimension changed: expected {base_dimension}, "
            f"got {observed_base_dimension}"
        )
    return {
        "schema_version": SCALAR_FEATURE_SCHEMA,
        "feature_map_id": feature_map_id,
        "base_feature_dimension": base_dimension,
        "condition_encoding": condition_encoding,
        "condition_feature_dimension": encoded_condition_dimension,
        "total_feature_dimension": (
            base_dimension + encoded_condition_dimension
        ),
        "max_k": max_k,
        "local_radius": radius,
    }


def _validate_feature_schema(
    value: Mapping, method_id: str, vocab: Mapping[str, Sequence[str]]
) -> dict:
    _strict_keys(value, _FEATURE_SCHEMA_KEYS, label="feature_schema")
    expected = _feature_schema(method_id, vocab)
    if dict(value) != expected:
        raise ScalarFittedStateError(
            "feature_schema differs from frozen method feature semantics"
        )
    return expected


def _fit_config(method_id: str) -> dict:
    uses_formal_weights = method_id in COMPONENT_ASSIGNMENT_REQUIRED_METHOD_IDS
    return {
        "schema_version": SCALAR_FIT_CONFIG_SCHEMA,
        "method_id": method_id,
        "row_order_rule": ROW_ORDER_RULE,
        "target_field": "training_target",
        "ridge_alpha_grid": (
            [float(value) for value in DEFAULT_RIDGE_ALPHAS]
            if method_id in RIDGE_METHOD_IDS
            else []
        ),
        "ridge_cv_rule": RIDGE_CV_RULE if method_id in RIDGE_METHOD_IDS else None,
        "ridge_cv_scoring_policy": (
            RIDGE_CV_SCORING_POLICY if method_id in RIDGE_METHOD_IDS else None
        ),
        "training_weight_policy": (
            FORMAL_WEIGHT_POLICY if uses_formal_weights else "NOT_USED_ZERO_DELTA"
        ),
        "weighted_risk_normalization": (
            "SUM_TO_ONE" if uses_formal_weights else "NOT_APPLICABLE"
        ),
        "ridge_training_policy": (
            RIDGE_TRAINING_POLICY if method_id in RIDGE_METHOD_IDS else None
        ),
        "prior_training_policy": (
            PRIOR_TRAINING_POLICY
            if method_id == "SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR"
            else None
        ),
        "nearest_neighbor_train_chunk_size": (
            DEFAULT_NN_TRAIN_CHUNK_SIZE
            if method_id == "PAIRED_FEATURE_NEAREST_NEIGHBOR"
            else None
        ),
        "nearest_neighbor_scaling_policy": (
            NN_SCALING_POLICY
            if method_id == "PAIRED_FEATURE_NEAREST_NEIGHBOR"
            else None
        ),
        "nearest_neighbor_tie_rule": (
            NN_TIE_RULE
            if method_id == "PAIRED_FEATURE_NEAREST_NEIGHBOR"
            else None
        ),
    }


def _validate_fit_config(value: Mapping, method_id: str) -> dict:
    _strict_keys(value, _FIT_CONFIG_KEYS, label="fit_config")
    expected = _fit_config(method_id)
    if dict(value) != expected:
        raise ScalarFittedStateError(
            "fit_config differs from the frozen scalar method configuration"
        )
    return expected


def _component_map(assignment: Mapping) -> dict[str, str]:
    return {
        item["row_id"]: item["source_component_id"]
        for item in assignment["assignments"]
    }


def _formal_cell_component_row_weights(
    cell_ids: Sequence[str], source_component_ids: Sequence[str]
) -> np.ndarray:
    """Return the frozen sum-to-one cell -> global-component -> row risk.

    ``source_component_ids`` are global source-component identities.  The
    hierarchy first gives every typed estimand cell equal mass, then gives each
    global component represented in that cell equal mass, and finally divides
    a cell/component mass equally over its rows.  Row replication inside one
    cell/component therefore cannot change that cell/component's total risk.
    """

    cells = [str(value) for value in cell_ids]
    components = [str(value) for value in source_component_ids]
    if not cells or len(cells) != len(components):
        raise ScalarFittedStateError(
            "formal cell/component weight inputs must be non-empty and aligned"
        )
    if any(not value for value in cells) or any(not value for value in components):
        raise ScalarFittedStateError(
            "formal cell/component IDs must be non-empty"
        )
    hierarchy: dict[str, dict[str, list[int]]] = {}
    for index, (cell, component) in enumerate(
        zip(cells, components, strict=True)
    ):
        hierarchy.setdefault(cell, {}).setdefault(component, []).append(index)
    weights = np.zeros(len(cells), dtype=np.float64)
    n_cells = len(hierarchy)
    for cell in sorted(hierarchy):
        component_rows = hierarchy[cell]
        n_components = len(component_rows)
        for component in sorted(component_rows):
            indices = component_rows[component]
            value = 1.0 / (n_cells * n_components * len(indices))
            weights[np.asarray(indices, dtype=np.int64)] = value
    total = math.fsum(float(value) for value in weights)
    if not math.isfinite(total) or total <= 0.0:
        raise ScalarFittedStateError("formal training weights are invalid")
    weights /= total
    if (
        not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
        or not math.isclose(
            math.fsum(float(value) for value in weights),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ScalarFittedStateError(
            "formal training weights must be finite, positive, and sum to one"
        )
    return weights


def _formal_cell_component_row_mean(
    values: Sequence[float],
    cell_ids: Sequence[str],
    source_component_ids: Sequence[str],
) -> float:
    """Compute the formal hierarchy with canonical nested ``math.fsum``."""

    numeric = [float(value) for value in values]
    cells = [str(value) for value in cell_ids]
    components = [str(value) for value in source_component_ids]
    if (
        not numeric
        or len(numeric) != len(cells)
        or len(numeric) != len(components)
        or any(not math.isfinite(value) for value in numeric)
    ):
        raise ScalarFittedStateError(
            "formal weighted mean inputs must be finite, non-empty, and aligned"
        )
    hierarchy: dict[str, dict[str, list[float]]] = {}
    for value, cell, component in zip(
        numeric, cells, components, strict=True
    ):
        hierarchy.setdefault(cell, {}).setdefault(component, []).append(value)
    cell_values = []
    for cell in sorted(hierarchy):
        component_values = []
        for component in sorted(hierarchy[cell]):
            rows = hierarchy[cell][component]
            component_values.append(math.fsum(rows) / len(rows))
        cell_values.append(math.fsum(component_values) / len(component_values))
    result = math.fsum(cell_values) / len(cell_values)
    if not math.isfinite(result):
        raise ScalarFittedStateError("formal weighted mean is non-finite")
    return result


def _formal_cell_component_row_weighted_mse(
    y: Sequence[float],
    prediction: Sequence[float],
    cell_ids: Sequence[str],
    source_component_ids: Sequence[str],
) -> float:
    observed = np.asarray(y, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    if observed.shape != predicted.shape or observed.ndim != 1:
        raise ScalarFittedStateError("formal MSE inputs must be aligned vectors")
    squared = (predicted - observed) ** 2
    if not np.all(np.isfinite(squared)):
        raise ScalarFittedStateError("formal MSE inputs must be finite")
    return _formal_cell_component_row_mean(
        squared.tolist(), cell_ids, source_component_ids
    )


def _training_design_binding(
    rows: Sequence[Mapping],
    method_id: str,
    component_by_row_id: Mapping[str, str],
) -> dict:
    cells = [str(row["cell_id"]) for row in rows]
    uses_components = method_id in COMPONENT_ASSIGNMENT_REQUIRED_METHOD_IDS
    components = (
        [component_by_row_id[row["row_id"]] for row in rows]
        if uses_components
        else []
    )
    design = [
        {
            "row_id": row["row_id"],
            "cell_id": row["cell_id"],
            "source_component_id": (
                component_by_row_id[row["row_id"]] if uses_components else None
            ),
        }
        for row in rows
    ]
    weights = (
        _formal_cell_component_row_weights(cells, components).tolist()
        if uses_components
        else []
    )
    return {
        "schema_version": SCALAR_TRAINING_DESIGN_SCHEMA,
        "design_sha256": _canonical_sha256(design),
        "formal_weights_sha256": _canonical_sha256(weights),
        "training_weight_policy": (
            FORMAL_WEIGHT_POLICY if uses_components else "NOT_USED_ZERO_DELTA"
        ),
        "ordered_cell_ids_sha256": _canonical_sha256(cells),
        "ordered_source_component_ids_sha256": _canonical_sha256(components),
        "cell_ids": sorted(set(cells)),
        "source_component_ids": sorted(set(components)),
    }


def _weighted_feature_mean(
    matrix_value: np.ndarray, sample_weight: np.ndarray
) -> np.ndarray:
    x = np.asarray(matrix_value, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    return np.asarray(
        [
            math.fsum(
                float(weights[row] * x[row, column])
                for row in range(len(x))
            )
            for column in range(x.shape[1])
        ],
        dtype=np.float64,
    )


def _fit_standardized_ridge(
    features: np.ndarray,
    target: np.ndarray,
    alpha: float,
    sample_weight: np.ndarray,
) -> dict:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ScalarFittedStateError("ridge features must be a non-empty matrix")
    if y.ndim != 1 or len(y) != len(x):
        raise ScalarFittedStateError("ridge target must align to feature rows")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ScalarFittedStateError("ridge features and target must be finite")
    weights = np.asarray(sample_weight, dtype=np.float64)
    if (
        weights.shape != (len(x),)
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
        or not math.isclose(
            math.fsum(float(value) for value in weights),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ScalarFittedStateError(
            "ridge formal sample weights must be positive and sum to one"
        )
    alpha = _finite_float(alpha, label="ridge alpha")
    if alpha <= 0.0:
        raise ScalarFittedStateError("ridge alpha must be positive")

    mean = _weighted_feature_mean(x, weights)
    variance = _weighted_feature_mean((x - mean) ** 2, weights)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale < 1e-12] = 1.0
    standardized = (x - mean) / scale
    y_mean = math.fsum(
        float(weight * value) for weight, value in zip(weights, y, strict=True)
    )
    centered = y - y_mean
    square_root_weight = np.sqrt(weights)
    weighted_features = standardized * square_root_weight[:, None]
    weighted_target = centered * square_root_weight
    n_rows, n_features = standardized.shape
    if n_features <= n_rows:
        solver = "PRIMAL_SMALLER_FEATURE_SYSTEM"
        dimension = n_features
        system = (
            weighted_features.T @ weighted_features
            + alpha * np.eye(n_features, dtype=np.float64)
        )
        rhs = weighted_features.T @ weighted_target
        try:
            coef = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(system) @ rhs
    else:
        solver = "DUAL_SMALLER_SAMPLE_SYSTEM"
        dimension = n_rows
        kernel = weighted_features @ weighted_features.T
        system = kernel + alpha * np.eye(n_rows, dtype=np.float64)
        try:
            dual = np.linalg.solve(system, weighted_target)
        except np.linalg.LinAlgError:
            dual = np.linalg.pinv(system) @ weighted_target
        coef = weighted_features.T @ dual
    return {
        "mean": mean,
        "scale": scale,
        "y_mean": y_mean,
        "coef": np.asarray(coef, dtype=np.float64),
        "alpha": alpha,
        "solver": solver,
        "square_system_dimension": int(dimension),
        "n_train_rows": int(n_rows),
        "n_features": int(n_features),
    }


def _ridge_predict(model: Mapping, features: np.ndarray) -> np.ndarray:
    standardized = (
        np.asarray(features, dtype=np.float64) - model["mean"]
    ) / model["scale"]
    return standardized @ model["coef"] + model["y_mean"]


def _fit_grouped_ridge_state(
    rows: Sequence[Mapping],
    method_id: str,
    vocab: Mapping[str, Sequence[str]],
    component_by_row_id: Mapping[str, str],
) -> tuple[dict, dict]:
    base = _base_feature_matrix(rows, method_id)
    feature_schema = _feature_schema(
        method_id, vocab, observed_base_dimension=base.shape[1]
    )
    features = np.concatenate([base, _condition_matrix(rows, vocab)], axis=1)
    target = np.asarray(
        [float(row["training_target"]) for row in rows], dtype=np.float64
    )
    groups = np.asarray(
        [component_by_row_id[row["row_id"]] for row in rows], dtype=object
    )
    cells = np.asarray([row["cell_id"] for row in rows], dtype=object)
    full_weights = _formal_cell_component_row_weights(cells, groups)
    levels = sorted(set(groups.tolist()))
    if len(levels) < 3:
        selected = float(DEFAULT_RIDGE_ALPHAS[len(DEFAULT_RIDGE_ALPHAS) // 2])
        cv_status = "FIXED_MIDDLE_ALPHA_TOO_FEW_COMPONENTS"
        scores: dict[float, float] = {}
        n_folds = 0
        fold_assignment: list[dict] = []
    else:
        n_folds = min(5, len(levels))
        fold_of = {group: index % n_folds for index, group in enumerate(levels)}
        fold_assignment = [
            {"source_component_id": group, "fold": fold_of[group]}
            for group in levels
        ]
        scores = {}
        for alpha in DEFAULT_RIDGE_ALPHAS:
            oof = np.full(len(rows), np.nan, dtype=np.float64)
            for fold in range(n_folds):
                valid = np.asarray(
                    [fold_of[group] == fold for group in groups], dtype=bool
                )
                if not np.any(valid) or not np.any(~valid):
                    raise ScalarFittedStateError(
                        "grouped ridge CV produced an empty train or validation fold"
                    )
                fold_weights = _formal_cell_component_row_weights(
                    cells[~valid], groups[~valid]
                )
                model = _fit_standardized_ridge(
                    features[~valid],
                    target[~valid],
                    float(alpha),
                    fold_weights,
                )
                oof[valid] = _ridge_predict(model, features[valid])
            if not np.all(np.isfinite(oof)):
                raise ScalarFittedStateError(
                    "grouped ridge CV did not produce exactly one OOF prediction per row"
                )
            scores[float(alpha)] = _formal_cell_component_row_weighted_mse(
                target, oof, cells, groups
            )
        selected = min(scores, key=lambda alpha: (scores[alpha], alpha))
        cv_status = "TRAIN_ONLY_GLOBAL_SOURCE_COMPONENT_GROUPED_FORMAL_OOF_CV"
    model = _fit_standardized_ridge(
        features, target, selected, full_weights
    )
    state = {
        "kind": "STANDARDIZED_RIDGE",
        "scaler_mean": model["mean"].tolist(),
        "scaler_scale": model["scale"].tolist(),
        "y_mean": model["y_mean"],
        "coef": model["coef"].tolist(),
        "alpha": model["alpha"],
        "solver": model["solver"],
        "square_system_dimension": model["square_system_dimension"],
        "n_train_rows": model["n_train_rows"],
        "n_features": model["n_features"],
        "cv_status": cv_status,
        "cv_n_global_source_components": len(levels),
        "cv_n_folds": n_folds,
        "cv_fold_assignment_sha256": _canonical_sha256(fold_assignment),
        "cv_formal_oof_weighted_mse": [
            {
                "alpha": float(alpha),
                "formal_oof_weighted_mse": scores[alpha],
            }
            for alpha in sorted(scores)
        ],
    }
    return state, feature_schema


def _cell_component_values(
    values: Sequence[float],
    cell_ids: Sequence[str],
    source_component_ids: Sequence[str],
) -> tuple[list[dict], float]:
    hierarchy: dict[str, dict[str, list[float]]] = {}
    for value, cell, component in zip(
        values, cell_ids, source_component_ids, strict=True
    ):
        hierarchy.setdefault(str(cell), {}).setdefault(
            str(component), []
        ).append(float(value))
    summaries = []
    for cell in sorted(hierarchy):
        for component in sorted(hierarchy[cell]):
            rows = hierarchy[cell][component]
            summaries.append({
                "cell_id": cell,
                "source_component_id": component,
                "n_rows": len(rows),
                "row_mean": math.fsum(rows) / len(rows),
            })
    return summaries, _formal_cell_component_row_mean(
        values, cell_ids, source_component_ids
    )


def _fit_prior_state(
    rows: Sequence[Mapping], component_by_row_id: Mapping[str, str]
) -> dict:
    by_condition: dict[tuple[str, str, str], list[tuple[float, str, str]]] = {}
    global_values: list[float] = []
    global_cells: list[str] = []
    global_components: list[str] = []
    for row in rows:
        condition = _condition(row)
        component = component_by_row_id[row["row_id"]]
        value = float(row["training_target"])
        cell = str(row["cell_id"])
        global_values.append(value)
        global_cells.append(cell)
        global_components.append(component)
        by_condition.setdefault(condition, []).append((value, cell, component))
    fallback_values, fallback = _cell_component_values(
        global_values, global_cells, global_components
    )
    condition_values = []
    for condition in sorted(by_condition):
        values = [item[0] for item in by_condition[condition]]
        cells = [item[1] for item in by_condition[condition]]
        components = [item[2] for item in by_condition[condition]]
        summaries, formal_value = _cell_component_values(
            values, cells, components
        )
        condition_values.append({
            "endpoint": condition[0],
            "publication_context_id": condition[1],
            "region": condition[2],
            "cell_ids": sorted(set(cells)),
            "source_component_ids": sorted(set(components)),
            "cell_component_values": summaries,
            "formal_value": formal_value,
        })
    return {
        "kind": "FORMAL_CELL_COMPONENT_ROW_BALANCED_TRAIN_PRIOR",
        "condition_formal_values": condition_values,
        "fallback_cell_component_values": fallback_values,
        "fallback_value": fallback,
    }


def _fit_nn_state(
    rows: Sequence[Mapping],
    vocab: Mapping[str, Sequence[str]],
    component_by_row_id: Mapping[str, str],
) -> tuple[dict, dict]:
    base = _base_feature_matrix(rows, "PAIRED_FEATURE_NEAREST_NEIGHBOR")
    feature_schema = _feature_schema(
        "PAIRED_FEATURE_NEAREST_NEIGHBOR",
        vocab,
        observed_base_dimension=base.shape[1],
    )
    features = np.concatenate([base, _condition_matrix(rows, vocab)], axis=1)
    cells = [str(row["cell_id"]) for row in rows]
    components = [component_by_row_id[row["row_id"]] for row in rows]
    weights = _formal_cell_component_row_weights(cells, components)
    mean = _weighted_feature_mean(features, weights)
    variance = _weighted_feature_mean((features - mean) ** 2, weights)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale < 1e-12] = 1.0
    standardized = (features - mean) / scale
    if not np.all(np.isfinite(standardized)):
        raise ScalarFittedStateError("nearest-neighbor state is non-finite")
    return {
        "kind": "STANDARDIZED_EXACT_NEAREST_NEIGHBOR",
        "scaler_mean": mean.tolist(),
        "scaler_scale": scale.tolist(),
        "train_row_ids": [row["row_id"] for row in rows],
        "train_cell_ids": cells,
        "train_source_component_ids": components,
        "train_standardized_features": standardized.tolist(),
        "train_targets": [float(row["training_target"]) for row in rows],
        "tie_atol": NN_TIE_ATOL,
        "tie_rule": NN_TIE_RULE,
    }, feature_schema


def _load_component_assignment_for_fit(
    method_id: str,
    rows: Sequence[Mapping],
    *,
    train_development_outcomes_sha256: str,
    non_data_authority_hashes: Mapping,
    component_assignment_authority_rows: Sequence[Mapping] | None,
    component_assignment_path: Path | None,
    component_assignment_sha256: str | None,
    source_component_split_manifest_source_sha256: str | None,
    source_component_split_manifest_projection_sha256: str | None,
) -> tuple[dict | None, dict[str, str]]:
    required_values = (
        component_assignment_path,
        component_assignment_sha256,
        source_component_split_manifest_source_sha256,
        source_component_split_manifest_projection_sha256,
    )
    if method_id not in COMPONENT_ASSIGNMENT_REQUIRED_METHOD_IDS:
        if component_assignment_authority_rows is not None or any(
            value is not None for value in required_values
        ):
            raise ScalarFittedStateError(
                "ZERO_DELTA cannot accept an unused component-assignment authority"
            )
        return None, {}
    if any(value is None for value in required_values):
        raise ScalarFittedStateError(
            f"{method_id} HOLD: exact train component-assignment authority is required"
        )
    authority_rows = (
        rows
        if component_assignment_authority_rows is None
        else _normalized_typed_rows(
            component_assignment_authority_rows,
            partition=TRAIN_PARTITION,
            label="component_assignment_authority_rows",
            require_single_estimand=False,
        )
    )
    requested_ids = {str(row["row_id"]) for row in rows}
    authority_ids = {str(row["row_id"]) for row in authority_rows}
    if not requested_ids.issubset(authority_ids):
        raise ScalarFittedStateError(
            "fitted estimand rows are not a subset of the shared component authority"
        )
    assignment = load_train_source_component_assignment_exact(
        Path(component_assignment_path),
        str(component_assignment_sha256),
        expected_train_rows=authority_rows,
        expected_train_development_outcomes_sha256=(
            train_development_outcomes_sha256
        ),
        expected_non_data_authority_hashes=non_data_authority_hashes,
        expected_source_component_split_manifest_source_sha256=str(
            source_component_split_manifest_source_sha256
        ),
        expected_source_component_split_manifest_projection_sha256=str(
            source_component_split_manifest_projection_sha256
        ),
    )
    binding = {
        "schema_version": TRAIN_COMPONENT_ASSIGNMENT_SCHEMA,
        "sha256": str(component_assignment_sha256),
        "source_component_split_manifest_source_sha256": assignment[
            "source_component_split_manifest_source_sha256"
        ],
        "source_component_split_manifest_projection_sha256": assignment[
            "source_component_split_manifest_projection_sha256"
        ],
    }
    full_component_map = _component_map(assignment)
    return binding, {
        row_id: full_component_map[row_id] for row_id in requested_ids
    }


def fit_scalar_fitted_state(
    method_id: str,
    train_rows: Sequence[Mapping],
    *,
    information_set_fingerprint: str,
    train_development_outcomes_sha256: str,
    non_data_authority_hashes: Mapping,
    component_assignment_authority_rows: Sequence[Mapping] | None = None,
    component_assignment_path: Path | None = None,
    component_assignment_sha256: str | None = None,
    source_component_split_manifest_source_sha256: str | None = None,
    source_component_split_manifest_projection_sha256: str | None = None,
) -> dict:
    """Fit one frozen scalar method from canonical train-only typed rows."""

    method = _method_id(method_id)
    rows = _normalized_typed_rows(
        train_rows, partition=TRAIN_PARTITION, label="train_rows"
    )
    fingerprint = _method_information_set_fingerprint(
        method,
        information_set_fingerprint,
        label="information_set_fingerprint",
    )
    outcomes_sha = _digest(
        train_development_outcomes_sha256,
        label="train_development_outcomes_sha256",
    )
    authorities = _non_data_authority_hashes(non_data_authority_hashes)
    component_binding, component_by_row_id = _load_component_assignment_for_fit(
        method,
        rows,
        train_development_outcomes_sha256=outcomes_sha,
        non_data_authority_hashes=authorities,
        component_assignment_authority_rows=component_assignment_authority_rows,
        component_assignment_path=component_assignment_path,
        component_assignment_sha256=component_assignment_sha256,
        source_component_split_manifest_source_sha256=(
            source_component_split_manifest_source_sha256
        ),
        source_component_split_manifest_projection_sha256=(
            source_component_split_manifest_projection_sha256
        ),
    )
    vocab = _condition_vocab(rows)
    if method == "ZERO_DELTA":
        state = {"kind": "ZERO_DELTA", "constant": 0.0}
        feature_schema = _feature_schema(method, vocab)
    elif method == "SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR":
        state = _fit_prior_state(rows, component_by_row_id)
        feature_schema = _feature_schema(method, vocab)
    elif method in RIDGE_METHOD_IDS:
        state, feature_schema = _fit_grouped_ridge_state(
            rows, method, vocab, component_by_row_id
        )
    else:
        state, feature_schema = _fit_nn_state(
            rows, vocab, component_by_row_id
        )
    config = _fit_config(method)
    design_binding = _training_design_binding(
        rows, method, component_by_row_id
    )
    payload = {
        "schema_version": SCALAR_FITTED_STATE_SCHEMA,
        "artifact_role": SCALAR_FITTED_STATE_ROLE,
        "status": SCALAR_FITTED_STATE_STATUS,
        "method_id": method,
        "information_set": METHOD_INFORMATION_SETS[method],
        "information_set_fingerprint": fingerprint,
        "estimand_id": rows[0]["estimand_id"],
        "train_development_outcomes_sha256": outcomes_sha,
        "non_data_authority_hashes": authorities,
        "train_row_binding": _train_row_binding(rows),
        "source_component_assignment_binding": component_binding,
        "training_design_binding": design_binding,
        "fit_config": config,
        "fit_config_sha256": _canonical_sha256(config),
        "feature_schema": feature_schema,
        "condition_vocab": vocab,
        "state": state,
        "ordinary_only": True,
        "internal_test_access": False,
        "sealed_access": False,
        "downstream_unlock": False,
    }
    return validate_scalar_fitted_state(payload)


def _float_vector(
    value: Sequence[object], *, label: str, expected_length: int
) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ScalarFittedStateError(f"{label} must be a numeric sequence")
    result = [
        _finite_float(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    ]
    if len(result) != expected_length:
        raise ScalarFittedStateError(
            f"{label} length differs from frozen feature dimension"
        )
    return result


def _sorted_unique_ids(value: object, *, label: str) -> list[str]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
    ):
        raise ScalarFittedStateError(f"{label} must be a sequence")
    result = [
        _feature_text(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    ]
    if result != sorted(set(result)):
        raise ScalarFittedStateError(f"{label} must be sorted unique")
    return result


def _cell_component_summaries(
    value: Sequence[Mapping], *, label: str
) -> list[dict]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or not value
    ):
        raise ScalarFittedStateError(f"{label} must be a non-empty sequence")
    result = []
    for index, item in enumerate(value):
        _strict_keys(
            item, _CELL_COMPONENT_VALUE_KEYS, label=f"{label}[{index}]"
        )
        result.append({
            "cell_id": _identifier(
                item.get("cell_id"), label=f"{label}[{index}].cell_id"
            ),
            "source_component_id": _feature_text(
                item.get("source_component_id"),
                label=f"{label}[{index}].source_component_id",
            ),
            "n_rows": _positive_int(
                item.get("n_rows"), label=f"{label}[{index}].n_rows"
            ),
            "row_mean": _finite_float(
                item.get("row_mean"), label=f"{label}[{index}].row_mean"
            ),
        })
    keys = [
        (item["cell_id"], item["source_component_id"]) for item in result
    ]
    if keys != sorted(set(keys)):
        raise ScalarFittedStateError(
            f"{label} must be sorted by unique cell/component identity"
        )
    return result


def _summary_formal_value(summaries: Sequence[Mapping]) -> float:
    by_cell: dict[str, list[float]] = {}
    for item in summaries:
        by_cell.setdefault(str(item["cell_id"]), []).append(
            float(item["row_mean"])
        )
    cell_values = [
        math.fsum(by_cell[cell]) / len(by_cell[cell]) for cell in sorted(by_cell)
    ]
    return math.fsum(cell_values) / len(cell_values)


def _validate_prior_state(value: Mapping) -> dict:
    _strict_keys(value, _PRIOR_STATE_KEYS, label="prior state")
    if value.get("kind") != "FORMAL_CELL_COMPONENT_ROW_BALANCED_TRAIN_PRIOR":
        raise ScalarFittedStateError("prior state kind is invalid")
    conditions = value.get("condition_formal_values")
    if (
        isinstance(conditions, (str, bytes))
        or not isinstance(conditions, Sequence)
        or not conditions
    ):
        raise ScalarFittedStateError(
            "prior condition mapping must be a non-empty sequence"
        )
    normalized_conditions = []
    keys = []
    for index, item in enumerate(conditions):
        _strict_keys(item, _PRIOR_CONDITION_KEYS, label=f"prior condition {index}")
        key = (
            _feature_text(
                item.get("endpoint"), label=f"prior condition {index}.endpoint"
            ),
            _feature_text(
                item.get("publication_context_id"),
                label=f"prior condition {index}.publication_context_id",
            ),
            _feature_text(
                item.get("region"), label=f"prior condition {index}.region"
            ),
        )
        parsed_cells = _sorted_unique_ids(
            item.get("cell_ids"), label=f"prior condition {index}.cell_ids"
        )
        parsed_components = _sorted_unique_ids(
            item.get("source_component_ids"),
            label=f"prior condition {index}.source_component_ids",
        )
        if not parsed_cells or not parsed_components:
            raise ScalarFittedStateError(
                "prior condition cell/component IDs must be non-empty"
            )
        summaries = _cell_component_summaries(
            item.get("cell_component_values"),
            label=f"prior condition {index}.cell_component_values",
        )
        if parsed_cells != sorted({entry["cell_id"] for entry in summaries}):
            raise ScalarFittedStateError(
                "prior condition cell IDs differ from its summaries"
            )
        if parsed_components != sorted({
            entry["source_component_id"] for entry in summaries
        }):
            raise ScalarFittedStateError(
                "prior condition component IDs differ from its summaries"
            )
        formal = _finite_float(
            item.get("formal_value"),
            label=f"prior condition {index}.formal_value",
        )
        if formal != _summary_formal_value(summaries):
            raise ScalarFittedStateError(
                "prior condition formal value is not recomputable"
            )
        keys.append(key)
        normalized_conditions.append({
            "endpoint": key[0],
            "publication_context_id": key[1],
            "region": key[2],
            "cell_ids": parsed_cells,
            "source_component_ids": parsed_components,
            "cell_component_values": summaries,
            "formal_value": formal,
        })
    if keys != sorted(set(keys)):
        raise ScalarFittedStateError("prior condition mapping must be sorted unique")
    fallback_values = _cell_component_summaries(
        value.get("fallback_cell_component_values"),
        label="prior fallback cell/component values",
    )
    fallback = _finite_float(value.get("fallback_value"), label="prior fallback value")
    if fallback != _summary_formal_value(fallback_values):
        raise ScalarFittedStateError(
            "prior fallback is not formally cell/component/row balanced"
        )
    return {
        "kind": "FORMAL_CELL_COMPONENT_ROW_BALANCED_TRAIN_PRIOR",
        "condition_formal_values": normalized_conditions,
        "fallback_cell_component_values": fallback_values,
        "fallback_value": fallback,
    }


def _validate_ridge_state(
    value: Mapping, *, feature_dimension: int, n_train_rows: int
) -> dict:
    _strict_keys(value, _RIDGE_STATE_KEYS, label="ridge state")
    if value.get("kind") != "STANDARDIZED_RIDGE":
        raise ScalarFittedStateError("ridge state kind is invalid")
    mean = _float_vector(
        value.get("scaler_mean"),
        label="ridge scaler_mean",
        expected_length=feature_dimension,
    )
    scale = _float_vector(
        value.get("scaler_scale"),
        label="ridge scaler_scale",
        expected_length=feature_dimension,
    )
    if any(item <= 0.0 for item in scale):
        raise ScalarFittedStateError("ridge scaler_scale must be positive")
    coef = _float_vector(
        value.get("coef"), label="ridge coef", expected_length=feature_dimension
    )
    y_mean = _finite_float(value.get("y_mean"), label="ridge y_mean")
    alpha = _finite_float(value.get("alpha"), label="ridge alpha")
    if alpha not in DEFAULT_RIDGE_ALPHAS:
        raise ScalarFittedStateError("ridge alpha is outside the frozen grid")
    if value.get("n_train_rows") != n_train_rows:
        raise ScalarFittedStateError("ridge n_train_rows binding mismatch")
    if value.get("n_features") != feature_dimension:
        raise ScalarFittedStateError("ridge n_features binding mismatch")
    expected_solver = (
        "PRIMAL_SMALLER_FEATURE_SYSTEM"
        if feature_dimension <= n_train_rows
        else "DUAL_SMALLER_SAMPLE_SYSTEM"
    )
    if value.get("solver") != expected_solver:
        raise ScalarFittedStateError("ridge solver does not use the smaller system")
    expected_dimension = min(feature_dimension, n_train_rows)
    if value.get("square_system_dimension") != expected_dimension:
        raise ScalarFittedStateError("ridge square-system dimension is invalid")
    n_components = _positive_int(
        value.get("cv_n_global_source_components"),
        label="ridge cv_n_global_source_components",
    )
    n_folds_raw = value.get("cv_n_folds")
    if isinstance(n_folds_raw, bool) or not isinstance(n_folds_raw, int):
        raise ScalarFittedStateError("ridge cv_n_folds must be an integer")
    if n_folds_raw < 0 or n_folds_raw > 5:
        raise ScalarFittedStateError("ridge cv_n_folds is outside frozen bounds")
    fold_sha = _digest(
        value.get("cv_fold_assignment_sha256"),
        label="ridge cv_fold_assignment_sha256",
    )
    scores = value.get("cv_formal_oof_weighted_mse")
    if isinstance(scores, (str, bytes)) or not isinstance(scores, Sequence):
        raise ScalarFittedStateError("ridge CV scores must be a sequence")
    normalized_scores = []
    for index, item in enumerate(scores):
        _strict_keys(item, _CV_SCORE_KEYS, label=f"ridge CV score {index}")
        score_alpha = _finite_float(item.get("alpha"), label="ridge CV alpha")
        score = _finite_float(
            item.get("formal_oof_weighted_mse"), label="ridge CV score"
        )
        if score_alpha not in DEFAULT_RIDGE_ALPHAS or score < 0.0:
            raise ScalarFittedStateError("ridge CV score is outside frozen bounds")
        normalized_scores.append({
            "alpha": score_alpha,
            "formal_oof_weighted_mse": score,
        })
    if [item["alpha"] for item in normalized_scores] != sorted({
        item["alpha"] for item in normalized_scores
    }):
        raise ScalarFittedStateError("ridge CV alpha records must be sorted unique")
    cv_status = value.get("cv_status")
    if cv_status == "FIXED_MIDDLE_ALPHA_TOO_FEW_COMPONENTS":
        if (
            normalized_scores
            or alpha != DEFAULT_RIDGE_ALPHAS[2]
            or n_components >= 3
            or n_folds_raw != 0
            or fold_sha != _canonical_sha256([])
        ):
            raise ScalarFittedStateError("fixed-alpha ridge CV state is inconsistent")
    elif cv_status == "TRAIN_ONLY_GLOBAL_SOURCE_COMPONENT_GROUPED_FORMAL_OOF_CV":
        if n_components < 3 or n_folds_raw != min(5, n_components):
            raise ScalarFittedStateError(
                "grouped ridge CV component/fold counts are inconsistent"
            )
        if len(normalized_scores) != len(DEFAULT_RIDGE_ALPHAS):
            raise ScalarFittedStateError("grouped ridge CV scores are incomplete")
        selected = min(
            normalized_scores,
            key=lambda item: (item["formal_oof_weighted_mse"], item["alpha"]),
        )["alpha"]
        if alpha != selected:
            raise ScalarFittedStateError("ridge alpha does not match frozen CV rule")
    else:
        raise ScalarFittedStateError("ridge CV status is invalid")
    return {
        "kind": "STANDARDIZED_RIDGE",
        "scaler_mean": mean,
        "scaler_scale": scale,
        "y_mean": y_mean,
        "coef": coef,
        "alpha": alpha,
        "solver": expected_solver,
        "square_system_dimension": expected_dimension,
        "n_train_rows": n_train_rows,
        "n_features": feature_dimension,
        "cv_status": cv_status,
        "cv_n_global_source_components": n_components,
        "cv_n_folds": n_folds_raw,
        "cv_fold_assignment_sha256": fold_sha,
        "cv_formal_oof_weighted_mse": normalized_scores,
    }


def _validate_nn_state(
    value: Mapping, *, feature_dimension: int, train_binding: Mapping
) -> dict:
    _strict_keys(value, _NN_STATE_KEYS, label="nearest-neighbor state")
    if value.get("kind") != "STANDARDIZED_EXACT_NEAREST_NEIGHBOR":
        raise ScalarFittedStateError("nearest-neighbor state kind is invalid")
    mean = _float_vector(
        value.get("scaler_mean"),
        label="nearest-neighbor scaler_mean",
        expected_length=feature_dimension,
    )
    scale = _float_vector(
        value.get("scaler_scale"),
        label="nearest-neighbor scaler_scale",
        expected_length=feature_dimension,
    )
    if any(item <= 0.0 for item in scale):
        raise ScalarFittedStateError("nearest-neighbor scaler_scale must be positive")
    row_ids = value.get("train_row_ids")
    if isinstance(row_ids, (str, bytes)) or not isinstance(row_ids, Sequence):
        raise ScalarFittedStateError("nearest-neighbor train_row_ids must be a sequence")
    parsed_ids = [
        _digest(row_id, label=f"nearest-neighbor train_row_ids[{index}]")
        for index, row_id in enumerate(row_ids)
    ]
    if parsed_ids != sorted(set(parsed_ids)):
        raise ScalarFittedStateError(
            "nearest-neighbor train_row_ids must be canonical sorted unique"
        )
    if len(parsed_ids) != train_binding["n_rows"]:
        raise ScalarFittedStateError("nearest-neighbor row count binding mismatch")
    ordered_sha = _canonical_sha256(parsed_ids)
    if ordered_sha != train_binding["ordered_row_ids_sha256"]:
        raise ScalarFittedStateError("nearest-neighbor row-order binding mismatch")
    raw_cells = value.get("train_cell_ids")
    raw_components = value.get("train_source_component_ids")
    if (
        isinstance(raw_cells, (str, bytes))
        or not isinstance(raw_cells, Sequence)
        or isinstance(raw_components, (str, bytes))
        or not isinstance(raw_components, Sequence)
        or len(raw_cells) != len(parsed_ids)
        or len(raw_components) != len(parsed_ids)
    ):
        raise ScalarFittedStateError(
            "nearest-neighbor cell/component identities must align to train rows"
        )
    parsed_cells = [
        _identifier(cell, label=f"nearest-neighbor train_cell_ids[{index}]")
        for index, cell in enumerate(raw_cells)
    ]
    parsed_components = [
        _feature_text(
            component,
            label=f"nearest-neighbor train_source_component_ids[{index}]",
        )
        for index, component in enumerate(raw_components)
    ]
    raw_matrix = value.get("train_standardized_features")
    if (
        isinstance(raw_matrix, (str, bytes))
        or not isinstance(raw_matrix, Sequence)
        or len(raw_matrix) != len(parsed_ids)
    ):
        raise ScalarFittedStateError(
            "nearest-neighbor train feature matrix row count is invalid"
        )
    feature_rows = [
        _float_vector(
            row,
            label=f"nearest-neighbor train feature row {index}",
            expected_length=feature_dimension,
        )
        for index, row in enumerate(raw_matrix)
    ]
    targets = _float_vector(
        value.get("train_targets"),
        label="nearest-neighbor train_targets",
        expected_length=len(parsed_ids),
    )
    if value.get("tie_atol") != NN_TIE_ATOL:
        raise ScalarFittedStateError("nearest-neighbor tie tolerance changed")
    if value.get("tie_rule") != NN_TIE_RULE:
        raise ScalarFittedStateError("nearest-neighbor tie rule changed")
    return {
        "kind": "STANDARDIZED_EXACT_NEAREST_NEIGHBOR",
        "scaler_mean": mean,
        "scaler_scale": scale,
        "train_row_ids": parsed_ids,
        "train_cell_ids": parsed_cells,
        "train_source_component_ids": parsed_components,
        "train_standardized_features": feature_rows,
        "train_targets": targets,
        "tie_atol": NN_TIE_ATOL,
        "tie_rule": NN_TIE_RULE,
    }


def _validate_training_design_binding(
    value: Mapping, *, method_id: str
) -> dict:
    _strict_keys(value, _TRAINING_DESIGN_KEYS, label="training_design_binding")
    if value.get("schema_version") != SCALAR_TRAINING_DESIGN_SCHEMA:
        raise ScalarFittedStateError(
            "training-design binding schema is unsupported"
        )
    cells = _sorted_unique_ids(
        value.get("cell_ids"), label="training_design_binding.cell_ids"
    )
    components = _sorted_unique_ids(
        value.get("source_component_ids"),
        label="training_design_binding.source_component_ids",
    )
    if not cells:
        raise ScalarFittedStateError(
            "training-design binding requires at least one typed cell ID"
        )
    uses_components = method_id in COMPONENT_ASSIGNMENT_REQUIRED_METHOD_IDS
    expected_policy = (
        FORMAL_WEIGHT_POLICY if uses_components else "NOT_USED_ZERO_DELTA"
    )
    if value.get("training_weight_policy") != expected_policy:
        raise ScalarFittedStateError(
            "training-design binding uses the wrong formal weight policy"
        )
    if uses_components and not components:
        raise ScalarFittedStateError(
            "weighted scalar training requires global source-component IDs"
        )
    if not uses_components and components:
        raise ScalarFittedStateError(
            "ZERO_DELTA cannot bind unused source-component IDs"
        )
    return {
        "schema_version": SCALAR_TRAINING_DESIGN_SCHEMA,
        "design_sha256": _digest(
            value.get("design_sha256"), label="training design sha256"
        ),
        "formal_weights_sha256": _digest(
            value.get("formal_weights_sha256"),
            label="training design formal weights sha256",
        ),
        "training_weight_policy": expected_policy,
        "ordered_cell_ids_sha256": _digest(
            value.get("ordered_cell_ids_sha256"),
            label="training design ordered cell IDs sha256",
        ),
        "ordered_source_component_ids_sha256": _digest(
            value.get("ordered_source_component_ids_sha256"),
            label="training design ordered component IDs sha256",
        ),
        "cell_ids": cells,
        "source_component_ids": components,
    }


def validate_scalar_fitted_state(payload: Mapping) -> dict:
    if not isinstance(payload, Mapping):
        raise ScalarFittedStateError("scalar fitted state must be an object")
    if payload.get("schema_version") != SCALAR_FITTED_STATE_SCHEMA:
        raise ScalarFittedStateError("unsupported scalar fitted-state schema")
    _strict_keys(payload, _STATE_KEYS, label="scalar fitted state")
    if payload.get("artifact_role") != SCALAR_FITTED_STATE_ROLE:
        raise ScalarFittedStateError("scalar fitted-state artifact role is invalid")
    if payload.get("status") != SCALAR_FITTED_STATE_STATUS:
        raise ScalarFittedStateError("scalar fitted-state status is invalid")
    method = _method_id(payload.get("method_id"))
    if payload.get("information_set") != METHOD_INFORMATION_SETS[method]:
        raise ScalarFittedStateError("scalar fitted-state information set is invalid")
    fingerprint = _method_information_set_fingerprint(
        method,
        payload.get("information_set_fingerprint"),
        label="information_set_fingerprint",
    )
    estimand_id = _identifier(payload.get("estimand_id"), label="estimand_id")
    outcomes_sha = _digest(
        payload.get("train_development_outcomes_sha256"),
        label="train_development_outcomes_sha256",
    )
    authorities = _non_data_authority_hashes(
        payload.get("non_data_authority_hashes")
    )
    train_binding = _validate_row_binding(
        payload.get("train_row_binding"), label="train_row_binding"
    )
    component_binding = payload.get("source_component_assignment_binding")
    if method in COMPONENT_ASSIGNMENT_REQUIRED_METHOD_IDS:
        _strict_keys(
            component_binding,
            _COMPONENT_BINDING_KEYS,
            label="source_component_assignment_binding",
        )
        normalized_component_binding = {
            "schema_version": component_binding.get("schema_version"),
            "sha256": _digest(
                component_binding.get("sha256"),
                label="source_component_assignment_binding.sha256",
            ),
            "source_component_split_manifest_source_sha256": _digest(
                component_binding.get(
                    "source_component_split_manifest_source_sha256"
                ),
                label=(
                    "source_component_assignment_binding."
                    "source_component_split_manifest_source_sha256"
                ),
            ),
            "source_component_split_manifest_projection_sha256": _digest(
                component_binding.get(
                    "source_component_split_manifest_projection_sha256"
                ),
                label=(
                    "source_component_assignment_binding."
                    "source_component_split_manifest_projection_sha256"
                ),
            ),
        }
        if (
            normalized_component_binding["schema_version"]
            != TRAIN_COMPONENT_ASSIGNMENT_SCHEMA
        ):
            raise ScalarFittedStateError(
                "component-assignment binding schema is invalid"
            )
        if (
            normalized_component_binding[
                "source_component_split_manifest_source_sha256"
            ]
            != authorities["source_component_split_manifest_sha256"]
        ):
            raise ScalarFittedStateError(
                "component-assignment binding differs from non-data authority"
            )
    else:
        if component_binding is not None:
            raise ScalarFittedStateError(
                "ZERO_DELTA cannot bind an unused component assignment"
            )
        normalized_component_binding = None
    design_binding = _validate_training_design_binding(
        payload.get("training_design_binding"), method_id=method
    )
    config = _validate_fit_config(payload.get("fit_config"), method)
    if payload.get("fit_config_sha256") != _canonical_sha256(config):
        raise ScalarFittedStateError("fit_config_sha256 is invalid")
    vocab = _validate_condition_vocab(payload.get("condition_vocab"))
    feature_schema = _validate_feature_schema(
        payload.get("feature_schema"), method, vocab
    )
    if method == "ZERO_DELTA":
        state = payload.get("state")
        _strict_keys(state, _ZERO_STATE_KEYS, label="zero state")
        if state.get("kind") != "ZERO_DELTA" or state.get("constant") != 0.0:
            raise ScalarFittedStateError("ZERO_DELTA state must be exact constant zero")
        normalized_state = {"kind": "ZERO_DELTA", "constant": 0.0}
    elif method == "SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR":
        normalized_state = _validate_prior_state(payload.get("state"))
    elif method in RIDGE_METHOD_IDS:
        normalized_state = _validate_ridge_state(
            payload.get("state"),
            feature_dimension=feature_schema["total_feature_dimension"],
            n_train_rows=train_binding["n_rows"],
        )
    else:
        normalized_state = _validate_nn_state(
            payload.get("state"),
            feature_dimension=feature_schema["total_feature_dimension"],
            train_binding=train_binding,
        )
        if sorted(set(normalized_state["train_cell_ids"])) != design_binding[
            "cell_ids"
        ]:
            raise ScalarFittedStateError(
                "nearest-neighbor cell IDs differ from training-design binding"
            )
        if sorted(set(normalized_state["train_source_component_ids"])) != (
            design_binding["source_component_ids"]
        ):
            raise ScalarFittedStateError(
                "nearest-neighbor component IDs differ from training-design binding"
            )
    _boundaries(payload, label="scalar fitted state")
    return {
        **dict(payload),
        "method_id": method,
        "information_set_fingerprint": fingerprint,
        "estimand_id": estimand_id,
        "train_development_outcomes_sha256": outcomes_sha,
        "non_data_authority_hashes": authorities,
        "train_row_binding": train_binding,
        "source_component_assignment_binding": normalized_component_binding,
        "training_design_binding": design_binding,
        "fit_config": config,
        "feature_schema": feature_schema,
        "condition_vocab": vocab,
        "state": normalized_state,
    }


def canonical_scalar_fitted_state_bytes(payload: Mapping) -> bytes:
    try:
        return canonical_json_bytes(validate_scalar_fitted_state(payload))
    except ArtifactIOError as exc:
        raise ScalarFittedStateError(str(exc)) from exc


def _ridge_predictions(
    state: Mapping, features: np.ndarray
) -> np.ndarray:
    mean = np.asarray(state["scaler_mean"], dtype=np.float64)
    scale = np.asarray(state["scaler_scale"], dtype=np.float64)
    coef = np.asarray(state["coef"], dtype=np.float64)
    return ((features - mean) / scale) @ coef + float(state["y_mean"])


def _prior_predictions(
    state: Mapping, rows: Sequence[Mapping]
) -> np.ndarray:
    lookup = {
        (
            item["endpoint"],
            item["publication_context_id"],
            item["region"],
        ): float(item["formal_value"])
        for item in state["condition_formal_values"]
    }
    fallback = float(state["fallback_value"])
    return np.asarray(
        [lookup.get(_condition(row), fallback) for row in rows],
        dtype=np.float64,
    )


def _nearest_neighbor_predictions(
    state: Mapping,
    features: np.ndarray,
    *,
    train_chunk_size: int,
) -> np.ndarray:
    chunk = _positive_int(train_chunk_size, label="nearest-neighbor train_chunk_size")
    mean = np.asarray(state["scaler_mean"], dtype=np.float64)
    scale = np.asarray(state["scaler_scale"], dtype=np.float64)
    train_features = np.asarray(
        state["train_standardized_features"], dtype=np.float64
    )
    target = np.asarray(state["train_targets"], dtype=np.float64)
    train_cells = list(state["train_cell_ids"])
    train_components = list(state["train_source_component_ids"])
    query_features = (np.asarray(features, dtype=np.float64) - mean) / scale
    predictions = []
    for query in query_features:
        minimum = np.inf
        for start in range(0, len(train_features), chunk):
            distance = np.sum(
                (train_features[start : start + chunk] - query) ** 2,
                axis=1,
            )
            if not np.all(np.isfinite(distance)):
                raise ScalarFittedStateError(
                    "nearest-neighbor distance replay is non-finite"
                )
            minimum = min(minimum, float(np.min(distance)))
        tied_indices: list[int] = []
        for start in range(0, len(train_features), chunk):
            distance = np.sum(
                (train_features[start : start + chunk] - query) ** 2,
                axis=1,
            )
            tied = np.isclose(
                distance, minimum, rtol=0.0, atol=NN_TIE_ATOL
            )
            for offset in np.flatnonzero(tied).tolist():
                tied_indices.append(start + int(offset))
        if not tied_indices:
            raise ScalarFittedStateError(
                "nearest-neighbor replay produced no exact tied minimum"
            )
        predictions.append(_formal_cell_component_row_mean(
            [float(target[index]) for index in tied_indices],
            [train_cells[index] for index in tied_indices],
            [train_components[index] for index in tied_indices],
        ))
    return np.asarray(predictions, dtype=np.float64)


def recompute_scalar_predictions(
    fitted_state: Mapping,
    development_rows: Sequence[Mapping],
    *,
    nearest_neighbor_train_chunk_size: int | None = None,
) -> list[dict]:
    """Recompute development predictions without accepting caller metrics."""

    payload = validate_scalar_fitted_state(fitted_state)
    rows = _normalized_typed_rows(
        development_rows,
        partition=DEVELOPMENT_PARTITION,
        label="development_rows",
    )
    if rows[0]["estimand_id"] != payload["estimand_id"]:
        raise ScalarFittedStateError(
            "development rows belong to a different fitted estimand"
        )
    method = payload["method_id"]
    if method == "ZERO_DELTA":
        if nearest_neighbor_train_chunk_size is not None:
            raise ScalarFittedStateError(
                "non-nearest-neighbor replay cannot accept a chunk override"
            )
        prediction = np.zeros(len(rows), dtype=np.float64)
    elif method == "SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR":
        if nearest_neighbor_train_chunk_size is not None:
            raise ScalarFittedStateError(
                "non-nearest-neighbor replay cannot accept a chunk override"
            )
        prediction = _prior_predictions(payload["state"], rows)
    elif method in RIDGE_METHOD_IDS:
        if nearest_neighbor_train_chunk_size is not None:
            raise ScalarFittedStateError(
                "non-nearest-neighbor replay cannot accept a chunk override"
            )
        features = _feature_matrix(
            rows, method, payload["condition_vocab"]
        )
        prediction = _ridge_predictions(payload["state"], features)
    else:
        features = _feature_matrix(
            rows, method, payload["condition_vocab"]
        )
        prediction = _nearest_neighbor_predictions(
            payload["state"],
            features,
            train_chunk_size=(
                payload["fit_config"]["nearest_neighbor_train_chunk_size"]
                if nearest_neighbor_train_chunk_size is None
                else nearest_neighbor_train_chunk_size
            ),
        )
    if prediction.shape != (len(rows),) or not np.all(np.isfinite(prediction)):
        raise ScalarFittedStateError(
            "scalar development prediction replay is incomplete or non-finite"
        )
    return [
        {
            "schema_version": "r3x_prediction_row_v1",
            "row_id": row["row_id"],
            "predicted_delta_mean": float(value),
            "predicted_delta_logvar": None,
        }
        for row, value in zip(rows, prediction)
    ]


def _rebuild_and_match_state(
    payload: Mapping,
    *,
    expected_train_rows: Sequence[Mapping],
    expected_method_id: str,
    expected_information_set_fingerprint: str,
    expected_train_development_outcomes_sha256: str,
    expected_non_data_authority_hashes: Mapping,
    component_assignment_authority_rows: Sequence[Mapping] | None,
    component_assignment_path: Path | None,
    expected_component_assignment_sha256: str | None,
    expected_source_component_split_manifest_source_sha256: str | None,
    expected_source_component_split_manifest_projection_sha256: str | None,
) -> dict:
    validated = validate_scalar_fitted_state(payload)
    method = _method_id(expected_method_id)
    if validated["method_id"] != method:
        raise ScalarFittedStateError("scalar fitted-state method binding mismatch")
    expected_fingerprint = _method_information_set_fingerprint(
        method,
        expected_information_set_fingerprint,
        label="expected_information_set_fingerprint",
    )
    if validated["information_set_fingerprint"] != expected_fingerprint:
        raise ScalarFittedStateError(
            "scalar fitted-state information fingerprint mismatch"
        )
    expected_outcomes = _digest(
        expected_train_development_outcomes_sha256,
        label="expected_train_development_outcomes_sha256",
    )
    if validated["train_development_outcomes_sha256"] != expected_outcomes:
        raise ScalarFittedStateError(
            "scalar fitted-state train/development authority mismatch"
        )
    expected_authorities = _non_data_authority_hashes(
        expected_non_data_authority_hashes
    )
    if validated["non_data_authority_hashes"] != expected_authorities:
        raise ScalarFittedStateError(
            "scalar fitted-state non-data authorities mismatch"
        )
    rebuilt = fit_scalar_fitted_state(
        method,
        expected_train_rows,
        information_set_fingerprint=expected_fingerprint,
        train_development_outcomes_sha256=expected_outcomes,
        non_data_authority_hashes=expected_authorities,
        component_assignment_authority_rows=component_assignment_authority_rows,
        component_assignment_path=component_assignment_path,
        component_assignment_sha256=expected_component_assignment_sha256,
        source_component_split_manifest_source_sha256=(
            expected_source_component_split_manifest_source_sha256
        ),
        source_component_split_manifest_projection_sha256=(
            expected_source_component_split_manifest_projection_sha256
        ),
    )
    if rebuilt != validated:
        raise ScalarFittedStateError(
            "scalar fitted state does not match independent exact train-row refit"
        )
    return validated


def write_scalar_fitted_state_exclusive(
    path: Path,
    payload: Mapping,
    *,
    expected_train_rows: Sequence[Mapping],
    expected_method_id: str,
    expected_information_set_fingerprint: str,
    expected_train_development_outcomes_sha256: str,
    expected_non_data_authority_hashes: Mapping,
    component_assignment_authority_rows: Sequence[Mapping] | None = None,
    component_assignment_path: Path | None = None,
    expected_component_assignment_sha256: str | None = None,
    expected_source_component_split_manifest_source_sha256: str | None = None,
    expected_source_component_split_manifest_projection_sha256: str | None = None,
) -> str:
    output = _assert_development_path(path, label="scalar fitted-state output")
    validated = _rebuild_and_match_state(
        payload,
        expected_train_rows=expected_train_rows,
        expected_method_id=expected_method_id,
        expected_information_set_fingerprint=expected_information_set_fingerprint,
        expected_train_development_outcomes_sha256=(
            expected_train_development_outcomes_sha256
        ),
        expected_non_data_authority_hashes=expected_non_data_authority_hashes,
        component_assignment_authority_rows=component_assignment_authority_rows,
        component_assignment_path=component_assignment_path,
        expected_component_assignment_sha256=expected_component_assignment_sha256,
        expected_source_component_split_manifest_source_sha256=(
            expected_source_component_split_manifest_source_sha256
        ),
        expected_source_component_split_manifest_projection_sha256=(
            expected_source_component_split_manifest_projection_sha256
        ),
    )
    try:
        return atomic_write_json_exclusive(
            output, validated, validator=validate_scalar_fitted_state
        )
    except ArtifactIOError as exc:
        raise ScalarFittedStateError(str(exc)) from exc


def load_scalar_fitted_state_exact(
    path: Path,
    expected_sha256: str,
    *,
    expected_train_rows: Sequence[Mapping],
    expected_method_id: str,
    expected_information_set_fingerprint: str,
    expected_train_development_outcomes_sha256: str,
    expected_non_data_authority_hashes: Mapping,
    component_assignment_authority_rows: Sequence[Mapping] | None = None,
    component_assignment_path: Path | None = None,
    expected_component_assignment_sha256: str | None = None,
    expected_source_component_split_manifest_source_sha256: str | None = None,
    expected_source_component_split_manifest_projection_sha256: str | None = None,
) -> dict:
    input_path = _assert_development_path(path, label="scalar fitted-state input")
    digest = _digest(expected_sha256, label="expected_sha256")
    try:
        raw = read_exact_regular_file(input_path, digest)
    except ArtifactIOError as exc:
        raise ScalarFittedStateError(str(exc)) from exc
    payload = validate_scalar_fitted_state(
        _parse_json_object(raw, label="scalar fitted state")
    )
    if canonical_scalar_fitted_state_bytes(payload) != raw:
        raise ScalarFittedStateError("scalar fitted-state bytes are not canonical")
    return _rebuild_and_match_state(
        payload,
        expected_train_rows=expected_train_rows,
        expected_method_id=expected_method_id,
        expected_information_set_fingerprint=expected_information_set_fingerprint,
        expected_train_development_outcomes_sha256=(
            expected_train_development_outcomes_sha256
        ),
        expected_non_data_authority_hashes=expected_non_data_authority_hashes,
        component_assignment_authority_rows=component_assignment_authority_rows,
        component_assignment_path=component_assignment_path,
        expected_component_assignment_sha256=expected_component_assignment_sha256,
        expected_source_component_split_manifest_source_sha256=(
            expected_source_component_split_manifest_source_sha256
        ),
        expected_source_component_split_manifest_projection_sha256=(
            expected_source_component_split_manifest_projection_sha256
        ),
    )
