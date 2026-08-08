"""Canonical, exact-replayable development state for the R3X HGB baseline.

The scikit-learn estimator itself is never pickled.  Only its numeric baseline
and non-categorical tree-node arrays are published as strict canonical JSON.
An exact load replays three locally hash-bound dependencies: caller-supplied
canonical fixture rows, a formal-cell/source-component assignment, and the
observed scikit-learn runtime qualification.  It then refits the estimator and
requires byte-equivalent state before returning it.  These mutually consistent
hashes are not an independent data or execution trust root.

This module deliberately does *not* make the current runtime a trusted
execution environment.  The available qualification is an observed local,
CPU, positive-fixture-namespace development replay with no real run-plan trust
root.  The ``SYNTHETIC_`` prefix is a fixture naming convention only, not data
provenance or an authorization boundary.
Consequently ``EXACT_EIGHT_COMPLETE`` and ``TRAINING_READY`` are permanently
false in this schema.  A future trusted execution path needs a new schema and
an independently rooted qualification; changing flags in this artifact is not
an upgrade path.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
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
from .baselines import (
    _component_cell_balanced_sample_weights,
    _condition_vocab,
    _feature_matrix,
    fit_optional_hist_gradient_boosting,
)
from .features import PAIRED_KMER_FEATURE_SCHEMA, paired_kmer_features
from .information_sets import (
    InformationSet,
    InformationSetError,
    information_set_fingerprint as canonical_information_set_fingerprint,
)
from .predictor_provenance import (
    DEVELOPMENT_PARTITION,
    NON_DATA_AUTHORITY_HASH_KEYS,
    TRAIN_PARTITION,
    PredictorProvenanceError,
    normalize_predictor_input_rows,
    predictor_input_jsonl_bytes,
)


HGB_METHOD_ID = "HISTOGRAM_GRADIENT_BOOSTING"
HGB_FITTED_STATE_SCHEMA = "r3x_hgb_numeric_fitted_state_v2"
HGB_ASSIGNMENT_SCHEMA = "r3x_train_hgb_formal_cell_component_assignment_v2"
HGB_ENVIRONMENT_SCHEMA = "r3x_hgb_observed_environment_qualification_v2"
HGB_TREE_SCHEMA = "r3x_sklearn_numeric_tree_nodes_v1"
HGB_FIT_CONFIG_SCHEMA = "r3x_hgb_fit_config_v2"

HGB_FITTED_STATE_ROLE = "ORDINARY_DEVELOPMENT_HGB_NUMERIC_FITTED_STATE"
HGB_FITTED_STATE_STATUS = (
    "COMPLETE_FIXTURE_NAMESPACE_DEVELOPMENT_EXACT_REPLAY_NOT_TRUST_ROOT"
)
HGB_ASSIGNMENT_ROLE = "ORDINARY_TRAIN_HGB_FORMAL_CELL_COMPONENT_ASSIGNMENT"
HGB_ASSIGNMENT_STATUS = "COMPLETE_EXACT_TRAIN_ASSIGNMENT"
HGB_ENVIRONMENT_ROLE = "ORDINARY_DEVELOPMENT_HGB_OBSERVED_ENVIRONMENT"
HGB_ENVIRONMENT_STATUS = "OBSERVED_LOCAL_RUNTIME_NOT_TRUST_ROOT_BOUND"

HGB_INFORMATION_SET_ENUM = InformationSet.PAIRED_SEQUENCE_NO_MEASURED_ANCHOR
HGB_INFORMATION_SET = HGB_INFORMATION_SET_ENUM.value
HGB_INFORMATION_SET_FINGERPRINT = canonical_information_set_fingerprint(
    HGB_INFORMATION_SET_ENUM
)
FORMAL_CELL_DEFINITION = (
    "(study_id,estimand_id,publication_context_id,edit_count)"
)
ENVIRONMENT_TRUST_STATUS = "NOT_CONNECTED_TO_REAL_RUN_PLAN_TRUST_ROOT"
ENVIRONMENT_SCOPE = (
    "CPU_POSITIVE_FIXTURE_NAMESPACE_DEVELOPMENT_ONLY_NOT_PROVENANCE"
)
ROW_ORDER_RULE = "CANONICAL_ASCENDING_ROW_ID"
OPENMP_THREAD_POLICY = "PRECAUTIONARY_THREADPOOLCTL_OPENMP_LIMIT_EXACTLY_1"
FIXTURE_NAMESPACE_PREFIX = "SYNTHETIC_"
FIXTURE_NAMESPACE_STATUS = (
    "POSITIVE_FIXTURE_NAMESPACE_ONLY_NOT_PROVENANCE_OR_TRUST_ROOT"
)
HGB_LOSS_SAMPLE_WEIGHT_POLICY = (
    "FORMAL_CELL_THEN_SOURCE_COMPONENT_THEN_ROW_BALANCED_"
    "LOSS_SAMPLE_WEIGHT_ONLY_V2"
)
HGB_BINNING_WEIGHT_POLICY = "SKLEARN_UNWEIGHTED_QUANTILE_BINNING"
HGB_LEAF_SUPPORT_POLICY = "RAW_ROW_COUNT_MIN_SAMPLES_LEAF_20"
HGB_SAMPLE_WEIGHT_NORMALIZATION = "MATH_FSUM_MEAN_WEIGHT_ONE_SUM_EQUALS_N_ROWS"
HGB_WEIGHT_SUM_ALGORITHM = "PYTHON_MATH_FSUM"
HGB_REPLICATION_INVARIANCE = "NOT_ASSERTED"
HGB_MISSING_FEATURE_POLICY = "FINITE_FEATURES_ONLY_MISSING_UNSUPPORTED"

EXACT_EIGHT_COMPLETE = False
TRAINING_READY = False
READINESS_BLOCKER = (
    "real independently trusted HGB environment qualification and external "
    "formal-cell projection are not connected to the run-plan trust root; "
    "the positive fixture namespace is not provenance"
)

MINIMUM_TRAIN_ROWS = 100
HGB_EXPLICIT_CONFIG = {
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "l2_regularization": 1.0,
    "random_state": 42,
    "early_stopping": False,
}
HGB_EFFECTIVE_SKLEARN_1_6_CONFIG = {
    "categorical_features": "from_dtype",
    "early_stopping": False,
    "interaction_cst": None,
    "l2_regularization": 1.0,
    "learning_rate": 0.05,
    "loss": "squared_error",
    "max_bins": 255,
    "max_depth": None,
    "max_features": 1.0,
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 20,
    "monotonic_cst": None,
    "n_iter_no_change": 10,
    "quantile": None,
    "random_state": 42,
    "scoring": "loss",
    "tol": 1e-7,
    "validation_fraction": 0.1,
    "verbose": 0,
    "warm_start": False,
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_PROTECTED_TOKENS = ("internal_test", "gse246381")
_PROTECTED_PATH_PARTS = {"test", "internal_test", "sealed", "restricted"}

_BOUNDARY_KEYS = {
    "ordinary_only",
    "internal_test_access",
    "sealed_access",
    "network_access",
    "gpu_execution",
    "downstream_unlock",
}
_ROW_BINDING_KEYS = {
    "n_rows",
    "ordered_row_ids_sha256",
    "row_set_sha256",
    "content_sha256",
}
_FORMAL_CELL_KEYS = {
    "study_id",
    "estimand_id",
    "publication_context_id",
    "edit_count",
}
_ASSIGNMENT_ROW_KEYS = {
    "row_id",
    "typed_cell_id",
    "source_component_id",
    "formal_cell",
    "formal_cell_sha256",
}
_ASSIGNMENT_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "information_set_fingerprint",
    "train_development_outcomes_sha256",
    "non_data_authority_hashes",
    "source_component_assignment_source_sha256",
    "source_component_assignment_projection_sha256",
    "formal_cell_assignment_source_sha256",
    "formal_cell_assignment_projection_sha256",
    "train_row_binding",
    "assignments",
    "n_typed_cells",
    "n_formal_cells",
    "typed_cell_to_formal_cell_mapping_sha256",
} | _BOUNDARY_KEYS
_ENVIRONMENT_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "execution_scope",
    "trust_root_binding_status",
    "python_implementation",
    "python_version",
    "numpy_version",
    "sklearn_version",
    "threadpoolctl_version",
    "openmp_thread_limit",
    "openmp_thread_policy",
    "sklearn_gradient_boosting_code_sha256",
    "sklearn_tree_predictor_code_sha256",
    "sklearn_compiled_predictor_sha256",
    "threadpoolctl_code_sha256",
    "r3x_hgb_fitted_state_code_sha256",
    "r3x_baselines_code_sha256",
    "r3x_features_code_sha256",
    "exact_eight_complete",
    "training_ready",
    "readiness_blocker",
} | _BOUNDARY_KEYS
_ASSIGNMENT_BINDING_KEYS = {
    "schema_version",
    "sha256",
    "source_component_assignment_source_sha256",
    "source_component_assignment_projection_sha256",
    "formal_cell_assignment_source_sha256",
    "formal_cell_assignment_projection_sha256",
    "n_typed_cells",
    "n_formal_cells",
    "typed_cell_to_formal_cell_mapping_sha256",
}
_ENVIRONMENT_BINDING_KEYS = {
    "schema_version",
    "sha256",
    "sklearn_version",
    "threadpoolctl_version",
    "openmp_thread_limit",
    "openmp_thread_policy",
    "sklearn_gradient_boosting_code_sha256",
    "sklearn_tree_predictor_code_sha256",
    "sklearn_compiled_predictor_sha256",
    "threadpoolctl_code_sha256",
    "r3x_hgb_fitted_state_code_sha256",
}
_FIT_CONFIG_KEYS = {
    "schema_version",
    "method_id",
    "feature_schema",
    "condition_features",
    "row_order_rule",
    "training_weight_policy",
    "formal_cell_definition",
    "binning_weight_policy",
    "leaf_support_policy",
    "sample_weight_normalization",
    "weight_sum_algorithm",
    "replication_invariance",
    "missing_feature_policy",
    "fixture_namespace_status",
    "openmp_thread_policy",
    "minimum_train_rows",
    "explicit_constructor_parameters",
    "effective_sklearn_parameters",
}
_WEIGHT_BINDING_KEYS = {
    "training_weight_policy",
    "formal_cell_definition",
    "sample_weight_normalization",
    "weight_sum_algorithm",
    "replication_invariance",
    "n_formal_cells",
    "ordered_formal_cell_sha256s_sha256",
    "ordered_source_component_ids_sha256",
    "ordered_sample_weights_sha256",
    "sample_weight_sum",
}
_NODE_KEYS = {
    "node_index",
    "value",
    "count",
    "feature_idx",
    "num_threshold",
    "missing_go_to_left",
    "left",
    "right",
    "gain",
    "depth",
    "is_leaf",
    "bin_threshold",
    "is_categorical",
    "bitset_idx",
}
_TREE_KEYS = {"iteration", "nodes"}
_MODEL_STATE_KEYS = {
    "tree_schema",
    "baseline_prediction",
    "n_features",
    "n_trees_per_iteration",
    "n_iterations",
    "trees",
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
    "cell_component_assignment_binding",
    "environment_qualification_binding",
    "feature_schema",
    "condition_vocab",
    "fit_config",
    "fit_config_sha256",
    "training_weight_binding",
    "model_state",
    "fixture_namespace_status",
    "exact_eight_complete",
    "training_ready",
    "readiness_blocker",
} | _BOUNDARY_KEYS


class HGBFittedStateError(RuntimeError):
    """Fail-closed HGB state, assignment, or environment error."""


def _strict_keys(value: Mapping, expected: set[str], *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise HGBFittedStateError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise HGBFittedStateError(
            f"{label} keys differ from strict schema: {sorted(actual ^ expected)}"
        )


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise HGBFittedStateError(f"{label} must be 64 lowercase hex")
    return value


def _hgb_information_set_fingerprint(value: object, *, label: str) -> str:
    parsed = _digest(value, label=label)
    try:
        canonical = canonical_information_set_fingerprint(
            InformationSet.PAIRED_SEQUENCE_NO_MEASURED_ANCHOR
        )
    except InformationSetError as exc:  # pragma: no cover - frozen registry bug
        raise HGBFittedStateError(
            "canonical HGB information-set fingerprint is unavailable"
        ) from exc
    if parsed != canonical or canonical != HGB_INFORMATION_SET_FINGERPRINT:
        raise HGBFittedStateError(
            f"{label} differs from the canonical fingerprint for "
            f"{HGB_INFORMATION_SET}"
        )
    return canonical


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise HGBFittedStateError(f"{label} must be a canonical identifier")
    lowered = value.lower()
    if (
        any(token in lowered for token in _PROTECTED_TOKENS)
        or lowered.startswith("sealed")
        or lowered.startswith("restricted")
    ):
        raise HGBFittedStateError(f"{label} enters a protected namespace")
    return value


def _version_text(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise HGBFittedStateError(f"{label} must be canonical non-empty text")
    return value


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise HGBFittedStateError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise HGBFittedStateError(f"{label} must be a finite number")
    return result


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HGBFittedStateError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: object, *, label: str) -> int:
    result = _nonnegative_int(value, label=label)
    if result == 0:
        raise HGBFittedStateError(f"{label} must be positive")
    return result


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
        raise HGBFittedStateError(f"value is not strict finite JSON: {exc}") from exc
    return hashlib.sha256(raw).hexdigest()


def _sequence_sha256(values: Sequence[object]) -> str:
    return _canonical_sha256(list(values))


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
        "network_access": False,
        "gpu_execution": False,
        "downstream_unlock": False,
    }
    for field, expected_value in expected.items():
        if value.get(field) is not expected_value:
            raise HGBFittedStateError(
                f"{label}.{field} violates ordinary-only development boundaries"
            )


def _boundary_payload() -> dict[str, bool]:
    return {
        "ordinary_only": True,
        "internal_test_access": False,
        "sealed_access": False,
        "network_access": False,
        "gpu_execution": False,
        "downstream_unlock": False,
    }


def _assert_development_path(path: Path, *, label: str) -> Path:
    candidate = Path(path)
    parts = [part.lower() for part in candidate.parts]
    if "development" not in parts:
        raise HGBFittedStateError(
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
            raise HGBFittedStateError(f"{label} enters a protected path namespace")
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
        raise HGBFittedStateError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise HGBFittedStateError(f"{label} must be a JSON object")
    return value


def _normalized_typed_rows(
    rows: Sequence[Mapping], *, partition: str, label: str
) -> list[dict]:
    try:
        normalized = normalize_predictor_input_rows(rows)
    except PredictorProvenanceError as exc:
        raise HGBFittedStateError(str(exc)) from exc
    if any(row["partition"] != partition for row in normalized):
        raise HGBFittedStateError(f"{label} must contain only {partition} rows")
    estimands = {row["estimand_id"] for row in normalized}
    if len(estimands) != 1:
        raise HGBFittedStateError(f"{label} must contain exactly one estimand")
    for row in normalized:
        _identifier(row["estimand_id"], label=f"{label}.estimand_id")
        _identifier(row["cell_id"], label=f"{label}.cell_id")
        _identifier(
            row["predictor_input"]["publication_context_id"],
            label=f"{label}.publication_context_id",
        )
    return normalized


def _train_row_binding(rows: Sequence[Mapping]) -> dict:
    row_ids = [row["row_id"] for row in rows]
    try:
        content = predictor_input_jsonl_bytes(rows)
    except PredictorProvenanceError as exc:
        raise HGBFittedStateError(str(exc)) from exc
    return {
        "n_rows": len(rows),
        "ordered_row_ids_sha256": _sequence_sha256(row_ids),
        "row_set_sha256": _sequence_sha256(sorted(row_ids)),
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


def _normalize_formal_cell(value: Mapping, *, label: str) -> dict:
    _strict_keys(value, _FORMAL_CELL_KEYS, label=label)
    edit_count = value.get("edit_count")
    if (
        isinstance(edit_count, bool)
        or not isinstance(edit_count, int)
        or edit_count not in {1, 3, 5}
    ):
        raise HGBFittedStateError(f"{label}.edit_count must be one of 1, 3, 5")
    return {
        "study_id": _identifier(value.get("study_id"), label=f"{label}.study_id"),
        "estimand_id": _identifier(
            value.get("estimand_id"), label=f"{label}.estimand_id"
        ),
        "publication_context_id": _identifier(
            value.get("publication_context_id"),
            label=f"{label}.publication_context_id",
        ),
        "edit_count": edit_count,
    }


def _formal_cell_sha256(cell: Mapping) -> str:
    normalized = _normalize_formal_cell(cell, label="formal_cell")
    # Hash the ordered mathematical tuple, not an arbitrary caller cell label.
    return _sequence_sha256([
        normalized["study_id"],
        normalized["estimand_id"],
        normalized["publication_context_id"],
        normalized["edit_count"],
    ])


def _typed_formal_cell_mapping_binding(
    assignments: Sequence[Mapping],
) -> dict[str, object]:
    """Require a bijection between registered and mathematical cells.

    A one-to-many mapping would give one registered metric cell multiple units
    of training mass.  A many-to-one mapping would silently merge registered
    cells.  Both are scientific-unit changes, not harmless metadata aliases.
    """

    typed_to_formal: dict[str, str] = {}
    formal_to_typed: dict[str, str] = {}
    for item in assignments:
        typed = str(item["typed_cell_id"])
        formal = str(item["formal_cell_sha256"])
        prior_formal = typed_to_formal.setdefault(typed, formal)
        if prior_formal != formal:
            raise HGBFittedStateError(
                "one typed_cell_id maps to multiple formal cells"
            )
        prior_typed = formal_to_typed.setdefault(formal, typed)
        if prior_typed != typed:
            raise HGBFittedStateError(
                "one formal cell maps to multiple typed_cell_id values"
            )
    ordered_mapping = [
        [typed, typed_to_formal[typed]] for typed in sorted(typed_to_formal)
    ]
    return {
        "n_typed_cells": len(typed_to_formal),
        "n_formal_cells": len(formal_to_typed),
        "typed_cell_to_formal_cell_mapping_sha256": _sequence_sha256(
            ordered_mapping
        ),
    }


def _normalize_assignment_rows(
    assignments: Sequence[Mapping], *, expected_rows: Sequence[Mapping] | None
) -> list[dict]:
    if (
        isinstance(assignments, (str, bytes))
        or not isinstance(assignments, Sequence)
        or not assignments
    ):
        raise HGBFittedStateError("assignments must be a non-empty sequence")
    parsed_by_id = {}
    for index, raw in enumerate(assignments):
        label = f"assignment[{index}]"
        _strict_keys(raw, _ASSIGNMENT_ROW_KEYS, label=label)
        row_id = _digest(raw.get("row_id"), label=f"{label}.row_id")
        if row_id in parsed_by_id:
            raise HGBFittedStateError("assignment row IDs must be unique")
        cell = _normalize_formal_cell(raw.get("formal_cell"), label=f"{label}.formal_cell")
        cell_sha = _digest(
            raw.get("formal_cell_sha256"), label=f"{label}.formal_cell_sha256"
        )
        if cell_sha != _formal_cell_sha256(cell):
            raise HGBFittedStateError("formal_cell_sha256 does not bind the formal tuple")
        parsed_by_id[row_id] = {
            "row_id": row_id,
            "typed_cell_id": _identifier(
                raw.get("typed_cell_id"), label=f"{label}.typed_cell_id"
            ),
            "source_component_id": _identifier(
                raw.get("source_component_id"),
                label=f"{label}.source_component_id",
            ),
            "formal_cell": cell,
            "formal_cell_sha256": cell_sha,
        }
    if expected_rows is None:
        ordered_ids = sorted(parsed_by_id)
    else:
        ordered_ids = [row["row_id"] for row in expected_rows]
        if set(ordered_ids) != set(parsed_by_id):
            raise HGBFittedStateError(
                "assignment rows must cover exactly the bound training row IDs"
            )
    result = [parsed_by_id[row_id] for row_id in ordered_ids]
    if expected_rows is not None:
        for row, assignment in zip(expected_rows, result, strict=True):
            inputs = row["predictor_input"]
            cell = assignment["formal_cell"]
            expected = {
                "typed_cell_id": row["cell_id"],
                "estimand_id": row["estimand_id"],
                "publication_context_id": inputs["publication_context_id"],
                "edit_count": inputs["edit_count"],
            }
            actual = {
                "typed_cell_id": assignment["typed_cell_id"],
                "estimand_id": cell["estimand_id"],
                "publication_context_id": cell["publication_context_id"],
                "edit_count": cell["edit_count"],
            }
            if actual != expected:
                raise HGBFittedStateError(
                    "formal-cell assignment conflicts with its typed predictor row"
                )
    _typed_formal_cell_mapping_binding(result)
    return result


def build_hgb_cell_component_assignment(
    train_rows: Sequence[Mapping],
    assignments: Sequence[Mapping],
    *,
    information_set_fingerprint: str,
    train_development_outcomes_sha256: str,
    non_data_authority_hashes: Mapping,
    source_component_assignment_source_sha256: str,
    source_component_assignment_projection_sha256: str,
    formal_cell_assignment_source_sha256: str,
    formal_cell_assignment_projection_sha256: str,
) -> dict:
    """Stage an externally projected assignment; fitters accept only its file.

    This builder is intentionally not an authorization primitive.  It makes
    the external projection canonical and testable, while all fit/load APIs
    require the resulting immutable file and its independently supplied hash.
    """

    rows = _normalized_typed_rows(
        train_rows, partition=TRAIN_PARTITION, label="train_rows"
    )
    authorities = _non_data_authority_hashes(non_data_authority_hashes)
    component_source = _digest(
        source_component_assignment_source_sha256,
        label="source_component_assignment_source_sha256",
    )
    if component_source != authorities["source_component_split_manifest_sha256"]:
        raise HGBFittedStateError(
            "component source differs from the frozen split-manifest authority"
        )
    assignment_role_hashes = {
        component_source,
        _digest(
            source_component_assignment_projection_sha256,
            label="source_component_assignment_projection_sha256",
        ),
        _digest(
            formal_cell_assignment_source_sha256,
            label="formal_cell_assignment_source_sha256",
        ),
        _digest(
            formal_cell_assignment_projection_sha256,
            label="formal_cell_assignment_projection_sha256",
        ),
    }
    if len(assignment_role_hashes) != 4:
        raise HGBFittedStateError(
            "HGB assignment source/projection roles cannot alias bytes"
        )
    normalized_assignments = _normalize_assignment_rows(
        assignments, expected_rows=rows
    )
    mapping_binding = _typed_formal_cell_mapping_binding(normalized_assignments)
    payload = {
        "schema_version": HGB_ASSIGNMENT_SCHEMA,
        "artifact_role": HGB_ASSIGNMENT_ROLE,
        "status": HGB_ASSIGNMENT_STATUS,
        "information_set_fingerprint": _hgb_information_set_fingerprint(
            information_set_fingerprint, label="information_set_fingerprint"
        ),
        "train_development_outcomes_sha256": _digest(
            train_development_outcomes_sha256,
            label="train_development_outcomes_sha256",
        ),
        "non_data_authority_hashes": authorities,
        "source_component_assignment_source_sha256": component_source,
        "source_component_assignment_projection_sha256": _digest(
            source_component_assignment_projection_sha256,
            label="source_component_assignment_projection_sha256",
        ),
        "formal_cell_assignment_source_sha256": _digest(
            formal_cell_assignment_source_sha256,
            label="formal_cell_assignment_source_sha256",
        ),
        "formal_cell_assignment_projection_sha256": _digest(
            formal_cell_assignment_projection_sha256,
            label="formal_cell_assignment_projection_sha256",
        ),
        "train_row_binding": _train_row_binding(rows),
        "assignments": normalized_assignments,
        **mapping_binding,
        **_boundary_payload(),
    }
    return validate_hgb_cell_component_assignment(payload)


def validate_hgb_cell_component_assignment(payload: Mapping) -> dict:
    if not isinstance(payload, Mapping):
        raise HGBFittedStateError("HGB assignment must be an object")
    if payload.get("schema_version") != HGB_ASSIGNMENT_SCHEMA:
        raise HGBFittedStateError("unsupported HGB assignment schema")
    _strict_keys(payload, _ASSIGNMENT_KEYS, label="HGB assignment")
    if payload.get("artifact_role") != HGB_ASSIGNMENT_ROLE:
        raise HGBFittedStateError("HGB assignment role is invalid")
    if payload.get("status") != HGB_ASSIGNMENT_STATUS:
        raise HGBFittedStateError("HGB assignment status is invalid")
    authorities = _non_data_authority_hashes(
        payload.get("non_data_authority_hashes")
    )
    component_source = _digest(
        payload.get("source_component_assignment_source_sha256"),
        label="source_component_assignment_source_sha256",
    )
    if component_source != authorities["source_component_split_manifest_sha256"]:
        raise HGBFittedStateError("HGB assignment component source is stale")
    binding = _validate_row_binding(
        payload.get("train_row_binding"), label="train_row_binding"
    )
    assignments = _normalize_assignment_rows(
        payload.get("assignments"), expected_rows=None
    )
    mapping_binding = _typed_formal_cell_mapping_binding(assignments)
    provided_mapping_binding = {
        "n_typed_cells": _positive_int(
            payload.get("n_typed_cells"), label="n_typed_cells"
        ),
        "n_formal_cells": _positive_int(
            payload.get("n_formal_cells"), label="n_formal_cells"
        ),
        "typed_cell_to_formal_cell_mapping_sha256": _digest(
            payload.get("typed_cell_to_formal_cell_mapping_sha256"),
            label="typed_cell_to_formal_cell_mapping_sha256",
        ),
    }
    for field, expected_value in mapping_binding.items():
        if provided_mapping_binding[field] != expected_value:
            raise HGBFittedStateError(
                f"HGB assignment {field} binding mismatch"
            )
    if payload.get("assignments") != assignments:
        raise HGBFittedStateError("HGB assignments are not in canonical row order")
    if len(assignments) != binding["n_rows"]:
        raise HGBFittedStateError("HGB assignment row count differs from binding")
    row_ids = [item["row_id"] for item in assignments]
    if _sequence_sha256(row_ids) != binding["ordered_row_ids_sha256"]:
        raise HGBFittedStateError("HGB assignment row order hash mismatch")
    if _sequence_sha256(sorted(row_ids)) != binding["row_set_sha256"]:
        raise HGBFittedStateError("HGB assignment row set hash mismatch")
    _hgb_information_set_fingerprint(
        payload.get("information_set_fingerprint"),
        label="information_set_fingerprint",
    )
    for field in (
        "train_development_outcomes_sha256",
        "source_component_assignment_projection_sha256",
        "formal_cell_assignment_source_sha256",
        "formal_cell_assignment_projection_sha256",
    ):
        _digest(payload.get(field), label=field)
    assignment_role_hashes = {
        component_source,
        payload["source_component_assignment_projection_sha256"],
        payload["formal_cell_assignment_source_sha256"],
        payload["formal_cell_assignment_projection_sha256"],
    }
    if len(assignment_role_hashes) != 4:
        raise HGBFittedStateError(
            "HGB assignment source/projection roles cannot alias bytes"
        )
    _boundaries(payload, label="HGB assignment")
    return {
        **dict(payload),
        "non_data_authority_hashes": authorities,
        "train_row_binding": binding,
        "assignments": assignments,
    }


def canonical_hgb_cell_component_assignment_bytes(payload: Mapping) -> bytes:
    try:
        return canonical_json_bytes(validate_hgb_cell_component_assignment(payload))
    except ArtifactIOError as exc:
        raise HGBFittedStateError(str(exc)) from exc


def _verify_assignment_bindings(
    payload: Mapping,
    *,
    expected_train_rows: Sequence[Mapping],
    expected_information_set_fingerprint: str,
    expected_train_development_outcomes_sha256: str,
    expected_non_data_authority_hashes: Mapping,
    expected_source_component_assignment_source_sha256: str,
    expected_source_component_assignment_projection_sha256: str,
    expected_formal_cell_assignment_source_sha256: str,
    expected_formal_cell_assignment_projection_sha256: str,
) -> list[dict]:
    rows = _normalized_typed_rows(
        expected_train_rows,
        partition=TRAIN_PARTITION,
        label="expected_train_rows",
    )
    expected = {
        "information_set_fingerprint": _hgb_information_set_fingerprint(
            expected_information_set_fingerprint,
            label="expected_information_set_fingerprint",
        ),
        "train_development_outcomes_sha256": _digest(
            expected_train_development_outcomes_sha256,
            label="expected_train_development_outcomes_sha256",
        ),
        "non_data_authority_hashes": _non_data_authority_hashes(
            expected_non_data_authority_hashes
        ),
        "source_component_assignment_source_sha256": _digest(
            expected_source_component_assignment_source_sha256,
            label="expected_source_component_assignment_source_sha256",
        ),
        "source_component_assignment_projection_sha256": _digest(
            expected_source_component_assignment_projection_sha256,
            label="expected_source_component_assignment_projection_sha256",
        ),
        "formal_cell_assignment_source_sha256": _digest(
            expected_formal_cell_assignment_source_sha256,
            label="expected_formal_cell_assignment_source_sha256",
        ),
        "formal_cell_assignment_projection_sha256": _digest(
            expected_formal_cell_assignment_projection_sha256,
            label="expected_formal_cell_assignment_projection_sha256",
        ),
        "train_row_binding": _train_row_binding(rows),
    }
    for field, value in expected.items():
        if payload[field] != value:
            raise HGBFittedStateError(f"HGB assignment {field} binding mismatch")
    checked = _normalize_assignment_rows(
        payload["assignments"], expected_rows=rows
    )
    if checked != payload["assignments"]:
        raise HGBFittedStateError("HGB assignment order differs from train rows")
    return rows


def write_hgb_cell_component_assignment_exclusive(
    path: Path,
    payload: Mapping,
    **expected_bindings,
) -> str:
    output = _assert_development_path(path, label="HGB assignment output")
    validated = validate_hgb_cell_component_assignment(payload)
    _verify_assignment_bindings(validated, **expected_bindings)
    try:
        return atomic_write_json_exclusive(
            output,
            validated,
            validator=validate_hgb_cell_component_assignment,
        )
    except ArtifactIOError as exc:
        raise HGBFittedStateError(str(exc)) from exc


def load_hgb_cell_component_assignment_exact(
    path: Path,
    expected_sha256: str,
    **expected_bindings,
) -> dict:
    input_path = _assert_development_path(path, label="HGB assignment input")
    digest = _digest(expected_sha256, label="expected assignment sha256")
    try:
        raw = read_exact_regular_file(input_path, digest)
    except ArtifactIOError as exc:
        raise HGBFittedStateError(str(exc)) from exc
    payload = validate_hgb_cell_component_assignment(
        _parse_json_object(raw, label="HGB assignment")
    )
    if canonical_hgb_cell_component_assignment_bytes(payload) != raw:
        raise HGBFittedStateError("HGB assignment bytes are not canonical")
    _verify_assignment_bindings(payload, **expected_bindings)
    return payload


def _sha256_regular_file(path: Path, *, label: str) -> str:
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        raise HGBFittedStateError(f"{label} is not a real regular file")
    return hashlib.sha256(candidate.read_bytes()).hexdigest()


def observe_hgb_environment_qualification() -> dict:
    """Observe the exact local runtime without claiming it is trusted."""

    try:
        import sklearn
        import sklearn.ensemble._hist_gradient_boosting._predictor as compiled
        import sklearn.ensemble._hist_gradient_boosting.gradient_boosting as gradient
        import sklearn.ensemble._hist_gradient_boosting.predictor as predictor
        import threadpoolctl
    except Exception as exc:  # pragma: no cover - optional environment failure
        raise HGBFittedStateError(
            f"scikit-learn HGB runtime is unavailable: {type(exc).__name__}: {exc}"
        ) from exc
    module_path = Path(__file__)
    baselines_path = module_path.with_name("baselines.py")
    features_path = module_path.with_name("features.py")
    payload = {
        "schema_version": HGB_ENVIRONMENT_SCHEMA,
        "artifact_role": HGB_ENVIRONMENT_ROLE,
        "status": HGB_ENVIRONMENT_STATUS,
        "execution_scope": ENVIRONMENT_SCOPE,
        "trust_root_binding_status": ENVIRONMENT_TRUST_STATUS,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "threadpoolctl_version": threadpoolctl.__version__,
        "openmp_thread_limit": 1,
        "openmp_thread_policy": OPENMP_THREAD_POLICY,
        "sklearn_gradient_boosting_code_sha256": _sha256_regular_file(
            Path(gradient.__file__), label="sklearn gradient-boosting code"
        ),
        "sklearn_tree_predictor_code_sha256": _sha256_regular_file(
            Path(predictor.__file__), label="sklearn tree-predictor code"
        ),
        "sklearn_compiled_predictor_sha256": _sha256_regular_file(
            Path(compiled.__file__), label="sklearn compiled predictor"
        ),
        "threadpoolctl_code_sha256": _sha256_regular_file(
            Path(threadpoolctl.__file__), label="threadpoolctl code"
        ),
        "r3x_hgb_fitted_state_code_sha256": _sha256_regular_file(
            module_path, label="R3X HGB fitted-state code"
        ),
        "r3x_baselines_code_sha256": _sha256_regular_file(
            baselines_path, label="R3X baseline code"
        ),
        "r3x_features_code_sha256": _sha256_regular_file(
            features_path, label="R3X feature code"
        ),
        "exact_eight_complete": False,
        "training_ready": False,
        "readiness_blocker": READINESS_BLOCKER,
        **_boundary_payload(),
    }
    return validate_hgb_environment_qualification(payload)


def validate_hgb_environment_qualification(payload: Mapping) -> dict:
    if not isinstance(payload, Mapping):
        raise HGBFittedStateError("HGB environment qualification must be an object")
    if payload.get("schema_version") != HGB_ENVIRONMENT_SCHEMA:
        raise HGBFittedStateError("unsupported HGB environment schema")
    _strict_keys(payload, _ENVIRONMENT_KEYS, label="HGB environment qualification")
    expected = {
        "artifact_role": HGB_ENVIRONMENT_ROLE,
        "status": HGB_ENVIRONMENT_STATUS,
        "execution_scope": ENVIRONMENT_SCOPE,
        "trust_root_binding_status": ENVIRONMENT_TRUST_STATUS,
        "exact_eight_complete": False,
        "training_ready": False,
        "readiness_blocker": READINESS_BLOCKER,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise HGBFittedStateError(f"HGB environment {field} is invalid")
    for field in (
        "python_implementation",
        "python_version",
        "numpy_version",
        "sklearn_version",
        "threadpoolctl_version",
    ):
        _version_text(payload.get(field), label=field)
    if payload.get("openmp_thread_limit") != 1:
        raise HGBFittedStateError("HGB environment openmp_thread_limit must be 1")
    if payload.get("openmp_thread_policy") != OPENMP_THREAD_POLICY:
        raise HGBFittedStateError("HGB environment OpenMP policy is invalid")
    for field in (
        "sklearn_gradient_boosting_code_sha256",
        "sklearn_tree_predictor_code_sha256",
        "sklearn_compiled_predictor_sha256",
        "threadpoolctl_code_sha256",
        "r3x_hgb_fitted_state_code_sha256",
        "r3x_baselines_code_sha256",
        "r3x_features_code_sha256",
    ):
        _digest(payload.get(field), label=field)
    _boundaries(payload, label="HGB environment qualification")
    return dict(payload)


def canonical_hgb_environment_qualification_bytes(payload: Mapping) -> bytes:
    try:
        return canonical_json_bytes(validate_hgb_environment_qualification(payload))
    except ArtifactIOError as exc:
        raise HGBFittedStateError(str(exc)) from exc


def write_hgb_environment_qualification_exclusive(
    path: Path, payload: Mapping
) -> str:
    output = _assert_development_path(path, label="HGB environment output")
    validated = validate_hgb_environment_qualification(payload)
    if validated != observe_hgb_environment_qualification():
        raise HGBFittedStateError(
            "HGB environment qualification differs from current runtime"
        )
    try:
        return atomic_write_json_exclusive(
            output,
            validated,
            validator=validate_hgb_environment_qualification,
        )
    except ArtifactIOError as exc:
        raise HGBFittedStateError(str(exc)) from exc


def load_hgb_environment_qualification_exact(
    path: Path, expected_sha256: str
) -> dict:
    input_path = _assert_development_path(path, label="HGB environment input")
    digest = _digest(expected_sha256, label="expected environment sha256")
    try:
        raw = read_exact_regular_file(input_path, digest)
    except ArtifactIOError as exc:
        raise HGBFittedStateError(str(exc)) from exc
    payload = validate_hgb_environment_qualification(
        _parse_json_object(raw, label="HGB environment qualification")
    )
    if canonical_hgb_environment_qualification_bytes(payload) != raw:
        raise HGBFittedStateError("HGB environment bytes are not canonical")
    if payload != observe_hgb_environment_qualification():
        raise HGBFittedStateError(
            "HGB environment qualification does not match current runtime"
        )
    return payload


def _flat_baseline_rows(
    rows: Sequence[Mapping],
    *,
    source_component_by_row_id: Mapping[str, str] | None = None,
) -> list[dict]:
    result = []
    for row in rows:
        inputs = row["predictor_input"]
        flat = {
            "record_id": row["row_id"],
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
        if row["partition"] == TRAIN_PARTITION:
            flat["delta"] = float(row["training_target"])
            if source_component_by_row_id is None:
                raise HGBFittedStateError(
                    "training feature rows require exact component assignment"
                )
            flat["source_component_id"] = source_component_by_row_id[row["row_id"]]
        result.append(flat)
    return result


def _fit_config() -> dict:
    return {
        "schema_version": HGB_FIT_CONFIG_SCHEMA,
        "method_id": HGB_METHOD_ID,
        "feature_schema": PAIRED_KMER_FEATURE_SCHEMA,
        "condition_features": (
            "TRAIN_ONLY_ENDPOINT_CONTEXT_REGION_ONE_HOT_WITH_UNK"
        ),
        "row_order_rule": ROW_ORDER_RULE,
        "training_weight_policy": HGB_LOSS_SAMPLE_WEIGHT_POLICY,
        "formal_cell_definition": FORMAL_CELL_DEFINITION,
        "binning_weight_policy": HGB_BINNING_WEIGHT_POLICY,
        "leaf_support_policy": HGB_LEAF_SUPPORT_POLICY,
        "sample_weight_normalization": HGB_SAMPLE_WEIGHT_NORMALIZATION,
        "weight_sum_algorithm": HGB_WEIGHT_SUM_ALGORITHM,
        "replication_invariance": HGB_REPLICATION_INVARIANCE,
        "missing_feature_policy": HGB_MISSING_FEATURE_POLICY,
        "fixture_namespace_status": FIXTURE_NAMESPACE_STATUS,
        "openmp_thread_policy": OPENMP_THREAD_POLICY,
        "minimum_train_rows": MINIMUM_TRAIN_ROWS,
        "explicit_constructor_parameters": dict(HGB_EXPLICIT_CONFIG),
        "effective_sklearn_parameters": dict(HGB_EFFECTIVE_SKLEARN_1_6_CONFIG),
    }


def _validate_fit_config(value: Mapping) -> dict:
    _strict_keys(value, _FIT_CONFIG_KEYS, label="HGB fit_config")
    expected = _fit_config()
    if dict(value) != expected:
        raise HGBFittedStateError("HGB fit_config differs from frozen semantics")
    return expected


def _normalize_loss_sample_weights(
    raw_weights: Sequence[Real], *, n_rows: int
) -> np.ndarray:
    """Normalize positive loss weights with stable summation.

    The absolute scale is frozen to mean weight one because sklearn's fixed L2
    penalty interacts with weight scale.  This does not make HGB replication
    invariant: quantile binning and min_samples_leaf remain raw-row based.
    """

    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows <= 0:
        raise HGBFittedStateError("HGB loss weights require positive n_rows")
    weights = np.asarray(raw_weights, dtype=np.float64)
    if weights.shape != (n_rows,):
        raise HGBFittedStateError("HGB loss weights do not align to training rows")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise HGBFittedStateError("HGB loss weights must be finite positive")
    total = math.fsum(float(value) for value in weights)
    if not math.isfinite(total) or total <= 0.0:
        raise HGBFittedStateError("HGB loss-weight sum is not finite positive")
    normalized = np.asarray(
        weights * (float(n_rows) / total), dtype=np.float64
    )
    if not np.all(np.isfinite(normalized)) or np.any(normalized <= 0.0):
        raise HGBFittedStateError("normalized HGB loss weights are invalid")
    normalized_sum = math.fsum(float(value) for value in normalized)
    tolerance = 8.0 * np.finfo(np.float64).eps * max(1.0, float(n_rows))
    if not math.isclose(
        normalized_sum, float(n_rows), rel_tol=0.0, abs_tol=tolerance
    ):
        raise HGBFittedStateError(
            "normalized HGB loss weights do not have frozen mean-one scale"
        )
    return normalized


def _training_weights(
    rows: Sequence[Mapping], assignment: Mapping
) -> tuple[np.ndarray, dict]:
    assignment_by_id = {
        item["row_id"]: item for item in assignment["assignments"]
    }
    formal_cell_ids = [
        assignment_by_id[row["row_id"]]["formal_cell_sha256"] for row in rows
    ]
    components = [
        assignment_by_id[row["row_id"]]["source_component_id"] for row in rows
    ]
    try:
        weights = _normalize_loss_sample_weights(
            _component_cell_balanced_sample_weights(
                formal_cell_ids, components
            ),
            n_rows=len(rows),
        )
    except Exception as exc:
        raise HGBFittedStateError(f"HGB training weights failed: {exc}") from exc
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise HGBFittedStateError("HGB training weights are not finite positive")
    binding = {
        "training_weight_policy": HGB_LOSS_SAMPLE_WEIGHT_POLICY,
        "formal_cell_definition": FORMAL_CELL_DEFINITION,
        "sample_weight_normalization": HGB_SAMPLE_WEIGHT_NORMALIZATION,
        "weight_sum_algorithm": HGB_WEIGHT_SUM_ALGORITHM,
        "replication_invariance": HGB_REPLICATION_INVARIANCE,
        "n_formal_cells": len(set(formal_cell_ids)),
        "ordered_formal_cell_sha256s_sha256": _sequence_sha256(formal_cell_ids),
        "ordered_source_component_ids_sha256": _sequence_sha256(components),
        "ordered_sample_weights_sha256": _sequence_sha256(
            [float(value) for value in weights]
        ),
        "sample_weight_sum": math.fsum(float(value) for value in weights),
    }
    return weights, binding


def recompute_hgb_training_weights(
    train_rows: Sequence[Mapping], assignment_payload: Mapping
) -> np.ndarray:
    """Recompute weights from a complete formal assignment, never cell strings."""

    rows = _normalized_typed_rows(
        train_rows, partition=TRAIN_PARTITION, label="train_rows"
    )
    assignment = validate_hgb_cell_component_assignment(assignment_payload)
    if assignment["train_row_binding"] != _train_row_binding(rows):
        raise HGBFittedStateError("HGB assignment does not bind these train rows")
    checked = _normalize_assignment_rows(
        assignment["assignments"], expected_rows=rows
    )
    if checked != assignment["assignments"]:
        raise HGBFittedStateError("HGB assignment is not aligned to train rows")
    return _training_weights(rows, assignment)[0]


def _new_estimator():
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except Exception as exc:  # pragma: no cover - optional environment failure
        raise HGBFittedStateError(
            f"scikit-learn HGB is unavailable: {type(exc).__name__}: {exc}"
        ) from exc
    model = HistGradientBoostingRegressor(**HGB_EXPLICIT_CONFIG)
    if model.get_params(deep=False) != HGB_EFFECTIVE_SKLEARN_1_6_CONFIG:
        raise HGBFittedStateError(
            "current sklearn effective HGB defaults differ from frozen config"
        )
    return model


def _serialize_model(model, *, n_features: int) -> dict:
    baseline = np.asarray(model._baseline_prediction, dtype=np.float64).reshape(-1)
    if baseline.shape != (1,) or not np.all(np.isfinite(baseline)):
        raise HGBFittedStateError("HGB baseline prediction is not one finite scalar")
    if getattr(model, "n_trees_per_iteration_", None) != 1:
        raise HGBFittedStateError("HGB replay supports scalar regression only")
    categorical = getattr(model, "is_categorical_", None)
    if categorical is not None and np.any(np.asarray(categorical, dtype=bool)):
        raise HGBFittedStateError("categorical HGB features are prohibited")
    trees = []
    for iteration, predictors in enumerate(model._predictors):
        if len(predictors) != 1:
            raise HGBFittedStateError("HGB iteration has multiple trees")
        nodes = predictors[0].nodes
        required_fields = {
            "value",
            "count",
            "feature_idx",
            "num_threshold",
            "missing_go_to_left",
            "left",
            "right",
            "gain",
            "depth",
            "is_leaf",
            "bin_threshold",
            "is_categorical",
            "bitset_idx",
        }
        if set(nodes.dtype.names or ()) != required_fields:
            raise HGBFittedStateError("sklearn HGB node dtype changed")
        serialized_nodes = []
        for node_index, node in enumerate(nodes):
            serialized_nodes.append({
                "node_index": node_index,
                "value": _finite_float(node["value"], label="node.value"),
                "count": int(node["count"]),
                "feature_idx": int(node["feature_idx"]),
                "num_threshold": _finite_float(
                    node["num_threshold"], label="node.num_threshold"
                ),
                "missing_go_to_left": bool(node["missing_go_to_left"]),
                "left": int(node["left"]),
                "right": int(node["right"]),
                "gain": _finite_float(node["gain"], label="node.gain"),
                "depth": int(node["depth"]),
                "is_leaf": bool(node["is_leaf"]),
                "bin_threshold": int(node["bin_threshold"]),
                "is_categorical": bool(node["is_categorical"]),
                "bitset_idx": int(node["bitset_idx"]),
            })
        trees.append({"iteration": iteration, "nodes": serialized_nodes})
    state = {
        "tree_schema": HGB_TREE_SCHEMA,
        "baseline_prediction": float(baseline[0]),
        "n_features": int(n_features),
        "n_trees_per_iteration": 1,
        "n_iterations": len(trees),
        "trees": trees,
    }
    return _validate_model_state(state)


def _validate_node(value: Mapping, *, index: int, n_features: int) -> dict:
    _strict_keys(value, _NODE_KEYS, label=f"HGB node[{index}]")
    node_index = _nonnegative_int(
        value.get("node_index"), label=f"node[{index}].node_index"
    )
    if node_index != index:
        raise HGBFittedStateError("HGB node indices must be contiguous canonical order")
    is_leaf = value.get("is_leaf")
    missing_left = value.get("missing_go_to_left")
    is_categorical = value.get("is_categorical")
    for field, item in (
        ("is_leaf", is_leaf),
        ("missing_go_to_left", missing_left),
        ("is_categorical", is_categorical),
    ):
        if not isinstance(item, bool):
            raise HGBFittedStateError(f"node[{index}].{field} must be boolean")
    if is_categorical:
        raise HGBFittedStateError("categorical HGB tree nodes are prohibited")
    feature_idx = _nonnegative_int(
        value.get("feature_idx"), label=f"node[{index}].feature_idx"
    )
    if not is_leaf and feature_idx >= n_features:
        raise HGBFittedStateError("HGB node feature index is out of range")
    return {
        "node_index": node_index,
        "value": _finite_float(value.get("value"), label=f"node[{index}].value"),
        "count": _positive_int(value.get("count"), label=f"node[{index}].count"),
        "feature_idx": feature_idx,
        "num_threshold": _finite_float(
            value.get("num_threshold"), label=f"node[{index}].num_threshold"
        ),
        "missing_go_to_left": missing_left,
        "left": _nonnegative_int(value.get("left"), label=f"node[{index}].left"),
        "right": _nonnegative_int(value.get("right"), label=f"node[{index}].right"),
        "gain": _finite_float(value.get("gain"), label=f"node[{index}].gain"),
        "depth": _nonnegative_int(value.get("depth"), label=f"node[{index}].depth"),
        "is_leaf": is_leaf,
        "bin_threshold": _nonnegative_int(
            value.get("bin_threshold"), label=f"node[{index}].bin_threshold"
        ),
        "is_categorical": False,
        "bitset_idx": _nonnegative_int(
            value.get("bitset_idx"), label=f"node[{index}].bitset_idx"
        ),
    }


def _validate_tree_graph(nodes: Sequence[Mapping]) -> None:
    n_nodes = len(nodes)
    if not n_nodes:
        raise HGBFittedStateError("HGB tree cannot be empty")
    visited = set()
    active = set()

    def visit(index: int) -> None:
        if not 0 <= index < n_nodes:
            raise HGBFittedStateError("HGB child index is out of range")
        if index in active:
            raise HGBFittedStateError("HGB tree contains a cycle")
        if index in visited:
            raise HGBFittedStateError("HGB tree node has multiple parents")
        active.add(index)
        node = nodes[index]
        if not node["is_leaf"]:
            left, right = node["left"], node["right"]
            if left == right or left == index or right == index:
                raise HGBFittedStateError("HGB split has invalid children")
            visit(left)
            visit(right)
        active.remove(index)
        visited.add(index)

    visit(0)
    if visited != set(range(n_nodes)):
        raise HGBFittedStateError("HGB tree contains unreachable nodes")


def _validate_model_state(value: Mapping) -> dict:
    _strict_keys(value, _MODEL_STATE_KEYS, label="HGB model_state")
    if value.get("tree_schema") != HGB_TREE_SCHEMA:
        raise HGBFittedStateError("unsupported HGB tree schema")
    n_features = _positive_int(value.get("n_features"), label="n_features")
    n_trees = _positive_int(
        value.get("n_trees_per_iteration"), label="n_trees_per_iteration"
    )
    if n_trees != 1:
        raise HGBFittedStateError("HGB state must contain one tree per iteration")
    n_iterations = _positive_int(
        value.get("n_iterations"), label="n_iterations"
    )
    if n_iterations != HGB_EXPLICIT_CONFIG["max_iter"]:
        raise HGBFittedStateError("HGB state does not contain all frozen iterations")
    raw_trees = value.get("trees")
    if isinstance(raw_trees, (str, bytes)) or not isinstance(raw_trees, Sequence):
        raise HGBFittedStateError("HGB trees must be a sequence")
    if len(raw_trees) != n_iterations:
        raise HGBFittedStateError("HGB tree count differs from n_iterations")
    trees = []
    for index, raw_tree in enumerate(raw_trees):
        _strict_keys(raw_tree, _TREE_KEYS, label=f"HGB tree[{index}]")
        if raw_tree.get("iteration") != index:
            raise HGBFittedStateError("HGB iterations must be contiguous canonical order")
        raw_nodes = raw_tree.get("nodes")
        if isinstance(raw_nodes, (str, bytes)) or not isinstance(raw_nodes, Sequence):
            raise HGBFittedStateError("HGB tree nodes must be a sequence")
        nodes = [
            _validate_node(node, index=node_index, n_features=n_features)
            for node_index, node in enumerate(raw_nodes)
        ]
        _validate_tree_graph(nodes)
        trees.append({"iteration": index, "nodes": nodes})
    return {
        "tree_schema": HGB_TREE_SCHEMA,
        "baseline_prediction": _finite_float(
            value.get("baseline_prediction"), label="baseline_prediction"
        ),
        "n_features": n_features,
        "n_trees_per_iteration": 1,
        "n_iterations": n_iterations,
        "trees": trees,
    }


def _predict_model_state(model_state: Mapping, features: np.ndarray) -> np.ndarray:
    state = _validate_model_state(model_state)
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != state["n_features"]:
        raise HGBFittedStateError("HGB replay feature matrix has wrong shape")
    if not np.all(np.isfinite(x)):
        raise HGBFittedStateError("HGB replay feature matrix must be finite")
    prediction = np.full(len(x), state["baseline_prediction"], dtype=np.float64)
    for tree in state["trees"]:
        nodes = tree["nodes"]
        for row_index, row in enumerate(x):
            node_index = 0
            while not nodes[node_index]["is_leaf"]:
                node = nodes[node_index]
                feature = float(row[node["feature_idx"]])
                if math.isnan(feature):
                    go_left = node["missing_go_to_left"]
                else:
                    go_left = feature <= node["num_threshold"]
                node_index = node["left"] if go_left else node["right"]
            prediction[row_index] += nodes[node_index]["value"]
    if not np.all(np.isfinite(prediction)):
        raise HGBFittedStateError("HGB replay produced non-finite predictions")
    return prediction


def _validate_weight_binding(value: Mapping, *, n_rows: int) -> dict:
    _strict_keys(value, _WEIGHT_BINDING_KEYS, label="training_weight_binding")
    if value.get("training_weight_policy") != HGB_LOSS_SAMPLE_WEIGHT_POLICY:
        raise HGBFittedStateError("HGB weight policy differs from frozen policy")
    if value.get("formal_cell_definition") != FORMAL_CELL_DEFINITION:
        raise HGBFittedStateError("HGB formal-cell definition is invalid")
    expected_semantics = {
        "sample_weight_normalization": HGB_SAMPLE_WEIGHT_NORMALIZATION,
        "weight_sum_algorithm": HGB_WEIGHT_SUM_ALGORITHM,
        "replication_invariance": HGB_REPLICATION_INVARIANCE,
    }
    for field, expected in expected_semantics.items():
        if value.get(field) != expected:
            raise HGBFittedStateError(
                f"HGB weight binding {field} differs from frozen semantics"
            )
    n_cells = _positive_int(value.get("n_formal_cells"), label="n_formal_cells")
    if n_cells > n_rows:
        raise HGBFittedStateError("HGB formal-cell count exceeds training rows")
    for field in (
        "ordered_formal_cell_sha256s_sha256",
        "ordered_source_component_ids_sha256",
        "ordered_sample_weights_sha256",
    ):
        _digest(value.get(field), label=field)
    weight_sum = _finite_float(value.get("sample_weight_sum"), label="sample_weight_sum")
    tolerance = 8.0 * np.finfo(np.float64).eps * max(1.0, float(n_rows))
    if not math.isclose(
        weight_sum, float(n_rows), rel_tol=0.0, abs_tol=tolerance
    ):
        raise HGBFittedStateError("HGB sample weights do not sum to training rows")
    return dict(value)


def _validate_assignment_binding(value: Mapping) -> dict:
    _strict_keys(value, _ASSIGNMENT_BINDING_KEYS, label="assignment binding")
    if value.get("schema_version") != HGB_ASSIGNMENT_SCHEMA:
        raise HGBFittedStateError("assignment binding schema is invalid")
    n_typed = _positive_int(
        value.get("n_typed_cells"), label="assignment binding.n_typed_cells"
    )
    n_formal = _positive_int(
        value.get("n_formal_cells"), label="assignment binding.n_formal_cells"
    )
    if n_typed != n_formal:
        raise HGBFittedStateError(
            "assignment binding typed/formal cell counts must be equal"
        )
    for field in _ASSIGNMENT_BINDING_KEYS - {
        "schema_version",
        "n_typed_cells",
        "n_formal_cells",
    }:
        _digest(value.get(field), label=f"assignment binding.{field}")
    return dict(value)


def _validate_environment_binding(value: Mapping) -> dict:
    _strict_keys(value, _ENVIRONMENT_BINDING_KEYS, label="environment binding")
    if value.get("schema_version") != HGB_ENVIRONMENT_SCHEMA:
        raise HGBFittedStateError("environment binding schema is invalid")
    _version_text(value.get("sklearn_version"), label="environment sklearn_version")
    _version_text(
        value.get("threadpoolctl_version"),
        label="environment threadpoolctl_version",
    )
    if value.get("openmp_thread_limit") != 1:
        raise HGBFittedStateError("environment OpenMP thread limit is invalid")
    if value.get("openmp_thread_policy") != OPENMP_THREAD_POLICY:
        raise HGBFittedStateError("environment OpenMP policy is invalid")
    for field in _ENVIRONMENT_BINDING_KEYS - {
        "schema_version",
        "sklearn_version",
        "threadpoolctl_version",
        "openmp_thread_limit",
        "openmp_thread_policy",
    }:
        _digest(value.get(field), label=f"environment binding.{field}")
    return dict(value)


def _validate_condition_vocab(value: object) -> list[list[str]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise HGBFittedStateError("condition_vocab must have three ordered blocks")
    result = []
    for index, levels in enumerate(value):
        if isinstance(levels, (str, bytes)) or not isinstance(levels, Sequence) or not levels:
            raise HGBFittedStateError("condition_vocab blocks must be non-empty")
        parsed = [
            _identifier(level, label=f"condition_vocab[{index}][{j}]")
            for j, level in enumerate(levels)
        ]
        if parsed != sorted(set(parsed)):
            raise HGBFittedStateError("condition_vocab levels must be sorted unique")
        result.append(parsed)
    return result


def validate_hgb_fitted_state(payload: Mapping) -> dict:
    if not isinstance(payload, Mapping):
        raise HGBFittedStateError("HGB fitted state must be an object")
    if payload.get("schema_version") != HGB_FITTED_STATE_SCHEMA:
        raise HGBFittedStateError("unsupported HGB fitted-state schema")
    _strict_keys(payload, _STATE_KEYS, label="HGB fitted state")
    expected = {
        "artifact_role": HGB_FITTED_STATE_ROLE,
        "status": HGB_FITTED_STATE_STATUS,
        "method_id": HGB_METHOD_ID,
        "information_set": HGB_INFORMATION_SET,
        "feature_schema": PAIRED_KMER_FEATURE_SCHEMA,
        "fixture_namespace_status": FIXTURE_NAMESPACE_STATUS,
        "exact_eight_complete": False,
        "training_ready": False,
        "readiness_blocker": READINESS_BLOCKER,
    }
    for field, expected_value in expected.items():
        if payload.get(field) != expected_value:
            raise HGBFittedStateError(f"HGB fitted state {field} is invalid")
    _identifier(payload.get("estimand_id"), label="estimand_id")
    _hgb_information_set_fingerprint(
        payload.get("information_set_fingerprint"),
        label="information_set_fingerprint",
    )
    _digest(
        payload.get("train_development_outcomes_sha256"),
        label="train_development_outcomes_sha256",
    )
    authorities = _non_data_authority_hashes(payload.get("non_data_authority_hashes"))
    row_binding = _validate_row_binding(payload.get("train_row_binding"), label="train_row_binding")
    assignment_binding = _validate_assignment_binding(
        payload.get("cell_component_assignment_binding")
    )
    environment_binding = _validate_environment_binding(
        payload.get("environment_qualification_binding")
    )
    vocab = _validate_condition_vocab(payload.get("condition_vocab"))
    fit_config = _validate_fit_config(payload.get("fit_config"))
    if payload.get("fit_config_sha256") != _canonical_sha256(fit_config):
        raise HGBFittedStateError("HGB fit_config SHA mismatch")
    weight_binding = _validate_weight_binding(
        payload.get("training_weight_binding"), n_rows=row_binding["n_rows"]
    )
    if weight_binding["n_formal_cells"] != assignment_binding["n_formal_cells"]:
        raise HGBFittedStateError(
            "HGB weight binding and assignment bind different formal-cell counts"
        )
    model_state = _validate_model_state(payload.get("model_state"))
    _boundaries(payload, label="HGB fitted state")
    return {
        **dict(payload),
        "non_data_authority_hashes": authorities,
        "train_row_binding": row_binding,
        "cell_component_assignment_binding": assignment_binding,
        "environment_qualification_binding": environment_binding,
        "condition_vocab": vocab,
        "fit_config": fit_config,
        "training_weight_binding": weight_binding,
        "model_state": model_state,
    }


def canonical_hgb_fitted_state_bytes(payload: Mapping) -> bytes:
    try:
        return canonical_json_bytes(validate_hgb_fitted_state(payload))
    except ArtifactIOError as exc:
        raise HGBFittedStateError(str(exc)) from exc


def _assignment_expected_kwargs(
    train_rows: Sequence[Mapping],
    *,
    information_set_fingerprint: str,
    train_development_outcomes_sha256: str,
    non_data_authority_hashes: Mapping,
    source_component_assignment_source_sha256: str,
    source_component_assignment_projection_sha256: str,
    formal_cell_assignment_source_sha256: str,
    formal_cell_assignment_projection_sha256: str,
) -> dict:
    return {
        "expected_train_rows": train_rows,
        "expected_information_set_fingerprint": information_set_fingerprint,
        "expected_train_development_outcomes_sha256": train_development_outcomes_sha256,
        "expected_non_data_authority_hashes": non_data_authority_hashes,
        "expected_source_component_assignment_source_sha256": (
            source_component_assignment_source_sha256
        ),
        "expected_source_component_assignment_projection_sha256": (
            source_component_assignment_projection_sha256
        ),
        "expected_formal_cell_assignment_source_sha256": (
            formal_cell_assignment_source_sha256
        ),
        "expected_formal_cell_assignment_projection_sha256": (
            formal_cell_assignment_projection_sha256
        ),
    }


def fit_hgb_fitted_state(
    train_rows: Sequence[Mapping],
    *,
    cell_component_assignment_path: Path,
    cell_component_assignment_sha256: str,
    environment_qualification_path: Path,
    environment_qualification_sha256: str,
    information_set_fingerprint: str,
    train_development_outcomes_sha256: str,
    non_data_authority_hashes: Mapping,
    source_component_assignment_source_sha256: str,
    source_component_assignment_projection_sha256: str,
    formal_cell_assignment_source_sha256: str,
    formal_cell_assignment_projection_sha256: str,
) -> dict:
    """Fit a positive fixture-namespace state from local exact dependencies.

    The namespace checks below are test-fixture conventions only.  They do not
    establish data provenance, ordinary-data authority, or execution trust.
    """

    rows = _normalized_typed_rows(
        train_rows, partition=TRAIN_PARTITION, label="train_rows"
    )
    if not rows[0]["estimand_id"].startswith(FIXTURE_NAMESPACE_PREFIX):
        raise HGBFittedStateError(
            "current HGB schema requires the positive fixture namespace; "
            "that namespace is not provenance or a trust root"
        )
    if len(rows) < MINIMUM_TRAIN_ROWS:
        raise HGBFittedStateError(
            f"HGB requires at least {MINIMUM_TRAIN_ROWS} training rows"
        )
    assignment = load_hgb_cell_component_assignment_exact(
        cell_component_assignment_path,
        cell_component_assignment_sha256,
        **_assignment_expected_kwargs(
            rows,
            information_set_fingerprint=information_set_fingerprint,
            train_development_outcomes_sha256=train_development_outcomes_sha256,
            non_data_authority_hashes=non_data_authority_hashes,
            source_component_assignment_source_sha256=(
                source_component_assignment_source_sha256
            ),
            source_component_assignment_projection_sha256=(
                source_component_assignment_projection_sha256
            ),
            formal_cell_assignment_source_sha256=formal_cell_assignment_source_sha256,
            formal_cell_assignment_projection_sha256=(
                formal_cell_assignment_projection_sha256
            ),
        ),
    )
    environment = load_hgb_environment_qualification_exact(
        environment_qualification_path, environment_qualification_sha256
    )
    if any(
        not item["formal_cell"]["study_id"].startswith(
            FIXTURE_NAMESPACE_PREFIX
        )
        for item in assignment["assignments"]
    ):
        raise HGBFittedStateError(
            "current HGB schema accepts only the positive fixture study "
            "namespace, which is not provenance"
        )
    component_by_id = {
        item["row_id"]: item["source_component_id"]
        for item in assignment["assignments"]
    }
    flat = _flat_baseline_rows(rows, source_component_by_row_id=component_by_id)
    cells = [
        item["formal_cell_sha256"] for item in assignment["assignments"]
    ]
    weights, weight_binding = _training_weights(rows, assignment)
    condition_vocab = _condition_vocab(flat)
    features = _feature_matrix(flat, paired_kmer_features, condition_vocab)
    target = np.asarray([row["training_target"] for row in rows], dtype=np.float64)
    model = _new_estimator()
    try:
        from threadpoolctl import threadpool_limits

        # A single OpenMP thread is a precautionary local fixture policy.
        # Multithread stability, crash behavior, and real-run qualification are
        # deliberately not asserted by this artifact.
        with threadpool_limits(limits=1, user_api="openmp"):
            model.fit(features, target, sample_weight=weights)
            sklearn_prediction = np.asarray(
                model.predict(features), dtype=np.float64
            )
    except Exception as exc:
        raise HGBFittedStateError(
            f"HGB fit failed: {type(exc).__name__}: {exc}"
        ) from exc
    model_state = _serialize_model(model, n_features=features.shape[1])
    replay_prediction = _predict_model_state(model_state, features)
    if not np.array_equal(sklearn_prediction, replay_prediction):
        raise HGBFittedStateError(
            "numeric JSON tree replay differs exactly from sklearn predictions"
        )

    # Independently exercise the public baseline path on the same exact rows.
    with threadpool_limits(limits=1, user_api="openmp"):
        baseline = fit_optional_hist_gradient_boosting(
            flat,
            flat,
            execution_authorized=True,
            minimum_train_rows=MINIMUM_TRAIN_ROWS,
            training_cell_ids=cells,
        )
    if baseline.status != "RUN" or not np.array_equal(
        sklearn_prediction, np.asarray(baseline.predictions, dtype=np.float64)
    ):
        raise HGBFittedStateError(
            "serialized HGB fit differs from fit_optional_hist_gradient_boosting"
        )

    fit_config = _fit_config()
    payload = {
        "schema_version": HGB_FITTED_STATE_SCHEMA,
        "artifact_role": HGB_FITTED_STATE_ROLE,
        "status": HGB_FITTED_STATE_STATUS,
        "method_id": HGB_METHOD_ID,
        "information_set": HGB_INFORMATION_SET,
        "information_set_fingerprint": _hgb_information_set_fingerprint(
            information_set_fingerprint, label="information_set_fingerprint"
        ),
        "estimand_id": rows[0]["estimand_id"],
        "train_development_outcomes_sha256": _digest(
            train_development_outcomes_sha256,
            label="train_development_outcomes_sha256",
        ),
        "non_data_authority_hashes": _non_data_authority_hashes(
            non_data_authority_hashes
        ),
        "train_row_binding": _train_row_binding(rows),
        "cell_component_assignment_binding": {
            "schema_version": HGB_ASSIGNMENT_SCHEMA,
            "sha256": _digest(
                cell_component_assignment_sha256,
                label="cell_component_assignment_sha256",
            ),
            "source_component_assignment_source_sha256": assignment[
                "source_component_assignment_source_sha256"
            ],
            "source_component_assignment_projection_sha256": assignment[
                "source_component_assignment_projection_sha256"
            ],
            "formal_cell_assignment_source_sha256": assignment[
                "formal_cell_assignment_source_sha256"
            ],
            "formal_cell_assignment_projection_sha256": assignment[
                "formal_cell_assignment_projection_sha256"
            ],
            "n_typed_cells": assignment["n_typed_cells"],
            "n_formal_cells": assignment["n_formal_cells"],
            "typed_cell_to_formal_cell_mapping_sha256": assignment[
                "typed_cell_to_formal_cell_mapping_sha256"
            ],
        },
        "environment_qualification_binding": {
            "schema_version": HGB_ENVIRONMENT_SCHEMA,
            "sha256": _digest(
                environment_qualification_sha256,
                label="environment_qualification_sha256",
            ),
            "sklearn_version": environment["sklearn_version"],
            "threadpoolctl_version": environment["threadpoolctl_version"],
            "openmp_thread_limit": environment["openmp_thread_limit"],
            "openmp_thread_policy": environment["openmp_thread_policy"],
            "sklearn_gradient_boosting_code_sha256": environment[
                "sklearn_gradient_boosting_code_sha256"
            ],
            "sklearn_tree_predictor_code_sha256": environment[
                "sklearn_tree_predictor_code_sha256"
            ],
            "sklearn_compiled_predictor_sha256": environment[
                "sklearn_compiled_predictor_sha256"
            ],
            "threadpoolctl_code_sha256": environment[
                "threadpoolctl_code_sha256"
            ],
            "r3x_hgb_fitted_state_code_sha256": environment[
                "r3x_hgb_fitted_state_code_sha256"
            ],
        },
        "feature_schema": PAIRED_KMER_FEATURE_SCHEMA,
        "condition_vocab": [list(values) for values in condition_vocab],
        "fit_config": fit_config,
        "fit_config_sha256": _canonical_sha256(fit_config),
        "training_weight_binding": weight_binding,
        "model_state": model_state,
        "fixture_namespace_status": FIXTURE_NAMESPACE_STATUS,
        "exact_eight_complete": False,
        "training_ready": False,
        "readiness_blocker": READINESS_BLOCKER,
        **_boundary_payload(),
    }
    return validate_hgb_fitted_state(payload)


def _recompute_hgb_predictions_untrusted_state_for_tests_only(
    fitted_state: Mapping, development_rows: Sequence[Mapping]
) -> np.ndarray:
    """Traverse a schema-valid raw state without proving that it was fitted.

    This helper is deliberately private and exists for numeric-replay unit
    tests.  Evidence code must use ``load_and_recompute_hgb_predictions_exact``
    so an arbitrary schema-valid tree cannot bypass the exact training refit.
    """

    state = validate_hgb_fitted_state(fitted_state)
    rows = _normalized_typed_rows(
        development_rows,
        partition=DEVELOPMENT_PARTITION,
        label="development_rows",
    )
    if any(row["estimand_id"] != state["estimand_id"] for row in rows):
        raise HGBFittedStateError("development rows use a different estimand")
    flat = _flat_baseline_rows(rows)
    vocab = tuple(tuple(levels) for levels in state["condition_vocab"])
    features = _feature_matrix(flat, paired_kmer_features, vocab)
    return _predict_model_state(state["model_state"], features)


def _verify_fitted_state_bindings(
    payload: Mapping,
    *,
    expected_train_rows: Sequence[Mapping],
    expected_cell_component_assignment_path: Path,
    expected_cell_component_assignment_sha256: str,
    expected_environment_qualification_path: Path,
    expected_environment_qualification_sha256: str,
    expected_information_set_fingerprint: str,
    expected_train_development_outcomes_sha256: str,
    expected_non_data_authority_hashes: Mapping,
    expected_source_component_assignment_source_sha256: str,
    expected_source_component_assignment_projection_sha256: str,
    expected_formal_cell_assignment_source_sha256: str,
    expected_formal_cell_assignment_projection_sha256: str,
    refit: bool,
) -> None:
    rows = _normalized_typed_rows(
        expected_train_rows,
        partition=TRAIN_PARTITION,
        label="expected_train_rows",
    )
    simple_expected = {
        "information_set_fingerprint": _hgb_information_set_fingerprint(
            expected_information_set_fingerprint,
            label="expected_information_set_fingerprint",
        ),
        "train_development_outcomes_sha256": _digest(
            expected_train_development_outcomes_sha256,
            label="expected_train_development_outcomes_sha256",
        ),
        "non_data_authority_hashes": _non_data_authority_hashes(
            expected_non_data_authority_hashes
        ),
        "train_row_binding": _train_row_binding(rows),
    }
    for field, value in simple_expected.items():
        if payload[field] != value:
            raise HGBFittedStateError(f"HGB fitted-state {field} binding mismatch")
    assignment_sha = _digest(
        expected_cell_component_assignment_sha256,
        label="expected_cell_component_assignment_sha256",
    )
    environment_sha = _digest(
        expected_environment_qualification_sha256,
        label="expected_environment_qualification_sha256",
    )
    if payload["cell_component_assignment_binding"]["sha256"] != assignment_sha:
        raise HGBFittedStateError("HGB fitted-state assignment SHA mismatch")
    if payload["environment_qualification_binding"]["sha256"] != environment_sha:
        raise HGBFittedStateError("HGB fitted-state environment SHA mismatch")
    if refit:
        expected = fit_hgb_fitted_state(
            rows,
            cell_component_assignment_path=expected_cell_component_assignment_path,
            cell_component_assignment_sha256=assignment_sha,
            environment_qualification_path=expected_environment_qualification_path,
            environment_qualification_sha256=environment_sha,
            information_set_fingerprint=expected_information_set_fingerprint,
            train_development_outcomes_sha256=(
                expected_train_development_outcomes_sha256
            ),
            non_data_authority_hashes=expected_non_data_authority_hashes,
            source_component_assignment_source_sha256=(
                expected_source_component_assignment_source_sha256
            ),
            source_component_assignment_projection_sha256=(
                expected_source_component_assignment_projection_sha256
            ),
            formal_cell_assignment_source_sha256=(
                expected_formal_cell_assignment_source_sha256
            ),
            formal_cell_assignment_projection_sha256=(
                expected_formal_cell_assignment_projection_sha256
            ),
        )
        if canonical_hgb_fitted_state_bytes(payload) != canonical_hgb_fitted_state_bytes(expected):
            raise HGBFittedStateError(
                "loaded HGB state differs from exact training-row refit"
            )


def write_hgb_fitted_state_exclusive(
    path: Path,
    payload: Mapping,
    **expected_bindings,
) -> str:
    output = _assert_development_path(path, label="HGB fitted-state output")
    if output.exists() or output.is_symlink():
        raise HGBFittedStateError(f"artifact output already exists: {output}")
    validated = validate_hgb_fitted_state(payload)
    _verify_fitted_state_bindings(validated, refit=True, **expected_bindings)
    try:
        return atomic_write_json_exclusive(
            output, validated, validator=validate_hgb_fitted_state
        )
    except ArtifactIOError as exc:
        raise HGBFittedStateError(str(exc)) from exc


def load_hgb_fitted_state_exact(
    path: Path,
    expected_sha256: str,
    **expected_bindings,
) -> dict:
    input_path = _assert_development_path(path, label="HGB fitted-state input")
    digest = _digest(expected_sha256, label="expected fitted-state sha256")
    try:
        raw = read_exact_regular_file(input_path, digest)
    except ArtifactIOError as exc:
        raise HGBFittedStateError(str(exc)) from exc
    payload = validate_hgb_fitted_state(
        _parse_json_object(raw, label="HGB fitted state")
    )
    if canonical_hgb_fitted_state_bytes(payload) != raw:
        raise HGBFittedStateError("HGB fitted-state bytes are not canonical")
    _verify_fitted_state_bindings(payload, refit=True, **expected_bindings)
    return payload


def load_and_recompute_hgb_predictions_exact(
    fitted_state_path: Path,
    fitted_state_sha256: str,
    development_rows: Sequence[Mapping],
    **expected_bindings,
) -> np.ndarray:
    """Evidence-facing prediction after exact byte load and training refit."""

    loaded = load_hgb_fitted_state_exact(
        fitted_state_path,
        fitted_state_sha256,
        **expected_bindings,
    )
    return _recompute_hgb_predictions_untrusted_state_for_tests_only(
        loaded, development_rows
    )


__all__ = [
    "EXACT_EIGHT_COMPLETE",
    "TRAINING_READY",
    "READINESS_BLOCKER",
    "FIXTURE_NAMESPACE_PREFIX",
    "FIXTURE_NAMESPACE_STATUS",
    "FORMAL_CELL_DEFINITION",
    "HGB_LOSS_SAMPLE_WEIGHT_POLICY",
    "HGB_INFORMATION_SET_FINGERPRINT",
    "HGB_ASSIGNMENT_SCHEMA",
    "HGB_ENVIRONMENT_SCHEMA",
    "HGB_FITTED_STATE_SCHEMA",
    "HGB_METHOD_ID",
    "HGBFittedStateError",
    "build_hgb_cell_component_assignment",
    "canonical_hgb_cell_component_assignment_bytes",
    "canonical_hgb_environment_qualification_bytes",
    "canonical_hgb_fitted_state_bytes",
    "fit_hgb_fitted_state",
    "load_hgb_cell_component_assignment_exact",
    "load_and_recompute_hgb_predictions_exact",
    "load_hgb_environment_qualification_exact",
    "load_hgb_fitted_state_exact",
    "observe_hgb_environment_qualification",
    "recompute_hgb_training_weights",
    "validate_hgb_cell_component_assignment",
    "validate_hgb_environment_qualification",
    "validate_hgb_fitted_state",
    "write_hgb_cell_component_assignment_exclusive",
    "write_hgb_environment_qualification_exclusive",
    "write_hgb_fitted_state_exclusive",
]
