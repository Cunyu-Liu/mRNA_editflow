"""Exact-replayable train/development neural fitted states for R3X.

This module is deliberately independent of the historical ``run_ordinary``
runner.  It consumes only the positive, typed predictor-input schema, a
separately bound train/development source-component assignment, and the public
R3X model/training kernels.  It never accepts an internal-test or sealed row.

The implementation is a development evidence kernel, not an authority issuer.
Callers remain responsible for exact-loading independently issued ordinary
train/development authorities before invoking it.  A fitted state cannot
unlock an internal-test outcome, a sealed dataset, Route B, or a publication
claim.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import torch

from .artifact_graph import ArtifactGraphError, prediction_jsonl_bytes
from .artifact_io import (
    ArtifactIOError,
    atomic_publish_bytes,
    atomic_write_json_exclusive,
    canonical_json_bytes,
    read_exact_regular_file,
)
from .information_sets import InformationSetError, information_set_fingerprint
from .metrics import MetricProtocolError, component_balanced_cell_metrics
from .models import (
    ModelConfig,
    derive_replay_edit_tensors,
    derive_wrong_local_feature_positions,
)
from .neural_method_registry import (
    CONDITION_UNK_TRAINING_VARIANT_ID,
    POSITION_AGNOSTIC_TRAINING_VARIANT_ID,
    PRIMARY_TRAINING_VARIANT_ID,
    REGISTRY_SCHEMA,
    WRONG_LOCAL_TRAINING_VARIANT_ID,
    NeuralMethodRegistryError,
    canonical_registry_payload,
    compile_registered_development_model,
    registry_sha256,
)
from .predictor_provenance import (
    DEVELOPMENT_PARTITION,
    ENSEMBLE_RULE,
    ENSEMBLE_VARIANCE_FORMULA,
    FROZEN_NEURAL_SEEDS,
    NON_DATA_AUTHORITY_HASH_KEYS,
    TRAIN_PARTITION,
    VARIANCE_FLOOR,
    PredictorProvenanceError,
    normalize_predictor_input_rows,
    predictor_input_jsonl_bytes,
)
from .protocol import ProtocolHoldError, assert_ordinary_path
from .train import (
    SIGN_GATE,
    TrainingProtocolError,
    apply_condition_unknown_dropout,
    balanced_epoch_indices,
    effect_loss,
    select_checkpoint,
)


NEURAL_DEVELOPMENT_STATE_SCHEMA = "r3x_neural_development_fitted_state_v1"
NEURAL_DEVELOPMENT_STATE_ROLE = "ORDINARY_DEVELOPMENT_EXACT_NEURAL_FITTED_STATE"
NEURAL_DEVELOPMENT_STATE_STATUS = "COMPLETE_EXACT_REPLAYABLE_DEVELOPMENT_FIT"
NEURAL_COMPONENT_ASSIGNMENT_SCHEMA = (
    "r3x_train_development_source_component_assignment_v1"
)
NEURAL_COMPONENT_ASSIGNMENT_ROLE = (
    "ORDINARY_TRAIN_DEVELOPMENT_SOURCE_COMPONENT_ASSIGNMENT"
)
NEURAL_COMPONENT_ASSIGNMENT_STATUS = (
    "COMPLETE_EXACT_TRAIN_DEVELOPMENT_SOURCE_COMPONENT_ASSIGNMENT"
)
NEURAL_ENSEMBLE_SCHEMA = "r3x_neural_development_five_seed_ensemble_v1"
NEURAL_ENSEMBLE_ROLE = "ORDINARY_DEVELOPMENT_FIXED_FIVE_SEED_ENSEMBLE"
NEURAL_ENSEMBLE_STATUS = "COMPLETE_EXACT_DEVELOPMENT_PREDICTION_ENSEMBLE"

PAIRED_INFORMATION_SET = "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR"
ALLOWED_EDIT_COUNTS = frozenset({1, 3, 5})
ACTION_POLICY = "EQUAL_LENGTH_SUBSTITUTION_ONLY_EDIT_COUNT_1_3_5_V1"
SAMPLER_POLICY = (
    "DETERMINISTIC_EQUAL_CELL_ROUND_ROBIN_GLOBAL_COMPONENT_WITHIN_CELL_V1"
)
SELECTION_METRIC_POLICY = (
    "DEV_CELL_MACRO_COMPONENT_BALANCED_SIGN_WEIGHTED_RANK_DIAGNOSTIC_NLL_V1"
)
ROW_ORDER_POLICY = "CANONICAL_ASCENDING_ROW_ID"
DETERMINISM_POLICY = "CPU_FLOAT32_SINGLE_THREAD_DETERMINISTIC_ALGORITHMS_V1"
RESOURCE_PREFLIGHT_POLICY = (
    "BEFORE_MODEL_CONSTRUCTION_ROWS_TOKENS_FORWARD_AND_CONSERVATIVE_PARAMETER_BYTES_V1"
)
PARAMETER_MATCH_POLICY = "REGISTRY_EXACT_ACTUAL_CONFIG_MAXIMUM_GAP_0P05_V2"
# Backward-compatible public alias for the six original methods.  New controls
# bind their exact variant through the versioned method registry.
TRAINING_VARIANT_ID = PRIMARY_TRAINING_VARIANT_ID
STATE_TENSOR_DIGEST_POLICY = "SORTED_KEY_DTYPE_SHAPE_LITTLE_ENDIAN_RAW_BYTES_V1"
EXACT_RECOMPUTE_POLICY = (
    "EXACT_LOAD_REGISTERED_CLASS_STATE_THEN_CANONICAL_DEVELOPMENT_PREDICTION_BYTES_V1"
)
BOUNDARY = {
    "ordinary_only": True,
    "internal_test_access": False,
    "sealed_access": False,
    "downstream_unlock": False,
    "execution_authorized": False,
    "independent_authority_verified": False,
    "training_ready": False,
    "publication_evidence": False,
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_PROTECTED_TOKENS = ("internal_test", "gse246381")
_PROTECTED_PATH_PARTS = {"test", "internal_test", "sealed", "restricted"}
_BASE_TO_TOKEN = {"A": 1, "C": 2, "G": 3, "U": 4}

_ROW_BINDING_KEYS = {
    "n_rows",
    "ordered_row_ids_sha256",
    "row_set_sha256",
    "content_sha256",
}
_ASSIGNMENT_ROW_KEYS = {"row_id", "partition", "source_component_id"}
_ASSIGNMENT_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "information_set",
    "information_set_fingerprint",
    "train_development_outcomes_sha256",
    "non_data_authority_hashes",
    "source_component_split_manifest_source_sha256",
    "source_component_split_manifest_projection_sha256",
    "partition_row_bindings",
    "assignments",
    "assignment_sha256",
} | set(BOUNDARY)
_FIT_CONFIG_KEYS = {
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "huber_weight",
    "nll_weight",
    "sign_weight",
    "sign_temperature",
    "gradient_clip_norm",
    "condition_unknown_dropout",
    "max_rows_per_fit",
    "max_sequence_tokens_per_batch",
    "max_forward_equivalents_per_fit",
    "max_total_forward_equivalents_five_seeds",
    "max_peak_parameter_bytes_per_fit",
    "max_wall_time_seconds_per_fit",
    "device",
}
_STATE_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "run_start_sha256",
    "development_activation_sha256",
    "train_development_outcomes_sha256",
    "non_data_authority_hashes",
    "estimand_id",
    "method_id",
    "training_variant_id",
    "architecture_id",
    "registry_schema",
    "registry_sha256",
    "compiled_binding",
    "compiled_binding_sha256",
    "kernel_code_sha256",
    "models_code_sha256",
    "train_code_sha256",
    "model_config",
    "model_config_sha256",
    "fit_config",
    "fit_config_sha256",
    "information_set",
    "information_set_fingerprint",
    "seed",
    "partition_row_bindings",
    "component_assignment_binding",
    "training_design_sha256",
    "development_target_sha256",
    "action_policy",
    "allowed_edit_counts",
    "action_replay_sha256",
    "condition_vocab",
    "condition_vocab_sha256",
    "row_order_policy",
    "sampler_policy",
    "sampler_epoch_manifests",
    "selection_metric_policy",
    "checkpoint_selection",
    "checkpoint_selection_sha256",
    "selected_epoch",
    "development_predictions",
    "development_prediction_sha256",
    "resource_preflight_policy",
    "resource_preflight",
    "parameter_match_policy",
    "determinism_policy",
    "state_tensor_digest_policy",
    "state_shapes",
    "state_dtypes",
    "state_tensor_sha256",
    "state_dict",
    "exact_recompute_policy",
} | set(BOUNDARY)
_ENSEMBLE_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "method_id",
    "training_variant_id",
    "architecture_id",
    "registry_sha256",
    "model_config_sha256",
    "fit_config_sha256",
    "information_set",
    "information_set_fingerprint",
    "estimand_id",
    "partition_row_bindings",
    "seeds",
    "seed_state_tensor_sha256",
    "seed_checkpoint_selection_sha256",
    "seed_development_prediction_sha256",
    "ensemble_rule",
    "ensemble_variance_formula",
    "variance_floor",
    "ensemble_predictions",
    "ensemble_prediction_sha256",
} | set(BOUNDARY)


class NeuralDevelopmentStateError(RuntimeError):
    """Fail-closed neural development fit, state, or replay error."""


@dataclass(frozen=True, slots=True)
class NeuralDevelopmentFitConfig:
    """Fully frozen fit and resource configuration.

    Resource limits are part of the scientific configuration, not optional
    operational hints.  Deterministic bounds are checked before the registered
    model compiler is called; the wall-time limit is enforced during fitting.
    """

    epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    huber_weight: float = 1.0
    nll_weight: float = 0.25
    sign_weight: float = 0.25
    sign_temperature: float = 1.0
    gradient_clip_norm: float = 5.0
    condition_unknown_dropout: float = 0.15
    max_rows_per_fit: int = 100_000
    max_sequence_tokens_per_batch: int = 1_000_000
    max_forward_equivalents_per_fit: int = 100_000_000
    max_total_forward_equivalents_five_seeds: int = 500_000_000
    max_peak_parameter_bytes_per_fit: int = 200_000_000
    max_wall_time_seconds_per_fit: float = 86_400.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        _validate_fit_config(asdict(self))


def _strict_keys(value: Mapping, expected: set[str], *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise NeuralDevelopmentStateError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise NeuralDevelopmentStateError(
            f"{label} keys differ from strict schema: {sorted(actual ^ expected)}"
        )


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise NeuralDevelopmentStateError(f"{label} must be 64 lowercase hex")
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise NeuralDevelopmentStateError(f"{label} must be a canonical identifier")
    lowered = value.lower()
    if (
        any(token in lowered for token in _PROTECTED_TOKENS)
        or lowered.startswith("sealed")
        or lowered.startswith("restricted")
    ):
        raise NeuralDevelopmentStateError(f"{label} enters a protected namespace")
    return value


def _feature_text(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise NeuralDevelopmentStateError(
            f"{label} must be canonical non-empty feature text"
        )
    lowered = value.lower()
    if (
        any(token in lowered for token in _PROTECTED_TOKENS)
        or lowered.startswith("sealed")
        or lowered.startswith("restricted")
    ):
        raise NeuralDevelopmentStateError(f"{label} enters a protected namespace")
    return value


def _finite(value: object, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise NeuralDevelopmentStateError(f"{label} must be finite numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise NeuralDevelopmentStateError(f"{label} must be finite and positive")
    return result


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise NeuralDevelopmentStateError(f"{label} must be a positive integer")
    return value


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NeuralDevelopmentStateError(
            f"value is not canonical finite JSON: {exc}"
        ) from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _boundaries(value: Mapping, *, label: str) -> None:
    for field, expected in BOUNDARY.items():
        if value.get(field) is not expected:
            raise NeuralDevelopmentStateError(
                f"{label}.{field} violates the ordinary development boundary"
            )


def _non_data_authority_hashes(value: Mapping) -> dict[str, str]:
    _strict_keys(
        value,
        set(NON_DATA_AUTHORITY_HASH_KEYS),
        label="non_data_authority_hashes",
    )
    return {
        name: _digest(value[name], label=f"non_data_authority_hashes.{name}")
        for name in NON_DATA_AUTHORITY_HASH_KEYS
    }


def _assert_development_path(path: Path, *, label: str) -> Path:
    candidate = Path(path)
    parts = [part.lower() for part in candidate.parts]
    if "development" not in parts:
        raise NeuralDevelopmentStateError(
            f"{label} must be beneath a literal development namespace"
        )
    if any(
        part in _PROTECTED_PATH_PARTS
        or "internal_test" in part
        or "gse246381" in part
        or part.startswith("sealed")
        or part.startswith("restricted")
        for part in parts
    ):
        raise NeuralDevelopmentStateError(f"{label} enters a protected namespace")
    try:
        assert_ordinary_path(candidate, purpose=label)
    except ProtocolHoldError as exc:
        raise NeuralDevelopmentStateError(str(exc)) from exc
    return candidate


def _normalize_rows(
    rows: Sequence[Mapping], *, partition: str, label: str
) -> list[dict]:
    try:
        normalized = normalize_predictor_input_rows(rows)
    except PredictorProvenanceError as exc:
        raise NeuralDevelopmentStateError(str(exc)) from exc
    if any(row["partition"] != partition for row in normalized):
        raise NeuralDevelopmentStateError(
            f"{label} must contain only {partition} predictor rows"
        )
    estimands = {row["estimand_id"] for row in normalized}
    if len(estimands) != 1:
        raise NeuralDevelopmentStateError(
            f"{label} must contain exactly one primary estimand"
        )
    for row in normalized:
        _identifier(row["estimand_id"], label=f"{label}.estimand_id")
        _identifier(row["cell_id"], label=f"{label}.cell_id")
        for field in ("endpoint", "publication_context_id", "region"):
            _feature_text(
                row["predictor_input"][field],
                label=f"{label}.predictor_input.{field}",
            )
    return normalized


def _row_binding(rows: Sequence[Mapping]) -> dict:
    row_ids = [row["row_id"] for row in rows]
    try:
        content = predictor_input_jsonl_bytes(rows)
    except PredictorProvenanceError as exc:
        raise NeuralDevelopmentStateError(str(exc)) from exc
    return {
        "n_rows": len(rows),
        "ordered_row_ids_sha256": _canonical_sha256(row_ids),
        "row_set_sha256": _canonical_sha256(sorted(row_ids)),
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


def _partition_row_bindings(train_rows: Sequence[Mapping], dev_rows: Sequence[Mapping]) -> dict:
    return {
        "train": _row_binding(train_rows),
        "development": _row_binding(dev_rows),
    }


def _validate_partition_row_bindings(value: Mapping) -> dict:
    _strict_keys(value, {"train", "development"}, label="partition_row_bindings")
    result = {
        key: _validate_row_binding(value[key], label=f"partition_row_bindings.{key}")
        for key in ("train", "development")
    }
    if result["train"]["row_set_sha256"] == result["development"]["row_set_sha256"]:
        raise NeuralDevelopmentStateError(
            "train and development row sets cannot alias"
        )
    return result


def _normalize_assignment_rows(
    values: Sequence[Mapping],
    *,
    expected_row_ids: Sequence[str],
    expected_partitions: Mapping[str, str] | None = None,
) -> list[dict]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise NeuralDevelopmentStateError("component assignments must be a sequence")
    normalized = []
    for index, value in enumerate(values):
        _strict_keys(value, _ASSIGNMENT_ROW_KEYS, label=f"assignment {index}")
        partition = value.get("partition")
        if partition not in {TRAIN_PARTITION, DEVELOPMENT_PARTITION}:
            raise NeuralDevelopmentStateError(
                f"assignment {index}.partition is not ordinary train/development"
            )
        normalized.append({
            "row_id": _digest(value.get("row_id"), label=f"assignment {index}.row_id"),
            "partition": partition,
            "source_component_id": _identifier(
                value.get("source_component_id"),
                label=f"assignment {index}.source_component_id",
            ),
        })
    normalized.sort(key=lambda item: item["row_id"])
    row_ids = [item["row_id"] for item in normalized]
    if len(row_ids) != len(set(row_ids)):
        raise NeuralDevelopmentStateError("component assignment row IDs repeat")
    if row_ids != sorted(expected_row_ids):
        raise NeuralDevelopmentStateError(
            "component assignments must exactly cover train and development rows"
        )
    if expected_partitions is not None and any(
        expected_partitions.get(item["row_id"]) != item["partition"]
        for item in normalized
    ):
        raise NeuralDevelopmentStateError(
            "component assignment partition differs from the bound typed row"
        )
    return normalized


def build_neural_source_component_assignment(
    train_rows: Sequence[Mapping],
    development_rows: Sequence[Mapping],
    assignments: Sequence[Mapping],
    *,
    information_set_fingerprint_sha256: str,
    train_development_outcomes_sha256: str,
    non_data_authority_hashes: Mapping,
    source_component_split_manifest_source_sha256: str,
    source_component_split_manifest_projection_sha256: str,
) -> dict:
    """Bind an externally projected component assignment to exact typed rows."""

    train = _normalize_rows(train_rows, partition=TRAIN_PARTITION, label="train_rows")
    development = _normalize_rows(
        development_rows,
        partition=DEVELOPMENT_PARTITION,
        label="development_rows",
    )
    if {row["estimand_id"] for row in train} != {
        row["estimand_id"] for row in development
    }:
        raise NeuralDevelopmentStateError("train/development estimands differ")
    try:
        canonical_fingerprint = information_set_fingerprint(PAIRED_INFORMATION_SET)
    except InformationSetError as exc:
        raise NeuralDevelopmentStateError(str(exc)) from exc
    if information_set_fingerprint_sha256 != canonical_fingerprint:
        raise NeuralDevelopmentStateError(
            "component assignment information-set fingerprint is not canonical"
        )
    authorities = _non_data_authority_hashes(non_data_authority_hashes)
    source_sha = _digest(
        source_component_split_manifest_source_sha256,
        label="source_component_split_manifest_source_sha256",
    )
    if source_sha != authorities["source_component_split_manifest_sha256"]:
        raise NeuralDevelopmentStateError(
            "component source SHA differs from the frozen non-data authority"
        )
    expected_partitions = {
        row["row_id"]: row["partition"] for row in (*train, *development)
    }
    normalized_assignments = _normalize_assignment_rows(
        assignments,
        expected_row_ids=[row["row_id"] for row in (*train, *development)],
        expected_partitions=expected_partitions,
    )
    assignment_core = {
        "schema_version": NEURAL_COMPONENT_ASSIGNMENT_SCHEMA,
        "artifact_role": NEURAL_COMPONENT_ASSIGNMENT_ROLE,
        "status": NEURAL_COMPONENT_ASSIGNMENT_STATUS,
        "information_set": PAIRED_INFORMATION_SET,
        "information_set_fingerprint": canonical_fingerprint,
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
        "partition_row_bindings": _partition_row_bindings(train, development),
        "assignments": normalized_assignments,
        **BOUNDARY,
    }
    assignment_core["assignment_sha256"] = _canonical_sha256(assignment_core)
    return validate_neural_source_component_assignment(assignment_core)


def validate_neural_source_component_assignment(payload: Mapping) -> dict:
    if not isinstance(payload, Mapping) or payload.get("schema_version") != NEURAL_COMPONENT_ASSIGNMENT_SCHEMA:
        raise NeuralDevelopmentStateError("unsupported neural component assignment schema")
    _strict_keys(payload, _ASSIGNMENT_KEYS, label="neural component assignment")
    if payload.get("artifact_role") != NEURAL_COMPONENT_ASSIGNMENT_ROLE:
        raise NeuralDevelopmentStateError("component assignment role changed")
    if payload.get("status") != NEURAL_COMPONENT_ASSIGNMENT_STATUS:
        raise NeuralDevelopmentStateError("component assignment status changed")
    if payload.get("information_set") != PAIRED_INFORMATION_SET:
        raise NeuralDevelopmentStateError("component assignment information set changed")
    canonical_fingerprint = information_set_fingerprint(PAIRED_INFORMATION_SET)
    if payload.get("information_set_fingerprint") != canonical_fingerprint:
        raise NeuralDevelopmentStateError(
            "component assignment information-set fingerprint changed"
        )
    authorities = _non_data_authority_hashes(payload.get("non_data_authority_hashes"))
    source_sha = _digest(
        payload.get("source_component_split_manifest_source_sha256"),
        label="source_component_split_manifest_source_sha256",
    )
    if source_sha != authorities["source_component_split_manifest_sha256"]:
        raise NeuralDevelopmentStateError("component assignment source authority mismatch")
    _digest(
        payload.get("source_component_split_manifest_projection_sha256"),
        label="source_component_split_manifest_projection_sha256",
    )
    _digest(
        payload.get("train_development_outcomes_sha256"),
        label="train_development_outcomes_sha256",
    )
    bindings = _validate_partition_row_bindings(payload.get("partition_row_bindings"))
    assignments_raw = payload.get("assignments")
    expected_ids = [
        item.get("row_id")
        for item in assignments_raw
    ] if isinstance(assignments_raw, Sequence) and not isinstance(assignments_raw, (str, bytes)) else []
    assignments = _normalize_assignment_rows(
        assignments_raw, expected_row_ids=expected_ids
    )
    if assignments != assignments_raw:
        raise NeuralDevelopmentStateError(
            "component assignments are not in canonical row_id order"
        )
    if len(assignments) != bindings["train"]["n_rows"] + bindings["development"]["n_rows"]:
        raise NeuralDevelopmentStateError("component assignment row count is stale")
    for partition, binding in (
        (TRAIN_PARTITION, bindings["train"]),
        (DEVELOPMENT_PARTITION, bindings["development"]),
    ):
        row_ids = sorted(
            item["row_id"] for item in assignments if item["partition"] == partition
        )
        if len(row_ids) != binding["n_rows"]:
            raise NeuralDevelopmentStateError(
                "component assignment partition row count is stale"
            )
        if (
            _canonical_sha256(row_ids) != binding["ordered_row_ids_sha256"]
            or _canonical_sha256(row_ids) != binding["row_set_sha256"]
        ):
            raise NeuralDevelopmentStateError(
                "component assignment row IDs differ from partition binding"
            )
    _boundaries(payload, label="component assignment")
    core = {key: copy.deepcopy(value) for key, value in payload.items() if key != "assignment_sha256"}
    if payload.get("assignment_sha256") != _canonical_sha256(core):
        raise NeuralDevelopmentStateError("component assignment digest is stale")
    return copy.deepcopy(dict(payload))


def canonical_neural_source_component_assignment_bytes(payload: Mapping) -> bytes:
    try:
        return canonical_json_bytes(
            validate_neural_source_component_assignment(payload)
        )
    except ArtifactIOError as exc:
        raise NeuralDevelopmentStateError(str(exc)) from exc


def _verify_neural_component_assignment_inputs(
    payload: Mapping,
    train_rows: Sequence[Mapping],
    development_rows: Sequence[Mapping],
    *,
    expected_information_set_fingerprint: str,
    expected_train_development_outcomes_sha256: str,
    expected_non_data_authority_hashes: Mapping,
    expected_source_component_split_manifest_source_sha256: str,
    expected_source_component_split_manifest_projection_sha256: str,
) -> dict:
    validated = validate_neural_source_component_assignment(payload)
    train = _normalize_rows(
        train_rows, partition=TRAIN_PARTITION, label="expected train_rows"
    )
    development = _normalize_rows(
        development_rows,
        partition=DEVELOPMENT_PARTITION,
        label="expected development_rows",
    )
    expected = {
        "information_set_fingerprint": _digest(
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
        "source_component_split_manifest_source_sha256": _digest(
            expected_source_component_split_manifest_source_sha256,
            label="expected_source_component_split_manifest_source_sha256",
        ),
        "source_component_split_manifest_projection_sha256": _digest(
            expected_source_component_split_manifest_projection_sha256,
            label="expected_source_component_split_manifest_projection_sha256",
        ),
        "partition_row_bindings": _partition_row_bindings(train, development),
    }
    for field, value in expected.items():
        if validated[field] != value:
            raise NeuralDevelopmentStateError(
                f"component assignment differs from expected {field}"
            )
    if expected["information_set_fingerprint"] != information_set_fingerprint(
        PAIRED_INFORMATION_SET
    ):
        raise NeuralDevelopmentStateError(
            "expected component assignment fingerprint is not canonical"
        )
    if (
        expected["source_component_split_manifest_source_sha256"]
        != expected["non_data_authority_hashes"][
            "source_component_split_manifest_sha256"
        ]
    ):
        raise NeuralDevelopmentStateError(
            "expected component source SHA differs from expected authority"
        )
    return validated


def write_neural_source_component_assignment_exclusive(
    path: Path,
    payload: Mapping,
    *,
    expected_train_rows: Sequence[Mapping],
    expected_development_rows: Sequence[Mapping],
    expected_information_set_fingerprint: str,
    expected_train_development_outcomes_sha256: str,
    expected_non_data_authority_hashes: Mapping,
    expected_source_component_split_manifest_source_sha256: str,
    expected_source_component_split_manifest_projection_sha256: str,
) -> str:
    candidate = _assert_development_path(
        Path(path), label="neural component assignment output"
    )
    validated = _verify_neural_component_assignment_inputs(
        payload,
        expected_train_rows,
        expected_development_rows,
        expected_information_set_fingerprint=expected_information_set_fingerprint,
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
            candidate,
            validated,
            validator=validate_neural_source_component_assignment,
        )
    except ArtifactIOError as exc:
        raise NeuralDevelopmentStateError(str(exc)) from exc


def _strict_json_object(raw: bytes, *, label: str) -> dict:
    def no_duplicates(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicates)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise NeuralDevelopmentStateError(
            f"{label} is not strict JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise NeuralDevelopmentStateError(f"{label} must be a JSON object")
    return value


def load_neural_source_component_assignment_exact(
    path: Path,
    expected_sha256: str,
    *,
    expected_train_rows: Sequence[Mapping],
    expected_development_rows: Sequence[Mapping],
    expected_information_set_fingerprint: str,
    expected_train_development_outcomes_sha256: str,
    expected_non_data_authority_hashes: Mapping,
    expected_source_component_split_manifest_source_sha256: str,
    expected_source_component_split_manifest_projection_sha256: str,
) -> dict:
    candidate = _assert_development_path(
        Path(path), label="neural component assignment input"
    )
    try:
        raw = read_exact_regular_file(
            candidate, _digest(expected_sha256, label="expected_sha256")
        )
    except ArtifactIOError as exc:
        raise NeuralDevelopmentStateError(str(exc)) from exc
    payload = _strict_json_object(raw, label="neural component assignment")
    validated = _verify_neural_component_assignment_inputs(
        payload,
        expected_train_rows,
        expected_development_rows,
        expected_information_set_fingerprint=expected_information_set_fingerprint,
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
    if canonical_neural_source_component_assignment_bytes(validated) != raw:
        raise NeuralDevelopmentStateError(
            "neural component assignment bytes are not canonical exact JSON"
        )
    return validated


def _validate_fit_config(value: Mapping) -> dict:
    _strict_keys(value, _FIT_CONFIG_KEYS, label="fit_config")
    result = dict(value)
    for field in ("epochs", "batch_size", "max_rows_per_fit", "max_sequence_tokens_per_batch", "max_forward_equivalents_per_fit", "max_total_forward_equivalents_five_seeds", "max_peak_parameter_bytes_per_fit"):
        result[field] = _positive_int(result.get(field), label=f"fit_config.{field}")
    for field in ("learning_rate", "sign_temperature", "gradient_clip_norm", "max_wall_time_seconds_per_fit"):
        result[field] = _finite(result.get(field), label=f"fit_config.{field}", positive=True)
    for field in ("weight_decay", "huber_weight", "nll_weight", "sign_weight"):
        result[field] = _finite(result.get(field), label=f"fit_config.{field}")
        if result[field] < 0.0:
            raise NeuralDevelopmentStateError(f"fit_config.{field} cannot be negative")
    if result["sign_weight"] <= 0.0:
        raise NeuralDevelopmentStateError("fit_config.sign_weight must remain positive")
    dropout = _finite(
        result.get("condition_unknown_dropout"),
        label="fit_config.condition_unknown_dropout",
    )
    if not 0.0 < dropout < 1.0:
        raise NeuralDevelopmentStateError(
            "fit_config.condition_unknown_dropout must be in (0,1)"
        )
    result["condition_unknown_dropout"] = dropout
    if result.get("device") != "cpu":
        raise NeuralDevelopmentStateError(
            "neural development fitted-state kernel is frozen to CPU"
        )
    return result


def _fit_config_dict(config: NeuralDevelopmentFitConfig | Mapping) -> dict:
    if isinstance(config, NeuralDevelopmentFitConfig):
        return _validate_fit_config(asdict(config))
    return _validate_fit_config(config)


def _normalize_dev_targets(
    development_rows: Sequence[Mapping], values: Mapping
) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise NeuralDevelopmentStateError("development targets must be an exact mapping")
    expected = {row["row_id"] for row in development_rows}
    if set(values) != expected:
        raise NeuralDevelopmentStateError(
            "development targets must exactly cover the development row set"
        )
    return {
        row_id: _finite(values[row_id], label=f"development target {row_id}")
        for row_id in sorted(expected)
    }


def _component_map(assignment: Mapping) -> dict[str, str]:
    validated = validate_neural_source_component_assignment(assignment)
    return {
        item["row_id"]: item["source_component_id"]
        for item in validated["assignments"]
    }


def _validate_action_replay(rows: Sequence[Mapping], cfg: ModelConfig) -> str:
    replay_records = []
    for row in rows:
        predictor_input = row["predictor_input"]
        source = predictor_input["normalized_source_sequence"]
        candidate = predictor_input["normalized_candidate_sequence"]
        if len(source) != len(candidate) or len(source) > cfg.max_len:
            raise NeuralDevelopmentStateError(
                "neural rows require equal-length sequences within frozen max_len"
            )
        edits = predictor_input["edit_list"]
        edit_count = predictor_input["edit_count"]
        if edit_count not in ALLOWED_EDIT_COUNTS or len(edits) != edit_count:
            raise NeuralDevelopmentStateError(
                "actual primary edit count must be exactly one of {1,3,5}"
            )
        replay = list(source)
        changed_positions = [index for index, pair in enumerate(zip(source, candidate, strict=True)) if pair[0] != pair[1]]
        if len(changed_positions) != edit_count:
            raise NeuralDevelopmentStateError(
                "actual source/candidate changed-position count differs from edit_count"
            )
        for edit in edits:
            if edit["op"] != "SUB":
                raise NeuralDevelopmentStateError("only SUB actions are admitted")
            position = edit["pos"]
            if replay[position] != edit["ref"] or edit["alt"] == edit["ref"]:
                raise NeuralDevelopmentStateError("SUB action ref/alt replay is invalid")
            replay[position] = edit["alt"]
        if "".join(replay) != candidate:
            raise NeuralDevelopmentStateError("SUB actions do not replay to candidate")
        replay_records.append({
            "row_id": row["row_id"],
            "edit_count": edit_count,
            "changed_positions": changed_positions,
            "source_sha256": hashlib.sha256(source.encode("ascii")).hexdigest(),
            "candidate_sha256": hashlib.sha256(candidate.encode("ascii")).hexdigest(),
        })
    return _canonical_sha256(replay_records)


def _condition_vocab(train_rows: Sequence[Mapping], cfg: ModelConfig) -> dict:
    endpoints = sorted({row["predictor_input"]["endpoint"] for row in train_rows})
    contexts = sorted({
        row["predictor_input"]["publication_context_id"] for row in train_rows
    })
    if len(endpoints) + 1 > cfg.n_endpoints or len(contexts) + 1 > cfg.n_contexts:
        raise NeuralDevelopmentStateError(
            "train-only condition vocabulary exceeds ModelConfig capacity"
        )
    return {
        "endpoint": {"__UNK__": 0, **{name: index + 1 for index, name in enumerate(endpoints)}},
        "context": {"__UNK__": 0, **{name: index + 1 for index, name in enumerate(contexts)}},
    }


def _validate_condition_vocab(value: Mapping, cfg: ModelConfig) -> dict:
    _strict_keys(value, {"endpoint", "context"}, label="condition_vocab")
    normalized = {}
    for axis, capacity in (("endpoint", cfg.n_endpoints), ("context", cfg.n_contexts)):
        mapping = value.get(axis)
        if not isinstance(mapping, Mapping) or mapping.get("__UNK__") != 0:
            raise NeuralDevelopmentStateError(f"condition_vocab.{axis} lacks __UNK__=0")
        result = {}
        for key, index in mapping.items():
            _feature_text(key, label=f"condition_vocab.{axis}.{key}")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise NeuralDevelopmentStateError(f"condition_vocab.{axis} index is invalid")
            result[key] = index
        if sorted(result.values()) != list(range(len(result))) or len(result) > capacity:
            raise NeuralDevelopmentStateError(f"condition_vocab.{axis} is non-contiguous or exceeds capacity")
        normalized[axis] = dict(sorted(result.items()))
    return normalized


def _tensorize(rows: Sequence[Mapping], vocab: Mapping, cfg: ModelConfig) -> dict[str, torch.Tensor]:
    n_rows = len(rows)
    max_length = max(len(row["predictor_input"]["normalized_source_sequence"]) for row in rows)
    if max_length > cfg.max_len:
        raise NeuralDevelopmentStateError("sequence exceeds ModelConfig.max_len")
    source = torch.zeros((n_rows, max_length), dtype=torch.long)
    candidate = torch.zeros((n_rows, max_length), dtype=torch.long)
    mask = torch.zeros((n_rows, max_length), dtype=torch.bool)
    endpoint = torch.zeros(n_rows, dtype=torch.long)
    context = torch.zeros(n_rows, dtype=torch.long)
    for index, row in enumerate(rows):
        predictor_input = row["predictor_input"]
        source_sequence = predictor_input["normalized_source_sequence"]
        candidate_sequence = predictor_input["normalized_candidate_sequence"]
        length = len(source_sequence)
        source[index, :length] = torch.tensor([_BASE_TO_TOKEN[base] for base in source_sequence], dtype=torch.long)
        candidate[index, :length] = torch.tensor([_BASE_TO_TOKEN[base] for base in candidate_sequence], dtype=torch.long)
        mask[index, :length] = True
        endpoint[index] = int(vocab["endpoint"].get(predictor_input["endpoint"], 0))
        context[index] = int(vocab["context"].get(predictor_input["publication_context_id"], 0))
    output = {
        "source": source,
        "candidate": candidate,
        "source_mask": mask,
        "candidate_mask": mask.clone(),
        "endpoint": endpoint,
        "context": context,
    }
    if all(row["training_target"] is not None for row in rows):
        output["target"] = torch.tensor(
            [float(row["training_target"]) for row in rows], dtype=torch.float32
        )
    return output


def _index(tensors: Mapping[str, torch.Tensor], indices: Sequence[int]) -> dict[str, torch.Tensor]:
    index = torch.as_tensor(list(indices), dtype=torch.long)
    return {key: value[index] for key, value in tensors.items()}


def _forward(model: torch.nn.Module, tensors: Mapping[str, torch.Tensor]) -> dict:
    return model(
        tensors["source"],
        tensors["candidate"],
        tensors["source_mask"],
        tensors["candidate_mask"],
        tensors["endpoint"],
        tensors["context"],
    )


def _preflight_registered_training_variant(
    method_id: str,
    train_tensors: Mapping[str, torch.Tensor],
    development_tensors: Mapping[str, torch.Tensor],
    cfg: ModelConfig,
) -> tuple[str, str]:
    """Validate method-specific control support before model construction."""

    entry = _method_registry_entry(method_id)
    variant = entry.get("training_variant_id")
    intervention = entry.get("control_intervention")
    admitted = {
        PRIMARY_TRAINING_VARIANT_ID,
        WRONG_LOCAL_TRAINING_VARIANT_ID,
        CONDITION_UNK_TRAINING_VARIANT_ID,
        POSITION_AGNOSTIC_TRAINING_VARIANT_ID,
    }
    if variant not in admitted or not isinstance(intervention, str):
        raise NeuralDevelopmentStateError(
            "registered training variant/control intervention is unsupported"
        )
    if variant == WRONG_LOCAL_TRAINING_VARIANT_ID:
        try:
            for tensors in (train_tensors, development_tensors):
                positions, ref, _alt, edit_mask = derive_replay_edit_tensors(
                    tensors["source"],
                    tensors["candidate"],
                    tensors["source_mask"],
                    tensors["candidate_mask"],
                    max_edits=cfg.max_edits,
                )
                derive_wrong_local_feature_positions(
                    tensors["source"],
                    tensors["source_mask"],
                    positions,
                    ref,
                    edit_mask,
                )
        except ValueError as exc:
            raise NeuralDevelopmentStateError(str(exc)) from exc
    return variant, intervention


def _predict(model: torch.nn.Module, tensors: Mapping[str, torch.Tensor], *, batch_size: int) -> tuple[list[float], list[float]]:
    means: list[float] = []
    logvars: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(tensors["source"]), batch_size):
            batch = _index(tensors, range(start, min(start + batch_size, len(tensors["source"]))))
            outputs = _forward(model, batch)
            means.extend(float(value) for value in outputs["mean"].detach().cpu().tolist())
            logvars.extend(float(value) for value in outputs["logvar"].detach().cpu().tolist())
    if not all(math.isfinite(value) for value in (*means, *logvars)):
        raise NeuralDevelopmentStateError("neural prediction contains non-finite values")
    return means, logvars


def _sampler_rows(train_rows: Sequence[Mapping], components: Mapping[str, str]) -> list[dict]:
    return [{
        "study": row["cell_id"],
        "endpoint": "FORMAL_CELL",
        "publication_context_id": "FORMAL_CELL",
        "edit_count": 1,
        "source_component_id": components[row["row_id"]],
    } for row in train_rows]


def _sampler_manifest(
    train_rows: Sequence[Mapping], components: Mapping[str, str], indices: Sequence[int], *, epoch: int
) -> dict:
    ordered_row_ids = [train_rows[index]["row_id"] for index in indices]
    by_cell: dict[str, int] = {}
    by_cell_component: dict[str, int] = {}
    for index in indices:
        row = train_rows[index]
        cell = row["cell_id"]
        component = components[row["row_id"]]
        by_cell[cell] = by_cell.get(cell, 0) + 1
        key = f"{cell}\u001f{component}"
        by_cell_component[key] = by_cell_component.get(key, 0) + 1
    if len(set(by_cell.values())) != 1:
        raise NeuralDevelopmentStateError("balanced sampler failed equal cell quotas")
    expected_cell_components = {
        (row["cell_id"], components[row["row_id"]]) for row in train_rows
    }
    observed_cell_components = {
        (train_rows[index]["cell_id"], components[train_rows[index]["row_id"]])
        for index in indices
    }
    if observed_cell_components != expected_cell_components:
        raise NeuralDevelopmentStateError(
            "balanced sampler omitted a frozen cell/component in this epoch"
        )
    return {
        "epoch": epoch,
        "n_presentations": len(indices),
        "ordered_row_ids_sha256": _canonical_sha256(ordered_row_ids),
        "cell_counts_sha256": _canonical_sha256(by_cell),
        "cell_component_counts_sha256": _canonical_sha256(by_cell_component),
        "n_cells": len(by_cell),
        "n_cell_components": len(by_cell_component),
    }


def _method_registry_entry(method_id: str) -> dict:
    payload = canonical_registry_payload()
    entries = [item for item in payload["methods"] if item["method_id"] == method_id]
    if len(entries) != 1:
        raise NeuralDevelopmentStateError(f"method is not uniquely registered: {method_id}")
    return entries[0]


def _conservative_parameter_bytes(cfg: ModelConfig, method_id: str) -> tuple[int, int]:
    d = cfg.d_model
    embeddings = d * (cfg.vocab_size + cfg.max_len + cfg.n_endpoints + cfg.n_contexts)
    transformer = cfg.n_layers * (8 * d * d + 4 * d * cfg.ff_dim + 32 * d)
    route_heads_and_controls = 256 * d * d + 32 * d * cfg.pair_rank + 100_000
    per_model = embeddings + transformer + route_heads_and_controls
    model_count = 1 + int(_method_registry_entry(method_id)["matched_reference_method_id"] is not None)
    return per_model * model_count * 4, model_count


def _resource_preflight(
    train_rows: Sequence[Mapping], development_rows: Sequence[Mapping], cfg: ModelConfig,
    fit_config: Mapping, method_id: str, components: Mapping[str, str],
) -> dict:
    n_rows = len(train_rows) + len(development_rows)
    n_cells = len({row["cell_id"] for row in train_rows})
    cell_quota = int(math.ceil(len(train_rows) / n_cells))
    component_counts_by_cell: dict[str, set[str]] = {}
    for row in train_rows:
        component_counts_by_cell.setdefault(row["cell_id"], set()).add(
            components[row["row_id"]]
        )
    maximum_components_per_cell = max(map(len, component_counts_by_cell.values()))
    if maximum_components_per_cell > cell_quota:
        raise NeuralDevelopmentStateError(
            "balanced sampler cannot cover every frozen cell/component under its bounded epoch quota"
        )
    train_presentations_per_epoch = n_cells * cell_quota
    pair_forward_rows = (
        fit_config["epochs"] * (train_presentations_per_epoch + len(development_rows))
        + len(development_rows)
    )
    forward_equivalents = 2 * pair_forward_rows
    maximum_sequence_length = max(
        len(row["predictor_input"]["normalized_source_sequence"])
        for row in (*train_rows, *development_rows)
    )
    peak_batch_tokens = 2 * min(
        fit_config["batch_size"],
        max(train_presentations_per_epoch, len(development_rows)),
    ) * maximum_sequence_length
    parameter_bytes, preallocation_model_count = _conservative_parameter_bytes(cfg, method_id)
    failures = []
    if n_rows > fit_config["max_rows_per_fit"]:
        failures.append("rows")
    if peak_batch_tokens > fit_config["max_sequence_tokens_per_batch"]:
        failures.append("sequence_tokens_per_batch")
    if forward_equivalents > fit_config["max_forward_equivalents_per_fit"]:
        failures.append("forward_equivalents")
    if parameter_bytes > fit_config["max_peak_parameter_bytes_per_fit"]:
        failures.append("conservative_parameter_bytes")
    if failures:
        raise NeuralDevelopmentStateError(
            "resource preflight failed before model construction: " + ", ".join(failures)
        )
    return {
        "n_rows": n_rows,
        "n_train_rows": len(train_rows),
        "n_development_rows": len(development_rows),
        "n_train_cells": n_cells,
        "maximum_components_per_train_cell": maximum_components_per_cell,
        "train_presentations_per_epoch": train_presentations_per_epoch,
        "pair_forward_rows": pair_forward_rows,
        "sequence_encoder_forward_equivalents": forward_equivalents,
        "maximum_sequence_length": maximum_sequence_length,
        "peak_batch_sequence_tokens": peak_batch_tokens,
        "conservative_peak_parameter_bytes": parameter_bytes,
        "preallocation_model_count": preallocation_model_count,
        "five_seed_forward_equivalents": 5 * forward_equivalents,
        "status": "PASS_BEFORE_MODEL_CONSTRUCTION",
    }


def _development_metric_candidate(
    rows: Sequence[Mapping], targets: Mapping[str, float], components: Mapping[str, str],
    means: Sequence[float], logvars: Sequence[float], *, epoch: int,
) -> dict:
    by_cell: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_cell.setdefault(row["cell_id"], []).append(index)
    cell_values = []
    rank_defined = True
    for cell in sorted(by_cell):
        indices = by_cell[cell]
        try:
            metrics = component_balanced_cell_metrics(
                [targets[rows[index]["row_id"]] for index in indices],
                [means[index] for index in indices],
                [components[rows[index]["row_id"]] for index in indices],
                logvar=[logvars[index] for index in indices],
            )
        except MetricProtocolError as exc:
            raise NeuralDevelopmentStateError(str(exc)) from exc
        if metrics["sign_accuracy"] is None or metrics["gaussian_nll"] is None:
            raise NeuralDevelopmentStateError(
                f"development cell {cell} has undefined sign or NLL"
            )
        rank = metrics["effect_weighted_spearman_diagnostic"]
        if rank is None:
            rank_defined = False
            rank = -1.0
        cell_values.append((float(metrics["sign_accuracy"]), float(rank), float(metrics["gaussian_nll"])))
    return {
        "epoch": epoch,
        "macro_sign_accuracy": math.fsum(value[0] for value in cell_values) / len(cell_values),
        "macro_delta_spearman": (
            math.fsum(value[1] for value in cell_values) / len(cell_values)
            if rank_defined else -1.0
        ),
        "macro_gaussian_nll": math.fsum(value[2] for value in cell_values) / len(cell_values),
        "rank_metric_status": "DEFINED_ALL_CELLS" if rank_defined else "HOLD_UNDEFINED_RANK_SENTINEL_MINUS_ONE",
        "selection_metric_policy": SELECTION_METRIC_POLICY,
    }


def _prediction_rows(rows: Sequence[Mapping], means: Sequence[float], logvars: Sequence[float]) -> list[dict]:
    predictions = [{
        "schema_version": "r3x_prediction_row_v1",
        "row_id": row["row_id"],
        "predicted_delta_mean": float(mean),
        "predicted_delta_logvar": float(logvar),
    } for row, mean, logvar in zip(rows, means, logvars, strict=True)]
    try:
        prediction_jsonl_bytes(predictions)
    except ArtifactGraphError as exc:
        raise NeuralDevelopmentStateError(str(exc)) from exc
    return predictions


def _prediction_sha256(predictions: Sequence[Mapping]) -> str:
    try:
        return hashlib.sha256(prediction_jsonl_bytes(predictions)).hexdigest()
    except ArtifactGraphError as exc:
        raise NeuralDevelopmentStateError(str(exc)) from exc


def _state_tensor_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise NeuralDevelopmentStateError(f"state value is not tensor: {key}")
        tensor = value.detach().cpu().contiguous()
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise NeuralDevelopmentStateError(f"state tensor is non-finite: {key}")
        if tensor.dtype not in {torch.float32, torch.int64, torch.bool}:
            raise NeuralDevelopmentStateError(f"unsupported state dtype: {key}={tensor.dtype}")
        header = _canonical_bytes({
            "key": key,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        })
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        if tensor.dtype is torch.bool:
            raw = tensor.to(torch.uint8).numpy().tobytes(order="C")
        else:
            array = tensor.numpy()
            little_endian_dtype = array.dtype.newbyteorder("<")
            raw = array.astype(little_endian_dtype, copy=False).tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


@contextlib.contextmanager
def _deterministic_cpu(seed: int) -> Iterator[None]:
    previous_rng = torch.random.get_rng_state()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        yield
    finally:
        torch.random.set_rng_state(previous_rng)
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)


def _source_hashes() -> tuple[str, str, str]:
    here = Path(__file__)
    return (
        hashlib.sha256(here.read_bytes()).hexdigest(),
        hashlib.sha256(here.with_name("models.py").read_bytes()).hexdigest(),
        hashlib.sha256(here.with_name("train.py").read_bytes()).hexdigest(),
    )


def fit_neural_development_state(
    method_id: str,
    cfg: ModelConfig,
    fit_config: NeuralDevelopmentFitConfig | Mapping,
    train_rows: Sequence[Mapping],
    development_rows: Sequence[Mapping],
    development_targets: Mapping,
    source_component_assignment: Mapping,
    *,
    seed: int,
    expected_registry_sha256: str,
    run_start_sha256: str,
    development_activation_sha256: str,
) -> dict:
    """Fit one frozen seed and return a strict development-only fitted state."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in FROZEN_NEURAL_SEEDS:
        raise NeuralDevelopmentStateError("seed is not one of the five frozen seeds")
    _digest(run_start_sha256, label="run_start_sha256")
    _digest(development_activation_sha256, label="development_activation_sha256")
    expected_registry = _digest(expected_registry_sha256, label="expected_registry_sha256")
    if expected_registry != registry_sha256():
        raise NeuralDevelopmentStateError("expected neural registry SHA is stale")
    if not isinstance(cfg, ModelConfig):
        raise NeuralDevelopmentStateError("cfg must be a frozen ModelConfig")
    config = _fit_config_dict(fit_config)
    train = _normalize_rows(train_rows, partition=TRAIN_PARTITION, label="train_rows")
    development = _normalize_rows(
        development_rows, partition=DEVELOPMENT_PARTITION, label="development_rows"
    )
    estimand_ids = {row["estimand_id"] for row in (*train, *development)}
    if len(estimand_ids) != 1:
        raise NeuralDevelopmentStateError("train/development estimands differ")
    estimand_id = next(iter(estimand_ids))
    row_bindings = _partition_row_bindings(train, development)
    assignment = validate_neural_source_component_assignment(source_component_assignment)
    if assignment["partition_row_bindings"] != row_bindings:
        raise NeuralDevelopmentStateError("component assignment row binding differs from fit rows")
    components = _component_map(assignment)
    targets = _normalize_dev_targets(development, development_targets)
    action_sha = _validate_action_replay((*train, *development), cfg)
    info_fingerprint = information_set_fingerprint(PAIRED_INFORMATION_SET)
    if assignment["information_set_fingerprint"] != info_fingerprint:
        raise NeuralDevelopmentStateError("component assignment information set differs")
    resource = _resource_preflight(
        train, development, cfg, config, method_id, components
    )
    vocab = _condition_vocab(train, cfg)
    train_tensor = _tensorize(train, vocab, cfg)
    development_tensor = _tensorize(development, vocab, cfg)
    training_variant_id, control_intervention = (
        _preflight_registered_training_variant(
            method_id, train_tensor, development_tensor, cfg
        )
    )
    sampler_input = _sampler_rows(train, components)
    training_design_sha = _canonical_sha256({
        "partition_row_bindings": row_bindings,
        "component_assignment_sha256": assignment["assignment_sha256"],
        "development_targets_sha256": _canonical_sha256(targets),
        "train_development_outcomes_sha256": assignment["train_development_outcomes_sha256"],
        "action_replay_sha256": action_sha,
        "condition_vocab": vocab,
        "training_variant_id": training_variant_id,
        "control_intervention": control_intervention,
    })
    started = time.monotonic()
    with _deterministic_cpu(seed):
        try:
            model, compiled_binding = compile_registered_development_model(
                method_id,
                cfg,
                expected_registry_sha256=expected_registry,
            )
        except NeuralMethodRegistryError as exc:
            raise NeuralDevelopmentStateError(str(exc)) from exc
        if compiled_binding["preallocation_peak_parameter_bytes_upper_bound"] != resource["conservative_peak_parameter_bytes"]:
            raise NeuralDevelopmentStateError(
                "local resource preflight differs from registry preallocation formula"
            )
        if compiled_binding["action_budget_validated_by_compiler"] is not False:
            raise NeuralDevelopmentStateError("registry unexpectedly claimed action validation")
        if (
            compiled_binding.get("training_variant_id") != training_variant_id
            or compiled_binding.get("control_intervention") != control_intervention
            or compiled_binding.get("training_ready") is not False
            or compiled_binding.get("publication") is not False
            or compiled_binding.get("publication_evidence") is not False
        ):
            raise NeuralDevelopmentStateError(
                "compiled registry control/boundary binding changed"
            )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        epoch_candidates: list[dict] = []
        sampler_manifests: list[dict] = []
        selected_state: dict[str, torch.Tensor] | None = None
        selected_epoch_so_far: int | None = None
        for epoch in range(1, config["epochs"] + 1):
            if time.monotonic() - started > config["max_wall_time_seconds_per_fit"]:
                raise NeuralDevelopmentStateError("wall-time resource limit exceeded during fit")
            model.train()
            try:
                indices = balanced_epoch_indices(sampler_input, seed=seed + epoch)
                presented = _index(train_tensor, indices)
                masked_endpoint, masked_context, _ = apply_condition_unknown_dropout(
                    presented["endpoint"],
                    presented["context"],
                    rate=config["condition_unknown_dropout"],
                    seed=seed * 100_000 + epoch,
                )
            except TrainingProtocolError as exc:
                raise NeuralDevelopmentStateError(str(exc)) from exc
            presented["endpoint"] = masked_endpoint
            presented["context"] = masked_context
            sampler_manifests.append(
                _sampler_manifest(train, components, indices, epoch=epoch)
            )
            for start in range(0, len(indices), config["batch_size"]):
                if time.monotonic() - started > config["max_wall_time_seconds_per_fit"]:
                    raise NeuralDevelopmentStateError("wall-time resource limit exceeded during fit")
                batch = _index(
                    presented,
                    range(start, min(start + config["batch_size"], len(indices))),
                )
                optimizer.zero_grad(set_to_none=True)
                outputs = _forward(model, batch)
                try:
                    losses = effect_loss(
                        outputs,
                        batch["target"],
                        huber_weight=config["huber_weight"],
                        nll_weight=config["nll_weight"],
                        sign_weight=config["sign_weight"],
                        sign_temperature=config["sign_temperature"],
                    )
                except TrainingProtocolError as exc:
                    raise NeuralDevelopmentStateError(str(exc)) from exc
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
                optimizer.step()
            means, logvars = _predict(
                model, development_tensor, batch_size=config["batch_size"]
            )
            candidate = _development_metric_candidate(
                development, targets, components, means, logvars, epoch=epoch
            )
            epoch_candidates.append(candidate)
            try:
                selection_so_far = select_checkpoint(epoch_candidates, sign_gate=SIGN_GATE)
            except TrainingProtocolError as exc:
                raise NeuralDevelopmentStateError(str(exc)) from exc
            if selection_so_far["selected_epoch"] != selected_epoch_so_far:
                selected_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                selected_epoch_so_far = selection_so_far["selected_epoch"]
        try:
            selection = select_checkpoint(epoch_candidates, sign_gate=SIGN_GATE)
        except TrainingProtocolError as exc:
            raise NeuralDevelopmentStateError(str(exc)) from exc
        if selected_state is None or selected_epoch_so_far != selection["selected_epoch"]:
            raise NeuralDevelopmentStateError("selected checkpoint state was not retained")
        model.load_state_dict(selected_state, strict=True)
        final_means, final_logvars = _predict(
            model, development_tensor, batch_size=config["batch_size"]
        )

    predictions = _prediction_rows(development, final_means, final_logvars)
    state_dict = {key: value.detach().cpu().clone() for key, value in selected_state.items()}
    kernel_sha, models_sha, train_sha = _source_hashes()
    compiled_binding_sha = _canonical_sha256(compiled_binding)
    selection_payload = {
        **selection,
        "epoch_candidates": epoch_candidates,
        "selection_metric_policy": SELECTION_METRIC_POLICY,
        "sign_gate_unchanged": SIGN_GATE,
        "best_seed_selection": False,
    }
    state = {
        "schema_version": NEURAL_DEVELOPMENT_STATE_SCHEMA,
        "artifact_role": NEURAL_DEVELOPMENT_STATE_ROLE,
        "status": NEURAL_DEVELOPMENT_STATE_STATUS,
        "run_start_sha256": run_start_sha256,
        "development_activation_sha256": development_activation_sha256,
        "train_development_outcomes_sha256": assignment["train_development_outcomes_sha256"],
        "non_data_authority_hashes": assignment["non_data_authority_hashes"],
        "estimand_id": estimand_id,
        "method_id": method_id,
        "training_variant_id": training_variant_id,
        "architecture_id": compiled_binding["architecture_id"],
        "registry_schema": REGISTRY_SCHEMA,
        "registry_sha256": expected_registry,
        "compiled_binding": compiled_binding,
        "compiled_binding_sha256": compiled_binding_sha,
        "kernel_code_sha256": kernel_sha,
        "models_code_sha256": models_sha,
        "train_code_sha256": train_sha,
        "model_config": asdict(cfg),
        "model_config_sha256": _canonical_sha256(asdict(cfg)),
        "fit_config": config,
        "fit_config_sha256": _canonical_sha256(config),
        "information_set": PAIRED_INFORMATION_SET,
        "information_set_fingerprint": info_fingerprint,
        "seed": seed,
        "partition_row_bindings": row_bindings,
        "component_assignment_binding": {
            "assignment_sha256": assignment["assignment_sha256"],
            "information_set_fingerprint": assignment["information_set_fingerprint"],
            "train_development_outcomes_sha256": assignment["train_development_outcomes_sha256"],
            "source_component_split_manifest_source_sha256": assignment["source_component_split_manifest_source_sha256"],
            "source_component_split_manifest_projection_sha256": assignment["source_component_split_manifest_projection_sha256"],
        },
        "training_design_sha256": training_design_sha,
        "development_target_sha256": _canonical_sha256(targets),
        "action_policy": ACTION_POLICY,
        "allowed_edit_counts": [1, 3, 5],
        "action_replay_sha256": action_sha,
        "condition_vocab": vocab,
        "condition_vocab_sha256": _canonical_sha256(vocab),
        "row_order_policy": ROW_ORDER_POLICY,
        "sampler_policy": SAMPLER_POLICY,
        "sampler_epoch_manifests": sampler_manifests,
        "selection_metric_policy": SELECTION_METRIC_POLICY,
        "checkpoint_selection": selection_payload,
        "checkpoint_selection_sha256": _canonical_sha256(selection_payload),
        "selected_epoch": selection["selected_epoch"],
        "development_predictions": predictions,
        "development_prediction_sha256": _prediction_sha256(predictions),
        "resource_preflight_policy": RESOURCE_PREFLIGHT_POLICY,
        "resource_preflight": resource,
        "parameter_match_policy": PARAMETER_MATCH_POLICY,
        "determinism_policy": DETERMINISM_POLICY,
        "state_tensor_digest_policy": STATE_TENSOR_DIGEST_POLICY,
        "state_shapes": {key: list(value.shape) for key, value in state_dict.items()},
        "state_dtypes": {key: str(value.dtype) for key, value in state_dict.items()},
        "state_tensor_sha256": _state_tensor_sha256(state_dict),
        "state_dict": state_dict,
        "exact_recompute_policy": EXACT_RECOMPUTE_POLICY,
        **BOUNDARY,
    }
    return validate_neural_development_state(state)


def validate_neural_development_state(payload: Mapping) -> dict:
    if not isinstance(payload, Mapping) or payload.get("schema_version") != NEURAL_DEVELOPMENT_STATE_SCHEMA:
        raise NeuralDevelopmentStateError("unsupported neural development state schema")
    _strict_keys(payload, _STATE_KEYS, label="neural development state")
    if payload.get("artifact_role") != NEURAL_DEVELOPMENT_STATE_ROLE:
        raise NeuralDevelopmentStateError("neural state role changed")
    if payload.get("status") != NEURAL_DEVELOPMENT_STATE_STATUS:
        raise NeuralDevelopmentStateError("neural state status changed")
    _boundaries(payload, label="neural state")
    for field in ("run_start_sha256", "development_activation_sha256", "train_development_outcomes_sha256", "training_design_sha256", "development_target_sha256", "action_replay_sha256", "compiled_binding_sha256", "kernel_code_sha256", "models_code_sha256", "train_code_sha256", "model_config_sha256", "fit_config_sha256", "information_set_fingerprint", "condition_vocab_sha256", "checkpoint_selection_sha256", "development_prediction_sha256", "state_tensor_sha256"):
        _digest(payload.get(field), label=field)
    if payload.get("information_set") != PAIRED_INFORMATION_SET:
        raise NeuralDevelopmentStateError("neural state information set changed")
    if payload.get("information_set_fingerprint") != information_set_fingerprint(PAIRED_INFORMATION_SET):
        raise NeuralDevelopmentStateError("neural state information-set fingerprint is stale")
    if payload.get("registry_schema") != REGISTRY_SCHEMA or payload.get("registry_sha256") != registry_sha256():
        raise NeuralDevelopmentStateError("neural state registry binding is stale")
    method_id = _identifier(payload.get("method_id"), label="method_id")
    registered_entry = _method_registry_entry(method_id)
    if payload.get("training_variant_id") != registered_entry.get(
        "training_variant_id"
    ):
        raise NeuralDevelopmentStateError(
            "neural state training variant differs from the exact registry"
        )
    _identifier(payload.get("architecture_id"), label="architecture_id")
    _identifier(payload.get("estimand_id"), label="estimand_id")
    if isinstance(payload.get("seed"), bool) or payload.get("seed") not in FROZEN_NEURAL_SEEDS:
        raise NeuralDevelopmentStateError("neural state seed is not frozen")
    authorities = _non_data_authority_hashes(payload.get("non_data_authority_hashes"))
    if payload["train_development_outcomes_sha256"] == authorities["analysis_code_tree_manifest_sha256"]:
        raise NeuralDevelopmentStateError("data and code authority hashes cannot alias")
    bindings = _validate_partition_row_bindings(payload.get("partition_row_bindings"))
    binding = payload.get("compiled_binding")
    if not isinstance(binding, Mapping) or payload["compiled_binding_sha256"] != _canonical_sha256(binding):
        raise NeuralDevelopmentStateError("compiled registry binding digest is stale")
    if binding.get("method_id") != payload["method_id"] or binding.get("architecture_id") != payload["architecture_id"]:
        raise NeuralDevelopmentStateError("compiled registry method/architecture binding changed")
    if binding.get("registry_sha256") != payload["registry_sha256"] or binding.get("execution_authorized") is not False:
        raise NeuralDevelopmentStateError("compiled registry authority boundary changed")
    if (
        binding.get("training_variant_id") != payload["training_variant_id"]
        or binding.get("control_intervention")
        != registered_entry.get("control_intervention")
        or binding.get("training_ready") is not False
        or binding.get("publication") is not False
        or binding.get("publication_evidence") is not False
    ):
        raise NeuralDevelopmentStateError(
            "compiled registry control/publication boundary changed"
        )
    if binding.get("action_budget_validated_by_compiler") is not False:
        raise NeuralDevelopmentStateError("compiler cannot self-validate row action budgets")
    try:
        cfg = ModelConfig(**dict(payload.get("model_config") or {}))
    except (TypeError, ValueError) as exc:
        raise NeuralDevelopmentStateError(f"model_config is invalid: {exc}") from exc
    if payload["model_config_sha256"] != _canonical_sha256(asdict(cfg)):
        raise NeuralDevelopmentStateError("model_config digest is stale")
    if (
        binding.get("model_config") != asdict(cfg)
        or binding.get("model_config_sha256") != payload["model_config_sha256"]
        or binding.get("models_code_sha256") != payload["models_code_sha256"]
    ):
        raise NeuralDevelopmentStateError(
            "compiled registry binding differs from state model/config code"
        )
    config = _validate_fit_config(payload.get("fit_config"))
    if payload["fit_config_sha256"] != _canonical_sha256(config):
        raise NeuralDevelopmentStateError("fit_config digest is stale")
    vocab = _validate_condition_vocab(payload.get("condition_vocab"), cfg)
    if payload["condition_vocab_sha256"] != _canonical_sha256(vocab):
        raise NeuralDevelopmentStateError("condition vocabulary digest is stale")
    component_binding = payload.get("component_assignment_binding")
    _strict_keys(component_binding, {"assignment_sha256", "information_set_fingerprint", "train_development_outcomes_sha256", "source_component_split_manifest_source_sha256", "source_component_split_manifest_projection_sha256"}, label="component_assignment_binding")
    for key, value in component_binding.items():
        _digest(value, label=f"component_assignment_binding.{key}")
    if component_binding["source_component_split_manifest_source_sha256"] != authorities["source_component_split_manifest_sha256"]:
        raise NeuralDevelopmentStateError("component assignment source authority changed")
    if (
        component_binding["information_set_fingerprint"]
        != payload["information_set_fingerprint"]
        or component_binding["train_development_outcomes_sha256"]
        != payload["train_development_outcomes_sha256"]
    ):
        raise NeuralDevelopmentStateError(
            "component assignment information/data authority binding changed"
        )
    if payload.get("action_policy") != ACTION_POLICY or payload.get("allowed_edit_counts") != [1, 3, 5]:
        raise NeuralDevelopmentStateError("action support policy changed")
    if payload.get("row_order_policy") != ROW_ORDER_POLICY or payload.get("sampler_policy") != SAMPLER_POLICY:
        raise NeuralDevelopmentStateError("row or sampler policy changed")
    manifests = payload.get("sampler_epoch_manifests")
    if not isinstance(manifests, list) or len(manifests) != config["epochs"]:
        raise NeuralDevelopmentStateError("sampler epoch manifests are incomplete")
    if [item.get("epoch") for item in manifests] != list(range(1, config["epochs"] + 1)):
        raise NeuralDevelopmentStateError("sampler epoch manifests are out of order")
    if payload.get("selection_metric_policy") != SELECTION_METRIC_POLICY:
        raise NeuralDevelopmentStateError("selection metric policy changed")
    selection = payload.get("checkpoint_selection")
    if not isinstance(selection, Mapping) or payload["checkpoint_selection_sha256"] != _canonical_sha256(selection):
        raise NeuralDevelopmentStateError("checkpoint selection digest is stale")
    if selection.get("sign_gate") != SIGN_GATE or selection.get("sign_gate_unchanged") != SIGN_GATE:
        raise NeuralDevelopmentStateError("sign gate changed")
    if selection.get("best_seed_selection") is not False:
        raise NeuralDevelopmentStateError("best-seed selection is prohibited")
    if payload.get("selected_epoch") != selection.get("selected_epoch"):
        raise NeuralDevelopmentStateError("selected epoch binding changed")
    predictions = payload.get("development_predictions")
    if not isinstance(predictions, list) or len(predictions) != bindings["development"]["n_rows"]:
        raise NeuralDevelopmentStateError("development predictions are incomplete")
    if payload["development_prediction_sha256"] != _prediction_sha256(predictions):
        raise NeuralDevelopmentStateError("development prediction digest is stale")
    if payload.get("resource_preflight_policy") != RESOURCE_PREFLIGHT_POLICY or payload.get("parameter_match_policy") != PARAMETER_MATCH_POLICY:
        raise NeuralDevelopmentStateError("resource or parameter-match policy changed")
    resource = payload.get("resource_preflight")
    if not isinstance(resource, Mapping) or resource.get("status") != "PASS_BEFORE_MODEL_CONSTRUCTION":
        raise NeuralDevelopmentStateError("resource preflight is absent or failed")
    if resource.get("n_train_rows") != bindings["train"]["n_rows"] or resource.get("n_development_rows") != bindings["development"]["n_rows"]:
        raise NeuralDevelopmentStateError("resource row accounting is stale")
    if payload.get("determinism_policy") != DETERMINISM_POLICY or payload.get("state_tensor_digest_policy") != STATE_TENSOR_DIGEST_POLICY or payload.get("exact_recompute_policy") != EXACT_RECOMPUTE_POLICY:
        raise NeuralDevelopmentStateError("determinism or replay policy changed")
    kernel_sha, models_sha, train_sha = _source_hashes()
    if (payload["kernel_code_sha256"], payload["models_code_sha256"], payload["train_code_sha256"]) != (kernel_sha, models_sha, train_sha):
        raise NeuralDevelopmentStateError("current replay code differs from fitted state")
    state = payload.get("state_dict")
    shapes = payload.get("state_shapes")
    dtypes = payload.get("state_dtypes")
    if not isinstance(state, Mapping) or not state or not isinstance(shapes, Mapping) or not isinstance(dtypes, Mapping) or set(state) != set(shapes) or set(state) != set(dtypes):
        raise NeuralDevelopmentStateError("state tensor metadata is incomplete")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or shapes[key] != list(value.shape) or dtypes[key] != str(value.dtype):
            raise NeuralDevelopmentStateError(f"state tensor metadata mismatch: {key}")
    if payload["state_tensor_sha256"] != _state_tensor_sha256(state):
        raise NeuralDevelopmentStateError("state tensor digest is stale")
    # Avoid an unbounded second copy of a potentially large tensor state.  The
    # caller receives a new top-level mapping; every evidence consumer must
    # validate again after any mutation and exact replay compares tensor bytes.
    return dict(payload)


def recompute_neural_development_predictions(
    state: Mapping,
    development_rows: Sequence[Mapping],
) -> list[dict]:
    """Exact-load the registered class/state and recompute development bytes."""

    payload = validate_neural_development_state(state)
    development = _normalize_rows(
        development_rows, partition=DEVELOPMENT_PARTITION, label="development_rows"
    )
    if _row_binding(development) != payload["partition_row_bindings"]["development"]:
        raise NeuralDevelopmentStateError("development rows differ from fitted state")
    cfg = ModelConfig(**payload["model_config"])
    _validate_action_replay(development, cfg)
    config = payload["fit_config"]
    vocab = payload["condition_vocab"]
    tensors = _tensorize(development, vocab, cfg)
    with _deterministic_cpu(payload["seed"]):
        try:
            model, binding = compile_registered_development_model(
                payload["method_id"],
                cfg,
                expected_registry_sha256=payload["registry_sha256"],
            )
        except NeuralMethodRegistryError as exc:
            raise NeuralDevelopmentStateError(str(exc)) from exc
        if _canonical_sha256(binding) != payload["compiled_binding_sha256"]:
            raise NeuralDevelopmentStateError("recompiled method binding differs from state")
        model.load_state_dict(payload["state_dict"], strict=True)
        means, logvars = _predict(model, tensors, batch_size=config["batch_size"])
    predictions = _prediction_rows(development, means, logvars)
    if _prediction_sha256(predictions) != payload["development_prediction_sha256"]:
        raise NeuralDevelopmentStateError(
            "loaded state does not exactly recompute development prediction bytes"
        )
    return predictions


def rebuild_and_match_neural_development_state(
    state: Mapping,
    train_rows: Sequence[Mapping],
    development_rows: Sequence[Mapping],
    development_targets: Mapping,
    source_component_assignment: Mapping,
) -> dict:
    """Refit from bound rows and fail unless state/selection/predictions match."""

    expected = validate_neural_development_state(state)
    rebuilt = fit_neural_development_state(
        expected["method_id"],
        ModelConfig(**expected["model_config"]),
        expected["fit_config"],
        train_rows,
        development_rows,
        development_targets,
        source_component_assignment,
        seed=expected["seed"],
        expected_registry_sha256=expected["registry_sha256"],
        run_start_sha256=expected["run_start_sha256"],
        development_activation_sha256=expected["development_activation_sha256"],
    )
    compared = (
        "training_design_sha256",
        "compiled_binding_sha256",
        "checkpoint_selection_sha256",
        "development_prediction_sha256",
        "state_tensor_sha256",
    )
    mismatches = [field for field in compared if rebuilt[field] != expected[field]]
    if mismatches:
        raise NeuralDevelopmentStateError(
            "exact neural refit differs from fitted state: " + ", ".join(mismatches)
        )
    return rebuilt


def save_neural_development_state_exclusive(path: Path, payload: Mapping) -> str:
    candidate = _assert_development_path(Path(path), label="neural state output")
    validated = validate_neural_development_state(payload)
    buffer = io.BytesIO()
    torch.save(validated, buffer)

    def replay(raw: bytes) -> None:
        try:
            decoded = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise NeuralDevelopmentStateError(
                f"neural state cannot be safely replayed: {type(exc).__name__}: {exc}"
            ) from exc
        validate_neural_development_state(decoded)

    try:
        return atomic_publish_bytes(candidate, buffer.getvalue(), replay_validator=replay)
    except ArtifactIOError as exc:
        raise NeuralDevelopmentStateError(str(exc)) from exc


def load_neural_development_state_exact(path: Path, expected_sha256: str) -> dict:
    candidate = _assert_development_path(Path(path), label="neural state input")
    try:
        raw = read_exact_regular_file(candidate, _digest(expected_sha256, label="expected_sha256"))
    except ArtifactIOError as exc:
        raise NeuralDevelopmentStateError(str(exc)) from exc
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise NeuralDevelopmentStateError(
            f"neural state cannot be safely decoded: {type(exc).__name__}: {exc}"
        ) from exc
    return validate_neural_development_state(payload)


def ensemble_neural_development_states(states: Sequence[Mapping]) -> dict:
    if isinstance(states, (str, bytes)) or not isinstance(states, Sequence):
        raise NeuralDevelopmentStateError("five-seed ensemble states must be a sequence")
    validated = [validate_neural_development_state(state) for state in states]
    if [state["seed"] for state in validated] != list(FROZEN_NEURAL_SEEDS):
        raise NeuralDevelopmentStateError("ensemble must use all five seeds in frozen order")
    common_fields = (
        "method_id", "training_variant_id", "architecture_id", "registry_sha256", "model_config_sha256",
        "fit_config_sha256", "information_set", "information_set_fingerprint",
        "estimand_id", "partition_row_bindings", "training_design_sha256",
    )
    for field in common_fields:
        if any(state[field] != validated[0][field] for state in validated[1:]):
            raise NeuralDevelopmentStateError(f"five-seed state binding differs: {field}")
    row_ids = [row["row_id"] for row in validated[0]["development_predictions"]]
    seed_maps = []
    for state in validated:
        mapping = {row["row_id"]: row for row in state["development_predictions"]}
        if list(mapping) != row_ids:
            raise NeuralDevelopmentStateError("five-seed prediction row order differs")
        seed_maps.append(mapping)
    predictions = []
    for row_id in row_ids:
        means = [float(mapping[row_id]["predicted_delta_mean"]) for mapping in seed_maps]
        logvars = [float(mapping[row_id]["predicted_delta_logvar"]) for mapping in seed_maps]
        mean = math.fsum(means) / len(means)
        variance = (
            math.fsum(math.exp(logvar) + value * value for value, logvar in zip(means, logvars, strict=True)) / len(means)
            - mean * mean
        )
        predictions.append({
            "schema_version": "r3x_prediction_row_v1",
            "row_id": row_id,
            "predicted_delta_mean": mean,
            "predicted_delta_logvar": math.log(max(variance, VARIANCE_FLOOR)),
        })
    first = validated[0]
    manifest = {
        "schema_version": NEURAL_ENSEMBLE_SCHEMA,
        "artifact_role": NEURAL_ENSEMBLE_ROLE,
        "status": NEURAL_ENSEMBLE_STATUS,
        "method_id": first["method_id"],
        "training_variant_id": first["training_variant_id"],
        "architecture_id": first["architecture_id"],
        "registry_sha256": first["registry_sha256"],
        "model_config_sha256": first["model_config_sha256"],
        "fit_config_sha256": first["fit_config_sha256"],
        "information_set": first["information_set"],
        "information_set_fingerprint": first["information_set_fingerprint"],
        "estimand_id": first["estimand_id"],
        "partition_row_bindings": first["partition_row_bindings"],
        "seeds": list(FROZEN_NEURAL_SEEDS),
        "seed_state_tensor_sha256": [state["state_tensor_sha256"] for state in validated],
        "seed_checkpoint_selection_sha256": [state["checkpoint_selection_sha256"] for state in validated],
        "seed_development_prediction_sha256": [state["development_prediction_sha256"] for state in validated],
        "ensemble_rule": ENSEMBLE_RULE,
        "ensemble_variance_formula": ENSEMBLE_VARIANCE_FORMULA,
        "variance_floor": VARIANCE_FLOOR,
        "ensemble_predictions": predictions,
        "ensemble_prediction_sha256": _prediction_sha256(predictions),
        **BOUNDARY,
    }
    return validate_neural_development_ensemble(manifest)


def validate_neural_development_ensemble(payload: Mapping) -> dict:
    if not isinstance(payload, Mapping) or payload.get("schema_version") != NEURAL_ENSEMBLE_SCHEMA:
        raise NeuralDevelopmentStateError("unsupported neural ensemble schema")
    _strict_keys(payload, _ENSEMBLE_KEYS, label="neural development ensemble")
    if payload.get("artifact_role") != NEURAL_ENSEMBLE_ROLE or payload.get("status") != NEURAL_ENSEMBLE_STATUS:
        raise NeuralDevelopmentStateError("neural ensemble role/status changed")
    _boundaries(payload, label="neural ensemble")
    if payload.get("seeds") != list(FROZEN_NEURAL_SEEDS):
        raise NeuralDevelopmentStateError("neural ensemble seeds changed")
    method_id = _identifier(payload.get("method_id"), label="method_id")
    if payload.get("training_variant_id") != _method_registry_entry(method_id).get(
        "training_variant_id"
    ):
        raise NeuralDevelopmentStateError(
            "neural ensemble training variant differs from the exact registry"
        )
    if payload.get("information_set") != PAIRED_INFORMATION_SET or payload.get("information_set_fingerprint") != information_set_fingerprint(PAIRED_INFORMATION_SET):
        raise NeuralDevelopmentStateError("neural ensemble information set changed")
    if payload.get("ensemble_rule") != ENSEMBLE_RULE or payload.get("ensemble_variance_formula") != ENSEMBLE_VARIANCE_FORMULA or payload.get("variance_floor") != VARIANCE_FLOOR:
        raise NeuralDevelopmentStateError("neural ensemble aggregation rule changed")
    for field in ("registry_sha256", "model_config_sha256", "fit_config_sha256", "information_set_fingerprint", "ensemble_prediction_sha256"):
        _digest(payload.get(field), label=field)
    for field in ("seed_state_tensor_sha256", "seed_checkpoint_selection_sha256", "seed_development_prediction_sha256"):
        values = payload.get(field)
        if not isinstance(values, list) or len(values) != 5:
            raise NeuralDevelopmentStateError(f"{field} is incomplete")
        for value in values:
            _digest(value, label=field)
    _validate_partition_row_bindings(payload.get("partition_row_bindings"))
    predictions = payload.get("ensemble_predictions")
    if not isinstance(predictions, list) or payload["ensemble_prediction_sha256"] != _prediction_sha256(predictions):
        raise NeuralDevelopmentStateError("ensemble prediction digest is stale")
    return copy.deepcopy(dict(payload))


def fit_neural_development_five_seed_ensemble(
    method_id: str,
    cfg: ModelConfig,
    fit_config: NeuralDevelopmentFitConfig | Mapping,
    train_rows: Sequence[Mapping],
    development_rows: Sequence[Mapping],
    development_targets: Mapping,
    source_component_assignment: Mapping,
    *,
    expected_registry_sha256: str,
    run_start_sha256: str,
    development_activation_sha256: str,
) -> tuple[list[dict], dict]:
    """Fit all five seeds without any best-seed selection."""

    config = _fit_config_dict(fit_config)
    train = _normalize_rows(train_rows, partition=TRAIN_PARTITION, label="train_rows")
    development = _normalize_rows(development_rows, partition=DEVELOPMENT_PARTITION, label="development_rows")
    assignment = validate_neural_source_component_assignment(
        source_component_assignment
    )
    if assignment["partition_row_bindings"] != _partition_row_bindings(
        train, development
    ):
        raise NeuralDevelopmentStateError(
            "component assignment row binding differs from five-seed fit rows"
        )
    resource = _resource_preflight(
        train, development, cfg, config, method_id, _component_map(assignment)
    )
    if resource["five_seed_forward_equivalents"] > config["max_total_forward_equivalents_five_seeds"]:
        raise NeuralDevelopmentStateError(
            "five-seed resource budget failed before model construction"
        )
    states = [
        fit_neural_development_state(
            method_id,
            cfg,
            config,
            train,
            development,
            development_targets,
            source_component_assignment,
            seed=seed,
            expected_registry_sha256=expected_registry_sha256,
            run_start_sha256=run_start_sha256,
            development_activation_sha256=development_activation_sha256,
        )
        for seed in FROZEN_NEURAL_SEEDS
    ]
    return states, ensemble_neural_development_states(states)


__all__ = [
    "ACTION_POLICY",
    "ALLOWED_EDIT_COUNTS",
    "DETERMINISM_POLICY",
    "NEURAL_COMPONENT_ASSIGNMENT_SCHEMA",
    "NEURAL_DEVELOPMENT_STATE_SCHEMA",
    "NEURAL_ENSEMBLE_SCHEMA",
    "TRAINING_VARIANT_ID",
    "NeuralDevelopmentFitConfig",
    "NeuralDevelopmentStateError",
    "build_neural_source_component_assignment",
    "canonical_neural_source_component_assignment_bytes",
    "ensemble_neural_development_states",
    "fit_neural_development_five_seed_ensemble",
    "fit_neural_development_state",
    "load_neural_development_state_exact",
    "load_neural_source_component_assignment_exact",
    "rebuild_and_match_neural_development_state",
    "recompute_neural_development_predictions",
    "save_neural_development_state_exclusive",
    "validate_neural_development_ensemble",
    "validate_neural_development_state",
    "validate_neural_source_component_assignment",
    "write_neural_source_component_assignment_exclusive",
]
