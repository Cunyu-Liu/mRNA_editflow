"""Versioned neural method-to-architecture bindings for R3X development.

This module closes one narrow ambiguity: a method identifier cannot be mapped
to whichever model class happens to accept the same ``ModelConfig``.  It is a
code-level compiler and *not* a data-access or execution authorization.  A
future development runner must still exact-load its independent authorities
before calling it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import torch
from torch import nn

from .models import (
    BoundedLocalGlobalExactPotentialCritic,
    BoundedLocalConditionUnknownControl,
    BoundedLocalParameterMatchedUnrestrictedCritic,
    BoundedLocalWrongPositionControl,
    ConditionAwareParameterMatchedUnrestrictedSiamese,
    EditAwareFullPairUnrestrictedCritic,
    ExactPotentialCritic,
    FullPairUnrestrictedSiamese,
    ModelConfig,
    ReplayDerivedSparseEditSetAdditiveCritic,
    ReplayDerivedSparseEditSetConditionUnknownControl,
    ReplayDerivedSparseEditSetPositionAgnosticControl,
    ReplayDerivedSparseEditSetWrongPositionControl,
    count_parameters,
    require_parameter_match,
)


REGISTRY_SCHEMA = "r3x_neural_method_architecture_registry_v4"
FACTORY_VERSION = "R3X_EXACT_CLASS_FACTORY_V4"
MAXIMUM_MATCHED_PARAMETER_GAP = 0.05
MAXIMUM_PREALLOCATION_PARAMETER_UPPER_BOUND = 50_000_000
FLOAT32_PARAMETER_BYTES = 4
MAXIMUM_PREALLOCATION_PARAMETER_BYTES = (
    MAXIMUM_PREALLOCATION_PARAMETER_UPPER_BOUND * FLOAT32_PARAMETER_BYTES
)
PAIRED_INFORMATION_SET = "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR"
PRIMARY_PAIR_POLICY = "EQUAL_LENGTH_SUBSTITUTION_ONLY_EDIT_COUNT_1_3_5_V1"
ACTION_BUDGET_ENFORCEMENT = "RUNNER_ENFORCEMENT_REQUIRED"
CONSTRUCTION_DEVICE = "cpu"
FLOATING_DTYPE = "torch.float32"
PREALLOCATION_POLICY = "CANDIDATE_PLUS_REFERENCE_FLOAT32_PARAMETER_BYTES_V1"

PRIMARY_TRAINING_VARIANT_ID = "PRIMARY"
WRONG_LOCAL_TRAINING_VARIANT_ID = (
    "DETERMINISTIC_COMPATIBLE_WRONG_LOCAL_POSITION_ALL_PARTITIONS_V1"
)
CONDITION_UNK_TRAINING_VARIANT_ID = "CONDITION_UNK_ONLY_ALL_PARTITIONS_V1"
POSITION_AGNOSTIC_TRAINING_VARIANT_ID = (
    "POSITION_AGNOSTIC_ARCHITECTURE_MATCHED_CONTROL_V2"
)
NO_CONTROL_INTERVENTION = "NONE"
WRONG_LOCAL_CONTROL_INTERVENTION = (
    "RELOCATE_ONLY_LOCAL_FEATURES_TO_COMPATIBLE_UNEDITED_SOURCE_POSITIONS"
)
CONDITION_UNK_CONTROL_INTERVENTION = (
    "REPLACE_ENDPOINT_AND_CONTEXT_IDS_WITH_EXPLICIT_UNK_ZERO"
)
POSITION_AGNOSTIC_CONTROL_INTERVENTION = (
    "REMOVE_ACTUAL_EDIT_POSITION_FROM_PREDICTION_USE_GLOBAL_POOLED_SOURCE_"
    "AND_REF_ALT_PSEUDOCOORDINATE_USING_SAME_PARAMETERS"
)
B_INTERACTION_STATUS = "NOT_RUN_ACTION_SUPPORT"

A0_METHOD_ID = "A0_EXACT_POTENTIAL_V1"
A0_CONTROL_METHOD_ID = "A0_CONDITION_AWARE_MATCHED_CONTROL_V2"
A1_METHOD_ID = "A1_BOUNDED_LOCAL_EXACT_POTENTIAL_V2"
A1_CONTROL_METHOD_ID = "A1_BOUNDED_LOCAL_MATCHED_CONTROL_V2"
A1_WRONG_LOCAL_METHOD_ID = "A1_BOUNDED_LOCAL_WRONG_POSITION_CONTROL_V1"
A1_UNK_ONLY_METHOD_ID = "A1_BOUNDED_LOCAL_CONDITION_UNK_ONLY_V1"
FULL_PAIR_METHOD_ID = "FULL_PAIR_UNRESTRICTED_SIAMESE_V1"
EDIT_AWARE_FULL_PAIR_METHOD_ID = "FULL_PAIR_EDIT_AWARE_UNRESTRICTED_SIAMESE_V2"
B_ADDITIVE_METHOD_ID = "B_SPARSE_EDIT_SET_ADDITIVE_V1"
B_MATCHED_CONTROL_METHOD_ID = (
    "B_SPARSE_EDIT_SET_POSITION_AGNOSTIC_MATCHED_CONTROL_V1"
)
B_WRONG_LOCAL_METHOD_ID = "B_SPARSE_EDIT_SET_WRONG_POSITION_CONTROL_V1"
B_UNK_ONLY_METHOD_ID = "B_SPARSE_EDIT_SET_CONDITION_UNK_ONLY_V1"


class NeuralMethodRegistryError(RuntimeError):
    """A neural method, architecture, or parameter-match binding is invalid."""


@dataclass(frozen=True, slots=True)
class _MethodSpec:
    """Immutable exact-class crosswalk entry.

    The canonical strings are captured when this module is imported.  Registry
    serialization therefore never consults a caller-mutable factory table or
    trusts metadata reported by a constructed object.
    """

    method_id: str
    architecture_id: str
    model_class: type[nn.Module]
    factory_qualified_name: str
    scientific_role: str
    matched_reference_method_id: str | None
    maximum_parameter_gap_fraction: float | None
    training_variant_id: str
    control_intervention: str


def _method_spec(
    *,
    method_id: str,
    model_class: type[nn.Module],
    scientific_role: str,
    matched_reference_method_id: str | None,
    maximum_parameter_gap_fraction: float | None,
    training_variant_id: str = PRIMARY_TRAINING_VARIANT_ID,
    control_intervention: str = NO_CONTROL_INTERVENTION,
) -> _MethodSpec:
    return _MethodSpec(
        method_id=method_id,
        architecture_id=str(model_class.architecture_id),
        model_class=model_class,
        factory_qualified_name=(
            f"{model_class.__module__}:{model_class.__qualname__}"
        ),
        scientific_role=scientific_role,
        matched_reference_method_id=matched_reference_method_id,
        maximum_parameter_gap_fraction=maximum_parameter_gap_fraction,
        training_variant_id=training_variant_id,
        control_intervention=control_intervention,
    )


_METHOD_SPECS: tuple[_MethodSpec, ...] = (
    _method_spec(
        method_id=FULL_PAIR_METHOD_ID,
        model_class=FullPairUnrestrictedSiamese,
        scientific_role="EXACT_EIGHT_HISTORICAL_POOLED_BASELINE",
        matched_reference_method_id=A0_METHOD_ID,
        maximum_parameter_gap_fraction=MAXIMUM_MATCHED_PARAMETER_GAP,
    ),
    _method_spec(
        method_id=EDIT_AWARE_FULL_PAIR_METHOD_ID,
        model_class=EditAwareFullPairUnrestrictedCritic,
        scientific_role="DEVELOPMENT_EDIT_AWARE_COMPARATOR_NOT_EXACT_EIGHT_ALIAS",
        matched_reference_method_id=A0_METHOD_ID,
        maximum_parameter_gap_fraction=MAXIMUM_MATCHED_PARAMETER_GAP,
    ),
    _method_spec(
        method_id=A0_METHOD_ID,
        model_class=ExactPotentialCritic,
        scientific_role="ROUTE_A0_EXACT_POTENTIAL",
        matched_reference_method_id=None,
        maximum_parameter_gap_fraction=None,
    ),
    _method_spec(
        method_id=A0_CONTROL_METHOD_ID,
        model_class=ConditionAwareParameterMatchedUnrestrictedSiamese,
        scientific_role="ROUTE_A0_NONINTEGRABLE_MATCHED_CONTROL",
        matched_reference_method_id=A0_METHOD_ID,
        maximum_parameter_gap_fraction=MAXIMUM_MATCHED_PARAMETER_GAP,
    ),
    _method_spec(
        method_id=A1_METHOD_ID,
        model_class=BoundedLocalGlobalExactPotentialCritic,
        scientific_role="ROUTE_A1_BOUNDED_LOCAL_EXACT_POTENTIAL",
        matched_reference_method_id=None,
        maximum_parameter_gap_fraction=None,
    ),
    _method_spec(
        method_id=A1_CONTROL_METHOD_ID,
        model_class=BoundedLocalParameterMatchedUnrestrictedCritic,
        scientific_role="ROUTE_A1_NONINTEGRABLE_MATCHED_CONTROL",
        matched_reference_method_id=A1_METHOD_ID,
        maximum_parameter_gap_fraction=MAXIMUM_MATCHED_PARAMETER_GAP,
    ),
    _method_spec(
        method_id=A1_WRONG_LOCAL_METHOD_ID,
        model_class=BoundedLocalWrongPositionControl,
        scientific_role="ROUTE_A1_WRONG_LOCAL_POSITION_CONTROL_DEVELOPMENT_ONLY",
        matched_reference_method_id=A1_METHOD_ID,
        maximum_parameter_gap_fraction=MAXIMUM_MATCHED_PARAMETER_GAP,
        training_variant_id=WRONG_LOCAL_TRAINING_VARIANT_ID,
        control_intervention=WRONG_LOCAL_CONTROL_INTERVENTION,
    ),
    _method_spec(
        method_id=A1_UNK_ONLY_METHOD_ID,
        model_class=BoundedLocalConditionUnknownControl,
        scientific_role="ROUTE_A1_CONDITION_UNK_ONLY_CONTROL_DEVELOPMENT_ONLY",
        matched_reference_method_id=A1_METHOD_ID,
        maximum_parameter_gap_fraction=MAXIMUM_MATCHED_PARAMETER_GAP,
        training_variant_id=CONDITION_UNK_TRAINING_VARIANT_ID,
        control_intervention=CONDITION_UNK_CONTROL_INTERVENTION,
    ),
    _method_spec(
        method_id=B_ADDITIVE_METHOD_ID,
        model_class=ReplayDerivedSparseEditSetAdditiveCritic,
        scientific_role="ROUTE_B_REPLAY_DERIVED_SPARSE_EDIT_SET_ADDITIVE_DEVELOPMENT_ONLY",
        matched_reference_method_id=None,
        maximum_parameter_gap_fraction=None,
    ),
    _method_spec(
        method_id=B_MATCHED_CONTROL_METHOD_ID,
        model_class=ReplayDerivedSparseEditSetPositionAgnosticControl,
        scientific_role="ROUTE_B_POSITION_AGNOSTIC_PARAMETER_MATCHED_CONTROL_DEVELOPMENT_ONLY",
        matched_reference_method_id=B_ADDITIVE_METHOD_ID,
        maximum_parameter_gap_fraction=MAXIMUM_MATCHED_PARAMETER_GAP,
        training_variant_id=POSITION_AGNOSTIC_TRAINING_VARIANT_ID,
        control_intervention=POSITION_AGNOSTIC_CONTROL_INTERVENTION,
    ),
    _method_spec(
        method_id=B_WRONG_LOCAL_METHOD_ID,
        model_class=ReplayDerivedSparseEditSetWrongPositionControl,
        scientific_role="ROUTE_B_WRONG_LOCAL_POSITION_CONTROL_DEVELOPMENT_ONLY",
        matched_reference_method_id=B_ADDITIVE_METHOD_ID,
        maximum_parameter_gap_fraction=MAXIMUM_MATCHED_PARAMETER_GAP,
        training_variant_id=WRONG_LOCAL_TRAINING_VARIANT_ID,
        control_intervention=WRONG_LOCAL_CONTROL_INTERVENTION,
    ),
    _method_spec(
        method_id=B_UNK_ONLY_METHOD_ID,
        model_class=ReplayDerivedSparseEditSetConditionUnknownControl,
        scientific_role="ROUTE_B_CONDITION_UNK_ONLY_CONTROL_DEVELOPMENT_ONLY",
        matched_reference_method_id=B_ADDITIVE_METHOD_ID,
        maximum_parameter_gap_fraction=MAXIMUM_MATCHED_PARAMETER_GAP,
        training_variant_id=CONDITION_UNK_TRAINING_VARIANT_ID,
        control_intervention=CONDITION_UNK_CONTROL_INTERVENTION,
    ),
)

_METHOD_BY_ID: Mapping[str, _MethodSpec] = MappingProxyType(
    {item.method_id: item for item in _METHOD_SPECS}
)
# Kept as a read-only diagnostic view for callers that need to inspect the
# exact classes.  Neither compilation nor canonical serialization trusts a
# separately mutable factory map.
_FACTORIES: Mapping[str, type[nn.Module]] = MappingProxyType(
    {item.method_id: item.model_class for item in _METHOD_SPECS}
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _model_config_payload(cfg: ModelConfig) -> dict[str, int]:
    payload = asdict(cfg)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in payload.values()):
        raise NeuralMethodRegistryError("ModelConfig must contain only frozen integers")
    return payload


def _preallocation_parameter_upper_bound(cfg: ModelConfig) -> int:
    """Conservative integer guard evaluated before constructing torch modules."""

    d = cfg.d_model
    embeddings = d * (
        cfg.vocab_size + cfg.max_len + cfg.n_endpoints + cfg.n_contexts
    )
    transformer = cfg.n_layers * (
        8 * d * d + 4 * d * cfg.ff_dim + 32 * d
    )
    route_heads_and_controls = 256 * d * d + 32 * d * cfg.pair_rank + 100_000
    return embeddings + transformer + route_heads_and_controls


def canonical_registry_payload() -> dict:
    methods = []
    for spec in _METHOD_SPECS:
        methods.append(
            {
                "method_id": spec.method_id,
                "architecture_id": spec.architecture_id,
                "scientific_role": spec.scientific_role,
                "matched_reference_method_id": spec.matched_reference_method_id,
                "maximum_parameter_gap_fraction": (
                    spec.maximum_parameter_gap_fraction
                ),
                "training_variant_id": spec.training_variant_id,
                "control_intervention": spec.control_intervention,
                "factory_id": FACTORY_VERSION,
                "factory_qualified_name": spec.factory_qualified_name,
                "information_set": PAIRED_INFORMATION_SET,
                "primary_pair_policy": PRIMARY_PAIR_POLICY,
                "allowed_primary_edit_counts": [1, 3, 5],
                "maximum_primary_edits": 5,
                "action_budget_enforcement": ACTION_BUDGET_ENFORCEMENT,
                "action_budget_validated_by_compiler": False,
                "ordinary_development_only": True,
                "execution_authorized": False,
                "training_ready": False,
                "publication": False,
                "publication_evidence": False,
            }
        )
    payload = {
        "schema_version": REGISTRY_SCHEMA,
        "factory_version": FACTORY_VERSION,
        "methods": methods,
        "method_order": [item["method_id"] for item in methods],
        "information_set": PAIRED_INFORMATION_SET,
        "action_budget_enforcement": ACTION_BUDGET_ENFORCEMENT,
        "action_budget_validated_by_compiler": False,
        "construction_device": CONSTRUCTION_DEVICE,
        "floating_dtype": FLOATING_DTYPE,
        "preallocation_policy": PREALLOCATION_POLICY,
        "maximum_preallocation_parameter_upper_bound": (
            MAXIMUM_PREALLOCATION_PARAMETER_UPPER_BOUND
        ),
        "maximum_preallocation_parameter_bytes": (
            MAXIMUM_PREALLOCATION_PARAMETER_BYTES
        ),
        "ordinary_development_only": True,
        "internal_test_access": False,
        "sealed_access": False,
        "downstream_unlock": False,
        "execution_authorized": False,
        "training_ready": False,
        "publication": False,
        "publication_evidence": False,
        "b_interaction_status": B_INTERACTION_STATUS,
    }
    return payload


def registry_sha256() -> str:
    return _sha256(canonical_registry_payload())


def _spec(method_id: str) -> _MethodSpec:
    if not isinstance(method_id, str):
        raise NeuralMethodRegistryError("method_id must be a string")
    try:
        return _METHOD_BY_ID[method_id]
    except KeyError as exc:
        raise NeuralMethodRegistryError(
            f"unregistered neural method: {method_id!r}"
        ) from exc


def _require_expected_registry_sha256(expected_registry_sha256: str) -> str:
    if (
        not isinstance(expected_registry_sha256, str)
        or len(expected_registry_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_registry_sha256)
    ):
        raise NeuralMethodRegistryError(
            "expected_registry_sha256 must be a lowercase SHA-256 digest"
        )
    actual = registry_sha256()
    if expected_registry_sha256 != actual:
        raise NeuralMethodRegistryError(
            "caller-provided expected registry SHA-256 does not match the canonical registry"
        )
    return actual


def _verify_exact_instance(
    model: nn.Module,
    spec: _MethodSpec,
    *,
    role: str,
) -> dict[str, object]:
    if type(model) is not spec.model_class:
        raise NeuralMethodRegistryError(
            f"{role} factory returned a non-exact registered class"
        )
    if getattr(model, "architecture_id", None) != spec.architecture_id:
        raise NeuralMethodRegistryError(
            f"{role} model differs from the frozen architecture ID"
        )
    if getattr(model, "information_set", None) != PAIRED_INFORMATION_SET:
        raise NeuralMethodRegistryError(
            f"{role} model differs from the frozen paired information set"
        )

    parameter_dtypes: set[str] = set()
    buffer_dtypes: set[str] = set()
    for name, parameter in model.named_parameters():
        if parameter.device.type != "cpu":
            raise NeuralMethodRegistryError(
                f"{role} parameter {name!r} was not constructed on CPU"
            )
        if parameter.dtype is not torch.float32:
            raise NeuralMethodRegistryError(
                f"{role} parameter {name!r} was not constructed as float32"
            )
        parameter_dtypes.add(str(parameter.dtype))
    for name, buffer in model.named_buffers():
        if buffer.device.type != "cpu":
            raise NeuralMethodRegistryError(
                f"{role} buffer {name!r} was not constructed on CPU"
            )
        if (
            buffer.is_floating_point() or buffer.is_complex()
        ) and buffer.dtype is not torch.float32:
            raise NeuralMethodRegistryError(
                f"{role} floating buffer {name!r} was not constructed as float32"
            )
        buffer_dtypes.add(str(buffer.dtype))
    if not parameter_dtypes:
        raise NeuralMethodRegistryError(f"{role} model has no registered parameters")
    return {
        "device": "cpu",
        "parameter_dtypes": sorted(parameter_dtypes),
        "buffer_dtypes": sorted(buffer_dtypes),
    }


def _construct_cpu_float32(
    candidate_spec: _MethodSpec,
    reference_spec: _MethodSpec | None,
    cfg: ModelConfig,
) -> tuple[nn.Module, nn.Module | None, dict[str, object], dict[str, object] | None]:
    """Construct exact classes only from a qualified CPU/float32 environment."""

    if torch.get_default_dtype() is not torch.float32:
        raise NeuralMethodRegistryError(
            "neural registry construction requires the float32 default dtype"
        )
    default_device = torch.get_default_device()
    if default_device.type != "cpu":
        raise NeuralMethodRegistryError(
            "neural registry construction requires the CPU default device"
        )
    with torch.device("cpu"):
        candidate = candidate_spec.model_class(cfg)
        reference = (
            reference_spec.model_class(cfg)
            if reference_spec is not None
            else None
        )
    candidate_state = _verify_exact_instance(
        candidate, candidate_spec, role="candidate"
    )
    reference_state = None
    if reference is not None and reference_spec is not None:
        reference_state = _verify_exact_instance(
            reference, reference_spec, role="matched reference"
        )
    return candidate, reference, candidate_state, reference_state


def compile_registered_development_model(
    method_id: str,
    cfg: ModelConfig,
    *,
    expected_registry_sha256: str,
) -> tuple[nn.Module, dict[str, object]]:
    """Instantiate the exact registered class and verify its actual parameter claim.

    The returned binding remains development-only and explicitly non-authorizing.
    Callers must not treat its existence as permission to open outcome-bearing
    data or to train a model.
    """

    spec = _spec(method_id)
    bound_registry_sha256 = _require_expected_registry_sha256(
        expected_registry_sha256
    )
    config_payload = _model_config_payload(cfg)
    reference_spec = (
        _spec(spec.matched_reference_method_id)
        if spec.matched_reference_method_id is not None
        else None
    )
    per_model_upper_bound = _preallocation_parameter_upper_bound(cfg)
    preallocation_model_count = 1 + int(reference_spec is not None)
    peak_parameter_upper_bound = per_model_upper_bound * preallocation_model_count
    peak_parameter_bytes_upper_bound = (
        peak_parameter_upper_bound * FLOAT32_PARAMETER_BYTES
    )
    if (
        peak_parameter_upper_bound > MAXIMUM_PREALLOCATION_PARAMETER_UPPER_BOUND
        or peak_parameter_bytes_upper_bound > MAXIMUM_PREALLOCATION_PARAMETER_BYTES
    ):
        raise NeuralMethodRegistryError(
            "candidate-plus-reference configuration exceeds the frozen "
            "pre-allocation parameter-byte guard"
        )
    model, reference, candidate_state, reference_state = _construct_cpu_float32(
        spec, reference_spec, cfg
    )
    architecture_id = spec.architecture_id
    parameter_count = count_parameters(model)
    reference_method_id = spec.matched_reference_method_id
    parameter_match = None
    if reference_method_id is not None and reference is not None:
        try:
            parameter_match = require_parameter_match(
                reference,
                model,
                maximum_gap_fraction=float(spec.maximum_parameter_gap_fraction),
            )
        except ValueError as exc:
            raise NeuralMethodRegistryError(str(exc)) from exc
    model_source = Path(__file__).with_name("models.py")
    registry_source = Path(__file__)
    binding = {
        "schema_version": "r3x_compiled_neural_method_binding_v3",
        "registry_schema": REGISTRY_SCHEMA,
        "registry_sha256": bound_registry_sha256,
        "caller_expected_registry_sha256": expected_registry_sha256,
        "registry_code_sha256": hashlib.sha256(
            registry_source.read_bytes()
        ).hexdigest(),
        "method_id": method_id,
        "architecture_id": architecture_id,
        "factory_id": FACTORY_VERSION,
        "factory_qualified_name": spec.factory_qualified_name,
        "scientific_role": spec.scientific_role,
        "training_variant_id": spec.training_variant_id,
        "control_intervention": spec.control_intervention,
        "information_set": PAIRED_INFORMATION_SET,
        "primary_pair_policy": PRIMARY_PAIR_POLICY,
        "allowed_primary_edit_counts": [1, 3, 5],
        "maximum_primary_edits": 5,
        "action_budget_enforcement": ACTION_BUDGET_ENFORCEMENT,
        "action_budget_validated_by_compiler": False,
        "model_config": config_payload,
        "model_config_sha256": _sha256(config_payload),
        "models_code_sha256": hashlib.sha256(model_source.read_bytes()).hexdigest(),
        "parameter_count": parameter_count,
        "preallocation_parameter_upper_bound": per_model_upper_bound,
        "preallocation_model_count": preallocation_model_count,
        "preallocation_peak_parameter_upper_bound": peak_parameter_upper_bound,
        "preallocation_peak_parameter_bytes_upper_bound": (
            peak_parameter_bytes_upper_bound
        ),
        "maximum_preallocation_parameter_bytes": (
            MAXIMUM_PREALLOCATION_PARAMETER_BYTES
        ),
        "preallocation_policy": PREALLOCATION_POLICY,
        "construction_device": CONSTRUCTION_DEVICE,
        "floating_dtype": FLOATING_DTYPE,
        "candidate_state_tensor_contract": candidate_state,
        "reference_state_tensor_contract": reference_state,
        "matched_reference_method_id": reference_method_id,
        "parameter_match": parameter_match,
        "ordinary_development_only": True,
        "internal_test_access": False,
        "sealed_access": False,
        "downstream_unlock": False,
        "execution_authorized": False,
        "training_ready": False,
        "publication": False,
        "publication_evidence": False,
        "b_interaction_status": B_INTERACTION_STATUS,
    }
    return model, binding


__all__ = [
    "A0_CONTROL_METHOD_ID",
    "A0_METHOD_ID",
    "A1_CONTROL_METHOD_ID",
    "A1_METHOD_ID",
    "A1_UNK_ONLY_METHOD_ID",
    "A1_WRONG_LOCAL_METHOD_ID",
    "B_ADDITIVE_METHOD_ID",
    "B_INTERACTION_STATUS",
    "B_MATCHED_CONTROL_METHOD_ID",
    "B_UNK_ONLY_METHOD_ID",
    "B_WRONG_LOCAL_METHOD_ID",
    "CONDITION_UNK_TRAINING_VARIANT_ID",
    "EDIT_AWARE_FULL_PAIR_METHOD_ID",
    "FACTORY_VERSION",
    "FULL_PAIR_METHOD_ID",
    "MAXIMUM_MATCHED_PARAMETER_GAP",
    "NeuralMethodRegistryError",
    "POSITION_AGNOSTIC_TRAINING_VARIANT_ID",
    "PRIMARY_TRAINING_VARIANT_ID",
    "REGISTRY_SCHEMA",
    "WRONG_LOCAL_TRAINING_VARIANT_ID",
    "canonical_registry_payload",
    "compile_registered_development_model",
    "registry_sha256",
]
