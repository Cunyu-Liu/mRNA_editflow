"""Typed, development-safe non-data authority projections for R3X.

These artifacts are generated outside the training process.  This module does
not read a full scientific authority and does not construct a projection from
one.  It validates, publishes, and exactly reloads seven deliberately minimal
typed projections whose allowlists exclude designated row-, sequence-, and
result-bearing fields.  That is a bounded capability, not proof that arbitrary
producer-chosen strings are semantically harmless: activation remains not
training-ready and relies on independent pre-outcome snapshots and typed row
authority replay.

The code-tree projection is the sole authority that refers to other files.
Those references are restricted to ordinary repository code beneath the
caller-provided run root and their exact bytes are replayed on both write and
load.  Data, outputs, checkpoints, results, tests, path traversal, symlinks,
and hard links are refused.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import stat
from pathlib import Path
from types import MappingProxyType
from typing import AbstractSet, Any, Callable, Mapping

from .baseline_selection import SELECTION_SCHEMA as DEVELOPMENT_SELECTOR_SCHEMA
from .artifact_io import (
    ArtifactIOError,
    canonical_json_bytes,
)
from .protocol import (
    AUTHORIZED_ROUTE,
    AUTHORIZED_SEQUENCE,
    REQUIRED_NEGATIVE_CONTROLS,
    ProtocolHoldError,
    canonical_sha256,
)


TYPED_DEVELOPMENT_SAFE = "TYPED_DEVELOPMENT_SAFE"
TYPED_DEVELOPMENT_SAFE_BUNDLE_NOT_TRAINING_READY = (
    "TYPED_DEVELOPMENT_SAFE_BUNDLE_NOT_TRAINING_READY"
)
SIGN_ACCURACY_GATE = 0.60
FROZEN_SEEDS = (11, 23, 42, 67, 101)
FROZEN_NEGATIVE_CONTROLS = tuple(REQUIRED_NEGATIVE_CONTROLS)

ROUTE_AUTHORIZATION = "route_authorization"
ENDPOINT_ESTIMAND_REGISTRY = "endpoint_estimand_registry"
METRIC_ZERO_POLICY_REGISTRY = "metric_zero_policy_registry"
MODEL_OPTIMIZER_SEED_CONFIG = "model_optimizer_seed_config"
ANALYSIS_CODE_TREE_MANIFEST = "analysis_code_tree_manifest"
METHOD_PROTOCOL = "method_protocol"
PAIRED_BASELINE_POLICY = "paired_strongest_baseline_comparison_policy"

SUPPORTED_AUTHORITIES = (
    ROUTE_AUTHORIZATION,
    ENDPOINT_ESTIMAND_REGISTRY,
    METRIC_ZERO_POLICY_REGISTRY,
    MODEL_OPTIMIZER_SEED_CONFIG,
    ANALYSIS_CODE_TREE_MANIFEST,
    METHOD_PROTOCOL,
    PAIRED_BASELINE_POLICY,
)

SCHEMAS = MappingProxyType({
    name: f"r3x_typed_safe_{name}_v1" for name in SUPPORTED_AUTHORITIES
})
ROLES = MappingProxyType({
    name: f"TYPED_DEVELOPMENT_SAFE_{name.upper()}"
    for name in SUPPORTED_AUTHORITIES
})

DEVELOPMENT_PROJECTION_HASH_KEYS = {
    "inclusion_manifest_development_projection",
    "context_source_crosswalk_development_projection",
    "source_component_split_manifest_development_projection",
}

COMMON_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "authority_name",
    "source_authority_sha256",
    "train_development_outcomes_sha256",
    "development_projection_hashes",
    "sign_accuracy_gate",
    "sign_gate_change_authorized",
    "ordinary_only",
    "training_ready",
    "sealed_access",
    "downstream_unlock",
}

ROUTE_KEYS = COMMON_KEYS | {
    "authorization_id",
    "selected_route",
    "authorized_sequence",
    "code_implementation_authorized",
    "ordinary_training_authorized_when_all_inputs_frozen",
    "c3_positioning",
    "hard_boundaries",
}
ROUTE_BOUNDARY_KEYS = {
    "ordinary_only",
    "sealed_member_access",
    "formal_sealed_evaluator",
    "sign_accuracy_gate",
    "sign_gate_change_authorized",
    "post_hoc_metric_threshold_seed_or_endpoint_selection",
    "legacy_signed_ratio_enrichment",
    "o0_g1_unlock",
    "authority_contract_overwrite",
    "active_contract_mutation",
}

ESTIMAND_KEYS = COMMON_KEYS | {"entries"}
ESTIMAND_ENTRY_KEYS = {
    "study",
    "endpoint",
    "estimand_id",
    "raw_source_field",
    "raw_candidate_field",
    "raw_unit",
    "direction",
    "transform",
    "valid_domain",
    "role",
    "replicate_policy",
}

METRIC_KEYS = COMMON_KEYS | {
    "zero_policy",
    "primary_cell_definition",
    "component_weighting",
    "ranking_aggregation",
    "undefined_primary_cell_policy",
    "seed_policy",
    "selection_tie_policy",
    "primary_sign_gate",
}

CONFIG_KEYS = COMMON_KEYS | {
    "seeds",
    "seed_aggregation",
    "estimand_training_scope",
    "condition_unknown_dropout",
    "optimizer",
    "loss",
    "checkpoint_selection",
    "model",
    "frozen_environment_execution_qualification_sha256",
    "histogram_gradient_boosting_execution_qualification_asserted",
    "execution_qualification_replay_status",
    "foundation_baseline_status",
}
OPTIMIZER_KEYS = {"name", "learning_rate", "weight_decay"}
LOSS_KEYS = {"huber_weight", "nll_weight", "sign_weight", "sign_temperature"}
CHECKPOINT_SELECTION_KEYS = {"rule", "selection_partition"}
MODEL_KEYS = {
    "architecture_id",
    "max_sequence_length",
    "embedding_dim",
    "hidden_dim",
    "num_layers",
    "dropout",
}

CODE_KEYS = COMMON_KEYS | {
    "code_root_relative_path",
    "files",
    "tree_sha256",
    "trusted_code_snapshot_sha256",
}
CODE_ENTRY_KEYS = {"path", "sha256"}
ALLOWED_CODE_EXTENSIONS = {".py", ".pyi"}
ALLOWED_CODE_TOP_LEVEL = {"scripts", "src", "models"}
FORBIDDEN_CODE_PARTS = {
    ".git",
    "data",
    "dataset",
    "datasets",
    "output",
    "outputs",
    "checkpoint",
    "checkpoints",
    "result",
    "results",
    "prediction",
    "predictions",
    "artifact",
    "artifacts",
    "run",
    "runs",
    "sealed",
    "restricted",
    "test",
    "tests",
    "__pycache__",
}

METHOD_KEYS = COMMON_KEYS | {
    "selected_route",
    "execution_order",
    "seeds",
    "information_set",
    "estimand_training_scope",
    "method_ids",
    "negative_controls",
    "training_policy",
    "compute_matching",
    "stop_rules",
    "hard_boundaries",
}
TRAINING_POLICY_KEYS = {
    "sampler",
    "mean_loss",
    "uncertainty_loss",
    "sign_probability",
    "sign_bce_variance_gradient",
    "hard_sign_rule",
    "rank_loss",
    "checkpoint_selection",
    "selection_partition",
}
COMPUTE_MATCHING_KEYS = {
    "same_training_rows",
    "same_seeds",
    "same_epoch_and_optimizer_budget",
    "a_control_parameter_difference_max_fraction",
    "report_parameter_count",
    "report_total_forward_equivalents",
    "report_wall_time_and_peak_memory",
}
METHOD_BOUNDARY_KEYS = {
    "ordinary_only",
    "sealed_member_access",
    "formal_sealed_evaluator",
    "O0_G1_E0_unlock",
    "active_contract_mutation",
    "sign_gate_change_authorized",
}
MINIMUM_METHOD_IDS = frozenset({
    "ZERO_DELTA",
    "SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR",
    "EDIT_ONLY_RIDGE",
    "CANDIDATE_KMER_RIDGE",
    "PAIRED_LOCAL_GLOBAL_KMER_RIDGE",
    "PAIRED_FEATURE_NEAREST_NEIGHBOR",
    "HISTOGRAM_GRADIENT_BOOSTING",
    "FULL_PAIR_UNRESTRICTED_SIAMESE",
    "A_EXACT_POTENTIAL_DIFFERENCE",
    "A_PARAMETER_MATCHED_UNRESTRICTED_CONTROL",
    "B_SPARSE_EDIT_SET_ADDITIVE",
})
ALLOWED_METHOD_IDS = MINIMUM_METHOD_IDS | frozenset({
    "EDIT_ONLY_SIGN_LOGISTIC_DIAGNOSTIC",
    "R2_STYLE_NO_EXPLICIT_STUDY_ADAPTATION",
    "B_SPARSE_EDIT_SET_INTERACTION",
})
FROZEN_STOP_RULES = (
    "ANY_PROTECTED_INPUT",
    "ANY_COMPONENT_PARTITION_OVERLAP",
    "INCOMPLETE_TYPED_AUTHORITY",
    "A_INVARIANT_OR_MATCHED_CONTROL_FAILURE",
    "B_CONTROL_OR_ADDITIVE_FAILURE",
    "PRIMARY_SIGN_GATE_BELOW_0_60",
    "POST_HOC_CHANGE_REQUIRED",
)

PAIRED_KEYS = COMMON_KEYS | {
    "policy_id",
    "baseline_suite_status",
    "primary_candidate_model_id",
    "information_set",
    "comparison_family_id",
    "comparison_family_member_ids",
    "primary_comparison_member_id",
    "eligible_baseline_method_ids",
    "required_baseline_method_ids",
    "development_selector_required",
    "development_selector_schema",
    "frozen_environment_execution_qualification_sha256",
    "histogram_gradient_boosting_execution_qualification_asserted",
    "execution_qualification_replay_status",
    "foundation_baseline_status",
    "primary_sign_inference",
}
INFERENCE_KEYS = {"metric_id", "alpha", "confidence", "bootstrap", "permutation"}
BOOTSTRAP_KEYS = {
    "n_resamples",
    "seed",
    "min_components_per_cell",
    "max_attempt_multiplier",
}
PERMUTATION_KEYS = {
    "n_permutations",
    "seed",
    "exact_component_limit",
    "min_components_per_cell",
}
FROZEN_BASELINE_METHOD_ORDER = (
    "ZERO_DELTA",
    "SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR",
    "EDIT_ONLY_RIDGE",
    "CANDIDATE_KMER_RIDGE",
    "PAIRED_LOCAL_GLOBAL_KMER_RIDGE",
    "PAIRED_FEATURE_NEAREST_NEIGHBOR",
    "HISTOGRAM_GRADIENT_BOOSTING",
    "FULL_PAIR_UNRESTRICTED_SIAMESE",
)
MINIMUM_BASELINE_METHOD_IDS = frozenset(FROZEN_BASELINE_METHOD_ORDER)

HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
SAFE_ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.json$")
PROHIBITED_STRING = re.compile(
    r"(?:internal[_ -]?test|test[_ -]?outcome|observed[_ -]?(?:result|outcome|value)|"
    r"ground[_ -]?truth|gse246381|(?<![a-z0-9])restricted(?![a-z0-9])|"
    r"sealed[_ -]?(?:member|data|path))",
    re.IGNORECASE,
)
MAX_CODE_FILE_BYTES = 2 * 1024 * 1024
MAX_AUTHORITY_BYTES = 4 * 1024 * 1024
MAX_IDENTIFIER_LENGTH = 160
MAX_ESTIMAND_ENTRIES = 64
MAX_CODE_FILES = 4096
OPEN_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd
LINK_SUPPORTS_DIR_FD = os.link in os.supports_dir_fd
UNLINK_SUPPORTS_DIR_FD = os.unlink in os.supports_dir_fd
BUNDLE_REF_KEYS = {"path", "sha256"}
BUNDLE_SHARED_FIELDS = (
    "train_development_outcomes_sha256",
    "development_projection_hashes",
    "sign_accuracy_gate",
    "sign_gate_change_authorized",
    "ordinary_only",
    "training_ready",
    "sealed_access",
    "downstream_unlock",
)
REMAINING_TYPED_ROW_AUTHORITIES = [
    "inclusion_manifest_development_projection",
    "context_source_crosswalk_development_projection",
    "source_component_split_manifest_development_projection",
]


class TypedSafeAuthorityError(ProtocolHoldError):
    """Structured fail-closed error for typed safe projections."""


def _hold(code: str, message: str, **details: Any) -> None:
    raise TypedSafeAuthorityError(code, message, details=details or None)


def _strict_keys(value: object, expected: AbstractSet[str], *, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        _hold("HOLD_TYPED_SAFE_SCHEMA", f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        _hold(
            "HOLD_TYPED_SAFE_SCHEMA",
            f"{label} keys differ from the exact schema",
            missing=sorted(expected - actual),
            unexpected=sorted(actual - expected),
        )
    return value


def _exact_match(value: object, expected: object) -> bool:
    """Compare policy literals without Python's bool/int equality aliasing."""

    if isinstance(expected, Mapping):
        return (
            isinstance(value, Mapping)
            and set(value) == set(expected)
            and all(_exact_match(value[key], nested) for key, nested in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(value, list)
            and len(value) == len(expected)
            and all(_exact_match(item, frozen) for item, frozen in zip(value, expected))
        )
    if isinstance(expected, bool):
        return value is expected
    if isinstance(expected, float):
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) == expected
        )
    return type(value) is type(expected) and value == expected


def _require_exact(value: object, expected: object, *, label: str) -> None:
    if not _exact_match(value, expected):
        _hold("HOLD_TYPED_SAFE_SCHEMA", f"{label} differs from the exact policy")


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        _hold("HOLD_TYPED_SAFE_SCHEMA", f"{label} must be lowercase SHA-256")
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        _hold("HOLD_TYPED_SAFE_SCHEMA", f"{label} is not a safe identifier")
    if len(value) > MAX_IDENTIFIER_LENGTH:
        _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", f"{label} is too long")
    if HEX64.fullmatch(value):
        _hold(
            "HOLD_TYPED_SAFE_PROHIBITED_CONTENT",
            f"{label} cannot carry a digest-shaped row identifier",
        )
    if PROHIBITED_STRING.search(value):
        _hold("HOLD_TYPED_SAFE_PROHIBITED_CONTENT", f"{label} names prohibited support")
    return value


def _finite_number(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
    open_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _hold("HOLD_TYPED_SAFE_SCHEMA", f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        _hold("HOLD_TYPED_SAFE_SCHEMA", f"{label} must be finite")
    if minimum is not None and (
        number < minimum or (open_minimum and number == minimum)
    ):
        _hold("HOLD_TYPED_SAFE_SCHEMA", f"{label} is below its supported range")
    if maximum is not None and number > maximum:
        _hold("HOLD_TYPED_SAFE_SCHEMA", f"{label} exceeds its supported range")
    return number


def _bounded_int(
    value: object,
    *,
    label: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _hold("HOLD_TYPED_SAFE_SCHEMA", f"{label} must be an integer")
    if not minimum <= value <= maximum:
        _hold("HOLD_TYPED_SAFE_SCHEMA", f"{label} is outside its supported range")
    return value


def _safe_id_list(
    value: object,
    *,
    label: str,
    sorted_required: bool = True,
    maximum: int = 128,
) -> list[str]:
    if not isinstance(value, list) or not value:
        _hold("HOLD_TYPED_SAFE_SCHEMA", f"{label} must be a non-empty list")
    if len(value) > maximum:
        _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", f"{label} is too large")
    normalized = [_identifier(item, label=f"{label} item") for item in value]
    if len(normalized) != len(set(normalized)):
        _hold("HOLD_TYPED_SAFE_SCHEMA", f"{label} must be unique")
    if sorted_required and normalized != sorted(normalized):
        _hold("HOLD_TYPED_SAFE_SCHEMA", f"{label} must be sorted")
    return normalized


def _reject_prohibited_strings(value: object, *, path: str = "projection") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_prohibited_strings(nested, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_prohibited_strings(nested, path=f"{path}[{index}]")
    elif isinstance(value, str) and PROHIBITED_STRING.search(value):
        _hold(
            "HOLD_TYPED_SAFE_PROHIBITED_CONTENT",
            "typed projection contains prohibited test/result support",
            field_path=path,
        )


def _validate_common(
    payload: Mapping,
    *,
    authority_name: str,
    expected_keys: AbstractSet[str],
) -> None:
    _strict_keys(payload, expected_keys, label=authority_name)
    if authority_name not in SUPPORTED_AUTHORITIES:
        _hold(
            "HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY",
            f"unsupported typed authority: {authority_name}",
        )
    expected = {
        "schema_version": SCHEMAS[authority_name],
        "artifact_role": ROLES[authority_name],
        "status": TYPED_DEVELOPMENT_SAFE,
        "authority_name": authority_name,
        "sign_gate_change_authorized": False,
        "ordinary_only": True,
        "training_ready": False,
        "sealed_access": False,
        "downstream_unlock": False,
    }
    for field, value in expected.items():
        if not _exact_match(payload.get(field), value):
            _hold("HOLD_TYPED_SAFE_SCHEMA", f"{authority_name} {field} must be {value!r}")
    gate = _finite_number(
        payload.get("sign_accuracy_gate"),
        label="sign_accuracy_gate",
        minimum=SIGN_ACCURACY_GATE,
        maximum=SIGN_ACCURACY_GATE,
    )
    if gate != SIGN_ACCURACY_GATE:
        _hold("HOLD_SIGN_GATE_MUTATION", "sign gate must remain exactly 0.60")
    source_sha = _digest(payload.get("source_authority_sha256"), label="source authority")
    train_sha = _digest(
        payload.get("train_development_outcomes_sha256"),
        label="train/development outcomes",
    )
    projections = _strict_keys(
        payload.get("development_projection_hashes"),
        DEVELOPMENT_PROJECTION_HASH_KEYS,
        label="development_projection_hashes",
    )
    projection_hashes = [
        _digest(projections[name], label=f"development projection {name}")
        for name in sorted(projections)
    ]
    if len({source_sha, train_sha, *projection_hashes}) != 5:
        _hold("HOLD_TYPED_SAFE_BINDING", "authority and development bindings must be distinct")
    _reject_prohibited_strings(payload)


def _validate_baseline_execution_qualification(payload: Mapping) -> None:
    qualification_sha = _digest(
        payload.get("frozen_environment_execution_qualification_sha256"),
        label="frozen environment execution qualification",
    )
    bound_digests = {
        payload["source_authority_sha256"],
        payload["train_development_outcomes_sha256"],
        *payload["development_projection_hashes"].values(),
    }
    if qualification_sha in bound_digests:
        _hold(
            "HOLD_TYPED_SAFE_BINDING",
            "execution qualification must bind an independent frozen certificate",
        )
    if (
        payload.get("histogram_gradient_boosting_execution_qualification_asserted")
        is not True
    ):
        _hold(
            "HOLD_BASELINE_SUITE_NOT_QUALIFIED",
            "histogram gradient boosting must be frozen executable before suite completion",
        )
    if (
        payload.get("execution_qualification_replay_status")
        != "CERTIFICATE_HASH_BOUND_NOT_REPLAYED"
    ):
        _hold(
            "HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY",
            "execution qualification capability must remain honestly non-replayed",
        )
    if (
        payload.get("foundation_baseline_status")
        != "NOT_RUN_EXPOSURE_LICENSE_QUALIFICATION_UNKNOWN"
    ):
        _hold(
            "HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY",
            "foundation baseline must remain NOT_RUN while exposure/license is unknown",
        )


def validate_route_authorization(payload: Mapping) -> dict:
    _validate_common(payload, authority_name=ROUTE_AUTHORIZATION, expected_keys=ROUTE_KEYS)
    _identifier(payload.get("authorization_id"), label="authorization_id")
    if payload.get("selected_route") != AUTHORIZED_ROUTE:
        _hold("HOLD_TYPED_SAFE_SCHEMA", "route authorization selects a different route")
    if payload.get("authorized_sequence") != list(AUTHORIZED_SEQUENCE):
        _hold("HOLD_TYPED_SAFE_SCHEMA", "authorized sequence differs from the frozen route")
    if payload.get("code_implementation_authorized") is not True:
        _hold("HOLD_TYPED_SAFE_SCHEMA", "code implementation must be authorized")
    if payload.get("ordinary_training_authorized_when_all_inputs_frozen") is not True:
        _hold(
            "HOLD_TYPED_SAFE_SCHEMA",
            "ordinary training authorization must remain conditional on frozen inputs",
        )
    if (
        payload.get("c3_positioning")
        != "PROPOSED_NOT_ACTIVE_DGM_DOOB_DERIVED_ADAPTATION"
    ):
        _hold(
            "HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY",
            "only the frozen DGM/Doob-derived C3 positioning is supported",
        )
    boundaries = _strict_keys(
        payload.get("hard_boundaries"), ROUTE_BOUNDARY_KEYS, label="route hard_boundaries"
    )
    expected_boundaries = {
        "ordinary_only": True,
        "sealed_member_access": False,
        "formal_sealed_evaluator": False,
        "sign_accuracy_gate": SIGN_ACCURACY_GATE,
        "sign_gate_change_authorized": False,
        "post_hoc_metric_threshold_seed_or_endpoint_selection": False,
        "legacy_signed_ratio_enrichment": "PROHIBITED",
        "o0_g1_unlock": False,
        "authority_contract_overwrite": False,
        "active_contract_mutation": False,
    }
    _require_exact(
        boundaries,
        expected_boundaries,
        label="route hard boundaries",
    )
    return dict(payload)


def validate_endpoint_estimand_registry(payload: Mapping) -> dict:
    _validate_common(
        payload,
        authority_name=ENDPOINT_ESTIMAND_REGISTRY,
        expected_keys=ESTIMAND_KEYS,
    )
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        _hold("HOLD_TYPED_SAFE_SCHEMA", "estimand entries must be non-empty")
    if len(entries) > MAX_ESTIMAND_ENTRIES:
        _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "estimand registry is too large")
    normalized_order: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(entries):
        entry = _strict_keys(raw, ESTIMAND_ENTRY_KEYS, label=f"estimand entry {index}")
        study = _identifier(entry.get("study"), label="study")
        endpoint = _identifier(entry.get("endpoint"), label="endpoint")
        estimand = _identifier(entry.get("estimand_id"), label="estimand_id")
        key = (study, endpoint)
        if key in seen:
            _hold("HOLD_TYPED_SAFE_SCHEMA", "study/endpoint estimand keys must be unique")
        seen.add(key)
        normalized_order.append((study, endpoint, estimand))
        for field in ("raw_source_field", "raw_candidate_field", "raw_unit"):
            _identifier(entry.get(field), label=field)
        if entry.get("direction") not in {
            "HIGHER_IS_MORE_STABLE",
            "LOWER_IS_MORE_STABLE",
        }:
            _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "unsupported estimand direction")
        if entry.get("transform") != "IDENTITY_DIFFERENCE":
            _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "unsupported estimand transform")
        if entry.get("valid_domain") != "FINITE":
            _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "unsupported estimand domain")
        if entry.get("role") != "PRIMARY":
            _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "only primary estimands are supported")
        if entry.get("replicate_policy") != "SINGLE_OBSERVATION_ONLY":
            _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "unsupported replicate policy")
    if normalized_order != sorted(normalized_order):
        _hold("HOLD_TYPED_SAFE_SCHEMA", "estimand entries must be canonically sorted")
    return dict(payload)


def validate_metric_zero_policy_registry(payload: Mapping) -> dict:
    _validate_common(
        payload,
        authority_name=METRIC_ZERO_POLICY_REGISTRY,
        expected_keys=METRIC_KEYS,
    )
    expected = {
        "zero_policy": "EXCLUDE_TRUE_ZERO",
        "primary_cell_definition": "STUDY_ESTIMAND_PUBLICATION_CONTEXT_EDIT_COUNT",
        "component_weighting": "EQUAL_COMPONENT_THEN_EQUAL_ROW_WITHIN_COMPONENT",
        "ranking_aggregation": "WITHIN_SOURCE_POOL_THEN_CELL_MACRO",
        "undefined_primary_cell_policy": "HOLD_PRIMARY_CELL_UNDEFINED",
        "seed_policy": "FIXED_FIVE_MODEL_PREDICTION_ENSEMBLE",
        "selection_tie_policy": "EXPECTED_OUTCOME_OVER_CUTOFF_TIE_BLOCK",
        "primary_sign_gate": SIGN_ACCURACY_GATE,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", f"unsupported metric policy: {field}")
    return dict(payload)


def validate_model_optimizer_seed_config(payload: Mapping) -> dict:
    _validate_common(
        payload,
        authority_name=MODEL_OPTIMIZER_SEED_CONFIG,
        expected_keys=CONFIG_KEYS,
    )
    if payload.get("seeds") != list(FROZEN_SEEDS):
        _hold("HOLD_TYPED_SAFE_SCHEMA", "optimizer config must freeze five seeds")
    if payload.get("seed_aggregation") != "FIXED_PREDICTION_ENSEMBLE":
        _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "unsupported seed aggregation")
    if payload.get("estimand_training_scope") != "INDEPENDENT_MODEL_PER_PRIMARY_ESTIMAND":
        _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "unsupported estimand training scope")
    _validate_baseline_execution_qualification(payload)
    _finite_number(
        payload.get("condition_unknown_dropout"),
        label="condition_unknown_dropout",
        minimum=0.0,
        maximum=1.0,
        open_minimum=True,
    )
    optimizer = _strict_keys(payload.get("optimizer"), OPTIMIZER_KEYS, label="optimizer")
    if optimizer.get("name") != "Adam":
        _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "only Adam is supported")
    _finite_number(
        optimizer.get("learning_rate"), label="learning_rate", minimum=0.0, maximum=1.0, open_minimum=True
    )
    _finite_number(optimizer.get("weight_decay"), label="weight_decay", minimum=0.0, maximum=1.0)
    loss = _strict_keys(payload.get("loss"), LOSS_KEYS, label="loss")
    _finite_number(loss.get("huber_weight"), label="huber_weight", minimum=0.0, maximum=100.0)
    _finite_number(loss.get("nll_weight"), label="nll_weight", minimum=0.0, maximum=100.0)
    _finite_number(loss.get("sign_weight"), label="sign_weight", minimum=0.0, maximum=100.0)
    _finite_number(
        loss.get("sign_temperature"), label="sign_temperature", minimum=0.0, maximum=100.0, open_minimum=True
    )
    selection = _strict_keys(
        payload.get("checkpoint_selection"),
        CHECKPOINT_SELECTION_KEYS,
        label="checkpoint_selection",
    )
    if selection.get("rule") != "SIGN_GATE_THEN_SPEARMAN_THEN_NLL_THEN_EARLIEST_EPOCH":
        _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "unsupported checkpoint selection rule")
    if selection.get("selection_partition") != "development":
        _hold("HOLD_TYPED_SAFE_PROHIBITED_CONTENT", "checkpoint selection must be development-only")
    model = _strict_keys(payload.get("model"), MODEL_KEYS, label="model")
    _identifier(model.get("architecture_id"), label="architecture_id")
    _bounded_int(model.get("max_sequence_length"), label="max_sequence_length", minimum=1, maximum=8192)
    _bounded_int(model.get("embedding_dim"), label="embedding_dim", minimum=1, maximum=16384)
    _bounded_int(model.get("hidden_dim"), label="hidden_dim", minimum=1, maximum=65536)
    _bounded_int(model.get("num_layers"), label="num_layers", minimum=1, maximum=128)
    _finite_number(model.get("dropout"), label="model dropout", minimum=0.0, maximum=1.0)
    return dict(payload)


def _checked_run_root(run_root: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(run_root)))
    if root.is_symlink() or not root.is_dir():
        _hold("HOLD_TYPED_SAFE_PATH", "run root must be an existing real directory")
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        _hold("HOLD_TYPED_SAFE_PATH", f"run root cannot be resolved: {exc}")
    if root != resolved:
        _hold("HOLD_TYPED_SAFE_PATH", "run root cannot traverse symlinks")
    return root


def _path_beneath(root: Path, path: Path, *, label: str) -> Path:
    candidate = Path(os.path.abspath(os.fspath(path)))
    try:
        candidate.relative_to(root)
    except ValueError:
        _hold("HOLD_TYPED_SAFE_PATH", f"{label} must remain beneath the run root")
    return candidate


def _assert_no_symlink_components(root: Path, path: Path, *, label: str) -> None:
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            _hold("HOLD_TYPED_SAFE_PATH", f"{label} path is unavailable: {exc}")
        if stat.S_ISLNK(metadata.st_mode):
            _hold("HOLD_TYPED_SAFE_PATH", f"{label} cannot traverse a symlink")


def _validate_code_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        _hold("HOLD_TYPED_SAFE_SCHEMA", "code path must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        _hold("HOLD_TYPED_SAFE_PATH", "code paths must be normalized safe relative paths")
    if not path.parts or path.parts[0] not in ALLOWED_CODE_TOP_LEVEL:
        _hold("HOLD_TYPED_SAFE_PATH", "code path is outside the supported repository code roots")
    lowered = {part.lower() for part in path.parts}
    if lowered & FORBIDDEN_CODE_PARTS:
        _hold("HOLD_TYPED_SAFE_PROHIBITED_CONTENT", "code path names data/result/test support")
    if path.suffix.lower() not in ALLOWED_CODE_EXTENSIONS:
        _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "unsupported code file extension")
    return path


def _normalize_trusted_code_snapshot(snapshot: Mapping | None) -> list[dict[str, str]]:
    if snapshot is None:
        _hold(
            "HOLD_EXTERNAL_CODE_SNAPSHOT_REVIEW_REQUIRED",
            "code-tree safety requires an independently frozen path/hash allowlist",
        )
    if not isinstance(snapshot, Mapping) or not snapshot:
        _hold(
            "HOLD_EXTERNAL_CODE_SNAPSHOT_REVIEW_REQUIRED",
            "trusted code snapshot must be a non-empty external mapping",
        )
    if len(snapshot) > MAX_CODE_FILES:
        _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "trusted code snapshot is too large")
    normalized: list[dict[str, str]] = []
    for raw_path, raw_digest in snapshot.items():
        relative = _validate_code_path(raw_path)
        normalized.append(
            {
                "path": relative.as_posix(),
                "sha256": _digest(
                    raw_digest,
                    label=f"trusted code file {relative.as_posix()}",
                ),
            }
        )
    normalized.sort(key=lambda item: item["path"])
    if len({item["path"] for item in normalized}) != len(normalized):
        _hold("HOLD_TYPED_SAFE_SCHEMA", "trusted code snapshot paths must be unique")
    return normalized


def _open_absolute_directory_nofollow(path: Path) -> int:
    if (
        not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
        or not OPEN_SUPPORTS_DIR_FD
    ):
        _hold(
            "HOLD_EXTERNAL_CODE_SNAPSHOT_REVIEW_REQUIRED",
            "platform lacks descriptor-relative no-follow directory replay",
        )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = -1
    try:
        descriptor = os.open(os.sep, flags)
        for part in path.parts[1:]:
            next_descriptor = os.open(
                part,
                flags,
                dir_fd=descriptor,
            )
            metadata = os.fstat(next_descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(next_descriptor)
                _hold("HOLD_TYPED_SAFE_PATH", "code root component is not a directory")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except TypedSafeAuthorityError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        _hold("HOLD_TYPED_SAFE_PATH", f"code root cannot be opened safely: {exc}")


def _open_directory_beneath_nofollow(root: Path, parts: tuple[str, ...]) -> int:
    descriptor = _open_absolute_directory_nofollow(root)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        for part in parts:
            if not part or part in {".", ".."} or os.sep in part:
                _hold("HOLD_TYPED_SAFE_PATH", "directory component is not safe")
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            metadata = os.fstat(next_descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(next_descriptor)
                _hold("HOLD_TYPED_SAFE_PATH", "namespace component is not a directory")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except TypedSafeAuthorityError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        _hold("HOLD_TYPED_SAFE_PATH", f"namespace cannot be opened safely: {exc}")


def _read_regular_file_at_exact(
    parent_descriptor: int,
    name: str,
    expected_sha256: str,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = -1
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            _hold("HOLD_TYPED_SAFE_PATH", f"{label} must be a regular single-link file")
        if before.st_size > maximum_bytes:
            _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", f"{label} is too large")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1 << 20, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", f"{label} is too large")
        after = os.fstat(descriptor)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or after.st_nlink != 1
        ):
            _hold("HOLD_TYPED_SAFE_PATH", f"{label} changed during exact read")
        raw = b"".join(chunks)
    except TypedSafeAuthorityError:
        raise
    except OSError as exc:
        _hold("HOLD_TYPED_SAFE_PATH", f"{label} cannot be opened safely: {exc}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        _hold("HOLD_TYPED_SAFE_HASH_MISMATCH", f"{label} SHA-256 mismatch")
    return raw


def _verify_namespace_binding(
    root: Path,
    parts: tuple[str, ...],
    pinned_descriptor: int,
) -> None:
    try:
        replay_descriptor = _open_directory_beneath_nofollow(root, parts)
    except TypedSafeAuthorityError as exc:
        _hold(
            "HOLD_UNSUPPORTED_ON_MUTABLE_NAMESPACE",
            f"artifact namespace changed during operation: {exc}",
        )
    try:
        pinned = os.fstat(pinned_descriptor)
        replayed = os.fstat(replay_descriptor)
        if pinned.st_dev != replayed.st_dev or pinned.st_ino != replayed.st_ino:
            _hold(
                "HOLD_UNSUPPORTED_ON_MUTABLE_NAMESPACE",
                "artifact namespace directory identity changed during operation",
            )
    finally:
        os.close(replay_descriptor)


def _secure_read_authority_artifact(
    root: Path,
    candidate: Path,
    expected_sha256: str,
) -> bytes:
    parent_parts = ("development", "authorities")
    parent_descriptor = _open_directory_beneath_nofollow(root, parent_parts)
    try:
        raw = _read_regular_file_at_exact(
            parent_descriptor,
            candidate.name,
            expected_sha256,
            maximum_bytes=MAX_AUTHORITY_BYTES,
            label="typed safe authority artifact",
        )
        _verify_namespace_binding(root, parent_parts, parent_descriptor)
        return raw
    finally:
        os.close(parent_descriptor)


def _write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise OSError("short write while publishing typed authority")
        written += count


def _secure_publish_authority_artifact(
    root: Path,
    candidate: Path,
    raw: bytes,
) -> str:
    if not LINK_SUPPORTS_DIR_FD or not UNLINK_SUPPORTS_DIR_FD:
        _hold(
            "HOLD_UNSUPPORTED_ON_MUTABLE_NAMESPACE",
            "platform lacks descriptor-relative no-replace artifact publication",
        )
    expected_sha256 = hashlib.sha256(raw).hexdigest()
    parent_parts = ("development", "authorities")
    parent_descriptor = _open_directory_beneath_nofollow(root, parent_parts)
    staging_descriptor = -1
    staging_name: str | None = None
    published = False
    try:
        create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            create_flags |= os.O_CLOEXEC
        for _ in range(128):
            proposed = f".{candidate.name}.{secrets.token_hex(16)}.staging"
            try:
                staging_descriptor = os.open(
                    proposed,
                    create_flags,
                    0o600,
                    dir_fd=parent_descriptor,
                )
                staging_name = proposed
                break
            except FileExistsError:
                continue
        if staging_descriptor < 0 or staging_name is None:
            _hold("HOLD_TYPED_SAFE_WRITE", "cannot allocate exclusive staging artifact")
        _write_all(staging_descriptor, raw)
        os.fsync(staging_descriptor)
        staging_metadata = os.fstat(staging_descriptor)
        if (
            not stat.S_ISREG(staging_metadata.st_mode)
            or staging_metadata.st_nlink != 1
            or staging_metadata.st_size != len(raw)
        ):
            _hold("HOLD_TYPED_SAFE_WRITE", "staging artifact metadata is invalid")
        os.close(staging_descriptor)
        staging_descriptor = -1
        try:
            os.link(
                staging_name,
                candidate.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            _hold("HOLD_TYPED_SAFE_WRITE", f"artifact output already exists: {candidate}")
        published = True
        os.fsync(parent_descriptor)
        os.unlink(staging_name, dir_fd=parent_descriptor)
        staging_name = None
        os.fsync(parent_descriptor)
        replayed = _read_regular_file_at_exact(
            parent_descriptor,
            candidate.name,
            expected_sha256,
            maximum_bytes=MAX_AUTHORITY_BYTES,
            label="published typed safe authority artifact",
        )
        if replayed != raw:
            _hold("HOLD_TYPED_SAFE_WRITE", "published artifact bytes changed during replay")
        _verify_namespace_binding(root, parent_parts, parent_descriptor)
        return expected_sha256
    except TypedSafeAuthorityError:
        raise
    except OSError as exc:
        _hold("HOLD_TYPED_SAFE_WRITE", f"descriptor-relative publication failed: {exc}")
    finally:
        if staging_descriptor >= 0:
            os.close(staging_descriptor)
        if staging_name is not None:
            try:
                os.unlink(staging_name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
            except OSError:
                pass
        # Never delete or overwrite a visible final name after publication.
        if published:
            pass
        os.close(parent_descriptor)


def _read_code_exact(root: Path, relative: Path, expected_sha256: str) -> bytes:
    _path_beneath(root, root / relative, label="code file")
    directory_descriptor = _open_absolute_directory_nofollow(root)
    file_descriptor = -1
    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        file_flags = os.O_RDONLY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            directory_flags |= os.O_CLOEXEC
            file_flags |= os.O_CLOEXEC
        for part in relative.parts[:-1]:
            next_descriptor = os.open(
                part,
                directory_flags,
                dir_fd=directory_descriptor,
            )
            metadata = os.fstat(next_descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(next_descriptor)
                _hold("HOLD_TYPED_SAFE_PATH", "code path component is not a directory")
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        file_descriptor = os.open(
            relative.parts[-1],
            file_flags,
            dir_fd=directory_descriptor,
        )
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            _hold("HOLD_TYPED_SAFE_PATH", "code file must be a regular single-link file")
        if before.st_size > MAX_CODE_FILE_BYTES:
            _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "code file exceeds safe replay size")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(file_descriptor, min(1 << 20, MAX_CODE_FILE_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_CODE_FILE_BYTES:
                _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "code file exceeds safe replay size")
        after = os.fstat(file_descriptor)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or after.st_nlink != 1
        ):
            _hold("HOLD_TYPED_SAFE_PATH", "code file metadata changed during exact replay")
        raw = b"".join(chunks)
    except TypedSafeAuthorityError:
        raise
    except OSError as exc:
        _hold("HOLD_TYPED_SAFE_PATH", f"code file cannot be opened safely: {exc}")
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        _hold("HOLD_TYPED_SAFE_HASH_MISMATCH", "code replay SHA-256 mismatch")
    try:
        text = raw.decode("utf-8")
    except UnicodeError:
        _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "code file must be UTF-8 text")
    if PROHIBITED_STRING.search(text):
        _hold("HOLD_TYPED_SAFE_PROHIBITED_CONTENT", "code file references prohibited test/result support")
    return raw


def _validate_code_snapshot_binding(
    payload: Mapping,
    trusted_code_snapshot: Mapping | None,
) -> list[dict[str, str]]:
    trusted = _normalize_trusted_code_snapshot(trusted_code_snapshot)
    declared_snapshot = _digest(
        payload.get("trusted_code_snapshot_sha256"),
        label="trusted code snapshot",
    )
    actual_snapshot = canonical_sha256(trusted)
    if declared_snapshot != actual_snapshot:
        _hold(
            "HOLD_EXTERNAL_CODE_SNAPSHOT_REVIEW_REQUIRED",
            "code manifest does not bind the independently frozen snapshot",
        )
    declared_files = [dict(entry) for entry in payload.get("files", [])]
    if declared_files != trusted:
        _hold(
            "HOLD_EXTERNAL_CODE_SNAPSHOT_REVIEW_REQUIRED",
            "code manifest differs from the independently frozen path/hash allowlist",
        )
    return trusted


def _replay_code_tree(
    run_root: Path,
    payload: Mapping,
    trusted_code_snapshot: Mapping | None,
) -> None:
    root = _checked_run_root(run_root)
    _validate_code_snapshot_binding(payload, trusted_code_snapshot)
    code_root_value = payload.get("code_root_relative_path")
    if code_root_value != ".":
        _hold(
            "HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY",
            "typed code projection currently supports only the run-root repository",
        )
    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        _hold("HOLD_TYPED_SAFE_SCHEMA", "code manifest files must be non-empty")
    if len(entries) > MAX_CODE_FILES:
        _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "code manifest is too large")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw_entry in enumerate(entries):
        entry = _strict_keys(raw_entry, CODE_ENTRY_KEYS, label=f"code entry {index}")
        relative = _validate_code_path(entry.get("path"))
        key = relative.as_posix()
        if key in seen:
            _hold("HOLD_TYPED_SAFE_SCHEMA", "code manifest paths must be unique")
        seen.add(key)
        digest = _digest(entry.get("sha256"), label=f"code file {key}")
        _read_code_exact(root, relative, digest)
        normalized.append({"path": key, "sha256": digest})
    if normalized != sorted(normalized, key=lambda item: item["path"]):
        _hold("HOLD_TYPED_SAFE_SCHEMA", "code manifest entries must be sorted")
    declared = _digest(payload.get("tree_sha256"), label="code tree")
    actual = canonical_sha256(normalized)
    if declared != actual:
        _hold("HOLD_TYPED_SAFE_HASH_MISMATCH", "code tree semantic digest mismatch")


def validate_analysis_code_tree_manifest(
    payload: Mapping,
    *,
    trusted_code_snapshot: Mapping | None = None,
) -> dict:
    _validate_common(
        payload,
        authority_name=ANALYSIS_CODE_TREE_MANIFEST,
        expected_keys=CODE_KEYS,
    )
    if payload.get("code_root_relative_path") != ".":
        _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "unsupported code root")
    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        _hold("HOLD_TYPED_SAFE_SCHEMA", "code manifest files must be non-empty")
    if len(entries) > MAX_CODE_FILES:
        _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "code manifest is too large")
    normalized: list[dict[str, str]] = []
    for index, raw_entry in enumerate(entries):
        entry = _strict_keys(raw_entry, CODE_ENTRY_KEYS, label=f"code entry {index}")
        relative = _validate_code_path(entry.get("path"))
        digest = _digest(entry.get("sha256"), label=f"code file {relative.as_posix()}")
        normalized.append({"path": relative.as_posix(), "sha256": digest})
    if len({item["path"] for item in normalized}) != len(normalized):
        _hold("HOLD_TYPED_SAFE_SCHEMA", "code manifest paths must be unique")
    if normalized != sorted(normalized, key=lambda item: item["path"]):
        _hold("HOLD_TYPED_SAFE_SCHEMA", "code manifest entries must be sorted")
    if payload.get("tree_sha256") != canonical_sha256(normalized):
        _hold("HOLD_TYPED_SAFE_HASH_MISMATCH", "code tree digest is stale")
    _validate_code_snapshot_binding(payload, trusted_code_snapshot)
    return dict(payload)


def validate_method_protocol(payload: Mapping) -> dict:
    _validate_common(payload, authority_name=METHOD_PROTOCOL, expected_keys=METHOD_KEYS)
    if payload.get("selected_route") != AUTHORIZED_ROUTE:
        _hold("HOLD_TYPED_SAFE_SCHEMA", "method protocol selects a different route")
    if payload.get("execution_order") != list(AUTHORIZED_SEQUENCE):
        _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "unsupported method execution order")
    if payload.get("seeds") != list(FROZEN_SEEDS):
        _hold("HOLD_TYPED_SAFE_SCHEMA", "method protocol must freeze five seeds")
    if payload.get("information_set") != "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR":
        _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "unsupported information set")
    if payload.get("estimand_training_scope") != "INDEPENDENT_MODEL_PER_PRIMARY_ESTIMAND":
        _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "unsupported estimand training scope")
    methods = _safe_id_list(payload.get("method_ids"), label="method_ids")
    if not MINIMUM_METHOD_IDS.issubset(methods):
        _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "method protocol omits required methods")
    if not set(methods).issubset(ALLOWED_METHOD_IDS):
        _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "method protocol adds an unsupported method")
    if payload.get("negative_controls") != list(FROZEN_NEGATIVE_CONTROLS):
        _hold("HOLD_TYPED_SAFE_SCHEMA", "negative controls differ from the frozen family")
    training = _strict_keys(
        payload.get("training_policy"), TRAINING_POLICY_KEYS, label="training_policy"
    )
    expected_training = {
        "sampler": "STUDY_ENDPOINT_CONTEXT_AND_SOURCE_COMPONENT_BALANCED",
        "mean_loss": "HUBER",
        "uncertainty_loss": "GAUSSIAN_NLL",
        "sign_probability": "NORMAL_CDF_OF_MU_OVER_SIGMA",
        "sign_bce_variance_gradient": "DETACHED",
        "hard_sign_rule": "RAW_PREDICTED_DELTA_MEAN_GE_ZERO",
        "rank_loss": "DISABLED_UNLESS_SEPARATE_PUBLICATION_RANK_POLICY_QUALIFIED",
        "checkpoint_selection": "SIGN_GATE_THEN_SPEARMAN_THEN_NLL_THEN_EARLIEST_EPOCH",
        "selection_partition": "development",
    }
    if not _exact_match(training, expected_training):
        _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "unsupported training policy")
    compute = _strict_keys(
        payload.get("compute_matching"), COMPUTE_MATCHING_KEYS, label="compute_matching"
    )
    expected_compute = {
        "same_training_rows": True,
        "same_seeds": True,
        "same_epoch_and_optimizer_budget": True,
        "a_control_parameter_difference_max_fraction": 0.05,
        "report_parameter_count": True,
        "report_total_forward_equivalents": True,
        "report_wall_time_and_peak_memory": True,
    }
    if not _exact_match(compute, expected_compute):
        _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "unsupported compute-matching policy")
    if payload.get("stop_rules") != list(FROZEN_STOP_RULES):
        _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "stop rules differ from the safe schema")
    boundaries = _strict_keys(
        payload.get("hard_boundaries"), METHOD_BOUNDARY_KEYS, label="method hard_boundaries"
    )
    expected_boundaries = {
        "ordinary_only": True,
        "sealed_member_access": False,
        "formal_sealed_evaluator": False,
        "O0_G1_E0_unlock": False,
        "active_contract_mutation": False,
        "sign_gate_change_authorized": False,
    }
    _require_exact(
        boundaries,
        expected_boundaries,
        label="method hard boundaries",
    )
    return dict(payload)


def validate_paired_strongest_baseline_comparison_policy(payload: Mapping) -> dict:
    _validate_common(
        payload,
        authority_name=PAIRED_BASELINE_POLICY,
        expected_keys=PAIRED_KEYS,
    )
    expected_identity = {
        "policy_id": "R3X_PAIRED_STRONGEST_BASELINE_POLICY_V1",
        "primary_candidate_model_id": "A_EXACT_POTENTIAL_DIFFERENCE",
        "comparison_family_id": "R3X_PRIMARY_PAIRED_FAMILY_V1",
        "comparison_family_member_ids": [
            "primary_candidate_vs_development_selected_baseline_sign"
        ],
        "primary_comparison_member_id": (
            "primary_candidate_vs_development_selected_baseline_sign"
        ),
    }
    for field, expected in expected_identity.items():
        _require_exact(payload.get(field), expected, label=f"paired policy {field}")
    if payload.get("baseline_suite_status") != "PRETEST_POLICY_FROZEN":
        _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "baseline suite pretest policy is not frozen")
    if payload.get("information_set") != "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR":
        _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "paired policy information set is unsupported")
    _validate_baseline_execution_qualification(payload)
    if payload.get("development_selector_required") is not True:
        _hold(
            "HOLD_TYPED_SAFE_SCHEMA",
            "a separate development-only baseline selector is required",
        )
    if payload.get("development_selector_schema") != DEVELOPMENT_SELECTOR_SCHEMA:
        _hold(
            "HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY",
            "unsupported development selector schema",
        )
    members = _safe_id_list(
        payload.get("comparison_family_member_ids"),
        label="comparison_family_member_ids",
    )
    if payload.get("primary_comparison_member_id") not in members:
        _hold("HOLD_TYPED_SAFE_SCHEMA", "primary comparison is absent from the family")
    eligible = _safe_id_list(
        payload.get("eligible_baseline_method_ids"),
        label="eligible_baseline_method_ids",
        sorted_required=False,
    )
    baselines = _safe_id_list(
        payload.get("required_baseline_method_ids"),
        label="required_baseline_method_ids",
        sorted_required=False,
    )
    if eligible != list(FROZEN_BASELINE_METHOD_ORDER):
        _hold(
            "HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY",
            "paired selector candidates differ from the frozen contract order",
        )
    if baselines != list(FROZEN_BASELINE_METHOD_ORDER):
        _hold(
            "HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY",
            "paired required baselines differ from the frozen contract order",
        )
    if set(baselines) != MINIMUM_BASELINE_METHOD_IDS:
        _hold(
            "HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY",
            "paired policy baseline suite differs from the supported exact set",
        )
    if set(eligible) != MINIMUM_BASELINE_METHOD_IDS:
        _hold(
            "HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY",
            "paired selector candidates differ from the supported exact set",
        )
    if not set(baselines).issubset(eligible):
        _hold("HOLD_TYPED_SAFE_SCHEMA", "required baselines must be eligible comparators")
    if payload.get("primary_candidate_model_id") in eligible:
        _hold("HOLD_TYPED_SAFE_SCHEMA", "the primary candidate cannot self-compare as a baseline")
    inference = _strict_keys(
        payload.get("primary_sign_inference"), INFERENCE_KEYS, label="primary_sign_inference"
    )
    if inference.get("metric_id") != "MACRO_COMPONENT_CELL_BALANCED_SIGN_ACCURACY":
        _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "paired primary metric is unsupported")
    _require_exact(inference.get("alpha"), 0.05, label="paired alpha")
    _require_exact(inference.get("confidence"), 0.95, label="paired confidence")
    bootstrap = _strict_keys(inference.get("bootstrap"), BOOTSTRAP_KEYS, label="bootstrap")
    permutation = _strict_keys(
        inference.get("permutation"), PERMUTATION_KEYS, label="permutation"
    )
    _require_exact(
        bootstrap,
        {
            "n_resamples": 2000,
            "seed": 23,
            "min_components_per_cell": 10,
            "max_attempt_multiplier": 100,
        },
        label="paired bootstrap policy",
    )
    _require_exact(
        permutation,
        {
            "n_permutations": 10000,
            "seed": 42,
            "exact_component_limit": 20,
            "min_components_per_cell": 10,
        },
        label="paired permutation policy",
    )
    return dict(payload)


VALIDATORS: Mapping[str, Callable[[Mapping], dict]] = MappingProxyType({
    ROUTE_AUTHORIZATION: validate_route_authorization,
    ENDPOINT_ESTIMAND_REGISTRY: validate_endpoint_estimand_registry,
    METRIC_ZERO_POLICY_REGISTRY: validate_metric_zero_policy_registry,
    MODEL_OPTIMIZER_SEED_CONFIG: validate_model_optimizer_seed_config,
    ANALYSIS_CODE_TREE_MANIFEST: validate_analysis_code_tree_manifest,
    METHOD_PROTOCOL: validate_method_protocol,
    PAIRED_BASELINE_POLICY: validate_paired_strongest_baseline_comparison_policy,
})


def validate_typed_safe_authority(
    payload: Mapping,
    *,
    trusted_code_snapshot: Mapping | None = None,
) -> dict:
    if not isinstance(payload, Mapping):
        _hold("HOLD_TYPED_SAFE_SCHEMA", "typed safe authority must be an object")
    name = payload.get("authority_name")
    if name not in VALIDATORS:
        _hold(
            "HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY",
            f"unsupported typed safe authority: {name!r}",
        )
    if name == ANALYSIS_CODE_TREE_MANIFEST:
        return validate_analysis_code_tree_manifest(
            payload,
            trusted_code_snapshot=trusted_code_snapshot,
        )
    return VALIDATORS[name](payload)


def validate_typed_safe_bundle(
    authorities: Mapping,
    *,
    trusted_code_snapshot: Mapping | None = None,
) -> dict:
    """Cross-bind all seven typed-safe authorities without claiming activation."""

    typed = _strict_keys(
        authorities,
        set(SUPPORTED_AUTHORITIES),
        label="typed safe authority bundle",
    )
    normalized: dict[str, dict] = {}
    for name in SUPPORTED_AUTHORITIES:
        authority = validate_typed_safe_authority(
            typed[name],
            trusted_code_snapshot=(
                trusted_code_snapshot
                if name == ANALYSIS_CODE_TREE_MANIFEST
                else None
            ),
        )
        if authority["authority_name"] != name:
            _hold(
                "HOLD_TYPED_SAFE_BUNDLE_BINDING",
                "typed safe authority occupies the wrong bundle slot",
            )
        normalized[name] = authority

    anchor = normalized[SUPPORTED_AUTHORITIES[0]]
    for name, authority in normalized.items():
        for field in BUNDLE_SHARED_FIELDS:
            if not _exact_match(authority[field], anchor[field]):
                _hold(
                    "HOLD_TYPED_SAFE_BUNDLE_BINDING",
                    f"typed safe bundle disagrees on {field}",
                    authority_name=name,
                )

    source_hashes = [
        normalized[name]["source_authority_sha256"]
        for name in SUPPORTED_AUTHORITIES
    ]
    if len(source_hashes) != len(set(source_hashes)):
        _hold(
            "HOLD_TYPED_SAFE_BUNDLE_BINDING",
            "each source authority must have a unique byte commitment",
        )
    data_hashes = {
        anchor["train_development_outcomes_sha256"],
        *anchor["development_projection_hashes"].values(),
    }
    if set(source_hashes) & data_hashes:
        _hold(
            "HOLD_TYPED_SAFE_BUNDLE_BINDING",
            "source authorities cannot alias development data bindings",
        )

    config_qualification = normalized[MODEL_OPTIMIZER_SEED_CONFIG][
        "frozen_environment_execution_qualification_sha256"
    ]
    paired_qualification = normalized[PAIRED_BASELINE_POLICY][
        "frozen_environment_execution_qualification_sha256"
    ]
    if config_qualification != paired_qualification:
        _hold(
            "HOLD_TYPED_SAFE_BUNDLE_BINDING",
            "config and paired policy bind different execution qualifications",
        )

    return {
        "status": TYPED_DEVELOPMENT_SAFE_BUNDLE_NOT_TRAINING_READY,
        "training_ready": False,
        "execution_qualification_replay_status": (
            "CERTIFICATE_HASH_BOUND_NOT_REPLAYED"
        ),
        "remaining_required_typed_row_authorities": list(
            REMAINING_TYPED_ROW_AUTHORITIES
        ),
        "authorities": normalized,
    }


def canonical_typed_safe_authority_bytes(
    payload: Mapping,
    *,
    trusted_code_snapshot: Mapping | None = None,
) -> bytes:
    try:
        raw = canonical_json_bytes(
            validate_typed_safe_authority(
                payload,
                trusted_code_snapshot=trusted_code_snapshot,
            )
        )
    except ArtifactIOError as exc:
        _hold("HOLD_TYPED_SAFE_SCHEMA", str(exc))
    if len(raw) > MAX_AUTHORITY_BYTES:
        _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "typed safe authority is too large")
    return raw


def _parse_json_object(raw: bytes) -> dict:
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
        _hold("HOLD_TYPED_SAFE_JSON", f"typed safe authority is not strict JSON: {exc}")
    if not isinstance(payload, dict):
        _hold("HOLD_TYPED_SAFE_JSON", "typed safe authority must be a JSON object")
    return payload


def _checked_artifact_path(run_root: Path, path: Path, *, output: bool) -> tuple[Path, Path]:
    root = _checked_run_root(run_root)
    candidate = _path_beneath(root, path, label="typed authority artifact")
    relative = candidate.relative_to(root)
    if (
        len(relative.parts) != 3
        or relative.parts[:2] != ("development", "authorities")
        or not SAFE_ARTIFACT_NAME.fullmatch(relative.name)
        or ".." in relative.name
    ):
        _hold(
            "HOLD_TYPED_SAFE_PATH",
            "typed authority artifact must be one JSON file directly under development/authorities",
        )
    if output:
        parent = candidate.parent
        if parent.is_symlink() or not parent.is_dir():
            _hold("HOLD_TYPED_SAFE_PATH", "typed authority parent must be an existing real directory")
        _assert_no_symlink_components(
            root,
            parent,
            label="typed authority parent",
        )
    else:
        _assert_no_symlink_components(root, candidate, label="typed authority artifact")
    return root, candidate


def write_typed_safe_authority_exclusive(
    run_root: Path,
    path: Path,
    payload: Mapping,
    *,
    trusted_code_snapshot: Mapping | None = None,
) -> str:
    """Validate and exclusively publish one externally generated projection."""

    root, candidate = _checked_artifact_path(run_root, path, output=True)
    validated = validate_typed_safe_authority(
        payload,
        trusted_code_snapshot=trusted_code_snapshot,
    )
    raw = canonical_typed_safe_authority_bytes(
        validated,
        trusted_code_snapshot=trusted_code_snapshot,
    )
    if validated["authority_name"] == ANALYSIS_CODE_TREE_MANIFEST:
        _replay_code_tree(root, validated, trusted_code_snapshot)
    digest = _secure_publish_authority_artifact(root, candidate, raw)
    if validated["authority_name"] == ANALYSIS_CODE_TREE_MANIFEST:
        # Close the publication window as far as a hash-bound manifest can:
        # exact-replay again after the immutable artifact name is visible.
        _replay_code_tree(root, validated, trusted_code_snapshot)
    return digest


def load_typed_safe_authority_exact(
    run_root: Path,
    path: Path,
    expected_sha256: str,
    *,
    expected_authority_name: str | None = None,
    trusted_code_snapshot: Mapping | None = None,
) -> dict:
    """Exactly read, validate, and replay one typed safe projection."""

    root, candidate = _checked_artifact_path(run_root, path, output=False)
    _digest(expected_sha256, label="typed safe authority SHA-256")
    raw = _secure_read_authority_artifact(root, candidate, expected_sha256)
    if len(raw) > MAX_AUTHORITY_BYTES:
        _hold("HOLD_UNSUPPORTED_TYPED_SAFE_AUTHORITY", "typed safe authority is too large")
    payload = validate_typed_safe_authority(
        _parse_json_object(raw),
        trusted_code_snapshot=trusted_code_snapshot,
    )
    if raw != canonical_typed_safe_authority_bytes(
        payload,
        trusted_code_snapshot=trusted_code_snapshot,
    ):
        _hold("HOLD_TYPED_SAFE_JSON", "typed safe authority bytes are not canonical")
    if expected_authority_name is not None and payload["authority_name"] != expected_authority_name:
        _hold("HOLD_TYPED_SAFE_BINDING", "typed safe authority name binding mismatch")
    if payload["authority_name"] == ANALYSIS_CODE_TREE_MANIFEST:
        _replay_code_tree(root, payload, trusted_code_snapshot)
    return payload


def load_typed_safe_bundle_exact(
    run_root: Path,
    authority_refs: Mapping,
    *,
    trusted_code_snapshot: Mapping | None = None,
) -> dict:
    """Exact-load all seven artifacts, then reject any mixed-run bundle."""

    root = _checked_run_root(run_root)
    refs = _strict_keys(
        authority_refs,
        set(SUPPORTED_AUTHORITIES),
        label="typed safe authority bundle refs",
    )
    authorities: dict[str, dict] = {}
    seen_paths: set[str] = set()
    seen_artifact_hashes: set[str] = set()
    for name in SUPPORTED_AUTHORITIES:
        ref = _strict_keys(refs[name], BUNDLE_REF_KEYS, label=f"{name} bundle ref")
        relative_value = ref.get("path")
        if not isinstance(relative_value, str) or not relative_value:
            _hold("HOLD_TYPED_SAFE_PATH", f"{name} bundle ref path is invalid")
        relative = Path(relative_value)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != relative_value
        ):
            _hold(
                "HOLD_TYPED_SAFE_PATH",
                f"{name} bundle ref must be a normalized relative path",
            )
        path_key = relative.as_posix()
        if path_key in seen_paths:
            _hold(
                "HOLD_TYPED_SAFE_BUNDLE_BINDING",
                "typed safe bundle refs must use distinct paths",
            )
        seen_paths.add(path_key)
        artifact_sha = _digest(ref.get("sha256"), label=f"{name} artifact")
        if artifact_sha in seen_artifact_hashes:
            _hold(
                "HOLD_TYPED_SAFE_BUNDLE_BINDING",
                "typed safe bundle refs must bind distinct artifacts",
            )
        seen_artifact_hashes.add(artifact_sha)
        authorities[name] = load_typed_safe_authority_exact(
            root,
            root / relative,
            artifact_sha,
            expected_authority_name=name,
            trusted_code_snapshot=(
                trusted_code_snapshot
                if name == ANALYSIS_CODE_TREE_MANIFEST
                else None
            ),
        )
    return validate_typed_safe_bundle(
        authorities,
        trusted_code_snapshot=trusted_code_snapshot,
    )


__all__ = [
    "ANALYSIS_CODE_TREE_MANIFEST",
    "DEVELOPMENT_PROJECTION_HASH_KEYS",
    "ENDPOINT_ESTIMAND_REGISTRY",
    "METRIC_ZERO_POLICY_REGISTRY",
    "METHOD_PROTOCOL",
    "MODEL_OPTIMIZER_SEED_CONFIG",
    "PAIRED_BASELINE_POLICY",
    "ROUTE_AUTHORIZATION",
    "SCHEMAS",
    "SUPPORTED_AUTHORITIES",
    "TYPED_DEVELOPMENT_SAFE",
    "TYPED_DEVELOPMENT_SAFE_BUNDLE_NOT_TRAINING_READY",
    "TypedSafeAuthorityError",
    "canonical_typed_safe_authority_bytes",
    "load_typed_safe_authority_exact",
    "load_typed_safe_bundle_exact",
    "validate_analysis_code_tree_manifest",
    "validate_endpoint_estimand_registry",
    "validate_method_protocol",
    "validate_metric_zero_policy_registry",
    "validate_model_optimizer_seed_config",
    "validate_paired_strongest_baseline_comparison_policy",
    "validate_route_authorization",
    "validate_typed_safe_authority",
    "validate_typed_safe_bundle",
    "write_typed_safe_authority_exclusive",
]
