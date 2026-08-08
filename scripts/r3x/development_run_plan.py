"""Canonical ordinary train/development planning scaffold without execution.

This module performs no fitting, accelerator discovery, or data-stage release.
It converts exact, pre-bound development inputs into one immutable plan whose
scientific method order, row slices, proposed configuration, and resource
budget can be replayed before a later runner is implemented.  It deliberately
does not establish an external trust root, a development capability, or
training authorization; callers must never use this artifact as a permission
token.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import stat
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any, AbstractSet, Mapping, Sequence

from .artifact_io import (
    ArtifactIOError,
    atomic_publish_bytes,
    canonical_json_bytes,
    read_exact_regular_file,
)
from .protocol import ProtocolHoldError, canonical_sha256
from .typed_safe_authorities import (
    DEVELOPMENT_SELECTOR_SCHEMA,
    ENDPOINT_ESTIMAND_REGISTRY,
    FROZEN_BASELINE_METHOD_ORDER as TYPED_FROZEN_BASELINE_METHOD_ORDER,
    MINIMUM_BASELINE_METHOD_IDS,
    MODEL_OPTIMIZER_SEED_CONFIG,
    PAIRED_BASELINE_POLICY,
    SCHEMAS as TYPED_AUTHORITY_SCHEMAS,
    TypedSafeAuthorityError,
    load_typed_safe_authority_exact,
)


RUN_PLAN_SCHEMA = "r3x_development_run_plan_v4"
RUN_PLAN_ROLE = "ORDINARY_TRAIN_DEVELOPMENT_PLAN_SCAFFOLD_NO_EXECUTION"
RUN_PLAN_STATUS = "DEVELOPMENT_PLAN_SCAFFOLD_NOT_AUTHORIZED"

EXECUTABLE_CONFIG_SCHEMA = "r3x_development_executable_config_v4"
EXECUTABLE_CONFIG_ROLE = "ORDINARY_DEVELOPMENT_EXECUTABLE_CONFIG"
EXECUTABLE_CONFIG_STATUS = "FROZEN_PLAN_INPUT_CONFIG_NOT_COMPILED"
V1_EXECUTABLE_CONFIG_SCHEMA = TYPED_AUTHORITY_SCHEMAS[
    MODEL_OPTIMIZER_SEED_CONFIG
]

ROW_INVENTORY_SCHEMA = "r3x_development_row_inventory_v1"
ROW_INVENTORY_ROLE = "ORDINARY_TRAIN_DEVELOPMENT_ROW_INVENTORY"
ROW_INVENTORY_STATUS = "FROZEN_TYPED_ROW_BINDING"

DEVELOPMENT_RELEASE_SCHEMA = "r3x_development_release_hash_declaration_v2"
DEVELOPMENT_RELEASE_ROLE = "ORDINARY_DEVELOPMENT_PLAN_RELEASE"
DEVELOPMENT_RELEASE_STATUS = "HASH_DECLARATION_BOUND_NOT_INDEPENDENTLY_TRUSTED"

HGB_QUALIFICATION_SCHEMA = "r3x_hgb_execution_qualification_v3"
HGB_QUALIFICATION_ROLE = "ORDINARY_DEVELOPMENT_HGB_QUALIFICATION"
HGB_QUALIFICATION_STATUS = "HASH_DECLARATION_BOUND_NOT_REPLAYED"

ORDINARY_TRAIN = "ORDINARY_TRAIN"
ORDINARY_DEVELOPMENT = "ORDINARY_DEVELOPMENT"
PARTITIONS = (ORDINARY_TRAIN, ORDINARY_DEVELOPMENT)
CAPABILITY_SCOPE = "ORDINARY_TRAIN_AND_ORDINARY_DEVELOPMENT"
FROZEN_SEEDS = (11, 23, 42, 67, 101)
SIGN_ACCURACY_GATE = 0.60
PAIRED_KMER_FEATURE_SCHEMA = (
    "R3X_PAIRED_LOCAL_GLOBAL_KMER_V2_PER_EDIT_WINDOW_MEAN_NO_CROSS_WINDOW_KMERS"
)
EDIT_ONLY_FEATURE_SCHEMA = "R3X_EDIT_ONLY_FEATURES_V1"
CANDIDATE_KMER_FEATURE_SCHEMA = "R3X_CANDIDATE_KMER_FREQUENCIES_V1"
RIDGE_SOLVER_POLICY = "SMALLER_PRIMAL_OR_DUAL_SQUARE_SYSTEM_V2"
FORMAL_SCALAR_WEIGHT_POLICY = (
    "CELL_THEN_GLOBAL_SOURCE_COMPONENT_THEN_ROW_"
    "SUM_TO_ONE_WEIGHTED_RISK_V1"
)
RIDGE_TRAINING_WEIGHT_POLICY = (
    "FORMAL_WEIGHTED_SCALER_INTERCEPT_AND_SQUARED_ERROR_PLUS_ALPHA_L2_V2"
)
RIDGE_CV_SCORING_POLICY = (
    "ONE_OOF_PREDICTION_PER_ROW_THEN_CELL_COMPONENT_ROW_WEIGHTED_MSE_V1"
)
PRIOR_TRAINING_POLICY = (
    "CONDITION_LOCAL_AND_GLOBAL_FALLBACK_CELL_COMPONENT_ROW_MEAN_V2"
)
NEAREST_NEIGHBOR_SEARCH_POLICY = (
    "EXACT_TWO_PASS_CHUNKED_DISTANCE_CANONICAL_FSUM_TIES_V2"
)
NEAREST_NEIGHBOR_SCALING_POLICY = (
    "TRAIN_ONLY_FORMAL_WEIGHTED_MEAN_VARIANCE_V2"
)
NEAREST_NEIGHBOR_TIE_AGGREGATION_POLICY = (
    "CANONICAL_ROW_ID_CELL_COMPONENT_ROW_MATH_FSUM_MEAN_V2"
)
HGB_TRAINING_WEIGHT_POLICY = (
    "FORMAL_CELL_THEN_SOURCE_COMPONENT_THEN_ROW_BALANCED_"
    "LOSS_SAMPLE_WEIGHT_ONLY_V2"
)
HGB_WEIGHT_NORMALIZATION = (
    "MATH_FSUM_MEAN_WEIGHT_ONE_SUM_EQUALS_N_ROWS"
)
FROZEN_BASELINE_METHOD_ORDER = tuple(TYPED_FROZEN_BASELINE_METHOD_ORDER)
if frozenset(FROZEN_BASELINE_METHOD_ORDER) != MINIMUM_BASELINE_METHOD_IDS:
    raise RuntimeError("typed exact-eight order and supported baseline set diverged")
RIDGE_METHOD_IDS = (
    "CANDIDATE_KMER_RIDGE",
    "EDIT_ONLY_RIDGE",
    "PAIRED_LOCAL_GLOBAL_KMER_RIDGE",
)
SINGLE_FIT_METHOD_IDS = (
    "HISTOGRAM_GRADIENT_BOOSTING",
    "PAIRED_FEATURE_NEAREST_NEIGHBOR",
    "SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR",
    "ZERO_DELTA",
)
NEURAL_METHOD_ID = "FULL_PAIR_UNRESTRICTED_SIAMESE"

PROJECTION_NAMES = (
    "inclusion_manifest_development_projection",
    "context_source_crosswalk_development_projection",
    "source_component_split_manifest_development_projection",
)
ROW_SOURCE_NAMES = (
    "inclusion_manifest",
    "context_source_crosswalk",
    "source_component_split_manifest",
)
SOURCE_REF_NAMES = (
    "independent_development_release",
    "paired_baseline_policy",
    "endpoint_estimand_registry",
    "executable_config",
    "hgb_qualification",
    "row_inventory",
)
APPROVED_RELEASE_REF_NAMES = tuple(
    name for name in SOURCE_REF_NAMES if name != "independent_development_release"
)
SOURCE_NAMESPACES = {
    "independent_development_release": ("development", "releases"),
    "paired_baseline_policy": ("development", "authorities"),
    "endpoint_estimand_registry": ("development", "authorities"),
    "executable_config": ("development", "run_plan_inputs"),
    "hgb_qualification": ("development", "qualifications"),
    "row_inventory": ("development", "run_plan_inputs"),
}
RUN_PLAN_NAMESPACE = ("development", "run_plans")

REF_KEYS = {"relative_path", "sha256"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")
MAX_JSON_BYTES = 8 * 1024 * 1024
PROTECTED_COMPACT_TOKENS = (
    "internaltest",
    "testoutcome",
    "sealed",
    "downstream",
    "protected",
    "gse246381",
)


class DevelopmentRunPlanError(ProtocolHoldError):
    """Structured HOLD raised before execution or artifact publication."""


def _hold(code: str, message: str, **details: Any) -> None:
    raise DevelopmentRunPlanError(code, message, details=details or None)


def _strict_keys(
    value: object,
    expected: AbstractSet[str],
    *,
    label: str,
) -> Mapping:
    if not isinstance(value, Mapping):
        _hold("HOLD_DEVELOPMENT_RUN_PLAN_SCHEMA", f"{label} must be an object")
    actual = set(value)
    if actual != set(expected):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_SCHEMA",
            f"{label} keys differ from the exact schema",
            missing=sorted(set(expected) - actual),
            extra=sorted(actual - set(expected)),
        )
    if any(not isinstance(key, str) for key in value):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_SCHEMA",
            f"{label} keys must be strings",
        )
    return value


def _exact(value: object, expected: object, *, label: str) -> None:
    if isinstance(expected, bool):
        matches = value is expected
    else:
        matches = type(value) is type(expected) and value == expected
    if not matches:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_SCHEMA",
            f"{label} differs from the exact policy",
        )


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_SCHEMA",
            f"{label} must be lowercase SHA-256",
        )
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_SCHEMA",
            f"{label} must be a safe non-empty identifier",
        )
    return value


def _bounded_int(
    value: object,
    *,
    label: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_SCHEMA",
            f"{label} must be an integer",
        )
    if value < minimum or value > maximum:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_SCHEMA",
            f"{label} is outside its supported range",
        )
    return value


def _finite_number(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
    open_minimum: bool = False,
    open_maximum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_SCHEMA",
            f"{label} must be numeric",
        )
    number = float(value)
    if not math.isfinite(number):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_SCHEMA",
            f"{label} must be finite",
        )
    if minimum is not None and (
        number <= minimum if open_minimum else number < minimum
    ):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_SCHEMA",
            f"{label} is below its supported range",
        )
    if maximum is not None and (
        number >= maximum if open_maximum else number > maximum
    ):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_SCHEMA",
            f"{label} exceeds its supported range",
        )
    return number


def _compact_security_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) not in {"Cf", "Cc"}
    )
    return "".join(character for character in normalized if character.isalnum())


def _scan_protected(value: object, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            _scan_protected(key_text, path=f"{path}::<field_name>")
            _scan_protected(child, path=f"{path}.{key_text}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _scan_protected(child, path=f"{path}[{index}]")
        return
    if not isinstance(value, str):
        return
    compact = _compact_security_text(value)
    for token in PROTECTED_COMPACT_TOKENS:
        if token in compact:
            _hold(
                "HOLD_DEVELOPMENT_RUN_PLAN_PROTECTED_CONTENT",
                "run-plan input/output contains a future-stage or protected token",
                field_path=path,
            )


class _DuplicateJSONKey(ValueError):
    pass


def _pairs_no_duplicates(pairs: Sequence[tuple[str, object]]) -> dict:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _parse_json_object(raw: bytes, *, label: str) -> dict:
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_JSON_BYTES:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_JSON",
            f"{label} bytes are empty or too large",
        )
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, _DuplicateJSONKey, ValueError) as exc:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_JSON",
            f"{label} is not strict duplicate-free JSON",
            error=str(exc),
        )
    if not isinstance(payload, dict):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_JSON",
            f"{label} must be a JSON object",
        )
    return payload


def _checked_run_root(run_root: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(run_root)))
    if root.is_symlink() or not root.is_dir():
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_PATH",
            "run root must be an existing real directory",
        )
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_PATH",
            f"run root cannot be resolved: {exc}",
        )
    if root != resolved:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_PATH",
            "run root cannot traverse a symbolic link",
        )
    return root


def _normalized_relative_path(
    value: object,
    *,
    label: str,
    required_prefix: Sequence[str],
) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_PATH",
            f"{label} must be a normalized POSIX relative path",
        )
    _scan_protected(value, path=f"$.{label}")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or value != relative.as_posix()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or any(SAFE_PATH_COMPONENT.fullmatch(part) is None for part in relative.parts)
        or tuple(relative.parts[: len(required_prefix)]) != tuple(required_prefix)
        or relative.suffix != ".json"
    ):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_PATH",
            f"{label} escapes or differs from its required namespace",
        )
    return relative


def _reference_shape(
    value: object,
    *,
    label: str,
    required_prefix: Sequence[str],
) -> dict[str, str]:
    raw = _strict_keys(value, REF_KEYS, label=label)
    relative = _normalized_relative_path(
        raw.get("relative_path"),
        label=f"{label}.relative_path",
        required_prefix=required_prefix,
    )
    return {
        "relative_path": relative.as_posix(),
        "sha256": _digest(raw.get("sha256"), label=f"{label}.sha256"),
    }


def _normalize_source_refs(source_refs: Mapping) -> dict[str, dict[str, str]]:
    raw = _strict_keys(
        source_refs,
        set(SOURCE_REF_NAMES),
        label="run-plan source refs",
    )
    normalized = {
        name: _reference_shape(
            raw[name],
            label=f"source_refs.{name}",
            required_prefix=SOURCE_NAMESPACES[name],
        )
        for name in SOURCE_REF_NAMES
    }
    paths = [item["relative_path"] for item in normalized.values()]
    digests = [item["sha256"] for item in normalized.values()]
    if len(paths) != len(set(paths)) or len(digests) != len(set(digests)):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_BINDING",
            "source refs must bind distinct paths and artifact bytes",
        )
    return normalized


def _existing_ref_path(
    root: Path,
    reference: Mapping[str, str],
    *,
    label: str,
) -> Path:
    relative = PurePosixPath(reference["relative_path"])
    candidate = root.joinpath(*relative.parts)
    current = root
    try:
        for part in relative.parts:
            current = current / part
            metadata = os.lstat(current)
            if stat.S_ISLNK(metadata.st_mode):
                _hold(
                    "HOLD_DEVELOPMENT_RUN_PLAN_PATH",
                    f"{label} cannot traverse a symbolic link",
                )
        metadata = os.lstat(candidate)
    except FileNotFoundError:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_DEPENDENCY",
            f"{label} does not exist",
        )
    except OSError as exc:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_PATH",
            f"{label} cannot be inspected safely: {exc}",
        )
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_PATH",
            f"{label} must be a regular single-link file",
        )
    try:
        if candidate.resolve(strict=True) != candidate:
            _hold(
                "HOLD_DEVELOPMENT_RUN_PLAN_PATH",
                f"{label} path changed during resolution",
            )
    except OSError as exc:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_PATH",
            f"{label} cannot be resolved: {exc}",
        )
    return candidate


def _read_ref_bytes(
    root: Path,
    reference: Mapping[str, str],
    *,
    label: str,
) -> bytes:
    path = _existing_ref_path(root, reference, label=label)
    try:
        return read_exact_regular_file(path, reference["sha256"])
    except ArtifactIOError as exc:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_DEPENDENCY",
            f"{label} exact read failed: {exc}",
        )


def _load_new_json_ref(
    root: Path,
    reference: Mapping[str, str],
    *,
    label: str,
    validator,
) -> dict:
    raw = _read_ref_bytes(root, reference, label=label)
    payload = _parse_json_object(raw, label=label)
    validated = validator(payload)
    try:
        canonical = canonical_json_bytes(validated)
    except ArtifactIOError as exc:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_JSON",
            f"{label} is not canonical JSON: {exc}",
        )
    if canonical != raw:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_JSON",
            f"{label} bytes are not canonical",
        )
    return validated


def _load_typed_ref(
    root: Path,
    reference: Mapping[str, str],
    *,
    authority_name: str,
    label: str,
) -> dict:
    path = _existing_ref_path(root, reference, label=label)
    try:
        return load_typed_safe_authority_exact(
            root,
            path,
            reference["sha256"],
            expected_authority_name=authority_name,
        )
    except TypedSafeAuthorityError as exc:
        code = (
            "HOLD_DEVELOPMENT_RUN_PLAN_PATH"
            if "PATH" in exc.code
            else "HOLD_DEVELOPMENT_RUN_PLAN_DEPENDENCY"
        )
        _hold(
            code,
            f"{label} typed exact replay failed: {exc}",
            upstream_code=exc.code,
        )


def _projection_hashes(value: object, *, label: str) -> dict[str, str]:
    raw = _strict_keys(value, set(PROJECTION_NAMES), label=label)
    normalized = {
        name: _digest(raw[name], label=f"{label}.{name}")
        for name in PROJECTION_NAMES
    }
    if len(set(normalized.values())) != len(normalized):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_BINDING",
            f"{label} values must be distinct",
        )
    return normalized


def validate_hgb_qualification(payload: Mapping) -> dict:
    _scan_protected(payload)
    keys = {
        "schema_version",
        "artifact_role",
        "status",
        "qualification_id",
        "method_id",
        "qualification_scope",
        "device_class",
        "environment_sha256",
        "code_tree_sha256",
        "verifier_code_sha256",
        "synthetic_smoke_sha256",
        "execution_replay_status",
        "ordinary_only",
        "network_access",
        "gpu_execution",
    }
    raw = _strict_keys(payload, keys, label="HGB qualification")
    expected = {
        "schema_version": HGB_QUALIFICATION_SCHEMA,
        "artifact_role": HGB_QUALIFICATION_ROLE,
        "status": HGB_QUALIFICATION_STATUS,
        "method_id": "HISTOGRAM_GRADIENT_BOOSTING",
        "qualification_scope": "FROZEN_HGB_POLICY_ONLY",
        "device_class": "CPU",
        "execution_replay_status": "HASH_DECLARATION_ONLY_NOT_REPLAYED",
        "ordinary_only": True,
        "network_access": False,
        "gpu_execution": False,
    }
    for field, value in expected.items():
        _exact(raw.get(field), value, label=f"HGB qualification.{field}")
    _identifier(raw.get("qualification_id"), label="HGB qualification_id")
    digests = [
        _digest(raw.get(field), label=f"HGB qualification.{field}")
        for field in (
            "environment_sha256",
            "code_tree_sha256",
            "verifier_code_sha256",
            "synthetic_smoke_sha256",
        )
    ]
    if len(digests) != len(set(digests)):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_BINDING",
            "HGB qualification roles cannot alias bytes",
        )
    return copy.deepcopy(dict(raw))


def _validate_model_config(value: object) -> dict:
    keys = {
        "vocab_size",
        "d_model",
        "n_heads",
        "n_layers",
        "ff_dim",
        "max_len",
        "n_endpoints",
        "n_contexts",
        "max_edits",
        "pair_rank",
        "dropout",
        "dropout_consumption_status",
    }
    raw = _strict_keys(value, keys, label="executable config.model")
    integers = {
        "vocab_size": (5, 5),
        "d_model": (8, 16384),
        "n_heads": (1, 256),
        "n_layers": (1, 128),
        "ff_dim": (1, 65536),
        "max_len": (1, 8192),
        "n_endpoints": (2, 65536),
        "n_contexts": (2, 65536),
        "max_edits": (5, 5),
        "pair_rank": (1, 4096),
    }
    normalized = {
        field: _bounded_int(
            raw.get(field), label=f"model.{field}", minimum=bounds[0], maximum=bounds[1]
        )
        for field, bounds in integers.items()
    }
    if normalized["d_model"] % normalized["n_heads"] != 0:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_SCHEMA",
            "model.d_model must be divisible by model.n_heads",
        )
    dropout = _finite_number(
        raw.get("dropout"), label="model.dropout", minimum=0.0, maximum=1.0
    )
    if dropout != 0.0:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_CONFIG",
            "the bound model implementation only supports explicit zero dropout",
        )
    _exact(
        raw.get("dropout_consumption_status"),
        "EXPLICIT_ZERO_MATCHES_BOUND_MODEL_IMPLEMENTATION",
        label="model.dropout_consumption_status",
    )
    return copy.deepcopy(dict(raw))


def validate_executable_config(payload: Mapping) -> dict:
    if isinstance(payload, Mapping) and payload.get("schema_version") == (
        V1_EXECUTABLE_CONFIG_SCHEMA
    ):
        _hold(
            "HOLD_DEVELOPMENT_EXECUTABLE_CONFIG_V1_INCOMPLETE",
            "typed config v1 lacks the complete executable model, fit, and resource fields",
        )
    _scan_protected(payload)
    keys = {
        "schema_version",
        "artifact_role",
        "status",
        "source_authority_sha256",
        "train_development_outcomes_sha256",
        "development_projection_hashes",
        "code_tree_sha256",
        "ordinary_only",
        "seeds",
        "seed_aggregation",
        "estimand_training_scope",
        "model",
        "optimizer",
        "loss",
        "checkpoint_selection",
        "training",
        "prior_policy",
        "ridge_policy",
        "hgb_policy",
        "nearest_neighbor_policy",
        "resource_budget",
    }
    raw = _strict_keys(payload, keys, label="executable config v4")
    expected = {
        "schema_version": EXECUTABLE_CONFIG_SCHEMA,
        "artifact_role": EXECUTABLE_CONFIG_ROLE,
        "status": EXECUTABLE_CONFIG_STATUS,
        "ordinary_only": True,
        "seed_aggregation": "FIXED_PREDICTION_ENSEMBLE",
        "estimand_training_scope": "INDEPENDENT_MODEL_PER_PRIMARY_ESTIMAND",
    }
    for field, value in expected.items():
        _exact(raw.get(field), value, label=f"executable config.{field}")
    source_sha = _digest(
        raw.get("source_authority_sha256"), label="config source authority"
    )
    data_sha = _digest(
        raw.get("train_development_outcomes_sha256"),
        label="config train/development outcomes",
    )
    code_sha = _digest(raw.get("code_tree_sha256"), label="config code tree")
    projections = _projection_hashes(
        raw.get("development_projection_hashes"),
        label="config development projections",
    )
    if {source_sha, data_sha, code_sha} & set(projections.values()):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_BINDING",
            "config source/data/code roles cannot alias projection bytes",
        )
    if raw.get("seeds") != list(FROZEN_SEEDS):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_CONFIG",
            "executable config must freeze exactly five registered seeds",
        )
    _validate_model_config(raw.get("model"))

    optimizer = _strict_keys(
        raw.get("optimizer"),
        {"name", "learning_rate", "weight_decay"},
        label="executable config.optimizer",
    )
    _exact(optimizer.get("name"), "Adam", label="optimizer.name")
    _finite_number(
        optimizer.get("learning_rate"),
        label="optimizer.learning_rate",
        minimum=0.0,
        maximum=1.0,
        open_minimum=True,
    )
    _finite_number(
        optimizer.get("weight_decay"),
        label="optimizer.weight_decay",
        minimum=0.0,
        maximum=1.0,
    )

    loss = _strict_keys(
        raw.get("loss"),
        {"huber_weight", "nll_weight", "sign_weight", "sign_temperature"},
        label="executable config.loss",
    )
    normalized_loss_weights = {}
    for field in ("huber_weight", "nll_weight", "sign_weight"):
        normalized_loss_weights[field] = _finite_number(
            loss.get(field), label=f"loss.{field}", minimum=0.0, maximum=100.0
        )
    if normalized_loss_weights["sign_weight"] <= 0.0:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_CONFIG",
            "the frozen primary sign objective requires sign_weight > 0",
        )
    if sum(normalized_loss_weights.values()) <= 0.0:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_CONFIG",
            "at least one training loss weight must be positive",
        )
    _finite_number(
        loss.get("sign_temperature"),
        label="loss.sign_temperature",
        minimum=0.0,
        maximum=100.0,
        open_minimum=True,
    )

    selection = _strict_keys(
        raw.get("checkpoint_selection"),
        {"rule", "selection_partition"},
        label="executable config.checkpoint_selection",
    )
    _exact(
        selection.get("rule"),
        "SIGN_GATE_THEN_SPEARMAN_THEN_NLL_THEN_EARLIEST_EPOCH",
        label="checkpoint_selection.rule",
    )
    _exact(
        selection.get("selection_partition"),
        ORDINARY_DEVELOPMENT,
        label="checkpoint_selection.selection_partition",
    )

    training = _strict_keys(
        raw.get("training"),
        {
            "epochs",
            "batch_size",
            "device_class",
            "gradient_clip_norm",
            "condition_unknown_dropout",
        },
        label="executable config.training",
    )
    _bounded_int(training.get("epochs"), label="training.epochs", minimum=1, maximum=100000)
    _bounded_int(
        training.get("batch_size"), label="training.batch_size", minimum=1, maximum=1000000
    )
    if training.get("device_class") not in {"CPU", "CUDA"}:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_CONFIG",
            "training.device_class must be frozen as CPU or CUDA",
        )
    _finite_number(
        training.get("gradient_clip_norm"),
        label="training.gradient_clip_norm",
        minimum=0.0,
        maximum=1_000_000.0,
        open_minimum=True,
    )
    _finite_number(
        training.get("condition_unknown_dropout"),
        label="training.condition_unknown_dropout",
        minimum=0.0,
        maximum=1.0,
        open_minimum=True,
        open_maximum=True,
    )

    prior = _strict_keys(
        raw.get("prior_policy"),
        {
            "training_weight_policy",
            "weighted_risk_normalization",
            "aggregation_policy",
        },
        label="executable config.prior_policy",
    )
    _exact(
        prior.get("training_weight_policy"),
        FORMAL_SCALAR_WEIGHT_POLICY,
        label="prior.training_weight_policy",
    )
    _exact(
        prior.get("weighted_risk_normalization"),
        "SUM_TO_ONE",
        label="prior.weighted_risk_normalization",
    )
    _exact(
        prior.get("aggregation_policy"),
        PRIOR_TRAINING_POLICY,
        label="prior.aggregation_policy",
    )

    ridge = _strict_keys(
        raw.get("ridge_policy"),
        {
            "feature_schema_by_method",
            "solver_policy",
            "training_weight_policy",
            "weighted_risk_normalization",
            "standardization_policy",
            "intercept_policy",
            "objective_policy",
            "alpha_grid",
            "max_component_folds",
            "minimum_components_for_cv",
            "standardization_floor",
            "selection_metric",
            "tie_rule",
        },
        label="executable config.ridge_policy",
    )
    expected_ridge_features = {
        "EDIT_ONLY_RIDGE": EDIT_ONLY_FEATURE_SCHEMA,
        "CANDIDATE_KMER_RIDGE": CANDIDATE_KMER_FEATURE_SCHEMA,
        "PAIRED_LOCAL_GLOBAL_KMER_RIDGE": PAIRED_KMER_FEATURE_SCHEMA,
    }
    _exact(
        ridge.get("feature_schema_by_method"),
        expected_ridge_features,
        label="ridge.feature_schema_by_method",
    )
    _exact(
        ridge.get("solver_policy"),
        RIDGE_SOLVER_POLICY,
        label="ridge.solver_policy",
    )
    _exact(
        ridge.get("training_weight_policy"),
        FORMAL_SCALAR_WEIGHT_POLICY,
        label="ridge.training_weight_policy",
    )
    _exact(
        ridge.get("weighted_risk_normalization"),
        "SUM_TO_ONE",
        label="ridge.weighted_risk_normalization",
    )
    _exact(
        ridge.get("standardization_policy"),
        "FORMAL_WEIGHTED_TRAIN_ONLY_MEAN_VARIANCE_V2",
        label="ridge.standardization_policy",
    )
    _exact(
        ridge.get("intercept_policy"),
        "FORMAL_WEIGHTED_TARGET_MEAN_UNPENALIZED_V1",
        label="ridge.intercept_policy",
    )
    _exact(
        ridge.get("objective_policy"),
        RIDGE_TRAINING_WEIGHT_POLICY,
        label="ridge.objective_policy",
    )
    alpha_grid = ridge.get("alpha_grid")
    if not isinstance(alpha_grid, list) or not alpha_grid:
        _hold("HOLD_DEVELOPMENT_RUN_PLAN_SCHEMA", "ridge alpha_grid must be non-empty")
    normalized_alphas = [
        _finite_number(
            value,
            label="ridge alpha",
            minimum=0.0,
            maximum=1_000_000.0,
            open_minimum=True,
        )
        for value in alpha_grid
    ]
    _exact(
        normalized_alphas,
        [0.01, 0.1, 1.0, 10.0, 100.0],
        label="ridge.alpha_grid",
    )
    _exact(ridge.get("max_component_folds"), 5, label="ridge.max_component_folds")
    _exact(
        ridge.get("minimum_components_for_cv"),
        3,
        label="ridge.minimum_components_for_cv",
    )
    _exact(
        ridge.get("standardization_floor"),
        1e-12,
        label="ridge.standardization_floor",
    )
    _exact(
        ridge.get("selection_metric"),
        RIDGE_CV_SCORING_POLICY,
        label="ridge.selection_metric",
    )
    _exact(ridge.get("tie_rule"), "LOWEST_ALPHA", label="ridge.tie_rule")

    hgb = _strict_keys(
        raw.get("hgb_policy"),
        {
            "feature_set",
            "feature_schema",
            "training_weight_policy",
            "weight_normalization",
            "learning_rate",
            "max_iter",
            "max_leaf_nodes",
            "l2_regularization",
            "random_state",
            "minimum_train_rows",
            "early_stopping",
            "qualification_sha256",
        },
        label="executable config.hgb_policy",
    )
    _exact(
        hgb.get("feature_set"),
        "PAIRED_LOCAL_GLOBAL_KMER_WITH_CONDITION_UNK",
        label="hgb.feature_set",
    )
    _exact(
        hgb.get("feature_schema"),
        PAIRED_KMER_FEATURE_SCHEMA,
        label="hgb.feature_schema",
    )
    _exact(
        hgb.get("training_weight_policy"),
        HGB_TRAINING_WEIGHT_POLICY,
        label="hgb.training_weight_policy",
    )
    _exact(
        hgb.get("weight_normalization"),
        HGB_WEIGHT_NORMALIZATION,
        label="hgb.weight_normalization",
    )
    _exact(hgb.get("learning_rate"), 0.05, label="hgb.learning_rate")
    _exact(hgb.get("max_iter"), 200, label="hgb.max_iter")
    _exact(hgb.get("max_leaf_nodes"), 15, label="hgb.max_leaf_nodes")
    _exact(
        hgb.get("l2_regularization"), 1.0, label="hgb.l2_regularization"
    )
    _exact(hgb.get("random_state"), 42, label="hgb.random_state")
    _exact(hgb.get("minimum_train_rows"), 100, label="hgb.minimum_train_rows")
    if not isinstance(hgb.get("early_stopping"), bool):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_CONFIG",
            "hgb.early_stopping must be a boolean",
        )
    _exact(hgb.get("early_stopping"), False, label="hgb.early_stopping")
    _digest(hgb.get("qualification_sha256"), label="hgb qualification")

    nearest = _strict_keys(
        raw.get("nearest_neighbor_policy"),
        {
            "feature_set",
            "feature_schema",
            "distance",
            "scaling",
            "training_weight_policy",
            "weighted_risk_normalization",
            "standardization_floor",
            "tie_absolute_tolerance",
            "tie_relative_tolerance",
            "tie_aggregation",
            "search_policy",
            "train_chunk_size",
        },
        label="executable config.nearest_neighbor_policy",
    )
    expected_nearest = {
        "feature_set": "PAIRED_LOCAL_GLOBAL_KMER_WITH_CONDITION_UNK",
        "feature_schema": PAIRED_KMER_FEATURE_SCHEMA,
        "distance": "SQUARED_EUCLIDEAN",
        "scaling": NEAREST_NEIGHBOR_SCALING_POLICY,
        "training_weight_policy": FORMAL_SCALAR_WEIGHT_POLICY,
        "weighted_risk_normalization": "SUM_TO_ONE",
        "tie_aggregation": NEAREST_NEIGHBOR_TIE_AGGREGATION_POLICY,
        "search_policy": NEAREST_NEIGHBOR_SEARCH_POLICY,
        "train_chunk_size": 4096,
    }
    for field, value in expected_nearest.items():
        _exact(nearest.get(field), value, label=f"nearest_neighbor.{field}")
    _exact(
        nearest.get("standardization_floor"),
        1e-12,
        label="nearest_neighbor.standardization_floor",
    )
    _exact(
        nearest.get("tie_absolute_tolerance"),
        1e-12,
        label="nearest_neighbor.tie_absolute_tolerance",
    )
    _exact(
        nearest.get("tie_relative_tolerance"),
        0.0,
        label="nearest_neighbor.tie_relative_tolerance",
    )

    budget = _strict_keys(
        raw.get("resource_budget"),
        {
            "max_wall_time_seconds_per_fit",
            "max_peak_memory_bytes_per_fit",
            "max_forward_equivalents_per_fit",
            "max_total_fit_units",
            "max_estimands",
            "max_rows",
            "max_sequence_tokens_per_batch",
        },
        label="executable config.resource_budget",
    )
    for field in (
        "max_wall_time_seconds_per_fit",
        "max_peak_memory_bytes_per_fit",
        "max_forward_equivalents_per_fit",
        "max_total_fit_units",
        "max_estimands",
        "max_rows",
        "max_sequence_tokens_per_batch",
    ):
        _bounded_int(
            budget.get(field), label=f"resource_budget.{field}", minimum=1, maximum=10**18
        )
    return copy.deepcopy(dict(raw))


def validate_row_inventory(payload: Mapping) -> dict:
    _scan_protected(payload)
    keys = {
        "schema_version",
        "artifact_role",
        "status",
        "capability_scope",
        "ordinary_only",
        "train_development_outcomes_sha256",
        "source_authority_sha256",
        "development_projection_hashes",
        "partition_row_set_sha256",
        "rows",
    }
    raw = _strict_keys(payload, keys, label="row inventory")
    expected = {
        "schema_version": ROW_INVENTORY_SCHEMA,
        "artifact_role": ROW_INVENTORY_ROLE,
        "status": ROW_INVENTORY_STATUS,
        "capability_scope": CAPABILITY_SCOPE,
        "ordinary_only": True,
    }
    for field, value in expected.items():
        _exact(raw.get(field), value, label=f"row inventory.{field}")
    data_sha = _digest(
        raw.get("train_development_outcomes_sha256"),
        label="row inventory data authority",
    )
    source_raw = _strict_keys(
        raw.get("source_authority_sha256"),
        set(ROW_SOURCE_NAMES),
        label="row inventory source authorities",
    )
    source_hashes = {
        name: _digest(source_raw[name], label=f"row source {name}")
        for name in ROW_SOURCE_NAMES
    }
    projections = _projection_hashes(
        raw.get("development_projection_hashes"),
        label="row inventory development projections",
    )
    partition_raw = _strict_keys(
        raw.get("partition_row_set_sha256"),
        set(PARTITIONS),
        label="row inventory partition hashes",
    )
    partition_hashes = {
        partition: _digest(
            partition_raw[partition], label=f"row set {partition}"
        )
        for partition in PARTITIONS
    }
    role_hashes = [data_sha, *source_hashes.values(), *projections.values()]
    if len(role_hashes) != len(set(role_hashes)):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_BINDING",
            "data, source, and projection bindings must be globally unique",
        )
    if len(set(partition_hashes.values())) != len(PARTITIONS):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_BINDING",
            "ordinary train and development row sets cannot alias",
        )

    rows = raw.get("rows")
    if not isinstance(rows, list) or not rows:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_SCHEMA",
            "row inventory rows must be a non-empty list",
        )
    row_keys = {
        "row_id",
        "partition",
        "study",
        "endpoint",
        "estimand_id",
        "source_component_id",
        "source_group_id_v2",
    }
    normalized_rows: list[dict] = []
    component_partition: dict[str, str] = {}
    group_partition: dict[str, str] = {}
    for index, item in enumerate(rows):
        row = _strict_keys(item, row_keys, label=f"row inventory row {index}")
        row_id = _digest(row.get("row_id"), label=f"row {index} row_id")
        partition = row.get("partition")
        if partition not in PARTITIONS:
            _hold(
                "HOLD_DEVELOPMENT_RUN_PLAN_PARTITION",
                "row inventory can express only ordinary train/development partitions",
            )
        for field in (
            "study",
            "endpoint",
            "estimand_id",
            "source_component_id",
            "source_group_id_v2",
        ):
            _identifier(row.get(field), label=f"row {index}.{field}")
        component = str(row["source_component_id"])
        previous = component_partition.setdefault(component, str(partition))
        if previous != partition:
            _hold(
                "HOLD_DEVELOPMENT_RUN_PLAN_BINDING",
                "one source component crosses ordinary partitions",
                source_component_id=component,
            )
        group = str(row["source_group_id_v2"])
        previous_group_partition = group_partition.setdefault(group, str(partition))
        if previous_group_partition != partition:
            _hold(
                "HOLD_DEVELOPMENT_RUN_PLAN_BINDING",
                "one source ranking group crosses ordinary partitions",
                source_group_id_v2=group,
            )
        normalized_rows.append(copy.deepcopy(dict(row)))
        if normalized_rows[-1]["row_id"] != row_id:
            raise AssertionError("unreachable normalized row identity drift")
    if normalized_rows != sorted(normalized_rows, key=lambda item: item["row_id"]):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_SCHEMA",
            "row inventory rows must be sorted by row_id",
        )
    row_ids = [row["row_id"] for row in normalized_rows]
    if len(row_ids) != len(set(row_ids)):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_BINDING",
            "row inventory row IDs must be unique",
        )
    for partition in PARTITIONS:
        ids = sorted(
            row["row_id"] for row in normalized_rows if row["partition"] == partition
        )
        if not ids:
            _hold(
                "HOLD_DEVELOPMENT_RUN_PLAN_PARTITION",
                f"row inventory has no rows for {partition}",
            )
        if canonical_sha256(ids) != partition_hashes[partition]:
            _hold(
                "HOLD_DEVELOPMENT_RUN_PLAN_BINDING",
                f"row-set commitment is stale for {partition}",
            )
    return copy.deepcopy(dict(raw))


def validate_development_release(payload: Mapping) -> dict:
    _scan_protected(payload)
    keys = {
        "schema_version",
        "artifact_role",
        "status",
        "release_id",
        "release_scope",
        "producer_id",
        "verifier_id",
        "verifier_code_sha256",
        "approved_code_tree_sha256",
        "approved_artifact_sha256",
        "projection_binding_sha256",
        "independence_status",
        "plan_construction_authorized",
        "training_execution_authorized",
        "gpu_execution_authorized",
        "future_stage_access_authorized",
        "ordinary_only",
    }
    raw = _strict_keys(payload, keys, label="independent development release")
    expected = {
        "schema_version": DEVELOPMENT_RELEASE_SCHEMA,
        "artifact_role": DEVELOPMENT_RELEASE_ROLE,
        "status": DEVELOPMENT_RELEASE_STATUS,
        "release_scope": "ORDINARY_TRAIN_AND_ORDINARY_DEVELOPMENT_PLAN_ONLY",
        "independence_status": "SELF_REPORTED_DISTINCT_IDS_NOT_TRUST_ROOT_VERIFIED",
        "plan_construction_authorized": True,
        "training_execution_authorized": False,
        "gpu_execution_authorized": False,
        "future_stage_access_authorized": False,
        "ordinary_only": True,
    }
    for field, value in expected.items():
        _exact(raw.get(field), value, label=f"development release.{field}")
    _identifier(raw.get("release_id"), label="development release.release_id")
    producer = _identifier(raw.get("producer_id"), label="development release.producer_id")
    verifier = _identifier(raw.get("verifier_id"), label="development release.verifier_id")
    if producer == verifier:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_RELEASE",
            "development release producer and verifier must be distinct",
        )
    _digest(
        raw.get("verifier_code_sha256"), label="development release verifier code"
    )
    _digest(
        raw.get("approved_code_tree_sha256"), label="development release code tree"
    )
    _digest(
        raw.get("projection_binding_sha256"),
        label="development release projection binding",
    )
    approved_raw = _strict_keys(
        raw.get("approved_artifact_sha256"),
        set(APPROVED_RELEASE_REF_NAMES),
        label="development release approved artifacts",
    )
    approved = [
        _digest(approved_raw[name], label=f"approved artifact {name}")
        for name in APPROVED_RELEASE_REF_NAMES
    ]
    if len(approved) != len(set(approved)):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_BINDING",
            "development release approved artifacts must be distinct",
        )
    return copy.deepcopy(dict(raw))


def _build_estimand_slices(
    rows: Sequence[Mapping],
    estimand_registry: Mapping,
) -> list[dict]:
    entries = [
        dict(entry)
        for entry in estimand_registry["entries"]
        if entry.get("role") == "PRIMARY"
    ]
    if not entries:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_ESTIMAND",
            "typed estimand registry has no primary entries",
        )
    by_key: dict[tuple[str, str], dict] = {}
    for entry in entries:
        key = (str(entry["study"]), str(entry["endpoint"]))
        if key in by_key:
            _hold(
                "HOLD_DEVELOPMENT_RUN_PLAN_ESTIMAND",
                "typed estimand study/endpoint keys are duplicated",
            )
        by_key[key] = entry

    support: dict[tuple[str, str], set[str]] = {key: set() for key in by_key}
    by_estimand: dict[str, dict[str, list[str]]] = {}
    registered: dict[str, set[tuple[str, str]]] = {}
    for key, entry in by_key.items():
        registered.setdefault(str(entry["estimand_id"]), set()).add(key)
    for row in rows:
        key = (str(row["study"]), str(row["endpoint"]))
        entry = by_key.get(key)
        if entry is None or row["estimand_id"] != entry["estimand_id"]:
            _hold(
                "HOLD_DEVELOPMENT_RUN_PLAN_ESTIMAND",
                "row is not covered by the typed primary estimand registry",
                row_id=row["row_id"],
            )
        support[key].add(str(row["partition"]))
        grouped = by_estimand.setdefault(
            str(row["estimand_id"]), {partition: [] for partition in PARTITIONS}
        )
        grouped[str(row["partition"])].append(str(row["row_id"]))
    for key, partitions in support.items():
        if partitions != set(PARTITIONS):
            _hold(
                "HOLD_DEVELOPMENT_RUN_PLAN_ESTIMAND",
                "every typed primary study/endpoint requires both ordinary partitions",
                study=key[0],
                endpoint=key[1],
            )

    slices: list[dict] = []
    for estimand_id in sorted(registered):
        grouped = by_estimand.get(estimand_id)
        if grouped is None or any(not grouped[partition] for partition in PARTITIONS):
            _hold(
                "HOLD_DEVELOPMENT_RUN_PLAN_ESTIMAND",
                "every typed primary estimand requires both ordinary partitions",
                estimand_id=estimand_id,
            )
        core = {
            "estimand_id": estimand_id,
            "registered_study_endpoints": [
                {"study": study, "endpoint": endpoint}
                for study, endpoint in sorted(registered[estimand_id])
            ],
            "partitions": [
                {
                    "partition": partition,
                    "row_ids": sorted(grouped[partition]),
                }
                for partition in PARTITIONS
            ],
        }
        slices.append({**core, "slice_sha256": canonical_sha256(core)})
    return slices


def _planning_budget_summary(
    slices: Sequence[Mapping],
    config: Mapping,
) -> dict:
    """Conservatively count planned fit units without claiming executability."""

    ridge_policy = config["ridge_policy"]
    hgb_policy = config["hgb_policy"]
    training = config["training"]
    model = config["model"]
    budget = config["resource_budget"]
    per_estimand: list[dict] = []
    for item in slices:
        partition_rows = {
            part["partition"]: len(part["row_ids"])
            for part in item["partitions"]
        }
        train_rows = partition_rows[ORDINARY_TRAIN]
        development_rows = partition_rows[ORDINARY_DEVELOPMENT]
        if train_rows < hgb_policy["minimum_train_rows"]:
            _hold(
                "HOLD_DEVELOPMENT_RUN_PLAN_HGB_SUPPORT",
                "an estimand has fewer training rows than the frozen HGB minimum",
                estimand_id=item["estimand_id"],
                train_rows=train_rows,
                required_rows=hgb_policy["minimum_train_rows"],
            )
        if train_rows < ridge_policy["minimum_components_for_cv"]:
            _hold(
                "HOLD_DEVELOPMENT_RUN_PLAN_RIDGE_SUPPORT",
                "an estimand cannot support the frozen component-CV minimum",
                estimand_id=item["estimand_id"],
            )
        # Row count is an upper bound on component count.  Using it here makes
        # the planning budget conservative; the future executable compiler must
        # replay the exact component assignment before fitting.
        ridge_cv_folds = min(
            ridge_policy["max_component_folds"],
            train_rows,
        )
        ridge_fit_units = len(RIDGE_METHOD_IDS) * (
            ridge_cv_folds * len(ridge_policy["alpha_grid"]) + 1
        )
        scalar_fit_units = ridge_fit_units + len(SINGLE_FIT_METHOD_IDS)
        neural_fit_units = len(FROZEN_SEEDS)
        total_fit_units = scalar_fit_units + neural_fit_units
        per_estimand.append(
            {
                "estimand_id": item["estimand_id"],
                "train_row_count": train_rows,
                "development_row_count": development_rows,
                "ridge_cv_fold_upper_bound": ridge_cv_folds,
                "scalar_fit_unit_upper_bound": scalar_fit_units,
                "neural_fit_units": neural_fit_units,
                "total_fit_unit_upper_bound": total_fit_units,
            }
        )
    total_fit_units = sum(item["total_fit_unit_upper_bound"] for item in per_estimand)
    if total_fit_units > budget["max_total_fit_units"]:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_RESOURCE_BUDGET",
            "the conservative fit-unit count exceeds the frozen total budget",
            planned_fit_units=total_fit_units,
            maximum_fit_units=budget["max_total_fit_units"],
        )
    tokens_per_batch = 2 * training["batch_size"] * model["max_len"]
    if tokens_per_batch > budget["max_sequence_tokens_per_batch"]:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_RESOURCE_BUDGET",
            "source/candidate tokens per batch exceed the frozen budget",
            planned_tokens=tokens_per_batch,
            maximum_tokens=budget["max_sequence_tokens_per_batch"],
        )
    return {
        "status": "CONSERVATIVE_PLAN_BUDGET_ONLY_NOT_IMPLEMENTATION_DRY_RUN",
        "per_estimand": per_estimand,
        "total_fit_unit_upper_bound": total_fit_units,
        "source_candidate_tokens_per_batch": tokens_per_batch,
        "implementation_dry_run_completed": False,
    }


def _cross_bind(
    *,
    refs: Mapping[str, Mapping[str, str]],
    release: Mapping,
    hgb: Mapping,
    paired: Mapping,
    estimand: Mapping,
    config: Mapping,
    inventory: Mapping,
) -> dict:
    approved = release["approved_artifact_sha256"]
    expected_approved = {
        name: refs[name]["sha256"] for name in APPROVED_RELEASE_REF_NAMES
    }
    if approved != expected_approved:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_RELEASE",
            "development release binds different input artifacts",
        )
    projection_hashes = dict(inventory["development_projection_hashes"])
    if release["projection_binding_sha256"] != canonical_sha256(projection_hashes):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_RELEASE",
            "development release projection binding is stale",
        )
    if release["approved_code_tree_sha256"] != config["code_tree_sha256"]:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_RELEASE",
            "development release binds a different code tree",
        )
    if hgb["code_tree_sha256"] != config["code_tree_sha256"]:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_BINDING",
            "HGB qualification and executable config bind different code trees",
        )
    hgb_sha = refs["hgb_qualification"]["sha256"]
    if (
        config["hgb_policy"]["qualification_sha256"] != hgb_sha
        or paired["frozen_environment_execution_qualification_sha256"] != hgb_sha
    ):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_BINDING",
            "HGB qualification is not exact-bound by config and paired policy",
        )

    data_hashes = {
        paired["train_development_outcomes_sha256"],
        estimand["train_development_outcomes_sha256"],
        config["train_development_outcomes_sha256"],
        inventory["train_development_outcomes_sha256"],
    }
    if len(data_hashes) != 1:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_BINDING",
            "typed authorities bind different train/development bytes",
        )
    for authority in (paired, estimand, config):
        if authority["development_projection_hashes"] != projection_hashes:
            _hold(
                "HOLD_DEVELOPMENT_RUN_PLAN_BINDING",
                "typed authorities bind different development projections",
            )

    authority_sources = {
        "paired_baseline_policy": paired["source_authority_sha256"],
        "endpoint_estimand_registry": estimand["source_authority_sha256"],
        "executable_config": config["source_authority_sha256"],
        **{
            name: inventory["source_authority_sha256"][name]
            for name in ROW_SOURCE_NAMES
        },
    }
    semantic_hashes = [
        next(iter(data_hashes)),
        *authority_sources.values(),
        *projection_hashes.values(),
    ]
    if len(semantic_hashes) != len(set(semantic_hashes)):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_BINDING",
            "data, source-authority, and projection roles cannot alias bytes",
        )
    artifact_hashes = [ref["sha256"] for ref in refs.values()]
    if set(artifact_hashes) & set(semantic_hashes):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_BINDING",
            "artifact and semantic source/projection roles cannot alias bytes",
        )
    return {
        "train_development_outcomes_sha256": next(iter(data_hashes)),
        "authority_source_sha256": authority_sources,
        "development_projection_hashes": projection_hashes,
    }


def build_development_run_plan(
    run_root: Path,
    *,
    source_refs: Mapping,
    expected_development_release_sha256: str,
) -> dict:
    """Exact-load a caller-bound DAG and build a non-authorizing scaffold."""

    root = _checked_run_root(run_root)
    refs = _normalize_source_refs(source_refs)
    expected_release_sha = _digest(
        expected_development_release_sha256,
        label="externally pre-bound development release",
    )
    if refs["independent_development_release"]["sha256"] != expected_release_sha:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_RELEASE",
            "development release ref differs from the externally pre-bound digest",
        )

    release = _load_new_json_ref(
        root,
        refs["independent_development_release"],
        label="independent development release",
        validator=validate_development_release,
    )
    hgb = _load_new_json_ref(
        root,
        refs["hgb_qualification"],
        label="HGB execution qualification",
        validator=validate_hgb_qualification,
    )
    paired = _load_typed_ref(
        root,
        refs["paired_baseline_policy"],
        authority_name=PAIRED_BASELINE_POLICY,
        label="typed paired baseline policy",
    )
    estimand = _load_typed_ref(
        root,
        refs["endpoint_estimand_registry"],
        authority_name=ENDPOINT_ESTIMAND_REGISTRY,
        label="typed endpoint estimand registry",
    )
    config = _load_new_json_ref(
        root,
        refs["executable_config"],
        label="development executable config",
        validator=validate_executable_config,
    )
    inventory = _load_new_json_ref(
        root,
        refs["row_inventory"],
        label="typed development row inventory",
        validator=validate_row_inventory,
    )
    bindings = _cross_bind(
        refs=refs,
        release=release,
        hgb=hgb,
        paired=paired,
        estimand=estimand,
        config=config,
        inventory=inventory,
    )

    baseline_order = list(paired["eligible_baseline_method_ids"])
    if (
        baseline_order != list(FROZEN_BASELINE_METHOD_ORDER)
        or paired["required_baseline_method_ids"] != baseline_order
    ):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_BASELINE_SET",
            "typed paired policy does not exact-bind the eight-baseline order",
        )
    if config["seeds"] != list(FROZEN_SEEDS):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_CONFIG",
            "executable config seeds changed after validation",
        )
    slices = _build_estimand_slices(inventory["rows"], estimand)
    budget = config["resource_budget"]
    if len(slices) > budget["max_estimands"] or len(inventory["rows"]) > budget["max_rows"]:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_RESOURCE_BUDGET",
            "typed row inventory exceeds the frozen resource budget",
        )
    planning_budget = _planning_budget_summary(slices, config)

    core = {
        "schema_version": RUN_PLAN_SCHEMA,
        "artifact_role": RUN_PLAN_ROLE,
        "status": RUN_PLAN_STATUS,
        "capability_scope": CAPABILITY_SCOPE,
        "source_refs": copy.deepcopy(refs),
        "external_release_sha256": expected_release_sha,
        "train_development_outcomes_sha256": bindings[
            "train_development_outcomes_sha256"
        ],
        "authority_source_sha256": bindings["authority_source_sha256"],
        "development_projection_hashes": bindings[
            "development_projection_hashes"
        ],
        "partition_row_set_sha256": copy.deepcopy(
            inventory["partition_row_set_sha256"]
        ),
        "eligible_baseline_method_order": baseline_order,
        "seeds": list(FROZEN_SEEDS),
        "estimand_slices": slices,
        "planning_budget_summary": planning_budget,
        "executable_config": copy.deepcopy(config),
        "scientific_policy": {
            "primary_sign_gate": SIGN_ACCURACY_GATE,
            "information_set": paired["information_set"],
            "selector_required": paired["development_selector_required"],
            "selector_schema": paired["development_selector_schema"],
            "seed_aggregation": config["seed_aggregation"],
            "estimand_training_scope": config["estimand_training_scope"],
        },
        "execution_state": {
            "ordinary_only": True,
            "training_ready": False,
            "execution_authorized": False,
            "external_trust_root_verified": False,
            "development_capability_bound": False,
            "training_executed": False,
            "gpu_used": False,
            "network_used": False,
            "unreleased_partition_access": False,
            "unreleased_outcome_access": False,
            "future_stage_access": False,
            "publication_claim_authorized": False,
        },
    }
    plan = {**core, "plan_sha256": canonical_sha256(core)}
    return validate_development_run_plan(plan)


def _validate_estimand_slices(value: object) -> list[dict]:
    if not isinstance(value, list) or not value:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_SCHEMA",
            "estimand_slices must be a non-empty list",
        )
    normalized: list[dict] = []
    seen_estimands: set[str] = set()
    seen_rows: set[str] = set()
    for index, item in enumerate(value):
        raw = _strict_keys(
            item,
            {"estimand_id", "registered_study_endpoints", "partitions", "slice_sha256"},
            label=f"estimand slice {index}",
        )
        estimand_id = _identifier(raw.get("estimand_id"), label=f"slice {index}.estimand_id")
        if estimand_id in seen_estimands:
            _hold(
                "HOLD_DEVELOPMENT_RUN_PLAN_ESTIMAND",
                "estimand slices must be unique",
            )
        seen_estimands.add(estimand_id)
        endpoints = raw.get("registered_study_endpoints")
        if not isinstance(endpoints, list) or not endpoints:
            _hold(
                "HOLD_DEVELOPMENT_RUN_PLAN_SCHEMA",
                "registered_study_endpoints must be non-empty",
            )
        normalized_endpoints = []
        for endpoint_index, endpoint in enumerate(endpoints):
            entry = _strict_keys(
                endpoint,
                {"study", "endpoint"},
                label=f"slice {index} endpoint {endpoint_index}",
            )
            normalized_endpoints.append(
                {
                    "study": _identifier(entry.get("study"), label="registered study"),
                    "endpoint": _identifier(
                        entry.get("endpoint"), label="registered endpoint"
                    ),
                }
            )
        if normalized_endpoints != sorted(
            normalized_endpoints, key=lambda entry: (entry["study"], entry["endpoint"])
        ) or len({(entry["study"], entry["endpoint"]) for entry in normalized_endpoints}) != len(
            normalized_endpoints
        ):
            _hold(
                "HOLD_DEVELOPMENT_RUN_PLAN_SCHEMA",
                "registered study/endpoints must be sorted and unique",
            )
        partitions = raw.get("partitions")
        if not isinstance(partitions, list) or len(partitions) != len(PARTITIONS):
            _hold(
                "HOLD_DEVELOPMENT_RUN_PLAN_PARTITION",
                "each estimand requires exactly two ordinary partitions",
            )
        normalized_partitions = []
        for expected_partition, partition_value in zip(PARTITIONS, partitions):
            partition = _strict_keys(
                partition_value,
                {"partition", "row_ids"},
                label=f"slice {index} partition",
            )
            _exact(
                partition.get("partition"),
                expected_partition,
                label=f"slice {index} partition literal",
            )
            ids = partition.get("row_ids")
            if not isinstance(ids, list) or not ids:
                _hold(
                    "HOLD_DEVELOPMENT_RUN_PLAN_PARTITION",
                    "each ordinary estimand partition requires row IDs",
                )
            normalized_ids = [
                _digest(row_id, label=f"slice {index} row_id") for row_id in ids
            ]
            if normalized_ids != sorted(set(normalized_ids)):
                _hold(
                    "HOLD_DEVELOPMENT_RUN_PLAN_BINDING",
                    "slice row IDs must be sorted and unique",
                )
            if seen_rows.intersection(normalized_ids):
                _hold(
                    "HOLD_DEVELOPMENT_RUN_PLAN_BINDING",
                    "one row cannot appear in multiple estimand slices",
                )
            seen_rows.update(normalized_ids)
            normalized_partitions.append(
                {"partition": expected_partition, "row_ids": normalized_ids}
            )
        core = {
            "estimand_id": estimand_id,
            "registered_study_endpoints": normalized_endpoints,
            "partitions": normalized_partitions,
        }
        if raw.get("slice_sha256") != canonical_sha256(core):
            _hold(
                "HOLD_DEVELOPMENT_RUN_PLAN_BINDING",
                "estimand slice digest is stale",
            )
        normalized.append({**core, "slice_sha256": raw["slice_sha256"]})
    if [item["estimand_id"] for item in normalized] != sorted(seen_estimands):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_SCHEMA",
            "estimand slices must be canonically sorted",
        )
    return normalized


def validate_development_run_plan(payload: Mapping) -> dict:
    _scan_protected(payload)
    keys = {
        "schema_version",
        "artifact_role",
        "status",
        "capability_scope",
        "source_refs",
        "external_release_sha256",
        "train_development_outcomes_sha256",
        "authority_source_sha256",
        "development_projection_hashes",
        "partition_row_set_sha256",
        "eligible_baseline_method_order",
        "seeds",
        "estimand_slices",
        "planning_budget_summary",
        "executable_config",
        "scientific_policy",
        "execution_state",
        "plan_sha256",
    }
    raw = _strict_keys(payload, keys, label="development run plan")
    expected = {
        "schema_version": RUN_PLAN_SCHEMA,
        "artifact_role": RUN_PLAN_ROLE,
        "status": RUN_PLAN_STATUS,
        "capability_scope": CAPABILITY_SCOPE,
    }
    for field, value in expected.items():
        _exact(raw.get(field), value, label=f"run plan.{field}")
    refs = _normalize_source_refs(raw.get("source_refs"))
    external_release = _digest(
        raw.get("external_release_sha256"), label="run plan external release"
    )
    if refs["independent_development_release"]["sha256"] != external_release:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_RELEASE",
            "run plan external release binding is inconsistent",
        )
    data_sha = _digest(
        raw.get("train_development_outcomes_sha256"),
        label="run plan train/development outcomes",
    )
    source_raw = _strict_keys(
        raw.get("authority_source_sha256"),
        {
            "paired_baseline_policy",
            "endpoint_estimand_registry",
            "executable_config",
            *ROW_SOURCE_NAMES,
        },
        label="run plan authority source hashes",
    )
    authority_sources = {
        name: _digest(source_raw[name], label=f"run plan source {name}")
        for name in source_raw
    }
    projections = _projection_hashes(
        raw.get("development_projection_hashes"),
        label="run plan development projections",
    )
    semantic_hashes = [data_sha, *authority_sources.values(), *projections.values()]
    if len(semantic_hashes) != len(set(semantic_hashes)):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_BINDING",
            "run plan data/source/projection roles cannot alias bytes",
        )
    if set(semantic_hashes) & {ref["sha256"] for ref in refs.values()}:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_BINDING",
            "run plan artifact and semantic roles cannot alias bytes",
        )
    partition_raw = _strict_keys(
        raw.get("partition_row_set_sha256"),
        set(PARTITIONS),
        label="run plan partition row-set hashes",
    )
    partition_hashes = {
        partition: _digest(
            partition_raw[partition], label=f"run plan row set {partition}"
        )
        for partition in PARTITIONS
    }
    order = raw.get("eligible_baseline_method_order")
    if order != list(FROZEN_BASELINE_METHOD_ORDER):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_BASELINE_SET",
            "run plan must contain the exact eight baseline methods",
        )
    if raw.get("seeds") != list(FROZEN_SEEDS):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_CONFIG",
            "run plan seeds differ from the frozen five",
        )
    slices = _validate_estimand_slices(raw.get("estimand_slices"))
    for partition in PARTITIONS:
        ids = sorted(
            row_id
            for item in slices
            for part in item["partitions"]
            if part["partition"] == partition
            for row_id in part["row_ids"]
        )
        if canonical_sha256(ids) != partition_hashes[partition]:
            _hold(
                "HOLD_DEVELOPMENT_RUN_PLAN_BINDING",
                f"run plan partition row-set hash is stale for {partition}",
            )
    config = validate_executable_config(raw.get("executable_config"))
    if (
        config["train_development_outcomes_sha256"] != data_sha
        or config["development_projection_hashes"] != projections
        or config["source_authority_sha256"]
        != authority_sources["executable_config"]
        or config["seeds"] != raw["seeds"]
        or config["hgb_policy"]["qualification_sha256"]
        != refs["hgb_qualification"]["sha256"]
    ):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_BINDING",
            "embedded executable config disagrees with run-plan bindings",
        )
    expected_budget_summary = _planning_budget_summary(slices, config)
    if raw.get("planning_budget_summary") != expected_budget_summary:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_RESOURCE_BUDGET",
            "planning budget summary differs from the deterministic replay",
        )
    policy = _strict_keys(
        raw.get("scientific_policy"),
        {
            "primary_sign_gate",
            "information_set",
            "selector_required",
            "selector_schema",
            "seed_aggregation",
            "estimand_training_scope",
        },
        label="run plan scientific policy",
    )
    if _finite_number(
        policy.get("primary_sign_gate"),
        label="scientific policy sign gate",
        minimum=0.0,
        maximum=1.0,
    ) != SIGN_ACCURACY_GATE:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_SCHEMA",
            "scientific policy sign gate changed",
        )
    expected_policy = {
        "information_set": "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        "selector_required": True,
        "seed_aggregation": "FIXED_PREDICTION_ENSEMBLE",
        "estimand_training_scope": "INDEPENDENT_MODEL_PER_PRIMARY_ESTIMAND",
    }
    for field, value in expected_policy.items():
        _exact(policy.get(field), value, label=f"scientific policy.{field}")
    _exact(
        policy.get("selector_schema"),
        DEVELOPMENT_SELECTOR_SCHEMA,
        label="scientific policy.selector_schema",
    )
    execution = _strict_keys(
        raw.get("execution_state"),
        {
            "ordinary_only",
            "training_ready",
            "execution_authorized",
            "external_trust_root_verified",
            "development_capability_bound",
            "training_executed",
            "gpu_used",
            "network_used",
            "unreleased_partition_access",
            "unreleased_outcome_access",
            "future_stage_access",
            "publication_claim_authorized",
        },
        label="run plan execution state",
    )
    expected_execution = {
        "ordinary_only": True,
        "training_ready": False,
        "execution_authorized": False,
        "external_trust_root_verified": False,
        "development_capability_bound": False,
        "training_executed": False,
        "gpu_used": False,
        "network_used": False,
        "unreleased_partition_access": False,
        "unreleased_outcome_access": False,
        "future_stage_access": False,
        "publication_claim_authorized": False,
    }
    for field, value in expected_execution.items():
        _exact(execution.get(field), value, label=f"execution state.{field}")
    plan_sha = _digest(raw.get("plan_sha256"), label="run plan internal digest")
    core = copy.deepcopy(dict(raw))
    core.pop("plan_sha256")
    if plan_sha != canonical_sha256(core):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_BINDING",
            "run plan internal digest is stale",
        )
    return copy.deepcopy(dict(raw))


def canonical_development_run_plan_bytes(payload: Mapping) -> bytes:
    validated = validate_development_run_plan(payload)
    try:
        return canonical_json_bytes(validated)
    except ArtifactIOError as exc:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_JSON",
            f"run plan cannot be canonicalized: {exc}",
        )


def _rebuild_from_plan(run_root: Path, payload: Mapping) -> dict:
    validated = validate_development_run_plan(payload)
    rebuilt = build_development_run_plan(
        run_root,
        source_refs=validated["source_refs"],
        expected_development_release_sha256=validated[
            "external_release_sha256"
        ],
    )
    if canonical_development_run_plan_bytes(rebuilt) != (
        canonical_development_run_plan_bytes(validated)
    ):
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_REPLAY",
            "run plan differs from exact source replay",
        )
    return validated


def _checked_output_path(run_root: Path, path: Path) -> Path:
    root = _checked_run_root(run_root)
    candidate = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_PATH",
            "run-plan output must remain beneath the run root",
        )
    relative_posix = _normalized_relative_path(
        relative.as_posix(),
        label="run-plan output",
        required_prefix=RUN_PLAN_NAMESPACE,
    )
    parent = root
    try:
        for part in relative_posix.parts[:-1]:
            parent = parent / part
            metadata = os.lstat(parent)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                _hold(
                    "HOLD_DEVELOPMENT_RUN_PLAN_PATH",
                    "run-plan output parent must be a real directory",
                )
    except FileNotFoundError:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_PATH",
            "run-plan output parent must already exist",
        )
    if candidate.exists() or candidate.is_symlink():
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_WRITE",
            "run-plan output already exists",
        )
    return candidate


def _checked_plan_input_path(run_root: Path, path: Path) -> Path:
    root = _checked_run_root(run_root)
    candidate = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_PATH",
            "run-plan input must remain beneath the run root",
        )
    reference = _reference_shape(
        {
            "relative_path": relative.as_posix(),
            "sha256": "0" * 64,
        },
        label="run-plan input",
        required_prefix=RUN_PLAN_NAMESPACE,
    )
    return _existing_ref_path(root, reference, label="development run plan")


def write_development_run_plan_exclusive(
    run_root: Path,
    path: Path,
    payload: Mapping,
) -> str:
    """Replay every dependency, then atomically publish without overwrite."""

    root = _checked_run_root(run_root)
    validated = _rebuild_from_plan(root, payload)
    candidate = _checked_output_path(root, path)
    raw = canonical_development_run_plan_bytes(validated)

    def replay(data: bytes) -> None:
        parsed = _parse_json_object(data, label="published development run plan")
        replayed = validate_development_run_plan(parsed)
        if canonical_development_run_plan_bytes(replayed) != data:
            _hold(
                "HOLD_DEVELOPMENT_RUN_PLAN_JSON",
                "published development run plan is not canonical",
            )

    try:
        digest = atomic_publish_bytes(candidate, raw, replay_validator=replay)
    except DevelopmentRunPlanError:
        raise
    except ArtifactIOError as exc:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_WRITE",
            f"run-plan publication failed: {exc}",
        )
    loaded = load_development_run_plan_exact(root, candidate, digest)
    if canonical_development_run_plan_bytes(loaded) != raw:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_REPLAY",
            "published run plan changed during exact load",
        )
    return digest


def load_development_run_plan_exact(
    run_root: Path,
    path: Path,
    expected_sha256: str,
) -> dict:
    """Exact-load canonical bytes and rebuild the scaffold's declared DAG.

    This checks byte-level replay only.  It does not verify an external trust
    root or grant training capability.
    """

    root = _checked_run_root(run_root)
    candidate = _checked_plan_input_path(root, path)
    digest = _digest(expected_sha256, label="development run-plan artifact")
    try:
        raw = read_exact_regular_file(candidate, digest)
    except ArtifactIOError as exc:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_DEPENDENCY",
            f"development run-plan exact read failed: {exc}",
        )
    payload = _parse_json_object(raw, label="development run plan")
    validated = validate_development_run_plan(payload)
    if canonical_development_run_plan_bytes(validated) != raw:
        _hold(
            "HOLD_DEVELOPMENT_RUN_PLAN_JSON",
            "development run-plan bytes are not canonical",
        )
    return _rebuild_from_plan(root, validated)


__all__ = [
    "CAPABILITY_SCOPE",
    "DEVELOPMENT_RELEASE_ROLE",
    "DEVELOPMENT_RELEASE_SCHEMA",
    "DEVELOPMENT_RELEASE_STATUS",
    "DevelopmentRunPlanError",
    "EXECUTABLE_CONFIG_ROLE",
    "EXECUTABLE_CONFIG_SCHEMA",
    "EXECUTABLE_CONFIG_STATUS",
    "FROZEN_SEEDS",
    "FORMAL_SCALAR_WEIGHT_POLICY",
    "RIDGE_TRAINING_WEIGHT_POLICY",
    "RIDGE_CV_SCORING_POLICY",
    "PRIOR_TRAINING_POLICY",
    "NEAREST_NEIGHBOR_SCALING_POLICY",
    "NEAREST_NEIGHBOR_TIE_AGGREGATION_POLICY",
    "HGB_TRAINING_WEIGHT_POLICY",
    "HGB_WEIGHT_NORMALIZATION",
    "HGB_QUALIFICATION_ROLE",
    "HGB_QUALIFICATION_SCHEMA",
    "HGB_QUALIFICATION_STATUS",
    "ORDINARY_DEVELOPMENT",
    "ORDINARY_TRAIN",
    "ROW_INVENTORY_ROLE",
    "ROW_INVENTORY_SCHEMA",
    "ROW_INVENTORY_STATUS",
    "RUN_PLAN_ROLE",
    "RUN_PLAN_SCHEMA",
    "RUN_PLAN_STATUS",
    "build_development_run_plan",
    "canonical_development_run_plan_bytes",
    "load_development_run_plan_exact",
    "validate_development_release",
    "validate_development_run_plan",
    "validate_executable_config",
    "validate_hgb_qualification",
    "validate_row_inventory",
    "write_development_run_plan_exclusive",
]
