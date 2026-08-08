"""Independent ordinary train/development exact-eight orchestration.

This module is deliberately separate from the historical ordinary runner.  It
accepts only an exact V4 development plan, canonical target-bearing
``ORDINARY_TRAIN`` predictor rows, target-free ``ORDINARY_DEVELOPMENT`` rows,
and one information-set-independent shared train component assignment.  Every
external reference is rejected for protected names before any file content is
read.

The current implementation can fit and exact-replay the six deterministic
scalar baselines and the historical FullPair fixed-five-seed primary model.
The synthetic-only HGB numeric state is still not real-run-plan-qualified, so
HGB alone receives an immutable per-method HOLD artifact and the suite result
remains PARTIAL.  This schema has no code path that fabricates exact-eight
completion or converts development evidence into execution/publication
authority.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import stat
import unicodedata
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .artifact_graph import ArtifactGraphError, prediction_jsonl_bytes
from .artifact_io import (
    ArtifactIOError,
    atomic_publish_bytes,
    atomic_write_json_exclusive,
    canonical_json_bytes,
    read_exact_regular_file,
)
from .development_run_plan import (
    FROZEN_BASELINE_METHOD_ORDER,
    ORDINARY_DEVELOPMENT,
    ORDINARY_TRAIN,
    RUN_PLAN_SCHEMA,
    SIGN_ACCURACY_GATE,
    load_development_run_plan_exact,
    validate_row_inventory,
)
from .information_sets import information_set_fingerprint
from .models import ModelConfig
from .neural_development_state import (
    TRAINING_VARIANT_ID,
    NeuralDevelopmentFitConfig,
    NeuralDevelopmentStateError,
    build_neural_source_component_assignment,
    ensemble_neural_development_states,
    fit_neural_development_five_seed_ensemble,
    load_neural_development_state_exact,
    load_neural_source_component_assignment_exact,
    recompute_neural_development_predictions,
    save_neural_development_state_exclusive,
    validate_neural_development_ensemble,
    write_neural_source_component_assignment_exclusive,
)
from .neural_method_registry import (
    FULL_PAIR_METHOD_ID as REGISTERED_FULL_PAIR_METHOD_ID,
    registry_sha256,
)
from .predictor_provenance import (
    FROZEN_NEURAL_SEEDS,
    NON_DATA_AUTHORITY_HASH_KEYS,
    parse_predictor_input_jsonl,
)
from .protocol import canonical_sha256
from .scalar_fitted_state import (
    METHOD_INFORMATION_SETS,
    SUPPORTED_SCALAR_METHOD_IDS,
    ScalarFittedStateError,
    fit_scalar_fitted_state,
    load_scalar_fitted_state_exact,
    load_train_source_component_assignment_exact,
    recompute_scalar_predictions,
    write_scalar_fitted_state_exclusive,
)
from .typed_safe_authorities import (
    PAIRED_BASELINE_POLICY,
    load_typed_safe_authority_exact,
)


REQUEST_SCHEMA = "r3x_development_exact_eight_request_v2"
METHOD_EVIDENCE_SCHEMA = "r3x_development_exact_eight_method_evidence_v2"
RESULT_SCHEMA = "r3x_development_exact_eight_result_v2"
DEVELOPMENT_TARGETS_SCHEMA = "r3x_development_targets_v1"

METHOD_EVIDENCE_ROLE = "ORDINARY_DEVELOPMENT_EXACT_EIGHT_METHOD_EVIDENCE"
RESULT_ROLE = "ORDINARY_TRAIN_DEVELOPMENT_EXACT_EIGHT_ORCHESTRATION"

SCALAR_COMPLETE_STATUS = "COMPLETE_EXACT_SCALAR_STATE_AND_PREDICTIONS"
NEURAL_COMPLETE_STATUS = "COMPLETE_EXACT_NEURAL_FIVE_SEED_STATE_AND_ENSEMBLE"
METHOD_HOLD_STATUS = "HOLD_REAL_QUALIFIED_METHOD_STATE_ABSENT"
PARTIAL_STATUS = (
    "PARTIAL_SEVEN_METHODS_COMPLETE_HGB_REAL_STATE_HOLD"
)
COMPLETE_STATUS = "COMPLETE_ALL_EIGHT_REAL_QUALIFIED_AND_EXACT_REPLAYED"

HGB_METHOD_ID = "HISTOGRAM_GRADIENT_BOOSTING"
FULL_PAIR_METHOD_ID = "FULL_PAIR_UNRESTRICTED_SIAMESE"
PAIRED_INFORMATION_SET = "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR"

HGB_HOLD_CODE = "HOLD_HGB_REAL_RUN_PLAN_QUALIFICATION_ABSENT"
HGB_HOLD_REASON = (
    "current HGB numeric fitted-state API is synthetic-fixture-only, declares "
    "TRAINING_READY=false, and is not connected to the real run-plan trust root"
)

RESOURCE_ACCOUNTING_STATUS = (
    "STATIC_FIT_UNITS_AND_ZERO_NEURAL_FORWARDS_BOUND_"
    "RUNTIME_WALL_AND_PEAK_MEMORY_NOT_INDEPENDENTLY_QUALIFIED"
)
NEURAL_RESOURCE_ACCOUNTING_STATUS = (
    "EXACT_FIVE_SEED_PREFLIGHT_AND_FORWARD_EQUIVALENTS_BOUND_"
    "RUNTIME_WALL_NOT_INDEPENDENTLY_QUALIFIED"
)
AUTHORITY_BINDING_STATUS = (
    "CALLER_PREBOUND_ALL_TEN_SEVEN_CROSS_BOUND_TO_EXACT_RUN_PLAN"
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
_REF_KEYS = {"relative_path", "sha256"}
_REQUEST_KEYS = {
    "schema_version",
    "run_id",
    "run_plan_ref",
    "train_predictor_input_ref",
    "development_predictor_input_ref",
    "development_targets_ref",
    "shared_component_assignment_ref",
    "expected_non_data_authority_hashes",
    "expected_neural_registry_sha256",
    "output_relative_dir",
    "ordinary_only",
    "primary_sign_gate",
    "sign_gate_change_authorized",
}
_DEVELOPMENT_TARGETS_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "train_development_outcomes_sha256",
    "development_projection_hashes",
    "partition",
    "partition_row_set_sha256",
    "targets",
    "targets_sha256",
    "ordinary_only",
    "protected_partition_access",
    "future_stage_access",
    "execution_authorized",
    "publication_evidence",
}
_DEVELOPMENT_TARGET_ROW_KEYS = {"row_id", "estimand_id", "target"}
_DEVELOPMENT_PROJECTION_NAMES = {
    "inclusion_manifest_development_projection",
    "context_source_crosswalk_development_projection",
    "source_component_split_manifest_development_projection",
}
_PREDICTION_KEYS = {
    "schema_version",
    "row_id",
    "predicted_delta_mean",
    "predicted_delta_logvar",
}
class DevelopmentExactEightError(RuntimeError):
    """Structured fail-closed error raised before any unsupported claim."""

    def __init__(
        self, code: str, message: str, *, details: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.details = None if details is None else dict(details)


def _hold(code: str, message: str, **details: Any) -> None:
    raise DevelopmentExactEightError(
        code, message, details=details or None
    )


def _strict_keys(value: object, expected: set[str], *, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_SCHEMA", f"{label} must be an object")
    actual = set(value)
    if actual != expected or any(not isinstance(key, str) for key in value):
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_SCHEMA",
            f"{label} keys differ from the exact schema",
            missing=sorted(expected - actual),
            extra=sorted(actual - expected),
        )
    return value


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_SCHEMA",
            f"{label} must be lowercase SHA-256",
        )
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_SCHEMA",
            f"{label} must be a safe non-empty identifier",
        )
    return value


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_SCHEMA", f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_SCHEMA", f"{label} must be finite")
    return parsed


def _compact_security_text(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _contains_protected_token(value: object) -> bool:
    compact = _compact_security_text(value)
    always_forbidden = (
        "internaltest",
        "testoutcome",
        "outcome",
        "sealed",
        "gse246381",
    )
    if any(token in compact for token in always_forbidden):
        return True
    # The frozen baseline name FULL_PAIR_UNRESTRICTED_SIAMESE is legitimate;
    # do not confuse its English adjective with a protected namespace.
    namespace_text = compact.replace("unrestricted", "")
    return "restricted" in namespace_text or "protected" in namespace_text


def _scan_preaccess(value: object, *, path: str = "$") -> None:
    """Reject protected request fields and values before any content read."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _contains_protected_token(key):
                _hold(
                    "HOLD_DEVELOPMENT_EXACT_EIGHT_PROTECTED_REQUEST",
                    "protected field name is forbidden before access",
                    path=path,
                )
            _scan_preaccess(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _scan_preaccess(nested, path=f"{path}[{index}]")
    elif isinstance(value, (str, Path)):
        if _contains_protected_token(value):
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_PROTECTED_REQUEST",
                "protected string or path is forbidden before access",
                path=path,
            )


def _normalize_ref(
    value: object, *, label: str, required_prefix: tuple[str, ...]
) -> dict[str, str]:
    raw = _strict_keys(value, _REF_KEYS, label=label)
    relative = raw.get("relative_path")
    if not isinstance(relative, str) or not relative or "\\" in relative:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_PATH",
            f"{label}.relative_path is invalid",
        )
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != relative
        or tuple(pure.parts[: len(required_prefix)]) != required_prefix
    ):
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_PATH",
            f"{label} is outside its frozen namespace",
        )
    _scan_preaccess(relative, path=f"{label}.relative_path")
    return {
        "relative_path": relative,
        "sha256": _digest(raw.get("sha256"), label=f"{label}.sha256"),
    }


def _authorities(value: object) -> dict[str, str]:
    raw = _strict_keys(
        value,
        set(NON_DATA_AUTHORITY_HASH_KEYS),
        label="expected_non_data_authority_hashes",
    )
    return {
        key: _digest(raw[key], label=f"authority.{key}")
        for key in NON_DATA_AUTHORITY_HASH_KEYS
    }


def validate_development_exact_eight_request(payload: Mapping) -> dict:
    _scan_preaccess(payload)
    raw = _strict_keys(payload, _REQUEST_KEYS, label="exact-eight request")
    if raw.get("schema_version") != REQUEST_SCHEMA:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_SCHEMA",
            "unsupported exact-eight request schema",
        )
    if raw.get("ordinary_only") is not True:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_BOUNDARY",
            "exact-eight request must remain ordinary-only",
        )
    if raw.get("sign_gate_change_authorized") is not False:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_SIGN_GATE",
            "sign gate changes are not authorized",
        )
    if _finite_float(
        raw.get("primary_sign_gate"), label="primary_sign_gate"
    ) != SIGN_ACCURACY_GATE:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_SIGN_GATE",
            "primary sign gate must remain exactly 0.60",
        )
    run_id = _identifier(raw.get("run_id"), label="run_id")
    output = raw.get("output_relative_dir")
    if not isinstance(output, str) or "\\" in output:
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_PATH", "output path is invalid")
    output_pure = PurePosixPath(output)
    if (
        output_pure.is_absolute()
        or any(part in {"", ".", ".."} for part in output_pure.parts)
        or output_pure.as_posix() != output
        or tuple(output_pure.parts[:2]) != ("development", "exact_eight")
        or output_pure.name != run_id
    ):
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_PATH",
            "output path must be development/exact_eight/<run_id>",
        )
    refs = {
        "run_plan_ref": _normalize_ref(
            raw.get("run_plan_ref"),
            label="run_plan_ref",
            required_prefix=("development", "run_plans"),
        ),
        "train_predictor_input_ref": _normalize_ref(
            raw.get("train_predictor_input_ref"),
            label="train_predictor_input_ref",
            required_prefix=("development", "predictor_inputs"),
        ),
        "development_predictor_input_ref": _normalize_ref(
            raw.get("development_predictor_input_ref"),
            label="development_predictor_input_ref",
            required_prefix=("development", "predictor_inputs"),
        ),
        "development_targets_ref": _normalize_ref(
            raw.get("development_targets_ref"),
            label="development_targets_ref",
            required_prefix=("development", "targets"),
        ),
        "shared_component_assignment_ref": _normalize_ref(
            raw.get("shared_component_assignment_ref"),
            label="shared_component_assignment_ref",
            required_prefix=("development", "assignments"),
        ),
    }
    if len({item["sha256"] for item in refs.values()}) != len(refs):
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_BINDING",
            "request artifact roles cannot alias identical bytes",
        )
    expected_registry = _digest(
        raw.get("expected_neural_registry_sha256"),
        label="expected_neural_registry_sha256",
    )
    if expected_registry != registry_sha256():
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REGISTRY",
            "request neural registry SHA differs from the current primary registry",
        )
    return {
        "schema_version": REQUEST_SCHEMA,
        "run_id": run_id,
        **refs,
        "expected_non_data_authority_hashes": _authorities(
            raw.get("expected_non_data_authority_hashes")
        ),
        "expected_neural_registry_sha256": expected_registry,
        "output_relative_dir": output,
        "ordinary_only": True,
        "primary_sign_gate": SIGN_ACCURACY_GATE,
        "sign_gate_change_authorized": False,
    }


def _checked_root(run_root: Path) -> Path:
    _scan_preaccess(os.fspath(run_root), path="run_root")
    root = Path(os.path.abspath(os.fspath(run_root)))
    try:
        metadata = os.lstat(root)
    except OSError as exc:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_PATH",
            f"run root is unavailable: {exc}",
        )
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_PATH",
            "run root must be a real directory",
        )
    return root


def _path_from_relative(
    root: Path, relative: str, *, label: str, require_exists: bool
) -> Path:
    pure = PurePosixPath(relative)
    current = root
    for index, part in enumerate(pure.parts):
        current = current / part
        if not require_exists and index == len(pure.parts) - 1:
            break
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_PATH",
                f"{label} path is unavailable: {exc}",
            )
        if stat.S_ISLNK(metadata.st_mode):
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_PATH",
                f"{label} path cannot traverse a symlink",
            )
        if index < len(pure.parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_PATH",
                f"{label} parent is not a real directory",
            )
    return current


def _ref_path(root: Path, reference: Mapping, *, label: str) -> Path:
    return _path_from_relative(
        root, str(reference["relative_path"]), label=label, require_exists=True
    )


def _make_ref(root: Path, path: Path, digest: str) -> dict[str, str]:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_PATH",
            "published artifact escaped the run root",
        )
    return {"relative_path": relative, "sha256": _digest(digest, label="artifact")}


class _DuplicateJSONKey(ValueError):
    pass


def _unique_object(pairs: Sequence[tuple[str, object]]) -> dict:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise _DuplicateJSONKey(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _parse_json_object(raw: bytes, *, label: str) -> dict:
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_object
        )
    except (UnicodeError, json.JSONDecodeError, _DuplicateJSONKey) as exc:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_JSON",
            f"{label} is not strict JSON: {exc}",
        )
    if not isinstance(value, dict):
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_JSON", f"{label} must be an object")
    return value


def _load_row_inventory_exact(root: Path, plan: Mapping) -> dict:
    reference = plan["source_refs"]["row_inventory"]
    path = _ref_path(root, reference, label="row inventory")
    try:
        raw = read_exact_regular_file(path, reference["sha256"])
    except ArtifactIOError as exc:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_DEPENDENCY",
            f"row inventory exact read failed: {exc}",
        )
    payload = validate_row_inventory(_parse_json_object(raw, label="row inventory"))
    try:
        canonical = canonical_json_bytes(payload)
    except ArtifactIOError as exc:
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_JSON", str(exc))
    if canonical != raw:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_JSON",
            "row inventory bytes are not canonical",
        )
    return payload


def _load_predictor_rows_exact(
    root: Path, reference: Mapping, *, partition: str, label: str
) -> list[dict]:
    path = _ref_path(root, reference, label=label)
    try:
        raw = read_exact_regular_file(path, reference["sha256"])
        rows = parse_predictor_input_jsonl(raw)
    except (ArtifactIOError, RuntimeError) as exc:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_PREDICTOR_INPUT",
            f"{label} exact replay failed: {exc}",
        )
    if any(row["partition"] != partition for row in rows):
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_PARTITION",
            f"{label} contains a non-{partition} row",
        )
    return rows


def validate_development_targets(payload: Mapping) -> dict:
    """Validate target-bearing ordinary-development selection authority bytes."""

    raw = _strict_keys(
        payload, _DEVELOPMENT_TARGETS_KEYS, label="development targets"
    )
    expected_identity = {
        "schema_version": DEVELOPMENT_TARGETS_SCHEMA,
        "artifact_role": "ORDINARY_DEVELOPMENT_EXACT_TARGETS",
        "status": "COMPLETE_EXACT_DEVELOPMENT_TARGET_BINDING",
        "partition": ORDINARY_DEVELOPMENT,
        "ordinary_only": True,
        "protected_partition_access": False,
        "future_stage_access": False,
        "execution_authorized": False,
        "publication_evidence": False,
    }
    for field, expected in expected_identity.items():
        if raw.get(field) != expected:
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_TARGETS",
                f"development target {field} drift",
            )
    _digest(
        raw.get("train_development_outcomes_sha256"),
        label="development targets outcomes SHA",
    )
    _digest(
        raw.get("partition_row_set_sha256"),
        label="development targets row-set SHA",
    )
    projections = _strict_keys(
        raw.get("development_projection_hashes"),
        _DEVELOPMENT_PROJECTION_NAMES,
        label="development target projection hashes",
    )
    for name, value in projections.items():
        _digest(value, label=f"development target projection {name}")
    targets_raw = raw.get("targets")
    if not isinstance(targets_raw, list) or not targets_raw:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_TARGETS",
            "development targets must be a non-empty list",
        )
    targets = []
    for index, item in enumerate(targets_raw):
        target = _strict_keys(
            item,
            _DEVELOPMENT_TARGET_ROW_KEYS,
            label=f"development target row {index}",
        )
        targets.append({
            "row_id": _digest(target.get("row_id"), label="target row_id"),
            "estimand_id": _identifier(
                target.get("estimand_id"), label="target estimand_id"
            ),
            "target": _finite_float(target.get("target"), label="target value"),
        })
    if targets != sorted(targets, key=lambda item: item["row_id"]):
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_TARGETS",
            "development target rows must be in canonical row_id order",
        )
    row_ids = [item["row_id"] for item in targets]
    if len(row_ids) != len(set(row_ids)):
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_TARGETS",
            "development target row IDs repeat",
        )
    if raw.get("partition_row_set_sha256") != canonical_sha256(row_ids):
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_TARGETS",
            "development target row-set commitment is stale",
        )
    if raw.get("targets_sha256") != canonical_sha256(targets):
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_TARGETS",
            "development target value commitment is stale",
        )
    return copy.deepcopy(dict(raw))


def _load_development_targets_exact(
    root: Path,
    reference: Mapping,
    *,
    plan: Mapping,
    development_rows: Sequence[Mapping],
) -> tuple[dict, dict[str, float]]:
    path = _ref_path(root, reference, label="ordinary development targets")
    try:
        raw = read_exact_regular_file(path, reference["sha256"])
    except ArtifactIOError as exc:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_TARGETS",
            f"development targets exact read failed: {exc}",
        )
    payload = validate_development_targets(
        _parse_json_object(raw, label="development targets")
    )
    try:
        canonical = canonical_json_bytes(payload)
    except ArtifactIOError as exc:
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_TARGETS", str(exc))
    if canonical != raw:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_TARGETS",
            "development target bytes are not canonical",
        )
    if (
        payload["train_development_outcomes_sha256"]
        != plan["train_development_outcomes_sha256"]
        or payload["development_projection_hashes"]
        != plan["development_projection_hashes"]
        or payload["partition_row_set_sha256"]
        != plan["partition_row_set_sha256"][ORDINARY_DEVELOPMENT]
    ):
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_TARGETS",
            "development targets differ from the exact run-plan authority",
        )
    expected = {
        row["row_id"]: row["estimand_id"] for row in development_rows
    }
    actual = {item["row_id"]: item["estimand_id"] for item in payload["targets"]}
    if actual != expected:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_TARGETS",
            "development targets do not exactly cover bound predictor rows/estimands",
        )
    return payload, {
        item["row_id"]: float(item["target"])
        for item in payload["targets"]
    }


def _load_plan_context(root: Path, request: Mapping) -> tuple[dict, dict, dict]:
    plan_ref = request["run_plan_ref"]
    plan_path = _ref_path(root, plan_ref, label="development run plan")
    try:
        plan = load_development_run_plan_exact(
            root, plan_path, plan_ref["sha256"]
        )
    except RuntimeError as exc:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_RUN_PLAN",
            f"development run plan exact replay failed: {exc}",
        )
    if plan.get("schema_version") != RUN_PLAN_SCHEMA:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_RUN_PLAN",
            "only the exact-order V4 development run plan is supported",
        )
    if plan["scientific_policy"]["primary_sign_gate"] != SIGN_ACCURACY_GATE:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_SIGN_GATE",
            "run plan sign gate differs from 0.60",
        )
    paired_ref = plan["source_refs"]["paired_baseline_policy"]
    paired_path = _ref_path(root, paired_ref, label="paired baseline policy")
    try:
        paired = load_typed_safe_authority_exact(
            root,
            paired_path,
            paired_ref["sha256"],
            expected_authority_name=PAIRED_BASELINE_POLICY,
        )
    except RuntimeError as exc:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_DEPENDENCY",
            f"paired policy exact replay failed: {exc}",
        )
    order = list(FROZEN_BASELINE_METHOD_ORDER)
    if (
        paired["eligible_baseline_method_ids"] != order
        or paired["required_baseline_method_ids"] != order
        or plan["eligible_baseline_method_order"] != order
    ):
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_METHOD_ORDER",
            "paired policy and run plan do not bind the same contract order",
        )
    inventory = _load_row_inventory_exact(root, plan)
    return plan, paired, inventory


def _cross_bind_authorities(
    plan: Mapping, expected: Mapping[str, str]
) -> None:
    source = plan["authority_source_sha256"]
    config = plan["executable_config"]
    crosswalk = {
        "inclusion_manifest_sha256": source["inclusion_manifest"],
        "context_source_crosswalk_sha256": source[
            "context_source_crosswalk"
        ],
        "endpoint_estimand_registry_sha256": source[
            "endpoint_estimand_registry"
        ],
        "source_component_split_manifest_sha256": source[
            "source_component_split_manifest"
        ],
        "model_optimizer_seed_config_sha256": source["executable_config"],
        "analysis_code_tree_manifest_sha256": config["code_tree_sha256"],
        "paired_strongest_baseline_comparison_policy_sha256": source[
            "paired_baseline_policy"
        ],
    }
    mismatched = sorted(
        key for key, value in crosswalk.items() if expected[key] != value
    )
    if mismatched:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_AUTHORITY_BINDING",
            "caller authority hashes disagree with the exact run plan",
            mismatched=mismatched,
        )


def _partition_ids(plan: Mapping, partition: str) -> list[str]:
    return sorted(
        row_id
        for item in plan["estimand_slices"]
        for part in item["partitions"]
        if part["partition"] == partition
        for row_id in part["row_ids"]
    )


def _cross_bind_rows(
    *,
    plan: Mapping,
    inventory: Mapping,
    train_rows: Sequence[Mapping],
    development_rows: Sequence[Mapping],
) -> None:
    by_partition = {
        ORDINARY_TRAIN: list(train_rows),
        ORDINARY_DEVELOPMENT: list(development_rows),
    }
    inventory_by_id = {row["row_id"]: row for row in inventory["rows"]}
    for partition, rows in by_partition.items():
        ids = [row["row_id"] for row in rows]
        if ids != _partition_ids(plan, partition):
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_ROW_SET",
                f"{partition} predictor rows differ from the exact plan row set",
            )
        if canonical_sha256(ids) != plan["partition_row_set_sha256"][partition]:
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_ROW_SET",
                f"{partition} predictor row-set commitment is stale",
            )
        for row in rows:
            inventory_row = inventory_by_id.get(row["row_id"])
            if inventory_row is None:
                _hold(
                    "HOLD_DEVELOPMENT_EXACT_EIGHT_ROW_SET",
                    "predictor row is absent from the exact row inventory",
                    row_id=row["row_id"],
                )
            if (
                inventory_row["partition"] != row["partition"]
                or inventory_row["estimand_id"] != row["estimand_id"]
                or inventory_row["endpoint"] != row["predictor_input"]["endpoint"]
            ):
                _hold(
                    "HOLD_DEVELOPMENT_EXACT_EIGHT_ROW_BINDING",
                    "predictor row partition/estimand/endpoint disagrees with inventory",
                    row_id=row["row_id"],
                )


def _load_shared_assignment(
    *,
    root: Path,
    request: Mapping,
    plan: Mapping,
    train_rows: Sequence[Mapping],
) -> dict:
    reference = request["shared_component_assignment_ref"]
    path = _ref_path(root, reference, label="shared component assignment")
    source_sha = plan["authority_source_sha256"][
        "source_component_split_manifest"
    ]
    projection_sha = plan["development_projection_hashes"][
        "source_component_split_manifest_development_projection"
    ]
    try:
        return load_train_source_component_assignment_exact(
            path,
            reference["sha256"],
            expected_train_rows=train_rows,
            expected_train_development_outcomes_sha256=plan[
                "train_development_outcomes_sha256"
            ],
            expected_non_data_authority_hashes=request[
                "expected_non_data_authority_hashes"
            ],
            expected_source_component_split_manifest_source_sha256=source_sha,
            expected_source_component_split_manifest_projection_sha256=(
                projection_sha
            ),
        )
    except ScalarFittedStateError as exc:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_ASSIGNMENT",
            f"shared component assignment exact replay failed: {exc}",
        )


def _cross_bind_assignment(
    *,
    assignment: Mapping,
    inventory: Mapping,
    train_rows: Sequence[Mapping],
) -> dict[str, str]:
    predictor_by_id = {row["row_id"]: row for row in train_rows}
    inventory_by_id = {row["row_id"]: row for row in inventory["rows"]}
    components: dict[str, str] = {}
    for item in assignment["assignments"]:
        row_id = item["row_id"]
        predictor = predictor_by_id[row_id]
        inventory_row = inventory_by_id[row_id]
        if (
            item["partition"] != ORDINARY_TRAIN
            or item["cell_id"] != predictor["cell_id"]
            or item["source_component_id"]
            != inventory_row["source_component_id"]
        ):
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_ASSIGNMENT",
                "shared assignment disagrees with predictor cell or inventory component",
                row_id=row_id,
            )
        components[row_id] = item["source_component_id"]
    return components


def _validate_budget(
    *,
    plan: Mapping,
    train_rows: Sequence[Mapping],
    development_rows: Sequence[Mapping],
    components: Mapping[str, str],
) -> dict:
    budget = copy.deepcopy(plan["executable_config"]["resource_budget"])
    all_rows = len(train_rows) + len(development_rows)
    if all_rows > budget["max_rows"]:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_RESOURCE_BUDGET",
            "predictor rows exceed the frozen max_rows budget",
        )
    if len(plan["estimand_slices"]) > budget["max_estimands"]:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_RESOURCE_BUDGET",
            "estimands exceed the frozen max_estimands budget",
        )
    ridge = plan["executable_config"]["ridge_policy"]
    scalar_fit_units = 0
    per_estimand = []
    for item in plan["estimand_slices"]:
        train_ids = next(
            part["row_ids"]
            for part in item["partitions"]
            if part["partition"] == ORDINARY_TRAIN
        )
        n_components = len({components[row_id] for row_id in train_ids})
        if n_components < ridge["minimum_components_for_cv"]:
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_COMPONENT_SUPPORT",
                "actual global components cannot support frozen ridge CV",
                estimand_id=item["estimand_id"],
                n_components=n_components,
            )
        folds = min(ridge["max_component_folds"], n_components)
        units = 3 * (folds * len(ridge["alpha_grid"]) + 1) + 3
        scalar_fit_units += units
        per_estimand.append({
            "estimand_id": item["estimand_id"],
            "n_global_train_components": n_components,
            "ridge_component_folds": folds,
            "scalar_fit_units": units,
            "neural_fit_units": len(FROZEN_NEURAL_SEEDS),
        })
    neural_fit_units = len(plan["estimand_slices"]) * len(FROZEN_NEURAL_SEEDS)
    total_fit_units = scalar_fit_units + neural_fit_units
    if total_fit_units > budget["max_total_fit_units"]:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_RESOURCE_BUDGET",
            "actual scalar plus neural fit units exceed the frozen total budget",
        )
    return {
        "resource_budget": budget,
        "resource_budget_sha256": canonical_sha256(budget),
        "accounting_status": RESOURCE_ACCOUNTING_STATUS,
        "per_estimand": per_estimand,
        "executed_scalar_fit_units": scalar_fit_units,
        "executed_neural_fit_units": neural_fit_units,
        "executed_total_fit_units": total_fit_units,
        "neural_forward_equivalents_per_scalar_fit": 0,
    }


def _prediction_bytes(rows: Sequence[Mapping]) -> bytes:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_PREDICTION",
            "prediction rows must be a non-empty sequence",
        )
    normalized = []
    for index, row in enumerate(rows):
        raw = _strict_keys(row, _PREDICTION_KEYS, label=f"prediction {index}")
        if raw.get("schema_version") != "r3x_prediction_row_v1":
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_PREDICTION",
                "unsupported prediction row schema",
            )
        if raw.get("predicted_delta_logvar") is not None:
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_PREDICTION",
                "scalar prediction log-variance must be null",
            )
        normalized.append({
            "schema_version": "r3x_prediction_row_v1",
            "row_id": _digest(raw.get("row_id"), label="prediction.row_id"),
            "predicted_delta_mean": _finite_float(
                raw.get("predicted_delta_mean"), label="predicted_delta_mean"
            ),
            "predicted_delta_logvar": None,
        })
    normalized.sort(key=lambda item: item["row_id"])
    if len({item["row_id"] for item in normalized}) != len(normalized):
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_PREDICTION",
            "prediction row IDs must be unique",
        )
    return b"".join(
        (
            json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for item in normalized
    )


def _parse_prediction_bytes(raw: bytes) -> list[dict]:
    if not raw or not raw.endswith(b"\n"):
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_PREDICTION",
            "prediction JSONL must be non-empty and end in LF",
        )
    parsed = []
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as exc:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_PREDICTION",
            f"prediction JSONL is not UTF-8: {exc}",
        )
    for index, line in enumerate(lines):
        if not line:
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_PREDICTION",
                "prediction JSONL cannot contain blank lines",
            )
        try:
            value = json.loads(line, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, _DuplicateJSONKey) as exc:
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_PREDICTION",
                f"prediction line {index + 1} is invalid: {exc}",
            )
        parsed.append(value)
    canonical = _prediction_bytes(parsed)
    if canonical != raw:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_PREDICTION",
            "prediction JSONL bytes are not canonical",
        )
    return sorted(parsed, key=lambda item: item["row_id"])


def _write_prediction_exclusive(path: Path, rows: Sequence[Mapping]) -> str:
    raw = _prediction_bytes(rows)
    try:
        return atomic_publish_bytes(
            path, raw, replay_validator=_parse_prediction_bytes
        )
    except ArtifactIOError as exc:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_WRITE",
            f"prediction publication failed: {exc}",
        )


def _neural_prediction_bytes(rows: Sequence[Mapping]) -> bytes:
    try:
        raw = prediction_jsonl_bytes(rows)
    except ArtifactGraphError as exc:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REPLAY",
            f"neural prediction rows are invalid: {exc}",
        )
    if any(row.get("predicted_delta_logvar") is None for row in rows):
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REPLAY",
            "neural predictions require finite log-variance",
        )
    return raw


def _parse_neural_prediction_bytes(raw: bytes) -> list[dict]:
    if not raw or not raw.endswith(b"\n"):
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REPLAY",
            "neural prediction JSONL must be non-empty and end in LF",
        )
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as exc:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REPLAY",
            f"neural prediction JSONL is not UTF-8: {exc}",
        )
    parsed = []
    for index, line in enumerate(lines):
        if not line:
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REPLAY",
                "neural prediction JSONL cannot contain blank lines",
            )
        try:
            value = json.loads(line, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, _DuplicateJSONKey) as exc:
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REPLAY",
                f"neural prediction line {index + 1} is invalid: {exc}",
            )
        if not isinstance(value, Mapping):
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REPLAY",
                "neural prediction row must be an object",
            )
        parsed.append(dict(value))
    if _neural_prediction_bytes(parsed) != raw:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REPLAY",
            "neural prediction JSONL bytes are not canonical",
        )
    return sorted(parsed, key=lambda item: item["row_id"])


def _write_neural_prediction_exclusive(
    path: Path, rows: Sequence[Mapping]
) -> str:
    raw = _neural_prediction_bytes(rows)
    try:
        return atomic_publish_bytes(
            path, raw, replay_validator=_parse_neural_prediction_bytes
        )
    except ArtifactIOError as exc:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_WRITE",
            f"neural prediction publication failed: {exc}",
        )


def _write_neural_ensemble_exclusive(path: Path, payload: Mapping) -> str:
    try:
        return atomic_write_json_exclusive(
            path,
            validate_neural_development_ensemble(payload),
            validator=validate_neural_development_ensemble,
        )
    except (ArtifactIOError, NeuralDevelopmentStateError) as exc:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REPLAY",
            f"neural ensemble publication failed: {exc}",
        )


def _load_neural_ensemble_exact(path: Path, expected_sha256: str) -> dict:
    try:
        raw = read_exact_regular_file(path, expected_sha256)
        payload = validate_neural_development_ensemble(
            _parse_json_object(raw, label="neural ensemble")
        )
        canonical = canonical_json_bytes(payload)
    except (ArtifactIOError, NeuralDevelopmentStateError) as exc:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REPLAY",
            f"neural ensemble exact replay failed: {exc}",
        )
    if canonical != raw:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REPLAY",
            "neural ensemble bytes are not canonical",
        )
    return payload


def _estimand_rows(
    rows: Sequence[Mapping], estimand_id: str
) -> list[dict]:
    selected = [row for row in rows if row["estimand_id"] == estimand_id]
    if not selected:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_ESTIMAND",
            "plan estimand has no predictor rows",
            estimand_id=estimand_id,
        )
    return selected


def _scalar_kwargs(
    *,
    method_id: str,
    root: Path,
    request: Mapping,
    plan: Mapping,
    all_train_rows: Sequence[Mapping],
) -> dict:
    values = {
        "information_set_fingerprint": information_set_fingerprint(
            METHOD_INFORMATION_SETS[method_id]
        ),
        "train_development_outcomes_sha256": plan[
            "train_development_outcomes_sha256"
        ],
        "non_data_authority_hashes": request[
            "expected_non_data_authority_hashes"
        ],
    }
    if method_id != "ZERO_DELTA":
        assignment_ref = request["shared_component_assignment_ref"]
        values.update({
            "component_assignment_authority_rows": all_train_rows,
            "component_assignment_path": _ref_path(
                root, assignment_ref, label="shared component assignment"
            ),
            "component_assignment_sha256": assignment_ref["sha256"],
            "source_component_split_manifest_source_sha256": plan[
                "authority_source_sha256"
            ]["source_component_split_manifest"],
            "source_component_split_manifest_projection_sha256": plan[
                "development_projection_hashes"
            ]["source_component_split_manifest_development_projection"],
        })
    return values


def _state_writer_kwargs(
    *,
    method_id: str,
    train_rows: Sequence[Mapping],
    root: Path,
    request: Mapping,
    plan: Mapping,
    all_train_rows: Sequence[Mapping],
) -> dict:
    common = _scalar_kwargs(
        method_id=method_id,
        root=root,
        request=request,
        plan=plan,
        all_train_rows=all_train_rows,
    )
    values = {
        "expected_train_rows": train_rows,
        "expected_method_id": method_id,
        "expected_information_set_fingerprint": common.pop(
            "information_set_fingerprint"
        ),
        "expected_train_development_outcomes_sha256": common.pop(
            "train_development_outcomes_sha256"
        ),
        "expected_non_data_authority_hashes": common.pop(
            "non_data_authority_hashes"
        ),
    }
    if method_id != "ZERO_DELTA":
        values.update({
            "component_assignment_authority_rows": common.pop(
                "component_assignment_authority_rows"
            ),
            "component_assignment_path": common.pop(
                "component_assignment_path"
            ),
            "expected_component_assignment_sha256": common.pop(
                "component_assignment_sha256"
            ),
            "expected_source_component_split_manifest_source_sha256": (
                common.pop("source_component_split_manifest_source_sha256")
            ),
            "expected_source_component_split_manifest_projection_sha256": (
                common.pop("source_component_split_manifest_projection_sha256")
            ),
        })
    if common:
        raise AssertionError(f"unconsumed scalar writer bindings: {sorted(common)}")
    return values


def _neural_model_config(plan: Mapping) -> ModelConfig:
    model = plan["executable_config"]["model"]
    try:
        return ModelConfig(**{
            key: model[key]
            for key in (
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
            )
        })
    except (KeyError, TypeError, ValueError) as exc:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_CONFIG",
            f"run-plan model cannot compile the primary FullPair config: {exc}",
        )


def _neural_fit_config(plan: Mapping) -> NeuralDevelopmentFitConfig:
    config = plan["executable_config"]
    training = config["training"]
    if training["device_class"] != "CPU":
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_CONFIG",
            "development exact replay is CPU-only and cannot reinterpret a CUDA plan",
        )
    optimizer = config["optimizer"]
    loss = config["loss"]
    budget = config["resource_budget"]
    try:
        return NeuralDevelopmentFitConfig(
            epochs=training["epochs"],
            batch_size=training["batch_size"],
            learning_rate=optimizer["learning_rate"],
            weight_decay=optimizer["weight_decay"],
            huber_weight=loss["huber_weight"],
            nll_weight=loss["nll_weight"],
            sign_weight=loss["sign_weight"],
            sign_temperature=loss["sign_temperature"],
            gradient_clip_norm=training["gradient_clip_norm"],
            condition_unknown_dropout=training["condition_unknown_dropout"],
            max_rows_per_fit=budget["max_rows"],
            max_sequence_tokens_per_batch=budget[
                "max_sequence_tokens_per_batch"
            ],
            max_forward_equivalents_per_fit=budget[
                "max_forward_equivalents_per_fit"
            ],
            max_total_forward_equivalents_five_seeds=(
                len(FROZEN_NEURAL_SEEDS)
                * budget["max_forward_equivalents_per_fit"]
            ),
            max_peak_parameter_bytes_per_fit=budget[
                "max_peak_memory_bytes_per_fit"
            ],
            max_wall_time_seconds_per_fit=budget[
                "max_wall_time_seconds_per_fit"
            ],
            device="cpu",
        )
    except (KeyError, TypeError, ValueError, NeuralDevelopmentStateError) as exc:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_CONFIG",
            f"run-plan fit/resource config cannot compile FullPair: {exc}",
        )


def _neural_run_hashes(request: Mapping, plan: Mapping) -> tuple[str, str]:
    run_start_sha256 = canonical_sha256({
        "domain": "R3X_DEVELOPMENT_EXACT_EIGHT_FULLPAIR_RUN_START_V1",
        "run_id": request["run_id"],
        "request_sha256": canonical_sha256(request),
        "run_plan_artifact_sha256": request["run_plan_ref"]["sha256"],
        "run_plan_internal_sha256": plan["plan_sha256"],
    })
    activation_sha256 = canonical_sha256({
        "domain": "R3X_DEVELOPMENT_EXACT_EIGHT_FULLPAIR_ACTIVATION_V1",
        "run_start_sha256": run_start_sha256,
        "development_targets_ref": request["development_targets_ref"],
        "development_row_set_sha256": plan["partition_row_set_sha256"][
            ORDINARY_DEVELOPMENT
        ],
        "train_development_outcomes_sha256": plan[
            "train_development_outcomes_sha256"
        ],
        "ordinary_only": True,
        "execution_authorized": False,
    })
    return run_start_sha256, activation_sha256


def _neural_assignment_kwargs(
    *,
    request: Mapping,
    plan: Mapping,
    train_rows: Sequence[Mapping],
    development_rows: Sequence[Mapping],
) -> dict:
    return {
        "expected_train_rows": train_rows,
        "expected_development_rows": development_rows,
        "expected_information_set_fingerprint": information_set_fingerprint(
            PAIRED_INFORMATION_SET
        ),
        "expected_train_development_outcomes_sha256": plan[
            "train_development_outcomes_sha256"
        ],
        "expected_non_data_authority_hashes": request[
            "expected_non_data_authority_hashes"
        ],
        "expected_source_component_split_manifest_source_sha256": plan[
            "authority_source_sha256"
        ]["source_component_split_manifest"],
        "expected_source_component_split_manifest_projection_sha256": plan[
            "development_projection_hashes"
        ]["source_component_split_manifest_development_projection"],
    }


def _derive_neural_assignment(
    *,
    request: Mapping,
    plan: Mapping,
    train_rows: Sequence[Mapping],
    development_rows: Sequence[Mapping],
    train_components: Mapping[str, str],
    inventory_components: Mapping[str, str],
) -> tuple[dict, str]:
    assignments = [
        {
            "row_id": row["row_id"],
            "partition": row["partition"],
            "source_component_id": (
                train_components[row["row_id"]]
                if row["partition"] == ORDINARY_TRAIN
                else inventory_components[row["row_id"]]
            ),
        }
        for row in (*train_rows, *development_rows)
    ]
    try:
        assignment = build_neural_source_component_assignment(
            train_rows,
            development_rows,
            assignments,
            information_set_fingerprint_sha256=information_set_fingerprint(
                PAIRED_INFORMATION_SET
            ),
            train_development_outcomes_sha256=plan[
                "train_development_outcomes_sha256"
            ],
            non_data_authority_hashes=request[
                "expected_non_data_authority_hashes"
            ],
            source_component_split_manifest_source_sha256=plan[
                "authority_source_sha256"
            ]["source_component_split_manifest"],
            source_component_split_manifest_projection_sha256=plan[
                "development_projection_hashes"
            ]["source_component_split_manifest_development_projection"],
        )
    except (KeyError, NeuralDevelopmentStateError) as exc:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_ASSIGNMENT",
            f"paired neural assignment derivation failed: {exc}",
        )
    derivation_sha256 = canonical_sha256({
        "domain": "R3X_FULLPAIR_ASSIGNMENT_FROM_SHARED_V2_AND_ROW_INVENTORY_V1",
        "shared_component_assignment_ref": request[
            "shared_component_assignment_ref"
        ],
        "row_inventory_ref": plan["source_refs"]["row_inventory"],
        "information_set_fingerprint": information_set_fingerprint(
            PAIRED_INFORMATION_SET
        ),
        "assignments": assignment["assignments"],
        "assignment_sha256": assignment["assignment_sha256"],
    })
    return assignment, derivation_sha256


def _method_common(
    *,
    method_id: str,
    order_index: int,
    request: Mapping,
    plan: Mapping,
    budget: Mapping,
) -> dict:
    information_set = METHOD_INFORMATION_SETS.get(
        method_id, PAIRED_INFORMATION_SET
    )
    return {
        "schema_version": METHOD_EVIDENCE_SCHEMA,
        "artifact_role": METHOD_EVIDENCE_ROLE,
        "method_id": method_id,
        "order_index": order_index,
        "information_set": information_set,
        "information_set_fingerprint": information_set_fingerprint(
            information_set
        ),
        "run_plan_artifact_sha256": request["run_plan_ref"]["sha256"],
        "run_plan_internal_sha256": plan["plan_sha256"],
        "train_predictor_input_ref": copy.deepcopy(
            request["train_predictor_input_ref"]
        ),
        "development_predictor_input_ref": copy.deepcopy(
            request["development_predictor_input_ref"]
        ),
        "development_targets_ref": copy.deepcopy(
            request["development_targets_ref"]
        ),
        "shared_component_assignment_ref": copy.deepcopy(
            request["shared_component_assignment_ref"]
        ),
        "train_development_outcomes_sha256": plan[
            "train_development_outcomes_sha256"
        ],
        "non_data_authority_hashes": copy.deepcopy(
            request["expected_non_data_authority_hashes"]
        ),
        "neural_registry_sha256": request[
            "expected_neural_registry_sha256"
        ],
        "authority_binding_status": AUTHORITY_BINDING_STATUS,
        "primary_sign_gate": SIGN_ACCURACY_GATE,
        "resource_budget_sha256": budget["resource_budget_sha256"],
        "ordinary_only": True,
        "protected_partition_access": False,
        "future_stage_access": False,
        "sign_gate_change_authorized": False,
        "publication_claim_authorized": False,
    }


def _validate_evidence_ref(value: object, *, label: str) -> dict[str, str]:
    return _normalize_ref(
        value,
        label=label,
        required_prefix=("development", "exact_eight"),
    )


def validate_development_exact_eight_method_evidence(payload: Mapping) -> dict:
    common_keys = {
        "schema_version",
        "artifact_role",
        "status",
        "method_kind",
        "method_id",
        "order_index",
        "information_set",
        "information_set_fingerprint",
        "run_plan_artifact_sha256",
        "run_plan_internal_sha256",
        "train_predictor_input_ref",
        "development_predictor_input_ref",
        "development_targets_ref",
        "shared_component_assignment_ref",
        "train_development_outcomes_sha256",
        "non_data_authority_hashes",
        "neural_registry_sha256",
        "authority_binding_status",
        "primary_sign_gate",
        "resource_budget_sha256",
        "estimands",
        "ordinary_only",
        "protected_partition_access",
        "future_stage_access",
        "sign_gate_change_authorized",
        "publication_claim_authorized",
    }
    status = payload.get("status") if isinstance(payload, Mapping) else None
    if status == METHOD_HOLD_STATUS:
        expected_keys = common_keys | {
            "hold_code",
            "hold_reason",
            "missing_requirements",
        }
    else:
        expected_keys = common_keys | {"resource_accounting"}
    raw = _strict_keys(payload, expected_keys, label="method evidence")
    if raw.get("schema_version") != METHOD_EVIDENCE_SCHEMA:
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_SCHEMA", "bad method evidence schema")
    if raw.get("artifact_role") != METHOD_EVIDENCE_ROLE:
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_SCHEMA", "bad method evidence role")
    method_id = _identifier(raw.get("method_id"), label="method_id")
    if method_id not in FROZEN_BASELINE_METHOD_ORDER:
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_METHOD_ORDER", "unknown method")
    order_index = raw.get("order_index")
    if isinstance(order_index, bool) or not isinstance(order_index, int):
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_SCHEMA", "order_index must be integer")
    if FROZEN_BASELINE_METHOD_ORDER[order_index] != method_id:
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_METHOD_ORDER", "method index drift")
    expected_information_set = METHOD_INFORMATION_SETS.get(
        method_id, PAIRED_INFORMATION_SET
    )
    if raw.get("information_set") != expected_information_set:
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_INFORMATION_SET", "information set drift")
    expected_fingerprint = information_set_fingerprint(expected_information_set)
    if raw.get("information_set_fingerprint") != expected_fingerprint:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_INFORMATION_SET",
            "information-set fingerprint drift",
        )
    for field in (
        "run_plan_artifact_sha256",
        "run_plan_internal_sha256",
        "train_development_outcomes_sha256",
        "resource_budget_sha256",
        "neural_registry_sha256",
    ):
        _digest(raw.get(field), label=field)
    _normalize_ref(
        raw.get("train_predictor_input_ref"),
        label="train predictor ref",
        required_prefix=("development", "predictor_inputs"),
    )
    _normalize_ref(
        raw.get("development_predictor_input_ref"),
        label="development predictor ref",
        required_prefix=("development", "predictor_inputs"),
    )
    _normalize_ref(
        raw.get("development_targets_ref"),
        label="development targets ref",
        required_prefix=("development", "targets"),
    )
    _normalize_ref(
        raw.get("shared_component_assignment_ref"),
        label="shared assignment ref",
        required_prefix=("development", "assignments"),
    )
    _authorities(raw.get("non_data_authority_hashes"))
    if raw.get("neural_registry_sha256") != registry_sha256():
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REGISTRY",
            "method evidence neural registry SHA is stale",
        )
    if raw.get("authority_binding_status") != AUTHORITY_BINDING_STATUS:
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_AUTHORITY_BINDING", "authority status drift")
    if raw.get("primary_sign_gate") != SIGN_ACCURACY_GATE:
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_SIGN_GATE", "method sign gate drift")
    expected_boundaries = {
        "ordinary_only": True,
        "protected_partition_access": False,
        "future_stage_access": False,
        "sign_gate_change_authorized": False,
        "publication_claim_authorized": False,
    }
    for field, expected in expected_boundaries.items():
        if raw.get(field) is not expected:
            _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_BOUNDARY", f"{field} drift")
    estimands = raw.get("estimands")
    if not isinstance(estimands, list) or not estimands:
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_SCHEMA", "estimands must be non-empty")
    if status == SCALAR_COMPLETE_STATUS:
        if method_id not in SUPPORTED_SCALAR_METHOD_IDS or raw.get("method_kind") != "SCALAR":
            _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_SCHEMA", "invalid scalar evidence")
        accounting = _strict_keys(
            raw.get("resource_accounting"),
            {"status", "fit_units", "neural_forward_equivalents"},
            label="resource_accounting",
        )
        if accounting.get("status") != RESOURCE_ACCOUNTING_STATUS:
            _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_RESOURCE_BUDGET", "resource status drift")
        if (
            isinstance(accounting.get("fit_units"), bool)
            or not isinstance(accounting.get("fit_units"), int)
            or accounting["fit_units"] < 1
            or accounting.get("neural_forward_equivalents") != 0
        ):
            _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_RESOURCE_BUDGET", "resource counts invalid")
        normalized_estimands = []
        expected_est_keys = {
            "estimand_id",
            "slice_sha256",
            "train_row_binding",
            "development_row_binding",
            "fitted_state_ref",
            "prediction_ref",
            "replay_status",
        }
        for index, item in enumerate(estimands):
            est = _strict_keys(item, expected_est_keys, label=f"estimand {index}")
            _identifier(est.get("estimand_id"), label="estimand_id")
            _digest(est.get("slice_sha256"), label="slice_sha256")
            for binding_name in ("train_row_binding", "development_row_binding"):
                binding = _strict_keys(
                    est.get(binding_name),
                    {"n_rows", "row_set_sha256"},
                    label=binding_name,
                )
                if (
                    isinstance(binding.get("n_rows"), bool)
                    or not isinstance(binding.get("n_rows"), int)
                    or binding["n_rows"] < 1
                ):
                    _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_ROW_SET", "invalid row count")
                _digest(binding.get("row_set_sha256"), label="row_set_sha256")
            _validate_evidence_ref(est.get("fitted_state_ref"), label="fitted state ref")
            _validate_evidence_ref(est.get("prediction_ref"), label="prediction ref")
            if est.get("replay_status") != "EXACT_REFIT_AND_PREDICTION_REPLAY_COMPLETE":
                _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_REPLAY", "scalar replay status drift")
            normalized_estimands.append(copy.deepcopy(dict(est)))
    elif status == NEURAL_COMPLETE_STATUS:
        if method_id != FULL_PAIR_METHOD_ID or raw.get("method_kind") != "NEURAL":
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_SCHEMA",
                "only the contract FullPair method may carry neural evidence",
            )
        accounting = _strict_keys(
            raw.get("resource_accounting"),
            {
                "status",
                "fit_units",
                "neural_forward_equivalents",
                "maximum_conservative_peak_parameter_bytes",
                "fit_config_sha256",
            },
            label="neural resource_accounting",
        )
        if accounting.get("status") != NEURAL_RESOURCE_ACCOUNTING_STATUS:
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_RESOURCE_BUDGET",
                "neural resource status drift",
            )
        for field in (
            "fit_units",
            "neural_forward_equivalents",
            "maximum_conservative_peak_parameter_bytes",
        ):
            value = accounting.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                _hold(
                    "HOLD_DEVELOPMENT_EXACT_EIGHT_RESOURCE_BUDGET",
                    f"neural resource {field} is invalid",
                )
        _digest(accounting.get("fit_config_sha256"), label="fit_config_sha256")
        normalized_estimands = []
        expected_est_keys = {
            "estimand_id",
            "slice_sha256",
            "train_row_binding",
            "development_row_binding",
            "registered_method_id",
            "training_variant_id",
            "run_start_sha256",
            "development_activation_sha256",
            "derived_component_assignment_ref",
            "component_assignment_derivation_sha256",
            "development_target_sha256",
            "seed_states",
            "ensemble_ref",
            "prediction_ref",
            "replay_status",
        }
        for index, item in enumerate(estimands):
            est = _strict_keys(item, expected_est_keys, label=f"neural estimand {index}")
            _identifier(est.get("estimand_id"), label="estimand_id")
            for field in (
                "slice_sha256",
                "run_start_sha256",
                "development_activation_sha256",
                "component_assignment_derivation_sha256",
                "development_target_sha256",
            ):
                _digest(est.get(field), label=field)
            if (
                est.get("registered_method_id") != REGISTERED_FULL_PAIR_METHOD_ID
                or est.get("training_variant_id") != TRAINING_VARIANT_ID
            ):
                _hold(
                    "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REGISTRY",
                    "FullPair registry method or PRIMARY variant drift",
                )
            for binding_name in ("train_row_binding", "development_row_binding"):
                binding = _strict_keys(
                    est.get(binding_name),
                    {"n_rows", "row_set_sha256"},
                    label=binding_name,
                )
                if (
                    isinstance(binding.get("n_rows"), bool)
                    or not isinstance(binding.get("n_rows"), int)
                    or binding["n_rows"] < 1
                ):
                    _hold(
                        "HOLD_DEVELOPMENT_EXACT_EIGHT_ROW_SET",
                        "invalid neural row count",
                    )
                _digest(binding.get("row_set_sha256"), label="row_set_sha256")
            _validate_evidence_ref(
                est.get("derived_component_assignment_ref"),
                label="derived neural assignment ref",
            )
            _validate_evidence_ref(est.get("ensemble_ref"), label="ensemble ref")
            _validate_evidence_ref(est.get("prediction_ref"), label="prediction ref")
            seed_states = est.get("seed_states")
            if not isinstance(seed_states, list) or [
                seed.get("seed") if isinstance(seed, Mapping) else None
                for seed in seed_states
            ] != list(FROZEN_NEURAL_SEEDS):
                _hold(
                    "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REPLAY",
                    "neural seed states are not the frozen five in exact order",
                )
            for seed_index, seed_item in enumerate(seed_states):
                seed = _strict_keys(
                    seed_item,
                    {
                        "seed",
                        "state_ref",
                        "state_tensor_sha256",
                        "checkpoint_selection_sha256",
                        "development_prediction_sha256",
                        "resource_preflight_sha256",
                    },
                    label=f"seed state {seed_index}",
                )
                _validate_evidence_ref(seed.get("state_ref"), label="neural state ref")
                for field in (
                    "state_tensor_sha256",
                    "checkpoint_selection_sha256",
                    "development_prediction_sha256",
                    "resource_preflight_sha256",
                ):
                    _digest(seed.get(field), label=field)
            if est.get("replay_status") != (
                "EXACT_FIVE_STATE_LOAD_PREDICTION_AND_ENSEMBLE_REPLAY_COMPLETE"
            ):
                _hold(
                    "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REPLAY",
                    "neural replay status drift",
                )
            normalized_estimands.append(copy.deepcopy(dict(est)))
        if accounting["fit_units"] != len(estimands) * len(FROZEN_NEURAL_SEEDS):
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_RESOURCE_BUDGET",
                "neural fit-unit accounting differs from five seeds per estimand",
            )
    elif status == METHOD_HOLD_STATUS:
        if method_id != HGB_METHOD_ID:
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_SCHEMA",
                "HGB is the only method permitted to remain HOLD",
            )
        if raw.get("method_kind") != "HGB":
            _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_SCHEMA", "hold method kind drift")
        if raw.get("hold_code") != HGB_HOLD_CODE or raw.get("hold_reason") != HGB_HOLD_REASON:
            _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_HOLD", "method HOLD drift")
        missing = raw.get("missing_requirements")
        if not isinstance(missing, list) or not missing or any(
            not isinstance(item, str) or not item for item in missing
        ):
            _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_HOLD", "missing requirements invalid")
        normalized_estimands = []
        for index, item in enumerate(estimands):
            est = _strict_keys(
                item,
                {"estimand_id", "slice_sha256", "status", "hold_code"},
                label=f"held estimand {index}",
            )
            _identifier(est.get("estimand_id"), label="estimand_id")
            _digest(est.get("slice_sha256"), label="slice_sha256")
            if est.get("status") != METHOD_HOLD_STATUS or est.get("hold_code") != HGB_HOLD_CODE:
                _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_HOLD", "estimand HOLD drift")
            normalized_estimands.append(copy.deepcopy(dict(est)))
    else:
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_SCHEMA", "unsupported method evidence status")
    if estimands != sorted(estimands, key=lambda item: item["estimand_id"]):
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_SCHEMA", "estimands must be sorted")
    return copy.deepcopy(dict(raw))


def _canonical_method_evidence_bytes(payload: Mapping) -> bytes:
    try:
        return canonical_json_bytes(
            validate_development_exact_eight_method_evidence(payload)
        )
    except ArtifactIOError as exc:
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_JSON", str(exc))


def _write_method_evidence_exclusive(path: Path, payload: Mapping) -> str:
    try:
        return atomic_write_json_exclusive(
            path,
            validate_development_exact_eight_method_evidence(payload),
            validator=validate_development_exact_eight_method_evidence,
        )
    except ArtifactIOError as exc:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_WRITE",
            f"method evidence publication failed: {exc}",
        )


def _fit_scalar_method(
    *,
    root: Path,
    output_dir: Path,
    request: Mapping,
    plan: Mapping,
    all_train_rows: Sequence[Mapping],
    all_development_rows: Sequence[Mapping],
    method_id: str,
    order_index: int,
    budget: Mapping,
) -> tuple[dict, dict[str, str]]:
    evidence_estimands = []
    fit_units = 0
    budget_by_estimand = {
        item["estimand_id"]: item for item in budget["per_estimand"]
    }
    for slice_item in plan["estimand_slices"]:
        estimand_id = slice_item["estimand_id"]
        train_rows = _estimand_rows(all_train_rows, estimand_id)
        development_rows = _estimand_rows(all_development_rows, estimand_id)
        scalar_kwargs = _scalar_kwargs(
            method_id=method_id,
            root=root,
            request=request,
            plan=plan,
            all_train_rows=all_train_rows,
        )
        try:
            state = fit_scalar_fitted_state(
                method_id, train_rows, **scalar_kwargs
            )
        except ScalarFittedStateError as exc:
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_SCALAR_FIT",
                f"{method_id}/{estimand_id} scalar fit failed: {exc}",
            )
        slug = hashlib.sha256(estimand_id.encode("utf-8")).hexdigest()[:16]
        state_path = output_dir / f"{order_index:02d}-{method_id}-{slug}-state.json"
        writer_kwargs = _state_writer_kwargs(
            method_id=method_id,
            train_rows=train_rows,
            root=root,
            request=request,
            plan=plan,
            all_train_rows=all_train_rows,
        )
        try:
            state_sha = write_scalar_fitted_state_exclusive(
                state_path, state, **writer_kwargs
            )
            loaded = load_scalar_fitted_state_exact(
                state_path, state_sha, **writer_kwargs
            )
            predictions = recompute_scalar_predictions(
                loaded, development_rows
            )
        except ScalarFittedStateError as exc:
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_SCALAR_REPLAY",
                f"{method_id}/{estimand_id} scalar replay failed: {exc}",
            )
        prediction_path = (
            output_dir / f"{order_index:02d}-{method_id}-{slug}-predictions.jsonl"
        )
        prediction_sha = _write_prediction_exclusive(
            prediction_path, predictions
        )
        try:
            replay_raw = read_exact_regular_file(prediction_path, prediction_sha)
        except ArtifactIOError as exc:
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_REPLAY",
                f"prediction exact read failed: {exc}",
            )
        if replay_raw != _prediction_bytes(
            recompute_scalar_predictions(loaded, development_rows)
        ):
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_REPLAY",
                "published scalar predictions differ from exact recomputation",
            )
        per = budget_by_estimand[estimand_id]
        fit_units += (
            per["scalar_fit_units"] - 3
        ) // 3 if method_id.endswith("RIDGE") else 1
        evidence_estimands.append({
            "estimand_id": estimand_id,
            "slice_sha256": slice_item["slice_sha256"],
            "train_row_binding": {
                "n_rows": len(train_rows),
                "row_set_sha256": canonical_sha256(
                    [row["row_id"] for row in train_rows]
                ),
            },
            "development_row_binding": {
                "n_rows": len(development_rows),
                "row_set_sha256": canonical_sha256(
                    [row["row_id"] for row in development_rows]
                ),
            },
            "fitted_state_ref": _make_ref(root, state_path, state_sha),
            "prediction_ref": _make_ref(
                root, prediction_path, prediction_sha
            ),
            "replay_status": "EXACT_REFIT_AND_PREDICTION_REPLAY_COMPLETE",
        })
    payload = {
        **_method_common(
            method_id=method_id,
            order_index=order_index,
            request=request,
            plan=plan,
            budget=budget,
        ),
        "status": SCALAR_COMPLETE_STATUS,
        "method_kind": "SCALAR",
        "resource_accounting": {
            "status": RESOURCE_ACCOUNTING_STATUS,
            "fit_units": fit_units,
            "neural_forward_equivalents": 0,
        },
        "estimands": sorted(
            evidence_estimands, key=lambda item: item["estimand_id"]
        ),
    }
    validated = validate_development_exact_eight_method_evidence(payload)
    evidence_path = output_dir / f"{order_index:02d}-{method_id}-evidence.json"
    evidence_sha = _write_method_evidence_exclusive(evidence_path, validated)
    return validated, _make_ref(root, evidence_path, evidence_sha)


def _assert_neural_state_bindings(
    state: Mapping,
    *,
    request: Mapping,
    plan: Mapping,
    estimand_id: str,
    assignment: Mapping,
    development_targets: Mapping[str, float],
    model_config: ModelConfig,
    fit_config: NeuralDevelopmentFitConfig,
    run_start_sha256: str,
    development_activation_sha256: str,
) -> None:
    expected = {
        "method_id": REGISTERED_FULL_PAIR_METHOD_ID,
        "training_variant_id": TRAINING_VARIANT_ID,
        "registry_sha256": request["expected_neural_registry_sha256"],
        "information_set": PAIRED_INFORMATION_SET,
        "information_set_fingerprint": information_set_fingerprint(
            PAIRED_INFORMATION_SET
        ),
        "train_development_outcomes_sha256": plan[
            "train_development_outcomes_sha256"
        ],
        "non_data_authority_hashes": request[
            "expected_non_data_authority_hashes"
        ],
        "estimand_id": estimand_id,
        "run_start_sha256": run_start_sha256,
        "development_activation_sha256": development_activation_sha256,
        "model_config": asdict(model_config),
        "fit_config": asdict(fit_config),
        "development_target_sha256": canonical_sha256(development_targets),
    }
    for field, value in expected.items():
        if state.get(field) != value:
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_BINDING",
                f"neural state differs from exact orchestrator {field}",
                estimand_id=estimand_id,
            )
    component_binding = state["component_assignment_binding"]
    expected_component_binding = {
        "assignment_sha256": assignment["assignment_sha256"],
        "information_set_fingerprint": information_set_fingerprint(
            PAIRED_INFORMATION_SET
        ),
        "train_development_outcomes_sha256": plan[
            "train_development_outcomes_sha256"
        ],
        "source_component_split_manifest_source_sha256": plan[
            "authority_source_sha256"
        ]["source_component_split_manifest"],
        "source_component_split_manifest_projection_sha256": plan[
            "development_projection_hashes"
        ]["source_component_split_manifest_development_projection"],
    }
    if component_binding != expected_component_binding:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_BINDING",
            "neural state component assignment differs from shared-authority derivation",
            estimand_id=estimand_id,
        )
    if (
        state["checkpoint_selection"]["sign_gate"] != SIGN_ACCURACY_GATE
        or state["checkpoint_selection"]["sign_gate_unchanged"]
        != SIGN_ACCURACY_GATE
    ):
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_SIGN_GATE",
            "neural checkpoint selection changed the primary sign gate",
        )
    resource = state["resource_preflight"]
    budget = plan["executable_config"]["resource_budget"]
    if (
        resource["sequence_encoder_forward_equivalents"]
        > budget["max_forward_equivalents_per_fit"]
        or resource["peak_batch_sequence_tokens"]
        > budget["max_sequence_tokens_per_batch"]
        or resource["conservative_peak_parameter_bytes"]
        > budget["max_peak_memory_bytes_per_fit"]
    ):
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_RESOURCE_BUDGET",
            "neural state resource preflight exceeds exact run-plan limits",
        )
    expected_boundaries = {
        "ordinary_only": True,
        "internal_test_access": False,
        "sealed_access": False,
        "downstream_unlock": False,
        "execution_authorized": False,
        "independent_authority_verified": False,
        "training_ready": False,
        "publication_evidence": False,
    }
    for field, value in expected_boundaries.items():
        if state.get(field) is not value:
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_BOUNDARY",
                f"neural state {field} boundary drift",
            )


def _fit_neural_method(
    *,
    root: Path,
    output_dir: Path,
    request: Mapping,
    plan: Mapping,
    all_train_rows: Sequence[Mapping],
    all_development_rows: Sequence[Mapping],
    development_targets: Mapping[str, float],
    train_components: Mapping[str, str],
    inventory_components: Mapping[str, str],
    method_id: str,
    order_index: int,
    budget: Mapping,
) -> tuple[dict, dict[str, str]]:
    if method_id != FULL_PAIR_METHOD_ID:
        raise AssertionError("neural method path is reserved for contract FullPair")
    model_config = _neural_model_config(plan)
    fit_config = _neural_fit_config(plan)
    run_start_sha256, activation_sha256 = _neural_run_hashes(request, plan)
    evidence_estimands = []
    total_forward_equivalents = 0
    maximum_peak_parameter_bytes = 0
    fit_units = 0
    for slice_item in plan["estimand_slices"]:
        estimand_id = slice_item["estimand_id"]
        train_rows = _estimand_rows(all_train_rows, estimand_id)
        development_rows = _estimand_rows(all_development_rows, estimand_id)
        target_map = {
            row["row_id"]: development_targets[row["row_id"]]
            for row in development_rows
        }
        assignment, derivation_sha256 = _derive_neural_assignment(
            request=request,
            plan=plan,
            train_rows=train_rows,
            development_rows=development_rows,
            train_components=train_components,
            inventory_components=inventory_components,
        )
        slug = hashlib.sha256(estimand_id.encode("utf-8")).hexdigest()[:16]
        assignment_path = output_dir / (
            f"{order_index:02d}-{method_id}-{slug}-paired-assignment.json"
        )
        assignment_kwargs = _neural_assignment_kwargs(
            request=request,
            plan=plan,
            train_rows=train_rows,
            development_rows=development_rows,
        )
        try:
            assignment_sha = write_neural_source_component_assignment_exclusive(
                assignment_path,
                assignment,
                **assignment_kwargs,
            )
            loaded_assignment = load_neural_source_component_assignment_exact(
                assignment_path,
                assignment_sha,
                **assignment_kwargs,
            )
            if loaded_assignment != assignment:
                raise NeuralDevelopmentStateError(
                    "derived paired assignment differs after exact load"
                )
            states, ensemble = fit_neural_development_five_seed_ensemble(
                REGISTERED_FULL_PAIR_METHOD_ID,
                model_config,
                fit_config,
                train_rows,
                development_rows,
                target_map,
                loaded_assignment,
                expected_registry_sha256=request[
                    "expected_neural_registry_sha256"
                ],
                run_start_sha256=run_start_sha256,
                development_activation_sha256=activation_sha256,
            )
        except NeuralDevelopmentStateError as exc:
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_FIT",
                f"{method_id}/{estimand_id} five-seed fit failed: {exc}",
            )
        loaded_states = []
        seed_records = []
        for seed, state in zip(FROZEN_NEURAL_SEEDS, states, strict=True):
            state_path = output_dir / (
                f"{order_index:02d}-{method_id}-{slug}-seed-{seed}-state.pt"
            )
            try:
                state_sha = save_neural_development_state_exclusive(
                    state_path, state
                )
                loaded = load_neural_development_state_exact(
                    state_path, state_sha
                )
                _assert_neural_state_bindings(
                    loaded,
                    request=request,
                    plan=plan,
                    estimand_id=estimand_id,
                    assignment=loaded_assignment,
                    development_targets=target_map,
                    model_config=model_config,
                    fit_config=fit_config,
                    run_start_sha256=run_start_sha256,
                    development_activation_sha256=activation_sha256,
                )
                recomputed = recompute_neural_development_predictions(
                    loaded, development_rows
                )
            except NeuralDevelopmentStateError as exc:
                _hold(
                    "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REPLAY",
                    f"{method_id}/{estimand_id}/seed-{seed} replay failed: {exc}",
                )
            if _neural_prediction_bytes(recomputed) != _neural_prediction_bytes(
                loaded["development_predictions"]
            ):
                _hold(
                    "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REPLAY",
                    "loaded neural state predictions differ from exact recomputation",
                    estimand_id=estimand_id,
                    seed=seed,
                )
            resource = loaded["resource_preflight"]
            total_forward_equivalents += resource[
                "sequence_encoder_forward_equivalents"
            ]
            maximum_peak_parameter_bytes = max(
                maximum_peak_parameter_bytes,
                resource["conservative_peak_parameter_bytes"],
            )
            loaded_states.append(loaded)
            seed_records.append({
                "seed": seed,
                "state_ref": _make_ref(root, state_path, state_sha),
                "state_tensor_sha256": loaded["state_tensor_sha256"],
                "checkpoint_selection_sha256": loaded[
                    "checkpoint_selection_sha256"
                ],
                "development_prediction_sha256": loaded[
                    "development_prediction_sha256"
                ],
                "resource_preflight_sha256": canonical_sha256(resource),
            })
        try:
            replayed_ensemble = ensemble_neural_development_states(
                loaded_states
            )
            if canonical_json_bytes(replayed_ensemble) != canonical_json_bytes(
                ensemble
            ):
                raise NeuralDevelopmentStateError(
                    "five-seed ensemble differs after exact state reload"
                )
        except (ArtifactIOError, NeuralDevelopmentStateError) as exc:
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REPLAY",
                f"{method_id}/{estimand_id} ensemble replay failed: {exc}",
            )
        ensemble_path = output_dir / (
            f"{order_index:02d}-{method_id}-{slug}-ensemble.json"
        )
        ensemble_sha = _write_neural_ensemble_exclusive(
            ensemble_path, replayed_ensemble
        )
        loaded_ensemble = _load_neural_ensemble_exact(
            ensemble_path, ensemble_sha
        )
        prediction_path = output_dir / (
            f"{order_index:02d}-{method_id}-{slug}-predictions.jsonl"
        )
        prediction_sha = _write_neural_prediction_exclusive(
            prediction_path, loaded_ensemble["ensemble_predictions"]
        )
        try:
            prediction_raw = read_exact_regular_file(
                prediction_path, prediction_sha
            )
        except ArtifactIOError as exc:
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REPLAY",
                f"neural ensemble prediction exact read failed: {exc}",
            )
        if prediction_raw != _neural_prediction_bytes(
            ensemble_neural_development_states(loaded_states)[
                "ensemble_predictions"
            ]
        ):
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REPLAY",
                "published ensemble prediction differs from five loaded states",
            )
        fit_units += len(FROZEN_NEURAL_SEEDS)
        evidence_estimands.append({
            "estimand_id": estimand_id,
            "slice_sha256": slice_item["slice_sha256"],
            "train_row_binding": {
                "n_rows": len(train_rows),
                "row_set_sha256": canonical_sha256(
                    [row["row_id"] for row in train_rows]
                ),
            },
            "development_row_binding": {
                "n_rows": len(development_rows),
                "row_set_sha256": canonical_sha256(
                    [row["row_id"] for row in development_rows]
                ),
            },
            "registered_method_id": REGISTERED_FULL_PAIR_METHOD_ID,
            "training_variant_id": TRAINING_VARIANT_ID,
            "run_start_sha256": run_start_sha256,
            "development_activation_sha256": activation_sha256,
            "derived_component_assignment_ref": _make_ref(
                root, assignment_path, assignment_sha
            ),
            "component_assignment_derivation_sha256": derivation_sha256,
            "development_target_sha256": canonical_sha256(target_map),
            "seed_states": seed_records,
            "ensemble_ref": _make_ref(root, ensemble_path, ensemble_sha),
            "prediction_ref": _make_ref(
                root, prediction_path, prediction_sha
            ),
            "replay_status": (
                "EXACT_FIVE_STATE_LOAD_PREDICTION_AND_ENSEMBLE_REPLAY_COMPLETE"
            ),
        })
    payload = {
        **_method_common(
            method_id=method_id,
            order_index=order_index,
            request=request,
            plan=plan,
            budget=budget,
        ),
        "status": NEURAL_COMPLETE_STATUS,
        "method_kind": "NEURAL",
        "resource_accounting": {
            "status": NEURAL_RESOURCE_ACCOUNTING_STATUS,
            "fit_units": fit_units,
            "neural_forward_equivalents": total_forward_equivalents,
            "maximum_conservative_peak_parameter_bytes": (
                maximum_peak_parameter_bytes
            ),
            "fit_config_sha256": canonical_sha256(asdict(fit_config)),
        },
        "estimands": sorted(
            evidence_estimands, key=lambda item: item["estimand_id"]
        ),
    }
    validated = validate_development_exact_eight_method_evidence(payload)
    evidence_path = output_dir / f"{order_index:02d}-{method_id}-evidence.json"
    evidence_sha = _write_method_evidence_exclusive(evidence_path, validated)
    return validated, _make_ref(root, evidence_path, evidence_sha)


def _held_method(
    *,
    root: Path,
    output_dir: Path,
    request: Mapping,
    plan: Mapping,
    method_id: str,
    order_index: int,
    budget: Mapping,
) -> tuple[dict, dict[str, str]]:
    if method_id != HGB_METHOD_ID:
        raise AssertionError("HGB is the only supported held-method path")
    kind = "HGB"
    code = HGB_HOLD_CODE
    reason = HGB_HOLD_REASON
    missing = [
        "real-run-plan-bound observed environment qualification",
        "real ordinary-train HGB assignment derived from shared V2 authority",
        "real exact numeric refit and development prediction replay",
    ]
    payload = {
        **_method_common(
            method_id=method_id,
            order_index=order_index,
            request=request,
            plan=plan,
            budget=budget,
        ),
        "status": METHOD_HOLD_STATUS,
        "method_kind": kind,
        "hold_code": code,
        "hold_reason": reason,
        "missing_requirements": missing,
        "estimands": [
            {
                "estimand_id": item["estimand_id"],
                "slice_sha256": item["slice_sha256"],
                "status": METHOD_HOLD_STATUS,
                "hold_code": code,
            }
            for item in plan["estimand_slices"]
        ],
    }
    validated = validate_development_exact_eight_method_evidence(payload)
    evidence_path = output_dir / f"{order_index:02d}-{method_id}-hold.json"
    evidence_sha = _write_method_evidence_exclusive(evidence_path, validated)
    return validated, _make_ref(root, evidence_path, evidence_sha)


def validate_development_exact_eight_result(payload: Mapping) -> dict:
    keys = {
        "schema_version",
        "artifact_role",
        "status",
        "request",
        "request_sha256",
        "run_plan_internal_sha256",
        "paired_policy_ref",
        "row_inventory_ref",
        "method_order",
        "primary_sign_gate",
        "information_set_fingerprints",
        "partition_row_set_sha256",
        "authority_binding_status",
        "resource_budget_binding",
        "methods",
        "completion",
        "ordinary_only",
        "protected_partition_access",
        "future_stage_access",
        "sign_gate_change_authorized",
        "publication_claim_authorized",
    }
    raw = _strict_keys(payload, keys, label="exact-eight result")
    if raw.get("schema_version") != RESULT_SCHEMA or raw.get("artifact_role") != RESULT_ROLE:
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_SCHEMA", "result identity drift")
    request = validate_development_exact_eight_request(raw.get("request"))
    if raw.get("request_sha256") != canonical_sha256(request):
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_BINDING", "request SHA drift")
    _digest(raw.get("run_plan_internal_sha256"), label="run plan internal")
    _normalize_ref(
        raw.get("paired_policy_ref"),
        label="paired policy ref",
        required_prefix=("development", "authorities"),
    )
    _normalize_ref(
        raw.get("row_inventory_ref"),
        label="row inventory ref",
        required_prefix=("development", "run_plan_inputs"),
    )
    order = raw.get("method_order")
    if order != list(FROZEN_BASELINE_METHOD_ORDER):
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_METHOD_ORDER", "result method order drift")
    if raw.get("primary_sign_gate") != SIGN_ACCURACY_GATE:
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_SIGN_GATE", "result sign gate drift")
    fingerprints = _strict_keys(
        raw.get("information_set_fingerprints"), set(order), label="fingerprints"
    )
    for method_id in order:
        expected = information_set_fingerprint(
            METHOD_INFORMATION_SETS.get(method_id, PAIRED_INFORMATION_SET)
        )
        if fingerprints[method_id] != expected:
            _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_INFORMATION_SET", "result fingerprint drift")
    partitions = _strict_keys(
        raw.get("partition_row_set_sha256"),
        {ORDINARY_TRAIN, ORDINARY_DEVELOPMENT},
        label="partition row sets",
    )
    for value in partitions.values():
        _digest(value, label="partition row set")
    if raw.get("authority_binding_status") != AUTHORITY_BINDING_STATUS:
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_AUTHORITY_BINDING", "result authority status drift")
    budget = _strict_keys(
        raw.get("resource_budget_binding"),
        {
            "resource_budget",
            "resource_budget_sha256",
            "accounting_status",
            "executed_scalar_fit_units",
            "executed_neural_fit_units",
            "executed_total_fit_units",
            "neural_forward_equivalents_per_scalar_fit",
            "neural_forward_equivalents",
            "maximum_conservative_peak_parameter_bytes",
        },
        label="resource budget binding",
    )
    if budget.get("resource_budget_sha256") != canonical_sha256(budget.get("resource_budget")):
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_RESOURCE_BUDGET", "budget SHA drift")
    integer_counts = (
        "executed_scalar_fit_units",
        "executed_neural_fit_units",
        "executed_total_fit_units",
        "neural_forward_equivalents",
        "maximum_conservative_peak_parameter_bytes",
    )
    if (
        budget.get("accounting_status") != RESOURCE_ACCOUNTING_STATUS
        or budget.get("neural_forward_equivalents_per_scalar_fit") != 0
        or any(
            isinstance(budget.get(field), bool)
            or not isinstance(budget.get(field), int)
            or budget[field] < 1
            for field in integer_counts
        )
        or budget["executed_total_fit_units"]
        != budget["executed_scalar_fit_units"]
        + budget["executed_neural_fit_units"]
        or budget["executed_total_fit_units"]
        > budget["resource_budget"]["max_total_fit_units"]
        or budget["maximum_conservative_peak_parameter_bytes"]
        > budget["resource_budget"]["max_peak_memory_bytes_per_fit"]
    ):
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_RESOURCE_BUDGET", "budget accounting drift")
    methods = raw.get("methods")
    if not isinstance(methods, list) or len(methods) != len(order):
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_METHOD_ORDER", "result method records incomplete")
    normalized_methods = []
    for index, item in enumerate(methods):
        record = _strict_keys(
            item,
            {"order_index", "method_id", "status", "evidence_ref"},
            label=f"method record {index}",
        )
        if record.get("order_index") != index or record.get("method_id") != order[index]:
            _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_METHOD_ORDER", "method record drift")
        if record.get("status") not in {
            SCALAR_COMPLETE_STATUS,
            NEURAL_COMPLETE_STATUS,
            METHOD_HOLD_STATUS,
        }:
            _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_SCHEMA", "method status drift")
        _validate_evidence_ref(record.get("evidence_ref"), label="method evidence ref")
        normalized_methods.append(copy.deepcopy(dict(record)))
    complete_count = sum(
        item["status"] in {SCALAR_COMPLETE_STATUS, NEURAL_COMPLETE_STATUS}
        for item in methods
    )
    hold_count = sum(item["status"] == METHOD_HOLD_STATUS for item in methods)
    completion = _strict_keys(
        raw.get("completion"),
        {"complete_method_count", "held_method_count", "exact_eight_complete"},
        label="completion",
    )
    expected_completion = {
        "complete_method_count": complete_count,
        "held_method_count": hold_count,
        "exact_eight_complete": False,
    }
    if completion != expected_completion:
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_COMPLETION", "completion counts drift")
    expected_status_by_method = {
        **{
            method_id: SCALAR_COMPLETE_STATUS
            for method_id in SUPPORTED_SCALAR_METHOD_IDS
        },
        HGB_METHOD_ID: METHOD_HOLD_STATUS,
        FULL_PAIR_METHOD_ID: NEURAL_COMPLETE_STATUS,
    }
    if (
        complete_count != 7
        or hold_count != 1
        or any(
            item["status"] != expected_status_by_method[item["method_id"]]
            for item in methods
        )
        or raw.get("status") != PARTIAL_STATUS
    ):
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_COMPLETION", "result status overclaims completion")
    expected_boundaries = {
        "ordinary_only": True,
        "protected_partition_access": False,
        "future_stage_access": False,
        "sign_gate_change_authorized": False,
        "publication_claim_authorized": False,
    }
    for field, expected in expected_boundaries.items():
        if raw.get(field) is not expected:
            _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_BOUNDARY", f"result {field} drift")
    return copy.deepcopy(dict(raw))


def _canonical_result_bytes(payload: Mapping) -> bytes:
    try:
        return canonical_json_bytes(validate_development_exact_eight_result(payload))
    except ArtifactIOError as exc:
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_JSON", str(exc))


def _prepare_context(root: Path, request: Mapping) -> dict:
    plan, paired, inventory = _load_plan_context(root, request)
    _cross_bind_authorities(plan, request["expected_non_data_authority_hashes"])
    train_rows = _load_predictor_rows_exact(
        root,
        request["train_predictor_input_ref"],
        partition=ORDINARY_TRAIN,
        label="ordinary train predictor rows",
    )
    development_rows = _load_predictor_rows_exact(
        root,
        request["development_predictor_input_ref"],
        partition=ORDINARY_DEVELOPMENT,
        label="ordinary development predictor rows",
    )
    _cross_bind_rows(
        plan=plan,
        inventory=inventory,
        train_rows=train_rows,
        development_rows=development_rows,
    )
    development_targets_payload, development_targets = (
        _load_development_targets_exact(
            root,
            request["development_targets_ref"],
            plan=plan,
            development_rows=development_rows,
        )
    )
    assignment = _load_shared_assignment(
        root=root,
        request=request,
        plan=plan,
        train_rows=train_rows,
    )
    components = _cross_bind_assignment(
        assignment=assignment, inventory=inventory, train_rows=train_rows
    )
    inventory_components = {
        row["row_id"]: row["source_component_id"]
        for row in inventory["rows"]
    }
    budget = _validate_budget(
        plan=plan,
        train_rows=train_rows,
        development_rows=development_rows,
        components=components,
    )
    return {
        "plan": plan,
        "paired": paired,
        "inventory": inventory,
        "train_rows": train_rows,
        "development_rows": development_rows,
        "development_targets_payload": development_targets_payload,
        "development_targets": development_targets,
        "assignment": assignment,
        "components": components,
        "inventory_components": inventory_components,
        "budget": budget,
    }


def execute_development_exact_eight(
    run_root: Path, request_payload: Mapping
) -> tuple[dict, dict[str, str]]:
    """Fit/replay all currently qualified methods and publish a PARTIAL result.

    The request is fully scanned before the first content read.  Output parents
    must already exist and every final artifact name is exclusive.
    """

    _scan_preaccess(request_payload)
    _scan_preaccess(os.fspath(run_root), path="run_root")
    request = validate_development_exact_eight_request(request_payload)
    root = _checked_root(run_root)
    output_dir = _path_from_relative(
        root,
        request["output_relative_dir"],
        label="exact-eight output directory",
        require_exists=True,
    )
    try:
        output_metadata = os.lstat(output_dir)
    except OSError as exc:
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_PATH", f"output directory missing: {exc}")
    if stat.S_ISLNK(output_metadata.st_mode) or not stat.S_ISDIR(output_metadata.st_mode):
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_PATH", "output must be a real directory")
    context = _prepare_context(root, request)
    plan = context["plan"]
    budget = context["budget"]
    method_records = []
    evidence_payloads = []
    for order_index, method_id in enumerate(FROZEN_BASELINE_METHOD_ORDER):
        if method_id in SUPPORTED_SCALAR_METHOD_IDS:
            evidence, evidence_ref = _fit_scalar_method(
                root=root,
                output_dir=output_dir,
                request=request,
                plan=plan,
                all_train_rows=context["train_rows"],
                all_development_rows=context["development_rows"],
                method_id=method_id,
                order_index=order_index,
                budget=budget,
            )
        elif method_id == FULL_PAIR_METHOD_ID:
            evidence, evidence_ref = _fit_neural_method(
                root=root,
                output_dir=output_dir,
                request=request,
                plan=plan,
                all_train_rows=context["train_rows"],
                all_development_rows=context["development_rows"],
                development_targets=context["development_targets"],
                train_components=context["components"],
                inventory_components=context["inventory_components"],
                method_id=method_id,
                order_index=order_index,
                budget=budget,
            )
        else:
            evidence, evidence_ref = _held_method(
                root=root,
                output_dir=output_dir,
                request=request,
                plan=plan,
                method_id=method_id,
                order_index=order_index,
                budget=budget,
            )
        evidence_payloads.append(evidence)
        method_records.append({
            "order_index": order_index,
            "method_id": method_id,
            "status": evidence["status"],
            "evidence_ref": evidence_ref,
        })
    complete_count = sum(
        item["status"] in {SCALAR_COMPLETE_STATUS, NEURAL_COMPLETE_STATUS}
        for item in method_records
    )
    held_count = len(method_records) - complete_count
    neural_evidence = next(
        item
        for item in evidence_payloads
        if item["method_id"] == FULL_PAIR_METHOD_ID
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "artifact_role": RESULT_ROLE,
        "status": PARTIAL_STATUS,
        "request": copy.deepcopy(request),
        "request_sha256": canonical_sha256(request),
        "run_plan_internal_sha256": plan["plan_sha256"],
        "paired_policy_ref": copy.deepcopy(
            plan["source_refs"]["paired_baseline_policy"]
        ),
        "row_inventory_ref": copy.deepcopy(plan["source_refs"]["row_inventory"]),
        "method_order": list(FROZEN_BASELINE_METHOD_ORDER),
        "primary_sign_gate": SIGN_ACCURACY_GATE,
        "information_set_fingerprints": {
            method_id: information_set_fingerprint(
                METHOD_INFORMATION_SETS.get(method_id, PAIRED_INFORMATION_SET)
            )
            for method_id in FROZEN_BASELINE_METHOD_ORDER
        },
        "partition_row_set_sha256": copy.deepcopy(
            plan["partition_row_set_sha256"]
        ),
        "authority_binding_status": AUTHORITY_BINDING_STATUS,
        "resource_budget_binding": {
            "resource_budget": copy.deepcopy(budget["resource_budget"]),
            "resource_budget_sha256": budget["resource_budget_sha256"],
            "accounting_status": budget["accounting_status"],
            "executed_scalar_fit_units": budget["executed_scalar_fit_units"],
            "executed_neural_fit_units": budget["executed_neural_fit_units"],
            "executed_total_fit_units": budget["executed_total_fit_units"],
            "neural_forward_equivalents_per_scalar_fit": 0,
            "neural_forward_equivalents": neural_evidence[
                "resource_accounting"
            ]["neural_forward_equivalents"],
            "maximum_conservative_peak_parameter_bytes": neural_evidence[
                "resource_accounting"
            ]["maximum_conservative_peak_parameter_bytes"],
        },
        "methods": method_records,
        "completion": {
            "complete_method_count": complete_count,
            "held_method_count": held_count,
            "exact_eight_complete": False,
        },
        "ordinary_only": True,
        "protected_partition_access": False,
        "future_stage_access": False,
        "sign_gate_change_authorized": False,
        "publication_claim_authorized": False,
    }
    validated = validate_development_exact_eight_result(result)
    result_path = output_dir / "exact-eight-result.json"
    try:
        result_sha = atomic_write_json_exclusive(
            result_path,
            validated,
            validator=validate_development_exact_eight_result,
        )
    except ArtifactIOError as exc:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_WRITE",
            f"result publication failed: {exc}",
        )
    result_ref = _make_ref(root, result_path, result_sha)
    replayed = load_development_exact_eight_result_exact(root, result_ref)
    if _canonical_result_bytes(replayed) != _canonical_result_bytes(validated):
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_REPLAY",
            "published result differs from full exact replay",
        )
    return replayed, result_ref


def _load_method_evidence_exact(
    root: Path, reference: Mapping
) -> dict:
    path = _ref_path(root, reference, label="method evidence")
    try:
        raw = read_exact_regular_file(path, reference["sha256"])
    except ArtifactIOError as exc:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_DEPENDENCY",
            f"method evidence exact read failed: {exc}",
        )
    payload = validate_development_exact_eight_method_evidence(
        _parse_json_object(raw, label="method evidence")
    )
    if _canonical_method_evidence_bytes(payload) != raw:
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_JSON", "method evidence is not canonical")
    return payload


def _cross_bind_method_evidence(
    evidence: Mapping,
    *,
    request: Mapping,
    plan: Mapping,
    budget: Mapping,
) -> None:
    expected = _method_common(
        method_id=evidence["method_id"],
        order_index=evidence["order_index"],
        request=request,
        plan=plan,
        budget=budget,
    )
    for field, value in expected.items():
        if evidence.get(field) != value:
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_BINDING",
                f"method evidence differs from exact context field {field}",
                method_id=evidence["method_id"],
            )
    slice_by_estimand = {
        item["estimand_id"]: item["slice_sha256"]
        for item in plan["estimand_slices"]
    }
    if {
        item["estimand_id"]: item["slice_sha256"]
        for item in evidence["estimands"]
    } != slice_by_estimand:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_BINDING",
            "method evidence estimand slices differ from exact run plan",
            method_id=evidence["method_id"],
        )


def _replay_neural_estimand(
    *,
    root: Path,
    request: Mapping,
    plan: Mapping,
    context: Mapping,
    unit: Mapping,
) -> tuple[int, int]:
    estimand_id = unit["estimand_id"]
    train_rows = _estimand_rows(context["train_rows"], estimand_id)
    development_rows = _estimand_rows(
        context["development_rows"], estimand_id
    )
    target_map = {
        row["row_id"]: context["development_targets"][row["row_id"]]
        for row in development_rows
    }
    expected_row_bindings = {
        "train_row_binding": {
            "n_rows": len(train_rows),
            "row_set_sha256": canonical_sha256(
                [row["row_id"] for row in train_rows]
            ),
        },
        "development_row_binding": {
            "n_rows": len(development_rows),
            "row_set_sha256": canonical_sha256(
                [row["row_id"] for row in development_rows]
            ),
        },
    }
    if any(
        unit[field] != value for field, value in expected_row_bindings.items()
    ):
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_ROW_SET",
            "neural evidence row binding differs from exact predictor rows",
            estimand_id=estimand_id,
        )
    expected_assignment, expected_derivation_sha = _derive_neural_assignment(
        request=request,
        plan=plan,
        train_rows=train_rows,
        development_rows=development_rows,
        train_components=context["components"],
        inventory_components=context["inventory_components"],
    )
    assignment_path = _ref_path(
        root,
        unit["derived_component_assignment_ref"],
        label="derived neural assignment",
    )
    assignment_kwargs = _neural_assignment_kwargs(
        request=request,
        plan=plan,
        train_rows=train_rows,
        development_rows=development_rows,
    )
    try:
        assignment = load_neural_source_component_assignment_exact(
            assignment_path,
            unit["derived_component_assignment_ref"]["sha256"],
            **assignment_kwargs,
        )
    except NeuralDevelopmentStateError as exc:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REPLAY",
            f"derived neural assignment exact replay failed: {exc}",
        )
    if (
        assignment != expected_assignment
        or unit["component_assignment_derivation_sha256"]
        != expected_derivation_sha
        or unit["development_target_sha256"] != canonical_sha256(target_map)
    ):
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_BINDING",
            "neural assignment or target derivation differs from exact shared inputs",
            estimand_id=estimand_id,
        )
    model_config = _neural_model_config(plan)
    fit_config = _neural_fit_config(plan)
    run_start_sha256, activation_sha256 = _neural_run_hashes(request, plan)
    if (
        unit["run_start_sha256"] != run_start_sha256
        or unit["development_activation_sha256"] != activation_sha256
    ):
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_BINDING",
            "neural run-start or development-activation binding drift",
        )
    loaded_states = []
    total_forward_equivalents = 0
    maximum_peak_parameter_bytes = 0
    for seed_record in unit["seed_states"]:
        state_path = _ref_path(
            root, seed_record["state_ref"], label="neural fitted state"
        )
        try:
            state = load_neural_development_state_exact(
                state_path, seed_record["state_ref"]["sha256"]
            )
            _assert_neural_state_bindings(
                state,
                request=request,
                plan=plan,
                estimand_id=estimand_id,
                assignment=assignment,
                development_targets=target_map,
                model_config=model_config,
                fit_config=fit_config,
                run_start_sha256=run_start_sha256,
                development_activation_sha256=activation_sha256,
            )
            predictions = recompute_neural_development_predictions(
                state, development_rows
            )
        except NeuralDevelopmentStateError as exc:
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REPLAY",
                f"neural state exact replay failed: {exc}",
            )
        expected_seed_record = {
            "seed": state["seed"],
            "state_ref": seed_record["state_ref"],
            "state_tensor_sha256": state["state_tensor_sha256"],
            "checkpoint_selection_sha256": state[
                "checkpoint_selection_sha256"
            ],
            "development_prediction_sha256": state[
                "development_prediction_sha256"
            ],
            "resource_preflight_sha256": canonical_sha256(
                state["resource_preflight"]
            ),
        }
        if seed_record != expected_seed_record or _neural_prediction_bytes(
            predictions
        ) != _neural_prediction_bytes(state["development_predictions"]):
            _hold(
                "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REPLAY",
                "neural seed record or prediction recomputation drift",
                seed=state["seed"],
            )
        resource = state["resource_preflight"]
        total_forward_equivalents += resource[
            "sequence_encoder_forward_equivalents"
        ]
        maximum_peak_parameter_bytes = max(
            maximum_peak_parameter_bytes,
            resource["conservative_peak_parameter_bytes"],
        )
        loaded_states.append(state)
    try:
        replayed_ensemble = ensemble_neural_development_states(loaded_states)
    except NeuralDevelopmentStateError as exc:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REPLAY",
            f"five loaded neural states cannot re-form ensemble: {exc}",
        )
    ensemble_path = _ref_path(
        root, unit["ensemble_ref"], label="neural ensemble"
    )
    ensemble = _load_neural_ensemble_exact(
        ensemble_path, unit["ensemble_ref"]["sha256"]
    )
    try:
        ensemble_matches = canonical_json_bytes(ensemble) == canonical_json_bytes(
            replayed_ensemble
        )
    except ArtifactIOError as exc:
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REPLAY", str(exc))
    prediction_path = _ref_path(
        root, unit["prediction_ref"], label="neural ensemble prediction"
    )
    try:
        prediction_raw = read_exact_regular_file(
            prediction_path, unit["prediction_ref"]["sha256"]
        )
    except ArtifactIOError as exc:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REPLAY",
            f"neural ensemble prediction exact read failed: {exc}",
        )
    if (
        not ensemble_matches
        or prediction_raw
        != _neural_prediction_bytes(replayed_ensemble["ensemble_predictions"])
    ):
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_NEURAL_REPLAY",
            "ensemble artifact/prediction differs from five exact-loaded states",
        )
    return total_forward_equivalents, maximum_peak_parameter_bytes


def load_development_exact_eight_result_exact(
    run_root: Path, result_ref: Mapping
) -> dict:
    """Exact-load and independently refit every completed scalar artifact."""

    _scan_preaccess(result_ref)
    _scan_preaccess(os.fspath(run_root), path="run_root")
    root = _checked_root(run_root)
    reference = _normalize_ref(
        result_ref,
        label="exact-eight result ref",
        required_prefix=("development", "exact_eight"),
    )
    path = _ref_path(root, reference, label="exact-eight result")
    try:
        raw = read_exact_regular_file(path, reference["sha256"])
    except ArtifactIOError as exc:
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_DEPENDENCY",
            f"result exact read failed: {exc}",
        )
    result = validate_development_exact_eight_result(
        _parse_json_object(raw, label="exact-eight result")
    )
    if _canonical_result_bytes(result) != raw:
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_JSON", "result bytes are not canonical")
    request = result["request"]
    context = _prepare_context(root, request)
    plan = context["plan"]
    if (
        result["run_plan_internal_sha256"] != plan["plan_sha256"]
        or result["partition_row_set_sha256"]
        != plan["partition_row_set_sha256"]
        or result["paired_policy_ref"]
        != plan["source_refs"]["paired_baseline_policy"]
        or result["row_inventory_ref"] != plan["source_refs"]["row_inventory"]
    ):
        _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_BINDING", "result/run-plan binding drift")
    budget = context["budget"]
    result_budget = result["resource_budget_binding"]
    if (
        result_budget["resource_budget"] != budget["resource_budget"]
        or result_budget["resource_budget_sha256"]
        != budget["resource_budget_sha256"]
        or result_budget["executed_scalar_fit_units"]
        != budget["executed_scalar_fit_units"]
        or result_budget["executed_neural_fit_units"]
        != budget["executed_neural_fit_units"]
        or result_budget["executed_total_fit_units"]
        != budget["executed_total_fit_units"]
    ):
        _hold(
            "HOLD_DEVELOPMENT_EXACT_EIGHT_RESOURCE_BUDGET",
            "result resource budget differs from exact recomputation",
        )
    for record in result["methods"]:
        evidence = _load_method_evidence_exact(root, record["evidence_ref"])
        method_id = record["method_id"]
        if (
            evidence["method_id"] != method_id
            or evidence["order_index"] != record["order_index"]
            or evidence["status"] != record["status"]
        ):
            _hold("HOLD_DEVELOPMENT_EXACT_EIGHT_BINDING", "method evidence ref drift")
        _cross_bind_method_evidence(
            evidence,
            request=request,
            plan=plan,
            budget=budget,
        )
        if evidence["status"] == METHOD_HOLD_STATUS:
            continue
        if evidence["status"] == NEURAL_COMPLETE_STATUS:
            forward_equivalents = 0
            maximum_peak_parameter_bytes = 0
            for unit in evidence["estimands"]:
                unit_forwards, unit_peak = _replay_neural_estimand(
                    root=root,
                    request=request,
                    plan=plan,
                    context=context,
                    unit=unit,
                )
                forward_equivalents += unit_forwards
                maximum_peak_parameter_bytes = max(
                    maximum_peak_parameter_bytes, unit_peak
                )
            accounting = evidence["resource_accounting"]
            if (
                accounting["fit_units"] != budget["executed_neural_fit_units"]
                or accounting["neural_forward_equivalents"]
                != forward_equivalents
                or accounting["maximum_conservative_peak_parameter_bytes"]
                != maximum_peak_parameter_bytes
                or accounting["fit_config_sha256"]
                != canonical_sha256(asdict(_neural_fit_config(plan)))
                or result_budget["neural_forward_equivalents"]
                != forward_equivalents
                or result_budget["maximum_conservative_peak_parameter_bytes"]
                != maximum_peak_parameter_bytes
            ):
                _hold(
                    "HOLD_DEVELOPMENT_EXACT_EIGHT_RESOURCE_BUDGET",
                    "neural evidence resource accounting differs from exact states",
                )
            continue
        for unit in evidence["estimands"]:
            estimand_id = unit["estimand_id"]
            train_rows = _estimand_rows(context["train_rows"], estimand_id)
            development_rows = _estimand_rows(
                context["development_rows"], estimand_id
            )
            if (
                unit["train_row_binding"]
                != {
                    "n_rows": len(train_rows),
                    "row_set_sha256": canonical_sha256(
                        [row["row_id"] for row in train_rows]
                    ),
                }
                or unit["development_row_binding"]
                != {
                    "n_rows": len(development_rows),
                    "row_set_sha256": canonical_sha256(
                        [row["row_id"] for row in development_rows]
                    ),
                }
            ):
                _hold(
                    "HOLD_DEVELOPMENT_EXACT_EIGHT_ROW_SET",
                    "scalar evidence row binding differs from exact predictor rows",
                    estimand_id=estimand_id,
                )
            writer_kwargs = _state_writer_kwargs(
                method_id=method_id,
                train_rows=train_rows,
                root=root,
                request=request,
                plan=plan,
                all_train_rows=context["train_rows"],
            )
            state_path = _ref_path(
                root, unit["fitted_state_ref"], label="scalar fitted state"
            )
            try:
                state = load_scalar_fitted_state_exact(
                    state_path,
                    unit["fitted_state_ref"]["sha256"],
                    **writer_kwargs,
                )
            except ScalarFittedStateError as exc:
                _hold(
                    "HOLD_DEVELOPMENT_EXACT_EIGHT_SCALAR_REPLAY",
                    f"scalar fitted-state exact replay failed: {exc}",
                )
            expected_prediction = _prediction_bytes(
                recompute_scalar_predictions(state, development_rows)
            )
            prediction_path = _ref_path(
                root, unit["prediction_ref"], label="scalar prediction"
            )
            try:
                actual_prediction = read_exact_regular_file(
                    prediction_path, unit["prediction_ref"]["sha256"]
                )
            except ArtifactIOError as exc:
                _hold(
                    "HOLD_DEVELOPMENT_EXACT_EIGHT_REPLAY",
                    f"scalar prediction exact replay failed: {exc}",
                )
            if actual_prediction != expected_prediction:
                _hold(
                    "HOLD_DEVELOPMENT_EXACT_EIGHT_REPLAY",
                    "scalar prediction differs from exact loaded-state recomputation",
                )
    return result


__all__ = [
    "COMPLETE_STATUS",
    "DEVELOPMENT_TARGETS_SCHEMA",
    "DevelopmentExactEightError",
    "HGB_HOLD_CODE",
    "METHOD_EVIDENCE_SCHEMA",
    "METHOD_HOLD_STATUS",
    "NEURAL_COMPLETE_STATUS",
    "PARTIAL_STATUS",
    "REQUEST_SCHEMA",
    "RESULT_SCHEMA",
    "SCALAR_COMPLETE_STATUS",
    "execute_development_exact_eight",
    "load_development_exact_eight_result_exact",
    "validate_development_exact_eight_method_evidence",
    "validate_development_exact_eight_request",
    "validate_development_exact_eight_result",
    "validate_development_targets",
]
