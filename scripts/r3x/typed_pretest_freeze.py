"""Typed, development-only pre-test freeze for the R3X ordinary route.

The historical staged-commit freeze proves only that caller-supplied bytes have
the caller-supplied hashes.  This module deliberately does more:

* the exact eight-baseline order is derived from an exact-loaded typed paired
  policy, never from a caller-owned list;
* deterministic scalar fits and the five neural seeds are distinct unit kinds,
  rather than one method/control/estimand/seed Cartesian product;
* every predictor provenance artifact is independently replayed through an
  explicit verifier registry;
* every neural checkpoint-selection envelope, checkpoint envelope, and binary
  development model state is exact-loaded and cross-bound; and
* every synthetic development control artifact supplied to this module is
  independently replayed.  This direct verifier route is a migration scaffold,
  not the authoritative pre-test control registry and not control
  qualification.  A NOT_RUN control is representable for an honest HOLD.

This module does not disclose test features or outcomes and does not authorize
their disclosure.  A successfully replayed artifact remains development-only,
``training_ready=false``, and ``downstream_unlock=false``.  Capability-separated
feature/outcome stage activation remains an external blocker.  In particular,
artifacts emitted by this module are not acceptable inputs to the feature
stage: ``scripts.r3x.pretest_control_registry`` must be exact-loaded and must
eventually report ``can_satisfy_pretest_freeze=true``.  The current authoritative
registry reports false, so this module records that unresolved blocker instead
of manufacturing a pre-test PASS.

The existing :mod:`scripts.r3x.control_summary` schema is intentionally not an
input here: it binds the ordinary internal-test row set and completed control
outcomes.  It is a post-prediction/outcome artifact, not a pre-test control
readiness authority.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Sequence

from .artifact_graph import (
    ArtifactGraphError,
    make_artifact_ref,
    read_artifact_ref,
    resolve_artifact_ref,
    validate_artifact_ref,
)
from .artifact_io import (
    ArtifactIOError,
    atomic_write_json_exclusive,
    canonical_json_bytes,
    read_exact_regular_file,
)
from .baseline_selection import (
    BaselineSelectionError,
    QUALIFIED_STATUS,
    load_development_baseline_selection_exact,
    replay_development_baseline_evidence,
)
from .development_checkpoint import (
    DevelopmentCheckpointError,
    load_development_checkpoint_exact,
    validate_development_checkpoint as validate_development_model_state,
)
from .predictor_provenance import (
    FROZEN_NEURAL_SEEDS,
    NEURAL_KIND,
    NON_DATA_AUTHORITY_HASH_KEYS,
    SCALAR_KIND,
    PredictorProvenanceError,
    PredictorVerifier,
    replay_predictor_provenance,
    validate_development_checkpoint as validate_checkpoint_envelope,
    validate_development_checkpoint_selection,
)
from .protocol import (
    REQUIRED_NEGATIVE_CONTROLS,
    SIGN_ACCURACY_GATE,
    ProtocolHoldError,
    assert_ordinary_path,
    canonical_sha256,
)
from .typed_safe_authorities import (
    MINIMUM_BASELINE_METHOD_IDS,
    PAIRED_BASELINE_POLICY,
    TypedSafeAuthorityError,
    load_typed_safe_authority_exact,
)


UNIT_REGISTRY_SCHEMA = "r3x_typed_predictor_unit_registry_v1"
UNIT_REGISTRY_ROLE = "ORDINARY_DEVELOPMENT_TYPED_PREDICTOR_UNIT_REGISTRY"
UNIT_REGISTRY_READY = "READY_FOR_STRICT_DEVELOPMENT_SYNTHETIC_REPLAY_ONLY"
UNIT_REGISTRY_HOLD = "HOLD_REQUIRED_CONTROL_NOT_RUN"

PRETEST_FREEZE_SCHEMA = "r3x_typed_pretest_freeze_v1"
PRETEST_FREEZE_ROLE = "ORDINARY_DEVELOPMENT_TYPED_PRETEST_FREEZE"
PRETEST_FREEZE_STATUS = (
    "HOLD_EXACT_DEVELOPMENT_ARTIFACTS_FROZEN_EXTERNAL_BINDINGS_REQUIRED"
)
SCIENTIFIC_SCOPE = "DEVELOPMENT_ONLY_PRETEST_ARTIFACT_FREEZE"
SHIPPED_CAPABILITY_STATUS = (
    "HOLD_PENDING_PRETEST_CONTROL_REGISTRY_DEVELOPMENT_CONTROL_EVIDENCE_"
    "EXECUTION_AND_ACTIVATION_BINDING"
)

SCALAR_FIT = "SCALAR_FIT"
NEURAL_SEED = "NEURAL_SEED"
CONTROL_RUN = "CONTROL_RUN"
CONTROL_NOT_RUN = "CONTROL_NOT_RUN"
UNIT_KINDS = frozenset({SCALAR_FIT, NEURAL_SEED, CONTROL_RUN, CONTROL_NOT_RUN})

CONTROL_VERIFIER_SCHEMA = "r3x_pretest_control_verifier_v1"
CONTROL_REPLAY_RESULT_SCHEMA = "r3x_pretest_control_replay_result_v1"
CONTROL_REPLAY_ACTIVE = "ACTIVE_FROZEN_REPLAYED"

CURRENT_CONTROL_SUMMARY_PRETEST_BLOCKER = (
    "HOLD_CONTROL_SUMMARY_V1_BINDS_INTERNAL_TEST_AND_IS_NOT_PRETEST_ADMISSIBLE"
)
DIRECT_CONTROL_VERIFIER_SCOPE = (
    "SYNTHETIC_SCAFFOLD_ONLY_NOT_PRETEST_CONTROL_QUALIFICATION"
)
PRETEST_CONTROL_REGISTRY_SCHEMA_REQUIRED = (
    "r3x_pretest_train_development_control_registry_v1"
)
REMAINING_BLOCKERS = (
    "PRETEST_CONTROL_REGISTRY_V1_REQUIRED",
    "PRETEST_CONTROL_REGISTRY_CAN_SATISFY_FREEZE_FALSE",
    "ALL_REQUIRED_DEVELOPMENT_CONTROL_EVIDENCE_NOT_ESTABLISHED",
    "EXECUTION_QUALIFICATION_REPLAY_NOT_BOUND",
    "ACTIVATION_CAPABILITY_BINDING_NOT_ESTABLISHED",
    "FEATURE_STAGE_CAPABILITY_NOT_ACTIVATED",
    "OUTCOME_STAGE_CAPABILITY_NOT_ACTIVATED",
)

FULL_PAIR_METHOD_ID = "FULL_PAIR_UNRESTRICTED_SIAMESE"
REQUIRED_NEURAL_METHOD_IDS = frozenset({FULL_PAIR_METHOD_ID})
REQUIRED_SCALAR_METHOD_IDS = frozenset(
    set(MINIMUM_BASELINE_METHOD_IDS) - set(REQUIRED_NEURAL_METHOD_IDS)
)
REQUIRED_CONTROL_IDS = tuple(REQUIRED_NEGATIVE_CONTROLS)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_METHOD_ID = re.compile(r"^[A-Z][A-Z0-9_]*$")
_REASON = re.compile(r"^[A-Z][A-Z0-9_]{2,127}$")
_PROHIBITED_PATH_PARTS = frozenset(
    {"test", "internal_test", "sealed", "restricted", "gse246381"}
)
_PROTECTED_CONTROL_BYTES = re.compile(
    rb"(?:internal[_ -]?test|test[_ -]?(?:path|hash|support|label|outcome|feature|row)|"
    rb"sealed|restricted|gse246381|ground[_ -]?truth|observed[_ -]?outcome)",
    re.IGNORECASE,
)

_EXPECTATION_KEYS = {
    "run_start_sha256",
    "run_plan_sha256",
    "development_capability_sha256",
    "train_development_outcomes_sha256",
    "non_data_authority_hashes",
    "information_set_fingerprint",
    "required_estimand_cells",
}
_REQUIRED_ESTIMAND_KEYS = {"estimand_id", "cells"}
_REQUIRED_CELL_KEYS = {"cell_id", "registered_source_group_ids_v2"}

_REGISTRY_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "scientific_scope",
    "run_start_sha256",
    "run_plan_sha256",
    "development_capability_sha256",
    "train_development_outcomes_sha256",
    "development_bindings_sha256",
    "information_set_fingerprint",
    "required_estimand_cells_sha256",
    "paired_policy_ref",
    "eligible_baseline_method_order",
    "eligible_baseline_method_set_sha256",
    "required_control_ids",
    "units",
    "control_authority_scope",
    "pretest_control_registry_schema_required",
    "pretest_control_registry_v1_replayed",
    "pretest_control_registry_can_satisfy_freeze",
    "control_qualification_passed",
    "feature_stage_acceptable",
    "sign_accuracy_gate",
    "sign_gate_change_authorized",
    "ordinary_only",
    "internal_test_feature_access",
    "internal_test_outcome_access",
    "sealed_access",
    "downstream_unlock",
    "training_ready",
    "capability_ready",
    "shipped_capability_status",
}
_BASELINE_UNIT_KEYS = {
    "unit_id",
    "unit_kind",
    "method_id",
    "seed",
    "predictor_provenance_ref",
}
_CONTROL_RUN_UNIT_KEYS = {
    "unit_id",
    "unit_kind",
    "control_id",
    "control_artifact_ref",
    "control_verifier_id",
    "control_verifier_code_sha256",
}
_CONTROL_NOT_RUN_UNIT_KEYS = {
    "unit_id",
    "unit_kind",
    "control_id",
    "reason_code",
}

_FREEZE_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "scientific_scope",
    "run_start_sha256",
    "run_plan_sha256",
    "development_capability_sha256",
    "train_development_outcomes_sha256",
    "development_bindings_sha256",
    "information_set_fingerprint",
    "required_estimand_cells_sha256",
    "paired_policy_ref",
    "typed_unit_registry_ref",
    "development_selector_ref",
    "eligible_baseline_method_order",
    "eligible_baseline_method_set_sha256",
    "selected_baseline_method_id",
    "required_control_ids",
    "strict_replay",
    "remaining_blockers",
    "control_authority_scope",
    "pretest_control_registry_schema_required",
    "pretest_control_registry_v1_replayed",
    "pretest_control_registry_can_satisfy_freeze",
    "control_qualification_passed",
    "feature_stage_acceptable",
    "sign_accuracy_gate",
    "sign_gate_change_authorized",
    "ordinary_only",
    "internal_test_feature_access",
    "internal_test_outcome_access",
    "sealed_access",
    "downstream_unlock",
    "training_ready",
    "capability_ready",
    "shipped_capability_status",
}
_STRICT_REPLAY_KEYS = {
    "typed_paired_policy_replayed",
    "selector_replayed",
    "predictor_provenance_replayed",
    "checkpoint_selection_replayed",
    "development_model_state_replayed",
    "required_controls_replayed",
    "synthetic_control_scaffold_replayed",
    "pretest_control_registry_v1_replayed",
    "control_qualification_passed",
    "baseline_method_count",
    "scalar_fit_unit_count",
    "neural_seed_unit_count",
    "control_run_unit_count",
}
_CONTROL_REPLAY_RESULT_KEYS = {
    "schema_version",
    "control_id",
    "status",
    "control_artifact_sha256",
    "run_start_sha256",
    "development_capability_sha256",
    "ordinary_only",
    "internal_test_feature_access",
    "internal_test_outcome_access",
    "sealed_access",
    "downstream_unlock",
}

_PROJECTION_TO_NONDATA = {
    "inclusion_manifest_development_projection": "inclusion_manifest_sha256",
    "context_source_crosswalk_development_projection": (
        "context_source_crosswalk_sha256"
    ),
    "source_component_split_manifest_development_projection": (
        "source_component_split_manifest_sha256"
    ),
}
_NONDATA_STATE_NAME_MAP = {
    key: key.removesuffix("_sha256") for key in NON_DATA_AUTHORITY_HASH_KEYS
}


class TypedPretestFreezeError(ProtocolHoldError):
    """Structured fail-closed error for typed pre-test freeze operations."""


def _hold(code: str, message: str, **details: object) -> None:
    raise TypedPretestFreezeError(code, message, details=details or None)


@dataclass(frozen=True)
class PretestFreezeExpectations:
    """Development-only trust inputs supplied outside the freeze artifact."""

    run_start_sha256: str
    run_plan_sha256: str
    development_capability_sha256: str
    train_development_outcomes_sha256: str
    non_data_authority_hashes: Mapping[str, str]
    information_set_fingerprint: str
    required_estimand_cells: Sequence[Mapping]


@dataclass(frozen=True)
class PretestControlVerifier:
    """Explicit replay registration for synthetic development control fixtures.

    The callable receives the checked run root, the normalized immutable
    control-artifact reference, and a fixed development-only context.  It must
    return the exact structured replay result defined by this module.

    This object is deliberately *not* the closed verifier registry defined by
    :mod:`scripts.r3x.pretest_control_registry`.  Successful direct replay here
    cannot establish control qualification, cannot satisfy the authoritative
    pre-test registry, and must never be accepted by the feature stage.
    """

    verifier_id: str
    code_sha256: str
    replay: Callable[[Path, Mapping, Mapping], Mapping]
    schema_version: str = CONTROL_VERIFIER_SCHEMA


def _strict_keys(value: object, expected: set[str], *, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        _hold("HOLD_TYPED_PRETEST_SCHEMA", f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        _hold(
            "HOLD_TYPED_PRETEST_SCHEMA",
            f"{label} keys differ from the exact schema",
            differing_keys=sorted(actual ^ expected),
        )
    return value


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        _hold("HOLD_TYPED_PRETEST_DIGEST", f"{label} must be 64 lowercase hex")
    return value


def _safe_id(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        _hold("HOLD_TYPED_PRETEST_SCHEMA", f"{label} is not a safe identifier")
    return value


def _method_id(value: object, *, label: str = "method_id") -> str:
    if not isinstance(value, str) or not _METHOD_ID.fullmatch(value):
        _hold("HOLD_TYPED_PRETEST_SCHEMA", f"{label} is not a frozen method ID")
    return value


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json_object(raw: bytes, *, label: str) -> dict:
    try:
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_json_object
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        _hold("HOLD_TYPED_PRETEST_JSON", f"{label} is not strict JSON", error=str(exc))
    if not isinstance(payload, dict):
        _hold("HOLD_TYPED_PRETEST_JSON", f"{label} must be a JSON object")
    return payload


def _checked_run_root(run_root: Path) -> Path:
    root = Path(run_root)
    try:
        assert_ordinary_path(root, purpose="typed pre-test run root")
    except ProtocolHoldError as exc:
        _hold("HOLD_TYPED_PRETEST_PATH", str(exc))
    if not root.is_dir() or root.is_symlink():
        _hold("HOLD_TYPED_PRETEST_PATH", "run root must be an existing real directory")
    return root.resolve(strict=True)


def _development_ref(value: Mapping, *, label: str) -> dict:
    try:
        reference = validate_artifact_ref(value)
    except ArtifactGraphError as exc:
        _hold("HOLD_TYPED_PRETEST_REFERENCE", f"{label}: {exc}")
    parts = [part.lower() for part in PurePosixPath(reference["relative_path"]).parts]
    if "development" not in parts:
        _hold(
            "HOLD_TYPED_PRETEST_REFERENCE",
            f"{label} must remain beneath a literal development namespace",
        )
    if any(
        part in _PROHIBITED_PATH_PARTS
        or "internal_test" in part
        or "gse246381" in part
        or "sealed" in part
        for part in parts
    ):
        _hold(
            "HOLD_TYPED_PRETEST_PROTECTED_REFERENCE",
            f"{label} names a protected stage or dataset",
        )
    return reference


def _checked_output_path(run_root: Path, path: Path, *, label: str) -> Path:
    root = _checked_run_root(run_root)
    candidate = Path(path)
    try:
        assert_ordinary_path(candidate, purpose=label)
        parent = candidate.parent.resolve(strict=True)
    except (ProtocolHoldError, FileNotFoundError, OSError) as exc:
        _hold("HOLD_TYPED_PRETEST_PATH", f"{label} path is unavailable: {exc}")
    if not parent.is_relative_to(root) or candidate.parent.is_symlink():
        _hold("HOLD_TYPED_PRETEST_PATH", f"{label} must remain beneath the run root")
    relative = parent.relative_to(root) / candidate.name
    _development_ref(
        {"relative_path": relative.as_posix(), "sha256": "0" * 64},
        label=label,
    )
    if candidate.exists() or candidate.is_symlink():
        _hold("HOLD_TYPED_PRETEST_NO_OVERWRITE", f"{label} already exists")
    return candidate


def _checked_input_path(
    run_root: Path,
    path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> Path:
    root = _checked_run_root(run_root)
    candidate = Path(path)
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(root).as_posix()
    except (FileNotFoundError, OSError, ValueError) as exc:
        _hold("HOLD_TYPED_PRETEST_PATH", f"{label} is outside the run root: {exc}")
    reference = _development_ref(
        {
            "relative_path": relative,
            "sha256": _digest(expected_sha256, label=f"{label} SHA-256"),
        },
        label=label,
    )
    try:
        checked = resolve_artifact_ref(root, reference)
    except ArtifactGraphError as exc:
        _hold("HOLD_TYPED_PRETEST_PATH", f"{label}: {exc}")
    if checked.resolve(strict=True) != resolved:
        _hold("HOLD_TYPED_PRETEST_PATH", f"{label} path changed during resolution")
    return checked


def _load_json_exact(path: Path, expected_sha256: str, *, label: str) -> tuple[bytes, dict]:
    try:
        raw = read_exact_regular_file(Path(path), _digest(expected_sha256, label=label))
    except ArtifactIOError as exc:
        _hold("HOLD_TYPED_PRETEST_EXACT_READ", f"{label}: {exc}")
    return raw, _parse_json_object(raw, label=label)


def _normalized_required_estimand_cells(values: Sequence[Mapping]) -> list[dict]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        _hold(
            "HOLD_TYPED_PRETEST_SCHEMA",
            "required_estimand_cells must be a non-empty sequence",
        )
    estimands = []
    for index, raw in enumerate(values):
        _strict_keys(raw, _REQUIRED_ESTIMAND_KEYS, label=f"required estimand {index}")
        estimand_id = _safe_id(raw.get("estimand_id"), label="estimand_id")
        raw_cells = raw.get("cells")
        if (
            isinstance(raw_cells, (str, bytes))
            or not isinstance(raw_cells, Sequence)
            or not raw_cells
        ):
            _hold("HOLD_TYPED_PRETEST_SCHEMA", "every required estimand needs cells")
        cells = []
        for cell_index, raw_cell in enumerate(raw_cells):
            _strict_keys(
                raw_cell,
                _REQUIRED_CELL_KEYS,
                label=f"required cell {cell_index}",
            )
            cell_id = _safe_id(raw_cell.get("cell_id"), label="cell_id")
            groups = raw_cell.get("registered_source_group_ids_v2")
            if (
                not isinstance(groups, list)
                or not groups
                or groups != sorted(groups)
                or len(groups) != len(set(groups))
            ):
                _hold(
                    "HOLD_TYPED_PRETEST_SCHEMA",
                    "registered source groups must be non-empty sorted and unique",
                )
            normalized_groups = [
                _safe_id(group, label="source_group_id_v2") for group in groups
            ]
            cells.append(
                {
                    "cell_id": cell_id,
                    "registered_source_group_ids_v2": normalized_groups,
                }
            )
        if cells != sorted(cells, key=lambda item: item["cell_id"]) or len(
            {cell["cell_id"] for cell in cells}
        ) != len(cells):
            _hold("HOLD_TYPED_PRETEST_SCHEMA", "required cells are not canonical")
        estimands.append({"estimand_id": estimand_id, "cells": cells})
    if estimands != sorted(estimands, key=lambda item: item["estimand_id"]) or len(
        {item["estimand_id"] for item in estimands}
    ) != len(estimands):
        _hold("HOLD_TYPED_PRETEST_SCHEMA", "required estimands are not canonical")
    return estimands


def _nondata_hashes(value: Mapping) -> dict[str, str]:
    raw = _strict_keys(
        value,
        set(NON_DATA_AUTHORITY_HASH_KEYS),
        label="non_data_authority_hashes",
    )
    return {
        name: _digest(raw[name], label=f"non_data_authority_hashes.{name}")
        for name in NON_DATA_AUTHORITY_HASH_KEYS
    }


def _expectations(value: PretestFreezeExpectations | Mapping) -> dict:
    if isinstance(value, PretestFreezeExpectations):
        raw = {
            "run_start_sha256": value.run_start_sha256,
            "run_plan_sha256": value.run_plan_sha256,
            "development_capability_sha256": value.development_capability_sha256,
            "train_development_outcomes_sha256": (
                value.train_development_outcomes_sha256
            ),
            "non_data_authority_hashes": value.non_data_authority_hashes,
            "information_set_fingerprint": value.information_set_fingerprint,
            "required_estimand_cells": value.required_estimand_cells,
        }
    else:
        raw = _strict_keys(value, _EXPECTATION_KEYS, label="freeze expectations")
    normalized = {
        "run_start_sha256": _digest(raw.get("run_start_sha256"), label="run start"),
        "run_plan_sha256": _digest(raw.get("run_plan_sha256"), label="run plan"),
        "development_capability_sha256": _digest(
            raw.get("development_capability_sha256"),
            label="development capability",
        ),
        "train_development_outcomes_sha256": _digest(
            raw.get("train_development_outcomes_sha256"),
            label="train/development outcomes",
        ),
        "non_data_authority_hashes": _nondata_hashes(
            raw.get("non_data_authority_hashes")
        ),
        "information_set_fingerprint": _digest(
            raw.get("information_set_fingerprint"),
            label="information-set fingerprint",
        ),
        "required_estimand_cells": _normalized_required_estimand_cells(
            raw.get("required_estimand_cells")
        ),
    }
    return normalized


def _development_bindings_sha256(expected: Mapping) -> str:
    return canonical_sha256(dict(expected))


def _load_paired_policy(
    run_root: Path,
    paired_policy_ref: Mapping,
    expected: Mapping,
) -> tuple[dict, dict, tuple[str, ...]]:
    root = _checked_run_root(run_root)
    reference = _development_ref(paired_policy_ref, label="typed paired policy")
    try:
        path = resolve_artifact_ref(root, reference)
        policy = load_typed_safe_authority_exact(
            root,
            path,
            reference["sha256"],
            expected_authority_name=PAIRED_BASELINE_POLICY,
        )
    except (ArtifactGraphError, TypedSafeAuthorityError) as exc:
        _hold("HOLD_TYPED_PRETEST_PAIRED_POLICY", str(exc))
    order = tuple(policy["eligible_baseline_method_ids"])
    if set(order) != set(MINIMUM_BASELINE_METHOD_IDS) or len(order) != len(
        MINIMUM_BASELINE_METHOD_IDS
    ):
        _hold(
            "HOLD_TYPED_PRETEST_BASELINE_SET",
            "typed paired policy does not provide the exact supported eight-baseline set",
        )
    if policy["required_baseline_method_ids"] != list(order):
        _hold(
            "HOLD_TYPED_PRETEST_BASELINE_SET",
            "required and eligible baseline order differ",
        )
    if policy["train_development_outcomes_sha256"] != expected[
        "train_development_outcomes_sha256"
    ]:
        _hold(
            "HOLD_TYPED_PRETEST_BINDING",
            "paired policy binds different train/development outcomes",
        )
    nondata = expected["non_data_authority_hashes"]
    if nondata["paired_strongest_baseline_comparison_policy_sha256"] != reference[
        "sha256"
    ]:
        _hold(
            "HOLD_TYPED_PRETEST_BINDING",
            "paired policy reference differs from development authority binding",
        )
    for projection_key, nondata_key in _PROJECTION_TO_NONDATA.items():
        if policy["development_projection_hashes"][projection_key] != nondata[
            nondata_key
        ]:
            _hold(
                "HOLD_TYPED_PRETEST_BINDING",
                f"paired policy development projection differs: {projection_key}",
            )
    return policy, reference, order


def _predictor_ref_by_method(payload: Mapping) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for unit in payload["units"]:
        if unit["unit_kind"] not in {SCALAR_FIT, NEURAL_SEED}:
            continue
        method = unit["method_id"]
        reference = unit["predictor_provenance_ref"]
        if method in result and result[method] != reference:
            _hold(
                "HOLD_TYPED_PRETEST_UNIT_REGISTRY",
                f"{method} uses more than one predictor provenance artifact",
            )
        result[method] = reference
    return result


def _normalize_baseline_unit(value: Mapping) -> dict:
    raw = _strict_keys(value, _BASELINE_UNIT_KEYS, label="baseline unit")
    kind = raw.get("unit_kind")
    if kind not in {SCALAR_FIT, NEURAL_SEED}:
        _hold("HOLD_TYPED_PRETEST_UNIT_KIND", "baseline unit kind is invalid")
    method = _method_id(raw.get("method_id"))
    seed = raw.get("seed")
    if kind == SCALAR_FIT:
        if seed is not None:
            _hold("HOLD_TYPED_PRETEST_UNIT_KIND", "scalar fit cannot carry a seed")
        expected_id = f"BASELINE:{method}:SCALAR_FIT"
    else:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in FROZEN_NEURAL_SEEDS:
            _hold("HOLD_TYPED_PRETEST_UNIT_KIND", "neural seed is not frozen")
        expected_id = f"BASELINE:{method}:NEURAL_SEED:{seed}"
    if raw.get("unit_id") != expected_id:
        _hold("HOLD_TYPED_PRETEST_UNIT_ID", "baseline unit ID is not deterministic")
    return {
        "unit_id": expected_id,
        "unit_kind": kind,
        "method_id": method,
        "seed": seed,
        "predictor_provenance_ref": _development_ref(
            raw.get("predictor_provenance_ref"),
            label=f"{method} predictor provenance",
        ),
    }


def _normalize_control_unit(value: Mapping) -> dict:
    if not isinstance(value, Mapping):
        _hold("HOLD_TYPED_PRETEST_SCHEMA", "control unit must be an object")
    kind = value.get("unit_kind")
    if kind == CONTROL_RUN:
        raw = _strict_keys(value, _CONTROL_RUN_UNIT_KEYS, label="control run unit")
        control_id = _safe_id(raw.get("control_id"), label="control_id")
        expected_id = f"CONTROL:{control_id}:RUN"
        if raw.get("unit_id") != expected_id:
            _hold("HOLD_TYPED_PRETEST_UNIT_ID", "control unit ID is not deterministic")
        return {
            "unit_id": expected_id,
            "unit_kind": CONTROL_RUN,
            "control_id": control_id,
            "control_artifact_ref": _development_ref(
                raw.get("control_artifact_ref"),
                label=f"{control_id} control artifact",
            ),
            "control_verifier_id": _safe_id(
                raw.get("control_verifier_id"), label="control_verifier_id"
            ),
            "control_verifier_code_sha256": _digest(
                raw.get("control_verifier_code_sha256"),
                label="control verifier code",
            ),
        }
    if kind == CONTROL_NOT_RUN:
        raw = _strict_keys(
            value, _CONTROL_NOT_RUN_UNIT_KEYS, label="control NOT_RUN unit"
        )
        control_id = _safe_id(raw.get("control_id"), label="control_id")
        reason = raw.get("reason_code")
        if not isinstance(reason, str) or not _REASON.fullmatch(reason):
            _hold(
                "HOLD_TYPED_PRETEST_CONTROL_NOT_RUN",
                "CONTROL_NOT_RUN requires a stable reason_code",
            )
        expected_id = f"CONTROL:{control_id}:NOT_RUN"
        if raw.get("unit_id") != expected_id:
            _hold("HOLD_TYPED_PRETEST_UNIT_ID", "control unit ID is not deterministic")
        return {
            "unit_id": expected_id,
            "unit_kind": CONTROL_NOT_RUN,
            "control_id": control_id,
            "reason_code": reason,
        }
    _hold("HOLD_TYPED_PRETEST_UNIT_KIND", "control unit kind is invalid")


def _boundaries(payload: Mapping, *, label: str) -> None:
    exact = {
        "control_authority_scope": DIRECT_CONTROL_VERIFIER_SCOPE,
        "pretest_control_registry_schema_required": (
            PRETEST_CONTROL_REGISTRY_SCHEMA_REQUIRED
        ),
        "pretest_control_registry_v1_replayed": False,
        "pretest_control_registry_can_satisfy_freeze": False,
        "control_qualification_passed": False,
        "feature_stage_acceptable": False,
        "sign_accuracy_gate": SIGN_ACCURACY_GATE,
        "sign_gate_change_authorized": False,
        "ordinary_only": True,
        "internal_test_feature_access": False,
        "internal_test_outcome_access": False,
        "sealed_access": False,
        "downstream_unlock": False,
        "training_ready": False,
        "capability_ready": False,
    }
    for field, expected in exact.items():
        value = payload.get(field)
        same = value is expected if isinstance(expected, bool) else (
            type(value) is type(expected) and value == expected
        )
        if not same:
            _hold(
                "HOLD_TYPED_PRETEST_BOUNDARY",
                f"{label} boundary changed: {field}",
            )


def validate_typed_unit_registry(payload: Mapping) -> dict:
    """Validate the exact typed registry schema without trusting its refs."""

    raw = _strict_keys(payload, _REGISTRY_KEYS, label="typed unit registry")
    if raw.get("schema_version") != UNIT_REGISTRY_SCHEMA:
        _hold("HOLD_TYPED_PRETEST_SCHEMA", "unsupported typed unit registry schema")
    if raw.get("artifact_role") != UNIT_REGISTRY_ROLE:
        _hold("HOLD_TYPED_PRETEST_SCHEMA", "typed unit registry role changed")
    if raw.get("scientific_scope") != SCIENTIFIC_SCOPE:
        _hold("HOLD_TYPED_PRETEST_BOUNDARY", "typed unit registry scope escalated")
    for field in (
        "run_start_sha256",
        "run_plan_sha256",
        "development_capability_sha256",
        "train_development_outcomes_sha256",
        "development_bindings_sha256",
        "information_set_fingerprint",
        "required_estimand_cells_sha256",
    ):
        _digest(raw.get(field), label=field)
    policy_ref = _development_ref(raw.get("paired_policy_ref"), label="paired policy")
    order_value = raw.get("eligible_baseline_method_order")
    if not isinstance(order_value, list):
        _hold("HOLD_TYPED_PRETEST_BASELINE_SET", "baseline order must be a list")
    order = [_method_id(method) for method in order_value]
    if len(order) != len(set(order)) or set(order) != set(MINIMUM_BASELINE_METHOD_IDS):
        _hold(
            "HOLD_TYPED_PRETEST_BASELINE_SET",
            "unit registry must bind the exact eight-baseline set",
        )
    if raw.get("eligible_baseline_method_set_sha256") != canonical_sha256(
        sorted(order)
    ):
        _hold("HOLD_TYPED_PRETEST_BASELINE_SET", "baseline set digest is stale")
    if raw.get("required_control_ids") != list(REQUIRED_CONTROL_IDS):
        _hold(
            "HOLD_TYPED_PRETEST_CONTROL_SET",
            "unit registry does not bind all required controls in frozen order",
        )
    raw_units = raw.get("units")
    if isinstance(raw_units, (str, bytes)) or not isinstance(raw_units, Sequence):
        _hold("HOLD_TYPED_PRETEST_UNIT_REGISTRY", "units must be a sequence")
    units = []
    for item in raw_units:
        if not isinstance(item, Mapping):
            _hold("HOLD_TYPED_PRETEST_UNIT_REGISTRY", "unit must be an object")
        kind = item.get("unit_kind")
        if kind in {SCALAR_FIT, NEURAL_SEED}:
            units.append(_normalize_baseline_unit(item))
        elif kind in {CONTROL_RUN, CONTROL_NOT_RUN}:
            units.append(_normalize_control_unit(item))
        else:
            _hold("HOLD_TYPED_PRETEST_UNIT_KIND", f"unsupported unit kind: {kind}")

    expected_units = []
    for method in order:
        method_units = [
            unit
            for unit in units
            if unit["unit_kind"] in {SCALAR_FIT, NEURAL_SEED}
            and unit["method_id"] == method
        ]
        if method in REQUIRED_NEURAL_METHOD_IDS:
            if [unit["seed"] for unit in method_units] != list(FROZEN_NEURAL_SEEDS) or any(
                unit["unit_kind"] != NEURAL_SEED for unit in method_units
            ):
                _hold(
                    "HOLD_TYPED_PRETEST_UNIT_REGISTRY",
                    f"{method} must have exactly the five frozen neural seed units",
                )
        else:
            if len(method_units) != 1 or method_units[0]["unit_kind"] != SCALAR_FIT:
                _hold(
                    "HOLD_TYPED_PRETEST_UNIT_REGISTRY",
                    f"{method} must have exactly one deterministic scalar-fit unit",
                )
        expected_units.extend(method_units)

    control_units = [
        unit for unit in units if unit["unit_kind"] in {CONTROL_RUN, CONTROL_NOT_RUN}
    ]
    if [unit["control_id"] for unit in control_units] != list(REQUIRED_CONTROL_IDS):
        _hold(
            "HOLD_TYPED_PRETEST_CONTROL_SET",
            "control units must cover the exact frozen control order",
        )
    expected_units.extend(control_units)
    if units != expected_units:
        _hold("HOLD_TYPED_PRETEST_UNIT_REGISTRY", "unit order is not canonical")
    unit_ids = [unit["unit_id"] for unit in units]
    if len(unit_ids) != len(set(unit_ids)):
        _hold("HOLD_TYPED_PRETEST_UNIT_REGISTRY", "unit IDs are not unique")
    all_controls_run = all(unit["unit_kind"] == CONTROL_RUN for unit in control_units)
    expected_status = UNIT_REGISTRY_READY if all_controls_run else UNIT_REGISTRY_HOLD
    if raw.get("status") != expected_status:
        _hold("HOLD_TYPED_PRETEST_UNIT_REGISTRY", "registry status is not recomputable")
    _boundaries(raw, label="typed unit registry")
    if raw.get("shipped_capability_status") != SHIPPED_CAPABILITY_STATUS:
        _hold(
            "HOLD_TYPED_PRETEST_BOUNDARY",
            "typed unit registry shipped capability status changed",
        )
    return {
        **dict(raw),
        "paired_policy_ref": policy_ref,
        "eligible_baseline_method_order": order,
        "units": units,
    }


def canonical_typed_unit_registry_bytes(payload: Mapping) -> bytes:
    try:
        return canonical_json_bytes(validate_typed_unit_registry(payload))
    except ArtifactIOError as exc:
        _hold("HOLD_TYPED_PRETEST_JSON", str(exc))


def _parse_canonical_json_ref(
    run_root: Path,
    reference: Mapping,
    *,
    label: str,
    validator: Callable[[Mapping], dict],
) -> tuple[bytes, dict]:
    try:
        raw = read_artifact_ref(run_root, reference)
    except ArtifactGraphError as exc:
        _hold("HOLD_TYPED_PRETEST_EXACT_READ", f"{label}: {exc}")
    parsed = _parse_json_object(raw, label=label)
    try:
        normalized = validator(parsed)
    except PredictorProvenanceError as exc:
        _hold("HOLD_TYPED_PRETEST_CHECKPOINT", f"{label}: {exc}")
    try:
        canonical = canonical_json_bytes(normalized)
    except ArtifactIOError as exc:
        _hold("HOLD_TYPED_PRETEST_JSON", f"{label}: {exc}")
    if canonical != raw:
        _hold("HOLD_TYPED_PRETEST_JSON", f"{label} bytes are not canonical")
    return raw, normalized


def _state_authority_hashes(expected: Mapping) -> tuple[dict, dict]:
    nondata = expected["non_data_authority_hashes"]
    return (
        {"train_development_outcomes": expected["train_development_outcomes_sha256"]},
        {
            _NONDATA_STATE_NAME_MAP[name]: nondata[name]
            for name in NON_DATA_AUTHORITY_HASH_KEYS
        },
    )


def _replay_neural_model_states(
    run_root: Path,
    provenance: Mapping,
    expected: Mapping,
) -> int:
    data_hashes, nondata_hashes = _state_authority_hashes(expected)
    state_count = 0
    for estimand in provenance["estimands"]:
        for cell in estimand["cells"]:
            artifacts = cell["predictor_artifacts"]
            if artifacts["kind"] != NEURAL_KIND:
                _hold(
                    "HOLD_TYPED_PRETEST_UNIT_KIND",
                    "neural method contains a non-neural cell",
                )
            for seed_artifact in artifacts["seed_artifacts"]:
                selection_ref = _development_ref(
                    seed_artifact["checkpoint_selection_ref"],
                    label="development checkpoint selection",
                )
                _, selection = _parse_canonical_json_ref(
                    run_root,
                    selection_ref,
                    label="development checkpoint selection",
                    validator=validate_development_checkpoint_selection,
                )
                checkpoint_ref = _development_ref(
                    seed_artifact["checkpoint_ref"],
                    label="development checkpoint envelope",
                )
                _, checkpoint = _parse_canonical_json_ref(
                    run_root,
                    checkpoint_ref,
                    label="development checkpoint envelope",
                    validator=validate_checkpoint_envelope,
                )
                if checkpoint["checkpoint_selection_ref"] != selection_ref:
                    _hold(
                        "HOLD_TYPED_PRETEST_CHECKPOINT",
                        "checkpoint envelope binds a different selection artifact",
                    )
                state_ref = _development_ref(
                    checkpoint["state_ref"], label="development model state"
                )
                try:
                    state_path = resolve_artifact_ref(run_root, state_ref)
                    state = load_development_checkpoint_exact(
                        state_path, state_ref["sha256"]
                    )
                    validated_state = validate_development_model_state(
                        state,
                        expected_architecture_id=checkpoint["architecture_id"],
                        expected_development_data_authority_hashes=data_hashes,
                        expected_development_nondata_authority_hashes=nondata_hashes,
                        expected_run_start_sha256=expected["run_start_sha256"],
                        expected_run_plan_sha256=expected["run_plan_sha256"],
                        expected_development_activation_sha256=expected[
                            "development_capability_sha256"
                        ],
                        expected_method_id=provenance["method_id"],
                        expected_training_variant_id="PRIMARY",
                        expected_seed=seed_artifact["seed"],
                        expected_estimand_binding_sha256=estimand[
                            "estimand_binding_sha256"
                        ],
                        expected_information_set_fingerprint=provenance[
                            "information_set_fingerprint"
                        ],
                        expected_partition_row_set_sha256={
                            "train": cell["train_row_set_sha256"],
                            "development": cell["development_row_set_sha256"],
                        },
                    )
                except (ArtifactGraphError, DevelopmentCheckpointError) as exc:
                    _hold(
                        "HOLD_TYPED_PRETEST_MODEL_STATE",
                        f"development model state replay failed: {exc}",
                    )
                if validated_state["checkpoint_selection_sha256"] != selection_ref[
                    "sha256"
                ]:
                    _hold(
                        "HOLD_TYPED_PRETEST_MODEL_STATE",
                        "model state binds a different checkpoint selection",
                    )
                if validated_state["selected_epoch"] != selection["selected_epoch"]:
                    _hold(
                        "HOLD_TYPED_PRETEST_MODEL_STATE",
                        "model state selected epoch differs from selection envelope",
                    )
                if validated_state["training_config_sha256"] != provenance[
                    "predictor_config_sha256"
                ]:
                    _hold(
                        "HOLD_TYPED_PRETEST_MODEL_STATE",
                        "model state training config differs from predictor provenance",
                    )
                if validated_state["information_set"] != provenance["information_set"]:
                    _hold(
                        "HOLD_TYPED_PRETEST_MODEL_STATE",
                        "model state information set differs from predictor provenance",
                    )
                state_count += 1
    if state_count < len(FROZEN_NEURAL_SEEDS):
        _hold(
            "HOLD_TYPED_PRETEST_MODEL_STATE",
            "neural provenance does not bind a complete five-seed state chain",
        )
    return state_count


def _replay_predictor(
    run_root: Path,
    reference: Mapping,
    *,
    method_id: str,
    expected: Mapping,
    predictor_verifiers: Mapping[str, PredictorVerifier] | None,
) -> tuple[dict, str, int]:
    if not isinstance(predictor_verifiers, Mapping) or not predictor_verifiers:
        _hold(
            "HOLD_TYPED_PRETEST_PREDICTOR_VERIFIER",
            "an explicit predictor verifier registry is required",
        )
    try:
        provenance = replay_predictor_provenance(
            run_root,
            reference,
            verifiers=predictor_verifiers,
            expected_run_start_sha256=expected["run_start_sha256"],
            expected_activation_manifest_sha256=expected[
                "development_capability_sha256"
            ],
            expected_train_development_outcomes_sha256=expected[
                "train_development_outcomes_sha256"
            ],
            expected_non_data_authority_hashes=expected[
                "non_data_authority_hashes"
            ],
            expected_method_id=method_id,
            expected_information_set_fingerprint=expected[
                "information_set_fingerprint"
            ],
        )
    except PredictorProvenanceError as exc:
        _hold(
            "HOLD_TYPED_PRETEST_PREDICTOR_PROVENANCE",
            f"{method_id} predictor provenance failed exact replay: {exc}",
        )
    kinds = {
        cell["predictor_artifacts"]["kind"]
        for estimand in provenance["estimands"]
        for cell in estimand["cells"]
    }
    if len(kinds) != 1:
        _hold(
            "HOLD_TYPED_PRETEST_UNIT_KIND",
            f"{method_id} mixes predictor kinds across cells",
        )
    kind = next(iter(kinds))
    if method_id in REQUIRED_NEURAL_METHOD_IDS and kind != NEURAL_KIND:
        _hold(
            "HOLD_TYPED_PRETEST_UNIT_KIND",
            f"{method_id} must use the fixed five-seed neural provenance",
        )
    if method_id in REQUIRED_SCALAR_METHOD_IDS and kind != SCALAR_KIND:
        _hold(
            "HOLD_TYPED_PRETEST_UNIT_KIND",
            f"{method_id} must use one deterministic scalar fit",
        )
    state_count = (
        _replay_neural_model_states(run_root, provenance, expected)
        if kind == NEURAL_KIND
        else 0
    )
    return provenance, kind, state_count


def _registered_control_verifier(
    unit: Mapping,
    registry: Mapping[str, PretestControlVerifier] | None,
) -> PretestControlVerifier:
    if not isinstance(registry, Mapping) or not registry:
        _hold(
            "HOLD_TYPED_PRETEST_CONTROL_VERIFIER",
            "an explicit pre-test control verifier registry is required",
        )
    verifier = registry.get(unit["control_verifier_id"])
    if not isinstance(verifier, PretestControlVerifier):
        _hold(
            "HOLD_TYPED_PRETEST_CONTROL_VERIFIER",
            f"missing registered verifier for {unit['control_id']}",
        )
    if verifier.verifier_id != unit["control_verifier_id"]:
        _hold("HOLD_TYPED_PRETEST_CONTROL_VERIFIER", "control verifier ID drifted")
    if verifier.schema_version != CONTROL_VERIFIER_SCHEMA:
        _hold("HOLD_TYPED_PRETEST_CONTROL_VERIFIER", "control verifier schema drifted")
    if verifier.code_sha256 != unit["control_verifier_code_sha256"]:
        _hold("HOLD_TYPED_PRETEST_CONTROL_VERIFIER", "control verifier code hash drifted")
    if not callable(verifier.replay):
        _hold("HOLD_TYPED_PRETEST_CONTROL_VERIFIER", "control verifier is not callable")
    return verifier


def _validate_control_replay_result(
    result: Mapping,
    *,
    unit: Mapping,
    expected: Mapping,
) -> dict:
    raw = _strict_keys(
        result, _CONTROL_REPLAY_RESULT_KEYS, label="control replay result"
    )
    exact = {
        "schema_version": CONTROL_REPLAY_RESULT_SCHEMA,
        "control_id": unit["control_id"],
        "status": CONTROL_REPLAY_ACTIVE,
        "control_artifact_sha256": unit["control_artifact_ref"]["sha256"],
        "run_start_sha256": expected["run_start_sha256"],
        "development_capability_sha256": expected[
            "development_capability_sha256"
        ],
        "ordinary_only": True,
        "internal_test_feature_access": False,
        "internal_test_outcome_access": False,
        "sealed_access": False,
        "downstream_unlock": False,
    }
    if dict(raw) != exact:
        _hold(
            "HOLD_TYPED_PRETEST_CONTROL_INACTIVE",
            f"{unit['control_id']} verifier did not return the exact active replay result",
        )
    return exact


def _replay_control(
    run_root: Path,
    unit: Mapping,
    *,
    expected: Mapping,
    control_verifiers: Mapping[str, PretestControlVerifier] | None,
) -> dict:
    verifier = _registered_control_verifier(unit, control_verifiers)
    try:
        raw = read_artifact_ref(run_root, unit["control_artifact_ref"])
    except ArtifactGraphError as exc:
        _hold(
            "HOLD_TYPED_PRETEST_CONTROL_ARTIFACT",
            f"{unit['control_id']} control artifact failed exact read: {exc}",
        )
    if _PROTECTED_CONTROL_BYTES.search(raw):
        _hold(
            "HOLD_TYPED_PRETEST_CONTROL_PROTECTED_CONTENT",
            f"{unit['control_id']} control artifact contains test/outcome/support content",
            blocker=CURRENT_CONTROL_SUMMARY_PRETEST_BLOCKER,
        )
    context = {
        "run_start_sha256": expected["run_start_sha256"],
        "development_capability_sha256": expected[
            "development_capability_sha256"
        ],
        "train_development_outcomes_sha256": expected[
            "train_development_outcomes_sha256"
        ],
        "information_set_fingerprint": expected["information_set_fingerprint"],
        "control_artifact_sha256": unit["control_artifact_ref"]["sha256"],
        "control_artifact_bytes_sha256": hashlib.sha256(raw).hexdigest(),
    }
    try:
        result = verifier.replay(
            _checked_run_root(run_root),
            dict(unit["control_artifact_ref"]),
            context,
        )
    except TypedPretestFreezeError:
        raise
    except Exception as exc:
        _hold(
            "HOLD_TYPED_PRETEST_CONTROL_VERIFIER",
            f"{unit['control_id']} verifier failed closed: {type(exc).__name__}: {exc}",
        )
    if isinstance(result, bool) or not isinstance(result, Mapping):
        _hold(
            "HOLD_TYPED_PRETEST_CONTROL_VERIFIER",
            "control verifier must return a structured result, not a boolean",
        )
    return _validate_control_replay_result(result, unit=unit, expected=expected)


def _baseline_units(
    order: Sequence[str],
    refs: Mapping[str, Mapping],
    *,
    run_root: Path,
    expected: Mapping,
    predictor_verifiers: Mapping[str, PredictorVerifier] | None,
) -> list[dict]:
    if not isinstance(refs, Mapping) or set(refs) != set(order):
        _hold(
            "HOLD_TYPED_PRETEST_BASELINE_SET",
            "predictor provenance refs must cover the exact paired-policy baseline set",
        )
    units = []
    seen_refs: set[tuple[str, str]] = set()
    for method in order:
        reference = _development_ref(
            refs[method], label=f"{method} predictor provenance"
        )
        ref_identity = (reference["relative_path"], reference["sha256"])
        if ref_identity in seen_refs:
            _hold(
                "HOLD_TYPED_PRETEST_REFERENCE_ALIAS",
                "different baseline methods cannot share one provenance artifact",
            )
        seen_refs.add(ref_identity)
        _, kind, _ = _replay_predictor(
            run_root,
            reference,
            method_id=method,
            expected=expected,
            predictor_verifiers=predictor_verifiers,
        )
        if kind == SCALAR_KIND:
            units.append(
                {
                    "unit_id": f"BASELINE:{method}:SCALAR_FIT",
                    "unit_kind": SCALAR_FIT,
                    "method_id": method,
                    "seed": None,
                    "predictor_provenance_ref": reference,
                }
            )
        else:
            units.extend(
                {
                    "unit_id": f"BASELINE:{method}:NEURAL_SEED:{seed}",
                    "unit_kind": NEURAL_SEED,
                    "method_id": method,
                    "seed": seed,
                    "predictor_provenance_ref": reference,
                }
                for seed in FROZEN_NEURAL_SEEDS
            )
    return units


def _control_units(
    specifications: Mapping[str, Mapping],
    *,
    run_root: Path,
    expected: Mapping,
    control_verifiers: Mapping[str, PretestControlVerifier] | None,
) -> list[dict]:
    if not isinstance(specifications, Mapping) or set(specifications) != set(
        REQUIRED_CONTROL_IDS
    ):
        _hold(
            "HOLD_TYPED_PRETEST_CONTROL_SET",
            "control specifications must cover all required controls exactly",
        )
    units = []
    seen_refs: set[tuple[str, str]] = set()
    for control_id in REQUIRED_CONTROL_IDS:
        spec = specifications[control_id]
        if not isinstance(spec, Mapping):
            _hold("HOLD_TYPED_PRETEST_CONTROL_SET", "control specification is not an object")
        status = spec.get("status")
        if status == "RUN":
            _strict_keys(
                spec,
                {"status", "control_artifact_ref", "control_verifier_id"},
                label=f"{control_id} control specification",
            )
            verifier_id = _safe_id(
                spec.get("control_verifier_id"), label="control_verifier_id"
            )
            verifier = None if not isinstance(control_verifiers, Mapping) else (
                control_verifiers.get(verifier_id)
            )
            if not isinstance(verifier, PretestControlVerifier):
                _hold(
                    "HOLD_TYPED_PRETEST_CONTROL_VERIFIER",
                    f"missing registered verifier for {control_id}",
                )
            reference = _development_ref(
                spec.get("control_artifact_ref"), label=f"{control_id} control artifact"
            )
            identity = (reference["relative_path"], reference["sha256"])
            if identity in seen_refs:
                _hold(
                    "HOLD_TYPED_PRETEST_REFERENCE_ALIAS",
                    "required controls cannot share one artifact",
                )
            seen_refs.add(identity)
            unit = {
                "unit_id": f"CONTROL:{control_id}:RUN",
                "unit_kind": CONTROL_RUN,
                "control_id": control_id,
                "control_artifact_ref": reference,
                "control_verifier_id": verifier_id,
                "control_verifier_code_sha256": _digest(
                    verifier.code_sha256, label="control verifier code"
                ),
            }
            _replay_control(
                run_root,
                unit,
                expected=expected,
                control_verifiers=control_verifiers,
            )
            units.append(unit)
            continue
        if status == "NOT_RUN":
            _strict_keys(
                spec,
                {"status", "reason_code"},
                label=f"{control_id} NOT_RUN specification",
            )
            reason = spec.get("reason_code")
            if not isinstance(reason, str) or not _REASON.fullmatch(reason):
                _hold(
                    "HOLD_TYPED_PRETEST_CONTROL_NOT_RUN",
                    f"{control_id} NOT_RUN reason is invalid",
                )
            units.append(
                {
                    "unit_id": f"CONTROL:{control_id}:NOT_RUN",
                    "unit_kind": CONTROL_NOT_RUN,
                    "control_id": control_id,
                    "reason_code": reason,
                }
            )
            continue
        _hold(
            "HOLD_TYPED_PRETEST_CONTROL_INACTIVE",
            f"{control_id} must be explicitly RUN or NOT_RUN",
        )
    return units


def _registry_payload(
    *,
    expected: Mapping,
    paired_policy_ref: Mapping,
    order: Sequence[str],
    units: Sequence[Mapping],
) -> dict:
    control_units = [
        unit for unit in units if unit["unit_kind"] in {CONTROL_RUN, CONTROL_NOT_RUN}
    ]
    all_controls_run = all(unit["unit_kind"] == CONTROL_RUN for unit in control_units)
    return validate_typed_unit_registry(
        {
            "schema_version": UNIT_REGISTRY_SCHEMA,
            "artifact_role": UNIT_REGISTRY_ROLE,
            "status": UNIT_REGISTRY_READY if all_controls_run else UNIT_REGISTRY_HOLD,
            "scientific_scope": SCIENTIFIC_SCOPE,
            "run_start_sha256": expected["run_start_sha256"],
            "run_plan_sha256": expected["run_plan_sha256"],
            "development_capability_sha256": expected[
                "development_capability_sha256"
            ],
            "train_development_outcomes_sha256": expected[
                "train_development_outcomes_sha256"
            ],
            "development_bindings_sha256": _development_bindings_sha256(expected),
            "information_set_fingerprint": expected["information_set_fingerprint"],
            "required_estimand_cells_sha256": canonical_sha256(
                expected["required_estimand_cells"]
            ),
            "paired_policy_ref": dict(paired_policy_ref),
            "eligible_baseline_method_order": list(order),
            "eligible_baseline_method_set_sha256": canonical_sha256(sorted(order)),
            "required_control_ids": list(REQUIRED_CONTROL_IDS),
            "units": [dict(unit) for unit in units],
            "control_authority_scope": DIRECT_CONTROL_VERIFIER_SCOPE,
            "pretest_control_registry_schema_required": (
                PRETEST_CONTROL_REGISTRY_SCHEMA_REQUIRED
            ),
            "pretest_control_registry_v1_replayed": False,
            "pretest_control_registry_can_satisfy_freeze": False,
            "control_qualification_passed": False,
            "feature_stage_acceptable": False,
            "sign_accuracy_gate": SIGN_ACCURACY_GATE,
            "sign_gate_change_authorized": False,
            "ordinary_only": True,
            "internal_test_feature_access": False,
            "internal_test_outcome_access": False,
            "sealed_access": False,
            "downstream_unlock": False,
            "training_ready": False,
            "capability_ready": False,
            "shipped_capability_status": SHIPPED_CAPABILITY_STATUS,
        }
    )


def build_typed_unit_registry(
    run_root: Path,
    *,
    paired_policy_ref: Mapping,
    baseline_predictor_provenance_refs: Mapping[str, Mapping],
    control_specifications: Mapping[str, Mapping],
    expectations: PretestFreezeExpectations | Mapping,
    predictor_verifiers: Mapping[str, PredictorVerifier] | None,
    control_verifiers: Mapping[str, PretestControlVerifier] | None,
) -> dict:
    """Build a development replay registry, never a control-qualification PASS.

    ``control_verifiers`` is a synthetic migration scaffold.  This function
    does not load ``PRETEST_CONTROL_REGISTRY_SCHEMA_REQUIRED`` and consequently
    fixes ``pretest_control_registry_can_satisfy_freeze`` and
    ``feature_stage_acceptable`` to false.
    """

    root = _checked_run_root(run_root)
    expected = _expectations(expectations)
    _, policy_ref, order = _load_paired_policy(root, paired_policy_ref, expected)
    baseline_units = _baseline_units(
        order,
        baseline_predictor_provenance_refs,
        run_root=root,
        expected=expected,
        predictor_verifiers=predictor_verifiers,
    )
    controls = _control_units(
        control_specifications,
        run_root=root,
        expected=expected,
        control_verifiers=control_verifiers,
    )
    return _registry_payload(
        expected=expected,
        paired_policy_ref=policy_ref,
        order=order,
        units=[*baseline_units, *controls],
    )


def _control_specs_from_registry(payload: Mapping) -> dict[str, dict]:
    specs = {}
    for unit in payload["units"]:
        if unit["unit_kind"] == CONTROL_RUN:
            specs[unit["control_id"]] = {
                "status": "RUN",
                "control_artifact_ref": unit["control_artifact_ref"],
                "control_verifier_id": unit["control_verifier_id"],
            }
        elif unit["unit_kind"] == CONTROL_NOT_RUN:
            specs[unit["control_id"]] = {
                "status": "NOT_RUN",
                "reason_code": unit["reason_code"],
            }
    return specs


def _rebuild_typed_unit_registry(
    run_root: Path,
    payload: Mapping,
    *,
    expectations: PretestFreezeExpectations | Mapping,
    predictor_verifiers: Mapping[str, PredictorVerifier] | None,
    control_verifiers: Mapping[str, PretestControlVerifier] | None,
) -> dict:
    validated = validate_typed_unit_registry(payload)
    expected = _expectations(expectations)
    refs = _predictor_ref_by_method(validated)
    rebuilt = build_typed_unit_registry(
        run_root,
        paired_policy_ref=validated["paired_policy_ref"],
        baseline_predictor_provenance_refs=refs,
        control_specifications=_control_specs_from_registry(validated),
        expectations=expected,
        predictor_verifiers=predictor_verifiers,
        control_verifiers=control_verifiers,
    )
    if rebuilt != validated:
        _hold(
            "HOLD_TYPED_PRETEST_REPLAY_DRIFT",
            "typed unit registry changed during strict replay",
        )
    return validated


def write_typed_unit_registry_exclusive(
    run_root: Path,
    path: Path,
    payload: Mapping,
    *,
    expectations: PretestFreezeExpectations | Mapping,
    predictor_verifiers: Mapping[str, PredictorVerifier] | None,
    control_verifiers: Mapping[str, PretestControlVerifier] | None,
) -> str:
    candidate = _checked_output_path(run_root, path, label="typed unit registry output")
    replayed = _rebuild_typed_unit_registry(
        run_root,
        payload,
        expectations=expectations,
        predictor_verifiers=predictor_verifiers,
        control_verifiers=control_verifiers,
    )
    try:
        return atomic_write_json_exclusive(
            candidate, replayed, validator=validate_typed_unit_registry
        )
    except ArtifactIOError as exc:
        _hold("HOLD_TYPED_PRETEST_WRITE", str(exc))


def load_typed_unit_registry_exact(
    run_root: Path,
    path: Path,
    expected_sha256: str,
    *,
    expectations: PretestFreezeExpectations | Mapping,
    predictor_verifiers: Mapping[str, PredictorVerifier] | None,
    control_verifiers: Mapping[str, PretestControlVerifier] | None,
) -> dict:
    root = _checked_run_root(run_root)
    candidate = _checked_input_path(
        root,
        path,
        expected_sha256,
        label="typed unit registry",
    )
    raw, parsed = _load_json_exact(
        candidate, expected_sha256, label="typed unit registry"
    )
    validated = validate_typed_unit_registry(parsed)
    if canonical_typed_unit_registry_bytes(validated) != raw:
        _hold("HOLD_TYPED_PRETEST_JSON", "typed unit registry bytes are not canonical")
    return _rebuild_typed_unit_registry(
        root,
        validated,
        expectations=expectations,
        predictor_verifiers=predictor_verifiers,
        control_verifiers=control_verifiers,
    )


def _load_registry_ref_exact(
    run_root: Path,
    reference: Mapping,
    *,
    expectations: PretestFreezeExpectations | Mapping,
    predictor_verifiers: Mapping[str, PredictorVerifier] | None,
    control_verifiers: Mapping[str, PretestControlVerifier] | None,
) -> dict:
    ref = _development_ref(reference, label="typed unit registry")
    try:
        path = resolve_artifact_ref(run_root, ref)
    except ArtifactGraphError as exc:
        _hold("HOLD_TYPED_PRETEST_REFERENCE", f"typed unit registry: {exc}")
    return load_typed_unit_registry_exact(
        run_root,
        path,
        ref["sha256"],
        expectations=expectations,
        predictor_verifiers=predictor_verifiers,
        control_verifiers=control_verifiers,
    )


def _selector_matches_registry(
    run_root: Path,
    selector: Mapping,
    registry: Mapping,
    *,
    expected: Mapping,
    predictor_verifiers: Mapping[str, PredictorVerifier] | None,
) -> None:
    registry_refs = _predictor_ref_by_method(registry)
    for method in registry["eligible_baseline_method_order"]:
        try:
            evidence = replay_development_baseline_evidence(
                run_root,
                selector["development_evidence_refs"][method],
                predictor_verifiers=predictor_verifiers,
                expected_run_start_sha256=expected["run_start_sha256"],
                expected_activation_manifest_sha256=expected[
                    "development_capability_sha256"
                ],
                expected_train_development_outcomes_sha256=expected[
                    "train_development_outcomes_sha256"
                ],
                expected_non_data_authority_hashes=expected[
                    "non_data_authority_hashes"
                ],
                expected_method_id=method,
                expected_information_set_fingerprint=expected[
                    "information_set_fingerprint"
                ],
            )
        except BaselineSelectionError as exc:
            _hold(
                "HOLD_TYPED_PRETEST_SELECTOR",
                f"{method} selector evidence failed strict replay: {exc}",
            )
        if evidence["predictor_provenance_ref"] != registry_refs[method]:
            _hold(
                "HOLD_TYPED_PRETEST_SELECTOR",
                f"{method} selector evidence binds a different predictor provenance",
            )


def _strict_replay_summary(registry: Mapping) -> dict:
    scalar_count = sum(
        unit["unit_kind"] == SCALAR_FIT for unit in registry["units"]
    )
    neural_count = sum(
        unit["unit_kind"] == NEURAL_SEED for unit in registry["units"]
    )
    control_count = sum(
        unit["unit_kind"] == CONTROL_RUN for unit in registry["units"]
    )
    return {
        "typed_paired_policy_replayed": True,
        "selector_replayed": True,
        "predictor_provenance_replayed": True,
        "checkpoint_selection_replayed": True,
        "development_model_state_replayed": True,
        "required_controls_replayed": False,
        "synthetic_control_scaffold_replayed": True,
        "pretest_control_registry_v1_replayed": False,
        "control_qualification_passed": False,
        "baseline_method_count": len(MINIMUM_BASELINE_METHOD_IDS),
        "scalar_fit_unit_count": scalar_count,
        "neural_seed_unit_count": neural_count,
        "control_run_unit_count": control_count,
    }


def validate_typed_pretest_freeze(payload: Mapping) -> dict:
    """Validate exact freeze bytes; strict ref replay is performed by load/write."""

    raw = _strict_keys(payload, _FREEZE_KEYS, label="typed pre-test freeze")
    exact = {
        "schema_version": PRETEST_FREEZE_SCHEMA,
        "artifact_role": PRETEST_FREEZE_ROLE,
        "status": PRETEST_FREEZE_STATUS,
        "scientific_scope": SCIENTIFIC_SCOPE,
    }
    for field, expected in exact.items():
        if raw.get(field) != expected:
            _hold("HOLD_TYPED_PRETEST_SCHEMA", f"freeze field changed: {field}")
    for field in (
        "run_start_sha256",
        "run_plan_sha256",
        "development_capability_sha256",
        "train_development_outcomes_sha256",
        "development_bindings_sha256",
        "information_set_fingerprint",
        "required_estimand_cells_sha256",
    ):
        _digest(raw.get(field), label=field)
    policy_ref = _development_ref(raw.get("paired_policy_ref"), label="paired policy")
    registry_ref = _development_ref(
        raw.get("typed_unit_registry_ref"), label="typed unit registry"
    )
    selector_ref = _development_ref(
        raw.get("development_selector_ref"), label="development selector"
    )
    refs = [
        (policy_ref["relative_path"], policy_ref["sha256"]),
        (registry_ref["relative_path"], registry_ref["sha256"]),
        (selector_ref["relative_path"], selector_ref["sha256"]),
    ]
    if len(refs) != len(set(refs)):
        _hold("HOLD_TYPED_PRETEST_REFERENCE_ALIAS", "freeze refs must be distinct")
    order_value = raw.get("eligible_baseline_method_order")
    if not isinstance(order_value, list):
        _hold("HOLD_TYPED_PRETEST_BASELINE_SET", "freeze baseline order must be a list")
    order = [_method_id(method) for method in order_value]
    if len(order) != len(set(order)) or set(order) != set(MINIMUM_BASELINE_METHOD_IDS):
        _hold("HOLD_TYPED_PRETEST_BASELINE_SET", "freeze baseline set is not exact")
    if raw.get("eligible_baseline_method_set_sha256") != canonical_sha256(
        sorted(order)
    ):
        _hold("HOLD_TYPED_PRETEST_BASELINE_SET", "freeze baseline-set digest is stale")
    selected = _method_id(
        raw.get("selected_baseline_method_id"), label="selected baseline method"
    )
    if selected not in set(order):
        _hold("HOLD_TYPED_PRETEST_SELECTOR", "selected method is not policy eligible")
    if raw.get("required_control_ids") != list(REQUIRED_CONTROL_IDS):
        _hold("HOLD_TYPED_PRETEST_CONTROL_SET", "freeze control set changed")
    replay = _strict_keys(
        raw.get("strict_replay"), _STRICT_REPLAY_KEYS, label="strict replay summary"
    )
    expected_replay = {
        "typed_paired_policy_replayed": True,
        "selector_replayed": True,
        "predictor_provenance_replayed": True,
        "checkpoint_selection_replayed": True,
        "development_model_state_replayed": True,
        "required_controls_replayed": False,
        "synthetic_control_scaffold_replayed": True,
        "pretest_control_registry_v1_replayed": False,
        "control_qualification_passed": False,
        "baseline_method_count": len(MINIMUM_BASELINE_METHOD_IDS),
        "scalar_fit_unit_count": len(REQUIRED_SCALAR_METHOD_IDS),
        "neural_seed_unit_count": len(FROZEN_NEURAL_SEEDS),
        "control_run_unit_count": len(REQUIRED_CONTROL_IDS),
    }
    if dict(replay) != expected_replay:
        _hold("HOLD_TYPED_PRETEST_REPLAY_DRIFT", "strict replay summary is not exact")
    if raw.get("remaining_blockers") != list(REMAINING_BLOCKERS):
        _hold("HOLD_TYPED_PRETEST_BOUNDARY", "remaining blockers changed")
    _boundaries(raw, label="typed pre-test freeze")
    if raw.get("shipped_capability_status") != SHIPPED_CAPABILITY_STATUS:
        _hold(
            "HOLD_TYPED_PRETEST_BOUNDARY",
            "typed pre-test freeze shipped capability status changed",
        )
    return {
        **dict(raw),
        "paired_policy_ref": policy_ref,
        "typed_unit_registry_ref": registry_ref,
        "development_selector_ref": selector_ref,
        "eligible_baseline_method_order": order,
        "strict_replay": dict(replay),
    }


def canonical_typed_pretest_freeze_bytes(payload: Mapping) -> bytes:
    try:
        return canonical_json_bytes(validate_typed_pretest_freeze(payload))
    except ArtifactIOError as exc:
        _hold("HOLD_TYPED_PRETEST_JSON", str(exc))


def build_typed_pretest_freeze(
    run_root: Path,
    *,
    typed_unit_registry_ref: Mapping,
    development_selector_ref: Mapping,
    expectations: PretestFreezeExpectations | Mapping,
    predictor_verifiers: Mapping[str, PredictorVerifier] | None,
    control_verifiers: Mapping[str, PretestControlVerifier] | None,
) -> dict:
    """Build a strict development evidence freeze that remains an explicit HOLD.

    The direct controls replayed here are only synthetic migration scaffolding.
    This function neither consumes nor replaces the authoritative pre-test
    control registry; its result is therefore never feature-stage acceptable.
    """

    root = _checked_run_root(run_root)
    expected = _expectations(expectations)
    registry_ref = _development_ref(
        typed_unit_registry_ref, label="typed unit registry"
    )
    registry = _load_registry_ref_exact(
        root,
        registry_ref,
        expectations=expected,
        predictor_verifiers=predictor_verifiers,
        control_verifiers=control_verifiers,
    )
    if registry["status"] != UNIT_REGISTRY_READY:
        not_run = [
            unit["control_id"]
            for unit in registry["units"]
            if unit["unit_kind"] == CONTROL_NOT_RUN
        ]
        _hold(
            "HOLD_TYPED_PRETEST_CONTROL_NOT_RUN",
            "required controls are not all active and replayed",
            control_ids=not_run,
        )
    selector_ref = _development_ref(
        development_selector_ref, label="development selector"
    )
    try:
        selector_path = resolve_artifact_ref(root, selector_ref)
        selector = load_development_baseline_selection_exact(
            root,
            selector_path,
            selector_ref["sha256"],
            expected_run_start_sha256=expected["run_start_sha256"],
            expected_activation_manifest_sha256=expected[
                "development_capability_sha256"
            ],
            expected_train_development_outcomes_sha256=expected[
                "train_development_outcomes_sha256"
            ],
            expected_non_data_authority_hashes=expected[
                "non_data_authority_hashes"
            ],
            expected_eligible_method_order=registry[
                "eligible_baseline_method_order"
            ],
            expected_required_estimand_cells=expected["required_estimand_cells"],
            predictor_verifiers=predictor_verifiers,
        )
    except (ArtifactGraphError, BaselineSelectionError) as exc:
        _hold("HOLD_TYPED_PRETEST_SELECTOR", f"selector failed strict replay: {exc}")
    if selector["status"] != QUALIFIED_STATUS:
        _hold(
            "HOLD_TYPED_PRETEST_SIGN_GATE",
            "development selector did not meet the unchanged 0.60 sign gate",
        )
    if selector["information_set_fingerprint"] != expected[
        "information_set_fingerprint"
    ]:
        _hold("HOLD_TYPED_PRETEST_SELECTOR", "selector information set changed")
    _selector_matches_registry(
        root,
        selector,
        registry,
        expected=expected,
        predictor_verifiers=predictor_verifiers,
    )
    replay_summary = _strict_replay_summary(registry)
    payload = {
        "schema_version": PRETEST_FREEZE_SCHEMA,
        "artifact_role": PRETEST_FREEZE_ROLE,
        "status": PRETEST_FREEZE_STATUS,
        "scientific_scope": SCIENTIFIC_SCOPE,
        "run_start_sha256": expected["run_start_sha256"],
        "run_plan_sha256": expected["run_plan_sha256"],
        "development_capability_sha256": expected[
            "development_capability_sha256"
        ],
        "train_development_outcomes_sha256": expected[
            "train_development_outcomes_sha256"
        ],
        "development_bindings_sha256": _development_bindings_sha256(expected),
        "information_set_fingerprint": expected["information_set_fingerprint"],
        "required_estimand_cells_sha256": canonical_sha256(
            expected["required_estimand_cells"]
        ),
        "paired_policy_ref": registry["paired_policy_ref"],
        "typed_unit_registry_ref": registry_ref,
        "development_selector_ref": selector_ref,
        "eligible_baseline_method_order": registry[
            "eligible_baseline_method_order"
        ],
        "eligible_baseline_method_set_sha256": registry[
            "eligible_baseline_method_set_sha256"
        ],
        "selected_baseline_method_id": selector["selected_method_id"],
        "required_control_ids": list(REQUIRED_CONTROL_IDS),
        "strict_replay": replay_summary,
        "remaining_blockers": list(REMAINING_BLOCKERS),
        "control_authority_scope": DIRECT_CONTROL_VERIFIER_SCOPE,
        "pretest_control_registry_schema_required": (
            PRETEST_CONTROL_REGISTRY_SCHEMA_REQUIRED
        ),
        "pretest_control_registry_v1_replayed": False,
        "pretest_control_registry_can_satisfy_freeze": False,
        "control_qualification_passed": False,
        "feature_stage_acceptable": False,
        "sign_accuracy_gate": SIGN_ACCURACY_GATE,
        "sign_gate_change_authorized": False,
        "ordinary_only": True,
        "internal_test_feature_access": False,
        "internal_test_outcome_access": False,
        "sealed_access": False,
        "downstream_unlock": False,
        "training_ready": False,
        "capability_ready": False,
        "shipped_capability_status": SHIPPED_CAPABILITY_STATUS,
    }
    return validate_typed_pretest_freeze(payload)


def _rebuild_typed_pretest_freeze(
    run_root: Path,
    payload: Mapping,
    *,
    expectations: PretestFreezeExpectations | Mapping,
    predictor_verifiers: Mapping[str, PredictorVerifier] | None,
    control_verifiers: Mapping[str, PretestControlVerifier] | None,
) -> dict:
    validated = validate_typed_pretest_freeze(payload)
    rebuilt = build_typed_pretest_freeze(
        run_root,
        typed_unit_registry_ref=validated["typed_unit_registry_ref"],
        development_selector_ref=validated["development_selector_ref"],
        expectations=expectations,
        predictor_verifiers=predictor_verifiers,
        control_verifiers=control_verifiers,
    )
    if rebuilt != validated:
        _hold(
            "HOLD_TYPED_PRETEST_REPLAY_DRIFT",
            "typed pre-test freeze changed during full strict replay",
        )
    return validated


def write_typed_pretest_freeze_exclusive(
    run_root: Path,
    path: Path,
    payload: Mapping,
    *,
    expectations: PretestFreezeExpectations | Mapping,
    predictor_verifiers: Mapping[str, PredictorVerifier] | None,
    control_verifiers: Mapping[str, PretestControlVerifier] | None,
) -> str:
    candidate = _checked_output_path(run_root, path, label="typed pre-test freeze output")
    replayed = _rebuild_typed_pretest_freeze(
        run_root,
        payload,
        expectations=expectations,
        predictor_verifiers=predictor_verifiers,
        control_verifiers=control_verifiers,
    )
    try:
        return atomic_write_json_exclusive(
            candidate, replayed, validator=validate_typed_pretest_freeze
        )
    except ArtifactIOError as exc:
        _hold("HOLD_TYPED_PRETEST_WRITE", str(exc))


def load_typed_pretest_freeze_exact(
    run_root: Path,
    path: Path,
    expected_sha256: str,
    *,
    expectations: PretestFreezeExpectations | Mapping,
    predictor_verifiers: Mapping[str, PredictorVerifier] | None,
    control_verifiers: Mapping[str, PretestControlVerifier] | None,
) -> dict:
    root = _checked_run_root(run_root)
    candidate = _checked_input_path(
        root,
        path,
        expected_sha256,
        label="typed pre-test freeze",
    )
    raw, parsed = _load_json_exact(
        candidate, expected_sha256, label="typed pre-test freeze"
    )
    validated = validate_typed_pretest_freeze(parsed)
    if canonical_typed_pretest_freeze_bytes(validated) != raw:
        _hold("HOLD_TYPED_PRETEST_JSON", "typed pre-test freeze bytes are not canonical")
    return _rebuild_typed_pretest_freeze(
        root,
        validated,
        expectations=expectations,
        predictor_verifiers=predictor_verifiers,
        control_verifiers=control_verifiers,
    )


def typed_unit_registry_ref(run_root: Path, path: Path, sha256: str) -> dict:
    """Create and exact-read the canonical ref used by a freeze artifact."""

    try:
        reference = make_artifact_ref(run_root, path, sha256)
    except ArtifactGraphError as exc:
        _hold("HOLD_TYPED_PRETEST_REFERENCE", str(exc))
    return _development_ref(reference, label="typed unit registry")


__all__ = [
    "CONTROL_NOT_RUN",
    "CONTROL_REPLAY_ACTIVE",
    "CONTROL_REPLAY_RESULT_SCHEMA",
    "CONTROL_RUN",
    "CURRENT_CONTROL_SUMMARY_PRETEST_BLOCKER",
    "DIRECT_CONTROL_VERIFIER_SCOPE",
    "NEURAL_SEED",
    "PRETEST_FREEZE_SCHEMA",
    "PRETEST_FREEZE_STATUS",
    "PRETEST_CONTROL_REGISTRY_SCHEMA_REQUIRED",
    "PretestControlVerifier",
    "PretestFreezeExpectations",
    "REMAINING_BLOCKERS",
    "REQUIRED_CONTROL_IDS",
    "SCALAR_FIT",
    "SHIPPED_CAPABILITY_STATUS",
    "TypedPretestFreezeError",
    "UNIT_REGISTRY_HOLD",
    "UNIT_REGISTRY_READY",
    "UNIT_REGISTRY_SCHEMA",
    "build_typed_pretest_freeze",
    "build_typed_unit_registry",
    "canonical_typed_pretest_freeze_bytes",
    "canonical_typed_unit_registry_bytes",
    "load_typed_pretest_freeze_exact",
    "load_typed_unit_registry_exact",
    "typed_unit_registry_ref",
    "validate_typed_pretest_freeze",
    "validate_typed_unit_registry",
    "write_typed_pretest_freeze_exclusive",
    "write_typed_unit_registry_exclusive",
]
