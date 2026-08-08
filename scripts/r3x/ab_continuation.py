"""Strict ordinary-development A/B continuation decisions.

This module is a decision compiler, not a trainer.  It consumes only exact,
hash-addressed artifacts beneath an ordinary ``development`` namespace and
recomputes every decision-bearing quantity from committed prediction rows.
It deliberately refuses internal-test and sealed namespaces, never changes the
0.60 sign gate, and cannot authorize downstream execution.

The decision boundary is intentionally stronger than a JSON schema check:

* the V3 strongest-baseline selector is exact-loaded and fully replayed;
* every candidate/control evidence artifact is replayed through its trusted
  predictor verifier;
* neural evidence must be the frozen five-seed prediction ensemble (metrics
  averaged across seeds are not an accepted input);
* method/config/architecture/parameter claims are recompiled from the exact
  neural registry binding;
* all methods must share the exact information set, registered cells, row
  order/set, evaluation rows, and global component assignment;
* paired component-bootstrap and synchronized component-swap statistics are
  recomputed as one pre-frozen Holm family.

Missing required references yield a deterministic development HOLD.  Complete
evidence that misses a scientific gate yields a scientific FAIL.  Malformed,
tampered, non-canonical, or cross-partition inputs raise and are never converted
into a softer result.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import torch

from .artifact_graph import (
    ArtifactGraphError,
    PRIMARY_INFORMATION_SET,
    parse_prediction_jsonl,
    read_artifact_ref,
    validate_artifact_ref,
)
from .artifact_io import (
    ArtifactIOError,
    atomic_write_json_exclusive,
    canonical_json_bytes,
)
from .baseline_selection import (
    DEVELOPMENT_PARTITION,
    SELECTION_SCHEMA,
    BaselineSelectionError,
    load_development_baseline_selection_exact,
    parse_development_evaluation_jsonl,
    replay_development_baseline_evidence,
)
from .continuation_neural_evidence import (
    CONTINUATION_NEURAL_EVIDENCE_SCHEMA,
    ContinuationNeuralEvidenceError,
    load_continuation_neural_evidence_exact,
)
from .models import ModelConfig
from .metrics import MetricProtocolError
from .neural_method_registry import (
    A0_METHOD_ID,
    A1_CONTROL_METHOD_ID,
    A1_METHOD_ID,
    A1_UNK_ONLY_METHOD_ID,
    A1_WRONG_LOCAL_METHOD_ID,
    B_ADDITIVE_METHOD_ID,
    B_MATCHED_CONTROL_METHOD_ID,
    B_UNK_ONLY_METHOD_ID,
    B_WRONG_LOCAL_METHOD_ID,
    MAXIMUM_MATCHED_PARAMETER_GAP,
    REGISTRY_SCHEMA,
    NeuralMethodRegistryError,
    canonical_registry_payload,
    compile_registered_development_model,
    registry_sha256,
)
from .predictor_provenance import (
    ENSEMBLE_RULE,
    FROZEN_NEURAL_SEEDS,
    NEURAL_KIND,
    PredictorProvenanceError,
    PredictorVerifier,
    replay_predictor_provenance,
    validate_development_checkpoint,
)
from .protocol import SIGN_ACCURACY_GATE
from .statistics import (
    PRIMARY_PAIRED_METRIC_ID,
    _paired_sign_scores,
    holm_adjust,
    paired_primary_sign_bootstrap,
    paired_primary_sign_permutation,
)


POLICY_SCHEMA = "r3x_ab_continuation_policy_v3"
MODEL_BINDING_SCHEMA = "r3x_continuation_model_binding_evidence_v1"
A_DECISION_SCHEMA = "r3x_a_continuation_decision_v2"
B_DECISION_SCHEMA = "r3x_b_continuation_decision_v2"

POLICY_ROLE = "ORDINARY_DEVELOPMENT_AB_CONTINUATION_POLICY"
MODEL_BINDING_ROLE = "ORDINARY_DEVELOPMENT_RECOMPILED_MODEL_BINDING"
A_DECISION_ROLE = "ORDINARY_DEVELOPMENT_A_CONTINUATION_DECISION"
B_DECISION_ROLE = "ORDINARY_DEVELOPMENT_B_CONTINUATION_DECISION"

GO_STATUS = "GO_DEVELOPMENT_CONTINUATION"
FAIL_STATUS = "FAIL_SCIENTIFIC_CONTINUATION_CRITERIA"
HOLD_STATUS = "HOLD_MISSING_OR_UNDEFINED_DEVELOPMENT_EVIDENCE"
HOLD_A_STATUS = "HOLD_A_CONTINUATION_NOT_GO"

C3_STATUS = "PROPOSED_REDLINE_NOT_ACTIVE"
C3_GUIDANCE_POSITION = "DGM_DOOB_DERIVED_ADAPTATION_NOT_EXACT_TARGET_SAMPLER"
B_INTERACTION_STATUS = "NOT_RUN_ACTION_SUPPORT"

POSITIVE_RULE = "POSITIVE_EFFECT_CI_LOWER_GT_ZERO_AND_HOLM_P_LE_ALPHA"
MATCHED_DIRECTION = "CANDIDATE_GREATER_THAN_MATCHED_CONTROL"
CONTROL_NONREPLICATION_RULE = (
    "CANDIDATE_SIGNIFICANTLY_GREATER_THAN_CONTROL_NOT_ZERO_EFFECT_CLAIM"
)

A_ROLE_ORDER = (
    "candidate",
    "mechanism_reference",
    "matched_control",
    "wrong_local_control",
    "unk_control",
)
B_ROLE_ORDER = (
    "candidate",
    "a_reference",
    "matched_control",
    "wrong_local_control",
    "unk_control",
)

_EXACT_METHOD_ID_BY_ROUTE_ROLE = {
    "A": {
        "candidate": A1_METHOD_ID,
        "mechanism_reference": A0_METHOD_ID,
        "matched_control": A1_CONTROL_METHOD_ID,
        "wrong_local_control": A1_WRONG_LOCAL_METHOD_ID,
        "unk_control": A1_UNK_ONLY_METHOD_ID,
    },
    "B": {
        "candidate": B_ADDITIVE_METHOD_ID,
        "a_reference": A1_METHOD_ID,
        "matched_control": B_MATCHED_CONTROL_METHOD_ID,
        "wrong_local_control": B_WRONG_LOCAL_METHOD_ID,
        "unk_control": B_UNK_ONLY_METHOD_ID,
    },
}

A_COMPARISON_LAYOUT = (
    ("A_CANDIDATE_VS_SELECTED_COMPARATOR", "candidate", "SELECTED_COMPARATOR"),
    ("A_CANDIDATE_VS_MECHANISM_REFERENCE", "candidate", "mechanism_reference"),
    ("A_CANDIDATE_VS_MATCHED_CONTROL", "candidate", "matched_control"),
    ("A_CANDIDATE_VS_WRONG_LOCAL_CONTROL", "candidate", "wrong_local_control"),
    ("A_CANDIDATE_VS_UNK_CONTROL", "candidate", "unk_control"),
)
B_COMPARISON_LAYOUT = (
    ("B_CANDIDATE_VS_SELECTED_COMPARATOR", "candidate", "SELECTED_COMPARATOR"),
    ("B_CANDIDATE_VS_A_REFERENCE", "candidate", "a_reference"),
    ("B_CANDIDATE_VS_MATCHED_CONTROL", "candidate", "matched_control"),
    ("B_CANDIDATE_VS_WRONG_LOCAL_CONTROL", "candidate", "wrong_local_control"),
    ("B_CANDIDATE_VS_UNK_CONTROL", "candidate", "unk_control"),
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_METHOD_ID = re.compile(r"^[A-Z][A-Z0-9_]*$")

_METHOD_SPEC_KEYS = {
    "method_id",
    "architecture_id",
    "scientific_role",
    "requires_parameter_match_to_role",
}
_COMPARISON_POLICY_KEYS = {"comparison_id", "left_role", "right_role", "rule"}
_BOOTSTRAP_KEYS = {
    "n_resamples",
    "seed",
    "min_components_per_cell",
    "max_attempt_multiplier",
}
_PERMUTATION_KEYS = {
    "n_permutations",
    "seed",
    "exact_component_limit",
    "min_components_per_cell",
}
_POLICY_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "policy_id",
    "partition",
    "information_set",
    "selector_schema",
    "development_selector_ref",
    "sign_gate",
    "registered_seed_ids",
    "ensemble_rule",
    "neural_evidence_schema",
    "registry_schema",
    "registry_sha256",
    "maximum_parameter_gap_fraction",
    "paired_metric_id",
    "paired_family_alpha",
    "paired_confidence",
    "paired_bootstrap",
    "paired_permutation",
    "matched_control_direction",
    "control_nonreplication_rule",
    "a_method_roles",
    "b_method_roles",
    "a_model_binding_refs",
    "b_model_binding_refs",
    "a_comparisons",
    "b_comparisons",
    "b_interaction_status",
    "c3_status",
    "c3_guidance_position",
    "ordinary_development_only",
    "internal_test_access",
    "sealed_access",
    "downstream_unlock",
}
_MODEL_BINDING_KEYS = {
    "schema_version",
    "artifact_role",
    "method_id",
    "architecture_id",
    "scientific_role",
    "information_set",
    "information_set_fingerprint",
    "predictor_config_sha256",
    "registry_schema",
    "registry_sha256",
    "compiled_binding",
    "compiled_binding_sha256",
    "parameter_match_qualified",
    "ordinary_development_only",
    "internal_test_access",
    "sealed_access",
    "downstream_unlock",
}
_PAIR_RESULT_KEYS = {
    "comparison_id",
    "left_role",
    "right_role",
    "left_method_id",
    "right_method_id",
    "rule",
    "left_raw_sign_utility",
    "right_raw_sign_utility",
    "oriented_improvement",
    "confidence_interval",
    "raw_p_value_two_sided",
    "holm_adjusted_p_value",
    "alpha",
    "bootstrap",
    "permutation",
    "passed",
}
_PAIRED_FAMILY_KEYS = {
    "family_id",
    "metric_id",
    "comparison_order",
    "raw_p_values_two_sided",
    "holm_adjusted_p_values",
    "alpha",
    "confidence",
    "results",
    "all_qualified",
}
_CHECK_KEYS = {
    "a_prerequisite_go",
    "evidence_complete",
    "same_information_set",
    "same_registered_cells",
    "same_row_set_and_order",
    "same_component_assignment",
    "five_seed_ensembles",
    "exact_neural_state_receipts",
    "independent_train_refit_verified",
    "independent_component_assignment_replay_verified",
    "independent_development_target_replay_verified",
    "same_fit_config_across_roles",
    "model_registry_config_bound",
    "parameter_match_qualified",
    "all_registered_cells_defined",
    "candidate_sign_gate_met",
    "candidate_beats_selected_comparator",
    "candidate_beats_mechanism_or_a_reference",
    "candidate_beats_matched_control",
    "wrong_local_does_not_replicate",
    "unk_control_does_not_replicate",
    "paired_family_qualified",
}
_BINDING_KEYS = {
    "run_start_sha256",
    "activation_manifest_sha256",
    "train_development_outcomes_sha256",
    "non_data_authority_hashes",
    "information_set_fingerprint",
    "registered_cell_ids_sha256",
    "ordered_row_ids_sha256",
    "row_set_sha256",
    "component_assignment_sha256",
}
_DECISION_KEYS = {
    "schema_version",
    "artifact_role",
    "route",
    "status",
    "decision",
    "reason_codes",
    "policy_ref",
    "policy_sha256",
    "selector_ref",
    "selector_status",
    "selected_comparator_method_id",
    "comparator_qualification",
    "method_evidence_refs",
    "model_binding_refs",
    "neural_evidence_refs",
    "a_decision_ref",
    "bindings",
    "registered_cell_ids",
    "n_registered_cells",
    "candidate_method_id",
    "candidate_macro_sign_accuracy",
    "sign_gate",
    "sign_gate_met",
    "verified_five_seed_roles",
    "verified_model_binding_roles",
    "verified_neural_receipt_roles",
    "paired_family",
    "scientific_checks",
    "b_interaction_status",
    "c3_status",
    "c3_guidance_position",
    "ordinary_development_only",
    "internal_test_access",
    "sealed_access",
    "downstream_unlock",
}


class ABContinuationError(RuntimeError):
    """Fail-closed continuation policy, evidence, or replay error."""


def _strict_keys(value: Mapping, expected: set[str], *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ABContinuationError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ABContinuationError(
            f"{label} keys differ from strict schema: {sorted(actual ^ expected)}"
        )


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise ABContinuationError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ABContinuationError(f"{label} must be a stable identifier")
    return value


def _method_id(value: object, *, label: str = "method_id") -> str:
    if not isinstance(value, str) or not _METHOD_ID.fullmatch(value):
        raise ABContinuationError(f"{label} must be a frozen uppercase method ID")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ABContinuationError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ABContinuationError(f"{label} must be finite")
    return number


def _positive_integer(value: object, *, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ABContinuationError(f"{label} must be an integer >= {minimum}")
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
        raise ABContinuationError(f"value is not canonical finite JSON: {exc}") from exc
    return hashlib.sha256(raw).hexdigest()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict:
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key!r}")
        output[key] = value
    return output


def _parse_json_object(raw: bytes, *, label: str) -> dict:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ABContinuationError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ABContinuationError(f"{label} must be a JSON object")
    return payload


def _development_ref(value: Mapping, *, label: str) -> dict:
    try:
        reference = validate_artifact_ref(value)
    except ArtifactGraphError as exc:
        raise ABContinuationError(str(exc)) from exc
    parts = tuple(
        part.lower() for part in PurePosixPath(reference["relative_path"]).parts
    )
    if "development" not in parts:
        raise ABContinuationError(f"{label} must be beneath a literal development namespace")
    if any(
        part in {"test", "internal_test", "sealed", "restricted"}
        or "internal_test" in part
        or "gse246381" in part
        for part in parts
    ):
        raise ABContinuationError(f"{label} enters a prohibited partition namespace")
    return reference


def _read_ref(run_root: Path, reference: Mapping, *, label: str) -> bytes:
    normalized = _development_ref(reference, label=label)
    try:
        return read_artifact_ref(Path(run_root), normalized)
    except ArtifactGraphError as exc:
        raise ABContinuationError(str(exc)) from exc


def _registry_entries() -> dict[str, dict]:
    payload = canonical_registry_payload()
    return {entry["method_id"]: entry for entry in payload["methods"]}


def _method_specs(
    value: Mapping,
    *,
    route: str,
    role_order: Sequence[str],
    registry_entries: Mapping[str, Mapping],
) -> dict[str, dict]:
    if not isinstance(value, Mapping) or set(value) != set(role_order):
        raise ABContinuationError(
            f"{route} method roles must be exactly {list(role_order)}"
        )
    expected_method_ids = _EXACT_METHOD_ID_BY_ROUTE_ROLE.get(route)
    if expected_method_ids is None or set(role_order) != set(expected_method_ids):
        raise ABContinuationError(f"{route} exact method-role layout is not registered")
    normalized = {}
    method_ids = []
    for role in role_order:
        spec = value[role]
        _strict_keys(spec, _METHOD_SPEC_KEYS, label=f"{route} method role {role}")
        method = _method_id(spec.get("method_id"), label=f"{route}.{role}.method_id")
        if method != expected_method_ids[role]:
            raise ABContinuationError(
                f"{route}.{role} method ID differs from the frozen exact registry role"
            )
        entry = registry_entries.get(method)
        if entry is None:
            raise ABContinuationError(
                f"{route}.{role} method is absent from the exact neural registry"
            )
        architecture = _identifier(
            spec.get("architecture_id"), label=f"{route}.{role}.architecture_id"
        )
        scientific_role = _identifier(
            spec.get("scientific_role"), label=f"{route}.{role}.scientific_role"
        )
        if architecture != entry["architecture_id"]:
            raise ABContinuationError(
                f"{route}.{role} architecture differs from the exact registry"
            )
        if scientific_role != entry["scientific_role"]:
            raise ABContinuationError(
                f"{route}.{role} scientific role differs from the exact registry"
            )
        target = spec.get("requires_parameter_match_to_role")
        if target is not None and target not in role_order:
            raise ABContinuationError(
                f"{route}.{role} parameter-match target is not a route role"
            )
        normalized[role] = {
            "method_id": method,
            "architecture_id": architecture,
            "scientific_role": scientific_role,
            "requires_parameter_match_to_role": target,
        }
        method_ids.append(method)
    if len(method_ids) != len(set(method_ids)):
        raise ABContinuationError(f"{route} method roles must use distinct method IDs")
    expected_targets = {
        "candidate": None,
        "matched_control": "candidate",
        "wrong_local_control": "candidate",
        "unk_control": "candidate",
        **(
            {"mechanism_reference": None}
            if route == "A"
            else {"a_reference": None}
        ),
    }
    for role, expected_target in expected_targets.items():
        if normalized[role]["requires_parameter_match_to_role"] != expected_target:
            raise ABContinuationError(
                f"{route}.{role} parameter-match target differs from the frozen role design"
            )
    return normalized


def _comparison_specs(
    value: Sequence[Mapping],
    *,
    route: str,
    expected_layout: Sequence[tuple[str, str, str]],
) -> list[dict]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ABContinuationError(f"{route} comparisons must be a sequence")
    if len(value) != len(expected_layout):
        raise ABContinuationError(f"{route} comparison family is incomplete")
    normalized = []
    for index, (item, expected) in enumerate(zip(value, expected_layout)):
        _strict_keys(item, _COMPARISON_POLICY_KEYS, label=f"{route} comparison {index}")
        comparison_id = _identifier(
            item.get("comparison_id"), label=f"{route} comparison_id"
        )
        left = _identifier(item.get("left_role"), label=f"{route} left_role")
        right = _identifier(item.get("right_role"), label=f"{route} right_role")
        if (comparison_id, left, right) != expected:
            raise ABContinuationError(
                f"{route} comparison order/direction differs from the frozen layout"
            )
        if item.get("rule") != POSITIVE_RULE:
            raise ABContinuationError(f"{route} comparison rule is not frozen")
        normalized.append({
            "comparison_id": comparison_id,
            "left_role": left,
            "right_role": right,
            "rule": POSITIVE_RULE,
        })
    return normalized


def _policy_model_binding_refs(
    value: Mapping,
    *,
    route: str,
    role_order: Sequence[str],
) -> dict[str, dict]:
    if not isinstance(value, Mapping) or set(value) != set(role_order):
        raise ABContinuationError(
            f"{route} model-binding refs must cover exactly {list(role_order)}"
        )
    return {
        role: _development_ref(
            value[role], label=f"{route} pre-frozen model binding {role}"
        )
        for role in role_order
    }


def validate_continuation_policy(payload: Mapping) -> dict:
    """Validate a pre-outcome policy against the exact current registry."""

    _strict_keys(payload, _POLICY_KEYS, label="A/B continuation policy")
    if payload.get("schema_version") != POLICY_SCHEMA:
        raise ABContinuationError("unsupported A/B continuation policy schema")
    if payload.get("artifact_role") != POLICY_ROLE or payload.get("status") != "FROZEN_PRE_OUTCOME":
        raise ABContinuationError("continuation policy must be frozen before outcomes")
    _identifier(payload.get("policy_id"), label="policy_id")
    if payload.get("partition") != DEVELOPMENT_PARTITION:
        raise ABContinuationError("continuation policy is not ordinary development")
    if payload.get("information_set") != PRIMARY_INFORMATION_SET:
        raise ABContinuationError("continuation policy information set is invalid")
    if payload.get("selector_schema") != SELECTION_SCHEMA:
        raise ABContinuationError("continuation policy must require selector V3")
    selector_ref = _development_ref(
        payload.get("development_selector_ref"),
        label="pre-frozen development selector",
    )
    if _finite(payload.get("sign_gate"), label="sign_gate") != SIGN_ACCURACY_GATE:
        raise ABContinuationError("the frozen 0.60 sign gate cannot be changed")
    if payload.get("registered_seed_ids") != list(FROZEN_NEURAL_SEEDS):
        raise ABContinuationError("continuation policy must freeze exactly five seeds")
    if payload.get("ensemble_rule") != ENSEMBLE_RULE:
        raise ABContinuationError("continuation policy ensemble rule is invalid")
    if payload.get("neural_evidence_schema") != CONTINUATION_NEURAL_EVIDENCE_SCHEMA:
        raise ABContinuationError(
            "continuation policy must require the exact neural-state receipt schema"
        )
    if payload.get("registry_schema") != REGISTRY_SCHEMA:
        raise ABContinuationError("continuation policy registry schema is stale")
    bound_registry_sha = _digest(payload.get("registry_sha256"), label="registry_sha256")
    if bound_registry_sha != registry_sha256():
        raise ABContinuationError("continuation policy registry SHA does not match current code")
    gap = _finite(
        payload.get("maximum_parameter_gap_fraction"),
        label="maximum_parameter_gap_fraction",
    )
    if gap != MAXIMUM_MATCHED_PARAMETER_GAP:
        raise ABContinuationError("parameter-match maximum must remain exactly five percent")
    if payload.get("paired_metric_id") != PRIMARY_PAIRED_METRIC_ID:
        raise ABContinuationError("continuation paired metric is not the frozen sign utility")
    alpha = _finite(payload.get("paired_family_alpha"), label="paired_family_alpha")
    confidence = _finite(payload.get("paired_confidence"), label="paired_confidence")
    if not 0.0 < alpha < 1.0 or not math.isclose(
        confidence, 1.0 - alpha, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ABContinuationError("paired alpha/confidence are inconsistent")
    bootstrap = payload.get("paired_bootstrap")
    _strict_keys(bootstrap, _BOOTSTRAP_KEYS, label="paired_bootstrap")
    _positive_integer(bootstrap.get("n_resamples"), label="bootstrap.n_resamples", minimum=2000)
    _positive_integer(bootstrap.get("seed"), label="bootstrap.seed", minimum=0)
    bootstrap_min = _positive_integer(
        bootstrap.get("min_components_per_cell"),
        label="bootstrap.min_components_per_cell",
        minimum=10,
    )
    _positive_integer(
        bootstrap.get("max_attempt_multiplier"),
        label="bootstrap.max_attempt_multiplier",
    )
    permutation = payload.get("paired_permutation")
    _strict_keys(permutation, _PERMUTATION_KEYS, label="paired_permutation")
    _positive_integer(
        permutation.get("n_permutations"),
        label="permutation.n_permutations",
        minimum=10000,
    )
    _positive_integer(permutation.get("seed"), label="permutation.seed", minimum=0)
    if permutation.get("exact_component_limit") != 20:
        raise ABContinuationError("paired exact component limit must remain 20")
    permutation_min = _positive_integer(
        permutation.get("min_components_per_cell"),
        label="permutation.min_components_per_cell",
        minimum=10,
    )
    if bootstrap_min != permutation_min:
        raise ABContinuationError("bootstrap and permutation cell support must match")
    if payload.get("matched_control_direction") != MATCHED_DIRECTION:
        raise ABContinuationError("matched-control direction is not pre-frozen")
    if payload.get("control_nonreplication_rule") != CONTROL_NONREPLICATION_RULE:
        raise ABContinuationError("control non-replication rule is not frozen")
    registry_entries = _registry_entries()
    a_roles = _method_specs(
        payload.get("a_method_roles"),
        route="A",
        role_order=A_ROLE_ORDER,
        registry_entries=registry_entries,
    )
    b_roles = _method_specs(
        payload.get("b_method_roles"),
        route="B",
        role_order=B_ROLE_ORDER,
        registry_entries=registry_entries,
    )
    if b_roles["a_reference"]["method_id"] != a_roles["candidate"]["method_id"]:
        raise ABContinuationError("B a_reference must be the frozen A candidate method")
    a_binding_refs = _policy_model_binding_refs(
        payload.get("a_model_binding_refs"),
        route="A",
        role_order=A_ROLE_ORDER,
    )
    b_binding_refs = _policy_model_binding_refs(
        payload.get("b_model_binding_refs"),
        route="B",
        role_order=B_ROLE_ORDER,
    )
    a_comparisons = _comparison_specs(
        payload.get("a_comparisons"),
        route="A",
        expected_layout=A_COMPARISON_LAYOUT,
    )
    b_comparisons = _comparison_specs(
        payload.get("b_comparisons"),
        route="B",
        expected_layout=B_COMPARISON_LAYOUT,
    )
    if payload.get("b_interaction_status") != B_INTERACTION_STATUS:
        raise ABContinuationError("B interaction must remain NOT_RUN_ACTION_SUPPORT")
    if payload.get("c3_status") != C3_STATUS:
        raise ABContinuationError("C3 must remain inactive")
    if payload.get("c3_guidance_position") != C3_GUIDANCE_POSITION:
        raise ABContinuationError("C3 guidance claim boundary changed")
    if payload.get("ordinary_development_only") is not True:
        raise ABContinuationError("continuation policy must remain development-only")
    if payload.get("internal_test_access") is not False:
        raise ABContinuationError("continuation policy cannot access internal test")
    if payload.get("sealed_access") is not False or payload.get("downstream_unlock") is not False:
        raise ABContinuationError("continuation policy cannot access sealed or unlock downstream")
    normalized = dict(payload)
    normalized["development_selector_ref"] = selector_ref
    normalized["a_method_roles"] = a_roles
    normalized["b_method_roles"] = b_roles
    normalized["a_model_binding_refs"] = a_binding_refs
    normalized["b_model_binding_refs"] = b_binding_refs
    normalized["a_comparisons"] = a_comparisons
    normalized["b_comparisons"] = b_comparisons
    return normalized


def canonical_continuation_policy_bytes(payload: Mapping) -> bytes:
    try:
        return canonical_json_bytes(validate_continuation_policy(payload))
    except ArtifactIOError as exc:
        raise ABContinuationError(str(exc)) from exc


def write_continuation_policy_exclusive(path: Path, payload: Mapping) -> str:
    _assert_development_output(Path(path), label="continuation policy output")
    try:
        return atomic_write_json_exclusive(
            Path(path), payload, validator=validate_continuation_policy
        )
    except ArtifactIOError as exc:
        raise ABContinuationError(str(exc)) from exc


def load_continuation_policy_exact(run_root: Path, reference: Mapping) -> dict:
    raw = _read_ref(run_root, reference, label="continuation policy reference")
    policy = validate_continuation_policy(
        _parse_json_object(raw, label="continuation policy")
    )
    if canonical_continuation_policy_bytes(policy) != raw:
        raise ABContinuationError("continuation policy bytes are not canonical")
    return policy


def _recompile_binding(compiled: Mapping) -> dict:
    if not isinstance(compiled, Mapping):
        raise ABContinuationError("compiled_binding must be an object")
    method = _method_id(compiled.get("method_id"), label="compiled_binding.method_id")
    config = compiled.get("model_config")
    if not isinstance(config, Mapping):
        raise ABContinuationError("compiled binding model_config must be an object")
    try:
        cfg = ModelConfig(**dict(config))
        with torch.random.fork_rng(devices=[], enabled=True):
            model, expected = compile_registered_development_model(
                method,
                cfg,
                expected_registry_sha256=_digest(
                    compiled.get("registry_sha256"),
                    label="compiled_binding.registry_sha256",
                ),
            )
        del model
    except (TypeError, NeuralMethodRegistryError) as exc:
        raise ABContinuationError(f"model binding cannot be recompiled: {exc}") from exc
    if dict(compiled) != expected:
        raise ABContinuationError("compiled model binding differs from exact registry recompilation")
    return expected


def build_model_binding_evidence(
    *,
    method_id: str,
    model_config: ModelConfig,
    expected_registry_sha256: str,
    predictor_config_sha256: str,
    information_set_fingerprint: str,
) -> dict:
    """Recompile a model/config binding without loading data or training."""

    try:
        with torch.random.fork_rng(devices=[], enabled=True):
            model, compiled = compile_registered_development_model(
                method_id,
                model_config,
                expected_registry_sha256=expected_registry_sha256,
            )
        del model
    except NeuralMethodRegistryError as exc:
        raise ABContinuationError(str(exc)) from exc
    match = compiled.get("parameter_match")
    qualified = None if match is None else bool(match.get("matched") is True)
    payload = {
        "schema_version": MODEL_BINDING_SCHEMA,
        "artifact_role": MODEL_BINDING_ROLE,
        "method_id": compiled["method_id"],
        "architecture_id": compiled["architecture_id"],
        "scientific_role": compiled["scientific_role"],
        "information_set": compiled["information_set"],
        "information_set_fingerprint": _digest(
            information_set_fingerprint, label="information_set_fingerprint"
        ),
        "predictor_config_sha256": _digest(
            predictor_config_sha256, label="predictor_config_sha256"
        ),
        "registry_schema": compiled["registry_schema"],
        "registry_sha256": compiled["registry_sha256"],
        "compiled_binding": compiled,
        "compiled_binding_sha256": _canonical_sha256(compiled),
        "parameter_match_qualified": qualified,
        "ordinary_development_only": True,
        "internal_test_access": False,
        "sealed_access": False,
        "downstream_unlock": False,
    }
    return validate_model_binding_evidence(payload)


def validate_model_binding_evidence(payload: Mapping) -> dict:
    _strict_keys(payload, _MODEL_BINDING_KEYS, label="continuation model binding")
    if payload.get("schema_version") != MODEL_BINDING_SCHEMA:
        raise ABContinuationError("unsupported continuation model-binding schema")
    if payload.get("artifact_role") != MODEL_BINDING_ROLE:
        raise ABContinuationError("continuation model-binding role is invalid")
    compiled = _recompile_binding(payload.get("compiled_binding"))
    for field in (
        "method_id",
        "architecture_id",
        "scientific_role",
        "information_set",
        "registry_schema",
        "registry_sha256",
    ):
        if payload.get(field) != compiled[field]:
            raise ABContinuationError(f"model-binding {field} differs from recompilation")
    if payload.get("information_set") != PRIMARY_INFORMATION_SET:
        raise ABContinuationError("model binding information set is invalid")
    _digest(payload.get("information_set_fingerprint"), label="information_set_fingerprint")
    _digest(payload.get("predictor_config_sha256"), label="predictor_config_sha256")
    if payload.get("compiled_binding_sha256") != _canonical_sha256(compiled):
        raise ABContinuationError("compiled model-binding SHA is invalid")
    match = compiled.get("parameter_match")
    qualified = None if match is None else bool(match.get("matched") is True)
    if payload.get("parameter_match_qualified") is not qualified:
        raise ABContinuationError("parameter-match qualification is not recomputable")
    if payload.get("ordinary_development_only") is not True:
        raise ABContinuationError("model binding must remain development-only")
    if payload.get("internal_test_access") is not False:
        raise ABContinuationError("model binding cannot access internal test")
    if payload.get("sealed_access") is not False or payload.get("downstream_unlock") is not False:
        raise ABContinuationError("model binding cannot access sealed or unlock downstream")
    return dict(payload)


def canonical_model_binding_evidence_bytes(payload: Mapping) -> bytes:
    try:
        return canonical_json_bytes(validate_model_binding_evidence(payload))
    except ArtifactIOError as exc:
        raise ABContinuationError(str(exc)) from exc


def write_model_binding_evidence_exclusive(path: Path, payload: Mapping) -> str:
    _assert_development_output(Path(path), label="model-binding evidence output")
    try:
        return atomic_write_json_exclusive(
            Path(path), payload, validator=validate_model_binding_evidence
        )
    except ArtifactIOError as exc:
        raise ABContinuationError(str(exc)) from exc


def load_model_binding_evidence_exact(run_root: Path, reference: Mapping) -> dict:
    raw = _read_ref(run_root, reference, label="model-binding evidence reference")
    binding = validate_model_binding_evidence(
        _parse_json_object(raw, label="model-binding evidence")
    )
    if canonical_model_binding_evidence_bytes(binding) != raw:
        raise ABContinuationError("model-binding evidence bytes are not canonical")
    return binding


def _selector_path(run_root: Path, reference: Mapping) -> Path:
    normalized = _development_ref(reference, label="selector reference")
    # ``load_development_baseline_selection_exact`` performs the descriptor-level
    # no-symlink/hash checks.  This lexical join is only its input path.
    return Path(run_root).joinpath(*PurePosixPath(normalized["relative_path"]).parts)


def _assert_development_output(path: Path, *, label: str) -> None:
    parts = tuple(part.lower() for part in Path(path).parts)
    if "development" not in parts:
        raise ABContinuationError(f"{label} must be beneath a literal development namespace")
    if any(
        part in {"test", "internal_test", "sealed", "restricted"}
        or "internal_test" in part
        or "gse246381" in part
        for part in parts
    ):
        raise ABContinuationError(f"{label} enters a prohibited partition namespace")


def _assert_output_within_run_root(run_root: Path, path: Path, *, label: str) -> None:
    try:
        root = Path(run_root).resolve(strict=True)
        candidate = Path(path).resolve(strict=False)
    except OSError as exc:
        raise ABContinuationError(
            f"{label} containment cannot be established: {exc}"
        ) from exc
    if not root.is_dir() or not candidate.is_relative_to(root):
        raise ABContinuationError(f"{label} must remain beneath the exact run root")


def _load_selector(
    run_root: Path,
    reference: Mapping,
    *,
    selector_context: Mapping,
) -> dict:
    expected_keys = {
        "expected_run_start_sha256",
        "expected_activation_manifest_sha256",
        "expected_train_development_outcomes_sha256",
        "expected_non_data_authority_hashes",
        "expected_eligible_method_order",
        "expected_required_estimand_cells",
        "predictor_verifiers",
    }
    _strict_keys(selector_context, expected_keys, label="selector replay context")
    normalized = _development_ref(reference, label="selector reference")
    try:
        return load_development_baseline_selection_exact(
            Path(run_root),
            _selector_path(run_root, normalized),
            normalized["sha256"],
            **dict(selector_context),
        )
    except BaselineSelectionError as exc:
        raise ABContinuationError(f"selector exact replay failed: {exc}") from exc


def _load_method_bundle(
    run_root: Path,
    reference: Mapping,
    *,
    selector: Mapping,
    method_id: str,
    predictor_verifiers: Mapping[str, PredictorVerifier] | None,
) -> dict:
    try:
        evidence = replay_development_baseline_evidence(
            Path(run_root),
            reference,
            predictor_verifiers=predictor_verifiers,
            expected_run_start_sha256=selector["run_start_sha256"],
            expected_activation_manifest_sha256=selector["activation_manifest_sha256"],
            expected_train_development_outcomes_sha256=selector[
                "train_development_outcomes_sha256"
            ],
            expected_non_data_authority_hashes=selector["non_data_authority_hashes"],
            expected_method_id=method_id,
            expected_information_set_fingerprint=selector[
                "information_set_fingerprint"
            ],
        )
        provenance = replay_predictor_provenance(
            Path(run_root),
            evidence["predictor_provenance_ref"],
            verifiers=predictor_verifiers,
            expected_run_start_sha256=evidence["run_start_sha256"],
            expected_activation_manifest_sha256=evidence["activation_manifest_sha256"],
            expected_train_development_outcomes_sha256=evidence[
                "train_development_outcomes_sha256"
            ],
            expected_non_data_authority_hashes=evidence["non_data_authority_hashes"],
            expected_method_id=method_id,
            expected_information_set_fingerprint=evidence[
                "information_set_fingerprint"
            ],
        )
    except (BaselineSelectionError, PredictorProvenanceError) as exc:
        raise ABContinuationError(f"method evidence replay failed for {method_id}: {exc}") from exc
    neural_cells = [
        cell
        for estimand in provenance["estimands"]
        for cell in estimand["cells"]
    ]
    five_seed = bool(neural_cells) and all(
        cell["predictor_artifacts"]["kind"] == NEURAL_KIND
        and cell["predictor_artifacts"]["seeds"] == list(FROZEN_NEURAL_SEEDS)
        and cell["predictor_artifacts"]["ensemble_rule"] == ENSEMBLE_RULE
        and cell["predictor_artifacts"]["ensemble_prediction_ref"]
        == cell["prediction_jsonl_ref"]
        for cell in neural_cells
    )
    architecture_ids = set()
    if five_seed:
        for cell in neural_cells:
            for seed_artifact in cell["predictor_artifacts"]["seed_artifacts"]:
                raw = _read_ref(
                    run_root,
                    seed_artifact["checkpoint_ref"],
                    label="development checkpoint reference",
                )
                checkpoint = validate_development_checkpoint(
                    _parse_json_object(raw, label="development checkpoint")
                )
                if canonical_json_bytes(checkpoint) != raw:
                    raise ABContinuationError("development checkpoint bytes are not canonical")
                architecture_ids.add(checkpoint["architecture_id"])
    vectors = _evidence_vectors(run_root, evidence)
    return {
        "evidence": evidence,
        "provenance": provenance,
        "five_seed_verified": five_seed,
        "checkpoint_architecture_ids": sorted(architecture_ids),
        **vectors,
    }


def _evidence_vectors(run_root: Path, evidence: Mapping) -> dict:
    rows = []
    predictions = []
    cell_ids = []
    for estimand in evidence["estimands"]:
        for cell in estimand["cells"]:
            try:
                prediction_rows = parse_prediction_jsonl(
                    read_artifact_ref(Path(run_root), cell["prediction_jsonl_ref"])
                )
                evaluation_rows = parse_development_evaluation_jsonl(
                    read_artifact_ref(
                        Path(run_root), cell["development_evaluation_jsonl_ref"]
                    )
                )
            except (ArtifactGraphError, BaselineSelectionError) as exc:
                raise ABContinuationError(f"development vector replay failed: {exc}") from exc
            if [row["row_id"] for row in prediction_rows] != [
                row["row_id"] for row in evaluation_rows
            ]:
                raise ABContinuationError("prediction/evaluation row order differs")
            frozen_cell_id = f"{estimand['estimand_id']}::{cell['cell_id']}"
            cell_ids.append(frozen_cell_id)
            for evaluation, prediction in zip(evaluation_rows, prediction_rows):
                rows.append({
                    "r3_input_row_sha256": evaluation["row_id"],
                    "study": "R3X_ORDINARY_DEVELOPMENT",
                    "estimand_id": estimand["estimand_id"],
                    "context_id_v2": cell["cell_id"],
                    "actual_edit_count": 1,
                    "source_component_id": evaluation["source_component_id"],
                    "delta": float(evaluation["true_delta"]),
                })
                predictions.append(float(prediction["predicted_delta_mean"]))
    row_ids = [row["r3_input_row_sha256"] for row in rows]
    if not rows or len(row_ids) != len(set(row_ids)):
        raise ABContinuationError("development evidence rows must be non-empty and unique")
    if cell_ids != sorted(cell_ids) or len(cell_ids) != len(set(cell_ids)):
        raise ABContinuationError(
            "registered estimand/cell IDs must be canonical and unique"
        )
    component_assignment = [
        {
            "row_id": row["r3_input_row_sha256"],
            "cell_id": f"{row['estimand_id']}::{row['context_id_v2']}",
            "source_component_id": row["source_component_id"],
        }
        for row in rows
    ]
    return {
        "rows": rows,
        "predictions": predictions,
        "registered_cell_ids": cell_ids,
        "ordered_row_ids_sha256": _canonical_sha256(row_ids),
        "row_set_sha256": _canonical_sha256(sorted(row_ids)),
        "component_assignment_sha256": _canonical_sha256(component_assignment),
    }


def _validate_role_binding(
    *,
    role: str,
    role_spec: Mapping,
    bundle: Mapping,
    binding: Mapping,
    all_role_specs: Mapping[str, Mapping],
    maximum_gap: float,
) -> None:
    evidence = bundle["evidence"]
    if binding["method_id"] != role_spec["method_id"] or evidence["method_id"] != role_spec["method_id"]:
        raise ABContinuationError(f"{role} method binding/evidence mismatch")
    if binding["architecture_id"] != role_spec["architecture_id"]:
        raise ABContinuationError(f"{role} architecture binding mismatch")
    if binding["scientific_role"] != role_spec["scientific_role"]:
        raise ABContinuationError(f"{role} scientific role binding mismatch")
    if binding["information_set_fingerprint"] != evidence["information_set_fingerprint"]:
        raise ABContinuationError(f"{role} information-set fingerprint mismatch")
    if binding["predictor_config_sha256"] != evidence["non_data_authority_hashes"][
        "model_optimizer_seed_config_sha256"
    ]:
        raise ABContinuationError(f"{role} predictor-config authority mismatch")
    if bundle["checkpoint_architecture_ids"] != [binding["architecture_id"]]:
        raise ABContinuationError(f"{role} checkpoints do not use one exact registered architecture")
    target_role = role_spec["requires_parameter_match_to_role"]
    if target_role is not None:
        target_method = all_role_specs[target_role]["method_id"]
        compiled = binding["compiled_binding"]
        if compiled.get("matched_reference_method_id") != target_method:
            raise ABContinuationError(f"{role} matched-reference method is not pre-frozen")
        match = compiled.get("parameter_match")
        if not isinstance(match, Mapping) or match.get("matched") is not True:
            raise ABContinuationError(f"{role} parameter match is not qualified")
        gap = _finite(match.get("gap_fraction"), label=f"{role} parameter gap")
        if _finite(
            match.get("maximum_gap_fraction"),
            label=f"{role} maximum parameter gap",
        ) != maximum_gap:
            raise ABContinuationError(
                f"{role} parameter-match maximum differs from the frozen policy"
            )
        if gap > maximum_gap:
            raise ABContinuationError(f"{role} parameter gap exceeds the frozen maximum")


def _validate_role_neural_receipt(
    *,
    role: str,
    role_spec: Mapping,
    bundle: Mapping,
    binding: Mapping,
    receipt: Mapping,
    selector: Mapping,
) -> None:
    """Bind one exact neural receipt to the generic evidence and frozen model.

    This validates state-load/development-replay evidence only.  The caller must
    separately require ``independent_train_refit_verified`` before any GO.
    """

    evidence = bundle["evidence"]
    expected_equal = {
        "predictor_provenance_ref": evidence["predictor_provenance_ref"],
        "method_id": role_spec["method_id"],
        "architecture_id": binding["architecture_id"],
        "registry_schema": binding["registry_schema"],
        "registry_sha256": binding["registry_sha256"],
        "compiled_binding_sha256": binding["compiled_binding_sha256"],
        "model_config_sha256": binding["compiled_binding"]["model_config_sha256"],
        "information_set_fingerprint": evidence["information_set_fingerprint"],
        "predictor_config_authority_sha256": evidence["non_data_authority_hashes"][
            "model_optimizer_seed_config_sha256"
        ],
        "run_start_sha256": selector["run_start_sha256"],
        "activation_manifest_sha256": selector["activation_manifest_sha256"],
        "train_development_outcomes_sha256": selector[
            "train_development_outcomes_sha256"
        ],
        "non_data_authority_hashes": selector["non_data_authority_hashes"],
        "registered_cell_ids": bundle["registered_cell_ids"],
        "continuation_development_ordered_row_ids_sha256": bundle[
            "ordered_row_ids_sha256"
        ],
        "continuation_development_row_set_sha256": bundle["row_set_sha256"],
    }
    compiled = binding["compiled_binding"]
    expected_equal["training_variant_id"] = compiled["training_variant_id"]
    expected_equal["control_intervention"] = compiled["control_intervention"]
    for field, expected in expected_equal.items():
        if receipt.get(field) != expected:
            raise ABContinuationError(
                f"{role} neural receipt {field} differs from exact continuation binding"
            )
    if receipt.get("information_set") != PRIMARY_INFORMATION_SET:
        raise ABContinuationError(f"{role} neural receipt information set changed")
    if receipt.get("ensemble_rule") != ENSEMBLE_RULE:
        raise ABContinuationError(f"{role} neural receipt ensemble rule changed")
    estimands = receipt.get("estimands")
    if not isinstance(estimands, list) or not estimands:
        raise ABContinuationError(f"{role} neural receipt lacks estimands")
    if any(item.get("seeds") != list(FROZEN_NEURAL_SEEDS) for item in estimands):
        raise ABContinuationError(f"{role} neural receipt seed binding changed")
    receipt_prediction_layout = [
        {
            "estimand_id": estimand.get("estimand_id"),
            "cell_id": cell.get("cell_id"),
            "ensemble_prediction_ref": cell.get("ensemble_prediction_ref"),
        }
        for estimand in estimands
        for cell in estimand.get("cells", [])
    ]
    provenance = bundle.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ABContinuationError(f"{role} generic provenance is absent")
    provenance_prediction_layout = [
        {
            "estimand_id": estimand.get("estimand_id"),
            "cell_id": cell.get("cell_id"),
            "ensemble_prediction_ref": cell.get("prediction_jsonl_ref"),
        }
        for estimand in provenance.get("estimands", [])
        for cell in estimand.get("cells", [])
    ]
    if (
        not receipt_prediction_layout
        or receipt_prediction_layout != provenance_prediction_layout
    ):
        raise ABContinuationError(
            f"{role} neural receipt ensemble refs differ from decision provenance"
        )


def _same_vector_bindings(bundles: Mapping[str, Mapping]) -> dict:
    first = next(iter(bundles.values()))
    fields = (
        "registered_cell_ids",
        "ordered_row_ids_sha256",
        "row_set_sha256",
        "component_assignment_sha256",
    )
    return {
        field: all(bundle[field] == first[field] for bundle in bundles.values())
        for field in fields
    }


def _pair_result(
    *,
    comparison: Mapping,
    left_method_id: str,
    right_method_id: str,
    rows: Sequence[Mapping],
    left_prediction: Sequence[float],
    right_prediction: Sequence[float],
    policy: Mapping,
) -> dict:
    try:
        score = _paired_sign_scores(rows, left_prediction, right_prediction)
        bootstrap = paired_primary_sign_bootstrap(
            rows,
            left_prediction,
            right_prediction,
            confidence=policy["paired_confidence"],
            **policy["paired_bootstrap"],
        )
        permutation = paired_primary_sign_permutation(
            rows,
            left_prediction,
            right_prediction,
            **policy["paired_permutation"],
        )
    except MetricProtocolError as exc:
        # Undefined inferential support is evidence absence (HOLD), whereas an
        # implementation error must propagate rather than be mislabeled.
        raise ABContinuationError(f"UNDEFINED_PAIRED_INFERENCE:{type(exc).__name__}:{exc}") from exc
    return {
        "comparison_id": comparison["comparison_id"],
        "left_role": comparison["left_role"],
        "right_role": comparison["right_role"],
        "left_method_id": left_method_id,
        "right_method_id": right_method_id,
        "rule": POSITIVE_RULE,
        "left_raw_sign_utility": float(score["candidate"]),
        "right_raw_sign_utility": float(score["baseline"]),
        "oriented_improvement": float(score["improvement"]),
        "confidence_interval": [
            float(value)
            for value in bootstrap["confidence_interval_oriented_improvement"]
        ],
        "raw_p_value_two_sided": float(permutation["p_value_two_sided"]),
        "holm_adjusted_p_value": None,
        "alpha": float(policy["paired_family_alpha"]),
        "bootstrap": bootstrap,
        "permutation": permutation,
        "passed": False,
    }


def _paired_family(
    *,
    route: str,
    comparisons: Sequence[Mapping],
    bundles: Mapping[str, Mapping],
    comparator_bundle: Mapping,
    selector: Mapping,
    policy: Mapping,
) -> dict:
    results = []
    for comparison in comparisons:
        left_role = comparison["left_role"]
        right_role = comparison["right_role"]
        left = bundles[left_role]
        if right_role == "SELECTED_COMPARATOR":
            right = comparator_bundle
            right_method = selector["selected_method_id"]
        else:
            right = bundles[right_role]
            right_method = right["evidence"]["method_id"]
        results.append(
            _pair_result(
                comparison=comparison,
                left_method_id=left["evidence"]["method_id"],
                right_method_id=right_method,
                rows=left["rows"],
                left_prediction=left["predictions"],
                right_prediction=right["predictions"],
                policy=policy,
            )
        )
    raw = {item["comparison_id"]: item["raw_p_value_two_sided"] for item in results}
    adjusted = holm_adjust(raw)
    for item in results:
        item["holm_adjusted_p_value"] = adjusted[item["comparison_id"]]
        item["passed"] = bool(
            item["oriented_improvement"] > 0.0
            and item["confidence_interval"][0] > 0.0
            and item["holm_adjusted_p_value"] <= item["alpha"]
        )
    order = [item["comparison_id"] for item in results]
    return {
        "family_id": f"R3X_{route}_CONTINUATION_PRIMARY_SIGN_FAMILY_V1",
        "metric_id": PRIMARY_PAIRED_METRIC_ID,
        "comparison_order": order,
        "raw_p_values_two_sided": {key: raw[key] for key in order},
        "holm_adjusted_p_values": {key: adjusted[key] for key in order},
        "alpha": float(policy["paired_family_alpha"]),
        "confidence": float(policy["paired_confidence"]),
        "results": results,
        "all_qualified": all(item["passed"] for item in results),
    }


def _empty_checks(*, route: str, a_prerequisite_go: bool | None) -> dict:
    return {
        "a_prerequisite_go": a_prerequisite_go,
        "evidence_complete": False,
        "same_information_set": None,
        "same_registered_cells": None,
        "same_row_set_and_order": None,
        "same_component_assignment": None,
        "five_seed_ensembles": None,
        "exact_neural_state_receipts": None,
        "independent_train_refit_verified": None,
        "independent_component_assignment_replay_verified": None,
        "independent_development_target_replay_verified": None,
        "same_fit_config_across_roles": None,
        "model_registry_config_bound": None,
        "parameter_match_qualified": None,
        "all_registered_cells_defined": None,
        "candidate_sign_gate_met": None,
        "candidate_beats_selected_comparator": None,
        "candidate_beats_mechanism_or_a_reference": None,
        "candidate_beats_matched_control": None,
        "wrong_local_does_not_replicate": None,
        "unk_control_does_not_replicate": None,
        "paired_family_qualified": None,
    }


def _decision_base(
    *,
    route: str,
    policy_ref: Mapping,
    policy: Mapping,
    selector_ref: Mapping,
    selector: Mapping,
    evidence_refs: Mapping,
    model_binding_refs: Mapping,
    neural_evidence_refs: Mapping,
    a_decision_ref: Mapping | None,
) -> dict:
    return {
        "schema_version": A_DECISION_SCHEMA if route == "A" else B_DECISION_SCHEMA,
        "artifact_role": A_DECISION_ROLE if route == "A" else B_DECISION_ROLE,
        "route": route,
        "status": HOLD_STATUS,
        "decision": "HOLD",
        "reason_codes": [],
        "policy_ref": _development_ref(policy_ref, label="policy reference"),
        "policy_sha256": _development_ref(policy_ref, label="policy reference")["sha256"],
        "selector_ref": _development_ref(selector_ref, label="selector reference"),
        "selector_status": selector["status"],
        "selected_comparator_method_id": selector["selected_method_id"],
        "comparator_qualification": (
            "QUALIFIED" if selector["can_satisfy_pretest_freeze"] else "RECOVERY_ONLY"
        ),
        "method_evidence_refs": {
            role: _development_ref(ref, label=f"{route} {role} evidence")
            for role, ref in evidence_refs.items()
        },
        "model_binding_refs": {
            role: _development_ref(ref, label=f"{route} {role} model binding")
            for role, ref in model_binding_refs.items()
        },
        "neural_evidence_refs": {
            role: _development_ref(ref, label=f"{route} {role} neural receipt")
            for role, ref in neural_evidence_refs.items()
        },
        "a_decision_ref": (
            None
            if a_decision_ref is None
            else _development_ref(a_decision_ref, label="A decision reference")
        ),
        "bindings": None,
        "registered_cell_ids": None,
        "n_registered_cells": None,
        "candidate_method_id": policy[f"{route.lower()}_method_roles"]["candidate"]["method_id"],
        "candidate_macro_sign_accuracy": None,
        "sign_gate": SIGN_ACCURACY_GATE,
        "sign_gate_met": None,
        "verified_five_seed_roles": [],
        "verified_model_binding_roles": [],
        "verified_neural_receipt_roles": [],
        "paired_family": None,
        "scientific_checks": _empty_checks(
            route=route, a_prerequisite_go=None if route == "A" else False
        ),
        "b_interaction_status": B_INTERACTION_STATUS,
        "c3_status": C3_STATUS,
        "c3_guidance_position": C3_GUIDANCE_POSITION,
        "ordinary_development_only": True,
        "internal_test_access": False,
        "sealed_access": False,
        "downstream_unlock": False,
    }


def _build_route_decision(
    run_root: Path,
    *,
    route: str,
    policy_ref: Mapping,
    selector_ref: Mapping,
    method_evidence_refs: Mapping[str, Mapping],
    model_binding_refs: Mapping[str, Mapping],
    neural_evidence_refs: Mapping[str, Mapping] | None,
    selector_context: Mapping,
    predictor_verifiers: Mapping[str, PredictorVerifier] | None,
    a_decision_ref: Mapping | None = None,
    a_decision: Mapping | None = None,
) -> dict:
    policy = load_continuation_policy_exact(run_root, policy_ref)
    normalized_selector_ref = _development_ref(
        selector_ref, label="selector reference"
    )
    if normalized_selector_ref != policy["development_selector_ref"]:
        raise ABContinuationError(
            "decision selector ref differs from the pre-outcome policy freeze"
        )
    selector = _load_selector(run_root, selector_ref, selector_context=selector_context)
    roles = A_ROLE_ORDER if route == "A" else B_ROLE_ORDER
    role_specs = policy[f"{route.lower()}_method_roles"]
    comparisons = policy[f"{route.lower()}_comparisons"]
    if not isinstance(method_evidence_refs, Mapping):
        raise ABContinuationError("method_evidence_refs must be an object")
    if not isinstance(model_binding_refs, Mapping):
        raise ABContinuationError("model_binding_refs must be an object")
    if neural_evidence_refs is None:
        neural_evidence_refs = {}
    if not isinstance(neural_evidence_refs, Mapping):
        raise ABContinuationError("neural_evidence_refs must be an object")
    unknown_evidence = sorted(set(method_evidence_refs) - set(roles))
    unknown_bindings = sorted(set(model_binding_refs) - set(roles))
    unknown_neural = sorted(set(neural_evidence_refs) - set(roles))
    if unknown_evidence or unknown_bindings or unknown_neural:
        raise ABContinuationError(
            "continuation inputs contain unknown roles: "
            f"evidence={unknown_evidence}, bindings={unknown_bindings}, "
            f"neural_receipts={unknown_neural}"
        )
    expected_binding_refs = policy[f"{route.lower()}_model_binding_refs"]
    for role, reference in model_binding_refs.items():
        normalized_reference = _development_ref(
            reference, label=f"{route} {role} model binding"
        )
        if normalized_reference != expected_binding_refs[role]:
            raise ABContinuationError(
                f"{route} {role} model binding differs from the pre-outcome policy freeze"
            )
    decision = _decision_base(
        route=route,
        policy_ref=policy_ref,
        policy=policy,
        selector_ref=selector_ref,
        selector=selector,
        evidence_refs=method_evidence_refs,
        model_binding_refs=model_binding_refs,
        neural_evidence_refs=neural_evidence_refs,
        a_decision_ref=a_decision_ref,
    )
    checks = decision["scientific_checks"]
    if route == "B":
        if a_decision is None:
            raise ABContinuationError("B continuation requires an exact A decision")
        if a_decision.get("policy_ref") != decision["policy_ref"]:
            raise ABContinuationError("B and A decisions do not share one exact policy ref")
        if a_decision.get("selector_ref") != decision["selector_ref"]:
            raise ABContinuationError("B and A decisions do not share one exact selector ref")
        if a_decision.get("decision") != "GO":
            decision["status"] = HOLD_A_STATUS
            decision["reason_codes"] = ["A_CONTINUATION_NOT_GO"]
            return validate_continuation_decision(decision)
        checks["a_prerequisite_go"] = True
        if a_decision["candidate_method_id"] != role_specs["a_reference"]["method_id"]:
            raise ABContinuationError("B policy A reference differs from exact A decision")
        a_bindings = a_decision.get("bindings")
        if not isinstance(a_bindings, Mapping):
            raise ABContinuationError("A GO decision lacks exact data/component bindings")
        for field in (
            "run_start_sha256",
            "activation_manifest_sha256",
            "train_development_outcomes_sha256",
            "non_data_authority_hashes",
            "information_set_fingerprint",
        ):
            expected = (
                selector[field]
                if field != "information_set_fingerprint"
                else selector["information_set_fingerprint"]
            )
            if a_bindings[field] != expected:
                raise ABContinuationError(f"A/B prerequisite {field} binding mismatch")
    missing_evidence = [role for role in roles if role not in method_evidence_refs]
    missing_bindings = [role for role in roles if role not in model_binding_refs]
    missing_neural = [role for role in roles if role not in neural_evidence_refs]
    if missing_evidence or missing_bindings or missing_neural:
        decision["reason_codes"] = [
            *[f"MISSING_METHOD_EVIDENCE:{role}" for role in missing_evidence],
            *[f"MISSING_MODEL_BINDING:{role}" for role in missing_bindings],
            *[f"MISSING_NEURAL_STATE_RECEIPT:{role}" for role in missing_neural],
        ]
        return validate_continuation_decision(decision)

    bundles = {}
    bindings = {}
    neural_receipts = {}
    for role in roles:
        bundles[role] = _load_method_bundle(
            run_root,
            method_evidence_refs[role],
            selector=selector,
            method_id=role_specs[role]["method_id"],
            predictor_verifiers=predictor_verifiers,
        )
        bindings[role] = load_model_binding_evidence_exact(
            run_root, model_binding_refs[role]
        )
        try:
            neural_receipts[role] = load_continuation_neural_evidence_exact(
                run_root, neural_evidence_refs[role]
            )
        except ContinuationNeuralEvidenceError as exc:
            raise ABContinuationError(
                f"{role} neural-state receipt exact replay failed: {exc}"
            ) from exc
    comparator_ref = selector["development_evidence_refs"][selector["selected_method_id"]]
    comparator = _load_method_bundle(
        run_root,
        comparator_ref,
        selector=selector,
        method_id=selector["selected_method_id"],
        predictor_verifiers=predictor_verifiers,
    )

    all_bundles = {**bundles, "SELECTED_COMPARATOR": comparator}
    vector_checks = _same_vector_bindings(all_bundles)
    if not all(vector_checks.values()):
        raise ABContinuationError(
            "candidate, controls, and selected comparator do not share exact cells/rows/components"
        )
    five_seed_roles = [role for role in roles if bundles[role]["five_seed_verified"]]
    if five_seed_roles != list(roles):
        decision["reason_codes"] = [
            f"NOT_FIXED_FIVE_SEED_PREDICTION_ENSEMBLE:{role}"
            for role in roles
            if role not in five_seed_roles
        ]
        checks.update({
            "evidence_complete": True,
            "same_information_set": True,
            "same_registered_cells": True,
            "same_row_set_and_order": True,
            "same_component_assignment": True,
            "five_seed_ensembles": False,
            "model_registry_config_bound": None,
            "parameter_match_qualified": None,
            "all_registered_cells_defined": True,
        })
        return validate_continuation_decision(decision)

    receipt_binding_failures = []
    for role in roles:
        _validate_role_binding(
            role=role,
            role_spec=role_specs[role],
            bundle=bundles[role],
            binding=bindings[role],
            all_role_specs=role_specs,
            maximum_gap=policy["maximum_parameter_gap_fraction"],
        )
        try:
            _validate_role_neural_receipt(
                role=role,
                role_spec=role_specs[role],
                bundle=bundles[role],
                binding=bindings[role],
                receipt=neural_receipts[role],
                selector=selector,
            )
        except ABContinuationError:
            receipt_binding_failures.append(role)

    if receipt_binding_failures:
        decision["reason_codes"] = [
            f"NEURAL_STATE_RECEIPT_BINDING_MISMATCH:{role}"
            for role in receipt_binding_failures
        ]
        return validate_continuation_decision(decision)

    decision["verified_neural_receipt_roles"] = list(roles)
    checks["exact_neural_state_receipts"] = True
    fit_config_shas = {receipt.get("fit_config_sha256") for receipt in neural_receipts.values()}
    checks["same_fit_config_across_roles"] = len(fit_config_shas) == 1
    independent_requirements = (
        (
            "independent_train_refit_verified",
            "NEURAL_RECEIPT_NO_INDEPENDENT_TRAIN_REFIT",
        ),
        (
            "independent_component_assignment_replay_verified",
            "NEURAL_RECEIPT_NO_INDEPENDENT_COMPONENT_ASSIGNMENT_REPLAY",
        ),
        (
            "independent_development_target_replay_verified",
            "NEURAL_RECEIPT_NO_INDEPENDENT_DEVELOPMENT_TARGET_REPLAY",
        ),
    )
    reasons = []
    for field, reason in independent_requirements:
        qualified_roles = [
            role for role in roles if neural_receipts[role].get(field) is True
        ]
        checks[field] = qualified_roles == list(roles)
        reasons.extend(
            f"{reason}:{role}" for role in roles if role not in qualified_roles
        )
    if not checks["same_fit_config_across_roles"]:
        reasons.append("NEURAL_RECEIPT_FIT_CONFIG_MISMATCH_ACROSS_ROLES")
    if reasons:
        decision["reason_codes"] = reasons
        return validate_continuation_decision(decision)

    candidate = bundles["candidate"]
    candidate_sign = float(
        candidate["evidence"]["aggregate_development_metrics"][
            "macro_component_cell_sign_accuracy"
        ]
    )
    direct_score = _paired_sign_scores(
        candidate["rows"], candidate["predictions"], [0.0] * len(candidate["rows"])
    )["candidate"]
    if not math.isclose(candidate_sign, direct_score, rel_tol=0.0, abs_tol=1e-12):
        raise ABContinuationError("candidate macro sign differs from exact prediction replay")

    decision["bindings"] = {
        "run_start_sha256": selector["run_start_sha256"],
        "activation_manifest_sha256": selector["activation_manifest_sha256"],
        "train_development_outcomes_sha256": selector[
            "train_development_outcomes_sha256"
        ],
        "non_data_authority_hashes": selector["non_data_authority_hashes"],
        "information_set_fingerprint": selector["information_set_fingerprint"],
        "registered_cell_ids_sha256": _canonical_sha256(
            candidate["registered_cell_ids"]
        ),
        "ordered_row_ids_sha256": candidate["ordered_row_ids_sha256"],
        "row_set_sha256": candidate["row_set_sha256"],
        "component_assignment_sha256": candidate["component_assignment_sha256"],
    }
    if route == "B":
        for field in (
            "registered_cell_ids_sha256",
            "ordered_row_ids_sha256",
            "row_set_sha256",
            "component_assignment_sha256",
        ):
            if decision["bindings"][field] != a_decision["bindings"][field]:
                raise ABContinuationError(
                    f"B evidence {field} differs from the exact A GO row universe"
                )
    decision["registered_cell_ids"] = list(candidate["registered_cell_ids"])
    decision["n_registered_cells"] = len(candidate["registered_cell_ids"])
    decision["candidate_macro_sign_accuracy"] = candidate_sign
    decision["sign_gate_met"] = candidate_sign >= SIGN_ACCURACY_GATE
    decision["verified_five_seed_roles"] = list(roles)
    decision["verified_model_binding_roles"] = list(roles)
    checks.update({
        "evidence_complete": True,
        "same_information_set": True,
        "same_registered_cells": vector_checks["registered_cell_ids"],
        "same_row_set_and_order": vector_checks["ordered_row_ids_sha256"]
        and vector_checks["row_set_sha256"],
        "same_component_assignment": vector_checks["component_assignment_sha256"],
        "five_seed_ensembles": True,
        "model_registry_config_bound": True,
        "parameter_match_qualified": True,
        "all_registered_cells_defined": True,
        "candidate_sign_gate_met": decision["sign_gate_met"],
    })
    try:
        family = _paired_family(
            route=route,
            comparisons=comparisons,
            bundles=bundles,
            comparator_bundle=comparator,
            selector=selector,
            policy=policy,
        )
    except ABContinuationError as exc:
        if not str(exc).startswith("UNDEFINED_PAIRED_INFERENCE:"):
            raise
        decision["reason_codes"] = [str(exc)]
        return validate_continuation_decision(decision)
    decision["paired_family"] = family
    by_id = {item["comparison_id"]: item["passed"] for item in family["results"]}
    prefix = route
    checks.update({
        "candidate_beats_selected_comparator": by_id[
            f"{prefix}_CANDIDATE_VS_SELECTED_COMPARATOR"
        ],
        "candidate_beats_mechanism_or_a_reference": by_id[
            f"{prefix}_CANDIDATE_VS_"
            + ("MECHANISM_REFERENCE" if route == "A" else "A_REFERENCE")
        ],
        "candidate_beats_matched_control": by_id[
            f"{prefix}_CANDIDATE_VS_MATCHED_CONTROL"
        ],
        "wrong_local_does_not_replicate": by_id[
            f"{prefix}_CANDIDATE_VS_WRONG_LOCAL_CONTROL"
        ],
        "unk_control_does_not_replicate": by_id[
            f"{prefix}_CANDIDATE_VS_UNK_CONTROL"
        ],
        "paired_family_qualified": family["all_qualified"],
    })
    required = [
        value for key, value in checks.items()
        if key != "a_prerequisite_go" or route == "B"
    ]
    scientific_go = all(value is True for value in required)
    if scientific_go:
        decision["status"] = GO_STATUS
        decision["decision"] = "GO"
        decision["reason_codes"] = []
    else:
        decision["status"] = FAIL_STATUS
        decision["decision"] = "FAIL"
        reasons = []
        if not decision["sign_gate_met"]:
            reasons.append("CANDIDATE_SIGN_ACCURACY_BELOW_0.60")
        reasons.extend(
            f"FAILED_PAIRED_COMPARISON:{item['comparison_id']}"
            for item in family["results"]
            if not item["passed"]
        )
        decision["reason_codes"] = reasons
    return validate_continuation_decision(decision)


def build_a_continuation_decision(
    run_root: Path,
    *,
    policy_ref: Mapping,
    selector_ref: Mapping,
    method_evidence_refs: Mapping[str, Mapping],
    model_binding_refs: Mapping[str, Mapping],
    neural_evidence_refs: Mapping[str, Mapping] | None = None,
    selector_context: Mapping,
    predictor_verifiers: Mapping[str, PredictorVerifier] | None,
) -> dict:
    return _build_route_decision(
        run_root,
        route="A",
        policy_ref=policy_ref,
        selector_ref=selector_ref,
        method_evidence_refs=method_evidence_refs,
        model_binding_refs=model_binding_refs,
        neural_evidence_refs=neural_evidence_refs,
        selector_context=selector_context,
        predictor_verifiers=predictor_verifiers,
    )


def build_b_continuation_decision(
    run_root: Path,
    *,
    policy_ref: Mapping,
    selector_ref: Mapping,
    a_decision_ref: Mapping,
    method_evidence_refs: Mapping[str, Mapping],
    model_binding_refs: Mapping[str, Mapping],
    neural_evidence_refs: Mapping[str, Mapping] | None = None,
    selector_context: Mapping,
    predictor_verifiers: Mapping[str, PredictorVerifier] | None,
    a_replay_context: Mapping,
) -> dict:
    a_decision = load_continuation_decision_exact(
        run_root, a_decision_ref, replay_context=a_replay_context
    )
    if a_decision["route"] != "A":
        raise ABContinuationError("B prerequisite reference is not an A decision")
    return _build_route_decision(
        run_root,
        route="B",
        policy_ref=policy_ref,
        selector_ref=selector_ref,
        method_evidence_refs=method_evidence_refs,
        model_binding_refs=model_binding_refs,
        neural_evidence_refs=neural_evidence_refs,
        selector_context=selector_context,
        predictor_verifiers=predictor_verifiers,
        a_decision_ref=a_decision_ref,
        a_decision=a_decision,
    )


def _validate_pair_result(value: Mapping, *, alpha: float) -> dict:
    _strict_keys(value, _PAIR_RESULT_KEYS, label="paired comparison result")
    for field in ("comparison_id", "left_role", "right_role", "left_method_id", "right_method_id"):
        _identifier(value.get(field), label=field)
    if value.get("rule") != POSITIVE_RULE:
        raise ABContinuationError("paired comparison rule changed")
    left = _finite(value.get("left_raw_sign_utility"), label="left sign utility")
    right = _finite(value.get("right_raw_sign_utility"), label="right sign utility")
    improvement = _finite(value.get("oriented_improvement"), label="oriented improvement")
    if not math.isclose(improvement, left - right, rel_tol=0.0, abs_tol=1e-12):
        raise ABContinuationError("paired improvement does not equal left minus right")
    ci = value.get("confidence_interval")
    if not isinstance(ci, list) or len(ci) != 2:
        raise ABContinuationError("paired confidence interval must have two endpoints")
    ci = [_finite(item, label="paired confidence endpoint") for item in ci]
    if ci[0] > ci[1]:
        raise ABContinuationError("paired confidence interval is reversed")
    raw_p = _finite(value.get("raw_p_value_two_sided"), label="raw p-value")
    adjusted = _finite(value.get("holm_adjusted_p_value"), label="Holm p-value")
    if not 0.0 <= raw_p <= 1.0 or not 0.0 <= adjusted <= 1.0:
        raise ABContinuationError("paired p-values must be probabilities")
    if _finite(value.get("alpha"), label="paired alpha") != alpha:
        raise ABContinuationError("paired result alpha differs from family alpha")
    bootstrap = value.get("bootstrap")
    permutation = value.get("permutation")
    if not isinstance(bootstrap, Mapping) or not isinstance(permutation, Mapping):
        raise ABContinuationError("paired result lacks recomputed statistics")
    if bootstrap.get("observed_oriented_improvement") != improvement:
        raise ABContinuationError("bootstrap observed improvement differs")
    if bootstrap.get("confidence_interval_oriented_improvement") != ci:
        raise ABContinuationError("bootstrap confidence interval differs")
    if permutation.get("observed_oriented_improvement") != improvement:
        raise ABContinuationError("permutation observed improvement differs")
    if permutation.get("p_value_two_sided") != raw_p:
        raise ABContinuationError("permutation p-value differs")
    passed = improvement > 0.0 and ci[0] > 0.0 and adjusted <= alpha
    if value.get("passed") is not passed:
        raise ABContinuationError("paired comparison pass is not recomputable")
    return dict(value)


def _validate_paired_family(value: Mapping) -> dict:
    _strict_keys(value, _PAIRED_FAMILY_KEYS, label="paired comparison family")
    _identifier(value.get("family_id"), label="paired family_id")
    if value.get("metric_id") != PRIMARY_PAIRED_METRIC_ID:
        raise ABContinuationError("paired family metric changed")
    order = value.get("comparison_order")
    if not isinstance(order, list) or not order or len(order) != len(set(order)):
        raise ABContinuationError("paired comparison order must be non-empty and unique")
    alpha = _finite(value.get("alpha"), label="paired family alpha")
    confidence = _finite(value.get("confidence"), label="paired family confidence")
    if not math.isclose(confidence, 1.0 - alpha, rel_tol=0.0, abs_tol=1e-12):
        raise ABContinuationError("paired family confidence differs from alpha")
    raw = value.get("raw_p_values_two_sided")
    adjusted = value.get("holm_adjusted_p_values")
    if not isinstance(raw, Mapping) or not isinstance(adjusted, Mapping):
        raise ABContinuationError("paired family p-values must be objects")
    if set(raw) != set(order) or set(adjusted) != set(order):
        raise ABContinuationError("paired family p-values do not match frozen membership")
    checked_raw = {key: _finite(raw[key], label=f"raw p-value {key}") for key in order}
    expected_adjusted = holm_adjust(checked_raw)
    if dict(adjusted) != {key: expected_adjusted[key] for key in order}:
        raise ABContinuationError("paired family Holm values are not recomputable")
    results = value.get("results")
    if not isinstance(results, list) or [item.get("comparison_id") for item in results] != order:
        raise ABContinuationError("paired results do not follow frozen family order")
    checked_results = [_validate_pair_result(item, alpha=alpha) for item in results]
    for item in checked_results:
        key = item["comparison_id"]
        if item["raw_p_value_two_sided"] != checked_raw[key]:
            raise ABContinuationError("paired result raw p-value differs from family")
        if item["holm_adjusted_p_value"] != expected_adjusted[key]:
            raise ABContinuationError("paired result Holm p-value differs from family")
    qualified = all(item["passed"] for item in checked_results)
    if value.get("all_qualified") is not qualified:
        raise ABContinuationError("paired family qualification is not recomputable")
    return dict(value)


def validate_continuation_decision(payload: Mapping) -> dict:
    """Validate internal consistency; exact-load additionally replays all refs."""

    _strict_keys(payload, _DECISION_KEYS, label="continuation decision")
    route = payload.get("route")
    if route not in {"A", "B"}:
        raise ABContinuationError("continuation route must be A or B")
    expected_schema = A_DECISION_SCHEMA if route == "A" else B_DECISION_SCHEMA
    expected_role = A_DECISION_ROLE if route == "A" else B_DECISION_ROLE
    if payload.get("schema_version") != expected_schema or payload.get("artifact_role") != expected_role:
        raise ABContinuationError("continuation decision schema/role mismatch")
    status = payload.get("status")
    decision = payload.get("decision")
    allowed = {
        GO_STATUS: "GO",
        FAIL_STATUS: "FAIL",
        HOLD_STATUS: "HOLD",
        HOLD_A_STATUS: "HOLD",
    }
    if allowed.get(status) != decision:
        raise ABContinuationError("continuation status/decision mismatch")
    reasons = payload.get("reason_codes")
    if not isinstance(reasons, list) or any(not isinstance(item, str) or not item for item in reasons):
        raise ABContinuationError("reason_codes must be a string list")
    if decision == "GO" and reasons:
        raise ABContinuationError("GO continuation cannot carry failure reasons")
    if decision != "GO" and not reasons:
        raise ABContinuationError("non-GO continuation requires reason codes")
    policy_ref = _development_ref(payload.get("policy_ref"), label="decision policy ref")
    if payload.get("policy_sha256") != policy_ref["sha256"]:
        raise ABContinuationError("decision policy SHA differs from exact ref")
    _development_ref(payload.get("selector_ref"), label="decision selector ref")
    _identifier(payload.get("selector_status"), label="selector_status")
    _method_id(payload.get("selected_comparator_method_id"), label="selected comparator")
    if payload.get("comparator_qualification") not in {"QUALIFIED", "RECOVERY_ONLY"}:
        raise ABContinuationError("comparator qualification is invalid")
    for field in (
        "method_evidence_refs",
        "model_binding_refs",
        "neural_evidence_refs",
    ):
        refs = payload.get(field)
        if not isinstance(refs, Mapping):
            raise ABContinuationError(f"{field} must be an object")
        for role, reference in refs.items():
            _identifier(role, label=f"{field} role")
            _development_ref(reference, label=f"{field}.{role}")
    if route == "A" and payload.get("a_decision_ref") is not None:
        raise ABContinuationError("A decision cannot reference another A decision")
    if route == "B" and payload.get("a_decision_ref") is not None:
        _development_ref(payload["a_decision_ref"], label="B prerequisite A decision")
    binding = payload.get("bindings")
    registered_cells = payload.get("registered_cell_ids")
    n_cells = payload.get("n_registered_cells")
    candidate_sign = payload.get("candidate_macro_sign_accuracy")
    sign_gate_met = payload.get("sign_gate_met")
    if binding is None:
        if any(value is not None for value in (registered_cells, n_cells, candidate_sign, sign_gate_met)):
            raise ABContinuationError("incomplete HOLD cannot carry partial decision metrics")
    else:
        _strict_keys(binding, _BINDING_KEYS, label="continuation bindings")
        for field in (
            "run_start_sha256",
            "activation_manifest_sha256",
            "train_development_outcomes_sha256",
            "information_set_fingerprint",
            "registered_cell_ids_sha256",
            "ordered_row_ids_sha256",
            "row_set_sha256",
            "component_assignment_sha256",
        ):
            _digest(binding.get(field), label=f"bindings.{field}")
        authorities = binding.get("non_data_authority_hashes")
        if not isinstance(authorities, Mapping) or not authorities:
            raise ABContinuationError("non-data authority bindings are empty")
        for name, value in authorities.items():
            _identifier(name, label="authority hash key")
            _digest(value, label=f"authority hash {name}")
        if not isinstance(registered_cells, list) or not registered_cells:
            raise ABContinuationError("registered cells must be a non-empty list")
        for cell in registered_cells:
            _identifier(cell, label="registered cell")
        if registered_cells != sorted(registered_cells) or len(
            registered_cells
        ) != len(set(registered_cells)):
            raise ABContinuationError("registered cells must be sorted and unique")
        if binding["registered_cell_ids_sha256"] != _canonical_sha256(registered_cells):
            raise ABContinuationError("registered-cell SHA is invalid")
        if n_cells != len(registered_cells):
            raise ABContinuationError("registered-cell count is invalid")
        sign = _finite(candidate_sign, label="candidate macro sign accuracy")
        if not 0.0 <= sign <= 1.0:
            raise ABContinuationError("candidate sign accuracy must be in [0,1]")
        if sign_gate_met is not (sign >= SIGN_ACCURACY_GATE):
            raise ABContinuationError("candidate sign-gate result is not recomputable")
    _method_id(payload.get("candidate_method_id"), label="candidate_method_id")
    if _finite(payload.get("sign_gate"), label="sign_gate") != SIGN_ACCURACY_GATE:
        raise ABContinuationError("decision changed the 0.60 sign gate")
    for field in (
        "verified_five_seed_roles",
        "verified_model_binding_roles",
        "verified_neural_receipt_roles",
    ):
        roles = payload.get(field)
        if not isinstance(roles, list) or len(roles) != len(set(roles)):
            raise ABContinuationError(f"{field} must be a unique list")
        for role in roles:
            _identifier(role, label=field)
    family = payload.get("paired_family")
    if family is not None:
        _validate_paired_family(family)
    checks = payload.get("scientific_checks")
    _strict_keys(checks, _CHECK_KEYS, label="scientific_checks")
    for key, value in checks.items():
        if value not in {None, True, False}:
            raise ABContinuationError(f"scientific check {key} must be boolean or null")
    if decision == "GO":
        required = [
            value for key, value in checks.items()
            if key != "a_prerequisite_go" or route == "B"
        ]
        if not all(value is True for value in required):
            raise ABContinuationError("GO continuation has an unmet scientific check")
        if family is None or family.get("all_qualified") is not True:
            raise ABContinuationError("GO continuation lacks a qualified paired family")
        role_order = A_ROLE_ORDER if route == "A" else B_ROLE_ORDER
        if payload.get("verified_neural_receipt_roles") != list(role_order):
            raise ABContinuationError(
                "GO continuation lacks every exact neural-state receipt"
            )
    if payload.get("b_interaction_status") != B_INTERACTION_STATUS:
        raise ABContinuationError("B interaction status changed")
    if payload.get("c3_status") != C3_STATUS or payload.get("c3_guidance_position") != C3_GUIDANCE_POSITION:
        raise ABContinuationError("C3 boundary changed")
    if payload.get("ordinary_development_only") is not True:
        raise ABContinuationError("continuation decision must remain development-only")
    if payload.get("internal_test_access") is not False:
        raise ABContinuationError("continuation decision cannot access internal test")
    if payload.get("sealed_access") is not False or payload.get("downstream_unlock") is not False:
        raise ABContinuationError("continuation decision cannot access sealed or unlock downstream")
    return dict(payload)


def canonical_continuation_decision_bytes(payload: Mapping) -> bytes:
    try:
        return canonical_json_bytes(validate_continuation_decision(payload))
    except ArtifactIOError as exc:
        raise ABContinuationError(str(exc)) from exc


def _rebuild_from_payload(
    run_root: Path,
    payload: Mapping,
    *,
    replay_context: Mapping,
) -> dict:
    expected = {"selector_context", "predictor_verifiers", "a_replay_context"}
    _strict_keys(replay_context, expected, label="continuation replay context")
    route = payload["route"]
    common = {
        "run_root": run_root,
        "policy_ref": payload["policy_ref"],
        "selector_ref": payload["selector_ref"],
        "method_evidence_refs": payload["method_evidence_refs"],
        "model_binding_refs": payload["model_binding_refs"],
        "neural_evidence_refs": payload["neural_evidence_refs"],
        "selector_context": replay_context["selector_context"],
        "predictor_verifiers": replay_context["predictor_verifiers"],
    }
    if route == "A":
        return build_a_continuation_decision(**common)
    if payload["a_decision_ref"] is None:
        raise ABContinuationError("B decision lacks exact A prerequisite reference")
    a_context = replay_context.get("a_replay_context")
    if not isinstance(a_context, Mapping):
        raise ABContinuationError("B decision requires A replay context")
    return build_b_continuation_decision(
        **common,
        a_decision_ref=payload["a_decision_ref"],
        a_replay_context=a_context,
    )


def write_continuation_decision_exclusive(
    run_root: Path,
    path: Path,
    payload: Mapping,
    *,
    replay_context: Mapping,
) -> str:
    _assert_development_output(Path(path), label="continuation decision output")
    _assert_output_within_run_root(
        Path(run_root), Path(path), label="continuation decision output"
    )
    validated = validate_continuation_decision(payload)
    rebuilt = _rebuild_from_payload(run_root, validated, replay_context=replay_context)
    if rebuilt != validated:
        raise ABContinuationError("continuation decision differs from exact evidence replay")
    try:
        return atomic_write_json_exclusive(
            Path(path), rebuilt, validator=validate_continuation_decision
        )
    except ArtifactIOError as exc:
        raise ABContinuationError(str(exc)) from exc


def load_continuation_decision_exact(
    run_root: Path,
    reference: Mapping,
    *,
    replay_context: Mapping,
) -> dict:
    raw = _read_ref(run_root, reference, label="continuation decision reference")
    payload = validate_continuation_decision(
        _parse_json_object(raw, label="continuation decision")
    )
    if canonical_continuation_decision_bytes(payload) != raw:
        raise ABContinuationError("continuation decision bytes are not canonical")
    rebuilt = _rebuild_from_payload(run_root, payload, replay_context=replay_context)
    if rebuilt != payload:
        raise ABContinuationError("continuation decision does not match exact replay")
    return payload


__all__ = [
    "ABContinuationError",
    "A_DECISION_SCHEMA",
    "B_DECISION_SCHEMA",
    "B_INTERACTION_STATUS",
    "C3_GUIDANCE_POSITION",
    "C3_STATUS",
    "FAIL_STATUS",
    "GO_STATUS",
    "HOLD_A_STATUS",
    "HOLD_STATUS",
    "MODEL_BINDING_SCHEMA",
    "POLICY_SCHEMA",
    "build_a_continuation_decision",
    "build_b_continuation_decision",
    "build_model_binding_evidence",
    "canonical_continuation_decision_bytes",
    "canonical_continuation_policy_bytes",
    "canonical_model_binding_evidence_bytes",
    "load_continuation_decision_exact",
    "load_continuation_policy_exact",
    "load_model_binding_evidence_exact",
    "validate_continuation_decision",
    "validate_continuation_policy",
    "validate_model_binding_evidence",
    "write_continuation_decision_exclusive",
    "write_continuation_policy_exclusive",
    "write_model_binding_evidence_exclusive",
]
