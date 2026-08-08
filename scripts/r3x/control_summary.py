"""Evidence-replayed gate for all preregistered R3X negative controls.

The summary is deliberately not a bag of self-attested booleans.  A PASS or
FAIL result must point at immutable evidence bytes and must be independently
replayed by a caller-supplied verifier.  Training-data perturbations additionally
bind the exact train-only transform manifest.  The verifier must return the
ordinary-internal-test row binding it observed, so changing the evaluation set
cannot be hidden behind an otherwise plausible result.

This artifact is ordinary-only.  It never authorizes sealed access or unlocks a
downstream phase.  Its strongest possible status is therefore only
``PASS_ALL_PREREGISTERED_CONTROLS`` within the ordinary run.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .action_support import ActionSupportHoldError, validate_action_support_policy
from .artifact_graph import (
    ArtifactGraphError,
    read_artifact_ref,
    validate_artifact_ref,
)
from .artifact_io import (
    ArtifactIOError,
    atomic_write_json_exclusive,
    canonical_json_bytes,
    read_exact_regular_file,
)
from .checkpoint import FROZEN_SEEDS
from .protocol import ProtocolHoldError, assert_ordinary_path


SCHEMA = "r3x_negative_control_summary_v1"
ARTIFACT_ROLE = "ORDINARY_PREREGISTERED_NEGATIVE_CONTROL_GATE"
PASS_ALL = "PASS_ALL_PREREGISTERED_CONTROLS"
HOLD = "HOLD"

ACTION_SUPPORT_QUALIFIED = "PUBLICATION_QUALIFIED"
ACTION_SUPPORT_NOT_QUALIFIED = "NOT_QUALIFIED"

PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT_RUN"
UNDEFINED = "UNDEFINED"
RESULT_OUTCOMES = {PASS, FAIL, NOT_RUN, UNDEFINED}
ENTRY_STATUSES = {PASS, FAIL, HOLD, "NOT_APPLICABLE"}

TRAIN_DATA_PERTURBATION = "TRAIN_DATA_PERTURBATION"
ALGEBRAIC_INVARIANT = "ALGEBRAIC_INVARIANT"
QUALIFIED_MODEL_COMPARISON = "QUALIFIED_MODEL_COMPARISON"

EVIDENCE_SCOPE_BY_CLASS = {
    TRAIN_DATA_PERTURBATION: (
        "PRIMARY_VS_CONTROL_PAIRED_DEVELOPMENT_AND_ORDINARY_INTERNAL_TEST_RECOMPUTED"
    ),
    ALGEBRAIC_INVARIANT: "A_IDENTITY_AND_SWAP_ALGEBRAIC_INVARIANTS_RECOMPUTED",
    QUALIFIED_MODEL_COMPARISON: (
        "B_INTERACTION_VS_ADDITIVE_ELIGIBLE_CELL_AND_PAIRED_INFERENCE_RECOMPUTED"
    ),
}
VERIFIED_PARTITIONS_BY_CLASS = {
    TRAIN_DATA_PERTURBATION: [
        "ORDINARY_DEVELOPMENT",
        "ORDINARY_INTERNAL_TEST",
    ],
    ALGEBRAIC_INVARIANT: [],
    QUALIFIED_MODEL_COMPARISON: ["ORDINARY_INTERNAL_TEST"],
}

CONTROL_IDS = (
    "SOURCE_CANDIDATE_PAIRING_SHUFFLE",
    "CONTEXT_LABEL_PERMUTATION",
    "A_IDENTITY_AND_SWAP_INVARIANTS",
    "SOURCE_CONTEXT_SHUFFLE_WITHIN_ACTION",
    "EDIT_POSITION_SHUFFLE_WITHIN_SOURCE",
    "TARGET_TOKEN_SHUFFLE_WITHIN_CELL",
    "B_ADDITIVE_VERSUS_INTERACTION",
)

FULL_PAIR = "FULL_PAIR_UNRESTRICTED_SIAMESE"
R2_STYLE = "R2_STYLE_NO_EXPLICIT_STUDY_ADAPTATION"
A_EXACT = "A_EXACT_POTENTIAL_DIFFERENCE"
A_MATCHED = "A_PARAMETER_MATCHED_UNRESTRICTED_CONTROL"
B_ADDITIVE = "B_SPARSE_EDIT_SET_ADDITIVE"
B_INTERACTION = "B_SPARSE_EDIT_SET_INTERACTION"

DATA_CRITERION = (
    "UNPERTURBED_METHOD_OUTPERFORMS_TRAIN_PERTURBATION_ON_FROZEN_PRIMARY_UTILITY_"
    "IN_DEVELOPMENT_AND_ORDINARY_INTERNAL_TEST_WITH_PAIRED_COMPONENT_INFERENCE"
)

# The ordering is part of the artifact schema and follows the frozen method
# protocol rather than lexical order.
CONTROL_SPECS = {
    "SOURCE_CANDIDATE_PAIRING_SHUFFLE": {
        "control_class": TRAIN_DATA_PERTURBATION,
        "base_methods": (FULL_PAIR, R2_STYLE, A_EXACT, A_MATCHED),
        "optional_interaction": False,
        "outcome_criterion": DATA_CRITERION,
        "train_only_invariant": "CONTROL_TRANSFORM_INPUT_AND_OUTPUT_ARE_FROZEN_TRAIN_ROWS_ONLY",
    },
    "CONTEXT_LABEL_PERMUTATION": {
        "control_class": TRAIN_DATA_PERTURBATION,
        "base_methods": (FULL_PAIR, R2_STYLE, A_EXACT, A_MATCHED, B_ADDITIVE),
        "optional_interaction": True,
        "outcome_criterion": DATA_CRITERION,
        "train_only_invariant": "CONTROL_TRANSFORM_INPUT_AND_OUTPUT_ARE_FROZEN_TRAIN_ROWS_ONLY",
    },
    "A_IDENTITY_AND_SWAP_INVARIANTS": {
        "control_class": ALGEBRAIC_INVARIANT,
        "base_methods": (A_EXACT,),
        "optional_interaction": False,
        "outcome_criterion": (
            "IDENTITY_MEAN_IS_EXACTLY_ZERO_AND_SWAP_MEAN_IS_EXACTLY_ANTISYMMETRIC_"
            "AND_SWAP_LOGVAR_IS_EXACTLY_SYMMETRIC"
        ),
        "train_only_invariant": "NOT_APPLICABLE_ALGEBRAIC_INVARIANT",
    },
    "SOURCE_CONTEXT_SHUFFLE_WITHIN_ACTION": {
        "control_class": TRAIN_DATA_PERTURBATION,
        "base_methods": (B_ADDITIVE,),
        "optional_interaction": True,
        "outcome_criterion": DATA_CRITERION,
        "train_only_invariant": "CONTROL_TRANSFORM_INPUT_AND_OUTPUT_ARE_FROZEN_TRAIN_ROWS_ONLY",
    },
    "EDIT_POSITION_SHUFFLE_WITHIN_SOURCE": {
        "control_class": TRAIN_DATA_PERTURBATION,
        "base_methods": (B_ADDITIVE,),
        "optional_interaction": True,
        "outcome_criterion": DATA_CRITERION,
        "train_only_invariant": "CONTROL_TRANSFORM_INPUT_AND_OUTPUT_ARE_FROZEN_TRAIN_ROWS_ONLY",
    },
    "TARGET_TOKEN_SHUFFLE_WITHIN_CELL": {
        "control_class": TRAIN_DATA_PERTURBATION,
        "base_methods": (B_ADDITIVE,),
        "optional_interaction": True,
        "outcome_criterion": DATA_CRITERION,
        "train_only_invariant": "CONTROL_TRANSFORM_INPUT_AND_OUTPUT_ARE_FROZEN_TRAIN_ROWS_ONLY",
    },
    "B_ADDITIVE_VERSUS_INTERACTION": {
        "control_class": QUALIFIED_MODEL_COMPARISON,
        "base_methods": (),
        "optional_interaction": True,
        "outcome_criterion": (
            "PUBLICATION_QUALIFIED_INTERACTION_OUTPERFORMS_ADDITIVE_IN_EVERY_"
            "ELIGIBLE_CELL_WITH_FROZEN_PAIRED_COMPONENT_INFERENCE"
        ),
        "train_only_invariant": "NOT_APPLICABLE_MODEL_COMPARISON",
    },
}

HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_REASON = re.compile(r"^[A-Z][A-Z0-9_]{2,127}$")

SUMMARY_KEYS = {
    "schema_version",
    "artifact_role",
    "method_protocol_ref",
    "action_support_status",
    "action_support_policy_ref",
    "ordinary_internal_test_row_binding_sha256",
    "controls",
    "overall_status",
    "ordinary_only",
    "sealed_access",
    "downstream_unlock",
}
CONTROL_KEYS = {
    "control_id",
    "control_class",
    "required_method_ids",
    "conditionally_excluded_method_ids",
    "outcome_criterion",
    "train_only_invariant",
    "train_control_manifest_ref",
    "method_results",
    "status",
}
RESULT_KEYS = {
    "method_id",
    "outcome",
    "evidence_ref",
    "ordinary_internal_test_row_binding_sha256",
    "reason_code",
}
VERIFIER_RESULT_KEYS = {
    "control_id",
    "method_id",
    "outcome",
    "ordinary_internal_test_row_binding_sha256",
    "train_only_verified",
    "train_control_manifest_sha256",
    "action_support_policy_sha256",
    "verification_scope",
    "verified_partitions",
}

RUN_MANIFEST_KEYS = {
    "schema_version",
    "name",
    "scope",
    "seed",
    "n_rows",
    "ordered_input_row_ids_sha256",
    "input_sha256",
    "output_sha256",
    "partition_validation",
    "test_partition_modified",
    "sealed_access",
}
NOT_RUN_MANIFEST_KEYS = {
    "schema_version",
    "name",
    "scope",
    "seed",
    "n_rows",
    "ordered_input_row_ids_sha256",
    "input_sha256",
    "partition_validation",
    "status",
    "reason",
    "sealed_access",
}

EvidenceVerifier = Callable[[bytes, Mapping[str, object]], Mapping[str, object]]


class NegativeControlProtocolError(RuntimeError):
    """Fail-closed negative-control schema or evidence-replay error."""


def _strict_keys(value: Mapping, expected: set[str], *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise NegativeControlProtocolError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise NegativeControlProtocolError(
            f"{label} keys differ from strict schema: {sorted(actual ^ expected)}"
        )


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise NegativeControlProtocolError(f"{label} must be 64 lowercase hex")
    return value


def _optional_digest(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return _digest(value, label=label)


def _reference(value: object, *, label: str) -> dict | None:
    if value is None:
        return None
    try:
        return validate_artifact_ref(value)  # type: ignore[arg-type]
    except ArtifactGraphError as exc:
        raise NegativeControlProtocolError(f"{label}: {exc}") from exc


def _expected_methods(control_id: str, action_support_status: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    spec = CONTROL_SPECS[control_id]
    required = tuple(spec["base_methods"])
    excluded: tuple[str, ...] = ()
    if bool(spec["optional_interaction"]):
        if action_support_status == ACTION_SUPPORT_QUALIFIED:
            required = (*required, B_INTERACTION)
        else:
            excluded = (B_INTERACTION,)
    return required, excluded


def _entry_status(required: Sequence[str], outcomes: Sequence[str]) -> str:
    if not required:
        return "NOT_APPLICABLE"
    if any(value == FAIL for value in outcomes):
        return FAIL
    if any(value in {NOT_RUN, UNDEFINED} for value in outcomes):
        return HOLD
    if outcomes and all(value == PASS for value in outcomes):
        return PASS
    return HOLD


def _summary_status(entries: Sequence[Mapping]) -> str:
    return (
        PASS_ALL
        if all(entry["status"] in {PASS, "NOT_APPLICABLE"} for entry in entries)
        else HOLD
    )


def _normalize_result(
    value: Mapping,
    *,
    control_id: str,
    method_id: str,
    control_class: str,
    expected_test_binding: str,
) -> dict:
    _strict_keys(value, RESULT_KEYS, label=f"{control_id}.{method_id} result")
    if value.get("method_id") != method_id:
        raise NegativeControlProtocolError(
            f"{control_id} result method_id order/binding differs"
        )
    outcome = value.get("outcome")
    if outcome not in RESULT_OUTCOMES:
        raise NegativeControlProtocolError(f"{control_id}.{method_id} outcome is invalid")
    evidence_ref = _reference(
        value.get("evidence_ref"), label=f"{control_id}.{method_id}.evidence_ref"
    )
    binding = _optional_digest(
        value.get("ordinary_internal_test_row_binding_sha256"),
        label=f"{control_id}.{method_id} internal-test row binding",
    )
    reason = value.get("reason_code")

    if outcome in {PASS, FAIL}:
        if evidence_ref is None:
            raise NegativeControlProtocolError(
                f"{control_id}.{method_id} {outcome} requires immutable evidence_ref"
            )
        if reason is not None:
            raise NegativeControlProtocolError(
                f"{control_id}.{method_id} completed outcome cannot carry reason_code"
            )
        if control_class == ALGEBRAIC_INVARIANT:
            if binding is not None:
                raise NegativeControlProtocolError(
                    "algebraic invariant evidence must not pretend to bind test rows"
                )
        elif binding != expected_test_binding:
            raise NegativeControlProtocolError(
                f"{control_id}.{method_id} changed ordinary internal-test row binding"
            )
    else:
        if not isinstance(reason, str) or not SAFE_REASON.fullmatch(reason):
            raise NegativeControlProtocolError(
                f"{control_id}.{method_id} {outcome} requires a stable reason_code"
            )
        # Evidence may document an undefined or interrupted calculation, but it
        # can never convert that state into PASS.
        if control_class == ALGEBRAIC_INVARIANT:
            if binding is not None:
                raise NegativeControlProtocolError(
                    "algebraic invariant non-result cannot bind test rows"
                )
        elif binding not in {None, expected_test_binding}:
            raise NegativeControlProtocolError(
                f"{control_id}.{method_id} non-result changed internal-test rows"
            )

    return {
        "method_id": method_id,
        "outcome": str(outcome),
        "evidence_ref": evidence_ref,
        "ordinary_internal_test_row_binding_sha256": binding,
        "reason_code": reason,
    }


def validate_negative_control_summary(payload: Mapping) -> dict:
    """Validate exact schema and recompute every declared gate status."""

    _strict_keys(payload, SUMMARY_KEYS, label="negative-control summary")
    if payload.get("schema_version") != SCHEMA:
        raise NegativeControlProtocolError("unsupported negative-control summary schema")
    if payload.get("artifact_role") != ARTIFACT_ROLE:
        raise NegativeControlProtocolError("negative-control artifact role is invalid")
    if payload.get("ordinary_only") is not True:
        raise NegativeControlProtocolError("negative controls must remain ordinary-only")
    if payload.get("sealed_access") is not False:
        raise NegativeControlProtocolError("negative controls cannot access sealed data")
    if payload.get("downstream_unlock") is not False:
        raise NegativeControlProtocolError("negative controls cannot unlock downstream phases")

    method_protocol_ref = _reference(
        payload.get("method_protocol_ref"), label="method_protocol_ref"
    )
    if method_protocol_ref is None:
        raise NegativeControlProtocolError("method_protocol_ref is required")
    action_status = payload.get("action_support_status")
    if action_status not in {ACTION_SUPPORT_QUALIFIED, ACTION_SUPPORT_NOT_QUALIFIED}:
        raise NegativeControlProtocolError("action_support_status is invalid")
    action_ref = _reference(
        payload.get("action_support_policy_ref"), label="action_support_policy_ref"
    )
    if action_status == ACTION_SUPPORT_QUALIFIED and action_ref is None:
        raise NegativeControlProtocolError(
            "publication-qualified B interaction requires action-support evidence"
        )
    if action_status == ACTION_SUPPORT_NOT_QUALIFIED and action_ref is not None:
        raise NegativeControlProtocolError(
            "unqualified B interaction cannot carry a qualifying policy reference"
        )

    test_binding = _digest(
        payload.get("ordinary_internal_test_row_binding_sha256"),
        label="ordinary_internal_test_row_binding_sha256",
    )
    raw_controls = payload.get("controls")
    if isinstance(raw_controls, (str, bytes)) or not isinstance(raw_controls, Sequence):
        raise NegativeControlProtocolError("controls must be the ordered seven-item sequence")
    if len(raw_controls) != len(CONTROL_IDS):
        raise NegativeControlProtocolError("controls must contain exactly seven entries")

    controls: list[dict] = []
    for index, control_id in enumerate(CONTROL_IDS):
        raw = raw_controls[index]
        _strict_keys(raw, CONTROL_KEYS, label=f"control[{index}]")
        if raw.get("control_id") != control_id:
            raise NegativeControlProtocolError(
                "control order/identity differs from the frozen method protocol"
            )
        spec = CONTROL_SPECS[control_id]
        if raw.get("control_class") != spec["control_class"]:
            raise NegativeControlProtocolError(f"{control_id} class differs from policy")
        if raw.get("outcome_criterion") != spec["outcome_criterion"]:
            raise NegativeControlProtocolError(f"{control_id} criterion differs from policy")
        if raw.get("train_only_invariant") != spec["train_only_invariant"]:
            raise NegativeControlProtocolError(
                f"{control_id} train-only invariant differs from policy"
            )

        required, excluded = _expected_methods(control_id, str(action_status))
        if raw.get("required_method_ids") != list(required):
            raise NegativeControlProtocolError(
                f"{control_id} required method applicability differs from policy"
            )
        if raw.get("conditionally_excluded_method_ids") != list(excluded):
            raise NegativeControlProtocolError(
                f"{control_id} conditional method applicability differs from policy"
            )

        manifest_ref = _reference(
            raw.get("train_control_manifest_ref"),
            label=f"{control_id}.train_control_manifest_ref",
        )
        if spec["control_class"] != TRAIN_DATA_PERTURBATION and manifest_ref is not None:
            raise NegativeControlProtocolError(
                f"{control_id} cannot carry a training perturbation manifest"
            )

        raw_results = raw.get("method_results")
        if isinstance(raw_results, (str, bytes)) or not isinstance(raw_results, Sequence):
            raise NegativeControlProtocolError(f"{control_id}.method_results must be a sequence")
        if len(raw_results) != len(required):
            raise NegativeControlProtocolError(
                f"{control_id}.method_results must exactly cover applicable methods"
            )
        results = [
            _normalize_result(
                raw_results[result_index],
                control_id=control_id,
                method_id=method_id,
                control_class=str(spec["control_class"]),
                expected_test_binding=test_binding,
            )
            for result_index, method_id in enumerate(required)
        ]
        if (
            spec["control_class"] == TRAIN_DATA_PERTURBATION
            and any(item["outcome"] in {PASS, FAIL} for item in results)
            and manifest_ref is None
        ):
            raise NegativeControlProtocolError(
                f"{control_id} completed result requires a frozen train-only manifest"
            )
        computed_status = _entry_status(required, [item["outcome"] for item in results])
        if raw.get("status") not in ENTRY_STATUSES or raw.get("status") != computed_status:
            raise NegativeControlProtocolError(
                f"{control_id} status is not the recomputed result status {computed_status}"
            )
        controls.append(
            {
                "control_id": control_id,
                "control_class": spec["control_class"],
                "required_method_ids": list(required),
                "conditionally_excluded_method_ids": list(excluded),
                "outcome_criterion": spec["outcome_criterion"],
                "train_only_invariant": spec["train_only_invariant"],
                "train_control_manifest_ref": manifest_ref,
                "method_results": results,
                "status": computed_status,
            }
        )

    overall = _summary_status(controls)
    if payload.get("overall_status") != overall:
        raise NegativeControlProtocolError(
            f"overall_status is not the recomputed status {overall}"
        )
    return {
        "schema_version": SCHEMA,
        "artifact_role": ARTIFACT_ROLE,
        "method_protocol_ref": method_protocol_ref,
        "action_support_status": action_status,
        "action_support_policy_ref": action_ref,
        "ordinary_internal_test_row_binding_sha256": test_binding,
        "controls": controls,
        "overall_status": overall,
        "ordinary_only": True,
        "sealed_access": False,
        "downstream_unlock": False,
    }


def build_not_run_negative_control_summary(
    *,
    method_protocol_ref: Mapping,
    ordinary_internal_test_row_binding_sha256: str,
    reason_code: str,
    action_support_status: str = ACTION_SUPPORT_NOT_QUALIFIED,
    action_support_policy_ref: Mapping | None = None,
    train_control_manifest_refs: Mapping[str, Mapping] | None = None,
) -> dict:
    """Build an explicit HOLD when preregistered controls were not executed.

    This is the integration path for a runner that has completed the ordinary
    prediction artifacts but has not run the control variants.  Missing
    perturbation manifests are representable only because every applicable
    result is ``NOT_RUN``; they can never be upgraded to PASS in this artifact.
    A later control execution must publish a new, no-overwrite summary.
    """

    if not isinstance(reason_code, str) or not SAFE_REASON.fullmatch(reason_code):
        raise NegativeControlProtocolError("NOT_RUN summary requires a stable reason_code")
    manifest_refs = dict(train_control_manifest_refs or {})
    unknown = set(manifest_refs) - set(CONTROL_IDS)
    if unknown:
        raise NegativeControlProtocolError(
            f"unknown train-control manifest IDs: {sorted(unknown)}"
        )
    controls = []
    for control_id in CONTROL_IDS:
        spec = CONTROL_SPECS[control_id]
        required, excluded = _expected_methods(control_id, action_support_status)
        results = [
            {
                "method_id": method_id,
                "outcome": NOT_RUN,
                "evidence_ref": None,
                "ordinary_internal_test_row_binding_sha256": None,
                "reason_code": reason_code,
            }
            for method_id in required
        ]
        controls.append(
            {
                "control_id": control_id,
                "control_class": spec["control_class"],
                "required_method_ids": list(required),
                "conditionally_excluded_method_ids": list(excluded),
                "outcome_criterion": spec["outcome_criterion"],
                "train_only_invariant": spec["train_only_invariant"],
                "train_control_manifest_ref": (
                    manifest_refs.get(control_id)
                    if spec["control_class"] == TRAIN_DATA_PERTURBATION
                    else None
                ),
                "method_results": results,
                "status": _entry_status(
                    required, [result["outcome"] for result in results]
                ),
            }
        )
    payload = {
        "schema_version": SCHEMA,
        "artifact_role": ARTIFACT_ROLE,
        "method_protocol_ref": dict(method_protocol_ref),
        "action_support_status": action_support_status,
        "action_support_policy_ref": (
            None if action_support_policy_ref is None else dict(action_support_policy_ref)
        ),
        "ordinary_internal_test_row_binding_sha256": (
            ordinary_internal_test_row_binding_sha256
        ),
        "controls": controls,
        "overall_status": _summary_status(controls),
        "ordinary_only": True,
        "sealed_access": False,
        "downstream_unlock": False,
    }
    return validate_negative_control_summary(payload)


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(raw: bytes, *, label: str) -> dict:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_json_object)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise NegativeControlProtocolError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise NegativeControlProtocolError(f"{label} must be a JSON object")
    return value


def _validate_method_protocol(raw: bytes) -> dict:
    payload = _parse_json(raw, label="method protocol")
    if payload.get("schema_version") != "r3_ordinary_experiment_protocol_v1":
        raise NegativeControlProtocolError("method protocol schema is unsupported")
    if payload.get("negative_controls") != list(CONTROL_IDS):
        raise NegativeControlProtocolError(
            "method protocol does not bind the exact seven negative controls"
        )
    boundaries = payload.get("hard_boundaries")
    if not isinstance(boundaries, Mapping):
        raise NegativeControlProtocolError("method protocol lacks hard boundaries")
    if boundaries.get("ordinary_only") is not True:
        raise NegativeControlProtocolError("method protocol is not ordinary-only")
    if boundaries.get("sealed_member_access") is not False:
        raise NegativeControlProtocolError("method protocol permits sealed access")
    if boundaries.get("O0_G1_E0_unlock") is not False:
        raise NegativeControlProtocolError("method protocol permits downstream unlock")
    return dict(payload)


def _validate_train_manifest(raw: bytes, *, control_id: str) -> dict:
    payload = _parse_json(raw, label=f"{control_id} train-control manifest")
    is_not_run = payload.get("status") == "NOT_RUN_WITH_REASON"
    _strict_keys(
        payload,
        NOT_RUN_MANIFEST_KEYS if is_not_run else RUN_MANIFEST_KEYS,
        label=f"{control_id} train-control manifest",
    )
    if payload.get("schema_version") != "R3X_NEGATIVE_CONTROL_MANIFEST_V1":
        raise NegativeControlProtocolError(f"{control_id} manifest schema is unsupported")
    if payload.get("name") != control_id:
        raise NegativeControlProtocolError(f"{control_id} manifest name differs")
    if payload.get("scope") != "ORDINARY_TRAIN_PARTITION_ONLY":
        raise NegativeControlProtocolError(f"{control_id} manifest is not train-only")
    if payload.get("sealed_access") is not False:
        raise NegativeControlProtocolError(f"{control_id} manifest accessed sealed data")
    seed = payload.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in FROZEN_SEEDS:
        raise NegativeControlProtocolError(f"{control_id} manifest seed is not frozen")
    n_rows = payload.get("n_rows")
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows < 1:
        raise NegativeControlProtocolError(f"{control_id} manifest row count is invalid")
    _digest(payload.get("ordered_input_row_ids_sha256"), label="ordered train row IDs")
    _digest(payload.get("input_sha256"), label="train-control input_sha256")
    if is_not_run:
        if payload.get("partition_validation") != "ALL_INPUT_ROWS_EQUAL_TRAIN":
            raise NegativeControlProtocolError(f"{control_id} NOT_RUN manifest is not train-only")
        if not isinstance(payload.get("reason"), str) or not payload.get("reason"):
            raise NegativeControlProtocolError(f"{control_id} NOT_RUN manifest lacks reason")
    else:
        _digest(payload.get("output_sha256"), label="train-control output_sha256")
        if payload.get("partition_validation") != "ALL_INPUT_AND_OUTPUT_ROWS_EQUAL_TRAIN":
            raise NegativeControlProtocolError(f"{control_id} transform escaped train-only scope")
        if payload.get("test_partition_modified") is not False:
            raise NegativeControlProtocolError(f"{control_id} modified the test partition")
    return dict(payload)


def _validate_verifier_result(
    value: Mapping,
    *,
    control: Mapping,
    result: Mapping,
    summary: Mapping,
    manifest_ref: Mapping | None,
    action_ref: Mapping | None,
) -> None:
    _strict_keys(value, VERIFIER_RESULT_KEYS, label="evidence verifier result")
    control_id = control["control_id"]
    method_id = result["method_id"]
    if value.get("control_id") != control_id or value.get("method_id") != method_id:
        raise NegativeControlProtocolError("evidence verifier returned the wrong control/method binding")
    if value.get("outcome") != result["outcome"]:
        raise NegativeControlProtocolError(
            f"evidence replay contradicts claimed outcome for {control_id}.{method_id}"
        )
    expected_scope = EVIDENCE_SCOPE_BY_CLASS[control["control_class"]]
    if value.get("verification_scope") != expected_scope:
        raise NegativeControlProtocolError(
            f"evidence verifier did not recompute the required scope for {control_id}.{method_id}"
        )
    if value.get("verified_partitions") != VERIFIED_PARTITIONS_BY_CLASS[
        control["control_class"]
    ]:
        raise NegativeControlProtocolError(
            f"evidence verifier did not replay the required partitions for {control_id}.{method_id}"
        )

    observed_binding = _optional_digest(
        value.get("ordinary_internal_test_row_binding_sha256"),
        label="verifier internal-test row binding",
    )
    if control["control_class"] == ALGEBRAIC_INVARIANT:
        if observed_binding is not None:
            raise NegativeControlProtocolError("algebraic verifier cannot claim test-row evidence")
    elif observed_binding != summary["ordinary_internal_test_row_binding_sha256"]:
        raise NegativeControlProtocolError(
            f"evidence replay detected modified internal-test rows for {control_id}.{method_id}"
        )

    manifest_sha = _optional_digest(
        value.get("train_control_manifest_sha256"),
        label="verifier train-control manifest SHA-256",
    )
    train_verified = value.get("train_only_verified")
    if control["control_class"] == TRAIN_DATA_PERTURBATION:
        if train_verified is not True or manifest_ref is None or manifest_sha != manifest_ref["sha256"]:
            raise NegativeControlProtocolError(
                f"evidence verifier did not prove train-only binding for {control_id}.{method_id}"
            )
    elif train_verified is not None or manifest_sha is not None:
        raise NegativeControlProtocolError("non-perturbation verifier returned a training manifest")

    action_sha = _optional_digest(
        value.get("action_support_policy_sha256"),
        label="verifier action-support policy SHA-256",
    )
    if method_id == B_INTERACTION:
        if action_ref is None or action_sha != action_ref["sha256"]:
            raise NegativeControlProtocolError(
                "B interaction evidence is not bound to publication-qualified action support"
            )
    elif action_sha is not None:
        raise NegativeControlProtocolError("non-interaction verifier returned action-support evidence")


def verify_negative_control_summary_references(
    run_root: Path,
    payload: Mapping,
    *,
    evidence_verifier: EvidenceVerifier | None,
    verifier_context: Mapping[str, object] | None = None,
) -> dict:
    """Read every reference exactly and independently replay all completed outcomes.

    ``evidence_verifier`` is mandatory for PASS/FAIL results.  It must return a
    structured result, not a boolean.  This makes the result source explicit and
    lets an integrating runner recompute metrics/invariants from its own context.
    """

    summary = validate_negative_control_summary(payload)
    try:
        method_protocol_raw = read_artifact_ref(run_root, summary["method_protocol_ref"])
    except ArtifactGraphError as exc:
        raise NegativeControlProtocolError(f"method protocol reference: {exc}") from exc
    method_protocol = _validate_method_protocol(method_protocol_raw)

    action_policy = None
    action_ref = summary["action_support_policy_ref"]
    if action_ref is not None:
        try:
            action_raw = read_artifact_ref(run_root, action_ref)
        except ArtifactGraphError as exc:
            raise NegativeControlProtocolError(f"action-support reference: {exc}") from exc
        action_payload = _parse_json(action_raw, label="action-support policy")
        try:
            action_policy = validate_action_support_policy(action_payload)
        except ActionSupportHoldError as exc:
            raise NegativeControlProtocolError(str(exc)) from exc
        if action_policy.get("status") != ACTION_SUPPORT_QUALIFIED:
            raise NegativeControlProtocolError("B interaction action support is not qualified")

    seen_paths = {summary["method_protocol_ref"]["relative_path"]}
    if action_ref is not None:
        if action_ref["relative_path"] in seen_paths:
            raise NegativeControlProtocolError("artifact reference path is reused")
        seen_paths.add(action_ref["relative_path"])

    for control in summary["controls"]:
        manifest = None
        manifest_raw = None
        manifest_ref = control["train_control_manifest_ref"]
        if manifest_ref is not None:
            if manifest_ref["relative_path"] in seen_paths:
                raise NegativeControlProtocolError("artifact reference path is reused")
            seen_paths.add(manifest_ref["relative_path"])
            try:
                manifest_raw = read_artifact_ref(run_root, manifest_ref)
            except ArtifactGraphError as exc:
                raise NegativeControlProtocolError(
                    f"{control['control_id']} manifest reference: {exc}"
                ) from exc
            manifest = _validate_train_manifest(
                manifest_raw, control_id=control["control_id"]
            )
            completed = any(
                result["outcome"] in {PASS, FAIL}
                for result in control["method_results"]
            )
            if completed and manifest.get("status") == "NOT_RUN_WITH_REASON":
                raise NegativeControlProtocolError(
                    f"{control['control_id']} claims completed evidence from a NOT_RUN transform"
                )

        for result in control["method_results"]:
            evidence_ref = result["evidence_ref"]
            evidence_raw = None
            if evidence_ref is not None:
                if evidence_ref["relative_path"] in seen_paths:
                    raise NegativeControlProtocolError("artifact reference path is reused")
                seen_paths.add(evidence_ref["relative_path"])
                try:
                    evidence_raw = read_artifact_ref(run_root, evidence_ref)
                except ArtifactGraphError as exc:
                    raise NegativeControlProtocolError(
                        f"{control['control_id']}.{result['method_id']} evidence reference: {exc}"
                    ) from exc

            if result["outcome"] in {PASS, FAIL}:
                if evidence_raw is None:
                    raise NegativeControlProtocolError("completed outcome lacks evidence bytes")
                if evidence_verifier is None:
                    raise NegativeControlProtocolError(
                        "PASS/FAIL negative-control evidence requires an independent verifier"
                    )
                metadata = {
                    "control_id": control["control_id"],
                    "control_class": control["control_class"],
                    "method_id": result["method_id"],
                    "outcome_criterion": control["outcome_criterion"],
                    "expected_outcome": result["outcome"],
                    "required_verification_scope": EVIDENCE_SCOPE_BY_CLASS[
                        control["control_class"]
                    ],
                    "required_verified_partitions": list(
                        VERIFIED_PARTITIONS_BY_CLASS[control["control_class"]]
                    ),
                    "ordinary_internal_test_row_binding_sha256": summary[
                        "ordinary_internal_test_row_binding_sha256"
                    ],
                    "train_control_manifest": manifest,
                    "train_control_manifest_bytes": manifest_raw,
                    "train_control_manifest_sha256": (
                        None if manifest_ref is None else manifest_ref["sha256"]
                    ),
                    "action_support_policy": action_policy,
                    "action_support_policy_sha256": (
                        None if action_ref is None else action_ref["sha256"]
                    ),
                    "method_protocol": method_protocol,
                    "verifier_context": (
                        None if verifier_context is None else dict(verifier_context)
                    ),
                }
                try:
                    verification = evidence_verifier(evidence_raw, metadata)
                except Exception as exc:
                    raise NegativeControlProtocolError(
                        f"evidence verifier failed for {control['control_id']}.{result['method_id']}: {exc}"
                    ) from exc
                if isinstance(verification, bool) or not isinstance(verification, Mapping):
                    raise NegativeControlProtocolError(
                        "evidence verifier must return a structured result, not a self-attested bool"
                    )
                _validate_verifier_result(
                    verification,
                    control=control,
                    result=result,
                    summary=summary,
                    manifest_ref=manifest_ref,
                    action_ref=action_ref,
                )
    return summary


def _checked_summary_path(run_root: Path, path: Path) -> Path:
    root = Path(run_root)
    candidate = Path(path)
    try:
        assert_ordinary_path(root, purpose="negative-control run root")
        assert_ordinary_path(candidate, purpose="negative-control summary")
    except ProtocolHoldError as exc:
        raise NegativeControlProtocolError(str(exc)) from exc
    if not root.is_dir() or root.is_symlink():
        raise NegativeControlProtocolError("negative-control run root must be a real directory")
    resolved_root = root.resolve(strict=True)
    try:
        resolved_parent = candidate.parent.resolve(strict=True)
    except FileNotFoundError as exc:
        raise NegativeControlProtocolError("summary parent directory does not exist") from exc
    if not resolved_parent.is_relative_to(resolved_root):
        raise NegativeControlProtocolError("summary path must remain inside the run root")
    if candidate.is_symlink():
        raise NegativeControlProtocolError("summary path cannot be a symlink")
    return candidate


def write_negative_control_summary_exclusive(
    path: Path,
    payload: Mapping,
    *,
    run_root: Path,
    evidence_verifier: EvidenceVerifier | None,
    verifier_context: Mapping[str, object] | None = None,
) -> str:
    """Replay evidence, then atomically publish without overwriting a path."""

    candidate = _checked_summary_path(run_root, path)
    normalized = verify_negative_control_summary_references(
        run_root,
        payload,
        evidence_verifier=evidence_verifier,
        verifier_context=verifier_context,
    )
    try:
        return atomic_write_json_exclusive(
            candidate, normalized, validator=validate_negative_control_summary
        )
    except ArtifactIOError as exc:
        raise NegativeControlProtocolError(str(exc)) from exc


def load_negative_control_summary(
    path: Path,
    expected_sha256: str,
    *,
    run_root: Path,
    evidence_verifier: EvidenceVerifier | None,
    verifier_context: Mapping[str, object] | None = None,
) -> dict:
    """Load exact canonical bytes and replay every evidence reference."""

    candidate = _checked_summary_path(run_root, path)
    digest = _digest(expected_sha256, label="negative-control summary SHA-256")
    try:
        raw = read_exact_regular_file(candidate, digest)
    except ArtifactIOError as exc:
        raise NegativeControlProtocolError(str(exc)) from exc
    parsed = _parse_json(raw, label="negative-control summary")
    normalized = validate_negative_control_summary(parsed)
    if canonical_json_bytes(normalized) != raw:
        raise NegativeControlProtocolError("negative-control summary bytes are not canonical")
    return verify_negative_control_summary_references(
        run_root,
        normalized,
        evidence_verifier=evidence_verifier,
        verifier_context=verifier_context,
    )
