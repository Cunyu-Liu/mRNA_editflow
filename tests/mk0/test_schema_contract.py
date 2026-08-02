"""Static schema/configuration checks for the frozen MK0-v1 interfaces."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from mrna_editflow.core.mk0.samplers import (
    constrained_single_event_first_order,
    sampler_result_to_schema_record,
)
from mrna_editflow.core.mk0.state_action import (
    action_to_schema_record,
    apply_action,
    state_to_schema_record,
    termination_to_schema_record,
    validate_schema_facing_record,
)
from mrna_editflow.core.mk0.alignment_coupling import (
    BLANK,
    build_alignment,
    reconstruct_alignment,
)
from mrna_editflow.core.mk0.types import (
    ActionType,
    AtomicAction,
    EditState,
    TerminationReason,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "schemas"

SCHEMA_FILES = (
    "edit_state_v1.schema.json",
    "edit_action_v1.schema.json",
    "edit_trajectory_v1.schema.json",
    "termination_event_v1.schema.json",
    "coupling_manifest_v1.schema.json",
)

TERMINATION_REASONS = {
    "LEARNED_STOP",
    "FORCED_BUDGET",
    "FORCED_NO_LEGAL_EDIT_ACTION",
    "FORCED_ZERO_REMAINING_INTEGRATED_HAZARD",
    "FORCED_TIME_HORIZON",
    "FAILED_NUMERICAL",
}


def _load(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def _walk(value):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def test_all_mandatory_schemas_are_strict_draft_2020_12_json() -> None:
    for name in SCHEMA_FILES:
        schema = _load(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert schema["required"]
        assert schema["$id"].endswith(name)


def test_edit_state_schema_excludes_training_only_target_fields() -> None:
    state = _load("edit_state_v1.schema.json")
    property_names = {str(key).lower() for key in state["properties"]}
    assert "target_condition" in property_names
    assert (
        not {
            "z_aux",
            "z_src",
            "z_t",
            "z_tar",
            "target_sequence",
            "target_alignment",
            "remaining_target_edits",
        }
        & property_names
    )
    assert state["properties"]["external_time"]["minimum"] == 0.0
    assert state["properties"]["external_time"]["maximum"] == 1.0
    assert state["properties"]["time_direction"]["const"] == (
        "source_at_0_to_target_at_1"
    )
    assert state["properties"]["run_state"]["enum"] == ["ACTIVE", "HALTED"]


def test_action_schema_freezes_atomic_budget_and_current_coordinates() -> None:
    action = _load("edit_action_v1.schema.json")
    assert action["properties"]["action_type"]["enum"] == [
        "INS",
        "SUB",
        "DEL",
        "STOP",
    ]
    assert action["properties"]["coordinate_system"]["const"] == (
        "pre_action_current_state_zero_based"
    )
    discriminants = {
        branch["if"]["properties"]["action_type"]["const"]: branch["then"]
        for branch in action["allOf"]
    }
    assert set(discriminants) == {"INS", "SUB", "DEL", "STOP"}
    assert discriminants["INS"]["properties"]["budget_cost"]["const"] == 1
    assert discriminants["SUB"]["properties"]["budget_cost"]["const"] == 1
    assert discriminants["DEL"]["properties"]["budget_cost"]["const"] == 1
    assert discriminants["STOP"]["properties"]["budget_cost"]["const"] == 0


def test_termination_schema_never_conflates_learned_forced_or_numerical() -> None:
    termination = _load("termination_event_v1.schema.json")
    assert set(termination["properties"]["reason"]["enum"]) == TERMINATION_REASONS
    assert termination["properties"]["consumes_edit_budget"]["const"] is False
    branches = termination["allOf"]
    learned = next(
        branch["then"]
        for branch in branches
        if branch["if"]["properties"]["reason"].get("const") == "LEARNED_STOP"
    )
    assert learned["properties"]["learned_stop"]["const"] is True
    assert learned["properties"]["forced_termination"]["const"] is False
    assert learned["properties"]["inside_ctmc"]["const"] is True
    numerical = next(
        branch["then"]
        for branch in branches
        if branch["if"]["properties"]["reason"].get("const") == "FAILED_NUMERICAL"
    )
    assert numerical["properties"]["learned_stop"]["const"] is False
    assert numerical["properties"]["forced_termination"]["const"] is False
    assert numerical["properties"]["inside_ctmc"]["const"] is False
    assert "diagnostic" in numerical["required"]


def test_trajectory_schema_is_replayable_and_never_claims_exact_gillespie() -> None:
    trajectory = _load("edit_trajectory_v1.schema.json")
    assert trajectory["properties"]["exact_gillespie"]["const"] is False
    assert set(trajectory["properties"]["sampler"]["enum"]) == {
        "paper_first_order_parallel",
        "constrained_single_event_first_order",
    }
    assert trajectory["properties"]["time_direction"]["const"] == (
        "source_at_0_to_target_at_1"
    )
    replay_required = set(trajectory["properties"]["replay"]["required"])
    assert {
        "status",
        "replayed_step_count",
        "state_hash_match_fraction",
    } <= replay_required
    step_required = set(trajectory["$defs"]["step"]["required"])
    assert {
        "t_start",
        "t_end",
        "substep_h",
        "total_hazard",
        "event_probability",
        "event_uniform",
        "outcome",
        "state_hash_before",
        "state_hash_after",
        "rate_recomputed_after_step",
        "adaptive_subdivision_count",
        "candidate_actions_hash",
        "candidate_rates_hash",
        "parallel_trials",
        "parallel_actions",
    } <= step_required
    assert {
        "sampler_config",
        "remaining_hazard_certificate",
    } <= set(trajectory["required"])


def test_coupling_schema_freezes_latent_product_path_and_isolation() -> None:
    coupling = _load("coupling_manifest_v1.schema.json")
    assert coupling["properties"]["path_is_observed"]["const"] is False
    assert coupling["properties"]["path_semantics"]["const"] == ("latent_algorithmic")
    assert coupling["properties"]["joint_path"]["const"] == (
        "independent_switch_clock_product"
    )
    isolation = coupling["properties"]["target_auxiliary_isolation"]["properties"]
    assert isolation["z_aux_training_only"]["const"] is True
    assert isolation["rate_network_receives_z_aux"]["const"] is False
    assert coupling["properties"]["schedule"]["properties"]["rho"]["const"] == (
        "kappa_derivative/(1-kappa)"
    )


def test_no_schema_permits_exact_gillespie_true() -> None:
    for name in SCHEMA_FILES:
        schema = _load(name)
        for node in _walk(schema):
            if isinstance(node, dict) and "exact_gillespie" in node:
                declaration = node["exact_gillespie"]
                assert not (
                    isinstance(declaration, dict)
                    and (
                        declaration.get("const") is True
                        or True in declaration.get("enum", [])
                    )
                )


def test_runtime_state_and_all_four_action_records_match_schema_facing_shape() -> None:
    initial = EditState.initial(
        "ACG",
        budget=4,
        context={
            "assay": "toy_assay",
            "cell_or_tissue": "toy_cell",
            "endpoint": "toy_endpoint",
        },
    )
    state_record = state_to_schema_record(
        initial, source_id="toy-source", external_time=0.25
    )
    validate_schema_facing_record(state_record, "state")
    assert state_record["x_src"] == state_record["x_current"] == "ACG"
    assert state_record["termination"] is None
    assert state_record["time_direction"] == "source_at_0_to_target_at_1"
    assert not {
        "Z_aux",
        "target_sequence",
        "target_alignment",
        "remaining_target_edits",
    } & set(state_record)

    actions = (
        AtomicAction(ActionType.INS, 1, "U"),
        AtomicAction(ActionType.SUB, 0, "C"),
        AtomicAction(ActionType.DEL, 1),
        AtomicAction(ActionType.STOP),
    )
    for index, action in enumerate(actions):
        transition = apply_action(initial, action, min_length=1, max_length=8)
        record = action_to_schema_record(
            transition,
            action_id=f"action-{index}",
            external_time=0.25,
        )
        validate_schema_facing_record(record, "action")
        assert record["action_type"] == action.kind.value
        assert record["budget_cost"] == int(action.kind != ActionType.STOP)
        assert len(record["pre_state_hash"]) == len(record["post_state_hash"]) == 64
        json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("reason", tuple(TerminationReason))
def test_runtime_termination_records_keep_learned_forced_and_numerical_disjoint(
    reason: TerminationReason,
) -> None:
    kwargs = (
        {"diagnostic": "injected numerical failure"}
        if reason == TerminationReason.FAILED_NUMERICAL
        else {}
    )
    record = termination_to_schema_record(
        reason=reason,
        external_time=0.5,
        state_hash_before="a" * 64,
        state_hash_after="b" * 64,
        **kwargs,
    )
    assert record["learned_stop"] == (reason == TerminationReason.LEARNED_STOP)
    assert record["forced_termination"] == (
        reason
        in {
            TerminationReason.FORCED_BUDGET,
            TerminationReason.FORCED_NO_LEGAL_EDIT_ACTION,
            TerminationReason.FORCED_ZERO_REMAINING_INTEGRATED_HAZARD,
            TerminationReason.FORCED_TIME_HORIZON,
        }
    )
    assert record["inside_ctmc"] == (reason == TerminationReason.LEARNED_STOP)
    assert record["consumes_edit_budget"] is False
    if reason == TerminationReason.FORCED_ZERO_REMAINING_INTEGRATED_HAZARD:
        assert record["remaining_integrated_total_hazard"] == 0.0
    validate_schema_facing_record(record, "termination")


def _coupling_schema_record() -> dict:
    alignment = build_alignment("AC", "AG")
    source, target = reconstruct_alignment(alignment)
    encode_blank = lambda token: "EPSILON" if token == BLANK else token
    return {
        "schema_version": "coupling_manifest_v1",
        "source_id": "source",
        "target_id": "target",
        "coupling_type": alignment.coupling_type,
        "alignment_algorithm_version": alignment.algorithm_version,
        "alignment_cost": alignment.cost,
        "tie_break_rule": alignment.tie_break_rule,
        "alignment_hash": alignment.alignment_hash,
        "path_is_observed": False,
        "path_semantics": "latent_algorithmic",
        "z_src": [encode_blank(column.source_token) for column in alignment.columns],
        "z_tar": [encode_blank(column.target_token) for column in alignment.columns],
        "source_reconstruction": source,
        "target_reconstruction": target,
        "joint_path": "independent_switch_clock_product",
        "schedule": {
            "name": "cubic",
            "kappa": "t**3",
            "kappa_derivative": "3*t**2",
            "rho": "kappa_derivative/(1-kappa)",
            "time_eps": 1.0e-4,
        },
        "target_auxiliary_isolation": {
            "z_aux_training_only": True,
            "rate_network_receives_z_aux": False,
            "leakage_test_status": "PASS",
        },
        "rejected_path_ledger": [],
    }


def test_runtime_coupling_record_validates_and_rejects_observed_or_extra_target_data() -> (
    None
):
    record = _coupling_schema_record()
    validate_schema_facing_record(record, "coupling")
    with pytest.raises(ValueError):
        validate_schema_facing_record({**record, "path_is_observed": True}, "coupling")
    with pytest.raises(ValueError):
        validate_schema_facing_record(
            {**record, "target_alignment_feature": [1, 2, 3]}, "coupling"
        )

    malformed_records = (
        {**record, "alignment_cost": -1},
        {**record, "z_src": ["T"]},
        {**record, "source_reconstruction": "T"},
        {**record, "schedule": {**record["schedule"], "name": "quadratic"}},
        {**record, "schedule": {**record["schedule"], "time_eps": 0.0}},
    )
    for malformed in malformed_records:
        with pytest.raises(ValueError):
            validate_schema_facing_record(malformed, "coupling")


def test_runtime_trajectory_record_validates_all_nested_schema_records() -> None:
    initial = EditState.initial("AC", budget=2)
    stop_rates = lambda _state, _time: {AtomicAction(ActionType.STOP): 100.0}
    result = constrained_single_event_first_order(
        initial,
        stop_rates,
        step_size=0.1,
        stability_hazard=100.0,
        min_length=1,
        max_length=6,
        seed=20_260_802,
    )
    trajectory = sampler_result_to_schema_record(
        result,
        stop_rates,
        trajectory_id="trajectory-0",
        source_id="source",
    )
    validate_schema_facing_record(trajectory, "trajectory")
    with pytest.raises(ValueError):
        validate_schema_facing_record(
            {**trajectory, "exact_gillespie": True}, "trajectory"
        )
    with pytest.raises(ValueError):
        validate_schema_facing_record(
            {**trajectory, "sampler": "unregistered_sampler"}, "trajectory"
        )
    malformed_records = (
        {
            **trajectory,
            "sampler_semantics": "exact_event_time",
        },
        {
            **trajectory,
            "replay": {**trajectory["replay"], "status": "NOT_RUN"},
        },
        {
            **trajectory,
            "steps": [{"garbage": True}],
        },
    )
    for malformed in malformed_records:
        with pytest.raises(ValueError):
            validate_schema_facing_record(malformed, "trajectory")


def test_runtime_schema_validator_rejects_additional_properties_bad_time_and_bad_hash() -> (
    None
):
    record = state_to_schema_record(
        EditState.initial("AC", budget=2),
        source_id="source",
        external_time=0.5,
    )
    malformed_cases = (
        {**record, "undeclared_target_alignment": "forbidden"},
        {**record, "external_time": 1.5},
        {**record, "state_hash": "not-a-sha256"},
        {**record, "x_src": ""},
        {**record, "initial_budget": -1, "remaining_budget": -1},
        {
            **record,
            "context": {**record["context"], "assay": ""},
        },
        {
            **record,
            "target_condition": {
                **record["target_condition"],
                "direction": "unregistered",
            },
        },
    )
    for malformed in malformed_cases:
        with pytest.raises(ValueError):
            validate_schema_facing_record(malformed, "state")


def test_runtime_action_validator_rejects_unknown_type_wrong_budget_and_bad_hash() -> (
    None
):
    initial = EditState.initial("AC", budget=2)
    transition = apply_action(
        initial,
        AtomicAction(ActionType.SUB, 0, "G"),
        min_length=1,
        max_length=6,
    )
    record = action_to_schema_record(transition, action_id="sub-0", external_time=0.5)
    malformed_cases = (
        {**record, "action_type": "REPAIR"},
        {**record, "budget_cost": 0},
        {**record, "post_state_hash": "not-a-sha256"},
        {**record, "undeclared": True},
        {**record, "nucleotide": "T"},
        {**record, "token_index": -1},
    )
    for malformed in malformed_cases:
        with pytest.raises(ValueError):
            validate_schema_facing_record(malformed, "action")


def test_runtime_termination_validator_rejects_bad_time_hash_and_hazard() -> None:
    record = termination_to_schema_record(
        reason=TerminationReason.LEARNED_STOP,
        external_time=0.5,
        state_hash_before="a" * 64,
        state_hash_after="b" * 64,
        instantaneous_edit_hazard=1.0,
        instantaneous_stop_hazard=2.0,
    )
    malformed_cases = (
        {**record, "external_time": math.inf},
        {**record, "state_hash_before": "not-a-sha256"},
        {**record, "instantaneous_edit_hazard": -1.0},
        {**record, "learned_stop": False},
    )
    for malformed in malformed_cases:
        with pytest.raises(ValueError):
            validate_schema_facing_record(malformed, "termination")
