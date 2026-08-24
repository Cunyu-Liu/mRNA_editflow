from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.route_a_v3.run_route2_xeditflow_matched_controls_v4 import (
    _critic_projection_rows_for_states_v4,
    maximum_control_round_forward_equivalents_v4,
    validate_matched_control_config_v4,
)
from core.route2_legal_xeditflow import apply_action, initial_state, legal_actions
from core.route2_xeditcritic_training_data_v3 import UNKNOWN_CATEGORY


def _config() -> dict:
    return {
        "schema_version": "route_a_v3_route2_xeditflow_matched_control_run_config.v4",
        "method_id": "simple_rate_guidance",
        "base_flow_training_seed": 20260913,
        "kappa": 0.5,
        "temperature": 1.0,
        "beta_max": 2.0,
        "particle_count": 32,
        "candidate_cap": 32,
        "ess_threshold": 16.0,
        "resampling": "STRATIFIED",
        "forward_equivalent_ceiling_per_source": 320,
        "terminal_critic_forwards_by_member": [8, 4, 1],
        "maximum_sampling_rounds": 32,
        "action_space": "SUB+STOP",
        "replay_check": True,
        "decoder_seed_base": 20261001,
        "expected_source_count": 891,
        "critic_refit_runtime_config_paths": {
            "20260908": "/mnt/a.json",
            "20260909": "/mnt/b.json",
            "20260910": "/mnt/c.json",
        },
        "physical_gpu_index": 4,
        "device": "cuda:4",
        "output_dir": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/final/controls",
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def test_v4_matched_control_config_freezes_final_seeds_and_budget() -> None:
    validate_matched_control_config_v4(_config())
    for method in (
        "unguided_setflow",
        "first_order_guidance",
        "simple_rate_guidance",
        "generate_then_rerank",
    ):
        config = copy.deepcopy(_config())
        config["method_id"] = method
        validate_matched_control_config_v4(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("base_flow_training_seed", 20260911, "seed"),
        ("particle_count", 64, "particle"),
        ("candidate_cap", 64, "candidate"),
        ("forward_equivalent_ceiling_per_source", 321, "compute"),
        ("maximum_sampling_rounds", 31, "round"),
        ("decoder_seed_base", 20261002, "decoder"),
        ("physical_gpu_index", 6, "GPU"),
    ],
)
def test_v4_matched_control_config_rejects_protocol_drift(
    field: str, value: object, message: str
) -> None:
    config = copy.deepcopy(_config())
    config[field] = value
    with pytest.raises(Exception, match=message):
        validate_matched_control_config_v4(config)


def test_v4_control_round_bound_charges_replay_and_member_batches() -> None:
    physical = (4, 8, 32)
    assert maximum_control_round_forward_equivalents_v4(
        5, method_id="unguided_setflow", critic_physical_batches=physical
    ) == 90
    assert maximum_control_round_forward_equivalents_v4(
        5, method_id="generate_then_rerank", critic_physical_batches=physical
    ) == 90
    assert maximum_control_round_forward_equivalents_v4(
        5, method_id="simple_rate_guidance", critic_physical_batches=physical
    ) == 90 + (9 + 4 * 8) + (5 + 4 * 4) + (2 + 4 * 1)


def test_v4_control_critic_projection_is_study_neutral_and_complete() -> None:
    root = initial_state("AA", budget=1, assay_id="assay", context_id="context")
    edited = apply_action(
        root, next(action for action in legal_actions(root) if action.kind == "SUB")
    )
    rows = _critic_projection_rows_for_states_v4(
        [root, edited],
        source={
            "source_key": "source",
            "source_sequence": "AA",
            "region": "5UTR",
            "assay_id": "assay",
            "biological_context_id": "context",
        },
        representative={
            "task_id": "task",
            "source_group_id": "group",
            "endpoint_descriptor": {"quantity_family": "RNA_ABUNDANCE"},
        },
    )
    assert rows[0]["source_relative_edits"] == []
    assert len(rows[1]["source_relative_edits"]) == 1
    assert all(row["study_unit_id"] == UNKNOWN_CATEGORY for row in rows)
    assert all(row["dummy_target_for_inference_only"] is True for row in rows)


def test_v4_matched_runner_records_fixed_modes_replay_and_pending_rerank() -> None:
    source = Path(
        "scripts/route_a_v3/run_route2_xeditflow_matched_controls_v4.py"
    ).read_text(encoding="utf-8")
    assert "stratified_trajectory_mode_ids_v4" in source
    assert "result[\"candidates\"] == replay[\"candidates\"]" in source
    assert "trajectory_critic_forwards_actual_by_member" in source
    assert '"terminal_rerank_pending": method == "generate_then_rerank"' in source
    assert "used + worst_round > 320" in source
