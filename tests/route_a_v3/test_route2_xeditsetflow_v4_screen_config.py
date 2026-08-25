from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/route_a_v3_route2_xeditsetflow_v4_screen_v1.json"


def _load() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_screen_has_one_selectable_full_and_one_real_single_mode_control() -> None:
    config = _load()
    runs = config["required_screen_runs"]
    assert [run["run_id"] for run in runs] == ["v4_full", "v4_single_mode"]
    assert [run["run_id"] for run in runs if run["selectable"]] == ["v4_full"]
    assert [run["mode_count"] for run in runs] == [8, 1]
    assert [run["mode_information_weight"] for run in runs] == [0.05, 0.0]


def test_training_and_checkpoint_schedule_are_frozen_without_active_generation() -> None:
    training = _load()["training"]
    assert training["screen_seed"] == 20260911
    assert training["pass_count"] == 10
    assert training["saved_checkpoint_passes"] == [4, 6, 8, 10]
    assert training["early_stopping"] is False
    assert training["validation_generation_during_training"] is False
    assert training["checkpoint_selection_after_terminal_training_only"] is True


def test_formal_capacity_and_source_level_objective_match_protocol() -> None:
    config = _load()
    architecture = config["architecture"]
    objective = config["objective"]
    assert architecture["formal_full_trainable_parameter_count"] == 100_099_998
    assert architecture["formal_single_mode_trainable_parameter_count"] == 98_628_717
    assert 95_000_000 <= architecture["formal_full_trainable_parameter_count"] <= 110_000_000
    assert objective == {
        "common_set_marginal_weight": 1.0,
        "source_candidate_coverage_weight": 0.5,
        "remaining_count_weight": 0.2,
        "full_mode_information_weight": 0.05,
        "single_mode_information_weight": 0.0,
        "per_candidate_constraints_must_remain_separate": True,
        "unconditional_action_entropy_bonus": False,
        "critic_or_independent_evaluator_in_gradient": False,
    }


def test_validation_has_exact_891_by_32_budget_and_mode_stratification() -> None:
    config = _load()
    validation = config["validation_generation"]
    assert config["data_geometry"]["expected_validation_source_record_count"] == 15_327
    assert config["data_geometry"]["eligible_validation_source_count"] == 891
    assert validation["eligible_source_count"] == 891
    assert validation["candidate_cap_per_source"] == 32
    assert validation["trajectory_count_per_source"] == 32
    assert validation["full_mode_initial_trajectories_per_mode"] == 1
    assert validation["full_mode_remaining_trajectories"] == 24
    assert validation["extra_retry_or_duplicate_rejection"] is False


def test_strict_floors_relative_gates_and_protected_reads_cannot_drift() -> None:
    config = _load()
    eligibility = config["checkpoint_eligibility_and_selection"]
    gate = config["screen_gate"]
    assert eligibility["maximum_common_validation_nll"] == 2.06809
    assert eligibility["minimum_source_macro_recovery"] == 0.35
    assert eligibility["minimum_source_macro_top_k_recovery"] == 0.2
    assert eligibility["minimum_source_macro_unique_candidate_rate"] == 0.9
    assert gate["minimum_recovery_margin_over_terminal_f2"] == 0.05
    assert gate["minimum_top_k_margin_over_terminal_f2"] == 0.03
    assert gate["minimum_unique_margin_over_terminal_f2"] == 0.15
    assert gate["minimum_recovery_margin_over_single_mode"] == 0.03
    assert gate["minimum_unique_margin_over_single_mode"] == 0.05
    assert gate["maximum_development_test_outcome_reads"] == 0
    assert gate["maximum_new_evaluation_outcome_reads"] == 0
    assert config["development_test_outcomes_accessed"] is False
    assert config["new_final_evaluation_outcomes_accessed"] is False


def test_launch_barrier_does_not_depend_on_critic_performance() -> None:
    barrier = _load()["launch_barrier"]
    assert barrier["all_five_c3_launch_head_jobs_must_be_terminal"] is True
    assert barrier["c3_terminal_summaries_must_be_read_exactly_once"] is True
    assert barrier["a100_current_head_sync_and_tests_must_pass"] is True
    assert barrier["formal_parameter_preflight_must_pass"] is True
    assert barrier["critic_screen_or_test_result_required"] is False
