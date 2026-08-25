from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/route_a_v3_route2_xedit_v4_method_repair_protocol_v1.json"


def _load() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_v4_takeover_keeps_c3_natural_terminal_but_blocks_all_c3_downstream() -> None:
    protocol = _load()
    takeover = protocol["takeover"]
    assert protocol["status"].startswith("FROZEN_PROSPECTIVE")
    assert len(takeover["c3_screen_jobs"]) == 5
    assert takeover["c3_jobs_must_finish_naturally"] is True
    assert takeover["c3_terminal_summaries_read_exactly_once"] is True
    assert takeover["c3_active_performance_curve_read_allowed"] is False
    assert takeover["c3_can_trigger_confirmation"] is False
    assert takeover["c3_can_trigger_development_test"] is False
    assert takeover["c3_can_trigger_all_development_refit"] is False
    assert takeover["c3_can_trigger_loso"] is False
    assert takeover["c3_can_trigger_guidance"] is False
    assert takeover["v3_artifacts_read_only"] is True
    assert takeover["v4_parameter_update_blocked_until_all_c3_jobs_terminal"] is True


def test_v4_protected_outcomes_remain_closed_before_v4_confirmation() -> None:
    protocol = _load()
    protected = protocol["protected_outcomes"]
    state = protocol["freeze_state"]
    assert protected["development_test"].startswith("CLOSED_UNTIL_CRITIC_V4_")
    assert protected["development_test_projection"].startswith("ATOMIC_")
    assert protected["new_final_evaluation"].startswith("CLOSED_UNTIL_V4_")
    assert state["development_test_outcome_reads"] == 0
    assert state["new_evaluation_outcome_reads"] == 0
    assert state["v4_optimizer_attempts_started"] == 0


def test_v4_critic_capacity_batches_and_memory_preflight_are_exactly_frozen() -> None:
    critic = _load()["critic_v4"]
    capacity = critic["trainable_parameter_range"]
    memory = critic["memory_preflight"]
    assert (capacity["minimum"], capacity["maximum"]) == (120_000_000, 180_000_000)
    assert (capacity["design_target_minimum"], capacity["design_target_maximum"]) == (
        165_000_000,
        175_000_000,
    )
    assert memory["physical_batch_candidates"] == [4, 8, 16, 32]
    assert memory["minimum_physical_batch"] == 4
    assert memory["effective_batch"] == 32
    assert memory["peak_allocated_memory_gib_minimum"] is None
    assert memory["minimum_occupancy_gate_enabled"] is False
    assert memory["peak_allocated_memory_gib_maximum"] == 35
    assert memory["measurement"] == "TORCH_CUDA_MAX_MEMORY_ALLOCATED"
    assert memory["cpu_fallback"] is False
    assert memory["artificial_padding_or_unused_tensor_allowed"] is False


def test_v401_resource_amendment_is_frozen_before_attempt5_or_performance_read() -> None:
    amendment = _load()["prospective_resource_amendment_v401"]
    assert amendment["authorized_by_user"] is True
    assert amendment["attempt_5_authorized"] is True
    assert amendment["screen_authorized_only_after_attempt_5_dual_pass"] is True
    assert amendment["architecture_training_and_scientific_gates_changed"] is False
    assert amendment["development_validation_performance_read_before_amendment"] is False
    assert amendment["development_test_outcome_reads"] == 0
    assert amendment["new_final_evaluation_outcome_reads"] == 0


def test_v4_critic_architecture_losses_controls_and_seeds_are_prefrozen() -> None:
    critic = _load()["critic_v4"]
    architecture = critic["architecture"]
    trunk = architecture["edit_trunk"]
    mixture = architecture["endpoint_semantic_mixture"]
    assert critic["screen_seed"] == 20260907
    assert critic["confirmation_seeds"] == [20260908, 20260909, 20260910]
    assert critic["additional_screen_or_confirmation_seed_authorized"] is False
    assert trunk["block_count"] == 12
    assert trunk["self_attention_block_count"] == trunk["cross_attention_block_count"] == 6
    assert (trunk["width"], trunk["attention_heads"], trunk["ffn_width"]) == (768, 12, 3072)
    assert mixture["shared_ffn_always_enabled"] is True
    assert (mixture["semantic_expert_count"], mixture["expert_bottleneck_width"], mixture["top_k"]) == (
        4,
        256,
        2,
    )
    assert mixture["router_receives_study_identity"] is False
    assert mixture["router_receives_outcome"] is False
    assert architecture["strict_swap_antisymmetry"] is True
    assert architecture["identity_pair_prediction"] == 0.0
    assert architecture["formal_upper_encoder"] == {
        "source": "PRETRAINED_MRNABERT_BLOCKS_6_THROUGH_11",
        "block_count": 6,
        "embedding_layer_retained": False,
        "bottom_six_retained_in_trainable_module": False,
        "all_retained_parameters_trainable": True,
    }
    assert architecture["readout_fusion"]["concatenated_width"] == 6 * 768
    assert architecture["readout_fusion"]["hidden_width"] == 2560
    assert architecture["readout_fusion"]["output_width"] == 768
    assert architecture["formal_mrnabert_upper_six_trainable_parameter_count"] == 56_664_576
    assert architecture["semantic_expert_bank_scope"] == (
        "ONE_SHARED_FOUR_EXPERT_BANK_REUSED_ACROSS_ALL_12_BLOCKS"
    )
    assert architecture["local_geometry_proxy_trainable_parameter_count"] == 170_481_733
    assert critic["training"]["pass_3_8_loss"]["effective_task_batch_soft_spearman"] == 0.25
    assert critic["training"]["soft_spearman"] == {
        "rank_method": "PAIRWISE_SIGMOID_SOFT_RANK",
        "temperature": 0.2,
        "target_ties": "MID_RANK",
    }
    assert len(critic["screen_package"]["candidate_controls"]) == 4
    assert critic["screen_package"]["mechanism_ablations"] == ["V4-NO-CROSS", "V4-NO-MOE"]


def test_v4_critic_gates_require_absolute_reference_baseline_control_and_ablation_gains() -> None:
    critic = _load()["critic_v4"]
    screen = critic["screen_gate"]
    confirmation = critic["confirmation_gate"]
    assert screen["spearman_threshold_formula"] == "MAX(0.30,C3_REFERENCE+0.05,C0_V4+0.10)"
    assert screen["minimum_absolute_task_macro_spearman"] == 0.30
    assert screen["minimum_margin_over_c3_reference"] == 0.05
    assert screen["minimum_margin_over_c0_v4"] == 0.10
    assert screen["minimum_aggregate_margin_over_complete_candidate_bundle_permutation"] == 0.05
    assert screen["minimum_permutation_task_wins"] == 5
    assert screen["minimum_margin_over_no_cross"] == screen["minimum_margin_over_no_moe"] == 0.02
    assert screen["any_failure_status"] == "XEDITCRITIC_V4_SCREEN_NO_GO"
    assert confirmation["minimum_each_seed_task_macro_spearman"] == 0.30
    assert confirmation["minimum_median_task_macro_spearman"] == 0.35
    assert confirmation["minimum_each_seed_margin_over_c0_v4"] == 0.10
    assert confirmation["minimum_median_margin_over_c0_v4"] == 0.12
    assert confirmation["all_three_seeds_required"] is True


def test_v4_setflow_capacity_source_objective_and_mode_mechanism_are_frozen() -> None:
    flow = _load()["setflow_v4"]
    capacity = flow["trainable_parameter_range"]
    architecture = flow["architecture"]
    source = flow["source_level_training"]
    assert (capacity["minimum"], capacity["maximum"]) == (80_000_000, 150_000_000)
    assert (capacity["design_target_minimum"], capacity["design_target_maximum"]) == (
        95_000_000,
        110_000_000,
    )
    assert (architecture["block_count"], architecture["width"], architecture["attention_heads"]) == (18, 640, 10)
    assert architecture["ffn_width"] == 2560
    assert architecture["latent_mode_count"] == 8
    assert architecture["mode_fixed_for_entire_trajectory"] is True
    assert architecture["hard_legality_mask_before_normalization"] is True
    assert source["unique_terminal_edit_sets_per_source"] is True
    assert source["terminal_candidates_equal_weight_within_source"] is True
    assert source["states_per_source_per_pass"] == 4
    assert source["critic_target_used"] is False
    assert source["independent_evaluator_used"] is False
    assert flow["loss"] == {
        "common_set_marginal": 1.0,
        "source_candidate_coverage": 0.5,
        "remaining_count": 0.2,
        "mode_information": 0.05,
        "coverage_applies_per_compatible_candidate_before_source_mean": True,
        "mode_information_prevents_identical_modes_and_aggregate_collapse": True,
        "unconditional_action_entropy_bonus": 0.0,
    }


def test_v4_setflow_checkpoint_selection_and_three_seed_gate_cannot_drift() -> None:
    flow = _load()["setflow_v4"]
    selection = flow["checkpoint_selection"]
    eligible = selection["eligible_checkpoint_requirements"]
    screen = flow["screen_gate"]
    assert flow["screen_seed"] == 20260911
    assert flow["confirmation_seeds"] == [20260912, 20260913, 20260914]
    assert flow["additional_screen_or_confirmation_seed_authorized"] is False
    assert flow["training"]["saved_checkpoint_passes"] == [4, 6, 8, 10]
    assert flow["training"]["active_generation_metric_read_during_training"] is False
    assert eligible == {
        "maximum_common_validation_nll": 2.06809,
        "minimum_source_macro_recovery": 0.35,
        "minimum_source_macro_top_k_recovery": 0.2,
        "minimum_source_macro_unique_candidate_rate": 0.9,
        "all_g0_correctness_required": True,
    }
    assert selection["tie_break_order"] == [
        "HIGHER_RECOVERY",
        "HIGHER_TOP_K_RECOVERY",
        "LOWER_COMMON_NLL",
        "EARLIER_PASS",
    ]
    assert selection["no_eligible_checkpoint_status"] == "XEDITSETFLOW_V4_SCREEN_NO_GO"
    assert screen["minimum_recovery_margin_over_terminal_f2"] == 0.05
    assert screen["minimum_top_k_margin_over_terminal_f2"] == 0.03
    assert screen["minimum_unique_rate_margin_over_terminal_f2"] == 0.15
    assert flow["confirmation_gate"]["all_three_seeds_required"] is True


def test_v4_guidance_and_failure_policy_remain_closed_and_nonadaptive() -> None:
    protocol = _load()
    guided = protocol["guided_v4"]
    failure = protocol["terminal_failure_policy"]
    assert guided["status"].startswith("BLOCKED_UNTIL_CRITIC_V4_")
    assert guided["free_action_ratio_head_allowed"] is False
    assert guided["setflow_latent_mode_is_fixed_trajectory_state"] is True
    grid = guided["guidance_grid"]
    assert len(grid["kappa"]) * len(grid["tau"]) * len(grid["beta_max"]) == 18
    assert grid["additional_combinations_authorized"] is False
    assert guided["smc"]["forward_equivalent_ceiling_per_source"] == 320
    assert failure["lower_threshold_after_result"] is False
    assert failure["add_seed_after_result"] is False
    assert failure["drop_control_or_baseline_after_result"] is False
    assert failure["repeat_terminal_experiment"] is False
    assert failure["read_test_to_retune"] is False
    assert failure["invent_v5_without_user_discussion"] is False
