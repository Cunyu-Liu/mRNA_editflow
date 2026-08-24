"""Prospective readiness and one-shot guidance-screen gates for XEditFlow V4."""

from __future__ import annotations

import math
from itertools import product
from typing import Any, Mapping

import numpy as np


class XEditFlowGateV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowGateV4Error(message)


def _finite(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} is nonfinite")
    return result


GUIDANCE_GRID_V4 = tuple(
    (kappa, temperature, beta_max)
    for kappa, temperature, beta_max in product(
        (0.0, 0.5, 1.0), (0.5, 1.0), (0.5, 1.0, 2.0)
    )
)
GUIDANCE_SCREEN_BASE_FLOW_SEED_V4 = 20260912


def authorize_xeditflow_guidance_v4(
    critic_readiness: Mapping[str, Any],
    setflow_confirmation: Mapping[str, Any],
) -> dict[str, Any]:
    """Require both independently terminal V4 readiness decisions.

    The Critic's single atomic TEST access has already occurred at this point.  It
    is carried forward as a count, not reopened by the guidance authorizer.
    """

    critic_ready = (
        critic_readiness.get("status") == "CRITIC_V4_READY_FOR_GUIDANCE"
        and critic_readiness.get("three_seed_passed") is True
        and critic_readiness.get("frozen_test_passed") is True
        and critic_readiness.get("all_development_refit_complete") is True
        and critic_readiness.get("loso_readiness_passed") is True
        and critic_readiness.get("guidance_authorized") is True
        and int(critic_readiness.get("development_test_access_event_count", -1)) == 1
        and critic_readiness.get("general_test_projection_persisted") is False
        and critic_readiness.get("development_test_outcomes_accessed_after_atomic_test")
        is False
        and critic_readiness.get("new_final_evaluation_outcomes_accessed") is False
    )
    setflow_ready = (
        setflow_confirmation.get("status") == "XEDITSETFLOW_V4_G0_READY"
        and setflow_confirmation.get("required_seeds")
        == [20260912, 20260913, 20260914]
        and int(setflow_confirmation.get("development_test_outcome_reads", -1)) == 0
        and int(setflow_confirmation.get("new_final_evaluation_outcome_reads", -1))
        == 0
        and setflow_confirmation.get("critic_used") is False
        and setflow_confirmation.get("independent_evaluator_used") is False
    )
    authorized = critic_ready and setflow_ready
    return {
        "schema_version": "route_a_v3_route2_xeditflow_v4_guidance_authorization.v1",
        "status": (
            "XEDITFLOW_V4_GUIDANCE_AUTHORIZED"
            if authorized
            else "XEDITFLOW_V4_GUIDANCE_BLOCKED"
        ),
        "critic_ready": critic_ready,
        "setflow_ready": setflow_ready,
        "guidance_authorized": authorized,
        "base_flow_training_seed": GUIDANCE_SCREEN_BASE_FLOW_SEED_V4,
        "guidance_grid_combination_count": len(GUIDANCE_GRID_V4),
        "development_test_reopened": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_authorized": False,
        "new_final_evaluation_outcome_reads": 0,
    }


def adjudicate_guidance_screen_v4(
    results: Mapping[tuple[float, float, float], Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze one of exactly eighteen combinations by the preregistered order."""

    _require(
        set(results) == set(GUIDANCE_GRID_V4) and len(results) == 18,
        "V4 guidance screen must contain exactly the 18 frozen combinations",
    )
    rows: list[dict[str, Any]] = []
    for combination in GUIDANCE_GRID_V4:
        result = results[combination]
        _require(
            result.get("status")
            == "XEDITFLOW_V4_GUIDANCE_SCREEN_COMBINATION_COMPLETE",
            f"V4 guidance screen combination is incomplete: {combination}",
        )
        _require(
            int(result.get("base_flow_training_seed", -1))
            == GUIDANCE_SCREEN_BASE_FLOW_SEED_V4,
            f"V4 guidance screen base-flow seed differs: {combination}",
        )
        _require(
            tuple(float(value) for value in result.get("combination", ()))
            == combination,
            f"V4 guidance screen combination identity differs: {combination}",
        )
        _require(
            result.get("setflow_mode_is_fixed_trajectory_state") is True
            and result.get("free_action_ratio_head_used") is False
            and result.get("all_network_forwards_separately_charged") is True,
            f"V4 guidance mechanism or compute accounting differs: {combination}",
        )
        _require(
            result.get("development_test_outcomes_accessed_after_atomic_test")
            is False
            and int(result.get("new_final_evaluation_outcome_reads", -1)) == 0,
            f"V4 guidance screen accessed a protected outcome: {combination}",
        )
        row = {
            "combination": combination,
            "closed_source_macro_ndcg": _finite(
                result.get("closed_source_macro_ndcg"), "closed NDCG"
            ),
            "closed_source_macro_normalized_regret": _finite(
                result.get("closed_source_macro_normalized_regret"),
                "closed normalized regret",
            ),
            "independent_evaluator_paired_margin": _finite(
                result.get("independent_evaluator_paired_margin"),
                "independent evaluator paired margin",
            ),
            "open_source_macro_candidate_recovery": _finite(
                result.get("open_source_macro_candidate_recovery"),
                "open candidate recovery",
            ),
            "total_forward_equivalents": int(
                result.get("total_forward_equivalents", -1)
            ),
        }
        _require(
            0 <= row["total_forward_equivalents"] <= 320,
            f"V4 guidance screen compute differs: {combination}",
        )
        rows.append(row)
    selected = min(
        rows,
        key=lambda row: (
            -row["closed_source_macro_ndcg"],
            row["closed_source_macro_normalized_regret"],
            -row["independent_evaluator_paired_margin"],
            -row["open_source_macro_candidate_recovery"],
            row["total_forward_equivalents"],
            row["combination"],
        ),
    )
    return {
        "schema_version": "route_a_v3_route2_xeditflow_v4_guidance_screen_gate.v1",
        "status": "XEDITFLOW_V4_GUIDANCE_SCREEN_FROZEN",
        "base_flow_training_seed": GUIDANCE_SCREEN_BASE_FLOW_SEED_V4,
        "combination_count": 18,
        "selected_kappa": selected["combination"][0],
        "selected_temperature": selected["combination"][1],
        "selected_beta_max": selected["combination"][2],
        "selection_order": [
            "MAX_CLOSED_SOURCE_MACRO_NDCG",
            "MIN_CLOSED_SOURCE_MACRO_NORMALIZED_REGRET",
            "MAX_INDEPENDENT_EVALUATOR_PAIRED_MARGIN",
            "MAX_OPEN_SOURCE_MACRO_CANDIDATE_RECOVERY",
            "MIN_TOTAL_FORWARD_EQUIVALENTS",
        ],
        "selected_metrics": {
            key: value for key, value in selected.items() if key != "combination"
        },
        "setflow_mode_is_fixed_trajectory_state": True,
        "free_action_ratio_head_allowed": False,
        "additional_grid_combination_authorized": False,
        "development_test_reopened": False,
        "new_final_evaluation_authorized": False,
    }


def adjudicate_guided_three_seed_v4(
    payloads: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen V4 publication gate to three matched Flow seeds."""

    required_seeds = (20260912, 20260913, 20260914)
    _require(
        tuple(sorted(payloads)) == required_seeds and len(payloads) == 3,
        "V4 guided comparison requires exactly the three frozen base-flow seeds",
    )
    required_methods = {
        "full_soft_value_smc",
        "unguided_setflow",
        "first_order_guidance",
        "simple_rate_guidance",
        "generate_then_rerank",
        "strongest_matched_baseline",
    }
    seed_results: dict[str, Any] = {}
    minimum_ndcg_improvements: list[float] = []
    evaluator_margins: list[float] = []
    for seed in required_seeds:
        payload = payloads[seed]
        methods = payload.get("methods")
        _require(
            isinstance(methods, Mapping) and set(methods) == required_methods,
            f"V4 guided comparison method set differs: {seed}",
        )
        _require(
            payload.get("setflow_mode_is_fixed_trajectory_state") is True
            and payload.get("free_action_ratio_head_used") is False
            and payload.get("all_network_forwards_separately_charged") is True
            and payload.get("matched_compute_schema") == "MatchedComputeRecordV4",
            f"V4 guided mechanism or compute accounting differs: {seed}",
        )
        _require(
            payload.get("development_test_outcomes_accessed_after_atomic_test")
            is False
            and int(payload.get("new_final_evaluation_outcome_reads", -1)) == 0
            and payload.get("independent_evaluator_in_gradient") is False,
            f"V4 guided comparison accessed a protected outcome or evaluator gradient: {seed}",
        )
        full = methods["full_soft_value_smc"]
        unguided = methods["unguided_setflow"]
        baseline = methods["strongest_matched_baseline"]
        full_ndcg = _finite(full.get("closed_source_macro_ndcg"), "full NDCG")
        unguided_ndcg = _finite(
            unguided.get("closed_source_macro_ndcg"), "unguided NDCG"
        )
        baseline_ndcg = _finite(
            baseline.get("closed_source_macro_ndcg"), "baseline NDCG"
        )
        auxiliary_ndcgs = {
            name: _finite(
                methods[name].get("closed_source_macro_ndcg"), f"{name} NDCG"
            )
            for name in (
                "first_order_guidance",
                "simple_rate_guidance",
                "generate_then_rerank",
            )
        }
        improvement_unguided = full_ndcg - unguided_ndcg
        improvement_baseline = full_ndcg - baseline_ndcg
        minimum_ndcg_improvements.append(
            min(improvement_unguided, improvement_baseline)
        )
        full_regret = _finite(
            full.get("closed_source_macro_normalized_regret"), "full regret"
        )
        unguided_regret = _finite(
            unguided.get("closed_source_macro_normalized_regret"),
            "unguided regret",
        )
        baseline_regret = _finite(
            baseline.get("closed_source_macro_normalized_regret"),
            "baseline regret",
        )
        evaluator_margin = _finite(
            full.get("independent_evaluator_margin_over_strongest_baseline"),
            "independent evaluator margin",
        )
        evaluator_margins.append(evaluator_margin)
        ndcg_ci = payload.get("source_paired_ndcg_improvement_ci_95")
        evaluator_ci = payload.get(
            "source_paired_independent_evaluator_margin_ci_95"
        )
        _require(
            isinstance(ndcg_ci, Mapping)
            and set(ndcg_ci) == {"over_unguided", "over_strongest_baseline"}
            and all(
                isinstance(value, list) and len(value) == 2
                for value in ndcg_ci.values()
            )
            and isinstance(evaluator_ci, list)
            and len(evaluator_ci) == 2,
            f"V4 guided paired-bootstrap intervals differ: {seed}",
        )
        checks = {
            "ndcg_over_unguided_at_least_0_05": improvement_unguided >= 0.05,
            "ndcg_over_strongest_baseline_at_least_0_05": improvement_baseline
            >= 0.05,
            "ndcg_beats_first_order_simple_rate_and_rerank": all(
                full_ndcg > value for value in auxiliary_ndcgs.values()
            ),
            "ndcg_bootstrap_ci_over_unguided_positive": float(
                ndcg_ci["over_unguided"][0]
            )
            > 0.0,
            "ndcg_bootstrap_ci_over_strongest_positive": float(
                ndcg_ci["over_strongest_baseline"][0]
            )
            > 0.0,
            "regret_reduction_over_unguided_at_least_10pct": unguided_regret
            > 0.0
            and (unguided_regret - full_regret) / unguided_regret >= 0.10,
            "regret_reduction_over_strongest_at_least_10pct": baseline_regret
            > 0.0
            and (baseline_regret - full_regret) / baseline_regret >= 0.10,
            "top_1_recall_not_below_unguided": _finite(
                full.get("closed_source_macro_top_1_recall"), "full top-1"
            )
            >= _finite(
                unguided.get("closed_source_macro_top_1_recall"), "unguided top-1"
            ),
            "top_1_recall_not_below_strongest": _finite(
                full.get("closed_source_macro_top_1_recall"), "full top-1"
            )
            >= _finite(
                baseline.get("closed_source_macro_top_1_recall"), "baseline top-1"
            ),
            "open_recovery_at_least_0_25": _finite(
                full.get("open_source_macro_candidate_recovery"), "open recovery"
            )
            >= 0.25,
            "open_top_k_recovery_at_least_0_15": _finite(
                full.get("open_source_macro_top_k_recovery"), "open top-k recovery"
            )
            >= 0.15,
            "unique_rate_at_least_0_90": _finite(
                full.get("open_source_macro_unique_candidate_rate"),
                "open unique rate",
            )
            >= 0.90,
            "independent_evaluator_margin_positive": evaluator_margin > 0.0,
            "independent_evaluator_ci_lower_positive": float(evaluator_ci[0])
            > 0.0,
            "hard_legality_100pct": _finite(
                full.get("hard_legality_rate"), "hard legality"
            )
            == 1.0,
            "failure_counters_zero": all(
                int(full.get(key, -1)) == 0
                for key in (
                    "edit_budget_violation_count",
                    "candidate_budget_violation_count",
                    "trajectory_replay_failure_count",
                    "numerical_failure_count",
                )
            ),
            "matched_compute_ceiling_met": 0
            <= int(full.get("maximum_forward_equivalents_per_source", -1))
            <= 320
            and payload.get("all_methods_matched_compute_ceiling_met") is True,
            "protected_outcome_and_mechanism_checks_pass": True,
        }
        seed_results[str(seed)] = {
            "ndcg_improvement_over_unguided": improvement_unguided,
            "ndcg_improvement_over_strongest_baseline": improvement_baseline,
            "independent_evaluator_margin": evaluator_margin,
            "checks": checks,
            "passed": all(checks.values()),
        }
    median_min_ndcg_improvement = float(np.median(minimum_ndcg_improvements))
    median_evaluator_margin = float(np.median(evaluator_margins))
    cohort_checks = {
        "all_three_seed_checks_pass": all(
            row["passed"] for row in seed_results.values()
        ),
        "median_min_ndcg_improvement_at_least_0_07": median_min_ndcg_improvement
        >= 0.07,
        "median_independent_evaluator_margin_at_least_0_10": median_evaluator_margin
        >= 0.10,
    }
    passed = all(cohort_checks.values())
    reward_exploitation = any(
        payloads[seed].get("critic_self_score_increased") is True
        and not all(
            seed_results[str(seed)]["checks"][key]
            for key in (
                "ndcg_over_unguided_at_least_0_05",
                "ndcg_over_strongest_baseline_at_least_0_05",
                "regret_reduction_over_unguided_at_least_10pct",
                "regret_reduction_over_strongest_at_least_10pct",
                "independent_evaluator_margin_positive",
                "independent_evaluator_ci_lower_positive",
            )
        )
        for seed in required_seeds
    )
    return {
        "schema_version": "route_a_v3_route2_xeditflow_v4_three_seed_gate.v1",
        "status": "XEDITFLOW_V4_PASS" if passed else "XEDITFLOW_V4_NO_GO",
        "required_seeds": list(required_seeds),
        "seed_results": seed_results,
        "median_min_ndcg_improvement": median_min_ndcg_improvement,
        "median_independent_evaluator_margin": median_evaluator_margin,
        "cohort_checks": cohort_checks,
        "reward_exploitation": reward_exploitation,
        "new_final_evaluation_authorized": passed,
        "additional_training_seed_authorized": False,
        "submission_ready": False,
    }
