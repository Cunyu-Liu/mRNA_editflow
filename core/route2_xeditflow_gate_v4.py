"""Prospective readiness and one-shot guidance-screen gates for XEditFlow V4."""

from __future__ import annotations

import math
from itertools import product
from typing import Any, Mapping


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
