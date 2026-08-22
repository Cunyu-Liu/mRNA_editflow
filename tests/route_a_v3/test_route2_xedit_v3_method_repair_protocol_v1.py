from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/route_a_v3_route2_xedit_v3_method_repair_protocol_v1.json"
PROJECTION = ROOT / "configs/route_a_v3_route2_xedit_v3_development_projection_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_xedit_v3_protocol_freezes_strict_critic_gates_and_exact_seeds() -> None:
    protocol = _load(PROTOCOL)
    critic = protocol["critic"]
    assert protocol["status"].startswith("FROZEN_PROSPECTIVE")
    assert critic["screen_seed"] == 20260830
    assert critic["confirmation_seeds"] == [20260831, 20260901, 20260902]
    assert critic["additional_seed_authorized"] is False
    assert critic["three_seed_gate"] == {
        "minimum_each_seed_task_macro_spearman": 0.25,
        "minimum_median_task_macro_spearman": 0.3,
        "minimum_each_seed_margin_over_c0": 0.07,
        "minimum_median_margin_over_c0": 0.1,
        "minimum_positive_tasks_each_seed": 8,
        "minimum_task_wins_over_c0_each_seed": 6,
        "maximum_task_macro_standardized_mae": 1.7,
        "paired_source_group_bootstrap_ci_lower_bound": ">0",
    }
    assert [arm["arm_id"] for arm in critic["screen_arms"] if arm["selectable"]] == ["C2", "C3"]


def test_xedit_v3_protocol_keeps_test_evaluation_and_guidance_closed() -> None:
    protocol = _load(PROTOCOL)
    projection = _load(PROJECTION)
    assert projection["included_splits"] == ["TRAIN", "VALIDATION"]
    assert projection["development_test_outcomes_accessed"] is False
    assert projection["evaluation_outcomes_accessed"] is False
    assert protocol["protected_outcomes"]["development_test"].startswith("CLOSED_")
    assert protocol["protected_outcomes"]["new_final_evaluation"].startswith("CLOSED_")
    assert protocol["guided"]["status"].startswith("BLOCKED_")
    assert set(protocol["setflow"]["action_space"]) == {"SUB", "STOP"}
    assert set(protocol["setflow"]["excluded_actions"]) == {"INS", "DEL"}


def test_xedit_v3_flow_capacity_and_guidance_budget_are_prefrozen() -> None:
    protocol = _load(PROTOCOL)
    flow = protocol["setflow"]
    assert flow["screen_seed"] == 20260903
    assert flow["confirmation_seeds"] == [20260904, 20260905, 20260906]
    assert [arm["arm_id"] for arm in flow["screen_arms"] if arm["selectable"]] == ["F2", "F3"]
    assert flow["screen_gate"]["minimum_source_macro_candidate_recovery"] == 0.25
    assert flow["screen_gate"]["minimum_source_macro_top_k_recovery"] == 0.15
    assert flow["screen_gate"]["minimum_source_macro_unique_candidate_rate"] == 0.9
    grid = protocol["guided"]["guidance_grid"]
    assert len(grid["kappa"]) * len(grid["tau"]) * len(grid["beta_max"]) == grid["combination_count"] == 18
    assert protocol["guided"]["smc"]["forward_equivalent_ceiling_per_source"] == 320
    assert protocol["terminal_failure_policy"]["add_seed_after_result"] is False
