from __future__ import annotations

import copy
import json

import pytest

from core.route2_xeditflow_gate_v4 import GUIDANCE_GRID_V4
from scripts.route_a_v3.adjudicate_route2_xeditflow_guidance_screen_v4 import (
    assemble_screen_results_v4,
)


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _component(value: float) -> str:
    return str(float(value)).replace(".", "p")


def _manifest(tmp_path):
    compute_path = tmp_path / "compute.jsonl"
    compute = {
        "schema_version": "MatchedComputeRecordV4",
        "total_forward_equivalents": 300,
        "critic_forwards_by_member": [4, 4, 4],
        "all_network_forwards_separately_charged": True,
        "terminal_critic_reservation_reconciled": True,
        "failure_counters": {
            "edit_budget_violation_count": 0,
            "candidate_budget_violation_count": 0,
            "replay_failure_count": 0,
            "numerical_failure_count": 0,
        },
    }
    compute_path.write_text(
        "".join(json.dumps({**compute, "source_key": f"s{i}"}) + "\n" for i in range(891)),
        encoding="utf-8",
    )
    result_paths = []
    for index, combination in enumerate(GUIDANCE_GRID_V4):
        kappa, temperature, beta_max = combination
        combination_id = (
            f"kappa_{_component(kappa)}_temperature_{_component(temperature)}"
            f"_beta_{_component(beta_max)}"
        )
        method = f"xeditflow_v4_guidance_screen_{combination_id}"
        root = tmp_path / str(index)
        common = {
            "method_id": method,
            "base_flow_training_seed": 20260912,
            "kappa": kappa,
            "temperature": temperature,
            "beta_max": beta_max,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        smc = root / "smc.json"
        critic = root / "critic.json"
        closed = root / "closed.json"
        open_metric = root / "open.json"
        evaluator_summary = root / "evaluator_summary.json"
        evaluator_metric = root / "evaluator_metric.json"
        _write(
            smc,
            {
                **common,
                "status": (
                    "XEDITFLOW_V4_SMC_GENERATION_COMPLETE_PENDING_TERMINAL_CRITIC_SCORING"
                ),
                "setflow_mode_is_fixed_trajectory_state": True,
                "free_action_ratio_head_used": False,
            },
        )
        _write(
            critic,
            {
                **common,
                "status": "XEDITFLOW_V4_CANDIDATE_CRITIC_SCORING_COMPLETE",
                "source_count": 891,
                "maximum_total_forward_equivalents_per_source": 300,
                "forward_equivalent_ceiling_per_source": 320,
            },
        )
        _write(
            closed,
            {
                **common,
                "status": "XEDITFLOW_V4_CLOSED_NEIGHBORHOOD_COMPLETE",
                "source_macro_ndcg": 0.5 + index / 1000,
                "source_macro_normalized_regret": 0.4,
            },
        )
        _write(
            open_metric,
            {
                **common,
                "status": "XEDITFLOW_V4_OPEN_GENERATION_METRICS_COMPLETE",
                "source_macro_candidate_recovery": 0.35,
            },
        )
        _write(
            evaluator_summary,
            {
                "status": "FROZEN_INDEPENDENT_EVALUATOR_SCORING_COMPLETE",
                "evaluation_outcomes_used_to_select_evaluator": 0,
                "independent_evaluator_in_gradient": False,
                "development_test_outcomes_accessed_after_atomic_test": False,
                "new_final_evaluation_outcome_reads": 0,
            },
        )
        _write(
            evaluator_metric,
            {
                "status": (
                    "XEDITFLOW_V4_INDEPENDENT_EVALUATOR_COMPARISON_COMPLETE"
                ),
                "method_id": method,
                "base_flow_training_seed": 20260912,
                "combination": list(combination),
                "paired_margin_over_strongest_baseline": 0.1,
                "independent_evaluator_used_for_gradient": False,
                "development_test_outcomes_accessed_after_atomic_test": False,
                "new_final_evaluation_outcome_reads": 0,
            },
        )
        result_paths.append(
            {
                "combination_id": combination_id,
                "combination": list(combination),
                "smc_summary_path": str(smc),
                "critic_summary_path": str(critic),
                "matched_compute_path": str(compute_path),
                "closed_summary_path": str(closed),
                "open_metric_path": str(open_metric),
                "independent_evaluator_summary_path": str(evaluator_summary),
                "independent_evaluator_metric_path": str(evaluator_metric),
            }
        )
    return {
        "schema_version": (
            "route_a_v3_route2_xeditflow_v4_value_config_manifest.v1"
        ),
        "status": "XEDITFLOW_V4_VALUE_CONFIGS_PREPARED_NOT_STARTED",
        "guidance_result_paths": result_paths,
    }


def test_v4_adjudicator_reads_exact_terminal_grid_and_keeps_frozen_order(
    tmp_path,
) -> None:
    results = assemble_screen_results_v4(_manifest(tmp_path))
    assert set(results) == set(GUIDANCE_GRID_V4)
    assert all(row["total_forward_equivalents"] == 300 for row in results.values())
    assert max(
        results,
        key=lambda combination: results[combination]["closed_source_macro_ndcg"],
    ) == GUIDANCE_GRID_V4[-1]


def test_v4_adjudicator_rejects_unreconciled_compute_or_protected_read(
    tmp_path,
) -> None:
    manifest = _manifest(tmp_path)
    changed = copy.deepcopy(manifest)
    evaluator_path = changed["guidance_result_paths"][0][
        "independent_evaluator_metric_path"
    ]
    evaluator = json.loads(open(evaluator_path, encoding="utf-8").read())
    evaluator["new_final_evaluation_outcome_reads"] = 1
    open(evaluator_path, "w", encoding="utf-8").write(json.dumps(evaluator))
    with pytest.raises(Exception, match="protected outcome"):
        assemble_screen_results_v4(changed)
