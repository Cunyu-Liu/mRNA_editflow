from __future__ import annotations

import json

import pytest

from core.route2_xeditflow_gate_v3 import GUIDANCE_GRID_V3, adjudicate_guidance_screen_v3
from scripts.route_a_v3.adjudicate_route2_xeditflow_guidance_screen_v3 import assemble_screen_results_v3


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _manifest(tmp_path):
    jobs = []
    for index, combination in enumerate(GUIDANCE_GRID_V3):
        root = tmp_path / str(index)
        closed = root / "closed"
        smc = root / "smc"
        common = {
            "development_test_outcomes_accessed": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        _write(closed / "run_summary.json", {
            **common,
            "status": "XEDITFLOW_V3_CLOSED_NEIGHBORHOOD_COMPLETE",
            "source_macro_ndcg": 0.5 + index / 1000,
            "source_macro_normalized_regret": 0.4,
        })
        _write(smc / "run_summary.json", {
            **common,
            "status": "XEDITFLOW_V3_SMC_GENERATION_COMPLETE",
            "maximum_forward_equivalents_per_source": 200,
            "reserved_terminal_critic_forwards_per_source": 3,
        })
        open_path = root / "open.json"
        evaluator_path = root / "evaluator.json"
        _write(open_path, {**common, "source_macro_candidate_recovery": 0.3})
        _write(evaluator_path, {**common, "paired_margin_over_strongest_baseline": 0.1})
        jobs.append({
            "combination": list(combination),
            "closed_config": {"output_dir": str(closed)},
            "smc_config": {"output_dir": str(smc)},
            "open_generation_metric_path": str(open_path),
            "independent_evaluator_metric_path": str(evaluator_path),
        })
    return {
        "schema_version": "route_a_v3_route2_xeditflow_guidance_screen_manifest.v1",
        "status": "XEDITFLOW_V3_GUIDANCE_SCREEN_CONFIGS_PREPARED",
        "guidance_jobs": jobs,
    }


def test_guidance_adjudicator_assembles_exact_grid_and_freezes_one_winner(tmp_path) -> None:
    results = assemble_screen_results_v3(_manifest(tmp_path))
    assert set(results) == set(GUIDANCE_GRID_V3)
    gate = adjudicate_guidance_screen_v3(results)
    assert gate["status"] == "XEDITFLOW_V3_GUIDANCE_SCREEN_FROZEN"
    assert (
        gate["selected_kappa"], gate["selected_temperature"], gate["selected_beta_max"]
    ) == GUIDANCE_GRID_V3[-1]


def test_guidance_adjudicator_rejects_protected_outcome_contamination(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    path = manifest["guidance_jobs"][0]["independent_evaluator_metric_path"]
    payload = json.loads(open(path).read())
    payload["development_test_outcomes_accessed"] = True
    open(path, "w").write(json.dumps(payload))
    with pytest.raises(Exception, match="protected outcome"):
        assemble_screen_results_v3(manifest)
