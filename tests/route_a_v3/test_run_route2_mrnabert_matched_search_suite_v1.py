from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/run_route2_mrnabert_matched_search_suite_v1.py"
SCHEDULER = ROOT / "scripts/route_a_v3/schedule_route2_mrnabert_postselection_controls_v1.sh"


def _load():
    spec = importlib.util.spec_from_file_location("matched_mrnabert_search_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config():
    return {
        "schema_version": "route_a_v3_route2_mrnabert_matched_search_protocol.v1",
        "required_method_ids": [
            "random_legal", "greedy", "beam", "genetic", "local_search", "generate_then_rerank"
        ],
        "critic_budget_rule": "EXACT_GUIDED_CRITIC_CANDIDATE_FORWARD_EQUIVALENTS_PER_SOURCE",
        "candidate_generation_only": True,
        "strongest_method_selection_in_this_suite": False,
        "evaluation_outcomes_accessed": False,
        "critic_checkpoint_path": "/critic.pt",
        "source_manifest_path": "/sources.jsonl",
        "mrnabert_model_path": "/model",
        "reward_policy_path": "/policy.json",
        "guided_compute_by_source_path": "/compute.jsonl",
        "independent_evaluator_adjudication_path": "/evaluator.json",
        "device": "cuda:0",
        "physical_gpu_index": 0,
        "beam_width": 16,
        "genetic_population_size": 32,
        "oversample_factor": 8,
        "exhaustive_space_limit": 4096,
        "seed": 7,
        "output_directory": "/output",
    }


def _readiness():
    return {
        "schema_version": "route_a_v3_route2_readiness_input.v1",
        "critic": {"final_refit_checkpoint": "/critic.pt"},
    }


def _adjudication():
    return {
        "guided_unlocked": True,
        "critic_status": "CRITIC_READY_FOR_GUIDANCE",
        "flow_status": "FLOW_G0_READY",
    }


def _independent_evaluator_adjudication(status="INDEPENDENT_GENERATION_EVALUATOR_QUALIFIED"):
    return {
        "schema_version": "route_a_v3_route2_independent_generation_evaluator_adjudication.v1",
        "status": status,
        "development_test_outcomes_accessed": False,
        "evaluation_outcomes_accessed": False,
    }


def _guided():
    return {
        "status": "GUIDED_XEDITFLOW_DEVELOPMENT_COMPLETE",
        "matched_search_budget_rule": "EXACT_GUIDED_CRITIC_CANDIDATE_FORWARD_EQUIVALENTS_PER_SOURCE",
        "per_source_compute_path": "/compute.jsonl",
    }


def test_exact_guided_budgets_cover_each_source() -> None:
    module = _load()
    budgets = module.validate_inputs(
        _config(),
        _readiness(),
        _adjudication(),
        _independent_evaluator_adjudication(),
        _guided(),
        [
            {"source_key": "S1", "critic_candidate_forward_equivalent_count": 101},
            {"source_key": "S2", "critic_candidate_forward_equivalent_count": 202},
        ],
        [{"source_key": "S1"}, {"source_key": "S2"}],
    )
    assert budgets == {"S1": 101, "S2": 202}
    with pytest.raises(module.MatchedSearchSuiteError, match="exactly cover"):
        module.validate_inputs(
            _config(), _readiness(), _adjudication(), _independent_evaluator_adjudication(), _guided(),
            [{"source_key": "S1", "critic_candidate_forward_equivalent_count": 101}],
            [{"source_key": "S1"}, {"source_key": "S2"}],
        )


def test_commands_are_serial_mrnabert_search_jobs_without_evaluator() -> None:
    module = _load()
    commands = module.build_commands(_config())
    assert [row["method_id"] for row in commands] == list(module.EXPECTED_METHODS)
    for row in commands:
        command = row["command"]
        assert "--mrnabert-model-path" in command
        assert "--reward-policy" in command
        assert "--critic-budget-by-source" in command
        assert "--max-critic-forwards" not in command
        assert not any("independent_evaluator" in value for value in command)


def test_unqualified_evaluator_cannot_be_relabelled_as_selection() -> None:
    module = _load()
    config = _config()
    config["strongest_method_selection_in_this_suite"] = True
    with pytest.raises(module.MatchedSearchSuiteError, match="scientific method selection"):
        module.validate_inputs(
            config,
            _readiness(),
            _adjudication(),
            _independent_evaluator_adjudication(),
            _guided(),
            [{"source_key": "S1", "critic_candidate_forward_equivalent_count": 1}],
            [{"source_key": "S1"}],
        )


def test_no_go_evaluator_is_frozen_but_cannot_select_a_method() -> None:
    module = _load()
    budgets = module.validate_inputs(
        _config(), _readiness(), _adjudication(),
        _independent_evaluator_adjudication("INDEPENDENT_GENERATION_EVALUATOR_NO_GO"),
        _guided(),
        [{"source_key": "S1", "critic_candidate_forward_equivalent_count": 3}],
        [{"source_key": "S1"}],
    )
    assert budgets == {"S1": 3}


def test_scheduler_runs_matched_search_only_after_guided_generation() -> None:
    source = SCHEDULER.read_text(encoding="utf-8")
    guided = source.index("run_route2_guided_xeditflow_v1.py")
    matched = source.index("run_route2_mrnabert_matched_search_suite_v1.py")
    assert guided < matched
    assert "readiness_stop_before_guided_xeditflow" in source[:guided]
