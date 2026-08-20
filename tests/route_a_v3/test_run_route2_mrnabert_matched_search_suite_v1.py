from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/run_route2_mrnabert_matched_search_suite_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("matched_mrnabert_search_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config():
    return {
        "schema_version": "route_a_v3_route2_mrnabert_critic_v2_matched_search_protocol.v1",
        "status": "WAITING_FOR_CRITIC_V2_GUIDED_XEDITFLOW",
        "required_method_ids": [
            "random_legal", "greedy", "beam", "genetic", "local_search", "generate_then_rerank"
        ],
        "critic_budget_rule": "GUIDED_TOTAL_FORWARD_EQUIVALENTS_AS_SEARCH_CRITIC_CAP_PER_SOURCE",
        "candidate_generation_only": True,
        "strongest_method_selection_in_this_suite": False,
        "evaluation_outcomes_accessed": False,
        "readiness_input_path": "/unused",
        "readiness_adjudication_path": "/unused",
        "guided_summary_path": "/unused",
        "critic_checkpoint_path": "/unused",
        "source_manifest_path": "/sources.jsonl",
        "mrnabert_model_path": "/model",
        "reward_policy_path": "/policy.json",
        "selected_attention_backend": "PYTORCH_SDPA_AUTO",
        "guided_compute_by_source_path": "/unused",
        "independent_evaluator_adjudication_path": "/evaluator.json",
        "device": "cuda:0",
        "physical_gpu_index": 0,
        "beam_width": 16,
        "genetic_population_size": 32,
        "oversample_factor": 8,
        "exhaustive_space_limit": 4096,
        "seed": 20260825,
        "output_directory": "/output",
    }


def _readiness():
    return {
        "schema_version": "route_a_v3_route2_mrnabert_critic_v2_guidance_readiness_input.v1",
        "critic": {"refit_checkpoint": "/unused"},
        "guided_generation_executed": False,
        "evaluation_opened_by_readiness_builder": False,
    }


def _adjudication():
    return {
        "schema_version": "route_a_v3_route2_mrnabert_critic_v2_guidance_readiness_adjudication.v1",
        "guided_unlocked": True,
        "critic_status": "CRITIC_READY_FOR_GUIDANCE",
        "flow_status": "FLOW_G0_READY",
        "guided_generation_status": "GUIDED_XEDITFLOW_DEVELOPMENT_ALLOWED",
        "guided_generation_executed": False,
        "evaluation_opened": False,
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
        "schema_version": "route_a_v3_route2_mrnabert_critic_v2_guided_xeditflow_development.v1",
        "status": "GUIDED_XEDITFLOW_DEVELOPMENT_COMPLETE",
        "matched_search_budget_rule": "GUIDED_TOTAL_FORWARD_EQUIVALENTS_AS_SEARCH_CRITIC_CAP_PER_SOURCE",
        "per_source_compute_path": "/unused",
        "evaluation_outcomes_read": 0,
        "generated_candidates_grant_canonical_credit": False,
        "biological_optimization_established": False,
    }


def test_exact_guided_budgets_cover_each_source() -> None:
    module = _load()
    config = _config()
    config.update({
        "readiness_input_path": str(module.EXPECTED_READINESS_INPUT),
        "readiness_adjudication_path": str(module.EXPECTED_READINESS_ADJUDICATION),
        "critic_checkpoint_path": str(module.EXPECTED_CRITIC_CHECKPOINT),
        "guided_summary_path": str(module.EXPECTED_GUIDED_ROOT / "guided_summary.json"),
        "guided_compute_by_source_path": str(module.EXPECTED_GUIDED_ROOT / "guided_compute_by_source.jsonl"),
    })
    readiness = _readiness()
    readiness["critic"]["refit_checkpoint"] = str(module.EXPECTED_CRITIC_CHECKPOINT)
    guided = _guided()
    guided["per_source_compute_path"] = str(module.EXPECTED_GUIDED_ROOT / "guided_compute_by_source.jsonl")
    budgets = module.validate_inputs(
        config,
        readiness,
        _adjudication(),
        _independent_evaluator_adjudication(),
        guided,
        [
            {"source_key": "S1", "matched_search_critic_forward_budget": 101},
            {"source_key": "S2", "matched_search_critic_forward_budget": 202},
        ],
        [{"source_key": "S1"}, {"source_key": "S2"}],
    )
    assert budgets == {"S1": 101, "S2": 202}
    with pytest.raises(module.MatchedSearchSuiteError, match="exactly cover"):
        module.validate_inputs(
            config, readiness, _adjudication(), _independent_evaluator_adjudication(), guided,
            [{"source_key": "S1", "matched_search_critic_forward_budget": 101}],
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
        assert command[command.index("--attention-backend") + 1] == "PYTORCH_SDPA_AUTO"
        assert "--critic-budget-by-source" in command
        assert "--max-critic-forwards" not in command
        assert not any("independent_evaluator" in value for value in command)


def test_unqualified_evaluator_cannot_be_relabelled_as_selection() -> None:
    module = _load()
    config = _config()
    config.update({
        "readiness_input_path": str(module.EXPECTED_READINESS_INPUT),
        "readiness_adjudication_path": str(module.EXPECTED_READINESS_ADJUDICATION),
        "critic_checkpoint_path": str(module.EXPECTED_CRITIC_CHECKPOINT),
        "guided_summary_path": str(module.EXPECTED_GUIDED_ROOT / "guided_summary.json"),
        "guided_compute_by_source_path": str(module.EXPECTED_GUIDED_ROOT / "guided_compute_by_source.jsonl"),
    })
    config["strongest_method_selection_in_this_suite"] = True
    with pytest.raises(module.MatchedSearchSuiteError, match="scientific method selection"):
        module.validate_inputs(
            config,
            _readiness(),
            _adjudication(),
            _independent_evaluator_adjudication(),
            _guided(),
            [{"source_key": "S1", "matched_search_critic_forward_budget": 1}],
            [{"source_key": "S1"}],
        )


def test_no_go_evaluator_is_frozen_but_cannot_select_a_method() -> None:
    module = _load()
    config = _config()
    config.update({
        "readiness_input_path": str(module.EXPECTED_READINESS_INPUT),
        "readiness_adjudication_path": str(module.EXPECTED_READINESS_ADJUDICATION),
        "critic_checkpoint_path": str(module.EXPECTED_CRITIC_CHECKPOINT),
        "guided_summary_path": str(module.EXPECTED_GUIDED_ROOT / "guided_summary.json"),
        "guided_compute_by_source_path": str(module.EXPECTED_GUIDED_ROOT / "guided_compute_by_source.jsonl"),
    })
    readiness = _readiness()
    readiness["critic"]["refit_checkpoint"] = str(module.EXPECTED_CRITIC_CHECKPOINT)
    guided = _guided()
    guided["per_source_compute_path"] = str(module.EXPECTED_GUIDED_ROOT / "guided_compute_by_source.jsonl")
    budgets = module.validate_inputs(
        config, readiness, _adjudication(),
        _independent_evaluator_adjudication("INDEPENDENT_GENERATION_EVALUATOR_NO_GO"),
        guided,
        [{"source_key": "S1", "matched_search_critic_forward_budget": 3}],
        [{"source_key": "S1"}],
    )
    assert budgets == {"S1": 3}


def test_repository_config_uses_only_critic_v2_artifacts() -> None:
    module = _load()
    current = json.loads(
        (
            ROOT
            / "configs/route_a_v3_route2_mrnabert_critic_v2_matched_search_development_gpu0_v1.json"
        ).read_text(encoding="utf-8")
    )
    historical = json.loads(
        (
            ROOT
            / "configs/route_a_v3_route2_mrnabert_matched_search_development_gpu0_v1.json"
        ).read_text(encoding="utf-8")
    )
    module.validate_config_boundary(current)
    assert historical["status"] == "RETIRED_HISTORICAL_V1_READINESS_PATH_NOT_AUTHORIZED"
    with pytest.raises(module.MatchedSearchSuiteError, match="historical"):
        module.validate_config_boundary(historical)
