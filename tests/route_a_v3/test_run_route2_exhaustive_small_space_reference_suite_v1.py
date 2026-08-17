import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "route_a_v3" / "run_route2_exhaustive_small_space_reference_suite_v1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("exhaustive_reference_suite_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _inputs():
    config = {
        "schema_version": "route_a_v3_route2_exhaustive_small_space_reference.v1",
        "scientific_role": "REAL_SMALL_SPACE_EXHAUSTIVE_GUIDING_CRITIC_REFERENCE_NOT_FULL_COHORT_STRONGEST_SELECTOR",
        "source_cohort_count": 190,
        "legal_space_size_per_source": 151,
        "candidate_budget_per_source": 32,
        "critic_forward_budget_per_source": 256,
        "forward_equivalent_budget_per_source": 320,
        "source_manifest_path": "/mnt/subset.jsonl",
        "measured_neighborhood_path": "/mnt/subset_measured.jsonl",
        "guiding_checkpoint_path": "/mnt/guide.pt",
        "independent_evaluator_checkpoint_path": "/mnt/evaluator.pt",
        "candidate_output_path": "/mnt/candidates.jsonl",
        "independent_scored_output_path": "/mnt/scored.jsonl",
        "generation_evaluation_output_path": "/mnt/evaluation.json",
        "comparison_output_path": "/mnt/comparison.json",
        "beam_width": 16,
        "genetic_population_size": 32,
        "oversample_factor": 8,
        "exhaustive_space_limit": 4096,
        "seed": 20260816,
        "device": "cuda:6",
        "physical_gpu_index": 6,
        "evaluation_outcomes_accessed": False,
        "full_cohort_strongest_selector_eligible": False,
        "guided_xeditflow_allowed": False,
    }
    scoring = {
        "evaluator_checkpoint_path": "/mnt/evaluator.pt",
        "guiding_checkpoint_path": "/mnt/guide.pt",
        "source_manifest_path": "/mnt/subset.jsonl",
        "candidate_path": "/mnt/candidates.jsonl",
        "output_path": "/mnt/scored.jsonl",
        "evaluator_frozen_before_candidate_generation": True,
        "evaluation_outcomes_used_to_select_evaluator": 0,
        "device": "cuda:6",
        "physical_gpu_index": 6,
    }
    protocol = {
        "candidate_budget_per_source": 32,
        "search_critic_forward_budget_per_source": 256,
        "forward_equivalent_budget_per_source": 320,
        "guiding_checkpoint_path": "/mnt/guide.pt",
        "independent_evaluator_checkpoint_path": "/mnt/evaluator.pt",
        "candidate_support_mode": "OPEN_GENERATED_SUPPORT",
        "evaluation_release_state": "CLOSED",
        "execution_device": "cuda:6",
        "physical_gpu_index": 6,
    }
    suite = {
        "status": "MATCHED_GENERATION_BASELINE_SUITE_COMPLETED",
        "evaluation_outcomes_accessed": False,
        "guided_xeditflow_run": False,
    }
    adjudication = {
        "status": "INDEPENDENT_GENERATION_EVALUATOR_QUALIFIED",
        "candidate_rerun_authorized": True,
        "evaluation_outcomes_accessed": False,
    }
    return config, scoring, protocol, suite, adjudication


def test_validated_inputs_build_the_four_frozen_stages() -> None:
    module = _load_module()
    config, scoring, protocol, suite, adjudication = _inputs()
    module.validate_inputs(config, scoring, protocol, suite, adjudication)
    commands = module.build_commands(config, Path("scoring.json"), protocol, Path("reference.json"))
    assert [row["stage"] for row in commands] == [
        "exhaustive_candidate_generation",
        "independent_evaluator_scoring",
        "development_open_support_evaluation",
        "small_space_reference_comparison",
    ]
    generation = commands[0]["command"]
    assert generation[generation.index("--method") + 1] == "exhaustive"
    assert generation[generation.index("--device") + 1] == "cuda:6"
    assert generation[generation.index("--max-critic-forwards") + 1] == "256"
    evaluation = commands[2]["command"]
    assert evaluation[evaluation.index("--measured-neighborhood") + 1] == "/mnt/subset_measured.jsonl"
    assert evaluation[evaluation.index("--candidate-support-mode") + 1] == "OPEN_GENERATED_SUPPORT"


def test_exhaustive_reference_may_use_another_cuda_device_than_full_suite() -> None:
    module = _load_module()
    config, scoring, protocol, suite, adjudication = _inputs()
    config["device"] = "cuda:2"
    config["physical_gpu_index"] = 2
    scoring["device"] = "cuda:2"
    scoring["physical_gpu_index"] = 2
    module.validate_inputs(config, scoring, protocol, suite, adjudication)
    commands = module.build_commands(config, Path("scoring.json"), protocol, Path("reference.json"))
    generation = commands[0]["command"]
    assert generation[generation.index("--device") + 1] == "cuda:2"
    assert generation[generation.index("--physical-gpu-index") + 1] == "2"


def test_exhaustive_reference_rejects_cpu_execution() -> None:
    module = _load_module()
    config, scoring, protocol, suite, adjudication = _inputs()
    config["device"] = "cpu"
    scoring["device"] = "cpu"
    with pytest.raises(module.ExhaustiveReferenceSuiteError):
        module.validate_inputs(config, scoring, protocol, suite, adjudication)


def test_validation_rejects_incomplete_full_suite() -> None:
    module = _load_module()
    config, scoring, protocol, suite, adjudication = _inputs()
    suite["status"] = "MATCHED_GENERATION_BASELINE_SUITE_FAILED"
    with pytest.raises(module.ExhaustiveReferenceSuiteError):
        module.validate_inputs(config, scoring, protocol, suite, adjudication)


def test_comparison_output_requires_measured_open_support_without_zero_imputation() -> None:
    module = _load_module()
    comparison = {
        "status": "SMALL_SPACE_EXHAUSTIVE_GUIDING_CRITIC_REFERENCE_COMPLETED",
        "measured_neighborhood_comparison_included": True,
        "measured_candidate_support_mode": "OPEN_GENERATED_SUPPORT",
        "unknown_generated_outcomes_treated_as_zero": False,
        "measured_superiority_claim_established": False,
    }
    module.validate_comparison_output(comparison)
    comparison["unknown_generated_outcomes_treated_as_zero"] = True
    with pytest.raises(module.ExhaustiveReferenceSuiteError):
        module.validate_comparison_output(comparison)


def test_stage_failure_is_terminal_and_preserves_log_paths(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=7),
    )
    with pytest.raises(module.StageFailure) as exc:
        module.run_stage({"stage": "candidate", "command": ["false"]}, tmp_path)
    assert exc.value.result["return_code"] == 7
    assert exc.value.result["stdout_path"].endswith("candidate.stdout.log")
    assert exc.value.result["stderr_path"].endswith("candidate.stderr.log")
