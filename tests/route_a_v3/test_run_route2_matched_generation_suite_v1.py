import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "route_a_v3" / "run_route2_matched_generation_suite_v1.py"
PROTOCOL = ROOT / "configs" / "route_a_v3_route2_generation_matched_compute_repair_protocol_v1.json"
JOBS = ROOT / "configs" / "route_a_v3_route2_generation_independent_evaluator_jobs_gpu6_v1.json"
FLOW_CONFIG = ROOT / "configs" / "route_a_v3_route2_base_flow_g0_matched_compute_candidates_seed20260816_gpu6_v1.json"
RUNTIME_PROTOCOL = ROOT / "configs" / "route_a_v3_route2_generation_matched_runtime_valid_protocol_v1.json"
RUNTIME_JOBS = ROOT / "configs" / "route_a_v3_route2_generation_runtime_valid_evaluator_jobs_gpu6_v1.json"
RUNTIME_FLOW_CONFIG = ROOT / "configs" / "route_a_v3_route2_base_flow_g0_matched_runtime_valid_seed20260816_gpu6_v1.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("matched_generation_suite_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _qualified_adjudication() -> dict:
    return {
        "status": "INDEPENDENT_GENERATION_EVALUATOR_QUALIFIED",
        "candidate_rerun_authorized": True,
        "development_test_outcomes_accessed": False,
        "evaluation_outcomes_accessed": False,
    }


def test_suite_plan_covers_exact_methods_and_frozen_budgets(tmp_path: Path) -> None:
    module = _load_module()
    protocol = _load(PROTOCOL)
    jobs = _load(JOBS)
    flow_config = _load(FLOW_CONFIG)
    suite = module.validate_suite_inputs(
        protocol,
        jobs,
        _qualified_adjudication(),
        flow_config,
    )

    assert suite["required_methods"] == protocol["required_method_ids"]
    assert len(suite["required_methods"]) == 7
    assert set(suite["job_by_method"]) == set(protocol["required_method_ids"])
    assert suite["gpu_stage_execution_mode"] == module.PARALLEL_QUALITY_ONLY
    assert suite["runtime_comparison_valid"] is False

    generation = module.build_generation_commands(protocol, FLOW_CONFIG, suite)
    assert {spec["name"] for spec in generation} == set(protocol["required_method_ids"])
    for spec in generation:
        command = spec["command"]
        if spec["name"] == "unguided_learned_base_flow_g0":
            assert str(module.FLOW_SCRIPT) in command
            continue
        assert command[command.index("--max-critic-forwards") + 1] == "256"
        assert command[command.index("--beam-width") + 1] == "16"
        assert command[command.index("--genetic-population-size") + 1] == "32"
        assert command[command.index("--oversample-factor") + 1] == "8"
        assert command[command.index("--exhaustive-space-limit") + 1] == "4096"
        assert command[command.index("--device") + 1] == "cuda:6"
        assert command[command.index("--physical-gpu-index") + 1] == "6"

    scoring = module.build_scoring_configs(jobs, suite, tmp_path / "scoring")
    assert {spec["name"] for spec in scoring} == set(protocol["required_method_ids"])
    for config_path in (tmp_path / "scoring").glob("*.json"):
        config = _load(config_path)
        assert config["evaluator_frozen_before_candidate_generation"] is True
        assert config["evaluation_outcomes_used_to_select_evaluator"] == 0
        assert config["device"] == "cuda:6"
        assert config["physical_gpu_index"] == 6
        assert config["evaluator_checkpoint_path"] != config["guiding_checkpoint_path"]

    evaluations = module.build_evaluation_commands(protocol, suite)
    for spec in evaluations:
        command = spec["command"]
        assert command[command.index("--candidate-support-mode") + 1] == "OPEN_GENERATED_SUPPORT"
        assert command[command.index("--measured-neighborhood-pool") + 1] == "DEVELOPMENT"
        assert command[command.index("--evaluation-release-state") + 1] == "CLOSED"


def test_runtime_valid_suite_requires_serial_same_gpu_execution() -> None:
    module = _load_module()
    protocol = _load(RUNTIME_PROTOCOL)
    suite = module.validate_suite_inputs(
        protocol,
        _load(RUNTIME_JOBS),
        _qualified_adjudication(),
        _load(RUNTIME_FLOW_CONFIG),
    )

    assert suite["gpu_stage_execution_mode"] == module.SERIAL_RUNTIME_VALID
    assert suite["runtime_comparison_valid"] is True
    assert protocol["parallel_gpu_jobs_allowed"] is False
    assert protocol["runtime_comparison_allowed"] is True


def test_runtime_valid_suite_rejects_parallel_gpu_jobs() -> None:
    module = _load_module()
    protocol = _load(RUNTIME_PROTOCOL)
    protocol["parallel_gpu_jobs_allowed"] = True
    with pytest.raises(module.MatchedGenerationSuiteError):
        module.validate_suite_inputs(
            protocol,
            _load(RUNTIME_JOBS),
            _qualified_adjudication(),
            _load(RUNTIME_FLOW_CONFIG),
        )


def test_suite_rejects_flow_seed_mismatch() -> None:
    module = _load_module()
    flow_config = _load(FLOW_CONFIG)
    flow_config["seed"] += 1
    with pytest.raises(module.MatchedGenerationSuiteError):
        module.validate_suite_inputs(
            _load(PROTOCOL),
            _load(JOBS),
            _qualified_adjudication(),
            flow_config,
        )


def test_serial_stage_records_independent_method_wall_times(tmp_path: Path) -> None:
    module = _load_module()
    results = module.run_serial_stage(
        "runtime_test",
        [
            {"name": "first", "command": [sys.executable, "-c", "pass"]},
            {"name": "second", "command": [sys.executable, "-c", "pass"]},
        ],
        tmp_path / "logs",
    )

    assert [row["name"] for row in results] == ["first", "second"]
    assert all(row["return_code"] == 0 for row in results)
    assert all(row["runtime_measurement_valid"] is True for row in results)
    assert all(row["wall_time_seconds"] >= 0.0 for row in results)


def test_runtime_summary_binds_wall_time_and_peak_vram(tmp_path: Path) -> None:
    module = _load_module()
    evaluation_root = tmp_path / "evaluations"
    evaluation_root.mkdir()
    flow_root = tmp_path / "flow"
    flow_root.mkdir()
    methods = ["random_legal", module.UNGUIDED_METHOD]
    jobs = {}
    for index, method_id in enumerate(methods, start=1):
        (evaluation_root / f"{method_id}_evaluation_v2.json").write_text(
            json.dumps(
                {
                    "generation": {
                        "per_source": {
                            "source": {"peak_vram_mb": None if method_id == module.UNGUIDED_METHOD else 100.0 + index}
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        scored_output = tmp_path / f"{method_id}.jsonl"
        scored_output.with_suffix(scored_output.suffix + ".summary.json").write_text(
            json.dumps({"peak_vram_mb": 200.0 + index}),
            encoding="utf-8",
        )
        jobs[method_id] = {"output_path": str(scored_output)}
    (flow_root / "final_summary.json").write_text(
        json.dumps({"peak_vram_mb": 150.0}),
        encoding="utf-8",
    )
    stage_results = {
        "candidate_generation": [
            {
                "name": method_id,
                "wall_time_seconds": float(index),
                "runtime_measurement_valid": True,
            }
            for index, method_id in enumerate(methods, start=1)
        ],
        "independent_evaluator_scoring": [
            {
                "name": method_id,
                "wall_time_seconds": float(index + 2),
                "runtime_measurement_valid": True,
            }
            for index, method_id in enumerate(methods, start=1)
        ],
    }
    result = module.build_method_runtime_summary(
        {"independent_evaluation_output_root": str(evaluation_root)},
        {"output_directory": str(flow_root)},
        {
            "required_methods": methods,
            "job_by_method": jobs,
            "gpu_stage_execution_mode": module.SERIAL_RUNTIME_VALID,
            "runtime_comparison_valid": True,
        },
        stage_results,
    )

    assert result["runtime_comparison_valid"] is True
    assert result["method_runtime"]["random_legal"]["candidate_generation_peak_vram_mb"] == 101.0
    assert result["method_runtime"][module.UNGUIDED_METHOD]["candidate_generation_peak_vram_mb"] == 150.0
    assert result["method_runtime"]["random_legal"]["total_serial_gpu_wall_time_seconds"] == 4.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "INDEPENDENT_GENERATION_EVALUATOR_NO_GO"),
        ("candidate_rerun_authorized", False),
        ("development_test_outcomes_accessed", True),
        ("evaluation_outcomes_accessed", True),
    ],
)
def test_suite_rejects_unqualified_or_exposed_evaluator(field: str, value: object) -> None:
    module = _load_module()
    adjudication = _qualified_adjudication()
    adjudication[field] = value
    with pytest.raises(module.MatchedGenerationSuiteError):
        module.validate_suite_inputs(
            _load(PROTOCOL),
            _load(JOBS),
            adjudication,
            _load(FLOW_CONFIG),
        )
