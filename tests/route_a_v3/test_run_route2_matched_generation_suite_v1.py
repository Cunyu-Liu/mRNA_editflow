import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "route_a_v3" / "run_route2_matched_generation_suite_v1.py"
PROTOCOL = ROOT / "configs" / "route_a_v3_route2_generation_matched_compute_repair_protocol_v1.json"
JOBS = ROOT / "configs" / "route_a_v3_route2_generation_independent_evaluator_jobs_gpu6_v1.json"
FLOW_CONFIG = ROOT / "configs" / "route_a_v3_route2_base_flow_g0_matched_compute_candidates_seed20260816_gpu6_v1.json"


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
    assert suite["evaluator_qualified"] is True
    assert len(suite["required_methods"]) == 7
    assert set(suite["job_by_method"]) == set(protocol["required_method_ids"])

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


@pytest.mark.parametrize(
    "field,value",
    [
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


def test_evaluator_no_go_allows_candidate_generation_but_not_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    protocol = _load(PROTOCOL)
    jobs = _load(JOBS)
    flow_config = _load(FLOW_CONFIG)
    adjudication = {
        "status": "INDEPENDENT_GENERATION_EVALUATOR_NO_GO",
        "candidate_rerun_authorized": False,
        "development_test_outcomes_accessed": False,
        "evaluation_outcomes_accessed": False,
    }
    suite = module.validate_suite_inputs(protocol, jobs, adjudication, flow_config)
    assert suite["evaluator_qualified"] is False

    paths = {
        tmp_path / "protocol.json": protocol,
        tmp_path / "jobs.json": jobs,
        tmp_path / "adjudication.json": adjudication,
        tmp_path / "flow.json": flow_config,
    }
    monkeypatch.setattr(module, "_read", lambda path: paths[path])
    stages = []

    def fake_stage(stage_name, specs, log_directory):
        stages.append(stage_name)
        return [{"name": spec["name"], "return_code": 0} for spec in specs]

    monkeypatch.setattr(module, "run_parallel_stage", fake_stage)
    output = tmp_path / "summary.json"
    result = module.execute(
        protocol_path=tmp_path / "protocol.json",
        jobs_path=tmp_path / "jobs.json",
        evaluator_adjudication_path=tmp_path / "adjudication.json",
        flow_config_path=tmp_path / "flow.json",
        output_summary_path=output,
    )
    assert stages == ["candidate_generation"]
    assert result["status"] == "MATCHED_GENERATION_CANDIDATES_COMPLETED_EVALUATOR_NO_GO"
    assert result["independent_evaluator_scoring_run"] is False
    assert result["strongest_generation_baseline_selected"] is False
    assert result["evaluation_outcomes_accessed"] is False


def test_parallel_stage_persists_each_job_wall_time(tmp_path: Path) -> None:
    module = _load_module()
    results = module.run_parallel_stage(
        "timed",
        [
            {
                "name": "short",
                "command": [sys.executable, "-c", "import time; time.sleep(0.02)"],
            },
            {
                "name": "long",
                "command": [sys.executable, "-c", "import time; time.sleep(0.10)"],
            },
        ],
        tmp_path / "logs",
    )
    assert [row["name"] for row in results] == ["short", "long"]
    assert all(row["return_code"] == 0 for row in results)
    assert all(row["wall_time_seconds"] > 0.0 for row in results)
    assert results[1]["wall_time_seconds"] > results[0]["wall_time_seconds"]
    assert all(Path(row["stdout_path"]).is_file() for row in results)
    assert all(Path(row["stderr_path"]).is_file() for row in results)
