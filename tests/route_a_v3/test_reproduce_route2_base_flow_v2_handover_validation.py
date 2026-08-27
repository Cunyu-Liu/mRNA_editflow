from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from scripts.route_a_v3 import (
    reproduce_route2_base_flow_v2_handover_validation as module,
)


CURRENT_HEAD = "a" * 40
FINAL_HEAD = "b" * 40
TERMINAL_COMMIT = "c" * 40


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _git_observation() -> dict[str, Any]:
    return {
        "head_return_code": 0,
        "observed_git_head": CURRENT_HEAD,
        "status_return_code": 0,
        "status_stdout": "",
        "status_stderr": "",
        "worktree_clean": True,
    }


def _write_final_authority(
    tmp_path: Path, gate_status: str = "XEDITFLOW_V4_PASS"
) -> dict[str, Path]:
    runtime_path = tmp_path / "final" / "runtime.json"
    adjudication_path = tmp_path / "final" / "final_adjudication.json"
    launch_path = tmp_path / "final" / "launch.json"
    jobs = {
        f"job_{index:03d}": {
            "status": "TERMINAL_COMPLETE",
            "return_code": 0,
            "terminal_artifact_kind": "SUCCESS",
        }
        for index in range(module.EXPECTED_FINAL_JOB_COUNT)
    }
    _write_json(
        runtime_path,
        {
            "schema_version": module.FINAL_RUNTIME_SCHEMA,
            "status": module.FINAL_TERMINAL_STATUS,
            "git_head": FINAL_HEAD,
            "experiment_head": "d" * 40,
            "guidance_runner_head": "e" * 40,
            "jobs": jobs,
            "first_terminal_failure": None,
            "active_performance_output_read": False,
            "development_test_reopened": False,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcome_reads": 0,
        },
    )
    passed = gate_status == "XEDITFLOW_V4_PASS"
    _write_json(
        adjudication_path,
        {
            "schema_version": module.FINAL_ADJUDICATION_SCHEMA,
            "status": module.FINAL_TERMINAL_STATUS,
            "gate": {
                "schema_version": module.FINAL_GATE_SCHEMA,
                "status": gate_status,
                "new_final_evaluation_authorized": passed,
                "additional_training_seed_authorized": False,
                "submission_ready": False,
            },
            "predictor_generator_baselines_metrics_policy_frozen": True,
            "new_final_evaluation_authorized": passed,
            "additional_training_seed_authorized": False,
            "submission_ready": False,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcomes_accessed": False,
        },
    )
    _write_json(
        launch_path,
        {
            "schema_version": module.FINAL_LAUNCH_SCHEMA,
            "status": "XEDITFLOW_V4_FINAL_SCHEDULER_LAUNCHED",
            "git_head": FINAL_HEAD,
            "experiment_head": "d" * 40,
            "guidance_runner_head": "e" * 40,
            "runtime_manifest": str(runtime_path),
            "final_adjudication_path": str(adjudication_path),
            "development_test_reopened": False,
            "new_final_evaluation_outcome_reads": 0,
        },
    )
    return {
        "final_launch_receipt_path": launch_path,
        "final_runtime_path": runtime_path,
        "final_adjudication_path": adjudication_path,
    }


def _write_terminal_inputs(tmp_path: Path, *, full_cohort: bool) -> dict[str, Path]:
    source_manifest = tmp_path / "development" / "source.jsonl"
    measured = tmp_path / "development" / "measured.private.jsonl"
    candidate_dir = tmp_path / "terminal_candidates"
    candidates = candidate_dir / "trajectories.private.jsonl"
    training_dir = tmp_path / "training"
    checkpoint = training_dir / "best.pt"
    training_summary = training_dir / "training_summary.json"
    provenance = training_dir / "training_attempt.json"
    config = candidate_dir / "validation_config.json"
    validation_summary = candidate_dir / "validation_summary.json"
    expected = tmp_path / "tracked_expected.csv"
    source_count = module.EXPECTED_SOURCE_COUNT if full_cohort else 1
    source_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    measured_rows: list[dict[str, Any]] = []
    for index in range(source_count):
        source_key = f"source_{index:04d}"
        source_rows.append(
            {
                "source_key": source_key,
                "source_sequence": "AAAA",
                "edit_budget": 1,
                "candidate_budget": module.EXPECTED_CANDIDATE_CAP,
            }
        )
        for candidate_index in range(module.EXPECTED_CANDIDATE_CAP):
            first = candidate_index < module.EXPECTED_CANDIDATE_CAP // 2
            candidate_rows.append(
                {
                    "method_id": module.METHOD_ID,
                    "source_key": source_key,
                    "candidate_sequence": "CAAA" if first else "GAAA",
                    "terminal_cause": "BUDGET_EXHAUSTED",
                    "generation_score": 1.0 if first else 0.0,
                    "generator_nfe": 1,
                    "critic_forwards": 0,
                    "independent_evaluator_forwards": 0,
                }
            )
        measured_rows.extend(
            (
                {
                    "source_key": source_key,
                    "candidate_sequence": "CAAA",
                    "measured_direction_normalized_delta": 1.0,
                    "pool_assignment": "DEVELOPMENT",
                },
                {
                    "source_key": source_key,
                    "candidate_sequence": "GAAA",
                    "measured_direction_normalized_delta": 0.0,
                    "pool_assignment": "DEVELOPMENT",
                },
            )
        )
    _write_jsonl(source_manifest, source_rows)
    _write_jsonl(candidates, candidate_rows)
    _write_jsonl(measured, measured_rows)
    _write_json(
        config,
        {
            "schema_version": module.TERMINAL_CONFIG_SCHEMA,
            "checkpoint_path": str(checkpoint),
            "source_eligibility_manifest": str(source_manifest),
            "seed": 20260816,
            "device": "cuda:2",
            "physical_gpu_index": 2,
            "guided_critic_used": False,
            "evaluation_outcomes_accessed": False,
            "output_directory": str(candidate_dir),
        },
    )
    _write_json(
        training_summary,
        {
            "schema_version": module.TERMINAL_TRAINING_SCHEMA,
            "status": "LEARNED_BASE_FLOW_GPU_UPDATE_COMPLETE",
            "seed": 20260816,
            "physical_gpu_index": 0,
            "torch_device": "cuda:0",
            "cpu_fallback_used": False,
            "cuda_training_tensors_verified": True,
            "optimizer_steps": 12,
            "parameter_changed": True,
            "development_test_outcomes_evaluated": False,
            "evaluation_records_read": 0,
            "guided_critic_used": False,
            "biological_optimization_established": False,
            "cuda_device_index": 0,
            "cuda_device_uuid": "GPU-TRAINING",
            "cuda_total_memory_mb": 81920.0,
        },
    )
    _write_json(
        provenance,
        {
            "status": "COMPLETED",
            "code_commit": TERMINAL_COMMIT,
            "seed": 20260816,
            "output_directory": str(training_dir),
            "device": "cuda:0",
            "physical_gpu_index": 0,
            "evaluation_record_count": 0,
        },
    )
    _write_json(
        validation_summary,
        {
            "schema_version": module.TERMINAL_VALIDATION_SCHEMA,
            "status": "FLOW_G0_READY",
            "source_budget_cohort_count": module.EXPECTED_SOURCE_COUNT,
            "trajectory_count": module.EXPECTED_CANDIDATE_COUNT,
            "hard_legality_rate": 1.0,
            "edit_budget_violation_count": 0,
            "candidate_budget_violation_count": 0,
            "numerical_failure_count": 0,
            "trajectory_replay_failure_count": 0,
            "learned_parameter_update_checkpoint_loaded": True,
            "physical_gpu_index": 2,
            "device": "cuda:2",
            "cpu_fallback_used": False,
            "checkpoint_gpu_parameter_update_provenance_verified": True,
            "checkpoint_training_device": "cuda:0",
            "checkpoint_training_physical_gpu_index": 0,
            "checkpoint_cpu_fallback_used": False,
            "checkpoint_training_seed": 20260816,
            "checkpoint_training_optimizer_steps": 4,
            "checkpoint_parameter_changed": True,
            "checkpoint_cuda_training_tensors_verified": True,
            "checkpoint_training_cuda_device_index": 0,
            "checkpoint_training_cuda_device_uuid": "GPU-TRAINING",
            "checkpoint_training_cuda_total_memory_mb": 81920.0,
            "evaluation_outcomes_read": 0,
            "guided_critic_used": False,
            "generated_candidates_grant_canonical_credit": False,
            "biological_optimization_established": False,
        },
    )
    with expected.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "method_id",
                "source_count",
                "candidate_count",
                "candidate_cap_per_source",
                "hard_legality_rate",
                "edit_budget_violation_count",
                "candidate_budget_violation_count",
                "source_macro_unique_candidate_rate",
                "source_macro_candidate_recovery_rate",
                "source_macro_measured_top_k_recovery_at_k",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "method_id": module.METHOD_ID,
                "source_count": module.EXPECTED_SOURCE_COUNT,
                "candidate_count": module.EXPECTED_CANDIDATE_COUNT,
                "candidate_cap_per_source": module.EXPECTED_CANDIDATE_CAP,
                "hard_legality_rate": 1.0,
                "edit_budget_violation_count": 0,
                "candidate_budget_violation_count": 0,
                "source_macro_unique_candidate_rate": 0.0625,
                "source_macro_candidate_recovery_rate": 1.0,
                "source_macro_measured_top_k_recovery_at_k": 1.0,
            }
        )
    return {
        "terminal_config_path": config,
        "terminal_validation_summary_path": validation_summary,
        "terminal_training_summary_path": training_summary,
        "terminal_provenance_path": provenance,
        "source_manifest_path": source_manifest,
        "terminal_candidates_path": candidates,
        "development_measured_neighborhood_path": measured,
        "tracked_expected_csv_path": expected,
    }


def _kwargs(
    tmp_path: Path,
    *,
    gate_status: str = "XEDITFLOW_V4_PASS",
    full_cohort: bool = True,
) -> dict[str, Any]:
    return {
        **_write_final_authority(tmp_path, gate_status),
        **_write_terminal_inputs(tmp_path, full_cohort=full_cohort),
        "expected_head": CURRENT_HEAD,
        "output_dir": tmp_path / "handover_reproduction",
        "command": ["python", "reproduce.py", "--synthetic-test"],
        "git_observation": _git_observation(),
    }


@pytest.mark.parametrize(
    "gate_status",
    ("XEDITFLOW_V4_PASS", "XEDITFLOW_V4_NO_GO"),
)
def test_pass_and_no_go_both_allow_cpu_aggregation_only_reproduction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate_status: str,
) -> None:
    def forbidden_subprocess(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no generation/GPU subprocess may start")

    monkeypatch.setattr(module.subprocess, "run", forbidden_subprocess)
    monkeypatch.setattr(module.subprocess, "Popen", forbidden_subprocess)
    arguments = _kwargs(tmp_path, gate_status=gate_status)
    output_dir = arguments["output_dir"]
    result = module.run_reproduction(**arguments)

    assert result["status"] == "BASE_FLOW_V2_HANDOVER_R3_REPRODUCTION_COMPLETE"
    assert result["final_gate_status"] == gate_status
    assert {
        "command.txt",
        "environment.txt",
        "git_status.txt",
        "config.snapshot.json",
        "stdout.log",
        "metrics.json",
        "comparison_to_frozen.json",
        "README.md",
    } == {path.name for path in output_dir.iterdir()}
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["generation"]["source_count"] == 891
    assert metrics["generation"]["candidate_count"] == 28_512
    assert metrics["measured_neighborhood"][
        "source_macro_closed_measured_ndcg_at_k"
    ] is None
    assert metrics["additional_protected_outcome_reads"] == 0
    assert metrics["model_forward_run"] is False
    assert metrics["cuda_queried"] is False
    assert metrics["cpu_fallback_used"] is False
    snapshot = json.loads(
        (output_dir / "config.snapshot.json").read_text(encoding="utf-8")
    )
    assert snapshot["terminal_validation_summary"]["status"] == "FLOW_G0_READY"
    assert snapshot["terminal_provenance"][
        "terminal_validation_candidate_count"
    ] == 28_512
    environment = (output_dir / "environment.txt").read_text(encoding="utf-8")
    assert "execution_mode=CPU_AGGREGATION_ONLY" in environment
    assert "generation_run=false" in environment
    assert "gpu_validation_run=false" in environment
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "validation_summary.json" in readme
    assert "891 sources / 28,512 candidates / hard legality 1.0" in readme


@pytest.mark.parametrize(
    "mutate,expected_message",
    (
        (
            lambda runtime: runtime.update(
                {"status": "XEDITFLOW_V4_FINAL_COMPARISON_RUNNING"}
            ),
            "not the exact terminal runtime",
        ),
        (
            lambda runtime: runtime.update(
                {"new_final_evaluation_outcome_reads": 1}
            ),
            "new Evaluation outcome reads",
        ),
    ),
)
def test_nonterminal_or_protected_violation_stops_before_data_readers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, Any]], None],
    expected_message: str,
) -> None:
    arguments = _kwargs(tmp_path, full_cohort=False)
    runtime_path = arguments["final_runtime_path"]
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    mutate(runtime)
    _write_json(runtime_path, runtime)

    def protected_reader(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("protected Development rows opened before authority")

    monkeypatch.setattr(module.evaluator, "load_source_manifest", protected_reader)
    monkeypatch.setattr(module.evaluator, "_read_jsonl", protected_reader)
    with pytest.raises(
        module.BaseFlowV2HandoverReproductionError, match=expected_message
    ):
        module.run_reproduction(**arguments)
    assert not arguments["output_dir"].exists()


@pytest.mark.parametrize(
    "summary_state,expected_message",
    (
        ("MISSING", "cannot read terminal Base Flow producer validation summary"),
        ("NONTERMINAL", "not exact terminal FLOW_G0_READY"),
    ),
)
def test_missing_or_nonterminal_producer_summary_stops_before_data_readers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    summary_state: str,
    expected_message: str,
) -> None:
    arguments = _kwargs(tmp_path, full_cohort=False)
    summary_path = arguments["terminal_validation_summary_path"]
    if summary_state == "MISSING":
        summary_path.unlink()
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["status"] = "FLOW_G0_VALIDATION_FAIL"
        _write_json(summary_path, summary)

    def protected_reader(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("protected Development rows opened before producer terminal")

    monkeypatch.setattr(module.evaluator, "load_source_manifest", protected_reader)
    monkeypatch.setattr(module.evaluator, "_read_jsonl", protected_reader)
    with pytest.raises(
        module.BaseFlowV2HandoverReproductionError, match=expected_message
    ):
        module.run_reproduction(**arguments)
    assert not arguments["output_dir"].exists()


@pytest.mark.parametrize("checkpoint_optimizer_steps", (0, 13))
def test_checkpoint_optimizer_steps_must_be_positive_and_not_exceed_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_optimizer_steps: int,
) -> None:
    arguments = _kwargs(tmp_path, full_cohort=False)
    summary_path = arguments["terminal_validation_summary_path"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["checkpoint_training_optimizer_steps"] = checkpoint_optimizer_steps
    _write_json(summary_path, summary)

    def protected_reader(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("protected Development rows opened after bad provenance")

    monkeypatch.setattr(module.evaluator, "load_source_manifest", protected_reader)
    monkeypatch.setattr(module.evaluator, "_read_jsonl", protected_reader)
    with pytest.raises(
        module.BaseFlowV2HandoverReproductionError,
        match="checkpoint GPU provenance differs",
    ):
        module.run_reproduction(**arguments)
    assert not arguments["output_dir"].exists()


def test_recompute_calls_shared_evaluator_on_terminal_rows(tmp_path: Path) -> None:
    inputs = _write_terminal_inputs(tmp_path, full_cohort=False)
    result = module.recompute_terminal_metrics(
        inputs["source_manifest_path"],
        inputs["terminal_candidates_path"],
        inputs["development_measured_neighborhood_path"],
        k=10,
    )
    assert result["shared_evaluator_module"] == (
        "scripts.route_a_v3.evaluate_route2_generation_v1"
    )
    assert result["frozen_evaluation_json_copied"] is False
    assert result["generation"]["source_macro_unique_candidate_rate"] == 0.0625
    assert result["measured_neighborhood"][
        "source_macro_candidate_recovery_rate"
    ] == 1.0
    assert result["measured_neighborhood"][
        "source_macro_measured_top_k_recovery_at_k"
    ] == 1.0


def _comparison_metrics(
    *,
    unique_rate: float = 0.0,
    recovery_rate: float = 0.0,
    top_k_recovery: float = 0.0,
    closed_ndcg: float | None = None,
    defined_count: int = 0,
) -> dict[str, Any]:
    generation_per_source = {
        f"source_{index:04d}": {
            "candidate_count": module.EXPECTED_CANDIDATE_CAP,
            "candidate_budget": module.EXPECTED_CANDIDATE_CAP,
        }
        for index in range(module.EXPECTED_SOURCE_COUNT)
    }
    measured_per_source = {
        f"source_{index:04d}": {
            "closed_measured_ndcg_at_k": closed_ndcg,
            "closed_measured_ndcg_status": module.OPEN_SUPPORT_STATUS,
        }
        for index in range(module.EXPECTED_SOURCE_COUNT)
    }
    return {
        "generation": {
            "method_id": module.METHOD_ID,
            "source_count": module.EXPECTED_SOURCE_COUNT,
            "candidate_count": module.EXPECTED_CANDIDATE_COUNT,
            "hard_legality_rate": 1.0,
            "edit_budget_violation_count": 0,
            "candidate_budget_violation_count": 0,
            "source_macro_unique_candidate_rate": unique_rate,
            "per_source": generation_per_source,
        },
        "measured_neighborhood": {
            "candidate_support_mode": "OPEN_GENERATED_SUPPORT",
            "unknown_generated_candidates_are_zero_gain": False,
            "source_macro_candidate_recovery_rate": recovery_rate,
            "source_macro_measured_top_k_recovery_at_k": top_k_recovery,
            "source_macro_closed_measured_ndcg_at_k": closed_ndcg,
            "source_closed_measured_ndcg_defined_count": defined_count,
            "per_source": measured_per_source,
        },
    }


def _expected_row() -> dict[str, str]:
    return {
        "method_id": module.METHOD_ID,
        "source_count": str(module.EXPECTED_SOURCE_COUNT),
        "candidate_count": str(module.EXPECTED_CANDIDATE_COUNT),
        "candidate_cap_per_source": str(module.EXPECTED_CANDIDATE_CAP),
        "hard_legality_rate": "1.0",
        "edit_budget_violation_count": "0",
        "candidate_budget_violation_count": "0",
        "source_macro_unique_candidate_rate": "0.0",
        "source_macro_candidate_recovery_rate": "0.0",
        "source_macro_measured_top_k_recovery_at_k": "0.0",
    }


def test_continuous_metric_absolute_tolerance_has_an_exact_boundary() -> None:
    result = module.compare_to_tracked_expected(
        _comparison_metrics(
            unique_rate=module.FLOAT_ABS_TOLERANCE,
            recovery_rate=module.FLOAT_ABS_TOLERANCE,
            top_k_recovery=module.FLOAT_ABS_TOLERANCE,
        ),
        _expected_row(),
    )
    assert result["matched"] is True
    with pytest.raises(
        module.BaseFlowV2HandoverReproductionError,
        match="exceeds absolute tolerance",
    ):
        module.compare_to_tracked_expected(
            _comparison_metrics(
                unique_rate=module.FLOAT_ABS_TOLERANCE * 1.000001
            ),
            _expected_row(),
        )


def test_numeric_zero_closed_ndcg_is_rejected_as_defined() -> None:
    with pytest.raises(
        module.BaseFlowV2HandoverReproductionError,
        match="must remain undefined",
    ):
        module.compare_to_tracked_expected(
            _comparison_metrics(closed_ndcg=0.0, defined_count=0),
            _expected_row(),
        )


@pytest.mark.parametrize("existing_kind", ("output", "partial"))
def test_existing_output_or_partial_is_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_kind: str,
) -> None:
    output = tmp_path / "handover_reproduction"
    existing = output if existing_kind == "output" else output.with_name(
        output.name + ".partial"
    )
    existing.mkdir()
    marker = existing / "marker.txt"
    marker.write_text("retain\n", encoding="utf-8")

    def forbidden_read(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("inputs must not open when output identity is occupied")

    monkeypatch.setattr(module, "_read_json", forbidden_read)
    with pytest.raises(
        module.BaseFlowV2HandoverReproductionError, match="already exists"
    ):
        module.run_reproduction(
            final_launch_receipt_path=tmp_path / "launch.json",
            final_runtime_path=tmp_path / "runtime.json",
            final_adjudication_path=tmp_path / "adjudication.json",
            terminal_config_path=tmp_path / "config.json",
            terminal_validation_summary_path=tmp_path / "validation_summary.json",
            terminal_training_summary_path=tmp_path / "training_summary.json",
            terminal_provenance_path=tmp_path / "provenance.json",
            source_manifest_path=tmp_path / "source.jsonl",
            terminal_candidates_path=tmp_path / "candidates.jsonl",
            development_measured_neighborhood_path=tmp_path / "measured.jsonl",
            tracked_expected_csv_path=tmp_path / "expected.csv",
            expected_head=CURRENT_HEAD,
            output_dir=output,
            git_observation=_git_observation(),
        )
    assert marker.read_text(encoding="utf-8") == "retain\n"
