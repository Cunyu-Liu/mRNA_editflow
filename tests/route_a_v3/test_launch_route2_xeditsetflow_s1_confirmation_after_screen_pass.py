from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.route_a_v3.launch_route2_xeditsetflow_s1_confirmation_after_screen_pass as launcher


HEAD = "a" * 40


def test_confirmation_launcher_is_bound_to_the_permitted_execution_branch() -> None:
    assert launcher.BRANCH == "route-a-v3-v403-no-vram-gate-20260827"


def _protocol(tmp_path: Path) -> dict:
    return {
        "runner_outputs": {
            "runtime_config_root_template": str(
                tmp_path / "configs_{runner_git_head}"
            ),
            "training_runtime_root_template": str(
                tmp_path / "training_{runner_git_head}"
            ),
            "training_log_root_template": str(
                tmp_path / "training_logs_{runner_git_head}"
            ),
            "posttraining_runtime_root_template": str(
                tmp_path / "posttraining_{runner_git_head}"
            ),
            "posttraining_log_root_template": str(
                tmp_path / "posttraining_logs_{runner_git_head}"
            ),
            "confirmation_gate_output_template": str(
                tmp_path / "posttraining_{runner_git_head}/confirmation_gate.json"
            ),
            "authorization_output_template": str(
                tmp_path / "authorization_{runner_git_head}.json"
            ),
            "prelaunch_failure_template": str(
                tmp_path / "prelaunch_{runner_git_head}.failed.json"
            ),
        }
    }


def _configs(tmp_path: Path) -> dict[int, Path]:
    result = {}
    for seed in launcher.CONFIRMATION_SEEDS:
        path = tmp_path / f"seed_{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "output_root": str(tmp_path / f"training/seed_{seed}"),
                    "screen_gate_path": str(tmp_path / "screen_gate.json"),
                    "screen_selected_checkpoint_pass": 8,
                    "screen_provenance": {"screen_runner_git_head": launcher.SCREEN_HEAD},
                }
            )
        )
        result[seed] = path
    return result


def _terminal_screen(tmp_path: Path) -> tuple[dict, dict, dict, Path, Path]:
    runtime_path = tmp_path / "runtime.json"
    gate_path = tmp_path / "screen_gate.json"
    gate_path.write_text("{}\n")
    adjudication_failure = tmp_path / "screen_gate.failed.json"
    training_queues = []
    validation_queues = []
    training_states = {}
    validation_states = {}
    for index, run_id in enumerate(("v4_s1_full", "v4_s1_single_mode")):
        output = tmp_path / f"training_{run_id}"
        output.mkdir()
        summary = output / "training_summary.json"
        summary.write_text("{}")
        failure = output / "failure.json"
        key = f"training:{run_id}"
        job = {
            "job_key": key,
            "run_id": run_id,
            "physical_gpu_index": index,
            "terminal_summary": str(summary),
            "terminal_failure": str(failure),
        }
        training_queues.append({"physical_gpu_index": index, "jobs": [job]})
        training_states[key] = {
            "status": "TERMINAL_COMPLETE",
            "terminal_artifact_kind": "SUMMARY",
            "return_code": 0,
            "run_id": run_id,
            "physical_gpu_index": index,
            "terminal_summary": str(summary),
            "terminal_failure": str(failure),
        }
    for index in range(8):
        output = tmp_path / f"validation_{index}"
        output.mkdir()
        summary = output / "validation_summary.json"
        summary.write_text("{}")
        failure = output.with_name(output.name + ".failed.json")
        key = f"validation:{index}"
        job = {
            "job_key": key,
            "run_id": "v4_s1_full" if index < 4 else "v4_s1_single_mode",
            "checkpoint_pass": (4, 6, 8, 10)[index % 4],
            "physical_gpu_index": index % 6,
            "terminal_summary": str(summary),
            "terminal_failure": str(failure),
        }
        validation_queues.append(
            {"physical_gpu_index": index % 6, "jobs": [job]}
        )
        validation_states[key] = {
            "status": "TERMINAL_COMPLETE",
            "terminal_artifact_kind": "SUMMARY",
            "return_code": 0,
            "run_id": job["run_id"],
            "checkpoint_pass": job["checkpoint_pass"],
            "physical_gpu_index": job["physical_gpu_index"],
            "terminal_summary": str(summary),
            "terminal_failure": str(failure),
        }
    schedule = {
        "runtime_manifest": str(runtime_path),
        "training_queues": training_queues,
        "validation_queues": validation_queues,
        "adjudication": {
            "gate_path": str(gate_path),
            "failure_path": str(adjudication_failure),
        },
    }
    runtime = {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v4_s1_screen_runtime.v1"
        ),
        "status": "XEDITSETFLOW_V4_S1_SCREEN_AND_GATE_TERMINAL",
        "git_head": launcher.SCREEN_HEAD,
        "objective_identity": launcher.OBJECTIVE_IDENTITY,
        "cross_state_candidate_mode_responsibility_weight": .05,
        "first_terminal_failure": None,
        "free_memory_gate_applied": False,
        "active_performance_output_read": False,
        "training_jobs": training_states,
        "validation_jobs": validation_states,
        "adjudication": {
            "status": "TERMINAL_COMPLETE",
            "terminal_artifact_kind": "GATE",
            "return_code": 0,
            "gate_present": True,
            "failure_present": False,
            "gate_path": str(gate_path),
            "failure_path": str(adjudication_failure),
        },
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    gate = {
        "status": "XEDITSETFLOW_V4_S1_SCREEN_PASS",
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    return schedule, runtime, gate, runtime_path, gate_path


def test_training_schedule_is_three_full_only_and_low_memory_never_changes_it(
    tmp_path: Path,
) -> None:
    diagnostics = {
        gpu: {
            "name": "NVIDIA A100-SXM4-80GB",
            "free_memory_mib": 1 if gpu == 0 else 79_000,
            "total_memory_mib": 81_920,
        }
        for gpu in range(6)
    }
    probes = {
        gpu: {"device_class": "A100", "device_type": "cuda", "dtype": "BF16"}
        for gpu in (0, 1, 2)
    }
    schedule = launcher.build_training_schedule_s1(
        _protocol(tmp_path),
        tmp_path / "manifest.json",
        tmp_path / "authorization.json",
        _configs(tmp_path),
        (0, 1, 2),
        diagnostics,
        probes,
        {"shared": {}, "setflow": {}},
        runner_head=HEAD,
    )
    jobs = [job for queue in schedule["gpu_queues"] for job in queue["jobs"]]
    assert len(jobs) == 3
    assert {job["run_id"] for job in jobs} == {"v4_s1_full"}
    assert [job["training_seed"] for job in jobs] == [20260912, 20260913, 20260914]
    assert [queue["physical_gpu_index"] for queue in schedule["gpu_queues"]] == [0, 1, 2]
    assert schedule["single_mode_training_job_count"] == 0
    assert schedule["free_memory_gate_applied"] is False
    assert schedule["gpu_diagnostics_before_launch"]["0"]["free_memory_mib"] == 1


def test_screen_runtime_requires_exact_two_plus_eight_zero_exit_unique_summaries(
    tmp_path: Path,
) -> None:
    schedule, runtime, gate, runtime_path, gate_path = _terminal_screen(tmp_path)
    launcher.validate_screen_runtime_terminal_s1(
        schedule,
        runtime,
        gate,
        runtime_path=runtime_path,
        gate_path=gate_path,
    )
    first = next(iter(runtime["validation_jobs"].values()))
    first["return_code"] = 3
    with pytest.raises(launcher.XEditSetFlowS1ConfirmationLaunchError):
        launcher.validate_screen_runtime_terminal_s1(
            schedule,
            runtime,
            gate,
            runtime_path=runtime_path,
            gate_path=gate_path,
        )
    first["return_code"] = 0
    first["terminal_summary"] = str(tmp_path / "wrong_seed_summary.json")
    with pytest.raises(launcher.XEditSetFlowS1ConfirmationLaunchError):
        launcher.validate_screen_runtime_terminal_s1(
            schedule,
            runtime,
            gate,
            runtime_path=runtime_path,
            gate_path=gate_path,
        )


def test_known_unseeded_screen_head_cannot_authorize_confirmation() -> None:
    repair = {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v4_s1_seed_initialization_repair.v1"
        ),
        "status": (
            "XEDITSETFLOW_V4_S1_SEED_INITIALIZATION_REPAIR_FROZEN_BEFORE_INDEPENDENT_RETRY"
        ),
        "affected_family": {"runner_git_head": launcher.SCREEN_HEAD},
        "defect": {"affected_family_can_authorize_successor": False},
    }
    with pytest.raises(
        launcher.XEditSetFlowS1ConfirmationLaunchError,
        match="uncontrolled parameter initialization",
    ):
        launcher.require_seed_valid_screen_head_s1(
            repair, screen_head=launcher.SCREEN_HEAD
        )


def test_screen_no_go_stops_before_any_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    family = tmp_path / "family"

    def reject(*_args, **_kwargs):
        raise RuntimeError("S1 screen gate is NO_GO")

    monkeypatch.setattr(launcher, "validate_screen_pass_barrier_s1", reject)
    with pytest.raises(RuntimeError, match="NO_GO"):
        launcher.validate_screen_bundle_s1(
            {},
            {},
            {},
            {},
            {},
            {},
            {"status": "XEDITSETFLOW_V4_S1_SCREEN_NO_GO"},
            screen_schedule_path=tmp_path / "schedule.json",
            screen_runtime_path=tmp_path / "runtime.json",
            screen_runtime_config_path=tmp_path / "runtime_config.json",
            screen_authorization_path=tmp_path / "authorization.json",
            screen_gate_path=tmp_path / "screen_gate.json",
        )
    assert not family.exists()


def test_inventory_failure_is_sibling_evidence_before_all_family_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = (tmp_path / "configs", tmp_path / "training", tmp_path / "posttraining")
    failure = tmp_path / "audit" / "prelaunch.failed.json"

    def fail(_gpus):
        raise launcher.XEditSetFlowS1GpuError(
            "inventory parse failed", reason="OUTPUT_PARSE_FAILED"
        )

    monkeypatch.setattr(launcher, "gpu_diagnostics", fail)
    with pytest.raises(launcher.XEditSetFlowS1GpuError):
        launcher.perform_gpu_preflight_s1(
            runner_head=HEAD,
            configured_gpus=tuple(range(6)),
            selected_gpus=(0, 1, 2),
            failure_path=failure,
            family_roots=roots,
        )
    payload = json.loads(failure.read_text())
    assert payload["failure_stage"] == "GPU0_5_INVENTORY"
    assert payload["family_roots_created"] == [False, False, False]
    assert not any(path.exists() for path in roots)


def test_both_receipts_must_cover_new_s1_modules_not_only_old_counts() -> None:
    commands = ["pytest old_group.py" for _ in range(8)]
    receipt = {"focused_tests": {"command": commands}}
    with pytest.raises(
        launcher.XEditSetFlowS1ConfirmationLaunchError,
        match="S1-specific focused coverage",
    ):
        launcher.validate_s1_receipt_marker_coverage(receipt, label="old")
    commands[0] += " " + " ".join(launcher.S1_FOCUSED_TEST_MARKERS)
    launcher.validate_s1_receipt_marker_coverage(receipt, label="current")


def test_protected_drift_and_scheduler_launch_failure_are_fail_closed(
    tmp_path: Path,
) -> None:
    schedule, runtime, gate, runtime_path, gate_path = _terminal_screen(tmp_path)
    runtime["development_test_outcome_reads"] = 1
    with pytest.raises(
        launcher.XEditSetFlowS1ConfirmationLaunchError, match="protected"
    ):
        launcher.validate_screen_runtime_terminal_s1(
            schedule,
            runtime,
            gate,
            runtime_path=runtime_path,
            gate_path=gate_path,
        )
    evidence = tmp_path / "scheduler_launch.failed.json"
    launcher.write_scheduler_launch_failure_s1(
        evidence,
        runner_head=HEAD,
        command_line=["python", "scheduler.py", "--schedule", "schedule.json"],
        schedule_path=tmp_path / "schedule.json",
        runtime_path=tmp_path / "runtime.json",
        error=OSError("Popen failed"),
    )
    payload = json.loads(evidence.read_text())
    assert payload["failure_stage"] == (
        "CONFIRMATION_TRAINING_SCHEDULER_PROCESS_LAUNCH"
    )
    assert payload["scheduler_command"] == [
        "python",
        "scheduler.py",
        "--schedule",
        "schedule.json",
    ]
    assert payload["scheduler_started"] is False


def test_cli_requires_explicit_screen_runtime() -> None:
    source = Path(launcher.__file__).read_text()
    assert 'parser.add_argument("--screen-runtime", required=True, type=Path)' in source
