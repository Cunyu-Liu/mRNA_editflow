from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.route_a_v3.launch_route2_xeditsetflow_s1_screen_after_v403_terminal as launcher


def _config(tmp_path: Path) -> dict:
    root = tmp_path / "s1_{runner_git_head}"
    return {
        "schema_version": launcher.CONFIG_SCHEMA,
        "training": {"screen_seed": 20260911},
        "objective": {
            "identity": launcher.OBJECTIVE_IDENTITY,
            "cross_state_candidate_mode_responsibility_weight": .05,
            "cross_state_candidate_mode_responsibility_weight_sweep": False,
        },
        "required_screen_runs": [
            {"run_id": "v4_s1_full", "mode_count": 8, "mode_information_weight": .05, "cross_state_candidate_mode_responsibility_weight": .05, "selectable": True},
            {"run_id": "v4_s1_single_mode", "mode_count": 1, "mode_information_weight": 0.0, "cross_state_candidate_mode_responsibility_weight": .05, "selectable": False},
        ],
        "gpu_policy": {"physical_gpu_scope": [0, 1, 2, 3, 4, 5], "cuda_bf16_only": True, "cpu_fallback": False},
        "screen_gate": {"success_status": "XEDITSETFLOW_V4_S1_SCREEN_PASS", "failure_status": "XEDITSETFLOW_V4_S1_SCREEN_NO_GO"},
        "family_paths": {
            "runtime_root_template": str(root),
            "schedule_template": str(root / "schedule.json"),
            "runtime_template": str(root / "runtime.json"),
            "training_output_template": str(root / "training/{run_id}"),
            "validation_output_template": str(root / "validation/{run_id}"),
            "screen_gate_template": str(root / "screen_gate.json"),
            "log_root_template": str(root / "logs"),
            "authorization_template": str(tmp_path / "authorization_{runner_git_head}.json"),
            "prelaunch_failure_template": str(tmp_path / "failure_{runner_git_head}.json"),
        },
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def test_s1_launcher_validates_exact_frozen_config(tmp_path: Path) -> None:
    config = _config(tmp_path)
    launcher.validate_config(config)
    config["objective"]["cross_state_candidate_mode_responsibility_weight"] = .04
    with pytest.raises(launcher.XEditSetFlowS1LaunchError, match="objective"):
        launcher.validate_config(config)


def test_s1_launcher_inventory_records_memory_but_never_uses_it_as_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(
            returncode=0,
            stdout="0, NVIDIA A100-SXM4-80GB, 1, 81920\n1, NVIDIA A100-SXM4-80GB, 70000, 81920\n",
            stderr="",
        ),
    )
    diagnostics = launcher.gpu_diagnostics([0, 1])
    assert diagnostics[0]["free_memory_mib"] == 1
    assert diagnostics[1]["free_memory_mib"] == 70000
    source = Path(launcher.__file__).read_text()
    assert '"free_memory_gate_applied": False' in source
    assert "required_free_memory" not in source
    assert ">= free" not in source and "sorted(diagnostics" not in source


def test_s1_schedule_is_exact_two_plus_eight_and_uses_all_configured_gpus(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    diagnostics = {gpu: {"name": "A100"} for gpu in range(6)}
    probes = {gpu: {"device_class": "A100"} for gpu in range(6)}
    schedule = launcher.build_schedule(
        config,
        head="a" * 40,
        runtime_config_path=tmp_path / "runtime_config.json",
        authorization_path=tmp_path / "authorization.json",
        diagnostics=diagnostics,
        probes=probes,
    )
    training = [job for queue in schedule["training_queues"] for job in queue["jobs"]]
    validation = [job for queue in schedule["validation_queues"] for job in queue["jobs"]]
    assert len(training) == 2 and {job["run_id"] for job in training} == set(launcher.RUN_IDS)
    assert len(validation) == 8
    assert {job["physical_gpu_index"] for job in validation} == set(range(6))
    assert schedule["free_memory_gate_applied"] is False
    assert schedule["cross_state_candidate_mode_responsibility_weight"] == .05


def test_s1_gpu_failure_evidence_is_sibling_and_does_not_create_family_root(
    tmp_path: Path,
) -> None:
    family = tmp_path / "experiments" / "family"
    failure = tmp_path / "audits" / "family.failed.json"
    error = launcher.XEditSetFlowS1GpuError("inventory failed", reason="OUTPUT_PARSE_FAILED")
    launcher.write_prelaunch_failure(
        failure,
        head="a" * 40,
        family_root=family,
        error=error,
        diagnostics={},
        completed_probes={},
    )
    payload = json.loads(failure.read_text())
    assert payload["family_root_created"] is False
    assert payload["gpu_job_started"] is False
    assert payload["automatic_retry_attempted"] is False
    assert payload["free_memory_gate_applied"] is False
    assert not family.exists()


def test_s1_scheduler_launch_failure_is_durable_and_blocks_family_reuse(
    tmp_path: Path,
) -> None:
    family = tmp_path / "family"
    family.mkdir()
    failure = family / "scheduler_launch.failed.json"
    launcher.write_scheduler_launch_failure(
        failure,
        head="a" * 40,
        schedule_path=family / "schedule.json",
        runtime_path=family / "runtime.json",
        scheduler_command=["python", "scheduler.py", "--schedule", "schedule.json"],
        error=OSError("process creation failed"),
    )
    payload = json.loads(failure.read_text())
    assert payload["status"] == (
        "XEDITSETFLOW_V4_S1_SCHEDULER_LAUNCH_TECHNICAL_FAILURE"
    )
    assert payload["scheduler_started"] is False
    assert payload["gpu_job_started"] is False
    assert payload["automatic_retry_attempted"] is False
    assert payload["failure_stage"] == "SCHEDULER_PROCESS_LAUNCH"
    assert payload["scheduler_command"][1] == "scheduler.py"
    with pytest.raises(launcher.XEditSetFlowS1LaunchError, match="already exists"):
        launcher.write_scheduler_launch_failure(
            failure,
            head="a" * 40,
            schedule_path=family / "schedule.json",
            runtime_path=family / "runtime.json",
            scheduler_command=["python", "scheduler.py"],
            error=OSError("second attempt"),
        )


def test_s1_cuda_probe_failure_retains_failed_gpu_command_and_prior_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = subprocess.CalledProcessError(
        3,
        ["python", "-c", "probe", "1"],
        output="probe stdout",
        stderr="probe stderr",
    )
    monkeypatch.setattr(launcher, "command", lambda _arguments: (_ for _ in ()).throw(error))
    with pytest.raises(launcher.XEditSetFlowS1GpuError) as caught:
        launcher.cuda_bf16_probe(1)
    failure = tmp_path / "probe.failed.json"
    launcher.write_prelaunch_failure(
        failure,
        head="a" * 40,
        family_root=tmp_path / "family",
        error=caught.value,
        diagnostics={0: {"name": "A100", "free_memory_mib": 1}},
        completed_probes={0: {"device_class": "A100", "device_type": "cuda"}},
    )
    payload = json.loads(failure.read_text())
    assert payload["failure_stage"] == "A100_CUDA_BF16_PROBE"
    assert payload["failed_physical_gpu_index"] == 1
    assert payload["probe_command"][-1] == "1"
    assert payload["stdout"] == "probe stdout"
    assert payload["completed_cuda_bf16_probes"]["0"]["device_type"] == "cuda"


def test_s1_cuda_probe_nonobject_json_is_a_device_bound_parse_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        launcher,
        "command",
        lambda _arguments: SimpleNamespace(returncode=0, stdout="[]", stderr=""),
    )
    with pytest.raises(launcher.XEditSetFlowS1GpuError) as caught:
        launcher.cuda_bf16_probe(4)
    assert caught.value.reason == "CUDA_BF16_PROBE_OUTPUT_PARSE_FAILED"
    assert caught.value.failed_physical_gpu_index == 4
    assert caught.value.probe_command[-1] == "4"


def test_s1_launcher_consumes_tracked_facts_and_both_exact_head_receipts() -> None:
    source = Path(launcher.__file__).read_text()
    for token in (
        "validate_repo_fact_audits(config)",
        "validate_shared_runner_verification_receipt",
        "require_runner_verification_receipt_v403",
        'command(["git", "rev-parse", f"origin/{BRANCH}"])',
        "XEDITSETFLOW_V403_RECOVERED_SCREEN_TERMINAL_NO_GO_RECORDED",
        "XEDITCRITIC_V403_FULL_TERMINAL_SUMMARY_RECORDED",
        "XEDITSETFLOW_V4_S1_PROTOCOL_AND_RUNNER_FROZEN_NO_ATTEMPT",
        "XEDITSETFLOW_V4_S1_SEED_INITIALIZATION_REPAIR_FROZEN_BEFORE_INDEPENDENT_RETRY",
        "PARAMETER_INITIALIZATION_SEED_APPLIED_AFTER_MODEL_CONSTRUCTION",
    ):
        assert token in source
    assert "recovery_runtime" not in source
    assert "recovered_screen_gate" not in source


def test_s1_launcher_accepts_the_actual_tracked_repo_fact_audits() -> None:
    config = launcher.read_json(launcher.CONFIG)
    launcher.validate_config(config)
    audits = launcher.validate_repo_fact_audits(config)
    assert set(audits) == {
        "v403_terminal_no_go",
        "critic_v403_full_terminal",
        "s1_mechanism_authorization",
        "s1_seed_initialization_repair",
    }
    repair = audits["s1_seed_initialization_repair"]
    assert repair["defect"]["affected_family_can_authorize_successor"] is False
    assert repair["repair_contract"]["same_screen_seed"] == 20260911
    assert repair["repair_contract"]["threshold_reduction_authorized"] is False
