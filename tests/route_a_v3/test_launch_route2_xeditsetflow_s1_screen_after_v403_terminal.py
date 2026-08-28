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


def _invalidation_receipt(*, scientific: bool) -> dict:
    if scientific:
        status = launcher.OLD_S1_SCIENTIFIC_INVALIDATION_STATUS
        terminal_class = "SCIENTIFIC_GATE_TERMINAL"
        runtime_status = launcher.OLD_S1_SCIENTIFIC_TERMINAL_STATUS
    else:
        status = launcher.OLD_S1_TECHNICAL_INVALIDATION_STATUS
        terminal_class = "TECHNICAL_FAILURE_TERMINAL"
        runtime_status = launcher.OLD_S1_TECHNICAL_TERMINAL_STATUS
    return {
        "schema_version": launcher.OLD_S1_TERMINAL_INVALIDATION_SCHEMA,
        "status": status,
        "terminal_class": terminal_class,
        "old_runner_git_head": launcher.INVALIDATED_S1_RUNNER_HEAD,
        "old_runtime_path": str(launcher.INVALIDATED_S1_RUNTIME),
        "old_runtime_status": runtime_status,
        "screen_seed": launcher.INVALIDATED_S1_SCREEN_SEED,
        "run_ids": list(launcher.INVALIDATED_S1_RUN_IDS),
        "objective_identity": launcher.OBJECTIVE_IDENTITY,
        "cross_state_candidate_mode_responsibility_weight": launcher.OBJECTIVE_WEIGHT,
        "scheduler_process_gone": True,
        "terminal_jobs": [{"job_key": f"job:{index}"} for index in range(10)],
        "terminal_adjudication": {"status": "TERMINAL"},
        "known_defect": {
            "identity": launcher.OLD_S1_DEFECT_IDENTITY,
            "model_construction_consumes_cpu_rng": True,
            "nominal_seed_controlled_parameter_initialization": False,
            "matched_full_single_initialization_established": False,
            "affected_family_can_authorize_successor": False,
        },
        "nominal_terminal_retained_as_execution_evidence": True,
        "nominal_terminal_rewritten": False,
        "scientific_successor_authorized": False,
        "successor_authorized": False,
        "same_family_retry_authorized": False,
        "old_family_artifacts_read_only": True,
        "old_runtime_read_count_this_transition": 1,
        "gpu_inventory_or_probe_executed": False,
        "gpu_or_model_execution_started": False,
        "protected_outcome_payload_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
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


@pytest.mark.parametrize("scientific", [True, False])
def test_s1_launcher_accepts_both_exact_old_terminal_invalidation_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scientific: bool
) -> None:
    receipt = tmp_path / "old_terminal_invalidation.json"
    receipt.write_text(json.dumps(_invalidation_receipt(scientific=scientific)))
    monkeypatch.setattr(launcher, "OLD_S1_TERMINAL_INVALIDATION_RECEIPT", receipt)
    consumed = launcher.consume_old_s1_terminal_invalidation_receipt(receipt)
    assert consumed["successor_authorized"] is False
    assert consumed["same_family_retry_authorized"] is False
    assert consumed["terminal_class"] == (
        "SCIENTIFIC_GATE_TERMINAL"
        if scientific
        else "TECHNICAL_FAILURE_TERMINAL"
    )


def test_s1_launcher_rejects_partial_or_authorizing_invalidation_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "old_terminal_invalidation.json"
    monkeypatch.setattr(launcher, "OLD_S1_TERMINAL_INVALIDATION_RECEIPT", receipt)
    partial = receipt.with_suffix(receipt.suffix + ".partial")
    partial.write_text("{}")
    with pytest.raises(launcher.XEditSetFlowS1LaunchError, match="partial"):
        launcher.consume_old_s1_terminal_invalidation_receipt(receipt)
    partial.unlink()
    payload = _invalidation_receipt(scientific=True)
    payload["successor_authorized"] = True
    receipt.write_text(json.dumps(payload))
    with pytest.raises(launcher.XEditSetFlowS1LaunchError, match="authorization"):
        launcher.consume_old_s1_terminal_invalidation_receipt(receipt)


def _command_identity(arguments: list[str]) -> SimpleNamespace:
    if arguments[:2] == ["git", "status"]:
        value = ""
    elif arguments[:3] == ["git", "branch", "--show-current"]:
        value = launcher.BRANCH
    elif arguments[:3] == ["git", "rev-parse", "HEAD"]:
        value = "a" * 40
    elif arguments[:2] == ["git", "rev-parse"]:
        value = "a" * 40
    else:
        raise AssertionError(arguments)
    return SimpleNamespace(stdout=value + ("\n" if value else ""), stderr="")


def test_s1_launcher_checks_invalidation_after_repo_facts_before_receipts_or_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = []
    config = _config(tmp_path)
    monkeypatch.setattr(launcher, "command", _command_identity)
    monkeypatch.setattr(launcher, "read_json", lambda path: config if path == launcher.CONFIG else {})
    monkeypatch.setattr(launcher, "validate_config", lambda _config: events.append("config"))
    monkeypatch.setattr(
        launcher,
        "validate_repo_fact_audits",
        lambda _config: events.append("repo_facts") or {},
    )

    def stop_at_invalidation(_path: Path) -> dict:
        events.append("old_terminal_invalidation")
        raise launcher.XEditSetFlowS1LaunchError("sentinel invalidation stop")

    monkeypatch.setattr(
        launcher,
        "consume_old_s1_terminal_invalidation_receipt",
        stop_at_invalidation,
    )
    monkeypatch.setattr(
        launcher,
        "consume_receipts",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("receipts consumed first")),
    )
    monkeypatch.setattr(
        launcher,
        "gpu_diagnostics",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("GPU inventory ran first")),
    )
    with pytest.raises(launcher.XEditSetFlowS1LaunchError, match="sentinel"):
        launcher.run("a" * 40)
    assert events == ["config", "repo_facts", "old_terminal_invalidation"]
    assert not Path(config["family_paths"]["runtime_root_template"].format(runner_git_head="a" * 40)).exists()


def test_s1_launcher_missing_invalidation_receipt_blocks_gpu_and_family_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    missing = tmp_path / "missing_terminal_invalidation.json"
    monkeypatch.setattr(launcher, "OLD_S1_TERMINAL_INVALIDATION_RECEIPT", missing)
    monkeypatch.setattr(launcher, "command", _command_identity)
    monkeypatch.setattr(launcher, "read_json", lambda path: config if path == launcher.CONFIG else {})
    monkeypatch.setattr(launcher, "validate_config", lambda _config: None)
    monkeypatch.setattr(launcher, "validate_repo_fact_audits", lambda _config: {})
    monkeypatch.setattr(
        launcher,
        "consume_receipts",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("receipts consumed")),
    )
    monkeypatch.setattr(
        launcher,
        "gpu_diagnostics",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("GPU inventory ran")),
    )
    with pytest.raises(launcher.XEditSetFlowS1LaunchError, match="absent"):
        launcher.run("a" * 40)
    family = Path(config["family_paths"]["runtime_root_template"].format(runner_git_head="a" * 40))
    assert not family.exists()


def test_s1_launcher_explicitly_rejects_invalidated_930_head_before_any_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        launcher,
        "command",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("repository inspected")),
    )
    monkeypatch.setattr(
        launcher,
        "gpu_diagnostics",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("GPU inventory ran")),
    )
    with pytest.raises(launcher.XEditSetFlowS1LaunchError, match="invalidated 930"):
        launcher.run(launcher.INVALIDATED_S1_RUNNER_HEAD)


def test_s1_launcher_source_order_places_terminal_barrier_before_receipts_gpu_and_family() -> None:
    source = Path(launcher.__file__).read_text()
    repo_facts = source.index("audits = validate_repo_fact_audits(config)")
    invalidation = source.index(
        "old_s1_terminal_invalidation = consume_old_s1_terminal_invalidation_receipt("
    )
    receipts = source.index("receipts = consume_receipts(")
    gpu = source.index("diagnostics = gpu_diagnostics(range(6))")
    family = source.index("family_root.mkdir(parents=True)")
    assert repo_facts < invalidation < receipts < gpu < family
