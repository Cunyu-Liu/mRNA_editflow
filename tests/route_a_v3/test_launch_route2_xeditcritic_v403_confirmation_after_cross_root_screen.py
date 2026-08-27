from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import scripts.route_a_v3.launch_route2_xedit_v4_confirmation_posttraining_after_terminal as posttraining
import scripts.route_a_v3.launch_route2_xeditcritic_v403_confirmation_after_cross_root_screen as launcher


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _source_authorization(run_id: str, head: str) -> dict:
    return {
        "schema_version": (
            "route_a_v3_route2_xeditcritic_v4_"
            "screen_launch_authorization.v1"
        ),
        "status": "XEDITCRITIC_V4_SCREEN_LAUNCH_AUTHORIZED",
        "authorized_git_head": head,
        "authorized_run_ids": list(launcher.ARM_ORDER),
        "barriers": {key: True for key in launcher.SOURCE_BARRIERS},
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def _gate(tmp_path: Path) -> dict:
    arm_sources = {}
    for run_id in launcher.ARM_ORDER:
        head = launcher.expected_source_head(run_id)
        authorization_path = tmp_path / "authorizations" / f"{run_id}.json"
        _write_json(
            authorization_path, _source_authorization(run_id, head)
        )
        arm_sources[run_id] = {
            "summary_path": str(launcher.expected_summary_paths()[run_id]),
            "training_git_head": head,
            "source_role": launcher.expected_source_role(run_id),
            "launch_authorization_path": str(authorization_path),
            "launch_authorization_schema_version": (
                "route_a_v3_route2_xeditcritic_v4_"
                "screen_launch_authorization.v1"
            ),
            "launch_authorization_status": (
                "XEDITCRITIC_V4_SCREEN_LAUNCH_AUTHORIZED"
            ),
            "authorized_git_head": head,
            "run_id_authorization_verified": True,
            "authorization_protected_outcome_reads_verified_zero": True,
        }
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_screen_gate.v1",
        "status": "XEDITCRITIC_V4_SCREEN_PASS",
        "passed": True,
        "selectable_model": "V4-FULL",
        "screen_seed": 20260907,
        "checks": {key: True for key in launcher.SCREEN_CHECKS},
        "confirmation_authorized": True,
        "development_test_authorized": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
        "cross_root_transition": {
            "schema_version": (
                "route_a_v3_route2_xeditcritic_v403_cross_root_input.v1"
            ),
            "ordered_run_ids": list(launcher.ARM_ORDER),
            "arm_sources": arm_sources,
            "historical_c0_git_head": launcher.C0_GIT_HEAD,
            "repaired_full_and_controls_git_head": (
                launcher.TRAINING_GIT_HEAD
            ),
            "full_runtime_path": str(launcher.FULL_RUNTIME),
            "full_runtime_status": (
                "XEDITCRITIC_V403_FULL_RECOVERY_TERMINAL"
            ),
            "control_runtime_path": str(launcher.CONTROL_RUNTIME),
            "control_runtime_status": (
                "XEDITCRITIC_V403_CONTROL_RECOVERY_"
                "ALL_SIX_SUMMARIES_TERMINAL"
            ),
            "frozen_config_path": str(
                launcher.TRAINING_WORKTREE
                / "configs/route_a_v3_route2_xeditcritic_v4_screen_v1.json"
            ),
            "legacy_gate_path": str(launcher.LEGACY_GATE),
            "legacy_gate_preserved": True,
            "terminal_summary_payloads_read": 8,
            "full_retrained": False,
            "c0_retrained": False,
            "old_v402_stopped_process_resumed": False,
            "scientific_thresholds_changed": False,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    }


def _base_inputs() -> tuple[dict, dict]:
    return (
        json.loads(launcher.BASE_CONFIG.read_text(encoding="utf-8")),
        json.loads(launcher.BASE_PROTOCOL.read_text(encoding="utf-8")),
    )


def _diagnostics() -> dict[int, dict]:
    return {
        gpu: {
            "name": f"GPU-{gpu}",
            "free_memory_mib": 0,
            "total_memory_mib": 40960,
        }
        for gpu in launcher.PHYSICAL_GPUS
    }


def _probes() -> dict[int, dict]:
    return {
        gpu: {
            "physical_gpu_index": gpu,
            "device_name": f"GPU-{gpu}",
            "device_type": "cuda",
            "dtype": "BFLOAT16",
            "cuda_available": True,
            "bf16_supported": True,
            "cpu_fallback_used": False,
        }
        for gpu in launcher.PHYSICAL_GPUS
    }


def _focused_test_commands() -> list[str]:
    return [
        "python -m pytest -q "
        "test_score_route2_xeditflow_closed_frozen_methods_v3.py "
        "test_launch_route2_xeditflow_v403_guidance_after_dual_readiness.py "
        "test_launch_route2_xeditflow_v4_guidance_authorization_after_dual_readiness.py "
        "test_launch_route2_xeditflow_v4_guidance_screen_after_authorization.py "
        "test_launch_route2_xeditflow_v4_final_after_guidance_screen.py "
        "test_launch_route2_xeditcritic_v403_confirmation_after_cross_root_screen.py "
        "test_launch_route2_xeditsetflow_v403_recovered_confirmation_posttraining.py "
        "test_authorize_route2_xeditsetflow_v403_recovered_confirmation.py",
        "python -m pytest -q "
        "test_transition_adjudicate_route2_xeditcritic_v403_cross_root_screen.py "
        "test_prepare_route2_xeditcritic_v4_confirmation_configs.py "
        "test_route2_xeditcritic_v4_confirmation_runtime.py",
        "python -m pytest -q "
        "test_run_route2_xeditsetflow_v402_terminal_validation_scheduler.py "
        "test_adjudicate_route2_xeditsetflow_v4_confirmation.py "
        "test_route2_xeditsetflow_training_v4.py "
        "test_route2_xeditsetflow_s1_protocol.py "
        "test_route2_xeditsetflow_s1.py "
        "test_train_route2_xeditsetflow_s1.py "
        "test_validate_route2_xeditsetflow_s1_checkpoint.py "
        "test_route2_xeditsetflow_gate_s1.py "
        "test_run_route2_xeditsetflow_s1_screen_scheduler.py "
        "test_launch_route2_xeditsetflow_s1_screen_after_v403_terminal.py",
        "python -m pytest -q "
        "test_run_route2_xeditflow_v4_guidance_screen_scheduler.py "
        "test_adjudicate_route2_xeditflow_guidance_screen_v4.py "
        "test_route2_xeditflow_guidance_v4.py",
        "python -m pytest -q "
        "test_train_route2_xeditcritic_v4.py "
        "test_run_route2_xeditcritic_v4_loso_scheduler.py",
        "python -m pytest -q "
        "test_run_route2_xedit_v4_confirmation_training_scheduler.py "
        "test_run_route2_xedit_v4_confirmation_posttraining_scheduler.py",
        "python -m pytest -q "
        "test_launch_route2_xedit_v4_confirmation_training_after_screen_pass.py "
        "test_launch_route2_xedit_v4_confirmation_posttraining_after_terminal.py "
        "test_launch_route2_xeditsetflow_v403_recovered_confirmation.py "
        "test_launch_route2_xeditcritic_v403_controls_after_full.py "
        "test_launch_route2_xeditcritic_v4_atomic_frozen_test_after_confirmation.py "
        "test_launch_route2_xeditcritic_v4_refit_after_atomic_test.py "
        "test_launch_route2_xeditcritic_v4_loso_after_refits.py",
        "python -m pytest -q "
        "test_prepare_route2_xeditflow_final_generation_configs_v4.py "
        "test_evaluate_route2_xeditflow_closed_scores_v4.py "
        "test_compare_route2_xeditflow_independent_evaluator_v4.py "
        "test_xeditflow_v4_final_evidence_chain.py "
        "test_run_route2_xeditflow_strongest_timing_v4.py "
        "test_reproduce_route2_base_flow_v2_handover_validation.py "
        "test_export_route2_xeditflow_v4_terminal_training_ledger.py",
    ]


def _runner_receipt(runner_head: str) -> dict:
    return {
        "schema_version": launcher.RUNNER_VERIFICATION_RECEIPT_SCHEMA,
        "status": launcher.RUNNER_VERIFICATION_RECEIPT_PASS,
        "runner_git_head": runner_head,
        "worktree_clean": True,
        "focused_tests": {
            "command": _focused_test_commands(),
            "passed": True,
            "passed_count": 203,
            "failed_count": 0,
            "isolated_process_groups": True,
            "group_passed_counts": [75, 14, 4, 26, 14, 8, 36, 26],
        },
        "v332_tests": {
            "command": ["python", "-m", "pytest", "tests/*v332*.py"],
            "passed": True,
            "passed_count": 96,
            "failed_count": 0,
        },
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def test_fixed_cross_root_pass_and_all_eight_authorizations_are_accepted(
    tmp_path: Path,
) -> None:
    gate = _gate(tmp_path)
    launcher.validate_cross_root_gate(gate)
    authorizations = launcher.load_and_validate_source_authorizations(gate)
    assert tuple(authorizations) == launcher.ARM_ORDER
    assert authorizations["c0_v4"]["authorized_git_head"] == (
        launcher.C0_GIT_HEAD
    )
    assert all(
        authorizations[run_id]["authorized_git_head"]
        == launcher.TRAINING_GIT_HEAD
        for run_id in launcher.ARM_ORDER[1:]
    )


def test_sort_keys_persisted_pass_gate_round_trip_preserves_semantic_arm_order(
    tmp_path: Path,
) -> None:
    gate_path = tmp_path / "persisted_cross_root_gate.json"
    gate_path.write_text(
        json.dumps(_gate(tmp_path), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    persisted = json.loads(gate_path.read_text(encoding="utf-8"))
    transition = persisted["cross_root_transition"]
    assert tuple(transition["arm_sources"]) != launcher.ARM_ORDER
    assert set(transition["arm_sources"]) == set(launcher.ARM_ORDER)
    assert transition["ordered_run_ids"] == list(launcher.ARM_ORDER)

    launcher.validate_cross_root_gate(persisted)
    authorizations = launcher.load_and_validate_source_authorizations(
        persisted
    )
    assert tuple(authorizations) == launcher.ARM_ORDER


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda gate: gate.update(
                status="XEDITCRITIC_V4_SCREEN_NO_GO",
                passed=False,
                confirmation_authorized=False,
            ),
            "not terminal PASS",
        ),
        (
            lambda gate: gate.update(development_test_outcome_reads=1),
            "protected outcome read",
        ),
        (
            lambda gate: gate["cross_root_transition"].update(
                legacy_gate_preserved=False
            ),
            "transition provenance changed",
        ),
        (
            lambda gate: gate["cross_root_transition"]["arm_sources"][
                "c0_v4"
            ].update(authorized_git_head=launcher.TRAINING_GIT_HEAD),
            "provenance changed for c0_v4",
        ),
        (
            lambda gate: gate["cross_root_transition"]["arm_sources"][
                "v4_no_moe"
            ].update(authorized_git_head=launcher.C0_GIT_HEAD),
            "provenance changed for v4_no_moe",
        ),
    ],
)
def test_gate_fails_closed_on_nonpass_protected_or_wrong_provenance(
    tmp_path: Path, mutation, error: str
) -> None:
    gate = _gate(tmp_path)
    mutation(gate)
    with pytest.raises(Exception, match=error):
        launcher.validate_cross_root_gate(gate)


def test_source_authorization_artifact_must_match_embedded_provenance(
    tmp_path: Path,
) -> None:
    gate = _gate(tmp_path)
    path = Path(
        gate["cross_root_transition"]["arm_sources"]["v4_no_cross"][
            "launch_authorization_path"
        ]
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["authorized_git_head"] = launcher.C0_GIT_HEAD
    _write_json(path, payload)
    with pytest.raises(Exception, match="v4_no_cross"):
        launcher.load_and_validate_source_authorizations(gate)


def test_confirmation_configs_change_only_the_screen_gate_binding(
    tmp_path: Path,
) -> None:
    base_config, base_protocol = _base_inputs()
    gate = _gate(tmp_path)
    derived, configs = launcher.build_confirmation_configs(
        base_config, base_protocol, gate
    )
    assert base_protocol["screen_gate_path"] == str(launcher.LEGACY_GATE)
    assert derived["screen_gate_path"] == str(launcher.CROSS_ROOT_GATE)
    assert all(
        derived[key] == value
        for key, value in base_protocol.items()
        if key != "screen_gate_path"
    )
    assert [config["training_seed"] for config in configs] == list(
        launcher.CONFIRMATION_SEEDS
    )
    assert all(
        config["screen_gate_path"] == str(launcher.CROSS_ROOT_GATE)
        and config["required_confirmation_run_ids"]
        == ["v4_full", "c0_v4"]
        for config in configs
    )


def test_exact_runner_rejects_changed_repaired_training_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def unchanged(command_args, **kwargs):
        commands.append(list(command_args))
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(
        launcher,
        "command",
        unchanged,
    )
    receipt = launcher.validate_runner_training_semantics("c" * 40)
    assert receipt["training_semantics_unchanged"] is True
    assert receipt["repaired_screen_git_head"] == launcher.TRAINING_GIT_HEAD
    assert receipt["training_semantics_baseline_git_head"] == (
        launcher.TRAINING_SEMANTICS_BASELINE_HEAD
    )
    assert commands == [
        [
            "git",
            "diff",
            "--name-only",
            launcher.TRAINING_SEMANTICS_BASELINE_HEAD,
            "c" * 40,
            "--",
            *launcher.TRAINING_SEMANTIC_PATHS,
        ]
    ]
    monkeypatch.setattr(
        launcher,
        "command",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="scripts/route_a_v3/train_route2_xeditcritic_v4.py\n"
        ),
    )
    with pytest.raises(Exception, match="training semantics changed"):
        launcher.validate_runner_training_semantics("d" * 40)


def test_committed_successor_head_preserves_audited_training_semantics(
) -> None:
    current_head = launcher.command(
        ["git", "rev-parse", "HEAD"]
    ).stdout.strip()
    receipt = launcher.validate_runner_training_semantics(current_head)
    assert receipt["training_semantics_baseline_git_head"] == (
        launcher.TRAINING_SEMANTICS_BASELINE_HEAD
    )
    assert receipt["training_semantic_diff_paths"] == []
    assert receipt["training_semantics_unchanged"] is True
    assert receipt["training_semantics_baseline_audit"] == str(
        launcher.TRAINING_SEMANTICS_BASELINE_AUDIT
    )


def test_training_semantics_reaudit_rejects_changed_critic_classification() -> None:
    audit = launcher.read_json(launcher.TRAINING_SEMANTICS_BASELINE_AUDIT)
    launcher.validate_training_semantics_baseline_audit(audit)
    audit["critic_confirmation_training_semantics_changed"] = True
    with pytest.raises(Exception, match="re-audit"):
        launcher.validate_training_semantics_baseline_audit(audit)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda receipt: receipt.update(runner_git_head="d" * 40),
            "exact-HEAD clean PASS",
        ),
        (
            lambda receipt: receipt.update(worktree_clean=False),
            "exact-HEAD clean PASS",
        ),
        (
            lambda receipt: receipt["focused_tests"].update(
                passed=False, failed_count=1
            ),
            "focused tests",
        ),
        (
            lambda receipt: receipt["v332_tests"].update(
                passed=False, failed_count=1
            ),
            "V3.3.2 tests",
        ),
        (
            lambda receipt: receipt.update(development_test_outcome_reads=1),
            "protected outcome read",
        ),
    ],
)
def test_runner_receipt_fails_closed_on_wrong_head_dirty_or_failed_tests(
    mutation, error: str
) -> None:
    runner_head = "c" * 40
    receipt = _runner_receipt(runner_head)
    mutation(receipt)
    with pytest.raises(Exception, match=error):
        launcher.validate_runner_verification_receipt(
            receipt,
            runner_head=runner_head,
            receipt_path=launcher.runner_verification_receipt_path(
                runner_head
            ),
        )


def test_runner_receipt_accepts_complete_eight_group_coverage() -> None:
    runner_head = "c" * 40
    launcher.validate_runner_verification_receipt(
        _runner_receipt(runner_head),
        runner_head=runner_head,
        receipt_path=launcher.runner_verification_receipt_path(runner_head),
    )


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("six_groups", "focused tests"),
        (
            "c5db_group7_missing_new_markers",
            "lacks required test-module coverage",
        ),
        (
            "missing_setflow_authorization_marker",
            "lacks required test-module coverage",
        ),
        (
            "missing_handover_validation_marker",
            "lacks required test-module coverage",
        ),
        (
            "missing_terminal_training_ledger_marker",
            "lacks required test-module coverage",
        ),
        ("group_sum_mismatch", "focused tests"),
        ("focused_below_c5db_floor", "focused tests"),
        ("not_isolated", "focused tests"),
        ("nonpositive_group", "focused tests"),
        ("v332_not_96", "V3.3.2 tests"),
        ("v332_glob_absent", "V3.3.2 tests"),
    ],
)
def test_runner_receipt_coverage_fails_closed(
    case: str, error: str
) -> None:
    runner_head = "c" * 40
    receipt = _runner_receipt(runner_head)
    focused = receipt["focused_tests"]
    if case == "six_groups":
        focused["command"] = focused["command"][:6]
        focused["group_passed_counts"] = focused[
            "group_passed_counts"
        ][:6]
        focused["passed_count"] = sum(focused["group_passed_counts"])
    elif case == "c5db_group7_missing_new_markers":
        focused["command"][6] = (
            "python -m pytest -q "
            "test_launch_route2_xedit_v4_confirmation_training_after_screen_pass.py "
            "test_launch_route2_xedit_v4_confirmation_posttraining_after_terminal.py "
            "test_launch_route2_xeditsetflow_v403_recovered_confirmation.py"
        )
    elif case == "missing_setflow_authorization_marker":
        focused["command"][0] = focused["command"][0].replace(
            "test_authorize_route2_xeditsetflow_v403_recovered_confirmation.py",
            "",
        )
    elif case == "missing_handover_validation_marker":
        focused["command"][7] = focused["command"][7].replace(
            "test_reproduce_route2_base_flow_v2_handover_validation.py",
            "",
        )
    elif case == "missing_terminal_training_ledger_marker":
        focused["command"][7] = focused["command"][7].replace(
            "test_export_route2_xeditflow_v4_terminal_training_ledger.py",
            "",
        )
    elif case == "group_sum_mismatch":
        focused["group_passed_counts"][0] += 1
    elif case == "focused_below_c5db_floor":
        focused["group_passed_counts"][0] -= 1
        focused["passed_count"] = 202
    elif case == "not_isolated":
        focused["isolated_process_groups"] = False
    elif case == "nonpositive_group":
        focused["group_passed_counts"][1] += focused[
            "group_passed_counts"
        ][0]
        focused["group_passed_counts"][0] = 0
    elif case == "v332_not_96":
        receipt["v332_tests"]["passed_count"] = 95
    elif case == "v332_glob_absent":
        receipt["v332_tests"]["command"][-1] = "tests/v332.py"
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(case)

    with pytest.raises(Exception, match=error):
        launcher.validate_runner_verification_receipt(
            receipt,
            runner_head=runner_head,
            receipt_path=launcher.runner_verification_receipt_path(
                runner_head
            ),
        )


@pytest.mark.parametrize(
    "marker", launcher.FOCUSED_GROUP_REQUIRED_TEST_MARKERS[2][2:]
)
def test_runner_receipt_requires_each_s1_focused_marker(
    marker: str,
) -> None:
    runner_head = "c" * 40
    receipt = _runner_receipt(runner_head)
    receipt["focused_tests"]["command"][2] = receipt["focused_tests"][
        "command"
    ][2].replace(marker, "")

    with pytest.raises(
        Exception, match="lacks required test-module coverage"
    ):
        launcher.validate_runner_verification_receipt(
            receipt,
            runner_head=runner_head,
            receipt_path=launcher.runner_verification_receipt_path(
                runner_head
            ),
        )


def test_manifest_is_standard_consumer_shape_and_binds_exact_runner(
    tmp_path: Path,
) -> None:
    base_config, base_protocol = _base_inputs()
    protocol, configs = launcher.build_confirmation_configs(
        base_config, base_protocol, _gate(tmp_path)
    )
    protocol = copy.deepcopy(protocol)
    protocol["runtime_config_root"] = str(tmp_path / "configs")
    protocol["run_root"] = str(tmp_path / "runs")
    for config in configs:
        seed = int(config["training_seed"])
        config["output_root"] = str(tmp_path / "runs" / f"seed_{seed}")
    runner_head = "c" * 40
    manifest = launcher.materialize_config_package(
        configs, protocol, runner_head=runner_head
    )
    paths = launcher.validate_manifest(manifest, runner_head=runner_head)
    assert set(paths) == set(launcher.CONFIRMATION_SEEDS)
    assert manifest["schema_version"] == (
        "route_a_v3_route2_xeditcritic_v4_confirmation_config_manifest.v1"
    )
    assert manifest["required_run_ids"] == ["v4_full", "c0_v4"]
    assert manifest["screen_gate_path"] == str(launcher.CROSS_ROOT_GATE)


def test_authorization_is_trainer_and_posttraining_compatible(
    tmp_path: Path,
) -> None:
    gate = _gate(tmp_path)
    source = launcher.load_and_validate_source_authorizations(gate)
    runner_head = "d" * 40
    receipt = _runner_receipt(runner_head)
    authorization = launcher.build_confirmation_authorization(
        gate,
        source,
        receipt,
        runner_head=runner_head,
        runner_verification_receipt_path_value=(
            launcher.runner_verification_receipt_path(runner_head)
        ),
    )
    assert authorization["schema_version"] == (
        "route_a_v3_route2_xeditcritic_v4_"
        "confirmation_launch_authorization.v1"
    )
    assert authorization["status"] == (
        "XEDITCRITIC_V4_CONFIRMATION_LAUNCH_AUTHORIZED"
    )
    assert authorization["authorized_git_head"] == runner_head
    assert authorization["authorized_seeds"] == list(
        launcher.CONFIRMATION_SEEDS
    )
    assert authorization["authorized_run_ids"] == ["v4_full", "c0_v4"]
    assert authorization["runner_current_head_verification"][
        "runner_git_head"
    ] == runner_head
    assert all(
        authorization["barriers"].get(key) is True
        for key in (
            "screen_gate_passed",
            "a100_current_head_focused_tests_passed",
            "a100_current_head_v332_tests_passed",
            "bottom_six_cache_terminal_complete",
            "formal_parameter_preflight_passed",
            "formal_memory_preflight_passed",
        )
    )


def test_schedule_is_exact_three_seeds_two_arms_on_fixed_gpu_zero_to_five(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = {}
    for seed in launcher.CONFIRMATION_SEEDS:
        path = tmp_path / f"seed_{seed}.json"
        _write_json(path, {"output_root": str(tmp_path / f"run_{seed}")})
        configs[seed] = path
    runner_head = "e" * 40
    schedule = launcher.build_schedule(
        configs,
        tmp_path / "authorization.json",
        tmp_path / "manifest.json",
        _diagnostics(),
        _probes(),
        runner_head=runner_head,
        runtime_manifest=tmp_path / "runtime.json",
        log_root=tmp_path / "logs",
        preflight_peak_allocated_gib=14.9,
    )
    jobs = [queue["jobs"][0] for queue in schedule["gpu_queues"]]
    assert [queue["physical_gpu_index"] for queue in schedule["gpu_queues"]] == list(
        launcher.PHYSICAL_GPUS
    )
    assert [
        (job["training_seed"], job["run_id"]) for job in jobs
    ] == [
        (20260908, "v4_full"),
        (20260908, "c0_v4"),
        (20260909, "v4_full"),
        (20260909, "c0_v4"),
        (20260910, "v4_full"),
        (20260910, "c0_v4"),
    ]
    for queue, job in zip(schedule["gpu_queues"], jobs, strict=True):
        command = job["command"]
        gpu_flag = command.index("--physical-gpu-index")
        assert command.count("--physical-gpu-index") == 1
        assert command[gpu_flag + 1] == str(queue["physical_gpu_index"])
        assert command[gpu_flag + 2] == "--launch-authorization"
        assert len(command) == 10
    assert schedule["free_memory_gate_applied"] is False
    assert schedule["gpu_sort_applied"] is False
    assert all(
        row["free_memory_mib"] == 0
        for row in schedule["gpu_memory_diagnostics_before_launch"].values()
    )

    runtime = {
        "status": "V4_CONFIRMATION_TRAINING_ALL_JOBS_TERMINAL",
        "git_head": runner_head,
        "eligible_components": ["critic"],
        "jobs": {
            job["job_key"]: {"terminal_artifact_kind": "SUMMARY"}
            for job in jobs
        },
        "active_performance_output_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    assert posttraining.validate_confirmation_runtime(
        runtime, head=runner_head
    ) == ("critic",)

    import scripts.route_a_v3.train_route2_xeditcritic_v4 as trainer

    parsed: list[dict] = []

    def fake_run(config, **kwargs):
        parsed.append({"config": config, **kwargs})
        return {"status": "ARGPARSE_ACCEPTED"}

    monkeypatch.setattr(trainer, "run", fake_run)
    for job in jobs:
        monkeypatch.setattr(
            sys, "argv", [str(launcher.TRAINER), *job["command"][2:]]
        )
        trainer.main()
    assert len(parsed) == 6
    assert [row["physical_gpu_index"] for row in parsed] == list(
        launcher.PHYSICAL_GPUS
    )


def test_one_shot_receipt_rejects_duplicate_before_any_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "attempt.json"
    attempt.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "ATTEMPT_RECEIPT", attempt)
    protocol = {
        "runtime_config_root": str(tmp_path / "configs"),
        "run_root": str(tmp_path / "runs"),
    }
    with pytest.raises(Exception, match="canonical confirmation attempt"):
        launcher.ensure_one_shot_targets_absent(
            protocol=protocol,
            authorization_root=tmp_path / "authorization",
            runtime_root=tmp_path / "runtime",
            log_root=tmp_path / "logs",
        )
    assert not (tmp_path / "configs").exists()
    assert not (tmp_path / "runtime").exists()


def test_launcher_records_memory_but_never_gates_or_sorts_by_it() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert '"free_memory_gate_applied": False' in source
    assert '"gpu_sort_applied": False' in source
    assert "required_free_memory" not in source
    assert "free_memory_mib >=" not in source
    assert "sorted(diagnostics" not in source
    assert "torch.cuda.is_available()" in source
    assert "torch.cuda.is_bf16_supported()" in source
    assert "CUDA_BF16_PROBE_SILENT_CPU_FALLBACK" in source
    assert "STOPPED_BEFORE_CONFIRMATION_LAUNCH_CUDA_FAILURE" in source


def test_gate_and_provenance_validation_precede_one_shot_write_and_launch() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    run_source = source[source.index("def run(current_head: str)") :]
    gate_validation = run_source.index("validate_cross_root_gate(gate)")
    provenance_validation = run_source.index(
        "load_and_validate_source_authorizations(gate)"
    )
    attempt_write = run_source.index(
        "write_new_atomic(\n        ATTEMPT_RECEIPT"
    )
    scheduler_launch = run_source.index("process = subprocess.Popen(")
    assert gate_validation < provenance_validation < attempt_write < scheduler_launch
