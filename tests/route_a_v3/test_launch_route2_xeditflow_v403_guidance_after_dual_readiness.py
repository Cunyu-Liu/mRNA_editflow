from __future__ import annotations

import copy
import inspect
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.route_a_v3.launch_route2_xeditcritic_v403_confirmation_after_cross_root_screen as critic_launcher
import scripts.route_a_v3.launch_route2_xeditflow_v403_guidance_after_dual_readiness as launcher


BRIDGE_HEAD = "a" * 40
CRITIC_RUNNER_HEAD = "b" * 40
SETFLOW_POSTTRAINING_HEAD = "c" * 40
SETFLOW_TRAINING_RUNNER_HEAD = "d" * 40
SETFLOW_TRAINING_HEAD = "e" * 40
SETFLOW_VALIDATION_HEAD = "f" * 40
CRITIC_PREFLIGHT_HEAD = "1" * 40
SETFLOW_PREFLIGHT_HEAD = "2" * 40
CRITIC_CACHE_HEAD = "3" * 40
SETFLOW_CACHE_HEAD = "4" * 40


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


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
        "test_launch_route2_xeditsetflow_s1_screen_after_v403_terminal.py "
        "test_route2_xeditsetflow_confirmation_s1.py "
        "test_launch_route2_xeditsetflow_s1_confirmation_after_screen_pass.py "
        "test_launch_route2_xeditsetflow_s1_confirmation_posttraining.py "
        "test_adjudicate_route2_xeditsetflow_s1_confirmation.py",
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


def _runner_receipt(head: str = BRIDGE_HEAD) -> dict:
    return {
        "schema_version": (
            "route_a_v3_route2_xedit_v403_successor_runner_"
            "verification_receipt.v1"
        ),
        "status": "XEDIT_V403_SUCCESSOR_RUNNER_VERIFICATION_PASS",
        "runner_git_head": head,
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


def test_runner_verification_receipt_is_exact_head_terminal_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    route2 = tmp_path / "route2"
    monkeypatch.setattr(launcher, "ROOT", route2)
    monkeypatch.setattr(critic_launcher, "ROOT", route2)
    path = launcher.expected_runner_verification_receipt_path_v403(BRIDGE_HEAD)

    with pytest.raises(Exception, match="is absent"):
        launcher.consume_runner_verification_receipt_v403(path, BRIDGE_HEAD)

    stale = _runner_receipt("9" * 40)
    _write(path, stale)
    with pytest.raises(Exception, match="exact-HEAD clean PASS"):
        launcher.consume_runner_verification_receipt_v403(path, BRIDGE_HEAD)

    failed = _runner_receipt()
    failed["focused_tests"]["passed"] = False
    _write(path, failed)
    with pytest.raises(Exception, match="failed or incomplete focused"):
        launcher.consume_runner_verification_receipt_v403(path, BRIDGE_HEAD)

    wrong_v332_count = _runner_receipt()
    wrong_v332_count["v332_tests"]["passed_count"] = 95
    _write(path, wrong_v332_count)
    with pytest.raises(Exception, match="failed or incomplete V3.3.2 tests"):
        launcher.consume_runner_verification_receipt_v403(path, BRIDGE_HEAD)

    _write(path, _runner_receipt())
    lineage = launcher.consume_runner_verification_receipt_v403(
        path, BRIDGE_HEAD
    )
    assert lineage["runner_git_head"] == BRIDGE_HEAD
    assert lineage["focused_tests"]["passed_count"] == 203
    assert lineage["v332_tests"]["passed_count"] == 96


def _setflow_package(tmp_path: Path) -> tuple[dict, ...]:
    recovered_protocol_path = tmp_path / "recovered_protocol.json"
    runtime_path = tmp_path / "posttraining" / "runtime.json"
    gate_path = tmp_path / "confirmation_gate.json"
    manifest_path = tmp_path / "configs" / "manifest.json"
    schedule_path = tmp_path / "posttraining" / "schedule.json"
    recovered_screen_gate = str(tmp_path / "recovered_screen_gate.json")
    recovered_protocol = {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v4_confirmation_protocol.v1"
        ),
        "status": "FROZEN_PROSPECTIVE_BEFORE_SCREEN_OR_CONFIRMATION_RESULT",
        "required_seeds": list(launcher.SEEDS),
        "additional_seed_authorized": False,
        "confirmation_gate_output": str(gate_path),
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
        "validation_recovery_provenance": {
            "training_git_head": SETFLOW_TRAINING_HEAD,
            "validation_git_head": SETFLOW_VALIDATION_HEAD,
            "recovered_screen_gate_path": recovered_screen_gate,
            "parameter_update_count": 0,
            "scientific_thresholds_changed": False,
        },
    }
    config_paths = []
    for seed in launcher.SEEDS:
        path = tmp_path / "configs" / f"seed_{seed}.json"
        _write(
            path,
            {
                "schema_version": (
                    "route_a_v3_route2_xeditsetflow_v4_confirmation_runtime.v1"
                ),
                "training_seed": seed,
                "screen_gate_path": recovered_screen_gate,
                "development_test_outcomes_accessed": False,
                "new_final_evaluation_outcomes_accessed": False,
            },
        )
        config_paths.append(str(path))
    manifest = {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v4_confirmation_config_manifest.v1"
        ),
        "status": "THREE_CONFIRMATION_CONFIGS_PREPARED_NOT_STARTED",
        "selected_model": "v4_full",
        "required_seeds": list(launcher.SEEDS),
        "config_paths": config_paths,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    bindings = {
        "protocol_path": str(recovered_protocol_path),
        "runner_git_head": SETFLOW_TRAINING_RUNNER_HEAD,
        "config_manifest_path": str(manifest_path),
        "confirmation_gate_output": str(gate_path),
        "posttraining_runtime_root": str(runtime_path.parent),
    }
    schedule = {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v403_recovered_"
            "confirmation_posttraining_schedule.v1"
        ),
        "status": "FROZEN_RECOVERED_CONFIRMATION_POSTTRAINING_SCHEDULE",
        "git_head": SETFLOW_POSTTRAINING_HEAD,
        "orchestration_git_head": SETFLOW_POSTTRAINING_HEAD,
        "training_runner_git_head": SETFLOW_TRAINING_RUNNER_HEAD,
        "training_git_head": SETFLOW_TRAINING_HEAD,
        "recovery_validation_git_head": SETFLOW_VALIDATION_HEAD,
        "eligible_components": ["setflow"],
        "runtime_manifest": str(runtime_path),
        "posttraining_bindings": bindings,
        "free_memory_gate_applied": False,
        "cpu_fallback_used": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    runtime = {
        "schema_version": (
            "route_a_v3_route2_xedit_v4_confirmation_posttraining_runtime.v1"
        ),
        "status": "V4_CONFIRMATION_POSTTRAINING_ALL_TERMINAL",
        "git_head": SETFLOW_POSTTRAINING_HEAD,
        "eligible_components": ["setflow"],
        "validation_jobs": {
            f"job-{index}": {
                "status": "TERMINAL_COMPLETE",
                "terminal_artifact_kind": "SUMMARY",
            }
            for index in range(12)
        },
        "adjudications": {
            "setflow": {
                "status": "TERMINAL_COMPLETE",
                "terminal_artifact_kind": "SUMMARY",
                "gate_path": str(gate_path),
            }
        },
        "active_performance_output_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    gate = {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v4_confirmation_gate.v1"
        ),
        "status": "XEDITSETFLOW_V4_G0_READY",
        "required_seeds": list(launcher.SEEDS),
        "seed_results": {
            str(seed): {"passed": True} for seed in launcher.SEEDS
        },
        "additional_seed_authorized": False,
        "development_test_authorized": False,
        "guidance_authorized": False,
        "critic_used": False,
        "independent_evaluator_used": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    return (
        schedule,
        runtime,
        gate,
        recovered_protocol,
        manifest,
        schedule_path,
        runtime_path,
        gate_path,
        recovered_protocol_path,
        manifest_path,
    )


def test_setflow_lineage_consumes_recovered_runtime_gate_and_three_configs(
    tmp_path: Path,
) -> None:
    package = _setflow_package(tmp_path)
    lineage = launcher.validate_setflow_lineage_v403(
        *package[:5],
        schedule_path=package[5],
        runtime_path=package[6],
        gate_path=package[7],
        recovered_protocol_path=package[8],
        manifest_path=package[9],
    )
    assert lineage["posttraining_runner_git_head"] == SETFLOW_POSTTRAINING_HEAD
    assert lineage["training_runner_git_head"] == SETFLOW_TRAINING_RUNNER_HEAD
    assert lineage["training_git_head"] == SETFLOW_TRAINING_HEAD
    assert lineage["recovery_validation_git_head"] == SETFLOW_VALIDATION_HEAD
    assert set(lineage["runtime_config_paths"]) == {
        str(seed) for seed in launcher.SEEDS
    }

    wrong_runtime = copy.deepcopy(package[1])
    wrong_runtime["git_head"] = SETFLOW_TRAINING_RUNNER_HEAD
    with pytest.raises(Exception, match="not exact G0 terminal"):
        launcher.validate_setflow_lineage_v403(
            package[0],
            wrong_runtime,
            *package[2:5],
            schedule_path=package[5],
            runtime_path=package[6],
            gate_path=package[7],
            recovered_protocol_path=package[8],
            manifest_path=package[9],
        )


def _critic_ready() -> tuple[dict, dict, dict]:
    readiness_path = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/ready.json"
    runtime = {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_loso_runtime.v1",
        "status": "CRITIC_V4_READY_FOR_GUIDANCE",
        "git_head": CRITIC_RUNNER_HEAD,
        "readiness": {
            "terminal_artifact_kind": "SUMMARY",
            "readiness_status": "CRITIC_V4_READY_FOR_GUIDANCE",
            "guidance_authorized": True,
            "summary_path": readiness_path,
        },
        "active_performance_output_read": False,
        "development_test_access_event_count_before_loso": 1,
        "development_test_outcome_reads_during_loso": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    readiness = {
        "schema_version": (
            "route_a_v3_route2_xeditcritic_v4_guidance_readiness.v1"
        ),
        "status": "CRITIC_V4_READY_FOR_GUIDANCE",
        "three_seed_passed": True,
        "frozen_test_passed": True,
        "all_development_refit_complete": True,
        "loso_readiness_passed": True,
        "guidance_authorized": True,
        "development_test_access_event_count": 1,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    refit = {
        "status": "XEDITCRITIC_V4_ALL_DEVELOPMENT_REFIT_COMPLETE",
        "required_seeds": [20260908, 20260909, 20260910],
        "completed_refit_count": 3,
        "refit_pass_count": 8,
        "loso_authorized": True,
        "checkpoints": [
            {"seed": seed, "checkpoint_path": f"/mnt/checkpoint_{seed}.pt"}
            for seed in (20260908, 20260909, 20260910)
        ],
        "development_test_outcomes_accessed_during_refit": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    return runtime, readiness, refit


def test_critic_lineage_keeps_canonical_inputs_and_own_runner_head() -> None:
    runtime, readiness, refit = _critic_ready()
    readiness_path = Path(runtime["readiness"]["summary_path"])
    lineage = launcher.validate_critic_lineage_v403(
        runtime,
        readiness,
        refit,
        runtime_path=Path("/mnt/runtime.json"),
        readiness_path=readiness_path,
        refit_manifest_path=Path("/mnt/refit.json"),
    )
    assert lineage["training_runner_git_head"] == CRITIC_RUNNER_HEAD
    assert lineage["readiness_path"] == str(readiness_path)


def test_preflight_cache_and_training_heads_are_not_collapsed() -> None:
    critic_authorization_path = Path("/mnt/critic_authorization.json")
    setflow_authorization_path = Path("/mnt/setflow_authorization.json")
    source_cache = Path("/mnt/source_cache.pt")
    critic_preflight = {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_preflight.v1",
        "status": "XEDITCRITIC_V4_PREFLIGHT_PASS",
        "passed": True,
        "git_head": CRITIC_PREFLIGHT_HEAD,
        "authorization_path": str(critic_authorization_path),
        "cpu_fallback_used": False,
        "cuda_device_name": "A100",
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    identity = {"model_id": "mRNA-BERT", "record_count": 3}
    setflow_preflight = {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_preflight.v1",
        "status": "XEDITSETFLOW_V4_PREFLIGHT_PASS",
        "passed": True,
        "git_head": SETFLOW_PREFLIGHT_HEAD,
        "cpu_fallback_used": False,
        "torch_device": "cuda:4",
        "precision": "BF16",
        "source_token_cache_identity": identity,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    critic_authorization = {
        "schema_version": (
            "route_a_v3_route2_xeditcritic_v4_preflight_authorization.v1"
        ),
        "status": "XEDITCRITIC_V4_PREFLIGHT_AUTHORIZED",
        "authorized_git_head": CRITIC_PREFLIGHT_HEAD,
        "cache_experiment_head": CRITIC_CACHE_HEAD,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    setflow_authorization = {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v4_preflight_authorization.v1"
        ),
        "status": "XEDITSETFLOW_V4_PREFLIGHT_AUTHORIZED",
        "authorized_git_head": SETFLOW_PREFLIGHT_HEAD,
        "cache_experiment_head": SETFLOW_CACHE_HEAD,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    critic_cache = {
        "schema_version": (
            "route_a_v3_route2_frozen_bottom_encoder_chunk_cache_summary.v4"
        ),
        "status": "XEDITCRITIC_V4_BOTTOM_SIX_CACHE_COMPLETE",
        "git_head": CRITIC_CACHE_HEAD,
        "cpu_fallback": False,
        "development_test_outcomes_accessed": False,
        "evaluation_outcomes_accessed": False,
    }
    setflow_cache = {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v4_source_cache_adoption_receipt.v1"
        ),
        "status": "XEDITSETFLOW_V4_SOURCE_CACHE_ADOPTED_READ_ONLY",
        "git_head": SETFLOW_CACHE_HEAD,
        "legacy_cache_path": str(source_cache),
        "source_token_cache_identity": identity,
        "cpu_fallback_used": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    lineage = launcher.validate_preflight_cache_lineage_v403(
        critic_preflight,
        setflow_preflight,
        critic_authorization,
        setflow_authorization,
        critic_cache,
        setflow_cache,
        critic_preflight_path=Path("/mnt/critic_preflight.json"),
        setflow_preflight_path=Path("/mnt/setflow_preflight.json"),
        critic_authorization_path=critic_authorization_path,
        setflow_authorization_path=setflow_authorization_path,
        critic_cache_summary_path=Path("/mnt/critic_cache.json"),
        setflow_cache_receipt_path=Path("/mnt/setflow_cache.json"),
        source_token_cache_path=source_cache,
    )
    assert lineage["critic_preflight_runner_git_head"] == CRITIC_PREFLIGHT_HEAD
    assert lineage["setflow_preflight_runner_git_head"] == SETFLOW_PREFLIGHT_HEAD
    assert lineage["critic_cache_experiment_git_head"] == CRITIC_CACHE_HEAD
    assert lineage["setflow_cache_experiment_git_head"] == SETFLOW_CACHE_HEAD
    assert len(set(lineage.values()) & {
        CRITIC_PREFLIGHT_HEAD,
        SETFLOW_PREFLIGHT_HEAD,
        CRITIC_CACHE_HEAD,
        SETFLOW_CACHE_HEAD,
    }) == 4


def test_real_prefrozen_strongest_score_table_format_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    route2 = tmp_path / "route2"
    monkeypatch.setattr(launcher, "ROOT", route2)
    measured_path = route2 / "cohort/measured_neighborhood.private.jsonl"
    score_path = route2 / "baseline/frozen_method_scores.private.jsonl"
    summary_path = route2 / "baseline/run_summary.json"
    authorization_path = route2 / "authorization/guidance.json"
    measured_rows = [
        {"source_key": "source-1", "candidate_sequence": "ACGU"},
        {"source_key": "source-2", "candidate_sequence": "UGCA"},
    ]
    score_rows = [
        {
            "source_key": row["source_key"],
            "candidate_sequence": row["candidate_sequence"],
            "frozen_method_score": float(index) / 10.0,
            "method_id": "strongest_matched_baseline",
            "base_flow_training_seed": launcher.STRONGEST_SCORE_SEED,
            "score_used_measured_outcome": False,
        }
        for index, row in enumerate(measured_rows)
    ]
    measured_path.parent.mkdir(parents=True)
    measured_path.write_text(
        "".join(json.dumps(row) + "\n" for row in measured_rows),
        encoding="utf-8",
    )
    score_path.parent.mkdir(parents=True)
    score_path.write_text(
        "".join(json.dumps(row) + "\n" for row in score_rows),
        encoding="utf-8",
    )
    _write(authorization_path, {})
    summary = {
        "schema_version": (
            "route_a_v3_route2_xeditflow_closed_frozen_scores.v3"
        ),
        "status": "XEDITFLOW_V3_CLOSED_FROZEN_SCORES_COMPLETE",
        "method_id": "strongest_matched_baseline",
        "base_flow_training_seed": launcher.STRONGEST_SCORE_SEED,
        "source_count": 891,
        "measured_candidate_count": len(score_rows),
        "score_path": str(score_path),
        "score_provider": "FROZEN_GENETIC_GUIDING_CHECKPOINT",
        "strongest_guiding_forward_calls": len(score_rows),
        "frozen_baseline_reselected": False,
        "measured_outcome_used_for_score": False,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
        "cpu_fallback_used": False,
        "cuda_device_index": 3,
        "cuda_device_name": "NVIDIA A100-SXM4-80GB",
        "cuda_parent_uuid_matches_declared_physical_index": True,
        "guidance_authorization_path": str(authorization_path),
        "guidance_authorization_schema_version": (
            "route_a_v3_route2_xeditflow_v4_guidance_authorization.v1"
        ),
        "guidance_authorization_status": "XEDITFLOW_V4_GUIDANCE_AUTHORIZED",
    }
    _write(summary_path, summary)
    lineage = launcher.validate_strongest_closed_score_artifact_v403(
        summary,
        summary_path=summary_path,
        score_table_path=score_path,
        measured_neighborhood_path=measured_path,
        authorization_path=authorization_path,
        physical_gpu_index=3,
    )
    assert lineage["method_id"] == "strongest_matched_baseline"
    assert lineage["measured_candidate_count"] == 2
    assert lineage["guidance_winner_used"] is False

    score_path.write_text(
        json.dumps(
            {
                "source_key": "source-1",
                "candidate_sequence": "ACGU",
                "candidate_probability": 0.5,
                "method_id": "full_soft_value_smc",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="exactly cover|row schema"):
        launcher.validate_strongest_closed_score_artifact_v403(
            {
                **summary,
                "measured_candidate_count": 1,
                "strongest_guiding_forward_calls": 1,
            },
            summary_path=summary_path,
            score_table_path=score_path,
            measured_neighborhood_path=measured_path,
            authorization_path=authorization_path,
            physical_gpu_index=3,
        )


def test_strongest_scorer_failure_is_recorded_once_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    route2 = tmp_path / "route2"
    monkeypatch.setattr(launcher, "ROOT", route2)
    output_dir = route2 / "experiments/pre_guidance_strongest"
    failure_path = output_dir.with_name(output_dir.name + ".bridge_failure.json")
    protocol = {
        "authorization_output": str(route2 / "authorization.json"),
        "source_eligibility_manifest": str(route2 / "sources.jsonl"),
        "measured_neighborhood_path": str(route2 / "measured.jsonl"),
        "strongest_generation_baseline_path": str(route2 / "strongest.json"),
        "baseline_selection_input_path": str(route2 / "selection.json"),
        "strongest_closed_score_config_path": str(
            route2 / "runtime/strongest_score.json"
        ),
        "strongest_closed_score_failure_path": str(failure_path),
        "strongest_closed_score_summary_path": str(
            output_dir / "run_summary.json"
        ),
        "strongest_closed_score_table_path": str(
            output_dir / "frozen_method_scores.private.jsonl"
        ),
    }
    calls = []

    def fail_once(arguments):
        calls.append(list(arguments))
        scorer_failure_path = output_dir.with_name(output_dir.name + ".failed.json")
        _write(
            scorer_failure_path,
            {
                "status": "STOPPED_WITH_EVIDENCE",
                "error_type": "RuntimeError",
                "error": "CUDA is unavailable; CPU fallback is forbidden",
                "cuda_available": False,
                "requested_cuda_observation": {
                    "cuda_device_index": 4,
                    "cuda_device_name": None,
                },
                "cpu_fallback_used": False,
            },
        )
        raise subprocess.CalledProcessError(
            2,
            arguments,
            output="producer stdout",
            stderr="CUDA is unavailable; CPU fallback is forbidden",
        )

    monkeypatch.setattr(launcher, "command", fail_once)
    with pytest.raises(Exception, match="stopped with evidence"):
        launcher.materialize_strongest_closed_score_v403(
            protocol, physical_gpu_index=4
        )
    assert len(calls) == 1
    evidence = json.loads(failure_path.read_text(encoding="utf-8"))
    assert evidence["returncode"] == 2
    assert evidence["command"] == calls[0]
    assert evidence["stdout"] == "producer stdout"
    assert "CUDA is unavailable" in evidence["stderr"]
    assert evidence["physical_gpu_index"] == 4
    assert evidence["cuda_available"] is False
    assert evidence["scorer_error_type"] == "RuntimeError"
    assert "CPU fallback is forbidden" in evidence["scorer_error"]
    assert evidence["scorer_cpu_fallback_used"] is False
    assert evidence["cpu_fallback_used"] is False
    assert evidence["automatic_retry_attempted"] is False
    assert evidence["development_test_outcome_reads"] == 0
    assert evidence["new_final_evaluation_outcome_reads"] == 0


def test_derived_protocol_changes_only_allowed_paths_and_provenance() -> None:
    base = json.loads(launcher.BASE_PROTOCOL.read_text(encoding="utf-8"))
    setflow = {
        "confirmation_gate_path": (
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"
            "experiments/xeditsetflow_v4/recovered/confirmation_gate.json"
        ),
        "runtime_config_paths": {
            str(seed): (
                "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"
                f"runtime_configs/xeditsetflow_v4/recovered/seed_{seed}.json"
            )
            for seed in launcher.SEEDS
        },
        "posttraining_schedule_path": "/mnt/setflow_schedule.json",
        "posttraining_runtime_path": "/mnt/setflow_runtime.json",
        "posttraining_runner_git_head": SETFLOW_POSTTRAINING_HEAD,
        "training_runner_git_head": SETFLOW_TRAINING_RUNNER_HEAD,
        "training_git_head": SETFLOW_TRAINING_HEAD,
        "recovery_validation_git_head": SETFLOW_VALIDATION_HEAD,
        "recovered_protocol_path": "/home/recovered_protocol.json",
        "config_manifest_path": "/mnt/manifest.json",
    }
    critic = {
        "loso_runtime_path": "/mnt/critic_runtime.json",
        "training_runner_git_head": CRITIC_RUNNER_HEAD,
        "readiness_path": base["critic_readiness_path"],
        "refit_manifest_path": base["critic_refit_manifest_path"],
    }
    prerequisites = {
        "critic_preflight_runner_git_head": CRITIC_PREFLIGHT_HEAD,
        "setflow_preflight_runner_git_head": SETFLOW_PREFLIGHT_HEAD,
        "critic_cache_experiment_git_head": CRITIC_CACHE_HEAD,
        "setflow_cache_experiment_git_head": SETFLOW_CACHE_HEAD,
        "successor_runner_verification": {
            "receipt_path": str(
                launcher.expected_runner_verification_receipt_path_v403(
                    BRIDGE_HEAD
                )
            ),
            "runner_git_head": BRIDGE_HEAD,
            "worktree_clean": True,
        },
    }
    derived = launcher.build_derived_guidance_protocol_v403(
        base,
        runner_head=BRIDGE_HEAD,
        setflow_lineage=setflow,
        critic_lineage=critic,
        prerequisite_lineage=prerequisites,
    )
    changed = {key for key in base if base[key] != derived[key]}
    assert changed == launcher.DERIVED_EXISTING_PATH_FIELDS
    assert set(derived) == set(base) | launcher.DERIVED_NEW_FIELDS
    assert all(
        derived[key] == base[key]
        for key in base
        if key not in launcher.DERIVED_EXISTING_PATH_FIELDS
    )
    assert derived["guidance_grid"] == base["guidance_grid"]
    assert derived["value_to_go"] == base["value_to_go"]
    assert derived["smc"] == base["smc"]
    assert derived["protected_outcomes"] == base["protected_outcomes"]
    assert "guidance_v403_recovered_" in derived["authorization_output"]
    score_output = Path(derived["strongest_closed_score_summary_path"]).parent
    screen_output = Path(derived["guidance_screen_output_root"])
    score_config = Path(derived["strongest_closed_score_config_path"])
    screen_config_root = Path(derived["runtime_config_root"])
    assert not score_output.is_relative_to(screen_output)
    assert not screen_output.is_relative_to(score_output)
    assert not score_config.is_relative_to(screen_config_root)
    assert derived["strongest_closed_score_table_path"].endswith(
        "/frozen_method_scores.private.jsonl"
    )
    assert (
        derived["v403_recovery_provenance"]["prerequisites"]
        ["successor_runner_verification"]["runner_git_head"]
        == BRIDGE_HEAD
    )

    changed_science = copy.deepcopy(derived)
    changed_science["guidance_grid"]["kappa"] = [0.0]
    with pytest.raises(Exception, match="non-path science"):
        launcher.validate_derived_guidance_protocol_v403(
            base, changed_science
        )


def test_final_successor_carries_every_explicit_path_and_independent_head() -> None:
    route2 = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
    protocol = route2 / "runtime_configs/guidance_v403/protocol.json"
    guidance_runtime = route2 / "experiments/guidance_v403/runtime.json"
    base = json.loads(launcher.BASE_PROTOCOL.read_text(encoding="utf-8"))
    derived = {
        "guidance_screen_output_root": str(
            route2 / "experiments/guidance_v403/screen_seed_20260912"
        ),
        "strongest_closed_score_table_path": str(
            route2
            / "experiments/guidance_v403/strongest/frozen_method_scores.private.jsonl"
        ),
        "strongest_closed_score_summary_path": str(
            route2 / "experiments/guidance_v403/strongest/run_summary.json"
        ),
        "final_runtime_root": str(route2 / "experiments/guidance_v403/final"),
        "final_log_root": str(route2 / "logs/guidance_v403/final"),
    }
    guidance_gate = route2 / "experiments/guidance_v403/guidance_screen_gate.json"
    prerequisites = {
        "critic_preflight_path": str(route2 / "preflight/critic.json"),
        "critic_preflight_runner_git_head": CRITIC_PREFLIGHT_HEAD,
        "setflow_preflight_path": str(route2 / "preflight/setflow.json"),
        "setflow_preflight_runner_git_head": SETFLOW_PREFLIGHT_HEAD,
    }
    successor = launcher.build_final_successor_v403(
        current_head=BRIDGE_HEAD,
        protocol_path=protocol,
        guidance_runtime_path=guidance_runtime,
        guidance_screen_gate_path=guidance_gate,
        derived=derived,
        prerequisite_lineage=prerequisites,
        strongest_score_lineage={
            "status": "MATERIALIZED_BEFORE_V4_GUIDANCE_SCREEN",
            "method_id": "strongest_matched_baseline",
            "score_table_path": derived["strongest_closed_score_table_path"],
            "summary_path": derived["strongest_closed_score_summary_path"],
            "guidance_winner_used": False,
            "baseline_reselected_for_v4": False,
        },
    )
    assert successor["current_head"] == BRIDGE_HEAD
    assert successor["protocol_path"] == str(protocol)
    assert successor["guidance_runtime_path"] == str(guidance_runtime)
    assert successor["strongest_closed_score_table_path"] == (
        derived["strongest_closed_score_table_path"]
    )
    resolution = successor["guidance_selection_resolution"]
    assert resolution["status"] == (
        "UNRESOLVED_UNTIL_FROZEN_GUIDANCE_SCREEN_GATE"
    )
    assert resolution["guidance_screen_gate_path"] == str(guidance_gate)
    assert resolution["winner_resolves_only"] == [
        "kappa",
        "temperature",
        "beta_max",
    ]
    assert resolution["strongest_closed_score_table_independent_of_winner"] is True
    assert resolution["single_guidance_winner_bound_before_frozen_gate"] is False
    assert successor["critic_preflight_runner_git_head"] == CRITIC_PREFLIGHT_HEAD
    assert successor["setflow_preflight_runner_git_head"] == SETFLOW_PREFLIGHT_HEAD
    assert successor["execution_runtime_root"] == derived["final_runtime_root"]
    assert successor["execution_log_root"] == derived["final_log_root"]


def test_no_winner_preselection_is_required_for_mock_screen_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    route2 = tmp_path / "route2"
    monkeypatch.setattr(launcher, "ROOT", route2)
    monkeypatch.setattr(critic_launcher, "ROOT", route2)
    base = json.loads(launcher.BASE_PROTOCOL.read_text(encoding="utf-8"))
    base["critic_readiness_path"] = str(route2 / "critic/readiness.json")
    base["critic_refit_manifest_path"] = str(route2 / "critic/refit.json")
    base["source_token_cache_path"] = str(route2 / "cache/source.pt")
    base_path = tmp_path / "base_protocol.json"
    _write(base_path, base)

    paths = {
        name: route2 / f"inputs/{name}.json"
        for name in (
            "setflow_schedule",
            "setflow_recovered_protocol",
            "setflow_runtime",
            "setflow_gate",
            "setflow_manifest",
            "critic_runtime",
            "critic_preflight",
            "setflow_preflight",
            "critic_cache",
            "setflow_cache",
            "critic_authorization",
            "setflow_authorization",
        )
    }
    for path in paths.values():
        _write(path, {})
    runner_receipt_path = (
        launcher.expected_runner_verification_receipt_path_v403(BRIDGE_HEAD)
    )
    _write(runner_receipt_path, _runner_receipt())
    _write(
        paths["setflow_schedule"],
        {
            "posttraining_bindings": {
                "config_manifest_path": str(paths["setflow_manifest"])
            }
        },
    )
    _write(
        paths["critic_preflight"],
        {"authorization_path": str(paths["critic_authorization"])},
    )
    for path in (
        Path(base["critic_readiness_path"]),
        Path(base["critic_refit_manifest_path"]),
        Path(base["source_token_cache_path"]),
    ):
        _write(path, {}) if path.suffix == ".json" else path.parent.mkdir(
            parents=True, exist_ok=True
        )
    Path(base["source_token_cache_path"]).touch()

    setflow_lineage = {
        "confirmation_gate_path": str(paths["setflow_gate"]),
        "runtime_config_paths": {
            str(seed): str(route2 / f"setflow/seed_{seed}.json")
            for seed in launcher.SEEDS
        },
        "posttraining_schedule_path": str(paths["setflow_schedule"]),
        "posttraining_runtime_path": str(paths["setflow_runtime"]),
        "posttraining_runner_git_head": SETFLOW_POSTTRAINING_HEAD,
        "training_runner_git_head": SETFLOW_TRAINING_RUNNER_HEAD,
        "training_git_head": SETFLOW_TRAINING_HEAD,
        "recovery_validation_git_head": SETFLOW_VALIDATION_HEAD,
        "recovered_protocol_path": str(paths["setflow_recovered_protocol"]),
        "config_manifest_path": str(paths["setflow_manifest"]),
    }
    critic_lineage = {
        "loso_runtime_path": str(paths["critic_runtime"]),
        "training_runner_git_head": CRITIC_RUNNER_HEAD,
        "readiness_path": base["critic_readiness_path"],
        "refit_manifest_path": base["critic_refit_manifest_path"],
    }
    prerequisite_lineage = {
        "critic_preflight_path": str(paths["critic_preflight"]),
        "critic_preflight_authorization_path": str(
            paths["critic_authorization"]
        ),
        "critic_preflight_runner_git_head": CRITIC_PREFLIGHT_HEAD,
        "critic_cache_summary_path": str(paths["critic_cache"]),
        "critic_cache_experiment_git_head": CRITIC_CACHE_HEAD,
        "setflow_preflight_path": str(paths["setflow_preflight"]),
        "setflow_preflight_authorization_path": str(
            paths["setflow_authorization"]
        ),
        "setflow_preflight_runner_git_head": SETFLOW_PREFLIGHT_HEAD,
        "setflow_cache_receipt_path": str(paths["setflow_cache"]),
        "setflow_cache_experiment_git_head": SETFLOW_CACHE_HEAD,
    }
    monkeypatch.setattr(
        launcher,
        "validate_setflow_lineage_v403",
        lambda *args, **kwargs: setflow_lineage,
    )
    monkeypatch.setattr(
        launcher,
        "validate_critic_lineage_v403",
        lambda *args, **kwargs: critic_lineage,
    )
    monkeypatch.setattr(
        launcher,
        "validate_preflight_cache_lineage_v403",
        lambda *args, **kwargs: prerequisite_lineage,
    )
    monkeypatch.setattr(
        launcher,
        "command",
        lambda arguments: SimpleNamespace(
            stdout=(BRIDGE_HEAD + "\n")
            if arguments == ["git", "rev-parse", "HEAD"]
            else ""
        ),
    )
    monkeypatch.setattr(
        launcher,
        "authorize_guidance",
        lambda *args, **kwargs: {
            "status": "XEDITFLOW_V4_GUIDANCE_AUTHORIZED",
            "guidance_authorized": True,
        },
    )
    materialized = {"called": False}

    def mock_strongest(protocol, *, physical_gpu_index):
        materialized["called"] = True
        return {
            "status": "MATERIALIZED_BEFORE_V4_GUIDANCE_SCREEN",
            "method_id": "strongest_matched_baseline",
            "score_table_path": protocol["strongest_closed_score_table_path"],
            "summary_path": protocol["strongest_closed_score_summary_path"],
            "guidance_winner_used": False,
            "baseline_reselected_for_v4": False,
            "physical_gpu_index": physical_gpu_index,
        }

    monkeypatch.setattr(
        launcher, "materialize_strongest_closed_score_v403", mock_strongest
    )
    launched = {"screen": False}

    def mock_screen(*args, **kwargs):
        launched["screen"] = True
        derived = json.loads(Path(kwargs["protocol_path"]).read_text())
        return {
            "runtime_manifest": str(route2 / "screen_execution/runtime.json"),
            "guidance_screen_gate_path": str(
                Path(derived["guidance_screen_output_root"])
                / "guidance_screen_gate.json"
            ),
            "scheduler_pid": 1234,
        }

    monkeypatch.setattr(launcher, "launch_guidance_screen", mock_screen)
    assert "strongest_closed_score_table_path" not in inspect.signature(
        launcher.run
    ).parameters

    receipt = launcher.run(
        BRIDGE_HEAD,
        runner_verification_receipt_path=runner_receipt_path,
        base_protocol_path=base_path,
        setflow_posttraining_schedule_path=paths["setflow_schedule"],
        setflow_recovered_protocol_path=paths["setflow_recovered_protocol"],
        setflow_posttraining_runtime_path=paths["setflow_runtime"],
        setflow_confirmation_gate_path=paths["setflow_gate"],
        critic_loso_runtime_path=paths["critic_runtime"],
        critic_readiness_path=Path(base["critic_readiness_path"]),
        critic_refit_manifest_path=Path(base["critic_refit_manifest_path"]),
        critic_preflight_path=paths["critic_preflight"],
        setflow_preflight_path=paths["setflow_preflight"],
        critic_cache_summary_path=paths["critic_cache"],
        setflow_cache_receipt_path=paths["setflow_cache"],
        setflow_preflight_authorization_path=paths["setflow_authorization"],
        strongest_score_physical_gpu=3,
        protocol_output=None,
    )
    assert launched["screen"] is True
    assert materialized["called"] is True
    assert receipt["final_successor"]["status"] == (
        "NOT_LAUNCHED_BEFORE_GUIDANCE_SCREEN_TERMINAL"
    )
    assert receipt["final_successor"]["strongest_closed_score_table_path"].endswith(
        "/frozen_method_scores.private.jsonl"
    )
    resolution = receipt["final_successor"]["guidance_selection_resolution"]
    assert resolution["winner_resolves_only"] == [
        "kappa",
        "temperature",
        "beta_max",
    ]
    assert resolution["single_guidance_winner_bound_before_frozen_gate"] is False
    assert receipt["runner_verification_receipt"]["runner_git_head"] == BRIDGE_HEAD
