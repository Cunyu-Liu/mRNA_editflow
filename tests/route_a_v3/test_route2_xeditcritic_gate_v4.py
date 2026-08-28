from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from core.route2_xeditcritic_gate_v4 import (
    adjudicate_critic_confirmation_v4,
    adjudicate_critic_frozen_test_v4,
    build_critic_confirmation_seed_payload_v4,
    evaluate_xeditcritic_v4_screen,
)
from scripts.route_a_v3.adjudicate_route2_xeditcritic_v4_screen import (
    run as adjudicate_screen,
)
import scripts.route_a_v3.adjudicate_route2_xeditcritic_v4_screen as critic_screen_adjudicator


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/route_a_v3_route2_xeditcritic_v4_screen_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _summary(run: dict, rho: float, *, mae: float = 1.2) -> dict:
    applicable = _config()["screen_gate"]["permutation_applicable_tasks"]
    remaining = [f"OTHER_TASK_{index}" for index in range(3)]
    tasks = applicable + remaining
    candidate_permutation = bool(run.get("candidate_bundle_permutation", False))
    output_directory = Path(_config()["output_root"]) / run["run_id"]
    initialization_scope = (
        "NOT_CLAIMED_DIFFERENT_C0_ARCHITECTURE"
        if run["model"] == "C0-V4"
        else "NOT_CLAIMED_PARAMETER_MATCHED_DIFFERENT_MODULE"
        if run["mechanism"] == "NO_CROSS"
        else "SHARED_V4_CONSTRUCTOR_WITHIN_IDENTICAL_ARCHITECTURE"
    )
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_screen_run.v2",
        "status": "TERMINAL_XEDITCRITIC_V4_SCREEN_RUN_COMPLETE",
        "run_id": run["run_id"],
        "model_kind": run["model"],
        "control_mode": "CANDIDATE_BUNDLE_PERMUTATION" if candidate_permutation else run["control"],
        "mechanism_mode": run["mechanism"],
        "candidate_bundle_permutation": candidate_permutation,
        "candidate_permutation_summary": {
            "complete_candidate_bundle_permuted": candidate_permutation,
            "exact_source_task_strata": candidate_permutation,
            "eligible_tasks": applicable if candidate_permutation else [],
        },
        "seed": 20260907,
        "parameter_initialization_seed": 20260907,
        "parameter_initialization_seed_applied_before_model_construction": True,
        "parameter_initialization_tensor_identity_scope": initialization_scope,
        "training_git_head": "a" * 40,
        "cuda_available": True,
        "cuda_device": "cuda:0",
        "cuda_device_name": "NVIDIA A100-SXM4-80GB",
        "a100_device_verified": True,
        "bf16_supported": True,
        "train_record_count": 89580,
        "validation_record_count": 18293,
        "pass_count": 8,
        "selected_pass": 8,
        "update_count": 22416,
        "selection_policy": "FINAL_PASS_8_FIXED_NO_VALIDATION_PEAK_RESELECTION",
        "physical_batch_size": 8,
        "effective_batch_size": 32,
        "singleton_forward_count": 0,
        "precision": "BF16_FORWARD_FP32_EFFECTIVE_OBJECTIVE",
        "cpu_fallback_used": False,
        "parameter_changed": True,
        "output_directory": str(output_directory),
        "training_summary_path": str(output_directory / "run_summary.json"),
        "checkpoint_path": str(
            output_directory / "final_pass_8_checkpoint.pt"
        ),
        "training_attempt_path": str(output_directory / "training_attempt.json"),
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
        "passes": [{"validation_metric_read": False} for _ in range(8)],
        "capacity": {
            "trainable_parameter_count": 1_000_000 if run["model"] == "C0-V4" else 170_481_733
        },
        "peak_vram_bytes": 30 * 1024**3,
        "final_validation": {
            "task_count": 9,
            "task_macro_spearman": rho,
            "task_macro_standardized_mae": mae,
            "positive_task_count": 9 if rho > 0 else 0,
            "prediction_std": 0.5,
            "tasks": {
                task: {"spearman": rho, "standardized_mae": mae, "record_count": 10}
                for task in tasks
            },
        },
    }


def _passing_package() -> tuple[dict, dict[str, dict], dict]:
    config = _config()
    rho = {
        "c0_v4": 0.20,
        "v4_full": 0.40,
        "v4_source_only": 0.25,
        "v4_edit_metadata_only": 0.26,
        "v4_no_candidate_sequence": 0.27,
        "v4_candidate_bundle_permutation": 0.30,
        "v4_no_cross": 0.35,
        "v4_no_moe": 0.36,
    }
    summaries = {
        run["run_id"]: _summary(run, rho[run["run_id"]])
        for run in config["required_screen_runs"]
    }
    preflight = {
        "status": "XEDITCRITIC_V4_PREFLIGHT_PASS",
        "passed": True,
        "trainable_parameter_count": 170_481_733,
        "selected_physical_batch": 8,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    return config, summaries, preflight


def _expected_heads(summaries: dict[str, dict]) -> dict[str, str]:
    return {
        run_id: str(summary["training_git_head"])
        for run_id, summary in summaries.items()
    }


def test_v4_screen_gate_passes_only_the_full_strict_package() -> None:
    config, summaries, preflight = _passing_package()
    result = evaluate_xeditcritic_v4_screen(
        config,
        summaries,
        c3_reference_spearman=0.20,
        preflight=preflight,
        expected_training_git_heads=_expected_heads(summaries),
    )
    assert result["status"] == "XEDITCRITIC_V4_SCREEN_PASS"
    assert result["minimum_required_task_macro_spearman"] == pytest.approx(0.30)
    assert result["c0_task_win_count"] == 9
    assert result["permutation_task_win_count"] == 6
    assert result["confirmation_authorized"] is True
    assert result["development_test_authorized"] is False


def test_v4_screen_gate_is_no_go_for_c3_formula_control_or_mechanism_failure() -> None:
    config, summaries, preflight = _passing_package()
    result = evaluate_xeditcritic_v4_screen(
        config,
        summaries,
        c3_reference_spearman=0.36,
        preflight=preflight,
        expected_training_git_heads=_expected_heads(summaries),
    )
    assert result["minimum_required_task_macro_spearman"] == pytest.approx(0.41)
    assert result["status"] == "XEDITCRITIC_V4_SCREEN_NO_GO"
    config, summaries, preflight = _passing_package()
    summaries["v4_no_moe"]["final_validation"]["task_macro_spearman"] = 0.39
    for row in summaries["v4_no_moe"]["final_validation"]["tasks"].values():
        row["spearman"] = 0.39
    result = evaluate_xeditcritic_v4_screen(
        config,
        summaries,
        c3_reference_spearman=0.20,
        preflight=preflight,
        expected_training_git_heads=_expected_heads(summaries),
    )
    assert result["checks"]["no_moe_margin"] is False
    assert result["passed"] is False


def test_v4_screen_gate_requires_five_of_exact_six_permutation_task_wins() -> None:
    config, summaries, preflight = _passing_package()
    applicable = config["screen_gate"]["permutation_applicable_tasks"]
    for task in applicable[:2]:
        summaries["v4_candidate_bundle_permutation"]["final_validation"]["tasks"][task]["spearman"] = 0.45
    summaries["v4_candidate_bundle_permutation"]["final_validation"]["task_macro_spearman"] = (
        2 * 0.45 + 7 * 0.30
    ) / 9
    result = evaluate_xeditcritic_v4_screen(
        config,
        summaries,
        c3_reference_spearman=0.20,
        preflight=preflight,
        expected_training_git_heads=_expected_heads(summaries),
    )
    assert result["permutation_task_win_count"] == 4
    assert result["checks"]["permutation_five_of_six_tasks"] is False
    assert result["status"] == "XEDITCRITIC_V4_SCREEN_NO_GO"


def test_v4_screen_gate_rejects_any_protected_read_or_parameter_drift() -> None:
    config, summaries, preflight = _passing_package()
    protected = copy.deepcopy(summaries)
    protected["v4_full"]["development_test_outcome_reads"] = 1
    try:
        evaluate_xeditcritic_v4_screen(
            config,
            protected,
            c3_reference_spearman=0.20,
            preflight=preflight,
            expected_training_git_heads=_expected_heads(protected),
        )
    except Exception as exc:
        assert "Development TEST" in str(exc)
    else:
        raise AssertionError("protected read was accepted")
    drift = copy.deepcopy(summaries)
    drift["v4_no_cross"]["capacity"]["trainable_parameter_count"] += 1
    try:
        evaluate_xeditcritic_v4_screen(
            config,
            drift,
            c3_reference_spearman=0.20,
            preflight=preflight,
            expected_training_git_heads=_expected_heads(drift),
        )
    except Exception as exc:
        assert "parameter count" in str(exc)
    else:
        raise AssertionError("parameter drift was accepted")


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    (
        ("parameter_initialization_seed", 7, "initialization evidence"),
        (
            "parameter_initialization_seed_applied_before_model_construction",
            False,
            "initialization evidence",
        ),
        ("cuda_available", False, "CUDA/A100/BF16"),
        ("cuda_device", "cpu", "CUDA/A100/BF16"),
        ("a100_device_verified", False, "CUDA/A100/BF16"),
        ("bf16_supported", False, "CUDA/A100/BF16"),
        ("cpu_fallback_used", True, "CUDA/A100/BF16"),
        ("training_git_head", "b" * 39, "Git HEAD"),
        ("checkpoint_path", "/wrong/checkpoint.pt", "path binding"),
    ),
)
def test_v4_screen_rejects_initialization_device_head_or_path_drift(
    field: str, invalid: object, message: str
) -> None:
    config, summaries, preflight = _passing_package()
    expected_heads = _expected_heads(summaries)
    summaries["v4_full"][field] = invalid
    with pytest.raises(Exception, match=message):
        evaluate_xeditcritic_v4_screen(
            config,
            summaries,
            c3_reference_spearman=0.20,
            preflight=preflight,
            expected_training_git_heads=expected_heads,
        )


def test_v4_screen_peak_vram_is_positive_diagnostic_not_a_ceiling() -> None:
    config, summaries, preflight = _passing_package()
    summaries["v4_full"]["peak_vram_bytes"] = 80 * 1024**3
    result = evaluate_xeditcritic_v4_screen(
        config,
        summaries,
        c3_reference_spearman=0.20,
        preflight=preflight,
        expected_training_git_heads=_expected_heads(summaries),
    )
    assert result["status"] == "XEDITCRITIC_V4_SCREEN_PASS"
    assert result["peak_vram_diagnostic_gib_by_run"]["v4_full"] == 80.0


def test_v4_screen_rejects_authorized_runner_head_drift() -> None:
    config, summaries, preflight = _passing_package()
    heads = _expected_heads(summaries)
    heads["v4_no_moe"] = "b" * 40
    with pytest.raises(Exception, match="training Git HEAD"):
        evaluate_xeditcritic_v4_screen(
            config,
            summaries,
            c3_reference_spearman=0.20,
            preflight=preflight,
            expected_training_git_heads=heads,
        )


def test_v4_screen_mixed_provenance_limits_v1_to_existing_full_and_c0() -> None:
    config, summaries, preflight = _passing_package()
    historical = {
        "c0_v4": (
            "93703adec7a4c76b4466d3aaae8684620bee985a",
            "HISTORICAL_MATCHED_C0_TERMINAL_SUMMARY",
        ),
        "v4_full": (
            "f34ab7d865bb2477bfe24c1d0a7c9f5301a24cea",
            "CURRENT_V403_REPAIRED_FULL_TERMINAL_SUMMARY",
        ),
    }
    control_head = "c" * 40
    provenance = {}
    for run_id, summary in summaries.items():
        is_historical = run_id in historical
        head, role = historical.get(
            run_id, (control_head, "CURRENT_HEAD_CONTROL_TERMINAL_SUMMARY")
        )
        if is_historical:
            summary["schema_version"] = (
                "route_a_v3_route2_xeditcritic_v4_screen_run.v1"
            )
        else:
            summary["training_git_head"] = control_head
            output_directory = Path("/tmp/current-head-controls") / run_id
            summary.update(
                {
                    "output_directory": str(output_directory),
                    "training_summary_path": str(
                        output_directory / "run_summary.json"
                    ),
                    "checkpoint_path": str(
                        output_directory / "final_pass_8_checkpoint.pt"
                    ),
                    "training_attempt_path": str(
                        output_directory / "training_attempt.json"
                    ),
                }
            )
        provenance[run_id] = {
            "run_id": run_id,
            "summary_path": summary["training_summary_path"],
            "training_git_head": head,
            "source_role": role,
            "authorized_git_head": head,
            "legacy_terminal_summary": is_historical,
            "run_id_authorization_verified": True,
            "authorization_protected_outcome_reads_verified_zero": True,
        }
    expected_heads = {
        run_id: control_head
        for run_id in summaries
        if run_id not in historical
    }
    result = evaluate_xeditcritic_v4_screen(
        config,
        summaries,
        c3_reference_spearman=0.20,
        preflight=preflight,
        terminal_provenance=provenance,
        expected_training_git_heads=expected_heads,
    )
    assert result["status"] == "XEDITCRITIC_V4_SCREEN_PASS"
    correct_checkpoint = summaries["v4_no_moe"]["checkpoint_path"]
    summaries["v4_no_moe"]["checkpoint_path"] = "/tmp/wrong/checkpoint.pt"
    with pytest.raises(Exception, match="path binding"):
        evaluate_xeditcritic_v4_screen(
            config,
            summaries,
            c3_reference_spearman=0.20,
            preflight=preflight,
            terminal_provenance=provenance,
            expected_training_git_heads=expected_heads,
        )
    summaries["v4_no_moe"]["checkpoint_path"] = correct_checkpoint
    summaries["v4_no_moe"]["schema_version"] = (
        "route_a_v3_route2_xeditcritic_v4_screen_run.v1"
    )
    provenance["v4_no_moe"]["legacy_terminal_summary"] = True
    with pytest.raises(Exception, match="mixed legacy/nonlegacy"):
        evaluate_xeditcritic_v4_screen(
            config,
            summaries,
            c3_reference_spearman=0.20,
            preflight=preflight,
            terminal_provenance=provenance,
            expected_training_git_heads=expected_heads,
        )


def test_adjudicator_keeps_terminal_technical_failure_out_of_scientific_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    config["output_root"] = str(tmp_path / "runs")
    config["screen_gate_output"] = str(tmp_path / "runs" / "screen_gate.json")
    config["c3_read_once_reference_adjudication"] = str(
        tmp_path / "c3_reference.json"
    )
    Path(config["output_root"]).mkdir()
    for row in config["required_screen_runs"]:
        run_directory = Path(config["output_root"]) / row["run_id"]
        run_directory.mkdir()
        (run_directory / "failure.json").write_text(
            json.dumps(
                {
                    "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
                    "error_type": "SyntheticFailure",
                }
            ),
            encoding="utf-8",
        )
    Path(config["c3_read_once_reference_adjudication"]).write_text(
        json.dumps(
            {
                "status": "C3_V4_REFERENCE_READ_ONCE_COMPLETE",
                "terminal_summaries_read_count": 5,
                "c3_reference_task_macro_spearman": 0.2,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            }
        ),
        encoding="utf-8",
    )
    replace_calls: list[tuple[Path, Path]] = []
    original_replace = os.replace

    def recording_replace(source: str | Path, destination: str | Path) -> None:
        replace_calls.append((Path(source), Path(destination)))
        original_replace(source, destination)

    monkeypatch.setattr(critic_screen_adjudicator.os, "replace", recording_replace)
    with pytest.raises(Exception, match="scientific gate was not written"):
        adjudicate_screen(config, expected_runner_head="a" * 40)
    output = Path(config["screen_gate_output"])
    failure_output = output.with_name(output.stem + ".failed.json")
    result = json.loads(failure_output.read_text(encoding="utf-8"))
    assert result["status"] == "XEDITCRITIC_V4_SCREEN_TECHNICAL_FAILURE"
    assert result["technical_failure_run_ids"] == sorted(
        row["run_id"] for row in config["required_screen_runs"]
    )
    assert result["confirmation_authorized"] is False
    assert result["development_test_authorized"] is False
    failure_partial = failure_output.with_suffix(failure_output.suffix + ".partial")
    assert replace_calls == [(failure_partial, failure_output)]
    assert not output.exists()
    assert failure_output.exists()
    assert not failure_partial.exists()


def test_adjudicator_refuses_a_stale_partial_gate(tmp_path: Path) -> None:
    config = _config()
    output = tmp_path / "runs" / "screen_gate.json"
    config["screen_gate_output"] = str(output)
    output.parent.mkdir(parents=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text("interrupted", encoding="utf-8")

    with pytest.raises(Exception, match="partial Critic V4 screen gate already exists"):
        adjudicate_screen(config, expected_runner_head="a" * 40)

    assert not output.exists()
    assert partial.read_text(encoding="utf-8") == "interrupted"


def _confirmation_summary(run: dict, seed: int, rho: float, mae: float = 1.2) -> dict:
    result = _summary(run, rho, mae=mae)
    result["schema_version"] = "route_a_v3_route2_xeditcritic_v4_confirmation_run.v2"
    result["status"] = "TERMINAL_XEDITCRITIC_V4_CONFIRMATION_RUN_COMPLETE"
    result["run_stage"] = "CONFIRMATION"
    result["seed"] = seed
    result["parameter_initialization_seed"] = seed
    output_directory = Path(f"/tmp/critic-confirmation/seed_{seed}") / run["run_id"]
    result.update(
        {
            "output_directory": str(output_directory),
            "training_summary_path": str(output_directory / "run_summary.json"),
            "checkpoint_path": str(
                output_directory / "final_pass_8_checkpoint.pt"
            ),
            "training_attempt_path": str(
                output_directory / "training_attempt.json"
            ),
        }
    )
    return result


def _confirmation_payloads() -> dict:
    config = _config()
    runs = {row["run_id"]: row for row in config["required_screen_runs"]}
    return {
        seed: {
            "candidate_summary": _confirmation_summary(runs["v4_full"], seed, 0.40),
            "baseline_summary": _confirmation_summary(runs["c0_v4"], seed, 0.20, mae=1.3),
            "bootstrap": {
                "analysis_unit": "SOURCE_GROUP_WITHIN_TASK",
                "task_count": 9,
                "source_group_count": 18,
                "bootstrap_iterations": 10000,
                "defined_bootstrap_iterations": 10000,
                "point_task_macro_spearman_difference": 0.20,
                "task_macro_spearman_difference_ci_95": [0.10, 0.30],
            },
            "training_config_identity": {
                "training_seed": seed,
                "run_stage": "CONFIRMATION",
                "required_run_ids": ["v4_full", "c0_v4"],
                "output_root": f"/tmp/critic-confirmation/seed_{seed}",
                "confirmation_runner_git_head": "a" * 40,
            },
        }
        for seed in (20260908, 20260909, 20260910)
    }


def test_v4_confirmation_gate_requires_three_seed_strict_cohort() -> None:
    config, _, preflight = _passing_package()
    result = adjudicate_critic_confirmation_v4(
        config, _confirmation_payloads(), preflight=preflight
    )
    assert result["status"] == "XEDITCRITIC_V4_THREE_SEED_PASS"
    assert result["development_test_authorized"] is True
    assert result["atomic_development_test_only"] is True
    assert result["additional_seed_authorized"] is False
    payloads = _confirmation_payloads()
    payloads[20260910]["candidate_summary"]["final_validation"][
        "task_macro_spearman"
    ] = 0.29
    for row in payloads[20260910]["candidate_summary"]["final_validation"][
        "tasks"
    ].values():
        row["spearman"] = 0.29
    payloads[20260910]["candidate_summary"]["final_validation"][
        "positive_task_count"
    ] = 9
    payloads[20260910]["bootstrap"][
        "point_task_macro_spearman_difference"
    ] = 0.09
    result = adjudicate_critic_confirmation_v4(config, payloads, preflight=preflight)
    assert result["status"] == "XEDITCRITIC_V4_THREE_SEED_NO_GO"
    assert result["development_test_authorized"] is False


@pytest.mark.parametrize(
    ("target", "field", "invalid", "message"),
    (
        ("candidate_summary", "parameter_initialization_seed", 7, "initialization"),
        ("candidate_summary", "cpu_fallback_used", True, "CUDA/A100/BF16"),
        ("candidate_summary", "bf16_supported", False, "CUDA/A100/BF16"),
        ("candidate_summary", "training_git_head", "b" * 40, "Git HEAD"),
        (
            "candidate_summary",
            "training_summary_path",
            "/tmp/critic-confirmation/seed_20260909/v4_full/run_summary.json",
            "path binding",
        ),
    ),
)
def test_v4_confirmation_rejects_training_evidence_drift(
    target: str, field: str, invalid: object, message: str
) -> None:
    config, _, preflight = _passing_package()
    payloads = _confirmation_payloads()
    payloads[20260908][target][field] = invalid
    with pytest.raises(Exception, match=message):
        adjudicate_critic_confirmation_v4(config, payloads, preflight=preflight)


def test_v4_confirmation_rejects_runner_head_drift_across_seeds() -> None:
    config, _, preflight = _passing_package()
    payloads = _confirmation_payloads()
    payloads[20260910]["training_config_identity"][
        "confirmation_runner_git_head"
    ] = "b" * 40
    for summary_key in ("candidate_summary", "baseline_summary"):
        payloads[20260910][summary_key]["training_git_head"] = "b" * 40
    with pytest.raises(Exception, match="differs across seeds"):
        adjudicate_critic_confirmation_v4(config, payloads, preflight=preflight)


def test_v4_confirmation_bootstrap_is_exact_source_group_pairing() -> None:
    candidate = []
    baseline = []
    for task_index in range(9):
        for group_index in range(2):
            for row_index, target in enumerate((-1.0, 0.0, 1.0)):
                common = {
                    "record_id": f"{task_index}-{group_index}-{row_index}",
                    "source_group_id": f"task-{task_index}-group-{group_index}",
                    "task_id": f"task-{task_index}",
                    "target": target,
                    "scaled_target": target,
                }
                candidate.append({**common, "prediction": target})
                baseline.append({**common, "prediction": -target})
    result = build_critic_confirmation_seed_payload_v4(
        {},
        {},
        candidate,
        baseline,
        {
            "training_seed": 20260908,
            "run_stage": "CONFIRMATION",
            "required_confirmation_run_ids": ["v4_full", "c0_v4"],
            "output_root": "/tmp/critic-confirmation/seed_20260908",
            "confirmation_runner_git_head": "a" * 40,
        },
        seed=20260908,
        bootstrap_seed=2026090801,
    )
    assert result["bootstrap"]["bootstrap_iterations"] == 10000
    assert result["bootstrap"]["task_macro_spearman_difference_ci_95"][0] > 0.0


def test_v4_frozen_test_gate_requires_single_atomic_access_and_strict_metrics() -> None:
    common = {
        "status": "ATOMIC_FROZEN_DEVELOPMENT_TEST_EVALUATION_COMPLETE",
        "test_record_count": 18292,
        "development_test_outcomes_accessed": True,
        "development_test_access_event_count": 1,
        "general_test_projection_persisted": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    candidate = {
        **common,
        "test_metrics": {
            "task_count": 9,
            "task_macro_spearman": 0.35,
            "task_macro_standardized_mae": 1.2,
            "positive_task_count": 9,
        },
    }
    baseline = {
        **common,
        "test_metrics": {
            "task_count": 9,
            "task_macro_spearman": 0.20,
            "task_macro_standardized_mae": 1.3,
            "positive_task_count": 8,
        },
    }
    bootstrap = {
        "analysis_unit": "SOURCE_GROUP_WITHIN_TASK",
        "bootstrap_iterations": 10000,
        "point_task_macro_spearman_difference": 0.15,
        "task_macro_spearman_difference_ci_95": [0.05, 0.25],
    }
    result = adjudicate_critic_frozen_test_v4(candidate, baseline, bootstrap)
    assert result["status"] == "XEDITCRITIC_V4_FROZEN_TEST_PASS"
    assert result["all_development_refit_authorized"] is True
    candidate["general_test_projection_persisted"] = True
    with pytest.raises(Exception, match="single and atomic"):
        adjudicate_critic_frozen_test_v4(candidate, baseline, bootstrap)
