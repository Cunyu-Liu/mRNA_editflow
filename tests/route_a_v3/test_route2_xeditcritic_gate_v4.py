from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from core.route2_xeditcritic_gate_v4 import evaluate_xeditcritic_v4_screen
from scripts.route_a_v3.adjudicate_route2_xeditcritic_v4_screen import (
    run as adjudicate_screen,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/route_a_v3_route2_xeditcritic_v4_screen_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _summary(run: dict, rho: float, *, mae: float = 1.2) -> dict:
    applicable = _config()["screen_gate"]["permutation_applicable_tasks"]
    remaining = [f"OTHER_TASK_{index}" for index in range(3)]
    tasks = applicable + remaining
    candidate_permutation = bool(run.get("candidate_bundle_permutation", False))
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_screen_run.v1",
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
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
        "passes": [{"validation_metric_read": False} for _ in range(8)],
        "capacity": {
            "trainable_parameter_count": 1_000_000 if run["model"] == "C0-V4" else 173_692_549
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
        "trainable_parameter_count": 173_692_549,
        "selected_physical_batch": 8,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    return config, summaries, preflight


def test_v4_screen_gate_passes_only_the_full_strict_package() -> None:
    config, summaries, preflight = _passing_package()
    result = evaluate_xeditcritic_v4_screen(
        config,
        summaries,
        c3_reference_spearman=0.20,
        preflight=preflight,
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
    )
    assert result["minimum_required_task_macro_spearman"] == pytest.approx(0.41)
    assert result["status"] == "XEDITCRITIC_V4_SCREEN_NO_GO"
    config, summaries, preflight = _passing_package()
    summaries["v4_no_moe"]["final_validation"]["task_macro_spearman"] = 0.39
    for row in summaries["v4_no_moe"]["final_validation"]["tasks"].values():
        row["spearman"] = 0.39
    result = evaluate_xeditcritic_v4_screen(
        config, summaries, c3_reference_spearman=0.20, preflight=preflight
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
        config, summaries, c3_reference_spearman=0.20, preflight=preflight
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
            config, protected, c3_reference_spearman=0.20, preflight=preflight
        )
    except Exception as exc:
        assert "Development TEST" in str(exc)
    else:
        raise AssertionError("protected read was accepted")
    drift = copy.deepcopy(summaries)
    drift["v4_no_cross"]["capacity"]["trainable_parameter_count"] += 1
    try:
        evaluate_xeditcritic_v4_screen(
            config, drift, c3_reference_spearman=0.20, preflight=preflight
        )
    except Exception as exc:
        assert "parameter count" in str(exc)
    else:
        raise AssertionError("parameter drift was accepted")


def test_adjudicator_turns_any_terminal_technical_failure_into_no_go(
    tmp_path: Path,
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
    result = adjudicate_screen(config)
    assert result["status"] == "XEDITCRITIC_V4_SCREEN_NO_GO"
    assert result["technical_failure_run_ids"] == sorted(
        row["run_id"] for row in config["required_screen_runs"]
    )
    assert result["confirmation_authorized"] is False
    assert result["development_test_authorized"] is False
    assert Path(config["screen_gate_output"]).exists()
