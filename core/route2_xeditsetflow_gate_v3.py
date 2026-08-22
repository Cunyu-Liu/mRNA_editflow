"""Frozen screen and confirmation gates for XEditSetFlow V3."""

from __future__ import annotations

import math
from typing import Any, Mapping


class XEditSetFlowGateV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditSetFlowGateV3Error(message)


def _finite(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    result = float(value)
    _require(math.isfinite(result), f"{label} is nonfinite")
    return result


def adjudicate_setflow_screen_v3(
    f0_replay: Mapping[str, Any],
    training: Mapping[str, Mapping[str, Any]],
    validation: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _require(set(training) == {"f1", "f2", "f3"}, "SetFlow screen training arms are incomplete")
    _require(set(validation) == {"f1", "f2", "f3"}, "SetFlow screen validation arms are incomplete")
    _require(f0_replay.get("status") == "FROZEN_BASE_FLOW_V2_COMMON_SET_NLL_REPLAY_COMPLETE", "F0 common-NLL replay is not terminal")
    f0_nll = _finite(f0_replay.get("common_validation_set_marginal_nll"), "F0 common NLL")
    _require(f0_nll > 0.0, "F0 common NLL is not positive")
    rows: dict[str, dict[str, Any]] = {}
    for arm in ("f1", "f2", "f3"):
        train = training[arm]
        valid = validation[arm]
        _require(train.get("status") == "XEDITSETFLOW_V3_GPU_TRAINING_COMPLETE", f"{arm} training is not terminal")
        _require(valid.get("status") in {"FLOW_G0_READY", "FLOW_G0_VALIDATION_FAIL"}, f"{arm} validation is not terminal")
        _require(int(train.get("seed", -1)) == int(valid.get("seed", -2)) == 20260903, f"{arm} screen seed changed")
        _require(train.get("development_test_outcomes_accessed") is False and valid.get("development_test_outcomes_accessed") is False, f"{arm} accessed Development TEST")
        _require(train.get("evaluation_outcomes_accessed") is False and valid.get("evaluation_outcomes_accessed") is False, f"{arm} accessed Evaluation")
        nll = _finite(train.get("best_validation_common_set_marginal_nll"), f"{arm} common NLL")
        recovery = _finite(valid.get("source_macro_candidate_recovery_rate"), f"{arm} recovery")
        top_k = _finite(valid.get("source_macro_measured_top_k_recovery_at_k"), f"{arm} top-k recovery")
        unique = _finite(valid.get("source_macro_unique_candidate_rate"), f"{arm} unique rate")
        relative_nll_improvement = (f0_nll - nll) / f0_nll
        checks = {
            "common_nll_improvement_at_least_10pct": relative_nll_improvement >= 0.10,
            "source_macro_recovery_at_least_0_25": recovery >= 0.25,
            "source_macro_top_k_recovery_at_least_0_15": top_k >= 0.15,
            "source_macro_unique_rate_at_least_0_90": unique >= 0.90,
            "hard_legality_100pct": _finite(valid.get("hard_legality_rate"), f"{arm} legality") == 1.0,
            "edit_budget_violation_zero": int(valid.get("edit_budget_violation_count", -1)) == 0,
            "candidate_budget_violation_zero": int(valid.get("candidate_budget_violation_count", -1)) == 0,
            "trajectory_replay_failure_zero": int(valid.get("trajectory_replay_failure_count", -1)) == 0,
            "numerical_failure_zero": int(valid.get("numerical_failure_count", -1)) == 0,
        }
        selectable = arm in {"f2", "f3"}
        rows[arm] = {
            "selectable": selectable,
            "common_validation_set_marginal_nll": nll,
            "f0_common_validation_set_marginal_nll": f0_nll,
            "relative_common_nll_improvement": relative_nll_improvement,
            "source_macro_candidate_recovery_rate": recovery,
            "source_macro_measured_top_k_recovery_at_k": top_k,
            "source_macro_unique_candidate_rate": unique,
            "checks": checks,
            "passes_screen_gate": selectable and all(checks.values()),
        }
    passed = [arm for arm in ("f2", "f3") if rows[arm]["passes_screen_gate"]]
    selected = None
    selection_reason = "NO_SELECTABLE_ARM_PASSED"
    if len(passed) == 1:
        selected = passed[0]
        selection_reason = "ONLY_PASSING_SELECTABLE_ARM"
    elif len(passed) == 2:
        recovery_difference = abs(
            rows["f2"]["source_macro_candidate_recovery_rate"]
            - rows["f3"]["source_macro_candidate_recovery_rate"]
        )
        if recovery_difference > 0.01 and not math.isclose(
            recovery_difference, 0.01, rel_tol=0.0, abs_tol=1e-12
        ):
            selected = max(passed, key=lambda arm: rows[arm]["source_macro_candidate_recovery_rate"])
            selection_reason = "RECOVERY_DIFFERENCE_EXCEEDS_0_01"
        else:
            f2_top = rows["f2"]["source_macro_measured_top_k_recovery_at_k"]
            f3_top = rows["f3"]["source_macro_measured_top_k_recovery_at_k"]
            if f2_top != f3_top:
                selected = "f2" if f2_top > f3_top else "f3"
                selection_reason = "TOP_K_RECOVERY_TIE_BREAK"
            else:
                f2_nll = rows["f2"]["common_validation_set_marginal_nll"]
                f3_nll = rows["f3"]["common_validation_set_marginal_nll"]
                if f2_nll != f3_nll:
                    selected = "f2" if f2_nll < f3_nll else "f3"
                    selection_reason = "COMMON_SET_NLL_TIE_BREAK"
                else:
                    selected = "f2"
                    selection_reason = "SMALLER_F2_FINAL_TIE_BREAK"
    return {
        "schema_version": "route_a_v3_route2_xeditsetflow_screen_gate.v3",
        "status": "XEDITSETFLOW_V3_SCREEN_PASS" if selected else "XEDITSETFLOW_V3_SCREEN_NO_GO",
        "screen_seed": 20260903,
        "arms": rows,
        "selected_arm": selected,
        "selection_reason": selection_reason,
        "confirmation_authorized": selected is not None,
        "development_test_authorized": False,
        "guidance_authorized": False,
        "additional_seed_authorized": False,
    }
