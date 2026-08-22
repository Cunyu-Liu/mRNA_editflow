"""Frozen screen and confirmation gates for XEditSetFlow V3."""

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any, Mapping


class XEditSetFlowGateV3Error(RuntimeError):
    pass


SELECTABLE_TRAINABLE_PARAMETER_COUNTS_V3 = {
    "f2": 16_179_014,
    "f3": 42_197_158,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditSetFlowGateV3Error(message)


def _finite(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    result = float(value)
    _require(math.isfinite(result), f"{label} is nonfinite")
    return result


def _require_unguided_validation_identity_v3(
    validation: Mapping[str, Any], *, arm: str, seed: int
) -> None:
    _require(
        validation.get("arm") == arm
        and int(validation.get("seed", -1)) == seed
        and validation.get("method_id")
        == f"unguided_xeditsetflow_v3_{arm}_seed{seed}",
        "SetFlow unguided validation method identity differs",
    )
    _require(
        int(validation.get("source_count", -1)) == 891
        and int(validation.get("candidate_count", -1)) == 28_512
        and int(validation.get("trajectory_forward_batch_size", -1)) == 64,
        "SetFlow unguided validation cohort or execution budget differs",
    )
    _require(
        validation.get("cpu_fallback_used") is False
        and int(validation.get("parameter_update_count", -1)) == 0
        and validation.get("guided_critic_used") is False
        and validation.get("independent_evaluator_used") is False,
        "SetFlow unguided validation provenance differs",
    )
    _require(
        validation.get("development_test_outcomes_accessed") is False
        and int(validation.get("evaluation_records_read", -1)) == 0
        and validation.get("evaluation_outcomes_accessed") is False,
        "SetFlow unguided validation accessed protected outcome",
    )
    _require(
        validation.get("generated_candidates_grant_canonical_credit") is False
        and validation.get("biological_optimization_established") is False,
        "SetFlow unguided validation overclaims generated-candidate evidence",
    )
    small_graph = validation.get("small_graph_reference")
    _require(
        isinstance(small_graph, Mapping)
        and small_graph.get("status") == "PASS"
        and _finite(small_graph.get("total_variation"), "small-graph total variation")
        <= _finite(small_graph.get("tolerance"), "small-graph tolerance"),
        "SetFlow small-graph distribution exactness is absent or failed",
    )


def adjudicate_setflow_screen_v3(
    f0_replay: Mapping[str, Any],
    training: Mapping[str, Mapping[str, Any]],
    validation: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _require(set(training) == {"f1", "f2", "f3"}, "SetFlow screen training arms are incomplete")
    _require(set(validation) == {"f1", "f2", "f3"}, "SetFlow screen validation arms are incomplete")
    _require(f0_replay.get("status") == "FROZEN_BASE_FLOW_V2_COMMON_SET_NLL_REPLAY_COMPLETE", "F0 common-NLL replay is not terminal")
    _require(
        int(f0_replay.get("selected_epoch", -1)) == 1
        and int(f0_replay.get("trainable_parameter_count", -1)) == 817_957
        and int(f0_replay.get("parameter_update_count", -1)) == 0
        and f0_replay.get("parameter_changed_during_replay") is False,
        "F0 replay is not the frozen read-only Base Flow V2 reference",
    )
    _require(
        int(f0_replay.get("validation_record_count", -1)) == 15924
        and int(f0_replay.get("validation_states_per_record", -1)) == 2,
        "F0 common-validation cohort differs",
    )
    _require(
        f0_replay.get("development_test_outcomes_accessed") is False
        and f0_replay.get("evaluation_outcomes_accessed") is False
        and f0_replay.get("critic_score_used") is False
        and f0_replay.get("independent_evaluator_used") is False,
        "F0 replay accessed protected/model-selection information",
    )
    f0_nll = _finite(f0_replay.get("common_validation_set_marginal_nll"), "F0 common NLL")
    _require(f0_nll > 0.0, "F0 common NLL is not positive")
    rows: dict[str, dict[str, Any]] = {}
    for arm in ("f1", "f2", "f3"):
        train = training[arm]
        valid = validation[arm]
        _require(train.get("status") == "XEDITSETFLOW_V3_GPU_TRAINING_COMPLETE", f"{arm} training is not terminal")
        _require(valid.get("status") in {"FLOW_G0_READY", "FLOW_G0_VALIDATION_FAIL"}, f"{arm} validation is not terminal")
        _require(int(train.get("seed", -1)) == int(valid.get("seed", -2)) == 20260903, f"{arm} screen seed changed")
        _require(
            str(train.get("arm")) == str(valid.get("arm")) == arm
            and train.get("run_stage") == "SCREEN"
            and train.get("selectable") is (arm in {"f2", "f3"}),
            f"{arm} screen arm/role identity differs",
        )
        _require(
            int(train.get("train_record_count", -1)) == 68294
            and int(train.get("validation_record_count", -1)) == 15924
            and int(train.get("states_per_record_per_pass", -1)) == 2
            and int(train.get("effective_batch_size", -1)) == 32
            and int(train.get("maximum_passes", -1)) == 12,
            f"{arm} screen split or training budget differs",
        )
        _require(
            train.get("training_precision") == "BF16"
            and train.get("parameter_changed") is True
            and train.get("cpu_fallback_used") is False
            and int(train.get("development_test_record_count_withheld", -1)) == 18292
            and train.get("critic_score_used") is False
            and train.get("independent_evaluator_used") is False,
            f"{arm} screen training provenance differs",
        )
        trainable_parameter_count = int(train.get("trainable_parameter_count", -1))
        _require(trainable_parameter_count > 0, f"{arm} trainable parameter count is absent")
        if arm in SELECTABLE_TRAINABLE_PARAMETER_COUNTS_V3:
            _require(
                trainable_parameter_count
                == SELECTABLE_TRAINABLE_PARAMETER_COUNTS_V3[arm],
                f"{arm} trainable parameter count differs from the frozen arm",
            )
        _require_unguided_validation_identity_v3(
            valid, arm=arm, seed=20260903
        )
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


SETFLOW_CONFIRMATION_SEEDS_V3 = (20260904, 20260905, 20260906)


def require_setflow_confirmation_authorization_v3(
    config: Mapping[str, Any], *, arm: str
) -> None:
    seed = int(config["seed"])
    stage = str(config.get("run_stage", "SCREEN" if seed == 20260903 else "CONFIRMATION"))
    if stage == "SCREEN":
        _require(seed == 20260903, "SetFlow screen seed changed")
        return
    _require(stage == "CONFIRMATION", "SetFlow run stage is unsupported")
    _require(seed in SETFLOW_CONFIRMATION_SEEDS_V3, "SetFlow confirmation seed is undeclared")
    gate = json.loads(Path(str(config["screen_gate_path"])).read_text(encoding="utf-8"))
    selected = str(gate.get("selected_arm"))
    _require(
        gate.get("status") == "XEDITSETFLOW_V3_SCREEN_PASS"
        and gate.get("confirmation_authorized") is True,
        "SetFlow screen does not authorize confirmation",
    )
    _require(
        str(config.get("selected_arm")) == selected == arm,
        "SetFlow confirmation arm differs from the screen selection",
    )


def adjudicate_setflow_confirmation_v3(
    training: Mapping[int, Mapping[str, Any]],
    validation: Mapping[int, Mapping[str, Any]],
    *,
    selected_arm: str,
) -> dict[str, Any]:
    _require(selected_arm in {"f2", "f3"}, "SetFlow confirmation arm is not F2/F3")
    required = set(SETFLOW_CONFIRMATION_SEEDS_V3)
    _require(
        set(training) == set(validation) == required,
        "SetFlow confirmation requires exactly the three frozen seeds",
    )
    seed_results = {}
    for seed in SETFLOW_CONFIRMATION_SEEDS_V3:
        trained = training[seed]
        valid = validation[seed]
        _require(
            trained.get("status") == "XEDITSETFLOW_V3_GPU_TRAINING_COMPLETE",
            f"SetFlow confirmation training is not terminal: {seed}",
        )
        _require(
            valid.get("status") in {"FLOW_G0_READY", "FLOW_G0_VALIDATION_FAIL"},
            f"SetFlow confirmation validation is not terminal: {seed}",
        )
        _require(
            str(trained.get("arm")) == str(valid.get("arm")) == selected_arm,
            f"SetFlow confirmation arm differs: {seed}",
        )
        _require(
            trained.get("run_stage") == "CONFIRMATION"
            and trained.get("selectable") is True,
            f"SetFlow confirmation role differs: {seed}",
        )
        _require(
            int(trained.get("seed", -1)) == int(valid.get("seed", -2)) == seed,
            f"SetFlow confirmation seed differs: {seed}",
        )
        _require(
            trained.get("development_test_outcomes_accessed") is False
            and valid.get("development_test_outcomes_accessed") is False
            and trained.get("evaluation_outcomes_accessed") is False
            and valid.get("evaluation_outcomes_accessed") is False,
            f"SetFlow confirmation accessed protected outcome: {seed}",
        )
        _require(
            int(trained.get("train_record_count", -1)) == 68294
            and int(trained.get("validation_record_count", -1)) == 15924
            and int(trained.get("states_per_record_per_pass", -1)) == 2
            and int(trained.get("effective_batch_size", -1)) == 32
            and int(trained.get("maximum_passes", -1)) == 12
            and trained.get("training_precision") == "BF16"
            and trained.get("parameter_changed") is True
            and trained.get("cpu_fallback_used") is False
            and trained.get("critic_score_used") is False
            and trained.get("independent_evaluator_used") is False,
            f"SetFlow confirmation training provenance differs: {seed}",
        )
        _require(
            int(trained.get("trainable_parameter_count", -1))
            == SELECTABLE_TRAINABLE_PARAMETER_COUNTS_V3[selected_arm],
            f"SetFlow confirmation trainable parameter count differs: {seed}",
        )
        _require_unguided_validation_identity_v3(
            valid, arm=selected_arm, seed=seed
        )
        recovery = _finite(
            valid.get("source_macro_candidate_recovery_rate"), f"seed {seed} recovery"
        )
        top_k = _finite(
            valid.get("source_macro_measured_top_k_recovery_at_k"), f"seed {seed} top-k"
        )
        unique = _finite(
            valid.get("source_macro_unique_candidate_rate"), f"seed {seed} unique"
        )
        checks = {
            "source_macro_recovery_at_least_0_25": recovery >= 0.25,
            "source_macro_top_k_recovery_at_least_0_15": top_k >= 0.15,
            "source_macro_unique_rate_at_least_0_90": unique >= 0.90,
            "hard_legality_100pct": _finite(
                valid.get("hard_legality_rate"), f"seed {seed} legality"
            )
            == 1.0,
            "edit_budget_violation_zero": int(
                valid.get("edit_budget_violation_count", -1)
            )
            == 0,
            "candidate_budget_violation_zero": int(
                valid.get("candidate_budget_violation_count", -1)
            )
            == 0,
            "trajectory_replay_failure_zero": int(
                valid.get("trajectory_replay_failure_count", -1)
            )
            == 0,
            "numerical_failure_zero": int(valid.get("numerical_failure_count", -1))
            == 0,
            "protected_outcome_reads_zero": True,
        }
        seed_results[str(seed)] = {
            "source_macro_candidate_recovery_rate": recovery,
            "source_macro_measured_top_k_recovery_at_k": top_k,
            "source_macro_unique_candidate_rate": unique,
            "checks": checks,
            "passed": all(checks.values()),
        }
    passed = all(row["passed"] for row in seed_results.values())
    return {
        "schema_version": "route_a_v3_route2_xeditsetflow_confirmation_gate.v1",
        "status": (
            "XEDITSETFLOW_V3_CONFIRMATION_PASS"
            if passed
            else "XEDITSETFLOW_V3_CONFIRMATION_NO_GO"
        ),
        "selected_arm": selected_arm,
        "required_seeds": list(SETFLOW_CONFIRMATION_SEEDS_V3),
        "seed_results": seed_results,
        "flow_status": "FLOW_G0_READY" if passed else "FLOW_G0_NOT_READY",
        "guidance_authorized": False,
        "development_test_authorized": False,
        "new_final_evaluation_authorized": False,
        "additional_seed_authorized": False,
    }
