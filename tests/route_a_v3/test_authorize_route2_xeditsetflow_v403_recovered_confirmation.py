from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from core.route2_xeditsetflow_runtime_v4 import (
    require_setflow_v4_confirmation_launch_authorization,
)
from scripts.route_a_v3.authorize_route2_xeditsetflow_v403_recovered_confirmation import (
    CONFIRMATION_SEEDS,
    RUNNER_VERIFICATION_RECEIPT_PASS,
    RUNNER_VERIFICATION_RECEIPT_SCHEMA,
    SCREEN_EXPERIMENT_HEAD,
    TRAINING_HEAD,
    VALIDATION_HEAD,
    build_recovered_confirmation_authorization_v403,
    require_recovery_terminal_v403,
    require_runner_verification_receipt_v403,
    require_science_protocol_unchanged_v403,
)
from scripts.route_a_v3.prepare_route2_xeditsetflow_v4_confirmation_configs import (
    build_confirmation_configs_v4,
)
from tests.route_a_v3.test_route2_xeditsetflow_runtime_v4 import (
    _authorization,
    _config,
)


ROOT = Path(__file__).resolve().parents[2]
BASE_PROTOCOL = json.loads(
    (
        ROOT
        / "configs/route_a_v3_route2_xeditsetflow_v4_confirmation_protocol_v1.json"
    ).read_text(encoding="utf-8")
)
DERIVED_PROTOCOL = json.loads(
    (
        ROOT
        / "configs/route_a_v3_route2_xeditsetflow_v403_recovered_confirmation_protocol_v1.json"
    ).read_text(encoding="utf-8")
)


def _recovery_config() -> dict:
    config = _config()
    provenance = DERIVED_PROTOCOL["validation_recovery_provenance"]
    return {
        **config,
        "status": "VALIDATION_ONLY_RECOVERY_FROM_TERMINAL_V4_CHECKPOINTS",
        "validation_output_root": "/tmp/recovered-validation",
        "screen_gate_output_path": provenance["recovered_screen_gate_path"],
        "validation_recovery": {
            "training_git_head": TRAINING_HEAD,
            "validation_git_head": VALIDATION_HEAD,
            "original_technical_gate": provenance["original_technical_gate_path"],
            "parameter_updates": 0,
            "training_reused": True,
            "scientific_thresholds_changed": False,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    }


def _recovery_runtime(*, terminal: bool = True) -> dict:
    provenance = DERIVED_PROTOCOL["validation_recovery_provenance"]
    return {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v403_validation_recovery_runtime.v1"
        ),
        "status": (
            "XEDITSETFLOW_V403_VALIDATION_RECOVERY_AND_GATE_TERMINAL"
            if terminal
            else "XEDITSETFLOW_V403_VALIDATION_RECOVERY_RUNNING"
        ),
        "git_head": VALIDATION_HEAD,
        "source_screen_head": TRAINING_HEAD,
        "experiment_head": SCREEN_EXPERIMENT_HEAD,
        "setflow_adjudication": {
            "status": "TERMINAL_COMPLETE" if terminal else "PENDING",
            "gate_present": terminal,
            "gate_path": provenance["recovered_screen_gate_path"],
        },
        "validation_jobs": {
            f"job_{index}": {
                "status": "TERMINAL_COMPLETE" if terminal else "RUNNING",
                "terminal_artifact_kind": "SUMMARY" if terminal else None,
            }
            for index in range(8)
        },
        "critic_failure_payload_reads": 0,
        "active_performance_output_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def _recovered_gate(*, passed: bool = True) -> dict:
    return {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_screen_gate.v1",
        "status": (
            "XEDITSETFLOW_V4_SCREEN_PASS"
            if passed
            else "XEDITSETFLOW_V4_SCREEN_NO_GO"
        ),
        "confirmation_authorized": passed,
        "confirmation_seeds": list(CONFIRMATION_SEEDS) if passed else [],
        "selected_checkpoint_pass": 8 if passed else None,
        "additional_seed_authorized": False,
        "development_test_authorized": False,
        "guidance_authorized": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
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
        "test_adjudicate_route2_xeditsetflow_v4_confirmation.py",
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


def _runner_receipt(*, runner_head: str = "c" * 40) -> dict:
    return {
        "schema_version": RUNNER_VERIFICATION_RECEIPT_SCHEMA,
        "status": RUNNER_VERIFICATION_RECEIPT_PASS,
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


def _receipt_path(runner_head: str) -> str:
    return DERIVED_PROTOCOL["runner_outputs"][
        "runner_verification_receipt_template"
    ].format(runner_git_head=runner_head)


def _build(
    *, runner_head: str = "c" * 40, runner_receipt: dict | None = None
) -> dict:
    authorization, preflight, source_data = _authorization(head=TRAINING_HEAD)
    return build_recovered_confirmation_authorization_v403(
        BASE_PROTOCOL,
        DERIVED_PROTOCOL,
        _config(),
        authorization,
        preflight,
        source_data,
        _recovery_config(),
        _recovery_runtime(),
        _recovered_gate(),
        runner_receipt or _runner_receipt(runner_head=runner_head),
        current_runner_head=runner_head,
        runner_verification_receipt_path=_receipt_path(runner_head),
    )


def test_recovered_authorization_preserves_three_distinct_heads() -> None:
    result = _build()
    assert result["authorized_git_head"] == "c" * 40
    assert result["training_git_head"] == TRAINING_HEAD
    assert result["validation_git_head"] == VALIDATION_HEAD
    assert len({result["authorized_git_head"], TRAINING_HEAD, VALIDATION_HEAD}) == 3
    assert result["authorized_seeds"] == list(CONFIRMATION_SEEDS)
    assert result["recovery_parameter_update_count"] == 0
    assert result["scientific_thresholds_changed"] is False
    assert result["source_screen_head_test_evidence"] == {
        "source_screen_git_head": TRAINING_HEAD,
        "source_screen_head_focused_tests_passed": True,
        "source_screen_head_v332_tests_passed": True,
    }
    runner_evidence = result["runner_current_head_verification"]
    assert runner_evidence["runner_git_head"] == "c" * 40
    assert runner_evidence["receipt_path"] == _receipt_path("c" * 40)
    assert result["barriers"]["a100_current_head_focused_tests_passed"] is True
    assert result["barriers"]["a100_current_head_v332_tests_passed"] is True


def test_existing_prepare_entry_accepts_recovery_derived_inputs() -> None:
    configs = build_confirmation_configs_v4(
        _recovery_config(), DERIVED_PROTOCOL, _recovered_gate()
    )
    assert [config["training_seed"] for config in configs] == list(
        CONFIRMATION_SEEDS
    )
    assert all(
        config["screen_gate_path"]
        == DERIVED_PROTOCOL["validation_recovery_provenance"][
            "recovered_screen_gate_path"
        ]
        for config in configs
    )
    assert all(
        config["validation_recovery"]["training_git_head"] == TRAINING_HEAD
        and config["validation_recovery"]["validation_git_head"]
        == VALIDATION_HEAD
        and config["validation_recovery"]["parameter_updates"] == 0
        for config in configs
    )


def test_core_consumes_only_receipt_derived_current_head_barriers() -> None:
    runner_head = "c" * 40
    result = _build(runner_head=runner_head)
    config = build_confirmation_configs_v4(
        _recovery_config(), DERIVED_PROTOCOL, _recovered_gate()
    )[0]
    _, preflight, source_data = _authorization(head=TRAINING_HEAD)
    require_setflow_v4_confirmation_launch_authorization(
        config,
        result,
        preflight,
        source_data,
        _recovered_gate(),
        run_id="v4_full",
        current_git_head=runner_head,
    )


def test_recovered_authorization_accepts_complete_eight_group_runner_receipt() -> None:
    require_runner_verification_receipt_v403(
        _runner_receipt(), current_runner_head="c" * 40
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
def test_recovered_authorization_runner_receipt_coverage_fails_closed(
    case: str, error: str
) -> None:
    receipt = _runner_receipt()
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
        require_runner_verification_receipt_v403(
            receipt, current_runner_head="c" * 40
        )


def test_recovered_authorization_rejects_nonterminal_or_no_go_recovery() -> None:
    with pytest.raises(Exception, match="not exact dual-HEAD terminal"):
        require_recovery_terminal_v403(
            DERIVED_PROTOCOL,
            _recovery_runtime(terminal=False),
            _recovered_gate(),
        )
    with pytest.raises(Exception, match="does not authorize exact confirmation"):
        require_recovery_terminal_v403(
            DERIVED_PROTOCOL,
            _recovery_runtime(),
            _recovered_gate(passed=False),
        )


def test_recovered_authorization_rejects_science_or_protected_read_drift() -> None:
    changed_protocol = copy.deepcopy(DERIVED_PROTOCOL)
    changed_protocol["checkpoint_gate"]["minimum_source_macro_recovery"] = 0.34
    with pytest.raises(Exception, match="confirmation science changed"):
        require_science_protocol_unchanged_v403(
            BASE_PROTOCOL, changed_protocol
        )

    protected_gate = _recovered_gate()
    protected_gate["development_test_outcome_reads"] = 1
    with pytest.raises(Exception, match="protected outcome read"):
        require_recovery_terminal_v403(
            DERIVED_PROTOCOL, _recovery_runtime(), protected_gate
        )


def test_recovered_authorization_rejects_screen_authorization_at_wrong_head() -> None:
    authorization, preflight, source_data = _authorization(head="b" * 40)
    with pytest.raises(Exception, match="another Git HEAD"):
        build_recovered_confirmation_authorization_v403(
            BASE_PROTOCOL,
            DERIVED_PROTOCOL,
            _config(),
            authorization,
            preflight,
            source_data,
            _recovery_config(),
            _recovery_runtime(),
            _recovered_gate(),
            _runner_receipt(),
            current_runner_head="c" * 40,
            runner_verification_receipt_path=_receipt_path("c" * 40),
        )


def test_recovered_authorization_requires_exact_runner_receipt() -> None:
    wrong_head = _runner_receipt(runner_head="d" * 40)
    with pytest.raises(Exception, match="not exact-HEAD PASS"):
        _build(runner_receipt=wrong_head)

    failed = _runner_receipt()
    failed["focused_tests"] = {
        **failed["focused_tests"],
        "passed": False,
        "passed_count": 34,
        "failed_count": 1,
    }
    with pytest.raises(Exception, match="failed or incomplete focused tests"):
        _build(runner_receipt=failed)


def test_recovered_authorization_rejects_runner_receipt_protected_read() -> None:
    receipt = _runner_receipt()
    receipt["development_test_outcome_reads"] = 1
    with pytest.raises(Exception, match="protected outcome read"):
        _build(runner_receipt=receipt)
