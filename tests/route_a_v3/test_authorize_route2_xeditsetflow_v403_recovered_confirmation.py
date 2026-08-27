from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.route_a_v3.authorize_route2_xeditsetflow_v403_recovered_confirmation import (
    CONFIRMATION_SEEDS,
    SCREEN_EXPERIMENT_HEAD,
    TRAINING_HEAD,
    VALIDATION_HEAD,
    build_recovered_confirmation_authorization_v403,
    require_recovery_terminal_v403,
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


def _build(*, runner_head: str = "c" * 40) -> dict:
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
        current_runner_head=runner_head,
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
            current_runner_head="c" * 40,
        )
