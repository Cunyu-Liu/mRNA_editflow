from __future__ import annotations

import copy

import pytest

from scripts.route_a_v3.authorize_route2_xeditsetflow_v4_confirmation import (
    build_confirmation_authorization_v4,
)
from tests.route_a_v3.test_route2_xeditsetflow_runtime_v4 import (
    _authorization,
    _config,
)


def _screen_gate() -> dict:
    return {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_screen_gate.v1",
        "status": "XEDITSETFLOW_V4_SCREEN_PASS",
        "confirmation_authorized": True,
        "confirmation_seeds": [20260912, 20260913, 20260914],
        "selected_checkpoint_pass": 8,
        "additional_seed_authorized": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def test_authorization_is_exact_full_only_three_seed_scope() -> None:
    screen_authorization, preflight, source_data = _authorization()
    result = build_confirmation_authorization_v4(
        _config(),
        screen_authorization,
        preflight,
        source_data,
        _screen_gate(),
        current_git_head="abc",
    )
    assert result["status"] == "XEDITSETFLOW_V4_CONFIRMATION_LAUNCH_AUTHORIZED"
    assert result["authorized_seeds"] == [20260912, 20260913, 20260914]
    assert result["authorized_run_id"] == "v4_full"
    assert result["development_test_authorized"] is False
    assert result["guidance_authorized"] is False


def test_authorization_rejects_no_go_or_protected_read() -> None:
    screen_authorization, preflight, source_data = _authorization()
    gate = _screen_gate()
    gate["status"] = "XEDITSETFLOW_V4_SCREEN_NO_GO"
    with pytest.raises(RuntimeError):
        build_confirmation_authorization_v4(
            _config(),
            screen_authorization,
            preflight,
            source_data,
            gate,
            current_git_head="abc",
        )
    authorization = copy.deepcopy(screen_authorization)
    authorization["development_test_outcome_reads"] = 1
    with pytest.raises(Exception):
        build_confirmation_authorization_v4(
            _config(),
            authorization,
            preflight,
            source_data,
            _screen_gate(),
            current_git_head="abc",
        )
