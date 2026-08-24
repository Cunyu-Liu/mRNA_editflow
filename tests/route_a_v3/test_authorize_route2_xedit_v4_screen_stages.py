from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.route_a_v3.authorize_route2_xedit_v4_screen_stages import (
    build_preflight_authorization_v4,
    build_screen_launch_authorization_v4,
)


ROOT = Path(__file__).resolve().parents[2]
CRITIC_CONFIG = json.loads((ROOT / "configs/route_a_v3_route2_xeditcritic_v4_screen_v1.json").read_text())
FLOW_CONFIG = json.loads((ROOT / "configs/route_a_v3_route2_xeditsetflow_v4_screen_v1.json").read_text())
HEAD = "a" * 40


def _c3():
    return {
        "status": "C3_V4_REFERENCE_READ_ONCE_COMPLETE",
        "terminal_summaries_read_count": 5,
        "c3_terminal_artifacts_retained": True,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def _a100():
    return {
        "repository_sync": {
            "head_after": HEAD,
            "old_launch_jobs_active_before_sync": False,
            "shared_history_rewritten": False,
        },
        "a100_current_head_verification": {
            "critic_focused_failed": 0,
            "setflow_focused_failed": 0,
            "exact_v332_passed": 96,
            "exact_v332_failed": 0,
        },
        "protected_data": {
            "development_test_outcomes_accessed": False,
            "new_final_evaluation_outcomes_accessed": False,
        },
    }


def _critic_cache():
    return {
        "schema_version": "route_a_v3_route2_frozen_bottom_encoder_chunk_cache_summary.v4",
        "status": "XEDITCRITIC_V4_BOTTOM_SIX_CACHE_COMPLETE",
        "record_count": 107873,
        "raw_sequence_payload_written": 0,
        "label_or_outcome_payload_written": 0,
        "development_test_outcomes_accessed": False,
        "evaluation_outcomes_accessed": False,
    }


def _flow_cache():
    return {
        "schema_version": "route_a_v3_route2_setflow_source_token_cache_summary.v3",
        "status": "XEDITSETFLOW_V3_SOURCE_TOKEN_CACHE_COMPLETE",
        "raw_sequence_payload_written": 0,
        "outcome_value_access_count": 0,
        "development_test_outcomes_accessed": False,
        "evaluation_outcomes_accessed": False,
    }


def test_preflight_authorizations_supply_exact_runner_barriers() -> None:
    critic = build_preflight_authorization_v4(
        "critic", _c3(), _a100(), _critic_cache(), current_git_head=HEAD
    )
    assert critic["status"] == "XEDITCRITIC_V4_PREFLIGHT_AUTHORIZED"
    assert critic["barriers"]["bottom_six_cache_terminal_complete"] is True
    flow = build_preflight_authorization_v4(
        "setflow", _c3(), _a100(), _flow_cache(), current_git_head=HEAD
    )
    assert flow["status"] == "XEDITSETFLOW_V4_PREFLIGHT_AUTHORIZED"
    assert flow["barriers"]["source_token_cache_terminal_complete"] is True


def test_preflight_authorization_rejects_old_head_or_incomplete_c3() -> None:
    audit = _a100()
    audit["repository_sync"]["head_after"] = "b" * 40
    with pytest.raises(Exception, match="exact current HEAD"):
        build_preflight_authorization_v4(
            "critic", _c3(), audit, _critic_cache(), current_git_head=HEAD
        )
    c3 = _c3()
    c3["terminal_summaries_read_count"] = 4
    with pytest.raises(Exception, match="read-once"):
        build_preflight_authorization_v4(
            "setflow", c3, _a100(), _flow_cache(), current_git_head=HEAD
        )


def test_screen_authorizations_match_formal_preflight_and_run_package() -> None:
    critic_preflight = {
        "status": "XEDITCRITIC_V4_PREFLIGHT_PASS",
        "passed": True,
        "git_head": HEAD,
        "trainable_parameter_count": 170000000,
        "selected_peak_allocated_gib": 30.0,
        "selected_physical_batch": 8,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    critic = build_screen_launch_authorization_v4(
        "critic", CRITIC_CONFIG, _c3(), _a100(), _critic_cache(),
        critic_preflight, None, current_git_head=HEAD,
    )
    assert critic["status"] == "XEDITCRITIC_V4_SCREEN_LAUNCH_AUTHORIZED"
    assert set(critic["authorized_run_ids"]) == {
        row["run_id"] for row in CRITIC_CONFIG["required_screen_runs"]
    }

    flow_preflight = {
        "status": "XEDITSETFLOW_V4_PREFLIGHT_PASS",
        "passed": True,
        "git_head": HEAD,
        "full_trainable_parameter_count": FLOW_CONFIG["architecture"]["formal_full_trainable_parameter_count"],
        "single_mode_trainable_parameter_count": FLOW_CONFIG["architecture"]["formal_single_mode_trainable_parameter_count"],
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    data = {
        "status": "XEDITSETFLOW_V4_SOURCE_LEVEL_DATA_AUDIT_PASS",
        "validation_source_count": 891,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    flow = build_screen_launch_authorization_v4(
        "setflow", FLOW_CONFIG, _c3(), _a100(), _flow_cache(),
        flow_preflight, data, current_git_head=HEAD,
    )
    assert flow["status"] == "XEDITSETFLOW_V4_SCREEN_LAUNCH_AUTHORIZED"
    assert flow["barriers"]["source_level_data_audit_passed"] is True


def test_screen_authorization_rejects_preflight_or_source_data_drift() -> None:
    preflight = {
        "status": "XEDITCRITIC_V4_PREFLIGHT_PASS",
        "passed": True,
        "git_head": HEAD,
        "trainable_parameter_count": 119999999,
        "selected_peak_allocated_gib": 30.0,
        "selected_physical_batch": 8,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    with pytest.raises(Exception, match="parameter/memory"):
        build_screen_launch_authorization_v4(
            "critic", CRITIC_CONFIG, _c3(), _a100(), _critic_cache(),
            preflight, None, current_git_head=HEAD,
        )
