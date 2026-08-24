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
            "remote_worktree_clean_after": True,
            "shared_history_rewritten": False,
        },
        "a100_current_head_verification": {
            "verified_git_head": HEAD,
            "critic_focused_total_passed": 100,
            "critic_focused_failed": 0,
            "setflow_focused_passed": 60,
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
        "unique_sequence_count": 43730,
        "embedding_width": 768,
        "model_id": "YYLY66/mRNABERT@a1eb7df25804d23f08646e1cb996b234d7208a40",
        "chunk_nucleotides": 1000,
        "chunk_overlap": 64,
        "local_context_radius": 32,
        "frozen_encoder_blocks": [0, 1, 2, 3, 4, 5],
        "trainable_encoder_blocks": [6, 7, 8, 9, 10, 11],
        "raw_sequence_payload_written": 0,
        "label_or_outcome_payload_written": 0,
        "development_test_outcomes_accessed": False,
        "evaluation_outcomes_accessed": False,
    }


def _flow_cache():
    return {
        "schema_version": "route_a_v3_route2_setflow_source_token_cache_summary.v3",
        "status": "XEDITSETFLOW_V3_SOURCE_TOKEN_CACHE_COMPLETE",
        "projection_record_count": 107873,
        "eligible_record_count": 84218,
        "unique_source_count": 19303,
        "unique_source_token_count": 2817781,
        "maximum_source_length": 837,
        "embedding_width": 768,
        "model_id": "YYLY66/mRNABERT@a1eb7df25804d23f08646e1cb996b234d7208a40",
        "raw_sequence_payload_written": 0,
        "outcome_value_access_count": 0,
        "development_test_outcomes_accessed": False,
        "evaluation_outcomes_accessed": False,
    }


def _critic_cache_receipt():
    return {
        "model_id": "YYLY66/mRNABERT@a1eb7df25804d23f08646e1cb996b234d7208a40",
        "record_count": 107873,
        "unique_sequence_count": 43730,
        "embedding_width": 768,
        "frozen_encoder_blocks": [0, 1, 2, 3, 4, 5],
        "trainable_encoder_blocks": [6, 7, 8, 9, 10, 11],
        "chunk_length": 1000,
        "chunk_overlap": 64,
        "local_context_radius": 32,
        "special_token_offset": 1,
    }


def _flow_cache_receipt():
    return {
        "model_id": "YYLY66/mRNABERT@a1eb7df25804d23f08646e1cb996b234d7208a40",
        "record_count": 84218,
        "unique_source_count": 19303,
        "token_count": 2817781,
        "maximum_source_length": 837,
        "embedding_width": 768,
        "tokenization_policy": "UTR_SINGLE_NUCLEOTIDE_SPACE_SEPARATED_DNA_ALPHABET_ONE_LEADING_SPECIAL",
        "chunk_policy": "ONE_COMPLETE_CHUNK_MAXIMUM_1000_NUCLEOTIDES",
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


def test_preflight_authorization_requires_clean_tested_current_head() -> None:
    no_tests = _a100()
    no_tests["a100_current_head_verification"]["critic_focused_total_passed"] = 0
    with pytest.raises(Exception, match="did not run and pass"):
        build_preflight_authorization_v4(
            "critic", _c3(), no_tests, _critic_cache(), current_git_head=HEAD
        )

    wrong_test_head = _a100()
    wrong_test_head["a100_current_head_verification"]["verified_git_head"] = "b" * 40
    with pytest.raises(Exception, match="did not run and pass"):
        build_preflight_authorization_v4(
            "critic", _c3(), wrong_test_head, _critic_cache(), current_git_head=HEAD
        )

    dirty = _a100()
    dirty["repository_sync"]["remote_worktree_clean_after"] = False
    with pytest.raises(Exception, match="not synchronized"):
        build_preflight_authorization_v4(
            "setflow", _c3(), dirty, _flow_cache(), current_git_head=HEAD
        )


def test_preflight_authorization_rejects_wrong_cache_identity() -> None:
    critic_cache = _critic_cache()
    critic_cache["local_context_radius"] = 16
    with pytest.raises(Exception, match="not terminal and isolated"):
        build_preflight_authorization_v4(
            "critic", _c3(), _a100(), critic_cache, current_git_head=HEAD
        )

    flow_cache = _flow_cache()
    flow_cache["model_id"] = "another-model"
    with pytest.raises(Exception, match="not terminal and isolated"):
        build_preflight_authorization_v4(
            "setflow", _c3(), _a100(), flow_cache, current_git_head=HEAD
        )


def test_screen_authorizations_match_formal_preflight_and_run_package() -> None:
    critic_preflight = {
        "status": "XEDITCRITIC_V4_PREFLIGHT_PASS",
        "passed": True,
        "git_head": HEAD,
        "trainable_parameter_count": 170000000,
        "selected_peak_allocated_gib": 30.0,
        "selected_physical_batch": 8,
        "bottom_six_cache_identity": _critic_cache_receipt(),
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
        "source_token_cache_identity": _flow_cache_receipt(),
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
        "bottom_six_cache_identity": _critic_cache_receipt(),
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    with pytest.raises(Exception, match="parameter/memory"):
        build_screen_launch_authorization_v4(
            "critic", CRITIC_CONFIG, _c3(), _a100(), _critic_cache(),
            preflight, None, current_git_head=HEAD,
        )


def test_screen_authorization_rejects_missing_tensor_cache_identity_receipt() -> None:
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
    with pytest.raises(Exception, match="cache identity receipt is absent"):
        build_screen_launch_authorization_v4(
            "critic", CRITIC_CONFIG, _c3(), _a100(), _critic_cache(),
            critic_preflight, None, current_git_head=HEAD,
        )

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
    with pytest.raises(Exception, match="cache identity receipt is absent"):
        build_screen_launch_authorization_v4(
            "setflow", FLOW_CONFIG, _c3(), _a100(), _flow_cache(),
            flow_preflight, data, current_git_head=HEAD,
        )
