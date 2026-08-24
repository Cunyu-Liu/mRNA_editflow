from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.route_a_v3.train_route2_xeditcritic_v4 import (
    XEditCriticTrainingV4RunnerError,
    critic_v4_run_stage_seed,
    evaluation_index_batches_v4,
    require_confirmation_launch_authorization_v4,
    require_screen_launch_authorization_v4,
    screen_run_spec_v4,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/route_a_v3_route2_xeditcritic_v4_screen_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _authorization(config: dict) -> dict:
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_screen_launch_authorization.v1",
        "status": "XEDITCRITIC_V4_SCREEN_LAUNCH_AUTHORIZED",
        "authorized_git_head": "head",
        "authorized_run_ids": [
            row["run_id"] for row in config["required_screen_runs"]
        ],
        "barriers": {
            "all_five_c3_jobs_terminal": True,
            "c3_terminal_summaries_read_exactly_once": True,
            "a100_current_head_focused_tests_passed": True,
            "a100_current_head_v332_tests_passed": True,
            "bottom_six_cache_terminal_complete": True,
            "formal_parameter_preflight_passed": True,
            "formal_memory_preflight_passed": True,
        },
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def _preflight() -> dict:
    return {
        "status": "XEDITCRITIC_V4_PREFLIGHT_PASS",
        "passed": True,
        "selected_physical_batch": 8,
        "trainable_parameter_count": 173_692_549,
        "selected_peak_allocated_gib": 29.0,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def test_screen_run_specs_bind_all_exact_frozen_controls_and_ablations() -> None:
    config = _config()
    permutation = screen_run_spec_v4(config, "v4_candidate_bundle_permutation")
    assert permutation.control_mode == "CANDIDATE_BUNDLE_PERMUTATION"
    assert permutation.candidate_bundle_permutation is True
    assert permutation.selectable is False
    assert screen_run_spec_v4(config, "v4_no_cross").mechanism_mode == "NO_CROSS"
    assert screen_run_spec_v4(config, "v4_full").selectable is True
    assert screen_run_spec_v4(config, "c0_v4").model_kind == "C0-V4"
    with pytest.raises(XEditCriticTrainingV4RunnerError, match="exact frozen"):
        screen_run_spec_v4(config, "v4_unregistered")


def test_validation_batch_padding_does_not_add_measured_rows() -> None:
    batches = evaluation_index_batches_v4(18, 8)
    assert [valid for _, valid in batches] == [8, 8, 2]
    assert all(len(indices) == 8 for indices, _ in batches)
    assert sum(valid for _, valid in batches) == 18
    assert batches[-1][0][:2] == [16, 17]
    assert batches[-1][0][2:] == [0, 1, 2, 3, 4, 5]


def test_launch_authorization_requires_every_c3_sync_cache_and_preflight_barrier() -> None:
    config = _config()
    authorization = _authorization(config)
    require_screen_launch_authorization_v4(
        config,
        authorization,
        _preflight(),
        run_id="v4_full",
        physical_batch_size=8,
        current_git_head="head",
    )
    authorization["barriers"]["c3_terminal_summaries_read_exactly_once"] = False
    with pytest.raises(XEditCriticTrainingV4RunnerError, match="barrier"):
        require_screen_launch_authorization_v4(
            config,
            authorization,
            _preflight(),
            run_id="v4_full",
            physical_batch_size=8,
            current_git_head="head",
        )


def test_launch_authorization_rejects_head_batch_memory_parameter_or_protected_read_drift() -> None:
    config = _config()
    authorization = _authorization(config)
    with pytest.raises(XEditCriticTrainingV4RunnerError, match="another Git HEAD"):
        require_screen_launch_authorization_v4(
            config,
            authorization,
            _preflight(),
            run_id="v4_full",
            physical_batch_size=8,
            current_git_head="different",
        )
    preflight = _preflight()
    preflight["selected_peak_allocated_gib"] = 19.9
    with pytest.raises(XEditCriticTrainingV4RunnerError, match="20–35"):
        require_screen_launch_authorization_v4(
            config,
            authorization,
            preflight,
            run_id="v4_full",
            physical_batch_size=8,
            current_git_head="head",
        )
    preflight = _preflight()
    preflight["development_test_outcome_reads"] = 1
    with pytest.raises(XEditCriticTrainingV4RunnerError, match="TEST read"):
        require_screen_launch_authorization_v4(
            config,
            authorization,
            preflight,
            run_id="v4_full",
            physical_batch_size=8,
            current_git_head="head",
        )


def test_confirmation_authorization_is_exact_three_seed_full_c0_scope() -> None:
    config = _config()
    config.update(
        {
            "schema_version": "route_a_v3_route2_xeditcritic_v4_confirmation_runtime.v1",
            "run_stage": "CONFIRMATION",
            "training_seed": 20260908,
            "required_confirmation_run_ids": ["v4_full", "c0_v4"],
        }
    )
    authorization = {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_confirmation_launch_authorization.v1",
        "status": "XEDITCRITIC_V4_CONFIRMATION_LAUNCH_AUTHORIZED",
        "authorized_git_head": "head",
        "authorized_seeds": [20260908, 20260909, 20260910],
        "authorized_run_ids": ["v4_full", "c0_v4"],
        "barriers": {
            "screen_gate_passed": True,
            "a100_current_head_focused_tests_passed": True,
            "a100_current_head_v332_tests_passed": True,
            "bottom_six_cache_terminal_complete": True,
            "formal_parameter_preflight_passed": True,
            "formal_memory_preflight_passed": True,
        },
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    screen_gate = {
        "status": "XEDITCRITIC_V4_SCREEN_PASS",
        "passed": True,
        "confirmation_authorized": True,
        "development_test_authorized": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    assert critic_v4_run_stage_seed(config, "v4_full") == (
        "CONFIRMATION",
        20260908,
    )
    require_confirmation_launch_authorization_v4(
        config,
        authorization,
        _preflight(),
        screen_gate,
        run_id="v4_full",
        physical_batch_size=8,
        current_git_head="head",
    )
    with pytest.raises(XEditCriticTrainingV4RunnerError, match="scope"):
        critic_v4_run_stage_seed(config, "v4_no_moe")
    preflight = _preflight()
    preflight["development_test_outcome_reads"] = 1
    with pytest.raises(XEditCriticTrainingV4RunnerError, match="protected read"):
        require_confirmation_launch_authorization_v4(
            config,
            authorization,
            preflight,
            screen_gate,
            run_id="v4_full",
            physical_batch_size=8,
            current_git_head="head",
        )
