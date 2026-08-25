from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/route_a_v3_route2_xeditcritic_v4_screen_v1.json"


def _load() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_v4_screen_has_exactly_one_selectable_full_and_all_required_controls() -> None:
    config = _load()
    runs = config["required_screen_runs"]
    assert [run["run_id"] for run in runs] == [
        "c0_v4",
        "v4_full",
        "v4_source_only",
        "v4_edit_metadata_only",
        "v4_no_candidate_sequence",
        "v4_candidate_bundle_permutation",
        "v4_no_cross",
        "v4_no_moe",
    ]
    assert [run["run_id"] for run in runs if run["selectable"]] == ["v4_full"]
    assert {run["mechanism"] for run in runs} >= {"FULL", "NO_CROSS", "NO_MOE", "RAW_BASELINE"}


def test_v4_screen_freezes_exact_seed_updates_losses_and_checkpoint() -> None:
    config = _load()
    geometry = config["data_geometry"]
    training = config["training"]
    assert training["screen_seed"] == 20260907
    assert geometry["effective_batch_size"] == 32
    assert geometry["updates_per_pass"] == 2802
    assert geometry["pass_count"] == 8
    assert geometry["total_optimizer_updates"] == 22416
    assert training["pass_1_2_loss"] == {
        "huber": 1.0,
        "pairwise": 0.25,
        "soft_spearman": 0.0,
        "router_balance": 0.0,
    }
    assert training["pass_3_8_loss"] == {
        "huber": 1.0,
        "pairwise": 0.5,
        "soft_spearman": 0.25,
        "router_balance": 0.01,
    }
    assert training["checkpoint_selection"].startswith("FINAL_PASS_8_FIXED")


def test_v4_screen_capacity_and_memory_preflight_cannot_silently_shrink() -> None:
    config = _load()
    architecture = config["architecture"]
    memory = config["memory_preflight"]
    assert architecture["local_geometry_proxy_trainable_parameter_count"] == 170_481_733
    assert architecture["semantic_expert_bank_scope"] == (
        "ONE_SHARED_FOUR_EXPERT_BANK_REUSED_ACROSS_ALL_12_BLOCKS"
    )
    assert (
        architecture["design_target_minimum_trainable_parameter_count"],
        architecture["design_target_maximum_trainable_parameter_count"],
    ) == (165_000_000, 175_000_000)
    assert memory["physical_batch_candidates"] == [4, 8, 16, 32]
    assert memory["minimum_physical_batch"] == 4
    assert (memory["minimum_peak_allocated_gib"], memory["maximum_peak_allocated_gib"]) == (20.0, 35.0)
    assert memory["measurement"] == "TORCH_CUDA_MAX_MEMORY_ALLOCATED"
    assert memory["cpu_fallback"] is False
    assert memory["artificial_padding_or_unused_tensor"] is False


def test_v4_cache_online_alignment_cohort_and_tolerances_are_frozen() -> None:
    alignment = _load()["cache_online_alignment"]
    assert alignment == {
        "sequence_count": 8,
        "selection": "LENGTH_SORTED_EVEN_QUANTILES_OVER_LEXICOGRAPHIC_CACHE_SEQUENCE_INDICES",
        "maximum_absolute_tolerance": 0.02,
        "mean_absolute_tolerance": 0.005,
        "maximum_sequences_per_batch": 8,
        "batch_token_budget": 4096,
        "attention_backend": "PYTORCH_SDPA_AUTO",
        "raw_sequence_payload_written": 0,
        "target_value_accessed": False,
        "validation_metric_read": False,
    }


def test_v4_screen_launch_is_blocked_until_c3_cache_tests_and_preflight_finish() -> None:
    barrier = _load()["launch_barrier"]
    assert len(barrier["c3_required_run_ids"]) == 5
    assert barrier["all_c3_jobs_must_be_terminal"] is True
    assert barrier["c3_terminal_summaries_must_be_read_exactly_once"] is True
    assert barrier["a100_current_head_sync_and_tests_must_pass"] is True
    assert barrier["bottom_six_cache_must_be_terminal_complete"] is True
    assert barrier["formal_parameter_and_memory_preflight_must_pass"] is True
    assert barrier["c3_can_authorize_v4_downstream"] is False


def test_v4_screen_keeps_test_evaluation_and_extra_seed_closed() -> None:
    config = _load()
    assert config["status"].startswith("FROZEN_BEFORE_V4_")
    assert config["development_test_outcomes_accessed"] is False
    assert config["new_final_evaluation_outcomes_accessed"] is False
    assert config["additional_screen_seed_authorized"] is False
    assert config["c3_reference"]["role"].endswith("NEVER_TEST_AUTHORIZATION")
    assert config["screen_gate"]["maximum_development_test_outcome_reads"] == 0
    assert config["screen_gate"]["maximum_new_evaluation_outcome_reads"] == 0
