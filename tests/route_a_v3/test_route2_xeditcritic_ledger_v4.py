from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.route2_xeditcritic_ledger_v4 import (
    XEditCriticLedgerV4Error,
    critic_v4_attempt_config,
    critic_v4_attempt_details,
    critic_v4_ledger_paths,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/route_a_v3_route2_xeditcritic_v4_screen_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_v4_attempt_identity_and_candidate_control_match_exact_frozen_run() -> None:
    attempt = critic_v4_attempt_config(
        _config(),
        run_id="v4_candidate_bundle_permutation",
        physical_gpu_index=3,
        physical_batch_size=8,
    )
    assert attempt["attempt_id"] == "xeditcritic_v4_screen_seed20260907::v4_candidate_bundle_permutation"
    assert attempt["candidate_control"] == "CANDIDATE_BUNDLE_PERMUTATION"
    assert attempt["device"] == "cuda:3"
    assert attempt["batch_size"] == 32
    assert attempt["pretrained_feature_cache_path"].endswith("frozen_bottom_six_chunk_cache_v1.pt")


def test_v4_raw_baseline_has_same_information_but_no_pretrained_cache_identity() -> None:
    attempt = critic_v4_attempt_config(
        _config(),
        run_id="c0_v4",
        physical_gpu_index=0,
        physical_batch_size=32,
    )
    assert attempt["model_kind"] == "C0-V4"
    assert attempt["metadata_mode"] == "OUTCOME_FREE_ENDPOINT_DESCRIPTORS"
    assert attempt["pretrained_model_id"] == ""
    assert attempt["pretrained_feature_cache_path"] == ""
    assert attempt["critic_position_features"] == "RAW_FULL_CONTEXT"


def test_v4_attempt_rejects_unknown_run_gpu_or_batch() -> None:
    with pytest.raises(XEditCriticLedgerV4Error, match="exact frozen"):
        critic_v4_attempt_config(
            _config(), run_id="v4_extra", physical_gpu_index=0, physical_batch_size=4
        )
    with pytest.raises(XEditCriticLedgerV4Error, match="outside 0–5"):
        critic_v4_attempt_config(
            _config(), run_id="v4_full", physical_gpu_index=6, physical_batch_size=4
        )
    with pytest.raises(XEditCriticLedgerV4Error, match="undeclared"):
        critic_v4_attempt_config(
            _config(), run_id="v4_full", physical_gpu_index=0, physical_batch_size=2
        )


def test_v4_attempt_details_keep_protected_counts_and_zero_reads() -> None:
    details = critic_v4_attempt_details(
        _config(),
        trainable_parameter_count=173_692_549,
        physical_batch_size=8,
        peak_vram_mb=30000,
    )
    assert details["record_counts"] == {"TRAIN": 89580, "VALIDATION": 18293}
    assert details["development_test_record_count_withheld"] == 18292
    assert details["evaluation_record_count"] == 0
    assert details["protected_outcome_reads"] == {
        "development_test": 0,
        "new_final_evaluation": 0,
    }
    assert critic_v4_ledger_paths(_config(), Path("/mnt/run")) == (
        Path(_config()["experiment_ledger_path"]),
        Path("/mnt/run/training_attempt.json"),
    )


def test_confirmation_attempt_identity_is_seed_specific_and_full_c0_only() -> None:
    config = _config()
    config.update(
        {
            "run_stage": "CONFIRMATION",
            "training_seed": 20260909,
            "required_confirmation_run_ids": ["v4_full", "c0_v4"],
        }
    )
    result = critic_v4_attempt_config(
        config,
        run_id="v4_full",
        physical_gpu_index=3,
        physical_batch_size=8,
    )
    assert result["attempt_id"] == "xeditcritic_v4_confirmation_seed20260909::v4_full"
    assert result["attempt_purpose"] == "XEDITCRITIC_V4_CONFIRMATION"
    assert result["seed"] == 20260909
    with pytest.raises(XEditCriticLedgerV4Error, match="undeclared run"):
        critic_v4_attempt_config(
            config,
            run_id="v4_no_moe",
            physical_gpu_index=3,
            physical_batch_size=8,
        )


def test_posttest_attempt_identity_distinguishes_refit_and_loso_folds() -> None:
    config = _config()
    config.update(
        {
            "schema_version": "route_a_v3_route2_xeditcritic_v4_posttest_runtime.v1",
            "run_stage": "REFIT",
            "training_seed": 20260908,
            "required_posttest_run_ids": ["v4_full"],
            "held_out_study": None,
        }
    )
    refit = critic_v4_attempt_config(
        config, run_id="v4_full", physical_gpu_index=0, physical_batch_size=8
    )
    assert refit["attempt_id"] == "xeditcritic_v4_refit_seed20260908::v4_full"
    assert refit["result_stage"] == "ALL_DEVELOPMENT_REFIT"

    config.update(
        {
            "run_stage": "LOSO",
            "required_posttest_run_ids": ["v4_full", "c0_v4"],
            "held_out_study": "GSE269595",
        }
    )
    loso = critic_v4_attempt_config(
        config, run_id="c0_v4", physical_gpu_index=1, physical_batch_size=8
    )
    assert loso["attempt_id"] == (
        "xeditcritic_v4_loso_seed20260908_gse269595::c0_v4"
    )
    assert loso["result_stage"] == "DEVELOPMENT_TEST_PRESERVING_LOSO"
