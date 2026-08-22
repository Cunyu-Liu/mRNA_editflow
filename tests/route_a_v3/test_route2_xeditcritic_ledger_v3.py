from __future__ import annotations

from pathlib import Path

from core.route2_xeditcritic_ledger_v3 import critic_v3_attempt_config, critic_v3_attempt_details, critic_v3_ledger_paths


def _config():
    return {
        "screen_seed": 20260830, "passes": 8, "head_learning_rate": 3e-4,
        "edit_site_cache": "/mnt/cache.pt", "batch_size": 32, "weight_decay": 1e-4,
        "huber_delta": 1.0, "withheld_development_test_record_count": 18292,
        "experiment_ledger_path": "/mnt/attempts.csv",
    }


def test_attempt_identity_is_unique_per_exact_screen_artifact() -> None:
    config = critic_v3_attempt_config(
        _config(), run_id="c2_source_only", arm="C2", control_mode="SOURCE_ONLY",
        candidate_bundle_permutation=False, physical_gpu_index=3,
    )
    assert config["attempt_id"] == "xeditcritic_v3_screen_seed20260830::c2_source_only"
    assert config["candidate_control"] == "SOURCE_ONLY"
    assert config["device"] == "cuda:3"
    assert config["pretrained_feature_cache_path"] == "/mnt/cache.pt"


def test_attempt_details_keep_protected_counts_and_projection_counts() -> None:
    details = critic_v3_attempt_details(
        _config(), trainable_parameter_count=29_000_000,
        train_record_count=89580, validation_record_count=18293,
    )
    assert details["record_counts"] == {"TRAIN": 89580, "VALIDATION": 18293}
    assert details["development_test_record_count_withheld"] == 18292
    assert details["evaluation_record_count"] == 0
    assert critic_v3_ledger_paths(_config(), Path("/mnt/run")) == (
        Path("/mnt/attempts.csv"), Path("/mnt/run/training_attempt.json")
    )
