from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.route2_xeditcritic_ledger_v3 import (
    critic_v3_attempt_config,
    critic_v3_attempt_details,
    critic_v3_ledger_paths,
    critic_v3_seed_and_stage,
    require_critic_v3_confirmation_authorization,
    require_critic_v3_posttest_authorization,
)


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


def test_confirmation_attempt_uses_exact_seed_and_stage_identity() -> None:
    raw = {**_config(), "run_stage": "CONFIRMATION", "seed": 20260901}
    config = critic_v3_attempt_config(
        raw, run_id="c3", arm="C3", control_mode="NONE",
        candidate_bundle_permutation=False, physical_gpu_index=4,
    )
    assert critic_v3_seed_and_stage(raw) == (20260901, "CONFIRMATION")
    assert config["attempt_id"] == "xeditcritic_v3_confirmation_seed20260901::c3"
    assert config["attempt_purpose"] == "XEDITCRITIC_V3_CONFIRMATION"
    assert config["run_mode"] == "FROZEN_CONFIRMATION"
    assert config["seed"] == 20260901
    with pytest.raises(ValueError, match="undeclared"):
        critic_v3_seed_and_stage({**raw, "seed": 20260903})


def test_confirmation_requires_terminal_screen_selected_arm(tmp_path: Path) -> None:
    gate_path = tmp_path / "screen_gate.json"
    gate_path.write_text(json.dumps({
        "status": "XEDITCRITIC_V3_SCREEN_PASS",
        "confirmation_authorized": True,
        "selected_arm": "C3",
    }))
    config = {
        **_config(), "run_stage": "CONFIRMATION", "seed": 20260831,
        "screen_gate_path": str(gate_path), "selected_arm": "C3",
    }
    require_critic_v3_confirmation_authorization(config, arm="C0")
    require_critic_v3_confirmation_authorization(config, arm="C3")
    with pytest.raises(ValueError, match="not selected"):
        require_critic_v3_confirmation_authorization(config, arm="C2")
    gate_path.write_text(json.dumps({"status": "XEDITCRITIC_V3_SCREEN_NO_GO"}))
    with pytest.raises(ValueError, match="does not authorize"):
        require_critic_v3_confirmation_authorization(config, arm="C3")


def test_refit_and_loso_require_frozen_test_and_three_refits(tmp_path: Path) -> None:
    three = tmp_path / "three.json"
    atomic = tmp_path / "atomic.json"
    refit = tmp_path / "refit.json"
    three.write_text(json.dumps({
        "status": "XEDITCRITIC_V3_THREE_SEED_PASS", "selected_arm": "C2"
    }))
    atomic.write_text(json.dumps({
        "status": "ATOMIC_FROZEN_DEVELOPMENT_TEST_TERMINAL",
        "frozen_test_gate": {
            "status": "XEDITCRITIC_V3_FROZEN_TEST_PASS",
            "all_development_refit_authorized": True,
        },
    }))
    refit.write_text(json.dumps({
        "status": "XEDITCRITIC_V3_ALL_DEVELOPMENT_REFIT_COMPLETE",
        "required_seeds": [20260831, 20260901, 20260902],
        "completed_refit_count": 3,
    }))
    base = {
        **_config(),
        "seed": 20260831,
        "selected_arm": "C2",
        "three_seed_gate_path": str(three),
        "atomic_frozen_test_path": str(atomic),
    }
    require_critic_v3_posttest_authorization(
        {**base, "run_stage": "REFIT"}, arm="C2"
    )
    with pytest.raises(ValueError, match="only authorizes"):
        require_critic_v3_posttest_authorization(
            {**base, "run_stage": "REFIT"}, arm="C0"
        )
    loso = {
        **base,
        "run_stage": "LOSO",
        "held_out_study": "GSE269595",
        "refit_manifest_path": str(refit),
    }
    require_critic_v3_posttest_authorization(loso, arm="C2")
    require_critic_v3_posttest_authorization(loso, arm="C0")
    loso["held_out_study"] = "GSE_UNKNOWN"
    with pytest.raises(ValueError, match="undeclared"):
        require_critic_v3_posttest_authorization(loso, arm="C2")
