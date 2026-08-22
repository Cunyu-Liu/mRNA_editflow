from __future__ import annotations

import pytest

from core.route2_xeditsetflow_f0_replay_v3 import FrozenF0ReplayV3Error, validate_frozen_f0_for_common_replay_v3


def _inputs():
    checkpoint = {
        "completed_epoch": 1,
        "model_config": {"hidden_dim": 256, "position_progress_features": True},
        "training_provenance": {"parameter_changed": True, "cpu_fallback_used": False, "optimizer_steps": 1068},
    }
    summary = {
        "selected_epoch": 1,
        "trainable_parameter_count": 817_957,
        "status": "LEARNED_BASE_FLOW_GPU_UPDATE_COMPLETE",
        "best_validation_nll": 5.512483521877043,
    }
    return checkpoint, summary


def test_only_terminal_epoch_one_f0_is_accepted_for_read_only_replay() -> None:
    checkpoint, summary = _inputs()
    result = validate_frozen_f0_for_common_replay_v3(checkpoint, summary)
    assert result["selected_epoch"] == 1
    assert result["trainable_parameter_count"] == 817_957


def test_replay_rejects_a_retrained_or_reselected_f0() -> None:
    checkpoint, summary = _inputs()
    checkpoint["completed_epoch"] = 2
    with pytest.raises(FrozenF0ReplayV3Error, match="selected epoch 1"):
        validate_frozen_f0_for_common_replay_v3(checkpoint, summary)
