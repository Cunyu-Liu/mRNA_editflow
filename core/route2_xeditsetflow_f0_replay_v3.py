"""Frozen Base Flow V2 provenance boundary for SetFlow V3 common-NLL replay."""

from __future__ import annotations

from typing import Any, Mapping


class FrozenF0ReplayV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FrozenF0ReplayV3Error(message)


def validate_frozen_f0_for_common_replay_v3(
    checkpoint: Mapping[str, Any], training_summary: Mapping[str, Any]
) -> dict[str, Any]:
    provenance = checkpoint.get("training_provenance") or {}
    _require(int(checkpoint.get("completed_epoch", -1)) == 1, "F0 replay checkpoint is not selected epoch 1")
    _require(int(training_summary.get("selected_epoch", -1)) == 1, "F0 terminal selection changed")
    _require(int(training_summary.get("trainable_parameter_count", -1)) == 817_957, "F0 terminal capacity changed")
    _require(training_summary.get("status") == "LEARNED_BASE_FLOW_GPU_UPDATE_COMPLETE", "F0 training is not terminal")
    _require(provenance.get("parameter_changed") is True, "F0 checkpoint has no learned update")
    _require(provenance.get("cpu_fallback_used") is False, "F0 checkpoint used CPU fallback")
    _require(int(provenance.get("optimizer_steps", 0)) > 0, "F0 checkpoint has no optimizer steps")
    model_config = checkpoint.get("model_config") or {}
    _require(
        int(model_config.get("hidden_dim", -1)) == 256
        and model_config.get("position_progress_features") is True,
        "F0 checkpoint architecture changed",
    )
    return {
        "selected_epoch": 1,
        "trainable_parameter_count": 817_957,
        "historical_best_validation_next_action_nll": float(training_summary["best_validation_nll"]),
        "optimizer_steps_at_selected_checkpoint": int(provenance["optimizer_steps"]),
    }
