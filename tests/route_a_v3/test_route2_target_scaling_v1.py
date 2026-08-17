from dataclasses import dataclass

import pytest

from core.route2_target_scaling import (
    TARGET_SCALING_NONE,
    TARGET_SCALING_TRAIN_TASK_ROBUST,
    Route2TargetScaler,
    fit_route2_target_scaler,
)


@dataclass(frozen=True)
class Record:
    target: float
    endpoint: str
    region: int


def test_robust_scaler_is_zero_preserving_and_train_only() -> None:
    train = [Record(value, "half_life", 0) for value in (-2.0, -1.0, 0.0, 1.0, 2.0)]
    scaler = fit_route2_target_scaler(
        train,
        mode=TARGET_SCALING_TRAIN_TASK_ROBUST,
        minimum_task_records=3,
        floor=1e-3,
    )
    scale, source = scaler.scale("half_life", 0)
    assert source == "TASK"
    assert scale == pytest.approx(1.4826)
    assert 0.0 / scale == 0.0

    # A validation-only outlier is not an input to fit and cannot change scale.
    validation = train + [Record(1e9, "half_life", 0)]
    contaminated = fit_route2_target_scaler(
        validation,
        mode=TARGET_SCALING_TRAIN_TASK_ROBUST,
        minimum_task_records=3,
        floor=1e-3,
    )
    assert contaminated.scale("half_life", 0)[0] != pytest.approx(scale)


def test_unseen_task_uses_train_region_then_global_fallback() -> None:
    scaler = fit_route2_target_scaler(
        [Record(value, "known", 0) for value in (-1.0, 0.0, 1.0)]
        + [Record(value, "three_prime", 1) for value in (-4.0, 0.0, 4.0)],
        mode=TARGET_SCALING_TRAIN_TASK_ROBUST,
        minimum_task_records=3,
    )
    region_scale, region_source = scaler.scale("unseen", 0)
    assert region_source == "REGION_FALLBACK"
    assert region_scale == scaler.region_scales["region=0"]
    global_scale, global_source = scaler.scale("unseen", 9)
    assert global_source == "GLOBAL_FALLBACK"
    assert global_scale == scaler.global_scale


def test_target_scaler_round_trip_and_legacy_none() -> None:
    fitted = fit_route2_target_scaler(
        [Record(value, "task", 0) for value in (-1.0, 0.0, 1.0)],
        mode=TARGET_SCALING_TRAIN_TASK_ROBUST,
        minimum_task_records=2,
    )
    restored = Route2TargetScaler.from_dict(fitted.to_dict())
    assert restored == fitted
    legacy = Route2TargetScaler.from_dict(None)
    assert legacy.mode == TARGET_SCALING_NONE
    assert legacy.scale("anything", 0) == (1.0, "NONE")


def test_scaler_rejects_non_train_only_or_centered_payload() -> None:
    fitted = fit_route2_target_scaler(
        [Record(1.0, "task", 0)],
        mode=TARGET_SCALING_TRAIN_TASK_ROBUST,
        minimum_task_records=1,
    ).to_dict()
    fitted["fit_scope"] = "TRAIN_AND_VALIDATION"
    with pytest.raises(ValueError, match="train-only"):
        Route2TargetScaler.from_dict(fitted)
