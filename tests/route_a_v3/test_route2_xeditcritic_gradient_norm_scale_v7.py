from __future__ import annotations

import math

import torch

from core.route2_xeditcritic_gradient_norm_scale_v7 import (
    GradientNormScalerV7,
    XEditCriticV7GradientNormScaleError,
)


def _geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def test_high_norm_task_downweighted_low_norm_upweighted() -> None:
    scaler = GradientNormScalerV7(ema_alpha=0.5)
    multipliers = scaler.frozen_multipliers({"task-a": 10.0, "task-b": 0.1})
    # the higher-norm task receives a multiplier below 1 and vice versa
    assert multipliers["task-a"] < 1.0
    assert multipliers["task-b"] > 1.0


def test_multipliers_center_on_geometric_mean() -> None:
    scaler = GradientNormScalerV7(ema_alpha=0.5)
    scaler.ingest("task-a", 10.0)
    scaler.ingest("task-b", 1.0)
    scaler.ingest("task-c", 0.2)
    # geometric mean of all EMAs should equal the reference
    emas = list(scaler.ema_norms().values())
    assert math.isclose(scaler.reference(), _geometric_mean(emas), rel_tol=1e-12)
    # each multiplier centers the EMA onto the reference
    for task, ema in scaler.ema_norms().items():
        multiplier = scaler.ingest(task, ema)  # no-op EMA, returns current multiplier
        assert math.isclose(ema * multiplier, scaler.reference(), rel_tol=1e-12)


def test_scale_returns_scaled_gradient_multiplier_norm() -> None:
    scaler = GradientNormScalerV7(ema_alpha=0.5)
    gradient = torch.tensor([3.0, 4.0])  # norm = 5
    scaled, multiplier, norm = scaler.scale("task-a", gradient)
    assert math.isclose(norm, 5.0, rel_tol=1e-12)
    assert math.isclose(multiplier, scaler.reference() / 5.0, rel_tol=1e-12)
    assert torch.allclose(scaled, gradient * multiplier)


def test_ema_smooths_toward_steady_value() -> None:
    scaler = GradientNormScalerV7(ema_alpha=0.1)
    for _ in range(50):
        scaler.ingest("task-a", 2.0)
    scaler.ingest("task-b", 2.0)  # second task makes centering active
    # after many identical updates the EMA converges toward 2.0
    assert math.isclose(scaler.ema_norms()["task-a"], 2.0, rel_tol=1e-6)


def test_guard_rejects_nonfinite_and_nonpositive_norms() -> None:
    scaler = GradientNormScalerV7()
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        try:
            scaler.ingest("task-x", bad)
        except XEditCriticV7GradientNormScaleError:
            continue
        raise AssertionError(f"expected rejection for norm={bad}")


def test_frozen_multipliers_geometric_mean_is_one() -> None:
    scaler = GradientNormScalerV7(ema_alpha=0.5)
    multipliers = scaler.frozen_multipliers({"task-a": 10.0, "task-b": 1.0, "task-c": 0.2})
    product = _geometric_mean(list(multipliers.values()))
    assert math.isclose(product, 1.0, rel_tol=1e-12)


def test_reset_clears_state() -> None:
    scaler = GradientNormScalerV7()
    scaler.ingest("task-a", 1.0)
    assert scaler.reference() is not None
    scaler.reset()
    assert scaler.ema_norms() == {}
    assert scaler.reference() is None
