"""Unit tests for the SetFlow V5 successor runtime (CPU-only)."""

from __future__ import annotations

import math

import pytest

from core.route2_xeditsetflow_runtime_v5 import (
    gate_b0_convergence_judgment,
    screen_run_spec_v5,
    setflow_v5_learning_rate_factor,
)


def _config(runs=None):
    return {
        "required_screen_runs": runs or [
            {
                "run_id": "b_fix1",
                "mode_count": 1,
                "mode_information_weight": 0.0,
                "coverage_weight": 0.10,
                "selectable": True,
                "architecture_profile": "V4_FULL",
            },
            {
                "run_id": "b_arch1",
                "mode_count": 1,
                "mode_information_weight": 0.0,
                "coverage_weight": 0.0,
                "selectable": True,
                "architecture_profile": "A1",
            },
        ]
    }


def test_screen_run_spec_v5_resolves_config_arms():
    config = _config()
    spec = screen_run_spec_v5(config, "b_fix1")
    assert spec.run_id == "b_fix1"
    assert spec.mode_count == 1
    assert spec.coverage_weight == pytest.approx(0.10)
    assert spec.architecture_profile == "V4_FULL"
    spec2 = screen_run_spec_v5(config, "b_arch1")
    assert spec2.architecture_profile == "A1"


def test_screen_run_spec_v5_unknown_run_rejected():
    with pytest.raises(RuntimeError):
        screen_run_spec_v5(_config(), "not_an_arm")


def test_screen_run_spec_v5_duplicate_run_rejected():
    config = _config(runs=[
        {"run_id": "dup", "mode_count": 1, "mode_information_weight": 0.0,
         "coverage_weight": 0.0, "selectable": True, "architecture_profile": "A1"},
        {"run_id": "dup", "mode_count": 1, "mode_information_weight": 0.0,
         "coverage_weight": 0.0, "selectable": True, "architecture_profile": "A1"},
    ])
    with pytest.raises(RuntimeError):
        screen_run_spec_v5(config, "dup")


def test_gate_b0_converged_true():
    result = gate_b0_convergence_judgment(
        [
            {"mean_train_total_loss": 0.50},
            {"mean_train_total_loss": 0.40},
            {"mean_train_total_loss": 0.39},
        ],
        window=2,
        tolerance=0.05,
    )
    assert result["converged"] is True
    assert result["relative_drop_over_window"] < 0.05


def test_gate_b0_not_converged():
    result = gate_b0_convergence_judgment(
        [
            {"mean_train_total_loss": 2.0},
            {"mean_train_total_loss": 1.8},
            {"mean_train_total_loss": 1.0},
        ],
        window=2,
        tolerance=0.05,
    )
    assert result["converged"] is False


def test_learning_rate_factor_warmup_and_cosine():
    total = 100
    first = setflow_v5_learning_rate_factor(0, total_updates=total, warmup_fraction=0.1)
    assert first == pytest.approx(1.0 / 10.0)
    mid = setflow_v5_learning_rate_factor(55, total_updates=total, warmup_fraction=0.1)
    assert 0.1 <= mid <= 1.0
    late = setflow_v5_learning_rate_factor(99, total_updates=total, warmup_fraction=0.1)
    assert late == pytest.approx(0.1, abs=1e-6)


def test_temperature_control_identity():
    from core.route2_xeditsetflow_temperature_control_v5 import temper_mode_prior_v5

    prior = (0.6, 0.4)
    tempered = temper_mode_prior_v5(prior, temperature=1.0)
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(tempered, prior))


def test_architecture_parameter_bands():
    import torch

    from core.route2_xeditsetflow_runtime_v5 import (
        build_setflow_screen_model_v5,
        require_setflow_v5_trainable_parameter_range,
        screen_run_spec_v5,
    )
    from core.route2_xeditsetflow_v4 import XEditSetFlowV4

    vocab = {
        "assay": {str(i): i for i in range(7)},
        "context": {str(i): i for i in range(28)},
        "quantity": {str(i): i for i in range(6)},
        "measurement": {str(i): i for i in range(5)},
        "numerator": {str(i): i for i in range(6)},
        "denominator": {str(i): i for i in range(6)},
    }
    config = {
        "required_screen_runs": [
            {
                "run_id": "b_arch1",
                "mode_count": 1,
                "mode_information_weight": 0.0,
                "coverage_weight": 0.10,
                "selectable": True,
                "architecture_profile": "A1",
            }
        ],
        "architecture": {
            "frozen_source_mrnabert_width": 768,
            "formal_endpoint_vocab_cardinalities": {
                "assay": 7, "context": 28, "quantity": 6,
                "measurement": 5, "numerator": 6, "denominator": 6,
            },
            "architecture_profiles": {
                "A1": {
                    "model_width": 384, "depth": 6, "attention_heads": 8,
                    "ffn_width": 1536, "local_attention_window": 64,
                    "mode_residual_rank": 32, "stop_bottleneck_width": 64,
                    "dropout": 0.15,
                }
            },
        },
    }
    model, capacity = build_setflow_screen_model_v5(config, vocab, run_id="b_arch1")
    assert 5_000_000 <= capacity["trainable_parameter_count"] <= 20_000_000
    # A1 band guard
    spec = screen_run_spec_v5(config, "b_arch1")
    capacity2 = require_setflow_v5_trainable_parameter_range(model, spec)
    assert capacity2["trainable_parameter_count"] == capacity["trainable_parameter_count"]
