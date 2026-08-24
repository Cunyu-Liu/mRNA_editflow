from __future__ import annotations

import math

import torch
import pytest

from core.route2_xeditsetflow_v4 import (
    XEditSetFlowV4,
    common_set_marginal_loss_v4,
    mixture_setflow_loss_v4,
    require_setflow_v4_trainable_parameter_range,
    select_trajectory_mode_rates_v4,
)


def _model(*, modes: int = 4) -> XEditSetFlowV4:
    return XEditSetFlowV4(
        assay_count=2,
        context_count=2,
        quantity_count=2,
        measurement_count=2,
        numerator_count=2,
        denominator_count=2,
        pretrained_width=8,
        model_width=40,
        depth=2,
        heads=5,
        ffn_width=80,
        local_attention_window=4,
        mode_count=modes,
        mode_residual_rank=8,
        stop_bottleneck_width=8,
        dropout=0.0,
        activation_checkpointing=False,
    )


def _batch() -> dict[str, torch.Tensor]:
    source = torch.tensor([[0, 0, 0, 0, 4, 4], [0, 1, 2, 3, 0, 1]])
    current = source.clone()
    current[0, 1] = 2
    return {
        "source_tokens": source,
        "current_tokens": current,
        "padding_mask": source == 4,
        "source_pretrained_tokens": torch.randn(2, 6, 8),
        "remaining_budget": torch.tensor([2, 0]),
        "quantity_ids": torch.tensor([1, 1]),
        "measurement_ids": torch.tensor([1, 1]),
        "numerator_ids": torch.tensor([1, 1]),
        "denominator_ids": torch.tensor([1, 1]),
        "assay_ids": torch.tensor([1, 1]),
        "context_ids": torch.tensor([1, 1]),
        "region_ids": torch.tensor([0, 1]),
    }


def test_model_emits_smoothed_mode_prior_and_masks_illegal_rates_before_use() -> None:
    model = _model()
    output = model(_batch())
    assert output["mode_rates"].shape == (2, 4, 25)
    assert output["legal_action_mask"].shape == (2, 25)
    assert output["mode_prior"].shape == (2, 4)
    assert torch.allclose(output["mode_prior"].sum(dim=1), torch.ones(2))
    assert (output["mode_prior"] >= 0.125 - 1e-7).all()
    assert torch.equal(
        output["mode_rates"],
        torch.where(
            output["legal_action_mask"][:, None, :],
            output["mode_rates"],
            torch.zeros_like(output["mode_rates"]),
        ),
    )
    assert not output["legal_action_mask"][1].any()


def test_trajectory_mode_selection_uses_the_fixed_declared_mode() -> None:
    rates = torch.arange(2 * 4 * 5).reshape(2, 4, 5).float()
    selected = select_trajectory_mode_rates_v4(rates, torch.tensor([3, 1]))
    assert torch.equal(selected[0], rates[0, 3])
    assert torch.equal(selected[1], rates[1, 1])


def _loss_bundle(candidate_order=(0, 1)):
    rates = torch.tensor(
        [
            [
                [9.0, 1.0, 1.0],
                [1.0, 9.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        ]
    )
    candidate = torch.tensor(
        [
            [[True, False, False], [False, True, False]],
            [[False, False, False], [False, False, False]],
        ]
    )[:, candidate_order]
    batch = {
        "common_positive_action_mask": torch.tensor(
            [[True, True, False], [False, False, False]]
        ),
        "candidate_positive_action_mask": candidate,
        "candidate_valid_mask": torch.tensor(
            [[True, True], [True, True]]
        )[:, candidate_order],
        "remaining_count_soft_target": torch.tensor(
            [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        ),
        "structural_budget_exhausted": torch.tensor([False, True]),
    }
    output = {
        "mode_rates": rates,
        "legal_action_mask": torch.tensor(
            [[True, True, True], [False, False, False]]
        ),
        "mode_prior": torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
        "remaining_count_logits": torch.zeros(2, 6),
    }
    return output, batch


def test_loss_keeps_each_candidate_separate_and_is_candidate_order_invariant() -> None:
    output, batch = _loss_bundle((0, 1))
    output["mode_rates"] = output["mode_rates"].requires_grad_()
    differentiable_rates = output["mode_rates"]
    forward = mixture_setflow_loss_v4(output, batch)
    output, reversed_batch = _loss_bundle((1, 0))
    reverse = mixture_setflow_loss_v4(output, reversed_batch)
    assert torch.allclose(forward.total, reverse.total)
    assert forward.active_state_count == 1
    assert forward.active_candidate_constraint_count == 2
    assert torch.isfinite(forward.mode_information)
    forward.total.backward()
    assert differentiable_rates.grad is not None
    assert torch.isfinite(differentiable_rates.grad).all()


def test_single_mode_control_has_exact_zero_information_loss() -> None:
    output, batch = _loss_bundle((0, 1))
    output["mode_rates"] = output["mode_rates"][:, :1]
    output["mode_prior"] = torch.ones(2, 1)
    loss = mixture_setflow_loss_v4(output, batch)
    assert loss.mode_information.item() == 0.0


def test_smoothed_prior_and_mode_information_match_hand_calculation() -> None:
    model = _model(modes=2)
    with torch.no_grad():
        model.mode_router.weight.zero_()
        model.mode_router.bias.copy_(torch.tensor([math.log(3.0), 0.0]))
    prior = model(_batch())["mode_prior"]
    assert torch.allclose(
        prior,
        torch.tensor([[0.625, 0.375], [0.625, 0.375]]),
        atol=1e-7,
    )

    output, batch = _loss_bundle((0, 1))
    loss = mixture_setflow_loss_v4(output, batch)
    expected_mutual_information = 0.9 * math.log(1.8) + 0.1 * math.log(0.2)
    assert loss.mode_information.item() == pytest.approx(
        -expected_mutual_information,
        abs=1e-7,
    )


def test_common_nll_uses_mixture_mass_and_excludes_structural_state() -> None:
    output, batch = _loss_bundle((0, 1))
    result = common_set_marginal_loss_v4(
        output,
        batch["common_positive_action_mask"],
        batch["structural_budget_exhausted"],
        torch.tensor([2.0, 9.0]),
    )
    expected_mass = 10.0 / 11.0
    assert result.loss.item() == pytest.approx(-torch.log(torch.tensor(expected_mass)).item())
    assert result.active_weight.item() == 2.0
    assert result.active_state_count == 1


def test_formal_full_capacity_is_95_to_110m_and_single_mode_diff_is_below_two_percent() -> None:
    full = XEditSetFlowV4(
        assay_count=7,
        context_count=28,
        quantity_count=6,
        measurement_count=5,
        numerator_count=6,
        denominator_count=6,
    )
    single = XEditSetFlowV4(
        assay_count=7,
        context_count=28,
        quantity_count=6,
        measurement_count=5,
        numerator_count=6,
        denominator_count=6,
        mode_count=1,
    )
    capacity = require_setflow_v4_trainable_parameter_range(full)
    assert capacity["trainable_parameter_count"] == 100_099_998
    assert single.trainable_parameter_count == 98_628_717
    assert 95_000_000 <= capacity["trainable_parameter_count"] <= 110_000_000
    difference = full.trainable_parameter_count - single.trainable_parameter_count
    assert 0 < difference / full.trainable_parameter_count < 0.02
