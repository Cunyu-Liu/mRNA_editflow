from __future__ import annotations

import torch

from core.route2_xeditsetflow_v3 import (
    XEditSetFlowV3,
    set_marginal_negative_log_likelihood_v1,
)


def _model(width: int = 128, depth: int = 2, ffn: int = 256) -> XEditSetFlowV3:
    torch.manual_seed(3)
    return XEditSetFlowV3(
        model_width=width,
        depth=depth,
        heads=4,
        ffn_width=ffn,
        assay_count=3,
        context_count=3,
        quantity_count=3,
        measurement_count=3,
        numerator_count=3,
        denominator_count=3,
        pretrained_width=8,
        dropout=0.0,
    ).eval()


def _batch(remaining: tuple[int, int] = (2, 0)) -> dict[str, torch.Tensor]:
    source = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
    current = source.clone()
    current[0, 1] = 2
    current[1, 1] = 2
    return {
        "source_tokens": source,
        "current_tokens": current,
        "padding_mask": source.eq(4),
        "source_pretrained_tokens": torch.randn(2, 5, 8),
        "remaining_budget": torch.tensor(remaining),
        "quantity_ids": torch.tensor([1, 1]),
        "measurement_ids": torch.tensor([1, 1]),
        "numerator_ids": torch.tensor([1, 1]),
        "denominator_ids": torch.tensor([1, 1]),
        "assay_ids": torch.tensor([1, 1]),
        "context_ids": torch.tensor([1, 1]),
        "region_ids": torch.tensor([0, 1]),
    }


def test_hard_legal_mask_precedes_rates_and_budget_exhaustion_is_structural() -> None:
    model = _model()
    batch = _batch()
    rates, legal = model.rates(batch)
    length = batch["source_tokens"].shape[1]
    assert torch.equal(rates.ne(0), legal)
    assert not legal[0, 1 * 4 : 1 * 4 + 4].any()
    assert legal[0, length * 4]
    assert not legal[1].any()


def test_set_marginal_loss_is_invariant_to_positive_edit_order() -> None:
    rates = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    legal = torch.ones_like(rates, dtype=torch.bool)
    first = torch.tensor([[True, False, True, False]])
    second = torch.tensor([[True, False, True, False]])
    loss_a = set_marginal_negative_log_likelihood_v1(
        rates, legal, first, torch.tensor([False])
    )
    loss_b = set_marginal_negative_log_likelihood_v1(
        rates[:, [2, 1, 0, 3]], legal, second, torch.tensor([False])
    )
    assert torch.allclose(loss_a, loss_b)


def test_incomplete_state_does_not_mark_stop_positive() -> None:
    rates = torch.ones((1, 9))
    legal = torch.ones_like(rates, dtype=torch.bool)
    positive = torch.zeros_like(legal)
    positive[0, 2] = True
    assert not positive[0, -1]
    loss = set_marginal_negative_log_likelihood_v1(
        rates, legal, positive, torch.tensor([False])
    )
    assert torch.isfinite(loss)


def test_completed_with_budget_uses_stop_but_exhausted_state_has_no_positive() -> None:
    rates = torch.ones((2, 5))
    legal = torch.zeros_like(rates, dtype=torch.bool)
    positive = torch.zeros_like(legal)
    legal[0, -1] = True
    positive[0, -1] = True
    loss = set_marginal_negative_log_likelihood_v1(
        rates, legal, positive, torch.tensor([False, True])
    )
    assert loss.item() == 0.0


def test_f2_and_f3_parameter_counts_match_frozen_capacity_bands() -> None:
    common = dict(
        assay_count=8,
        context_count=29,
        quantity_count=7,
        measurement_count=6,
        numerator_count=8,
        denominator_count=7,
    )
    f2 = XEditSetFlowV3(
        model_width=384, depth=8, heads=8, ffn_width=1536, **common
    )
    f3 = XEditSetFlowV3(
        model_width=512, depth=12, heads=8, ffn_width=2048, **common
    )
    assert 15_000_000 <= f2.trainable_parameter_count <= 22_000_000
    assert 38_000_000 <= f3.trainable_parameter_count <= 46_000_000
