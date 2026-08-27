from __future__ import annotations

import copy

import pytest
import torch

import core.route2_xeditsetflow_s1 as s1
from core.route2_xeditsetflow_s1 import (
    S1_RUN_ROLE_TO_V4,
    mixture_setflow_loss_s1,
    screen_run_spec_s1,
)
from core.route2_xeditsetflow_v4 import mixture_setflow_loss_v4


def _screen_config() -> dict:
    return {
        "required_screen_runs": [
            {
                "run_id": "v4_s1_full",
                "mode_count": 8,
                "mode_information_weight": 0.05,
                "selectable": True,
            },
            {
                "run_id": "v4_s1_single_mode",
                "mode_count": 1,
                "mode_information_weight": 0.0,
                "selectable": False,
            },
        ],
        "architecture": {"identity": "frozen-v4"},
    }


def test_s1_roles_are_distinct_while_model_builder_uses_scoped_v4_architecture_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _screen_config()
    before = copy.deepcopy(config)
    captured: dict = {}
    sentinel_model = object()
    sentinel_capacity = {"trainable_parameter_count": 123}

    def fake_builder(config_view, vocabs, *, run_id):
        captured["config"] = config_view
        captured["vocabs"] = vocabs
        captured["run_id"] = run_id
        return sentinel_model, sentinel_capacity

    monkeypatch.setattr(s1, "build_setflow_screen_model_v4", fake_builder)
    vocabs = {"assay": {"__UNK__": 0}}
    model, capacity = s1.build_setflow_screen_model_s1(
        config, vocabs, run_id="v4_s1_full"
    )
    assert model is sentinel_model
    assert capacity is sentinel_capacity
    assert S1_RUN_ROLE_TO_V4 == {
        "v4_s1_full": "v4_full",
        "v4_s1_single_mode": "v4_single_mode",
    }
    assert captured == {
        "config": {
            **config,
            "required_screen_runs": [
                {
                    "run_id": "v4_full",
                    "mode_count": 8,
                    "mode_information_weight": 0.05,
                    "selectable": True,
                }
            ],
        },
        "vocabs": vocabs,
        "run_id": "v4_full",
    }
    assert config == before
    assert screen_run_spec_s1(config, "v4_s1_full").run_id == "v4_s1_full"


def _loss_bundle(*, swapped_rows: tuple[int, ...] = ()) -> tuple[dict, dict]:
    aligned = torch.tensor(
        [
            [9.0, 1.0],
            [1.0, 9.0],
        ]
    )
    swapped = aligned.flip(0)
    rates = torch.stack(
        [swapped if row in swapped_rows else aligned for row in range(3)]
        + [torch.zeros_like(aligned)]
    )
    candidate_positive = torch.tensor(
        [
            [[True, False], [False, True]],
            [[True, False], [False, True]],
            [[True, False], [False, True]],
            [[False, False], [False, False]],
        ]
    )
    batch = {
        "common_positive_action_mask": torch.tensor(
            [
                [True, True],
                [True, True],
                [True, True],
                [False, False],
            ]
        ),
        "candidate_positive_action_mask": candidate_positive,
        "candidate_valid_mask": torch.ones(4, 2, dtype=torch.bool),
        "remaining_count_soft_target": torch.tensor(
            [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]] * 3
            + [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        ),
        "structural_budget_exhausted": torch.tensor(
            [False, False, False, True]
        ),
    }
    output = {
        "mode_rates": rates,
        "legal_action_mask": torch.tensor(
            [[True, True], [True, True], [True, True], [False, False]]
        ),
        "mode_prior": torch.full((4, 2), 0.5),
        "remaining_count_logits": torch.zeros(4, 6),
    }
    return output, batch


def _s1_loss(
    output: dict,
    batch: dict,
    *,
    canonical_candidate_indices: torch.Tensor | None = None,
):
    return mixture_setflow_loss_s1(
        output,
        batch,
        state_slots=torch.tensor([0, 1, 2, 3]),
        source_occurrence_ids=torch.tensor([0, 0, 0, 0]),
        canonical_candidate_indices=(
            torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
            if canonical_candidate_indices is None
            else canonical_candidate_indices
        ),
        cross_state_candidate_mode_responsibility_weight=0.05,
    )


def _permute_candidates(
    batch: dict, canonical: torch.Tensor, orders: tuple[tuple[int, int], ...]
) -> tuple[dict, torch.Tensor]:
    permuted = dict(batch)
    for key in ("candidate_positive_action_mask", "candidate_valid_mask"):
        value = batch[key]
        permuted[key] = torch.stack(
            [value[row, list(order)] for row, order in enumerate(orders)]
        )
    return permuted, torch.stack(
        [canonical[row, list(order)] for row, order in enumerate(orders)]
    )


def test_swapped_nonroot_modes_have_positive_forward_kl_and_finite_backward() -> None:
    aligned_output, batch = _loss_bundle()
    aligned = _s1_loss(aligned_output, batch)
    swapped_output, batch = _loss_bundle(swapped_rows=(1,))
    swapped_output["mode_rates"] = swapped_output["mode_rates"].requires_grad_()
    swapped = _s1_loss(swapped_output, batch)
    assert aligned.cross_state_candidate_mode_responsibility.item() == 0.0
    assert swapped.cross_state_candidate_mode_responsibility.item() > 0.0
    assert swapped.active_responsibility_constraint_count == 4
    assert swapped.active_responsibility_candidate_count == 2
    assert swapped.active_responsibility_occurrence_count == 1
    swapped.total.backward()
    assert swapped_output["mode_rates"].grad is not None
    assert torch.isfinite(swapped_output["mode_rates"].grad).all()


def test_each_state_can_permute_candidate_rows_without_changing_s1_loss() -> None:
    output, batch = _loss_bundle(swapped_rows=(1,))
    canonical = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
    baseline = _s1_loss(output, batch, canonical_candidate_indices=canonical)
    permuted_batch, permuted_canonical = _permute_candidates(
        batch,
        canonical,
        ((1, 0), (0, 1), (1, 0), (0, 1)),
    )
    permuted = _s1_loss(
        output,
        permuted_batch,
        canonical_candidate_indices=permuted_canonical,
    )
    assert torch.equal(
        baseline.cross_state_candidate_mode_responsibility,
        permuted.cross_state_candidate_mode_responsibility,
    )
    assert torch.equal(baseline.total, permuted.total)


def test_canonical_identity_poison_fails_closed() -> None:
    output, batch = _loss_bundle()
    poisoned = torch.tensor([[0, 1], [0, 0], [0, 1], [0, 1]])
    with pytest.raises(RuntimeError, match="duplicate canonical candidate identity"):
        _s1_loss(output, batch, canonical_candidate_indices=poisoned)


def test_structural_nonroot_state_is_excluded_from_responsibility_reduction() -> None:
    output, batch = _loss_bundle(swapped_rows=(2,))
    batch["structural_budget_exhausted"] = torch.tensor(
        [False, False, True, True]
    )
    batch["common_positive_action_mask"][2].zero_()
    batch["candidate_positive_action_mask"][2].zero_()
    output["legal_action_mask"][2].zero_()
    output["mode_rates"][2].zero_()
    result = _s1_loss(output, batch)
    assert result.cross_state_candidate_mode_responsibility.item() == 0.0
    assert result.active_responsibility_constraint_count == 2
    assert result.active_responsibility_candidate_count == 2
    assert result.active_responsibility_occurrence_count == 1


def test_root_posterior_is_a_detached_target() -> None:
    output, batch = _loss_bundle(swapped_rows=(1,))
    output["mode_rates"] = output["mode_rates"].requires_grad_()
    result = _s1_loss(output, batch)
    result.cross_state_candidate_mode_responsibility.backward()
    gradient = output["mode_rates"].grad
    assert gradient is not None
    assert torch.equal(gradient[0], torch.zeros_like(gradient[0]))
    assert torch.count_nonzero(gradient[1]).item() > 0


def test_reduction_means_states_then_candidates_then_occurrences() -> None:
    first_output, first_batch = _loss_bundle(swapped_rows=(1,))
    first = _s1_loss(first_output, first_batch)

    second_output, second_batch = _loss_bundle(swapped_rows=(1,))
    second_batch["structural_budget_exhausted"] = torch.tensor(
        [False, False, True, True]
    )
    second_batch["common_positive_action_mask"][2].zero_()
    second_batch["candidate_positive_action_mask"][2].zero_()
    second_output["legal_action_mask"][2].zero_()
    second_output["mode_rates"][2].zero_()
    second = _s1_loss(second_output, second_batch)

    combined_output = {
        key: torch.cat((first_output[key], second_output[key]), dim=0)
        for key in first_output
    }
    combined_batch = {
        key: torch.cat((first_batch[key], second_batch[key]), dim=0)
        for key in first_batch
    }
    combined = mixture_setflow_loss_s1(
        combined_output,
        combined_batch,
        state_slots=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
        source_occurrence_ids=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        canonical_candidate_indices=torch.tensor(
            [[0, 1], [0, 1], [0, 1], [0, 1]] * 2
        ),
        cross_state_candidate_mode_responsibility_weight=0.05,
    )
    expected = 0.5 * (
        first.cross_state_candidate_mode_responsibility
        + second.cross_state_candidate_mode_responsibility
    )
    assert torch.allclose(
        combined.cross_state_candidate_mode_responsibility,
        expected,
        atol=1e-7,
    )
    assert combined.active_responsibility_constraint_count == 6
    assert combined.active_responsibility_candidate_count == 4
    assert combined.active_responsibility_occurrence_count == 2


def test_single_mode_s1_is_exactly_the_old_v4_total_and_gradients() -> None:
    output, batch = _loss_bundle(swapped_rows=(1,))
    old_rates = output["mode_rates"][:, :1].clone().requires_grad_()
    old_output = {
        **output,
        "mode_rates": old_rates,
        "mode_prior": torch.ones(4, 1),
    }
    old = mixture_setflow_loss_v4(old_output, batch)
    old.total.backward()
    old_gradient = old_rates.grad.detach().clone()

    s1_rates = output["mode_rates"][:, :1].clone().requires_grad_()
    s1_output = {
        **output,
        "mode_rates": s1_rates,
        "mode_prior": torch.ones(4, 1),
    }
    successor = _s1_loss(s1_output, batch)
    assert successor.total is successor.base_v4.total
    assert torch.equal(successor.total, old.total)
    assert successor.cross_state_candidate_mode_responsibility.item() == 0.0
    assert successor.active_responsibility_constraint_count == 4
    successor.total.backward()
    assert torch.equal(s1_rates.grad, old_gradient)
