from __future__ import annotations

import math
from typing import get_type_hints

import pytest
import torch

from core.route2_xedit_v4_interfaces import (
    CriticPredictionV4,
    CriticStateBatchV4,
    MixtureSetMarginalTargetV4,
    SetFlowCheckpointDecisionV4,
    SetFlowSourceBatchV4,
)
from core.route2_xeditcritic_batch_v4 import XEditCriticCollatorV4
from core.route2_xeditcritic_v4 import (
    EndpointSemanticMixtureV4,
    XEditCriticV4,
)
from core.route2_xeditsetflow_gate_v4 import select_checkpoint_v4
from core.route2_xeditsetflow_training_v4 import (
    collate_setflow_source_states_v4,
)
from core.route2_xeditsetflow_v4 import XEditSetFlowV4, mixture_setflow_loss_v4


def test_named_batch_and_checkpoint_interfaces_are_used_by_real_consumers() -> None:
    assert get_type_hints(XEditCriticCollatorV4.__call__)["return"] is CriticStateBatchV4
    assert get_type_hints(XEditCriticV4.forward)["batch"] is CriticStateBatchV4
    assert (
        get_type_hints(collate_setflow_source_states_v4)["return"]
        is SetFlowSourceBatchV4
    )
    assert get_type_hints(XEditSetFlowV4.forward)["batch"] is SetFlowSourceBatchV4
    assert (
        get_type_hints(select_checkpoint_v4)["return"]
        is SetFlowCheckpointDecisionV4
    )
    assert {
        "edit_source_chunk_indices",
        "edit_candidate_chunk_indices",
        "edit_source_token_centers",
        "edit_candidate_token_centers",
        "edit_source_window_starts",
        "edit_source_window_ends",
        "edit_candidate_window_starts",
        "edit_candidate_window_ends",
    } <= CriticStateBatchV4.__required_keys__
    assert {
        "state_slots",
        "source_occurrence_ids",
        "canonical_candidate_indices",
    } <= SetFlowSourceBatchV4.__required_keys__


def test_critic_prediction_v4_freezes_three_seed_mean_sd_reward_and_neutrality() -> None:
    prediction = CriticPredictionV4.from_seed_predictions(
        {20260908: 0.2, 20260909: 0.4, 20260910: 0.6},
        uncertainty_penalty_kappa=0.5,
    )
    expected_sd = math.sqrt((0.2**2 + 0.0**2 + 0.2**2) / 3.0)
    assert prediction.ensemble_mean == pytest.approx(0.4)
    assert prediction.ensemble_sd == pytest.approx(expected_sd)
    assert prediction.standardized_reward == pytest.approx(0.4 - 0.5 * expected_sd)
    assert prediction.study_neutral is True
    artifact = prediction.to_artifact()
    assert set(artifact["per_seed_predictions"]) == {
        "20260908",
        "20260909",
        "20260910",
    }
    assert artifact["study_neutral"] is True

    with pytest.raises(Exception, match="exact frozen three seeds"):
        CriticPredictionV4.from_seed_predictions(
            {20260908: 0.2, 20260909: 0.4},
            uncertainty_penalty_kappa=0.5,
        )
    with pytest.raises(Exception, match="nonfinite"):
        CriticPredictionV4.from_seed_predictions(
            {20260908: 0.2, 20260909: float("nan"), 20260910: 0.6},
            uncertainty_penalty_kappa=0.5,
        )


def test_endpoint_semantic_mixture_is_the_expert_module_used_by_v4_blocks() -> None:
    module = EndpointSemanticMixtureV4(
        width=8,
        bottleneck_width=4,
        expert_count=4,
        semantic_routing=True,
    )
    values = torch.randn(2, 3, 2, 8)
    route = torch.tensor(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.5],
        ]
    )
    output = module(values, route)
    assert output.shape == values.shape
    assert torch.isfinite(output).all()


def test_mixture_target_interface_keeps_candidates_separate_from_union_mass() -> None:
    batch = {
        "common_positive_action_mask": torch.tensor([[True, True, False]]),
        "candidate_positive_action_mask": torch.tensor(
            [[[True, False, False], [False, True, False]]]
        ),
        "candidate_valid_mask": torch.tensor([[True, True]]),
        "remaining_count_soft_target": torch.tensor(
            [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
        ),
        "structural_budget_exhausted": torch.tensor([False]),
    }
    target = MixtureSetMarginalTargetV4.from_source_batch(batch)
    assert target.candidate_positive_action_mask.shape == (1, 2, 3)
    output = {
        "mode_rates": torch.tensor([[[9.0, 1.0, 1.0], [1.0, 9.0, 1.0]]]),
        "legal_action_mask": torch.tensor([[True, True, True]]),
        "mode_prior": torch.tensor([[0.5, 0.5]]),
        "remaining_count_logits": torch.zeros(1, 6),
    }
    loss = mixture_setflow_loss_v4(output, batch)
    assert loss.active_candidate_constraint_count == 2

    drifted = dict(batch)
    drifted["candidate_valid_mask"] = torch.tensor([[True]])
    with pytest.raises(Exception, match="candidate validity geometry"):
        MixtureSetMarginalTargetV4.from_source_batch(drifted)
