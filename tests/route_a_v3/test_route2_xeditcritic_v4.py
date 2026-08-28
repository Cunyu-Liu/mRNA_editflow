from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from core.route2_xeditcritic_v4 import (
    EndpointSemanticRouterV4,
    LocalCrossAttentionBlockV4,
    LocalPooledResidualBlockV4,
    TrainableUpperSixTransformerV4,
    XEditCriticV4,
    XEditCriticV4Error,
    gather_ragged_local_contexts_v4,
    require_v4_trainable_parameter_range,
)


class _Upper(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(width, width)

    def forward(self, hidden, attention_mask):
        return self.projection(hidden) * attention_mask.unsqueeze(-1)


def _model(*, control_mode: str = "NONE", mechanism_mode: str = "FULL", dropout: float = 0.0) -> XEditCriticV4:
    torch.manual_seed(7)
    return XEditCriticV4(
        upper_encoder=_Upper(16),
        study_count=3,
        assay_count=3,
        context_count=4,
        quantity_count=5,
        measurement_count=6,
        numerator_count=4,
        denominator_count=4,
        control_mode=control_mode,
        mechanism_mode=mechanism_mode,
        pretrained_width=16,
        model_width=16,
        block_count=4,
        heads=4,
        ffn_width=32,
        expert_count=2,
        expert_bottleneck_width=8,
        expert_top_k=1,
        base_embedding_width=4,
        raw_hidden_dim=17,
        raw_depth=2,
        readout_hidden_width=32,
        dropout=dropout,
        minimum_physical_batch=4,
    )


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
    batch_size = 4
    chunk_count = 6
    token_count = 12
    width = 16
    source_tokens = torch.tensor(
        [
            [0, 1, 2, 3, 0, 1],
            [1, 1, 2, 2, 3, 3],
            [0, 0, 1, 1, 2, 2],
            [3, 2, 1, 0, 3, 2],
        ]
    )
    candidate_tokens = source_tokens.clone()
    candidate_tokens[0, 2] = 3
    candidate_tokens[1, 4] = 0
    candidate_tokens[2, 1] = 2
    # Record 3 is an identity pair.  The padded edit column is retained so the
    # physical batch has one common ragged geometry.
    record_edit_offsets = torch.tensor([0, 1, 2, 3, 3])
    edit_padding_mask = torch.tensor(
        [[False], [False], [False], [True]], dtype=torch.bool
    )
    source_chunk_indices = torch.tensor([0, 1, 2])
    candidate_chunk_indices = torch.tensor([3, 4, 5])
    centers = torch.tensor([5, 6, 4])
    starts = torch.tensor([2, 3, 1])
    ends = torch.tensor([9, 10, 8])
    source_base = torch.tensor([[2], [3], [1], [4]])
    candidate_base = torch.tensor([[3], [0], [2], [4]])
    return {
        "chunk_hidden": torch.randn(chunk_count, token_count, width),
        "chunk_attention_mask": torch.ones(chunk_count, token_count, dtype=torch.bool),
        "record_edit_offsets": record_edit_offsets,
        "edit_source_chunk_indices": source_chunk_indices,
        "edit_candidate_chunk_indices": candidate_chunk_indices,
        "edit_source_token_centers": centers,
        "edit_candidate_token_centers": centers,
        "edit_source_window_starts": starts,
        "edit_source_window_ends": ends,
        "edit_candidate_window_starts": starts,
        "edit_candidate_window_ends": ends,
        "source_tokens": source_tokens,
        "candidate_tokens": candidate_tokens,
        "padding_mask": torch.zeros_like(source_tokens, dtype=torch.bool),
        "source_edit_base_ids": source_base,
        "candidate_edit_base_ids": candidate_base,
        "normalized_edit_positions": torch.tensor([[0.4], [0.8], [0.2], [0.0]]),
        "edit_padding_mask": edit_padding_mask,
        "record_source_global": torch.randn(batch_size, width),
        "record_candidate_global": torch.randn(batch_size, width),
        "study_ids": torch.tensor([0, 1, 2, 0]),
        "assay_ids": torch.tensor([1, 2, 1, 0]),
        "context_ids": torch.tensor([2, 3, 1, 0]),
        "quantity_ids": torch.tensor([1, 4, 2, 0]),
        "measurement_ids": torch.tensor([2, 5, 3, 0]),
        "numerator_ids": torch.tensor([1, 3, 2, 0]),
        "denominator_ids": torch.tensor([2, 3, 1, 0]),
        "region_ids": torch.tensor([0, 1, 0, 1]),
    }


def _swap(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    swapped = copy.copy(batch)
    for left, right in (
        ("source_tokens", "candidate_tokens"),
        ("record_source_global", "record_candidate_global"),
        ("source_edit_base_ids", "candidate_edit_base_ids"),
        ("edit_source_chunk_indices", "edit_candidate_chunk_indices"),
        ("edit_source_token_centers", "edit_candidate_token_centers"),
        ("edit_source_window_starts", "edit_candidate_window_starts"),
        ("edit_source_window_ends", "edit_candidate_window_ends"),
    ):
        swapped[left], swapped[right] = batch[right], batch[left]
    return swapped


def test_v4_full_is_strictly_antisymmetric_and_identity_is_exact_zero() -> None:
    model = _model().eval()
    batch = _batch()
    prediction = model(batch)["mean"]
    reverse = model(_swap(batch))["mean"]
    assert torch.equal(prediction, -reverse)
    assert prediction[3].item() == 0.0


def test_v4_identity_stays_zero_in_training_with_shared_dropout_masks() -> None:
    model = _model(dropout=0.1).train()
    batch = _batch()
    for _ in range(3):
        assert model(batch)["mean"][3].item() == 0.0


def test_v4_rejects_singleton_or_subminimum_physical_batches() -> None:
    model = _model()
    batch = _batch()
    smaller = {
        key: value[:3] if isinstance(value, torch.Tensor) and value.shape[:1] == (4,) else value
        for key, value in batch.items()
    }
    smaller["record_edit_offsets"] = torch.tensor([0, 1, 2, 3])
    with pytest.raises(XEditCriticV4Error, match="physical batch"):
        model(smaller)


def test_semantic_router_selects_exactly_top_two_without_study_input() -> None:
    router = EndpointSemanticRouterV4(8, expert_count=4, top_k=2)
    weights, balance = router(torch.randn(6, 8))
    assert torch.equal((weights > 0).sum(dim=1), torch.full((6,), 2))
    assert torch.allclose(weights.sum(dim=1), torch.ones(6))
    assert balance.ndim == 0 and torch.isfinite(balance)
    assert not any("study" in name for name, _ in router.named_parameters())


def test_semantic_router_accepts_autocast_promoted_softmax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = EndpointSemanticRouterV4(8, expert_count=4, top_k=2).to(
        dtype=torch.bfloat16
    )
    original_softmax = torch.softmax

    def promoted_softmax(values: torch.Tensor, *, dim: int) -> torch.Tensor:
        return original_softmax(values.float(), dim=dim)

    monkeypatch.setattr(torch, "softmax", promoted_softmax)
    weights, balance = router(torch.randn(6, 8, dtype=torch.bfloat16))
    assert weights.dtype == torch.bfloat16
    assert torch.equal((weights > 0).sum(dim=1), torch.full((6,), 2))
    assert torch.allclose(
        weights.float().sum(dim=1),
        torch.ones(6),
        atol=1e-2,
        rtol=0.0,
    )
    assert balance.dtype == torch.float32 and torch.isfinite(balance)


def test_local_context_gather_ignores_every_token_outside_declared_windows() -> None:
    batch = _batch()
    upper = batch["chunk_hidden"].clone()
    gathered = gather_ragged_local_contexts_v4(
        upper_chunk_hidden=upper,
        batch=batch,
        edit_padding_mask=batch["edit_padding_mask"],
    )
    changed = upper.clone()
    # Every active window is within token indices 1..9.  Tokens 0, 10, and 11
    # must not enter a local source or candidate context.
    changed[:, 0] = 1000
    changed[:, 10:] = -1000
    regathered = gather_ragged_local_contexts_v4(
        upper_chunk_hidden=changed,
        batch=batch,
        edit_padding_mask=batch["edit_padding_mask"],
    )
    for key in ("source_context", "candidate_context", "source_site", "candidate_site"):
        assert torch.equal(gathered[key], regathered[key])


def test_no_cross_and_no_moe_are_parameter_matched_without_unused_padding() -> None:
    full = _model(mechanism_mode="FULL")
    no_cross = _model(mechanism_mode="NO_CROSS")
    no_moe = _model(mechanism_mode="NO_MOE")
    assert full.trainable_parameter_count == no_cross.trainable_parameter_count
    assert full.trainable_parameter_count == no_moe.trainable_parameter_count
    assert any(isinstance(module, LocalCrossAttentionBlockV4) for module in full.modules())
    assert any(isinstance(module, LocalPooledResidualBlockV4) for module in no_cross.modules())
    for model in (full, no_cross, no_moe):
        output = model(_batch())
        loss = output["mean"].square().mean() + output["router_balance_loss"]
        loss.backward()
        assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_identical_v4_control_constructors_share_seeded_initial_tensors() -> None:
    full = _model()
    full_state = full.state_dict()
    for control_mode, mechanism_mode in (
        ("SOURCE_ONLY", "FULL"),
        ("EDIT_METADATA_ONLY", "FULL"),
        ("NO_CANDIDATE_SEQUENCE", "FULL"),
        ("NONE", "NO_MOE"),
    ):
        control = _model(
            control_mode=control_mode, mechanism_mode=mechanism_mode
        )
        control_state = control.state_dict()
        assert control_state.keys() == full_state.keys()
        assert all(
            torch.equal(full_state[name], control_state[name])
            for name in full_state
        )

    no_cross = _model(mechanism_mode="NO_CROSS")
    assert no_cross.state_dict().keys() != full_state.keys()


def test_candidate_information_controls_keep_full_parameter_geometry() -> None:
    batch = _batch()
    full_count = _model().trainable_parameter_count
    for control in ("SOURCE_ONLY", "EDIT_METADATA_ONLY", "NO_CANDIDATE_SEQUENCE"):
        model = _model(control_mode=control)
        assert model.trainable_parameter_count == full_count
        assert torch.isfinite(model(batch)["mean"]).all()


def test_study_identity_only_enters_scale_and_unknown_scale_is_one() -> None:
    model = _model()
    study_parameters = [name for name, _ in model.named_parameters() if "study" in name]
    assert study_parameters == ["study_calibration.known_log_scale"]
    with torch.no_grad():
        model.study_calibration.known_log_scale.copy_(torch.tensor([0.2, -0.4]))
    scales = model.study_calibration.scale(torch.tensor([0, 1, 2]))
    assert scales[0].item() == 1.0


def test_default_v4_geometry_hits_the_prefrozen_design_target_on_meta_device() -> None:
    with torch.device("meta"):
        upper = TrainableUpperSixTransformerV4()
        model = XEditCriticV4(
            upper_encoder=upper,
            study_count=8,
            assay_count=16,
            context_count=64,
            quantity_count=7,
            measurement_count=7,
            numerator_count=7,
            denominator_count=7,
        )
    capacity = require_v4_trainable_parameter_range(model)
    assert 165_000_000 <= capacity["trainable_parameter_count"] <= 175_000_000
    assert sum(capacity["module_counts"].values()) == capacity["trainable_parameter_count"]


def test_formal_upper_count_and_one_shared_four_expert_bank_hit_exact_capacity() -> None:
    class FormalUpperCount(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.parameters_for_count = nn.Parameter(
                torch.empty(56_664_576, device="meta")
            )

        def forward(self, hidden, attention_mask):
            return hidden

    with torch.device("meta"):
        model = XEditCriticV4(
            upper_encoder=FormalUpperCount(),
            study_count=8,
            assay_count=7,
            context_count=28,
            quantity_count=6,
            measurement_count=5,
            numerator_count=6,
            denominator_count=6,
        )
    expert_ids = {id(block.experts) for block in model.blocks}
    assert len(expert_ids) == 1
    capacity = require_v4_trainable_parameter_range(model)
    assert capacity["trainable_parameter_count"] == 170_481_733
