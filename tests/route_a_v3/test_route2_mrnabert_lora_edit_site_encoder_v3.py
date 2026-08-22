from __future__ import annotations

import torch

from scripts.route_a_v3.route2_mrnabert_lora_edit_site_encoder_v3 import (
    populate_edit_features_from_hidden_v3,
)


def test_online_feature_extraction_matches_one_leading_special_and_radius_window() -> None:
    # CLS, 40 nucleotide tokens, SEP.  Nucleotide p has value p+1.
    values = torch.arange(42, dtype=torch.float32)
    source_hidden = values.reshape(1, 42, 1)
    candidate_hidden = (values + 100).reshape(1, 42, 1)
    mask = torch.ones((1, 42), dtype=torch.long)
    batch = {
        "padding_mask": torch.zeros((1, 40), dtype=torch.bool),
        "edit_padding_mask": torch.tensor([[False, False, False]]),
        "edit_positions": torch.tensor([[0, 20, 39]]),
    }
    result = populate_edit_features_from_hidden_v3(
        batch,
        source_hidden=source_hidden,
        candidate_hidden=candidate_hidden,
        source_attention_mask=mask,
        candidate_attention_mask=mask,
    )
    assert result["source_site"].flatten().tolist() == [1.0, 21.0, 40.0]
    assert result["candidate_site"].flatten().tolist() == [101.0, 121.0, 140.0]
    assert result["source_window_mean"][0, 0, 0].item() == 9.0
    assert result["source_window_mean"][0, 1, 0].item() == 21.0
    assert result["source_window_max"][0, 2, 0].item() == 40.0
    assert result["source_global"].item() == 20.5


def test_padded_edit_slots_are_zero_and_keep_ragged_width() -> None:
    hidden = torch.randn(1, 12, 4, requires_grad=True)
    mask = torch.ones((1, 12), dtype=torch.long)
    batch = {
        "padding_mask": torch.zeros((1, 10), dtype=torch.bool),
        "edit_padding_mask": torch.tensor([[False, True, True]]),
        "edit_positions": torch.tensor([[5, 0, 0]]),
    }
    result = populate_edit_features_from_hidden_v3(
        batch,
        source_hidden=hidden,
        candidate_hidden=hidden,
        source_attention_mask=mask,
        candidate_attention_mask=mask,
    )
    assert result["source_site"].shape == (1, 3, 4)
    assert torch.equal(result["source_site"][0, 1:], torch.zeros(2, 4))
    result["source_site"].sum().backward()
    assert hidden.grad is not None
