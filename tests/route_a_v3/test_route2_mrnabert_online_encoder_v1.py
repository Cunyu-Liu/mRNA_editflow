from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.route_a_v3 import route2_mrnabert_online_encoder_v1 as encoder
from scripts.route_a_v3 import validate_route2_mrnabert_online_encoder_v1 as validation


def test_online_tokenization_and_chunk_pooling_match_cache_policy() -> None:
    assert encoder.normalize_rna("ACTN") == "ACUN"
    assert encoder.format_utr_chunk("AUGCN") == "A T G C N"
    assert encoder.sequence_chunks("A" * 11, 5) == ["A" * 5, "A" * 5, "A"]
    hidden = torch.tensor([[[1.0], [3.0], [99.0]]])
    attention = torch.tensor([[1, 1, 0]])
    assert encoder.pool_last_hidden(hidden, attention).item() == 2.0


def test_online_batches_use_special_token_budget() -> None:
    chunks = [(str(index), "A" * length) for index, length in enumerate((50, 60, 500, 900))]
    batches = list(encoder.chunk_batches(chunks, 3, 1000))
    assert [len(batch) for batch in batches] == [2, 1, 1]


def test_sdpa_adapter_matches_manual_packed_qkv_attention() -> None:
    torch.manual_seed(20260817)
    qkv = torch.randn(2, 7, 3, 4, 8)
    bias = torch.randn(2, 4, 7, 7) * 0.1
    query = qkv[:, :, 0].permute(0, 2, 1, 3)
    key = qkv[:, :, 1].permute(0, 2, 3, 1)
    value = qkv[:, :, 2].permute(0, 2, 1, 3)
    scores = torch.matmul(query, key) / (query.shape[-1] ** 0.5)
    expected = torch.matmul(torch.softmax(scores + bias, dim=-1), value).permute(
        0, 2, 1, 3
    )
    observed = encoder.pytorch_sdpa_qkvpacked(qkv, bias)
    assert torch.allclose(observed, expected, atol=1e-5, rtol=1e-5)


def test_attention_backend_names_are_explicit() -> None:
    assert encoder.ATTENTION_BACKENDS == {
        "OFFICIAL_PYTORCH_FALLBACK",
        "PYTORCH_SDPA_AUTO",
    }


def test_validation_sampling_spans_full_record_order() -> None:
    assert validation._sample_indices(10, 4) == [0, 3, 6, 9]
    assert validation._sample_indices(3, 8) == [0, 1, 2]


def test_novel_candidate_is_legal_and_not_in_known_set() -> None:
    known = {"AAAA", "CAAA", "GAAA"}
    candidate = validation._novel_single_substitution("AAAA", known)
    assert candidate not in known
    assert len(candidate) == 4
    assert sum(left != right for left, right in zip(candidate, "AAAA")) == 1


def test_invalid_sequence_and_impossible_novel_candidate_fail() -> None:
    with pytest.raises(encoder.OnlineEncoderError, match="unsupported"):
        encoder.normalize_rna("AX")
    known = {
        replacement + "A" for replacement in "ACGU"
    } | {
        "A" + replacement for replacement in "ACGU"
    }
    with pytest.raises(validation.OnlineEncoderValidationError, match="could not construct"):
        validation._novel_single_substitution("AA", known)
