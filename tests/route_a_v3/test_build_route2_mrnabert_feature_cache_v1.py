from __future__ import annotations

import pytest
import torch

from scripts.route_a_v3 import build_route2_mrnabert_feature_cache_v1 as builder


def test_long_utr_is_fully_partitioned_without_truncation() -> None:
    sequence = "A" * 1874
    chunks = builder._sequence_chunks(sequence, 1000)
    assert [len(chunk) for chunk in chunks] == [1000, 874]
    assert "".join(chunks) == sequence


def test_official_utr_tokenization_is_single_letter_and_uses_dna_alphabet() -> None:
    assert builder._format_utr_chunk("AUGCN") == "A T G C N"
    with pytest.raises(builder.DeltaTrainingError, match="unsupported base"):
        builder._format_utr_chunk("AX")


def test_embedding_batches_include_special_tokens_in_budget() -> None:
    chunks = [(str(index), "A" * length) for index, length in enumerate([50, 60, 500, 900])]
    batches = list(builder._batches(chunks, maximum_sequences=3, token_budget=1000))
    assert [len(batch) for batch in batches] == [2, 1, 1]
    assert all(
        len(batch) <= 3
        and max(len(chunk) + 2 for _key, chunk in batch) * len(batch) <= 1000
        for batch in batches
    )


def test_pooling_matches_official_attention_masked_mean() -> None:
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    attention = torch.tensor([[1, 1, 0]])
    assert torch.equal(
        builder._pool_last_hidden(hidden, attention),
        torch.tensor([[2.0, 3.0]]),
    )
