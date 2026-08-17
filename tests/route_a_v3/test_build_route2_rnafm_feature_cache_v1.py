from __future__ import annotations

from scripts.route_a_v3 import build_route2_rnafm_feature_cache_v1 as builder


def test_long_sequences_are_fully_partitioned_without_truncation() -> None:
    sequence = "A" * 1874
    chunks = builder._sequence_chunks(sequence, 1000)
    assert [len(chunk) for chunk in chunks] == [1000, 874]
    assert "".join(chunks) == sequence


def test_embedding_batches_obey_sequence_and_token_budgets() -> None:
    chunks = [(str(index), "A" * length) for index, length in enumerate([50, 60, 500, 900])]
    batches = list(builder._batches(chunks, maximum_sequences=3, token_budget=1000))
    assert [len(batch) for batch in batches] == [2, 1, 1]
    assert all(
        len(batch) <= 3 and max(len(chunk) for _key, chunk in batch) * len(batch) <= 1000
        for batch in batches
    )
