from __future__ import annotations

import pytest

from data.utr_benchmark_v2.near_neighbors import NEAR_NEIGHBOR_ALGORITHM_ID
from data.utr_benchmark_v2.near_neighbors import NEAR_NEIGHBOR_CANDIDATE_BACKEND
from data.utr_benchmark_v2.near_neighbors import NEAR_NEIGHBOR_DISTANCE_BACKEND
from data.utr_benchmark_v2.near_neighbors import NearNeighborError
from data.utr_benchmark_v2.near_neighbors import _banded_distance_at_most
from data.utr_benchmark_v2.near_neighbors import build_near_neighbor_clusters


def test_six_block_candidates_cannot_miss_five_distributed_edits() -> None:
    source_blocks = [
        "AAAAAA",
        "CCCCCC",
        "GGGGGG",
        "UUUUUU",
        "ACACAC",
        "GUGUGU",
    ]
    candidate_blocks = [
        "CAAAAA",
        "ACCCCC",
        "AGGGGG",
        "AUUUUU",
        "CCACAC",
        source_blocks[-1],
    ]
    source = "".join(source_blocks)
    candidate = "".join(candidate_blocks)
    clusters = build_near_neighbor_clusters(
        {"source-record": (source,), "candidate-record": (candidate,)}
    )

    assert (
        clusters.record_clusters["source-record"]
        == clusters.record_clusters["candidate-record"]
    )
    assert clusters.binding["algorithm"] == NEAR_NEIGHBOR_ALGORITHM_ID
    assert (
        clusters.binding["candidate_deduplication"]
        == NEAR_NEIGHBOR_CANDIDATE_BACKEND
    )
    assert clusters.binding["resource_policy"] == "uncapped_exact_completion"
    assert clusters.binding["distance_backend"] == NEAR_NEIGHBOR_DISTANCE_BACKEND
    assert clusters.binding["edit_distance_threshold"] == 5
    assert clusters.binding["candidate_generation_complete"] is True
    assert clusters.binding["qualifying_pair_count"] == 1


def test_short_long_indel_pair_is_found_and_distance_six_is_rejected() -> None:
    within = build_near_neighbor_clusters(
        {"short": ("A",), "long": ("CCCCCA",)}
    )
    assert within.record_clusters["short"] == within.record_clusters["long"]

    outside = build_near_neighbor_clusters(
        {"left": ("AAAAAA",), "right": ("CCCCCC",)}
    )
    assert outside.record_clusters["left"] != outside.record_clusters["right"]
    assert outside.binding["qualifying_pair_count"] == 0


def test_near_neighbor_clustering_is_order_stable_and_fails_closed() -> None:
    first = build_near_neighbor_clusters(
        {"b": ("CCCCCCCC",), "a": ("AAAAAAAA",)}
    )
    second = build_near_neighbor_clusters(
        {"a": ("AAAAAAAA",), "b": ("CCCCCCCC",)}
    )
    assert first.binding == second.binding
    assert first.record_clusters == second.record_clusters

    with pytest.raises(
        NearNeighborError, match="STOP_RULE_B0_NEAR_NEIGHBOR_COMPLEXITY"
    ):
        build_near_neighbor_clusters(
            {"a": ("AAAAAAAA",), "b": ("CCCCCCCC",)},
            max_sequences=1,
        )


def test_exact_candidate_spool_retains_an_explicit_fail_closed_cap() -> None:
    with pytest.raises(
        NearNeighborError,
        match="STOP_RULE_B0_NEAR_NEIGHBOR_COMPLEXITY",
    ):
        build_near_neighbor_clusters(
            {
                "a": ("AAAAAA",),
                "b": ("AAAAAC",),
                "c": ("AAAACC",),
            },
            max_candidate_pairs=1,
        )


def test_distance_backend_preserves_thresholded_exact_distance() -> None:
    assert _banded_distance_at_most("AAAAACCCC", "AAAAA", 5) == 4
    assert _banded_distance_at_most("AAAAAA", "CCCCCC", 5) is None
