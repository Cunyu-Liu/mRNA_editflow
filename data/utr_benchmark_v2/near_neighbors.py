"""Exact global near-neighbor clustering for label-free sequence states.

The candidate generator uses the edit-distance pigeonhole principle.  A
sequence is split into ``k + 1`` non-empty blocks.  If another sequence is
within ``k`` primitive Levenshtein edits, at least one block is untouched and
must occur as an exact substring in the other sequence.  Every generated pair
is then verified with exact banded Levenshtein distance; block collisions are
never treated as edges by themselves.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Tuple


NEAR_NEIGHBOR_EDIT_THRESHOLD = 5
NEAR_NEIGHBOR_BLOCK_COUNT = NEAR_NEIGHBOR_EDIT_THRESHOLD + 1
NEAR_NEIGHBOR_ALGORITHM_ID = (
    "six_block_pigeonhole_all_substring_candidates_exact_banded_levenshtein_v1"
)


class NearNeighborError(ValueError):
    """Raised when an exact near-neighbor audit cannot finish safely."""


@dataclass(frozen=True)
class NearNeighborClusters:
    """Frozen global clustering plus its auditable manifest binding."""

    binding: Mapping[str, Any]
    record_clusters: Mapping[str, str]
    sequence_clusters: Mapping[str, str]


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_limit(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise NearNeighborError(f"{name} must be a positive integer")


def _validate_sequence(sequence: str) -> None:
    if (
        not isinstance(sequence, str)
        or not sequence
        or sequence != sequence.upper()
        or any(base not in "ACGU" for base in sequence)
    ):
        raise NearNeighborError(
            "near-neighbor states must be non-empty uppercase A/C/G/U RNA"
        )


def _balanced_blocks(sequence: str) -> Tuple[str, ...]:
    """Return six deterministic non-empty blocks, or none when too short."""

    if len(sequence) < NEAR_NEIGHBOR_BLOCK_COUNT:
        return ()
    quotient, remainder = divmod(
        len(sequence), NEAR_NEIGHBOR_BLOCK_COUNT
    )
    blocks = []
    start = 0
    for block_index in range(NEAR_NEIGHBOR_BLOCK_COUNT):
        width = quotient + (block_index < remainder)
        blocks.append(sequence[start : start + width])
        start += width
    if start != len(sequence) or any(not block for block in blocks):
        raise AssertionError("six-block partition is not exhaustive")
    return tuple(blocks)


def _block_lengths_for_sequence_length(sequence_length: int) -> Tuple[int, ...]:
    if sequence_length < NEAR_NEIGHBOR_BLOCK_COUNT:
        return ()
    quotient, remainder = divmod(
        sequence_length, NEAR_NEIGHBOR_BLOCK_COUNT
    )
    return (
        (quotient, quotient + 1)
        if remainder
        else (quotient,)
    )


def _band_cell_count(left_length: int, right_length: int, band: int) -> int:
    return sum(
        max(
            0,
            min(right_length, i + band) - max(0, i - band) + 1,
        )
        for i in range(left_length + 1)
    )


def _banded_distance_at_most(
    left: str,
    right: str,
    threshold: int,
) -> int | None:
    """Return exact Levenshtein distance when <= threshold, else ``None``."""

    if abs(len(left) - len(right)) > threshold:
        return None
    previous: Dict[int, int] = {
        j: j for j in range(0, min(len(right), threshold) + 1)
    }
    for i in range(1, len(left) + 1):
        current: Dict[int, int] = {}
        for j in range(
            max(0, i - threshold),
            min(len(right), i + threshold) + 1,
        ):
            best = threshold + 1
            if j in previous:
                best = min(best, previous[j] + 1)
            if j - 1 in current:
                best = min(best, current[j - 1] + 1)
            if j > 0 and j - 1 in previous:
                best = min(
                    best,
                    previous[j - 1] + (left[i - 1] != right[j - 1]),
                )
            if best <= threshold:
                current[j] = best
        previous = current
        if not previous:
            return None
    distance = previous.get(len(right))
    return distance if distance is not None and distance <= threshold else None


def build_near_neighbor_clusters(
    record_states: Mapping[str, Iterable[str]],
    *,
    max_sequences: int = 100_000,
    max_block_postings: int = 600_000,
    max_substring_probes: int = 50_000_000,
    max_candidate_pairs: int = 1_000_000,
    max_exact_dp_cells: int = 100_000_000,
) -> NearNeighborClusters:
    """Build exact global edit-distance-5 connected components.

    The input must contain the complete shortest-path state closure for every
    eligible record.  Every limit is fail-closed and no partial clustering is
    returned.
    """

    for name, value in (
        ("max_sequences", max_sequences),
        ("max_block_postings", max_block_postings),
        ("max_substring_probes", max_substring_probes),
        ("max_candidate_pairs", max_candidate_pairs),
        ("max_exact_dp_cells", max_exact_dp_cells),
    ):
        _validate_limit(name, value)
    if not isinstance(record_states, Mapping) or not record_states:
        raise NearNeighborError("record_states must be a non-empty mapping")

    normalized_records: Dict[str, Tuple[str, ...]] = {}
    sequence_set: set[str] = set()
    for raw_record_id, raw_states in sorted(
        record_states.items(), key=lambda item: str(item[0])
    ):
        record_id = str(raw_record_id).strip()
        if not record_id:
            raise NearNeighborError("record_states contains an empty record_id")
        states = tuple(sorted(set(raw_states)))
        if not states:
            raise NearNeighborError(f"record {record_id} has no sequence states")
        for sequence in states:
            _validate_sequence(sequence)
        normalized_records[record_id] = states
        sequence_set.update(states)

    sequences = tuple(sorted(sequence_set))
    if len(sequences) > max_sequences:
        raise NearNeighborError(
            "STOP_RULE_B0_NEAR_NEIGHBOR_COMPLEXITY: exact global clustering "
            f"has {len(sequences)} states, exceeding {max_sequences}; "
            "no approximation was emitted"
        )
    sequence_index = {
        sequence: index for index, sequence in enumerate(sequences)
    }

    block_index: Dict[Tuple[int, str], set[int]] = {}
    block_postings = 0
    for index, sequence in enumerate(sequences):
        for block in sorted(set(_balanced_blocks(sequence))):
            key = (len(block), block)
            owners = block_index.setdefault(key, set())
            if index not in owners:
                owners.add(index)
                block_postings += 1
                if block_postings > max_block_postings:
                    raise NearNeighborError(
                        "STOP_RULE_B0_NEAR_NEIGHBOR_COMPLEXITY: six-block "
                        f"index exceeded {max_block_postings} postings; "
                        "no approximation was emitted"
                    )

    candidate_pairs: set[Tuple[int, int]] = set()

    def add_candidate(left: int, right: int) -> None:
        if left == right:
            return
        pair = (left, right) if left < right else (right, left)
        if pair in candidate_pairs:
            return
        if (
            abs(len(sequences[pair[0]]) - len(sequences[pair[1]]))
            > NEAR_NEIGHBOR_EDIT_THRESHOLD
        ):
            return
        candidate_pairs.add(pair)
        if len(candidate_pairs) > max_candidate_pairs:
            raise NearNeighborError(
                "STOP_RULE_B0_NEAR_NEIGHBOR_COMPLEXITY: pigeonhole "
                f"candidate set exceeded {max_candidate_pairs} pairs; "
                "no approximation was emitted"
            )

    short_indices = [
        index
        for index, sequence in enumerate(sequences)
        if len(sequence) < NEAR_NEIGHBOR_BLOCK_COUNT
    ]
    for offset, left in enumerate(short_indices):
        for right in short_indices[offset + 1 :]:
            add_candidate(left, right)

    substring_probes = 0
    for target_index, target in enumerate(sequences):
        possible_block_lengths: set[int] = set()
        minimum_query_length = max(
            NEAR_NEIGHBOR_BLOCK_COUNT,
            len(target) - NEAR_NEIGHBOR_EDIT_THRESHOLD,
        )
        maximum_query_length = len(target) + NEAR_NEIGHBOR_EDIT_THRESHOLD
        for query_length in range(
            minimum_query_length, maximum_query_length + 1
        ):
            possible_block_lengths.update(
                _block_lengths_for_sequence_length(query_length)
            )
        for block_length in sorted(possible_block_lengths):
            if block_length > len(target):
                continue
            observed_substrings = {
                target[start : start + block_length]
                for start in range(len(target) - block_length + 1)
            }
            substring_probes += len(observed_substrings)
            if substring_probes > max_substring_probes:
                raise NearNeighborError(
                    "STOP_RULE_B0_NEAR_NEIGHBOR_COMPLEXITY: exact "
                    f"substring scan exceeded {max_substring_probes} probes; "
                    "no approximation was emitted"
                )
            for substring in sorted(observed_substrings):
                for owner in sorted(
                    block_index.get((block_length, substring), ())
                ):
                    add_candidate(owner, target_index)

    union_find = _UnionFind(len(sequences))
    qualifying_pairs: list[Tuple[int, int]] = []
    exact_dp_cells = 0
    for left, right in sorted(candidate_pairs):
        exact_dp_cells += _band_cell_count(
            len(sequences[left]),
            len(sequences[right]),
            NEAR_NEIGHBOR_EDIT_THRESHOLD,
        )
        if exact_dp_cells > max_exact_dp_cells:
            raise NearNeighborError(
                "STOP_RULE_B0_NEAR_NEIGHBOR_COMPLEXITY: exact banded "
                f"verification exceeded {max_exact_dp_cells} DP cells; "
                "no approximation was emitted"
            )
        distance = _banded_distance_at_most(
            sequences[left],
            sequences[right],
            NEAR_NEIGHBOR_EDIT_THRESHOLD,
        )
        if distance is not None:
            qualifying_pairs.append((left, right))
            union_find.union(left, right)

    indices_by_root: Dict[int, list[int]] = {}
    for index in range(len(sequences)):
        indices_by_root.setdefault(union_find.find(index), []).append(index)
    sequence_clusters: Dict[str, str] = {}
    cluster_sizes: list[int] = []
    for indices in indices_by_root.values():
        members = tuple(sequences[index] for index in sorted(indices))
        cluster_id = (
            f"near{NEAR_NEIGHBOR_EDIT_THRESHOLD}:"
            + _sha256_payload(members)
        )
        cluster_sizes.append(len(members))
        for sequence in members:
            sequence_clusters[sequence] = cluster_id

    record_clusters: Dict[str, str] = {}
    for record_id, states in normalized_records.items():
        cluster_ids = {sequence_clusters[state] for state in states}
        if len(cluster_ids) != 1:
            raise NearNeighborError(
                "record shortest-path states span multiple edit-distance-5 "
                f"clusters: {record_id}; candidate completeness cannot be "
                "certified"
            )
        record_clusters[record_id] = next(iter(cluster_ids))

    sequence_hashes = {
        sequence: hashlib.sha256(sequence.encode("utf-8")).hexdigest()
        for sequence in sequences
    }
    binding = {
        "algorithm": NEAR_NEIGHBOR_ALGORITHM_ID,
        "edit_distance_threshold": NEAR_NEIGHBOR_EDIT_THRESHOLD,
        "pigeonhole_block_count": NEAR_NEIGHBOR_BLOCK_COUNT,
        "scope": (
            "global_all_shortest_path_sequence_states_of_all_split_eligible_records"
        ),
        "label_fields_read": [],
        "candidate_generation_complete": True,
        "false_negative_guarantee": (
            "six_nonempty_source_blocks; every_distance_lte_5_pair_shares_at_"
            "least_one_untouched_block_found_by_all_substring_scan"
        ),
        "exact_verification": "banded_levenshtein",
        "sequence_count": len(sequences),
        "record_count": len(normalized_records),
        "block_posting_count": block_postings,
        "substring_probe_count": substring_probes,
        "candidate_pair_count": len(candidate_pairs),
        "qualifying_pair_count": len(qualifying_pairs),
        "cluster_count": len(indices_by_root),
        "maximum_cluster_sequence_count": max(cluster_sizes, default=0),
        "exact_dp_cell_count": exact_dp_cells,
        "sequence_universe_sha256": _sha256_payload(
            sorted(sequence_hashes.values())
        ),
        "qualifying_edges_sha256": _sha256_payload(
            [
                (
                    sequence_hashes[sequences[left]],
                    sequence_hashes[sequences[right]],
                )
                for left, right in qualifying_pairs
            ]
        ),
        "cluster_assignment_sha256": _sha256_payload(
            sorted(
                (
                    sequence_hashes[sequence],
                    cluster_id,
                )
                for sequence, cluster_id in sequence_clusters.items()
            )
        ),
        "record_assignment_sha256": _sha256_payload(
            sorted(record_clusters.items())
        ),
        "resource_limits": {
            "max_sequences": max_sequences,
            "max_block_postings": max_block_postings,
            "max_substring_probes": max_substring_probes,
            "max_candidate_pairs": max_candidate_pairs,
            "max_exact_dp_cells": max_exact_dp_cells,
        },
    }
    return NearNeighborClusters(
        binding=binding,
        record_clusters=dict(sorted(record_clusters.items())),
        sequence_clusters=dict(sorted(sequence_clusters.items())),
    )


__all__ = [
    "NEAR_NEIGHBOR_ALGORITHM_ID",
    "NEAR_NEIGHBOR_BLOCK_COUNT",
    "NEAR_NEIGHBOR_EDIT_THRESHOLD",
    "NearNeighborClusters",
    "NearNeighborError",
    "build_near_neighbor_clusters",
]
