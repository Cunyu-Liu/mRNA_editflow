from __future__ import annotations

import json
from dataclasses import is_dataclass
from itertools import product
from pathlib import Path

import pytest

from data.utr_benchmark_v2.path_states import PathStateError
from data.utr_benchmark_v2.path_states import minimum_alignment_state_closure
from data.utr_benchmark_v2.symbolic_path_states import (
    StreamingPathStateError,
)
from data.utr_benchmark_v2.symbolic_path_states import (
    minimum_alignment_state_summary,
)


WITNESS_SOURCE = (
    "CUAACUGAGAAGGGCGUAGGCGCCGUGCUUUUGCUCCCCGCGCGCUGUUUUUCUCGCUGAC"
    "UUUCAGCGGGCGGAAAAGCCUCGGCCUGCCGCCUUCCACCGUUCAUUCUAGAGCAAACAAA"
    "AAAUGUC"
)
WITNESS_CANDIDATE = (
    "CUAACUGAGAAGGGCGUAGGCGCCGUGCUUUUGCUCCCCGCGCGCUGUUUUUCUCGCGGAA"
    "AAGCCUCGGCCUGCCGCCUUCCACCGUUCAUUCUAGAGCAAACAAAAAAUGUC"
)
WITNESS_LAYER_NODE_COUNTS = (
    1,
    18,
    152,
    788,
    2_762,
    6_917,
    12_872,
    18_248,
    19_901,
    16_629,
    10_458,
    4_744,
    1_436,
    264,
    26,
    1,
)

EXACT_SUMMARY_FIELDS = (
    "minimum_edit_count",
    "minimum_alignment_count",
    "minimum_state_path_count",
    "reachable_node_count",
    "reachable_transition_count",
    "evaluated_primitive_action_count",
    "evaluated_state_dp_cell_count",
    "reachable_states_sha256",
)


def _fresh_workspace(tmp_path: Path, name: str) -> Path:
    workspace = tmp_path / name
    assert not workspace.exists()
    return workspace


def _exact_signature(summary: object) -> tuple[object, ...]:
    return tuple(getattr(summary, field) for field in EXACT_SUMMARY_FIELDS) + (
        tuple(getattr(summary, "layer_node_counts")),
        getattr(summary, "algorithm"),
        getattr(summary, "exact"),
    )


@pytest.mark.parametrize(
    ("source", "candidate"),
    (
        ("AC", "CA"),
        ("AA", "CC"),
        ("AAA", "A"),
        ("A", "AAA"),
        ("ACGU", "AGU"),
    ),
)
def test_bounded_symbolic_summary_matches_every_legacy_exact_counter(
    tmp_path: Path,
    source: str,
    candidate: str,
) -> None:
    legacy = minimum_alignment_state_closure(source, candidate)
    summary = minimum_alignment_state_summary(
        source,
        candidate,
        workspace=_fresh_workspace(
            tmp_path,
            f"parity-{source}-{candidate}",
        ),
        chunk_size=2,
    )

    assert is_dataclass(summary)
    assert summary.exact is True
    assert isinstance(summary.algorithm, str)
    assert summary.algorithm
    assert not hasattr(summary, "reachable_states")
    for field in EXACT_SUMMARY_FIELDS:
        assert getattr(summary, field) == getattr(legacy, field)
    assert sum(summary.layer_node_counts) == summary.reachable_node_count
    assert len(summary.layer_node_counts) == summary.minimum_edit_count + 1


@pytest.mark.parametrize(
    ("source", "candidate"),
    (
        ("ACACA", "ACA"),
        ("ACA", "ACACA"),
        ("AAAA", "AA"),
        ("AA", "AAAA"),
    ),
)
def test_pure_indel_symbolic_layers_match_legacy_sequence_identity_layers(
    tmp_path: Path,
    source: str,
    candidate: str,
) -> None:
    legacy = minimum_alignment_state_closure(source, candidate)
    expected_layer_counts = [0] * (legacy.minimum_edit_count + 1)
    for state in legacy.reachable_states:
        depth = abs(len(state) - len(source))
        expected_layer_counts[depth] += 1

    summary = minimum_alignment_state_summary(
        source,
        candidate,
        known_minimum_edit_count=abs(len(source) - len(candidate)),
        workspace=_fresh_workspace(
            tmp_path,
            f"pure-{source}-{candidate}",
        ),
        chunk_size=1,
    )

    assert summary.exact is True
    assert tuple(summary.layer_node_counts) == tuple(expected_layer_counts)
    assert summary.reachable_node_count == len(legacy.reachable_states)
    assert summary.reachable_states_sha256 == legacy.reachable_states_sha256


@pytest.mark.parametrize(
    ("source", "candidate"),
    (
        ("AC", "CA"),
        ("ACACA", "ACA"),
    ),
)
def test_chunk_size_cannot_change_any_exact_result(
    tmp_path: Path,
    source: str,
    candidate: str,
) -> None:
    signatures = []
    for chunk_size in (1, 2, 7, 64):
        summary = minimum_alignment_state_summary(
            source,
            candidate,
            workspace=_fresh_workspace(
                tmp_path,
                f"chunk-{source}-{candidate}-{chunk_size}",
            ),
            chunk_size=chunk_size,
        )
        signatures.append(_exact_signature(summary))

    assert signatures[1:] == signatures[:-1]


def test_exhaustive_small_streaming_closures_match_legacy(
    tmp_path: Path,
) -> None:
    endpoints = [
        "".join(bases) for length in (1, 2) for bases in product("AC", repeat=length)
    ]
    for source in endpoints:
        for candidate in endpoints:
            legacy = minimum_alignment_state_closure(source, candidate)
            summary = minimum_alignment_state_summary(
                source,
                candidate,
                workspace=_fresh_workspace(
                    tmp_path,
                    f"exhaustive-{source}-{candidate}",
                ),
                chunk_size=3,
            )
            for field in EXACT_SUMMARY_FIELDS:
                assert getattr(summary, field) == getattr(legacy, field)


@pytest.mark.parametrize(
    ("source", "candidate", "limits"),
    (
        ("AA", "CC", {"max_reachable_states": 3}),
        ("AAA", "A", {"max_neighbor_expansions": 2}),
        ("AC", "CA", {"max_state_dp_cells": 1}),
        ("ACGU", "UGCA", {"max_dag_cells": 1}),
        ("AA", "CC", {"max_spill_bytes": 1}),
    ),
)
def test_symbolic_limits_fail_closed_without_returning_partial_summary(
    tmp_path: Path,
    source: str,
    candidate: str,
    limits: dict[str, int],
) -> None:
    assert issubclass(StreamingPathStateError, PathStateError)
    workspace = _fresh_workspace(
        tmp_path,
        f"limit-{source}-{candidate}-{sorted(limits)}",
    )
    with pytest.raises(
        StreamingPathStateError,
        match="STOP_RULE_B0_PATH_STATE_COMPLEXITY|exceeding",
    ):
        minimum_alignment_state_summary(
            source,
            candidate,
            workspace=workspace,
            chunk_size=2,
            **limits,
        )
    assert (workspace / "failure.json").is_file()
    assert '"status": "FAILED_WITH_EVIDENCE"' in (workspace / "status.json").read_text(
        encoding="utf-8"
    )


def test_chunk_size_and_workspace_ownership_fail_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(StreamingPathStateError, match="chunk_size"):
        minimum_alignment_state_summary(
            "AC",
            "CA",
            workspace=_fresh_workspace(tmp_path, "invalid-chunk"),
            chunk_size=0,
        )

    existing_empty = tmp_path / "existing-empty"
    existing_empty.mkdir()
    with pytest.raises(StreamingPathStateError, match="workspace"):
        minimum_alignment_state_summary(
            "AC",
            "CA",
            workspace=existing_empty,
            chunk_size=1,
        )

    existing_nonempty = tmp_path / "existing-nonempty"
    existing_nonempty.mkdir()
    (existing_nonempty / "foreign.txt").write_text(
        "must not be overwritten\n",
        encoding="utf-8",
    )
    with pytest.raises(StreamingPathStateError, match="workspace"):
        minimum_alignment_state_summary(
            "AC",
            "CA",
            workspace=existing_nonempty,
            chunk_size=1,
        )
    assert (existing_nonempty / "foreign.txt").read_text(
        encoding="utf-8"
    ) == "must not be overwritten\n"

    invalid_merge = _fresh_workspace(tmp_path, "invalid-open-chunks")
    with pytest.raises(StreamingPathStateError, match="max_open_chunks"):
        minimum_alignment_state_summary(
            "AC",
            "CA",
            workspace=invalid_merge,
            max_open_chunks=1,
        )
    assert not invalid_merge.exists()


def test_zero_edit_terminal_workspace_bytes_are_exact_and_guarded(
    tmp_path: Path,
) -> None:
    completed_workspace = _fresh_workspace(tmp_path, "zero-edit-complete")
    summary = minimum_alignment_state_summary(
        "A",
        "A",
        workspace=completed_workspace,
    )
    observed_bytes = sum(
        path.stat().st_size for path in completed_workspace.rglob("*") if path.is_file()
    )
    assert summary.spill_bytes == observed_bytes

    with pytest.raises(
        StreamingPathStateError,
        match="STOP_RULE_B0_PATH_STATE_COMPLEXITY",
    ):
        minimum_alignment_state_summary(
            "A",
            "A",
            workspace=_fresh_workspace(tmp_path, "zero-edit-stop"),
            max_spill_bytes=observed_bytes - 1,
        )


def test_progress_callback_runs_inside_a_record_and_can_fail_closed(
    tmp_path: Path,
) -> None:
    observed: list[dict] = []

    def stop_during_expansion(progress: dict) -> None:
        observed.append(dict(progress))
        if progress["phase"] == "expand_layer":
            raise StreamingPathStateError("SAFE_RESOURCE_PAUSE: fixture")

    workspace = _fresh_workspace(tmp_path, "callback-stop")
    with pytest.raises(StreamingPathStateError, match="SAFE_RESOURCE_PAUSE"):
        minimum_alignment_state_summary(
            "AAAA",
            "AA",
            workspace=workspace,
            max_reachable_states=100,
            progress_callback=stop_during_expansion,
            progress_interval=1,
        )

    assert any(item["phase"] == "alignment_statistics_complete" for item in observed)
    assert any(item["phase"] == "expand_layer" for item in observed)
    assert (
        json.loads((workspace / "status.json").read_text(encoding="utf-8"))["status"]
        == "FAILED_WITH_EVIDENCE"
    )
    assert (workspace / "failure.json").is_file()


def test_real_witness_streams_exact_summary_without_materializing_states(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        StreamingPathStateError,
        match="STOP_RULE_B0_PATH_STATE_COMPLEXITY",
    ):
        minimum_alignment_state_summary(
            WITNESS_SOURCE,
            WITNESS_CANDIDATE,
            known_minimum_edit_count=15,
            workspace=_fresh_workspace(tmp_path, "witness-default-stop"),
            chunk_size=257,
        )

    summary = minimum_alignment_state_summary(
        WITNESS_SOURCE,
        WITNESS_CANDIDATE,
        known_minimum_edit_count=15,
        workspace=_fresh_workspace(tmp_path, "witness-exact"),
        chunk_size=257,
        max_reachable_states=100_000,
    )

    assert is_dataclass(summary)
    assert summary.exact is True
    assert not hasattr(summary, "reachable_states")
    assert summary.minimum_edit_count == 15
    assert summary.minimum_alignment_count == 2_340
    assert tuple(summary.layer_node_counts) == WITNESS_LAYER_NODE_COUNTS
    assert summary.reachable_node_count == 95_217
    assert summary.reachable_transition_count == 751_771
    assert summary.minimum_state_path_count == 3_934_510_691_993
    assert summary.evaluated_primitive_action_count == 1_205_477
    assert summary.evaluated_state_dp_cell_count == 0
    assert (
        summary.reachable_states_sha256
        == "900076096ad75979a1b592b6d14fd7647dfe54c39b4cee80a053937de9411332"
    )
