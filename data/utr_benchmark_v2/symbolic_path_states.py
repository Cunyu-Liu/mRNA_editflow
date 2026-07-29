"""Isolated exact symbolic/streaming prototype for B0 capacity diagnosis.

This module deliberately does not replace or modify the production
``minimum_alignment_state_closure`` entrypoint.  It preserves the same
all-shortest dynamic-edit, sequence-identity, transition-collapse, and path
count semantics while moving layer materialization to caller-owned external
storage.

The pure-indel certificate counts the exact subsequence language by shortest
path layer.  It certifies only state/layer counts.  Edges, state paths,
primitive actions, and the state digest are still computed by the exact
external-memory traversal.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TextIO

from data.utr_benchmark_v2.path_states import (
    DEFAULT_MAX_REACHABLE_STATES,
)
from data.utr_benchmark_v2.path_states import PathStateError
from data.utr_benchmark_v2.path_states import (
    _distance_reducing_neighbors,
)
from data.utr_benchmark_v2.path_states import _one_primitive_edit_apart
from data.utr_benchmark_v2.path_states import minimum_alignment_statistics


SYMBOLIC_PURE_INDEL_ALGORITHM_ID = (
    "distinct_subsequence_next_occurrence_dfa_layer_count_v1"
)
STREAMING_SUMMARY_ALGORITHM_ID = (
    "external_sort_merge_all_shortest_dynamic_edit_state_summary_v1"
)
DEFAULT_MAX_SPILL_BYTES = 20_000_000_000
DEFAULT_MAX_OPEN_CHUNKS = 128


class StreamingPathStateError(PathStateError):
    """Raised when the isolated exact prototype cannot finish safely."""


@dataclass(frozen=True)
class PureIndelStateCertificate:
    """Exact state-language node counts for a pure-indel endpoint pair."""

    source_sequence_sha256: str
    candidate_sequence_sha256: str
    minimum_edit_count: int
    layer_node_counts: tuple[int, ...]
    reachable_node_count: int
    proof_sha256: str
    algorithm: str = SYMBOLIC_PURE_INDEL_ALGORITHM_ID
    exact: bool = True
    certifies_edges: bool = False
    certifies_state_paths: bool = False


@dataclass(frozen=True)
class MinimumAlignmentStateSummary:
    """Exact closure summary without returning an in-memory state tuple."""

    source_sequence: str
    candidate_sequence: str
    minimum_edit_count: int
    minimum_alignment_count: int
    minimum_state_path_count: int
    reachable_node_count: int
    reachable_transition_count: int
    evaluated_primitive_action_count: int
    evaluated_state_dp_cell_count: int
    reachable_states_sha256: str
    layer_node_counts: tuple[int, ...]
    layer_files: tuple[str, ...]
    layer_files_sha256: tuple[str, ...]
    spill_file_count: int
    spill_bytes: int
    symbolic_certificate_sha256: str | None
    algorithm: str = STREAMING_SUMMARY_ALGORITHM_ID
    exact: bool = True
    state_identity: str = "full_sequence_string"
    approximation_emitted: bool = False
    production_gate_changed: bool = False


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )


def _atomic_json(path: Path, value: object) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("x", encoding="utf-8", newline="") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _validate_positive_limit(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise StreamingPathStateError(f"{name} must be a positive integer")


def _next_occurrence_table(sequence: str) -> list[dict[str, int]]:
    next_positions = [
        {base: len(sequence) for base in "ACGU"} for _ in range(len(sequence) + 1)
    ]
    for position in range(len(sequence) - 1, -1, -1):
        next_positions[position] = next_positions[position + 1].copy()
        next_positions[position][sequence[position]] = position
    return next_positions


def pure_indel_state_certificate(
    source: str,
    candidate: str,
    *,
    known_minimum_edit_count: int | None = None,
    max_dag_cells: int = 1_000_000,
) -> PureIndelStateCertificate:
    """Count the exact pure-indel state language by execution depth.

    Every distinct accepted string follows one path through the
    next-occurrence automaton using its unique leftmost embedding in the
    longer endpoint.  A second deterministic state tracks the greedily matched
    prefix of the shorter endpoint, so acceptance is equivalent to containing
    that endpoint as a subsequence.
    """

    expected_distance = abs(len(source) - len(candidate))
    if (
        known_minimum_edit_count is not None
        and known_minimum_edit_count != expected_distance
    ):
        raise StreamingPathStateError(
            "pure-indel certificate requires edit distance equal to the "
            "endpoint length difference"
        )
    try:
        statistics = minimum_alignment_statistics(
            source,
            candidate,
            known_minimum_edit_count=expected_distance,
            max_dag_cells=max_dag_cells,
        )
    except PathStateError as error:
        raise StreamingPathStateError(str(error)) from error
    if statistics.minimum_edit_count != expected_distance:
        raise StreamingPathStateError(
            "endpoint pair is not a pure-indel shortest-path problem"
        )
    if source == candidate:
        layer_counts = (1,)
    else:
        longer, shorter = (
            (source, candidate) if len(source) > len(candidate) else (candidate, source)
        )
        next_positions = _next_occurrence_table(longer)
        # (leftmost-embedding cursor, greedily matched shorter prefix) -> count
        current: dict[tuple[int, int], int] = {(0, 0): 1}
        accepted_by_length = [0] * (len(longer) + 1)
        for length in range(1, len(longer) + 1):
            following: dict[tuple[int, int], int] = defaultdict(int)
            for (cursor, matched), count in current.items():
                for base in "ACGU":
                    occurrence = next_positions[cursor][base]
                    if occurrence == len(longer):
                        continue
                    next_matched = matched
                    if matched < len(shorter) and base == shorter[matched]:
                        next_matched += 1
                    following[(occurrence + 1, next_matched)] += count
            current = dict(following)
            accepted_by_length[length] = sum(
                count
                for (_cursor, matched), count in current.items()
                if matched == len(shorter)
            )

        layer_values = [0] * (expected_distance + 1)
        for length in range(len(shorter), len(longer) + 1):
            depth = (
                len(source) - length
                if len(source) > len(candidate)
                else length - len(source)
            )
            layer_values[depth] = accepted_by_length[length]
        layer_counts = tuple(layer_values)

    if (
        len(layer_counts) != expected_distance + 1
        or layer_counts[0] != 1
        or layer_counts[-1] != 1
        or any(count < 1 for count in layer_counts)
    ):
        raise StreamingPathStateError(
            "pure-indel symbolic certificate violated endpoint/layer invariants"
        )
    payload = {
        "algorithm": SYMBOLIC_PURE_INDEL_ALGORITHM_ID,
        "candidate_sequence_sha256": _sha256_bytes(candidate.encode("utf-8")),
        "layer_node_counts": layer_counts,
        "minimum_edit_count": expected_distance,
        "source_sequence_sha256": _sha256_bytes(source.encode("utf-8")),
    }
    return PureIndelStateCertificate(
        source_sequence_sha256=payload["source_sequence_sha256"],
        candidate_sequence_sha256=payload["candidate_sequence_sha256"],
        minimum_edit_count=expected_distance,
        layer_node_counts=layer_counts,
        reachable_node_count=sum(layer_counts),
        proof_sha256=_canonical_sha256(payload),
    )


def _write_layer(path: Path, rows: Iterable[tuple[str, int]]) -> int:
    partial = path.with_name(path.name + ".partial")
    count = 0
    with partial.open("x", encoding="utf-8", newline="") as handle:
        for sequence, path_count in rows:
            handle.write(f"{sequence}\t{path_count}\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    return count


def _iter_layer(path: Path) -> Iterator[tuple[str, int]]:
    previous: str | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            sequence, separator, raw_count = raw_line.rstrip("\n").partition("\t")
            if (
                not separator
                or not sequence
                or not raw_count.isdigit()
                or int(raw_count) < 1
            ):
                raise StreamingPathStateError(
                    f"invalid layer row at {path}:{line_number}"
                )
            if previous is not None and sequence <= previous:
                raise StreamingPathStateError(
                    f"layer is not strictly sorted and unique: {path}"
                )
            previous = sequence
            yield sequence, int(raw_count)


def _write_contribution_chunk(
    path: Path,
    rows: Sequence[tuple[str, int]],
) -> int:
    ordered = sorted(rows, key=lambda row: row[0])
    with path.open("x", encoding="utf-8", newline="") as handle:
        for sequence, contribution in ordered:
            handle.write(f"{sequence}\t{contribution}\n")
        handle.flush()
    return path.stat().st_size


def _iter_contributions(path: Path) -> Iterator[tuple[str, int]]:
    previous: str | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            sequence, separator, raw_count = raw_line.rstrip("\n").partition("\t")
            if (
                not separator
                or not sequence
                or not raw_count.isdigit()
                or int(raw_count) < 1
            ):
                raise StreamingPathStateError(
                    f"invalid contribution row at {path}:{line_number}"
                )
            if previous is not None and sequence < previous:
                raise StreamingPathStateError(
                    f"contribution chunk is not sorted: {path}"
                )
            previous = sequence
            yield sequence, int(raw_count)


def _merge_contribution_chunks(
    chunks: Sequence[Path],
    output: Path,
    *,
    prior_node_count: int,
    max_reachable_states: int,
) -> int:
    if not chunks:
        raise StreamingPathStateError(
            "shortest-action closure lost every path before the endpoint"
        )
    iterators = [_iter_contributions(path) for path in chunks]
    merged = heapq.merge(*iterators, key=lambda row: row[0])
    partial = output.with_name(output.name + ".partial")
    node_count = 0
    current_sequence: str | None = None
    current_count = 0
    with partial.open("x", encoding="utf-8", newline="") as handle:
        for sequence, contribution in merged:
            if current_sequence is None:
                current_sequence = sequence
                current_count = contribution
                continue
            if sequence == current_sequence:
                current_count += contribution
                continue
            node_count += 1
            if prior_node_count + node_count > max_reachable_states:
                raise StreamingPathStateError(
                    "STOP_RULE_B0_PATH_STATE_COMPLEXITY: exact "
                    f"shortest-action closure exceeded "
                    f"{max_reachable_states} reachable states; "
                    "no approximation was emitted"
                )
            handle.write(f"{current_sequence}\t{current_count}\n")
            current_sequence = sequence
            current_count = contribution
        if current_sequence is not None:
            node_count += 1
            if prior_node_count + node_count > max_reachable_states:
                raise StreamingPathStateError(
                    "STOP_RULE_B0_PATH_STATE_COMPLEXITY: exact "
                    f"shortest-action closure exceeded "
                    f"{max_reachable_states} reachable states; "
                    "no approximation was emitted"
                )
            handle.write(f"{current_sequence}\t{current_count}\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, output)
    return node_count


def _merge_contribution_runs(
    chunks: Sequence[Path],
    output: Path,
) -> int:
    """Aggregate sorted contribution chunks into one sorted reusable run."""

    if not chunks:
        raise StreamingPathStateError("cannot merge an empty chunk group")
    iterators = [_iter_contributions(path) for path in chunks]
    merged = heapq.merge(*iterators, key=lambda row: row[0])
    current_sequence: str | None = None
    current_count = 0
    with output.open("x", encoding="utf-8", newline="") as handle:
        for sequence, contribution in merged:
            if current_sequence is None:
                current_sequence = sequence
                current_count = contribution
                continue
            if sequence == current_sequence:
                current_count += contribution
                continue
            handle.write(f"{current_sequence}\t{current_count}\n")
            current_sequence = sequence
            current_count = contribution
        if current_sequence is not None:
            handle.write(f"{current_sequence}\t{current_count}\n")
        handle.flush()
    return output.stat().st_size


def _global_state_digest(layer_paths: Sequence[Path]) -> str:
    iterators = (
        (sequence for sequence, _count in _iter_layer(path)) for path in layer_paths
    )
    digest = hashlib.sha256()
    previous: str | None = None
    for sequence in heapq.merge(*iterators):
        if previous is not None and sequence == previous:
            raise StreamingPathStateError(
                "a shortest edit path revisited a sequence state across layers"
            )
        digest.update(sequence.encode("utf-8"))
        digest.update(b"\n")
        previous = sequence
    return digest.hexdigest()


def _workspace_bytes(workspace: Path) -> int:
    return sum(path.stat().st_size for path in workspace.rglob("*") if path.is_file())


def minimum_alignment_state_summary(
    source: str,
    candidate: str,
    *,
    workspace: Path,
    known_minimum_edit_count: int | None = None,
    max_dag_cells: int = 1_000_000,
    max_reachable_states: int = DEFAULT_MAX_REACHABLE_STATES,
    max_neighbor_expansions: int = 5_000_000,
    max_state_dp_cells: int = 50_000_000,
    max_spill_bytes: int = DEFAULT_MAX_SPILL_BYTES,
    max_open_chunks: int = DEFAULT_MAX_OPEN_CHUNKS,
    chunk_size: int = 50_000,
) -> MinimumAlignmentStateSummary:
    """Return an exact external-memory closure summary.

    ``workspace`` must not exist.  All completed layers and any failure
    evidence are retained there.  This function never returns a partial
    summary and never changes a production resource limit.
    """

    for name, value in (
        ("max_dag_cells", max_dag_cells),
        ("max_reachable_states", max_reachable_states),
        ("max_neighbor_expansions", max_neighbor_expansions),
        ("max_state_dp_cells", max_state_dp_cells),
        ("max_spill_bytes", max_spill_bytes),
        ("max_open_chunks", max_open_chunks),
        ("chunk_size", chunk_size),
    ):
        _validate_positive_limit(name, value)
    workspace = Path(workspace)
    if workspace.exists():
        raise StreamingPathStateError(f"workspace must not already exist: {workspace}")
    if not workspace.parent.exists():
        raise StreamingPathStateError(
            f"workspace parent does not exist: {workspace.parent}"
        )
    workspace.mkdir()
    layers_dir = workspace / "layers"
    spills_dir = workspace / "spills"
    layers_dir.mkdir()
    spills_dir.mkdir()
    _atomic_json(
        workspace / "status.json",
        {
            "algorithm": STREAMING_SUMMARY_ALGORITHM_ID,
            "exact": False,
            "production_gate_changed": False,
            "status": "RUNNING",
        },
    )

    try:
        try:
            statistics = minimum_alignment_statistics(
                source,
                candidate,
                known_minimum_edit_count=known_minimum_edit_count,
                max_dag_cells=max_dag_cells,
            )
        except PathStateError as error:
            raise StreamingPathStateError(str(error)) from error
        minimum_edit_count = statistics.minimum_edit_count
        symbolic_certificate: PureIndelStateCertificate | None = None
        if minimum_edit_count == abs(len(source) - len(candidate)):
            symbolic_certificate = pure_indel_state_certificate(
                source,
                candidate,
                known_minimum_edit_count=minimum_edit_count,
                max_dag_cells=max_dag_cells,
            )
            _atomic_json(
                workspace / "symbolic_certificate.json",
                asdict(symbolic_certificate),
            )

        layer_paths = [layers_dir / "layer-00000.tsv"]
        _write_layer(layer_paths[0], ((source, 1),))
        layer_counts = [1]
        total_nodes = 1
        transition_count = 0
        primitive_action_count = 0
        state_dp_cell_count = 0
        spill_paths: list[Path] = []
        current_layer = layer_paths[0]
        accounted_bytes = _workspace_bytes(workspace)

        for depth in range(1, minimum_edit_count):
            remaining_before_edit = minimum_edit_count - depth + 1
            contribution_buffer: list[tuple[str, int]] = []
            depth_chunks: list[Path] = []
            chunk_index = 0

            def flush_chunk() -> None:
                nonlocal accounted_bytes, chunk_index
                if not contribution_buffer:
                    return
                chunk_path = spills_dir / (
                    f"depth-{depth:05d}-chunk-{chunk_index:06d}.tsv"
                )
                written_bytes = _write_contribution_chunk(
                    chunk_path,
                    contribution_buffer,
                )
                accounted_bytes += written_bytes
                contribution_buffer.clear()
                depth_chunks.append(chunk_path)
                spill_paths.append(chunk_path)
                chunk_index += 1
                if accounted_bytes > max_spill_bytes:
                    raise StreamingPathStateError(
                        "STOP_RULE_B0_PATH_STATE_COMPLEXITY: exact "
                        f"external-memory closure exceeded "
                        f"{max_spill_bytes} spill bytes; "
                        "no approximation was emitted"
                    )

            for state, state_path_count in _iter_layer(current_layer):
                try:
                    (
                        neighbors,
                        evaluated_actions,
                        evaluated_cells,
                    ) = _distance_reducing_neighbors(
                        state,
                        candidate,
                        remaining_before_edit,
                    )
                except PathStateError as error:
                    raise StreamingPathStateError(str(error)) from error
                primitive_action_count += evaluated_actions
                if primitive_action_count > max_neighbor_expansions:
                    raise StreamingPathStateError(
                        "STOP_RULE_B0_PATH_STATE_COMPLEXITY: exact "
                        f"shortest-action closure exceeded "
                        f"{max_neighbor_expansions} evaluated primitive "
                        "actions; no approximation was emitted"
                    )
                state_dp_cell_count += evaluated_cells
                if state_dp_cell_count > max_state_dp_cells:
                    raise StreamingPathStateError(
                        "STOP_RULE_B0_PATH_STATE_COMPLEXITY: exact "
                        f"shortest-action closure exceeded "
                        f"{max_state_dp_cells} state DP cells; "
                        "no approximation was emitted"
                    )
                transition_count += len(neighbors)
                for neighbor in neighbors:
                    contribution_buffer.append((neighbor, state_path_count))
                    if len(contribution_buffer) >= chunk_size:
                        flush_chunk()
            flush_chunk()

            merge_pass = 0
            while len(depth_chunks) > max_open_chunks:
                next_runs: list[Path] = []
                for group_index, offset in enumerate(
                    range(0, len(depth_chunks), max_open_chunks)
                ):
                    run_path = spills_dir / (
                        f"depth-{depth:05d}-pass-{merge_pass:03d}-"
                        f"run-{group_index:06d}.tsv"
                    )
                    written_bytes = _merge_contribution_runs(
                        depth_chunks[offset : offset + max_open_chunks],
                        run_path,
                    )
                    accounted_bytes += written_bytes
                    spill_paths.append(run_path)
                    next_runs.append(run_path)
                    if accounted_bytes > max_spill_bytes:
                        raise StreamingPathStateError(
                            "STOP_RULE_B0_PATH_STATE_COMPLEXITY: exact "
                            f"external-memory closure exceeded "
                            f"{max_spill_bytes} spill bytes; "
                            "no approximation was emitted"
                        )
                depth_chunks = next_runs
                merge_pass += 1

            next_layer = layers_dir / f"layer-{depth:05d}.tsv"
            next_count = _merge_contribution_chunks(
                depth_chunks,
                next_layer,
                prior_node_count=total_nodes,
                max_reachable_states=max_reachable_states,
            )
            layer_paths.append(next_layer)
            layer_counts.append(next_count)
            total_nodes += next_count
            accounted_bytes += next_layer.stat().st_size
            if accounted_bytes > max_spill_bytes:
                raise StreamingPathStateError(
                    "STOP_RULE_B0_PATH_STATE_COMPLEXITY: exact "
                    f"external-memory closure exceeded "
                    f"{max_spill_bytes} spill bytes; "
                    "no approximation was emitted"
                )
            current_layer = next_layer

        if minimum_edit_count == 0:
            minimum_state_path_count = 1
        else:
            if total_nodes + 1 > max_reachable_states:
                raise StreamingPathStateError(
                    "STOP_RULE_B0_PATH_STATE_COMPLEXITY: exact "
                    f"shortest-action closure exceeded "
                    f"{max_reachable_states} reachable states; "
                    "no approximation was emitted"
                )
            minimum_state_path_count = 0
            penultimate_count = 0
            for state, state_path_count in _iter_layer(current_layer):
                if not _one_primitive_edit_apart(state, candidate):
                    raise StreamingPathStateError(
                        "penultimate state is not one edit from the candidate"
                    )
                minimum_state_path_count += state_path_count
                penultimate_count += 1
            transition_count += penultimate_count
            final_layer = layers_dir / f"layer-{minimum_edit_count:05d}.tsv"
            _write_layer(
                final_layer,
                ((candidate, minimum_state_path_count),),
            )
            accounted_bytes += final_layer.stat().st_size
            if accounted_bytes > max_spill_bytes:
                raise StreamingPathStateError(
                    "STOP_RULE_B0_PATH_STATE_COMPLEXITY: exact "
                    f"external-memory closure exceeded "
                    f"{max_spill_bytes} spill bytes; "
                    "no approximation was emitted"
                )
            layer_paths.append(final_layer)
            layer_counts.append(1)
            total_nodes += 1

        if symbolic_certificate is not None and (
            tuple(layer_counts) != symbolic_certificate.layer_node_counts
            or total_nodes != symbolic_certificate.reachable_node_count
        ):
            raise StreamingPathStateError(
                "symbolic pure-indel certificate disagrees with streamed "
                "sequence-identity layer counts"
            )

        state_digest = _global_state_digest(layer_paths)
        layer_hashes = tuple(_sha256_file(path) for path in layer_paths)
        summary = MinimumAlignmentStateSummary(
            source_sequence=source,
            candidate_sequence=candidate,
            minimum_edit_count=minimum_edit_count,
            minimum_alignment_count=statistics.minimum_alignment_count,
            minimum_state_path_count=minimum_state_path_count,
            reachable_node_count=total_nodes,
            reachable_transition_count=transition_count,
            evaluated_primitive_action_count=primitive_action_count,
            evaluated_state_dp_cell_count=state_dp_cell_count,
            reachable_states_sha256=state_digest,
            layer_node_counts=tuple(layer_counts),
            layer_files=tuple(
                path.relative_to(workspace).as_posix() for path in layer_paths
            ),
            layer_files_sha256=layer_hashes,
            spill_file_count=len(spill_paths),
            spill_bytes=_workspace_bytes(workspace),
            symbolic_certificate_sha256=(
                symbolic_certificate.proof_sha256
                if symbolic_certificate is not None
                else None
            ),
        )
        _atomic_json(workspace / "summary.json", asdict(summary))
        _atomic_json(
            workspace / "status.json",
            {
                "algorithm": STREAMING_SUMMARY_ALGORITHM_ID,
                "exact": True,
                "production_gate_changed": False,
                "status": "EXACT_COMPLETED",
                "summary_sha256": _sha256_file(workspace / "summary.json"),
            },
        )
        return summary
    except Exception as error:
        wrapped = (
            error
            if isinstance(error, StreamingPathStateError)
            else StreamingPathStateError(f"exact streaming prototype failed: {error}")
        )
        _atomic_json(
            workspace / "failure.json",
            {
                "error_type": type(error).__name__,
                "message": str(error),
                "no_approximation_emitted": True,
                "partial_result_returned": False,
            },
        )
        _atomic_json(
            workspace / "status.json",
            {
                "algorithm": STREAMING_SUMMARY_ALGORITHM_ID,
                "exact": False,
                "production_gate_changed": False,
                "status": "FAILED_WITH_EVIDENCE",
            },
        )
        raise wrapped from error


__all__ = [
    "DEFAULT_MAX_OPEN_CHUNKS",
    "DEFAULT_MAX_SPILL_BYTES",
    "MinimumAlignmentStateSummary",
    "PureIndelStateCertificate",
    "STREAMING_SUMMARY_ALGORITHM_ID",
    "SYMBOLIC_PURE_INDEL_ALGORITHM_ID",
    "StreamingPathStateError",
    "minimum_alignment_state_summary",
    "pure_indel_state_certificate",
]
