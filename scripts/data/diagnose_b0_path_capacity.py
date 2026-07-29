#!/usr/bin/env python3
"""Build an isolated, replayable, label-free B0 path-capacity diagnostic.

This entrypoint is intentionally separate from every formal B0 driver.  It
does not change a production gate, does not build a split, and cannot emit B0
acceptance.  Exact record closures use the external-memory prototype in
``symbolic_path_states.py``.  A resource stop is retained as evidence and is
never replaced by a sampled or single-traceback result.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import importlib.metadata
import json
import os
import platform
import resource
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import jsonschema

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.utr_benchmark_v2.near_neighbors import (
    NEAR_NEIGHBOR_ALGORITHM_ID,
)
from data.utr_benchmark_v2.path_states import ALGORITHM_ID
from data.utr_benchmark_v2.path_states import (
    PRIMITIVE_ACTION_EVALUATION_SCOPE,
)
from data.utr_benchmark_v2.path_states import STATE_CLOSURE_SCOPE
from data.utr_benchmark_v2.path_states import STATE_PATH_COUNT_SCOPE
from data.utr_benchmark_v2.path_states import minimum_alignment_statistics
from data.utr_benchmark_v2.split_graph import REGIONS
from data.utr_benchmark_v2.split_graph import record_structural_sha256
from data.utr_benchmark_v2.split_graph import (
    select_split_eligible_records,
)
from data.utr_benchmark_v2.symbolic_path_states import (
    StreamingPathStateError,
)
from data.utr_benchmark_v2.symbolic_path_states import (
    minimum_alignment_state_summary,
)


GOAL_CONTRACT_SHA256 = (
    "c3dc5875868d847b8519fee40b14c43b65e4c5948dc5c3b98101ca61a5671dd5"
)
WITNESS_RECORD_ID = "GSE217518:record:025e56d3b64660abb559dcbd"
WITNESS_CANONICAL_JSONL_LINE = 39_913
WITNESS_SOURCE = (
    "CUAACUGAGAAGGGCGUAGGCGCCGUGCUUUUGCUCCCCGCGCGCUGUUUUUCUCGCUGAC"
    "UUUCAGCGGGCGGAAAAGCCUCGGCCUGCCGCCUUCCACCGUUCAUUCUAGAGCAAACAAA"
    "AAAUGUC"
)
WITNESS_CANDIDATE = (
    "CUAACUGAGAAGGGCGUAGGCGCCGUGCUUUUGCUCCCCGCGCGCUGUUUUUCUCGCGGAA"
    "AAGCCUCGGCCUGCCGCCUUCCACCGUUCAUUCUAGAGCAAACAAAAAAUGUC"
)
WITNESS_EXPECTED = {
    "reachable_node_count": 95_217,
    "reachable_transition_count": 751_771,
    "minimum_state_path_count": 3_934_510_691_993,
    "evaluated_primitive_action_count": 1_205_477,
    "evaluated_state_dp_cell_count": 0,
    "reachable_states_sha256": (
        "900076096ad75979a1b592b6d14fd7647dfe54c39b4cee80a053937de9411332"
    ),
}
FROZEN_B0_LIMITS = {
    "max_dag_cells": 1_000_000,
    "max_reachable_states": 50_000,
    "max_neighbor_expansions": 5_000_000,
    "max_state_dp_cells": 50_000_000,
    "max_sequences": 100_000,
    "max_block_postings": 600_000,
    "max_substring_probes": 50_000_000,
    "max_candidate_pairs": 1_000_000,
    "max_exact_dp_cells": 100_000_000,
}
FORBIDDEN_TOP_LEVEL_LABEL_FIELDS = frozenset(
    {
        "label",
        "label_value",
        "labels",
        "final_label",
        "measured_label",
        "measured_value",
        "target_value",
        "activity_value",
    }
)
CONFIG_SAFETY_FIELDS = frozenset(
    {
        "max_dag_cells_per_record",
        "max_reachable_states_per_record",
        "max_neighbor_expansions_per_record",
        "max_state_dp_cells_per_record",
        "max_spill_bytes_per_record",
        "chunk_size",
        "max_open_chunks",
        "minimum_free_bytes",
        "max_rss_bytes",
        "max_wall_seconds",
        "max_spill_bytes",
        "heartbeat_seconds",
    }
)


class CapacityDiagnosticError(ValueError):
    """Raised when diagnostic provenance or exactness cannot be certified."""


class SafeCapacityPause(CapacityDiagnosticError):
    """Raised for an audited resource pause that may be replayed fresh."""


@dataclass(frozen=True)
class StructuralSelection:
    """Label-free, fully accounted structural capacity selection."""

    records: tuple[Mapping[str, Any], ...]
    eligible_endpoints: tuple[str, ...]
    source_record_count: int
    split_eligible_record_count: int
    selected_record_count: int
    excluded_record_count: int
    record_ids_sha256: str
    selected_record_ids_sha256: str
    structural_store_sha256: str
    exclusion_reason_counts: Mapping[str, int]
    label_fields_read: tuple[str, ...] = ()
    canonical_label_store_opened: bool = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def _record_ids_sha256(values: Sequence[str]) -> str:
    ordered = sorted(values)
    if len(ordered) != len(set(ordered)):
        raise CapacityDiagnosticError("structural store contains duplicate record IDs")
    payload = (("\n".join(ordered) + "\n") if ordered else "").encode("utf-8")
    return _sha256_bytes(payload)


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CapacityDiagnosticError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise CapacityDiagnosticError(f"JSON root must be an object: {path}")
    return value


def _write_json_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _write_text_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
        )
        handle.flush()


def _artifact_ref(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    rendered = str(resolved)
    if root is not None:
        try:
            rendered = resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return {
        "path": rendered,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _claim_boundary() -> dict[str, Any]:
    return {
        "formal_b0_attempt_started": False,
        "b0_gate_values_changed": False,
        "b0_phase_acceptance_claimed": False,
        "scientific_result_claimed": False,
        "budget_change_authorized": False,
        "approximation_emitted": False,
        "allowed_claim": "CAPACITY_DIAGNOSTIC_ONLY",
    }


def _create_exclusive_run_root(path: Path) -> None:
    if path.exists():
        raise CapacityDiagnosticError(f"diagnostic output root already exists: {path}")
    if not path.parent.exists():
        raise CapacityDiagnosticError(
            f"diagnostic output parent does not exist: {path.parent}"
        )
    path.mkdir()


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    expected_top = {
        "schema_version",
        "selection",
        "frozen_b0_limits",
        "diagnostic_safety_limits",
    }
    if set(config) != expected_top:
        raise CapacityDiagnosticError(
            "capacity config has missing or unexpected top-level fields"
        )
    if config["schema_version"] != "b0_capacity_config.v1":
        raise CapacityDiagnosticError("capacity config schema version is invalid")
    if dict(config["frozen_b0_limits"]) != FROZEN_B0_LIMITS:
        raise CapacityDiagnosticError(
            "frozen B0 limits differ from the unchanged production gates"
        )
    selection = config["selection"]
    if (
        not isinstance(selection, Mapping)
        or set(selection) != {"regions", "minimum_edit_distance"}
        or list(selection["regions"]) != list(REGIONS)
        or selection["minimum_edit_distance"] != 2
    ):
        raise CapacityDiagnosticError(
            "selection must be the frozen two-region edit-distance>=2 census"
        )
    safety = config["diagnostic_safety_limits"]
    if not isinstance(safety, Mapping) or set(safety) != CONFIG_SAFETY_FIELDS:
        raise CapacityDiagnosticError(
            "diagnostic safety limit scope is incomplete or ambiguous"
        )
    for name, value in safety.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise CapacityDiagnosticError(
                f"diagnostic safety limit {name} must be a positive integer"
            )
    if safety["heartbeat_seconds"] < 300:
        raise CapacityDiagnosticError(
            "diagnostic heartbeat must follow the low-frequency >=300s cadence"
        )
    if safety["max_open_chunks"] < 2:
        raise CapacityDiagnosticError("max_open_chunks must be at least two")
    if (
        safety["max_reachable_states_per_record"]
        <= FROZEN_B0_LIMITS["max_reachable_states"]
    ):
        raise CapacityDiagnosticError(
            "diagnostic state envelope must be able to measure the frozen stop"
        )
    return json.loads(json.dumps(config))


def _load_structural_selection(
    *,
    candidate_store: Path,
    d1_snapshot: Path,
    minimum_edit_distance: int,
    witness_record_id: str,
) -> StructuralSelection:
    """Validate and stream the frozen label-free structural store."""

    if minimum_edit_distance < 2:
        raise CapacityDiagnosticError(
            "capacity selection may not include single-edit production rows"
        )
    snapshot = _read_json(d1_snapshot)
    try:
        global_stores = snapshot["global_stores"]
        structural_meta = global_stores["sealed_label_free_candidate_store"]
        canonical_meta = global_stores["canonical_label_store"]
    except (KeyError, TypeError) as error:
        raise CapacityDiagnosticError(
            "D1 snapshot lacks frozen global-store bindings"
        ) from error
    if Path(str(structural_meta["path"])).resolve() != candidate_store.resolve():
        raise CapacityDiagnosticError(
            "candidate-store path differs from the frozen structural store"
        )
    actual_store_ref = _artifact_ref(candidate_store)
    if actual_store_ref["bytes"] != structural_meta.get("bytes") or actual_store_ref[
        "sha256"
    ] != structural_meta.get("sha256"):
        raise CapacityDiagnosticError(
            "frozen structural store byte/hash binding failed"
        )
    if structural_meta.get("record_ids_sha256") != global_stores.get(
        "record_ids_sha256"
    ) or structural_meta.get("record_ids_sha256") != canonical_meta.get(
        "record_ids_sha256"
    ):
        raise CapacityDiagnosticError(
            "canonical/structural record-ID binding is inconsistent"
        )

    record_ids: list[str] = []
    selected: list[Mapping[str, Any]] = []
    endpoints: set[str] = set()
    exclusion_reasons: Counter[str] = Counter()
    split_eligible_count = 0
    seen: set[str] = set()
    with candidate_store.open("r", encoding="utf-8", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise CapacityDiagnosticError(
                    f"structural store JSON error at line {line_number}"
                ) from error
            if not isinstance(record, Mapping):
                raise CapacityDiagnosticError(
                    f"structural store row {line_number} is not an object"
                )
            forbidden = sorted(FORBIDDEN_TOP_LEVEL_LABEL_FIELDS & set(record))
            if forbidden:
                raise CapacityDiagnosticError(
                    "label-free structural store unexpectedly contains label "
                    f"fields at line {line_number}: {forbidden}"
                )
            record_id = str(record.get("record_id") or "").strip()
            if not record_id or record_id in seen:
                raise CapacityDiagnosticError(
                    f"invalid or duplicate record_id at line {line_number}"
                )
            seen.add(record_id)
            record_ids.append(record_id)
            eligible, excluded = select_split_eligible_records(
                [record],
                regions=REGIONS,
            )
            if excluded:
                exclusion_reasons[str(excluded[0]["reason"])] += 1
                continue
            if len(eligible) != 1:
                raise CapacityDiagnosticError(
                    f"split eligibility is ambiguous for {record_id}"
                )
            split_eligible_count += 1
            source = str(record.get("source_sequence") or "")
            candidate = str(record.get("candidate_sequence") or "")
            endpoints.update((source, candidate))
            edit_distance = record.get("edit_distance")
            if (
                isinstance(edit_distance, bool)
                or not isinstance(edit_distance, int)
                or edit_distance < 0
            ):
                raise CapacityDiagnosticError(f"invalid edit_distance for {record_id}")
            if edit_distance < minimum_edit_distance:
                exclusion_reasons[f"edit_distance_below_{minimum_edit_distance}"] += 1
                continue
            selected.append(
                {
                    **record,
                    "_canonical_jsonl_line": line_number,
                }
            )

    observed_ids_sha = _record_ids_sha256(record_ids)
    if len(record_ids) != structural_meta.get(
        "records"
    ) or observed_ids_sha != structural_meta.get("record_ids_sha256"):
        raise CapacityDiagnosticError(
            "frozen structural store record count/ID digest binding failed"
        )
    witness_rows = [row for row in selected if row["record_id"] == witness_record_id]
    if len(witness_rows) != 1:
        raise CapacityDiagnosticError(
            "frozen first witness is absent or duplicated in capacity selection"
        )
    ordered = tuple(
        witness_rows
        + sorted(
            (row for row in selected if row["record_id"] != witness_record_id),
            key=lambda row: (
                int(row["_canonical_jsonl_line"]),
                str(row["record_id"]),
            ),
        )
    )
    selected_ids = [str(row["record_id"]) for row in ordered]
    return StructuralSelection(
        records=ordered,
        eligible_endpoints=tuple(sorted(endpoints)),
        source_record_count=len(record_ids),
        split_eligible_record_count=split_eligible_count,
        selected_record_count=len(ordered),
        excluded_record_count=len(record_ids) - len(ordered),
        record_ids_sha256=observed_ids_sha,
        selected_record_ids_sha256=_record_ids_sha256(selected_ids),
        structural_store_sha256=actual_store_ref["sha256"],
        exclusion_reason_counts=dict(sorted(exclusion_reasons.items())),
    )


def _git_binding(
    *,
    project_root: Path,
    git_dir: Path,
    expected_head: str,
) -> dict[str, Any]:
    command_prefix = [
        "git",
        f"--git-dir={git_dir}",
        f"--work-tree={project_root}",
    ]
    head = (
        subprocess.run(
            [*command_prefix, "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        .stdout.decode("utf-8", errors="strict")
        .strip()
    )
    status = subprocess.run(
        [
            *command_prefix,
            "status",
            "--porcelain=v2",
            "--untracked-files=all",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    if head != expected_head:
        raise CapacityDiagnosticError(
            f"clean code commit drift: expected {expected_head}, observed {head}"
        )
    if status:
        raise CapacityDiagnosticError(
            "diagnostic requires a clean authoritative Git worktree"
        )
    return {
        "project_root": str(project_root.resolve()),
        "head": head,
        "clean": True,
        "dirty_state_sha256": _sha256_bytes(status),
    }


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _system_metric(run_root: Path, started_monotonic: float) -> dict[str, Any]:
    disk = shutil.disk_usage(run_root)
    run_bytes = sum(
        path.stat().st_size for path in run_root.rglob("*") if path.is_file()
    )
    return {
        "captured_at_utc": _utc_now(),
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "free_bytes": disk.free,
        "peak_rss_bytes": _peak_rss_bytes(),
        "run_bytes": run_bytes,
        "total_bytes": disk.total,
        "used_bytes": disk.used,
    }


def _check_global_resources(
    metric: Mapping[str, Any],
    safety: Mapping[str, int],
) -> None:
    if metric["free_bytes"] < safety["minimum_free_bytes"]:
        raise SafeCapacityPause(
            "minimum free disk safety line reached; evidence preserved"
        )
    if metric["peak_rss_bytes"] > safety["max_rss_bytes"]:
        raise SafeCapacityPause(
            "maximum diagnostic RSS safety line reached; evidence preserved"
        )
    if metric["elapsed_seconds"] > safety["max_wall_seconds"]:
        raise SafeCapacityPause(
            "maximum diagnostic wall-time safety line reached; evidence preserved"
        )
    if metric["run_bytes"] > safety["max_spill_bytes"]:
        raise SafeCapacityPause(
            "maximum diagnostic run-byte safety line reached; evidence preserved"
        )


def _iter_layer_sequences(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        previous: str | None = None
        for line_number, raw_line in enumerate(handle, start=1):
            sequence, separator, _count = raw_line.rstrip("\n").partition("\t")
            if not separator or not sequence:
                raise CapacityDiagnosticError(
                    f"invalid layer row at {path}:{line_number}"
                )
            if previous is not None and sequence <= previous:
                raise CapacityDiagnosticError(
                    f"layer state ordering is not strict: {path}"
                )
            previous = sequence
            yield sequence


def _iter_sequence_file(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        previous: str | None = None
        for line_number, raw_line in enumerate(handle, start=1):
            sequence = raw_line.rstrip("\n")
            if not sequence:
                raise CapacityDiagnosticError(
                    f"empty sequence row at {path}:{line_number}"
                )
            if previous is not None and sequence <= previous:
                raise CapacityDiagnosticError(
                    f"sequence universe is not strictly sorted: {path}"
                )
            previous = sequence
            yield sequence


def _merge_unique_iterators(
    iterators: Sequence[Iterator[str]],
    output: Path,
) -> tuple[int, str]:
    partial = output.with_name(output.name + ".partial")
    count = 0
    digest = hashlib.sha256()
    previous: str | None = None
    with partial.open("x", encoding="utf-8", newline="") as handle:
        for sequence in heapq.merge(*iterators):
            if sequence == previous:
                continue
            handle.write(sequence + "\n")
            digest.update(sequence.encode("utf-8"))
            digest.update(b"\n")
            previous = sequence
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, output)
    return count, digest.hexdigest()


def _record_state_universe(
    workspace: Path,
    layer_files: Sequence[str],
) -> Path:
    output = workspace / "state_universe.tsv"
    iterators = [
        _iter_layer_sequences(workspace / relative) for relative in layer_files
    ]
    _merge_unique_iterators(iterators, output)
    return output


def _merge_global_universe(
    *,
    run_root: Path,
    endpoints: Sequence[str],
    record_state_files: Sequence[Path],
    max_open_chunks: int,
) -> tuple[Path, int, str]:
    global_root = run_root / "global"
    spills_root = global_root / "spills"
    global_root.mkdir()
    spills_root.mkdir()
    endpoints_path = global_root / "eligible_endpoints.tsv"
    _write_text_exclusive(
        endpoints_path,
        "".join(sequence + "\n" for sequence in sorted(set(endpoints))),
    )
    current = [endpoints_path, *record_state_files]
    merge_pass = 0
    while len(current) > max_open_chunks:
        following: list[Path] = []
        for group_index, offset in enumerate(range(0, len(current), max_open_chunks)):
            output = spills_root / (f"pass-{merge_pass:03d}-run-{group_index:06d}.tsv")
            _merge_unique_iterators(
                [
                    _iter_sequence_file(path)
                    for path in current[offset : offset + max_open_chunks]
                ],
                output,
            )
            following.append(output)
        current = following
        merge_pass += 1
    output = global_root / "state_universe.tsv"
    count, digest = _merge_unique_iterators(
        [_iter_sequence_file(path) for path in current],
        output,
    )
    return output, count, digest


def _failure_dimension(message: str) -> str | None:
    mapping = {
        "reachable states": "max_reachable_states",
        "primitive actions": "max_neighbor_expansions",
        "state DP cells": "max_state_dp_cells",
        "DAG": "max_dag_cells",
    }
    return next(
        (dimension for phrase, dimension in mapping.items() if phrase in message),
        None,
    )


def _completed_layer_node_lower_bound(workspace: Path) -> int:
    total = 0
    layers = workspace / "layers"
    if not layers.exists():
        return 1
    for path in sorted(layers.glob("layer-*.tsv")):
        total += sum(1 for _ in _iter_layer_sequences(path))
    return max(total, 1)


def _run_record(
    *,
    record: Mapping[str, Any],
    ordinal: int,
    run_root: Path,
    safety: Mapping[str, int],
) -> tuple[dict[str, Any], Path | None]:
    record_id = str(record["record_id"])
    source = str(record["source_sequence"])
    candidate = str(record["candidate_sequence"])
    known_distance = int(record["edit_distance"])
    started = time.monotonic()
    statistics = minimum_alignment_statistics(
        source,
        candidate,
        known_minimum_edit_count=known_distance,
        max_dag_cells=safety["max_dag_cells_per_record"],
    )
    workspace = (
        run_root
        / "record_workspaces"
        / f"{ordinal:06d}-{hashlib.sha256(record_id.encode()).hexdigest()[:12]}"
    )
    workspace.parent.mkdir(exist_ok=True)
    summary = None
    state_file: Path | None = None
    try:
        summary = minimum_alignment_state_summary(
            source,
            candidate,
            workspace=workspace,
            known_minimum_edit_count=known_distance,
            max_dag_cells=safety["max_dag_cells_per_record"],
            max_reachable_states=safety["max_reachable_states_per_record"],
            max_neighbor_expansions=safety["max_neighbor_expansions_per_record"],
            max_state_dp_cells=safety["max_state_dp_cells_per_record"],
            max_spill_bytes=safety["max_spill_bytes_per_record"],
            max_open_chunks=safety["max_open_chunks"],
            chunk_size=safety["chunk_size"],
        )
        exact = {
            "reachable_node_count": summary.reachable_node_count,
            "reachable_transition_count": (summary.reachable_transition_count),
            "minimum_state_path_count": summary.minimum_state_path_count,
            "evaluated_primitive_action_count": (
                summary.evaluated_primitive_action_count
            ),
            "evaluated_state_dp_cell_count": (summary.evaluated_state_dp_cell_count),
            "reachable_states_sha256": summary.reachable_states_sha256,
        }
        exceeded = []
        if statistics.evaluated_dag_cell_count > FROZEN_B0_LIMITS["max_dag_cells"]:
            exceeded.append("max_dag_cells")
        if summary.reachable_node_count > FROZEN_B0_LIMITS["max_reachable_states"]:
            exceeded.append("max_reachable_states")
        if (
            summary.evaluated_primitive_action_count
            > FROZEN_B0_LIMITS["max_neighbor_expansions"]
        ):
            exceeded.append("max_neighbor_expansions")
        if (
            summary.evaluated_state_dp_cell_count
            > FROZEN_B0_LIMITS["max_state_dp_cells"]
        ):
            exceeded.append("max_state_dp_cells")
        state_file = _record_state_universe(
            workspace,
            summary.layer_files,
        )
        capacity = {"exact": exact, "lower_bound": None}
        outcome = "EXACT_COMPLETED"
        evidence = {
            "counts_exact": True,
            "state_set_complete": True,
            "no_approximation_emitted": True,
            "usable_for_b0_acceptance": False,
        }
        stop = None
        spill_bytes = summary.spill_bytes
    except StreamingPathStateError as error:
        message = str(error)
        if "spill bytes" in message:
            raise SafeCapacityPause(message) from error
        dimension = _failure_dimension(message)
        if dimension is None:
            raise CapacityDiagnosticError(
                f"unclassified exact prototype failure for {record_id}: {message}"
            ) from error
        node_lower_bound = _completed_layer_node_lower_bound(workspace)
        lower_bound = {
            "reachable_node_count": node_lower_bound,
            "reachable_transition_count": max(node_lower_bound - 1, 0),
            "minimum_state_path_count": 1,
            "evaluated_primitive_action_count": 0,
            "evaluated_state_dp_cell_count": 0,
        }
        capacity = {"exact": None, "lower_bound": lower_bound}
        outcome = "LOWER_BOUND_STOPPED"
        evidence = {
            "counts_exact": False,
            "state_set_complete": False,
            "no_approximation_emitted": True,
            "usable_for_b0_acceptance": False,
        }
        observed_lower_bound = max(
            FROZEN_B0_LIMITS[dimension] + 1,
            lower_bound.get(
                {
                    "max_reachable_states": "reachable_node_count",
                    "max_neighbor_expansions": ("evaluated_primitive_action_count"),
                    "max_state_dp_cells": "evaluated_state_dp_cell_count",
                    "max_dag_cells": "reachable_node_count",
                }[dimension],
                1,
            ),
        )
        stop = {
            "stop_rule": "STOP_RULE_B0_PATH_STATE_COMPLEXITY",
            "dimension": dimension,
            "frozen_limit": FROZEN_B0_LIMITS[dimension],
            "observed_lower_bound": observed_lower_bound,
            "message": message,
        }
        exceeded = [dimension]
        spill_bytes = sum(
            path.stat().st_size for path in workspace.rglob("*") if path.is_file()
        )

    row = {
        "schema_version": "b0_capacity_record.v1",
        "ordinal": ordinal,
        "record_id": record_id,
        "dataset_id": str(record["dataset_id"]),
        "region": str(record["region"]),
        "canonical_jsonl_line": int(record["_canonical_jsonl_line"]),
        "input_record_structural_sha256": record_structural_sha256(record),
        "source_sequence_sha256": _sha256_bytes(source.encode("utf-8")),
        "candidate_sequence_sha256": _sha256_bytes(candidate.encode("utf-8")),
        "source_length": len(source),
        "candidate_length": len(candidate),
        "alignment_statistics": {
            "minimum_edit_count": statistics.minimum_edit_count,
            "minimum_alignment_count": statistics.minimum_alignment_count,
            "evaluated_dag_cell_count": statistics.evaluated_dag_cell_count,
            "counts_exact": True,
        },
        "outcome": outcome,
        "capacity": capacity,
        "frozen_gate_assessment": {
            "would_pass_frozen_b0_limits": not exceeded,
            "exceeded_limits": exceeded,
        },
        "evidence_semantics": evidence,
        "stop": stop,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": _peak_rss_bytes(),
        "spill_bytes": spill_bytes,
    }
    return row, state_file


def _verify_witness(
    record: Mapping[str, Any],
    row: Mapping[str, Any],
) -> None:
    if (
        record["record_id"] != WITNESS_RECORD_ID
        or int(record["_canonical_jsonl_line"]) != WITNESS_CANONICAL_JSONL_LINE
        or record["source_sequence"] != WITNESS_SOURCE
        or record["candidate_sequence"] != WITNESS_CANDIDATE
    ):
        raise CapacityDiagnosticError(
            "frozen witness identity, line, or endpoint sequence drifted"
        )
    if row["outcome"] != "EXACT_COMPLETED":
        raise CapacityDiagnosticError("frozen witness did not complete exactly")
    if row["capacity"]["exact"] != WITNESS_EXPECTED:
        raise CapacityDiagnosticError(
            "frozen witness exact counters or digest failed parity"
        )


def _runtime_manifest() -> dict[str, Any]:
    executable = Path(sys.executable).resolve(strict=True)
    return {
        "captured_at_utc": _utc_now(),
        "executable": str(executable),
        "executable_bytes": executable.stat().st_size,
        "executable_sha256": _sha256_file(executable),
        "implementation": platform.python_implementation(),
        "jsonschema_version": importlib.metadata.version("jsonschema"),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "workload_class": "NON_NEURAL_DATA_BENCHMARK",
        "cuda_required": False,
        "cuda_used": False,
    }


def _find_required_artifact(
    snapshot: Mapping[str, Any],
    suffix: str,
) -> Path:
    artifacts = snapshot.get("required_artifacts")
    if not isinstance(artifacts, Mapping):
        raise CapacityDiagnosticError("D1 snapshot lacks required-artifact bindings")
    matches = [value for key, value in artifacts.items() if str(key).endswith(suffix)]
    if len(matches) != 1 or not isinstance(matches[0], Mapping):
        raise CapacityDiagnosticError(
            f"D1 snapshot has ambiguous required artifact: {suffix}"
        )
    return Path(str(matches[0]["path"]))


def _code_files(project_root: Path) -> list[dict[str, Any]]:
    entries = (
        (
            "data/utr_benchmark_v2/path_states.py",
            "PATH_STATE_ORACLE",
        ),
        (
            "data/utr_benchmark_v2/near_neighbors.py",
            "NEAR_NEIGHBOR_ORACLE",
        ),
        (
            "data/utr_benchmark_v2/split_graph.py",
            "SELECTION_ORACLE",
        ),
        (
            "data/utr_benchmark_v2/symbolic_path_states.py",
            "STREAMING_PROTOTYPE",
        ),
        (
            "scripts/data/diagnose_b0_path_capacity.py",
            "DIAGNOSTIC_ENTRYPOINT",
        ),
        (
            "schemas/b0_capacity_diagnostic.schema.json",
            "DIAGNOSTIC_SCHEMA",
        ),
    )
    result = []
    for relative, role in entries:
        path = project_root / relative
        ref = _artifact_ref(path, root=project_root)
        result.append({**ref, "role": role})
    return result


def _write_replay_script(
    *,
    run_root: Path,
    argv: Sequence[str],
) -> None:
    rendered: list[str] = []
    replace_next = False
    for token in argv:
        if replace_next:
            rendered.append('"${1}"')
            replace_next = False
        else:
            rendered.append(shlex.quote(token))
            if token == "--output-root":
                replace_next = True
    body = (
        "#!/bin/sh\n"
        "set -eu\n"
        'if [ "$#" -ne 1 ]; then\n'
        '  echo "usage: replay.sh FRESH_OUTPUT_ROOT" >&2\n'
        "  exit 2\n"
        "fi\n"
        "exec " + " ".join(rendered) + "\n"
    )
    path = run_root / "replay.sh"
    _write_text_exclusive(path, body)
    path.chmod(0o750)


def _diagnostic_summary(
    *,
    selection: StructuralSelection,
    rows: Sequence[Mapping[str, Any]],
    global_state_count: int,
    global_state_digest: str,
    global_state_exact: bool,
    global_state_path: Path,
) -> dict[str, Any]:
    exact_rows = [row for row in rows if row["outcome"] == "EXACT_COMPLETED"]
    lower_rows = [row for row in rows if row["outcome"] == "LOWER_BOUND_STOPPED"]
    exact_node_rows = [
        (
            int(row["capacity"]["exact"]["reachable_node_count"]),
            str(row["record_id"]),
        )
        for row in exact_rows
    ]
    above_state_gate = [
        row["record_id"]
        for row in exact_rows
        if row["capacity"]["exact"]["reachable_node_count"]
        > FROZEN_B0_LIMITS["max_reachable_states"]
    ]
    edit_class_counts: Counter[str] = Counter()
    for row in rows:
        source_length = int(row["source_length"])
        candidate_length = int(row["candidate_length"])
        distance = int(row["alignment_statistics"]["minimum_edit_count"])
        edit_class_counts[
            (
                "pure_indel"
                if abs(source_length - candidate_length) == distance
                else "mixed"
            )
        ] += 1
    maximum = max(exact_node_rows, default=(0, None))
    return {
        "schema_version": "b0_capacity_summary.v1",
        "formal_b0_attempt_started": False,
        "b0_gate_values_changed": False,
        "budget_change_authorized": False,
        "scientific_result_claimed": False,
        "source_record_count": selection.source_record_count,
        "split_eligible_record_count": (selection.split_eligible_record_count),
        "scheduled_multi_edit_record_count": len(rows),
        "exact_completed_record_count": len(exact_rows),
        "lower_bound_record_count": len(lower_rows),
        "edit_class_counts": dict(sorted(edit_class_counts.items())),
        "sum_exact_per_record_reachable_nodes": sum(
            node_count for node_count, _record_id in exact_node_rows
        ),
        "maximum_exact_record_reachable_nodes": maximum[0],
        "maximum_exact_record_id": maximum[1],
        "records_above_frozen_state_gate": sorted(above_state_gate),
        "records_above_frozen_state_gate_count": len(above_state_gate),
        "global_unique_eligible_endpoint_count": len(selection.eligible_endpoints),
        "global_unique_state_count": global_state_count,
        "global_state_universe_sha256": global_state_digest,
        "global_state_universe_exact": global_state_exact,
        "global_state_universe_role": (
            "EXACT"
            if global_state_exact
            else "LOWER_BOUND_FROM_ENDPOINTS_AND_EXACT_COMPLETED_RECORDS"
        ),
        "global_state_universe_artifact": _artifact_ref(global_state_path),
        "usable_for_b0_acceptance": False,
        "allowed_interpretation": "CAPACITY_DIAGNOSTIC_ONLY",
    }


def _checksum_index(
    run_root: Path,
    paths: Iterable[Path],
) -> dict[str, Any]:
    entries = []
    for path in sorted(
        {item.resolve() for item in paths},
        key=lambda item: str(item),
    ):
        ref = _artifact_ref(path, root=run_root)
        entries.append(ref)
    return {
        "schema_version": "b0_capacity_checksums.v1",
        "artifact_count": len(entries),
        "artifacts": entries,
    }


def _validate_manifest(
    manifest: Mapping[str, Any],
    schema_path: Path,
) -> None:
    schema = _read_json(schema_path)
    validator = jsonschema.Draft202012Validator(
        schema,
        format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER,
    )
    errors = sorted(
        validator.iter_errors(manifest),
        key=lambda error: list(error.path),
    )
    if errors:
        rendered = "; ".join(
            f"{'/'.join(str(part) for part in error.path)}: {error.message}"
            for error in errors[:10]
        )
        raise CapacityDiagnosticError(
            f"terminal diagnostic manifest failed schema validation: {rendered}"
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--d1-snapshot", type=Path, required=True)
    parser.add_argument("--d1-acceptance", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--candidate-store", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--git-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-code-commit",
        required=True,
    )
    parser.add_argument(
        "--scope",
        choices=("witness", "census"),
        default="census",
    )
    parser.add_argument("--parent-diagnostic-id")
    return parser.parse_args(argv)


def _validate_cli_paths(args: argparse.Namespace) -> None:
    for name in (
        "config",
        "d1_snapshot",
        "d1_acceptance",
        "contract",
        "candidate_store",
        "project_root",
        "git_dir",
    ):
        path = Path(getattr(args, name))
        if not path.is_absolute() or not path.exists():
            raise CapacityDiagnosticError(
                f"--{name.replace('_', '-')} must be an existing absolute path"
            )
    if len(args.expected_code_commit) != 40 or any(
        base not in "0123456789abcdef" for base in args.expected_code_commit
    ):
        raise CapacityDiagnosticError(
            "--expected-code-commit must be a lowercase 40-hex Git commit"
        )
    if not args.output_root.is_absolute():
        raise CapacityDiagnosticError("--output-root must be absolute")
    if not args.output_root.name.startswith("B0_capacity_"):
        raise CapacityDiagnosticError(
            "output root name must be a B0_capacity diagnostic ID"
        )


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    _validate_cli_paths(args)
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    _create_exclusive_run_root(args.output_root)
    run_root = args.output_root
    for relative in ("logs", "provenance", "records"):
        (run_root / relative).mkdir()
    events_path = run_root / "logs/events.jsonl"
    metrics_path = run_root / "logs/system_metrics.jsonl"
    _append_jsonl(
        events_path,
        {
            "at_utc": started_at,
            "event": "DIAGNOSTIC_STARTED",
            "formal_b0_attempt_started": False,
        },
    )

    config = _validate_config(_read_json(args.config))
    _write_json_exclusive(run_root / "resolved_config.json", config)
    contract_ref = _artifact_ref(args.contract)
    if contract_ref["sha256"] != GOAL_CONTRACT_SHA256:
        raise CapacityDiagnosticError("active scientific contract hash drifted")
    git = _git_binding(
        project_root=args.project_root,
        git_dir=args.git_dir,
        expected_head=args.expected_code_commit,
    )
    snapshot = _read_json(args.d1_snapshot)
    snapshot_ref = _artifact_ref(args.d1_snapshot)
    acceptance_ref = _artifact_ref(args.d1_acceptance)
    if (
        acceptance_ref["bytes"] != snapshot["acceptance"]["bytes"]
        or acceptance_ref["sha256"] != snapshot["acceptance"]["sha256"]
    ):
        raise CapacityDiagnosticError(
            "D1 acceptance differs from the frozen snapshot binding"
        )
    build_manifest_path = Path(str(snapshot["build_manifest"]["path"]))
    build_manifest_ref = _artifact_ref(build_manifest_path)
    if (
        build_manifest_ref["bytes"] != snapshot["build_manifest"]["bytes"]
        or build_manifest_ref["sha256"] != snapshot["build_manifest"]["sha256"]
    ):
        raise CapacityDiagnosticError(
            "D1 build manifest differs from the frozen snapshot binding"
        )
    ambiguity_path = _find_required_artifact(
        snapshot,
        "edit_script_ambiguity_report.json",
    )
    ambiguity_ref = _artifact_ref(ambiguity_path)

    selection = _load_structural_selection(
        candidate_store=args.candidate_store,
        d1_snapshot=args.d1_snapshot,
        minimum_edit_distance=config["selection"]["minimum_edit_distance"],
        witness_record_id=WITNESS_RECORD_ID,
    )
    if args.scope == "witness":
        selection = replace(
            selection,
            records=(selection.records[0],),
            selected_record_count=1,
            excluded_record_count=selection.source_record_count - 1,
            selected_record_ids_sha256=_record_ids_sha256([WITNESS_RECORD_ID]),
        )
    selection_manifest = {
        "schema_version": "b0_capacity_selection.v1",
        "scope": args.scope,
        "source_store_role": "D1_SEALED_LABEL_FREE_CANDIDATE_STORE",
        "selection_algorithm": (
            "split_graph.select_split_eligible_records+edit_distance_gte_2"
        ),
        "record_order": ("FROZEN_WITNESS_FIRST_THEN_CANONICAL_JSONL_LINE_ASCENDING"),
        "source_record_count": selection.source_record_count,
        "split_eligible_record_count": (selection.split_eligible_record_count),
        "selected_record_count": selection.selected_record_count,
        "excluded_record_count": selection.excluded_record_count,
        "record_ids_sha256": selection.record_ids_sha256,
        "selected_record_ids_sha256": (selection.selected_record_ids_sha256),
        "selected_records": [
            {
                "canonical_jsonl_line": int(record["_canonical_jsonl_line"]),
                "record_id": str(record["record_id"]),
                "structural_sha256": record_structural_sha256(record),
            }
            for record in selection.records
        ],
        "exclusion_reason_counts": dict(selection.exclusion_reason_counts),
        "label_fields_read": [],
        "canonical_label_store_opened": False,
    }
    selection_manifest_path = run_root / "provenance/selection_manifest.json"
    _write_json_exclusive(selection_manifest_path, selection_manifest)

    runtime = _runtime_manifest()
    runtime_path = run_root / "provenance/runtime_manifest.json"
    _write_json_exclusive(runtime_path, runtime)
    python_launcher_path = run_root / "provenance/python_launcher.json"
    _write_json_exclusive(
        python_launcher_path,
        {
            "path": runtime["executable"],
            "bytes": runtime["executable_bytes"],
            "sha256": runtime["executable_sha256"],
        },
    )
    code_files = _code_files(args.project_root)
    code_manifest_path = run_root / "provenance/code_manifest.json"
    _write_json_exclusive(
        code_manifest_path,
        {
            "code_commit": args.expected_code_commit,
            "files": code_files,
        },
    )
    input_manifest_path = run_root / "provenance/input_manifest.json"
    structural_meta = snapshot["global_stores"]["sealed_label_free_candidate_store"]
    canonical_meta = snapshot["global_stores"]["canonical_label_store"]
    _write_json_exclusive(
        input_manifest_path,
        {
            "contract": contract_ref,
            "d1_snapshot": snapshot_ref,
            "d1_acceptance": acceptance_ref,
            "d1_build_manifest": build_manifest_ref,
            "ambiguity_report": ambiguity_ref,
            "structural_store": structural_meta,
            "canonical_store_metadata_only": {
                **canonical_meta,
                "opened": False,
            },
            "label_fields_read": [],
        },
    )
    argv_exact = [sys.executable, *sys.argv]
    command_path = run_root / "command.json"
    _write_json_exclusive(
        command_path,
        {
            "exact_argv": argv_exact,
            "exact_argv_sha256": _canonical_sha256(argv_exact),
            "output_root_replay_policy": (
                "replace_with_a_fresh_nonexistent_output_root"
            ),
        },
    )
    _write_replay_script(run_root=run_root, argv=argv_exact)

    rows: list[dict[str, Any]] = []
    record_state_files: list[Path] = []
    state = "COMPLETED"
    stop_reason: str | None = None
    failure: dict[str, Any] | None = None
    safety = config["diagnostic_safety_limits"]
    for ordinal, record in enumerate(selection.records):
        metric = _system_metric(run_root, started_monotonic)
        _append_jsonl(metrics_path, metric)
        _check_global_resources(metric, safety)
        _append_jsonl(
            events_path,
            {
                "at_utc": _utc_now(),
                "event": "RECORD_STARTED",
                "ordinal": ordinal,
                "record_id": record["record_id"],
            },
        )
        try:
            row, state_file = _run_record(
                record=record,
                ordinal=ordinal,
                run_root=run_root,
                safety=safety,
            )
        except SafeCapacityPause as error:
            state = "SAFE_PAUSED"
            stop_reason = str(error)
            failure_path = run_root / "failure/resource_pause.json"
            _write_json_exclusive(
                failure_path,
                {
                    "failed_at_utc": _utc_now(),
                    "failure_class": "SAFE_RESOURCE_PAUSE",
                    "reason": stop_reason,
                    "partial_outputs_preserved": True,
                    "resume_policy": ("fresh_segment_only_after_hash_revalidation"),
                },
            )
            failure = {
                "failed_at_utc": _utc_now(),
                "failure_class": "SAFE_RESOURCE_PAUSE",
                "reason": stop_reason,
                "evidence": _artifact_ref(failure_path, root=run_root),
                "partial_outputs_preserved": True,
                "resume_allowed": True,
            }
            break
        rows.append(row)
        if state_file is not None:
            record_state_files.append(state_file)
        if ordinal == 0:
            _verify_witness(record, row)
        _append_jsonl(
            events_path,
            {
                "at_utc": _utc_now(),
                "event": "RECORD_TERMINAL",
                "ordinal": ordinal,
                "outcome": row["outcome"],
                "record_id": row["record_id"],
            },
        )

    if not rows or rows[0]["record_id"] != WITNESS_RECORD_ID:
        raise CapacityDiagnosticError(
            "diagnostic segment ended without an exact frozen witness"
        )
    records_path = run_root / "records/results.jsonl"
    _write_text_exclusive(
        records_path,
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
        ),
    )
    shard_index_path = run_root / "records/shard_index.json"
    _write_json_exclusive(
        shard_index_path,
        {
            "schema_version": "b0_capacity_shard_index.v1",
            "row_count": len(rows),
            "record_ids_sha256": _record_ids_sha256(
                [str(row["record_id"]) for row in rows]
            ),
            "shards": [_artifact_ref(records_path, root=run_root)],
        },
    )

    global_path, global_count, global_digest = _merge_global_universe(
        run_root=run_root,
        endpoints=selection.eligible_endpoints,
        record_state_files=record_state_files,
        max_open_chunks=safety["max_open_chunks"],
    )
    global_exact = (
        state == "COMPLETED"
        and len(rows) == len(selection.records)
        and all(row["outcome"] == "EXACT_COMPLETED" for row in rows)
    )
    summary = _diagnostic_summary(
        selection=selection,
        rows=rows,
        global_state_count=global_count,
        global_state_digest=global_digest,
        global_state_exact=global_exact,
        global_state_path=global_path,
    )
    summary_path = run_root / "capacity_summary.json"
    _write_json_exclusive(summary_path, summary)

    outcome_counts = Counter(row["outcome"] for row in rows)
    exact_count = outcome_counts["EXACT_COMPLETED"]
    lower_count = outcome_counts["LOWER_BOUND_STOPPED"]
    all_accounted = len(rows) == len(selection.records)
    if state == "COMPLETED" and not all_accounted:
        raise CapacityDiagnosticError(
            "completed diagnostic has unaccounted scheduled records"
        )
    marker_name = "DONE" if state == "COMPLETED" else "SAFE_PAUSED"
    marker_path = run_root / marker_name
    _write_text_exclusive(marker_path, "")
    terminal_lock = run_root / "terminal.lock"
    _write_text_exclusive(terminal_lock, "")
    status_path = run_root / "status.json"
    _write_json_exclusive(
        status_path,
        {
            "state": state,
            "terminal": True,
            "formal_b0_attempt_started": False,
            "b0_accepted": False,
            "b0_frozen": False,
            "scientific_result_claimed": False,
            "stop_reason": stop_reason,
        },
    )
    _append_jsonl(
        events_path,
        {
            "at_utc": _utc_now(),
            "event": "DIAGNOSTIC_TERMINAL",
            "state": state,
        },
    )

    durable_paths = [
        run_root / "resolved_config.json",
        selection_manifest_path,
        runtime_path,
        python_launcher_path,
        code_manifest_path,
        input_manifest_path,
        command_path,
        run_root / "replay.sh",
        events_path,
        metrics_path,
        records_path,
        shard_index_path,
        summary_path,
        global_path,
        run_root / "global/eligible_endpoints.tsv",
        marker_path,
        terminal_lock,
        status_path,
    ]
    if failure is not None:
        durable_paths.append(run_root / failure["evidence"]["path"])
    checksums_path = run_root / "artifact_checksums.json"
    _write_json_exclusive(
        checksums_path,
        _checksum_index(run_root, durable_paths),
    )

    completed_at = _utc_now()
    manifest = {
        "artifact_type": "b0_capacity_diagnostic",
        "schema_version": "b0_capacity_diagnostic.v1",
        "diagnostic_id": run_root.name,
        "parent_diagnostic_id": args.parent_diagnostic_id,
        "state": state,
        "terminal": True,
        "started_at_utc": started_at,
        "ended_at_utc": completed_at,
        "workload_class": "NON_NEURAL_DATA_BENCHMARK",
        "purpose": "B0_PATH_CAPACITY_DIAGNOSTIC_ONLY",
        "goal_contract": {
            "id": "utr_editflow_goal_v2",
            "sha256": GOAL_CONTRACT_SHA256,
            "repository_snapshot": ("docs/contracts/mrna_latest_build_contract_v2.md"),
        },
        "claim_boundary": _claim_boundary(),
        "selection": {
            "source_store_role": ("D1_SEALED_LABEL_FREE_CANDIDATE_STORE"),
            "selection_algorithm": (
                "split_graph.select_split_eligible_records+edit_distance_gte_2"
            ),
            "regions": list(REGIONS),
            "record_order": (
                "FROZEN_WITNESS_FIRST_THEN_CANONICAL_JSONL_LINE_ASCENDING"
            ),
            "source_record_count": selection.source_record_count,
            "selected_record_count": selection.selected_record_count,
            "excluded_record_count": selection.excluded_record_count,
            "selected_record_ids_sha256": (selection.selected_record_ids_sha256),
            "label_fields_read": [],
            "canonical_label_store_opened": False,
            "selection_manifest": _artifact_ref(
                selection_manifest_path,
                root=run_root,
            ),
        },
        "witnesses": [
            {
                "witness_kind": ("FROZEN_FIRST_PATH_COMPLEXITY_WITNESS"),
                "record_id": WITNESS_RECORD_ID,
                "canonical_jsonl_line": WITNESS_CANONICAL_JSONL_LINE,
                "source_length": len(WITNESS_SOURCE),
                "candidate_length": len(WITNESS_CANDIDATE),
                "expected": WITNESS_EXPECTED,
                "observed_record_ordinal": 0,
                "parity_passed": True,
            }
        ],
        "provenance": {
            "git": git,
            "exact_argv": argv_exact,
            "exact_argv_sha256": _canonical_sha256(argv_exact),
            "resolved_config": _artifact_ref(
                run_root / "resolved_config.json",
                root=run_root,
            ),
            "runtime_manifest": _artifact_ref(
                runtime_path,
                root=run_root,
            ),
            "python_launcher": _artifact_ref(
                python_launcher_path,
                root=run_root,
            ),
            "d1_acceptance": acceptance_ref,
            "d1_build_manifest": build_manifest_ref,
            "canonical_store_metadata_only": {
                "path": str(canonical_meta["path"]),
                "bytes": int(canonical_meta["bytes"]),
                "sha256": str(canonical_meta["sha256"]),
                "record_count": int(canonical_meta["records"]),
                "record_ids_sha256": str(canonical_meta["record_ids_sha256"]),
                "content_sha256": str(canonical_meta["sha256"]),
                "opened": False,
            },
            "structural_store": {
                "path": str(structural_meta["path"]),
                "bytes": int(structural_meta["bytes"]),
                "sha256": str(structural_meta["sha256"]),
                "record_count": int(structural_meta["records"]),
                "record_ids_sha256": str(structural_meta["record_ids_sha256"]),
                "structural_content_sha256": str(structural_meta["sha256"]),
                "role": "D1_SEALED_LABEL_FREE_CANDIDATE_STORE",
            },
            "ambiguity_report": ambiguity_ref,
            "code_files": code_files,
        },
        "algorithm_contract": {
            "path_state_algorithm": ALGORITHM_ID,
            "state_closure_scope": STATE_CLOSURE_SCOPE,
            "state_path_count_scope": STATE_PATH_COUNT_SCOPE,
            "primitive_action_evaluation_scope": (PRIMITIVE_ACTION_EVALUATION_SCOPE),
            "near_neighbor_algorithm": NEAR_NEIGHBOR_ALGORITHM_ID,
            "near_neighbor_edit_distance_threshold": 5,
            "frozen_b0_limits": FROZEN_B0_LIMITS,
            "diagnostic_safety_limits": {
                "minimum_free_bytes": safety["minimum_free_bytes"],
                "max_rss_bytes": safety["max_rss_bytes"],
                "max_wall_seconds": safety["max_wall_seconds"],
                "max_spill_bytes": safety["max_spill_bytes"],
                "max_dag_cells_per_record": safety["max_dag_cells_per_record"],
                "max_reachable_states_per_record": safety[
                    "max_reachable_states_per_record"
                ],
                "max_neighbor_expansions_per_record": safety[
                    "max_neighbor_expansions_per_record"
                ],
                "max_state_dp_cells_per_record": safety[
                    "max_state_dp_cells_per_record"
                ],
                "max_spill_bytes_per_record": safety["max_spill_bytes_per_record"],
                "chunk_size": safety["chunk_size"],
                "max_open_chunks": safety["max_open_chunks"],
                "heartbeat_seconds": safety["heartbeat_seconds"],
            },
        },
        "accounting": {
            "scheduled_record_count": len(selection.records),
            "terminal_record_count": len(rows),
            "exact_completed_count": exact_count,
            "lower_bound_count": lower_count,
            "failed_record_count": 0,
            "outcome_counts": {
                "EXACT_COMPLETED": exact_count,
                "LOWER_BOUND_STOPPED": lower_count,
            },
            "all_scheduled_records_accounted": all_accounted,
            "accounting_reconciled": all_accounted,
            "record_results": _artifact_ref(
                records_path,
                root=run_root,
            ),
            "record_shard_index": _artifact_ref(
                shard_index_path,
                root=run_root,
            ),
            "capacity_summary": _artifact_ref(
                summary_path,
                root=run_root,
            ),
        },
        "result_semantics": {
            "exact_capacity_complete": global_exact,
            "lower_bounds_present": lower_count > 0,
            "no_approximation_emitted": True,
            "usable_for_budget_decision": (args.scope == "census" and global_exact),
            "usable_for_b0_acceptance": False,
        },
        "stop_reason": stop_reason,
        "completion_seal": (
            {
                "sealed_at_utc": completed_at,
                "terminal_marker": "DONE",
                "terminal_marker_ref": _artifact_ref(
                    marker_path,
                    root=run_root,
                ),
                "artifact_checksum_index": _artifact_ref(
                    checksums_path,
                    root=run_root,
                ),
                "record_shard_index": _artifact_ref(
                    shard_index_path,
                    root=run_root,
                ),
            }
            if state == "COMPLETED"
            else None
        ),
        "failure": failure,
    }
    schema_path = args.project_root / "schemas/b0_capacity_diagnostic.schema.json"
    _validate_manifest(manifest, schema_path)
    manifest_path = run_root / "diagnostic_manifest.json"
    _write_json_exclusive(manifest_path, manifest)
    return {
        "diagnostic_manifest": _artifact_ref(
            manifest_path,
            root=run_root,
        ),
        "state": state,
        "capacity_summary": _artifact_ref(
            summary_path,
            root=run_root,
        ),
        "formal_b0_attempt_started": False,
        "b0_accepted": False,
        "b0_frozen": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = run(args)
    except Exception as error:
        output_root = getattr(args, "output_root", None)
        if isinstance(output_root, Path) and output_root.exists():
            failure_root = output_root / "failure"
            failure_root.mkdir(exist_ok=True)
            failure_path = failure_root / "terminal_failure.json"
            if not failure_path.exists():
                _write_json_exclusive(
                    failure_path,
                    {
                        "at_utc": _utc_now(),
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "formal_b0_attempt_started": False,
                        "b0_gate_values_changed": False,
                        "partial_outputs_preserved": True,
                    },
                )
            failed_marker = output_root / "FAILED"
            if not failed_marker.exists():
                _write_text_exclusive(failed_marker, "")
        print(
            json.dumps(
                {
                    "status": "FAILED_WITH_EVIDENCE",
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "formal_b0_attempt_started": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
