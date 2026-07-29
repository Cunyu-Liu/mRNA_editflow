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
import re
import resource
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import jsonschema

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.utr_benchmark_v2.near_neighbors import (
    NEAR_NEIGHBOR_ALGORITHM_ID,
)
from data.utr_benchmark_v2.d1_builder import CANDIDATE_STORE_FIELDS
from data.utr_benchmark_v2.d1_builder import candidate_store_label_paths
from data.utr_benchmark_v2.path_states import ALGORITHM_ID
from data.utr_benchmark_v2.path_states import PathStateError
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
from scripts.data.validate_d1_canonical_snapshot import validate_snapshot
from scripts.execution.acceptance_semantics import validate_phase_acceptance


GOAL_CONTRACT_SHA256 = (
    "c3dc5875868d847b8519fee40b14c43b65e4c5948dc5c3b98101ca61a5671dd5"
)
ALLOWED_OUTPUT_PARENT = Path("/mnt/cunyuliu/mrna_editflow_b0_capacity")
FORMAL_ATTEMPT_ROOT = Path("/mnt/cunyuliu/mrna_editflow_d1_b0")
PROGRESS_CHECK_INTERVAL = 10_000
RESOURCE_LIMIT_SCOPE = (
    "COMPUTE_AND_GLOBAL_MERGE_THROUGH_FORCED_PRETERMINAL_CHECK;"
    "IMMUTABLE_SEAL_OVERHEAD_RECORDED_BUT_NOT_BUDGETED"
)
STREAM_CAPTURE_SCOPE = (
    "POST_RUN_ROOT_CREATION_THROUGH_TERMINAL_COMPUTE_EVENT;"
    "SEAL_FAILURES_FAIL_CLOSED_WITHOUT_VERIFIED_MARKER"
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
DIAGNOSTIC_LIMIT_FIELD_BY_DIMENSION = {
    "max_dag_cells": "max_dag_cells_per_record",
    "max_reachable_states": "max_reachable_states_per_record",
    "max_neighbor_expansions": "max_neighbor_expansions_per_record",
    "max_state_dp_cells": "max_state_dp_cells_per_record",
}


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


@dataclass
class ResourceWatchdog:
    """Check hard run limits frequently while logging only low-frequency heartbeats."""

    run_root: Path
    metrics_path: Path
    started_monotonic: float
    safety: Mapping[str, int]
    last_logged_monotonic: float

    def check(
        self,
        progress: Mapping[str, Any] | None = None,
        *,
        force_log: bool = False,
    ) -> None:
        metric = _system_metric(self.run_root, self.started_monotonic)
        now = time.monotonic()
        if (
            force_log
            or now - self.last_logged_monotonic >= self.safety["heartbeat_seconds"]
        ):
            _append_jsonl(
                self.metrics_path,
                {
                    **metric,
                    "progress": dict(progress or {}),
                },
            )
            self.last_logged_monotonic = now
        _check_global_resources(metric, self.safety)

    def prototype_callback(self, progress: Mapping[str, Any]) -> None:
        try:
            self.check(progress)
        except SafeCapacityPause as error:
            raise StreamingPathStateError(f"SAFE_RESOURCE_PAUSE: {error}") from error


@dataclass
class VerificationResourceGuard:
    """Apply the declared diagnostic envelope to independent replay scratch."""

    scratch_root: Path
    scratch_byte_limit: int
    safety: Mapping[str, int]
    started_monotonic: float

    def check(self, _progress: Mapping[str, Any] | None = None) -> None:
        scratch_bytes = sum(
            path.stat().st_size
            for path in self.scratch_root.rglob("*")
            if path.is_file()
        )
        if (
            shutil.disk_usage(self.scratch_root.parent).free
            < self.safety["minimum_free_bytes"]
        ):
            raise CapacityDiagnosticError(
                "verification reached the minimum-free-disk safety line"
            )
        if scratch_bytes > self.scratch_byte_limit:
            raise CapacityDiagnosticError(
                "verification reached the scratch-byte safety line"
            )
        if _peak_rss_bytes() > self.safety["max_rss_bytes"]:
            raise CapacityDiagnosticError("verification reached the RSS safety line")
        if time.monotonic() - self.started_monotonic > self.safety["max_wall_seconds"]:
            raise CapacityDiagnosticError(
                "verification reached the wall-time safety line"
            )


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
    allowed_fields = set(CANDIDATE_STORE_FIELDS)
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
            unsealed = sorted(set(record) - allowed_fields)
            if unsealed:
                raise CapacityDiagnosticError(
                    "label-free structural store contains fields outside the "
                    f"frozen D1 allowlist at line {line_number}: {unsealed}"
                )
            nested_label_paths = candidate_store_label_paths(record)
            if nested_label_paths:
                raise CapacityDiagnosticError(
                    "label-free structural store contains recursively detected "
                    f"label paths at line {line_number}: {nested_label_paths}"
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


def _apply_diagnostic_scope(
    selection: StructuralSelection,
    *,
    scope: str,
) -> StructuralSelection:
    if scope == "census":
        return selection
    if scope != "witness":
        raise CapacityDiagnosticError(f"unsupported diagnostic scope: {scope}")
    witness = selection.records[0]
    if witness["record_id"] != WITNESS_RECORD_ID:
        raise CapacityDiagnosticError("frozen witness is not first in selection")
    exclusion_counts = Counter(selection.exclusion_reason_counts)
    exclusion_counts["scope_witness_only_unscheduled"] += len(selection.records) - 1
    return replace(
        selection,
        records=(witness,),
        eligible_endpoints=tuple(
            sorted(
                {
                    str(witness["source_sequence"]),
                    str(witness["candidate_sequence"]),
                }
            )
        ),
        selected_record_count=1,
        excluded_record_count=selection.source_record_count - 1,
        selected_record_ids_sha256=_record_ids_sha256([WITNESS_RECORD_ID]),
        exclusion_reason_counts=dict(sorted(exclusion_counts.items())),
    )


def _selection_manifest_payload(
    selection: StructuralSelection,
    *,
    scope: str,
) -> dict[str, Any]:
    """Render the only accepted selection manifest for a live recomputation."""

    return {
        "schema_version": "b0_capacity_selection.v1",
        "scope": scope,
        "source_store_role": "D1_SEALED_LABEL_FREE_CANDIDATE_STORE",
        "selection_algorithm": (
            "split_graph.select_split_eligible_records+edit_distance_gte_2"
        ),
        "record_order": "FROZEN_WITNESS_FIRST_THEN_CANONICAL_JSONL_LINE_ASCENDING",
        "source_record_count": selection.source_record_count,
        "split_eligible_record_count": selection.split_eligible_record_count,
        "selected_record_count": selection.selected_record_count,
        "excluded_record_count": selection.excluded_record_count,
        "record_ids_sha256": selection.record_ids_sha256,
        "selected_record_ids_sha256": selection.selected_record_ids_sha256,
        "eligible_endpoint_count": len(selection.eligible_endpoints),
        "state_universe_scope": (
            "FULL_CENSUS" if scope == "census" else "FROZEN_WITNESS_SUBSET"
        ),
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
        "canonical_label_store_opened_by_selection": False,
        "resume_supported": False,
        "resume_policy": ("replay_into_a_fresh_root_after_all_hashes_are_revalidated"),
    }


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
        "git_dir": str(git_dir.resolve()),
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
    *,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    progress_phase: str = "merge_unique_sequences",
) -> tuple[int, str]:
    partial = output.with_name(output.name + ".partial")
    count = 0
    input_count = 0
    digest = hashlib.sha256()
    previous: str | None = None
    with partial.open("x", encoding="utf-8", newline="") as handle:
        for input_index, sequence in enumerate(heapq.merge(*iterators), start=1):
            input_count = input_index
            if (
                progress_callback is not None
                and input_index % PROGRESS_CHECK_INTERVAL == 0
            ):
                progress_callback(
                    {
                        "phase": progress_phase,
                        "processed_input_sequences": input_index,
                        "unique_output_sequences": count,
                    }
                )
            if sequence == previous:
                continue
            handle.write(sequence + "\n")
            digest.update(sequence.encode("utf-8"))
            digest.update(b"\n")
            previous = sequence
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": progress_phase,
                    "processed_input_sequences": input_count,
                    "unique_output_sequences": count,
                    "final_callback": True,
                }
            )
    os.replace(partial, output)
    return count, digest.hexdigest()


def _record_state_universe(
    workspace: Path,
    layer_files: Sequence[str],
    *,
    expected_count: int,
    expected_digest: str,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> Path:
    output = workspace / "state_universe.tsv"
    iterators = [
        _iter_layer_sequences(workspace / relative) for relative in layer_files
    ]
    observed_count, observed_digest = _merge_unique_iterators(
        iterators,
        output,
        progress_callback=progress_callback,
        progress_phase="record_state_universe",
    )
    if observed_count != expected_count or observed_digest != expected_digest:
        raise CapacityDiagnosticError(
            "record state-universe merge disagrees with exact summary"
        )
    return output


def _alignment_statistics_payload(statistics: Any) -> dict[str, Any]:
    return {
        "minimum_edit_count": statistics.minimum_edit_count,
        "minimum_alignment_count": statistics.minimum_alignment_count,
        "evaluated_dag_cell_count": statistics.evaluated_dag_cell_count,
        "counts_exact": True,
    }


def _exact_summary_payload(summary: Any) -> dict[str, Any]:
    return {
        "reachable_node_count": summary.reachable_node_count,
        "reachable_transition_count": summary.reachable_transition_count,
        "minimum_state_path_count": summary.minimum_state_path_count,
        "evaluated_primitive_action_count": (summary.evaluated_primitive_action_count),
        "evaluated_state_dp_cell_count": summary.evaluated_state_dp_cell_count,
        "reachable_states_sha256": summary.reachable_states_sha256,
    }


def _frozen_gate_assessment(
    alignment_statistics: Mapping[str, Any],
    exact: Mapping[str, Any],
) -> dict[str, Any]:
    exceeded: list[str] = []
    if (
        int(alignment_statistics["evaluated_dag_cell_count"])
        > FROZEN_B0_LIMITS["max_dag_cells"]
    ):
        exceeded.append("max_dag_cells")
    if int(exact["reachable_node_count"]) > FROZEN_B0_LIMITS["max_reachable_states"]:
        exceeded.append("max_reachable_states")
    if (
        int(exact["evaluated_primitive_action_count"])
        > FROZEN_B0_LIMITS["max_neighbor_expansions"]
    ):
        exceeded.append("max_neighbor_expansions")
    if (
        int(exact["evaluated_state_dp_cell_count"])
        > FROZEN_B0_LIMITS["max_state_dp_cells"]
    ):
        exceeded.append("max_state_dp_cells")
    return {
        "would_pass_frozen_b0_limits": not exceeded,
        "exceeded_limits": exceeded,
    }


def _recompute_exact_record_evidence(
    *,
    record: Mapping[str, Any],
    ordinal: int,
    scratch_parent: Path,
    safety: Mapping[str, int],
    verification_started: float,
) -> dict[str, Any]:
    """Replay one claimed exact row from live endpoints in isolated scratch."""

    record_id = str(record["record_id"])
    source = str(record["source_sequence"])
    candidate = str(record["candidate_sequence"])
    known_distance = int(record["edit_distance"])
    try:
        with tempfile.TemporaryDirectory(
            prefix=(
                f".b0-record-verify-{ordinal:06d}-"
                f"{hashlib.sha256(record_id.encode()).hexdigest()[:12]}-"
            ),
            dir=str(scratch_parent.resolve(strict=True)),
        ) as temporary:
            scratch = Path(temporary)
            workspace = scratch / "workspace"
            guard = VerificationResourceGuard(
                scratch_root=scratch,
                scratch_byte_limit=safety["max_spill_bytes_per_record"],
                safety=safety,
                started_monotonic=verification_started,
            )
            guard.check()
            statistics = minimum_alignment_statistics(
                source,
                candidate,
                known_minimum_edit_count=known_distance,
                max_dag_cells=safety["max_dag_cells_per_record"],
            )
            guard.check()
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
                progress_callback=guard.check,
                progress_interval=PROGRESS_CHECK_INTERVAL,
            )
            exact = _exact_summary_payload(summary)
            state_path = _record_state_universe(
                workspace,
                summary.layer_files,
                expected_count=summary.reachable_node_count,
                expected_digest=summary.reachable_states_sha256,
                progress_callback=guard.check,
            )
            guard.check()
            return {
                "alignment_statistics": _alignment_statistics_payload(statistics),
                "exact": exact,
                "frozen_gate_assessment": _frozen_gate_assessment(
                    _alignment_statistics_payload(statistics),
                    exact,
                ),
                "state_universe": {
                    "bytes": state_path.stat().st_size,
                    "sha256": _sha256_file(state_path),
                },
            }
    except (
        CapacityDiagnosticError,
        OSError,
        PathStateError,
        StreamingPathStateError,
    ) as error:
        raise CapacityDiagnosticError(
            f"exact row replay failed for {record_id}: {error}"
        ) from error


def _merge_global_universe(
    *,
    run_root: Path,
    endpoints: Sequence[str],
    record_state_files: Sequence[Path],
    max_open_chunks: int,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
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
                progress_callback=progress_callback,
                progress_phase="global_state_universe_intermediate_merge",
            )
            following.append(output)
        current = following
        merge_pass += 1
    output = global_root / "state_universe.tsv"
    count, digest = _merge_unique_iterators(
        [_iter_sequence_file(path) for path in current],
        output,
        progress_callback=progress_callback,
        progress_phase="global_state_universe_final_merge",
    )
    return output, count, digest


def _sequence_file_stats(path: Path) -> tuple[int, str]:
    """Validate canonical sequence ordering and return newline-bound statistics."""

    count = 0
    digest = hashlib.sha256()
    for sequence in _iter_sequence_file(path):
        count += 1
        digest.update(sequence.encode("utf-8"))
        digest.update(b"\n")
    return count, digest.hexdigest()


def _recompute_global_union(
    *,
    input_paths: Sequence[Path],
    safety: Mapping[str, int],
    scratch_parent: Path,
    verification_started: float | None = None,
) -> tuple[int, str, int, str]:
    """Independently replay a bounded fan-in union outside the sealed bundle."""

    if not input_paths:
        raise CapacityDiagnosticError("global-union verification has no inputs")
    max_open_chunks = safety["max_open_chunks"]
    if max_open_chunks < 2:
        raise CapacityDiagnosticError(
            "global-union verification requires max_open_chunks >= 2"
        )
    started = time.monotonic() if verification_started is None else verification_started
    with tempfile.TemporaryDirectory(
        prefix=".b0-capacity-verify-",
        dir=str(scratch_parent.resolve(strict=True)),
    ) as temporary:
        scratch = Path(temporary)
        guard = VerificationResourceGuard(
            scratch_root=scratch,
            scratch_byte_limit=safety["max_spill_bytes"],
            safety=safety,
            started_monotonic=started,
        )
        guard.check()
        current = list(input_paths)
        merge_pass = 0
        while len(current) > max_open_chunks:
            following: list[Path] = []
            for group_index, offset in enumerate(
                range(0, len(current), max_open_chunks)
            ):
                output = scratch / (f"pass-{merge_pass:03d}-run-{group_index:06d}.tsv")
                _merge_unique_iterators(
                    [
                        _iter_sequence_file(path)
                        for path in current[offset : offset + max_open_chunks]
                    ],
                    output,
                    progress_callback=guard.check,
                    progress_phase="bundle_verification_intermediate_merge",
                )
                following.append(output)
            current = following
            merge_pass += 1
        output = scratch / "recomputed-global-state-universe.tsv"
        count, digest = _merge_unique_iterators(
            [_iter_sequence_file(path) for path in current],
            output,
            progress_callback=guard.check,
            progress_phase="bundle_verification_final_merge",
        )
        guard.check()
        return count, digest, output.stat().st_size, _sha256_file(output)


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


def _record_workspace_path(
    run_root: Path,
    *,
    ordinal: int,
    record_id: str,
) -> Path:
    return (
        run_root
        / "record_workspaces"
        / f"{ordinal:06d}-{hashlib.sha256(record_id.encode()).hexdigest()[:12]}"
    )


def _run_record(
    *,
    record: Mapping[str, Any],
    ordinal: int,
    run_root: Path,
    safety: Mapping[str, int],
    watchdog: ResourceWatchdog,
) -> tuple[dict[str, Any], Path | None]:
    record_id = str(record["record_id"])
    source = str(record["source_sequence"])
    candidate = str(record["candidate_sequence"])
    known_distance = int(record["edit_distance"])
    started = time.monotonic()
    workspace = _record_workspace_path(
        run_root,
        ordinal=ordinal,
        record_id=record_id,
    )
    workspace.parent.mkdir(exist_ok=True)
    alignment_statistics: dict[str, Any]
    state_file: Path | None = None
    try:
        statistics = minimum_alignment_statistics(
            source,
            candidate,
            known_minimum_edit_count=known_distance,
            max_dag_cells=safety["max_dag_cells_per_record"],
        )
        alignment_statistics = _alignment_statistics_payload(statistics)
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
            progress_callback=watchdog.prototype_callback,
            progress_interval=PROGRESS_CHECK_INTERVAL,
        )
        exact = _exact_summary_payload(summary)
        gate_assessment = _frozen_gate_assessment(
            alignment_statistics,
            exact,
        )
        state_file = _record_state_universe(
            workspace,
            summary.layer_files,
            expected_count=summary.reachable_node_count,
            expected_digest=summary.reachable_states_sha256,
            progress_callback=watchdog.prototype_callback,
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
    except (StreamingPathStateError, PathStateError) as error:
        message = str(error)
        if isinstance(error, StreamingPathStateError) and message.startswith(
            "SAFE_RESOURCE_PAUSE:"
        ):
            raise SafeCapacityPause(
                message.removeprefix("SAFE_RESOURCE_PAUSE:").strip()
            ) from error
        if "spill bytes" in message:
            raise SafeCapacityPause(message) from error
        dimension = _failure_dimension(message)
        if dimension is None:
            raise CapacityDiagnosticError(
                f"unclassified exact prototype failure for {record_id}: {message}"
            ) from error
        if not isinstance(error, StreamingPathStateError):
            if dimension != "max_dag_cells":
                raise CapacityDiagnosticError(
                    "path-state oracle failure was not the typed DAG capacity stop "
                    f"for {record_id}: {message}"
                ) from error
            alignment_statistics = {
                "minimum_edit_count": known_distance,
                "minimum_alignment_count": None,
                "evaluated_dag_cell_count": (safety["max_dag_cells_per_record"] + 1),
                "counts_exact": False,
            }
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
        observed_metric = (
            int(alignment_statistics["evaluated_dag_cell_count"])
            if dimension == "max_dag_cells"
            else int(
                lower_bound[
                    {
                        "max_reachable_states": "reachable_node_count",
                        "max_neighbor_expansions": ("evaluated_primitive_action_count"),
                        "max_state_dp_cells": "evaluated_state_dp_cell_count",
                    }[dimension]
                ]
            )
        )
        diagnostic_limit = int(safety[DIAGNOSTIC_LIMIT_FIELD_BY_DIMENSION[dimension]])
        observed_lower_bound = max(
            FROZEN_B0_LIMITS[dimension] + 1,
            diagnostic_limit + 1,
            observed_metric,
        )
        stop = {
            "stop_rule": "STOP_RULE_B0_PATH_STATE_COMPLEXITY",
            "dimension": dimension,
            "frozen_limit": FROZEN_B0_LIMITS[dimension],
            "observed_lower_bound": observed_lower_bound,
            "message": message,
        }
        gate_assessment = {
            "would_pass_frozen_b0_limits": False,
            "exceeded_limits": [dimension],
        }
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
        "alignment_statistics": alignment_statistics,
        "outcome": outcome,
        "state_universe_artifact": (
            _artifact_ref(state_file, root=run_root) if state_file is not None else None
        ),
        "capacity": capacity,
        "frozen_gate_assessment": gate_assessment,
        "evidence_semantics": evidence,
        "stop": stop,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": _peak_rss_bytes(),
        "spill_bytes": spill_bytes,
    }
    return row, state_file


@dataclass
class FileDescriptorCapture:
    """Capture OS-level stdout/stderr through the terminal compute event."""

    stdout_path: Path
    stderr_path: Path
    saved_stdout_fd: int | None = None
    saved_stderr_fd: int | None = None
    stdout_handle: Any = None
    stderr_handle: Any = None

    def start(self) -> None:
        if self.saved_stdout_fd is not None or self.saved_stderr_fd is not None:
            raise CapacityDiagnosticError("file-descriptor capture already started")
        sys.stdout.flush()
        sys.stderr.flush()
        self.stdout_handle = self.stdout_path.open("ab", buffering=0)
        self.stderr_handle = self.stderr_path.open("ab", buffering=0)
        try:
            self.saved_stdout_fd = os.dup(1)
            self.saved_stderr_fd = os.dup(2)
            os.dup2(self.stdout_handle.fileno(), 1)
            os.dup2(self.stderr_handle.fileno(), 2)
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        if self.saved_stdout_fd is not None:
            os.dup2(self.saved_stdout_fd, 1)
            os.close(self.saved_stdout_fd)
            self.saved_stdout_fd = None
        if self.saved_stderr_fd is not None:
            os.dup2(self.saved_stderr_fd, 2)
            os.close(self.saved_stderr_fd)
            self.saved_stderr_fd = None
        for handle in (self.stdout_handle, self.stderr_handle):
            if handle is not None:
                os.fsync(handle.fileno())
                handle.close()
        self.stdout_handle = None
        self.stderr_handle = None


_ACTIVE_FD_CAPTURE: FileDescriptorCapture | None = None


def _stop_active_fd_capture() -> None:
    global _ACTIVE_FD_CAPTURE
    if _ACTIVE_FD_CAPTURE is not None:
        _ACTIVE_FD_CAPTURE.stop()
        _ACTIVE_FD_CAPTURE = None


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
        "resource_limit_scope": RESOURCE_LIMIT_SCOPE,
    }


def _validate_d1_snapshot_trust(
    *,
    snapshot_path: Path,
    acceptance_path: Path,
    project_root: Path,
    git_dir: Path,
) -> dict[str, Any]:
    """Recompute the frozen D1 snapshot before structural selection.

    The existing validator hashes the canonical label store as opaque bytes
    while rebuilding the snapshot, but never parses that JSONL or exposes a
    value to this diagnostic.  Selection itself opens only the sealed
    label-free candidate store.
    """

    try:
        relative_snapshot = snapshot_path.resolve(strict=True).relative_to(
            project_root.resolve(strict=True)
        )
    except ValueError as error:
        raise CapacityDiagnosticError(
            "D1 snapshot must be a tracked file inside the project root"
        ) from error
    command_prefix = [
        "git",
        f"--git-dir={git_dir}",
        f"--work-tree={project_root}",
    ]
    tracked = subprocess.run(
        [
            *command_prefix,
            "ls-files",
            "--error-unmatch",
            "--",
            relative_snapshot.as_posix(),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if tracked.returncode != 0:
        raise CapacityDiagnosticError(
            "D1 snapshot is not tracked by the authoritative Git repository"
        )
    committed_bytes = subprocess.run(
        [
            *command_prefix,
            "show",
            f"HEAD:{relative_snapshot.as_posix()}",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    live_bytes = snapshot_path.read_bytes()
    if committed_bytes != live_bytes:
        raise CapacityDiagnosticError(
            "D1 snapshot differs from the clean authoritative Git blob"
        )
    prior_git_dir = os.environ.get("GIT_DIR")
    prior_git_work_tree = os.environ.get("GIT_WORK_TREE")
    os.environ["GIT_DIR"] = str(git_dir.resolve(strict=True))
    os.environ["GIT_WORK_TREE"] = str(project_root.resolve(strict=True))
    try:
        validation_errors = validate_snapshot(
            snapshot_path,
            repo_root=project_root,
        )
    finally:
        if prior_git_dir is None:
            os.environ.pop("GIT_DIR", None)
        else:
            os.environ["GIT_DIR"] = prior_git_dir
        if prior_git_work_tree is None:
            os.environ.pop("GIT_WORK_TREE", None)
        else:
            os.environ["GIT_WORK_TREE"] = prior_git_work_tree
    if validation_errors:
        raise CapacityDiagnosticError(
            "D1 snapshot exact live recomputation failed: "
            + "; ".join(validation_errors)
        )
    acceptance = _read_json(acceptance_path)
    acceptance_errors = validate_phase_acceptance(
        "D1",
        acceptance,
        require_pass=True,
    )
    if acceptance_errors:
        raise CapacityDiagnosticError(
            "D1 acceptance semantic validation failed: " + "; ".join(acceptance_errors)
        )
    return {
        "schema_version": "b0_d1_snapshot_validation.v1",
        "status": "PASS",
        "snapshot": _artifact_ref(snapshot_path),
        "tracked_repository_path": relative_snapshot.as_posix(),
        "tracked_git_blob_sha256": _sha256_bytes(committed_bytes),
        "authoritative_git_dir": str(git_dir.resolve(strict=True)),
        "authoritative_git_work_tree": str(project_root.resolve(strict=True)),
        "native_worktree_git_metadata_used": False,
        "exact_live_recomputation_passed": True,
        "d1_acceptance_semantic_passed": True,
        "selection_store": "D1_SEALED_LABEL_FREE_CANDIDATE_STORE",
        "canonical_label_store_access": {
            "mode": "OPAQUE_SHA256_INTEGRITY_VALIDATION_ONLY",
            "opened_for_integrity_hash": True,
            "jsonl_parsed": False,
            "label_values_accessed": False,
            "used_for_selection": False,
            "used_for_capacity": False,
        },
        "scientific_result_claimed": False,
    }


def _find_required_artifact(
    snapshot: Mapping[str, Any],
    suffix: str,
) -> tuple[Path, dict[str, Any]]:
    artifacts = snapshot.get("required_artifacts")
    if not isinstance(artifacts, Mapping):
        raise CapacityDiagnosticError("D1 snapshot lacks required-artifact bindings")
    matches = [value for key, value in artifacts.items() if str(key).endswith(suffix)]
    if len(matches) != 1 or not isinstance(matches[0], Mapping):
        raise CapacityDiagnosticError(
            f"D1 snapshot has ambiguous required artifact: {suffix}"
        )
    expected = matches[0]
    path = Path(str(expected["path"]))
    actual = _artifact_ref(path)
    if actual["bytes"] != expected.get("bytes") or actual["sha256"] != expected.get(
        "sha256"
    ):
        raise CapacityDiagnosticError(
            f"D1 required artifact drifted from snapshot binding: {suffix}"
        )
    return path, actual


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


def _render_replay_script(
    argv: Sequence[str],
    launch_cwd: Path,
) -> str:
    rendered: list[str] = []
    replace_next = False
    replacements = 0
    for token in argv:
        if replace_next:
            rendered.append('"${1}"')
            replace_next = False
            replacements += 1
        elif token.startswith("--output-root="):
            rendered.append('--output-root="${1}"')
            replacements += 1
        else:
            rendered.append(shlex.quote(token))
            if token == "--output-root":
                replace_next = True
    if replace_next or replacements != 1:
        raise CapacityDiagnosticError(
            "exact argv must contain exactly one replaceable --output-root"
        )
    return (
        "#!/bin/sh\n"
        "set -eu\n"
        'if [ "$#" -ne 1 ]; then\n'
        '  echo "usage: replay.sh FRESH_OUTPUT_ROOT" >&2\n'
        "  exit 2\n"
        "fi\n"
        f"cd {shlex.quote(str(launch_cwd.resolve(strict=True)))}\n"
        "exec " + " ".join(rendered) + "\n"
    )


def _write_replay_script(
    *,
    run_root: Path,
    argv: Sequence[str],
    launch_cwd: Path,
) -> None:
    body = _render_replay_script(argv, launch_cwd)
    path = run_root / "replay.sh"
    _write_text_exclusive(path, body)
    path.chmod(0o750)


def _diagnostic_summary(
    *,
    run_root: Path,
    diagnostic_scope: str,
    selection: StructuralSelection,
    rows: Sequence[Mapping[str, Any]],
    global_state_count: int,
    global_state_digest: str,
    global_state_exact: bool,
    global_state_role: str,
    global_state_path: Path,
) -> dict[str, Any]:
    allowed_global_roles = {
        "EXACT",
        "LOWER_BOUND_FROM_ENDPOINTS_AND_EXACT_COMPLETED_RECORDS",
        "LOWER_BOUND_FROM_ELIGIBLE_ENDPOINTS_ONLY",
    }
    if global_state_role not in allowed_global_roles:
        raise CapacityDiagnosticError("global state-universe role is invalid")
    if (global_state_role == "EXACT") is not global_state_exact:
        raise CapacityDiagnosticError(
            "global state-universe role contradicts exactness"
        )
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
        "diagnostic_scope": diagnostic_scope,
        "state_universe_scope": (
            "FULL_CENSUS" if diagnostic_scope == "census" else "FROZEN_WITNESS_SUBSET"
        ),
        "source_record_count": selection.source_record_count,
        "split_eligible_record_count": (selection.split_eligible_record_count),
        "scheduled_multi_edit_record_count": selection.selected_record_count,
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
        "full_census_state_universe_exact": (
            diagnostic_scope == "census" and global_state_exact
        ),
        "global_state_universe_role": global_state_role,
        "global_state_universe_artifact": _artifact_ref(
            global_state_path,
            root=run_root,
        ),
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


def _bundle_path(run_root: Path, relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise CapacityDiagnosticError(
            f"bundle reference must be a safe relative path: {relative}"
        )
    try:
        resolved = (run_root / raw).resolve(strict=True)
    except OSError as error:
        raise CapacityDiagnosticError(
            f"bundle reference does not exist: {relative}"
        ) from error
    try:
        resolved.relative_to(run_root.resolve(strict=True))
    except ValueError as error:
        raise CapacityDiagnosticError(
            f"bundle reference escapes the run root: {relative}"
        ) from error
    return resolved


def _verify_bundle_ref(run_root: Path, reference: Mapping[str, Any]) -> Path:
    path = _bundle_path(run_root, str(reference.get("path") or ""))
    actual = _artifact_ref(path, root=run_root)
    if actual != dict(reference):
        raise CapacityDiagnosticError(
            f"bundle artifact reference mismatch: {reference.get('path')}"
        )
    return path


def _read_jsonl_objects(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise CapacityDiagnosticError(
                    f"bundle JSONL parse error at {path}:{line_number}"
                ) from error
            if not isinstance(row, Mapping):
                raise CapacityDiagnosticError(
                    f"bundle JSONL row is not an object at {path}:{line_number}"
                )
            rows.append(row)
    return rows


def _verify_external_ref(
    reference: Mapping[str, Any],
    path: Path,
    *,
    root: Path | None = None,
) -> None:
    actual = _artifact_ref(path, root=root)
    if dict(reference) != actual:
        raise CapacityDiagnosticError(
            f"external artifact reference mismatch: {reference.get('path')}"
        )


def _validate_replay_provenance(
    *,
    run_root: Path,
    manifest: Mapping[str, Any],
) -> argparse.Namespace:
    provenance = manifest["provenance"]
    exact_argv = provenance["exact_argv"]
    if (
        not isinstance(exact_argv, list)
        or len(exact_argv) < 2
        or any(not isinstance(token, str) for token in exact_argv)
        or provenance.get("exact_argv_sha256") != _canonical_sha256(exact_argv)
    ):
        raise CapacityDiagnosticError("bundle exact argv is invalid")
    try:
        launcher = Path(exact_argv[0]).resolve(strict=True)
        entrypoint = Path(exact_argv[1]).resolve(strict=True)
        launch_cwd = Path(str(provenance["launch_cwd"])).resolve(strict=True)
        project_root = Path(str(provenance["git"]["project_root"])).resolve(strict=True)
        git_dir = Path(str(provenance["git"]["git_dir"])).resolve(strict=True)
    except (KeyError, OSError) as error:
        raise CapacityDiagnosticError(
            "bundle launcher, entrypoint, cwd, or Git binding does not exist"
        ) from error
    if launch_cwd != project_root:
        raise CapacityDiagnosticError(
            "bundle launch cwd differs from the authoritative project root"
        )
    if exact_argv[0] != str(launcher) or exact_argv[1] != str(entrypoint):
        raise CapacityDiagnosticError(
            "bundle launcher or entrypoint is not captured canonically"
        )
    expected_entrypoint = (
        project_root / "scripts/data/diagnose_b0_path_capacity.py"
    ).resolve(strict=True)
    if entrypoint != expected_entrypoint:
        raise CapacityDiagnosticError(
            "bundle exact argv does not use the authoritative entrypoint"
        )
    try:
        replay_args = _parse_args(exact_argv[2:])
    except SystemExit as error:
        raise CapacityDiagnosticError(
            "bundle exact argv cannot be parsed by the frozen entrypoint"
        ) from error
    expected_bindings = {
        "output_root": run_root.resolve(strict=True),
        "project_root": project_root,
        "git_dir": git_dir,
    }
    for field, expected in expected_bindings.items():
        try:
            observed = Path(getattr(replay_args, field)).resolve(strict=True)
        except OSError as error:
            raise CapacityDiagnosticError(
                f"bundle replay argument does not exist: {field}"
            ) from error
        if observed != expected:
            raise CapacityDiagnosticError(
                f"bundle replay argument differs from provenance: {field}"
            )
    selection = manifest["selection"]
    if (
        replay_args.expected_code_commit != provenance["git"]["head"]
        or replay_args.scope != selection["diagnostic_scope"]
        or replay_args.parent_diagnostic_id != manifest["parent_diagnostic_id"]
    ):
        raise CapacityDiagnosticError(
            "bundle replay scope, parent, or code commit differs from manifest"
        )

    resolved_config_path = _verify_bundle_ref(
        run_root,
        provenance["resolved_config"],
    )
    resolved_config = _read_json(resolved_config_path)
    if _validate_config(_read_json(replay_args.config)) != resolved_config:
        raise CapacityDiagnosticError(
            "bundle replay config differs from the resolved frozen config"
        )

    input_manifest = _read_json(run_root / "provenance/input_manifest.json")
    _verify_external_ref(input_manifest["contract"], replay_args.contract)
    _verify_external_ref(provenance["d1_snapshot"], replay_args.d1_snapshot)
    _verify_external_ref(provenance["d1_acceptance"], replay_args.d1_acceptance)
    if (
        input_manifest.get("d1_snapshot") != provenance["d1_snapshot"]
        or input_manifest.get("d1_snapshot_validation")
        != provenance["d1_snapshot_validation"]
        or input_manifest.get("d1_acceptance") != provenance["d1_acceptance"]
        or input_manifest.get("d1_build_manifest") != provenance["d1_build_manifest"]
        or input_manifest.get("ambiguity_report") != provenance["ambiguity_report"]
        or input_manifest.get("label_fields_read") != []
    ):
        raise CapacityDiagnosticError(
            "bundle input manifest differs from terminal provenance"
        )
    structural_store = provenance["structural_store"]
    actual_structural = _artifact_ref(replay_args.candidate_store)
    if any(
        structural_store.get(field) != actual_structural[field]
        for field in ("path", "bytes", "sha256")
    ):
        raise CapacityDiagnosticError(
            "bundle replay candidate store differs from structural provenance"
        )
    input_structural = input_manifest.get("structural_store", {})
    if any(
        input_structural.get(input_field) != structural_store.get(provenance_field)
        for input_field, provenance_field in (
            ("path", "path"),
            ("bytes", "bytes"),
            ("sha256", "sha256"),
            ("records", "record_count"),
            ("record_ids_sha256", "record_ids_sha256"),
        )
    ):
        raise CapacityDiagnosticError(
            "bundle input manifest structural store is inconsistent"
        )
    input_canonical = input_manifest.get("canonical_store_metadata_only", {})
    canonical = provenance["canonical_store_metadata_only"]
    if any(
        input_canonical.get(input_field) != canonical.get(provenance_field)
        for input_field, provenance_field in (
            ("path", "path"),
            ("bytes", "bytes"),
            ("sha256", "sha256"),
            ("records", "record_count"),
            ("record_ids_sha256", "record_ids_sha256"),
            ("selection_opened", "selection_opened"),
            ("capacity_algorithm_opened", "capacity_algorithm_opened"),
            ("opaque_integrity_hash_opened", "opaque_integrity_hash_opened"),
            ("jsonl_parsed", "jsonl_parsed"),
            ("label_values_accessed", "label_values_accessed"),
        )
    ):
        raise CapacityDiagnosticError(
            "bundle input manifest canonical-store metadata is inconsistent"
        )
    if manifest["goal_contract"]["sha256"] != input_manifest["contract"]["sha256"]:
        raise CapacityDiagnosticError(
            "bundle contract hash differs between manifest and replay input"
        )

    runtime_path = _verify_bundle_ref(run_root, provenance["runtime_manifest"])
    runtime = _read_json(runtime_path)
    launcher_binding_path = _verify_bundle_ref(
        run_root,
        provenance["python_launcher"],
    )
    launcher_binding = _read_json(launcher_binding_path)
    actual_launcher = {
        "path": str(launcher),
        "bytes": launcher.stat().st_size,
        "sha256": _sha256_file(launcher),
    }
    if launcher_binding != actual_launcher or (
        runtime.get("executable") != actual_launcher["path"]
        or runtime.get("executable_bytes") != actual_launcher["bytes"]
        or runtime.get("executable_sha256") != actual_launcher["sha256"]
        or runtime.get("resource_limit_scope") != RESOURCE_LIMIT_SCOPE
    ):
        raise CapacityDiagnosticError(
            "bundle Python launcher or runtime binding is invalid"
        )

    code_files = provenance["code_files"]
    code_manifest = _read_json(run_root / "provenance/code_manifest.json")
    if code_manifest != {
        "code_commit": provenance["git"]["head"],
        "files": code_files,
    }:
        raise CapacityDiagnosticError(
            "bundle code manifest differs from terminal provenance"
        )
    entrypoint_refs = [
        reference
        for reference in code_files
        if reference.get("role") == "DIAGNOSTIC_ENTRYPOINT"
    ]
    if len(entrypoint_refs) != 1:
        raise CapacityDiagnosticError(
            "bundle code manifest lacks one diagnostic entrypoint"
        )
    for reference in code_files:
        relative = Path(str(reference.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise CapacityDiagnosticError(
                "bundle code-file reference is not repository-relative"
            )
        expected = {
            **_artifact_ref(project_root / relative, root=project_root),
            "role": reference["role"],
        }
        if dict(reference) != expected:
            raise CapacityDiagnosticError(
                f"bundle code-file reference drifted: {relative}"
            )
    if entrypoint_refs[0]["path"] != ("scripts/data/diagnose_b0_path_capacity.py"):
        raise CapacityDiagnosticError(
            "bundle entrypoint role is bound to the wrong code file"
        )

    command = _read_json(run_root / "command.json")
    if command != {
        "exact_argv": exact_argv,
        "exact_argv_sha256": provenance["exact_argv_sha256"],
        "launch_cwd": str(launch_cwd),
        "output_root_replay_policy": ("replace_with_a_fresh_nonexistent_output_root"),
    }:
        raise CapacityDiagnosticError("command manifest differs from provenance")
    replay_path = run_root / "replay.sh"
    if replay_path.read_text(encoding="utf-8") != _render_replay_script(
        exact_argv,
        launch_cwd,
    ):
        raise CapacityDiagnosticError(
            "replay script is not an exact rendering of captured argv and cwd"
        )
    if replay_path.stat().st_mode & 0o111 == 0:
        raise CapacityDiagnosticError("replay script is not executable")
    return replay_args


def _validate_frozen_witness_row(row: Mapping[str, Any]) -> None:
    """Require every terminal bundle to retain the frozen exact first witness."""

    if (
        row.get("ordinal") != 0
        or row.get("record_id") != WITNESS_RECORD_ID
        or row.get("dataset_id") != "GSE217518"
        or row.get("region") != "five_utr"
        or row.get("canonical_jsonl_line") != WITNESS_CANONICAL_JSONL_LINE
        or row.get("source_length") != len(WITNESS_SOURCE)
        or row.get("candidate_length") != len(WITNESS_CANDIDATE)
        or row.get("source_sequence_sha256")
        != _sha256_bytes(WITNESS_SOURCE.encode("utf-8"))
        or row.get("candidate_sequence_sha256")
        != _sha256_bytes(WITNESS_CANDIDATE.encode("utf-8"))
        or row.get("alignment_statistics", {}).get("minimum_edit_count") != 15
        or row.get("alignment_statistics", {}).get("minimum_alignment_count") != 2_340
        or row.get("alignment_statistics", {}).get("counts_exact") is not True
        or row.get("outcome") != "EXACT_COMPLETED"
        or row.get("capacity", {}).get("exact") != WITNESS_EXPECTED
        or row.get("capacity", {}).get("lower_bound") is not None
        or row.get("evidence_semantics", {}).get("counts_exact") is not True
        or row.get("evidence_semantics", {}).get("state_set_complete") is not True
        or row.get("stop") is not None
    ):
        raise CapacityDiagnosticError(
            "terminal bundle does not retain the frozen exact first witness"
        )


def _validate_external_parent_authorization(
    *,
    parent_id: str,
    parent_authorization: Mapping[str, Any] | None,
    child_manifest: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    schema_path: Path,
) -> None:
    """Validate the actual frozen witness bundle, not only its path strings."""

    parent_root = ALLOWED_OUTPUT_PARENT / parent_id
    expected_parent_paths = {
        "diagnostic_manifest": parent_root / "diagnostic_manifest.json",
        "verified_marker": parent_root / "VERIFIED",
        "bundle_seal": parent_root / "bundle_seal.json",
        "terminal_lock": parent_root / "terminal.lock",
        "process_result": parent_root / "provenance/process_result.json",
    }
    if (
        not isinstance(parent_authorization, Mapping)
        or parent_authorization.get("diagnostic_id") != parent_id
        or any(
            parent_authorization.get(field, {}).get("path") != str(path)
            for field, path in expected_parent_paths.items()
        )
    ):
        raise CapacityDiagnosticError(
            "census bundle does not freeze the exact parent authorization"
        )
    if not parent_root.is_dir():
        raise CapacityDiagnosticError(
            "census parent authorization directory does not exist"
        )

    _validate_bundle(run_root=parent_root, schema_path=schema_path)
    _validate_verified_marker(parent_root)
    parent_manifest = _read_json(parent_root / "diagnostic_manifest.json")
    parent_config = _read_json(parent_root / "resolved_config.json")
    child_provenance = child_manifest["provenance"]
    parent_provenance = parent_manifest.get("provenance", {})
    if (
        parent_manifest.get("diagnostic_id") != parent_id
        or parent_manifest.get("state") != "COMPLETED"
        or parent_manifest.get("parent_diagnostic_id") is not None
        or parent_manifest.get("selection", {}).get("diagnostic_scope") != "witness"
        or parent_manifest.get("goal_contract") != child_manifest.get("goal_contract")
        or parent_manifest.get("algorithm_contract")
        != child_manifest.get("algorithm_contract")
        or parent_provenance.get("git") != child_provenance.get("git")
        or parent_provenance.get("d1_snapshot") != child_provenance.get("d1_snapshot")
        or parent_provenance.get("d1_acceptance")
        != child_provenance.get("d1_acceptance")
        or parent_provenance.get("structural_store")
        != child_provenance.get("structural_store")
        or parent_config != dict(resolved_config)
    ):
        raise CapacityDiagnosticError(
            "census parent is not the completed witness for this code/data/config"
        )
    try:
        for field, path in expected_parent_paths.items():
            _verify_external_ref(parent_authorization[field], path)
    except (KeyError, OSError, CapacityDiagnosticError) as error:
        raise CapacityDiagnosticError(
            "census parent authorization bytes or SHA-256 do not match"
        ) from error


def _validate_bundle(*, run_root: Path, schema_path: Path) -> None:
    """Independently recompute accounting, references, and the detached seal."""

    verification_started = time.monotonic()
    manifest_path = run_root / "diagnostic_manifest.json"
    manifest = _read_json(manifest_path)
    _validate_manifest(manifest, schema_path)
    provenance = manifest["provenance"]
    replay_args = _validate_replay_provenance(
        run_root=run_root,
        manifest=manifest,
    )
    verified_bundle_refs: dict[str, Path] = {}
    for field in (
        "resolved_config",
        "runtime_manifest",
        "python_launcher",
        "d1_snapshot_validation",
        "process_result",
    ):
        verified_bundle_refs[field] = _verify_bundle_ref(
            run_root,
            provenance[field],
        )
    resolved_config = _validate_config(
        _read_json(verified_bundle_refs["resolved_config"])
    )
    saved_d1_validation = _read_json(verified_bundle_refs["d1_snapshot_validation"])
    live_d1_validation = _validate_d1_snapshot_trust(
        snapshot_path=replay_args.d1_snapshot,
        acceptance_path=replay_args.d1_acceptance,
        project_root=replay_args.project_root,
        git_dir=replay_args.git_dir,
    )
    if saved_d1_validation != live_d1_validation:
        raise CapacityDiagnosticError(
            "saved D1 validation differs from exact live recomputation"
        )
    selection = manifest["selection"]
    recomputed_selection = _load_structural_selection(
        candidate_store=replay_args.candidate_store,
        d1_snapshot=replay_args.d1_snapshot,
        minimum_edit_distance=resolved_config["selection"]["minimum_edit_distance"],
        witness_record_id=WITNESS_RECORD_ID,
    )
    recomputed_selection = _apply_diagnostic_scope(
        recomputed_selection,
        scope=selection["diagnostic_scope"],
    )
    accounting = manifest["accounting"]
    parent_id = manifest["parent_diagnostic_id"]
    parent_authorization = manifest["parent_authorization"]
    if selection["diagnostic_scope"] == "witness":
        if parent_id is not None or parent_authorization is not None:
            raise CapacityDiagnosticError(
                "witness bundle may not claim a parent authorization"
            )
    else:
        _validate_external_parent_authorization(
            parent_id=str(parent_id),
            parent_authorization=parent_authorization,
            child_manifest=manifest,
            resolved_config=resolved_config,
            schema_path=schema_path,
        )
    selection_manifest_path = _verify_bundle_ref(
        run_root,
        selection["selection_manifest"],
    )
    selection_manifest = _read_json(selection_manifest_path)
    expected_selection_manifest = _selection_manifest_payload(
        recomputed_selection,
        scope=selection["diagnostic_scope"],
    )
    if selection_manifest != expected_selection_manifest:
        raise CapacityDiagnosticError(
            "bundle selection differs from live D1 structural recomputation"
        )
    expected_terminal_selection = {
        "source_store_role": "D1_SEALED_LABEL_FREE_CANDIDATE_STORE",
        "selection_algorithm": (
            "split_graph.select_split_eligible_records+edit_distance_gte_2"
        ),
        "diagnostic_scope": selection["diagnostic_scope"],
        "state_universe_scope": (
            "FULL_CENSUS"
            if selection["diagnostic_scope"] == "census"
            else "FROZEN_WITNESS_SUBSET"
        ),
        "regions": list(REGIONS),
        "record_order": ("FROZEN_WITNESS_FIRST_THEN_CANONICAL_JSONL_LINE_ASCENDING"),
        "source_record_count": recomputed_selection.source_record_count,
        "selected_record_count": recomputed_selection.selected_record_count,
        "excluded_record_count": recomputed_selection.excluded_record_count,
        "selected_record_ids_sha256": (recomputed_selection.selected_record_ids_sha256),
        "label_fields_read": [],
        "canonical_label_store_opened_by_selection": False,
        "selection_manifest": _artifact_ref(
            selection_manifest_path,
            root=run_root,
        ),
    }
    if dict(selection) != expected_terminal_selection:
        raise CapacityDiagnosticError(
            "terminal selection differs from live D1 structural recomputation"
        )
    records_path = _verify_bundle_ref(run_root, accounting["record_results"])
    rows = _read_jsonl_objects(records_path)
    schema = _read_json(schema_path)
    row_validator = jsonschema.Draft202012Validator(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$ref": "#/$defs/recordResult",
            "$defs": schema["$defs"],
        },
        format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER,
    )
    for ordinal, row in enumerate(rows):
        row_errors = list(row_validator.iter_errors(row))
        if row_errors:
            raise CapacityDiagnosticError(
                f"terminal record row {ordinal} failed schema validation: "
                f"{row_errors[0].message}"
            )
    shard_index = _read_json(
        _verify_bundle_ref(run_root, accounting["record_shard_index"])
    )
    summary = _read_json(_verify_bundle_ref(run_root, accounting["capacity_summary"]))

    source_count = recomputed_selection.source_record_count
    selected_count = recomputed_selection.selected_record_count
    excluded_count = recomputed_selection.excluded_record_count
    if source_count != selected_count + excluded_count:
        raise CapacityDiagnosticError(
            "bundle selection accounting does not reconcile source records"
        )

    outcomes = Counter(str(row.get("outcome")) for row in rows)
    exact_count = outcomes["EXACT_COMPLETED"]
    lower_count = outcomes["LOWER_BOUND_STOPPED"]
    scheduled = int(accounting["scheduled_record_count"])
    terminal = int(accounting["terminal_record_count"])
    all_accounted = terminal == scheduled
    expected_accounting = {
        "scheduled_record_count": selected_count,
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
    }
    for field, expected in expected_accounting.items():
        if accounting.get(field) != expected:
            raise CapacityDiagnosticError(
                f"bundle terminal accounting mismatch: {field}"
            )
    if terminal != exact_count + lower_count:
        raise CapacityDiagnosticError(
            "bundle outcome counts do not sum to terminal records"
        )
    if manifest["state"] == "COMPLETED" and not all_accounted:
        raise CapacityDiagnosticError(
            "completed bundle contains unaccounted scheduled records"
        )

    expected_records = recomputed_selection.records[: len(rows)]
    observed_ids = [str(row["record_id"]) for row in rows]
    if observed_ids != [str(item["record_id"]) for item in expected_records]:
        raise CapacityDiagnosticError(
            "terminal rows are not the scheduled selection prefix"
        )
    if not rows:
        raise CapacityDiagnosticError("terminal bundle has no frozen witness row")
    exact_state_files: list[Path] = []
    for ordinal, (row, expected_record) in enumerate(zip(rows, expected_records)):
        source = str(expected_record["source_sequence"])
        candidate = str(expected_record["candidate_sequence"])
        expected_identity = {
            "ordinal": ordinal,
            "record_id": str(expected_record["record_id"]),
            "dataset_id": str(expected_record["dataset_id"]),
            "region": str(expected_record["region"]),
            "canonical_jsonl_line": int(expected_record["_canonical_jsonl_line"]),
            "input_record_structural_sha256": record_structural_sha256(expected_record),
            "source_sequence_sha256": _sha256_bytes(source.encode("utf-8")),
            "candidate_sequence_sha256": _sha256_bytes(candidate.encode("utf-8")),
            "source_length": len(source),
            "candidate_length": len(candidate),
        }
        if any(row.get(field) != value for field, value in expected_identity.items()):
            raise CapacityDiagnosticError(
                "terminal row identity differs from live structural selection"
            )
        if row.get("alignment_statistics", {}).get("minimum_edit_count") != int(
            expected_record["edit_distance"]
        ):
            raise CapacityDiagnosticError(
                "terminal row edit distance differs from live structural selection"
            )
        workspace = _record_workspace_path(
            run_root,
            ordinal=ordinal,
            record_id=str(row["record_id"]),
        )
        expected_state_path = workspace / "state_universe.tsv"
        if row["outcome"] == "EXACT_COMPLETED":
            state_path = _verify_bundle_ref(
                run_root,
                row["state_universe_artifact"],
            )
            try:
                expected_resolved = expected_state_path.resolve(strict=True)
            except OSError as error:
                raise CapacityDiagnosticError(
                    "exact row lacks its deterministic state-universe file"
                ) from error
            if state_path != expected_resolved:
                raise CapacityDiagnosticError(
                    "exact row state-universe artifact is not at its "
                    "deterministic workspace path"
                )
            state_count, state_digest = _sequence_file_stats(state_path)
            exact = row["capacity"]["exact"]
            if (
                state_count != exact["reachable_node_count"]
                or state_digest != exact["reachable_states_sha256"]
                or row["state_universe_artifact"]["sha256"] != state_digest
            ):
                raise CapacityDiagnosticError(
                    "exact row counters or digest differ from its state universe"
                )
            replayed = _recompute_exact_record_evidence(
                record=expected_record,
                ordinal=ordinal,
                scratch_parent=run_root.parent,
                safety=resolved_config["diagnostic_safety_limits"],
                verification_started=verification_started,
            )
            if (
                row["alignment_statistics"] != replayed["alignment_statistics"]
                or exact != replayed["exact"]
                or row["frozen_gate_assessment"] != replayed["frozen_gate_assessment"]
                or row["state_universe_artifact"]["bytes"]
                != replayed["state_universe"]["bytes"]
                or row["state_universe_artifact"]["sha256"]
                != replayed["state_universe"]["sha256"]
            ):
                raise CapacityDiagnosticError(
                    "exact row differs from independent endpoint replay"
                )
            exact_state_files.append(state_path)
        else:
            if row["state_universe_artifact"] is not None:
                raise CapacityDiagnosticError(
                    "lower-bound row may not advertise a complete state universe"
                )
            if expected_state_path.exists():
                raise CapacityDiagnosticError(
                    "lower-bound row retains an unbound completed state universe"
                )
            stop = row["stop"]
            dimension = str(stop["dimension"])
            diagnostic_limit = resolved_config["diagnostic_safety_limits"][
                DIAGNOSTIC_LIMIT_FIELD_BY_DIMENSION[dimension]
            ]
            if (
                stop["frozen_limit"] != FROZEN_B0_LIMITS[dimension]
                or stop["observed_lower_bound"] <= stop["frozen_limit"]
                or stop["observed_lower_bound"] <= diagnostic_limit
                or row["frozen_gate_assessment"]
                != {
                    "would_pass_frozen_b0_limits": False,
                    "exceeded_limits": [dimension],
                }
            ):
                raise CapacityDiagnosticError(
                    "lower-bound row stop or frozen-gate assessment is invalid"
                )
    _validate_frozen_witness_row(rows[0])
    observed_complete_state_files = {
        path.resolve(strict=True)
        for path in run_root.glob("record_workspaces/*/state_universe.tsv")
        if path.is_file()
    }
    if observed_complete_state_files != set(exact_state_files):
        raise CapacityDiagnosticError(
            "completed record state universes are not exactly bound to exact rows"
        )
    observed_ids_sha256 = _record_ids_sha256(observed_ids)
    if (
        shard_index.get("row_count") != len(rows)
        or shard_index.get("record_ids_sha256") != observed_ids_sha256
        or shard_index.get("shards") != [_artifact_ref(records_path, root=run_root)]
    ):
        raise CapacityDiagnosticError("record shard index does not bind terminal rows")
    if (
        summary.get("source_record_count") != source_count
        or summary.get("split_eligible_record_count")
        != recomputed_selection.split_eligible_record_count
        or summary.get("scheduled_multi_edit_record_count") != scheduled
        or summary.get("exact_completed_record_count") != exact_count
        or summary.get("lower_bound_record_count") != lower_count
        or summary.get("diagnostic_scope") != selection_manifest.get("scope")
        or summary.get("state_universe_scope")
        != selection_manifest.get("state_universe_scope")
    ):
        raise CapacityDiagnosticError(
            "capacity summary counts or scope do not reconcile"
        )

    global_path = _verify_bundle_ref(
        run_root,
        summary["global_state_universe_artifact"],
    )
    endpoints_path = _bundle_path(run_root, "global/eligible_endpoints.tsv")
    observed_endpoints = tuple(_iter_sequence_file(endpoints_path))
    if observed_endpoints != recomputed_selection.eligible_endpoints:
        raise CapacityDiagnosticError(
            "eligible endpoints differ from live D1 structural recomputation"
        )
    endpoint_count = len(observed_endpoints)
    if (
        selection_manifest.get("eligible_endpoint_count") != endpoint_count
        or summary.get("global_unique_eligible_endpoint_count") != endpoint_count
    ):
        raise CapacityDiagnosticError(
            "eligible endpoint count differs from global-universe provenance"
        )

    global_exact = (
        manifest["state"] == "COMPLETED"
        and all_accounted
        and all(row["outcome"] == "EXACT_COMPLETED" for row in rows)
    )
    full_census_exact = selection_manifest.get("scope") == "census" and global_exact
    global_role = summary.get("global_state_universe_role")
    if global_exact:
        expected_global_role = "EXACT"
    elif global_role in {
        "LOWER_BOUND_FROM_ENDPOINTS_AND_EXACT_COMPLETED_RECORDS",
        "LOWER_BOUND_FROM_ELIGIBLE_ENDPOINTS_ONLY",
    }:
        expected_global_role = global_role
    else:
        raise CapacityDiagnosticError(
            "capacity summary global-universe role is invalid"
        )
    if (
        summary.get("global_state_universe_exact") is not global_exact
        or summary.get("full_census_state_universe_exact") is not full_census_exact
        or summary.get("global_state_universe_role") != expected_global_role
    ):
        raise CapacityDiagnosticError(
            "capacity summary exactness semantics are not recomputable"
        )
    if expected_global_role in {
        "EXACT",
        "LOWER_BOUND_FROM_ENDPOINTS_AND_EXACT_COMPLETED_RECORDS",
    }:
        expected_global_path = (run_root / "global/state_universe.tsv").resolve(
            strict=False
        )
        if global_path != expected_global_path:
            raise CapacityDiagnosticError(
                "merged global-universe role does not bind the canonical artifact"
            )
        (
            global_count,
            global_sha256,
            global_bytes,
            global_file_sha256,
        ) = _recompute_global_union(
            input_paths=[endpoints_path, *exact_state_files],
            safety=resolved_config["diagnostic_safety_limits"],
            scratch_parent=run_root.parent,
            verification_started=verification_started,
        )
        if global_sha256 != global_file_sha256:
            raise CapacityDiagnosticError(
                "recomputed global universe has noncanonical digest semantics"
            )
    else:
        if global_exact or manifest["state"] != "SAFE_PAUSED":
            raise CapacityDiagnosticError(
                "endpoint-only global role is valid only for a safe pause"
            )
        expected_failure_path = (
            run_root / "failure/global_merge_resource_pause.json"
        ).resolve(strict=False)
        failure = manifest.get("failure")
        if (
            not isinstance(failure, Mapping)
            or not isinstance(failure.get("evidence"), Mapping)
            or _verify_bundle_ref(run_root, failure["evidence"])
            != expected_failure_path
        ):
            raise CapacityDiagnosticError(
                "endpoint-only global role lacks the bound merge-pause evidence"
            )
        if global_path != endpoints_path:
            raise CapacityDiagnosticError(
                "endpoint-only global role does not bind eligible endpoints"
            )
        global_count, global_sha256 = _sequence_file_stats(endpoints_path)
        global_bytes = endpoints_path.stat().st_size
        global_file_sha256 = _sha256_file(endpoints_path)
    if (
        summary.get("global_unique_state_count") != global_count
        or summary.get("global_state_universe_sha256") != global_sha256
        or summary["global_state_universe_artifact"].get("bytes") != global_bytes
        or summary["global_state_universe_artifact"].get("sha256") != global_file_sha256
    ):
        raise CapacityDiagnosticError(
            "global universe is not the replayed exact union for its declared role"
        )
    expected_summary = _diagnostic_summary(
        run_root=run_root,
        diagnostic_scope=selection_manifest["scope"],
        selection=recomputed_selection,
        rows=rows,
        global_state_count=global_count,
        global_state_digest=global_sha256,
        global_state_exact=global_exact,
        global_state_role=expected_global_role,
        global_state_path=global_path,
    )
    if summary != expected_summary:
        raise CapacityDiagnosticError(
            "capacity summary differs from full independent recomputation"
        )
    result_semantics = manifest["result_semantics"]
    if (
        result_semantics.get("exact_capacity_complete") is not full_census_exact
        or result_semantics.get("lower_bounds_present") is not (lower_count > 0)
        or result_semantics.get("usable_for_budget_decision") is not full_census_exact
        or result_semantics.get("no_approximation_emitted") is not True
        or result_semantics.get("usable_for_b0_acceptance") is not False
    ):
        raise CapacityDiagnosticError(
            "result semantics disagree with recomputed census exactness"
        )

    checksum_path = _bundle_path(run_root, "artifact_checksums.json")
    checksum_index = _read_json(checksum_path)
    indexed = checksum_index.get("artifacts")
    if not isinstance(indexed, list) or checksum_index.get("artifact_count") != len(
        indexed
    ):
        raise CapacityDiagnosticError("artifact checksum index count is invalid")
    indexed_paths: set[str] = set()
    for reference in indexed:
        if not isinstance(reference, Mapping):
            raise CapacityDiagnosticError("artifact checksum entry is not an object")
        _verify_bundle_ref(run_root, reference)
        relative = str(reference["path"])
        if relative in indexed_paths:
            raise CapacityDiagnosticError("artifact checksum paths are duplicated")
        indexed_paths.add(relative)
    excluded_from_index = {
        "artifact_checksums.json",
        "bundle_seal.json",
        "terminal.lock",
        "VERIFIED",
    }
    expected_indexed_paths = {
        path.relative_to(run_root).as_posix()
        for path in run_root.rglob("*")
        if path.is_file()
        and path.relative_to(run_root).as_posix() not in excluded_from_index
    }
    if indexed_paths != expected_indexed_paths:
        raise CapacityDiagnosticError(
            "artifact checksum index does not cover the complete pre-seal bundle"
        )

    marker_name = "DONE" if manifest["state"] == "COMPLETED" else "SAFE_PAUSED"
    marker_ref = _artifact_ref(run_root / marker_name, root=run_root)
    if manifest["state"] == "COMPLETED":
        completion = manifest["completion_seal"]
        if (
            completion.get("terminal_marker") != marker_name
            or completion.get("terminal_marker_ref") != marker_ref
            or completion.get("artifact_checksum_index_path")
            != "artifact_checksums.json"
            or completion.get("bundle_seal_path") != "bundle_seal.json"
            or completion.get("terminal_lock_path") != "terminal.lock"
            or completion.get("record_shard_index")
            != accounting.get("record_shard_index")
        ):
            raise CapacityDiagnosticError(
                "manifest completion seal declaration is inconsistent"
            )

    bundle_seal_path = run_root / "bundle_seal.json"
    bundle_seal = _read_json(bundle_seal_path)
    expected_seal_refs = {
        "diagnostic_manifest": _artifact_ref(manifest_path, root=run_root),
        "artifact_checksum_index": _artifact_ref(
            checksum_path,
            root=run_root,
        ),
        "terminal_marker": marker_ref,
        "status": _artifact_ref(run_root / "status.json", root=run_root),
        "process_result": _artifact_ref(
            run_root / "provenance/process_result.json",
            root=run_root,
        ),
    }
    if (
        bundle_seal.get("schema_version") != "b0_capacity_bundle_seal.v1"
        or bundle_seal.get("diagnostic_id") != manifest["diagnostic_id"]
        or bundle_seal.get("state") != manifest["state"]
        or any(
            bundle_seal.get(key) != value for key, value in expected_seal_refs.items()
        )
    ):
        raise CapacityDiagnosticError("detached bundle seal is invalid")

    terminal_lock = _read_json(run_root / "terminal.lock")
    if terminal_lock != {
        "schema_version": "b0_capacity_terminal_lock.v1",
        "diagnostic_id": manifest["diagnostic_id"],
        "state": manifest["state"],
        "sealed": True,
        "bundle_seal": _artifact_ref(bundle_seal_path, root=run_root),
    }:
        raise CapacityDiagnosticError("terminal lock does not bind the detached seal")

    process_result = _read_json(run_root / "provenance/process_result.json")
    expected_return_code = 0 if manifest["state"] == "COMPLETED" else 3
    if (
        process_result.get("schema_version") != "b0_capacity_process_result.v1"
        or process_result.get("terminal_state") != manifest["state"]
        or process_result.get("return_code") != expected_return_code
        or process_result.get("stdout")
        != _artifact_ref(run_root / "logs/stdout.log", root=run_root)
        or process_result.get("stderr")
        != _artifact_ref(run_root / "logs/stderr.log", root=run_root)
        or process_result.get("capture_mode") != "OS_FD_DUP2"
        or process_result.get("capture_complete") is not True
        or process_result.get("capture_scope") != STREAM_CAPTURE_SCOPE
        or process_result.get("return_code_semantics")
        != "COMMITTED_AFTER_BUNDLE_VALIDATION_AND_VERIFIED_MARKER"
    ):
        raise CapacityDiagnosticError(
            "process result does not bind captured streams and terminal return code"
        )


def _verified_marker_payload(
    *,
    run_root: Path,
    state: str,
    verified_at_utc: str,
    postseal_system_metric: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "b0_capacity_verified_marker.v1",
        "diagnostic_id": run_root.name,
        "state": state,
        "verified_at_utc": verified_at_utc,
        "bundle_verifier": "PASSED",
        "committed_return_code": 0 if state == "COMPLETED" else 3,
        "return_code_semantics": (
            "NO_FALLIBLE_ACTIONS_AFTER_VERIFIED_MARKER_VALIDATION"
        ),
        "resource_limit_scope": RESOURCE_LIMIT_SCOPE,
        "postseal_system_metric": dict(postseal_system_metric),
        "diagnostic_manifest": _artifact_ref(
            run_root / "diagnostic_manifest.json",
            root=run_root,
        ),
        "bundle_seal": _artifact_ref(
            run_root / "bundle_seal.json",
            root=run_root,
        ),
        "terminal_lock": _artifact_ref(
            run_root / "terminal.lock",
            root=run_root,
        ),
        "process_result": _artifact_ref(
            run_root / "provenance/process_result.json",
            root=run_root,
        ),
    }


def _validate_verified_marker(run_root: Path) -> None:
    marker = _read_json(run_root / "VERIFIED")
    manifest = _read_json(run_root / "diagnostic_manifest.json")
    metric = marker.get("postseal_system_metric")
    if not isinstance(metric, Mapping) or any(
        name not in metric
        for name in (
            "captured_at_utc",
            "elapsed_seconds",
            "free_bytes",
            "peak_rss_bytes",
            "run_bytes",
            "total_bytes",
            "used_bytes",
        )
    ):
        raise CapacityDiagnosticError(
            "VERIFIED marker lacks the post-seal system metric"
        )
    expected = _verified_marker_payload(
        run_root=run_root,
        state=str(manifest["state"]),
        verified_at_utc=str(marker.get("verified_at_utc")),
        postseal_system_metric=metric,
    )
    if marker != expected:
        raise CapacityDiagnosticError(
            "VERIFIED marker does not bind the validated terminal bundle"
        )


def _validate_parent_witness(
    *,
    parent_diagnostic_id: str,
    project_root: Path,
    expected_code_commit: str,
    d1_snapshot: Path,
    candidate_store: Path,
    resolved_config: Mapping[str, Any],
) -> tuple[dict[str, Any], StructuralSelection]:
    parent_root = ALLOWED_OUTPUT_PARENT / parent_diagnostic_id
    if not parent_root.is_dir():
        raise CapacityDiagnosticError("census parent witness diagnostic does not exist")
    schema_path = project_root / "schemas/b0_capacity_diagnostic.schema.json"
    _validate_bundle(run_root=parent_root, schema_path=schema_path)
    _validate_verified_marker(parent_root)
    manifest = _read_json(parent_root / "diagnostic_manifest.json")
    selection_manifest = _read_json(parent_root / "provenance/selection_manifest.json")
    summary = _read_json(parent_root / "capacity_summary.json")
    rows = _read_jsonl_objects(parent_root / "records/results.jsonl")
    current_selection = _load_structural_selection(
        candidate_store=candidate_store,
        d1_snapshot=d1_snapshot,
        minimum_edit_distance=resolved_config["selection"]["minimum_edit_distance"],
        witness_record_id=WITNESS_RECORD_ID,
    )
    witness_record = current_selection.records[0]
    structural_ref = _artifact_ref(candidate_store)
    parent_structural = manifest.get("provenance", {}).get("structural_store", {})
    algorithm_contract = manifest.get("algorithm_contract", {})
    witness_row_valid = (
        len(rows) == 1
        and rows[0].get("ordinal") == 0
        and rows[0].get("record_id") == WITNESS_RECORD_ID
        and rows[0].get("canonical_jsonl_line") == WITNESS_CANONICAL_JSONL_LINE
        and rows[0].get("source_length") == len(WITNESS_SOURCE)
        and rows[0].get("candidate_length") == len(WITNESS_CANDIDATE)
        and rows[0].get("input_record_structural_sha256")
        == record_structural_sha256(witness_record)
        and rows[0].get("source_sequence_sha256")
        == _sha256_bytes(WITNESS_SOURCE.encode("utf-8"))
        and rows[0].get("candidate_sequence_sha256")
        == _sha256_bytes(WITNESS_CANDIDATE.encode("utf-8"))
        and rows[0].get("outcome") == "EXACT_COMPLETED"
        and rows[0].get("capacity", {}).get("exact") == WITNESS_EXPECTED
        and rows[0].get("capacity", {}).get("lower_bound") is None
        and rows[0].get("alignment_statistics", {}).get("counts_exact") is True
        and rows[0].get("evidence_semantics", {}).get("counts_exact") is True
        and rows[0].get("evidence_semantics", {}).get("state_set_complete") is True
    )
    if (
        manifest.get("state") != "COMPLETED"
        or manifest.get("parent_diagnostic_id") is not None
        or manifest.get("diagnostic_id") != parent_diagnostic_id
        or manifest.get("selection", {}).get("diagnostic_scope") != "witness"
        or manifest.get("selection", {}).get("selected_record_count") != 1
        or manifest.get("accounting", {}).get("scheduled_record_count") != 1
        or manifest.get("accounting", {}).get("terminal_record_count") != 1
        or manifest.get("accounting", {}).get("exact_completed_count") != 1
        or selection_manifest.get("scope") != "witness"
        or selection_manifest.get("selected_record_count") != 1
        or len(selection_manifest.get("selected_records", [])) != 1
        or selection_manifest.get("selected_records", [{}])[0].get("record_id")
        != WITNESS_RECORD_ID
        or summary.get("state_universe_scope") != "FROZEN_WITNESS_SUBSET"
        or summary.get("global_state_universe_exact") is not True
        or summary.get("global_unique_state_count")
        != WITNESS_EXPECTED["reachable_node_count"]
        or summary.get("global_state_universe_sha256")
        != WITNESS_EXPECTED["reachable_states_sha256"]
        or summary.get("full_census_state_universe_exact") is not False
        or summary.get("diagnostic_scope") != "witness"
        or manifest.get("result_semantics", {}).get("exact_capacity_complete")
        is not False
        or manifest.get("result_semantics", {}).get("usable_for_budget_decision")
        is not False
        or manifest.get("provenance", {}).get("git", {}).get("head")
        != expected_code_commit
        or manifest.get("provenance", {}).get("d1_snapshot", {}).get("sha256")
        != _sha256_file(d1_snapshot)
        or _read_json(parent_root / "resolved_config.json") != dict(resolved_config)
        or algorithm_contract.get("frozen_b0_limits") != FROZEN_B0_LIMITS
        or algorithm_contract.get("diagnostic_safety_limits")
        != resolved_config["diagnostic_safety_limits"]
        or any(
            parent_structural.get(field) != structural_ref[field]
            for field in ("path", "bytes", "sha256")
        )
        or not witness_row_valid
    ):
        raise CapacityDiagnosticError(
            "census parent is not the exact completed witness for this code/data"
        )
    return (
        {
            "diagnostic_id": parent_diagnostic_id,
            "diagnostic_manifest": _artifact_ref(
                parent_root / "diagnostic_manifest.json"
            ),
            "verified_marker": _artifact_ref(parent_root / "VERIFIED"),
            "bundle_seal": _artifact_ref(parent_root / "bundle_seal.json"),
            "terminal_lock": _artifact_ref(parent_root / "terminal.lock"),
            "process_result": _artifact_ref(
                parent_root / "provenance/process_result.json"
            ),
        },
        current_selection,
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
    try:
        allowed_parent = ALLOWED_OUTPUT_PARENT.resolve(strict=True)
        output_parent = args.output_root.parent.resolve(strict=True)
    except OSError as error:
        raise CapacityDiagnosticError(
            "diagnostic output parent must already exist"
        ) from error
    if output_parent != allowed_parent:
        raise CapacityDiagnosticError(
            "diagnostic output root must be a direct child of "
            f"{ALLOWED_OUTPUT_PARENT}"
        )
    resolved_output = args.output_root.resolve(strict=False)
    if resolved_output.is_relative_to(FORMAL_ATTEMPT_ROOT.resolve(strict=False)):
        raise CapacityDiagnosticError(
            "diagnostic output may not be inside the formal D1/B0 attempt tree"
        )
    if not args.output_root.name.startswith("B0_capacity_"):
        raise CapacityDiagnosticError(
            "output root name must be a B0_capacity diagnostic ID"
        )
    if (
        re.fullmatch(
            r"B0_capacity_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{7,40}",
            args.output_root.name,
        )
        is None
    ):
        raise CapacityDiagnosticError(
            "output root diagnostic ID has an invalid timestamp/hash shape"
        )
    if args.scope == "witness" and args.parent_diagnostic_id is not None:
        raise CapacityDiagnosticError(
            "the first witness diagnostic may not declare a parent"
        )
    if args.scope == "census" and not args.parent_diagnostic_id:
        raise CapacityDiagnosticError(
            "the census requires the exact completed witness diagnostic ID"
        )
    if args.parent_diagnostic_id is not None and (
        re.fullmatch(
            r"B0_capacity_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{7,40}",
            args.parent_diagnostic_id,
        )
        is None
        or args.parent_diagnostic_id == args.output_root.name
    ):
        raise CapacityDiagnosticError("parent diagnostic ID is invalid")
    entrypoint = Path(sys.argv[0])
    expected_entrypoint = (
        args.project_root / "scripts/data/diagnose_b0_path_capacity.py"
    ).resolve(strict=True)
    if not entrypoint.is_absolute() or entrypoint.resolve(strict=True) != (
        expected_entrypoint
    ):
        raise CapacityDiagnosticError(
            "diagnostic must be launched through its absolute repository entrypoint"
        )
    if Path.cwd().resolve(strict=True) != args.project_root.resolve(strict=True):
        raise CapacityDiagnosticError(
            "diagnostic launch cwd must equal the authoritative project root"
        )


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    global _ACTIVE_FD_CAPTURE
    _validate_cli_paths(args)
    config = _validate_config(_read_json(args.config))
    parent_authorization: dict[str, Any] | None = None
    prevalidated_selection: StructuralSelection | None = None
    if args.scope == "census":
        parent_authorization, prevalidated_selection = _validate_parent_witness(
            parent_diagnostic_id=args.parent_diagnostic_id,
            project_root=args.project_root,
            expected_code_commit=args.expected_code_commit,
            d1_snapshot=args.d1_snapshot,
            candidate_store=args.candidate_store,
            resolved_config=config,
        )
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    _create_exclusive_run_root(args.output_root)
    run_root = args.output_root
    for relative in ("logs", "provenance", "records"):
        (run_root / relative).mkdir()
    events_path = run_root / "logs/events.jsonl"
    metrics_path = run_root / "logs/system_metrics.jsonl"
    stdout_path = run_root / "logs/stdout.log"
    stderr_path = run_root / "logs/stderr.log"
    _write_text_exclusive(stdout_path, "")
    _write_text_exclusive(stderr_path, "")
    capture = FileDescriptorCapture(
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    capture.start()
    _ACTIVE_FD_CAPTURE = capture
    _append_jsonl(
        events_path,
        {
            "at_utc": started_at,
            "event": "DIAGNOSTIC_STARTED",
            "formal_b0_attempt_started": False,
        },
    )

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
    d1_validation = _validate_d1_snapshot_trust(
        snapshot_path=args.d1_snapshot,
        acceptance_path=args.d1_acceptance,
        project_root=args.project_root,
        git_dir=args.git_dir,
    )
    d1_validation_path = run_root / "provenance/d1_snapshot_validation.json"
    _write_json_exclusive(d1_validation_path, d1_validation)
    build_manifest_path = Path(str(snapshot["build_manifest"]["path"]))
    build_manifest_ref = _artifact_ref(build_manifest_path)
    if (
        build_manifest_ref["bytes"] != snapshot["build_manifest"]["bytes"]
        or build_manifest_ref["sha256"] != snapshot["build_manifest"]["sha256"]
    ):
        raise CapacityDiagnosticError(
            "D1 build manifest differs from the frozen snapshot binding"
        )
    ambiguity_path, ambiguity_ref = _find_required_artifact(
        snapshot,
        "edit_script_ambiguity_report.json",
    )

    selection = prevalidated_selection or _load_structural_selection(
        candidate_store=args.candidate_store,
        d1_snapshot=args.d1_snapshot,
        minimum_edit_distance=config["selection"]["minimum_edit_distance"],
        witness_record_id=WITNESS_RECORD_ID,
    )
    selection = _apply_diagnostic_scope(selection, scope=args.scope)
    selection_manifest = _selection_manifest_payload(
        selection,
        scope=args.scope,
    )
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
            "d1_snapshot_validation": _artifact_ref(
                d1_validation_path,
                root=run_root,
            ),
            "d1_acceptance": acceptance_ref,
            "d1_build_manifest": build_manifest_ref,
            "ambiguity_report": ambiguity_ref,
            "structural_store": structural_meta,
            "canonical_store_metadata_only": {
                **canonical_meta,
                "selection_opened": False,
                "capacity_algorithm_opened": False,
                "opaque_integrity_hash_opened": True,
                "jsonl_parsed": False,
                "label_values_accessed": False,
            },
            "label_fields_read": [],
        },
    )
    launch_cwd = Path.cwd().resolve(strict=True)
    argv_exact = [str(Path(sys.executable).resolve(strict=True)), *sys.argv]
    command_path = run_root / "command.json"
    _write_json_exclusive(
        command_path,
        {
            "exact_argv": argv_exact,
            "exact_argv_sha256": _canonical_sha256(argv_exact),
            "launch_cwd": str(launch_cwd),
            "output_root_replay_policy": (
                "replace_with_a_fresh_nonexistent_output_root"
            ),
        },
    )
    _write_replay_script(
        run_root=run_root,
        argv=argv_exact,
        launch_cwd=launch_cwd,
    )

    rows: list[dict[str, Any]] = []
    record_state_files: list[Path] = []
    state = "COMPLETED"
    stop_reason: str | None = None
    failure: dict[str, Any] | None = None
    safety = config["diagnostic_safety_limits"]
    watchdog = ResourceWatchdog(
        run_root=run_root,
        metrics_path=metrics_path,
        started_monotonic=started_monotonic,
        safety=safety,
        last_logged_monotonic=started_monotonic,
    )
    watchdog.check({"phase": "pre_record_census"}, force_log=True)
    for ordinal, record in enumerate(selection.records):
        try:
            watchdog.check(
                {
                    "phase": "record_boundary",
                    "ordinal": ordinal,
                    "record_id": record["record_id"],
                }
            )
            _append_jsonl(
                events_path,
                {
                    "at_utc": _utc_now(),
                    "event": "RECORD_STARTED",
                    "ordinal": ordinal,
                    "record_id": record["record_id"],
                },
            )
            row, state_file = _run_record(
                record=record,
                ordinal=ordinal,
                run_root=run_root,
                safety=safety,
                watchdog=watchdog,
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

    global_merge_completed = False
    try:
        global_path, global_count, global_digest = _merge_global_universe(
            run_root=run_root,
            endpoints=selection.eligible_endpoints,
            record_state_files=record_state_files,
            max_open_chunks=safety["max_open_chunks"],
            progress_callback=watchdog.check,
        )
        global_merge_completed = True
        watchdog.check(
            {
                "phase": "post_global_merge",
                "global_unique_state_count": global_count,
            },
            force_log=True,
        )
    except SafeCapacityPause as error:
        state = "SAFE_PAUSED"
        stop_reason = str(error)
        failure_path = run_root / "failure/global_merge_resource_pause.json"
        _write_json_exclusive(
            failure_path,
            {
                "failed_at_utc": _utc_now(),
                "failure_class": "SAFE_RESOURCE_PAUSE",
                "reason": stop_reason,
                "partial_outputs_preserved": True,
                "resume_policy": "fresh_segment_only_after_hash_revalidation",
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
        if not global_merge_completed:
            global_path = run_root / "global/eligible_endpoints.tsv"
            global_count, global_digest = _sequence_file_stats(global_path)
    if state == "COMPLETED":
        try:
            watchdog.check(
                {
                    "phase": "forced_preterminal_resource_gate",
                    "terminal_record_count": len(rows),
                    "global_unique_state_count": global_count,
                },
                force_log=True,
            )
        except SafeCapacityPause as error:
            state = "SAFE_PAUSED"
            stop_reason = str(error)
            failure_path = run_root / "failure/preterminal_resource_pause.json"
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
    global_exact = (
        state == "COMPLETED"
        and len(rows) == len(selection.records)
        and all(row["outcome"] == "EXACT_COMPLETED" for row in rows)
    )
    global_state_role = (
        "EXACT"
        if global_exact
        else (
            "LOWER_BOUND_FROM_ENDPOINTS_AND_EXACT_COMPLETED_RECORDS"
            if global_merge_completed
            else "LOWER_BOUND_FROM_ELIGIBLE_ENDPOINTS_ONLY"
        )
    )
    summary = _diagnostic_summary(
        run_root=run_root,
        diagnostic_scope=args.scope,
        selection=selection,
        rows=rows,
        global_state_count=global_count,
        global_state_digest=global_digest,
        global_state_exact=global_exact,
        global_state_role=global_state_role,
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
    terminal_lock = run_root / "terminal.lock"
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
    _stop_active_fd_capture()
    process_result_path = run_root / "provenance/process_result.json"
    _write_json_exclusive(
        process_result_path,
        {
            "schema_version": "b0_capacity_process_result.v1",
            "terminal_state": state,
            "return_code": 0 if state == "COMPLETED" else 3,
            "stdout": _artifact_ref(stdout_path, root=run_root),
            "stderr": _artifact_ref(stderr_path, root=run_root),
            "capture_mode": "OS_FD_DUP2",
            "capture_complete": True,
            "capture_scope": STREAM_CAPTURE_SCOPE,
            "return_code_semantics": (
                "COMMITTED_AFTER_BUNDLE_VALIDATION_AND_VERIFIED_MARKER"
            ),
        },
    )

    completed_at = _utc_now()
    manifest = {
        "artifact_type": "b0_capacity_diagnostic",
        "schema_version": "b0_capacity_diagnostic.v1",
        "diagnostic_id": run_root.name,
        "parent_diagnostic_id": args.parent_diagnostic_id,
        "parent_authorization": parent_authorization,
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
            "diagnostic_scope": args.scope,
            "state_universe_scope": (
                "FULL_CENSUS" if args.scope == "census" else "FROZEN_WITNESS_SUBSET"
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
            "canonical_label_store_opened_by_selection": False,
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
            "launch_cwd": str(launch_cwd),
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
            "d1_snapshot": snapshot_ref,
            "d1_snapshot_validation": _artifact_ref(
                d1_validation_path,
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
                "selection_opened": False,
                "capacity_algorithm_opened": False,
                "opaque_integrity_hash_opened": True,
                "jsonl_parsed": False,
                "label_values_accessed": False,
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
            "process_result": _artifact_ref(
                process_result_path,
                root=run_root,
            ),
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
            "exact_capacity_complete": (args.scope == "census" and global_exact),
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
                "terminal_marker_ref": {
                    "path": "DONE",
                    "bytes": 0,
                    "sha256": hashlib.sha256(b"").hexdigest(),
                },
                "artifact_checksum_index_path": "artifact_checksums.json",
                "bundle_seal_path": "bundle_seal.json",
                "terminal_lock_path": "terminal.lock",
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
    _write_text_exclusive(marker_path, "")
    checksums_path = run_root / "artifact_checksums.json"
    preseal_paths = [path for path in run_root.rglob("*") if path.is_file()]
    _write_json_exclusive(
        checksums_path,
        _checksum_index(run_root, preseal_paths),
    )
    bundle_seal_path = run_root / "bundle_seal.json"
    _write_json_exclusive(
        bundle_seal_path,
        {
            "schema_version": "b0_capacity_bundle_seal.v1",
            "diagnostic_id": run_root.name,
            "state": state,
            "sealed_at_utc": completed_at,
            "diagnostic_manifest": _artifact_ref(
                manifest_path,
                root=run_root,
            ),
            "artifact_checksum_index": _artifact_ref(
                checksums_path,
                root=run_root,
            ),
            "terminal_marker": _artifact_ref(marker_path, root=run_root),
            "status": _artifact_ref(status_path, root=run_root),
            "process_result": _artifact_ref(
                process_result_path,
                root=run_root,
            ),
        },
    )
    _write_json_exclusive(
        terminal_lock,
        {
            "schema_version": "b0_capacity_terminal_lock.v1",
            "diagnostic_id": run_root.name,
            "state": state,
            "sealed": True,
            "bundle_seal": _artifact_ref(
                bundle_seal_path,
                root=run_root,
            ),
        },
    )
    _validate_bundle(run_root=run_root, schema_path=schema_path)
    result = {
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
        "return_code": 0 if state == "COMPLETED" else 3,
    }
    verified_path = run_root / "VERIFIED"
    _write_json_exclusive(
        verified_path,
        _verified_marker_payload(
            run_root=run_root,
            state=state,
            verified_at_utc=_utc_now(),
            postseal_system_metric=_system_metric(
                run_root,
                started_monotonic,
            ),
        ),
    )
    _validate_verified_marker(run_root)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = run(args)
    except Exception as error:
        _stop_active_fd_capture()
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
    _stop_active_fd_capture()
    return int(result["return_code"])


if __name__ == "__main__":
    raise SystemExit(main())
