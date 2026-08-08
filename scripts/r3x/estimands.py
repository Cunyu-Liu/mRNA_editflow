"""Fail-closed helpers for independent primary-estimand training.

The R3X targets may have unrelated physical scales.  This module therefore
partitions the frozen ordinary dataset before any optimizer is constructed and
binds every model/prediction shard to one frozen primary-estimand registry
entry.  It performs no I/O and does not inspect protected data.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from .protocol import canonical_sha256
from .report import ReportProtocolError


PARTITIONS = ("train", "development", "internal_test")
PREDICTION_SHARD_SCHEMA = "r3x_primary_estimand_prediction_shard_v1"
MERGED_PREDICTION_SCHEMA = "r3x_merged_primary_estimand_predictions_v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SEMANTIC_FIELDS = (
    "study",
    "endpoint",
    "raw_source_field",
    "raw_candidate_field",
    "raw_unit",
    "direction",
    "transform",
    "valid_domain",
    "role",
    "replicate_policy",
    "source_citation",
    "source_evidence_sha256",
)
_SHARED_ESTIMAND_FIELDS = (
    "raw_unit", "direction", "transform", "valid_domain", "role",
    "replicate_policy",
)


def _fail(message: str) -> None:
    raise ReportProtocolError(message)


def _require_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{label} must be a non-empty string")
    return value.strip()


def _require_sha256(value: object, *, label: str) -> str:
    text = _require_text(value, label=label)
    if not _HEX64.fullmatch(text):
        _fail(f"{label} must be 64 lowercase hexadecimal characters")
    return text


def stable_row_id(row: Mapping) -> str:
    """Return the frozen row identity used for prediction alignment."""

    if not isinstance(row, Mapping):
        _fail("estimand rows must be mappings")
    return _require_text(
        row.get("r3_input_row_sha256") or row.get("record_id"),
        label="estimand row identity",
    )


def row_order_sha256(row_ids: Sequence[str]) -> str:
    """Hash an already ordered, unique row-ID sequence.

    This intentionally matches the existing R3X prediction-row hash encoding:
    a compact JSON array, with order included in the digest.
    """

    ids = tuple(_require_text(value, label="prediction row identity") for value in row_ids)
    if len(ids) != len(set(ids)):
        _fail("prediction row identities must be unique")
    return hashlib.sha256(
        json.dumps(ids, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def canonical_row_ids(rows: Sequence[Mapping]) -> tuple[str, ...]:
    """Return the canonical, input-order-invariant row order."""

    ids = tuple(sorted(stable_row_id(row) for row in rows))
    if len(ids) != len(set(ids)):
        _fail("primary-estimand rows contain duplicate identities")
    return ids


def _canonical_rows(rows: Sequence[Mapping]) -> tuple[dict, ...]:
    indexed = {stable_row_id(row): dict(row) for row in rows}
    if len(indexed) != len(rows):
        _fail("primary-estimand rows contain duplicate identities")
    return tuple(indexed[row_id] for row_id in sorted(indexed))


@dataclass(frozen=True)
class PrimaryEstimandSlice:
    """One independently trainable primary estimand and its three partitions."""

    estimand_id: str
    studies: tuple[str, ...]
    endpoints: tuple[str, ...]
    registry_sha256: str
    registry_entry_set_sha256: str
    train: tuple[dict, ...]
    development: tuple[dict, ...]
    internal_test: tuple[dict, ...]

    def rows_for(self, partition: str) -> tuple[dict, ...]:
        if partition not in PARTITIONS:
            _fail(f"unknown primary-estimand partition: {partition}")
        return getattr(self, partition)

    def row_ids_for(self, partition: str) -> tuple[str, ...]:
        return tuple(stable_row_id(row) for row in self.rows_for(partition))

    def row_order_sha256_for(self, partition: str) -> str:
        return row_order_sha256(self.row_ids_for(partition))

    @property
    def partition_row_order_sha256(self) -> dict[str, str]:
        return {
            partition: self.row_order_sha256_for(partition)
            for partition in PARTITIONS
        }

    @property
    def partition_binding_sha256(self) -> str:
        return canonical_sha256({
            "schema_version": "r3x_primary_estimand_partition_binding_v1",
            "estimand_id": self.estimand_id,
            "registry_sha256": self.registry_sha256,
            "registry_entry_set_sha256": self.registry_entry_set_sha256,
            "ordered_row_ids": {
                partition: list(self.row_ids_for(partition))
                for partition in PARTITIONS
            },
        })


def _validate_embedded_binding(
    row: Mapping,
    entry: Mapping,
    *,
    registry_sha256: str,
) -> None:
    status = row.get("estimand_status")
    if status is not None and status != "FROZEN_REGISTRY_APPLIED":
        _fail("row estimand status is not FROZEN_REGISTRY_APPLIED")
    binding = row.get("estimand_binding")
    if binding is None:
        return
    if not isinstance(binding, Mapping):
        _fail("embedded estimand_binding must be an object")
    embedded_sha = binding.get("registry_sha256")
    if embedded_sha is not None and embedded_sha != registry_sha256:
        _fail("row estimand_binding registry SHA-256 disagrees with frozen authority")
    for field in _SEMANTIC_FIELDS:
        if field in entry and binding.get(field) != entry.get(field):
            _fail(f"row estimand_binding disagrees with registry field {field}")


def build_primary_estimand_slices(
    rows: Sequence[Mapping],
    estimand_registry: Mapping,
    *,
    estimand_registry_sha256: str,
    require_all_primary_registry_entries: bool = True,
) -> tuple[PrimaryEstimandSlice, ...]:
    """Validate and partition rows into independent primary-estimand slices.

    Every represented primary estimand must have non-empty train, development,
    and internal-test partitions.  By default every primary registry entry must
    also be represented, preventing a silent endpoint drop from looking like a
    complete run.
    """

    registry_sha = _require_sha256(
        estimand_registry_sha256, label="endpoint estimand registry SHA-256"
    )
    if not isinstance(estimand_registry, Mapping):
        _fail("endpoint estimand registry must be an object")
    if estimand_registry.get("status") != "FROZEN":
        _fail("endpoint estimand registry must declare status=FROZEN")
    entries = estimand_registry.get("entries")
    if not isinstance(entries, list) or not entries:
        _fail("endpoint estimand registry lacks entries")

    by_key: dict[tuple[str, str], Mapping] = {}
    by_id: dict[str, list[Mapping]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            _fail("endpoint estimand registry entries must be objects")
        if entry.get("role") != "PRIMARY":
            continue
        study = _require_text(entry.get("study"), label="registry study")
        endpoint = _require_text(entry.get("endpoint"), label="registry endpoint")
        estimand_id = _require_text(entry.get("estimand_id"), label="registry estimand_id")
        key = (study, endpoint)
        if key in by_key:
            _fail("primary estimand registry study/endpoint keys must be unique")
        by_key[key] = entry
        by_id.setdefault(estimand_id, []).append(entry)
    if not by_id:
        _fail("endpoint estimand registry contains no primary estimands")

    grouped: dict[str, dict[str, list[Mapping]]] = {
        estimand_id: {partition: [] for partition in PARTITIONS}
        for estimand_id in by_id
    }
    seen_row_ids: set[str] = set()
    represented: set[str] = set()
    represented_keys: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            _fail("primary-estimand rows must be objects")
        row_id = stable_row_id(row)
        if row_id in seen_row_ids:
            _fail("primary-estimand rows contain duplicate identities")
        seen_row_ids.add(row_id)
        study = _require_text(row.get("study"), label="row study")
        endpoint_values = {
            _require_text(row[field], label=f"row {field}")
            for field in ("endpoint", "endpoint_id")
            if row.get(field) is not None
        }
        if len(endpoint_values) != 1:
            _fail("row endpoint and endpoint_id aliases must resolve to one value")
        endpoint = endpoint_values.pop()
        estimand_id = _require_text(row.get("estimand_id"), label="row estimand_id")
        partition = _require_text(row.get("partition"), label="row partition")
        if partition not in PARTITIONS:
            _fail(f"row partition is outside the frozen three-way split: {partition}")
        entry = by_key.get((study, endpoint))
        if entry is None or entry.get("estimand_id") != estimand_id:
            _fail("row is not semantically bound to one frozen primary estimand")
        if entry not in by_id.get(estimand_id, []):
            _fail("row estimand binding is ambiguous")
        _validate_embedded_binding(row, entry, registry_sha256=registry_sha)
        grouped[estimand_id][partition].append(row)
        represented.add(estimand_id)
        represented_keys.add((study, endpoint))

    if not rows:
        _fail("primary-estimand training dataset is empty")
    if require_all_primary_registry_entries and represented_keys != set(by_key):
        missing = sorted(set(by_key) - represented_keys)
        _fail(
            "primary registry study/endpoint entries are absent from the ready dataset: "
            + ", ".join(f"{study}/{endpoint}" for study, endpoint in missing)
        )

    slices = []
    for estimand_id in sorted(represented):
        missing = [partition for partition in PARTITIONS if not grouped[estimand_id][partition]]
        if missing:
            _fail(
                f"primary estimand {estimand_id} has empty frozen partitions: "
                + ", ".join(missing)
            )
        entry_set = sorted(
            by_id[estimand_id],
            key=lambda item: (str(item["study"]), str(item["endpoint"])),
        )
        signatures = {
            tuple((field, entry.get(field)) for field in _SHARED_ESTIMAND_FIELDS)
            for entry in entry_set
        }
        if len(signatures) != 1:
            _fail(
                f"primary estimand {estimand_id} has inconsistent scientific semantics across registry entries"
            )
        slices.append(PrimaryEstimandSlice(
            estimand_id=estimand_id,
            studies=tuple(sorted({str(entry["study"]) for entry in entry_set})),
            endpoints=tuple(sorted({str(entry["endpoint"]) for entry in entry_set})),
            registry_sha256=registry_sha,
            registry_entry_set_sha256=canonical_sha256(entry_set),
            train=_canonical_rows(grouped[estimand_id]["train"]),
            development=_canonical_rows(grouped[estimand_id]["development"]),
            internal_test=_canonical_rows(grouped[estimand_id]["internal_test"]),
        ))
    return tuple(slices)


def _finite_vector(value: object, *, label: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        _fail(f"{label} must align exactly with the estimand row order")
    output = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            _fail(f"{label} must contain only finite numbers")
        output.append(float(item))
    return tuple(output)


def validate_prediction_shard(
    estimand_slice: PrimaryEstimandSlice,
    shard: Mapping,
    *,
    partition: str = "internal_test",
) -> dict:
    """Validate one immutable prediction shard and return a normalized copy."""

    if not isinstance(shard, Mapping) or shard.get("schema_version") != PREDICTION_SHARD_SCHEMA:
        _fail("prediction shard has an unsupported schema")
    if shard.get("estimand_id") != estimand_slice.estimand_id:
        _fail("prediction shard estimand_id disagrees with its training slice")
    if shard.get("partition") != partition:
        _fail("prediction shard partition disagrees with requested merge partition")
    if shard.get("registry_sha256") != estimand_slice.registry_sha256:
        _fail("prediction shard registry SHA-256 disagrees with its training slice")
    if shard.get("registry_entry_set_sha256") != estimand_slice.registry_entry_set_sha256:
        _fail("prediction shard registry-entry-set SHA-256 disagrees with its training slice")
    seed = shard.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        _fail("prediction shard seed must be a non-negative integer")
    expected_ids = estimand_slice.row_ids_for(partition)
    row_ids = shard.get("row_ids")
    if not isinstance(row_ids, (list, tuple)) or tuple(row_ids) != expected_ids:
        _fail("prediction shard row order/identity disagrees with canonical estimand order")
    expected_hash = estimand_slice.row_order_sha256_for(partition)
    if shard.get("row_order_sha256") != expected_hash:
        _fail("prediction shard row-order SHA-256 disagrees with canonical estimand order")
    mean = _finite_vector(shard.get("mean"), label="prediction mean", length=len(expected_ids))
    logvar_value = shard.get("logvar")
    logvar = None if logvar_value is None else _finite_vector(
        logvar_value, label="prediction logvar", length=len(expected_ids)
    )
    architecture_id = _require_text(shard.get("architecture_id"), label="architecture_id")
    information_set = _require_text(shard.get("information_set"), label="information_set")
    fingerprint = _require_sha256(
        shard.get("information_set_fingerprint"), label="information-set fingerprint"
    )
    parameter_count = shard.get("parameter_count")
    if isinstance(parameter_count, bool) or not isinstance(parameter_count, int) or parameter_count < 1:
        _fail("prediction shard parameter_count must be a positive integer")
    return {
        "schema_version": PREDICTION_SHARD_SCHEMA,
        "estimand_id": estimand_slice.estimand_id,
        "partition": partition,
        "seed": seed,
        "registry_sha256": estimand_slice.registry_sha256,
        "registry_entry_set_sha256": estimand_slice.registry_entry_set_sha256,
        "architecture_id": architecture_id,
        "information_set": information_set,
        "information_set_fingerprint": fingerprint,
        "parameter_count": parameter_count,
        "row_ids": list(expected_ids),
        "row_order_sha256": expected_hash,
        "row_set_sha256": expected_hash,
        "mean": list(mean),
        "logvar": None if logvar is None else list(logvar),
    }


def prediction_shard_from_seed_run(
    estimand_slice: PrimaryEstimandSlice,
    seed_run: Mapping,
    *,
    partition: str = "internal_test",
    prediction_field: str = "test_predictions",
) -> dict:
    """Bind a per-estimand seed run to its canonical prediction shard."""

    if not isinstance(seed_run, Mapping) or seed_run.get("status") != "RUN":
        _fail("per-estimand seed run must declare status=RUN")
    if seed_run.get("estimand_id") != estimand_slice.estimand_id:
        _fail("per-estimand seed run is bound to the wrong estimand")
    if seed_run.get("registry_sha256") != estimand_slice.registry_sha256:
        _fail("per-estimand seed run is bound to the wrong registry bytes")
    if seed_run.get("registry_entry_set_sha256") != estimand_slice.registry_entry_set_sha256:
        _fail("per-estimand seed run is bound to the wrong registry entry set")
    prediction = seed_run.get(prediction_field)
    if not isinstance(prediction, Mapping):
        _fail(f"per-estimand seed run lacks {prediction_field}")
    expected_hash = estimand_slice.row_order_sha256_for(partition)
    supplied_hash = prediction.get("row_order_sha256", prediction.get("row_set_sha256"))
    shard = {
        "schema_version": PREDICTION_SHARD_SCHEMA,
        "estimand_id": estimand_slice.estimand_id,
        "partition": partition,
        "seed": seed_run.get("seed"),
        "registry_sha256": seed_run.get("registry_sha256"),
        "registry_entry_set_sha256": seed_run.get("registry_entry_set_sha256"),
        "architecture_id": seed_run.get("architecture_id"),
        "information_set": seed_run.get("information_set"),
        "information_set_fingerprint": seed_run.get("information_set_fingerprint"),
        "parameter_count": seed_run.get("parameter_count"),
        "row_ids": prediction.get("row_ids"),
        "row_order_sha256": supplied_hash,
        "mean": prediction.get("mean"),
        "logvar": prediction.get("logvar"),
    }
    if supplied_hash != expected_hash:
        _fail("per-estimand seed prediction hash is not its canonical row-order hash")
    return validate_prediction_shard(estimand_slice, shard, partition=partition)


def merge_prediction_shards(
    estimand_slices: Sequence[PrimaryEstimandSlice],
    shards: Sequence[Mapping],
    *,
    partition: str = "internal_test",
) -> dict:
    """Merge exactly one validated shard per estimand in global canonical order."""

    slice_index = {item.estimand_id: item for item in estimand_slices}
    if len(slice_index) != len(estimand_slices) or not slice_index:
        _fail("estimand slices must be non-empty with unique estimand_id values")
    raw_index: dict[str, Mapping] = {}
    for shard in shards:
        if not isinstance(shard, Mapping):
            _fail("prediction shards must be objects")
        estimand_id = _require_text(shard.get("estimand_id"), label="prediction shard estimand_id")
        if estimand_id in raw_index:
            _fail("prediction merge received duplicate estimand shards")
        raw_index[estimand_id] = shard
    if set(raw_index) != set(slice_index):
        _fail("prediction merge requires exactly one shard for every estimand slice")
    normalized = {
        estimand_id: validate_prediction_shard(
            slice_index[estimand_id], raw_index[estimand_id], partition=partition
        )
        for estimand_id in sorted(slice_index)
    }
    seeds = {shard["seed"] for shard in normalized.values()}
    architectures = {shard["architecture_id"] for shard in normalized.values()}
    information_sets = {shard["information_set"] for shard in normalized.values()}
    fingerprints = {shard["information_set_fingerprint"] for shard in normalized.values()}
    registry_hashes = {item.registry_sha256 for item in estimand_slices}
    logvar_modes = {shard["logvar"] is None for shard in normalized.values()}
    if len(seeds) != 1:
        _fail("prediction shards use different optimization seeds")
    if len(architectures) != 1:
        _fail("prediction shards use different architectures")
    if len(information_sets) != 1 or len(fingerprints) != 1:
        _fail("prediction shards use unequal information-set contracts")
    if len(registry_hashes) != 1:
        _fail("prediction shards are bound to different estimand registry authorities")
    if len(logvar_modes) != 1:
        _fail("prediction shards mix uncertainty-bearing and point-only outputs")

    values: dict[str, tuple[float, float | None, str]] = {}
    for estimand_id, shard in normalized.items():
        logvar = shard["logvar"]
        for index, row_id in enumerate(shard["row_ids"]):
            if row_id in values:
                _fail("prediction shards contain overlapping row identities")
            values[row_id] = (
                shard["mean"][index],
                None if logvar is None else logvar[index],
                estimand_id,
            )
    row_ids = tuple(sorted(values))
    merged_hash = row_order_sha256(row_ids)
    has_logvar = next(iter(logvar_modes)) is False
    return {
        "schema_version": MERGED_PREDICTION_SCHEMA,
        "status": "RUN",
        "estimand_training_scope": "INDEPENDENT_MODEL_PER_PRIMARY_ESTIMAND",
        "partition": partition,
        "seed": next(iter(seeds)),
        "architecture_id": next(iter(architectures)),
        "information_set": next(iter(information_sets)),
        "information_set_fingerprint": next(iter(fingerprints)),
        "registry_sha256": next(iter(registry_hashes)),
        "parameter_count_scope": "SUM_OF_INDEPENDENT_ESTIMAND_MODELS",
        "parameter_count": sum(shard["parameter_count"] for shard in normalized.values()),
        "parameter_count_by_estimand": {
            estimand_id: shard["parameter_count"]
            for estimand_id, shard in normalized.items()
        },
        "row_ids": list(row_ids),
        "row_order_sha256": merged_hash,
        "row_set_sha256": merged_hash,
        "mean": [values[row_id][0] for row_id in row_ids],
        "logvar": [values[row_id][1] for row_id in row_ids] if has_logvar else None,
        "estimand_id_by_row": [values[row_id][2] for row_id in row_ids],
        "shard_row_order_sha256": {
            estimand_id: shard["row_order_sha256"]
            for estimand_id, shard in normalized.items()
        },
        "registry_entry_set_sha256_by_estimand": {
            estimand_id: shard["registry_entry_set_sha256"]
            for estimand_id, shard in normalized.items()
        },
    }


def validate_estimand_seed_run(
    estimand_slice: PrimaryEstimandSlice,
    seed_run: Mapping,
    *,
    partition: str = "internal_test",
) -> dict:
    """Public validation alias used by per-estimand training orchestration."""

    return prediction_shard_from_seed_run(
        estimand_slice, seed_run, partition=partition
    )


def merge_estimand_seed_runs(
    estimand_slices: Sequence[PrimaryEstimandSlice],
    seed_runs: Sequence[Mapping],
    *,
    partition: str = "internal_test",
) -> dict:
    """Validate per-estimand runs, merge predictions, and retain run lineage."""

    slice_index = {item.estimand_id: item for item in estimand_slices}
    run_index: dict[str, Mapping] = {}
    shards = []
    for run in seed_runs:
        if not isinstance(run, Mapping):
            _fail("per-estimand seed runs must be objects")
        estimand_id = _require_text(run.get("estimand_id"), label="seed-run estimand_id")
        if estimand_id in run_index:
            _fail("seed-run merge received duplicate estimands")
        estimand_slice = slice_index.get(estimand_id)
        if estimand_slice is None:
            _fail("seed run has no corresponding primary-estimand slice")
        run_index[estimand_id] = run
        shards.append(validate_estimand_seed_run(estimand_slice, run, partition=partition))
    if set(run_index) != set(slice_index):
        _fail("seed-run merge requires exactly one run for every primary estimand")
    merged = merge_prediction_shards(estimand_slices, shards, partition=partition)
    return {
        "schema_version": "r3x_merged_primary_estimand_seed_run_v1",
        "status": "RUN",
        "estimand_training_scope": "INDEPENDENT_MODEL_PER_PRIMARY_ESTIMAND",
        "seed": merged["seed"],
        "architecture_id": merged["architecture_id"],
        "information_set": merged["information_set"],
        "information_set_fingerprint": merged["information_set_fingerprint"],
        "parameter_count": merged["parameter_count"],
        "parameter_count_scope": merged["parameter_count_scope"],
        "parameter_count_by_estimand": merged["parameter_count_by_estimand"],
        "test_predictions": {
            key: merged[key]
            for key in (
                "row_ids", "row_order_sha256", "row_set_sha256", "mean", "logvar",
                "estimand_id_by_row", "shard_row_order_sha256",
                "registry_entry_set_sha256_by_estimand",
            )
        },
        "estimand_runs": {
            estimand_id: dict(run_index[estimand_id])
            for estimand_id in sorted(run_index)
        },
    }
