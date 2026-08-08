"""Train/development-only estimand slicing for staged R3X execution.

Unlike :mod:`scripts.r3x.estimands`, this module has no internal-test field or
API.  It is intended for the first disclosure stage, where neither test
features nor test outcomes are available to the process.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from .estimands import row_order_sha256, stable_row_id
from .protocol import ProtocolHoldError, assert_ordinary_record, canonical_sha256


DEVELOPMENT_PARTITIONS = ("train", "development")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SHARED_ESTIMAND_FIELDS = (
    "raw_unit",
    "direction",
    "transform",
    "valid_domain",
    "replicate_policy",
)


def _hold(code: str, message: str, **details) -> None:
    raise ProtocolHoldError(code, message, details=details)


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _hold("HOLD_DEVELOPMENT_ESTIMAND_AUTHORITY", f"{label} must be non-empty text")
    return value.strip()


def _sha256(value: object, *, label: str) -> str:
    text = _text(value, label=label).lower()
    if not _HEX64.fullmatch(text):
        _hold("HOLD_DEVELOPMENT_ESTIMAND_AUTHORITY", f"{label} must be lowercase SHA-256")
    return text


def _canonical_rows(rows: Sequence[Mapping]) -> tuple[dict, ...]:
    indexed: dict[str, dict] = {}
    for row in rows:
        row_id = stable_row_id(row)
        if not _HEX64.fullmatch(row_id):
            _hold(
                "HOLD_DEVELOPMENT_ROW_IDENTITY",
                "staged development rows require lowercase SHA-256 identities",
                row_id=row_id,
            )
        if row_id in indexed:
            _hold(
                "HOLD_DEVELOPMENT_ROW_IDENTITY",
                "staged development rows contain duplicate identities",
                row_id=row_id,
            )
        indexed[row_id] = dict(row)
    return tuple(indexed[row_id] for row_id in sorted(indexed))


@dataclass(frozen=True)
class DevelopmentEstimandSlice:
    """One primary estimand exposed only through train/development rows."""

    estimand_id: str
    studies: tuple[str, ...]
    endpoints: tuple[str, ...]
    registry_sha256: str
    registry_entry_set_sha256: str
    train: tuple[dict, ...]
    development: tuple[dict, ...]

    def rows_for(self, partition: str) -> tuple[dict, ...]:
        if partition not in DEVELOPMENT_PARTITIONS:
            _hold(
                "HOLD_TEST_PARTITION_NOT_DISCLOSED",
                "development estimand slices expose only train/development",
                partition=partition,
            )
        return getattr(self, partition)

    def row_ids_for(self, partition: str) -> tuple[str, ...]:
        return tuple(stable_row_id(row) for row in self.rows_for(partition))

    def row_order_sha256_for(self, partition: str) -> str:
        return row_order_sha256(self.row_ids_for(partition))

    @property
    def partition_row_set_sha256(self) -> dict[str, str]:
        return {
            partition: canonical_sha256(sorted(self.row_ids_for(partition)))
            for partition in DEVELOPMENT_PARTITIONS
        }

    @property
    def partition_binding_sha256(self) -> str:
        return canonical_sha256({
            "schema_version": "r3x_development_estimand_partition_binding_v1",
            "estimand_id": self.estimand_id,
            "registry_sha256": self.registry_sha256,
            "registry_entry_set_sha256": self.registry_entry_set_sha256,
            "ordered_row_ids": {
                partition: list(self.row_ids_for(partition))
                for partition in DEVELOPMENT_PARTITIONS
            },
            "internal_test_disclosed": False,
            "sealed_access": False,
        })


def build_development_estimand_slices(
    rows: Sequence[Mapping],
    estimand_registry: Mapping,
    *,
    estimand_registry_sha256: str,
    require_all_primary_registry_entries: bool = True,
) -> tuple[DevelopmentEstimandSlice, ...]:
    """Validate and split an outcome-bearing train/development authority.

    Any internal-test row, missing/unknown partition, protected record,
    unregistered endpoint, or incomplete study/endpoint partition support is a
    structured HOLD before a model or output artifact is created.
    """

    registry_sha = _sha256(
        estimand_registry_sha256, label="endpoint estimand registry SHA-256"
    )
    if not isinstance(estimand_registry, Mapping):
        _hold("HOLD_DEVELOPMENT_ESTIMAND_AUTHORITY", "estimand registry must be an object")
    if estimand_registry.get("schema_version") != "r3_endpoint_estimand_registry_v1":
        _hold("HOLD_DEVELOPMENT_ESTIMAND_AUTHORITY", "unsupported estimand registry schema")
    if estimand_registry.get("status") != "FROZEN":
        _hold("HOLD_DEVELOPMENT_ESTIMAND_AUTHORITY", "estimand registry must be FROZEN")
    raw_entries = estimand_registry.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        _hold("HOLD_DEVELOPMENT_ESTIMAND_AUTHORITY", "estimand registry lacks entries")

    by_key: dict[tuple[str, str], dict] = {}
    by_id: dict[str, list[dict]] = {}
    for raw in raw_entries:
        if not isinstance(raw, Mapping) or raw.get("role") != "PRIMARY":
            continue
        entry = dict(raw)
        study = _text(entry.get("study"), label="registry study")
        endpoint = _text(entry.get("endpoint"), label="registry endpoint")
        estimand_id = _text(entry.get("estimand_id"), label="registry estimand_id")
        key = (study, endpoint)
        if key in by_key:
            _hold(
                "HOLD_DEVELOPMENT_ESTIMAND_AUTHORITY",
                "primary registry study/endpoint keys must be unique",
                study=study,
                endpoint=endpoint,
            )
        by_key[key] = entry
        by_id.setdefault(estimand_id, []).append(entry)
    if not by_key:
        _hold("HOLD_DEVELOPMENT_ESTIMAND_AUTHORITY", "registry has no PRIMARY entries")
    for estimand_id, entries in by_id.items():
        reference = entries[0]
        for entry in entries[1:]:
            for field in _SHARED_ESTIMAND_FIELDS:
                if entry.get(field) != reference.get(field):
                    _hold(
                        "HOLD_DEVELOPMENT_ESTIMAND_AUTHORITY",
                        "shared estimand semantics differ across studies/endpoints",
                        estimand_id=estimand_id,
                        field=field,
                    )

    grouped: dict[str, dict[str, list[dict]]] = {}
    represented_keys: set[tuple[str, str]] = set()
    support: dict[tuple[str, str], set[str]] = {}
    seen_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            _hold("HOLD_DEVELOPMENT_ROW_SCHEMA", "development rows must be objects")
        assert_ordinary_record(row, purpose="staged train/development row")
        partition = row.get("partition")
        if partition not in DEVELOPMENT_PARTITIONS:
            code = (
                "HOLD_TEST_PARTITION_PREMATURELY_DISCLOSED"
                if partition == "internal_test"
                else "HOLD_DEVELOPMENT_PARTITION_AUTHORITY"
            )
            _hold(code, "train/development authority contains a forbidden partition", partition=partition)
        row_id = stable_row_id(row)
        if not _HEX64.fullmatch(row_id) or row_id in seen_ids:
            _hold(
                "HOLD_DEVELOPMENT_ROW_IDENTITY",
                "row IDs must be unique lowercase SHA-256 values",
                row_id=row_id,
            )
        seen_ids.add(row_id)
        study = _text(row.get("study"), label="row study")
        endpoint = _text(
            row.get("endpoint") or row.get("endpoint_id"), label="row endpoint"
        )
        entry = by_key.get((study, endpoint))
        if entry is None:
            _hold(
                "HOLD_DEVELOPMENT_ESTIMAND_BINDING",
                "row is not covered by a PRIMARY estimand entry",
                study=study,
                endpoint=endpoint,
            )
        estimand_id = _text(row.get("estimand_id"), label="row estimand_id")
        if estimand_id != entry.get("estimand_id"):
            _hold(
                "HOLD_DEVELOPMENT_ESTIMAND_BINDING",
                "row estimand_id disagrees with frozen registry",
                study=study,
                endpoint=endpoint,
                estimand_id=estimand_id,
            )
        delta = row.get("delta")
        if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(float(delta)):
            _hold("HOLD_DEVELOPMENT_TARGET", "development delta must be finite", row_id=row_id)
        for field in ("source_component_id", "source_group_id_v2"):
            _text(row.get(field), label=f"row {field}")
        grouped.setdefault(estimand_id, {name: [] for name in DEVELOPMENT_PARTITIONS})[
            partition
        ].append(dict(row))
        represented_keys.add((study, endpoint))
        support.setdefault((study, endpoint), set()).add(partition)

    if not rows:
        _hold("HOLD_DEVELOPMENT_EMPTY", "train/development authority has no rows")
    if require_all_primary_registry_entries and represented_keys != set(by_key):
        _hold(
            "HOLD_DEVELOPMENT_ESTIMAND_COVERAGE",
            "not every PRIMARY registry entry is represented",
            missing=sorted(set(by_key) - represented_keys),
            unexpected=sorted(represented_keys - set(by_key)),
        )
    incomplete = {
        f"{study}|{endpoint}": sorted(set(DEVELOPMENT_PARTITIONS) - partitions)
        for (study, endpoint), partitions in sorted(support.items())
        if partitions != set(DEVELOPMENT_PARTITIONS)
    }
    if incomplete:
        _hold(
            "HOLD_DEVELOPMENT_ESTIMAND_COVERAGE",
            "every represented study/endpoint requires train and development support",
            incomplete=incomplete,
        )

    slices = []
    for estimand_id, partitions in sorted(grouped.items()):
        entries = by_id[estimand_id]
        if any(not partitions[name] for name in DEVELOPMENT_PARTITIONS):
            _hold(
                "HOLD_DEVELOPMENT_ESTIMAND_COVERAGE",
                "each estimand requires train and development rows",
                estimand_id=estimand_id,
            )
        entry_set_sha = canonical_sha256(
            sorted(
                (dict(entry) for entry in entries),
                key=lambda entry: (str(entry.get("study")), str(entry.get("endpoint"))),
            )
        )
        slices.append(DevelopmentEstimandSlice(
            estimand_id=estimand_id,
            studies=tuple(sorted({_text(entry.get("study"), label="study") for entry in entries})),
            endpoints=tuple(sorted({_text(entry.get("endpoint"), label="endpoint") for entry in entries})),
            registry_sha256=registry_sha,
            registry_entry_set_sha256=entry_set_sha,
            train=_canonical_rows(partitions["train"]),
            development=_canonical_rows(partitions["development"]),
        ))
    return tuple(slices)
