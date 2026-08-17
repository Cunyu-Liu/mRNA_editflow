#!/usr/bin/env python3
"""Restrict the Development measured neighborhood to an exhaustive source subset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


class ExhaustiveMeasuredNeighborhoodError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExhaustiveMeasuredNeighborhoodError(message)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _require(bool(rows), f"input is empty: {path}")
    return rows


def build(
    source_rows: list[Mapping[str, Any]],
    measured_rows: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_keys: set[str] = set()
    for row in source_rows:
        source_key = str(row["source_key"])
        _require(source_key not in source_keys, f"source duplicated: {source_key}")
        _require(
            row.get("evaluation_outcomes_included") is False,
            "Evaluation outcome entered exhaustive source manifest",
        )
        source_keys.add(source_key)

    selected: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    seen_candidates: set[tuple[str, str]] = set()
    for raw in measured_rows:
        row = dict(raw)
        source_key = str(row["source_key"])
        if source_key not in source_keys:
            continue
        _require(
            row.get("pool_assignment") == "DEVELOPMENT",
            "non-Development outcome entered exhaustive measured neighborhood",
        )
        _require(
            row.get("split") == "VALIDATION",
            "non-Validation outcome entered exhaustive measured neighborhood",
        )
        candidate = str(row["candidate_sequence"]).upper().replace("T", "U")
        key = (source_key, candidate)
        _require(key not in seen_candidates, f"measured candidate duplicated: {source_key}/{candidate}")
        seen_candidates.add(key)
        selected.append(row)
        selected_counts[source_key] += 1

    _require(
        set(selected_counts) == source_keys,
        "measured neighborhood does not cover every exhaustive source",
    )
    summary = {
        "schema_version": "route_a_v3_route2_exhaustive_measured_neighborhood.v1",
        "status": "DEVELOPMENT_EXHAUSTIVE_MEASURED_NEIGHBORHOOD_READY",
        "input_source_count": len(source_keys),
        "input_measured_row_count": len(measured_rows),
        "selected_source_count": len(selected_counts),
        "selected_measured_row_count": len(selected),
        "excluded_measured_row_count": len(measured_rows) - len(selected),
        "minimum_rows_per_source": min(selected_counts.values()),
        "maximum_rows_per_source": max(selected_counts.values()),
        "pool_assignment": "DEVELOPMENT",
        "split": "VALIDATION",
        "evaluation_outcomes_accessed": False,
        "scientific_role": "DEVELOPMENT_SMALL_SPACE_REFERENCE_EVALUATION_ONLY",
    }
    return selected, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--measured-neighborhood", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"output exists: {args.output}")
    _require(not args.summary.exists(), f"summary exists: {args.summary}")
    rows, summary = build(
        _read_jsonl(args.source_manifest),
        _read_jsonl(args.measured_neighborhood),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
