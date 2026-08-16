#!/usr/bin/env python3
"""Build Development source/action-space eligibility for Route 2 generation."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ALPHABET = set("ACGU")


class EligibilityError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EligibilityError(message)


def _normalize(sequence: Any) -> str:
    value = str(sequence).upper().replace("T", "U")
    _require(value and set(value) <= ALPHABET, "sequence is outside the RNA alphabet")
    return value


def _finite(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    result = float(value)
    _require(math.isfinite(result), f"{label} is not finite")
    return result


def legal_space_size(source_length: int, edit_budget: int) -> int:
    return sum(math.comb(source_length, count) * 3 ** count for count in range(min(source_length, edit_budget) + 1))


def load_split_manifest(path: Path, requested_split: str) -> set[str]:
    selected = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            _require(row["pool_assignment"] == "DEVELOPMENT", "Evaluation record entered generation eligibility")
            if row["split"] == requested_split:
                selected.add(str(row["canonical_record_id"]))
    _require(selected, f"Development split is empty: {requested_split}")
    return selected


def load_grouped_records(canonical_paths: Iterable[Path], selected_ids: set[str]) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    records: dict[str, dict[str, Any]] = {}
    for path in canonical_paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                record_id = str(row["canonical_record_id"])
                if record_id not in selected_ids:
                    continue
                _require(record_id not in records, f"canonical record duplicated: {record_id}")
                _require(row["pool_assignment"] == "DEVELOPMENT", "Evaluation outcome entered generation eligibility")
                source = _normalize(row["source_sequence"])
                candidate = _normalize(row["candidate_sequence"])
                _require(len(source) == len(candidate), f"length-changing record reached SUB generation: {record_id}")
                records[record_id] = {
                    "record_id": record_id,
                    "study_unit_id": str(row["study_unit_id"]),
                    "source_id": str(row["source_id"]),
                    "biological_context_id": str(row["biological_context_id"]),
                    "endpoint_id": str(row["endpoint_id"]),
                    "region": str(row["region"]),
                    "assay_id": str(row["assay_id"]),
                    "source": source,
                    "candidate": candidate,
                    "edit_count": sum(left != right for left, right in zip(source, candidate)),
                    "outcome": _finite(row["direction_normalized_delta"], "measured outcome"),
                }
    _require(set(records) == selected_ids, "canonical inputs do not exactly cover the selected split")
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records.values():
        grouped[(row["study_unit_id"], row["source_id"], row["biological_context_id"], row["endpoint_id"])].append(row)
    return grouped


def build_eligibility(
    groups: Mapping[tuple[str, str, str, str], list[dict[str, Any]]],
    *,
    requested_split: str,
    edit_budgets: tuple[int, ...],
    candidate_budget: int,
    minimum_measured_candidates: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    _require(edit_budgets == (1, 3, 5), "edit-budget cohorts must remain 1/3/5")
    _require(candidate_budget > 0 and minimum_measured_candidates >= 2, "candidate thresholds are invalid")
    source_rows = []
    measured_rows = []
    exclusions = Counter()
    for group_key in sorted(groups):
        rows = groups[group_key]
        sources = {row["source"] for row in rows}
        regions = {row["region"] for row in rows}
        assays = {row["assay_id"] for row in rows}
        _require(len(sources) == 1, f"source sequence differs within group: {group_key}")
        _require(len(regions) == 1, f"region differs within group: {group_key}")
        _require(len(assays) == 1, f"assay differs within group: {group_key}")
        source = next(iter(sources))
        by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_candidate[row["candidate"]].append(row)
        collapsed: dict[str, dict[str, Any]] = {}
        ambiguous_candidate_outcome = False
        for sequence, members in by_candidate.items():
            outcomes = {member["outcome"] for member in members}
            if len(outcomes) != 1:
                exclusions["AMBIGUOUS_DUPLICATE_CANDIDATE_OUTCOME_SOURCE_GROUP"] += 1
                ambiguous_candidate_outcome = True
                break
            collapsed[sequence] = {
                "candidate_sequence": sequence,
                "measured_direction_normalized_delta": next(iter(outcomes)),
                "canonical_record_ids": sorted(member["record_id"] for member in members),
                "edit_count": members[0]["edit_count"],
            }
        if ambiguous_candidate_outcome:
            continue
        for budget in edit_budgets:
            eligible = [row for row in collapsed.values() if row["edit_count"] <= budget]
            if len(eligible) < minimum_measured_candidates:
                exclusions[f"B{budget}_TOO_FEW_MEASURED_CANDIDATES"] += 1
                continue
            source_key = "::".join((*group_key, f"B{budget}"))
            source_rows.append({
                "schema_version": "route_a_v3_route2_generation_source_eligibility.v1",
                "source_key": source_key,
                "study_unit_id": group_key[0],
                "source_id": group_key[1],
                "biological_context_id": group_key[2],
                "endpoint_id": group_key[3],
                "region": next(iter(regions)),
                "assay_id": next(iter(assays)),
                "source_sequence": source,
                "development_split": requested_split,
                "edit_budget": budget,
                "candidate_budget": min(candidate_budget, legal_space_size(len(source), budget)),
                "measured_candidate_count": len(eligible),
                "generated_candidates_grant_canonical_credit": False,
                "evaluation_outcomes_included": False,
            })
            for row in sorted(eligible, key=lambda item: item["candidate_sequence"]):
                measured_rows.append({
                    "schema_version": "route_a_v3_route2_development_measured_neighborhood.v1",
                    "source_key": source_key,
                    **row,
                    "pool_assignment": "DEVELOPMENT",
                    "split": requested_split,
                })
    _require(source_rows, "no generation-eligible source/action-space cohort")
    summary = {
        "schema_version": "route_a_v3_route2_generation_eligibility_summary.v1",
        "status": "DEVELOPMENT_GENERATION_ELIGIBILITY_MATERIALIZED",
        "development_split": requested_split,
        "source_budget_cohort_count": len(source_rows),
        "unique_source_group_count": len({row["source_key"].rsplit("::", 1)[0] for row in source_rows}),
        "measured_neighborhood_row_count": len(measured_rows),
        "edit_budget_cohort_counts": dict(sorted(Counter(f"B{row['edit_budget']}" for row in source_rows).items())),
        "exclusions": dict(sorted(exclusions.items())),
        "evaluation_records_read": 0,
        "generated_candidates_grant_canonical_credit": False,
    }
    return source_rows, measured_rows, summary


def execute(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"output already exists: {output_dir}")
    selected = load_split_manifest(Path(config["development_manifest"]), str(config["development_split"]))
    groups = load_grouped_records([Path(path) for path in config["canonical_paths"]], selected)
    source_rows, measured_rows, summary = build_eligibility(
        groups,
        requested_split=str(config["development_split"]),
        edit_budgets=tuple(int(value) for value in config["edit_budgets"]),
        candidate_budget=int(config["candidate_budget"]),
        minimum_measured_candidates=int(config["minimum_measured_candidates"]),
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        (temporary / "source_eligibility.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in source_rows), encoding="utf-8"
        )
        (temporary / "measured_neighborhood.private.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in measured_rows), encoding="utf-8"
        )
        (temporary / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.rename(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = execute(config, args.output_dir or Path(config["output_directory"]))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
