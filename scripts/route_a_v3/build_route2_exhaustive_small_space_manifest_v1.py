#!/usr/bin/env python3
"""Select real Route 2 cohorts that fit a complete matched-budget search."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.route_a_v3.run_route2_search_generation_baselines_v1 import (
    legal_space_size,
)


class ExhaustiveManifestError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExhaustiveManifestError(message)


def build(
    rows: list[Mapping[str, Any]],
    *,
    max_critic_forwards: int,
    exhaustive_space_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require(max_critic_forwards > 0, "critic-forward budget must be positive")
    _require(exhaustive_space_limit > 0, "exhaustive-space limit must be positive")
    selected = []
    excluded_by_reason: Counter[str] = Counter()
    spaces = []
    seen = set()
    for raw in rows:
        row = dict(raw)
        source_key = str(row["source_key"])
        _require(source_key not in seen, f"duplicate source cohort: {source_key}")
        seen.add(source_key)
        _require(
            row.get("evaluation_outcomes_included") is False,
            "Evaluation outcome entered exhaustive source eligibility",
        )
        _require(
            row.get("generated_candidates_grant_canonical_credit") is False,
            "generated candidate canonical credit was enabled",
        )
        source = str(row["source_sequence"]).upper().replace("T", "U")
        space = legal_space_size(len(source), int(row["edit_budget"]))
        if space > exhaustive_space_limit:
            excluded_by_reason["ABOVE_EXHAUSTIVE_SPACE_LIMIT"] += 1
            continue
        if space > max_critic_forwards:
            excluded_by_reason["ABOVE_MATCHED_CRITIC_FORWARD_BUDGET"] += 1
            continue
        selected.append({**row, "exhaustive_legal_space_size": space})
        spaces.append(space)
    _require(selected, "no real source cohort fits complete matched-budget search")
    summary = {
        "schema_version": "route_a_v3_route2_exhaustive_small_space_manifest.v1",
        "status": "OUTCOME_BLIND_MATCHED_BUDGET_EXHAUSTIVE_SUBSET_READY",
        "input_source_cohort_count": len(rows),
        "selected_source_cohort_count": len(selected),
        "excluded_source_cohort_count": len(rows) - len(selected),
        "excluded_by_reason": dict(sorted(excluded_by_reason.items())),
        "max_critic_forwards_per_source": max_critic_forwards,
        "exhaustive_space_limit": exhaustive_space_limit,
        "minimum_selected_legal_space_size": min(spaces),
        "maximum_selected_legal_space_size": max(spaces),
        "evaluation_outcomes_accessed": False,
        "generated_candidates_grant_canonical_credit": False,
        "scientific_role": "REAL_SMALL_SPACE_ORACLE_REFERENCE_NOT_FULL_COHORT_STRONGEST_SELECTOR",
    }
    return selected, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--max-critic-forwards", type=int, required=True)
    parser.add_argument("--exhaustive-space-limit", type=int, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output_manifest.exists(), f"output exists: {args.output_manifest}")
    _require(not args.output_summary.exists(), f"output exists: {args.output_summary}")
    rows = [
        json.loads(line)
        for line in args.source_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected, summary = build(
        rows,
        max_critic_forwards=args.max_critic_forwards,
        exhaustive_space_limit=args.exhaustive_space_limit,
    )
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
