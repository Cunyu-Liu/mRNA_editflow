#!/usr/bin/env python3
"""Per-task decomposition of the SetFlow V5 B2 guided vs unguided adjudication.

Pure post-hoc analysis (no gate logic touched): reads the B2 adjudication JSON
(per_source block), joins source_key -> endpoint_id via the generation
eligibility manifest, and aggregates recovery / hit@1 / deltas per task.

This delivers the preregistered diagnostic requirement: the per-task guided
Delta table used for the three-way attribution (critic quality vs guidance
form vs base) and as the V5-critic baseline guidance profile for V8 (H-V8e).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

PER_TASK_SCHEMA = "route_a_v3_route2_guided_setflow_v5_b2_per_task_decomposition.v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"JSONL input is empty: {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjudication", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    adjudication = json.loads(arguments.adjudication.read_text(encoding="utf-8"))
    per_source = adjudication.get("per_source")
    if not per_source:
        raise RuntimeError(
            "adjudication JSON has no per_source block - cannot decompose by task"
        )

    manifest_rows = _read_jsonl(arguments.source_manifest)
    source_meta = {
        str(row["source_key"]): {
            "endpoint_id": str(row.get("endpoint_id", "UNKNOWN")),
            "study_unit_id": str(row.get("study_unit_id", "UNKNOWN")),
            "region": str(row.get("region", "UNKNOWN")),
        }
        for row in manifest_rows
    }

    buckets: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {
            "unguided_recovery": [],
            "guided_recovery": [],
            "unguided_hit_at_1": [],
            "guided_hit_at_1": [],
        }
    )
    missing_meta = 0
    for source_key, arms in per_source.items():
        meta = source_meta.get(str(source_key))
        if meta is None:
            missing_meta += 1
            continue
        endpoint = meta["endpoint_id"]
        bucket = buckets[endpoint]
        bucket["unguided_recovery"].append(
            float(arms["unguided"]["candidate_recovery_rate"])
        )
        bucket["guided_recovery"].append(
            float(arms["guided"]["candidate_recovery_rate"])
        )
        bucket["unguided_hit_at_1"].append(float(arms["unguided"]["hit_at_1"]))
        bucket["guided_hit_at_1"].append(float(arms["guided"]["hit_at_1"]))

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else float("nan")

    per_task: dict[str, Any] = {}
    for endpoint in sorted(buckets):
        bucket = buckets[endpoint]
        n = len(bucket["unguided_recovery"])
        unguided_recovery = _mean(bucket["unguided_recovery"])
        guided_recovery = _mean(bucket["guided_recovery"])
        unguided_hit = _mean(bucket["unguided_hit_at_1"])
        guided_hit = _mean(bucket["guided_hit_at_1"])
        per_task[endpoint] = {
            "source_count": n,
            "unguided_recovery": unguided_recovery,
            "guided_recovery": guided_recovery,
            "delta_recovery": guided_recovery - unguided_recovery,
            "unguided_hit_at_1": unguided_hit,
            "guided_hit_at_1": guided_hit,
            "delta_hit_at_1": guided_hit - unguided_hit,
            "region": source_meta[
                next(k for k in per_source if source_meta.get(str(k), {}).get("endpoint_id") == endpoint)
            ]["region"]
            if endpoint in {m["endpoint_id"] for m in source_meta.values()}
            else "UNKNOWN",
        }

    overall_guided = float(
        adjudication["guided"]["source_macro_candidate_recovery_rate"]
    )
    overall_unguided = float(
        adjudication["unguided"]["source_macro_candidate_recovery_rate"]
    )
    decomposition_sum_check = sum(
        task["source_count"] for task in per_task.values()
    )

    result = {
        "schema_version": PER_TASK_SCHEMA,
        "status": "B2_PER_TASK_DECOMPOSITION_COMPLETE",
        "note": (
            "post-hoc diagnostic decomposition; gate logic untouched; "
            "per-task deltas are descriptive (no per-task bootstrap, "
            "overall gate CI in the parent adjudication)"
        ),
        "adjudication_input": str(arguments.adjudication),
        "gate_b2_passed": adjudication.get("gate_b2_passed"),
        "gate_b3_passed": adjudication.get("gate_b3_passed"),
        "overall": {
            "unguided_recovery": overall_unguided,
            "guided_recovery": overall_guided,
            "delta_recovery": overall_guided - overall_unguided,
            "decomposed_source_count": decomposition_sum_check,
            "sources_missing_manifest_meta": missing_meta,
        },
        "per_task": per_task,
    }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"{'endpoint':44s} {'n':>4s} {'unguided':>9s} {'guided':>9s} {'delta':>8s} {'d_hit1':>8s}")
    for endpoint, task in per_task.items():
        print(
            f"{endpoint:44s} {task['source_count']:4d} "
            f"{task['unguided_recovery']:9.4f} {task['guided_recovery']:9.4f} "
            f"{task['delta_recovery']:+8.4f} {task['delta_hit_at_1']:+8.4f}"
        )
    print(f"TOTAL decomposed sources: {decomposition_sum_check}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
