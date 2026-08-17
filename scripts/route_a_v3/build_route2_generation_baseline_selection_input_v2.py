#!/usr/bin/env python3
"""Compose the matched-budget independent-evaluator baseline selection input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


class GenerationSelectionInputError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GenerationSelectionInputError(message)


def _read(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required input is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"input is not an object: {path}")
    return value


def build(protocol: Mapping[str, Any], jobs: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        protocol["schema_version"] == "route_a_v3_route2_generation_matched_compute_repair_protocol.v1",
        "unexpected protocol schema",
    )
    _require(
        jobs["schema_version"] == "route_a_v3_route2_generation_independent_evaluator_jobs.v1",
        "unexpected evaluator-jobs schema",
    )
    required_methods = [str(value) for value in protocol["required_method_ids"]]
    job_by_method = {str(row["method_id"]): row for row in jobs["jobs"]}
    _require(len(job_by_method) == len(jobs["jobs"]), "evaluator job method is duplicated")
    _require(set(job_by_method) == set(required_methods), "evaluator jobs do not cover required methods")
    evaluation_root = Path(str(protocol["independent_evaluation_output_root"]))
    entries = []
    for method_id in required_methods:
        job = job_by_method[method_id]
        evaluation_path = evaluation_root / f"{method_id}_evaluation_v2.json"
        scored_output = Path(str(job["output_path"]))
        summary_path = scored_output.with_suffix(scored_output.suffix + ".summary.json")
        entries.append({
            "method_id": method_id,
            "evaluation": _read(evaluation_path),
            "independent_evaluator_summary": _read(summary_path),
        })
    return {
        "schema_version": "route_a_v3_route2_generation_baseline_selection_input.v2",
        "selection_pool": "DEVELOPMENT_MEASURED_NEIGHBORHOOD",
        "selection_evidence_mode": "INDEPENDENT_EVALUATOR_OPEN_SUPPORT",
        "evaluation_release_state": "CLOSED",
        "bootstrap_iterations": int(protocol["selection_bootstrap_iterations"]),
        "bootstrap_seed": int(protocol["selection_bootstrap_seed"]),
        "forward_equivalent_budget_per_source": int(protocol["forward_equivalent_budget_per_source"]),
        "critic_forward_budget_per_source": int(protocol["search_critic_forward_budget_per_source"]),
        "required_method_ids": required_methods,
        "baseline_evaluations": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--evaluator-jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"output already exists: {args.output}")
    result = build(_read(args.protocol), _read(args.evaluator_jobs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema_version": result["schema_version"],
        "method_count": len(result["baseline_evaluations"]),
        "selection_evidence_mode": result["selection_evidence_mode"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
