#!/usr/bin/env python3
"""Compare full-cohort baselines with the small-space exhaustive critic reference."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]


class ExhaustiveReferenceComparisonError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExhaustiveReferenceComparisonError(message)


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required input is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"input is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    _require(path.is_file(), f"required input is absent: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(rows, f"input is empty: {path}")
    return rows


def _finite(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    result = float(value)
    _require(math.isfinite(result), f"{label} is not finite")
    return result


def _group(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_key"])].append(row)
    return dict(grouped)


def _best_critic(rows: list[Mapping[str, Any]]) -> tuple[str, float]:
    ordered = sorted(
        (
            (str(row["candidate_sequence"]), _finite(row["critic_score"], "critic score"))
            for row in rows
        ),
        key=lambda item: (-item[1], item[0]),
    )
    return ordered[0]


def _max_evaluator_uplift(rows: list[Mapping[str, Any]]) -> float:
    uplifts = []
    for row in rows:
        score = _finite(row["independent_evaluator_score"], "independent evaluator score")
        source_score = _finite(
            row["source_independent_evaluator_score"],
            "source independent evaluator score",
        )
        uplifts.append(score - source_score)
    return max(uplifts)


def compare(
    *,
    config: Mapping[str, Any],
    source_rows: list[Mapping[str, Any]],
    exhaustive_rows: list[Mapping[str, Any]],
    full_method_rows: Mapping[str, list[Mapping[str, Any]]],
    exhaustive_scoring_summary: Mapping[str, Any],
    full_suite_summary: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        config.get("schema_version") == "route_a_v3_route2_exhaustive_small_space_reference.v1",
        "unexpected exhaustive-reference schema",
    )
    _require(config.get("evaluation_outcomes_accessed") is False, "Evaluation was accessed")
    _require(config.get("full_cohort_strongest_selector_eligible") is False, "subset entered strongest selector")
    _require(config.get("guided_xeditflow_allowed") is False, "guided XEditFlow entered reference")
    _require(
        full_suite_summary.get("status") == "MATCHED_GENERATION_BASELINE_SUITE_COMPLETED",
        "full-cohort matched suite is incomplete",
    )
    _require(
        exhaustive_scoring_summary.get("status") == "FROZEN_INDEPENDENT_EVALUATOR_SCORING_COMPLETE"
        and exhaustive_scoring_summary.get("cpu_fallback_used") is False,
        "exhaustive reference lacks completed GPU independent scoring",
    )

    source_keys = {str(row["source_key"]) for row in source_rows}
    _require(len(source_keys) == int(config["source_cohort_count"]), "small-space source count differs")
    _require(
        all(int(row["candidate_budget"]) == int(config["candidate_budget_per_source"]) for row in source_rows),
        "small-space candidate budget differs",
    )
    exhaustive_by_source = _group(exhaustive_rows)
    _require(set(exhaustive_by_source) == source_keys, "exhaustive source coverage differs")
    _require({str(row["method_id"]) for row in exhaustive_rows} == {"exhaustive"}, "reference method changed")

    exhaustive_reference = {}
    for source_key in sorted(source_keys):
        rows = exhaustive_by_source[source_key]
        _require(
            0 < len(rows) <= int(config["candidate_budget_per_source"]),
            f"exhaustive candidate count differs: {source_key}",
        )
        best_sequence, best_score = _best_critic(rows)
        exhaustive_reference[source_key] = {
            "critic_best_candidate": best_sequence,
            "critic_best_score": best_score,
            "independent_evaluator_max_uplift_over_source": _max_evaluator_uplift(rows),
        }

    method_summaries = []
    for method_id, rows in sorted(full_method_rows.items()):
        subset_rows = [row for row in rows if str(row["source_key"]) in source_keys]
        by_source = _group(subset_rows)
        _require(set(by_source) == source_keys, f"full method source coverage differs: {method_id}")
        critic_gaps = []
        recovered = []
        evaluator_advantages = []
        per_source = {}
        for source_key in sorted(source_keys):
            method_rows = by_source[source_key]
            method_best_sequence, method_best_score = _best_critic(method_rows)
            reference = exhaustive_reference[source_key]
            critic_gap = reference["critic_best_score"] - method_best_score
            _require(critic_gap >= -1e-7, f"method exceeds exhaustive critic optimum: {method_id}/{source_key}")
            critic_gap = max(0.0, critic_gap)
            optimum_recovered = any(
                str(row["candidate_sequence"]) == reference["critic_best_candidate"]
                for row in method_rows
            )
            evaluator_advantage = (
                _max_evaluator_uplift(method_rows)
                - reference["independent_evaluator_max_uplift_over_source"]
            )
            critic_gaps.append(critic_gap)
            recovered.append(float(optimum_recovered))
            evaluator_advantages.append(evaluator_advantage)
            per_source[source_key] = {
                "exhaustive_critic_best_candidate": reference["critic_best_candidate"],
                "method_critic_best_candidate": method_best_sequence,
                "critic_optimality_gap": critic_gap,
                "critic_optimum_recovered": optimum_recovered,
                "independent_evaluator_advantage_over_exhaustive_critic_top32": evaluator_advantage,
            }
        method_summaries.append(
            {
                "method_id": method_id,
                "source_count": len(source_keys),
                "source_macro_critic_optimality_gap": sum(critic_gaps) / len(critic_gaps),
                "critic_optimum_recovery_rate": sum(recovered) / len(recovered),
                "source_macro_independent_evaluator_advantage_over_exhaustive_critic_top32": (
                    sum(evaluator_advantages) / len(evaluator_advantages)
                ),
                "per_source": per_source,
            }
        )

    return {
        "schema_version": "route_a_v3_route2_exhaustive_small_space_reference_comparison.v1",
        "status": "SMALL_SPACE_EXHAUSTIVE_GUIDING_CRITIC_REFERENCE_COMPLETED",
        "scientific_role": config["scientific_role"],
        "source_count": len(source_keys),
        "legal_space_size_per_source": int(config["legal_space_size_per_source"]),
        "candidate_budget_per_source": int(config["candidate_budget_per_source"]),
        "critic_forward_budget_per_source": int(config["critic_forward_budget_per_source"]),
        "full_cohort_strongest_selector_eligible": False,
        "method_summaries": method_summaries,
        "evaluation_outcomes_accessed": False,
        "guided_xeditflow_run": False,
        "scientific_claim_status": "DEVELOPMENT_SMALL_SPACE_REFERENCE_ONLY",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = _read_json(args.config)
    output = args.output or Path(str(config["comparison_output_path"]))
    _require(not output.exists(), f"output already exists: {output}")
    source_rows = _read_jsonl(Path(str(config["source_manifest_path"])))
    exhaustive_rows = _read_jsonl(Path(str(config["independent_scored_output_path"])))
    jobs = _read_json(REPO_ROOT / str(config["full_cohort_jobs_config"]))
    full_method_rows = {
        str(job["method_id"]): _read_jsonl(Path(str(job["output_path"])))
        for job in jobs["jobs"]
    }
    exhaustive_summary = _read_json(
        Path(str(config["independent_scored_output_path"])).with_suffix(".jsonl.summary.json")
    )
    full_suite_summary = _read_json(Path(str(config["full_cohort_suite_summary_path"])))
    result = compare(
        config=config,
        source_rows=source_rows,
        exhaustive_rows=exhaustive_rows,
        full_method_rows=full_method_rows,
        exhaustive_scoring_summary=exhaustive_summary,
        full_suite_summary=full_suite_summary,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "source_count": result["source_count"],
        "method_count": len(result["method_summaries"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
