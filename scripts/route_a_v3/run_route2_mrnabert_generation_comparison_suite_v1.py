#!/usr/bin/env python3
"""Score and compare guided XEditFlow with frozen matched Development baselines."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCORE_SCRIPT = REPO_ROOT / "scripts/route_a_v3/score_route2_generation_independent_evaluator_v1.py"
EVALUATE_SCRIPT = REPO_ROOT / "scripts/route_a_v3/evaluate_route2_generation_v1.py"


class GenerationComparisonError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GenerationComparisonError(message)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    _require(path.is_file(), f"{label} is absent: {path}")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    _require(rows and all(isinstance(row, dict) for row in rows), f"{label} is empty or malformed")
    return rows


def _finite(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    result = float(value)
    _require(math.isfinite(result), f"{label} is not finite")
    return result


def _paired_bootstrap(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    _require(set(left) == set(right) and left, "paired bootstrap source keys differ")
    keys = sorted(left)
    differences = np.asarray([left[key] - right[key] for key in keys], dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(keys), size=(iterations, len(keys)))
    bootstrap = differences[sampled].mean(axis=1)
    return {
        "source_count": len(keys),
        "point_difference": float(differences.mean()),
        "ci_95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "iterations": iterations,
        "seed": seed,
    }


def validate_protocol(config: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    _require(
        config.get("schema_version")
        == "route_a_v3_route2_mrnabert_generation_comparison_protocol.v1",
        "generation comparison schema differs",
    )
    guided = str(config["guided_method_id"])
    baselines = tuple(str(value) for value in config["required_baseline_method_ids"])
    _require(guided not in baselines and len(baselines) == len(set(baselines)), "guided/baseline method identities overlap")
    _require(len(baselines) == 7 and "unguided_learned_base_flow_g0" in baselines, "seven required baselines are not frozen")
    _require(set(config["candidate_paths"]) == {guided, *baselines}, "candidate paths do not cover exact methods")
    _require(
        config.get("candidate_support_mode") == "OPEN_GENERATED_SUPPORT"
        and config.get("measured_neighborhood_pool") == "DEVELOPMENT"
        and config.get("evaluation_release_state") == "CLOSED"
        and config.get("evaluation_outcomes_accessed") is False,
        "generation comparison opened Evaluation or misclassified support",
    )
    _require(int(config["bootstrap_iterations"]) >= 1000, "bootstrap iteration count is too small")
    return guided, baselines


def method_statistics(
    method_id: str,
    evaluation: Mapping[str, Any],
    budgets: Mapping[str, int],
    *,
    guided_method_id: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    _require(evaluation.get("schema_version") == "route_a_v3_route2_generation_evaluation.v2", f"evaluation schema differs: {method_id}")
    _require(evaluation.get("evaluation_release_state") == "CLOSED" and evaluation.get("measured_neighborhood_pool") == "DEVELOPMENT", f"Evaluation opened or pool differs: {method_id}")
    generation = evaluation["generation"]
    measured = evaluation["measured_neighborhood"]
    _require(generation["method_id"] == method_id, f"method identity differs: {method_id}")
    _require(generation["hard_legality_rate"] == 1.0 and int(generation["edit_budget_violation_count"]) == 0 and int(generation["candidate_budget_violation_count"]) == 0, f"legality or candidate budget failed: {method_id}")
    _require(generation["generated_candidates_grant_canonical_credit"] is False, f"canonical credit enabled: {method_id}")
    _require(measured["candidate_support_mode"] == "OPEN_GENERATED_SUPPORT" and measured["unknown_generated_candidates_are_zero_gain"] is False, f"unknown outcomes were treated as zero: {method_id}")
    per_source = generation["per_source"]
    _require(set(per_source) == set(budgets), f"source coverage differs: {method_id}")
    uplift: dict[str, float] = {}
    generation_compute = []
    for source_key, row in per_source.items():
        _require(int(row["candidate_count"]) == int(row["candidate_budget"]), f"candidate count differs: {method_id}/{source_key}")
        score = row["independent_evaluator_score"]
        _require(score is not None and int(score["count"]) > 0, f"independent evaluator score absent: {method_id}/{source_key}")
        uplift[source_key] = _finite(score["max_uplift_over_source"], f"independent uplift {method_id}/{source_key}")
        compute = row["compute"]
        used = _finite(compute["generator_nfe"], "generator NFE") + _finite(compute["critic_forwards"], "critic forwards")
        _require(used <= budgets[source_key], f"generation exceeded matched total budget: {method_id}/{source_key}")
        if method_id == guided_method_id:
            _require(used == budgets[source_key], f"guided accounting differs from matched budget: {source_key}")
        if method_id not in {guided_method_id, "unguided_learned_base_flow_g0"}:
            _require(int(compute["critic_forward_budget"]) == budgets[source_key], f"search critic cap differs: {method_id}/{source_key}")
        generation_compute.append(used)
    statistics = {
        "method_id": method_id,
        "source_count": len(per_source),
        "source_macro_independent_evaluator_max_uplift": float(np.mean(list(uplift.values()))),
        "source_macro_measured_top_k_recovery_at_k": measured["source_macro_measured_top_k_recovery_at_k"],
        "source_macro_candidate_recovery_rate": measured["source_macro_candidate_recovery_rate"],
        "source_macro_unique_candidate_rate": generation["source_macro_unique_candidate_rate"],
        "source_macro_pairwise_hamming_diversity": generation["source_macro_pairwise_hamming_diversity"],
        "mean_generation_forward_equivalents_per_source": float(np.mean(generation_compute)),
        "maximum_generation_forward_equivalents_per_source": float(np.max(generation_compute)),
    }
    return statistics, uplift


def select_comparison(
    config: Mapping[str, Any],
    evaluations: Mapping[str, Mapping[str, Any]],
    budgets: Mapping[str, int],
) -> dict[str, Any]:
    guided, baselines = validate_protocol(config)
    _require(set(evaluations) == {guided, *baselines}, "evaluation methods are incomplete")
    stats: dict[str, dict[str, Any]] = {}
    uplifts: dict[str, dict[str, float]] = {}
    for method_id, evaluation in evaluations.items():
        stats[method_id], uplifts[method_id] = method_statistics(
            method_id, evaluation, budgets, guided_method_id=guided
        )
    point_leader = max(
        baselines,
        key=lambda method: (
            stats[method]["source_macro_independent_evaluator_max_uplift"],
            stats[method]["source_macro_measured_top_k_recovery_at_k"],
            -stats[method]["mean_generation_forward_equivalents_per_source"],
            method,
        ),
    )
    equivalent = [point_leader]
    baseline_comparisons = []
    for index, method in enumerate(baselines):
        if method == point_leader:
            continue
        comparison = _paired_bootstrap(
            uplifts[point_leader], uplifts[method],
            iterations=int(config["bootstrap_iterations"]),
            seed=int(config["bootstrap_seed"]) + index,
        )
        baseline_comparisons.append({"left": point_leader, "right": method, **comparison})
        if comparison["ci_95"][0] <= 0.0 <= comparison["ci_95"][1]:
            equivalent.append(method)
    strongest = min(
        equivalent,
        key=lambda method: (
            stats[method]["mean_generation_forward_equivalents_per_source"],
            method,
        ),
    )
    guided_comparison = _paired_bootstrap(
        uplifts[guided], uplifts[strongest],
        iterations=int(config["bootstrap_iterations"]),
        seed=int(config["bootstrap_seed"]) + 100,
    )
    lower, upper = guided_comparison["ci_95"]
    stable_advantage = lower > 0.0
    return {
        "schema_version": "route_a_v3_route2_mrnabert_generation_comparison.v1",
        "status": (
            "DEVELOPMENT_INDEPENDENT_EVALUATOR_GUIDED_ADVANTAGE"
            if stable_advantage
            else "DEVELOPMENT_NO_STABLE_GUIDED_ADVANTAGE"
        ),
        "guided_method_id": guided,
        "strongest_generation_baseline_id": strongest,
        "baseline_point_leader_id": point_leader,
        "baseline_bootstrap_equivalent_method_ids": sorted(equivalent),
        "baseline_comparisons": baseline_comparisons,
        "guided_vs_strongest_baseline": guided_comparison,
        "guided_advantage_ci_excludes_zero": stable_advantage,
        "guided_disadvantage_ci_excludes_zero": upper < 0.0,
        "all_method_statistics": [stats[method] for method in [guided, *baselines]],
        "matched_budget_rule": "GUIDED_TOTAL_FORWARD_EQUIVALENTS_AS_SEARCH_CRITIC_CAP_PER_SOURCE",
        "source_specific_budget_count": len(budgets),
        "selection_pool": "DEVELOPMENT_INDEPENDENT_EVALUATOR_OPEN_GENERATED_SUPPORT",
        "evaluation_outcomes_accessed": False,
        "measured_biological_improvement_established": False,
        "generated_candidates_grant_canonical_credit": False,
        "scientific_claim_status": "INDEPENDENT_EVALUATOR_DEVELOPMENT_ONLY_NOT_EXTERNAL_OR_MEASURED_SUCCESS",
    }


def _run(command: list[str], stdout_path: Path, stderr_path: Path) -> None:
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, text=True, check=False)
    _require(result.returncode == 0, f"comparison child failed: {command[1]}")


def execute(config_path: Path) -> dict[str, Any]:
    config = _read_json(config_path, "generation comparison config")
    guided, baselines = validate_protocol(config)
    output_directory = Path(str(config["output_directory"]))
    _require(not output_directory.exists(), f"comparison output exists: {output_directory}")
    adjudication = _read_json(Path(str(config["independent_evaluator_adjudication_path"])), "independent evaluator adjudication")
    _require(adjudication.get("evaluation_outcomes_accessed") is False and adjudication.get("development_test_outcomes_accessed") is False, "independent evaluator accessed protected outcomes")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir()
    if adjudication.get("status") != "INDEPENDENT_GENERATION_EVALUATOR_QUALIFIED":
        result = {
            "schema_version": "route_a_v3_route2_mrnabert_generation_comparison.v1",
            "status": "BLOCKED_INDEPENDENT_EVALUATOR_NO_GO",
            "independent_evaluator_status": adjudication.get("status"),
            "candidate_generation_preserved": True,
            "strongest_generation_baseline_selected": False,
            "evaluation_outcomes_accessed": False,
            "scientific_claim_status": "NOT_ESTABLISHED",
        }
        (output_directory / "final_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
    _require(adjudication.get("candidate_rerun_authorized") is True, "qualified evaluator did not authorize candidate scoring")
    compute_rows = _read_jsonl(Path(str(config["guided_compute_by_source_path"])), "guided compute")
    budgets = {str(row["source_key"]): int(row["matched_search_critic_forward_budget"]) for row in compute_rows}
    _require(len(budgets) == len(compute_rows) and all(value > 0 for value in budgets.values()), "guided budget table is invalid")
    methods = (guided, *baselines)
    scoring_configs = output_directory / "scoring_configs"
    scored_directory = output_directory / "scored_candidates"
    evaluation_directory = output_directory / "evaluations"
    logs = output_directory / "logs"
    for directory in (scoring_configs, scored_directory, evaluation_directory, logs):
        directory.mkdir()
    started = time.time()
    evaluations = {}
    for method in methods:
        candidate_path = Path(str(config["candidate_paths"][method]))
        scoring_config = {
            "schema_version": "route_a_v3_route2_generation_independent_evaluator_job.v1",
            "method_id": method,
            "evaluator_checkpoint_path": config["independent_evaluator_checkpoint_path"],
            "guiding_checkpoint_path": config["guiding_checkpoint_path"],
            "source_manifest_path": config["source_manifest_path"],
            "evaluator_frozen_before_candidate_generation": True,
            "evaluation_outcomes_used_to_select_evaluator": 0,
            "device": config["device"],
            "physical_gpu_index": config["physical_gpu_index"],
            "candidate_path": str(candidate_path),
        }
        scoring_config_path = scoring_configs / f"{method}.json"
        scoring_config_path.write_text(json.dumps(scoring_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        scored_path = scored_directory / f"{method}.private.jsonl"
        _run(
            [sys.executable, str(SCORE_SCRIPT), "--config", str(scoring_config_path), "--output", str(scored_path)],
            logs / f"{method}.score.stdout.log", logs / f"{method}.score.stderr.log",
        )
        evaluation_path = evaluation_directory / f"{method}.json"
        _run(
            [
                sys.executable, str(EVALUATE_SCRIPT),
                "--source-manifest", str(config["source_manifest_path"]),
                "--candidates", str(scored_path),
                "--measured-neighborhood", str(config["measured_neighborhood_path"]),
                "--measured-neighborhood-pool", "DEVELOPMENT",
                "--candidate-support-mode", "OPEN_GENERATED_SUPPORT",
                "--evaluation-release-state", "CLOSED",
                "--k", str(config["k"]),
                "--output", str(evaluation_path),
            ],
            logs / f"{method}.evaluate.stdout.log", logs / f"{method}.evaluate.stderr.log",
        )
        evaluations[method] = _read_json(evaluation_path, f"{method} evaluation")
    result = select_comparison(config, evaluations, budgets)
    result["independent_evaluator_status"] = adjudication["status"]
    result["wall_time_seconds"] = time.time() - started
    result["evaluation_paths"] = {
        method: str(evaluation_directory / f"{method}.json") for method in methods
    }
    (output_directory / "comparison_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    (output_directory / "comparison_summary.json").write_text(serialized, encoding="utf-8")
    (output_directory / "final_summary.json").write_text(serialized, encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    execute(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
