#!/usr/bin/env python3
"""Freeze the strongest legal Development generation baseline under matched budgets."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping


class GenerationBaselineSelectionError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GenerationBaselineSelectionError(message)


def _finite(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    result = float(value)
    _require(math.isfinite(result), f"{label} is not finite")
    return result


def _optional_finite(value: Any, label: str) -> float | None:
    return None if value is None else _finite(value, label)


def _budget_signature(
    generation: Mapping[str, Any],
    *,
    forward_equivalent_budget_per_source: float,
) -> tuple[dict[str, int], dict[str, int] | None]:
    signature = {}
    critic_signature = {}
    critic_budget_presence = set()
    for source_key, result in generation["per_source"].items():
        compute = result["compute"]
        forward_budget = compute["critic_forward_budget"]
        critic_budget_presence.add(forward_budget is not None)
        signature[str(source_key)] = int(result["candidate_budget"])
        generator_nfe = _finite(compute["generator_nfe"], "generator NFE")
        critic_forwards = _finite(compute["critic_forwards"], "critic forwards")
        evaluator_forwards = _finite(
            compute["independent_evaluator_forwards"], "independent evaluator forwards"
        )
        total_forwards = _finite(
            compute["total_forward_equivalents"], "total forward equivalents"
        )
        if forward_budget is None:
            _require(critic_forwards == 0.0, f"unbudgeted critic calls occurred: {source_key}")
        _require(
            min(generator_nfe, critic_forwards, evaluator_forwards, total_forwards) >= 0.0,
            f"negative forward accounting occurred: {source_key}",
        )
        _require(
            math.isclose(
                total_forwards,
                generator_nfe + critic_forwards + evaluator_forwards,
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
            f"total forward-equivalent accounting does not close: {source_key}",
        )
        _require(
            total_forwards <= forward_equivalent_budget_per_source,
            f"matched forward-equivalent budget exceeded: {source_key}",
        )
        if forward_budget is not None:
            critic_signature[str(source_key)] = int(forward_budget)
    _require(len(signature) == int(generation["source_count"]), "per-source results do not close to source count")
    _require(len(critic_budget_presence) == 1, "critic budget presence differs within one method")
    return signature, critic_signature if critic_signature else None


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_source_bootstrap(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    maximize: bool,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    _require(set(left) == set(right), "paired bootstrap source eligibility differs")
    keys = sorted(left)
    _require(len(keys) >= 2, "paired source bootstrap requires at least two sources")
    rng = random.Random(seed)
    differences = []
    for _ in range(iterations):
        sampled = [keys[rng.randrange(len(keys))] for _ in keys]
        left_mean = sum(left[key] for key in sampled) / len(sampled)
        right_mean = sum(right[key] for key in sampled) / len(sampled)
        differences.append(left_mean - right_mean if maximize else right_mean - left_mean)
    point_left = sum(left.values()) / len(left)
    point_right = sum(right.values()) / len(right)
    return {
        "analysis_unit": "SOURCE",
        "source_count": len(keys),
        "bootstrap_iterations": iterations,
        "defined_bootstrap_iterations": len(differences),
        "point_leader_advantage": point_left - point_right if maximize else point_right - point_left,
        "leader_advantage_ci_95": [_quantile(differences, 0.025), _quantile(differences, 0.975)],
    }


def select(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        payload["schema_version"] == "route_a_v3_route2_generation_baseline_selection_input.v2",
        "unexpected selection input schema",
    )
    _require(payload["selection_pool"] == "DEVELOPMENT_MEASURED_NEIGHBORHOOD", "selection is not Development measured data")
    _require(payload["evaluation_release_state"] == "CLOSED", "Evaluation was opened during baseline selection")
    bootstrap_iterations = int(payload["bootstrap_iterations"])
    _require(bootstrap_iterations >= 1000, "generation bootstrap budget is below 1000 iterations")
    bootstrap_seed = int(payload["bootstrap_seed"])
    _require(
        "forward_equivalent_budget_per_source" in payload,
        "matched forward-equivalent budget is missing",
    )
    forward_equivalent_budget_per_source = _finite(
        payload["forward_equivalent_budget_per_source"],
        "forward-equivalent budget per source",
    )
    _require(
        forward_equivalent_budget_per_source > 0.0
        and forward_equivalent_budget_per_source.is_integer(),
        "forward-equivalent budget per source must be a positive integer",
    )
    required_methods = {str(value) for value in payload["required_method_ids"]}
    entries = payload["baseline_evaluations"]
    _require(entries, "no generation baseline evaluations were provided")
    methods = [str(entry["method_id"]) for entry in entries]
    _require(len(methods) == len(set(methods)), "generation baseline evaluation is duplicated")
    _require(set(methods) == required_methods, "generation baseline method set is incomplete or changed")

    candidates = []
    matched_candidate_signature = None
    matched_critic_signature = None
    for entry in entries:
        method_id = str(entry["method_id"])
        evaluation = entry["evaluation"]
        _require(evaluation["schema_version"] == "route_a_v3_route2_generation_evaluation.v2", f"evaluation schema changed: {method_id}")
        _require(evaluation["evaluation_release_state"] == "CLOSED", f"Evaluation opened for {method_id}")
        _require(evaluation["measured_neighborhood_pool"] == "DEVELOPMENT", f"measured pool is not Development: {method_id}")
        generation = evaluation["generation"]
        _require(generation["method_id"] == method_id, f"method id differs inside evaluation: {method_id}")
        _require(generation["hard_legality_rate"] == 1.0, f"illegal candidates produced by {method_id}")
        _require(generation["edit_budget_violation_count"] == 0, f"edit-budget violation produced by {method_id}")
        _require(generation["candidate_budget_violation_count"] == 0, f"candidate-budget violation produced by {method_id}")
        _require(generation["generated_candidates_grant_canonical_credit"] is False, f"canonical credit enabled for {method_id}")
        candidate_signature, critic_signature = _budget_signature(
            generation,
            forward_equivalent_budget_per_source=forward_equivalent_budget_per_source,
        )
        if matched_candidate_signature is None:
            matched_candidate_signature = candidate_signature
        else:
            _require(candidate_signature == matched_candidate_signature, f"source/candidate budgets differ for {method_id}")
        if critic_signature is not None:
            _require(
                set(critic_signature.values()) == {int(forward_equivalent_budget_per_source)},
                f"critic budget does not use the matched forward-equivalent budget for {method_id}",
            )
            if matched_critic_signature is None:
                matched_critic_signature = critic_signature
            else:
                _require(critic_signature == matched_critic_signature, f"critic budgets differ for {method_id}")

        measured = evaluation["measured_neighborhood"]
        _require(
            measured["candidate_support_mode"] == "CLOSED_MEASURED_SUPPORT",
            (
                "open generated support cannot be selected by measured NDCG because "
                f"unknown outcomes are not zero gain; independent evaluator required: {method_id}"
            ),
        )
        _require(
            measured["unknown_generated_candidates_are_zero_gain"] is False,
            f"unknown generated candidates were assigned zero gain: {method_id}",
        )
        source_count = int(measured["source_count"])
        _require(source_count == int(generation["source_count"]), f"measured source count differs for {method_id}")
        ndcg_defined_count = int(measured["source_closed_measured_ndcg_defined_count"])
        regret_defined_count = int(measured["source_normalized_regret_defined_count"])
        _require(0 < ndcg_defined_count <= source_count, f"measured NDCG is not defined for any source: {method_id}")
        _require(0 <= regret_defined_count <= source_count, f"regret defined count is invalid: {method_id}")
        mean_total_forwards = sum(
            _finite(result["compute"]["total_forward_equivalents"], "total forward equivalents")
            for result in generation["per_source"].values()
        ) / source_count
        per_source = measured["per_source"]
        _require(set(per_source) == set(generation["per_source"]), f"measured source coverage differs for {method_id}")
        ndcg_by_source = {
            str(source_key): _finite(row["closed_measured_ndcg_at_k"], f"source NDCG for {method_id}/{source_key}")
            for source_key, row in per_source.items()
            if row["closed_measured_ndcg_at_k"] is not None
        }
        regret_by_source = {
            str(source_key): _finite(row["normalized_regret"], f"source regret for {method_id}/{source_key}")
            for source_key, row in per_source.items()
            if row["normalized_regret"] is not None
        }
        _require(len(ndcg_by_source) == ndcg_defined_count, f"source NDCG closure differs for {method_id}")
        _require(len(regret_by_source) == regret_defined_count, f"source regret closure differs for {method_id}")
        candidates.append({
            "method_id": method_id,
            "source_count": source_count,
            "source_closed_measured_ndcg_defined_count": ndcg_defined_count,
            "source_normalized_regret_defined_count": regret_defined_count,
            "source_macro_closed_measured_ndcg_at_k": _optional_finite(
                measured["source_macro_closed_measured_ndcg_at_k"], f"closed measured NDCG for {method_id}"
            ),
            "source_macro_normalized_regret": _optional_finite(
                measured["source_macro_normalized_regret"], f"normalized regret for {method_id}"
            ),
            "source_macro_measured_top_k_recovery_at_k": _finite(
                measured["source_macro_measured_top_k_recovery_at_k"], f"measured top-k recovery for {method_id}"
            ),
            "source_macro_candidate_recovery_rate": _finite(
                measured["source_macro_candidate_recovery_rate"], f"candidate recovery for {method_id}"
            ),
            "mean_total_forward_equivalents_per_source": mean_total_forwards,
            "critic_budget_class": "MATCHED_CRITIC_BUDGET" if critic_signature is not None else "NO_CRITIC_CALLS",
            "ndcg_by_source": ndcg_by_source,
            "regret_by_source": regret_by_source,
        })

    _require(
        len({candidate["source_closed_measured_ndcg_defined_count"] for candidate in candidates}) == 1,
        "closed measured NDCG source eligibility differs across methods",
    )
    _require(
        len({frozenset(candidate["ndcg_by_source"]) for candidate in candidates}) == 1,
        "measured NDCG source identities differ across methods",
    )

    finite_ndcg = any(candidate["source_macro_closed_measured_ndcg_at_k"] is not None for candidate in candidates)
    comparisons = []
    if finite_ndcg:
        ranked = sorted(candidates, key=lambda candidate: (
            candidate["source_macro_closed_measured_ndcg_at_k"] is None,
            0.0 if candidate["source_macro_closed_measured_ndcg_at_k"] is None else -candidate["source_macro_closed_measured_ndcg_at_k"],
            candidate["source_macro_normalized_regret"] is None,
            math.inf if candidate["source_macro_normalized_regret"] is None else candidate["source_macro_normalized_regret"],
            -candidate["source_macro_measured_top_k_recovery_at_k"],
            candidate["mean_total_forward_equivalents_per_source"],
            candidate["method_id"],
        ))
        selection_metric = "DEVELOPMENT_CLOSED_MEASURED_NDCG_THEN_REGRET_THEN_COST"
        point_winner = ranked[0]
        uncertainty_equivalent = [point_winner]
        for comparison_index, candidate in enumerate(ranked[1:]):
            if candidate["source_macro_closed_measured_ndcg_at_k"] is None:
                continue
            comparison = paired_source_bootstrap(
                point_winner["ndcg_by_source"], candidate["ndcg_by_source"],
                maximize=True, iterations=bootstrap_iterations,
                seed=bootstrap_seed + comparison_index,
            )
            comparisons.append({
                "metric": "CLOSED_MEASURED_NDCG_AT_K",
                "point_leader_method_id": point_winner["method_id"],
                "candidate_method_id": candidate["method_id"],
                **comparison,
            })
            lower, upper = comparison["leader_advantage_ci_95"]
            if lower <= 0.0 <= upper:
                uncertainty_equivalent.append(candidate)
    else:
        _require(
            any(candidate["source_macro_normalized_regret"] is not None for candidate in candidates),
            "all measured NDCG and regret values are undefined",
        )
        ranked = sorted(candidates, key=lambda candidate: (
            candidate["source_macro_normalized_regret"] is None,
            math.inf if candidate["source_macro_normalized_regret"] is None else candidate["source_macro_normalized_regret"],
            -candidate["source_macro_measured_top_k_recovery_at_k"],
            candidate["mean_total_forward_equivalents_per_source"],
            candidate["method_id"],
        ))
        selection_metric = "DEVELOPMENT_MEASURED_REGRET_ALL_CLOSED_NDCG_UNDEFINED"
        _require(
            len({frozenset(candidate["regret_by_source"]) for candidate in candidates}) == 1,
            "measured regret source identities differ across methods",
        )
        point_winner = ranked[0]
        uncertainty_equivalent = [point_winner]
        for comparison_index, candidate in enumerate(ranked[1:]):
            if candidate["source_macro_normalized_regret"] is None:
                continue
            comparison = paired_source_bootstrap(
                point_winner["regret_by_source"], candidate["regret_by_source"],
                maximize=False, iterations=bootstrap_iterations,
                seed=bootstrap_seed + comparison_index,
            )
            comparisons.append({
                "metric": "NORMALIZED_REGRET",
                "point_leader_method_id": point_winner["method_id"],
                "candidate_method_id": candidate["method_id"],
                **comparison,
            })
            lower, upper = comparison["leader_advantage_ci_95"]
            if lower <= 0.0 <= upper:
                uncertainty_equivalent.append(candidate)
    winner = min(
        uncertainty_equivalent,
        key=lambda candidate: (
            candidate["mean_total_forward_equivalents_per_source"],
            candidate["method_id"],
        ),
    )
    for candidate in candidates:
        candidate.pop("ndcg_by_source")
        candidate.pop("regret_by_source")
    return {
        "schema_version": "route_a_v3_route2_strongest_generation_baseline.v2",
        "status": "DEVELOPMENT_STRONGEST_GENERATION_BASELINE_FROZEN",
        "strongest_generation_baseline_id": winner["method_id"],
        "selection_primary_metric": selection_metric,
        "point_leader_method_id": point_winner["method_id"],
        "bootstrap_uncertainty_equivalent_method_ids": sorted(
            candidate["method_id"] for candidate in uncertainty_equivalent
        ),
        "paired_source_bootstrap": comparisons,
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": bootstrap_seed,
        "uncertainty_tiebreak": "LOWEST_MEAN_TOTAL_FORWARD_EQUIVALENTS_THEN_METHOD_ID",
        "matched_source_and_candidate_budget": True,
        "matched_forward_equivalent_budget": True,
        "forward_equivalent_budget_per_source": forward_equivalent_budget_per_source,
        "critic_budget_matched_within_critic_using_methods": True,
        "generator_nfe_accounting_validated": True,
        "required_method_ids": sorted(required_methods),
        "all_candidates_ranked": ranked,
        "evaluation_outcomes_accessed": False,
        "main_guided_model_result_used_for_selection": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"output already exists: {args.output}")
    result = select(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
