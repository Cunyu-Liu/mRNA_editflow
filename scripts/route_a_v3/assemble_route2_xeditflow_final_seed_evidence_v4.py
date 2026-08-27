#!/usr/bin/env python3
"""Assemble one V4 seed's six-method terminal evidence and paired intervals."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditflow_value_training_v4 import BASE_FLOW_SEEDS_V4


METHODS_V4 = {
    "full_soft_value_smc",
    "unguided_setflow",
    "first_order_guidance",
    "simple_rate_guidance",
    "generate_then_rerank",
    "strongest_matched_baseline",
}
V4_GENERATED_METHODS = METHODS_V4 - {"strongest_matched_baseline"}


class XEditFlowFinalSeedEvidenceV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowFinalSeedEvidenceV4Error(message)


def _finite(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} is nonfinite")
    return result


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _require(
        bool(rows) and all(isinstance(row, dict) for row in rows),
        f"JSONL input is empty or invalid: {path}",
    )
    return rows


def source_paired_mean_bootstrap_ci_v4(
    differences: Sequence[float], *, iterations: int, seed: int
) -> list[float]:
    values = np.asarray(differences, dtype=np.float64)
    _require(
        values.ndim == 1
        and len(values) >= 2
        and bool(np.isfinite(values).all()),
        "V4 final paired differences are invalid",
    )
    _require(iterations == 10_000, "V4 final paired bootstrap count changed")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _require_unprotected(payload: Mapping[str, Any], label: str) -> None:
    _require(
        payload.get("development_test_outcomes_accessed_after_atomic_test") is False
        and payload.get("new_final_evaluation_outcomes_accessed") is False,
        f"V4 final seed evidence accessed protected outcome: {label}",
    )


def _defined_ndcg_by_source(
    closed: Mapping[str, Any], label: str
) -> dict[str, float]:
    _require(
        closed.get("status") == "XEDITFLOW_V4_CLOSED_NEIGHBORHOOD_COMPLETE"
        and closed.get("undefined_sources_are_not_filled_with_zero") is True,
        f"V4 closed evidence is incomplete or zero-filled: {label}",
    )
    rows = closed.get("per_source")
    _require(
        isinstance(rows, Mapping) and bool(rows),
        f"V4 closed per-source evidence is absent: {label}",
    )
    result: dict[str, float] = {}
    for source_key, row in rows.items():
        if row.get("status") == "DEFINED":
            result[str(source_key)] = _finite(
                row.get("ndcg"), f"V4 closed source NDCG {label}/{source_key}"
            )
        else:
            _require(
                row.get("ndcg") is None,
                f"V4 undefined closed source was assigned NDCG: {label}/{source_key}",
            )
    _require(bool(result), f"V4 closed method has no defined source: {label}")
    return result


def _present_closed_metric_by_source(
    closed: Mapping[str, Any], *, label: str, metric: str
) -> dict[str, float]:
    rows = closed.get("per_source")
    _require(
        isinstance(rows, Mapping) and bool(rows),
        f"V4 closed per-source evidence is absent: {label}",
    )
    result = {
        str(source_key): _finite(
            row.get(metric), f"V4 closed source {metric} {label}/{source_key}"
        )
        for source_key, row in rows.items()
        if row.get(metric) is not None
    }
    _require(
        bool(result),
        f"V4 closed method has no source-defined {metric}: {label}",
    )
    return result


def _source_max_critic_score(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    seed: int,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        _require(
            str(row.get("method_id")) == method
            and int(row.get("base_flow_training_seed", -1)) == seed
            and row.get("development_test_outcomes_accessed_after_atomic_test")
            is False
            and row.get("new_final_evaluation_outcomes_accessed") is False,
            f"V4 critic diagnostic candidate provenance differs: {method}",
        )
        grouped.setdefault(str(row["source_key"]), []).append(
            _finite(row.get("critic_self_score"), f"V4 critic self score {method}")
        )
    _require(bool(grouped), f"V4 critic diagnostic rows are empty: {method}")
    return {key: max(values) for key, values in grouped.items()}


def assemble_final_seed_evidence_v4(
    evidence: Mapping[str, Mapping[str, Any]],
    *,
    base_flow_training_seed: int,
    selected_combination: Sequence[float],
    equal_wall_time_sensitivity: Mapping[str, Any],
    full_independent_evaluator: Mapping[str, Any],
    full_candidate_rows: Sequence[Mapping[str, Any]],
    unguided_candidate_rows: Sequence[Mapping[str, Any]],
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    seed = int(base_flow_training_seed)
    _require(seed in BASE_FLOW_SEEDS_V4, "V4 final evidence SetFlow seed differs")
    combination = tuple(float(value) for value in selected_combination)
    _require(
        len(combination) == 3
        and combination[0] in {0.0, 0.5, 1.0}
        and combination[1] in {0.5, 1.0}
        and combination[2] in {0.5, 1.0, 2.0},
        "V4 final evidence selected combination differs",
    )
    _require(set(evidence) == METHODS_V4, "V4 final evidence method inventory differs")
    _require(
        equal_wall_time_sensitivity.get("status")
        == "XEDITFLOW_V4_EQUAL_WALL_TIME_SENSITIVITY_COMPLETE"
        and int(equal_wall_time_sensitivity.get("base_flow_training_seed", -1))
        == seed
        and isinstance(equal_wall_time_sensitivity.get("methods"), Mapping)
        and set(equal_wall_time_sensitivity["methods"]) == METHODS_V4
        and equal_wall_time_sensitivity.get(
            "five_v4_methods_use_terminal_scoring_reconciled_compute"
        )
        is True
        and equal_wall_time_sensitivity.get("all_network_forwards_separately_charged")
        is True
        and equal_wall_time_sensitivity.get("matched_compute_schema")
        == "MatchedComputeRecordV4",
        "V4 final equal-wall evidence is incomplete",
    )
    _require_unprotected(equal_wall_time_sensitivity, "equal-wall sensitivity")
    common_prefix_count = int(
        equal_wall_time_sensitivity.get("common_source_prefix_count", -1)
    )
    _require(2 <= common_prefix_count <= 891, "V4 equal-wall prefix differs")
    _require(
        full_independent_evaluator.get("status")
        == "XEDITFLOW_V4_INDEPENDENT_EVALUATOR_COMPARISON_COMPLETE"
        and full_independent_evaluator.get("analysis_unit") == "SOURCE"
        and int(full_independent_evaluator.get("base_flow_training_seed", -1))
        == seed
        and tuple(
            float(value) for value in full_independent_evaluator.get("combination", ())
        )
        == combination
        and full_independent_evaluator.get("independent_evaluator_used_for_gradient")
        is False
        and full_independent_evaluator.get(
            "development_test_outcomes_accessed_after_atomic_test"
        )
        is False
        and int(full_independent_evaluator.get("new_final_evaluation_outcome_reads", -1))
        == 0,
        "V4 final independent-evaluator evidence is incomplete",
    )
    method_results: dict[str, Any] = {}
    ndcg_by_method: dict[str, dict[str, float]] = {}
    source_inventory: dict[str, set[str]] = {}
    all_compute_ok = True
    for method in sorted(METHODS_V4):
        bundle = evidence[method]
        expected_bundle = (
            {"closed", "open", "generation", "terminal_critic"}
            if method in V4_GENERATED_METHODS
            else {"closed", "open", "generation"}
        )
        _require(
            set(bundle) == expected_bundle,
            f"V4 final method bundle differs: {method}",
        )
        closed = bundle["closed"]
        opened = bundle["open"]
        generation = bundle["generation"]
        for label, payload in (
            ("closed", closed),
            ("open", opened),
            ("generation", generation),
        ):
            _require_unprotected(payload, f"{method}/{label}")
        _require(
            str(closed.get("method_id"))
            == str(opened.get("method_id"))
            == str(generation.get("method_id"))
            == method
            and int(closed.get("base_flow_training_seed", -1))
            == int(opened.get("base_flow_training_seed", -1))
            == int(generation.get("base_flow_training_seed", -1))
            == seed,
            f"V4 final method or seed identity differs: {method}",
        )
        _require(
            opened.get("status") == "XEDITFLOW_V4_OPEN_GENERATION_METRICS_COMPLETE",
            f"V4 open evidence is incomplete: {method}",
        )
        if method in V4_GENERATED_METHODS:
            closed_combination = tuple(
                _finite(
                    closed.get(key),
                    f"V4 closed combination {method}/{key}",
                )
                for key in ("kappa", "temperature", "beta_max")
            )
            open_combination = tuple(
                _finite(
                    opened.get(key),
                    f"V4 open combination {method}/{key}",
                )
                for key in ("kappa", "temperature", "beta_max")
            )
            _require(
                generation.get("status")
                == "XEDITFLOW_V4_SMC_GENERATION_COMPLETE_PENDING_TERMINAL_CRITIC_SCORING"
                and generation.get("setflow_mode_is_fixed_trajectory_state") is True
                and generation.get("free_action_ratio_head_used") is False,
                f"V4 generation mechanism evidence differs: {method}",
            )
            observed_combination = tuple(
                float(generation.get(key, -1))
                for key in ("kappa", "temperature", "beta_max")
            )
            _require(
                closed_combination
                == open_combination
                == observed_combination
                == combination,
                f"V4 generation combination differs: {method}",
            )
            terminal = bundle["terminal_critic"]
            _require_unprotected(terminal, f"{method}/terminal_critic")
            _require(
                terminal.get("status")
                == "XEDITFLOW_V4_CANDIDATE_CRITIC_SCORING_COMPLETE"
                and str(terminal.get("method_id")) == method
                and int(terminal.get("base_flow_training_seed", -1)) == seed
                and tuple(
                    float(terminal.get(key, -1))
                    for key in ("kappa", "temperature", "beta_max")
                )
                == combination
                and terminal.get("reservation_reconciled_for_every_source") is True
                and terminal.get("candidate_support_unchanged_by_terminal_rerank")
                is True,
                f"V4 terminal Critic evidence differs: {method}",
            )
            maximum_compute = int(
                terminal.get("maximum_total_forward_equivalents_per_source", -1)
            )
            replay_failures = int(generation.get("replay_failure_count", -1))
        else:
            _require(
                generation.get("status")
                == "XEDITFLOW_V4_STRONGEST_BASELINE_ADAPTER_COMPLETE"
                and generation.get("frozen_baseline_reselected_for_v4") is False,
                "V4 strongest baseline adapter evidence differs",
            )
            maximum_compute = int(
                generation.get("maximum_forward_equivalents_per_source", -1)
            )
            replay_failures = int(
                generation.get("trajectory_replay_failure_count", -1)
            )
        compute_ok = 0 <= maximum_compute <= 320
        all_compute_ok = all_compute_ok and compute_ok
        ndcg_by_method[method] = _defined_ndcg_by_source(closed, method)
        regret_by_source = _present_closed_metric_by_source(
            closed, label=method, metric="normalized_regret"
        )
        top_1_by_source = _present_closed_metric_by_source(
            closed, label=method, metric="top_1_recall"
        )
        source_inventory[method] = {str(key) for key in closed["per_source"]}
        macro_ndcg = _finite(
            closed.get("source_macro_ndcg"), f"V4 closed macro NDCG {method}"
        )
        macro_regret = _finite(
            closed.get("source_macro_normalized_regret"),
            f"V4 closed macro regret {method}",
        )
        macro_top_1 = _finite(
            closed.get("source_macro_top_1_recall"),
            f"V4 closed macro top-1 {method}",
        )
        _require(
            math.isclose(
                macro_ndcg,
                float(np.mean(list(ndcg_by_method[method].values()))),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"V4 closed macro NDCG is not the defined-source mean: {method}",
        )
        _require(
            math.isclose(
                macro_regret,
                float(np.mean(list(regret_by_source.values()))),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"V4 closed macro regret is not the source-defined mean: {method}",
        )
        _require(
            math.isclose(
                macro_top_1,
                float(np.mean(list(top_1_by_source.values()))),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"V4 closed macro top-1 is not the source-defined mean: {method}",
        )
        equal_wall = equal_wall_time_sensitivity["methods"][method]
        _require(
            isinstance(equal_wall, Mapping)
            and "A100" in str(equal_wall.get("accelerator_name", "")).upper(),
            f"V4 equal-wall accelerator differs: {method}",
        )
        metrics = {
            "closed_source_macro_ndcg": macro_ndcg,
            "closed_source_macro_normalized_regret": macro_regret,
            "closed_source_macro_top_1_recall": macro_top_1,
            "open_source_macro_candidate_recovery": _finite(
                opened.get("source_macro_candidate_recovery"),
                f"V4 open recovery {method}",
            ),
            "open_source_macro_top_k_recovery": _finite(
                opened.get("source_macro_top_k_recovery"),
                f"V4 open top-k {method}",
            ),
            "open_source_macro_unique_candidate_rate": _finite(
                opened.get("source_macro_unique_candidate_rate"),
                f"V4 open unique {method}",
            ),
            "independent_evaluator_margin_over_strongest_baseline": (
                _finite(
                    full_independent_evaluator.get(
                        "paired_margin_over_strongest_baseline"
                    ),
                    "V4 full evaluator margin",
                )
                if method == "full_soft_value_smc"
                else 0.0
            ),
            "hard_legality_rate": _finite(
                opened.get("hard_legality_rate"), f"V4 hard legality {method}"
            ),
            "edit_budget_violation_count": int(
                generation.get("edit_budget_violation_count", -1)
            ),
            "candidate_budget_violation_count": int(
                generation.get("candidate_budget_violation_count", -1)
            ),
            "trajectory_replay_failure_count": replay_failures,
            "numerical_failure_count": int(
                generation.get("numerical_failure_count", -1)
            ),
            "maximum_forward_equivalents_per_source": maximum_compute,
            "full_cohort_generation_wall_time_seconds": _finite(
                equal_wall.get("full_cohort_generation_wall_time_seconds"),
                f"V4 full-cohort wall time {method}",
            ),
            "equal_wall_common_prefix_generation_wall_time_seconds": _finite(
                equal_wall.get("common_prefix_generation_wall_time_seconds"),
                f"V4 equal-wall prefix time {method}",
            ),
            "equal_wall_common_prefix_source_count": common_prefix_count,
            "peak_vram_mb": _finite(
                equal_wall.get("peak_vram_mb"), f"V4 peak VRAM {method}"
            ),
            "equal_wall_source_macro_ndcg": _finite(
                equal_wall.get("source_macro_ndcg"),
                f"V4 equal-wall NDCG {method}",
            ),
            "equal_wall_source_macro_normalized_regret": _finite(
                equal_wall.get("source_macro_normalized_regret"),
                f"V4 equal-wall regret {method}",
            ),
            "equal_wall_source_macro_top_1_recall": _finite(
                equal_wall.get("source_macro_top_1_recall"),
                f"V4 equal-wall top-1 {method}",
            ),
        }
        method_results[method] = {
            "schema_version": "route_a_v3_route2_xeditflow_matched_method_metrics.v4",
            "status": "XEDITFLOW_V4_MATCHED_METHOD_METRICS_COMPLETE",
            "method_role": method,
            "base_flow_training_seed": seed,
            "selected_combination": list(combination),
            "metrics": metrics,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
    reference = "full_soft_value_smc"
    _require(
        all(value == source_inventory[reference] for value in source_inventory.values())
        and all(
            set(value) == set(ndcg_by_method[reference])
            for value in ndcg_by_method.values()
        ),
        "V4 closed methods do not share exact measured source support",
    )
    full_ndcg = ndcg_by_method[reference]
    ndcg_cis: dict[str, list[float]] = {}
    for offset, (label, comparison) in enumerate(
        (
            ("over_unguided", "unguided_setflow"),
            ("over_strongest_baseline", "strongest_matched_baseline"),
        )
    ):
        common = sorted(full_ndcg)
        ndcg_cis[label] = source_paired_mean_bootstrap_ci_v4(
            [full_ndcg[key] - ndcg_by_method[comparison][key] for key in common],
            iterations=bootstrap_iterations,
            seed=int(bootstrap_seed) + offset,
        )
    evaluator_differences = full_independent_evaluator.get("per_source_paired_margin")
    _require(
        isinstance(evaluator_differences, Mapping)
        and len(evaluator_differences) >= 2,
        "V4 evaluator paired sources are absent",
    )
    evaluator_values = [
        _finite(value, "V4 per-source evaluator margin")
        for value in evaluator_differences.values()
    ]
    _require(
        math.isclose(
            _finite(
                full_independent_evaluator.get(
                    "paired_margin_over_strongest_baseline"
                ),
                "V4 evaluator point margin",
            ),
            float(np.mean(evaluator_values)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "V4 evaluator point margin is not the paired source mean",
    )
    evaluator_ci = source_paired_mean_bootstrap_ci_v4(
        evaluator_values,
        iterations=bootstrap_iterations,
        seed=int(bootstrap_seed) + 2,
    )
    full_self = _source_max_critic_score(
        full_candidate_rows, method="full_soft_value_smc", seed=seed
    )
    unguided_self = _source_max_critic_score(
        unguided_candidate_rows, method="unguided_setflow", seed=seed
    )
    common_self = sorted(set(full_self) & set(unguided_self))
    _require(bool(common_self), "V4 critic self-score source pairing is empty")
    critic_self_score_increased = float(
        np.mean([full_self[key] - unguided_self[key] for key in common_self])
    ) > 0.0
    bootstrap = {
        "schema_version": "route_a_v3_route2_xeditflow_source_paired_bootstrap.v4",
        "status": "XEDITFLOW_V4_SOURCE_PAIRED_BOOTSTRAP_COMPLETE",
        "base_flow_training_seed": seed,
        "selected_combination": list(combination),
        "analysis_unit": "SOURCE",
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": int(bootstrap_seed),
        "source_paired_ndcg_improvement_ci_95": ndcg_cis,
        "source_paired_independent_evaluator_margin_ci_95": evaluator_ci,
        "critic_self_score_increased": critic_self_score_increased,
        "all_methods_matched_compute_ceiling_met": all_compute_ok,
        "setflow_mode_is_fixed_trajectory_state": True,
        "free_action_ratio_head_used": False,
        "all_network_forwards_separately_charged": True,
        "matched_compute_schema": "MatchedComputeRecordV4",
        "closed_source_count": len(source_inventory[reference]),
        "defined_closed_source_count": len(ndcg_by_method[reference]),
        "closed_method_source_support_exactly_matched": True,
        "undefined_closed_sources_filled_with_zero": False,
        "independent_evaluator_in_gradient": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcome_reads": 0,
    }
    return {
        "method_results": method_results,
        "paired_bootstrap": bootstrap,
        "equal_wall_time_sensitivity": dict(equal_wall_time_sensitivity),
    }


def write_final_seed_evidence_v4(
    payload: Mapping[str, Any], *, output_dir: Path
) -> dict[str, Any]:
    _require(
        not output_dir.exists(), f"V4 final seed evidence exists: {output_dir}"
    )
    output_dir.mkdir(parents=True)
    method_paths: dict[str, str] = {}
    for method, result in payload["method_results"].items():
        path = output_dir / f"{method}.json"
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        method_paths[method] = str(path)
    bootstrap_path = output_dir / "paired_bootstrap.json"
    bootstrap_path.write_text(
        json.dumps(payload["paired_bootstrap"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    equal_wall_path = output_dir / "equal_wall_time_sensitivity.json"
    equal_wall_path.write_text(
        json.dumps(payload["equal_wall_time_sensitivity"], indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    row = {
        "base_flow_training_seed": int(
            payload["paired_bootstrap"]["base_flow_training_seed"]
        ),
        "methods": method_paths,
        "paired_bootstrap_path": str(bootstrap_path),
        "equal_wall_time_sensitivity_path": str(equal_wall_path),
    }
    (output_dir / "seed_manifest_row.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    config = _json(arguments.config)
    _require(
        config.get("schema_version")
        == "route_a_v3_route2_xeditflow_final_seed_evidence_config.v4",
        "unexpected V4 final seed evidence config",
    )
    evidence: dict[str, dict[str, Any]] = {}
    for method, paths in config["methods"].items():
        bundle = {
            "closed": _json(Path(paths["closed_summary_path"])),
            "open": _json(Path(paths["open_summary_path"])),
            "generation": _json(Path(paths["generation_summary_path"])),
        }
        if "terminal_critic_summary_path" in paths:
            bundle["terminal_critic"] = _json(
                Path(paths["terminal_critic_summary_path"])
            )
        evidence[method] = bundle
    payload = assemble_final_seed_evidence_v4(
        evidence,
        base_flow_training_seed=int(config["base_flow_training_seed"]),
        selected_combination=config["selected_combination"],
        equal_wall_time_sensitivity=_json(
            Path(config["equal_wall_time_sensitivity_path"])
        ),
        full_independent_evaluator=_json(
            Path(config["full_independent_evaluator_path"])
        ),
        full_candidate_rows=_jsonl(Path(config["full_candidate_path"])),
        unguided_candidate_rows=_jsonl(Path(config["unguided_candidate_path"])),
        bootstrap_iterations=int(config["bootstrap_iterations"]),
        bootstrap_seed=int(config["bootstrap_seed"]),
    )
    result = write_final_seed_evidence_v4(payload, output_dir=arguments.output_dir)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
