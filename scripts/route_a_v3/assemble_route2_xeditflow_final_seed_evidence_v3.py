#!/usr/bin/env python3
"""Assemble one frozen seed's six-method metrics and paired bootstraps."""

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


METHODS = {
    "full_soft_value_smc",
    "unguided_setflow",
    "first_order_guidance",
    "simple_rate_guidance",
    "generate_then_rerank",
    "strongest_matched_baseline",
}
SEEDS = (20260904, 20260905, 20260906)
GENERATION_COMPLETE_STATUSES = {
    "XEDITFLOW_V3_SMC_GENERATION_COMPLETE",
    "XEDITFLOW_V3_MATCHED_CONTROL_GENERATION_COMPLETE",
    "XEDITFLOW_V3_STRONGEST_BASELINE_ADAPTER_COMPLETE",
}


class XEditFlowFinalSeedEvidenceV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowFinalSeedEvidenceV3Error(message)


def _finite(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    result = float(value)
    _require(math.isfinite(result), f"{label} is nonfinite")
    return result


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(bool(rows) and all(isinstance(row, dict) for row in rows), f"JSONL input is empty or invalid: {path}")
    return rows


def source_paired_mean_bootstrap_ci_v3(
    differences: Sequence[float],
    *,
    iterations: int,
    seed: int,
) -> list[float]:
    values = np.asarray(differences, dtype=np.float64)
    _require(values.ndim == 1 and len(values) >= 2 and bool(np.isfinite(values).all()), "final paired differences are invalid")
    _require(iterations == 10_000, "final paired bootstrap count changed")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _require_unprotected(payload: Mapping[str, Any], label: str) -> None:
    _require(
        payload.get("development_test_outcomes_accessed") is False
        and payload.get("new_final_evaluation_outcomes_accessed") is False,
        f"final seed evidence accessed protected outcome: {label}",
    )


def _defined_ndcg_by_source(closed: Mapping[str, Any], label: str) -> dict[str, float]:
    _require(closed.get("status") == "XEDITFLOW_V3_CLOSED_NEIGHBORHOOD_COMPLETE", f"closed evidence is incomplete: {label}")
    _require(closed.get("undefined_sources_are_not_filled_with_zero") is True, f"closed undefined policy differs: {label}")
    rows = closed.get("per_source")
    _require(isinstance(rows, Mapping) and bool(rows), f"closed per-source evidence is absent: {label}")
    result = {}
    for source_key, row in rows.items():
        if row.get("status") == "DEFINED":
            result[str(source_key)] = _finite(row.get("ndcg"), f"closed source NDCG {label}/{source_key}")
        else:
            _require(row.get("ndcg") is None, f"undefined closed source was assigned NDCG: {label}/{source_key}")
    _require(bool(result), f"closed method has no defined source: {label}")
    return result


def _source_max_critic_score(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = _finite(row.get("critic_self_score"), f"critic self score {label}")
        grouped.setdefault(str(row["source_key"]), []).append(value)
    _require(bool(grouped), f"critic diagnostic candidate rows are empty: {label}")
    return {key: max(values) for key, values in grouped.items()}


def assemble_final_seed_evidence_v3(
    evidence: Mapping[str, Mapping[str, Any]],
    *,
    base_flow_training_seed: int,
    equal_wall_time_sensitivity: Mapping[str, Any],
    full_independent_evaluator: Mapping[str, Any],
    full_candidate_rows: Sequence[Mapping[str, Any]],
    unguided_candidate_rows: Sequence[Mapping[str, Any]],
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    _require(base_flow_training_seed in SEEDS, "final evidence base-flow seed differs")
    _require(set(evidence) == METHODS, "final evidence method inventory differs")
    _require(
        equal_wall_time_sensitivity.get("status")
        == "XEDITFLOW_V3_EQUAL_WALL_TIME_SENSITIVITY_COMPLETE"
        and int(equal_wall_time_sensitivity.get("base_flow_training_seed", -1))
        == base_flow_training_seed
        and isinstance(equal_wall_time_sensitivity.get("methods"), Mapping)
        and set(equal_wall_time_sensitivity["methods"]) == METHODS,
        "final equal-wall-time sensitivity evidence is incomplete",
    )
    _require_unprotected(equal_wall_time_sensitivity, "equal-wall-time sensitivity")
    common_prefix_count = int(
        equal_wall_time_sensitivity.get("common_source_prefix_count", -1)
    )
    _require(2 <= common_prefix_count <= 891, "final equal-wall common prefix differs")
    _require(
        full_independent_evaluator.get("status") == "XEDITFLOW_V3_INDEPENDENT_EVALUATOR_COMPARISON_COMPLETE"
        and full_independent_evaluator.get("analysis_unit") == "SOURCE",
        "final independent-evaluator evidence is incomplete",
    )
    _require_unprotected(full_independent_evaluator, "full independent evaluator")
    method_results = {}
    ndcg_by_method = {}
    closed_source_inventory_by_method = {}
    all_compute_ok = True
    for method in sorted(METHODS):
        bundle = evidence[method]
        _require(set(bundle) == {"closed", "open", "generation"}, f"final method bundle differs: {method}")
        closed = bundle["closed"]
        opened = bundle["open"]
        generation = bundle["generation"]
        for label, payload in (("closed", closed), ("open", opened), ("generation", generation)):
            _require_unprotected(payload, f"{method}/{label}")
        _require(opened.get("status") == "XEDITFLOW_V3_OPEN_GENERATION_METRICS_COMPLETE", f"open evidence is incomplete: {method}")
        _require(generation.get("status") in GENERATION_COMPLETE_STATUSES, f"generation evidence is incomplete: {method}")
        _require(
            str(closed.get("method_id")) == str(opened.get("method_id")) == method,
            f"final method role differs: {method}",
        )
        _require(str(generation.get("method_id")) == method, f"final generation role differs: {method}")
        observed_seed = int(generation.get("base_flow_training_seed", -1))
        _require(observed_seed == base_flow_training_seed, f"final method seed differs: {method}")
        ndcg_by_method[method] = _defined_ndcg_by_source(closed, method)
        closed_source_inventory_by_method[method] = set(
            str(source_key) for source_key in closed["per_source"]
        )
        closed_macro_ndcg = _finite(
            closed.get("source_macro_ndcg"), f"closed macro NDCG {method}"
        )
        _require(
            math.isclose(
                closed_macro_ndcg,
                float(np.mean(list(ndcg_by_method[method].values()))),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"closed macro NDCG does not equal defined-source mean: {method}",
        )
        maximum_compute = int(generation.get("maximum_forward_equivalents_per_source", -1))
        compute_ok = 0 <= maximum_compute <= 320
        all_compute_ok = all_compute_ok and compute_ok
        equal_wall = equal_wall_time_sensitivity["methods"][method]
        _require(
            isinstance(equal_wall, Mapping)
            and "A100" in str(equal_wall.get("accelerator_name", "")).upper(),
            f"final equal-wall accelerator differs: {method}",
        )
        metrics = {
            "closed_source_macro_ndcg": closed_macro_ndcg,
            "closed_source_macro_normalized_regret": _finite(closed.get("source_macro_normalized_regret"), f"closed macro regret {method}"),
            "closed_source_macro_top_1_recall": _finite(closed.get("source_macro_top_1_recall"), f"closed macro top-1 {method}"),
            "open_source_macro_candidate_recovery": _finite(opened.get("source_macro_candidate_recovery"), f"open recovery {method}"),
            "open_source_macro_top_k_recovery": _finite(opened.get("source_macro_top_k_recovery"), f"open top-k {method}"),
            "open_source_macro_unique_candidate_rate": _finite(opened.get("source_macro_unique_candidate_rate"), f"open unique {method}"),
            "independent_evaluator_margin_over_strongest_baseline": (
                _finite(full_independent_evaluator.get("paired_margin_over_strongest_baseline"), "full evaluator margin")
                if method == "full_soft_value_smc"
                else 0.0
            ),
            "hard_legality_rate": _finite(opened.get("hard_legality_rate"), f"hard legality {method}"),
            "edit_budget_violation_count": int(generation.get("edit_budget_violation_count", -1)),
            "candidate_budget_violation_count": int(generation.get("candidate_budget_violation_count", -1)),
            "trajectory_replay_failure_count": int(generation.get("trajectory_replay_failure_count", -1)),
            "numerical_failure_count": int(generation.get("numerical_failure_count", -1)),
            "maximum_forward_equivalents_per_source": maximum_compute,
            "full_cohort_generation_wall_time_seconds": _finite(
                equal_wall.get("full_cohort_generation_wall_time_seconds"),
                f"full-cohort wall time {method}",
            ),
            "equal_wall_common_prefix_generation_wall_time_seconds": _finite(
                equal_wall.get("common_prefix_generation_wall_time_seconds"),
                f"common-prefix wall time {method}",
            ),
            "equal_wall_common_prefix_source_count": common_prefix_count,
            "peak_vram_mb": _finite(
                equal_wall.get("peak_vram_mb"), f"peak VRAM {method}"
            ),
            "equal_wall_source_macro_ndcg": _finite(
                equal_wall.get("source_macro_ndcg"),
                f"equal-wall NDCG {method}",
            ),
            "equal_wall_source_macro_normalized_regret": _finite(
                equal_wall.get("source_macro_normalized_regret"),
                f"equal-wall regret {method}",
            ),
            "equal_wall_source_macro_top_1_recall": _finite(
                equal_wall.get("source_macro_top_1_recall"),
                f"equal-wall top-1 {method}",
            ),
        }
        method_results[method] = {
            "schema_version": "route_a_v3_route2_xeditflow_matched_method_metrics.v3",
            "status": "XEDITFLOW_V3_MATCHED_METHOD_METRICS_COMPLETE",
            "method_role": method,
            "base_flow_training_seed": base_flow_training_seed,
            "metrics": metrics,
            "development_test_outcomes_accessed": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
    reference_method = "full_soft_value_smc"
    _require(
        all(
            inventory == closed_source_inventory_by_method[reference_method]
            for inventory in closed_source_inventory_by_method.values()
        )
        and all(
            set(values) == set(ndcg_by_method[reference_method])
            for values in ndcg_by_method.values()
        ),
        "closed methods do not share the exact measured source support",
    )
    full_ndcg = ndcg_by_method["full_soft_value_smc"]
    ndcg_cis = {}
    for label, other_method in (
        ("over_unguided", "unguided_setflow"),
        ("over_strongest_baseline", "strongest_matched_baseline"),
    ):
        common = sorted(full_ndcg)
        _require(len(common) >= 2, f"paired closed source support is too small: {label}")
        differences = [full_ndcg[key] - ndcg_by_method[other_method][key] for key in common]
        ndcg_cis[label] = source_paired_mean_bootstrap_ci_v3(
            differences,
            iterations=bootstrap_iterations,
            seed=int(bootstrap_seed) + (0 if label == "over_unguided" else 1),
        )
    evaluator_differences = full_independent_evaluator.get("per_source_paired_margin")
    _require(isinstance(evaluator_differences, Mapping) and len(evaluator_differences) >= 2, "final evaluator paired sources are absent")
    evaluator_values = [
        _finite(value, "per-source evaluator margin")
        for value in evaluator_differences.values()
    ]
    _require(
        math.isclose(
            _finite(
                full_independent_evaluator.get("paired_margin_over_strongest_baseline"),
                "full evaluator margin",
            ),
            float(np.mean(evaluator_values)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "independent-evaluator point margin does not equal the paired source mean",
    )
    evaluator_ci = source_paired_mean_bootstrap_ci_v3(
        evaluator_values,
        iterations=bootstrap_iterations,
        seed=int(bootstrap_seed) + 2,
    )
    full_self = _source_max_critic_score(full_candidate_rows, "full")
    unguided_self = _source_max_critic_score(unguided_candidate_rows, "unguided")
    common_self = sorted(set(full_self) & set(unguided_self))
    _require(bool(common_self), "critic self-score source pairing is empty")
    critic_self_score_increased = float(
        np.mean([full_self[key] - unguided_self[key] for key in common_self])
    ) > 0.0
    bootstrap = {
        "schema_version": "route_a_v3_route2_xeditflow_source_paired_bootstrap.v3",
        "status": "XEDITFLOW_V3_SOURCE_PAIRED_BOOTSTRAP_COMPLETE",
        "base_flow_training_seed": base_flow_training_seed,
        "analysis_unit": "SOURCE",
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": int(bootstrap_seed),
        "source_paired_ndcg_improvement_ci_95": ndcg_cis,
        "source_paired_independent_evaluator_margin_ci_95": evaluator_ci,
        "critic_self_score_increased": critic_self_score_increased,
        "all_methods_matched_compute_ceiling_met": all_compute_ok,
        "closed_source_count": len(closed_source_inventory_by_method[reference_method]),
        "defined_closed_source_count": len(ndcg_by_method[reference_method]),
        "closed_method_source_support_exactly_matched": True,
        "undefined_closed_sources_filled_with_zero": False,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    return {
        "method_results": method_results,
        "paired_bootstrap": bootstrap,
        "equal_wall_time_sensitivity": dict(equal_wall_time_sensitivity),
    }


def write_final_seed_evidence_v3(
    payload: Mapping[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    _require(not output_dir.exists(), f"final seed evidence output exists: {output_dir}")
    output_dir.mkdir(parents=True)
    method_paths = {}
    for method, result in payload["method_results"].items():
        path = output_dir / f"{method}.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        method_paths[method] = str(path)
    bootstrap_path = output_dir / "paired_bootstrap.json"
    bootstrap_path.write_text(json.dumps(payload["paired_bootstrap"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    equal_wall_path = output_dir / "equal_wall_time_sensitivity.json"
    equal_wall_path.write_text(
        json.dumps(payload["equal_wall_time_sensitivity"], indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    manifest_row = {
        "base_flow_training_seed": int(payload["paired_bootstrap"]["base_flow_training_seed"]),
        "methods": method_paths,
        "paired_bootstrap_path": str(bootstrap_path),
        "equal_wall_time_sensitivity_path": str(equal_wall_path),
    }
    (output_dir / "seed_manifest_row.json").write_text(json.dumps(manifest_row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = _json(args.config)
    _require(config.get("schema_version") == "route_a_v3_route2_xeditflow_final_seed_evidence_config.v1", "unexpected final seed evidence config")
    evidence = {
        method: {
            "closed": _json(Path(paths["closed_summary_path"])),
            "open": _json(Path(paths["open_summary_path"])),
            "generation": _json(Path(paths["generation_summary_path"])),
        }
        for method, paths in config["methods"].items()
    }
    payload = assemble_final_seed_evidence_v3(
        evidence,
        base_flow_training_seed=int(config["base_flow_training_seed"]),
        equal_wall_time_sensitivity=_json(
            Path(config["equal_wall_time_sensitivity_path"])
        ),
        full_independent_evaluator=_json(Path(config["full_independent_evaluator_path"])),
        full_candidate_rows=_jsonl(Path(config["full_candidate_path"])),
        unguided_candidate_rows=_jsonl(Path(config["unguided_candidate_path"])),
        bootstrap_iterations=int(config["bootstrap_iterations"]),
        bootstrap_seed=int(config["bootstrap_seed"]),
    )
    result = write_final_seed_evidence_v3(payload, output_dir=args.output_dir)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
