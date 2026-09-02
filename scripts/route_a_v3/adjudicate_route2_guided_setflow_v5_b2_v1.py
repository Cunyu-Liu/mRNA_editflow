#!/usr/bin/env python3
"""Adjudicate Gate B2 for guided vs unguided SetFlow V5 generation.

Consumes the two candidate files produced by
run_route2_guided_xeditsetflow_v5_v1.py (or the terminal screen validation
trajectories as the unguided arm), computes per-arm source-macro recovery,
measured top-k recovery, and hit@1 via the frozen
measured_neighborhood_metrics, then runs a source-group paired cluster
bootstrap (2,000 iterations, seed 20260816) on Delta recovery and Delta hit@1.

Gate B2 passes when the guided recovery improvement is at least +0.05 with a
bootstrap CI that does not cross zero and hit@1 does not degrade.  Gate B3 is
reported as guided recovery >= 0.35.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.route_a_v3.evaluate_route2_generation_v1 import (
    evaluate_generation,
    load_source_manifest,
    measured_neighborhood_metrics,
    validate_measured_pool,
)

B2_SCHEMA = "route_a_v3_route2_guided_setflow_v5_b2_adjudication.v1"
BOOTSTRAP_ITERATIONS = 2_000
BOOTSTRAP_SEED = 20_260_816
RECOVERY_IMPROVEMENT_MINIMUM = 0.05
GUIDED_RECOVERY_B3_MINIMUM = 0.35
MEASURED_TOP_K = 10


class GuidedB2AdjudicationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GuidedB2AdjudicationError(message)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                _require(isinstance(row, dict), f"JSONL row is not an object: {path}")
                rows.append(row)
    _require(bool(rows), f"JSONL input is empty: {path}")
    return rows


def _measured_pool_by_source(
    measured_rows: Sequence[Mapping[str, Any]],
) -> dict[str, set[str]]:
    pools: dict[str, set[str]] = defaultdict(set)
    for row in measured_rows:
        pools[str(row["source_key"])].add(
            str(row["candidate_sequence"]).upper().replace("T", "U")
        )
    return dict(pools)


def per_source_arm_metrics(
    manifest: Mapping[str, Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    measured_rows: Sequence[Mapping[str, Any]],
    measured_pools: Mapping[str, set[str]],
) -> dict[str, dict[str, Any]]:
    """Per-source recovery, top-k recovery, and hit@1 for one arm."""

    methods = {str(row["method_id"]) for row in candidates}
    _require(len(methods) == 1, "one arm must contain exactly one method_id")
    at_k = measured_neighborhood_metrics(
        manifest,
        list(candidates),
        list(measured_rows),
        k=MEASURED_TOP_K,
        candidate_support_mode="OPEN_GENERATED_SUPPORT",
    )
    at_one = measured_neighborhood_metrics(
        manifest,
        list(candidates),
        list(measured_rows),
        k=1,
        candidate_support_mode="OPEN_GENERATED_SUPPORT",
    )
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_source[str(row["source_key"])].append(row)
    per_source: dict[str, dict[str, Any]] = {}
    for source_key in sorted(manifest):
        rows = by_source[source_key]
        _require(bool(rows), f"arm has no candidates for source: {source_key}")
        ranked: dict[str, float] = {}
        for row in rows:
            sequence = str(row["candidate_sequence"]).upper().replace("T", "U")
            score = float(row["generation_score"])
            ranked[sequence] = max(ranked.get(sequence, -math.inf), score)
        sequences = sorted(ranked)
        scores = np.asarray([ranked[sequence] for sequence in sequences], dtype=float)
        top_score = float(np.max(scores))
        top_block_share = 1.0 / sum(1 for score in scores if score == top_score)
        measured_keys = measured_pools.get(source_key, set())
        top1_generated_is_measured = sum(
            top_block_share
            for sequence, score in zip(sequences, scores)
            if score == top_score and sequence in measured_keys
        )
        pool = at_k["per_source"][source_key]
        per_source[source_key] = {
            "candidate_recovery_rate": pool["candidate_recovery_rate"],
            "measured_top_k_recovery_at_k": pool["measured_top_k_recovery_at_k"],
            "hit_at_1": at_one["per_source"][source_key][
                "measured_top_k_recovery_at_k"
            ],
            "top1_generated_is_measured": top1_generated_is_measured,
        }
    return per_source


def source_group_bootstrap(
    group_keys: dict[str, list[str]],
    guided: Mapping[str, Mapping[str, float]],
    unguided: Mapping[str, Mapping[str, float]],
    metric: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    group_ids = sorted(group_keys)
    deltas = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled = rng.integers(0, len(group_ids), size=len(group_ids))
        guided_values: list[float] = []
        unguided_values: list[float] = []
        for group_index in sampled:
            for source_key in group_keys[group_ids[group_index]]:
                guided_values.append(float(guided[source_key][metric]))
                unguided_values.append(float(unguided[source_key][metric]))
        deltas[iteration] = float(np.mean(guided_values)) - float(
            np.mean(unguided_values)
        )
    point_estimate = float(
        np.mean([guided[key][metric] for key in guided])
        - np.mean([unguided[key][metric] for key in unguided])
    )
    return {
        "iterations": iterations,
        "seed": seed,
        "metric": metric,
        "delta_point_estimate": point_estimate,
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "bootstrap_delta_mean": float(np.mean(deltas)),
    }


def adjudicate(
    unguided_candidates: Sequence[Mapping[str, Any]],
    guided_candidates: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Mapping[str, Any]],
    measured_rows: Sequence[Mapping[str, Any]],
    source_group_by_key: Mapping[str, str],
    legality: Mapping[str, Mapping[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    unguided_sources = {str(row["source_key"]) for row in unguided_candidates}
    guided_sources = {str(row["source_key"]) for row in guided_candidates}
    _require(
        unguided_sources == guided_sources == set(manifest),
        "guided and unguided arms do not cover the exact same source cohort",
    )
    measured_pools = _measured_pool_by_source(measured_rows)
    unguided_metrics = per_source_arm_metrics(
        manifest, unguided_candidates, measured_rows, measured_pools
    )
    guided_metrics = per_source_arm_metrics(
        manifest, guided_candidates, measured_rows, measured_pools
    )
    group_keys: dict[str, list[str]] = defaultdict(list)
    for source_key in sorted(manifest):
        group_keys[str(source_group_by_key[source_key])].append(source_key)
    _require(bool(group_keys), "source-group mapping is empty")
    delta_recovery = source_group_bootstrap(
        group_keys,
        guided_metrics,
        unguided_metrics,
        "candidate_recovery_rate",
        iterations=iterations,
        seed=seed,
    )
    delta_hit_at_1 = source_group_bootstrap(
        group_keys,
        guided_metrics,
        unguided_metrics,
        "hit_at_1",
        iterations=iterations,
        seed=seed + 1,
    )
    unguided_recovery = float(
        np.mean([row["candidate_recovery_rate"] for row in unguided_metrics.values()])
    )
    guided_recovery = float(
        np.mean([row["candidate_recovery_rate"] for row in guided_metrics.values()])
    )
    unguided_hit_at_1 = float(
        np.mean([row["hit_at_1"] for row in unguided_metrics.values()])
    )
    guided_hit_at_1 = float(
        np.mean([row["hit_at_1"] for row in guided_metrics.values()])
    )
    unguided_top_k = float(
        np.mean(
            [row["measured_top_k_recovery_at_k"] for row in unguided_metrics.values()]
        )
    )
    guided_top_k = float(
        np.mean(
            [row["measured_top_k_recovery_at_k"] for row in guided_metrics.values()]
        )
    )
    recovery_criterion = (
        delta_recovery["delta_point_estimate"] >= RECOVERY_IMPROVEMENT_MINIMUM
        and delta_recovery["ci_low"] > 0.0
    )
    hit_at_1_non_degradation = delta_hit_at_1["delta_point_estimate"] >= 0.0
    legality_ok = all(
        bool(arm.get("hard_legality_rate") == 1.0)
        and int(arm.get("edit_budget_violation_count", -1)) == 0
        and int(arm.get("candidate_budget_violation_count", -1)) == 0
        for arm in legality.values()
    )
    _require(bool(legality), "legality evidence is empty")
    return {
        "schema_version": B2_SCHEMA,
        "status": "GUIDED_SETFLOW_V5_B2_ADJUDICATION_COMPLETE",
        "gate_b2_rule": (
            "GUIDED_RECOVERY_IMPROVEMENT_GE_0.05_WITH_CI_NOT_CROSSING_ZERO_"
            "AND_HIT_AT_1_NOT_DEGRADED"
        ),
        "gate_b3_rule": "GUIDED_RECOVERY_GE_0.35",
        "bootstrap": {
            "iterations": iterations,
            "seed": seed,
            "unit": "SOURCE_GROUP_PAIRED_CLUSTER",
            "group_count": len(group_keys),
            "source_count": len(manifest),
        },
        "metric_definitions": {
            "recovery": (
                "source_macro_candidate_recovery_rate at k=10 "
                "(OPEN_GENERATED_SUPPORT)"
            ),
            "top_k_recovery": "source_macro_measured_top_k_recovery_at_k at k=10",
            "hit_at_1": "source_macro_measured_top_k_recovery_at_k evaluated at k=1",
            "top1_generated_is_measured": (
                "tie-aware probability that the top-ranked generated candidate is "
                "a measured neighborhood member (diagnostic, not gated)"
            ),
        },
        "unguided": {
            "method_id": str(unguided_candidates[0]["method_id"]),
            "source_macro_candidate_recovery_rate": unguided_recovery,
            "source_macro_measured_top_k_recovery_at_k": unguided_top_k,
            "hit_at_1": unguided_hit_at_1,
        },
        "guided": {
            "method_id": str(guided_candidates[0]["method_id"]),
            "source_macro_candidate_recovery_rate": guided_recovery,
            "source_macro_measured_top_k_recovery_at_k": guided_top_k,
            "hit_at_1": guided_hit_at_1,
        },
        "delta_recovery": delta_recovery,
        "delta_hit_at_1": delta_hit_at_1,
        "gate_b2_recovery_improvement_minimum": RECOVERY_IMPROVEMENT_MINIMUM,
        "gate_b2_recovery_ci_not_crossing_zero": bool(delta_recovery["ci_low"] > 0.0),
        "gate_b2_recovery_criterion_passed": bool(recovery_criterion),
        "gate_b2_hit_at_1_non_degradation_passed": bool(hit_at_1_non_degradation),
        "legality": dict(legality),
        "legality_ok": bool(legality_ok),
        "gate_b2_passed": bool(
            recovery_criterion and hit_at_1_non_degradation and legality_ok
        ),
        "gate_b3_passed": bool(guided_recovery >= GUIDED_RECOVERY_B3_MINIMUM),
        "gate_b3_guided_recovery_minimum": GUIDED_RECOVERY_B3_MINIMUM,
        "per_source": {
            source_key: {
                "unguided": unguided_metrics[source_key],
                "guided": guided_metrics[source_key],
            }
            for source_key in sorted(manifest)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unguided-candidates", required=True, type=Path)
    parser.add_argument("--guided-candidates", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--measured-neighborhood", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--unguided-arm-summary",
        type=Path,
        default=None,
        help="optional arm summary JSON carrying legality counters",
    )
    parser.add_argument(
        "--guided-arm-summary",
        type=Path,
        default=None,
        help="optional arm summary JSON carrying legality counters",
    )
    parser.add_argument(
        "--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS
    )
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise GuidedB2AdjudicationError(
            f"terminal B2 adjudication already exists: {arguments.output}"
        )
    unguided_candidates = _read_jsonl(arguments.unguided_candidates)
    guided_candidates = _read_jsonl(arguments.guided_candidates)
    measured_rows = _read_jsonl(arguments.measured_neighborhood)
    validate_measured_pool(measured_rows, "DEVELOPMENT", "CLOSED")
    source_rows = _read_jsonl(arguments.source_manifest)
    source_group_by_key = {
        str(row["source_key"]): str(row["source_id"]) for row in source_rows
    }
    full_manifest = load_source_manifest(arguments.source_manifest)
    covered = {str(row["source_key"]) for row in unguided_candidates}
    manifest = {key: full_manifest[key] for key in sorted(covered)}
    measured_rows = [
        row for row in measured_rows if str(row["source_key"]) in covered
    ]
    legality: dict[str, Any] = {}
    for arm_name, summary_path, candidates in (
        ("unguided", arguments.unguided_arm_summary, unguided_candidates),
        ("guided", arguments.guided_arm_summary, guided_candidates),
    ):
        if summary_path is not None:
            summary = _read_json(summary_path)
            legality[arm_name] = {
                "hard_legality_rate": summary.get("hard_legality_rate"),
                "edit_budget_violation_count": summary.get(
                    "edit_budget_violation_count"
                ),
                "candidate_budget_violation_count": summary.get(
                    "candidate_budget_violation_count"
                ),
            }
        else:
            generation = evaluate_generation(manifest, list(candidates))
            legality[arm_name] = {
                "hard_legality_rate": generation["hard_legality_rate"],
                "edit_budget_violation_count": generation[
                    "edit_budget_violation_count"
                ],
                "candidate_budget_violation_count": generation[
                    "candidate_budget_violation_count"
                ],
            }
    result = adjudicate(
        unguided_candidates,
        guided_candidates,
        manifest,
        measured_rows,
        source_group_by_key,
        legality,
        iterations=int(arguments.bootstrap_iterations),
        seed=int(arguments.bootstrap_seed),
    )
    result["inputs"] = {
        "unguided_candidates": str(arguments.unguided_candidates),
        "guided_candidates": str(arguments.guided_candidates),
        "source_manifest": str(arguments.source_manifest),
        "measured_neighborhood": str(arguments.measured_neighborhood),
        "unguided_arm_summary": (
            None
            if arguments.unguided_arm_summary is None
            else str(arguments.unguided_arm_summary)
        ),
        "guided_arm_summary": (
            None
            if arguments.guided_arm_summary is None
            else str(arguments.guided_arm_summary)
        ),
    }
    result["evaluation_outcome_reads"] = 0
    result["new_final_evaluation_outcome_reads"] = 0
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "gate_b2_passed",
                    "gate_b3_passed",
                    "delta_recovery",
                    "delta_hit_at_1",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
