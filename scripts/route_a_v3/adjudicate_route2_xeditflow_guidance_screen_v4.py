#!/usr/bin/env python3
"""Assemble and freeze exactly one of the 18 V4 guidance combinations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditflow_gate_v4 import (
    GUIDANCE_GRID_V4,
    adjudicate_guidance_screen_v4,
)
from core.route2_xeditflow_value_training_v4 import (
    validate_value_training_provenance_v4,
)


class GuidanceScreenAdjudicationV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GuidanceScreenAdjudicationV4Error(message)


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


def _identity(payload: Mapping[str, Any]) -> tuple[str, int, tuple[float, ...]]:
    return (
        str(payload.get("method_id")),
        int(payload.get("base_flow_training_seed", -1)),
        tuple(
            float(payload.get(key, -1))
            for key in ("kappa", "temperature", "beta_max")
        ),
    )


def _protected(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("development_test_outcomes_accessed_after_atomic_test")
        is False
        and (
            payload.get("new_final_evaluation_outcomes_accessed") is False
            or int(payload.get("new_final_evaluation_outcome_reads", -1)) == 0
        )
    )


def assemble_screen_results_v4(
    manifest: Mapping[str, Any],
) -> dict[tuple[float, float, float], dict[str, Any]]:
    _require(
        manifest.get("schema_version")
        == "route_a_v3_route2_xeditflow_v4_value_config_manifest.v1"
        and manifest.get("status")
        == "XEDITFLOW_V4_VALUE_CONFIGS_PREPARED_NOT_STARTED",
        "unexpected or incomplete V4 guidance manifest",
    )
    paths = manifest.get("guidance_result_paths")
    _require(
        isinstance(paths, list) and len(paths) == 18,
        "V4 guidance manifest must expose exactly 18 result chains",
    )
    results: dict[tuple[float, float, float], dict[str, Any]] = {}
    for row in paths:
        combination = tuple(float(value) for value in row.get("combination", ()))
        _require(
            combination in GUIDANCE_GRID_V4 and combination not in results,
            "V4 guidance result combination differs or is duplicated",
        )
        smc = _json(Path(str(row["smc_summary_path"])))
        critic = _json(Path(str(row["critic_summary_path"])))
        closed = _json(Path(str(row["closed_summary_path"])))
        open_metrics = _json(Path(str(row["open_metric_path"])))
        evaluator_summary = _json(
            Path(str(row["independent_evaluator_summary_path"]))
        )
        evaluator = _json(Path(str(row["independent_evaluator_metric_path"])))
        _require(
            smc.get("status")
            == "XEDITFLOW_V4_SMC_GENERATION_COMPLETE_PENDING_TERMINAL_CRITIC_SCORING"
            and critic.get("status")
            == "XEDITFLOW_V4_CANDIDATE_CRITIC_SCORING_COMPLETE"
            and closed.get("status")
            == "XEDITFLOW_V4_CLOSED_NEIGHBORHOOD_COMPLETE"
            and open_metrics.get("status")
            == "XEDITFLOW_V4_OPEN_GENERATION_METRICS_COMPLETE"
            and evaluator_summary.get("status")
            == "FROZEN_INDEPENDENT_EVALUATOR_SCORING_COMPLETE"
            and evaluator.get("status")
            == "XEDITFLOW_V4_INDEPENDENT_EVALUATOR_COMPARISON_COMPLETE",
            f"V4 guidance result chain is incomplete: {combination}",
        )
        expected_identity = (
            f"xeditflow_v4_guidance_screen_{row['combination_id']}",
            20260912,
            combination,
        )
        _require(
            all(
                _identity(payload) == expected_identity
                for payload in (smc, critic, closed, open_metrics)
            )
            and str(evaluator.get("method_id")) == expected_identity[0]
            and int(evaluator.get("base_flow_training_seed", -1)) == 20260912
            and tuple(float(value) for value in evaluator.get("combination", ()))
            == combination,
            f"V4 guidance result identity differs: {combination}",
        )
        value_checkpoint_path = str(row.get("value_checkpoint_path", ""))
        _require(
            smc.get("value_checkpoint_path") == value_checkpoint_path,
            f"V4 guidance value checkpoint path differs: {combination}",
        )
        value_training_provenance = validate_value_training_provenance_v4(
            smc.get("value_training_provenance", {}),
            base_flow_training_seed=20260912,
            value_checkpoint_path=value_checkpoint_path,
        )
        _require(
            all(
                _protected(payload)
                for payload in (
                    smc,
                    critic,
                    closed,
                    open_metrics,
                    evaluator_summary,
                    evaluator,
                )
            )
            and evaluator_summary.get(
                "evaluation_outcomes_used_to_select_evaluator"
            )
            == 0
            and evaluator_summary.get("independent_evaluator_in_gradient")
            is False
            and evaluator.get("independent_evaluator_used_for_gradient") is False,
            f"V4 guidance screen accessed a protected outcome or evaluator gradient: {combination}",
        )
        compute_rows = _jsonl(Path(str(row["matched_compute_path"])))
        _require(
            len(compute_rows) == int(critic.get("source_count", -1)) == 891,
            f"V4 matched-compute source coverage differs: {combination}",
        )
        _require(
            all(
                int(compute["total_forward_equivalents"]) <= 320
                and len(compute.get("critic_forwards_by_member", ())) == 3
                and compute.get("all_network_forwards_separately_charged")
                is True
                and compute.get("terminal_critic_reservation_reconciled") is True
                and all(int(value) == 0 for value in compute["failure_counters"].values())
                for compute in compute_rows
            ),
            f"V4 network compute was not separately reconciled or failed: {combination}",
        )
        maximum_compute = max(
            int(compute["total_forward_equivalents"]) for compute in compute_rows
        )
        _require(
            maximum_compute
            == int(critic.get("maximum_total_forward_equivalents_per_source", -1))
            and int(critic.get("forward_equivalent_ceiling_per_source", -1))
            == 320,
            f"V4 matched-compute summary differs: {combination}",
        )
        results[combination] = {
            "status": "XEDITFLOW_V4_GUIDANCE_SCREEN_COMBINATION_COMPLETE",
            "base_flow_training_seed": 20260912,
            "combination": list(combination),
            "closed_source_macro_ndcg": closed["source_macro_ndcg"],
            "closed_source_macro_normalized_regret": closed[
                "source_macro_normalized_regret"
            ],
            "independent_evaluator_paired_margin": evaluator[
                "paired_margin_over_strongest_baseline"
            ],
            "open_source_macro_candidate_recovery": open_metrics[
                "source_macro_candidate_recovery"
            ],
            "total_forward_equivalents": maximum_compute,
            "value_checkpoint_path": value_checkpoint_path,
            "value_training_provenance": value_training_provenance,
            "setflow_mode_is_fixed_trajectory_state": smc[
                "setflow_mode_is_fixed_trajectory_state"
            ],
            "free_action_ratio_head_used": smc["free_action_ratio_head_used"],
            "all_network_forwards_separately_charged": True,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcome_reads": 0,
        }
    _require(
        set(results) == set(GUIDANCE_GRID_V4),
        "V4 guidance result grid differs",
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(
        not args.output.exists(),
        f"terminal V4 guidance adjudication exists: {args.output}",
    )
    results = assemble_screen_results_v4(_json(args.manifest))
    gate = adjudicate_guidance_screen_v4(results)
    selected = (
        float(gate["selected_kappa"]),
        float(gate["selected_temperature"]),
        float(gate["selected_beta_max"]),
    )
    selected_result = results[selected]
    gate["selected_value_checkpoint_path"] = selected_result[
        "value_checkpoint_path"
    ]
    gate["selected_value_training_provenance"] = selected_result[
        "value_training_provenance"
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, sort_keys=True))


if __name__ == "__main__":
    main()
