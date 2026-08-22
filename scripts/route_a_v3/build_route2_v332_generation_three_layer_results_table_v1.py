#!/usr/bin/env python3
"""Build the terminal Generation critic/independent/measured evidence table.

The table keeps the three evidence layers in separate columns and never treats
an unavailable measured metric as zero.  It reads only a versioned aggregate
snapshot plus existing terminal reporting tables; it does not open candidate
payloads, Development TEST, or new final Evaluation outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = (
    ROOT
    / "audits/route_a_v3_route2_v332_generation_three_layer_terminal_snapshot_v1.json"
)
DEFAULT_GENERATION_TABLE = (
    ROOT / "docs/paper/route2_v332_generation_baseline_table_v1.csv"
)
DEFAULT_MATCHED_BUDGET_MATRIX = (
    ROOT / "docs/paper/route2_v332_matched_budget_baseline_matrix_v1.csv"
)
DEFAULT_OUTPUT_TABLE = (
    ROOT / "docs/paper/route2_v332_generation_three_layer_results_table_v1.csv"
)
DEFAULT_OUTPUT_AUDIT = (
    ROOT / "audits/route_a_v3_route2_v332_generation_three_layer_results_table_v1.json"
)

EXECUTED_METHOD_ORDER = (
    "random_legal",
    "greedy",
    "beam",
    "genetic",
    "local_search",
    "generate_then_rerank",
    "unguided_learned_base_flow_g0",
)
GUIDED_METHOD_ORDER = (
    "first_order_rate_guidance",
    "frozen_critic_xeditflow",
)

FIELDS = (
    "result_row_id",
    "method_id",
    "result_status",
    "source_count",
    "candidate_count",
    "critic_layer_status",
    "source_macro_critic_max_uplift_over_source",
    "source_macro_critic_mean_uplift_over_source",
    "critic_score_defined_source_count",
    "independent_layer_status",
    "source_macro_independent_evaluator_max_uplift_over_source",
    "measured_layer_status",
    "source_macro_candidate_recovery_rate",
    "source_macro_measured_top_k_recovery_at_k",
    "source_macro_recovered_measured_ndcg_at_k",
    "recovered_measured_ndcg_defined_source_count",
    "recovered_measured_ndcg_role",
    "source_macro_closed_measured_ndcg_at_k",
    "closed_measured_ndcg_defined_source_count",
    "source_macro_normalized_regret",
    "normalized_regret_defined_source_count",
    "unknown_generated_candidates_are_zero_gain",
    "generated_candidates_grant_canonical_credit",
    "cross_layer_numeric_ranking_allowed",
    "biological_improvement_claim_allowed",
    "development_test_read",
    "new_final_evaluation_read",
    "guided_required",
    "guided_executed",
    "claim_boundary",
    "evidence_locator",
)


class GenerationThreeLayerInputError(RuntimeError):
    """Frozen terminal inputs do not support the declared three-layer row."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GenerationThreeLayerInputError(message)


def _blank_row() -> dict[str, Any]:
    return {field: "" for field in FIELDS}


def _executed_rows(
    snapshot: Mapping[str, Any],
    generation_table: Sequence[Mapping[str, str]],
    matched_budget_matrix: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    snapshot_by_method = {row["method_id"]: row for row in snapshot["methods"]}
    generation_by_method = {row["method_id"]: row for row in generation_table}
    budget_by_method = {
        row["method_id"]: row
        for row in matched_budget_matrix
        if row["track"] == "GENERATION"
    }
    expected = set(EXECUTED_METHOD_ORDER)
    _require(set(snapshot_by_method) == expected, "terminal snapshot method set changed")
    _require(set(generation_by_method) == expected, "generation result method set changed")

    rows: list[dict[str, Any]] = []
    for index, method_id in enumerate(EXECUTED_METHOD_ORDER, start=1):
        terminal = snapshot_by_method[method_id]
        generation = generation_by_method[method_id]
        budget = budget_by_method[method_id]
        measured = terminal["measured"]
        source_count = int(generation["source_count"])
        candidate_count = int(generation["candidate_count"])

        _require(
            budget["result_status"] == "EXECUTED_TERMINAL_MATCHED_DEVELOPMENT",
            f"{method_id} terminal execution status changed",
        )
        _require(source_count == terminal["generation_source_count"] == 891,
                 f"{method_id} source count changed")
        _require(measured["source_count"] == source_count,
                 f"{method_id} measured source count changed")
        _require(
            float(generation["source_macro_candidate_recovery_rate"])
            == measured["source_macro_candidate_recovery_rate"],
            f"{method_id} candidate recovery changed",
        )
        _require(
            float(generation["source_macro_measured_top_k_recovery_at_k"])
            == measured["source_macro_measured_top_k_recovery_at_k"],
            f"{method_id} measured top-k recovery changed",
        )
        _require(
            measured["source_closed_measured_ndcg_defined_count"] == 0
            and measured["source_macro_closed_measured_ndcg_at_k"] is None,
            f"{method_id} closed measured NDCG boundary changed",
        )
        _require(
            measured["source_macro_normalized_regret"] is None,
            f"{method_id} source-macro normalized regret unexpectedly became numeric",
        )
        _require(
            measured["unknown_generated_candidates_are_zero_gain"] is False,
            f"{method_id} unknown generated candidates were relabeled as zero gain",
        )

        if method_id == "unguided_learned_base_flow_g0":
            _require(terminal["critic_defined_source_count"] == 0,
                     "unguided Base Flow unexpectedly has critic self-scores")
            _require(terminal["critic_candidate_count_sum"] is None,
                     "unguided Base Flow unexpectedly has critic candidate counts")
            _require(terminal["source_macro_critic_max_uplift_over_source"] is None,
                     "unguided Base Flow unexpectedly has critic uplift")
            critic_status = "NOT_APPLICABLE_NO_CRITIC_CALLS"
            critic_max: Any = ""
            critic_mean: Any = ""
        else:
            _require(terminal["critic_defined_source_count"] == source_count,
                     f"{method_id} critic source coverage is incomplete")
            _require(terminal["critic_candidate_count_sum"] == candidate_count,
                     f"{method_id} critic candidate coverage changed")
            _require(
                terminal["source_macro_critic_max_uplift_over_source"] is not None
                and terminal["source_macro_critic_mean_uplift_over_source"] is not None,
                f"{method_id} critic self-score aggregate is missing",
            )
            critic_status = "TERMINAL_GUIDING_CRITIC_SELF_SCORE_891_SOURCE"
            critic_max = terminal["source_macro_critic_max_uplift_over_source"]
            critic_mean = terminal["source_macro_critic_mean_uplift_over_source"]

        row = _blank_row()
        row.update(
            {
                "result_row_id": f"G-3L-{index:02d}",
                "method_id": method_id,
                "result_status": "EXECUTED_TERMINAL_MATCHED_DEVELOPMENT",
                "source_count": source_count,
                "candidate_count": candidate_count,
                "critic_layer_status": critic_status,
                "source_macro_critic_max_uplift_over_source": critic_max,
                "source_macro_critic_mean_uplift_over_source": critic_mean,
                "critic_score_defined_source_count": terminal[
                    "critic_defined_source_count"
                ],
                "independent_layer_status": (
                    "TERMINAL_FROZEN_EVALUATOR_DEVELOPMENT_METHOD_SELECTION_ONLY"
                ),
                "source_macro_independent_evaluator_max_uplift_over_source": float(
                    generation[
                        "source_macro_independent_evaluator_max_uplift_over_source"
                    ]
                ),
                "measured_layer_status": (
                    "SPARSE_MEASURED_RECOVERY_ONLY_CLOSED_NDCG_UNDEFINED"
                ),
                "source_macro_candidate_recovery_rate": measured[
                    "source_macro_candidate_recovery_rate"
                ],
                "source_macro_measured_top_k_recovery_at_k": measured[
                    "source_macro_measured_top_k_recovery_at_k"
                ],
                "source_macro_recovered_measured_ndcg_at_k": measured[
                    "source_macro_recovered_measured_ndcg_at_k"
                ],
                "recovered_measured_ndcg_defined_source_count": measured[
                    "source_recovered_measured_ndcg_defined_count"
                ],
                "recovered_measured_ndcg_role": (
                    "CONDITIONAL_ON_RECOVERED_MEASURED_CANDIDATES_NOT_CLOSED_SUPPORT"
                ),
                "source_macro_closed_measured_ndcg_at_k": "",
                "closed_measured_ndcg_defined_source_count": 0,
                "source_macro_normalized_regret": "",
                "normalized_regret_defined_source_count": measured[
                    "source_normalized_regret_defined_count"
                ],
                "unknown_generated_candidates_are_zero_gain": "false",
                "generated_candidates_grant_canonical_credit": "false",
                "cross_layer_numeric_ranking_allowed": "false",
                "biological_improvement_claim_allowed": "false",
                "development_test_read": "false",
                "new_final_evaluation_read": "false",
                "guided_required": "false",
                "guided_executed": "false",
                "claim_boundary": (
                    "Critic self-score and independent evaluator are Development computational "
                    "signals; measured evidence is sparse recovery with zero closed-NDCG-defined "
                    "sources. No layer establishes biological improvement."
                ),
                "evidence_locator": (
                    "audits/route_a_v3_route2_v332_generation_three_layer_terminal_snapshot_v1.json;"
                    "docs/paper/route2_v332_generation_baseline_table_v1.csv"
                ),
            }
        )
        rows.append(row)
    return rows


def _guided_boundary_rows(
    matched_budget_matrix: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    budget_by_method = {
        row["method_id"]: row
        for row in matched_budget_matrix
        if row["track"] == "GENERATION"
    }
    rows: list[dict[str, Any]] = []
    for offset, method_id in enumerate(GUIDED_METHOD_ORDER, start=8):
        budget = budget_by_method[method_id]
        _require(
            budget["result_status"] == "NOT_RUN_CRITIC_V2_NO_GO",
            f"guided {method_id} is no longer closed by Critic V2 NO-GO",
        )
        _require(
            budget["guided_required"] == "true"
            and budget["guided_executed"] == "false",
            f"guided {method_id} execution boundary changed",
        )
        row = _blank_row()
        row.update(
            {
                "result_row_id": f"G-3L-{offset:02d}",
                "method_id": method_id,
                "result_status": "NOT_RUN_CRITIC_V2_NO_GO",
                "source_count": 891,
                "critic_layer_status": "NOT_RUN_CRITIC_V2_NO_GO",
                "independent_layer_status": "NOT_RUN_CRITIC_V2_NO_GO",
                "measured_layer_status": "NOT_RUN_CRITIC_V2_NO_GO",
                "generated_candidates_grant_canonical_credit": "false",
                "cross_layer_numeric_ranking_allowed": "false",
                "biological_improvement_claim_allowed": "false",
                "development_test_read": "false",
                "new_final_evaluation_read": "false",
                "guided_required": "true",
                "guided_executed": "false",
                "claim_boundary": (
                    "Critic V2 terminal NO-GO prohibits guided execution; all three evidence "
                    "layers are unavailable and remain blank rather than zero."
                ),
                "evidence_locator": budget["evidence_locator"],
            }
        )
        rows.append(row)
    return rows


def derive_rows_and_audit(
    snapshot: Mapping[str, Any],
    generation_table: Sequence[Mapping[str, str]],
    matched_budget_matrix: Sequence[Mapping[str, str]],
    *,
    source_paths: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require(
        snapshot["status"]
        == "TERMINAL_GENERATION_LAYER_AGGREGATES_CAPTURED_WITHOUT_CANDIDATE_PAYLOAD",
        "three-layer terminal snapshot status changed",
    )
    _require(
        snapshot["aggregation_policy"]
        == {
            "source_unit_count": 891,
            "critic_max_uplift": (
                "MEAN_OVER_PER_SOURCE_CRITIC_SCORE_MAX_UPLIFT_OVER_SOURCE"
            ),
            "critic_mean_uplift": (
                "MEAN_OVER_PER_SOURCE_CRITIC_SCORE_MEAN_UPLIFT_OVER_SOURCE"
            ),
            "candidate_payload_opened": False,
            "generated_candidate_outcome_opened": False,
            "missing_numeric_value_substituted_with_zero": False,
        },
        "three-layer aggregation or candidate-access boundary changed",
    )
    _require(
        snapshot["protected_outcomes"]
        == {
            "development_test_read": False,
            "new_final_evaluation_read": False,
            "guided_xeditflow_run": False,
        },
        "three-layer snapshot opened a protected outcome",
    )

    executed = _executed_rows(snapshot, generation_table, matched_budget_matrix)
    guided = _guided_boundary_rows(matched_budget_matrix)
    rows = executed + guided
    _require(len(rows) == 9, "three-layer result row count changed")
    _require(len({row["result_row_id"] for row in rows}) == len(rows),
             "three-layer result row IDs are not unique")

    critic_numeric = [
        row
        for row in executed
        if row["source_macro_critic_max_uplift_over_source"] != ""
    ]
    independent_numeric = [
        row
        for row in executed
        if row["source_macro_independent_evaluator_max_uplift_over_source"] != ""
    ]
    measured_numeric = [
        row for row in executed if row["source_macro_candidate_recovery_rate"] != ""
    ]
    critic_leader = max(
        critic_numeric,
        key=lambda row: float(row["source_macro_critic_max_uplift_over_source"]),
    )
    independent_leader = max(
        independent_numeric,
        key=lambda row: float(
            row["source_macro_independent_evaluator_max_uplift_over_source"]
        ),
    )
    measured_leader = max(
        measured_numeric,
        key=lambda row: float(row["source_macro_candidate_recovery_rate"]),
    )

    audit = {
        "schema_version": "route_a_v3_route2_v332_generation_three_layer_results_table.v1",
        "status": "GENERATION_EVIDENCE_LAYERS_SEPARATED_NO_GUIDED_OR_BIOLOGICAL_CLAIM",
        "row_count": len(rows),
        "executed_terminal_method_rows": len(executed),
        "guided_no_go_boundary_rows": len(guided),
        "numeric_coverage": {
            "critic_self_score_rows": len(critic_numeric),
            "independent_evaluator_rows": len(independent_numeric),
            "measured_candidate_recovery_rows": len(measured_numeric),
            "closed_measured_ndcg_rows": sum(
                row["source_macro_closed_measured_ndcg_at_k"] != "" for row in rows
            ),
            "biological_improvement_claim_rows": sum(
                row["biological_improvement_claim_allowed"] == "true" for row in rows
            ),
        },
        "descriptive_development_layer_leaders": {
            "critic_self_score": {
                "method_id": critic_leader["method_id"],
                "source_macro_critic_max_uplift_over_source": critic_leader[
                    "source_macro_critic_max_uplift_over_source"
                ],
            },
            "independent_evaluator": {
                "method_id": independent_leader["method_id"],
                "source_macro_independent_evaluator_max_uplift_over_source": (
                    independent_leader[
                        "source_macro_independent_evaluator_max_uplift_over_source"
                    ]
                ),
            },
            "measured_candidate_recovery": {
                "method_id": measured_leader["method_id"],
                "source_macro_candidate_recovery_rate": measured_leader[
                    "source_macro_candidate_recovery_rate"
                ],
            },
            "leader_comparison_is_cross_layer_numeric_ranking": False,
        },
        "layer_interpretation": {
            "critic_and_independent_leader_same": (
                critic_leader["method_id"] == independent_leader["method_id"]
            ),
            "measured_candidate_recovery_leader_differs": (
                measured_leader["method_id"] != independent_leader["method_id"]
            ),
            "recovered_measured_ndcg_is_conditional_support_only": True,
            "unknown_generated_candidates_are_zero_gain": False,
            "cross_layer_numeric_ranking_allowed": False,
            "critic_or_independent_substitutes_for_measured_outcome": False,
        },
        "three_layer_reporting_complete": True,
        "guided_generation_comparison_complete": False,
        "headline_generation_improvement_established": False,
        "new_training_attempt_created": False,
        "protected_outcomes": {
            "development_test_read": False,
            "new_final_evaluation_read": False,
            "guided_xeditflow_run": False,
        },
        "scientific_claim_status": "NOT_ESTABLISHED",
        "source_data": dict(source_paths),
    }
    return rows, audit


def build_table(
    *,
    snapshot_path: Path = DEFAULT_SNAPSHOT,
    generation_table_path: Path = DEFAULT_GENERATION_TABLE,
    matched_budget_matrix_path: Path = DEFAULT_MATCHED_BUDGET_MATRIX,
    output_table_path: Path = DEFAULT_OUTPUT_TABLE,
    output_audit_path: Path = DEFAULT_OUTPUT_AUDIT,
) -> dict[str, Any]:
    if output_table_path.exists() or output_audit_path.exists():
        raise FileExistsError("refusing to overwrite versioned three-layer output")
    snapshot = _read_json(snapshot_path)
    generation_table = _read_csv(generation_table_path)
    matched_budget_matrix = _read_csv(matched_budget_matrix_path)
    rows, audit = derive_rows_and_audit(
        snapshot,
        generation_table,
        matched_budget_matrix,
        source_paths={
            "terminal_snapshot": str(snapshot_path),
            "generation_table": str(generation_table_path),
            "matched_budget_matrix": str(matched_budget_matrix_path),
        },
    )
    output_table_path.parent.mkdir(parents=True, exist_ok=True)
    with output_table_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    audit["table_path"] = str(output_table_path)
    output_audit_path.parent.mkdir(parents=True, exist_ok=True)
    output_audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--generation-table", type=Path, default=DEFAULT_GENERATION_TABLE)
    parser.add_argument(
        "--matched-budget-matrix", type=Path, default=DEFAULT_MATCHED_BUDGET_MATRIX
    )
    parser.add_argument("--output-table", type=Path, default=DEFAULT_OUTPUT_TABLE)
    parser.add_argument("--output-audit", type=Path, default=DEFAULT_OUTPUT_AUDIT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    audit = build_table(
        snapshot_path=args.snapshot,
        generation_table_path=args.generation_table,
        matched_budget_matrix_path=args.matched_budget_matrix,
        output_table_path=args.output_table,
        output_audit_path=args.output_audit,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
