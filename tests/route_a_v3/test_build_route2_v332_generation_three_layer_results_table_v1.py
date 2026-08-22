import csv
import json
from pathlib import Path

import pytest

from scripts.route_a_v3.build_route2_v332_generation_three_layer_results_table_v1 import (
    DEFAULT_SNAPSHOT,
    GenerationThreeLayerInputError,
    build_table,
)


def _build(tmp_path: Path, *, snapshot_path: Path = DEFAULT_SNAPSHOT):
    table = tmp_path / "three_layer.csv"
    audit_path = tmp_path / "three_layer.json"
    audit = build_table(
        snapshot_path=snapshot_path,
        output_table_path=table,
        output_audit_path=audit_path,
    )
    with table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return audit, rows, table, audit_path


def test_builder_separates_critic_independent_and_sparse_measured_layers(
    tmp_path: Path,
) -> None:
    audit, rows, table, audit_path = _build(tmp_path)

    assert audit["status"] == (
        "GENERATION_EVIDENCE_LAYERS_SEPARATED_NO_GUIDED_OR_BIOLOGICAL_CLAIM"
    )
    assert len(rows) == audit["row_count"] == 9
    assert audit["executed_terminal_method_rows"] == 7
    assert audit["guided_no_go_boundary_rows"] == 2
    assert audit["numeric_coverage"] == {
        "critic_self_score_rows": 6,
        "independent_evaluator_rows": 7,
        "measured_candidate_recovery_rows": 7,
        "closed_measured_ndcg_rows": 0,
        "biological_improvement_claim_rows": 0,
    }
    assert audit["three_layer_reporting_complete"] is True
    assert audit["guided_generation_comparison_complete"] is False
    assert audit["headline_generation_improvement_established"] is False
    assert audit["scientific_claim_status"] == "NOT_ESTABLISHED"

    executed = [
        row
        for row in rows
        if row["result_status"] == "EXECUTED_TERMINAL_MATCHED_DEVELOPMENT"
    ]
    guided = [row for row in rows if row["result_status"] == "NOT_RUN_CRITIC_V2_NO_GO"]
    assert len(executed) == 7
    assert len(guided) == 2
    assert {int(row["source_count"]) for row in executed} == {891}
    assert all(row["independent_layer_status"].endswith("METHOD_SELECTION_ONLY") for row in executed)
    assert all(row["measured_layer_status"].startswith("SPARSE_MEASURED_RECOVERY") for row in executed)
    assert all(row["source_macro_closed_measured_ndcg_at_k"] == "" for row in rows)
    assert all(row["source_macro_normalized_regret"] == "" for row in rows)
    assert all(row["cross_layer_numeric_ranking_allowed"] == "false" for row in rows)
    assert all(row["biological_improvement_claim_allowed"] == "false" for row in rows)
    assert all(row["development_test_read"] == "false" for row in rows)
    assert all(row["new_final_evaluation_read"] == "false" for row in rows)

    critic = [
        row for row in executed if row["source_macro_critic_max_uplift_over_source"]
    ]
    assert len(critic) == 6
    assert {int(row["critic_score_defined_source_count"]) for row in critic} == {891}
    flow = next(row for row in executed if row["method_id"] == "unguided_learned_base_flow_g0")
    assert flow["critic_layer_status"] == "NOT_APPLICABLE_NO_CRITIC_CALLS"
    assert flow["source_macro_critic_max_uplift_over_source"] == ""
    assert flow["critic_score_defined_source_count"] == "0"
    assert float(flow["source_macro_candidate_recovery_rate"]) == 0.20286195286195285

    leaders = audit["descriptive_development_layer_leaders"]
    assert leaders["critic_self_score"]["method_id"] == "genetic"
    assert leaders["independent_evaluator"]["method_id"] == "genetic"
    assert leaders["measured_candidate_recovery"]["method_id"] == (
        "unguided_learned_base_flow_g0"
    )
    assert leaders["leader_comparison_is_cross_layer_numeric_ranking"] is False
    assert audit["layer_interpretation"]["measured_candidate_recovery_leader_differs"] is True

    assert all(row["candidate_count"] == "" for row in guided)
    assert all(row["critic_layer_status"] == "NOT_RUN_CRITIC_V2_NO_GO" for row in guided)
    assert all(row["guided_required"] == "true" for row in guided)
    assert all(row["guided_executed"] == "false" for row in guided)
    with pytest.raises(FileExistsError):
        build_table(output_table_path=table, output_audit_path=audit_path)


def test_builder_rejects_unknown_generated_candidates_encoded_as_zero_gain(
    tmp_path: Path,
) -> None:
    snapshot = json.loads(DEFAULT_SNAPSHOT.read_text(encoding="utf-8"))
    snapshot["methods"][0]["measured"][
        "unknown_generated_candidates_are_zero_gain"
    ] = True
    bad_snapshot = tmp_path / "bad_snapshot.json"
    bad_snapshot.write_text(json.dumps(snapshot), encoding="utf-8")

    with pytest.raises(
        GenerationThreeLayerInputError,
        match="unknown generated candidates were relabeled as zero gain",
    ):
        _build(tmp_path / "bad", snapshot_path=bad_snapshot)
