import csv
from pathlib import Path

import pytest

from scripts.route_a_v3.build_route2_v332_error_domain_shift_analysis_table_v1 import (
    DEFAULT_EVALUATOR_TABLE,
    ErrorDomainShiftInputError,
    build_table,
)


def _build(tmp_path: Path, *, evaluator_table_path: Path = DEFAULT_EVALUATOR_TABLE):
    table_path = tmp_path / "error_domain_shift.csv"
    audit_path = tmp_path / "error_domain_shift.json"
    audit = build_table(
        evaluator_table_path=evaluator_table_path,
        output_table_path=table_path,
        output_audit_path=audit_path,
    )
    with table_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return audit, rows, table_path, audit_path


def test_builder_separates_development_error_and_historical_shift_layers(tmp_path: Path) -> None:
    audit, rows, table_path, audit_path = _build(tmp_path)

    assert audit["status"] == "ERROR_AND_DOMAIN_SHIFT_ANALYSIS_REPORTED_NO_CAUSAL_OR_FINAL_CONFIRMATION_CLAIM"
    assert audit["row_count"] == len(rows) == 12
    assert audit["development_task_rows"] == 9
    assert audit["historical_seed_rows"] == 3
    assert audit["development_validation_record_count_sum"] == 18293
    assert audit["historical_transfer_record_count"] == 8068
    assert audit["metric_layer_separation"] == {
        "development_rows_with_critic_metrics": 9,
        "development_rows_with_independent_evaluator_metrics": 9,
        "historical_rows_with_zero_shot_seed_metrics": 3,
        "cross_layer_missing_metrics_are_blank": True,
        "cross_layer_numeric_pooling_allowed": False,
    }

    development = [row for row in rows if row["analysis_layer"].startswith("DEVELOPMENT")]
    historical = [row for row in rows if row["analysis_layer"].startswith("HISTORICAL")]
    assert len(development) == 9
    assert len(historical) == 3
    assert {row["region"] for row in development} == {"5UTR", "3UTR"}
    assert len({row["assay_id"] for row in development}) == 7
    assert all(row["historical_seed"] == "" for row in development)
    assert all(row["critic_v2_full_spearman"] == "" for row in historical)
    assert all(row["independent_evaluator_spearman"] == "" for row in historical)
    assert all(row["external_confirmation_eligible"] == "false" for row in rows)
    assert all(row["development_test_read"] == "false" for row in rows)
    assert all(row["new_final_evaluation_read"] == "false" for row in rows)
    assert all(row["guided_xeditflow_run"] == "false" for row in rows)
    assert all(row["post_hoc_noncausal"] == "true" for row in development)
    assert all(row["post_hoc_noncausal"] == "false" for row in historical)

    geometry = audit["development_failure_geometry"]
    assert geometry["critic_v2_spearman_win_count_vs_strongest_baseline"] == 4
    assert geometry["critic_v2_spearman_loss_count_vs_strongest_baseline"] == 5
    assert geometry["critic_v2_standardized_mae_better_task_count_vs_strongest_baseline"] == 0
    assert geometry["critic_v2_standardized_mae_worse_task_count_vs_strongest_baseline"] == 9
    assert geometry["n48_exclusion_is_replacement_endpoint"] is False
    assert geometry["causal_mechanism_established"] is False
    assert audit["assay_context_resolution"]["within_assay_context_specific_error_metrics_available"] is False
    assert audit["historical_domain_shift"]["rank_improvement_ci_excludes_zero_seed_count"] == 2
    assert audit["historical_domain_shift"]["baseline_mae_minus_model_mae_negative_seed_count"] == 3
    assert audit["historical_domain_shift"]["preregistered_pass"] is False
    assert audit["error_domain_shift_analysis_complete"] is True
    assert audit["external_transfer_established"] is False
    assert audit["scientific_claim_status"] == "NOT_ESTABLISHED"

    with pytest.raises(FileExistsError):
        build_table(output_table_path=table_path, output_audit_path=audit_path)


def test_builder_rejects_task_record_count_mismatch(tmp_path: Path) -> None:
    with DEFAULT_EVALUATOR_TABLE.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    rows[0]["record_count"] = "731"
    bad_table = tmp_path / "bad_evaluator.csv"
    with bad_table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ErrorDomainShiftInputError, match="record count mismatch"):
        _build(tmp_path / "bad", evaluator_table_path=bad_table)
