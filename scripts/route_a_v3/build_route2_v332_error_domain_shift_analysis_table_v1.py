#!/usr/bin/env python3
"""Build the terminal Route 2 error/domain-shift analysis table.

The output deliberately separates two evidence layers:

* nine frozen Development Validation task-region diagnostics; and
* three historically outcome-exposed GSE232572 zero-shot seed diagnostics.

Metrics that do not exist in a layer are left blank.  The builder reads only
versioned aggregate tables, audits and converter metadata.  It does not open
Development TEST, a new final Evaluation outcome, or candidate-level payloads.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CRITIC_TABLE = ROOT / "docs/paper/route2_v332_critic_v2_task_diagnostic_table_v1.csv"
DEFAULT_EVALUATOR_TABLE = ROOT / "docs/paper/route2_v332_independent_evaluator_task_table_v1.csv"
DEFAULT_CRITIC_AUDIT = ROOT / "audits/route_a_v3_route2_critic_v2_task_failure_diagnostic_v1.json"
DEFAULT_HISTORICAL_AUDIT = ROOT / "audits/route_a_v3_route2_gse232572_zero_shot_summary_v1.json"
DEFAULT_DATASET_TABLE = ROOT / "docs/paper/route2_v332_dataset_qualification_table_v1.csv"
DEFAULT_OUTPUT_TABLE = ROOT / "docs/paper/route2_v332_error_domain_shift_analysis_table_v1.csv"
DEFAULT_OUTPUT_AUDIT = ROOT / "audits/route_a_v3_route2_v332_error_domain_shift_analysis_table_v1.json"

CONVERTER_PATHS = (
    ROOT / "configs/route_a_v3_route2_gse114002_converter_v1.json",
    ROOT / "configs/route_a_v3_route2_encsr854ruf_converter_v1.json",
    ROOT / "configs/route_a_v3_route2_gse269595_converter_v1.json",
    ROOT / "configs/route_a_v3_route2_gse186455_converter_v1.json",
    ROOT / "configs/route_a_v3_route2_gse217518_converter_v1.json",
    ROOT / "configs/route_a_v3_route2_gse200304_converter_v1.json",
    ROOT / "configs/route_a_v3_route2_gse149487_converter_v1.json",
    ROOT / "configs/route_a_v3_route2_gse232572_converter_v1.json",
)

TASK_ORDER = (
    "MEAN_RIBOSOME_LOAD::region=0",
    "MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE::region=1",
    "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS::region=1",
    "PUBLISHED_REF_VS_ALT_ACTIVITY_LMM_LOG2_FOLD_CHANGE::region=1",
    "RNA_HALF_LIFE_MINUTES::region=0",
    "RNA_HALF_LIFE_MINUTES::region=1",
    "TOTAL_POLYSOME_TRANSLATION_EFFICIENCY::region=1",
    "te_log2_polysome_over_totalrna::region=0",
    "transcript_log2_totalrna_over_dna::region=0",
)

TASK_TO_STUDY = {
    "MEAN_RIBOSOME_LOAD::region=0": "GSE114002",
    "MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE::region=1": "ENCSR854RUF",
    "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS::region=1": "GSE269595",
    "PUBLISHED_REF_VS_ALT_ACTIVITY_LMM_LOG2_FOLD_CHANGE::region=1": "GSE186455",
    "RNA_HALF_LIFE_MINUTES::region=0": "GSE217518",
    "RNA_HALF_LIFE_MINUTES::region=1": "GSE217518",
    "TOTAL_POLYSOME_TRANSLATION_EFFICIENCY::region=1": "GSE200304",
    "te_log2_polysome_over_totalrna::region=0": "GSE149487",
    "transcript_log2_totalrna_over_dna::region=0": "GSE149487",
}

REGION_INDEX_TO_NAME = {"0": "5UTR", "1": "3UTR"}

FIELDS = (
    "result_row_id",
    "analysis_layer",
    "study_id",
    "task_id",
    "endpoint_id",
    "region",
    "assay_id",
    "biological_context_scope",
    "record_count",
    "evidence_stage",
    "critic_v2_full_spearman",
    "critic_v2_candidate_permutation_spearman",
    "critic_v2_source_only_spearman",
    "critic_v2_source_edit_metadata_spearman",
    "critic_v2_strongest_same_information_baseline_spearman",
    "critic_v2_full_minus_strongest_baseline_spearman",
    "critic_v2_full_standardized_mae",
    "critic_v2_strongest_baseline_standardized_mae",
    "critic_v2_full_minus_strongest_baseline_standardized_mae",
    "independent_evaluator_spearman",
    "independent_evaluator_mae",
    "independent_evaluator_standardized_mae",
    "historical_seed",
    "historical_model_spearman",
    "historical_strongest_baseline_spearman",
    "historical_model_minus_baseline_spearman",
    "historical_model_minus_baseline_spearman_ci95_lower",
    "historical_model_minus_baseline_spearman_ci95_upper",
    "historical_baseline_minus_model_mae",
    "historical_baseline_minus_model_mae_ci95_lower",
    "historical_baseline_minus_model_mae_ci95_upper",
    "historical_preregistered_pass",
    "external_confirmation_eligible",
    "outcome_exposure_status",
    "development_test_read",
    "new_final_evaluation_read",
    "guided_xeditflow_run",
    "post_hoc_noncausal",
    "failure_or_shift_interpretation",
    "claim_boundary",
    "evidence_locator",
)


class ErrorDomainShiftInputError(RuntimeError):
    """Frozen terminal inputs do not support the declared table."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ErrorDomainShiftInputError(message)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _blank_row() -> dict[str, Any]:
    return {field: "" for field in FIELDS}


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _endpoint_ids(study: Mapping[str, Any]) -> list[str]:
    if "endpoint_id" in study:
        return [str(study["endpoint_id"])]
    return [str(value) for value in study["endpoint_ids"]]


def _context_scope(study_id: str, study: Mapping[str, Any]) -> str:
    if "biological_context_id" in study:
        return str(study["biological_context_id"])
    if "biological_context_ids" in study:
        return "|".join(str(value) for value in study["biological_context_ids"])
    if study_id == "GSE269595":
        perturbations = "|".join(study["biological_context_perturbations"])
        reporters = "|".join(study["distal_reporter_contexts"])
        return f"HEK293FT;perturbations={perturbations};distal_reporters={reporters}"
    raise ErrorDomainShiftInputError(f"no biological context metadata for {study_id}")


def _study_metadata(converter_paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for path in converter_paths:
        study = _json(path)["study"]
        study_id = str(study["study_unit_id"])
        _require(study_id not in metadata, f"duplicate converter metadata for {study_id}")
        metadata[study_id] = {
            "study_id": study_id,
            "regions": [str(value) for value in study.get("regions", [study.get("region")])],
            "assay_id": str(study["assay_id"]),
            "endpoint_ids": _endpoint_ids(study),
            "biological_context_scope": _context_scope(study_id, study),
            "converter_path": _display_path(path),
        }
    expected = set(TASK_TO_STUDY.values()) | {"GSE232572"}
    _require(set(metadata) == expected, "converter study set changed")
    return metadata


def _task_parts(task_id: str) -> tuple[str, str]:
    endpoint_id, region_part = task_id.rsplit("::region=", 1)
    _require(region_part in REGION_INDEX_TO_NAME, f"unknown region index in {task_id}")
    return endpoint_id, REGION_INDEX_TO_NAME[region_part]


def derive_rows_and_audit(
    *,
    critic_rows: Sequence[Mapping[str, str]],
    evaluator_rows: Sequence[Mapping[str, str]],
    critic_audit: Mapping[str, Any],
    historical_audit: Mapping[str, Any],
    dataset_rows: Sequence[Mapping[str, str]],
    metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    critic_by_task = {row["task_id"]: row for row in critic_rows}
    evaluator_by_task = {row["task_id"]: row for row in evaluator_rows}
    _require(tuple(critic_by_task) == TASK_ORDER, "Critic V2 task order or task set changed")
    _require(tuple(evaluator_by_task) == TASK_ORDER, "evaluator task order or task set changed")

    protected = critic_audit["protected_outcomes"]
    _require(protected["development_test_opened"] is False, "Development TEST was opened")
    _require(protected["evaluation_opened"] is False, "a new final Evaluation was opened")
    _require(protected["guided_generation_authorized"] is False, "guided generation became authorized")
    _require(critic_audit["analysis_scope"] == "TERMINAL_DEVELOPMENT_VALIDATION_ONLY_POST_HOC_FAILURE_GEOMETRY", "Critic V2 analysis scope changed")

    dataset_by_study = {row["study_unit_id"]: row for row in dataset_rows}
    historical_status = dataset_by_study["GSE232572"]
    _require(
        historical_status["current_analysis_role_v332"]
        == "HISTORICAL_OUTCOME_EXPOSED_TRANSFER_DIAGNOSTIC_NOT_FINAL_CONFIRMATION",
        "GSE232572 current role changed",
    )
    _require(
        historical_status["outcome_exposure"] == "OUTCOME_EXPOSED_BY_EXISTING_ZERO_SHOT",
        "GSE232572 outcome-exposure boundary changed",
    )
    _require(historical_audit["preregistered_pass"] is False, "historical transfer pass status changed")
    _require(historical_audit["evaluation_record_count"] == 8068, "historical record count changed")

    rows: list[dict[str, Any]] = []
    for index, task_id in enumerate(TASK_ORDER, start=1):
        critic = critic_by_task[task_id]
        evaluator = evaluator_by_task[task_id]
        _require(critic["record_count"] == evaluator["record_count"], f"record count mismatch for {task_id}")
        endpoint_id, region = _task_parts(task_id)
        study_id = TASK_TO_STUDY[task_id]
        study = metadata[study_id]
        _require(endpoint_id in study["endpoint_ids"], f"endpoint metadata mismatch for {task_id}")
        _require(region in study["regions"], f"region metadata mismatch for {task_id}")

        spearman_margin = float(critic["full_minus_strongest_baseline_spearman"])
        standardized_mae_margin = float(critic["full_minus_strongest_baseline_standardized_mae"])
        evaluator_spearman = float(evaluator["spearman"])
        interpretation = (
            f"Critic V2 Spearman {'exceeds' if spearman_margin > 0 else 'does not exceed'} "
            f"the strongest same-information baseline; standardized MAE is "
            f"{'worse' if standardized_mae_margin > 0 else 'not worse'} than that baseline. "
            f"Independent-evaluator task Spearman is "
            f"{'positive' if evaluator_spearman > 0 else 'non-positive'}."
        )
        row = _blank_row()
        row.update(
            {
                "result_row_id": f"EDS-D-{index:02d}",
                "analysis_layer": "DEVELOPMENT_VALIDATION_TASK_ERROR_DIAGNOSTIC",
                "study_id": study_id,
                "task_id": task_id,
                "endpoint_id": endpoint_id,
                "region": region,
                "assay_id": study["assay_id"],
                "biological_context_scope": study["biological_context_scope"],
                "record_count": int(critic["record_count"]),
                "evidence_stage": "TERMINAL_FROZEN_DEVELOPMENT_VALIDATION_ONLY",
                "critic_v2_full_spearman": float(critic["full_spearman"]),
                "critic_v2_candidate_permutation_spearman": float(critic["candidate_permutation_spearman"]),
                "critic_v2_source_only_spearman": float(critic["source_only_spearman"]),
                "critic_v2_source_edit_metadata_spearman": float(critic["source_edit_metadata_spearman"]),
                "critic_v2_strongest_same_information_baseline_spearman": float(critic["strongest_baseline_spearman"]),
                "critic_v2_full_minus_strongest_baseline_spearman": spearman_margin,
                "critic_v2_full_standardized_mae": float(critic["full_standardized_mae"]),
                "critic_v2_strongest_baseline_standardized_mae": float(critic["strongest_baseline_standardized_mae"]),
                "critic_v2_full_minus_strongest_baseline_standardized_mae": standardized_mae_margin,
                "independent_evaluator_spearman": evaluator_spearman,
                "independent_evaluator_mae": float(evaluator["mae"]),
                "independent_evaluator_standardized_mae": float(evaluator["standardized_mae"]),
                "external_confirmation_eligible": "false",
                "outcome_exposure_status": "DEVELOPMENT_EXPOSED",
                "development_test_read": "false",
                "new_final_evaluation_read": "false",
                "guided_xeditflow_run": "false",
                "post_hoc_noncausal": "true",
                "failure_or_shift_interpretation": interpretation,
                "claim_boundary": "Within-Development task heterogeneity and error geometry only; not external transfer, causal attribution or biological validation.",
                "evidence_locator": (
                    "docs/paper/route2_v332_critic_v2_task_diagnostic_table_v1.csv;"
                    "docs/paper/route2_v332_independent_evaluator_task_table_v1.csv;"
                    f"{study['converter_path']}"
                ),
            }
        )
        rows.append(row)

    historical_meta = metadata["GSE232572"]
    for index, result in enumerate(historical_audit["paired_results"], start=1):
        rank_ci = result["task_macro_spearman_improvement_ci_95"]
        mae_ci = result["baseline_mae_minus_model_mae_ci_95"]
        row = _blank_row()
        row.update(
            {
                "result_row_id": f"EDS-H-{index:02d}",
                "analysis_layer": "HISTORICAL_OUTCOME_EXPOSED_ZERO_SHOT_DOMAIN_SHIFT",
                "study_id": "GSE232572",
                "task_id": "GSE232572::historical_zero_shot",
                "endpoint_id": historical_meta["endpoint_ids"][0],
                "region": historical_meta["regions"][0],
                "assay_id": historical_meta["assay_id"],
                "biological_context_scope": historical_meta["biological_context_scope"],
                "record_count": historical_audit["evaluation_record_count"],
                "evidence_stage": "HISTORICALLY_OUTCOME_EXPOSED_TRANSFER_DIAGNOSTIC_NOT_FINAL_CONFIRMATION",
                "historical_seed": result["seed"],
                "historical_model_spearman": result["task_macro_spearman_model"],
                "historical_strongest_baseline_spearman": result["task_macro_spearman_baseline"],
                "historical_model_minus_baseline_spearman": result["task_macro_spearman_improvement"],
                "historical_model_minus_baseline_spearman_ci95_lower": rank_ci[0],
                "historical_model_minus_baseline_spearman_ci95_upper": rank_ci[1],
                "historical_baseline_minus_model_mae": result["baseline_mae_minus_model_mae"],
                "historical_baseline_minus_model_mae_ci95_lower": mae_ci[0],
                "historical_baseline_minus_model_mae_ci95_upper": mae_ci[1],
                "historical_preregistered_pass": "false",
                "external_confirmation_eligible": "false",
                "outcome_exposure_status": "HISTORICALLY_OUTCOME_EXPOSED_NOT_FINAL_CONFIRMATION",
                "development_test_read": "false",
                "new_final_evaluation_read": "false",
                "guided_xeditflow_run": "false",
                "post_hoc_noncausal": "false",
                "failure_or_shift_interpretation": (
                    "Historical zero-shot rank point improvement does not establish stable transfer; "
                    "the preregistered cross-seed rule fails and MAE favors the baseline for this seed."
                ),
                "claim_boundary": "Negative historical transfer evidence only; outcome exposure prohibits use as an unbiased final confirmation.",
                "evidence_locator": (
                    "audits/route_a_v3_route2_gse232572_zero_shot_summary_v1.json;"
                    f"{historical_meta['converter_path']};"
                    "docs/paper/route2_v332_dataset_qualification_table_v1.csv"
                ),
            }
        )
        rows.append(row)

    development_rows = rows[: len(TASK_ORDER)]
    historical_rows = rows[len(TASK_ORDER) :]
    region_summaries: dict[str, dict[str, Any]] = {}
    for region in ("5UTR", "3UTR"):
        subset = [row for row in development_rows if row["region"] == region]
        region_summaries[region] = {
            "task_count": len(subset),
            "validation_record_count_sum": sum(row["record_count"] for row in subset),
            "critic_v2_mean_full_minus_strongest_baseline_spearman": sum(
                row["critic_v2_full_minus_strongest_baseline_spearman"] for row in subset
            ) / len(subset),
            "critic_v2_mean_full_minus_strongest_baseline_standardized_mae": sum(
                row["critic_v2_full_minus_strongest_baseline_standardized_mae"] for row in subset
            ) / len(subset),
            "independent_evaluator_task_macro_spearman": sum(
                row["independent_evaluator_spearman"] for row in subset
            ) / len(subset),
            "interpretation": "DESCRIPTIVE_POST_HOC_REGION_STRATUM_CONFOUNDED_BY_STUDY_ASSAY_CONTEXT_ENDPOINT_AND_TASK_SIZE",
        }

    exact = critic_audit["exact_observations"]
    audit = {
        "schema_version": "route_a_v3_route2_v332_error_domain_shift_analysis_table.v1",
        "status": "ERROR_AND_DOMAIN_SHIFT_ANALYSIS_REPORTED_NO_CAUSAL_OR_FINAL_CONFIRMATION_CLAIM",
        "row_count": len(rows),
        "development_task_rows": len(development_rows),
        "historical_seed_rows": len(historical_rows),
        "development_validation_record_count_sum": sum(row["record_count"] for row in development_rows),
        "historical_transfer_record_count": historical_audit["evaluation_record_count"],
        "metric_layer_separation": {
            "development_rows_with_critic_metrics": sum(bool(row["critic_v2_full_spearman"] != "") for row in rows),
            "development_rows_with_independent_evaluator_metrics": sum(bool(row["independent_evaluator_spearman"] != "") for row in rows),
            "historical_rows_with_zero_shot_seed_metrics": sum(bool(row["historical_seed"] != "") for row in rows),
            "cross_layer_missing_metrics_are_blank": True,
            "cross_layer_numeric_pooling_allowed": False,
        },
        "development_failure_geometry": {
            "task_count": exact["task_count"],
            "minimum_task_record_count": exact["minimum_task_record_count"],
            "maximum_task_record_count": exact["maximum_task_record_count"],
            "maximum_to_minimum_task_record_count_ratio": exact["maximum_to_minimum_task_record_count_ratio"],
            "critic_v2_spearman_win_count_vs_strongest_baseline": exact["full_spearman_win_count_vs_strongest_baseline"],
            "critic_v2_spearman_loss_count_vs_strongest_baseline": exact["full_spearman_loss_count_vs_strongest_baseline"],
            "critic_v2_nine_task_mean_spearman_margin_vs_strongest_baseline": exact["nine_task_mean_spearman_margin_vs_strongest_baseline"],
            "critic_v2_standardized_mae_better_task_count_vs_strongest_baseline": exact["full_standardized_mae_better_task_count_vs_strongest_baseline"],
            "critic_v2_standardized_mae_worse_task_count_vs_strongest_baseline": exact["full_standardized_mae_worse_task_count_vs_strongest_baseline"],
            "critic_v2_nine_task_mean_standardized_mae_margin_vs_strongest_baseline": exact["nine_task_mean_standardized_mae_margin_vs_strongest_baseline"],
            "two_n48_task_sum_spearman_margin_vs_strongest_baseline": exact["two_n48_task_sum_spearman_margin_vs_strongest_baseline"],
            "remaining_seven_task_post_hoc_mean_spearman_margin_vs_strongest_baseline": exact["remaining_seven_task_post_hoc_mean_spearman_margin_vs_strongest_baseline"],
            "independent_evaluator_positive_task_count": sum(row["independent_evaluator_spearman"] > 0 for row in development_rows),
            "independent_evaluator_spearman_min": min(row["independent_evaluator_spearman"] for row in development_rows),
            "independent_evaluator_spearman_max": max(row["independent_evaluator_spearman"] for row in development_rows),
            "independent_evaluator_max_standardized_mae": max(row["independent_evaluator_standardized_mae"] for row in development_rows),
            "n48_exclusion_is_replacement_endpoint": False,
            "causal_mechanism_established": False,
        },
        "descriptive_region_summaries": region_summaries,
        "assay_context_resolution": {
            "development_assay_count": len({row["assay_id"] for row in development_rows}),
            "development_study_count": len({row["study_id"] for row in development_rows}),
            "task_rows_aggregate_one_or_more_biological_contexts": True,
            "within_assay_context_specific_error_metrics_available": False,
            "context_effect_claim_allowed": False,
        },
        "historical_domain_shift": {
            "study_id": historical_audit["study_unit_id"],
            "seed_count": len(historical_rows),
            "rank_improvement_ci_excludes_zero_seed_count": sum(
                row["historical_model_minus_baseline_spearman_ci95_lower"] > 0
                for row in historical_rows
            ),
            "baseline_mae_minus_model_mae_negative_seed_count": sum(
                row["historical_baseline_minus_model_mae"] < 0 for row in historical_rows
            ),
            "preregistered_pass": historical_audit["preregistered_pass"],
            "outcome_exposure_status": "HISTORICALLY_OUTCOME_EXPOSED_NOT_FINAL_CONFIRMATION",
            "external_confirmation_eligible": False,
        },
        "protected_outcomes": {
            "development_test_read": False,
            "new_final_evaluation_read": False,
            "guided_xeditflow_run": False,
        },
        "interpretation_limits": [
            "Development task and region summaries are post hoc diagnostics, not replacement endpoints or causal effects.",
            "Region summaries are confounded by study, assay, context, endpoint and task size.",
            "Task rows aggregate biological contexts, so within-assay context-specific error is not identifiable from the persisted terminal aggregates.",
            "GSE232572 is historically outcome-exposed and cannot serve as final confirmation.",
            "Heterogeneous raw MAE values are not pooled across endpoints; standardized MAE is reported per task only.",
        ],
        "error_domain_shift_analysis_complete": True,
        "external_transfer_established": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }
    return rows, audit


def build_table(
    *,
    critic_table_path: Path = DEFAULT_CRITIC_TABLE,
    evaluator_table_path: Path = DEFAULT_EVALUATOR_TABLE,
    critic_audit_path: Path = DEFAULT_CRITIC_AUDIT,
    historical_audit_path: Path = DEFAULT_HISTORICAL_AUDIT,
    dataset_table_path: Path = DEFAULT_DATASET_TABLE,
    converter_paths: Sequence[Path] = CONVERTER_PATHS,
    output_table_path: Path = DEFAULT_OUTPUT_TABLE,
    output_audit_path: Path = DEFAULT_OUTPUT_AUDIT,
) -> dict[str, Any]:
    if output_table_path.exists() or output_audit_path.exists():
        raise FileExistsError("refusing to overwrite an existing error/domain-shift artifact")
    rows, audit = derive_rows_and_audit(
        critic_rows=_csv(critic_table_path),
        evaluator_rows=_csv(evaluator_table_path),
        critic_audit=_json(critic_audit_path),
        historical_audit=_json(historical_audit_path),
        dataset_rows=_csv(dataset_table_path),
        metadata=_study_metadata(converter_paths),
    )
    output_table_path.parent.mkdir(parents=True, exist_ok=True)
    with output_table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    audit["table_path"] = _display_path(output_table_path)
    audit["source_paths"] = {
        "critic_table": _display_path(critic_table_path),
        "evaluator_table": _display_path(evaluator_table_path),
        "critic_audit": _display_path(critic_audit_path),
        "historical_audit": _display_path(historical_audit_path),
        "dataset_table": _display_path(dataset_table_path),
        "converters": [_display_path(path) for path in converter_paths],
    }
    output_audit_path.parent.mkdir(parents=True, exist_ok=True)
    output_audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic-table", type=Path, default=DEFAULT_CRITIC_TABLE)
    parser.add_argument("--evaluator-table", type=Path, default=DEFAULT_EVALUATOR_TABLE)
    parser.add_argument("--critic-audit", type=Path, default=DEFAULT_CRITIC_AUDIT)
    parser.add_argument("--historical-audit", type=Path, default=DEFAULT_HISTORICAL_AUDIT)
    parser.add_argument("--dataset-table", type=Path, default=DEFAULT_DATASET_TABLE)
    parser.add_argument("--output-table", type=Path, default=DEFAULT_OUTPUT_TABLE)
    parser.add_argument("--output-audit", type=Path, default=DEFAULT_OUTPUT_AUDIT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        json.dumps(
            build_table(
                critic_table_path=args.critic_table,
                evaluator_table_path=args.evaluator_table,
                critic_audit_path=args.critic_audit,
                historical_audit_path=args.historical_audit,
                dataset_table_path=args.dataset_table,
                output_table_path=args.output_table,
                output_audit_path=args.output_audit,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
