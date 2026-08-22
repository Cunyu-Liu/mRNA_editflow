#!/usr/bin/env python3
"""Build the V3.3.2 native/common-task/architecture-controlled results table."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = ROOT / "audits/route_a_v3_route2_v332_three_track_terminal_input_snapshot_v1.json"
DEFAULT_INVENTORY = ROOT / "configs/route_a_v3_route2_baseline_inventory_v1.json"
DEFAULT_PROVENANCE = ROOT / "configs/route_a_v3_route2_external_model_provenance_v1.json"
DEFAULT_EXTERNAL_HPO = ROOT / "audits/route_a_v3_route2_external_hpo_selection_v1.json"
DEFAULT_BOTTLENECK = ROOT / "audits/route_a_v3_route2_rnafm_bottleneck_adapter_gpu_audit_v1.json"
DEFAULT_NEURAL = ROOT / "audits/route_a_v3_route2_neural_hpo_selection_v1.json"
DEFAULT_LEGACY = ROOT / "audits/route_a_v3_route2_legacy_baseline_task_macro_replay_v1.json"
DEFAULT_CRITIC = ROOT / "audits/route_a_v3_route2_critic_v2_control_terminal_no_go_v1.json"
DEFAULT_GENERATION = ROOT / "docs/paper/route2_v332_generation_baseline_table_v1.csv"
DEFAULT_GEOMETRY = ROOT / "audits/route_a_v3_route2_generation_action_space_geometry_v1.json"
DEFAULT_ALIGNED_A1 = ROOT / "configs/route_a_v3_route2_aligned_a1_multistudy_validation_config_v1.json"
DEFAULT_TABLE = ROOT / "docs/paper/route2_v332_three_track_results_table_v1.csv"
DEFAULT_AUDIT = ROOT / "audits/route_a_v3_route2_v332_three_track_results_table_v1.json"

EXTERNAL_EXECUTED = {
    "Optimus5Prime", "FramePool", "RNA-FM_MULTI_MOLECULE_CONVERSION",
    "UTR-LM", "APARENT", "RNA-FM_MULTI_MOLECULE_BOTTLENECK_ADAPTER",
}
EXTERNAL_LIMITED = {"RiNALMo", "Orthrus", "APARENT-Perturb"}
GENERATION_TERMINAL = {
    "random_legal", "greedy", "beam", "genetic", "local_search",
    "generate_then_rerank", "unguided_learned_base_flow_g0",
}
NEURAL_PROFILES = {
    "DELTA_DIAGNOSTIC_0_08M",
    "DELTA_MAIN_2M",
    "NEURAL_MAIN_2M_CANDIDATE_CNN",
    "NEURAL_MAIN_2M_FULL_PAIR_CNN",
    "NEURAL_MAIN_2M_SIAMESE_CNN",
    "NEURAL_MAIN_2M_SMALL_TRANSFORMER",
}

FIELDNAMES = (
    "row_id", "track", "comparison_axis", "comparison_group", "method_id",
    "task_scope", "result_status", "primary_metric_name", "primary_metric_value",
    "secondary_metric_name", "secondary_metric_value", "record_or_source_count",
    "parameter_count", "common_canonical_split", "information_match_status",
    "native_track_status", "headline_horizontal_comparison_eligible",
    "scientific_success_established", "development_test_accessed",
    "new_final_evaluation_accessed", "guided_executed", "primary_limitation",
    "evidence_locator",
)


class ThreeTrackInputError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ThreeTrackInputError(message)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _value(value: Any) -> str:
    return "" if value is None else str(value)


def _row(
    row_id: str,
    track: str,
    axis: str,
    group: str,
    method: str,
    task_scope: str,
    status: str,
    limitation: str,
    evidence: str,
    *,
    primary_name: str = "",
    primary_value: Any = None,
    secondary_name: str = "",
    secondary_value: Any = None,
    count: Any = None,
    parameters: Any = None,
    common_split: bool = False,
    information: str = "NOT_APPLICABLE",
    native_status: str = "NOT_APPLICABLE",
    headline: bool = False,
    guided: bool = False,
) -> dict[str, str]:
    return {
        "row_id": row_id,
        "track": track,
        "comparison_axis": axis,
        "comparison_group": group,
        "method_id": method,
        "task_scope": task_scope,
        "result_status": status,
        "primary_metric_name": primary_name,
        "primary_metric_value": _value(primary_value),
        "secondary_metric_name": secondary_name,
        "secondary_metric_value": _value(secondary_value),
        "record_or_source_count": _value(count),
        "parameter_count": _value(parameters),
        "common_canonical_split": str(common_split).lower(),
        "information_match_status": information,
        "native_track_status": native_status,
        "headline_horizontal_comparison_eligible": str(headline).lower(),
        "scientific_success_established": "false",
        "development_test_accessed": "false",
        "new_final_evaluation_accessed": "false",
        "guided_executed": str(guided).lower(),
        "primary_limitation": limitation,
        "evidence_locator": evidence,
    }


def _validate_inputs(
    snapshot: Mapping[str, Any], inventory: Mapping[str, Any], provenance: Mapping[str, Any],
    external_hpo: Mapping[str, Any], bottleneck: Mapping[str, Any], neural: Mapping[str, Any],
    legacy: Mapping[str, Any], critic: Mapping[str, Any], generation: Sequence[Mapping[str, str]],
    geometry: Mapping[str, Any], aligned_a1: Mapping[str, Any],
) -> None:
    _require(snapshot["status"] == "FROZEN_DEVELOPMENT_TERMINAL_FIELDS_CAPTURED_FOR_THREE_TRACK_TABLE", "three-track snapshot status changed")
    _require(set(snapshot["external_common_task_results"]) == EXTERNAL_EXECUTED, "external snapshot set changed")
    _require(len(snapshot["arch_controlled_prediction_results"]) == 9, "architecture snapshot count changed")
    _require(all(value is False for value in snapshot["protected_outcomes"].values()), "snapshot opened a protected outcome")
    _require(inventory["evaluation_outcomes_accessed"] is False, "baseline inventory accessed Evaluation")
    executed = {row["model_id"]: row for row in inventory["prediction_common_task_adapters"]}
    limited = {row["model_id"]: row for row in inventory["prediction_literature_only_or_not_executed"]}
    _require(set(executed) == EXTERNAL_EXECUTED and set(limited) == EXTERNAL_LIMITED, "external inventory set changed")
    _require({row["model_id"] for row in provenance["artifacts"]} == {"Optimus5Prime", "FramePool", "RNA-FM_MULTI_MOLECULE_CONVERSION", "UTR-LM", "APARENT"}, "external provenance set changed")
    _require(provenance["evaluation_outcomes_accessed"] is False, "provenance accessed Evaluation")
    snap = snapshot["external_common_task_results"]
    _require(external_hpo["selection_pool"] == "DEVELOPMENT_VALIDATION" and external_hpo["development_test_outcomes_accessed"] is False and external_hpo["evaluation_outcomes_accessed"] is False, "external HPO boundary changed")
    _require(external_hpo["rnafm"]["validation_spearman"] == snap["RNA-FM_MULTI_MOLECULE_CONVERSION"]["task_macro_spearman"], "RNA-FM snapshot mismatch")
    _require(external_hpo["utrlm"]["validation_spearman"] == snap["UTR-LM"]["task_macro_spearman"], "UTR-LM snapshot mismatch")
    _require(external_hpo["fixed_native_adapters"]["aparent"]["validation_spearman"] == snap["APARENT"]["task_macro_spearman"], "APARENT snapshot mismatch")
    _require(bottleneck["selected_validation_spearman"] == snap["RNA-FM_MULTI_MOLECULE_BOTTLENECK_ADAPTER"]["task_macro_spearman"] and bottleneck["evaluation_outcomes_accessed"] is False, "bottleneck snapshot mismatch")
    _require(neural["selection_pool"] == "DEVELOPMENT_VALIDATION" and neural["development_test_outcomes_accessed"] is False and neural["evaluation_outcomes_accessed"] is False, "neural HPO boundary changed")
    _require(NEURAL_PROFILES <= set(neural["selections"]), "required neural profiles changed")
    _require(legacy["status"] == "LEGACY_BEST_OBSERVED_VALIDATION_REFERENCE_FROZEN" and legacy["development_test_outcomes_accessed"] is False and legacy["evaluation_outcomes_accessed"] is False, "legacy reference boundary changed")
    control = critic["control_screen"]
    _require(control["status"] == "CRITIC_V2_CONTROLS_DO_NOT_SUPPORT_THREE_FROZEN_SEEDS" and set(control["arms"]) == {"full", "candidate_permutation", "source_only", "source_edit_metadata"}, "Critic V2 control set changed")
    _require(critic["protected_outcomes"]["development_test_outcomes_accessed"] is False and critic["protected_outcomes"]["evaluation_outcomes_accessed"] is False and critic["protected_outcomes"]["guided_generation_authorized"] is False, "Critic V2 protected boundary changed")
    _require({row["method_id"] for row in generation} == GENERATION_TERMINAL and len(generation) == 7, "generation terminal set changed")
    protocol = geometry["protocol_boundary"]
    _require(protocol["action_types_in_scope"] == ["SUB", "STOP"] and protocol["action_types_out_of_scope"] == ["INS", "DEL"] and protocol["guided_xeditflow_run"] is False and protocol["development_test_outcomes_read"] == 0 and protocol["new_final_evaluation_outcomes_read"] == 0, "generation geometry boundary changed")
    _require(aligned_a1["status"] == "A1_ONLY_MULTISTUDY_ABLATION_COMPLETE_INPUTS_FROZEN" and aligned_a1["evaluation_outcomes_accessed"] is False, "aligned-A1 input boundary changed")


def _native_rows(inventory: Mapping[str, Any]) -> list[dict[str, str]]:
    executed = {row["model_id"]: row for row in inventory["prediction_common_task_adapters"]}
    limited = {row["model_id"]: row for row in inventory["prediction_literature_only_or_not_executed"]}
    order = ["Optimus5Prime", "FramePool", "RNA-FM_MULTI_MOLECULE_CONVERSION", "UTR-LM", "APARENT", "RNA-FM_MULTI_MOLECULE_BOTTLENECK_ADAPTER", "RiNALMo", "Orthrus", "APARENT-Perturb", "mRNABERT"]
    rows = []
    for index, model in enumerate(order, start=1):
        if model in executed:
            native = executed[model]["native_track_status"]
            status = native
            limitation = "No original-paper numeric benchmark reproduction is retained in the Route 2 evidence packet."
            if model == "RNA-FM_MULTI_MOLECULE_BOTTLENECK_ADAPTER":
                native = status = "NOT_APPLICABLE_DERIVED_COMMON_TASK_ADAPTER"
                limitation = "This learned bottleneck is a Route 2 adapter, not an original native model task."
        elif model in limited:
            native = status = limited[model]["status"]
            limitation = limited[model]["reason"]
        else:
            native = status = "OFFICIAL_ROUTE2_INTEGRATION_COMPLETE_NATIVE_PAPER_TASK_METRIC_NOT_REPRODUCED"
            limitation = "Official checkpoint/tokenizer integration is real, but no original-paper native-task metric is reproduced in this table."
        rows.append(_row(
            f"A-{index:02d}", "NATIVE_REPRODUCTION", "ORIGINAL_PAPER_TASK_CAPABILITY",
            f"NATIVE::{model}", model, "ORIGINAL_PAPER_TASK_OR_INTERFACE", status,
            limitation, "configs/route_a_v3_route2_external_model_provenance_v1.json;configs/route_a_v3_route2_baseline_inventory_v1.json",
            information="NATIVE_TASK_NOT_CURRENT_CANONICAL_ESTIMAND", native_status=native,
        ))
    return rows


def _common_rows(snapshot: Mapping[str, Any], inventory: Mapping[str, Any], critic: Mapping[str, Any], legacy: Mapping[str, Any]) -> list[dict[str, str]]:
    executed = {row["model_id"]: row for row in inventory["prediction_common_task_adapters"]}
    rows = []
    for index, model in enumerate(sorted(EXTERNAL_EXECUTED), start=1):
        item = snapshot["external_common_task_results"][model]
        inv = executed[model]
        rows.append(_row(
            f"B-{index:02d}", "COMMON_SOURCE_RELATIVE_TASK", "EXTERNAL_COMMON_TASK_ADAPTER",
            item["task"], model, item["task"], inv["common_task_status"],
            "Development-only; native parity and independent external-transfer claims remain unavailable.", item["source_path"],
            primary_name="task_macro_spearman", primary_value=item["task_macro_spearman"],
            secondary_name="mae", secondary_value=item["mae"], count=item["record_count"],
            common_split=True, information="SAME_CANONICAL_SOURCE_CANDIDATE_AND_OUTCOME_NO_EXTRA_LABELS", native_status=inv["native_track_status"], headline=True,
        ))
    limited = {row["model_id"]: row for row in inventory["prediction_literature_only_or_not_executed"]}
    for offset, model in enumerate(sorted(EXTERNAL_LIMITED), start=7):
        item = limited[model]
        rows.append(_row(
            f"B-{offset:02d}", "COMMON_SOURCE_RELATIVE_TASK", "EXTERNAL_COMMON_TASK_ADAPTER",
            "NOT_EXECUTED_OR_TASK_MISMATCH", model, "NOT_AVAILABLE", item["status"],
            item["reason"], "configs/route_a_v3_route2_baseline_inventory_v1.json",
            information="NOT_EXECUTED_ON_COMMON_CANONICAL_TASK", native_status=item["status"],
        ))
    control = critic["control_screen"]
    full = control["arms"]["full"]
    strongest = control["strongest_same_information_baseline"]
    rows.append(_row(
        "B-10", "COMMON_SOURCE_RELATIVE_TASK", "NINE_TASK_PRIMARY_CONTROL_GATE",
        "NINE_TASK_MULTI_STUDY_DEVELOPMENT_VALIDATION", "mRNABERT_CRITIC_V2_FULL",
        "NINE_TASK_MULTI_STUDY_SOURCE_RELATIVE", control["status"],
        "Terminal negative result: the full arm did not beat the strongest same-information baseline.",
        "audits/route_a_v3_route2_critic_v2_control_terminal_no_go_v1.json",
        primary_name="task_macro_spearman", primary_value=full["task_macro_spearman"],
        secondary_name="task_macro_standardized_mae", secondary_value=full["task_macro_standardized_mae"],
        common_split=True, information="TRANSFERABLE_CONTEXT_MATCHED_CONTROL_SCREEN", native_status="NOT_A_NATIVE_RESULT", headline=True,
    ))
    rows.append(_row(
        "B-11", "COMMON_SOURCE_RELATIVE_TASK", "NINE_TASK_PRIMARY_CONTROL_GATE",
        "NINE_TASK_MULTI_STUDY_DEVELOPMENT_VALIDATION", strongest["baseline_id"],
        "NINE_TASK_MULTI_STUDY_SOURCE_RELATIVE", "EXECUTED_TERMINAL_SAME_INFORMATION_BASELINE",
        "Strongest baseline for the Critic V2 frozen gate, not a cross-task winner for external adapter groups.",
        "audits/route_a_v3_route2_critic_v2_control_terminal_no_go_v1.json",
        primary_name="task_macro_spearman", primary_value=strongest["task_macro_spearman"],
        secondary_name="task_macro_standardized_mae", secondary_value=strongest["task_macro_standardized_mae"],
        common_split=True, information="TRANSFERABLE_CONTEXT_MATCHED_CONTROL_SCREEN", native_status="NOT_A_NATIVE_RESULT", headline=True,
    ))
    best = legacy["strongest"]
    rows.append(_row(
        "B-12", "COMMON_SOURCE_RELATIVE_TASK", "LEGACY_OBSERVED_REFERENCE",
        "NINE_TASK_MULTI_STUDY_DEVELOPMENT_VALIDATION", best["baseline_id"],
        "NINE_TASK_MULTI_STUDY_SOURCE_RELATIVE", legacy["status"],
        legacy["same_information_caveat"], "audits/route_a_v3_route2_legacy_baseline_task_macro_replay_v1.json",
        primary_name="task_macro_spearman", primary_value=best["task_macro_spearman"],
        secondary_name="task_macro_standardized_mae", secondary_value=best["common_train_robust_task_macro_standardized_mae"],
        count=legacy["record_count"], parameters=best["parameter_count"], common_split=True,
        information="FULL_CONTEXT_NOT_SAME_INFORMATION_AS_CRITIC_V2", native_status="NOT_A_NATIVE_RESULT", headline=False,
    ))
    return rows


def _arch_rows(snapshot: Mapping[str, Any], neural: Mapping[str, Any], critic: Mapping[str, Any], generation: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows = []
    axis_by_snapshot = {
        "candidate_only_ridge": "ABSOLUTE_VS_SOURCE_RELATIVE;CANDIDATE_ONLY_VS_SOURCE_PLUS_CANDIDATE",
        "absolute_candidate_difference_ridge": "ABSOLUTE_VS_SOURCE_RELATIVE",
        "delta_main_2m_sequence_and_region_only_lr3e4": "NO_CONTEXT_VS_CONTEXT_AWARE",
        "delta_main_2m_5utr_multistudy_lr3e4": "REGION_SUBSET;SINGLE_VS_MULTI_STUDY_CONTEXT",
        "delta_main_2m_3utr_multistudy_lr3e4": "REGION_SUBSET;SINGLE_VS_MULTI_STUDY_CONTEXT",
        "delta_main_2m_a1_only_multistudy_lr3e4": "A1_ONLY_VS_A1_PLUS_A2",
        "delta_main_2m_gse200304_only_lr3e4": "SINGLE_STUDY_VS_MULTI_STUDY",
        "delta_main_2m_study_scale_calibration_lr3e4": "NO_CONTEXT_VS_CONTEXT_AWARE;STUDY_CALIBRATION",
        "delta_main_2m_without_study_balance_lr3e4": "STUDY_BALANCE_ABLATION",
    }
    for index, (method, item) in enumerate(snapshot["arch_controlled_prediction_results"].items(), start=1):
        limitation = "Development validation only."
        if item["metric_scope"] != "NINE_TASK_MACRO":
            limitation = "Subset-pooled metric has a different task/record scope and cannot be ranked directly against the nine-task macro rows."
        if method == "delta_main_2m_a1_only_multistudy_lr3e4":
            limitation = "Aligned A1 comparison inputs are frozen, but the aligned result was not materialized; direct A1-only versus A1+A2 ranking is prohibited."
        rows.append(_row(
            f"C-{index:02d}", "ARCH_CONTROLLED", axis_by_snapshot[method],
            f"PREDICTION::{item['metric_scope']}", method, item["metric_scope"],
            "EXECUTED_DEVELOPMENT_VALIDATION", limitation, item["source_path"],
            primary_name="task_macro_spearman" if item["metric_scope"] == "NINE_TASK_MACRO" else "pooled_spearman",
            primary_value=item["task_macro_spearman"], secondary_name=item["secondary_metric_name"],
            secondary_value=item["secondary_metric_value"], count=item["record_count"], common_split=True,
            information="FROZEN_SUBSET_OR_COMMON_SPLIT_DECLARED_PER_ROW",
        ))
    profile_order = [
        ("DELTA_DIAGNOSTIC_0_08M", "PARAMETER_SCALE;SCRATCH_ARCHITECTURE"),
        ("DELTA_MAIN_2M", "ABSOLUTE_VS_SOURCE_RELATIVE;PARAMETER_SCALE;CONTEXT_AWARE"),
        ("NEURAL_MAIN_2M_CANDIDATE_CNN", "CANDIDATE_ONLY_VS_SOURCE_PLUS_CANDIDATE"),
        ("NEURAL_MAIN_2M_FULL_PAIR_CNN", "CANDIDATE_ONLY_VS_SOURCE_PLUS_CANDIDATE;PAIR_ARCHITECTURE"),
        ("NEURAL_MAIN_2M_SIAMESE_CNN", "PAIR_ARCHITECTURE"),
        ("NEURAL_MAIN_2M_SMALL_TRANSFORMER", "PAIR_ARCHITECTURE"),
    ]
    for offset, (profile, axis) in enumerate(profile_order, start=10):
        selection = neural["selections"][profile]
        selected = selection["all_trials_ranked"][0]
        _require(selected["trial_id"] == selection["selected_trial_id"], f"selected neural trial ordering changed: {profile}")
        rows.append(_row(
            f"C-{offset:02d}", "ARCH_CONTROLLED", axis, "PREDICTION::NINE_TASK_HPO_SELECTION",
            selected["baseline_id"], "NINE_TASK_MULTI_STUDY_SOURCE_RELATIVE", "EXECUTED_DEVELOPMENT_VALIDATION_HPO_SELECTED",
            "Architecture/parameter-band Development result; causal attribution is limited to rows with matched information and parameter band.",
            selected["validation_evaluation_path"], primary_name="task_macro_spearman",
            primary_value=selected["task_macro_spearman"], secondary_name="source_macro_mae_raw_heterogeneous_scale",
            secondary_value=selected["source_macro_mae"], parameters=selected["parameter_count"], common_split=True,
            information="FULL_CONTEXT_COMMON_SPLIT;PARAMETER_BAND_DECLARED",
        ))
    for offset, arm_id in enumerate(("full", "candidate_permutation", "source_only", "source_edit_metadata"), start=16):
        arm = critic["control_screen"]["arms"][arm_id]
        rows.append(_row(
            f"C-{offset:02d}", "ARCH_CONTROLLED", "CANDIDATE_SIGNAL_CONTROL",
            "CRITIC_V2_SAME_BUDGET_CONTROL_SCREEN", f"critic_v2_{arm_id}",
            "NINE_TASK_TRANSFERABLE_CONTEXT", "EXECUTED_TERMINAL_CRITIC_V2_CONTROL",
            "Matched control-screen result; current status does not support confirmation seeds or guidance.",
            "audits/route_a_v3_route2_critic_v2_control_terminal_no_go_v1.json",
            primary_name="task_macro_spearman", primary_value=arm["task_macro_spearman"],
            secondary_name="task_macro_standardized_mae", secondary_value=arm["task_macro_standardized_mae"],
            common_split=True, information="MATCHED_BUDGET_WITH_ARM_SPECIFIC_INFORMATION_REMOVAL",
        ))
    generation_by_id = {row["method_id"]: row for row in generation}
    generation_order = ["random_legal", "greedy", "beam", "genetic", "local_search", "generate_then_rerank", "unguided_learned_base_flow_g0"]
    for offset, method in enumerate(generation_order, start=20):
        item = generation_by_id[method]
        rows.append(_row(
            f"C-{offset:02d}", "ARCH_CONTROLLED", "RANDOM_SEARCH_VS_FLOW;RERANK_VS_SAMPLING;SAMPLER_COMPARISON",
            "GENERATION_MATCHED_DEVELOPMENT_SUB_STOP", method, "891_SOURCE_OPEN_GENERATED_SUPPORT",
            "EXECUTED_TERMINAL_MATCHED_DEVELOPMENT", "Independent-evaluator Development selection only; no biological-improvement claim.",
            "docs/paper/route2_v332_generation_baseline_table_v1.csv",
            primary_name="source_macro_independent_evaluator_max_uplift_over_source",
            primary_value=item["source_macro_independent_evaluator_max_uplift_over_source"],
            secondary_name="mean_total_forward_equivalents_per_source",
            secondary_value=item["mean_total_forward_equivalents_per_source"], count=item["source_count"],
            common_split=True, information="MATCHED_SOURCE_ACTION_CANDIDATE_AND_FORWARD_BUDGET_WITH_DECLARED_NO_CRITIC_FLOW_EXCEPTION",
        ))
    for row_id, method, axis in [
        ("C-27", "first_order_rate_guidance", "UNGUIDED_VS_FIRST_ORDER_VS_POTENTIAL_GUIDANCE"),
        ("C-28", "frozen_critic_xeditflow", "UNGUIDED_VS_FIRST_ORDER_VS_POTENTIAL_GUIDANCE;RERANK_VS_GUIDED_SAMPLING"),
    ]:
        rows.append(_row(
            row_id, "ARCH_CONTROLLED", axis, "GENERATION_GUIDANCE_CLOSED", method,
            "891_SOURCE_OPEN_GENERATED_SUPPORT", "NOT_RUN_CRITIC_V2_NO_GO",
            "Critic V2 terminal NO-GO prohibits guided execution for the current cohort.",
            "audits/route_a_v3_route2_critic_v2_control_terminal_no_go_v1.json",
            common_split=True, information="PROSPECTIVE_MATCHED_PROTOCOL_NOT_EXECUTED", guided=False,
        ))
    rows.append(_row(
        "C-29", "ARCH_CONTROLLED", "SCRATCH_VS_FROZEN_EMBEDDING", "PREDICTION_CAUSAL_ATTRIBUTION_GAP",
        "scratch_vs_frozen_embedding_matched_contrast", "NINE_TASK_MULTI_STUDY_SOURCE_RELATIVE",
        "NOT_CAUSALLY_IDENTIFIABLE_FROM_CURRENT_TERMINAL_RUNS",
        "Scratch and frozen-embedding runs differ in parameterization, metadata/optimization regime or HPO context; no matched causal contrast is claimed.",
        "audits/route_a_v3_route2_legacy_baseline_task_macro_replay_v1.json;audits/route_a_v3_route2_critic_v2_control_terminal_no_go_v1.json",
        common_split=True, information="NOT_MATCHED_FOR_CAUSAL_ATTRIBUTION",
    ))
    rows.append(_row(
        "C-30", "ARCH_CONTROLLED", "GENERIC_TRUNK_VS_REGION_ADAPTER", "GENERATION_CAUSAL_ATTRIBUTION_GAP",
        "generic_trunk_vs_region_adapter_matched_contrast", "DEVELOPMENT_SUB_STOP",
        "NOT_TERMINAL_MATCHED_CONTRAST",
        "The current terminal generation suite does not isolate a generic trunk from the region adapter under an otherwise matched run.",
        "audits/route_a_v3_route2_generation_action_space_geometry_v1.json",
        common_split=True, information="NOT_EXECUTED_AS_MATCHED_ABLATION",
    ))
    return rows


def _validate_rows(rows: Sequence[Mapping[str, str]]) -> None:
    _require(len(rows) == 52 and len({row["row_id"] for row in rows}) == 52, "three-track table must have 52 unique rows")
    _require(Counter(row["track"] for row in rows) == {"NATIVE_REPRODUCTION": 10, "COMMON_SOURCE_RELATIVE_TASK": 12, "ARCH_CONTROLLED": 30}, "three-track row counts changed")
    native = [row for row in rows if row["track"] == "NATIVE_REPRODUCTION"]
    _require(all(not row["primary_metric_value"] and not row["secondary_metric_value"] for row in native), "native status-only track acquired a numeric result")
    _require(all(row["headline_horizontal_comparison_eligible"] == "false" for row in native), "native row entered headline comparison")
    common_numeric = [row for row in rows if row["track"] == "COMMON_SOURCE_RELATIVE_TASK" and row["primary_metric_value"]]
    _require(len(common_numeric) == 9, "common-task numeric row count changed")
    _require(sum(row["headline_horizontal_comparison_eligible"] == "true" for row in rows) == 8, "headline-eligible common-task count changed")
    arch_numeric = [row for row in rows if row["track"] == "ARCH_CONTROLLED" and row["primary_metric_value"]]
    _require(len(arch_numeric) == 26, "architecture-controlled numeric row count changed")
    _require(all(row["scientific_success_established"] == "false" for row in rows), "scientific success was overcalled")
    _require(all(row["development_test_accessed"] == "false" and row["new_final_evaluation_accessed"] == "false" and row["guided_executed"] == "false" for row in rows), "protected outcome entered three-track table")


def build_table(
    *, snapshot_path: Path = DEFAULT_SNAPSHOT, inventory_path: Path = DEFAULT_INVENTORY,
    provenance_path: Path = DEFAULT_PROVENANCE, external_hpo_path: Path = DEFAULT_EXTERNAL_HPO,
    bottleneck_path: Path = DEFAULT_BOTTLENECK, neural_path: Path = DEFAULT_NEURAL,
    legacy_path: Path = DEFAULT_LEGACY, critic_path: Path = DEFAULT_CRITIC,
    generation_path: Path = DEFAULT_GENERATION, geometry_path: Path = DEFAULT_GEOMETRY,
    aligned_a1_path: Path = DEFAULT_ALIGNED_A1, table_path: Path = DEFAULT_TABLE,
    audit_path: Path = DEFAULT_AUDIT, overwrite: bool = False,
) -> dict[str, Any]:
    paths = [snapshot_path, inventory_path, provenance_path, external_hpo_path, bottleneck_path, neural_path, legacy_path, critic_path, generation_path, geometry_path, aligned_a1_path, table_path, audit_path]
    snapshot_path, inventory_path, provenance_path, external_hpo_path, bottleneck_path, neural_path, legacy_path, critic_path, generation_path, geometry_path, aligned_a1_path, table_path, audit_path = [path.resolve() for path in paths]
    for path in (table_path, audit_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing three-track artifact: {path}")
    snapshot = _load_json(snapshot_path); inventory = _load_json(inventory_path); provenance = _load_json(provenance_path)
    external_hpo = _load_json(external_hpo_path); bottleneck = _load_json(bottleneck_path); neural = _load_json(neural_path)
    legacy = _load_json(legacy_path); critic = _load_json(critic_path); generation = _load_csv(generation_path)
    geometry = _load_json(geometry_path); aligned_a1 = _load_json(aligned_a1_path)
    _validate_inputs(snapshot, inventory, provenance, external_hpo, bottleneck, neural, legacy, critic, generation, geometry, aligned_a1)
    rows = _native_rows(inventory) + _common_rows(snapshot, inventory, critic, legacy) + _arch_rows(snapshot, neural, critic, generation)
    _validate_rows(rows)
    table_path.parent.mkdir(parents=True, exist_ok=True); audit_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES); writer.writeheader(); writer.writerows(rows)
    audit = {
        "schema_version": "route_a_v3_route2_v332_three_track_results_table.v1",
        "status": "THREE_TRACK_REPORTING_TABLE_RENDERED_EXECUTION_GAPS_DECLARED",
        "authority": {
            "scientific_contract": "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/提示词/mrna 数据gate转向后的合同.md",
            "execution_protocol": "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/提示词/mrna V3.3.2 执行提示词.md",
        },
        "table_path": _display(table_path),
        "input_snapshot": _display(snapshot_path),
        "row_count": 52,
        "track_counts": {"NATIVE_REPRODUCTION": 10, "COMMON_SOURCE_RELATIVE_TASK": 12, "ARCH_CONTROLLED": 30},
        "numeric_result_counts": {"NATIVE_REPRODUCTION": 0, "COMMON_SOURCE_RELATIVE_TASK": 9, "ARCH_CONTROLLED": 26},
        "headline_horizontal_comparison_eligible_rows": 8,
        "reporting_table_complete": True,
        "three_track_benchmark_execution_complete": False,
        "execution_gaps": [
            "No original-paper numeric native reproduction is retained for any Track A row.",
            "The aligned A1-only versus A1+A2 inputs are frozen but the aligned result is not materialized.",
            "Scratch versus frozen embedding is not a matched causal contrast in current terminal runs.",
            "Generic trunk versus region adapter is not a terminal matched generation contrast.",
            "First-order and frozen-critic guided generation remain unrun after Critic V2 NO-GO.",
        ],
        "interpretation_rules": {
            "native_results_enter_current_headline": False,
            "common_task_results_may_be_compared_only_within_same_task_scope": True,
            "architecture_controlled_results_enter_headline_horizontal_ranking": False,
            "subset_pooled_metrics_ranked_against_nine_task_macro": False,
            "task_mismatch_models_enter_headline": False,
        },
        "protected_outcomes": {"development_test_read": False, "new_final_evaluation_read": False, "guided_xeditflow_run": False, "generated_candidates_opened": False},
        "scientific_claim_status": "NOT_ESTABLISHED",
        "submission_ready": False,
        "new_training_attempt_created": False,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT); parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE); parser.add_argument("--external-hpo", type=Path, default=DEFAULT_EXTERNAL_HPO)
    parser.add_argument("--bottleneck", type=Path, default=DEFAULT_BOTTLENECK); parser.add_argument("--neural", type=Path, default=DEFAULT_NEURAL)
    parser.add_argument("--legacy", type=Path, default=DEFAULT_LEGACY); parser.add_argument("--critic", type=Path, default=DEFAULT_CRITIC)
    parser.add_argument("--generation", type=Path, default=DEFAULT_GENERATION); parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--aligned-a1", type=Path, default=DEFAULT_ALIGNED_A1); parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT); parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(json.dumps(build_table(snapshot_path=args.snapshot, inventory_path=args.inventory, provenance_path=args.provenance, external_hpo_path=args.external_hpo, bottleneck_path=args.bottleneck, neural_path=args.neural, legacy_path=args.legacy, critic_path=args.critic, generation_path=args.generation, geometry_path=args.geometry, aligned_a1_path=args.aligned_a1, table_path=args.table, audit_path=args.audit, overwrite=args.overwrite), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
