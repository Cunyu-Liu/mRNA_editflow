#!/usr/bin/env python3
"""Build the Route 2 V3.3.2 Prediction/Generation baseline inventory matrix.

This is deliberately a coverage/status matrix, not a result table.  It reads
only frozen inventories, terminal Development audits, and the already-rendered
seven-method generation table.  It never opens Development TEST, a new final
Evaluation outcome, generated candidates, or model checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_INVENTORY = ROOT / "configs/route_a_v3_route2_baseline_inventory_v1.json"
DEFAULT_CLASSICAL_CONFIG = ROOT / "configs/route_a_v3_route2_classical_hpo_v1.json"
DEFAULT_ELASTIC_AUDIT = ROOT / "audits/route_a_v3_route2_classical_elastic_recovery_v1.json"
DEFAULT_NEURAL_AUDIT = ROOT / "audits/route_a_v3_route2_neural_hpo_selection_v1.json"
DEFAULT_CRITIC_AUDIT = ROOT / "audits/route_a_v3_route2_critic_v2_control_terminal_no_go_v1.json"
DEFAULT_GENERATION_TABLE = ROOT / "docs/paper/route2_v332_generation_baseline_table_v1.csv"
DEFAULT_GEOMETRY_AUDIT = ROOT / "audits/route_a_v3_route2_generation_action_space_geometry_v1.json"
DEFAULT_MINIMUM_PACKAGE_TABLE = ROOT / "docs/paper/route2_v332_minimum_benchmark_package_table_v1.csv"
DEFAULT_PAIRWISE_CONFIG = ROOT / "configs/route_a_v3_route2_delta_main_2m_huber_plus_pairwise_lr3e4_v1.json"
DEFAULT_LISTWISE_CONFIG = ROOT / "configs/route_a_v3_route2_delta_main_2m_huber_plus_listwise_lr3e4_v1.json"
DEFAULT_TABLE = ROOT / "docs/paper/route2_v332_baseline_matrix_v1.csv"
DEFAULT_AUDIT = ROOT / "audits/route_a_v3_route2_v332_baseline_matrix_v1.json"

EXPECTED_CLASSICAL_IDS = {
    "global_mean",
    "majority_sign",
    "context_only_mean",
    "source_group_mean",
    "kmer_context_ridge",
    "source_centered_ridge",
    "source_only_ridge",
    "candidate_only_ridge",
    "edit_position_only_ridge",
    "ref_alt_only_ridge",
    "gc_mfe_motif_ridge",
    "candidate_permutation_ridge",
    "edit_context_elastic_net",
    "xgboost_full",
    "absolute_candidate_difference_ridge",
}
EXPECTED_NEURAL_PROFILES = {
    "DELTA_DIAGNOSTIC_0_08M",
    "DELTA_MAIN_2M",
    "DELTA_MEDIUM_0_5M",
    "NEURAL_MAIN_2M_CANDIDATE_CNN",
    "NEURAL_MAIN_2M_FULL_PAIR_CNN",
    "NEURAL_MAIN_2M_SIAMESE_CNN",
    "NEURAL_MAIN_2M_SMALL_TRANSFORMER",
    "NEURAL_MEDIUM_0_5M_CANDIDATE_CNN",
    "NEURAL_MEDIUM_0_5M_FULL_PAIR_CNN",
    "NEURAL_MEDIUM_0_5M_SIAMESE_CNN",
    "NEURAL_MEDIUM_0_5M_SMALL_TRANSFORMER",
}
EXPECTED_EXTERNAL_EXECUTED = {
    "Optimus5Prime",
    "FramePool",
    "RNA-FM_MULTI_MOLECULE_CONVERSION",
    "UTR-LM",
    "APARENT",
    "RNA-FM_MULTI_MOLECULE_BOTTLENECK_ADAPTER",
}
EXPECTED_EXTERNAL_LIMITED = {"RiNALMo", "Orthrus", "APARENT-Perturb"}
EXPECTED_GENERATION_TERMINAL = {
    "random_legal",
    "greedy",
    "beam",
    "genetic",
    "local_search",
    "generate_then_rerank",
    "unguided_learned_base_flow_g0",
}

FIELDNAMES = (
    "matrix_row_id",
    "track",
    "baseline_family",
    "contract_requirement",
    "implementation_id",
    "task_interface",
    "input_information",
    "execution_status_v332",
    "contract_coverage_status",
    "current_scope",
    "native_track_status",
    "terminal_evidence_class",
    "headline_eligible_now",
    "development_test_accessed",
    "new_final_evaluation_accessed",
    "guided_required",
    "guided_executed",
    "primary_limitation",
    "evidence_locator",
)


class BaselineMatrixInputError(RuntimeError):
    """A frozen baseline input no longer matches its declared V3.3.2 shape."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BaselineMatrixInputError(message)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row(
    row_id: str,
    track: str,
    family: str,
    requirement: str,
    implementation: str,
    interface: str,
    information: str,
    status: str,
    coverage: str,
    scope: str,
    limitation: str,
    evidence: str,
    *,
    native: str = "NOT_APPLICABLE",
    evidence_class: str = "FROZEN_DEVELOPMENT_ARTIFACT",
    guided_required: bool = False,
    guided_executed: bool = False,
) -> dict[str, str]:
    return {
        "matrix_row_id": row_id,
        "track": track,
        "baseline_family": family,
        "contract_requirement": requirement,
        "implementation_id": implementation,
        "task_interface": interface,
        "input_information": information,
        "execution_status_v332": status,
        "contract_coverage_status": coverage,
        "current_scope": scope,
        "native_track_status": native,
        "terminal_evidence_class": evidence_class,
        "headline_eligible_now": "false",
        "development_test_accessed": "false",
        "new_final_evaluation_accessed": "false",
        "guided_required": str(guided_required).lower(),
        "guided_executed": str(guided_executed).lower(),
        "primary_limitation": limitation,
        "evidence_locator": evidence,
    }


def _validate_inputs(
    inventory: Mapping[str, Any],
    classical: Sequence[Mapping[str, Any]],
    elastic: Mapping[str, Any],
    neural: Mapping[str, Any],
    critic: Mapping[str, Any],
    generation_rows: Sequence[Mapping[str, str]],
    geometry: Mapping[str, Any],
    minimum_rows: Sequence[Mapping[str, str]],
    pairwise: Mapping[str, Any],
    listwise: Mapping[str, Any],
) -> None:
    _require(
        inventory["status"]
        == "DEVELOPMENT_BASELINE_RESULTS_PARTIALLY_COMPLETED_PRIMARY_MRNABERT_HPO_RUNNING",
        "the frozen baseline inventory no longer carries the expected stale running label",
    )
    _require(inventory["evaluation_outcomes_accessed"] is False, "baseline inventory accessed Evaluation")
    _require(
        {row["baseline_id"] for row in classical} == EXPECTED_CLASSICAL_IDS
        and len(classical) == 15,
        "classical baseline inventory changed",
    )
    _require(
        elastic["status"] == "MANDATORY_ELASTIC_NET_RECOVERED_ON_DEVELOPMENT_VALIDATION"
        and elastic["development_test_outcomes_accessed"] is False
        and elastic["evaluation_outcomes_accessed"] is False,
        "elastic-net terminal/protected-outcome boundary changed",
    )
    _require(
        neural["status"] == "NEURAL_HPO_LEARNING_RATES_FROZEN_BY_PROFILE"
        and neural["selection_pool"] == "DEVELOPMENT_VALIDATION"
        and neural["development_test_outcomes_accessed"] is False
        and neural["evaluation_outcomes_accessed"] is False,
        "neural HPO terminal/protected-outcome boundary changed",
    )
    _require(set(neural["selections"]) == EXPECTED_NEURAL_PROFILES, "neural HPO profile set changed")
    _require(
        {row["model_id"] for row in inventory["prediction_common_task_adapters"]}
        == EXPECTED_EXTERNAL_EXECUTED,
        "executed common-task adapter set changed",
    )
    _require(
        {row["model_id"] for row in inventory["prediction_literature_only_or_not_executed"]}
        == EXPECTED_EXTERNAL_LIMITED,
        "literature-only/task-mismatch adapter set changed",
    )
    control = critic["control_screen"]
    _require(
        control["status"] == "CRITIC_V2_CONTROLS_DO_NOT_SUPPORT_THREE_FROZEN_SEEDS"
        and control["supports_three_frozen_seeds"] is False
        and all(arm["ledger_status"] == "COMPLETED" for arm in control["arms"].values())
        and set(control["arms"]) == {"full", "candidate_permutation", "source_only", "source_edit_metadata"},
        "Critic V2 terminal control screen changed",
    )
    _require(
        critic["protected_outcomes"]["development_test_outcomes_accessed"] is False
        and critic["protected_outcomes"]["evaluation_outcomes_accessed"] is False
        and critic["protected_outcomes"]["guided_generation_authorized"] is False,
        "Critic V2 protected-outcome boundary changed",
    )
    _require(
        {row["method_id"] for row in generation_rows} == EXPECTED_GENERATION_TERMINAL
        and len(generation_rows) == 7,
        "terminal matched generation method set changed",
    )
    _require(
        all(float(row["hard_legality_rate"]) == 1.0 for row in generation_rows)
        and all(int(row["edit_budget_violation_count"]) == 0 for row in generation_rows)
        and all(int(row["candidate_budget_violation_count"]) == 0 for row in generation_rows),
        "terminal generation legality/budget boundary changed",
    )
    protocol = geometry["protocol_boundary"]
    _require(
        geometry["status"] == "DEVELOPMENT_BENCHMARK_ACTION_SPACE_GEOMETRY_COMPLETE"
        and protocol["action_types_in_scope"] == ["SUB", "STOP"]
        and protocol["action_types_out_of_scope"] == ["INS", "DEL"]
        and protocol["candidate_support_mode"] == "OPEN_GENERATED_SUPPORT"
        and protocol["generated_candidates_grant_canonical_credit"] is False
        and protocol["guided_xeditflow_run"] is False
        and protocol["development_test_outcomes_read"] == 0
        and protocol["new_final_evaluation_outcomes_read"] == 0,
        "generation action-space/protected-outcome boundary changed",
    )
    minimum = {row["requirement_id"]: row for row in minimum_rows}
    _require(
        minimum["MBP-08"]["status"] == "COMPLETE_DEVELOPMENT_ONLY"
        and minimum["MBP-09"]["status"] == "COMPLETE_DEVELOPMENT_ONLY"
        and minimum["MBP-10"]["status"] == "PARTIAL_GUIDED_NOT_AUTHORIZED",
        "minimum-package baseline status changed",
    )
    _require(
        pairwise["loss_kind"] == "huber_plus_pairwise"
        and listwise["loss_kind"] == "huber_plus_listwise",
        "pairwise/listwise configuration identity changed",
    )


def _build_rows(
    inventory: Mapping[str, Any],
    generation_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    classical_evidence = "configs/route_a_v3_route2_classical_hpo_v1.json;MBP-08"
    internal = [
        ("P-IC-01", "global mean", "global_mean", "no sequence; global target mean", "EXECUTED_DEVELOPMENT_VALIDATION", "SATISFIED_STANDALONE", "Global mean is a sanity control, not a transfer claim."),
        ("P-IC-02", "study mean", "context_only_mean", "study + biological context + endpoint", "MAPPED_COMPOSITE_DEVELOPMENT_CONTROL", "SATISFIED_COMPOSITE_WITH_LIMIT", "No standalone study-only mean is retained; the executed grouped mean is finer than the named requirement."),
        ("P-IC-03", "source mean", "source_group_mean", "study + source + biological context + endpoint", "EXECUTED_DEVELOPMENT_VALIDATION", "SATISFIED_STANDALONE", "Source-group mean cannot establish candidate-specific signal."),
        ("P-IC-04", "majority sign", "majority_sign", "target sign prevalence only", "EXECUTED_DEVELOPMENT_VALIDATION", "SATISFIED_STANDALONE", "Sign prevalence is a sanity control."),
        ("P-IC-05", "edit-position only", "edit_position_only_ridge", "edit position only", "EXECUTED_DEVELOPMENT_VALIDATION", "SATISFIED_STANDALONE", "Position-only signal does not establish sequence understanding."),
        ("P-IC-06", "ref-to-alt identity only", "ref_alt_only_ridge", "reference and alternate nucleotide identity", "EXECUTED_DEVELOPMENT_VALIDATION", "SATISFIED_STANDALONE", "Identity-only signal omits broader sequence context."),
        ("P-IC-07", "GC/MFE/motif", "gc_mfe_motif_ridge", "GC, MFE, and motif summaries", "EXECUTED_DEVELOPMENT_VALIDATION", "SATISFIED_STANDALONE", "Hand-crafted summaries are a diagnostic baseline."),
        ("P-IC-08", "anchor/source only", "source_only_ridge|critic_v2_source_only", "source/anchor and metadata; no candidate sequence", "EXECUTED_DEVELOPMENT_AND_TERMINAL_CONTROL", "SATISFIED_STANDALONE", "Candidate-specific predictive signal is intentionally removed."),
        ("P-IC-09", "candidate only", "candidate_only_ridge", "candidate sequence and metadata; no explicit source", "EXECUTED_DEVELOPMENT_VALIDATION", "SATISFIED_STANDALONE", "No explicit source-relative comparison is available."),
        ("P-IC-10", "candidate permutation", "candidate_permutation_ridge|critic_v2_candidate_permutation", "within-stratum permuted candidate", "EXECUTED_DEVELOPMENT_AND_TERMINAL_CONTROL", "SATISFIED_STANDALONE", "Permutation destroys the intended candidate pairing while preserving the declared stratum."),
        ("P-IC-11", "study/context only", "context_only_mean", "study + biological context + endpoint", "EXECUTED_DEVELOPMENT_VALIDATION", "SATISFIED_STANDALONE", "No source or candidate sequence is used."),
    ]
    rows = [
        _row(
            row_id, "PREDICTION", "INTERNAL_CONTROL", requirement, implementation,
            "COMMON_SOURCE_RELATIVE_TASK", information, status, coverage,
            "DEVELOPMENT_VALIDATION_ONLY", limitation,
            classical_evidence + (";audits/route_a_v3_route2_critic_v2_control_terminal_no_go_v1.json" if "critic_v2" in implementation else ""),
            evidence_class="TERMINAL_CONTROL_AUDIT" if "critic_v2" in implementation else "MINIMUM_PACKAGE_AGGREGATE_PLUS_CONFIG",
        )
        for row_id, requirement, implementation, information, status, coverage, limitation in internal
    ]

    classical = [
        ("P-CL-01", "k-mer ridge", "kmer_context_ridge", "source/candidate k-mer and context features", "EXECUTED_DEVELOPMENT_VALIDATION", "SATISFIED_STANDALONE", "Linear source-relative comparator only."),
        ("P-CL-02", "edit/context elastic net", "edit_context_elastic_net", "full edit/context feature matrix", "EXECUTED_TERMINAL_DEVELOPMENT_VALIDATION", "SATISFIED_STANDALONE", "Recovered solver is terminal on Development validation only."),
        ("P-CL-03", "source-centered linear", "source_centered_ridge", "source-centered sequence/context features", "EXECUTED_DEVELOPMENT_VALIDATION", "SATISFIED_STANDALONE", "Linear source-centered comparator only."),
        ("P-CL-04", "XGBoost", "xgboost_full", "full edit/context feature matrix", "EXECUTED_DEVELOPMENT_VALIDATION", "SATISFIED_STANDALONE", "Tree comparator remains Development-only."),
        ("P-CL-05", "absolute-candidate predictor", "absolute_candidate_difference_ridge::candidate_component", "absolute candidate features", "EXECUTED_COMPONENT_WITHIN_ABSOLUTE_DIFFERENCE_PIPELINE", "SATISFIED_COMPONENT_WITH_LIMIT", "The absolute candidate score is retained only as a component; no standalone delta headline is reported."),
        ("P-CL-06", "absolute(candidate)-absolute(source)", "absolute_candidate_difference_ridge", "absolute candidate and source features", "EXECUTED_DEVELOPMENT_VALIDATION", "SATISFIED_STANDALONE", "Only the source-relative difference is used for Route 2 comparison."),
    ]
    for row_id, requirement, implementation, information, status, coverage, limitation in classical:
        evidence = classical_evidence
        evidence_class = "MINIMUM_PACKAGE_AGGREGATE_PLUS_CONFIG"
        if implementation == "edit_context_elastic_net":
            evidence += ";audits/route_a_v3_route2_classical_elastic_recovery_v1.json"
            evidence_class = "TERMINAL_DEVELOPMENT_AUDIT"
        rows.append(_row(
            row_id, "PREDICTION", "CLASSICAL", requirement, implementation,
            "COMMON_SOURCE_RELATIVE_TASK", information, status, coverage,
            "DEVELOPMENT_VALIDATION_ONLY", limitation, evidence,
            evidence_class=evidence_class,
        ))

    neural_evidence = "audits/route_a_v3_route2_neural_hpo_selection_v1.json;MBP-08"
    neural = [
        ("P-NN-01", "small CNN", "neural_main_2m_candidate_cnn", "candidate sequence + shared task metadata", "EXECUTED_DEVELOPMENT_VALIDATION", "SATISFIED_STANDALONE", "Candidate-only CNN lacks an explicit source branch."),
        ("P-NN-02", "small Transformer", "neural_main_2m_small_transformer", "source + candidate + signed/absolute representation difference", "EXECUTED_DEVELOPMENT_VALIDATION", "SATISFIED_STANDALONE", "Small Transformer comparator is Development-only."),
        ("P-NN-03", "anchored CNN", "delta_main_2m::delta_anchored_position_aware_antisymmetric", "source + candidate + edit identity + normalized position", "EXECUTED_DEVELOPMENT_VALIDATION", "SATISFIED_STANDALONE", "This is the non-pretrained anchored CNN family, not the mRNABERT primary critic."),
        ("P-NN-04", "siamese source/candidate encoder", "neural_main_2m_siamese_cnn", "shared source/candidate CNN encoders + difference features", "EXECUTED_DEVELOPMENT_VALIDATION", "SATISFIED_STANDALONE", "Siamese comparator is Development-only."),
        ("P-NN-05", "full-pair encoder", "neural_main_2m_full_pair_cnn", "joint source/candidate/edit/position channels", "EXECUTED_DEVELOPMENT_VALIDATION", "SATISFIED_STANDALONE", "Full-pair comparator is Development-only."),
        ("P-NN-06", "ordinal pairwise ranker", "delta_main_2m_huber_plus_pairwise", "anchored pair encoder + pairwise ranking loss", "CONFIGURED_NOT_TERMINAL_INDEPENDENT_BASELINE", "CONFIGURED_NOT_TERMINAL", "A runnable loss configuration exists, but no independent terminal selection is present in the frozen neural HPO audit."),
        ("P-NN-07", "listwise ranker", "delta_main_2m_huber_plus_listwise", "anchored pair encoder + listwise ranking loss", "CONFIGURED_NOT_TERMINAL_INDEPENDENT_BASELINE", "CONFIGURED_NOT_TERMINAL", "A runnable loss configuration exists, but no independent terminal selection is present in the frozen neural HPO audit."),
    ]
    for row_id, requirement, implementation, information, status, coverage, limitation in neural:
        evidence = neural_evidence
        evidence_class = "TERMINAL_DEVELOPMENT_HPO_SELECTION"
        scope = "DEVELOPMENT_VALIDATION_ONLY"
        if status.startswith("CONFIGURED"):
            config_name = "pairwise" if "pairwise" in implementation else "listwise"
            evidence = f"configs/route_a_v3_route2_delta_main_2m_huber_plus_{config_name}_lr3e4_v1.json;audits/route_a_v3_route2_neural_hpo_selection_v1.json"
            evidence_class = "CONFIGURATION_WITHOUT_INDEPENDENT_TERMINAL_SELECTION"
            scope = "CONFIGURATION_ONLY"
        rows.append(_row(
            row_id, "PREDICTION", "NEURAL", requirement, implementation,
            "COMMON_SOURCE_RELATIVE_TASK", information, status, coverage, scope,
            limitation, evidence, evidence_class=evidence_class,
        ))

    executed_adapters = sorted(inventory["prediction_common_task_adapters"], key=lambda item: item["model_id"])
    for index, adapter in enumerate(executed_adapters, start=1):
        rows.append(_row(
            f"P-EXT-{index:02d}", "PREDICTION", "TASK_SPECIFIC_FOUNDATION",
            f"task-specific/foundation common-task adapter: {adapter['model_id']}",
            adapter["model_id"], "COMMON_SOURCE_RELATIVE_TASK", adapter["prediction_transform"],
            "EXECUTED_COMMON_TASK_DEVELOPMENT_VALIDATION",
            "SATISFIED_COMMON_TASK_WITH_NATIVE_LIMIT", "DEVELOPMENT_VALIDATION_ONLY",
            "Common-task execution does not establish native-repository parity or independent external transfer.",
            "configs/route_a_v3_route2_baseline_inventory_v1.json;MBP-09",
            native=adapter["native_track_status"],
            evidence_class="FROZEN_COMMON_TASK_INVENTORY",
        ))
    limited_adapters = sorted(inventory["prediction_literature_only_or_not_executed"], key=lambda item: item["model_id"])
    for offset, adapter in enumerate(limited_adapters, start=7):
        coverage = "LITERATURE_ONLY_TASK_MISMATCH" if adapter["status"].endswith("TASK_MISMATCH") else "LITERATURE_ONLY_NOT_EXECUTED"
        rows.append(_row(
            f"P-EXT-{offset:02d}", "PREDICTION", "TASK_SPECIFIC_FOUNDATION",
            f"task-specific/foundation reference: {adapter['model_id']}", adapter["model_id"],
            "NATIVE_OR_LITERATURE_REFERENCE", "not executed on the frozen common source-relative interface",
            adapter["status"], coverage, "LITERATURE_ONLY", adapter["reason"],
            "configs/route_a_v3_route2_baseline_inventory_v1.json;MBP-09",
            native="NOT_EXECUTED_OR_TASK_MISMATCH",
            evidence_class="FROZEN_LIMITATION_INVENTORY",
        ))
    rows.append(_row(
        "P-EXT-10", "PREDICTION", "TASK_SPECIFIC_FOUNDATION",
        "primary foundation predictor reference", "mRNABERT_EDIT_CENTERED_CRITIC_V2",
        "COMMON_SOURCE_RELATIVE_TASK_CONTROL_GATE", "frozen mRNABERT source/candidate features + Delta head",
        "TERMINAL_CRITIC_V2_NO_GO", "PRIMARY_REFERENCE_NOT_BASELINE_WIN",
        "DEVELOPMENT_CONTROL_SCREEN_ONLY",
        "The full arm did not beat the strongest same-information baseline, so three-seed confirmation and guidance stayed closed.",
        "audits/route_a_v3_route2_critic_v2_control_terminal_no_go_v1.json",
        native="PRETRAINED_ENCODER_FROZEN",
        evidence_class="TERMINAL_CONTROL_GATE_AUDIT",
    ))

    generation_by_id = {row["method_id"]: row for row in generation_rows}
    generation_specs = [
        ("G-01", "random legal", "random_legal", "legal random SUB+STOP", "EXECUTED_TERMINAL_MATCHED_DEVELOPMENT", "SATISFIED_TERMINAL_MATCHED", "Random legal is a matched search control."),
        ("G-02", "exhaustive only where enumerable", "exhaustive", "enumerated 151-state small-space subset", "SMALL_SPACE_REFERENCE_NOT_FULL_COHORT_EXECUTION", "SATISFIED_SMALL_SPACE_WITH_LIMIT", "Only an outcome-blind 190-source enumerable subset was prepared; it is not full-cohort selector evidence."),
        ("G-03", "greedy", "greedy", "legal greedy SUB+STOP", "EXECUTED_TERMINAL_MATCHED_DEVELOPMENT", "SATISFIED_TERMINAL_MATCHED", "Terminal matched Development method."),
        ("G-04", "beam", "beam", "legal beam SUB+STOP", "EXECUTED_TERMINAL_MATCHED_DEVELOPMENT", "SATISFIED_TERMINAL_MATCHED", "Terminal matched Development method."),
        ("G-05", "genetic", "genetic", "legal genetic SUB+STOP", "EXECUTED_TERMINAL_MATCHED_DEVELOPMENT", "SATISFIED_TERMINAL_MATCHED", "Terminal matched Development method."),
        ("G-06", "simulated annealing/local search", "local_search", "legal local search SUB+STOP", "EXECUTED_TERMINAL_MATCHED_DEVELOPMENT", "SATISFIED_TERMINAL_MATCHED", "Terminal matched Development method; generation wall time was not retained."),
        ("G-07", "generate-N-then-rerank", "generate_then_rerank", "legal proposal generation followed by frozen reranking", "EXECUTED_TERMINAL_MATCHED_DEVELOPMENT", "SATISFIED_TERMINAL_MATCHED", "Terminal matched Development method."),
        ("G-08", "unguided base flow", "unguided_learned_base_flow_g0", "position/progress Base Flow with legal SUB+STOP", "EXECUTED_TERMINAL_MATCHED_DEVELOPMENT", "SATISFIED_TERMINAL_MATCHED", "FLOW_G0_READY is engineering readiness only and establishes no biological optimization."),
        ("G-09", "first-order/rate guidance", "first_order_rate_guidance", "prospective frozen-critic first-order/rate feedback", "NOT_RUN_CRITIC_V2_NO_GO", "NOT_AUTHORIZED_DEPENDENCY_NO_GO", "Critic V2 NO-GO prohibits the current guided route."),
        ("G-10", "frozen-critic XEditFlow", "frozen_critic_xeditflow", "prospective Legal XEditFlow with frozen critic", "NOT_RUN_CRITIC_V2_NO_GO", "NOT_AUTHORIZED_DEPENDENCY_NO_GO", "Critic V2 NO-GO prohibits the current guided route."),
        ("G-11", "masked discrete flow/diffusion", "masked_discrete_flow_or_diffusion", "literature reference with nonmatching action/task interface", "LITERATURE_ONLY_TASK_MISMATCH", "LITERATURE_ONLY_TASK_MISMATCH", "No jointly executable matched Route 2 source-relative SUB+STOP interface was established."),
    ]
    for row_id, requirement, implementation, information, status, coverage, limitation in generation_specs:
        evidence = "configs/route_a_v3_route2_baseline_inventory_v1.json;audits/route_a_v3_route2_generation_action_space_geometry_v1.json"
        evidence_class = "FROZEN_LIMITATION_INVENTORY"
        scope = "DEVELOPMENT_GENERATION_OPEN_SUPPORT"
        guided_required = implementation in {"first_order_rate_guidance", "frozen_critic_xeditflow"}
        if implementation in generation_by_id:
            evidence = "docs/paper/route2_v332_generation_baseline_table_v1.csv;audits/route_a_v3_route2_generation_action_space_geometry_v1.json"
            evidence_class = "TERMINAL_MATCHED_DEVELOPMENT_TABLE"
        elif guided_required:
            evidence += ";audits/route_a_v3_route2_critic_v2_control_terminal_no_go_v1.json;MBP-10"
            evidence_class = "TERMINAL_DEPENDENCY_NO_GO"
            scope = "NOT_RUN"
        elif implementation == "masked_discrete_flow_or_diffusion":
            scope = "LITERATURE_ONLY"
        rows.append(_row(
            row_id, "GENERATION", "SEARCH_FLOW", requirement, implementation,
            "DEVELOPMENT_SUB_STOP_OPEN_GENERATED_SUPPORT", information, status, coverage,
            scope, limitation, evidence, evidence_class=evidence_class,
            guided_required=guided_required, guided_executed=False,
        ))
    return rows


def _validate_rows(rows: Sequence[Mapping[str, str]]) -> None:
    _require(len(rows) == 45, "baseline matrix must contain exactly 45 contract rows")
    _require(len({row["matrix_row_id"] for row in rows}) == 45, "matrix row IDs are not unique")
    _require(Counter(row["track"] for row in rows) == {"PREDICTION": 34, "GENERATION": 11}, "track counts changed")
    _require(
        Counter(row["baseline_family"] for row in rows)
        == {"INTERNAL_CONTROL": 11, "CLASSICAL": 6, "NEURAL": 7, "TASK_SPECIFIC_FOUNDATION": 10, "SEARCH_FLOW": 11},
        "baseline-family counts changed",
    )
    forbidden_result_columns = {
        "spearman", "mae", "ndcg", "uplift", "recovery", "wall_time_seconds",
        "generation_peak_vram_mb", "candidate_count", "nfe",
    }
    _require(not (set(FIELDNAMES) & forbidden_result_columns), "result metrics entered the inventory matrix")
    _require(all(row["headline_eligible_now"] == "false" for row in rows), "a baseline row became headline eligible")
    _require(all(row["development_test_accessed"] == "false" for row in rows), "Development TEST entered the matrix")
    _require(all(row["new_final_evaluation_accessed"] == "false" for row in rows), "new final Evaluation entered the matrix")
    _require(all(row["guided_executed"] == "false" for row in rows), "guided execution entered the matrix")


def build_matrix(
    *,
    baseline_inventory_path: Path = DEFAULT_BASELINE_INVENTORY,
    classical_config_path: Path = DEFAULT_CLASSICAL_CONFIG,
    elastic_audit_path: Path = DEFAULT_ELASTIC_AUDIT,
    neural_audit_path: Path = DEFAULT_NEURAL_AUDIT,
    critic_audit_path: Path = DEFAULT_CRITIC_AUDIT,
    generation_table_path: Path = DEFAULT_GENERATION_TABLE,
    geometry_audit_path: Path = DEFAULT_GEOMETRY_AUDIT,
    minimum_package_table_path: Path = DEFAULT_MINIMUM_PACKAGE_TABLE,
    pairwise_config_path: Path = DEFAULT_PAIRWISE_CONFIG,
    listwise_config_path: Path = DEFAULT_LISTWISE_CONFIG,
    table_path: Path = DEFAULT_TABLE,
    audit_path: Path = DEFAULT_AUDIT,
    overwrite: bool = False,
) -> dict[str, Any]:
    paths = [
        baseline_inventory_path, classical_config_path, elastic_audit_path,
        neural_audit_path, critic_audit_path, generation_table_path,
        geometry_audit_path, minimum_package_table_path, pairwise_config_path,
        listwise_config_path, table_path, audit_path,
    ]
    (
        baseline_inventory_path, classical_config_path, elastic_audit_path,
        neural_audit_path, critic_audit_path, generation_table_path,
        geometry_audit_path, minimum_package_table_path, pairwise_config_path,
        listwise_config_path, table_path, audit_path,
    ) = [path.resolve() for path in paths]
    for path in (table_path, audit_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing baseline matrix artifact: {path}")

    inventory = _load_json(baseline_inventory_path)
    classical_raw = _load_json(classical_config_path)
    classical = classical_raw if isinstance(classical_raw, list) else classical_raw["baselines"]
    elastic = _load_json(elastic_audit_path)
    neural = _load_json(neural_audit_path)
    critic = _load_json(critic_audit_path)
    generation_rows = _load_csv(generation_table_path)
    geometry = _load_json(geometry_audit_path)
    minimum_rows = _load_csv(minimum_package_table_path)
    pairwise = _load_json(pairwise_config_path)
    listwise = _load_json(listwise_config_path)
    _validate_inputs(
        inventory, classical, elastic, neural, critic, generation_rows, geometry,
        minimum_rows, pairwise, listwise,
    )
    rows = _build_rows(inventory, generation_rows)
    _validate_rows(rows)

    table_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    execution_counts = Counter(row["execution_status_v332"] for row in rows)
    coverage_counts = Counter(row["contract_coverage_status"] for row in rows)
    audit = {
        "schema_version": "route_a_v3_route2_v332_baseline_matrix.v1",
        "status": "BASELINE_INVENTORY_MATRIX_RENDERED_DEVELOPMENT_ONLY",
        "authority": {
            "scientific_contract": "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/提示词/mrna 数据gate转向后的合同.md",
            "execution_protocol": "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/提示词/mrna V3.3.2 执行提示词.md",
            "authority_rule": "V3.3.2 terminal audits override stale running and preterminal generation labels.",
        },
        "table_path": _display_path(table_path),
        "matrix_is_result_table": False,
        "row_count": len(rows),
        "track_counts": dict(sorted(Counter(row["track"] for row in rows).items())),
        "family_counts": dict(sorted(Counter(row["baseline_family"] for row in rows).items())),
        "execution_status_counts": dict(sorted(execution_counts.items())),
        "coverage_status_counts": dict(sorted(coverage_counts.items())),
        "contract_coverage_summary": {
            "prediction_internal_control_rows": 11,
            "prediction_classical_rows": 6,
            "prediction_neural_rows": 7,
            "prediction_neural_terminal_independent_rows": 5,
            "prediction_neural_configured_not_terminal_independent_rows": 2,
            "prediction_external_common_task_executed_rows": 6,
            "prediction_external_literature_or_task_mismatch_rows": 3,
            "prediction_primary_reference_terminal_no_go_rows": 1,
            "generation_terminal_matched_rows": 7,
            "generation_small_space_only_rows": 1,
            "generation_guided_not_authorized_rows": 2,
            "generation_literature_task_mismatch_rows": 1,
        },
        "stale_status_overrides": [
            {
                "artifact": _display_path(baseline_inventory_path),
                "stale_field": "primary mRNABERT HPO running",
                "current_authority": "Critic V2 terminal NO-GO control audit",
            },
            {
                "artifact": _display_path(baseline_inventory_path),
                "stale_field": "preterminal generation execution labels",
                "current_authority": "seven-method terminal generation table and action-space geometry audit",
            },
        ],
        "minimum_package_alignment": {
            "MBP-08": "COMPLETE_DEVELOPMENT_ONLY",
            "MBP-09": "COMPLETE_DEVELOPMENT_ONLY",
            "MBP-10": "PARTIAL_GUIDED_NOT_AUTHORIZED",
            "minimum_package_complete": False,
            "submission_ready": False,
        },
        "separate_future_artifacts": {
            "native_common_arch_three_track_results_table_built": False,
            "prediction_generation_matched_budget_numeric_matrix_built": False,
        },
        "protected_outcomes": {
            "development_test_read": False,
            "new_final_evaluation_read": False,
            "guided_xeditflow_run": False,
            "generated_candidates_opened": False,
        },
        "claim_boundary": "The matrix establishes auditable Development coverage and declared gaps only; it contains no result metric and establishes no external superiority, biological optimization, Route A success, or final Evaluation claim.",
        "scientific_claim_status": "NOT_ESTABLISHED",
        "new_training_attempt_created": False,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-inventory", type=Path, default=DEFAULT_BASELINE_INVENTORY)
    parser.add_argument("--classical-config", type=Path, default=DEFAULT_CLASSICAL_CONFIG)
    parser.add_argument("--elastic-audit", type=Path, default=DEFAULT_ELASTIC_AUDIT)
    parser.add_argument("--neural-audit", type=Path, default=DEFAULT_NEURAL_AUDIT)
    parser.add_argument("--critic-audit", type=Path, default=DEFAULT_CRITIC_AUDIT)
    parser.add_argument("--generation-table", type=Path, default=DEFAULT_GENERATION_TABLE)
    parser.add_argument("--geometry-audit", type=Path, default=DEFAULT_GEOMETRY_AUDIT)
    parser.add_argument("--minimum-package-table", type=Path, default=DEFAULT_MINIMUM_PACKAGE_TABLE)
    parser.add_argument("--pairwise-config", type=Path, default=DEFAULT_PAIRWISE_CONFIG)
    parser.add_argument("--listwise-config", type=Path, default=DEFAULT_LISTWISE_CONFIG)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    audit = build_matrix(
        baseline_inventory_path=args.baseline_inventory,
        classical_config_path=args.classical_config,
        elastic_audit_path=args.elastic_audit,
        neural_audit_path=args.neural_audit,
        critic_audit_path=args.critic_audit,
        generation_table_path=args.generation_table,
        geometry_audit_path=args.geometry_audit,
        minimum_package_table_path=args.minimum_package_table,
        pairwise_config_path=args.pairwise_config,
        listwise_config_path=args.listwise_config,
        table_path=args.table,
        audit_path=args.audit,
        overwrite=args.overwrite,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
