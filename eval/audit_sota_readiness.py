"""Unified SOTA-readiness audit for pending mRNA-EditFlow evidence.

This module does not run benchmarks. It combines the result-level audits for
the current blocking evidence streams:

* region-specialized adapter comparisons;
* protein-conditioned CDS CAI-GC sweep.
* multi-objective head256/head1024 claim-language audit.
* external SOTA protocol/dry-run readiness.
* frozen-foundation matched-budget protocol readiness.

The goal is to make over-claiming hard: a section can be numerically negative
or non-significant and still be complete evidence, but missing files, non-finite
statistics, or violated hard constraints keep the overall readiness false.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Mapping, Optional, Sequence

from mrna_editflow.eval.artifact_contract import normalize_run_mode, paper_builder_gate, validate_report_output_namespaces, write_paper_report_sidecars

from mrna_editflow.eval.audit_protein_conditioned_gc_sweep import (
    audit_protein_conditioned_gc_sweep,
)
from mrna_editflow.eval.audit_region_adapter_results import audit_region_adapter_results
from mrna_editflow.eval.audit_multiobjective_scaleup_claims import (
    audit_multiobjective_scaleup_claims,
)


def _rel(path: str, root: str) -> str:
    return os.path.relpath(path, root) if os.path.isabs(path) else path


def _gc_prefix(project_root: str, slice_name: str) -> str:
    return os.path.join(
        project_root,
        "benchmark",
        f"protein_conditioned_cds_gc_sweep_{slice_name}",
    )


def _region_audit_prefix(project_root: str, slice_name: str) -> str:
    return os.path.join(
        project_root,
        "benchmark",
        f"region_adapter_result_audit_{slice_name}",
    )


def _external_sota_dir(project_root: str) -> str:
    return os.path.join(project_root, "benchmark", "external_sota", "dry_run_t5_head1024")


def _external_input_pack_dir(project_root: str) -> str:
    return os.path.join(project_root, "benchmark", "external_sota", "input_pack_t5_head1024")


def _external_real_run_audit_path(project_root: str) -> str:
    return os.path.join(project_root, "docs", "external_sota_real_run_audit.json")


def _t5_external_utr_comparison_path(project_root: str) -> str:
    return os.path.join(
        project_root,
        "docs",
        "t5_external_utr_baseline_comparison.json",
    )


def _frozen_protocol_dir(project_root: str, slice_name: str) -> str:
    return os.path.join(project_root, "benchmark", f"frozen_backbone_protocol_{slice_name}")


T1_T7_BUNDLE_REPORTS: tuple[dict[str, str], ...] = (
    {
        "name": "t1_t7_ledger",
        "json": "benchmark/t1_t7_evidence_status_head256.json",
        "md": "benchmark/t1_t7_evidence_status_head256.md",
    },
    {
        "name": "t1_runtime",
        "json": "benchmark/t1_runtime_report_head256_head1024.json",
        "md": "benchmark/t1_runtime_report_head256_head1024.md",
    },
    {
        "name": "t2_t3_distribution_novelty",
        "json": "benchmark/t2_t3_distribution_novelty_report_head256_head1024.json",
        "md": "benchmark/t2_t3_distribution_novelty_report_head256_head1024.md",
    },
    {
        "name": "multi_scale_sequence_spectrum",
        "json": "benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k.json",
        "md": "benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k.md",
    },
    {
        "name": "t4_protein_identity_cai_gc",
        "json": "benchmark/t4_protein_identity_cai_gc_report_head256.json",
        "md": "benchmark/t4_protein_identity_cai_gc_report_head256.md",
    },
    {
        "name": "t5_edit_budget_curve",
        "json": "benchmark/edit_budget_curve_report_head256_head1024.json",
        "md": "benchmark/edit_budget_curve_report_head256_head1024.md",
    },
    {
        "name": "t6_length_curve",
        "json": "benchmark/t6_length_curve_report_head256_head1024.json",
        "md": "benchmark/t6_length_curve_report_head256_head1024.md",
    },
    {
        "name": "t7_motif_frame",
        "json": "benchmark/t7_motif_frame_report_head256.json",
        "md": "benchmark/t7_motif_frame_report_head256.md",
    },
    {
        "name": "t7_motif_edit",
        "json": "benchmark/t7_motif_edit_benchmark_head256/summary.json",
        "md": "benchmark/t7_motif_edit_benchmark_head256/summary.md",
    },
)

DATA_SCALEUP_READINESS_REPORT = "docs/data_scaleup_readiness.json"
T2_T3_DISTRIBUTION_NOVELTY_REPORT = (
    "benchmark/t2_t3_distribution_novelty_report_head256_head1024.json"
)


def _load_json_if_exists(path: str) -> Optional[Mapping[str, object]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _summary_from_report(project_root: str, rel_path: str) -> tuple[Optional[Mapping[str, object]], str]:
    """Load a JSON report and return its summary-like mapping plus relative path."""
    path = os.path.join(project_root, rel_path)
    payload = _load_json_if_exists(path)
    if payload is None:
        return None, rel_path
    summary = payload.get("summary", payload)
    if not isinstance(summary, Mapping):
        return {}, rel_path
    return summary, rel_path


def _num(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _all_constraint_rows_exact_1(rows: object) -> bool:
    if not isinstance(rows, list) or not rows:
        return False
    constraint_keys = (
        "legal_fraction",
        "mean_protein_identity",
        "within_budget_fraction",
        "reading_frame_intact_fraction",
    )
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        for key in constraint_keys:
            if key in row and _num(row.get(key)) != 1.0:
                return False
    return True


def _check_t1_t7_report(name: str, payload: Optional[Mapping[str, object]]) -> bool:
    if payload is None:
        return False
    if name == "t1_t7_ledger":
        tasks = payload.get("tasks", [])
        if not isinstance(tasks, list):
            return False
        task_names = {row.get("task") for row in tasks if isinstance(row, Mapping)}
        statuses_ready = all(
            isinstance(row, Mapping) and str(row.get("status", "")).startswith("ready")
            for row in tasks
        )
        return (
            payload.get("artifact_kind") == "t1_t7_evidence_status"
            and task_names == {"T1", "T2", "T3", "T4", "T5", "T6", "T7"}
            and statuses_ready
            and bool(payload.get("claim_policy"))
        )
    if name == "t1_runtime":
        interp = payload.get("interpretation", {})
        rows = payload.get("rows", [])
        return (
            payload.get("artifact_kind") == "t1_runtime_report"
            and isinstance(rows, list)
            and len(rows) >= 1
            and isinstance(interp, Mapping)
            and interp.get("strict_hardware_benchmark_ready") is False
        )
    if name == "t2_t3_distribution_novelty":
        interp = payload.get("interpretation", {})
        rows = payload.get("rows", [])
        return (
            payload.get("artifact_kind") == "t2_t3_distribution_novelty_report"
            and isinstance(rows, list)
            and bool(rows)
            and all(isinstance(row, Mapping) and row.get("status") == "complete" for row in rows)
            and isinstance(interp, Mapping)
            and interp.get("primary_head256_distribution_collapse_flag") is False
            and interp.get("primary_head256_de_novo_overclaim_flag") is True
        )
    if name == "multi_scale_sequence_spectrum":
        summary = payload.get("summary", {})
        figures = payload.get("figures", {})
        base = payload.get("base_composition", {})
        regions = base.get("regions", {}) if isinstance(base, Mapping) else {}
        required_figures = (
            "base_composition_full_svg",
            "base_composition_five_utr_svg",
            "base_composition_cds_svg",
            "base_composition_three_utr_svg",
            "length_histogram_svg",
            "gc_histogram_svg",
            "kmer_top_delta_svg",
            "codon_pair_top_delta_svg",
        )
        return (
            payload.get("artifact_kind") == "multi_scale_sequence_spectrum_audit"
            and isinstance(summary, Mapping)
            and summary.get("ready_for_distribution_figure_audit") is True
            and isinstance(base, Mapping)
            and isinstance(regions, Mapping)
            and all(region in regions for region in ("five_utr", "cds", "three_utr"))
            and all(
                isinstance(base.get("full", {}).get("candidate", {}).get(nt), (int, float))
                for nt in ("A", "C", "G", "U")
            )
            and isinstance(figures, Mapping)
            and all(bool(figures.get(key)) for key in required_figures)
        )
    if name == "t4_protein_identity_cai_gc":
        summary = payload.get("summary", {})
        return (
            payload.get("artifact_kind") == "t4_protein_identity_cai_gc_report"
            and isinstance(summary, Mapping)
            and summary.get("ready") is True
            and summary.get("hard_constraints_exact_1") is True
            and summary.get("codon_level_metrics_ready") is True
            and summary.get("external_baselines_configured") is False
            and summary.get("true_mfe_structure_metric_available") is False
        )
    if name == "t5_edit_budget_curve":
        head256 = payload.get("head256_mo_grpo", {})
        head1024 = payload.get("head1024_mo_pareto", {})
        return (
            payload.get("artifact_kind") == "edit_budget_curve_report"
            and isinstance(head256, Mapping)
            and isinstance(head1024, Mapping)
            and head256.get("status") == "complete"
            and head1024.get("status") == "complete"
            and _all_constraint_rows_exact_1(head256.get("rows"))
            and _all_constraint_rows_exact_1(head1024.get("rows"))
        )
    if name == "t6_length_curve":
        head256 = payload.get("head256_stagea10k", {})
        head1024 = payload.get("head1024_stagea10k", {})
        return (
            payload.get("artifact_kind") == "t6_length_curve_report"
            and isinstance(head256, Mapping)
            and isinstance(head1024, Mapping)
            and head256.get("status") == "complete"
            and head1024.get("status") == "complete"
            and head256.get("pending_target_length_deltas") == []
            and head1024.get("pending_target_length_deltas") == []
            and _all_constraint_rows_exact_1(head256.get("rows"))
            and _all_constraint_rows_exact_1(head1024.get("rows"))
        )
    if name == "t7_motif_frame":
        rows = payload.get("rows", [])
        frame_rows = [
            row for row in rows
            if isinstance(row, Mapping) and row.get("metric") == "reading_frame_intact_fraction"
        ]
        return (
            payload.get("artifact_kind") == "t7_motif_frame_report"
            and bool(frame_rows)
            and all(_num(row.get("mean")) == 1.0 for row in frame_rows)
        )
    if name == "t7_motif_edit":
        aggregate = payload.get("aggregate", {})
        if not isinstance(aggregate, Mapping):
            return False
        required_exact = (
            "insert_mean_protein_identity",
            "insert_reading_frame_intact_fraction",
            "insert_within_budget_fraction",
            "excise_mean_protein_identity",
            "excise_reading_frame_intact_fraction",
            "excise_within_budget_fraction",
        )
        success = (
            _num(aggregate.get("insert_success_fraction", {}).get("mean"))
            if isinstance(aggregate.get("insert_success_fraction"), Mapping)
            else None
        )
        excise_success = (
            _num(aggregate.get("excise_success_fraction", {}).get("mean"))
            if isinstance(aggregate.get("excise_success_fraction"), Mapping)
            else None
        )
        return (
            payload.get("artifact_kind") == "t7_motif_edit_benchmark"
            and success is not None and success >= 0.99
            and excise_success is not None and excise_success >= 0.99
            and all(
                isinstance(aggregate.get(key), Mapping)
                and _num(aggregate.get(key, {}).get("mean")) == 1.0
                for key in required_exact
            )
        )
    return False


def _audit_t1_t7_evidence_bundle(project_root: str) -> dict[str, object]:
    reports = []
    missing_artifacts = []
    failed_checks = []
    for spec in T1_T7_BUNDLE_REPORTS:
        name = spec["name"]
        json_path = os.path.join(project_root, spec["json"])
        md_path = os.path.join(project_root, spec["md"])
        payload = _load_json_if_exists(json_path)
        json_exists = payload is not None
        md_exists = os.path.exists(md_path)
        check_ok = _check_t1_t7_report(name, payload)
        report = {
            "name": name,
            "json_path": spec["json"],
            "md_path": spec["md"],
            "json_exists": json_exists,
            "md_exists": md_exists,
            "check_ok": check_ok,
        }
        reports.append(report)
        if not json_exists:
            missing_artifacts.append(spec["json"])
        if not md_exists:
            missing_artifacts.append(spec["md"])
        if json_exists and not check_ok:
            failed_checks.append(name)
    ready = (
        bool(reports)
        and all(bool(row["json_exists"]) and bool(row["md_exists"]) and bool(row["check_ok"]) for row in reports)
    )
    return {
        "artifact_kind": "t1_t7_evidence_bundle_audit",
        "summary": {
            "bundle_ready": ready,
            "n_reports_ready": sum(
                1 for row in reports
                if bool(row["json_exists"]) and bool(row["md_exists"]) and bool(row["check_ok"])
            ),
            "n_reports_expected": len(reports),
            "missing_artifacts": missing_artifacts,
            "failed_checks": failed_checks,
            "claim_policy": (
                "Bundle readiness means internal T1-T7 proxy/constraint reports "
                "are complete and boundary flags are auditable. It does not mean "
                "external SOTA, wet-lab, or de novo claims are complete."
            ),
        },
        "reports": reports,
    }


def _audit_external_sota_protocol(project_root: str) -> dict[str, object]:
    dry_run_dir = _external_sota_dir(project_root)
    input_pack_dir = _external_input_pack_dir(project_root)
    summary_path = os.path.join(dry_run_dir, "summary.json")
    runtime_path = os.path.join(dry_run_dir, "runtime.json")
    table_path = os.path.join(dry_run_dir, "table.md")
    input_pack_summary_path = os.path.join(input_pack_dir, "summary.json")
    input_pack_table_path = os.path.join(input_pack_dir, "table.md")
    input_pack_cds_path = os.path.join(input_pack_dir, "cds_protein_inputs.jsonl")
    input_pack_utr5_path = os.path.join(input_pack_dir, "utr5_inputs.jsonl")
    input_pack_schema_path = os.path.join(input_pack_dir, "metric_schema.json")
    real_run_audit_path = _external_real_run_audit_path(project_root)
    t5_utr_comparison_path = _t5_external_utr_comparison_path(project_root)
    summary = _load_json_if_exists(summary_path)
    runtime = _load_json_if_exists(runtime_path)
    input_pack = _load_json_if_exists(input_pack_summary_path)
    real_run_audit = _load_json_if_exists(real_run_audit_path)
    t5_utr_comparison = _load_json_if_exists(t5_utr_comparison_path)
    rows = summary.get("rows", []) if summary else []
    if not isinstance(rows, list):
        rows = []
    dataset = summary.get("dataset", {}) if summary else {}
    if not isinstance(dataset, Mapping):
        dataset = {}
    runtime_dataset_sha = runtime.get("dataset_sha256") if runtime else None
    required_real_run_metadata = []
    contract = summary.get("artifact_contract", {}) if summary else {}
    if isinstance(contract, Mapping):
        metadata = contract.get("required_real_run_metadata", [])
        if isinstance(metadata, list):
            required_real_run_metadata = metadata
    required_metadata_ok = all(
        item in required_real_run_metadata
        for item in (
            "dataset.sha256",
            "dataset.split_name",
            "dataset.seed",
            "dataset.record_count_effective",
            "runtime.elapsed_s",
            "hardware",
        )
    )
    dataset_ok = (
        bool(dataset.get("exists"))
        and isinstance(dataset.get("sha256"), str)
        and bool(dataset.get("sha256"))
        and dataset.get("sha256") == runtime_dataset_sha
        and isinstance(dataset.get("record_count_effective"), int)
        and int(dataset.get("record_count_effective", 0)) > 0
    )
    rows_ok = bool(rows) and all(
        isinstance(row, Mapping)
        and row.get("model_name")
        and row.get("status") in {"not_configured", "executable_ready"}
        and isinstance(row.get("protocol_difference"), str)
        and bool(row.get("protocol_difference"))
        and row.get("metrics") == {}
        for row in rows
    )
    protocol_ready = (
        summary is not None
        and runtime is not None
        and os.path.exists(table_path)
        and summary.get("status") == "dry_run_complete"
        and dataset_ok
        and rows_ok
        and required_metadata_ok
    )
    input_pack_outputs = input_pack.get("outputs", {}) if isinstance(input_pack, Mapping) else {}
    if not isinstance(input_pack_outputs, Mapping):
        input_pack_outputs = {}
    input_pack_ready = (
        isinstance(input_pack, Mapping)
        and input_pack.get("artifact_kind") == "external_sota_input_pack"
        and input_pack.get("ready_for_external_real_run") is True
        and input_pack.get("ready_for_external_sota_claim") is False
        and isinstance(input_pack.get("n_cds_protein_rows"), int)
        and int(input_pack.get("n_cds_protein_rows", 0)) > 0
        and isinstance(input_pack.get("n_utr5_rows"), int)
        and int(input_pack.get("n_utr5_rows", 0)) > 0
        and os.path.exists(input_pack_table_path)
        and os.path.exists(input_pack_cds_path)
        and os.path.exists(input_pack_utr5_path)
        and os.path.exists(input_pack_schema_path)
        and bool(input_pack_outputs.get("cds_protein_jsonl_sha256"))
        and bool(input_pack_outputs.get("utr5_jsonl_sha256"))
        and bool(input_pack_outputs.get("metric_schema_json_sha256"))
    )
    missing_artifacts = [
        _rel(path, project_root)
        for path in (summary_path, runtime_path, table_path)
        if not os.path.exists(path)
    ]
    input_pack_missing_artifacts = [
        _rel(path, project_root)
        for path in (
            input_pack_summary_path,
            input_pack_table_path,
            input_pack_cds_path,
            input_pack_utr5_path,
            input_pack_schema_path,
        )
        if not os.path.exists(path)
    ]
    real_run_summary = real_run_audit.get("summary", {}) if isinstance(real_run_audit, Mapping) else {}
    if not isinstance(real_run_summary, Mapping):
        real_run_summary = {}
    real_run_audit_complete = (
        isinstance(real_run_audit, Mapping)
        and real_run_audit.get("artifact_kind") == "external_sota_real_run_audit"
        and real_run_summary.get("audit_complete") is True
    )
    t5_utr_summary = (
        t5_utr_comparison.get("summary", {})
        if isinstance(t5_utr_comparison, Mapping)
        else {}
    )
    if not isinstance(t5_utr_summary, Mapping):
        t5_utr_summary = {}
    ready_for_external_real_metric_table = bool(
        real_run_summary.get("ready_for_external_real_metric_table")
    )
    ready_for_external_sota_metric_claim = bool(
        real_run_summary.get("ready_for_external_sota_metric_claim")
    )
    input_pack_groups = input_pack.get("models", {}) if isinstance(input_pack, Mapping) else {}
    if not isinstance(input_pack_groups, Mapping):
        input_pack_groups = {}
    input_pack_models: set[str] = set()
    for values in input_pack_groups.values():
        if isinstance(values, list):
            input_pack_models.update(str(value) for value in values)
    dry_run_models = {
        str(row.get("model_name"))
        for row in rows
        if isinstance(row, Mapping) and row.get("model_name")
    }
    real_run_rows = real_run_audit.get("rows", []) if isinstance(real_run_audit, Mapping) else []
    if not isinstance(real_run_rows, list):
        real_run_rows = []
    real_run_models = {
        str(row.get("model_name"))
        for row in real_run_rows
        if isinstance(row, Mapping) and row.get("model_name")
    }
    model_set_consistent = bool(
        dry_run_models
        and input_pack_models
        and real_run_models
        and dry_run_models == input_pack_models == real_run_models
    )
    real_run_missing_artifacts = (
        [] if os.path.exists(real_run_audit_path) else [_rel(real_run_audit_path, project_root)]
    )
    return {
        "artifact_kind": "external_sota_protocol_audit",
        "summary_path": _rel(summary_path, project_root),
        "runtime_path": _rel(runtime_path, project_root),
        "table_path": _rel(table_path, project_root),
        "input_pack_summary_path": _rel(input_pack_summary_path, project_root),
        "real_run_audit_path": _rel(real_run_audit_path, project_root),
        "t5_external_utr_comparison_path": _rel(
            t5_utr_comparison_path, project_root
        ),
        "summary": {
            "protocol_ready": protocol_ready,
            "input_pack_ready": input_pack_ready,
            "real_run_audit_complete": real_run_audit_complete,
            "ready_for_external_real_metric_table": ready_for_external_real_metric_table,
            "ready_for_external_sota_metric_claim": ready_for_external_sota_metric_claim,
            "model_set_consistent": model_set_consistent,
            "dry_run_models": sorted(dry_run_models),
            "input_pack_models": sorted(input_pack_models),
            "real_run_audit_models": sorted(real_run_models),
            "summary_exists": summary is not None,
            "runtime_exists": runtime is not None,
            "table_exists": os.path.exists(table_path),
            "input_pack_summary_exists": input_pack is not None,
            "real_run_audit_exists": real_run_audit is not None,
            "t5_external_utr_comparison_exists": t5_utr_comparison is not None,
            "t5_utr_descriptive_table_ready": bool(
                t5_utr_summary.get("ready_for_t5_utr_descriptive_table")
            ),
            "t5_utr_model_only_head_to_head_ready": bool(
                t5_utr_summary.get("ready_for_model_only_head_to_head")
            ),
            "t5_utr_mef_superiority_claim_ready": bool(
                t5_utr_summary.get("ready_for_mef_superiority_claim")
            ),
            "dataset_ok": dataset_ok,
            "rows_ok": rows_ok,
            "required_metadata_ok": required_metadata_ok,
            "n_models": len(rows),
            "n_executable_ready": summary.get("n_executable_ready") if summary else None,
            "n_input_pack_cds_protein_rows": (
                input_pack.get("n_cds_protein_rows") if isinstance(input_pack, Mapping) else None
            ),
            "n_input_pack_utr5_rows": (
                input_pack.get("n_utr5_rows") if isinstance(input_pack, Mapping) else None
            ),
            "n_input_pack_skipped_invalid_cds": (
                input_pack.get("n_skipped_invalid_cds") if isinstance(input_pack, Mapping) else None
            ),
            "missing_artifacts": missing_artifacts,
            "input_pack_missing_artifacts": input_pack_missing_artifacts,
            "real_run_missing_artifacts": real_run_missing_artifacts,
            "n_real_run_models_expected": real_run_summary.get("n_models_expected"),
            "n_real_run_models_measured": real_run_summary.get("n_models_measured"),
            "n_real_run_models_invalid": real_run_summary.get("n_models_invalid"),
            "n_real_run_models_missing": real_run_summary.get("n_models_missing"),
            "real_metric_policy": (
                contract.get("real_metric_policy") if isinstance(contract, Mapping) else None
            ),
            "input_pack_claim_policy": (
                input_pack.get("claim_policy") if isinstance(input_pack, Mapping) else None
            ),
        },
        "rows": rows,
        "real_run_rows": (
            real_run_rows
        ),
    }


def _audit_frozen_foundation_protocol(project_root: str, slice_name: str) -> dict[str, object]:
    protocol_dir = _frozen_protocol_dir(project_root, slice_name)
    summary_path = os.path.join(protocol_dir, "summary.json")
    table_path = os.path.join(protocol_dir, "table.md")
    leakage_path = os.path.join(protocol_dir, "leakage.json")
    summary = _load_json_if_exists(summary_path)
    leakage = _load_json_if_exists(leakage_path)
    runs = summary.get("runs", []) if summary else []
    if not isinstance(runs, list):
        runs = []
    gate = summary.get("leakage_gate", {}) if summary else {}
    if not isinstance(gate, Mapping):
        gate = {}
    matched = summary.get("matched_budget", {}) if summary else {}
    if not isinstance(matched, Mapping):
        matched = {}
    real_runs = [
        row for row in runs if isinstance(row, Mapping) and bool(row.get("is_real"))
    ]
    stub_runs = [
        row for row in runs if isinstance(row, Mapping) and not bool(row.get("is_real"))
    ]
    runs_ok = (
        bool(real_runs)
        and bool(stub_runs)
        and all(bool(row.get("finite_loss")) for row in runs if isinstance(row, Mapping))
        and all(bool(row.get("valid_quality_signal")) for row in real_runs)
        and all(not bool(row.get("valid_quality_signal")) for row in stub_runs)
    )
    protocol_ready = (
        summary is not None
        and os.path.exists(table_path)
        and leakage is not None
        and bool(gate.get("enabled"))
        and bool(gate.get("audited"))
        and bool(gate.get("passed"))
        and int(gate.get("exact_match_count", -1)) == 0
        and bool(matched.get("trainable_params_consistent"))
        and int(summary.get("n_real_arms", 0)) >= 1
        and int(summary.get("n_stub_arms", 0)) >= 1
        and runs_ok
    )
    missing_artifacts = [
        _rel(path, project_root)
        for path in (summary_path, table_path, leakage_path)
        if not os.path.exists(path)
    ]
    return {
        "artifact_kind": "frozen_foundation_protocol_audit",
        "summary_path": _rel(summary_path, project_root),
        "table_path": _rel(table_path, project_root),
        "leakage_path": _rel(leakage_path, project_root),
        "summary": {
            "protocol_ready": protocol_ready,
            "summary_exists": summary is not None,
            "table_exists": os.path.exists(table_path),
            "leakage_exists": leakage is not None,
            "leakage_gate_passed": bool(gate.get("passed")),
            "matched_budget": bool(matched.get("trainable_params_consistent")),
            "n_real_arms": summary.get("n_real_arms") if summary else None,
            "n_stub_arms": summary.get("n_stub_arms") if summary else None,
            "runs_ok": runs_ok,
            "missing_artifacts": missing_artifacts,
            "claim_policy": (
                "Protocol ready means leakage-gated matched-budget plumbing is "
                "auditable. Placeholder external arms remain non-quotable until "
                "real foundation checkpoints are installed."
            ),
        },
        "runs": runs,
    }


def _positive_sota_claim_gate(
    *,
    project_root: str,
    all_sections_ready: bool,
    external_sota: Mapping[str, object],
    multiobjective: Mapping[str, object],
) -> dict[str, object]:
    """Return strict paper-claim gates separate from evidence-completeness gates."""
    data_summary, data_path = _summary_from_report(project_root, DATA_SCALEUP_READINESS_REPORT)
    t2_t3_path = os.path.join(project_root, T2_T3_DISTRIBUTION_NOVELTY_REPORT)
    t2_t3 = _load_json_if_exists(t2_t3_path)
    t2_t3_interpretation = (
        t2_t3.get("interpretation", {}) if isinstance(t2_t3, Mapping) else {}
    )
    if not isinstance(t2_t3_interpretation, Mapping):
        t2_t3_interpretation = {}

    external_summary = external_sota.get("summary", {})
    if not isinstance(external_summary, Mapping):
        external_summary = {}
    external_rows = external_sota.get("real_run_rows", [])
    if not isinstance(external_rows, list):
        external_rows = []
    external_metrics_ready = bool(
        external_summary.get("ready_for_external_sota_metric_claim")
    )

    mo_summary = multiobjective.get("summary", {})
    if not isinstance(mo_summary, Mapping):
        mo_summary = {}

    data_summary = data_summary or {}
    ready_for_real_te_or_stability = bool(
        data_summary.get("ready_for_real_te_or_stability_claim")
    )
    ready_for_true_scale_law = bool(data_summary.get("ready_for_true_scale_law_claim"))
    de_novo_overclaim_flag = t2_t3_interpretation.get("primary_head256_de_novo_overclaim_flag")
    ready_for_full_de_novo = de_novo_overclaim_flag is False
    head1024_strict_vs_te_only = bool(
        mo_summary.get("head1024_vs_te_only_strict_claim_allowed")
    )
    head256_strict_proxy = bool(mo_summary.get("head256_fusion_vs_te_only_all_strict"))
    ready_for_wet_lab = False

    block_reasons = []
    if not all_sections_ready:
        block_reasons.append("internal_evidence_sections_incomplete")
    if not external_metrics_ready:
        block_reasons.append("external_sota_real_metrics_missing")
    if not ready_for_full_de_novo:
        block_reasons.append("full_de_novo_evidence_missing_or_overclaim_flagged")
    if not ready_for_real_te_or_stability:
        block_reasons.append("real_mpra_te_or_stability_data_missing")
    if not ready_for_true_scale_law:
        block_reasons.append("true_data_model_step_scale_law_missing")
    if not head1024_strict_vs_te_only:
        block_reasons.append("head1024_fusion_vs_strong_te_only_not_strict")
    if not ready_for_wet_lab:
        block_reasons.append("wet_lab_validation_missing")

    positive_sota_claim_ready = (
        all_sections_ready
        and external_metrics_ready
        and ready_for_full_de_novo
        and ready_for_real_te_or_stability
        and ready_for_true_scale_law
        and head1024_strict_vs_te_only
        and ready_for_wet_lab
    )
    return {
        "positive_sota_claim_ready": positive_sota_claim_ready,
        "ready_for_internal_proxy_constrained_optimization_claim": bool(
            all_sections_ready and head256_strict_proxy
        ),
        "ready_for_external_sota_metric_claim": external_metrics_ready,
        "ready_for_full_de_novo_claim": ready_for_full_de_novo,
        "ready_for_real_te_or_stability_claim": ready_for_real_te_or_stability,
        "ready_for_true_scale_law_claim": ready_for_true_scale_law,
        "ready_for_wet_lab_claim": ready_for_wet_lab,
        "head256_fusion_vs_te_only_strict_proxy": head256_strict_proxy,
        "head1024_fusion_vs_strong_te_only_strict": head1024_strict_vs_te_only,
        "external_models_measured": external_summary.get("n_real_run_models_measured"),
        "external_models_expected": external_summary.get("n_real_run_models_expected"),
        "de_novo_overclaim_flag": de_novo_overclaim_flag,
        "block_reasons": block_reasons,
        "evidence_paths": {
            "data_scaleup_readiness": data_path,
            "t2_t3_distribution_novelty": T2_T3_DISTRIBUTION_NOVELTY_REPORT,
            "multiobjective_scaleup_claims": (
                "docs/multiobjective_scaleup_claim_audit_head256_head1024.json"
            ),
            "external_sota_dry_run": "benchmark/external_sota/dry_run_t5_head1024/summary.json",
            "external_sota_real_run_audit": "docs/external_sota_real_run_audit.json",
        },
        "allowed_claim_scope": (
            "Constrained local full-length mRNA optimization/reranking with "
            "proxy/offline T1-T7 evidence. Do not state full de novo, wet-lab, "
            "external SOTA, or true scale-law claims until the blocking gates clear."
        ),
    }


def _audit_sota_readiness_development(
    *,
    project_root: str,
    slice_name: str = "head256",
    top_k: int = 64,
    out_json: Optional[str] = None,
    out_md: Optional[str] = None,
) -> dict[str, object]:
    """Run read-only readiness audits for the current SOTA-gate artifacts."""
    region = audit_region_adapter_results(
        project_root=project_root,
        slice_name=slice_name,
        top_k=top_k,
    )
    prefix = _gc_prefix(project_root, slice_name)
    gc_sweep = audit_protein_conditioned_gc_sweep(
        summary_json=f"{prefix}.summary.json",
        jsonl_path=f"{prefix}.jsonl",
        md_path=f"{prefix}.md",
        project_root=project_root,
    )
    external_sota = _audit_external_sota_protocol(project_root)
    frozen_protocol = _audit_frozen_foundation_protocol(project_root, slice_name)
    multiobjective = audit_multiobjective_scaleup_claims(project_root=project_root)
    t1_t7_bundle = _audit_t1_t7_evidence_bundle(project_root)
    region_audit_prefix = _region_audit_prefix(project_root, slice_name)
    expected_audit_artifacts = {
        "region_adapter": [
            f"{region_audit_prefix}.json",
            f"{region_audit_prefix}.md",
        ],
        "protein_conditioned_gc_sweep": [
            f"{prefix}.audit.json",
            f"{prefix}.audit.md",
        ],
    }
    persisted_audit_artifacts_exist = {
        section: all(os.path.exists(path) for path in paths)
        for section, paths in expected_audit_artifacts.items()
    }
    sections = {
        "region_adapter": {
            "ready": bool(region["summary"]["ready_for_sota_claim_audit"])
            and persisted_audit_artifacts_exist["region_adapter"],
            "persisted_audit_artifacts_exist": persisted_audit_artifacts_exist[
                "region_adapter"
            ],
            "audit": region,
        },
        "protein_conditioned_gc_sweep": {
            "ready": bool(gc_sweep["summary"]["ready_for_pareto_claim_audit"])
            and persisted_audit_artifacts_exist["protein_conditioned_gc_sweep"],
            "persisted_audit_artifacts_exist": persisted_audit_artifacts_exist[
                "protein_conditioned_gc_sweep"
            ],
            "audit": gc_sweep,
        },
        "external_sota_protocol": {
            "ready": bool(external_sota["summary"]["protocol_ready"])
            and bool(external_sota["summary"]["input_pack_ready"])
            and bool(external_sota["summary"]["real_run_audit_complete"])
            and bool(external_sota["summary"]["model_set_consistent"]),
            "persisted_audit_artifacts_exist": True,
            "audit": external_sota,
        },
        "multiobjective_scaleup_claims": {
            "ready": bool(
                multiobjective["summary"]["ready_for_full_hard_constraint_claim_audit"]
            ),
            "persisted_audit_artifacts_exist": True,
            "audit": multiobjective,
        },
        "frozen_foundation_protocol": {
            "ready": bool(frozen_protocol["summary"]["protocol_ready"]),
            "persisted_audit_artifacts_exist": True,
            "audit": frozen_protocol,
        },
        "t1_t7_evidence_bundle": {
            "ready": bool(t1_t7_bundle["summary"]["bundle_ready"]),
            "persisted_audit_artifacts_exist": True,
            "audit": t1_t7_bundle,
        },
    }
    pending_sections = [name for name, row in sections.items() if not bool(row["ready"])]
    missing_artifacts = []
    region_missing = region["summary"].get("missing_artifacts", [])
    if isinstance(region_missing, list):
        for item in region_missing:
            if isinstance(item, Mapping):
                missing_artifacts.append(
                    {"section": "region_adapter", "path": item.get("path")}
                )
    gc_missing = gc_sweep["summary"].get("missing_artifacts", [])
    if isinstance(gc_missing, list):
        for path in gc_missing:
            missing_artifacts.append(
                {"section": "protein_conditioned_gc_sweep", "path": path}
            )
    for section, paths in expected_audit_artifacts.items():
        for path in paths:
            if not os.path.exists(path):
                missing_artifacts.append({"section": section, "path": _rel(path, project_root)})
    external_missing = external_sota["summary"].get("missing_artifacts", [])
    if isinstance(external_missing, list):
        for path in external_missing:
            missing_artifacts.append({"section": "external_sota_protocol", "path": path})
    external_input_pack_missing = external_sota["summary"].get("input_pack_missing_artifacts", [])
    if isinstance(external_input_pack_missing, list):
        for path in external_input_pack_missing:
            missing_artifacts.append({"section": "external_sota_input_pack", "path": path})
    external_real_run_missing = external_sota["summary"].get("real_run_missing_artifacts", [])
    if isinstance(external_real_run_missing, list):
        for path in external_real_run_missing:
            missing_artifacts.append({"section": "external_sota_real_run_audit", "path": path})
    frozen_missing = frozen_protocol["summary"].get("missing_artifacts", [])
    if isinstance(frozen_missing, list):
        for path in frozen_missing:
            missing_artifacts.append({"section": "frozen_foundation_protocol", "path": path})
    t1_t7_missing = t1_t7_bundle["summary"].get("missing_artifacts", [])
    if isinstance(t1_t7_missing, list):
        for path in t1_t7_missing:
            missing_artifacts.append({"section": "t1_t7_evidence_bundle", "path": path})
    t1_t7_failed = t1_t7_bundle["summary"].get("failed_checks", [])
    if isinstance(t1_t7_failed, list):
        for name in t1_t7_failed:
            missing_artifacts.append(
                {"section": "t1_t7_evidence_bundle", "path": f"failed_check:{name}"}
            )
    for row in multiobjective.get("comparisons", []):
        if isinstance(row, Mapping) and not bool(row.get("exists")):
            missing_artifacts.append(
                {"section": "multiobjective_scaleup_claims", "path": row.get("source")}
            )
    for row in multiobjective.get("summary_constraints", []):
        if isinstance(row, Mapping) and not bool(row.get("exists")):
            missing_artifacts.append(
                {"section": "multiobjective_scaleup_claims", "path": row.get("path")}
            )
    all_sections_ready = not pending_sections
    positive_claim_gate = _positive_sota_claim_gate(
        project_root=project_root,
        all_sections_ready=all_sections_ready,
        external_sota=external_sota,
        multiobjective=multiobjective,
    )
    payload = {
        "artifact_kind": "sota_readiness_audit",
        "slice": slice_name,
        "top_k": int(top_k),
        "summary": {
            "all_ready_for_sota_claim_audit": all_sections_ready,
            "positive_sota_claim_ready": positive_claim_gate["positive_sota_claim_ready"],
            "ready_for_internal_proxy_constrained_optimization_claim": positive_claim_gate[
                "ready_for_internal_proxy_constrained_optimization_claim"
            ],
            "ready_for_external_sota_metric_claim": positive_claim_gate[
                "ready_for_external_sota_metric_claim"
            ],
            "ready_for_full_de_novo_claim": positive_claim_gate[
                "ready_for_full_de_novo_claim"
            ],
            "ready_for_real_te_or_stability_claim": positive_claim_gate[
                "ready_for_real_te_or_stability_claim"
            ],
            "ready_for_true_scale_law_claim": positive_claim_gate[
                "ready_for_true_scale_law_claim"
            ],
            "ready_for_wet_lab_claim": positive_claim_gate["ready_for_wet_lab_claim"],
            "positive_sota_block_reasons": positive_claim_gate["block_reasons"],
            "allowed_claim_scope": positive_claim_gate["allowed_claim_scope"],
            "pending_sections": pending_sections,
            "n_sections_ready": len(sections) - len(pending_sections),
            "n_sections_expected": len(sections),
            "missing_artifacts": missing_artifacts,
            "persisted_audit_artifacts_exist": persisted_audit_artifacts_exist,
            "claim_policy": (
                "Ready only means evidence is complete and hard constraints are auditable; "
                "effect sizes and p-values still determine whether claims are positive, "
                "borderline, non-significant, or negative."
            ),
        },
        "positive_claim_gate": positive_claim_gate,
        "sections": sections,
    }
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
    if out_md:
        write_markdown(payload, out_md)
    return payload


def audit_sota_readiness(
    *,
    project_root: str,
    slice_name: str = "head256",
    top_k: int = 64,
    out_json: Optional[str] = None,
    out_md: Optional[str] = None,
    run_mode: str = "development",
    artifact_paths: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    if normalize_run_mode(run_mode) == "paper":
        payload = paper_builder_gate("sota_readiness_audit", project_root, artifact_paths, __file__)
        validate_report_output_namespaces(payload, (out_json, out_md))
        if out_json:
            os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
        if out_md:
            os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
            with open(out_md, "w", encoding="utf-8") as fh:
                fh.write("# Paper SOTA Readiness Gate\n\n")
                fh.write(f"- Status: `{payload['status']}`\n")
                for reason in payload["block_reasons"]:
                    fh.write(f"- Block reason: `{reason}`\n")
        write_paper_report_sidecars(payload, (out_json, out_md))
        return payload
    payload = _audit_sota_readiness_development(
        project_root=project_root,
        slice_name=slice_name,
        top_k=top_k,
        out_json=None,
        out_md=None,
    )
    payload.update({"claim_tier": "development_only", "paper_eligible": False})
    validate_report_output_namespaces(payload, (out_json, out_md))
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
    if out_md:
        write_markdown(payload, out_md)
    return payload


def write_markdown(payload: Mapping[str, object], out_md: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
    summary = payload.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    sections = payload.get("sections", {})
    if not isinstance(sections, Mapping):
        sections = {}
    lines = [
        "# mRNA-EditFlow SOTA Readiness Audit",
        "",
        f"- Slice: {payload.get('slice')}",
        f"- Top-k: {payload.get('top_k')}",
        f"- All ready for SOTA claim audit: {summary.get('all_ready_for_sota_claim_audit')}",
        f"- Positive SOTA claim ready: {summary.get('positive_sota_claim_ready')}",
        (
            "- Internal proxy constrained-optimization claim ready: "
            f"{summary.get('ready_for_internal_proxy_constrained_optimization_claim')}"
        ),
        f"- External SOTA metric claim ready: {summary.get('ready_for_external_sota_metric_claim')}",
        f"- Full de novo claim ready: {summary.get('ready_for_full_de_novo_claim')}",
        f"- Real TE/stability claim ready: {summary.get('ready_for_real_te_or_stability_claim')}",
        f"- True scale-law claim ready: {summary.get('ready_for_true_scale_law_claim')}",
        f"- Wet-lab claim ready: {summary.get('ready_for_wet_lab_claim')}",
        f"- Positive SOTA blockers: {summary.get('positive_sota_block_reasons')}",
        f"- Allowed claim scope: {summary.get('allowed_claim_scope')}",
        f"- Sections ready: {summary.get('n_sections_ready')}/{summary.get('n_sections_expected')}",
        f"- Pending sections: {summary.get('pending_sections')}",
        f"- Claim policy: {summary.get('claim_policy')}",
        "",
        "| section | ready | key status |",
        "|---|---:|---|",
    ]
    for name, row in sections.items():
        if not isinstance(row, Mapping):
            continue
        audit = row.get("audit", {})
        audit_summary = audit.get("summary", {}) if isinstance(audit, Mapping) else {}
        if not isinstance(audit_summary, Mapping):
            audit_summary = {}
        if name == "region_adapter":
            status = (
                f"compare files {audit_summary.get('n_compare_files_found')}/"
                f"{audit_summary.get('n_compare_files_expected')}; "
                f"constraints exact 1={audit_summary.get('all_constraints_exact_1')}; "
                f"persisted audit={row.get('persisted_audit_artifacts_exist')}"
            )
        elif name == "protein_conditioned_gc_sweep":
            status = (
                f"points={audit_summary.get('n_points')}; "
                f"identity exact 1={audit_summary.get('all_points_identity_exact_1')}; "
                f"pareto metadata={audit_summary.get('pareto_metadata_ok')}; "
                f"persisted audit={row.get('persisted_audit_artifacts_exist')}"
            )
        elif name == "external_sota_protocol":
            status = (
                f"protocol ready={audit_summary.get('protocol_ready')}; "
                f"input pack ready={audit_summary.get('input_pack_ready')}; "
                f"real-run audit={audit_summary.get('real_run_audit_complete')}; "
                f"model set consistent={audit_summary.get('model_set_consistent')}; "
                f"measured={audit_summary.get('n_real_run_models_measured')}/"
                f"{audit_summary.get('n_real_run_models_expected')}; "
                f"T5 UTR descriptive/model-only="
                f"{audit_summary.get('t5_utr_descriptive_table_ready')}/"
                f"{audit_summary.get('t5_utr_model_only_head_to_head_ready')}; "
                f"input rows={audit_summary.get('n_input_pack_cds_protein_rows')}/"
                f"{audit_summary.get('n_input_pack_utr5_rows')}; "
                f"models={audit_summary.get('n_models')}; "
                f"executable ready={audit_summary.get('n_executable_ready')}; "
                "real metrics not claimed"
            )
        elif name == "multiobjective_scaleup_claims":
            status = (
                f"comparison rows={audit_summary.get('comparison_rows_ready')}; "
                f"hard constraints={audit_summary.get('summary_constraints_complete')}; "
                f"head256 strict={audit_summary.get('head256_fusion_vs_te_only_all_strict')}; "
                "head1024 vs te_only strict="
                f"{audit_summary.get('head1024_vs_te_only_strict_claim_allowed')}; "
                f"best signal={audit_summary.get('head1024_vs_te_only_best_signal')}"
            )
        elif name == "frozen_foundation_protocol":
            status = (
                f"protocol ready={audit_summary.get('protocol_ready')}; "
                f"leakage gate={audit_summary.get('leakage_gate_passed')}; "
                f"matched budget={audit_summary.get('matched_budget')}; "
                f"real/stub arms={audit_summary.get('n_real_arms')}/"
                f"{audit_summary.get('n_stub_arms')}; real metrics not claimed"
            )
        elif name == "t1_t7_evidence_bundle":
            status = (
                f"reports ready={audit_summary.get('n_reports_ready')}/"
                f"{audit_summary.get('n_reports_expected')}; "
                f"failed checks={audit_summary.get('failed_checks')}; "
                "proxy/constraint reports only"
            )
        else:
            status = ""
        lines.append(f"| {name} | {row.get('ready')} | {status} |")
    lines.extend(["", "## Missing Artifacts", "", "| section | path |", "|---|---|"])
    for item in summary.get("missing_artifacts", []):
        if isinstance(item, Mapping):
            lines.append(f"| {item.get('section')} | `{item.get('path')}` |")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--slice", dest="slice_name", default="head256")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--run-mode", choices=("development", "paper"), default="development")
    parser.add_argument("--paper-artifact", action="append", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    payload = audit_sota_readiness(
        project_root=args.project_root,
        slice_name=args.slice_name,
        top_k=args.top_k,
        out_json=args.out_json,
        out_md=args.out_md,
        run_mode=args.run_mode,
        artifact_paths=args.paper_artifact,
    )
    print(json.dumps({"summary": payload.get("summary", {}), "out_json": args.out_json, "out_md": args.out_md}, sort_keys=True))
    if args.strict and not payload.get("summary", {}).get("all_ready_for_sota_claim_audit", False):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
