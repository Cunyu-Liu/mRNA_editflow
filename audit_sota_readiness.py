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


def _frozen_protocol_dir(project_root: str, slice_name: str) -> str:
    return os.path.join(project_root, "benchmark", f"frozen_backbone_protocol_{slice_name}")


def _load_json_if_exists(path: str) -> Optional[Mapping[str, object]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _audit_external_sota_protocol(project_root: str) -> dict[str, object]:
    dry_run_dir = _external_sota_dir(project_root)
    summary_path = os.path.join(dry_run_dir, "summary.json")
    runtime_path = os.path.join(dry_run_dir, "runtime.json")
    table_path = os.path.join(dry_run_dir, "table.md")
    summary = _load_json_if_exists(summary_path)
    runtime = _load_json_if_exists(runtime_path)
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
    missing_artifacts = [
        _rel(path, project_root)
        for path in (summary_path, runtime_path, table_path)
        if not os.path.exists(path)
    ]
    return {
        "artifact_kind": "external_sota_protocol_audit",
        "summary_path": _rel(summary_path, project_root),
        "runtime_path": _rel(runtime_path, project_root),
        "table_path": _rel(table_path, project_root),
        "summary": {
            "protocol_ready": protocol_ready,
            "summary_exists": summary is not None,
            "runtime_exists": runtime is not None,
            "table_exists": os.path.exists(table_path),
            "dataset_ok": dataset_ok,
            "rows_ok": rows_ok,
            "required_metadata_ok": required_metadata_ok,
            "n_models": len(rows),
            "n_executable_ready": summary.get("n_executable_ready") if summary else None,
            "missing_artifacts": missing_artifacts,
            "real_metric_policy": (
                contract.get("real_metric_policy") if isinstance(contract, Mapping) else None
            ),
        },
        "rows": rows,
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


def audit_sota_readiness(
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
            "ready": bool(external_sota["summary"]["protocol_ready"]),
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
    frozen_missing = frozen_protocol["summary"].get("missing_artifacts", [])
    if isinstance(frozen_missing, list):
        for path in frozen_missing:
            missing_artifacts.append({"section": "frozen_foundation_protocol", "path": path})
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
    payload = {
        "artifact_kind": "sota_readiness_audit",
        "slice": slice_name,
        "top_k": int(top_k),
        "summary": {
            "all_ready_for_sota_claim_audit": not pending_sections,
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
        "sections": sections,
    }
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
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    payload = audit_sota_readiness(
        project_root=args.project_root,
        slice_name=args.slice_name,
        top_k=args.top_k,
        out_json=args.out_json,
        out_md=args.out_md,
    )
    print(json.dumps({"summary": payload["summary"], "out_json": args.out_json, "out_md": args.out_md}, sort_keys=True))
    if args.strict and not payload["summary"]["all_ready_for_sota_claim_audit"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
