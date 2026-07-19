"""Build paper Table 3: external baseline readiness and runtime protocol.

This table summarizes the external SOTA dry-run contract. It is deliberately
not a performance table while all external rows are ``not_configured``. Real
external TE/F1/runtime metrics remain forbidden until executable adapters write
measured outputs under ``benchmark/external_sota/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Mapping, Optional, Sequence

from mrna_editflow.eval.artifact_contract import normalize_run_mode, paper_builder_gate, validate_report_output_namespaces, write_paper_report_sidecars


CLAIM_POLICY = (
    "Table 3 is an external-baseline readiness/protocol table. Do not report "
    "external TE/F1/runtime performance or SOTA comparisons until a row is "
    "executable_ready and a real adapter writes measured outputs under "
    "benchmark/external_sota/."
)
DEFAULT_SUMMARY = "benchmark/external_sota/dry_run_t5_head1024/summary.json"
DEFAULT_RUNTIME = "benchmark/external_sota/dry_run_t5_head1024/runtime.json"
DEFAULT_TABLE = "benchmark/external_sota/dry_run_t5_head1024/table.md"
DEFAULT_INPUT_PACK = "benchmark/external_sota/input_pack_t5_head1024/summary.json"
DEFAULT_REAL_RUN_AUDIT = "docs/external_sota_real_run_audit.json"


def _load_json(path: str) -> Mapping[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _path(project_root: str, rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(project_root, rel)


def _sha256_file(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_optional_json(path: str) -> Optional[Mapping[str, object]]:
    if not os.path.exists(path):
        return None
    return _load_json(path)


def _candidate_audit_text(row: Mapping[str, object]) -> str:
    audit = row.get("candidate_audit", [])
    if not isinstance(audit, Sequence) or isinstance(audit, (str, bytes)):
        return ""
    parts = []
    for item in audit:
        if isinstance(item, Mapping):
            parts.append(f"{item.get('candidate')}={item.get('status')}")
    return "; ".join(parts)


def _input_pack_models(input_pack: Optional[Mapping[str, object]]) -> set[str]:
    if not isinstance(input_pack, Mapping):
        return set()
    groups = input_pack.get("models", {})
    if not isinstance(groups, Mapping):
        return set()
    models: set[str] = set()
    for values in groups.values():
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            models.update(str(value) for value in values)
    return models


def _required_metadata_ok(summary: Mapping[str, object], runtime: Mapping[str, object]) -> bool:
    dataset = summary.get("dataset", {})
    contract = summary.get("artifact_contract", {})
    if not isinstance(dataset, Mapping) or not isinstance(contract, Mapping):
        return False
    required = contract.get("required_real_run_metadata", [])
    if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
        return False
    required_set = set(str(item) for item in required)
    expected = {
        "dataset.sha256",
        "dataset.split_name",
        "dataset.seed",
        "dataset.record_count_effective",
        "runtime.elapsed_s",
        "hardware",
    }
    return (
        expected <= required_set
        and bool(dataset.get("exists"))
        and bool(dataset.get("sha256"))
        and dataset.get("sha256") == runtime.get("dataset_sha256")
        and isinstance(runtime.get("elapsed_s"), (int, float))
        and isinstance(runtime.get("hardware"), Mapping)
    )


def _build_paper_table3_development(
    project_root: str,
    *,
    summary_rel: str = DEFAULT_SUMMARY,
    runtime_rel: str = DEFAULT_RUNTIME,
    table_rel: str = DEFAULT_TABLE,
    input_pack_rel: str = DEFAULT_INPUT_PACK,
    real_run_audit_rel: str = DEFAULT_REAL_RUN_AUDIT,
) -> dict[str, object]:
    summary_path = _path(project_root, summary_rel)
    runtime_path = _path(project_root, runtime_rel)
    table_path = _path(project_root, table_rel)
    input_pack_path = _path(project_root, input_pack_rel)
    real_run_audit_path = _path(project_root, real_run_audit_rel)
    summary = _load_json(summary_path)
    runtime = _load_json(runtime_path)
    input_pack = _load_optional_json(input_pack_path)
    real_run_audit = _load_optional_json(real_run_audit_path)
    rows_raw = summary.get("rows", [])
    rows_raw = rows_raw if isinstance(rows_raw, Sequence) and not isinstance(rows_raw, (str, bytes)) else []
    dataset = summary.get("dataset", {})
    dataset = dataset if isinstance(dataset, Mapping) else {}
    hardware = summary.get("hardware", {})
    hardware = hardware if isinstance(hardware, Mapping) else {}
    real_metric_policy = ""
    contract = summary.get("artifact_contract", {})
    if isinstance(contract, Mapping):
        real_metric_policy = str(contract.get("real_metric_policy", ""))
    real_run_rows_raw = real_run_audit.get("rows", []) if isinstance(real_run_audit, Mapping) else []
    if not isinstance(real_run_rows_raw, Sequence) or isinstance(real_run_rows_raw, (str, bytes)):
        real_run_rows_raw = []
    real_run_rows_by_model = {
        row.get("model_name"): row
        for row in real_run_rows_raw
        if isinstance(row, Mapping) and row.get("model_name")
    }
    dry_run_models = {
        str(row.get("model_name"))
        for row in rows_raw
        if isinstance(row, Mapping) and row.get("model_name")
    }
    input_pack_models = _input_pack_models(input_pack)
    real_run_models = {str(name) for name in real_run_rows_by_model}
    model_set_consistent = bool(
        dry_run_models
        and input_pack_models
        and real_run_models
        and dry_run_models == input_pack_models == real_run_models
    )

    rows = []
    for row in rows_raw:
        if not isinstance(row, Mapping):
            continue
        status = row.get("status")
        executable_ready = status == "executable_ready"
        measured = real_run_rows_by_model.get(row.get("model_name"))
        measured_ready = bool(isinstance(measured, Mapping) and measured.get("real_metric_ready") is True)
        real_runtime_ready = bool(isinstance(measured, Mapping) and measured.get("real_runtime_ready") is True)
        rows.append(
            {
                "model_name": row.get("model_name"),
                "family": row.get("family"),
                "dry_run_status": status,
                "executable": row.get("executable"),
                "executable_source": row.get("executable_source"),
                "command_candidates": row.get("command_candidates", []),
                "candidate_audit": row.get("candidate_audit", []),
                "candidate_audit_text": _candidate_audit_text(row),
                "expected_inputs": row.get("expected_inputs"),
                "expected_outputs": row.get("expected_outputs"),
                "protocol_difference": row.get("protocol_difference"),
                "dataset_split": dataset.get("split_name"),
                "dataset_sha256": dataset.get("sha256"),
                "dataset_records": dataset.get("record_count_effective"),
                "dry_run_elapsed_s": runtime.get("elapsed_s"),
                "hardware_label": hardware.get("label"),
                "real_run_status": measured.get("status") if isinstance(measured, Mapping) else "missing_audit_row",
                "real_metric_ready": measured_ready,
                "real_runtime_ready": real_runtime_ready,
                "success_fraction": measured.get("success_fraction") if isinstance(measured, Mapping) else None,
                "hard_constraints_exact_1": (
                    measured.get("hard_constraints_exact_1") if isinstance(measured, Mapping) else None
                ),
                "real_run_failure_reasons": (
                    measured.get("failure_reasons") if isinstance(measured, Mapping) else []
                ),
                "claim_language": (
                    "measured_external_metrics_available_not_sota_claim"
                    if measured_ready
                    else (
                        "adapter_executable_but_real_metrics_still_required"
                        if executable_ready
                        else "not_configured_no_external_metric_claim"
                    )
                ),
                "next_action": (
                    "Run head-to-head comparison and claim-language audit before any superiority claim."
                    if measured_ready
                    else (
                        "Run real adapter and write measured outputs under benchmark/external_sota/."
                        if executable_ready
                        else "Install executable or set documented *_BIN/PATH candidate, then rerun dry-run and real adapter."
                    )
                ),
            }
        )
    n_executable_ready = sum(1 for row in rows if row["dry_run_status"] == "executable_ready")
    n_not_configured = sum(1 for row in rows if row["dry_run_status"] == "not_configured")
    metadata_ok = _required_metadata_ok(summary, runtime)
    input_pack_ready = (
        isinstance(input_pack, Mapping)
        and input_pack.get("artifact_kind") == "external_sota_input_pack"
        and input_pack.get("ready_for_external_real_run") is True
        and isinstance(input_pack.get("n_cds_protein_rows"), int)
        and isinstance(input_pack.get("n_utr5_rows"), int)
    )
    input_pack_summary = {
        "present": input_pack is not None,
        "ready_for_external_real_run": bool(input_pack_ready),
        "path": input_pack_rel,
        "sha256": _sha256_file(input_pack_path),
        "n_cds_protein_rows": input_pack.get("n_cds_protein_rows") if isinstance(input_pack, Mapping) else None,
        "n_utr5_rows": input_pack.get("n_utr5_rows") if isinstance(input_pack, Mapping) else None,
        "n_skipped_invalid_cds": input_pack.get("n_skipped_invalid_cds") if isinstance(input_pack, Mapping) else None,
    }
    real_run_summary_raw = real_run_audit.get("summary", {}) if isinstance(real_run_audit, Mapping) else {}
    real_run_summary_raw = real_run_summary_raw if isinstance(real_run_summary_raw, Mapping) else {}
    real_run_summary = {
        "present": real_run_audit is not None,
        "path": real_run_audit_rel,
        "sha256": _sha256_file(real_run_audit_path),
        "audit_complete": bool(real_run_summary_raw.get("audit_complete")),
        "ready_for_external_real_metric_table": bool(
            real_run_summary_raw.get("ready_for_external_real_metric_table")
        ),
        "ready_for_external_sota_metric_claim": bool(
            real_run_summary_raw.get("ready_for_external_sota_metric_claim")
        ),
        "ready_for_external_sota_claim": bool(
            real_run_summary_raw.get("ready_for_external_sota_claim")
        ),
        "n_models_expected": real_run_summary_raw.get("n_models_expected"),
        "n_models_measured": real_run_summary_raw.get("n_models_measured"),
        "n_models_invalid": real_run_summary_raw.get("n_models_invalid"),
        "n_models_missing": real_run_summary_raw.get("n_models_missing"),
    }
    ready_for_real_metric_table = bool(real_run_summary["ready_for_external_real_metric_table"])
    return {
        "artifact_kind": "paper_table3_external_baseline_readiness",
        "project_root": os.path.abspath(project_root),
        "claim_policy": CLAIM_POLICY,
        "sources": {
            "summary_json": summary_rel,
            "summary_sha256": _sha256_file(summary_path),
            "runtime_json": runtime_rel,
            "runtime_sha256": _sha256_file(runtime_path),
            "table_md": table_rel,
            "table_sha256": _sha256_file(table_path),
            "input_pack_summary_json": input_pack_rel,
            "input_pack_summary_sha256": _sha256_file(input_pack_path),
            "real_run_audit_json": real_run_audit_rel,
            "real_run_audit_sha256": _sha256_file(real_run_audit_path),
        },
        "dataset": dict(dataset),
        "hardware": dict(hardware),
        "input_pack": input_pack_summary,
        "real_run_audit": real_run_summary,
        "model_set_audit": {
            "dry_run_models": sorted(dry_run_models),
            "input_pack_models": sorted(input_pack_models),
            "real_run_audit_models": sorted(real_run_models),
            "model_set_consistent": model_set_consistent,
            "missing_from_dry_run": sorted((input_pack_models | real_run_models) - dry_run_models),
            "missing_from_input_pack": sorted((dry_run_models | real_run_models) - input_pack_models),
            "missing_from_real_run_audit": sorted((dry_run_models | input_pack_models) - real_run_models),
        },
        "summary": {
            "task_id": summary.get("task_id"),
            "n_models": len(rows),
            "n_executable_ready": n_executable_ready,
            "n_not_configured": n_not_configured,
            "metadata_contract_ok": metadata_ok,
            "input_pack_ready": bool(input_pack_ready),
            "real_run_audit_complete": bool(real_run_summary["audit_complete"]),
            "model_set_consistent": model_set_consistent,
            "ready_for_protocol_table": bool(rows) and metadata_ok,
            "ready_for_full_external_protocol_bundle": bool(
                rows
                and metadata_ok
                and input_pack_ready
                and real_run_summary["audit_complete"]
                and model_set_consistent
            ),
            "ready_for_real_metric_table": ready_for_real_metric_table,
            "ready_for_external_sota_metric_claim": bool(
                real_run_summary["ready_for_external_sota_metric_claim"]
            ),
            "ready_for_external_sota_claim": False,
            "real_metric_policy": real_metric_policy,
        },
        "rows": rows,
    }


def build_paper_table3(
    project_root: str,
    run_mode: str = "development",
    artifact_paths: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    if normalize_run_mode(run_mode) == "paper":
        return paper_builder_gate("paper_table3_external_baselines", project_root, artifact_paths, __file__)
    report = _build_paper_table3_development(project_root)
    report.update({"claim_tier": "development_only", "paper_eligible": False})
    return report


def write_report_json(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return path


def write_report_markdown(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows = report.get("rows", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        rows = []
    summary = report.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    dataset = report.get("dataset", {})
    if not isinstance(dataset, Mapping):
        dataset = {}
    hardware = report.get("hardware", {})
    if not isinstance(hardware, Mapping):
        hardware = {}
    input_pack = report.get("input_pack", {})
    if not isinstance(input_pack, Mapping):
        input_pack = {}
    real_run = report.get("real_run_audit", {})
    if not isinstance(real_run, Mapping):
        real_run = {}
    model_set = report.get("model_set_audit", {})
    if not isinstance(model_set, Mapping):
        model_set = {}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Paper Table 3: External Baseline Readiness\n\n")
        fh.write(f"- Claim policy: {report.get('claim_policy', CLAIM_POLICY)}\n")
        fh.write(
            f"- Task: `{summary.get('task_id')}`; executable ready: "
            f"`{summary.get('n_executable_ready')}` / `{summary.get('n_models')}`; "
            f"real metric table ready: `{summary.get('ready_for_real_metric_table')}`\n"
        )
        fh.write(
            f"- Dataset: split=`{dataset.get('split_name')}`, records=`{dataset.get('record_count_effective')}`, "
            f"sha256=`{dataset.get('sha256')}`\n"
        )
        fh.write(
            f"- Hardware: label=`{hardware.get('label')}`, host=`{hardware.get('hostname')}`, "
            f"machine=`{hardware.get('machine')}`\n\n"
        )
        fh.write("## External Input Pack\n\n")
        fh.write(
            f"- Present: `{input_pack.get('present')}`; "
            f"ready for external real run: `{input_pack.get('ready_for_external_real_run')}`; "
            f"path: `{input_pack.get('path')}`\n"
        )
        fh.write(
            f"- CDS/protein-conditioned rows: `{input_pack.get('n_cds_protein_rows')}`; "
            f"5'UTR-only rows: `{input_pack.get('n_utr5_rows')}`; "
            f"skipped invalid CDS: `{input_pack.get('n_skipped_invalid_cds')}`; "
            f"sha256: `{input_pack.get('sha256')}`\n\n"
        )
        fh.write("## External Real-Run Audit\n\n")
        fh.write(
            f"- Present: `{real_run.get('present')}`; audit complete: "
            f"`{real_run.get('audit_complete')}`; path: `{real_run.get('path')}`\n"
        )
        fh.write(
            f"- Measured models: `{real_run.get('n_models_measured')}` / "
            f"`{real_run.get('n_models_expected')}`; invalid: "
            f"`{real_run.get('n_models_invalid')}`; missing: "
            f"`{real_run.get('n_models_missing')}`; real metric table ready: "
            f"`{real_run.get('ready_for_external_real_metric_table')}`; "
            f"sha256: `{real_run.get('sha256')}`\n\n"
        )
        fh.write("## Model Set Consistency\n\n")
        fh.write(
            f"- Consistent: `{model_set.get('model_set_consistent')}`; "
            f"dry-run: `{model_set.get('dry_run_models')}`; "
            f"input pack: `{model_set.get('input_pack_models')}`; "
            f"real-run audit: `{model_set.get('real_run_audit_models')}`\n"
        )
        fh.write(
            f"- Missing from dry-run: `{model_set.get('missing_from_dry_run')}`; "
            f"missing from input pack: `{model_set.get('missing_from_input_pack')}`; "
            f"missing from real-run audit: `{model_set.get('missing_from_real_run_audit')}`\n\n"
        )
        fh.write("| Model | Dry-run status | Real-run status | Candidate audit | Dataset/runtime audit | Protocol gap | Claim language |\n")
        fh.write("|---|---|---|---|---|---|---|\n")
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            real_text = (
                f"{row.get('real_run_status')}; metric_ready={row.get('real_metric_ready')}; "
                f"success={row.get('success_fraction')}; constraints={row.get('hard_constraints_exact_1')}"
            )
            dataset_text = (
                f"split={row.get('dataset_split')}; records={row.get('dataset_records')}; "
                f"dry_run_elapsed_s={row.get('dry_run_elapsed_s')}; hardware={row.get('hardware_label')}"
            )
            fh.write(
                f"| {row.get('model_name')} | `{row.get('dry_run_status')}` | "
                f"{real_text} | {row.get('candidate_audit_text')} | {dataset_text} | "
                f"{row.get('protocol_difference')} | {row.get('claim_language')} |\n"
            )
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--out-json", default="docs/paper_table3_external_baseline_readiness.json")
    parser.add_argument("--out-md", default="docs/paper_table3_external_baseline_readiness.md")
    parser.add_argument("--run-mode", choices=("development", "paper"), default="development")
    parser.add_argument("--paper-artifact", action="append", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    project_root = os.path.abspath(args.project_root)
    report = build_paper_table3(project_root, args.run_mode, args.paper_artifact)
    out_json = args.out_json if os.path.isabs(args.out_json) else os.path.join(project_root, args.out_json)
    out_md = args.out_md if os.path.isabs(args.out_md) else os.path.join(project_root, args.out_md)
    validate_report_output_namespaces(report, (out_json, out_md))
    write_report_json(report, out_json)
    write_report_markdown(report, out_md)
    write_paper_report_sidecars(report, (out_json, out_md))
    print(json.dumps({"json_path": out_json, "markdown_path": out_md}, sort_keys=True))
    if args.run_mode == "paper" and args.paper_artifact and not report["paper_eligible"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLAIM_POLICY",
    "build_paper_table3",
    "write_report_json",
    "write_report_markdown",
    "main",
]
