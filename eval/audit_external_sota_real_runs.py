"""Audit measured outputs from real external SOTA adapters.

The input-pack and dry-run artifacts prove that a fair protocol is ready, but
they do not prove that LinearDesign, EnsembleDesign, codonGPT, Prot2RNA or
UTailoR has actually been executed. This module audits the measured output
contract that future real adapters must write before Table 3 can report
external metrics.

No external executable is launched here and no metric is fabricated. Missing
real-run outputs are recorded as missing evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from typing import Mapping, Optional, Sequence

from mrna_editflow.baselines.external_sota_input_pack import (
    CDS_MODELS,
    UTR_MODELS,
    external_metric_schema,
)
from mrna_editflow.core.constants import is_valid_cds, translate


CLAIM_POLICY = (
    "External real-run audit validates adapter-written measured outputs against "
    "the standardized input pack and metric schema. Passing this audit permits "
    "reporting external metric rows, but does not by itself prove that MEF beats "
    "external SOTA methods."
)
DEFAULT_INPUT_PACK = "benchmark/external_sota/input_pack_t5_head1024/summary.json"
DEFAULT_REAL_RUN_DIR = "benchmark/external_sota/real_runs_t5_head1024"
DEFAULT_OUT_JSON = "docs/external_sota_real_run_audit.json"
DEFAULT_OUT_MD = "docs/external_sota_real_run_audit.md"


def _path(project_root: str, rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(project_root, rel)


def _rel(path: str, project_root: str) -> str:
    return os.path.relpath(path, project_root) if os.path.isabs(path) else path


def _load_json(path: str) -> Optional[Mapping[str, object]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_jsonl(path: str) -> list[dict[str, object]]:
    if not os.path.exists(path):
        return []
    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_no} must contain a JSON object")
            rows.append(dict(payload))
    return rows


def _sha256_file(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _nested(payload: Mapping[str, object], dotted: str) -> object:
    cur: object = payload
    for part in dotted.split("."):
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(part)
    return cur


def _resolve_pack_path(input_pack_summary_path: str, input_pack: Mapping[str, object], key: str, default_name: str) -> str:
    outputs = _mapping(input_pack.get("outputs"))
    value = outputs.get(key)
    if isinstance(value, str):
        candidate = value if os.path.isabs(value) else os.path.join(os.path.dirname(input_pack_summary_path), value)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(os.path.dirname(input_pack_summary_path), default_name)


def _model_task_family(model_name: str) -> Optional[str]:
    if model_name in CDS_MODELS:
        return "cds_protein_conditioned"
    if model_name in UTR_MODELS:
        return "utr5_only"
    return None


def _model_dir_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def _resolve_model_dir(
    real_run_dir: str,
    model_name: str,
    output_filename: str,
) -> tuple[str, str]:
    variants = (
        ("paper_default_10000_steps", "UTRGAN_paper10000"),
        ("canonical", _model_dir_name(model_name)),
    ) if model_name == "UTRGAN" else (
        ("canonical", _model_dir_name(model_name)),
    )
    for variant, directory_name in variants:
        model_dir = os.path.join(real_run_dir, directory_name)
        if (
            os.path.isfile(os.path.join(model_dir, "summary.json"))
            and os.path.isfile(os.path.join(model_dir, output_filename))
        ):
            return model_dir, variant
    variant, directory_name = variants[-1]
    return os.path.join(real_run_dir, directory_name), variant


def _expected_models(input_pack: Optional[Mapping[str, object]], models: Optional[Sequence[str]]) -> list[str]:
    if models:
        return [str(item) for item in models]
    pack_models = _mapping(input_pack.get("models") if input_pack else {})
    cds = pack_models.get("cds_protein_conditioned", list(CDS_MODELS))
    utr = pack_models.get("utr5_only", list(UTR_MODELS))
    result: list[str] = []
    if isinstance(cds, Sequence) and not isinstance(cds, (str, bytes)):
        result.extend(str(item) for item in cds)
    if isinstance(utr, Sequence) and not isinstance(utr, (str, bytes)):
        result.extend(str(item) for item in utr)
    return result


def _required_metadata_checks(
    summary: Mapping[str, object],
    *,
    model_name: str,
    task_family: str,
    input_pack: Mapping[str, object],
    input_pack_summary_path: str,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    dataset = _mapping(input_pack.get("dataset"))
    outputs = _mapping(input_pack.get("outputs"))
    expected = {
        "artifact_kind": "external_sota_real_run_summary",
        "model_name": model_name,
        "task_family": task_family,
        "input_pack.summary_sha256": _sha256_file(input_pack_summary_path),
        "dataset.records_jsonl_sha256": dataset.get("records_jsonl_sha256"),
        "dataset.split_name": dataset.get("split_name"),
        "dataset.seed": dataset.get("seed"),
    }
    for key, expected_value in expected.items():
        observed = summary.get(key) if "." not in key else _nested(summary, key)
        if observed != expected_value:
            failures.append(f"{key}_mismatch")
    for key in (
        "input_pack.cds_protein_jsonl_sha256",
        "input_pack.utr5_jsonl_sha256",
    ):
        output_key = key.split(".")[-1]
        expected_value = outputs.get(output_key)
        observed = _nested(summary, key)
        if expected_value and observed != expected_value:
            failures.append(f"{key}_mismatch")
    if not _finite(_nested(summary, "runtime.elapsed_s")):
        failures.append("runtime.elapsed_s_missing_or_nonfinite")
    if not isinstance(summary.get("hardware"), Mapping):
        failures.append("hardware_missing")
    if not _nested(summary, "executable.path"):
        failures.append("executable.path_missing")
    if not _nested(summary, "executable.version"):
        failures.append("executable.version_missing")
    return not failures, failures


def _required_summary_fields_ok(
    summary: Mapping[str, object],
    required_fields: Sequence[object],
) -> tuple[bool, list[str]]:
    missing_or_bad: list[str] = []
    for field in required_fields:
        key = str(field)
        if key not in summary:
            missing_or_bad.append(key)
            continue
        if key.startswith("n_"):
            if not isinstance(summary.get(key), int):
                missing_or_bad.append(key)
        elif not _finite(summary.get(key)):
            missing_or_bad.append(key)
    return not missing_or_bad, missing_or_bad


def _required_row_fields_ok(
    rows: Sequence[Mapping[str, object]],
    required_fields: Sequence[object],
) -> tuple[bool, list[dict[str, object]]]:
    bad_rows: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        missing = [str(field) for field in required_fields if str(field) not in row]
        nonfinite = [
            str(field)
            for field in required_fields
            if str(field) in row
            and str(field)
            in {
                "wall_clock_s",
                "protein_identity",
                "cai",
                "gc",
                "gc3",
                "codon_usage_kl_vs_native",
                "codon_pair_kl_vs_native",
                "te_proxy",
                "te_proxy_delta_vs_native",
                "uaug_count",
                "kozak_score",
                "start_accessibility_proxy",
            }
            and not _finite(row.get(str(field)))
        ]
        if missing or nonfinite:
            bad_rows.append({"row_index": idx, "missing_fields": missing, "nonfinite_fields": nonfinite})
    return not bad_rows, bad_rows[:20]


def _audit_cds_constraints(
    output_rows: Sequence[Mapping[str, object]],
    input_by_id: Mapping[str, Mapping[str, object]],
) -> tuple[bool, list[dict[str, object]]]:
    failures: list[dict[str, object]] = []
    for idx, row in enumerate(output_rows):
        transcript_id = row.get("transcript_id")
        source = input_by_id.get(str(transcript_id))
        designed_cds = row.get("designed_cds")
        reasons: list[str] = []
        if row.get("valid_cds") is not True:
            reasons.append("valid_cds_not_true")
        if row.get("protein_identity_exact_1") is not True:
            reasons.append("protein_identity_exact_1_not_true")
        if not _finite(row.get("protein_identity")) or float(row.get("protein_identity", 0.0)) != 1.0:
            reasons.append("protein_identity_not_exact_1")
        if not isinstance(designed_cds, str) or not is_valid_cds(designed_cds):
            reasons.append("designed_cds_invalid")
        elif source:
            target_with_stop = str(source.get("protein_target_with_stop", ""))
            target_no_stop = str(source.get("protein_target", ""))
            designed_protein = translate(designed_cds)
            if designed_protein not in {target_with_stop, target_no_stop}:
                reasons.append("translated_protein_mismatch")
        else:
            reasons.append("transcript_id_not_in_input_pack")
        if reasons:
            failures.append({"row_index": idx, "transcript_id": transcript_id, "reasons": reasons})
    return not failures, failures[:20]


def _audit_utr5_constraints(
    output_rows: Sequence[Mapping[str, object]],
    input_by_id: Mapping[str, Mapping[str, object]],
) -> tuple[bool, list[dict[str, object]]]:
    failures: list[dict[str, object]] = []
    for idx, row in enumerate(output_rows):
        transcript_id = row.get("transcript_id")
        reasons: list[str] = []
        if str(transcript_id) not in input_by_id:
            reasons.append("transcript_id_not_in_input_pack")
        if not isinstance(row.get("designed_five_utr"), str):
            reasons.append("designed_five_utr_missing")
        for key in ("cds_unchanged", "three_utr_unchanged", "protein_identity_exact_1"):
            if row.get(key) is not True:
                reasons.append(f"{key}_not_true")
        if reasons:
            failures.append({"row_index": idx, "transcript_id": transcript_id, "reasons": reasons})
    return not failures, failures[:20]


def _audit_one_model(
    *,
    model_name: str,
    project_root: str,
    real_run_dir: str,
    input_pack: Mapping[str, object],
    input_pack_summary_path: str,
    input_rows: Sequence[Mapping[str, object]],
    schema: Mapping[str, object],
) -> dict[str, object]:
    task_family = _model_task_family(model_name)
    if task_family is None:
        return {
            "model_name": model_name,
            "status": "unknown_model_family",
            "task_family": None,
            "real_metric_ready": False,
            "failure_reasons": ["unknown_model_family"],
        }
    output_filename = "cds_outputs.jsonl" if task_family == "cds_protein_conditioned" else "utr5_outputs.jsonl"
    model_dir, evidence_variant = _resolve_model_dir(
        real_run_dir,
        model_name,
        output_filename,
    )
    output_path = os.path.join(model_dir, output_filename)
    summary_path = os.path.join(model_dir, "summary.json")
    summary = _load_json(summary_path)
    output_rows = _load_jsonl(output_path)
    expected_input_count = len(input_rows)
    input_by_id = {str(row.get("transcript_id")): row for row in input_rows}
    failure_reasons: list[str] = []
    if summary is None:
        failure_reasons.append("summary_missing")
    if not os.path.exists(output_path):
        failure_reasons.append("outputs_jsonl_missing")
    if summary is None or not os.path.exists(output_path):
        return {
            "model_name": model_name,
            "status": "missing",
            "task_family": task_family,
            "summary_path": _rel(summary_path, project_root),
            "outputs_jsonl": _rel(output_path, project_root),
            "selected_evidence_variant": evidence_variant,
            "expected_input_rows": expected_input_count,
            "n_outputs": 0,
            "n_failures": None,
            "real_metric_ready": False,
            "real_runtime_ready": False,
            "hard_constraints_exact_1": False,
            "failure_reasons": failure_reasons,
        }

    task_schema = _mapping(schema.get(task_family))
    required_output_fields = task_schema.get("required_output_jsonl_fields", [])
    required_summary_fields = task_schema.get("required_summary_fields", [])
    metadata_ok, metadata_failures = _required_metadata_checks(
        summary,
        model_name=model_name,
        task_family=task_family,
        input_pack=input_pack,
        input_pack_summary_path=input_pack_summary_path,
    )
    summary_fields_ok, bad_summary_fields = _required_summary_fields_ok(summary, required_summary_fields)
    rows_fields_ok, bad_output_rows = _required_row_fields_ok(output_rows, required_output_fields)

    n_inputs = summary.get("n_inputs")
    n_outputs = summary.get("n_outputs")
    n_failures = summary.get("n_failures")
    eligibility = _mapping(summary.get("eligibility"))
    protocol_subset = bool(
        model_name == "UTailoR"
        and eligibility.get("policy")
        == "official_input_length_25_100_strict"
    )
    if protocol_subset:
        expected_output_ids = {
            str(row.get("transcript_id"))
            for row in input_rows
            if 25
            <= len(
                "".join(
                    str(row.get("native_five_utr") or "")
                    .upper()
                    .replace("T", "U")
                    .split()
                )
            )
            <= 100
        }
        n_eligible = eligibility.get("n_eligible_inputs")
        n_ineligible = eligibility.get("n_ineligible_inputs")
        eligibility_contract_ok = bool(
            n_eligible == len(expected_output_ids)
            and isinstance(n_ineligible, int)
            and n_eligible + n_ineligible == expected_input_count
        )
        input_coverage_ok = bool(
            eligibility_contract_ok
            and n_inputs == expected_input_count
            and n_outputs == len(output_rows)
            and isinstance(n_failures, int)
            and isinstance(n_outputs, int)
            and n_outputs + n_failures == n_eligible
        )
    else:
        expected_output_ids = set(input_by_id)
        eligibility_contract_ok = True
        input_coverage_ok = (
            n_inputs == expected_input_count
            and n_outputs == len(output_rows)
            and isinstance(n_failures, int)
            and isinstance(n_outputs, int)
            and n_outputs + n_failures == expected_input_count
        )
    if not input_coverage_ok:
        failure_reasons.append("input_output_coverage_mismatch")
    transcript_ids = [str(row.get("transcript_id")) for row in output_rows]
    if len(transcript_ids) != len(set(transcript_ids)):
        failure_reasons.append("duplicate_transcript_id")
    if any(transcript_id not in input_by_id for transcript_id in transcript_ids):
        failure_reasons.append("output_transcript_not_in_input_pack")
    if set(transcript_ids) != expected_output_ids:
        failure_reasons.append("expected_output_transcript_coverage_mismatch")
    if any(row.get("model_name") != model_name for row in output_rows):
        failure_reasons.append("row_model_name_mismatch")

    if task_family == "cds_protein_conditioned":
        constraints_ok, constraint_failures = _audit_cds_constraints(output_rows, input_by_id)
    else:
        constraints_ok, constraint_failures = _audit_utr5_constraints(output_rows, input_by_id)

    failure_reasons.extend(metadata_failures)
    if bad_summary_fields:
        failure_reasons.append("required_summary_fields_missing_or_nonfinite")
    if bad_output_rows:
        failure_reasons.append("required_output_fields_missing_or_nonfinite")
    if not constraints_ok:
        failure_reasons.append("hard_constraints_failed")

    real_metric_ready = (
        metadata_ok
        and summary_fields_ok
        and rows_fields_ok
        and input_coverage_ok
        and constraints_ok
        and eligibility_contract_ok
        and not any(row.get("model_name") != model_name for row in output_rows)
        and len(transcript_ids) == len(set(transcript_ids))
        and all(transcript_id in input_by_id for transcript_id in transcript_ids)
        and set(transcript_ids) == expected_output_ids
    )
    success_denominator = len(expected_output_ids)
    success_fraction = (
        float(len(output_rows) / success_denominator)
        if success_denominator
        else 0.0
    )
    protocol_fidelity = summary.get("protocol_fidelity", "unspecified")
    protocol_fidelity_sufficient = (
        summary.get("protocol_fidelity_sufficient_for_sota_reproduction") is True
    )
    return {
        "model_name": model_name,
        "status": "measured" if real_metric_ready else "invalid",
        "task_family": task_family,
        "summary_path": _rel(summary_path, project_root),
        "outputs_jsonl": _rel(output_path, project_root),
        "selected_evidence_variant": evidence_variant,
        "summary_sha256": _sha256_file(summary_path),
        "outputs_sha256": _sha256_file(output_path),
        "expected_input_rows": expected_input_count,
        "expected_eligible_input_rows": success_denominator,
        "protocol_subset": protocol_subset,
        "eligibility_policy": eligibility.get("policy"),
        "eligibility_contract_ok": eligibility_contract_ok,
        "n_inputs": n_inputs,
        "n_outputs": len(output_rows),
        "n_failures": n_failures,
        "success_fraction": success_fraction,
        "metadata_ok": metadata_ok,
        "summary_fields_ok": summary_fields_ok,
        "output_fields_ok": rows_fields_ok,
        "input_coverage_ok": input_coverage_ok,
        "hard_constraints_exact_1": constraints_ok,
        "real_metric_ready": real_metric_ready,
        "real_runtime_ready": metadata_ok and _finite(_nested(summary, "runtime.elapsed_s")),
        "protocol_fidelity": protocol_fidelity,
        "protocol_fidelity_sufficient_for_sota_reproduction": protocol_fidelity_sufficient,
        "failure_reasons": sorted(set(failure_reasons)),
        "metadata_failures": metadata_failures,
        "bad_summary_fields": bad_summary_fields,
        "bad_output_rows": bad_output_rows,
        "constraint_failures": constraint_failures,
    }


def audit_external_sota_real_runs(
    *,
    project_root: str,
    input_pack_rel: str = DEFAULT_INPUT_PACK,
    real_run_dir_rel: str = DEFAULT_REAL_RUN_DIR,
    models: Optional[Sequence[str]] = None,
    out_json: Optional[str] = None,
    out_md: Optional[str] = None,
) -> dict[str, object]:
    """Audit real external adapter outputs without running external tools."""
    input_pack_path = _path(project_root, input_pack_rel)
    real_run_dir = _path(project_root, real_run_dir_rel)
    input_pack = _load_json(input_pack_path)
    input_pack_ready = (
        isinstance(input_pack, Mapping)
        and input_pack.get("artifact_kind") == "external_sota_input_pack"
        and input_pack.get("ready_for_external_real_run") is True
    )
    if input_pack:
        schema = _mapping(input_pack.get("metric_schema")) or external_metric_schema()
        cds_path = _resolve_pack_path(input_pack_path, input_pack, "cds_protein_jsonl", "cds_protein_inputs.jsonl")
        utr_path = _resolve_pack_path(input_pack_path, input_pack, "utr5_jsonl", "utr5_inputs.jsonl")
        cds_inputs = _load_jsonl(cds_path)
        utr_inputs = _load_jsonl(utr_path)
        expected_models = _expected_models(input_pack, models)
    else:
        schema = external_metric_schema()
        cds_path = os.path.join(os.path.dirname(input_pack_path), "cds_protein_inputs.jsonl")
        utr_path = os.path.join(os.path.dirname(input_pack_path), "utr5_inputs.jsonl")
        cds_inputs = []
        utr_inputs = []
        expected_models = _expected_models(None, models)

    rows = []
    for model_name in expected_models:
        task_family = _model_task_family(model_name)
        input_rows = cds_inputs if task_family == "cds_protein_conditioned" else utr_inputs
        rows.append(
            _audit_one_model(
                model_name=model_name,
                project_root=project_root,
                real_run_dir=real_run_dir,
                input_pack=input_pack or {},
                input_pack_summary_path=input_pack_path,
                input_rows=input_rows,
                schema=schema,
            )
        )
    n_measured = sum(1 for row in rows if row.get("status") == "measured")
    n_invalid = sum(1 for row in rows if row.get("status") == "invalid")
    n_missing = sum(1 for row in rows if row.get("status") == "missing")
    all_measured = bool(rows) and n_measured == len(rows)
    n_protocol_sufficient = sum(
        1
        for row in rows
        if row.get("protocol_fidelity_sufficient_for_sota_reproduction") is True
    )
    all_protocol_sufficient = bool(rows) and n_protocol_sufficient == len(rows)
    payload = {
        "artifact_kind": "external_sota_real_run_audit",
        "project_root": os.path.abspath(project_root),
        "claim_policy": CLAIM_POLICY,
        "input_pack": {
            "path": input_pack_rel,
            "exists": input_pack is not None,
            "sha256": _sha256_file(input_pack_path),
            "ready_for_external_real_run": bool(input_pack_ready),
            "cds_protein_jsonl": _rel(cds_path, project_root),
            "cds_protein_jsonl_sha256": _sha256_file(cds_path),
            "utr5_jsonl": _rel(utr_path, project_root),
            "utr5_jsonl_sha256": _sha256_file(utr_path),
        },
        "real_run_dir": real_run_dir_rel,
        "summary": {
            "audit_complete": bool(input_pack_ready and rows),
            "n_models_expected": len(rows),
            "n_models_measured": n_measured,
            "n_models_invalid": n_invalid,
            "n_models_missing": n_missing,
            "n_models_protocol_fidelity_sufficient": n_protocol_sufficient,
            "ready_for_external_real_metric_table": all_measured,
            "ready_for_external_sota_metric_claim": (
                all_measured and all_protocol_sufficient
            ),
            "ready_for_external_sota_claim": False,
            "claim_boundary": (
                "A complete real metric table only permits reporting external "
                "metric rows. It does not establish MEF superiority without a "
                "separate head-to-head comparison and claim-language audit."
            ),
        },
        "rows": rows,
    }
    if out_json:
        write_report_json(payload, out_json)
    if out_md:
        write_report_markdown(payload, out_md)
    return payload


def write_report_json(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return path


def write_report_markdown(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    summary = _mapping(report.get("summary"))
    input_pack = _mapping(report.get("input_pack"))
    rows = report.get("rows", [])
    rows = rows if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) else []
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# External SOTA Real-Run Audit\n\n")
        fh.write(f"- Claim policy: {report.get('claim_policy', CLAIM_POLICY)}\n")
        fh.write(
            f"- Input pack: `{input_pack.get('path')}`; ready: "
            f"`{input_pack.get('ready_for_external_real_run')}`; "
            f"sha256: `{input_pack.get('sha256')}`\n"
        )
        fh.write(
            f"- Measured models: `{summary.get('n_models_measured')}` / "
            f"`{summary.get('n_models_expected')}`; invalid: "
            f"`{summary.get('n_models_invalid')}`; missing: "
            f"`{summary.get('n_models_missing')}`; protocol-fidelity sufficient: "
            f"`{summary.get('n_models_protocol_fidelity_sufficient')}`\n"
        )
        fh.write(
            f"- Ready for external real metric table: "
            f"`{summary.get('ready_for_external_real_metric_table')}`; "
            f"ready for external SOTA claim: "
            f"`{summary.get('ready_for_external_sota_claim')}`\n\n"
        )
        fh.write("| Model | Task family | Status | Outputs | Success | Constraints | Protocol fidelity | Failure reasons |\n")
        fh.write("|---|---|---|---:|---:|---|---|---|\n")
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            reasons = row.get("failure_reasons", [])
            reason_text = ", ".join(str(item) for item in reasons) if isinstance(reasons, Sequence) else str(reasons)
            fh.write(
                f"| {row.get('model_name')} | {row.get('task_family')} | "
                f"`{row.get('status')}` | {row.get('n_outputs')} / "
                f"{row.get('expected_input_rows')} | "
                f"{row.get('success_fraction', 0.0):.4f} | "
                f"`{row.get('hard_constraints_exact_1')}` | "
                f"{row.get('protocol_fidelity')} "
                f"(sufficient={row.get('protocol_fidelity_sufficient_for_sota_reproduction')}) | "
                f"{reason_text} |\n"
            )
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--input-pack", default=DEFAULT_INPUT_PACK)
    parser.add_argument("--real-run-dir", default=DEFAULT_REAL_RUN_DIR)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", default=DEFAULT_OUT_MD)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    project_root = os.path.abspath(args.project_root)
    out_json = args.out_json if os.path.isabs(args.out_json) else os.path.join(project_root, args.out_json)
    out_md = args.out_md if os.path.isabs(args.out_md) else os.path.join(project_root, args.out_md)
    payload = audit_external_sota_real_runs(
        project_root=project_root,
        input_pack_rel=args.input_pack,
        real_run_dir_rel=args.real_run_dir,
        models=args.models,
        out_json=out_json,
        out_md=out_md,
    )
    print(json.dumps({"json_path": out_json, "markdown_path": out_md, "summary": payload["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLAIM_POLICY",
    "DEFAULT_INPUT_PACK",
    "DEFAULT_REAL_RUN_DIR",
    "audit_external_sota_real_runs",
    "write_report_json",
    "write_report_markdown",
    "main",
]
