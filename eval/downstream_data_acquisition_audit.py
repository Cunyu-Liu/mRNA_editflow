"""Audit acquisition gates for real MPRA/TE and stability data.

The project already has predictor protocols and a downstream manifest builder.
This report keeps the remaining data-acquisition work explicit: source table,
official split, manifest completeness, predictor audit, and claim boundary. It
is read-only and never downloads data or promotes protocol evidence to real
TE/stability results.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from typing import Mapping, Optional, Sequence


CLAIM_POLICY = (
    "Downstream data acquisition audit is planning and provenance evidence. "
    "Do not claim real MPRA/TE or stability performance until source tables are "
    "downloaded with license/source URLs, official splits are preserved, dataset "
    "manifests are complete, held-out predictor reports are generated, and "
    "leakage documentation is attached."
)

DATASET_SPECS: tuple[dict[str, object], ...] = (
    {
        "dataset": "mpra_te",
        "task": "real_te_predictor",
        "source_reference": "Sample et al. 2019 MPRA / Optimus-style 5'UTR MRL table",
        "source_status": "needs_download_and_license_verification",
        "required_columns": ("sequence", "mrl", "split"),
        "env_vars": ("MRNA_EDITFLOW_MPRA_TE_TABLE", "MPRA_TE_TABLE", "MPRA_TABLE"),
        "raw_globs": (
            "data/raw/*mpra*.csv",
            "data/raw/*mpra*.tsv",
            "data/raw/*mrl*.csv",
            "data/raw/*mrl*.tsv",
            "data/processed/*mpra*.csv",
            "data/processed/*mpra*.tsv",
            "data/processed/*mrl*.csv",
            "data/processed/*mrl*.tsv",
        ),
        "manifest_globs": (
            "data/processed/*mpra*.data_manifest.json",
            "data/processed/*mpra*manifest*.json",
        ),
        "predictor_report_globs": (
            "benchmark/mpra_te_predictor*/report.json",
            "benchmark/mpra_te_predictor*.json",
        ),
        "predictor_ready_key": "ready_for_mpra_te_predictor_audit",
        "builder_command": (
            "python -m mrna_editflow.eval.build_downstream_table_manifest "
            "--dataset mpra_te --input <csv_or_tsv> --source-url <verified_url>"
        ),
    },
    {
        "dataset": "stability_half_life",
        "task": "real_stability_predictor",
        "source_reference": "External mRNA stability/half-life table, e.g. mRNABench-compatible labelled task",
        "source_status": "needs_dataset_selection_download_and_license_verification",
        "required_columns": ("sequence", "half_life|stability|stability_score|degradation_rate", "split"),
        "env_vars": (
            "MRNA_EDITFLOW_STABILITY_TABLE",
            "STABILITY_TABLE",
            "HALF_LIFE_TABLE",
        ),
        "raw_globs": (
            "data/raw/*stability*.csv",
            "data/raw/*stability*.tsv",
            "data/raw/*half_life*.csv",
            "data/raw/*half_life*.tsv",
            "data/raw/*degradation*.csv",
            "data/raw/*degradation*.tsv",
            "data/processed/*stability*.csv",
            "data/processed/*stability*.tsv",
            "data/processed/*half_life*.csv",
            "data/processed/*half_life*.tsv",
            "data/processed/*degradation*.csv",
            "data/processed/*degradation*.tsv",
        ),
        "manifest_globs": (
            "data/processed/*stability*.data_manifest.json",
            "data/processed/*half_life*.data_manifest.json",
            "data/processed/*degradation*.data_manifest.json",
        ),
        "predictor_report_globs": (
            "benchmark/stability_predictor*/report.json",
            "benchmark/stability_predictor*.json",
        ),
        "predictor_ready_key": "ready_for_stability_predictor_audit",
        "builder_command": (
            "python -m mrna_editflow.eval.build_downstream_table_manifest "
            "--dataset stability_half_life --input <csv_or_tsv> --source-url <verified_url>"
        ),
    },
)


def _path(project_root: str, rel_or_abs: str) -> str:
    return rel_or_abs if os.path.isabs(rel_or_abs) else os.path.join(project_root, rel_or_abs)


def _rel(project_root: str, path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        return os.path.relpath(path, project_root)
    except ValueError:
        return path


def _sha256(path: str) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json_if_exists(path: Optional[str]) -> Optional[Mapping[str, object]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _glob_existing(project_root: str, patterns: Sequence[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(_path(project_root, pattern)))
    return sorted(set(paths))


def _env_candidates(project_root: str, env_vars: Sequence[str]) -> list[dict[str, object]]:
    rows = []
    for name in env_vars:
        value = os.environ.get(name)
        if not value:
            continue
        path = _path(project_root, value) if not os.path.isabs(value) else value
        rows.append(
            {
                "env_var": name,
                "path": _rel(project_root, path),
                "exists": os.path.exists(path),
                "sha256": _sha256(path),
            }
        )
    return rows


def _latest_json(project_root: str, patterns: Sequence[str]) -> tuple[Optional[str], Optional[Mapping[str, object]]]:
    paths = _glob_existing(project_root, patterns)
    if not paths:
        return None, None
    latest = max(paths, key=os.path.getmtime)
    return latest, _load_json_if_exists(latest)


def _dataset_manifest_rows(project_root: str) -> dict[str, Mapping[str, object]]:
    payload = _load_json_if_exists(_path(project_root, "docs/dataset_manifest_audit.json"))
    if not payload:
        return {}
    rows = payload.get("rows", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return {}
    return {
        str(row.get("dataset")): row
        for row in rows
        if isinstance(row, Mapping) and row.get("dataset")
    }


def _input_table_status(project_root: str, dataset: str) -> Mapping[str, object]:
    payload = _load_json_if_exists(_path(project_root, "docs/data_scaleup_readiness.json"))
    if not payload:
        return {}
    audit = payload.get("mpra_te_stability_audit")
    if not isinstance(audit, Mapping):
        return {}
    key = "mpra_input_table" if dataset == "mpra_te" else "stability_input_table"
    status = audit.get(key)
    return status if isinstance(status, Mapping) else {}


def _predictor_report_summary(
    project_root: str,
    patterns: Sequence[str],
    ready_key: str,
) -> dict[str, object]:
    path, payload = _latest_json(project_root, patterns)
    if payload is None:
        return {
            "path": None,
            "exists": False,
            "sha256": None,
            "ready_for_predictor_audit": False,
            "synthetic_smoke_only": None,
        }
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), Mapping) else {}
    return {
        "path": _rel(project_root, path),
        "exists": True,
        "sha256": _sha256(path) if path else None,
        "ready_for_predictor_audit": bool(summary.get(ready_key)),
        "synthetic_smoke_only": bool(summary.get("synthetic_smoke_only")),
        "external_input_provided": bool(summary.get("external_input_provided")),
    }


def _manifest_summary(
    project_root: str,
    dataset: str,
    manifest_globs: Sequence[str],
) -> dict[str, object]:
    rows = _dataset_manifest_rows(project_root)
    audit_row = rows.get(dataset, {})
    paths = _glob_existing(project_root, manifest_globs)
    latest = max(paths, key=os.path.getmtime) if paths else None
    return {
        "path": _rel(project_root, latest),
        "exists": bool(latest),
        "sha256": _sha256(latest) if latest else None,
        "audit_complete": bool(audit_row.get("complete")),
        "missing_fields": audit_row.get("missing_fields", []),
        "records_sha256_verified": (
            audit_row.get("records", {}).get("sha256_matches")
            if isinstance(audit_row.get("records"), Mapping)
            else None
        ),
    }


def _raw_table_summary(
    project_root: str,
    raw_globs: Sequence[str],
    env_vars: Sequence[str],
) -> dict[str, object]:
    paths = _glob_existing(project_root, raw_globs)
    return {
        "candidate_paths": [
            {
                "path": _rel(project_root, path),
                "size_bytes": os.path.getsize(path),
                "sha256": _sha256(path),
            }
            for path in paths
            if os.path.isfile(path)
        ],
        "env_candidates": _env_candidates(project_root, env_vars),
    }


def audit_dataset(project_root: str, spec: Mapping[str, object]) -> dict[str, object]:
    dataset = str(spec["dataset"])
    env_vars = tuple(str(x) for x in spec["env_vars"])
    raw_globs = tuple(str(x) for x in spec["raw_globs"])
    manifest_globs = tuple(str(x) for x in spec["manifest_globs"])
    predictor_globs = tuple(str(x) for x in spec["predictor_report_globs"])
    ready_key = str(spec["predictor_ready_key"])

    raw = _raw_table_summary(project_root, raw_globs, env_vars)
    table_status = _input_table_status(project_root, dataset)
    manifest = _manifest_summary(project_root, dataset, manifest_globs)
    predictor = _predictor_report_summary(project_root, predictor_globs, ready_key)
    builder_exists = os.path.exists(_path(project_root, "eval/build_downstream_table_manifest.py"))

    source_table_ready = bool(table_status.get("ready_for_predictor_input"))
    manifest_complete = bool(manifest.get("audit_complete"))
    predictor_ready = bool(predictor.get("ready_for_predictor_audit"))
    missing: list[str] = []
    if not raw["candidate_paths"] and not raw["env_candidates"]:
        missing.append("source_table")
    if not source_table_ready:
        missing.append("schema_valid_official_split")
    if not builder_exists:
        missing.append("manifest_builder")
    if not manifest_complete:
        missing.append("complete_dataset_manifest")
    if not predictor_ready:
        missing.append("heldout_predictor_report")
    missing.append("leakage_documentation")

    if not raw["candidate_paths"] and not raw["env_candidates"]:
        status = "needs_source_table_download"
    elif not source_table_ready:
        status = "source_table_present_needs_schema_or_split_fix"
    elif not manifest_complete:
        status = "schema_ready_needs_manifest"
    elif not predictor_ready:
        status = "manifest_ready_needs_heldout_predictor"
    else:
        status = "predictor_audit_ready_needs_leakage_documentation"

    return {
        "dataset": dataset,
        "task": spec.get("task"),
        "status": status,
        "source_reference": spec.get("source_reference"),
        "source_status": spec.get("source_status"),
        "required_columns": list(spec.get("required_columns", ())),
        "env_vars": list(env_vars),
        "raw_table": raw,
        "input_table_status": dict(table_status),
        "manifest": manifest,
        "predictor_report": predictor,
        "manifest_builder": {
            "exists": builder_exists,
            "path": "eval/build_downstream_table_manifest.py",
            "command": spec.get("builder_command"),
        },
        "missing_gates": missing,
        "ready_for_acquisition_claim": False,
        "ready_for_real_te_or_stability_claim": False,
    }


def build_downstream_data_acquisition_audit(project_root: str) -> dict[str, object]:
    project_root = os.path.abspath(project_root)
    rows = [audit_dataset(project_root, spec) for spec in DATASET_SPECS]
    return {
        "artifact_kind": "downstream_data_acquisition_audit",
        "project_root": project_root,
        "claim_policy": CLAIM_POLICY,
        "summary": {
            "n_datasets": len(rows),
            "n_source_tables_present": sum(
                1 for row in rows
                if row["raw_table"]["candidate_paths"] or row["raw_table"]["env_candidates"]
            ),
            "n_schema_ready": sum(
                1 for row in rows
                if row.get("input_table_status", {}).get("ready_for_predictor_input")
            ),
            "n_manifests_complete": sum(1 for row in rows if row["manifest"]["audit_complete"]),
            "n_predictor_audits_ready": sum(
                1 for row in rows if row["predictor_report"]["ready_for_predictor_audit"]
            ),
            "ready_for_real_te_or_stability_claim": False,
            "incomplete_datasets": [
                row["dataset"]
                for row in rows
                if row["missing_gates"]
            ],
        },
        "rows": rows,
    }


def write_report_json(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return path


def write_report_markdown(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    summary = report.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    rows = report.get("rows", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        rows = []
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Downstream Data Acquisition Audit\n\n")
        fh.write(f"- Claim policy: {report.get('claim_policy', CLAIM_POLICY)}\n")
        fh.write(
            f"- Source tables present: `{summary.get('n_source_tables_present')}/"
            f"{summary.get('n_datasets')}`; schema ready: "
            f"`{summary.get('n_schema_ready')}/{summary.get('n_datasets')}`; "
            f"complete manifests: `{summary.get('n_manifests_complete')}/"
            f"{summary.get('n_datasets')}`; predictor audits ready: "
            f"`{summary.get('n_predictor_audits_ready')}/{summary.get('n_datasets')}`\n"
        )
        fh.write(
            f"- Ready for real TE/stability claim: "
            f"`{summary.get('ready_for_real_te_or_stability_claim')}`\n"
        )
        fh.write(f"- Incomplete datasets: `{summary.get('incomplete_datasets')}`\n\n")
        fh.write("| Dataset | Status | Raw candidates | Manifest complete | Predictor ready | Missing gates |\n")
        fh.write("|---|---|---:|---:|---:|---|\n")
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            raw = row.get("raw_table", {})
            raw_count = 0
            if isinstance(raw, Mapping):
                raw_count = len(raw.get("candidate_paths", [])) + len(raw.get("env_candidates", []))
            manifest = row.get("manifest", {}) if isinstance(row.get("manifest"), Mapping) else {}
            predictor = row.get("predictor_report", {}) if isinstance(row.get("predictor_report"), Mapping) else {}
            fh.write(
                f"| {row.get('dataset')} | {row.get('status')} | {raw_count} | "
                f"`{manifest.get('audit_complete')}` | "
                f"`{predictor.get('ready_for_predictor_audit')}` | "
                f"`{row.get('missing_gates')}` |\n"
            )
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--out-json", default="docs/downstream_data_acquisition_audit.json")
    parser.add_argument("--out-md", default="docs/downstream_data_acquisition_audit.md")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    project_root = os.path.abspath(args.project_root)
    report = build_downstream_data_acquisition_audit(project_root)
    out_json = args.out_json if os.path.isabs(args.out_json) else os.path.join(project_root, args.out_json)
    out_md = args.out_md if os.path.isabs(args.out_md) else os.path.join(project_root, args.out_md)
    write_report_json(report, out_json)
    write_report_markdown(report, out_md)
    print(json.dumps({"json_path": out_json, "markdown_path": out_md}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLAIM_POLICY",
    "DATASET_SPECS",
    "audit_dataset",
    "build_downstream_data_acquisition_audit",
    "write_report_json",
    "write_report_markdown",
    "main",
]
