"""Audit dataset manifest completeness for data scale-up evidence.

The public corpus builders already write dataset manifests, but paper-grade
data-scale claims require a stricter contract: source URL, SHA256, record
counts, drop/attrition statistics, split statistics, and local record SHA
verification when the record file is available. This module is read-only and
does not build or modify corpora.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from typing import Mapping, Optional, Sequence


CLAIM_POLICY = (
    "Dataset manifest audit is reproducibility-governance evidence. Do not "
    "claim data scale-up or leakage-safe training until every required data "
    "version has source URL, SHA256, raw/clean record counts, drop statistics, "
    "split statistics, and record-file SHA verification where applicable."
)

DATASET_SPECS: tuple[dict[str, object], ...] = (
    {
        "dataset": "gencode_human_transcripts",
        "required_for": "base_training_corpus",
        "manifest_globs": ("data/processed/gencode_human_transcripts.data_manifest.json",),
        "split_report_globs": ("benchmark/gencode_family_leakage_protocol/report.json",),
        "split_status_globs": ("benchmark/gencode_family_leakage_protocol/status.json",),
    },
    {
        "dataset": "refseq_human_rna",
        "required_for": "refseq_scaleup_corpus",
        "manifest_globs": ("data/processed/refseq_human_rna.data_manifest.json",),
        "split_report_globs": ("benchmark/refseq_family_leakage_protocol/report.json",),
        "split_status_globs": ("benchmark/refseq_family_leakage_protocol/status.json",),
    },
    {
        "dataset": "mpra_te",
        "required_for": "real_te_predictor",
        "manifest_globs": (
            "data/processed/*mpra*.data_manifest.json",
            "data/processed/*mpra*manifest*.json",
        ),
    },
    {
        "dataset": "stability_half_life",
        "required_for": "real_stability_predictor",
        "manifest_globs": (
            "data/processed/*stability*.data_manifest.json",
            "data/processed/*half_life*.data_manifest.json",
            "data/processed/*degradation*.data_manifest.json",
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


def _load_json(path: str) -> Mapping[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _sha256(path: str) -> Optional[str]:
    if not os.path.exists(path) or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _local_candidate(project_root: str, declared_path: object) -> Optional[str]:
    if not isinstance(declared_path, str) or not declared_path:
        return None
    if os.path.isabs(declared_path):
        if os.path.exists(declared_path):
            return declared_path
        marker = "/mrna_editflow/"
        if marker in declared_path:
            return os.path.join(project_root, declared_path.split(marker, 1)[1])
        return declared_path
    return os.path.join(project_root, declared_path)


def _first_existing_manifest(project_root: str, patterns: Sequence[str]) -> Optional[str]:
    candidates: list[str] = []
    for pattern in patterns:
        candidates.extend(glob.glob(_path(project_root, pattern)))
    candidates = sorted(set(candidates), key=os.path.getmtime, reverse=True)
    return candidates[0] if candidates else None


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _files(manifest: Mapping[str, object]) -> list[Mapping[str, object]]:
    dataset = _as_mapping(manifest.get("dataset"))
    files = dataset.get("files", [])
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes)):
        return []
    return [row for row in files if isinstance(row, Mapping)]


def _has_source_url(manifest: Mapping[str, object]) -> bool:
    dataset = _as_mapping(manifest.get("dataset"))
    if isinstance(dataset.get("registry_url"), str) and dataset.get("registry_url"):
        return True
    return any(
        isinstance(row.get("public_url"), str) and bool(row.get("public_url"))
        for row in _files(manifest)
    )


def _has_raw_sha(manifest: Mapping[str, object]) -> bool:
    return any(
        isinstance(row.get("sha256"), str) and len(str(row.get("sha256"))) == 64
        for row in _files(manifest)
    )


def _has_record_counts(manifest: Mapping[str, object]) -> bool:
    raw = _as_mapping(manifest.get("raw_summary"))
    clean = _as_mapping(manifest.get("clean_summary"))
    return isinstance(raw.get("n_records"), int) and isinstance(clean.get("n_records"), int)


def _has_drop_stats(manifest: Mapping[str, object]) -> bool:
    drops = manifest.get("cleaning_drop_counts")
    return isinstance(drops, Mapping) and bool(drops) and "total" in drops and "kept" in drops


def _has_split_stats(manifest: Mapping[str, object]) -> bool:
    for key in ("split_stats", "splits", "split_counts"):
        value = manifest.get(key)
        if isinstance(value, Mapping) and value:
            return True
    return False


def _split_sidecar_audit(
    project_root: str,
    report_patterns: Sequence[str],
    status_patterns: Sequence[str] = (),
) -> dict[str, object]:
    path = _first_existing_manifest(project_root, report_patterns)
    status_path = _first_existing_manifest(project_root, status_patterns)
    if path is None:
        status_payload = _load_json(status_path) if status_path is not None else {}
        progress = _as_mapping(status_payload.get("progress"))
        return {
            "exists": False,
            "path": None,
            "split_stats_ready": False,
            "split_stats_pending": bool(status_path),
            "status_path": _rel(project_root, status_path),
            "status_sha256": _sha256(status_path) if status_path else None,
            "status": status_payload.get("status") if isinstance(status_payload, Mapping) else None,
            "last_event": progress.get("last_event"),
            "last_loadavg": progress.get("last_loadavg"),
        }
    payload = _load_json(path)
    summary = _as_mapping(payload.get("summary"))
    split = _as_mapping(payload.get("split"))
    split_ready = bool(
        summary.get("split_ready")
        and isinstance(split.get("n_train"), int)
        and isinstance(split.get("n_val"), int)
        and isinstance(split.get("n_test"), int)
    )
    return {
        "exists": True,
        "path": _rel(project_root, path),
        "sha256": _sha256(path),
        "split_stats_ready": split_ready,
        "split_stats_pending": False,
        "status_path": _rel(project_root, status_path),
        "status_sha256": _sha256(status_path) if status_path else None,
        "synthetic_smoke_only": bool(summary.get("synthetic_smoke_only")),
        "external_records_provided": bool(summary.get("external_records_provided")),
        "n_train": split.get("n_train"),
        "n_val": split.get("n_val"),
        "n_test": split.get("n_test"),
        "n_clusters": split.get("n_clusters"),
    }


def _records_audit(project_root: str, manifest: Mapping[str, object]) -> dict[str, object]:
    declared = manifest.get("records_path")
    local_path = _local_candidate(project_root, declared)
    exists = bool(local_path and os.path.exists(local_path))
    declared_sha = manifest.get("records_sha256")
    actual_sha = _sha256(local_path) if exists and local_path else None
    return {
        "declared_path": declared if isinstance(declared, str) else None,
        "local_path": _rel(project_root, local_path),
        "exists": exists,
        "declared_sha256": declared_sha if isinstance(declared_sha, str) else None,
        "actual_sha256": actual_sha,
        "sha256_matches": bool(actual_sha and actual_sha == declared_sha),
    }


def audit_one_manifest(
    project_root: str,
    dataset: str,
    required_for: str,
    manifest_globs: Sequence[str],
    split_report_globs: Sequence[str] = (),
    split_status_globs: Sequence[str] = (),
) -> dict[str, object]:
    split_sidecar = _split_sidecar_audit(
        project_root,
        split_report_globs,
        split_status_globs,
    )
    manifest_path = _first_existing_manifest(project_root, manifest_globs)
    if manifest_path is None:
        return {
            "dataset": dataset,
            "required_for": required_for,
            "manifest_path": None,
            "manifest_exists": False,
            "complete": False,
            "missing_fields": [
                "manifest",
                "source_url",
                "raw_file_sha256",
                "records_sha256",
                "record_counts",
                "drop_stats",
                "split_stats",
            ],
            "split_sidecar": split_sidecar,
        }

    manifest = _load_json(manifest_path)
    dataset_payload = _as_mapping(manifest.get("dataset"))
    records = _records_audit(project_root, manifest)
    split_stats_ready = _has_split_stats(manifest) or bool(
        split_sidecar.get("split_stats_ready")
        and split_sidecar.get("external_records_provided")
        and not split_sidecar.get("synthetic_smoke_only")
    )
    checks = {
        "dataset_name_present": isinstance(dataset_payload.get("name"), str),
        "source_url": _has_source_url(manifest),
        "raw_file_sha256": _has_raw_sha(manifest),
        "records_path": isinstance(manifest.get("records_path"), str),
        "records_sha256": isinstance(manifest.get("records_sha256"), str)
        and len(str(manifest.get("records_sha256"))) == 64,
        "record_counts": _has_record_counts(manifest),
        "drop_stats": _has_drop_stats(manifest),
        "split_stats": split_stats_ready,
    }
    missing = [name for name, ok in checks.items() if not ok]
    complete = not missing and bool(records.get("sha256_matches") or not records.get("exists"))
    return {
        "dataset": dataset,
        "required_for": required_for,
        "manifest_path": _rel(project_root, manifest_path),
        "manifest_exists": True,
        "manifest_sha256": _sha256(manifest_path),
        "manifest_dataset_name": dataset_payload.get("name"),
        "checks": checks,
        "records": records,
        "split_sidecar": split_sidecar,
        "complete": complete,
        "missing_fields": missing,
        "raw_n_records": _as_mapping(manifest.get("raw_summary")).get("n_records"),
        "clean_n_records": _as_mapping(manifest.get("clean_summary")).get("n_records"),
    }


def build_dataset_manifest_audit(project_root: str) -> dict[str, object]:
    project_root = os.path.abspath(project_root)
    rows = [
        audit_one_manifest(
            project_root,
            dataset=str(spec["dataset"]),
            required_for=str(spec["required_for"]),
            manifest_globs=tuple(str(x) for x in spec["manifest_globs"]),
            split_report_globs=tuple(str(x) for x in spec.get("split_report_globs", ())),
            split_status_globs=tuple(str(x) for x in spec.get("split_status_globs", ())),
        )
        for spec in DATASET_SPECS
    ]
    incomplete = [row["dataset"] for row in rows if not row.get("complete")]
    pending_split_sidecars = [
        row["dataset"]
        for row in rows
        if _as_mapping(row.get("split_sidecar")).get("split_stats_pending")
    ]
    return {
        "artifact_kind": "dataset_manifest_audit",
        "project_root": project_root,
        "claim_policy": CLAIM_POLICY,
        "summary": {
            "n_datasets": len(rows),
            "n_manifests_present": sum(1 for row in rows if row.get("manifest_exists")),
            "n_complete": sum(1 for row in rows if row.get("complete")),
            "all_required_dataset_manifests_complete": len(incomplete) == 0,
            "ready_for_data_scale_claim": False,
            "incomplete_datasets": incomplete,
            "pending_split_sidecars": pending_split_sidecars,
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
        fh.write("# Dataset Manifest Audit\n\n")
        fh.write(f"- Claim policy: {report.get('claim_policy', CLAIM_POLICY)}\n")
        fh.write(
            f"- Complete manifests: `{summary.get('n_complete')}/"
            f"{summary.get('n_datasets')}`; all complete: "
            f"`{summary.get('all_required_dataset_manifests_complete')}`; "
            f"ready for data-scale claim: `{summary.get('ready_for_data_scale_claim')}`\n"
        )
        fh.write(f"- Incomplete datasets: `{summary.get('incomplete_datasets')}`\n")
        fh.write(f"- Pending split sidecars: `{summary.get('pending_split_sidecars')}`\n\n")
        fh.write("| Dataset | Manifest | Complete | Missing fields | Records SHA verified | Split sidecar |\n")
        fh.write("|---|---|---:|---|---:|---|\n")
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            records = _as_mapping(row.get("records"))
            sidecar = _as_mapping(row.get("split_sidecar"))
            sidecar_display = sidecar.get("path") or sidecar.get("status_path")
            if sidecar.get("split_stats_pending"):
                sidecar_display = f"pending:{sidecar_display}"
            fh.write(
                f"| {row.get('dataset')} | `{row.get('manifest_path')}` | "
                f"`{row.get('complete')}` | `{row.get('missing_fields')}` | "
                f"`{records.get('sha256_matches')}` | `{sidecar_display}` |\n"
            )
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--out-json", default="docs/dataset_manifest_audit.json")
    parser.add_argument("--out-md", default="docs/dataset_manifest_audit.md")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    project_root = os.path.abspath(args.project_root)
    report = build_dataset_manifest_audit(project_root)
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
    "audit_one_manifest",
    "build_dataset_manifest_audit",
    "write_report_json",
    "write_report_markdown",
    "main",
]
