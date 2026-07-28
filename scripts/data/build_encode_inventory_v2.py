#!/usr/bin/env python3
"""Build a fail-closed 62-row ENCSR854RUF inventory from reconstruction evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_FILES = 62
FIELDS = (
    "encode_accession",
    "file_name",
    "path",
    "expected_bytes",
    "actual_bytes",
    "status",
    "sha256",
    "checksum_origin",
    "assay_context",
    "sequence_role",
    "label_role",
    "license",
    "downstream_overlap_status",
    "admitted_as_intervention_data",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("files"), list):
        return payload["files"]
    raise ValueError("manifest must be a record list or contain a files list")


def _key(record: dict[str, Any]) -> str:
    value = record.get("encode_accession") or record.get("accession")
    if not value:
        value = record.get("name")
    if not value:
        raise ValueError("manifest record lacks accession and name")
    return str(value)


def build_inventory(
    reconstruction_manifest: Path,
    source_manifest: Path | None = None,
    rehash: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reconstruction_payload = _load_json(reconstruction_manifest)
    reconstructed = {_key(item): item for item in _records(reconstruction_payload)}

    if source_manifest is None and isinstance(reconstruction_payload, dict):
        candidate = reconstruction_payload.get("source_manifest")
        if candidate:
            candidate_path = Path(candidate)
            if candidate_path.is_file():
                source_manifest = candidate_path
    source_records = (
        _records(_load_json(source_manifest))
        if source_manifest is not None
        else list(reconstructed.values())
    )
    expected = {_key(item): item for item in source_records}
    if len(expected) != EXPECTED_FILES:
        raise ValueError(
            f"expected source manifest with {EXPECTED_FILES} unique files, got {len(expected)}"
        )

    rows: list[dict[str, Any]] = []
    for accession in sorted(expected):
        source = expected[accession]
        observed = reconstructed.get(accession)
        name = str(source.get("name") or (observed or {}).get("name") or accession)
        expected_bytes = int(source.get("bytes") or (observed or {}).get("bytes") or 0)
        path_text = str((observed or {}).get("path") or "")
        path = Path(path_text) if path_text else None
        actual_bytes = path.stat().st_size if path is not None and path.is_file() else 0
        sha256 = str((observed or {}).get("sha256") or "")
        status = str((observed or {}).get("status") or "MISSING_MANIFEST_ROW")
        valid = (
            observed is not None
            and status == "VERIFIED"
            and path is not None
            and path.is_file()
            and expected_bytes > 0
            and actual_bytes == expected_bytes
            and len(sha256) == 64
        )
        checksum_origin = "downloader_verified_manifest"
        if rehash and valid:
            actual_sha256 = _sha256(path)
            checksum_origin = "inventory_rehash"
            if actual_sha256 != sha256:
                valid = False
                status = "CHECKSUM_MISMATCH"
            sha256 = actual_sha256
        if not valid and status == "VERIFIED":
            status = "INCOMPLETE_OR_INVALID_EVIDENCE"
        rows.append(
            {
                "encode_accession": accession,
                "file_name": name,
                "path": path_text,
                "expected_bytes": expected_bytes,
                "actual_bytes": actual_bytes,
                "status": status,
                "sha256": sha256,
                "checksum_origin": checksum_origin,
                "assay_context": "MPRAu_raw_RNA_DNA_reporter_reads",
                "sequence_role": "observational_or_pretraining_input_candidate",
                "label_role": "not_admitted_as_intervention_label_in_D0",
                "license": (
                    "ENCODE_unrestricted_public_release|"
                    "ENA_INSDC_free_unrestricted_access_with_citation"
                ),
                "downstream_overlap_status": "UNKNOWN_REQUIRES_D1_B0_AUDIT",
                "admitted_as_intervention_data": "false",
            }
        )

    verified = sum(row["status"] == "VERIFIED" for row in rows)
    actual_total = sum(int(row["actual_bytes"]) for row in rows)
    expected_total = sum(int(row["expected_bytes"]) for row in rows)
    existing_paths = [
        Path(str(row["path"]))
        for row in rows
        if row["path"] and Path(str(row["path"])).is_file()
    ]
    storage_free = (
        shutil.disk_usage(existing_paths[0].parent).free if existing_paths else None
    )
    complete = (
        len(rows) == EXPECTED_FILES
        and verified == EXPECTED_FILES
        and actual_total == expected_total
        and all(len(str(row["sha256"])) == 64 for row in rows)
    )
    summary = {
        "schema_version": "encode_62_inventory_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_id": "utr_editflow_goal_v2",
        "dataset_id": "ENCSR854RUF",
        "source_manifest": str(source_manifest) if source_manifest else None,
        "source_manifest_sha256": (
            _sha256(source_manifest) if source_manifest is not None else None
        ),
        "reconstruction_manifest": str(reconstruction_manifest),
        "reconstruction_manifest_sha256": _sha256(reconstruction_manifest),
        "expected_files": EXPECTED_FILES,
        "inventory_rows": len(rows),
        "verified_files": verified,
        "incomplete_files": EXPECTED_FILES - verified,
        "expected_bytes": expected_total,
        "actual_bytes": actual_total,
        "complete": complete,
        "evidence_role": "observational_pretraining_candidate_only",
        "admitted_as_intervention_data": False,
        "downstream_overlap_status": "UNKNOWN_REQUIRES_D1_B0_AUDIT",
        "rehash_performed": rehash,
        "io_plan": {
            "inventory_read_mode": (
                "full_sha256_rehash"
                if rehash
                else "stat_plus_downloader_verified_sha256"
            ),
            "bytes_for_full_rehash": expected_total,
            "redundant_concurrent_rehash_forbidden": True,
            "additional_raw_copy_planned": False,
        },
        "disk_plan": {
            "raw_bytes": expected_total,
            "storage_filesystem_free_bytes": storage_free,
            "inventory_output_bytes_expected_lt": 10 * 1024 * 1024,
            "temporary_raw_copy_bytes": 0,
        },
    }
    return rows, summary


def write_inventory(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_csv: Path,
    output_summary: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--rehash", action="store_true")
    args = parser.parse_args(argv)
    rows, summary = build_inventory(
        args.manifest,
        source_manifest=args.source_manifest,
        rehash=args.rehash,
    )
    write_inventory(rows, summary, args.output_csv, args.output_summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
