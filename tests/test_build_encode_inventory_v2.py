from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from scripts.data.build_encode_inventory_v2 import build_inventory, write_inventory

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fixtures(tmp_path):
    records = []
    completed = []
    for index in range(62):
        accession = f"ENCFF{index:06d}"
        path = tmp_path / f"{accession}.fastq.gz"
        path.write_bytes(f"payload-{index}".encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        source = {
            "encode_accession": accession,
            "name": path.name,
            "bytes": path.stat().st_size,
            "ok": True,
        }
        records.append(source)
        completed.append(
            {
                **source,
                "path": str(path),
                "status": "VERIFIED",
                "sha256": digest,
            }
        )
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(records), encoding="utf-8")
    reconstruction_path = tmp_path / "reconstruction.json"
    reconstruction_path.write_text(
        json.dumps(
            {
                "expected_files": 62,
                "source_manifest": str(source_path),
                "files": completed,
            }
        ),
        encoding="utf-8",
    )
    return source_path, reconstruction_path


def test_complete_inventory_requires_all_62_verified_rows(tmp_path):
    source, reconstruction = _fixtures(tmp_path)
    rows, summary = build_inventory(reconstruction, source_manifest=source)
    assert len(rows) == 62
    assert summary["complete"] is True
    assert summary["verified_files"] == 62
    assert len(summary["source_manifest_sha256"]) == 64
    assert len(summary["reconstruction_manifest_sha256"]) == 64
    assert summary["io_plan"]["bytes_for_full_rehash"] == summary["expected_bytes"]
    assert summary["io_plan"]["additional_raw_copy_planned"] is False
    assert summary["disk_plan"]["storage_filesystem_free_bytes"] > 0
    assert all(row["admitted_as_intervention_data"] == "false" for row in rows)
    assert all("ENA_INSDC_free_unrestricted_access" in row["license"] for row in rows)


def test_missing_reconstruction_row_is_explicit_and_incomplete(tmp_path):
    source, reconstruction = _fixtures(tmp_path)
    payload = json.loads(reconstruction.read_text(encoding="utf-8"))
    payload["files"].pop()
    reconstruction.write_text(json.dumps(payload), encoding="utf-8")
    rows, summary = build_inventory(reconstruction, source_manifest=source)
    assert len(rows) == 62
    assert summary["complete"] is False
    assert summary["verified_files"] == 61
    assert any(row["status"] == "MISSING_MANIFEST_ROW" for row in rows)

    output_csv = tmp_path / "inventory.csv"
    output_summary = tmp_path / "summary.json"
    write_inventory(rows, summary, output_csv, output_summary)
    assert output_csv.read_text(encoding="utf-8").count("\n") == 63
    assert json.loads(output_summary.read_text(encoding="utf-8"))["complete"] is False


def test_repository_inventory_snapshot_is_self_consistent_and_fail_closed():
    inventory_path = REPO_ROOT / "data_registry" / "encode_62_inventory_v2.csv"
    summary_path = (
        REPO_ROOT / "data_registry" / "encode_62_inventory_v2_summary.json"
    )
    assert inventory_path.is_file()
    assert summary_path.is_file()

    with inventory_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    verified = [row for row in rows if row["status"] == "VERIFIED"]
    incomplete = [row for row in rows if row["status"] != "VERIFIED"]
    assert len(rows) == summary["inventory_rows"] == summary["expected_files"] == 62
    assert len(verified) == summary["verified_files"]
    assert len(incomplete) == summary["incomplete_files"]
    assert summary["complete"] is (len(verified) == 62)
    assert sum(int(row["actual_bytes"]) for row in rows) == summary["actual_bytes"]
    assert sum(int(row["expected_bytes"]) for row in rows) == summary["expected_bytes"]
    assert all(len(row["sha256"]) == 64 for row in verified)
    assert all(row["actual_bytes"] == row["expected_bytes"] for row in verified)
    assert all(row["status"] for row in incomplete)
    assert all(row["admitted_as_intervention_data"] == "false" for row in rows)
    assert summary["admitted_as_intervention_data"] is False
