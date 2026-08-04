#!/usr/bin/env python3
"""Repair the strict pair schema's missing property definition and ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema-dir", type=Path, required=True)
    ap.add_argument("--archive-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    pair = args.schema_dir / "utr_edit_pair.schema.json"
    manifest = args.schema_dir / "SCHEMA_MANIFEST.json"
    sums = args.schema_dir / "SCHEMA_SHA256SUMS"
    sources = [pair, manifest, sums]
    for src in sources:
        dest = args.archive_dir / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    before = {str(p): sha256(p) for p in sources}

    doc = json.loads(pair.read_text(encoding="utf-8"))
    if "future_use_role" not in doc.get("required", []):
        raise RuntimeError("pair schema no longer has future_use_role as a required field")
    if "future_use_role" in doc.get("properties", {}):
        raise RuntimeError("pair schema already contains future_use_role")
    doc.setdefault("properties", {})["future_use_role"] = {"type": "string"}
    pair.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    schema_files = sorted(args.schema_dir.glob("*.schema.json"))
    old_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    entries = []
    for path in schema_files:
        schema_doc = json.loads(path.read_text(encoding="utf-8"))
        entries.append(
            {
                "$id": schema_doc["$id"],
                "contract_id": "utr_editflow_goal_v3.1_benchmark_first",
                "filename": path.name,
                "schema_version": "3.1",
                "sha256": sha256(path),
            }
        )
    new_manifest = dict(old_manifest)
    new_manifest["schemas"] = entries
    manifest.write_text(json.dumps(new_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sums.write_text(
        "\n".join(f"{entry['sha256']}  {entry['filename']}" for entry in entries) + "\n",
        encoding="utf-8",
    )

    after = {str(p): sha256(p) for p in sources}
    report = {
        "artifact_kind": "V3_1_STRICT_SCHEMA_REPAIR",
        "reason": "utr_edit_pair required future_use_role but omitted its property definition",
        "archive_dir": str(args.archive_dir),
        "before_sha256": before,
        "after_sha256": after,
        "schema_filename_set_unchanged": sorted(p.name for p in schema_files)
        == sorted(p.name for p in args.schema_dir.glob("*.schema.json")),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
