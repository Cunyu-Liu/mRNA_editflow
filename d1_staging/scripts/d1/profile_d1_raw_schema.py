#!/usr/bin/env python3
"""Metadata-only profile of assembled D1 raw records.

The profiler never emits sequence, label, or sample values.  It records only
schema shapes, bounded categorical counts, and source-file to D0-manifest
matching evidence needed before implementing a strict D1 builder.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", type=Path, required=True)
    ap.add_argument("--raw-asset-manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    manifest_by_name: dict[str, set[str]] = defaultdict(set)
    manifest_by_asset: dict[str, dict] = {}
    for line in args.raw_asset_manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        relpath = str(row.get("relpath") or "")
        name = Path(relpath).name
        if name:
            manifest_by_name[name].add(str(row.get("asset_id")))
        manifest_by_asset[str(row.get("asset_id"))] = {
            "sha256": row.get("sha256"),
            "relpath": relpath,
            "byte_size": row.get("byte_size"),
            "mapping_status": row.get("mapping_status"),
            "parse_status": row.get("parse_status"),
        }

    total = 0
    dataset = Counter()
    record_type = Counter()
    region = Counter()
    source_file = Counter()
    label_key = Counter()
    source_presence = Counter()
    candidate_presence = Counter()
    equality = Counter()
    edit_verified = Counter()
    metadata_key = Counter()
    source_file_match = Counter()
    source_file_assets: dict[str, set[str]] = defaultdict(set)
    first_record_shape = None

    with args.canonical.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            total += 1
            if first_record_shape is None:
                first_record_shape = {
                    "top_level_keys": sorted(rec),
                    "metadata_keys": sorted((rec.get("metadata") or {}).keys()),
                    "label_keys": sorted((rec.get("labels") or {}).keys()),
                    "edit_script_item_keys": sorted((rec.get("edit_script") or [{}])[0])
                    if rec.get("edit_script") else [],
                }
            ds = str(rec.get("dataset") or "")
            dataset[ds] += 1
            meta = rec.get("metadata") or {}
            record_type[str(meta.get("record_type") or "<MISSING>")] += 1
            metadata_key.update(meta.keys())
            region[str(rec.get("region") or "<MISSING>")] += 1
            sf = str(meta.get("source_file") or "<MISSING>")
            source_file[sf] += 1
            for key in (rec.get("labels") or {}):
                label_key[str(key)] += 1
            has_source = bool(rec.get("source_sequence"))
            has_candidate = bool(rec.get("candidate_sequence"))
            source_presence[str(has_source)] += 1
            candidate_presence[str(has_candidate)] += 1
            if has_source and has_candidate:
                equality["equal" if rec["source_sequence"] == rec["candidate_sequence"] else "different"] += 1
            else:
                equality["not_both_present"] += 1
            edit_verified[str(rec.get("edit_script_verified"))] += 1
            sf_name = Path(sf).name
            matches = manifest_by_name.get(sf_name, set())
            if matches:
                source_file_match["basename_match"] += 1
                source_file_assets[sf].update(matches)
            else:
                source_file_match["unmatched"] += 1

    report = {
        "artifact_kind": "D1_RAW_SCHEMA_PROFILE",
        "privacy_scope": "metadata_only_no_sequence_or_label_values",
        "input": str(args.canonical),
        "raw_asset_manifest": str(args.raw_asset_manifest),
        "total_records": total,
        "first_record_shape": first_record_shape,
        "dataset_counts": dict(sorted(dataset.items())),
        "record_type_counts": dict(sorted(record_type.items())),
        "region_counts": dict(sorted(region.items())),
        "label_key_counts": dict(sorted(label_key.items())),
        "source_presence_counts": dict(sorted(source_presence.items())),
        "candidate_presence_counts": dict(sorted(candidate_presence.items())),
        "source_candidate_equality_counts": dict(sorted(equality.items())),
        "edit_script_verified_counts": dict(sorted(edit_verified.items())),
        "metadata_keys": sorted(metadata_key),
        "source_file_counts": dict(sorted(source_file.items())),
        "source_file_match_counts": dict(sorted(source_file_match.items())),
        "source_file_manifest_assets": {
            key: sorted(value) for key, value in sorted(source_file_assets.items())
        },
        "manifest_asset_count": len(manifest_by_asset),
        "manifest_asset_metadata": {
            key: value for key, value in sorted(manifest_by_asset.items())
            if key in {asset for values in source_file_assets.values() for asset in values}
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
