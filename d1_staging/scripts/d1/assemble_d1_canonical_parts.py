#!/usr/bin/env python3
"""Assemble explicit per-dataset D1 adapter outputs without glob guessing."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", action="append", required=True)
    ap.add_argument("--legacy", action="append", default=[])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--legacy-output", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.legacy_output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    counts = Counter()
    files = []
    duplicate_ids = []

    def copy_parts(paths, output, bucket):
        nonlocal duplicate_ids
        with output.open("w", encoding="utf-8") as out:
            for raw_path in paths:
                path = Path(raw_path)
                file_count = 0
                with path.open("r", encoding="utf-8") as fh:
                    for lineno, line in enumerate(fh, 1):
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        rid = rec.get("record_id")
                        if not rid:
                            raise ValueError(f"{path}:{lineno}: missing record_id")
                        if bucket == "ordinary":
                            if rid in seen:
                                duplicate_ids.append({"record_id": rid, "path": str(path), "line": lineno})
                            seen.add(rid)
                        out.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
                        file_count += 1
                        accession = rec.get("accession", "UNKNOWN")
                        counts[f"{bucket}:{accession}:records"] += 1
                        if rec.get("source_sequence") and rec.get("candidate_sequence"):
                            counts[f"{bucket}:{accession}:paired"] += 1
                        elif rec.get("candidate_sequence"):
                            counts[f"{bucket}:{accession}:sequence_only"] += 1
                        else:
                            counts[f"{bucket}:{accession}:no_sequence"] += 1
                files.append({"path": str(path), "sha256": sha256_file(path), "records": file_count, "bucket": bucket})

    copy_parts(args.part, args.output, "ordinary")
    copy_parts(args.legacy, args.legacy_output, "legacy_quarantine")
    summary = {
        "artifact_kind": "D1_CANONICAL_PART_ASSEMBLY",
        "ordinary_parts": args.part,
        "legacy_parts": args.legacy,
        "files": files,
        "counts": dict(sorted(counts.items())),
        "duplicate_record_ids": duplicate_ids[:100],
        "duplicate_record_id_count": len(duplicate_ids),
        "ordinary_record_id_set_unique": not duplicate_ids,
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if not duplicate_ids else 1


if __name__ == "__main__":
    raise SystemExit(main())
