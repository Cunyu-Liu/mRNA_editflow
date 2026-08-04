#!/usr/bin/env python3
"""Aggregate identity-row provenance without returning row payloads."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    identity = Counter()
    record_types = Counter()
    with args.input.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            meta = row.get("metadata") or {}
            typ = meta.get("record_type", "<MISSING>")
            accession = row.get("accession", "<MISSING>")
            source_file = meta.get("source_file", "<MISSING>")
            library = meta.get("library", "<MISSING>")
            record_types[(accession, typ)] += 1
            if typ == "identity":
                identity[(accession, source_file, library)] += 1
    report = {
        "identity_groups": {"|".join(key): value for key, value in sorted(identity.items())},
        "record_types": {"|".join(key): value for key, value in sorted(record_types.items())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
