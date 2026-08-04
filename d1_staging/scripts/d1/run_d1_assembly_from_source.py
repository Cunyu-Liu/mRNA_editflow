#!/usr/bin/env python3
"""Run the explicit D1 assembler from a hash-bound source attempt."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ORDINARY = [
    "GSE114002.jsonl",
    "GSE114002_identity_support.jsonl",
    "GSE145046.jsonl",
    "GSE176581_support.jsonl",
    "GSE200304.jsonl",
    "GSE217518.jsonl",
    "ENCSR854RUF.jsonl",
    "GSE149487.jsonl",
    "GSE232572.jsonl",
    "GSE186455.jsonl",
]
LEGACY = ["GSE207584.jsonl", "GSE173083.jsonl"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--dest-root", type=Path, required=True)
    ap.add_argument("--assembler", type=Path, required=True)
    ap.add_argument("--python", type=Path, required=True)
    args = ap.parse_args()
    source_parts = args.source_root / "canonical_parts" / "ordinary"
    args.dest_root.mkdir(parents=True, exist_ok=True)
    (args.dest_root / "quarantine").mkdir(parents=True, exist_ok=True)
    ordinary_args = []
    legacy_args = []
    for name in ORDINARY:
        path = source_parts / name
        if not path.exists():
            raise FileNotFoundError(path)
        ordinary_args.extend(["--part", str(path)])
    for name in LEGACY:
        path = source_parts / name
        if not path.exists():
            raise FileNotFoundError(path)
        legacy_args.extend(["--legacy", str(path)])
    command = [
        str(args.python),
        str(args.assembler),
        *ordinary_args,
        *legacy_args,
        "--output",
        str(args.dest_root / "canonical" / "ordinary_d1_canonical_records.jsonl"),
        "--legacy-output",
        str(args.dest_root / "quarantine" / "legacy_records.jsonl"),
        "--summary",
        str(args.dest_root / "D1_PART_ASSEMBLY_SUMMARY.json"),
    ]
    log = (args.dest_root / "D1_PART_ASSEMBLY.log").open("w", encoding="utf-8")
    try:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    finally:
        log.close()
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
