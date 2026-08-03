#!/usr/bin/env python
"""D1-R (v3.1): freeze D1 manifest + SHA256SUMS for the technical canonical.

Walks the ordinary and restricted D1 output trees, computes SHA-256 of every
artifact, and emits:
  ordinary: D1_CANONICAL_MANIFEST.json + D1_SHA256SUMS
  restricted: SEALED_CANONICAL_MANIFEST.json + SEALED_CANONICAL_SHA256SUMS

This is a data-engineering tool: no training and no GPU work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def walk(dir: Path) -> dict[str, int]:
    """Return {rel_path: bytes} for all files under dir, sorted."""
    out = {}
    for p in sorted(dir.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(dir))] = p.stat().st_size
    return out


def write_sha256sums(dir: Path, files: dict[str, int], out_name: str, prefix: str):
    lines = []
    for rel in files:
        lines.append(f"{sha256_file(dir / rel)}  {prefix}{rel}")
    (dir / out_name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ordinary", required=True, help="ordinary output dir")
    ap.add_argument("--restricted", required=True, help="restricted run root (base)")
    ap.add_argument("--run-root", required=True, help="run root for manifests")
    ap.add_argument("--config-hash", default="v3.1-D1")
    args = ap.parse_args()

    od = Path(args.ordinary)
    rdir = Path(args.restricted) / "sealed_external" / "GSE246381"
    rr = Path(args.run_root)
    now = datetime.now(timezone.utc).isoformat()

    # ordinary
    od_files = walk(od)
    write_sha256sums(od, od_files, "D1_SHA256SUMS", "")
    od_manifest = {
        "phase": "D1-R",
        "status": "GENERATED",
        "generated_at_utc": now,
        "run_root": str(rr),
        "config_hash": args.config_hash,
        "ordinary_dir": str(od),
        "artifact_files": od_files,
        "sha256sums_file": "D1_SHA256SUMS",
    }
    (od / "D1_CANONICAL_MANIFEST.json").write_text(
        json.dumps(od_manifest, indent=2, sort_keys=True), encoding="utf-8")

    # restricted
    r_files = walk(rdir)
    write_sha256sums(rdir, r_files, "SEALED_CANONICAL_SHA256SUMS", "")
    r_manifest = {
        "phase": "D1-R",
        "status": "SEALED",
        "generated_at_utc": now,
        "config_hash": args.config_hash,
        "restricted_dir": str(rdir),
        "artifact_files": r_files,
        "sha256sums_file": "SEALED_CANONICAL_SHA256SUMS",
    }
    (rdir / "SEALED_CANONICAL_MANIFEST.json").write_text(
        json.dumps(r_manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({
        "ordinary_files": od_files,
        "restricted_files": r_files,
        "status": "DONE",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()