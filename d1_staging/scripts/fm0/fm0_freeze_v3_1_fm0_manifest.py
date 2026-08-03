#!/usr/bin/env python
"""FM0-A (v3.1): freeze FM0-A manifest + SHA256SUMS.

Walks the ordinary FM0 output tree and the restricted GSE246381 FM0 mirror,
computes SHA-256 of every artifact, and emits:
  ordinary: FOUNDATION_EXPOSURE_MANIFEST.json + FOUNDATION_EXPOSURE_SHA256SUMS
  restricted: FM0_AGGREGATE.json already written; SEALED_FM0_MANIFEST.json

No training and no GPU work.
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
    out = {}
    for p in sorted(dir.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(dir))] = p.stat().st_size
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="ordinary FM0 output dir")
    ap.add_argument("--restricted-dir", required=True, help="restricted GSE246381 dir")
    ap.add_argument("--run-root", required=True, help="run root for run-level artifacts")
    ap.add_argument("--config-hash", default="v3.1-FM0-A")
    args = ap.parse_args()

    d = Path(args.out_dir)
    rdir = Path(args.restricted_dir)
    rr = Path(args.run_root)
    now = datetime.now(timezone.utc).isoformat()

    files = walk(d)
    lines = []
    for rel in files:
        lines.append(f"{sha256_file(d / rel)}  {rel}")
    (d / "FOUNDATION_EXPOSURE_SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "phase": "FM0-A",
        "status": "GENERATED",
        "generated_at_utc": now,
        "run_root": str(rr),
        "config_hash": args.config_hash,
        "out_dir": str(d),
        "artifact_files": files,
        "sha256sums_file": "FOUNDATION_EXPOSURE_SHA256SUMS",
    }
    (d / "FOUNDATION_EXPOSURE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    # restricted manifest (mirror of FM0 aggregate)
    r_files = walk(rdir)
    r_manifest = {
        "phase": "FM0-A",
        "status": "SEALED",
        "generated_at_utc": now,
        "config_hash": args.config_hash,
        "restricted_dir": str(rdir),
        "artifact_files": r_files,
    }
    (rdir / "SEALED_FM0_MANIFEST.json").write_text(
        json.dumps(r_manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({
        "ordinary_files": files,
        "restricted_files": r_files,
        "status": "DONE",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()