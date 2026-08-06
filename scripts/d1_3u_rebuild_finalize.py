#!/usr/bin/env python3
"""D1 3U-A1 rebuild finalizer: close the rebuilt technical-canonical artifacts.

Generates D1_SHA256SUMS + D1_CANONICAL_MANIFEST.json for the rebuilt ordinary
staging dir produced by build_v3_1_technical_canonical.py, mirroring the v3.1
D1-R manifest format. Does NOT overwrite the legacy /mnt .../d1/ canonical.

Usage:
    python scripts/d1_3u_rebuild_finalize.py --dir /mnt/.../d1_3u_rebuild_staging/ordinary \
        --config-hash v3.1-D1 --phase D1-3U-REBUILD
"""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ARTIFACT_KEYS = [
    "effective_exposure_projection.jsonl",
    "endpoint_registry.jsonl",
    "exposure_records.jsonl",
    "functional_observation_candidates.jsonl",
    "functional_observations.jsonl",
    "group_assignments.jsonl",
    "group_registry.jsonl",
    "object_attributes.jsonl",
    "rejection_records.jsonl",
    "sequence_entities.jsonl",
    "transformation_edges.jsonl",
    "use_roles.jsonl",
    "utr_edit_pairs.jsonl",
    "utr_edit_relation_candidates.jsonl",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="ordinary output dir")
    ap.add_argument("--config-hash", default="v3.1-D1")
    ap.add_argument("--phase", default="D1-3U-REBUILD")
    args = ap.parse_args()

    d = Path(args.dir)
    sums = {}
    sizes = {}
    missing = []
    for key in ARTIFACT_KEYS:
        p = d / key
        if not p.exists():
            missing.append(key)
            continue
        sums[key] = sha256_file(p)
        sizes[key] = p.stat().st_size

    if missing:
        raise SystemExit(f"FAIL: missing artifacts: {missing}")

    sums_lines = "".join(f"{v}  {k}\n" for k, v in sums.items())
    (d / "D1_SHA256SUMS").write_text(sums_lines, encoding="utf-8")

    manifest = {
        "artifact_files": sizes,
        "config_hash": args.config_hash,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ordinary_dir": str(d),
        "phase": args.phase,
        "run_root": "/home/cunyuliu/mrna_editflow_goal/runs/d1_3u_rebuild_20260806",
        "sha256sums_file": "D1_SHA256SUMS",
        "status": "GENERATED",
    }
    (d / "D1_CANONICAL_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"finalized {len(sums)} artifacts in {d}")
    key = chr(117) + chr(116) + chr(114) + chr(95) + chr(101) + chr(100) + chr(105) + chr(116) + chr(95) + chr(112) + chr(97) + chr(105) + chr(114) + chr(115) + chr(46) + chr(106) + chr(115) + chr(111) + chr(110) + chr(108)
    print(f"utr_edit_pairs.jsonl bytes={sizes[key]} sha256={sums[key]}")


if __name__ == "__main__":
    main()
