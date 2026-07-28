#!/usr/bin/env python3
"""D0-03: download a MaveDB score set (scores + counts) by URN.

MaveDB API v1 is URN-addressed (free-text search is not exposed; verified
2026-07-28). D0-02 adopted no MaveDB candidate, so this script is not run for
the P0 batch; it is provided so any future MaveDB candidate can be downloaded
with the same checksum/manifest contract:

    python scripts/data/download_mavedb.py --urn urn:mavedb:00000001-a-1

Outputs under ``data/p0/mavedb_{urn_slug}/``: scores.csv (+ counts.csv when
published) and ``manifest.json`` with sha256 checksums.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_common import http_get_json, stream_download, write_manifest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
MAVEDB_API = "https://api.mavedb.org/api/v1/score-sets"


def download_mavedb(urn: str, dest: Path, retries: int = 3) -> dict:
    slug = urn.replace(":", "_")
    dataset_dir = dest / f"mavedb_{slug}"
    retrieved_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    meta = http_get_json(f"{MAVEDB_API}/{urn}")
    manifest = {
        "provider": "MaveDB",
        "accession": urn,
        "source_url": f"{MAVEDB_API}/{urn}",
        "retrieved_at_utc": retrieved_at,
        "title": meta.get("title", ""),
        "files": [],
        "skipped": [],
    }
    for kind in ("scores", "counts"):
        url = f"{MAVEDB_API}/{urn}/{kind}"
        dest_file = dataset_dir / f"{kind}.csv"
        rec = stream_download(url, dest_file, retries=retries)
        if not rec["downloaded"] and kind == "counts":
            # counts are optional for some score sets
            manifest["skipped"].append({"name": "counts.csv",
                                        "reason": rec.get("error", "unavailable")})
            continue
        manifest["files"].append(rec)
    write_manifest(dataset_dir, manifest)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urn", required=True)
    parser.add_argument("--dest", default=str(REPO_ROOT / "data/p0"))
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args(argv)

    manifest = download_mavedb(args.urn, Path(args.dest), retries=args.retries)
    n_ok = sum(1 for f in manifest["files"] if f["downloaded"])
    n_fail = sum(1 for f in manifest["files"] if not f["downloaded"])
    print(f"{args.urn}: {n_ok} downloaded, {n_fail} failed, "
          f"{len(manifest['skipped'])} skipped")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
