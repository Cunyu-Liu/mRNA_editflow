#!/usr/bin/env python3
"""D0-03: download ENCODE publication-data files with checksums.

For an ENCODE accession (e.g. ENCSR854RUF / MPRAu):

1. read ``/publication-data/{accession}/?format=json`` -> related files;
2. read every file's metadata (md5sum, file_size, href, file_type);
3. download files while the cumulative payload stays within ``--max-bytes``
   (default 5 GB). Files beyond the cap are recorded as ``deferred`` with
   their provider md5 — the accession and release stay pinned, and D0-04
   documents the reconstruction path. This prevents a single ~350 GB fastq
   set from exhausting the storage quota;
4. record per-file sha256 (recomputed) + provider md5 in ``manifest.json``.

Usage:
    python scripts/data/download_encode.py --accession ENCSR854RUF
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_common import (  # noqa: E402
    already_downloaded,
    http_get_json,
    stream_download,
    write_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ENCODE = "https://www.encodeproject.org"


def list_related_files(accession: str) -> list[dict]:
    data = http_get_json(f"{ENCODE}/publication-data/{accession}/?format=json")
    records = []
    for path in data.get("related_files", []):
        meta = http_get_json(f"{ENCODE}{path}?format=json")
        records.append({
            "file_accession": path.strip("/").split("/")[-1],
            "file_type": meta.get("file_type", ""),
            "output_type": meta.get("output_type", ""),
            "file_size": meta.get("file_size", 0),
            "md5sum": meta.get("md5sum", ""),
            "href": meta.get("href", ""),
        })
    return records


def download_encode(accession: str, dest: Path, max_bytes: int = 5 * (1 << 30),
                    retries: int = 3) -> dict:
    dataset_dir = dest / accession
    retrieved_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    related = list_related_files(accession)
    manifest = {
        "provider": "ENCODE",
        "accession": accession,
        "source_url": f"{ENCODE}/publication-data/{accession}/",
        "retrieved_at_utc": retrieved_at,
        "max_bytes_cap": max_bytes,
        "files": [],
        "skipped": [],
    }

    old_files = {}
    old_manifest = dataset_dir / "manifest.json"
    if old_manifest.is_file():
        import json
        old_files = {f["name"]: f for f in json.loads(old_manifest.read_text()).get("files", [])}

    spent = 0
    for rec in related:
        size = rec["file_size"] or 0
        name = f"{rec['file_accession']}{_suffix(rec['href'])}"
        url = f"{ENCODE}{rec['href']}" if rec["href"] else ""
        if not url:
            manifest["skipped"].append({"name": name, "size": size, "reason": "no href"})
            continue
        if spent + size > max_bytes:
            manifest["files"].append({
                "name": name, "url": url, "bytes": size, "sha256": "",
                "downloaded": False, "provider_md5": rec["md5sum"],
                "defer_reason": f"exceeds max_bytes cap ({max_bytes}); raw reads "
                                f"reconstructable on demand (see D0-04)",
            })
            continue
        dest_file = dataset_dir / name
        old = old_files.get(name)
        if old and old.get("downloaded") and already_downloaded(dest_file, old):
            manifest["files"].append(old)
            spent += old["bytes"]
            continue
        out = stream_download(url, dest_file, retries=retries)
        out["provider_md5"] = rec["md5sum"]
        if out["downloaded"]:
            spent += out["bytes"]
        manifest["files"].append(out)
    write_manifest(dataset_dir, manifest)
    return manifest


def _suffix(href: str) -> str:
    """'.fastq.gz' style suffix extracted from an ENCODE download href."""
    base = href.rsplit("/", 1)[-1] if href else ""
    if "." in base:
        return "." + base.split(".", 1)[1]
    return ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accession", required=True)
    parser.add_argument("--dest", default=str(REPO_ROOT / "data/p0"))
    parser.add_argument("--max-bytes", type=int, default=5 * (1 << 30))
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args(argv)

    manifest = download_encode(args.accession, Path(args.dest),
                               max_bytes=args.max_bytes, retries=args.retries)
    n_ok = sum(1 for f in manifest["files"] if f["downloaded"])
    n_defer = sum(1 for f in manifest["files"]
                  if not f["downloaded"] and f.get("defer_reason"))
    n_err = sum(1 for f in manifest["files"]
                if not f["downloaded"] and not f.get("defer_reason"))
    print(f"{args.accession}: {n_ok} downloaded, {n_defer} deferred (cap), "
          f"{n_err} failed, {len(manifest['skipped'])} skipped")
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
