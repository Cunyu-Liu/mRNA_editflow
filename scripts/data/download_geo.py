#!/usr/bin/env python3
"""D0-03: download GEO series supplementary files with checksums.

For a GEO series accession (GSExxxxx):

1. parse ``suppl/filelist.txt`` (name/size table); when the series has no
   filelist.txt, fall back to scraping the ``suppl/`` directory index;
2. download every listed file (per-sample supplementary tables). ``*_RAW.tar``
   archives are skipped by default because they duplicate the individually
   listed per-sample files; ``--include-archive`` overrides;
3. record per-file sha256/bytes in ``manifest.json`` next to the files.

Idempotent: files whose size+sha256 already match the manifest are skipped.

Usage:
    python scripts/data/download_geo.py --accession GSE145046
    python scripts/data/download_geo.py --accession GSE114002 --dest data/p0
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_common import (  # noqa: E402
    already_downloaded,
    http_get,
    http_head_size,
    stream_download,
    write_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo"

GSM_PATTERN = re.compile(r"(GSM\d+)")


def series_prefix(accession: str) -> str:
    """GSE145046 -> GSE145nnn"""
    return accession[:-3] + "nnn"


def suppl_base(accession: str) -> str:
    return f"{GEO_FTP}/series/{series_prefix(accession)}/{accession}/suppl"


def file_url(accession: str, filename: str) -> str:
    """Resolve a filelist.txt entry to its real URL.

    GEO stores series-level supplementary files under the series ``suppl/``
    directory, but per-sample files (named ``GSMxxxx_*``) under the sample
    page ``geo/samples/{GSMprefix}/{GSM}/suppl/``. filelist.txt mixes both.
    """
    match = GSM_PATTERN.match(filename)
    if match:
        gsm = match.group(1)
        return f"{GEO_FTP}/samples/{gsm[:-3]}nnn/{gsm}/suppl/{filename}"
    return f"{suppl_base(accession)}/{filename}"


def parse_filelist(text: str) -> list[dict]:
    """Parse suppl/filelist.txt rows into {name, size, kind} dicts."""
    files = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) < 3:
            continue
        kind, name = parts[0], parts[1]
        if kind not in ("Archive", "File"):
            continue
        size = None
        for token in reversed(parts[2:]):
            if re.fullmatch(r"\d+", token):
                size = int(token)
                break
        files.append({"name": name, "size": size,
                      "kind": "archive" if kind == "Archive" else "file"})
    return files


def parse_dir_index(html: str) -> list[dict]:
    """Fallback: scrape the suppl/ Apache-style index for file hrefs."""
    files = []
    for href in re.findall(r'href="([^"]+)"', html):
        if href.endswith("/") or href.startswith("/") or href.startswith("http"):
            continue  # directories, absolute links, external links
        files.append({"name": href, "size": None, "kind": "file"})
    return files


def list_supplementary(accession: str) -> list[dict]:
    base = suppl_base(accession)
    try:
        text = http_get(f"{base}/filelist.txt").decode("utf-8", errors="replace")
        if "<html" not in text[:200].lower():
            files = parse_filelist(text)
            if files:
                return files
    except Exception:
        pass
    html = http_get(f"{base}/").decode("utf-8", errors="replace")
    return parse_dir_index(html)


def download_geo(accession: str, dest: Path, include_archive: bool = False,
                 retries: int = 3, workers: int = 1) -> dict:
    dataset_dir = dest / accession
    base = suppl_base(accession)
    retrieved_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    listed = list_supplementary(accession)
    manifest = {
        "provider": "GEO",
        "accession": accession,
        "source_url": f"{base}/",
        "retrieved_at_utc": retrieved_at,
        "files": [],
        "skipped": [],
    }

    old_files = {}
    old_manifest = dataset_dir / "manifest.json"
    if old_manifest.is_file():
        import json
        old_files = {f["name"]: f for f in json.loads(old_manifest.read_text()).get("files", [])}

    todo: list[tuple[dict, str, Path, int | None]] = []
    for item in listed:
        name = item["name"]
        if item["kind"] == "archive" or name.endswith("_RAW.tar"):
            if not include_archive:
                manifest["skipped"].append(
                    {"name": name, "size": item.get("size"),
                     "reason": "RAW.tar archive duplicates per-sample files"})
                continue
        url = file_url(accession, name)
        dest_file = dataset_dir / name
        expected_size = item.get("size")
        if expected_size is None:
            expected_size = http_head_size(url)
        old = old_files.get(name)
        if old and old.get("downloaded") and already_downloaded(dest_file, old):
            manifest["files"].append(old)
            continue
        todo.append((item, url, dest_file, expected_size))

    def fetch(job: tuple[dict, str, Path, int | None]) -> dict:
        item, url, dest_file, expected_size = job
        rec = stream_download(url, dest_file, retries=retries)
        if expected_size is not None and rec["downloaded"] and rec["bytes"] != expected_size:
            rec["downloaded"] = False
            rec["error"] = f"size mismatch: got {rec['bytes']}, expected {expected_size}"
            dest_file.unlink(missing_ok=True)
        rec["expected_bytes"] = expected_size
        return rec

    if workers > 1 and len(todo) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            manifest["files"].extend(pool.map(fetch, todo))
    else:
        manifest["files"].extend(fetch(job) for job in todo)
    write_manifest(dataset_dir, manifest)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accession", required=True)
    parser.add_argument("--dest", default=str(REPO_ROOT / "data/p0"))
    parser.add_argument("--include-archive", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel download connections (each spawns curl)")
    args = parser.parse_args(argv)

    manifest = download_geo(args.accession, Path(args.dest),
                            include_archive=args.include_archive,
                            retries=args.retries, workers=args.workers)
    n_ok = sum(1 for f in manifest["files"] if f["downloaded"])
    n_fail = sum(1 for f in manifest["files"] if not f["downloaded"])
    print(f"{args.accession}: {n_ok} downloaded, {n_fail} failed, "
          f"{len(manifest['skipped'])} skipped")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
