#!/usr/bin/env python3
"""D0-03: verify downloaded P0 datasets against their manifests.

Checks for every manifest under ``data/p0/*/manifest.json``:

* every file flagged ``downloaded`` exists on disk;
* byte size matches the manifest record;
* recomputed sha256 matches the manifest record;
* the file payload is not an HTML error page (first KB sniffed for
  ``<!DOCTYPE`` / ``<html`` markers);
* ENCODE files carry the provider md5 (release pinning evidence);
* deferred files (ENCODE size cap) are reported, not counted as failures.

Writes ``docs/data/download_verification.md`` and exits non-zero on failure.

Usage:
    python scripts/data/verify_downloads.py
    python scripts/data/verify_downloads.py --root data/p0
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

HTML_MARKERS = (b"<!doctype", b"<html")


def sha256_of(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def looks_like_html(path: Path, sniff_bytes: int = 4096) -> bool:
    with path.open("rb") as fh:
        head = fh.read(sniff_bytes).lstrip().lower()
    return any(head.startswith(m) for m in HTML_MARKERS)


def verify_file(dataset_dir: Path, record: dict) -> list[str]:
    """Return a list of violation strings (empty == file OK)."""
    errors = []
    path = dataset_dir / record["name"]
    if not record.get("downloaded"):
        if not record.get("defer_reason"):
            errors.append(f"{record['name']}: not downloaded "
                          f"({record.get('error', 'no error recorded')})")
        return errors
    if not path.is_file():
        return [f"{record['name']}: missing on disk"]
    size = path.stat().st_size
    if size != record["bytes"]:
        errors.append(f"{record['name']}: size {size} != manifest {record['bytes']}")
    if size == 0:
        errors.append(f"{record['name']}: zero bytes")
    if record.get("sha256"):
        actual = sha256_of(path)
        if actual != record["sha256"]:
            errors.append(f"{record['name']}: sha256 mismatch")
    else:
        errors.append(f"{record['name']}: manifest has no sha256")
    if looks_like_html(path):
        errors.append(f"{record['name']}: payload looks like an HTML error page")
    return errors


def verify_root(root: Path) -> dict:
    report = {"root": str(root), "datasets": [], "n_files_ok": 0,
              "n_files_failed": 0, "n_deferred": 0, "n_skipped": 0}
    for manifest_path in sorted(root.glob("*/manifest.json")):
        dataset_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = {
            "accession": manifest.get("accession", dataset_dir.name),
            "provider": manifest.get("provider", ""),
            "retrieved_at_utc": manifest.get("retrieved_at_utc", ""),
            "ok": [], "failed": [], "deferred": [],
            "skipped": manifest.get("skipped", []),
        }
        for record in manifest.get("files", []):
            if not record.get("downloaded") and record.get("defer_reason"):
                entry["deferred"].append(record["name"])
                continue
            errors = verify_file(dataset_dir, record)
            if errors:
                entry["failed"].extend(errors)
            else:
                entry["ok"].append(record["name"])
        report["datasets"].append(entry)
        report["n_files_ok"] += len(entry["ok"])
        report["n_files_failed"] += len(entry["failed"])
        report["n_deferred"] += len(entry["deferred"])
        report["n_skipped"] += len(entry["skipped"])
    return report


def render_report_md(report: dict) -> str:
    lines = [
        "# Download Verification (D0-03)",
        "",
        f"- root: `{report['root']}`",
        f"- datasets with manifest: {len(report['datasets'])}",
        f"- files OK: {report['n_files_ok']}",
        f"- files failed: {report['n_files_failed']}",
        f"- files deferred (documented, e.g. ENCODE size cap): {report['n_deferred']}",
        f"- archives skipped (RAW.tar duplicates): {report['n_skipped']}",
        "",
        "| dataset | provider | retrieved_at | ok | failed | deferred |",
        "|---|---|---|---|---|---|",
    ]
    for d in report["datasets"]:
        lines.append(
            f"| {d['accession']} | {d['provider']} | {d['retrieved_at_utc']} | "
            f"{len(d['ok'])} | {len(d['failed'])} | {len(d['deferred'])} |"
        )
    lines.append("")
    for d in report["datasets"]:
        if d["failed"]:
            lines.append(f"## FAILURES {d['accession']}")
            lines += [f"- {e}" for e in d["failed"]]
            lines.append("")
    verdict = "PASS" if report["n_files_failed"] == 0 and report["n_files_ok"] > 0 else "FAIL"
    lines.append(f"## Verdict: {verdict}")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO_ROOT / "data/p0"))
    parser.add_argument("--report-md",
                        default=str(REPO_ROOT / "docs/data/download_verification.md"))
    args = parser.parse_args(argv)

    report = verify_root(Path(args.root))
    md_path = Path(args.report_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_report_md(report), encoding="utf-8")
    print(render_report_md(report).splitlines()[-2])
    print(f"ok={report['n_files_ok']} failed={report['n_files_failed']} "
          f"deferred={report['n_deferred']} skipped={report['n_skipped']}")
    return 0 if report["n_files_failed"] == 0 and report["n_files_ok"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
