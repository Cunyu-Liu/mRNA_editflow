#!/usr/bin/env python3
"""Download and verify ENA FASTQ files used to reconstruct ENCODE ENCSR854RUF raw reads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any


USER_AGENT = "mrna-editflow-d0/2.0 (utr_editflow_goal_v2)"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def looks_like_fastq_gzip(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(2) == b"\x1f\x8b"


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def verify(path: Path, expected_bytes: int, expected_sha256: str | None) -> tuple[bool, str | None]:
    if not path.is_file():
        return False, "missing"
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        return False, f"size:{actual_bytes}!={expected_bytes}"
    if not looks_like_fastq_gzip(path):
        return False, "not_gzip"
    actual_sha256 = sha256_file(path)
    if expected_sha256 and actual_sha256 != expected_sha256:
        return False, f"sha256:{actual_sha256}!={expected_sha256}"
    return True, actual_sha256


def download_one(record: dict[str, Any], dest: Path, timeout: int, retries: int) -> dict[str, Any]:
    accession = record["encode_accession"]
    expected_bytes = int(record["bytes"])
    final_path = dest / record["name"]
    part_path = final_path.with_name(final_path.name + ".part")
    final_path.parent.mkdir(parents=True, exist_ok=True)

    ok, detail = verify(final_path, expected_bytes, record.get("sha256"))
    if ok:
        return {**record, "path": str(final_path), "status": "VERIFIED", "downloaded": True, "sha256": detail}

    command = [
        "curl", "-fSL", "--http1.1", "--retry", str(retries), "--retry-delay", "10",
        "--continue-at", "-", "--connect-timeout", "60", "--max-time", str(timeout),
        "-A", USER_AGENT, "-o", str(part_path), record["url"],
    ]
    completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        return {**record, "path": str(part_path), "status": "DOWNLOAD_FAILED", "downloaded": False,
                "error": completed.stderr[-2000:]}

    ok, detail = verify(part_path, expected_bytes, record.get("sha256"))
    if not ok:
        return {**record, "path": str(part_path), "status": "VERIFY_FAILED", "downloaded": False,
                "error": detail}
    part_path.replace(final_path)
    return {**record, "path": str(final_path), "status": "VERIFIED", "downloaded": True, "sha256": detail}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=86400)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()

    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    if len(records) != 62 or any(not item.get("ok") for item in records):
        raise SystemExit("ENA manifest must contain 62 successful HEAD records")
    args.dest.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any] | None] = [None] * len(records)
    lock = Lock()

    def persist() -> None:
        with lock:
            atomic_write_json(args.output, {
                "schema_version": "ena_reconstruction_manifest.v1",
                "source_manifest": str(args.manifest),
                "destination": str(args.dest),
                "expected_files": len(records),
                "expected_bytes": sum(int(item["bytes"]) for item in records),
                "files": [item for item in results if item is not None],
            })

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, item, args.dest, args.timeout, args.retries): idx for idx, item in enumerate(records)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # retain a machine-readable failure record
                result = {**records[idx], "status": "EXCEPTION", "downloaded": False,
                          "error": f"{type(exc).__name__}: {exc}"}
            results[idx] = result
            persist()
            done = sum(item is not None for item in results)
            good = sum(item is not None and item.get("status") == "VERIFIED" for item in results)
            print(f"completed={done}/{len(records)} verified={good}", flush=True)

    final = [item for item in results if item is not None]
    atomic_write_json(args.output, {
        "schema_version": "ena_reconstruction_manifest.v1",
        "source_manifest": str(args.manifest),
        "destination": str(args.dest),
        "expected_files": len(records),
        "expected_bytes": sum(int(item["bytes"]) for item in records),
        "files": final,
    })
    return 0 if len(final) == 62 and all(item.get("status") == "VERIFIED" for item in final) else 2


if __name__ == "__main__":
    raise SystemExit(main())
