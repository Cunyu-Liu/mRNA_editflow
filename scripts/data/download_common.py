#!/usr/bin/env python3
"""Shared HTTP + checksummed download helpers for D0-03 download scripts."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

USER_AGENT = "mrna-editflow-d0/2.0 (mrna_editflow_single_active_contract)"


def http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def http_get_json(url: str, timeout: int = 60) -> dict:
    return json.loads(http_get(url, timeout=timeout).decode("utf-8"))


def http_head_size(url: str, timeout: int = 60) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length is not None else None
    except Exception:
        return None


def _curl_download(url: str, tmp: Path, timeout: int) -> None:
    """Download ``url`` to ``tmp`` using curl (much faster than urllib on
    high-latency links to NCBI; ~25x in practice). Raises on failure."""
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl not found on PATH")
    subprocess.run(
        [curl, "-fSL", "--retry", "0", "--connect-timeout", "30",
         "--max-time", str(timeout), "-A", USER_AGENT, "-o", str(tmp), url],
        check=True, capture_output=True)


def _sha256_and_size(path: Path) -> tuple[str, int]:
    sha = hashlib.sha256()
    nbytes = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
            nbytes += len(chunk)
    return sha.hexdigest(), nbytes


def stream_download(url: str, dest: Path, retries: int = 3, timeout: int = 600) -> dict:
    """Download ``url`` to ``dest`` while computing sha256. Returns file record.

    Retries with exponential backoff. A failed attempt never leaves a partial
    file behind (downloads go to ``dest.part`` first and are renamed).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            _curl_download(url, tmp, timeout)
            sha, nbytes = _sha256_and_size(tmp)
            tmp.replace(dest)
            return {"name": dest.name, "url": url, "bytes": nbytes,
                    "sha256": sha, "downloaded": True}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2 ** attempt)
    return {"name": dest.name, "url": url, "bytes": 0, "sha256": "",
            "downloaded": False, "error": last_error}


def already_downloaded(dest: Path, expected: dict) -> bool:
    """Idempotency check: file exists with matching size and sha256."""
    if not dest.is_file():
        return False
    if expected.get("bytes") is not None and dest.stat().st_size != expected["bytes"]:
        return False
    if expected.get("sha256"):
        sha = hashlib.sha256()
        with dest.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                sha.update(chunk)
        return sha.hexdigest() == expected["sha256"]
    return True


def write_manifest(dataset_dir: Path, manifest: dict) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
