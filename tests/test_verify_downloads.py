from __future__ import annotations

import hashlib
import json

from scripts.data.verify_downloads import verify_root


def _record(name: str, payload: bytes) -> dict:
    return {
        "name": name,
        "downloaded": True,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_verify_root_accepts_valid_file(tmp_path):
    dataset = tmp_path / "GSE123"
    dataset.mkdir()
    payload = b"not html\n"
    (dataset / "counts.txt").write_bytes(payload)
    (dataset / "manifest.json").write_text(
        json.dumps({"accession": "GSE123", "files": [_record("counts.txt", payload)]}),
        encoding="utf-8",
    )

    report = verify_root(tmp_path)

    assert report["n_files_ok"] == 1
    assert report["n_files_failed"] == 0


def test_verify_root_reports_unreadable_manifest(tmp_path):
    dataset = tmp_path / "ENCSR854RUF"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("", encoding="utf-8")

    report = verify_root(tmp_path)

    assert report["n_files_failed"] == 1
    assert "manifest unreadable" in report["datasets"][0]["failed"][0]


def test_verify_root_reports_partial_download(tmp_path):
    dataset = tmp_path / "GSE123"
    dataset.mkdir()
    (dataset / "counts.txt.part").write_bytes(b"partial")
    (dataset / "manifest.json").write_text(
        json.dumps({"accession": "GSE123", "files": []}), encoding="utf-8"
    )

    report = verify_root(tmp_path)

    assert report["n_files_failed"] == 1
    assert "partial file remains" in report["datasets"][0]["failed"][0]


def test_verify_root_rejects_html_payload(tmp_path):
    dataset = tmp_path / "GSE123"
    dataset.mkdir()
    payload = b"<!DOCTYPE html><html>error</html>"
    (dataset / "counts.txt").write_bytes(payload)
    (dataset / "manifest.json").write_text(
        json.dumps({"accession": "GSE123", "files": [_record("counts.txt", payload)]}),
        encoding="utf-8",
    )

    report = verify_root(tmp_path)

    assert report["n_files_failed"] == 1
    assert "HTML error page" in report["datasets"][0]["failed"][0]
