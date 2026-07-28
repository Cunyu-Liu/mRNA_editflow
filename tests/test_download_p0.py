"""Unit tests for D0-03 download scripts (network-free; all HTTP mocked)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))

from scripts.data import download_geo, verify_downloads  # noqa: E402
import download_common  # noqa: E402


FILELIST = """#Archive/File\tName\tTime\tSize\tType
Archive\tGSE999999_RAW.tar\t06/02/2020 17:19:32\t130426880\tTAR
File\tGSM1_counts.txt.gz\t02/09/2020 16:19:12\t4369006\tTXT
File\tGSM2_counts.txt.gz\t02/09/2020 16:19:14\t4434205\tTXT
"""

DIR_INDEX = """
<html><body>
<a href="/geo/series/GSE999nnn/GSE999999/">Parent</a>
<a href="GSE999999_hek_combined_umi_counts.csv.gz">f</a>
<a href="GSE999999_vglut_combined_umi_counts.csv.gz">f</a>
<a href="https://www.hhs.gov/vulnerability-disclosure-policy/index.html">x</a>
</body></html>
"""


def test_series_prefix():
    assert download_geo.series_prefix("GSE145046") == "GSE145nnn"
    assert download_geo.series_prefix("GSE246381") == "GSE246nnn"


def test_file_url_routes_sample_files_to_sample_pages():
    url = download_geo.file_url("GSE145046", "GSM4305122_1_read_count_x.txt.gz")
    assert url == ("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4305nnn/"
                   "GSM4305122/suppl/GSM4305122_1_read_count_x.txt.gz")
    series_url = download_geo.file_url("GSE246381", "GSE246381_hek_combined_umi_counts.csv.gz")
    assert series_url == ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE246nnn/"
                          "GSE246381/suppl/GSE246381_hek_combined_umi_counts.csv.gz")


def test_parse_filelist_rows():
    files = download_geo.parse_filelist(FILELIST)
    assert files == [
        {"name": "GSE999999_RAW.tar", "size": 130426880, "kind": "archive"},
        {"name": "GSM1_counts.txt.gz", "size": 4369006, "kind": "file"},
        {"name": "GSM2_counts.txt.gz", "size": 4434205, "kind": "file"},
    ]


def test_parse_dir_index_filters_dirs_and_external_links():
    files = download_geo.parse_dir_index(DIR_INDEX)
    names = [f["name"] for f in files]
    assert names == [
        "GSE999999_hek_combined_umi_counts.csv.gz",
        "GSE999999_vglut_combined_umi_counts.csv.gz",
    ]


def test_download_geo_skips_raw_tar_and_writes_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(download_geo, "list_supplementary",
                        lambda acc: download_geo.parse_filelist(FILELIST))
    payload = b"col1\tcol2\n1\t2\n"
    calls = []

    def fake_stream(url, dest, retries=3):
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        return {"name": dest.name, "url": url, "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(), "downloaded": True}

    monkeypatch.setattr(download_geo, "stream_download", fake_stream)
    manifest = download_geo.download_geo("GSE999999", tmp_path)
    assert len(manifest["files"]) == 2
    assert manifest["skipped"] == [
        {"name": "GSE999999_RAW.tar", "size": 130426880,
         "reason": "RAW.tar archive duplicates per-sample files"}
    ]
    assert not any("RAW.tar" in u for u in calls)
    on_disk = tmp_path / "GSE999999" / "manifest.json"
    assert on_disk.is_file()
    saved = json.loads(on_disk.read_text())
    assert saved["accession"] == "GSE999999"


def test_download_geo_recovers_size_matched_existing_file(tmp_path, monkeypatch):
    payload = b"already here"
    monkeypatch.setattr(
        download_geo, "list_supplementary",
        lambda acc: [{"name": "existing.txt", "size": len(payload), "kind": "file"}],
    )
    dataset = tmp_path / "GSE999999"
    dataset.mkdir()
    (dataset / "existing.txt").write_bytes(payload)

    def should_not_download(*args, **kwargs):
        raise AssertionError("size-matched existing file was unnecessarily downloaded")

    monkeypatch.setattr(download_geo, "stream_download", should_not_download)
    manifest = download_geo.download_geo("GSE999999", tmp_path)

    record = manifest["files"][0]
    assert record["downloaded"] is True
    assert record["recovered_existing"] is True
    assert record["bytes"] == len(payload)
    assert record["sha256"] == hashlib.sha256(payload).hexdigest()


def test_download_geo_flags_size_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(download_geo, "list_supplementary",
                        lambda acc: download_geo.parse_filelist(FILELIST))

    def fake_stream(url, dest, retries=3):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"tiny")
        return {"name": dest.name, "url": url, "bytes": 4,
                "sha256": hashlib.sha256(b"tiny").hexdigest(), "downloaded": True}

    monkeypatch.setattr(download_geo, "stream_download", fake_stream)
    manifest = download_geo.download_geo("GSE999999", tmp_path)
    assert all(not f["downloaded"] for f in manifest["files"])
    assert all("size mismatch" in f["error"] for f in manifest["files"])


def test_verify_detects_html_and_corruption(tmp_path):
    d = tmp_path / "GSE1"
    d.mkdir()
    good = b"real data payload"
    (d / "good.txt").write_bytes(good)
    (d / "bad.html").write_bytes(b"<!DOCTYPE html><title>Object not found!</title>")
    manifest = {
        "provider": "GEO", "accession": "GSE1", "files": [
            {"name": "good.txt", "bytes": len(good),
             "sha256": hashlib.sha256(good).hexdigest(), "downloaded": True},
            {"name": "bad.html", "bytes": 48,
             "sha256": hashlib.sha256(b"x" * 48).hexdigest(), "downloaded": True},
            {"name": "gone.txt", "bytes": 3, "sha256": "0" * 64, "downloaded": True},
            {"name": "defer.fastq.gz", "downloaded": False,
             "defer_reason": "cap", "bytes": 10},
        ], "skipped": []}
    (d / "manifest.json").write_text(json.dumps(manifest))
    report = verify_downloads.verify_root(tmp_path)
    assert report["n_files_ok"] == 1
    failed_text = "\n".join(report["datasets"][0]["failed"])
    assert "bad.html" in failed_text
    assert "gone.txt" in failed_text
    assert report["n_deferred"] == 1


def test_verify_verdict_md(tmp_path):
    d = tmp_path / "GSE1"
    d.mkdir()
    payload = b"abc"
    (d / "f.txt").write_bytes(payload)
    (d / "manifest.json").write_text(json.dumps({
        "provider": "GEO", "accession": "GSE1",
        "files": [{"name": "f.txt", "bytes": 3,
                   "sha256": hashlib.sha256(payload).hexdigest(),
                   "downloaded": True}], "skipped": []}))
    from scripts.data.verify_downloads import main
    rc = main(["--root", str(tmp_path), "--report-md", str(tmp_path / "r.md")])
    assert rc == 0
    assert "PASS" in (tmp_path / "r.md").read_text()


def test_stream_download_retry_then_success(tmp_path, monkeypatch):
    attempts = {"n": 0}

    def fake_curl(url, tmp, timeout):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise OSError("transient")
        tmp.write_bytes(b"payload")

    monkeypatch.setattr(download_common, "_curl_download", fake_curl)
    monkeypatch.setattr(download_common.time, "sleep", lambda s: None)
    rec = download_common.stream_download("https://x/f", tmp_path / "f.bin")
    assert rec["downloaded"] and rec["bytes"] == 7
    assert rec["sha256"] == hashlib.sha256(b"payload").hexdigest()
    assert attempts["n"] == 2


def test_stream_download_retains_partial_for_resume(tmp_path, monkeypatch):
    attempts = {"n": 0}
    dest = tmp_path / "f.bin"

    def fake_curl(url, tmp, timeout):
        attempts["n"] += 1
        if attempts["n"] == 1:
            tmp.write_bytes(b"partial")
            raise OSError("connection reset")
        assert tmp.read_bytes() == b"partial"
        tmp.write_bytes(b"payload")

    monkeypatch.setattr(download_common, "_curl_download", fake_curl)
    monkeypatch.setattr(download_common.time, "sleep", lambda s: None)
    rec = download_common.stream_download("https://x/f", dest, retries=2)

    assert rec["downloaded"] is True
    assert dest.read_bytes() == b"payload"
    assert not dest.with_suffix(dest.suffix + ".part").exists()
