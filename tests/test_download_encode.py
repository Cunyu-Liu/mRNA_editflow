from __future__ import annotations

import json

from scripts.data import download_encode


def test_download_encode_recovers_empty_manifest(tmp_path, monkeypatch):
    dataset = tmp_path / "ENCSR854RUF"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("", encoding="utf-8")
    monkeypatch.setattr(download_encode, "list_related_files", lambda accession: [])

    manifest = download_encode.download_encode("ENCSR854RUF", tmp_path)

    assert manifest["accession"] == "ENCSR854RUF"
    assert manifest["files"] == []
    saved = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    assert saved["provider"] == "ENCODE"


def test_download_encode_prefers_official_cloud_url(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        download_encode,
        "list_related_files",
        lambda accession: [{
            "file_accession": "ENCFFTEST",
            "href": "/files/ENCFFTEST/@@download/ENCFFTEST.fastq.gz",
            "cloud_url": "https://encode-public.s3.amazonaws.com/released/ENCFFTEST.fastq.gz",
            "file_size": 3,
            "md5sum": "abc",
        }],
    )

    def fake_stream(url, dest, retries=3):
        seen.append(url)
        return {"name": dest.name, "url": url, "bytes": 3, "sha256": "sha", "downloaded": True}

    monkeypatch.setattr(download_encode, "stream_download", fake_stream)
    manifest = download_encode.download_encode("ENCSR854RUF", tmp_path, max_bytes=10)

    assert manifest["files"][0]["url"].startswith("https://encode-public.s3.amazonaws.com/")
    assert seen == ["https://encode-public.s3.amazonaws.com/released/ENCFFTEST.fastq.gz"]


def test_download_encode_parallel_preserves_manifest_order(tmp_path, monkeypatch):
    records = [
        {"file_accession": "ENCFF_A", "href": "/a.fastq.gz", "cloud_url": "https://s3/a", "file_size": 3, "md5sum": "a"},
        {"file_accession": "ENCFF_B", "href": "/b.fastq.gz", "cloud_url": "https://s3/b", "file_size": 4, "md5sum": "b"},
    ]
    monkeypatch.setattr(download_encode, "list_related_files", lambda accession: records)

    def fake_stream(url, dest, retries=3):
        return {"name": dest.name, "url": url, "bytes": 3 if url.endswith("/a") else 4, "sha256": "sha", "downloaded": True}

    monkeypatch.setattr(download_encode, "stream_download", fake_stream)
    manifest = download_encode.download_encode("ENCSR854RUF", tmp_path, max_bytes=10, workers=2)

    assert [x["url"] for x in manifest["files"]] == ["https://s3/a", "https://s3/b"]
