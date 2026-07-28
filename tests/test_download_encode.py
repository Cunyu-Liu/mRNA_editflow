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
