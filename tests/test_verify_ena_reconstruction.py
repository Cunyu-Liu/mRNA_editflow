from __future__ import annotations

import gzip
import hashlib
import json

from scripts.data.verify_downloads import verify_ena_reconstruction


def test_verify_ena_reconstruction_accepts_62_verified_gzip_files(tmp_path):
    destination = tmp_path / "reconstructed"
    destination.mkdir()
    records = []
    for index in range(62):
        payload_path = destination / f"ENCFF{index:06d}.fastq.gz"
        with gzip.open(payload_path, "wb") as handle:
            handle.write(f"@read-{index}\nA\n+\nI\n".encode())
        digest = hashlib.sha256(payload_path.read_bytes()).hexdigest()
        records.append({
            "encode_accession": f"ENCFF{index:06d}",
            "bytes": payload_path.stat().st_size,
            "path": str(payload_path),
            "status": "VERIFIED",
            "sha256": digest,
        })
    manifest = tmp_path / "ena_manifest.json"
    manifest.write_text(json.dumps({
        "expected_files": 62,
        "destination": str(destination),
        "files": records,
    }), encoding="utf-8")

    result = verify_ena_reconstruction(manifest)

    assert result["ok"] is True
    assert result["n_files_ok"] == 62
    assert result["n_files_failed"] == 0
