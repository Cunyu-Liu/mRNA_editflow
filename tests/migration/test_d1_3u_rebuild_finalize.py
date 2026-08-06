"""Tests for the D1 3U-A1 rebuild finalizer (scripts/d1_3u_rebuild_finalize.py).

Exercises the manifest + SHA256SUMS generation on a small temp dir so the test
does not require the multi-GB /mnt staging artifacts.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "d1_3u_rebuild_finalize.py"

ARTIFACT_KEYS = [
    "effective_exposure_projection.jsonl",
    "endpoint_registry.jsonl",
    "exposure_records.jsonl",
    "functional_observation_candidates.jsonl",
    "functional_observations.jsonl",
    "group_assignments.jsonl",
    "group_registry.jsonl",
    "object_attributes.jsonl",
    "rejection_records.jsonl",
    "sequence_entities.jsonl",
    "transformation_edges.jsonl",
    "use_roles.jsonl",
    "utr_edit_pairs.jsonl",
    "utr_edit_relation_candidates.jsonl",
]


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_finalizer_generates_manifest_and_sums(tmp_path):
    d = tmp_path / "ordinary"
    d.mkdir()
    content = {k: f"line for {k}\n".encode() for k in ARTIFACT_KEYS}
    for k, v in content.items():
        (d / k).write_bytes(v)

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--dir", str(d), "--config-hash", "v3.1-D1", "--phase", "D1-3U-REBUILD"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr

    sums_file = d / "D1_SHA256SUMS"
    manifest_file = d / "D1_CANONICAL_MANIFEST.json"
    assert sums_file.exists()
    assert manifest_file.exists()

    lines = {}
    for line in sums_file.read_text().splitlines():
        if not line:
            continue
        h, fn = line.split("  ", 1)
        lines[fn] = h
    assert set(lines) == set(ARTIFACT_KEYS)
    for k, v in content.items():
        assert lines[k] == _sha256(v), f"mismatch sha256 for {k}"

    m = json.loads(manifest_file.read_text())
    assert m["phase"] == "D1-3U-REBUILD"
    assert m["config_hash"] == "v3.1-D1"
    assert m["status"] == "GENERATED"
    assert m["artifact_files"][k] == len(v)
