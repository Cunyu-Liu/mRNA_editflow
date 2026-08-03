"""Tests for the v3.1 D1-R technical canonical builder and validator."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts" / "d1"
BUILD = SCRIPT_DIR / "build_v3_1_technical_canonical.py"
VALIDATE = SCRIPT_DIR / "validate_v3_1_technical_canonical.py"

# A tiny canonical sample: one GSE114002 paired record + one GSE246381 record.
SAMPLE = [
    {
        "record_id": "GSE114002_t1",
        "accession": "GSE114002",
        "dataset": "sample2019",
        "region": "5'UTR",
        "source_sequence": "ACGTACGTACGT",
        "candidate_sequence": "ACGTATGTACGT",
        "edit_script": [{"op": "SUB", "pos": 5, "token": "T"}],
        "edit_script_verified": True,
        "labels": {"rl": 5.33},
        "metadata": {"record_type": "paired"},
    },
    {
        "record_id": "GSE246381_t1",
        "accession": "GSE246381",
        "dataset": "gse246381",
        "region": "5'UTR",
        "source_sequence": "GGGGCCCCTTTT",
        "candidate_sequence": "GGGGCCCCTTTG",
        "edit_script": [{"op": "SUB", "pos": 11, "token": "G"}],
        "edit_script_verified": True,
        "labels": {"hek_log2fc_umi": 0.5},
        "metadata": {},
    },
]


@pytest.fixture
def build_workspace(tmp_path):
    canonical = tmp_path / "canonical.jsonl"
    with open(canonical, "w", encoding="utf-8") as fh:
        for rec in SAMPLE:
            fh.write(json.dumps(rec) + "\n")
    out = tmp_path / "out"
    restricted = tmp_path / "restricted"
    subprocess.run(
        [sys.executable, str(BUILD), "--canonical", str(canonical),
         "--out", str(out), "--restricted-out", str(restricted)],
        check=True, capture_output=True, text=True,
    )
    return canonical, out, restricted


def _read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def test_sequences_emitted(build_workspace):
    _, out, _ = build_workspace
    seqs = _read_jsonl(out / "sequence_entities.jsonl")
    # 2 records x 2 sides = 4 sequences (GSE246381 is restricted, so only 2 in ordinary)
    assert len(seqs) == 2
    for s in seqs:
        assert s["sequence_id"]
        assert s["raw_sequence_sha256"]
        assert s["original_length"] == 12


def test_pairs_and_relations(build_workspace):
    _, out, _ = build_workspace
    pairs = _read_jsonl(out / "utr_edit_pairs.jsonl")
    rels = _read_jsonl(out / "utr_edit_relation_candidates.jsonl")
    assert len(pairs) == 1
    assert len(rels) == 1
    assert pairs[0]["scientific_track"] == "E"
    assert pairs[0]["immutable_base_future_use_role"] == "AWAITING_B0_GLOBAL_DISPOSITION"
    assert rels[0]["lifecycle_status"] == "ACCEPTED"


def test_observations_emitted(build_workspace):
    _, out, _ = build_workspace
    obs = _read_jsonl(out / "functional_observations.jsonl")
    assert len(obs) == 1  # only GSE114002 rl label in ordinary
    assert obs[0]["value"] == 5.33


def test_restricted_mirror_isolated(build_workspace):
    _, _, restricted = build_workspace
    rdir = restricted / "sealed_external" / "GSE246381"
    rseq = _read_jsonl(rdir / "sequence_entities.jsonl")
    assert len(rseq) == 2
    access = _read_jsonl(rdir / "ACCESS_LOG.jsonl")
    assert len(access) == 1
    assert access[0]["status"] == "COMPLETION"
    # chain links
    for ev in access:
        assert "event_sha256" in ev


def test_use_roles_and_exposure(build_workspace):
    _, out, _ = build_workspace
    roles = _read_jsonl(out / "use_roles.jsonl")
    assert len(roles) == 2
    for r in roles:
        assert r["authority_level"] == "ORDINARY"
    eff = _read_jsonl(out / "effective_exposure_projection.jsonl")
    assert len(eff) == 2
    assert all(e["effective_exposure"] == "AWAITING_B0_GLOBAL_DISPOSITION" for e in eff)


def test_validator_passes(build_workspace):
    _, out, restricted = build_workspace
    res = subprocess.run(
        [sys.executable, str(VALIDATE), "--dir", str(out), "--restricted-dir", str(restricted)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert '"status": "PASS"' in res.stdout


def test_validator_rejects_gse246381_leak(build_workspace, tmp_path):
    _, out, restricted = build_workspace
    # inject a leak into the ordinary sequence file
    leak_path = out / "sequence_entities.jsonl"
    with open(leak_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"sequence_id": "gse246381_leak", "sequence_scope": "5UTR",
                             "raw_sequence_sha256": "x" * 64, "normalized_sequence_sha256": "x" * 64,
                             "full_sequence_sha256": "x" * 64, "original_length": 1,
                             "region_scope": "5UTR"}) + "\n")
    res = subprocess.run(
        [sys.executable, str(VALIDATE), "--dir", str(out), "--restricted-dir", str(restricted)],
        capture_output=True, text=True,
    )
    assert res.returncode == 1
    assert "gse246381_leak" in res.stdout