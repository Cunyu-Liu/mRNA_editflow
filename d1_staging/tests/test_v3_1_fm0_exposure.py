"""FM0-A (v3.1) tests: candidates, exposure ledger, overlap, aggregate/commitment.

Run on server:
    cd /home/cunyuliu/mrna_editflow_goal/worktrees/v3_1_data_bench_closure_20260803
    /home/cunyuliu/miniconda3/envs/pc_cng/bin/python -m pytest \
        d1_staging/tests/test_v3_1_fm0_exposure.py -v
"""

import json
import os
import sys
from pathlib import Path

import pytest

FM0_SCRIPTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "fm0")
sys.path.insert(0, FM0_SCRIPTS)

from fm0_build_v3_1_fm0_exposure import (  # noqa: E402
    AUDIT_RUN,
    BACKBONE_OVERLAP,
    EXTERNAL_BACKBONE,
    build_candidates,
    build_clusters,
    build_ledger,
    build_gse246381_aggregate,
    build_gse246381_commitment,
    build_restricted_aggregate,
)


@pytest.fixture
def workspace(tmp_path):
    """A tiny synthetic D1 workspace with 2 ordinary accessions + cluster."""
    seq = tmp_path / "seq.jsonl"
    seq.write_text(
        "\n".join([
            json.dumps({"sequence_id": "GSE114002_A__src", "region_scope": "5UTR"}),
            json.dumps({"sequence_id": "GSE114002_A__cand", "region_scope": "5UTR"}),
            json.dumps({"sequence_id": "GSE200304_B__src", "region_scope": "3UTR"}),
            json.dumps({"sequence_id": "GSE200304_B__cand", "region_scope": "3UTR"}),
        ]) + "\n",
        encoding="utf-8",
    )
    pair = tmp_path / "pairs.jsonl"
    pair.write_text(
        "\n".join([
            json.dumps({"pair_id": "gse114002_pair_0"}),
            json.dumps({"pair_id": "gse200304_pair_0"}),
        ]) + "\n",
        encoding="utf-8",
    )
    obs = tmp_path / "obs.jsonl"
    obs.write_text(
        "\n".join([
            json.dumps({"observation_id": "gse114002_obs_0"}),
            json.dumps({"observation_id": "gse114002_obs_1"}),
            json.dumps({"observation_id": "gse200304_obs_0"}),
        ]) + "\n",
        encoding="utf-8",
    )
    return seq, pair, obs


def test_candidates_policy(workspace):
    seq, pair, obs = workspace
    clusters = build_clusters(seq, pair, obs)
    cand = build_candidates(clusters)
    assert cand["policy_ok"] is True
    kinds = [c["kind"] for c in cand["candidates"]]
    assert kinds == ["from_scratch_E", "supervised_F_to_E", "general_backbone"]
    # final alias points to eligible set
    assert cand["final_alias"]["alias"] == "from_scratch_E"
    assert "from_scratch_E" in cand["final_alias"]["eligible_set"]


def test_external_backbone_overlap_detected(workspace):
    seq, pair, obs = workspace
    clusters = build_clusters(seq, pair, obs)
    cand = build_candidates(clusters)
    backbone = next(c for c in cand["candidates"] if c["kind"] == "general_backbone")
    # In synthetic workspace only GSE114002/GSE200304 present; FIXED overlap map
    # marks GSE114002 DETECTED -> backbone not eligible
    assert backbone["eligible"] is False
    assert "GSE114002" in backbone["overlap_detected_accessions"]


def test_ledger_unique_keys_and_eligibility(workspace):
    seq, pair, obs = workspace
    clusters = build_clusters(seq, pair, obs)
    ledger = build_ledger(clusters)
    keys = [r["ledger_key"] for r in ledger]
    assert len(keys) == len(set(keys)), "ledger keys must be unique"

    # internal candidates always eligible+clean
    for r in ledger:
        if r["checkpoint_id"] in ("from_scratch_E", "supervised_F_to_E"):
            assert r["overlap_status"] == "CLEAN"
            assert r["eligible"] is True
        if r["checkpoint_id"] == EXTERNAL_BACKBONE["candidate_id"]:
            assert r["weight_sha256" if False else "weights_sha256"] == EXTERNAL_BACKBONE["weights_sha256"]
            # consistency: DETECTED -> not eligible
            if r["overlap_status"] == "DETECTED":
                assert r["eligible"] is False


def test_no_gse246381_in_ordinary_ledger(workspace):
    seq, pair, obs = workspace
    clusters = build_clusters(seq, pair, obs)
    # GSE246381 excluded from synthetic clusters; ensure builder never fabricates it
    assert "GSE246381" not in clusters
    ledger = build_ledger(clusters)
    assert not any(r["cluster_id"] == "GSE246381" for r in ledger)


def test_aggregate_commitment_no_member_data():
    agg = build_gse246381_aggregate([])
    assert agg["member_data_emitted_to_ordinary"] is False
    assert all(v == 0 for v in agg["analytic_or_final_counters"].values())
    r_agg = build_restricted_aggregate()
    assert all(v == 0 for v in r_agg["analytic_or_final_counters"].values())
    commit = build_gse246381_commitment(agg, r_agg)
    assert commit["member_data_emitted_to_ordinary"] is False
    assert commit["aggregate_sha256"]


def test_audit_run_constant():
    assert AUDIT_RUN == "audit_fm0a_v1"
    assert "GSE114002" in BACKBONE_OVERLAP
    assert BACKBONE_OVERLAP["GSE114002"]["overlap"] == "DETECTED"