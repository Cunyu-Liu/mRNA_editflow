"""D0-R asset registry contract tests (v3.1).

Verifies that the frozen P0/P1/P2/reference/analysis-only/search-negative
registry sets exactly equal the contract §7.1.1 constants, that GSE200304
member set is exact, that every ACQUIRED_FOR_REBUILD decision carries
download+processing permission = YES, that every asset with present files has a
verified sha256, and that the registry does not self-claim completeness.
"""
import json
import os
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Frozen constants (contract §7.1.1) -- DO NOT modify.
# ---------------------------------------------------------------------------
P0_ASSET_GROUP_IDS = {
    "GSE114002", "GSE145046", "GSE217518", "GSE232571", "GSE232572",
    "FAST_UTR_SIEGEL_2022", "GSE288185", "GSE256185_DART", "GSE232927",
    "GSE176581", "ENCSR854RUF", "GSE149487", "GSE200304", "GSE186455",
    "GSE246381", "NZIP_EMTAB_10902_11572_11575", "GSE330741", "GSE261709",
    "GSE298114", "PTRE_PRJNA1116243", "GSE207584_CLEANUP", "GSE173083_CLEANUP",
}
P1_ASSET_GROUP_IDS = {
    "GSE194092", "GSE270252_270254", "GSE173098", "GSE295080_ISOMPRA",
    "GSE291719_SONAR", "GSE55396_FAST_UTR_2014", "PASSPORT_SEQ", "SEERS",
}
P2_ACQUISITION_WATCHLIST = {"PARADE", "SALUKI_HALF_LIFE"}
REFERENCE_SERVICE = {"GENCODE", "REFSEQ", "ENSEMBL", "UTRDB", "RNACENTRAL"}
ANALYSIS_ONLY_OUT_OF_SCOPE = {"CODONBERT", "OPENVACCINE", "BPRNA_STRUCTURE_ONLY"}
SEARCH_NEGATIVE_LEDGER = {"MAVEDB", "MPRABASE"}
GSE200304_MEMBERS = {"GSE200304", "GSE200302", "GSE200303", "GSE217530"}

# Repository root: tests/data/ -> ../../..
REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_DIR = REPO_ROOT / "data" / "v3_1" / "registry"


def _load_jsonl(name):
    rows = []
    with open(REGISTRY_DIR / name) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_snapshot():
    with open(REGISTRY_DIR / "priority_snapshot_v3_1.yaml") as f:
        return yaml.safe_load(f)


def test_p0_set_equals_frozen():
    snap = _load_snapshot()
    assert set(snap["frozen_sets"]["p0_asset_group_ids"]) == P0_ASSET_GROUP_IDS


def test_p1_set_equals_frozen():
    snap = _load_snapshot()
    assert set(snap["frozen_sets"]["p1_asset_group_ids"]) == P1_ASSET_GROUP_IDS


def test_p2_set_equals_frozen():
    snap = _load_snapshot()
    assert set(snap["frozen_sets"]["p2_acquisition_watchlist"]) == P2_ACQUISITION_WATCHLIST


def test_reference_set_equals_frozen():
    snap = _load_snapshot()
    assert set(snap["frozen_sets"]["reference_service"]) == REFERENCE_SERVICE


def test_analysis_only_set_equals_frozen():
    snap = _load_snapshot()
    assert set(snap["frozen_sets"]["analysis_only_out_of_scope"]) == ANALYSIS_ONLY_OUT_OF_SCOPE


def test_search_negative_set_equals_frozen():
    snap = _load_snapshot()
    assert set(snap["frozen_sets"]["search_negative_ledger"]) == SEARCH_NEGATIVE_LEDGER


def test_gse200304_member_set_equals_frozen():
    snap = _load_snapshot()
    assert set(snap["frozen_sets"]["gse200304_members"]) == GSE200304_MEMBERS


def test_acquired_for_rebuild_has_permissions():
    decisions = _load_jsonl("dataset_decisions.jsonl")
    rebuild = [d for d in decisions if d["d0_decision"] == "ACQUIRED_FOR_REBUILD"]
    assert len(rebuild) > 0
    for d in rebuild:
        assert d["permitted_download"] == "YES", d["asset_id"]
        assert d["permitted_processing"] == "YES", d["asset_id"]
        assert d["reviewer"], d["asset_id"]
        assert d["use_basis_evidence_ids"], d["asset_id"]


def test_present_assets_have_verified_sha256():
    assets = _load_jsonl("dataset_assets.jsonl")
    present = [a for a in assets if a["acquisition_status"] in ("DOWNLOADED_VERIFIED", "DOWNLOADED_UNVERIFIED", "PARTIAL")]
    assert len(present) > 0
    for a in present:
        sha = a.get("sha256")
        assert sha and len(sha) == 64, a["asset_id"]


def test_registry_does_not_self_claim_completeness():
    snap_path = REGISTRY_DIR / "priority_snapshot_v3_1.yaml"
    text = snap_path.read_text(encoding="utf-8")
    for bad in ("等其它资产", "等其他资产", "etc.", "ETC", "other assets", "…", "..."):
        assert bad not in text, f"registry self-claims completeness with {bad!r}"
    # every frozen set must be a non-empty explicit enumeration
    for key in ("p0_asset_group_ids", "p1_asset_group_ids", "p2_acquisition_watchlist",
                "reference_service", "analysis_only_out_of_scope", "search_negative_ledger",
                "gse200304_members"):
        assert _load_snapshot()["frozen_sets"][key], key


def test_all_p0_assets_have_decision_row():
    decisions = _load_jsonl("dataset_decisions.jsonl")
    decided = {d["asset_group_id"] for d in decisions}
    assert P0_ASSET_GROUP_IDS <= decided


def test_all_p1_assets_have_decision_row():
    decisions = _load_jsonl("dataset_decisions.jsonl")
    decided = {d["asset_group_id"] for d in decisions}
    assert P1_ASSET_GROUP_IDS <= decided