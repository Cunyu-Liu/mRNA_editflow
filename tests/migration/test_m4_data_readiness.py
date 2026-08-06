"""M4 data-readiness gate tests.

Verifies the audit output is internally consistent and that the migration
alignment invariants hold (sealed isolation, no cross-region mixing, only
ACCEPTED_FOR_NEW_ROLE assets in pools). Does NOT assert that every benchmark is
data-ready — it asserts the audit correctly reports readiness/gaps.
"""
import collections
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

ART = REPO_ROOT / "artifacts" / "migration"
EXEC = REPO_ROOT / "docs" / "execution"

NEW_CONTRACT_ID = "mrna_xeditflow_goal_v1_1"
ACTIVE_BENCHMARKS = {
    "EditBench-5U-A1-Natural",
    "EditBench-5U-A2-Dense",
    "EditBench-3U-A1-Variant",
}


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def test_readiness_manifest_exists():
    p = ART / "M4_DATA_READINESS.json"
    assert p.exists(), "missing M4_DATA_READINESS.json"
    d = _load(p)
    assert d["contract_id"] == NEW_CONTRACT_ID
    assert set(d["sub_benchmarks"].keys()) == ACTIVE_BENCHMARKS


def test_sealed_gse246381_not_in_any_pool():
    d = _load(ART / "M4_DATA_READINESS.json")
    for b in d["sub_benchmarks"].values():
        accs = {a["asset_id"] for a in b["assets"]}
        assert "GSE246381" not in accs, f"GSE246381 leaked into {b}"
    assert d["alignment_ok"] is True, f"alignment violations: {d['alignment_violations']}"


def test_no_cross_region_mixing():
    d = _load(ART / "M4_DATA_READINESS.json")
    five = {a["asset_id"] for b in d["sub_benchmarks"].values() if b["region"] == "5UTR" for a in b["assets"]}
    three = {a["asset_id"] for b in d["sub_benchmarks"].values() if b["region"] == "3UTR" for a in b["assets"]}
    assert not (five & three), "cross-region pool mixing"


def test_pool_assets_all_accepted():
    d = _load(ART / "M4_DATA_READINESS.json")
    for b in d["sub_benchmarks"].values():
        for a in b["assets"]:
            assert a["pool_role"] == "ACCEPTED_FOR_NEW_ROLE", f"{a['asset_id']} role {a['pool_role']}"


def test_pool_assets_match_benchmark_registry():
    d = _load(ART / "M4_DATA_READINESS.json")
    bench = yaml.safe_load((EXEC / "xeditflow_benchmark_registry.yaml").read_text(encoding="utf-8"))
    by_id = {b["id"]: b for b in bench["sub_benchmarks"]}
    for bid, b in d["sub_benchmarks"].items():
        reg_ids = {a for a in by_id[bid]["asset_ids"]}
        audit_ids = {a["asset_id"] for a in b["assets"]}
        assert reg_ids == audit_ids, f"{bid} asset set mismatch audit vs registry"


def test_readiness_is_honest_not_fabricated():
    """The audit must not claim a benchmark is ready when it has data gaps.
    A benchmark is 'ready' only if every bound asset has D1 pairs."""
    d = _load(ART / "M4_DATA_READINESS.json")
    for bid, b in d["sub_benchmarks"].items():
        n_pairs = sum(1 for a in b["assets"] if a["d1_pairs"] > 0)
        assert b["n_assets_with_d1_pairs"] == n_pairs
        assert b["ready"] == (n_pairs == b["n_assets"]), f"{bid} ready flag inconsistent"


def test_d1_pairs_are_real_not_placeholder():
    d = _load(ART / "M4_DATA_READINESS.json")
    assert d["d1_pairs_total"] > 0, "no D1 pairs scanned"
    # total pairs in audit must equal sum across sub-benchmark assets (no fabricated count)
    total_in_pools = sum(
        a["d1_pairs"] for b in d["sub_benchmarks"].values() for a in b["assets"]
    )
    assert total_in_pools <= d["d1_pairs_total"], "pool pair counts exceed D1 total"