"""M3→terminal gates: migration finalization deliverables (§二十).

Verifies that the migration authority layer produced the named terminal artifacts
and that they are internally consistent (manifest hashes match files, crosswalk
rows match inventory, blocker rebind covers PENDING_BLOCKED assets + B3).
"""
import hashlib
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

ART = REPO_ROOT / "artifacts" / "migration"
REP = REPO_ROOT / "reports" / "migration"
EXEC = REPO_ROOT / "docs" / "execution"

NEW_CONTRACT_ID = "mrna_xeditflow_goal_v1_1"
TERMINAL_STATE = "MIGRATION_READY_FOR_DATA_REBUILD"


def _load_jsonl(p: Path):
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def test_final_report_exists_with_terminal_state():
    p = REP / "FINAL_MIGRATION_REPORT.md"
    assert p.exists(), "missing FINAL_MIGRATION_REPORT.md"
    txt = p.read_text(encoding="utf-8")
    assert TERMINAL_STATE in txt
    assert NEW_CONTRACT_ID in txt


def test_crosswalk_jsonl_matches_inventory():
    inv = _load_jsonl(ART / "MIGRATION_INVENTORY.jsonl")
    cross = _load_jsonl(ART / "OLD_TO_NEW_ARTIFACT_CROSSWALK.jsonl")
    assert len(cross) == len(inv), "crosswalk row count != inventory row count"
    inv_paths = {r["path"] for r in inv}
    cross_paths = {r["old_path"] for r in cross}
    assert inv_paths == cross_paths, "crosswalk old_path set != inventory path set"


def test_blocker_rebind_covers_pending_assets_and_b3():
    rows = _load_jsonl(ART / "OLD_BLOCKER_REBIND.jsonl")
    role = yaml.safe_load((EXEC / "xeditflow_asset_role_assignment.yaml").read_text(encoding="utf-8"))
    pending = {a["asset_id"] for a in role["assets"] if a["role"] == "PENDING_BLOCKED"}
    bound_assets = {r["asset_id"] for r in rows}
    assert pending <= bound_assets, f"pending assets missing from rebind: {pending - bound_assets}"
    b3 = [r for r in rows if r.get("blocker_id") == "B3"]
    assert len(b3) == 4, "expected 4 B3 model-engine blocker rows"
    for r in rows:
        assert r["contract_id"] == NEW_CONTRACT_ID


def test_manifest_hashes_match_files():
    m = json.loads((ART / "FINAL_MIGRATION_MANIFEST.json").read_text(encoding="utf-8"))
    assert m["contract_id"] == NEW_CONTRACT_ID
    assert m["terminal_state"] == TERMINAL_STATE
    for rel, expected in m["artifacts"].items():
        p = REPO_ROOT / rel
        assert p.exists(), f"manifest references missing file {rel}"
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert h == expected, f"manifest hash mismatch for {rel}"


def test_sha256sums_match_files():
    sums = (ART / "FINAL_MIGRATION_SHA256SUMS").read_text(encoding="utf-8")
    entries = {}
    for line in sums.strip().splitlines():
        h, rel = line.split("  ", 1)
        entries[rel] = h
    m = json.loads((ART / "FINAL_MIGRATION_MANIFEST.json").read_text(encoding="utf-8"))
    assert set(entries) == set(m["artifacts"]), "sha256sums file set != manifest file set"
    for rel, h in entries.items():
        p = REPO_ROOT / rel
        assert hashlib.sha256(p.read_bytes()).hexdigest() == h


def test_terminal_state_is_effect_model_ready():
    m = json.loads((ART / "FINAL_MIGRATION_MANIFEST.json").read_text(encoding="utf-8"))
    assert m["terminal_state"] == TERMINAL_STATE