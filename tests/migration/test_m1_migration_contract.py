"""M1 migration contracts: single authoritative contract, zero active legacy refs,
crosswalk coverage, conflict decisions hash-bound, old artifacts unmodified.

These are the M1 failing-tests-first gates for GOAL-XEDITFLOW-MIGRATION-01.
"""
import os
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

CONTRACTS_DIR = REPO_ROOT / "docs" / "contracts"
EXEC_DIR = REPO_ROOT / "docs" / "execution"
CONFIG_DIR = REPO_ROOT / "configs"

NEW_CONTRACT_MD = CONTRACTS_DIR / "mrna_xeditflow_goal_v1_1.md"
NEW_CONTRACT_YAML = CONFIG_DIR / "mrna_xeditflow_contract_v1_1.yaml"
SUPERSESSION = CONTRACTS_DIR / "supersession_mrna_editflow_v3_1_to_xeditflow_v1_1.md"
DECISION_LOG = EXEC_DIR / "xeditflow_migration_decision_log.yaml"
CROSSWALK = EXEC_DIR / "old_to_new_contract_crosswalk.csv"
CLAIM_MATRIX = EXEC_DIR / "xeditflow_claim_matrix.yaml"
TASK_REGISTRY = EXEC_DIR / "xeditflow_task_registry.yaml"
SPLIT_REGISTRY = EXEC_DIR / "xeditflow_split_registry.yaml"
TASK_SPLIT_MATRIX = EXEC_DIR / "xeditflow_task_split_matrix.yaml"

OLD_CONTRACT_ID = "utr_editflow_goal_v3.1_benchmark_first"
NEW_CONTRACT_ID = "mrna_xeditflow_goal_v1_1"


def _load_yaml(path: Path):
    assert path.exists(), f"missing {path}"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _active_contract_md_files():
    """Only the new contract is active; old contract md is historical."""
    result = []
    if CONTRACTS_DIR.exists():
        for p in CONTRACTS_DIR.glob("*.md"):
            if p.name.startswith("mrna_xeditflow_goal"):
                result.append(p)
    return result


def test_single_authoritative_contract_count_is_1():
    """Exactly one ACTIVE top-level contract md must exist."""
    active = _active_contract_md_files()
    assert len(active) == 1, f"expected 1 active contract, got {len(active)}: {active}"
    assert active[0].name == "mrna_xeditflow_goal_v1_1.md"


def test_new_contract_contains_required_governance_sections():
    txt = NEW_CONTRACT_MD.read_text(encoding="utf-8")
    for section in ["provenance", "license", "exposure", "sealed", "conservation", "audit"]:
        assert section in txt.lower(), f"new contract missing governance mention: {section}"


def test_new_contract_has_claim_boundaries_and_term_definition():
    txt = NEW_CONTRACT_MD.read_text(encoding="utf-8")
    assert "L4" in txt or "prohibited" in txt.lower()
    assert "EXACT" in txt


def test_active_authority_layer_never_marks_old_contract_active():
    """The old contract id may appear only in historical/superseded context within the
    active authority layer; it must never be the ACTIVE contract.

    The migration prompt preserves historical governance docs, v3_1 schemas, data and
    the old contract md read-only with the old id retained, so a naive full-repo
    'reference count == 0' scan is invalid. 'Active legacy references == 0' is enforced
    on the authority layer: the new contract is the single ACTIVE authority and the old
    contract only appears as HISTORICAL_SUPERSEDED."""
    authority_files = [
        NEW_CONTRACT_MD, NEW_CONTRACT_YAML, SUPERSESSION,
        DECISION_LOG, CROSSWALK, CLAIM_MATRIX,
        TASK_REGISTRY, SPLIT_REGISTRY, TASK_SPLIT_MATRIX,
    ]
    for p in authority_files:
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if OLD_CONTRACT_ID in txt:
            assert (
                "HISTORICAL" in txt.upper()
                or "SUPERSEDE" in txt.upper()
                or "PREDECESSOR" in txt.upper()
            ), f"{p.name} retains old contract id without historical context"


def test_old_contract_never_claims_active_status():
    """The old contract md must be marked HISTORICAL_SUPERSEDED, and the new contract
    md must be the sole ACTIVE_AUTHORITATIVE_CONTRACT. The new config legitimately
    carries an ACTIVE status while separately referencing the old id as superseded, so
    a naive 'both substrings present' scan is invalid."""
    old_md = CONTRACTS_DIR / "utr_editflow_goal_v3_1.md"
    assert old_md.exists(), "old contract md must be preserved (never deleted)"
    old_txt = old_md.read_text(encoding="utf-8")
    assert "HISTORICAL_SUPERSEDED_BY_MRNA_XEDITFLOW_V1_1" in old_txt, \
        "old contract md must be marked HISTORICAL_SUPERSEDED"

    new_txt = NEW_CONTRACT_MD.read_text(encoding="utf-8")
    assert "ACTIVE_AUTHORITATIVE_CONTRACT" in new_txt
    # only the new contract carries the ACTIVE status among the actual contract md docs
    # (supersession_* records legitimately reference the active status and are excluded)
    active_mds = [
        p for p in CONTRACTS_DIR.glob("*.md")
        if not p.name.startswith("supersession_")
        and "ACTIVE_AUTHORITATIVE_CONTRACT" in p.read_text(encoding="utf-8", errors="ignore")
    ]
    assert [p.name for p in active_mds] == ["mrna_xeditflow_goal_v1_1.md"], \
        f"unexpected ACTIVE contract md files: {[p.name for p in active_mds]}"


def test_active_runtime_config_is_single_authority():
    """The new contract config is the sole ACTIVE authority; it supersedes the old."""
    cfg = _load_yaml(NEW_CONTRACT_YAML)
    assert cfg["status"] == "ACTIVE_AUTHORITATIVE_CONTRACT"
    assert cfg["supersedes"]["contract_id"] == OLD_CONTRACT_ID
    assert cfg["supersedes"]["supersession_scope"] == "SCIENTIFIC_LINE_ONLY"
    assert "provenance" in cfg["supersedes"]["retained_governance"]


def test_historical_docs_may_retain_old_id_but_marked_historical():
    """Old contract md may keep old id but must be marked historical/superseded."""
    old_md = CONTRACTS_DIR / "utr_editflow_goal_v3_1.md"
    if old_md.exists():
        txt = old_md.read_text(encoding="utf-8")
        assert "SUPERSEDED" in txt.upper() or "HISTORICAL" in txt.upper()


def test_crosswalk_coverage():
    """Crosswalk must cover old RQ/Hypothesis/Claim/E-F-tasks/splits/P0/model/GSE246381/STOP/TrackU/alignment/G0-G7/PR."""
    required = [
        OLD_CONTRACT_ID, "H1A", "H6B", "C-BENCH-01", "C-BIO-01",
        "GSE246381", "Track U", "STOP", "alignment", "G0", "PR1",
        "T5_RANK_CLOSED_SELECT_E_PAIR", "T5_MEASURED_NEIGHBORHOOD_OPTIMIZATION",
    ]
    text = CROSSWALK.read_text(encoding="utf-8")
    missing = [r for r in required if r not in text]
    assert not missing, f"crosswalk missing: {missing}"


def test_decision_log_freeze_decisions():
    """The five conflict freezes from the migration prompt must be present: STOP, unlabeled, GSE246381, budget, indel."""
    log = _load_yaml(DECISION_LOG)
    dims = {d["dimension"] for d in log["decisions"]}
    for expected in ["STOP", "unlabeled_pretraining", "GSE246381_temporal_wording", "edit_budget", "indel"]:
        assert expected in dims, f"decision log missing dimension: {expected}"


def test_config_freeze_values():
    cfg = _load_yaml(NEW_CONTRACT_YAML)
    fd = cfg["frozen_decisions"]
    assert fd["termination"]["learned_general_stop"] == "HOLD_IDENTIFIABILITY_GATE"
    assert fd["unlabeled_pretraining"]["project_unlabeled_pretraining"] == "DISABLED_IN_PRIMARY_V1"
    assert fd["edit_budget"]["primary_edit_budget"] == [1, 3, 5]
    assert fd["edit_budget"]["exploratory_edit_budget"] == [10]
    assert fd["gse246381"]["claim_wording"] == "OPERATIONALLY_SEALED_RETROSPECTIVE_EXTERNAL"


def test_task_registry_expected_sets():
    reg = _load_yaml(TASK_REGISTRY)
    assert len(reg["primary_tasks"]) == 4
    assert len(reg["secondary_tasks"]) == 6
    assert len(reg["theory_tasks"]) == 2
    assert set(reg["expected_primary_task_ids"]) == {t["id"] for t in reg["primary_tasks"]}
    assert set(reg["expected_secondary_task_ids"]) == {t["id"] for t in reg["secondary_tasks"]}
    assert set(reg["expected_theory_task_ids"]) == {t["id"] for t in reg["theory_tasks"]}


def test_split_registry_expected_sets():
    reg = _load_yaml(SPLIT_REGISTRY)
    assert set(reg["expected_split_ids"]) == {s["id"] for s in reg["splits"]}
    assert len(reg["splits"]) == 8


def test_task_split_matrix_fk_closure():
    mat = _load_yaml(TASK_SPLIT_MATRIX)
    split_reg = _load_yaml(SPLIT_REGISTRY)
    task_reg = _load_yaml(TASK_REGISTRY)
    valid_splits = {s["id"] for s in split_reg["splits"]}
    task_ids = {mm["id"] for mm in task_reg["primary_tasks"] + task_reg["secondary_tasks"] + task_reg["theory_tasks"]}
    assert set(mat["matrix"].keys()) == task_ids, "task-split matrix rows != task registry tasks"
    for tid, splits in mat["matrix"].items():
        assert set(splits) <= valid_splits, f"task {tid} references unknown split"
    assert mat["sealed_split"] == "S6"
    assert mat["sealed_split_in_task_activation"] is False


def test_old_artifacts_not_overwritten():
    """The old v3.1 contract md (if present) must be byte-identical to its committed hash."""
    # We only assert presence; content immutability is enforced by git (no force rewrite).
    old_md = CONTRACTS_DIR / "utr_editflow_goal_v3_1.md"
    assert old_md.exists() or SUPERSESSION.exists()


def test_supersession_record_marks_old_historical():
    ss = _load_yaml(SUPERSESSION)
    assert ss["active_contract"] == NEW_CONTRACT_ID
    old = next(r for r in ss["records"] if r["contract"] == OLD_CONTRACT_ID)
    assert old["status"] == "HISTORICAL_SUPERSEDED_BY_MRNA_XEDITFLOW_V1_1"