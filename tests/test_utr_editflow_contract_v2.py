"""Contract tests for utr_editflow_contract_v2 (Phase C0-04 acceptance).

Run:
    pytest tests/test_utr_editflow_contract_v2.py -q

These tests validate that:
1. The v2 contract file family exists and is internally consistent.
2. The v2 contract boundaries are not violated by the active tree.
3. No active code references the superseded v1 contract as a constraint source.
4. The active task registry points to v2.
5. README is aligned with v2.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

CONTRACT_PATH = ROOT / "configs" / "utr_editflow_contract_v2.yaml"
SQ_PATH = ROOT / "docs" / "utr_editflow_scientific_question_v2.md"
CLAIM_PATH = ROOT / "docs" / "utr_editflow_claim_matrix_v2.md"
REGISTRY_PATH = ROOT / "docs" / "execution" / "task_registry_v2.yaml"
DECISION_LOG_PATH = ROOT / "docs" / "decision_log.md"
CONFLICT_MATRIX_PATH = ROOT / "docs" / "contracts" / "v2_contract_conflict_matrix.md"
README_PATH = ROOT / "README.md"

# Archived v1 contract locations (these MUST exist after Phase C0 archival)
V1_ARCHIVE_DIR = ROOT / "archive" / "legacy_predictor_first_v1"
V1_CONTRACTS_ARCHIVE = V1_ARCHIVE_DIR / "contracts_v1"
V1_SUPERSEDED_MARKER = V1_ARCHIVE_DIR / "SUPERSEDED.md"


# ---------------------------------------------------------------------------
# 1. v2 contract file family exists
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def contract():
    return yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def registry():
    return yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_v2_contract_file_exists():
    assert CONTRACT_PATH.exists(), f"Missing: {CONTRACT_PATH}"


def test_v2_scientific_question_file_exists():
    assert SQ_PATH.exists(), f"Missing: {SQ_PATH}"


def test_v2_claim_matrix_file_exists():
    assert CLAIM_PATH.exists(), f"Missing: {CLAIM_PATH}"


def test_v2_task_registry_file_exists():
    assert REGISTRY_PATH.exists(), f"Missing: {REGISTRY_PATH}"


def test_decision_log_exists():
    assert DECISION_LOG_PATH.exists(), f"Missing: {DECISION_LOG_PATH}"


def test_conflict_matrix_exists():
    assert CONFLICT_MATRIX_PATH.exists(), f"Missing: {CONFLICT_MATRIX_PATH}"


# ---------------------------------------------------------------------------
# 2. v2 contract internal consistency
# ---------------------------------------------------------------------------

def test_contract_id_is_v2(contract):
    assert contract["contract_id"] == "utr_editflow_contract_v2"


def test_contract_status_is_frozen(contract):
    assert contract["status"] == "FROZEN"


def test_contract_supersedes_v1(contract):
    assert "public_intervention_contract_v1" in contract["supersedes"]


def test_edit_flow_is_primary_method(contract):
    assert contract["core_boundaries"]["edit_flow_is_primary_method"] is True


def test_scope_is_utr_only(contract):
    assert set(contract["core_boundaries"]["scope"]) == {"five_utr", "three_utr"}


def test_cds_and_full_length_out_of_scope(contract):
    oos = set(contract["core_boundaries"]["out_of_scope"])
    assert "cds_synonymous_generation" in oos
    assert "full_length_mrna_joint_optimization" in oos
    assert "full_length_therapeutic_mrna_generation" in oos
    assert "cross_region_full_transcript_synergy" in oos


def test_new_wetlab_forbidden(contract):
    assert contract["core_boundaries"]["new_wetlab"] == "forbidden"


def test_training_device_gpu_only(contract):
    assert contract["core_boundaries"]["training_device"] == "GPU_only"


def test_gse246381_historically_exposed(contract):
    g = contract["gse246381_status"]
    assert g["historically_exposed"] is True
    assert g["evidence_grade"] == "E4"
    assert "sealed" in g["forbidden_wording"]
    assert "untouched" in g["forbidden_wording"] or "never-seen_external_test" in g["forbidden_wording"]


def test_encode_status_complete(contract):
    e = contract["encode_status"]
    assert e["download_status"] == "COMPLETE"
    assert e["raw_files"] == 62
    assert "provider_md5" in e["verification_method"]


def test_hypotheses_h1_through_h8(contract):
    hyps = contract["hypotheses"]
    for h in ["H1_edit_process_modeling", "H2_architecture_irreplacability",
              "H3_hard_constrained_validity", "H4_conditional_controllability",
              "H5_generative_advantage_over_search", "H6_cross_source_and_cross_study_transfer",
              "H7_foundation_model_value", "H8_5utr_3utr_unify_and_diff"]:
        assert h in hyps, f"Missing hypothesis: {h}"


def test_phases_include_mandatory_gates(contract):
    phases = contract["phases"]
    for p in ["C0", "D0", "D1", "B0", "FM0", "MK0", "EF0", "GP0", "FC0", "ME0", "MB0", "TR0", "ER0", "PP0"]:
        assert p in phases, f"Missing phase: {p}"
    assert "FM0_to_MK0_to_EF0" in contract["mandatory_freeze_gates"]
    assert "ME0_to_MB0_Freeze_to_MB0_Run" in contract["mandatory_freeze_gates"]


def test_forbidden_claims_include_key_items(contract):
    forbidden = contract["claims"]["forbidden"]
    assert "Generated candidates improve real therapeutic mRNA efficacy" in forbidden or \
           any("therapeutic mRNA efficacy" in c for c in forbidden)
    assert any("GSE246381" in c and "sealed" in c.lower() for c in forbidden)
    assert any("the first" in c.lower() for c in forbidden)


# ---------------------------------------------------------------------------
# 3. v2 task registry consistency
# ---------------------------------------------------------------------------

def test_registry_contract_id_is_v2(registry):
    assert registry["contract_id"] == "utr_editflow_contract_v2"


def test_registry_has_c0_phase_tasks(registry):
    c0_tasks = [t for t in registry["tasks"] if t["phase"] == "C0"]
    assert len(c0_tasks) >= 5, f"Expected >=5 C0 tasks, got {len(c0_tasks)}"
    task_ids = {t["task_id"] for t in c0_tasks}
    for tid in ["C0-01", "C0-02", "C0-03", "C0-04", "C0-05"]:
        assert tid in task_ids, f"Missing C0 task: {tid}"


def test_registry_c0_tasks_done(registry):
    for t in registry["tasks"]:
        if t["phase"] == "C0":
            assert t["status"] == "DONE", f"C0 task not DONE: {t['task_id']} status={t['status']}"


def test_registry_d0_03_and_d0_04_done(registry):
    for t in registry["tasks"]:
        if t["task_id"] in ("D0-03", "D0-04"):
            assert t["status"] == "DONE", f"{t['task_id']} status={t['status']}"


def test_registry_forward_only_phase_order(registry):
    """Phase execution is forward-only: downstream phases must be PENDING while upstream Gates not passed."""
    phase_order = ["C0", "D0", "D1", "B0", "FM0", "MK0", "EF0", "GP0", "FC0", "ME0", "MB0", "TR0", "ER0", "PP0"]
    statuses = {}
    for t in registry["tasks"]:
        p = t["phase"]
        statuses.setdefault(p, set()).add(t["status"])
    # No phase after D0 should have any DONE tasks (D0-05 still PENDING)
    for p in phase_order[2:]:  # D1 onward
        assert "DONE" not in statuses.get(p, set()), \
            f"Phase {p} has DONE tasks but upstream D0-05 is still PENDING"


# ---------------------------------------------------------------------------
# 4. v1 contract archived correctly
# ---------------------------------------------------------------------------

def test_v1_contract_archived():
    assert V1_CONTRACTS_ARCHIVE.exists(), f"Missing v1 archive: {V1_CONTRACTS_ARCHIVE}"
    assert (V1_CONTRACTS_ARCHIVE / "public_intervention_contract.yaml").exists()
    assert (V1_CONTRACTS_ARCHIVE / "public_intervention_claim_matrix.md").exists()
    assert (V1_CONTRACTS_ARCHIVE / "public_intervention_scientific_question.md").exists()
    assert (V1_CONTRACTS_ARCHIVE / "task_registry.yaml").exists()


def test_v1_superseded_marker_exists():
    assert V1_SUPERSEDED_MARKER.exists()
    text = V1_SUPERSEDED_MARKER.read_text(encoding="utf-8")
    assert "SUPERSEDED" in text
    assert "utr_editflow_contract_v2" in text


def test_v1_contract_not_in_active_configs():
    assert not (ROOT / "configs" / "public_intervention_contract.yaml").exists()


def test_v1_scientific_question_not_in_active_docs():
    assert not (ROOT / "docs" / "public_intervention_scientific_question.md").exists()


def test_v1_claim_matrix_not_in_active_docs():
    assert not (ROOT / "docs" / "public_intervention_claim_matrix.md").exists()


def test_v1_task_registry_not_in_active_execution():
    assert not (ROOT / "docs" / "execution" / "task_registry.yaml").exists()


def test_v1_training_scripts_not_in_root():
    for script in ["train_grpo.py", "train_dagger_ranker.py", "train_proposal_ranker.py",
                   "train_adapter.py", "train_backbone.py", "sample.py"]:
        assert not (ROOT / script).exists(), f"Legacy script still in root: {script}"


def test_legacy_rl_dir_not_in_root():
    assert not (ROOT / "rl").exists()


def test_legacy_benchmark_dirs_not_in_root():
    assert not (ROOT / "benchmark").exists()
    assert not (ROOT / "benchmark_v21").exists()
    assert not (ROOT / "data" / "nmi_benchmark_v2").exists()


def test_legacy_ckpts_not_in_root():
    assert not (ROOT / "ckpts").exists()


# ---------------------------------------------------------------------------
# 5. README alignment with v2
# ---------------------------------------------------------------------------

def test_readme_mentions_v2_contract():
    text = README_PATH.read_text(encoding="utf-8")
    assert "utr_editflow_contract_v2" in text


def test_readme_does_not_claim_full_length_primary():
    text = README_PATH.read_text(encoding="utf-8")
    # The old README title was "full-length protein-coding mRNA generation"
    # The new README must not claim full-length as the primary scope
    assert "full-length protein-coding mRNA generation" not in text


def test_readme_declares_utr_only():
    text = README_PATH.read_text(encoding="utf-8")
    assert "5′UTR" in text or "5'UTR" in text
    assert "3′UTR" in text or "3'UTR" in text


def test_readme_declares_edit_flow_primary():
    text = README_PATH.read_text(encoding="utf-8")
    assert "Edit Flow" in text


def test_readme_declares_no_wetlab():
    text = README_PATH.read_text(encoding="utf-8")
    assert "wet-lab" in text.lower() or "wetlab" in text.lower()


def test_readme_declares_gpu_only():
    text = README_PATH.read_text(encoding="utf-8")
    assert "GPU" in text


# ---------------------------------------------------------------------------
# 6. No active code references v1 contract as a constraint source
# ---------------------------------------------------------------------------

def test_no_active_v1_contract_id_in_data_registry():
    for p in [ROOT / "data_registry" / "intervention_candidates.yaml",
              ROOT / "data_registry" / "unavailable.yaml",
              ROOT / "data_registry" / "search_artifacts" / "intervention_candidates.raw.json"]:
        if p.exists():
            text = p.read_text(encoding="utf-8")
            assert "public_intervention_contract_v1" not in text, \
                f"v1 contract_id still referenced in active file: {p}"


def test_no_active_v1_user_agent_in_scripts():
    for p in [ROOT / "scripts" / "data" / "download_common.py",
              ROOT / "scripts" / "data" / "download_ena_reconstruction.py",
              ROOT / "scripts" / "data" / "systematic_search.py"]:
        if p.exists():
            text = p.read_text(encoding="utf-8")
            # User-agent strings should now reference v2
            assert "public_intervention_contract_v1" not in text, \
                f"v1 contract_id still referenced in active script: {p}"


def test_no_active_v1_contract_test_file():
    assert not (ROOT / "tests" / "test_public_intervention_contract.py").exists()


# ---------------------------------------------------------------------------
# 7. ENCODE status evidence alignment
# ---------------------------------------------------------------------------

def test_encode_manifest_status_complete():
    manifest_path = ROOT / "data" / "p0" / "ENCSR854RUF" / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("ENCODE manifest not present in this checkout")
    import json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest.get("status") == "RAW_READS_COMPLETE"
    assert "provider_md5" in manifest.get("verification_method", "")
    files = manifest.get("files", [])
    assert len(files) == 62
    for f in files:
        assert f["downloaded"] is True, f"File not marked downloaded: {f.get('name')}"
        assert "defer_reason" not in f, f"defer_reason still present: {f.get('name')}"


def test_download_verification_verdict_complete():
    path = ROOT / "docs" / "data" / "download_verification.md"
    text = path.read_text(encoding="utf-8")
    assert "verdict: `COMPLETE`" in text or "verdict: COMPLETE" in text
    assert "files deferred: 0" in text
