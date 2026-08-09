"""R1 evidence manifest tests: read the ordinary output-dir JSON artifacts, the
audit module behavior for strict-context-empty grouping, and the decision log.

These tests validate the *real* ordinary-run evidence produced during the R1
grouping/label audit, plus the formalized decision-log entry.  No sealed data
is read; the audit module already refuses protected paths/roles.
"""
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.m4_sparse.audit_ordinary_grouping import audit_rows  # noqa: E402

# Ordinary-only output directory (no database, no sealed data).
OUT_DIR = Path(
    "/home/cunyuliu/mrna_editflow_goal/ordinary_runs/"
    "R1_GROUPING_LABEL_AUDIT_20260809T141132Z"
)

DECISION_LOG = REPO_ROOT / "docs/execution/xeditflow_repair_decision_log.yaml"
R1_DECISION_ID = "XE-REPAIR-DEC-019"


def _row(**extra):
    row = {
        "study": "GSE114002",
        "benchmark": "5U-A1",
        "endpoint": "ep_rl",
        "context_id": "ctx",
        "source_id": "src",
        "source_sequence": "AAAA",
        "candidate_sequence": "CAAA",
        "candidate_id": "cand",
        "edit_list": [{"op": "SUB", "pos": 0, "token": "C"}],
        "edit_count": 1,
        "source_value": 0.0,
        "candidate_value": 1.0,
        "delta": 1.0,
    }
    row.update(extra)
    return row


def _load(name):
    path = OUT_DIR / name
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------- 1. output-dir JSON artifact structure ----------

@pytest.mark.skipif(not OUT_DIR.exists(), reason="R1 ordinary audit output dir not present")
def test_a_audit_json_structure_and_strict_context_empty():
    audit = _load("A_r1_grouping_audit.json")
    assert audit["status"] == "ORDINARY_ONLY_GROUPING_DIAGNOSTIC"
    assert audit["audit_version"] == "r1_ordinary_grouping_audit_v3"
    assert audit["publication_grouping_qualified"] is False
    assert isinstance(audit["n_records"], int)
    assert isinstance(audit["n_delta_defined_records"], int)
    assert audit["n_delta_defined_records"] == 103199
    assert audit["sealed_rows_seen"] == 0
    # strict context grouping is empty; every delta-defined row lacks context
    groups = audit["source_sequence_candidate_groups"]
    assert groups["n_groups"] == 0
    assert groups["n_rows_missing_safe_group_key"] == 103199
    assert groups["missing_group_key_atom_counts"]["context"] == 103199
    assert groups["qualification_status"] == "PROVISIONAL_REQUIRES_BIOLOGICAL_PROVENANCE"
    # all 9 grouping atoms are missing (missing_fraction == 1.0)
    atoms = audit["missing_grouping_atoms"]
    assert len(atoms) == 9
    for atom in ("biological_parent_group", "gene_group", "context_id",
                 "assay_context", "tile_family_group"):
        assert atoms[atom]["missing_fraction"] == pytest.approx(1.0)


@pytest.mark.skipif(not OUT_DIR.exists(), reason="R1 ordinary audit output dir not present")
def test_results_manifest_matches_known_grouping_conclusions():
    r1 = _load("results.json")
    r2 = _load("results2.json")
    # B2 exact (delta-defined GSE114002) groups meet the thresholds
    assert r2["B2_exact_n_rows"] == 55043
    assert r2["B2_exact_groups"] >= 31000          # 31498
    assert r2["B2_exact_groups_ge2"] >= 12000      # 12923
    assert r2["B2_exact_rows_in_ge2"] == 36468
    assert r2["B2_exact_max"] >= 40                # 40
    # all-row B2 also meets the same thresholds
    assert r1["B2_n_groups"] >= 31000              # 31576
    assert r1["B2_n_groups_ge2"] >= 12000          # 12956
    assert r1["B2_max"] >= 40                      # 40
    # numeric types
    for k in ("B2_n_groups", "B2_n_groups_ge2", "B2_max", "B2_n_rows"):
        assert isinstance(r1[k], int)
    # E_split: biological_parent missing ~93.4% (pair), gene only ~40%
    split = r1["E_split"]
    assert split["pair_BIOLOGICAL_PARENT_frac"] == pytest.approx(0.0664)
    assert split["pair_GENE_frac"] == pytest.approx(0.4014)
    # G: study-macro frac_pos ~0.5547, macro spearman ~0.03
    assert r2["G_study_delta"]["study_macro_frac_pos"] == pytest.approx(0.5547)
    assert r2["G_spearman_delta_editcount"]["macro_rho"] == pytest.approx(0.03, abs=0.01)


@pytest.mark.skipif(not OUT_DIR.exists(), reason="R1 ordinary audit output dir not present")
def test_gse217518_label_scale_and_extreme_values_untrustworthy():
    r1 = _load("results.json")
    r3 = _load("results3.json")
    hek = r1["C_delta_hek"]
    assert isinstance(hek["n"], int)
    assert hek["min"] < -2_000_000          # pathological min -2,867,981
    # 12.1% of HEK deltas have |delta| > 1000
    assert r3["C_hek_abs_gt_1000"] / r3["C_hek_n"] == pytest.approx(0.121, abs=0.001)
    # replicate row-CV spans 0.16 .. 0.66
    cvs = [r1["C_raw_files"][k]["row_cv"]["median_row_cv"] for k in r1["C_raw_files"]]
    assert min(cvs) <= 0.17
    assert max(cvs) >= 0.65
    # scale differs by 2-3 orders of magnitude vs other studies
    g217_med = r3["F_median_abs_delta_by_study"]["GSE217518"]["median_abs_delta"]
    assert g217_med > 10.0


# ---------- 2. audit_rows reproduces strict-context-empty grouping ----------

def test_audit_rows_reports_zero_groups_for_missing_context():
    rows = [
        _row(source_sequence="AAAA", candidate_sequence="CAAA"),
        _row(source_sequence="AAAA", candidate_sequence="GAAA"),
    ]
    for r in rows:
        r["context_id"] = None
        r["assay_context"] = None
    groups = audit_rows(rows)["source_sequence_candidate_groups"]
    # reproduces the strict-context-empty behavior: n_groups == 0
    assert groups["n_groups"] == 0
    assert groups["n_rows_missing_safe_group_key"] == 2
    assert groups["missing_group_key_atom_counts"]["context"] == 2


def test_audit_rows_groups_ok_when_context_present():
    rows = [
        _row(source_sequence="AAAA", candidate_sequence="CAAA"),
        _row(source_sequence="AAAA", candidate_sequence="GAAA"),
    ]
    groups = audit_rows(rows)["source_sequence_candidate_groups"]
    assert groups["n_groups"] == 1
    assert groups["n_groups_with_at_least_2_distinct_candidates"] == 1
    assert groups["n_rows_missing_safe_group_key"] == 0


# ---------- 3. decision log entry presence + YAML validity ----------

def test_decision_log_has_r1_entry_and_valid_yaml():
    assert DECISION_LOG.exists(), f"decision log missing: {DECISION_LOG}"
    with open(DECISION_LOG, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    ids = [d["id"] for d in doc["decisions"]]
    assert R1_DECISION_ID in ids, f"{R1_DECISION_ID} not found"
    entry = next(d for d in doc["decisions"] if d["id"] == R1_DECISION_ID)
    assert entry["dimension"] == "r1_grouping_label_evidence"
    assert "EVIDENCE_ESTABLISHED_ROUTE_C_READY" in entry["status"]
    decision = entry["decision"]
    assert "GSE114002" in decision
    assert "GSE217518" in decision
    assert "NOT_JOINABLE" in decision
    assert "NOT_QUALIFIED" in decision
    assert "reports/migration/R1_GROUPING_LABEL_EVIDENCE_20260809.md" in entry["evidence_artifact"]
