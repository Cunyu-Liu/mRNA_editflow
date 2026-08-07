"""X0-X CDS-B1 rebuild-audit unit tests (GSE207584 iCodon zebrafish reporter).

Phase X0-X (3'UTR & CDS transfer) — PURE DEVELOPMENT PREPARATION tests.  They
do NOT touch the frozen 5' primary model, do NOT access sealed labels, and do
NOT trigger the formal X0-X gate.

These tests verify the pure, data-free logic in
`scripts/x0x/cds_b1_rebuild_audit.py`:
  * variant grouping / sequence collection from CSV+FASTA,
  * the sequence-recovery blocker proof,
  * S7 protein-family-disjoint split conservation (disjoint + exhaustive +
    family-atomic),
  * family group-registry rankable flags,
  * functional-observation emission (aggregate + replicate endpoints),
  * family-anchor CDS emission with per-variant-sequences flagged blocked.

They use small synthetic CSV/FASTA fixtures and never fabricate a measured
per-variant synonymous sequence (the honest blocker contract).
"""
from __future__ import annotations

import io
import gzip

import pytest

from scripts.x0x import cds_b1_rebuild_audit as audit


# ---------------------------------------------------------------------------
# synthetic fixtures
# ---------------------------------------------------------------------------

_COLS = (["Protein_id", "Group", "Name"]
         + [f"zf_library_{t}_{r}" for t in ("2h", "5h", "8h") for r in (1, 2, 3)])


def _csv_bytes(rows: "list[list]") -> bytes:
    import csv
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(_COLS)
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode("utf-8")


def _write_gzip_csv(path, rows):
    path.write_bytes(gzip.compress(_csv_bytes(rows)))


def _write_gzip_fasta(path, seqs):
    out = io.StringIO()
    for name, seq in seqs.items():
        out.write(f">{name}\n{seq}\n")
    path.write_bytes(gzip.compress(out.getvalue().encode("utf-8")))


def _load(tmp_path, rows, seqs):
    """Build variants from synthetic CSV rows + FASTA seqs (temp files)."""
    csv_path = tmp_path / "perfect.csv.gz"
    fasta_path = tmp_path / "ref.fasta.gz"
    _write_gzip_csv(csv_path, rows)
    _write_gzip_fasta(fasta_path, seqs)
    perf = audit.load_csv_gz(csv_path)
    fasta = audit.load_fasta(fasta_path)
    return audit.build_variants(perf, fasta)


# A valid CDS: AUG(M)-UUU(F)-AAA(K)-UUC(F)-UAA(stop), protein "MFKF"
_SEQ1 = "AUGUUUAAAUUCUAA"
# synonymous variant of the same protein: AUG-UUU-AAG(K)-UUC-UAA
_SEQ1B = "AUGUUUAAGUUCUAA"


# ---------------------------------------------------------------------------
# variant grouping + sequence collection
# ---------------------------------------------------------------------------

def test_build_variants_groups_by_protein_group(tmp_path):
    rows = [
        ["P1", "G1", "N1", 1, 2, 3, 1, 2, 3, 1, 2, 3],
        ["P1", "G1", "N1", 4, 5, 6, 4, 5, 6, 4, 5, 6],   # same group, new Name
        ["P1", "G2", "N2", 7, 8, 9, 7, 8, 9, 7, 8, 9],
    ]
    variants = _load(tmp_path, rows, {"N1": _SEQ1, "N2": _SEQ1B})
    assert ("P1", "G1") in variants
    assert ("P1", "G2") in variants
    assert len(variants) == 2
    assert set(variants[("P1", "G1")].names) == {"N1"}
    assert set(variants[("P1", "G1")].seqs) == {_SEQ1}


def test_build_variants_recoverable_translated_protein(tmp_path):
    rows = [["P1", "G1", "N1", 1, 2, 3, 1, 2, 3, 1, 2, 3]]
    variants = _load(tmp_path, rows, {"N1": _SEQ1})
    assert variants[("P1", "G1")].protein_str == "MFKF"


# ---------------------------------------------------------------------------
# sequence-recovery blocker proof
# ---------------------------------------------------------------------------

def test_blocker_detects_blocked_when_groups_share_sequence_set(tmp_path):
    # P1: G1 and G2 both reference Names that share the SAME underlying FASTA
    # sequences -> the FASTA cannot distinguish the codon schemes -> BLOCKED.
    rows = [
        ["P1", "G1", "N1", 1, 2, 3, 1, 2, 3, 1, 2, 3],
        ["P1", "G2", "N1", 4, 5, 6, 4, 5, 6, 4, 5, 6],
    ]
    variants = _load(tmp_path, rows, {"N1": _SEQ1})
    res = audit.check_sequence_blocker(variants)
    assert res["n_proteins"] == 1
    assert res["sequence_blocked_proteins"] == 1
    assert res["sequence_recovery"] == "BLOCKED"


def test_blocker_distinguishes_distinct_group_sequences(tmp_path):
    # P1: G1 -> N1 (seq1), G2 -> N2 (seq1B, distinct codon scheme)
    rows = [
        ["P1", "G1", "N1", 1, 2, 3, 1, 2, 3, 1, 2, 3],
        ["P1", "G2", "N2", 4, 5, 6, 4, 5, 6, 4, 5, 6],
    ]
    variants = _load(tmp_path, rows, {"N1": _SEQ1, "N2": _SEQ1B})
    res = audit.check_sequence_blocker(variants)
    assert res["proteins_with_distinct_group_sequences"] == 1
    assert res["sequence_blocked_proteins"] == 0
    assert res["sequence_recovery"] == "PARTIAL"


def test_blocker_ignores_proteins_without_any_sequence(tmp_path):
    # no FASTA sequences available -> neither blocked nor distinct
    rows = [["P1", "G1", "N1", 1, 2, 3, 1, 2, 3, 1, 2, 3]]
    variants = _load(tmp_path, rows, {})
    res = audit.check_sequence_blocker(variants)
    assert res["sequence_blocked_proteins"] == 0
    assert res["proteins_with_distinct_group_sequences"] == 0


# ---------------------------------------------------------------------------
# S7 protein-family-disjoint split conservation
# ---------------------------------------------------------------------------

def test_s7_split_disjoint_exhaustive_family_atomic():
    fams = [
        {"family_id": "fam_P1", "rankable": True},
        {"family_id": "fam_P2", "rankable": True},
        {"family_id": "fam_P3", "rankable": False},
        {"family_id": "fam_P4", "rankable": True},
    ]
    s7 = audit.build_s7_split(fams, seed=42, train_frac=0.50, val_frac=0.25)
    assert set(s7) == {"fam_P1", "fam_P2", "fam_P3", "fam_P4"}
    # exhaustive: every family assigned exactly one split
    assert all(v in ("train", "val", "test") for v in s7.values())
    train = {f for f, v in s7.items() if v == "train"}
    val = {f for f, v in s7.items() if v == "val"}
    test = {f for f, v in s7.items() if v == "test"}
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert train | val | test == set(s7)


def test_s7_split_deterministic_by_seed():
    fams = [{"family_id": f"fam_P{i}"} for i in range(20)]
    s7a = audit.build_s7_split(fams, seed=7, train_frac=0.70, val_frac=0.15)
    s7b = audit.build_s7_split(fams, seed=7, train_frac=0.70, val_frac=0.15)
    assert s7a == s7b


# ---------------------------------------------------------------------------
# group registry + rankable flags
# ---------------------------------------------------------------------------

def test_group_registry_rankable_flags(tmp_path):
    rows = [
        ["P1", "G1", "N1", 1, 2, 3, 1, 2, 3, 1, 2, 3],
        ["P1", "G2", "N2", 4, 5, 6, 4, 5, 6, 4, 5, 6],  # P1 rankable (2 groups)
        ["P2", "G1", "N3", 7, 8, 9, 7, 8, 9, 7, 8, 9],  # P2 singleton
    ]
    variants = _load(tmp_path, rows, {"N1": _SEQ1, "N2": _SEQ1B, "N3": _SEQ1})
    reg = audit.build_group_registry(variants)
    by_id = {g["family_id"]: g for g in reg}
    assert by_id["fam_P1"]["rankable"] is True
    assert by_id["fam_P1"]["n_variants"] == 2
    assert by_id["fam_P2"]["rankable"] is False
    assert by_id["fam_P2"]["n_variants"] == 1


# ---------------------------------------------------------------------------
# functional observations (aggregate + replicate endpoints)
# ---------------------------------------------------------------------------

def test_functional_observations_emit_mean_and_replicate_endpoints(tmp_path):
    rows = [
        ["P1", "G1", "N1", 1, 2, 3, 1, 2, 3, 1, 2, 3],
        ["P1", "G1", "N1", 4, 5, 6, 4, 5, 6, 4, 5, 6],
    ]
    variants = _load(tmp_path, rows, {"N1": _SEQ1})
    obs = audit.build_functional_observations(variants)
    # per variant: 3 mean endpoints (2h/5h/8h) + 9 replicate endpoints
    assert len(obs) == 3 + 9
    eps = {o["endpoint_id"] for o in obs}
    assert "ep_zf_library_2h_mean" in eps
    assert "ep_zf_library_2h_1" in eps
    assert "ep_zf_library_8h_3" in eps
    # 2h mean endpoint across both rows + 3 replicates: (1+2+3+4+5+6)/6 = 3.5
    for o in obs:
        if o["endpoint_id"] == "ep_zf_library_2h_mean":
            assert o["value"] == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# sequence entities: family-anchor CDS + blocked per-variant flag
# ---------------------------------------------------------------------------

def test_sequence_entities_emit_anchor_and_flag_blocked(tmp_path):
    rows = [
        ["P1", "G1", "N1", 1, 2, 3, 1, 2, 3, 1, 2, 3],
        ["P1", "G2", "N1", 4, 5, 6, 4, 5, 6, 4, 5, 6],
    ]
    variants = _load(tmp_path, rows, {"N1": _SEQ1})
    entities = audit.build_sequence_entities(variants)
    assert len(entities) == 1  # one anchor per protein
    e = entities[0]
    assert e["sequence_id"] == "fam_P1"
    assert e["protein"] == "MFKF"
    assert e["per_variant_distinct_sequences_recoverable"] is False
    assert e["n_codons"] == 5
    assert isinstance(e["anchor_dna_sha256"], str)
    assert isinstance(e["anchor_rna_sha256"], str)


# ---------------------------------------------------------------------------
# aggregate_variant timepoint means
# ---------------------------------------------------------------------------

def test_aggregate_variant_means_ignore_none(tmp_path):
    rows = [
        ["P1", "G1", "N1", 1, "", 3, 1, 2, 3, 1, 2, 3],
        ["P1", "G1", "N1", 5, 6, 7, 5, 6, 7, 5, 6, 7],
    ]
    variants = _load(tmp_path, rows, {"N1": _SEQ1})
    agg = audit.aggregate_variant(variants[("P1", "G1")])
    # 2h values across rows: [1, (missing), 3, 5, 6, 7] -> mean 4.4 (missing skipped)
    assert agg["2h"] == pytest.approx(4.4)
    # 5h/8h: [1,2,3,5,6,7] -> mean 4.0
    assert agg["5h"] == pytest.approx(4.0)
    assert agg["8h"] == pytest.approx(4.0)
