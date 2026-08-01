"""Unit tests for GSE217518 UTR sequence reconstruction.

Tests HGVS c. notation parsing, mRNA coordinate conversion, and variant
application (SNV/del/ins) to UTR sequences.

Run: pytest d1_staging/tests/test_reconstruct_gse217518.py -v
"""

import os
import sys

import pytest

# Make the d1 scripts dir importable
HERE = os.path.dirname(os.path.abspath(__file__))
D1_SCRIPTS = os.path.join(HERE, "..", "scripts", "d1")
sys.path.insert(0, D1_SCRIPTS)

from reconstruct_gse217518_sequences import (  # noqa: E402
    parse_hgvs_c,
    hgvs_to_mrna_pos,
    apply_variant_to_utr,
    parse_cds_coordinates,
    parse_mrna_sequence,
)


# ---------------------------------------------------------------------------
# parse_hgvs_c
# ---------------------------------------------------------------------------

class TestParseHgvsC:
    """Test HGVS c. notation parsing."""

    def test_snv_5utr(self):
        """c.-134G>A → 5'UTR SNV at position 134."""
        r = parse_hgvs_c("NM_000518.4(HBB):c.-134G>A")
        assert r is not None
        assert r["position_type"] == "5utr"
        assert r["position"] == 134
        assert r["var_type"] == "snv"
        assert r["ref"] == "G"
        assert r["alt"] == "A"

    def test_snv_3utr(self):
        """c.*113G>A → 3'UTR SNV at position 113."""
        r = parse_hgvs_c("NM_000518.4(HBB):c.*113G>A")
        assert r is not None
        assert r["position_type"] == "3utr"
        assert r["position"] == 113
        assert r["var_type"] == "snv"
        assert r["ref"] == "G"
        assert r["alt"] == "A"

    def test_snv_cds(self):
        """c.134G>A → CDS SNV (should be skipped in UTR extraction)."""
        r = parse_hgvs_c("NM_000518.4(HBB):c.134G>A")
        assert r is not None
        assert r["position_type"] == "cds"
        assert r["position"] == 134

    def test_snv_pipe_notation(self):
        """c.-134G|A → pipe notation normalized to >."""
        r = parse_hgvs_c("NM_000518.4(HBB):c.-134G|A")
        assert r is not None
        assert r["position_type"] == "5utr"
        assert r["ref"] == "G"
        assert r["alt"] == "A"

    def test_del_3utr(self):
        """c.*110_*114delTAAAA → 3'UTR deletion of TAAAA at positions 110-114."""
        r = parse_hgvs_c("NM_000518.4(HBB):c.*110_*114delTAAAA")
        assert r is not None
        assert r["position_type"] == "3utr"
        assert r["position"] == 110
        assert r["position2"] == 114
        assert r["var_type"] == "del"
        assert r["ref"] == "TAAAA"

    def test_del_5utr(self):
        """c.-50_-45delATGCAT → 5'UTR deletion."""
        r = parse_hgvs_c("NM_000518.4(HBB):c.-50_-45delATGCAT")
        assert r is not None
        assert r["position_type"] == "5utr"
        assert r["position"] == -50
        assert r["position2"] == -45
        assert r["var_type"] == "del"
        assert r["ref"] == "ATGCAT"

    def test_ins_3utr(self):
        """c.*100_*101insAT → 3'UTR insertion of AT."""
        r = parse_hgvs_c("NM_000518.4(HBB):c.*100_*101insAT")
        assert r is not None
        assert r["position_type"] == "3utr"
        assert r["position"] == 100
        assert r["position2"] == 101
        assert r["var_type"] == "ins"
        assert r["alt"] == "AT"

    def test_ins_5utr(self):
        """c.-100_-99insGC → 5'UTR insertion of GC."""
        r = parse_hgvs_c("NM_000518.4(HBB):c.-100_-99insGC")
        assert r is not None
        assert r["position_type"] == "5utr"
        assert r["position"] == -100
        assert r["position2"] == -99
        assert r["var_type"] == "ins"
        assert r["alt"] == "GC"

    def test_splice_site(self):
        """c.1394-1G>T → splice site variant."""
        r = parse_hgvs_c("NM_000518.4(HBB):c.1394-1G>T")
        assert r is not None
        assert r["position_type"] == "splice"

    def test_unknown_notation(self):
        """Non-standard notation → unknown position_type."""
        r = parse_hgvs_c("NM_000518.4(HBB):c.(G-T)_+125537_to_termination_codon")
        assert r is not None
        assert r["position_type"] == "unknown"

    def test_no_c_notation(self):
        """No c. prefix → returns None."""
        r = parse_hgvs_c("NM_000518.4(HBB):p.V123M")
        assert r is None

    def test_del_without_seq(self):
        """Deletion without specifying the deleted sequence."""
        r = parse_hgvs_c("NM_000518.4(HBB):c.*110_*114del")
        assert r is not None
        assert r["var_type"] == "del"
        assert r["ref"] == ""


# ---------------------------------------------------------------------------
# hgvs_to_mrna_pos
# ---------------------------------------------------------------------------

class TestHgvsToMrnaPos:
    """Test HGVS c. position to mRNA position conversion."""

    def test_5utr_position(self):
        """c.-1 should map to the base just before CDS start."""
        # cds_start=50 (1-indexed), c.-1 → 0-indexed position 48
        pos = hgvs_to_mrna_pos(1, "5utr", cds_start_1idx=50, cds_end_1idx=100)
        assert pos == 48

    def test_5utr_position_far(self):
        """c.-10 → 0-indexed position 39."""
        pos = hgvs_to_mrna_pos(10, "5utr", cds_start_1idx=50, cds_end_1idx=100)
        assert pos == 39

    def test_5utr_out_of_range(self):
        """c.-100 with cds_start=50 → negative position → None."""
        pos = hgvs_to_mrna_pos(100, "5utr", cds_start_1idx=50, cds_end_1idx=100)
        assert pos is None

    def test_3utr_position(self):
        """c.*1 should map to the base just after CDS end."""
        # cds_end=100 (1-indexed), c.*1 → 0-indexed position 100
        pos = hgvs_to_mrna_pos(1, "3utr", cds_start_1idx=50, cds_end_1idx=100)
        assert pos == 100

    def test_3utr_position_far(self):
        """c.*50 → 0-indexed position 149."""
        pos = hgvs_to_mrna_pos(50, "3utr", cds_start_1idx=50, cds_end_1idx=100)
        assert pos == 149

    def test_cds_position(self):
        """c.1 should map to the CDS start position."""
        # cds_start=50, c.1 → 0-indexed position 49
        pos = hgvs_to_mrna_pos(1, "cds", cds_start_1idx=50, cds_end_1idx=100)
        assert pos == 49


# ---------------------------------------------------------------------------
# apply_variant_to_utr
# ---------------------------------------------------------------------------

class TestApplyVariantToUtr:
    """Test variant application to UTR sequences."""

    def test_snv_3utr(self):
        """Apply a 3'UTR SNV: ref matches, alt replaces."""
        # UTR sequence: "ATCG" (3'UTR starts after CDS end)
        # cds_end=10, c.*2 → mRNA pos 11, UTR pos = 11 - 10 = 1 → 'T'
        utr = "ATCG"
        var_info = {
            "position_type": "3utr",
            "position": 2,
            "var_type": "snv",
            "ref": "T",
            "alt": "G",
        }
        result = apply_variant_to_utr(utr, var_info, cds_start_1idx=1, cds_end_1idx=10)
        assert result is not None
        source, candidate, status = result
        assert status == "ok"
        assert source == "ATCG"
        assert candidate == "AGCG"

    def test_snv_5utr(self):
        """Apply a 5'UTR SNV."""
        # UTR sequence: "ATCG" (5'UTR = mRNA[0:cds_start-1])
        # cds_start=5, c.-3 → mRNA pos 5-3-1=1, UTR pos = 1 → 'T'
        utr = "ATCG"
        var_info = {
            "position_type": "5utr",
            "position": 3,
            "var_type": "snv",
            "ref": "T",
            "alt": "G",
        }
        result = apply_variant_to_utr(utr, var_info, cds_start_1idx=5, cds_end_1idx=10)
        assert result is not None
        source, candidate, status = result
        assert status == "ok"
        assert source == "ATCG"
        assert candidate == "AGCG"

    def test_snv_ref_mismatch(self):
        """Ref allele doesn't match the UTR sequence → None."""
        utr = "ATCG"
        var_info = {
            "position_type": "3utr",
            "position": 2,
            "var_type": "snv",
            "ref": "A",  # Wrong ref (actual is 'T')
            "alt": "G",
        }
        result = apply_variant_to_utr(utr, var_info, cds_start_1idx=1, cds_end_1idx=10)
        assert result is None

    def test_snv_out_of_range(self):
        """Position beyond UTR length → None."""
        utr = "ATCG"
        var_info = {
            "position_type": "3utr",
            "position": 100,
            "var_type": "snv",
            "ref": "A",
            "alt": "G",
        }
        result = apply_variant_to_utr(utr, var_info, cds_start_1idx=1, cds_end_1idx=10)
        assert result is None

    def test_del_3utr(self):
        """Apply a 3'UTR deletion: remove a range of bases."""
        # UTR: "ATTTTTCG", c.*2_*6delTTTT
        # cds_end=10, c.*2 → UTR pos 1, c.*6 → UTR pos 5
        # Delete positions 1-5 (TTTTT), leaving "ACG"... wait:
        # UTR pos 1 = 'T', UTR pos 5 = 'T'
        # del_seq = UTR[1:6] = "TTTTT"
        utr = "ATTTTTCG"
        var_info = {
            "position_type": "3utr",
            "position": 2,
            "position2": 6,
            "var_type": "del",
            "ref": "TTTTT",
            "alt": "",
        }
        result = apply_variant_to_utr(utr, var_info, cds_start_1idx=1, cds_end_1idx=10)
        assert result is not None
        source, candidate, status = result
        assert status == "ok"
        assert source == "ATTTTTCG"
        assert candidate == "ACG"

    def test_del_ref_mismatch(self):
        """Deletion ref doesn't match → None."""
        utr = "ATTTTTCG"
        var_info = {
            "position_type": "3utr",
            "position": 2,
            "position2": 6,
            "var_type": "del",
            "ref": "AAAAA",  # Wrong
            "alt": "",
        }
        result = apply_variant_to_utr(utr, var_info, cds_start_1idx=1, cds_end_1idx=10)
        assert result is None

    def test_ins_3utr(self):
        """Apply a 3'UTR insertion: insert sequence between positions."""
        # UTR: "ATCG", c.*1_*2insGG
        # cds_end=10, c.*1 → UTR pos 0, c.*2 → UTR pos 1
        # Insert after max(0, 1) = 1
        utr = "ATCG"
        var_info = {
            "position_type": "3utr",
            "position": 1,
            "position2": 2,
            "var_type": "ins",
            "ref": "",
            "alt": "GG",
        }
        result = apply_variant_to_utr(utr, var_info, cds_start_1idx=1, cds_end_1idx=10)
        assert result is not None
        source, candidate, status = result
        assert status == "ok"
        assert source == "ATCG"
        assert candidate == "ATGGCG"

    def test_cds_variant_returns_none(self):
        """CDS variants should return None (not applicable to UTR)."""
        var_info = {
            "position_type": "cds",
            "position": 10,
            "var_type": "snv",
            "ref": "A",
            "alt": "G",
        }
        result = apply_variant_to_utr("ATCG", var_info, cds_start_1idx=1, cds_end_1idx=10)
        assert result is None

    def test_snv_no_ref_check(self):
        """When ref is empty, skip ref verification."""
        utr = "ATCG"
        var_info = {
            "position_type": "3utr",
            "position": 2,
            "var_type": "snv",
            "ref": "",  # No ref specified
            "alt": "G",
        }
        result = apply_variant_to_utr(utr, var_info, cds_start_1idx=1, cds_end_1idx=10)
        assert result is not None
        source, candidate, status = result
        assert status == "ok"
        assert candidate == "AGCG"


# ---------------------------------------------------------------------------
# parse_genbank_cds
# ---------------------------------------------------------------------------

class TestParseCdsCoordinates:
    """Test GenBank CDS coordinate parsing."""

    def test_basic_genbank(self):
        """Parse a minimal GenBank record with CDS info."""
        genbank = """LOCUS       NM_TEST      200 bp    mRNA     linear   PRI 01-JAN-2024
DEFINITION  Test sequence.
ACCESSION   NM_TEST
VERSION     NM_TEST.1
FEATURES             Location/Qualifiers
     source          1..200
     CDS             51..150
                     /gene="TEST"
                     /translation="MTEST"
ORIGIN
        1 atgcgatcgc atcgatcgat cgatcgatcga tcgatcgatcg atcgatcgat
//
"""
        result = parse_cds_coordinates(genbank)
        assert result is not None
        cds_start, cds_end = result
        assert cds_start == 51
        assert cds_end == 150

    def test_no_cds(self):
        """GenBank without CDS feature → None."""
        genbank = """LOCUS       NM_TEST      200 bp    mRNA     linear   PRI 01-JAN-2024
DEFINITION  Test sequence.
ACCESSION   NM_TEST
VERSION     NM_TEST.1
FEATURES             Location/Qualifiers
     source          1..200
ORIGIN
        1 atgcgatcgc atcgatcgat cgatcgatcga tcgatcgatcg atcgatcgat
//
"""
        result = parse_cds_coordinates(genbank)
        assert result is None

    def test_complement_cds(self):
        """CDS on complement strand."""
        genbank = """LOCUS       NM_TEST      200 bp    mRNA     linear   PRI 01-JAN-2024
FEATURES             Location/Qualifiers
     CDS             complement(51..150)
ORIGIN
        1 atgcgatcgc atcgatcgat cgatcgatcga tcgatcgatcg atcgatcgat
//
"""
        result = parse_cds_coordinates(genbank)
        assert result is not None
        cds_start, cds_end = result
        assert cds_start == 51
        assert cds_end == 150


# ---------------------------------------------------------------------------
# parse_mrna_sequence
# ---------------------------------------------------------------------------

class TestParseMrnaSequence:
    """Test GenBank sequence extraction."""

    def test_extract_sequence(self):
        """Extract sequence from ORIGIN section."""
        genbank = """LOCUS       NM_TEST       13 bp    mRNA     linear   PRI 01-JAN-2024
ORIGIN
        1 atgcgatcgat cg
//
"""
        seq = parse_mrna_sequence(genbank)
        assert seq == "ATGCGATCGATCG"

    def test_no_origin(self):
        """No ORIGIN section → empty string."""
        genbank = "LOCUS       NM_TEST       12 bp\nDEFINITION  Test."
        seq = parse_mrna_sequence(genbank)
        assert seq == ""

    def test_u_to_t_conversion(self):
        """RNA 'U' bases converted to 'T'."""
        genbank = """LOCUS       NM_TEST        4 bp    mRNA     linear   PRI 01-JAN-2024
ORIGIN
        1 acgu
//
"""
        seq = parse_mrna_sequence(genbank)
        assert seq == "ACGT"


# ---------------------------------------------------------------------------
# Integration: parse + apply
# ---------------------------------------------------------------------------

class TestParseAndApplyIntegration:
    """Integration test: parse HGVS notation then apply variant."""

    def test_snv_3utr_full_pipeline(self):
        """Parse c.*3G>A and apply to a 3'UTR sequence."""
        # Build a UTR where c.*3 maps to a known position
        # cds_end=10, c.*3 → mRNA pos 12, UTR pos = 12 - 10 = 2
        utr = "ATGCATGC"  # pos 2 = 'G'
        var_info = parse_hgvs_c("NM_TEST.1(GENE):c.*3G>A")
        assert var_info is not None
        assert var_info["position_type"] == "3utr"

        result = apply_variant_to_utr(utr, var_info, cds_start_1idx=1, cds_end_1idx=10)
        assert result is not None
        source, candidate, status = result
        assert status == "ok"
        assert source[2] == "G"
        assert candidate[2] == "A"
        assert len(source) == len(candidate)  # SNV preserves length

    def test_del_3utr_full_pipeline(self):
        """Parse c.*2_*3delTA and apply to a 3'UTR sequence."""
        # cds_end=10, c.*2 → UTR pos 1, c.*3 → UTR pos 2
        utr = "ATAGCG"  # pos 1-2 = "TA"
        var_info = parse_hgvs_c("NM_TEST.1(GENE):c.*2_*3delTA")
        assert var_info is not None

        result = apply_variant_to_utr(utr, var_info, cds_start_1idx=1, cds_end_1idx=10)
        assert result is not None
        source, candidate, status = result
        assert status == "ok"
        assert len(candidate) == len(source) - 2  # 2bp deletion
        assert candidate == "AGCG"
