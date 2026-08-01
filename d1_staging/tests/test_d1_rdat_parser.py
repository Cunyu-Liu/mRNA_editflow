"""Unit tests for GSE173083 .rdat parser (D1-01).

Tests the _parse_rdat helper and extract_gse173083 integration.

Run: pytest d1_staging/tests/test_d1_rdat_parser.py -v
"""

import os
import sys
import json
import tempfile
from pathlib import Path

import pytest

# Make the d1 scripts dir importable
HERE = os.path.dirname(os.path.abspath(__file__))
D1_SCRIPTS = os.path.join(HERE, "..", "scripts", "d1")
sys.path.insert(0, D1_SCRIPTS)

from build_canonical_records import _parse_rdat, extract_gse173083  # noqa: E402


# ---------------------------------------------------------------------------
# .rdat fixture
# ---------------------------------------------------------------------------

SAMPLE_RDAT = """\
RDAT_VERSION\t0.34
NAME\tEteRNA Cloud Lab
SEQUENCE\tXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
STRUCTURE\t....................................
OFFSET\t0
COMMENT\tOutput of MAPseeker v1.3

ANNOTATION\texperimentType:StandardState\ttemperature:50C

ANNOTATION_DATA:1\tMAPseq:design_name:testing\tMAPseq:project_name:Roll your own structure\tMAPseq:ID:9830366\tsequence:GGAAAUUUGCUGAUCUUCAGCAUCAGGAAUACUGAUGUCCUUUGCCCGCCAGCGGCAAGGCAGGAUAACCUACCAUUCGUGGUAGGAAAAGAAACAACAACAACAAC\tsignal_to_noise:medium:3.746
ANNOTATION_DATA:2\tMAPseq:design_name:ZZ-1\tMAPseq:project_name:Roll your own structure\tMAPseq:ID:9831675\tsequence:GGAAAACGGCUCUGGGUAGCAGGAAACUAGCUACGCAGAGCCGUCAGCAGGGAAACCUGGUGACGAGACGGUGCGUUCGCGCACCGAAAAGAAACAACAACAACAAC\tsignal_to_noise:weak:0.964
ANNOTATION_DATA:3\tMAPseq:design_name:ZZ-2\tMAPseq:project_name:Roll your own structure\tMAPseq:ID:9831995\tsequence:GGAAACGACUAUACCGUCGAGGGAUAGUCACGGACUCCGACCGGAGGAAUAUCCGGAUGUCGGAGUAACGGGCCGUUCGCGGCCCGAAAAGAAACAACAACAACAAC\tsignal_to_noise:good:7.183
"""


@pytest.fixture
def rdat_file(tmp_path):
    """Create a sample .rdat file."""
    p = tmp_path / "GSM5259588_RYOS1_50C_0000.rdat"
    p.write_text(SAMPLE_RDAT)
    return p


@pytest.fixture
def gse173083_dir(tmp_path):
    """Create a GSE173083 data dir with two .rdat files (same designs, diff conditions)."""
    d = tmp_path / "GSE173083"
    d.mkdir()
    (d / "GSM5259588_RYOS1_50C_0000.rdat").write_text(SAMPLE_RDAT)
    # Second condition with different SNR for same designs
    cond2 = SAMPLE_RDAT.replace("RYOS1_50C", "RYOS1_MG50")
    cond2_lines = cond2.split("\n")
    # Modify SNR values to test per-condition tracking
    cond2_lines[-4] = cond2_lines[-4].replace("medium:3.746", "good:5.0")
    cond2_lines[-3] = cond2_lines[-3].replace("weak:0.964", "medium:1.5")
    cond2_lines[-2] = cond2_lines[-2].replace("good:7.183", "good:8.0")
    (d / "GSM5259588_RYOS1_MG50_0000.rdat").write_text("\n".join(cond2_lines))
    return d


# ---------------------------------------------------------------------------
# _parse_rdat
# ---------------------------------------------------------------------------


class TestParseRdat:
    def test_parses_annotation_data_lines(self, rdat_file):
        entries = _parse_rdat(rdat_file)
        assert len(entries) == 3

    def test_extracts_sequence(self, rdat_file):
        entries = _parse_rdat(rdat_file)
        assert entries[0]["sequence"] == \
            "GGAAAUUUGCUGAUCUUCAGCAUCAGGAAUACUGAUGUCCUUUGCCCGCCAGCGGCAAGGCAGGAUAACCUACCAUUCGUGGUAGGAAAAGAAACAACAACAACAAC"

    def test_extracts_mapsq_id(self, rdat_file):
        entries = _parse_rdat(rdat_file)
        assert entries[0]["mapseq_id"] == "9830366"
        assert entries[1]["mapseq_id"] == "9831675"
        assert entries[2]["mapseq_id"] == "9831995"

    def test_extracts_design_name(self, rdat_file):
        entries = _parse_rdat(rdat_file)
        assert entries[0]["design_name"] == "testing"
        assert entries[1]["design_name"] == "ZZ-1"

    def test_extracts_snr(self, rdat_file):
        entries = _parse_rdat(rdat_file)
        assert entries[0]["snr_quality"] == "medium"
        assert entries[0]["snr_value"] == pytest.approx(3.746)
        assert entries[2]["snr_quality"] == "good"
        assert entries[2]["snr_value"] == pytest.approx(7.183)

    def test_skips_non_annotation_lines(self, rdat_file):
        """Header lines (RDAT_VERSION, SEQUENCE, etc.) should not produce entries."""
        entries = _parse_rdat(rdat_file)
        # Only 3 ANNOTATION_DATA lines, not header lines
        assert len(entries) == 3
        for e in entries:
            assert e["sequence"]
            assert e["mapseq_id"]

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.rdat"
        p.write_text("")
        assert _parse_rdat(p) == []

    def test_no_sequence_field_skipped(self, tmp_path):
        """ANNOTATION_DATA without a sequence: field should be skipped."""
        p = tmp_path / "test.rdat"
        p.write_text("ANNOTATION_DATA:1\tMAPseq:ID:123\tsignal_to_noise:good:5.0\n")
        entries = _parse_rdat(p)
        assert len(entries) == 0


# ---------------------------------------------------------------------------
# extract_gse173083 integration
# ---------------------------------------------------------------------------


class TestExtractGse173083:
    def test_extracts_unique_designs(self, gse173083_dir):
        """Two .rdat files with same 3 designs → 3 unique records (dedup by ID)."""
        records = extract_gse173083(gse173083_dir.parent)
        assert len(records) == 3

    def test_record_id_format(self, gse173083_dir):
        records = extract_gse173083(gse173083_dir.parent)
        ids = {r["record_id"] for r in records}
        assert "GSE173083_rdat_9830366" in ids
        assert "GSE173083_rdat_9831675" in ids

    def test_region_and_data_role(self, gse173083_dir):
        records = extract_gse173083(gse173083_dir.parent)
        for r in records:
            assert r["region"] == "full_length"
            assert r["metadata"]["data_role"] == "D_A"

    def test_sequence_normalized(self, gse173083_dir):
        """U should be converted to T."""
        records = extract_gse173083(gse173083_dir.parent)
        for r in records:
            seq = r["candidate_sequence"]
            assert "U" not in seq
            assert all(c in "ACGT" for c in seq)

    def test_conditions_tracked(self, gse173083_dir):
        """Each design should have 2 conditions (50C + MG50) with SNR values."""
        records = extract_gse173083(gse173083_dir.parent)
        for r in records:
            assert r["metadata"]["n_conditions"] == 2
            conds = r["metadata"]["conditions"]
            assert "GSM5259588_RYOS1_50C_0000" in conds
            assert "GSM5259588_RYOS1_MG50_0000" in conds

    def test_snr_labels(self, gse173083_dir):
        """SNR mean/min/max labels should be present."""
        records = extract_gse173083(gse173083_dir.parent)
        r = records[0]
        assert "signal_to_noise_mean" in r["labels"]
        assert "signal_to_noise_min" in r["labels"]
        assert "signal_to_noise_max" in r["labels"]
        # Design 9830366: 50C=3.746, MG50=5.0
        assert r["labels"]["signal_to_noise_min"] == pytest.approx(3.746)
        assert r["labels"]["signal_to_noise_max"] == pytest.approx(5.0)
        assert r["labels"]["signal_to_noise_mean"] == pytest.approx((3.746 + 5.0) / 2)

    def test_no_edit_observational(self, gse173083_dir):
        """Records should be observational (no edit script)."""
        records = extract_gse173083(gse173083_dir.parent)
        for r in records:
            assert r["edit_script"] == []
            assert r["edit_script_verified"] is True
            assert r["metadata"]["record_type"] == "observational"

    def test_missing_dir_returns_empty(self, tmp_path):
        """When the accession directory doesn't exist, return empty list."""
        records = extract_gse173083(tmp_path)
        assert len(records) == 0

    def test_empty_dir_returns_incomplete(self, tmp_path):
        """When the directory exists but has no .rdat or xlsx, return incomplete."""
        d = tmp_path / "GSE173083"
        d.mkdir()
        records = extract_gse173083(tmp_path)
        assert len(records) == 1
        assert records[0]["metadata"]["record_type"] == "incomplete"
