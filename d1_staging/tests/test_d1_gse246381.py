"""Unit tests for GSE246381 reconstruction (D1-03).

Tests the parsing helpers (reverse_complement, parse_seqid, parse_gtf_5utr,
build_vglut_condition_map) and the extract_gse246381 integration.

Run: pytest d1_staging/tests/test_d1_gse246381.py -v
"""

import os
import sys
import json
import gzip
from pathlib import Path

import pytest

# Make the d1 scripts dir importable
HERE = os.path.dirname(os.path.abspath(__file__))
D1_SCRIPTS = os.path.join(HERE, "..", "scripts", "d1")
sys.path.insert(0, D1_SCRIPTS)

from reconstruct_gse246381_sequences import (  # noqa: E402
    reverse_complement,
    parse_seqid,
    parse_gtf_5utr,
    build_vglut_condition_map,
)
from build_canonical_records import extract_gse246381  # noqa: E402


# ---------------------------------------------------------------------------
# reverse_complement
# ---------------------------------------------------------------------------


class TestReverseComplement:
    def test_basic(self):
        assert reverse_complement("ACGT") == "ACGT"
        assert reverse_complement("AAAA") == "TTTT"
        assert reverse_complement("ATGC") == "GCAT"

    def test_empty(self):
        assert reverse_complement("") == ""


# ---------------------------------------------------------------------------
# parse_seqid
# ---------------------------------------------------------------------------


class TestParseSeqid:
    def test_standard(self):
        r = parse_seqid(
            "Variant;chr3:123954485;CA|C;Family=14179;"
            "ENST00000485727;REF;TTAAGCTTCA"
        )
        assert r is not None
        assert r["chrom"] == "chr3"
        assert r["pos"] == 123954485
        assert r["ref"] == "CA"
        assert r["alt"] == "C"
        assert r["family"] == "14179"
        assert r["enst"] == "ENST00000485727"
        assert r["allele"] == "REF"
        assert r["barcode"] == "TTAAGCTTCA"

    def test_snv(self):
        r = parse_seqid(
            "Variant;chr1:1000;A|G;Family=1;ENST000001;ALT;ACGTACGTAC"
        )
        assert r is not None
        assert r["ref"] == "A"
        assert r["alt"] == "G"
        assert r["allele"] == "ALT"

    def test_alt_allele(self):
        r = parse_seqid(
            "Variant;chrX:500;TATG|T;Family=99;ENST0000099;REF;GGGGGG"
        )
        assert r is not None
        assert r["ref"] == "TATG"
        assert r["alt"] == "T"

    def test_invalid(self):
        assert parse_seqid("invalid") is None
        assert parse_seqid("") is None
        assert parse_seqid(None) is None  # type: ignore[arg-type]
        # Must start with "Variant"
        assert parse_seqid("NotVariant;chr1:100;A|G") is None


# ---------------------------------------------------------------------------
# parse_gtf_5utr
# ---------------------------------------------------------------------------


def _write_gtf_fixture(path: Path):
    """Write a small GTF with + strand, - strand, and no-CDS transcripts."""
    lines = [
        # + strand transcript ENST000001 on chr1 (5'UTR before CDS)
        'chr1\tHAVANA\ttranscript\t100\t500\t.\t+\t.\t'
        'transcript_id "ENST000001.1"; gene_id "ENSG000001.1";',
        'chr1\tHAVANA\tUTR\t100\t199\t.\t+\t.\t'
        'transcript_id "ENST000001.1";',
        'chr1\tHAVANA\tCDS\t200\t400\t.\t+\t0\t'
        'transcript_id "ENST000001.1";',
        'chr1\tHAVANA\tUTR\t401\t500\t.\t+\t.\t'
        'transcript_id "ENST000001.1";',
        # - strand transcript ENST000002 on chr2 (5'UTR after CDS in genomic
        # coords, since - strand transcription goes right to left)
        'chr2\tHAVANA\ttranscript\t100\t500\t.\t-\t.\t'
        'transcript_id "ENST000002.1"; gene_id "ENSG000002.1";',
        'chr2\tHAVANA\tUTR\t100\t199\t.\t-\t.\t'
        'transcript_id "ENST000002.1";',
        'chr2\tHAVANA\tCDS\t200\t400\t.\t-\t0\t'
        'transcript_id "ENST000002.1";',
        'chr2\tHAVANA\tUTR\t401\t500\t.\t-\t.\t'
        'transcript_id "ENST000002.1";',
        # Transcript with no CDS (should be skipped)
        'chr3\tHAVANA\ttranscript\t100\t500\t.\t+\t.\t'
        'transcript_id "ENST000003.1"; gene_id "ENSG000003.1";',
        'chr3\tHAVANA\tUTR\t100\t199\t.\t+\t.\t'
        'transcript_id "ENST000003.1";',
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


class TestParseGtf5utr:
    def test_plus_strand_5utr(self, tmp_path):
        gtf = tmp_path / "test.gtf"
        _write_gtf_fixture(gtf)
        result = parse_gtf_5utr(gtf)
        assert "ENST000001" in result
        info = result["ENST000001"]
        assert info["chrom"] == "chr1"
        assert info["strand"] == "+"
        assert info["cds_start"] == 200
        assert info["cds_end"] == 400
        # 5'UTR is the UTR upstream of (before) the CDS
        assert info["utr5_exons"] == [(100, 199)]

    def test_minus_strand_5utr(self, tmp_path):
        gtf = tmp_path / "test.gtf"
        _write_gtf_fixture(gtf)
        result = parse_gtf_5utr(gtf)
        assert "ENST000002" in result
        info = result["ENST000002"]
        assert info["chrom"] == "chr2"
        assert info["strand"] == "-"
        assert info["cds_start"] == 200
        assert info["cds_end"] == 400
        # 5'UTR is the UTR downstream of (after) the CDS in genomic coords
        assert info["utr5_exons"] == [(401, 500)]

    def test_no_cds_transcript_skipped(self, tmp_path):
        gtf = tmp_path / "test.gtf"
        _write_gtf_fixture(gtf)
        result = parse_gtf_5utr(gtf)
        # ENST000003 has UTR but no CDS -> skipped
        assert "ENST000003" not in result

    def test_both_transcripts_present(self, tmp_path):
        gtf = tmp_path / "test.gtf"
        _write_gtf_fixture(gtf)
        result = parse_gtf_5utr(gtf)
        assert set(result.keys()) == {"ENST000001", "ENST000002"}


# ---------------------------------------------------------------------------
# build_vglut_condition_map
# ---------------------------------------------------------------------------


class TestBuildVglutConditionMap:
    def test_basic(self):
        cols = [
            "Vglut_Monosome_CreON-30-M-N",
            "Vglut_Polysome_CreON-28-P-N",
            "Vglut_DNA_CreOFF-27-D-F",
        ]
        cm = build_vglut_condition_map(cols)
        assert cm["Vglut_Monosome_CreON-30-M-N"] == ("CreON", "Monosome")
        assert cm["Vglut_Polysome_CreON-28-P-N"] == ("CreON", "Polysome")
        assert cm["Vglut_DNA_CreOFF-27-D-F"] == ("CreOFF", "DNA")

    def test_non_vglut(self):
        cols = ["SIC0228_HVF3VDSXY"]
        cm = build_vglut_condition_map(cols)
        assert cm == {}


# ---------------------------------------------------------------------------
# extract_gse246381 — integration tests with JSONL fixtures
# ---------------------------------------------------------------------------


def _write_reconstructed_fixture(path: Path):
    """Write a small reconstructed_utrs.jsonl with 2 records."""
    records = [
        {
            "record_id": "GSE246381_test1",
            "source_sequence": "ACGTACGTACGT",
            "candidate_sequence": "ACGTTCGTACGT",
            "region": "5'UTR",
            "variant_type": "snv",
            "labels": {"log2fc_te_creon": 0.5},
            "metadata": {
                "chrom": "chr1",
                "variant_position": 4,
                "enst": "ENST001",
            },
            "enst": "ENST001",
        },
        {
            "record_id": "GSE246381_test2",
            "source_sequence": "TTTTAAAA",
            "candidate_sequence": "TTTTAAA",
            "region": "5'UTR",
            "variant_type": "indel",
            "labels": {},
            "metadata": {
                "chrom": "chr2",
                "variant_position": 7,
                "enst": "ENST002",
            },
            "enst": "ENST002",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


@pytest.fixture
def gse246381_dir(tmp_path):
    """Create a GSE246381 data dir with a reconstructed_utrs.jsonl fixture."""
    d = tmp_path / "GSE246381"
    d.mkdir()
    _write_reconstructed_fixture(d / "reconstructed_utrs.jsonl")
    return d


class TestExtractGse246381:
    def test_returns_two_records(self, gse246381_dir):
        records = extract_gse246381(gse246381_dir.parent)
        assert len(records) == 2
        ids = {r["record_id"] for r in records}
        assert "GSE246381_test1" in ids
        assert "GSE246381_test2" in ids

    def test_canonical_record_fields(self, gse246381_dir):
        records = extract_gse246381(gse246381_dir.parent)
        for r in records:
            assert "edit_script" in r
            assert r["edit_script_verified"] is True
            assert r["edit_distance"] > 0
            assert "n_ins" in r
            assert "n_del" in r
            assert "n_sub" in r
            assert "path_ambiguity" in r
            assert r["source_sequence"] is not None
            assert r["candidate_sequence"] is not None
            assert r["source_sequence"] != r["candidate_sequence"]
            assert r["region"] == "5'UTR"
            assert r["dataset"] == "gse246381"
            assert r["accession"] == "GSE246381"

    def test_snv_record_edit_script(self, gse246381_dir):
        """The SNV record should have one substitution."""
        records = extract_gse246381(gse246381_dir.parent)
        snv = next(r for r in records if r["record_id"] == "GSE246381_test1")
        assert snv["metadata"]["variant_type"] == "snv"
        assert snv["n_sub"] == 1
        assert snv["edit_distance"] == 1
        assert snv["edit_script_verified"] is True

    def test_indel_record_edit_script(self, gse246381_dir):
        """The indel record should have a deletion (n_del >= 1)."""
        records = extract_gse246381(gse246381_dir.parent)
        indel = next(r for r in records if r["record_id"] == "GSE246381_test2")
        assert indel["metadata"]["variant_type"] == "indel"
        assert indel["n_del"] >= 1
        assert indel["edit_distance"] >= 1
        assert indel["edit_script_verified"] is True

    def test_edit_script_round_trip(self, gse246381_dir):
        """apply(edit_script, source) == candidate for all records."""
        from edit_script_core import EditOp, apply_edit_script
        records = extract_gse246381(gse246381_dir.parent)
        for r in records:
            ops = [EditOp.from_dict(o) for o in r["edit_script"]]
            assert apply_edit_script(r["source_sequence"], ops) == \
                r["candidate_sequence"]
            assert r["edit_script_verified"] is True

    def test_labels_propagated(self, gse246381_dir):
        """Labels from the JSONL should be propagated as floats."""
        records = extract_gse246381(gse246381_dir.parent)
        snv = next(r for r in records if r["record_id"] == "GSE246381_test1")
        assert "log2fc_te_creon" in snv["labels"]
        assert snv["labels"]["log2fc_te_creon"] == pytest.approx(0.5)
        # The indel record has empty labels
        indel = next(r for r in records if r["record_id"] == "GSE246381_test2")
        assert indel["labels"] == {}

    def test_metadata_fields(self, gse246381_dir):
        records = extract_gse246381(gse246381_dir.parent)
        snv = next(r for r in records if r["record_id"] == "GSE246381_test1")
        md = snv["metadata"]
        assert md["data_role"] == "D_C"
        assert md["variant_type"] == "snv"
        assert md["enst"] == "ENST001"
        assert md["chrom"] == "chr1"
        assert md["variant_position"] == 4

    def test_missing_dir_returns_incomplete(self, tmp_path):
        """When the accession dir/file is absent, return one incomplete record."""
        records = extract_gse246381(tmp_path)
        assert len(records) == 1
        r = records[0]
        assert r["record_id"] == "GSE246381_INCOMPLETE"
        assert r["metadata"]["record_type"] == "incomplete"
        assert r["source_sequence"] is None
        assert r["candidate_sequence"] is None
        assert r["edit_script"] == []
        assert r["edit_distance"] == 0
