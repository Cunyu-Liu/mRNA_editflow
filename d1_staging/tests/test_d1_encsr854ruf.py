"""Unit tests for ENCSR854RUF reconstruction & extractor (D1-03).

Tests the parsing helpers (reverse_complement, normalize_seq,
apply_variant_to_plus_strand) in reconstruct_encsr854ruf_sequences.py and
the extract_encsr854ruf integration in build_canonical_records.py.

Run: pytest d1_staging/tests/test_d1_encsr854ruf.py -v
"""

import json
import os
import sys
from pathlib import Path

import pytest

# Make the d1 scripts dir importable
HERE = os.path.dirname(os.path.abspath(__file__))
D1_SCRIPTS = os.path.join(HERE, "..", "scripts", "d1")
sys.path.insert(0, D1_SCRIPTS)

from reconstruct_encsr854ruf_sequences import (  # noqa: E402
    reverse_complement,
    normalize_seq,
    apply_variant_to_plus_strand,
)
from build_canonical_records import extract_encsr854ruf  # noqa: E402


# ---------------------------------------------------------------------------
# reverse_complement
# ---------------------------------------------------------------------------


class TestReverseComplement:
    def test_basic(self):
        # ACGT is its own reverse complement (palindrome)
        assert reverse_complement("ACGT") == "ACGT"
        assert reverse_complement("AAAA") == "TTTT"
        # N complements to N, so ACGTN -> TGCAN -> reversed NACGT
        assert reverse_complement("ACGTN") == "NACGT"

    def test_lowercase(self):
        # translate preserves case: acgt -> tgca -> reversed acgt
        assert reverse_complement("acgt") == "acgt"

    def test_empty(self):
        assert reverse_complement("") == ""


# ---------------------------------------------------------------------------
# normalize_seq
# ---------------------------------------------------------------------------


class TestNormalizeSeq:
    def test_basic(self):
        assert normalize_seq("acgt") == "ACGT"

    def test_rna(self):
        # U -> T, then N is not in ACGT so removed
        assert normalize_seq("AUGCN") == "ATGC"

    def test_mixed(self):
        # non-ACGT characters (., space, -) removed
        assert normalize_seq("a.c g-t") == "ACGT"

    def test_none(self):
        assert normalize_seq(None) == ""


# ---------------------------------------------------------------------------
# apply_variant_to_plus_strand
# ---------------------------------------------------------------------------


class TestApplyVariantToPlusStrand:
    def test_snv(self):
        # offset=2 is 'G'; replace with 'A' -> ACATACGT
        result = apply_variant_to_plus_strand("ACGTACGT", 2, "G", "A")
        assert result == "ACATACGT"

    def test_insertion(self):
        # offset=1 is 'C'; ref='C', alt='CA' inserts 'A' after 'C' -> ACAGT
        result = apply_variant_to_plus_strand("ACGT", 1, "C", "CA")
        assert result == "ACAGT"

    def test_deletion(self):
        # offset=1 is 'CA'; ref='CA', alt='C' removes the 'A' -> ACGT
        result = apply_variant_to_plus_strand("ACAGT", 1, "CA", "C")
        assert result == "ACGT"

    def test_ref_mismatch(self):
        # offset=1 is 'C' but ref says 'T' -> None
        result = apply_variant_to_plus_strand("ACGT", 1, "T", "G")
        assert result is None

    def test_offset_out_of_range(self):
        # offset=5 is beyond the 2-char sequence; actual slice is "" != "G" -> None
        result = apply_variant_to_plus_strand("AC", 5, "G", "T")
        assert result is None


# ---------------------------------------------------------------------------
# extract_encsr854ruf — integration tests with JSONL fixtures
# ---------------------------------------------------------------------------


def _write_jsonl_fixture(path: Path, records):
    """Write a list of dicts as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _make_record(
    record_id,
    source,
    candidate,
    gene_symbol="TEST",
    variant_type="snv",
    variant_position=4,
    ref_allele="A",
    alt_allele="T",
):
    return {
        "record_id": record_id,
        "source_sequence": source,
        "candidate_sequence": candidate,
        "region": "3'UTR",
        "variant_type": variant_type,
        "labels": {"log2FoldChange_Skew_HEK293FT": 1.5},
        "metadata": {
            "chrom": "chr1",
            "variant_position": variant_position,
            "gene_symbol": gene_symbol,
            "ref_allele": ref_allele,
            "alt_allele": alt_allele,
        },
    }


@pytest.fixture
def encsr854ruf_dir(tmp_path):
    """Create an ENCSR854RUF data dir with a small reconstructed_oligos.jsonl."""
    d = tmp_path / "ENCSR854RUF"
    d.mkdir()
    records = [
        # Valid SNV: A at position 4 -> T
        _make_record(
            "ENCSR854RUF_test1",
            source="ACGTACGTACGT",
            candidate="ACGTTCGTACGT",
            gene_symbol="GENE1",
            variant_position=4,
            ref_allele="A",
            alt_allele="T",
        ),
        # Valid SNV: different gene
        _make_record(
            "ENCSR854RUF_test2",
            source="CCCCCCCC",
            candidate="CCCTCCCC",
            gene_symbol="GENE2",
            variant_position=3,
            ref_allele="C",
            alt_allele="T",
        ),
        # Skipped: source == candidate
        _make_record(
            "ENCSR854RUF_identical",
            source="ACGTACGT",
            candidate="ACGTACGT",
        ),
        # Skipped: empty candidate
        _make_record(
            "ENCSR854RUF_empty",
            source="ACGT",
            candidate="",
        ),
    ]
    _write_jsonl_fixture(d / "reconstructed_oligos.jsonl", records)
    return d


class TestExtractEncsr854ruf:
    def test_extracts_valid_records(self, encsr854ruf_dir):
        """Two valid SNV records should be extracted; identical/empty skipped."""
        records = extract_encsr854ruf(encsr854ruf_dir.parent)
        ids = {r["record_id"] for r in records}
        assert "ENCSR854RUF_test1" in ids
        assert "ENCSR854RUF_test2" in ids
        # Skipped records should NOT appear
        assert "ENCSR854RUF_identical" not in ids
        assert "ENCSR854RUF_empty" not in ids
        # Exactly 2 records (skips filtered out)
        assert len(records) == 2

    def test_canonical_record_fields(self, encsr854ruf_dir):
        """Each extracted record has the full canonical_record field set."""
        records = extract_encsr854ruf(encsr854ruf_dir.parent)
        for r in records:
            assert r["dataset"] == "encsr854ruf_mprau"
            assert r["accession"] == "ENCSR854RUF"
            assert r["region"] == "3'UTR"
            assert r["source_sequence"] is not None
            assert r["candidate_sequence"] is not None
            assert r["source_sequence"] != r["candidate_sequence"]
            assert "edit_script" in r
            assert isinstance(r["edit_script"], list)
            assert r["edit_script_verified"] is True
            assert r["edit_distance"] >= 1
            assert r["n_sub"] >= 1
            assert r["path_ambiguity"] >= 1
            assert isinstance(r["labels"], dict)
            assert isinstance(r["metadata"], dict)

    def test_labels_passed_through(self, encsr854ruf_dir):
        """Labels from the JSONL fixture should propagate to the record."""
        records = extract_encsr854ruf(encsr854ruf_dir.parent)
        r1 = next(r for r in records if r["record_id"] == "ENCSR854RUF_test1")
        assert "log2FoldChange_Skew_HEK293FT" in r1["labels"]
        assert r1["labels"]["log2FoldChange_Skew_HEK293FT"] == pytest.approx(1.5)

    def test_metadata_passed_through(self, encsr854ruf_dir):
        """Raw metadata fields should propagate, with data_role=D_C."""
        records = extract_encsr854ruf(encsr854ruf_dir.parent)
        r1 = next(r for r in records if r["record_id"] == "ENCSR854RUF_test1")
        md = r1["metadata"]
        assert md["data_role"] == "D_C"
        assert md["gene_symbol"] == "GENE1"
        assert md["variant_type"] == "snv"
        assert md["chrom"] == "chr1"
        assert md["variant_position"] == 4
        assert md["ref_allele"] == "A"
        assert md["alt_allele"] == "T"

    def test_edit_script_round_trip(self, encsr854ruf_dir):
        """apply(edit_script, source) == candidate for all extracted records."""
        from edit_script_core import EditOp, apply_edit_script
        records = extract_encsr854ruf(encsr854ruf_dir.parent)
        assert len(records) >= 2
        for r in records:
            ops = [EditOp.from_dict(o) for o in r["edit_script"]]
            assert apply_edit_script(r["source_sequence"], ops) == r["candidate_sequence"]
            assert r["edit_script_verified"] is True

    def test_sequence_normalized(self, encsr854ruf_dir):
        """Extracted sequences should contain only ACGT (no U, no N, no lowercase)."""
        records = extract_encsr854ruf(encsr854ruf_dir.parent)
        for r in records:
            for seq_field in ("source_sequence", "candidate_sequence"):
                seq = r[seq_field]
                assert all(c in "ACGT" for c in seq), \
                    f"non-ACGT char in {r['record_id']}.{seq_field}"

    def test_not_found_returns_incomplete(self, tmp_path):
        """When reconstructed_oligos.jsonl is missing, return 1 incomplete record."""
        # tmp_path exists but has no ENCSR854RUF dir
        records = extract_encsr854ruf(tmp_path)
        assert len(records) == 1
        r = records[0]
        assert r["record_id"] == "ENCSR854RUF_INCOMPLETE"
        assert r["accession"] == "ENCSR854RUF"
        assert r["source_sequence"] is None
        assert r["candidate_sequence"] is None
        assert r["edit_script"] == []
        assert r["metadata"]["record_type"] == "incomplete"
        assert r["metadata"]["data_role"] == "D_C"

    def test_empty_dir_returns_incomplete(self, tmp_path):
        """When the accession dir exists but the JSONL is missing, return incomplete."""
        d = tmp_path / "ENCSR854RUF"
        d.mkdir()
        (d / "other_file.txt").write_text("placeholder")
        records = extract_encsr854ruf(tmp_path)
        assert len(records) == 1
        assert records[0]["record_id"] == "ENCSR854RUF_INCOMPLETE"
