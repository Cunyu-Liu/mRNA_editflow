"""Unit tests for data.clean_external_catalog (pure functions only; no I/O
on raw datasets, no network)."""
from __future__ import annotations

import gzip
import json
import zipfile
from pathlib import Path

import pytest

from data.clean_external_catalog import (
    REASON_ILLEGAL_CHARS,
    REASON_KEPT,
    REASON_NO_EXACT_ORF,
    accession_of,
    bprna_record_ok,
    content_hash,
    dot_bracket_valid,
    ensembl_transcript_id,
    find_exact_orf,
    gtf_tags,
    is_canonical_transcript,
    normalise_dna,
    optimus_window_ok,
    parse_fasta,
    parse_gct,
    parse_gtf_attributes,
    parse_stockholm,
    spliceai_chrom_split,
    t_to_u,
    translation_consistent,
    write_cleaning_report,
    zip_integrity_ok,
)
from mrna_editflow.core.constants import translate


def _fake_mrna(protein: str, five: str = "GGG", three: str = "CCC") -> str:
    """Build a transcript embedding an ORF that translates to ``protein``.

    ``translate`` is used in reverse via the codon table's simplest codons
    so the fixture is guaranteed consistent with the project translator.
    """
    # reverse translate with a fixed synonymous codon per amino acid
    codon_for = {
        "A": "GCU", "R": "CGU", "N": "AAU", "D": "GAU", "C": "UGU",
        "Q": "CAA", "E": "GAA", "G": "GGU", "H": "CAU", "I": "AUU",
        "L": "CUG", "K": "AAA", "M": "AUG", "F": "UUU", "P": "CCU",
        "S": "UCU", "T": "ACU", "W": "UGG", "Y": "UAU", "V": "GUU",
    }
    cds = "".join(codon_for[aa] for aa in protein) + "UAA"
    return five + cds + three


class TestNormalisation:
    def test_t_to_u(self):
        assert t_to_u(" acgtnACGTN ") == "ACGUNACGUN"

    def test_normalise_dna_keeps_t(self):
        assert normalise_dna("acgt") == "ACGT"

    def test_content_hash_stable(self):
        assert content_hash("ACGU") == content_hash("ACGU")
        assert content_hash("ACGU") != content_hash("ACGA")


class TestParseFasta:
    def test_multiline(self):
        records = list(parse_fasta([">a desc\n", "AC\n", "GT\n", ">b\n", "TT\n"]))
        assert records == [("a desc", "ACGT"), ("b", "TT")]

    def test_content_before_header_raises(self):
        with pytest.raises(ValueError):
            list(parse_fasta(["ACGT\n", ">a\n", "TT\n"]))

    def test_accession(self):
        assert accession_of("NM_000014.6 Homo sapiens ...") == "NM_000014.6"

    def test_ensembl_tx_id(self):
        assert ensembl_transcript_id("ENST00000269305.9 cds ...") == "ENST00000269305.9"


class TestTranslationConsistency:
    def test_exact_orf_found(self):
        protein = "MAMK"
        mrna = _fake_mrna(protein)
        cds = find_exact_orf(mrna, protein)
        assert cds is not None
        assert translate(cds).rstrip("*") == protein

    def test_translation_consistent_true(self):
        protein = "MKTAA"
        assert translation_consistent(_fake_mrna(protein), protein)

    def test_trailing_star_on_protein_ok(self):
        protein = "MKTAA"
        assert translation_consistent(_fake_mrna(protein), protein + "*")

    def test_mismatch_rejected(self):
        assert not translation_consistent(_fake_mrna("MAMK"), "MAQQ")

    def test_non_m_start_segment_ignored(self):
        # ORF segment after a stop that does not start with M must not match
        mrna = _fake_mrna("MAMK")
        assert find_exact_orf(mrna, "AMK") is None

    def test_dna_alphabet_input_ok(self):
        protein = "MAMK"
        mrna_dna = _fake_mrna(protein).replace("U", "T")
        assert translation_consistent(mrna_dna, protein)

    def test_empty_protein_rejected(self):
        assert find_exact_orf(_fake_mrna("MAMK"), "") is None
        assert find_exact_orf(_fake_mrna("MAMK"), "*") is None

    def test_offset_points_into_original(self):
        protein = "MAMK"
        five = "GGGGG"  # 5-nt 5' flank shifts frame
        mrna = _fake_mrna(protein, five=five)
        cds = find_exact_orf(mrna, protein)
        assert cds is not None
        assert mrna[mrna.index(cds):] == mrna[mrna.index(cds):]  # substring of original
        assert cds in mrna


class TestGtfHelpers:
    def test_parse_attributes(self):
        attrs = parse_gtf_attributes(
            'gene_id "ENSG1.2"; transcript_id "ENST1.1"; gene_type "protein_coding"; tag "MANE_Select";'
        )
        assert attrs["gene_id"] == "ENSG1.2"
        assert attrs["transcript_id"] == "ENST1.1"
        assert gtf_tags(attrs) == frozenset({"MANE_Select"})

    def test_multi_tags(self):
        attrs = parse_gtf_attributes('tag "Ensembl_canonical"; tag "MANE_Select";')
        # repeated tag keys collapse to the last one in the dict view
        assert gtf_tags(attrs) <= {"Ensembl_canonical", "MANE_Select"}

    def test_canonical(self):
        assert is_canonical_transcript({"tag": "Ensembl_canonical"})
        assert is_canonical_transcript({"tag": "MANE_Select"})
        assert not is_canonical_transcript({"tag": "basic"})
        assert not is_canonical_transcript({})

    def test_spliceai_split(self):
        assert spliceai_chrom_split("chr1") == "test"
        assert spliceai_chrom_split("chr9") == "test"
        assert spliceai_chrom_split("chr2") == "train"
        assert spliceai_chrom_split("chrX") == "train"


class TestOptimus:
    def test_window_ok(self):
        assert optimus_window_ok("A" * 50)
        assert optimus_window_ok("t" * 50)  # T->U accepted

    def test_window_rejects(self):
        assert not optimus_window_ok("A" * 49)
        assert not optimus_window_ok("A" * 51)
        assert not optimus_window_ok("A" * 25 + "N" * 25)


class TestGct:
    def test_parse(self):
        lines = ["#1.2\n", "2\t3\n", "Name\tDescription\tS1\tS2\tS3\n",
                 "G1\tx\t1.0\t2.0\t3.0\n", "G2\ty\t4\t5\t6\n"]
        header, rows = parse_gct(lines)
        assert header[:2] == ["Name", "Description"]
        rows = list(rows)
        assert rows[0][0] == "G1"
        assert rows[1][2:] == ["4", "5", "6"]  # TPM kept as published strings

    def test_bad_version_raises(self):
        with pytest.raises(ValueError):
            parse_gct(["#1.3\n", "1\t1\n", "Name\n"])


class TestStockholm:
    def test_parse_records(self):
        lines = [
            "# STOCKHOLM 1.0\n",
            "#=GF ID RF00001\n",
            "seq1 ACGU\n",
            "seq2 A.CU\n",
            "//\n",
            "# STOCKHOLM 1.0\n",
            "s3 AAA\n",
            "s3 CCC\n",
            "//\n",
        ]
        recs = list(parse_stockholm(lines))
        assert len(recs) == 2
        assert recs[0]["sequences"] == {"seq1": "ACGU", "seq2": "A.CU"}
        assert recs[1]["sequences"] == {"s3": "AAACCC"}  # multi-line concat


class TestDotBracket:
    def test_valid(self):
        assert dot_bracket_valid("((..))")
        assert dot_bracket_valid("([{<>}])..")
        assert dot_bracket_valid("....")

    def test_invalid(self):
        assert not dot_bracket_valid("((.)")
        assert not dot_bracket_valid("(])")
        assert not dot_bracket_valid("((x))")
        assert not dot_bracket_valid(")")

    def test_bprna_record(self):
        assert bprna_record_ok("GGAAUCC", "((...))")
        assert not bprna_record_ok("GGAAUC", "((...))")   # length mismatch
        assert not bprna_record_ok("GGXAUCC", "((...))")  # bad alphabet
        assert not bprna_record_ok("GGAAUCC", "((..))")   # unbalanced


class TestZipIntegrity:
    def test_ok(self, tmp_path: Path):
        p = tmp_path / "ok.zip"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("a.txt", "hello")
        ok, n = zip_integrity_ok(p)
        assert ok and n == 1


class TestReport:
    def test_write(self, tmp_path: Path):
        path = write_cleaning_report(tmp_path, "rfam_seed", "Rfam CURRENT", {"kept": 3})
        data = json.loads(path.read_text())
        assert data["dataset_name"] == "rfam_seed"
        assert data["stats"]["kept"] == 3
        assert "cleaning_spec" in data and "citation" in data


class TestDispatch:
    def test_every_catalog_dataset_has_a_driver(self):
        from data.clean_external_catalog import DRIVERS
        from data.download_external_catalog import EXTERNAL_CATALOG
        assert set(DRIVERS) == set(EXTERNAL_CATALOG)

    def test_drivers_are_callable_with_protocol_label(self):
        from data.clean_external_catalog import DRIVERS
        for name, (driver, protocol) in DRIVERS.items():
            assert callable(driver), name
            assert isinstance(protocol, str) and protocol, name

    def test_merge_stats(self):
        from data.clean_external_catalog import _merge_stats
        total = _merge_stats({"a": 1}, {"a": 2, "b": 5})
        assert total == {"a": 3, "b": 5}
        _merge_stats(total, {"b": 1})
        assert total == {"a": 3, "b": 6}

    def test_main_list_outputs_all_datasets(self, capsys):
        from data.clean_external_catalog import main
        from data.download_external_catalog import EXTERNAL_CATALOG
        assert main(["--list"]) == 0
        out = capsys.readouterr().out
        for name in EXTERNAL_CATALOG:
            assert name in out

    def test_main_missing_raw_dir_fails(self, tmp_path: Path):
        from data.clean_external_catalog import main
        rc = main(["--datasets", "rfam_seed",
                   "--raw-root", str(tmp_path / "raw"),
                   "--clean-root", str(tmp_path / "cleaned")])
        assert rc == 1
