"""Unit tests for GSE149487 extractor (D1-01).

Tests the parsing helpers (_parse_6c_description, _parse_6a_coordinate,
_aggregate_6c_cpm, _find_utr_seq_6a) and the extract_gse149487 integration.

Run: pytest d1_staging/tests/test_d1_gse149487.py -v
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Make the d1 scripts dir importable
HERE = os.path.dirname(os.path.abspath(__file__))
D1_SCRIPTS = os.path.join(HERE, "..", "scripts", "d1")
sys.path.insert(0, D1_SCRIPTS)

from build_canonical_records import (  # noqa: E402
    _parse_6c_description,
    _parse_6a_coordinate,
    _aggregate_6c_cpm,
    _find_utr_seq_6a,
    extract_gse149487,
)


# ---------------------------------------------------------------------------
# _parse_6c_description
# ---------------------------------------------------------------------------


class TestParse6cDescription:
    def test_snv_basic(self):
        r = _parse_6c_description("ABCA1_C_A_chr9_107665961_107690527")
        assert r is not None
        assert r["type"] == "snv"
        assert r["gene"] == "ABCA1"
        assert r["ref"] == "C"
        assert r["alt"] == "A"
        assert r["chr"] == "chr9"
        assert r["start"] == 107665961
        assert r["end"] == 107690527
        assert r["suffix"] is None

    def test_snv_with_potential_suffix(self):
        r = _parse_6c_description("ADAM32_C_T_chr8_38965236_38965236_potential")
        assert r is not None
        assert r["type"] == "snv"
        assert r["gene"] == "ADAM32"
        assert r["ref"] == "C"
        assert r["alt"] == "T"
        assert r["start"] == 38965236
        assert r["end"] == 38965236
        assert r["suffix"] == "potential"

    def test_snv_point_variant_start_eq_end(self):
        """SNV where start == end (single position variant)."""
        r = _parse_6c_description("TP53_G_A_chr17_7577538_7577538")
        assert r is not None
        assert r["type"] == "snv"
        assert r["start"] == r["end"] == 7577538

    def test_wt_description(self):
        r = _parse_6c_description("ALDH1B1_WT_chr9_38392661_38395745")
        assert r is not None
        assert r["type"] == "wt"
        assert r["gene"] == "ALDH1B1"
        assert r["chr"] == "chr9"
        assert r["start"] == 38392661
        assert r["end"] == 38395745

    def test_haplotype_ref(self):
        r = _parse_6c_description("COX7C_ref_chr5_85913784_85913872")
        assert r is not None
        assert r["type"] == "haplotype"
        assert r["gene"] == "COX7C"
        assert r["haplo"] == "ref"

    def test_haplotype_alt(self):
        r = _parse_6c_description("MAT1A_alt_chr10_82049180_82049434")
        assert r is not None
        assert r["type"] == "haplotype"
        assert r["haplo"] == "alt"

    def test_na_description(self):
        r = _parse_6c_description("ADAL_NA_chr15_43622554_43627305")
        assert r is not None
        assert r["type"] == "na"
        assert r["gene"] == "ADAL"

    def test_invalid_returns_none(self):
        assert _parse_6c_description("invalid_format") is None
        assert _parse_6c_description("") is None
        assert _parse_6c_description(None) is None  # type: ignore[arg-type]

    def test_lowercase_alleles_not_snv(self):
        """Lowercase ref/alt should not match SNV format (alleles are ACGT uppercase)."""
        # 'a' is lowercase, not [ACGT]
        r = _parse_6c_description("GENE_a_T_chr1_100_200")
        assert r is None


# ---------------------------------------------------------------------------
# _parse_6a_coordinate
# ---------------------------------------------------------------------------


class TestParse6aCoordinate:
    def test_mutant(self):
        r = _parse_6a_coordinate("RPS6_chr9_19380199_A_T_UTR5")
        assert r is not None
        assert r["type"] == "mut"
        assert r["gene"] == "RPS6"
        assert r["chr"] == "chr9"
        assert r["pos"] == 19380199
        assert r["ref"] == "A"
        assert r["alt"] == "T"

    def test_wt(self):
        r = _parse_6a_coordinate("RPS6_chr9_19380199_WT_UTR5")
        assert r is not None
        assert r["type"] == "wt"
        assert r["gene"] == "RPS6"
        assert r["chr"] == "chr9"
        assert r["pos"] == 19380199

    def test_invalid_returns_none(self):
        assert _parse_6a_coordinate("invalid") is None
        assert _parse_6a_coordinate("") is None
        assert _parse_6a_coordinate(None) is None  # type: ignore[arg-type]

    def test_missing_utr5_suffix(self):
        assert _parse_6a_coordinate("RPS6_chr9_19380199_A_T") is None

    def test_chrX_chrY_supported(self):
        r = _parse_6a_coordinate("GENE_chrX_12345_G_A_UTR5")
        assert r is not None
        assert r["chr"] == "chrX"


# ---------------------------------------------------------------------------
# _aggregate_6c_cpm
# ---------------------------------------------------------------------------


class TestAggregate6cCpm:
    def test_basic_aggregation(self):
        """Mean CPM per description across multiple barcodes."""
        df = pd.DataFrame({
            "description": ["A", "A", "B", "B"],
            "barcode": ["bc1", "bc2", "bc3", "bc4"],
            "293T_TotalRNA_rep1": [2.0, 4.0, 10.0, 20.0],
            "293T_DNA_rep1": [1.0, 3.0, 5.0, 7.0],
        })
        agg = _aggregate_6c_cpm(df)
        assert agg["A"]["293T_TotalRNA_rep1"] == pytest.approx(3.0)
        assert agg["A"]["293T_DNA_rep1"] == pytest.approx(2.0)
        assert agg["B"]["293T_TotalRNA_rep1"] == pytest.approx(15.0)
        assert agg["B"]["293T_DNA_rep1"] == pytest.approx(6.0)

    def test_skips_duplicate_dot1_columns(self):
        """The .1 columns (right-half duplicates) should be skipped."""
        df = pd.DataFrame({
            "description": ["A", "A"],
            "barcode": ["bc1", "bc2"],
            "293T_TotalRNA_rep1": [2.0, 4.0],
            "293T_DNA_rep1": [1.0, 3.0],
            "barcode.1": ["bc1", "bc2"],
            "293T_TotalRNA_rep1.1": [99.0, 99.0],  # duplicate, should be skipped
            "293T_polysome_rep1": [5.0, 7.0],
        })
        agg = _aggregate_6c_cpm(df)
        # Should NOT contain the .1 duplicate
        assert "293T_TotalRNA_rep1.1" not in agg["A"]
        assert "barcode.1" not in agg["A"]
        # Should contain the non-duplicate columns
        assert "293T_TotalRNA_rep1" in agg["A"]
        assert "293T_DNA_rep1" in agg["A"]
        assert "293T_polysome_rep1" in agg["A"]
        assert agg["A"]["293T_polysome_rep1"] == pytest.approx(6.0)

    def test_skips_unnamed_columns(self):
        df = pd.DataFrame({
            "description": ["A"],
            "barcode": ["bc1"],
            "Unnamed: 8": [99.0],
            "293T_TotalRNA_rep1": [2.0],
        })
        agg = _aggregate_6c_cpm(df)
        assert "Unnamed: 8" not in agg["A"]
        assert agg["A"]["293T_TotalRNA_rep1"] == pytest.approx(2.0)

    def test_handles_nan_values(self):
        """NaN values should be ignored in mean (default pandas behavior)."""
        df = pd.DataFrame({
            "description": ["A", "A"],
            "barcode": ["bc1", "bc2"],
            "293T_TotalRNA_rep1": [2.0, float("nan")],
        })
        agg = _aggregate_6c_cpm(df)
        assert agg["A"]["293T_TotalRNA_rep1"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# _find_utr_seq_6a
# ---------------------------------------------------------------------------


class TestFindUtrSeq6a:
    @pytest.fixture
    def lookup(self):
        return {
            ("RPS6", "chr9"): [
                {"type": "mut", "pos": 19380199, "ref": "A", "alt": "T",
                 "seq": "ACGTACGT"},
                {"type": "wt", "pos": 19380199, "seq": "ACGTACGA"},
            ],
            ("GRHL3", "chr1"): [
                {"type": "mut", "pos": 24649532, "ref": "G", "alt": "A",
                 "seq": "GGGG"},
                {"type": "wt", "pos": 24649532, "seq": "GGGA"},
                # Second WT entry at different pos in same UTR range
                {"type": "wt", "pos": 24649600, "seq": "TTTT"},
            ],
        }

    def test_find_mutant(self, lookup):
        seq = _find_utr_seq_6a(
            lookup, "RPS6", "chr9", 19380190, 19380210,
            ref="A", alt="T", want_wt=False,
        )
        assert seq == "ACGTACGT"

    def test_find_wt(self, lookup):
        seq = _find_utr_seq_6a(
            lookup, "RPS6", "chr9", 19380190, 19380210, want_wt=True,
        )
        assert seq == "ACGTACGA"

    def test_pos_out_of_range_returns_none(self, lookup):
        seq = _find_utr_seq_6a(
            lookup, "RPS6", "chr9", 100, 200, want_wt=True,
        )
        assert seq is None

    def test_missing_gene_chrom_returns_none(self, lookup):
        seq = _find_utr_seq_6a(
            lookup, "UNKNOWN", "chr1", 1, 1000, want_wt=True,
        )
        assert seq is None

    def test_wrong_alleles_returns_none(self, lookup):
        seq = _find_utr_seq_6a(
            lookup, "RPS6", "chr9", 19380190, 19380210,
            ref="C", alt="G", want_wt=False,
        )
        assert seq is None

    def test_multiple_wt_matches_prefers_mid_range(self, lookup):
        """When multiple WT entries match the range, prefer the one closest to mid."""
        # GRHL3 has two WT entries at pos 24649532 and 24649600.
        # Range [24649500, 24649700] has mid=24649600 → should pick pos=24649600 (TTTT).
        seq = _find_utr_seq_6a(
            lookup, "GRHL3", "chr1", 24649500, 24649700, want_wt=True,
        )
        assert seq == "TTTT"

    def test_point_variant_range(self, lookup):
        """SNV where start == end == pos should match."""
        seq = _find_utr_seq_6a(
            lookup, "RPS6", "chr9", 19380199, 19380199,
            ref="A", alt="T", want_wt=False,
        )
        assert seq == "ACGTACGT"


# ---------------------------------------------------------------------------
# extract_gse149487 — integration tests with xlsx fixtures
# ---------------------------------------------------------------------------


def _write_moesm8_fixture(path: Path):
    """Write a small MOESM8 xlsx with sheets 6a, 6d, 6e."""
    # 6a: WT + mutant pairs for 2 genes
    df_6a = pd.DataFrame({
        "5' UTR length (bp)": [42, 42, 46, 46],
        "Gene name": ["RPS6", "RPS6", "GRHL3", "GRHL3"],
        "5' UTR genomic coordinate": [
            "RPS6_chr9_19380199_A_T_UTR5",
            "RPS6_chr9_19380199_WT_UTR5",
            "GRHL3_chr1_24649532_G_A_UTR5",
            "GRHL3_chr1_24649532_WT_UTR5",
        ],
        "sequence of 5' UTR": [
            "CCTCTTTTCCGTGGCGCCTCGGAGGCGTTCAGCTGCATCAAG",  # mutant (T at pos)
            "CCTCTTTTCCGTGGCGCCTCGGAGGCGTTCAGCTGCTTCAAG",  # WT (A at pos)
            "AGAAGATGTGCCAAACTGTTAAGAGTGGTTATTTCTGAGCAGAAGA",  # mutant (A)
            "AGAAGATGTGCCAAACTGTTAAGAGTGGTTATTTCTGAGCAGGAGA",  # WT (G)
        ],
    })
    # 6d: transcript-significant pair
    df_6d = pd.DataFrame({
        "Mutant": ["RPS6_A_T_chr9_19380199_19380250"],
        "wt": ["RPS6_WT_chr9_19380199_19380250"],
        "p value": [1.5e-10],
        "log fold change": [-0.45],
        "padj fdr": [3.0e-8],
    })
    # 6e: TE-significant pair
    df_6e = pd.DataFrame({
        "Mutant": ["RPS6_A_T_chr9_19380199_19380250"],
        "wt": ["RPS6_WT_chr9_19380199_19380250"],
        "p value": [2.0e-8],
        "log fold change": [-0.32],
        "padj fdr": [5.0e-6],
    })
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df_6a.to_excel(w, sheet_name="6a 5' UTR sequences", index=False)
        df_6d.to_excel(w, sheet_name="6d transcript FDR<0.1", index=False)
        df_6e.to_excel(w, sheet_name="6e TE FDR<0.1", index=False)


def _write_lim6c_fixture(path: Path):
    """Write a small Lim 6c xlsx with description+barcode+CPMs.

    Includes:
      - 2 barcodes for an SNV mutant description (RPS6_A_T_...)
      - 2 barcodes for the matching WT description (RPS6_WT_...)
      - 1 barcode for an SNV without a matching 6a WT (forces obs record)
    """
    rows = []
    # SNV mutant description with 2 barcodes
    for bc, cpm in [("BC1", 3.0), ("BC2", 5.0)]:
        rows.append({
            "description": "RPS6_A_T_chr9_19380199_19380250",
            "barcode": bc,
            "293T_TotalRNA_rep1": cpm,
            "293T_DNA_rep1": cpm + 1.0,
            "293T_TotalRNA_rep2": cpm + 0.5,
            "293T_DNA_rep2": cpm + 1.5,
            "293T_TotalRNA_rep3": cpm + 0.2,
            "293T_DNA_rep3": cpm + 1.2,
            "Unnamed: 8": None,
            "description.1": "RPS6_A_T_chr9_19380199_19380250",
            "barcode.1": bc,
            "293T_TotalRNA_rep1.1": cpm,
            "293T_polysome_rep1": cpm - 0.5,
            "293T_TotalRNA_rep2.1": cpm + 0.5,
            "293T_polysome_rep2": cpm - 0.3,
            "293T_TotalRNA_rep3.1": cpm + 0.2,
            "293T_polysome_rep3": cpm - 0.1,
        })
    # WT description with 2 barcodes
    for bc, cpm in [("BC3", 2.0), ("BC4", 4.0)]:
        rows.append({
            "description": "RPS6_WT_chr9_19380199_19380250",
            "barcode": bc,
            "293T_TotalRNA_rep1": cpm,
            "293T_DNA_rep1": cpm + 1.0,
            "293T_TotalRNA_rep2": cpm + 0.5,
            "293T_DNA_rep2": cpm + 1.5,
            "293T_TotalRNA_rep3": cpm + 0.2,
            "293T_DNA_rep3": cpm + 1.2,
            "Unnamed: 8": None,
            "description.1": "RPS6_WT_chr9_19380199_19380250",
            "barcode.1": bc,
            "293T_TotalRNA_rep1.1": cpm,
            "293T_polysome_rep1": cpm + 0.5,
            "293T_TotalRNA_rep2.1": cpm + 0.5,
            "293T_polysome_rep2": cpm + 0.3,
            "293T_TotalRNA_rep3.1": cpm + 0.2,
            "293T_polysome_rep3": cpm + 0.1,
        })
    # SNV mutant WITHOUT a matching WT in 6a (GRHL3 has only one WT at same pos)
    # Use a fake variant allele so mutant matches but no WT (impossible since
    # GRHL3 has WT). Instead, use a gene not in 6a lookup at all.
    rows.append({
        "description": "FAKE_GENE_G_A_chr1_100_200",
        "barcode": "BC5",
        "293T_TotalRNA_rep1": 1.0,
        "293T_DNA_rep1": 2.0,
        "293T_TotalRNA_rep2": 1.5,
        "293T_DNA_rep2": 2.5,
        "293T_TotalRNA_rep3": 1.2,
        "293T_DNA_rep3": 2.2,
        "Unnamed: 8": None,
        "description.1": "FAKE_GENE_G_A_chr1_100_200",
        "barcode.1": "BC5",
        "293T_TotalRNA_rep1.1": 1.0,
        "293T_polysome_rep1": 0.5,
        "293T_TotalRNA_rep2.1": 1.5,
        "293T_polysome_rep2": 0.6,
        "293T_TotalRNA_rep3.1": 1.2,
        "293T_polysome_rep3": 0.4,
    })
    # GRHL3 SNV — should pair with GRHL3 WT
    rows.append({
        "description": "GRHL3_G_A_chr1_24649532_24649600",
        "barcode": "BC6",
        "293T_TotalRNA_rep1": 8.0,
        "293T_DNA_rep1": 9.0,
        "293T_TotalRNA_rep2": 8.5,
        "293T_DNA_rep2": 9.5,
        "293T_TotalRNA_rep3": 8.2,
        "293T_DNA_rep3": 9.2,
        "Unnamed: 8": None,
        "description.1": "GRHL3_G_A_chr1_24649532_24649600",
        "barcode.1": "BC6",
        "293T_TotalRNA_rep1.1": 8.0,
        "293T_polysome_rep1": 7.5,
        "293T_TotalRNA_rep2.1": 8.5,
        "293T_polysome_rep2": 7.6,
        "293T_TotalRNA_rep3.1": 8.2,
        "293T_polysome_rep3": 7.4,
    })
    df = pd.DataFrame(rows)
    df.to_excel(path, sheet_name="Sheet1", index=False, engine="openpyxl")


@pytest.fixture
def gse149487_dir(tmp_path):
    """Create a GSE149487 data dir with MOESM8 + Lim 6c fixtures."""
    d = tmp_path / "GSE149487"
    d.mkdir()
    _write_moesm8_fixture(d / "41467_2021_24445_MOESM8_ESM.xlsx")
    _write_lim6c_fixture(d / "Lim_et_al_Supp_Tbl_6c_293T.xlsx")
    return d


class TestExtractGse149487:
    def test_paired_d_c_record_created(self, gse149487_dir):
        """RPS6 SNV should produce a paired D_C record with verified edit_script."""
        records = extract_gse149487(gse149487_dir.parent)
        # Find the RPS6 paired record
        rps6_paired = [
            r for r in records
            if r["record_id"].startswith("GSE149487_snv_RPS6_")
        ]
        assert len(rps6_paired) == 1
        r = rps6_paired[0]
        assert r["region"] == "5'UTR"
        assert r["metadata"]["data_role"] == "D_C"
        # Paired D_C records have no record_type (only observational/incomplete do)
        assert r["metadata"].get("record_type") != "incomplete"
        assert r["source_sequence"] is not None
        assert r["candidate_sequence"] is not None
        assert r["edit_script_verified"] is True
        # source != candidate (it's a real SNV)
        assert r["source_sequence"] != r["candidate_sequence"]
        # edit_distance should be >= 1 (at least one substitution)
        assert r["edit_distance"] >= 1
        # n_sub should be >= 1
        assert r["n_sub"] >= 1

    def test_record_id_format(self, gse149487_dir):
        records = extract_gse149487(gse149487_dir.parent)
        ids = {r["record_id"] for r in records}
        # SNV paired record
        assert "GSE149487_snv_RPS6_A_T_chr9_19380199_19380250" in ids
        # GRHL3 paired record
        assert "GSE149487_snv_GRHL3_G_A_chr1_24649532_24649600" in ids

    def test_wt_control_observational(self, gse149487_dir):
        """The WT description in 6c should produce an observational record."""
        records = extract_gse149487(gse149487_dir.parent)
        wt_recs = [
            r for r in records
            if r["record_id"].startswith("GSE149487_wt_")
        ]
        assert len(wt_recs) >= 1
        for r in wt_recs:
            assert r["metadata"]["record_type"] == "observational"
            assert r["source_sequence"] is None
            assert r["candidate_sequence"] is not None
            assert r["edit_script"] == []

    def test_labels_include_cpm_means(self, gse149487_dir):
        """Labels should include per-description mean CPMs (mutant_ and wt_)."""
        records = extract_gse149487(gse149487_dir.parent)
        rps6 = next(
            r for r in records
            if r["record_id"] == "GSE149487_snv_RPS6_A_T_chr9_19380199_19380250"
        )
        labels = rps6["labels"]
        # mutant CPMs (mean of 3.0 and 5.0 = 4.0)
        assert "mutant_293T_TotalRNA_rep1" in labels
        assert labels["mutant_293T_TotalRNA_rep1"] == pytest.approx(4.0)
        # polysome (mean of -0.5 and -0.3 = -0.4... actually (3.0-0.5)+(5.0-0.3) /2 = 3.6)
        assert "mutant_293T_polysome_rep1" in labels
        # wt CPMs (mean of 2.0 and 4.0 = 3.0)
        assert "wt_293T_TotalRNA_rep1" in labels
        assert labels["wt_293T_TotalRNA_rep1"] == pytest.approx(3.0)
        # No duplicate .1 columns in labels
        assert not any(".1" in k for k in labels)

    def test_labels_include_significance(self, gse149487_dir):
        """Labels should include 6d (transcript) and 6e (TE) significance."""
        records = extract_gse149487(gse149487_dir.parent)
        rps6 = next(
            r for r in records
            if r["record_id"] == "GSE149487_snv_RPS6_A_T_chr9_19380199_19380250"
        )
        labels = rps6["labels"]
        # 6d transcript significance
        assert "transcript_p_value" in labels
        assert labels["transcript_p_value"] == pytest.approx(1.5e-10)
        assert "transcript_log_fold_change" in labels
        assert labels["transcript_log_fold_change"] == pytest.approx(-0.45)
        # 6e TE significance
        assert "te_p_value" in labels
        assert labels["te_p_value"] == pytest.approx(2.0e-8)
        assert "te_log_fold_change" in labels

    def test_no_duplicate_columns_in_labels(self, gse149487_dir):
        """No .1 duplicate columns should leak into labels."""
        records = extract_gse149487(gse149487_dir.parent)
        for r in records:
            for k in r["labels"]:
                assert ".1" not in k, f"label {k} contains .1 suffix"
                assert "Unnamed" not in k

    def test_metadata_fields(self, gse149487_dir):
        records = extract_gse149487(gse149487_dir.parent)
        rps6 = next(
            r for r in records
            if r["record_id"] == "GSE149487_snv_RPS6_A_T_chr9_19380199_19380250"
        )
        md = rps6["metadata"]
        assert md["variant_type"] == "snv"
        assert md["gene"] == "RPS6"
        assert md["chrom"] == "chr9"
        assert md["ref"] == "A"
        assert md["alt"] == "T"
        assert md["pos_start"] == 19380199
        assert md["pos_end"] == 19380250

    def test_missing_dir_returns_empty(self, tmp_path):
        """When the accession dir doesn't exist, return empty list."""
        records = extract_gse149487(tmp_path)
        assert records == []

    def test_missing_files_returns_incomplete(self, tmp_path):
        """When the dir exists but no MOESM8/Lim 6c files, return incomplete."""
        d = tmp_path / "GSE149487"
        d.mkdir()
        (d / "other_file.txt").write_text("placeholder")
        records = extract_gse149487(tmp_path)
        assert len(records) == 1
        assert records[0]["metadata"]["record_type"] == "incomplete"
        assert records[0]["record_id"] == "GSE149487_INCOMPLETE"

    def test_edit_script_round_trip(self, gse149487_dir):
        """apply(edit_script, source) == candidate for all paired records."""
        from edit_script_core import EditOp, apply_edit_script
        records = extract_gse149487(gse149487_dir.parent)
        paired = [
            r for r in records
            if r["metadata"].get("variant_type") == "snv"
            and r["source_sequence"] is not None
        ]
        assert len(paired) >= 2  # RPS6 + GRHL3
        for r in paired:
            ops = [EditOp.from_dict(o) for o in r["edit_script"]]
            assert apply_edit_script(r["source_sequence"], ops) == r["candidate_sequence"]
            assert r["edit_script_verified"] is True

    def test_sequence_normalized(self, gse149487_dir):
        """U should be converted to T; sequences should be ACGT only."""
        records = extract_gse149487(gse149487_dir.parent)
        for r in records:
            for seq_field in ("source_sequence", "candidate_sequence"):
                seq = r.get(seq_field)
                if seq is None:
                    continue
                assert "U" not in seq
                assert all(c in "ACGT" for c in seq), \
                    f"non-ACGT char in {r['record_id']}.{seq_field}"

    def test_grhl3_paired(self, gse149487_dir):
        """GRHL3 SNV should also produce a paired record."""
        records = extract_gse149487(gse149487_dir.parent)
        grhl3 = [
            r for r in records
            if "GRHL3" in r["record_id"]
            and r["record_id"].startswith("GSE149487_snv_")
        ]
        assert len(grhl3) == 1
        assert grhl3[0]["source_sequence"] != grhl3[0]["candidate_sequence"]
        assert grhl3[0]["edit_script_verified"] is True

    def test_snv_without_6a_match_skipped(self, gse149487_dir):
        """FAKE_GENE description in 6c has no 6a entry → skipped (not in records)."""
        records = extract_gse149487(gse149487_dir.parent)
        fake_recs = [r for r in records if "FAKE_GENE" in r["record_id"]]
        assert len(fake_recs) == 0
