"""P3-05 Task 4: Phase-Aware CDS Encoding — 100% alignment verification.

Verifies that codon_phases() in core/schema.py satisfies all P3-05 requirements:
- CDS codon boundary aligned to real cds_start
- BOS and 5'UTR length do not change codon phase
- Codon representation maps only to CDS codon
- UTR does not perform pseudo codon pooling
- Synonymous candidate set from genetic code
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from core.constants import (
    START_CODON, STOP_CODONS, CODON_TABLE, SYNONYMOUS_CODONS,
    PHASE_NONE, REGION_5UTR, REGION_CDS, REGION_3UTR,
)
from core.schema import MRNARecord


class TestPhaseAlignment:
    """Verify codon_phases() is 100% correctly aligned."""

    def test_phase_zero_at_cds_start(self):
        """First CDS nucleotide must have phase 0."""
        record = MRNARecord(
            transcript_id="t1",
            five_utr="ACGU" * 10,  # 40 nt
            cds=START_CODON + "GCU" * 5 + "UAA",
            three_utr="UGCU" * 10,
        )
        phases = record.codon_phases()
        assert phases[record.cds_start] == 0, \
            f"First CDS nt must be phase 0, got {phases[record.cds_start]}"

    def test_phase_cycles_012_through_cds(self):
        """CDS phases must cycle 0, 1, 2, 0, 1, 2, ... through CDS."""
        record = MRNARecord(
            transcript_id="t1",
            five_utr="ACGU" * 10,
            cds=START_CODON + "GCU" * 10 + "UAA",  # 36 nt CDS = 12 codons
            three_utr="UGCU" * 10,
        )
        phases = record.codon_phases()
        cds_phases = phases[record.cds_start:record.cds_end]
        expected = [0, 1, 2] * 12  # 36 nt = 12 codons
        assert cds_phases == expected, \
            f"CDS phases must cycle 012, got {cds_phases[:9]}..."

    def test_utr_has_phase_none(self):
        """5'UTR and 3'UTR must have PHASE_NONE."""
        record = MRNARecord(
            transcript_id="t1",
            five_utr="ACGU" * 10,  # 40 nt
            cds=START_CODON + "GCU" * 5 + "UAA",
            three_utr="UGCU" * 10,  # 40 nt
        )
        phases = record.codon_phases()
        utr5_phases = phases[:record.cds_start]
        utr3_phases = phases[record.cds_end:]
        assert all(p == PHASE_NONE for p in utr5_phases), \
            "5'UTR must be all PHASE_NONE"
        assert all(p == PHASE_NONE for p in utr3_phases), \
            "3'UTR must be all PHASE_NONE"

    def test_utr_length_does_not_shift_cds_phase(self):
        """Changing 5'UTR length must NOT change CDS phase pattern."""
        cds = START_CODON + "GCU" * 5 + "UAA"
        # Same CDS, different 5'UTR lengths
        r1 = MRNARecord(transcript_id="t1", five_utr="A", cds=cds, three_utr="U")
        r2 = MRNARecord(transcript_id="t2", five_utr="ACGU" * 25, cds=cds, three_utr="U")
        r3 = MRNARecord(transcript_id="t3", five_utr="", cds=cds, three_utr="U")

        p1 = r1.codon_phases()[r1.cds_start:r1.cds_end]
        p2 = r2.codon_phases()[r2.cds_start:r2.cds_end]
        p3 = r3.codon_phases()[r3.cds_start:r3.cds_end]

        assert p1 == p2 == p3, \
            "CDS phase pattern must be identical regardless of 5'UTR length"
        assert p1 == [0, 1, 2] * (len(cds) // 3)

    def test_phase_length_matches_sequence(self):
        """codon_phases() length must equal sequence length."""
        record = MRNARecord(
            transcript_id="t1",
            five_utr="ACGU" * 10,
            cds=START_CODON + "GCU" * 10 + "UAA",
            three_utr="UGCU" * 10,
        )
        phases = record.codon_phases()
        assert len(phases) == len(record.seq), \
            f"Phase length {len(phases)} != seq length {len(record.seq)}"

    def test_region_ids_match_regions(self):
        """region_ids() must correctly label 5'UTR, CDS, 3'UTR."""
        record = MRNARecord(
            transcript_id="t1",
            five_utr="ACGU" * 10,  # 40 nt
            cds=START_CODON + "GCU" * 5 + "UAA",  # 18 nt
            three_utr="UGCU" * 10,  # 40 nt
        )
        regions = record.region_ids()
        assert len(regions) == len(record.seq)
        assert all(r == REGION_5UTR for r in regions[:record.cds_start])
        assert all(r == REGION_CDS for r in regions[record.cds_start:record.cds_end])
        assert all(r == REGION_3UTR for r in regions[record.cds_end:])


class TestSynonymousCodonSet:
    """Verify synonymous candidate set is from genetic code."""

    def test_synonymous_codons_preserve_protein(self):
        """Every codon in SYNONYMOUS_CODONS[codon] must encode same amino acid."""
        for codon, synonyms in SYNONYMOUS_CODONS.items():
            if codon not in CODON_TABLE:
                continue
            aa = CODON_TABLE[codon]
            for syn in synonyms:
                assert syn in CODON_TABLE, f"{syn} not in CODON_TABLE"
                assert CODON_TABLE[syn] == aa, \
                    f"{syn} encodes {CODON_TABLE[syn]} but {codon} encodes {aa}"

    def test_start_and_stop_not_in_synonymous(self):
        """START and STOP codons should not appear in other codons' synonymous sets."""
        for codon, synonyms in SYNONYMOUS_CODONS.items():
            if codon not in CODON_TABLE:
                continue
            for stop in STOP_CODONS:
                assert stop not in synonyms, \
                    f"Stop codon {stop} found in synonyms of {codon}"
            # START_CODON can be in its own synonyms (Methionine has only AUG)
            if codon != START_CODON:
                assert START_CODON not in synonyms, \
                    f"START_CODON found in synonyms of {codon}"

    def test_synonymous_set_is_reflexive(self):
        """If A is synonymous to B, then B is synonymous to A."""
        for codon, synonyms in SYNONYMOUS_CODONS.items():
            for syn in synonyms:
                if syn in SYNONYMOUS_CODONS:
                    assert codon in SYNONYMOUS_CODONS[syn], \
                        f"{codon} not in SYNONYMOUS_CODONS[{syn}] but {syn} is in SYNONYMOUS_CODONS[{codon}]"


class TestNoPseudoCodonPooling:
    """Verify UTR does not perform pseudo codon pooling."""

    def test_utr_phases_are_all_none(self):
        """UTR positions must all be PHASE_NONE — no pseudo codon phase."""
        # Even if 5'UTR length is a multiple of 3, it should NOT get codon phases
        record = MRNARecord(
            transcript_id="t1",
            five_utr="ACG" * 10,  # 30 nt, multiple of 3
            cds=START_CODON + "GCU" * 5 + "UAA",
            three_utr="ACG" * 10,  # 30 nt, multiple of 3
        )
        phases = record.codon_phases()
        utr5 = phases[:record.cds_start]
        utr3 = phases[record.cds_end:]
        assert all(p == PHASE_NONE for p in utr5), \
            "5'UTR must not have pseudo codon phases even if length is multiple of 3"
        assert all(p == PHASE_NONE for p in utr3), \
            "3'UTR must not have pseudo codon phases even if length is multiple of 3"

    def test_cds_phase_independent_of_utr_mod3(self):
        """CDS phase pattern is the same whether UTR length is mod 0, 1, or 2."""
        cds = START_CODON + "GCU" * 5 + "UAA"
        # UTR lengths: 30 (mod 0), 31 (mod 1), 32 (mod 2)
        r0 = MRNARecord(transcript_id="t0", five_utr="A" * 30, cds=cds, three_utr="U")
        r1 = MRNARecord(transcript_id="t1", five_utr="A" * 31, cds=cds, three_utr="U")
        r2 = MRNARecord(transcript_id="t2", five_utr="A" * 32, cds=cds, three_utr="U")

        p0 = r0.codon_phases()[r0.cds_start:r0.cds_end]
        p1 = r1.codon_phases()[r1.cds_start:r1.cds_end]
        p2 = r2.codon_phases()[r2.cds_start:r2.cds_end]

        assert p0 == p1 == p2 == [0, 1, 2] * (len(cds) // 3), \
            "CDS phases must be identical regardless of UTR length mod 3"
