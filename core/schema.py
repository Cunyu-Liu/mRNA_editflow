"""On-disk record schema shared between the data pipeline and the model.

Defines the canonical representation of a processed mRNA transcript and the
per-token region/phase encoding used by the region-aware model head. Keeping
this contract in one place lets the data sub-pipeline and the model be built
independently while staying interoperable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .constants import (
    NUC_TO_ID, PAD_TOKEN, PHASE_NONE,
    REGION_5UTR, REGION_CDS, REGION_3UTR,
)


@dataclass
class MRNARecord:
    """A cleaned, region-annotated mRNA transcript.

    Attributes
    ----------
    transcript_id: stable identifier (e.g. RefSeq accession or synthetic id).
    five_utr / cds / three_utr: RNA strings over ``A C G U`` (T already -> U).
    seq: convenience concatenation ``five_utr + cds + three_utr``.
    species: optional source species (used by ortholog coupling).
    """
    transcript_id: str
    five_utr: str
    cds: str
    three_utr: str
    species: str = "human"
    metadata: dict = field(default_factory=dict)

    @property
    def seq(self) -> str:
        return self.five_utr + self.cds + self.three_utr

    @property
    def cds_start(self) -> int:
        return len(self.five_utr)

    @property
    def cds_end(self) -> int:
        return len(self.five_utr) + len(self.cds)

    def region_ids(self) -> List[int]:
        """Per-nucleotide region id list aligned to :pyattr:`seq`.

        Complexity: O(len(seq)).
        """
        return (
            [REGION_5UTR] * len(self.five_utr)
            + [REGION_CDS] * len(self.cds)
            + [REGION_3UTR] * len(self.three_utr)
        )

    def codon_phases(self) -> List[int]:
        """Per-nucleotide codon phase (0/1/2 inside CDS, PHASE_NONE elsewhere).

        Complexity: O(len(seq)).
        """
        phases = [PHASE_NONE] * len(self.five_utr)
        phases += [i % 3 for i in range(len(self.cds))]
        phases += [PHASE_NONE] * len(self.three_utr)
        return phases

    def token_ids(self) -> List[int]:
        """Per-nucleotide token id list over the RNA alphabet.

        Illegal characters map to PAD (should not occur post-cleaning).
        Complexity: O(len(seq)).
        """
        return [NUC_TO_ID.get(ch, PAD_TOKEN) for ch in self.seq]

    def to_dict(self) -> dict:
        return {
            "transcript_id": self.transcript_id,
            "five_utr": self.five_utr,
            "cds": self.cds,
            "three_utr": self.three_utr,
            "species": self.species,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MRNARecord":
        return cls(
            transcript_id=d["transcript_id"],
            five_utr=d.get("five_utr", ""),
            cds=d.get("cds", ""),
            three_utr=d.get("three_utr", ""),
            species=d.get("species", "human"),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass
class PrecomputedFeatures:
    """Offline-computed thermodynamic / structural features for one transcript.

    All arrays are aligned to the transcript nt positions. ``pairing_prob`` is
    optional (may be None when a folding tool is unavailable and a fallback is
    used). ``mfe`` is a scalar global minimum free energy estimate.
    """
    transcript_id: str
    mfe: float
    start_accessibility: float  # unpaired prob near start codon (higher=better)
    pairing_prob: Optional[List[float]] = field(default=None)

    def to_dict(self) -> dict:
        return {
            "transcript_id": self.transcript_id,
            "mfe": self.mfe,
            "start_accessibility": self.start_accessibility,
            "pairing_prob": self.pairing_prob,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PrecomputedFeatures":
        return cls(
            transcript_id=d["transcript_id"],
            mfe=float(d["mfe"]),
            start_accessibility=float(d["start_accessibility"]),
            pairing_prob=d.get("pairing_prob"),
        )


__all__ = ["MRNARecord", "PrecomputedFeatures"]
