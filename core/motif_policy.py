"""P0-05: hard motif policy for the minimal-edit legal-action builder.

Filters edit actions that would introduce known deleterious motifs:

* upstream AUG (uAUG) in the 5'UTR — spurious translation initiation
* cryptic splice donor / acceptor (proxies: new "AGGU" donor-like or
  "UUCAG"/"CCAG" acceptor-like motifs introduced by the edit)
* homopolymer runs >= 6 nt introduced by the edit
* premature in-frame stop codons in the CDS
* start/stop codon violations (start codon not AUG, terminal codon not stop)
* reading-frame violations (CDS length not multiple of 3, or protein changed)

Motif checks are *diff-based* where applicable: an edit is rejected only if
it INCREASES the violation count relative to the parent record, so legal
source sequences that already contain benign motif instances remain editable.
Structural checks (premature stop, start/stop, reading frame) are absolute.
"""
from __future__ import annotations

import re
from typing import List, Sequence

from core.constants import START_CODON, STOP_CODONS, translate
from core.schema import MRNARecord

HOMOPOLYMER_MIN_RUN = 6
# Cryptic splice-site proxies (simplified, documented):
#   donor-like:    AGGU  (approximates the AG|GURAGU 5'SS consensus)
#   acceptor-like: UUCAG / CCAG (pyrimidine-tract + AG 3'SS)
SPLICE_DONOR_MOTIFS = ("AGGU",)
SPLICE_ACCEPTOR_MOTIFS = ("UUCAG", "CCAG")


def _count_overlapping(seq: str, motif: str) -> int:
    return len(re.findall(f"(?={motif})", seq))


def _max_homopolymer_run(seq: str) -> int:
    best = run = 0
    prev = ""
    for ch in seq:
        run = run + 1 if ch == prev else 1
        prev = ch
        best = max(best, run)
    return best


def _n_inframe_stops(cds: str) -> int:
    """Number of in-frame stop codons excluding the terminal codon."""
    n_codons = len(cds) // 3
    stops = 0
    for i in range(n_codons - 1):  # exclude terminal stop
        if cds[3 * i:3 * i + 3] in STOP_CODONS:
            stops += 1
    return stops


# ---------------------------------------------------------------------------
# Individual violation checks (parent vs child)
# ---------------------------------------------------------------------------

def creates_upstream_aug(parent: MRNARecord, child: MRNARecord) -> bool:
    """Edit introduces a new upstream AUG in the 5'UTR."""
    return (_count_overlapping(child.five_utr, START_CODON)
            > _count_overlapping(parent.five_utr, START_CODON))


def creates_cryptic_splice_site(parent: MRNARecord, child: MRNARecord) -> bool:
    """Edit introduces a new cryptic splice donor/acceptor proxy motif."""
    for motif in SPLICE_DONOR_MOTIFS + SPLICE_ACCEPTOR_MOTIFS:
        if (_count_overlapping(child.seq, motif)
                > _count_overlapping(parent.seq, motif)):
            return True
    return False


def creates_homopolymer(parent: MRNARecord, child: MRNARecord,
                        min_run: int = HOMOPOLYMER_MIN_RUN) -> bool:
    """Edit extends/creates a homopolymer run of length >= min_run."""
    child_max = _max_homopolymer_run(child.seq)
    if child_max < min_run:
        return False
    return child_max > _max_homopolymer_run(parent.seq)


def creates_premature_stop(parent: MRNARecord, child: MRNARecord) -> bool:
    """Edit introduces an in-frame premature stop codon in the CDS."""
    return _n_inframe_stops(child.cds) > _n_inframe_stops(parent.cds)


def violates_start_stop(child: MRNARecord) -> bool:
    """Start codon must be AUG; terminal codon must be a stop (absolute)."""
    cds = child.cds
    if len(cds) < 6 or len(cds) % 3 != 0:
        return True
    if cds[:3] != START_CODON:
        return True
    if cds[-3:] not in STOP_CODONS:
        return True
    return False


def violates_reading_frame(parent: MRNARecord, child: MRNARecord) -> bool:
    """CDS length multiple of 3 and protein product preserved (absolute)."""
    if len(child.cds) % 3 != 0:
        return True
    return translate(child.cds) != translate(parent.cds)


# ---------------------------------------------------------------------------
# Aggregate API
# ---------------------------------------------------------------------------

def hard_motif_violations(parent: MRNARecord, child: MRNARecord) -> List[str]:
    """Return the list of hard-motif rule names violated by child vs parent."""
    violations: List[str] = []
    if violates_reading_frame(parent, child):
        violations.append("reading_frame")
    if violates_start_stop(child):
        violations.append("start_stop")
    if creates_premature_stop(parent, child):
        violations.append("premature_stop")
    if creates_upstream_aug(parent, child):
        violations.append("upstream_aug")
    if creates_cryptic_splice_site(parent, child):
        violations.append("cryptic_splice")
    if creates_homopolymer(parent, child):
        violations.append("homopolymer")
    return violations


def is_hard_legal(parent: MRNARecord, child: MRNARecord) -> bool:
    """True iff applying an edit to parent producing child breaks no hard motif."""
    return len(hard_motif_violations(parent, child)) == 0


def filter_hard_motif_candidates(
    parent: MRNARecord,
    candidates: Sequence[MRNARecord],
) -> List[MRNARecord]:
    """Keep only candidates that pass the hard motif policy."""
    return [c for c in candidates if is_hard_legal(parent, c)]


__all__ = [
    "HOMOPOLYMER_MIN_RUN",
    "SPLICE_DONOR_MOTIFS",
    "SPLICE_ACCEPTOR_MOTIFS",
    "creates_upstream_aug",
    "creates_cryptic_splice_site",
    "creates_homopolymer",
    "creates_premature_stop",
    "violates_start_stop",
    "violates_reading_frame",
    "hard_motif_violations",
    "is_hard_legal",
    "filter_hard_motif_candidates",
]
