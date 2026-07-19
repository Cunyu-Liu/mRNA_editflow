"""Corpus cleaning / normalisation for mRNA-EditFlow.

Turns raw (possibly DNA-alphabet, possibly over-length) transcripts into
canonical region-annotated :class:`~mrna_editflow.core.schema.MRNARecord`
objects that satisfy the model's invariants, and reports *why* records were
rejected so the manifest can audit corpus attrition.

Cleaning contract
-----------------
* Normalise ``T -> U`` and uppercase; strip whitespace.
* Alphabet: only ``A C G U`` survive (any other symbol -> drop).
* CDS must be a valid ORF: ``AUG`` start, length a multiple of 3, a single
  terminal stop, and no internal stop (delegated to
  :func:`~mrna_editflow.core.constants.is_valid_cds`, with granular reasons
  attributed before the final assertion).
* Length caps (``truncate-or-drop`` policy):
    - 5'UTR over ``max_5utr`` -> **truncate from the 5' end**, keeping the
      3'-proximal window (preserves the Kozak / start context).
    - 3'UTR over ``max_3utr`` -> **truncate from the 3' end**, keeping the
      5'-proximal window (preserves the stop-proximal context).
    - CDS over ``max_cds`` -> **drop** (a CDS cannot be truncated without
      destroying its frame / terminal stop, so truncation is unsafe).

Complexity: O(len(seq)) per record.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from mrna_editflow.core.config import DataConfig
from mrna_editflow.core.constants import (
    NUC_VOCAB,
    START_CODON,
    is_valid_cds,
    translate,
)
from mrna_editflow.core.schema import MRNARecord

# Canonical rejection-reason keys (stable strings for the manifest & tests).
REASON_OK = "ok"
REASON_EMPTY_CDS = "empty_cds"
REASON_ILLEGAL_CHARS = "illegal_chars"
REASON_FRAME = "frame_not_multiple_of_3"
REASON_CDS_TOO_SHORT = "cds_too_short"
REASON_BAD_START = "bad_start_codon"
REASON_NO_TERMINAL_STOP = "no_terminal_stop"
REASON_INTERNAL_STOP = "internal_stop"
REASON_CDS_TOO_LONG = "cds_too_long"

# Every reason that removes a record (used to pre-seed the stats dict).
DROP_REASONS: Tuple[str, ...] = (
    REASON_EMPTY_CDS,
    REASON_ILLEGAL_CHARS,
    REASON_FRAME,
    REASON_CDS_TOO_SHORT,
    REASON_BAD_START,
    REASON_NO_TERMINAL_STOP,
    REASON_INTERNAL_STOP,
    REASON_CDS_TOO_LONG,
)

_VALID_CHARS = frozenset(NUC_VOCAB)


def _normalise(seq: str) -> str:
    """Uppercase, strip whitespace, and map DNA ``T`` -> RNA ``U``.

    Complexity: O(len(seq)).
    """
    return seq.strip().upper().replace("T", "U")


def _has_only_acgu(seq: str) -> bool:
    """True iff every character is in ``{A,C,G,U}`` (empty string is allowed)."""
    return all(ch in _VALID_CHARS for ch in seq)


def _classify_cds(cds: str) -> str:
    """Return :data:`REASON_OK` or the most specific CDS rejection reason.

    Mirrors the individual predicates inside
    :func:`~mrna_editflow.core.constants.is_valid_cds` so we can attribute a
    precise cause. Complexity: O(len(cds)).
    """
    if len(cds) == 0:
        return REASON_EMPTY_CDS
    if len(cds) % 3 != 0:
        return REASON_FRAME
    if len(cds) < 6:
        return REASON_CDS_TOO_SHORT
    if cds[:3] != START_CODON:
        return REASON_BAD_START
    prot = translate(cds)
    if not prot.endswith("*"):
        return REASON_NO_TERMINAL_STOP
    if "*" in prot[:-1]:
        return REASON_INTERNAL_STOP
    return REASON_OK


def _clean_one(
    raw: MRNARecord, cfg: DataConfig
) -> Tuple[Optional[MRNARecord], str, bool, bool]:
    """Clean a single record.

    Returns ``(record_or_none, reason, truncated_5, truncated_3)`` where
    ``reason`` is :data:`REASON_OK` on success. Truncation flags let the caller
    tally how many survivors were length-clipped. Complexity: O(len(seq)).
    """
    five = _normalise(raw.five_utr)
    cds = _normalise(raw.cds)
    three = _normalise(raw.three_utr)

    # Alphabet gate (whole record). UTRs may be empty; CDS emptiness handled below.
    if not (_has_only_acgu(five) and _has_only_acgu(cds) and _has_only_acgu(three)):
        return None, REASON_ILLEGAL_CHARS, False, False

    reason = _classify_cds(cds)
    if reason != REASON_OK:
        return None, reason, False, False

    # CDS over the cap cannot be safely truncated -> drop.
    if len(cds) > cfg.max_cds:
        return None, REASON_CDS_TOO_LONG, False, False

    # UTRs are free-length -> truncate toward the CDS-proximal window.
    trunc5 = len(five) > cfg.max_5utr
    trunc3 = len(three) > cfg.max_3utr
    if trunc5:
        five = five[len(five) - cfg.max_5utr:]   # keep 3'-proximal (near start)
    if trunc3:
        three = three[: cfg.max_3utr]            # keep 5'-proximal (near stop)

    cleaned = MRNARecord(
        transcript_id=raw.transcript_id,
        five_utr=five, cds=cds, three_utr=three, species=raw.species,
    )
    # Safety net: the composite validator must agree with our granular checks.
    assert is_valid_cds(cleaned.cds), "post-clean CDS unexpectedly invalid"
    return cleaned, REASON_OK, trunc5, trunc3


def clean_record(raw: MRNARecord, cfg: Optional[DataConfig] = None) -> Optional[MRNARecord]:
    """Clean one record, returning the canonical record or ``None`` if rejected.

    ``cfg`` defaults to :class:`DataConfig` so the documented one-argument
    signature ``clean_record(raw)`` works. Complexity: O(len(seq)).
    """
    if cfg is None:
        cfg = DataConfig()
    record, _reason, _t5, _t3 = _clean_one(raw, cfg)
    return record


def clean_corpus(
    records: List[MRNARecord], cfg: Optional[DataConfig] = None
) -> Tuple[List[MRNARecord], Dict[str, int]]:
    """Clean a corpus, returning survivors and a drop-statistics dict.

    The stats dict contains a count for every reason in :data:`DROP_REASONS`
    plus ``kept``, ``total``, ``truncated_5utr`` and ``truncated_3utr`` (the
    latter two count *survivors* whose UTRs were clipped, not drops).

    Complexity: O(sum of transcript lengths).
    """
    if cfg is None:
        cfg = DataConfig()
    stats: Dict[str, int] = {r: 0 for r in DROP_REASONS}
    stats["kept"] = 0
    stats["total"] = len(records)
    stats["truncated_5utr"] = 0
    stats["truncated_3utr"] = 0

    survivors: List[MRNARecord] = []
    for raw in records:
        record, reason, trunc5, trunc3 = _clean_one(raw, cfg)
        if record is None:
            stats[reason] += 1
            continue
        survivors.append(record)
        stats["kept"] += 1
        if trunc5:
            stats["truncated_5utr"] += 1
        if trunc3:
            stats["truncated_3utr"] += 1
    return survivors, stats


__all__ = [
    "clean_record",
    "clean_corpus",
    "DROP_REASONS",
    "REASON_OK",
    "REASON_EMPTY_CDS",
    "REASON_ILLEGAL_CHARS",
    "REASON_FRAME",
    "REASON_CDS_TOO_SHORT",
    "REASON_BAD_START",
    "REASON_NO_TERMINAL_STOP",
    "REASON_INTERNAL_STOP",
    "REASON_CDS_TOO_LONG",
]
