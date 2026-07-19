"""K-mer nearest-neighbour leakage audit for mRNA foundation-model experiments.

Frozen mRNA foundation encoders can silently inflate downstream performance if
the evaluation transcripts, or very close homologues, were present in the
training/pretraining corpus. This module provides an offline, dependency-light
audit that compares a query set against a reference corpus using exact sequence
matches and k-mer nearest neighbours.

For a query k-mer set ``Q`` and reference k-mer set ``R``:

``Jaccard(Q,R) = |Q ∩ R| / |Q ∪ R|``

``Containment(Q,R) = |Q ∩ R| / min(|Q|, |R|)``.

Containment is intentionally reported because mRNA datasets often include
truncated isoforms; a short test transcript contained in a longer reference can
have modest Jaccard but high leakage risk.

Implementation
--------------
The reference corpus is indexed as an inverted map ``kmer -> reference_ids``.
For each query, only references sharing at least one k-mer are scored. This is
exact for all non-zero-overlap pairs while avoiding an ``O(N*M)`` full matrix.

Complexity is ``O(total_reference_kmers + total_query_kmers + candidate_hits)``
time and ``O(total_reference_kmers)`` memory, where ``candidate_hits`` is the
number of shared-kmer postings visited by all queries.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Optional, Sequence

from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.download_mrna import load_records_jsonl


def normalize_rna(seq: str) -> str:
    """Normalize RNA/DNA text to an RNA string. Complexity: ``O(len(seq))``."""
    return "".join(str(seq or "").upper().replace("T", "U").split())


def sequence_of(record: object) -> str:
    """Return the full sequence for ``MRNARecord`` or dict-like rows."""
    if isinstance(record, MRNARecord):
        return record.seq
    if isinstance(record, Mapping):
        seq = record.get("seq")
        if isinstance(seq, str):
            return normalize_rna(seq)
        return normalize_rna(
            str(record.get("five_utr", ""))
            + str(record.get("cds", ""))
            + str(record.get("three_utr", ""))
        )
    return normalize_rna(str(record))


def record_id(record: object, fallback: int) -> str:
    """Return a stable display id for a record. Complexity: ``O(1)``."""
    if isinstance(record, MRNARecord):
        return record.transcript_id
    if isinstance(record, Mapping):
        return str(record.get("transcript_id", record.get("id", fallback)))
    return str(fallback)


def kmer_set(seq: str, k: int) -> set[str]:
    """Return distinct k-mers of ``seq``.

    Sequences shorter than ``k`` are represented by the full normalized sequence
    so that exact short records can still be detected. Complexity:
    ``O(max(1, len(seq)-k+1) * k)`` due to string slicing.
    """
    s = normalize_rna(seq)
    if not s:
        return set()
    if k <= 0:
        raise ValueError("k must be positive")
    if len(s) < k:
        return {s}
    return {s[i:i + k] for i in range(len(s) - k + 1)}


@dataclass(frozen=True)
class LeakageHit:
    """One query-reference nearest-neighbour hit."""

    reference_id: str
    jaccard: float
    containment: float
    shared_kmers: int
    reference_kmers: int
    exact_sequence_match: bool

    def to_dict(self) -> dict[str, object]:
        return dict(asdict(self))


@dataclass(frozen=True)
class LeakageRecordAudit:
    """Leakage audit result for one query record."""

    query_id: str
    query_length: int
    query_kmers: int
    top_hits: tuple[LeakageHit, ...]
    max_jaccard: float
    max_containment: float
    exact_sequence_match: bool
    flagged: bool

    def to_dict(self) -> dict[str, object]:
        payload = dict(asdict(self))
        payload["top_hits"] = [hit.to_dict() for hit in self.top_hits]
        return payload


def _reference_index(records: Sequence[object], k: int) -> tuple[list[str], list[str], list[set[str]], dict[str, list[int]]]:
    ids: list[str] = []
    seqs: list[str] = []
    kmers: list[set[str]] = []
    index: dict[str, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        rid = record_id(record, i)
        seq = sequence_of(record)
        ks = kmer_set(seq, k)
        ids.append(rid)
        seqs.append(seq)
        kmers.append(ks)
        for kmer in ks:
            index[kmer].append(i)
    return ids, seqs, kmers, index


def _score_query(
    query_seq: str,
    query_kmers: set[str],
    *,
    ref_ids: Sequence[str],
    ref_seqs: Sequence[str],
    ref_kmers: Sequence[set[str]],
    index: Mapping[str, Sequence[int]],
    top_k: int,
) -> tuple[tuple[LeakageHit, ...], int]:
    candidate_counts: Counter[int] = Counter()
    postings_visited = 0
    for kmer in query_kmers:
        refs = index.get(kmer, ())
        postings_visited += len(refs)
        candidate_counts.update(refs)
    hits: list[LeakageHit] = []
    for ref_idx, shared in candidate_counts.items():
        rks = ref_kmers[ref_idx]
        qn = len(query_kmers)
        rn = len(rks)
        union = qn + rn - shared
        jaccard = float(shared / union) if union else 0.0
        containment = float(shared / max(1, min(qn, rn)))
        hits.append(
            LeakageHit(
                reference_id=ref_ids[ref_idx],
                jaccard=jaccard,
                containment=containment,
                shared_kmers=int(shared),
                reference_kmers=rn,
                exact_sequence_match=bool(query_seq == ref_seqs[ref_idx]),
            )
        )
    hits.sort(
        key=lambda item: (
            item.exact_sequence_match,
            item.containment,
            item.jaccard,
            item.shared_kmers,
        ),
        reverse=True,
    )
    return tuple(hits[: max(1, int(top_k))]), postings_visited


def audit_leakage(
    query_records: Sequence[object],
    reference_records: Sequence[object],
    *,
    k: int = 15,
    top_k: int = 3,
    jaccard_threshold: float = 0.80,
    containment_threshold: float = 0.95,
) -> dict[str, object]:
    """Audit query/reference overlap using exact k-mer nearest neighbours.

    A query is flagged when it has an exact sequence match, or its best hit
    reaches either ``jaccard_threshold`` or ``containment_threshold``.

    Complexity follows the module docstring.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    ref_ids, ref_seqs, ref_kmers, index = _reference_index(reference_records, k)
    rows: list[LeakageRecordAudit] = []
    postings_total = 0
    for i, query in enumerate(query_records):
        qid = record_id(query, i)
        seq = sequence_of(query)
        qks = kmer_set(seq, k)
        hits, postings = _score_query(
            seq,
            qks,
            ref_ids=ref_ids,
            ref_seqs=ref_seqs,
            ref_kmers=ref_kmers,
            index=index,
            top_k=top_k,
        )
        postings_total += postings
        max_j = max((hit.jaccard for hit in hits), default=0.0)
        max_c = max((hit.containment for hit in hits), default=0.0)
        exact = any(hit.exact_sequence_match for hit in hits)
        flagged = bool(exact or max_j >= jaccard_threshold or max_c >= containment_threshold)
        rows.append(
            LeakageRecordAudit(
                query_id=qid,
                query_length=len(seq),
                query_kmers=len(qks),
                top_hits=hits,
                max_jaccard=max_j,
                max_containment=max_c,
                exact_sequence_match=exact,
                flagged=flagged,
            )
        )
    flagged = [row for row in rows if row.flagged]
    return {
        "config": {
            "k": int(k),
            "top_k": int(top_k),
            "jaccard_threshold": float(jaccard_threshold),
            "containment_threshold": float(containment_threshold),
        },
        "summary": {
            "n_query": len(query_records),
            "n_reference": len(reference_records),
            "flagged_count": len(flagged),
            "flagged_fraction": float(len(flagged) / max(1, len(rows))),
            "exact_match_count": int(sum(row.exact_sequence_match for row in rows)),
            "max_jaccard": max((row.max_jaccard for row in rows), default=0.0),
            "max_containment": max((row.max_containment for row in rows), default=0.0),
            "postings_visited": int(postings_total),
            "reference_index_kmers": int(len(index)),
        },
        "per_query": [row.to_dict() for row in rows],
    }


def write_leakage_report(result: Mapping[str, object], out_json: str, out_md: Optional[str] = None) -> tuple[str, Optional[str]]:
    """Write JSON and optional Markdown report. Complexity: ``O(report_size)``."""
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    if out_md is None:
        return out_json, None
    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
    summary = result.get("summary", {})
    per_query = result.get("per_query", [])
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("# mRNA-EditFlow Leakage Audit\n\n")
        fh.write("| Metric | Value |\n|---|---:|\n")
        if isinstance(summary, Mapping):
            for key in (
                "n_query",
                "n_reference",
                "flagged_count",
                "flagged_fraction",
                "exact_match_count",
                "max_jaccard",
                "max_containment",
            ):
                fh.write(f"| `{key}` | {summary.get(key, '')} |\n")
        fh.write("\n## Top Query Hits\n\n")
        fh.write("| Query | Flagged | Max Jaccard | Max Containment | Top Reference | Exact |\n")
        fh.write("|---|---:|---:|---:|---|---:|\n")
        if isinstance(per_query, Iterable):
            for row in per_query:
                if not isinstance(row, Mapping):
                    continue
                hits = row.get("top_hits", [])
                top_ref = ""
                exact = False
                if isinstance(hits, Sequence) and hits:
                    hit = hits[0]
                    if isinstance(hit, Mapping):
                        top_ref = str(hit.get("reference_id", ""))
                        exact = bool(hit.get("exact_sequence_match", False))
                fh.write(
                    f"| {row.get('query_id', '')} | {row.get('flagged', False)} | "
                    f"{float(row.get('max_jaccard', 0.0)):.4f} | "
                    f"{float(row.get('max_containment', 0.0)):.4f} | "
                    f"{top_ref} | {exact} |\n"
                )
    return out_json, out_md


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-jsonl", required=True)
    parser.add_argument("--reference-jsonl", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--kmer", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--jaccard-threshold", type=float, default=0.80)
    parser.add_argument("--containment-threshold", type=float, default=0.95)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    result = audit_leakage(
        load_records_jsonl(args.query_jsonl),
        load_records_jsonl(args.reference_jsonl),
        k=args.kmer,
        top_k=args.top_k,
        jaccard_threshold=args.jaccard_threshold,
        containment_threshold=args.containment_threshold,
    )
    out_json, out_md = write_leakage_report(result, args.out_json, args.out_md)
    print(json.dumps({"out_json": out_json, "out_md": out_md, "summary": result["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LeakageHit",
    "LeakageRecordAudit",
    "normalize_rna",
    "kmer_set",
    "audit_leakage",
    "write_leakage_report",
    "main",
]
