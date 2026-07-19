"""Synonymous codon-lattice dynamic-programming baseline.

This module provides a small, executable CDS-only baseline for comparison with
structure/codon optimizers such as LinearDesign and EnsembleDesign. It is not a
drop-in reimplementation of those external systems. Instead, it gives the paper
pipeline a transparent no-heavy-dependency codon-lattice optimizer that obeys
the same biological invariant:

``translate(CDS_source) = translate(CDS_optimized)``.

The optimizer fixes the start codon and terminal stop codon. Every internal
codon is replaced only by a synonymous codon for the same amino acid.

Objective
---------
For source codons ``x_1 ... x_n`` and selected codons ``c_1 ... c_n``, maximize

``sum_i U_i(c_i) + sum_{i>1} B(c_{i-1}, c_i)``

where

``U_i(c) = alpha log CAI(c) - beta (GC(c)-g*)^2 - gamma 1[c != x_i]``

and ``B`` is a dinucleotide boundary stability proxy. With an optional maximum
number of codon changes ``K``, the dynamic program is

``DP[i,k,c] = U_i(c) + max_p DP[i-1,k-1[c!=x_i],p] + B(p,c)``.

Complexity is ``O(n * K * C^2)`` time and ``O(n * K * C)`` backpointers, with
``C <= 6`` for the standard genetic code.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Sequence

from mrna_editflow.core.constants import (
    CODON_TABLE,
    START_CODON,
    STOP_CODONS,
    SYNONYMOUS_CODONS,
    is_valid_cds,
    translate,
)
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.download_mrna import load_records_jsonl, write_records_jsonl
from mrna_editflow.eval.metrics import cai, codon_weights_from_reference, gc_fraction


@dataclass(frozen=True)
class CodonLatticeDPConfig:
    """Configuration for the codon-lattice DP baseline.

    ``max_codon_changes=None`` means no explicit change budget. Scores are in
    arbitrary additive units. ``cai_weight`` rewards frequent synonymous codons,
    ``gc_weight`` penalizes codon GC deviation from ``target_gc``,
    ``boundary_weight`` rewards stable codon-boundary dinucleotides, and
    ``change_penalty`` discourages excessive synonymous rewrites.

    Construction complexity is ``O(1)``.
    """

    cai_weight: float = 1.0
    gc_weight: float = 0.10
    target_gc: float = 0.55
    boundary_weight: float = 0.05
    change_penalty: float = 0.0
    max_codon_changes: Optional[int] = None


@dataclass(frozen=True)
class CodonLatticeDPResult:
    """Result from optimizing one CDS.

    ``objective`` is the final DP score. ``codon_changes`` counts synonymous
    codon replacements relative to the source CDS, not nucleotide edits.

    Conversion to dict is ``O(1)`` plus sequence length for stored strings.
    """

    source_cds: str
    optimized_cds: str
    objective: float
    codon_changes: int
    source_cai: float
    optimized_cai: float
    source_gc: float
    optimized_gc: float
    source_protein: str
    optimized_protein: str

    def to_dict(self) -> dict[str, object]:
        return dict(asdict(self))


_BOUNDARY_DINUCLEOTIDE_STABILITY = {
    "GC": 2.0,
    "CG": 2.0,
    "GG": 1.5,
    "CC": 1.5,
    "AU": 1.0,
    "UA": 1.0,
    "GU": 0.5,
    "UG": 0.5,
}


def _codons(cds: str) -> list[str]:
    """Split an in-frame CDS into codons. Complexity: ``O(len(cds))``."""
    return [cds[i:i + 3] for i in range(0, len(cds), 3)]


def _codon_gc(codon: str) -> float:
    """Return GC fraction of one codon. Complexity: ``O(1)``."""
    return float((codon.count("G") + codon.count("C")) / 3.0)


def boundary_stability_score(left: str, right: str) -> float:
    """Return a local codon-boundary stability proxy.

    The score uses the dinucleotide spanning the codon boundary,
    ``left[-1] + right[0]``. It is a transparent stand-in for a first-order
    folding/lattice term; true external structure optimizers should replace it
    in paper-grade comparisons.

    Complexity is ``O(1)``.
    """
    if len(left) != 3 or len(right) != 3:
        raise ValueError("left and right codons must have length 3")
    return float(_BOUNDARY_DINUCLEOTIDE_STABILITY.get(left[-1] + right[0], 0.0))


def _unary_score(
    codon: str,
    source_codon: str,
    config: CodonLatticeDPConfig,
    codon_weights: Mapping[str, float],
) -> float:
    if CODON_TABLE.get(codon) == "*":
        return 0.0
    weight = max(float(codon_weights.get(codon, 1e-6)), 1e-6)
    cai_score = config.cai_weight * math.log(weight)
    gc_penalty = config.gc_weight * (_codon_gc(codon) - config.target_gc) ** 2
    change_penalty = config.change_penalty if codon != source_codon else 0.0
    return float(cai_score - gc_penalty - change_penalty)


def _choices_for_position(codon: str, position: int, n_codons: int) -> list[str]:
    """Return legal codon choices at one CDS position.

    Start and terminal stop codons are fixed. Internal codons are replaced only
    by synonymous codons for the same amino acid. Complexity is ``O(C)``.
    """
    if position == 0:
        return [codon]
    if position == n_codons - 1:
        return [codon]
    aa = CODON_TABLE.get(codon)
    if aa is None or aa == "*":
        return [codon]
    return list(SYNONYMOUS_CODONS[aa])


def optimize_cds_synonymous(
    cds: str,
    *,
    config: Optional[CodonLatticeDPConfig] = None,
    codon_weights: Optional[Mapping[str, float]] = None,
) -> CodonLatticeDPResult:
    """Optimize a CDS over the synonymous codon lattice.

    The dynamic program maximizes the objective stated in the module docstring.
    A candidate is legal if the translated amino-acid sequence is unchanged and
    the terminal stop remains terminal. Invalid CDS inputs raise ``ValueError``
    rather than being silently repaired.

    Complexity is ``O(n * K * C^2)`` time and ``O(n * K * C)`` memory, where
    ``n`` is codon count, ``K`` is the codon-change budget and ``C`` is maximum
    synonymous choices per amino acid.
    """
    cfg = config or CodonLatticeDPConfig()
    weights = dict(codon_weights or {})
    if not weights:
        weights = {codon: 1.0 for codon in CODON_TABLE if CODON_TABLE[codon] != "*"}
    cds = str(cds).upper().replace("T", "U")
    if not is_valid_cds(cds):
        raise ValueError("optimize_cds_synonymous requires a valid in-frame CDS")
    if not cds.startswith(START_CODON) or cds[-3:] not in STOP_CODONS:
        raise ValueError("CDS must have AUG start and terminal stop")

    source_codons = _codons(cds)
    n = len(source_codons)
    choices = [_choices_for_position(codon, i, n) for i, codon in enumerate(source_codons)]
    if cfg.max_codon_changes is None:
        # No edit-budget dimension is needed when all synonymous rewrites are
        # allowed; change_penalty is already captured by the unary score.
        simple_back: list[dict[str, str]] = []
        simple_prev: dict[str, float] = {}
        for codon in choices[0]:
            simple_prev[codon] = _unary_score(codon, source_codons[0], cfg, weights)
        simple_back.append({})

        for i in range(1, n):
            cur: dict[str, float] = {}
            cur_back: dict[str, str] = {}
            for prev_codon, prev_score in simple_prev.items():
                for codon in choices[i]:
                    score = (
                        prev_score
                        + _unary_score(codon, source_codons[i], cfg, weights)
                        + cfg.boundary_weight * boundary_stability_score(prev_codon, codon)
                    )
                    if codon not in cur or score > cur[codon]:
                        cur[codon] = float(score)
                        cur_back[codon] = prev_codon
            if not cur:
                raise ValueError("no feasible synonymous codon path")
            simple_prev = cur
            simple_back.append(cur_back)

        best_codon, best_score = max(simple_prev.items(), key=lambda item: item[1])
        path = [best_codon]
        key_codon = best_codon
        for i in range(n - 1, 0, -1):
            key_codon = simple_back[i][key_codon]
            path.append(key_codon)
        optimized_codons = list(reversed(path))
        optimized = "".join(optimized_codons)
    else:
        max_changes = int(cfg.max_codon_changes)
        if max_changes < 0:
            raise ValueError("max_codon_changes must be non-negative")

        back: list[dict[tuple[int, str], tuple[int, str]]] = []
        prev: dict[tuple[int, str], float] = {}

        for codon in choices[0]:
            changed = int(codon != source_codons[0])
            if changed <= max_changes:
                prev[(changed, codon)] = _unary_score(codon, source_codons[0], cfg, weights)
        back.append({})

        for i in range(1, n):
            cur: dict[tuple[int, str], float] = {}
            cur_back: dict[tuple[int, str], tuple[int, str]] = {}
            for (prev_changes, prev_codon), prev_score in prev.items():
                for codon in choices[i]:
                    changed = int(codon != source_codons[i])
                    total_changes = prev_changes + changed
                    if total_changes > max_changes:
                        continue
                    score = (
                        prev_score
                        + _unary_score(codon, source_codons[i], cfg, weights)
                        + cfg.boundary_weight * boundary_stability_score(prev_codon, codon)
                    )
                    key = (total_changes, codon)
                    if key not in cur or score > cur[key]:
                        cur[key] = float(score)
                        cur_back[key] = (prev_changes, prev_codon)
            if not cur:
                raise ValueError("no feasible synonymous codon path under max_codon_changes")
            prev = cur
            back.append(cur_back)

        best_key, best_score = max(prev.items(), key=lambda item: item[1])
        path = [best_key[1]]
        key = best_key
        for i in range(n - 1, 0, -1):
            key = back[i][key]
            path.append(key[1])
        optimized_codons = list(reversed(path))
        optimized = "".join(optimized_codons)
    source_protein = translate(cds)
    optimized_protein = translate(optimized)
    if source_protein != optimized_protein:
        raise RuntimeError("DP invariant violated: optimized CDS changed protein")
    codon_changes = sum(a != b for a, b in zip(source_codons, optimized_codons))
    return CodonLatticeDPResult(
        source_cds=cds,
        optimized_cds=optimized,
        objective=float(best_score),
        codon_changes=int(codon_changes),
        source_cai=cai(cds, weights),
        optimized_cai=cai(optimized, weights),
        source_gc=gc_fraction(cds),
        optimized_gc=gc_fraction(optimized),
        source_protein=source_protein,
        optimized_protein=optimized_protein,
    )


def optimize_record_cds(
    record: MRNARecord,
    *,
    config: Optional[CodonLatticeDPConfig] = None,
    codon_weights: Optional[Mapping[str, float]] = None,
) -> tuple[MRNARecord, CodonLatticeDPResult]:
    """Optimize only the CDS of one record and preserve both UTRs.

    Complexity follows :func:`optimize_cds_synonymous`.
    """
    result = optimize_cds_synonymous(record.cds, config=config, codon_weights=codon_weights)
    optimized = MRNARecord(
        transcript_id=record.transcript_id,
        five_utr=record.five_utr,
        cds=result.optimized_cds,
        three_utr=record.three_utr,
        species=record.species,
    )
    return optimized, result


def optimize_records_cds(
    records: Sequence[MRNARecord],
    *,
    config: Optional[CodonLatticeDPConfig] = None,
    codon_weights: Optional[Mapping[str, float]] = None,
) -> tuple[list[MRNARecord], list[CodonLatticeDPResult]]:
    """Optimize a batch of records. Complexity is the sum of per-record DP costs."""
    optimized = []
    results = []
    for record in records:
        rec, result = optimize_record_cds(record, config=config, codon_weights=codon_weights)
        optimized.append(rec)
        results.append(result)
    return optimized, results


def summarize_results(results: Sequence[CodonLatticeDPResult]) -> dict[str, object]:
    """Summarize codon-lattice DP outputs for reports.

    Complexity is ``O(N)`` over records.
    """
    if not results:
        return {
            "n": 0,
            "mean_codon_changes": 0.0,
            "mean_delta_cai": 0.0,
            "mean_delta_gc": 0.0,
            "protein_identity_fraction": 0.0,
        }
    return {
        "n": len(results),
        "mean_codon_changes": float(sum(r.codon_changes for r in results) / len(results)),
        "mean_source_cai": float(sum(r.source_cai for r in results) / len(results)),
        "mean_optimized_cai": float(sum(r.optimized_cai for r in results) / len(results)),
        "mean_delta_cai": float(sum(r.optimized_cai - r.source_cai for r in results) / len(results)),
        "mean_source_gc": float(sum(r.source_gc for r in results) / len(results)),
        "mean_optimized_gc": float(sum(r.optimized_gc for r in results) / len(results)),
        "mean_delta_gc": float(sum(r.optimized_gc - r.source_gc for r in results) / len(results)),
        "protein_identity_fraction": float(
            sum(r.source_protein == r.optimized_protein for r in results) / len(results)
        ),
    }


def run_codon_lattice_dp(
    records: Sequence[MRNARecord],
    *,
    out_jsonl: str,
    out_json: str,
    limit: Optional[int] = None,
    config: Optional[CodonLatticeDPConfig] = None,
    reference_records: Optional[Sequence[MRNARecord]] = None,
) -> dict[str, object]:
    """Run the executable codon-lattice baseline and write artifacts."""
    selected = list(records[: int(limit)]) if limit is not None else list(records)
    weights = codon_weights_from_reference(reference_records or selected)
    optimized, results = optimize_records_cds(selected, config=config, codon_weights=weights)
    write_records_jsonl(optimized, out_jsonl)
    payload = {
        "config": asdict(config or CodonLatticeDPConfig()),
        "n_records": len(selected),
        "out_jsonl": out_jsonl,
        "summary": summarize_results(results),
        "per_record": [result.to_dict() for result in results],
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return payload


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-codon-changes", type=int, default=None)
    parser.add_argument("--target-gc", type=float, default=0.55)
    parser.add_argument("--cai-weight", type=float, default=1.0)
    parser.add_argument("--gc-weight", type=float, default=0.10)
    parser.add_argument("--boundary-weight", type=float, default=0.05)
    parser.add_argument("--change-penalty", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    config = CodonLatticeDPConfig(
        cai_weight=args.cai_weight,
        gc_weight=args.gc_weight,
        target_gc=args.target_gc,
        boundary_weight=args.boundary_weight,
        change_penalty=args.change_penalty,
        max_codon_changes=args.max_codon_changes,
    )
    payload = run_codon_lattice_dp(
        load_records_jsonl(args.records_jsonl),
        out_jsonl=args.out_jsonl,
        out_json=args.out_json,
        limit=args.limit,
        config=config,
    )
    print(json.dumps({"out_json": args.out_json, "out_jsonl": args.out_jsonl, "summary": payload["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CodonLatticeDPConfig",
    "CodonLatticeDPResult",
    "boundary_stability_score",
    "optimize_cds_synonymous",
    "optimize_record_cds",
    "optimize_records_cds",
    "summarize_results",
    "run_codon_lattice_dp",
    "main",
]
