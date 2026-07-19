"""Predictor-guided 5'UTR local-search baseline.

This module implements an executable UTR-only baseline in the spirit of
Optimus+GA and UTailoR-style predictor-guided optimization. It is intentionally
not a claim that those external systems have been reproduced. Instead, it gives
the MEF paper pipeline a transparent, no-network local-search comparator that:

* edits only the 5'UTR,
* preserves the original CDS and 3'UTR exactly,
* scores candidates with the independent :class:`LocalTranslationOracle`, and
* writes auditable JSON/JSONL artifacts for the SOTA gap report.

For source 5'UTR ``x`` and candidate ``y``, the search maximizes

``F(y) = TE_oracle(y, c) - alpha (GC(y)-g*)^2 - beta uAUG(y)``

subject to an edit budget ``d_edit(x, y) <= K`` and optional length-delta bound
``|len(y)-len(x)| <= D``. ``c`` is the first CDS context used by the oracle.
The optimizer uses beam search over substitution/insertion/deletion operators:

``B_0 = {x}``

``B_t = top_B({ op(y) : y in B_{t-1}, op in O_5UTR(y) })``

and returns the best candidate observed across ``B_0 ... B_K``. With beam width
``B``, budget ``K``, selected editable positions ``P`` and oracle feature cost
``Q``, time is ``O(K * B * P * Q)`` and memory is ``O(B * L)`` for UTR length
``L``. The implementation selects biologically salient positions around the
start codon and known inhibitory motifs to keep long public transcripts
tractable while remaining deterministic and fully auditable.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional, Sequence

from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.download_mrna import load_records_jsonl, write_records_jsonl
from mrna_editflow.eval.metrics import edit_distance, gc_fraction, normalize_rna
from mrna_editflow.eval.oracle import LocalTranslationOracle

RNA_ALPHABET = ("A", "C", "G", "U")
DEFAULT_BAD_UTR_MOTIFS = ("AUG", "UAA", "UAG", "UGA", "GGGG", "CCCC")


@dataclass(frozen=True)
class UTRLocalSearchConfig:
    """Configuration for the 5'UTR local-search baseline.

    ``edit_budget`` bounds the number of elementary edit operators considered.
    ``beam_width`` controls the number of candidates retained after each depth.
    ``start_window_nt`` prioritizes edits near the CDS start where Kozak and
    accessibility features are most sensitive. ``max_edit_positions`` caps the
    number of positions expanded per state; motif positions are kept first and
    then positions nearest the start codon are added.

    Construction complexity is ``O(1)``.
    """

    edit_budget: int = 3
    beam_width: int = 32
    max_length_delta: int = 8
    start_window_nt: int = 90
    max_edit_positions: int = 120
    gc_target: float = 0.52
    gc_penalty_weight: float = 0.0
    uaug_penalty_weight: float = 0.0
    allow_substitution: bool = True
    allow_insertion: bool = True
    allow_deletion: bool = True
    bad_motifs: tuple[str, ...] = DEFAULT_BAD_UTR_MOTIFS


@dataclass(frozen=True)
class UTRLocalSearchResult:
    """Result from optimizing one transcript's 5'UTR.

    ``utr_edit_distance`` is the exact Levenshtein distance between source and
    optimized 5'UTRs. ``search_steps`` is the beam depth at which the returned
    candidate was found. Since every path applies one elementary edit per depth,
    the returned candidate always satisfies ``utr_edit_distance <= search_steps``.

    Conversion to dict is ``O(path_length + sequence_length)``.
    """

    transcript_id: str
    source_five_utr: str
    optimized_five_utr: str
    source_te: float
    optimized_te: float
    source_mrl: float
    optimized_mrl: float
    source_objective: float
    optimized_objective: float
    delta_te: float
    delta_mrl: float
    utr_edit_distance: int
    length_delta: int
    search_steps: int
    operations: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping. Complexity is ``O(fields)``."""
        payload = dict(asdict(self))
        payload["operations"] = list(self.operations)
        return payload


@dataclass(frozen=True)
class _SearchState:
    """Internal beam-search state.

    ``score`` is the objective ``F``. ``te`` and ``mrl`` are stored separately
    so summaries can report oracle improvements without re-scoring.
    """

    seq: str
    score: float
    te: float
    mrl: float
    steps: int
    operations: tuple[str, ...]


_WORKER_CONFIG: Optional[UTRLocalSearchConfig] = None
_WORKER_ORACLE: Optional[LocalTranslationOracle] = None


def _init_search_worker(config: UTRLocalSearchConfig) -> None:
    """Initialize one reusable oracle per local-search worker."""
    global _WORKER_CONFIG, _WORKER_ORACLE
    _WORKER_CONFIG = config
    _WORKER_ORACLE = LocalTranslationOracle()


def _optimize_record_worker(
    record: MRNARecord,
) -> tuple[MRNARecord, UTRLocalSearchResult]:
    """Optimize one record inside a process-pool worker."""
    if _WORKER_CONFIG is None or _WORKER_ORACLE is None:
        raise RuntimeError("UTR local-search worker was not initialized")
    return optimize_record_five_utr(
        record,
        config=_WORKER_CONFIG,
        oracle=_WORKER_ORACLE,
    )


def _normalise_utr(seq: str) -> str:
    """Normalize a 5'UTR and remove invalid characters.

    The public pipeline should already emit clean RNA strings. This function is
    deliberately strict in output but tolerant in input, mirroring evaluation
    metrics: DNA ``T`` is converted to RNA ``U`` and non-ACGU characters are
    dropped. Complexity is ``O(L)``.
    """
    return "".join(ch for ch in normalize_rna(seq) if ch in RNA_ALPHABET)


def _find_motif_positions(seq: str, motifs: Sequence[str]) -> set[int]:
    """Return all positions covered by any motif occurrence.

    For motif length ``m`` and sequence length ``L``, each motif scan is
    ``O(L*m)`` in the worst case through Python substring search. The bounded
    motif list keeps this linear in practice for benchmark UTRs.
    """
    positions: set[int] = set()
    for motif in motifs:
        motif = _normalise_utr(motif)
        if not motif:
            continue
        start = 0
        while True:
            idx = seq.find(motif, start)
            if idx < 0:
                break
            positions.update(range(idx, idx + len(motif)))
            start = idx + 1
    return positions


def _editable_positions(seq: str, config: UTRLocalSearchConfig) -> list[int]:
    """Select deterministic biologically salient 5'UTR edit positions.

    The selected set is

    ``P = P_motif union P_start_window``.

    ``P_motif`` covers upstream AUG/stop-like/poly-GC motifs, while
    ``P_start_window`` is the last ``start_window_nt`` nucleotides before the
    CDS. If ``|P|`` exceeds ``max_edit_positions``, motif positions are retained
    first and the remaining slots are filled from positions closest to the CDS
    start. Complexity is ``O(L * M + L)`` for ``M`` motifs.
    """
    if not seq:
        return []
    n = len(seq)
    start_window = max(0, n - max(0, int(config.start_window_nt)))
    motif_positions = _find_motif_positions(seq, config.bad_motifs)
    window_positions = set(range(start_window, n))
    selected = motif_positions | window_positions
    cap = int(config.max_edit_positions)
    if cap > 0 and len(selected) > cap:
        ordered: list[int] = []
        for pos in sorted(motif_positions):
            if pos in selected and pos not in ordered:
                ordered.append(pos)
        for pos in range(n - 1, -1, -1):
            if pos in selected and pos not in ordered:
                ordered.append(pos)
            if len(ordered) >= cap:
                break
        selected = set(ordered[:cap])
    return sorted(pos for pos in selected if 0 <= pos < n)


def _insertion_slots(seq: str, positions: Sequence[int]) -> list[int]:
    """Map editable positions to insertion slots.

    A slot ``s`` inserts before ``seq[s]``; slot ``len(seq)`` appends. For every
    editable position ``p`` we include slots ``p`` and ``p+1``. Complexity is
    ``O(|P|)``.
    """
    if not seq:
        return [0]
    slots = {0, len(seq)}
    for pos in positions:
        slots.add(pos)
        slots.add(pos + 1)
    return sorted(slot for slot in slots if 0 <= slot <= len(seq))


def _objective_from_score(
    score: Mapping[str, object],
    *,
    config: UTRLocalSearchConfig,
) -> tuple[float, float, float]:
    """Convert oracle output into ``(objective, TE, MRL)``.

    The objective follows the module formula. ``ensemble_te`` and
    ``ensemble_mrl`` are used because the local oracle exposes two independent
    deterministic predictors. Complexity is ``O(1)``.
    """
    te = float(score.get("ensemble_te", score.get("te", 0.0)))
    mrl = float(score.get("ensemble_mrl", score.get("mrl", 0.0)))
    features = score.get("features", {})
    gc_value = 0.0
    uaug_count = 0.0
    if isinstance(features, Mapping):
        gc_value = float(features.get("gc", 0.0))
        uaug_count = float(features.get("uaug_count", 0.0))
    objective = (
        te
        - float(config.gc_penalty_weight) * (gc_value - float(config.gc_target)) ** 2
        - float(config.uaug_penalty_weight) * uaug_count
    )
    return float(objective), float(te), float(mrl)


def _score_state(
    seq: str,
    *,
    record: MRNARecord,
    config: UTRLocalSearchConfig,
    oracle: LocalTranslationOracle,
    steps: int,
    operations: Sequence[str],
) -> _SearchState:
    """Score a candidate UTR and return a beam state.

    The CDS start context is fixed to the source record's first four codons,
    so the baseline cannot alter CDS-derived Kozak evidence. Complexity is the
    oracle feature cost, ``O(len(seq))`` for the deterministic fallback.
    """
    score = oracle.score_utr(seq, record.cds[:12])
    objective, te, mrl = _objective_from_score(score, config=config)
    return _SearchState(
        seq=seq,
        score=objective,
        te=te,
        mrl=mrl,
        steps=int(steps),
        operations=tuple(operations),
    )


def _candidate_sequences(
    state: _SearchState,
    *,
    source_len: int,
    config: UTRLocalSearchConfig,
) -> list[tuple[str, str]]:
    """Expand one beam state by legal UTR edit operations.

    Returned pairs are ``(new_sequence, operation_label)``. The operator set is
    substitution, insertion and deletion constrained by ``max_length_delta``.
    Complexity is ``O(P * |alphabet|)`` for selected positions ``P``.
    """
    seq = state.seq
    positions = _editable_positions(seq, config)
    out: list[tuple[str, str]] = []
    max_delta = max(0, int(config.max_length_delta))
    if config.allow_substitution:
        for pos in positions:
            old = seq[pos]
            for nt in RNA_ALPHABET:
                if nt != old:
                    out.append((seq[:pos] + nt + seq[pos + 1:], f"sub:{pos}:{old}>{nt}"))
    if config.allow_insertion and abs((len(seq) + 1) - source_len) <= max_delta:
        for slot in _insertion_slots(seq, positions):
            for nt in RNA_ALPHABET:
                out.append((seq[:slot] + nt + seq[slot:], f"ins:{slot}:{nt}"))
    if config.allow_deletion and seq and abs((len(seq) - 1) - source_len) <= max_delta:
        for pos in positions:
            out.append((seq[:pos] + seq[pos + 1:], f"del:{pos}:{seq[pos]}"))
    return out


def _state_sort_key(state: _SearchState) -> tuple[float, float, int, str]:
    """Sort states by objective, TE, shorter edit path and sequence.

    Higher objective and TE are better; fewer steps breaks ties toward simpler
    candidates. Complexity is ``O(1)``.
    """
    return (state.score, state.te, -state.steps, state.seq)


def optimize_record_five_utr(
    record: MRNARecord,
    *,
    config: Optional[UTRLocalSearchConfig] = None,
    oracle: Optional[LocalTranslationOracle] = None,
) -> tuple[MRNARecord, UTRLocalSearchResult]:
    """Optimize only the 5'UTR of one transcript with beam search.

    The algorithm never edits ``record.cds`` or ``record.three_utr``. It scores
    the source UTR, expands up to ``edit_budget`` beam layers, deduplicates by
    sequence, and returns the best objective seen at any depth. Complexity is
    ``O(K * B * P * Q)`` as described in the module docstring.
    """
    cfg = config or UTRLocalSearchConfig()
    if cfg.edit_budget < 0:
        raise ValueError("edit_budget must be non-negative")
    if cfg.beam_width <= 0:
        raise ValueError("beam_width must be positive")
    pred = oracle or LocalTranslationOracle()
    source_utr = _normalise_utr(record.five_utr)
    source_record = MRNARecord(
        transcript_id=record.transcript_id,
        five_utr=source_utr,
        cds=record.cds,
        three_utr=record.three_utr,
        species=record.species,
    )
    source_state = _score_state(
        source_utr,
        record=source_record,
        config=cfg,
        oracle=pred,
        steps=0,
        operations=(),
    )
    best = source_state
    beam = [source_state]
    seen_global = {source_utr}
    for depth in range(1, int(cfg.edit_budget) + 1):
        scored: list[_SearchState] = []
        seen_layer: set[str] = set()
        for state in beam:
            for seq, op in _candidate_sequences(state, source_len=len(source_utr), config=cfg):
                if seq in seen_layer:
                    continue
                seen_layer.add(seq)
                scored.append(
                    _score_state(
                        seq,
                        record=source_record,
                        config=cfg,
                        oracle=pred,
                        steps=depth,
                        operations=state.operations + (op,),
                    )
                )
        if not scored:
            break
        scored.sort(key=_state_sort_key, reverse=True)
        beam = []
        for state in scored:
            if state.seq in seen_global:
                continue
            beam.append(state)
            seen_global.add(state.seq)
            if len(beam) >= int(cfg.beam_width):
                break
        if not beam:
            break
        layer_best = max(beam, key=_state_sort_key)
        if _state_sort_key(layer_best) > _state_sort_key(best):
            best = layer_best

    optimized = MRNARecord(
        transcript_id=record.transcript_id,
        five_utr=best.seq,
        cds=record.cds,
        three_utr=record.three_utr,
        species=record.species,
    )
    dist = edit_distance(source_utr, best.seq)
    result = UTRLocalSearchResult(
        transcript_id=record.transcript_id,
        source_five_utr=source_utr,
        optimized_five_utr=best.seq,
        source_te=source_state.te,
        optimized_te=best.te,
        source_mrl=source_state.mrl,
        optimized_mrl=best.mrl,
        source_objective=source_state.score,
        optimized_objective=best.score,
        delta_te=best.te - source_state.te,
        delta_mrl=best.mrl - source_state.mrl,
        utr_edit_distance=int(dist),
        length_delta=len(best.seq) - len(source_utr),
        search_steps=best.steps,
        operations=best.operations,
    )
    if result.utr_edit_distance > int(cfg.edit_budget):
        raise RuntimeError("UTR local-search invariant violated: edit budget exceeded")
    return optimized, result


def optimize_records_five_utr(
    records: Sequence[MRNARecord],
    *,
    config: Optional[UTRLocalSearchConfig] = None,
    oracle: Optional[LocalTranslationOracle] = None,
    workers: int = 1,
) -> tuple[list[MRNARecord], list[UTRLocalSearchResult]]:
    """Optimize a batch of records. Complexity is the sum of per-record costs."""
    worker_count = max(1, int(workers))
    cfg = config or UTRLocalSearchConfig()
    if worker_count > 1:
        if oracle is not None:
            raise ValueError("parallel UTR local search does not accept a custom oracle")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_search_worker,
            initargs=(cfg,),
        ) as pool:
            pairs = list(pool.map(_optimize_record_worker, records, chunksize=1))
        return (
            [record for record, _result in pairs],
            [result for _record, result in pairs],
        )
    pred = oracle or LocalTranslationOracle()
    optimized: list[MRNARecord] = []
    results: list[UTRLocalSearchResult] = []
    for record in records:
        out_record, result = optimize_record_five_utr(record, config=cfg, oracle=pred)
        optimized.append(out_record)
        results.append(result)
    return optimized, results


def summarize_results(
    results: Sequence[UTRLocalSearchResult],
    *,
    optimized_records: Optional[Sequence[MRNARecord]] = None,
    source_records: Optional[Sequence[MRNARecord]] = None,
) -> dict[str, object]:
    """Summarize UTR local-search results for reports.

    The reported mean delta is ``mean(TE_optimized - TE_source)``. Constraint
    fractions verify the baseline stayed in the UTR-only setting. Complexity is
    ``O(N)`` for ``N`` records.
    """
    if not results:
        return {
            "n": 0,
            "mean_source_te": 0.0,
            "mean_optimized_te": 0.0,
            "mean_delta_te": 0.0,
            "mean_edit_distance": 0.0,
            "cds_unchanged_fraction": 1.0,
            "three_utr_unchanged_fraction": 1.0,
        }
    summary = {
        "n": len(results),
        "mean_source_te": float(sum(r.source_te for r in results) / len(results)),
        "mean_optimized_te": float(sum(r.optimized_te for r in results) / len(results)),
        "mean_delta_te": float(sum(r.delta_te for r in results) / len(results)),
        "mean_source_mrl": float(sum(r.source_mrl for r in results) / len(results)),
        "mean_optimized_mrl": float(sum(r.optimized_mrl for r in results) / len(results)),
        "mean_delta_mrl": float(sum(r.delta_mrl for r in results) / len(results)),
        "mean_edit_distance": float(sum(r.utr_edit_distance for r in results) / len(results)),
        "mean_length_delta": float(sum(r.length_delta for r in results) / len(results)),
        "mean_source_gc": float(sum(gc_fraction(r.source_five_utr) for r in results) / len(results)),
        "mean_optimized_gc": float(sum(gc_fraction(r.optimized_five_utr) for r in results) / len(results)),
    }
    if optimized_records is not None and source_records is not None:
        pairs = list(zip(source_records, optimized_records))
        summary["cds_unchanged_fraction"] = float(
            sum(src.cds == out.cds for src, out in pairs) / max(1, len(pairs))
        )
        summary["three_utr_unchanged_fraction"] = float(
            sum(src.three_utr == out.three_utr for src, out in pairs) / max(1, len(pairs))
        )
    else:
        summary["cds_unchanged_fraction"] = 1.0
        summary["three_utr_unchanged_fraction"] = 1.0
    return summary


def run_utr_local_search(
    records: Sequence[MRNARecord],
    *,
    out_jsonl: str,
    out_json: str,
    limit: Optional[int] = None,
    config: Optional[UTRLocalSearchConfig] = None,
    workers: int = 1,
) -> dict[str, object]:
    """Run the executable UTR local-search baseline and write artifacts.

    ``out_jsonl`` stores optimized transcript records. ``out_json`` stores the
    config, aggregate summary and per-record audit trail. Complexity is the sum
    of per-record beam-search costs plus ``O(N)`` serialization.
    """
    start = time.perf_counter()
    selected = list(records[: int(limit)]) if limit is not None else list(records)
    cfg = config or UTRLocalSearchConfig()
    worker_count = max(1, int(workers))
    optimized, results = optimize_records_five_utr(
        selected,
        config=cfg,
        workers=worker_count,
    )
    elapsed_s = float(time.perf_counter() - start)
    write_records_jsonl(optimized, out_jsonl)
    payload = {
        "config": asdict(cfg),
        "n_records": len(selected),
        "out_jsonl": out_jsonl,
        "runtime": {
            "elapsed_s": elapsed_s,
            "mean_wall_clock_s": elapsed_s / max(1, len(selected)),
            "workers": worker_count,
        },
        "summary": summarize_results(
            results,
            optimized_records=optimized,
            source_records=selected,
        ),
        "per_record": [result.to_dict() for result in results],
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return payload


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments. Complexity is ``O(number_of_args)``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--edit-budget", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--max-length-delta", type=int, default=8)
    parser.add_argument("--start-window-nt", type=int, default=90)
    parser.add_argument("--max-edit-positions", type=int, default=120)
    parser.add_argument("--gc-target", type=float, default=0.52)
    parser.add_argument("--gc-penalty-weight", type=float, default=0.0)
    parser.add_argument("--uaug-penalty-weight", type=float, default=0.0)
    parser.add_argument("--disable-insertion", action="store_true")
    parser.add_argument("--disable-deletion", action="store_true")
    parser.add_argument("--disable-substitution", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Complexity is dominated by :func:`run_utr_local_search`."""
    args = _parse_args(argv)
    config = UTRLocalSearchConfig(
        edit_budget=args.edit_budget,
        beam_width=args.beam_width,
        max_length_delta=args.max_length_delta,
        start_window_nt=args.start_window_nt,
        max_edit_positions=args.max_edit_positions,
        gc_target=args.gc_target,
        gc_penalty_weight=args.gc_penalty_weight,
        uaug_penalty_weight=args.uaug_penalty_weight,
        allow_substitution=not args.disable_substitution,
        allow_insertion=not args.disable_insertion,
        allow_deletion=not args.disable_deletion,
    )
    payload = run_utr_local_search(
        load_records_jsonl(args.records_jsonl),
        out_jsonl=args.out_jsonl,
        out_json=args.out_json,
        limit=args.limit,
        config=config,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "out_jsonl": args.out_jsonl,
                "summary": payload["summary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "UTRLocalSearchConfig",
    "UTRLocalSearchResult",
    "optimize_record_five_utr",
    "optimize_records_five_utr",
    "run_utr_local_search",
    "summarize_results",
    "main",
]
