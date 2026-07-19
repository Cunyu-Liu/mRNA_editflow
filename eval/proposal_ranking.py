"""Audit model ranking of legal mRNA edit proposals.

The oracle-guided ablation shows that high-TE neighbours exist inside the hard
constraint set, but the model-only decoder may fail to rank them highly. This
module builds the missing evidence layer: for each source record it enumerates
legal one-step proposals, scores them by the MEF CTMC field and by an
independent TE oracle, then reports the ranking gap.

Mathematical protocol
---------------------
For a source record ``x`` and legal proposal set ``C(x)``, the model assigns

``s_M(c) = lambda_op(i | x, t=0.5) p_op(a | i, x, t=0.5)``,

while the teacher oracle assigns

``s_T(c) = TE(c) - TE(x)``.

The core regret is

``R(x) = max_c TE(c) - TE(argmax_c s_M(c))``.

``R=0`` means the model top-1 proposal is also oracle-optimal within the
enumerated pool. The JSONL rows include ``teacher_score=s_T`` and
``student_score=log(max(s_M, eps))`` so the same artifact can supervise a
pairwise distillation objective

``log(1 + exp(-sign(s_T(i)-s_T(j)) * (s_M(i)-s_M(j))))``.

Complexity per record is one transformer forward plus oracle scoring for the
candidate pool: ``O(model_forward + |C| * oracle_cost)``. For T5, the default
pool is UTR substitutions only, ``|C| <= 3 * (|5UTR| + |3UTR|)``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.download_mrna import load_records_jsonl
from mrna_editflow.eval.oracle import LocalTranslationOracle
from mrna_editflow.sample import (
    _delete_nt,
    _insert_nt_after,
    _model_out_for_record,
    _replace_nt,
    _synonymous_substitution_candidates,
    _utr_delete_candidates,
    _utr_insert_candidates,
    _utr_substitution_candidates,
    load_stage_a_checkpoint,
)


@dataclass(frozen=True)
class ProposalCandidate:
    """One materialised edit proposal with model and oracle scores."""

    transcript_id: str
    task_id: str
    op: str
    pos: int
    nt: str
    model_score: float
    source_te: float
    oracle_te: float
    delta_te: float
    model_rank: int
    oracle_rank: int
    student_score: float
    teacher_score: float
    candidate: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _te_from_score(score: Mapping[str, object]) -> float:
    return _as_float(score.get("ensemble_te", score.get("te", 0.0)))


def _proposal_pool(
    record: MRNARecord,
    out: Mapping[str, object],
    task_id: str,
    target_length: Optional[int],
) -> list[tuple[str, float, int, str, MRNARecord]]:
    """Return legal one-step proposals materialised as records.

    Each tuple is ``(op, model_score, pos, nt, candidate_record)``. The model
    score is the CTMC operation intensity already computed by the sampler.
    """
    tid = task_id.upper()
    suffix = f"{record.transcript_id}_{tid.lower()}_proposal"
    rows: list[tuple[str, float, int, str, MRNARecord]] = []
    if tid in {"T2", "T3", "T5"}:
        for score, pos, nt in _utr_substitution_candidates(record, out):
            rows.append(("sub", float(score), int(pos), str(nt), _replace_nt(record, pos, nt, suffix)))
        return rows
    if tid == "T4":
        for score, pos, nt in _synonymous_substitution_candidates(record, out):
            rows.append(("sub", float(score), int(pos), str(nt), _replace_nt(record, pos, nt, suffix)))
        return rows
    if tid == "T6":
        if target_length is None:
            raise ValueError("T6 proposal audit requires target_length")
        delta = int(target_length) - len(record.seq)
        if delta == 0:
            return rows
        if delta > 0:
            for score, pos, nt in _utr_insert_candidates(record, out):
                rows.append(("ins", float(score), int(pos), str(nt), _insert_nt_after(record, pos, nt, suffix)))
        else:
            for score, pos, nt in _utr_delete_candidates(record, out):
                rows.append(("del", float(score), int(pos), "", _delete_nt(record, pos, suffix)))
        return rows
    raise ValueError("proposal ranking audit supports T2/T3/T4/T5/T6")


def _rank_positions(values: Sequence[float], *, descending: bool = True) -> dict[int, int]:
    order = sorted(
        range(len(values)),
        key=lambda i: ((-values[i] if descending else values[i]), i),
    )
    return {idx: rank + 1 for rank, idx in enumerate(order)}


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _min(values: Sequence[float]) -> float:
    return float(min(values)) if values else 0.0


def _max(values: Sequence[float]) -> float:
    return float(max(values)) if values else 0.0


def summarise_record_rows(
    transcript_id: str,
    rows: Sequence[ProposalCandidate],
    *,
    top_k: int,
    source_te: float = 0.0,
) -> dict[str, object]:
    """Summarise one source record's ranking gap.

    ``top_k`` measures whether the oracle-best candidate is reachable by a
    model top-k decoder. Complexity is ``O(|C|)`` because ranks are precomputed.
    """
    if not rows:
        return {
            "transcript_id": transcript_id,
            "n_candidates": 0,
            "source_te": float(source_te),
            "model_top_te": float(source_te),
            "oracle_top_te": float(source_te),
            "model_regret": 0.0,
            "oracle_best_model_rank": 0,
            "model_top_oracle_rank": 0,
            "oracle_best_in_model_top_k": False,
        }
    model_top = min(rows, key=lambda r: r.model_rank)
    oracle_top = min(rows, key=lambda r: r.oracle_rank)
    k = len(rows) if int(top_k) <= 0 else max(1, int(top_k))
    return {
        "transcript_id": transcript_id,
        "n_candidates": len(rows),
        "source_te": float(model_top.source_te),
        "model_top_te": float(model_top.oracle_te),
        "oracle_top_te": float(oracle_top.oracle_te),
        "model_regret": float(oracle_top.oracle_te - model_top.oracle_te),
        "oracle_best_model_rank": int(oracle_top.model_rank),
        "model_top_oracle_rank": int(model_top.oracle_rank),
        "oracle_best_in_model_top_k": bool(oracle_top.model_rank <= k),
    }


def score_record_proposals(
    record: MRNARecord,
    model,
    backbone,
    *,
    task_id: str = "T5",
    oracle: Optional[object] = None,
    device: Optional[str] = None,
    target_length: Optional[int] = None,
    candidate_cap: int = 0,
    top_k: int = 32,
) -> tuple[list[ProposalCandidate], dict[str, object]]:
    """Score one source record's legal one-step proposal pool."""
    pred = oracle if oracle is not None else LocalTranslationOracle()
    if not hasattr(pred, "score_record"):
        raise TypeError("oracle must provide score_record(record)")
    out = _model_out_for_record(record, model, backbone, device or next(model.parameters()).device)
    raw_pool = _proposal_pool(record, out, task_id, target_length)
    raw_pool.sort(key=lambda item: item[1], reverse=True)
    if int(candidate_cap) > 0:
        raw_pool = raw_pool[: max(1, min(int(candidate_cap), len(raw_pool)))]

    source_te = _te_from_score(pred.score_record(record))  # type: ignore[attr-defined]
    model_scores = [float(row[1]) for row in raw_pool]
    oracle_tes = [_te_from_score(pred.score_record(row[4])) for row in raw_pool]  # type: ignore[attr-defined]
    model_ranks = _rank_positions(model_scores, descending=True)
    oracle_ranks = _rank_positions(oracle_tes, descending=True)
    rows = [
        ProposalCandidate(
            transcript_id=record.transcript_id,
            task_id=task_id.upper(),
            op=op,
            pos=int(pos),
            nt=str(nt),
            model_score=float(model_score),
            source_te=float(source_te),
            oracle_te=float(oracle_te),
            delta_te=float(oracle_te - source_te),
            model_rank=int(model_ranks[i]),
            oracle_rank=int(oracle_ranks[i]),
            student_score=float(math.log(max(float(model_score), 1e-20))),
            teacher_score=float(oracle_te - source_te),
            candidate=cand.to_dict(),
        )
        for i, (op, model_score, pos, nt, cand) in enumerate(raw_pool)
        for oracle_te in [oracle_tes[i]]
    ]
    rows.sort(key=lambda row: row.model_rank)
    return rows, summarise_record_rows(record.transcript_id, rows, top_k=top_k, source_te=source_te)


def _aggregate_record_summaries(record_summaries: Sequence[Mapping[str, object]]) -> dict[str, object]:
    with_candidates = [
        row for row in record_summaries
        if _as_float(row.get("n_candidates")) > 0.0
    ]
    regrets = [_as_float(row.get("model_regret")) for row in record_summaries]
    hit = [1.0 if row.get("oracle_best_in_model_top_k") else 0.0 for row in with_candidates]
    n_candidates = [_as_float(row.get("n_candidates")) for row in record_summaries]
    model_top_te = [_as_float(row.get("model_top_te")) for row in record_summaries]
    oracle_top_te = [_as_float(row.get("oracle_top_te")) for row in record_summaries]
    source_te = [_as_float(row.get("source_te")) for row in record_summaries]
    return {
        "n_records": len(record_summaries),
        "n_records_with_candidates": len(with_candidates),
        "n_candidates": int(sum(n_candidates)),
        "mean_source_te": _mean(source_te),
        "mean_model_top_te": _mean(model_top_te),
        "mean_oracle_top_te": _mean(oracle_top_te),
        "mean_model_regret": _mean(regrets),
        "max_model_regret": _max(regrets),
        "oracle_best_in_model_top_k_fraction": _mean(hit),
        "candidate_pool_min": _min(n_candidates),
        "candidate_pool_mean": _mean(n_candidates),
        "candidate_pool_max": _max(n_candidates),
    }


def run_proposal_ranking_audit(
    records: Sequence[MRNARecord],
    *,
    checkpoint_path: str,
    out_json: str,
    out_jsonl: Optional[str] = None,
    task_id: str = "T5",
    limit: Optional[int] = None,
    device: Optional[str] = None,
    candidate_cap: int = 0,
    top_k: int = 32,
    target_length_delta: int = 0,
    oracle: Optional[object] = None,
) -> dict[str, object]:
    """Run a proposal ranking audit and write JSON/JSONL artifacts."""
    _cfg, backbone, model = load_stage_a_checkpoint(checkpoint_path, device=device)
    selected = list(records[: int(limit)]) if limit is not None else list(records)
    pred = oracle if oracle is not None else LocalTranslationOracle()
    record_summaries: list[dict[str, object]] = []
    candidate_count = 0
    if out_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)), exist_ok=True)
        jsonl_fh = open(out_jsonl, "w", encoding="utf-8")
    else:
        jsonl_fh = None
    try:
        for rec in selected:
            target_length = len(rec.seq) + int(target_length_delta) if task_id.upper() == "T6" else None
            rows, summary = score_record_proposals(
                rec,
                model,
                backbone,
                task_id=task_id,
                oracle=pred,
                device=device,
                target_length=target_length,
                candidate_cap=candidate_cap,
                top_k=top_k,
            )
            record_summaries.append(summary)
            candidate_count += len(rows)
            if jsonl_fh is not None:
                for row in rows:
                    jsonl_fh.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")
    finally:
        if jsonl_fh is not None:
            jsonl_fh.close()

    result: dict[str, object] = {
        "config": {
            "checkpoint_path": checkpoint_path,
            "task_id": task_id.upper(),
            "limit": limit,
            "device": device,
            "candidate_cap": int(candidate_cap),
            "top_k": int(top_k),
            "target_length_delta": int(target_length_delta),
            "out_jsonl": out_jsonl,
        },
        "aggregate": _aggregate_record_summaries(record_summaries),
        "per_record": record_summaries,
    }
    result["aggregate"]["n_candidates"] = candidate_count  # exact after filtering
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    return result


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit model ranking of legal mRNA proposals")
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-jsonl", default=None)
    parser.add_argument("--task-id", default="T5")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--candidate-cap", type=int, default=0, help="<=0 evaluates the full legal pool")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--target-length-delta", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    result = run_proposal_ranking_audit(
        load_records_jsonl(args.records_jsonl),
        checkpoint_path=args.checkpoint,
        out_json=args.out_json,
        out_jsonl=args.out_jsonl,
        task_id=args.task_id,
        limit=args.limit,
        device=args.device,
        candidate_cap=args.candidate_cap,
        top_k=args.top_k,
        target_length_delta=args.target_length_delta,
    )
    print(json.dumps({"out_json": args.out_json, "aggregate": result["aggregate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
