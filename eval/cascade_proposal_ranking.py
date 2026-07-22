"""Two-stage proposal-ranking cascade audit for mRNA-EditFlow.

The source-aware hybrid ranker improves top-k recall of oracle-best proposals,
while the full-pool/sequential rankers improve top-1 precision. This module
audits the natural two-stage pipeline:

1. A recall checkpoint ranks the full legal proposal pool ``C(x)`` and keeps
   ``K`` candidates.
2. A precision checkpoint reranks only that retained set and chooses top-1.

For source ``x`` and proposal ``c``, each model score is the CTMC intensity

``s_M(c)=lambda_op(i|x,t=0.5) p_op(a|i,x,t=0.5)``.

The cascade prediction is

``c* = argmax_{c in TopK_recall(C(x))} s_precision(c)``.

The main metric is full-pool regret

``R_cascade(x)=max_{c in C(x)} TE(c) - TE(c*)``.

This measures whether a high-recall model can expose the oracle-best region of
the legal edit space while a precision model improves the final top-1 choice.
Complexity per record is two transformer forward passes plus oracle scoring of
the proposal pool: ``O(2*model_forward + |C|*oracle_cost + |C|log|C|)``.
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
from mrna_editflow.eval.proposal_ranking import (
    _as_float,
    _mean,
    _proposal_pool,
    _rank_positions,
    _te_from_score,
)
from mrna_editflow.sample import _model_out_for_record, load_stage_a_checkpoint
from mrna_editflow.rl.action_scoring import action_log_score_float


@dataclass(frozen=True)
class CascadeCandidate:
    """One legal candidate scored by recall model, precision model and oracle."""

    transcript_id: str
    task_id: str
    op: str
    pos: int
    nt: str
    recall_score: float
    precision_score: float
    source_te: float
    oracle_te: float
    delta_te: float
    recall_rank: int
    precision_rank_full: int
    precision_rank_in_recall: int
    oracle_rank: int
    in_recall_top_k: bool
    candidate: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping. Complexity is ``O(fields)``."""
        return dict(asdict(self))


def _model_score_for_op(out: Mapping[str, object], op: str, pos: int, nt: str) -> float:
    """Return the shared decoder/ranker log score for one proposal."""
    return action_log_score_float(out, op, pos, nt or None)


def summarise_cascade_record(
    transcript_id: str,
    rows: Sequence[CascadeCandidate],
    *,
    recall_top_k: int,
    source_te: float = 0.0,
) -> dict[str, object]:
    """Summarize one record's two-stage ranking outcome.

    ``recall_top_k<=0`` means the precision model sees the full pool. Complexity
    is ``O(|C|)`` because ranks are precomputed.
    """
    if not rows:
        return {
            "transcript_id": transcript_id,
            "n_candidates": 0,
            "recall_top_k": int(recall_top_k),
            "source_te": float(source_te),
            "cascade_top_te": float(source_te),
            "recall_top_te": float(source_te),
            "precision_full_top_te": float(source_te),
            "oracle_top_te": float(source_te),
            "cascade_regret": 0.0,
            "recall_model_regret": 0.0,
            "precision_full_regret": 0.0,
            "oracle_best_in_recall_top_k": False,
        }
    recall_top = min(rows, key=lambda row: row.recall_rank)
    precision_full_top = min(rows, key=lambda row: row.precision_rank_full)
    oracle_top = min(rows, key=lambda row: row.oracle_rank)
    retained = [row for row in rows if row.in_recall_top_k]
    cascade_top = min(retained, key=lambda row: row.precision_rank_in_recall) if retained else recall_top
    return {
        "transcript_id": transcript_id,
        "n_candidates": len(rows),
        "recall_top_k": int(recall_top_k),
        "source_te": float(recall_top.source_te),
        "cascade_top_te": float(cascade_top.oracle_te),
        "recall_top_te": float(recall_top.oracle_te),
        "precision_full_top_te": float(precision_full_top.oracle_te),
        "oracle_top_te": float(oracle_top.oracle_te),
        "cascade_regret": float(oracle_top.oracle_te - cascade_top.oracle_te),
        "recall_model_regret": float(oracle_top.oracle_te - recall_top.oracle_te),
        "precision_full_regret": float(oracle_top.oracle_te - precision_full_top.oracle_te),
        "oracle_best_in_recall_top_k": bool(oracle_top.in_recall_top_k),
        "cascade_top_recall_rank": int(cascade_top.recall_rank),
        "cascade_top_precision_full_rank": int(cascade_top.precision_rank_full),
        "oracle_best_recall_rank": int(oracle_top.recall_rank),
    }


def _aggregate_record_summaries(record_summaries: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate cascade record summaries. Complexity is ``O(N)``."""
    with_candidates = [row for row in record_summaries if _as_float(row.get("n_candidates")) > 0.0]
    n_candidates = [_as_float(row.get("n_candidates")) for row in record_summaries]
    hit = [1.0 if row.get("oracle_best_in_recall_top_k") else 0.0 for row in with_candidates]
    return {
        "n_records": len(record_summaries),
        "n_records_with_candidates": len(with_candidates),
        "n_candidates": int(sum(n_candidates)),
        "mean_source_te": _mean([_as_float(row.get("source_te")) for row in record_summaries]),
        "mean_cascade_top_te": _mean([_as_float(row.get("cascade_top_te")) for row in record_summaries]),
        "mean_recall_top_te": _mean([_as_float(row.get("recall_top_te")) for row in record_summaries]),
        "mean_precision_full_top_te": _mean([_as_float(row.get("precision_full_top_te")) for row in record_summaries]),
        "mean_oracle_top_te": _mean([_as_float(row.get("oracle_top_te")) for row in record_summaries]),
        "mean_cascade_regret": _mean([_as_float(row.get("cascade_regret")) for row in record_summaries]),
        "mean_recall_model_regret": _mean([_as_float(row.get("recall_model_regret")) for row in record_summaries]),
        "mean_precision_full_regret": _mean([_as_float(row.get("precision_full_regret")) for row in record_summaries]),
        "oracle_best_in_recall_top_k_fraction": _mean(hit),
        "candidate_pool_mean": _mean(n_candidates),
        "candidate_pool_max": float(max(n_candidates)) if n_candidates else 0.0,
    }


def score_record_cascade(
    record: MRNARecord,
    recall_model,
    recall_backbone,
    precision_model,
    precision_backbone,
    *,
    task_id: str = "T5",
    oracle: Optional[object] = None,
    device: Optional[str] = None,
    target_length: Optional[int] = None,
    candidate_cap: int = 0,
    recall_top_k: int = 32,
) -> tuple[list[CascadeCandidate], dict[str, object]]:
    """Score one record with recall and precision rankers."""
    pred = oracle if oracle is not None else LocalTranslationOracle()
    if not hasattr(pred, "score_record"):
        raise TypeError("oracle must provide score_record(record)")
    dev = device or next(recall_model.parameters()).device
    recall_out = _model_out_for_record(record, recall_model, recall_backbone, dev)
    precision_out = _model_out_for_record(record, precision_model, precision_backbone, dev)
    raw_pool = _proposal_pool(record, recall_out, task_id, target_length)
    raw_pool.sort(key=lambda item: item[1], reverse=True)
    if int(candidate_cap) > 0:
        raw_pool = raw_pool[: max(1, min(int(candidate_cap), len(raw_pool)))]
    source_te = _te_from_score(pred.score_record(record))  # type: ignore[attr-defined]
    recall_scores = [float(row[1]) for row in raw_pool]
    precision_scores = [
        _model_score_for_op(precision_out, op, pos, nt)
        for op, _recall_score, pos, nt, _cand in raw_pool
    ]
    oracle_tes = [_te_from_score(pred.score_record(row[4])) for row in raw_pool]  # type: ignore[attr-defined]
    recall_ranks = _rank_positions(recall_scores, descending=True)
    precision_full_ranks = _rank_positions(precision_scores, descending=True)
    oracle_ranks = _rank_positions(oracle_tes, descending=True)
    k = len(raw_pool) if int(recall_top_k) <= 0 else max(1, min(int(recall_top_k), len(raw_pool)))
    retained_indices = {idx for idx, rank in recall_ranks.items() if rank <= k}
    retained_precision_values = [
        precision_scores[idx] if idx in retained_indices else float("-inf")
        for idx in range(len(raw_pool))
    ]
    precision_in_recall_ranks = _rank_positions(retained_precision_values, descending=True)
    rows = [
        CascadeCandidate(
            transcript_id=record.transcript_id,
            task_id=task_id.upper(),
            op=str(op),
            pos=int(pos),
            nt=str(nt),
            recall_score=float(recall_score),
            precision_score=float(precision_scores[i]),
            source_te=float(source_te),
            oracle_te=float(oracle_tes[i]),
            delta_te=float(oracle_tes[i] - source_te),
            recall_rank=int(recall_ranks[i]),
            precision_rank_full=int(precision_full_ranks[i]),
            precision_rank_in_recall=int(precision_in_recall_ranks[i]),
            oracle_rank=int(oracle_ranks[i]),
            in_recall_top_k=bool(i in retained_indices),
            candidate=cand.to_dict(),
        )
        for i, (op, recall_score, pos, nt, cand) in enumerate(raw_pool)
    ]
    rows.sort(key=lambda row: row.recall_rank)
    return rows, summarise_cascade_record(
        record.transcript_id,
        rows,
        recall_top_k=recall_top_k,
        source_te=source_te,
    )


def run_cascade_proposal_ranking_audit(
    records: Sequence[MRNARecord],
    *,
    recall_checkpoint: str,
    precision_checkpoint: str,
    out_json: str,
    out_jsonl: Optional[str] = None,
    task_id: str = "T5",
    limit: Optional[int] = None,
    device: Optional[str] = None,
    candidate_cap: int = 0,
    recall_top_k: int = 32,
    target_length_delta: int = 0,
    oracle: Optional[object] = None,
) -> dict[str, object]:
    """Run two-stage cascade audit and write JSON/JSONL artifacts."""
    _recall_cfg, recall_backbone, recall_model = load_stage_a_checkpoint(recall_checkpoint, device=device)
    _precision_cfg, precision_backbone, precision_model = load_stage_a_checkpoint(precision_checkpoint, device=device)
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
            rows, summary = score_record_cascade(
                rec,
                recall_model,
                recall_backbone,
                precision_model,
                precision_backbone,
                task_id=task_id,
                oracle=pred,
                device=device,
                target_length=target_length,
                candidate_cap=candidate_cap,
                recall_top_k=recall_top_k,
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
            "recall_checkpoint": recall_checkpoint,
            "precision_checkpoint": precision_checkpoint,
            "task_id": task_id.upper(),
            "limit": limit,
            "device": device,
            "candidate_cap": int(candidate_cap),
            "recall_top_k": int(recall_top_k),
            "target_length_delta": int(target_length_delta),
            "out_jsonl": out_jsonl,
        },
        "aggregate": _aggregate_record_summaries(record_summaries),
        "per_record": record_summaries,
    }
    result["aggregate"]["n_candidates"] = candidate_count
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    return result


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit two-stage recall/precision proposal ranking")
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--recall-checkpoint", required=True)
    parser.add_argument("--precision-checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-jsonl", default=None)
    parser.add_argument("--task-id", default="T5")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--candidate-cap", type=int, default=0)
    parser.add_argument("--recall-top-k", type=int, default=32)
    parser.add_argument("--target-length-delta", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    result = run_cascade_proposal_ranking_audit(
        load_records_jsonl(args.records_jsonl),
        recall_checkpoint=args.recall_checkpoint,
        precision_checkpoint=args.precision_checkpoint,
        out_json=args.out_json,
        out_jsonl=args.out_jsonl,
        task_id=args.task_id,
        limit=args.limit,
        device=args.device,
        candidate_cap=args.candidate_cap,
        recall_top_k=args.recall_top_k,
        target_length_delta=args.target_length_delta,
    )
    print(json.dumps({"out_json": args.out_json, "aggregate": result["aggregate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CascadeCandidate",
    "run_cascade_proposal_ranking_audit",
    "score_record_cascade",
    "summarise_cascade_record",
    "main",
]
