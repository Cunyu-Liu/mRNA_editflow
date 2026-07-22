"""Model evaluation adapter for offline proposal-ranker validation."""
from __future__ import annotations

from typing import Mapping, Sequence

import torch

from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.rl.ranking_metrics import RankingCandidate, compute_ranking_metrics


def evaluate_ranker_validation(
    *,
    records_by_id: Mapping[str, MRNARecord],
    teachers_by_id: Mapping[str, Sequence[object]],
    score_row,
    forward_record,
    model,
    backbone,
    device: torch.device,
    candidate_cap: int = 0,
) -> dict[str, object]:
    """Score validation candidates with no gradient updates and aggregate metrics."""
    was_training = bool(model.training)
    model.eval()
    candidates: dict[str, list[RankingCandidate]] = {}
    try:
        with torch.no_grad():
            for transcript_id in sorted(teachers_by_id):
                record = records_by_id.get(transcript_id)
                if record is None:
                    continue
                out = forward_record(record, model, backbone, device)
                candidates[transcript_id] = [
                    RankingCandidate(
                        op=str(row.op),
                        teacher_delta=float(row.teacher_score),
                        model_score=float(score_row(out, row).detach().float().cpu()),
                    )
                    for row in teachers_by_id[transcript_id]
                ]
    finally:
        model.train(was_training)
    return compute_ranking_metrics(candidates, candidate_cap=candidate_cap)


__all__ = ["evaluate_ranker_validation"]
