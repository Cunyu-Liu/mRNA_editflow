"""Cell-offset auxiliary head and HEK293FT discrimination for Critic V6.

D1 (2026-08-31 frozen): the six-cell MPRAU pair mean is the main prediction
target, but the model must not ignore the per-cell axis.  The auxiliary head
regresses each record's scaled deviation from its pair mean (target - pair
mean), conditioned on the endpoint semantics that already encode the cell
line.  The auxiliary acceptance criterion is HEK293FT distinguishability:
HEK293FT offsets correlate only 0.17-0.20 with the other five cell lines, so a
head that actually uses the cell channel must separate HEK293FT's predicted
offsets from the others (AUC over cells, rho per held-in cell).

Both the head and the discrimination probe are pure and CPU-testable.  The
head participates in training only through the frozen 32-vector detached
objective path (see ``effective_prediction_objective_v4``), so it adds a
separate scalar loss without reopening the replay machinery.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class XEditCriticCellOffsetV6Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticCellOffsetV6Error(message)


HEK293FT_CONTEXT = "HEK293FT"


class CellOffsetHeadV6(nn.Module):
    """Small MLP that maps endpoint semantics onto a per-cell offset scalar.

    The offset is a scalar added to the (antisymmetric) shared-effect
    prediction to reproduce the observed per-cell delta.  The head consumes a
    per-record condition vector (which already includes the context/cell
    embedding) and emits one offset per record.
    """

    def __init__(self, condition_width: int = 768, hidden_width: int = 256) -> None:
        super().__init__()
        _require(condition_width > 0 and hidden_width > 0, "offset head widths are invalid")
        self.condition_projection = nn.Linear(condition_width, hidden_width)
        self.activation = nn.GELU()
        self.offset_head = nn.Linear(hidden_width, 1)
        self.layer_norm = nn.LayerNorm(hidden_width)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        _require(condition.ndim == 2, "offset head condition must be batch x width")
        hidden = self.activation(self.condition_projection(condition))
        hidden = self.layer_norm(hidden)
        offset = self.offset_head(hidden).squeeze(-1)
        _require(
            bool(torch.isfinite(offset).all().item()),
            "cell-offset head output is nonfinite",
        )
        return offset


def cell_offset_loss_v6(
    offset_predictions: torch.Tensor,
    offset_targets: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
    delta: float = 1.0,
) -> dict[str, torch.Tensor | int]:
    """Weighted Huber auxiliary loss on scaled per-cell offsets (32-vector).

    ``offset_targets`` are the per-record scaled deviations from the pair
    mean: (raw_target - pair_mean) / target_scale.  Predictions come from the
    auxiliary head conditioned on endpoint semantics.  Returns the scalar loss
    and the detached 32-vector gradient used by the frozen VJP machinery.
    """

    _require(
        offset_predictions.shape == offset_targets.shape
        and offset_predictions.ndim == 1,
        "cell-offset prediction/target geometry differs",
    )
    values = offset_predictions.detach().clone().requires_grad_(True)
    per_record = F.huber_loss(
        values,
        offset_targets,
        reduction="none",
        delta=float(delta),
    )
    if sample_weights is not None:
        _require(
            sample_weights.shape == per_record.shape,
            "cell-offset sample weights are misaligned",
        )
        _require(bool((sample_weights.sum() > 0).item()), "cell-offset weights sum to zero")
        loss = (per_record * sample_weights).sum() / sample_weights.sum()
    else:
        loss = per_record.mean()
    _require(bool(torch.isfinite(loss).item()), "cell-offset loss is nonfinite")
    gradient = torch.autograd.grad(loss, values)[0].detach()
    return {
        "loss": loss.detach(),
        "gradient": gradient,
        "record_count": int(offset_targets.numel()),
    }


def cell_offset_discrimination_v6(
    predicted_offsets: Sequence[float],
    contexts: Sequence[str],
    *,
    target_context: str = HEK293FT_CONTEXT,
) -> dict[str, object]:
    """W1-b auxiliary acceptance probe: is the HEK293FT offset distinguishable?

    Returns the AUROC of treating 'predicted offset magnitude' as a HEK293FT
    vs not-HEK293FT score, plus the Spearman rho of the offset predictor across
    cells, when both are computable.  A head that ignores the cell channel
    produces AUC ~ 0.5.
    """

    from scipy.stats import spearmanr

    _require(
        len(predicted_offsets) == len(contexts) and bool(predicted_offsets),
        "cell-offset discrimination bundles are misaligned or empty",
    )
    labels = [1.0 if str(context) == target_context else 0.0 for context in contexts]
    positives = sum(labels)
    _require(
        0 < positives < len(labels),
        "target context is absent or universal in the probe bundle",
    )
    ordered = sorted(
        ((float(score), 1.0 if label else 0.0, index) for index, (score, label) in enumerate(zip(predicted_offsets, labels))),
        key=lambda item: item[0],
    )
    negative_count = len(ordered) - positives
    # Mann-Whitney U over ranks: AUC = U / (n_pos * n_neg).  Ties share one rank.
    element_ranks: list[float] = [0.0] * len(ordered)
    cursor = 0
    while cursor < len(ordered):
        anchor = ordered[cursor][0]
        end = cursor
        while end + 1 < len(ordered) and ordered[end + 1][0] == anchor:
            end += 1
        mid_rank = (cursor + end) / 2.0 + 1.0
        for index in range(cursor, end + 1):
            element_ranks[ordered[index][2]] = mid_rank
        cursor = end + 1
    positive_rank_sum = sum(
        element_ranks[index] for (_, label, index) in ordered if label == 1.0
    )
    auc = (
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negative_count)
        if positives * negative_count > 0
        else 0.5
    )

    rho = None
    if len(predicted_offsets) >= 3:
        rho = float(spearmanr(predicted_offsets, labels).statistic)
    return {
        "target_context": target_context,
        "auroc": float(auc),
        "spearman_with_label": rho,
        "record_count": len(labels),
    }