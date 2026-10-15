"""Route A anchored SparseEditFormer model.

Subclasses the frozen migration ``SparseEditFormer`` and adds one new head
input: the measured source_value is projected into the shared context vector.
When ``source_value`` is not provided (or SOURCE_ANCHOR_INJECT is off), the
forward pass is exactly the frozen baseline model, so a control run is trivial.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from scripts.m4_sparse.model import SparseEditFormer


class AnchoredSparseEditFormer(SparseEditFormer):
    """SparseEditFormer that can inject the measured source value into its
    context vector to strengthen source anchoring."""

    def __init__(self, cfg):
        super().__init__(cfg)
        d = cfg.HIDDEN_DIM
        self.source_value_proj = nn.Linear(1, d)

    def forward(self, src, cand, edit_feat, study_id, endpoint_id, bench_id,
                source_value=None):
        z = self._context_vec(src, cand, edit_feat, study_id, endpoint_id,
                              bench_id)
        if source_value is not None:
            sv = torch.as_tensor(source_value, dtype=z.dtype, device=z.device)
            az = self.source_value_proj(sv.unsqueeze(-1))
            z = z + az
        mean = self.mean_head(z).squeeze(-1)
        logvar = self.logvar_head(z).squeeze(-1)
        sign = self.sign_head(z).squeeze(-1)
        rank = self.rank_head(z).squeeze(-1)
        return {"mean": mean, "logvar": logvar, "sign": sign, "rank": rank}
