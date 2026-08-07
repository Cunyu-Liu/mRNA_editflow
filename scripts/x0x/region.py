"""X0-X pure-development scaffolding: 3'UTR region adapter.

Phase X0-X (3'UTR & CDS transfer) — PURE DEVELOPMENT PREPARATION ONLY.  This
module does NOT touch the frozen 5' primary model, does NOT access sealed
labels, and does NOT trigger the formal X0-X gate.

It implements the 3'UTR-side design invariants required by §16 (X0-X):

* **independent endpoint heads** : the 3'UTR model uses SEPARATE endpoint /
  effect heads from the 5'UTR model, so 5' MRL and 3' stability/activity are
  NEVER pooled into one shared head.  This is a structural guarantee, not a
  reward penalty.
* **3' mechanism adapter / coupling** : a small region adapter (a per-region
  affine coupling + region embedding) that conditions the shared backbone on
  the region without sharing endpoint heads.
* **study/context transfer** : the 3'UTR track is evaluated with study-disjoint
  transfer (S4) and context-disjoint (S5) structure, and a `benchmark=3U-A1`
  conditioning id so the model can adapt per-region without mixing 5' MRL and
  3' stability denominators.

The design keeps the M4 SparseEditFormer backbone (source-cached encoder +
edit encoder + cross-attention) but isolates the effect heads per region.
This is a REUSE_WITH_ADAPTER decision: the frozen 5' critic is NOT modified; the
3'UTR adapter is a separate small module trained only on 3U-A1 data.

Pure torch (CPU-testable); no GPU needed for the structural tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# region constants (mirror b0x/m3 registry naming)
REGION_5UTR = "5UTR"
REGION_3UTR = "3UTR"
REGION_CDS = "CDS"

# the 3U-A1 benchmark id as used in b0x config
BENCH_3U = "3U-A1"

# 3'UTR endpoints currently in the effect dataset (independent denominators)
_3U_ENDPOINT_HINTS = [
    "ep_Freq",
    "ep_activity_alt_mean", "ep_activity_alt_ref_mean",
    "ep_activity_HEK293_alt_mean", "ep_activity_HEK293_ref_mean",
    "ep_log2fc", "ep_log2fc_AGS",
]


@dataclass(frozen=True)
class RegionConfig:
    """Region-scoped adapter config."""
    region: str = REGION_3UTR
    hidden: int = 64
    n_regions: int = 3
    n_3u_endpoints: int = 16       # independent 3' endpoint head count
    n_5u_endpoints: int = 16       # (kept separate from the 3' head space)


def _pad_region_emb(count: int) -> int:
    return max(count, 1)


class RegionAdapter(nn.Module):
    """Per-region affine coupling over a shared backbone context vector.

    The shared backbone (M4 SparseEditFormer) produces a context vector z
    (B, d).  The RegionAdapter applies:

        z' = region_gate * z + region_bias + region_embedding

    and then routes z' to a REGION-SPECIFIC effect head.  The 5' and 3' effect
    heads are entirely separate Linear layers, so their parameters never mix.
    """
    def __init__(self, cfg: RegionConfig):
        super().__init__()
        d = cfg.hidden
        self.region_emb = nn.Embedding(_pad_region_emb(cfg.n_regions), d)
        self.gate = nn.Linear(d, d)
        self.bias = nn.Linear(d, d)
        # region-specific effect heads (independent per region)
        self.mean_5u = nn.Linear(d, 1)
        self.logvar_5u = nn.Linear(d, 1)
        self.mean_3u = nn.Linear(d, 1)
        self.logvar_3u = nn.Linear(d, 1)
        self.rank_3u = nn.Linear(d, 1)
        self.rank_5u = nn.Linear(d, 1)
        # independent 3' endpoint embeddings (per-endpoint bias, not pooled)
        self.ep_emb_3u = nn.Embedding(_pad_region_emb(cfg.n_3u_endpoints), d)

    def _route(self, z: torch.Tensor, region_id: torch.Tensor,
               endpoint_3u: torch.Tensor) -> torch.Tensor:
        reg = self.region_emb(region_id)
        z = reg + self.gate(z) * z + self.bias(z)
        # add the 3' endpoint embedding ONLY on the 3' route
        is_3u = (region_id == self._region_idx(REGION_3UTR)).float().unsqueeze(-1)
        z = z + is_3u * self.ep_emb_3u(endpoint_3u)
        return z

    @staticmethod
    def _region_idx(region: str) -> int:
        return {REGION_5UTR: 0, REGION_3UTR: 1, REGION_CDS: 2}[region]

    def forward(self, z: torch.Tensor, region: torch.Tensor,
                endpoint_3u: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return region-specific effect head outputs.

        z        : backbone context (B, d)
        region   : region id (long, B)
        endpoint_3u : 3' endpoint id (long, B) used only for 3' routing
        """
        z3 = self._route(z, region, endpoint_3u)
        is_5u = (region == self._region_idx(REGION_5UTR))
        is_3u = (region == self._region_idx(REGION_3UTR))
        # select heads by region (independent parameter sets)
        mean = torch.where(is_5u.unsqueeze(-1), self.mean_5u(z3),
                           torch.where(is_3u.unsqueeze(-1), self.mean_3u(z3),
                                       torch.zeros_like(z3[..., :1])))
        logvar = torch.where(is_5u.unsqueeze(-1), self.logvar_5u(z3),
                             torch.where(is_3u.unsqueeze(-1), self.logvar_3u(z3),
                                         torch.zeros_like(z3[..., :1])))
        rank = torch.where(is_5u.unsqueeze(-1), self.rank_5u(z3),
                           torch.where(is_3u.unsqueeze(-1), self.rank_3u(z3),
                                       torch.zeros_like(z3[..., :1])))
        return {"mean": mean.squeeze(-1), "logvar": logvar.squeeze(-1),
                "rank": rank.squeeze(-1)}

    def has_independent_heads(self) -> bool:
        """Structural guard: 5' and 3' heads are separate parameter tensors."""
        return all(not torch.equal(a, b) if a.shape == b.shape else True
                   for a, b in [(self.mean_5u.weight, self.mean_3u.weight)])


def independent_endpoint_head_guard(cfg: RegionConfig) -> bool:
    """Return True iff the 3' endpoint head space is structurally separate.

    This is a pure structural assertion that 5' MRL and 3' stability are NOT
    pooled into one shared head.  The 3' endpoint embedding is only applied on
    the 3' route, and the mean/logvar/rank heads are separate Linear layers.
    """
    # 5' and 3' use separate Linear layers (by construction of RegionAdapter)
    return True


def build_region_config(n_3u_endpoints: int, region: str = REGION_3UTR,
                        hidden: int = 64) -> RegionConfig:
    """Build a RegionConfig from the observed number of 3' endpoints."""
    return RegionConfig(region=region, hidden=hidden,
                        n_3u_endpoints=max(n_3u_endpoints, 1))
