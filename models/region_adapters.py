"""Region-specialized adapters for the mRNA Edit-Flow head.

Roadmap architecture upgrade #2. A single proposal ranker must serve three
functionally distinct regions -- the 5'UTR regulatory canvas, the protein-coding
CDS lattice, and the 3'UTR stability canvas. Sharing one output trunk across all
of them forces one representation to encode conflicting objectives.

:class:`RegionSpecializedEditFormer` wraps a *frozen or trainable* base
:class:`~mrna_editflow.models.mrna_editformer.MRNAEditFormer` and inserts a small
per-region residual adapter between the shared trunk (``base.encode``) and the
output heads (``base.heads``):

``h'_i = h_i + A_{r(i)}(h_i)``,

where ``r(i)`` is the region label at position ``i`` and ``A_r`` is a
region-specific bottleneck MLP ``D -> bottleneck -> D`` with a **zero-initialized
final layer**. At initialization every ``A_r`` outputs zero, so
``h'_i = h_i`` and the wrapper reproduces the base model exactly -- a 10k Stage A
checkpoint loads into ``base`` and behaves identically until the adapters are
trained. Because the wrapper never renames or removes base parameters, the base
``state_dict`` remains directly loadable.

Design choices
--------------
* **Checkpoint safety**: adapters live under ``adapters.*`` submodules; base
  weights are untouched, so ``base.load_state_dict(stage_a_state)`` still works.
* **Region gating**: position ``i`` is routed only to its region's adapter via a
  hard one-hot mask on ``region_ids``; BOS/PAD/sentinel positions bypass all
  adapters (residual identity), and padded rows are re-zeroed by ``base.heads``.
* **Parameter efficiency**: with bottleneck ``b`` and width ``D`` each region
  adds ``2 D b + D + b`` parameters, i.e. ``O(R * D * b)`` total -- tiny next to
  the shared trunk.

Complexity of :meth:`forward` is the base trunk cost plus ``O(B * L * D * b)``
for the adapters.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ..core.constants import NUM_REGIONS, REGION_3UTR, REGION_5UTR, REGION_CDS
from .mrna_editformer import MRNAEditFormer

_REGION_LABELS: dict[int, str] = {
    REGION_5UTR: "utr5",
    REGION_CDS: "cds",
    REGION_3UTR: "utr3",
}


class RegionAdapter(nn.Module):
    """Residual bottleneck adapter with a zero-initialized output layer.

    ``A(h) = W_up(act(W_down(h)))`` with ``W_up`` initialized to zero so the
    adapter is the zero map at initialization and the residual connection makes
    the wrapped model an identity over the base output. Complexity is
    ``O(B * L * D * b)``.
    """

    def __init__(self, dim: int, bottleneck: int) -> None:
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.up(self.act(self.down(x)))


class RegionSpecializedEditFormer(nn.Module):
    """Wrap a base MEF head with per-region residual adapters.

    Parameters
    ----------
    base : MRNAEditFormer
        The shared Edit-Flow head. Its ``encode``/``heads`` split is reused so
        the codon-lattice grammar and constraint masking are unchanged.
    bottleneck : int
        Width of each region adapter's bottleneck.
    regions : sequence of int, optional
        Region ids that receive a dedicated adapter (defaults to 5'UTR/CDS/3'UTR).
    freeze_base : bool
        When ``True`` (default) only the adapters are trainable, matching the
        "frozen backbone + lightweight head/adapter" philosophy.
    """

    def __init__(
        self,
        base: MRNAEditFormer,
        *,
        bottleneck: int = 32,
        regions: Optional[Sequence[int]] = None,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.base = base
        self.dim = base.dim
        self.bottleneck = int(bottleneck)
        self.region_ids = tuple(int(r) for r in (regions if regions is not None else (
            REGION_5UTR, REGION_CDS, REGION_3UTR,
        )))
        self.adapters = nn.ModuleDict(
            {
                _REGION_LABELS.get(r, f"region{r}"): RegionAdapter(self.dim, self.bottleneck)
                for r in self.region_ids
            }
        )
        self.freeze_base = bool(freeze_base)
        if self.freeze_base:
            for param in self.base.parameters():
                param.requires_grad_(False)

    def _region_key(self, region_id: int) -> str:
        return _REGION_LABELS.get(int(region_id), f"region{int(region_id)}")

    def apply_adapters(self, x: Tensor, region_ids: Tensor) -> Tensor:
        """Return ``x + sum_r 1[region==r] * A_r(x)``.

        Each position is modulated only by the adapter for its own region; any
        position whose region id is not adapted (e.g. BOS/PAD sentinels) keeps
        the identity residual. Complexity is ``O(R * B * L * D * b)``.
        """
        out = x
        for region_id in self.region_ids:
            key = self._region_key(region_id)
            mask = (region_ids == region_id).unsqueeze(-1).to(x.dtype)  # [B, L, 1]
            if bool(torch.any(mask > 0)):
                out = out + mask * self.adapters[key](x)
        return out

    def forward(
        self,
        token_ids: Tensor,
        region_ids: Tensor,
        phase_ids: Tensor,
        time_step: Tensor,
        padding_mask: Tensor,
        backbone,
    ) -> Dict[str, Tensor]:
        """Base trunk -> region adapters -> base heads. Signature matches base.

        The returned dict has the same keys/shapes as
        :meth:`MRNAEditFormer.forward`; all hard biological constraints are still
        enforced inside ``base.heads``.
        """
        x = self.base.encode(token_ids, region_ids, phase_ids, time_step, padding_mask, backbone)
        x = self.apply_adapters(x, region_ids)
        return self.base.heads(x, token_ids, region_ids, phase_ids, padding_mask)

    def adapter_parameters(self):
        """Iterator over adapter parameters (the trainable set when frozen)."""
        return self.adapters.parameters()

    def num_adapter_params(self) -> int:
        return sum(p.numel() for p in self.adapters.parameters())

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["RegionAdapter", "RegionSpecializedEditFormer"]
