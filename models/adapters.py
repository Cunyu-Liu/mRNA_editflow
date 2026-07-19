"""Parameter-efficient adaptation of a frozen MEF model to downstream tasks.

:class:`AdapterWrappedMEF` freezes both the backbone *and* a pretrained
:class:`~mrna_editflow.models.mrna_editformer.MRNAEditFormer` head, then inserts
a small number of **trainable** components:

* **Bottleneck adapters** -- one per transformer block, applied to the block
  output. Each is a down-project (``model_dim -> rank``) / nonlinearity /
  up-project (``rank -> model_dim``) residual module (Houlsby-style). ``rank`` is
  configurable (default 4, e.g. 4/16).
* **Property-FiLM task head** -- a light head conditioned on a task property
  (e.g. a target translation-efficiency bucket) via FiLM, re-predicting the same
  edit-flow output contract ``(rates, ins_probs, sub_probs, aux)`` so downstream
  sampling code is unchanged.

Only the adapters + task head (and their FiLM/embedding tables) train; every
pretrained parameter keeps ``requires_grad=False``. This yields large sample /
compute savings when adapting one base model to many property targets.

Design
------
We do not surgically rewrite the frozen head's internal blocks. Instead we run
the frozen head's own trunk (backbone -> in_proj -> blocks -> final_norm) to get
frozen hidden states, then let trainable adapters + the property head transform
those hidden states into the task outputs. This keeps the frozen graph intact
(so freezing is trivially correct and auditable) while adding a small trainable
delta.

Complexity: adapters add O(B * L * model_dim * rank) FLOPs per block -- linear in
sequence length, negligible next to the frozen O(L^2) attention.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.constants import VOCAB_MODEL_SIZE, REGION_CDS, V as NUC_V
from ..core.config import ModelConfig
from ..core.mrna_flow_utils import synonymous_nt_sub_mask
from .mrna_editformer import MRNAEditFormer, FiLM, _finite_neg_mask_value

class BottleneckAdapter(nn.Module):
    """Houlsby-style residual bottleneck adapter.

    ``y = x + up(act(down(norm(x))))`` with ``down: D->r``, ``up: r->D``. The
    up-projection is zero-initialised so the adapter starts as identity (safe to
    bolt onto a pretrained model).
    """

    def __init__(self, dim: int, rank: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, dim)
        self.act = nn.GELU()
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.up(self.act(self.down(self.norm(x))))


class AdapterWrappedMEF(nn.Module):
    """Frozen backbone + frozen MEF head + trainable adapters & property head.

    Parameters
    ----------
    backbone : FrozenBackbone
        Frozen encoder (its params are frozen here regardless of prior state).
    base_head : MRNAEditFormer
        Pretrained edit-flow head; frozen here.
    adapter_rank : int
        Bottleneck width (e.g. 4 or 16).
    num_property_buckets : int
        Cardinality of the conditioning property (e.g. target-TE buckets).
    vocab_size : int
        Token vocab for the ins/sub heads.
    """

    def __init__(
        self,
        backbone,
        base_head: MRNAEditFormer,
        adapter_rank: int = 4,
        num_property_buckets: int = 8,
        vocab_size: int = VOCAB_MODEL_SIZE,
    ):
        super().__init__()
        self.backbone = backbone
        self.base_head = base_head
        self.cfg: ModelConfig = base_head.cfg
        self.dim = base_head.dim
        self.vocab_size = vocab_size
        self.adapter_rank = adapter_rank

        # Freeze everything pretrained.
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.base_head.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.base_head.eval()

        # --- trainable pieces ---
        self.adapters = nn.ModuleList(
            [BottleneckAdapter(self.dim, adapter_rank) for _ in range(len(base_head.blocks))]
        )
        # Property conditioning (e.g. target-TE bucket) + FiLM into the head.
        self.property_emb = nn.Embedding(num_property_buckets, self.dim)
        self.property_film = FiLM(self.dim, self.dim)
        self.task_norm = nn.LayerNorm(self.dim)
        # Task heads mirror the base output contract.
        self.rates_out = nn.Sequential(
            nn.Linear(self.dim, self.dim), nn.SiLU(), nn.Linear(self.dim, 3)
        )
        self.ins_logits = nn.Sequential(
            nn.Linear(self.dim, self.dim), nn.SiLU(), nn.Linear(self.dim, vocab_size)
        )
        self.sub_logits = nn.Sequential(
            nn.Linear(self.dim, self.dim), nn.SiLU(), nn.Linear(self.dim, vocab_size)
        )
        self.use_aux = base_head.use_aux
        if self.use_aux:
            self.aux_struct = nn.Sequential(
                nn.Linear(self.dim, self.dim), nn.SiLU(), nn.Linear(self.dim, 2)
            )

    # ------------------------------------------------------------------
    def _frozen_trunk(
        self,
        token_ids: Tensor,
        region_ids: Tensor,
        phase_ids: Tensor,
        time_step: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        """Run the frozen head trunk with trainable adapters after each block.

        Correctness of gradient routing relies on ``requires_grad=False`` on all
        frozen params (backbone + base head): activation gradients still flow
        *through* the frozen blocks so every adapter (not just the last) receives
        gradient, while the frozen weights accumulate none. The pure-prefix
        (embedding / projection / positional / time) has no trainable descendants
        upstream of the first adapter, so it is computed under ``no_grad`` to save
        memory.
        """
        head = self.base_head
        length = token_ids.shape[1]
        device = token_ids.device

        with torch.no_grad():
            feats = self.backbone.embed(token_ids, region_ids, padding_mask)
            x = head.in_proj(feats)
            cond = head._conditioning(region_ids, phase_ids, time_step, length)
            if head.use_rope:
                from .mrna_editformer import _build_rope_cache

                rope = _build_rope_cache(length, head.dim // head.cfg.num_heads, device)
            else:
                pos = torch.arange(length, device=device).clamp(
                    max=head.pos_emb.num_embeddings - 1
                )
                x = x + head.pos_emb(pos).unsqueeze(0)
                rope = None
            x = x + head.time_emb(time_step).unsqueeze(1).expand(-1, length, -1)

        # Frozen blocks are run IN-GRAPH (not under no_grad): their weights have
        # requires_grad=False so they get no gradient, but activation gradients
        # propagate back through them to every adapter in the stack.
        for block, adapter in zip(head.blocks, self.adapters):
            x = block(x, cond, padding_mask, rope)
            x = adapter(x)  # trainable delta

        x = head.final_norm(x)
        return x

    def _apply_codon_constraints(self, rates, sub_logits, token_ids, region_ids, phase_ids):
        b, length = token_ids.shape
        is_cds = region_ids == REGION_CDS
        if self.cfg.use_codon_constraint:
            allowed_nt = synonymous_nt_sub_mask(token_ids, region_ids, phase_ids)
            allowed_full = torch.zeros(
                (b, length, self.vocab_size), dtype=torch.bool, device=token_ids.device
            )
            allowed_full[:, :, :NUC_V] = allowed_nt
            sub_logits = sub_logits.masked_fill(
                ~allowed_full, _finite_neg_mask_value(sub_logits.dtype)
            )
            ins_rate, sub_rate, del_rate = rates[..., 0], rates[..., 1], rates[..., 2]
            if not self.cfg.codon_indel:
                keep = (~is_cds).to(dtype=rates.dtype)
                ins_rate, del_rate = ins_rate * keep, del_rate * keep
            else:
                codon_start = is_cds & (phase_ids == 0)
                allow = (~is_cds | codon_start).to(dtype=rates.dtype)
                ins_rate, del_rate = ins_rate * allow, del_rate * allow
            rates = torch.stack([ins_rate, sub_rate, del_rate], dim=-1)
        return rates, sub_logits

    # ------------------------------------------------------------------
    def forward(
        self,
        token_ids: Tensor,
        region_ids: Tensor,
        phase_ids: Tensor,
        time_step: Tensor,
        padding_mask: Tensor,
        property_bucket: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Same output contract as :meth:`MRNAEditFormer.forward`.

        ``property_bucket``: optional ``[B]`` long tensor selecting the target
        property (e.g. TE bucket); defaults to bucket 0.
        """
        b, length = token_ids.shape
        device = token_ids.device
        x = self._frozen_trunk(token_ids, region_ids, phase_ids, time_step, padding_mask)

        # Property-FiLM conditioning on the trainable task trunk.
        if property_bucket is None:
            property_bucket = torch.zeros(b, dtype=torch.long, device=device)
        prop = self.property_emb(property_bucket.clamp(0, self.property_emb.num_embeddings - 1))
        prop = prop.unsqueeze(1).expand(-1, length, -1)
        x = self.property_film(self.task_norm(x), prop)

        rates = F.softplus(self.rates_out(x))
        ins_logits = self.ins_logits(x)
        sub_logits = self.sub_logits(x)
        rates, sub_logits = self._apply_codon_constraints(
            rates, sub_logits, token_ids, region_ids, phase_ids
        )
        ins_probs = F.softmax(ins_logits, dim=-1)
        sub_probs = F.softmax(sub_logits, dim=-1)

        valid = (~padding_mask).unsqueeze(-1).to(x.dtype)
        rates = torch.nan_to_num(rates * valid, nan=0.0, posinf=1e4, neginf=0.0)
        ins_probs = torch.nan_to_num(ins_probs * valid, nan=0.0)
        sub_probs = torch.nan_to_num(sub_probs * valid, nan=0.0)
        aux = None
        if self.use_aux:
            aux = torch.nan_to_num(self.aux_struct(x) * valid, nan=0.0, posinf=1e4, neginf=-1e4)
        return {"rates": rates, "ins_probs": ins_probs, "sub_probs": sub_probs, "aux": aux}

    # ------------------------------------------------------------------
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def num_frozen_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)


__all__ = ["AdapterWrappedMEF", "BottleneckAdapter"]
