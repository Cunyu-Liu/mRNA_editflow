"""FlowRateNet: source-conditioned legal Edit Flow base model (F0-X).

A single small from-scratch backbone (mirrors the SparseEditFormer source-cached
encoder conventions) that produces per-position substitution rate logits over the
5'UTR.  Primary action set is UTR5_SUB; rates are non-negative (softplus);
the legal action mask is applied to the logits BEFORE normalization (hard mask).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from scripts.m4_sparse.config import MAX_SEQ_LEN, NUC_ORDER


class FlowRateNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.HIDDEN_DIM
        self.cfg = cfg
        self.max_len = getattr(cfg, "MAX_SEQ_LEN", MAX_SEQ_LEN)

        # shared nucleotide stem over the 4-channel one-hot (ACGU)
        self.stem = nn.Sequential(
            nn.Conv1d(4, d, cfg.CONV_KS, padding=cfg.CONV_KS // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(d, d, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pos = nn.Parameter(torch.zeros(1, self.max_len, d))
        nn.init.normal_(self.pos, std=0.02)

        # source encoder (cached source context) + current-state encoder
        src_layer = nn.TransformerEncoderLayer(
            d, cfg.NHEAD, cfg.DIM_FF, dropout=cfg.DROP, batch_first=True,
            activation="gelu")
        self.src_encoder = nn.TransformerEncoder(src_layer, num_layers=cfg.N_LAYERS)
        cand_layer = nn.TransformerEncoderLayer(
            d, cfg.NHEAD, cfg.DIM_FF, dropout=cfg.DROP, batch_first=True,
            activation="gelu")
        self.cand_encoder = nn.TransformerEncoder(cand_layer, num_layers=1)

        # budget conditioning (fixed budget k in {1,3,5}; embed up to 6)
        self.budget_emb = nn.Embedding(6, d)

        # per-position substitution rate head: (pos, nt) -> logit
        self.rate_head = nn.Linear(d, len(NUC_ORDER))

    def _encode(self, onehot: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        h = self.stem(onehot.permute(0, 2, 1)).permute(0, 2, 1)
        h = h[:, :self.max_len, :] + self.pos
        return encoder(h)

    def forward(self, src: torch.Tensor, cur: torch.Tensor,
                budget_idx: torch.Tensor, editable: torch.Tensor) -> Dict:
        """src, cur: [B,L,4] one-hot; budget_idx: [B] long; editable: [B,L,4] bool.

        Returns dict with raw logits, hard-masked logits, non-negative rates,
        and the normalized legal policy.
        """
        hs = self._encode(src, self.src_encoder)          # cached source [B,L,d]
        hc = self._encode(cur, self.cand_encoder)         # current state [B,L,d]
        bud = self.budget_emb(budget_idx).unsqueeze(1)    # [B,1,d]
        ctx = hs + hc + bud                               # source-conditioned
        logits = self.rate_head(ctx)                      # [B,L,4]

        from .flow import apply_hard_mask, nonnegative_rates, policy_from_masked_logits
        masked = apply_hard_mask(logits, editable)
        rates = nonnegative_rates(masked, editable)
        policy = policy_from_masked_logits(masked, editable)
        return {
            "logits": logits,
            "masked_logits": masked,
            "rates": rates,
            "policy": policy,
        }

    def _featurize_state(self, seq, device):
        vec = torch.zeros((1, self.max_len, len(NUC_ORDER)), device=device)
        for i, ch in enumerate(seq[:self.max_len]):
            if ch in "ACGU":
                vec[0, i, "ACGU".index(ch)] = 1.0
        return vec

    def policy_fn(self, state, actions):
        """Adapter for FirstOrderConstrainedSampler: scores over legal actions.

        Builds the source (from state.source_seq) and current (from
        state.seq) encodings, applies the hard legal mask, and returns the
        masked logits for actions (so the sampler softmax reproduces the
        model policy).  Source-conditioned per-state (each source anchored by
        its own sequence).
        """
        import torch
        dev = next(self.parameters()).device
        src_vec = self._featurize_state(state.source_seq, dev)
        cur_vec = self._featurize_state(state.seq, dev)
        editable = torch.zeros((1, self.max_len, len(NUC_ORDER)), dtype=torch.bool,
                               device=dev)
        for i in range(min(len(state.seq), self.max_len)):
            if bool(state.editable[i]):
                for nt in "ACGU":
                    if nt != state.seq[i]:
                        editable[0, i, "ACGU".index(nt)] = True
        budget_idx = torch.tensor([min(state.budget_remaining, 5)],
                                  dtype=torch.long, device=dev)
        with torch.no_grad():
            out = self.forward(src_vec, cur_vec, budget_idx, editable)
            masked = out["masked_logits"][0]  # [L,4]
        scores = []
        for a in actions:
            if a.pos < self.max_len:
                scores.append(float(masked[a.pos, "ACGU".index(a.target_nt)]))
            else:
                scores.append(-float('inf'))
        return scores


__all__ = ["FlowRateNet"]
