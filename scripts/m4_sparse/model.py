"""SparseEditFormer: source-relative mRNA effect model (M4).

Single small from-scratch backbone (NOT an ensemble of foundation models).
Components implemented:
  * source-cached encoder   : source sequence encoded once, cached; the
                              candidate "reads against" it via cross-attention
                              (sparse edit attention).
  * explicit edit encoder   : MLP over 12-dim edit features (op/pos/token).
  * endpoint/context heads  : study / endpoint / benchmark conditioning.
  * mean/variance head      : heteroscedastic regression (mean + log-var).
  * sign/beneficial head    : binary classification of sign(delta).
  * pairwise/listwise rank  : scalar score for ranking candidates.
  * inverse consistency     : f(src,cand) + f(cand,src) ~ 0 (regularizer in
                              train.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SparseEditFormer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.HIDDEN_DIM
        self.cfg = cfg

        # shared nucleotide stem over the 4-channel one-hot (ACGU)
        self.stem = nn.Sequential(
            nn.Conv1d(4, d, cfg.CONV_KS, padding=cfg.CONV_KS // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(d, d, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pos = nn.Parameter(torch.zeros(1, cfg.MAX_SEQ_LEN, d))
        nn.init.normal_(self.pos, std=0.02)

        # source encoder (self-attention over cached source)
        src_layer = nn.TransformerEncoderLayer(
            d, cfg.NHEAD, cfg.DIM_FF, dropout=cfg.DROP, batch_first=True,
            activation="gelu")
        self.src_encoder = nn.TransformerEncoder(src_layer, num_layers=cfg.N_LAYERS)

        # candidate encoder (self-attention within candidate)
        cand_layer = nn.TransformerEncoderLayer(
            d, cfg.NHEAD, cfg.DIM_FF, dropout=cfg.DROP, batch_first=True,
            activation="gelu")
        self.cand_encoder = nn.TransformerEncoder(cand_layer, num_layers=1)

        # cross-attention: candidate reads against cached source
        self.cross_attn = nn.MultiheadAttention(d, cfg.NHEAD, dropout=cfg.DROP,
                                                batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.pool_proj = nn.Linear(d * 2, d)

        # explicit edit/action encoder
        self.edit_mlp = nn.Sequential(
            nn.Linear(cfg.EDIT_FEAT_DIM, d), nn.ReLU(inplace=True), nn.Linear(d, d))

        # endpoint/context-specific heads (conditioning)
        self.study_emb = nn.Embedding(max(cfg.N_STUDIES, 1), d)
        self.endpoint_emb = nn.Embedding(max(cfg.N_ENDPOINTS, 1), d)
        self.bench_emb = nn.Embedding(max(cfg.N_BENCHMARKS, 1), d)

        # mean / variance, sign, ranking heads
        self.mean_head = nn.Linear(d, 1)
        self.logvar_head = nn.Linear(d, 1)
        self.sign_head = nn.Linear(d, 1)
        self.rank_head = nn.Linear(d, 1)

    def _context_vec(self, src, cand, edit_feat, study_id, endpoint_id, bench_id):
        """Encode candidate relative to cached source -> context vector (B,d)."""
        hs = self.stem(src.permute(0, 2, 1)).permute(0, 2, 1) + self.pos
        hs = self.src_encoder(hs)  # cached source tokens (B,L,d)
        hc = self.stem(cand.permute(0, 2, 1)).permute(0, 2, 1) + self.pos
        hc = self.cand_encoder(hc)
        cross, _ = self.cross_attn(hc, hs, hs)      # cand reads cached source
        hc = self.norm(hc + cross)
        pooled = torch.cat([hc.mean(dim=1), hc.max(dim=1).values], dim=-1)
        vec = self.pool_proj(pooled)
        ev = self.edit_mlp(edit_feat)
        ctx = (self.study_emb(study_id) + self.endpoint_emb(endpoint_id)
               + self.bench_emb(bench_id))
        return vec + ev + ctx

    def forward(self, src, cand, edit_feat, study_id, endpoint_id, bench_id):
        z = self._context_vec(src, cand, edit_feat, study_id, endpoint_id, bench_id)
        mean = self.mean_head(z).squeeze(-1)
        logvar = self.logvar_head(z).squeeze(-1)
        sign = self.sign_head(z).squeeze(-1)
        rank = self.rank_head(z).squeeze(-1)
        return {"mean": mean, "logvar": logvar, "sign": sign, "rank": rank}

    def predict_delta(self, src, cand, edit_feat, study_id, endpoint_id, bench_id,
                      anchor=None):
        """Return (delta_pred, sign_logit, rank_score).

        anchor: optional measured source_value (B,) used when the model was
        trained on candidate_value (TARGET="candidate_value").  If anchor given,
        delta = mean - anchor.
        """
        out = self.forward(src, cand, edit_feat, study_id, endpoint_id, bench_id)
        delta = out["mean"]
        if anchor is not None:
            delta = delta - anchor
        return delta, out["sign"], out["rank"]
