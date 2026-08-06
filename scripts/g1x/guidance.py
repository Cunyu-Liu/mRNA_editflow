"""G1-X guidance integration: frozen base flow + frozen M4 critic + trained guidance head.

Phase G1-X real-mRNA guidance integration.  We FREEZE the F0-X base flow
(FlowRateNet) and the M4 SparseEditFormer critic, and train ONLY a small
guidance head: a per-position additive effect field (GuidanceRatioNet).

The guidance head predicts, for a source/current sequence, the endpoint-delta
effect of substituting each nucleotide at each editable position.  It is
trained on MEASURED single-edit pairs (edit_count==1), where the per-position
effect is directly the measured delta (well-posed), with an L2 prior pushing
non-observed cells toward 0.  This is the first-order guidance field.

Guidance policies (methods compared in G1-X):
  * no_guidance      : base FlowRateNet policy only.
  * first_order      : base + beta * guidance_head per-action effect (first-order
                       expansion of the effect field).
  * rate_cfg         : base + beta * (frozen critic candidate delta), i.e.
                       classifier-free / rate CFG on the critic reward.
  * latent_cfg       : base + beta * (frozen critic rank head) -- guidance on the
                       critic's ranking signal.
  * dgm_learned      : pure learned-ratio guidance (NO base flow): guided logits =
                       beta * (frozen critic candidate delta).  This is the
                       *learned/approximate rate guidance* (per G0-X wording
                       boundary: the ratio is from a learned critic, not the true
                       q/p1, so it is NOT exact guidance).
  * generate_then_rerank: base flow generates a candidate set, frozen critic
                       reranks them (composite; produced by the runner).

All policies return hard-masked guided logits over the legal actions so the
sampler softmax reproduces a legal, entropy-reportable distribution.  The
mathematical cores of the base flow and critic are NOT rewritten.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from scripts.m4_sparse.config import MAX_SEQ_LEN, NUC_ORDER
from scripts.m4_sparse.dataset import edit_features, normalize, one_hot
from scripts.f0x.flow import LegalAction, apply_action, enumerate_legal_actions

G1_HIDDEN = 32
G1_LR = 1e-3
G1_EPOCHS = 3
G1_BATCH = 512


# ---------------------------------------------------------------------------
# guidance head
# ---------------------------------------------------------------------------

class GuidanceRatioNet(nn.Module):
    """Per-position additive effect field (first-order guidance head).

    f(src_seq)[pos, nt] = predicted endpoint-delta effect of substituting the
    current nucleotide at `pos` with `nt`.  Trained on measured single-edit
    pairs; non-observed cells pushed toward 0 by an L2 prior.
    """
    def __init__(self, hidden: int = G1_HIDDEN):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, hidden, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.effect_head = nn.Linear(hidden, len(NUC_ORDER))

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """src: [B,L,4] one-hot -> [B,L,4] per-position-nt effect."""
        h = self.stem(src.permute(0, 2, 1)).permute(0, 2, 1)
        return self.effect_head(h)

    @torch.no_grad()
    def effects(self, src_vec: torch.Tensor) -> np.ndarray:
        """[B,L,4] effect field as numpy (padding positions -> 0)."""
        self.eval()
        e = self(src_vec)
        return e.cpu().numpy()


def _single_edit_rows(rows: List[dict]) -> List[dict]:
    return [r for r in rows
            if len([e for e in r.get("edit_list", []) if e["op"] == "SUB"]) == 1]


def train_guidance_head(train_rows: List[dict], device, hidden: int = G1_HIDDEN,
                        epochs: int = G1_EPOCHS, batch: int = G1_BATCH,
                        lr: float = G1_LR) -> GuidanceRatioNet:
    """Train GuidanceRatioNet on measured single-edit deltas.

    Target: for the single edit (pos, nt) with measured delta d, the head's
    cell (pos, nt) is regressed to d; all other cells get an L2 prior toward 0.
    """
    rows = _single_edit_rows(train_rows)
    if not rows:
        raise RuntimeError("no single-edit measured rows available to train guidance head")
    net = GuidanceRatioNet(hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    srcs = np.stack([one_hot(r["source_sequence"]) for r in rows])
    Y = np.zeros((len(rows), MAX_SEQ_LEN, len(NUC_ORDER)), dtype=np.float32)
    M = np.zeros((len(rows), MAX_SEQ_LEN, len(NUC_ORDER)), dtype=np.float32)
    for i, r in enumerate(rows):
        s = normalize(r["source_sequence"])
        for e in r.get("edit_list", []):
            if e["op"] == "SUB":
                pos = int(e["pos"])
                tok = str(e["token"]).upper().replace("T", "U")
                if pos < MAX_SEQ_LEN and tok in NUC_ORDER:
                    Y[i, pos, NUC_ORDER.index(tok)] = float(r["delta"] or 0.0)
                    M[i, pos, NUC_ORDER.index(tok)] = 1.0
    n = len(rows)
    rng = np.random.RandomState(0)
    net.train()
    last = None
    for _ in range(epochs):
        perm = rng.permutation(n)
        for b0 in range(0, n, batch):
            idx = perm[b0:b0 + batch]
            src = torch.tensor(srcs[idx]).to(device)
            y = torch.tensor(Y[idx]).to(device)
            m = torch.tensor(M[idx]).to(device)
            pred = net(src)
            # masked L1 on observed cells + L2 prior on all cells
            mse_obs = ((pred - y) ** 2 * m).sum() / max(m.sum(), 1.0)
            mse_prior = ((pred - y) ** 2 * (1 - m)).mean()
            loss = mse_obs + 1e-3 * mse_prior
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.item())
    net.eval()
    return net


# ---------------------------------------------------------------------------
# frozen critic scorer
# ---------------------------------------------------------------------------

def critic_score(model, vocab, device, src_seq: str, cand_seq: str,
                 study: str, endpoint: str, bench: str,
                 source_value: Optional[float]) -> Dict:
    """Frozen M4 critic score for a (source, candidate) pair.

    Returns mean (candidate-value pred), logvar, rank, and delta = mean - source.
    """
    src = torch.tensor(one_hot(src_seq)).unsqueeze(0).to(device)
    cand = torch.tensor(one_hot(cand_seq)).unsqueeze(0).to(device)
    # build edit list source->candidate (single-pass diff for the first edit)
    ef = _edit_feat_for(src_seq, cand_seq)
    ef = torch.tensor(ef).unsqueeze(0).to(device)
    sid = torch.tensor([vocab["study"].get(study, 0)]).to(device)
    eid = torch.tensor([vocab["endpoint"].get(endpoint, 0)]).to(device)
    bid = torch.tensor([vocab["benchmark"].get(bench, 0)]).to(device)
    with torch.no_grad():
        out = model(src, cand, ef, sid, eid, bid)
    mean = float(out["mean"].item())
    lv = float(out["logvar"].item())
    rank = float(out["rank"].item())
    delta = mean - source_value if source_value is not None else mean
    return {"mean": mean, "logvar": lv, "rank": rank, "delta": delta}


def _edit_feat_for(src: str, cand: str) -> np.ndarray:
    """Minimal edit list source->candidate (SUB-only, first differing bases)."""
    edits = []
    s, c = normalize(src), normalize(cand)
    for i in range(min(len(s), len(c))):
        if s[i] != c[i]:
            edits.append({"op": "SUB", "pos": i, "token": c[i]})
    if len(c) > len(s):
        for i in range(len(s), len(c)):
            edits.append({"op": "SUB", "pos": i, "token": c[i]})
    return edit_features(edits, max(len(s), 1))


def critic_scores_batch(critic, vocab, device, src_seq: str, cand_seqs: List[str],
                        study: Optional[str], endpoint: Optional[str],
                        bench: str, source_value: Optional[float]) -> Dict:
    """Frozen critic scores for many candidate seqs from one source (one forward).

    Returns dict of numpy arrays: mean, rank, logvar, delta (= mean - source).
    """
    B = len(cand_seqs)
    if B == 0:
        return {"mean": np.zeros(0), "rank": np.zeros(0),
                "logvar": np.zeros(0), "delta": np.zeros(0)}
    src = torch.tensor(one_hot(src_seq)).unsqueeze(0).repeat(B, 1, 1).to(device)
    cand = torch.stack([torch.tensor(one_hot(c)) for c in cand_seqs]).to(device)
    ef = torch.stack([torch.tensor(_edit_feat_for(src_seq, c)) for c in cand_seqs]).to(device)
    sid = torch.full((B,), vocab["study"].get(study, 0), dtype=torch.long, device=device)
    eid = torch.full((B,), vocab["endpoint"].get(endpoint, 0), dtype=torch.long, device=device)
    bid = torch.full((B,), vocab["benchmark"].get(bench, 0), dtype=torch.long, device=device)
    with torch.no_grad():
        out = critic(src, cand, ef, sid, eid, bid)
    mean = out["mean"].cpu().numpy()
    rank = out["rank"].cpu().numpy()
    lv = out["logvar"].cpu().numpy()
    delta = mean - source_value if source_value is not None else mean
    return {"mean": mean, "rank": rank, "logvar": lv, "delta": delta}


# ---------------------------------------------------------------------------
# guidance step policies
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    l = np.asarray(logits, float)
    l = l - l.max()
    e = np.exp(l)
    return e / e.sum()


def base_step_policy(net, device, ctx=None):
    """no_guidance: pure base FlowRateNet policy."""
    def policy(state, actions):
        bl = np.asarray(net.policy_fn(state, actions), dtype=float)
        return {"guided_logits": bl, "base_logits": bl, "ratio": None,
                "critic_mean": None, "critic_logvar": None}
    return policy


def _guidance_head_effects(head, state, device):
    """Per-action guidance-head effect for the current state."""
    vec = torch.tensor(one_hot(state.seq)).unsqueeze(0).to(device)
    eff = head.effects(vec)[0]  # [L,4]
    return eff


def first_order_step_policy(base_net, head, beta, device, ctx=None):
    """first_order: base + beta * guidance-head per-action effect."""
    def policy(state, actions):
        bl = np.asarray(base_net.policy_fn(state, actions), dtype=float)
        eff = _guidance_head_effects(head, state, device)
        gl = np.empty(len(actions), dtype=float)
        for i, a in enumerate(actions):
            if a.pos < eff.shape[0]:
                gl[i] = bl[i] + beta * float(eff[a.pos, "ACGU".index(a.target_nt)])
            else:
                gl[i] = bl[i]
        return {"guided_logits": gl, "base_logits": bl, "ratio": None,
                "critic_mean": None, "critic_logvar": None}
    return policy


def _ctx(**kw):
    """Default biological context (neutral ids) if a source has no provenance."""
    return {"study": None, "endpoint": None, "bench": None,
            "source_value": None, **kw}


def _critic_reward_policy(base_net, critic, vocab, beta, device,
                          use_rank: bool, ctx):
    """rate_cfg (use_rank=False, critic delta) or latent_cfg (use_rank=True,
    critic rank head).  guided = base + beta * critic_reward_after_action.
    All legal actions' critic scores are computed in ONE batched forward."""
    c = _ctx(**(ctx or {}))
    # resolve conditioning ids once (fall back to id 0 for unknowns)
    study = c["study"] if c["study"] in vocab["study"] else None
    endpoint = c["endpoint"] if c["endpoint"] in vocab["endpoint"] else None
    bench = c["bench"] if c["bench"] in vocab["benchmark"] else "5U-A1"

    def policy(state, actions):
        bl = np.asarray(base_net.policy_fn(state, actions), dtype=float)
        cands = [apply_action(state, a).seq for a in actions]
        sc = critic_scores_batch(critic, vocab, device, state.source_seq, cands,
                                 study, endpoint, bench, c["source_value"])
        reward = sc["rank"] if use_rank else sc["delta"]
        gl = bl + beta * reward
        return {"guided_logits": gl, "base_logits": bl, "ratio": None,
                "critic_mean": float(sc["mean"].mean()) if len(sc["mean"]) else None,
                "critic_logvar": float(sc["logvar"].mean()) if len(sc["logvar"]) else None}
    return policy


def rate_cfg_step_policy(base_net, critic, vocab, beta, device, ctx=None):
    return _critic_reward_policy(base_net, critic, vocab, beta, device,
                                 use_rank=False, ctx=ctx)


def latent_cfg_step_policy(base_net, critic, vocab, beta, device, ctx=None):
    return _critic_reward_policy(base_net, critic, vocab, beta, device,
                                 use_rank=True, ctx=ctx)


def dgm_learned_step_policy(critic, vocab, beta, device, ctx=None):
    """dgm_learned: pure learned-ratio guidance (no base flow).

    guided_logits = beta * critic candidate delta.  This is the learned /
    approximate rate guidance (G0-X wording boundary: learned critic ratio,
    NOT exact q/p1).  All legal actions' critic scores in one batched forward.
    """
    c = _ctx(**(ctx or {}))
    study = c["study"] if c["study"] in vocab["study"] else None
    endpoint = c["endpoint"] if c["endpoint"] in vocab["endpoint"] else None
    bench = c["bench"] if c["bench"] in vocab["benchmark"] else "5U-A1"

    def policy(state, actions):
        cands = [apply_action(state, a).seq for a in actions]
        sc = critic_scores_batch(critic, vocab, device, state.source_seq, cands,
                                 study, endpoint, bench, c["source_value"])
        gl = beta * sc["delta"]
        return {"guided_logits": gl, "base_logits": None, "ratio": None,
                "critic_mean": float(sc["mean"].mean()) if len(sc["mean"]) else None,
                "critic_logvar": float(sc["logvar"].mean()) if len(sc["logvar"]) else None}
    return policy


POLICY_BUILDERS = {
    "no_guidance": lambda net, head, critic, vocab, beta, device, ctx: base_step_policy(net, device, ctx),
    "first_order": lambda net, head, critic, vocab, beta, device, ctx: first_order_step_policy(net, head, beta, device, ctx),
    "rate_cfg": lambda net, head, critic, vocab, beta, device, ctx: rate_cfg_step_policy(net, critic, vocab, beta, device, ctx),
    "latent_cfg": lambda net, head, critic, vocab, beta, device, ctx: latent_cfg_step_policy(net, critic, vocab, beta, device, ctx),
    "dgm_learned": lambda net, head, critic, vocab, beta, device, ctx: dgm_learned_step_policy(critic, vocab, beta, device, ctx),
}


__all__ = [
    "GuidanceRatioNet", "train_guidance_head", "critic_score",
    "base_step_policy", "first_order_step_policy", "rate_cfg_step_policy",
    "latent_cfg_step_policy", "dgm_learned_step_policy", "POLICY_BUILDERS",
]