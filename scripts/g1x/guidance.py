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
  * latent_cfg       : base + beta * frozen critic predicted delta mean.  R3-P0
                       forbids an independent rank head, so this label is a
                       compatibility alias with distinct policy accounting.
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
from scripts.m4_sparse.dataset import (
    UNKNOWN_CONTEXT,
    edit_features,
    encode_pair,
    normalize,
    one_hot,
)
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
            if len(r.get("edit_list", [])) == 1
            and r["edit_list"][0].get("op") == "SUB"]


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
    encoded = [
        encode_pair(r["source_sequence"], r["candidate_sequence"],
                    r.get("edit_list"))
        for r in rows
    ]
    srcs = np.stack([item[0] for item in encoded])
    Y = np.zeros((len(rows), MAX_SEQ_LEN, len(NUC_ORDER)), dtype=np.float32)
    M = np.zeros((len(rows), MAX_SEQ_LEN, len(NUC_ORDER)), dtype=np.float32)
    for i, r in enumerate(rows):
        window_start = encoded[i][4]
        s = normalize(r["source_sequence"])
        for e in r.get("edit_list", []):
            if e["op"] == "SUB":
                pos = int(e["pos"])
                tok = str(e["token"]).upper().replace("T", "U")
                local_pos = pos - window_start
                if 0 <= local_pos < MAX_SEQ_LEN and tok in NUC_ORDER:
                    Y[i, local_pos, NUC_ORDER.index(tok)] = float(r["delta"])
                    M[i, local_pos, NUC_ORDER.index(tok)] = 1.0
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

    Returns mean-head output, logvar, rank, and delta under the checkpoint's
    frozen TARGET semantics (direct for delta; measured-anchor subtraction for
    candidate_value).
    """
    edits = _edits_for(src_seq, cand_seq)
    src_x, cand_x, src_mask, cand_mask, _ = encode_pair(
        src_seq, cand_seq, edits)
    src = torch.tensor(src_x).unsqueeze(0).to(device)
    cand = torch.tensor(cand_x).unsqueeze(0).to(device)
    ef = torch.tensor(edit_features(edits, max(len(normalize(src_seq)), 1))) \
        .unsqueeze(0).to(device)
    src_mask = torch.tensor(src_mask).unsqueeze(0).to(device)
    cand_mask = torch.tensor(cand_mask).unsqueeze(0).to(device)
    sid = torch.tensor([_context_id(vocab["study"], study, "study")]).to(device)
    eid = torch.tensor([_context_id(vocab["endpoint"], endpoint, "endpoint")]).to(device)
    bid = torch.tensor([_context_id(vocab["benchmark"], bench, "benchmark")]).to(device)
    with torch.no_grad():
        out = model(src, cand, ef, sid, eid, bid,
                    src_pad_mask=src_mask, cand_pad_mask=cand_mask)
    mean = float(out["mean"].item())
    lv = float(out["logvar"].item())
    delta = _delta_from_mean(model, mean, source_value)
    rank = float(delta)
    return {"mean": mean, "logvar": lv, "rank": rank, "delta": delta}


def _edits_for(src: str, cand: str) -> List[dict]:
    """Canonical SUB edit list source->candidate for equal-length sequences."""
    edits = []
    s, c = normalize(src), normalize(cand)
    if len(s) != len(c):
        raise ValueError("R2 G1 critic supports equal-length SUB candidates only")
    for i in range(min(len(s), len(c))):
        if s[i] != c[i]:
            edits.append({"op": "SUB", "pos": i, "token": c[i]})
    return edits


def _edit_feat_for(src: str, cand: str) -> np.ndarray:
    edits = _edits_for(src, cand)
    return edit_features(edits, max(len(normalize(src)), 1))


def _context_id(mapping: Dict[str, int], value: Optional[str], name: str) -> int:
    if UNKNOWN_CONTEXT not in mapping:
        raise ValueError("R2 %s vocabulary has no reserved __UNK__ id" % name)
    if value in mapping:
        return int(mapping[value])
    return int(mapping[UNKNOWN_CONTEXT])


def _delta_from_mean(model, mean, source_value):
    """Apply the checkpoint's frozen regression-target semantics exactly."""
    target = getattr(getattr(model, "cfg", None), "TARGET", None)
    if target == "candidate_value":
        if source_value is None:
            raise ValueError(
                "candidate_value critic requires a measured source_value anchor"
            )
        return mean - source_value
    if target in (None, "delta"):
        # ``None`` is retained only for small test doubles; every production R2
        # checkpoint is schema-validated and has an explicit TARGET.
        return mean
    raise ValueError("unsupported R2 critic TARGET %r" % target)


def critic_scores_batch(critic, vocab, device, src_seq: str, cand_seqs: List[str],
                        study: Optional[str], endpoint: Optional[str],
                        bench: str, source_value: Optional[float]) -> Dict:
    """Frozen critic scores for many candidate seqs from one source (one forward).

    Returns dict of numpy arrays: mean, rank, logvar, and TARGET-aware delta.
    """
    B = len(cand_seqs)
    if B == 0:
        return {"mean": np.zeros(0), "rank": np.zeros(0),
                "logvar": np.zeros(0), "delta": np.zeros(0)}
    encoded = []
    edits_by_candidate = []
    for candidate in cand_seqs:
        edits = _edits_for(src_seq, candidate)
        edits_by_candidate.append(edits)
        encoded.append(encode_pair(src_seq, candidate, edits))
    src = torch.tensor(np.stack([item[0] for item in encoded])).to(device)
    cand = torch.tensor(np.stack([item[1] for item in encoded])).to(device)
    src_mask = torch.tensor(np.stack([item[2] for item in encoded])).to(device)
    cand_mask = torch.tensor(np.stack([item[3] for item in encoded])).to(device)
    ef = torch.tensor(np.stack([
        edit_features(edits, max(len(normalize(src_seq)), 1))
        for edits in edits_by_candidate
    ])).to(device)
    sid = torch.full((B,), _context_id(vocab["study"], study, "study"),
                     dtype=torch.long, device=device)
    eid = torch.full((B,), _context_id(vocab["endpoint"], endpoint, "endpoint"),
                     dtype=torch.long, device=device)
    bid = torch.full((B,), _context_id(vocab["benchmark"], bench, "benchmark"),
                     dtype=torch.long, device=device)
    with torch.no_grad():
        out = critic(src, cand, ef, sid, eid, bid,
                     src_pad_mask=src_mask, cand_pad_mask=cand_mask)
    mean = out["mean"].cpu().numpy()
    lv = out["logvar"].cpu().numpy()
    delta = _delta_from_mean(critic, mean, source_value)
    rank = np.asarray(delta)
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


def _guidance_head_action_effects(head, state, actions, device):
    """Score every legal action in its own edit-centred source window."""
    if not actions:
        return np.zeros(0, dtype=float)
    windows = []
    local_positions = []
    for action in actions:
        candidate = apply_action(state, action).seq
        edits = [{"op": "SUB", "pos": action.pos, "token": action.target_nt}]
        src_x, _, _, _, start = encode_pair(state.seq, candidate, edits)
        windows.append(src_x)
        local_positions.append(action.pos - start)
    effects = head.effects(torch.tensor(np.stack(windows)).to(device))
    return np.asarray([
        effects[i, local_positions[i], NUC_ORDER.index(action.target_nt)]
        for i, action in enumerate(actions)
    ], dtype=float)


def first_order_step_policy(base_net, head, beta, device, ctx=None):
    """first_order: base + beta * guidance-head per-action effect."""
    def policy(state, actions):
        bl = np.asarray(base_net.policy_fn(state, actions), dtype=float)
        effects = _guidance_head_action_effects(head, state, actions, device)
        gl = bl + beta * effects
        return {"guided_logits": gl, "base_logits": bl, "ratio": None,
                "critic_mean": None, "critic_logvar": None}
    return policy


def _ctx(**kw):
    """Default biological context (neutral ids) if a source has no provenance."""
    return {"study": UNKNOWN_CONTEXT, "endpoint": UNKNOWN_CONTEXT,
            "bench": UNKNOWN_CONTEXT,
            "source_value": None, **kw}


def _require_qualified_rank_supervision(critic) -> None:
    """Block every critic-driven G1 policy for an effect-only checkpoint."""
    metadata = getattr(critic, "rank_supervision", None)
    if (not isinstance(metadata, dict)
            or metadata.get("status") != "QUALIFIED"
            or int(metadata.get("qualified_rank_pair_count", 0)) <= 0):
        raise RuntimeError(
            "G1 critic policy requires qualified within-pool rank supervision; "
            "zero/unknown qualified pairs are effect-only and fail closed"
        )


def _critic_reward_policy(base_net, critic, vocab, beta, device,
                          use_rank: bool, ctx):
    """Critic guidance on predicted delta mean after rank qualification.

    ``use_rank`` is retained only so the pre-registered strategy labels remain
    stable.  Both branches consume the same predicted-delta score; no separate
    rank head exists in R3-P0.
    """
    _require_qualified_rank_supervision(critic)
    c = _ctx(**(ctx or {}))
    # Resolve conditioning ids once; unseen contexts route only to the trained
    # reserved UNK embedding.
    study = c["study"] if c["study"] in vocab["study"] else UNKNOWN_CONTEXT
    endpoint = (c["endpoint"] if c["endpoint"] in vocab["endpoint"]
                else UNKNOWN_CONTEXT)
    bench = c["bench"] if c["bench"] in vocab["benchmark"] else UNKNOWN_CONTEXT

    def policy(state, actions):
        bl = np.asarray(base_net.policy_fn(state, actions), dtype=float)
        cands = [apply_action(state, a).seq for a in actions]
        sc = critic_scores_batch(critic, vocab, device, state.source_seq, cands,
                                 study, endpoint, bench, c["source_value"])
        reward = sc["delta"]
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
    _require_qualified_rank_supervision(critic)
    c = _ctx(**(ctx or {}))
    study = c["study"] if c["study"] in vocab["study"] else UNKNOWN_CONTEXT
    endpoint = (c["endpoint"] if c["endpoint"] in vocab["endpoint"]
                else UNKNOWN_CONTEXT)
    bench = c["bench"] if c["bench"] in vocab["benchmark"] else UNKNOWN_CONTEXT

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
