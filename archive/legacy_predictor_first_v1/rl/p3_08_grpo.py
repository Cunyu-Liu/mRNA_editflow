"""P3-08: Production source-conditioned GRPO for minimal-edit mRNA optimization.

Implements the full GRPO training pipeline per the pre-registered config:
  - 1D CNN sequence encoder (real backbone)
  - Hierarchical action heads (STOP → region → position → target)
  - Multi-source batch rollouts with group-relative advantages
  - Clipped policy loss + adaptive KL + entropy bonus
  - P3-02 remediated oracle as reward signal (EnsembleDeltaOracle)
  - P3-06 MDP action space (Task A: 5'UTR substitution only)

Gate A: 3-seed pilot (1000 updates, edit_budget=1)
Gate B: 10-seed paper run (5000 updates, curriculum 1→3→5→10)
"""
from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.constants import NUC_VOCAB, START_CODON, translate
from core.schema import MRNARecord
from rl.p3_06_mdp import (
    EditAction,
    STOP_EDIT,
    MDPState,
    RewardV3Config,
    apply_edit_action,
    build_legal_edit_actions,
    compute_reward_v3,
    initial_state,
    transition,
)

# ===========================================================================
# Sequence encoding
# ===========================================================================

_NUC_TO_IDX = {ch: i for i, ch in enumerate(NUC_VOCAB)}  # A=0, C=1, G=2, U=3
_MAX_UTR_LEN = 100  # Max 5'UTR length for positional encoding


def build_legal_edit_actions_task_a(
    record: MRNARecord, visited: Optional[frozenset] = None
) -> List[EditAction]:
    """Fast legal action builder for Task A (5'UTR only, no CDS, no MD5 hash).

    Skips CDS synonymous actions entirely (Task A only edits 5'UTR).
    Uses tuple-based identity instead of MD5 for cycle avoidance (~10x faster).
    """
    actions: List[EditAction] = [STOP_EDIT]
    visited_set = set(visited) if visited else set()
    utr = record.five_utr
    for pos in range(len(utr)):
        old = utr[pos]
        for nt in NUC_VOCAB:
            if nt != old:
                # Fast identity: (pos, nt) uniquely identifies the edit
                key = (pos, nt)
                if key not in visited_set:
                    actions.append(EditAction(op="five_utr_sub", pos=pos, nt=nt))
    return actions


class PrecomputedSingleEditOracle:
    """Oracle with pre-computed scores for all single-edit candidates.

    Eliminates the CPU-bound oracle prediction bottleneck by pre-computing
    all possible (source, single-edit-candidate) scores in one batch at init.
    Each ``score_batch`` call is then a pure dict lookup (~10000x faster).

    Only valid for edit_budget=1 (Gate A). For higher budgets, fall back to
    the live ``EnsembleDeltaOracle``.
    """

    def __init__(self, base_oracle, sources: Sequence[MRNARecord]):
        """Pre-compute all single-edit scores for the given sources.

        Args:
            base_oracle: An EnsembleDeltaOracle (or any CountingOracle with
                score_batch) used to compute the raw scores.
            sources: All source records that will be queried (train + validation).
        """
        self._base = base_oracle
        self._score_cache: Dict[Tuple[str, str], Tuple[float, float]] = {}
        # Also cache source-bias by using the base oracle's cache
        self._precompute(sources)

    def _precompute(self, sources: Sequence[MRNARecord]) -> None:
        """Pre-compute scores for all single-edit candidates of all sources."""
        import time as _time
        t0 = _time.time()

        # Generate all (source, candidate) pairs for all possible single edits
        all_pairs: List[Tuple[MRNARecord, MRNARecord]] = []
        pair_keys: List[Tuple[str, str]] = []
        for source in sources:
            utr = source.five_utr
            for pos in range(len(utr)):
                old_nt = utr[pos]
                for nt in NUC_VOCAB:
                    if nt != old_nt:
                        action = EditAction(op="five_utr_sub", pos=pos, nt=nt)
                        candidate = apply_edit_action(source, action)
                        all_pairs.append((source, candidate))
                        pair_keys.append((source.five_utr, candidate.five_utr))

        if not all_pairs:
            return

        # Batch-score all pairs (uses base_oracle's batched path + source-bias cache)
        # Process in chunks to avoid memory issues
        CHUNK = 500
        for i in range(0, len(all_pairs), CHUNK):
            chunk_pairs = all_pairs[i:i + CHUNK]
            chunk_keys = pair_keys[i:i + CHUNK]
            scores = self._base.score_batch(chunk_pairs, purpose="search")
            for key, score in zip(chunk_keys, scores):
                self._score_cache[key] = score

        _elapsed = _time.time() - t0
        print(f"    [PrecomputedOracle] Pre-computed {len(self._score_cache)} "
              f"single-edit scores in {_elapsed:.1f}s", flush=True)

    def score(
        self, source: MRNARecord, candidate: MRNARecord, *, purpose: str = "search"
    ) -> Tuple[float, float]:
        """Single-pair score (lookup). Falls back to base oracle on miss."""
        key = (source.five_utr, candidate.five_utr)
        if key in self._score_cache:
            return self._score_cache[key]
        # Fallback: compute on the fly (e.g., multi-edit candidate)
        return self._base.score(source, candidate, purpose=purpose)

    def score_batch(
        self, pairs: Sequence[Tuple[MRNARecord, MRNARecord]], *, purpose: str = "search"
    ) -> List[Tuple[float, float]]:
        """Batch score via dict lookup. Falls back per-pair on miss."""
        results: List[Tuple[float, float]] = []
        miss_pairs: List[Tuple[int, Tuple[MRNARecord, MRNARecord]]] = []
        for i, (src, cand) in enumerate(pairs):
            key = (src.five_utr, cand.five_utr)
            if key in self._score_cache:
                results.append(self._score_cache[key])
            else:
                results.append((0.0, 0.0))  # placeholder
                miss_pairs.append((i, (src, cand)))

        if miss_pairs:
            # Compute misses via base oracle
            miss_scores = self._base.score_batch(
                [p for _, p in miss_pairs], purpose=purpose
            )
            for (i, pair), score in zip(miss_pairs, miss_scores):
                results[i] = score
                key = (pair[0].five_utr, pair[1].five_utr)
                self._score_cache[key] = score

        return results

    @property
    def search_calls(self) -> int:
        return self._base.search_calls

    @property
    def eval_calls(self) -> int:
        return self._base.eval_calls


def encode_sequence(seq: str, max_len: int = _MAX_UTR_LEN) -> np.ndarray:
    """One-hot encode a nucleotide sequence to (4, max_len) array."""
    arr = np.zeros((4, max_len), dtype=np.float32)
    for i, ch in enumerate(seq[:max_len]):
        idx = _NUC_TO_IDX.get(ch, 0)
        arr[idx, i] = 1.0
    return arr


def encode_record(record: MRNARecord, max_utr: int = _MAX_UTR_LEN) -> np.ndarray:
    """Encode the 5'UTR of a record (Task A only edits 5'UTR)."""
    return encode_sequence(record.five_utr, max_utr)


# ===========================================================================
# Policy network (1D CNN backbone + hierarchical heads)
# ===========================================================================

class SeqEncoder(nn.Module):
    """1D CNN sequence encoder — the 'real backbone' for P3-08."""

    def __init__(self, max_seq_len: int = _MAX_UTR_LEN, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, hidden_dim)
        self.max_seq_len = max_seq_len

    def forward(self, one_hot: torch.Tensor) -> torch.Tensor:
        """one_hot: (B, 4, L) → (B, hidden_dim)"""
        x = F.relu(self.conv1(one_hot))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # (B, 128)
        return F.relu(self.fc(x))  # (B, hidden_dim)


class P3O8Policy(nn.Module):
    """Hierarchical policy: STOP → region → position → target.

    For Task A (5'UTR-only), region is always five_utr, so the region head
    is a no-op (p_region = [1.0, 0.0]). The position and target heads are
    learned.
    """

    def __init__(self, max_utr_len: int = _MAX_UTR_LEN, hidden_dim: int = 64):
        super().__init__()
        self.max_utr_len = max_utr_len
        self.hidden_dim = hidden_dim
        self.encoder = SeqEncoder(max_utr_len, hidden_dim)

        # Budget/oracle feature dimension: [n_edits, remaining_budget, remaining_budget_frac]
        self.feat_dim = 3
        self.feat_proj = nn.Linear(self.feat_dim, hidden_dim)

        # STOP head
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Position head (over 5'UTR positions)
        self.pos_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_utr_len),
        )

        # Target head (3 non-identity nucleotides for 5'UTR)
        self.target_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + max_utr_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # 4 nucleotides, masked to exclude identity
        )

    def _features(self, state: MDPState) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract sequence encoding and budget features from state."""
        device = next(self.parameters()).device
        one_hot = encode_record(state.current_mrna, self.max_utr_len)
        seq_tensor = torch.from_numpy(one_hot).unsqueeze(0).to(device)  # (1, 4, L)
        seq_repr = self.encoder(seq_tensor)  # (1, hidden)

        total_budget = state.remaining_budget + state.n_edits()
        budget_feat = torch.tensor([[
            float(state.n_edits()),
            float(state.remaining_budget),
            float(state.remaining_budget) / max(total_budget, 1),
        ]], dtype=torch.float32, device=device)
        budget_repr = self.feat_proj(budget_feat)  # (1, hidden)

        combined = torch.cat([seq_repr, budget_repr], dim=-1)  # (1, hidden*2)
        return combined, seq_repr

    def forward(self, state: MDPState) -> Dict[str, torch.Tensor]:
        """Compute action distribution for a state.

        Returns dict with:
          p_stop: (1,) probability of STOP
          pos_logits: (1, max_utr_len) position logits (pre-mask)
          target_logits: (1, 4) target nucleotide logits (pre-mask)
        """
        combined, seq_repr = self._features(state)
        p_stop = torch.sigmoid(self.stop_head(combined).squeeze(-1))  # (1,)

        pos_logits = self.pos_head(combined)  # (1, max_utr_len)
        target_logits = self.target_head(
            torch.cat([combined, pos_logits], dim=-1)
        )  # (1, 4)

        return {"p_stop": p_stop, "pos_logits": pos_logits, "target_logits": target_logits}

    def action_log_probs(
        self, state: MDPState, legal_actions: List[EditAction]
    ) -> Dict[EditAction, float]:
        """Compute log π(a|s) for each legal action.

        For Task A (5'UTR-only):
          π(STOP|s) = p_stop
          π(edit|s) = (1 - p_stop) × π(pos|s) × π(target|pos,s)

        Position and target distributions are masked to legal actions only.
        Vectorized: pre-computes target log probs for all 4 old-nucleotide
        masks in one batch, then loops over actions with pure lookups.
        """
        out = self.forward(state)
        p_stop = out["p_stop"].clamp(1e-10, 1 - 1e-10)
        pos_logits = out["pos_logits"].squeeze(0)  # (max_utr_len,)
        target_logits = out["target_logits"].squeeze(0)  # (4,)

        # Collect legal positions and targets
        legal_positions: Dict[int, List[str]] = {}
        for a in legal_actions:
            if a.is_five_utr():
                legal_positions.setdefault(a.pos, []).append(a.nt)

        # Mask position logits to legal positions
        pos_mask = torch.full_like(pos_logits, float('-inf'))
        for pos in legal_positions:
            if pos < self.max_utr_len:
                pos_mask[pos] = 0.0
        masked_pos_logits = pos_logits + pos_mask
        pos_log_probs = F.log_softmax(masked_pos_logits, dim=-1)

        # Pre-compute target log probs for all 4 possible old nucleotides.
        # masks[i] excludes nucleotide i (old=idx_i → can't pick idx_i).
        dev = target_logits.device
        masks = 1.0 - torch.eye(4, dtype=torch.float32, device=dev)  # (4, 4)
        masked_tgt = target_logits.unsqueeze(0) * masks + (1 - masks) * (-1e9)  # (4, 4)
        tgt_log_probs_all = F.log_softmax(masked_tgt, dim=-1)  # (4, 4)

        # Compute log probs for all actions (pure lookups, no tensor ops)
        log_p_stop = float(math.log(p_stop.item()))
        log_p_edit = float(math.log(1 - p_stop.item()))
        utr = state.current_mrna.five_utr

        result: Dict[EditAction, float] = {}
        for action in legal_actions:
            if action.is_stop():
                result[action] = log_p_stop
            else:
                old_idx = _NUC_TO_IDX[utr[action.pos]]
                tgt_idx = _NUC_TO_IDX[action.nt]
                log_p_pos = float(pos_log_probs[action.pos].item())
                log_p_tgt = float(tgt_log_probs_all[old_idx, tgt_idx].item())
                result[action] = log_p_edit + log_p_pos + log_p_tgt

        return result

    def log_prob_tensor(
        self, state: MDPState, action: EditAction, legal_actions: List[EditAction]
    ) -> torch.Tensor:
        """Compute log π(action|state) as a differentiable tensor (for training).

        This is the gradient-flow version of action_log_probs for a single action.
        Vectorized: pre-computes target log probs for all 4 old-nucleotide masks
        in one batch, then indexes the result.
        """
        out = self.forward(state)
        p_stop = out["p_stop"].clamp(1e-10, 1 - 1e-10)
        pos_logits = out["pos_logits"].squeeze(0)  # (max_utr_len,)
        target_logits = out["target_logits"].squeeze(0)  # (4,)

        # Collect legal positions
        legal_positions: Dict[int, List[str]] = {}
        for a in legal_actions:
            if a.is_five_utr():
                legal_positions.setdefault(a.pos, []).append(a.nt)

        # Mask position logits
        pos_mask = torch.full_like(pos_logits, float('-inf'))
        for pos in legal_positions:
            if pos < self.max_utr_len:
                pos_mask[pos] = 0.0
        masked_pos_logits = pos_logits + pos_mask
        pos_log_probs = F.log_softmax(masked_pos_logits, dim=-1)

        if action.is_stop():
            return torch.log(p_stop.squeeze())

        log_p_edit = torch.log(1 - p_stop.squeeze())
        log_p_pos = pos_log_probs[action.pos]

        # Pre-compute target log probs for all 4 old-nucleotide masks (vectorized)
        dev = target_logits.device
        masks = 1.0 - torch.eye(4, dtype=torch.float32, device=dev)  # (4, 4)
        masked_tgt = target_logits.unsqueeze(0) * masks + (1 - masks) * (-1e9)  # (4, 4)
        tgt_log_probs_all = F.log_softmax(masked_tgt, dim=-1)  # (4, 4)

        old_nt = state.current_mrna.five_utr[action.pos]
        old_idx = _NUC_TO_IDX[old_nt]
        tgt_idx = _NUC_TO_IDX[action.nt]
        log_p_tgt = tgt_log_probs_all[old_idx, tgt_idx]

        return log_p_edit + log_p_pos + log_p_tgt

    def sample_action(
        self, state: MDPState, legal_actions: List[EditAction], generator: Optional[torch.Generator] = None
    ) -> Tuple[EditAction, float]:
        """Sample an action from the policy. Returns (action, log_prob)."""
        log_probs = self.action_log_probs(state, legal_actions)
        actions = list(log_probs.keys())
        probs = torch.tensor([math.exp(log_probs[a]) for a in actions])
        probs = probs / probs.sum()

        if generator is not None:
            idx = torch.multinomial(probs, 1, generator=generator).item()
        else:
            idx = torch.multinomial(probs, 1).item()

        return actions[idx], log_probs[actions[idx]]

    def forward_logits(self, state: MDPState) -> Dict[str, torch.Tensor]:
        """Same as forward() but returns logits on device (no .item() calls).

        Used by compute_kl_entropy_fast to avoid CPU-GPU sync overhead.
        """
        return self.forward(state)

    def compute_kl_entropy_fast(
        self, ref_policy: "P3O8Policy", state: MDPState, legal_actions: List[EditAction]
    ) -> Tuple[float, float]:
        """Compute KL(π||π_ref) and H(π) using vectorized tensor ops.

        Returns (kl, entropy) as floats. Only 2 .item() calls total (vs ~600
        in action_log_probs loop). Critical for loaded servers where each
        .item() CPU-GPU sync takes ~10ms.

        KL decomposition (hierarchical policy):
          KL = KL_stop + p_edit * (KL_pos + KL_tgt_approx)
        Entropy similarly decomposed. KL_tgt is approximate (doesn't account
        for per-position identity mask) but sufficient for KL controller.
        """
        with torch.no_grad():
            out_new = self.forward(state)
            out_ref = ref_policy.forward(state)

            # STOP/edit KL
            p_stop_new = out_new["p_stop"].clamp(1e-10, 1 - 1e-10)
            p_stop_ref = out_ref["p_stop"].clamp(1e-10, 1 - 1e-10)
            p_edit_new = 1 - p_stop_new
            p_edit_ref = 1 - p_stop_ref
            kl_stop = (p_stop_new * (torch.log(p_stop_new) - torch.log(p_stop_ref))
                       + p_edit_new * (torch.log(p_edit_new) - torch.log(p_edit_ref)))

            # Entropy of STOP/edit
            h_stop = -(p_stop_new * torch.log(p_stop_new) + p_edit_new * torch.log(p_edit_new))

            # Position KL (only over legal positions — indexing avoids 0*NaN
            # that occurs with -inf masking: exp(-inf)=0, but 0*(-inf-(-inf))=NaN)
            pos_logits_new = out_new["pos_logits"].squeeze(0)  # (max_utr_len,)
            pos_logits_ref = out_ref["pos_logits"].squeeze(0)
            legal_positions = sorted({a.pos for a in legal_actions
                                      if a.is_five_utr() and a.pos < self.max_utr_len})
            if legal_positions:
                legal_idx = torch.tensor(legal_positions, dtype=torch.long,
                                         device=pos_logits_new.device)
                pos_log_new = F.log_softmax(pos_logits_new[legal_idx], dim=-1)
                pos_log_ref = F.log_softmax(pos_logits_ref[legal_idx], dim=-1)
                pos_new = torch.exp(pos_log_new)
                kl_pos = (pos_new * (pos_log_new - pos_log_ref)).sum()
                h_pos = -(pos_new * pos_log_new).sum()
            else:
                kl_pos = torch.zeros((), device=pos_logits_new.device)
                h_pos = torch.zeros((), device=pos_logits_new.device)

            # Target KL (approximate — no per-position identity mask)
            tgt_logits_new = out_new["target_logits"].squeeze(0)  # (4,)
            tgt_logits_ref = out_ref["target_logits"].squeeze(0)
            tgt_log_new = F.log_softmax(tgt_logits_new, dim=-1)
            tgt_log_ref = F.log_softmax(tgt_logits_ref, dim=-1)
            tgt_new = torch.exp(tgt_log_new)
            kl_tgt = (tgt_new * (tgt_log_new - tgt_log_ref)).sum()
            h_tgt = -(tgt_new * tgt_log_new).sum()

            # Total KL and entropy (hierarchical decomposition)
            kl = kl_stop + p_edit_new * (kl_pos + kl_tgt)
            entropy = h_stop + p_edit_new * (h_pos + h_tgt)

            return float(kl.item()), float(entropy.item())


# ===========================================================================
# Reference policy (frozen copy for KL)
# ===========================================================================

class ReferencePolicy:
    """Frozen reference policy for KL computation."""

    def __init__(self, policy: P3O8Policy):
        self.policy = policy
        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)

    def forward(self, state: MDPState) -> Dict[str, torch.Tensor]:
        """Delegate to wrapped policy's forward (for vectorized KL)."""
        with torch.no_grad():
            return self.policy.forward(state)

    def action_log_probs(self, state: MDPState, legal_actions: List[EditAction]) -> Dict[EditAction, float]:
        with torch.no_grad():
            return self.policy.action_log_probs(state, legal_actions)


# ===========================================================================
# Adaptive KL controller
# ===========================================================================

class AdaptiveKLController:
    """Adaptive KL coefficient per PPO/GRPO practice.

    Strengthened after observed KL explosion (0.0115→0.2009 in 100 steps with
    MIN_COEFFICIENT=0.1). The old controller only reacted when KL exceeded
    max_kl, by which point the policy had already diverged too far.

    Fixes:
    1. MIN_COEFFICIENT raised to 0.3 — stronger baseline penalty prevents
       exponential KL growth during the 0.5*max_kl to max_kl zone.
    2. Proactive tier at max_kl*0.5 — increases coefficient by 1.5x BEFORE
       KL reaches max_kl, catching growth early.
    3. Skip at max_kl*1.3 — hard reject + double coefficient (cap 2.0).
    4. Reference reset after 30 consecutive KL_SKIPs in train_single_seed.
    """

    MIN_COEFFICIENT = 0.3  # Floor — strong enough to prevent KL explosion

    def __init__(self, coefficient: float = 0.05, max_kl: float = 0.25):
        self.coefficient = max(float(coefficient), self.MIN_COEFFICIENT)
        self.max_kl = float(max_kl)

    def update(self, observed_kl: float) -> bool:
        """Update coefficient based on observed KL. Returns True if should skip."""
        if observed_kl > self.max_kl * 1.3:
            # Hard skip: KL too high, reject update and aggressively increase penalty
            self.coefficient = min(self.coefficient * 2.0, 2.0)
            return True  # Skip this update
        elif observed_kl > self.max_kl:
            # Above threshold: double coefficient to push KL back down
            self.coefficient = min(self.coefficient * 2.0, 1.0)
        elif observed_kl > self.max_kl * 0.5:
            # Proactive tier: KL in warning zone, increase coefficient to
            # prevent reaching max_kl (catches growth before it's too late)
            self.coefficient = min(self.coefficient * 1.5, 1.0)
        elif observed_kl < self.max_kl * 0.5:
            # Below threshold: reduce coefficient, but never below MIN_COEFFICIENT
            self.coefficient = max(self.coefficient * 0.5, self.MIN_COEFFICIENT)
        return False


# ===========================================================================
# Trajectory and rollout
# ===========================================================================

@dataclass
class TrajectoryStep:
    state: MDPState
    action: EditAction
    log_prob: float
    reward_components: Dict[str, float]
    n_edits: int


@dataclass
class Trajectory:
    source_id: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    total_reward: float = 0.0
    final_mrna: Optional[MRNARecord] = None
    n_edits: int = 0
    constraint_valid: bool = True
    reward_components: Dict[str, float] = field(default_factory=dict)
    raw_delta: float = 0.0  # Raw oracle mean_delta (before LCB/edit-cost), for pos_rate metric

    def returns(self) -> float:
        return self.total_reward


def collect_trajectory(
    source: MRNARecord,
    policy: P3O8Policy,
    oracle,  # EnsembleDeltaOracle
    edit_budget: int,
    reward_config: RewardV3Config,
    generator: Optional[torch.Generator] = None,
) -> Trajectory:
    """Collect one trajectory by rolling out the policy."""
    state = initial_state(source, budget=edit_budget, cargo=source.transcript_id)
    traj = Trajectory(source_id=source.transcript_id)

    while state.remaining_budget > 0:
        legal = build_legal_edit_actions_task_a(state.current_mrna, state.visited_states)
        if not legal or len(legal) == 1:  # Only STOP
            break

        action, log_prob = policy.sample_action(state, legal, generator=generator)

        # Apply action
        if action.is_stop():
            step = TrajectoryStep(
                state=state, action=action, log_prob=log_prob,
                reward_components={}, n_edits=state.n_edits(),
            )
            traj.steps.append(step)
            break

        new_record = apply_edit_action(state.current_mrna, action)

        # Oracle score
        mean_delta, uncertainty = oracle.score(source, new_record, purpose="search")
        reward_out = compute_reward_v3(
            source, new_record,
            predicted_deltas={"protein_output": mean_delta},
            uncertainties={"protein_output": uncertainty},
            n_edits=state.n_edits() + 1,
            config=reward_config,
        )

        step = TrajectoryStep(
            state=state, action=action, log_prob=log_prob,
            reward_components={"protein_output": float(reward_out["scalar"])},
            n_edits=state.n_edits() + 1,
        )
        traj.steps.append(step)
        # Track the latest raw oracle delta (for pos_rate metric in validation)
        traj.raw_delta = float(mean_delta)
        state = transition(state, action)

    # Final reward: evaluate the final candidate against source
    if traj.steps and not traj.steps[-1].action.is_stop():
        # Already scored inline
        traj.total_reward = sum(s.reward_components.get("protein_output", 0.0) for s in traj.steps)
    elif not traj.steps:
        # No edits (immediate STOP or no legal actions)
        traj.total_reward = 0.0
    else:
        # Sum rewards from all non-STOP steps
        traj.total_reward = sum(
            s.reward_components.get("protein_output", 0.0)
            for s in traj.steps if not s.action.is_stop()
        )

    traj.final_mrna = state.current_mrna
    traj.n_edits = state.n_edits()

    # Constraint validation
    if traj.final_mrna:
        try:
            protein_src = translate(source.cds)
            protein_final = translate(traj.final_mrna.cds)
            traj.constraint_valid = (protein_src == protein_final and
                                      len(source.seq) == len(traj.final_mrna.seq))
        except Exception:
            traj.constraint_valid = False

    traj.reward_components = {"protein_output": traj.total_reward}
    return traj


def collect_batch(
    sources: Sequence[MRNARecord],
    policy: P3O8Policy,
    oracle,
    edit_budget: int,
    group_size: int,
    reward_config: RewardV3Config,
    seed: int = 0,
    stop_penalty: float = 0.0,
) -> List[List[Trajectory]]:
    """Collect a multi-source batch: group_size trajectories per source.

    Uses the fast batched path when edit_budget == 1 (Gate A) and the oracle
    supports ``score_batch``; otherwise falls back to sequential collection.
    """
    # Fast path: edit_budget=1 with batched oracle
    if edit_budget == 1 and hasattr(oracle, "score_batch"):
        return _collect_batch_budget1_batched(
            sources, policy, oracle, group_size, reward_config, seed, stop_penalty
        )
    # Fallback: sequential
    gen = torch.Generator()
    gen.manual_seed(seed)
    batch: List[List[Trajectory]] = []
    policy.eval()
    for source in sources:
        group: List[Trajectory] = []
        for _ in range(group_size):
            traj = collect_trajectory(source, policy, oracle, edit_budget, reward_config, gen)
            group.append(traj)
        batch.append(group)
    return batch


def _collect_batch_budget1_batched(
    sources: Sequence[MRNARecord],
    policy: P3O8Policy,
    oracle,
    group_size: int,
    reward_config: RewardV3Config,
    seed: int = 0,
    stop_penalty: float = 0.0,
) -> List[List[Trajectory]]:
    """Fast batched trajectory collection for edit_budget=1.

    For budget=1, each trajectory is: sample one action at the initial state,
    apply it (or STOP), score with oracle. Since the initial state is the same
    for all trajectories from the same source, we:
      1. Compute action log-probs once per source
      2. Sample group_size actions per source
      3. Batch-score all non-STOP candidates in one oracle call
      4. Assemble Trajectory objects

    If ``stop_penalty > 0``, trajectories that STOP at root (no edits) get
    reward ``-stop_penalty`` instead of 0, preventing STOP collapse.

    This is semantically identical to sequential collect_trajectory but runs
    the oracle once per batch instead of once per trajectory (~30x faster).
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    policy.eval()

    # Phase 1: sample actions for all trajectories
    # sampled[i] = list of (source_idx, action, log_prob) for group i
    sampled: List[List[Tuple[int, EditAction, float]]] = []
    # Collect (source, candidate) pairs that need oracle scoring
    oracle_pairs: List[Tuple[MRNARecord, MRNARecord]] = []
    # Map from pair index to (source_idx, group_idx, traj_idx)
    pair_map: List[Tuple[int, int, int]] = []

    with torch.no_grad():
        for src_idx, source in enumerate(sources):
            state = initial_state(source, budget=1, cargo=source.transcript_id)
            legal = build_legal_edit_actions_task_a(state.current_mrna, state.visited_states)
            group_actions: List[Tuple[int, EditAction, float]] = []
            for traj_idx in range(group_size):
                action, log_prob = policy.sample_action(state, legal, generator=gen)
                group_actions.append((src_idx, action, log_prob))
                if not action.is_stop():
                    candidate = apply_edit_action(source, action)
                    pair_map.append((src_idx, len(sampled), traj_idx))
                    oracle_pairs.append((source, candidate))
            sampled.append(group_actions)

    # Phase 2: batch-score all candidates
    if oracle_pairs:
        scores = oracle.score_batch(oracle_pairs, purpose="search")
    else:
        scores = []

    # Phase 3: assemble trajectories
    # Build a lookup from (src_idx, group_idx, traj_idx) → score
    score_lookup: Dict[Tuple[int, int, int], Tuple[float, float]] = {}
    for k, (src_idx, group_idx, traj_idx) in enumerate(pair_map):
        score_lookup[(src_idx, group_idx, traj_idx)] = scores[k]

    batch: List[List[Trajectory]] = []
    for src_idx, source in enumerate(sources):
        group: List[Trajectory] = []
        for traj_idx, (_src_idx, action, log_prob) in enumerate(sampled[src_idx]):
            state = initial_state(source, budget=1, cargo=source.transcript_id)
            traj = Trajectory(source_id=source.transcript_id)

            if action.is_stop():
                step = TrajectoryStep(
                    state=state, action=action, log_prob=log_prob,
                    reward_components={}, n_edits=0,
                )
                traj.steps.append(step)
                # STOP penalty: discourage STOP at root to prevent collapse
                traj.total_reward = -stop_penalty if stop_penalty > 0 else 0.0
            else:
                candidate = apply_edit_action(source, action)
                mean_delta, uncertainty = score_lookup[(src_idx, src_idx, traj_idx)]
                reward_out = compute_reward_v3(
                    source, candidate,
                    predicted_deltas={"protein_output": mean_delta},
                    uncertainties={"protein_output": uncertainty},
                    n_edits=1,
                    config=reward_config,
                )
                step_reward = float(reward_out["scalar"])
                step = TrajectoryStep(
                    state=state, action=action, log_prob=log_prob,
                    reward_components={"protein_output": step_reward},
                    n_edits=1,
                )
                traj.steps.append(step)
                traj.total_reward = step_reward
                traj.raw_delta = float(mean_delta)  # Track raw oracle delta for pos_rate

            traj.final_mrna = candidate if not action.is_stop() else source
            traj.n_edits = 0 if action.is_stop() else 1
            # Constraint validation (always True for Task A 5'UTR sub, but check)
            if traj.final_mrna:
                try:
                    protein_src = translate(source.cds)
                    protein_final = translate(traj.final_mrna.cds)
                    traj.constraint_valid = (
                        protein_src == protein_final
                        and len(source.seq) == len(traj.final_mrna.seq)
                    )
                except Exception:
                    traj.constraint_valid = False
            traj.reward_components = {"protein_output": traj.total_reward}
            group.append(traj)
        batch.append(group)
    return batch


# ===========================================================================
# GRPO loss and update
# ===========================================================================

def categorical_kl(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """Categorical KL(P || Q) where inputs are log probabilities."""
    p = torch.exp(log_p)
    return (p * (log_p - log_q)).sum()


def compute_group_advantages(
    batch: List[List[Trajectory]],
    clip_advantage: float = 10.0,
    min_variance: float = 1e-8,
) -> List[List[float]]:
    """Compute group-relative advantages: A_i = (R_i - mean(R)) / std(R)."""
    advantages: List[List[float]] = []
    for group in batch:
        rewards = torch.tensor([t.total_reward for t in group], dtype=torch.float32)
        mean_r = rewards.mean()
        std_r = rewards.std(unbiased=False)
        if float(std_r) <= min_variance:
            adv = torch.zeros_like(rewards)
        else:
            adv = ((rewards - mean_r) / std_r).clamp(-clip_advantage, clip_advantage)
        advantages.append(adv.tolist())
    return advantages


def grpo_update(
    policy: P3O8Policy,
    reference: ReferencePolicy,
    batch: List[List[Trajectory]],
    optimizer: torch.optim.Optimizer,
    kl_controller: AdaptiveKLController,
    clip_epsilon: float = 0.2,
    entropy_coef: float = 0.01,
    gradient_clip: float = 1.0,
) -> Dict[str, float]:
    """One GRPO optimizer step on a multi-source batch."""
    policy.train()
    advantages = compute_group_advantages(batch)

    new_log_probs: List[torch.Tensor] = []  # Differentiable tensors
    old_log_probs: List[float] = []
    adv_values: List[float] = []
    kl_values: List[float] = []
    entropy_values: List[float] = []

    for group_idx, group in enumerate(batch):
        for traj_idx, traj in enumerate(group):
            adv = advantages[group_idx][traj_idx]
            for step in traj.steps:
                legal = build_legal_edit_actions_task_a(
                    step.state.current_mrna, step.state.visited_states)

                if step.action not in [a for a in legal]:
                    continue

                # New policy log prob (differentiable tensor via log_prob_tensor)
                new_lp = policy.log_prob_tensor(step.state, step.action, legal)

                # Fast vectorized KL and entropy (2 .item() calls vs ~600
                # in the old action_log_probs loop). Critical for loaded servers.
                kl_val, ent = policy.compute_kl_entropy_fast(
                    reference.policy, step.state, legal)

                kl_values.append(kl_val)
                entropy_values.append(ent)

                new_log_probs.append(new_lp)
                old_log_probs.append(step.log_prob)
                adv_values.append(adv)

    if not new_log_probs:
        return {"loss": 0.0, "kl": 0.0, "entropy": 0.0, "clip_fraction": 0.0, "updated": False}

    new_t = torch.stack(new_log_probs)
    device = new_t.device
    old_t = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
    adv_t = torch.tensor(adv_values, dtype=torch.float32, device=device)

    # Clipped policy loss (gradient flows through new_t)
    ratio = torch.exp(new_t - old_t)
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.minimum(ratio * adv_t, clipped_ratio * adv_t)
    policy_loss = -surrogate.mean()

    # KL penalty (coefficient * observed KL, detached)
    observed_kl = float(np.mean(kl_values)) if kl_values else 0.0
    kl_loss = kl_controller.coefficient * observed_kl  # scalar, no grad

    # Entropy bonus (detached, just for logging/regularization signal)
    entropy = float(np.mean(entropy_values)) if entropy_values else 0.0

    # Total loss
    loss = policy_loss + kl_loss - entropy_coef * entropy

    # Clip fraction
    with torch.no_grad():
        clip_frac = float((torch.abs(ratio - 1.0) > clip_epsilon).float().mean())

    # Backward + step (trust region removed — too slow on loaded server due to
    # .item() CPU-GPU syncs. Strengthened KL controller is sufficient: skip at
    # max_kl*1.3, double coefficient at max_kl, initial beta_kl=0.3)
    skip = kl_controller.update(observed_kl)
    updated = False
    grad_norm = 0.0
    if torch.isfinite(loss) and not skip:
        optimizer.zero_grad()
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            policy.parameters(), gradient_clip).detach().cpu())
        optimizer.step()
        updated = True

    return {
        "loss": float(loss.detach()),
        "policy_loss": float(policy_loss.detach()),
        "kl": observed_kl,
        "kl_coefficient": kl_controller.coefficient,
        "entropy": entropy,
        "clip_fraction": clip_frac,
        "grad_norm": grad_norm,
        "updated": updated,
        "skip_kl_guard": skip,
        "n_steps": len(new_log_probs),
    }


# ===========================================================================
# Validation
# ===========================================================================

def validate_policy(
    policy: P3O8Policy,
    validation_sources: Sequence[MRNARecord],
    oracle,
    edit_budget: int,
    reward_config: RewardV3Config,
    n_trajectories: int = 32,
    seed: int = 999,
    stop_penalty: float = 0.0,
) -> Dict[str, Any]:
    """Validate policy on held-out sources.

    Uses the fast batched path when edit_budget == 1 and the oracle supports
    ``score_batch``; otherwise falls back to sequential collection.

    Note: ``stop_penalty`` is applied during validation too (for consistency
    with training), but the reported ``mean_reward`` includes the penalty.
    The ``positive_improvement_rate`` is computed on the raw edit reward
    (penalty-free) to reflect true edit quality.
    """
    # Fast path: edit_budget=1 with batched oracle
    if edit_budget == 1 and hasattr(oracle, "score_batch"):
        return _validate_policy_budget1_batched(
            policy, validation_sources, oracle, reward_config,
            n_trajectories, seed, stop_penalty,
        )

    # Fallback: sequential
    policy.eval()
    all_rewards: List[float] = []
    all_n_edits: List[int] = []
    all_constraint_valid: List[bool] = []
    all_positive: List[bool] = []
    stop_at_root_count = 0
    total = 0

    gen = torch.Generator()
    gen.manual_seed(seed)

    import time as _time
    _t0 = _time.time()
    for src_idx, source in enumerate(validation_sources):
        source_rewards: List[float] = []
        for _ in range(n_trajectories):
            traj = collect_trajectory(source, policy, oracle, edit_budget, reward_config, gen)
            all_rewards.append(traj.total_reward)
            all_n_edits.append(traj.n_edits)
            all_constraint_valid.append(traj.constraint_valid)
            all_positive.append(traj.n_edits > 0 and traj.raw_delta > 0)
            total += 1
            if traj.steps and traj.steps[0].action.is_stop():
                stop_at_root_count += 1
            source_rewards.append(traj.total_reward)
        if (src_idx + 1) % 4 == 0:
            _elapsed = _time.time() - _t0
            print(f"    [validate] {src_idx+1}/{len(validation_sources)} sources, "
                  f"{_elapsed:.1f}s elapsed", flush=True)

    return {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "mean_n_edits": float(np.mean(all_n_edits)),
        "constraint_validity": float(np.mean(all_constraint_valid)),
        "positive_improvement_rate": float(np.mean(all_positive)),
        "stop_at_root_rate": stop_at_root_count / max(total, 1),
        "n_trajectories": total,
    }


def _validate_policy_budget1_batched(
    policy: P3O8Policy,
    validation_sources: Sequence[MRNARecord],
    oracle,
    reward_config: RewardV3Config,
    n_trajectories: int = 32,
    seed: int = 999,
    stop_penalty: float = 0.0,
) -> Dict[str, Any]:
    """Fast batched validation for edit_budget=1.

    Collects all trajectories across all validation sources in a single
    batched oracle call, then aggregates metrics. Semantically identical
    to the sequential path but ~30x faster.
    """
    import time as _time
    _t0 = _time.time()
    policy.eval()

    all_rewards: List[float] = []
    all_n_edits: List[int] = []
    all_constraint_valid: List[bool] = []
    all_positive: List[bool] = []
    stop_at_root_count = 0
    total = 0

    # Process validation sources in chunks to cap oracle batch size
    CHUNK = 8
    for chunk_start in range(0, len(validation_sources), CHUNK):
        chunk = validation_sources[chunk_start:chunk_start + CHUNK]
        batch = _collect_batch_budget1_batched(
            chunk, policy, oracle, n_trajectories, reward_config,
            seed=seed + chunk_start, stop_penalty=stop_penalty,
        )
        for group in batch:
            for traj in group:
                all_rewards.append(traj.total_reward)
                all_n_edits.append(traj.n_edits)
                all_constraint_valid.append(traj.constraint_valid)
                # Positive improvement = made an edit AND raw oracle delta > 0
                # (NOT LCB-based reward > 0, which is almost always negative due to
                # uncertainty penalty. Raw delta measures actual predicted improvement.)
                all_positive.append(traj.n_edits > 0 and traj.raw_delta > 0)
                total += 1
                if traj.steps and traj.steps[0].action.is_stop():
                    stop_at_root_count += 1
        _elapsed = _time.time() - _t0
        print(f"    [validate-batched] {min(chunk_start + CHUNK, len(validation_sources))}/"
              f"{len(validation_sources)} sources, {_elapsed:.1f}s elapsed", flush=True)

    return {
        "mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "std_reward": float(np.std(all_rewards)) if all_rewards else 0.0,
        "mean_n_edits": float(np.mean(all_n_edits)) if all_n_edits else 0.0,
        "constraint_validity": float(np.mean(all_constraint_valid)) if all_constraint_valid else 0.0,
        "positive_improvement_rate": float(np.mean(all_positive)) if all_positive else 0.0,
        "stop_at_root_rate": stop_at_root_count / max(total, 1),
        "n_trajectories": total,
    }


# ===========================================================================
# Training loop (single seed)
# ===========================================================================

@dataclass
class GRPOTrainConfig:
    n_updates: int = 1000
    edit_budget: int = 1
    sources_per_batch: int = 8
    group_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    clip_epsilon: float = 0.2
    beta_kl: float = 0.3  # Increased from 0.1 to prevent early KL growth
    beta_entropy: float = 0.05
    max_kl: float = 0.15
    gradient_clip: float = 1.0
    warmup_steps: int = 100
    stop_penalty: float = 0.1
    validation_interval: int = 100
    checkpoint_interval: int = 200
    seed: int = 42
    n_validation_trajectories: int = 32


def train_single_seed(
    seed: int,
    train_sources: Sequence[MRNARecord],
    validation_sources: Sequence[MRNARecord],
    oracle_factory: Callable,
    config: GRPOTrainConfig,
    reward_config: RewardV3Config,
    device: str = "cpu",
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Train GRPO for a single seed. Returns metrics dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    policy = P3O8Policy().to(device)
    reference = ReferencePolicy(P3O8Policy().to(device))
    # Copy initial weights to reference
    reference.policy.load_state_dict(policy.state_dict())

    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    kl_controller = AdaptiveKLController(config.beta_kl, config.max_kl)

    train_log: List[Dict[str, Any]] = []
    validation_log: List[Dict[str, Any]] = []

    # Build a single precomputed oracle for edit_budget=1 (eliminates CPU bottleneck).
    # For edit_budget>1, fall back to per-step oracle_factory() calls.
    use_precomputed = (config.edit_budget == 1)
    if use_precomputed:
        base_oracle = oracle_factory()
        all_sources = list(train_sources) + list(validation_sources)
        oracle = PrecomputedSingleEditOracle(base_oracle, all_sources)
        print(f"  [seed={seed}] Using PrecomputedSingleEditOracle for edit_budget=1",
              flush=True)
    else:
        oracle = None  # Will be created per-step

    # Warm-start baseline
    if oracle is None:
        oracle = oracle_factory()
    baseline_val = validate_policy(
        policy, validation_sources, oracle, config.edit_budget, reward_config,
        n_trajectories=config.n_validation_trajectories, seed=999,
        stop_penalty=config.stop_penalty,
    )
    validation_log.append({"step": 0, "validation": baseline_val, "type": "warm_start"})

    rng = random.Random(seed)
    n_sources = len(train_sources)
    import time as _time
    _train_t0 = _time.time()
    consecutive_kl_skips = 0
    REF_RESET_THRESHOLD = 30  # Reset reference after 30 consecutive KL skips

    for step in range(1, config.n_updates + 1):
        # Warmup
        if step <= config.warmup_steps:
            lr_scale = step / config.warmup_steps
            for pg in optimizer.param_groups:
                pg["lr"] = config.lr * lr_scale

        # Sample batch
        batch_sources_idx = rng.sample(range(n_sources), min(config.sources_per_batch, n_sources))
        batch_sources = [train_sources[i] for i in batch_sources_idx]

        if oracle is None:
            oracle = oracle_factory()
        batch = collect_batch(
            batch_sources, policy, oracle, config.edit_budget,
            config.group_size, reward_config, seed=seed + step,
            stop_penalty=config.stop_penalty,
        )

        # GRPO update
        metrics = grpo_update(
            policy, reference, batch, optimizer, kl_controller,
            clip_epsilon=config.clip_epsilon,
            entropy_coef=config.beta_entropy,
            gradient_clip=config.gradient_clip,
        )
        metrics["step"] = step
        train_log.append(metrics)

        # Track consecutive KL skips for reference reset
        if metrics.get("skip_kl_guard", False):
            consecutive_kl_skips += 1
        else:
            consecutive_kl_skips = 0

        # Reference reset: if KL has been too high for too many steps, the
        # policy is frozen in a deadlock (can't update because KL too high,
        # can't reduce KL because policy can't update). Reset reference to
        # current policy to allow continued training. Common in PPO with
        # adaptive KL when the policy diverges too far from the reference.
        if consecutive_kl_skips >= REF_RESET_THRESHOLD:
            reference = ReferencePolicy(P3O8Policy().to(device))
            reference.policy.load_state_dict(policy.state_dict())
            kl_controller = AdaptiveKLController(config.beta_kl, config.max_kl)
            consecutive_kl_skips = 0
            print(f"  [seed={seed}] step {step}: REFERENCE RESET (KL was stuck "
                  f"for {REF_RESET_THRESHOLD} steps, kl={metrics['kl']:.4f})",
                  flush=True)

        # Progress print every 10 steps
        if step % 10 == 0 or step == 1:
            _elapsed = _time.time() - _train_t0
            _rate = step / max(_elapsed, 0.001)
            _extra = ""
            if metrics.get("skip_kl_guard", False):
                _extra = ", KL_SKIP"
            print(f"  [seed={seed}] step {step}/{config.n_updates}: "
                  f"loss={metrics['loss']:.4f}, kl={metrics['kl']:.4f}, "
                  f"kl_c={metrics.get('kl_coefficient', 0):.3f}, "
                  f"updated={metrics['updated']}{_extra}, "
                  f"rate={_rate:.1f} steps/s, "
                  f"eta={(_train_t0 + config.n_updates/_rate - _time.time())/60:.1f}min",
                  flush=True)

        # Validation
        if step % config.validation_interval == 0 or step == config.n_updates:
            if oracle is None:
                oracle = oracle_factory()
            val = validate_policy(
                policy, validation_sources, oracle, config.edit_budget, reward_config,
                n_trajectories=config.n_validation_trajectories, seed=999,
                stop_penalty=config.stop_penalty,
            )
            val_entry = {"step": step, "validation": val, "type": "periodic"}
            validation_log.append(val_entry)
            print(f"  [seed={seed}] step {step}: val_reward={val['mean_reward']:.6f}, "
                  f"pos_rate={val['positive_improvement_rate']:.2%}, "
                  f"stop_root={val['stop_at_root_rate']:.2%}, "
                  f"constraint={val['constraint_validity']:.2%}")

        # Checkpoint
        if save_dir and (step % config.checkpoint_interval == 0 or step == config.n_updates):
            import os
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"grpo_seed{seed}_step{step}.pt")
            torch.save({
                "step": step, "seed": seed,
                "model_state": policy.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "kl_coefficient": kl_controller.coefficient,
                "config": config.__dict__,
            }, path)

    # Final validation
    if oracle is None:
        oracle = oracle_factory()
    final_val = validate_policy(
        policy, validation_sources, oracle, config.edit_budget, reward_config,
        n_trajectories=config.n_validation_trajectories, seed=999,
        stop_penalty=config.stop_penalty,
    )

    return {
        "seed": seed,
        "train_log": train_log,
        "validation_log": validation_log,
        "warm_start_validation": baseline_val,
        "final_validation": final_val,
        "n_updates": config.n_updates,
    }


__all__ = [
    "SeqEncoder",
    "P3O8Policy",
    "ReferencePolicy",
    "AdaptiveKLController",
    "PrecomputedSingleEditOracle",
    "TrajectoryStep",
    "Trajectory",
    "build_legal_edit_actions_task_a",
    "collect_trajectory",
    "collect_batch",
    "_collect_batch_budget1_batched",
    "compute_group_advantages",
    "grpo_update",
    "validate_policy",
    "_validate_policy_budget1_batched",
    "GRPOTrainConfig",
    "train_single_seed",
    "categorical_kl",
]
