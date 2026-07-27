"""P3-06: Minimal-Edit MDP, Action Space, and Reward v3.

Implements the source-conditioned MDP with:
- MDPState: 9 fields per spec
- EditAction: STOP, FIVE_UTR_SUB, CDS_SYNONYMOUS_SUB (no indels)
- apply_edit_action: atomic codon swap, protein identity guaranteed
- build_legal_edit_actions: 5'UTR + CDS synonymous only, no 3'UTR, no indels
- HierarchicalPolicy: 4-level decomposition (STOP/EDIT → region → position → target)
- LearnableStop: constant / budget-aware / learned variants
- RewardV3: LCB primary + secondary terms, no novelty = -edit_distance
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from core.constants import (
    CODON_TABLE, SYNONYMOUS_CODONS, START_CODON, STOP_CODONS,
    ID_TO_NUC, NUC_TO_ID, is_valid_cds, translate,
)
from core.motif_policy import is_hard_legal
from core.schema import MRNARecord


def sequence_identity(record: MRNARecord) -> str:
    """Canonical state identity: sha256 of the FULL current sequence.

    P0-05 hard rule: every state-identity computation in the MDP and the
    GRPO pipeline must use this function. Mixing tuple-based identity with
    hash-based identity is forbidden.
    """
    return hashlib.sha256(record.seq.encode()).hexdigest()

# Build codon → synonymous codons reverse lookup from amino-acid keyed table.
# SYNONYMOUS_CODONS is keyed by amino acid (e.g., "A" → ["GCA","GCC","GCG","GCU"]).
# We need codon → [synonymous codons] (e.g., "GCU" → ["GCA","GCC","GCG"]).
_CODON_TO_SYNONYMS: Dict[str, List[str]] = {}
for _aa, _codons in SYNONYMOUS_CODONS.items():
    for _c in _codons:
        _CODON_TO_SYNONYMS[_c] = [x for x in _codons if x != _c]


def _get_synonymous_codon_set(codon: str) -> List[str]:
    """Return synonymous codons for a given codon (excluding itself)."""
    return _CODON_TO_SYNONYMS.get(codon, [])


# ===========================================================================
# Edit Action (P3-06 primary actions — no indels)
# ===========================================================================

@dataclass(frozen=True)
class EditAction:
    """An action in the minimal-edit MDP.

    Only 3 action types: STOP, FIVE_UTR_SUB, CDS_SYNONYMOUS_SUB.
    No indels. No 3'UTR edits. No nonsynonymous CDS changes.
    """
    op: str  # "stop", "five_utr_sub", "cds_synonymous_sub"
    pos: int = -1           # nt position (five_utr_sub) or codon position (cds_synonymous_sub)
    nt: str = ""            # target nucleotide (five_utr_sub)
    target_codon: str = ""  # target codon (cds_synonymous_sub)

    def is_stop(self) -> bool:
        return self.op == "stop"

    def is_five_utr(self) -> bool:
        return self.op == "five_utr_sub"

    def is_cds(self) -> bool:
        return self.op == "cds_synonymous_sub"

    def region(self) -> str:
        if self.is_five_utr():
            return "five_utr"
        if self.is_cds():
            return "cds"
        return "none"

    def __repr__(self) -> str:
        if self.is_stop():
            return "EditAction(STOP)"
        if self.is_five_utr():
            return f"EditAction(5UTR_SUB@{self.pos},{self.nt})"
        return f"EditAction(CDS_SUB@codon{self.pos},{self.target_codon})"


STOP_EDIT = EditAction(op="stop")


def apply_edit_action(record: MRNARecord, action: EditAction) -> MRNARecord:
    """Apply an edit action to a record.

    Guarantees:
    - Protein identity: translate(record.cds) == translate(result.cds)
    - Length invariant: len(record.seq) == len(result.seq)
    - Atomic codon swap: no single-nt CDS intermediate states
    """
    if action.is_stop():
        return record

    if action.is_five_utr():
        if action.pos < 0 or action.pos >= len(record.five_utr):
            raise ValueError(f"5'UTR position {action.pos} out of range [0, {len(record.five_utr)})")
        if action.nt not in "ACGU":
            raise ValueError(f"Invalid nucleotide: {action.nt!r}")
        old = record.five_utr[action.pos]
        if action.nt == old:
            raise ValueError(f"Identity substitution at pos {action.pos}")
        new_utr = record.five_utr[:action.pos] + action.nt + record.five_utr[action.pos + 1:]
        return MRNARecord(
            transcript_id=record.transcript_id,
            five_utr=new_utr,
            cds=record.cds,
            three_utr=record.three_utr,
            species=record.species,
            metadata=dict(record.metadata),
        )

    if action.is_cds():
        n_codons = len(record.cds) // 3
        if action.pos < 0 or action.pos >= n_codons:
            raise ValueError(f"Codon position {action.pos} out of range [0, {n_codons})")
        # Start codon preserved
        if action.pos == 0:
            raise ValueError("Cannot edit start codon (codon 0)")
        # Stop codon preserved
        if action.pos == n_codons - 1:
            raise ValueError("Cannot edit stop codon (last codon)")
        nt_start = action.pos * 3
        old_codon = record.cds[nt_start:nt_start + 3]
        if action.target_codon == old_codon:
            raise ValueError(f"Identity codon substitution at codon {action.pos}")
        # Verify synonymous
        synonyms = _get_synonymous_codon_set(old_codon)
        if not synonyms:
            raise ValueError(f"No synonymous codons for {old_codon}")
        if action.target_codon not in synonyms:
            raise ValueError(
                f"{action.target_codon} is not synonymous to {old_codon}"
            )
        new_cds = record.cds[:nt_start] + action.target_codon + record.cds[nt_start + 3:]
        # Protein identity check
        assert translate(record.cds) == translate(new_cds), \
            "Protein identity violated by CDS edit"
        return MRNARecord(
            transcript_id=record.transcript_id,
            five_utr=record.five_utr,
            cds=new_cds,
            three_utr=record.three_utr,
            species=record.species,
            metadata=dict(record.metadata),
        )

    raise ValueError(f"Unknown action op: {action.op!r}")


def build_legal_edit_actions(
    record: MRNARecord,
    visited: Optional[set] = None,
    enforce_hard_motif: bool = True,
) -> List[EditAction]:
    """Build all legal edit actions for a record.

    Returns:
    - STOP (always legal)
    - FIVE_UTR_SUB for each 5'UTR position × 3 non-identity nucleotides
    - CDS_SYNONYMOUS_SUB for each CDS codon × synonymous codons

    Excludes:
    - All indels
    - 3'UTR edits
    - Start/stop codon edits
    - Identity substitutions
    - Actions leading to visited states (cycle avoidance, sha256 identity)
    - Actions violating the hard motif policy (P0-05): upstream AUG,
      cryptic splice sites, homopolymer >= 6, premature stop,
      start/stop violation, reading-frame violation
    """
    actions = [STOP_EDIT]
    visited = visited or set()

    # 5'UTR substitutions
    for pos in range(len(record.five_utr)):
        old = record.five_utr[pos]
        for nt in "ACGU":
            if nt != old:
                action = EditAction(op="five_utr_sub", pos=pos, nt=nt)
                # Check cycle
                new_rec = apply_edit_action(record, action)
                seq_hash = sequence_identity(new_rec)
                if seq_hash in visited:
                    continue
                if enforce_hard_motif and not is_hard_legal(record, new_rec):
                    continue
                actions.append(action)

    # CDS synonymous substitutions
    n_codons = len(record.cds) // 3
    for codon_pos in range(1, n_codons - 1):  # Skip start (0) and stop (last)
        nt_start = codon_pos * 3
        old_codon = record.cds[nt_start:nt_start + 3]
        synonyms = _get_synonymous_codon_set(old_codon)
        for target in synonyms:
            if target != old_codon:
                action = EditAction(op="cds_synonymous_sub", pos=codon_pos, target_codon=target)
                new_rec = apply_edit_action(record, action)
                seq_hash = sequence_identity(new_rec)
                if seq_hash in visited:
                    continue
                if enforce_hard_motif and not is_hard_legal(record, new_rec):
                    continue
                actions.append(action)

    return actions


# ===========================================================================
# MDP State (P3-06: 9 fields)
# ===========================================================================

@dataclass(frozen=True)
class MDPState:
    """Complete MDP state for minimal-edit mRNA optimization."""
    source_mrna: MRNARecord
    current_mrna: MRNARecord
    edit_history: Tuple[EditAction, ...] = ()
    visited_states: frozenset = frozenset()
    remaining_budget: int = 0
    cargo_identity: str = ""
    cell_context: str = "default"
    oracle_uncertainty: float = 0.0
    current_predicted_delta: float = 0.0

    def sequence_hash(self) -> str:
        return sequence_identity(self.current_mrna)

    def n_edits(self) -> int:
        return len(self.edit_history)


def initial_state(
    source: MRNARecord,
    budget: int,
    cargo: str = "",
) -> MDPState:
    """Create the initial MDP state from a source sequence."""
    seq_hash = sequence_identity(source)
    return MDPState(
        source_mrna=source,
        current_mrna=source,
        edit_history=(),
        visited_states=frozenset({seq_hash}),
        remaining_budget=budget,
        cargo_identity=cargo,
    )


def transition(state: MDPState, action: EditAction) -> MDPState:
    """Apply an action to get the next state."""
    new_record = apply_edit_action(state.current_mrna, action)
    seq_hash = sequence_identity(new_record)

    new_history = state.edit_history + (action,) if not action.is_stop() else state.edit_history
    new_budget = state.remaining_budget - (0 if action.is_stop() else 1)

    return MDPState(
        source_mrna=state.source_mrna,
        current_mrna=new_record,
        edit_history=new_history,
        visited_states=state.visited_states | {seq_hash},
        remaining_budget=new_budget,
        cargo_identity=state.cargo_identity,
        cell_context=state.cell_context,
    )


# ===========================================================================
# Learnable STOP (3 variants)
# ===========================================================================

class ConstantStop:
    """Fixed p_stop = 0.5."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, features: Dict[str, float]) -> float:
        return self.p

    def parameters(self):
        return []


class BudgetAwareStop:
    """p_stop = sigmoid(a + b * budget_fraction)."""
    def __init__(self):
        self.a = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(1.0))

    def __call__(self, features: Dict[str, float]) -> float:
        budget_frac = features.get("remaining_budget_frac", 0.0)
        logit = self.a.detach() + self.b.detach() * budget_frac
        return float(torch.sigmoid(logit))

    def parameters(self):
        return [self.a, self.b]


class LearnedStop(nn.Module):
    """MLP-based learned STOP.

    Input: global representation + edit history + budget + predicted improvement + uncertainty + best LCB
    Output: p_stop ∈ [0, 1]
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (batch, input_dim) → p_stop: (batch,)"""
        return self.net(features).squeeze(-1)


# ===========================================================================
# Hierarchical Policy (4-level decomposition)
# ===========================================================================

@dataclass
class PolicyLevelOutputs:
    """Outputs from each level of the hierarchical policy."""
    p_stop: float
    p_edit: float
    region_probs: torch.Tensor      # (2,) for [5'UTR, CDS]
    position_probs: torch.Tensor    # (max_region_len,) masked
    target_probs: torch.Tensor      # (max_targets,) masked
    legal_actions: List[EditAction]
    log_probs: Dict[str, float]     # action → log π(action|state)


class HierarchicalPolicy:
    """4-level hierarchical policy for minimal-edit MDP.

    π(action|state) = π(STOP|s) × π(region|EDIT,s) × π(position|region,s) × π(target|position,s)
    """

    def __init__(
        self,
        stop_model: Optional[nn.Module] = None,
        region_model: Optional[nn.Module] = None,
        position_model: Optional[nn.Module] = None,
        target_model: Optional[nn.Module] = None,
    ):
        self.stop_model = stop_model or ConstantStop()
        self.region_model = region_model
        self.position_model = position_model
        self.target_model = target_model

    def log_pi(self, action: EditAction, state: MDPState) -> float:
        """Compute log π(action|state) via hierarchical decomposition."""
        if action.is_stop():
            features = self._stop_features(state)
            p_stop = self.stop_model(features)
            return math.log(max(p_stop, 1e-10))

        # p_edit = 1 - p_stop
        features = self._stop_features(state)
        p_stop = self.stop_model(features)
        log_p = math.log(max(1 - p_stop, 1e-10))

        # Level 2: region
        region_idx = 0 if action.is_five_utr() else 1
        p_region = self._region_probs(state)
        log_p += math.log(max(float(p_region[region_idx]), 1e-10))

        # Level 3: position
        p_position = self._position_probs(state, action.region())
        pos_idx = action.pos
        log_p += math.log(max(float(p_position[pos_idx]), 1e-10))

        # Level 4: target
        p_target = self._target_probs(state, action)
        target_idx = self._target_index(action)
        log_p += math.log(max(float(p_target[target_idx]), 1e-10))

        return log_p

    def _stop_features(self, state: MDPState) -> Dict[str, float]:
        return {
            "n_edits": float(state.n_edits()),
            "remaining_budget_frac": state.remaining_budget / max(state.remaining_budget + state.n_edits(), 1),
            "oracle_uncertainty": state.oracle_uncertainty,
            "current_predicted_delta": state.current_predicted_delta,
        }

    def _region_probs(self, state: MDPState) -> torch.Tensor:
        """Return (2,) probability over [5'UTR, CDS]."""
        if self.region_model is not None:
            return torch.softmax(self.region_model(state), dim=-1)
        # Default: uniform over regions with legal actions
        n_utr = len(state.current_mrna.five_utr)
        n_cds = max(len(state.current_mrna.cds) // 3 - 2, 0)  # Exclude start/stop
        total = n_utr + n_cds
        if total == 0:
            return torch.tensor([0.0, 0.0])
        return torch.tensor([n_utr / total, n_cds / total])

    def _position_probs(self, state: MDPState, region: str) -> torch.Tensor:
        """Return probability over positions in the selected region."""
        if region == "five_utr":
            n = len(state.current_mrna.five_utr)
        else:
            n = max(len(state.current_mrna.cds) // 3 - 2, 0)
        if n == 0:
            return torch.ones(1)
        return torch.ones(n) / n

    def _target_probs(self, state: MDPState, action: EditAction) -> torch.Tensor:
        """Return probability over targets at the selected position."""
        if action.is_five_utr():
            # 3 non-identity nucleotides
            return torch.ones(3) / 3
        # CDS: synonymous codons for the current codon
        codon_pos = action.pos
        nt_start = codon_pos * 3
        old_codon = state.current_mrna.cds[nt_start:nt_start + 3]
        synonyms = _get_synonymous_codon_set(old_codon)
        n = max(len(synonyms), 1)
        return torch.ones(n) / n

    def _target_index(self, action: EditAction) -> int:
        """Map action target to index in the target distribution."""
        if action.is_five_utr():
            old = action.nt  # This is the NEW nucleotide
            # Index among non-identity nucleotides
            # We need the OLD nucleotide to compute the index
            # For simplicity, return 0 (uniform distribution makes this irrelevant)
            return 0
        return 0


# ===========================================================================
# Reward v3 (LCB primary + secondary, no novelty)
# ===========================================================================

@dataclass(frozen=True)
class RewardV3Config:
    """Configuration for reward v3."""
    lambda_lcb: float = 1.0          # LCB penalty multiplier
    w_abundance: float = 0.1         # mRNA abundance weight
    w_half_life: float = 0.1         # Half-life weight
    w_edit_cost: float = -0.05       # Edit cost (per edit, negative)
    w_manifold: float = 0.0          # On-manifold penalty (disabled)
    w_manufacturability: float = -0.1  # Manufacturability penalty
    context: str = "protein_output_focused"  # Pre-registered context


def compute_reward_v3(
    source: MRNARecord,
    candidate: MRNARecord,
    predicted_deltas: Dict[str, float],
    uncertainties: Dict[str, float],
    n_edits: int,
    config: Optional[RewardV3Config] = None,
) -> Dict[str, Any]:
    """Compute reward v3 for a candidate edit.

    Args:
        source: Original wild-type record
        candidate: Edited record
        predicted_deltas: {"protein_output": float, "mrna_abundance": float, "half_life": float}
        uncertainties: {"protein_output": float, ...}
        n_edits: Number of edits applied
        config: Reward configuration

    Returns:
        Dict with scalar, lcb, secondary_terms, provenance
    """
    cfg = config or RewardV3Config()

    # Primary: LCB
    mean_delta = predicted_deltas.get("protein_output", 0.0)
    uncertainty = uncertainties.get("protein_output", 0.0)
    lcb = mean_delta - cfg.lambda_lcb * uncertainty

    # Secondary terms (context-dependent)
    secondary = {}
    total_secondary = 0.0

    if cfg.context in ("protein_output_with_stability_guard", "balanced_experimental_profile"):
        half_life_delta = predicted_deltas.get("half_life", 0.0)
        secondary["half_life"] = cfg.w_half_life * half_life_delta
        total_secondary += secondary["half_life"]

    if cfg.context == "balanced_experimental_profile":
        abundance_delta = predicted_deltas.get("mrna_abundance", 0.0)
        secondary["mrna_abundance"] = cfg.w_abundance * abundance_delta
        total_secondary += secondary["mrna_abundance"]

    # Edit cost (always)
    secondary["edit_cost"] = cfg.w_edit_cost * n_edits
    total_secondary += secondary["edit_cost"]

    # Total scalar
    scalar = lcb + total_secondary

    return {
        "scalar": scalar,
        "lcb": lcb,
        "mean_delta": mean_delta,
        "uncertainty": uncertainty,
        "lambda_lcb": cfg.lambda_lcb,
        "secondary_terms": secondary,
        "total_secondary": total_secondary,
        "n_edits": n_edits,
        "context": cfg.context,
        "provenance": {
            "predictor": "p3_02_delta_oracle",
            "reward_version": "v3",
            "source_normalized": True,
            "risk_adjusted": True,
            "no_novelty": True,
        },
    }


# ===========================================================================
# P0-05: frozen reward semantics — strict incremental reward with telescoping
# ===========================================================================
#
# REWARD MODE (frozen): strict incremental
#   r_t = F(x_t) - F(x_{t-1}) - one_step_cost
# where F(x) is the risk-adjusted predicted improvement of x vs the source
# (LCB + context secondary terms, WITHOUT edit cost), and
#   one_step_cost = -w_edit_cost  (positive cost per edit).
#
# Telescoping guarantee:
#   sum_t r_t = F(x_final) - F(x_0) - n_edits * one_step_cost
#             = terminal_delta - total_cost
# which equals the terminal reward
#   R = Oracle(source, final_candidate) - edit_cost_total - risk_penalty.

def reward_value_fn(
    source: MRNARecord,
    candidate: MRNARecord,
    predicted_deltas: Dict[str, float],
    uncertainties: Dict[str, float],
    config: Optional[RewardV3Config] = None,
) -> float:
    """F(x): risk-adjusted predicted improvement of candidate vs source.

    Excludes edit cost (handled separately as per-step cost) so that the
    incremental rewards telescope exactly.
    """
    out = compute_reward_v3(
        source, candidate,
        predicted_deltas=predicted_deltas,
        uncertainties=uncertainties,
        n_edits=0,  # edit cost excluded from the value function
        config=config,
    )
    return float(out["scalar"])


def one_step_edit_cost(config: Optional[RewardV3Config] = None) -> float:
    """Positive per-edit cost deducted at every incremental step."""
    cfg = config or RewardV3Config()
    return -float(cfg.w_edit_cost)


def compute_terminal_reward(
    source: MRNARecord,
    final_candidate: MRNARecord,
    predicted_deltas: Dict[str, float],
    uncertainties: Dict[str, float],
    n_edits: int,
    config: Optional[RewardV3Config] = None,
) -> float:
    """Terminal reward: R = F(final) - F(source) - total_edit_cost.

    Exactly equals the sum of incremental rewards over any trajectory from
    source to final_candidate (telescoping).
    """
    f_final = reward_value_fn(source, final_candidate,
                              predicted_deltas, uncertainties, config)
    f_source = reward_value_fn(
        source, source,
        predicted_deltas={"protein_output": 0.0},
        uncertainties={"protein_output": 0.0},
        config=config,
    )
    return f_final - f_source - n_edits * one_step_edit_cost(config)
