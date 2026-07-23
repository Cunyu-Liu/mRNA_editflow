"""P3-04 Task 4: Training reward with risk adjustment and hard-constraint gating.

Implements ``build_training_reward`` which converts raw reward components into
a training signal that:

* Is source-normalized (delta vs source baseline, not absolute).
* Risk-adjusted (uncertainty penalty enters the advantage).
* Hard-failure gated (protein identity / GC / length violations zero-out
  reward and force STOP preference).
* STOP-preferred when all editing directions are negative.

This module is imported by the production GRPO path to replace the audit-only
``risk_adjusted_reward`` that was previously computed but never entered the
gradient.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class HardConstraintStatus:
    """Results of hard-constraint validation."""
    protein_identity: bool = True
    gc_in_range: bool = True
    length_in_range: bool = True
    cds_valid: bool = True

    @property
    def all_pass(self) -> bool:
        return self.protein_identity and self.gc_in_range and self.length_in_range and self.cds_valid

    def to_dict(self) -> Dict[str, bool]:
        return {
            "protein_identity": self.protein_identity,
            "gc_in_range": self.gc_in_range,
            "length_in_range": self.length_in_range,
            "cds_valid": self.cds_valid,
        }


@dataclass(frozen=True)
class TrainingReward:
    """The reward signal that actually enters the policy gradient."""
    scalar: float                          # The final scalar reward used for advantage
    source_normalized_delta: float         # reward - source_baseline (always relative)
    risk_adjusted_delta: float             # After uncertainty penalty
    hard_constraint_gated: bool            # True if hard failure zeroed the reward
    stop_preferred: bool                   # True if all-negative → STOP preferred
    raw_components: Dict[str, float]       # Original component deltas
    reward_provenance: Dict[str, Any]      # Full audit trail

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scalar": self.scalar,
            "source_normalized_delta": self.source_normalized_delta,
            "risk_adjusted_delta": self.risk_adjusted_delta,
            "hard_constraint_gated": self.hard_constraint_gated,
            "stop_preferred": self.stop_preferred,
            "raw_components": self.raw_components,
            "reward_provenance": self.reward_provenance,
        }


def build_training_reward(
    raw_components: Mapping[str, float],
    uncertainty: float,
    hard_constraint_status: HardConstraintStatus,
    source_baseline: float = 0.0,
    context: Optional[Mapping[str, Any]] = None,
    *,
    stop_reward: float = 0.0,
    uncertainty_penalty_multiplier: float = 1.0,
    soft_objective_keys: Sequence[str] = ("te", "access", "agreement"),
) -> TrainingReward:
    """Convert raw reward components into a training-grade reward signal.

    Parameters
    ----------
    raw_components : mapping
        Per-component deltas (e.g. ``{"te": 0.1, "access": -0.05, ...}``).
        These must already be source-normalized deltas (candidate − source).
    uncertainty : float
        Prediction uncertainty of the oracle for this trajectory.
    hard_constraint_status : HardConstraintStatus
        Whether protein identity, GC range, and length constraints passed.
    source_baseline : float
        The source sequence's baseline reward (should be 0.0 if components
        are already deltas).  Kept for explicitness.
    context : mapping, optional
        Additional context (e.g. cargo_id, task_id, preference weights).
    stop_reward : float
        Reward assigned to STOP action (default 0.0 — neutral).
    uncertainty_penalty_multiplier : float
        How strongly uncertainty penalises the reward (default 1.0).
    soft_objective_keys : sequence
        Keys in ``raw_components`` that are "soft" objectives (used for
        all-negative detection).

    Returns
    -------
    TrainingReward
        The scalar reward that enters the advantage, plus full provenance.

    Guarantees
    ----------
    1. **Source-normalized:** The scalar is always relative to source_baseline.
    2. **Hard-failure gating:** If any hard constraint fails, scalar is set to
       a large negative value (−1.0) and STOP is preferred.
    3. **All-negative → STOP preferred:** If all soft objectives are ≤ 0,
       scalar is clamped to ≤ stop_reward, making STOP the better action.
    4. **Risk-adjusted:** Uncertainty subtracts from the reward (penalises
       uncertain improvements).
    5. **Auditable:** Full provenance recorded.
    """
    context = dict(context or {})
    raw_dict = dict(raw_components)

    # --- Step 1: Compute soft-objective sum ---
    soft_sum = sum(raw_dict.get(k, 0.0) for k in soft_objective_keys)
    all_soft_negative = all(raw_dict.get(k, 0.0) <= 0.0 for k in soft_objective_keys)

    # --- Step 2: Compute cost terms (edit_cost, novelty, repeated_motif) ---
    cost_keys = [k for k in raw_dict if k not in soft_objective_keys]
    cost_sum = sum(raw_dict.get(k, 0.0) for k in cost_keys)

    # --- Step 3: Risk-adjusted delta ---
    risk_adjusted = soft_sum + cost_sum - uncertainty_penalty_multiplier * uncertainty

    # --- Step 4: Hard-constraint gating ---
    hard_gated = not hard_constraint_status.all_pass

    # --- Step 5: Determine final scalar ---
    if hard_gated:
        # Hard failure: reward is strongly negative, STOP preferred
        scalar = -1.0
        stop_preferred = True
        gating_reason = "hard_constraint_violation"
    elif all_soft_negative:
        # All soft objectives negative: STOP is at least as good as editing
        scalar = min(risk_adjusted, stop_reward)
        stop_preferred = True
        gating_reason = "all_negative_stop_preferred"
    else:
        # Normal case: risk-adjusted reward
        scalar = risk_adjusted
        stop_preferred = False
        gating_reason = "normal"

    # --- Step 6: Source normalization check ---
    # The scalar should already be a delta; we just record the baseline.
    source_normalized_delta = scalar  # Already relative to source

    # --- Step 7: Build provenance ---
    provenance: Dict[str, Any] = {
        "source_baseline": source_baseline,
        "soft_sum": soft_sum,
        "cost_sum": cost_sum,
        "uncertainty": uncertainty,
        "uncertainty_penalty": uncertainty_penalty_multiplier * uncertainty,
        "risk_adjusted_before_gating": risk_adjusted,
        "hard_constraint_status": hard_constraint_status.to_dict(),
        "hard_gated": hard_gated,
        "all_soft_negative": all_soft_negative,
        "stop_preferred": stop_preferred,
        "gating_reason": gating_reason,
        "stop_reward": stop_reward,
        "context": context,
        "soft_objective_keys": list(soft_objective_keys),
        "cost_keys": cost_keys,
    }

    return TrainingReward(
        scalar=scalar,
        source_normalized_delta=source_normalized_delta,
        risk_adjusted_delta=risk_adjusted,
        hard_constraint_gated=hard_gated,
        stop_preferred=stop_preferred,
        raw_components=raw_dict,
        reward_provenance=provenance,
    )


def stop_reward_vector(component_names: Sequence[str]) -> Dict[str, float]:
    """Return the reward vector for a STOP action (all zeros = source baseline)."""
    return {name: 0.0 for name in component_names}


__all__ = [
    "HardConstraintStatus",
    "TrainingReward",
    "build_training_reward",
    "stop_reward_vector",
]
