"""P2-05: Real mRNA design MDP backed by a predictive oracle.

Implements the same MDP interface as :class:`mrna_editflow.rl.tiny_mdp.TinyMDP`
but operates on real :class:`MRNARecord` sequences with a predictive oracle
(Oracle #3, GBT regressor) providing the reward signal.

Reward design (sparse terminal)
-------------------------------
Following TinyMDP's convention, the per-step reward is 0 for non-terminal
steps, and the terminal step receives the **total delta predicted value**
between the final edited sequence and the initial wild-type sequence:

    r_t = 0                                                  (t < T-1)
    r_{T-1} = oracle.predict(s_final.seq) - oracle.predict(s_initial.seq)

This yields ``G_0 = gamma^(T-1) * r_{T-1}`` (discounted terminal gain),
which is exactly the signal GRPO's group-normalized terminal advantage
operates on. The design matches the P2-05 spec ("group-normalized
terminal advantage") and gives the cleanest mapping from "gain/edit"
metrics to the RL objective.

Reward qualifier
----------------
The production reward signal is the **delta predicted TE** from Oracle #3
(P1-05 GBT regressor), which is a *predicted / internal proxy* for
translation efficiency. Any claim that GRPO "improves TE" MUST be
qualified as "improves predicted TE (internal proxy)" until P2-01
multi-region oracle validation completes.

Region restriction
------------------
The MDP itself does NOT enforce region-restricted editing (5'UTR / CDS /
3'UTR). Region restriction is the responsibility of the Policy via
``build_legal_action_mask`` (see ``rl/synergy.py:build_region_restricted_mask``
for the counterfactual single-region pattern). The ``region`` field on
this MDP is informational only and propagates into metadata for audit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

import numpy as np

from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.rl.action_space import Action


# ---------------------------------------------------------------------------
# Oracle protocol (structural typing)
# ---------------------------------------------------------------------------


class OracleLike(Protocol):
    """Minimal protocol for a predictive oracle.

    Any object with a ``predict(sequences: Sequence[str]) -> np.ndarray``
    method satisfies this protocol (e.g. :class:`GBTOracle`).
    """

    def predict(self, sequences: Sequence[str]) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# RealMRNAMDP
# ---------------------------------------------------------------------------


@dataclass
class RealMRNAMDP:
    """Real mRNA design MDP backed by a predictive oracle.

    The agent starts from ``initial_record`` and edits it to maximize
    ``oracle.predict(seq)``. The terminal reward is the delta predicted
    value between the final edited sequence and the initial wild-type
    sequence.

    Parameters
    ----------
    initial_record : MRNARecord
        Starting state (typically a wild-type mRNA from the split contract).
    oracle : OracleLike
        The reward oracle (e.g. :class:`GBTOracle`). Must expose
        ``predict(sequences: Sequence[str]) -> np.ndarray``.
    max_steps : int
        Maximum trajectory length (edit budget). Default 3 (matches the
        ``--edit-budget 3`` used in P2-03 baselines).
    gamma : float
        Discount factor. Default 0.99.
    region : str
        Informational: which region edits are restricted to
        (``"full"``, ``"five_utr"``, ``"cds"``, ``"three_utr"``).
        Default ``"full"``. The actual restriction is enforced by the
        Policy's action mask, not by this MDP.
    reward_field : str
        Field name for metadata / audit. Default
        ``"predicted_te_internal_proxy"`` to comply with the project
        constraint that any "improves TE" claim MUST be qualified as
        "predicted / internal proxy" until P2-01 completes.
    """

    initial_record: MRNARecord
    oracle: OracleLike
    max_steps: int = 3
    gamma: float = 0.99
    region: str = "full"
    reward_field: str = "predicted_te_internal_proxy"

    # Cached initial prediction to avoid recompute across trajectory groups
    # (GRPO collects N trajectories from the same starting state).
    _initial_pred: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if self.region not in ("full", "five_utr", "cds", "three_utr"):
            raise ValueError(
                f"region must be one of full/five_utr/cds/three_utr, got {self.region!r}"
            )
        if not self.reward_field:
            raise ValueError("reward_field must be non-empty")

    # ------------------------------------------------------------------
    # MDP interface (mirrors TinyMDP)
    # ------------------------------------------------------------------

    def initial_state(self) -> MRNARecord:
        """Return the initial state (the wild-type record)."""
        return self.initial_record

    def reward(
        self,
        state: MRNARecord,
        action: Action,
        next_state: MRNARecord,
        step: int,
    ) -> float:
        """Compute the per-step reward.

        - Non-terminal steps: 0 (sparse).
        - Terminal step (STOP or ``step + 1 >= max_steps``):
          ``oracle.predict(next_state.seq) - oracle.predict(initial.seq)``.

        STOP actions yield ``next_state == state`` by definition of
        :func:`apply_action`, so a STOP at any step produces a terminal
        reward of ``predict(state) - predict(initial)`` — the gain
        accumulated up to that point.
        """
        is_terminal = action.is_stop() or (step + 1) >= self.max_steps
        if not is_terminal:
            return 0.0
        if self._initial_pred is None:
            self._initial_pred = self._predict(self.initial_record.seq)
        final_pred = self._predict(next_state.seq)
        return float(final_pred - self._initial_pred)

    def is_terminal(self, action: Action, step: int) -> bool:
        """Terminal on STOP or when ``step + 1 >= max_steps``."""
        return action.is_stop() or (step + 1) >= self.max_steps

    # ------------------------------------------------------------------
    # Oracle helper
    # ------------------------------------------------------------------

    def _predict(self, seq: str) -> float:
        """Predict a single sequence via the oracle (returns Python float)."""
        return float(self.oracle.predict([seq])[0])

    def reset_cache(self) -> None:
        """Clear the cached initial prediction.

        Call this if the MDP is reused with a different ``initial_record``
        (which shouldn't happen in normal use, but is useful for tests).
        """
        self._initial_pred = None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def to_metadata(self) -> dict:
        """Return a JSON-serializable metadata dict for audit / reproducibility."""
        return {
            "mdp_type": "RealMRNAMDP",
            "max_steps": int(self.max_steps),
            "gamma": float(self.gamma),
            "region": str(self.region),
            "reward_field": str(self.reward_field),
            "reward_design": "sparse_terminal_delta_predicted_value",
            "initial_transcript_id": str(self.initial_record.transcript_id),
            "initial_seq_length": len(self.initial_record.seq),
        }
