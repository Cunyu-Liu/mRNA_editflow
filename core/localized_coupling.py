"""Localized Edit Flow coupling: sequence- and structure-aware rate boosting.

This module implements the "localized Edit Flow" extension described in the
Edit Flow paper, where an edit at one position increases the probability of
nearby edits. Two couplers are provided:

* :class:`SequenceLocalizedCoupling` -- after an edit at position ``i``, the
  edit rates at positions ``i +/- w`` are multiplicatively boosted by
  ``(1 + alpha * decay(distance))``.  The decay kernel is configurable
  (exponential ``exp(-d / sigma)`` by default, or linear).

* :class:`StructureLocalizedCoupling` -- extends sequence-localized coupling to
  RNA secondary structure.  After an edit at position ``i``, rates are boosted
  at:

  1. **Sequence neighbours** (``+/- w`` nt) -- same as the sequence coupler.
  2. **Base-pair partners** -- positions ``j`` with high pairing probability
     with ``i`` (requires a ``pairing_matrix``).  This enables compensatory
     double substitutions: e.g. a G->A edit in a stem boosts the rate at the
     paired C position so a compensatory C->U edit can restore the Watson-Crick
     pair.
  3. **Stem-loop members** -- positions near a base-pair partner ``j`` (within
     ``stem_window``), since stems are contiguous stacks of base pairs.

  When only a 1-D ``pairing_prob`` (per-nt pairing probability, as stored in
  :class:`~mrna_editflow.schema.PrecomputedFeatures`) is available, the coupler
  falls back to a coarser heuristic: nearby positions with high pairing
  probability are boosted, approximating "same structural element" membership.

Both couplers are pure tensor/numpy operations: no GPU, no trained model, no
network.  This makes them fully unit-testable with synthetic rate tensors and
edit trajectories.

Complexity: ``O(E * w)`` for sequence coupling and
``O(E * (w + P * s))`` for structure coupling, where ``E`` = number of edits,
``w`` = sequence window, ``P`` = number of base-pair partners per edit, and
``s`` = stem window.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor


# ===========================================================================
# Helpers
# ===========================================================================
def _to_float_tensor(
    x: Union[Tensor, np.ndarray, Sequence[float], None],
    *,
    length: Optional[int] = None,
) -> Optional[Tensor]:
    """Convert *x* to a 1-D ``float32`` tensor, validating length if given.

    Accepts ``None`` (passes through), tensors, numpy arrays, or Python
    sequences.  Complexity: O(len(x)).
    """
    if x is None:
        return None
    if isinstance(x, Tensor):
        t = x.detach().cpu().float()
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x.astype(np.float32))
    else:
        t = torch.tensor(list(x), dtype=torch.float32)
    t = t.squeeze()
    if t.dim() != 1:
        raise ValueError(f"expected a 1-D array, got shape {tuple(t.shape)}")
    if length is not None and t.shape[0] != length:
        raise ValueError(
            f"length mismatch: expected {length}, got {t.shape[0]}"
        )
    return t


def _to_float_matrix(
    x: Union[Tensor, np.ndarray, None],
) -> Optional[Tensor]:
    """Convert *x* to a 2-D ``float32`` tensor (CPU). Complexity: O(L^2)."""
    if x is None:
        return None
    if isinstance(x, Tensor):
        t = x.detach().cpu().float()
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x.astype(np.float32))
    else:
        raise TypeError(f"unsupported type for pairing_matrix: {type(x)}")
    if t.dim() != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {tuple(t.shape)}")
    if t.shape[0] != t.shape[1]:
        raise ValueError(f"pairing matrix must be square, got {tuple(t.shape)}")
    return t


# ===========================================================================
# Sequence-localized coupling
# ===========================================================================
@dataclass
class SequenceLocalizedCouplingConfig:
    """Configuration for :class:`SequenceLocalizedCoupling`.

    Attributes
    ----------
    window : int
        Half-window ``w``.  Positions in ``[i-w, i+w]`` (excluding ``i``) are
        boosted after an edit at ``i``.
    alpha : float
        Boost strength.  The multiplicative factor applied to a neighbour at
        distance ``d`` is ``(1 + alpha * decay(d))``.
    sigma : float
        Decay length-scale (used by the exponential kernel).
    decay_kind : str
        ``"exponential"`` (``exp(-d / sigma)``) or ``"linear"``
        (``max(0, 1 - d / (window + 1))``).
    """

    window: int = 5
    alpha: float = 1.0
    sigma: float = 2.0
    decay_kind: str = "exponential"


class SequenceLocalizedCoupling:
    """Boost edit rates at sequence-neighbour positions after each edit.

    Given a trajectory of edit positions and a per-position (or per-position,
    per-operation) rate tensor, this coupler computes a multiplicative boost
    factor for every position and returns the boosted rates::

        boost(j) = prod_{i in edits} (1 + alpha * decay(|j - i|))
        rates_out[j] = rates[j] * boost(j)

    The edit position itself (``d == 0``) is excluded from the boost.

    Usage::

        coupler = SequenceLocalizedCoupling(config)
        boosted = coupler.apply(rates, edit_positions=[10, 15])
        # or equivalently:
        boosted = coupler(rates, [10, 15])

    Complexity: ``O(E * w)`` for ``E`` edits and window ``w``.
    """

    def __init__(
        self,
        config: SequenceLocalizedCouplingConfig = SequenceLocalizedCouplingConfig(),
    ) -> None:
        self.config = config

    # ------------------------------------------------------------------
    def decay(self, distance: int) -> float:
        """Decay factor for *distance* (>= 0).

        - ``"exponential"``: ``exp(-d / sigma)``.
        - ``"linear"``: ``max(0, 1 - d / (window + 1))``.

        Complexity: O(1).
        """
        d = abs(int(distance))
        cfg = self.config
        if cfg.decay_kind == "exponential":
            s = max(float(cfg.sigma), 1e-12)
            return float(math.exp(-d / s))
        if cfg.decay_kind == "linear":
            denom = max(int(cfg.window) + 1, 1)
            return max(0.0, 1.0 - d / denom)
        raise ValueError(
            f"unknown decay_kind {cfg.decay_kind!r}; "
            "expected 'exponential' or 'linear'"
        )

    # ------------------------------------------------------------------
    def apply(
        self,
        rates: Tensor,
        edit_positions: Sequence[int],
    ) -> Tensor:
        """Return a boosted copy of *rates* given a trajectory of edits.

        Parameters
        ----------
        rates : Tensor
            Per-position rates, shape ``[L]`` or ``[L, num_ops]``.
        edit_positions : sequence of int
            Positions where edits occurred (the trajectory).

        Returns
        -------
        Tensor
            Same shape as *rates*, with neighbour positions multiplicatively
            boosted.

        Complexity: ``O(E * w)`` for ``E`` edits and window ``w``.
        """
        if rates.dim() not in (1, 2):
            raise ValueError(
                f"rates must be 1-D [L] or 2-D [L, num_ops], got {rates.dim()}-D"
            )
        L = rates.shape[0]
        cfg = self.config
        w = int(cfg.window)

        # Compute the cumulative boost factor per position.
        boost = torch.ones(L, dtype=torch.float32)
        for i in edit_positions:
            i = int(i)
            lo = max(0, i - w)
            hi = min(L, i + w + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                d = abs(j - i)
                boost[j] *= (1.0 + cfg.alpha * self.decay(d))

        out = rates.detach().clone().float()
        if out.dim() == 1:
            out = out * boost
        else:
            out = out * boost.unsqueeze(-1)
        return out

    # Convenient alias so the coupler is callable.
    def __call__(
        self,
        rates: Tensor,
        edit_positions: Sequence[int],
    ) -> Tensor:
        return self.apply(rates, edit_positions)


# ===========================================================================
# Structure-localized coupling
# ===========================================================================
@dataclass
class StructureLocalizedCouplingConfig:
    """Configuration for :class:`StructureLocalizedCoupling`.

    Attributes
    ----------
    window : int
        Half-window for sequence-neighbour boosts.
    alpha_seq : float
        Boost strength for sequence neighbours.
    alpha_pair : float
        Boost strength for base-pair partners.  Applied as
        ``(1 + alpha_pair * pairing_matrix[i, j])``.
    alpha_stem : float
        Boost strength for stem-loop members (positions near a base-pair
        partner).
    sigma : float
        Decay length-scale for the sequence-neighbour and stem-member kernels.
    decay_kind : str
        ``"exponential"`` or ``"linear"`` (same as
        :class:`SequenceLocalizedCouplingConfig`).
    pair_threshold : float
        Minimum pairing probability for a position to be considered a base-pair
        partner (used with ``pairing_matrix``) or a paired position (used with
        ``pairing_prob``).
    stem_window : int
        Half-window around each base-pair partner for stem-member boosts.
    structure_window : int
        Extended window used when only ``pairing_prob`` (1-D) is available.
        Nearby positions with high pairing probability within this window are
        boosted as approximate structural neighbours.
    """

    window: int = 5
    alpha_seq: float = 1.0
    alpha_pair: float = 2.0
    alpha_stem: float = 1.5
    sigma: float = 2.0
    decay_kind: str = "exponential"
    pair_threshold: float = 0.3
    stem_window: int = 3
    structure_window: int = 15


class StructureLocalizedCoupling:
    """Boost edit rates using RNA secondary-structure information.

    After an edit at position ``i``, rates are boosted at three classes of
    positions:

    1. **Sequence neighbours** (``+/- window``) -- identical to
       :class:`SequenceLocalizedCoupling`.
    2. **Base-pair partners** -- positions ``j`` where
       ``pairing_matrix[i, j] > pair_threshold``.  Boost factor:
       ``(1 + alpha_pair * pairing_matrix[i, j])``.
    3. **Stem-loop members** -- positions within ``stem_window`` of each
       base-pair partner ``j``.  Boost factor:
       ``(1 + alpha_stem * decay(|k - j|))``.

    When only a 1-D ``pairing_prob`` is available (no pairing matrix), the
    coupler falls back to a coarser heuristic: positions ``j`` within
    ``structure_window`` of the edit that have ``pairing_prob[j] >
    pair_threshold`` are boosted by ``(1 + alpha_pair * pairing_prob[j] *
    decay(|j - i|))``.  This approximates "same structural element" membership
    without requiring the full L-by-L base-pair probability matrix.

    The three boost sources are combined multiplicatively::

        total_boost(j) = seq_boost(j) * pair_boost(j) * stem_boost(j)

    This enables compensatory double substitutions: a G->A edit in a stem
    boosts the rate at its paired C position, allowing a subsequent C->U edit
    to restore the Watson-Crick pair.

    Usage with a pairing matrix::

        coupler = StructureLocalizedCoupling(config)
        boosted = coupler(rates, [10], pairing_matrix=bp_matrix)

    Usage with per-nt pairing probabilities (from
    :class:`~mrna_editflow.schema.PrecomputedFeatures`)::

        boosted = coupler(rates, [10], pairing_prob=features.pairing_prob)

    Complexity: ``O(E * (w + P * s))`` for ``E`` edits, ``w`` = sequence
    window, ``P`` = number of base-pair partners per edit, ``s`` = stem window.
    With only ``pairing_prob`` the cost is ``O(E * structure_window)``.
    """

    def __init__(
        self,
        config: StructureLocalizedCouplingConfig = StructureLocalizedCouplingConfig(),
    ) -> None:
        self.config = config
        # Reuse the sequence coupler for neighbour boosts.
        self._seq_coupler = SequenceLocalizedCoupling(
            SequenceLocalizedCouplingConfig(
                window=config.window,
                alpha=config.alpha_seq,
                sigma=config.sigma,
                decay_kind=config.decay_kind,
            )
        )

    # ------------------------------------------------------------------
    def decay(self, distance: int) -> float:
        """Decay factor for *distance*. Delegates to the sequence coupler.

        Complexity: O(1).
        """
        return self._seq_coupler.decay(distance)

    # ------------------------------------------------------------------
    def apply(
        self,
        rates: Tensor,
        edit_positions: Sequence[int],
        *,
        pairing_prob: Union[Tensor, np.ndarray, Sequence[float], None] = None,
        pairing_matrix: Union[Tensor, np.ndarray, None] = None,
    ) -> Tensor:
        """Return a structure-aware boosted copy of *rates*.

        Parameters
        ----------
        rates : Tensor
            Per-position rates, shape ``[L]`` or ``[L, num_ops]``.
        edit_positions : sequence of int
            Positions where edits occurred.
        pairing_prob : 1-D array, optional
            Per-nucleotide pairing probability (e.g. from
            :class:`~mrna_editflow.schema.PrecomputedFeatures`).  Used as a
            coarse fallback when ``pairing_matrix`` is not available.
        pairing_matrix : 2-D array, optional
            Full ``[L, L]`` base-pair probability matrix.  When provided,
            precise base-pair partner and stem-member boosts are applied.

        Returns
        -------
        Tensor
            Same shape as *rates* with structure-localized boosts applied.

        Raises
        ------
        ValueError
            If neither ``pairing_prob`` nor ``pairing_matrix`` is provided, or
            if dimensions do not match ``rates``.

        Complexity: ``O(E * (w + P * s))`` with ``pairing_matrix``;
        ``O(E * structure_window)`` with only ``pairing_prob``.
        """
        if rates.dim() not in (1, 2):
            raise ValueError(
                f"rates must be 1-D [L] or 2-D [L, num_ops], got {rates.dim()}-D"
            )
        L = rates.shape[0]
        if not edit_positions:
            return rates.detach().clone().float()

        pp = _to_float_tensor(pairing_prob, length=L)
        pm = _to_float_matrix(pairing_matrix)
        if pm is not None and pm.shape[0] != L:
            raise ValueError(
                f"pairing_matrix dimension {pm.shape[0]} != rates length {L}"
            )
        if pm is None and pp is None:
            raise ValueError(
                "StructureLocalizedCoupling requires pairing_prob or "
                "pairing_matrix"
            )

        # --- Sequence-neighbour boost (same as SequenceLocalizedCoupling) ---
        boost = torch.ones(L, dtype=torch.float32)
        cfg = self.config

        for i in edit_positions:
            i = int(i)
            self._apply_sequence_boost(boost, i, L, cfg)
            if pm is not None:
                self._apply_matrix_boost(boost, i, L, pm, cfg)
            else:
                self._apply_prob_boost(boost, i, L, pp, cfg)  # type: ignore[arg-type]

        out = rates.detach().clone().float()
        if out.dim() == 1:
            out = out * boost
        else:
            out = out * boost.unsqueeze(-1)
        return out

    # ------------------------------------------------------------------
    def _apply_sequence_boost(
        self,
        boost: Tensor,
        i: int,
        L: int,
        cfg: StructureLocalizedCouplingConfig,
    ) -> None:
        """Multiply sequence-neighbour boost factors into *boost* (in-place).

        Complexity: O(window).
        """
        w = int(cfg.window)
        lo = max(0, i - w)
        hi = min(L, i + w + 1)
        for j in range(lo, hi):
            if j == i:
                continue
            d = abs(j - i)
            boost[j] *= (1.0 + cfg.alpha_seq * self.decay(d))

    # ------------------------------------------------------------------
    def _apply_matrix_boost(
        self,
        boost: Tensor,
        i: int,
        L: int,
        pm: Tensor,
        cfg: StructureLocalizedCouplingConfig,
    ) -> None:
        """Apply base-pair partner and stem-member boosts from *pm*.

        For each partner ``j`` of ``i`` (``pm[i, j] > pair_threshold``):
        - Partner boost: ``boost[j] *= (1 + alpha_pair * pm[i, j])``
        - Stem-member boost: positions ``k`` within ``stem_window`` of ``j``
          get ``boost[k] *= (1 + alpha_stem * decay(|k - j|))``

        Complexity: O(P * stem_window) where P = number of partners of ``i``.
        """
        if i >= pm.shape[0]:
            return
        row = pm[i]
        partners = (row > cfg.pair_threshold).nonzero(as_tuple=True)[0]
        sw = int(cfg.stem_window)
        for j_t in partners:
            j = int(j_t.item())
            if j == i:
                continue
            prob = float(row[j].item())
            # Base-pair partner boost.
            boost[j] *= (1.0 + cfg.alpha_pair * prob)
            # Stem-loop member boost (positions near the partner).
            lo = max(0, j - sw)
            hi = min(L, j + sw + 1)
            for k in range(lo, hi):
                if k == j or k == i:
                    continue
                d = abs(k - j)
                boost[k] *= (1.0 + cfg.alpha_stem * self.decay(d))

    # ------------------------------------------------------------------
    def _apply_prob_boost(
        self,
        boost: Tensor,
        i: int,
        L: int,
        pp: Tensor,
        cfg: StructureLocalizedCouplingConfig,
    ) -> None:
        """Coarse structure boost using only 1-D pairing probabilities.

        Positions ``j`` within ``structure_window`` of ``i`` that have
        ``pairing_prob[j] > pair_threshold`` are boosted by
        ``(1 + alpha_pair * pairing_prob[j] * decay(|j - i|))``.

        This approximates "same structural element" membership without a full
        pairing matrix.

        Complexity: O(structure_window).
        """
        sw = int(cfg.structure_window)
        lo = max(0, i - sw)
        hi = min(L, i + sw + 1)
        for j in range(lo, hi):
            if j == i:
                continue
            pj = float(pp[j].item())
            if pj <= cfg.pair_threshold:
                continue
            d = abs(j - i)
            boost[j] *= (1.0 + cfg.alpha_pair * pj * self.decay(d))

    # ------------------------------------------------------------------
    def __call__(
        self,
        rates: Tensor,
        edit_positions: Sequence[int],
        *,
        pairing_prob: Union[Tensor, np.ndarray, Sequence[float], None] = None,
        pairing_matrix: Union[Tensor, np.ndarray, None] = None,
    ) -> Tensor:
        return self.apply(
            rates,
            edit_positions,
            pairing_prob=pairing_prob,
            pairing_matrix=pairing_matrix,
        )


__all__ = [
    "SequenceLocalizedCouplingConfig",
    "SequenceLocalizedCoupling",
    "StructureLocalizedCouplingConfig",
    "StructureLocalizedCoupling",
]
