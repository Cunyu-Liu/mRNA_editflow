"""Flow-specific diagnostics for mRNA Edit-Flow trajectories (P0-8).

This module computes trajectory-level metrics that probe the *behaviour* of an
Edit-Flow decoder (the true :class:`~mrna_editflow.core.ctmc_sampler.CTMCSampler`
tau-leaping sampler or the historical
:class:`~mrna_editflow.core.proposal_editor.ProposalEditor`). The metrics are:

* :func:`rate_calibration` -- compare predicted CTMC rates to the empirical
  edit frequencies actually realised along the trajectory.
* :func:`path_legality_check` -- verify that no intermediate state in the
  trajectory violates the mRNA edit grammar (frameshift, nonsynonymous
  substitution, region-boundary violation, start/stop tampering).
* :func:`endpoint_recovery` -- measure how close the final sequence is to the
  target (Levenshtein, normalised recovery, protein match).
* :func:`trajectory_edit_distance` -- Levenshtein distance at each step
  (relative to the initial sequence and, optionally, to a reference/target).
* :func:`rate_temporal_profile` -- how the predicted rates evolve over the
  flow time ``t`` (trend, peak, monotonicity).

All functions accept a :class:`Trajectory` (defined here) and return a metrics
dict. They are pure-Python/numpy: no GPU, no trained model, no network. This
makes them fully unit-testable with synthetic trajectories.

Trajectory contract
-------------------
A :class:`Trajectory` is a list of :class:`TrajectoryStep` objects. Each step
carries the post-edit sequence (``sequence``), the flow time (``t``), the edit
applied at that step (``action``) or the set of simultaneous events
(``events``), and optional per-position predicted rates (``rates``) and
per-position region ids (``region_ids``). Step 0 is conventionally the initial
state with no action. Any field may be ``None``; each metric degrades
gracefully and reports ``"available": False`` when the data it needs is absent.

Complexity: all functions are ``O(S * (L + E))`` for ``S`` steps, sequence
length ``L`` and per-step events ``E``; the Levenshtein-based ones add
``O(S * L^2)`` DP cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .constants import (
    CODON_TABLE,
    REGION_CDS,
    START_CODON,
    STOP_CODONS,
    is_valid_cds,
    translate,
)


# ===========================================================================
# Trajectory data model
# ===========================================================================
@dataclass
class TrajectoryStep:
    """One step of a decoder trajectory (post-edit state).

    Attributes
    ----------
    step : int
        Step index (0 = initial state, no action).
    t : float
        Flow time at which the model was queried for this step.
    sequence : str
        Full RNA sequence (5'UTR + CDS + 3'UTR) AFTER the edit at this step.
        For step 0 this is the initial sequence.
    action : mapping, optional
        The single edit applied to reach this state, e.g.
        ``{"op": "sub", "pos": 12, "nt": "G", "old_nt": "A"}``. ``None`` for
        the initial state.
    events : list of mapping, optional
        Simultaneous events (tau-leaping). Used when ``action`` is ``None``.
    rates : mapping, optional
        Per-position predicted rates at this step, keyed by operation:
        ``{"ins": [...], "sub": [...], "del": [...]}`` where each value is a
        sequence of per-position rate values (the model's ``rates[..., op]``
        column). Used by :func:`rate_calibration` and
        :func:`rate_temporal_profile`.
    region_ids : sequence of int, optional
        Per-nucleotide region ids aligned to ``sequence`` (for legality checks
        on trajectories whose length changes via UTR indels).
    """

    step: int = 0
    t: float = 0.0
    sequence: str = ""
    action: Optional[Mapping[str, Any]] = None
    events: List[Mapping[str, Any]] = field(default_factory=list)
    rates: Optional[Mapping[str, Any]] = None
    region_ids: Optional[Sequence[int]] = None


@dataclass
class Trajectory:
    """A full decoder trajectory.

    Attributes
    ----------
    steps : list of TrajectoryStep
        Ordered steps; ``steps[0]`` is the initial state.
    initial_sequence : str
        Convenience copy of the initial sequence (``steps[0].sequence``).
    target_sequence : str, optional
        Target sequence the decoder was steering toward.
    final_sequence : str, optional
        Final sequence (defaults to the last step's sequence).
    cds_start, cds_end : int, optional
        Constant CDS boundaries valid for sub-only trajectories. For indel
        trajectories supply per-step ``region_ids`` instead.
    """

    steps: List[TrajectoryStep] = field(default_factory=list)
    initial_sequence: str = ""
    target_sequence: Optional[str] = None
    final_sequence: Optional[str] = None
    cds_start: Optional[int] = None
    cds_end: Optional[int] = None


# ===========================================================================
# Internal helpers
# ===========================================================================
def _levenshtein(a: str, b: str) -> int:
    """Levenshtein edit distance via a numpy DP table.

    Complexity: ``O(len(a) * len(b))``.
    """
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = np.zeros((m + 1, n + 1), dtype=np.int64)
    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)
    a_chars = a
    for i in range(1, m + 1):
        ai = a_chars[i - 1]
        row = dp[i]
        prev = dp[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost)
    return int(dp[m, n])


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, returning 0.0 on degenerate input. Complexity: O(N)."""
    if x.size < 2:
        return 0.0
    xm = x - x.mean()
    ym = y - y.mean()
    denom = float(np.sqrt(float((xm * xm).sum()) * float((ym * ym).sum())))
    if denom <= 0.0:
        return 0.0
    return float((xm * ym).sum() / denom)


def _rank(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) of ``a`` (ties get the mean rank). Complexity: O(N log N)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
    # Resolve ties by averaging ranks.
    sorted_a = a[order]
    i = 0
    while i < a.size:
        j = i + 1
        while j < a.size and sorted_a[j] == sorted_a[i]:
            j += 1
        if j > i + 1:
            ranks[order[i:j]] = float(np.mean(ranks[order[i:j]]))
        i = j
    return ranks


def _linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Least-squares ``y = slope * x + intercept``. Complexity: O(N)."""
    if x.size < 2:
        return 0.0, float(y.mean()) if y.size else 0.0
    xm = x - x.mean()
    ym = y - y.mean()
    var = float((xm * xm).sum())
    if var <= 0.0:
        return 0.0, float(y.mean())
    slope = float((xm * ym).sum() / var)
    intercept = float(y.mean() - slope * x.mean())
    return slope, intercept


def _ece(pred: np.ndarray, emp: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error over ``n_bins`` probability bins.

    Both arrays are interpreted as predicted probabilities paired with 0/1
    empirical outcomes. Complexity: ``O(N)``.
    """
    if pred.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        if k == n_bins - 1:
            mask = (pred >= lo) & (pred <= hi)
        else:
            mask = (pred >= lo) & (pred < hi)
        if not np.any(mask):
            continue
        ece += float(np.mean(pred[mask]) - np.mean(emp[mask])) ** 2 * int(np.sum(mask))
    return float(np.sqrt(ece / max(int(pred.size), 1)))


def _cds_span(region_ids: Sequence[int]) -> Tuple[Optional[int], Optional[int]]:
    """Return ``(cds_start, cds_end)`` from per-position region ids.

    ``cds_end`` is exclusive. Returns ``(None, None)`` if there is no CDS.
    Complexity: ``O(len(region_ids))``.
    """
    start: Optional[int] = None
    end: Optional[int] = None
    for i, r in enumerate(region_ids):
        if int(r) == REGION_CDS:
            if start is None:
                start = i
            end = i + 1
    return start, end


def _resolve_cds_boundaries(
    trajectory: Trajectory,
    cds_start: Optional[int],
    cds_end: Optional[int],
    region_ids: Optional[Sequence[int]],
) -> Tuple[Optional[int], Optional[int], bool]:
    """Resolve constant CDS boundaries for the trajectory.

    Returns ``(cds_start, cds_end, available)`` where ``available`` is False
    when no region information is available at all. Complexity: O(L).
    """
    if cds_start is not None and cds_end is not None:
        return cds_start, cds_end, True
    rid = region_ids
    if rid is None and trajectory.steps:
        rid = trajectory.steps[0].region_ids
    if rid is not None:
        cs, ce = _cds_span(rid)
        cs = cs if cds_start is None else cds_start
        ce = ce if cds_end is None else cds_end
        return cs, ce, cs is not None
    cs = trajectory.cds_start if cds_start is None else cds_start
    ce = trajectory.cds_end if cds_end is None else cds_end
    return cs, ce, cs is not None and ce is not None


def _event_iter(step: TrajectoryStep) -> List[Mapping[str, Any]]:
    """Yield the events applied at ``step`` (action first, then simultaneous)."""
    evs: List[Mapping[str, Any]] = []
    if step.action:
        evs.append(step.action)
    evs.extend(step.events)
    return evs


# ===========================================================================
# Metric 1: rate calibration
# ===========================================================================
def rate_calibration(trajectory: Trajectory) -> Dict[str, Any]:
    """Compare predicted CTMC rates to empirical edit frequencies.

    For every step that carries predicted ``rates`` we build the predicted
    event distribution at ``(op, position)`` granularity by normalising the
    per-position rate ``lambda_{op, pos}`` by the step's total rate mass. The
    empirical outcome is ``1 / n_events`` for each ``(op, pos)`` actually edited
    that step and ``0`` otherwise. Aggregated across steps this yields paired
    ``(predicted_prob, empirical_freq)`` vectors used for:

    * Pearson and Spearman correlation,
    * MAE and RMSE,
    * a least-squares calibration slope/intercept,
    * a binned Expected Calibration Error (ECE).

    Parameters
    ----------
    trajectory : Trajectory
        Steps must carry ``rates`` (``{"ins","sub","del"}`` per-position) and
        ``action``/``events`` for the comparison to be meaningful.

    Returns
    -------
    dict
        ``{"available": bool, "n_steps": int, "n_event_slots": int,
        "pearson_r": float, "spearman_r": float, "mae": float, "rmse": float,
        "calibration_slope": float, "calibration_intercept": float,
        "ece": float, "predicted_mean": float, "empirical_mean": float}``.

    Complexity: ``O(S * (L + E))``.
    """
    pred: List[float] = []
    emp: List[float] = []
    n_steps = 0
    for st in trajectory.steps:
        if not st.rates:
            continue
        n_steps += 1
        rate_map: Dict[Tuple[str, int], float] = {}
        total = 0.0
        for op in ("ins", "sub", "del"):
            arr = np.asarray(st.rates.get(op, []), dtype=float).ravel()
            for pos, r in enumerate(arr):
                rate_map[(op, int(pos))] = float(r)
                total += float(r)
        if total <= 0.0:
            continue
        occurred: set = set()
        for ev in _event_iter(st):
            op = str(ev.get("op", "")).lower()
            pos = ev.get("pos")
            if op in ("ins", "sub", "del") and pos is not None:
                occurred.add((op, int(pos)))
        n_ev = max(1, len(occurred))
        for key, r in rate_map.items():
            pred.append(r / total)
            emp.append((1.0 / n_ev) if key in occurred else 0.0)

    if not pred:
        return {
            "available": False,
            "n_steps": n_steps,
            "n_event_slots": 0,
            "reason": "no per-step rate data",
        }
    pred_arr = np.asarray(pred, dtype=float)
    emp_arr = np.asarray(emp, dtype=float)
    slope, intercept = _linear_fit(pred_arr, emp_arr)
    return {
        "available": True,
        "n_steps": n_steps,
        "n_event_slots": int(pred_arr.size),
        "pearson_r": _pearson(pred_arr, emp_arr),
        "spearman_r": _pearson(_rank(pred_arr), _rank(emp_arr)),
        "mae": float(np.mean(np.abs(pred_arr - emp_arr))),
        "rmse": float(np.sqrt(np.mean((pred_arr - emp_arr) ** 2))),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "ece": _ece(pred_arr, emp_arr, n_bins=10),
        "predicted_mean": float(pred_arr.mean()),
        "empirical_mean": float(emp_arr.mean()),
    }


# ===========================================================================
# Metric 2: path legality check
# ===========================================================================
def path_legality_check(
    trajectory: Trajectory,
    *,
    cds_start: Optional[int] = None,
    cds_end: Optional[int] = None,
    region_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Verify a trajectory never passes through an illegal mRNA state.

    For each step's sequence the following are checked (when region info is
    available):

    * **Frameshift**: the CDS length is a multiple of 3.
    * **Nonsynonymous**: the translated protein equals the initial protein.
    * **Start codon**: the CDS begins with ``AUG``.
    * **Stop codon**: the CDS ends with a stop codon and has no internal stop.
    * **Boundary**: each applied edit respects the region grammar -- indels
      never occur inside CDS, and CDS substitutions are synonymous (verified
      via the action's ``old_nt`` when present).

    Parameters
    ----------
    trajectory : Trajectory
    cds_start, cds_end : int, optional
        Constant CDS boundaries (valid for sub-only trajectories).
    region_ids : sequence of int, optional
        Per-position region ids of the initial sequence (used to derive CDS
        boundaries when ``cds_start``/``cds_end`` are not given).

    Returns
    -------
    dict
        ``{"is_legal": bool, "region_info_available": bool, "n_steps_checked":
        int, "n_frameshift_violations": int, "n_nonsynonymous_violations":
        int, "n_boundary_violations": int, "n_start_codon_violations": int,
        "n_stop_codon_violations": int, "violation_steps": [...],
        "violation_types": {step: [types, ...]}}``.

    Complexity: ``O(S * (L + E))``.
    """
    cs, ce, region_available = _resolve_cds_boundaries(
        trajectory, cds_start, cds_end, region_ids
    )
    counts = {
        "frameshift": 0,
        "nonsynonymous": 0,
        "boundary": 0,
        "start_codon": 0,
        "stop_codon": 0,
    }
    violation_steps: List[int] = []
    violation_types: Dict[int, List[str]] = {}
    n_checked = 0
    init_protein: Optional[str] = None

    for st in trajectory.steps:
        seq = st.sequence
        if not seq:
            continue
        n_checked += 1
        step_violations: List[str] = []

        # Per-step region ids override the constant boundaries when present
        # (indel trajectories change length and region layout).
        step_cs, step_ce = cs, ce
        if st.region_ids is not None:
            s, e = _cds_span(st.region_ids)
            step_cs = s if s is not None else step_cs
            step_ce = e if e is not None else step_ce

        if step_cs is not None and step_ce is not None and step_ce > step_cs:
            cds = seq[step_cs:step_ce]
            # Frameshift.
            if len(cds) % 3 != 0:
                counts["frameshift"] += 1
                step_violations.append("frameshift")
            else:
                # Start codon.
                if len(cds) >= 3 and cds[:3] != START_CODON:
                    counts["start_codon"] += 1
                    step_violations.append("start_codon")
                # Stop codon / premature stop.
                prot = translate(cds)
                if len(cds) >= 3 and cds[-3:] not in STOP_CODONS:
                    counts["stop_codon"] += 1
                    step_violations.append("stop_codon")
                elif "*" in prot[:-1]:
                    counts["stop_codon"] += 1
                    step_violations.append("premature_stop")
                # Nonsynonymous (vs initial protein).
                if init_protein is None:
                    init_protein = prot
                elif prot != init_protein:
                    counts["nonsynonymous"] += 1
                    step_violations.append("nonsynonymous")

            # Boundary checks on the edit that produced this step.
            for ev in _event_iter(st):
                vtype = _action_boundary_violation(ev, seq, step_cs, step_ce)
                if vtype is not None:
                    counts["boundary"] += 1
                    step_violations.append(f"boundary:{vtype}")

        if step_violations:
            violation_steps.append(st.step)
            violation_types[st.step] = step_violations

    n_violations = sum(counts.values())
    return {
        "is_legal": n_violations == 0,
        "region_info_available": bool(region_available),
        "n_steps_checked": n_checked,
        "n_frameshift_violations": counts["frameshift"],
        "n_nonsynonymous_violations": counts["nonsynonymous"],
        "n_boundary_violations": counts["boundary"],
        "n_start_codon_violations": counts["start_codon"],
        "n_stop_codon_violations": counts["stop_codon"],
        "violation_steps": violation_steps,
        "violation_types": violation_types,
    }


def _action_boundary_violation(
    ev: Mapping[str, Any],
    post_seq: str,
    cds_start: Optional[int],
    cds_end: Optional[int],
) -> Optional[str]:
    """Classify a region-grammar violation of a single edit, or ``None``.

    ``post_seq`` is the sequence AFTER the edit; the edit's ``old_nt`` (when
    present) is used to reconstruct the pre-edit codon for synonymous checking.
    Complexity: O(1).
    """
    op = str(ev.get("op", "")).lower()
    pos = ev.get("pos")
    if pos is None or op not in ("ins", "sub", "del"):
        return None
    pos = int(pos)
    in_cds = (
        cds_start is not None
        and cds_end is not None
        and cds_start <= pos < cds_end
    )
    if op in ("ins", "del"):
        return "cds_indel" if in_cds else None
    if op == "sub" and in_cds:
        if cds_start is None or cds_end is None:
            return None
        phase = (pos - cds_start) % 3
        cstart = pos - phase
        if cstart < cds_start or cstart + 3 > cds_end or cstart + 3 > len(post_seq):
            return "boundary"
        post_codon = post_seq[cstart : cstart + 3]
        old_nt = ev.get("old_nt")
        if old_nt is None:
            return None  # cannot verify synonymity without the pre-edit base
        pre_codon = post_codon[:phase] + str(old_nt) + post_codon[phase + 1 :]
        if CODON_TABLE.get(pre_codon) != CODON_TABLE.get(post_codon):
            return "nonsynonymous"
        if pre_codon == START_CODON or pre_codon in STOP_CODONS:
            return "start_stop_tampered"
        return None
    return None


# ===========================================================================
# Metric 3: endpoint recovery
# ===========================================================================
def endpoint_recovery(
    trajectory: Trajectory,
    target_sequence: Optional[str] = None,
) -> Dict[str, Any]:
    """Measure how close the final sequence is to the target.

    Parameters
    ----------
    trajectory : Trajectory
    target_sequence : str, optional
        Override for ``trajectory.target_sequence``.

    Returns
    -------
    dict
        ``{"available": bool, "levenshtein": int, "normalized_recovery": float,
        "exact_match": bool, "length_diff": int, "protein_match": bool|None,
        "cds_levenshtein": int|None}``. ``normalized_recovery`` is
        ``1 - dist / max(len(final), len(target), 1)``.

    Complexity: ``O(L_final * L_target)`` (Levenshtein DP).
    """
    target = target_sequence if target_sequence is not None else trajectory.target_sequence
    if trajectory.final_sequence is not None:
        final = trajectory.final_sequence
    elif trajectory.steps:
        final = trajectory.steps[-1].sequence
    else:
        final = trajectory.initial_sequence
    if target is None:
        return {"available": False, "reason": "no target sequence"}
    dist = _levenshtein(final, target)
    max_len = max(len(final), len(target), 1)
    out: Dict[str, Any] = {
        "available": True,
        "levenshtein": int(dist),
        "normalized_recovery": float(1.0 - dist / max_len),
        "exact_match": bool(final == target),
        "length_diff": int(len(final) - len(target)),
    }
    cs, ce = trajectory.cds_start, trajectory.cds_end
    if (
        cs is not None
        and ce is not None
        and ce <= len(final)
        and ce <= len(target)
    ):
        out["protein_match"] = bool(translate(final[cs:ce]) == translate(target[cs:ce]))
        out["cds_levenshtein"] = int(_levenshtein(final[cs:ce], target[cs:ce]))
    else:
        out["protein_match"] = None
        out["cds_levenshtein"] = None
    return out


# ===========================================================================
# Metric 4: trajectory edit distance
# ===========================================================================
def trajectory_edit_distance(
    trajectory: Trajectory,
    *,
    reference: Optional[str] = None,
) -> Dict[str, Any]:
    """Levenshtein distance at each step.

    By default the distance is measured from the **initial** sequence
    (``trajectory.initial_sequence`` or ``steps[0].sequence``). When
    ``reference`` is given (or ``trajectory.target_sequence`` is set), a second
    per-step series ``distances_to_reference`` is reported.

    Parameters
    ----------
    trajectory : Trajectory
    reference : str, optional
        Reference sequence (e.g. the target) for the second distance series.

    Returns
    -------
    dict
        ``{"available": bool, "n_steps": int, "distances": [...],
        "max": int, "final": int, "monotonic_non_decreasing": bool,
        "distances_to_reference": [...]}``.

    Complexity: ``O(S * L^2)`` (one Levenshtein DP per step).
    """
    seqs = [st.sequence for st in trajectory.steps]
    if not seqs:
        return {"available": False, "n_steps": 0}
    base = reference if reference is not None else (trajectory.initial_sequence or seqs[0])
    distances = [int(_levenshtein(base, s)) for s in seqs]
    out: Dict[str, Any] = {
        "available": True,
        "n_steps": len(seqs),
        "distances": distances,
        "max": int(max(distances)) if distances else 0,
        "final": int(distances[-1]) if distances else 0,
        "monotonic_non_decreasing": bool(
            all(distances[i] <= distances[i + 1] for i in range(len(distances) - 1))
        ),
    }
    ref = reference if reference is not None else trajectory.target_sequence
    if ref is not None:
        out["distances_to_reference"] = [int(_levenshtein(ref, s)) for s in seqs]
    return out


# ===========================================================================
# Metric 5: rate temporal profile
# ===========================================================================
def rate_temporal_profile(trajectory: Trajectory) -> Dict[str, Any]:
    """Summarise how predicted rates change over flow time ``t``.

    For every step carrying ``rates``, the per-operation total rate
    (``sum_pos lambda_{op, pos}``) and the mean rate across operations are
    recorded against the step's time ``t``. We then report the linear trend
    (slope/intercept of total rate vs ``t``), the peak time and rate, and
    whether the total rate is monotonically decreasing over the trajectory.

    Parameters
    ----------
    trajectory : Trajectory
        Steps must carry ``t`` and ``rates``.

    Returns
    -------
    dict
        ``{"available": bool, "n_steps": int, "times": [...],
        "total_rates": {"ins": [...], "sub": [...], "del": [...]},
        "mean_rates": [...], "slope": float, "intercept": float,
        "peak_t": float, "peak_rate": float, "monotonic_decreasing": bool}``.

    Complexity: ``O(S * L)``.
    """
    times: List[float] = []
    totals: Dict[str, List[float]] = {"ins": [], "sub": [], "del": []}
    means: List[float] = []
    for st in trajectory.steps:
        if not st.rates:
            continue
        times.append(float(st.t))
        step_totals: List[float] = []
        for op in ("ins", "sub", "del"):
            arr = np.asarray(st.rates.get(op, []), dtype=float).ravel()
            t = float(arr.sum()) if arr.size else 0.0
            totals[op].append(t)
            step_totals.append(t)
        means.append(float(np.mean(step_totals)) if step_totals else 0.0)

    if not times:
        return {"available": False, "n_steps": 0}
    agg = np.asarray(
        [totals["ins"][i] + totals["sub"][i] + totals["del"][i] for i in range(len(times))],
        dtype=float,
    )
    t_arr = np.asarray(times, dtype=float)
    slope, intercept = _linear_fit(t_arr, agg)
    peak_idx = int(np.argmax(agg)) if agg.size else 0
    return {
        "available": True,
        "n_steps": len(times),
        "times": times,
        "total_rates": totals,
        "mean_rates": means,
        "slope": float(slope),
        "intercept": float(intercept),
        "peak_t": float(times[peak_idx]) if times else 0.0,
        "peak_rate": float(agg[peak_idx]) if agg.size else 0.0,
        "monotonic_decreasing": bool(
            all(agg[i] >= agg[i + 1] for i in range(len(agg) - 1))
        ),
    }


__all__ = [
    "Trajectory",
    "TrajectoryStep",
    "rate_calibration",
    "path_legality_check",
    "endpoint_recovery",
    "trajectory_edit_distance",
    "rate_temporal_profile",
]
