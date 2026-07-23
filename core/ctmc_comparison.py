"""CTMC vs. ProposalEditor comparison framework (P1-5).

This module provides a side-by-side comparison harness that runs both the true
:class:`~mrna_editflow.core.ctmc_sampler.CTMCSampler` (tau-leaping over a
time-varying rate field) and the historical
:class:`~mrna_editflow.core.proposal_editor.ProposalEditor` (fixed-time
sequential proposal ranker) on the same source/target pair, then reports a
unified metrics dictionary.

Metrics computed for each sampler
---------------------------------
* **Final sequence quality** -- via a user-supplied or default scoring
  function (default: normalised Levenshtein recovery to the target).
* **Number of edits** -- event count for CTMC, Levenshtein distance for the
  proposal editor.
* **Protein identity preserved** -- whether the translated CDS protein of the
  final sequence matches the source (CDS-only; relevant for synonymous-edit
  tasks).
* **Total compute time** -- wall-clock seconds measured with
  :func:`time.perf_counter`.
* **Trajectory edit distance profile** -- per-step Levenshtein distance from
  the initial sequence (and optionally to the target), computed via
  :func:`~mrna_editflow.core.flow_diagnostics.trajectory_edit_distance`.
* **Endpoint recovery** -- Levenshtein distance, normalised recovery, exact
  match and protein match of the final sequence vs. the target, via
  :func:`~mrna_editflow.core.flow_diagnostics.endpoint_recovery`.
* **Path legality** -- verifies that no intermediate state in the trajectory
  violates the mRNA edit grammar, via
  :func:`~mrna_editflow.core.flow_diagnostics.path_legality_check`.

The framework is model-agnostic: a single ``model_fn`` callable is shared
between both samplers.  All code is CPU-testable: no GPU or trained model is
required (a dummy ``model_fn`` returning random rate tensors suffices for
unit tests).

Complexity: one comparison costs ``O(F_ctmc * K + F_prop * E + S * L^2)``
where ``F`` = model forward cost, ``K`` = CTMC steps, ``E`` = edit budget,
``S`` = trajectory steps, and ``L`` = sequence length (the ``L^2`` comes from
the Levenshtein DP in the edit-distance profile).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .constants import (
    ID_TO_NUC,
    is_valid_cds,
    translate,
)
from .schema import MRNARecord
from .ctmc_sampler import CTMCConfig, CTMCSampler, CTMCTrajectory
from .flow_diagnostics import (
    Trajectory,
    TrajectoryStep,
    endpoint_recovery,
    path_legality_check,
    trajectory_edit_distance,
)
from .proposal_editor import ProposalEditor, ProposalEditorConfig

# Type aliases -- keep them consistent with the samplers' own aliases.
ModelFn = Callable[..., Dict[str, torch.Tensor]]
ScoringFn = Callable[[str, Optional[str]], float]


# ===========================================================================
# Configuration
# ===========================================================================
@dataclass
class ComparisonConfig:
    """Configuration for :class:`CTMCComparison`.

    Attributes
    ----------
    ctmc_n_steps : int
        Number of CTMC time-discretisation steps (``K``).
    ctmc_max_events_per_step : int
        Maximum simultaneous events per CTMC tau-leap.
    ctmc_grammar_safe : bool
        Whether CTMC events are filtered through the region grammar.
    proposal_edit_budget : int
        Upper bound on the number of ProposalEditor edits.
    proposal_task_id : str
        Task type for the ProposalEditor (``T2``/``T3``/``T4``/``T5``/``T6``).
    proposal_time_step : float
        Fixed bridge time ``t`` used by the ProposalEditor for scoring.
    proposal_top_k : int
        Top-k legal proposals retained per ProposalEditor step.
    seed : int
        RNG seed shared by both samplers for reproducibility.
    device : str
        Torch device string (``"cpu"`` for testing).
    """

    ctmc_n_steps: int = 50
    ctmc_max_events_per_step: int = 5
    ctmc_grammar_safe: bool = True
    proposal_edit_budget: int = 3
    proposal_task_id: str = "T5"
    proposal_time_step: float = 0.5
    proposal_top_k: int = 8
    seed: int = 42
    device: str = "cpu"


# ===========================================================================
# Helpers
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
    for i in range(1, m + 1):
        ai = a[i - 1]
        row = dp[i]
        prev = dp[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost)
    return int(dp[m, n])


def _default_scoring_fn(final_seq: str, target_seq: Optional[str]) -> float:
    """Default quality score: normalised Levenshtein recovery to *target*.

    Returns ``0.0`` when no target is available (neutral score).

    Complexity: ``O(len(final) * len(target))``.
    """
    if target_seq is None:
        return 0.0
    dist = _levenshtein(final_seq, target_seq)
    max_len = max(len(final_seq), len(target_seq), 1)
    return float(1.0 - dist / max_len)


def _tokens_to_seq(tokens: Sequence[int]) -> str:
    """Convert a token-id sequence to an RNA string.

    Tokens outside ``[0, 3]`` map to ``'N'``.

    Complexity: ``O(len(tokens))``.
    """
    return "".join(ID_TO_NUC.get(int(t), "N") for t in tokens)


def _replay_ctmc_sequences(
    source_tokens: Sequence[int],
    steps: List[Dict[str, Any]],
) -> List[str]:
    """Reconstruct per-step sequences by replaying CTMC events.

    The CTMC sampler stores events as ``(op, pos, token)`` tuples per step.
    Events within a step are applied right-to-left (by descending position),
    matching :func:`mrna_editflow.core.ctmc_sampler._apply_events`.

    Parameters
    ----------
    source_tokens : sequence of int
        Token ids of the source (initial) sequence.
    steps : list of dict
        ``CTMCTrajectory.steps`` -- each dict has an ``"events"`` key.

    Returns
    -------
    list of str
        Sequence strings: ``[source, after_step_0, after_step_1, ...]``.

    Complexity: ``O(S * E)`` for ``S`` steps and ``E`` events per step.
    """
    cur: List[int] = [int(t) for t in source_tokens]
    sequences: List[str] = [_tokens_to_seq(cur)]
    for step_info in steps:
        events = step_info.get("events", [])
        # Apply right-to-left (same ordering as _apply_events in ctmc_sampler).
        for op, pos, token in sorted(events, key=lambda e: -int(e[1])):
            pos_i = int(pos)
            if op == "sub" and 0 <= pos_i < len(cur):
                cur[pos_i] = int(token)
            elif op == "del" and 0 <= pos_i < len(cur):
                del cur[pos_i]
            elif op == "ins":
                cur.insert(pos_i, int(token))
        sequences.append(_tokens_to_seq(cur))
    return sequences


def _build_ctmc_trajectory(
    source: MRNARecord,
    ctmc_traj: CTMCTrajectory,
) -> Trajectory:
    """Build a :class:`Trajectory` from a :class:`CTMCTrajectory`.

    Intermediate sequences are reconstructed by replaying events (see
    :func:`_replay_ctmc_sequences`).  Each step's flow time ``t`` is taken
    from the CTMC trajectory.

    Complexity: ``O(S * E)`` for sequence replay.
    """
    source_tokens = source.token_ids()
    sequences = _replay_ctmc_sequences(source_tokens, ctmc_traj.steps)
    steps: List[TrajectoryStep] = []
    for k, seq in enumerate(sequences):
        if k == 0:
            t_val = 0.0
            events: List[Dict[str, Any]] = []
        else:
            step_idx = k - 1
            t_val = (
                float(ctmc_traj.steps[step_idx]["t"])
                if step_idx < len(ctmc_traj.steps)
                else 1.0
            )
            events = [
                {
                    "op": str(op),
                    "pos": int(pos),
                    "nt": ID_TO_NUC.get(int(token), "N"),
                }
                for op, pos, token in ctmc_traj.steps[step_idx].get("events", [])
            ]
        steps.append(
            TrajectoryStep(step=k, t=t_val, sequence=seq, events=events)
        )
    cds_start = source.cds_start if source.cds else None
    cds_end = source.cds_end if source.cds else None
    return Trajectory(
        steps=steps,
        initial_sequence=sequences[0] if sequences else "",
        final_sequence=sequences[-1] if sequences else "",
        target_sequence=None,
        cds_start=cds_start,
        cds_end=cds_end,
    )


def _build_proposal_trajectory(
    source: MRNARecord,
    result: MRNARecord,
    time_step: float,
) -> Trajectory:
    """Build a :class:`Trajectory` from a ProposalEditor result.

    The ProposalEditor only exposes the final edited record (no intermediate
    states), so the trajectory has two steps: source (``t=0``) and final
    (``t=time_step``).

    Complexity: ``O(1)``.
    """
    cds_start = source.cds_start if source.cds else None
    cds_end = source.cds_end if source.cds else None
    return Trajectory(
        steps=[
            TrajectoryStep(step=0, t=0.0, sequence=source.seq),
            TrajectoryStep(step=1, t=float(time_step), sequence=result.seq),
        ],
        initial_sequence=source.seq,
        final_sequence=result.seq,
        target_sequence=None,
        cds_start=cds_start,
        cds_end=cds_end,
    )


def _protein_identity(
    source: MRNARecord,
    final_seq: str,
    cds_start: int,
    cds_end: int,
) -> Dict[str, Any]:
    """Check whether protein identity is preserved between source and final.

    Parameters
    ----------
    source : MRNARecord
        The original (pre-edit) record.
    final_seq : str
        The full final sequence (5'UTR + CDS + 3'UTR) after editing.
    cds_start, cds_end : int
        CDS boundaries in the source coordinates.

    Returns
    -------
    dict
        ``{"preserved": bool|None, "source_protein": str,
        "final_protein": str, "source_cds_valid": bool,
        "final_cds_valid": bool}``.

    Complexity: ``O(len(cds))``.
    """
    source_cds = source.cds
    if not source_cds:
        return {
            "preserved": None,
            "reason": "source has no CDS",
            "source_protein": "",
            "final_protein": "",
            "source_cds_valid": False,
            "final_cds_valid": False,
        }
    if cds_end > len(final_seq):
        return {
            "preserved": None,
            "reason": "final sequence shorter than CDS end",
            "source_protein": translate(source_cds),
            "final_protein": "",
            "source_cds_valid": is_valid_cds(source_cds),
            "final_cds_valid": False,
        }
    final_cds = final_seq[cds_start:cds_end]
    source_prot = translate(source_cds)
    final_prot = translate(final_cds)
    return {
        "preserved": bool(source_prot == final_prot),
        "source_protein": source_prot,
        "final_protein": final_prot,
        "source_cds_valid": is_valid_cds(source_cds),
        "final_cds_valid": is_valid_cds(final_cds),
    }


def _compute_sampler_metrics(
    source: MRNARecord,
    final_seq: str,
    target_seq: Optional[str],
    n_edits: int,
    elapsed: float,
    trajectory: Trajectory,
    scoring_fn: ScoringFn,
) -> Dict[str, Any]:
    """Compute the standard metrics for one sampler's output.

    Parameters
    ----------
    source : MRNARecord
        The original (pre-edit) record.
    final_seq : str
        Full final sequence after editing.
    target_seq : str, optional
        Target sequence for quality scoring (may be ``None``).
    n_edits : int
        Number of edits applied (from the sampler's own accounting).
    elapsed : float
        Wall-clock compute time in seconds.
    trajectory : Trajectory
        Flow-diagnostics trajectory for edit-distance profiling.
    scoring_fn : callable
        Scoring function ``(final_seq, target_seq) -> float``.

    Returns
    -------
    dict
        ``{"final_sequence", "final_length", "n_edits", "elapsed_seconds",
        "quality_score", "protein_identity", "trajectory_edit_distance",
        "endpoint_recovery", "path_legality"}``.

    Complexity: ``O(S * L^2)`` (dominated by the Levenshtein DP in
    :func:`trajectory_edit_distance` and :func:`endpoint_recovery`).
    """
    cds_start = source.cds_start
    cds_end = source.cds_end
    quality = scoring_fn(final_seq, target_seq)
    protein = _protein_identity(source, final_seq, cds_start, cds_end)
    # Attach the target for the edit-distance profile (distances_to_reference).
    trajectory.target_sequence = target_seq
    edit_dist_profile = trajectory_edit_distance(
        trajectory, reference=target_seq
    )
    recovery = endpoint_recovery(trajectory, target_sequence=target_seq)
    legality = path_legality_check(
        trajectory, cds_start=cds_start, cds_end=cds_end
    )
    return {
        "final_sequence": final_seq,
        "final_length": len(final_seq),
        "n_edits": int(n_edits),
        "elapsed_seconds": float(elapsed),
        "quality_score": float(quality),
        "protein_identity": protein,
        "trajectory_edit_distance": edit_dist_profile,
        "endpoint_recovery": recovery,
        "path_legality": legality,
    }


# ===========================================================================
# Comparison harness
# ===========================================================================
class CTMCComparison:
    """Side-by-side comparison of CTMCSampler and ProposalEditor.

    Given a shared model callable, this harness runs both samplers on the same
    source/target pair, computes a unified set of metrics for each, and
    produces a comparison dictionary with all metrics side-by-side.

    Usage::

        comparison = CTMCComparison(model_fn, ComparisonConfig(ctmc_n_steps=20))
        report = comparison.compare_pair(source_record, target_record)
        # report["ctmc"]     -- CTMC metrics
        # report["proposal"]  -- ProposalEditor metrics
        # report["comparison"] -- side-by-side deltas

    The framework is fully CPU-testable: pass a dummy ``model_fn`` that returns
    random rate tensors (no trained model needed).

    Complexity per pair: ``O(F_ctmc * K + F_prop * E + S * L^2)``.
    """

    def __init__(
        self,
        model_fn: ModelFn,
        config: ComparisonConfig = ComparisonConfig(),
        scoring_fn: Optional[ScoringFn] = None,
    ) -> None:
        self.model_fn = model_fn
        self.config = config
        self.scoring_fn: ScoringFn = scoring_fn or _default_scoring_fn

    # ------------------------------------------------------------------
    def compare_pair(
        self,
        source: MRNARecord,
        target: Optional[MRNARecord] = None,
        backbone: Any = None,
    ) -> Dict[str, Any]:
        """Run both samplers on one source/target pair and compare.

        Parameters
        ----------
        source : MRNARecord
            The source transcript to edit (copied by each sampler; never
            mutated).
        target : MRNARecord, optional
            Target transcript for quality scoring and edit-distance profiling.
            If ``None``, the source's own region/phase structure is used as the
            CTMC reference and quality scores default to ``0.0``.
        backbone : object, optional
            Frozen backbone passed through to ``model_fn``.

        Returns
        -------
        dict
            ``{"source_id", "target_id", "ctmc": {...}, "proposal": {...},
            "comparison": {...}}``.  Each sampler sub-dict contains the metrics
            from :func:`_compute_sampler_metrics`.  The ``"comparison"``
            sub-dict contains side-by-side deltas.

        Complexity: ``O(F_ctmc * K + F_prop * E + S * L^2)``.
        """
        cfg = self.config
        device = torch.device(cfg.device)
        target_seq = target.seq if target is not None else None

        # --- Run CTMCSampler ---
        ctmc_metrics = self._run_ctmc(source, target, target_seq, backbone, device)

        # --- Run ProposalEditor ---
        proposal_metrics = self._run_proposal(
            source, target_seq, backbone, device
        )

        # --- Side-by-side comparison ---
        return {
            "source_id": source.transcript_id,
            "target_id": target.transcript_id if target else None,
            "ctmc": ctmc_metrics,
            "proposal": proposal_metrics,
            "comparison": {
                "quality_diff": float(
                    ctmc_metrics["quality_score"]
                    - proposal_metrics["quality_score"]
                ),
                "speed_diff": float(
                    ctmc_metrics["elapsed_seconds"]
                    - proposal_metrics["elapsed_seconds"]
                ),
                "edits_diff": int(
                    ctmc_metrics["n_edits"] - proposal_metrics["n_edits"]
                ),
                "ctmc_faster": bool(
                    ctmc_metrics["elapsed_seconds"]
                    < proposal_metrics["elapsed_seconds"]
                ),
                "ctmc_higher_quality": bool(
                    ctmc_metrics["quality_score"]
                    > proposal_metrics["quality_score"]
                ),
                "ctmc_fewer_edits": bool(
                    ctmc_metrics["n_edits"] < proposal_metrics["n_edits"]
                ),
            },
        }

    # ------------------------------------------------------------------
    def _run_ctmc(
        self,
        source: MRNARecord,
        target: Optional[MRNARecord],
        target_seq: Optional[str],
        backbone: Any,
        device: torch.device,
    ) -> Dict[str, Any]:
        """Run the CTMCSampler and compute its metrics.

        Complexity: ``O(F * K + S * L^2)``.
        """
        cfg = self.config
        ctmc_config = CTMCConfig(
            n_steps=cfg.ctmc_n_steps,
            max_events_per_step=cfg.ctmc_max_events_per_step,
            grammar_safe=cfg.ctmc_grammar_safe,
            seed=cfg.seed,
        )
        ctmc_sampler = CTMCSampler(self.model_fn, ctmc_config)

        # The CTMC sampler uses token_ids/region_ids/phase_ids as the
        # region/phase *reference* (typically the target), and source_seq as
        # the starting point.  When no target is given, fall back to source.
        ref = target if target is not None else source
        ref_tokens = torch.tensor(
            [ref.token_ids()], dtype=torch.long, device=device
        )
        ref_region = torch.tensor(
            [ref.region_ids()], dtype=torch.long, device=device
        )
        ref_phase = torch.tensor(
            [ref.codon_phases()], dtype=torch.long, device=device
        )
        source_tokens = torch.tensor(
            [source.token_ids()], dtype=torch.long, device=device
        )

        t0 = time.perf_counter()
        ctmc_traj = ctmc_sampler.sample(
            ref_tokens,
            ref_region,
            ref_phase,
            backbone,
            device,
            source_seq=source_tokens,
        )
        ctmc_elapsed = time.perf_counter() - t0

        ctmc_final_seq = (
            _tokens_to_seq(ctmc_traj.final_tokens)
            if ctmc_traj.final_tokens
            else ""
        )
        ctmc_trajectory = _build_ctmc_trajectory(source, ctmc_traj)
        return _compute_sampler_metrics(
            source,
            ctmc_final_seq,
            target_seq,
            ctmc_traj.n_events,
            ctmc_elapsed,
            ctmc_trajectory,
            self.scoring_fn,
        )

    # ------------------------------------------------------------------
    def _run_proposal(
        self,
        source: MRNARecord,
        target_seq: Optional[str],
        backbone: Any,
        device: torch.device,
    ) -> Dict[str, Any]:
        """Run the ProposalEditor and compute its metrics.

        Complexity: ``O(F * E + S * L^2)``.
        """
        cfg = self.config
        proposal_config = ProposalEditorConfig(
            time_step=cfg.proposal_time_step,
            top_k=cfg.proposal_top_k,
            seed=cfg.seed,
        )
        proposal_editor = ProposalEditor(self.model_fn, proposal_config)

        t0 = time.perf_counter()
        proposal_result = proposal_editor.edit(
            source,
            task_id=cfg.proposal_task_id,
            edit_budget=cfg.proposal_edit_budget,
            backbone=backbone,
            device=cfg.device,
        )
        proposal_elapsed = time.perf_counter() - t0

        proposal_final_seq = proposal_result.seq
        proposal_trajectory = _build_proposal_trajectory(
            source, proposal_result, cfg.proposal_time_step
        )
        # The ProposalEditor does not expose an explicit edit count; use the
        # Levenshtein distance from source to final as a comparable proxy.
        proposal_n_edits = _levenshtein(source.seq, proposal_final_seq)
        return _compute_sampler_metrics(
            source,
            proposal_final_seq,
            target_seq,
            proposal_n_edits,
            proposal_elapsed,
            proposal_trajectory,
            self.scoring_fn,
        )

    # ------------------------------------------------------------------
    def compare_batch(
        self,
        pairs: Sequence[Tuple[MRNARecord, Optional[MRNARecord]]],
        backbone: Any = None,
    ) -> List[Dict[str, Any]]:
        """Run comparison over multiple source/target pairs.

        Parameters
        ----------
        pairs : sequence of (source, target) tuples
            Each tuple is a ``(MRNARecord, Optional[MRNARecord])`` pair.
            ``target`` may be ``None`` for pairs without a target.
        backbone : object, optional
            Frozen backbone passed through to ``model_fn``.

        Returns
        -------
        list of dict
            One comparison dictionary per pair (see :meth:`compare_pair`).
            If a pair raises an exception, its dict contains an ``"error"``
            key instead of sampler metrics, so one failure does not abort the
            whole batch.

        Complexity: ``O(N * (F * K + F * E + S * L^2))`` for ``N`` pairs.
        """
        results: List[Dict[str, Any]] = []
        for source, target in pairs:
            try:
                result = self.compare_pair(source, target, backbone)
            except Exception as exc:  # noqa: BLE001 -- batch resilience
                result = {
                    "source_id": source.transcript_id,
                    "target_id": target.transcript_id if target else None,
                    "error": str(exc),
                    "ctmc": None,
                    "proposal": None,
                    "comparison": None,
                }
            results.append(result)
        return results


# ===========================================================================
# Batch summarisation
# ===========================================================================
def summarize_batch(
    results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate a list of comparison results into a summary.

    Parameters
    ----------
    results : sequence of dict
        Output of :meth:`CTMCComparison.compare_batch`.

    Returns
    -------
    dict
        ``{"n_pairs", "n_success", "n_failed", "mean_ctmc_quality",
        "mean_proposal_quality", "mean_ctmc_time", "mean_proposal_time",
        "mean_ctmc_edits", "mean_proposal_edits",
        "ctmc_higher_quality_count", "ctmc_faster_count",
        "ctmc_fewer_edits_count"}``.

    Complexity: ``O(N)``.
    """
    valid = [r for r in results if r.get("error") is None]
    n_total = len(results)
    n_valid = len(valid)
    n_failed = n_total - n_valid

    if not valid:
        return {
            "n_pairs": n_total,
            "n_success": 0,
            "n_failed": n_failed,
            "mean_ctmc_quality": 0.0,
            "mean_proposal_quality": 0.0,
            "mean_ctmc_time": 0.0,
            "mean_proposal_time": 0.0,
            "mean_ctmc_edits": 0.0,
            "mean_proposal_edits": 0.0,
            "ctmc_higher_quality_count": 0,
            "ctmc_faster_count": 0,
            "ctmc_fewer_edits_count": 0,
        }

    ctmc_q = [r["ctmc"]["quality_score"] for r in valid]
    prop_q = [r["proposal"]["quality_score"] for r in valid]
    ctmc_t = [r["ctmc"]["elapsed_seconds"] for r in valid]
    prop_t = [r["proposal"]["elapsed_seconds"] for r in valid]
    ctmc_e = [r["ctmc"]["n_edits"] for r in valid]
    prop_e = [r["proposal"]["n_edits"] for r in valid]

    return {
        "n_pairs": n_total,
        "n_success": n_valid,
        "n_failed": n_failed,
        "mean_ctmc_quality": float(np.mean(ctmc_q)),
        "mean_proposal_quality": float(np.mean(prop_q)),
        "mean_ctmc_time": float(np.mean(ctmc_t)),
        "mean_proposal_time": float(np.mean(prop_t)),
        "mean_ctmc_edits": float(np.mean(ctmc_e)),
        "mean_proposal_edits": float(np.mean(prop_e)),
        "ctmc_higher_quality_count": sum(
            1 for r in valid if r["comparison"]["ctmc_higher_quality"]
        ),
        "ctmc_faster_count": sum(
            1 for r in valid if r["comparison"]["ctmc_faster"]
        ),
        "ctmc_fewer_edits_count": sum(
            1 for r in valid if r["comparison"]["ctmc_fewer_edits"]
        ),
    }


__all__ = [
    "ComparisonConfig",
    "CTMCComparison",
    "summarize_batch",
]
