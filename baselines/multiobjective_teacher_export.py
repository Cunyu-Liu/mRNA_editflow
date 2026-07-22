"""Multi-objective reward teacher export for mRNA-EditFlow.

This module realizes architecture upgrade #1 from the SOTA roadmap: a
multi-objective reward head. Instead of distilling a single translation
efficiency (TE) delta, it labels each legal one-step 5'UTR edit with a *vector*
of objective deltas and a Pareto-aware scalarization, so the proposal ranker can
learn to balance competing biological objectives rather than optimize TE alone.

Objectives per candidate ``y`` versus source ``x`` (all "higher is better"):

* ``te``     : ``TE_oracle(y) - TE_oracle(x)``
* ``mrl``    : ``MRL_oracle(y) - MRL_oracle(x)`` (scaled to a comparable range)
* ``cai``    : ``CAI(y) - CAI(x)`` over the fixed CDS (constant for UTR-only
  edits, but kept for full-transcript generality and audit completeness)
* ``gc``     : ``-( (GC(y)-g*)^2 - (GC(x)-g*)^2 )`` -- rewards moving GC toward
  the target ``g*``
* ``access`` : ``access(y) - access(x)`` -- start-codon accessibility proxy
* ``uaug``   : ``uAUG(x) - uAUG(y)`` -- rewards removing upstream AUGs

Scalarization / ranking fusion
------------------------------
Given per-objective weights ``w_k`` (summing to 1) and per-objective z-free
min-max normalization within a transcript's candidate pool,

``fuse(y) = sum_k w_k * norm_k(delta_k(y))``.

The normalized per-objective scores are emitted as ``source_scores`` so that
``train_proposal_ranker.py --pair-source-mode source_balanced`` applies a
Bradley-Terry loss independently per objective (a Pareto-front-preserving
formulation: each objective retains its own ordering), while ``teacher_score``
carries the scalarized fusion for global ranking or single-objective control.

Pareto fusion
-------------
Scalarization alone cannot recover the concave regions of a Pareto front, so
this module also computes an explicit **fast non-dominated sort** (the NSGA-II
ranking) over the objective-delta vectors of each transcript's candidate pool.
Candidate ``a`` *dominates* ``b`` when ``a`` is no worse on every objective and
strictly better on at least one. ``pareto_rank`` is the front index (``0`` is
the non-dominated front), ``pareto_front_size`` the size of that front, and
``pareto_fused_score`` in ``[0, 1]`` is a front-major, scalarization-minor score
(``1`` for the best front) so it can drive ranking directly. Selecting
``fusion_mode="pareto"`` makes ``teacher_score`` the Pareto-fused value; the
default ``fusion_mode="scalar"`` keeps the weighted scalarization for backward
compatibility. Both modes always emit the full Pareto metadata for audit.

Legacy standardized fusion
--------------------------
``fusion_mode="grpo_standardized"`` is a retained artifact-compatible name for
per-candidate-pool objective z-score aggregation (``grpo_fused_score``).  It
does **not** perform GRPO: there is no sampled policy rollout, policy-ratio
objective, KL control, or online policy update.  It remains available only for
reproducing prior teacher artifacts; new Stage 3 training consumes the emitted
reward vectors and preference distribution instead of treating this scalar as
an RL objective.

Complexity is ``O(N * P * |alphabet| * Q)`` for ``N`` records, ``P`` selected
5'UTR positions and oracle feature cost ``Q``; normalization is ``O(R)`` per
transcript for ``R`` candidate rows, and the non-dominated sort is ``O(R^2 * K)``
for ``K`` objectives (``R`` is the small per-transcript candidate pool).
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional, Sequence

from mrna_editflow.baselines.utr_local_search import (
    UTRLocalSearchConfig,
    _normalise_utr,
)
from mrna_editflow.baselines.utr_teacher_export import _one_step_candidates
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.download_mrna import load_records_jsonl
from mrna_editflow.data.split_contract import VerifiedSplitContract, load_and_verify_split_manifest
from mrna_editflow.eval.artifact_contract import (
    OracleContractError,
    build_run_metadata,
    load_and_verify_oracle_manifest,
    normalize_run_mode,
    prepare_scientific_records,
    require_paper_cli_inputs,
    validate_output_namespace,
    write_provenance_sidecar,
)
from mrna_editflow.eval.metrics import cai, edit_distance, gc_fraction
from mrna_editflow.eval.oracle import LocalTranslationOracle
from mrna_editflow.rl.reward_diagnostics import reward_correlation_report, write_reward_correlation_report
from mrna_editflow.rl.reward_vector import RewardComponent, RewardVector


OBJECTIVE_LABELS: tuple[str, ...] = ("te", "mrl", "cai", "gc", "access", "uaug")

DEFAULT_OBJECTIVE_WEIGHTS: dict[str, float] = {
    "te": 0.50,
    "mrl": 0.0,
    "cai": 0.0,
    "gc": 0.20,
    "access": 0.15,
    "uaug": 0.15,
}

FUSION_MODES: tuple[str, ...] = ("scalar", "pareto", "grpo_standardized")


@dataclass(frozen=True)
class MultiObjectiveConfig:
    """Configuration for multi-objective teacher export.

    ``gc_target`` is the GC fraction that the ``gc`` objective rewards moving
    toward. ``mrl_scale`` maps the oracle MRL range into a TE-comparable scale
    before normalization. ``weights`` are the scalarization weights; they are
    renormalized to sum to 1. ``candidate_cap`` deterministically keeps ranking
    extremes of the fused score. ``fusion_mode`` selects which fusion drives
    ``teacher_score``: ``"scalar"`` (weighted scalarization, default and
    backward compatible), ``"pareto"`` (fast non-dominated sort ranking) or
    ``"grpo_standardized"`` (legacy per-metric z-scored aggregation; not GRPO).
    """

    gc_target: float = 0.52
    mrl_scale: float = 10.0
    start_window_nt: int = 90
    max_edit_positions: int = 90
    max_length_delta: int = 1
    allow_substitution: bool = True
    allow_insertion: bool = True
    allow_deletion: bool = True
    candidate_cap: int = 0
    fusion_mode: str = "scalar"
    uncertainty_penalty: float = 0.0
    minimum_oracle_agreement: Optional[float] = None
    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_OBJECTIVE_WEIGHTS))

    def __post_init__(self) -> None:
        if self.fusion_mode not in FUSION_MODES:
            raise ValueError(
                f"fusion_mode must be one of {FUSION_MODES}, got {self.fusion_mode!r}"
            )
        if self.uncertainty_penalty < 0.0:
            raise ValueError("uncertainty_penalty must be non-negative")
        if self.minimum_oracle_agreement is not None and not 0.0 <= self.minimum_oracle_agreement <= 1.0:
            raise ValueError("minimum_oracle_agreement must be within [0, 1]")

    def normalized_weights(self) -> dict[str, float]:
        """Return objective weights restricted to known labels and summing to 1."""
        raw = {
            label: max(0.0, float(self.weights.get(label, 0.0)))
            for label in OBJECTIVE_LABELS
        }
        total = sum(raw.values())
        if total <= 0.0:
            return {label: 1.0 / len(OBJECTIVE_LABELS) for label in OBJECTIVE_LABELS}
        return {label: value / total for label, value in raw.items()}

    def to_utr_search_config(self) -> UTRLocalSearchConfig:
        """Build the UTR one-step enumeration config. Complexity ``O(1)``."""
        return UTRLocalSearchConfig(
            edit_budget=1,
            beam_width=1,
            max_length_delta=int(self.max_length_delta),
            start_window_nt=int(self.start_window_nt),
            max_edit_positions=int(self.max_edit_positions),
            gc_target=float(self.gc_target),
            allow_substitution=bool(self.allow_substitution),
            allow_insertion=bool(self.allow_insertion),
            allow_deletion=bool(self.allow_deletion),
        )


@dataclass(frozen=True)
class MultiObjectiveTeacherRow:
    """One legal edit labelled with a multi-objective reward vector."""

    transcript_id: str
    task_id: str
    op: str
    pos: Optional[int]
    nt: str
    teacher_score: float
    objective_deltas: Mapping[str, float]
    source_scores: Mapping[str, float]
    utr_edit_distance: int
    length_delta: int
    pareto_rank: int
    pareto_front_size: int
    pareto_fused_score: float
    scalar_fused_score: float
    grpo_fused_score: float
    reward_vector: Mapping[str, object] = field(default_factory=dict)
    raw_absolute_level: Mapping[str, float] = field(default_factory=dict)
    raw_delta_from_source: Mapping[str, float] = field(default_factory=dict)
    normalized_within_group: Mapping[str, float] = field(default_factory=dict)
    oracle_uncertainty: float = 0.0
    oracle_agreement: float = 1.0
    validity: Mapping[str, bool] = field(default_factory=dict)
    redundancy_groups: Sequence[str] = ()

    def to_dict(self) -> dict[str, object]:
        return dict(asdict(self))


def _transcript_seq(record: MRNARecord) -> str:
    """Full transcript string used for GC/accessibility objectives."""
    return f"{record.five_utr}{record.cds}{record.three_utr}"


def _objective_vector(
    record: MRNARecord,
    oracle: LocalTranslationOracle,
    cfg: MultiObjectiveConfig,
    codon_weights: Optional[Mapping[str, float]],
) -> dict[str, float]:
    """Return raw objective levels (higher is better where applicable).

    ``te``/``mrl``/``access``/``gc_sqerr``/``uaug`` come from the independent
    oracle features; ``cai`` uses the shared CDS codon-adaptation index. All are
    absolute levels; deltas are computed against the source. Complexity is the
    oracle feature cost plus ``O(len(cds))`` for CAI.
    """
    score = oracle.score_utr(record.five_utr, record.cds[:12])
    features = score.get("features", {})
    if not isinstance(features, Mapping):
        features = {}
    te = float(score.get("ensemble_te", score.get("te", 0.0)))
    mrl = float(score.get("ensemble_mrl", score.get("mrl", 0.0)))
    access = float(features.get("start_accessibility", 0.0))
    uaug = float(features.get("uaug_count", 0.0))
    gc_value = gc_fraction(_transcript_seq(record))
    gc_sqerr = (gc_value - float(cfg.gc_target)) ** 2
    cai_value = cai(record.cds, codon_weights)
    return {
        "te": te,
        "mrl": mrl,
        "cai": cai_value,
        "gc_sqerr": gc_sqerr,
        "access": access,
        "uaug": uaug,
        "uncertainty": float(score.get("uncertainty", 0.0)),
        "agreement": float(score.get("agreement", 1.0)),
    }


def _vector_reward(
    *,
    levels: Mapping[str, float],
    source_levels: Mapping[str, float],
    legacy_deltas: Mapping[str, float],
    normalized: Mapping[str, float],
    edit_cost: float,
    task_id: str,
) -> RewardVector:
    """Create the audited Stage 3 vector without trading hard constraints."""
    uncertainty = max(0.0, float(levels.get("uncertainty", 0.0)))
    components = [
        RewardComponent("te", float(levels["te"] - source_levels["te"]), "LocalTranslationOracle", "functional", True, uncertainty, True),
        RewardComponent("mrl", float(levels["mrl"] - source_levels["mrl"]), "LocalTranslationOracle", "functional", False, uncertainty, True),
        RewardComponent("access", float(levels["access"] - source_levels["access"]), "LocalTranslationOracle_shared_features", "structure", True, uncertainty, True),
        RewardComponent("gc_constraint", float(legacy_deltas["gc"]), "gc_interval_penalty", "manufacturability", True, None, True),
        RewardComponent("uaug", float(legacy_deltas["uaug"]), "uaug_counter", "safety", True, None, True),
        RewardComponent("cai", float(levels["cai"] - source_levels["cai"]), "codon_adaptation", "manufacturability", True, None, task_id.upper() == "T4"),
        RewardComponent("edit_cost", -abs(float(edit_cost)), "edit_distance", "edit_cost", True, None, True),
        RewardComponent("uncertainty", -uncertainty, "LocalTranslationOracle", "functional", True, uncertainty, True),
        RewardComponent("protein_identity", 0.0, "hard_action_gate", "hard_constraint", False, None, True),
        RewardComponent("cds_frame", 0.0, "hard_action_gate", "hard_constraint", False, None, True),
        RewardComponent("start_stop_codon", 0.0, "hard_action_gate", "hard_constraint", False, None, True),
        RewardComponent("allowed_editing_region", 0.0, "hard_action_gate", "hard_constraint", False, None, True),
    ]
    absolute = {
        "te": float(levels["te"]), "mrl": float(levels["mrl"]), "access": float(levels["access"]),
        "gc_constraint": -float(levels["gc_sqerr"]), "uaug": -float(levels["uaug"]),
        "cai": float(levels["cai"]), "edit_cost": -abs(float(edit_cost)), "uncertainty": -uncertainty,
    }
    raw_delta = {component.name: float(component.value) for component in components if component.category != "hard_constraint"}
    normalized_vector = {
        "te": float(normalized.get("te", 0.0)), "mrl": float(normalized.get("mrl", 0.0)),
        "access": float(normalized.get("access", 0.0)), "gc_constraint": float(normalized.get("gc", 0.0)),
        "uaug": float(normalized.get("uaug", 0.0)), "cai": float(normalized.get("cai", 0.0)),
    }
    return RewardVector(
        raw_absolute_level=absolute, raw_delta_from_source=raw_delta,
        normalized_within_group=normalized_vector, components=components,
        validity={component.name: component.valid for component in components},
        redundancy_groups=("te_mrl_derived", "te_access_shared_oracle_features"),
    )


def _deltas_from_levels(
    source: Mapping[str, float],
    cand: Mapping[str, float],
    cfg: MultiObjectiveConfig,
) -> dict[str, float]:
    """Signed "higher is better" objective deltas versus the source.

    Complexity is ``O(number_of_objectives)``.
    """
    return {
        "te": cand["te"] - source["te"],
        "mrl": (cand["mrl"] - source["mrl"]) / max(float(cfg.mrl_scale), 1e-8),
        "cai": cand["cai"] - source["cai"],
        # Lower squared GC error is better -> reward the reduction.
        "gc": source["gc_sqerr"] - cand["gc_sqerr"],
        "access": cand["access"] - source["access"],
        # Fewer uAUGs is better -> reward the reduction.
        "uaug": source["uaug"] - cand["uaug"],
    }


def _min_max_normalize(values: Sequence[float]) -> list[float]:
    """Min-max normalize into ``[0, 1]``; constant columns map to ``0.5``.

    Complexity is ``O(len(values))``.
    """
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 1e-12:
        return [0.5 for _ in values]
    return [float((v - lo) / span) for v in values]


def _zscore_standardize(values: Sequence[float]) -> list[float]:
    """Standardize to zero-mean/unit-std; constant columns map to ``0.0``.

    This legacy scalarization helper centers and scales each objective's raw
    delta within a candidate pool before aggregation. It is not a GRPO loss or
    policy-optimization procedure. Complexity is ``O(len(values))``.
    """
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = var ** 0.5
    if std <= 1e-12:
        return [0.0 for _ in values]
    return [float((v - mean) / std) for v in values]


def _dominates(a: Sequence[float], b: Sequence[float], *, eps: float = 1e-12) -> bool:
    """Return ``True`` when ``a`` Pareto-dominates ``b`` (maximization).

    ``a`` dominates ``b`` iff ``a`` is no worse on every objective and strictly
    better on at least one. Complexity is ``O(K)`` for ``K`` objectives.
    """
    no_worse = True
    strictly_better = False
    for av, bv in zip(a, b):
        if av < bv - eps:
            no_worse = False
            break
        if av > bv + eps:
            strictly_better = True
    return no_worse and strictly_better


def fast_non_dominated_sort(points: Sequence[Sequence[float]]) -> list[int]:
    """Assign an NSGA-II front index to each objective vector (maximization).

    Front ``0`` is the non-dominated set; front ``f+1`` is non-dominated once
    all lower fronts are removed. Returns a list of ranks aligned to ``points``.

    Complexity is ``O(R^2 * K)`` for ``R`` points and ``K`` objectives, which is
    cheap for the small per-transcript candidate pool.
    """
    n = len(points)
    if n == 0:
        return []
    domination_count = [0] * n           # how many points dominate i
    dominated: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(points[i], points[j]):
                dominated[i].append(j)
                domination_count[j] += 1
            elif _dominates(points[j], points[i]):
                dominated[j].append(i)
                domination_count[i] += 1
    ranks = [-1] * n
    current = [i for i in range(n) if domination_count[i] == 0]
    front = 0
    while current:
        nxt: list[int] = []
        for i in current:
            ranks[i] = front
            for j in dominated[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    nxt.append(j)
        current = nxt
        front += 1
    # Any residual (should not happen for a finite poset) gets the last front.
    for i in range(n):
        if ranks[i] < 0:
            ranks[i] = front
    return ranks


def score_record_multiobjective_rows(
    record: MRNARecord,
    *,
    config: Optional[MultiObjectiveConfig] = None,
    oracle: Optional[LocalTranslationOracle] = None,
    codon_weights: Optional[Mapping[str, float]] = None,
) -> list[MultiObjectiveTeacherRow]:
    """Return multi-objective teacher rows for one transcript.

    Each row carries per-objective normalized ``source_scores`` and a scalarized
    ``teacher_score``. CDS and 3'UTR are never mutated (UTR-only edits), so all
    protein/frame hard constraints are preserved by construction. Complexity is
    ``O(P * |alphabet| * Q + R)``.
    """
    cfg = config or MultiObjectiveConfig()
    pred = oracle or LocalTranslationOracle()
    search_cfg = cfg.to_utr_search_config()
    source_utr = _normalise_utr(record.five_utr)
    source_record = MRNARecord(
        transcript_id=f"{record.transcript_id}_mo_source",
        five_utr=source_utr,
        cds=record.cds,
        three_utr=record.three_utr,
        species=record.species,
    )
    source_levels = _objective_vector(source_record, pred, cfg, codon_weights)

    raw: list[tuple[str, Optional[int], str, int, int, dict[str, float], dict[str, float]]] = []
    for op, pos, nt, cand in _one_step_candidates(source_record, search_cfg):
        cand_levels = _objective_vector(cand, pred, cfg, codon_weights)
        if (
            cfg.minimum_oracle_agreement is not None
            and cand_levels["agreement"] < float(cfg.minimum_oracle_agreement)
        ):
            continue
        deltas = _deltas_from_levels(source_levels, cand_levels, cfg)
        raw.append(
            (
                op,
                int(pos),
                str(nt),
                int(edit_distance(source_utr, cand.five_utr)),
                int(len(cand.five_utr) - len(source_utr)),
                deltas,
                cand_levels,
            )
        )
    # STOP is a first-class auditable action. Its raw delta is exactly zero;
    # normalized values are still group-relative and never replace that gate.
    raw.append(("stop", None, "", 0, 0, {label: 0.0 for label in OBJECTIVE_LABELS}, dict(source_levels)))

    normalized: dict[str, list[float]] = {}
    for label in OBJECTIVE_LABELS:
        normalized[label] = _min_max_normalize([item[5][label] for item in raw])

    # Legacy per-metric standardization: z-score each objective's raw delta
    # across the candidate pool before scalar aggregation. This is not GRPO.
    grpo_std: dict[str, list[float]] = {}
    for label in OBJECTIVE_LABELS:
        grpo_std[label] = _zscore_standardize([item[5][label] for item in raw])

    # Pareto front assignment over the normalized objective vectors (all
    # "higher is better", so plain maximization dominance applies).
    pareto_labels = ("te", "gc", "access", "uaug")
    objective_points = [
        [normalized[label][idx] for label in pareto_labels] for idx in range(len(raw))
    ]
    pareto_ranks = fast_non_dominated_sort(objective_points)
    n_fronts = (max(pareto_ranks) + 1) if pareto_ranks else 1
    front_sizes: dict[int, int] = {}
    for rank in pareto_ranks:
        front_sizes[rank] = front_sizes.get(rank, 0) + 1

    weights = cfg.normalized_weights()
    rows: list[MultiObjectiveTeacherRow] = []
    for idx, (op, pos, nt, dist, ldelta, deltas, cand_levels) in enumerate(raw):
        source_scores = {
            label: float(normalized[label][idx])
            for label in OBJECTIVE_LABELS
            if label not in {"mrl", "cai"}
        }
        scalar_fused = float(
            sum(weights[label] * normalized[label][idx] for label in OBJECTIVE_LABELS)
        )
        # Legacy standardized fusion: weighted sum of per-metric z-scored
        # deltas. It is a scalar ranking feature, not a policy objective.
        grpo_fused = float(
            sum(weights[label] * grpo_std[label][idx] for label in OBJECTIVE_LABELS)
        )
        rank = int(pareto_ranks[idx])
        # Disjoint front bands in [0, 1): front k occupies
        # [(n_fronts-1-k)/n_fronts, (n_fronts-k)/n_fronts); the scalar score
        # orders candidates strictly within a band and never crosses it. When
        # everything is one front this degenerates to plain scalarization.
        base = float((n_fronts - 1 - rank) / n_fronts)
        pareto_fused = float(base + scalar_fused / float(n_fronts))
        pareto_fused = max(0.0, min(1.0, pareto_fused))
        if cfg.fusion_mode == "pareto":
            teacher_score = pareto_fused
        elif cfg.fusion_mode == "grpo_standardized":
            teacher_score = grpo_fused
        else:
            teacher_score = scalar_fused
        teacher_score -= float(cfg.uncertainty_penalty) * float(cand_levels["uncertainty"])
        reward = _vector_reward(
            levels=cand_levels,
            source_levels=source_levels,
            legacy_deltas=deltas,
            normalized={label: normalized[label][idx] for label in OBJECTIVE_LABELS},
            edit_cost=dist,
            task_id="T5",
        )
        rows.append(
            MultiObjectiveTeacherRow(
                transcript_id=record.transcript_id,
                task_id="T5",
                op=op,
                pos=pos,
                nt=nt,
                teacher_score=teacher_score,
                objective_deltas={label: float(deltas[label]) for label in OBJECTIVE_LABELS},
                source_scores=source_scores,
                utr_edit_distance=dist,
                length_delta=ldelta,
                pareto_rank=rank,
                pareto_front_size=int(front_sizes.get(rank, 1)),
                pareto_fused_score=pareto_fused,
                scalar_fused_score=scalar_fused,
                grpo_fused_score=grpo_fused,
                reward_vector=reward.to_dict(),
                raw_absolute_level=reward.raw_absolute_level,
                raw_delta_from_source=reward.raw_delta_from_source,
                normalized_within_group=reward.normalized_within_group,
                oracle_uncertainty=float(cand_levels["uncertainty"]),
                oracle_agreement=float(cand_levels["agreement"]),
                validity=reward.validity,
                redundancy_groups=reward.redundancy_groups,
            )
        )
    return _cap_rows(rows, int(cfg.candidate_cap))


def _cap_rows(
    rows: Sequence[MultiObjectiveTeacherRow], candidate_cap: int
) -> list[MultiObjectiveTeacherRow]:
    """Deterministically cap rows while preserving fused-score extremes.

    Complexity is ``O(R log R)``.
    """
    cap = int(candidate_cap)
    stops = [row for row in rows if row.op == "stop"]
    ordered = sorted(
        [row for row in rows if row.op != "stop"],
        key=lambda row: (-row.teacher_score, row.op, row.pos if row.pos is not None else -1, row.nt),
    )
    if cap <= 0 or len(ordered) <= cap:
        return ordered + stops
    selected: list[MultiObjectiveTeacherRow] = []
    lo = 0
    hi = len(ordered) - 1
    while len(selected) < cap and lo <= hi:
        selected.append(ordered[lo])
        lo += 1
        if len(selected) >= cap or lo > hi:
            break
        selected.append(ordered[hi])
        hi -= 1
    return selected + stops


def summarize_multiobjective_rows(
    rows_by_record: Sequence[Sequence[MultiObjectiveTeacherRow]],
) -> dict[str, object]:
    """Summarize a grouped multi-objective teacher export.

    Complexity is ``O(total_rows)``.
    """
    groups = [list(rows) for rows in rows_by_record]
    non_empty = [rows for rows in groups if rows]
    all_rows = [row for rows in non_empty for row in rows]
    summary: dict[str, object] = {
        "n_records": len(groups),
        "n_records_with_rows": len(non_empty),
        "n_rows": len(all_rows),
        "objectives": list(OBJECTIVE_LABELS),
    }
    if not all_rows:
        summary["mean_rows_per_record"] = 0.0
        return summary
    summary["mean_rows_per_record"] = float(len(all_rows) / max(1, len(non_empty)))
    for label in OBJECTIVE_LABELS:
        deltas = [float(row.objective_deltas[label]) for row in all_rows]
        summary[f"mean_delta_{label}"] = float(sum(deltas) / len(deltas))
        summary[f"max_delta_{label}"] = float(max(deltas))
        summary[f"min_delta_{label}"] = float(min(deltas))
    summary["mean_fused_teacher_score"] = float(
        sum(row.teacher_score for row in all_rows) / len(all_rows)
    )
    summary["sub_rows"] = int(sum(row.op == "sub" for row in all_rows))
    summary["ins_rows"] = int(sum(row.op == "ins" for row in all_rows))
    summary["del_rows"] = int(sum(row.op == "del" for row in all_rows))
    # Pareto-front statistics (front 0 is the non-dominated set per transcript).
    front0_rows = [row for row in all_rows if row.pareto_rank == 0]
    summary["pareto_front0_rows"] = int(len(front0_rows))
    summary["pareto_front0_fraction"] = float(len(front0_rows) / len(all_rows))
    summary["mean_pareto_rank"] = float(
        sum(row.pareto_rank for row in all_rows) / len(all_rows)
    )
    summary["max_pareto_rank"] = int(max(row.pareto_rank for row in all_rows))
    summary["mean_pareto_fused_score"] = float(
        sum(row.pareto_fused_score for row in all_rows) / len(all_rows)
    )
    summary["mean_grpo_fused_score"] = float(
        sum(row.grpo_fused_score for row in all_rows) / len(all_rows)
    )
    return summary


def export_multiobjective_teacher_jsonl(
    records: Sequence[MRNARecord],
    *,
    out_jsonl: str,
    out_json: str,
    limit: Optional[int] = None,
    config: Optional[MultiObjectiveConfig] = None,
    reference_records: Optional[Sequence[MRNARecord]] = None,
    run_mode: str = "development",
    split_contract: Optional[VerifiedSplitContract] = None,
    split_role: Optional[str] = None,
    oracle_manifest: Optional[str] = None,
    reward_correlation_json: Optional[str] = None,
    reward_correlation_markdown: Optional[str] = None,
) -> dict[str, object]:
    """Write multi-objective teacher rows and a summary JSON artifact.

    ``out_jsonl`` rows are consumable by ``train_proposal_ranker.py`` in both
    ``global`` (uses fused ``teacher_score``) and ``source_balanced`` (uses the
    per-objective ``source_scores``) modes. ``reference_records`` optionally
    define the CAI codon weights; when omitted, the shared default table is used.
    Complexity is ``O(N * P * |alphabet| * Q + rows_written)``.
    """
    cfg = config or MultiObjectiveConfig()
    run_mode = normalize_run_mode(run_mode)
    role_records, data_provenance = prepare_scientific_records(
        records,
        run_mode=run_mode,
        split_contract=split_contract,
        split_role=split_role,
        allowed_roles=("train",),
    )
    oracle_metadata = load_and_verify_oracle_manifest(oracle_manifest, run_mode=run_mode)
    if run_mode == "paper":
        raise OracleContractError(
            "paper teacher export has no independent-oracle execution adapter in this goal"
        )
    validate_output_namespace(out_jsonl, run_mode)
    validate_output_namespace(out_json, run_mode)
    selected = list(role_records[: int(limit)]) if limit is not None else list(role_records)
    scientific_validity = build_run_metadata(
        run_mode=run_mode,
        data_provenance=data_provenance,
        config=asdict(cfg),
        code_paths=(__file__,),
        oracle=oracle_metadata,
        extra_block_reasons=("heuristic_functional_oracle",),
        functional_claim=True,
    )
    oracle = LocalTranslationOracle()
    codon_weights: Optional[Mapping[str, float]] = None
    if reference_records is not None:
        from mrna_editflow.eval.metrics import codon_weights_from_reference

        codon_weights = codon_weights_from_reference(list(reference_records))
    grouped_rows = [
        score_record_multiobjective_rows(
            record,
            config=cfg,
            oracle=oracle,
            codon_weights=codon_weights,
        )
        for record in selected
    ]
    reward_vectors = [
        RewardVector.from_dict(row.reward_vector)
        for rows in grouped_rows for row in rows
        if row.reward_vector
    ]
    correlation = reward_correlation_report(reward_vectors)
    if reward_correlation_json or reward_correlation_markdown:
        if not reward_correlation_json or not reward_correlation_markdown:
            raise ValueError("reward correlation JSON and Markdown paths must be supplied together")
        write_reward_correlation_report(
            correlation, reward_correlation_json, reward_correlation_markdown
        )
    os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)), exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for rows in grouped_rows:
            for row in rows:
                fh.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")
    payload = {
        "config": {
            "gc_target": float(cfg.gc_target),
            "mrl_scale": float(cfg.mrl_scale),
            "start_window_nt": int(cfg.start_window_nt),
            "max_edit_positions": int(cfg.max_edit_positions),
            "max_length_delta": int(cfg.max_length_delta),
            "candidate_cap": int(cfg.candidate_cap),
            "fusion_mode": str(cfg.fusion_mode),
            "uncertainty_penalty": float(cfg.uncertainty_penalty),
            "minimum_oracle_agreement": cfg.minimum_oracle_agreement,
            "weights": cfg.normalized_weights(),
        },
        "n_records": len(selected),
        "out_jsonl": out_jsonl,
        "uses_reference_codon_weights": reference_records is not None,
        "summary": summarize_multiobjective_rows(grouped_rows),
        "reward_schema_version": 1,
        "reward_correlation_report": correlation,
        "reward_correlation_json": reward_correlation_json,
        "reward_correlation_markdown": reward_correlation_markdown,
        "scientific_validity": scientific_validity,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    payload["provenance_sidecar"] = write_provenance_sidecar(out_jsonl, scientific_validity)
    payload["summary_provenance_sidecar"] = write_provenance_sidecar(
        out_json, scientific_validity
    )
    return payload


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reference-records-jsonl", default=None)
    parser.add_argument("--candidate-cap", type=int, default=256)
    parser.add_argument("--gc-target", type=float, default=0.52)
    parser.add_argument("--mrl-scale", type=float, default=10.0)
    parser.add_argument("--uncertainty-penalty", type=float, default=0.0)
    parser.add_argument("--minimum-oracle-agreement", type=float, default=None)
    parser.add_argument("--reward-correlation-json", default="docs/reward_correlation_report.json")
    parser.add_argument("--reward-correlation-markdown", default="docs/reward_correlation_report.md")
    parser.add_argument("--start-window-nt", type=int, default=90)
    parser.add_argument("--max-edit-positions", type=int, default=90)
    parser.add_argument("--max-length-delta", type=int, default=1)
    parser.add_argument(
        "--fusion-mode",
        choices=list(FUSION_MODES),
        default="scalar",
        help="teacher_score fusion; grpo_standardized is a legacy z-score scalarization, not GRPO",
    )
    parser.add_argument("--w-te", type=float, default=DEFAULT_OBJECTIVE_WEIGHTS["te"])
    parser.add_argument("--w-mrl", type=float, default=DEFAULT_OBJECTIVE_WEIGHTS["mrl"])
    parser.add_argument("--w-cai", type=float, default=DEFAULT_OBJECTIVE_WEIGHTS["cai"])
    parser.add_argument("--w-gc", type=float, default=DEFAULT_OBJECTIVE_WEIGHTS["gc"])
    parser.add_argument("--w-access", type=float, default=DEFAULT_OBJECTIVE_WEIGHTS["access"])
    parser.add_argument("--w-uaug", type=float, default=DEFAULT_OBJECTIVE_WEIGHTS["uaug"])
    parser.add_argument("--run-mode", choices=("development", "paper"), default="development")
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--split-role", choices=("train", "val", "test"), default=None)
    parser.add_argument("--oracle-manifest", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    require_paper_cli_inputs(
        run_mode=args.run_mode,
        split_manifest=args.split_manifest,
        split_role=args.split_role,
        allowed_roles=("train",),
        oracle_manifest=args.oracle_manifest,
        require_oracle=True,
    )
    config = MultiObjectiveConfig(
        gc_target=args.gc_target,
        mrl_scale=args.mrl_scale,
        start_window_nt=args.start_window_nt,
        max_edit_positions=args.max_edit_positions,
        max_length_delta=args.max_length_delta,
        candidate_cap=args.candidate_cap,
        fusion_mode=args.fusion_mode,
        uncertainty_penalty=args.uncertainty_penalty,
        minimum_oracle_agreement=args.minimum_oracle_agreement,
        weights={
            "te": args.w_te,
            "mrl": args.w_mrl,
            "cai": args.w_cai,
            "gc": args.w_gc,
            "access": args.w_access,
            "uaug": args.w_uaug,
        },
    )
    reference = (
        load_records_jsonl(args.reference_records_jsonl)
        if args.reference_records_jsonl
        else None
    )
    payload = export_multiobjective_teacher_jsonl(
        load_records_jsonl(args.records_jsonl),
        out_jsonl=args.out_jsonl,
        out_json=args.out_json,
        limit=args.limit,
        config=config,
        reference_records=reference,
        run_mode=args.run_mode,
        split_contract=(
            load_and_verify_split_manifest(args.split_manifest, records_path=args.records_jsonl)
            if args.split_manifest else None
        ),
        split_role=args.split_role,
        oracle_manifest=args.oracle_manifest,
        reward_correlation_json=args.reward_correlation_json,
        reward_correlation_markdown=args.reward_correlation_markdown,
    )
    print(
        json.dumps(
            {"out_json": args.out_json, "out_jsonl": args.out_jsonl, "summary": payload["summary"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OBJECTIVE_LABELS",
    "DEFAULT_OBJECTIVE_WEIGHTS",
    "FUSION_MODES",
    "MultiObjectiveConfig",
    "MultiObjectiveTeacherRow",
    "fast_non_dominated_sort",
    "score_record_multiobjective_rows",
    "summarize_multiobjective_rows",
    "export_multiobjective_teacher_jsonl",
    "main",
]
