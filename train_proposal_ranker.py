"""TE-aware proposal ranking distillation for mRNA-EditFlow.

Stage A learns the Edit-Flow CTMC field from sequence couplings, while the
oracle-guided ablation shows that some legal local edits improve the TE proxy
but are not always ranked highly by the model. This training entry point turns
``eval.proposal_ranking`` JSONL artifacts into a direct ranking objective.

Mathematical objective
----------------------
For a source record ``x`` and two legal proposals ``i,j`` from the same source,
the teacher signal is

``y_i = TE(candidate_i) - TE(x)``.

The student score is the log CTMC operation intensity at ``t=0.5``:

``s_i = log(lambda_op(pos_i | x) p_op(nt_i | pos_i, x))``.

For pairs with ``|y_i-y_j| >= eps`` we minimise

``L_ij = w_ij softplus(- sign(y_i-y_j) (s_i-s_j) / tau)``,

where ``w_ij = max(|y_i-y_j|, min_pair_weight)``. This is a Bradley-Terry style
pairwise distillation loss: it does not require absolute TE calibration, only
the ordering of legal neighbours. Complexity per optimisation step is
``O(B * model_forward + B * P)`` for ``B`` source records and at most ``P``
pairs per source.

Hybrid teacher artifacts may include source-specific scores such as
``source_scores={"full": y_full, "utr": y_utr}``. With
``pair_source_mode="source_balanced"``, the same Bradley-Terry loss is applied
separately inside each source score, splitting the pair budget across sources:

``L = mean_s mean_(i,j in P_s) BT(y_s(i)-y_s(j), s_i-s_j)``.

This prevents full-pool precision and UTR-search recall from collapsing into a
single global ordering when they encode complementary scientific objectives.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.download_mrna import load_records_jsonl
from mrna_editflow.data.split_contract import (
    VerifiedSplitContract,
    assert_train_validation_disjoint,
    build_split_provenance,
    load_and_verify_split_manifest,
    sha256_file,
    verify_supplied_role_records,
)
from mrna_editflow.eval.artifact_contract import (
    ArtifactProvenanceError,
    build_run_metadata,
    load_artifact_provenance,
    normalize_run_mode,
    prepare_scientific_records,
    validate_output_namespace,
    verify_paper_artifact,
    verify_paper_checkpoint,
    verify_provenance_compatibility,
)
from mrna_editflow.sample import _record_tensors, load_stage_a_checkpoint
from mrna_editflow.rl.action_scoring import operation_log_score
from mrna_editflow.rl.preference_conditioning import (
    PreferenceConditionedObjectiveHead,
    normalized_preference,
    sample_dirichlet_preference,
)
from mrna_editflow.rl.reward_vector import RewardVector
from mrna_editflow.rl.validation import evaluate_ranker_validation
from mrna_editflow.train_backbone import _ensure_parent, _resolve_device, _set_seed, _write_profile


@dataclass(frozen=True)
class ProposalTeacherRow:
    """One teacher-labelled legal edit proposal."""

    transcript_id: str
    op: str
    pos: Optional[int]
    nt: str
    teacher_score: float
    source_scores: Mapping[str, float]
    task_id: str = "T5"
    reward_vector: Mapping[str, object] = field(default_factory=dict)
    raw_absolute_level: Mapping[str, float] = field(default_factory=dict)
    raw_delta_from_source: Mapping[str, float] = field(default_factory=dict)
    normalized_within_group: Mapping[str, float] = field(default_factory=dict)
    oracle_uncertainty: Optional[float] = None
    oracle_agreement: Optional[float] = None
    validity: Mapping[str, bool] = field(default_factory=dict)

    @classmethod
    def from_json(cls, row: Mapping[str, object]) -> "ProposalTeacherRow":
        raw_sources = row.get("source_scores", {})
        source_scores: dict[str, float] = {}
        if isinstance(raw_sources, Mapping):
            for key, value in raw_sources.items():
                try:
                    score = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(score):
                    source_scores[str(key)] = score
        if not source_scores:
            source_scores = {"teacher": float(row["teacher_score"])}
        return cls(
            transcript_id=str(row["transcript_id"]),
            task_id=str(row.get("task_id", "T5")).upper(),
            op=str(row["op"]),
            pos=(None if row.get("pos") is None else int(row["pos"])),
            nt=str(row.get("nt", "")),
            teacher_score=float(row["teacher_score"]),
            source_scores=source_scores,
            reward_vector=(dict(row["reward_vector"]) if isinstance(row.get("reward_vector"), Mapping) else {}),
            raw_absolute_level={
                str(k): float(v)
                for k, v in (row.get("raw_absolute_level") if isinstance(row.get("raw_absolute_level"), Mapping) else {}).items()
            },
            raw_delta_from_source={
                str(k): float(v)
                for k, v in (row.get("raw_delta_from_source") if isinstance(row.get("raw_delta_from_source"), Mapping) else row.get("objective_deltas") if isinstance(row.get("objective_deltas"), Mapping) else {}).items()
            },
            normalized_within_group={
                str(k): float(v)
                for k, v in (row.get("normalized_within_group") if isinstance(row.get("normalized_within_group"), Mapping) else row.get("source_scores") if isinstance(row.get("source_scores"), Mapping) else {}).items()
            },
            oracle_uncertainty=(None if row.get("oracle_uncertainty") is None else float(row["oracle_uncertainty"])),
            oracle_agreement=(None if row.get("oracle_agreement") is None else float(row["oracle_agreement"])),
            validity={
                str(k): bool(v)
                for k, v in (row.get("validity") if isinstance(row.get("validity"), Mapping) else {}).items()
            },
        )


@dataclass(frozen=True)
class PairSpec:
    """One pairwise teacher comparison selected for distillation."""

    i: int
    j: int
    teacher_delta: float
    source_label: str


def load_teacher_rows(path: str) -> dict[str, list[ProposalTeacherRow]]:
    """Load proposal-ranking JSONL grouped by source transcript id."""
    grouped: dict[str, list[ProposalTeacherRow]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = ProposalTeacherRow.from_json(json.loads(line))
            if math.isfinite(row.teacher_score):
                grouped.setdefault(row.transcript_id, []).append(row)
    return grouped


def _record_map(records: Sequence[MRNARecord]) -> dict[str, MRNARecord]:
    out: dict[str, MRNARecord] = {}
    for rec in records:
        out.setdefault(rec.transcript_id, rec)
    return out


def _forward_record(record: MRNARecord, model, backbone, device: torch.device) -> Mapping[str, torch.Tensor]:
    token_ids, region_ids, phase_ids, padding_mask = _record_tensors(record, device)
    t = torch.full((1, 1), 0.5, dtype=torch.float32, device=device)
    return model.forward(token_ids, region_ids, phase_ids, t, padding_mask, backbone)


def _operation_log_score(out: Mapping[str, torch.Tensor], row: ProposalTeacherRow) -> torch.Tensor:
    """Compatibility wrapper delegating to the shared differentiable scorer."""
    return operation_log_score(out, row.op, row.pos, row.nt or None)


def _row_reward_vector(row: ProposalTeacherRow) -> RewardVector:
    if row.reward_vector:
        return RewardVector.from_dict(row.reward_vector)
    return RewardVector.from_legacy(row.raw_delta_from_source or {"teacher": row.teacher_score}, row.source_scores)


def _objective_schema(rows: Sequence[ProposalTeacherRow]) -> tuple[str, ...]:
    names: set[str] = set()
    for row in rows:
        if not row.reward_vector:
            # Old scalar/source-balanced teachers retain their established
            # Bradley-Terry path rather than being silently reinterpreted.
            continue
        vector = _row_reward_vector(row)
        for component in vector.components:
            if component.valid and component.category != "hard_constraint" and component.independent:
                names.add(component.name)
    return tuple(sorted(names))


def _operation_id(op: str) -> int:
    return {"ins": 0, "sub": 1, "del": 2, "stop": 0}.get(str(op).lower(), 0)


def score_objectives(
    out: Mapping[str, torch.Tensor],
    rows: Sequence[ProposalTeacherRow],
    objective_head: PreferenceConditionedObjectiveHead,
) -> dict[str, torch.Tensor]:
    """Return independently parameterized objective scores for legal actions."""
    base = torch.stack([_operation_log_score(out, row) for row in rows])
    operation_ids = torch.tensor(
        [_operation_id(row.op) for row in rows], dtype=torch.long, device=base.device
    )
    return objective_head(base, operation_ids)


def load_objective_head_from_checkpoint(
    checkpoint_path: str, *, device: Optional[str] = None
) -> tuple[PreferenceConditionedObjectiveHead, dict[str, object]]:
    """Load vector reward heads for preference-conditioned inference."""
    payload = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must contain a mapping payload")
    schema = payload.get("objective_schema")
    state = payload.get("objective_head_state")
    if not isinstance(schema, Mapping) or not isinstance(state, Mapping):
        raise ValueError("checkpoint does not contain a Stage 3 objective head")
    names = schema.get("objective_names")
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)) or not names:
        raise ValueError("objective checkpoint schema has no objective_names")
    head = PreferenceConditionedObjectiveHead([str(name) for name in names])
    head.load_state_dict(state)  # type: ignore[arg-type]
    head.to(device or "cpu").eval()
    return head, dict(schema)


def _pair_indices_from_scores(
    scores: Sequence[float],
    *,
    max_pairs: int,
    min_teacher_delta: float,
) -> list[tuple[int, int]]:
    """Select deterministic high-vs-low candidate pairs from scalar scores.

    When the full pair set is small, all informative pairs are used. For large
    pools, we pair high-TE and low-TE extremes first, which maximises ranking
    signal under a fixed pair budget.

    Complexity is ``O(N^2)`` for small pools and ``O(N log N + max_pairs)`` for
    large pools.
    """
    n = len(scores)
    if n < 2:
        return []
    cap = max(1, int(max_pairs))
    total = n * (n - 1) // 2
    out: list[tuple[int, int]] = []
    if total <= cap:
        for i in range(n):
            for j in range(i + 1, n):
                if abs(float(scores[i]) - float(scores[j])) >= min_teacher_delta:
                    out.append((i, j))
        return out

    order = sorted(range(n), key=lambda idx: (-float(scores[idx]), idx))
    seen: set[tuple[int, int]] = set()
    hi = 0
    lo = n - 1
    while len(out) < cap and hi < lo:
        i, j = order[hi], order[lo]
        key = (min(i, j), max(i, j))
        if key not in seen and abs(float(scores[i]) - float(scores[j])) >= min_teacher_delta:
            out.append((i, j))
            seen.add(key)
        hi += 1
        if hi >= lo:
            hi = 0
            lo -= 1
    return out


def _pair_indices(
    rows: Sequence[ProposalTeacherRow],
    *,
    max_pairs: int,
    min_teacher_delta: float,
) -> list[tuple[int, int]]:
    """Backward-compatible global pair selection by ``teacher_score``."""
    return _pair_indices_from_scores(
        [row.teacher_score for row in rows],
        max_pairs=max_pairs,
        min_teacher_delta=min_teacher_delta,
    )


def _source_labels(rows: Sequence[ProposalTeacherRow]) -> list[str]:
    """Return sorted source labels present in teacher rows.

    Complexity is ``O(total_source_labels)``.
    """
    labels: set[str] = set()
    for row in rows:
        labels.update(str(label) for label in row.source_scores)
    return sorted(labels)


def _pair_specs(
    rows: Sequence[ProposalTeacherRow],
    *,
    max_pairs: int,
    min_teacher_delta: float,
    pair_source_mode: str,
) -> list[PairSpec]:
    """Select pairwise comparisons for global or source-aware training.

    ``global`` mode recovers the original behavior. ``source_balanced`` mode
    splits the pair budget across labels in ``row.source_scores`` and computes
    teacher deltas from the source-specific score. Complexity is
    ``O(S * (N log N + P_s))`` for ``S`` labels and source pair budgets ``P_s``.
    """
    mode = str(pair_source_mode)
    if mode == "global":
        return [
            PairSpec(i=i, j=j, teacher_delta=float(rows[i].teacher_score - rows[j].teacher_score), source_label="global")
            for i, j in _pair_indices(
                rows,
                max_pairs=max_pairs,
                min_teacher_delta=min_teacher_delta,
            )
        ]
    if mode != "source_balanced":
        raise ValueError(f"unsupported pair_source_mode {pair_source_mode!r}")

    labels = _source_labels(rows)
    if not labels:
        return []
    total_budget = max(1, int(max_pairs))
    base_budget = max(1, total_budget // len(labels))
    remainder = max(0, total_budget - base_budget * len(labels))
    specs: list[PairSpec] = []
    for label_idx, label in enumerate(labels):
        indexed = [
            (idx, float(row.source_scores[label]))
            for idx, row in enumerate(rows)
            if label in row.source_scores and math.isfinite(float(row.source_scores[label]))
        ]
        if len(indexed) < 2:
            continue
        local_indices = [item[0] for item in indexed]
        local_scores = [item[1] for item in indexed]
        budget = base_budget + (1 if label_idx < remainder else 0)
        for li, lj in _pair_indices_from_scores(
            local_scores,
            max_pairs=budget,
            min_teacher_delta=min_teacher_delta,
        ):
            i = local_indices[li]
            j = local_indices[lj]
            specs.append(
                PairSpec(
                    i=i,
                    j=j,
                    teacher_delta=float(rows[i].source_scores[label] - rows[j].source_scores[label]),
                    source_label=label,
                )
            )
    return specs


def _usable_transcripts(
    records_by_id: Mapping[str, MRNARecord],
    teachers_by_id: Mapping[str, Sequence[ProposalTeacherRow]],
    *,
    max_pairs_per_record: int,
    min_teacher_delta: float,
    pair_source_mode: str = "global",
) -> list[str]:
    tids: list[str] = []
    for tid, rows in teachers_by_id.items():
        if tid not in records_by_id or len(rows) < 2:
            continue
        if _pair_specs(
            rows,
            max_pairs=max_pairs_per_record,
            min_teacher_delta=min_teacher_delta,
            pair_source_mode=pair_source_mode,
        ):
            tids.append(tid)
    return sorted(tids)


def _save_ranker_checkpoint(
    path: str,
    *,
    cfg,
    backbone,
    model,
    base_checkpoint: str,
    teacher_jsonl: str,
    val_teacher_jsonl: str,
    step: int,
    best_loss: float,
    best_validation_metric: float,
    best_validation_step: int,
    validation_summary: Mapping[str, object],
    training_loss_at_best_step: float,
    trained_action_space: Mapping[str, object],
    checkpoint_metric: str,
    checkpoint_mode: str,
    objective_head: Optional[PreferenceConditionedObjectiveHead] = None,
    objective_schema: Optional[Mapping[str, object]] = None,
    preference_training: Optional[Mapping[str, object]] = None,
    early_stopping_reason: Optional[str] = None,
    scientific_validity: Optional[Mapping[str, object]] = None,
) -> None:
    _ensure_parent(path)
    action_space = dict(trained_action_space)
    torch.save(
        {
            "stage": "proposal_ranker",
            "step": int(step),
            "best_loss": float(best_loss),
            "base_checkpoint": base_checkpoint,
            "teacher_jsonl": teacher_jsonl,
            "val_teacher_jsonl": val_teacher_jsonl,
            "checkpoint_metric": checkpoint_metric,
            "checkpoint_mode": checkpoint_mode,
            "best_validation_metric": float(best_validation_metric),
            "best_validation_step": int(best_validation_step),
            "validation_summary": dict(validation_summary),
            "training_loss_at_best_step": float(training_loss_at_best_step),
            "early_stopping_reason": early_stopping_reason,
            "objective_head_state": (
                objective_head.state_dict() if objective_head is not None else None
            ),
            "objective_schema": dict(objective_schema or {}),
            "preference_training": dict(preference_training or {}),
            # Keep the nested object as the canonical extensible form, while
            # mirroring the three required fields at top level for tooling
            # that reads checkpoint metadata without knowing this wrapper.
            "trained_action_space": action_space,
            "trained_task": action_space.get("trained_task"),
            "trained_editable_regions": action_space.get("trained_editable_regions"),
            "trained_operations": action_space.get("trained_operations"),
            "config": asdict(cfg),
            "backbone_state": backbone.state_dict(),
            "model_state": model.state_dict(),
            "scientific_validity": dict(scientific_validity or {}),
        },
        path,
    )


def _partition_provenance(
    records: Sequence[MRNARecord],
    *,
    run_mode: str,
    split_contract: Optional[VerifiedSplitContract],
    role: str,
) -> tuple[list[MRNARecord], dict[str, object]]:
    """Verify role-specific inputs while retaining development compatibility."""
    if split_contract is not None:
        selected = verify_supplied_role_records(records, split_contract, role)
        provenance = build_split_provenance(split_contract, role)
    elif run_mode == "paper":
        raise ValueError("paper proposal-ranker training requires a split contract")
    else:
        selected = list(records)
        provenance = {
            "split_role": role,
            "records_count": len(selected),
            "selected_transcript_id_digest": None,
            "block_reasons": ["verified_split_manifest_missing"],
        }
    provenance = dict(provenance)
    provenance.update(
        {
            "run_mode": run_mode,
            "claim_tier": "paper" if run_mode == "paper" else "development_only",
            "paper_eligible": run_mode == "paper",
        }
    )
    return selected, provenance


def _require_teacher_alignment(
    records: Sequence[MRNARecord],
    teachers: Mapping[str, Sequence[ProposalTeacherRow]],
    *,
    partition: str,
) -> None:
    record_ids = {record.transcript_id for record in records}
    teacher_ids = set(teachers)
    missing = record_ids - teacher_ids
    extra = teacher_ids - record_ids
    if missing or extra:
        raise ValueError(
            f"{partition} teacher/record transcript mismatch; "
            f"missing_teacher={sorted(missing)[:1]}, unknown_teacher={sorted(extra)[:1]}"
        )


def _checkpoint_metric_key(metric: str, *, candidate_cap: int) -> str:
    allowed = {
        "val_mean_model_regret": "mean_model_regret",
        "val_oracle_best_recall_at_32": "oracle_best_recall_at_32",
        "val_ndcg_at_32": "ndcg_at_32",
    }
    if metric not in allowed:
        raise ValueError("checkpoint_metric must be one of " + ", ".join(sorted(allowed)))
    if int(candidate_cap) > 0:
        raise ValueError(
            "global checkpoint metrics require validation_candidate_cap=0; "
            "capped validation is restricted and cannot select a global-regret checkpoint"
        )
    return allowed[metric]


def _metric_improved(value: float, best: Optional[float], *, mode: str, minimum_improvement: float) -> bool:
    if best is None:
        return True
    if mode == "min":
        return value < best - float(minimum_improvement)
    return value > best + float(minimum_improvement)


def _annotate_checkpoint_termination(path: str, reason: str) -> None:
    """Add the final validation-only termination reason without replacing weights."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("proposal ranker checkpoint payload is malformed")
    updated = dict(payload)
    updated["early_stopping_reason"] = reason
    torch.save(updated, path)


def train_proposal_ranker(
    *,
    base_checkpoint: str,
    save_dir: str,
    profile_path: str,
    records: Optional[Sequence[MRNARecord]] = None,
    teacher_jsonl: Optional[str] = None,
    train_records: Optional[Sequence[MRNARecord]] = None,
    val_records: Optional[Sequence[MRNARecord]] = None,
    train_teacher_jsonl: Optional[str] = None,
    val_teacher_jsonl: Optional[str] = None,
    teacher_summary: Optional[str] = None,
    steps: int = 200,
    batch_records: int = 4,
    max_pairs_per_record: int = 32,
    lr: float = 2e-5,
    weight_decay: float = 0.0,
    device: Optional[str] = None,
    seed: int = 0,
    min_teacher_delta: float = 1e-6,
    temperature: float = 1.0,
    min_pair_weight: float = 0.01,
    pair_source_mode: str = "global",
    run_mode: str = "development",
    split_contract: Optional[VerifiedSplitContract] = None,
    split_role: Optional[str] = None,
    trained_task: Optional[str] = None,
    trained_editable_regions: Optional[Sequence[str]] = None,
    trained_operations: Optional[Sequence[str]] = None,
    checkpoint_metric: str = "val_mean_model_regret",
    checkpoint_mode: str = "min",
    validation_interval: int = 25,
    early_stopping_patience: int = 0,
    minimum_improvement: float = 0.0,
    validation_candidate_cap: int = 0,
    preference_alpha: Optional[Mapping[str, float]] = None,
    uncertainty_penalty: float = 0.0,
    minimum_oracle_agreement: Optional[float] = None,
) -> dict[str, object]:
    """Fine-tune only on train rows and select checkpoints only on validation."""
    run_mode = normalize_run_mode(run_mode)
    if checkpoint_mode not in {"min", "max"}:
        raise ValueError("checkpoint_mode must be 'min' or 'max'")
    if int(validation_interval) <= 0:
        raise ValueError("validation_interval must be positive")
    if float(minimum_improvement) < 0.0:
        raise ValueError("minimum_improvement must be non-negative")
    metric_key = _checkpoint_metric_key(
        checkpoint_metric, candidate_cap=validation_candidate_cap
    )
    resolved_train_records = list(train_records if train_records is not None else (records or ()))
    resolved_val_records = list(val_records or ())
    train_teacher_path = train_teacher_jsonl or teacher_jsonl
    # Preserve the established paper-mode provenance preflight for callers
    # migrating from the Stage 1 single-partition API.  It intentionally runs
    # before the Stage 2 validation-input error, so a mismatched teacher is
    # never obscured by an interface migration message.
    if (
        run_mode == "paper"
        and resolved_train_records
        and train_teacher_path
        and split_contract is not None
        and (not resolved_val_records or not val_teacher_jsonl)
    ):
        _legacy_train_records, legacy_provenance = prepare_scientific_records(
            resolved_train_records,
            run_mode=run_mode,
            split_contract=split_contract,
            split_role="train",
            allowed_roles=("train",),
        )
        verify_paper_checkpoint(base_checkpoint, legacy_provenance)
        legacy_teacher_provenance = verify_paper_artifact(train_teacher_path)
        verify_provenance_compatibility(
            legacy_provenance,
            legacy_teacher_provenance,
            require_same_role=True,
        )
    if not resolved_train_records or not resolved_val_records:
        raise ValueError("independent train_records and val_records are required")
    if not train_teacher_path or not val_teacher_jsonl:
        raise ValueError("independent train_teacher_jsonl and val_teacher_jsonl are required")
    _set_seed(int(seed))
    validate_output_namespace(save_dir, run_mode)
    validate_output_namespace(profile_path, run_mode)
    dev = _resolve_device(device)
    if split_role not in (None, "train"):
        raise ValueError("split_role is legacy-only; Stage 2 uses train and val roles")
    train_records_verified, data_provenance = _partition_provenance(
        resolved_train_records, run_mode=run_mode, split_contract=split_contract, role="train"
    )
    val_records_verified, val_data_provenance = _partition_provenance(
        resolved_val_records, run_mode=run_mode, split_contract=split_contract, role="val"
    )
    assert_train_validation_disjoint(
        train_records_verified,
        val_records_verified,
        split_contract=split_contract,
    )
    teacher_provenance: Optional[dict[str, object]] = None
    val_teacher_provenance: Optional[dict[str, object]] = None
    try:
        teacher_provenance = load_artifact_provenance(train_teacher_path)
    except ArtifactProvenanceError:
        if run_mode == "paper":
            raise
    try:
        val_teacher_provenance = load_artifact_provenance(val_teacher_jsonl)
    except ArtifactProvenanceError:
        if run_mode == "paper":
            raise
    if run_mode == "paper":
        verify_paper_checkpoint(base_checkpoint, data_provenance)
        teacher_provenance = verify_paper_artifact(train_teacher_path)
        verify_provenance_compatibility(
            data_provenance, teacher_provenance, require_same_role=True
        )
        val_teacher_provenance = verify_paper_artifact(val_teacher_jsonl)
        verify_provenance_compatibility(
            val_data_provenance, val_teacher_provenance, require_same_role=True
        )
        if teacher_summary is None:
            raise ValueError("paper ranker requires --teacher-summary")
        summary_provenance = verify_paper_artifact(teacher_summary)
        verify_provenance_compatibility(
            teacher_provenance, summary_provenance, require_same_role=True
        )
    scientific_validity = build_run_metadata(
        run_mode=run_mode,
        data_provenance=data_provenance,
        config={
            "steps": steps,
            "batch_records": batch_records,
            "max_pairs_per_record": max_pairs_per_record,
            "lr": lr,
            "weight_decay": weight_decay,
            "min_teacher_delta": min_teacher_delta,
            "temperature": temperature,
            "min_pair_weight": min_pair_weight,
            "pair_source_mode": pair_source_mode,
            "checkpoint_metric": checkpoint_metric,
            "checkpoint_mode": checkpoint_mode,
            "validation_interval": validation_interval,
            "early_stopping_patience": early_stopping_patience,
            "minimum_improvement": minimum_improvement,
            "validation_candidate_cap": validation_candidate_cap,
            "preference_alpha": dict(preference_alpha or {}),
            "uncertainty_penalty": uncertainty_penalty,
            "minimum_oracle_agreement": minimum_oracle_agreement,
        },
        code_paths=(__file__,),
        training_seed=int(seed),
        oracle=(
            teacher_provenance.get("oracle")
            if isinstance(teacher_provenance, Mapping)
            and isinstance(teacher_provenance.get("oracle"), Mapping)
            else None
        ),
        upstream={
            "base_checkpoint_sha256": sha256_file(base_checkpoint),
            "teacher_jsonl_sha256": sha256_file(train_teacher_path),
            "val_teacher_jsonl_sha256": sha256_file(val_teacher_jsonl),
            "teacher_summary_sha256": (
                sha256_file(teacher_summary) if teacher_summary else None
            ),
            "teacher_provenance_sidecar_sha256": (
                sha256_file(train_teacher_path + ".provenance.json")
                if os.path.isfile(train_teacher_path + ".provenance.json") else None
            ),
        },
        extra_block_reasons=(
            ("teacher_oracle_provenance_missing",)
            if teacher_provenance is None or val_teacher_provenance is None else ()
        ),
    )
    cfg, backbone, model = load_stage_a_checkpoint(base_checkpoint, device=str(dev))
    backbone.eval()
    model.train()
    teachers = load_teacher_rows(train_teacher_path)
    val_teachers = load_teacher_rows(val_teacher_jsonl)
    _require_teacher_alignment(train_records_verified, teachers, partition="train")
    _require_teacher_alignment(val_records_verified, val_teachers, partition="validation")
    all_teacher_rows = [row for rows in teachers.values() for row in rows]
    objective_names = _objective_schema(all_teacher_rows)
    objective_head = (
        PreferenceConditionedObjectiveHead(objective_names).to(dev)
        if objective_names else None
    )
    objective_schema = (
        objective_head.schema() if objective_head is not None else {"objective_names": [], "legacy_scalar_only": True}
    )
    alpha = {
        name: float((preference_alpha or {}).get(name, 1.0))
        for name in objective_names
    }
    inferred_task = str(trained_task or (all_teacher_rows[0].task_id if all_teacher_rows else "T5")).upper()
    inferred_ops = sorted({str(row.op).lower() for row in all_teacher_rows})
    inferred_regions = list(trained_editable_regions or (("cds",) if inferred_task == "T4" else ("utr5",)))
    default_ops = ("sub", "ins", "del") if inferred_task == "T5" else tuple(inferred_ops or ("sub",))
    action_space = {
        "trained_task": inferred_task,
        "trained_editable_regions": inferred_regions,
        "trained_operations": list(trained_operations or default_ops),
    }
    records_by_id = _record_map(train_records_verified)
    val_records_by_id = _record_map(val_records_verified)
    usable = _usable_transcripts(
        records_by_id,
        teachers,
        max_pairs_per_record=max_pairs_per_record,
        min_teacher_delta=min_teacher_delta,
        pair_source_mode=pair_source_mode,
    )
    if not usable:
        raise ValueError("no transcript has at least one informative teacher pair")
    trainable_parameters = list(model.parameters()) + (
        list(objective_head.parameters()) if objective_head is not None else []
    )
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(lr), weight_decay=float(weight_decay))
    os.makedirs(save_dir, exist_ok=True)
    _ensure_parent(profile_path)
    with open(profile_path, "w", encoding="utf-8") as fh:
        fh.write("")
    ckpt_path = os.path.join(save_dir, "proposal_ranker_best.pt")
    rng = random.Random(int(seed))
    best_loss = float("inf")
    best_validation_metric: Optional[float] = None
    best_validation_summary: dict[str, object] = {}
    best_validation_step = 0
    training_loss_at_best_step = float("nan")
    bad_validations = 0
    early_stopping_reason = "completed_max_steps"
    last_stats: dict[str, object] = {}
    tau = max(float(temperature), 1e-8)

    for step in range(1, int(steps) + 1):
        start = time.perf_counter()
        selected = [usable[rng.randrange(len(usable))] for _ in range(max(1, int(batch_records)))]
        optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        pair_count = 0
        delta_sum = 0.0
        pair_source_counts: dict[str, int] = {}
        for tid in selected:
            rows = teachers[tid]
            pairs = _pair_specs(
                rows,
                max_pairs=max_pairs_per_record,
                min_teacher_delta=min_teacher_delta,
                pair_source_mode=pair_source_mode,
            )
            if not pairs:
                continue
            out = _forward_record(records_by_id[tid], model, backbone, dev)
            scores = [_operation_log_score(out, row) for row in rows]
            vectors = [_row_reward_vector(row) for row in rows]
            objective_scores = (
                score_objectives(out, rows, objective_head)
                if objective_head is not None else None
            )
            sampled_preference = (
                sample_dirichlet_preference(alpha) if alpha else {}
            )
            for pair in pairs:
                if (
                    minimum_oracle_agreement is not None
                    and any(
                        row.oracle_agreement is not None
                        and float(row.oracle_agreement) < float(minimum_oracle_agreement)
                        for row in (rows[pair.i], rows[pair.j])
                    )
                ):
                    continue
                if objective_scores is not None:
                    component_deltas: dict[str, float] = {}
                    for name in objective_names:
                        left = vectors[pair.i].raw_delta_from_source.get(name)
                        right = vectors[pair.j].raw_delta_from_source.get(name)
                        if left is None or right is None:
                            continue
                        valid_left = vectors[pair.i].validity.get(name, True)
                        valid_right = vectors[pair.j].validity.get(name, True)
                        if not valid_left or not valid_right:
                            continue
                        delta = float(left) - float(right)
                        if abs(delta) < float(min_teacher_delta):
                            continue
                        direction = 1.0 if delta > 0.0 else -1.0
                        margin = direction * (
                            objective_scores[name][pair.i] - objective_scores[name][pair.j]
                        ) / tau
                        losses.append(F.softplus(-margin) * max(abs(delta), float(min_pair_weight)))
                        component_deltas[name] = delta
                    if component_deltas and sampled_preference:
                        effective_preference = dict(sampled_preference)
                        # The vector stores uncertainty as a negative component;
                        # kappa controls whether/how strongly it enters a
                        # sampled preference objective.  This remains offline
                        # preference distillation, not online RL.
                        if "uncertainty" in effective_preference:
                            effective_preference["uncertainty"] *= float(uncertainty_penalty)
                        preference = normalized_preference({
                            name: effective_preference[name]
                            for name in component_deltas if name in effective_preference
                        })
                        teacher_delta = sum(preference[name] * component_deltas[name] for name in preference)
                        combined_scores = sum(
                            preference[name] * objective_scores[name]
                            for name in preference
                        )
                        direction = 1.0 if teacher_delta > 0.0 else -1.0
                        margin = direction * (combined_scores[pair.i] - combined_scores[pair.j]) / tau
                        losses.append(F.softplus(-margin) * max(abs(teacher_delta), float(min_pair_weight)))
                    pair_count += 1
                    continue
                teacher_delta = float(pair.teacher_delta)
                direction = 1.0 if teacher_delta > 0.0 else -1.0
                margin = direction * (scores[pair.i] - scores[pair.j]) / tau
                weight = max(abs(teacher_delta), float(min_pair_weight))
                losses.append(F.softplus(-margin) * weight)
                pair_count += 1
                delta_sum += abs(teacher_delta)
                pair_source_counts[pair.source_label] = pair_source_counts.get(pair.source_label, 0) + 1
        if not losses:
            continue
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss).all():
            raise FloatingPointError("non-finite proposal ranking loss")
        loss.backward()
        grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        elapsed = max(time.perf_counter() - start, 1e-12)
        loss_value = float(loss.detach().cpu())
        best_loss = min(best_loss, loss_value)
        stats: dict[str, object] = {
            "stage": "proposal_ranker",
            "step": step,
            "loss": loss_value,
            "train_loss": loss_value,
            "pair_count": pair_count,
            "pair_source_mode": pair_source_mode,
            "pair_source_counts": dict(sorted(pair_source_counts.items())),
            "batch_records": len(selected),
            "usable_transcripts": len(usable),
            "mean_abs_teacher_delta": delta_sum / max(1, pair_count),
            "grad_norm": float(grad_norm_t.detach().cpu()),
            "records_per_s": len(selected) / elapsed,
            "finite_loss": True,
        }
        should_validate = step % int(validation_interval) == 0 or step == int(steps)
        if should_validate:
            validation_summary = evaluate_ranker_validation(
                records_by_id=val_records_by_id,
                teachers_by_id=val_teachers,
                score_row=_operation_log_score,
                forward_record=_forward_record,
                model=model,
                backbone=backbone,
                device=dev,
                candidate_cap=validation_candidate_cap,
            )
            value = float(validation_summary[metric_key])
            stats.update({f"val_{key}": value for key, value in validation_summary.items() if isinstance(value, (float, int))})
            stats["val_recall_at_32"] = float(
                validation_summary["oracle_best_recall_at_32"]
            )
            stats["validation_metric_name"] = checkpoint_metric
            stats["validation_metric_value"] = value
            if _metric_improved(
                value, best_validation_metric, mode=checkpoint_mode,
                minimum_improvement=minimum_improvement,
            ):
                best_validation_metric = value
                best_validation_summary = dict(validation_summary)
                best_validation_step = step
                training_loss_at_best_step = loss_value
                bad_validations = 0
                _save_ranker_checkpoint(
                    ckpt_path,
                    cfg=cfg,
                    backbone=backbone,
                    model=model,
                    base_checkpoint=base_checkpoint,
                    teacher_jsonl=train_teacher_path,
                    val_teacher_jsonl=val_teacher_jsonl,
                    step=step,
                    best_loss=best_loss,
                    best_validation_metric=value,
                    best_validation_step=step,
                    validation_summary=validation_summary,
                    training_loss_at_best_step=loss_value,
                    trained_action_space=action_space,
                    checkpoint_metric=checkpoint_metric,
                    checkpoint_mode=checkpoint_mode,
                    objective_head=objective_head,
                    objective_schema=objective_schema,
                    preference_training={
                        "mode": "dirichlet" if alpha else "legacy_scalar",
                        "alpha": alpha,
                        "uncertainty_penalty": float(uncertainty_penalty),
                        "minimum_oracle_agreement": minimum_oracle_agreement,
                    },
                    scientific_validity=scientific_validity,
                )
            else:
                bad_validations += 1
                if int(early_stopping_patience) > 0 and bad_validations >= int(early_stopping_patience):
                    early_stopping_reason = (
                        f"no_validation_improvement_for_{bad_validations}_intervals"
                    )
                    stats["early_stopping_reason"] = early_stopping_reason
                    _write_profile(profile_path, stats)
                    last_stats = stats
                    break
        _write_profile(profile_path, stats)
        last_stats = stats
    if best_validation_metric is None or not os.path.isfile(ckpt_path):
        raise RuntimeError("no validation checkpoint was produced")
    _annotate_checkpoint_termination(ckpt_path, early_stopping_reason)
    return {
        "stage": "proposal_ranker",
        "checkpoint_path": ckpt_path,
        "profile_path": profile_path,
        "best_loss": best_loss,
        "best_validation_metric": best_validation_metric,
        "best_validation_step": best_validation_step,
        "validation_summary": best_validation_summary,
        "training_loss_at_best_step": training_loss_at_best_step,
        "checkpoint_metric": checkpoint_metric,
        "checkpoint_mode": checkpoint_mode,
        "early_stopping_reason": early_stopping_reason,
        "last_stats": last_stats,
        "usable_transcripts": len(usable),
        "pair_source_mode": pair_source_mode,
        "trained_action_space": action_space,
        "objective_schema": objective_schema,
        "preference_training": {
            "mode": "dirichlet" if alpha else "legacy_scalar",
            "alpha": alpha,
            "uncertainty_penalty": float(uncertainty_penalty),
            "minimum_oracle_agreement": minimum_oracle_agreement,
        },
        "scientific_validity": scientific_validity,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune MEF proposal ranking from oracle-teacher JSONL")
    parser.add_argument("--records-jsonl", default=None, help="legacy alias for --train-records-jsonl")
    parser.add_argument("--teacher-jsonl", default=None, help="legacy alias for --train-teacher-jsonl")
    parser.add_argument("--train-records-jsonl", default=None)
    parser.add_argument("--val-records-jsonl", default=None)
    parser.add_argument("--train-teacher-jsonl", default=None)
    parser.add_argument("--val-teacher-jsonl", default=None)
    parser.add_argument("--teacher-summary", default=None)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--profile-path", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-records", type=int, default=4)
    parser.add_argument("--max-pairs-per-record", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-teacher-delta", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min-pair-weight", type=float, default=0.01)
    parser.add_argument(
        "--pair-source-mode",
        choices=("global", "source_balanced"),
        default="global",
        help=(
            "global uses teacher_score as before; source_balanced uses "
            "source_scores labels such as full/utr with a split pair budget"
        ),
    )
    parser.add_argument("--run-mode", choices=("development", "paper"), default="development")
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--split-role", choices=("train", "val", "test"), default=None)
    parser.add_argument("--trained-task", default=None)
    parser.add_argument("--trained-editable-region", action="append", default=None)
    parser.add_argument("--trained-operation", action="append", default=None)
    parser.add_argument("--checkpoint-metric", choices=("val_mean_model_regret", "val_oracle_best_recall_at_32", "val_ndcg_at_32"), default="val_mean_model_regret")
    parser.add_argument("--checkpoint-mode", choices=("min", "max"), default="min")
    parser.add_argument("--validation-interval", type=int, default=25)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--minimum-improvement", type=float, default=0.0)
    parser.add_argument("--validation-candidate-cap", type=int, default=0)
    parser.add_argument("--preference-alpha", action="append", default=None, metavar="OBJECTIVE=ALPHA")
    parser.add_argument("--uncertainty-penalty", type=float, default=0.0)
    parser.add_argument("--minimum-oracle-agreement", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    train_records_path = args.train_records_jsonl or args.records_jsonl
    train_teacher_path = args.train_teacher_jsonl or args.teacher_jsonl
    if not train_records_path or not args.val_records_jsonl or not train_teacher_path or not args.val_teacher_jsonl:
        raise SystemExit("Stage 2 requires --train-records-jsonl --val-records-jsonl --train-teacher-jsonl --val-teacher-jsonl")
    if args.run_mode == "paper" and not args.split_manifest:
        raise SystemExit("paper ranker validation requires --split-manifest")
    split_contract = (
        load_and_verify_split_manifest(args.split_manifest)
        if args.split_manifest else None
    )
    preference_alpha: dict[str, float] = {}
    for item in args.preference_alpha or ():
        if "=" not in item:
            raise SystemExit("--preference-alpha expects OBJECTIVE=ALPHA")
        name, value = item.split("=", 1)
        preference_alpha[name] = float(value)
    result = train_proposal_ranker(
        train_records=load_records_jsonl(train_records_path),
        val_records=load_records_jsonl(args.val_records_jsonl),
        train_teacher_jsonl=train_teacher_path,
        val_teacher_jsonl=args.val_teacher_jsonl,
        teacher_summary=args.teacher_summary,
        base_checkpoint=args.base_checkpoint,
        save_dir=args.save_dir,
        profile_path=args.profile_path,
        steps=args.steps,
        batch_records=args.batch_records,
        max_pairs_per_record=args.max_pairs_per_record,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
        min_teacher_delta=args.min_teacher_delta,
        temperature=args.temperature,
        min_pair_weight=args.min_pair_weight,
        pair_source_mode=args.pair_source_mode,
        run_mode=args.run_mode,
        split_contract=split_contract,
        split_role=args.split_role,
        trained_task=args.trained_task,
        trained_editable_regions=args.trained_editable_region,
        trained_operations=args.trained_operation,
        checkpoint_metric=args.checkpoint_metric,
        checkpoint_mode=args.checkpoint_mode,
        validation_interval=args.validation_interval,
        early_stopping_patience=args.early_stopping_patience,
        minimum_improvement=args.minimum_improvement,
        validation_candidate_cap=args.validation_candidate_cap,
        preference_alpha=preference_alpha or None,
        uncertainty_penalty=args.uncertainty_penalty,
        minimum_oracle_agreement=args.minimum_oracle_agreement,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
