"""Offline DAgger rollout, relabelling, and trajectory-teacher export.

The policy rollout is intentionally Oracle-free.  Only after trajectories have
been fully saved are their visited states enumerated and labelled with the
heuristic Oracle.  This prevents Oracle-guided action selection leakage.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch

from mrna_editflow.baselines.multiobjective_teacher_export import (
    MultiObjectiveConfig,
    MultiObjectiveTeacherRow,
    score_record_multiobjective_rows,
)
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.eval.oracle import LocalTranslationOracle
from mrna_editflow.rl.decoder_state import DecoderAction, DecoderState, choose_stop_aware_action, sequence_hash
from mrna_editflow.rl.trajectory_schema import (
    OfflineTrajectory,
    TrajectoryAction,
    TrajectoryState,
    TrajectoryTeacherRow,
    potential,
    potential_reward,
    validate_telescoping,
)
from mrna_editflow.sample import (
    _copy_record,
    _model_out_for_record,
    _replace_nt,
    _resolve_decoder_action_space,
    _utr_substitution_candidates,
)


@dataclass(frozen=True)
class DaggerRolloutConfig:
    task_id: str = "T5"
    edit_budget: int = 3
    proposal_top_k: int = 8
    proposal_temperature: float = 1.0
    allow_stop: bool = True
    stop_logit_bias: float = 0.0
    min_action_margin: float = 0.0
    editable_regions: Optional[tuple[str, ...]] = None
    allow_action_space_expansion: bool = False


def _snapshot(record: MRNARecord, *, state_id: str, step: int) -> TrajectoryState:
    return TrajectoryState(
        transcript_id=state_id,
        sequence=record.seq,
        five_utr=record.five_utr,
        cds=record.cds,
        three_utr=record.three_utr,
        step_index=int(step),
        species=record.species,
    )


def _state_record(snapshot: TrajectoryState) -> MRNARecord:
    return MRNARecord(
        transcript_id=snapshot.transcript_id,
        five_utr=snapshot.five_utr,
        cds=snapshot.cds,
        three_utr=snapshot.three_utr,
        species=snapshot.species,
        metadata={"dagger_step_index": int(snapshot.step_index)},
    )


def _checkpoint_hash(model: object, explicit: Optional[str]) -> str:
    digest = explicit or getattr(model, "_checkpoint_sha256", None)
    if not isinstance(digest, str) or not digest:
        raise ValueError("DAgger rollout requires policy_checkpoint_sha256 provenance")
    return digest


def rollout_model_guided_trajectory(
    record: MRNARecord,
    model: object,
    backbone: object,
    *,
    config: Optional[DaggerRolloutConfig] = None,
    policy_checkpoint_sha256: Optional[str] = None,
    seed: int = 0,
    device: Optional[str] = None,
    rollout_id: int = 0,
) -> OfflineTrajectory:
    """Run one constrained, Oracle-free model-guided rollout.

    No Oracle parameter is accepted.  Legal proposals and selection use only
    model CTMC scores plus the shared STOP/cycle machinery from ``sample.py``.
    """
    cfg = config or DaggerRolloutConfig()
    task_id = str(cfg.task_id).upper()
    # Stage 4 reuses the audited Stage 3 multiobjective teacher, whose legal
    # candidate enumerator is deliberately T5 5'UTR-only. Fail closed rather
    # than relabel a trajectory under a different editing grammar.
    if task_id != "T5":
        raise ValueError("offline DAgger relabelling currently supports only the Stage 3 T5 5'UTR teacher")
    checkpoint_sha = _checkpoint_hash(model, policy_checkpoint_sha256)
    enabled_regions, expanded = _resolve_decoder_action_space(
        model,
        task_id=task_id,
        requested_regions=cfg.editable_regions,
        allow_action_space_expansion=cfg.allow_action_space_expansion,
    )
    rng = random.Random(int(seed))
    dev = torch.device(device or next(model.parameters()).device)  # type: ignore[attr-defined]
    current = _copy_record(record, transcript_id=record.transcript_id)
    state = DecoderState(current.seq, max(0, int(cfg.edit_budget)), out_of_training_action_space=expanded)
    prefix = f"{record.transcript_id}__dagger_{checkpoint_sha[:12]}_{int(rollout_id):03d}"
    snapshots = [_snapshot(current, state_id=f"{prefix}_s000", step=0)]
    actions: list[TrajectoryAction] = []
    termination = "edit_budget"

    for step in range(max(0, int(cfg.edit_budget))):
        out = _model_out_for_record(current, model, backbone, dev)
        choices = _utr_substitution_candidates(current, out, enabled_regions)
        op = "sub"
        materialize = lambda pos, nt: _replace_nt(current, pos, nt, current.transcript_id)
        decoder_actions: list[DecoderAction] = []
        for score, pos, nt in choices:
            candidate = materialize(pos, nt)
            decoder_actions.append(DecoderAction(op, int(pos), str(nt) or None, float(score), sequence_hash(candidate.seq), current.seq[int(pos)]))
        chosen = choose_stop_aware_action(
            decoder_actions,
            state,
            rng,
            top_k=int(cfg.proposal_top_k),
            temperature=float(cfg.proposal_temperature),
            allow_stop=bool(cfg.allow_stop),
            stop_logit_bias=float(cfg.stop_logit_bias),
            min_action_margin=float(cfg.min_action_margin),
        )
        if chosen is None:
            actions.append(TrajectoryAction("stop", None, "", float(cfg.stop_logit_bias)))
            termination = "stop" if state.terminated_by_stop else "no_legal_action"
            break
        current = _replace_nt(current, int(chosen.pos), str(chosen.nt or ""), current.transcript_id)
        actions.append(TrajectoryAction(chosen.op, int(chosen.pos), str(chosen.nt or ""), float(chosen.log_score)))
        snapshots.append(_snapshot(current, state_id=f"{prefix}_s{step + 1:03d}", step=step + 1))
    return OfflineTrajectory(
        source_transcript_id=record.transcript_id,
        task_id=task_id,
        policy_checkpoint_sha256=checkpoint_sha,
        states=snapshots,
        action_history=actions,
        terminated=bool(state.terminated_by_stop),
        termination_reason=termination,
        cycle_rejections=int(state.cycle_rejections),
        rollout_temperature=float(cfg.proposal_temperature),
        rollout_top_k=int(cfg.proposal_top_k),
    )


def rollout_records(
    records: Sequence[MRNARecord], model: object, backbone: object, *, config: Optional[DaggerRolloutConfig] = None,
    policy_checkpoint_sha256: Optional[str] = None, seed: int = 0, device: Optional[str] = None,
) -> list[OfflineTrajectory]:
    """Roll out only supplied records; callers must pass the training partition."""
    return [
        rollout_model_guided_trajectory(
            record, model, backbone, config=config, policy_checkpoint_sha256=policy_checkpoint_sha256,
            seed=int(seed) + index, device=device, rollout_id=index,
        )
        for index, record in enumerate(records)
    ]


def _stop_properties(rows: Sequence[MultiObjectiveTeacherRow]) -> dict[str, float]:
    stop = next((row for row in rows if row.op == "stop"), None)
    if stop is None:
        raise RuntimeError("teacher relabeler must emit an explicit STOP row")
    return {str(key): float(value) for key, value in stop.raw_absolute_level.items()}


def relabel_trajectory(
    trajectory: OfflineTrajectory,
    *,
    config: Optional[MultiObjectiveConfig] = None,
    oracle: Optional[LocalTranslationOracle] = None,
) -> tuple[list[TrajectoryTeacherRow], list[MRNARecord], dict[str, object]]:
    """Enumerate and Oracle-label legal one-step actions at visited states.

    This function is the first point that constructs an Oracle.  It runs after
    the rollout trajectory is immutable, so it cannot affect action selection.
    """
    cfg = config or MultiObjectiveConfig()
    pred = oracle or LocalTranslationOracle()
    state_rows: list[tuple[TrajectoryState, MRNARecord, list[MultiObjectiveTeacherRow], dict[str, float]]] = []
    for snapshot in trajectory.states:
        state_record = _state_record(snapshot)
        rows = score_record_multiobjective_rows(state_record, config=cfg, oracle=pred)
        state_rows.append((snapshot, state_record, rows, _stop_properties(rows)))
    if not state_rows:
        return [], [], {"state_diversity": 0, "mean_trajectory_reward": 0.0}
    source_properties = dict(state_rows[0][3])
    trajectory_properties = [properties for _snapshot, _record, _rows, properties in state_rows]
    if not validate_telescoping(trajectory_properties):
        raise AssertionError("potential rewards failed telescoping validation")
    labels: list[TrajectoryTeacherRow] = []
    state_records: list[MRNARecord] = []
    for snapshot, state_record, rows, state_properties in state_rows:
        # The initial source already belongs to the original teacher set; only
        # genuinely visited intermediate states are appended to replay training.
        if snapshot.step_index > 0:
            state_records.append(state_record)
        history = tuple(trajectory.action_history[: snapshot.step_index])
        before = potential(state_properties)
        for row in rows:
            # The source state is already covered by the original one-step
            # teacher.  DAgger contributes only genuinely visited intermediate
            # states, whose IDs are emitted in ``out_states_jsonl``.
            if snapshot.step_index == 0:
                continue
            payload = row.to_dict()
            candidate_properties = {str(key): float(value) for key, value in row.raw_absolute_level.items()}
            after = potential(candidate_properties)
            labels.append(
                TrajectoryTeacherRow(
                    source_transcript_id=trajectory.source_transcript_id,
                    state_transcript_id=snapshot.transcript_id,
                    state_sequence=snapshot.sequence,
                    step_index=snapshot.step_index,
                    policy_checkpoint_sha256=trajectory.policy_checkpoint_sha256,
                    action_history=history,
                    candidate_action={"op": row.op, "pos": row.pos, "nt": row.nt},
                    reward_vector=row.reward_vector,
                    raw_delta=row.raw_delta_from_source,
                    raw_absolute_level=row.raw_absolute_level,
                    normalized_within_group=row.normalized_within_group,
                    source_properties=source_properties,
                    state_properties=state_properties,
                    candidate_properties=candidate_properties,
                    potential_before=before,
                    potential_after=after,
                    potential_reward=potential_reward(state_properties, candidate_properties),
                    terminated=bool(trajectory.terminated),
                    teacher_payload=payload,
                )
            )
    meta = {
        "source_transcript_id": trajectory.source_transcript_id,
        "policy_checkpoint_sha256": trajectory.policy_checkpoint_sha256,
        "state_diversity": len({snapshot.sequence for snapshot, _record, _rows, _properties in state_rows}),
        "mean_trajectory_reward": (
            sum(potential_reward(left, right) for left, right in zip(trajectory_properties, trajectory_properties[1:]))
            / max(1, len(trajectory_properties) - 1)
        ),
        "trajectory_telescoping_valid": True,
    }
    return labels, state_records, meta


def export_dagger_teacher_jsonl(
    trajectories: Sequence[OfflineTrajectory], *, out_jsonl: str, out_states_jsonl: str,
    out_trajectories_jsonl: str, out_summary_json: str, train_transcript_ids: Sequence[str],
    validation_transcript_ids: Sequence[str] = (), config: Optional[MultiObjectiveConfig] = None,
) -> dict[str, object]:
    """Write versioned offline DAgger teacher data; validation IDs are rejected."""
    train_ids, val_ids = set(train_transcript_ids), set(validation_transcript_ids)
    if train_ids & val_ids:
        raise ValueError("train and validation transcript IDs overlap")
    if any(item.source_transcript_id not in train_ids for item in trajectories):
        raise ValueError("DAgger teacher export may only consume training source transcripts")
    if any(item.source_transcript_id in val_ids for item in trajectories):
        raise ValueError("validation transcripts must never enter DAgger teacher training")
    all_rows: list[TrajectoryTeacherRow] = []
    all_states: list[MRNARecord] = []
    summaries: list[dict[str, object]] = []
    for trajectory in trajectories:
        rows, states, summary = relabel_trajectory(trajectory, config=config)
        all_rows.extend(rows)
        all_states.extend(states)
        summaries.append(summary)
    for path in (out_jsonl, out_states_jsonl, out_trajectories_jsonl, out_summary_json):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")
    seen_state_ids: set[str] = set()
    with open(out_states_jsonl, "w", encoding="utf-8") as fh:
        for state in all_states:
            if state.transcript_id in seen_state_ids:
                continue
            seen_state_ids.add(state.transcript_id)
            fh.write(json.dumps(state.to_dict(), sort_keys=True) + "\n")
    with open(out_trajectories_jsonl, "w", encoding="utf-8") as fh:
        for trajectory in trajectories:
            fh.write(json.dumps(trajectory.to_dict(), sort_keys=True) + "\n")
    payload = {
        "n_trajectories": len(trajectories),
        "n_teacher_rows": len(all_rows),
        "n_intermediate_state_records": len(seen_state_ids),
        "policy_versions": sorted({item.policy_checkpoint_sha256 for item in trajectories}),
        "state_diversity": len({item.state_sequence for item in all_rows}),
        "mean_trajectory_reward": (
            sum(float(item["mean_trajectory_reward"]) for item in summaries) / max(1, len(summaries))
        ),
        "all_trajectory_telescoping_valid": all(bool(item["trajectory_telescoping_valid"]) for item in summaries),
        "oracle_used_for_action_selection": False,
        "train_only": True,
        "out_jsonl": out_jsonl,
        "out_states_jsonl": out_states_jsonl,
        "out_trajectories_jsonl": out_trajectories_jsonl,
    }
    with open(out_summary_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return payload


__all__ = [
    "DaggerRolloutConfig", "rollout_model_guided_trajectory", "rollout_records",
    "relabel_trajectory", "export_dagger_teacher_jsonl",
]
