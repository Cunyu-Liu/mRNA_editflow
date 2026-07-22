"""Offline DAgger iteration runner for proposal-ranker distillation.

This module orchestrates ``rollout -> offline relabel -> replay mix -> ranker
retrain``.  It does not contain an RL loss, a policy ratio, or an online Oracle
action selector; all ranking optimisation remains the existing offline
Bradley--Terry/vector-preference distillation.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional, Sequence

from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.rl.dagger_teacher_export import (
    DaggerRolloutConfig,
    export_dagger_teacher_jsonl,
    rollout_records,
)
from mrna_editflow.rl.rollout_buffer import REPLAY_BUCKETS, ReplayBuffer, ReplayMixConfig, iteration_directory
from mrna_editflow.sample import load_stage_a_checkpoint
from mrna_editflow.train_proposal_ranker import train_proposal_ranker


@dataclass(frozen=True)
class DaggerIterationConfig:
    iteration: int
    task_id: str = "T5"
    rollout_edit_budget: int = 3
    rollout_top_k: int = 8
    rollout_temperature: float = 1.0
    rollout_allow_stop: bool = True
    rollout_stop_logit_bias: float = 0.0
    rollout_min_action_margin: float = 0.0
    replay_mix: ReplayMixConfig = field(default_factory=ReplayMixConfig)
    mixed_rows: int = 0
    seed: int = 0
    ranker_steps: int = 100
    ranker_batch_records: int = 4
    ranker_validation_interval: int = 25


def _read_jsonl(path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"JSONL row must be an object: {path}")
                rows.append(dict(value))
    return rows


def _write_jsonl(rows: Sequence[Mapping[str, object]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _write_records(records: Sequence[MRNARecord], path: str) -> None:
    _write_jsonl([record.to_dict() for record in records], path)


def _maximum_mixture_size(buffer: ReplayBuffer, mix: ReplayMixConfig) -> int:
    """Largest total that can satisfy all non-zero mix quotas exactly."""
    counts, weights = buffer.bucket_counts(), mix.normalized()
    upper = min(
        int(counts[bucket] / weight)
        for bucket in REPLAY_BUCKETS if weights[bucket] > 0.0
    )
    for total in range(max(0, upper), -1, -1):
        quotas = mix.quotas(total)
        if all(counts[bucket] >= quotas[bucket] for bucket in REPLAY_BUCKETS):
            return total
    return 0


def _as_hard_negatives(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Reuse low-scoring visited-state candidates as explicit hard negatives."""
    candidates = [dict(row) for row in rows if str(row.get("op", "")) != "stop"]
    candidates.sort(key=lambda row: (float(row.get("teacher_score", 0.0)), str(row.get("transcript_id", ""))))
    return candidates


def train_dagger_iteration(
    *,
    train_records: Sequence[MRNARecord],
    validation_records: Sequence[MRNARecord],
    original_train_teacher_jsonl: str,
    validation_teacher_jsonl: str,
    policy_checkpoint: str,
    output_root: str,
    config: DaggerIterationConfig,
    device: Optional[str] = None,
    hard_negative_jsonl: Optional[str] = None,
) -> dict[str, object]:
    """Run one non-overwritable, train-only offline DAgger iteration."""
    train_ids = {record.transcript_id for record in train_records}
    val_ids = {record.transcript_id for record in validation_records}
    if train_ids & val_ids:
        raise ValueError("training and validation records overlap")
    iteration_dir = iteration_directory(output_root, config.iteration)
    cfg, backbone, model = load_stage_a_checkpoint(policy_checkpoint, device=device)
    rollout_cfg = DaggerRolloutConfig(
        task_id=config.task_id,
        edit_budget=config.rollout_edit_budget,
        proposal_top_k=config.rollout_top_k,
        proposal_temperature=config.rollout_temperature,
        allow_stop=config.rollout_allow_stop,
        stop_logit_bias=config.rollout_stop_logit_bias,
        min_action_margin=config.rollout_min_action_margin,
    )
    trajectories = rollout_records(
        list(train_records), model, backbone, config=rollout_cfg,
        policy_checkpoint_sha256=getattr(model, "_checkpoint_sha256", None),
        seed=config.seed, device=device,
    )
    dagger_rows_path = os.path.join(iteration_dir, "dagger_teacher.jsonl")
    state_records_path = os.path.join(iteration_dir, "visited_state_records.jsonl")
    trajectories_path = os.path.join(iteration_dir, "trajectories.jsonl")
    relabel_summary_path = os.path.join(iteration_dir, "relabel_summary.json")
    relabel_summary = export_dagger_teacher_jsonl(
        trajectories,
        out_jsonl=dagger_rows_path,
        out_states_jsonl=state_records_path,
        out_trajectories_jsonl=trajectories_path,
        out_summary_json=relabel_summary_path,
        train_transcript_ids=sorted(train_ids),
        validation_transcript_ids=sorted(val_ids),
    )
    original_rows = _read_jsonl(original_train_teacher_jsonl)
    dagger_rows = _read_jsonl(dagger_rows_path)
    buffer = ReplayBuffer(
        policy_checkpoint_sha256=str(getattr(model, "_checkpoint_sha256")),
        iteration=int(config.iteration),
    )
    buffer.add(original_rows, bucket="original")
    buffer.add([row for row in dagger_rows if str(row.get("op")) != "stop"], bucket="rollout")
    buffer.add([row for row in dagger_rows if str(row.get("op")) == "stop"], bucket="stop")
    hard_negative_rows = _read_jsonl(hard_negative_jsonl) if hard_negative_jsonl else _as_hard_negatives(dagger_rows)
    buffer.add(hard_negative_rows, bucket="hard_negative")
    total = int(config.mixed_rows) if int(config.mixed_rows) > 0 else _maximum_mixture_size(buffer, config.replay_mix)
    if total <= 0:
        raise ValueError("no non-empty exact replay mixture can be formed")
    mixed_rows = buffer.sample_mixed(total, config=config.replay_mix, seed=config.seed)
    mixed_teacher_path = os.path.join(iteration_dir, "mixed_teacher.jsonl")
    buffer_path = os.path.join(iteration_dir, "replay_buffer.jsonl")
    buffer_manifest_path = os.path.join(iteration_dir, "replay_buffer_manifest.json")
    _write_jsonl(mixed_rows, mixed_teacher_path)
    buffer.write_jsonl(buffer_path)
    buffer_manifest = buffer.write_manifest(
        buffer_manifest_path,
        extra={"mix": asdict(config.replay_mix), "mixed_rows": len(mixed_rows)},
    )
    visited = [MRNARecord.from_dict(row) for row in _read_jsonl(state_records_path)]
    available_records = {record.transcript_id: record for record in list(train_records) + visited}
    if len(available_records) != len(train_records) + len(visited):
        raise ValueError("DAgger state IDs collide with source records")
    selected_ids = {str(row["transcript_id"]) for row in mixed_rows}
    unknown_ids = selected_ids - set(available_records)
    if unknown_ids:
        raise ValueError("mixed replay teacher references an unknown state record")
    # The ranker requires an exact teacher/record ID match.  A sampled replay
    # mixture intentionally contains only a subset of available source/states,
    # so train on exactly that subset rather than adding unlabeled records.
    all_records = [available_records[identifier] for identifier in sorted(selected_ids)]
    combined_records_path = os.path.join(iteration_dir, "mixed_train_records.jsonl")
    _write_records(all_records, combined_records_path)
    result = train_proposal_ranker(
        train_records=all_records,
        val_records=list(validation_records),
        train_teacher_jsonl=mixed_teacher_path,
        val_teacher_jsonl=validation_teacher_jsonl,
        base_checkpoint=policy_checkpoint,
        save_dir=os.path.join(iteration_dir, "ranker"),
        profile_path=os.path.join(iteration_dir, "ranker_profile.jsonl"),
        steps=int(config.ranker_steps),
        batch_records=int(config.ranker_batch_records),
        validation_interval=int(config.ranker_validation_interval),
        device=device,
        seed=int(config.seed),
    )
    validation = result.get("validation_summary", {})
    if not isinstance(validation, Mapping):
        validation = {}
    metrics = {
        "policy_version": getattr(model, "_checkpoint_sha256"),
        "buffer_size": int(buffer_manifest["buffer_size"]),
        "state_diversity": int(relabel_summary["state_diversity"]),
        "mean_trajectory_reward": float(relabel_summary["mean_trajectory_reward"]),
        "validation_regret": validation.get("mean_model_regret"),
        "positive_edit_precision": validation.get("positive_edit_precision_at_1"),
        "stop_rate": sum(1 for item in trajectories if item.terminated) / max(1, len(trajectories)),
        "cycle_rejection_rate": sum(item.cycle_rejections for item in trajectories) / max(1, len(trajectories)),
        "validation_transcripts_used_for_dagger": 0,
    }
    manifest = {
        "stage": "offline_dagger_ranker",
        "iteration": int(config.iteration),
        "config": asdict(config),
        "policy_checkpoint": policy_checkpoint,
        "relabel_summary": relabel_summary,
        "buffer_manifest": buffer_manifest,
        "metrics": metrics,
        "ranker_result": result,
        "oracle_used_for_action_selection": False,
        "online_grpo": False,
    }
    manifest_path = os.path.join(iteration_dir, "iteration_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return dict(manifest, iteration_dir=iteration_dir, iteration_manifest=manifest_path)


def _records(path: str) -> list[MRNARecord]:
    return [MRNARecord.from_dict(row) for row in _read_jsonl(path)]


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-records-jsonl", required=True)
    parser.add_argument("--validation-records-jsonl", required=True)
    parser.add_argument("--original-train-teacher-jsonl", required=True)
    parser.add_argument("--validation-teacher-jsonl", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--task-id", default="T5")
    parser.add_argument("--rollout-edit-budget", type=int, default=3)
    parser.add_argument("--rollout-top-k", type=int, default=8)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-no-stop", action="store_true", help="Testing-only: force edits until budget")
    parser.add_argument("--mixed-rows", type=int, default=0)
    parser.add_argument("--ranker-steps", type=int, default=100)
    parser.add_argument("--ranker-batch-records", type=int, default=4)
    parser.add_argument("--ranker-validation-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--hard-negative-jsonl", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    result = train_dagger_iteration(
        train_records=_records(args.train_records_jsonl), validation_records=_records(args.validation_records_jsonl),
        original_train_teacher_jsonl=args.original_train_teacher_jsonl,
        validation_teacher_jsonl=args.validation_teacher_jsonl, policy_checkpoint=args.policy_checkpoint,
        output_root=args.output_root,
        config=DaggerIterationConfig(
            iteration=args.iteration, task_id=args.task_id, rollout_edit_budget=args.rollout_edit_budget,
            rollout_top_k=args.rollout_top_k, rollout_temperature=args.rollout_temperature,
            rollout_allow_stop=not args.rollout_no_stop,
            mixed_rows=args.mixed_rows, ranker_steps=args.ranker_steps,
            ranker_batch_records=args.ranker_batch_records,
            ranker_validation_interval=args.ranker_validation_interval, seed=args.seed,
        ),
        device=args.device, hard_negative_jsonl=args.hard_negative_jsonl,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
