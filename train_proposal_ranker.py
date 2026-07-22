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
from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.download_mrna import load_records_jsonl
from mrna_editflow.data.split_contract import (
    VerifiedSplitContract,
    load_and_verify_split_manifest,
    sha256_file,
)
from mrna_editflow.eval.artifact_contract import (
    ArtifactProvenanceError,
    build_run_metadata,
    load_artifact_provenance,
    normalize_run_mode,
    prepare_scientific_records,
    require_paper_cli_inputs,
    validate_output_namespace,
    verify_paper_artifact,
    verify_paper_checkpoint,
    verify_provenance_compatibility,
)
from mrna_editflow.sample import _record_tensors, load_stage_a_checkpoint
from mrna_editflow.rl.action_scoring import operation_log_score
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
    step: int,
    best_loss: float,
    trained_action_space: Mapping[str, object],
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


def train_proposal_ranker(
    *,
    records: Sequence[MRNARecord],
    teacher_jsonl: str,
    base_checkpoint: str,
    save_dir: str,
    profile_path: str,
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
) -> dict[str, object]:
    """Fine-tune the MEF head to rank oracle-preferred legal proposals higher."""
    run_mode = normalize_run_mode(run_mode)
    _set_seed(int(seed))
    validate_output_namespace(save_dir, run_mode)
    validate_output_namespace(profile_path, run_mode)
    dev = _resolve_device(device)
    train_records, data_provenance = prepare_scientific_records(
        records,
        run_mode=run_mode,
        split_contract=split_contract,
        split_role=split_role,
        allowed_roles=("train",),
    )
    teacher_provenance: Optional[dict[str, object]] = None
    try:
        teacher_provenance = load_artifact_provenance(teacher_jsonl)
    except ArtifactProvenanceError:
        if run_mode == "paper":
            raise
    if run_mode == "paper":
        verify_paper_checkpoint(base_checkpoint, data_provenance)
        teacher_provenance = verify_paper_artifact(teacher_jsonl)
        verify_provenance_compatibility(
            data_provenance, teacher_provenance, require_same_role=True
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
            "teacher_jsonl_sha256": sha256_file(teacher_jsonl),
            "teacher_summary_sha256": (
                sha256_file(teacher_summary) if teacher_summary else None
            ),
            "teacher_provenance_sidecar_sha256": (
                sha256_file(teacher_jsonl + ".provenance.json")
                if os.path.isfile(teacher_jsonl + ".provenance.json") else None
            ),
        },
        extra_block_reasons=(
            ("teacher_oracle_provenance_missing",)
            if teacher_provenance is None else ()
        ),
    )
    cfg, backbone, model = load_stage_a_checkpoint(base_checkpoint, device=str(dev))
    backbone.eval()
    model.train()
    teachers = load_teacher_rows(teacher_jsonl)
    all_teacher_rows = [row for rows in teachers.values() for row in rows]
    inferred_task = str(trained_task or (all_teacher_rows[0].task_id if all_teacher_rows else "T5")).upper()
    inferred_ops = sorted({str(row.op).lower() for row in all_teacher_rows})
    inferred_regions = list(trained_editable_regions or (("cds",) if inferred_task == "T4" else ("utr5",)))
    default_ops = ("sub", "ins", "del") if inferred_task == "T5" else tuple(inferred_ops or ("sub",))
    action_space = {
        "trained_task": inferred_task,
        "trained_editable_regions": inferred_regions,
        "trained_operations": list(trained_operations or default_ops),
    }
    records_by_id = _record_map(train_records)
    usable = _usable_transcripts(
        records_by_id,
        teachers,
        max_pairs_per_record=max_pairs_per_record,
        min_teacher_delta=min_teacher_delta,
        pair_source_mode=pair_source_mode,
    )
    if not usable:
        raise ValueError("no transcript has at least one informative teacher pair")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    os.makedirs(save_dir, exist_ok=True)
    _ensure_parent(profile_path)
    with open(profile_path, "w", encoding="utf-8") as fh:
        fh.write("")
    ckpt_path = os.path.join(save_dir, "proposal_ranker_best.pt")
    rng = random.Random(int(seed))
    best_loss = float("inf")
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
            for pair in pairs:
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
        stats: dict[str, object] = {
            "stage": "proposal_ranker",
            "step": step,
            "loss": loss_value,
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
        _write_profile(profile_path, stats)
        last_stats = stats
        if loss_value < best_loss:
            best_loss = loss_value
            _save_ranker_checkpoint(
                ckpt_path,
                cfg=cfg,
                backbone=backbone,
                model=model,
                base_checkpoint=base_checkpoint,
                teacher_jsonl=teacher_jsonl,
                step=step,
                best_loss=best_loss,
                trained_action_space=action_space,
                scientific_validity=scientific_validity,
            )
    return {
        "stage": "proposal_ranker",
        "checkpoint_path": ckpt_path,
        "profile_path": profile_path,
        "best_loss": best_loss,
        "last_stats": last_stats,
        "usable_transcripts": len(usable),
        "pair_source_mode": pair_source_mode,
        "trained_action_space": action_space,
        "scientific_validity": scientific_validity,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune MEF proposal ranking from oracle-teacher JSONL")
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--teacher-jsonl", required=True)
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
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    require_paper_cli_inputs(
        run_mode=args.run_mode,
        split_manifest=args.split_manifest,
        split_role=args.split_role,
        allowed_roles=("train",),
    )
    split_contract = (
        load_and_verify_split_manifest(args.split_manifest, records_path=args.records_jsonl)
        if args.split_manifest else None
    )
    result = train_proposal_ranker(
        records=load_records_jsonl(args.records_jsonl),
        teacher_jsonl=args.teacher_jsonl,
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
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
