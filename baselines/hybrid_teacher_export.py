"""Hybrid proposal-teacher export for MEF ranker distillation.

The previous TE-ranker improves top-1 proposal ranking, while the UTR-teacher
ranker greatly improves top-k recall of oracle-best UTR edits. This module
combines both supervision sources into one ranker-compatible JSONL artifact.

For a source transcript ``x`` and candidate edit ``c``, each input teacher
provides a TE delta

``y_s(c) = TE_s(c) - TE_s(x)``.

Rows are keyed by ``(transcript_id, task_id, op, pos, nt)``. When multiple
teachers label the same candidate, the hybrid score is the weighted mean

``y_h(c) = sum_s w_s y_s(c) / sum_s w_s``.

Unique candidates keep their source score. The output rows contain the standard
``train_proposal_ranker`` fields ``transcript_id/op/pos/nt/teacher_score`` plus
audit metadata recording which teacher sources contributed. Per-transcript
capping is deterministic and alternates high-score and low-score extremes so
Bradley-Terry training retains both positive and negative ranking contrast.

Complexity is ``O(R log R)`` time and ``O(U)`` memory, where ``R`` is total input
rows and ``U`` is the number of unique candidate keys after deduplication.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional, Sequence

from mrna_editflow.data.split_contract import (
    VerifiedSplitContract,
    build_split_provenance,
    load_and_verify_split_manifest,
    sha256_file,
)
from mrna_editflow.eval.artifact_contract import (
    build_run_metadata,
    normalize_run_mode,
    require_paper_cli_inputs,
    upstream_data_provenance,
    validate_output_namespace,
    verify_provenance_compatibility,
    write_provenance_sidecar,
)


@dataclass(frozen=True)
class HybridTeacherRow:
    """One ranker-compatible hybrid teacher row.

    ``source_scores`` stores raw per-source TE deltas and ``source_weights``
    stores the weights used in the weighted mean. Conversion is ``O(S)`` for
    ``S`` contributing sources.
    """

    transcript_id: str
    task_id: str
    op: str
    pos: int
    nt: str
    teacher_score: float
    source_scores: Mapping[str, float]
    source_weights: Mapping[str, float]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping. Complexity is ``O(S)``."""
        payload = dict(asdict(self))
        payload["source_labels"] = sorted(self.source_scores)
        return payload


@dataclass
class _HybridAccumulator:
    transcript_id: str
    task_id: str
    op: str
    pos: int
    nt: str
    source_scores: dict[str, float] = field(default_factory=dict)
    source_weights: dict[str, float] = field(default_factory=dict)

    def add(self, label: str, score: float, weight: float) -> None:
        """Add or replace one source score.

        If the same source contributes the same candidate more than once, the
        value with larger absolute teacher magnitude is retained. This avoids
        accidental duplicate rows from overweighting one source. Complexity is
        ``O(1)``.
        """
        if label in self.source_scores and abs(self.source_scores[label]) >= abs(score):
            return
        self.source_scores[label] = float(score)
        self.source_weights[label] = float(weight)

    def to_row(self) -> HybridTeacherRow:
        """Convert accumulated source scores into a weighted hybrid row."""
        weighted = 0.0
        total = 0.0
        for label, score in self.source_scores.items():
            weight = max(0.0, float(self.source_weights.get(label, 0.0)))
            weighted += weight * float(score)
            total += weight
        if total <= 0.0:
            raise ValueError("hybrid teacher row has non-positive total source weight")
        score = weighted / total
        return HybridTeacherRow(
            transcript_id=self.transcript_id,
            task_id=self.task_id,
            op=self.op,
            pos=self.pos,
            nt=self.nt,
            teacher_score=float(score),
            source_scores=dict(sorted(self.source_scores.items())),
            source_weights=dict(sorted(self.source_weights.items())),
        )


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _row_key(row: Mapping[str, object]) -> tuple[str, str, str, int, str]:
    """Return the candidate identity key expected by MEF proposal ranking.

    Complexity is ``O(length_of_string_fields)``.
    """
    return (
        str(row["transcript_id"]),
        str(row.get("task_id", "T5")).upper(),
        str(row["op"]).lower(),
        int(row["pos"]),
        str(row.get("nt", "")),
    )


def _read_teacher_jsonl(
    path: str,
    *,
    label: str,
    weight: float,
    accumulators: dict[tuple[str, str, str, int, str], _HybridAccumulator],
) -> dict[str, int]:
    """Read one teacher JSONL into ``accumulators``.

    Invalid/non-finite rows are skipped and counted. Complexity is linear in
    file rows.
    """
    stats = {"rows": 0, "kept": 0, "skipped": 0}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            stats["rows"] += 1
            try:
                row = json.loads(line)
                key = _row_key(row)
                score = _as_float(row.get("teacher_score"), default=float("nan"))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                stats["skipped"] += 1
                continue
            if not math.isfinite(score):
                stats["skipped"] += 1
                continue
            if key not in accumulators:
                tid, task_id, op, pos, nt = key
                accumulators[key] = _HybridAccumulator(
                    transcript_id=tid,
                    task_id=task_id,
                    op=op,
                    pos=pos,
                    nt=nt,
                )
            accumulators[key].add(label, score, weight)
            stats["kept"] += 1
    return stats


def _cap_rows(rows: Sequence[HybridTeacherRow], max_rows: int) -> list[HybridTeacherRow]:
    """Cap one transcript's rows by alternating teacher extremes.

    Keeping only high-score rows would remove negative comparisons needed by
    Bradley-Terry loss. Alternating best/worst candidates preserves both
    ranking directions. Complexity is ``O(R log R)`` for ``R`` rows.
    """
    cap = int(max_rows)
    ordered = sorted(rows, key=lambda row: (-row.teacher_score, row.op, row.pos, row.nt))
    if cap <= 0 or len(ordered) <= cap:
        return ordered
    selected: list[HybridTeacherRow] = []
    lo = 0
    hi = len(ordered) - 1
    while len(selected) < cap and lo <= hi:
        selected.append(ordered[lo])
        lo += 1
        if len(selected) >= cap or lo > hi:
            break
        selected.append(ordered[hi])
        hi -= 1
    return selected


def _summarize(rows_by_record: Mapping[str, Sequence[HybridTeacherRow]], source_stats: Mapping[str, Mapping[str, int]]) -> dict[str, object]:
    """Summarize the hybrid teacher export. Complexity is ``O(U)``."""
    groups = [list(rows) for rows in rows_by_record.values()]
    non_empty = [rows for rows in groups if rows]
    all_rows = [row for rows in non_empty for row in rows]
    if not all_rows:
        return {
            "n_records": len(groups),
            "n_records_with_rows": 0,
            "n_rows": 0,
            "source_stats": dict(source_stats),
        }
    best = [max(rows, key=lambda row: row.teacher_score) for rows in non_empty]
    worst = [min(rows, key=lambda row: row.teacher_score) for rows in non_empty]
    source_counts: dict[str, int] = {}
    overlap_count = 0
    for row in all_rows:
        if len(row.source_scores) > 1:
            overlap_count += 1
        for label in row.source_scores:
            source_counts[label] = source_counts.get(label, 0) + 1
    return {
        "n_records": len(groups),
        "n_records_with_rows": len(non_empty),
        "n_rows": len(all_rows),
        "mean_rows_per_record": float(len(all_rows) / max(1, len(non_empty))),
        "mean_best_teacher_score": float(sum(row.teacher_score for row in best) / len(best)),
        "mean_worst_teacher_score": float(sum(row.teacher_score for row in worst) / len(worst)),
        "mean_abs_teacher_score": float(sum(abs(row.teacher_score) for row in all_rows) / len(all_rows)),
        "max_teacher_score": float(max(row.teacher_score for row in all_rows)),
        "min_teacher_score": float(min(row.teacher_score for row in all_rows)),
        "overlap_rows": int(overlap_count),
        "source_counts": dict(sorted(source_counts.items())),
        "source_stats": dict(source_stats),
    }


def export_hybrid_teacher_jsonl(
    *,
    full_jsonl: str,
    utr_jsonl: str,
    out_jsonl: str,
    out_json: str,
    full_weight: float = 1.0,
    utr_weight: float = 1.0,
    max_rows_per_record: int = 512,
    run_mode: str = "development",
    split_contract: Optional[VerifiedSplitContract] = None,
    split_role: Optional[str] = None,
) -> dict[str, object]:
    """Merge full-pool and UTR teacher rows into one JSONL artifact.

    The output is directly consumable by ``train_proposal_ranker.py``. Complexity
    is ``O(R log R)`` due to per-transcript deterministic capping.
    """
    run_mode = normalize_run_mode(run_mode)
    validate_output_namespace(out_jsonl, run_mode)
    validate_output_namespace(out_json, run_mode)
    data_provenance = upstream_data_provenance(
        (full_jsonl, utr_jsonl), run_mode=run_mode, require_same_role=True
    )
    if run_mode == "paper":
        if split_contract is None or split_role != "train":
            raise ValueError("paper hybrid teacher requires VerifiedSplitContract train role")
        verify_provenance_compatibility(
            build_split_provenance(split_contract, "train"),
            data_provenance,
            require_same_role=True,
        )
    scientific_validity = build_run_metadata(
        run_mode=run_mode,
        data_provenance=data_provenance,
        config={
            "full_weight": full_weight,
            "utr_weight": utr_weight,
            "max_rows_per_record": max_rows_per_record,
        },
        code_paths=(__file__,),
        oracle=(
            data_provenance.get("oracle")
            if isinstance(data_provenance.get("oracle"), Mapping) else None
        ),
        upstream={
            "full_jsonl_sha256": sha256_file(full_jsonl),
            "utr_jsonl_sha256": sha256_file(utr_jsonl),
        },
        functional_claim=True,
    )
    accumulators: dict[tuple[str, str, str, int, str], _HybridAccumulator] = {}
    source_stats = {
        "full": _read_teacher_jsonl(
            full_jsonl,
            label="full",
            weight=float(full_weight),
            accumulators=accumulators,
        ),
        "utr": _read_teacher_jsonl(
            utr_jsonl,
            label="utr",
            weight=float(utr_weight),
            accumulators=accumulators,
        ),
    }
    grouped: dict[str, list[HybridTeacherRow]] = {}
    for acc in accumulators.values():
        row = acc.to_row()
        grouped.setdefault(row.transcript_id, []).append(row)
    capped = {
        tid: _cap_rows(rows, int(max_rows_per_record))
        for tid, rows in sorted(grouped.items())
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)), exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for rows in capped.values():
            for row in rows:
                fh.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")
    payload = {
        "config": {
            "full_jsonl": full_jsonl,
            "utr_jsonl": utr_jsonl,
            "full_weight": float(full_weight),
            "utr_weight": float(utr_weight),
            "max_rows_per_record": int(max_rows_per_record),
        },
        "out_jsonl": out_jsonl,
        "summary": _summarize(capped, source_stats),
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
    parser.add_argument("--full-jsonl", required=True)
    parser.add_argument("--utr-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--full-weight", type=float, default=1.0)
    parser.add_argument("--utr-weight", type=float, default=1.0)
    parser.add_argument("--max-rows-per-record", type=int, default=512)
    parser.add_argument("--run-mode", choices=("development", "paper"), default="development")
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--split-role", choices=("train", "val", "test"), default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Complexity is dominated by JSONL streaming."""
    args = _parse_args(argv)
    require_paper_cli_inputs(
        run_mode=args.run_mode,
        split_manifest=args.split_manifest,
        split_role=args.split_role,
        allowed_roles=("train",),
    )
    payload = export_hybrid_teacher_jsonl(
        full_jsonl=args.full_jsonl,
        utr_jsonl=args.utr_jsonl,
        out_jsonl=args.out_jsonl,
        out_json=args.out_json,
        full_weight=args.full_weight,
        utr_weight=args.utr_weight,
        max_rows_per_record=args.max_rows_per_record,
        run_mode=args.run_mode,
        split_contract=(
            load_and_verify_split_manifest(args.split_manifest)
            if args.split_manifest else None
        ),
        split_role=args.split_role,
    )
    print(json.dumps({"out_json": args.out_json, "out_jsonl": args.out_jsonl, "summary": payload["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HybridTeacherRow",
    "export_hybrid_teacher_jsonl",
    "main",
]
