"""Mine cascade win/loss transcripts into a ranker-compatible teacher JSONL.

The cascade error analysis identifies where recall-then-precision decoding helps
or hurts. This module turns that diagnostic signal into the next teacher
artifact for proposal-ranker v2.

For a transcript ``r`` with mean cascade gain

``g_r = mean_s[(TE_cascade(r,s)-TE_source(r)) - (TE_base(r,s)-TE_source(r))]``,

we assign one of three groups:

* ``cascade_rescue`` when ``g_r`` is sufficiently positive or win fraction is
  high. These rows emphasize UTR/local rescue signals.
* ``cascade_precision`` when ``g_r`` is sufficiently negative or loss fraction
  is high. These rows emphasize the full-pool precision signal that cascade
  should not override.
* ``neutral`` otherwise.

The output keeps the standard ``train_proposal_ranker`` fields and augments
``source_scores`` with a group-specific label. With
``--pair-source-mode source_balanced``, the trainer will allocate pair budget to
these labels directly. Complexity is ``O(R log R)`` for ``R`` teacher rows due
to deterministic per-transcript capping.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
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
class CascadeTranscriptSignal:
    """Cascade win/loss summary for one transcript."""

    transcript_id: str
    cascade_gain_mean: float
    cascade_win_fraction: float
    cascade_loss_fraction: float
    source_te: float
    source_uaug_presence: int
    source_kozak_score: float
    source_gc: float
    five_utr_len: int
    group: str


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _load_cascade_signals(
    path: str,
    *,
    rescue_gain_threshold: float,
    precision_loss_threshold: float,
    min_win_fraction: float,
    min_loss_fraction: float,
    uaug_rescue_bonus: bool,
) -> dict[str, CascadeTranscriptSignal]:
    """Load transcript-level cascade diagnostic rows and assign groups.

    The grouping rule is deterministic:

    ``rescue`` if ``gain >= rescue_threshold`` or, when enabled, the source has
    uAUG and ``gain > 0`` with high win fraction.

    ``precision`` if ``gain <= precision_loss_threshold`` or high loss fraction
    with negative gain.

    Complexity is ``O(N)`` for ``N`` analysis rows.
    """
    signals: dict[str, CascadeTranscriptSignal] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                continue
            tid = str(row["transcript_id"])
            gain = _as_float(row.get("cascade_gain_mean"))
            win_frac = _as_float(row.get("cascade_win_fraction"))
            loss_frac = _as_float(row.get("cascade_loss_fraction"))
            uaug = _as_int(row.get("source_uaug_presence"))
            group = "neutral"
            rescue_by_uaug = bool(uaug_rescue_bonus and uaug and gain > 0.0 and win_frac >= min_win_fraction)
            if gain >= rescue_gain_threshold or rescue_by_uaug:
                group = "cascade_rescue"
            elif gain <= precision_loss_threshold or (gain < 0.0 and loss_frac >= min_loss_fraction):
                group = "cascade_precision"
            signals[tid] = CascadeTranscriptSignal(
                transcript_id=tid,
                cascade_gain_mean=gain,
                cascade_win_fraction=win_frac,
                cascade_loss_fraction=loss_frac,
                source_te=_as_float(row.get("source_te")),
                source_uaug_presence=uaug,
                source_kozak_score=_as_float(row.get("source_kozak_score")),
                source_gc=_as_float(row.get("source_gc")),
                five_utr_len=_as_int(row.get("five_utr_len")),
                group=group,
            )
    return signals


def _row_score(row: Mapping[str, object], source_label: Optional[str] = None) -> float:
    source_scores = row.get("source_scores", {})
    if source_label and isinstance(source_scores, Mapping) and source_label in source_scores:
        return _as_float(source_scores[source_label])
    return _as_float(row.get("teacher_score"))


def _row_identity(row: Mapping[str, object]) -> tuple[str, int, str]:
    return (str(row.get("op", "")), int(row.get("pos", -1)), str(row.get("nt", "")))


def _cap_rows(rows: Sequence[dict[str, object]], cap: int, source_label: Optional[str] = None) -> list[dict[str, object]]:
    """Keep high/low teacher extremes for one transcript.

    Alternating extremes preserves positive and negative Bradley-Terry
    comparisons. Complexity is ``O(R log R)`` for ``R`` rows.
    """
    if cap <= 0 or len(rows) <= cap:
        return sorted(rows, key=lambda row: (-_row_score(row, source_label), _row_identity(row)))
    ordered = sorted(rows, key=lambda row: (-_row_score(row, source_label), _row_identity(row)))
    selected: list[dict[str, object]] = []
    lo = 0
    hi = len(ordered) - 1
    seen: set[tuple[str, int, str]] = set()
    while len(selected) < cap and lo <= hi:
        for idx in (lo, hi):
            row = ordered[idx]
            key = _row_identity(row)
            if key not in seen:
                selected.append(row)
                seen.add(key)
                if len(selected) >= cap:
                    break
        lo += 1
        hi -= 1
    return selected


def _augment_row(row: Mapping[str, object], signal: Optional[CascadeTranscriptSignal]) -> dict[str, object]:
    """Add cascade group metadata and source-balanced labels to one row."""
    out = dict(row)
    teacher_score = _as_float(out.get("teacher_score"))
    source_scores = out.get("source_scores", {})
    scores: dict[str, float] = {}
    if isinstance(source_scores, Mapping):
        for key, value in source_scores.items():
            score = _as_float(value, default=float("nan"))
            if math.isfinite(score):
                scores[str(key)] = score
    if not scores:
        scores["teacher"] = teacher_score
    source_weights = out.get("source_weights", {})
    weights: dict[str, float] = {}
    if isinstance(source_weights, Mapping):
        for key, value in source_weights.items():
            weight = _as_float(value, default=1.0)
            if math.isfinite(weight) and weight > 0.0:
                weights[str(key)] = weight
    for label in scores:
        weights.setdefault(label, 1.0)

    group = "unmatched"
    if signal is not None:
        group = signal.group
        out["cascade_gain_mean"] = signal.cascade_gain_mean
        out["cascade_win_fraction"] = signal.cascade_win_fraction
        out["cascade_loss_fraction"] = signal.cascade_loss_fraction
        out["source_te_for_cascade_group"] = signal.source_te
        out["source_uaug_presence_for_cascade_group"] = signal.source_uaug_presence
        out["source_kozak_score_for_cascade_group"] = signal.source_kozak_score
        out["source_gc_for_cascade_group"] = signal.source_gc
        out["five_utr_len_for_cascade_group"] = signal.five_utr_len
        if signal.group == "cascade_rescue":
            scores["cascade_rescue"] = scores.get("utr", teacher_score)
            weights["cascade_rescue"] = 1.0
        elif signal.group == "cascade_precision":
            scores["cascade_precision"] = scores.get("full", teacher_score)
            weights["cascade_precision"] = 1.0
    out["cascade_teacher_group"] = group
    out["source_scores"] = dict(sorted(scores.items()))
    out["source_weights"] = dict(sorted(weights.items()))
    out["source_labels"] = sorted(scores)
    return out


def _summarize(rows_by_group: Mapping[str, Sequence[Mapping[str, object]]], signals: Mapping[str, CascadeTranscriptSignal]) -> dict[str, object]:
    """Summarize mined teacher output. Complexity is ``O(R + N)``."""
    group_counts = {group: len(rows) for group, rows in sorted(rows_by_group.items())}
    transcript_groups: dict[str, int] = {}
    for signal in signals.values():
        transcript_groups[signal.group] = transcript_groups.get(signal.group, 0) + 1
    label_counts: dict[str, int] = {}
    for rows in rows_by_group.values():
        for row in rows:
            source_scores = row.get("source_scores", {})
            if isinstance(source_scores, Mapping):
                for label in source_scores:
                    label_counts[str(label)] = label_counts.get(str(label), 0) + 1
    return {
        "n_rows": int(sum(group_counts.values())),
        "row_counts_by_group": group_counts,
        "transcript_counts_by_group": dict(sorted(transcript_groups.items())),
        "source_label_counts": dict(sorted(label_counts.items())),
    }


def mine_cascade_hard_negative_teacher(
    *,
    teacher_jsonl: str,
    cascade_records_jsonl: str,
    out_jsonl: str,
    out_json: str,
    out_md: Optional[str] = None,
    rescue_gain_threshold: float = 0.004,
    precision_loss_threshold: float = -0.004,
    min_win_fraction: float = 0.6,
    min_loss_fraction: float = 0.6,
    rescue_cap_per_record: int = 768,
    precision_cap_per_record: int = 768,
    neutral_cap_per_record: int = 128,
    unmatched_cap_per_record: int = 0,
    uaug_rescue_bonus: bool = True,
    run_mode: str = "development",
    split_contract: Optional[VerifiedSplitContract] = None,
    split_role: Optional[str] = None,
) -> dict[str, object]:
    """Create a source-balanced hard-negative teacher JSONL.

    The miner preserves the base teacher scores and only adds extra
    source-specific labels plus deterministic per-transcript capping. The output
    remains directly consumable by ``train_proposal_ranker.py``.
    """
    run_mode = normalize_run_mode(run_mode)
    validate_output_namespace(out_jsonl, run_mode)
    validate_output_namespace(out_json, run_mode)
    data_provenance = upstream_data_provenance(
        (teacher_jsonl, cascade_records_jsonl),
        run_mode=run_mode,
        require_same_role=True,
    )
    if run_mode == "paper":
        if split_contract is None or split_role != "train":
            raise ValueError("paper cascade teacher requires VerifiedSplitContract train role")
        verify_provenance_compatibility(
            build_split_provenance(split_contract, "train"),
            data_provenance,
            require_same_role=True,
        )
    scientific_validity = build_run_metadata(
        run_mode=run_mode,
        data_provenance=data_provenance,
        config={
            "rescue_gain_threshold": rescue_gain_threshold,
            "precision_loss_threshold": precision_loss_threshold,
            "min_win_fraction": min_win_fraction,
            "min_loss_fraction": min_loss_fraction,
        },
        code_paths=(__file__,),
        oracle=(
            data_provenance.get("oracle")
            if isinstance(data_provenance.get("oracle"), Mapping) else None
        ),
        upstream={
            "teacher_jsonl_sha256": sha256_file(teacher_jsonl),
            "cascade_records_jsonl_sha256": sha256_file(cascade_records_jsonl),
        },
        functional_claim=True,
    )
    signals = _load_cascade_signals(
        cascade_records_jsonl,
        rescue_gain_threshold=float(rescue_gain_threshold),
        precision_loss_threshold=float(precision_loss_threshold),
        min_win_fraction=float(min_win_fraction),
        min_loss_fraction=float(min_loss_fraction),
        uaug_rescue_bonus=bool(uaug_rescue_bonus),
    )
    rows_by_tid: dict[str, list[dict[str, object]]] = {}
    source_rows = 0
    with open(teacher_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            source_rows += 1
            row = json.loads(line)
            if not isinstance(row, Mapping) or "transcript_id" not in row:
                continue
            tid = str(row["transcript_id"])
            rows_by_tid.setdefault(tid, []).append(_augment_row(row, signals.get(tid)))

    selected_by_group: dict[str, list[dict[str, object]]] = {}
    selected_rows: list[dict[str, object]] = []
    caps = {
        "cascade_rescue": int(rescue_cap_per_record),
        "cascade_precision": int(precision_cap_per_record),
        "neutral": int(neutral_cap_per_record),
        "unmatched": int(unmatched_cap_per_record),
    }
    source_for_cap = {
        "cascade_rescue": "cascade_rescue",
        "cascade_precision": "cascade_precision",
        "neutral": None,
        "unmatched": None,
    }
    for tid, rows in sorted(rows_by_tid.items()):
        signal = signals.get(tid)
        group = signal.group if signal is not None else "unmatched"
        cap = caps.get(group, 0)
        if cap <= 0:
            continue
        capped = _cap_rows(rows, cap, source_label=source_for_cap.get(group))
        selected_rows.extend(capped)
        selected_by_group.setdefault(group, []).extend(capped)

    os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)), exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for row in selected_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    summary = _summarize(selected_by_group, signals)
    summary["config"] = {
        "teacher_jsonl": teacher_jsonl,
        "cascade_records_jsonl": cascade_records_jsonl,
        "source_rows": source_rows,
        "rescue_gain_threshold": float(rescue_gain_threshold),
        "precision_loss_threshold": float(precision_loss_threshold),
        "min_win_fraction": float(min_win_fraction),
        "min_loss_fraction": float(min_loss_fraction),
        "rescue_cap_per_record": int(rescue_cap_per_record),
        "precision_cap_per_record": int(precision_cap_per_record),
        "neutral_cap_per_record": int(neutral_cap_per_record),
        "unmatched_cap_per_record": int(unmatched_cap_per_record),
        "uaug_rescue_bonus": bool(uaug_rescue_bonus),
    }
    summary["jsonl_path"] = out_jsonl
    summary["markdown_path"] = out_md
    summary["scientific_validity"] = scientific_validity
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    summary["provenance_sidecar"] = write_provenance_sidecar(
        out_jsonl, scientific_validity
    )
    summary["summary_provenance_sidecar"] = write_provenance_sidecar(
        out_json, scientific_validity
    )
    if out_md:
        write_hard_negative_teacher_markdown(summary, out_md)
        summary["markdown_provenance_sidecar"] = write_provenance_sidecar(
            out_md, scientific_validity
        )
    return summary


def _fmt(value: object) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(x):
        return ""
    if abs(x) >= 100:
        return f"{x:.1f}"
    if abs(x) >= 1:
        return f"{x:.4f}"
    return f"{x:.5f}"


def write_hard_negative_teacher_markdown(summary: Mapping[str, object], path: str) -> str:
    """Write a compact Markdown summary for the mined teacher."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Cascade Hard-Negative Teacher Mining\n\n")
        fh.write("| Section | Key | Value |\n|---|---|---:|\n")
        for section in ("transcript_counts_by_group", "row_counts_by_group", "source_label_counts"):
            payload = summary.get(section, {})
            if not isinstance(payload, Mapping):
                continue
            for key, value in sorted(payload.items()):
                fh.write(f"| `{section}` | `{key}` | {_fmt(value)} |\n")
        cfg = summary.get("config", {})
        if isinstance(cfg, Mapping):
            fh.write("\n## Config\n\n")
            fh.write("| Key | Value |\n|---|---|\n")
            for key, value in sorted(cfg.items()):
                fh.write(f"| `{key}` | `{value}` |\n")
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-jsonl", required=True)
    parser.add_argument("--cascade-records-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--rescue-gain-threshold", type=float, default=0.004)
    parser.add_argument("--precision-loss-threshold", type=float, default=-0.004)
    parser.add_argument("--min-win-fraction", type=float, default=0.6)
    parser.add_argument("--min-loss-fraction", type=float, default=0.6)
    parser.add_argument("--rescue-cap-per-record", type=int, default=768)
    parser.add_argument("--precision-cap-per-record", type=int, default=768)
    parser.add_argument("--neutral-cap-per-record", type=int, default=128)
    parser.add_argument("--unmatched-cap-per-record", type=int, default=0)
    parser.add_argument("--disable-uaug-rescue-bonus", action="store_true")
    parser.add_argument("--run-mode", choices=("development", "paper"), default="development")
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--split-role", choices=("train", "val", "test"), default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    require_paper_cli_inputs(
        run_mode=args.run_mode,
        split_manifest=args.split_manifest,
        split_role=args.split_role,
        allowed_roles=("train",),
    )
    summary = mine_cascade_hard_negative_teacher(
        teacher_jsonl=args.teacher_jsonl,
        cascade_records_jsonl=args.cascade_records_jsonl,
        out_jsonl=args.out_jsonl,
        out_json=args.out_json,
        out_md=args.out_md,
        rescue_gain_threshold=args.rescue_gain_threshold,
        precision_loss_threshold=args.precision_loss_threshold,
        min_win_fraction=args.min_win_fraction,
        min_loss_fraction=args.min_loss_fraction,
        rescue_cap_per_record=args.rescue_cap_per_record,
        precision_cap_per_record=args.precision_cap_per_record,
        neutral_cap_per_record=args.neutral_cap_per_record,
        unmatched_cap_per_record=args.unmatched_cap_per_record,
        uaug_rescue_bonus=not bool(args.disable_uaug_rescue_bonus),
        run_mode=args.run_mode,
        split_contract=(
            load_and_verify_split_manifest(args.split_manifest)
            if args.split_manifest else None
        ),
        split_role=args.split_role,
    )
    print(json.dumps({"jsonl_path": summary["jsonl_path"], "json_path": args.out_json, "markdown_path": args.out_md}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CascadeTranscriptSignal",
    "mine_cascade_hard_negative_teacher",
    "write_hard_negative_teacher_markdown",
    "main",
]
