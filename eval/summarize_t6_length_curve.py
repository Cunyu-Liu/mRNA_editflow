"""Summarize T6 length-control multiseed curves.

This module is read-only with respect to benchmark inputs. It scans completed
``run_multiseed_benchmark`` summaries for fixed target-length deltas and writes
a compact JSON/Markdown report suitable for the T1-T7 evidence ledger.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
from typing import Mapping, Optional, Sequence


CLAIM_POLICY = (
    "Length-control curves are proxy/offline benchmarks. Do not claim wet-lab "
    "performance or full de novo SOTA from these results."
)
DEFAULT_CHECKPOINT = "ckpts/stage_a_public_full_10k_bs8ga4_seed0/stage_a_best.pt"
DEFAULT_DELTAS: tuple[int, ...] = (-30, -15, 0, 15, 30)
DEFAULT_SLICES: tuple[str, ...] = ("head256", "head1024")
SUMMARY_METRICS: tuple[str, ...] = (
    "mean_abs_length_error",
    "legal_fraction",
    "mean_protein_identity",
    "within_budget_fraction",
    "reading_frame_intact_fraction",
    "delta_oracle_te_vs_source",
    "mean_oracle_te",
    "mean_edit_distance",
)
EXPECTED_T6_SEEDS: tuple[int, ...] = tuple(range(10))


def _delta_name(delta: int) -> str:
    return f"neg{abs(delta)}" if delta < 0 else f"pos{delta}"


def _summary_path(project_root: str, slice_name: str, delta: int, top_k: int) -> str:
    return os.path.join(
        project_root,
        "benchmark",
        f"multiseed_t6_public_{slice_name}_stagea10k_len_{_delta_name(delta)}_top{top_k}",
        "multiseed_summary.json",
    )


def _summary_candidates(project_root: str, slice_name: str, delta: int, top_k: int) -> list[str]:
    """Return standard then non-standard completed summary candidates."""
    standard = _summary_path(project_root, slice_name, delta, top_k)
    pattern = os.path.join(
        project_root,
        "benchmark",
        f"multiseed_t6_public_{slice_name}_stagea10k_len_{_delta_name(delta)}_*_top{top_k}",
        "multiseed_summary.json",
    )
    candidates = [standard]
    for path in sorted(glob.glob(pattern)):
        if path != standard:
            candidates.append(path)
    return candidates


def _completed_summary_path(
    project_root: str,
    slice_name: str,
    delta: int,
    top_k: int,
) -> Optional[str]:
    for path in _summary_candidates(project_root, slice_name, delta, top_k):
        if os.path.exists(path) and _is_complete_summary_path(path):
            return path
    return None


def _rel(path: str, root: str) -> str:
    return os.path.relpath(path, root) if os.path.isabs(path) else path


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str) -> Mapping[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _summary_has_expected_seeds(
    summary: Mapping[str, object],
    expected_seeds: Sequence[int] = EXPECTED_T6_SEEDS,
) -> bool:
    """Return whether a multiseed summary satisfies the full 10-seed protocol."""
    expected = [int(seed) for seed in expected_seeds]
    config = summary.get("config", {})
    if not isinstance(config, Mapping):
        return False
    seeds = config.get("seeds")
    if not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)):
        return False
    try:
        if sorted(int(seed) for seed in seeds) != expected:
            return False
    except (TypeError, ValueError):
        return False

    per_seed = summary.get("per_seed")
    if not isinstance(per_seed, Sequence) or isinstance(per_seed, (str, bytes)):
        return False
    found = []
    for row in per_seed:
        if not isinstance(row, Mapping):
            return False
        seed = row.get("seed")
        try:
            found.append(int(seed))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
    if sorted(found) != expected:
        return False

    aggregate = summary.get("aggregate", {})
    if not isinstance(aggregate, Mapping):
        return False
    min_n = len(expected)
    for metric in SUMMARY_METRICS:
        entry = aggregate.get(metric, {})
        if not isinstance(entry, Mapping):
            return False
        n_value = entry.get("n")
        if not isinstance(n_value, (int, float)) or int(n_value) < min_n:
            return False
    return True


def _is_complete_summary_path(path: str) -> bool:
    try:
        summary = _load_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return _summary_has_expected_seeds(summary)


def _metric_mean(summary: Mapping[str, object], metric: str) -> float:
    aggregate = summary.get("aggregate", {})
    if not isinstance(aggregate, Mapping):
        raise ValueError("summary aggregate must be a JSON object")
    entry = aggregate.get(metric, {})
    if not isinstance(entry, Mapping):
        raise ValueError(f"missing aggregate metric {metric!r}")
    value = entry.get("mean")
    if not isinstance(value, (int, float)):
        raise ValueError(f"aggregate metric {metric!r} has no numeric mean")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"aggregate metric {metric!r} mean is not finite")
    return number


def _row_from_summary(path: str, project_root: str, delta: int) -> dict[str, object]:
    summary = _load_json(path)
    row: dict[str, object] = {
        "target_length_delta": int(delta),
        "summary_path": _rel(path, project_root),
        "summary_sha256": _sha256_file(path),
    }
    for metric in SUMMARY_METRICS:
        row[metric] = _metric_mean(summary, metric)
    return row


def _parse_running_delta(value: str) -> tuple[str, int]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("running delta must be formatted as SLICE:DELTA")
    slice_name, delta_text = value.split(":", 1)
    if not slice_name:
        raise argparse.ArgumentTypeError("running delta slice cannot be empty")
    try:
        delta = int(delta_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid running delta: {delta_text}") from exc
    return slice_name, delta


def summarize_slice(
    *,
    project_root: str,
    slice_name: str,
    target_length_deltas: Sequence[int] = DEFAULT_DELTAS,
    checkpoint: str = DEFAULT_CHECKPOINT,
    task_id: str = "T6",
    edit_budget: int = 30,
    top_k: int = 64,
    running_log: Optional[str] = None,
    running_target_length_deltas: Sequence[int] = (),
) -> dict[str, object]:
    """Summarize completed rows for one slice and mark missing deltas."""
    rows = []
    pending = []
    for delta in target_length_deltas:
        path = _completed_summary_path(project_root, slice_name, int(delta), top_k)
        if path is not None:
            rows.append(_row_from_summary(path, project_root, int(delta)))
        else:
            pending.append(int(delta))

    running_set = {int(delta) for delta in running_target_length_deltas}
    if not pending:
        status = "complete"
    elif running_set:
        status = "running"
    elif rows:
        status = "partial"
    else:
        status = "pending"

    return {
        "slice": slice_name,
        "status": status,
        "task_id": task_id,
        "checkpoint": checkpoint,
        "edit_budget": int(edit_budget),
        "top_k": int(top_k),
        "rows": rows,
        "pending_target_length_deltas": pending,
        "running_target_length_deltas": sorted(running_set),
        "running_log": running_log,
    }


def build_t6_length_curve_report(
    *,
    project_root: str,
    slices: Sequence[str] = DEFAULT_SLICES,
    target_length_deltas: Sequence[int] = DEFAULT_DELTAS,
    checkpoint: str = DEFAULT_CHECKPOINT,
    task_id: str = "T6",
    edit_budget: int = 30,
    top_k: int = 64,
    running_logs: Optional[Mapping[str, str]] = None,
    running_deltas: Optional[Mapping[str, Sequence[int]]] = None,
) -> dict[str, object]:
    """Build the full T6 length-control report payload."""
    running_logs = running_logs or {}
    running_deltas = running_deltas or {}
    payload: dict[str, object] = {
        "artifact_kind": "t6_length_curve_report",
        "claim_policy": CLAIM_POLICY,
    }
    for slice_name in slices:
        payload[f"{slice_name}_stagea10k"] = summarize_slice(
            project_root=project_root,
            slice_name=slice_name,
            target_length_deltas=target_length_deltas,
            checkpoint=checkpoint,
            task_id=task_id,
            edit_budget=edit_budget,
            top_k=top_k,
            running_log=running_logs.get(slice_name),
            running_target_length_deltas=running_deltas.get(slice_name, ()),
        )
    return payload


def write_report_json(payload: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path


def _fmt(value: object, digits: int = 5) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return "pending"


def _fmt_delta(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):+.5f}"
    return "pending"


def _constraint_text(row: Mapping[str, object]) -> str:
    metrics = (
        row.get("legal_fraction"),
        row.get("mean_protein_identity"),
        row.get("within_budget_fraction"),
        row.get("reading_frame_intact_fraction"),
    )
    if all(isinstance(value, (int, float)) and abs(float(value) - 1.0) <= 1e-12 for value in metrics):
        return "legal/protein/budget/frame = 1.0"
    return "check JSON"


def _section_title(key: str) -> str:
    slice_name = key.removesuffix("_stagea10k")
    return f"{slice_name.capitalize()} Stage A 10k"


def _write_section(lines: list[str], key: str, section: Mapping[str, object]) -> None:
    rows = section.get("rows", [])
    if not isinstance(rows, Sequence):
        rows = []
    pending = section.get("pending_target_length_deltas", [])
    if not isinstance(pending, Sequence):
        pending = []
    running = section.get("running_target_length_deltas", [])
    running_set = {int(delta) for delta in running if isinstance(delta, int)}

    lines.extend(
        [
            f"## {_section_title(key)}",
            "",
            "| target length delta | mean abs length error | delta_oracle_te_vs_source | mean_oracle_te | mean_edit_distance | constraints |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    indexed = {
        int(row["target_length_delta"]): row
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("target_length_delta"), int)
    }
    all_deltas = sorted(set(indexed) | {int(delta) for delta in pending if isinstance(delta, int)})
    for delta in all_deltas:
        row = indexed.get(delta)
        if row is None:
            state = "running" if delta in running_set else "pending"
            lines.append(f"| {delta:+d} | pending | pending | pending | pending | {state} |")
            continue
        lines.append(
            "| {delta:+d} | {length_error} | {delta_te} | {mean_te} | {edit_distance} | {constraints} |".format(
                delta=delta,
                length_error=_fmt(row.get("mean_abs_length_error")),
                delta_te=_fmt_delta(row.get("delta_oracle_te_vs_source")),
                mean_te=_fmt(row.get("mean_oracle_te")),
                edit_distance=_fmt(row.get("mean_edit_distance")),
                constraints=_constraint_text(row),
            )
        )
    lines.append("")


def write_report_markdown(payload: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = [
        "# T6 Length-Control Curve Report",
        "",
        "- Claim policy: offline proxy benchmark; not wet-lab performance or full de novo SOTA.",
    ]
    first_section = next(
        (value for key, value in payload.items() if key.endswith("_stagea10k") and isinstance(value, Mapping)),
        None,
    )
    if first_section:
        lines.extend(
            [
                f"- Checkpoint: `{first_section.get('checkpoint')}`",
                f"- Task: `{first_section.get('task_id')}`",
                f"- Edit budget: `{first_section.get('edit_budget')}`",
            ]
        )
    lines.append("")
    for key in (k for k in payload if k.endswith("_stagea10k")):
        section = payload.get(key, {})
        if isinstance(section, Mapping):
            _write_section(lines, key, section)

    sha_rows = []
    for key in (k for k in payload if k.endswith("_stagea10k")):
        section = payload.get(key, {})
        if not isinstance(section, Mapping):
            continue
        rows = section.get("rows", [])
        if not isinstance(rows, Sequence):
            continue
        for row in rows:
            if isinstance(row, Mapping):
                sha_rows.append(
                    (
                        str(section.get("slice")),
                        int(row.get("target_length_delta", 0)),
                        str(row.get("summary_sha256")),
                    )
                )
    if sha_rows:
        lines.extend(
            [
                "## Source Summary SHAs",
                "",
                "| slice | target length delta | summary SHA-256 |",
                "|---|---:|---|",
            ]
        )
        for slice_name, delta, digest in sha_rows:
            lines.append(f"| {slice_name} | {delta:+d} | `{digest}` |")
        lines.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines).rstrip() + "\n")
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out-json", default="benchmark/t6_length_curve_report_head256_head1024.json")
    parser.add_argument("--out-md", default="benchmark/t6_length_curve_report_head256_head1024.md")
    parser.add_argument("--slices", nargs="+", default=list(DEFAULT_SLICES))
    parser.add_argument("--target-length-deltas", nargs="+", type=int, default=list(DEFAULT_DELTAS))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--task-id", default="T6")
    parser.add_argument("--edit-budget", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument(
        "--running-log",
        action="append",
        default=[],
        metavar="SLICE:PATH",
        help="Optional running log annotation for a slice.",
    )
    parser.add_argument(
        "--running-delta",
        action="append",
        type=_parse_running_delta,
        default=[],
        metavar="SLICE:DELTA",
        help="Mark a missing target-length delta as currently running.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    project_root = os.path.abspath(args.project_root)

    running_logs: dict[str, str] = {}
    for item in args.running_log:
        if ":" not in item:
            raise SystemExit(f"--running-log must be formatted as SLICE:PATH, got {item!r}")
        slice_name, path = item.split(":", 1)
        running_logs[slice_name] = path

    running_deltas: dict[str, list[int]] = {}
    for slice_name, delta in args.running_delta:
        running_deltas.setdefault(slice_name, []).append(delta)

    payload = build_t6_length_curve_report(
        project_root=project_root,
        slices=args.slices,
        target_length_deltas=args.target_length_deltas,
        checkpoint=args.checkpoint,
        task_id=args.task_id,
        edit_budget=args.edit_budget,
        top_k=args.top_k,
        running_logs=running_logs,
        running_deltas=running_deltas,
    )
    out_json = args.out_json
    out_md = args.out_md
    if not os.path.isabs(out_json):
        out_json = os.path.join(project_root, out_json)
    if not os.path.isabs(out_md):
        out_md = os.path.join(project_root, out_md)
    write_report_json(payload, out_json)
    write_report_markdown(payload, out_md)
    print(json.dumps({"out_json": out_json, "out_md": out_md}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
