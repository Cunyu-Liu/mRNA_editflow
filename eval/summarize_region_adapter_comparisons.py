"""Summarize region-adapter comparison artifacts into a decision report.

The region-adapter eval writes several paired comparison JSON files:

* against the prior hard-negative v2 champion;
* optionally against current multi-objective baselines.

This module joins those files into one compact JSON/Markdown report so the
result can be audited without manually cross-reading multiple compare tables.
It is read-only and tolerates missing optional comparison files.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Mapping, Optional, Sequence


CONSTRAINT_METRICS: tuple[str, ...] = (
    "legal_fraction",
    "mean_protein_identity",
    "within_budget_fraction",
    "reading_frame_intact_fraction",
)
PRIMARY_METRIC = "delta_oracle_te_vs_source"
STRICT_ALPHA = 0.05
BORDERLINE_ALPHA = 0.10
CONSTRAINT_EXACT_TOL = 1e-12


def _load_json(path: str) -> Mapping[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _compare_path(project_root: str, baseline_label: str, slice_name: str) -> str:
    return os.path.join(
        project_root,
        "benchmark",
        f"region_adapter_vs_{baseline_label}_{slice_name}.json",
    )


def _baseline_labels(top_k: int) -> tuple[str, ...]:
    return (
        f"hardneg_v2_top{top_k}",
        f"mo_grpo_top{top_k}",
        f"mo_scalar_top{top_k}",
        f"mo_pareto_top{top_k}",
        f"mo_te_only_top{top_k}",
    )


def _finite_float(value: object) -> Optional[float]:
    if not isinstance(value, (int, float)):
        return None
    x = float(value)
    return x if math.isfinite(x) else None


def _constraint_ok(metrics: Mapping[str, object], threshold: float = 0.999) -> bool:
    values = [_finite_float(metrics.get(metric)) for metric in CONSTRAINT_METRICS]
    return all(value is not None and value >= threshold for value in values)


def _constraint_exact_1(metrics: Mapping[str, object]) -> bool:
    values = [_finite_float(metrics.get(metric)) for metric in CONSTRAINT_METRICS]
    return all(value is not None and abs(value - 1.0) <= CONSTRAINT_EXACT_TOL for value in values)


def _primary_row(comparisons: Mapping[str, object], baseline_label: str) -> Mapping[str, object]:
    baseline = comparisons.get(baseline_label, {})
    if not isinstance(baseline, Mapping):
        return {}
    row = baseline.get(PRIMARY_METRIC, {})
    return row if isinstance(row, Mapping) else {}


def _primary_signal(delta: object, paired_p: object) -> str:
    """Classify the primary TE delta without turning trends into claims."""
    delta_x = _finite_float(delta)
    p_x = _finite_float(paired_p)
    if delta_x is None or p_x is None:
        return "unverified"
    if delta_x <= 0:
        return "not_positive"
    if p_x < STRICT_ALPHA:
        return "strict_positive"
    if p_x < BORDERLINE_ALPHA:
        return "borderline_positive"
    return "positive_not_significant"


def summarize_region_adapter_comparisons(
    *,
    project_root: str,
    slice_name: str = "head256",
    top_k: int = 64,
    baselines: Optional[Sequence[str]] = None,
    out_json: Optional[str] = None,
    out_md: Optional[str] = None,
) -> dict[str, object]:
    """Join region-adapter compare JSON files into one decision report."""
    baseline_labels = tuple(baselines) if baselines is not None else _baseline_labels(top_k)
    by_run: dict[str, dict[str, object]] = {}
    compare_files: list[dict[str, object]] = []

    for baseline_label in baseline_labels:
        path = _compare_path(project_root, baseline_label, slice_name)
        rel_path = os.path.relpath(path, project_root)
        exists = os.path.exists(path)
        compare_files.append({"baseline": baseline_label, "path": rel_path, "exists": exists})
        if not exists:
            continue
        payload = _load_json(path)
        for row in payload.get("rows", []):
            if not isinstance(row, Mapping):
                continue
            run = str(row.get("run", ""))
            metric = str(row.get("metric", ""))
            if not run or not metric:
                continue
            entry = by_run.setdefault(
                run,
                {
                    "run": run,
                    "constraints": {},
                    "constraints_ok": False,
                    "constraints_exact_1": False,
                    "comparisons": {},
                },
            )
            run_mean = _finite_float(row.get("run_mean"))
            if metric in CONSTRAINT_METRICS and run_mean is not None:
                entry["constraints"][metric] = run_mean  # type: ignore[index]
            comp = entry["comparisons"].setdefault(baseline_label, {})  # type: ignore[index]
            comp[metric] = {
                "baseline_mean": row.get("baseline_mean"),
                "run_mean": row.get("run_mean"),
                "delta": row.get("delta"),
                "improvement": row.get("improvement"),
                "paired_p": row.get("paired_p"),
                "ci_low": row.get("ci_low"),
                "ci_high": row.get("ci_high"),
                "n_paired_seeds": row.get("n_paired_seeds"),
                "higher_is_better": row.get("higher_is_better"),
            }

    runs = []
    for run, entry in sorted(by_run.items()):
        constraints = entry.get("constraints", {})
        if isinstance(constraints, Mapping):
            entry["constraints_ok"] = _constraint_ok(constraints)
            entry["constraints_exact_1"] = _constraint_exact_1(constraints)
        comparisons = entry.get("comparisons", {})
        if not isinstance(comparisons, Mapping):
            comparisons = {}
        primary_by_baseline: dict[str, dict[str, object]] = {}
        for baseline_label in baseline_labels:
            primary = _primary_row(comparisons, baseline_label)
            if primary:
                delta = primary.get("delta")
                paired_p = primary.get("paired_p")
                primary_by_baseline[baseline_label] = {
                    "delta": delta,
                    "paired_p": paired_p,
                    "run_mean": primary.get("run_mean"),
                    "baseline_mean": primary.get("baseline_mean"),
                    "signal": _primary_signal(delta, paired_p),
                }
        entry["primary_by_baseline"] = primary_by_baseline
        primary = _primary_row(comparisons, f"hardneg_v2_top{top_k}")
        entry["primary_vs_hardneg_delta"] = (
            primary.get("delta") if isinstance(primary, Mapping) else None
        )
        entry["primary_vs_hardneg_paired_p"] = (
            primary.get("paired_p") if isinstance(primary, Mapping) else None
        )
        runs.append(entry)

    complete_files = [row for row in compare_files if row["exists"]]
    missing_compare_files = [row for row in compare_files if not row["exists"]]
    hardneg_baseline = f"hardneg_v2_top{top_k}"
    all_runs_constraint_ok = bool(runs) and all(bool(row.get("constraints_ok")) for row in runs)
    all_runs_constraints_exact_1 = bool(runs) and all(
        bool(row.get("constraints_exact_1")) for row in runs
    )
    hardneg_rows = [
        row
        for row in runs
        if isinstance(row.get("comparisons"), Mapping)
        and hardneg_baseline in row["comparisons"]  # type: ignore[operator]
    ]
    best_vs_hardneg = None
    if hardneg_rows:
        best_vs_hardneg = max(
            hardneg_rows,
            key=lambda row: float(row.get("primary_vs_hardneg_delta") or float("-inf")),
        ).get("run")
    best_run_by_baseline: dict[str, Optional[str]] = {}
    best_strict_positive_run_by_baseline: dict[str, Optional[str]] = {}
    best_borderline_positive_run_by_baseline: dict[str, Optional[str]] = {}
    for baseline_label in baseline_labels:
        rows_with_baseline = [
            row
            for row in runs
            if isinstance(row.get("primary_by_baseline"), Mapping)
            and baseline_label in row["primary_by_baseline"]  # type: ignore[operator]
        ]
        if rows_with_baseline:
            best_run_by_baseline[baseline_label] = max(
                rows_with_baseline,
                key=lambda row: float(
                    row["primary_by_baseline"][baseline_label].get("delta")  # type: ignore[index]
                    or float("-inf")
                ),
            ).get("run")  # type: ignore[assignment]
        else:
            best_run_by_baseline[baseline_label] = None
        strict_rows = [
            row
            for row in rows_with_baseline
            if row["primary_by_baseline"][baseline_label].get("signal") == "strict_positive"  # type: ignore[index]
        ]
        borderline_rows = [
            row
            for row in rows_with_baseline
            if row["primary_by_baseline"][baseline_label].get("signal") == "borderline_positive"  # type: ignore[index]
        ]
        if strict_rows:
            best_strict_positive_run_by_baseline[baseline_label] = max(
                strict_rows,
                key=lambda row: float(
                    row["primary_by_baseline"][baseline_label].get("delta")  # type: ignore[index]
                    or float("-inf")
                ),
            ).get("run")  # type: ignore[assignment]
        else:
            best_strict_positive_run_by_baseline[baseline_label] = None
        if borderline_rows:
            best_borderline_positive_run_by_baseline[baseline_label] = max(
                borderline_rows,
                key=lambda row: float(
                    row["primary_by_baseline"][baseline_label].get("delta")  # type: ignore[index]
                    or float("-inf")
                ),
            ).get("run")  # type: ignore[assignment]
        else:
            best_borderline_positive_run_by_baseline[baseline_label] = None
    payload = {
        "artifact_kind": "region_adapter_decision_report",
        "slice": slice_name,
        "top_k": int(top_k),
        "primary_metric": PRIMARY_METRIC,
        "strict_alpha": STRICT_ALPHA,
        "borderline_alpha": BORDERLINE_ALPHA,
        "constraint_exact_tolerance": CONSTRAINT_EXACT_TOL,
        "constraint_metrics": list(CONSTRAINT_METRICS),
        "compare_files": compare_files,
        "summary": {
            "n_compare_files_found": len(complete_files),
            "n_compare_files_expected": len(compare_files),
            "missing_compare_files": missing_compare_files,
            "n_region_adapter_runs": len(runs),
            "all_constraints_ok": all_runs_constraint_ok,
            "all_constraints_exact_1": all_runs_constraints_exact_1,
            "best_run_vs_hardneg": best_vs_hardneg,
            "best_run_by_baseline": best_run_by_baseline,
            "best_strict_positive_run_by_baseline": best_strict_positive_run_by_baseline,
            "best_borderline_positive_run_by_baseline": best_borderline_positive_run_by_baseline,
        },
        "runs": runs,
    }
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
    if out_md:
        write_markdown(payload, out_md)
    return payload


def _fmt(value: object) -> str:
    x = _finite_float(value)
    if x is None:
        return ""
    if abs(x) >= 1:
        return f"{x:.4f}"
    return f"{x:.5f}"


def write_markdown(payload: Mapping[str, object], out_md: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
    summary = payload.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    baseline_labels = [
        str(row.get("baseline"))
        for row in payload.get("compare_files", [])
        if isinstance(row, Mapping) and row.get("exists")
    ]
    if not baseline_labels:
        baseline_labels = [str(row.get("baseline")) for row in payload.get("compare_files", []) if isinstance(row, Mapping)]
    compact_baselines = baseline_labels[:5]
    comparison_headers = [f"vs {label} ΔTE/p" for label in compact_baselines]
    lines = [
        "# Region Adapter Decision Report",
        "",
        f"- Slice: {payload.get('slice')}",
        f"- Top-k: {payload.get('top_k')}",
        f"- Strict alpha: {payload.get('strict_alpha')}",
        f"- Borderline alpha: {payload.get('borderline_alpha')}",
        f"- Constraint exact tolerance: {payload.get('constraint_exact_tolerance')}",
        f"- Compare files: {summary.get('n_compare_files_found')}/{summary.get('n_compare_files_expected')}",
        f"- Missing compare files: {len(summary.get('missing_compare_files', []))}",
        f"- All constraints OK: {summary.get('all_constraints_ok')}",
        f"- All constraints exactly 1: {summary.get('all_constraints_exact_1')}",
        f"- Best run vs hardneg: {summary.get('best_run_vs_hardneg')}",
        "",
        "| run | constraints ok | constraints exact 1 | legal | protein | budget | frame | "
        + " | ".join(comparison_headers)
        + " |",
        "|---|---:|---:|---:|---:|---:|---:|" + "---:|" * len(comparison_headers),
    ]
    for row in payload.get("runs", []):
        if not isinstance(row, Mapping):
            continue
        constraints = row.get("constraints", {})
        if not isinstance(constraints, Mapping):
            constraints = {}
        primary_by_baseline = row.get("primary_by_baseline", {})
        if not isinstance(primary_by_baseline, Mapping):
            primary_by_baseline = {}
        comparison_cells = []
        for baseline_label in compact_baselines:
            primary = primary_by_baseline.get(baseline_label, {})
            if isinstance(primary, Mapping):
                comparison_cells.append(
                    f"{_fmt(primary.get('delta'))}/{_fmt(primary.get('paired_p'))}/{primary.get('signal', '')}"
                )
            else:
                comparison_cells.append("")
        lines.append(
            "| {run} | {ok} | {exact} | {legal} | {protein} | {budget} | {frame} | {comparisons} |".format(
                run=row.get("run"),
                ok=row.get("constraints_ok"),
                exact=row.get("constraints_exact_1"),
                legal=_fmt(constraints.get("legal_fraction")),
                protein=_fmt(constraints.get("mean_protein_identity")),
                budget=_fmt(constraints.get("within_budget_fraction")),
                frame=_fmt(constraints.get("reading_frame_intact_fraction")),
                comparisons=" | ".join(comparison_cells),
            )
        )
    lines.extend(["", "## Compare File Status", "", "| baseline | exists | path |", "|---|---:|---|"])
    for row in payload.get("compare_files", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(f"| {row.get('baseline')} | {row.get('exists')} | `{row.get('path')}` |")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--slice", dest="slice_name", default="head256")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--baseline", action="append", default=None)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    payload = summarize_region_adapter_comparisons(
        project_root=args.project_root,
        slice_name=args.slice_name,
        top_k=args.top_k,
        baselines=args.baseline,
        out_json=args.out_json,
        out_md=args.out_md,
    )
    print(json.dumps({"summary": payload["summary"], "out_json": args.out_json, "out_md": args.out_md}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
