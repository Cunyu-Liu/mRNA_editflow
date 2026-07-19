"""Audit completed region-adapter comparison artifacts.

This read-only audit is intentionally stricter than the decision report:
it checks that all expected compare files exist, all expected adapter runs are
present, hard constraints are exactly 1.0, and primary TE statistics are finite.
It does not require positive results; negative or non-significant deltas are
valid evidence, but they must be reported honestly through the signal labels.
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
EXPECTED_BASELINE_STEMS: tuple[str, ...] = (
    "hardneg_v2",
    "mo_grpo",
    "mo_scalar",
    "mo_pareto",
    "mo_te_only",
)
DEFAULT_MODES: tuple[str, ...] = ("utr5", "cds", "utr3", "all")


def _load_json(path: str) -> Mapping[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _finite_float(value: object) -> Optional[float]:
    if not isinstance(value, (int, float)):
        return None
    x = float(value)
    return x if math.isfinite(x) else None


def _baseline_labels(top_k: int) -> tuple[str, ...]:
    return tuple(f"{stem}_top{top_k}" for stem in EXPECTED_BASELINE_STEMS)


def _expected_runs(modes: Sequence[str], top_k: int) -> tuple[str, ...]:
    return tuple(f"region_adapter_{mode}_top{top_k}" for mode in modes)


def _compare_path(project_root: str, baseline_label: str, slice_name: str) -> str:
    return os.path.join(
        project_root,
        "benchmark",
        f"region_adapter_vs_{baseline_label}_{slice_name}.json",
    )


def _decision_path(project_root: str, slice_name: str) -> str:
    return os.path.join(
        project_root,
        "benchmark",
        f"region_adapter_decision_report_{slice_name}.json",
    )


def _is_exact_1(value: object, tolerance: float) -> bool:
    x = _finite_float(value)
    return x is not None and abs(x - 1.0) <= tolerance


def _index_decision_runs(decision: Optional[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    if not decision:
        return {}
    indexed: dict[str, Mapping[str, object]] = {}
    for row in decision.get("runs", []):
        if isinstance(row, Mapping) and row.get("run"):
            indexed[str(row["run"])] = row
    return indexed


def _primary_audit(row: Mapping[str, object], baseline_label: str) -> dict[str, object]:
    primary_by_baseline = row.get("primary_by_baseline", {})
    if not isinstance(primary_by_baseline, Mapping):
        primary_by_baseline = {}
    primary = primary_by_baseline.get(baseline_label, {})
    if not isinstance(primary, Mapping):
        primary = {}
    delta = _finite_float(primary.get("delta"))
    paired_p = _finite_float(primary.get("paired_p"))
    run_mean = _finite_float(primary.get("run_mean"))
    baseline_mean = _finite_float(primary.get("baseline_mean"))
    return {
        "baseline": baseline_label,
        "present": bool(primary),
        "delta": delta,
        "paired_p": paired_p,
        "run_mean": run_mean,
        "baseline_mean": baseline_mean,
        "signal": primary.get("signal") if primary else None,
        "primary_stats_finite": all(
            value is not None for value in (delta, paired_p, run_mean, baseline_mean)
        ),
    }


def audit_region_adapter_results(
    *,
    project_root: str,
    slice_name: str = "head256",
    top_k: int = 64,
    modes: Sequence[str] = DEFAULT_MODES,
    constraint_tolerance: float = 1e-12,
    out_json: Optional[str] = None,
    out_md: Optional[str] = None,
) -> dict[str, object]:
    """Audit region-adapter result artifacts for completeness and claim safety."""
    baseline_labels = _baseline_labels(top_k)
    expected_runs = _expected_runs(modes, top_k)
    compare_files = []
    for label in baseline_labels:
        path = _compare_path(project_root, label, slice_name)
        compare_files.append(
            {
                "baseline": label,
                "path": os.path.relpath(path, project_root),
                "exists": os.path.exists(path),
            }
        )

    decision_report_path = _decision_path(project_root, slice_name)
    decision_exists = os.path.exists(decision_report_path)
    decision = _load_json(decision_report_path) if decision_exists else None
    indexed_runs = _index_decision_runs(decision)

    run_audits = []
    for run in expected_runs:
        row = indexed_runs.get(run)
        constraints = row.get("constraints", {}) if row else {}
        if not isinstance(constraints, Mapping):
            constraints = {}
        constraint_exact = {
            metric: _is_exact_1(constraints.get(metric), constraint_tolerance)
            for metric in CONSTRAINT_METRICS
        }
        primary_by_baseline = (
            [_primary_audit(row, label) for label in baseline_labels] if row else []
        )
        run_audits.append(
            {
                "run": run,
                "present": row is not None,
                "constraints": dict(constraints),
                "constraint_exact": constraint_exact,
                "constraints_exact_1": bool(row) and all(constraint_exact.values()),
                "primary_by_baseline": primary_by_baseline,
                "all_primary_stats_finite": bool(row)
                and len(primary_by_baseline) == len(baseline_labels)
                and all(bool(item["primary_stats_finite"]) for item in primary_by_baseline),
            }
        )

    missing_artifacts = [
        item for item in compare_files if not bool(item["exists"])
    ]
    if not decision_exists:
        missing_artifacts.append(
            {
                "baseline": None,
                "path": os.path.relpath(decision_report_path, project_root),
                "exists": False,
            }
        )

    all_expected_compare_files_exist = all(bool(item["exists"]) for item in compare_files)
    all_expected_runs_present = all(bool(item["present"]) for item in run_audits)
    all_constraints_exact_1 = bool(run_audits) and all(
        bool(item["constraints_exact_1"]) for item in run_audits
    )
    all_primary_stats_finite = bool(run_audits) and all(
        bool(item["all_primary_stats_finite"]) for item in run_audits
    )
    ready = (
        decision_exists
        and all_expected_compare_files_exist
        and all_expected_runs_present
        and all_constraints_exact_1
        and all_primary_stats_finite
    )

    payload = {
        "artifact_kind": "region_adapter_result_audit",
        "slice": slice_name,
        "top_k": int(top_k),
        "expected_runs": list(expected_runs),
        "expected_baselines": list(baseline_labels),
        "constraint_metrics": list(CONSTRAINT_METRICS),
        "constraint_tolerance": constraint_tolerance,
        "decision_report": {
            "path": os.path.relpath(decision_report_path, project_root),
            "exists": decision_exists,
        },
        "compare_files": compare_files,
        "summary": {
            "ready_for_sota_claim_audit": ready,
            "n_compare_files_found": sum(1 for item in compare_files if item["exists"]),
            "n_compare_files_expected": len(compare_files),
            "decision_report_exists": decision_exists,
            "all_expected_compare_files_exist": all_expected_compare_files_exist,
            "all_expected_runs_present": all_expected_runs_present,
            "all_constraints_exact_1": all_constraints_exact_1,
            "all_primary_stats_finite": all_primary_stats_finite,
            "missing_artifacts": missing_artifacts,
        },
        "run_audits": run_audits,
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
    return "" if x is None else f"{x:.5f}"


def write_markdown(payload: Mapping[str, object], out_md: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
    summary = payload.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    baseline_labels = [str(label) for label in payload.get("expected_baselines", [])]
    lines = [
        "# Region Adapter Result Audit",
        "",
        f"- Slice: {payload.get('slice')}",
        f"- Top-k: {payload.get('top_k')}",
        f"- Ready for SOTA claim audit: {summary.get('ready_for_sota_claim_audit')}",
        f"- Compare files: {summary.get('n_compare_files_found')}/{summary.get('n_compare_files_expected')}",
        f"- Decision report exists: {summary.get('decision_report_exists')}",
        f"- All constraints exactly 1: {summary.get('all_constraints_exact_1')}",
        f"- All primary stats finite: {summary.get('all_primary_stats_finite')}",
        "",
        "| run | present | constraints exact 1 | primary stats finite | "
        + " | ".join(f"{label} signal" for label in baseline_labels)
        + " |",
        "|---|---:|---:|---:|" + "---:|" * len(baseline_labels),
    ]
    for row in payload.get("run_audits", []):
        if not isinstance(row, Mapping):
            continue
        primary_by_baseline = row.get("primary_by_baseline", [])
        signal_by_baseline = {}
        for primary in primary_by_baseline if isinstance(primary_by_baseline, list) else []:
            if isinstance(primary, Mapping):
                signal_by_baseline[str(primary.get("baseline"))] = str(primary.get("signal"))
        signals = [signal_by_baseline.get(label, "") for label in baseline_labels]
        lines.append(
            "| {run} | {present} | {constraints} | {primary} | {signals} |".format(
                run=row.get("run"),
                present=row.get("present"),
                constraints=row.get("constraints_exact_1"),
                primary=row.get("all_primary_stats_finite"),
                signals=" | ".join(signals),
            )
        )
    lines.extend(["", "## Missing Artifacts", "", "| path |", "|---|"])
    for item in summary.get("missing_artifacts", []):
        if isinstance(item, Mapping):
            lines.append(f"| `{item.get('path')}` |")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--slice", dest="slice_name", default="head256")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--mode", action="append", default=None)
    parser.add_argument("--constraint-tolerance", type=float, default=1e-12)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    payload = audit_region_adapter_results(
        project_root=args.project_root,
        slice_name=args.slice_name,
        top_k=args.top_k,
        modes=tuple(args.mode) if args.mode else DEFAULT_MODES,
        constraint_tolerance=args.constraint_tolerance,
        out_json=args.out_json,
        out_md=args.out_md,
    )
    print(json.dumps({"summary": payload["summary"], "out_json": args.out_json, "out_md": args.out_md}, sort_keys=True))
    if args.strict and not payload["summary"]["ready_for_sota_claim_audit"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
