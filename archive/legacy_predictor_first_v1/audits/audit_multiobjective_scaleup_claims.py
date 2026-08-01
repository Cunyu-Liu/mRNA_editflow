"""Audit multi-objective fusion scale-up claim language.

This module is read-only. It turns the completed head256/head1024 comparison
artifacts into an explicit claim ledger so borderline scale-up evidence cannot
be accidentally described as strictly significant.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Mapping, Optional, Sequence


PRIMARY_METRIC = "delta_oracle_te_vs_source"
STRICT_ALPHA = 0.05
BORDERLINE_ALPHA = 0.10
IDENTITY_TOL = 1e-12
SUMMARY_CONSTRAINT_METRICS: tuple[str, ...] = (
    "legal_fraction",
    "mean_protein_identity",
    "within_budget_fraction",
    "reading_frame_intact_fraction",
)
COMPARE_CONSTRAINT_METRICS: tuple[str, ...] = (
    "mean_protein_identity",
    "within_budget_fraction",
    "reading_frame_intact_fraction",
)
MO_MODES: tuple[str, ...] = ("te_only", "scalar", "pareto", "grpo")
SLICES: tuple[str, ...] = ("head256", "head1024")


def _rel(path: str, root: str) -> str:
    return os.path.relpath(path, root) if os.path.isabs(path) else path


def _load_json_if_exists(path: str) -> Optional[Mapping[str, object]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _finite_float(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def classify_signal(delta: object, paired_p: object) -> str:
    """Classify a paired comparison without overstating borderline evidence."""
    delta_x = _finite_float(delta)
    p_x = _finite_float(paired_p)
    if delta_x is None or p_x is None:
        return "missing"
    if delta_x > 0 and p_x < STRICT_ALPHA:
        return "strict_positive"
    if delta_x > 0 and p_x < BORDERLINE_ALPHA:
        return "borderline_positive"
    if delta_x > 0:
        return "positive_not_significant"
    if delta_x == 0:
        return "no_effect"
    return "negative"


def _claim_language(signal: str) -> str:
    if signal == "strict_positive":
        return "may_claim_strict_significant_positive"
    if signal == "borderline_positive":
        return "trend_or_borderline_only_no_strict_significance"
    if signal == "positive_not_significant":
        return "numeric_positive_only_no_significance_claim"
    if signal == "no_effect":
        return "no_effect_claim_only"
    if signal == "negative":
        return "negative_or_regression_claim"
    return "insufficient_evidence"


def _comparison_row(
    payload: Mapping[str, object],
    *,
    run_label: str,
    metric: str,
) -> Optional[Mapping[str, object]]:
    rows = payload.get("rows", [])
    if not isinstance(rows, Sequence):
        return None
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if row.get("run") == run_label and row.get("metric") == metric:
            return row
    return None


def _n_records(payload: Optional[Mapping[str, object]]) -> Optional[int]:
    if payload is None:
        return None
    baseline = payload.get("baseline", {})
    if not isinstance(baseline, Mapping):
        return None
    config = baseline.get("config", {})
    if not isinstance(config, Mapping):
        return None
    value = config.get("n_records")
    return int(value) if isinstance(value, int) else None


def _config_checks_ok(payload: Optional[Mapping[str, object]]) -> Optional[bool]:
    if payload is None:
        return None
    checks = payload.get("config_checks", [])
    if not isinstance(checks, Sequence):
        return None
    if not checks:
        return None
    return all(isinstance(row, Mapping) and bool(row.get("matches")) for row in checks)


def _comparison_constraint_status(
    payload: Optional[Mapping[str, object]],
    *,
    run_label: str,
) -> dict[str, object]:
    metrics: dict[str, Optional[float]] = {}
    missing: list[str] = []
    for metric in COMPARE_CONSTRAINT_METRICS:
        row = _comparison_row(payload, run_label=run_label, metric=metric) if payload else None
        value = _finite_float(row.get("run_mean")) if isinstance(row, Mapping) else None
        metrics[metric] = value
        if value is None:
            missing.append(metric)
    exact = bool(metrics) and not missing and all(
        value is not None and abs(value - 1.0) <= IDENTITY_TOL
        for value in metrics.values()
    )
    return {
        "metrics": metrics,
        "missing_metrics": missing,
        "available_constraints_exact_1": exact,
    }


def _audit_comparison(
    project_root: str,
    *,
    comparison_id: str,
    relative_path: str,
    run_label: str,
    expected_signal: Optional[str],
) -> dict[str, object]:
    path = os.path.join(project_root, relative_path)
    payload = _load_json_if_exists(path)
    primary = (
        _comparison_row(payload, run_label=run_label, metric=PRIMARY_METRIC)
        if payload is not None
        else None
    )
    delta = _finite_float(primary.get("delta")) if isinstance(primary, Mapping) else None
    paired_p = _finite_float(primary.get("paired_p")) if isinstance(primary, Mapping) else None
    run_mean = _finite_float(primary.get("run_mean")) if isinstance(primary, Mapping) else None
    baseline_mean = (
        _finite_float(primary.get("baseline_mean")) if isinstance(primary, Mapping) else None
    )
    signal = classify_signal(delta, paired_p)
    constraints = _comparison_constraint_status(payload, run_label=run_label)
    return {
        "comparison_id": comparison_id,
        "source": relative_path,
        "exists": payload is not None,
        "run": run_label,
        "metric": PRIMARY_METRIC,
        "n_records": _n_records(payload),
        "config_checks_ok": _config_checks_ok(payload),
        "primary_row_exists": primary is not None,
        "baseline_mean": baseline_mean,
        "run_mean": run_mean,
        "delta": delta,
        "paired_p": paired_p,
        "n_paired_seeds": (
            int(primary["n_paired_seeds"])
            if isinstance(primary, Mapping) and isinstance(primary.get("n_paired_seeds"), int)
            else None
        ),
        "signal": signal,
        "expected_signal": expected_signal,
        "expected_signal_met": expected_signal is None or signal == expected_signal,
        "claim_language": _claim_language(signal),
        "comparison_constraints": constraints,
    }


def _expected_comparisons() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run, expected in (
        ("mo_scalar_top64", "strict_positive"),
        ("mo_pareto_top64", "strict_positive"),
        ("mo_grpo_top64", "strict_positive"),
    ):
        rows.append(
            {
                "comparison_id": f"head256_{run}_vs_te_only",
                "relative_path": "benchmark/compare_mo_fusion_vs_te_only_head256.json",
                "run_label": run,
                "expected_signal": expected,
            }
        )
    for run, path in (
        ("mo_scalar_top64", "benchmark/compare_scalar_vs_hardneg_v2_head256.json"),
        ("mo_pareto", "benchmark/compare_pareto_vs_hardneg_v2_head256.json"),
        ("mo_grpo", "benchmark/compare_grpo_vs_hardneg_v2_head256.json"),
    ):
        rows.append(
            {
                "comparison_id": f"head256_{run}_vs_hardneg_v2",
                "relative_path": path,
                "run_label": run,
                "expected_signal": "strict_positive",
            }
        )
    rows.extend(
        [
            {
                "comparison_id": "head256_grpo_vs_scalar",
                "relative_path": "benchmark/compare_grpo_vs_scalar_head256.json",
                "run_label": "mo_grpo",
                "expected_signal": "positive_not_significant",
            },
            {
                "comparison_id": "head256_scalar_vs_pareto",
                "relative_path": "benchmark/compare_scalar_vs_pareto_head256.json",
                "run_label": "mo_scalar",
                "expected_signal": "positive_not_significant",
            },
        ]
    )
    for run, expected in (
        ("mo_scalar_top64", "positive_not_significant"),
        ("mo_pareto_top64", "borderline_positive"),
        ("mo_grpo_top64", "positive_not_significant"),
    ):
        rows.append(
            {
                "comparison_id": f"head1024_{run}_vs_te_only",
                "relative_path": "benchmark/compare_mo_fusion_vs_te_only_head1024.json",
                "run_label": run,
                "expected_signal": expected,
            }
        )
    for run in ("mo_scalar_top64", "mo_pareto_top64", "mo_grpo_top64"):
        rows.append(
            {
                "comparison_id": f"head1024_{run}_vs_hardneg_v2",
                "relative_path": "benchmark/compare_mo_fusion_vs_hardneg_v2_head1024.json",
                "run_label": run,
                "expected_signal": "strict_positive",
            }
        )
    return rows


def _summary_path(project_root: str, slice_name: str, mode: str) -> str:
    if mode == "hardneg_v2":
        name = f"multiseed_t5_public_{slice_name}_hardneg_v2_top64"
    else:
        name = f"multiseed_t5_public_{slice_name}_mo_{mode}_top64"
    return os.path.join(project_root, "benchmark", name, "multiseed_summary.json")


def _aggregate_metric(summary: Mapping[str, object], metric: str) -> Optional[float]:
    aggregate = summary.get("aggregate", {})
    if not isinstance(aggregate, Mapping):
        return None
    value = aggregate.get(metric)
    if isinstance(value, Mapping):
        return _finite_float(value.get("mean"))
    return _finite_float(value)


def _audit_summary_constraints(
    project_root: str,
    *,
    slice_name: str,
    mode: str,
) -> dict[str, object]:
    path = _summary_path(project_root, slice_name, mode)
    payload = _load_json_if_exists(path)
    metrics: dict[str, Optional[float]] = {}
    missing: list[str] = []
    if payload is not None:
        for metric in SUMMARY_CONSTRAINT_METRICS:
            value = _aggregate_metric(payload, metric)
            metrics[metric] = value
            if value is None:
                missing.append(metric)
    else:
        missing = list(SUMMARY_CONSTRAINT_METRICS)
        metrics = {metric: None for metric in SUMMARY_CONSTRAINT_METRICS}
    exact = payload is not None and not missing and all(
        value is not None and abs(value - 1.0) <= IDENTITY_TOL
        for value in metrics.values()
    )
    return {
        "slice": slice_name,
        "mode": mode,
        "path": _rel(path, project_root),
        "exists": payload is not None,
        "metrics": metrics,
        "missing_metrics": missing,
        "all_constraints_exact_1": exact,
    }


def audit_multiobjective_scaleup_claims(
    *,
    project_root: str,
    out_json: Optional[str] = None,
    out_md: Optional[str] = None,
) -> dict[str, object]:
    comparisons = [
        _audit_comparison(project_root, **row) for row in _expected_comparisons()
    ]
    summaries = [
        _audit_summary_constraints(project_root, slice_name=slice_name, mode=mode)
        for slice_name in SLICES
        for mode in (*MO_MODES, "hardneg_v2")
    ]
    comparison_rows_ready = all(
        row["exists"]
        and row["primary_row_exists"]
        and row["signal"] != "missing"
        and row["expected_signal_met"]
        for row in comparisons
    )
    available_compare_constraints_exact = all(
        bool(row["comparison_constraints"]["available_constraints_exact_1"])
        for row in comparisons
    )
    summary_constraints_complete = all(
        bool(row["all_constraints_exact_1"]) for row in summaries
    )
    head1024_vs_te = [
        row for row in comparisons if str(row["comparison_id"]).startswith("head1024_")
        and str(row["comparison_id"]).endswith("_vs_te_only")
    ]
    head1024_vs_hardneg = [
        row for row in comparisons if str(row["comparison_id"]).startswith("head1024_")
        and str(row["comparison_id"]).endswith("_vs_hardneg_v2")
    ]
    head256_vs_te = [
        row for row in comparisons if str(row["comparison_id"]).startswith("head256_")
        and str(row["comparison_id"]).endswith("_vs_te_only")
    ]
    head256_vs_hardneg = [
        row for row in comparisons if str(row["comparison_id"]).startswith("head256_")
        and str(row["comparison_id"]).endswith("_vs_hardneg_v2")
    ]
    payload = {
        "artifact_kind": "multiobjective_scaleup_claim_audit",
        "primary_metric": PRIMARY_METRIC,
        "summary": {
            "comparison_rows_ready": comparison_rows_ready,
            "expected_signal_contract_met": all(
                bool(row["expected_signal_met"]) for row in comparisons
            ),
            "available_compare_constraints_exact_1": available_compare_constraints_exact,
            "summary_constraints_complete": summary_constraints_complete,
            "ready_for_full_hard_constraint_claim_audit": (
                comparison_rows_ready
                and available_compare_constraints_exact
                and summary_constraints_complete
            ),
            "head256_fusion_vs_te_only_all_strict": all(
                row["signal"] == "strict_positive" for row in head256_vs_te
            ),
            "head256_fusion_vs_hardneg_v2_all_strict": all(
                row["signal"] == "strict_positive" for row in head256_vs_hardneg
            ),
            "head1024_vs_te_only_strict_claim_allowed": any(
                row["signal"] == "strict_positive" for row in head1024_vs_te
            ),
            "head1024_vs_te_only_best_signal": (
                "borderline_positive"
                if any(row["signal"] == "borderline_positive" for row in head1024_vs_te)
                else max((str(row["signal"]) for row in head1024_vs_te), default="missing")
            ),
            "head1024_vs_hardneg_v2_all_strict": all(
                row["signal"] == "strict_positive" for row in head1024_vs_hardneg
            ),
            "claim_policy": (
                "head256 fusion-vs-control claims may be strict only where "
                "paired p < 0.05. The completed head1024 scale-up versus the "
                "stronger te_only control is not strictly significant; pareto is "
                "borderline/trend only, while all head1024 fusion modes remain "
                "strictly positive versus the older hardneg_v2 baseline."
            ),
        },
        "comparisons": comparisons,
        "summary_constraints": summaries,
    }
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
    if out_md:
        write_markdown(payload, out_md)
    return payload


def write_markdown(payload: Mapping[str, object], out_md: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
    summary = payload.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    lines = [
        "# Multi-Objective Scale-Up Claim Audit",
        "",
        f"- Primary metric: `{payload.get('primary_metric')}`",
        f"- Comparison rows ready: {summary.get('comparison_rows_ready')}",
        f"- Full hard-constraint claim audit ready: {summary.get('ready_for_full_hard_constraint_claim_audit')}",
        f"- head256 fusion vs te_only all strict: {summary.get('head256_fusion_vs_te_only_all_strict')}",
        f"- head1024 vs te_only strict claim allowed: {summary.get('head1024_vs_te_only_strict_claim_allowed')}",
        f"- head1024 vs te_only best signal: {summary.get('head1024_vs_te_only_best_signal')}",
        f"- head1024 vs hardneg_v2 all strict: {summary.get('head1024_vs_hardneg_v2_all_strict')}",
        f"- Claim policy: {summary.get('claim_policy')}",
        "",
        "## Comparison Ledger",
        "",
        "| comparison | run | delta | paired p | signal | allowed language | source |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for row in payload.get("comparisons", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            f"| {row.get('comparison_id')} | {row.get('run')} | "
            f"{_fmt(row.get('delta'))} | {_fmt(row.get('paired_p'))} | "
            f"{row.get('signal')} | {row.get('claim_language')} | "
            f"`{row.get('source')}` |"
        )
    lines.extend(
        [
            "",
            "## Summary Constraint Evidence",
            "",
            "| slice | mode | exists | constraints exact 1 | missing metrics | path |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for row in payload.get("summary_constraints", []):
        if not isinstance(row, Mapping):
            continue
        missing = row.get("missing_metrics", [])
        lines.append(
            f"| {row.get('slice')} | {row.get('mode')} | {row.get('exists')} | "
            f"{row.get('all_constraints_exact_1')} | `{missing}` | "
            f"`{row.get('path')}` |"
        )
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _fmt(value: object) -> str:
    x = _finite_float(value)
    if x is None:
        return "NA"
    return f"{x:.6g}"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    payload = audit_multiobjective_scaleup_claims(
        project_root=args.project_root,
        out_json=args.out_json,
        out_md=args.out_md,
    )
    print(
        json.dumps(
            {"summary": payload["summary"], "out_json": args.out_json, "out_md": args.out_md},
            sort_keys=True,
        )
    )
    if args.strict and not payload["summary"]["ready_for_full_hard_constraint_claim_audit"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
