"""Compare multi-seed mRNA-EditFlow benchmark summaries.

The project now produces one ``multiseed_summary.json`` per decoding/training
setting. This module turns several such summaries into a paired comparison table
against a baseline, using seed-aligned metric differences whenever possible.

Statistical protocol
--------------------
For a metric ``m`` and a run ``R`` compared with baseline ``B``, we align common
seeds and compute paired differences

``d_s = m_R(s) - m_B(s)``.

The table reports ``mean(d_s)``, a non-parametric bootstrap CI over seeds, and a
two-sided paired sign-flip permutation p-value. For lower-is-better metrics the
``improvement`` column is ``-mean(d_s)`` so positive always means better than
baseline.

Complexity: ``O(R * M * S * P)`` where ``R`` is number of compared runs, ``M``
number of metrics, ``S`` number of common seeds and ``P`` permutations.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from mrna_editflow.eval.run_eval import bootstrap_ci, paired_permutation_pvalue
from mrna_editflow.eval.run_multiseed_benchmark import METRIC_SPECS


DEFAULT_REQUIRED_MATCHING_CONFIG: tuple[str, ...] = (
    "task_id",
    "edit_budget",
    "effective_proposal_top_k",
    "proposal_temperature",
    "n_records",
    "decoder_seeds",
    "target_length_delta",
    "guidance_scale",
    "target_te",
    "target_start_accessibility",
    "editable_regions",
    "max_pairwise_pairs",
    "max_novelty_sources",
)


@dataclass(frozen=True)
class BenchmarkRun:
    """Parsed multi-seed summary.

    ``seed_metrics`` maps ``seed -> {metric_name: value}``. Aggregate metrics are
    preserved for display when seed-level data are unavailable.
    """

    label: str
    path: str
    config: Mapping[str, object]
    aggregate: Mapping[str, Mapping[str, float]]
    seed_metrics: Mapping[int, Mapping[str, float]]
    training_seeds: frozenset[int]
    nested_seed_metrics: Mapping[int, Mapping[int, Mapping[str, float]]]


def _load_json(path: str) -> Mapping[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _normalise_config(config: Mapping[str, object]) -> dict[str, object]:
    """Return config with derived fields used for fair paired comparisons.

    Cascade decoding and single-ranker decoding expose the same scientific
    nuisance variable through different raw fields. For a single ranker the
    retained candidate width is ``proposal_top_k``; for the two-stage cascade it
    is ``cascade_recall_top_k`` because precision reranking only sees that
    retained set. We therefore add

    ``effective_proposal_top_k = cascade_recall_top_k`` for cascade runs,
    otherwise ``proposal_top_k``.

    The comparison itself is ``O(|config|)``. It does not mutate the loaded
    payload, and it fills legacy defaults so older benchmark summaries can be
    checked against newer cascade summaries without spurious mismatches.
    """
    out = dict(config)
    if "decoder_seeds" not in out and "seeds" in out:
        out["decoder_seeds"] = out["seeds"]
    uses_cascade = bool(out.get("cascade_recall_checkpoint_path"))
    if "decoder_family" not in out:
        if uses_cascade:
            out["decoder_family"] = "cascade"
        elif out.get("checkpoint_path"):
            out["decoder_family"] = "checkpoint_guided"
        else:
            out["decoder_family"] = "deterministic"
    if "effective_proposal_top_k" not in out:
        out["effective_proposal_top_k"] = (
            out.get("cascade_recall_top_k") if uses_cascade else out.get("proposal_top_k")
        )
    out.setdefault("cascade_recall_checkpoint_path", None)
    out.setdefault("cascade_recall_top_k", None)
    out.setdefault("target_length_delta", 0)
    out.setdefault("proposal_temperature", 1.0)
    out.setdefault("guidance_scale", 0.0)
    out.setdefault("target_te", None)
    out.setdefault("target_start_accessibility", None)
    out.setdefault("editable_regions", ["utr5", "utr3"])
    out.setdefault("max_novelty_sources", 0)
    return out


def load_benchmark_run(label: str, path: str) -> BenchmarkRun:
    """Load a ``multiseed_summary.json`` as :class:`BenchmarkRun`."""
    payload = _load_json(path)
    per_seed = payload.get("per_seed", [])
    seed_metrics: dict[int, Mapping[str, float]] = {}
    training_seeds: set[int] = set()
    nested_seed_metrics: dict[int, dict[int, Mapping[str, float]]] = {}
    if isinstance(per_seed, Sequence):
        for row in per_seed:
            if not isinstance(row, Mapping):
                continue
            try:
                seed = int(row.get("decoder_seed", row.get("seed", len(seed_metrics))))
            except (TypeError, ValueError):
                seed = len(seed_metrics)
            metrics = row.get("metrics", {})
            if isinstance(metrics, Mapping):
                numeric_metrics = {
                    str(k): float(v)
                    for k, v in metrics.items()
                    if isinstance(v, (int, float)) and math.isfinite(float(v))
                }
                seed_metrics.setdefault(seed, numeric_metrics)
            raw_training_seed = row.get("training_seed")
            if raw_training_seed is not None:
                training_seed = int(raw_training_seed)
                training_seeds.add(training_seed)
                if isinstance(metrics, Mapping):
                    nested_seed_metrics.setdefault(training_seed, {})[seed] = numeric_metrics
    aggregate = payload.get("aggregate", {})
    config = payload.get("config", {})
    if not isinstance(aggregate, Mapping):
        aggregate = {}
    if not isinstance(config, Mapping):
        config = {}
    return BenchmarkRun(
        label=label,
        path=path,
        config=_normalise_config(config),
        aggregate=aggregate,  # type: ignore[arg-type]
        seed_metrics=seed_metrics,
        training_seeds=frozenset(training_seeds),
        nested_seed_metrics=nested_seed_metrics,
    )


def _training_seed_means(run: BenchmarkRun, metric: str) -> dict[int, float]:
    """Average nested decoder observations within each independent training seed."""
    means: dict[int, float] = {}
    for training_seed, decoder_rows in run.nested_seed_metrics.items():
        values = [
            float(row[metric])
            for row in decoder_rows.values()
            if metric in row and math.isfinite(float(row[metric]))
        ]
        if values:
            means[int(training_seed)] = float(np.mean(values))
    return means


def _aggregate_mean(run: BenchmarkRun, metric: str) -> float:
    row = run.aggregate.get(metric, {})
    if isinstance(row, Mapping) and "mean" in row:
        try:
            return float(row["mean"])
        except (TypeError, ValueError):
            return 0.0
    values = [float(m[metric]) for m in run.seed_metrics.values() if metric in m]
    return float(np.mean(values)) if values else 0.0


def _direction(metric: str) -> bool:
    for spec in METRIC_SPECS:
        if spec.name == metric:
            return bool(spec.higher_is_better)
    return True


def _metric_description(metric: str) -> str:
    for spec in METRIC_SPECS:
        if spec.name == metric:
            return spec.description
    return ""


def validate_matching_config(
    baseline: BenchmarkRun,
    runs: Sequence[BenchmarkRun],
    fields: Sequence[str],
) -> list[dict[str, object]]:
    """Require selected config fields to match before paired comparison.

    Paired seed tests only justify a method claim when nuisance variables are
    controlled. For a field set ``F``, this check enforces

    ``config_B[f] = config_R[f]`` for every compared run ``R`` and field
    ``f in F``.

    The comparison cost is ``O(R * |F|)`` and is independent of the benchmark
    sample count. A mismatch raises ``ValueError`` so automated paper-table
    jobs fail closed instead of silently mixing checkpoints, search widths or
    record subsets.
    """
    checked: list[dict[str, object]] = []
    mismatches: list[str] = []
    for run in runs:
        for field in fields:
            baseline_value = baseline.config.get(field)
            run_value = run.config.get(field)
            checked.append(
                {
                    "run": run.label,
                    "field": field,
                    "baseline": baseline_value,
                    "run_value": run_value,
                    "matches": baseline_value == run_value,
                }
            )
            if baseline_value != run_value:
                mismatches.append(
                    f"{run.label}.{field}: baseline={baseline_value!r}, run={run_value!r}"
                )
    if mismatches:
        joined = "; ".join(mismatches)
        raise ValueError(f"benchmark config mismatch for required fields: {joined}")
    return checked


def compare_run_to_baseline(
    baseline: BenchmarkRun,
    run: BenchmarkRun,
    metrics: Sequence[str],
    *,
    n_bootstrap: int = 1000,
    n_permutations: int = 2000,
) -> list[dict[str, object]]:
    """Return paired comparison rows for ``run`` vs ``baseline``."""
    rows: list[dict[str, object]] = []
    paper_mode = (
        baseline.config.get("run_mode") == "paper"
        or run.config.get("run_mode") == "paper"
    )
    common_training_seeds = sorted(baseline.training_seeds & run.training_seeds)
    if paper_mode and (
        len(baseline.training_seeds) < 3
        or len(run.training_seeds) < 3
        or len(common_training_seeds) < 3
    ):
        raise ValueError(
            "paper headline significance requires at least three matched independent "
            "training seeds in both runs; "
            f"baseline={len(baseline.training_seeds)}, run={len(run.training_seeds)}, "
            f"matched={len(common_training_seeds)}"
        )
    common_seeds = sorted(set(baseline.seed_metrics) & set(run.seed_metrics))
    for metric in metrics:
        higher = _direction(metric)
        baseline_training_means = _training_seed_means(baseline, metric)
        run_training_means = _training_seed_means(run, metric)
        if paper_mode:
            paired_units = [
                seed for seed in common_training_seeds
                if seed in baseline_training_means and seed in run_training_means
            ]
            if len(paired_units) < 3:
                raise ValueError(
                    f"paper metric {metric!r} lacks three matched training-seed observations"
                )
            b_mean = float(np.mean([baseline_training_means[seed] for seed in paired_units]))
            r_mean = float(np.mean([run_training_means[seed] for seed in paired_units]))
        else:
            paired_units = common_seeds
            b_mean = _aggregate_mean(baseline, metric)
            r_mean = _aggregate_mean(run, metric)
        diffs = []
        b_values = []
        r_values = []
        for seed in paired_units:
            if paper_mode:
                b_val = baseline_training_means[seed]
                r_val = run_training_means[seed]
            else:
                b = baseline.seed_metrics[seed]
                r = run.seed_metrics[seed]
                if metric not in b or metric not in r:
                    continue
                b_val = float(b[metric])
                r_val = float(r[metric])
            if math.isfinite(b_val) and math.isfinite(r_val):
                b_values.append(b_val)
                r_values.append(r_val)
                diffs.append(r_val - b_val)
        if diffs:
            seed_for_ci = paired_units[: max(5, min(len(paired_units), len(diffs)))]
            if len(seed_for_ci) < 5:
                seed_for_ci = list(range(5))
            ci = bootstrap_ci(diffs, seeds=seed_for_ci, n_bootstrap=n_bootstrap)
            p = paired_permutation_pvalue(
                r_values,
                b_values,
                seed=seed_for_ci[0],
                n_permutations=n_permutations,
            )
            delta = float(ci["mean"])
            ci_low, ci_high = float(ci["low"]), float(ci["high"])
            n = int(ci["n"])
        else:
            delta = r_mean - b_mean
            ci_low = ci_high = delta
            p = 1.0
            n = 0
        improvement = delta if higher else -delta
        rows.append(
            {
                "run": run.label,
                "metric": metric,
                "baseline_mean": b_mean,
                "run_mean": r_mean,
                "delta": delta,
                "improvement": improvement,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "paired_p": float(p),
                "n_paired_seeds": n,
                "higher_is_better": higher,
                "description": _metric_description(metric),
                "inference_unit": (
                    "training_seed_with_nested_decoder_means"
                    if paper_mode else "decoder_seed_development_only"
                ),
                "baseline_training_seed_count": len(baseline.training_seeds),
                "run_training_seed_count": len(run.training_seeds),
                "matched_training_seed_count": len(common_training_seeds),
                "paper_significance_eligible": (
                    paper_mode
                    and len(baseline.training_seeds) >= 3
                    and len(run.training_seeds) >= 3
                    and len(common_training_seeds) >= 3
                ),
            }
        )
    return rows


def compare_benchmarks(
    baseline: BenchmarkRun,
    runs: Sequence[BenchmarkRun],
    metrics: Optional[Sequence[str]] = None,
    *,
    n_bootstrap: int = 1000,
    n_permutations: int = 2000,
    require_matching_config: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    """Compare all ``runs`` against ``baseline``."""
    if metrics is None:
        metrics = [spec.name for spec in METRIC_SPECS]
    config_checks = []
    if require_matching_config:
        config_checks = validate_matching_config(baseline, runs, require_matching_config)
    rows: list[dict[str, object]] = []
    for run in runs:
        rows.extend(
            compare_run_to_baseline(
                baseline,
                run,
                metrics,
                n_bootstrap=n_bootstrap,
                n_permutations=n_permutations,
            )
        )
    return {
        "baseline": {"label": baseline.label, "path": baseline.path, "config": dict(baseline.config)},
        "runs": [{"label": run.label, "path": run.path, "config": dict(run.config)} for run in runs],
        "metrics": list(metrics),
        "required_matching_config": list(require_matching_config or []),
        "config_checks": config_checks,
        "rows": rows,
    }


def _fmt(value: object) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(x):
        return ""
    if abs(x) >= 100:
        return f"{x:.2f}"
    if abs(x) >= 1:
        return f"{x:.4f}"
    return f"{x:.5f}"


def write_comparison_table(result: Mapping[str, object], path: str) -> str:
    """Write comparison rows as Markdown."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        baseline = result.get("baseline", {})
        label = baseline.get("label", "baseline") if isinstance(baseline, Mapping) else "baseline"
        fh.write("# mRNA-EditFlow Benchmark Comparison\n\n")
        fh.write(f"Baseline: `{label}`\n\n")
        fh.write("| Run | Metric | Baseline | Run | Delta | Improvement | 95% CI(delta) | paired p | n | Direction |\n")
        fh.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in result.get("rows", []):
            if not isinstance(row, Mapping):
                continue
            direction = "higher" if row.get("higher_is_better", True) else "lower"
            ci = f"[{_fmt(row.get('ci_low'))}, {_fmt(row.get('ci_high'))}]"
            fh.write(
                f"| {row.get('run', '')} | `{row.get('metric', '')}` | "
                f"{_fmt(row.get('baseline_mean'))} | {_fmt(row.get('run_mean'))} | "
                f"{_fmt(row.get('delta'))} | {_fmt(row.get('improvement'))} | "
                f"{ci} | {_fmt(row.get('paired_p'))} | {row.get('n_paired_seeds', 0)} | {direction} |\n"
            )
    return path


def _parse_labeled_path(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise ValueError("expected LABEL=PATH")
    label, path = text.split("=", 1)
    if not label or not path:
        raise ValueError("expected LABEL=PATH")
    return label, path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="LABEL=multiseed_summary.json")
    parser.add_argument("--run", action="append", required=True, help="LABEL=multiseed_summary.json")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--metrics", nargs="*", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument(
        "--require-matching-config",
        nargs="*",
        default=None,
        help=(
            "Config fields that must match between baseline and every run "
            "before writing a paired comparison table."
        ),
    )
    parser.add_argument(
        "--require-default-matching-config",
        action="store_true",
        help=(
            "Require the default paper-grade nuisance fields to match. This "
            "includes effective_proposal_top_k, which maps cascade_recall_top_k "
            "to proposal_top_k for fair cascade-vs-single-ranker comparisons."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    base_label, base_path = _parse_labeled_path(args.baseline)
    baseline = load_benchmark_run(base_label, base_path)
    runs = [load_benchmark_run(*_parse_labeled_path(item)) for item in args.run]
    required_fields = list(args.require_matching_config or [])
    if args.require_default_matching_config:
        required_fields = list(dict.fromkeys([*DEFAULT_REQUIRED_MATCHING_CONFIG, *required_fields]))
    result = compare_benchmarks(
        baseline,
        runs,
        metrics=args.metrics,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        require_matching_config=required_fields or None,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    write_comparison_table(result, args.out_md)
    print(json.dumps({"json_path": args.out_json, "table_path": args.out_md}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BenchmarkRun",
    "DEFAULT_REQUIRED_MATCHING_CONFIG",
    "load_benchmark_run",
    "validate_matching_config",
    "compare_run_to_baseline",
    "compare_benchmarks",
    "write_comparison_table",
    "main",
]
