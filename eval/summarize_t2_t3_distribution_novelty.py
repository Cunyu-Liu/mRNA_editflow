"""Summarize T2/T3 distribution and novelty evidence from multiseed runs.

This module turns completed ``run_multiseed_benchmark`` artifacts into a compact
paper-facing audit for T2 distribution preservation and T3 novelty/diversity.
It is intentionally read-only with respect to benchmark inputs and records when
pairwise diversity is sampled rather than exact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from typing import Mapping, Optional, Sequence


CLAIM_POLICY = (
    "T2/T3 metrics are proxy/offline distribution and novelty audits. Low "
    "distribution distance and non-zero novelty do not prove wet-lab efficacy, "
    "external SOTA reproduction, or full de novo generation."
)
EXPECTED_SEEDS: tuple[int, ...] = tuple(range(10))
DEFAULT_RUN_SPECS: tuple[dict[str, str], ...] = (
    {
        "label": "head256_mo_grpo",
        "slice": "head256",
        "decoder": "mo_grpo",
        "role": "primary_head256",
        "summary": "benchmark/multiseed_t5_public_head256_mo_grpo_top64/multiseed_summary.json",
    },
    {
        "label": "head256_mo_scalar",
        "slice": "head256",
        "decoder": "mo_scalar",
        "role": "fusion_head256",
        "summary": "benchmark/multiseed_t5_public_head256_mo_scalar_top64/multiseed_summary.json",
    },
    {
        "label": "head256_mo_pareto",
        "slice": "head256",
        "decoder": "mo_pareto",
        "role": "fusion_head256",
        "summary": "benchmark/multiseed_t5_public_head256_mo_pareto_top64/multiseed_summary.json",
    },
    {
        "label": "head256_mo_te_only",
        "slice": "head256",
        "decoder": "mo_te_only",
        "role": "te_control_head256",
        "summary": "benchmark/multiseed_t5_public_head256_mo_te_only_top64/multiseed_summary.json",
    },
    {
        "label": "head256_hardneg_v2",
        "slice": "head256",
        "decoder": "hardneg_v2",
        "role": "prior_champion_head256",
        "summary": "benchmark/multiseed_t5_public_head256_hardneg_v2_top64/multiseed_summary.json",
    },
    {
        "label": "head1024_mo_pareto",
        "slice": "head1024",
        "decoder": "mo_pareto",
        "role": "primary_head1024_borderline",
        "summary": "benchmark/multiseed_t5_public_head1024_mo_pareto_top64/multiseed_summary.json",
    },
    {
        "label": "head1024_mo_scalar",
        "slice": "head1024",
        "decoder": "mo_scalar",
        "role": "fusion_head1024",
        "summary": "benchmark/multiseed_t5_public_head1024_mo_scalar_top64/multiseed_summary.json",
    },
    {
        "label": "head1024_mo_grpo",
        "slice": "head1024",
        "decoder": "mo_grpo",
        "role": "fusion_head1024",
        "summary": "benchmark/multiseed_t5_public_head1024_mo_grpo_top64/multiseed_summary.json",
    },
    {
        "label": "head1024_mo_te_only",
        "slice": "head1024",
        "decoder": "mo_te_only",
        "role": "te_control_head1024",
        "summary": "benchmark/multiseed_t5_public_head1024_mo_te_only_top64/multiseed_summary.json",
    },
    {
        "label": "head1024_hardneg_v2",
        "slice": "head1024",
        "decoder": "hardneg_v2",
        "role": "prior_champion_head1024",
        "summary": "benchmark/multiseed_t5_public_head1024_hardneg_v2_top64/multiseed_summary.json",
    },
)
CONTEXT_METRICS: tuple[str, ...] = (
    "delta_oracle_te_vs_source",
    "mean_oracle_te",
    "legal_fraction",
    "mean_protein_identity",
    "within_budget_fraction",
    "reading_frame_intact_fraction",
)
T2_METRICS: tuple[str, ...] = (
    "kmer_js",
    "codon_usage_kl",
    "candidate_mean_gc",
    "source_mean_gc",
    "candidate_mean_length",
    "source_mean_length",
    "gc_quantile_distance",
    "length_quantile_distance",
    "combined_gc_length_distance",
    "embedding_frechet_proxy",
)
T3_METRICS: tuple[str, ...] = (
    "mean_novelty",
    "exact_source_match_fraction",
    "unique_fraction",
    "pairwise_diversity",
    "pairwise_diversity_exact",
    "pairwise_pairs_total",
    "pairwise_pairs_evaluated",
    "novelty_exact",
    "novelty_source_comparisons",
    "novelty_sources_total",
    "novelty_sources_evaluated_cap",
)


def _load_json(path: str) -> Mapping[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: str, root: str) -> str:
    return os.path.relpath(path, root) if os.path.isabs(path) else path


def _as_number(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _mean_entry(summary: Mapping[str, object], metric: str) -> Optional[dict[str, object]]:
    aggregate = summary.get("aggregate", {})
    if not isinstance(aggregate, Mapping):
        return None
    entry = aggregate.get(metric)
    if not isinstance(entry, Mapping):
        return None
    mean = _as_number(entry.get("mean"))
    if mean is None:
        return None
    out = {"mean": mean}
    for key in ("std", "low", "high", "n"):
        value = _as_number(entry.get(key))
        if value is not None:
            out[key] = int(value) if key == "n" else value
    return out


def _summary_has_expected_seeds(
    summary: Mapping[str, object],
    expected_seeds: Sequence[int] = EXPECTED_SEEDS,
) -> bool:
    expected = sorted(int(seed) for seed in expected_seeds)
    config = summary.get("config", {})
    per_seed = summary.get("per_seed", [])
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
    if not isinstance(per_seed, Sequence) or isinstance(per_seed, (str, bytes)):
        return False
    found = []
    for row in per_seed:
        if not isinstance(row, Mapping):
            return False
        try:
            found.append(int(row.get("seed")))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
    return sorted(found) == expected


def _resolve_eval_path(
    *,
    project_root: str,
    summary_dir: str,
    row: Mapping[str, object],
) -> Optional[str]:
    raw = row.get("eval_json_path")
    candidates = []
    if isinstance(raw, str) and raw:
        candidates.append(raw)
        marker = "/mrna_editflow/"
        if marker in raw:
            candidates.append(os.path.join(project_root, raw.split(marker, 1)[1]))
    seed = row.get("seed")
    try:
        seed_int = int(seed)  # type: ignore[arg-type]
        candidates.append(os.path.join(summary_dir, f"seed_{seed_int:03d}", "eval_summary.json"))
    except (TypeError, ValueError):
        pass
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _metric_group(eval_summary: Mapping[str, object], task_key: str, metric_key: str) -> Mapping[str, object]:
    task_metrics = eval_summary.get("task_metrics", {})
    if isinstance(task_metrics, Mapping):
        task_group = task_metrics.get(task_key)
        if isinstance(task_group, Mapping):
            return task_group
    metrics = eval_summary.get("metrics", {})
    if isinstance(metrics, Mapping):
        metric_group = metrics.get(metric_key)
        if isinstance(metric_group, Mapping):
            return metric_group
    return {}


def _summarize_values(values: Sequence[object]) -> Optional[dict[str, object]]:
    if not values:
        return None
    if all(isinstance(value, bool) for value in values):
        true_count = sum(1 for value in values if value)
        return {
            "n": len(values),
            "all_true": true_count == len(values),
            "true_fraction": true_count / len(values),
        }
    numbers = [_as_number(value) for value in values]
    numbers = [value for value in numbers if value is not None]
    if not numbers:
        return None
    return {
        "n": len(numbers),
        "mean": statistics.mean(numbers),
        "std": statistics.pstdev(numbers) if len(numbers) > 1 else 0.0,
        "low": min(numbers),
        "high": max(numbers),
    }


def _collect_seed_metrics(
    *,
    project_root: str,
    summary_path: str,
    summary: Mapping[str, object],
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]], list[int], list[int]]:
    summary_dir = os.path.dirname(summary_path)
    per_seed = summary.get("per_seed", [])
    if not isinstance(per_seed, Sequence) or isinstance(per_seed, (str, bytes)):
        return {}, {}, [], []
    t2_values: dict[str, list[object]] = {metric: [] for metric in T2_METRICS}
    t3_values: dict[str, list[object]] = {metric: [] for metric in T3_METRICS}
    found_seeds = []
    missing_seeds = []
    for row in per_seed:
        if not isinstance(row, Mapping):
            continue
        try:
            seed = int(row.get("seed"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        path = _resolve_eval_path(project_root=project_root, summary_dir=summary_dir, row=row)
        if path is None:
            missing_seeds.append(seed)
            continue
        eval_summary = _load_json(path)
        found_seeds.append(seed)
        t2_group = _metric_group(eval_summary, "T2", "distribution")
        t3_group = _metric_group(eval_summary, "T3", "diversity_novelty")
        for metric in T2_METRICS:
            if metric in t2_group:
                t2_values[metric].append(t2_group[metric])
        for metric in T3_METRICS:
            if metric in t3_group:
                t3_values[metric].append(t3_group[metric])
    t2 = {
        metric: stats
        for metric, values in t2_values.items()
        for stats in [_summarize_values(values)]
        if stats is not None
    }
    t3 = {
        metric: stats
        for metric, values in t3_values.items()
        for stats in [_summarize_values(values)]
        if stats is not None
    }
    return t2, t3, sorted(found_seeds), sorted(missing_seeds)


def _add_aggregate_fallbacks(
    metrics: dict[str, dict[str, object]],
    summary: Mapping[str, object],
    names: Sequence[str],
) -> None:
    for name in names:
        if name not in metrics:
            entry = _mean_entry(summary, name)
            if entry is not None:
                metrics[name] = entry


def _metric_mean(row: Mapping[str, object], section: str, metric: str) -> Optional[float]:
    group = row.get(section, {})
    if not isinstance(group, Mapping):
        return None
    entry = group.get(metric, {})
    if not isinstance(entry, Mapping):
        return None
    value = entry.get("mean")
    return _as_number(value)


def _metric_bool(row: Mapping[str, object], section: str, metric: str, key: str) -> Optional[bool]:
    group = row.get(section, {})
    if not isinstance(group, Mapping):
        return None
    entry = group.get(metric, {})
    if not isinstance(entry, Mapping):
        return None
    value = entry.get(key)
    return bool(value) if isinstance(value, bool) else None


def summarize_run(
    *,
    project_root: str,
    spec: Mapping[str, str],
    expected_seeds: Sequence[int] = EXPECTED_SEEDS,
) -> dict[str, object]:
    summary_rel = spec["summary"]
    summary_path = summary_rel if os.path.isabs(summary_rel) else os.path.join(project_root, summary_rel)
    base = {
        "label": spec["label"],
        "slice": spec["slice"],
        "decoder": spec["decoder"],
        "role": spec.get("role", ""),
        "summary_path": _rel(summary_path, project_root),
    }
    if not os.path.exists(summary_path):
        return {**base, "status": "missing", "missing_reason": "summary_not_found"}
    summary = _load_json(summary_path)
    config = summary.get("config", {})
    if not isinstance(config, Mapping):
        config = {}
    t2, t3, found_seed_evals, missing_seed_evals = _collect_seed_metrics(
        project_root=project_root,
        summary_path=summary_path,
        summary=summary,
    )
    _add_aggregate_fallbacks(t2, summary, T2_METRICS)
    _add_aggregate_fallbacks(t3, summary, T3_METRICS)
    context = {
        metric: entry
        for metric in CONTEXT_METRICS
        for entry in [_mean_entry(summary, metric)]
        if entry is not None
    }
    status = "complete" if _summary_has_expected_seeds(summary, expected_seeds) else "partial"
    if missing_seed_evals:
        status = "partial_seed_eval"
    return {
        **base,
        "status": status,
        "summary_sha256": _sha256_file(summary_path),
        "config": {
            "n_records": config.get("n_records"),
            "seeds": config.get("seeds"),
            "max_pairwise_pairs": config.get("max_pairwise_pairs"),
            "max_novelty_sources": config.get("max_novelty_sources"),
            "edit_budget": config.get("edit_budget"),
            "effective_proposal_top_k": config.get("effective_proposal_top_k"),
        },
        "context": context,
        "T2_distribution": t2,
        "T3_novelty_diversity": t3,
        "seed_eval_audit": {
            "found_seed_eval_summaries": found_seed_evals,
            "missing_seed_eval_summaries": missing_seed_evals,
            "n_found": len(found_seed_evals),
            "n_missing": len(missing_seed_evals),
        },
    }


def build_t2_t3_report(
    *,
    project_root: str,
    run_specs: Sequence[Mapping[str, str]] = DEFAULT_RUN_SPECS,
    expected_seeds: Sequence[int] = EXPECTED_SEEDS,
) -> dict[str, object]:
    rows = [
        summarize_run(project_root=project_root, spec=spec, expected_seeds=expected_seeds)
        for spec in run_specs
    ]
    primary = next((row for row in rows if row.get("label") == "head256_mo_grpo"), None)
    interpretation = {
        "claim_policy": CLAIM_POLICY,
        "primary_head256_distribution_collapse_flag": None,
        "primary_head256_de_novo_overclaim_flag": None,
        "pairwise_diversity_scope": (
            "pairwise_diversity_exact=false means the metric is sampled using "
            "max_pairwise_pairs, not all candidate pairs."
        ),
    }
    if isinstance(primary, Mapping):
        kmer_js = _metric_mean(primary, "T2_distribution", "kmer_js")
        combined = _metric_mean(primary, "T2_distribution", "combined_gc_length_distance")
        exact_match = _metric_mean(primary, "T3_novelty_diversity", "exact_source_match_fraction")
        interpretation["primary_head256_distribution_collapse_flag"] = bool(
            kmer_js is not None
            and combined is not None
            and not (kmer_js < 1e-3 and combined < 1e-2)
        )
        interpretation["primary_head256_de_novo_overclaim_flag"] = bool(
            exact_match is not None and exact_match > 0.0
        )
    return {
        "artifact_kind": "t2_t3_distribution_novelty_report",
        "project_root": os.path.abspath(project_root),
        "claim_policy": CLAIM_POLICY,
        "expected_seeds": [int(seed) for seed in expected_seeds],
        "rows": rows,
        "interpretation": interpretation,
    }


def write_report_json(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return path


def _fmt(value: object, digits: int = 5) -> str:
    number = _as_number(value)
    if number is None:
        return "NA"
    if abs(number) >= 1:
        return f"{number:.4f}"
    return f"{number:.{digits}f}"


def _display(row: Mapping[str, object], section: str, metric: str) -> str:
    value = _metric_mean(row, section, metric)
    return _fmt(value)


def _display_bool(row: Mapping[str, object], section: str, metric: str, key: str) -> str:
    value = _metric_bool(row, section, metric, key)
    return "NA" if value is None else str(value)


def write_report_markdown(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows = report.get("rows", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        rows = []
    interpretation = report.get("interpretation", {})
    if not isinstance(interpretation, Mapping):
        interpretation = {}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# T2/T3 Distribution And Novelty Audit\n\n")
        fh.write(f"- Claim policy: {report.get('claim_policy', CLAIM_POLICY)}\n")
        fh.write(
            "- Pairwise note: pairwise diversity can be sampled when "
            "`pairwise_diversity_exact=False`; use `pairwise_pairs_evaluated` "
            "and `pairwise_pairs_total` to state the scope.\n"
        )
        fh.write(
            "- Primary caution: non-zero exact source match means this is local "
            "optimization/rewriting evidence, not de novo generation evidence.\n\n"
        )
        fh.write(
            "| run | status | n | delta TE | kmer JS | codon KL | GC delta | length delta | "
            "GC/length dist | novelty | exact match | unique | pairwise diversity | pairwise scope | novelty exact |\n"
        )
        fh.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|\n")
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            cfg = row.get("config", {})
            n_records = cfg.get("n_records") if isinstance(cfg, Mapping) else "NA"
            cand_gc = _metric_mean(row, "T2_distribution", "candidate_mean_gc")
            src_gc = _metric_mean(row, "T2_distribution", "source_mean_gc")
            cand_len = _metric_mean(row, "T2_distribution", "candidate_mean_length")
            src_len = _metric_mean(row, "T2_distribution", "source_mean_length")
            gc_delta = cand_gc - src_gc if cand_gc is not None and src_gc is not None else None
            len_delta = cand_len - src_len if cand_len is not None and src_len is not None else None
            pair_eval = _metric_mean(row, "T3_novelty_diversity", "pairwise_pairs_evaluated")
            pair_total = _metric_mean(row, "T3_novelty_diversity", "pairwise_pairs_total")
            pair_exact = _display_bool(
                row, "T3_novelty_diversity", "pairwise_diversity_exact", "all_true"
            )
            pair_scope = f"{_fmt(pair_eval, 0)}/{_fmt(pair_total, 0)} exact={pair_exact}"
            fh.write(
                f"| {row.get('label', '')} | `{row.get('status', '')}` | {n_records} | "
                f"{_display(row, 'context', 'delta_oracle_te_vs_source')} | "
                f"{_display(row, 'T2_distribution', 'kmer_js')} | "
                f"{_display(row, 'T2_distribution', 'codon_usage_kl')} | "
                f"{_fmt(gc_delta)} | {_fmt(len_delta)} | "
                f"{_display(row, 'T2_distribution', 'combined_gc_length_distance')} | "
                f"{_display(row, 'T3_novelty_diversity', 'mean_novelty')} | "
                f"{_display(row, 'T3_novelty_diversity', 'exact_source_match_fraction')} | "
                f"{_display(row, 'T3_novelty_diversity', 'unique_fraction')} | "
                f"{_display(row, 'T3_novelty_diversity', 'pairwise_diversity')} | "
                f"{pair_scope} | "
                f"{_display_bool(row, 'T3_novelty_diversity', 'novelty_exact', 'all_true')} |\n"
            )
        fh.write("\n## Interpretation\n\n")
        fh.write(
            f"- Primary head256 distribution-collapse flag: "
            f"`{interpretation.get('primary_head256_distribution_collapse_flag')}`\n"
        )
        fh.write(
            f"- Primary head256 de-novo overclaim flag: "
            f"`{interpretation.get('primary_head256_de_novo_overclaim_flag')}`\n"
        )
        fh.write(f"- Scope: {interpretation.get('pairwise_diversity_scope', '')}\n")
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument(
        "--out-json",
        default="benchmark/t2_t3_distribution_novelty_report_head256_head1024.json",
    )
    parser.add_argument(
        "--out-md",
        default="benchmark/t2_t3_distribution_novelty_report_head256_head1024.md",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    project_root = os.path.abspath(args.project_root)
    report = build_t2_t3_report(project_root=project_root)
    out_json = args.out_json if os.path.isabs(args.out_json) else os.path.join(project_root, args.out_json)
    out_md = args.out_md if os.path.isabs(args.out_md) else os.path.join(project_root, args.out_md)
    write_report_json(report, out_json)
    write_report_markdown(report, out_md)
    print(json.dumps({"json_path": out_json, "markdown_path": out_md}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLAIM_POLICY",
    "DEFAULT_RUN_SPECS",
    "build_t2_t3_report",
    "summarize_run",
    "write_report_json",
    "write_report_markdown",
    "main",
]
