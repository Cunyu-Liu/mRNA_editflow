"""Summarize T1 runtime/throughput evidence from multiseed progress logs.

The report is an audit artifact, not a strict hardware benchmark. It separates
fully measured seed runtimes from resumed seeds, because resumed progress logs
can include scheduler gaps or previous work that should not be treated as model
throughput.
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
    "Runtime is measured from existing progress logs. Fully measured seed "
    "durations support approximate throughput; resumed seeds and observed "
    "wall-clock spans are audit evidence, not strict hardware-normalized speed "
    "claims."
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
    "legal_fraction",
    "mean_oracle_te",
    "delta_oracle_te_vs_source",
    "mean_oracle_mrl",
    "mean_protein_identity",
    "within_budget_fraction",
    "reading_frame_intact_fraction",
)


def _load_json(path: str) -> Mapping[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _load_jsonl(path: str) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                rows.append(payload)
    return rows


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: str, root: str) -> str:
    return os.path.relpath(path, root) if os.path.isabs(path) else path


def _as_number(value: object) -> Optional[float]:
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


def _resolve_path(project_root: str, summary_dir: str, raw: object, fallback_name: str) -> Optional[str]:
    candidates = []
    if isinstance(raw, str) and raw:
        candidates.append(raw)
        marker = "/mrna_editflow/"
        if marker in raw:
            candidates.append(os.path.join(project_root, raw.split(marker, 1)[1]))
    candidates.append(os.path.join(summary_dir, fallback_name))
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _summarize_numbers(values: Sequence[float]) -> dict[str, object]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "sum": sum(values),
    }


def _time(row: Mapping[str, object]) -> Optional[float]:
    return _as_number(row.get("time"))


def _event_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        event = row.get("event")
        if isinstance(event, str):
            out[event] = out.get(event, 0) + 1
    return out


def _runtime_from_progress(
    *,
    rows: Sequence[Mapping[str, object]],
    n_records: Optional[int],
) -> dict[str, object]:
    times = [_time(row) for row in rows]
    times = [value for value in times if value is not None]
    by_seed: dict[int, dict[str, Mapping[str, object]]] = {}
    complete_time = None
    for row in rows:
        if row.get("event") == "benchmark_complete":
            t = _time(row)
            if t is not None:
                complete_time = t
        seed = row.get("seed")
        try:
            seed_int = int(seed)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        event = row.get("event")
        if isinstance(event, str):
            by_seed.setdefault(seed_int, {})[event] = row

    measured_seed_rows = []
    generation_s = []
    eval_s = []
    seed_total_s = []
    n_candidates_measured = 0
    resumed_seeds = []
    measured_with_resume_marker = []
    incomplete_seeds = []
    for seed, events in sorted(by_seed.items()):
        start = _time(events.get("seed_start", {}))
        written = _time(events.get("seed_candidates_written", {}))
        evaluated = _time(events.get("seed_evaluated", {}))
        if start is None or written is None or evaluated is None:
            if "seed_resumed" in events:
                resumed_seeds.append(seed)
                continue
            incomplete_seeds.append(seed)
            continue
        if "seed_resumed" in events:
            measured_with_resume_marker.append(seed)
        gen = max(written - start, 0.0)
        ev = max(evaluated - written, 0.0)
        total = max(evaluated - start, 0.0)
        n_candidates = events.get("seed_candidates_written", {}).get("n_candidates")
        n_cand = int(n_candidates) if isinstance(n_candidates, (int, float)) else n_records
        if n_cand is None:
            n_cand = 0
        n_candidates_measured += int(n_cand)
        generation_s.append(gen)
        eval_s.append(ev)
        seed_total_s.append(total)
        measured_seed_rows.append(
            {
                "seed": seed,
                "n_candidates": int(n_cand),
                "generation_s": gen,
                "evaluation_s": ev,
                "seed_total_s": total,
                "records_per_s_total": (float(n_cand) / total) if total > 0 and n_cand else None,
            }
        )
    first_time = min(times) if times else None
    observed_elapsed_s = None
    if first_time is not None and complete_time is not None:
        observed_elapsed_s = max(complete_time - first_time, 0.0)
    measured_total_s = sum(seed_total_s)
    return {
        "event_counts": _event_counts(rows),
        "observed_elapsed_s": observed_elapsed_s,
        "observed_elapsed_scope": (
            "mixed_resume_wall_clock"
            if resumed_seeds
            else (
                "complete_seed_runtime_with_resume_markers"
                if measured_with_resume_marker
                else "complete_run_wall_clock"
            )
        ),
        "n_seed_events": len(by_seed),
        "measured_seeds": [row["seed"] for row in measured_seed_rows],
        "resumed_seeds": resumed_seeds,
        "measured_with_resume_marker": measured_with_resume_marker,
        "incomplete_seeds": incomplete_seeds,
        "n_measured_seeds": len(measured_seed_rows),
        "n_resumed_seeds": len(resumed_seeds),
        "n_incomplete_seeds": len(incomplete_seeds),
        "n_candidates_measured": n_candidates_measured,
        "generation_s": _summarize_numbers(generation_s),
        "evaluation_s": _summarize_numbers(eval_s),
        "seed_total_s": _summarize_numbers(seed_total_s),
        "measured_records_per_s_total": (
            n_candidates_measured / measured_total_s if measured_total_s > 0 else None
        ),
        "measured_seed_rows": measured_seed_rows,
    }


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
    summary_dir = os.path.dirname(summary_path)
    config = summary.get("config", {})
    if not isinstance(config, Mapping):
        config = {}
    progress_path = _resolve_path(
        project_root,
        summary_dir,
        config.get("progress_jsonl") or summary.get("progress_jsonl"),
        "multiseed_progress.jsonl",
    )
    context = {
        metric: entry
        for metric in CONTEXT_METRICS
        for entry in [_mean_entry(summary, metric)]
        if entry is not None
    }
    complete_summary = _summary_has_expected_seeds(summary, expected_seeds)
    if progress_path is None:
        return {
            **base,
            "status": "missing_progress",
            "summary_sha256": _sha256_file(summary_path),
            "config": dict(config),
            "context": context,
            "runtime": None,
        }
    rows = _load_jsonl(progress_path)
    runtime = _runtime_from_progress(rows=rows, n_records=_as_int(config.get("n_records")))
    status = "complete" if complete_summary and runtime["n_measured_seeds"] else "partial"
    return {
        **base,
        "status": status,
        "summary_sha256": _sha256_file(summary_path),
        "progress_path": _rel(progress_path, project_root),
        "progress_sha256": _sha256_file(progress_path),
        "config": {
            "n_records": config.get("n_records"),
            "seeds": config.get("seeds"),
            "edit_budget": config.get("edit_budget"),
            "effective_proposal_top_k": config.get("effective_proposal_top_k"),
            "decoder_family": config.get("decoder_family"),
        },
        "context": context,
        "runtime": runtime,
    }


def _as_int(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return int(value)
    return None


def build_t1_runtime_report(
    *,
    project_root: str,
    run_specs: Sequence[Mapping[str, str]] = DEFAULT_RUN_SPECS,
    expected_seeds: Sequence[int] = EXPECTED_SEEDS,
) -> dict[str, object]:
    rows = [
        summarize_run(project_root=project_root, spec=spec, expected_seeds=expected_seeds)
        for spec in run_specs
    ]
    return {
        "artifact_kind": "t1_runtime_report",
        "project_root": os.path.abspath(project_root),
        "claim_policy": CLAIM_POLICY,
        "expected_seeds": [int(seed) for seed in expected_seeds],
        "rows": rows,
        "interpretation": {
            "runtime_scope": (
                "Use measured seed runtimes for approximate throughput. Runs "
                "with resumed seeds should not be used for strict wall-clock "
                "comparisons."
            ),
            "strict_hardware_benchmark_ready": False,
        },
    }


def write_report_json(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return path


def _fmt(value: object, digits: int = 2) -> str:
    number = _as_number(value)
    if number is None:
        return "NA"
    return f"{number:.{digits}f}"


def _mean(row: Mapping[str, object], group_name: str, metric: str, key: str = "mean") -> Optional[float]:
    group = row.get(group_name)
    if not isinstance(group, Mapping):
        return None
    entry = group.get(metric)
    if not isinstance(entry, Mapping):
        return None
    return _as_number(entry.get(key))


def _runtime_value(row: Mapping[str, object], group: str, key: str = "mean") -> Optional[float]:
    runtime = row.get("runtime")
    if not isinstance(runtime, Mapping):
        return None
    entry = runtime.get(group)
    if not isinstance(entry, Mapping):
        return None
    return _as_number(entry.get(key))


def write_report_markdown(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows = report.get("rows", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        rows = []
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# T1 Runtime Audit\n\n")
        fh.write(f"- Claim policy: {report.get('claim_policy', CLAIM_POLICY)}\n")
        fh.write(
            "- Scope: measured seed runtimes are approximate throughput evidence; "
            "runs with resumed seeds are not strict wall-clock comparisons.\n\n"
        )
        fh.write(
            "| run | status | n | delta TE | mean TE | measured/resumed seeds | "
            "observed elapsed s | mean gen s | mean eval s | mean seed total s | measured records/s | scope |\n"
        )
        fh.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            cfg = row.get("config", {})
            n_records = cfg.get("n_records") if isinstance(cfg, Mapping) else "NA"
            runtime = row.get("runtime", {})
            if not isinstance(runtime, Mapping):
                runtime = {}
            measured = runtime.get("n_measured_seeds", "NA")
            resumed = runtime.get("n_resumed_seeds", "NA")
            fh.write(
                f"| {row.get('label', '')} | `{row.get('status', '')}` | {n_records} | "
                f"{_fmt(_mean(row, 'context', 'delta_oracle_te_vs_source'), 5)} | "
                f"{_fmt(_mean(row, 'context', 'mean_oracle_te'), 5)} | "
                f"{measured}/{resumed} | "
                f"{_fmt(runtime.get('observed_elapsed_s'), 2)} | "
                f"{_fmt(_runtime_value(row, 'generation_s'), 2)} | "
                f"{_fmt(_runtime_value(row, 'evaluation_s'), 2)} | "
                f"{_fmt(_runtime_value(row, 'seed_total_s'), 2)} | "
                f"{_fmt(runtime.get('measured_records_per_s_total'), 3)} | "
                f"{runtime.get('observed_elapsed_scope', 'NA')} |\n"
            )
        fh.write("\n## Interpretation\n\n")
        fh.write(
            "- Strict hardware benchmark ready: "
            f"`{report.get('interpretation', {}).get('strict_hardware_benchmark_ready') if isinstance(report.get('interpretation'), Mapping) else False}`\n"
        )
        fh.write(
            "- Runtime values are useful for methods/reporting context, but not for "
            "claiming speed SOTA without matched hardware and clean non-resumed runs.\n"
        )
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--out-json", default="benchmark/t1_runtime_report_head256_head1024.json")
    parser.add_argument("--out-md", default="benchmark/t1_runtime_report_head256_head1024.md")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    project_root = os.path.abspath(args.project_root)
    report = build_t1_runtime_report(project_root=project_root)
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
    "build_t1_runtime_report",
    "summarize_run",
    "write_report_json",
    "write_report_markdown",
    "main",
]
