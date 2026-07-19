"""Per-record cascade-vs-baseline error analysis for mRNA-EditFlow.

The multi-seed benchmark tells us whether a decoder wins on average. This
module answers the next question needed for SOTA chasing:

``where does the cascade win or lose?``

For each source transcript ``i`` and seed ``s`` we compare the paired TE gain

``g_i,s = (TE_cascade_i,s - TE_source_i,s)
        - (TE_baseline_i,s - TE_source_i,s)``

which reduces to ``TE_cascade_i,s - TE_baseline_i,s`` when both runs use the
same source records. We then aggregate ``mean_s g_i,s`` and stratify records by
source features such as UTR length, GC, uAUG/Kozak status and source TE. The
output is a paper-debug artifact for hard-negative teacher mining and v2 ranker
design, not a replacement for the paired benchmark table.

Complexity is ``O(S * N + N log N)`` for ``S`` common seeds and ``N`` records;
the ``N log N`` term comes from quantile binning and top win/loss ranking.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.download_mrna import load_records_jsonl
from mrna_editflow.eval import metrics


@dataclass(frozen=True)
class RecordCascadeComparison:
    """One source transcript aggregated across common benchmark seeds."""

    index: int
    transcript_id: str
    n_seeds: int
    source_te: float
    baseline_delta_te_mean: float
    cascade_delta_te_mean: float
    cascade_gain_mean: float
    cascade_gain_std: float
    cascade_win_fraction: float
    cascade_loss_fraction: float
    five_utr_len: int
    cds_len: int
    three_utr_len: int
    total_len: int
    source_gc: float
    source_kozak_score: float
    source_start_accessibility: float
    source_uaug_presence: int
    source_strong_kozak_presence: int
    source_polyA_presence: int


def _load_json(path: str) -> Mapping[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _seed_dirs(run_dir: str) -> dict[int, str]:
    """Return ``{seed: seed_dir}`` for completed seed directories."""
    out: dict[int, str] = {}
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(run_dir)
    for name in os.listdir(run_dir):
        if not name.startswith("seed_"):
            continue
        try:
            seed = int(name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        seed_dir = os.path.join(run_dir, name)
        if os.path.exists(os.path.join(seed_dir, "eval_summary.json")):
            out[seed] = seed_dir
    return out


def _per_record(summary: Mapping[str, object], key: str) -> list[float]:
    per = summary.get("per_record_metrics", {})
    if not isinstance(per, Mapping) or key not in per:
        raise ValueError(f"eval_summary.json missing per_record_metrics.{key}")
    values = per[key]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"per_record_metrics.{key} must be a sequence")
    out = []
    for value in values:
        x = float(value)  # type: ignore[arg-type]
        if not math.isfinite(x):
            raise ValueError(f"per_record_metrics.{key} contains non-finite value")
        out.append(x)
    return out


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float))) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return float(np.std(np.asarray(values, dtype=float), ddof=1)) if len(values) > 1 else 0.0


def _fraction(flags: Sequence[bool]) -> float:
    return float(sum(bool(x) for x in flags) / max(1, len(flags)))


def _safe_gc(record: MRNARecord) -> float:
    return metrics.gc_fraction(record.seq)


def _binary_motif(record: MRNARecord, motif_name: str) -> int:
    return int(metrics.detect_motifs(record).get(motif_name, 0) > 0)


def _source_feature_rows(
    sources: Sequence[MRNARecord],
    source_te_by_index: Sequence[float],
) -> list[dict[str, object]]:
    """Collect source-only explanatory features for stratified analysis."""
    rows = []
    for i, rec in enumerate(sources):
        kozak = metrics.kozak_uaug_stats([rec])
        rows.append(
            {
                "index": i,
                "transcript_id": rec.transcript_id,
                "five_utr_len": len(rec.five_utr),
                "cds_len": len(rec.cds),
                "three_utr_len": len(rec.three_utr),
                "total_len": len(rec.seq),
                "source_gc": _safe_gc(rec),
                "source_te": float(source_te_by_index[i]),
                "source_kozak_score": float(kozak["mean_kozak_score"]),
                "source_start_accessibility": metrics.start_accessibility_proxy(rec),
                "source_uaug_presence": _binary_motif(rec, "uAUG"),
                "source_strong_kozak_presence": _binary_motif(rec, "strong_kozak"),
                "source_polyA_presence": _binary_motif(rec, "polyA_signal"),
            }
        )
    return rows


def _quantile_bins(values: Sequence[float], labels: tuple[str, str, str] = ("low", "mid", "high")) -> list[str]:
    """Assign deterministic tertile labels; ties stay stable and finite."""
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    q1, q2 = np.quantile(arr, [1.0 / 3.0, 2.0 / 3.0])
    out = []
    for value in arr:
        if value <= q1:
            out.append(labels[0])
        elif value <= q2:
            out.append(labels[1])
        else:
            out.append(labels[2])
    return out


def _record_to_dict(row: RecordCascadeComparison) -> dict[str, object]:
    return {
        "index": row.index,
        "transcript_id": row.transcript_id,
        "n_seeds": row.n_seeds,
        "source_te": row.source_te,
        "baseline_delta_te_mean": row.baseline_delta_te_mean,
        "cascade_delta_te_mean": row.cascade_delta_te_mean,
        "cascade_gain_mean": row.cascade_gain_mean,
        "cascade_gain_std": row.cascade_gain_std,
        "cascade_win_fraction": row.cascade_win_fraction,
        "cascade_loss_fraction": row.cascade_loss_fraction,
        "five_utr_len": row.five_utr_len,
        "cds_len": row.cds_len,
        "three_utr_len": row.three_utr_len,
        "total_len": row.total_len,
        "source_gc": row.source_gc,
        "source_kozak_score": row.source_kozak_score,
        "source_start_accessibility": row.source_start_accessibility,
        "source_uaug_presence": row.source_uaug_presence,
        "source_strong_kozak_presence": row.source_strong_kozak_presence,
        "source_polyA_presence": row.source_polyA_presence,
    }


def _summarize_group(records: Sequence[RecordCascadeComparison], group_name: str, group_value: object) -> dict[str, object]:
    gains = [r.cascade_gain_mean for r in records]
    return {
        "group": group_name,
        "value": group_value,
        "n_records": len(records),
        "mean_cascade_gain": _mean(gains),
        "mean_baseline_delta_te": _mean([r.baseline_delta_te_mean for r in records]),
        "mean_cascade_delta_te": _mean([r.cascade_delta_te_mean for r in records]),
        "win_record_fraction": _fraction([r.cascade_gain_mean > 0.0 for r in records]),
        "loss_record_fraction": _fraction([r.cascade_gain_mean < 0.0 for r in records]),
        "mean_source_te": _mean([r.source_te for r in records]),
    }


def _group_records(records: Sequence[RecordCascadeComparison]) -> list[dict[str, object]]:
    """Create biologically interpretable stratified win/loss summaries."""
    if not records:
        return []
    rows = [_record_to_dict(r) for r in records]
    numeric_bins = {
        "five_utr_len_bin": _quantile_bins([float(r["five_utr_len"]) for r in rows]),
        "three_utr_len_bin": _quantile_bins([float(r["three_utr_len"]) for r in rows]),
        "source_te_bin": _quantile_bins([float(r["source_te"]) for r in rows]),
        "source_gc_bin": _quantile_bins([float(r["source_gc"]) for r in rows]),
    }
    groups: list[dict[str, object]] = []
    for group_name, labels in numeric_bins.items():
        by_value: dict[str, list[RecordCascadeComparison]] = {}
        for rec, label in zip(records, labels):
            by_value.setdefault(label, []).append(rec)
        for value, subset in sorted(by_value.items()):
            groups.append(_summarize_group(subset, group_name, value))
    for group_name in (
        "source_uaug_presence",
        "source_strong_kozak_presence",
        "source_polyA_presence",
    ):
        by_value_int: dict[int, list[RecordCascadeComparison]] = {}
        for rec in records:
            value = int(getattr(rec, group_name))
            by_value_int.setdefault(value, []).append(rec)
        for value, subset in sorted(by_value_int.items()):
            groups.append(_summarize_group(subset, group_name, value))
    return groups


def _load_sources(baseline_dir: str, cascade_dir: str) -> list[MRNARecord]:
    for run_dir in (baseline_dir, cascade_dir):
        path = os.path.join(run_dir, "sources.jsonl")
        if os.path.exists(path):
            return load_records_jsonl(path)
    raise FileNotFoundError("could not find sources.jsonl in either benchmark directory")


def run_cascade_error_analysis(
    *,
    baseline_dir: str,
    cascade_dir: str,
    out_json: str,
    out_jsonl: Optional[str] = None,
    out_md: Optional[str] = None,
    baseline_label: str = "baseline",
    cascade_label: str = "cascade",
    top_n: int = 20,
) -> dict[str, object]:
    """Run per-record win/loss analysis for two completed multiseed runs."""
    baseline_seeds = _seed_dirs(baseline_dir)
    cascade_seeds = _seed_dirs(cascade_dir)
    common_seeds = sorted(set(baseline_seeds) & set(cascade_seeds))
    if not common_seeds:
        raise ValueError("no common completed seeds found")

    sources = _load_sources(baseline_dir, cascade_dir)
    per_record_baseline: list[list[float]] = [[] for _ in sources]
    per_record_cascade: list[list[float]] = [[] for _ in sources]
    per_record_gain: list[list[float]] = [[] for _ in sources]
    source_te_first: Optional[list[float]] = None

    for seed in common_seeds:
        b_summary = _load_json(os.path.join(baseline_seeds[seed], "eval_summary.json"))
        c_summary = _load_json(os.path.join(cascade_seeds[seed], "eval_summary.json"))
        b_te = _per_record(b_summary, "oracle_ensemble_te")
        c_te = _per_record(c_summary, "oracle_ensemble_te")
        b_source = _per_record(b_summary, "source_oracle_ensemble_te")
        c_source = _per_record(c_summary, "source_oracle_ensemble_te")
        n = min(len(sources), len(b_te), len(c_te), len(b_source), len(c_source))
        if source_te_first is None:
            source_te_first = b_source[:n]
        for i in range(n):
            b_delta = float(b_te[i] - b_source[i])
            c_delta = float(c_te[i] - c_source[i])
            per_record_baseline[i].append(b_delta)
            per_record_cascade[i].append(c_delta)
            per_record_gain[i].append(c_delta - b_delta)

    if source_te_first is None:
        raise ValueError("no source TE values were found")
    features = _source_feature_rows(sources, source_te_first)
    records: list[RecordCascadeComparison] = []
    for i, rec in enumerate(sources):
        gains = per_record_gain[i]
        if not gains:
            continue
        feat = features[i]
        records.append(
            RecordCascadeComparison(
                index=i,
                transcript_id=rec.transcript_id,
                n_seeds=len(gains),
                source_te=float(feat["source_te"]),
                baseline_delta_te_mean=_mean(per_record_baseline[i]),
                cascade_delta_te_mean=_mean(per_record_cascade[i]),
                cascade_gain_mean=_mean(gains),
                cascade_gain_std=_std(gains),
                cascade_win_fraction=_fraction([x > 0.0 for x in gains]),
                cascade_loss_fraction=_fraction([x < 0.0 for x in gains]),
                five_utr_len=int(feat["five_utr_len"]),
                cds_len=int(feat["cds_len"]),
                three_utr_len=int(feat["three_utr_len"]),
                total_len=int(feat["total_len"]),
                source_gc=float(feat["source_gc"]),
                source_kozak_score=float(feat["source_kozak_score"]),
                source_start_accessibility=float(feat["source_start_accessibility"]),
                source_uaug_presence=int(feat["source_uaug_presence"]),
                source_strong_kozak_presence=int(feat["source_strong_kozak_presence"]),
                source_polyA_presence=int(feat["source_polyA_presence"]),
            )
        )

    gains = [r.cascade_gain_mean for r in records]
    sorted_records = sorted(records, key=lambda r: r.cascade_gain_mean, reverse=True)
    top_wins = [_record_to_dict(r) for r in sorted_records[: int(top_n)]]
    top_losses = [_record_to_dict(r) for r in sorted(records, key=lambda r: r.cascade_gain_mean)[: int(top_n)]]
    result: dict[str, object] = {
        "config": {
            "baseline_dir": baseline_dir,
            "cascade_dir": cascade_dir,
            "baseline_label": baseline_label,
            "cascade_label": cascade_label,
            "common_seeds": common_seeds,
            "n_records": len(records),
            "top_n": int(top_n),
        },
        "aggregate": {
            "mean_cascade_gain": _mean(gains),
            "std_cascade_gain": _std(gains),
            "win_record_fraction": _fraction([g > 0.0 for g in gains]),
            "loss_record_fraction": _fraction([g < 0.0 for g in gains]),
            "tie_record_fraction": _fraction([g == 0.0 for g in gains]),
            "mean_baseline_delta_te": _mean([r.baseline_delta_te_mean for r in records]),
            "mean_cascade_delta_te": _mean([r.cascade_delta_te_mean for r in records]),
        },
        "groups": _group_records(records),
        "top_wins": top_wins,
        "top_losses": top_losses,
        "jsonl_path": out_jsonl,
        "markdown_path": out_md,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    if out_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)), exist_ok=True)
        with open(out_jsonl, "w", encoding="utf-8") as fh:
            for row in records:
                fh.write(json.dumps(_record_to_dict(row), sort_keys=True) + "\n")
    if out_md:
        write_error_analysis_markdown(result, out_md)
    result["json_path"] = out_json
    return result


def _fmt(value: object) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(x):
        return ""
    if abs(x) >= 1:
        return f"{x:.4f}"
    return f"{x:.5f}"


def write_error_analysis_markdown(result: Mapping[str, object], path: str) -> str:
    """Write a compact Markdown report for the cascade error analysis."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    agg = result.get("aggregate", {})
    cfg = result.get("config", {})
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# mRNA-EditFlow Cascade Error Analysis\n\n")
        if isinstance(cfg, Mapping):
            fh.write(
                f"Baseline: `{cfg.get('baseline_label', 'baseline')}`; "
                f"Cascade: `{cfg.get('cascade_label', 'cascade')}`; "
                f"records: `{cfg.get('n_records', '')}`; "
                f"seeds: `{cfg.get('common_seeds', [])}`\n\n"
            )
        if isinstance(agg, Mapping):
            fh.write("| Metric | Value |\n|---|---:|\n")
            for key in (
                "mean_cascade_gain",
                "std_cascade_gain",
                "win_record_fraction",
                "loss_record_fraction",
                "mean_baseline_delta_te",
                "mean_cascade_delta_te",
            ):
                fh.write(f"| `{key}` | {_fmt(agg.get(key))} |\n")
            fh.write("\n")
        fh.write("## Stratified Groups\n\n")
        fh.write("| Group | Value | n | mean cascade gain | win fraction | baseline delta TE | cascade delta TE |\n")
        fh.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in result.get("groups", []):
            if not isinstance(row, Mapping):
                continue
            fh.write(
                f"| `{row.get('group', '')}` | {row.get('value', '')} | "
                f"{row.get('n_records', '')} | {_fmt(row.get('mean_cascade_gain'))} | "
                f"{_fmt(row.get('win_record_fraction'))} | "
                f"{_fmt(row.get('mean_baseline_delta_te'))} | "
                f"{_fmt(row.get('mean_cascade_delta_te'))} |\n"
            )
        fh.write("\n## Top Cascade Wins\n\n")
        _write_record_table(fh, result.get("top_wins", []))
        fh.write("\n## Top Cascade Losses\n\n")
        _write_record_table(fh, result.get("top_losses", []))
    return path


def _write_record_table(fh, rows: object) -> None:
    fh.write("| Transcript | gain | baseline delta | cascade delta | source TE | 5'UTR | 3'UTR | GC | uAUG |\n")
    fh.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    if not isinstance(rows, Sequence):
        return
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        fh.write(
            f"| `{row.get('transcript_id', '')}` | {_fmt(row.get('cascade_gain_mean'))} | "
            f"{_fmt(row.get('baseline_delta_te_mean'))} | "
            f"{_fmt(row.get('cascade_delta_te_mean'))} | "
            f"{_fmt(row.get('source_te'))} | {row.get('five_utr_len', '')} | "
            f"{row.get('three_utr_len', '')} | {_fmt(row.get('source_gc'))} | "
            f"{row.get('source_uaug_presence', '')} |\n"
        )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--cascade-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-jsonl", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--cascade-label", default="cascade")
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    result = run_cascade_error_analysis(
        baseline_dir=args.baseline_dir,
        cascade_dir=args.cascade_dir,
        out_json=args.out_json,
        out_jsonl=args.out_jsonl,
        out_md=args.out_md,
        baseline_label=args.baseline_label,
        cascade_label=args.cascade_label,
        top_n=args.top_n,
    )
    print(json.dumps({"json_path": result["json_path"], "markdown_path": result.get("markdown_path")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RecordCascadeComparison",
    "run_cascade_error_analysis",
    "write_error_analysis_markdown",
    "main",
]
