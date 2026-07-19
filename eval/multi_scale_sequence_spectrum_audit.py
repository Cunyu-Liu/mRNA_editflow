"""Multi-scale sequence spectrum audit for generated mRNA candidates.

This module compares generated candidates against source/training records with
explicit nucleotide composition, region-wise composition, length/GC histograms,
k-mer spectra and codon-pair spectra. It writes machine-readable JSON, a compact
Markdown report and dependency-free SVG figures.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import html
import json
import math
import os
from collections import Counter
from itertools import product
from typing import Iterable, Mapping, Optional, Sequence

from mrna_editflow.eval.run_eval import load_records
from mrna_editflow.eval.metrics import cds_of, five_utr_of, gc_fraction, sequence_of, three_utr_of


RNA = "ACGU"
CLAIM_POLICY = (
    "Multi-scale sequence spectrum audit is proxy/offline distribution evidence. "
    "It compares generated candidates with source or training records, but does "
    "not establish wet-lab efficacy, external SOTA reproduction, or full de novo "
    "generation."
)


def _valid(seq: str) -> str:
    seq = str(seq or "").upper().replace("T", "U")
    return "".join(ch for ch in seq if ch in RNA)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_records(paths: Sequence[str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        rows.extend(load_records(path))
    return rows


def _source_paths_from_globs(patterns: Sequence[str]) -> list[str]:
    out: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            out.extend(matches)
        elif os.path.exists(pattern):
            out.append(pattern)
    return out


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return xs[0]
    pos = max(0.0, min(1.0, q)) * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _hist(values: Sequence[float], bins: Sequence[float]) -> list[dict[str, object]]:
    if len(bins) < 2:
        raise ValueError("histogram bins need at least two edges")
    counts = [0 for _ in range(len(bins) - 1)]
    for value in values:
        x = float(value)
        placed = False
        for i in range(len(bins) - 1):
            if bins[i] <= x < bins[i + 1]:
                counts[i] += 1
                placed = True
                break
        if not placed and x == bins[-1]:
            counts[-1] += 1
    total = max(1, sum(counts))
    return [
        {
            "bin_start": float(bins[i]),
            "bin_end": float(bins[i + 1]),
            "count": int(counts[i]),
            "fraction": float(counts[i] / total),
        }
        for i in range(len(counts))
    ]


def _auto_bins(values_a: Sequence[float], values_b: Sequence[float], n_bins: int = 20) -> list[float]:
    vals = [float(x) for x in list(values_a) + list(values_b)]
    if not vals:
        return [0.0, 1.0]
    lo = min(vals)
    hi = max(vals)
    if math.isclose(lo, hi):
        pad = max(1.0, abs(lo) * 0.05)
        lo -= pad
        hi += pad
    width = (hi - lo) / max(1, int(n_bins))
    return [lo + width * i for i in range(int(n_bins) + 1)]


def _lengths(records: Sequence[object]) -> list[float]:
    return [float(len(_valid(sequence_of(r)))) for r in records]


def _gc_values(records: Sequence[object]) -> list[float]:
    return [float(gc_fraction(r)) for r in records]


def _composition(records: Sequence[object], region: str = "full") -> dict[str, float]:
    counts: Counter[str] = Counter()
    total = 0
    for record in records:
        if region == "five_utr":
            seq = _valid(five_utr_of(record))
        elif region == "cds":
            seq = _valid(cds_of(record))
        elif region == "three_utr":
            seq = _valid(three_utr_of(record))
        else:
            seq = _valid(sequence_of(record))
        counts.update(seq)
        total += len(seq)
    denom = max(1, total)
    return {base: float(counts.get(base, 0) / denom) for base in RNA}


def _region_lengths(records: Sequence[object]) -> dict[str, dict[str, float]]:
    regions = {
        "five_utr": [float(len(_valid(five_utr_of(r)))) for r in records],
        "cds": [float(len(_valid(cds_of(r)))) for r in records],
        "three_utr": [float(len(_valid(three_utr_of(r)))) for r in records],
    }
    return {
        name: {
            "mean": _mean(vals),
            "q05": _quantile(vals, 0.05),
            "q50": _quantile(vals, 0.50),
            "q95": _quantile(vals, 0.95),
        }
        for name, vals in regions.items()
    }


def _all_kmers(k: int) -> list[str]:
    return ["".join(parts) for parts in product(RNA, repeat=k)]


def _kmer_distribution(records: Sequence[object], k: int) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for record in records:
        seq = _valid(sequence_of(record))
        for i in range(max(0, len(seq) - k + 1)):
            counts[seq[i : i + k]] += 1
    keys = _all_kmers(k)
    total = sum(counts.values())
    if total <= 0:
        return {key: 1.0 / len(keys) for key in keys}
    return {key: float(counts.get(key, 0) / total) for key in keys}


def _codon_pair_distribution(records: Sequence[object]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for record in records:
        cds = _valid(cds_of(record))
        codons = [cds[i : i + 3] for i in range(0, len(cds) - len(cds) % 3, 3)]
        for a, b in zip(codons, codons[1:]):
            if len(a) == 3 and len(b) == 3:
                counts[f"{a}-{b}"] += 1
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {key: float(value / total) for key, value in counts.items()}


def _l1_distance(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    keys = set(a) | set(b)
    return float(sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys))


def _top_delta(
    candidates: Mapping[str, float],
    sources: Mapping[str, float],
    *,
    n: int = 20,
) -> list[dict[str, object]]:
    keys = sorted(
        set(candidates) | set(sources),
        key=lambda key: abs(float(candidates.get(key, 0.0)) - float(sources.get(key, 0.0))),
        reverse=True,
    )[: int(n)]
    return [
        {
            "feature": key,
            "candidate_fraction": float(candidates.get(key, 0.0)),
            "source_fraction": float(sources.get(key, 0.0)),
            "delta": float(candidates.get(key, 0.0) - sources.get(key, 0.0)),
        }
        for key in keys
    ]


def _summary_stats(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": _mean(values),
        "q05": _quantile(values, 0.05),
        "q50": _quantile(values, 0.50),
        "q95": _quantile(values, 0.95),
    }


def _svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#222}.title{font-size:16px;font-weight:bold}.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.cand{fill:#1677ff}.src{fill:#fa8c16}.line-cand{stroke:#1677ff;stroke-width:2;fill:none}.line-src{stroke:#fa8c16;stroke-width:2;fill:none}</style>',
    ]


def _write_grouped_bar_svg(
    path: str,
    labels: Sequence[str],
    cand_values: Sequence[float],
    src_values: Sequence[float],
    *,
    title: str,
    y_label: str = "fraction",
) -> str:
    width, height = 900, 420
    margin_l, margin_r, margin_t, margin_b = 70, 30, 50, 95
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    max_y = max([0.001] + [float(x) for x in cand_values] + [float(x) for x in src_values])
    max_y *= 1.15
    lines = _svg_header(width, height)
    lines.append(f'<text class="title" x="{margin_l}" y="28">{html.escape(title)}</text>')
    lines.append(f'<text x="{margin_l}" y="45"><tspan class="cand">候选</tspan> vs <tspan class="src">source/training</tspan></text>')
    lines.append(f'<line class="axis" x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}"/>')
    lines.append(f'<line class="axis" x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}"/>')
    for tick in range(6):
        frac = tick / 5
        y = margin_t + plot_h - frac * plot_h
        val = frac * max_y
        lines.append(f'<line class="grid" x1="{margin_l}" y1="{y:.1f}" x2="{margin_l + plot_w}" y2="{y:.1f}"/>')
        lines.append(f'<text x="8" y="{y + 4:.1f}">{val:.3f}</text>')
    n = max(1, len(labels))
    group_w = plot_w / n
    bar_w = min(18, group_w * 0.32)
    for i, label in enumerate(labels):
        cx = margin_l + group_w * (i + 0.5)
        for value, cls, dx in ((cand_values[i], "cand", -bar_w * 0.6), (src_values[i], "src", bar_w * 0.6)):
            h = 0.0 if max_y <= 0 else float(value) / max_y * plot_h
            x = cx + dx - bar_w / 2
            y = margin_t + plot_h - h
            lines.append(f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}"/>')
        lines.append(f'<text transform="translate({cx - 4:.1f},{margin_t + plot_h + 16}) rotate(55)">{html.escape(str(label))}</text>')
    lines.append(f'<text transform="translate(18,{margin_t + plot_h / 2:.1f}) rotate(-90)">{html.escape(y_label)}</text>')
    lines.append("</svg>")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _write_hist_svg(path: str, cand_hist: Sequence[Mapping[str, object]], src_hist: Sequence[Mapping[str, object]], *, title: str) -> str:
    labels = [f"{row['bin_start']:.2g}-{row['bin_end']:.2g}" for row in cand_hist]
    cand = [float(row["fraction"]) for row in cand_hist]
    src = [float(row["fraction"]) for row in src_hist]
    return _write_grouped_bar_svg(path, labels, cand, src, title=title)


def _write_top_delta_svg(path: str, rows: Sequence[Mapping[str, object]], *, title: str) -> str:
    labels = [str(row["feature"]) for row in rows]
    cand = [float(row["candidate_fraction"]) for row in rows]
    src = [float(row["source_fraction"]) for row in rows]
    return _write_grouped_bar_svg(path, labels, cand, src, title=title)


def _build_figures(report: Mapping[str, object], out_fig_dir: str) -> dict[str, str]:
    os.makedirs(out_fig_dir, exist_ok=True)
    figures = {}
    base = report["base_composition"]
    full = base["full"]
    figures["base_composition_full_svg"] = _write_grouped_bar_svg(
        os.path.join(out_fig_dir, "base_composition_full.svg"),
        list(RNA),
        [full["candidate"][b] for b in RNA],
        [full["source"][b] for b in RNA],
        title="Full-sequence A/C/G/U composition",
    )
    for region in ("five_utr", "cds", "three_utr"):
        row = base["regions"][region]
        figures[f"base_composition_{region}_svg"] = _write_grouped_bar_svg(
            os.path.join(out_fig_dir, f"base_composition_{region}.svg"),
            list(RNA),
            [row["candidate"][b] for b in RNA],
            [row["source"][b] for b in RNA],
            title=f"{region} A/C/G/U composition",
        )
    figures["length_histogram_svg"] = _write_hist_svg(
        os.path.join(out_fig_dir, "length_histogram.svg"),
        report["length_distribution"]["histogram"]["candidate"],
        report["length_distribution"]["histogram"]["source"],
        title="Full sequence length distribution",
    )
    figures["gc_histogram_svg"] = _write_hist_svg(
        os.path.join(out_fig_dir, "gc_histogram.svg"),
        report["gc_distribution"]["histogram"]["candidate"],
        report["gc_distribution"]["histogram"]["source"],
        title="GC fraction distribution",
    )
    figures["kmer_top_delta_svg"] = _write_top_delta_svg(
        os.path.join(out_fig_dir, "kmer_top_delta.svg"),
        report["kmer_spectrum"]["top_abs_delta"],
        title="Top k-mer spectrum differences",
    )
    figures["codon_pair_top_delta_svg"] = _write_top_delta_svg(
        os.path.join(out_fig_dir, "codon_pair_top_delta.svg"),
        report["codon_pair_spectrum"]["top_abs_delta"],
        title="Top codon-pair spectrum differences",
    )
    return figures


def build_multi_scale_sequence_spectrum_audit(
    *,
    candidate_paths: Sequence[str],
    source_paths: Sequence[str],
    out_fig_dir: Optional[str] = None,
    kmer_k: int = 3,
    top_n: int = 20,
) -> dict[str, object]:
    candidates = _read_records(candidate_paths)
    sources = _read_records(source_paths)
    if not candidates:
        raise ValueError("candidate records are empty")
    if not sources:
        raise ValueError("source records are empty")

    cand_lengths = _lengths(candidates)
    src_lengths = _lengths(sources)
    length_bins = _auto_bins(cand_lengths, src_lengths, n_bins=20)
    cand_gc = _gc_values(candidates)
    src_gc = _gc_values(sources)
    gc_bins = [i / 20 for i in range(21)]
    cand_kmer = _kmer_distribution(candidates, int(kmer_k))
    src_kmer = _kmer_distribution(sources, int(kmer_k))
    cand_pair = _codon_pair_distribution(candidates)
    src_pair = _codon_pair_distribution(sources)
    base_regions = {}
    for region in ("five_utr", "cds", "three_utr"):
        cand_comp = _composition(candidates, region)
        src_comp = _composition(sources, region)
        base_regions[region] = {
            "candidate": cand_comp,
            "source": src_comp,
            "delta": {base: cand_comp[base] - src_comp[base] for base in RNA},
            "l1_distance": _l1_distance(cand_comp, src_comp),
        }
    cand_full_comp = _composition(candidates, "full")
    src_full_comp = _composition(sources, "full")
    report: dict[str, object] = {
        "artifact_kind": "multi_scale_sequence_spectrum_audit",
        "claim_policy": CLAIM_POLICY,
        "inputs": {
            "candidate_paths": [
                {"path": path, "sha256": _sha256(path), "n_records": len(load_records(path))}
                for path in candidate_paths
            ],
            "source_paths": [
                {"path": path, "sha256": _sha256(path), "n_records": len(load_records(path))}
                for path in source_paths
            ],
        },
        "summary": {
            "n_candidates": len(candidates),
            "n_sources": len(sources),
            "ready_for_distribution_figure_audit": True,
            "ready_for_sota_claim": False,
            "base_composition_full_l1": _l1_distance(cand_full_comp, src_full_comp),
            "length_mean_delta": _mean(cand_lengths) - _mean(src_lengths),
            "gc_mean_delta": _mean(cand_gc) - _mean(src_gc),
            "kmer_l1": _l1_distance(cand_kmer, src_kmer),
            "codon_pair_l1": _l1_distance(cand_pair, src_pair),
        },
        "base_composition": {
            "full": {
                "candidate": cand_full_comp,
                "source": src_full_comp,
                "delta": {base: cand_full_comp[base] - src_full_comp[base] for base in RNA},
                "l1_distance": _l1_distance(cand_full_comp, src_full_comp),
            },
            "regions": base_regions,
        },
        "region_lengths": {
            "candidate": _region_lengths(candidates),
            "source": _region_lengths(sources),
        },
        "length_distribution": {
            "candidate": _summary_stats(cand_lengths),
            "source": _summary_stats(src_lengths),
            "histogram": {
                "candidate": _hist(cand_lengths, length_bins),
                "source": _hist(src_lengths, length_bins),
            },
        },
        "gc_distribution": {
            "candidate": _summary_stats(cand_gc),
            "source": _summary_stats(src_gc),
            "histogram": {
                "candidate": _hist(cand_gc, gc_bins),
                "source": _hist(src_gc, gc_bins),
            },
        },
        "kmer_spectrum": {
            "k": int(kmer_k),
            "l1_distance": _l1_distance(cand_kmer, src_kmer),
            "top_abs_delta": _top_delta(cand_kmer, src_kmer, n=top_n),
        },
        "codon_pair_spectrum": {
            "l1_distance": _l1_distance(cand_pair, src_pair),
            "top_abs_delta": _top_delta(cand_pair, src_pair, n=top_n),
        },
    }
    if out_fig_dir:
        report["figures"] = _build_figures(report, out_fig_dir)
    else:
        report["figures"] = {}
    return report


def write_report_json(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return path


def _fmt(value: object, digits: int = 5) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def write_report_markdown(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    summary = report.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    base = report.get("base_composition", {})
    figures = report.get("figures", {})
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Multi-Scale Sequence Spectrum Audit\n\n")
        fh.write(f"- Claim policy: {report.get('claim_policy', CLAIM_POLICY)}\n")
        fh.write(f"- Candidates: `{summary.get('n_candidates')}`; sources: `{summary.get('n_sources')}`\n")
        fh.write(f"- Full base-composition L1: `{_fmt(summary.get('base_composition_full_l1'))}`\n")
        fh.write(f"- Length mean delta: `{_fmt(summary.get('length_mean_delta'))}`\n")
        fh.write(f"- GC mean delta: `{_fmt(summary.get('gc_mean_delta'))}`\n")
        fh.write(f"- k-mer L1: `{_fmt(summary.get('kmer_l1'))}`; codon-pair L1: `{_fmt(summary.get('codon_pair_l1'))}`\n\n")
        fh.write("## Base Composition\n\n")
        fh.write("| region | base | candidate | source | delta |\n")
        fh.write("|---|---|---:|---:|---:|\n")
        full = base.get("full", {}) if isinstance(base, Mapping) else {}
        regions = {"full": full}
        raw_regions = base.get("regions", {}) if isinstance(base, Mapping) else {}
        if isinstance(raw_regions, Mapping):
            regions.update(raw_regions)
        for region, row in regions.items():
            if not isinstance(row, Mapping):
                continue
            cand = row.get("candidate", {})
            src = row.get("source", {})
            delta = row.get("delta", {})
            if not isinstance(cand, Mapping) or not isinstance(src, Mapping) or not isinstance(delta, Mapping):
                continue
            for nt in RNA:
                fh.write(
                    f"| {region} | {nt} | {_fmt(cand.get(nt))} | "
                    f"{_fmt(src.get(nt))} | {_fmt(delta.get(nt))} |\n"
                )
        fh.write("\n## Top k-mer Differences\n\n")
        fh.write("| k-mer | candidate | source | delta |\n")
        fh.write("|---|---:|---:|---:|\n")
        kmer = report.get("kmer_spectrum", {})
        rows = kmer.get("top_abs_delta", []) if isinstance(kmer, Mapping) else []
        if isinstance(rows, Sequence):
            for row in rows[:20]:
                if isinstance(row, Mapping):
                    fh.write(
                        f"| {row.get('feature')} | {_fmt(row.get('candidate_fraction'))} | "
                        f"{_fmt(row.get('source_fraction'))} | {_fmt(row.get('delta'))} |\n"
                    )
        fh.write("\n## Figures\n\n")
        if isinstance(figures, Mapping) and figures:
            for name, fig_path in sorted(figures.items()):
                fh.write(f"- `{name}`: `{fig_path}`\n")
        else:
            fh.write("- No figures were requested.\n")
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="*", default=[])
    parser.add_argument("--candidate-glob", nargs="*", default=[])
    parser.add_argument("--sources", nargs="*", default=[])
    parser.add_argument("--source-glob", nargs="*", default=[])
    parser.add_argument("--kmer-k", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--out-json", default="benchmark/multi_scale_sequence_spectrum_audit.json")
    parser.add_argument("--out-md", default="benchmark/multi_scale_sequence_spectrum_audit.md")
    parser.add_argument("--out-fig-dir", default="benchmark/multi_scale_sequence_spectrum_figures")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    candidate_paths = list(args.candidates) + _source_paths_from_globs(args.candidate_glob)
    source_paths = list(args.sources) + _source_paths_from_globs(args.source_glob)
    if not candidate_paths:
        raise SystemExit("at least one --candidates or --candidate-glob path is required")
    if not source_paths:
        raise SystemExit("at least one --sources or --source-glob path is required")
    report = build_multi_scale_sequence_spectrum_audit(
        candidate_paths=candidate_paths,
        source_paths=source_paths,
        out_fig_dir=args.out_fig_dir,
        kmer_k=args.kmer_k,
        top_n=args.top_n,
    )
    write_report_json(report, args.out_json)
    write_report_markdown(report, args.out_md)
    print(json.dumps({"out_json": args.out_json, "out_md": args.out_md, "out_fig_dir": args.out_fig_dir, "summary": report["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_multi_scale_sequence_spectrum_audit",
    "write_report_json",
    "write_report_markdown",
]
