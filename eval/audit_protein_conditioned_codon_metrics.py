"""Codon-level audit for protein-conditioned CDS design artifacts.

This audit complements the T4 CAI/GC report with metrics that match
protein-conditioned CDS generators such as codonGPT or Prot2RNA more directly:
native codon recovery, synonymous substitution rate, GC3, codon usage KL and
codon-pair KL. It is offline/proxy evidence only and does not run an external
baseline or wet-lab assay.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from itertools import product
from typing import Mapping, Optional, Sequence

from mrna_editflow.core.constants import CODON_TABLE, START_CODON, STOP_CODONS, is_valid_cds, translate
from mrna_editflow.data.prepare_codon import gc3_fraction


CLAIM_POLICY = (
    "Protein-conditioned codon metrics are offline CDS reconstruction/design "
    "evidence. Native codon recovery and codon-distribution distances are useful "
    "for codonGPT/Prot2RNA-style comparisons, but they do not establish external "
    "SOTA reproduction, full-length de novo generation, wet-lab expression, or "
    "true structure-aware optimization."
)
RNA_ALPHABET = "ACGU"
ALL_CODONS = ["".join(parts) for parts in product(RNA_ALPHABET, repeat=3)]


def _normalise_rna(seq: object) -> str:
    return "".join(str(seq or "").upper().replace("T", "U").split())


def _codons(cds: object) -> list[str]:
    seq = _normalise_rna(cds)
    limit = len(seq) - len(seq) % 3
    return [seq[i : i + 3] for i in range(0, limit, 3)]


def _safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    if not values:
        return float(default)
    return float(sum(values) / len(values))


def _fraction(flags: Sequence[bool]) -> float:
    if not flags:
        return 0.0
    return float(sum(1 for flag in flags if flag) / len(flags))


def _identity(a: str, b: str) -> float:
    a = a.rstrip("*")
    b = b.rstrip("*")
    if not a and not b:
        return 1.0
    denom = max(len(a), len(b), 1)
    return float(sum(1 for x, y in zip(a, b) if x == y) / denom)


def _distribution(codon_lists: Sequence[Sequence[str]], *, pairs: bool = False, eps: float = 1e-8) -> dict[str, float]:
    keys = (
        [a + "|" + b for a in ALL_CODONS for b in ALL_CODONS]
        if pairs
        else list(ALL_CODONS)
    )
    counts = Counter({key: float(eps) for key in keys})
    for codons in codon_lists:
        usable = list(codons)
        if pairs:
            for a, b in zip(usable, usable[1:]):
                if a in CODON_TABLE and b in CODON_TABLE:
                    counts[a + "|" + b] += 1.0
        else:
            for codon in usable:
                if codon in CODON_TABLE:
                    counts[codon] += 1.0
    total = float(sum(counts.values()))
    return {key: float(counts[key] / total) for key in keys}


def _kl(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    total = 0.0
    for key, pv in p.items():
        if pv <= 0:
            continue
        qv = max(float(q.get(key, 0.0)), 1e-12)
        total += float(pv) * math.log(float(pv) / qv)
    return float(total)


def _top_deltas(
    p: Mapping[str, float],
    q: Mapping[str, float],
    *,
    top_n: int,
) -> list[dict[str, object]]:
    rows = [
        {
            "token": key,
            "designed_frequency": float(p.get(key, 0.0)),
            "native_frequency": float(q.get(key, 0.0)),
            "delta": float(p.get(key, 0.0) - q.get(key, 0.0)),
            "abs_delta": abs(float(p.get(key, 0.0) - q.get(key, 0.0))),
        }
        for key in sorted(set(p) | set(q))
    ]
    rows.sort(key=lambda row: (-float(row["abs_delta"]), str(row["token"])))
    return rows[: max(0, int(top_n))]


def _row_metrics(row: Mapping[str, object]) -> dict[str, object]:
    designed_cds = _normalise_rna(row.get("designed_cds"))
    native_cds = _normalise_rna(row.get("native_cds"))
    seed_cds = _normalise_rna(row.get("seed_cds"))
    target_protein = str(row.get("protein") or "").rstrip("*")
    designed_codons = _codons(designed_cds)
    native_codons = _codons(native_cds)
    seed_codons = _codons(seed_cds)

    designed_protein = translate(designed_cds).rstrip("*") if designed_cds else ""
    native_protein = translate(native_cds).rstrip("*") if native_cds else ""
    target_identity = _identity(target_protein, designed_protein)
    native_identity = _identity(native_protein, designed_protein) if native_cds else None

    n_common_native = min(len(designed_codons), len(native_codons))
    n_common_seed = min(len(designed_codons), len(seed_codons))
    native_matches = sum(
        1 for a, b in zip(designed_codons[:n_common_native], native_codons[:n_common_native]) if a == b
    )
    seed_matches = sum(
        1 for a, b in zip(designed_codons[:n_common_seed], seed_codons[:n_common_seed]) if a == b
    )
    native_synonymous = 0
    native_nonsynonymous = 0
    for designed, native in zip(designed_codons[:n_common_native], native_codons[:n_common_native]):
        if designed == native:
            continue
        if CODON_TABLE.get(designed) == CODON_TABLE.get(native):
            native_synonymous += 1
        else:
            native_nonsynonymous += 1
    designed_start_ok = bool(designed_codons and designed_codons[0] == START_CODON)
    designed_stop_ok = bool(designed_codons and designed_codons[-1] in STOP_CODONS)
    native_stop_ok = bool(native_codons and native_codons[-1] in STOP_CODONS)
    return {
        "target_id": row.get("target_id") or row.get("transcript_id") or row.get("protein") or None,
        "n_designed_codons": len(designed_codons),
        "n_native_codons": len(native_codons),
        "n_seed_codons": len(seed_codons),
        "protein_identity": float(target_identity),
        "native_protein_identity": None if native_identity is None else float(native_identity),
        "designed_valid_cds": bool(is_valid_cds(designed_cds)),
        "designed_start_ok": designed_start_ok,
        "designed_terminal_stop_ok": designed_stop_ok,
        "native_terminal_stop_ok": native_stop_ok,
        "native_codon_recovery": float(native_matches / n_common_native) if n_common_native else None,
        "seed_codon_recovery": float(seed_matches / n_common_seed) if n_common_seed else None,
        "native_synonymous_substitution_fraction": float(native_synonymous / n_common_native)
        if n_common_native
        else None,
        "native_nonsynonymous_substitution_fraction": float(native_nonsynonymous / n_common_native)
        if n_common_native
        else None,
        "native_codon_edit_fraction": float((n_common_native - native_matches) / n_common_native)
        if n_common_native
        else None,
        "seed_codon_edit_fraction": float((n_common_seed - seed_matches) / n_common_seed)
        if n_common_seed
        else None,
        "designed_gc3": float(gc3_fraction(designed_cds)),
        "native_gc3": float(gc3_fraction(native_cds)) if native_cds else None,
        "seed_gc3": float(gc3_fraction(seed_cds)) if seed_cds else None,
    }


def _read_jsonl(path: str) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path} contains a non-object JSONL row")
            rows.append(payload)
    return rows


def build_protein_conditioned_codon_metrics(
    *,
    jsonl_path: str,
    project_root: str = ".",
    top_n: int = 20,
) -> dict[str, object]:
    """Build codon-level protein-conditioned CDS metrics from JSONL rows."""
    project_root = os.path.abspath(project_root)
    rows = _read_jsonl(jsonl_path)
    row_metrics = [_row_metrics(row) for row in rows]
    designed_lists = [_codons(row.get("designed_cds")) for row in rows]
    native_lists = [_codons(row.get("native_cds")) for row in rows if row.get("native_cds")]
    seed_lists = [_codons(row.get("seed_cds")) for row in rows if row.get("seed_cds")]

    designed_dist = _distribution(designed_lists)
    native_dist = _distribution(native_lists)
    seed_dist = _distribution(seed_lists)
    designed_pair_dist = _distribution(designed_lists, pairs=True)
    native_pair_dist = _distribution(native_lists, pairs=True)

    def nums(key: str) -> list[float]:
        values = [row.get(key) for row in row_metrics]
        return [float(value) for value in values if isinstance(value, (int, float))]

    n = len(row_metrics)
    summary = {
        "n_rows": n,
        "n_with_native_cds": len(native_lists),
        "ready_for_codon_level_claim_audit": bool(
            n > 0
            and len(native_lists) == n
            and _fraction([bool(row["designed_valid_cds"]) for row in row_metrics]) == 1.0
            and _fraction([float(row["protein_identity"]) >= 1.0 for row in row_metrics]) == 1.0
            and _fraction([bool(row["designed_start_ok"]) for row in row_metrics]) == 1.0
            and _fraction([bool(row["designed_terminal_stop_ok"]) for row in row_metrics]) == 1.0
        ),
        "protein_identity_eq_1_fraction": _fraction(
            [float(row["protein_identity"]) >= 1.0 for row in row_metrics]
        ),
        "native_protein_identity_eq_1_fraction": _fraction(
            [
                isinstance(row.get("native_protein_identity"), (int, float))
                and float(row["native_protein_identity"]) >= 1.0
                for row in row_metrics
            ]
        ),
        "designed_valid_cds_fraction": _fraction([bool(row["designed_valid_cds"]) for row in row_metrics]),
        "designed_start_ok_fraction": _fraction([bool(row["designed_start_ok"]) for row in row_metrics]),
        "designed_terminal_stop_ok_fraction": _fraction(
            [bool(row["designed_terminal_stop_ok"]) for row in row_metrics]
        ),
        "mean_native_codon_recovery": _safe_mean(nums("native_codon_recovery")),
        "mean_seed_codon_recovery": _safe_mean(nums("seed_codon_recovery")),
        "mean_native_codon_edit_fraction": _safe_mean(nums("native_codon_edit_fraction")),
        "mean_seed_codon_edit_fraction": _safe_mean(nums("seed_codon_edit_fraction")),
        "mean_native_synonymous_substitution_fraction": _safe_mean(
            nums("native_synonymous_substitution_fraction")
        ),
        "mean_native_nonsynonymous_substitution_fraction": _safe_mean(
            nums("native_nonsynonymous_substitution_fraction")
        ),
        "mean_designed_gc3": _safe_mean(nums("designed_gc3")),
        "mean_native_gc3": _safe_mean(nums("native_gc3")),
        "mean_seed_gc3": _safe_mean(nums("seed_gc3")),
        "designed_vs_native_codon_usage_kl": _kl(designed_dist, native_dist),
        "designed_vs_seed_codon_usage_kl": _kl(designed_dist, seed_dist),
        "designed_vs_native_codon_pair_kl": _kl(designed_pair_dist, native_pair_dist),
    }
    payload = {
        "artifact_kind": "protein_conditioned_codon_metrics_audit",
        "claim_policy": CLAIM_POLICY,
        "jsonl_path": os.path.relpath(jsonl_path, project_root)
        if os.path.isabs(jsonl_path)
        else jsonl_path,
        "project_root": project_root,
        "summary": summary,
        "row_metrics": row_metrics,
        "top_codon_frequency_deltas": _top_deltas(designed_dist, native_dist, top_n=top_n),
        "top_codon_pair_frequency_deltas": _top_deltas(
            designed_pair_dist, native_pair_dist, top_n=top_n
        ),
        "limitations": [
            "Native codon recovery can decrease when CAI/GC optimization intentionally chooses synonymous alternatives.",
            "This audit uses proxy distribution distances and does not include external codonGPT/Prot2RNA executable outputs.",
            "Protein identity exact-1 remains the hard gate; positive quality claims still require matched baselines.",
        ],
    }
    return payload


def write_report_json(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return path


def _fmt(value: object, digits: int = 5) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_report_markdown(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    summary = report.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    lines = [
        "# Protein-Conditioned Codon Metrics Audit",
        "",
        f"- Claim policy: {report.get('claim_policy', CLAIM_POLICY)}",
        f"- Ready for codon-level claim audit: `{summary.get('ready_for_codon_level_claim_audit')}`",
        f"- Rows: `{summary.get('n_rows')}`; rows with native CDS: `{summary.get('n_with_native_cds')}`",
        f"- Protein identity exact-1 fraction: `{_fmt(summary.get('protein_identity_eq_1_fraction'))}`",
        f"- Mean native codon recovery: `{_fmt(summary.get('mean_native_codon_recovery'))}`",
        f"- Mean native synonymous substitution fraction: `{_fmt(summary.get('mean_native_synonymous_substitution_fraction'))}`",
        f"- Mean native nonsynonymous substitution fraction: `{_fmt(summary.get('mean_native_nonsynonymous_substitution_fraction'))}`",
        f"- Designed vs native codon usage KL: `{_fmt(summary.get('designed_vs_native_codon_usage_kl'))}`",
        f"- Designed vs native codon-pair KL: `{_fmt(summary.get('designed_vs_native_codon_pair_kl'))}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| designed valid CDS fraction | {_fmt(summary.get('designed_valid_cds_fraction'))} |",
        f"| designed start codon fraction | {_fmt(summary.get('designed_start_ok_fraction'))} |",
        f"| designed terminal stop fraction | {_fmt(summary.get('designed_terminal_stop_ok_fraction'))} |",
        f"| mean designed GC3 | {_fmt(summary.get('mean_designed_gc3'))} |",
        f"| mean native GC3 | {_fmt(summary.get('mean_native_gc3'))} |",
        f"| mean seed GC3 | {_fmt(summary.get('mean_seed_gc3'))} |",
        "",
        "## Largest Codon Frequency Deltas",
        "",
        "| codon | designed | native | delta |",
        "|---|---:|---:|---:|",
    ]
    for row in report.get("top_codon_frequency_deltas", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            f"| `{row.get('token')}` | {_fmt(row.get('designed_frequency'))} | "
            f"{_fmt(row.get('native_frequency'))} | {_fmt(row.get('delta'))} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
        ]
    )
    for item in report.get("limitations", []):
        lines.append(f"- {item}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True, help="protein_conditioned_cds*.jsonl artifact")
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--out-json", default="benchmark/protein_conditioned_codon_metrics_head256.json")
    parser.add_argument("--out-md", default="benchmark/protein_conditioned_codon_metrics_head256.md")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    project_root = os.path.abspath(args.project_root)
    jsonl_path = args.jsonl if os.path.isabs(args.jsonl) else os.path.join(project_root, args.jsonl)
    report = build_protein_conditioned_codon_metrics(
        jsonl_path=jsonl_path,
        project_root=project_root,
        top_n=args.top_n,
    )
    out_json = args.out_json if os.path.isabs(args.out_json) else os.path.join(project_root, args.out_json)
    out_md = args.out_md if os.path.isabs(args.out_md) else os.path.join(project_root, args.out_md)
    write_report_json(report, out_json)
    write_report_markdown(report, out_md)
    print(json.dumps({"json_path": out_json, "markdown_path": out_md, "summary": report["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLAIM_POLICY",
    "build_protein_conditioned_codon_metrics",
    "write_report_json",
    "write_report_markdown",
    "main",
]
