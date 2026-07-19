"""Summarize T4 protein-identity and CDS CAI/GC evidence.

This report consolidates the CDS-only codon-lattice DP baseline, the
protein-conditioned CDS design artifact, and the protein-conditioned CAI/GC
sweep audit. It is a proxy/offline T4 report: it verifies hard protein identity
and CAI/GC tradeoff metadata, but it does not claim external LinearDesign,
EnsembleDesign, codonGPT, or wet-lab/MFE reproduction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from typing import Mapping, Optional, Sequence


CLAIM_POLICY = (
    "T4 evidence is CDS/protein-identity and CAI/GC proxy evidence. Protein "
    "identity is a hard exact-1 constraint. Do not treat this as true external "
    "LinearDesign/codonGPT reproduction, wet-lab validation, or full MFE "
    "structure optimization."
)
DEFAULT_SLICE = "head256"


def default_paths(slice_name: str = DEFAULT_SLICE) -> dict[str, str]:
    """Return canonical T4 evidence paths for one evaluation slice."""
    slice_name = str(slice_name or DEFAULT_SLICE)
    return {
        "t5_primary_summary": (
            f"benchmark/multiseed_t5_public_{slice_name}_mo_grpo_top64/"
            "multiseed_summary.json"
        ),
        "codon_lattice_dp": f"benchmark/codon_lattice_dp_{slice_name}.json",
        "protein_conditioned": f"benchmark/protein_conditioned_cds_{slice_name}.summary.json",
        "protein_conditioned_codon_metrics": (
            f"benchmark/protein_conditioned_codon_metrics_{slice_name}.json"
        ),
        "gc_sweep_summary": (
            f"benchmark/protein_conditioned_cds_gc_sweep_{slice_name}.summary.json"
        ),
        "gc_sweep_audit": f"benchmark/protein_conditioned_cds_gc_sweep_{slice_name}.audit.json",
    }


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


def _path(project_root: str, value: str) -> str:
    return value if os.path.isabs(value) else os.path.join(project_root, value)


def _num(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _summary_metric(payload: Mapping[str, object], key: str) -> Optional[float]:
    summary = payload.get("summary", {})
    if not isinstance(summary, Mapping):
        return None
    return _num(summary.get(key))


def _aggregate_metric(payload: Mapping[str, object], key: str) -> Optional[float]:
    aggregate = payload.get("aggregate", {})
    if not isinstance(aggregate, Mapping):
        return None
    entry = aggregate.get(key, {})
    if not isinstance(entry, Mapping):
        return None
    return _num(entry.get("mean"))


def _load_source(project_root: str, rel_path: str) -> tuple[Optional[Mapping[str, object]], dict[str, object]]:
    path = _path(project_root, rel_path)
    audit: dict[str, object] = {
        "path": _rel(path, project_root),
        "exists": os.path.exists(path),
    }
    if not os.path.exists(path):
        return None, audit
    audit["sha256"] = _sha256_file(path)
    payload = _load_json(path)
    return payload, audit


def _finite_all(values: Sequence[object]) -> bool:
    return all(_num(value) is not None for value in values)


def _point_summary(point: Mapping[str, object]) -> dict[str, object]:
    summary = point.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    return {
        "gc_weight": _num(point.get("gc_weight")),
        "is_pareto_front": bool(point.get("is_pareto_front")),
        "pareto_rank": int(point.get("pareto_rank", -1)) if isinstance(point.get("pareto_rank"), (int, float)) else None,
        "mean_designed_cai": _num(summary.get("mean_designed_cai")),
        "mean_designed_gc": _num(summary.get("mean_designed_gc")),
        "mean_abs_gc_error": _num(summary.get("mean_abs_gc_error")),
        "mean_designed_vs_native_cai_delta": _num(summary.get("mean_designed_vs_native_cai_delta")),
        "mean_designed_vs_native_gc_delta": _num(summary.get("mean_designed_vs_native_gc_delta")),
        "mean_codon_changes": _num(summary.get("mean_codon_changes")),
        "protein_identity_eq_1_fraction": _num(summary.get("protein_identity_eq_1_fraction")),
        "designed_cai_ge_native_fraction": _num(summary.get("designed_cai_ge_native_fraction")),
    }


def _best_point(points: Sequence[Mapping[str, object]], metric: str, *, maximize: bool) -> Optional[dict[str, object]]:
    usable = []
    for point in points:
        value = _num(point.get(metric))
        if value is not None:
            usable.append((value, point))
    if not usable:
        return None
    _, point = max(usable, key=lambda item: item[0]) if maximize else min(usable, key=lambda item: item[0])
    return dict(point)


def build_t4_report(project_root: str, slice_name: str = DEFAULT_SLICE) -> dict[str, object]:
    """Build the T4 report from existing benchmark artifacts."""
    sources: dict[str, dict[str, object]] = {}
    payloads: dict[str, Optional[Mapping[str, object]]] = {}
    paths = default_paths(slice_name)
    for name, rel_path in paths.items():
        payload, audit = _load_source(project_root, rel_path)
        payloads[name] = payload
        sources[name] = audit

    t5 = payloads["t5_primary_summary"] or {}
    dp = payloads["codon_lattice_dp"] or {}
    protein = payloads["protein_conditioned"] or {}
    codon_metrics = payloads["protein_conditioned_codon_metrics"] or {}
    sweep = payloads["gc_sweep_summary"] or {}
    sweep_audit = payloads["gc_sweep_audit"] or {}

    t5_identity = {
        "mean_protein_identity": _aggregate_metric(t5, "mean_protein_identity"),
        "legal_fraction": _aggregate_metric(t5, "legal_fraction"),
        "within_budget_fraction": _aggregate_metric(t5, "within_budget_fraction"),
        "reading_frame_intact_fraction": _aggregate_metric(t5, "reading_frame_intact_fraction"),
    }
    dp_summary = {
        "n": _summary_metric(dp, "n"),
        "protein_identity_fraction": _summary_metric(dp, "protein_identity_fraction"),
        "mean_source_cai": _summary_metric(dp, "mean_source_cai"),
        "mean_optimized_cai": _summary_metric(dp, "mean_optimized_cai"),
        "mean_delta_cai": _summary_metric(dp, "mean_delta_cai"),
        "mean_source_gc": _summary_metric(dp, "mean_source_gc"),
        "mean_optimized_gc": _summary_metric(dp, "mean_optimized_gc"),
        "mean_delta_gc": _summary_metric(dp, "mean_delta_gc"),
        "mean_codon_changes": _summary_metric(dp, "mean_codon_changes"),
    }
    protein_summary_payload = protein.get("summary", {}) if isinstance(protein, Mapping) else {}
    if not isinstance(protein_summary_payload, Mapping):
        protein_summary_payload = {}
    protein_summary = {
        "n": _num(protein_summary_payload.get("n")),
        "mean_protein_identity": _num(protein_summary_payload.get("mean_protein_identity")),
        "protein_identity_eq_1_fraction": _num(protein_summary_payload.get("protein_identity_eq_1_fraction")),
        "native_protein_identity_eq_1_fraction": _num(protein_summary_payload.get("native_protein_identity_eq_1_fraction")),
        "mean_seed_cai": _num(protein_summary_payload.get("mean_seed_cai")),
        "mean_native_cai": _num(protein_summary_payload.get("mean_native_cai")),
        "mean_designed_cai": _num(protein_summary_payload.get("mean_designed_cai")),
        "mean_designed_vs_native_cai_delta": _num(protein_summary_payload.get("mean_designed_vs_native_cai_delta")),
        "mean_seed_gc": _num(protein_summary_payload.get("mean_seed_gc")),
        "mean_native_gc": _num(protein_summary_payload.get("mean_native_gc")),
        "mean_designed_gc": _num(protein_summary_payload.get("mean_designed_gc")),
        "mean_designed_vs_native_gc_delta": _num(protein_summary_payload.get("mean_designed_vs_native_gc_delta")),
        "mean_codon_changes": _num(protein_summary_payload.get("mean_codon_changes")),
        "designed_cai_ge_native_fraction": _num(protein_summary_payload.get("designed_cai_ge_native_fraction")),
    }
    codon_metrics_summary_payload = (
        codon_metrics.get("summary", {}) if isinstance(codon_metrics, Mapping) else {}
    )
    if not isinstance(codon_metrics_summary_payload, Mapping):
        codon_metrics_summary_payload = {}
    codon_metrics_summary = {
        "ready_for_codon_level_claim_audit": bool(
            codon_metrics_summary_payload.get("ready_for_codon_level_claim_audit")
        ),
        "n_rows": _num(codon_metrics_summary_payload.get("n_rows")),
        "n_with_native_cds": _num(codon_metrics_summary_payload.get("n_with_native_cds")),
        "protein_identity_eq_1_fraction": _num(
            codon_metrics_summary_payload.get("protein_identity_eq_1_fraction")
        ),
        "native_protein_identity_eq_1_fraction": _num(
            codon_metrics_summary_payload.get("native_protein_identity_eq_1_fraction")
        ),
        "designed_valid_cds_fraction": _num(
            codon_metrics_summary_payload.get("designed_valid_cds_fraction")
        ),
        "designed_start_ok_fraction": _num(
            codon_metrics_summary_payload.get("designed_start_ok_fraction")
        ),
        "designed_terminal_stop_ok_fraction": _num(
            codon_metrics_summary_payload.get("designed_terminal_stop_ok_fraction")
        ),
        "mean_native_codon_recovery": _num(
            codon_metrics_summary_payload.get("mean_native_codon_recovery")
        ),
        "mean_seed_codon_recovery": _num(
            codon_metrics_summary_payload.get("mean_seed_codon_recovery")
        ),
        "mean_native_codon_edit_fraction": _num(
            codon_metrics_summary_payload.get("mean_native_codon_edit_fraction")
        ),
        "mean_native_synonymous_substitution_fraction": _num(
            codon_metrics_summary_payload.get("mean_native_synonymous_substitution_fraction")
        ),
        "mean_native_nonsynonymous_substitution_fraction": _num(
            codon_metrics_summary_payload.get("mean_native_nonsynonymous_substitution_fraction")
        ),
        "mean_designed_gc3": _num(codon_metrics_summary_payload.get("mean_designed_gc3")),
        "mean_native_gc3": _num(codon_metrics_summary_payload.get("mean_native_gc3")),
        "designed_vs_native_codon_usage_kl": _num(
            codon_metrics_summary_payload.get("designed_vs_native_codon_usage_kl")
        ),
        "designed_vs_native_codon_pair_kl": _num(
            codon_metrics_summary_payload.get("designed_vs_native_codon_pair_kl")
        ),
    }

    raw_points = sweep.get("points", []) if isinstance(sweep, Mapping) else []
    points = [_point_summary(point) for point in raw_points if isinstance(point, Mapping)]
    best_cai = _best_point(points, "mean_designed_cai", maximize=True)
    best_gc = _best_point(points, "mean_abs_gc_error", maximize=False)
    audit_summary = sweep_audit.get("summary", {}) if isinstance(sweep_audit, Mapping) else {}
    jsonl_audit = sweep_audit.get("jsonl_audit", {}) if isinstance(sweep_audit, Mapping) else {}
    if not isinstance(audit_summary, Mapping):
        audit_summary = {}
    if not isinstance(jsonl_audit, Mapping):
        jsonl_audit = {}
    gc_sweep_summary = {
        "sweep_kind": sweep.get("sweep_kind") if isinstance(sweep, Mapping) else None,
        "n_targets": _num(sweep.get("n_targets")) if isinstance(sweep, Mapping) else None,
        "n_points": len(points),
        "n_pareto_front": int(_num(audit_summary.get("n_pareto_front")) or 0),
        "ready_for_pareto_claim_audit": bool(audit_summary.get("ready_for_pareto_claim_audit")),
        "all_points_identity_exact_1": bool(audit_summary.get("all_points_identity_exact_1")),
        "pareto_metadata_ok": bool(audit_summary.get("pareto_metadata_ok")),
        "jsonl_rows": int(_num(jsonl_audit.get("n_rows")) or 0),
        "jsonl_all_row_identity_exact_1": bool(jsonl_audit.get("all_row_identity_exact_1")),
        "best_cai_point": best_cai,
        "best_gc_point": best_gc,
        "points": points,
    }

    hard_constraints_exact_1 = all(
        value == 1.0
        for value in (
            t5_identity["mean_protein_identity"],
            dp_summary["protein_identity_fraction"],
            protein_summary["mean_protein_identity"],
            protein_summary["protein_identity_eq_1_fraction"],
            protein_summary["native_protein_identity_eq_1_fraction"],
        )
    ) and bool(gc_sweep_summary["all_points_identity_exact_1"]) and bool(
        gc_sweep_summary["jsonl_all_row_identity_exact_1"]
    )
    proxy_metrics_finite = _finite_all(
        [
            dp_summary["mean_delta_cai"],
            dp_summary["mean_delta_gc"],
            protein_summary["mean_designed_vs_native_cai_delta"],
            protein_summary["mean_designed_vs_native_gc_delta"],
            codon_metrics_summary["mean_native_codon_recovery"],
            codon_metrics_summary["mean_native_nonsynonymous_substitution_fraction"],
            codon_metrics_summary["designed_vs_native_codon_usage_kl"],
            codon_metrics_summary["designed_vs_native_codon_pair_kl"],
        ]
    )
    codon_level_metrics_ready = (
        bool(codon_metrics_summary["ready_for_codon_level_claim_audit"])
        and codon_metrics_summary["protein_identity_eq_1_fraction"] == 1.0
        and codon_metrics_summary["native_protein_identity_eq_1_fraction"] == 1.0
        and codon_metrics_summary["designed_valid_cds_fraction"] == 1.0
        and codon_metrics_summary["designed_start_ok_fraction"] == 1.0
        and codon_metrics_summary["designed_terminal_stop_ok_fraction"] == 1.0
        and codon_metrics_summary["mean_native_nonsynonymous_substitution_fraction"] == 0.0
    )
    ready = (
        all(audit.get("exists") for audit in sources.values())
        and hard_constraints_exact_1
        and codon_level_metrics_ready
        and proxy_metrics_finite
        and bool(gc_sweep_summary["ready_for_pareto_claim_audit"])
    )
    return {
        "artifact_kind": "t4_protein_identity_cai_gc_report",
        "project_root": os.path.abspath(project_root),
        "slice": slice_name,
        "claim_policy": CLAIM_POLICY,
        "sources": sources,
        "summary": {
            "ready": ready,
            "hard_constraints_exact_1": hard_constraints_exact_1,
            "codon_level_metrics_ready": codon_level_metrics_ready,
            "proxy_metrics_finite": proxy_metrics_finite,
            "external_baselines_configured": False,
            "true_mfe_structure_metric_available": False,
        },
        "t5_primary_identity": t5_identity,
        "codon_lattice_dp": dp_summary,
        "protein_conditioned_cds": protein_summary,
        "protein_conditioned_codon_metrics": codon_metrics_summary,
        "gc_sweep": gc_sweep_summary,
        "interpretation": {
            "best_cai_tradeoff": best_cai,
            "best_gc_tradeoff": best_gc,
            "boundary": (
                "This is constructive DP/proxy evidence. It supports exact protein "
                "identity and CAI/GC tradeoff claims, but not external LinearDesign, "
                "EnsembleDesign, codonGPT, wet-lab, or true MFE claims."
            ),
        },
    }


def write_report_json(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return path


def _fmt(value: object, digits: int = 5) -> str:
    number = _num(value)
    if number is None:
        return "NA"
    if abs(number) >= 1:
        return f"{number:.4f}"
    return f"{number:.{digits}f}"


def write_report_markdown(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    summary = report.get("summary", {})
    dp = report.get("codon_lattice_dp", {})
    pc = report.get("protein_conditioned_cds", {})
    codon_metrics = report.get("protein_conditioned_codon_metrics", {})
    sweep = report.get("gc_sweep", {})
    interpretation = report.get("interpretation", {})
    if not isinstance(summary, Mapping):
        summary = {}
    if not isinstance(dp, Mapping):
        dp = {}
    if not isinstance(pc, Mapping):
        pc = {}
    if not isinstance(codon_metrics, Mapping):
        codon_metrics = {}
    if not isinstance(sweep, Mapping):
        sweep = {}
    if not isinstance(interpretation, Mapping):
        interpretation = {}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# T4 Protein Identity / CAI-GC Audit\n\n")
        fh.write(f"- Claim policy: {report.get('claim_policy', CLAIM_POLICY)}\n")
        fh.write(f"- Ready: `{summary.get('ready')}`\n")
        fh.write(f"- Hard constraints exact-1: `{summary.get('hard_constraints_exact_1')}`\n")
        fh.write(f"- Codon-level metrics ready: `{summary.get('codon_level_metrics_ready')}`\n")
        fh.write(f"- True MFE/structure metric available: `{summary.get('true_mfe_structure_metric_available')}`\n")
        fh.write(f"- External baselines configured: `{summary.get('external_baselines_configured')}`\n\n")
        fh.write("| Evidence stream | n | protein identity | CAI delta | GC delta | codon changes | note |\n")
        fh.write("|---|---:|---:|---:|---:|---:|---|\n")
        fh.write(
            f"| CDS local codon-lattice DP | {_fmt(dp.get('n'), 0)} | "
            f"{_fmt(dp.get('protein_identity_fraction'))} | "
            f"{_fmt(dp.get('mean_delta_cai'))} | {_fmt(dp.get('mean_delta_gc'))} | "
            f"{_fmt(dp.get('mean_codon_changes'))} | synonymous CDS edit baseline |\n"
        )
        fh.write(
            f"| Protein-conditioned CDS | {_fmt(pc.get('n'), 0)} | "
            f"{_fmt(pc.get('protein_identity_eq_1_fraction'))} | "
            f"{_fmt(pc.get('mean_designed_vs_native_cai_delta'))} | "
            f"{_fmt(pc.get('mean_designed_vs_native_gc_delta'))} | "
            f"{_fmt(pc.get('mean_codon_changes'))} | protein-to-CDS constructive design |\n"
        )
        fh.write("\n## Codon-Level Protein-Conditioned Metrics\n\n")
        fh.write(
            f"- Rows: `{_fmt(codon_metrics.get('n_rows'), 0)}`; rows with native CDS: "
            f"`{_fmt(codon_metrics.get('n_with_native_cds'), 0)}`\n"
        )
        fh.write(
            f"- Native codon recovery: `{_fmt(codon_metrics.get('mean_native_codon_recovery'))}`; "
            f"native synonymous substitution fraction: "
            f"`{_fmt(codon_metrics.get('mean_native_synonymous_substitution_fraction'))}`; "
            f"native nonsynonymous substitution fraction: "
            f"`{_fmt(codon_metrics.get('mean_native_nonsynonymous_substitution_fraction'))}`\n"
        )
        fh.write(
            f"- Designed/native codon usage KL: "
            f"`{_fmt(codon_metrics.get('designed_vs_native_codon_usage_kl'))}`; "
            f"codon-pair KL: "
            f"`{_fmt(codon_metrics.get('designed_vs_native_codon_pair_kl'))}`\n"
        )
        fh.write(
            f"- Designed GC3: `{_fmt(codon_metrics.get('mean_designed_gc3'))}`; "
            f"native GC3: `{_fmt(codon_metrics.get('mean_native_gc3'))}`\n"
        )
        best_cai = sweep.get("best_cai_point")
        best_gc = sweep.get("best_gc_point")
        if not isinstance(best_cai, Mapping):
            best_cai = {}
        if not isinstance(best_gc, Mapping):
            best_gc = {}
        fh.write("\n## GC Sweep Pareto Audit\n\n")
        fh.write(
            f"- Points: `{sweep.get('n_points')}`; Pareto front: `{sweep.get('n_pareto_front')}`; "
            f"JSONL rows: `{sweep.get('jsonl_rows')}`\n"
        )
        fh.write(
            f"- Best CAI point: gc_weight=`{_fmt(best_cai.get('gc_weight'), 1)}`, "
            f"mean CAI=`{_fmt(best_cai.get('mean_designed_cai'))}`, "
            f"mean abs GC error=`{_fmt(best_cai.get('mean_abs_gc_error'))}`\n"
        )
        fh.write(
            f"- Best GC-target point: gc_weight=`{_fmt(best_gc.get('gc_weight'), 1)}`, "
            f"mean CAI=`{_fmt(best_gc.get('mean_designed_cai'))}`, "
            f"mean abs GC error=`{_fmt(best_gc.get('mean_abs_gc_error'))}`\n"
        )
        fh.write("\n## Boundary\n\n")
        fh.write(f"- {interpretation.get('boundary')}\n")
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--slice", default=DEFAULT_SLICE)
    parser.add_argument("--out-json", default="benchmark/t4_protein_identity_cai_gc_report_head256.json")
    parser.add_argument("--out-md", default="benchmark/t4_protein_identity_cai_gc_report_head256.md")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    project_root = os.path.abspath(args.project_root)
    report = build_t4_report(project_root, slice_name=args.slice)
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
    "DEFAULT_SLICE",
    "build_t4_report",
    "default_paths",
    "write_report_json",
    "write_report_markdown",
    "main",
]
