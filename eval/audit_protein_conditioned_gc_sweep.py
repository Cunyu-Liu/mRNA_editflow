"""Audit protein-conditioned CDS CAI-GC sweep artifacts.

The sweep is allowed to reveal a CAI-GC tradeoff, including weak or negative
biological tradeoffs. This audit only checks whether the artifact is complete
and safe to interpret: protein identity must remain exactly 1.0, numeric
objectives must be finite, and the Pareto frontier metadata must be coherent.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Mapping, Optional, Sequence


EXPECTED_SWEEP_KIND = "protein_conditioned_cai_gc_pareto"
IDENTITY_TOL = 1e-12
SUMMARY_METRICS: tuple[str, ...] = (
    "mean_designed_cai",
    "mean_designed_gc",
    "mean_abs_gc_error",
    "mean_codon_changes",
    "protein_identity_eq_1_fraction",
)


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


def _rel(path: Optional[str], root: str) -> Optional[str]:
    if not path:
        return None
    return os.path.relpath(path, root) if os.path.isabs(path) else path


def _point_audit(point: Mapping[str, object], identity_tolerance: float) -> dict[str, object]:
    summary = point.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    metric_values = {metric: _finite_float(summary.get(metric)) for metric in SUMMARY_METRICS}
    rank = point.get("pareto_rank")
    rank_ok = isinstance(rank, int) and rank >= 0
    identity = metric_values["protein_identity_eq_1_fraction"]
    return {
        "gc_weight": _finite_float(point.get("gc_weight")),
        "target_gc": _finite_float(point.get("target_gc")),
        "pareto_rank": rank if rank_ok else None,
        "is_pareto_front": bool(point.get("is_pareto_front")),
        "summary_metrics": metric_values,
        "summary_metrics_finite": all(value is not None for value in metric_values.values()),
        "protein_identity_exact_1": identity is not None
        and abs(identity - 1.0) <= identity_tolerance,
        "pareto_rank_ok": rank_ok,
    }


def _read_jsonl_audit(jsonl_path: Optional[str], identity_tolerance: float) -> dict[str, object]:
    if not jsonl_path:
        return {"path": None, "exists": False, "n_rows": 0, "all_row_identity_exact_1": False}
    if not os.path.exists(jsonl_path):
        return {
            "path": jsonl_path,
            "exists": False,
            "n_rows": 0,
            "all_row_identity_exact_1": False,
        }
    n_rows = 0
    all_identity = True
    malformed_rows = 0
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            n_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed_rows += 1
                all_identity = False
                continue
            if not isinstance(row, Mapping):
                malformed_rows += 1
                all_identity = False
                continue
            identity = _finite_float(row.get("protein_identity"))
            if identity is None or abs(identity - 1.0) > identity_tolerance:
                all_identity = False
    return {
        "path": jsonl_path,
        "exists": True,
        "n_rows": n_rows,
        "malformed_rows": malformed_rows,
        "all_row_identity_exact_1": all_identity and malformed_rows == 0 and n_rows > 0,
    }


def audit_protein_conditioned_gc_sweep(
    *,
    summary_json: str,
    jsonl_path: Optional[str] = None,
    md_path: Optional[str] = None,
    project_root: str = ".",
    identity_tolerance: float = IDENTITY_TOL,
    out_json: Optional[str] = None,
    out_md: Optional[str] = None,
) -> dict[str, object]:
    """Audit a protein-conditioned CAI-GC sweep summary and optional detail files."""
    summary_exists = os.path.exists(summary_json)
    payload: Optional[Mapping[str, object]] = _load_json(summary_json) if summary_exists else None
    if payload and jsonl_path is None:
        inferred_jsonl = payload.get("out_jsonl")
        if isinstance(inferred_jsonl, str):
            jsonl_path = inferred_jsonl
    if payload and md_path is None:
        inferred_md = payload.get("out_md")
        if isinstance(inferred_md, str):
            md_path = inferred_md

    points = payload.get("points", []) if payload else []
    if not isinstance(points, list):
        points = []
    pareto_front = payload.get("pareto_front", []) if payload else []
    if not isinstance(pareto_front, list):
        pareto_front = []
    point_audits = [
        _point_audit(point, identity_tolerance)
        for point in points
        if isinstance(point, Mapping)
    ]
    pareto_front_weights = payload.get("pareto_front_gc_weights", []) if payload else []
    if not isinstance(pareto_front_weights, list):
        pareto_front_weights = []
    front_weights_from_points = sorted(
        float(point["gc_weight"])
        for point in point_audits
        if point["is_pareto_front"] and point["gc_weight"] is not None
    )
    front_weights_from_payload = sorted(
        float(weight)
        for weight in pareto_front_weights
        if _finite_float(weight) is not None
    )
    jsonl_audit = _read_jsonl_audit(jsonl_path, identity_tolerance)
    md_exists = bool(md_path and os.path.exists(md_path))

    expected_rows = None
    n_targets = None
    if payload:
        n_targets = payload.get("n_targets")
        if isinstance(n_targets, int):
            expected_rows = n_targets * len(point_audits)
    jsonl_row_count_ok = (
        expected_rows is not None
        and bool(jsonl_audit.get("exists"))
        and int(jsonl_audit.get("n_rows", -1)) == expected_rows
    )
    artifact_contract = payload.get("artifact_contract", {}) if payload else {}
    if not isinstance(artifact_contract, Mapping):
        artifact_contract = {}
    hard_constraint_ok = (
        artifact_contract.get("hard_constraint")
        == "protein_identity_eq_1_fraction must remain 1.0"
    )
    sweep_kind_ok = bool(payload) and payload.get("sweep_kind") == EXPECTED_SWEEP_KIND
    all_points_identity_exact_1 = bool(point_audits) and all(
        bool(point["protein_identity_exact_1"]) for point in point_audits
    )
    all_point_metrics_finite = bool(point_audits) and all(
        bool(point["summary_metrics_finite"]) for point in point_audits
    )
    pareto_metadata_ok = (
        bool(point_audits)
        and bool(pareto_front)
        and all(bool(point["pareto_rank_ok"]) for point in point_audits)
        and all(
            (point["pareto_rank"] == 0) == bool(point["is_pareto_front"])
            for point in point_audits
        )
        and front_weights_from_points == front_weights_from_payload
    )
    ready = (
        summary_exists
        and sweep_kind_ok
        and hard_constraint_ok
        and all_points_identity_exact_1
        and all_point_metrics_finite
        and pareto_metadata_ok
        and bool(jsonl_audit.get("all_row_identity_exact_1"))
        and jsonl_row_count_ok
        and md_exists
    )
    missing_artifacts = []
    if not summary_exists:
        missing_artifacts.append(summary_json)
    if jsonl_path and not bool(jsonl_audit.get("exists")):
        missing_artifacts.append(jsonl_path)
    if md_path and not md_exists:
        missing_artifacts.append(md_path)
    payload_out = {
        "artifact_kind": "protein_conditioned_gc_sweep_audit",
        "summary_json": _rel(summary_json, project_root),
        "jsonl_path": _rel(jsonl_path, project_root),
        "md_path": _rel(md_path, project_root),
        "identity_tolerance": identity_tolerance,
        "summary": {
            "ready_for_pareto_claim_audit": ready,
            "summary_exists": summary_exists,
            "sweep_kind_ok": sweep_kind_ok,
            "hard_constraint_contract_ok": hard_constraint_ok,
            "n_targets": n_targets,
            "n_points": len(point_audits),
            "n_pareto_front": len(pareto_front),
            "all_points_identity_exact_1": all_points_identity_exact_1,
            "all_point_metrics_finite": all_point_metrics_finite,
            "pareto_metadata_ok": pareto_metadata_ok,
            "jsonl_row_count_ok": jsonl_row_count_ok,
            "jsonl_all_row_identity_exact_1": bool(
                jsonl_audit.get("all_row_identity_exact_1")
            ),
            "md_exists": md_exists,
            "missing_artifacts": [_rel(path, project_root) for path in missing_artifacts],
        },
        "jsonl_audit": {
            **jsonl_audit,
            "path": _rel(str(jsonl_audit.get("path")), project_root)
            if jsonl_audit.get("path")
            else None,
        },
        "point_audits": point_audits,
    }
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(payload_out, fh, indent=2, sort_keys=True)
    if out_md:
        write_markdown(payload_out, out_md)
    return payload_out


def _fmt(value: object) -> str:
    x = _finite_float(value)
    return "" if x is None else f"{x:.5f}"


def write_markdown(payload: Mapping[str, object], out_md: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
    summary = payload.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    lines = [
        "# Protein-conditioned GC Sweep Audit",
        "",
        f"- Ready for Pareto claim audit: {summary.get('ready_for_pareto_claim_audit')}",
        f"- Summary JSON: `{payload.get('summary_json')}`",
        f"- JSONL: `{payload.get('jsonl_path')}`",
        f"- Markdown: `{payload.get('md_path')}`",
        f"- Points: {summary.get('n_points')}",
        f"- Pareto front size: {summary.get('n_pareto_front')}",
        f"- All point identity exactly 1: {summary.get('all_points_identity_exact_1')}",
        f"- All point metrics finite: {summary.get('all_point_metrics_finite')}",
        f"- Pareto metadata OK: {summary.get('pareto_metadata_ok')}",
        "",
        "| gc_weight | rank | front | identity=1 | CAI | GC | abs GC error |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in payload.get("point_audits", []):
        if not isinstance(point, Mapping):
            continue
        metrics = point.get("summary_metrics", {})
        if not isinstance(metrics, Mapping):
            metrics = {}
        lines.append(
            "| {gc_weight} | {rank} | {front} | {identity} | {cai} | {gc} | {err} |".format(
                gc_weight=_fmt(point.get("gc_weight")),
                rank=point.get("pareto_rank"),
                front=point.get("is_pareto_front"),
                identity=point.get("protein_identity_exact_1"),
                cai=_fmt(metrics.get("mean_designed_cai")),
                gc=_fmt(metrics.get("mean_designed_gc")),
                err=_fmt(metrics.get("mean_abs_gc_error")),
            )
        )
    lines.extend(["", "## Missing Artifacts", "", "| path |", "|---|"])
    for path in summary.get("missing_artifacts", []):
        lines.append(f"| `{path}` |")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--jsonl", dest="jsonl_path", default=None)
    parser.add_argument("--md", dest="md_path", default=None)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--identity-tolerance", type=float, default=IDENTITY_TOL)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    payload = audit_protein_conditioned_gc_sweep(
        summary_json=args.summary_json,
        jsonl_path=args.jsonl_path,
        md_path=args.md_path,
        project_root=args.project_root,
        identity_tolerance=args.identity_tolerance,
        out_json=args.out_json,
        out_md=args.out_md,
    )
    print(json.dumps({"summary": payload["summary"], "out_json": args.out_json, "out_md": args.out_md}, sort_keys=True))
    if args.strict and not payload["summary"]["ready_for_pareto_claim_audit"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
