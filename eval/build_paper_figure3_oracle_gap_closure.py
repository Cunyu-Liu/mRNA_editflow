"""Build paper Figure 3: offline oracle-gap closure curve spec.

The curve uses the matched head64 proposal-ranking diagnostics. It measures how
much of the candidate-pool oracle gap each ranker closes relative to the source
sequence. This is an offline proxy diagnostic, not a wet-lab or de novo SOTA
claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from typing import Mapping, Optional, Sequence

from mrna_editflow.eval.artifact_contract import normalize_run_mode, paper_builder_gate, validate_report_output_namespaces, write_paper_report_sidecars


CLAIM_POLICY = (
    "Figure 3 may show offline candidate-pool oracle-gap closure for matched "
    "proposal-ranking diagnostics. The oracle is a proxy upper bound within the "
    "generated candidate pool; it is not wet-lab validation, an external SOTA "
    "result, or proof of unconstrained de novo design."
)

RANKER_RUNS: tuple[dict[str, str], ...] = (
    {
        "id": "stage_a_base",
        "label": "Stage A base",
        "path": "benchmark/proposal_ranking_t5_base_full1k_head64.json",
    },
    {
        "id": "previous_te_ranker",
        "label": "Previous TE-ranker",
        "path": "benchmark/proposal_ranking_t5_ranker_full1k_final_head64.json",
    },
    {
        "id": "utr_teacher",
        "label": "UTR-teacher ranker",
        "path": "benchmark/proposal_ranking_t5_utr_teacher_head64.json",
    },
    {
        "id": "direct_hybrid",
        "label": "Direct hybrid teacher",
        "path": "benchmark/proposal_ranking_t5_hybrid_teacher_head64.json",
    },
    {
        "id": "full_then_utr",
        "label": "Full-then-UTR sequential",
        "path": "benchmark/proposal_ranking_t5_full1k_then_utr_teacher_head64.json",
    },
    {
        "id": "source_aware_hybrid",
        "label": "Source-aware hybrid",
        "path": "benchmark/proposal_ranking_t5_sourceaware_hybrid_teacher_head64.json",
    },
    {
        "id": "cascade_hardneg_v2",
        "label": "Cascade hard-negative v2",
        "path": "benchmark/proposal_ranking_t5_cascade_hardneg_teacher_head64.json",
    },
)


def _path(project_root: str, rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(project_root, rel)


def _load_json(path: str) -> Mapping[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _sha256_if_exists(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_audit(project_root: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in RANKER_RUNS:
        rel = run["path"]
        path = _path(project_root, rel)
        rows.append(
            {
                "id": run["id"],
                "path": rel,
                "exists": os.path.exists(path),
                "sha256": _sha256_if_exists(path),
                "n_bytes": os.path.getsize(path) if os.path.exists(path) else None,
            }
        )
    return rows


def _num(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _fmt(value: object, digits: int = 5) -> str:
    number = _num(value)
    if number is None:
        return "NA"
    if abs(number) >= 1:
        return f"{number:.4f}"
    return f"{number:.{digits}f}"


def _fmt_delta(value: object, digits: int = 5) -> str:
    number = _num(value)
    if number is None:
        return "NA"
    return f"{number:+.{digits}f}"


def _aggregate(payload: Mapping[str, object]) -> Mapping[str, object]:
    aggregate = payload.get("aggregate", {})
    return aggregate if isinstance(aggregate, Mapping) else {}


def _point(run: Mapping[str, str], payload: Mapping[str, object], index: int) -> dict[str, object]:
    agg = _aggregate(payload)
    source = _num(agg.get("mean_source_te"))
    oracle = _num(agg.get("mean_oracle_top_te"))
    model = _num(agg.get("mean_model_top_te"))
    gap = (oracle - source) if source is not None and oracle is not None else None
    delta_vs_source = (model - source) if model is not None and source is not None else None
    residual_gap = (oracle - model) if oracle is not None and model is not None else None
    closure = (
        delta_vs_source / gap
        if delta_vs_source is not None and gap is not None and gap > 0
        else None
    )
    return {
        "order": index,
        "run_id": run["id"],
        "label": run["label"],
        "source": run["path"],
        "n_records": agg.get("n_records"),
        "n_candidates": agg.get("n_candidates"),
        "mean_source_te": source,
        "mean_oracle_top_te": oracle,
        "mean_model_top_te": model,
        "oracle_gap": gap,
        "delta_vs_source": delta_vs_source,
        "residual_oracle_gap": residual_gap,
        "closure_fraction": closure,
        "mean_model_regret": agg.get("mean_model_regret"),
        "oracle_best_in_model_top_k_fraction": agg.get("oracle_best_in_model_top_k_fraction"),
    }


def _consistent(values: Sequence[Optional[float]], tol: float = 1e-12) -> bool:
    finite = [value for value in values if value is not None]
    return bool(finite) and max(finite) - min(finite) <= tol


def _chart_spec(points: Sequence[Mapping[str, object]]) -> dict[str, object]:
    values = [
        {
            "order": point.get("order"),
            "label": point.get("label"),
            "closure_fraction": point.get("closure_fraction"),
            "mean_model_top_te": point.get("mean_model_top_te"),
            "oracle_best_in_model_top_k_fraction": point.get(
                "oracle_best_in_model_top_k_fraction"
            ),
        }
        for point in points
    ]
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Offline oracle-gap closure over matched proposal-ranking diagnostics.",
        "data": {"values": values},
        "mark": {"type": "line", "point": True},
        "encoding": {
            "x": {"field": "label", "type": "nominal", "sort": {"field": "order"}},
            "y": {
                "field": "closure_fraction",
                "type": "quantitative",
                "title": "Oracle-gap closure fraction",
            },
            "tooltip": [
                {"field": "label", "type": "nominal"},
                {"field": "closure_fraction", "type": "quantitative"},
                {"field": "mean_model_top_te", "type": "quantitative"},
                {"field": "oracle_best_in_model_top_k_fraction", "type": "quantitative"},
            ],
        },
    }


def _build_paper_figure3_development(project_root: str) -> dict[str, object]:
    points = [
        _point(run, _load_json(_path(project_root, run["path"])), index)
        for index, run in enumerate(RANKER_RUNS)
    ]
    source_values = [_num(point.get("mean_source_te")) for point in points]
    oracle_values = [_num(point.get("mean_oracle_top_te")) for point in points]
    closures = [_num(point.get("closure_fraction")) for point in points]
    finite_closures = [value for value in closures if value is not None]
    best = max(
        points,
        key=lambda point: _num(point.get("closure_fraction")) or float("-inf"),
        default={},
    )
    source_audit = _source_audit(project_root)
    best_closure = _num(best.get("closure_fraction")) if isinstance(best, Mapping) else None
    oracle_gap_fully_closed = best_closure is not None and best_closure >= 1.0
    negative_closure_present = any(value < 0 for value in finite_closures)
    caption = (
        "Figure 3. Offline oracle-gap closure on the matched head64 proposal pool. "
        "The source-to-oracle gap is defined as mean_oracle_top_TE - mean_source_TE; "
        "each ranker closes (mean_model_top_TE - mean_source_TE) / gap. "
        f"The best current point is {best.get('label')} with closure fraction "
        f"{_fmt(best_closure)}, so most of the proxy oracle gap remains open. "
        "Negative closure for the Stage A base is retained rather than hidden."
    )
    return {
        "artifact_kind": "paper_figure3_oracle_gap_closure",
        "project_root": os.path.abspath(project_root),
        "claim_policy": CLAIM_POLICY,
        "summary": {
            "n_points": len(points),
            "source_files_ready": all(row["exists"] for row in source_audit),
            "ready_for_oracle_gap_figure_draft": len(points) == len(RANKER_RUNS),
            "source_oracle_consistent": _consistent(source_values) and _consistent(oracle_values),
            "best_run_id": best.get("run_id") if isinstance(best, Mapping) else None,
            "best_label": best.get("label") if isinstance(best, Mapping) else None,
            "best_closure_fraction": best_closure,
            "negative_closure_present": negative_closure_present,
            "oracle_gap_fully_closed": oracle_gap_fully_closed,
            "ready_for_oracle_sota_claim": False,
            "ready_for_wet_lab_claim": False,
        },
        "caption": caption,
        "points": points,
        "chart_spec": _chart_spec(points),
        "source_audit": source_audit,
    }


def build_paper_figure3(
    project_root: str,
    run_mode: str = "development",
    artifact_paths: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    if normalize_run_mode(run_mode) == "paper":
        return paper_builder_gate("paper_figure3_oracle_gap_closure", project_root, artifact_paths, __file__)
    report = _build_paper_figure3_development(project_root)
    report.update({"claim_tier": "development_only", "paper_eligible": False})
    return report


def write_report_json(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return path


def write_report_markdown(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    points = report.get("points", [])
    summary = report.get("summary", {})
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        points = []
    if not isinstance(summary, Mapping):
        summary = {}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Paper Figure 3: Oracle-Gap Closure Curve\n\n")
        fh.write(f"- Claim policy: {report.get('claim_policy', CLAIM_POLICY)}\n")
        fh.write(
            f"- Ready for oracle-gap figure draft: `{summary.get('ready_for_oracle_gap_figure_draft')}`; "
            f"ready for oracle/SOTA claim: `{summary.get('ready_for_oracle_sota_claim')}`; "
            f"ready for wet-lab claim: `{summary.get('ready_for_wet_lab_claim')}`\n"
        )
        fh.write(
            f"- Best point: `{summary.get('best_label')}` closure "
            f"`{_fmt(summary.get('best_closure_fraction'))}`; "
            f"negative closure present: `{summary.get('negative_closure_present')}`; "
            f"oracle gap fully closed: `{summary.get('oracle_gap_fully_closed')}`\n\n"
        )
        fh.write("## Caption\n\n")
        fh.write(str(report.get("caption", "")))
        fh.write("\n\n## Curve Points\n\n")
        fh.write(
            "| Order | Ranker | Model top TE | Source TE | Oracle top TE | "
            "Delta vs source | Closure fraction | Residual gap | Recall fraction |\n"
        )
        fh.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for point in points:
            if not isinstance(point, Mapping):
                continue
            fh.write(
                f"| {point.get('order')} | {point.get('label')} | "
                f"{_fmt(point.get('mean_model_top_te'))} | "
                f"{_fmt(point.get('mean_source_te'))} | "
                f"{_fmt(point.get('mean_oracle_top_te'))} | "
                f"{_fmt_delta(point.get('delta_vs_source'))} | "
                f"{_fmt(point.get('closure_fraction'))} | "
                f"{_fmt(point.get('residual_oracle_gap'))} | "
                f"{_fmt(point.get('oracle_best_in_model_top_k_fraction'))} |\n"
            )
        fh.write("\n## Vega-Lite Spec\n\n")
        fh.write("```json\n")
        fh.write(json.dumps(report.get("chart_spec", {}), indent=2, sort_keys=True))
        fh.write("\n```\n")
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--out-json", default="docs/paper_figure3_oracle_gap_closure.json")
    parser.add_argument("--out-md", default="docs/paper_figure3_oracle_gap_closure.md")
    parser.add_argument("--run-mode", choices=("development", "paper"), default="development")
    parser.add_argument("--paper-artifact", action="append", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    project_root = os.path.abspath(args.project_root)
    report = build_paper_figure3(project_root, args.run_mode, args.paper_artifact)
    out_json = args.out_json if os.path.isabs(args.out_json) else os.path.join(project_root, args.out_json)
    out_md = args.out_md if os.path.isabs(args.out_md) else os.path.join(project_root, args.out_md)
    validate_report_output_namespaces(report, (out_json, out_md))
    write_report_json(report, out_json)
    write_report_markdown(report, out_md)
    write_paper_report_sidecars(report, (out_json, out_md))
    print(json.dumps({"json_path": out_json, "markdown_path": out_md}, sort_keys=True))
    if args.run_mode == "paper" and args.paper_artifact and not report["paper_eligible"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLAIM_POLICY",
    "build_paper_figure3",
    "write_report_json",
    "write_report_markdown",
    "main",
]
