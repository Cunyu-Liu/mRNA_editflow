"""Build paper Figure 2: cascade recall/precision division spec.

The figure explains why MEF separates recall and precision rankers, while
preserving the current evidence boundary: the old cascade has a non-significant
positive TE trend, and hard-negative v2 direct top64 remains the current default
model-only decoding setting.
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
    "Figure 2 may illustrate source-aware recall followed by precision reranking. "
    "It must not claim that cascade decoding significantly beats sequential or "
    "hard-negative v2 decoding unless the paired TE comparison is significant."
)

SOURCE_FILES: tuple[str, ...] = (
    "benchmark/proposal_ranking_t5_ranker_full1k_final_head64.json",
    "benchmark/proposal_ranking_t5_sourceaware_hybrid_teacher_head64.json",
    "benchmark/cascade_sourceaware_to_sequential_head64_k64.json",
    "benchmark/compare_t5_head256_cascade_vs_seq_top64.json",
    "benchmark/compare_t5_head256_hardneg_v2_top64.json",
    "benchmark/compare_t5_head256_hardneg_v2_vs_cascade_top64.json",
    "benchmark/compare_cascade_10krecall_vs_hardneg_v2.json",
    "benchmark/cascade_error_analysis_head256_top64.json",
)
PRIMARY_METRIC = "delta_oracle_te_vs_source"


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
    for rel in SOURCE_FILES:
        path = _path(project_root, rel)
        rows.append(
            {
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


def _comparison_row(
    payload: Mapping[str, object],
    *,
    metric: str = PRIMARY_METRIC,
    run_contains: Optional[str] = None,
) -> Mapping[str, object]:
    rows = payload.get("rows", [])
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        for row in rows:
            if not isinstance(row, Mapping) or row.get("metric") != metric:
                continue
            if run_contains is not None and run_contains not in str(row.get("run", "")):
                continue
            return row
    return {}


def _constraint_rows_exact_1(payload: Mapping[str, object], run_contains: Optional[str] = None) -> bool:
    rows = payload.get("rows", [])
    wanted = ("legal_fraction", "within_budget_fraction", "mean_protein_identity", "reading_frame_intact_fraction")
    values: list[float] = []
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        for metric in wanted:
            for row in rows:
                if not isinstance(row, Mapping) or row.get("metric") != metric:
                    continue
                if run_contains is not None and run_contains not in str(row.get("run", "")):
                    continue
                value = _num(row.get("run_mean"))
                if value is not None:
                    values.append(value)
                break
    return len(values) == len(wanted) and all(abs(value - 1.0) <= 1e-12 for value in values)


def _nodes(values: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        {
            "id": "A",
            "label": "Full legal candidate pool",
            "role": "input",
            "detail": (
                f"{int(values.get('n_candidates') or 0)} candidates over "
                f"{int(values.get('n_records') or 0)} diagnostic records"
            ),
        },
        {
            "id": "B",
            "label": "Source-aware recall ranker",
            "role": "recall",
            "detail": (
                f"oracle-best top-k recall={_fmt(values.get('sourceaware_recall'))}; "
                f"regret={_fmt(values.get('sourceaware_regret'))}"
            ),
        },
        {
            "id": "C",
            "label": "Recall top-k set",
            "role": "candidate_filter",
            "detail": (
                f"k=64 retains oracle-best fraction={_fmt(values.get('cascade_recall_k64'))}"
            ),
        },
        {
            "id": "D",
            "label": "Precision reranker",
            "role": "precision",
            "detail": (
                f"full-pool regret={_fmt(values.get('precision_full_regret'))}; "
                f"cascade regret={_fmt(values.get('cascade_regret'))}"
            ),
        },
        {
            "id": "E",
            "label": "Cascade output audit",
            "role": "trend_result",
            "detail": (
                f"vs sequential delta={_fmt_delta(values.get('cascade_vs_seq_delta'))}; "
                f"p={_fmt(values.get('cascade_vs_seq_p'))}"
            ),
        },
        {
            "id": "F",
            "label": "Hard-negative v2 direct top64",
            "role": "current_default",
            "detail": (
                f"vs sequential delta={_fmt_delta(values.get('hardneg_vs_seq_delta'))}; "
                f"p={_fmt(values.get('hardneg_vs_seq_p'))}"
            ),
        },
        {
            "id": "G",
            "label": "Cascade win/loss mining",
            "role": "error_analysis",
            "detail": (
                f"win_fraction={_fmt(values.get('win_fraction'))}; "
                f"mean_gain={_fmt_delta(values.get('mean_cascade_gain'))}"
            ),
        },
        {
            "id": "H",
            "label": "Next training target",
            "role": "future_work",
            "detail": "Improve recall quality or precision ranker; do not simply claim cascade superiority.",
        },
    ]


def _edges() -> list[dict[str, object]]:
    return [
        {"source": "A", "target": "B", "label": "score broad candidate pool"},
        {"source": "B", "target": "C", "label": "retain high-recall top-k"},
        {"source": "C", "target": "D", "label": "rerank for top-1 TE"},
        {"source": "D", "target": "E", "label": "audit cascade output"},
        {"source": "A", "target": "F", "label": "direct precision baseline"},
        {"source": "E", "target": "G", "label": "mine cascade wins/losses"},
        {"source": "F", "target": "G", "label": "compare against default"},
        {"source": "G", "target": "H", "label": "train next teacher"},
    ]


def _mermaid(nodes: Sequence[Mapping[str, object]], edges: Sequence[Mapping[str, object]]) -> str:
    node_map = {str(node.get("id")): node for node in nodes}
    lines = ["flowchart LR"]
    for node in nodes:
        node_id = str(node.get("id"))
        label = str(node.get("label"))
        detail = str(node.get("detail"))
        lines.append(f'    {node_id}["{label}<br/>{detail}"]')
    for edge in edges:
        src = str(edge.get("source"))
        dst = str(edge.get("target"))
        if src not in node_map or dst not in node_map:
            continue
        lines.append(f"    {src} -->|{edge.get('label')}| {dst}")
    lines.extend(
        [
            "    classDef recall fill:#e6f7ff,stroke:#1677ff,color:#262626",
            "    classDef precision fill:#f9f0ff,stroke:#722ed1,color:#262626",
            "    classDef caution fill:#fff1f0,stroke:#cf1322,color:#262626",
            "    class B,C recall",
            "    class D,F precision",
            "    class E,H caution",
        ]
    )
    return "\n".join(lines)


def _build_paper_figure2_development(project_root: str) -> dict[str, object]:
    prev_ranker = _load_json(_path(project_root, SOURCE_FILES[0]))
    sourceaware = _load_json(_path(project_root, SOURCE_FILES[1]))
    cascade = _load_json(_path(project_root, SOURCE_FILES[2]))
    cascade_vs_seq = _load_json(_path(project_root, SOURCE_FILES[3]))
    hardneg_vs_seq = _load_json(_path(project_root, SOURCE_FILES[4]))
    hardneg_vs_cascade = _load_json(_path(project_root, SOURCE_FILES[5]))
    cascade10k_vs_hardneg = _load_json(_path(project_root, SOURCE_FILES[6]))
    error = _load_json(_path(project_root, SOURCE_FILES[7]))

    prev_agg = _aggregate(prev_ranker)
    source_agg = _aggregate(sourceaware)
    cascade_agg = _aggregate(cascade)
    error_agg = _aggregate(error)
    cascade_cmp = _comparison_row(cascade_vs_seq, run_contains="cascade")
    hardneg_seq_cmp = _comparison_row(hardneg_vs_seq, run_contains="hardneg")
    hardneg_cascade_cmp = _comparison_row(hardneg_vs_cascade, run_contains="hardneg")
    cascade10k_cmp = _comparison_row(cascade10k_vs_hardneg, run_contains="cascade")

    values: dict[str, object] = {
        "n_candidates": cascade_agg.get("n_candidates"),
        "n_records": cascade_agg.get("n_records"),
        "previous_ranker_recall": prev_agg.get("oracle_best_in_model_top_k_fraction"),
        "sourceaware_recall": source_agg.get("oracle_best_in_model_top_k_fraction"),
        "sourceaware_regret": source_agg.get("mean_model_regret"),
        "cascade_recall_k64": cascade_agg.get("oracle_best_in_recall_top_k_fraction"),
        "precision_full_regret": cascade_agg.get("mean_precision_full_regret"),
        "cascade_regret": cascade_agg.get("mean_cascade_regret"),
        "cascade_vs_seq_delta": cascade_cmp.get("delta"),
        "cascade_vs_seq_p": cascade_cmp.get("paired_p"),
        "cascade_vs_seq_run_mean": cascade_cmp.get("run_mean"),
        "hardneg_vs_seq_delta": hardneg_seq_cmp.get("delta"),
        "hardneg_vs_seq_p": hardneg_seq_cmp.get("paired_p"),
        "hardneg_vs_cascade_delta": hardneg_cascade_cmp.get("delta"),
        "hardneg_vs_cascade_p": hardneg_cascade_cmp.get("paired_p"),
        "cascade10k_vs_hardneg_delta": cascade10k_cmp.get("delta"),
        "cascade10k_vs_hardneg_p": cascade10k_cmp.get("paired_p"),
        "win_fraction": error_agg.get("win_record_fraction"),
        "mean_cascade_gain": error_agg.get("mean_cascade_gain"),
    }
    nodes = _nodes(values)
    edges = _edges()
    cascade_p = _num(values.get("cascade_vs_seq_p"))
    hardneg_p = _num(values.get("hardneg_vs_seq_p"))
    hard_constraints_exact = bool(
        _constraint_rows_exact_1(cascade_vs_seq, "cascade")
        and _constraint_rows_exact_1(hardneg_vs_seq, "hardneg")
        and _constraint_rows_exact_1(hardneg_vs_cascade, "hardneg")
    )
    caption = (
        "Figure 2. Cascade decoding separates a source-aware recall ranker from "
        "a precision reranker: the recall stage retains more oracle-best edits "
        f"(k64 retained fraction {_fmt(values.get('cascade_recall_k64'))}), "
        "then precision reranking selects a top-1 candidate under hard constraints. "
        f"The head256 cascade has a positive but non-significant TE trend versus "
        f"sequential precision (delta {_fmt_delta(values.get('cascade_vs_seq_delta'))}, "
        f"paired p {_fmt(values.get('cascade_vs_seq_p'))}); hard-negative v2 direct "
        f"top64 is the current stronger default versus sequential (delta "
        f"{_fmt_delta(values.get('hardneg_vs_seq_delta'))}, paired p "
        f"{_fmt(values.get('hardneg_vs_seq_p'))})."
    )
    source_audit = _source_audit(project_root)
    return {
        "artifact_kind": "paper_figure2_cascade_recall_precision",
        "project_root": os.path.abspath(project_root),
        "claim_policy": CLAIM_POLICY,
        "summary": {
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "source_files_ready": all(row["exists"] for row in source_audit),
            "ready_for_cascade_figure_draft": len(nodes) == 8 and len(edges) == 8,
            "recall_precision_roles_visible": True,
            "cascade_te_significant_vs_sequential": cascade_p is not None and cascade_p < 0.05,
            "hardneg_v2_significant_vs_sequential": hardneg_p is not None and hardneg_p < 0.05,
            "hardneg_v2_direct_default": True,
            "ready_for_cascade_positive_claim": False,
            "hard_constraints_exact_1": hard_constraints_exact,
        },
        "diagnostics": values,
        "caption": caption,
        "nodes": nodes,
        "edges": edges,
        "mermaid": _mermaid(nodes, edges),
        "source_audit": source_audit,
    }


def build_paper_figure2(
    project_root: str,
    run_mode: str = "development",
    artifact_paths: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    if normalize_run_mode(run_mode) == "paper":
        return paper_builder_gate("paper_figure2_cascade_recall_precision", project_root, artifact_paths, __file__)
    report = _build_paper_figure2_development(project_root)
    report.update({"claim_tier": "development_only", "paper_eligible": False})
    return report


def write_report_json(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return path


def write_report_markdown(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    nodes = report.get("nodes", [])
    edges = report.get("edges", [])
    summary = report.get("summary", {})
    diagnostics = report.get("diagnostics", {})
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)):
        nodes = []
    if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes)):
        edges = []
    if not isinstance(summary, Mapping):
        summary = {}
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Paper Figure 2: Cascade Recall / Precision Division\n\n")
        fh.write(f"- Claim policy: {report.get('claim_policy', CLAIM_POLICY)}\n")
        fh.write(
            f"- Ready for cascade figure draft: `{summary.get('ready_for_cascade_figure_draft')}`; "
            f"cascade TE significant vs sequential: `{summary.get('cascade_te_significant_vs_sequential')}`; "
            f"ready for cascade positive claim: `{summary.get('ready_for_cascade_positive_claim')}`\n"
        )
        fh.write(
            f"- Hard-negative v2 direct default: `{summary.get('hardneg_v2_direct_default')}`; "
            f"hard constraints exact-1: `{summary.get('hard_constraints_exact_1')}`; "
            f"source files ready: `{summary.get('source_files_ready')}`\n\n"
        )
        fh.write("## Mermaid Source\n\n")
        fh.write("```mermaid\n")
        fh.write(str(report.get("mermaid", "")))
        fh.write("\n```\n\n")
        fh.write("## Caption\n\n")
        fh.write(str(report.get("caption", "")))
        fh.write("\n\n## Key Diagnostics\n\n")
        fh.write("| Metric | Value |\n")
        fh.write("|---|---:|\n")
        for key in (
            "previous_ranker_recall",
            "sourceaware_recall",
            "cascade_recall_k64",
            "cascade_vs_seq_delta",
            "cascade_vs_seq_p",
            "hardneg_vs_seq_delta",
            "hardneg_vs_seq_p",
            "hardneg_vs_cascade_delta",
            "hardneg_vs_cascade_p",
            "cascade10k_vs_hardneg_delta",
            "cascade10k_vs_hardneg_p",
            "win_fraction",
            "mean_cascade_gain",
        ):
            fh.write(f"| `{key}` | {_fmt(diagnostics.get(key))} |\n")
        fh.write("\n## Node Ledger\n\n")
        fh.write("| ID | Label | Role | Detail |\n")
        fh.write("|---|---|---|---|\n")
        for row in nodes:
            if isinstance(row, Mapping):
                fh.write(
                    f"| {row.get('id')} | {row.get('label')} | "
                    f"{row.get('role')} | {row.get('detail')} |\n"
                )
        fh.write("\n## Edge Ledger\n\n")
        fh.write("| Source | Target | Label |\n")
        fh.write("|---|---|---|\n")
        for row in edges:
            if isinstance(row, Mapping):
                fh.write(f"| {row.get('source')} | {row.get('target')} | {row.get('label')} |\n")
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--out-json", default="docs/paper_figure2_cascade_recall_precision.json")
    parser.add_argument("--out-md", default="docs/paper_figure2_cascade_recall_precision.md")
    parser.add_argument("--run-mode", choices=("development", "paper"), default="development")
    parser.add_argument("--paper-artifact", action="append", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    project_root = os.path.abspath(args.project_root)
    report = build_paper_figure2(project_root, args.run_mode, args.paper_artifact)
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
    "build_paper_figure2",
    "write_report_json",
    "write_report_markdown",
    "main",
]
