"""Build paper Figure 1: full-length mRNA edit-flow algorithm spec.

The artifact is a renderable figure specification, not a camera-ready drawing.
It records the paper-facing nodes, edges, hard constraints, and caption language
so the algorithm figure remains consistent with the project's claim boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Mapping, Optional, Sequence

from mrna_editflow.eval.artifact_contract import normalize_run_mode, paper_builder_gate, validate_report_output_namespaces, write_paper_report_sidecars


CLAIM_POLICY = (
    "Figure 1 may describe full-length constrained edit-flow optimization over "
    "5'UTR + CDS + 3'UTR. It must not depict MEF as unconstrained de novo "
    "generation, external SOTA reproduction, or wet-lab validated design."
)

FULL_LENGTH_SEGMENTS: tuple[str, ...] = ("5'UTR", "CDS", "3'UTR")
SOURCE_FILES: tuple[str, ...] = (
    "README.md",
    "sample.py",
    "eval/run_multiseed_benchmark.py",
    "benchmark/t1_t7_evidence_status_head256.json",
)


def _path(project_root: str, rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(project_root, rel)


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


def _nodes() -> list[dict[str, object]]:
    return [
        {
            "id": "A",
            "label": "Public full-transcript source",
            "role": "input",
            "detail": "Existing mRNA records with 5'UTR + CDS + 3'UTR and CDS coordinates.",
        },
        {
            "id": "B",
            "label": "Region and phase annotation",
            "role": "representation",
            "detail": "Track nucleotide region, CDS codon phase, source length, motifs, and edit budget.",
        },
        {
            "id": "C",
            "label": "CTMC edit bridge",
            "role": "generator_training",
            "detail": "Align source/target transcripts and sample gap-free intermediate edit states.",
        },
        {
            "id": "D",
            "label": "Stage A edit-flow generator",
            "role": "generator",
            "detail": "Predict insert, substitute, and delete rates under region-aware context.",
        },
        {
            "id": "E",
            "label": "Hard-constraint masks",
            "role": "constraint_layer",
            "detail": "Apply legal sequence checks, CDS synonymous lattice, protein identity, frame, length, motif, and budget gates.",
        },
        {
            "id": "F",
            "label": "Candidate pool",
            "role": "decoding",
            "detail": "Generate constraint-safe local edits from the source transcript.",
        },
        {
            "id": "G",
            "label": "Multi-objective ranker / fusion",
            "role": "reranking",
            "detail": "Rank candidates with TE proxy, CAI/GC, novelty, source-aware recall, and fusion modes.",
        },
        {
            "id": "H",
            "label": "T1-T7 evaluation gates",
            "role": "audit",
            "detail": "Audit legality, distribution, novelty, protein identity, edit budget, length, motif, and frame.",
        },
        {
            "id": "I",
            "label": "Selected constrained local optimization",
            "role": "output",
            "detail": "Report proxy/offline gains only under exact hard constraints and explicit claim boundaries.",
        },
    ]


def _edges() -> list[dict[str, object]]:
    return [
        {"source": "A", "target": "B", "label": "parse full-length regions"},
        {"source": "B", "target": "C", "label": "align source and target edits"},
        {"source": "C", "target": "D", "label": "train edit rates"},
        {"source": "D", "target": "E", "label": "decode with masks"},
        {"source": "E", "target": "F", "label": "filter invalid edits"},
        {"source": "F", "target": "G", "label": "score and fuse objectives"},
        {"source": "G", "target": "H", "label": "audit T1-T7"},
        {"source": "H", "target": "I", "label": "select reportable candidates"},
        {"source": "E", "target": "H", "label": "hard constraints must remain exact-1"},
    ]


def _constraint_callouts() -> list[dict[str, str]]:
    return [
        {
            "constraint": "Full-length representation",
            "language": "The editable object is 5'UTR + CDS + 3'UTR, not CDS-only or UTR-only.",
        },
        {
            "constraint": "CDS protein identity",
            "language": "CDS edits are constrained by a synonymous codon lattice and exact protein identity.",
        },
        {
            "constraint": "Frame and legality",
            "language": "Reading frame, start/stop validity, alphabet, and legal transcript checks are hard gates.",
        },
        {
            "constraint": "Edit budget / length / motif",
            "language": "Local edit budget, target length, Kozak/uAUG/polyA, and motif controls are evaluated explicitly.",
        },
        {
            "constraint": "Claim boundary",
            "language": "Outputs are constrained local optimizations from existing sources, not full de novo SOTA or wet-lab validation.",
        },
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
        label = str(edge.get("label"))
        lines.append(f"    {src} -->|{label}| {dst}")
    lines.extend(
        [
            "    classDef constraint fill:#fff7e6,stroke:#d48806,color:#262626",
            "    classDef audit fill:#f6ffed,stroke:#389e0d,color:#262626",
            "    class E constraint",
            "    class H audit",
        ]
    )
    return "\n".join(lines)


def _build_paper_figure1_development(project_root: str) -> dict[str, object]:
    nodes = _nodes()
    edges = _edges()
    constraints = _constraint_callouts()
    mermaid = _mermaid(nodes, edges)
    source_audit = _source_audit(project_root)
    source_files_ready = all(row["exists"] for row in source_audit)
    caption = (
        "Figure 1. mRNA-EditFlow represents an existing full-length transcript "
        "as 5'UTR + CDS + 3'UTR, samples region-aware CTMC edit trajectories, "
        "applies hard constraint masks for CDS protein identity, frame, legality, "
        "edit budget, length and motifs, then reranks constraint-safe local edits "
        "with multi-objective fusion before T1-T7 audit. The figure supports a "
        "constrained local-optimization claim only, not unconstrained de novo "
        "SOTA or wet-lab validation."
    )
    return {
        "artifact_kind": "paper_figure1_full_length_edit_flow",
        "project_root": os.path.abspath(project_root),
        "claim_policy": CLAIM_POLICY,
        "summary": {
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "full_length_segments": list(FULL_LENGTH_SEGMENTS),
            "source_files_ready": source_files_ready,
            "ready_for_algorithm_figure_draft": len(nodes) == 9 and len(edges) == 9,
            "ready_for_full_de_novo_claim": False,
            "ready_for_wet_lab_claim": False,
            "hard_constraints_visible": True,
        },
        "caption": caption,
        "nodes": nodes,
        "edges": edges,
        "constraint_callouts": constraints,
        "mermaid": mermaid,
        "source_audit": source_audit,
    }


def build_paper_figure1(
    project_root: str,
    run_mode: str = "development",
    artifact_paths: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    if normalize_run_mode(run_mode) == "paper":
        return paper_builder_gate("paper_figure1_full_length_edit_flow", project_root, artifact_paths, __file__)
    report = _build_paper_figure1_development(project_root)
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
    constraints = report.get("constraint_callouts", [])
    summary = report.get("summary", {})
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)):
        nodes = []
    if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes)):
        edges = []
    if not isinstance(constraints, Sequence) or isinstance(constraints, (str, bytes)):
        constraints = []
    if not isinstance(summary, Mapping):
        summary = {}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Paper Figure 1: Full-Length Edit-Flow Algorithm\n\n")
        fh.write(f"- Claim policy: {report.get('claim_policy', CLAIM_POLICY)}\n")
        fh.write(
            f"- Ready for algorithm figure draft: `{summary.get('ready_for_algorithm_figure_draft')}`; "
            f"ready for full de novo claim: `{summary.get('ready_for_full_de_novo_claim')}`; "
            f"ready for wet-lab claim: `{summary.get('ready_for_wet_lab_claim')}`\n"
        )
        fh.write(
            f"- Full-length segments: `{summary.get('full_length_segments')}`; "
            f"hard constraints visible: `{summary.get('hard_constraints_visible')}`; "
            f"source files ready: `{summary.get('source_files_ready')}`\n\n"
        )
        fh.write("## Mermaid Source\n\n")
        fh.write("```mermaid\n")
        fh.write(str(report.get("mermaid", "")))
        fh.write("\n```\n\n")
        fh.write("## Caption\n\n")
        fh.write(str(report.get("caption", "")))
        fh.write("\n\n## Constraint Callouts\n\n")
        fh.write("| Constraint | Figure language |\n")
        fh.write("|---|---|\n")
        for row in constraints:
            if isinstance(row, Mapping):
                fh.write(f"| {row.get('constraint')} | {row.get('language')} |\n")
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
    parser.add_argument("--out-json", default="docs/paper_figure1_full_length_edit_flow.json")
    parser.add_argument("--out-md", default="docs/paper_figure1_full_length_edit_flow.md")
    parser.add_argument("--run-mode", choices=("development", "paper"), default="development")
    parser.add_argument("--paper-artifact", action="append", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    project_root = os.path.abspath(args.project_root)
    report = build_paper_figure1(project_root, args.run_mode, args.paper_artifact)
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
    "build_paper_figure1",
    "write_report_json",
    "write_report_markdown",
    "main",
]
