"""Build paper Table 4: architecture ablation evidence.

This table summarizes architectural modules that have existing audit artifacts:
region adapters/FiLM-style conditioning, synonymous codon masks/lattice, source-
aware teacher ranking, and cascade decoding. Negative and non-significant
results remain part of the table so the paper draft does not over-select only
positive findings.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Mapping, Optional, Sequence

from mrna_editflow.eval.artifact_contract import normalize_run_mode, paper_builder_gate, validate_report_output_namespaces, write_paper_report_sidecars


CLAIM_POLICY = (
    "Table 4 is an architecture-ablation table over existing proxy/offline "
    "artifacts. Negative or non-significant modules must be reported as such. "
    "Do not turn region-adapter or cascade trends into positive SOTA claims."
)


def _load_json(path: str) -> Mapping[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _path(project_root: str, rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(project_root, rel)


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


def _row_by_metric(payload: Mapping[str, object], metric: str, run_contains: Optional[str] = None) -> Optional[Mapping[str, object]]:
    rows = payload.get("rows", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return None
    for row in rows:
        if not isinstance(row, Mapping) or row.get("metric") != metric:
            continue
        if run_contains is not None and run_contains not in str(row.get("run", "")):
            continue
        return row
    return None


def _region_best(decision: Mapping[str, object], baseline: str = "hardneg_v2_top64") -> dict[str, object]:
    summary = decision.get("summary", {})
    runs = decision.get("runs", [])
    if not isinstance(summary, Mapping) or not isinstance(runs, Sequence):
        return {}
    best = summary.get("best_run_vs_hardneg") if baseline == "hardneg_v2_top64" else None
    if not isinstance(best, str):
        return {}
    for row in runs:
        if not isinstance(row, Mapping) or row.get("run") != best:
            continue
        primary = row.get("primary_by_baseline", {})
        if not isinstance(primary, Mapping):
            return {}
        stats = primary.get(baseline, {})
        if not isinstance(stats, Mapping):
            return {}
        return {
            "run": best,
            "delta": stats.get("delta"),
            "paired_p": stats.get("paired_p"),
            "run_mean": stats.get("run_mean"),
            "baseline_mean": stats.get("baseline_mean"),
            "signal": stats.get("signal"),
            "constraints_exact_1": row.get("constraints_exact_1"),
        }
    return {}


def _aggregate(payload: Mapping[str, object]) -> Mapping[str, object]:
    agg = payload.get("aggregate", {})
    return agg if isinstance(agg, Mapping) else {}


def _build_paper_table4_development(project_root: str) -> dict[str, object]:
    region256 = _load_json(_path(project_root, "benchmark/region_adapter_decision_report_head256.json"))
    region1024 = _load_json(_path(project_root, "benchmark/region_adapter_decision_report_head1024.json"))
    region_audit256 = _load_json(_path(project_root, "benchmark/region_adapter_result_audit_head256.json"))
    region_audit1024 = _load_json(_path(project_root, "benchmark/region_adapter_result_audit_head1024.json"))
    codon_dp = _load_json(_path(project_root, "benchmark/codon_lattice_dp_head256.json"))
    source_base = _load_json(_path(project_root, "benchmark/proposal_ranking_t5_base_full1k_head64.json"))
    source_ranker = _load_json(_path(project_root, "benchmark/proposal_ranking_t5_ranker_full1k_final_head64.json"))
    sourceaware = _load_json(_path(project_root, "benchmark/proposal_ranking_t5_sourceaware_hybrid_teacher_head64.json"))
    cascade_k64 = _load_json(_path(project_root, "benchmark/cascade_sourceaware_to_sequential_head64_k64.json"))
    cascade_vs_seq = _load_json(_path(project_root, "benchmark/compare_t5_head256_cascade_vs_seq_top64.json"))
    cascade10k = _load_json(_path(project_root, "benchmark/compare_cascade_10krecall_vs_hardneg_v2.json"))
    cascade_error = _load_json(_path(project_root, "benchmark/cascade_error_analysis_head256_top64.json"))

    rows: list[dict[str, object]] = []

    best256 = _region_best(region256)
    best1024 = _region_best(region1024)
    audit256_summary = region_audit256.get("summary", {})
    audit1024_summary = region_audit1024.get("summary", {})
    rows.append(
        {
            "module": "Region adapters / FiLM-style conditioning",
            "scope": "T5 head256/head1024 region-specialized adapters",
            "primary_metric": "delta_oracle_te_vs_source vs hardneg_v2",
            "result": (
                f"head256 best={best256.get('run')} delta={_fmt(best256.get('delta'))} "
                f"p={_fmt(best256.get('paired_p'))}; "
                f"head1024 best={best1024.get('run')} delta={_fmt(best1024.get('delta'))} "
                f"p={_fmt(best1024.get('paired_p'))}"
            ),
            "hard_constraints": (
                f"head256 exact1={audit256_summary.get('all_constraints_exact_1') if isinstance(audit256_summary, Mapping) else None}; "
                f"head1024 exact1={audit1024_summary.get('all_constraints_exact_1') if isinstance(audit1024_summary, Mapping) else None}"
            ),
            "signal": "negative_ablation",
            "claim_language": "Report as failed/negative ablation; hard constraints safe but TE regresses.",
            "sources": [
                "benchmark/region_adapter_decision_report_head256.json",
                "benchmark/region_adapter_decision_report_head1024.json",
            ],
        }
    )

    dp_summary = codon_dp.get("summary", {})
    if not isinstance(dp_summary, Mapping):
        dp_summary = {}
    rows.append(
        {
            "module": "Synonymous codon mask / codon-lattice DP",
            "scope": "T4 CDS-only synonymous optimization",
            "primary_metric": "protein identity and CAI/GC proxy",
            "result": (
                f"protein_identity={_fmt(dp_summary.get('protein_identity_fraction'))}; "
                f"delta_CAI={_fmt(dp_summary.get('mean_delta_cai'))}; "
                f"delta_GC={_fmt(dp_summary.get('mean_delta_gc'))}; "
                f"mean_codon_changes={_fmt(dp_summary.get('mean_codon_changes'))}"
            ),
            "hard_constraints": "protein identity = 1.0",
            "signal": "constructive_constraint_positive",
            "claim_language": "Supports exact CDS protein-preserving optimization; not external LinearDesign/codonGPT.",
            "sources": ["benchmark/codon_lattice_dp_head256.json"],
        }
    )

    base_agg = _aggregate(source_base)
    ranker_agg = _aggregate(source_ranker)
    source_agg = _aggregate(sourceaware)
    rows.append(
        {
            "module": "Source-aware teacher / recall ranker",
            "scope": "T5 proposal-ranking head64 diagnostic",
            "primary_metric": "oracle-best recall and model regret",
            "result": (
                f"base recall={_fmt(base_agg.get('oracle_best_in_model_top_k_fraction'))}; "
                f"ranker recall={_fmt(ranker_agg.get('oracle_best_in_model_top_k_fraction'))}; "
                f"source-aware recall={_fmt(source_agg.get('oracle_best_in_model_top_k_fraction'))}; "
                f"source-aware regret={_fmt(source_agg.get('mean_model_regret'))}"
            ),
            "hard_constraints": "proposal-ranking diagnostic only",
            "signal": "recall_positive_regret_mixed",
            "claim_language": "Source-aware teacher improves oracle-best recall, but does not by itself prove top-1 TE gain.",
            "sources": [
                "benchmark/proposal_ranking_t5_base_full1k_head64.json",
                "benchmark/proposal_ranking_t5_ranker_full1k_final_head64.json",
                "benchmark/proposal_ranking_t5_sourceaware_hybrid_teacher_head64.json",
            ],
        }
    )

    cascade_agg = _aggregate(cascade_k64)
    cascade_cmp = _row_by_metric(cascade_vs_seq, "delta_oracle_te_vs_source", "cascade")
    cascade10k_cmp = _row_by_metric(cascade10k, "delta_oracle_te_vs_source", "cascade")
    err_agg = cascade_error.get("aggregate", {})
    if not isinstance(err_agg, Mapping):
        err_agg = {}
    rows.append(
        {
            "module": "Source-aware cascade recall -> precision",
            "scope": "T5 cascade decoding head64",
            "primary_metric": "delta_oracle_te_vs_source and recall diagnostics",
            "result": (
                f"old cascade vs sequential delta={_fmt(cascade_cmp.get('delta') if isinstance(cascade_cmp, Mapping) else None)} "
                f"p={_fmt(cascade_cmp.get('paired_p') if isinstance(cascade_cmp, Mapping) else None)}; "
                f"oracle-best recall k64={_fmt(cascade_agg.get('oracle_best_in_recall_top_k_fraction'))}; "
                f"win_fraction={_fmt(err_agg.get('win_record_fraction'))}"
            ),
            "hard_constraints": "legal/protein/budget/frame = 1.0 in multiseed comparisons",
            "signal": "trend_not_significant",
            "claim_language": "Old cascade has small non-significant trend; use cautious language only.",
            "sources": [
                "benchmark/compare_t5_head256_cascade_vs_seq_top64.json",
                "benchmark/cascade_sourceaware_to_sequential_head64_k64.json",
                "benchmark/cascade_error_analysis_head256_top64.json",
            ],
        }
    )

    rows.append(
        {
            "module": "Stage A 10k recall -> hardneg_v2 precision cascade",
            "scope": "T5 head256 10k recall cascade",
            "primary_metric": "delta_oracle_te_vs_source vs hardneg_v2",
            "result": (
                f"delta={_fmt(cascade10k_cmp.get('delta') if isinstance(cascade10k_cmp, Mapping) else None)}; "
                f"p={_fmt(cascade10k_cmp.get('paired_p') if isinstance(cascade10k_cmp, Mapping) else None)}; "
                f"run_mean={_fmt(cascade10k_cmp.get('run_mean') if isinstance(cascade10k_cmp, Mapping) else None)}"
            ),
            "hard_constraints": "legal/protein/budget/frame = 1.0",
            "signal": "not_positive",
            "claim_language": "10k recall did not improve top-1 TE over hardneg_v2; report as negative cascade ablation.",
            "sources": ["benchmark/compare_cascade_10krecall_vs_hardneg_v2.json"],
        }
    )

    signal_counts: dict[str, int] = {}
    for row in rows:
        signal = str(row["signal"])
        signal_counts[signal] = signal_counts.get(signal, 0) + 1
    return {
        "artifact_kind": "paper_table4_architecture_ablation",
        "project_root": os.path.abspath(project_root),
        "claim_policy": CLAIM_POLICY,
        "summary": {
            "n_modules": len(rows),
            "signal_counts": signal_counts,
            "table_ready_for_architecture_draft": len(rows) == 5,
            "positive_sota_claim_ready": False,
            "negative_results_included": True,
        },
        "rows": rows,
    }


def build_paper_table4(
    project_root: str,
    run_mode: str = "development",
    artifact_paths: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    if normalize_run_mode(run_mode) == "paper":
        return paper_builder_gate("paper_table4_architecture_ablation", project_root, artifact_paths, __file__)
    report = _build_paper_table4_development(project_root)
    report.update({"claim_tier": "development_only", "paper_eligible": False})
    return report


def write_report_json(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return path


def write_report_markdown(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows = report.get("rows", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        rows = []
    summary = report.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Paper Table 4: Architecture Ablation\n\n")
        fh.write(f"- Claim policy: {report.get('claim_policy', CLAIM_POLICY)}\n")
        fh.write(
            f"- Modules: `{summary.get('n_modules')}`; signals: `{summary.get('signal_counts')}`; "
            f"positive SOTA claim ready: `{summary.get('positive_sota_claim_ready')}`\n\n"
        )
        fh.write("| Module | Scope | Main result | Constraints / scope | Signal | Claim language |\n")
        fh.write("|---|---|---|---|---|---|\n")
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            fh.write(
                f"| {row.get('module')} | {row.get('scope')} | {row.get('result')} | "
                f"{row.get('hard_constraints')} | {row.get('signal')} | {row.get('claim_language')} |\n"
            )
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--out-json", default="docs/paper_table4_architecture_ablation.json")
    parser.add_argument("--out-md", default="docs/paper_table4_architecture_ablation.md")
    parser.add_argument("--run-mode", choices=("development", "paper"), default="development")
    parser.add_argument("--paper-artifact", action="append", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    project_root = os.path.abspath(args.project_root)
    report = build_paper_table4(project_root, args.run_mode, args.paper_artifact)
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
    "build_paper_table4",
    "write_report_json",
    "write_report_markdown",
    "main",
]
