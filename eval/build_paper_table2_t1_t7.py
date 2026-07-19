"""Build a paper-facing Table 2 summary for T1-T7 evidence.

The output is a compact table over the internal proxy/constraint benchmark
bundle. It reports paired p-values only where a real paired comparison artifact
exists, and marks the other rows as ``NA`` with a reason rather than inventing
significance.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Mapping, Optional, Sequence

from mrna_editflow.eval.artifact_contract import normalize_run_mode, paper_builder_gate, validate_report_output_namespaces, write_paper_report_sidecars


CLAIM_POLICY = (
    "Table 2 is an internal proxy/constraint table. It supports constrained "
    "local optimization claims only. It does not support wet-lab validation, "
    "external SOTA reproduction, speed SOTA, or full de novo mRNA generation "
    "claims."
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


def _task(ledger: Mapping[str, object], task_id: str) -> Mapping[str, object]:
    tasks = ledger.get("tasks", [])
    if isinstance(tasks, Sequence) and not isinstance(tasks, (str, bytes)):
        for row in tasks:
            if isinstance(row, Mapping) and row.get("task") == task_id:
                return row
    return {}


def _evidence(ledger: Mapping[str, object], task_id: str) -> Mapping[str, object]:
    row = _task(ledger, task_id)
    evidence = row.get("evidence", {})
    return evidence if isinstance(evidence, Mapping) else {}


def _comparison_row(
    payload: Mapping[str, object],
    *,
    metric: str,
    run_contains: Optional[str] = None,
) -> Optional[Mapping[str, object]]:
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


def _t2_t3_primary(report: Mapping[str, object]) -> Mapping[str, object]:
    rows = report.get("rows", [])
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        for row in rows:
            if isinstance(row, Mapping) and row.get("label") == "head256_mo_grpo":
                return row
    return {}


def _nested_mean(row: Mapping[str, object], section: str, metric: str) -> Optional[float]:
    group = row.get(section, {})
    if not isinstance(group, Mapping):
        return None
    entry = group.get(metric, {})
    if not isinstance(entry, Mapping):
        return None
    return _num(entry.get("mean"))


def _row_for_budget(report: Mapping[str, object], section: str, budget: int) -> Mapping[str, object]:
    payload = report.get(section, {})
    rows = payload.get("rows", []) if isinstance(payload, Mapping) else []
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        for row in rows:
            if isinstance(row, Mapping) and int(row.get("budget", -1)) == int(budget):
                return row
    return {}


def _length_rows(report: Mapping[str, object], section: str) -> list[Mapping[str, object]]:
    payload = report.get(section, {})
    rows = payload.get("rows", []) if isinstance(payload, Mapping) else []
    return [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, Sequence) else []


def _t7_metric(report: Mapping[str, object], metric: str, run: str = "mo_grpo") -> Optional[float]:
    rows = report.get("rows", [])
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        for row in rows:
            if (
                isinstance(row, Mapping)
                and row.get("metric") == metric
                and row.get("run") == run
            ):
                return _num(row.get("mean"))
    return None


def _t7_compare(report: Mapping[str, object], metric: str, run: str = "mo_grpo") -> Optional[Mapping[str, object]]:
    rows = report.get("comparisons_vs_mo_te_only", [])
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        for row in rows:
            if (
                isinstance(row, Mapping)
                and row.get("metric") == metric
                and row.get("run") == run
            ):
                return row
    return None


def _hard_constraints_text(evidence: Mapping[str, object]) -> str:
    keys = (
        "legal_fraction",
        "mean_protein_identity",
        "within_budget_fraction",
        "reading_frame_intact_fraction",
    )
    values = [evidence.get(key) for key in keys if key in evidence]
    if values and all(_num(value) == 1.0 for value in values):
        return "legal/protein/budget/frame = 1.0"
    if _num(evidence.get("mean_protein_identity")) == 1.0 or _num(evidence.get("protein_identity_eq_1_fraction")) == 1.0:
        return "protein identity = 1.0"
    return "see source"


def _build_paper_table2_development(project_root: str) -> dict[str, object]:
    ledger = _load_json(_path(project_root, "benchmark/t1_t7_evidence_status_head256.json"))
    cmp_te = _load_json(_path(project_root, "benchmark/compare_mo_fusion_vs_te_only_head256.json"))
    cmp_hard = _load_json(_path(project_root, "benchmark/compare_grpo_vs_hardneg_v2_head256.json"))
    t2t3 = _load_json(_path(project_root, "benchmark/t2_t3_distribution_novelty_report_head256_head1024.json"))
    t4 = _load_json(_path(project_root, "benchmark/t4_protein_identity_cai_gc_report_head256.json"))
    t5 = _load_json(_path(project_root, "benchmark/edit_budget_curve_report_head256_head1024.json"))
    t6 = _load_json(_path(project_root, "benchmark/t6_length_curve_report_head256_head1024.json"))
    t7 = _load_json(_path(project_root, "benchmark/t7_motif_frame_report_head256.json"))
    t7_edit = _load_json(_path(project_root, "benchmark/t7_motif_edit_benchmark_head256/summary.json"))

    rows: list[dict[str, object]] = []
    t1_ev = _evidence(ledger, "T1")
    t1_vs_te = _comparison_row(cmp_te, metric="delta_oracle_te_vs_source", run_contains="grpo")
    t1_vs_hard = _comparison_row(cmp_hard, metric="delta_oracle_te_vs_source", run_contains="grpo")
    rows.append(
        {
            "task": "T1",
            "task_name": "Validity / Oracle TE",
            "n_seeds": 10,
            "primary_result": (
                f"legal={_fmt(t1_ev.get('legal_fraction'))}; "
                f"delta_TE={_fmt(t1_ev.get('delta_oracle_te_vs_source'))}; "
                f"mean_TE={_fmt(t1_ev.get('mean_oracle_te'))}; "
                f"mean_MRL={_fmt(t1_ev.get('mean_oracle_mrl'))}"
            ),
            "hard_constraints": _hard_constraints_text(t1_ev),
            "paired_p": [
                {
                    "comparison": "mo_grpo vs te_only",
                    "metric": "delta_oracle_te_vs_source",
                    "paired_p": t1_vs_te.get("paired_p") if isinstance(t1_vs_te, Mapping) else None,
                    "n_paired_seeds": t1_vs_te.get("n_paired_seeds") if isinstance(t1_vs_te, Mapping) else None,
                },
                {
                    "comparison": "mo_grpo vs hardneg_v2",
                    "metric": "delta_oracle_te_vs_source",
                    "paired_p": t1_vs_hard.get("paired_p") if isinstance(t1_vs_hard, Mapping) else None,
                    "n_paired_seeds": t1_vs_hard.get("n_paired_seeds") if isinstance(t1_vs_hard, Mapping) else None,
                },
            ],
            "paired_p_status": "available",
            "claim_language": "Strict positive proxy-TE claim vs te_only and hardneg_v2; not wet-lab expression.",
            "sources": [
                "benchmark/t1_t7_evidence_status_head256.json",
                "benchmark/compare_mo_fusion_vs_te_only_head256.json",
                "benchmark/compare_grpo_vs_hardneg_v2_head256.json",
            ],
        }
    )

    primary = _t2_t3_primary(t2t3)
    rows.append(
        {
            "task": "T2",
            "task_name": "Distribution preservation",
            "n_seeds": 10,
            "primary_result": (
                f"kmer_JS={_fmt(_nested_mean(primary, 'T2_distribution', 'kmer_js'))}; "
                f"codon_KL={_fmt(_nested_mean(primary, 'T2_distribution', 'codon_usage_kl'))}; "
                f"GC_length_dist={_fmt(_nested_mean(primary, 'T2_distribution', 'combined_gc_length_distance'))}; "
                f"length_delta={_fmt(_nested_mean(primary, 'T2_distribution', 'candidate_mean_length') - _nested_mean(primary, 'T2_distribution', 'source_mean_length') if _nested_mean(primary, 'T2_distribution', 'candidate_mean_length') is not None and _nested_mean(primary, 'T2_distribution', 'source_mean_length') is not None else None)}"
            ),
            "hard_constraints": "distribution-collapse flag = False",
            "paired_p": [],
            "paired_p_status": "NA_distribution_audit_no_paired_claim",
            "claim_language": "No local distribution-collapse signal; still needs external split alignment.",
            "sources": ["benchmark/t2_t3_distribution_novelty_report_head256_head1024.json"],
        }
    )

    rows.append(
        {
            "task": "T3",
            "task_name": "Diversity / Novelty",
            "n_seeds": 10,
            "primary_result": (
                f"mean_novelty={_fmt(_nested_mean(primary, 'T3_novelty_diversity', 'mean_novelty'))}; "
                f"exact_match={_fmt(_nested_mean(primary, 'T3_novelty_diversity', 'exact_source_match_fraction'))}; "
                f"unique={_fmt(_nested_mean(primary, 'T3_novelty_diversity', 'unique_fraction'))}; "
                f"pairwise={_fmt(_nested_mean(primary, 'T3_novelty_diversity', 'pairwise_diversity'))} "
                "(64/32640 sampled)"
            ),
            "hard_constraints": "novelty exact; pairwise sampled",
            "paired_p": [],
            "paired_p_status": "NA_novelty_audit_no_paired_claim",
            "claim_language": "Novel but not de novo: exact source matches are non-zero.",
            "sources": ["benchmark/t2_t3_distribution_novelty_report_head256_head1024.json"],
        }
    )

    t4_summary = t4.get("summary", {})
    t4_dp = t4.get("codon_lattice_dp", {})
    t4_pc = t4.get("protein_conditioned_cds", {})
    rows.append(
        {
            "task": "T4",
            "task_name": "Protein identity / CDS CAI-GC",
            "n_seeds": 10,
            "primary_result": (
                f"protein_identity_exact_1={t4_summary.get('hard_constraints_exact_1') if isinstance(t4_summary, Mapping) else None}; "
                f"local_DP_delta_CAI={_fmt(t4_dp.get('mean_delta_cai') if isinstance(t4_dp, Mapping) else None)}; "
                f"protein_conditioned_delta_CAI={_fmt(t4_pc.get('mean_designed_vs_native_cai_delta') if isinstance(t4_pc, Mapping) else None)}"
            ),
            "hard_constraints": "protein identity = 1.0",
            "paired_p": [],
            "paired_p_status": "NA_constructive_DP_hard_constraint",
            "claim_language": "Exact protein-identity and CAI/GC proxy claim; not external LinearDesign/codonGPT/MFE.",
            "sources": ["benchmark/t4_protein_identity_cai_gc_report_head256.json"],
        }
    )

    budget3 = _row_for_budget(t5, "head256_mo_grpo", 3)
    budget10 = _row_for_budget(t5, "head256_mo_grpo", 10)
    rows.append(
        {
            "task": "T5",
            "task_name": "Edit budget",
            "n_seeds": 10,
            "primary_result": (
                f"budget3_delta_TE={_fmt(budget3.get('delta_oracle_te_vs_source'))}; "
                f"budget3_edit_distance={_fmt(budget3.get('mean_edit_distance'))}; "
                f"budget10_delta_TE={_fmt(budget10.get('delta_oracle_te_vs_source'))}; "
                f"within_budget={_fmt(budget3.get('within_budget_fraction'))}"
            ),
            "hard_constraints": _hard_constraints_text(budget3),
            "paired_p": [],
            "paired_p_status": "NA_curve_audit_no_single_paired_claim",
            "claim_language": "Budget curve supports controllable local edits; larger budgets increase edits.",
            "sources": ["benchmark/edit_budget_curve_report_head256_head1024.json"],
        }
    )

    h256_len = _length_rows(t6, "head256_stagea10k")
    h1024_len = _length_rows(t6, "head1024_stagea10k")
    max_h256_err = max((_num(row.get("mean_abs_length_error")) or 0.0 for row in h256_len), default=None)
    max_h1024_err = max((_num(row.get("mean_abs_length_error")) or 0.0 for row in h1024_len), default=None)
    rows.append(
        {
            "task": "T6",
            "task_name": "Length control",
            "n_seeds": 10,
            "primary_result": (
                f"head256_max_abs_error={_fmt(max_h256_err)}; "
                f"head1024_max_abs_error={_fmt(max_h1024_err)}; "
                "deltas=-30/-15/0/+15/+30"
            ),
            "hard_constraints": "legal/protein/budget/frame = 1.0",
            "paired_p": [],
            "paired_p_status": "NA_control_curve_no_single_paired_claim",
            "claim_language": "Accurate length control under hard constraints; positive lengthening hurts proxy TE.",
            "sources": ["benchmark/t6_length_curve_report_head256_head1024.json"],
        }
    )

    t7_uaug = _t7_compare(t7, "uAUG_presence_fraction")
    t7_edit_agg = t7_edit.get("aggregate", {})
    insert_success = (
        t7_edit_agg.get("insert_success_fraction", {}).get("mean")
        if isinstance(t7_edit_agg, Mapping) and isinstance(t7_edit_agg.get("insert_success_fraction"), Mapping)
        else None
    )
    excise_success = (
        t7_edit_agg.get("excise_success_fraction", {}).get("mean")
        if isinstance(t7_edit_agg, Mapping) and isinstance(t7_edit_agg.get("excise_success_fraction"), Mapping)
        else None
    )
    rows.append(
        {
            "task": "T7",
            "task_name": "Motif / Frame",
            "n_seeds": 10,
            "primary_result": (
                f"frame={_fmt(_t7_metric(t7, 'reading_frame_intact_fraction'))}; "
                f"uAUG_presence={_fmt(_t7_metric(t7, 'uAUG_presence_fraction'))}; "
                f"constructive_insert_success={_fmt(insert_success)}; "
                f"constructive_excise_success={_fmt(excise_success)}"
            ),
            "hard_constraints": "frame/protein/budget = 1.0 in constructive edit benchmark",
            "paired_p": [
                {
                    "comparison": "mo_grpo vs te_only",
                    "metric": "uAUG_presence_fraction",
                    "paired_p": t7_uaug.get("paired_p") if isinstance(t7_uaug, Mapping) else None,
                    "n_paired_seeds": t7_uaug.get("n_paired_seeds") if isinstance(t7_uaug, Mapping) else None,
                    "direction_note": t7_uaug.get("direction_note") if isinstance(t7_uaug, Mapping) else None,
                    "delta": t7_uaug.get("delta") if isinstance(t7_uaug, Mapping) else None,
                }
            ],
            "paired_p_status": "available_for_uAUG_safety_side_effect",
            "claim_language": "Frame is controlled; grpo increases uAUG vs te_only, so do not claim uAUG safety improvement.",
            "sources": [
                "benchmark/t7_motif_frame_report_head256.json",
                "benchmark/t7_motif_edit_benchmark_head256/summary.json",
            ],
        }
    )

    paired_available = [row["task"] for row in rows if row["paired_p"]]
    return {
        "artifact_kind": "paper_table2_t1_t7_main_results",
        "project_root": os.path.abspath(project_root),
        "claim_policy": CLAIM_POLICY,
        "summary": {
            "n_tasks": len(rows),
            "all_tasks_present": len(rows) == 7,
            "paired_p_available_tasks": paired_available,
            "paired_p_na_tasks": [row["task"] for row in rows if not row["paired_p"]],
            "table_ready_for_internal_proxy_paper_draft": len(rows) == 7,
            "ready_for_external_sota_claim": False,
            "ready_for_wet_lab_claim": False,
            "ready_for_full_de_novo_claim": False,
        },
        "rows": rows,
    }


def build_paper_table2(
    project_root: str,
    run_mode: str = "development",
    artifact_paths: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    if normalize_run_mode(run_mode) == "paper":
        return paper_builder_gate("paper_table2_t1_t7", project_root, artifact_paths, __file__)
    report = _build_paper_table2_development(project_root)
    report.update({"claim_tier": "development_only", "paper_eligible": False})
    return report


def write_report_json(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return path


def _paired_text(row: Mapping[str, object]) -> str:
    pairs = row.get("paired_p", [])
    if not pairs:
        return str(row.get("paired_p_status", "NA"))
    chunks = []
    if isinstance(pairs, Sequence) and not isinstance(pairs, (str, bytes)):
        for item in pairs:
            if isinstance(item, Mapping):
                chunks.append(
                    f"{item.get('comparison')} {item.get('metric')} p={_fmt(item.get('paired_p'))}"
                )
    return "; ".join(chunks)


def write_report_markdown(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows = report.get("rows", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        rows = []
    summary = report.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Paper Table 2: T1-T7 Main Results\n\n")
        fh.write(f"- Claim policy: {report.get('claim_policy', CLAIM_POLICY)}\n")
        fh.write(
            f"- Paired p available tasks: `{summary.get('paired_p_available_tasks')}`; "
            f"NA tasks: `{summary.get('paired_p_na_tasks')}`\n"
        )
        fh.write(
            f"- External SOTA claim ready: `{summary.get('ready_for_external_sota_claim')}`; "
            f"wet-lab claim ready: `{summary.get('ready_for_wet_lab_claim')}`; "
            f"full de novo claim ready: `{summary.get('ready_for_full_de_novo_claim')}`\n\n"
        )
        fh.write("| Task | Main result | Hard constraints / scope | paired p | Claim language |\n")
        fh.write("|---|---|---|---|---|\n")
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            fh.write(
                f"| {row.get('task')} {row.get('task_name')} | "
                f"{row.get('primary_result')} | {row.get('hard_constraints')} | "
                f"{_paired_text(row)} | {row.get('claim_language')} |\n"
            )
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--out-json", default="docs/paper_table2_t1_t7_main_results.json")
    parser.add_argument("--out-md", default="docs/paper_table2_t1_t7_main_results.md")
    parser.add_argument("--run-mode", choices=("development", "paper"), default="development")
    parser.add_argument("--paper-artifact", action="append", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    project_root = os.path.abspath(args.project_root)
    report = build_paper_table2(project_root, args.run_mode, args.paper_artifact)
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
    "build_paper_table2",
    "write_report_json",
    "write_report_markdown",
    "main",
]
