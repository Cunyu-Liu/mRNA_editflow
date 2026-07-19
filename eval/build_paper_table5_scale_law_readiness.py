"""Build paper Table 5: scale-law readiness evidence.

This table is intentionally conservative. Existing artifacts support an
evaluation-head-size scale-up audit, edit-budget and length-control curves, and
progress-log runtime evidence. They do not yet support a true controlled
training-data-size x model-size x training-steps scaling law.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Mapping, Optional, Sequence

from mrna_editflow.eval.artifact_contract import normalize_run_mode, paper_builder_gate, validate_report_output_namespaces, write_paper_report_sidecars


CLAIM_POLICY = (
    "Table 5 is a scale-law readiness table over existing proxy/offline "
    "artifacts. It may report evaluation-head-size, edit-budget, length-control, "
    "and progress-log runtime audits, but it must not claim a true data-size x "
    "model-size x training-steps scaling law until those controlled sweeps exist."
)

CONSTRAINT_METRICS: tuple[str, ...] = (
    "legal_fraction",
    "mean_protein_identity",
    "within_budget_fraction",
    "reading_frame_intact_fraction",
)
REQUIRED_SCALE_LAW_AXES: tuple[str, ...] = (
    "training_data_size",
    "model_size",
    "training_steps",
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


def _fmt_delta(value: object, digits: int = 5) -> str:
    number = _num(value)
    if number is None:
        return "NA"
    return f"{number:+.{digits}f}"


def _comparison(payload: Mapping[str, object], comparison_id: str) -> Mapping[str, object]:
    rows = payload.get("comparisons", [])
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        for row in rows:
            if isinstance(row, Mapping) and row.get("comparison_id") == comparison_id:
                return row
    return {}


def _constraints_exact(row: Mapping[str, object]) -> bool:
    values = [_num(row.get(metric)) for metric in CONSTRAINT_METRICS]
    return all(value is not None and abs(value - 1.0) <= 1e-12 for value in values)


def _section_rows(payload: Mapping[str, object], section: str) -> list[Mapping[str, object]]:
    section_payload = payload.get(section, {})
    rows = section_payload.get("rows", []) if isinstance(section_payload, Mapping) else []
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _curve_summary(
    payload: Mapping[str, object],
    section: str,
    *,
    x_key: str,
) -> dict[str, object]:
    rows = sorted(
        _section_rows(payload, section),
        key=lambda row: (_num(row.get(x_key)) is None, _num(row.get(x_key)) or 0.0),
    )
    deltas = [_num(row.get("delta_oracle_te_vs_source")) for row in rows]
    finite_deltas = [value for value in deltas if value is not None]
    monotonic = bool(finite_deltas) and all(
        finite_deltas[i] <= finite_deltas[i + 1] + 1e-12
        for i in range(len(finite_deltas) - 1)
    )
    best = max(
        rows,
        key=lambda row: _num(row.get("delta_oracle_te_vs_source")) or float("-inf"),
        default={},
    )
    return {
        "section": section,
        "n_rows": len(rows),
        "status": (
            payload.get(section, {}).get("status")
            if isinstance(payload.get(section, {}), Mapping)
            else None
        ),
        "x_values": [row.get(x_key) for row in rows],
        "first_delta": finite_deltas[0] if finite_deltas else None,
        "last_delta": finite_deltas[-1] if finite_deltas else None,
        "best_x": best.get(x_key) if isinstance(best, Mapping) else None,
        "best_delta": best.get("delta_oracle_te_vs_source") if isinstance(best, Mapping) else None,
        "monotonic_non_decreasing_delta": monotonic,
        "constraints_exact_1": bool(rows) and all(_constraints_exact(row) for row in rows),
    }


def _runtime_row(payload: Mapping[str, object], label: str) -> Mapping[str, object]:
    rows = payload.get("rows", [])
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        for row in rows:
            if isinstance(row, Mapping) and row.get("label") == label:
                return row
    return {}


def _runtime_stats(row: Mapping[str, object]) -> dict[str, object]:
    runtime = row.get("runtime", {})
    config = row.get("config", {})
    context = row.get("context", {})
    delta = None
    if isinstance(context, Mapping):
        metric = context.get("delta_oracle_te_vs_source", {})
        if isinstance(metric, Mapping):
            delta = metric.get("mean")
    seed_total = runtime.get("seed_total_s", {}) if isinstance(runtime, Mapping) else {}
    return {
        "label": row.get("label"),
        "n_records": config.get("n_records") if isinstance(config, Mapping) else None,
        "delta_oracle_te_vs_source": delta,
        "mean_seed_total_s": seed_total.get("mean") if isinstance(seed_total, Mapping) else None,
        "records_per_s_total": (
            runtime.get("measured_records_per_s_total") if isinstance(runtime, Mapping) else None
        ),
        "observed_elapsed_scope": (
            runtime.get("observed_elapsed_scope") if isinstance(runtime, Mapping) else None
        ),
    }


def _ratio(numerator: object, denominator: object) -> Optional[float]:
    num = _num(numerator)
    den = _num(denominator)
    if num is None or den is None or den == 0:
        return None
    return num / den


def _length_summary(payload: Mapping[str, object], section: str) -> dict[str, object]:
    rows = _section_rows(payload, section)
    targets = sorted(
        int(row["target_length_delta"])
        for row in rows
        if isinstance(row.get("target_length_delta"), int)
    )
    abs_errors = [_num(row.get("mean_abs_length_error")) for row in rows]
    finite_errors = [value for value in abs_errors if value is not None]
    positive_deltas = [
        _num(row.get("delta_oracle_te_vs_source"))
        for row in rows
        if (_num(row.get("target_length_delta")) or 0.0) > 0
    ]
    finite_positive_deltas = [value for value in positive_deltas if value is not None]
    return {
        "section": section,
        "status": (
            payload.get(section, {}).get("status")
            if isinstance(payload.get(section, {}), Mapping)
            else None
        ),
        "targets": targets,
        "targets_complete": targets == [-30, -15, 0, 15, 30],
        "max_mean_abs_length_error": max(finite_errors) if finite_errors else None,
        "positive_lengthening_min_delta_te": (
            min(finite_positive_deltas) if finite_positive_deltas else None
        ),
        "constraints_exact_1": bool(rows) and all(_constraints_exact(row) for row in rows),
    }


def _read_jsonl(path: str) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, Mapping):
                rows.append(row)
    return rows


def _latest_stage_a_scalelaw_sweep(project_root: str) -> dict[str, object]:
    bench_dir = _path(project_root, "benchmark")
    if not os.path.isdir(bench_dir):
        return {"status": "missing", "plan_ready": False}
    candidates: list[tuple[float, str, str]] = []
    for name in os.listdir(bench_dir):
        if not name.startswith("stage_a_scalelaw"):
            continue
        plan_path = os.path.join(bench_dir, name, "plan.json")
        if os.path.exists(plan_path):
            candidates.append((os.path.getmtime(plan_path), name, plan_path))
    if not candidates:
        return {"status": "missing", "plan_ready": False}

    _mtime, sweep_id, plan_path = max(candidates, key=lambda item: item[0])
    plan = _load_json(plan_path)
    axes = plan.get("axes", {})
    if not isinstance(axes, Mapping):
        axes = {}
    progress_path = os.path.join(os.path.dirname(plan_path), "progress.jsonl")
    summary_path = os.path.join(os.path.dirname(plan_path), "summary.json")
    progress_rows = _read_jsonl(progress_path)
    summary = _load_json(summary_path) if os.path.exists(summary_path) else {}
    n_runs = int(plan.get("n_runs", 0)) if isinstance(plan.get("n_runs"), int) else 0
    n_complete = (
        int(summary.get("n_complete", 0))
        if isinstance(summary, Mapping) and isinstance(summary.get("n_complete"), int)
        else 0
    )
    n_incomplete = (
        int(summary.get("n_incomplete", max(n_runs - n_complete, 0)))
        if isinstance(summary, Mapping) and isinstance(summary.get("n_incomplete"), int)
        else max(n_runs - n_complete, 0)
    )
    last_event = progress_rows[-1] if progress_rows else {}
    last_event_name = str(last_event.get("event")) if isinstance(last_event, Mapping) else None
    load_wait_events = [
        row for row in progress_rows if isinstance(row, Mapping) and row.get("event") == "load_gate_wait"
    ]
    status = "complete" if n_runs > 0 and n_complete == n_runs else "queued_or_running"
    return {
        "status": status,
        "plan_ready": True,
        "sweep_id": sweep_id,
        "plan_path": os.path.relpath(plan_path, project_root),
        "progress_path": os.path.relpath(progress_path, project_root),
        "summary_path": os.path.relpath(summary_path, project_root),
        "summary_exists": os.path.exists(summary_path),
        "axes": {
            "data_sizes": list(axes.get("data_sizes", [])) if isinstance(axes.get("data_sizes"), Sequence) else [],
            "model_sizes": list(axes.get("model_sizes", [])) if isinstance(axes.get("model_sizes"), Sequence) else [],
            "step_counts": list(axes.get("step_counts", [])) if isinstance(axes.get("step_counts"), Sequence) else [],
            "seeds": list(axes.get("seeds", [])) if isinstance(axes.get("seeds"), Sequence) else [],
        },
        "n_runs": n_runs,
        "n_complete": n_complete,
        "n_incomplete": n_incomplete,
        "last_event": last_event_name,
        "latest_loadavg": last_event.get("loadavg") if isinstance(last_event, Mapping) else None,
        "load_gate_wait_events": len(load_wait_events),
        "claim_policy": plan.get("claim_policy"),
    }


def _build_paper_table5_development(project_root: str) -> dict[str, object]:
    scale = _load_json(_path(project_root, "docs/multiobjective_scaleup_claim_audit_head256_head1024.json"))
    runtime = _load_json(_path(project_root, "benchmark/t1_runtime_report_head256_head1024.json"))
    budget = _load_json(_path(project_root, "benchmark/edit_budget_curve_report_head256_head1024.json"))
    length = _load_json(_path(project_root, "benchmark/t6_length_curve_report_head256_head1024.json"))

    scale_summary = scale.get("summary", {})
    if not isinstance(scale_summary, Mapping):
        scale_summary = {}
    head256_grpo = _comparison(scale, "head256_mo_grpo_top64_vs_te_only")
    head1024_pareto = _comparison(scale, "head1024_mo_pareto_top64_vs_te_only")
    head1024_hardneg = _comparison(scale, "head1024_mo_pareto_top64_vs_hardneg_v2")
    yield_contraction = (
        (_num(head1024_pareto.get("delta")) is not None)
        and (_num(head256_grpo.get("delta")) is not None)
        and (_num(head1024_pareto.get("delta")) < _num(head256_grpo.get("delta")))
    )

    budget256 = _curve_summary(budget, "head256_mo_grpo", x_key="budget")
    budget1024 = _curve_summary(budget, "head1024_mo_pareto", x_key="budget")
    length256 = _length_summary(length, "head256_stagea10k")
    length1024 = _length_summary(length, "head1024_stagea10k")

    runtime256 = _runtime_stats(_runtime_row(runtime, "head256_mo_grpo"))
    runtime1024 = _runtime_stats(_runtime_row(runtime, "head1024_mo_pareto"))
    controlled_sweep = _latest_stage_a_scalelaw_sweep(project_root)
    has_controlled_plan = bool(controlled_sweep.get("plan_ready"))
    controlled_sweep_complete = (
        has_controlled_plan
        and controlled_sweep.get("n_runs") == controlled_sweep.get("n_complete")
        and int(controlled_sweep.get("n_runs") or 0) > 0
    )
    controlled_axes = controlled_sweep.get("axes", {})
    if not isinstance(controlled_axes, Mapping):
        controlled_axes = {}

    if has_controlled_plan:
        required_axis_row = {
            "scale_dimension": "Training data x model size x steps",
            "evidence": "load-gated controlled Stage A scale-law sweep plan",
            "result": (
                f"sweep={controlled_sweep.get('sweep_id')}; "
                f"data_sizes={controlled_axes.get('data_sizes')}; "
                f"model_sizes={controlled_axes.get('model_sizes')}; "
                f"steps={controlled_axes.get('step_counts')}; "
                f"runs complete={controlled_sweep.get('n_complete')}/"
                f"{controlled_sweep.get('n_runs')}; "
                f"last_event={controlled_sweep.get('last_event')}"
            ),
            "signal": (
                "controlled_scale_law_sweep_complete_pending_downstream"
                if controlled_sweep_complete
                else "required_scale_law_axes_queued_incomplete"
            ),
            "claim_language": (
                "Required scale-law axes are now planned/queued, but true scale-law "
                "claims remain blocked until runs complete and downstream audits are generated."
            ),
            "sources": [
                controlled_sweep.get("plan_path"),
                controlled_sweep.get("progress_path"),
            ],
        }
    else:
        required_axis_row = {
            "scale_dimension": "Training data x model size x steps",
            "evidence": "controlled scaling axes required for a true scale law",
            "result": "missing controlled sweeps for training_data_size, model_size, and training_steps",
            "signal": "required_scale_law_axes_missing",
            "claim_language": (
                "True scale-law claim is not ready; complete P3/P4 controlled "
                "sweeps before claiming data/model/step scaling."
            ),
            "sources": ["docs/next_steps_sota_roadmap.md"],
        }

    rows: list[dict[str, object]] = [
        {
            "scale_dimension": "Evaluation head-size scale-up",
            "evidence": "head256 vs head1024 multi-objective fusion comparisons",
            "result": (
                f"head256 grpo vs te_only: delta={_fmt_delta(head256_grpo.get('delta'))}, "
                f"p={_fmt(head256_grpo.get('paired_p'))}, signal={head256_grpo.get('signal')}; "
                f"head1024 pareto vs te_only: delta={_fmt_delta(head1024_pareto.get('delta'))}, "
                f"p={_fmt(head1024_pareto.get('paired_p'))}, signal={head1024_pareto.get('signal')}"
            ),
            "signal": "observed_yield_contraction",
            "claim_language": (
                "Head256 is strictly positive, but the stronger head1024 te_only "
                "control reduces the fusion gain to borderline/trend evidence only."
            ),
            "sources": ["docs/multiobjective_scaleup_claim_audit_head256_head1024.json"],
        },
        {
            "scale_dimension": "Legacy-control scale-up",
            "evidence": "head1024 fusion modes vs old hardneg_v2 baseline",
            "result": (
                f"head1024 pareto vs hardneg_v2: delta={_fmt_delta(head1024_hardneg.get('delta'))}, "
                f"p={_fmt(head1024_hardneg.get('paired_p'))}, signal={head1024_hardneg.get('signal')}; "
                f"all modes strict={scale_summary.get('head1024_vs_hardneg_v2_all_strict')}"
            ),
            "signal": "strict_vs_legacy_control_only",
            "claim_language": (
                "Strictly positive versus the older hardneg_v2 baseline, but this "
                "does not override the non-strict head1024 result versus te_only."
            ),
            "sources": ["docs/multiobjective_scaleup_claim_audit_head256_head1024.json"],
        },
        {
            "scale_dimension": "Edit-budget curve",
            "evidence": "budgets 1/2/3/5/10 at head256 and head1024",
            "result": (
                f"head256 best budget={budget256.get('best_x')} delta={_fmt_delta(budget256.get('best_delta'))}; "
                f"head1024 best budget={budget1024.get('best_x')} delta={_fmt_delta(budget1024.get('best_delta'))}; "
                f"monotonic=({budget256.get('monotonic_non_decreasing_delta')}, "
                f"{budget1024.get('monotonic_non_decreasing_delta')})"
            ),
            "signal": "controlled_budget_curve_complete",
            "claim_language": (
                "Larger edit budgets increase proxy TE under exact constraints; "
                "edit budget is an intervention-capacity sweep, not data/model/step scaling."
            ),
            "sources": ["benchmark/edit_budget_curve_report_head256_head1024.json"],
        },
        {
            "scale_dimension": "Length-control curve",
            "evidence": "target deltas -30/-15/0/+15/+30 at head256 and head1024",
            "result": (
                f"targets complete=({length256.get('targets_complete')}, {length1024.get('targets_complete')}); "
                f"max mean abs length error={_fmt(max(_num(length256.get('max_mean_abs_length_error')) or 0.0, _num(length1024.get('max_mean_abs_length_error')) or 0.0))}; "
                f"positive lengthening min delta=({ _fmt_delta(length256.get('positive_lengthening_min_delta_te'))}, "
                f"{_fmt_delta(length1024.get('positive_lengthening_min_delta_te'))})"
            ),
            "signal": "control_curve_complete_performance_mixed",
            "claim_language": (
                "Length control is complete and constraint-safe, but positive "
                "lengthening reduces proxy TE; this is control evidence, not a positive scale law."
            ),
            "sources": ["benchmark/t6_length_curve_report_head256_head1024.json"],
        },
        {
            "scale_dimension": "Runtime / throughput audit",
            "evidence": "progress-log timing for primary head256/head1024 runs",
            "result": (
                f"head256 records/s={_fmt(runtime256.get('records_per_s_total'))}, "
                f"mean seed s={_fmt(runtime256.get('mean_seed_total_s'))}; "
                f"head1024 records/s={_fmt(runtime1024.get('records_per_s_total'))}, "
                f"mean seed s={_fmt(runtime1024.get('mean_seed_total_s'))}; "
                f"throughput ratio={_fmt(_ratio(runtime1024.get('records_per_s_total'), runtime256.get('records_per_s_total')))}"
            ),
            "signal": "progress_log_runtime_audit_only",
            "claim_language": (
                "Useful runtime audit from progress logs only; do not report this "
                "as a strict hardware benchmark or speed SOTA."
            ),
            "sources": ["benchmark/t1_runtime_report_head256_head1024.json"],
        },
        required_axis_row,
    ]

    hard_constraints_exact = bool(
        scale_summary.get("available_compare_constraints_exact_1")
        and scale_summary.get("summary_constraints_complete")
        and budget256.get("constraints_exact_1")
        and budget1024.get("constraints_exact_1")
        and length256.get("constraints_exact_1")
        and length1024.get("constraints_exact_1")
    )
    partial_axes = [
        {"axis": "evaluation_head_size", "status": "available_partial", "values": [256, 1024]},
        {"axis": "edit_budget", "status": "available_control_curve", "values": budget256.get("x_values")},
        {"axis": "target_length_delta", "status": "available_control_curve", "values": length256.get("targets")},
        {"axis": "runtime_progress_logs", "status": "available_audit", "values": ["head256", "head1024"]},
    ]
    if has_controlled_plan:
        partial_axes.extend(
            [
                {
                    "axis": "training_data_size",
                    "status": "queued_controlled_sweep",
                    "values": controlled_axes.get("data_sizes"),
                },
                {
                    "axis": "model_size",
                    "status": "queued_controlled_sweep",
                    "values": controlled_axes.get("model_sizes"),
                },
                {
                    "axis": "training_steps",
                    "status": "queued_controlled_sweep",
                    "values": controlled_axes.get("step_counts"),
                },
            ]
        )
    missing_required_axes = [] if has_controlled_plan else list(REQUIRED_SCALE_LAW_AXES)
    incomplete_required_axes = (
        [] if controlled_sweep_complete else list(REQUIRED_SCALE_LAW_AXES)
    )
    return {
        "artifact_kind": "paper_table5_scale_law_readiness",
        "project_root": os.path.abspath(project_root),
        "claim_policy": CLAIM_POLICY,
        "summary": {
            "n_rows": len(rows),
            "table_ready_for_scale_law_readiness_draft": len(rows) == 6,
            "ready_for_true_scale_law_claim": False,
            "ready_for_monotonic_scale_law_claim": False,
            "yield_contraction_flag": bool(yield_contraction),
            "head1024_vs_te_only_strict_claim_allowed": scale_summary.get(
                "head1024_vs_te_only_strict_claim_allowed"
            ),
            "head1024_vs_te_only_best_signal": scale_summary.get(
                "head1024_vs_te_only_best_signal"
            ),
            "hard_constraints_exact_1": hard_constraints_exact,
            "missing_required_axes": missing_required_axes,
            "incomplete_required_axes": incomplete_required_axes,
            "controlled_sweep_plan_ready": has_controlled_plan,
            "controlled_sweep_status": controlled_sweep.get("status"),
            "controlled_sweep_complete": controlled_sweep_complete,
            "controlled_sweep_n_runs": controlled_sweep.get("n_runs"),
            "controlled_sweep_n_complete": controlled_sweep.get("n_complete"),
            "controlled_sweep_last_event": controlled_sweep.get("last_event"),
            "partial_axes": partial_axes,
        },
        "scaleup_audit": {
            "head256_grpo_vs_te_only": head256_grpo,
            "head1024_pareto_vs_te_only": head1024_pareto,
            "head1024_pareto_vs_hardneg_v2": head1024_hardneg,
        },
        "curve_audit": {
            "budget_head256": budget256,
            "budget_head1024": budget1024,
            "length_head256": length256,
            "length_head1024": length1024,
        },
        "runtime_audit": {
            "head256_mo_grpo": runtime256,
            "head1024_mo_pareto": runtime1024,
        },
        "controlled_sweep_audit": controlled_sweep,
        "rows": rows,
    }


def build_paper_table5(
    project_root: str,
    run_mode: str = "development",
    artifact_paths: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    if normalize_run_mode(run_mode) == "paper":
        return paper_builder_gate("paper_table5_scale_law_readiness", project_root, artifact_paths, __file__)
    report = _build_paper_table5_development(project_root)
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
        fh.write("# Paper Table 5: Scale-Law Readiness\n\n")
        fh.write(f"- Claim policy: {report.get('claim_policy', CLAIM_POLICY)}\n")
        fh.write(
            f"- Ready for readiness draft: `{summary.get('table_ready_for_scale_law_readiness_draft')}`; "
            f"ready for true scale-law claim: `{summary.get('ready_for_true_scale_law_claim')}`; "
            f"yield contraction flag: `{summary.get('yield_contraction_flag')}`; "
            f"head1024 best signal: `{summary.get('head1024_vs_te_only_best_signal')}`\n"
        )
        fh.write(
            f"- Missing required scale-law axes: `{summary.get('missing_required_axes')}`; "
            f"incomplete required axes: `{summary.get('incomplete_required_axes')}`; "
            f"hard constraints exact-1: `{summary.get('hard_constraints_exact_1')}`\n"
        )
        fh.write(
            f"- Controlled sweep plan ready: `{summary.get('controlled_sweep_plan_ready')}`; "
            f"status: `{summary.get('controlled_sweep_status')}`; "
            f"complete: `{summary.get('controlled_sweep_complete')}`; "
            f"runs complete: `{summary.get('controlled_sweep_n_complete')}/"
            f"{summary.get('controlled_sweep_n_runs')}`; "
            f"last event: `{summary.get('controlled_sweep_last_event')}`\n\n"
        )
        fh.write("| Scale dimension | Evidence | Main result | Signal | Claim language |\n")
        fh.write("|---|---|---|---|---|\n")
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            fh.write(
                f"| {row.get('scale_dimension')} | {row.get('evidence')} | "
                f"{row.get('result')} | {row.get('signal')} | {row.get('claim_language')} |\n"
            )
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--out-json", default="docs/paper_table5_scale_law_readiness.json")
    parser.add_argument("--out-md", default="docs/paper_table5_scale_law_readiness.md")
    parser.add_argument("--run-mode", choices=("development", "paper"), default="development")
    parser.add_argument("--paper-artifact", action="append", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    project_root = os.path.abspath(args.project_root)
    report = build_paper_table5(project_root, args.run_mode, args.paper_artifact)
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
    "build_paper_table5",
    "write_report_json",
    "write_report_markdown",
    "main",
]
