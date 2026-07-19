"""Audit downstream evaluation readiness for Stage A scale-law sweeps.

The Stage A sweep is only training infrastructure evidence until completed
checkpoints are evaluated by proposal-ranking and T1-T7 style downstream
reports. This module keeps that post-training evidence gap explicit.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from typing import Mapping, Optional, Sequence


CLAIM_POLICY = (
    "Stage A downstream evaluation readiness is post-training governance "
    "evidence. Do not claim a true data/model/step scale law until all planned "
    "Stage A runs complete and downstream proposal-ranking, T1-T7 aggregate, "
    "and scale-law trend audits are generated."
)

DEFAULT_SWEEP_GLOB = "benchmark/stage_a_scalelaw*/status.json"
DOWNSTREAM_ROOT = "benchmark/stage_a_scalelaw_downstream"
AGGREGATE_REPORT = "docs/stage_a_scalelaw_downstream_eval_summary.json"
TREND_REPORT = "docs/stage_a_scalelaw_downstream_trend_audit.json"


def _path(project_root: str, rel_or_abs: str) -> str:
    return rel_or_abs if os.path.isabs(rel_or_abs) else os.path.join(project_root, rel_or_abs)


def _rel(project_root: str, path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        return os.path.relpath(path, project_root)
    except ValueError:
        return path


def _local_candidate(project_root: str, declared_path: object) -> Optional[str]:
    if not isinstance(declared_path, str) or not declared_path:
        return None
    if os.path.isabs(declared_path):
        if os.path.exists(declared_path):
            return declared_path
        marker = "/mrna_editflow/"
        if marker in declared_path:
            return os.path.join(project_root, declared_path.split(marker, 1)[1])
        return declared_path
    return os.path.join(project_root, declared_path)


def _load_json(path: str) -> Mapping[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _sha256(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _latest_status_path(project_root: str) -> Optional[str]:
    paths = glob.glob(_path(project_root, DEFAULT_SWEEP_GLOB))
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


def _file_audit(project_root: str, path: object) -> dict[str, object]:
    local = _local_candidate(project_root, path)
    exists = bool(local and os.path.exists(local))
    return {
        "declared_path": path if isinstance(path, str) else None,
        "local_path": _rel(project_root, local),
        "exists": exists,
        "sha256": _sha256(local) if exists else None,
    }


def _run_expected_paths(run_id: str) -> dict[str, str]:
    run_dir = os.path.join(DOWNSTREAM_ROOT, run_id)
    return {
        "proposal_ranking": os.path.join(run_dir, "proposal_ranking_t5.json"),
        "t1_t7_eval": os.path.join(run_dir, "t1_t7_eval_summary.json"),
        "runtime_audit": os.path.join(run_dir, "runtime_audit.json"),
    }


def _run_rows(project_root: str, status: Mapping[str, object]) -> list[dict[str, object]]:
    raw_runs = status.get("runs", [])
    if not isinstance(raw_runs, Sequence) or isinstance(raw_runs, (str, bytes)):
        raw_runs = []
    rows: list[dict[str, object]] = []
    for row in raw_runs:
        if not isinstance(row, Mapping):
            continue
        run_id = str(row.get("run_id") or "")
        expected = _run_expected_paths(run_id)
        downstream = {
            name: _file_audit(project_root, rel_path)
            for name, rel_path in expected.items()
        }
        checkpoint = _file_audit(
            project_root,
            row.get("paths", {}).get("checkpoint")
            if isinstance(row.get("paths"), Mapping)
            else None,
        )
        training_complete = str(row.get("status")) == "complete" and bool(
            row.get("checkpoint_exists") or checkpoint.get("exists")
        )
        downstream_ready = all(item.get("exists") for item in downstream.values())
        rows.append(
            {
                "run_id": run_id,
                "data_size": row.get("data_size"),
                "model_size": row.get("model_size"),
                "steps": row.get("steps"),
                "seed": row.get("seed"),
                "training_status": row.get("status"),
                "training_complete": training_complete,
                "checkpoint": checkpoint,
                "expected_downstream": downstream,
                "downstream_ready": downstream_ready,
                "status": (
                    "downstream_ready"
                    if downstream_ready and training_complete
                    else "blocked_on_training"
                    if not training_complete
                    else "downstream_missing"
                ),
            }
        )
    return rows


def build_stage_a_downstream_eval_readiness(project_root: str) -> dict[str, object]:
    project_root = os.path.abspath(project_root)
    status_path = _latest_status_path(project_root)
    if status_path is None:
        return {
            "artifact_kind": "stage_a_downstream_eval_readiness",
            "project_root": project_root,
            "claim_policy": CLAIM_POLICY,
            "summary": {
                "status": "missing_stage_a_sweep_status",
                "n_runs": 0,
                "n_training_complete": 0,
                "n_downstream_ready": 0,
                "aggregate_report_ready": False,
                "trend_report_ready": False,
                "ready_for_stage_a_downstream_eval_claim": False,
                "ready_for_true_scale_law_claim": False,
            },
            "rows": [],
        }
    status = _load_json(status_path)
    summary = status.get("summary", {}) if isinstance(status.get("summary"), Mapping) else {}
    rows = _run_rows(project_root, status)
    aggregate = _file_audit(project_root, AGGREGATE_REPORT)
    trend = _file_audit(project_root, TREND_REPORT)
    n_runs = len(rows)
    n_training_complete = sum(1 for row in rows if row.get("training_complete"))
    n_downstream_ready = sum(1 for row in rows if row.get("downstream_ready"))
    all_training_complete = n_runs > 0 and n_training_complete == n_runs
    all_downstream_ready = n_runs > 0 and n_downstream_ready == n_runs
    ready_for_downstream = bool(all_training_complete and all_downstream_ready and aggregate.get("exists"))
    if not all_training_complete:
        status_label = "blocked_on_stage_a_sweep"
    elif not all_downstream_ready:
        status_label = "downstream_eval_missing"
    elif not aggregate.get("exists"):
        status_label = "aggregate_downstream_report_missing"
    elif not trend.get("exists"):
        status_label = "trend_audit_missing"
    else:
        status_label = "downstream_eval_ready"
    return {
        "artifact_kind": "stage_a_downstream_eval_readiness",
        "project_root": project_root,
        "claim_policy": CLAIM_POLICY,
        "stage_a_status_path": _rel(project_root, status_path),
        "stage_a_status_sha256": _sha256(status_path),
        "stage_a_summary": summary,
        "summary": {
            "status": status_label,
            "n_runs": n_runs,
            "n_training_complete": n_training_complete,
            "n_downstream_ready": n_downstream_ready,
            "aggregate_report_ready": bool(aggregate.get("exists")),
            "trend_report_ready": bool(trend.get("exists")),
            "ready_for_stage_a_downstream_eval_claim": ready_for_downstream,
            "ready_for_true_scale_law_claim": False,
            "missing_or_incomplete": [
                name
                for name, ok in (
                    ("stage_a_training_complete", all_training_complete),
                    ("per_run_downstream_reports", all_downstream_ready),
                    ("aggregate_downstream_report", bool(aggregate.get("exists"))),
                    ("scale_law_trend_audit", bool(trend.get("exists"))),
                )
                if not ok
            ],
        },
        "aggregate_report": aggregate,
        "trend_report": trend,
        "rows": rows,
    }


def write_report_json(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return path


def write_report_markdown(report: Mapping[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    summary = report.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    rows = report.get("rows", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        rows = []
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Stage A Downstream Evaluation Readiness\n\n")
        fh.write(f"- Claim policy: {report.get('claim_policy', CLAIM_POLICY)}\n")
        fh.write(
            f"- Status: `{summary.get('status')}`; training complete: "
            f"`{summary.get('n_training_complete')}/{summary.get('n_runs')}`; "
            f"downstream ready: `{summary.get('n_downstream_ready')}/{summary.get('n_runs')}`\n"
        )
        fh.write(
            f"- Aggregate report ready: `{summary.get('aggregate_report_ready')}`; "
            f"trend report ready: `{summary.get('trend_report_ready')}`; "
            f"ready for true scale-law claim: `{summary.get('ready_for_true_scale_law_claim')}`\n"
        )
        fh.write(f"- Missing or incomplete: `{summary.get('missing_or_incomplete')}`\n\n")
        fh.write("| Run | Train | Downstream | Expected reports |\n")
        fh.write("|---|---:|---:|---|\n")
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            expected = row.get("expected_downstream", {})
            paths = []
            if isinstance(expected, Mapping):
                for name, item in expected.items():
                    if isinstance(item, Mapping):
                        paths.append(f"{name}:{item.get('local_path')}")
            fh.write(
                f"| `{row.get('run_id')}` | `{row.get('training_complete')}` | "
                f"`{row.get('downstream_ready')}` | `{'; '.join(paths)}` |\n"
            )
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--out-json", default="docs/stage_a_downstream_eval_readiness.json")
    parser.add_argument("--out-md", default="docs/stage_a_downstream_eval_readiness.md")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    project_root = os.path.abspath(args.project_root)
    report = build_stage_a_downstream_eval_readiness(project_root)
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
    "build_stage_a_downstream_eval_readiness",
    "write_report_json",
    "write_report_markdown",
    "main",
]
