"""Summarize a Stage A scale-law sweep directory.

The sweep launcher writes a plan before any training starts, then appends
progress events while waiting for load gates and launching runs. This summarizer
is read-only: it turns the current plan/progress/profile/checkpoint state into
JSON/Markdown so queued, running, failed, and completed states are auditable.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Mapping, Optional, Sequence


CLAIM_POLICY = (
    "Stage A scale-law sweep status is infrastructure evidence only. Do not "
    "claim a data/model/step scale law until planned runs complete and downstream "
    "proposal-ranking or T1-T7 audits are generated."
)


def _load_json(path: str) -> Mapping[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


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


def _tail(path: str, n: int = 8) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = [line.rstrip("\n") for line in fh]
    return lines[-max(0, int(n)) :]


def _profile_stats(path: str) -> dict[str, object]:
    rows = _read_jsonl(path)
    last = rows[-1] if rows else {}
    losses = [
        float(row["loss"])
        for row in rows
        if isinstance(row.get("loss"), (int, float))
    ]
    return {
        "exists": os.path.exists(path),
        "n_rows": len(rows),
        "last_step": last.get("step") if isinstance(last, Mapping) else None,
        "last_loss": last.get("loss") if isinstance(last, Mapping) else None,
        "best_loss": min(losses) if losses else None,
        "last_samples_per_s": (
            last.get("samples_per_s") if isinstance(last, Mapping) else None
        ),
        "last_event": last,
    }


def _run_status(run: Mapping[str, object]) -> dict[str, object]:
    profile_path = str(run.get("profile_path", ""))
    checkpoint_path = str(run.get("checkpoint_path", ""))
    log_path = str(run.get("log_path", ""))
    metadata_path = str(run.get("metadata_path", ""))
    steps = int(run.get("steps", 0)) if isinstance(run.get("steps"), int) else 0
    profile = _profile_stats(profile_path)
    checkpoint_exists = os.path.exists(checkpoint_path)
    metadata = _load_json(metadata_path) if os.path.exists(metadata_path) else {}
    n_rows = int(profile["n_rows"])
    if checkpoint_exists and n_rows >= steps and steps > 0:
        status = "complete"
    elif n_rows > 0:
        status = "running_or_partial"
    elif metadata.get("status") == "train_command_exited":
        status = "exited_without_profile"
    else:
        status = "queued"
    return {
        "run_id": run.get("run_id"),
        "data_size": run.get("data_size"),
        "model_size": run.get("model_size"),
        "steps": steps,
        "seed": run.get("seed"),
        "status": status,
        "checkpoint_exists": checkpoint_exists,
        "profile": profile,
        "metadata_exists": os.path.exists(metadata_path),
        "metadata_status": metadata.get("status") if isinstance(metadata, Mapping) else None,
        "log_exists": os.path.exists(log_path),
        "log_tail": _tail(log_path),
        "paths": {
            "checkpoint": checkpoint_path,
            "profile": profile_path,
            "metadata": metadata_path,
            "log": log_path,
        },
    }


def summarize_sweep(sweep_dir: str) -> dict[str, object]:
    sweep_dir = os.path.abspath(sweep_dir)
    plan_path = os.path.join(sweep_dir, "plan.json")
    progress_path = os.path.join(sweep_dir, "progress.jsonl")
    summary_path = os.path.join(sweep_dir, "summary.json")
    plan = _load_json(plan_path)
    progress = _read_jsonl(progress_path)
    previous_summary = _load_json(summary_path) if os.path.exists(summary_path) else {}
    runs_raw = plan.get("runs", [])
    runs = [
        _run_status(row)
        for row in runs_raw
        if isinstance(row, Mapping)
    ] if isinstance(runs_raw, Sequence) and not isinstance(runs_raw, (str, bytes)) else []
    status_counts: dict[str, int] = {}
    for row in runs:
        status = str(row.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    load_waits = [
        row for row in progress if isinstance(row, Mapping) and row.get("event") == "load_gate_wait"
    ]
    launch_events = [
        row for row in progress if isinstance(row, Mapping) and row.get("event") in {"launch", "launched"}
    ]
    last_event = progress[-1] if progress else {}
    complete = status_counts.get("complete", 0)
    n_runs = len(runs)
    return {
        "artifact_kind": "stage_a_scalelaw_sweep_status",
        "claim_policy": CLAIM_POLICY,
        "sweep_dir": sweep_dir,
        "plan_path": plan_path,
        "progress_path": progress_path,
        "previous_summary_path": summary_path,
        "previous_summary_exists": os.path.exists(summary_path),
        "plan": {
            "artifact_kind": plan.get("artifact_kind"),
            "axes": plan.get("axes", {}),
            "n_runs": plan.get("n_runs"),
            "source_records_sha256": plan.get("source_records_sha256"),
            "source_record_count": plan.get("source_record_count"),
        },
        "summary": {
            "n_runs": n_runs,
            "n_complete": complete,
            "n_incomplete": n_runs - complete,
            "status_counts": status_counts,
            "all_runs_complete": n_runs > 0 and complete == n_runs,
            "n_progress_events": len(progress),
            "n_load_gate_wait_events": len(load_waits),
            "n_launch_events": len(launch_events),
            "last_event": last_event.get("event") if isinstance(last_event, Mapping) else None,
            "last_loadavg": last_event.get("loadavg") if isinstance(last_event, Mapping) else None,
            "ready_for_scale_law_claim": False,
            "previous_summary_n_complete": (
                previous_summary.get("n_complete") if isinstance(previous_summary, Mapping) else None
            ),
        },
        "progress_tail": progress[-12:],
        "runs": runs,
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
    runs = report.get("runs", [])
    if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)):
        runs = []
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Stage A Scale-Law Sweep Status\n\n")
        fh.write(f"- Claim policy: {report.get('claim_policy', CLAIM_POLICY)}\n")
        fh.write(f"- Sweep dir: `{report.get('sweep_dir')}`\n")
        fh.write(
            f"- Complete/incomplete: `{summary.get('n_complete')}/"
            f"{summary.get('n_incomplete')}`; "
            f"last event: `{summary.get('last_event')}`; "
            f"last loadavg: `{summary.get('last_loadavg')}`\n"
        )
        fh.write(
            f"- Load-gate waits: `{summary.get('n_load_gate_wait_events')}`; "
            f"launch events: `{summary.get('n_launch_events')}`; "
            f"ready for scale-law claim: `{summary.get('ready_for_scale_law_claim')}`\n\n"
        )
        fh.write("| run | data | model | steps | status | profile rows | best loss | checkpoint |\n")
        fh.write("|---|---:|---|---:|---|---:|---:|---|\n")
        for row in runs:
            if not isinstance(row, Mapping):
                continue
            profile = row.get("profile", {})
            if not isinstance(profile, Mapping):
                profile = {}
            fh.write(
                f"| `{row.get('run_id')}` | {row.get('data_size')} | "
                f"{row.get('model_size')} | {row.get('steps')} | {row.get('status')} | "
                f"{profile.get('n_rows')} | {profile.get('best_loss')} | "
                f"`{row.get('checkpoint_exists')}` |\n"
            )
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    sweep_dir = os.path.abspath(args.sweep_dir)
    out_json = args.out_json or os.path.join(sweep_dir, "status.json")
    out_md = args.out_md or os.path.join(sweep_dir, "status.md")
    report = summarize_sweep(sweep_dir)
    write_report_json(report, out_json)
    write_report_markdown(report, out_md)
    print(json.dumps({"json_path": out_json, "markdown_path": out_md}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLAIM_POLICY",
    "summarize_sweep",
    "write_report_json",
    "write_report_markdown",
    "main",
]
