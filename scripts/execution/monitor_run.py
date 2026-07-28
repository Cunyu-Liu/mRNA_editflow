#!/usr/bin/env python3
"""Single-pass, low-frequency health evaluator for a registered run."""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


def _non_finite_paths(value, prefix: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        found.append(prefix)
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(_non_finite_paths(item, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_non_finite_paths(item, f"{prefix}[{index}]"))
    return found


def scan_metrics(path: Path) -> dict[str, object]:
    invalid_json_lines: list[int] = []
    non_finite: list[dict[str, object]] = []
    lines = 0
    if not path.is_file():
        return {"exists": False, "lines": 0, "invalid_json_lines": [], "non_finite": []}
    for lines, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            invalid_json_lines.append(lines)
            continue
        for item in _non_finite_paths(record):
            non_finite.append({"line": lines, "path": item})
    return {
        "exists": True,
        "lines": lines,
        "invalid_json_lines": invalid_json_lines,
        "non_finite": non_finite,
    }


def evaluate(run_root: Path, now_epoch: float | None = None, stall_seconds: float = 900) -> dict:
    now_epoch = now_epoch if now_epoch is not None else datetime.now(timezone.utc).timestamp()
    metrics_path = run_root / "logs/metrics.jsonl"
    events_path = run_root / "logs/events.jsonl"
    metrics = scan_metrics(metrics_path)
    heartbeat_candidates = [p for p in (metrics_path, events_path) if p.exists()]
    heartbeat_age = (
        now_epoch - max(path.stat().st_mtime for path in heartbeat_candidates)
        if heartbeat_candidates
        else None
    )
    reasons: list[str] = []
    if metrics["invalid_json_lines"]:
        reasons.append("INVALID_METRICS_JSON")
    if metrics["non_finite"]:
        reasons.append("NON_FINITE_METRIC")
    if heartbeat_age is None or heartbeat_age > stall_seconds:
        reasons.append("STALL_HEARTBEAT")
    state = "SAFE_PAUSED" if reasons else "RUNNING"
    return {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root.resolve()),
        "metrics": metrics,
        "heartbeat_age_seconds": heartbeat_age,
        "stall_threshold_seconds": stall_seconds,
        "state_recommendation": state,
        "reasons": reasons,
        "automatic_process_termination": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stall-seconds", type=float, default=900)
    parser.add_argument("--write-event", action="store_true")
    args = parser.parse_args(argv)
    report = evaluate(args.run_root, stall_seconds=args.stall_seconds)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.write_event:
        events = args.run_root / "logs/events.jsonl"
        events.parent.mkdir(parents=True, exist_ok=True)
        with events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(report, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return 1 if report["state_recommendation"] == "SAFE_PAUSED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
