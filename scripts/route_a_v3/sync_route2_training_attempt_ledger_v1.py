#!/usr/bin/env python3
"""Backfill/update the Route 2 training ledger from existing run directories."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_experiment_ledger import (
    build_training_attempt_row,
    canonical_dataset_ids,
    record_training_attempt,
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _running_details(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    included_studies = config.get("included_study_unit_ids") or canonical_dataset_ids(
        config.get("canonical_paths")
    )
    counts = Counter()
    manifest_path = Path(config["development_manifest"])
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("pool_assignment") != "DEVELOPMENT":
                continue
            if included_studies and row.get("study_unit_id") not in included_studies:
                continue
            counts[str(row["split"])] += 1
    stage = str(config.get("result_stage", ""))
    if stage in {"HPO_VALIDATION_ONLY", "FROZEN_DEVELOPMENT_VALIDATION"}:
        record_counts = {"TRAIN": counts["TRAIN"], "VALIDATION": counts["VALIDATION"]}
        withheld = counts["TEST"]
    elif stage == "FROZEN_DEVELOPMENT_TEST":
        record_counts = {"TRAIN": counts["TRAIN"] + counts["VALIDATION"], "TEST": counts["TEST"]}
        withheld = 0
    elif stage == "FINAL_ALL_DEVELOPMENT_REFIT":
        record_counts = {"TRAIN": sum(counts.values())}
        withheld = 0
    else:
        record_counts = dict(counts)
        withheld = 0
    return {
        "started_at": datetime.fromtimestamp(config_path.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
        "included_study_unit_ids": included_studies,
        "included_regions": config.get("included_regions") or ["5UTR", "3UTR"],
        "record_counts": record_counts,
        "development_test_record_count_withheld": withheld,
        "evaluation_record_count": 0,
    }


def sync_run(run_dir: Path, ledger_path: Path) -> dict[str, Any]:
    config_path = run_dir / "training_config.json"
    if not config_path.exists():
        raise ValueError(f"training_config.json is missing: {run_dir}")
    config = _load(config_path)
    summary_path = run_dir / "training_summary.json"
    failure_path = run_dir.with_name(run_dir.name + ".failed.json")
    if summary_path.exists():
        status = "COMPLETED"
        details = _load(summary_path)
    elif failure_path.exists():
        status = "FAILED"
        failure = _load(failure_path)
        details = {
            "error_type": failure.get("error_type", ""),
            "error": failure.get("error", failure.get("message", "")),
            "evaluation_record_count": 0,
        }
    else:
        status = "RUNNING"
        details = _running_details(config, config_path)
    row = build_training_attempt_row(
        config,
        run_dir,
        status,
        repository_root=REPO_ROOT,
        details=details,
    )
    record_training_attempt(ledger_path, run_dir / "training_attempt.json", row)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument("--runs-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    run_dirs = list(args.run_dir)
    for runs_root in args.runs_root:
        run_dirs.extend(path.parent for path in runs_root.rglob("training_config.json"))
    run_dirs = sorted(set(run_dirs))
    if not run_dirs:
        parser.error("at least one --run-dir or --runs-root is required")
    rows = [sync_run(path, args.ledger) for path in run_dirs]
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    print(json.dumps({"attempt_count": len(rows), "statuses": statuses}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
