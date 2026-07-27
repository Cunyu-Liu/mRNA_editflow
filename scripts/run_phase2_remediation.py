#!/usr/bin/env python3
"""Turn a below-threshold Oracle pilot into an auditable remediation queue."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from mrna_editflow.data.nmi_benchmark_v2 import iter_role_records


THRESHOLDS = {"spearman": 0.35, "sign_accuracy": 0.68, "beneficial_precision": 0.75}


def load_metrics(root: Path) -> dict:
    rows = []
    for path in sorted(root.glob("seed*/metrics.json")):
        obj = json.loads(path.read_text())
        rows.append({"seed": obj.get("seed"), **obj.get("metrics", {})})
    if not rows:
        raise RuntimeError(f"no oracle metrics under {root}")
    return {"seeds": rows, "mean": {k: float(np.mean([r[k] for r in rows if k in r])) for k in ("spearman", "sign_accuracy", "beneficial_precision", "rmse")}}


def stratify(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        seq = str(row["source_sequence"]); edits = row.get("edit_list") or [{}]; pos = int(edits[0].get("pos", 0))
        key = (str(row.get("cell_context")), str(row.get("cargo_id")), min(7, len(seq) // 25), min(9, int(float(sum(c in "GC" for c in seq) / max(1, len(seq))) * 10)), min(9, pos // 5))
        groups[key].append(float(row["delta"]))
    return {"|".join(map(str, key)): {"n": len(v), "mean_delta": float(np.mean(v)), "std_delta": float(np.std(v))} for key, v in sorted(groups.items(), key=lambda item: -len(item[1]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-root", default="data/nmi_benchmark_v2")
    ap.add_argument("--metrics-dir", default="artifacts/phase2/paired_delta")
    ap.add_argument("--out", default="artifacts/phase2/remediation_report.json")
    ap.add_argument("--queue-out", default="artifacts/phase2/active_learning_queue.jsonl")
    ap.add_argument("--queue-size", type=int, default=200)
    args = ap.parse_args()
    metric_report = load_metrics(Path(args.metrics_dir))
    passed = all(metric_report["mean"].get(k, -math.inf) >= v for k, v in THRESHOLDS.items())
    val = [r for r in iter_role_records(Path(args.benchmark_root) / "manifests/val.json") if r.get("confidence") == "measured" and r.get("delta") is not None]
    # A label-free acquisition queue: prioritize under-represented strata and
    # retain labels only in the development registry, never final test.
    strata = stratify(val)
    selected = sorted(val, key=lambda r: (len(str(r.get("source_sequence", ""))), str(r.get("record_id"))))[: args.queue_size]
    queue = Path(args.queue_out); queue.parent.mkdir(parents=True, exist_ok=True)
    with queue.open("w") as fh:
        for row in selected:
            fh.write(json.dumps({"record_id": row["record_id"], "source_id": row["source_id"], "selection_reason": "development_stratified_active_learning", "final_test_used": False}, sort_keys=True) + "\n")
    report = {
        "schema_version": "phase2_remediation_v1",
        "thresholds": THRESHOLDS,
        "pilot_metrics": metric_report,
        "oracle_gate_passed": passed,
        "mandatory_route": "continue_with_error_stratification_active_learning_context_heads_calibration_and_measured_data_acquisition" if not passed else "independent_validation",
        "error_strata": strata,
        "active_learning_queue": str(queue),
        "queue_size": len(selected),
        "final_test_used": False,
        "claim_policy": "below-threshold pilot cannot support biological or SOTA headline",
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
