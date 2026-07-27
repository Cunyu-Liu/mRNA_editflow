#!/usr/bin/env python3
"""Create a fail-closed Phase 2 remediation and active-learning artifact."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from mrna_editflow.data.nmi_benchmark_v2 import iter_role_records


THRESHOLDS = {
    "spearman": 0.35,
    "sign_accuracy": 0.68,
    "top10_enrichment": 1.75,
    "beneficial_precision": 0.75,
    "ece": 0.10,
}


def load_metrics(root: Path) -> dict:
    rows = []
    for path in sorted(root.glob("**/metrics.json")):
        obj = json.loads(path.read_text())
        metrics = obj.get("metrics", {})
        rows.append({"seed": obj.get("seed"), "backbone": obj.get("backbone"), "recipe": obj.get("recipe"), **metrics})
    if not rows:
        raise RuntimeError(f"no oracle metrics under {root}")
    keys = ["spearman", "sign_accuracy", "top10_enrichment", "beneficial_precision", "ece", "rmse"]
    mean = {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys if any(key in row for row in rows)}
    return {"runs": rows, "mean": mean}


def stratify(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        if row.get("task_kind") != "local_delta" or row.get("data_layer") != "C_source_matched_intervention":
            continue
        sequence = str(row["source_sequence"])
        edits = row.get("edit_list") or [{}]
        position = int(edits[0].get("pos", 0))
        key = (
            str(row.get("cell_context")), str(row.get("cargo_id")),
            min(7, len(sequence) // 25),
            min(9, int(float(sum(c in "GC" for c in sequence) / max(1, len(sequence))) * 10)),
            min(9, position // 5),
        )
        groups[key].append(float(row["delta"]))
    return {
        "|".join(map(str, key)): {
            "n": len(values), "mean_delta": float(np.mean(values)), "std_delta": float(np.std(values)),
        }
        for key, values in sorted(groups.items(), key=lambda item: -len(item[1]))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", default="data/nmi_benchmark_v2")
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--queue-out", required=True)
    parser.add_argument("--queue-size", type=int, default=200)
    args = parser.parse_args()
    metric_report = load_metrics(Path(args.metrics_dir))
    passed = all(
        (metric_report["mean"].get(key, -math.inf) >= value if key != "ece" else metric_report["mean"].get(key, math.inf) <= value)
        for key, value in THRESHOLDS.items()
    )
    val = [
        row for row in iter_role_records(Path(args.benchmark_root) / "manifests/val.json")
        if row.get("confidence") == "measured" and row.get("delta") is not None
        and row.get("task_kind") == "local_delta" and row.get("data_layer") == "C_source_matched_intervention"
        and row.get("local_delta_eligible")
    ]
    strata = stratify(val)
    selected = sorted(val, key=lambda row: (len(str(row.get("source_sequence", ""))), str(row.get("record_id"))))[: args.queue_size]
    queue = Path(args.queue_out)
    queue.parent.mkdir(parents=True, exist_ok=True)
    with queue.open("w") as handle:
        for row in selected:
            handle.write(json.dumps({
                "record_id": row["record_id"], "source_id": row["source_id"],
                "cargo_id": row.get("cargo_id"), "cell_context": row.get("cell_context"),
                "assay": row.get("assay"), "selection_reason": "development_stratified_active_learning",
                "label_status": "requires_new_source_matched_measurement", "final_test_used": False,
            }, sort_keys=True) + "\n")
    report = {
        "schema_version": "phase2_reliable_local_delta_remediation_v2",
        "thresholds": THRESHOLDS, "pilot_metrics": metric_report,
        "oracle_gate_passed": passed,
        "mandatory_route": "independent_validation" if passed else "continue_with_error_stratification_active_learning_context_heads_assay_effects_hierarchical_calibration_and_measured_data_acquisition",
        "error_stratification": {"status": "complete", "strata": strata},
        "active_learning_candidate_acquisition": {"status": "queue_created_pending_new_measurement", "queue": str(queue), "n": len(selected)},
        "context_specific_heads": {"status": "model_contract_present_but_requires_new_context_labels"},
        "assay_specific_random_effects": {"status": "assay_projection_present_random_effect_fit_pending"},
        "hierarchical_bayesian_calibration": {"status": "blocked_until_independent_calibration_labels", "global_measured_calibration": "available_in_phase2_runner"},
        "source_matched_measured_data": {"status": "external_wet_lab_acquisition_required"},
        "narrow_predictable_domain": {"status": "pending_preregistered_error_stratification_decision"},
        "queue_size": len(selected), "final_test_used": False,
        "claim_policy": "below-threshold or proxy-only evidence cannot support biological or SOTA headline",
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
