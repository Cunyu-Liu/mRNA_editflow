#!/usr/bin/env python3
"""Write structural evidence for the registered independent Phase 2 axis."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from mrna_editflow.data.nmi_benchmark_v2 import manifest_sha256
from mrna_editflow.train.train_paired_delta import file_sha256
from scripts.evaluate_phase2_oracle import INDEPENDENT_ASSAY_NAME, load_final_rows


def unique_counts(rows: list[dict], field: str) -> dict[str, int]:
    return dict(collections.Counter(str(row.get(field)) for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", default="data/nmi_benchmark_v2")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.benchmark_root)
    role = "test_ood"
    role_manifest_path = root / "manifests" / f"{role}.json"
    role_manifest = json.loads(role_manifest_path.read_text())
    records_path = root / str(role_manifest["records_path"])
    rows = load_final_rows(root, role, "independent_assay")
    report = {
        "schema_version": "phase2_independent_axis_audit_v1",
        "status": "structural_pass" if len(rows) == 703 else "blocked",
        "role": role,
        "selection_filter": f"assay={INDEPENDENT_ASSAY_NAME}",
        "record_count": len(rows),
        "local_delta_eligible": all(
            row.get("task_kind") == "local_delta"
            and row.get("data_layer") == "C_source_matched_intervention"
            and bool(row.get("local_delta_eligible"))
            for row in rows
        ),
        "unique_assay": unique_counts(rows, "assay"),
        "unique_cargo": unique_counts(rows, "cargo_id"),
        "unique_batch": unique_counts(rows, "batch"),
        "unique_cell_context": unique_counts(rows, "cell_context"),
        "role_manifest_sha256": manifest_sha256(role_manifest_path),
        "records_sha256": file_sha256(str(records_path)),
        "test_assay_role_is_not_local_delta_ground_truth": not bool(
            json.loads((root / "manifests" / "test_assay.json").read_text()).get("local_delta_ground_truth")
        ),
        "final_metrics_computed": False,
        "candidate_freeze_verified": False,
        "claim_policy": "structural axis evidence only; no biological or performance claim",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "structural_pass" or not report["local_delta_eligible"]:
        raise SystemExit("independent axis structural audit failed")


if __name__ == "__main__":
    main()
