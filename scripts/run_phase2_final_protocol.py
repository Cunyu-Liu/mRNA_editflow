#!/usr/bin/env python3
"""Run Phase 2 validation selection, pre-unblinding freeze, and final gates."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


ALIASES = (("test_id", "test_v2_untouched"), ("test_ood", "independent_assay"))


def select_validation_checkpoint(metrics_dir: Path) -> dict:
    runs = []
    for path in sorted(metrics_dir.glob("**/metrics.json")):
        obj = json.loads(path.read_text())
        if bool(obj.get("final_test_used", True)):
            continue
        checkpoint = Path(str(obj.get("checkpoint", "")))
        if not checkpoint.exists():
            continue
        metrics = obj.get("metrics", {})
        def val(name: str, default: float) -> float:
            value = metrics.get(name, default)
            return float(value) if math.isfinite(float(value)) else default
        runs.append({
            "path": str(path),
            "checkpoint": str(checkpoint),
            "seed": obj.get("seed"),
            "backbone": obj.get("backbone"),
            "recipe": obj.get("recipe"),
            "metrics": metrics,
            "sort_key": (
                val("spearman", -math.inf),
                val("sign_accuracy", -math.inf),
                val("beneficial_precision", -math.inf),
                -val("ece", math.inf),
                -val("rmse", math.inf),
                str(checkpoint),
            ),
        })
    if not runs:
        raise SystemExit(f"no validation-only checkpoints under {metrics_dir}")
    return max(runs, key=lambda row: row["sort_key"])


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", default="data/nmi_benchmark_v2")
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--foundation-leakage-audit", default=None)
    args = parser.parse_args()
    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = select_validation_checkpoint(metrics_dir)
    checkpoint = Path(selected["checkpoint"])
    config = json.loads(json.dumps(selected["metrics"]))  # keep summary JSON-native
    # Read the training configuration from the checkpoint without loading model
    # tensors; evaluator/selector need the exact sequence/edit window.
    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    train_cfg = payload.get("config", {})
    max_len = int(train_cfg.get("max_len", 256))
    max_edits = int(train_cfg.get("max_edits", 10))
    common = [
        "--benchmark-root", args.benchmark_root,
        "--checkpoint", str(checkpoint),
        "--device", args.device,
        "--batch-size", str(args.batch_size),
        "--max-len", str(max_len),
        "--max-edits", str(max_edits),
    ]
    reports = []
    selection_artifacts = []
    for role, alias in ALIASES:
        selection_path = out_dir / f"selection_{alias}.json"
        freeze_path = out_dir / f"freeze_{alias}.json"
        report_path = out_dir / f"final_{alias}.json"
        run([
            sys.executable, "scripts/select_phase2_candidates.py", *common,
            "--role", role, "--alias", alias, "--out", str(selection_path),
        ])
        freeze_command = [
            sys.executable, "scripts/freeze_phase2_candidate_manifest.py",
            "--benchmark-root", args.benchmark_root, "--role", role,
            "--alias", alias, "--selection-artifact", str(selection_path),
            "--out", str(freeze_path), "--attest-before-unblinding",
        ]
        run(freeze_command)
        eval_command = [
            sys.executable, "scripts/evaluate_phase2_oracle.py", *common,
            "--role", role, "--alias", alias,
            "--candidate-freeze-manifest", str(freeze_path),
            "--allow-final-labels", "--out-json", str(report_path),
        ]
        if args.foundation_leakage_audit:
            eval_command.extend(["--foundation-leakage-audit", args.foundation_leakage_audit])
        run(eval_command)
        reports.append(json.loads(report_path.read_text()))
        selection_artifacts.append(str(selection_path.resolve()))

    summary = {
        "schema_version": "phase2_final_protocol_v1",
        "validation_selection_rule": (
            "validation_spearman_desc, sign_accuracy_desc, beneficial_precision_desc, "
            "ece_asc, rmse_asc, checkpoint_path_asc"
        ),
        "selected_validation_run": selected,
        "selection_artifacts": selection_artifacts,
        "final_reports": [str((out_dir / f"final_{alias}.json").resolve()) for _, alias in ALIASES],
        "reports": reports,
        "final_test_used": True,
        "claim_policy": "only the reports after label-free freeze support final gate assessment",
    }
    (out_dir / "protocol_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
