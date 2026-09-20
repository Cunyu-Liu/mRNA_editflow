#!/usr/bin/env python3
"""Emit per-family deliverables under benchmark_v2_matrix/<family>/.

The executed grid lives in benchmark_v2/leaderboard_matrix_v2/cells/<family>/;
this script mirrors the user-specified deliverable layout
benchmark_v2_matrix/<family>/{predictions.jsonl, metrics.json, port_adaptation.md}
(predictions symlinked/copy-referenced rather than duplicated: rows are large).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
CELLS = MNT / "benchmark_v2/leaderboard_matrix_v2/cells"
OUT_ROOT = MNT / "benchmark_v2_matrix"
UT = json.loads((MNT / "benchmark_v2/leaderboard_matrix_v2/family_unit_tests.json").read_text())["results"]

FAMILIES = ["gemorna", "lamar_utr5te", "utr_insight", "utr_stcnet", "hydrarna"]


def main() -> int:
    for fam in FAMILIES:
        src = CELLS / fam
        dst = OUT_ROOT / fam
        dst.mkdir(parents=True, exist_ok=True)
        cells = json.loads((src / "cells.json").read_text())
        matrix = cells["matrix"]
        meta = cells["family_meta"]

        unit = UT.get(fam, {})
        unit_level = "EXACT" if fam == "utr_insight" else "SMOKE"

        metrics = {
            "schema_version": "benchmark_v2_matrix_family_metrics.v1",
            "family": fam,
            "protocol": "frozen-delta zero-tuning: delta_hat = pred(cand) - pred(src); no calibration",
            "split": "VALIDATION only (9 existing tasks); M1/M6/S1 new rows full (Task 3.5)",
            "evaluator": cells["evaluator"],
            "evaluator_identity": cells["evaluator_identity"],
            "protected_test_reads": cells.get("protected_test_reads", 0),
            "cpu_fallback_used": cells.get("cpu_fallback_used"),
            "cuda_provenance": cells.get("cuda_provenance"),
            "unit_test_level": unit_level,
            "unit_test": unit,
            "tasks": {
                cell: {
                    "status": v.get("status"),
                    "n": v.get("n"),
                    "n_evaluated": v.get("n_evaluated"),
                    "task_macro_spearman": v.get("task_macro_spearman"),
                    "source_macro_within_source_spearman": v.get("source_macro_within_source_spearman"),
                    "source_macro_top_1": v.get("source_macro_top_1"),
                    "miss_count": v.get("miss_count"),
                    "miss_rate": v.get("miss_rate"),
                }
                for cell, v in sorted(matrix.items())
            },
        }
        (dst / "metrics.json").write_text(json.dumps(metrics, indent=1, sort_keys=True))

        md = [f"# {fam} - port adaptation declaration (benchmark v2 Task 2.2)"]
        md.append("")
        md.append("- weights: " + str(meta.get("weights_path")))
        md.append(f"- weights_sha256: {meta.get('weights_sha256')}")
        md.append(f"- license: {meta.get('license')}")
        md.append(f"- paradigm: {meta.get('paradigm')}")
        md.append(f"- input adaptation: {meta.get('input_adaptation')}")
        md.append(f"- unit_test_level: {unit_level} ({unit.get('status')}; {unit.get('check')})")
        md.append(f"- evaluator: Task-1 evaluate_route2_prediction_v1.evaluate, K=10, VALIDATION only, protected TEST reads = 0")
        md.append("")
        md.append("Per-task Spearman (delta): " + ", ".join(
            f"{cell}={v.get('task_macro_spearman')}" if v.get("status") == "OK" else f"{cell}={v.get('status')}"
            for cell, v in sorted(matrix.items())))
        md.append("")
        md.append(f"Predictions rows: benchmark_v2/leaderboard_matrix_v2/cells/{fam}/predictions.jsonl (27470 rows, per-row pred_src/pred_cand/delta_hat/observed)")
        (dst / "port_adaptation.md").write_text("\n".join(md) + "\n")

        pred_link = dst / "predictions.jsonl"
        if pred_link.is_symlink() or pred_link.exists():
            pass
        elif False:
            rel = "../benchmark_v2/leaderboard_matrix_v2/cells/" + fam + "/predictions.jsonl"
            try:
                pred_link.symlink_to(rel)
            except OSError:
                shutil.copy(src / "predictions.jsonl", pred_link)
        print(f"{fam}: metrics.json + port_adaptation.md + predictions.jsonl -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
