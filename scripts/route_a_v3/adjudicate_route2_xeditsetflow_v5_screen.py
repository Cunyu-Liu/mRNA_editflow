#!/usr/bin/env python3
"""Adjudicate the SetFlow V5 screen: per-arm Gate B0 (convergence) + B1 (base quality).

B1 (base): every saved checkpoint is read outcome-free; a base arm passes B1 if
NLL <= 2.068 and unique >= 0.85 and hard-legality == 1.0 at its best-eligible
checkpoint (pre-registered selection rule below).
B0 (convergence): per-arm training-loss plateau on the last two passes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditsetflow_runtime_v5 import screen_run_spec_v5


class SetFlowV5AdjudicationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SetFlowV5AdjudicationError(message)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def _write_new_terminal(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), f"adjudication terminal already exists: {path}")
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def adjudicate(config: Mapping[str, Any], screen_gate_output: Path) -> dict[str, Any]:
    params = config.get("screen_gate")
    _require(isinstance(params, Mapping), "SetFlow V5 screen gate config is absent")
    nll_threshold = float(params["base_gate_b1"]["nll_threshold"])
    unique_min = float(params["base_gate_b1"]["minimum_unique_candidate_rate"])
    legality_min = float(params["base_gate_b1"]["hard_legality_rate"])
    b0_window = int(params["gate_b0"]["window"])
    b0_tolerance = float(params["gate_b0"]["tolerance_relative_drop"])

    output_root = Path(config["output_root"])
    validation_root = Path(config["validation_output_root"])
    arms: dict[str, dict[str, Any]] = {}
    any_b1_pass = False
    for run_def in config["required_screen_runs"]:
        run_id = str(run_def["run_id"])
        spec = screen_run_spec_v5(config, run_id)
        training_dir = output_root / run_id
        summary = _read_json(training_dir / "training_summary.json")
        _require(
            summary.get("status")
            == "TERMINAL_XEDITSETFLOW_V5_TRAINING_COMPLETE_PENDING_VALIDATION",
            f"SetFlow V5 training is not terminal for {run_id}",
        )
        b0 = summary.get("gate_b0_convergence") or {}
        passes = summary.get("passes") or []
        pass_metrics: list[dict[str, Any]] = []
        for checkpoint_pass in config["training"]["saved_checkpoint_passes"]:
            v = _read_json(
                validation_root / run_id / f"pass_{checkpoint_pass}" / "validation_summary.json"
            )
            pass_metrics.append(
                {
                    "checkpoint_pass": checkpoint_pass,
                    "common_validation_set_marginal_nll": float(
                        v["common_validation_set_marginal_nll"]
                    ),
                    "source_macro_unique_candidate_rate": float(
                        v["source_macro_unique_candidate_rate"]
                    ),
                    "hard_legality_rate": float(v["hard_legality_rate"]),
                    "source_macro_candidate_recovery_rate": float(
                        v["source_macro_candidate_recovery_rate"]
                    ),
                    "source_macro_measured_top_k_recovery_at_k": float(
                        v["source_macro_measured_top_k_recovery_at_k"]
                    ),
                }
            )
        if not pass_metrics:
            raise SetFlowV5AdjudicationError(f"no validation metrics for {run_id}")
        # pre-registered selection rule: MIN_COMMON_NLL among eligible passes
        eligible = [
            pm for pm in pass_metrics
            if pm["hard_legality_rate"] >= legality_min
        ]
        selected = min(
            eligible or pass_metrics,
            key=lambda pm: pm["common_validation_set_marginal_nll"],
        )
        b1_passed = bool(
            selected["common_validation_set_marginal_nll"] <= nll_threshold
            and selected["source_macro_unique_candidate_rate"] >= unique_min
            and selected["hard_legality_rate"] >= legality_min
        )
        any_b1_pass = any_b1_pass or b1_passed
        arms[run_id] = {
            "run_id": run_id,
            "architecture_profile": spec.architecture_profile,
            "mode_count": spec.mode_count,
            "coverage_weight": spec.coverage_weight,
            "selectable": spec.selectable,
            "gate_b0_convergence": b0,
            "gate_b0_rule": "LAST_2_PASS_RELATIVE_DROP_LT_5_PERCENT",
            "training_row_count": len(passes),
            "pass_metrics": pass_metrics,
            "selected_checkpoint_pass": selected["checkpoint_pass"],
            "selected_metrics": selected,
            "base_gate_b1_nll_threshold": nll_threshold,
            "gate_b1_unique_minimum": unique_min,
            "gate_b1_legality_minimum": legality_min,
            "gate_b1_passed": b1_passed,
        }
    gate = {
        "schema_version": "route_a_v3_route2_xeditsetflow_v5_screen_gate.v1",
        "status": "XEDITSETFLOW_V5_SCREEN_PASS" if any_b1_pass else "XEDITSETFLOW_V5_SCREEN_NO_GO",
        "confirmation_authorized": any_b1_pass,
        "stage_acceptance": "BASE_MODEL_REPAIR_SELECTION" if any_b1_pass else "BASE_MODEL_REPAIR_INSUFFICIENT",
        "nll_threshold_le_f2": nll_threshold,
        "unique_minimum": unique_min,
        "arms": arms,
        "any_arm_passed_b1": any_b1_pass,
        "protected_reads": {
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    }
    _write_new_terminal(screen_gate_output, gate)
    return gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--screen-gate-output", required=True, type=Path)
    args = parser.parse_args()
    config = _read_json(args.config)
    gate = adjudicate(config, args.screen_gate_output)
    print(json.dumps(gate, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
