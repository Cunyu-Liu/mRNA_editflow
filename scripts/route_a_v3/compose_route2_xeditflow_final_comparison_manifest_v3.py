#!/usr/bin/env python3
"""Compose the exact three-seed final comparison manifest for adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.route_a_v3.adjudicate_route2_xeditflow_final_v3 import METHODS, SEEDS


class XEditFlowFinalManifestV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowFinalManifestV3Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def compose_final_comparison_manifest_v3(
    seed_rows: Sequence[Mapping[str, Any]],
    guidance_screen_gate: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        guidance_screen_gate.get("status") == "XEDITFLOW_V3_GUIDANCE_SCREEN_FROZEN",
        "final comparison guidance combination is not frozen",
    )
    by_seed = {}
    for row in seed_rows:
        seed = int(row.get("base_flow_training_seed", -1))
        _require(seed in SEEDS and seed not in by_seed, "final comparison seed row differs")
        _require(isinstance(row.get("methods"), Mapping) and set(row["methods"]) == METHODS, f"final seed method inventory differs: {seed}")
        _require(bool(str(row.get("paired_bootstrap_path", ""))), f"final seed bootstrap path is absent: {seed}")
        _require(
            bool(str(row.get("equal_wall_time_sensitivity_path", ""))),
            f"final seed equal-wall sensitivity path is absent: {seed}",
        )
        by_seed[seed] = dict(row)
    _require(tuple(sorted(by_seed)) == SEEDS, "final comparison requires exactly three seed rows")
    return {
        "schema_version": "route_a_v3_route2_xeditflow_final_comparison_manifest.v1",
        "status": "XEDITFLOW_V3_FINAL_COMPARISON_RESULTS_COMPLETE",
        "guidance_screen_status": "XEDITFLOW_V3_GUIDANCE_SCREEN_FROZEN",
        "selected_kappa": float(guidance_screen_gate["selected_kappa"]),
        "selected_temperature": float(guidance_screen_gate["selected_temperature"]),
        "selected_beta_max": float(guidance_screen_gate["selected_beta_max"]),
        "seeds": [by_seed[seed] for seed in SEEDS],
        "additional_seed_authorized": False,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guidance-screen-gate", type=Path, required=True)
    parser.add_argument("--seed-row", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"final comparison manifest exists: {args.output}")
    result = compose_final_comparison_manifest_v3(
        [_json(path) for path in args.seed_row],
        _json(args.guidance_screen_gate),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
