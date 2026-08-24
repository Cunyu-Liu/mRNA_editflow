#!/usr/bin/env python3
"""Compose the exact three terminal V4 seed-evidence rows for adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from core.route2_xeditflow_value_training_v4 import BASE_FLOW_SEEDS_V4
from scripts.route_a_v3.assemble_route2_xeditflow_final_seed_evidence_v4 import (
    METHODS_V4,
)


class XEditFlowFinalComparisonComposeV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowFinalComparisonComposeV4Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def compose_final_comparison_manifest_v4(
    config: Mapping[str, Any],
    guidance_gate: Mapping[str, Any],
    seed_rows: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    _require(
        config.get("schema_version")
        == "route_a_v3_route2_xeditflow_final_comparison_compose_config.v4",
        "unexpected V4 final comparison compose config",
    )
    _require(
        guidance_gate.get("schema_version")
        == "route_a_v3_route2_xeditflow_v4_guidance_screen_gate.v1"
        and guidance_gate.get("status") == "XEDITFLOW_V4_GUIDANCE_SCREEN_FROZEN"
        and int(guidance_gate.get("base_flow_training_seed", -1)) == 20260912
        and int(guidance_gate.get("combination_count", -1)) == 18,
        "V4 final comparison guidance screen is not frozen",
    )
    _require(
        set(seed_rows) == set(BASE_FLOW_SEEDS_V4),
        "V4 final comparison seed-row inventory differs",
    )
    rows = []
    for seed in BASE_FLOW_SEEDS_V4:
        row = dict(seed_rows[seed])
        _require(
            int(row.get("base_flow_training_seed", -1)) == seed
            and isinstance(row.get("methods"), Mapping)
            and set(row["methods"]) == METHODS_V4
            and str(row.get("paired_bootstrap_path", "")).startswith(
                "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"
            )
            and str(row.get("equal_wall_time_sensitivity_path", "")).startswith(
                "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"
            )
            and all(
                str(path).startswith(
                    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"
                )
                for path in row["methods"].values()
            ),
            f"V4 final comparison seed row differs: {seed}",
        )
        rows.append(row)
    _require(
        config.get("development_test_outcomes_accessed_after_atomic_test") is False
        and config.get("new_final_evaluation_outcomes_accessed") is False,
        "V4 final comparison compose config accessed protected outcomes",
    )
    return {
        "schema_version": "route_a_v3_route2_xeditflow_final_comparison_manifest.v4",
        "status": "XEDITFLOW_V4_FINAL_COMPARISON_RESULTS_COMPLETE",
        "guidance_screen_status": "XEDITFLOW_V4_GUIDANCE_SCREEN_FROZEN",
        "selected_combination": [
            float(guidance_gate["selected_kappa"]),
            float(guidance_gate["selected_temperature"]),
            float(guidance_gate["selected_beta_max"]),
        ],
        "seeds": rows,
        "additional_training_seed_authorized": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()
    config = _json(arguments.config)
    output = Path(str(config["output_path"]))
    _require(not output.exists(), f"V4 final comparison manifest exists: {output}")
    path_map = config.get("seed_manifest_row_paths")
    _require(
        isinstance(path_map, Mapping)
        and set(path_map) == {str(seed) for seed in BASE_FLOW_SEEDS_V4},
        "V4 final comparison seed-row path inventory differs",
    )
    result = compose_final_comparison_manifest_v4(
        config,
        _json(Path(str(config["guidance_screen_gate_path"]))),
        {
            seed: _json(Path(str(path_map[str(seed)])))
            for seed in BASE_FLOW_SEEDS_V4
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
