#!/usr/bin/env python3
"""Prepare exactly three full-model SetFlow V4 confirmation configs after PASS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def build_confirmation_configs_v4(
    base: Mapping[str, Any],
    protocol: Mapping[str, Any],
    screen_gate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _require(
        base.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_screen_config.v1",
        "unexpected SetFlow V4 base screen config",
    )
    _require(
        protocol.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_confirmation_protocol.v1"
        and protocol.get("status")
        == "FROZEN_PROSPECTIVE_BEFORE_SCREEN_OR_CONFIRMATION_RESULT",
        "SetFlow V4 confirmation protocol is not prospectively frozen",
    )
    _require(
        screen_gate.get("status") == "XEDITSETFLOW_V4_SCREEN_PASS"
        and screen_gate.get("confirmation_authorized") is True,
        "SetFlow V4 screen does not authorize confirmation",
    )
    selected_checkpoint_pass = screen_gate.get("selected_checkpoint_pass")
    _require(
        isinstance(selected_checkpoint_pass, int)
        and not isinstance(selected_checkpoint_pass, bool)
        and selected_checkpoint_pass in {4, 6, 8, 10},
        "SetFlow V4 screen selected no valid checkpoint",
    )
    seeds = [int(seed) for seed in protocol["required_seeds"]]
    _require(
        seeds == [20260912, 20260913, 20260914] and len(set(seeds)) == 3,
        "SetFlow V4 confirmation seed cohort changed",
    )
    _require(
        protocol.get("selected_model") == "v4_full"
        and protocol.get("additional_seed_authorized") is False,
        "SetFlow V4 confirmation model or seed authorization changed",
    )
    policy = protocol["training_policy"]
    training = base["training"]
    _require(int(training["pass_count"]) == int(policy["passes"]) == 10, "confirmation passes changed")
    _require(training["saved_checkpoint_passes"] == policy["saved_checkpoint_passes"] == [4, 6, 8, 10], "confirmation checkpoints changed")
    _require(float(training["learning_rate"]) == float(policy["learning_rate"]), "confirmation learning rate changed")
    _require(float(training["weight_decay"]) == float(policy["weight_decay"]), "confirmation weight decay changed")
    _require(float(training["gradient_clip_norm"]) == float(policy["gradient_clip_norm"]), "confirmation clipping changed")
    _require(float(training["warmup_fraction"]) == float(policy["warmup_fraction"]), "confirmation warmup changed")
    _require(training["validation_generation_during_training"] is False, "confirmation active Validation generation was enabled")
    run_root = Path(str(protocol["run_root"]))
    result: list[dict[str, Any]] = []
    for seed in seeds:
        seed_root = run_root / f"seed_{seed}"
        result.append(
            {
                **dict(base),
                "schema_version": "route_a_v3_route2_xeditsetflow_v4_confirmation_runtime.v1",
                "status": "FROZEN_CONFIRMATION_CONFIG_NOT_STARTED",
                "run_stage": "CONFIRMATION",
                "training_seed": seed,
                "selected_model": "v4_full",
                "screen_gate_path": str(protocol["screen_gate_path"]),
                "screen_selected_checkpoint_pass": selected_checkpoint_pass,
                "output_root": str(seed_root),
                "validation_output_root": str(
                    seed_root / "outcome_free_validation_generation"
                ),
                "confirmation_gate_output": str(
                    protocol["confirmation_gate_output"]
                ),
                "required_confirmation_seeds": seeds,
                "additional_seed_authorized": False,
                "development_test_outcomes_accessed": False,
                "new_final_evaluation_outcomes_accessed": False,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--screen-gate", required=True, type=Path)
    arguments = parser.parse_args()
    base = json.loads(arguments.base_config.read_text(encoding="utf-8"))
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    screen_gate = json.loads(arguments.screen_gate.read_text(encoding="utf-8"))
    configs = build_confirmation_configs_v4(base, protocol, screen_gate)
    config_root = Path(str(protocol["runtime_config_root"]))
    run_root = Path(str(protocol["run_root"]))
    _require(not config_root.exists(), f"SetFlow V4 config root exists: {config_root}")
    _require(not run_root.exists(), f"SetFlow V4 confirmation root exists: {run_root}")
    config_root.mkdir(parents=True)
    paths = []
    for config in configs:
        path = config_root / f"seed_{config['training_seed']}.json"
        path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        paths.append(str(path))
    manifest = {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_confirmation_config_manifest.v1",
        "status": "THREE_CONFIRMATION_CONFIGS_PREPARED_NOT_STARTED",
        "selected_model": "v4_full",
        "required_seeds": [config["training_seed"] for config in configs],
        "config_paths": paths,
        "development_test_authorized": False,
        "guidance_authorized": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    (config_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
