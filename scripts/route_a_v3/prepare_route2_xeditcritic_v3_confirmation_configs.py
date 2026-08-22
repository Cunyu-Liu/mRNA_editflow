#!/usr/bin/env python3
"""Prepare the exact three Critic V3 confirmation configs after screen PASS."""

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


def build_confirmation_configs(
    base: Mapping[str, Any],
    protocol: Mapping[str, Any],
    screen_gate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _require(
        base.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v3_screen_config.v1",
        "unexpected Critic V3 base screen config",
    )
    _require(
        protocol.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v3_confirmation_protocol.v1",
        "unexpected Critic V3 confirmation protocol",
    )
    _require(
        protocol.get("status")
        == "FROZEN_PROSPECTIVE_BEFORE_SCREEN_OR_CONFIRMATION_OUTCOME_READ",
        "Critic V3 confirmation protocol is not prospectively frozen",
    )
    _require(
        screen_gate.get("status") == "XEDITCRITIC_V3_SCREEN_PASS"
        and screen_gate.get("confirmation_authorized") is True,
        "Critic V3 screen does not authorize confirmation",
    )
    selected = str(screen_gate.get("selected_arm"))
    _require(
        selected in set(protocol["selectable_full_arms"]),
        "Critic V3 screen selected an unauthorized arm",
    )
    seeds = [int(seed) for seed in protocol["required_seeds"]]
    _require(
        seeds == [20260831, 20260901, 20260902] and len(set(seeds)) == 3,
        "Critic V3 confirmation seed cohort differs from the freeze",
    )
    _require(
        protocol.get("additional_seed_authorized") is False,
        "Critic V3 confirmation unexpectedly authorizes another seed",
    )
    for key, value in protocol["training_policy"].items():
        _require(base.get(key) == value, f"confirmation training policy differs: {key}")
    run_root = Path(str(protocol["run_root"])) / selected.lower()
    configs = []
    for seed in seeds:
        config = {
            **dict(base),
            "schema_version": "route_a_v3_route2_xeditcritic_v3_confirmation_runtime.v1",
            "status": "FROZEN_CONFIRMATION_CONFIG_NOT_STARTED",
            "run_stage": "CONFIRMATION",
            "seed": seed,
            "selected_arm": selected,
            "screen_gate_path": str(protocol["screen_gate_path"]),
            "output_root": str(run_root / f"seed{seed}"),
            "required_confirmation_seeds": seeds,
            "additional_seed_authorized": False,
            "development_test_outcomes_accessed": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        configs.append(config)
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--screen-gate", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    screen_gate = json.loads(args.screen_gate.read_text(encoding="utf-8"))
    configs = build_confirmation_configs(base, protocol, screen_gate)
    config_root = Path(str(protocol["runtime_config_root"]))
    run_root = Path(str(protocol["run_root"]))
    _require(not config_root.exists(), f"confirmation config root already exists: {config_root}")
    _require(not run_root.exists(), f"confirmation run root already exists: {run_root}")
    config_root.mkdir(parents=True)
    paths = []
    for config in configs:
        path = config_root / f"seed{config['seed']}.json"
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths.append(str(path))
    manifest = {
        "schema_version": "route_a_v3_route2_xeditcritic_v3_confirmation_config_manifest.v1",
        "status": "THREE_CONFIRMATION_CONFIGS_PREPARED_NOT_STARTED",
        "selected_arm": configs[0]["selected_arm"],
        "required_seeds": [config["seed"] for config in configs],
        "config_paths": paths,
        "matched_baseline_arm": "C0",
        "development_test_authorized": False,
        "guidance_authorized": False,
    }
    (config_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
