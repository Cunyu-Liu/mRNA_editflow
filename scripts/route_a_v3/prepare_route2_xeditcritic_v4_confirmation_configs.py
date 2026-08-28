#!/usr/bin/env python3
"""Prepare exactly three matched V4-FULL/C0 Critic confirmation configs after PASS."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def build_critic_confirmation_configs_v4(
    base: Mapping[str, Any],
    protocol: Mapping[str, Any],
    screen_gate: Mapping[str, Any],
    *,
    confirmation_runner_git_head: str,
) -> list[dict[str, Any]]:
    _require(
        re.fullmatch(r"[0-9a-f]{40}", confirmation_runner_git_head) is not None,
        "Critic V4 confirmation runner Git HEAD is invalid",
    )
    _require(
        base.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_screen_config.v1",
        "unexpected Critic V4 screen config",
    )
    _require(
        protocol.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_confirmation_protocol.v1"
        and protocol.get("status")
        == "FROZEN_PROSPECTIVE_BEFORE_SCREEN_OR_CONFIRMATION_RESULT",
        "Critic V4 confirmation protocol is not prospectively frozen",
    )
    _require(
        screen_gate.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_screen_gate.v1"
        and screen_gate.get("status") == "XEDITCRITIC_V4_SCREEN_PASS"
        and screen_gate.get("passed") is True
        and screen_gate.get("confirmation_authorized") is True
        and screen_gate.get("development_test_authorized") is False,
        "Critic V4 screen does not authorize confirmation",
    )
    _require(
        int(screen_gate.get("development_test_outcome_reads", -1)) == 0
        and int(screen_gate.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "Critic V4 screen gate reports a protected read",
    )
    seeds = [int(seed) for seed in protocol["required_seeds"]]
    _require(
        seeds == [20260908, 20260909, 20260910] and len(set(seeds)) == 3,
        "Critic V4 confirmation seed cohort changed",
    )
    _require(
        protocol.get("selected_model_run_id") == "v4_full"
        and protocol.get("matched_baseline_run_id") == "c0_v4"
        and protocol.get("additional_seed_authorized") is False,
        "Critic V4 confirmation model, baseline, or seed scope changed",
    )
    run_root = Path(str(protocol["run_root"]))
    results = []
    for seed in seeds:
        seed_root = run_root / f"seed_{seed}"
        results.append(
            {
                **dict(base),
                "schema_version": "route_a_v3_route2_xeditcritic_v4_confirmation_runtime.v1",
                "status": "FROZEN_CONFIRMATION_CONFIG_NOT_STARTED",
                "run_stage": "CONFIRMATION",
                "training_seed": seed,
                "confirmation_runner_git_head": confirmation_runner_git_head,
                "required_confirmation_run_ids": ["v4_full", "c0_v4"],
                "bootstrap_seed": int(
                    protocol["bootstrap_seed_by_training_seed"][str(seed)]
                ),
                "screen_gate_path": str(protocol["screen_gate_path"]),
                "output_root": str(seed_root),
                "confirmation_gate_output": str(
                    protocol["confirmation_gate_output"]
                ),
                "required_confirmation_seeds": seeds,
                "additional_seed_authorized": False,
                "development_test_outcomes_accessed": False,
                "new_final_evaluation_outcomes_accessed": False,
            }
        )
    return results


def materialize_critic_confirmation_configs_v4(
    configs: list[dict[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    runner_heads = {
        str(config.get("confirmation_runner_git_head", ""))
        for config in configs
    }
    _require(
        len(runner_heads) == 1
        and re.fullmatch(r"[0-9a-f]{40}", next(iter(runner_heads)))
        is not None,
        "Critic V4 confirmation config runner heads differ",
    )
    config_root = Path(str(protocol["runtime_config_root"]))
    run_root = Path(str(protocol["run_root"]))
    staging = config_root.with_name(config_root.name + ".partial")
    _require(not config_root.exists(), f"Critic V4 config root exists: {config_root}")
    _require(not staging.exists(), f"Critic V4 partial config root exists: {staging}")
    _require(not run_root.exists(), f"Critic V4 confirmation root exists: {run_root}")
    staging.mkdir(parents=True)
    paths = []
    for config in configs:
        filename = f"seed_{config['training_seed']}.json"
        (staging / filename).write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        paths.append(str(config_root / filename))
    manifest = {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_confirmation_config_manifest.v1",
        "status": "THREE_MATCHED_CONFIRMATION_CONFIGS_PREPARED_NOT_STARTED",
        "required_run_ids": ["v4_full", "c0_v4"],
        "required_seeds": [config["training_seed"] for config in configs],
        "confirmation_runner_git_head": next(iter(runner_heads)),
        "config_paths": paths,
        "development_test_authorized": False,
        "guidance_authorized": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(staging, config_root)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--screen-gate", required=True, type=Path)
    arguments = parser.parse_args()
    base = json.loads(arguments.base_config.read_text(encoding="utf-8"))
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    screen_gate = json.loads(arguments.screen_gate.read_text(encoding="utf-8"))
    runner_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    configs = build_critic_confirmation_configs_v4(
        base,
        protocol,
        screen_gate,
        confirmation_runner_git_head=runner_head,
    )
    manifest = materialize_critic_confirmation_configs_v4(configs, protocol)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
