#!/usr/bin/env python3
"""Build the exact six V4 kappa-by-temperature value-target packages."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditflow_value_training_v4 import assemble_value_targets_v4


class XEditFlowValueTargetBuilderV4Error(RuntimeError):
    pass


VALUE_TARGET_GRID_V4 = tuple(
    (kappa, temperature)
    for kappa in (0.0, 0.5, 1.0)
    for temperature in (0.5, 1.0)
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowValueTargetBuilderV4Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                _require(
                    isinstance(row, dict),
                    f"JSONL row is not an object: {path}",
                )
                rows.append(row)
    _require(bool(rows), f"JSONL input is empty: {path}")
    return rows


def _component(value: float) -> str:
    return str(float(value)).replace(".", "p")


def build(config: dict[str, Any], *, output_root: Path) -> dict[str, Any]:
    _require(
        config.get("schema_version")
        == "route_a_v3_route2_xeditflow_value_target_grid_build_config.v4",
        "unexpected V4 value target grid config schema",
    )
    _require(
        config.get("stage") == "GUIDANCE_SCREEN"
        and int(config.get("base_flow_training_seed", -1)) == 20260912,
        "V4 value target grid is not the frozen screen seed",
    )
    _require(
        tuple(tuple(float(value) for value in row) for row in config.get("grid", ()))
        == VALUE_TARGET_GRID_V4,
        "V4 value target grid changed",
    )
    _require(
        output_root == Path(str(config.get("output_root", ""))),
        "V4 value target output path differs from frozen config",
    )
    _require(
        not output_root.exists(),
        f"terminal V4 value target grid already exists: {output_root}",
    )
    rollout_summary = _json(Path(config["rollout_summary_path"]))
    score_summary = _json(Path(config["critic_score_summary_path"]))
    _require(
        rollout_summary.get("status")
        == "XEDITFLOW_V4_VALUE_ROLLOUTS_COMPLETE_PENDING_CRITIC_SCORING"
        and int(rollout_summary.get("base_flow_training_seed", -1)) == 20260912
        and rollout_summary.get("fixed_seed_replayable") is True
        and int(rollout_summary.get("fixed_seed_replay_failure_count", -1)) == 0,
        "V4 value target grid requires replay-checked screen rollouts",
    )
    _require(
        score_summary.get("status")
        == "XEDITFLOW_V4_VALUE_CRITIC_SCORING_COMPLETE"
        and score_summary.get("critic_seeds")
        == [20260908, 20260909, 20260910]
        and score_summary.get("study_policy")
        == "UNKNOWN_STUDY_SCALE_FIXED_1"
        and score_summary.get("trajectory_mode_used_as_critic_input") is False,
        "V4 value target grid requires exact frozen critic scores",
    )
    expected_count = int(rollout_summary["terminal_rollout_count"])
    _require(
        int(score_summary.get("terminal_rollout_count", -1)) == expected_count,
        "V4 value target rollout and critic-score counts differ",
    )
    states = _jsonl(Path(config["train_state_path"]))
    rollouts = _jsonl(Path(config["frozen_rollout_score_path"]))
    _require(
        len(states) == int(rollout_summary["state_mode_count"])
        and len(rollouts) == expected_count == len(states) * 8,
        "V4 value target input geometry differs",
    )
    critic_readiness = _json(Path(config["critic_readiness_path"]))
    setflow_confirmation = _json(Path(config["setflow_confirmation_path"]))
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()
    packages: list[dict[str, Any]] = []
    for kappa, temperature in VALUE_TARGET_GRID_V4:
        payload = assemble_value_targets_v4(
            states,
            rollouts,
            critic_readiness=critic_readiness,
            setflow_confirmation=setflow_confirmation,
            base_flow_training_seed=20260912,
            kappa=kappa,
            temperature=temperature,
        )
        directory = output_root / (
            f"kappa_{_component(kappa)}_temperature_{_component(temperature)}"
        )
        directory.mkdir()
        target_path = directory / "value_targets.pt"
        partial_target = target_path.with_suffix(".pt.partial")
        torch.save(payload, partial_target)
        os.replace(partial_target, target_path)
        summary = {key: value for key, value in payload.items() if key != "records"}
        summary.update(
            {
                "value_target_path": str(target_path),
                "raw_outcome_values_persisted": False,
                "trajectory_mode_used_as_critic_input": False,
            }
        )
        (directory / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        packages.append(
            {
                "kappa": kappa,
                "temperature": temperature,
                "value_target_path": str(target_path),
                "summary_path": str(directory / "summary.json"),
                "state_mode_count": int(payload["state_mode_count"]),
            }
        )
    result = {
        "schema_version": "route_a_v3_route2_xeditflow_value_target_grid.v4",
        "status": "XEDITFLOW_V4_SIX_VALUE_TARGET_PACKAGES_COMPLETE",
        "stage": "GUIDANCE_SCREEN",
        "base_flow_training_seed": 20260912,
        "package_count": len(packages),
        "packages": packages,
        "beta_max_used_in_target": False,
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    result = build(_json(arguments.config), output_root=arguments.output_root)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
