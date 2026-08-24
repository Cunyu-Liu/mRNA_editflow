#!/usr/bin/env python3
"""Build one gate-selected V4 value target for a non-screen final seed."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditflow_value_training_v4 import assemble_value_targets_v4


class XEditFlowFinalValueTargetV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowFinalValueTargetV4Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _require(
        bool(rows) and all(isinstance(row, dict) for row in rows),
        f"JSONL input is empty or invalid: {path}",
    )
    return rows


def validate_final_value_target_config_v4(
    config: Mapping[str, Any], guidance_gate: Mapping[str, Any]
) -> None:
    _require(
        config.get("schema_version")
        == "route_a_v3_route2_xeditflow_final_value_target_config.v4",
        "unexpected V4 final value target config",
    )
    seed = int(config.get("base_flow_training_seed", -1))
    _require(
        seed in {20260913, 20260914},
        "V4 final value target is only for non-screen frozen seeds",
    )
    _require(
        guidance_gate.get("schema_version")
        == "route_a_v3_route2_xeditflow_v4_guidance_screen_gate.v1"
        and guidance_gate.get("status") == "XEDITFLOW_V4_GUIDANCE_SCREEN_FROZEN"
        and int(guidance_gate.get("base_flow_training_seed", -1)) == 20260912
        and int(guidance_gate.get("combination_count", -1)) == 18,
        "V4 final value target requires the frozen screen gate",
    )
    _require(
        (
            float(config.get("kappa", -1)),
            float(config.get("temperature", -1)),
        )
        == (
            float(guidance_gate["selected_kappa"]),
            float(guidance_gate["selected_temperature"]),
        )
        and "beta_max" not in config,
        "V4 final value target differs from selected kappa/temperature",
    )
    _require(
        config.get("independent_evaluator_used") is False
        and config.get("development_test_outcomes_accessed_after_atomic_test")
        is False
        and config.get("new_final_evaluation_outcomes_accessed") is False,
        "V4 final value target accessed a protected input",
    )
    for field in (
        "train_state_path",
        "frozen_rollout_score_path",
        "rollout_summary_path",
        "critic_score_summary_path",
        "critic_readiness_path",
        "setflow_confirmation_path",
        "guidance_screen_gate_path",
        "output_dir",
    ):
        _require(
            str(config.get(field, "")).startswith(
                "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"
            ),
            f"V4 final value target {field} left Route 2 /mnt",
        )


def build(config: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
    guidance_gate = _json(Path(config["guidance_screen_gate_path"]))
    validate_final_value_target_config_v4(config, guidance_gate)
    _require(output_dir == Path(str(config["output_dir"])), "V4 output path differs")
    _require(not output_dir.exists(), f"terminal V4 final value target exists: {output_dir}")
    seed = int(config["base_flow_training_seed"])
    rollout_summary = _json(Path(config["rollout_summary_path"]))
    score_summary = _json(Path(config["critic_score_summary_path"]))
    _require(
        rollout_summary.get("status")
        == "XEDITFLOW_V4_VALUE_ROLLOUTS_COMPLETE_PENDING_CRITIC_SCORING"
        and int(rollout_summary.get("base_flow_training_seed", -1)) == seed
        and rollout_summary.get("fixed_seed_replayable") is True
        and int(rollout_summary.get("fixed_seed_replay_failure_count", -1)) == 0,
        "V4 final value target requires replay-checked seed rollouts",
    )
    _require(
        score_summary.get("status") == "XEDITFLOW_V4_VALUE_CRITIC_SCORING_COMPLETE"
        and int(score_summary.get("base_flow_training_seed", -1)) == seed
        and score_summary.get("critic_seeds") == [20260908, 20260909, 20260910]
        and score_summary.get("study_policy") == "UNKNOWN_STUDY_SCALE_FIXED_1"
        and score_summary.get("trajectory_mode_used_as_critic_input") is False,
        "V4 final value target requires exact frozen critic scores",
    )
    states = _jsonl(Path(config["train_state_path"]))
    rollouts = _jsonl(Path(config["frozen_rollout_score_path"]))
    expected = int(rollout_summary["terminal_rollout_count"])
    _require(
        len(states) == int(rollout_summary["state_mode_count"])
        and len(rollouts) == expected == len(states) * 8
        and int(score_summary.get("terminal_rollout_count", -1)) == expected,
        "V4 final value target input geometry differs",
    )
    payload = assemble_value_targets_v4(
        states,
        rollouts,
        critic_readiness=_json(Path(config["critic_readiness_path"])),
        setflow_confirmation=_json(Path(config["setflow_confirmation_path"])),
        base_flow_training_seed=seed,
        kappa=float(config["kappa"]),
        temperature=float(config["temperature"]),
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    target_path = output_dir / "value_targets.pt"
    partial = target_path.with_suffix(".pt.partial")
    torch.save(payload, partial)
    os.replace(partial, target_path)
    result = {key: value for key, value in payload.items() if key != "records"}
    result.update(
        {
            "schema_version": "route_a_v3_route2_xeditflow_final_value_target.v4",
            "status": "XEDITFLOW_V4_FINAL_VALUE_TARGET_COMPLETE",
            "value_target_path": str(target_path),
            "selected_by_guidance_screen_seed": 20260912,
            "beta_max_used_in_target": False,
            "raw_outcome_values_persisted": False,
            "independent_evaluator_used": False,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
    )
    (output_dir / "run_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    result = build(_json(arguments.config), output_dir=arguments.output_dir)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
