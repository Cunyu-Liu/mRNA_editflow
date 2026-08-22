#!/usr/bin/env python3
"""Compose the guidance-screen preparer input from terminal value rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditflow_gate_v3 import authorize_xeditflow_guidance_v3


class GuidanceScreenInputV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GuidanceScreenInputV3Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def build_guidance_screen_prepare_input_v3(
    protocol: Mapping[str, Any],
    critic_readiness: Mapping[str, Any],
    setflow_confirmation: Mapping[str, Any],
    critic_refit_manifest: Mapping[str, Any],
    setflow_runtime: Mapping[str, Any],
    value_rollout_summary: Mapping[str, Any],
    *,
    physical_gpu_index: int,
) -> dict[str, Any]:
    _require(
        protocol.get("schema_version")
        == "route_a_v3_route2_xeditflow_v3_guidance_protocol.v1",
        "unexpected XEditFlow V3 guidance protocol",
    )
    authorization = authorize_xeditflow_guidance_v3(
        critic_readiness, setflow_confirmation
    )
    _require(authorization["guidance_authorized"] is True, "guidance screen input remains blocked before readiness")
    checkpoints = {
        int(row["seed"]): str(row["checkpoint_path"])
        for row in critic_refit_manifest.get("checkpoints", ())
    }
    _require(
        critic_refit_manifest.get("status")
        == "XEDITCRITIC_V3_ALL_DEVELOPMENT_REFIT_COMPLETE"
        and tuple(sorted(checkpoints)) == (20260831, 20260901, 20260902),
        "guidance screen Critic refit manifest differs",
    )
    selected = str(setflow_confirmation.get("selected_arm"))
    _require(selected in {"f2", "f3"}, "guidance screen SetFlow arm differs")
    _require(
        int(setflow_runtime.get("seed", -1)) == 20260904
        and setflow_runtime.get("selected_arm") == selected,
        "guidance screen SetFlow runtime differs",
    )
    _require(
        value_rollout_summary.get("status")
        == "XEDITFLOW_V3_VALUE_ROLLOUTS_COMPLETE"
        and int(value_rollout_summary.get("base_flow_training_seed", -1)) == 20260904
        and value_rollout_summary.get("setflow_arm") == selected
        and int(value_rollout_summary.get("rollouts_per_state", -1)) == 8,
        "guidance screen value rollout artifact is incomplete",
    )
    _require(
        value_rollout_summary.get("study_neutral") is True
        and value_rollout_summary.get("independent_evaluator_used") is False
        and value_rollout_summary.get("development_test_outcomes_accessed") is False
        and value_rollout_summary.get("new_final_evaluation_outcomes_accessed") is False,
        "guidance screen value rollout provenance differs",
    )
    _require(int(physical_gpu_index) in set(range(6)), "guidance screen GPU is outside 0-5")
    return {
        "schema_version": "route_a_v3_route2_xeditflow_guidance_screen_prepare.v1",
        "base_flow_training_seed": 20260904,
        "setflow_arm": selected,
        "physical_gpu_index": int(physical_gpu_index),
        "output_root": str(protocol["guidance_screen_output_root"]),
        "critic_readiness_path": str(protocol["critic_readiness_path"]),
        "setflow_confirmation_path": str(protocol["setflow_confirmation_path"]),
        "train_state_path": str(value_rollout_summary["state_path"]),
        "frozen_rollout_score_path": str(value_rollout_summary["frozen_rollout_score_path"]),
        "source_token_cache_path": str(setflow_runtime["source_token_cache_path"]),
        "experiment_ledger_path": str(setflow_runtime["experiment_ledger_path"]),
        "setflow_checkpoint_path": str(Path(setflow_runtime["output_root"]) / selected / "best.pt"),
        "source_eligibility_manifest": str(setflow_runtime["source_eligibility_manifest"]),
        "validation_projection_path": str(setflow_runtime["validation_projection_path"]),
        "measured_neighborhood_path": str(setflow_runtime["measured_neighborhood_path"]),
        "decoder_seed_base": int(protocol["decoder_seed_base"]),
        "guiding_checkpoint_path": checkpoints[20260831],
        "critic_refit_manifest_path": str(protocol["critic_refit_manifest_path"]),
        "mrnabert_model_path": str(protocol["mrnabert_model_path"]),
        "independent_evaluator_checkpoint_path": str(protocol["independent_evaluator_checkpoint_path"]),
        "strongest_generation_baseline_path": str(protocol["strongest_generation_baseline_path"]),
        "baseline_selection_input_path": str(protocol["baseline_selection_input_path"]),
        "independent_evaluator_bootstrap_iterations": int(protocol["independent_evaluator_bootstrap_iterations"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--value-rollout-summary", type=Path, required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"guidance screen preparer input exists: {args.output}")
    protocol = _json(args.protocol)
    result = build_guidance_screen_prepare_input_v3(
        protocol,
        _json(Path(protocol["critic_readiness_path"])),
        _json(Path(protocol["setflow_confirmation_path"])),
        _json(Path(protocol["critic_refit_manifest_path"])),
        _json(Path(protocol["setflow_confirmation_runtime_config_path"])),
        _json(args.value_rollout_summary),
        physical_gpu_index=args.physical_gpu_index,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
