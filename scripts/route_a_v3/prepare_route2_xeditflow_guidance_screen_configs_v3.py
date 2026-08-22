#!/usr/bin/env python3
"""Prepare the exact six value jobs and eighteen frozen guidance combinations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditflow_gate_v3 import GUIDANCE_GRID_V3, authorize_xeditflow_guidance_v3
from core.route2_xeditflow_guidance_v3 import TERMINAL_CRITIC_FORWARD_RESERVATION_V3


class GuidanceScreenConfigV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GuidanceScreenConfigV3Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def combination_id_v3(kappa: float, temperature: float, beta_max: float | None = None) -> str:
    stem = f"k{kappa:g}_t{temperature:g}"
    return stem if beta_max is None else f"{stem}_b{beta_max:g}"


def build_guidance_screen_configs_v3(config: Mapping[str, Any]) -> dict[str, Any]:
    _require(config.get("schema_version") == "route_a_v3_route2_xeditflow_guidance_screen_prepare.v1", "unexpected guidance prepare schema")
    _require(int(config.get("base_flow_training_seed", -1)) == 20260904, "guidance screen seed changed")
    _require(str(config.get("setflow_arm")) in {"f2", "f3"}, "guidance screen arm is not selectable")
    _require(int(config.get("physical_gpu_index", -1)) in set(range(6)), "guidance screen GPU is outside 0-5")
    _require(str(config.get("output_root", "")).startswith("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"), "guidance outputs left Route 2 /mnt")
    authorization = authorize_xeditflow_guidance_v3(
        _json(Path(config["critic_readiness_path"])),
        _json(Path(config["setflow_confirmation_path"])),
    )
    _require(authorization["guidance_authorized"] is True, "guidance configs remain blocked before readiness")
    _require(int(config.get("independent_evaluator_bootstrap_iterations", -1)) == 10_000, "guidance evaluator bootstrap count changed")
    output_root = Path(config["output_root"])
    gpu = int(config["physical_gpu_index"])
    value_jobs = []
    for kappa in (0.0, 0.5, 1.0):
        for temperature in (0.5, 1.0):
            value_id = combination_id_v3(kappa, temperature)
            target_output = output_root / "value_targets" / value_id
            value_output = output_root / "value_models" / value_id
            target_config = {
                "schema_version": "route_a_v3_route2_xeditflow_value_target_build_config.v1",
                "train_state_path": str(config["train_state_path"]),
                "frozen_rollout_score_path": str(config["frozen_rollout_score_path"]),
                "critic_readiness_path": str(config["critic_readiness_path"]),
                "setflow_confirmation_path": str(config["setflow_confirmation_path"]),
                "base_flow_training_seed": 20260904,
                "kappa": kappa,
                "temperature": temperature,
                "output_dir": str(target_output),
            }
            training_config = {
                "schema_version": "route_a_v3_route2_xeditflow_value_training_config.v1",
                "value_target_path": str(target_output / "value_targets.pt"),
                "critic_readiness_path": str(config["critic_readiness_path"]),
                "setflow_confirmation_path": str(config["setflow_confirmation_path"]),
                "source_token_cache_path": str(config["source_token_cache_path"]),
                "experiment_ledger_path": str(config["experiment_ledger_path"]),
                "base_flow_training_seed": 20260904,
                "kappa": kappa,
                "temperature": temperature,
                "passes": 8,
                "batch_size": 32,
                "precision": "BF16",
                "learning_rate": 3e-4,
                "weight_decay": 1e-4,
                "gradient_clip_norm": 1.0,
                "dropout": 0.10,
                "checkpoint_selection": "FINAL_PASS_NO_EPOCH_RESELECTION",
                "physical_gpu_index": gpu,
                "device": f"cuda:{gpu}",
                "output_dir": str(value_output),
            }
            value_jobs.append(
                {
                    "value_id": value_id,
                    "kappa": kappa,
                    "temperature": temperature,
                    "target_config": target_config,
                    "training_config": training_config,
                }
            )
    guidance_jobs = []
    for kappa, temperature, beta_max in GUIDANCE_GRID_V3:
        combination_id = combination_id_v3(kappa, temperature, beta_max)
        value_id = combination_id_v3(kappa, temperature)
        value_checkpoint = output_root / "value_models" / value_id / "value_checkpoint.pt"
        common = {
            "critic_readiness_path": str(config["critic_readiness_path"]),
            "setflow_confirmation_path": str(config["setflow_confirmation_path"]),
            "setflow_arm": str(config["setflow_arm"]),
            "setflow_checkpoint_path": str(config["setflow_checkpoint_path"]),
            "value_checkpoint_path": str(value_checkpoint),
            "source_token_cache_path": str(config["source_token_cache_path"]),
            "source_eligibility_manifest": str(config["source_eligibility_manifest"]),
            "validation_projection_path": str(config["validation_projection_path"]),
            "expected_source_count": 891,
            "base_flow_training_seed": 20260904,
            "kappa": kappa,
            "temperature": temperature,
            "beta_max": beta_max,
            "physical_gpu_index": gpu,
            "device": f"cuda:{gpu}",
            "method_id": f"xeditflow_v3_guidance_screen_{combination_id}",
        }
        smc_output = output_root / "guidance_screen" / combination_id / "open_smc"
        closed_output = output_root / "guidance_screen" / combination_id / "closed"
        raw_candidate_path = smc_output / "generated_candidates.private.jsonl"
        critic_score_output = smc_output / "critic_ensemble"
        candidate_path = critic_score_output / "critic_scored_candidates.private.jsonl"
        evaluator_scored_path = smc_output / "independent_evaluator_scored_candidates.private.jsonl"
        smc_config = {
            **common,
            "schema_version": "route_a_v3_route2_xeditflow_smc_run_config.v1",
            "particle_count": 32,
            "candidate_cap": 32,
            "ess_threshold": 16.0,
            "resampling": "STRATIFIED",
            "forward_equivalent_ceiling_per_source": 320,
            "reserved_terminal_critic_forwards": TERMINAL_CRITIC_FORWARD_RESERVATION_V3,
            "maximum_sampling_rounds": 32,
            "action_space": "SUB+STOP",
            "replay_check": True,
            "decoder_seed_base": int(config["decoder_seed_base"]),
            "output_dir": str(smc_output),
        }
        closed_config = {
            **common,
            "schema_version": "route_a_v3_route2_xeditflow_closed_neighborhood_config.v1",
            "potential_kind": "SOFT_VALUE",
            "measured_neighborhood_path": str(config["measured_neighborhood_path"]),
            "pool_assignment": "DEVELOPMENT",
            "split": "VALIDATION",
            "maximum_enumerated_edits": 5,
            "maximum_permutation_paths": 120,
            "enumeration": "ALL_EDIT_PERMUTATIONS_EXACT_SUM",
            "analysis_unit": "SOURCE",
            "undefined_source_policy": "EXCLUDE_NOT_ZERO_FILL",
            "output_dir": str(closed_output),
        }
        critic_ensemble_config = {
            "schema_version": "route_a_v3_route2_xeditflow_critic_ensemble_score_config.v1",
            "critic_readiness_path": str(config["critic_readiness_path"]),
            "setflow_confirmation_path": str(config["setflow_confirmation_path"]),
            "critic_refit_manifest_path": str(config["critic_refit_manifest_path"]),
            "mrnabert_model_path": str(config["mrnabert_model_path"]),
            "source_eligibility_manifest": str(config["source_eligibility_manifest"]),
            "validation_projection_path": str(config["validation_projection_path"]),
            "candidate_path": str(raw_candidate_path),
            "method_id": common["method_id"],
            "base_flow_training_seed": 20260904,
            "kappa": kappa,
            "critic_batch_size": 256,
            "critic_online_microbatch_size": 4,
            "physical_gpu_index": gpu,
            "device": f"cuda:{gpu}",
            "output_dir": str(critic_score_output),
        }
        open_metric_config = {
            "schema_version": "route_a_v3_route2_xeditflow_open_generation_config.v1",
            "pool_assignment": "DEVELOPMENT",
            "candidate_support_mode": "OPEN_GENERATED_SUPPORT",
            "undefined_outcome_policy": "UNKNOWN_NOT_ZERO",
            "source_eligibility_manifest": str(config["source_eligibility_manifest"]),
            "candidate_path": str(candidate_path),
            "measured_neighborhood_path": str(config["measured_neighborhood_path"]),
            "measured_top_k": 10,
        }
        evaluator_config = {
            "schema_version": "route_a_v3_route2_generation_independent_evaluator_job.v1",
            "method_id": common["method_id"],
            "evaluator_checkpoint_path": str(config["independent_evaluator_checkpoint_path"]),
            "guiding_checkpoint_path": str(config["guiding_checkpoint_path"]),
            "source_manifest_path": str(config["source_eligibility_manifest"]),
            "evaluator_frozen_before_candidate_generation": True,
            "evaluation_outcomes_used_to_select_evaluator": 0,
            "device": f"cuda:{gpu}",
            "physical_gpu_index": gpu,
            "candidate_path": str(candidate_path),
        }
        evaluator_comparison_config = {
            "schema_version": "route_a_v3_route2_xeditflow_independent_evaluator_comparison_config.v1",
            "strongest_baseline_path": str(config["strongest_generation_baseline_path"]),
            "baseline_selection_input_path": str(config["baseline_selection_input_path"]),
            "source_eligibility_manifest": str(config["source_eligibility_manifest"]),
            "guided_scored_candidate_path": str(evaluator_scored_path),
            "bootstrap_iterations": 10_000,
            "bootstrap_seed": int(config["decoder_seed_base"]) + len(guidance_jobs),
        }
        guidance_jobs.append(
            {
                "combination_id": combination_id,
                "combination": [kappa, temperature, beta_max],
                "smc_config": smc_config,
                "closed_config": closed_config,
                "critic_ensemble_config": critic_ensemble_config,
                "open_metric_config": open_metric_config,
                "independent_evaluator_config": evaluator_config,
                "independent_evaluator_comparison_config": evaluator_comparison_config,
                "independent_evaluator_scored_candidate_path": str(evaluator_scored_path),
                "open_generation_metric_path": str(smc_output / "generation_metrics.json"),
                "independent_evaluator_metric_path": str(smc_output / "independent_evaluator_metrics.json"),
            }
        )
    _require(len(value_jobs) == 6 and len(guidance_jobs) == 18, "guidance job count differs")
    return {
        "schema_version": "route_a_v3_route2_xeditflow_guidance_screen_manifest.v1",
        "status": "XEDITFLOW_V3_GUIDANCE_SCREEN_CONFIGS_PREPARED",
        "base_flow_training_seed": 20260904,
        "value_job_count": 6,
        "guidance_combination_count": 18,
        "value_jobs": value_jobs,
        "guidance_jobs": guidance_jobs,
        "additional_grid_combination_authorized": False,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def write_manifest_v3(payload: Mapping[str, Any], output_dir: Path) -> None:
    _require(not output_dir.exists(), f"guidance config output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    for job in payload["value_jobs"]:
        for kind in ("target", "training"):
            path = output_dir / f"value_{kind}_{job['value_id']}.json"
            path.write_text(json.dumps(job[f"{kind}_config"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for job in payload["guidance_jobs"]:
        for kind in (
            "smc",
            "closed",
            "critic_ensemble",
            "open_metric",
            "independent_evaluator",
            "independent_evaluator_comparison",
        ):
            path = output_dir / f"{kind}_{job['combination_id']}.json"
            path.write_text(json.dumps(job[f"{kind}_config"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = build_guidance_screen_configs_v3(_json(args.config))
    write_manifest_v3(payload, args.output_dir)
    print(json.dumps({key: value for key, value in payload.items() if key not in {"value_jobs", "guidance_jobs"}}, sort_keys=True))


if __name__ == "__main__":
    main()
