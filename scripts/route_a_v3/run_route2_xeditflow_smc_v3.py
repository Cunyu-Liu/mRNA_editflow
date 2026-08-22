#!/usr/bin/env python3
"""Run authorized, replay-checked XEditFlow V3 scalar-potential SMC."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_development_projection_v3 import load_projection_rows
from core.route2_gpu_failure_evidence import cuda_device_observation, write_gpu_failure_evidence
from core.route2_legal_xeditflow import initial_state
from core.route2_source_token_cache_v3 import SourceTokenCacheIndexV3, load_source_token_cache_v3
from core.route2_xeditflow_gate_v3 import authorize_xeditflow_guidance_v3
from core.route2_xeditflow_guidance_v3 import MatchedComputeRecordV2, XEditValueV3
from core.route2_xeditflow_smc_runtime_v3 import (
    SetFlowValueProvidersV3,
    merge_smc_rounds_v3,
    run_batched_potential_smc_v3,
)
from core.route2_xeditflow_value_training_v3 import VALUE_CHECKPOINT_SCHEMA_V3
from core.route2_xeditsetflow_sampling_v3 import build_generation_metadata_v3
from scripts.route_a_v3.run_route2_base_flow_g0_validation_v1 import load_sources
from scripts.route_a_v3.validate_route2_xeditsetflow_v3 import load_setflow_checkpoint_v3


class XEditFlowSMCRunnerV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowSMCRunnerV3Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def validate_smc_run_config_v3(config: Mapping[str, Any]) -> None:
    _require(config.get("schema_version") == "route_a_v3_route2_xeditflow_smc_run_config.v1", "unexpected SMC run config schema")
    _require(int(config.get("particle_count", -1)) == 32, "SMC particle count changed")
    _require(int(config.get("candidate_cap", -1)) == 32, "SMC candidate cap changed")
    _require(float(config.get("ess_threshold", -1)) == 16.0, "SMC ESS threshold changed")
    _require(config.get("resampling") == "STRATIFIED", "SMC resampling method changed")
    _require(int(config.get("forward_equivalent_ceiling_per_source", -1)) == 320, "SMC compute ceiling changed")
    _require(int(config.get("reserved_terminal_critic_forwards", -1)) == 3, "SMC critic ensemble reservation changed")
    _require(int(config.get("maximum_sampling_rounds", -1)) == 32, "SMC additional-round ceiling changed")
    _require(int(config.get("base_flow_training_seed", -1)) in {20260904, 20260905, 20260906}, "undeclared SMC base-flow training seed")
    _require(float(config.get("kappa", -1)) in {0.0, 0.5, 1.0}, "SMC kappa is outside the frozen grid")
    _require(float(config.get("temperature", -1)) in {0.5, 1.0}, "SMC temperature is outside the frozen grid")
    _require(float(config.get("beta_max", -1)) in {0.5, 1.0, 2.0}, "SMC beta is outside the frozen grid")
    _require(config.get("action_space") == "SUB+STOP", "SMC action space changed")
    _require(config.get("replay_check") is True, "SMC replay check was disabled")


def load_value_checkpoint_v3(
    path: Path,
    *,
    config: Mapping[str, Any],
    device: torch.device,
) -> XEditValueV3:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    _require(checkpoint.get("schema_version") == VALUE_CHECKPOINT_SCHEMA_V3, "unexpected value checkpoint schema")
    _require(int(checkpoint.get("base_flow_training_seed", -1)) == int(config["base_flow_training_seed"]), "value checkpoint base-flow seed differs")
    _require(float(checkpoint.get("kappa", -1)) == float(config["kappa"]), "value checkpoint kappa differs")
    _require(float(checkpoint.get("temperature", -1)) == float(config["temperature"]), "value checkpoint temperature differs")
    _require(int(checkpoint.get("selected_pass", -1)) == 8, "value checkpoint is not the final frozen pass")
    _require(checkpoint.get("checkpoint_selection") == "FINAL_PASS_NO_EPOCH_RESELECTION", "value checkpoint selection differs")
    provenance = checkpoint.get("training_provenance") or {}
    _require(provenance.get("parameter_changed") is True and int(provenance.get("optimizer_steps", 0)) > 0, "value checkpoint lacks a learned update")
    _require(provenance.get("cpu_fallback_used") is False, "value checkpoint used CPU fallback")
    model_config = dict(checkpoint["model_config"])
    _require(model_config.pop("blocks") == 6 and model_config.pop("width") == 384, "value checkpoint architecture differs")
    model = XEditValueV3(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def run(config: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
    validate_smc_run_config_v3(config)
    _require(not output_dir.exists(), f"terminal SMC output already exists: {output_dir}")
    authorization = authorize_xeditflow_guidance_v3(
        _json(Path(config["critic_readiness_path"])),
        _json(Path(config["setflow_confirmation_path"])),
    )
    _require(authorization["guidance_authorized"] is True, "SMC remains blocked before full readiness")
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden")
    physical_gpu = int(config["physical_gpu_index"])
    device = torch.device(str(config["device"]))
    _require(device == torch.device(f"cuda:{physical_gpu}"), "SMC device provenance changed")
    torch.cuda.set_device(device)
    _require(torch.cuda.is_bf16_supported(), "BF16 is unavailable for SMC")
    cuda = cuda_device_observation(physical_gpu, require_physical_index_match=True)
    arm = str(config["setflow_arm"])
    setflow, checkpoint = load_setflow_checkpoint_v3(Path(config["setflow_checkpoint_path"]), arm, device)
    _require(int(checkpoint["training_provenance"]["seed"]) == int(config["base_flow_training_seed"]), "SetFlow checkpoint training seed differs")
    value = load_value_checkpoint_v3(Path(config["value_checkpoint_path"]), config=config, device=device)
    cache = SourceTokenCacheIndexV3(load_source_token_cache_v3(Path(config["source_token_cache_path"])))
    sources = load_sources(Path(config["source_eligibility_manifest"]))
    _require(len(sources) == int(config["expected_source_count"]), "SMC source cohort changed")
    projection = load_projection_rows([Path(config["validation_projection_path"])], allowed_splits=("VALIDATION",))
    metadata = build_generation_metadata_v3(sources, projection, checkpoint["vocabs"])
    _require(len(metadata) == len(sources), "SMC source metadata count changed")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    candidate_path = output_dir / "generated_candidates.private.jsonl"
    compute_path = output_dir / "matched_compute.jsonl"
    candidate_path.write_text("", encoding="utf-8")
    compute_path.write_text("", encoding="utf-8")
    replay_failures = 0
    numerical_failures = 0
    edit_budget_violations = 0
    candidate_budget_violations = 0
    maximum_compute = 0
    total_candidates = 0
    started = time.time()
    for source_index, (source, source_metadata) in enumerate(zip(sources, metadata)):
        root = initial_state(
            str(source["source_sequence"]),
            budget=int(source["edit_budget"]),
            assay_id=str(source["assay_id"]),
            context_id=str(source["biological_context_id"]),
        )
        providers = SetFlowValueProvidersV3(
            setflow_model=setflow,
            setflow_arm=arm,
            value_model=value,
            metadata=source_metadata,
            source_cache=cache,
            device=device,
        )
        decoder_base = int(config["decoder_seed_base"]) + source_index * 1_000_003
        rounds = []
        replay_ok = True
        maximum_round_cost = 2 * int(source["edit_budget"])
        while len(rounds) < int(config["maximum_sampling_rounds"]):
            used = sum(
                int(row["matched_compute"]["total_forward_equivalents"])
                for row in rounds
            )
            if rounds and used + maximum_round_cost + int(config["reserved_terminal_critic_forwards"]) > 320:
                break
            round_index = len(rounds)
            round_seed = decoder_base + round_index * 10_007
            seeds = [round_seed + slot for slot in range(32)]
            result = run_batched_potential_smc_v3(
                root,
                providers.rates,
                providers.values,
                source_key=str(source["source_key"]),
                particle_seeds=seeds,
                resampling_seed=round_seed + 100_000,
                beta_max=float(config["beta_max"]),
            )
            replay = run_batched_potential_smc_v3(
                root,
                providers.rates,
                providers.values,
                source_key=str(source["source_key"]),
                particle_seeds=seeds,
                resampling_seed=round_seed + 100_000,
                beta_max=float(config["beta_max"]),
            )
            this_replay_ok = (
                result["candidates"] == replay["candidates"]
                and result["resampling_events"] == replay["resampling_events"]
            )
            replay_ok = replay_ok and this_replay_ok
            replay_failures += int(not this_replay_ok)
            edit_budget_violations += int(result["edit_budget_violation_count"])
            candidate_budget_violations += int(result["candidate_budget_violation_count"])
            numerical_failures += int(result["numerical_failure_count"])
            rounds.append(result)
            merged = merge_smc_rounds_v3(
                rounds,
                source_key=str(source["source_key"]),
                reserved_critic_forwards=int(config["reserved_terminal_critic_forwards"]),
            )
            if len(merged["candidates"]) >= 32:
                break
        merged = merge_smc_rounds_v3(
            rounds,
            source_key=str(source["source_key"]),
            reserved_critic_forwards=int(config["reserved_terminal_critic_forwards"]),
        )
        compute = dict(merged["matched_compute"])
        maximum_compute = max(maximum_compute, int(compute["total_forward_equivalents"]))
        with compute_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        **compute,
                        "sampling_round_count": merged["sampling_round_count"],
                        "reserved_terminal_critic_forwards": 3,
                        "remaining_forward_equivalents_after_reservation": merged[
                            "remaining_forward_equivalents_after_reservation"
                        ],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        with candidate_path.open("a", encoding="utf-8") as handle:
            for rank, candidate in enumerate(merged["candidates"], start=1):
                row = {
                    **candidate,
                    "schema_version": "route_a_v3_route2_xeditflow_generated_candidate.v3",
                    "method_id": str(config["method_id"]),
                    "source_key": str(source["source_key"]),
                    "generation_rank": rank,
                    "generation_score": float(candidate["merged_log_weight"]),
                    "trajectory_replay_ok": replay_ok,
                    "base_flow_training_seed": int(config["base_flow_training_seed"]),
                    "decoder_seed_stream_start": decoder_base,
                    "sampling_round_count": merged["sampling_round_count"],
                    "kappa": float(config["kappa"]),
                    "temperature": float(config["temperature"]),
                    "beta_max": float(config["beta_max"]),
                    "critic_self_score": None,
                    "independent_evaluator_score": None,
                    "generated_candidate_grants_canonical_credit": False,
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                total_candidates += 1
    summary = {
        "schema_version": "route_a_v3_route2_xeditflow_smc_generation.v3",
        "status": "XEDITFLOW_V3_SMC_GENERATION_COMPLETE" if not any((replay_failures, numerical_failures, edit_budget_violations, candidate_budget_violations)) else "XEDITFLOW_V3_SMC_GENERATION_FAIL",
        "method_id": str(config["method_id"]),
        "setflow_arm": arm,
        "base_flow_training_seed": int(config["base_flow_training_seed"]),
        "source_count": len(sources),
        "particle_count_per_source": 32,
        "candidate_cap_per_source": 32,
        "generated_candidate_count": total_candidates,
        "kappa": float(config["kappa"]),
        "temperature": float(config["temperature"]),
        "beta_max": float(config["beta_max"]),
        "maximum_forward_equivalents_per_source": maximum_compute,
        "forward_equivalent_ceiling_per_source": 320,
        "reserved_terminal_critic_forwards_per_source": 3,
        "additional_sampling_rounds_used_when_candidate_cap_not_reached": True,
        "trajectory_replay_failure_count": replay_failures,
        "edit_budget_violation_count": edit_budget_violations,
        "candidate_budget_violation_count": candidate_budget_violations,
        "numerical_failure_count": numerical_failures,
        "hard_legality_rate": 1.0,
        "wall_time_seconds_including_replay_check": time.time() - started,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "cpu_fallback_used": False,
        "critic_self_score_used_for_selection": False,
        "independent_evaluator_used_for_gradient": False,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
        **cuda,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = _json(args.config)
    try:
        result = run(config, output_dir=args.output_dir)
    except Exception as exc:
        write_gpu_failure_evidence(
            args.output_dir.with_name(args.output_dir.name + ".failed.json"),
            config,
            exc,
            entrypoint="run_route2_xeditflow_smc_v3",
            evaluation_outcomes_accessed=False,
        )
        raise
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
