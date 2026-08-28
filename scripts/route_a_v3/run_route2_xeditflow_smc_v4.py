#!/usr/bin/env python3
"""Run authorized, replay-checked, mode-fixed XEditFlow V4 SMC."""

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
from core.route2_gpu_failure_evidence import (
    cuda_device_observation,
    write_gpu_failure_evidence,
)
from core.route2_legal_xeditflow import initial_state
from core.route2_source_token_cache_v3 import (
    SourceTokenCacheIndexV3,
    load_source_token_cache_v3,
)
from core.route2_xeditflow_equal_wall_time_v3 import EQUAL_WALL_TIME_SCOPE_V3
from core.route2_xeditflow_gate_v4 import (
    authorize_xeditflow_guidance_v4,
    require_selected_guidance_combination_v4,
)
from core.route2_xeditflow_smc_runtime_v4 import (
    SetFlowModeValueProvidersV4,
    combine_primary_and_replay_compute_v4,
    merge_smc_rounds_v4,
    run_batched_mode_fixed_potential_smc_v4,
)
from core.route2_xeditflow_value_training_v4 import (
    BASE_FLOW_SEEDS_V4,
    load_value_checkpoint_v4,
)
from core.route2_xeditsetflow_sampling_v3 import build_generation_metadata_v3
from core.route2_xeditsetflow_sampling_v4 import (
    root_mode_priors_v4,
    stratified_trajectory_mode_ids_v4,
)
from scripts.route_a_v3.generate_route2_xeditflow_value_rollouts_v4 import (
    _selected_checkpoint_pass_v4,
)
from scripts.route_a_v3.run_route2_base_flow_g0_validation_v1 import load_sources
from scripts.route_a_v3.validate_route2_xeditsetflow_v4_checkpoint import (
    load_checkpoint_v4,
)


class XEditFlowSMCRunnerV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowSMCRunnerV4Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def validate_smc_run_config_v4(config: Mapping[str, Any]) -> None:
    _require(
        config.get("schema_version")
        == "route_a_v3_route2_xeditflow_smc_run_config.v4",
        "unexpected V4 SMC run config schema",
    )
    _require(int(config.get("particle_count", -1)) == 32, "V4 SMC particle count changed")
    _require(int(config.get("candidate_cap", -1)) == 32, "V4 SMC candidate cap changed")
    _require(float(config.get("ess_threshold", -1)) == 16.0, "V4 SMC ESS threshold changed")
    _require(config.get("resampling") == "STRATIFIED", "V4 SMC resampling changed")
    _require(
        int(config.get("forward_equivalent_ceiling_per_source", -1)) == 320,
        "V4 SMC compute ceiling changed",
    )
    critic_forwards = list(config.get("terminal_critic_forwards_by_member", ()))
    _require(
        len(critic_forwards) == 3
        and all(int(value) in {1, 2, 4, 8} for value in critic_forwards),
        "V4 SMC terminal critic reservation changed",
    )
    _require(
        int(config.get("maximum_sampling_rounds", -1)) == 32,
        "V4 SMC additional-round ceiling changed",
    )
    _require(
        int(config.get("base_flow_training_seed", -1)) in BASE_FLOW_SEEDS_V4,
        "V4 SMC SetFlow seed changed",
    )
    _require(float(config.get("kappa", -1)) in {0.0, 0.5, 1.0}, "V4 SMC kappa differs")
    _require(float(config.get("temperature", -1)) in {0.5, 1.0}, "V4 SMC temperature differs")
    _require(float(config.get("beta_max", -1)) in {0.5, 1.0, 2.0}, "V4 SMC beta differs")
    _require(config.get("action_space") == "SUB+STOP", "V4 SMC action space changed")
    _require(config.get("replay_check") is True, "V4 SMC replay check was disabled")
    _require(
        int(config.get("decoder_seed_base", -1)) == 20261001,
        "V4 SMC decoder seed base changed",
    )
    _require(int(config.get("expected_source_count", -1)) == 891, "V4 SMC source count changed")
    physical_gpu = int(config.get("physical_gpu_index", -1))
    _require(physical_gpu in range(6), "V4 SMC GPU is outside 0-5")
    _require(
        str(config.get("device")) == f"cuda:{physical_gpu}",
        "V4 SMC device provenance changed",
    )
    _require(
        config.get("independent_evaluator_used") is False
        and config.get("development_test_outcomes_accessed_after_atomic_test") is False
        and config.get("new_final_evaluation_outcomes_accessed") is False,
        "V4 SMC config accessed a protected outcome or evaluator",
    )
    if config.get("method_id") == "full_soft_value_smc":
        _require(
            str(config.get("guidance_screen_gate_path", "")).startswith(
                "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"
            ),
            "V4 final SMC guidance screen path differs",
        )


def maximum_round_forward_equivalents_v4(edit_budget: int) -> int:
    """Worst-case primary plus replay compute for one 32-particle round."""

    _require(edit_budget in {1, 3, 5}, "V4 SMC edit budget differs")
    return 2 * edit_budget * (1 + 8 + 1)


def terminal_critic_forward_reservation_v4(
    refit_manifest: Mapping[str, Any],
) -> tuple[int, int, int]:
    """Derive actual per-member terminal batches from frozen physical batches."""

    refit_rows = sorted(
        refit_manifest.get("checkpoints", ()),
        key=lambda row: int(row["seed"]),
    )
    _require(
        refit_manifest.get("status")
        == "XEDITCRITIC_V4_ALL_DEVELOPMENT_REFIT_COMPLETE"
        and [int(row["seed"]) for row in refit_rows]
        == [20260908, 20260909, 20260910],
        "V4 SMC critic refit manifest differs",
    )
    physical_batches = [int(row.get("physical_batch_size", -1)) for row in refit_rows]
    _require(
        all(value in {4, 8, 16, 32} for value in physical_batches),
        "V4 SMC critic refit physical batch differs",
    )
    return tuple(math.ceil(32 / value) for value in physical_batches)


def run(config: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
    validate_smc_run_config_v4(config)
    if config.get("method_id") == "full_soft_value_smc":
        require_selected_guidance_combination_v4(
            _json(Path(config["guidance_screen_gate_path"])), config
        )
    _require(output_dir == Path(str(config["output_dir"])), "V4 SMC output differs from config")
    _require(not output_dir.exists(), f"terminal V4 SMC output exists: {output_dir}")
    critic_readiness = _json(Path(config["critic_readiness_path"]))
    setflow_confirmation = _json(Path(config["setflow_confirmation_path"]))
    authorization = authorize_xeditflow_guidance_v4(
        critic_readiness, setflow_confirmation
    )
    _require(
        authorization["guidance_authorized"] is True,
        "V4 SMC remains blocked before joint readiness",
    )
    refit_manifest = _json(Path(config["critic_refit_manifest_path"]))
    expected_critic_reservation = terminal_critic_forward_reservation_v4(
        refit_manifest
    )
    _require(
        expected_critic_reservation
        == tuple(int(value) for value in config["terminal_critic_forwards_by_member"]),
        "V4 SMC terminal critic reservation does not match physical batches",
    )
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    _require(
        not os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES remapping is forbidden",
    )
    physical_gpu = int(config["physical_gpu_index"])
    device = torch.device(str(config["device"]))
    torch.cuda.set_device(device)
    _require(torch.cuda.is_bf16_supported(), "BF16 is unavailable for V4 SMC")
    cuda = cuda_device_observation(physical_gpu, require_physical_index_match=True)

    seed = int(config["base_flow_training_seed"])
    checkpoint_pass = _selected_checkpoint_pass_v4(
        setflow_confirmation, seed=seed
    )
    runtime_config = _json(Path(config["setflow_runtime_config_path"]))
    setflow, checkpoint, _training_summary = load_checkpoint_v4(
        runtime_config,
        run_id="v4_full",
        checkpoint_pass=checkpoint_pass,
        device=device,
    )
    _require(int(checkpoint.get("seed", -1)) == seed, "V4 SMC SetFlow seed differs")
    value, value_training_provenance = load_value_checkpoint_v4(
        Path(config["value_checkpoint_path"]),
        base_flow_training_seed=seed,
        kappa=float(config["kappa"]),
        temperature=float(config["temperature"]),
        device=device,
    )
    source_cache = SourceTokenCacheIndexV3(
        load_source_token_cache_v3(Path(config["source_token_cache_path"]))
    )
    sources = load_sources(Path(config["source_eligibility_manifest"]))
    _require(len(sources) == int(config["expected_source_count"]), "V4 SMC cohort changed")
    validation = load_projection_rows(
        [Path(config["validation_projection_path"])],
        allowed_splits=("VALIDATION",),
    )
    metadata = build_generation_metadata_v3(sources, validation, checkpoint["vocabs"])
    _require(len(metadata) == len(sources), "V4 SMC source metadata count changed")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    (output_dir / "run_config.json").write_text(
        json.dumps(dict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    candidate_path = output_dir / "generated_candidates.private.jsonl"
    compute_path = output_dir / "matched_compute.jsonl"
    candidate_path.write_text("", encoding="utf-8")
    compute_path.write_text("", encoding="utf-8")

    totals = {
        "replay_failure_count": 0,
        "edit_budget_violation_count": 0,
        "candidate_budget_violation_count": 0,
        "numerical_failure_count": 0,
    }
    maximum_compute = 0
    total_candidates = 0
    run_peak_vram_mb = 0.0
    started = time.time()
    critic_reservation = tuple(
        int(value) for value in config["terminal_critic_forwards_by_member"]
    )
    for source_index, (source, source_metadata) in enumerate(
        zip(sources, metadata, strict=True)
    ):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        source_started = time.perf_counter()
        root = initial_state(
            str(source["source_sequence"]),
            budget=int(source["edit_budget"]),
            assay_id=str(source["assay_id"]),
            context_id=str(source["biological_context_id"]),
        )
        providers = SetFlowModeValueProvidersV4(
            setflow_model=setflow,
            value_model=value,
            metadata=source_metadata,
            source_cache=source_cache,
            device=device,
        )
        priors, prior_compute = root_mode_priors_v4(
            setflow,
            [root],
            [source_metadata],
            source_cache=source_cache,
            device=device,
            forward_batch_size=1,
        )
        mode_ids = stratified_trajectory_mode_ids_v4(priors[0])
        prior_trunk = int(prior_compute.trunk_forward_batch_count)
        prior_modes = int(prior_compute.mode_head_forward_state_count)
        decoder_base = int(config["decoder_seed_base"]) + source_index * 1_000_003
        rounds: list[dict[str, Any]] = []
        replay_ok = True
        worst_round = maximum_round_forward_equivalents_v4(int(source["edit_budget"]))
        while len(rounds) < int(config["maximum_sampling_rounds"]):
            used = sum(
                int(row["matched_compute"]["total_forward_equivalents"])
                for row in rounds
            )
            reserved = prior_trunk + prior_modes + sum(critic_reservation)
            if rounds and used + worst_round + reserved > 320:
                break
            round_index = len(rounds)
            round_seed = decoder_base + round_index * 10_007
            particle_seeds = tuple(round_seed + slot for slot in range(32))
            result = run_batched_mode_fixed_potential_smc_v4(
                root,
                providers.rates,
                providers.values,
                source_key=str(source["source_key"]),
                particle_mode_ids=mode_ids,
                particle_seeds=particle_seeds,
                resampling_seed=round_seed + 100_000,
                beta_max=float(config["beta_max"]),
            )
            replay = run_batched_mode_fixed_potential_smc_v4(
                root,
                providers.rates,
                providers.values,
                source_key=str(source["source_key"]),
                particle_mode_ids=mode_ids,
                particle_seeds=particle_seeds,
                resampling_seed=round_seed + 100_000,
                beta_max=float(config["beta_max"]),
            )
            this_replay_ok = (
                result["candidates"] == replay["candidates"]
                and result["resampling_events"] == replay["resampling_events"]
            )
            replay_ok = replay_ok and this_replay_ok
            result = dict(result)
            result["matched_compute"] = combine_primary_and_replay_compute_v4(
                result["matched_compute"],
                replay["matched_compute"],
                replay_ok=this_replay_ok,
            )
            rounds.append(result)
            merged = merge_smc_rounds_v4(
                rounds,
                source_key=str(source["source_key"]),
                prior_trunk_forwards=prior_trunk,
                prior_mode_forwards=prior_modes,
                terminal_critic_forwards_by_member=critic_reservation,
            )
            if len(merged["candidates"]) >= 32:
                break
        merged = merge_smc_rounds_v4(
            rounds,
            source_key=str(source["source_key"]),
            prior_trunk_forwards=prior_trunk,
            prior_mode_forwards=prior_modes,
            terminal_critic_forwards_by_member=critic_reservation,
        )
        compute = dict(merged["matched_compute"])
        torch.cuda.synchronize(device)
        compute["source_equal_wall_time_seconds"] = time.perf_counter() - source_started
        compute["source_equal_wall_time_scope"] = EQUAL_WALL_TIME_SCOPE_V3
        compute["source_equal_wall_peak_vram_mb"] = (
            torch.cuda.max_memory_allocated(device) / 1024**2
        )
        compute["source_cuda_device_name"] = str(cuda["cuda_device_name"])
        compute["terminal_critic_forwards_are_reserved_pending_scoring"] = True
        run_peak_vram_mb = max(
            run_peak_vram_mb, float(compute["source_equal_wall_peak_vram_mb"])
        )
        maximum_compute = max(maximum_compute, int(compute["total_forward_equivalents"]))
        for key in totals:
            totals[key] += int(compute["failure_counters"][key])
        with compute_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(compute, sort_keys=True) + "\n")
        with candidate_path.open("a", encoding="utf-8") as handle:
            for rank, candidate in enumerate(merged["candidates"], start=1):
                handle.write(
                    json.dumps(
                        {
                            **candidate,
                            "schema_version": "route_a_v3_route2_xeditflow_generated_candidate.v4",
                            "method_id": str(config["method_id"]),
                            "source_key": str(source["source_key"]),
                            "generation_rank": rank,
                            "generation_score": float(candidate["merged_log_weight"]),
                            "trajectory_replay_ok": replay_ok,
                            "base_flow_training_seed": seed,
                            "decoder_seed_stream_start": decoder_base,
                            "sampling_round_count": merged["sampling_round_count"],
                            "kappa": float(config["kappa"]),
                            "temperature": float(config["temperature"]),
                            "beta_max": float(config["beta_max"]),
                            "trajectory_mode_fixed": True,
                            "critic_self_score": None,
                            "independent_evaluator_score": None,
                            "generated_candidate_grants_canonical_credit": False,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                total_candidates += 1
    failed = any(totals.values())
    summary = {
        "schema_version": "route_a_v3_route2_xeditflow_smc_generation.v4",
        "status": (
            "XEDITFLOW_V4_SMC_GENERATION_FAIL"
            if failed
            else "XEDITFLOW_V4_SMC_GENERATION_COMPLETE_PENDING_TERMINAL_CRITIC_SCORING"
        ),
        "method_id": str(config["method_id"]),
        "base_flow_training_seed": seed,
        "value_checkpoint_path": str(config["value_checkpoint_path"]),
        "value_training_provenance": value_training_provenance,
        "source_count": len(sources),
        "particle_count_per_round": 32,
        "candidate_cap_per_source": 32,
        "generated_candidate_count": total_candidates,
        "kappa": float(config["kappa"]),
        "temperature": float(config["temperature"]),
        "beta_max": float(config["beta_max"]),
        "maximum_forward_equivalents_per_source_including_reserved_critic": maximum_compute,
        "forward_equivalent_ceiling_per_source": 320,
        "terminal_critic_forwards_reserved_by_member": list(critic_reservation),
        "terminal_critic_scoring_performed": False,
        "additional_sampling_rounds_used_only_within_remaining_compute": True,
        **totals,
        "hard_legality_rate": 1.0,
        "wall_time_seconds_including_replay_check": time.time() - started,
        "peak_vram_mb": run_peak_vram_mb,
        "equal_wall_time_scope": EQUAL_WALL_TIME_SCOPE_V3,
        "setflow_mode_is_fixed_trajectory_state": True,
        "free_action_ratio_head_used": False,
        "cpu_fallback_used": False,
        "critic_self_score_used_for_selection": False,
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
        **cuda,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    config = _json(arguments.config)
    try:
        result = run(config, output_dir=arguments.output_dir)
    except Exception as exc:
        write_gpu_failure_evidence(
            arguments.output_dir.with_name(arguments.output_dir.name + ".failed.json"),
            config,
            exc,
            entrypoint="run_route2_xeditflow_smc_v4",
            evaluation_outcomes_accessed=False,
        )
        raise
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
