#!/usr/bin/env python3
"""Run frozen, mode-fixed XEditFlow V4 matched controls on CUDA/BF16."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_development_projection_v3 import load_projection_rows
from core.route2_gpu_failure_evidence import (
    cuda_device_observation,
    write_gpu_failure_evidence,
)
from core.route2_legal_xeditflow import FlowState, initial_state
from core.route2_source_token_cache_v3 import (
    SourceTokenCacheIndexV3,
    load_source_token_cache_v3,
)
from core.route2_xeditcritic_training_data_v3 import UNKNOWN_CATEGORY
from core.route2_xeditflow_equal_wall_time_v3 import EQUAL_WALL_TIME_SCOPE_V3
from core.route2_xeditflow_gate_v4 import authorize_xeditflow_guidance_v4
from core.route2_xeditflow_guidance_v3 import uncertainty_penalized_reward_v3
from core.route2_xeditflow_matched_methods_v4 import (
    CriticRewardBatchV4,
    ExactCriticRewardPotentialV4,
    SourceAnchoredFirstOrderPotentialV4,
    ZeroCriticPotentialV4,
    run_mode_fixed_matched_control_smc_v4,
)
from core.route2_xeditflow_smc_runtime_v4 import (
    SetFlowModeRateProviderV4,
    combine_primary_and_replay_compute_v4,
    merge_smc_rounds_v4,
)
from core.route2_xeditflow_value_training_v4 import (
    BASE_FLOW_SEEDS_V4,
    CRITIC_SEEDS_V4,
)
from core.route2_xeditsetflow_sampling_v3 import build_generation_metadata_v3
from core.route2_xeditsetflow_sampling_v4 import (
    root_mode_priors_v4,
    stratified_trajectory_mode_ids_v4,
)
from scripts.route_a_v3.generate_route2_xeditflow_value_rollouts_v4 import (
    _selected_checkpoint_pass_v4,
)
from scripts.route_a_v3.route2_mrnabert_bottom_six_encoder_v4 import (
    FrozenMRNABERTBottomSixEncoderV4,
)
from scripts.route_a_v3.run_route2_base_flow_g0_validation_v1 import load_sources
from scripts.route_a_v3.run_route2_xeditflow_smc_v4 import (
    terminal_critic_forward_reservation_v4,
)
from scripts.route_a_v3.score_route2_xeditflow_candidates_v4 import (
    _representatives_v4,
)
from scripts.route_a_v3.score_route2_xeditflow_value_rollouts_v4 import (
    _ephemeral_cache_view_v4,
    _load_refit_models_v4,
    _score_member_batch_v4,
)
from scripts.route_a_v3.validate_route2_xeditsetflow_v4_checkpoint import (
    load_checkpoint_v4,
)


METHODS_V4 = {
    "unguided_setflow",
    "first_order_guidance",
    "simple_rate_guidance",
    "generate_then_rerank",
}


class XEditFlowMatchedControlsV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowMatchedControlsV4Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def validate_matched_control_config_v4(config: Mapping[str, Any]) -> None:
    _require(
        config.get("schema_version")
        == "route_a_v3_route2_xeditflow_matched_control_run_config.v4",
        "unexpected V4 matched-control config schema",
    )
    _require(str(config.get("method_id")) in METHODS_V4, "unknown V4 matched control")
    _require(
        int(config.get("base_flow_training_seed", -1)) in BASE_FLOW_SEEDS_V4,
        "V4 matched-control SetFlow seed changed",
    )
    _require(float(config.get("kappa", -1)) in {0.0, 0.5, 1.0}, "V4 kappa differs")
    _require(
        float(config.get("temperature", -1)) in {0.5, 1.0},
        "V4 temperature differs",
    )
    _require(
        float(config.get("beta_max", -1)) in {0.5, 1.0, 2.0},
        "V4 beta differs",
    )
    _require(int(config.get("particle_count", -1)) == 32, "V4 particle count changed")
    _require(int(config.get("candidate_cap", -1)) == 32, "V4 candidate cap changed")
    _require(float(config.get("ess_threshold", -1)) == 16.0, "V4 ESS threshold changed")
    _require(config.get("resampling") == "STRATIFIED", "V4 resampling changed")
    _require(
        int(config.get("forward_equivalent_ceiling_per_source", -1)) == 320,
        "V4 compute ceiling changed",
    )
    reservations = tuple(
        int(value) for value in config.get("terminal_critic_forwards_by_member", ())
    )
    _require(
        len(reservations) == 3 and all(value in {1, 2, 4, 8} for value in reservations),
        "V4 terminal critic reservation changed",
    )
    _require(
        int(config.get("maximum_sampling_rounds", -1)) == 32,
        "V4 sampling-round ceiling changed",
    )
    _require(
        config.get("action_space") == "SUB+STOP" and config.get("replay_check") is True,
        "V4 action-space or replay policy changed",
    )
    _require(
        int(config.get("decoder_seed_base", -1)) == 20261001,
        "V4 decoder seed stream changed",
    )
    _require(int(config.get("expected_source_count", -1)) == 891, "V4 source count changed")
    runtime_paths = config.get("critic_refit_runtime_config_paths")
    _require(
        isinstance(runtime_paths, Mapping)
        and set(runtime_paths) == {str(seed) for seed in CRITIC_SEEDS_V4},
        "V4 critic runtime inventory differs",
    )
    gpu = int(config.get("physical_gpu_index", -1))
    _require(
        gpu in range(6) and str(config.get("device")) == f"cuda:{gpu}",
        "V4 matched-control GPU provenance differs",
    )
    _require(
        str(config.get("output_dir", "")).startswith(
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"
        ),
        "V4 matched-control output left Route 2 /mnt",
    )
    _require(
        config.get("independent_evaluator_used") is False
        and config.get("development_test_outcomes_accessed_after_atomic_test") is False
        and config.get("new_final_evaluation_outcomes_accessed") is False,
        "V4 matched control accessed a protected input",
    )


def maximum_control_round_forward_equivalents_v4(
    edit_budget: int,
    *,
    method_id: str,
    critic_physical_batches: Sequence[int],
) -> int:
    """Conservative primary+replay bound for one complete 32-particle round."""

    _require(edit_budget in {1, 3, 5}, "V4 matched-control edit budget differs")
    _require(method_id in METHODS_V4, "unknown V4 matched control")
    physical = tuple(int(value) for value in critic_physical_batches)
    _require(
        len(physical) == 3 and all(value in {4, 8, 16, 32} for value in physical),
        "V4 critic physical batches differ",
    )
    setflow_primary_and_replay = 2 * edit_budget * (1 + 8)
    if method_id in {"unguided_setflow", "generate_then_rerank"}:
        return setflow_primary_and_replay
    critic_primary = sum(
        math.ceil(33 / batch) + (edit_budget - 1) * math.ceil(32 / batch)
        for batch in physical
    )
    return setflow_primary_and_replay + critic_primary


def _critic_projection_rows_for_states_v4(
    states: Sequence[FlowState],
    *,
    source: Mapping[str, Any],
    representative: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _require(bool(states), "V4 matched-control critic state batch is empty")
    source_sequence = str(source["source_sequence"])
    source_key = str(source["source_key"])
    region = str(source["region"]).replace("′", "").replace("'", "")
    _require(region in {"5UTR", "3UTR"}, "V4 critic source region differs")
    rows = []
    for index, state in enumerate(states):
        _require(
            state.source_sequence == source_sequence
            and state.assay_id == str(source["assay_id"])
            and state.context_id == str(source["biological_context_id"]),
            "V4 critic state crossed source identity",
        )
        rows.append(
            {
                "canonical_record_id": f"control-{source_key}-{index:03d}",
                "split": "VALIDATION",
                "task_id": str(representative["task_id"]),
                "study_unit_id": UNKNOWN_CATEGORY,
                "source_group_id": str(representative["source_group_id"]),
                "assay_id": str(source["assay_id"]),
                "biological_context_id": str(source["biological_context_id"]),
                "region_id": 0 if region == "5UTR" else 1,
                "endpoint_id": "GENERATED_V4_STUDY_NEUTRAL",
                "endpoint_descriptor": dict(representative["endpoint_descriptor"]),
                "source_sequence": source_sequence,
                "candidate_sequence": state.current_sequence,
                "source_relative_edits": [
                    {
                        "position": int(position),
                        "source_base": source_sequence[int(position)],
                        "candidate_base": str(alt),
                    }
                    for position, alt in state.source_relative_edits
                ],
                "direction_normalized_delta": 0.0,
                "dummy_target_for_inference_only": True,
                "development_test_outcomes_accessed_after_atomic_test": False,
                "new_final_evaluation_outcomes_accessed": False,
            }
        )
    return rows


class BoundFrozenCriticEnsembleRewardV4:
    def __init__(
        self,
        parent: "FrozenCriticEnsembleRewardV4",
        source: Mapping[str, Any],
        representative: Mapping[str, Any],
    ) -> None:
        self.parent = parent
        self.source = source
        self.representative = representative

    def __call__(self, states: Sequence[FlowState]) -> CriticRewardBatchV4:
        rows = _critic_projection_rows_for_states_v4(
            states,
            source=self.source,
            representative=self.representative,
        )
        cache_view = _ephemeral_cache_view_v4(rows, encoder=self.parent.bottom_encoder)
        predictions = []
        calls = []
        for seed in CRITIC_SEEDS_V4:
            values, member_calls = _score_member_batch_v4(
                rows,
                model=self.parent.models[seed],
                checkpoint=self.parent.checkpoints[seed],
                cache_view=cache_view,
                device=self.parent.device,
            )
            predictions.append(values)
            calls.append(member_calls)
        matrix = torch.tensor(predictions, dtype=torch.float32).T
        _require(
            matrix.shape == (len(states), 3)
            and bool(torch.isfinite(matrix).all().item()),
            "V4 matched-control critic prediction matrix differs",
        )
        rewards = uncertainty_penalized_reward_v3(matrix, kappa=self.parent.kappa)
        return CriticRewardBatchV4(
            tuple(float(value) for value in rewards.tolist()),
            tuple(int(value) for value in calls),
        )


class FrozenCriticEnsembleRewardV4:
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        device: torch.device,
        kappa: float,
    ) -> None:
        self.device = device
        self.kappa = float(kappa)
        self.models, self.checkpoints, _runtimes = _load_refit_models_v4(
            config, device=device
        )
        self.bottom_encoder = FrozenMRNABERTBottomSixEncoderV4(
            Path(config["mrnabert_model_path"]),
            device,
            maximum_sequences_per_batch=int(
                config["bottom_six_maximum_sequences_per_batch"]
            ),
            batch_token_budget=int(config["bottom_six_batch_token_budget"]),
            attention_backend=str(config["attention_backend"]),
        )

    def bind_source(
        self,
        source: Mapping[str, Any],
        representative: Mapping[str, Any],
    ) -> BoundFrozenCriticEnsembleRewardV4:
        return BoundFrozenCriticEnsembleRewardV4(self, source, representative)


def _require_selected_guidance_v4(
    gate: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    _require(
        gate.get("schema_version")
        == "route_a_v3_route2_xeditflow_v4_guidance_screen_gate.v1"
        and gate.get("status") == "XEDITFLOW_V4_GUIDANCE_SCREEN_FROZEN"
        and int(gate.get("base_flow_training_seed", -1)) == 20260912
        and int(gate.get("combination_count", -1)) == 18,
        "V4 matched control requires the frozen guidance screen",
    )
    _require(
        (
            float(config["kappa"]),
            float(config["temperature"]),
            float(config["beta_max"]),
        )
        == (
            float(gate["selected_kappa"]),
            float(gate["selected_temperature"]),
            float(gate["selected_beta_max"]),
        ),
        "V4 matched control differs from the selected guidance combination",
    )


def run(config: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
    validate_matched_control_config_v4(config)
    _require(output_dir == Path(str(config["output_dir"])), "V4 output path differs")
    _require(not output_dir.exists(), f"terminal V4 matched-control output exists: {output_dir}")
    critic_readiness = _json(Path(config["critic_readiness_path"]))
    setflow_confirmation = _json(Path(config["setflow_confirmation_path"]))
    _require(
        authorize_xeditflow_guidance_v4(
            critic_readiness, setflow_confirmation
        )["guidance_authorized"]
        is True,
        "V4 matched controls remain blocked before joint readiness",
    )
    _require_selected_guidance_v4(
        _json(Path(config["guidance_screen_gate_path"])), config
    )
    refit_manifest = _json(Path(config["critic_refit_manifest_path"]))
    critic_reservation = terminal_critic_forward_reservation_v4(refit_manifest)
    _require(
        critic_reservation
        == tuple(int(value) for value in config["terminal_critic_forwards_by_member"]),
        "V4 matched-control terminal reservation differs from refit batches",
    )
    refit_rows = sorted(refit_manifest["checkpoints"], key=lambda row: int(row["seed"]))
    critic_physical_batches = tuple(
        int(row["physical_batch_size"]) for row in refit_rows
    )

    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    _require(
        not os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES remapping is forbidden",
    )
    gpu = int(config["physical_gpu_index"])
    device = torch.device(str(config["device"]))
    torch.cuda.set_device(device)
    _require(torch.cuda.is_bf16_supported(), "BF16 is unavailable for V4 controls")
    cuda = cuda_device_observation(gpu, require_physical_index_match=True)

    seed = int(config["base_flow_training_seed"])
    selected_pass = _selected_checkpoint_pass_v4(setflow_confirmation, seed=seed)
    runtime_config = _json(Path(config["setflow_runtime_config_path"]))
    setflow, checkpoint, _training_summary = load_checkpoint_v4(
        runtime_config,
        run_id="v4_full",
        checkpoint_pass=selected_pass,
        device=device,
    )
    _require(int(checkpoint.get("seed", -1)) == seed, "V4 SetFlow seed differs")
    source_cache = SourceTokenCacheIndexV3(
        load_source_token_cache_v3(Path(config["source_token_cache_path"]))
    )
    sources = load_sources(Path(config["source_eligibility_manifest"]))
    _require(len(sources) == 891, "V4 matched-control cohort changed")
    validation = load_projection_rows(
        [Path(config["validation_projection_path"])],
        allowed_splits=("VALIDATION",),
    )
    metadata = build_generation_metadata_v3(sources, validation, checkpoint["vocabs"])
    representatives = _representatives_v4(sources, validation)
    _require(
        len(metadata) == len(sources) == len(representatives),
        "V4 matched-control metadata coverage differs",
    )
    method = str(config["method_id"])
    sampling_method = "unguided_setflow" if method == "generate_then_rerank" else method
    critic = (
        FrozenCriticEnsembleRewardV4(
            config, device=device, kappa=float(config["kappa"])
        )
        if sampling_method in {"first_order_guidance", "simple_rate_guidance"}
        else None
    )

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
        rates = SetFlowModeRateProviderV4(
            setflow_model=setflow,
            metadata=source_metadata,
            source_cache=source_cache,
            device=device,
        )
        if sampling_method == "first_order_guidance":
            assert critic is not None
            potential = SourceAnchoredFirstOrderPotentialV4(
                root,
                critic.bind_source(source, representatives[str(source["source_key"])]),
            )
        elif sampling_method == "simple_rate_guidance":
            assert critic is not None
            potential = ExactCriticRewardPotentialV4(
                critic.bind_source(source, representatives[str(source["source_key"])])
            )
        else:
            potential = ZeroCriticPotentialV4()
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
        worst_round = maximum_control_round_forward_equivalents_v4(
            int(source["edit_budget"]),
            method_id=method,
            critic_physical_batches=critic_physical_batches,
        )
        rounds: list[dict[str, Any]] = []
        replay_ok = True
        while len(rounds) < int(config["maximum_sampling_rounds"]):
            used = (
                prior_trunk
                + prior_modes
                + sum(critic_reservation)
                + sum(
                    int(row["matched_compute"]["total_forward_equivalents"])
                    for row in rounds
                )
            )
            _require(
                bool(rounds) or used + worst_round <= 320,
                "V4 first matched-control round cannot fit the compute ceiling",
            )
            if rounds and used + worst_round > 320:
                break
            round_index = len(rounds)
            round_seed = decoder_base + round_index * 10_007
            particle_seeds = tuple(round_seed + slot for slot in range(32))
            result = run_mode_fixed_matched_control_smc_v4(
                root,
                rates.rates,
                potential,
                method_id=sampling_method,
                source_key=str(source["source_key"]),
                particle_mode_ids=mode_ids,
                particle_seeds=particle_seeds,
                resampling_seed=round_seed + 100_000,
                beta_max=float(config["beta_max"]),
            )
            replay = run_mode_fixed_matched_control_smc_v4(
                root,
                rates.rates,
                potential,
                method_id=sampling_method,
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
        _require(
            int(compute["total_forward_equivalents"]) <= 320,
            "V4 matched control exceeded the compute ceiling",
        )
        torch.cuda.synchronize(device)
        compute["source_equal_wall_time_seconds"] = time.perf_counter() - source_started
        compute["source_equal_wall_time_scope"] = EQUAL_WALL_TIME_SCOPE_V3
        compute["source_equal_wall_peak_vram_mb"] = (
            torch.cuda.max_memory_allocated(device) / 1024**2
        )
        compute["source_cuda_device_name"] = str(cuda["cuda_device_name"])
        compute["terminal_critic_forwards_are_reserved_pending_scoring"] = True
        compute["trajectory_critic_forwards_are_actual"] = True
        compute["terminal_critic_forwards_reserved_by_member"] = list(
            critic_reservation
        )
        compute["trajectory_critic_forwards_actual_by_member"] = [
            int(total) - int(reserved)
            for total, reserved in zip(
                compute["critic_forwards_by_member"],
                critic_reservation,
                strict=True,
            )
        ]
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
                            "method_id": method,
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
                            "critic_self_score_used_for_selection": False,
                            "terminal_rerank_pending": method == "generate_then_rerank",
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
        "schema_version": "route_a_v3_route2_xeditflow_matched_control_generation.v4",
        "status": (
            "XEDITFLOW_V4_MATCHED_CONTROL_GENERATION_FAIL"
            if failed
            else "XEDITFLOW_V4_SMC_GENERATION_COMPLETE_PENDING_TERMINAL_CRITIC_SCORING"
        ),
        "method_id": method,
        "base_flow_training_seed": seed,
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
        "terminal_rerank_pending": method == "generate_then_rerank",
        "additional_sampling_rounds_used_only_within_conservative_compute_bound": True,
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
            entrypoint="run_route2_xeditflow_matched_controls_v4",
            evaluation_outcomes_accessed=False,
        )
        raise
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
