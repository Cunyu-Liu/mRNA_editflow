#!/usr/bin/env python3
"""Run frozen matched XEditFlow V3 control methods on CUDA/BF16."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_development_projection_v3 import load_projection_rows
from core.route2_gpu_failure_evidence import cuda_device_observation, write_gpu_failure_evidence
from core.route2_legal_xeditflow import FlowState, initial_state, validate_state
from core.route2_source_token_cache_v3 import SourceTokenCacheIndexV3, load_source_token_cache_v3
from core.route2_xeditflow_gate_v3 import authorize_xeditflow_guidance_v3
from core.route2_xeditflow_guidance_v3 import MatchedComputeRecordV2
from core.route2_xeditflow_matched_methods_v3 import (
    CriticRewardBatchV3,
    ExactCriticRewardPotentialV3,
    SourceAnchoredFirstOrderPotentialV3,
    ZeroPotentialV3,
    merge_matched_control_rounds_v3,
    rerank_terminal_candidates_v3,
    run_batched_critic_potential_smc_v3,
)
from core.route2_xeditflow_smc_runtime_v3 import SetFlowRateProviderV3
from core.route2_xeditflow_value_training_v3 import CRITIC_SEEDS_V3
from core.route2_xeditsetflow_sampling_v3 import build_generation_metadata_v3
from scripts.route_a_v3.generate_route2_xeditflow_value_rollouts_v3 import (
    _load_critic_member_v3,
    _score_loaded_critic_member_rows_v3,
)
from scripts.route_a_v3.run_route2_base_flow_g0_validation_v1 import load_sources
from scripts.route_a_v3.score_route2_xeditflow_critic_ensemble_v3 import _representatives_v3
from scripts.route_a_v3.validate_route2_xeditsetflow_v3 import load_setflow_checkpoint_v3


METHODS = {
    "unguided_setflow",
    "first_order_guidance",
    "simple_rate_guidance",
    "generate_then_rerank",
}


class XEditFlowMatchedControlsV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowMatchedControlsV3Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def validate_matched_control_config_v3(config: Mapping[str, Any]) -> None:
    _require(config.get("schema_version") == "route_a_v3_route2_xeditflow_matched_control_run_config.v1", "unexpected matched-control config")
    _require(str(config.get("method_id")) in METHODS, "matched-control method differs")
    _require(int(config.get("base_flow_training_seed", -1)) in {20260904, 20260905, 20260906}, "matched-control base-flow seed differs")
    _require(str(config.get("setflow_arm")) in {"f2", "f3"}, "matched-control SetFlow arm differs")
    _require(float(config.get("kappa", -1)) in {0.0, 0.5, 1.0}, "matched-control kappa differs")
    _require(float(config.get("beta_max", -1)) in {0.5, 1.0, 2.0}, "matched-control beta differs")
    _require(int(config.get("particle_count", -1)) == 32, "matched-control particle count changed")
    _require(int(config.get("candidate_cap", -1)) == 32, "matched-control candidate cap changed")
    _require(float(config.get("ess_threshold", -1)) == 16.0, "matched-control ESS threshold changed")
    _require(config.get("resampling") == "STRATIFIED", "matched-control resampling changed")
    _require(int(config.get("forward_equivalent_ceiling_per_source", -1)) == 320, "matched-control compute ceiling changed")
    _require(int(config.get("maximum_sampling_rounds", -1)) == 32, "matched-control round ceiling changed")
    _require(int(config.get("reserved_terminal_critic_forwards", -1)) == 3, "matched-control terminal reservation changed")
    _require(int(config.get("critic_online_microbatch_size", -1)) == 4, "matched-control Critic microbatch changed")
    _require(config.get("action_space") == "SUB+STOP" and config.get("replay_check") is True, "matched-control legal/replay policy changed")
    gpu = int(config.get("physical_gpu_index", -1))
    _require(gpu in set(range(6)) and config.get("device") == f"cuda:{gpu}", "matched-control GPU provenance differs")
    _require(str(config.get("output_dir", "")).startswith("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"), "matched-control output left Route 2 /mnt")


def _critic_adapter_rows_for_states_v3(
    states: Sequence[FlowState],
    *,
    source_row: Mapping[str, Any],
    representative: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source = str(source_row["source_sequence"])
    region = str(source_row["region"]).replace("′", "").replace("'", "")
    _require(region in {"5UTR", "3UTR"}, "matched-control Critic region differs")
    rows = []
    for index, state in enumerate(states):
        _require(state.source_sequence == source, "matched-control Critic state crossed source")
        candidate = state.current_sequence
        edits = [
            {"position": position, "source_base": source[position], "candidate_base": alt}
            for position, alt in state.source_relative_edits
        ]
        rows.append(
            {
                "state_id": str(source_row["source_key"]),
                "rollout_index": index,
                "source_group_id": str(representative["source_group_id"]),
                "task_id": str(representative["task_id"]),
                "source_sequence": source,
                "candidate_sequence": candidate,
                "source_relative_edits": edits,
                "endpoint_descriptor": dict(representative["endpoint_descriptor"]),
                "assay_category": str(source_row["assay_id"]),
                "context_category": str(source_row["biological_context_id"]),
                "region_id": 0 if region == "5UTR" else 1,
            }
        )
    return rows


class FrozenCriticEnsembleRewardV3:
    """Three resident refit members with a source-local reward cache."""

    def __init__(
        self,
        *,
        checkpoint_paths: Mapping[int, Path],
        selected_arm: str,
        model_path: Path,
        device: torch.device,
        kappa: float,
        microbatch_size: int,
    ) -> None:
        _require(tuple(sorted(checkpoint_paths)) == CRITIC_SEEDS_V3, "matched-control Critic seed inventory differs")
        self.selected_arm = selected_arm
        self.device = device
        self.kappa = float(kappa)
        self.microbatch_size = int(microbatch_size)
        self.members = {}
        reference_vocabs = None
        for seed in CRITIC_SEEDS_V3:
            model, encoder, vocabs = _load_critic_member_v3(
                checkpoint_paths[seed],
                selected_arm=selected_arm,
                seed=seed,
                model_path=model_path,
                device=device,
            )
            if reference_vocabs is None:
                reference_vocabs = vocabs
            else:
                _require(vocabs == reference_vocabs, "matched-control Critic vocabularies differ across seeds")
            self.members[seed] = (model, encoder, vocabs)

    def bind_source(
        self,
        source_row: Mapping[str, Any],
        representative: Mapping[str, Any],
    ) -> "BoundCriticEnsembleRewardV3":
        return BoundCriticEnsembleRewardV3(self, source_row, representative)


class BoundCriticEnsembleRewardV3:
    def __init__(
        self,
        parent: FrozenCriticEnsembleRewardV3,
        source_row: Mapping[str, Any],
        representative: Mapping[str, Any],
    ) -> None:
        self.parent = parent
        self.source_row = source_row
        self.representative = representative
        self.cache: dict[str, float] = {}

    def __call__(self, states: Sequence[FlowState]) -> CriticRewardBatchV3:
        _require(bool(states), "matched-control Critic state batch is empty")
        missing_sequences = []
        representative_states = {}
        for state in states:
            sequence = state.current_sequence
            if sequence not in self.cache and sequence not in representative_states:
                missing_sequences.append(sequence)
                representative_states[sequence] = state
        member_calls = (0, 0, 0)
        if missing_sequences:
            missing_states = [representative_states[sequence] for sequence in missing_sequences]
            rows = _critic_adapter_rows_for_states_v3(
                missing_states,
                source_row=self.source_row,
                representative=self.representative,
            )
            predictions = []
            for seed in CRITIC_SEEDS_V3:
                model, encoder, vocabs = self.parent.members[seed]
                values = _score_loaded_critic_member_rows_v3(
                    rows,
                    model=model,
                    encoder=encoder,
                    vocabs=vocabs,
                    selected_arm=self.parent.selected_arm,
                    device=self.parent.device,
                    microbatch_size=self.parent.microbatch_size,
                )
                predictions.append(values)
            matrix = np.asarray(predictions, dtype=float).T
            _require(matrix.shape == (len(missing_sequences), 3) and bool(np.all(np.isfinite(matrix))), "matched-control Critic matrix differs")
            rewards = matrix.mean(axis=1) - self.parent.kappa * matrix.std(axis=1, ddof=0)
            for sequence, reward in zip(missing_sequences, rewards.tolist()):
                self.cache[sequence] = float(reward)
            member_calls = (1, 1, 1)
        return CriticRewardBatchV3(
            tuple(self.cache[state.current_sequence] for state in states),
            member_calls,
        )


def _terminal_state_from_candidate_v3(root: FlowState, candidate: Mapping[str, Any]) -> FlowState:
    sequence = str(candidate["candidate_sequence"])
    _require(len(sequence) == len(root.source_sequence), "matched-control terminal length differs")
    edits = tuple(
        (position, right)
        for position, (left, right) in enumerate(zip(root.source_sequence, sequence))
        if left != right
    )
    state = FlowState(
        source_sequence=root.source_sequence,
        current_sequence=sequence,
        source_relative_edits=edits,
        remaining_budget=root.remaining_budget - len(edits),
        assay_id=root.assay_id,
        context_id=root.context_id,
        terminal_cause=str(candidate["terminal_cause"]),
    )
    validate_state(state)
    _require(state.terminal_cause in {"EXPLICIT_STOP", "BUDGET_EXHAUSTED"}, "matched-control terminal cause differs")
    return state


def _final_compute_v3(
    merged_compute: Mapping[str, Any],
    terminal_member_calls: Sequence[int],
) -> dict[str, Any]:
    record = MatchedComputeRecordV2(
        source_key=str(merged_compute["source_key"]),
        base_flow_forwards=int(merged_compute["base_flow_forwards"]),
        value_forwards=int(merged_compute["value_forwards"]),
        critic_forwards_by_member=[
            int(before) + int(after)
            for before, after in zip(
                merged_compute["critic_forwards_by_member"], terminal_member_calls
            )
        ],
        candidate_count=int(merged_compute["candidate_count"]),
        wall_time_seconds=float(merged_compute["wall_time_seconds"]),
        peak_vram_mb=float(merged_compute.get("peak_vram_mb", 0.0)),
    )
    return record.to_dict()


def run(config: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
    validate_matched_control_config_v3(config)
    _require(output_dir == Path(config["output_dir"]), "matched-control output path differs")
    _require(not output_dir.exists(), f"matched-control output exists: {output_dir}")
    authorization = authorize_xeditflow_guidance_v3(
        _json(Path(config["critic_readiness_path"])),
        _json(Path(config["setflow_confirmation_path"])),
    )
    _require(authorization["guidance_authorized"] is True, "matched-control generation remains blocked before readiness")
    refit = _json(Path(config["critic_refit_manifest_path"]))
    _require(refit.get("status") == "XEDITCRITIC_V3_ALL_DEVELOPMENT_REFIT_COMPLETE", "matched-control Critic refit is incomplete")
    selected_arm = str(refit.get("selected_arm"))
    _require(selected_arm in {"C2", "C3"}, "matched-control Critic arm differs")
    checkpoints = {int(row["seed"]): Path(row["checkpoint_path"]) for row in refit["checkpoints"]}
    _require(tuple(sorted(checkpoints)) == CRITIC_SEEDS_V3, "matched-control refit checkpoint inventory differs")
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden")
    gpu = int(config["physical_gpu_index"])
    device = torch.device(str(config["device"]))
    torch.cuda.set_device(device)
    _require(torch.cuda.is_bf16_supported(), "BF16 is unavailable for matched controls")
    cuda = cuda_device_observation(gpu, require_physical_index_match=True)
    setflow, checkpoint = load_setflow_checkpoint_v3(
        Path(config["setflow_checkpoint_path"]), str(config["setflow_arm"]), device
    )
    _require(int(checkpoint["training_provenance"]["seed"]) == int(config["base_flow_training_seed"]), "matched-control SetFlow seed differs")
    cache = SourceTokenCacheIndexV3(load_source_token_cache_v3(Path(config["source_token_cache_path"])))
    sources = load_sources(Path(config["source_eligibility_manifest"]))
    _require(len(sources) == int(config["expected_source_count"]), "matched-control source cohort changed")
    projection = load_projection_rows([Path(config["validation_projection_path"])], allowed_splits=("VALIDATION",))
    metadata = build_generation_metadata_v3(sources, projection, checkpoint["vocabs"])
    _require(len(metadata) == len(sources), "matched-control metadata count changed")
    representatives = _representatives_v3(sources, projection)
    critic = FrozenCriticEnsembleRewardV3(
        checkpoint_paths=checkpoints,
        selected_arm=selected_arm,
        model_path=Path(config["mrnabert_model_path"]),
        device=device,
        kappa=float(config["kappa"]),
        microbatch_size=int(config["critic_online_microbatch_size"]),
    )
    method = str(config["method_id"])
    sampling_method = "unguided_setflow" if method == "generate_then_rerank" else method
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    candidate_path = output_dir / "generated_candidates.private.jsonl"
    compute_path = output_dir / "matched_compute.jsonl"
    candidate_path.write_text("", encoding="utf-8")
    compute_path.write_text("", encoding="utf-8")
    replay_failures = 0
    total_candidates = 0
    maximum_compute = 0
    started = time.time()
    for source_index, (source, source_metadata) in enumerate(zip(sources, metadata)):
        root = initial_state(
            str(source["source_sequence"]),
            budget=int(source["edit_budget"]),
            assay_id=str(source["assay_id"]),
            context_id=str(source["biological_context_id"]),
        )
        rates = SetFlowRateProviderV3(
            setflow_model=setflow,
            setflow_arm=str(config["setflow_arm"]),
            metadata=source_metadata,
            source_cache=cache,
            device=device,
        )
        reward = critic.bind_source(source, representatives[str(source["source_key"])])
        if sampling_method == "first_order_guidance":
            potential = SourceAnchoredFirstOrderPotentialV3(root, reward)
        elif sampling_method == "simple_rate_guidance":
            potential = ExactCriticRewardPotentialV3(reward)
        else:
            potential = ZeroPotentialV3()
        decoder_base = int(config["decoder_seed_base"]) + source_index * 1_000_003
        rounds = []
        replay_ok = True
        maximum_round_cost = int(source["edit_budget"]) * (
            4 if sampling_method in {"first_order_guidance", "simple_rate_guidance"} else 1
        )
        while len(rounds) < int(config["maximum_sampling_rounds"]):
            used = sum(int(row["matched_compute"]["total_forward_equivalents"]) for row in rounds)
            if rounds and used + maximum_round_cost + 3 > 320:
                break
            round_index = len(rounds)
            round_seed = decoder_base + round_index * 10_007
            seeds = [round_seed + slot for slot in range(32)]
            result = run_batched_critic_potential_smc_v3(
                root,
                rates.rates,
                potential,
                method_id=sampling_method,
                source_key=str(source["source_key"]),
                particle_seeds=seeds,
                resampling_seed=round_seed + 100_000,
                beta_max=float(config["beta_max"]),
            )
            replay = run_batched_critic_potential_smc_v3(
                root,
                rates.rates,
                potential,
                method_id=sampling_method,
                source_key=str(source["source_key"]),
                particle_seeds=seeds,
                resampling_seed=round_seed + 100_000,
                beta_max=float(config["beta_max"]),
            )
            this_replay_ok = result["candidates"] == replay["candidates"] and result["resampling_events"] == replay["resampling_events"]
            replay_ok = replay_ok and this_replay_ok
            replay_failures += int(not this_replay_ok)
            rounds.append(result)
            merged = merge_matched_control_rounds_v3(
                rounds,
                source_key=str(source["source_key"]),
                reserved_terminal_critic_forwards=3,
            )
            if len(merged["candidates"]) >= 32:
                break
        merged = merge_matched_control_rounds_v3(
            rounds,
            source_key=str(source["source_key"]),
            reserved_terminal_critic_forwards=3,
        )
        terminal_states = [_terminal_state_from_candidate_v3(root, row) for row in merged["candidates"]]
        terminal_rewards = reward(terminal_states)
        candidates = list(merged["candidates"])
        if method == "generate_then_rerank":
            candidates = rerank_terminal_candidates_v3(candidates, terminal_rewards.values)
        reward_by_sequence = {
            state.current_sequence: value
            for state, value in zip(terminal_states, terminal_rewards.values)
        }
        compute = _final_compute_v3(merged["matched_compute"], terminal_rewards.forward_batches_by_member)
        maximum_compute = max(maximum_compute, int(compute["total_forward_equivalents"]))
        with compute_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({**compute, "sampling_round_count": len(rounds)}, sort_keys=True) + "\n")
        with candidate_path.open("a", encoding="utf-8") as handle:
            for rank, candidate in enumerate(candidates, start=1):
                sequence = str(candidate["candidate_sequence"])
                generation_score = (
                    float(reward_by_sequence[sequence])
                    if method == "generate_then_rerank"
                    else float(candidate["merged_log_weight"])
                )
                row = {
                    **candidate,
                    "schema_version": "route_a_v3_route2_xeditflow_generated_candidate.v3",
                    "method_id": method,
                    "source_key": str(source["source_key"]),
                    "generation_rank": rank,
                    "generation_score": generation_score,
                    "terminal_flow_mass_score": float(candidate["merged_log_weight"]),
                    "critic_self_score": float(reward_by_sequence[sequence]),
                    "critic_self_score_used_for_selection": method == "generate_then_rerank",
                    "trajectory_replay_ok": replay_ok,
                    "base_flow_training_seed": int(config["base_flow_training_seed"]),
                    "decoder_seed_stream_start": decoder_base,
                    "sampling_round_count": len(rounds),
                    "generated_candidate_grants_canonical_credit": False,
                    "independent_evaluator_score": None,
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                total_candidates += 1
    summary = {
        "schema_version": "route_a_v3_route2_xeditflow_matched_control_generation.v3",
        "status": "XEDITFLOW_V3_MATCHED_CONTROL_GENERATION_COMPLETE" if replay_failures == 0 else "XEDITFLOW_V3_MATCHED_CONTROL_GENERATION_FAIL",
        "method_id": method,
        "sampling_method_id": sampling_method,
        "setflow_arm": str(config["setflow_arm"]),
        "critic_arm": selected_arm,
        "base_flow_training_seed": int(config["base_flow_training_seed"]),
        "critic_seeds": list(CRITIC_SEEDS_V3),
        "source_count": len(sources),
        "generated_candidate_count": total_candidates,
        "particle_count_per_source": 32,
        "candidate_cap_per_source": 32,
        "maximum_forward_equivalents_per_source": maximum_compute,
        "forward_equivalent_ceiling_per_source": 320,
        "trajectory_replay_failure_count": replay_failures,
        "edit_budget_violation_count": 0,
        "candidate_budget_violation_count": 0,
        "numerical_failure_count": 0,
        "hard_legality_rate": 1.0,
        "critic_self_score_used_for_selection": method == "generate_then_rerank",
        "independent_evaluator_used_for_gradient": False,
        "candidate_path": str(candidate_path),
        "compute_path": str(compute_path),
        "wall_time_seconds_including_replay_check": time.time() - started,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "cpu_fallback_used": False,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
        **cuda,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = _json(args.config)
    output_dir = Path(config["output_dir"])
    try:
        result = run(config, output_dir=output_dir)
    except Exception as exc:
        write_gpu_failure_evidence(
            output_dir.with_name(output_dir.name + ".failed.json"),
            config,
            exc,
            entrypoint="run_route2_xeditflow_matched_controls_v3",
            evaluation_outcomes_accessed=False,
        )
        raise
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
