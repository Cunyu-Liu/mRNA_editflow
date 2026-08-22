"""Batched 32-particle scalar-potential SMC for XEditFlow V3."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from core.route2_legal_xeditflow import STOP, FlowState, LegalAction, apply_action, legal_actions
from core.route2_source_token_cache_v3 import SourceTokenCacheIndexV3
from core.route2_xeditflow_guidance_v3 import (
    MatchedComputeRecordV2,
    beta_schedule_v3,
    deduplicate_terminal_candidates_v3,
    stratified_resample_v3,
)
from core.route2_xeditsetflow_runtime_v3 import setflow_arm_rates_v3
from core.route2_xeditsetflow_sampling_v3 import (
    BASE,
    TOKEN,
    SetFlowGenerationMetadataV3,
    collate_generation_states_v3,
)


class XEditFlowSMCRuntimeV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowSMCRuntimeV3Error(message)


@dataclass(frozen=True)
class BatchedRateRowV3:
    actions: tuple[LegalAction, ...]
    rates: tuple[float, ...]


def sample_base_proposal_v3(
    rows: Sequence[BatchedRateRowV3],
    *,
    uniforms: Sequence[float],
) -> list[int]:
    _require(bool(rows) and len(rows) == len(uniforms), "base proposal rows and uniforms differ")
    choices = []
    for row, uniform in zip(rows, uniforms, strict=True):
        rates = np.asarray(row.rates, dtype=float)
        _require(
            len(row.actions) == len(rates) > 0
            and np.all(np.isfinite(rates))
            and np.all(rates > 0.0),
            "base proposal contains an invalid legal rate",
        )
        _require(math.isfinite(float(uniform)) and 0.0 <= float(uniform) < 1.0, "base proposal uniform differs")
        cumulative = np.cumsum(rates) / float(np.sum(rates))
        choices.append(min(int(np.searchsorted(cumulative, float(uniform), side="right")), len(rates) - 1))
    return choices


def scalar_potential_rate_map_v3(
    state: FlowState,
    actions: Sequence[LegalAction],
    rate_provider: Callable[[Sequence[FlowState]], Sequence[BatchedRateRowV3]],
    value_provider: Callable[[Sequence[FlowState]], Sequence[float]],
    *,
    beta_max: float,
) -> dict[LegalAction, float]:
    """Evaluate the exact potential-guided rate for every hard-legal action."""

    hard_legal = tuple(legal_actions(state))
    _require(tuple(actions) == hard_legal, "guided rate request does not preserve hard-legal action order")
    rows = list(rate_provider([state]))
    _require(len(rows) == 1 and rows[0].actions == hard_legal, "guided base-rate bundle differs")
    children = [apply_action(state, action) for action in hard_legal]
    potentials = list(value_provider([state] + children))
    _require(len(potentials) == len(children) + 1, "guided value bundle differs")
    _require(all(math.isfinite(float(value)) for value in potentials), "guided value bundle is nonfinite")
    budget = state.edit_count + state.remaining_budget
    _require(budget in {1, 3, 5}, "guided state budget differs")
    beta = float(
        beta_schedule_v3(
            torch.tensor(state.edit_count / budget, dtype=torch.float64),
            beta_max=beta_max,
        )
    )
    result = {}
    for action, base_rate, child_value in zip(
        hard_legal, rows[0].rates, potentials[1:], strict=True
    ):
        rate = float(base_rate) * math.exp(
            beta * (float(child_value) - float(potentials[0]))
        )
        _require(math.isfinite(rate) and rate > 0.0, "guided exact legal rate is invalid")
        result[action] = rate
    return result


def run_batched_potential_smc_v3(
    root_state: FlowState,
    rate_provider: Callable[[Sequence[FlowState]], Sequence[BatchedRateRowV3]],
    value_provider: Callable[[Sequence[FlowState]], Sequence[float]],
    *,
    source_key: str,
    particle_seeds: Sequence[int],
    resampling_seed: int,
    beta_max: float,
) -> dict[str, Any]:
    """Use base actions as proposals and scalar potential differences as weights."""

    _require(root_state.terminal_cause is None, "SMC root is already terminal")
    root_budget = root_state.edit_count + root_state.remaining_budget
    _require(root_budget in {1, 3, 5}, "SMC root budget differs")
    _require(len(particle_seeds) == 32 and len(set(int(seed) for seed in particle_seeds)) == 32, "SMC requires 32 distinct particle seeds")
    generators = [np.random.default_rng(int(seed)) for seed in particle_seeds]
    particles = [root_state for _ in range(32)]
    trajectories: list[list[str]] = [[] for _ in range(32)]
    log_weights = np.full(32, -math.log(32.0), dtype=float)
    base_calls = 0
    value_calls = 0
    resampling_events = []
    step = 0
    started = time.time()
    while not all(state.terminal_cause is not None for state in particles):
        active = [index for index, state in enumerate(particles) if state.terminal_cause is None]
        _require(step < 5, "SMC active trajectory exceeded the five-edit ceiling")
        active_states = [particles[index] for index in active]
        rows = list(rate_provider(active_states))
        base_calls += 1
        _require(len(rows) == len(active_states), "batched base-rate provider count differs")
        for state, row in zip(active_states, rows, strict=True):
            legal = set(legal_actions(state))
            _require(len(row.actions) == len(set(row.actions)), "base-rate action is duplicated")
            _require(set(row.actions) == legal, "base-rate provider does not cover exactly hard-legal actions")
        uniforms = [generators[index].random() for index in active]
        choices = sample_base_proposal_v3(rows, uniforms=uniforms)
        children = [
            apply_action(state, row.actions[choice])
            for state, row, choice in zip(active_states, rows, choices, strict=True)
        ]
        potentials = list(value_provider(active_states + children))
        value_calls += 1
        _require(len(potentials) == 2 * len(active), "batched scalar potential count differs")
        _require(all(math.isfinite(float(value)) for value in potentials), "batched scalar potential is nonfinite")
        current_values = potentials[: len(active)]
        child_values = potentials[len(active) :]
        progress = torch.tensor(
            [state.edit_count / root_budget for state in active_states],
            dtype=torch.float64,
        )
        betas = beta_schedule_v3(progress, beta_max=beta_max).tolist()
        for offset, particle_index in enumerate(active):
            action = rows[offset].actions[choices[offset]]
            particles[particle_index] = children[offset]
            trajectories[particle_index].append(action.action_id)
            log_weights[particle_index] += float(betas[offset]) * (
                float(child_values[offset]) - float(current_values[offset])
            )
        step += 1
        resampling = stratified_resample_v3(
            log_weights, seed=int(resampling_seed) + step, threshold=16.0
        )
        resampling_events.append(
            {
                "step": step,
                "ess_before": float(resampling["ess_before"]),
                "resampled": bool(resampling["resampled"]),
            }
        )
        if resampling["resampled"]:
            ancestors = resampling["ancestor_indices"]
            particles = [particles[index] for index in ancestors]
            trajectories = [list(trajectories[index]) for index in ancestors]
            log_weights = np.asarray(resampling["log_weights_after"], dtype=float)
    terminals = [
        {
            "candidate_sequence": state.current_sequence,
            "terminal_cause": state.terminal_cause,
            "edit_count": state.edit_count,
            "log_weight": float(log_weights[index]),
            "trajectory_actions": trajectories[index],
            "particle_slot": index,
        }
        for index, state in enumerate(particles)
    ]
    candidates = deduplicate_terminal_candidates_v3(terminals, candidate_cap=32)
    compute = MatchedComputeRecordV2(
        source_key=source_key,
        base_flow_forwards=base_calls,
        value_forwards=value_calls,
        candidate_count=len(candidates),
        wall_time_seconds=time.time() - started,
    )
    return {
        "schema_version": "route_a_v3_route2_xeditflow_batched_potential_smc.v3",
        "status": "XEDITFLOW_V3_SMC_COMPLETE",
        "source_key": source_key,
        "particle_count": 32,
        "candidate_cap": 32,
        "completed_steps": step,
        "candidates": candidates,
        "resampling_events": resampling_events,
        "matched_compute": compute.to_dict(),
        "hard_legality_rate": 1.0,
        "edit_budget_violation_count": sum(state.edit_count > root_budget for state in particles),
        "candidate_budget_violation_count": int(len(candidates) > 32),
        "numerical_failure_count": 0,
        "proposal": "BASE_FLOW_TRANSITION",
        "incremental_importance_weight": "EXP_BETA_TIMES_SCALAR_POTENTIAL_DIFFERENCE",
        "free_action_ratio_head_used": False,
    }


def merge_smc_rounds_v3(
    round_results: Sequence[Mapping[str, Any]],
    *,
    source_key: str,
    reserved_critic_forwards: int = 3,
) -> dict[str, Any]:
    """Merge additional 32-particle rounds without exceeding matched compute."""

    _require(bool(round_results), "SMC round collection is empty")
    _require(reserved_critic_forwards == 3, "terminal critic ensemble reservation differs")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    base_calls = 0
    value_calls = 0
    wall_time = 0.0
    for result in round_results:
        _require(result.get("source_key") == source_key, "SMC round source differs")
        _require(result.get("status") == "XEDITFLOW_V3_SMC_COMPLETE", "SMC round is incomplete")
        compute = result["matched_compute"]
        base_calls += int(compute["base_flow_forwards"])
        value_calls += int(compute["value_forwards"])
        wall_time += float(compute["wall_time_seconds"])
        for candidate in result["candidates"]:
            grouped.setdefault(str(candidate["candidate_sequence"]), []).append(candidate)
    _require(base_calls + value_calls + reserved_critic_forwards <= 320, "additional SMC rounds exceed matched compute")
    candidates = []
    for sequence, members in grouped.items():
        weights = np.asarray([float(row["merged_log_weight"]) for row in members])
        _require(np.all(np.isfinite(weights)), "SMC round candidate mass is nonfinite")
        maximum = float(np.max(weights))
        merged = maximum + math.log(float(np.exp(weights - maximum).sum())) - math.log(len(round_results))
        representative = dict(max(members, key=lambda row: float(row["merged_log_weight"])))
        representative["candidate_sequence"] = sequence
        representative["merged_log_weight"] = merged
        representative["sampling_round_multiplicity"] = len(members)
        representative["particle_multiplicity"] = sum(int(row["particle_multiplicity"]) for row in members)
        candidates.append(representative)
    candidates.sort(key=lambda row: (-float(row["merged_log_weight"]), str(row["candidate_sequence"])))
    candidates = candidates[:32]
    compute = MatchedComputeRecordV2(
        source_key=source_key,
        base_flow_forwards=base_calls,
        value_forwards=value_calls,
        candidate_count=len(candidates),
        wall_time_seconds=wall_time,
    ).to_dict()
    return {
        "source_key": source_key,
        "sampling_round_count": len(round_results),
        "candidates": candidates,
        "matched_compute": compute,
        "reserved_terminal_critic_forwards": 3,
        "remaining_forward_equivalents_after_reservation": 320
        - compute["total_forward_equivalents"]
        - 3,
    }


class SetFlowValueProvidersV3:
    """CUDA/BF16 batched providers used by the formal SMC runner."""

    def __init__(
        self,
        *,
        setflow_model: nn.Module,
        setflow_arm: str,
        value_model: nn.Module,
        metadata: SetFlowGenerationMetadataV3,
        source_cache: SourceTokenCacheIndexV3,
        device: torch.device,
    ) -> None:
        _require(device.type == "cuda", "formal SMC providers require CUDA")
        self.setflow_model = setflow_model
        self.setflow_arm = setflow_arm
        self.value_model = value_model
        self.metadata = metadata
        self.source_cache = source_cache
        self.device = device

    def _batch(self, states: Sequence[FlowState]) -> dict[str, torch.Tensor]:
        batch = collate_generation_states_v3(
            states,
            [self.metadata] * len(states),
            source_cache=self.source_cache,
        )
        return {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}

    @torch.no_grad()
    def rates(self, states: Sequence[FlowState]) -> list[BatchedRateRowV3]:
        self.setflow_model.eval()
        batch = self._batch(states)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            rates, masks = setflow_arm_rates_v3(self.setflow_model, self.setflow_arm, batch)
        result = []
        padded_length = (rates.shape[1] - 1) // 4
        for row_index, state in enumerate(states):
            actions = tuple(legal_actions(state))
            values = []
            for action in actions:
                flat = padded_length * 4 if action.kind == STOP else int(action.position) * 4 + TOKEN[str(action.alt_base)]
                _require(bool(masks[row_index, flat].item()), "formal SMC model masked a hard-legal action")
                value = float(rates[row_index, flat].item())
                _require(math.isfinite(value) and value > 0.0, "formal SMC model legal rate is invalid")
                values.append(value)
            result.append(BatchedRateRowV3(actions=actions, rates=tuple(values)))
        return result

    @torch.no_grad()
    def values(self, states: Sequence[FlowState]) -> list[float]:
        self.value_model.eval()
        batch = self._batch(states)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            values = self.value_model(batch)
        _require(values.shape == (len(states),), "formal SMC value output differs")
        result = values.float().cpu().tolist()
        _require(all(math.isfinite(float(value)) for value in result), "formal SMC value is nonfinite")
        return result
