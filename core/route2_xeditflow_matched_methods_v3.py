"""Matched first-order, one-step critic, and terminal-rerank controls."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from core.route2_legal_xeditflow import FlowState, LegalAction, apply_action, legal_actions
from core.route2_xeditflow_guidance_v3 import (
    MatchedComputeRecordV2,
    TERMINAL_CRITIC_FORWARD_RESERVATION_V3,
    beta_schedule_v3,
    deduplicate_terminal_candidates_v3,
    stratified_resample_v3,
)
from core.route2_xeditflow_smc_runtime_v3 import BatchedRateRowV3, sample_base_proposal_v3


CRITIC_POTENTIAL_METHODS_V3 = {
    "first_order_guidance",
    "simple_rate_guidance",
}
MATCHED_CONTROL_METHODS_V3 = CRITIC_POTENTIAL_METHODS_V3 | {"unguided_setflow"}


class XEditFlowMatchedMethodsV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowMatchedMethodsV3Error(message)


@dataclass(frozen=True)
class CriticRewardBatchV3:
    values: tuple[float, ...]
    forward_batches_by_member: tuple[int, int, int]

    def __post_init__(self) -> None:
        _require(all(math.isfinite(float(value)) for value in self.values), "Critic reward batch is nonfinite")
        _require(
            len(self.forward_batches_by_member) == 3
            and min(self.forward_batches_by_member) >= 0,
            "Critic reward member accounting differs",
        )


@dataclass(frozen=True)
class CriticPotentialBatchV3:
    values: tuple[float, ...]
    forward_batches_by_member: tuple[int, int, int]


CriticRewardProviderV3 = Callable[[Sequence[FlowState]], CriticRewardBatchV3]
CriticPotentialProviderV3 = Callable[[Sequence[FlowState]], CriticPotentialBatchV3]
RateProviderV3 = Callable[[Sequence[FlowState]], Sequence[BatchedRateRowV3]]


def _validate_rate_row_v3(state: FlowState, row: BatchedRateRowV3) -> None:
    hard_legal = tuple(legal_actions(state))
    _require(row.actions == hard_legal, "matched method rate bundle differs from hard legality")
    _require(len(row.rates) == len(hard_legal), "matched method legal-rate count differs")
    values = np.asarray(row.rates, dtype=float)
    _require(
        bool(np.all(np.isfinite(values))) and bool(np.all(values > 0.0)),
        "matched method legal rate is invalid",
    )


class SourceAnchoredFirstOrderPotentialV3:
    """Lazy discrete first-order expansion around one immutable source."""

    def __init__(
        self,
        root_state: FlowState,
        reward_provider: CriticRewardProviderV3,
    ) -> None:
        _require(root_state.edit_count == 0 and root_state.terminal_cause is None, "first-order anchor is not an active source")
        self.root_state = root_state
        self.reward_provider = reward_provider
        self.source_reward: float | None = None
        self.coefficients: dict[tuple[int, str], float] = {}

    def __call__(self, states: Sequence[FlowState]) -> CriticPotentialBatchV3:
        _require(bool(states), "first-order potential state batch is empty")
        for state in states:
            _require(
                state.source_sequence == self.root_state.source_sequence,
                "first-order potential crossed source identity",
            )
        missing = sorted(
            {
                (int(position), str(alt))
                for state in states
                for position, alt in state.source_relative_edits
                if (int(position), str(alt)) not in self.coefficients
            }
        )
        member_calls = (0, 0, 0)
        if self.source_reward is None or missing:
            requests: list[FlowState] = []
            include_source = self.source_reward is None
            if include_source:
                requests.append(self.root_state)
            for position, alt in missing:
                action = LegalAction("SUB", position, alt)
                _require(action in legal_actions(self.root_state), "first-order requested an illegal source edit")
                requests.append(apply_action(self.root_state, action))
            scored = self.reward_provider(requests)
            _require(len(scored.values) == len(requests), "first-order reward count differs")
            member_calls = scored.forward_batches_by_member
            offset = 0
            if include_source:
                self.source_reward = float(scored.values[0])
                offset = 1
            assert self.source_reward is not None
            for edit, reward in zip(missing, scored.values[offset:]):
                self.coefficients[edit] = float(reward) - self.source_reward
        potentials = tuple(
            math.fsum(self.coefficients[(int(position), str(alt))] for position, alt in state.source_relative_edits)
            for state in states
        )
        _require(all(math.isfinite(value) for value in potentials), "first-order potential is nonfinite")
        return CriticPotentialBatchV3(potentials, member_calls)


class ExactCriticRewardPotentialV3:
    """Short-sighted scalar potential equal to the current frozen Critic reward."""

    def __init__(self, reward_provider: CriticRewardProviderV3) -> None:
        self.reward_provider = reward_provider

    def __call__(self, states: Sequence[FlowState]) -> CriticPotentialBatchV3:
        scored = self.reward_provider(states)
        _require(len(scored.values) == len(states), "one-step Critic reward count differs")
        return CriticPotentialBatchV3(scored.values, scored.forward_batches_by_member)


class ZeroPotentialV3:
    """The exact no-guidance control under the common proposal/sampler."""

    def __call__(self, states: Sequence[FlowState]) -> CriticPotentialBatchV3:
        _require(bool(states), "unguided potential state batch is empty")
        return CriticPotentialBatchV3(tuple(0.0 for _ in states), (0, 0, 0))


def run_batched_critic_potential_smc_v3(
    root_state: FlowState,
    rate_provider: RateProviderV3,
    potential_provider: CriticPotentialProviderV3,
    *,
    method_id: str,
    source_key: str,
    particle_seeds: Sequence[int],
    resampling_seed: int,
    beta_max: float,
) -> dict[str, Any]:
    """Use full-legal base proposals and a frozen Critic-derived scalar potential."""

    _require(method_id in MATCHED_CONTROL_METHODS_V3, "unknown matched-control method")
    _require(root_state.terminal_cause is None and root_state.edit_count == 0, "Critic-potential SMC root differs")
    root_budget = root_state.remaining_budget
    _require(root_budget in {1, 3, 5}, "Critic-potential SMC budget differs")
    _require(len(particle_seeds) == 32 and len(set(int(seed) for seed in particle_seeds)) == 32, "Critic-potential SMC requires 32 distinct particle seeds")
    generators = [np.random.default_rng(int(seed)) for seed in particle_seeds]
    particles = [root_state for _ in range(32)]
    trajectories: list[list[str]] = [[] for _ in particles]
    log_weights = np.full(32, -math.log(32.0), dtype=float)
    base_calls = 0
    critic_calls = [0, 0, 0]
    resampling_events = []
    step = 0
    started = time.time()
    while not all(state.terminal_cause is not None for state in particles):
        active = [index for index, state in enumerate(particles) if state.terminal_cause is None]
        _require(step < 5, "Critic-potential trajectory exceeded the five-edit ceiling")
        active_states = [particles[index] for index in active]
        rows = list(rate_provider(active_states))
        base_calls += 1
        _require(len(rows) == len(active_states), "Critic-potential base-rate count differs")
        for state, row in zip(active_states, rows):
            _validate_rate_row_v3(state, row)
        uniforms = [generators[index].random() for index in active]
        choices = sample_base_proposal_v3(rows, uniforms=uniforms)
        children = [
            apply_action(state, row.actions[choice])
            for state, row, choice in zip(active_states, rows, choices)
        ]
        potential_batch = potential_provider(active_states + children)
        _require(len(potential_batch.values) == 2 * len(active), "Critic-potential value count differs")
        if method_id == "unguided_setflow":
            _require(
                potential_batch.forward_batches_by_member == (0, 0, 0)
                and all(float(value) == 0.0 for value in potential_batch.values),
                "unguided control received a learned potential",
            )
        for member, count in enumerate(potential_batch.forward_batches_by_member):
            critic_calls[member] += int(count)
        current_values = potential_batch.values[: len(active)]
        child_values = potential_batch.values[len(active) :]
        progress = torch.tensor(
            [state.edit_count / root_budget for state in active_states],
            dtype=torch.float64,
        )
        betas = beta_schedule_v3(progress, beta_max=float(beta_max)).tolist()
        for offset, particle_index in enumerate(active):
            action = rows[offset].actions[choices[offset]]
            particles[particle_index] = children[offset]
            trajectories[particle_index].append(action.action_id)
            increment = float(betas[offset]) * (
                float(child_values[offset]) - float(current_values[offset])
            )
            _require(math.isfinite(increment), "Critic-potential weight increment is nonfinite")
            log_weights[particle_index] += increment
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
        critic_forwards_by_member=critic_calls,
        candidate_count=len(candidates),
        wall_time_seconds=time.time() - started,
    ).to_dict()
    return {
        "schema_version": "route_a_v3_route2_xeditflow_matched_control_smc.v3",
        "status": "XEDITFLOW_V3_MATCHED_CONTROL_SMC_COMPLETE",
        "method_id": method_id,
        "source_key": source_key,
        "particle_count": 32,
        "candidate_cap": 32,
        "completed_steps": step,
        "candidates": candidates,
        "resampling_events": resampling_events,
        "matched_compute": compute,
        "hard_legality_rate": 1.0,
        "edit_budget_violation_count": sum(state.edit_count > root_budget for state in particles),
        "candidate_budget_violation_count": int(len(candidates) > 32),
        "numerical_failure_count": 0,
        "proposal": "BASE_FLOW_TRANSITION",
        "incremental_importance_weight": (
            "UNITY"
            if method_id == "unguided_setflow"
            else "EXP_BETA_TIMES_CRITIC_DERIVED_SCALAR_POTENTIAL_DIFFERENCE"
        ),
        "first_order_anchor": "SOURCE" if method_id == "first_order_guidance" else None,
        "edit_interactions_present": method_id == "simple_rate_guidance",
        "free_action_ratio_head_used": False,
    }


def merge_matched_control_rounds_v3(
    round_results: Sequence[Mapping[str, Any]],
    *,
    source_key: str,
    reserved_terminal_critic_forwards: int = TERMINAL_CRITIC_FORWARD_RESERVATION_V3,
) -> dict[str, Any]:
    """Merge repeated 32-particle control rounds under the common ceiling."""

    _require(bool(round_results), "matched-control round collection is empty")
    _require(
        reserved_terminal_critic_forwards
        in {0, TERMINAL_CRITIC_FORWARD_RESERVATION_V3},
        "matched-control terminal reservation differs",
    )
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    base_calls = 0
    value_calls = 0
    critic_calls = [0, 0, 0]
    wall_time = 0.0
    method_id = str(round_results[0].get("method_id"))
    for result in round_results:
        _require(result.get("source_key") == source_key, "matched-control round source differs")
        _require(result.get("method_id") == method_id, "matched-control round method differs")
        _require(result.get("status") == "XEDITFLOW_V3_MATCHED_CONTROL_SMC_COMPLETE", "matched-control round is incomplete")
        compute = result["matched_compute"]
        base_calls += int(compute["base_flow_forwards"])
        value_calls += int(compute["value_forwards"])
        for member, count in enumerate(compute["critic_forwards_by_member"]):
            critic_calls[member] += int(count)
        wall_time += float(compute["wall_time_seconds"])
        for candidate in result["candidates"]:
            grouped.setdefault(str(candidate["candidate_sequence"]), []).append(candidate)
    candidates = []
    for sequence, members in grouped.items():
        weights = np.asarray([float(row["merged_log_weight"]) for row in members], dtype=float)
        _require(bool(np.all(np.isfinite(weights))), "matched-control candidate mass is nonfinite")
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
        critic_forwards_by_member=critic_calls,
        candidate_count=len(candidates),
        wall_time_seconds=wall_time,
    ).to_dict()
    _require(
        int(compute["total_forward_equivalents"]) + reserved_terminal_critic_forwards <= 320,
        "matched-control rounds exceed compute after terminal reservation",
    )
    return {
        "source_key": source_key,
        "method_id": method_id,
        "sampling_round_count": len(round_results),
        "candidates": candidates,
        "matched_compute": compute,
        "reserved_terminal_critic_forwards": reserved_terminal_critic_forwards,
        "remaining_forward_equivalents_after_reservation": 320
        - int(compute["total_forward_equivalents"])
        - reserved_terminal_critic_forwards,
    }


def rerank_terminal_candidates_v3(
    candidates: Sequence[Mapping[str, Any]],
    rewards: Sequence[float],
    *,
    candidate_cap: int = 32,
) -> list[dict[str, Any]]:
    """Rerank a fixed unguided terminal support without changing its members."""

    _require(candidate_cap == 32, "terminal rerank candidate cap differs")
    _require(bool(candidates) and len(candidates) == len(rewards), "terminal rerank input differs")
    sequences = [str(row["candidate_sequence"]) for row in candidates]
    _require(len(sequences) == len(set(sequences)), "terminal rerank support is not deduplicated")
    values = [float(value) for value in rewards]
    _require(all(math.isfinite(value) for value in values), "terminal rerank reward is nonfinite")
    rows = []
    for candidate, reward in zip(candidates, values):
        row = dict(candidate)
        row["terminal_rerank_reward"] = reward
        rows.append(row)
    rows.sort(key=lambda row: (-float(row["terminal_rerank_reward"]), str(row["candidate_sequence"])))
    result = rows[:candidate_cap]
    _require(
        {str(row["candidate_sequence"]) for row in result} <= set(sequences),
        "terminal rerank mutated candidate support",
    )
    return result
