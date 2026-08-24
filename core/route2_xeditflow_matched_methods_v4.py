"""Mode-fixed matched controls for the final XEditFlow V4 comparison."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from core.route2_legal_xeditflow import FlowState, LegalAction, apply_action, legal_actions
from core.route2_xeditflow_guidance_v3 import (
    beta_schedule_v3,
    deduplicate_terminal_candidates_v3,
    stratified_resample_v3,
)
from core.route2_xeditflow_guidance_v4 import (
    MatchedComputeRecordV4,
    SetFlowMixtureStateV4,
)
from core.route2_xeditflow_smc_runtime_v4 import (
    BatchedModeRateRowV4,
    sample_mode_base_proposal_v4,
)


MATCHED_CONTROL_METHODS_V4 = {
    "unguided_setflow",
    "first_order_guidance",
    "simple_rate_guidance",
}


class XEditFlowMatchedMethodsV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowMatchedMethodsV4Error(message)


@dataclass(frozen=True)
class CriticRewardBatchV4:
    values: tuple[float, ...]
    forward_batches_by_member: tuple[int, int, int]

    def __post_init__(self) -> None:
        _require(
            all(math.isfinite(float(value)) for value in self.values),
            "V4 critic reward batch is nonfinite",
        )
        _require(
            len(self.forward_batches_by_member) == 3
            and min(self.forward_batches_by_member) >= 0,
            "V4 critic reward member accounting differs",
        )


CriticRewardProviderV4 = Callable[[Sequence[FlowState]], CriticRewardBatchV4]
CriticPotentialProviderV4 = Callable[
    [Sequence[SetFlowMixtureStateV4]], CriticRewardBatchV4
]
ModeRateProviderV4 = Callable[
    [Sequence[SetFlowMixtureStateV4]], Sequence[BatchedModeRateRowV4]
]


def _flow_key(state: FlowState) -> tuple[Any, ...]:
    return (
        state.source_sequence,
        state.current_sequence,
        state.remaining_budget,
        state.terminal_cause,
        state.assay_id,
        state.context_id,
    )


class ExactCriticRewardPotentialV4:
    """Use the current frozen three-member Critic reward as a scalar potential."""

    def __init__(self, reward_provider: CriticRewardProviderV4) -> None:
        self.reward_provider = reward_provider

    def __call__(
        self, states: Sequence[SetFlowMixtureStateV4]
    ) -> CriticRewardBatchV4:
        _require(bool(states), "V4 exact critic potential state batch is empty")
        unique: list[FlowState] = []
        index_by_key: dict[tuple[Any, ...], int] = {}
        for state in states:
            key = _flow_key(state.flow_state)
            if key not in index_by_key:
                index_by_key[key] = len(unique)
                unique.append(state.flow_state)
        scored = self.reward_provider(unique)
        _require(
            len(scored.values) == len(unique),
            "V4 exact critic reward count differs",
        )
        return CriticRewardBatchV4(
            tuple(scored.values[index_by_key[_flow_key(state.flow_state)]] for state in states),
            scored.forward_batches_by_member,
        )


class SourceAnchoredFirstOrderPotentialV4:
    """Frozen source-anchored additive expansion shared across latent modes."""

    def __init__(
        self,
        root_state: FlowState,
        reward_provider: CriticRewardProviderV4,
    ) -> None:
        _require(
            root_state.edit_count == 0 and root_state.terminal_cause is None,
            "V4 first-order anchor is not an active source",
        )
        self.root_state = root_state
        self.reward_provider = reward_provider
        self.source_reward: float | None = None
        self.coefficients: dict[tuple[int, str], float] = {}

    def __call__(
        self, states: Sequence[SetFlowMixtureStateV4]
    ) -> CriticRewardBatchV4:
        _require(bool(states), "V4 first-order potential state batch is empty")
        _require(
            all(
                state.flow_state.source_sequence == self.root_state.source_sequence
                for state in states
            ),
            "V4 first-order potential crossed source identity",
        )
        missing = sorted(
            {
                (int(position), str(alt))
                for state in states
                for position, alt in state.flow_state.source_relative_edits
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
                _require(
                    action in legal_actions(self.root_state),
                    "V4 first-order requested an illegal source edit",
                )
                requests.append(apply_action(self.root_state, action))
            scored = self.reward_provider(requests)
            _require(
                len(scored.values) == len(requests),
                "V4 first-order reward count differs",
            )
            member_calls = scored.forward_batches_by_member
            offset = 0
            if include_source:
                self.source_reward = float(scored.values[0])
                offset = 1
            assert self.source_reward is not None
            for edit, reward in zip(missing, scored.values[offset:], strict=True):
                self.coefficients[edit] = float(reward) - self.source_reward
        values = tuple(
            math.fsum(
                self.coefficients[(int(position), str(alt))]
                for position, alt in state.flow_state.source_relative_edits
            )
            for state in states
        )
        return CriticRewardBatchV4(values, member_calls)


class ZeroCriticPotentialV4:
    def __call__(
        self, states: Sequence[SetFlowMixtureStateV4]
    ) -> CriticRewardBatchV4:
        _require(bool(states), "V4 zero potential state batch is empty")
        return CriticRewardBatchV4(
            tuple(0.0 for _ in states),
            (0, 0, 0),
        )


def run_mode_fixed_matched_control_smc_v4(
    root_state: FlowState,
    rate_provider: ModeRateProviderV4,
    potential_provider: CriticPotentialProviderV4,
    *,
    method_id: str,
    source_key: str,
    particle_mode_ids: Sequence[int],
    particle_seeds: Sequence[int],
    resampling_seed: int,
    beta_max: float,
) -> dict[str, Any]:
    """Run one 32-particle control round with the SetFlow mode fixed."""

    _require(method_id in MATCHED_CONTROL_METHODS_V4, "unknown V4 matched control")
    root_budget = root_state.edit_count + root_state.remaining_budget
    _require(
        root_state.terminal_cause is None
        and root_state.edit_count == 0
        and root_budget in {1, 3, 5},
        "V4 matched-control root differs",
    )
    _require(
        len(particle_mode_ids) == len(particle_seeds) == 32
        and len(set(int(seed) for seed in particle_seeds)) == 32
        and set(int(mode) for mode in particle_mode_ids) == set(range(8)),
        "V4 matched control particle or mode inventory differs",
    )
    particles = [
        SetFlowMixtureStateV4(root_state, int(mode))
        for mode in particle_mode_ids
    ]
    generators = [np.random.default_rng(int(seed)) for seed in particle_seeds]
    trajectories: list[list[str]] = [[] for _ in particles]
    log_weights = np.full(32, -math.log(32.0), dtype=float)
    trunk_forwards = 0
    mode_forwards = 0
    critic_forwards = [0, 0, 0]
    resampling_events: list[dict[str, Any]] = []
    step = 0
    started = time.time()
    while not all(
        particle.flow_state.terminal_cause is not None for particle in particles
    ):
        active = [
            index
            for index, particle in enumerate(particles)
            if particle.flow_state.terminal_cause is None
        ]
        _require(step < 5, "V4 matched-control trajectory exceeded five edits")
        current = [particles[index] for index in active]
        rows = list(rate_provider(current))
        trunk_forwards += 1
        mode_forwards += 8
        _require(
            len(rows) == len(current),
            "V4 matched-control rate provider count differs",
        )
        for state, row in zip(current, rows, strict=True):
            hard_legal = tuple(legal_actions(state.flow_state))
            rates = np.asarray(row.rates, dtype=float)
            _require(
                row.trajectory_mode_id == state.trajectory_mode_id
                and len(row.actions) == len(set(row.actions))
                and set(row.actions) == set(hard_legal)
                and len(rates) == len(hard_legal)
                and np.all(np.isfinite(rates))
                and np.all(rates > 0.0),
                "V4 matched-control legal rate bundle differs",
            )
        choices = sample_mode_base_proposal_v4(
            rows,
            uniforms=[generators[index].random() for index in active],
        )
        children = [
            SetFlowMixtureStateV4(
                apply_action(state.flow_state, row.actions[choice]),
                state.trajectory_mode_id,
            )
            for state, row, choice in zip(current, rows, choices, strict=True)
        ]
        potentials = potential_provider(current + children)
        _require(
            len(potentials.values) == 2 * len(current),
            "V4 matched-control potential count differs",
        )
        if method_id == "unguided_setflow":
            _require(
                potentials.forward_batches_by_member == (0, 0, 0)
                and all(float(value) == 0.0 for value in potentials.values),
                "V4 unguided control received a learned potential",
            )
        for member, count in enumerate(potentials.forward_batches_by_member):
            critic_forwards[member] += int(count)
        current_values = potentials.values[: len(current)]
        child_values = potentials.values[len(current) :]
        betas = beta_schedule_v3(
            torch.tensor(
                [state.flow_state.edit_count / root_budget for state in current],
                dtype=torch.float64,
            ),
            beta_max=float(beta_max),
        ).tolist()
        for offset, particle_index in enumerate(active):
            action = rows[offset].actions[choices[offset]]
            particles[particle_index] = children[offset]
            trajectories[particle_index].append(action.action_id)
            increment = float(betas[offset]) * (
                float(child_values[offset]) - float(current_values[offset])
            )
            _require(math.isfinite(increment), "V4 matched-control weight is nonfinite")
            log_weights[particle_index] += increment
        step += 1
        resampling = stratified_resample_v3(
            log_weights,
            seed=int(resampling_seed) + step,
            threshold=16.0,
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
            "candidate_sequence": particle.flow_state.current_sequence,
            "terminal_cause": particle.flow_state.terminal_cause,
            "edit_count": particle.flow_state.edit_count,
            "log_weight": float(log_weights[index]),
            "trajectory_actions": trajectories[index],
            "trajectory_mode_id": particle.trajectory_mode_id,
            "particle_slot": index,
        }
        for index, particle in enumerate(particles)
    ]
    candidates = deduplicate_terminal_candidates_v3(terminals, candidate_cap=32)
    for candidate in candidates:
        sequence = str(candidate["candidate_sequence"])
        candidate["contributing_mode_ids"] = sorted(
            {
                int(row["trajectory_mode_id"])
                for row in terminals
                if str(row["candidate_sequence"]) == sequence
            }
        )
    compute = MatchedComputeRecordV4(
        source_key=source_key,
        trunk_forwards=trunk_forwards,
        mode_forwards=mode_forwards,
        value_forwards=0,
        critic_forwards_by_member=critic_forwards,
        candidate_count=len(candidates),
        trajectory_count=32,
        wall_time_seconds=time.time() - started,
        edit_budget_violation_count=sum(
            particle.flow_state.edit_count > root_budget for particle in particles
        ),
        candidate_budget_violation_count=int(len(candidates) > 32),
    ).to_dict()
    return {
        "schema_version": "route_a_v3_route2_xeditflow_matched_control_smc.v4",
        "status": "XEDITFLOW_V4_SMC_COMPLETE",
        "method_id": method_id,
        "source_key": source_key,
        "particle_count": 32,
        "candidate_cap": 32,
        "completed_steps": step,
        "initial_particle_mode_ids": [int(value) for value in particle_mode_ids],
        "setflow_mode_is_fixed_trajectory_state": True,
        "mode_resampled_per_step": False,
        "resampling_copies_complete_mode_state": True,
        "candidates": candidates,
        "resampling_events": resampling_events,
        "matched_compute": compute,
        "hard_legality_rate": 1.0,
        "free_action_ratio_head_used": False,
        "potential_kind": {
            "unguided_setflow": "ZERO",
            "first_order_guidance": "SOURCE_ANCHORED_FIRST_ORDER_CRITIC",
            "simple_rate_guidance": "EXACT_CURRENT_CRITIC_REWARD",
        }[method_id],
        "critic_forward_is_not_value_forward": True,
    }
