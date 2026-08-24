"""Mode-aware scalar-potential guidance invariants for XEditFlow V4."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import torch

from core.route2_legal_xeditflow import FlowState, LegalAction, apply_action
from core.route2_xeditflow_guidance_v3 import (
    deduplicate_terminal_candidates_v3,
    potential_guided_rates_v3,
    smc_importance_transition_v3,
    stratified_resample_v3,
)


class XEditFlowGuidanceV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowGuidanceV4Error(message)


@dataclass(frozen=True)
class SetFlowMixtureStateV4:
    """The SetFlow state plus the mode selected once at trajectory start."""

    flow_state: FlowState
    trajectory_mode_id: int

    def __post_init__(self) -> None:
        _require(
            0 <= int(self.trajectory_mode_id) < 8,
            "XEditFlow V4 trajectory mode is outside the frozen eight modes",
        )


def apply_mode_fixed_action_v4(
    state: SetFlowMixtureStateV4, action: LegalAction
) -> SetFlowMixtureStateV4:
    return SetFlowMixtureStateV4(
        flow_state=apply_action(state.flow_state, action),
        trajectory_mode_id=state.trajectory_mode_id,
    )


def potential_guided_rates_v4(
    base_rates: torch.Tensor,
    legal_mask: torch.Tensor,
    current_potential: torch.Tensor,
    child_potential: torch.Tensor,
    *,
    progress: torch.Tensor,
    beta_max: float,
) -> torch.Tensor:
    """Apply the frozen scalar potential difference; no action-ratio parameter."""

    return potential_guided_rates_v3(
        base_rates,
        legal_mask,
        current_potential,
        child_potential,
        progress=progress,
        beta_max=beta_max,
    )


@dataclass(frozen=True)
class PotentialTransitionSetV4:
    actions: tuple[Any, ...]
    children: tuple[SetFlowMixtureStateV4, ...]
    base_rates: tuple[float, ...]
    current_potential: float
    child_potentials: tuple[float, ...]
    progress: float


def validate_mode_fixed_transition_v4(
    state: SetFlowMixtureStateV4, transition: PotentialTransitionSetV4
) -> None:
    count = len(transition.actions)
    _require(
        count > 0
        and len(transition.children) == count
        and len(transition.base_rates) == count
        and len(transition.child_potentials) == count,
        "XEditFlow V4 transition geometry differs",
    )
    _require(
        all(
            child.trajectory_mode_id == state.trajectory_mode_id
            for child in transition.children
        ),
        "XEditFlow V4 changed latent mode within a trajectory",
    )
    _require(
        math.isfinite(float(transition.current_potential))
        and all(math.isfinite(float(value)) for value in transition.child_potentials),
        "XEditFlow V4 scalar potential is nonfinite",
    )


@dataclass
class MatchedComputeRecordV4:
    source_key: str
    trunk_forwards: int = 0
    mode_forwards: int = 0
    value_forwards: int = 0
    critic_forwards_by_member: list[int] = field(default_factory=lambda: [0, 0, 0])
    candidate_count: int = 0
    trajectory_count: int = 0
    wall_time_seconds: float = 0.0
    peak_vram_mb: float = 0.0
    edit_budget_violation_count: int = 0
    candidate_budget_violation_count: int = 0
    replay_failure_count: int = 0
    numerical_failure_count: int = 0

    def add_critic_forwards(self, member_index: int, count: int = 1) -> None:
        _require(member_index in {0, 1, 2}, "critic ensemble member index differs")
        _require(count >= 0, "critic forward count is negative")
        self.critic_forwards_by_member[member_index] += int(count)

    @property
    def total_forward_equivalents(self) -> int:
        return int(
            self.trunk_forwards
            + self.mode_forwards
            + self.value_forwards
            + sum(self.critic_forwards_by_member)
        )

    def to_dict(self) -> dict[str, Any]:
        integer_counters = (
            self.trunk_forwards,
            self.mode_forwards,
            self.value_forwards,
            *self.critic_forwards_by_member,
            self.candidate_count,
            self.trajectory_count,
            self.edit_budget_violation_count,
            self.candidate_budget_violation_count,
            self.replay_failure_count,
            self.numerical_failure_count,
        )
        _require(min(integer_counters) >= 0, "matched-compute V4 counter is negative")
        _require(self.candidate_count <= 32, "matched-compute V4 candidate cap exceeded")
        _require(self.trajectory_count <= 32, "matched-compute V4 trajectory cap exceeded")
        _require(
            self.total_forward_equivalents <= 320,
            "matched-compute V4 forward ceiling exceeded",
        )
        _require(
            math.isfinite(self.wall_time_seconds)
            and self.wall_time_seconds >= 0.0
            and math.isfinite(self.peak_vram_mb)
            and self.peak_vram_mb >= 0.0,
            "matched-compute V4 resource measurement is invalid",
        )
        return {
            "schema_version": "MatchedComputeRecordV4",
            "source_key": self.source_key,
            "trunk_forwards": self.trunk_forwards,
            "mode_forwards": self.mode_forwards,
            "value_forwards": self.value_forwards,
            "critic_forwards_by_member": list(self.critic_forwards_by_member),
            "critic_forward_total": sum(self.critic_forwards_by_member),
            "all_network_forwards_separately_charged": True,
            "total_forward_equivalents": self.total_forward_equivalents,
            "forward_equivalent_ceiling": 320,
            "candidate_count": self.candidate_count,
            "candidate_cap": 32,
            "trajectory_count": self.trajectory_count,
            "trajectory_cap": 32,
            "wall_time_seconds": self.wall_time_seconds,
            "peak_vram_mb": self.peak_vram_mb,
            "failure_counters": {
                "edit_budget_violation_count": self.edit_budget_violation_count,
                "candidate_budget_violation_count": self.candidate_budget_violation_count,
                "replay_failure_count": self.replay_failure_count,
                "numerical_failure_count": self.numerical_failure_count,
            },
        }


def run_mode_fixed_scalar_potential_smc_v4(
    root_state: FlowState,
    transition_provider: Callable[
        [SetFlowMixtureStateV4], PotentialTransitionSetV4
    ],
    *,
    particle_mode_ids: Sequence[int],
    particle_seeds: Sequence[int],
    resampling_seed: int,
    beta_max: float,
    candidate_sequence: Callable[[FlowState], str],
) -> dict[str, Any]:
    """Replayable 32-particle SMC with mode carried through every lineage."""

    _require(
        len(particle_mode_ids) == len(particle_seeds) == 32,
        "formal XEditFlow V4 SMC requires exactly 32 modes and seeds",
    )
    _require(
        len(set(int(seed) for seed in particle_seeds)) == 32,
        "XEditFlow V4 particle seed stream is duplicated",
    )
    _require(
        set(int(mode) for mode in particle_mode_ids) == set(range(8)),
        "XEditFlow V4 initial particles do not cover all eight modes",
    )
    generators = [np.random.default_rng(int(seed)) for seed in particle_seeds]
    particles = [
        SetFlowMixtureStateV4(root_state, int(mode)) for mode in particle_mode_ids
    ]
    initial_modes = tuple(state.trajectory_mode_id for state in particles)
    trajectories: list[list[Any]] = [[] for _ in particles]
    log_weights = np.full(32, -math.log(32.0), dtype=float)
    resampling_events: list[dict[str, Any]] = []
    step = 0
    while not all(state.flow_state.terminal_cause is not None for state in particles):
        active = [
            index
            for index, state in enumerate(particles)
            if state.flow_state.terminal_cause is None
        ]
        _require(step <= 5, "XEditFlow V4 trajectory exceeded the frozen edit ceiling")
        for index in active:
            transition = transition_provider(particles[index])
            validate_mode_fixed_transition_v4(particles[index], transition)
            rates = torch.tensor([transition.base_rates], dtype=torch.float64)
            legal = torch.ones_like(rates, dtype=torch.bool)
            current = torch.tensor([transition.current_potential], dtype=torch.float64)
            children = torch.tensor([transition.child_potentials], dtype=torch.float64)
            progress = torch.tensor([transition.progress], dtype=torch.float64)
            uniform = torch.tensor([generators[index].random()], dtype=torch.float64)
            sampled = smc_importance_transition_v3(
                rates,
                legal,
                current,
                children,
                progress=progress,
                beta_max=beta_max,
                uniforms=uniform,
            )
            choice = int(sampled["choice_indices"][0])
            particles[index] = transition.children[choice]
            trajectories[index].append(transition.actions[choice])
            log_weights[index] += float(sampled["log_weight_increment"][0])
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
            "candidate_sequence": candidate_sequence(state.flow_state),
            "log_weight": float(log_weights[index]),
            "trajectory_actions": list(trajectories[index]),
            "trajectory_mode_id": state.trajectory_mode_id,
            "particle_slot": index,
        }
        for index, state in enumerate(particles)
    ]
    candidates = deduplicate_terminal_candidates_v3(terminals, candidate_cap=32)
    for candidate in candidates:
        sequence = candidate["candidate_sequence"]
        candidate["contributing_mode_ids"] = sorted(
            {
                int(row["trajectory_mode_id"])
                for row in terminals
                if row["candidate_sequence"] == sequence
            }
        )
    return {
        "schema_version": "route_a_v3_route2_xeditflow_mode_fixed_potential_smc.v4",
        "status": "XEDITFLOW_V4_SMC_COMPLETE",
        "particle_count": 32,
        "candidate_cap": 32,
        "completed_steps": step,
        "initial_particle_mode_ids": list(initial_modes),
        "setflow_mode_is_fixed_trajectory_state": True,
        "mode_resampled_per_step": False,
        "resampling_copies_complete_mode_state": True,
        "resampling_events": resampling_events,
        "candidates": candidates,
        "fixed_seed_replayable": True,
        "proposal": "V4_SETFLOW_MODE_CONDITIONED_BASE_TRANSITION",
        "incremental_importance_weight": (
            "EXP_BETA_TIMES_SINGLE_SCALAR_POTENTIAL_DIFFERENCE"
        ),
        "free_action_ratio_head_used": False,
    }
