"""Batched, mode-fixed scalar-potential SMC for XEditFlow V4."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from core.route2_legal_xeditflow import FlowState, LegalAction, apply_action, legal_actions
from core.route2_source_token_cache_v3 import SourceTokenCacheIndexV3
from core.route2_xeditflow_guidance_v3 import (
    beta_schedule_v3,
    deduplicate_terminal_candidates_v3,
    stratified_resample_v3,
)
from core.route2_xeditflow_guidance_v4 import MatchedComputeRecordV4, SetFlowMixtureStateV4
from core.route2_xeditsetflow_sampling_v3 import (
    TOKEN,
    SetFlowGenerationMetadataV3,
    collate_generation_states_v3,
)
from core.route2_xeditsetflow_v4 import XEditSetFlowV4, select_trajectory_mode_rates_v4


class XEditFlowSMCRuntimeV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowSMCRuntimeV4Error(message)


@dataclass(frozen=True)
class BatchedModeRateRowV4:
    actions: tuple[LegalAction, ...]
    rates: tuple[float, ...]
    trajectory_mode_id: int


def combine_primary_and_replay_compute_v4(
    primary: Mapping[str, Any],
    replay: Mapping[str, Any],
    *,
    replay_ok: bool,
) -> dict[str, Any]:
    """Count both executions while retaining one 32-particle sample round."""

    _require(
        primary.get("schema_version")
        == replay.get("schema_version")
        == "MatchedComputeRecordV4",
        "V4 primary/replay compute schema differs",
    )
    _require(
        str(primary.get("source_key")) == str(replay.get("source_key")),
        "V4 primary/replay compute source differs",
    )
    _require(
        int(primary.get("trajectory_count", -1))
        == int(replay.get("trajectory_count", -2))
        == 32,
        "V4 primary/replay is not one 32-particle round",
    )
    primary_failures = primary["failure_counters"]
    replay_failures = replay["failure_counters"]
    record = MatchedComputeRecordV4(
        source_key=str(primary["source_key"]),
        trunk_forwards=int(primary["trunk_forwards"])
        + int(replay["trunk_forwards"]),
        mode_forwards=int(primary["mode_forwards"])
        + int(replay["mode_forwards"]),
        value_forwards=int(primary["value_forwards"])
        + int(replay["value_forwards"]),
        critic_forwards_by_member=[
            int(left) + int(right)
            for left, right in zip(
                primary["critic_forwards_by_member"],
                replay["critic_forwards_by_member"],
                strict=True,
            )
        ],
        candidate_count=int(primary["candidate_count"]),
        trajectory_count=32,
        wall_time_seconds=float(primary["wall_time_seconds"])
        + float(replay["wall_time_seconds"]),
        peak_vram_mb=max(
            float(primary.get("peak_vram_mb", 0.0)),
            float(replay.get("peak_vram_mb", 0.0)),
        ),
        edit_budget_violation_count=int(
            primary_failures["edit_budget_violation_count"]
        )
        + int(replay_failures["edit_budget_violation_count"]),
        candidate_budget_violation_count=int(
            primary_failures["candidate_budget_violation_count"]
        )
        + int(replay_failures["candidate_budget_violation_count"]),
        replay_failure_count=int(
            primary_failures["replay_failure_count"]
        )
        + int(replay_failures["replay_failure_count"])
        + int(not replay_ok),
        numerical_failure_count=int(
            primary_failures["numerical_failure_count"]
        )
        + int(replay_failures["numerical_failure_count"]),
    )
    result = record.to_dict()
    result.update(
        {
            "primary_forward_equivalents": int(
                primary["total_forward_equivalents"]
            ),
            "replay_forward_equivalents": int(
                replay["total_forward_equivalents"]
            ),
            "replay_forwards_counted": True,
        }
    )
    return result


def merge_smc_rounds_v4(
    round_results: Sequence[Mapping[str, Any]],
    *,
    source_key: str,
    prior_trunk_forwards: int,
    prior_mode_forwards: int,
    terminal_critic_forwards_by_member: Sequence[int] = (1, 1, 1),
) -> dict[str, Any]:
    """Merge mode-fixed rounds and retain every charged V4 forward."""

    _require(bool(round_results), "V4 SMC round collection is empty")
    _require(
        prior_trunk_forwards >= 0 and prior_mode_forwards >= 0,
        "V4 root-mode prior compute is invalid",
    )
    critic_counts = [int(value) for value in terminal_critic_forwards_by_member]
    _require(
        len(critic_counts) == 3 and min(critic_counts) >= 0,
        "V4 terminal critic forward reservation differs",
    )
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    trunk = int(prior_trunk_forwards)
    mode = int(prior_mode_forwards)
    value = 0
    trajectories = 0
    wall_time = 0.0
    peak_vram = 0.0
    failure_totals = {
        "edit_budget_violation_count": 0,
        "candidate_budget_violation_count": 0,
        "replay_failure_count": 0,
        "numerical_failure_count": 0,
    }
    for result in round_results:
        _require(
            result.get("status") == "XEDITFLOW_V4_SMC_COMPLETE"
            and result.get("source_key") == source_key
            and result.get("setflow_mode_is_fixed_trajectory_state") is True
            and result.get("free_action_ratio_head_used") is False,
            "V4 SMC round identity or mechanism differs",
        )
        compute = result["matched_compute"]
        _require(
            compute.get("schema_version") == "MatchedComputeRecordV4"
            and int(compute.get("trajectory_count", -1)) == 32,
            "V4 SMC round compute identity differs",
        )
        trunk += int(compute["trunk_forwards"])
        mode += int(compute["mode_forwards"])
        value += int(compute["value_forwards"])
        for member_index, count in enumerate(
            compute["critic_forwards_by_member"]
        ):
            critic_counts[member_index] += int(count)
        trajectories += int(compute["trajectory_count"])
        wall_time += float(compute["wall_time_seconds"])
        peak_vram = max(peak_vram, float(compute.get("peak_vram_mb", 0.0)))
        for key in failure_totals:
            failure_totals[key] += int(compute["failure_counters"][key])
        for candidate in result["candidates"]:
            grouped.setdefault(str(candidate["candidate_sequence"]), []).append(
                candidate
            )
    candidates = []
    for sequence, members in grouped.items():
        weights = np.asarray(
            [float(row["merged_log_weight"]) for row in members], dtype=float
        )
        _require(
            np.all(np.isfinite(weights)),
            "V4 SMC round candidate mass is nonfinite",
        )
        maximum = float(np.max(weights))
        merged_weight = maximum + math.log(
            float(np.exp(weights - maximum).sum())
        ) - math.log(len(round_results))
        representative = dict(
            max(members, key=lambda row: float(row["merged_log_weight"]))
        )
        representative.update(
            {
                "candidate_sequence": sequence,
                "merged_log_weight": merged_weight,
                "sampling_round_multiplicity": len(members),
                "particle_multiplicity": sum(
                    int(row["particle_multiplicity"]) for row in members
                ),
                "contributing_mode_ids": sorted(
                    {
                        int(mode_id)
                        for row in members
                        for mode_id in row.get("contributing_mode_ids", ())
                    }
                ),
            }
        )
        candidates.append(representative)
    candidates.sort(
        key=lambda row: (
            -float(row["merged_log_weight"]),
            str(row["candidate_sequence"]),
        )
    )
    candidates = candidates[:32]
    compute = MatchedComputeRecordV4(
        source_key=source_key,
        trunk_forwards=trunk,
        mode_forwards=mode,
        value_forwards=value,
        critic_forwards_by_member=critic_counts,
        candidate_count=len(candidates),
        trajectory_count=trajectories,
        wall_time_seconds=wall_time,
        peak_vram_mb=peak_vram,
        **failure_totals,
    ).to_dict()
    return {
        "source_key": source_key,
        "sampling_round_count": len(round_results),
        "particle_count_per_round": 32,
        "trajectory_count": trajectories,
        "candidates": candidates,
        "matched_compute": compute,
        "root_mode_prior_forwards_counted": True,
        "terminal_critic_forwards_reserved_by_member": critic_counts,
        "remaining_forward_equivalents": 320
        - int(compute["total_forward_equivalents"]),
    }


def sample_mode_base_proposal_v4(
    rows: Sequence[BatchedModeRateRowV4], *, uniforms: Sequence[float]
) -> list[int]:
    _require(
        bool(rows) and len(rows) == len(uniforms),
        "V4 base proposal rows and uniforms differ",
    )
    choices: list[int] = []
    for row, uniform in zip(rows, uniforms, strict=True):
        rates = np.asarray(row.rates, dtype=float)
        _require(
            len(row.actions) == len(rates) > 0
            and np.all(np.isfinite(rates))
            and np.all(rates > 0.0),
            "V4 base proposal contains an invalid legal rate",
        )
        _require(
            math.isfinite(float(uniform)) and 0.0 <= float(uniform) < 1.0,
            "V4 base proposal uniform differs",
        )
        cumulative = np.cumsum(rates) / float(np.sum(rates))
        choices.append(
            min(
                int(np.searchsorted(cumulative, float(uniform), side="right")),
                len(rates) - 1,
            )
        )
    return choices


def scalar_potential_mode_rate_maps_v4(
    state: FlowState,
    rate_provider: Callable[
        [Sequence[SetFlowMixtureStateV4]], Sequence[BatchedModeRateRowV4]
    ],
    value_provider: Callable[[Sequence[SetFlowMixtureStateV4]], Sequence[float]],
    *,
    beta_max: float,
    mode_head_count: int = 8,
    value_forward_batch_size: int = 32,
) -> tuple[tuple[dict[LegalAction, float], ...], int]:
    """Evaluate exact guided rates for all modes in one shared state batch."""

    _require(state.terminal_cause is None, "V4 exact rate-map state is terminal")
    _require(mode_head_count == 8, "V4 exact rate-map mode count differs")
    _require(
        value_forward_batch_size == 32,
        "V4 exact rate-map value batch size differs",
    )
    hard_legal = tuple(legal_actions(state))
    _require(bool(hard_legal), "V4 exact rate-map has no legal action")
    current_states = [
        SetFlowMixtureStateV4(state, mode_id)
        for mode_id in range(mode_head_count)
    ]
    rows = list(rate_provider(current_states))
    _require(
        len(rows) == mode_head_count,
        "V4 exact rate-map provider count differs",
    )
    value_states: list[SetFlowMixtureStateV4] = []
    for mode_id, (current, row) in enumerate(
        zip(current_states, rows, strict=True)
    ):
        _require(
            row.trajectory_mode_id == mode_id
            and row.actions == hard_legal
            and len(row.rates) == len(hard_legal),
            "V4 exact rate-map action or mode bundle differs",
        )
        value_states.append(current)
        value_states.extend(
            SetFlowMixtureStateV4(
                apply_action(state, action), mode_id
            )
            for action in hard_legal
        )
    values: list[float] = []
    value_forward_calls = 0
    for start in range(0, len(value_states), value_forward_batch_size):
        values.extend(
            float(value)
            for value in value_provider(
                value_states[start : start + value_forward_batch_size]
            )
        )
        value_forward_calls += 1
    stride = len(hard_legal) + 1
    _require(
        len(values) == mode_head_count * stride
        and all(math.isfinite(float(value)) for value in values),
        "V4 exact rate-map scalar values differ",
    )
    budget = state.edit_count + state.remaining_budget
    _require(budget in {1, 3, 5}, "V4 exact rate-map budget differs")
    beta = float(
        beta_schedule_v3(
            torch.tensor(state.edit_count / budget, dtype=torch.float64),
            beta_max=beta_max,
        )
    )
    result = []
    for mode_id, row in enumerate(rows):
        offset = mode_id * stride
        current_value = float(values[offset])
        child_values = values[offset + 1 : offset + stride]
        rates = {}
        for action, base_rate, child_value in zip(
            hard_legal, row.rates, child_values, strict=True
        ):
            rate = float(base_rate) * math.exp(
                beta * (float(child_value) - current_value)
            )
            _require(
                math.isfinite(rate) and rate > 0.0,
                "V4 exact guided legal rate is invalid",
            )
            rates[action] = rate
        result.append(rates)
    return tuple(result), value_forward_calls


def run_batched_mode_fixed_potential_smc_v4(
    root_state: FlowState,
    rate_provider: Callable[
        [Sequence[SetFlowMixtureStateV4]], Sequence[BatchedModeRateRowV4]
    ],
    value_provider: Callable[[Sequence[SetFlowMixtureStateV4]], Sequence[float]],
    *,
    source_key: str,
    particle_mode_ids: Sequence[int],
    particle_seeds: Sequence[int],
    resampling_seed: int,
    beta_max: float,
    mode_head_count: int = 8,
) -> dict[str, Any]:
    """Run exactly 32 particles while carrying mode through every lineage."""

    _require(root_state.terminal_cause is None, "V4 SMC root is already terminal")
    root_budget = root_state.edit_count + root_state.remaining_budget
    _require(root_budget in {1, 3, 5}, "V4 SMC root budget differs")
    _require(
        len(particle_mode_ids) == len(particle_seeds) == 32,
        "V4 SMC requires exactly 32 particles",
    )
    _require(
        len(set(int(seed) for seed in particle_seeds)) == 32,
        "V4 SMC particle seed stream is duplicated",
    )
    _require(mode_head_count == 8, "V4 SMC mode count differs from the frozen eight")
    _require(
        set(int(mode) for mode in particle_mode_ids) == set(range(mode_head_count)),
        "V4 SMC initial particles do not cover every latent mode",
    )
    generators = [np.random.default_rng(int(seed)) for seed in particle_seeds]
    particles = [
        SetFlowMixtureStateV4(root_state, int(mode)) for mode in particle_mode_ids
    ]
    trajectories: list[list[str]] = [[] for _ in particles]
    log_weights = np.full(32, -math.log(32.0), dtype=float)
    trunk_forwards = 0
    mode_forwards = 0
    value_forwards = 0
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
        _require(step < 5, "V4 SMC active trajectory exceeded the five-edit ceiling")
        active_states = [particles[index] for index in active]
        rows = list(rate_provider(active_states))
        trunk_forwards += 1
        mode_forwards += mode_head_count
        _require(
            len(rows) == len(active_states),
            "V4 batched base-rate provider count differs",
        )
        for state, row in zip(active_states, rows, strict=True):
            hard_legal = tuple(legal_actions(state.flow_state))
            _require(
                row.trajectory_mode_id == state.trajectory_mode_id,
                "V4 rate provider changed the trajectory mode",
            )
            _require(
                len(row.actions) == len(set(row.actions))
                and set(row.actions) == set(hard_legal),
                "V4 rate provider does not cover exactly hard-legal actions",
            )
        choices = sample_mode_base_proposal_v4(
            rows, uniforms=[generators[index].random() for index in active]
        )
        children = [
            SetFlowMixtureStateV4(
                apply_action(state.flow_state, row.actions[choice]),
                state.trajectory_mode_id,
            )
            for state, row, choice in zip(active_states, rows, choices, strict=True)
        ]
        potentials = list(value_provider(active_states + children))
        value_forwards += 1
        _require(
            len(potentials) == 2 * len(active_states),
            "V4 batched scalar potential count differs",
        )
        _require(
            all(math.isfinite(float(value)) for value in potentials),
            "V4 batched scalar potential is nonfinite",
        )
        current_values = potentials[: len(active_states)]
        child_values = potentials[len(active_states) :]
        progress = torch.tensor(
            [state.flow_state.edit_count / root_budget for state in active_states],
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
                if row["candidate_sequence"] == sequence
            }
        )
    compute = MatchedComputeRecordV4(
        source_key=source_key,
        trunk_forwards=trunk_forwards,
        mode_forwards=mode_forwards,
        value_forwards=value_forwards,
        candidate_count=len(candidates),
        trajectory_count=32,
        wall_time_seconds=time.time() - started,
        edit_budget_violation_count=sum(
            particle.flow_state.edit_count > root_budget for particle in particles
        ),
        candidate_budget_violation_count=int(len(candidates) > 32),
    )
    return {
        "schema_version": "route_a_v3_route2_xeditflow_batched_mode_fixed_smc.v4",
        "status": "XEDITFLOW_V4_SMC_COMPLETE",
        "source_key": source_key,
        "particle_count": 32,
        "candidate_cap": 32,
        "completed_steps": step,
        "initial_particle_mode_ids": [int(mode) for mode in particle_mode_ids],
        "setflow_mode_is_fixed_trajectory_state": True,
        "mode_resampled_per_step": False,
        "resampling_copies_complete_mode_state": True,
        "candidates": candidates,
        "resampling_events": resampling_events,
        "matched_compute": compute.to_dict(),
        "hard_legality_rate": 1.0,
        "free_action_ratio_head_used": False,
        "incremental_importance_weight": (
            "EXP_BETA_TIMES_SINGLE_SCALAR_POTENTIAL_DIFFERENCE"
        ),
    }


class SetFlowModeRateProviderV4:
    """CUDA/BF16 provider that selects the trajectory's fixed V4 mode."""

    def __init__(
        self,
        *,
        setflow_model: XEditSetFlowV4,
        metadata: SetFlowGenerationMetadataV3,
        source_cache: SourceTokenCacheIndexV3,
        device: torch.device,
    ) -> None:
        _require(device.type == "cuda", "formal V4 SMC providers require CUDA")
        _require(setflow_model.mode_count == 8, "formal V4 SetFlow mode count differs")
        self.setflow_model = setflow_model
        self.metadata = metadata
        self.source_cache = source_cache
        self.device = device

    def _batch(
        self, states: Sequence[SetFlowMixtureStateV4]
    ) -> dict[str, torch.Tensor]:
        batch = collate_generation_states_v3(
            [state.flow_state for state in states],
            [self.metadata] * len(states),
            source_cache=self.source_cache,
        )
        return {
            key: value.to(self.device, non_blocking=True)
            for key, value in batch.items()
        }

    @torch.no_grad()
    def rates(
        self, states: Sequence[SetFlowMixtureStateV4]
    ) -> list[BatchedModeRateRowV4]:
        _require(bool(states), "formal V4 rate provider received no states")
        self.setflow_model.eval()
        batch = self._batch(states)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.setflow_model(batch)
        mode_ids = torch.tensor(
            [state.trajectory_mode_id for state in states],
            dtype=torch.long,
            device=self.device,
        )
        rates = select_trajectory_mode_rates_v4(output["mode_rates"], mode_ids)
        masks = output["legal_action_mask"]
        padded_length = (rates.shape[1] - 1) // 4
        result: list[BatchedModeRateRowV4] = []
        for row_index, state in enumerate(states):
            actions = tuple(legal_actions(state.flow_state))
            values: list[float] = []
            for action in actions:
                flat = (
                    padded_length * 4
                    if action.kind == "STOP"
                    else int(action.position) * 4 + TOKEN[str(action.alt_base)]
                )
                _require(
                    bool(masks[row_index, flat].item()),
                    "formal V4 SMC model masked a hard-legal action",
                )
                value = float(rates[row_index, flat].item())
                _require(
                    math.isfinite(value) and value > 0.0,
                    "formal V4 SMC model legal rate is invalid",
                )
                values.append(value)
            result.append(
                BatchedModeRateRowV4(
                    actions=actions,
                    rates=tuple(values),
                    trajectory_mode_id=state.trajectory_mode_id,
                )
            )
        return result


class SetFlowModeValueProvidersV4(SetFlowModeRateProviderV4):
    """Formal V4 SetFlow rate and one-scalar value providers."""

    def __init__(self, *, value_model: nn.Module, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.value_model = value_model

    @torch.no_grad()
    def values(self, states: Sequence[SetFlowMixtureStateV4]) -> list[float]:
        _require(bool(states), "formal V4 value provider received no states")
        self.value_model.eval()
        batch = self._batch(states)
        batch["trajectory_mode_ids"] = torch.tensor(
            [state.trajectory_mode_id for state in states],
            dtype=torch.long,
            device=self.device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            values = self.value_model(batch)
        _require(
            values.shape == (len(states),),
            "formal V4 SMC value output differs",
        )
        result = values.float().cpu().tolist()
        _require(
            all(math.isfinite(float(value)) for value in result),
            "formal V4 SMC value is nonfinite",
        )
        return result
