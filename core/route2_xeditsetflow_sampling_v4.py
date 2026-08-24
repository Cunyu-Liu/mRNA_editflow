"""Trajectory-fixed latent-mode sampling for XEditSetFlow V4."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from core.route2_legal_xeditflow import (
    STOP,
    FlowState,
    LegalAction,
    apply_action,
    legal_actions,
)
from core.route2_source_token_cache_v3 import SourceTokenCacheIndexV3
from core.route2_xeditsetflow_sampling_v3 import (
    SetFlowGenerationMetadataV3,
    collate_generation_states_v3,
)
from core.route2_xeditsetflow_v4 import (
    XEditSetFlowV4,
    select_trajectory_mode_rates_v4,
)


TOKEN = {"A": 0, "C": 1, "G": 2, "U": 3}
BASE = "ACGU"


class XEditSetFlowSamplingV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditSetFlowSamplingV4Error(message)


@dataclass(frozen=True)
class SetFlowSamplingComputeV4:
    trunk_forward_batch_count: int
    trunk_forward_state_count: int
    mode_head_forward_state_count: int


def largest_remainder_mode_counts_v4(
    prior: Sequence[float],
    *,
    trajectory_count: int = 32,
    minimum_per_mode: int = 1,
) -> tuple[int, ...]:
    """Allocate one per mode first, then largest-remainder under the prior."""

    values = [float(value) for value in prior]
    _require(bool(values), "SetFlow V4 mode prior is empty")
    _require(
        all(math.isfinite(value) and value >= 0.0 for value in values),
        "SetFlow V4 mode prior is invalid",
    )
    _require(
        math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-6),
        "SetFlow V4 mode prior is not normalized",
    )
    _require(minimum_per_mode == 1, "SetFlow V4 minimum mode allocation changed")
    _require(
        trajectory_count >= len(values),
        "SetFlow V4 has fewer trajectories than latent modes",
    )
    remaining = trajectory_count - len(values)
    quotas = [value * remaining for value in values]
    extra = [math.floor(quota) for quota in quotas]
    leftover = remaining - sum(extra)
    order = sorted(
        range(len(values)),
        key=lambda index: (-(quotas[index] - extra[index]), index),
    )
    for index in order[:leftover]:
        extra[index] += 1
    counts = tuple(1 + value for value in extra)
    _require(sum(counts) == trajectory_count, "SetFlow V4 mode allocation changed trajectory budget")
    _require(all(value >= 1 for value in counts), "SetFlow V4 omitted a latent mode")
    return counts


def stratified_trajectory_mode_ids_v4(
    prior: Sequence[float], *, trajectory_count: int = 32
) -> tuple[int, ...]:
    """Put one trajectory from every mode first, then append allocated extras."""

    counts = largest_remainder_mode_counts_v4(
        prior, trajectory_count=trajectory_count
    )
    mode_ids = list(range(len(counts)))
    for mode_index, count in enumerate(counts):
        mode_ids.extend([mode_index] * (count - 1))
    _require(len(mode_ids) == trajectory_count, "SetFlow V4 trajectory mode list changed size")
    _require(
        tuple(mode_ids[: len(counts)]) == tuple(range(len(counts))),
        "SetFlow V4 did not allocate every mode first",
    )
    return tuple(mode_ids)


def _move(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def root_mode_priors_v4(
    model: XEditSetFlowV4,
    roots: Sequence[FlowState],
    metadata: Sequence[SetFlowGenerationMetadataV3],
    *,
    source_cache: SourceTokenCacheIndexV3,
    device: torch.device,
    forward_batch_size: int,
) -> tuple[list[tuple[float, ...]], SetFlowSamplingComputeV4]:
    _require(device.type == "cuda", "formal SetFlow V4 prior inference requires CUDA")
    _require(bool(roots) and len(roots) == len(metadata), "SetFlow V4 root metadata differs")
    _require(forward_batch_size > 0, "SetFlow V4 forward batch is invalid")
    model.eval()
    priors: list[tuple[float, ...]] = []
    batches = 0
    states = 0
    for start in range(0, len(roots), forward_batch_size):
        active_roots = roots[start : start + forward_batch_size]
        active_metadata = metadata[start : start + forward_batch_size]
        batch = _move(
            collate_generation_states_v3(
                active_roots, active_metadata, source_cache=source_cache
            ),
            device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch)
        prior = output["mode_prior"].float()
        _require(
            bool(torch.isfinite(prior).all().item())
            and bool(
                torch.allclose(
                    prior.sum(dim=1),
                    torch.ones(prior.shape[0], device=device),
                    atol=1e-6,
                )
            ),
            "SetFlow V4 root mode prior is invalid",
        )
        priors.extend(tuple(float(value) for value in row) for row in prior.cpu())
        batches += 1
        states += len(active_roots)
    return priors, SetFlowSamplingComputeV4(
        trunk_forward_batch_count=batches,
        trunk_forward_state_count=states,
        mode_head_forward_state_count=states * model.mode_count,
    )


@torch.no_grad()
def sample_many_setflow_v4(
    model: XEditSetFlowV4,
    roots: Sequence[FlowState],
    metadata: Sequence[SetFlowGenerationMetadataV3],
    mode_ids: Sequence[int],
    seeds: Sequence[int],
    *,
    source_cache: SourceTokenCacheIndexV3,
    device: torch.device,
    forward_batch_size: int,
) -> tuple[
    list[tuple[FlowState, tuple[str, ...], int]], SetFlowSamplingComputeV4
]:
    """Sample with the declared mode held fixed for the whole trajectory."""

    _require(device.type == "cuda", "formal SetFlow V4 sampling requires CUDA")
    _require(
        len(roots) == len(metadata) == len(mode_ids) == len(seeds) and bool(roots),
        "SetFlow V4 generation inputs differ",
    )
    _require(
        all(0 <= int(mode) < model.mode_count for mode in mode_ids),
        "SetFlow V4 trajectory mode is invalid",
    )
    _require(forward_batch_size > 0, "SetFlow V4 forward batch is invalid")
    model.eval()
    states = list(roots)
    generators = [random.Random(int(seed)) for seed in seeds]
    action_ids: list[list[str]] = [[] for _ in roots]
    forward_counts = [0 for _ in roots]
    batch_count = 0
    state_count = 0
    while True:
        active = [
            index for index, state in enumerate(states) if state.terminal_cause is None
        ]
        if not active:
            break
        for start in range(0, len(active), forward_batch_size):
            indices = active[start : start + forward_batch_size]
            active_states = [states[index] for index in indices]
            active_metadata = [metadata[index] for index in indices]
            batch = _move(
                collate_generation_states_v3(
                    active_states, active_metadata, source_cache=source_cache
                ),
                device,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(batch)
            active_mode_ids = torch.tensor(
                [int(mode_ids[index]) for index in indices],
                dtype=torch.long,
                device=device,
            )
            rates = select_trajectory_mode_rates_v4(
                output["mode_rates"], active_mode_ids
            )
            masks = output["legal_action_mask"]
            batch_count += 1
            state_count += len(indices)
            weights = torch.where(
                masks, rates.double(), torch.zeros_like(rates, dtype=torch.float64)
            )
            totals = weights.sum(dim=1)
            _require(
                bool(torch.isfinite(weights).all().item())
                and bool(torch.isfinite(totals).all().item())
                and bool((totals > 0).all().item()),
                "SetFlow V4 produced an invalid exit-rate distribution",
            )
            cumulative = weights.cumsum(dim=1) / totals.unsqueeze(1)
            uniforms = torch.tensor(
                [generators[index].random() for index in indices],
                dtype=torch.float64,
                device=device,
            )
            choices = (cumulative < uniforms.unsqueeze(1)).sum(dim=1).clamp_max(
                rates.shape[1] - 1
            )
            rows = torch.arange(len(indices), device=device)
            _require(
                bool(masks[rows, choices].all().item()),
                "SetFlow V4 sampled a masked action",
            )
            padded_length = (rates.shape[1] - 1) // 4
            for trajectory_index, flat_index in zip(
                indices, choices.tolist(), strict=True
            ):
                state = states[trajectory_index]
                if flat_index == padded_length * 4:
                    action = LegalAction(STOP)
                else:
                    position, alt_index = divmod(flat_index, 4)
                    action = LegalAction("SUB", position, BASE[alt_index])
                _require(
                    action in legal_actions(state),
                    "SetFlow V4 selected an action outside hard legality",
                )
                states[trajectory_index] = apply_action(state, action)
                action_ids[trajectory_index].append(action.action_id)
                forward_counts[trajectory_index] += 1
    return [
        (states[index], tuple(action_ids[index]), forward_counts[index])
        for index in range(len(states))
    ], SetFlowSamplingComputeV4(
        trunk_forward_batch_count=batch_count,
        trunk_forward_state_count=state_count,
        mode_head_forward_state_count=state_count * model.mode_count,
    )


@torch.no_grad()
def setflow_rate_map_v4(
    model: XEditSetFlowV4,
    state: FlowState,
    metadata: SetFlowGenerationMetadataV3,
    mode_id: int,
    actions: Sequence[LegalAction],
    *,
    source_cache: SourceTokenCacheIndexV3,
    device: torch.device,
) -> dict[LegalAction, float]:
    _require(0 <= mode_id < model.mode_count, "SetFlow V4 rate-map mode is invalid")
    batch = _move(
        collate_generation_states_v3(
            [state], [metadata], source_cache=source_cache
        ),
        device,
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(batch)
    rates = output["mode_rates"][0, mode_id]
    mask = output["legal_action_mask"][0]
    length = len(state.source_sequence)
    result: dict[LegalAction, float] = {}
    for action in actions:
        index = (
            length * 4
            if action.kind == STOP
            else int(action.position) * 4 + TOKEN[str(action.alt_base)]
        )
        _require(bool(mask[index].item()), "SetFlow V4 rate map rejected a legal action")
        value = float(rates[index].item())
        _require(
            math.isfinite(value) and value > 0.0,
            "SetFlow V4 legal rate is invalid",
        )
        result[action] = value
    return result
