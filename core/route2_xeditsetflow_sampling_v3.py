"""Batched, replayable unguided sampling for XEditSetFlow V3."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from core.route2_legal_xeditflow import STOP, FlowState, LegalAction, apply_action, legal_actions
from core.route2_source_token_cache_v3 import SourceTokenCacheIndexV3
from core.route2_xeditsetflow_runtime_v3 import setflow_arm_rates_v3
from core.route2_xeditcritic_training_data_v3 import descriptor_category


TOKEN = {"A": 0, "C": 1, "G": 2, "U": 3}
BASE = "ACGU"
PAD = 4


class XEditSetFlowSamplingV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditSetFlowSamplingV3Error(message)


@dataclass(frozen=True)
class SetFlowGenerationMetadataV3:
    cache_record_id: str
    quantity: int
    measurement: int
    numerator: int
    denominator: int
    assay: int
    context: int
    region: int


def build_generation_metadata_v3(
    sources: Sequence[Mapping[str, Any]],
    validation_projection_rows: Sequence[Mapping[str, Any]],
    vocabs: Mapping[str, Mapping[str, int]],
) -> list[SetFlowGenerationMetadataV3]:
    """Match source cohorts to cached records without reading measured targets."""

    representatives: dict[tuple[str, str, str], tuple[str, Mapping[str, Any]]] = {}
    for row in validation_projection_rows:
        _require(row.get("split") == "VALIDATION", "non-Validation row entered generation metadata")
        if len(row["source_relative_edits"]) > 5:
            continue
        key = (
            str(row["source_sequence"]),
            str(row["endpoint_id"]),
            str(row["biological_context_id"]),
        )
        candidate = (str(row["canonical_record_id"]), row["endpoint_descriptor"])
        if key not in representatives or candidate[0] < representatives[key][0]:
            representatives[key] = candidate
    result: list[SetFlowGenerationMetadataV3] = []
    for source in sources:
        key = (
            str(source["source_sequence"]),
            str(source["endpoint_id"]),
            str(source["biological_context_id"]),
        )
        _require(key in representatives, "generation source lacks an eligible cached Validation record")
        record_id, descriptor = representatives[key]
        region = str(source["region"]).replace("′", "").replace("'", "")
        _require(region in {"5UTR", "3UTR"}, "generation source region is unsupported")
        categories = {
            "quantity": str(descriptor["quantity_family"]),
            "measurement": str(descriptor["measurement_form"]),
            "numerator": descriptor_category(descriptor["numerator_family"]),
            "denominator": descriptor_category(descriptor["denominator_family"]),
            "assay": str(source["assay_id"]),
            "context": str(source["biological_context_id"]),
        }
        result.append(
            SetFlowGenerationMetadataV3(
                cache_record_id=record_id,
                quantity=vocabs["quantity"].get(categories["quantity"], 0),
                measurement=vocabs["measurement"].get(categories["measurement"], 0),
                numerator=vocabs["numerator"].get(categories["numerator"], 0),
                denominator=vocabs["denominator"].get(categories["denominator"], 0),
                assay=vocabs["assay"].get(categories["assay"], 0),
                context=vocabs["context"].get(categories["context"], 0),
                region=0 if region == "5UTR" else 1,
            )
        )
    return result


def collate_generation_states_v3(
    states: Sequence[FlowState],
    metadata: Sequence[SetFlowGenerationMetadataV3],
    *,
    source_cache: SourceTokenCacheIndexV3,
) -> dict[str, torch.Tensor]:
    _require(bool(states) and len(states) == len(metadata), "generation state metadata differs")
    maximum = max(len(state.source_sequence) for state in states)
    width = int(source_cache.payload["embedding_width"])
    source = torch.full((len(states), maximum), PAD, dtype=torch.long)
    current = torch.full((len(states), maximum), PAD, dtype=torch.long)
    padding = torch.ones((len(states), maximum), dtype=torch.bool)
    pretrained = torch.zeros((len(states), maximum, width), dtype=torch.float16)
    for index, (state, meta) in enumerate(zip(states, metadata, strict=True)):
        length = len(state.source_sequence)
        _require(len(state.current_sequence) == length, "generation state length changed")
        source[index, :length] = torch.tensor([TOKEN[base] for base in state.source_sequence])
        current[index, :length] = torch.tensor([TOKEN[base] for base in state.current_sequence])
        padding[index, :length] = False
        cached = source_cache.tokens_for_record(meta.cache_record_id)
        _require(cached.shape[0] == length, "generation source cache does not align")
        pretrained[index, :length] = cached
    return {
        "source_tokens": source,
        "current_tokens": current,
        "padding_mask": padding,
        "source_pretrained_tokens": pretrained,
        "remaining_budget": torch.tensor([state.remaining_budget for state in states]),
        "quantity_ids": torch.tensor([item.quantity for item in metadata]),
        "measurement_ids": torch.tensor([item.measurement for item in metadata]),
        "numerator_ids": torch.tensor([item.numerator for item in metadata]),
        "denominator_ids": torch.tensor([item.denominator for item in metadata]),
        "assay_ids": torch.tensor([item.assay for item in metadata]),
        "context_ids": torch.tensor([item.context for item in metadata]),
        "region_ids": torch.tensor([item.region for item in metadata]),
    }


def _move(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def sample_many_setflow_v3(
    model: nn.Module,
    arm: str,
    roots: Sequence[FlowState],
    metadata: Sequence[SetFlowGenerationMetadataV3],
    seeds: Sequence[int],
    *,
    source_cache: SourceTokenCacheIndexV3,
    device: torch.device,
    forward_batch_size: int,
) -> tuple[list[tuple[FlowState, tuple[str, ...], int]], int]:
    _require(device.type == "cuda", "formal SetFlow sampling requires CUDA")
    _require(len(roots) == len(metadata) == len(seeds) and bool(roots), "generation inputs differ")
    _require(forward_batch_size > 0, "generation forward batch size is invalid")
    model.eval()
    states = list(roots)
    generators = [random.Random(int(seed)) for seed in seeds]
    action_ids: list[list[str]] = [[] for _ in roots]
    forward_counts = [0 for _ in roots]
    model_forward_batches = 0
    while True:
        active = [index for index, state in enumerate(states) if state.terminal_cause is None]
        if not active:
            break
        for start in range(0, len(active), forward_batch_size):
            indices = active[start : start + forward_batch_size]
            active_states = [states[index] for index in indices]
            active_meta = [metadata[index] for index in indices]
            batch = _move(
                collate_generation_states_v3(active_states, active_meta, source_cache=source_cache),
                device,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                rates, masks = setflow_arm_rates_v3(model, arm, batch)
            model_forward_batches += 1
            weights = torch.where(masks, rates.double(), torch.zeros_like(rates, dtype=torch.float64))
            totals = weights.sum(dim=1)
            _require(
                bool(torch.isfinite(weights).all().item())
                and bool(torch.isfinite(totals).all().item())
                and bool((totals > 0).all().item()),
                "SetFlow produced an invalid exit-rate distribution",
            )
            cumulative = weights.cumsum(dim=1) / totals.unsqueeze(1)
            uniforms = torch.tensor(
                [generators[index].random() for index in indices],
                dtype=torch.float64,
                device=device,
            )
            choices = (cumulative < uniforms.unsqueeze(1)).sum(dim=1).clamp_max(rates.shape[1] - 1)
            rows = torch.arange(len(indices), device=device)
            _require(bool(masks[rows, choices].all().item()), "SetFlow sampled a masked action")
            padded_length = (rates.shape[1] - 1) // 4
            for trajectory_index, flat_index in zip(indices, choices.tolist(), strict=True):
                state = states[trajectory_index]
                if flat_index == padded_length * 4:
                    action = LegalAction(STOP)
                else:
                    position, alt_index = divmod(flat_index, 4)
                    action = LegalAction("SUB", position, BASE[alt_index])
                _require(action in legal_actions(state), "SetFlow selected an action outside hard legality")
                child = apply_action(state, action)
                action_ids[trajectory_index].append(action.action_id)
                states[trajectory_index] = child
                forward_counts[trajectory_index] += 1
    return [
        (states[index], tuple(action_ids[index]), forward_counts[index])
        for index in range(len(states))
    ], model_forward_batches


@torch.no_grad()
def setflow_rate_map_v3(
    model: nn.Module,
    arm: str,
    state: FlowState,
    metadata: SetFlowGenerationMetadataV3,
    actions: Sequence[LegalAction],
    *,
    source_cache: SourceTokenCacheIndexV3,
    device: torch.device,
) -> dict[LegalAction, float]:
    batch = _move(collate_generation_states_v3([state], [metadata], source_cache=source_cache), device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        rates, mask = setflow_arm_rates_v3(model, arm, batch)
    length = len(state.source_sequence)
    result: dict[LegalAction, float] = {}
    for action in actions:
        index = length * 4 if action.kind == STOP else int(action.position) * 4 + TOKEN[str(action.alt_base)]
        _require(bool(mask[0, index].item()), "SetFlow rate map rejected a legal action")
        value = float(rates[0, index].item())
        _require(math.isfinite(value) and value > 0.0, "SetFlow legal rate is invalid")
        result[action] = value
    return result
