"""SetFlow V4-S1 inference-time temperature sweep runner (Direction E, V5).

Runs the frozen 5x5 (mode-prior temperature, STOP-rate scale) grid from
``core.route2_xeditsetflow_temperature_control_v5`` over one terminal S1
checkpoint on the outcome-free 891x32 Validation generation cohort,
reusing the frozen V4 validation data path (cohort, seeds, metrics) and the
V4 sampler loop with two explicit inference-time post-processors:

- mode priors are tempered BEFORE stratified mode allocation;
- the STOP column of the legal rate vector is scaled BEFORE normalization.

This is an inference-time diagnostic only: it writes its own JSON under
/mnt and never touches the frozen S1 family artifacts or gate outputs.

Usage (V5 prep worktree, after the S1 family reaches a terminal state):

    python3.10 scripts/route_a_v3/run_route2_xeditsetflow_s1_temperature_sweep_v5.py \
        --config <s1 runtime_config.json> \
        --run-id v4_s1_full \
        --checkpoint-pass 10 \
        --physical-gpu-index 6 \
        --output-json /mnt/.../audits/xeditsetflow_v4/s1_temperature_sweep_457e15ae.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

WORKTREE = Path(__file__).resolve().parents[2]
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from core.route2_legal_xeditflow import STOP, FlowState, LegalAction, apply_action, legal_actions
from core.route2_source_token_cache_v3 import (
    SourceTokenCacheIndexV3,
    load_source_token_cache_v3,
)
from core.route2_xeditsetflow_sampling_v3 import (
    SetFlowGenerationMetadataV3,
    collate_generation_states_v3,
    build_generation_metadata_v3,
)
from core.route2_xeditsetflow_sampling_v4 import (
    select_trajectory_mode_rates_v4,
    stratified_trajectory_mode_ids_v4,
)
from core.route2_xeditsetflow_temperature_control_v5 import (
    frozen_temperature_sweep_v5,
    temper_mode_prior_v5,
)
from scripts.route_a_v3.validate_route2_xeditsetflow_s1_checkpoint import (
    _read_json,
    _read_jsonl,
    _move,
    load_checkpoint_s1,
)
from core.route2_legal_xeditflow import initial_state
from scripts.route_a_v3.evaluate_route2_generation_v1 import (
    evaluate_generation,
    load_source_manifest,
    measured_neighborhood_metrics,
    validate_measured_pool,
)
from scripts.route_a_v3.run_route2_base_flow_g0_validation_v1 import load_sources

BASE = "ACGU"
SWEEP_SCHEMA = "route_a_v3_route2_xeditsetflow_s1_temperature_sweep_v5.v1"


class SetFlowTemperatureSweepV5Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SetFlowTemperatureSweepV5Error(message)


def stop_column_index_v5(flat_rate_width: int) -> int:
    """Flat index of the STOP column for a (positions x 4 + 1) rate vector."""

    _require(flat_rate_width >= 5, "flat rate vector is too narrow")
    _require((flat_rate_width - 1) % 4 == 0, "flat rate width is not 4k+1")
    return flat_rate_width - 1


@torch.no_grad()
def sample_many_setflow_v5(
    model,
    roots: Sequence[FlowState],
    metadata: Sequence[SetFlowGenerationMetadataV3],
    mode_ids: Sequence[int],
    seeds: Sequence[int],
    *,
    source_cache: SourceTokenCacheIndexV3,
    device: torch.device,
    forward_batch_size: int,
    stop_rate_scale: float = 1.0,
) -> list[tuple[FlowState, tuple[str, ...], int]]:
    """V4 sampler loop with one explicit STOP-column rescale (Direction E).

    Identical control flow to ``sample_many_setflow_v4``; the only change is
    multiplying the STOP column of the masked legal rates by
    ``stop_rate_scale`` before normalization. At ``stop_rate_scale == 1.0``
    the arithmetic is identical to the frozen V4 sampler.
    """

    _require(device.type == "cuda", "temperature sweep sampling requires CUDA")
    _require(
        math.isfinite(stop_rate_scale) and stop_rate_scale > 0.0,
        "STOP rate scale must be finite and positive",
    )
    model.eval()
    states = list(roots)
    generators = [random.Random(int(seed)) for seed in seeds]
    action_ids: list[list[str]] = [[] for _ in roots]
    forward_counts = [0 for _ in roots]
    while True:
        active = [
            index for index, state in enumerate(states)
            if state.terminal_cause is None
        ]
        if not active:
            break
        for start in range(0, len(active), forward_batch_size):
            indices = active[start : start + forward_batch_size]
            batch = _move(
                collate_generation_states_v3(
                    [states[index] for index in indices],
                    [metadata[index] for index in indices],
                    source_cache=source_cache,
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
            weights = torch.where(
                masks, rates.double(), torch.zeros_like(rates, dtype=torch.float64)
            )
            if stop_rate_scale != 1.0:
                stop_column = stop_column_index_v5(int(rates.shape[1]))
                stop_mask = masks[:, stop_column]
                _require(
                    bool((weights[:, stop_column][stop_mask] > 0).all().item()),
                    "STOP rate column is degenerate under scaling",
                )
                weights[:, stop_column] = weights[:, stop_column] * stop_rate_scale
            totals = weights.sum(dim=1)
            _require(
                bool(torch.isfinite(weights).all().item())
                and bool((totals > 0).all().item()),
                "V5 sweep produced an invalid exit-rate distribution",
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
                "V5 sweep sampled a masked action",
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
                    "V5 sweep selected an action outside hard legality",
                )
                states[trajectory_index] = apply_action(state, action)
                action_ids[trajectory_index].append(action.action_id)
                forward_counts[trajectory_index] += 1
    return [
        (states[index], tuple(action_ids[index]), forward_counts[index])
        for index in range(len(states))
    ]


def _sweep_cell(
    model,
    *,
    config: Mapping[str, Any],
    run_id: str,
    checkpoint,
    sources: Sequence[Mapping[str, Any]],
    source_roots: Sequence[FlowState],
    source_metadata: Sequence[SetFlowGenerationMetadataV3],
    priors: Sequence[Sequence[float]],
    cache: SourceTokenCacheIndexV3,
    device: torch.device,
    temperature: float,
    stop_scale: float,
) -> dict[str, Any]:
    """Run one (temperature, stop_scale) cell and return its metric record."""

    validation_generation = config["validation_generation"]
    tempered = [
        temper_mode_prior_v5(prior, temperature=temperature) for prior in priors
    ]
    roots: list[FlowState] = []
    trajectory_metadata: list[SetFlowGenerationMetadataV3] = []
    mode_ids: list[int] = []
    seeds: list[int] = []
    source_indices: list[int] = []
    decoder_seed_base = int(validation_generation["decoder_seed_base"])
    for source_index, (root, metadata, prior) in enumerate(
        zip(source_roots, source_metadata, tempered, strict=True)
    ):
        allocated_modes = stratified_trajectory_mode_ids_v4(prior)
        _require(
            len(allocated_modes) == 32,
            "V5 sweep source trajectory budget changed",
        )
        for trajectory_slot, mode_id in enumerate(allocated_modes):
            roots.append(root)
            trajectory_metadata.append(metadata)
            mode_ids.append(mode_id)
            seeds.append(
                decoder_seed_base + source_index * 1_000_003 + trajectory_slot
            )
            source_indices.append(source_index)
    sampled = sample_many_setflow_v5(
        model,
        roots,
        trajectory_metadata,
        mode_ids,
        seeds,
        source_cache=cache,
        device=device,
        forward_batch_size=64,
        stop_rate_scale=stop_scale,
    )
    method_id = (
        f"temperature_sweep_v5_t{temperature}_s{stop_scale}_{run_id}"
    )
    candidates = []
    for trajectory_index, (terminal, _actions, _forwards) in enumerate(sampled):
        source = sources[source_indices[trajectory_index]]
        candidates.append(
            {
                "method_id": method_id,
                "source_key": source["source_key"],
                "candidate_sequence": terminal.current_sequence,
                "terminal_cause": terminal.terminal_cause,
                "edit_count": terminal.edit_count,
                "trajectory_mode_id": mode_ids[trajectory_index],
                "trajectory_seed": seeds[trajectory_index],
                "generator_nfe": _forwards,
                "trunk_forwards": _forwards,
                "mode_head_forwards": _forwards * model.mode_count,
                "critic_forwards": 0,
                "independent_evaluator_forwards": 0,
                "generated_candidate_grants_canonical_credit": False,
            }
        )
    manifest = load_source_manifest(Path(config["source_eligibility_manifest"]))
    generation = evaluate_generation(manifest, candidates)
    measured_rows = _read_jsonl(Path(config["measured_neighborhood_path"]))
    validate_measured_pool(measured_rows, "DEVELOPMENT", "CLOSED")
    measured = measured_neighborhood_metrics(
        manifest,
        candidates,
        measured_rows,
        k=int(validation_generation["measured_top_k"]),
        candidate_support_mode="OPEN_GENERATED_SUPPORT",
    )
    return {
        "mode_prior_temperature": temperature,
        "stop_rate_scale": stop_scale,
        "method_id": method_id,
        "source_macro_unique_candidate_rate": generation[
            "source_macro_unique_candidate_rate"
        ],
        "source_macro_candidate_recovery_rate": measured[
            "source_macro_candidate_recovery_rate"
        ],
        "source_macro_measured_top_k_recovery_at_k": measured[
            "source_macro_measured_top_k_recovery_at_k"
        ],
        "legality_rate": generation.get("legality_rate"),
        "budget_violation_count": generation.get("budget_violation_count"),
        "trajectory_count": len(candidates),
    }


def run_sweep(
    config: Mapping[str, Any],
    *,
    run_id: str,
    checkpoint_pass: int,
    physical_gpu_index: int,
    output_json: Path,
) -> dict[str, Any]:
    """Execute the frozen sweep grid over one S1 checkpoint (inference only)."""

    _require(not output_json.exists(), f"sweep output already exists: {output_json}")
    _require(
        not os_environ_cuda_visible(),
        "CUDA_VISIBLE_DEVICES remapping is forbidden",
    )
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    device = torch.device(f"cuda:{physical_gpu_index}")
    torch.cuda.set_device(device)
    device_name = torch.cuda.get_device_name(device)
    _require("A100" in device_name, "selected GPU is not an A100")
    _require(torch.cuda.is_bf16_supported(), "BF16 is unavailable on selected GPU")

    model, checkpoint, _training_summary = load_checkpoint_s1(
        config,
        run_id=run_id,
        checkpoint_pass=checkpoint_pass,
        device=device,
    )
    validation_rows = _read_jsonl(Path(config["validation_projection_path"]))
    cache = SourceTokenCacheIndexV3(
        load_source_token_cache_v3(Path(config["source_token_cache_path"]))
    )
    sources = load_sources(Path(config["source_eligibility_manifest"]))
    _require(len(sources) == 891, "V5 sweep source cohort changed")
    source_metadata = build_generation_metadata_v3(
        sources, validation_rows, checkpoint["vocabs"]
    )
    source_roots = [
        initial_state(
            source["source_sequence"],
            budget=int(source["edit_budget"]),
            assay_id=str(source["assay_id"]),
            context_id=str(source["biological_context_id"]),
        )
        for source in sources
    ]
    from core.route2_xeditsetflow_sampling_v4 import root_mode_priors_v4

    priors, _compute = root_mode_priors_v4(
        model,
        source_roots,
        source_metadata,
        source_cache=cache,
        device=device,
        forward_batch_size=64,
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    cells: list[dict[str, Any]] = []
    for temperature, stop_scale in frozen_temperature_sweep_v5():
        cell = _sweep_cell(
            model,
            config=config,
            run_id=run_id,
            checkpoint=checkpoint,
            sources=sources,
            source_roots=source_roots,
            source_metadata=source_metadata,
            priors=priors,
            cache=cache,
            device=device,
            temperature=temperature,
            stop_scale=stop_scale,
        )
        cells.append(cell)
        print(json.dumps(cell, sort_keys=True), flush=True)
    result = {
        "schema_version": SWEEP_SCHEMA,
        "status": "XEDITSETFLOW_V4_S1_TEMPERATURE_SWEEP_V5_COMPLETE",
        "run_id": run_id,
        "checkpoint_pass": int(checkpoint_pass),
        "physical_gpu_index": int(physical_gpu_index),
        "cuda_device_name": device_name,
        "cpu_fallback_used": False,
        "grid": [list(pair) for pair in frozen_temperature_sweep_v5()],
        "cells": cells,
        "elapsed_seconds": time.time() - started,
        "peak_vram_mb": float(
            torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        ),
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
        "diagnostic_only_no_gate_change": True,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    partial = output_json.with_suffix(output_json.suffix + ".partial")
    _require(not partial.exists(), f"stale sweep partial exists: {partial}")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    partial.replace(output_json)
    return result


def os_environ_cuda_visible() -> bool:
    import os

    return bool(os.environ.get("CUDA_VISIBLE_DEVICES"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--run-id", required=True, choices=("v4_s1_full", "v4_s1_single_mode")
    )
    parser.add_argument("--checkpoint-pass", required=True, type=int, choices=(4, 6, 8, 10))
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    arguments = parser.parse_args()
    config = _read_json(arguments.config)
    try:
        result = run_sweep(
            config,
            run_id=arguments.run_id,
            checkpoint_pass=arguments.checkpoint_pass,
            physical_gpu_index=arguments.physical_gpu_index,
            output_json=arguments.output_json,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "schema_version": SWEEP_SCHEMA,
                    "status": "XEDITSETFLOW_V4_S1_TEMPERATURE_SWEEP_V5_FAILED",
                    "run_id": arguments.run_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "development_test_outcome_reads": 0,
                    "new_final_evaluation_outcome_reads": 0,
                }
            )
        )
        return 1
    print(json.dumps({k: v for k, v in result.items() if k != "cells"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
