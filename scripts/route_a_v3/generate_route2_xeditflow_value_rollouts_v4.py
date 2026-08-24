#!/usr/bin/env python3
"""Generate replay-checked, mode-fixed TRAIN rollouts for XEditFlow V4."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_development_projection_v3 import load_projection_rows
from core.route2_gpu_failure_evidence import cuda_device_observation, write_gpu_failure_evidence
from core.route2_source_token_cache_v3 import SourceTokenCacheIndexV3, load_source_token_cache_v3
from core.route2_xeditflow_gate_v4 import authorize_xeditflow_guidance_v4
from core.route2_xeditflow_value_rollouts_v4 import (
    build_value_train_state_rows_v4,
    flow_state_from_value_row_v4,
    generation_metadata_from_value_row_v4,
    terminal_rollout_row_v4,
    value_rollout_seed_v4,
)
from core.route2_xeditflow_value_training_v4 import BASE_FLOW_SEEDS_V4
from core.route2_xeditsetflow_sampling_v4 import sample_many_setflow_v4
from core.route2_xeditsetflow_training_v4 import (
    setflow_source_records_from_projection_rows_v4,
    setflow_source_vocabs_v4,
)
from scripts.route_a_v3.validate_route2_xeditsetflow_v4_checkpoint import (
    load_checkpoint_v4,
)


class XEditFlowValueRolloutRunnerV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowValueRolloutRunnerV4Error(message)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON root is not an object: {path}")
    return payload


def validate_value_rollout_config_v4(config: Mapping[str, Any]) -> None:
    _require(
        config.get("schema_version")
        == "route_a_v3_route2_xeditflow_value_rollout_config.v4",
        "unexpected V4 value rollout config schema",
    )
    _require(
        int(config.get("base_flow_training_seed", -1)) in BASE_FLOW_SEEDS_V4,
        "V4 value rollout base-flow seed changed",
    )
    _require(
        int(config.get("states_per_source", -1)) == 4,
        "V4 value states-per-source changed",
    )
    _require(
        int(config.get("state_pass_index", -1)) == 0,
        "V4 value state pass changed",
    )
    _require(
        int(config.get("rollouts_per_state_mode", -1)) == 8,
        "V4 value rollout K changed",
    )
    _require(
        int(config.get("sampling_state_batch_size", 0)) > 0,
        "V4 value sampling state batch is invalid",
    )
    _require(
        int(config.get("trajectory_forward_batch_size", 0)) > 0,
        "V4 value trajectory forward batch is invalid",
    )
    _require(
        config.get("fixed_seed_replay_check") is True,
        "V4 value rollout replay check was disabled",
    )
    physical_gpu = int(config.get("physical_gpu_index", -1))
    _require(
        physical_gpu in set(range(6)),
        "V4 value rollout GPU is outside 0-5",
    )
    _require(
        str(config.get("device")) == f"cuda:{physical_gpu}",
        "V4 value rollout device provenance changed",
    )
    _require(
        str(config.get("output_dir", "")).startswith(
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"
        ),
        "V4 value rollout outputs left Route 2 /mnt",
    )
    _require(
        config.get("independent_evaluator_used") is False,
        "independent evaluator entered V4 value rollouts",
    )
    _require(
        config.get("development_test_outcomes_accessed_after_atomic_test") is False,
        "V4 value rollout config reopened Development TEST",
    )
    _require(
        config.get("new_final_evaluation_outcomes_accessed") is False,
        "V4 value rollout config accessed Evaluation",
    )


def _selected_checkpoint_pass_v4(
    confirmation: Mapping[str, Any], *, seed: int
) -> int:
    _require(
        confirmation.get("status") == "XEDITSETFLOW_V4_G0_READY"
        and confirmation.get("required_seeds") == list(BASE_FLOW_SEEDS_V4),
        "V4 value rollouts require exact SetFlow G0 readiness",
    )
    row = confirmation.get("seed_results", {}).get(str(seed), {})
    selected = row.get("selected_checkpoint_pass")
    _require(
        row.get("passed") is True
        and isinstance(selected, int)
        and not isinstance(selected, bool)
        and selected in {4, 6, 8, 10},
        "V4 value rollout seed has no frozen selected checkpoint",
    )
    return int(selected)


def _replay_identity(rows: list[tuple[Any, tuple[str, ...], int]]) -> list[tuple[Any, ...]]:
    return [
        (
            state.current_sequence,
            state.terminal_cause,
            state.source_relative_edits,
            actions,
            forward_count,
        )
        for state, actions, forward_count in rows
    ]


def run(config: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
    validate_value_rollout_config_v4(config)
    _require(
        output_dir == Path(str(config["output_dir"])),
        "V4 value rollout output path differs from frozen config",
    )
    _require(
        not output_dir.exists(),
        f"terminal V4 value rollout output exists: {output_dir}",
    )
    critic_readiness = _json(Path(config["critic_readiness_path"]))
    setflow_confirmation = _json(Path(config["setflow_confirmation_path"]))
    authorization = authorize_xeditflow_guidance_v4(
        critic_readiness, setflow_confirmation
    )
    _require(
        authorization["guidance_authorized"] is True,
        "V4 value rollouts remain blocked before joint readiness",
    )
    seed = int(config["base_flow_training_seed"])
    selected_pass = _selected_checkpoint_pass_v4(
        setflow_confirmation, seed=seed
    )
    runtime_config = _json(Path(config["setflow_runtime_config_path"]))
    _require(
        runtime_config.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_confirmation_runtime.v1"
        and runtime_config.get("run_stage") == "CONFIRMATION"
        and int(runtime_config.get("training_seed", -1)) == seed,
        "V4 value rollout SetFlow runtime identity differs",
    )
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    _require(
        not os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES remapping is forbidden",
    )
    physical_gpu = int(config["physical_gpu_index"])
    device = torch.device(str(config["device"]))
    torch.cuda.set_device(device)
    _require(torch.cuda.is_bf16_supported(), "BF16 is unavailable for V4 rollouts")
    cuda = cuda_device_observation(
        physical_gpu, require_physical_index_match=True
    )
    setflow, checkpoint, _training_summary = load_checkpoint_v4(
        runtime_config,
        run_id="v4_full",
        checkpoint_pass=selected_pass,
        device=device,
    )
    _require(
        int(checkpoint.get("seed", -1)) == seed,
        "V4 value rollout SetFlow checkpoint seed differs",
    )
    train_rows = load_projection_rows(
        [Path(config["train_projection_path"])], allowed_splits=("TRAIN",)
    )
    source_records, source_audit = setflow_source_records_from_projection_rows_v4(
        train_rows
    )
    _require(
        len(source_records) == int(config["expected_train_source_count"]),
        "V4 value rollout TRAIN source count differs",
    )
    vocabs = setflow_source_vocabs_v4(source_records)
    _require(vocabs == checkpoint["vocabs"], "V4 value rollout SetFlow vocabulary differs")
    state_rows = build_value_train_state_rows_v4(
        source_records,
        vocabs,
        base_flow_training_seed=seed,
        state_pass_index=0,
    )
    source_cache = SourceTokenCacheIndexV3(
        load_source_token_cache_v3(Path(config["source_token_cache_path"]))
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    (output_dir / "run_config.json").write_text(
        json.dumps(dict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    state_path = output_dir / "train_state_modes.jsonl"
    terminal_path = output_dir / "terminal_rollouts.private.jsonl"
    with state_path.open("w", encoding="utf-8") as handle:
        for row in state_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    terminal_path.write_text("", encoding="utf-8")
    total_trunk_batch_forwards = 0
    total_trunk_state_forwards = 0
    total_mode_state_forwards = 0
    replay_trunk_batch_forwards = 0
    replay_trunk_state_forwards = 0
    replay_mode_state_forwards = 0
    replay_failures = 0
    terminal_causes: Counter[str] = Counter()
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    state_batch_size = int(config["sampling_state_batch_size"])
    with terminal_path.open("a", encoding="utf-8") as output:
        for state_start in range(0, len(state_rows), state_batch_size):
            active_rows = state_rows[state_start : state_start + state_batch_size]
            roots = []
            metadata = []
            modes = []
            seeds = []
            identities = []
            for state_offset, state_row in enumerate(active_rows):
                state_index = state_start + state_offset
                for rollout_index in range(8):
                    roots.append(flow_state_from_value_row_v4(state_row))
                    metadata.append(generation_metadata_from_value_row_v4(state_row))
                    modes.append(int(state_row["trajectory_mode_id"]))
                    seeds.append(
                        value_rollout_seed_v4(
                            state_index,
                            rollout_index,
                            base_flow_training_seed=seed,
                        )
                    )
                    identities.append((state_row, state_index, rollout_index))
            generated, compute = sample_many_setflow_v4(
                setflow,
                roots,
                metadata,
                modes,
                seeds,
                source_cache=source_cache,
                device=device,
                forward_batch_size=int(config["trajectory_forward_batch_size"]),
            )
            replayed, replay_compute = sample_many_setflow_v4(
                setflow,
                roots,
                metadata,
                modes,
                seeds,
                source_cache=source_cache,
                device=device,
                forward_batch_size=int(config["trajectory_forward_batch_size"]),
            )
            replay_ok = _replay_identity(generated) == _replay_identity(replayed)
            replay_failures += int(not replay_ok)
            _require(replay_ok, "V4 value rollout fixed-seed replay failed")
            total_trunk_batch_forwards += compute.trunk_forward_batch_count
            total_trunk_state_forwards += compute.trunk_forward_state_count
            total_mode_state_forwards += compute.mode_head_forward_state_count
            replay_trunk_batch_forwards += replay_compute.trunk_forward_batch_count
            replay_trunk_state_forwards += replay_compute.trunk_forward_state_count
            replay_mode_state_forwards += replay_compute.mode_head_forward_state_count
            for identity, terminal in zip(identities, generated, strict=True):
                state_row, state_index, rollout_index = identity
                terminal_state, actions, trajectory_forward_count = terminal
                terminal_causes[str(terminal_state.terminal_cause)] += 1
                row = terminal_rollout_row_v4(
                    state_row,
                    state_index=state_index,
                    rollout_index=rollout_index,
                    terminal_state=terminal_state,
                    trajectory_actions=actions,
                    trunk_forwards=trajectory_forward_count,
                    mode_forwards=trajectory_forward_count * setflow.mode_count,
                )
                output.write(json.dumps(row, sort_keys=True) + "\n")
    result = {
        "schema_version": "route_a_v3_route2_xeditflow_value_rollouts.v4",
        "status": "XEDITFLOW_V4_VALUE_ROLLOUTS_COMPLETE_PENDING_CRITIC_SCORING",
        "base_flow_training_seed": seed,
        "selected_setflow_checkpoint_pass": selected_pass,
        "train_source_count": len(source_records),
        "state_mode_count": len(state_rows),
        "rollouts_per_state_mode": 8,
        "terminal_rollout_count": len(state_rows) * 8,
        "trajectory_mode_count": 8,
        "terminal_causes": dict(terminal_causes),
        "primary_compute": {
            "trunk_forward_batch_count": total_trunk_batch_forwards,
            "trunk_forward_state_count": total_trunk_state_forwards,
            "mode_head_forward_state_count": total_mode_state_forwards,
        },
        "replay_compute": {
            "trunk_forward_batch_count": replay_trunk_batch_forwards,
            "trunk_forward_state_count": replay_trunk_state_forwards,
            "mode_head_forward_state_count": replay_mode_state_forwards,
        },
        "fixed_seed_replay_failure_count": replay_failures,
        "fixed_seed_replayable": True,
        "setflow_mode_is_fixed_trajectory_state": True,
        "source_audit": source_audit,
        "wall_time_seconds": time.time() - started,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "physical_gpu_index": physical_gpu,
        "cuda_device": cuda,
        "cpu_fallback_used": False,
        "critic_scoring_performed": False,
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    config = _json(arguments.config)
    try:
        result = run(config, output_dir=arguments.output_dir)
    except Exception as exc:
        write_gpu_failure_evidence(
            arguments.output_dir.with_name(
                arguments.output_dir.name + ".failed.json"
            ),
            config,
            exc,
            entrypoint="generate_route2_xeditflow_value_rollouts_v4",
            evaluation_outcomes_accessed=False,
        )
        raise
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
