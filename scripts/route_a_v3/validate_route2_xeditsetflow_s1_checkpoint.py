#!/usr/bin/env python3
"""Evaluate one terminal SetFlow V4 S1 checkpoint under the frozen 891x32 protocol."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_development_projection_v3 import load_projection_rows
from core.route2_gpu_failure_evidence import (
    cuda_device_observation,
)
from core.route2_legal_xeditflow import (
    exact_terminal_distribution,
    initial_state,
    jump_distribution,
)
from core.route2_source_token_cache_v3 import (
    SourceTokenCacheIndexV3,
    assemble_source_token_cache_v3,
    load_source_token_cache_v3,
)
from core.route2_xeditsetflow_s1 import (
    build_setflow_screen_model_s1,
    screen_run_spec_s1,
)
from core.route2_xeditsetflow_gate_s1 import (
    XEditSetFlowGateS1Error,
    require_matched_initialization_evidence_s1,
)
from core.route2_xeditsetflow_sampling_v3 import (
    SetFlowGenerationMetadataV3,
    build_generation_metadata_v3,
)
from core.route2_xeditsetflow_sampling_v4 import (
    root_mode_priors_v4,
    sample_many_setflow_v4,
    setflow_rate_map_v4,
    stratified_trajectory_mode_ids_v4,
)
from core.route2_xeditsetflow_training_v3 import (
    SetMarginalStateDatasetV3,
    collate_setflow_states_v3,
    setflow_records_from_projection_rows,
)
from core.route2_xeditsetflow_v4 import common_set_marginal_loss_v4
from scripts.route_a_v3.evaluate_route2_generation_v1 import (
    evaluate_generation,
    load_source_manifest,
    measured_neighborhood_metrics,
    validate_measured_pool,
)
from scripts.route_a_v3.run_route2_base_flow_g0_validation_v1 import load_sources
from scripts.route_a_v3.train_route2_xeditsetflow_s1 import (
    CONFIRMATION_CONFIG_SCHEMA,
    CONFIRMATION_RUN_ID,
    CONFIRMATION_SEEDS,
    OBJECTIVE_IDENTITY,
    OBJECTIVE_WEIGHT,
    SCREEN_CONFIG_SCHEMA,
    require_s1_launch_authorization,
    require_s1_confirmation_launch_authorization,
)


class SetFlowCheckpointValidationS1Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SetFlowCheckpointValidationS1Error(message)


def _require_matched_initialization_s1(
    payload: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    try:
        return require_matched_initialization_evidence_s1(payload, label=label)
    except XEditSetFlowGateS1Error as error:
        raise SetFlowCheckpointValidationS1Error(str(error)) from error


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    _require(bool(rows), f"validation input is empty: {path}")
    return rows


def _write_atomic_terminal_s1(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), f"terminal SetFlow V4 S1 artifact already exists: {path}")
    partial = path.with_suffix(path.suffix + ".partial")
    _require(
        not partial.exists(),
        f"partial SetFlow V4 S1 terminal artifact already exists: {partial}",
    )
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _git_head() -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _move(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def setflow_validation_stage_seed_s1(config: Mapping[str, Any]) -> tuple[str, int]:
    run_stage = str(config.get("run_stage", "SCREEN"))
    if run_stage == "SCREEN":
        _require(
            config.get("schema_version", SCREEN_CONFIG_SCHEMA)
            == SCREEN_CONFIG_SCHEMA,
            "SetFlow V4 S1 screen validation config changed",
        )
        seed = int(config["training"]["screen_seed"])
        _require(seed == 20260911, "SetFlow V4 S1 validation seed changed")
        return run_stage, seed
    _require(
        run_stage == "CONFIRMATION"
        and config.get("schema_version") == CONFIRMATION_CONFIG_SCHEMA
        and config.get("status") == "FROZEN_S1_CONFIRMATION_CONFIG_NOT_STARTED"
        and config.get("selected_model") == CONFIRMATION_RUN_ID,
        "SetFlow V4 S1 confirmation validation config changed",
    )
    seeds = tuple(int(seed) for seed in config.get("required_confirmation_seeds", ()))
    seed = int(config.get("training_seed", -1))
    _require(
        seeds == CONFIRMATION_SEEDS
        and seed in CONFIRMATION_SEEDS
        and config.get("additional_seed_authorized") is False,
        "SetFlow V4 S1 confirmation validation seed changed",
    )
    return run_stage, seed


def require_training_package_terminal_s1(
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    run_stage, training_seed = setflow_validation_stage_seed_s1(config)
    summaries: dict[str, dict[str, Any]] = {}
    matched_initialization_by_run: dict[str, dict[str, Any]] = {}
    required_runs = (
        ("v4_s1_full", "v4_s1_single_mode")
        if run_stage == "SCREEN"
        else (CONFIRMATION_RUN_ID,)
    )
    for required_run in required_runs:
        directory = Path(config["output_root"]) / required_run
        _require(
            not (directory / "failure.json").exists(),
            f"SetFlow V4 S1 training package has a technical failure: {required_run}",
        )
        summary = _read_json(directory / "training_summary.json")
        _require(
            summary.get("status")
            == "TERMINAL_XEDITSETFLOW_V4_S1_TRAINING_COMPLETE_PENDING_VALIDATION",
            f"SetFlow V4 S1 training package is not fully terminal: {required_run}",
        )
        _require(
            summary.get("run_id") == required_run
            and summary.get("run_stage", "SCREEN") == run_stage
            and int(summary.get("seed", -1)) == training_seed,
            f"SetFlow V4 S1 training stage or seed changed: {required_run}",
        )
        if run_stage == "CONFIRMATION":
            _require(
                summary.get("run_id") == CONFIRMATION_RUN_ID
                and summary.get("selected_model") == CONFIRMATION_RUN_ID,
                "SetFlow V4 S1 confirmation summary identity changed",
            )
        _require(
            summary.get("objective_identity") == OBJECTIVE_IDENTITY
            and float(
                summary.get(
                    "cross_state_candidate_mode_responsibility_weight", -1.0
                )
            )
            == OBJECTIVE_WEIGHT
            and int(summary.get("active_responsibility_constraint_count", 0))
            > 0,
            f"SetFlow V4 S1 objective provenance changed: {required_run}",
        )
        _require(
            summary.get("parameter_initialization_seed") == training_seed
            and summary.get(
                "parameter_initialization_seed_applied_before_model_construction"
            )
            is True,
            f"SetFlow V4 S1 parameter initialization provenance changed: {required_run}",
        )
        matched_initialization_by_run[required_run] = (
            _require_matched_initialization_s1(
                summary,
                label=f"SetFlow V4 S1 terminal training summary {required_run}",
            )
        )
        update_geometry = summary.get("update_geometry")
        _require(
            int(summary.get("completed_passes", -1)) == 10
            and summary.get("early_stopping_used") is False
            and isinstance(update_geometry, Mapping)
            and int(update_geometry.get("pass_count", -1)) == 10
            and int(update_geometry.get("total_optimizer_updates", 0)) > 0
            and int(summary.get("optimizer_update_count", -1))
            == int(update_geometry["total_optimizer_updates"])
            and summary.get("parameter_changed") is True,
            f"SetFlow V4 S1 terminal parameter-update evidence changed: {required_run}",
        )
        physical_gpu_index = summary.get("physical_gpu_index")
        _require(
            isinstance(physical_gpu_index, int)
            and not isinstance(physical_gpu_index, bool)
            and physical_gpu_index in range(6)
            and summary.get("torch_device") == f"cuda:{physical_gpu_index}"
            and "A100" in str(summary.get("device_name", ""))
            and summary.get("training_precision") == "BF16"
            and summary.get("cuda_available") is True
            and summary.get("bf16_supported") is True
            and summary.get("cpu_fallback_used") is False,
            f"SetFlow V4 S1 terminal CUDA/A100/BF16 evidence changed: {required_run}",
        )
        _require(
            summary.get("validation_generation_during_training") is False
            and summary.get("checkpoint_selection_status")
            == "PENDING_TERMINAL_OUTCOME_FREE_VALIDATION_GENERATION",
            f"SetFlow V4 S1 training consumed Validation generation early: {required_run}",
        )
        _require(
            int(summary.get("development_test_outcome_reads", -1)) == 0
            and int(summary.get("new_final_evaluation_outcome_reads", -1)) == 0,
            f"SetFlow V4 S1 training reports a protected read: {required_run}",
        )
        expected_checkpoint_paths = {
            str(checkpoint_pass): str(directory / f"pass_{checkpoint_pass}.pt")
            for checkpoint_pass in (4, 6, 8, 10)
        }
        _require(
            summary.get("saved_checkpoint_paths") == expected_checkpoint_paths,
            f"SetFlow V4 S1 checkpoint paths differ from the frozen run directory: {required_run}",
        )
        summaries[required_run] = summary
    if run_stage == "SCREEN":
        _require(
            matched_initialization_by_run["v4_s1_full"]
            == matched_initialization_by_run["v4_s1_single_mode"],
            "SetFlow V4 S1 terminal full/single canonical initialization differs",
        )
    return summaries


def require_training_package_provenance_s1(
    config: Mapping[str, Any],
) -> dict[str, str]:
    """Return the Git HEAD that produced each terminal training package.

    Checkpoint validation can legitimately run from a later Git revision than
    training.  The launch authorization therefore binds the code that produced
    the checkpoint, while the validation result records its own Git revision
    separately.
    """

    summaries = require_training_package_terminal_s1(config)
    heads: dict[str, str] = {}
    for run_id in summaries:
        directory = Path(config["output_root"]) / run_id
        training_config = _read_json(directory / "training_config.json")
        training_attempt = _read_json(directory / "training_attempt.json")
        config_head = str(training_config.get("authorized_git_head", ""))
        attempt_head = str(training_attempt.get("code_commit", ""))
        _require(
            re.fullmatch(r"[0-9a-f]{40}", config_head) is not None,
            f"SetFlow V4 S1 training Git provenance is invalid: {run_id}",
        )
        _require(
            attempt_head == config_head,
            f"SetFlow V4 S1 training config and attempt disagree on Git HEAD: {run_id}",
        )
        if str(config.get("run_stage", "SCREEN")) == "CONFIRMATION":
            _require(
                training_config.get("confirmation_runner_git_head")
                == config.get("confirmation_runner_git_head")
                == config_head,
                f"SetFlow V4 S1 confirmation training Git HEAD drifted: {run_id}",
            )
        heads[run_id] = config_head
    _require(
        len(set(heads.values())) == 1,
        "SetFlow V4 S1 screen arms were trained from different Git HEADs",
    )
    return heads


def load_checkpoint_s1(
    config: Mapping[str, Any],
    *,
    run_id: str,
    checkpoint_pass: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    _require(checkpoint_pass in {4, 6, 8, 10}, "SetFlow V4 S1 checkpoint pass is undeclared")
    run_stage, training_seed = setflow_validation_stage_seed_s1(config)
    _require(
        run_stage == "SCREEN" or run_id == CONFIRMATION_RUN_ID,
        "SetFlow V4 S1 confirmation only permits v4_s1_full",
    )
    training_directory = Path(config["output_root"]) / run_id
    training_summary = require_training_package_terminal_s1(config)[run_id]
    _require(
        training_summary.get("status")
        == "TERMINAL_XEDITSETFLOW_V4_S1_TRAINING_COMPLETE_PENDING_VALIDATION",
        "SetFlow V4 S1 training is not terminal before checkpoint validation",
    )
    _require(
        training_summary.get("validation_generation_during_training") is False
        and training_summary.get("checkpoint_selection_status")
        == "PENDING_TERMINAL_OUTCOME_FREE_VALIDATION_GENERATION",
        "SetFlow V4 S1 training read Validation generation or selected early",
    )
    _require(
        set(training_summary.get("saved_checkpoint_paths", {}))
        == {"4", "6", "8", "10"},
        "SetFlow V4 S1 terminal checkpoint package is incomplete",
    )
    checkpoint_path = Path(
        training_summary["saved_checkpoint_paths"][str(checkpoint_pass)]
    )
    _require(
        checkpoint_path == training_directory / f"pass_{checkpoint_pass}.pt",
        "SetFlow V4 S1 checkpoint path differs from the frozen run directory",
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    training_matched_initialization = _require_matched_initialization_s1(
        training_summary,
        label="SetFlow V4 S1 selected training summary",
    )
    checkpoint_matched_initialization = _require_matched_initialization_s1(
        checkpoint,
        label="SetFlow V4 S1 selected checkpoint",
    )
    _require(
        checkpoint_matched_initialization == training_matched_initialization,
        "SetFlow V4 S1 checkpoint matched-initialization evidence changed",
    )
    _require(
        checkpoint.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_s1_checkpoint.v1",
        "unexpected SetFlow V4 S1 checkpoint schema",
    )
    _require(
        checkpoint.get("run_id") == run_id
        and checkpoint.get("run_stage", "SCREEN") == run_stage
        and int(checkpoint.get("completed_pass", -1)) == checkpoint_pass
        and int(checkpoint.get("seed", -1)) == training_seed,
        "SetFlow V4 S1 checkpoint identity changed",
    )
    if run_stage == "CONFIRMATION":
        _require(
            checkpoint.get("selected_model") == CONFIRMATION_RUN_ID,
            "SetFlow V4 S1 confirmation checkpoint model identity changed",
        )
    _require(
        checkpoint.get("objective_identity") == OBJECTIVE_IDENTITY
        and float(
            checkpoint.get(
                "cross_state_candidate_mode_responsibility_weight", -1.0
            )
        )
        == OBJECTIVE_WEIGHT
        and int(checkpoint.get("active_responsibility_constraint_count", 0)) > 0,
        "SetFlow V4 S1 checkpoint objective provenance changed",
    )
    _require(
        checkpoint.get("parameter_initialization_seed") == training_seed
        and checkpoint.get(
            "parameter_initialization_seed_applied_before_model_construction"
        )
        is True
        and checkpoint.get("parameter_initialization_seed")
        == training_summary.get("parameter_initialization_seed"),
        "SetFlow V4 S1 checkpoint parameter initialization provenance changed",
    )
    physical_gpu_index = checkpoint.get("physical_gpu_index")
    _require(
        isinstance(physical_gpu_index, int)
        and not isinstance(physical_gpu_index, bool)
        and physical_gpu_index in range(6)
        and checkpoint.get("torch_device") == f"cuda:{physical_gpu_index}"
        and "A100" in str(checkpoint.get("device_name", ""))
        and checkpoint.get("training_precision") == "BF16"
        and checkpoint.get("cuda_available") is True
        and checkpoint.get("bf16_supported") is True
        and checkpoint.get("cpu_fallback_used") is False
        and int(checkpoint.get("development_test_outcome_reads", -1)) == 0
        and int(checkpoint.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "SetFlow V4 S1 checkpoint CUDA/A100/BF16 provenance changed",
    )
    model, capacity = build_setflow_screen_model_s1(
        config, checkpoint["vocabs"], run_id=run_id
    )
    _require(capacity == checkpoint["capacity"], "SetFlow V4 S1 checkpoint capacity changed")
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint, training_summary


@torch.no_grad()
def evaluate_common_validation_nll_s1(
    model: torch.nn.Module,
    validation_rows: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
    *,
    source_cache: SourceTokenCacheIndexV3,
    device: torch.device,
    state_seed: int,
    batch_size: int = 32,
) -> dict[str, Any]:
    records, eligibility = setflow_records_from_projection_rows(validation_rows)
    dataset = SetMarginalStateDatasetV3(
        records,
        checkpoint["vocabs"],
        seed=int(state_seed),
        states_per_record=2,
    )
    dataset.set_pass(0)
    indices = [
        (record_index, state_slot)
        for record_index in range(len(records))
        for state_slot in range(2)
    ]
    weighted = 0.0
    total_weight = 0.0
    active_count = 0
    forward_batch_count = 0
    forward_state_count = 0
    model.eval()
    for start in range(0, len(indices), batch_size):
        state_indices = indices[start : start + batch_size]
        batch = _move(
            collate_setflow_states_v3(
                [
                    dataset.state(record_index, state_slot)
                    for record_index, state_slot in state_indices
                ],
                source_cache=source_cache,
            ),
            device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch)
        result = common_set_marginal_loss_v4(
            output,
            batch["positive_action_mask"],
            batch["structural_budget_exhausted"],
            batch["sample_weight"],
        )
        weight = float(result.active_weight.cpu())
        weighted += float(result.loss.cpu()) * weight
        total_weight += weight
        active_count += result.active_state_count
        forward_batch_count += 1
        forward_state_count += len(state_indices)
    _require(total_weight > 0.0 and active_count > 0, "SetFlow V4 S1 common Validation set is empty")
    return {
        "common_validation_set_marginal_nll": weighted / total_weight,
        "validation_candidate_record_count": len(records),
        "validation_states_per_record": 2,
        "validation_active_state_count": active_count,
        "eligibility": eligibility,
        "forward_batch_count": forward_batch_count,
        "forward_state_count": forward_state_count,
        "mode_head_forward_state_count": forward_state_count * model.mode_count,
    }


def _enumerate_terminal_paths(root, rate_function):
    terminal = defaultdict(float)

    def visit(state, probability):
        if state.terminal_cause is not None:
            terminal[state] += probability
            return
        for _action, child, edge_probability in jump_distribution(
            state, rate_function, support_floor=1e-8
        ):
            visit(child, probability * edge_probability)

    visit(root, 1.0)
    return dict(terminal)


def small_graph_exact_check_s1(
    model,
    checkpoint: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    source = "AC"
    synthetic_cache = SourceTokenCacheIndexV3(
        assemble_source_token_cache_v3(
            [{"canonical_record_id": "small-graph", "source_sequence": source}],
            sequence_to_index={source: 0},
            encoded_tokens={0: torch.zeros(2, 768)},
            model_id="SYNTHETIC_ZERO_FEATURE_FOR_DISTRIBUTION_CHECK_ONLY",
            pretrained_parameter_count=113_389_056,
            attention_backend="NOT_APPLICABLE_SYNTHETIC_CHECK",
        )
    )
    metadata = SetFlowGenerationMetadataV3(
        "small-graph", 0, 0, 0, 0, 0, 0, 0
    )
    root = initial_state(source, budget=2, assay_id="__UNK__", context_id="__UNK__")
    priors, prior_compute = root_mode_priors_v4(
        model,
        [root],
        [metadata],
        source_cache=synthetic_cache,
        device=device,
        forward_batch_size=1,
    )
    dynamic_mixture = defaultdict(float)
    enumerated_mixture = defaultdict(float)
    per_mode_state_counts: dict[str, int] = {}
    for mode_id, prior in enumerate(priors[0]):
        def rate_function(state, actions, selected_mode=mode_id):
            return setflow_rate_map_v4(
                model,
                state,
                metadata,
                selected_mode,
                actions,
                source_cache=synthetic_cache,
                device=device,
            )

        dynamic = exact_terminal_distribution(
            root, rate_function, support_floor=1e-8
        )
        enumerated = _enumerate_terminal_paths(root, rate_function)
        per_mode_state_counts[str(mode_id)] = len(set(dynamic) | set(enumerated))
        for state, probability in dynamic.items():
            dynamic_mixture[state] += float(prior) * probability
        for state, probability in enumerated.items():
            enumerated_mixture[state] += float(prior) * probability
    states = set(dynamic_mixture) | set(enumerated_mixture)
    total_variation = 0.5 * math.fsum(
        abs(dynamic_mixture.get(state, 0.0) - enumerated_mixture.get(state, 0.0))
        for state in states
    )
    _require(total_variation <= 1e-12, "SetFlow V4 S1 small-graph mixture differs from enumeration")
    return {
        "source_length": 2,
        "edit_budget": 2,
        "mode_count": model.mode_count,
        "mode_prior": list(priors[0]),
        "per_mode_terminal_state_count": per_mode_state_counts,
        "mixture_terminal_state_count": len(states),
        "dynamic_probability_sum": math.fsum(dynamic_mixture.values()),
        "enumeration_probability_sum": math.fsum(enumerated_mixture.values()),
        "total_variation": total_variation,
        "tolerance": 1e-12,
        "status": "PASS",
        "prior_compute": prior_compute.__dict__,
        "benchmark_compute_excludes_small_graph_mechanics_test": True,
        "source_token_feature_policy": "SYNTHETIC_ZERO_FEATURE_MECHANICS_CHECK_NOT_PERFORMANCE",
    }


def validate_checkpoint(
    config: Mapping[str, Any],
    *,
    run_id: str,
    checkpoint_pass: int,
    authorization_path: Path,
    output_directory: Path,
    physical_gpu_index: int,
) -> dict[str, Any]:
    spec = screen_run_spec_s1(config, run_id)
    run_stage, training_seed = setflow_validation_stage_seed_s1(config)
    _require(
        run_stage == "SCREEN" or run_id == CONFIRMATION_RUN_ID,
        "SetFlow V4 S1 confirmation only permits v4_s1_full",
    )
    authorization = _read_json(authorization_path)
    preflight = _read_json(Path(config["preflight_output_path"]))
    source_data_audit = _read_json(Path(config["source_level_data_audit_path"]))
    validation_git_head = _git_head()
    training_git_heads = require_training_package_provenance_s1(config)
    training_git_head = training_git_heads[run_id]
    if run_stage == "SCREEN":
        require_s1_launch_authorization(
            config,
            authorization,
            preflight,
            source_data_audit,
            run_id=run_id,
            current_git_head=training_git_head,
        )
    else:
        require_s1_confirmation_launch_authorization(
            config,
            authorization,
            preflight,
            source_data_audit,
            run_id=run_id,
            training_seed=training_seed,
            current_git_head=training_git_head,
        )
    _require(not output_directory.exists(), "SetFlow V4 S1 checkpoint validation already exists")
    _require(
        set(config["gpu_policy"]["physical_gpu_scope"]) == set(range(6))
        and physical_gpu_index in config["gpu_policy"]["physical_gpu_scope"],
        "SetFlow V4 S1 validation GPU is outside 0–5",
    )
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden")
    device = torch.device(f"cuda:{physical_gpu_index}")
    torch.cuda.set_device(device)
    device_name = torch.cuda.get_device_name(device)
    _require("A100" in device_name, "selected GPU is not an A100")
    _require(torch.cuda.is_bf16_supported(), "BF16 is unavailable on selected GPU")
    cuda = cuda_device_observation(
        physical_gpu_index, require_physical_index_match=True
    )
    model, checkpoint, training_summary = load_checkpoint_s1(
        config,
        run_id=run_id,
        checkpoint_pass=checkpoint_pass,
        device=device,
    )
    training_summary_path = (
        Path(config["output_root"]) / run_id / "training_summary.json"
    )
    checkpoint_path = Path(
        training_summary["saved_checkpoint_paths"][str(checkpoint_pass)]
    )
    validation_rows = load_projection_rows(
        [Path(config["validation_projection_path"])],
        allowed_splits=("VALIDATION",),
    )
    _require(
        len(validation_rows)
        == int(config["data_geometry"]["expected_validation_projection_candidate_row_count"]),
        "SetFlow V4 S1 Validation projection count changed",
    )
    cache = SourceTokenCacheIndexV3(
        load_source_token_cache_v3(Path(config["source_token_cache_path"]))
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    common_nll = evaluate_common_validation_nll_s1(
        model,
        validation_rows,
        checkpoint,
        source_cache=cache,
        device=device,
        state_seed=int(
            config["validation_generation"]["common_validation_state_seed"]
        ),
        batch_size=32,
    )
    sources = load_sources(Path(config["source_eligibility_manifest"]))
    validation_generation = config["validation_generation"]
    _require(
        len(sources) == int(validation_generation["eligible_source_count"]) == 891,
        "SetFlow V4 S1 generation source cohort changed",
    )
    _require(
        all(
            int(source["candidate_budget"])
            == int(validation_generation["candidate_cap_per_source"])
            == 32
            for source in sources
        ),
        "SetFlow V4 S1 candidate cap changed",
    )
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
    forward_batch_size = 64
    priors, prior_compute = root_mode_priors_v4(
        model,
        source_roots,
        source_metadata,
        source_cache=cache,
        device=device,
        forward_batch_size=forward_batch_size,
    )
    roots = []
    trajectory_metadata = []
    mode_ids = []
    seeds = []
    source_indices = []
    aggregate_mode_allocations = Counter()
    decoder_seed_base = int(validation_generation["decoder_seed_base"])
    for source_index, (root, metadata, prior) in enumerate(
        zip(source_roots, source_metadata, priors, strict=True)
    ):
        allocated_modes = stratified_trajectory_mode_ids_v4(prior)
        _require(len(allocated_modes) == 32, "SetFlow V4 S1 source trajectory budget changed")
        for trajectory_slot, mode_id in enumerate(allocated_modes):
            roots.append(root)
            trajectory_metadata.append(metadata)
            mode_ids.append(mode_id)
            seeds.append(
                decoder_seed_base
                + source_index * 1_000_003
                + trajectory_slot
            )
            source_indices.append(source_index)
            aggregate_mode_allocations[mode_id] += 1
    _require(len(roots) == 891 * 32, "SetFlow V4 S1 trajectory count changed")
    sampled, primary_compute = sample_many_setflow_v4(
        model,
        roots,
        trajectory_metadata,
        mode_ids,
        seeds,
        source_cache=cache,
        device=device,
        forward_batch_size=forward_batch_size,
    )
    replayed, replay_compute = sample_many_setflow_v4(
        model,
        roots,
        trajectory_metadata,
        mode_ids,
        seeds,
        source_cache=cache,
        device=device,
        forward_batch_size=forward_batch_size,
    )
    candidates = []
    replay_failures = 0
    edit_budget_violations = 0
    terminal_causes = Counter()
    method_id = (
        f"unguided_xeditsetflow_v4_s1_{run_id}_pass{checkpoint_pass}_seed{training_seed}"
    )
    for trajectory_index, (first, second) in enumerate(
        zip(sampled, replayed, strict=True)
    ):
        terminal, actions, forwards = first
        replay_terminal, replay_actions, replay_forwards = second
        replay_ok = (
            terminal == replay_terminal
            and actions == replay_actions
            and forwards == replay_forwards
        )
        replay_failures += int(not replay_ok)
        source = sources[source_indices[trajectory_index]]
        edit_budget_violations += int(
            terminal.edit_count > int(source["edit_budget"])
        )
        terminal_causes[str(terminal.terminal_cause)] += 1
        candidates.append(
            {
                "method_id": method_id,
                "source_key": source["source_key"],
                "candidate_sequence": terminal.current_sequence,
                "terminal_cause": terminal.terminal_cause,
                "edit_count": terminal.edit_count,
                "trajectory_actions": list(actions),
                "trajectory_seed": seeds[trajectory_index],
                "trajectory_mode_id": mode_ids[trajectory_index],
                "trajectory_replay_ok": replay_ok,
                "generator_nfe": forwards,
                "trunk_forwards": forwards,
                "mode_head_forwards": forwards * model.mode_count,
                "critic_forwards": 0,
                "independent_evaluator_forwards": 0,
                "generated_candidate_grants_canonical_credit": False,
            }
        )
    empirical = Counter(
        (row["source_key"], row["candidate_sequence"]) for row in candidates
    )
    totals = Counter(row["source_key"] for row in candidates)
    for row in candidates:
        row["generation_score"] = math.log(
            empirical[(row["source_key"], row["candidate_sequence"])]
            / totals[row["source_key"]]
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
    small_graph = small_graph_exact_check_s1(model, checkpoint, device)
    numerical_failures = terminal_causes.get("NUMERICAL_FAILURE", 0)
    correctness = (
        generation["hard_legality_rate"] == 1.0
        and edit_budget_violations == 0
        and generation["candidate_budget_violation_count"] == 0
        and replay_failures == 0
        and numerical_failures == 0
        and small_graph["status"] == "PASS"
    )
    torch.cuda.synchronize(device)
    elapsed = time.time() - started
    compute = {
        "common_nll_trunk_forward_batch_count": common_nll["forward_batch_count"],
        "common_nll_trunk_forward_state_count": common_nll["forward_state_count"],
        "common_nll_mode_head_forward_state_count": common_nll[
            "mode_head_forward_state_count"
        ],
        "root_prior": prior_compute.__dict__,
        "primary_generation": primary_compute.__dict__,
        "replay_generation": replay_compute.__dict__,
        "trajectory_count": len(candidates),
        "candidate_count": len(candidates),
        "critic_forward_count": 0,
        "independent_evaluator_forward_count": 0,
        "small_graph_mechanics_test_excluded": True,
    }
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir()
    (output_directory / "trajectories.private.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidates),
        encoding="utf-8",
    )
    result = {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_s1_checkpoint_validation.v1",
        "status": "TERMINAL_XEDITSETFLOW_V4_S1_CHECKPOINT_VALIDATION_COMPLETE",
        "g0_status": "FLOW_G0_READY" if correctness else "FLOW_G0_VALIDATION_FAIL",
        "run_stage": run_stage,
        "run_id": run_id,
        **(
            {"selected_model": CONFIRMATION_RUN_ID}
            if run_stage == "CONFIRMATION"
            else {}
        ),
        "selectable": spec.selectable,
        "mode_count": spec.mode_count,
        "objective_identity": checkpoint["objective_identity"],
        "cross_state_candidate_mode_responsibility_weight": checkpoint[
            "cross_state_candidate_mode_responsibility_weight"
        ],
        "active_responsibility_constraint_count": checkpoint[
            "active_responsibility_constraint_count"
        ],
        "active_responsibility_candidate_count": checkpoint[
            "active_responsibility_candidate_count"
        ],
        "active_responsibility_occurrence_count": checkpoint[
            "active_responsibility_occurrence_count"
        ],
        "seed": training_seed,
        "parameter_initialization_seed": training_summary[
            "parameter_initialization_seed"
        ],
        "parameter_initialization_seed_applied_before_model_construction": (
            training_summary[
                "parameter_initialization_seed_applied_before_model_construction"
            ]
        ),
        "matched_initialization": _require_matched_initialization_s1(
            training_summary,
            label="SetFlow V4 S1 selected training summary",
        ),
        "checkpoint_pass": checkpoint_pass,
        "checkpoint_path": str(checkpoint_path),
        "training_summary_path": str(training_summary_path),
        "validation_summary_path": str(
            output_directory / "validation_summary.json"
        ),
        "training_git_head": training_git_head,
        "validation_git_head": validation_git_head,
        "training_and_validation_git_heads_differ": (
            training_git_head != validation_git_head
        ),
        "training_summary_status": training_summary["status"],
        "training_cuda_available": training_summary["cuda_available"],
        "training_bf16_supported": training_summary["bf16_supported"],
        "training_torch_device": training_summary["torch_device"],
        "training_device_name": training_summary["device_name"],
        "training_precision": training_summary["training_precision"],
        "training_cpu_fallback_used": training_summary["cpu_fallback_used"],
        "common_validation_set_marginal_nll": common_nll[
            "common_validation_set_marginal_nll"
        ],
        "common_validation": common_nll,
        "source_count": len(sources),
        "trajectory_count": len(candidates),
        "candidate_count": len(candidates),
        "candidate_cap_per_source": 32,
        "duplicate_retry_or_rejection_count": 0,
        "aggregate_mode_allocations": {
            str(key): value for key, value in sorted(aggregate_mode_allocations.items())
        },
        "hard_legality_rate": generation["hard_legality_rate"],
        "edit_budget_violation_count": edit_budget_violations,
        "candidate_budget_violation_count": generation[
            "candidate_budget_violation_count"
        ],
        "trajectory_replay_failure_count": replay_failures,
        "numerical_failure_count": numerical_failures,
        "source_macro_unique_candidate_rate": generation[
            "source_macro_unique_candidate_rate"
        ],
        "source_macro_candidate_recovery_rate": measured[
            "source_macro_candidate_recovery_rate"
        ],
        "source_macro_measured_top_k_recovery_at_k": measured[
            "source_macro_measured_top_k_recovery_at_k"
        ],
        "terminal_causes": dict(sorted(terminal_causes.items())),
        "small_graph_reference": small_graph,
        "compute": compute,
        "wall_time_seconds": elapsed,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        "physical_gpu_index": physical_gpu_index,
        "torch_device": str(device),
        "device_name": device_name,
        "precision": "BF16",
        "cuda_available": True,
        "bf16_supported": True,
        "cpu_fallback_used": False,
        "parameter_update_count": 0,
        "critic_used": False,
        "independent_evaluator_used": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
        "generated_candidates_grant_canonical_credit": False,
        "biological_optimization_established": False,
        "generation_metrics": generation,
        "measured_neighborhood_metrics": measured,
        **cuda,
    }
    _write_atomic_terminal_s1(
        output_directory / "validation_summary.json",
        result,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", required=True, choices=("v4_s1_full", "v4_s1_single_mode"))
    parser.add_argument("--checkpoint-pass", required=True, type=int, choices=(4, 6, 8, 10))
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    config = _read_json(arguments.config)
    run_stage, training_seed = setflow_validation_stage_seed_s1(config)
    output_directory = arguments.output_dir or (
        Path(config["validation_output_root"])
        / arguments.run_id
        / f"pass_{arguments.checkpoint_pass}"
    )
    try:
        result = validate_checkpoint(
            config,
            run_id=arguments.run_id,
            checkpoint_pass=arguments.checkpoint_pass,
            authorization_path=arguments.authorization,
            output_directory=output_directory,
            physical_gpu_index=arguments.physical_gpu_index,
        )
    except Exception as error:
        failure_path = output_directory.with_name(
            output_directory.name + ".failed.json"
        )
        _write_atomic_terminal_s1(
            failure_path,
            {
                "schema_version": "route_a_v3_route2_xeditsetflow_v4_s1_checkpoint_validation_failure.v1",
                "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
                "run_id": arguments.run_id,
                "run_stage": run_stage,
                "seed": training_seed,
                "checkpoint_pass": arguments.checkpoint_pass,
                "physical_gpu_index": arguments.physical_gpu_index,
                "cpu_fallback_used": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
