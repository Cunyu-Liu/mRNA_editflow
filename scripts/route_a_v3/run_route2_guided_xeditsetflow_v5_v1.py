#!/usr/bin/env python3
"""Guided vs unguided generation for one terminal SetFlow V5 checkpoint (Gate B2).

Runs the frozen 891x32 generation protocol of the V5 screen with matched
decoder seed streams: once with the unmodified V5 sampler (unguided arm) and
once with a frozen critic potential tilting every transition rate
(guided arm, U_q = U_p * exp(beta * (V(child) - V(state)))).

The guided critic family is selected by --critic-kind.  The pre-authorized
default is the terminal XEditCritic V5 screen checkpoint (the Critic V2
all-development refit never executed; see
route2_xeditcritic_v5_frozen_guidance_v1.py for the substitution record).

The tilt follows the frozen G0 reward policy
(configs/route_a_v3_route2_mrnabert_guidance_reward_policy_v1.json:
guidance_schedule=CONSTANT, guidance_strength=1.0, potential clip [-5, 5]),
which is the same transition rule as
core.route2_xeditflow_guidance_v3.potential_guided_rates_v3 evaluated at a
constant beta.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_gpu_failure_evidence import (
    cuda_device_observation,
)
from core.route2_legal_xeditflow import (
    STOP,
    FlowState,
    LegalAction,
    apply_action,
    initial_state,
    legal_actions,
)
from core.route2_source_token_cache_v3 import (
    SourceTokenCacheIndexV3,
    load_source_token_cache_v3,
)
from core.route2_xeditsetflow_sampling_v3 import (
    SetFlowGenerationMetadataV3,
    build_generation_metadata_v3,
    collate_generation_states_v3,
)
from core.route2_xeditsetflow_sampling_v4 import (
    BASE as _BASE,
    root_mode_priors_v4,
    sample_many_setflow_v4,
    select_trajectory_mode_rates_v4,
    stratified_trajectory_mode_ids_v4,
)
from core.route2_xeditsetflow_temperature_control_v5 import temper_mode_prior_v5
from core.route2_development_projection_v3 import load_projection_rows
from scripts.route_a_v3.evaluate_route2_generation_v1 import (
    evaluate_generation,
    load_source_manifest,
    measured_neighborhood_metrics,
    validate_measured_pool,
)
from scripts.route_a_v3.run_route2_base_flow_g0_validation_v1 import load_sources
from scripts.route_a_v3.route2_mrnabert_guided_critic_v1 import (
    FrozenRoute2MRNABERTCritic,
)
from scripts.route_a_v3.route2_xeditcritic_v5_frozen_guidance_v1 import (
    FrozenXEditCriticV5,
)
from scripts.route_a_v3.validate_route2_xeditsetflow_v5_checkpoint import (
    load_checkpoint_v5,
    sample_many_setflow_v5,
    setflow_validation_stage_seed_v5,
)

GUIDED_RUN_SCHEMA = "route_a_v3_route2_guided_xeditsetflow_v5_runner.v1"
ARM_SUMMARY_SCHEMA = "route_a_v3_route2_guided_xeditsetflow_v5_arm_summary.v1"
RUNNER_FAILURE_SCHEMA = (
    "route_a_v3_route2_guided_xeditsetflow_v5_runner_failure.v1"
)
FROZEN_REWARD_POLICY_PATH = (
    REPO_ROOT / "configs/route_a_v3_route2_mrnabert_guidance_reward_policy_v1.json"
)
EXPECTED_CRITIC_CHECKPOINT = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runs/mrnabert_critic_v2/"
    "all_development_refit_v1/seed20260823/delta_predictor_checkpoint.pt"
)
EXPECTED_MRNABERT_MODEL = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/"
    "mrnabert_a1eb7df25804d23f08646e1cb996b234d7208a40"
)
EXPECTED_XEDITCRITIC_V5_CHECKPOINT = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_v5/"
    "v5_screen_seed_20260907_runner_1113cd2c0dd9acb508f58782eecb40f458d2cab3/"
    "v5_full/final_pass_8_checkpoint.pt"
)
CRITIC_REGIONS = {"5UTR", "3UTR"}


class GuidedSetFlowV5Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GuidedSetFlowV5Error(message)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    _require(bool(rows), f"JSONL input is empty: {path}")
    return rows


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), f"terminal artifact already exists: {path}")
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _git_head() -> str:
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


def _flat_action(padded_length: int, flat_index: int) -> LegalAction:
    if flat_index == padded_length * 4:
        return LegalAction(STOP)
    position, alt_index = divmod(flat_index, 4)
    return LegalAction("SUB", position, _BASE[alt_index])


@torch.no_grad()
def sample_one_source_setflow_v5_guided(
    model: torch.nn.Module,
    root: FlowState,
    metadata: SetFlowGenerationMetadataV3,
    mode_ids: Sequence[int],
    seeds: Sequence[int],
    *,
    source_row: Mapping[str, Any],
    critic: FrozenRoute2MRNABERTCritic,
    beta: float,
    source_cache: SourceTokenCacheIndexV3,
    device: torch.device,
    forward_batch_size: int,
) -> tuple[list[tuple[FlowState, tuple[str, ...], int]], list[float]]:
    """Sample 32 trajectories from the potential-tilted V5 transition kernel.

    The tilt U_q(a) proportional to U_p(a) * exp(beta * (V(child_a) - V(state)))
    is applied to every legal action column of the V5 rate vector at each edit
    decision point, with V from the frozen Critic V2 potential (constant beta
    per the frozen G0 reward policy).  The per-trajectory uniform streams are
    identical to the unguided arm, so the two arms are seed-matched.
    """

    _require(device.type == "cuda", "guided SetFlow V5 sampling requires CUDA")
    _require(
        math.isfinite(beta) and beta >= 0.0,
        "guided SetFlow V5 beta must be finite and nonnegative",
    )
    _require(
        len(seeds) == 32 and len(mode_ids) == 32,
        "guided source trajectory budget differs",
    )
    endpoint_id = str(source_row["endpoint_id"])
    region = str(source_row["region"]).replace("′", "").replace("'", "")
    _require(region in CRITIC_REGIONS, "guided source region is unsupported")
    model.eval()
    states: list[FlowState] = [root for _ in seeds]
    generators = [random.Random(int(seed)) for seed in seeds]
    action_ids: list[list[str]] = [[] for _ in seeds]
    forwards = [0 for _ in seeds]
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
                    [metadata for _ in indices],
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
            padded_length = (int(rates.shape[1]) - 1) // 4
            stop_column = padded_length * 4
            query_states: list[FlowState] = []
            query_children: list[list[tuple[int, FlowState]]] = []
            for index in indices:
                state = states[index]
                columns: list[tuple[int, FlowState]] = []
                for action in legal_actions(state):
                    flat = (
                        stop_column
                        if action.kind == STOP
                        else int(action.position) * 4 + _BASE.index(str(action.alt_base))
                    )
                    columns.append((flat, apply_action(state, action)))
                query_states.append(state)
                query_children.append(columns)
            flat_query = [
                *[state for state in query_states],
                *[child for columns in query_children for _, child in columns],
            ]
            values = critic.potentials(
                flat_query,
                endpoint_id=endpoint_id,
                region=region,
                source_row=source_row,
            )
            current_potential = torch.tensor(
                values[: len(indices)], dtype=torch.float64, device=device
            )
            child_potential = (
                current_potential.unsqueeze(1).expand_as(weights).clone()
            )
            cursor = len(indices)
            for row, columns in enumerate(query_children):
                for flat, _child in columns:
                    child_potential[row, flat] = float(values[cursor])
                    cursor += 1
            _require(
                cursor == len(values),
                "guided child potential accounting does not close",
            )
            tilted = weights * torch.exp(
                beta * (child_potential - current_potential.unsqueeze(1))
            )
            guided = torch.where(masks, tilted, torch.zeros_like(tilted))
            totals = guided.sum(dim=1)
            _require(
                bool(torch.isfinite(guided).all().item())
                and bool((totals > 0).all().item()),
                "guided SetFlow V5 produced an invalid exit-rate distribution",
            )
            cumulative = guided.cumsum(dim=1) / totals.unsqueeze(1)
            uniforms = torch.tensor(
                [generators[index].random() for index in indices],
                dtype=torch.float64,
                device=device,
            )
            choices = (cumulative < uniforms.unsqueeze(1)).sum(dim=1).clamp_max(
                int(rates.shape[1]) - 1
            )
            rows = torch.arange(len(indices), device=device)
            _require(
                bool(masks[rows, choices].all().item()),
                "guided SetFlow V5 sampled a masked action",
            )
            for trajectory_index, flat_index in zip(
                indices, choices.tolist(), strict=True
            ):
                state = states[trajectory_index]
                action = _flat_action(padded_length, int(flat_index))
                _require(
                    action in legal_actions(state),
                    "guided SetFlow V5 selected an action outside hard legality",
                )
                states[trajectory_index] = apply_action(state, action)
                action_ids[trajectory_index].append(action.action_id)
                forwards[trajectory_index] += 1
    terminals = [
        (states[index], tuple(action_ids[index]), forwards[index])
        for index in range(len(seeds))
    ]
    terminal_potentials = critic.potentials(
        [terminal for terminal, _, _ in terminals],
        endpoint_id=endpoint_id,
        region=region,
        source_row=source_row,
    )
    return terminals, terminal_potentials


def build_arm_candidates(
    sampled: Sequence[tuple[FlowState, tuple[str, ...], int]],
    sources: Sequence[Mapping[str, Any]],
    source_indices: Sequence[int],
    seeds: Sequence[int],
    mode_ids: Sequence[int],
    *,
    method_id: str,
    mode_count: int,
    critic_forwards_by_source: dict[int, int] | None = None,
    terminal_potentials: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for trajectory_index, (terminal, actions, forwards) in enumerate(sampled):
        source = sources[source_indices[trajectory_index]]
        candidates.append(
            {
                "method_id": method_id,
                "source_key": source["source_key"],
                "candidate_sequence": terminal.current_sequence,
                "terminal_cause": terminal.terminal_cause,
                "edit_count": terminal.edit_count,
                "trajectory_actions": list(actions),
                "trajectory_seed": int(seeds[trajectory_index]),
                "trajectory_mode_id": int(mode_ids[trajectory_index]),
                "generator_nfe": int(forwards),
                "trunk_forwards": int(forwards),
                "mode_head_forwards": int(forwards) * int(mode_count),
                "critic_forwards": 0,
                "independent_evaluator_forwards": 0,
                "generated_candidate_grants_canonical_credit": False,
                **(
                    {"terminal_potential": float(terminal_potentials[trajectory_index])}
                    if terminal_potentials is not None
                    else {}
                ),
            }
        )
    if critic_forwards_by_source:
        for row, source_index in zip(candidates, source_indices, strict=True):
            if source_index in critic_forwards_by_source:
                row["critic_forwards"] = int(critic_forwards_by_source[source_index])
                critic_forwards_by_source.pop(source_index)
    empirical = Counter(
        (row["source_key"], row["candidate_sequence"]) for row in candidates
    )
    totals = Counter(row["source_key"] for row in candidates)
    for row in candidates:
        row["generation_score"] = math.log(
            empirical[(row["source_key"], row["candidate_sequence"])]
            / totals[row["source_key"]]
        )
    return candidates


def check_reference_reproduction(
    candidates: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reference = {
        (str(row["source_key"]), int(row["trajectory_seed"])): row
        for row in reference_rows
    }
    compared = 0
    matches = 0
    mismatches: list[dict[str, Any]] = []
    for row in candidates:
        key = (str(row["source_key"]), int(row["trajectory_seed"]))
        if key not in reference:
            continue
        compared += 1
        other = reference[key]
        same = (
            row["candidate_sequence"] == other["candidate_sequence"]
            and list(row["trajectory_actions"]) == list(other["trajectory_actions"])
            and row["terminal_cause"] == other["terminal_cause"]
            and int(row["edit_count"]) == int(other["edit_count"])
        )
        matches += int(same)
        if not same and len(mismatches) < 8:
            mismatches.append(
                {
                    "source_key": key[0],
                    "trajectory_seed": key[1],
                    "runner_actions": list(row["trajectory_actions"]),
                    "reference_actions": list(other["trajectory_actions"]),
                }
            )
    _require(compared > 0, "reference trajectories do not overlap the runner cohort")
    return {
        "reference_trajectory_compared": compared,
        "reference_trajectory_exact_matches": matches,
        "reference_trajectory_match_rate": matches / compared,
        "reference_mismatch_examples": mismatches,
    }


def evaluate_arm(
    arm_name: str,
    candidates: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Mapping[str, Any]],
    measured_rows: Sequence[Mapping[str, Any]],
    *,
    measured_top_k: int,
    compute: Mapping[str, Any],
) -> dict[str, Any]:
    generation = evaluate_generation(manifest, list(candidates))
    measured = measured_neighborhood_metrics(
        manifest,
        list(candidates),
        list(measured_rows),
        k=int(measured_top_k),
        candidate_support_mode="OPEN_GENERATED_SUPPORT",
    )
    return {
        "schema_version": ARM_SUMMARY_SCHEMA,
        "arm": arm_name,
        "status": "GUIDED_B2_ARM_COMPLETE",
        "method_id": str(candidates[0]["method_id"]),
        "source_count": len(manifest),
        "trajectory_count": len(candidates),
        "candidate_count": len(candidates),
        "candidate_cap_per_source": 32,
        "hard_legality_rate": generation["hard_legality_rate"],
        "edit_budget_violation_count": generation["edit_budget_violation_count"],
        "candidate_budget_violation_count": generation[
            "candidate_budget_violation_count"
        ],
        "source_macro_unique_candidate_rate": generation[
            "source_macro_unique_candidate_rate"
        ],
        "source_macro_candidate_recovery_rate": measured[
            "source_macro_candidate_recovery_rate"
        ],
        "source_macro_measured_top_k_recovery_at_k": measured[
            "source_macro_measured_top_k_recovery_at_k"
        ],
        "terminal_causes": generation["terminal_causes"],
        "compute": dict(compute),
        "generation_metrics": generation,
        "measured_neighborhood_metrics": measured,
    }


def execute(arguments: argparse.Namespace) -> dict[str, Any]:
    config = _read_json(Path(arguments.config))
    _require(
        config.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v5_screen_config.v1",
        "unexpected SetFlow V5 screen config schema",
    )
    run_stage, training_seed = setflow_validation_stage_seed_v5(config)
    _require(
        run_stage == "SCREEN",
        "guided B2 runner only supports the SCREEN stage",
    )
    run_id = str(arguments.run_id)
    checkpoint_pass = int(arguments.checkpoint_pass)
    screen_gate = _read_json(Path(arguments.screen_gate))
    gate_arm = screen_gate.get("arms", {}).get(run_id)
    _require(gate_arm is not None, "guided B2 run id is absent from the screen gate")
    _require(
        gate_arm.get("gate_b1_passed") is True
        and int(gate_arm.get("selected_checkpoint_pass", -1)) == checkpoint_pass,
        "guided B2 checkpoint differs from the gate-B1-selected checkpoint pass",
    )
    arms = [arm.strip() for arm in str(arguments.arms).split(",") if arm.strip()]
    _require(
        bool(arms) and set(arms) <= {"unguided", "guided"},
        "guided B2 arms must be a nonempty subset of {unguided, guided}",
    )
    output_directory = Path(arguments.output_dir)
    _require(
        not output_directory.exists(),
        f"guided B2 output directory already exists: {output_directory}",
    )
    _require(
        not os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES remapping is forbidden",
    )
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    physical_gpu_index = int(arguments.physical_gpu_index)
    device = torch.device(f"cuda:{physical_gpu_index}")
    torch.cuda.set_device(device)
    _require(torch.cuda.is_bf16_supported(), "BF16 is unavailable on selected GPU")
    free_bytes, _total = torch.cuda.mem_get_info(device)
    _require(
        free_bytes >= 8 * 1024**3,
        "selected GPU has less than 8 GiB free before launch",
    )
    cuda = cuda_device_observation(
        physical_gpu_index, require_physical_index_match=True
    )
    reward_policy = _read_json(FROZEN_REWARD_POLICY_PATH)
    if arguments.beta is None:
        _require(
            reward_policy["guidance_schedule"] == "CONSTANT",
            "frozen reward policy schedule differs from CONSTANT",
        )
        beta = float(reward_policy["guidance_strength"])
    else:
        beta = float(arguments.beta)
    _require(
        math.isfinite(beta) and beta >= 0.0,
        "guided beta must be finite and nonnegative",
    )
    model, checkpoint, training_summary = load_checkpoint_v5(
        config,
        run_id=run_id,
        checkpoint_pass=checkpoint_pass,
        device=device,
    )
    sources = load_sources(Path(config["source_eligibility_manifest"]))
    validation_generation = config["validation_generation"]
    _require(
        len(sources) == int(validation_generation["eligible_source_count"]) == 891,
        "guided B2 generation source cohort changed",
    )
    source_limit = int(arguments.source_limit)
    if source_limit > 0:
        sources = sources[:source_limit]
    _require(
        all(
            int(source["candidate_budget"])
            == int(validation_generation["candidate_cap_per_source"])
            == 32
            for source in sources
        ),
        "guided B2 candidate cap changed",
    )
    validation_rows = load_projection_rows(
        [Path(config["validation_projection_path"])],
        allowed_splits=("VALIDATION",),
    )
    cache = SourceTokenCacheIndexV3(
        load_source_token_cache_v3(Path(config["source_token_cache_path"]))
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
    mode_prior_temperature = float(
        validation_generation.get("mode_prior_temperature", 1.0)
    )
    stop_rate_scale = float(validation_generation.get("stop_rate_scale", 1.0))
    _require(
        mode_prior_temperature > 0.0 and stop_rate_scale > 0.0,
        "V5 generation knobs are invalid",
    )
    priors, prior_compute = root_mode_priors_v4(
        model,
        source_roots,
        source_metadata,
        source_cache=cache,
        device=device,
        forward_batch_size=forward_batch_size,
    )
    tempered = [
        temper_mode_prior_v5(prior, temperature=mode_prior_temperature)
        if mode_prior_temperature != 1.0
        else prior
        for prior in priors
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
        for trajectory_slot, mode_id in enumerate(
            stratified_trajectory_mode_ids_v4(prior)
        ):
            roots.append(root)
            trajectory_metadata.append(metadata)
            mode_ids.append(int(mode_id))
            seeds.append(
                decoder_seed_base + source_index * 1_000_003 + trajectory_slot
            )
            source_indices.append(source_index)
    _require(len(roots) == len(sources) * 32, "guided B2 trajectory count changed")
    measured_rows = _read_jsonl(Path(config["measured_neighborhood_path"]))
    validate_measured_pool(measured_rows, "DEVELOPMENT", "CLOSED")
    full_manifest = load_source_manifest(
        Path(config["source_eligibility_manifest"])
    )
    manifest = {
        str(source["source_key"]): full_manifest[str(source["source_key"])]
        for source in sources
    }
    if source_limit > 0:
        manifest_keys = set(manifest)
        measured_rows = [
            row
            for row in measured_rows
            if str(row["source_key"]) in manifest_keys
        ]
        _require(bool(measured_rows), "smoke source subset has no measured neighborhood")
    measured_top_k = int(validation_generation["measured_top_k"])
    started = time.time()
    arm_summaries: dict[str, Any] = {}
    reference_rows = (
        _read_jsonl(Path(arguments.reference_trajectories))
        if arguments.reference_trajectories
        else None
    )
    reproduction: dict[str, Any] | None = None
    if "unguided" in arms:
        method_id = (
            f"unguided_xeditsetflow_v5_{run_id}_pass{checkpoint_pass}"
            f"_seed{training_seed}"
        )
        if stop_rate_scale != 1.0:
            sampled = sample_many_setflow_v5(
                model,
                roots,
                trajectory_metadata,
                mode_ids,
                seeds,
                source_cache=cache,
                device=device,
                forward_batch_size=forward_batch_size,
                stop_rate_scale=stop_rate_scale,
            )
            sampler_compute: Mapping[str, Any] | None = None
        else:
            sampled, sampler_compute_obj = sample_many_setflow_v4(
                model,
                roots,
                trajectory_metadata,
                mode_ids,
                seeds,
                source_cache=cache,
                device=device,
                forward_batch_size=forward_batch_size,
            )
            sampler_compute = vars(sampler_compute_obj)
        candidates = build_arm_candidates(
            sampled,
            sources,
            source_indices,
            seeds,
            mode_ids,
            method_id=method_id,
            mode_count=int(model.mode_count),
        )
        if reference_rows is not None:
            reproduction = check_reference_reproduction(candidates, reference_rows)
            _require(
                reproduction["reference_trajectory_match_rate"] >= 0.99,
                "unguided arm failed to reproduce the reference trajectories",
            )
        trunk_forwards = sum(int(row["generator_nfe"]) for row in candidates)
        arm_summaries["unguided"] = evaluate_arm(
            "unguided",
            candidates,
            manifest,
            measured_rows,
            measured_top_k=measured_top_k,
            compute={
                "trunk_forwards": trunk_forwards,
                "mode_head_forwards": trunk_forwards * int(model.mode_count),
                "critic_candidate_forward_equivalents": 0,
                "critic_model_batch_forwards": 0,
                "sampler_compute": sampler_compute,
                "root_prior_compute": vars(prior_compute),
            },
        )
        arm_directory = output_directory / "unguided"
        arm_directory.mkdir(parents=True, exist_ok=True)
        (arm_directory / "generated_candidates.private.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidates),
            encoding="utf-8",
        )
        _write_atomic(arm_directory / "arm_summary.json", arm_summaries["unguided"])
        _require(
            arm_summaries["unguided"]["hard_legality_rate"] == 1.0
            and arm_summaries["unguided"]["edit_budget_violation_count"] == 0
            and arm_summaries["unguided"]["candidate_budget_violation_count"] == 0,
            "unguided arm violated a hard legality or budget constraint",
        )
    if "guided" in arms:
        transform = reward_policy["potential_transform"]
        critic_kind = str(arguments.critic_kind)
        if critic_kind == "v5":
            # Pre-authorized frozen-critic substitution: the Critic V2
            # all-development refit never executed (checkpoint absent, ~80h
            # serial refit), so the session task list froze the guided-arm
            # critic to the terminal gate-passing XEditCritic V5 screen
            # checkpoint under the unchanged frozen G0 reward policy.
            v5_checkpoint = Path(arguments.v5_critic_checkpoint)
            _require(
                v5_checkpoint.is_file(),
                f"frozen XEditCritic V5 checkpoint is absent: {v5_checkpoint}",
            )
            _require(
                Path(arguments.mrnabert_model).is_dir(),
                f"mRNABERT model directory is absent: {arguments.mrnabert_model}",
            )
            critic = FrozenXEditCriticV5(
                v5_checkpoint,
                Path(arguments.mrnabert_model),
                device,
                potential_minimum=float(transform["minimum"]),
                potential_maximum=float(transform["maximum"]),
            )
            method_id = (
                f"frozen_xeditcritic_v5_guided_xeditsetflow_v5_{run_id}"
                f"_pass{checkpoint_pass}_seed{training_seed}"
            )
        else:
            _require(
                Path(arguments.critic_checkpoint).is_file(),
                f"frozen Critic V2 checkpoint is absent: {arguments.critic_checkpoint}",
            )
            _require(
                Path(arguments.mrnabert_model).is_dir(),
                f"mRNABERT model directory is absent: {arguments.mrnabert_model}",
            )
            critic = FrozenRoute2MRNABERTCritic(
                Path(arguments.critic_checkpoint),
                Path(arguments.mrnabert_model),
                device,
                potential_minimum=float(transform["minimum"]),
                potential_maximum=float(transform["maximum"]),
                encoder_attention_backend=str(arguments.encoder_attention_backend),
            )
            method_id = (
                f"frozen_mrnabert_critic_v2_guided_xeditsetflow_v5_{run_id}"
                f"_pass{checkpoint_pass}_seed{training_seed}"
            )
        sampled: list[tuple[FlowState, tuple[str, ...], int]] = []
        terminal_potentials: list[float] = []
        critic_forwards_by_source: dict[int, int] = {}
        critic_batches_by_source: dict[int, int] = {}
        for source_index, source_row in enumerate(sources):
            critic.clear_source_caches()
            batch_start = critic.model_batch_forward_count
            equivalent_start = critic.candidate_forward_equivalent_count
            first = source_index * 32
            terminals, potentials = sample_one_source_setflow_v5_guided(
                model,
                source_roots[source_index],
                source_metadata[source_index],
                mode_ids[first : first + 32],
                seeds[first : first + 32],
                source_row=source_row,
                critic=critic,
                beta=beta,
                source_cache=cache,
                device=device,
                forward_batch_size=forward_batch_size,
            )
            sampled.extend(terminals)
            terminal_potentials.extend(potentials)
            critic_forwards_by_source[source_index] = (
                critic.candidate_forward_equivalent_count - equivalent_start
            )
            critic_batches_by_source[source_index] = (
                critic.model_batch_forward_count - batch_start
            )
            _require(
                critic_forwards_by_source[source_index] > 0,
                "guided source cohort made no critic forward calls",
            )
        candidates = build_arm_candidates(
            sampled,
            sources,
            source_indices,
            seeds,
            mode_ids,
            method_id=method_id,
            mode_count=int(model.mode_count),
            critic_forwards_by_source=critic_forwards_by_source,
            terminal_potentials=terminal_potentials,
        )
        trunk_forwards = sum(int(row["generator_nfe"]) for row in candidates)
        critic_equivalents = sum(critic_forwards_by_source.values())
        arm_summaries["guided"] = evaluate_arm(
            "guided",
            candidates,
            manifest,
            measured_rows,
            measured_top_k=measured_top_k,
            compute={
                "trunk_forwards": trunk_forwards,
                "mode_head_forwards": trunk_forwards * int(model.mode_count),
                "critic_candidate_forward_equivalents": critic_equivalents,
                "critic_model_batch_forwards": sum(critic_batches_by_source.values()),
                "critic_forward_budget_rule": (
                    "GUIDED_ARM_DECLARES_EXTRA_CRITIC_FORWARDS_SEPARATELY"
                ),
                "root_prior_compute": vars(prior_compute),
                **(
                    {
                        "critic_kind": "XEDITCRITIC_V5_FROZEN_GUIDANCE",
                        "critic_bottom_six_encoded_sequences": (
                            critic.encoded_sequence_count
                        ),
                        "critic_potential_query_count": (
                            critic.potential_query_count
                        ),
                        "critic_potential_newly_scored_count": (
                            critic.potential_newly_scored_count
                        ),
                        "critic_potential_memo_hit_count": (
                            critic.potential_query_count
                            - critic.potential_newly_scored_count
                        ),
                        "critic_scoring_batch_count": critic.scoring_batch_count,
                    }
                    if critic_kind == "v5"
                    else {}
                ),
            },
        )
        arm_directory = output_directory / "guided"
        arm_directory.mkdir(parents=True, exist_ok=True)
        (arm_directory / "generated_candidates.private.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidates),
            encoding="utf-8",
        )
        _write_atomic(arm_directory / "arm_summary.json", arm_summaries["guided"])
        _require(
            arm_summaries["guided"]["hard_legality_rate"] == 1.0
            and arm_summaries["guided"]["edit_budget_violation_count"] == 0
            and arm_summaries["guided"]["candidate_budget_violation_count"] == 0,
            "guided arm violated a hard legality or budget constraint",
        )
    torch.cuda.synchronize(device)
    result = {
        "schema_version": GUIDED_RUN_SCHEMA,
        "status": "GUIDED_XEDITSETFLOW_V5_B2_RUNNER_COMPLETE",
        "run_id": run_id,
        "checkpoint_pass": checkpoint_pass,
        "training_seed": training_seed,
        "run_stage": run_stage,
        "git_head": _git_head(),
        "screen_gate_path": str(Path(arguments.screen_gate)),
        "screen_gate_selected_checkpoint_pass": int(
            gate_arm["selected_checkpoint_pass"]
        ),
        "arms_executed": arms,
        "source_count": len(sources),
        "source_limit": source_limit,
        "trajectory_count_per_source": 32,
        "decoder_seed_base": decoder_seed_base,
        "mode_prior_temperature": mode_prior_temperature,
        "stop_rate_scale": stop_rate_scale,
        "measured_top_k": measured_top_k,
        "beta": beta,
        "beta_schedule": "CONSTANT_G0_FROZEN_REWARD_POLICY",
        "transition_rule": "BASE_TRANSITION_RATE_TIMES_EXP_POTENTIAL_DIFFERENCE",
        "critic_kind": str(arguments.critic_kind),
        "critic_checkpoint_path": str(
            arguments.v5_critic_checkpoint
            if str(arguments.critic_kind) == "v5"
            else arguments.critic_checkpoint
        ),
        **(
            {"critic_guidance_provenance": critic.guidance_provenance()}
            if "guided" in arms and str(arguments.critic_kind) == "v5"
            else {}
        ),
        "critic_reward_policy_path": str(FROZEN_REWARD_POLICY_PATH),
        "encoder_attention_backend": str(arguments.encoder_attention_backend),
        "arm_summaries": arm_summaries,
        **(
            {"unguided_reference_reproduction": reproduction}
            if reproduction is not None
            else {}
        ),
        "wall_time_seconds": time.time() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        "physical_gpu_index": physical_gpu_index,
        "torch_device": str(device),
        "precision": "BF16",
        "cpu_fallback_used": False,
        "evaluation_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
        "generated_candidates_grant_canonical_credit": False,
        **cuda,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_atomic(output_directory / "guided_run_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-pass", required=True, type=int)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--screen-gate", required=True, type=Path)
    parser.add_argument(
        "--arms",
        default="unguided,guided",
        help="comma-separated nonempty subset of {unguided, guided}",
    )
    parser.add_argument(
        "--source-limit",
        type=int,
        default=0,
        help="restrict to the first N sources (smoke test); 0 = all 891",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="constant guidance strength; defaults to the frozen G0 reward policy value",
    )
    parser.add_argument(
        "--critic-kind",
        default="v5",
        choices=["v2", "v5"],
        help=(
            "frozen guided-arm critic family; v5 is the pre-authorized "
            "substitution for the never-executed Critic V2 refit"
        ),
    )
    parser.add_argument(
        "--critic-checkpoint",
        type=Path,
        default=EXPECTED_CRITIC_CHECKPOINT,
    )
    parser.add_argument(
        "--v5-critic-checkpoint",
        type=Path,
        default=EXPECTED_XEDITCRITIC_V5_CHECKPOINT,
    )
    parser.add_argument(
        "--mrnabert-model",
        type=Path,
        default=EXPECTED_MRNABERT_MODEL,
    )
    parser.add_argument(
        "--encoder-attention-backend",
        default="OFFICIAL_PYTORCH_FALLBACK",
        choices=["OFFICIAL_PYTORCH_FALLBACK", "PYTORCH_SDPA_AUTO"],
    )
    parser.add_argument(
        "--reference-trajectories",
        type=Path,
        default=None,
        help="optional screen validation trajectories to verify unguided reproduction",
    )
    arguments = parser.parse_args()
    failure_config = {
        "config": str(arguments.config),
        "run_id": arguments.run_id,
        "checkpoint_pass": arguments.checkpoint_pass,
        "physical_gpu_index": arguments.physical_gpu_index,
        "device": f"cuda:{arguments.physical_gpu_index}",
        "output_directory": str(arguments.output_dir),
        "arms": arguments.arms,
        "critic_kind": str(arguments.critic_kind),
    }
    try:
        result = execute(arguments)
    except Exception as error:
        failure_path = arguments.output_dir.with_name(
            arguments.output_dir.name + ".failed.json"
        )
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        with failure_path.open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": RUNNER_FAILURE_SCHEMA,
                    "status": "STOPPED_WITH_EVIDENCE",
                    "entrypoint": "run_route2_guided_xeditsetflow_v5_v1",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "cpu_fallback_used": False,
                    "evaluation_outcomes_accessed": False,
                    "new_final_evaluation_outcome_reads": 0,
                    "requested_cuda_observation": cuda_device_observation(
                        arguments.physical_gpu_index
                    ),
                    **failure_config,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
