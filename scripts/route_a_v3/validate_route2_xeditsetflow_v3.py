#!/usr/bin/env python3
"""Validate unguided XEditSetFlow V3 recovery and G0 correctness."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_development_projection_v3 import load_projection_rows
from core.route2_gpu_failure_evidence import cuda_device_observation, write_gpu_failure_evidence
from core.route2_legal_xeditflow import exact_terminal_distribution, initial_state, jump_distribution
from core.route2_source_token_cache_v3 import SourceTokenCacheIndexV3, assemble_source_token_cache_v3, load_source_token_cache_v3
from core.route2_xeditsetflow_runtime_v3 import ARM_CONFIGS_V3, build_setflow_arm_v3
from core.route2_xeditsetflow_gate_v3 import require_setflow_confirmation_authorization_v3
from core.route2_xeditsetflow_sampling_v3 import (
    SetFlowGenerationMetadataV3,
    build_generation_metadata_v3,
    sample_many_setflow_v3,
    setflow_rate_map_v3,
)
from scripts.route_a_v3.evaluate_route2_generation_v1 import (
    evaluate_generation,
    load_source_manifest,
    measured_neighborhood_metrics,
    validate_measured_pool,
)
from scripts.route_a_v3.run_route2_base_flow_g0_validation_v1 import load_sources


class SetFlowValidationV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SetFlowValidationV3Error(message)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    _require(bool(rows), f"validation input is empty: {path}")
    return rows


def load_setflow_checkpoint_v3(
    checkpoint_path: Path, arm: str, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _require(checkpoint.get("schema_version") == "route_a_v3_route2_xeditsetflow_checkpoint.v3", "unexpected SetFlow checkpoint schema")
    _require(checkpoint.get("arm") == arm, "SetFlow checkpoint arm changed")
    model, expected = build_setflow_arm_v3(
        arm, vocabs=checkpoint["vocabs"], dropout=float(checkpoint["model_config"].get("dropout", 0.1))
    )
    _require(expected == checkpoint["model_config"], "SetFlow checkpoint model geometry changed")
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    provenance = checkpoint.get("training_provenance") or {}
    _require(
        int(provenance.get("optimizer_steps", 0)) > 0
        and provenance.get("parameter_changed") is True
        and provenance.get("cpu_fallback_used") is False
        and provenance.get("cuda_training_tensors_verified") is True,
        "SetFlow checkpoint lacks a learned CUDA update",
    )
    return model, checkpoint


def _enumerate_terminal_paths(root, rate_function):
    terminal = defaultdict(float)

    def visit(state, probability):
        if state.terminal_cause is not None:
            terminal[state] += probability
            return
        for _action, child, edge_probability in jump_distribution(state, rate_function, support_floor=1e-8):
            visit(child, probability * edge_probability)

    visit(root, 1.0)
    return dict(terminal)


def small_graph_exact_check_v3(model, arm: str, device: torch.device) -> dict[str, Any]:
    source = "AC"
    synthetic_cache = SourceTokenCacheIndexV3(assemble_source_token_cache_v3(
        [{"canonical_record_id": "small-graph", "source_sequence": source}],
        sequence_to_index={source: 0},
        encoded_tokens={0: torch.zeros(2, 768)},
        model_id="SYNTHETIC_ZERO_FEATURE_FOR_DISTRIBUTION_CHECK_ONLY",
        pretrained_parameter_count=113_389_056,
        attention_backend="NOT_APPLICABLE_SYNTHETIC_CHECK",
    ))
    metadata = SetFlowGenerationMetadataV3("small-graph", 0, 0, 0, 0, 0, 0, 0)
    root = initial_state(source, budget=2, assay_id="__UNK__", context_id="__UNK__")

    def rate_function(state, actions):
        return setflow_rate_map_v3(
            model, arm, state, metadata, actions, source_cache=synthetic_cache, device=device
        )

    dynamic = exact_terminal_distribution(root, rate_function, support_floor=1e-8)
    enumerated = _enumerate_terminal_paths(root, rate_function)
    states = set(dynamic) | set(enumerated)
    total_variation = 0.5 * math.fsum(
        abs(dynamic.get(state, 0.0) - enumerated.get(state, 0.0)) for state in states
    )
    _require(total_variation <= 1e-12, "SetFlow small-graph DP differs from independent path enumeration")
    return {
        "source_length": 2,
        "edit_budget": 2,
        "terminal_state_count": len(states),
        "dynamic_probability_sum": math.fsum(dynamic.values()),
        "enumeration_probability_sum": math.fsum(enumerated.values()),
        "total_variation": total_variation,
        "tolerance": 1e-12,
        "status": "PASS",
        "source_token_feature_policy": "SYNTHETIC_ZERO_FEATURE_MECHANICS_CHECK_NOT_PERFORMANCE",
    }


def validate(config: Mapping[str, Any], *, arm: str, output_dir: Path) -> dict[str, Any]:
    _require(arm in ARM_CONFIGS_V3, "unknown SetFlow validation arm")
    require_setflow_confirmation_authorization_v3(config, arm=arm)
    _require(not output_dir.exists(), f"terminal SetFlow validation output exists: {output_dir}")
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden")
    device = torch.device(str(config["device"]))
    physical_gpu = int(config["physical_gpu_index"])
    _require(device == torch.device(f"cuda:{physical_gpu}"), "SetFlow validation device provenance changed")
    torch.cuda.set_device(device)
    cuda = cuda_device_observation(physical_gpu, require_physical_index_match=True)
    checkpoint_path = Path(config["output_root"]) / arm / "best.pt"
    model, checkpoint = load_setflow_checkpoint_v3(checkpoint_path, arm, device)
    sources = load_sources(Path(config["source_eligibility_manifest"]))
    _require(len(sources) == 891, "SetFlow generation source cohort changed")
    _require(all(int(source["candidate_budget"]) == int(config["candidate_budget_per_source"]) == 32 for source in sources), "SetFlow candidate cap changed")
    validation_rows = load_projection_rows([Path(config["validation_projection_path"])], allowed_splits=("VALIDATION",))
    cache = SourceTokenCacheIndexV3(load_source_token_cache_v3(Path(config["source_token_cache_path"])))
    source_metadata = build_generation_metadata_v3(sources, validation_rows, checkpoint["vocabs"])
    roots = []
    trajectory_metadata = []
    seeds = []
    source_indices = []
    for source_index, (source, metadata) in enumerate(zip(sources, source_metadata, strict=True)):
        root = initial_state(
            source["source_sequence"],
            budget=int(source["edit_budget"]),
            assay_id=str(source["assay_id"]),
            context_id=str(source["biological_context_id"]),
        )
        for candidate_index in range(32):
            roots.append(root)
            trajectory_metadata.append(metadata)
            seeds.append(int(config["seed"]) + source_index * 1_000_003 + candidate_index)
            source_indices.append(source_index)
    started = time.time()
    sampled, primary_batches = sample_many_setflow_v3(
        model,
        arm,
        roots,
        trajectory_metadata,
        seeds,
        source_cache=cache,
        device=device,
        forward_batch_size=int(config["trajectory_forward_batch_size"]),
    )
    replayed, replay_batches = sample_many_setflow_v3(
        model,
        arm,
        roots,
        trajectory_metadata,
        seeds,
        source_cache=cache,
        device=device,
        forward_batch_size=int(config["trajectory_forward_batch_size"]),
    )
    elapsed = time.time() - started
    candidates = []
    replay_failures = 0
    budget_violations = 0
    terminal_causes = Counter()
    generator_nfe = 0
    method_id = f"unguided_xeditsetflow_v3_{arm}_seed{config['seed']}"
    for trajectory_index, (first, second) in enumerate(zip(sampled, replayed, strict=True)):
        terminal, actions, forwards = first
        replay_terminal, replay_actions, replay_forwards = second
        replay_ok = terminal == replay_terminal and actions == replay_actions and forwards == replay_forwards
        replay_failures += int(not replay_ok)
        source = sources[source_indices[trajectory_index]]
        budget_violations += int(terminal.edit_count > int(source["edit_budget"]))
        terminal_causes[str(terminal.terminal_cause)] += 1
        generator_nfe += forwards
        candidates.append({
            "method_id": method_id,
            "source_key": source["source_key"],
            "candidate_sequence": terminal.current_sequence,
            "terminal_cause": terminal.terminal_cause,
            "edit_count": terminal.edit_count,
            "trajectory_actions": list(actions),
            "trajectory_seed": seeds[trajectory_index],
            "trajectory_replay_ok": replay_ok,
            "generator_nfe": forwards,
            "critic_forwards": 0,
            "independent_evaluator_forwards": 0,
            "generated_candidate_grants_canonical_credit": False,
        })
    empirical = Counter((row["source_key"], row["candidate_sequence"]) for row in candidates)
    totals = Counter(row["source_key"] for row in candidates)
    for row in candidates:
        row["generation_score"] = math.log(empirical[(row["source_key"], row["candidate_sequence"])] / totals[row["source_key"]])
    manifest = load_source_manifest(Path(config["source_eligibility_manifest"]))
    generation = evaluate_generation(manifest, candidates)
    measured_rows = _read_jsonl(Path(config["measured_neighborhood_path"]))
    validate_measured_pool(measured_rows, "DEVELOPMENT", "CLOSED")
    measured = measured_neighborhood_metrics(
        manifest,
        candidates,
        measured_rows,
        k=int(config["measured_top_k"]),
        candidate_support_mode="OPEN_GENERATED_SUPPORT",
    )
    small_graph = small_graph_exact_check_v3(model, arm, device)
    numerical_failures = terminal_causes.get("NUMERICAL_FAILURE", 0)
    correctness = (
        generation["hard_legality_rate"] == 1.0
        and budget_violations == 0
        and generation["candidate_budget_violation_count"] == 0
        and replay_failures == 0
        and numerical_failures == 0
        and small_graph["status"] == "PASS"
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    (output_dir / "trajectories.private.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidates),
        encoding="utf-8",
    )
    result = {
        "schema_version": "route_a_v3_route2_xeditsetflow_unguided_validation.v3",
        "status": "FLOW_G0_READY" if correctness else "FLOW_G0_VALIDATION_FAIL",
        "arm": arm,
        "seed": int(config["seed"]),
        "method_id": method_id,
        "source_count": len(sources),
        "candidate_count": len(candidates),
        "hard_legality_rate": generation["hard_legality_rate"],
        "edit_budget_violation_count": budget_violations,
        "candidate_budget_violation_count": generation["candidate_budget_violation_count"],
        "trajectory_replay_failure_count": replay_failures,
        "numerical_failure_count": numerical_failures,
        "source_macro_unique_candidate_rate": generation["source_macro_unique_candidate_rate"],
        "source_macro_candidate_recovery_rate": measured["source_macro_candidate_recovery_rate"],
        "source_macro_measured_top_k_recovery_at_k": measured["source_macro_measured_top_k_recovery_at_k"],
        "terminal_causes": dict(sorted(terminal_causes.items())),
        "generator_nfe": generator_nfe,
        "model_forward_batch_count_with_replay": primary_batches + replay_batches,
        "trajectory_forward_batch_size": int(config["trajectory_forward_batch_size"]),
        "small_graph_reference": small_graph,
        "wall_time_seconds": elapsed,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "physical_gpu_index": physical_gpu,
        "torch_device": str(device),
        "cpu_fallback_used": False,
        "parameter_update_count": 0,
        "guided_critic_used": False,
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed": False,
        "evaluation_records_read": 0,
        "evaluation_outcomes_accessed": False,
        "generated_candidates_grant_canonical_credit": False,
        "biological_optimization_established": False,
        "generation_metrics": generation,
        "measured_neighborhood_metrics": measured,
        **cuda,
    }
    (output_dir / "validation_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=tuple(ARM_CONFIGS_V3))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or Path(config["validation_output_root"]) / args.arm
    try:
        result = validate(config, arm=args.arm, output_dir=output_dir)
    except Exception as exc:
        write_gpu_failure_evidence(
            output_dir.with_name(output_dir.name + ".failed.json"),
            config,
            exc,
            entrypoint="validate_route2_xeditsetflow_v3",
            evaluation_outcomes_accessed=False,
        )
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
