#!/usr/bin/env python3
"""Validate a learned Route 2 base-flow checkpoint on legal unguided trajectories."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_base_flow_model import Route2BaseFlowModel
from core.route2_gpu_failure_evidence import cuda_device_observation, write_gpu_failure_evidence
from core.route2_legal_xeditflow import (
    STOP,
    FlowState,
    LegalAction,
    apply_action,
    exact_terminal_distribution,
    initial_state,
    jump_distribution,
    legal_actions,
)


TOKEN = {"A": 0, "C": 1, "G": 2, "U": 3}
BASE = "ACGU"
REGION = {"5UTR": 0, "3UTR": 1}


class G0ValidationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G0ValidationError(message)


def _normalize(value: Any) -> str:
    sequence = str(value).upper().replace("T", "U")
    _require(sequence and set(sequence) <= set(BASE), "source is outside the RNA alphabet")
    return sequence


def load_sources(path: Path) -> list[dict[str, Any]]:
    sources = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = str(row["source_key"])
            _require(key not in seen, f"source eligibility duplicated: {key}")
            seen.add(key)
            _require(row["generated_candidates_grant_canonical_credit"] is False, "generated candidate credit was enabled")
            _require(row["evaluation_outcomes_included"] is False, "Evaluation outcome entered G0 source eligibility")
            sources.append({
                **row,
                "source_key": key,
                "source_sequence": _normalize(row["source_sequence"]),
                "edit_budget": int(row["edit_budget"]),
                "candidate_budget": int(row["candidate_budget"]),
            })
    _require(sources, "source eligibility is empty")
    return sources


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Route2BaseFlowModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def validate_checkpoint_training_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    physical_index = provenance.get("physical_gpu_index")
    observed_total = provenance.get("cuda_total_memory_mb")
    _require(
        provenance.get("optimizer_steps", 0) > 0
        and provenance.get("parameter_changed") is True
        and provenance.get("cuda_training_tensors_verified") is True
        and isinstance(physical_index, int)
        and not isinstance(physical_index, bool)
        and physical_index >= 0
        and provenance.get("torch_device") == f"cuda:{physical_index}"
        and provenance.get("cpu_fallback_used") is False
        and provenance.get("cuda_device_index") == physical_index
        and isinstance(provenance.get("cuda_device_uuid"), str)
        and bool(provenance.get("cuda_device_uuid"))
        and isinstance(observed_total, (int, float))
        and not isinstance(observed_total, bool)
        and math.isfinite(float(observed_total))
        and float(observed_total) > 0.0,
        "checkpoint does not prove a learned GPU parameter update with observed CUDA provenance",
    )
    return dict(provenance)


def _model_inputs(
    state: FlowState,
    *,
    region_id: int,
    assay_id: int,
    context_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    source = torch.tensor([[TOKEN[base] for base in state.source_sequence]], dtype=torch.long, device=device)
    current = torch.tensor([[TOKEN[base] for base in state.current_sequence]], dtype=torch.long, device=device)
    padding = torch.zeros_like(source, dtype=torch.bool)
    return (
        source,
        current,
        padding,
        torch.tensor([region_id], device=device),
        torch.tensor([assay_id], device=device),
        torch.tensor([context_id], device=device),
        torch.tensor([state.remaining_budget], device=device),
    )


@torch.no_grad()
def learned_rate_function(
    model: Route2BaseFlowModel,
    *,
    region_id: int,
    assay_id: int,
    context_id: int,
    device: torch.device,
):
    def score(state: FlowState, actions):
        rates, mask = model.rates(*_model_inputs(
            state,
            region_id=region_id,
            assay_id=assay_id,
            context_id=context_id,
            device=device,
        ))
        length = len(state.source_sequence)
        result = {}
        for action in actions:
            index = length * 4 if action.kind == STOP else int(action.position) * 4 + TOKEN[str(action.alt_base)]
            _require(bool(mask[0, index].item()), f"model mask rejected legal action: {action.action_id}")
            rate = float(rates[0, index].item())
            _require(math.isfinite(rate) and rate > 0.0, f"model rate is invalid: {action.action_id}")
            result[action] = max(0.0, rate - model.support_floor)
        return result

    return score


def sample_one(
    root: FlowState,
    rate_function,
    *,
    seed: int,
    device: torch.device,
) -> tuple[FlowState, tuple[str, ...], int]:
    _require(device.type == "cuda", "learned trajectory sampling requires CUDA")
    generator = torch.Generator(device=device).manual_seed(seed)
    state = root
    actions_taken = []
    forward_count = 0
    while state.terminal_cause is None:
        edges = jump_distribution(state, rate_function, support_floor=1e-8)
        probabilities = torch.tensor(
            [probability for _, _, probability in edges], dtype=torch.float64, device=device
        )
        choice = int(torch.multinomial(probabilities, 1, generator=generator).item())
        action, child, _probability = edges[choice]
        _require(action in legal_actions(state), "sampled action was not hard-legal")
        if action.kind != STOP:
            _require(action.position not in {position for position, _ in state.source_relative_edits}, "sampled a repeated edit")
        actions_taken.append(action.action_id)
        state = child
        forward_count += 1
    return state, tuple(actions_taken), forward_count


def _enumerate_paths(root: FlowState, rate_function) -> dict[FlowState, float]:
    terminal: dict[FlowState, float] = defaultdict(float)

    def visit(state: FlowState, probability: float) -> None:
        if state.terminal_cause is not None:
            terminal[state] += probability
            return
        for _action, child, jump_probability in jump_distribution(state, rate_function, support_floor=1e-8):
            visit(child, probability * jump_probability)

    visit(root, 1.0)
    _require(math.isclose(math.fsum(terminal.values()), 1.0, abs_tol=1e-12), "path enumeration mass does not close")
    return dict(terminal)


def learned_small_graph_check(model, checkpoint, device: torch.device) -> dict[str, Any]:
    assay_id = checkpoint["assay_vocab"].get("__UNK__", 0)
    context_id = checkpoint["context_vocab"].get("__UNK__", 0)
    root = initial_state("AC", budget=2, assay_id="__UNK__", context_id="__UNK__")
    rate_function = learned_rate_function(
        model, region_id=0, assay_id=assay_id, context_id=context_id, device=device
    )
    dp = exact_terminal_distribution(root, rate_function, support_floor=1e-8)
    enumeration = _enumerate_paths(root, rate_function)
    states = set(dp) | set(enumeration)
    total_variation = 0.5 * math.fsum(abs(dp.get(state, 0.0) - enumeration.get(state, 0.0)) for state in states)
    _require(total_variation <= 1e-12, "learned small-graph DP differs from complete path enumeration")
    return {
        "source_length": 2,
        "edit_budget": 2,
        "terminal_state_count": len(states),
        "dp_probability_sum": math.fsum(dp.values()),
        "enumeration_probability_sum": math.fsum(enumeration.values()),
        "total_variation": total_variation,
        "tolerance": 1e-12,
        "status": "PASS",
    }


def sampling_efficiency_summary(
    candidate_rows: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    *,
    generator_nfe: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    _require(candidate_rows, "no trajectories were sampled")
    _require(elapsed_seconds > 0.0, "sampling wall time is not positive")
    expected_by_source = {
        str(source["source_key"]): int(source["candidate_budget"])
        for source in sources
    }
    actual_by_source = Counter(str(row["source_key"]) for row in candidate_rows)
    _require(set(actual_by_source) == set(expected_by_source), "sampled source coverage differs")
    mismatch_count = sum(
        abs(actual_by_source[source_key] - candidate_budget)
        for source_key, candidate_budget in expected_by_source.items()
    )
    unique_by_source: dict[str, set[str]] = defaultdict(set)
    for row in candidate_rows:
        unique_by_source[str(row["source_key"])].add(str(row["candidate_sequence"]))
    source_unique_rates = [
        len(unique_by_source[source_key]) / actual_by_source[source_key]
        for source_key in expected_by_source
    ]
    trajectory_count = len(candidate_rows)
    unique_candidate_count = sum(len(values) for values in unique_by_source.values())
    sampling_invocation_count = 2 * trajectory_count
    generator_nfe_with_replay = 2 * generator_nfe
    return {
        "candidate_budget_violation_count": mismatch_count,
        "unique_candidate_count": unique_candidate_count,
        "duplicate_candidate_count": trajectory_count - unique_candidate_count,
        "global_unique_candidate_rate": unique_candidate_count / trajectory_count,
        "source_macro_unique_candidate_rate": math.fsum(source_unique_rates) / len(source_unique_rates),
        "mean_generator_nfe_per_trajectory": generator_nfe / trajectory_count,
        "validation_candidate_outputs_per_second": trajectory_count / elapsed_seconds,
        "validation_sampling_invocation_count": sampling_invocation_count,
        "validation_sampling_invocations_per_second": sampling_invocation_count / elapsed_seconds,
        "validation_generator_nfe_with_replay": generator_nfe_with_replay,
        "validation_generator_nfe_per_second_with_replay": generator_nfe_with_replay / elapsed_seconds,
        "replay_sampling_overhead_included_in_wall_time": True,
    }


def validate(
    model,
    checkpoint,
    sources: list[dict[str, Any]],
    *,
    device: torch.device,
    seed: int,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_rows = []
    terminal_causes = Counter()
    total_actions = 0
    replay_failures = 0
    budget_violations = 0
    started = time.time()
    for source_index, source_row in enumerate(sources):
        source = source_row["source_sequence"]
        root = initial_state(
            source,
            budget=source_row["edit_budget"],
            assay_id=str(source_row["assay_id"]),
            context_id=str(source_row["biological_context_id"]),
        )
        region_text = str(source_row["region"]).replace("′", "").replace("'", "")
        _require(region_text in REGION, f"unsupported region: {source_row['region']}")
        rate_function = learned_rate_function(
            model,
            region_id=REGION[region_text],
            assay_id=checkpoint["assay_vocab"].get(str(source_row["assay_id"]), 0),
            context_id=checkpoint["context_vocab"].get(str(source_row["biological_context_id"]), 0),
            device=device,
        )
        for candidate_index in range(source_row["candidate_budget"]):
            trajectory_seed = seed + source_index * 1_000_003 + candidate_index
            terminal, action_ids, forwards = sample_one(
                root, rate_function, seed=trajectory_seed, device=device
            )
            replay_terminal, replay_actions, replay_forwards = sample_one(
                root, rate_function, seed=trajectory_seed, device=device
            )
            replay_ok = terminal == replay_terminal and action_ids == replay_actions and forwards == replay_forwards
            replay_failures += int(not replay_ok)
            edit_count = terminal.edit_count
            budget_violations += int(edit_count > source_row["edit_budget"])
            total_actions += forwards
            terminal_causes[terminal.terminal_cause] += 1
            candidate_rows.append({
                "method_id": "unguided_learned_base_flow_g0",
                "source_key": source_row["source_key"],
                "candidate_sequence": terminal.current_sequence,
                "terminal_cause": terminal.terminal_cause,
                "edit_count": edit_count,
                "trajectory_actions": list(action_ids),
                "trajectory_seed": trajectory_seed,
                "trajectory_replay_ok": replay_ok,
                "generator_nfe": forwards,
                "critic_forwards": 0,
                "independent_evaluator_forwards": 0,
                "generated_candidate_grants_canonical_credit": False,
            })
        if progress is not None:
            progress({
                "event": "SOURCE_COHORT_COMPLETED",
                "completed_source_cohort_count": source_index + 1,
                "total_source_cohort_count": len(sources),
                "source_key": source_row["source_key"],
                "trajectory_count": len(candidate_rows),
                "generator_nfe": total_actions,
            })
    empirical_counts = Counter(
        (row["source_key"], row["candidate_sequence"])
        for row in candidate_rows
    )
    empirical_totals = Counter(row["source_key"] for row in candidate_rows)
    for row in candidate_rows:
        row["generation_score"] = math.log(
            empirical_counts[(row["source_key"], row["candidate_sequence"])]
            / empirical_totals[row["source_key"]]
        )
    elapsed = time.time() - started
    efficiency = sampling_efficiency_summary(
        candidate_rows,
        sources,
        generator_nfe=total_actions,
        elapsed_seconds=elapsed,
    )
    small_graph = learned_small_graph_check(model, checkpoint, device)
    numerical_failure_count = terminal_causes.get("NUMERICAL_FAILURE", 0)
    ready = (
        replay_failures == 0
        and budget_violations == 0
        and efficiency["candidate_budget_violation_count"] == 0
        and numerical_failure_count == 0
    )
    summary = {
        "schema_version": "route_a_v3_route2_base_flow_g0_validation.v1",
        "status": "FLOW_G0_READY" if ready else "FLOW_G0_VALIDATION_FAIL",
        "source_budget_cohort_count": len(sources),
        "trajectory_count": len(candidate_rows),
        "hard_legality_rate": 1.0,
        "edit_budget_violation_count": budget_violations,
        "numerical_failure_count": numerical_failure_count,
        "trajectory_replay_failure_count": replay_failures,
        "terminal_causes": dict(sorted(terminal_causes.items())),
        "generator_nfe": total_actions,
        "wall_time_seconds": elapsed,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "trajectory_sampling_device": str(device),
        "learned_parameter_update_checkpoint_loaded": True,
        "small_graph_reference": small_graph,
        "distinguishable_terminal_causes": [
            "EXPLICIT_STOP", "BUDGET_EXHAUSTED", "NO_LEGAL_ACTION", "NUMERICAL_FAILURE"
        ],
        "guided_critic_used": False,
        "evaluation_outcomes_read": 0,
        "generated_candidates_grant_canonical_credit": False,
        "biological_optimization_established": False,
        **efficiency,
    }
    return candidate_rows, summary


def execute(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"output already exists: {output_dir}")
    _require(str(config["device"]).startswith("cuda"), "G0 learned validation requires CUDA")
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden for physical-device provenance")
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    device = torch.device(str(config["device"]))
    _require(0 <= int(config["physical_gpu_index"]) < torch.cuda.device_count(), "physical GPU index is unavailable")
    _require(device.index == int(config["physical_gpu_index"]), "CUDA device index differs from declared physical GPU")
    torch.cuda.set_device(device)
    cuda_provenance = cuda_device_observation(int(config["physical_gpu_index"]), require_physical_index_match=True)
    sources = load_sources(Path(config["source_eligibility_manifest"]))
    model, checkpoint = load_model(Path(config["checkpoint_path"]), device)
    provenance = validate_checkpoint_training_provenance(checkpoint.get("training_provenance", {}))
    _require(
        0 <= int(provenance.get("physical_gpu_index", -1)) < torch.cuda.device_count(),
        "checkpoint physical GPU provenance is absent or unavailable",
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    serialized_config = json.dumps(dict(config), indent=2, sort_keys=True) + "\n"
    (output_dir / "validation_config.json").write_text(serialized_config, encoding="utf-8")
    (output_dir / "config.yaml").write_text(serialized_config, encoding="utf-8")
    log_path = output_dir / "validation.log"
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    def progress(row: Mapping[str, Any]) -> None:
        serialized = json.dumps(dict(row), sort_keys=True) + "\n"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(serialized)
        if row.get("event") == "SOURCE_COHORT_COMPLETED":
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(serialized)

    progress({
        "event": "VALIDATION_STARTED",
        "device": str(device),
        "physical_gpu_index": int(config["physical_gpu_index"]),
        "cuda_device_uuid": cuda_provenance["cuda_device_uuid"],
        "source_cohort_count": len(sources),
    })
    rows, summary = validate(
        model, checkpoint, sources, device=device, seed=int(config["seed"]), progress=progress
    )
    summary["physical_gpu_index"] = int(config["physical_gpu_index"])
    summary["device"] = str(device)
    summary["cpu_fallback_used"] = False
    summary["checkpoint_gpu_parameter_update_provenance_verified"] = True
    summary["checkpoint_training_device"] = str(provenance["torch_device"])
    summary["checkpoint_training_physical_gpu_index"] = int(provenance["physical_gpu_index"])
    summary["checkpoint_cpu_fallback_used"] = bool(provenance["cpu_fallback_used"])
    summary["checkpoint_training_seed"] = int(provenance["seed"])
    summary["checkpoint_training_optimizer_steps"] = int(provenance["optimizer_steps"])
    summary["checkpoint_parameter_changed"] = bool(provenance["parameter_changed"])
    summary["checkpoint_cuda_training_tensors_verified"] = bool(provenance["cuda_training_tensors_verified"])
    summary["checkpoint_training_cuda_device_index"] = int(provenance["cuda_device_index"])
    summary["checkpoint_training_cuda_device_uuid"] = str(provenance["cuda_device_uuid"])
    summary["checkpoint_training_cuda_total_memory_mb"] = float(provenance["cuda_total_memory_mb"])
    summary.update(cuda_provenance)
    (output_dir / "trajectories.private.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    serialized_summary = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (output_dir / "validation_summary.json").write_text(serialized_summary, encoding="utf-8")
    (output_dir / "final_summary.json").write_text(serialized_summary, encoding="utf-8")
    progress({"event": "VALIDATION_COMPLETED", "trajectory_count": summary["trajectory_count"]})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or Path(config["output_directory"])
    try:
        result = execute(config, output_dir)
    except Exception as exc:
        write_gpu_failure_evidence(
            output_dir.with_name(output_dir.name + ".failed.json"), config, exc,
            entrypoint="run_route2_base_flow_g0_validation_v1",
            evaluation_outcomes_accessed=False,
        )
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
