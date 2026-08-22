#!/usr/bin/env python3
"""Score the common closed neighborhood with frozen rerank/search methods."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_development_projection_v3 import load_projection_rows
from core.route2_gpu_failure_evidence import cuda_device_observation, write_gpu_failure_evidence
from core.route2_legal_xeditflow import FlowState, initial_state, validate_state
from core.route2_xeditflow_gate_v3 import authorize_xeditflow_guidance_v3
from core.route2_xeditflow_matched_methods_v3 import SourceAnchoredFirstOrderPotentialV3
from core.route2_xeditflow_value_training_v3 import CRITIC_SEEDS_V3
from scripts.route_a_v3.adapt_route2_xeditflow_strongest_baseline_v3 import (
    adapt_strongest_baseline_v3,
)
from scripts.route_a_v3.run_route2_base_flow_g0_validation_v1 import load_sources
from scripts.route_a_v3.run_route2_search_generation_baselines_v1 import TorchCheckpointScorer
from scripts.route_a_v3.run_route2_xeditflow_matched_controls_v3 import (
    FrozenCriticEnsembleRewardV3,
)
from scripts.route_a_v3.score_route2_xeditflow_critic_ensemble_v3 import _representatives_v3


METHODS_V3 = {
    "first_order_guidance",
    "simple_rate_guidance",
    "generate_then_rerank",
    "strongest_matched_baseline",
}


class XEditFlowClosedFrozenScoreV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowClosedFrozenScoreV3Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(bool(rows) and all(isinstance(row, dict) for row in rows), f"JSONL input is empty or invalid: {path}")
    return rows


def validate_closed_frozen_score_config_v3(config: Mapping[str, Any]) -> None:
    _require(config.get("schema_version") == "route_a_v3_route2_xeditflow_closed_frozen_score_config.v1", "unexpected closed frozen-score config")
    method = str(config.get("method_id"))
    _require(method in METHODS_V3, "closed frozen-score method differs")
    _require(config.get("pool_assignment") == "DEVELOPMENT" and config.get("split") == "VALIDATION", "closed frozen-score cohort differs")
    _require(int(config.get("base_flow_training_seed", -1)) in {20260904, 20260905, 20260906}, "closed frozen-score base-flow seed differs")
    _require(int(config.get("expected_source_count", -1)) == 891, "closed frozen-score source count differs")
    gpu = int(config.get("physical_gpu_index", -1))
    _require(gpu in set(range(6)) and config.get("device") == f"cuda:{gpu}", "closed frozen-score GPU provenance differs")
    if method != "strongest_matched_baseline":
        _require(float(config.get("kappa", -1)) in {0.0, 0.5, 1.0}, "closed rerank kappa differs")
        _require(int(config.get("critic_online_microbatch_size", -1)) == 4, "closed rerank Critic microbatch differs")
        _require(bool(config.get("critic_refit_manifest_path")), "closed rerank refit manifest is absent")
    else:
        _require(bool(config.get("strongest_generation_baseline_path")), "closed strongest artifact is absent")
        _require(bool(config.get("baseline_selection_input_path")), "closed strongest selection input is absent")


def _terminal_state_v3(root: FlowState, candidate_sequence: str) -> FlowState:
    candidate = str(candidate_sequence).upper().replace("T", "U")
    _require(len(candidate) == len(root.source_sequence), "closed frozen-score candidate length differs")
    edits = tuple(
        (index, right)
        for index, (left, right) in enumerate(zip(root.source_sequence, candidate))
        if left != right
    )
    _require(len(edits) <= root.remaining_budget, "closed frozen-score candidate exceeds edit budget")
    state = FlowState(
        source_sequence=root.source_sequence,
        current_sequence=candidate,
        source_relative_edits=edits,
        remaining_budget=root.remaining_budget - len(edits),
        assay_id=root.assay_id,
        context_id=root.context_id,
        terminal_cause=("BUDGET_EXHAUSTED" if len(edits) == root.remaining_budget else "EXPLICIT_STOP"),
    )
    validate_state(state)
    return state


def score_critic_states_for_method_v3(
    method: str,
    root: FlowState,
    states: list[FlowState],
    reward_provider,
):
    _require(
        method in {"first_order_guidance", "simple_rate_guidance", "generate_then_rerank"},
        "closed Critic-score method differs",
    )
    return (
        SourceAnchoredFirstOrderPotentialV3(root, reward_provider)(states)
        if method == "first_order_guidance"
        else reward_provider(states)
    )


def run(config: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
    validate_closed_frozen_score_config_v3(config)
    _require(output_dir == Path(config["output_dir"]), "closed frozen-score output path differs")
    _require(not output_dir.exists(), f"closed frozen-score output exists: {output_dir}")
    authorization = authorize_xeditflow_guidance_v3(
        _json(Path(config["critic_readiness_path"])),
        _json(Path(config["setflow_confirmation_path"])),
    )
    _require(authorization["guidance_authorized"] is True, "closed frozen scoring remains blocked before readiness")
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden")
    gpu = int(config["physical_gpu_index"])
    device = torch.device(str(config["device"]))
    torch.cuda.set_device(device)
    _require(torch.cuda.is_bf16_supported(), "BF16 is unavailable for closed frozen scoring")
    cuda = cuda_device_observation(gpu, require_physical_index_match=True)
    sources = load_sources(Path(config["source_eligibility_manifest"]))
    _require(len(sources) == int(config["expected_source_count"]), "closed frozen-score source cohort changed")
    source_by_key = {str(row["source_key"]): row for row in sources}
    measured = _jsonl(Path(config["measured_neighborhood_path"]))
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    for row in measured:
        _require(row.get("pool_assignment") == "DEVELOPMENT" and row.get("split") == "VALIDATION", "closed frozen-score measured row differs")
        source_key = str(row["source_key"])
        _require(source_key in source_by_key, "closed frozen-score measured source differs")
        key = (source_key, str(row["candidate_sequence"]))
        _require(key not in seen, "closed frozen-score candidate is duplicated")
        seen.add(key)
        by_source[source_key].append(row)
    _require(set(by_source) == set(source_by_key), "closed frozen-score source coverage differs")
    method = str(config["method_id"])
    critic = None
    representatives = None
    strongest_scorer = None
    if method != "strongest_matched_baseline":
        refit = _json(Path(config["critic_refit_manifest_path"]))
        _require(refit.get("status") == "XEDITCRITIC_V3_ALL_DEVELOPMENT_REFIT_COMPLETE", "closed rerank Critic refit is incomplete")
        selected_arm = str(refit.get("selected_arm"))
        _require(selected_arm in {"C2", "C3"}, "closed rerank Critic arm differs")
        checkpoints = {int(row["seed"]): Path(row["checkpoint_path"]) for row in refit["checkpoints"]}
        _require(tuple(sorted(checkpoints)) == CRITIC_SEEDS_V3, "closed rerank Critic seeds differ")
        projection = load_projection_rows([Path(config["validation_projection_path"])], allowed_splits=("VALIDATION",))
        representatives = _representatives_v3(sources, projection)
        critic = FrozenCriticEnsembleRewardV3(
            checkpoint_paths=checkpoints,
            selected_arm=selected_arm,
            model_path=Path(config["mrnabert_model_path"]),
            device=device,
            kappa=float(config["kappa"]),
            microbatch_size=int(config["critic_online_microbatch_size"]),
        )
    else:
        adapted = adapt_strongest_baseline_v3(
            _json(Path(config["strongest_generation_baseline_path"])),
            _json(Path(config["baseline_selection_input_path"])),
            base_flow_training_seed=int(config["base_flow_training_seed"]),
        )
        strongest_scorer = TorchCheckpointScorer(Path(adapted["guiding_checkpoint_path"]), str(device))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    score_path = output_dir / "frozen_method_scores.private.jsonl"
    started = time.time()
    score_count = 0
    forward_calls_by_member = [0, 0, 0]
    strongest_forward_calls = 0
    with score_path.open("w", encoding="utf-8") as output:
        for source_key in sorted(source_by_key):
            source = source_by_key[source_key]
            rows = by_source[source_key]
            if method != "strongest_matched_baseline":
                assert critic is not None and representatives is not None
                root = initial_state(
                    str(source["source_sequence"]),
                    budget=int(source["edit_budget"]),
                    assay_id=str(source["assay_id"]),
                    context_id=str(source["biological_context_id"]),
                )
                states = [_terminal_state_v3(root, str(row["candidate_sequence"])) for row in rows]
                reward = critic.bind_source(source, representatives[source_key])
                scored = score_critic_states_for_method_v3(method, root, states, reward)
                values = scored.values
                for member, count in enumerate(scored.forward_batches_by_member):
                    forward_calls_by_member[member] += int(count)
            else:
                assert strongest_scorer is not None
                bound = strongest_scorer.bind_source(source)
                values = tuple(bound(str(row["candidate_sequence"])) for row in rows)
                strongest_forward_calls += len(values)
            _require(len(values) == len(rows) and all(math.isfinite(float(value)) for value in values), "closed frozen-score output differs")
            for row, value in zip(rows, values):
                output.write(
                    json.dumps(
                        {
                            "source_key": source_key,
                            "candidate_sequence": str(row["candidate_sequence"]),
                            "frozen_method_score": float(value),
                            "method_id": method,
                            "base_flow_training_seed": int(config["base_flow_training_seed"]),
                            "score_used_measured_outcome": False,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                score_count += 1
    _require(score_count == len(measured), "closed frozen-score table does not exactly cover measured candidates")
    summary = {
        "schema_version": "route_a_v3_route2_xeditflow_closed_frozen_scores.v3",
        "status": "XEDITFLOW_V3_CLOSED_FROZEN_SCORES_COMPLETE",
        "method_id": method,
        "base_flow_training_seed": int(config["base_flow_training_seed"]),
        "source_count": len(sources),
        "measured_candidate_count": score_count,
        "score_path": str(score_path),
        "score_provider": (
            "SOURCE_ANCHORED_FIRST_ORDER_ADDITIVE_POTENTIAL"
            if method == "first_order_guidance"
            else "XEDITCRITIC_V3_ENSEMBLE_TERMINAL_REWARD"
            if method in {"simple_rate_guidance", "generate_then_rerank"}
            else "FROZEN_GENETIC_GUIDING_CHECKPOINT"
        ),
        "critic_forward_calls_by_member": forward_calls_by_member,
        "strongest_guiding_forward_calls": strongest_forward_calls,
        "frozen_baseline_reselected": False,
        "measured_outcome_used_for_score": False,
        "wall_time_seconds": time.time() - started,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "cpu_fallback_used": False,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
        **cuda,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = _json(args.config)
    output_dir = Path(config["output_dir"])
    try:
        result = run(config, output_dir=output_dir)
    except Exception as exc:
        write_gpu_failure_evidence(
            output_dir.with_name(output_dir.name + ".failed.json"),
            config,
            exc,
            entrypoint="score_route2_xeditflow_closed_frozen_methods_v3",
            evaluation_outcomes_accessed=False,
        )
        raise
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
