#!/usr/bin/env python3
"""Run frozen-critic guided SUB+STOP XEditFlow after readiness adjudication."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_gpu_failure_evidence import (
    cuda_device_observation,
    write_gpu_failure_evidence,
)
from core.route2_legal_xeditflow import FlowState, apply_action, initial_state
from scripts.route_a_v3.route2_mrnabert_guided_critic_v1 import (
    FrozenRoute2MRNABERTCritic,
)
from scripts.route_a_v3.run_route2_base_flow_g0_validation_v1 import (
    REGION,
    learned_rate_function,
    load_model,
    load_sources,
    sample_one,
    validate_checkpoint_training_provenance,
)


GUIDED_CONFIG_SCHEMA = (
    "route_a_v3_route2_mrnabert_critic_v2_guided_xeditflow_development.v1"
)
GUIDED_METHOD_ID = "frozen_mrnabert_critic_v2_guided_xeditflow_v1"
READINESS_INPUT_SCHEMA = (
    "route_a_v3_route2_mrnabert_critic_v2_guidance_readiness_input.v1"
)
READINESS_ADJUDICATION_SCHEMA = (
    "route_a_v3_route2_mrnabert_critic_v2_guidance_readiness_adjudication.v1"
)
ROUTE2_ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
EXPECTED_READINESS_INPUT = (
    ROUTE2_ROOT / "comparisons/mrnabert_critic_v2_guidance_readiness_input_v1.json"
)
EXPECTED_READINESS_ADJUDICATION = (
    ROUTE2_ROOT
    / "comparisons/mrnabert_critic_v2_guidance_readiness_adjudication_v1.json"
)
EXPECTED_CRITIC_CHECKPOINT = (
    ROUTE2_ROOT
    / "runs/mrnabert_critic_v2/all_development_refit_v1/seed20260823/delta_predictor_checkpoint.pt"
)


class GuidedRunError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GuidedRunError(message)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{label} root is not an object")
    return value


def validate_guided_config(config: Mapping[str, Any]) -> None:
    _require(
        config.get("schema_version") == GUIDED_CONFIG_SCHEMA
        and config.get("status") == "WAITING_FOR_CRITIC_V2_AND_FLOW_READINESS",
        "historical or unexpected guided config is not authorized",
    )
    _require(
        int(config.get("seed", -1)) == 20260825
        and Path(str(config.get("readiness_input_path")))
        == EXPECTED_READINESS_INPUT
        and Path(str(config.get("readiness_adjudication_path")))
        == EXPECTED_READINESS_ADJUDICATION
        and Path(str(config.get("critic_checkpoint_path")))
        == EXPECTED_CRITIC_CHECKPOINT,
        "Critic V2 guided artifact binding differs",
    )
    _require(
        config.get("matched_search_budget_rule")
        == "GUIDED_TOTAL_FORWARD_EQUIVALENTS_AS_SEARCH_CRITIC_CAP_PER_SOURCE"
        and config.get("evaluation_outcomes_accessed") is False
        and config.get("generated_candidates_grant_canonical_credit") is False
        and config.get("scientific_claim_status") == "NOT_ESTABLISHED",
        "guided protected-outcome or compute policy differs",
    )


def validate_readiness(
    readiness_input: Mapping[str, Any],
    adjudication: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    _require(
        readiness_input.get("schema_version") == READINESS_INPUT_SCHEMA
        and readiness_input.get("guided_generation_executed") is False
        and readiness_input.get("evaluation_opened_by_readiness_builder") is False,
        "readiness input schema differs",
    )
    _require(
        adjudication.get("schema_version")
        == READINESS_ADJUDICATION_SCHEMA
        and adjudication.get("guided_unlocked") is True
        and adjudication.get("critic_status") == "CRITIC_READY_FOR_GUIDANCE"
        and adjudication.get("flow_status") == "FLOW_G0_READY"
        and adjudication.get("guided_generation_status")
        == "GUIDED_XEDITFLOW_DEVELOPMENT_ALLOWED"
        and adjudication.get("guided_generation_executed") is False
        and adjudication.get("evaluation_opened") is False
        and adjudication.get("biological_optimization_established") is False,
        "critic and Flow are not ready for guided generation",
    )
    critic = readiness_input["critic"]
    flow = readiness_input["flow"]
    online_encoder = critic["online_encoder_validation"]
    _require(
        online_encoder.get("status")
        == "ONLINE_FROZEN_MRNABERT_MATCHES_CANONICAL_CACHE"
        and online_encoder.get("novel_candidate_encoding_supported") is True
        and online_encoder.get("evaluation_records_read") == 0,
        "online generated-candidate encoding is not ready",
    )
    _require(
        critic["reward_policy"].get(
            "evaluation_records_used_for_training_hpo_threshold_or_reward"
        )
        == 0,
        "Evaluation entered critic selection or reward",
    )
    _require(
        Path(critic["refit_checkpoint"])
        == Path(config["critic_checkpoint_path"]),
        "guided critic path differs from readiness evidence",
    )
    _require(
        Path(flow["checkpoint"])
        == Path(config["base_flow_checkpoint_path"]),
        "guided base-flow path differs from readiness evidence",
    )


def selected_attention_backend(adjudication: Mapping[str, Any]) -> str:
    _require(
        adjudication.get("schema_version")
        == "route_a_v3_route2_mrnabert_sdpa_backend_adjudication.v1"
        and adjudication.get("status") == "ONLINE_ENCODER_BACKEND_ADJUDICATED"
        and adjudication.get("evaluation_opened") is False,
        "online encoder backend adjudication is invalid",
    )
    backend = str(adjudication.get("selected_attention_backend"))
    _require(
        backend in {"OFFICIAL_PYTORCH_FALLBACK", "PYTORCH_SDPA_AUTO"},
        "online encoder backend selection is unknown",
    )
    return backend


def batched_guided_rate_function(
    base_rate_function,
    critic: FrozenRoute2MRNABERTCritic,
    *,
    endpoint_id: str,
    region: str,
    guidance_strength: float,
    counters: dict[str, int] | None = None,
):
    strength = float(guidance_strength)
    _require(math.isfinite(strength) and strength >= 0.0, "guidance strength is invalid")
    rate_cache: dict[FlowState, dict[Any, float]] = {}

    def score(state: FlowState, actions):
        if counters is not None:
            counters["guided_rate_requests"] = counters.get("guided_rate_requests", 0) + 1
        cached = rate_cache.get(state)
        if cached is not None:
            _require(set(cached) == set(actions), "cached rates do not cover the legal action set")
            if counters is not None:
                counters["guided_rate_cache_hits"] = counters.get("guided_rate_cache_hits", 0) + 1
            return cached
        base = base_rate_function(state, actions)
        _require(set(base) == set(actions), "base rates do not cover the legal action set")
        children = [apply_action(state, action) for action in actions]
        values = critic.potentials(
            [state, *children], endpoint_id=endpoint_id, region=region
        )
        current = values[0]
        if counters is not None:
            counters["base_flow_forwards"] = counters.get("base_flow_forwards", 0) + 1
        result = {}
        for action, child_value in zip(actions, values[1:]):
            rate = float(base[action])
            _require(math.isfinite(rate) and rate >= 0.0, "base rate is invalid")
            result[action] = rate * math.exp(strength * (child_value - current))
        rate_cache[state] = result
        if counters is not None:
            counters["unique_state_rate_evaluations"] = counters.get(
                "unique_state_rate_evaluations", 0
            ) + 1
        return result

    return score


def summarize_compute_rows(rows: list[Mapping[str, Any]]) -> dict[str, float | int]:
    _require(bool(rows), "guided per-source compute is empty")
    matched = [int(row["matched_search_critic_forward_budget"]) for row in rows]
    critic = [
        int(row["critic_candidate_forward_equivalent_count"]) for row in rows
    ]
    generator = [int(row["generator_nfe"]) for row in rows]
    _require(
        all(value > 0 for value in matched + critic)
        and all(value >= 0 for value in generator)
        and all(total == critic_value + generator_value for total, critic_value, generator_value in zip(matched, critic, generator)),
        "guided per-source forward accounting does not close",
    )
    return {
        "critic_candidate_forward_equivalent_count": sum(critic),
        "total_forward_equivalent_count": sum(matched),
        "matched_search_budget_minimum": min(matched),
        "matched_search_budget_maximum": max(matched),
        "matched_search_budget_mean": statistics.fmean(matched),
        "matched_search_budget_median": statistics.median(matched),
    }


def execute(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_guided_config(config)
    output_dir = Path(config["output_directory"])
    _require(not output_dir.exists(), f"guided output already exists: {output_dir}")
    readiness_input = _read_json(
        Path(config["readiness_input_path"]), "readiness input"
    )
    readiness_adjudication = _read_json(
        Path(config["readiness_adjudication_path"]), "readiness adjudication"
    )
    backend_adjudication = _read_json(
        Path(config["encoder_attention_backend_adjudication_path"]),
        "encoder attention backend adjudication",
    )
    attention_backend = selected_attention_backend(backend_adjudication)
    validate_readiness(readiness_input, readiness_adjudication, config)
    policy = _read_json(Path(config["reward_policy_path"]), "reward policy")
    _require(
        policy.get("status") == "PROSPECTIVELY_FROZEN_BEFORE_GUIDED_GENERATION"
        and policy.get("action_space") == "SUB_PLUS_STOP"
        and policy.get("uncertainty_in_guidance") == "DISABLED_DIAGNOSTIC_ONLY"
        and policy.get("evaluation_records_used_for_training_hpo_threshold_or_reward") == 0,
        "reward policy is not the frozen mean-only SUB+STOP policy",
    )
    _require(
        config.get("evaluation_outcomes_accessed") is False,
        "Evaluation is not available to guided Development",
    )
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA remapping is forbidden")
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    physical_gpu_index = int(config["physical_gpu_index"])
    device = torch.device(str(config["device"]))
    _require(device.type == "cuda" and device.index == physical_gpu_index, "CUDA index differs")
    _require(0 <= physical_gpu_index < torch.cuda.device_count(), "physical GPU is unavailable")
    torch.cuda.set_device(device)
    cuda_provenance = cuda_device_observation(
        physical_gpu_index, require_physical_index_match=True
    )
    sources = load_sources(Path(config["source_eligibility_manifest"]))
    base_model, base_checkpoint = load_model(
        Path(config["base_flow_checkpoint_path"]), device
    )
    validate_checkpoint_training_provenance(
        base_checkpoint.get("training_provenance", {})
    )
    transform = policy["potential_transform"]
    critic = FrozenRoute2MRNABERTCritic(
        Path(config["critic_checkpoint_path"]),
        Path(config["mrnabert_model_path"]),
        device,
        potential_minimum=float(transform["minimum"]),
        potential_maximum=float(transform["maximum"]),
        encoder_attention_backend=attention_backend,
    )
    rows = []
    terminal_causes = Counter()
    budget_violations = 0
    replay_probe_failures = 0
    generator_nfe = 0
    replay_probe_nfe = 0
    counters: dict[str, int] = {}
    per_source_compute = []
    started = time.time()
    for source_index, source_row in enumerate(sources):
        source_candidate_row_start = len(rows)
        critic.clear_source_caches()
        source_model_batches_start = critic.model_batch_forward_count
        source_candidate_equivalents_start = (
            critic.candidate_forward_equivalent_count
        )
        source_replay_model_batches = 0
        source_replay_candidate_equivalents = 0
        replay_actual_generator_forwards = 0
        replay_forwards = 0
        source_generator_nfe = 0
        source_trajectory_decision_count = 0
        source_rate_requests_start = counters.get("guided_rate_requests", 0)
        source_rate_cache_hits_start = counters.get("guided_rate_cache_hits", 0)
        source_unique_rate_evaluations_start = counters.get(
            "unique_state_rate_evaluations", 0
        )
        source = source_row["source_sequence"]
        _require(
            int(source_row["candidate_budget"]) > 0,
            "guided candidate budget must be positive",
        )
        region = str(source_row["region"]).replace("′", "").replace("'", "")
        _require(region in REGION, "source region is unsupported")
        root = initial_state(
            source,
            budget=source_row["edit_budget"],
            assay_id=str(source_row["assay_id"]),
            context_id=str(source_row["biological_context_id"]),
        )
        base_rate = learned_rate_function(
            base_model,
            region_id=REGION[region],
            assay_id=base_checkpoint["assay_vocab"].get(
                str(source_row["assay_id"]), 0
            ),
            context_id=base_checkpoint["context_vocab"].get(
                str(source_row["biological_context_id"]), 0
            ),
            device=device,
        )
        guided_rate = batched_guided_rate_function(
            base_rate,
            critic,
            endpoint_id=str(source_row["endpoint_id"]),
            region=region,
            guidance_strength=float(policy["guidance_strength"]),
            counters=counters,
        )
        source_critic_score = critic.potential(
            root,
            endpoint_id=str(source_row["endpoint_id"]),
            region=region,
        )
        for candidate_index in range(source_row["candidate_budget"]):
            trajectory_seed = int(config["seed"]) + source_index * 1_000_003 + candidate_index
            generator_forwards_start = counters.get("base_flow_forwards", 0)
            terminal, action_ids, forwards = sample_one(
                root, guided_rate, seed=trajectory_seed, device=device
            )
            actual_generator_forwards = (
                counters.get("base_flow_forwards", 0) - generator_forwards_start
            )
            if candidate_index == 0:
                replay_model_batches_start = critic.model_batch_forward_count
                replay_candidate_equivalents_start = (
                    critic.candidate_forward_equivalent_count
                )
                replay_generator_forwards_start = counters.get(
                    "base_flow_forwards", 0
                )
                replay, replay_actions, replay_forwards = sample_one(
                    root, guided_rate, seed=trajectory_seed, device=device
                )
                replay_actual_generator_forwards = (
                    counters.get("base_flow_forwards", 0)
                    - replay_generator_forwards_start
                )
                source_replay_model_batches += (
                    critic.model_batch_forward_count - replay_model_batches_start
                )
                source_replay_candidate_equivalents += (
                    critic.candidate_forward_equivalent_count
                    - replay_candidate_equivalents_start
                )
                replay_probe_failures += int(
                    replay != terminal or replay_actions != action_ids
                )
                replay_probe_nfe += replay_actual_generator_forwards
            generator_nfe += actual_generator_forwards
            source_generator_nfe += actual_generator_forwards
            source_trajectory_decision_count += forwards
            budget_violations += int(terminal.edit_count > source_row["edit_budget"])
            terminal_causes[terminal.terminal_cause] += 1
            terminal_critic_score = critic.potential(
                terminal,
                endpoint_id=str(source_row["endpoint_id"]),
                region=region,
            )
            rows.append({
                "method_id": GUIDED_METHOD_ID,
                "source_key": source_row["source_key"],
                "candidate_sequence": terminal.current_sequence,
                "terminal_cause": terminal.terminal_cause,
                "edit_count": terminal.edit_count,
                "trajectory_actions": list(action_ids),
                "trajectory_seed": trajectory_seed,
                "generator_nfe": actual_generator_forwards,
                "trajectory_decision_count": forwards,
                "critic_score": terminal_critic_score,
                "source_critic_score": source_critic_score,
                "generated_candidate_grants_canonical_credit": False,
            })
        source_model_batches = (
            critic.model_batch_forward_count
            - source_model_batches_start
            - source_replay_model_batches
        )
        source_candidate_equivalents = (
            critic.candidate_forward_equivalent_count
            - source_candidate_equivalents_start
            - source_replay_candidate_equivalents
        )
        source_rate_requests = (
            counters.get("guided_rate_requests", 0) - source_rate_requests_start
        )
        source_rate_cache_hits = (
            counters.get("guided_rate_cache_hits", 0) - source_rate_cache_hits_start
        )
        source_unique_rate_evaluations = (
            counters.get("unique_state_rate_evaluations", 0)
            - source_unique_rate_evaluations_start
        )
        _require(
            source_rate_requests
            == source_rate_cache_hits + source_unique_rate_evaluations,
            "guided state-rate cache accounting does not close",
        )
        _require(
            source_model_batches > 0 and source_candidate_equivalents > 0,
            "guided source cohort made no critic forward calls",
        )
        matched_search_critic_budget = (
            source_candidate_equivalents + source_generator_nfe
        )
        for row_index, row in enumerate(rows[source_candidate_row_start:]):
            row["critic_forwards"] = (
                source_candidate_equivalents if row_index == 0 else 0
            )
            row["critic_forward_budget"] = matched_search_critic_budget
            row["critic_forward_budget_rule"] = (
                "GUIDED_TOTAL_FORWARD_EQUIVALENTS_AS_SEARCH_CRITIC_CAP_PER_SOURCE"
            )
            row["independent_evaluator_forwards"] = 0
        per_source_compute.append({
            "source_key": source_row["source_key"],
            "edit_budget": int(source_row["edit_budget"]),
            "candidate_budget": int(source_row["candidate_budget"]),
            "generator_nfe": source_generator_nfe,
            "trajectory_decision_count": source_trajectory_decision_count,
            "guided_rate_request_count_including_replay_probes": source_rate_requests,
            "guided_rate_cache_hit_count_including_replay_probes": source_rate_cache_hits,
            "unique_state_rate_evaluation_count_including_replay_probes": (
                source_unique_rate_evaluations
            ),
            "guided_rate_cache_hit_rate_including_replay_probes": (
                source_rate_cache_hits / source_rate_requests
                if source_rate_requests
                else 0.0
            ),
            "critic_model_batch_forward_count": source_model_batches,
            "critic_candidate_forward_equivalent_count": source_candidate_equivalents,
            "total_forward_equivalent_count": (
                source_candidate_equivalents + source_generator_nfe
            ),
            "matched_search_critic_forward_budget": matched_search_critic_budget,
            "replay_probe_generator_nfe": replay_actual_generator_forwards,
            "replay_probe_trajectory_decision_count": replay_forwards,
            "replay_probe_critic_model_batch_forward_count": source_replay_model_batches,
            "replay_probe_critic_candidate_forward_equivalent_count": (
                source_replay_candidate_equivalents
            ),
        })
    _require(replay_probe_failures == 0, "guided fixed-seed replay failed")
    _require(budget_violations == 0, "guided trajectory exceeded edit budget")
    unique_candidate_count = len({
        (row["source_key"], row["candidate_sequence"]) for row in rows
    })
    forward_summary = summarize_compute_rows(per_source_compute)
    guided_model_batches = sum(
        row["critic_model_batch_forward_count"] for row in per_source_compute
    )
    summary = {
        "schema_version": GUIDED_CONFIG_SCHEMA,
        "status": "GUIDED_XEDITFLOW_DEVELOPMENT_COMPLETE",
        "trajectory_count": len(rows),
        "source_budget_cohort_count": len(sources),
        "hard_legality_rate": 1.0,
        "edit_budget_violation_count": budget_violations,
        "replay_probe_trajectory_count": len(sources),
        "replay_probe_failure_count": replay_probe_failures,
        "terminal_causes": dict(sorted(terminal_causes.items())),
        "unique_candidate_count": unique_candidate_count,
        "unique_candidate_rate": unique_candidate_count / len(rows),
        "generator_nfe": generator_nfe,
        "trajectory_decision_count": sum(
            int(row["trajectory_decision_count"]) for row in per_source_compute
        ),
        "replay_probe_generator_nfe": replay_probe_nfe,
        "base_flow_forward_count_including_replay_probes": counters.get(
            "base_flow_forwards", 0
        ),
        "guided_rate_request_count_including_replay_probes": counters.get(
            "guided_rate_requests", 0
        ),
        "guided_rate_cache_hit_count_including_replay_probes": counters.get(
            "guided_rate_cache_hits", 0
        ),
        "unique_state_rate_evaluation_count_including_replay_probes": counters.get(
            "unique_state_rate_evaluations", 0
        ),
        "guided_rate_cache_hit_rate_including_replay_probes": (
            counters.get("guided_rate_cache_hits", 0)
            / counters.get("guided_rate_requests", 1)
            if counters.get("guided_rate_requests", 0)
            else 0.0
        ),
        "critic_model_batch_forward_count": guided_model_batches,
        "critic_candidate_forward_equivalent_count": forward_summary[
            "critic_candidate_forward_equivalent_count"
        ],
        "total_forward_equivalent_count": forward_summary[
            "total_forward_equivalent_count"
        ],
        "critic_model_batch_forward_count_including_replay_probes": (
            critic.model_batch_forward_count
        ),
        "critic_candidate_forward_equivalent_count_including_replay_probes": (
            critic.candidate_forward_equivalent_count
        ),
        "matched_search_budget_rule": (
            "GUIDED_TOTAL_FORWARD_EQUIVALENTS_AS_SEARCH_CRITIC_CAP_PER_SOURCE"
        ),
        "matched_search_budget_minimum": forward_summary[
            "matched_search_budget_minimum"
        ],
        "matched_search_budget_maximum": forward_summary[
            "matched_search_budget_maximum"
        ],
        "matched_search_budget_mean": forward_summary[
            "matched_search_budget_mean"
        ],
        "matched_search_budget_median": forward_summary[
            "matched_search_budget_median"
        ],
        "per_source_compute_path": str(
            output_dir / "guided_compute_by_source.jsonl"
        ),
        "critic_parameter_updates": 0,
        "generator_parameter_updates": 0,
        "reward_signal": policy["reward_signal"],
        "encoder_attention_backend": attention_backend,
        "uncertainty_in_guidance": policy["uncertainty_in_guidance"],
        "evaluation_outcomes_read": 0,
        "generated_candidates_grant_canonical_credit": False,
        "biological_optimization_established": False,
        "wall_time_seconds": time.time() - started,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "device": str(device),
        "physical_gpu_index": physical_gpu_index,
        "cpu_fallback_used": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        **cuda_provenance,
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    (output_dir / "guided_config.json").write_text(
        json.dumps(dict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "generated_candidates.private.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output_dir / "guided_compute_by_source.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in per_source_compute
        ),
        encoding="utf-8",
    )
    serialized = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (output_dir / "guided_summary.json").write_text(serialized, encoding="utf-8")
    (output_dir / "final_summary.json").write_text(serialized, encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        result = execute(config)
    except Exception as exc:
        output = Path(config["output_directory"])
        write_gpu_failure_evidence(
            output.with_name(output.name + ".failed.json"),
            config,
            exc,
            entrypoint="run_route2_guided_xeditflow_v1",
            evaluation_outcomes_accessed=False,
        )
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
