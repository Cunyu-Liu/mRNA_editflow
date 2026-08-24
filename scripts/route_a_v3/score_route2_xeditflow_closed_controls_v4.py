#!/usr/bin/env python3
"""Score the common closed neighborhood for frozen V4 Critic controls."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_development_projection_v3 import load_projection_rows
from core.route2_gpu_failure_evidence import (
    cuda_device_observation,
    write_gpu_failure_evidence,
)
from core.route2_xeditflow_gate_v4 import authorize_xeditflow_guidance_v4
from core.route2_xeditflow_guidance_v3 import uncertainty_penalized_reward_v3
from core.route2_xeditflow_value_training_v4 import (
    BASE_FLOW_SEEDS_V4,
    CRITIC_SEEDS_V4,
)
from scripts.route_a_v3.route2_mrnabert_bottom_six_encoder_v4 import (
    FrozenMRNABERTBottomSixEncoderV4,
)
from scripts.route_a_v3.run_route2_base_flow_g0_validation_v1 import load_sources
from scripts.route_a_v3.run_route2_xeditflow_matched_controls_v4 import (
    _require_selected_guidance_v4,
)
from scripts.route_a_v3.score_route2_xeditflow_candidates_v4 import (
    _representatives_v4,
    candidate_projection_rows_v4,
)
from scripts.route_a_v3.score_route2_xeditflow_value_rollouts_v4 import (
    _ephemeral_cache_view_v4,
    _load_refit_models_v4,
    _score_member_batch_v4,
)


CLOSED_CRITIC_METHODS_V4 = {
    "first_order_guidance",
    "simple_rate_guidance",
    "generate_then_rerank",
}


class XEditFlowClosedControlScorerV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowClosedControlScorerV4Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _require(
        bool(rows) and all(isinstance(row, dict) for row in rows),
        f"JSONL input is empty or invalid: {path}",
    )
    return rows


def validate_closed_control_score_config_v4(config: Mapping[str, Any]) -> None:
    _require(
        config.get("schema_version")
        == "route_a_v3_route2_xeditflow_closed_control_score_config.v4",
        "unexpected V4 closed-control score config",
    )
    _require(
        str(config.get("method_id")) in CLOSED_CRITIC_METHODS_V4,
        "unknown V4 closed Critic control",
    )
    _require(
        int(config.get("base_flow_training_seed", -1)) in BASE_FLOW_SEEDS_V4
        and float(config.get("kappa", -1)) in {0.0, 0.5, 1.0}
        and float(config.get("temperature", -1)) in {0.5, 1.0}
        and float(config.get("beta_max", -1)) in {0.5, 1.0, 2.0},
        "V4 closed-control seed or guidance identity differs",
    )
    _require(
        tuple(int(seed) for seed in config.get("critic_seeds", ()))
        == CRITIC_SEEDS_V4,
        "V4 closed-control critic ensemble differs",
    )
    runtimes = config.get("critic_refit_runtime_config_paths")
    _require(
        isinstance(runtimes, Mapping)
        and set(runtimes) == {str(seed) for seed in CRITIC_SEEDS_V4},
        "V4 closed-control critic runtime inventory differs",
    )
    _require(
        config.get("pool_assignment") == "DEVELOPMENT"
        and config.get("split") == "VALIDATION"
        and int(config.get("expected_source_count", -1)) == 891
        and config.get("study_policy") == "UNKNOWN_STUDY_SCALE_FIXED_1",
        "V4 closed-control cohort or study policy differs",
    )
    gpu = int(config.get("physical_gpu_index", -1))
    _require(
        gpu in range(6) and str(config.get("device")) == f"cuda:{gpu}",
        "V4 closed-control GPU provenance differs",
    )
    _require(
        config.get("independent_evaluator_used") is False
        and config.get("measured_outcome_used_to_construct_score") is False
        and config.get("development_test_outcomes_accessed_after_atomic_test")
        is False
        and config.get("new_final_evaluation_outcomes_accessed") is False,
        "V4 closed-control protected-input policy differs",
    )
    for field in (
        "critic_readiness_path",
        "setflow_confirmation_path",
        "critic_refit_manifest_path",
        "mrnabert_model_path",
        "source_eligibility_manifest",
        "validation_projection_path",
        "measured_neighborhood_path",
        "guidance_screen_gate_path",
        "output_dir",
    ):
        _require(
            str(config.get(field, "")).startswith(
                "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"
            ),
            f"V4 closed-control {field} left Route 2 /mnt",
        )


def first_order_candidate_scores_v4(
    source_sequence: str,
    candidate_sequences: Sequence[str],
    *,
    single_state_rewards: Mapping[str, float],
) -> list[float]:
    """Reproduce the source-anchored additive potential used during sampling."""

    _require(
        source_sequence in single_state_rewards
        and all(
            len(candidate) == len(source_sequence)
            for candidate in candidate_sequences
        ),
        "V4 first-order closed score geometry differs",
    )
    source_reward = float(single_state_rewards[source_sequence])
    _require(math.isfinite(source_reward), "V4 first-order source reward is nonfinite")
    result = []
    for candidate in candidate_sequences:
        score = 0.0
        for position, (source_base, candidate_base) in enumerate(
            zip(source_sequence, candidate, strict=True)
        ):
            if source_base == candidate_base:
                continue
            single = (
                source_sequence[:position]
                + candidate_base
                + source_sequence[position + 1 :]
            )
            _require(
                single in single_state_rewards,
                "V4 first-order closed score lacks a single-edit coefficient",
            )
            reward = float(single_state_rewards[single])
            _require(
                math.isfinite(reward),
                "V4 first-order single-edit reward is nonfinite",
            )
            score += reward - source_reward
        _require(math.isfinite(score), "V4 first-order candidate score is nonfinite")
        result.append(score)
    return result


def _candidate_bundles_v4(
    sequences: Sequence[str], *, source_key: str
) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "route_a_v3_route2_xeditflow_generated_candidate.v4",
            "source_key": source_key,
            "generation_rank": rank,
            "candidate_sequence": sequence,
            "trajectory_replay_ok": True,
            "generated_candidate_grants_canonical_credit": False,
        }
        for rank, sequence in enumerate(sequences, start=1)
    ]


def _score_sequences_v4(
    sequences: Sequence[str],
    *,
    source: Mapping[str, Any],
    representative: Mapping[str, Any],
    models: Mapping[int, torch.nn.Module],
    checkpoints: Mapping[int, Mapping[str, Any]],
    bottom_encoder: FrozenMRNABERTBottomSixEncoderV4,
    device: torch.device,
    kappa: float,
) -> tuple[list[float], tuple[int, int, int]]:
    _require(bool(sequences), "V4 closed-control sequence batch is empty")
    rewards: list[float] = []
    calls = [0, 0, 0]
    for start in range(0, len(sequences), 32):
        chunk = sequences[start : start + 32]
        rows = candidate_projection_rows_v4(
            _candidate_bundles_v4(
                chunk, source_key=str(source["source_key"])
            ),
            source=source,
            representative=representative,
        )
        cache_view = _ephemeral_cache_view_v4(rows, encoder=bottom_encoder)
        predictions = []
        for member_index, seed in enumerate(CRITIC_SEEDS_V4):
            values, member_calls = _score_member_batch_v4(
                rows,
                model=models[seed],
                checkpoint=checkpoints[seed],
                cache_view=cache_view,
                device=device,
            )
            predictions.append(values)
            calls[member_index] += int(member_calls)
        matrix = torch.tensor(predictions, dtype=torch.float32).T
        _require(
            matrix.shape == (len(chunk), 3)
            and bool(torch.isfinite(matrix).all().item()),
            "V4 closed-control critic matrix differs",
        )
        chunk_rewards = uncertainty_penalized_reward_v3(
            matrix, kappa=float(kappa)
        )
        rewards.extend(float(value) for value in chunk_rewards.tolist())
    return rewards, tuple(calls)


def run(config: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
    validate_closed_control_score_config_v4(config)
    _require(output_dir == Path(str(config["output_dir"])), "V4 output path differs")
    _require(not output_dir.exists(), f"terminal V4 closed-control output exists: {output_dir}")
    critic_readiness = _json(Path(config["critic_readiness_path"]))
    setflow_confirmation = _json(Path(config["setflow_confirmation_path"]))
    _require(
        authorize_xeditflow_guidance_v4(
            critic_readiness, setflow_confirmation
        )["guidance_authorized"]
        is True,
        "V4 closed controls remain blocked before joint readiness",
    )
    _require_selected_guidance_v4(
        _json(Path(config["guidance_screen_gate_path"])), config
    )
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    _require(
        not os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES remapping is forbidden",
    )
    gpu = int(config["physical_gpu_index"])
    device = torch.device(str(config["device"]))
    torch.cuda.set_device(device)
    _require(torch.cuda.is_bf16_supported(), "BF16 is unavailable for V4 closed scores")
    cuda = cuda_device_observation(gpu, require_physical_index_match=True)
    models, checkpoints, _runtimes = _load_refit_models_v4(config, device=device)
    bottom_encoder = FrozenMRNABERTBottomSixEncoderV4(
        Path(config["mrnabert_model_path"]),
        device,
        maximum_sequences_per_batch=int(
            config["bottom_six_maximum_sequences_per_batch"]
        ),
        batch_token_budget=int(config["bottom_six_batch_token_budget"]),
        attention_backend=str(config["attention_backend"]),
    )
    sources = load_sources(Path(config["source_eligibility_manifest"]))
    _require(len(sources) == 891, "V4 closed-control source cohort changed")
    validation = load_projection_rows(
        [Path(config["validation_projection_path"])],
        allowed_splits=("VALIDATION",),
    )
    representatives = _representatives_v4(sources, validation)
    measured = _jsonl(Path(config["measured_neighborhood_path"]))
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in measured:
        _require(
            row.get("pool_assignment") == "DEVELOPMENT"
            and row.get("split") == "VALIDATION",
            "V4 closed-control score entered a non-Validation row",
        )
        by_source[str(row["source_key"])].append(row)
    source_by_key = {str(source["source_key"]): source for source in sources}
    _require(
        set(by_source) == set(source_by_key) == set(representatives),
        "V4 closed-control measured source coverage differs",
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    (output_dir / "run_config.json").write_text(
        json.dumps(dict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    score_path = output_dir / "frozen_method_scores.private.jsonl"
    compute_path = output_dir / "critic_compute_by_source.jsonl"
    score_path.write_text("", encoding="utf-8")
    compute_path.write_text("", encoding="utf-8")
    method = str(config["method_id"])
    total_calls = [0, 0, 0]
    scored_count = 0
    run_peak_vram_mb = 0.0
    started = time.time()
    for source in sources:
        source_key = str(source["source_key"])
        source_sequence = str(source["source_sequence"])
        candidate_sequences = [
            str(row["candidate_sequence"]) for row in by_source[source_key]
        ]
        _require(
            len(set(candidate_sequences)) == len(candidate_sequences),
            "V4 closed-control measured candidate is duplicated",
        )
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        source_started = time.perf_counter()
        if method == "first_order_guidance":
            single_sequences = {source_sequence}
            for candidate in candidate_sequences:
                _require(
                    len(candidate) == len(source_sequence),
                    "V4 closed-control candidate length differs",
                )
                for position, (left, right) in enumerate(
                    zip(source_sequence, candidate, strict=True)
                ):
                    if left != right:
                        single_sequences.add(
                            source_sequence[:position]
                            + right
                            + source_sequence[position + 1 :]
                        )
            ordered = [source_sequence] + sorted(single_sequences - {source_sequence})
            rewards, member_calls = _score_sequences_v4(
                ordered,
                source=source,
                representative=representatives[source_key],
                models=models,
                checkpoints=checkpoints,
                bottom_encoder=bottom_encoder,
                device=device,
                kappa=float(config["kappa"]),
            )
            scores = first_order_candidate_scores_v4(
                source_sequence,
                candidate_sequences,
                single_state_rewards=dict(zip(ordered, rewards, strict=True)),
            )
            score_kind = "SOURCE_ANCHORED_FIRST_ORDER_CRITIC_POTENTIAL"
        else:
            scores, member_calls = _score_sequences_v4(
                candidate_sequences,
                source=source,
                representative=representatives[source_key],
                models=models,
                checkpoints=checkpoints,
                bottom_encoder=bottom_encoder,
                device=device,
                kappa=float(config["kappa"]),
            )
            score_kind = "EXACT_TERMINAL_FROZEN_CRITIC_REWARD"
        torch.cuda.synchronize(device)
        source_wall = time.perf_counter() - source_started
        source_peak = torch.cuda.max_memory_allocated(device) / 1024**2
        run_peak_vram_mb = max(run_peak_vram_mb, source_peak)
        for index, count in enumerate(member_calls):
            total_calls[index] += int(count)
        with compute_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "source_key": source_key,
                        "critic_forwards_by_member": list(member_calls),
                        "candidate_count": len(candidate_sequences),
                        "wall_time_seconds": source_wall,
                        "peak_vram_mb": source_peak,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        with score_path.open("a", encoding="utf-8") as handle:
            for candidate, score in zip(candidate_sequences, scores, strict=True):
                handle.write(
                    json.dumps(
                        {
                            "schema_version": "route_a_v3_route2_xeditflow_closed_control_score.v4",
                            "method_id": method,
                            "base_flow_training_seed": int(
                                config["base_flow_training_seed"]
                            ),
                            "source_key": source_key,
                            "candidate_sequence": candidate,
                            "frozen_method_score": float(score),
                            "score_kind": score_kind,
                            "kappa": float(config["kappa"]),
                            "temperature": float(config["temperature"]),
                            "beta_max": float(config["beta_max"]),
                            "study_neutral": True,
                            "measured_outcome_used_to_construct_score": False,
                            "independent_evaluator_used": False,
                            "development_test_outcomes_accessed_after_atomic_test": False,
                            "new_final_evaluation_outcomes_accessed": False,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                scored_count += 1
    result = {
        "schema_version": "route_a_v3_route2_xeditflow_closed_control_scores.v4",
        "status": "XEDITFLOW_V4_CLOSED_CONTROL_SCORES_COMPLETE",
        "method_id": method,
        "base_flow_training_seed": int(config["base_flow_training_seed"]),
        "kappa": float(config["kappa"]),
        "temperature": float(config["temperature"]),
        "beta_max": float(config["beta_max"]),
        "source_count": len(sources),
        "measured_candidate_count": scored_count,
        "critic_seeds": list(CRITIC_SEEDS_V4),
        "critic_forward_counts_by_member": total_calls,
        "wall_time_seconds": time.time() - started,
        "peak_vram_mb": run_peak_vram_mb,
        "study_policy": "UNKNOWN_STUDY_SCALE_FIXED_1",
        "measured_outcome_used_to_construct_score": False,
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
        "cpu_fallback_used": False,
        **cuda,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    config = _json(arguments.config)
    try:
        result = run(config, output_dir=arguments.output_dir)
    except Exception as exc:
        write_gpu_failure_evidence(
            arguments.output_dir.with_name(arguments.output_dir.name + ".failed.json"),
            config,
            exc,
            entrypoint="score_route2_xeditflow_closed_controls_v4",
            evaluation_outcomes_accessed=False,
        )
        raise
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
