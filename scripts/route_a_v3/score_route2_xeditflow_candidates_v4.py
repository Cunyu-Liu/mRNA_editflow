#!/usr/bin/env python3
"""Score V4 SMC candidates and reconcile reserved critic compute."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
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
from core.route2_xeditcritic_training_data_v3 import UNKNOWN_CATEGORY
from core.route2_xeditflow_gate_v4 import authorize_xeditflow_guidance_v4
from core.route2_xeditflow_guidance_v3 import uncertainty_penalized_reward_v3
from core.route2_xeditflow_guidance_v4 import MatchedComputeRecordV4
from core.route2_xeditflow_value_training_v4 import CRITIC_SEEDS_V4
from scripts.route_a_v3.route2_mrnabert_bottom_six_encoder_v4 import (
    FrozenMRNABERTBottomSixEncoderV4,
)
from scripts.route_a_v3.run_route2_base_flow_g0_validation_v1 import load_sources
from scripts.route_a_v3.score_route2_xeditflow_value_rollouts_v4 import (
    _ephemeral_cache_view_v4,
    _load_refit_models_v4,
    _score_member_batch_v4,
)


class XEditFlowCandidateScorerV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowCandidateScorerV4Error(message)


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


def validate_candidate_score_config_v4(config: Mapping[str, Any]) -> None:
    _require(
        config.get("schema_version")
        == "route_a_v3_route2_xeditflow_candidate_critic_score_config.v4",
        "unexpected V4 candidate critic score config",
    )
    _require(
        tuple(int(seed) for seed in config.get("critic_seeds", ()))
        == CRITIC_SEEDS_V4,
        "V4 candidate critic seeds changed",
    )
    _require(
        int(config.get("base_flow_training_seed", -1)) == 20260912,
        "V4 candidate scorer base-flow seed changed",
    )
    _require(
        float(config.get("kappa", -1)) in {0.0, 0.5, 1.0},
        "V4 candidate scorer kappa differs",
    )
    _require(
        float(config.get("temperature", -1)) in {0.5, 1.0}
        and float(config.get("beta_max", -1)) in {0.5, 1.0, 2.0}
        and bool(str(config.get("method_id", ""))),
        "V4 candidate scorer combination identity differs",
    )
    _require(
        int(config.get("expected_source_count", -1)) == 891,
        "V4 candidate scorer source count differs",
    )
    _require(
        int(config.get("candidate_cap_per_source", -1)) == 32,
        "V4 candidate scorer cap differs",
    )
    _require(
        config.get("study_policy") == "UNKNOWN_STUDY_SCALE_FIXED_1"
        and config.get("prediction_scale")
        == "TASK_ROBUST_STANDARDIZED_EFFECT",
        "V4 candidate scorer inference policy differs",
    )
    runtime_paths = config.get("critic_refit_runtime_config_paths")
    _require(
        isinstance(runtime_paths, Mapping)
        and set(runtime_paths) == {str(seed) for seed in CRITIC_SEEDS_V4},
        "V4 candidate scorer runtime inventory differs",
    )
    gpu = int(config.get("physical_gpu_index", -1))
    _require(
        gpu in range(6) and str(config.get("device")) == f"cuda:{gpu}",
        "V4 candidate scorer GPU provenance differs",
    )
    _require(
        config.get("independent_evaluator_used") is False
        and config.get("critic_self_score_used_for_generation_or_selection")
        is False
        and config.get("development_test_outcomes_accessed_after_atomic_test")
        is False
        and config.get("new_final_evaluation_outcomes_accessed") is False,
        "V4 candidate scorer protected-input policy differs",
    )
    for field in (
        "generation_summary_path",
        "candidate_path",
        "generation_compute_path",
        "output_dir",
    ):
        _require(
            str(config.get(field, "")).startswith(
                "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"
            ),
            f"V4 candidate scorer {field} left Route 2 /mnt",
        )


def _representatives_v4(
    sources: Sequence[Mapping[str, Any]],
    projection: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in projection:
        _require(row.get("split") == "VALIDATION", "non-Validation row entered V4 scorer")
        key = (
            str(row["source_sequence"]),
            str(row["endpoint_id"]),
            str(row["biological_context_id"]),
        )
        if key not in by_key or str(row["canonical_record_id"]) < str(
            by_key[key]["canonical_record_id"]
        ):
            by_key[key] = row
    result = {}
    for source in sources:
        key = (
            str(source["source_sequence"]),
            str(source["endpoint_id"]),
            str(source["biological_context_id"]),
        )
        _require(key in by_key, "V4 generated source lacks Validation metadata")
        result[str(source["source_key"])] = by_key[key]
    return result


def candidate_projection_rows_v4(
    candidates: Sequence[Mapping[str, Any]],
    *,
    source: Mapping[str, Any],
    representative: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _require(bool(candidates) and len(candidates) <= 32, "V4 candidate source batch differs")
    source_key = str(source["source_key"])
    source_sequence = str(source["source_sequence"])
    region = str(source["region"]).replace("′", "").replace("'", "")
    _require(region in {"5UTR", "3UTR"}, "V4 candidate source region differs")
    rows = []
    for expected_rank, candidate in enumerate(candidates, start=1):
        _require(
            candidate.get("schema_version")
            == "route_a_v3_route2_xeditflow_generated_candidate.v4"
            and str(candidate.get("source_key")) == source_key
            and int(candidate.get("generation_rank", -1)) == expected_rank
            and candidate.get("trajectory_replay_ok") is True
            and candidate.get("generated_candidate_grants_canonical_credit") is False,
            "V4 candidate identity or replay differs",
        )
        sequence = str(candidate["candidate_sequence"])
        _require(
            len(sequence) == len(source_sequence)
            and set(sequence) <= set("ACGU"),
            "V4 candidate sequence geometry differs",
        )
        edits = [
            {"position": index, "source_base": left, "candidate_base": right}
            for index, (left, right) in enumerate(
                zip(source_sequence, sequence, strict=True)
            )
            if left != right
        ]
        _require(0 <= len(edits) <= 5, "V4 candidate edit budget differs")
        rows.append(
            {
                "canonical_record_id": f"generated-{source_key}-{expected_rank:02d}",
                "split": "VALIDATION",
                "task_id": str(representative["task_id"]),
                "study_unit_id": UNKNOWN_CATEGORY,
                "source_group_id": str(representative["source_group_id"]),
                "assay_id": str(source["assay_id"]),
                "biological_context_id": str(source["biological_context_id"]),
                "region_id": 0 if region == "5UTR" else 1,
                "endpoint_id": "GENERATED_V4_STUDY_NEUTRAL",
                "endpoint_descriptor": dict(representative["endpoint_descriptor"]),
                "source_sequence": source_sequence,
                "candidate_sequence": sequence,
                "source_relative_edits": edits,
                "direction_normalized_delta": 0.0,
                "dummy_target_for_inference_only": True,
                "development_test_outcomes_accessed_after_atomic_test": False,
                "new_final_evaluation_outcomes_accessed": False,
            }
        )
    return rows


def reconcile_candidate_compute_v4(
    compute: Mapping[str, Any],
    *,
    actual_critic_forwards_by_member: Sequence[int],
    scorer_wall_time_seconds: float,
    scorer_peak_vram_mb: float,
) -> dict[str, Any]:
    reserved = tuple(int(value) for value in compute["critic_forwards_by_member"])
    actual = tuple(int(value) for value in actual_critic_forwards_by_member)
    _require(
        len(reserved) == len(actual) == 3
        and all(0 < value <= cap for value, cap in zip(actual, reserved, strict=True)),
        "V4 actual critic compute exceeds or differs from reservation",
    )
    failures = compute["failure_counters"]
    generation_wall = float(
        compute.get("source_equal_wall_time_seconds", compute["wall_time_seconds"])
    )
    generation_peak = max(
        float(compute.get("peak_vram_mb", 0.0)),
        float(compute.get("source_equal_wall_peak_vram_mb", 0.0)),
    )
    record = MatchedComputeRecordV4(
        source_key=str(compute["source_key"]),
        trunk_forwards=int(compute["trunk_forwards"]),
        mode_forwards=int(compute["mode_forwards"]),
        value_forwards=int(compute["value_forwards"]),
        critic_forwards_by_member=list(actual),
        candidate_count=int(compute["candidate_count"]),
        trajectory_count=int(compute["trajectory_count"]),
        wall_time_seconds=generation_wall + float(scorer_wall_time_seconds),
        peak_vram_mb=max(generation_peak, float(scorer_peak_vram_mb)),
        edit_budget_violation_count=int(failures["edit_budget_violation_count"]),
        candidate_budget_violation_count=int(
            failures["candidate_budget_violation_count"]
        ),
        replay_failure_count=int(failures["replay_failure_count"]),
        numerical_failure_count=int(failures["numerical_failure_count"]),
    ).to_dict()
    record.update(
        {
            "terminal_critic_forwards_reserved_by_member": list(reserved),
            "terminal_critic_forwards_actual_by_member": list(actual),
            "terminal_critic_forwards_are_reserved_pending_scoring": False,
            "terminal_critic_reservation_reconciled": True,
            "source_equal_wall_time_seconds": generation_wall
            + float(scorer_wall_time_seconds),
            "source_equal_wall_time_scope": compute.get(
                "source_equal_wall_time_scope"
            ),
            "source_equal_wall_peak_vram_mb": max(
                generation_peak, float(scorer_peak_vram_mb)
            ),
            "source_cuda_device_name": compute.get("source_cuda_device_name"),
        }
    )
    return record


def run(config: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
    validate_candidate_score_config_v4(config)
    _require(output_dir == Path(str(config["output_dir"])), "V4 scorer output differs")
    _require(not output_dir.exists(), f"terminal V4 scorer output exists: {output_dir}")
    critic_readiness = _json(Path(config["critic_readiness_path"]))
    setflow_confirmation = _json(Path(config["setflow_confirmation_path"]))
    _require(
        authorize_xeditflow_guidance_v4(
            critic_readiness, setflow_confirmation
        )["guidance_authorized"]
        is True,
        "V4 candidate scoring remains blocked before joint readiness",
    )
    generation_summary = _json(Path(config["generation_summary_path"]))
    _require(
        generation_summary.get("status")
        == "XEDITFLOW_V4_SMC_GENERATION_COMPLETE_PENDING_TERMINAL_CRITIC_SCORING"
        and generation_summary.get("terminal_critic_scoring_performed") is False
        and int(generation_summary.get("base_flow_training_seed", -1)) == 20260912,
        "V4 candidate scorer requires terminal pending generation",
    )
    _require(
        str(generation_summary.get("method_id")) == str(config["method_id"])
        and float(generation_summary.get("kappa", -1)) == float(config["kappa"])
        and float(generation_summary.get("temperature", -1))
        == float(config["temperature"])
        and float(generation_summary.get("beta_max", -1))
        == float(config["beta_max"]),
        "V4 candidate scorer generation combination differs",
    )
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    _require(
        not os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES remapping is forbidden",
    )
    gpu = int(config["physical_gpu_index"])
    device = torch.device(str(config["device"]))
    torch.cuda.set_device(device)
    _require(torch.cuda.is_bf16_supported(), "BF16 is unavailable for V4 candidate scoring")
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
    _require(len(sources) == 891, "V4 candidate scorer cohort changed")
    source_by_key = {str(row["source_key"]): row for row in sources}
    validation = load_projection_rows(
        [Path(config["validation_projection_path"])],
        allowed_splits=("VALIDATION",),
    )
    representatives = _representatives_v4(sources, validation)
    candidates = _jsonl(Path(config["candidate_path"]))
    compute_rows = _jsonl(Path(config["generation_compute_path"]))
    compute_by_source = {str(row["source_key"]): row for row in compute_rows}
    by_source: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        _require(
            str(candidate.get("method_id")) == str(config["method_id"])
            and int(candidate.get("base_flow_training_seed", -1)) == 20260912
            and float(candidate.get("kappa", -1)) == float(config["kappa"])
            and float(candidate.get("temperature", -1))
            == float(config["temperature"])
            and float(candidate.get("beta_max", -1))
            == float(config["beta_max"]),
            "V4 candidate scorer candidate combination differs",
        )
        by_source.setdefault(str(candidate["source_key"]), []).append(candidate)
    _require(
        len(compute_by_source) == len(compute_rows) == 891
        and set(by_source) == set(source_by_key) == set(compute_by_source)
        and len(candidates)
        == int(generation_summary.get("generated_candidate_count", -1)),
        "V4 candidate scorer source coverage differs",
    )
    for rows in by_source.values():
        rows.sort(key=lambda row: int(row["generation_rank"]))

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    (output_dir / "run_config.json").write_text(
        json.dumps(dict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    scored_path = output_dir / "critic_scored_candidates.private.jsonl"
    compute_path = output_dir / "matched_compute.scored.jsonl"
    scored_path.write_text("", encoding="utf-8")
    compute_path.write_text("", encoding="utf-8")
    total_forwards = {seed: 0 for seed in CRITIC_SEEDS_V4}
    maximum_compute = 0
    run_peak_vram_mb = 0.0
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    for source in sources:
        source_key = str(source["source_key"])
        source_candidates = by_source[source_key]
        projection_rows = candidate_projection_rows_v4(
            source_candidates,
            source=source,
            representative=representatives[source_key],
        )
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        source_started = time.perf_counter()
        cache_view = _ephemeral_cache_view_v4(
            projection_rows, encoder=bottom_encoder
        )
        predictions_by_seed: dict[int, list[float]] = {}
        calls_by_seed: dict[int, int] = {}
        for seed in CRITIC_SEEDS_V4:
            predictions, calls = _score_member_batch_v4(
                projection_rows,
                model=models[seed],
                checkpoint=checkpoints[seed],
                cache_view=cache_view,
                device=device,
            )
            expected_calls = math.ceil(
                len(source_candidates) / int(checkpoints[seed]["physical_batch_size"])
            )
            _require(calls == expected_calls, "V4 candidate critic forward count differs")
            predictions_by_seed[seed] = predictions
            calls_by_seed[seed] = calls
            total_forwards[seed] += calls
        prediction_tensor = torch.tensor(
            [
                [predictions_by_seed[seed][index] for seed in CRITIC_SEEDS_V4]
                for index in range(len(source_candidates))
            ],
            dtype=torch.float32,
        )
        rewards = uncertainty_penalized_reward_v3(
            prediction_tensor, kappa=float(config["kappa"])
        )
        torch.cuda.synchronize(device)
        source_wall = time.perf_counter() - source_started
        source_peak = torch.cuda.max_memory_allocated(device) / 1024**2
        run_peak_vram_mb = max(run_peak_vram_mb, source_peak)
        _require(
            int(compute_by_source[source_key]["candidate_count"])
            == len(source_candidates)
            and compute_by_source[source_key].get(
                "terminal_critic_forwards_are_reserved_pending_scoring"
            )
            is True,
            "V4 generation compute is not pending exact candidate scoring",
        )
        reconciled = reconcile_candidate_compute_v4(
            compute_by_source[source_key],
            actual_critic_forwards_by_member=[
                calls_by_seed[seed] for seed in CRITIC_SEEDS_V4
            ],
            scorer_wall_time_seconds=source_wall,
            scorer_peak_vram_mb=source_peak,
        )
        maximum_compute = max(
            maximum_compute, int(reconciled["total_forward_equivalents"])
        )
        with compute_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(reconciled, sort_keys=True) + "\n")
        with scored_path.open("a", encoding="utf-8") as handle:
            for index, candidate in enumerate(source_candidates):
                values = prediction_tensor[index]
                handle.write(
                    json.dumps(
                        {
                            **candidate,
                            "critic_seeds": list(CRITIC_SEEDS_V4),
                            "calibrated_seed_predictions": values.tolist(),
                            "critic_ensemble_mean": float(values.mean().item()),
                            "critic_ensemble_sd": float(
                                values.std(unbiased=False).item()
                            ),
                            "uncertainty_penalty_kappa": float(config["kappa"]),
                            "calibrated_reward": float(rewards[index].item()),
                            "study_neutral": True,
                            "critic_self_score_is_diagnostic_only": True,
                            "critic_self_score_used_for_generation_or_selection": False,
                            "independent_evaluator_score": None,
                            "development_test_outcomes_accessed_after_atomic_test": False,
                            "new_final_evaluation_outcomes_accessed": False,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    result = {
        "schema_version": "route_a_v3_route2_xeditflow_candidate_critic_scores.v4",
        "status": "XEDITFLOW_V4_CANDIDATE_CRITIC_SCORING_COMPLETE",
        "base_flow_training_seed": 20260912,
        "method_id": str(config["method_id"]),
        "kappa": float(config["kappa"]),
        "temperature": float(config["temperature"]),
        "beta_max": float(config["beta_max"]),
        "source_count": len(sources),
        "candidate_count": len(candidates),
        "critic_seeds": list(CRITIC_SEEDS_V4),
        "critic_forward_counts_by_member": {
            str(seed): total_forwards[seed] for seed in CRITIC_SEEDS_V4
        },
        "maximum_total_forward_equivalents_per_source": maximum_compute,
        "forward_equivalent_ceiling_per_source": 320,
        "reservation_reconciled_for_every_source": True,
        "study_policy": "UNKNOWN_STUDY_SCALE_FIXED_1",
        "critic_self_score_used_for_generation_or_selection": False,
        "wall_time_seconds": time.time() - started,
        "peak_vram_mb": run_peak_vram_mb,
        "physical_gpu_index": gpu,
        "cuda_device": cuda,
        "cpu_fallback_used": False,
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
            entrypoint="score_route2_xeditflow_candidates_v4",
            evaluation_outcomes_accessed=False,
        )
        raise
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
