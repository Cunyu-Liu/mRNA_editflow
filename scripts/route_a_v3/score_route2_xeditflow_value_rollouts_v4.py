#!/usr/bin/env python3
"""Score V4 TRAIN rollouts with the frozen study-neutral refit ensemble."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_bottom_encoder_chunk_cache_v4 import (
    assemble_frozen_bottom_encoder_chunk_cache_v4,
)
from core.route2_gpu_failure_evidence import cuda_device_observation, write_gpu_failure_evidence
from core.route2_xeditcritic_batch_v4 import (
    FrozenBottomEncoderChunkCacheViewV4,
    XEditCriticCollatorV4,
    XEditCriticDatasetV4,
)
from core.route2_xeditcritic_training_data_v3 import (
    UNKNOWN_CATEGORY,
    records_from_projection_rows,
)
from core.route2_xeditflow_gate_v4 import authorize_xeditflow_guidance_v4
from core.route2_xeditflow_value_rollouts_v4 import frozen_rollout_score_row_v4
from core.route2_xeditflow_value_training_v4 import CRITIC_SEEDS_V4
from scripts.route_a_v3.route2_mrnabert_bottom_six_encoder_v4 import (
    FrozenMRNABERTBottomSixEncoderV4,
)
from scripts.route_a_v3.run_route2_xeditcritic_v4_atomic_frozen_test import _scaler
from scripts.route_a_v3.train_route2_xeditcritic_v4 import (
    _build_model,
    evaluation_index_batches_v4,
    screen_run_spec_v4,
)


class XEditFlowValueCriticScorerV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowValueCriticScorerV4Error(message)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON root is not an object: {path}")
    return payload


def _jsonl_batches(path: Path, batch_size: int) -> Iterator[list[dict[str, Any]]]:
    _require(batch_size > 0, "V4 critic scoring batch size is invalid")
    batch: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            _require(isinstance(row, dict), "V4 terminal rollout row is not an object")
            batch.append(row)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def validate_value_critic_score_config_v4(config: Mapping[str, Any]) -> None:
    _require(
        config.get("schema_version")
        == "route_a_v3_route2_xeditflow_value_critic_score_config.v4",
        "unexpected V4 value critic score config schema",
    )
    _require(
        tuple(int(seed) for seed in config.get("critic_seeds", ()))
        == CRITIC_SEEDS_V4,
        "V4 value critic seeds changed",
    )
    runtime_paths = config.get("critic_refit_runtime_config_paths")
    _require(
        isinstance(runtime_paths, Mapping)
        and set(runtime_paths) == {str(seed) for seed in CRITIC_SEEDS_V4},
        "V4 value critic refit runtime config paths differ",
    )
    _require(
        int(config.get("candidate_batch_size", 0)) > 0,
        "V4 value critic candidate batch size is invalid",
    )
    _require(
        config.get("study_policy") == "UNKNOWN_STUDY_SCALE_FIXED_1",
        "V4 value critic study policy changed",
    )
    _require(
        config.get("prediction_scale") == "TASK_ROBUST_STANDARDIZED_EFFECT",
        "V4 value critic prediction scale changed",
    )
    _require(
        config.get("trajectory_mode_used_as_critic_input") is False,
        "V4 trajectory mode entered critic inference",
    )
    physical_gpu = int(config.get("physical_gpu_index", -1))
    _require(
        physical_gpu in set(range(6)),
        "V4 value critic GPU is outside 0-5",
    )
    _require(
        str(config.get("device")) == f"cuda:{physical_gpu}",
        "V4 value critic device provenance changed",
    )
    for field in ("terminal_rollout_path", "output_dir"):
        _require(
            str(config.get(field, "")).startswith(
                "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"
            ),
            f"V4 value critic {field} left Route 2 /mnt",
        )
    _require(
        config.get("independent_evaluator_used") is False,
        "independent evaluator entered V4 value critic scoring",
    )
    _require(
        config.get("development_test_outcomes_accessed_after_atomic_test") is False,
        "V4 value critic scoring reopened Development TEST",
    )
    _require(
        config.get("new_final_evaluation_outcomes_accessed") is False,
        "V4 value critic scoring accessed Evaluation",
    )


def projection_rows_from_terminal_rollouts_v4(
    terminal_rows: Sequence[Mapping[str, Any]], *, global_start: int
) -> list[dict[str, Any]]:
    _require(bool(terminal_rows) and global_start >= 0, "V4 terminal score batch differs")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for offset, terminal in enumerate(terminal_rows):
        _require(
            terminal.get("schema_version")
            == "route_a_v3_route2_xeditflow_terminal_rollout.v4",
            "unexpected V4 terminal rollout schema",
        )
        _require(
            terminal.get("setflow_mode_is_fixed_trajectory_state") is True,
            "V4 terminal rollout did not preserve mode state",
        )
        source = str(terminal["source_sequence"])
        candidate = str(terminal["candidate_sequence"])
        edits = [dict(edit) for edit in terminal["source_relative_edits"]]
        _require(
            len(source) == len(candidate)
            and edits
            == [
                {
                    "position": index,
                    "source_base": left,
                    "candidate_base": right,
                }
                for index, (left, right) in enumerate(zip(source, candidate))
                if left != right
            ],
            "V4 terminal rollout candidate bundle differs",
        )
        record_id = f"generated-{global_start + offset:012d}"
        _require(record_id not in seen, "V4 generated critic identity is duplicated")
        seen.add(record_id)
        rows.append(
            {
                "canonical_record_id": record_id,
                "split": "VALIDATION",
                "task_id": str(terminal["task_id"]),
                "study_unit_id": UNKNOWN_CATEGORY,
                "source_group_id": str(terminal["source_group_id"]),
                "assay_id": str(terminal["assay_category"]),
                "biological_context_id": str(terminal["context_category"]),
                "region_id": int(terminal["region_id"]),
                "endpoint_id": "GENERATED_V4_STUDY_NEUTRAL",
                "endpoint_descriptor": dict(terminal["endpoint_descriptor"]),
                "source_sequence": source,
                "candidate_sequence": candidate,
                "source_relative_edits": edits,
                "direction_normalized_delta": 0.0,
                "dummy_target_for_inference_only": True,
                "development_test_outcomes_accessed_after_atomic_test": False,
                "new_final_evaluation_outcomes_accessed": False,
            }
        )
    return rows


def _ephemeral_cache_view_v4(
    rows: Sequence[Mapping[str, Any]],
    *,
    encoder: FrozenMRNABERTBottomSixEncoderV4,
) -> FrozenBottomEncoderChunkCacheViewV4:
    sequences = sorted(
        {
            str(sequence)
            for row in rows
            for sequence in (row["source_sequence"], row["candidate_sequence"])
        }
    )
    sequence_to_index = {sequence: index for index, sequence in enumerate(sequences)}
    encoded = encoder.encode_online(
        {index: sequence for index, sequence in enumerate(sequences)}
    )
    payload = assemble_frozen_bottom_encoder_chunk_cache_v4(
        rows,
        sequence_to_index=sequence_to_index,
        encoded=encoded,
        model_id="EPHEMERAL_V4_GENERATED_CANDIDATE_BOTTOM_SIX",
        pretrained_parameter_count=encoder.parameter_count,
        attention_backend=encoder.attention_backend,
    )
    return FrozenBottomEncoderChunkCacheViewV4(
        payload, {str(row["canonical_record_id"]) for row in rows}
    )


def _load_refit_models_v4(
    config: Mapping[str, Any], *, device: torch.device
) -> tuple[
    dict[int, torch.nn.Module],
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    manifest = _json(Path(config["critic_refit_manifest_path"]))
    _require(
        manifest.get("status") == "XEDITCRITIC_V4_ALL_DEVELOPMENT_REFIT_COMPLETE"
        and manifest.get("required_seeds") == list(CRITIC_SEEDS_V4)
        and int(manifest.get("completed_refit_count", -1)) == 3
        and int(manifest.get("refit_pass_count", -1)) == 8,
        "V4 value critic refit manifest differs",
    )
    checkpoint_paths = {
        int(row["seed"]): Path(row["checkpoint_path"])
        for row in manifest.get("checkpoints", ())
    }
    _require(
        tuple(sorted(checkpoint_paths)) == CRITIC_SEEDS_V4,
        "V4 value critic refit checkpoint seeds differ",
    )
    runtime_paths = config["critic_refit_runtime_config_paths"]
    models: dict[int, torch.nn.Module] = {}
    checkpoints: dict[int, dict[str, Any]] = {}
    runtimes: dict[int, dict[str, Any]] = {}
    for seed in CRITIC_SEEDS_V4:
        runtime = _json(Path(runtime_paths[str(seed)]))
        checkpoint = torch.load(
            checkpoint_paths[seed], map_location="cpu", weights_only=False
        )
        _require(
            runtime.get("run_stage") == "REFIT"
            and int(runtime.get("training_seed", -1)) == seed,
            "V4 value critic refit runtime identity differs",
        )
        _require(
            checkpoint.get("schema_version")
            == "route_a_v3_route2_xeditcritic_v4_refit_checkpoint.v1"
            and checkpoint.get("run_stage") == "REFIT"
            and checkpoint.get("run_id") == "v4_full"
            and int(checkpoint.get("seed", -1)) == seed
            and int(checkpoint.get("selected_pass", -1)) == 8
            and checkpoint.get("candidate_bundle_permutation") is False
            and int(checkpoint.get("development_test_outcome_reads", -1)) == 0
            and int(checkpoint.get("new_final_evaluation_outcome_reads", -1)) == 0,
            "V4 value critic refit checkpoint identity differs",
        )
        model, capacity = _build_model(
            runtime,
            screen_run_spec_v4(runtime, "v4_full"),
            checkpoint["vocabs"],
            device=device,
        )
        _require(
            capacity == checkpoint["capacity"],
            "V4 value critic refit checkpoint capacity changed",
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        models[seed] = model
        checkpoints[seed] = checkpoint
        runtimes[seed] = runtime
    return models, checkpoints, runtimes


def _move(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


@torch.no_grad()
def _score_member_batch_v4(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: torch.nn.Module,
    checkpoint: Mapping[str, Any],
    cache_view: FrozenBottomEncoderChunkCacheViewV4,
    device: torch.device,
) -> tuple[list[float], int]:
    records = records_from_projection_rows(rows)
    record_by_id = {record.record_id: record for record in records}
    dataset = XEditCriticDatasetV4(
        records,
        all_records=record_by_id,
        vocabs=checkpoint["vocabs"],
        target_scaler=_scaler(checkpoint["target_scaler"]),
        cache=None,
        neutral_studies={UNKNOWN_CATEGORY},
    )
    physical_batch = int(checkpoint["physical_batch_size"])
    collator = XEditCriticCollatorV4(
        cache_view, minimum_physical_batch=4
    )
    predictions: list[float] = []
    forward_count = 0
    for indices, valid_count in evaluation_index_batches_v4(
        len(dataset), physical_batch
    ):
        batch = _move(collator([dataset[index] for index in indices]), device)
        _require(
            bool((batch["study_ids"] == 0).all().item()),
            "V4 generated critic inference study scale is not unknown=1",
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            values = model(batch)["mean"].float()[:valid_count]
        _require(
            bool(torch.isfinite(values).all().item()),
            "V4 generated critic prediction is nonfinite",
        )
        predictions.extend(float(value) for value in values.cpu().tolist())
        forward_count += 1
    _require(
        len(predictions) == len(rows),
        "V4 generated critic prediction count differs",
    )
    return predictions, forward_count


def run(config: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
    validate_value_critic_score_config_v4(config)
    _require(
        output_dir == Path(str(config["output_dir"])),
        "V4 value critic output path differs from frozen config",
    )
    _require(
        not output_dir.exists(),
        f"terminal V4 value critic output exists: {output_dir}",
    )
    critic_readiness = _json(Path(config["critic_readiness_path"]))
    setflow_confirmation = _json(Path(config["setflow_confirmation_path"]))
    authorization = authorize_xeditflow_guidance_v4(
        critic_readiness, setflow_confirmation
    )
    _require(
        authorization["guidance_authorized"] is True,
        "V4 value critic scoring remains blocked before joint readiness",
    )
    rollout_summary = _json(Path(config["rollout_summary_path"]))
    _require(
        rollout_summary.get("status")
        == "XEDITFLOW_V4_VALUE_ROLLOUTS_COMPLETE_PENDING_CRITIC_SCORING"
        and rollout_summary.get("fixed_seed_replayable") is True
        and int(rollout_summary.get("fixed_seed_replay_failure_count", -1)) == 0
        and rollout_summary.get("critic_scoring_performed") is False,
        "V4 value critic scorer requires terminal replay-checked rollouts",
    )
    expected_count = int(rollout_summary["terminal_rollout_count"])
    _require(
        expected_count == int(config["expected_terminal_rollout_count"]),
        "V4 value critic expected rollout count differs",
    )
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    _require(
        not os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES remapping is forbidden",
    )
    physical_gpu = int(config["physical_gpu_index"])
    device = torch.device(str(config["device"]))
    torch.cuda.set_device(device)
    _require(torch.cuda.is_bf16_supported(), "BF16 is unavailable for V4 critic scoring")
    cuda = cuda_device_observation(
        physical_gpu, require_physical_index_match=True
    )
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
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    (output_dir / "run_config.json").write_text(
        json.dumps(dict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    score_path = output_dir / "critic_scored_rollouts.private.jsonl"
    score_path.write_text("", encoding="utf-8")
    forward_counts = {seed: 0 for seed in CRITIC_SEEDS_V4}
    processed = 0
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    with score_path.open("a", encoding="utf-8") as output:
        for terminal_rows in _jsonl_batches(
            Path(config["terminal_rollout_path"]),
            int(config["candidate_batch_size"]),
        ):
            projection_rows = projection_rows_from_terminal_rollouts_v4(
                terminal_rows, global_start=processed
            )
            cache_view = _ephemeral_cache_view_v4(
                projection_rows, encoder=bottom_encoder
            )
            predictions_by_seed: dict[int, list[float]] = {}
            for seed in CRITIC_SEEDS_V4:
                predictions, calls = _score_member_batch_v4(
                    projection_rows,
                    model=models[seed],
                    checkpoint=checkpoints[seed],
                    cache_view=cache_view,
                    device=device,
                )
                predictions_by_seed[seed] = predictions
                forward_counts[seed] += calls
            for row_index, terminal in enumerate(terminal_rows):
                members = {
                    seed: {
                        "state_id": str(terminal["state_id"]),
                        "rollout_index": int(terminal["rollout_index"]),
                        "trajectory_mode_id": int(terminal["trajectory_mode_id"]),
                        "critic_seed": seed,
                        "standardized_prediction": predictions_by_seed[seed][
                            row_index
                        ],
                        "study_neutral": True,
                    }
                    for seed in CRITIC_SEEDS_V4
                }
                output.write(
                    json.dumps(
                        frozen_rollout_score_row_v4(terminal, members),
                        sort_keys=True,
                    )
                    + "\n"
                )
            processed += len(terminal_rows)
    _require(processed == expected_count, "V4 value critic processed rollout count differs")
    result = {
        "schema_version": "route_a_v3_route2_xeditflow_value_critic_scores.v4",
        "status": "XEDITFLOW_V4_VALUE_CRITIC_SCORING_COMPLETE",
        "terminal_rollout_count": processed,
        "critic_seeds": list(CRITIC_SEEDS_V4),
        "critic_forward_counts_by_member": {
            str(seed): forward_counts[seed] for seed in CRITIC_SEEDS_V4
        },
        "study_policy": "UNKNOWN_STUDY_SCALE_FIXED_1",
        "prediction_scale": "TASK_ROBUST_STANDARDIZED_EFFECT",
        "trajectory_mode_used_as_critic_input": False,
        "wall_time_seconds": time.time() - started,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "physical_gpu_index": physical_gpu,
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
            entrypoint="score_route2_xeditflow_value_rollouts_v4",
            evaluation_outcomes_accessed=False,
        )
        raise
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
