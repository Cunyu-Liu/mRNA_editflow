#!/usr/bin/env python3
"""Generate K=8 TRAIN rollouts and score them with the frozen Critic V3 ensemble."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_development_projection_v3 import load_projection_rows
from core.route2_gpu_failure_evidence import cuda_device_observation, write_gpu_failure_evidence
from core.route2_mrnabert_lora_v3 import disabled_lora_residuals_v3
from core.route2_source_token_cache_v3 import SourceTokenCacheIndexV3, load_source_token_cache_v3
from core.route2_xeditcritic_training_data_v3 import RNA_TOKEN, descriptor_category
from core.route2_xeditcritic_v3 import XEditCriticV3
from core.route2_xeditflow_gate_v3 import authorize_xeditflow_guidance_v3
from core.route2_xeditflow_value_rollouts_v3 import (
    build_value_train_state_rows_v3,
    flow_state_from_value_row_v3,
    frozen_rollout_score_row_v3,
    generation_metadata_from_value_row_v3,
    terminal_rollout_row_v3,
    value_rollout_seed_v3,
)
from core.route2_xeditflow_value_training_v3 import CRITIC_SEEDS_V3
from core.route2_xeditsetflow_sampling_v3 import sample_many_setflow_v3
from core.route2_xeditsetflow_training_v3 import (
    setflow_records_from_projection_rows,
    setflow_vocabs,
)
from scripts.route_a_v3.route2_mrnabert_lora_edit_site_encoder_v3 import (
    TrainableMRNABERTEditSiteEncoderV3,
)
from scripts.route_a_v3.run_route2_xeditcritic_v3_atomic_frozen_test import _load_lora_state
from scripts.route_a_v3.train_route2_xeditcritic_v3 import (
    XEditCriticCollatorV3,
    _move,
)
from scripts.route_a_v3.validate_route2_xeditsetflow_v3 import load_setflow_checkpoint_v3


class XEditFlowValueRolloutRunnerV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowValueRolloutRunnerV3Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _jsonl_batches(path: Path, batch_size: int) -> Iterator[list[dict[str, Any]]]:
    _require(batch_size > 0, "JSONL batch size is invalid")
    batch: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            _require(isinstance(row, dict), f"JSONL row is not an object: {path}")
            batch.append(row)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def validate_value_rollout_config_v3(config: Mapping[str, Any]) -> None:
    _require(
        config.get("schema_version")
        == "route_a_v3_route2_xeditflow_value_rollout_config.v1",
        "unexpected value rollout config schema",
    )
    _require(int(config.get("base_flow_training_seed", -1)) == 20260904, "value rollout base-flow seed changed")
    _require(int(config.get("rollouts_per_state", -1)) == 8, "value rollout K changed")
    _require(int(config.get("states_per_record", -1)) == 2, "value state multiplicity changed")
    _require(int(config.get("state_pass_index", -1)) == 0, "value state pass changed")
    _require(str(config.get("setflow_arm")) in {"f2", "f3"}, "value rollout SetFlow arm is not selectable")
    _require(int(config.get("sampling_state_batch_size", 0)) > 0, "value sampling batch size is invalid")
    _require(int(config.get("trajectory_forward_batch_size", 0)) > 0, "value trajectory batch size is invalid")
    _require(int(config.get("critic_batch_size", 0)) > 0, "value critic batch size is invalid")
    _require(int(config.get("critic_online_microbatch_size", 0)) > 0, "value critic microbatch size is invalid")
    physical_gpu = int(config.get("physical_gpu_index", -1))
    _require(physical_gpu in set(range(6)), "value rollout GPU is outside 0-5")
    _require(str(config.get("device")) == f"cuda:{physical_gpu}", "value rollout device provenance changed")
    _require(str(config.get("output_dir", "")).startswith("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"), "value rollout outputs left Route 2 /mnt")
    _require(config.get("development_test_outcomes_accessed") is False, "value rollout config accessed Development TEST")
    _require(config.get("new_final_evaluation_outcomes_accessed") is False, "value rollout config accessed Evaluation")


def _critic_examples_v3(
    rows: Sequence[Mapping[str, Any]],
    vocabs: Mapping[str, Mapping[str, int]],
) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        source = str(row["source_sequence"])
        candidate = str(row["candidate_sequence"])
        _require(len(source) == len(candidate) <= 1000, "generated Critic sequence geometry differs")
        edits = tuple(
            (
                int(edit["position"]),
                str(edit["source_base"]),
                str(edit["candidate_base"]),
            )
            for edit in row["source_relative_edits"]
        )
        _require(
            edits
            == tuple(
                (index, left, right)
                for index, (left, right) in enumerate(zip(source, candidate))
                if left != right
            ),
            "generated Critic edit bundle differs",
        )
        descriptor = row["endpoint_descriptor"]
        examples.append(
            {
                "record_id": f"{row['state_id']}:{int(row['rollout_index'])}",
                "source_group": str(row["source_group_id"]),
                "task": str(row["task_id"]),
                "source": torch.tensor([RNA_TOKEN[base] for base in source], dtype=torch.long),
                "candidate": torch.tensor([RNA_TOKEN[base] for base in candidate], dtype=torch.long),
                "edits": edits,
                "target": 0.0,
                "scaled_target": 0.0,
                "target_scale": 1.0,
                "sample_weight": 1.0,
                "study": 0,
                "assay": vocabs["assay"].get(str(row["assay_category"]), 0),
                "context": vocabs["context"].get(str(row["context_category"]), 0),
                "quantity": vocabs["quantity"].get(str(descriptor["quantity_family"]), 0),
                "measurement": vocabs["measurement"].get(str(descriptor["measurement_form"]), 0),
                "numerator": vocabs["numerator"].get(descriptor_category(descriptor["numerator_family"]), 0),
                "denominator": vocabs["denominator"].get(descriptor_category(descriptor["denominator_family"]), 0),
                "region": int(row["region_id"]),
            }
        )
    return examples


def _critic_candidate_identity_v3(row: Mapping[str, Any]) -> tuple[Any, ...]:
    descriptor = row["endpoint_descriptor"]
    return (
        str(row["source_sequence"]),
        str(row["candidate_sequence"]),
        str(row["task_id"]),
        str(descriptor["quantity_family"]),
        str(descriptor["measurement_form"]),
        descriptor_category(descriptor["numerator_family"]),
        descriptor_category(descriptor["denominator_family"]),
        str(row["assay_category"]),
        str(row["context_category"]),
        int(row["region_id"]),
    )


def _load_critic_member_v3(
    checkpoint_path: Path,
    *,
    selected_arm: str,
    seed: int,
    model_path: Path,
    device: torch.device,
) -> tuple[XEditCriticV3, TrainableMRNABERTEditSiteEncoderV3, Mapping[str, Mapping[str, int]]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require(checkpoint.get("arm") == selected_arm, "value rollout Critic arm differs")
    _require(checkpoint.get("control_mode") == "NONE", "value rollout Critic control differs")
    _require(checkpoint.get("candidate_bundle_permutation") is False, "value rollout Critic is permuted")
    _require(int(checkpoint.get("seed", -1)) == seed, "value rollout Critic seed differs")
    vocabs = checkpoint["vocabs"]
    model = XEditCriticV3(
        arm=selected_arm,
        control_mode="NONE",
        study_count=len(vocabs["study"]),
        assay_count=len(vocabs["assay"]),
        context_count=len(vocabs["context"]),
        quantity_count=len(vocabs["quantity"]),
        measurement_count=len(vocabs["measurement"]),
        numerator_count=len(vocabs["numerator"]),
        denominator_count=len(vocabs["denominator"]),
        pretrained_width=768,
        dropout=0.10,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    encoder = TrainableMRNABERTEditSiteEncoderV3(
        model_path,
        device,
        rank=16,
        alpha=32.0,
        dropout=0.05,
    )
    if selected_arm == "C3":
        _require("lora_state_dict" in checkpoint, "C3 value rollout checkpoint lacks LoRA")
        _load_lora_state(encoder, checkpoint["lora_state_dict"])
    else:
        _require("lora_state_dict" not in checkpoint, "C2 value rollout checkpoint unexpectedly contains LoRA")
    encoder.eval()
    del checkpoint
    return model, encoder, vocabs


@torch.no_grad()
def _score_critic_member_v3(
    terminal_path: Path,
    output_path: Path,
    *,
    checkpoint_path: Path,
    selected_arm: str,
    seed: int,
    model_path: Path,
    device: torch.device,
    batch_size: int,
    microbatch_size: int,
) -> int:
    _require(not output_path.exists(), f"Critic member score output exists: {output_path}")
    model, encoder, vocabs = _load_critic_member_v3(
        checkpoint_path,
        selected_arm=selected_arm,
        seed=seed,
        model_path=model_path,
        device=device,
    )
    collator = XEditCriticCollatorV3(pretrained_width=768)
    count = 0
    with output_path.open("w", encoding="utf-8") as output:
        for rows in _jsonl_batches(terminal_path, batch_size):
            unique_rows = []
            identity_to_index: dict[tuple[Any, ...], int] = {}
            inverse = []
            for row in rows:
                identity = _critic_candidate_identity_v3(row)
                if identity not in identity_to_index:
                    identity_to_index[identity] = len(unique_rows)
                    unique_rows.append(row)
                inverse.append(identity_to_index[identity])
            raw_batch = collator(_critic_examples_v3(unique_rows, vocabs))
            unique_predictions: list[float] = []
            for start in range(0, len(unique_rows), microbatch_size):
                end = min(len(unique_rows), start + microbatch_size)
                indices = list(range(start, end))
                sliced = {
                    key: (
                        value[indices]
                        if isinstance(value, torch.Tensor)
                        else [value[index] for index in indices]
                        if isinstance(value, list)
                        else value
                    )
                    for key, value in raw_batch.items()
                }
                batch = _move(sliced, device)
                context = disabled_lora_residuals_v3(encoder.model) if selected_arm == "C2" else nullcontext()
                with context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    feature_batch = encoder.forward(batch)
                    predictions = model(feature_batch)["mean"].float().cpu().tolist()
                _require(len(predictions) == len(indices), "Critic member prediction count differs")
                unique_predictions.extend(float(value) for value in predictions)
            _require(len(unique_predictions) == len(unique_rows), "unique Critic prediction count differs")
            for row, unique_index in zip(rows, inverse):
                output.write(
                    json.dumps(
                        {
                            "state_id": str(row["state_id"]),
                            "rollout_index": int(row["rollout_index"]),
                            "critic_seed": seed,
                            "standardized_prediction": unique_predictions[unique_index],
                            "study_neutral": True,
                            "unknown_study_scale": 1.0,
                            "development_test_outcomes_accessed": False,
                            "new_final_evaluation_outcomes_accessed": False,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                count += 1
    del model, encoder
    torch.cuda.empty_cache()
    return count


def _line_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                _require(isinstance(value, dict), f"rollout member row is invalid: {path}")
                yield value


def _combine_member_scores_v3(
    terminal_path: Path,
    member_paths: Mapping[int, Path],
    output_path: Path,
) -> int:
    _require(not output_path.exists(), f"frozen rollout score output exists: {output_path}")
    sentinel = object()
    streams: list[Iterable[Any]] = [_line_rows(terminal_path)] + [
        _line_rows(member_paths[seed]) for seed in CRITIC_SEEDS_V3
    ]
    count = 0
    with output_path.open("w", encoding="utf-8") as output:
        for values in itertools.zip_longest(*streams, fillvalue=sentinel):
            _require(all(value is not sentinel for value in values), "Critic member score file lengths differ")
            terminal = values[0]
            members = {
                seed: values[index + 1]
                for index, seed in enumerate(CRITIC_SEEDS_V3)
            }
            output.write(json.dumps(frozen_rollout_score_row_v3(terminal, members), sort_keys=True) + "\n")
            count += 1
    return count


def run(config: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
    validate_value_rollout_config_v3(config)
    _require(output_dir == Path(config["output_dir"]), "value rollout output path differs from frozen config")
    _require(not output_dir.exists(), f"terminal value rollout output exists: {output_dir}")
    readiness = _json(Path(config["critic_readiness_path"]))
    setflow_confirmation = _json(Path(config["setflow_confirmation_path"]))
    authorization = authorize_xeditflow_guidance_v3(readiness, setflow_confirmation)
    _require(authorization["guidance_authorized"] is True, "value rollouts remain blocked before readiness")
    refit = _json(Path(config["critic_refit_manifest_path"]))
    _require(refit.get("status") == "XEDITCRITIC_V3_ALL_DEVELOPMENT_REFIT_COMPLETE", "value rollouts require three Critic refits")
    selected_arm = str(refit.get("selected_arm"))
    _require(selected_arm in {"C2", "C3"}, "value rollout selected Critic differs")
    checkpoints = {int(row["seed"]): Path(row["checkpoint_path"]) for row in refit.get("checkpoints", ())}
    _require(tuple(sorted(checkpoints)) == CRITIC_SEEDS_V3, "value rollout Critic refit seeds differ")
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden")
    physical_gpu = int(config["physical_gpu_index"])
    device = torch.device(str(config["device"]))
    torch.cuda.set_device(device)
    _require(torch.cuda.is_bf16_supported(), "BF16 is unavailable for value rollouts")
    cuda = cuda_device_observation(physical_gpu, require_physical_index_match=True)
    setflow, setflow_checkpoint = load_setflow_checkpoint_v3(
        Path(config["setflow_checkpoint_path"]), str(config["setflow_arm"]), device
    )
    _require(int(setflow_checkpoint["training_provenance"]["seed"]) == 20260904, "guidance screen SetFlow seed differs")
    train_rows = load_projection_rows(
        [Path(config["train_projection_path"])], allowed_splits=("TRAIN",)
    )
    records, eligibility = setflow_records_from_projection_rows(train_rows)
    _require(len(records) == int(config["expected_train_record_count"]), "value rollout TRAIN record count differs")
    vocabs = setflow_vocabs(records)
    _require(vocabs == setflow_checkpoint["vocabs"], "value rollout SetFlow vocabulary differs")
    states = build_value_train_state_rows_v3(
        records,
        vocabs,
        base_flow_training_seed=20260904,
        state_pass_index=0,
        states_per_record=2,
    )
    cache = SourceTokenCacheIndexV3(
        load_source_token_cache_v3(Path(config["source_token_cache_path"]))
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    (output_dir / "run_config.json").write_text(
        json.dumps(dict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    state_path = output_dir / "train_states.jsonl"
    terminal_path = output_dir / "terminal_rollouts.private.jsonl"
    with state_path.open("w", encoding="utf-8") as handle:
        for row in states:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    terminal_causes = Counter()
    base_flow_forwards = 0
    sampling_batches = 0
    started = time.time()
    with terminal_path.open("w", encoding="utf-8") as output:
        state_batch_size = int(config["sampling_state_batch_size"])
        for state_start in range(0, len(states), state_batch_size):
            batch_rows = states[state_start : state_start + state_batch_size]
            roots = []
            metadata = []
            seeds = []
            identities = []
            for local_index, row in enumerate(batch_rows):
                state_index = state_start + local_index
                for rollout_index in range(8):
                    roots.append(flow_state_from_value_row_v3(row))
                    metadata.append(generation_metadata_from_value_row_v3(row))
                    seeds.append(value_rollout_seed_v3(state_index, rollout_index))
                    identities.append((row, state_index, rollout_index))
            sampled, forward_batches = sample_many_setflow_v3(
                setflow,
                str(config["setflow_arm"]),
                roots,
                metadata,
                seeds,
                source_cache=cache,
                device=device,
                forward_batch_size=int(config["trajectory_forward_batch_size"]),
            )
            sampling_batches += forward_batches
            for identity, result in zip(identities, sampled):
                state_row, state_index, rollout_index = identity
                terminal, actions, forwards = result
                row = terminal_rollout_row_v3(
                    state_row,
                    state_index=state_index,
                    rollout_index=rollout_index,
                    terminal_state=terminal,
                    trajectory_actions=actions,
                    base_flow_forwards=forwards,
                )
                output.write(json.dumps(row, sort_keys=True) + "\n")
                terminal_causes[str(terminal.terminal_cause)] += 1
                base_flow_forwards += int(forwards)
    terminal_count = len(states) * 8
    _require(sum(terminal_causes.values()) == terminal_count, "value terminal rollout count differs")
    member_paths = {
        seed: output_dir / f"critic_member_seed{seed}.private.jsonl"
        for seed in CRITIC_SEEDS_V3
    }
    member_counts = {}
    for seed in CRITIC_SEEDS_V3:
        member_counts[seed] = _score_critic_member_v3(
            terminal_path,
            member_paths[seed],
            checkpoint_path=checkpoints[seed],
            selected_arm=selected_arm,
            seed=seed,
            model_path=Path(config["mrnabert_model_path"]),
            device=device,
            batch_size=int(config["critic_batch_size"]),
            microbatch_size=int(config["critic_online_microbatch_size"]),
        )
    _require(all(count == terminal_count for count in member_counts.values()), "value Critic member count differs")
    score_path = output_dir / "frozen_rollout_scores.private.jsonl"
    combined_count = _combine_member_scores_v3(terminal_path, member_paths, score_path)
    _require(combined_count == terminal_count, "combined value rollout count differs")
    summary = {
        "schema_version": "route_a_v3_route2_xeditflow_value_rollout_run.v3",
        "status": "XEDITFLOW_V3_VALUE_ROLLOUTS_COMPLETE",
        "base_flow_training_seed": 20260904,
        "setflow_arm": str(config["setflow_arm"]),
        "critic_arm": selected_arm,
        "critic_seeds": list(CRITIC_SEEDS_V3),
        "train_record_count": len(records),
        "over_budget_excluded_train_record_count": eligibility["skipped_over_budget_count"],
        "states_per_record": 2,
        "state_count": len(states),
        "rollouts_per_state": 8,
        "terminal_rollout_count": terminal_count,
        "terminal_cause_counts": dict(sorted(terminal_causes.items())),
        "base_flow_trajectory_forwards": base_flow_forwards,
        "base_flow_forward_batches": sampling_batches,
        "critic_member_prediction_counts": {str(seed): member_counts[seed] for seed in CRITIC_SEEDS_V3},
        "state_path": str(state_path),
        "terminal_rollout_path": str(terminal_path),
        "frozen_rollout_score_path": str(score_path),
        "wall_time_seconds": time.time() - started,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "cpu_fallback_used": False,
        "study_neutral": True,
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
        **cuda,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
            entrypoint="generate_route2_xeditflow_value_rollouts_v3",
            evaluation_outcomes_accessed=False,
        )
        raise
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
