#!/usr/bin/env python3
"""Measure formal Critic V4 capacity and training memory on TRAIN geometry."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_bottom_encoder_chunk_cache_v4 import (
    BottomEncodedChunkV4,
    BottomEncodedSequenceV4,
    load_frozen_bottom_encoder_chunk_cache_v4,
    require_frozen_bottom_encoder_chunk_cache_identity_v4,
)
from core.route2_development_projection_v3 import load_projection_rows
from core.route2_mrnabert_edit_site_features_v3 import ChunkSpan
from core.route2_xeditcritic_batch_v4 import (
    FrozenBottomEncoderChunkCacheViewV4,
    XEditCriticCollatorV4,
)
from core.route2_xeditcritic_training_data_v3 import (
    RNA_TOKEN,
)
from core.route2_xeditcritic_training_v4 import (
    XEditCriticTrainingV4Error,
    critic_v4_optimizer_parameter_groups,
    require_physical_gpu_scope_v4,
    select_physical_batch_from_memory_v4,
)
from core.route2_xeditcritic_v4 import (
    XEditCriticV4,
    require_v4_trainable_parameter_range,
)
from scripts.route_a_v3.route2_mrnabert_upper_six_encoder_v4 import (
    TrainableMRNABERTUpperSixEncoderV4,
)
from scripts.route_a_v3.route2_mrnabert_bottom_six_encoder_v4 import (
    FrozenMRNABERTBottomSixEncoderV4,
    compare_bottom_encoded_sequences_v4,
)
from scripts.route_a_v3.train_route2_xeditcritic_v3 import require_cuda


class XEditCriticPreflightV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticPreflightV4Error(message)


def require_preflight_authorization_v4(
    authorization: Mapping[str, Any],
    *,
    current_git_head: str,
) -> None:
    _require(
        authorization.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_preflight_authorization.v1",
        "Critic V4 preflight authorization schema is absent",
    )
    _require(
        authorization.get("status") == "XEDITCRITIC_V4_PREFLIGHT_AUTHORIZED",
        "Critic V4 preflight is not authorized",
    )
    _require(str(authorization.get("authorized_git_head")) == str(current_git_head), "Critic V4 preflight authorization is for another Git HEAD")
    barriers = authorization.get("barriers", {})
    required = (
        "all_five_c3_jobs_terminal",
        "c3_terminal_summaries_read_exactly_once",
        "a100_current_head_focused_tests_passed",
        "a100_current_head_v332_tests_passed",
        "bottom_six_cache_terminal_complete",
    )
    _require(all(barriers.get(key) is True for key in required), "a Critic V4 preflight barrier is not satisfied")
    _require(int(authorization.get("development_test_outcome_reads", -1)) == 0, "preflight authorization reports a Development TEST read")
    _require(int(authorization.get("new_final_evaluation_outcome_reads", -1)) == 0, "preflight authorization reports a new Evaluation read")


def select_train_geometry_records_v4(
    rows: Sequence[Mapping[str, Any]], *, count: int = 32
) -> list[Mapping[str, Any]]:
    """Select a fixed high-memory cohort without inspecting any target value."""

    train = [row for row in rows if str(row.get("split")) == "TRAIN"]
    _require(len(train) >= count, "TRAIN geometry has fewer than 32 records")
    selected = sorted(
        train,
        key=lambda row: (
            -len(row["source_relative_edits"]),
            -len(str(row["source_sequence"])),
            str(row["canonical_record_id"]),
        ),
    )[:count]
    _require(len({str(row["canonical_record_id"]) for row in selected}) == count, "preflight geometry repeats a record")
    return selected


def select_cache_online_alignment_sequences_v4(
    rows: Sequence[Mapping[str, Any]],
    cache_payload: Mapping[str, Any],
    *,
    count: int,
) -> dict[int, str]:
    """Select fixed length-stratified sequence indices without reading a target."""

    _require(count >= 2, "cache/online alignment requires at least two sequences")
    _require(
        all(str(row.get("split")) in {"TRAIN", "VALIDATION"} for row in rows),
        "protected split entered cache/online alignment",
    )
    sequences = sorted(
        {
            str(sequence)
            for row in rows
            for sequence in (row["source_sequence"], row["candidate_sequence"])
        }
    )
    _require(len(sequences) == int(cache_payload["sequence_lengths"].numel()), "alignment projection/cache sequence count changed")
    _require(len(sequences) >= count, "alignment cohort exceeds unique sequences")
    by_length = sorted(range(len(sequences)), key=lambda index: (len(sequences[index]), sequences[index]))
    ranks = [round(index * (len(by_length) - 1) / (count - 1)) for index in range(count)]
    _require(len(set(ranks)) == count, "alignment quantile ranks are duplicated")
    selected_indices = [by_length[rank] for rank in ranks]
    selected = {index: sequences[index] for index in selected_indices}
    _require(
        all(
            int(cache_payload["sequence_lengths"][index].item()) == len(sequence)
            for index, sequence in selected.items()
        ),
        "alignment sequence length differs from cache",
    )
    return selected


def cached_bottom_sequences_v4(
    cache_payload: Mapping[str, Any], sequence_indices: Sequence[int]
) -> dict[int, BottomEncodedSequenceV4]:
    """Reconstruct selected cached sequences without adding raw sequence payload."""

    result: dict[int, BottomEncodedSequenceV4] = {}
    sequence_count = int(cache_payload["sequence_lengths"].numel())
    for sequence_index in sequence_indices:
        _require(0 <= int(sequence_index) < sequence_count, "alignment sequence index is out of range")
        chunk_start = int(cache_payload["sequence_chunk_offsets"][sequence_index].item())
        chunk_end = int(cache_payload["sequence_chunk_offsets"][sequence_index + 1].item())
        chunks: list[BottomEncodedChunkV4] = []
        for chunk_index in range(chunk_start, chunk_end):
            token_start = int(cache_payload["chunk_token_offsets"][chunk_index].item())
            token_end = int(cache_payload["chunk_token_offsets"][chunk_index + 1].item())
            chunks.append(
                BottomEncodedChunkV4(
                    span=ChunkSpan(
                        int(cache_payload["chunk_starts"][chunk_index].item()),
                        int(cache_payload["chunk_ends"][chunk_index].item()),
                    ),
                    hidden=cache_payload["token_hidden"][token_start:token_end],
                    attention_mask=cache_payload["token_attention_mask"][token_start:token_end],
                    special_token_offset=int(
                        cache_payload["chunk_special_token_offsets"][chunk_index].item()
                    ),
                )
            )
        _require(bool(chunks), "alignment cached sequence has no chunks")
        result[int(sequence_index)] = BottomEncodedSequenceV4(
            chunks=tuple(chunks),
            global_residual=cache_payload["sequence_global_residuals"][sequence_index],
        )
    return result


def run_cache_online_alignment_v4(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    cache_payload: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Run the frozen CUDA cache/online equivalence check before memory preflight."""

    settings = config["cache_online_alignment"]
    selected = select_cache_online_alignment_sequences_v4(
        rows,
        cache_payload,
        count=int(settings["sequence_count"]),
    )
    _require(
        str(cache_payload["attention_backend"]) == str(settings["attention_backend"]),
        "cache/online attention backend changed",
    )
    cached = cached_bottom_sequences_v4(cache_payload, sorted(selected))
    encoder = FrozenMRNABERTBottomSixEncoderV4(
        Path(config["mrnabert_model_path"]),
        device,
        maximum_sequences_per_batch=int(settings["maximum_sequences_per_batch"]),
        batch_token_budget=int(settings["batch_token_budget"]),
        attention_backend=str(settings["attention_backend"]),
    )
    online = encoder.encode_online(selected)
    comparison = compare_bottom_encoded_sequences_v4(
        cached,
        online,
        maximum_absolute_tolerance=float(settings["maximum_absolute_tolerance"]),
        mean_absolute_tolerance=float(settings["mean_absolute_tolerance"]),
    )
    result = {
        **comparison,
        "selection": "LENGTH_SORTED_EVEN_QUANTILES_OVER_LEXICOGRAPHIC_CACHE_SEQUENCE_INDICES",
        "selected_sequence_indices": sorted(selected),
        "selected_sequence_lengths": [len(selected[index]) for index in sorted(selected)],
        "raw_sequence_payload_written": 0,
        "target_value_accessed": False,
        "validation_metric_read": False,
    }
    del encoder, online, cached
    torch.cuda.empty_cache()
    return result


def build_preflight_vocabs_v4(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    """Reproduce formal endpoint vocab geometry without indexing a target."""

    _require(bool(rows), "preflight vocab rows are empty")
    values: dict[str, set[str]] = {
        key: set()
        for key in (
            "study",
            "assay",
            "context",
            "quantity",
            "measurement",
            "numerator",
            "denominator",
        )
    }
    for row in rows:
        _require(str(row.get("split")) in {"TRAIN", "VALIDATION"}, "protected split entered preflight vocab")
        descriptor = row["endpoint_descriptor"]
        values["study"].add(str(row["study_unit_id"]))
        values["assay"].add(str(row["assay_id"]))
        values["context"].add(str(row["biological_context_id"]))
        values["quantity"].add(str(descriptor["quantity_family"]))
        values["measurement"].add(str(descriptor["measurement_form"]))
        values["numerator"].add(
            "__NONE__"
            if descriptor["numerator_family"] is None
            else str(descriptor["numerator_family"])
        )
        values["denominator"].add(
            "__NONE__"
            if descriptor["denominator_family"] is None
            else str(descriptor["denominator_family"])
        )
    return {
        field: {"__UNK__": 0}
        | {
            value: index + 1
            for index, value in enumerate(sorted(field_values))
        }
        for field, field_values in values.items()
    }


def preflight_example_v4(
    row: Mapping[str, Any],
    vocabs: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    """Convert only sequence/edit/descriptor geometry; target keys are ignored."""

    source = str(row["source_sequence"])
    candidate = str(row["candidate_sequence"])
    descriptor = row["endpoint_descriptor"]
    edits = tuple(
        (
            int(edit["position"]),
            str(edit["source_base"]),
            str(edit["candidate_base"]),
        )
        for edit in row["source_relative_edits"]
    )
    category = lambda value: "__NONE__" if value is None else str(value)
    return {
        "record_id": str(row["canonical_record_id"]),
        "cache_record_id": str(row["canonical_record_id"]),
        "source_group": str(row["source_group_id"]),
        "task": str(row["task_id"]),
        "source": torch.tensor([RNA_TOKEN[base] for base in source], dtype=torch.long),
        "candidate": torch.tensor([RNA_TOKEN[base] for base in candidate], dtype=torch.long),
        "edits": edits,
        # Structural placeholders satisfy the common collator only.  They are
        # never read by the preflight loss or used as biological targets.
        "target": 0.0,
        "scaled_target": 0.0,
        "target_scale": 1.0,
        "sample_weight": 1.0,
        "study": vocabs["study"].get(str(row["study_unit_id"]), 0),
        "assay": vocabs["assay"].get(str(row["assay_id"]), 0),
        "context": vocabs["context"].get(str(row["biological_context_id"]), 0),
        "quantity": vocabs["quantity"].get(str(descriptor["quantity_family"]), 0),
        "measurement": vocabs["measurement"].get(str(descriptor["measurement_form"]), 0),
        "numerator": vocabs["numerator"].get(category(descriptor["numerator_family"]), 0),
        "denominator": vocabs["denominator"].get(category(descriptor["denominator_family"]), 0),
        "region": int(row["region_id"]),
    }


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _build_model(
    config: Mapping[str, Any],
    vocabs: Mapping[str, Mapping[str, int]],
    *,
    device: torch.device,
) -> XEditCriticV4:
    architecture = config["architecture"]
    upper = TrainableMRNABERTUpperSixEncoderV4(
        Path(config["mrnabert_model_path"]),
        device,
        attention_backend=str(config["memory_preflight"]["attention_backend"]),
        activation_checkpointing=True,
    )
    return XEditCriticV4(
        upper_encoder=upper,
        study_count=len(vocabs["study"]),
        assay_count=len(vocabs["assay"]),
        context_count=len(vocabs["context"]),
        quantity_count=len(vocabs["quantity"]),
        measurement_count=len(vocabs["measurement"]),
        numerator_count=len(vocabs["numerator"]),
        denominator_count=len(vocabs["denominator"]),
        region_count=2,
        control_mode="NONE",
        mechanism_mode="FULL",
        pretrained_width=int(architecture["pretrained_width"]),
        model_width=int(architecture["model_width"]),
        block_count=int(architecture["edit_block_count"]),
        heads=int(architecture["attention_heads"]),
        ffn_width=int(architecture["ffn_width"]),
        expert_count=int(architecture["semantic_expert_count"]),
        expert_bottleneck_width=int(architecture["semantic_expert_bottleneck_width"]),
        expert_top_k=int(architecture["semantic_router_top_k"]),
        raw_hidden_dim=int(architecture["raw_hidden_dim"]),
        raw_depth=int(architecture["raw_depth"]),
        readout_hidden_width=int(architecture["readout_hidden_width"]),
        dropout=float(architecture["dropout"]),
        minimum_physical_batch=4,
        activation_checkpointing=True,
    ).to(device)


def _measure_one_batch(
    config: Mapping[str, Any],
    vocabs: Mapping[str, Mapping[str, int]],
    examples: Sequence[Mapping[str, Any]],
    collator: XEditCriticCollatorV4,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    _set_seed(int(config["training"]["screen_seed"]))
    torch.cuda.empty_cache()
    model = _build_model(config, vocabs, device=device)
    capacity = require_v4_trainable_parameter_range(model)
    rates = config["training"]["learning_rates"]
    optimizer = torch.optim.AdamW(
        critic_v4_optimizer_parameter_groups(
            model,
            head_learning_rate=float(rates["new_head_and_v4_trunk"]),
            semantic_learning_rate=float(rates["semantic_experts_and_router"]),
            upper_six_learning_rate=float(rates["mrnabert_top_six"]),
        ),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    batch = _move(collator(list(examples[:batch_size])), device)
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(batch)
        # Geometry-only scalar: it allocates the authentic forward/backward and
        # AdamW state without reading a target or Validation metric.
        geometry_loss = output["mean"].square().mean() + 0.01 * output["router_balance_loss"]
    _require(torch.isfinite(geometry_loss).item(), "preflight geometry loss is nonfinite")
    geometry_loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    _require(torch.isfinite(gradient_norm).item(), "preflight gradient norm is nonfinite")
    optimizer.step()
    torch.cuda.synchronize(device)
    peak_bytes = int(torch.cuda.max_memory_allocated(device))
    result = {
        "batch_size": batch_size,
        "peak_allocated_bytes": peak_bytes,
        "peak_allocated_gib": peak_bytes / 1024**3,
        "trainable_parameter_count": int(capacity["trainable_parameter_count"]),
        "module_counts": capacity["module_counts"],
        "cuda_device_name": torch.cuda.get_device_name(device),
        "forward_precision": "BF16",
        "activation_checkpointing": True,
        "optimizer_state_materialized": True,
        "target_value_accessed": False,
        "validation_metric_read": False,
        "passed_runtime": True,
    }
    del batch, optimizer, model, geometry_loss
    torch.cuda.empty_cache()
    return result


def run(
    config: Mapping[str, Any],
    *,
    physical_gpu_index: int,
    authorization_path: Path,
) -> dict[str, Any]:
    output_path = Path(config["preflight_output"])
    _require(not output_path.exists(), "Critic V4 preflight artifact already exists")
    partial_output = output_path.with_suffix(output_path.suffix + ".partial")
    _require(not partial_output.exists(), "partial Critic V4 preflight artifact already exists")
    current_head = _git_head()
    authorization = _load_json(authorization_path)
    require_preflight_authorization_v4(
        authorization, current_git_head=current_head
    )
    require_physical_gpu_scope_v4(config, physical_gpu_index)
    device = require_cuda(physical_gpu_index)
    started = time.time()
    projection_rows = load_projection_rows(
        [Path(path) for path in config["projection_paths"]]
    )
    # The frozen projection parser is TRAIN/VALIDATION-only.  Preflight uses
    # both splits solely to instantiate the final outcome-free vocab geometry;
    # selected memory rows come exclusively from TRAIN and target fields are
    # never indexed below.
    vocabs = build_preflight_vocabs_v4(projection_rows)
    selected_rows = select_train_geometry_records_v4(projection_rows, count=32)
    selected_ids = [str(row["canonical_record_id"]) for row in selected_rows]
    cache_payload = load_frozen_bottom_encoder_chunk_cache_v4(
        Path(config["bottom_six_cache"])
    )
    cache_identity = require_frozen_bottom_encoder_chunk_cache_identity_v4(
        cache_payload,
        expected_model_id=str(config["model_id"]),
        expected_record_count=int(config["data_geometry"]["expected_record_count"]),
        expected_unique_sequence_count=43730,
        expected_embedding_width=int(config["architecture"]["pretrained_width"]),
    )
    cache_online_alignment = run_cache_online_alignment_v4(
        config,
        projection_rows,
        cache_payload,
        device=device,
    )
    if cache_online_alignment["passed"] is not True:
        summary = {
            "schema_version": "route_a_v3_route2_xeditcritic_v4_preflight.v1",
            "status": "XEDITCRITIC_V4_PREFLIGHT_PAUSE_CACHE_ONLINE_MISMATCH",
            "passed": False,
            "git_head": current_head,
            "physical_gpu_index": physical_gpu_index,
            "cuda_device_name": torch.cuda.get_device_name(device),
            "bottom_six_cache_identity": cache_identity,
            "cache_online_alignment": cache_online_alignment,
            "target_value_accessed": False,
            "validation_metric_read": False,
            "memory_preflight_executed": False,
            "cpu_fallback_used": False,
            "elapsed_seconds": time.time() - started,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
            "authorization_path": str(authorization_path),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(partial_output, output_path)
        return summary
    cache = FrozenBottomEncoderChunkCacheViewV4(
        cache_payload, set(str(value) for value in cache_payload["record_ids"])
    )
    collator = XEditCriticCollatorV4(cache, minimum_physical_batch=4)
    examples = [preflight_example_v4(row, vocabs) for row in selected_rows]
    parameter_model = _build_model(config, vocabs, device=device)
    formal_capacity = require_v4_trainable_parameter_range(parameter_model)
    parameter_count = int(formal_capacity["trainable_parameter_count"])
    module_counts = formal_capacity["module_counts"]
    del parameter_model
    torch.cuda.empty_cache()
    measurements: dict[int, dict[str, Any] | None] = {}
    failure_messages: dict[int, str] = {}
    for batch_size in config["memory_preflight"]["physical_batch_candidates"]:
        try:
            measurements[int(batch_size)] = _measure_one_batch(
                config,
                vocabs,
                examples,
                collator,
                batch_size=int(batch_size),
                device=device,
            )
        except torch.cuda.OutOfMemoryError as exc:
            measurements[int(batch_size)] = None
            failure_messages[int(batch_size)] = f"{type(exc).__name__}: {exc}"
            torch.cuda.empty_cache()
    peak_map = {
        batch_size: None
        if measurement is None
        else float(measurement["peak_allocated_gib"])
        for batch_size, measurement in measurements.items()
    }
    measured_parameter_counts = {
        int(measurement["trainable_parameter_count"])
        for measurement in measurements.values()
        if measurement is not None
    }
    _require(
        not measured_parameter_counts
        or measured_parameter_counts == {parameter_count},
        "formal parameter count differs across batch preflights",
    )
    try:
        minimum_peak_gib = config["memory_preflight"][
            "minimum_peak_allocated_gib"
        ]
        _require(
            minimum_peak_gib is None,
            "Critic V4 lower memory-occupancy gate is not disabled",
        )
        selection = select_physical_batch_from_memory_v4(
            peak_map,
            minimum_peak_gib=None,
            maximum_peak_gib=float(config["memory_preflight"]["maximum_peak_allocated_gib"]),
        )
        passed = True
        status = "XEDITCRITIC_V4_PREFLIGHT_PASS"
        selection_error = None
    except XEditCriticTrainingV4Error as exc:
        passed = False
        status = "XEDITCRITIC_V4_PREFLIGHT_PAUSE"
        selection = {
            "selected_physical_batch": None,
            "selected_peak_allocated_gib": None,
        }
        selection_error = {
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    summary = {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_preflight.v1",
        "status": status,
        "passed": passed,
        "git_head": current_head,
        "physical_gpu_index": physical_gpu_index,
        "cuda_device_name": torch.cuda.get_device_name(device),
        "geometry_selection": "TRAIN_ONLY_DESCENDING_EDIT_COUNT_THEN_SEQUENCE_LENGTH_THEN_RECORD_ID",
        "selected_train_record_ids": selected_ids,
        "selected_train_record_count": len(selected_ids),
        "bottom_six_cache_identity": cache_identity,
        "cache_online_alignment": cache_online_alignment,
        "target_value_accessed": False,
        "validation_metric_read": False,
        "measurements": {
            str(batch): value for batch, value in sorted(measurements.items())
        },
        "runtime_failures": {
            str(batch): value for batch, value in sorted(failure_messages.items())
        },
        "trainable_parameter_count": parameter_count,
        "module_counts": module_counts,
        "selected_physical_batch": selection["selected_physical_batch"],
        "selected_peak_allocated_gib": selection["selected_peak_allocated_gib"],
        "selection_error": selection_error,
        "memory_measurement": "TORCH_CUDA_MAX_MEMORY_ALLOCATED",
        "artificial_padding_or_unused_tensor": False,
        "cpu_fallback_used": False,
        "elapsed_seconds": time.time() - started,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
        "authorization_path": str(authorization_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial_output, output_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--authorization", required=True, type=Path)
    arguments = parser.parse_args()
    config = _load_json(arguments.config)
    print(
        json.dumps(
            run(
                config,
                physical_gpu_index=arguments.physical_gpu_index,
                authorization_path=arguments.authorization,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
