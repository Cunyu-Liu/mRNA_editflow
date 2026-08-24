#!/usr/bin/env python3
"""Run the TRAIN-only source-data, capacity, BF16 batch, and memory preflight."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_development_projection_v3 import load_projection_rows
from core.route2_source_token_cache_v3 import (
    SourceTokenCacheIndexV3,
    load_source_token_cache_v3,
    require_source_token_cache_identity_v3,
)
from core.route2_xeditsetflow_runtime_v4 import build_setflow_screen_model_v4
from core.route2_xeditsetflow_training_v4 import (
    SetFlowSourceStateDatasetV4,
    collate_setflow_source_states_v4,
    setflow_source_records_from_projection_rows_v4,
    setflow_source_vocabs_v4,
)
from core.route2_xeditsetflow_v4 import mixture_setflow_loss_v4


class SetFlowPreflightV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SetFlowPreflightV4Error(message)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require_preflight_authorization_v4(
    authorization: Mapping[str, Any], *, current_git_head: str
) -> None:
    _require(
        authorization.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_preflight_authorization.v1",
        "SetFlow V4 preflight authorization schema is absent",
    )
    _require(
        authorization.get("status") == "XEDITSETFLOW_V4_PREFLIGHT_AUTHORIZED",
        "SetFlow V4 preflight is not authorized",
    )
    _require(
        str(authorization.get("authorized_git_head")) == str(current_git_head),
        "SetFlow V4 preflight authorization is for another Git HEAD",
    )
    barriers = authorization.get("barriers", {})
    required = (
        "all_five_c3_jobs_terminal",
        "c3_terminal_summaries_read_exactly_once",
        "a100_current_head_focused_tests_passed",
        "a100_current_head_v332_tests_passed",
        "source_token_cache_terminal_complete",
    )
    _require(
        all(barriers.get(key) is True for key in required),
        "a SetFlow V4 preflight barrier is not satisfied",
    )
    _require(
        int(authorization.get("development_test_outcome_reads", -1)) == 0,
        "SetFlow V4 preflight authorization reports a Development TEST read",
    )
    _require(
        int(authorization.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "SetFlow V4 preflight authorization reports a new Evaluation read",
    )


def select_train_geometry_sources_v4(records: Sequence[Any]) -> list[int]:
    """Select eight TRAIN sources by outcome-free length/edit geometry only."""

    _require(len(records) >= 8, "SetFlow V4 preflight has fewer than eight TRAIN sources")
    ranked = sorted(
        range(len(records)),
        key=lambda index: (
            -len(records[index].source),
            -max(len(edits) for edits in records[index].terminal_edit_sets),
            records[index].source_id,
        ),
    )
    return ranked[:8]


def _move(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def run_preflight(
    config: Mapping[str, Any],
    authorization: Mapping[str, Any],
    *,
    physical_gpu_index: int,
) -> dict[str, Any]:
    current_head = _git_head()
    require_preflight_authorization_v4(
        authorization, current_git_head=current_head
    )
    output_path = Path(config["preflight_output_path"])
    data_audit_path = Path(config["source_level_data_audit_path"])
    package_root = output_path.parent
    staging_root = package_root.with_name(package_root.name + ".partial")
    _require(data_audit_path.parent == package_root, "SetFlow V4 preflight and source audit must share one package root")
    _require(not package_root.exists(), "SetFlow V4 preflight package already exists")
    _require(not staging_root.exists(), "partial SetFlow V4 preflight package already exists")
    _require(not output_path.exists(), "SetFlow V4 preflight is already terminal")
    _require(not data_audit_path.exists(), "SetFlow V4 source data audit already exists")
    _require(
        physical_gpu_index in config["gpu_policy"]["physical_gpu_scope"],
        "SetFlow V4 preflight GPU is outside 0–5",
    )
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    _require(
        not os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES remapping is forbidden",
    )
    device = torch.device(f"cuda:{physical_gpu_index}")
    torch.cuda.set_device(device)
    _require(torch.cuda.is_bf16_supported(), "BF16 is unavailable on selected GPU")

    train_rows = load_projection_rows(
        [Path(config["train_projection_path"])], allowed_splits=("TRAIN",)
    )
    validation_rows = load_projection_rows(
        [Path(config["validation_projection_path"])],
        allowed_splits=("VALIDATION",),
    )
    geometry = config["data_geometry"]
    _require(
        len(train_rows)
        == int(geometry["expected_train_projection_candidate_row_count"]),
        "SetFlow V4 TRAIN projection count changed",
    )
    _require(
        len(validation_rows)
        == int(geometry["expected_validation_projection_candidate_row_count"]),
        "SetFlow V4 Validation projection count changed",
    )
    train_records, train_inventory = setflow_source_records_from_projection_rows_v4(
        train_rows
    )
    validation_records, validation_inventory = (
        setflow_source_records_from_projection_rows_v4(validation_rows)
    )
    _require(
        len(validation_records)
        == int(geometry["eligible_validation_source_count"]),
        "SetFlow V4 eligible Validation source count changed",
    )
    vocabs = setflow_source_vocabs_v4(train_records)
    expected_vocab = config["architecture"]["formal_endpoint_vocab_cardinalities"]
    _require(
        {field: len(vocab) for field, vocab in vocabs.items()} == expected_vocab,
        "SetFlow V4 outcome-free endpoint vocabulary changed",
    )
    cache_payload = load_source_token_cache_v3(Path(config["source_token_cache_path"]))
    source_token_cache_identity = require_source_token_cache_identity_v3(
        cache_payload,
        expected_model_id="YYLY66/mRNABERT@a1eb7df25804d23f08646e1cb996b234d7208a40",
        expected_record_count=84218,
        expected_unique_source_count=19303,
        expected_token_count=2817781,
        expected_maximum_source_length=837,
        expected_embedding_width=int(
            config["architecture"]["frozen_source_mrnabert_width"]
        ),
    )
    cache = SourceTokenCacheIndexV3(cache_payload)
    for record in [*train_records, *validation_records]:
        _require(
            record.cache_record_id in cache.record_to_row,
            "SetFlow V4 source record is absent from source-token cache",
        )
        _require(
            cache.tokens_for_record(record.cache_record_id).shape[0]
            == len(record.source),
            "SetFlow V4 source-token cache length does not align",
        )

    single, single_capacity = build_setflow_screen_model_v4(
        config, vocabs, run_id="v4_single_mode"
    )
    del single
    model, full_capacity = build_setflow_screen_model_v4(
        config, vocabs, run_id="v4_full"
    )
    dataset = SetFlowSourceStateDatasetV4(
        train_records,
        vocabs,
        seed=int(config["training"]["screen_seed"]),
    )
    dataset.set_pass(0)
    selected_indices = select_train_geometry_sources_v4(train_records)
    examples = [
        dataset.state(source_index, state_slot)
        for source_index in selected_indices
        for state_slot in range(4)
    ]
    batch = _move(
        collate_setflow_source_states_v4(examples, source_cache=cache), device
    )
    _require(batch["source_tokens"].shape[0] == 32, "SetFlow V4 preflight batch is not 32")
    torch.manual_seed(int(config["training"]["screen_seed"]))
    torch.cuda.manual_seed_all(int(config["training"]["screen_seed"]))
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        fused=True,
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(batch)
        loss = mixture_setflow_loss_v4(
            output,
            batch,
            coverage_weight=float(config["objective"]["source_candidate_coverage_weight"]),
            remaining_count_weight=float(config["objective"]["remaining_count_weight"]),
            mode_information_weight=float(config["objective"]["full_mode_information_weight"]),
        ).total
    _require(loss.is_cuda and torch.isfinite(loss).item(), "SetFlow V4 preflight loss is invalid")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(config["training"]["gradient_clip_norm"])
    )
    _require(torch.isfinite(gradient_norm).item(), "SetFlow V4 preflight gradient is nonfinite")
    optimizer.step()
    torch.cuda.synchronize(device)
    peak_bytes = int(torch.cuda.max_memory_allocated(device))
    data_audit = {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_source_level_data_audit.v1",
        "status": "XEDITSETFLOW_V4_SOURCE_LEVEL_DATA_AUDIT_PASS",
        "train_projection_candidate_row_count": len(train_rows),
        "validation_projection_candidate_row_count": len(validation_rows),
        "train_source_count": len(train_records),
        "validation_source_count": len(validation_records),
        "train_inventory": train_inventory,
        "validation_inventory": validation_inventory,
        "endpoint_vocab_cardinalities": {
            field: len(vocab) for field, vocab in vocabs.items()
        },
        "source_cache_record_count": len(cache_payload["record_ids"]),
        "source_token_cache_identity": source_token_cache_identity,
        "source_cache_raw_sequence_payload_written": int(
            cache_payload["raw_sequence_payload_written"]
        ),
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    result = {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_preflight.v1",
        "status": "XEDITSETFLOW_V4_PREFLIGHT_PASS",
        "passed": True,
        "git_head": current_head,
        "physical_gpu_index": physical_gpu_index,
        "torch_device": str(device),
        "precision": "BF16",
        "cpu_fallback_used": False,
        "full_trainable_parameter_count": int(
            full_capacity["trainable_parameter_count"]
        ),
        "single_mode_trainable_parameter_count": int(
            single_capacity["trainable_parameter_count"]
        ),
        "physical_and_effective_batch_size": 32,
        "source_token_cache_identity": source_token_cache_identity,
        "selected_train_source_indices": selected_indices,
        "selection_uses_outcome_free_geometry_only": True,
        "optimizer_state_materialized": True,
        "peak_memory_allocated_bytes": peak_bytes,
        "peak_memory_allocated_gib": peak_bytes / 1024**3,
        "elapsed_seconds": time.time() - started,
        "source_data_audit_path": str(data_audit_path),
        "validation_metric_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    staging_root.mkdir(parents=True)
    (staging_root / data_audit_path.name).write_text(
        json.dumps(data_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (staging_root / output_path.name).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(staging_root, package_root)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    authorization = json.loads(
        arguments.authorization.read_text(encoding="utf-8")
    )
    result = run_preflight(
        config,
        authorization,
        physical_gpu_index=arguments.physical_gpu_index,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
