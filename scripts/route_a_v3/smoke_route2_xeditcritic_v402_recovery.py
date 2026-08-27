#!/usr/bin/env python3
"""Run one target-free TRAIN sampler/CUDA smoke for Critic V4.0.2 recovery."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_bottom_encoder_chunk_cache_v4 import (
    load_frozen_bottom_encoder_chunk_cache_v4,
)
from core.route2_development_projection_v3 import load_projection_rows
from core.route2_xeditcritic_batch_v4 import (
    FrozenBottomEncoderChunkCacheViewV4,
    XEditCriticCollatorV4,
)
from core.route2_xeditcritic_training_data_v3 import XEditCriticRecordV3
from core.route2_xeditcritic_training_v4 import (
    FixedEffectiveTaskBatchSamplerV4,
    require_physical_gpu_scope_v4,
)
from scripts.route_a_v3.preflight_route2_xeditcritic_v4 import (
    _measure_one_batch,
    build_preflight_vocabs_v4,
    preflight_example_v4,
)
from scripts.route_a_v3.train_route2_xeditcritic_v3 import require_cuda


class V402RecoverySmokeError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V402RecoverySmokeError(message)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def sampler_records_without_targets_v402(
    rows: Sequence[Mapping[str, Any]],
) -> list[XEditCriticRecordV3]:
    """Build only the metadata consumed by the sampler; never index a target."""

    records = []
    for row in rows:
        if str(row.get("split")) != "TRAIN":
            continue
        descriptor = row["endpoint_descriptor"]
        category = lambda value: "__NONE__" if value is None else str(value)
        records.append(
            XEditCriticRecordV3(
                record_id=str(row["canonical_record_id"]),
                split="TRAIN",
                source="",
                candidate="",
                edits=(),
                target=0.0,
                task=str(row["task_id"]),
                study=str(row["study_unit_id"]),
                source_group=str(row["source_group_id"]),
                assay=str(row["assay_id"]),
                context=str(row["biological_context_id"]),
                region=int(row["region_id"]),
                quantity=str(descriptor["quantity_family"]),
                measurement=str(descriptor["measurement_form"]),
                numerator=category(descriptor["numerator_family"]),
                denominator=category(descriptor["denominator_family"]),
            )
        )
    _require(bool(records), "V4.0.2 TRAIN sampler records are empty")
    return sorted(records, key=lambda record: record.record_id)


def run(
    config: Mapping[str, Any],
    *,
    expected_head: str,
    physical_gpu_index: int,
    output: Path,
) -> dict[str, Any]:
    _require(_git_head() == expected_head, "V4.0.2 smoke Git HEAD differs")
    _require(not output.exists(), f"V4.0.2 smoke already exists: {output}")
    partial = output.with_suffix(output.suffix + ".partial")
    _require(not partial.exists(), f"V4.0.2 smoke partial exists: {partial}")
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden")
    require_physical_gpu_scope_v4(config, physical_gpu_index)

    rows = load_projection_rows([Path(path) for path in config["projection_paths"]])
    records = sampler_records_without_targets_v402(rows)
    geometry = config["data_geometry"]
    _require(len(records) == int(geometry["expected_train_count"]), "TRAIN count changed")
    sampler = FixedEffectiveTaskBatchSamplerV4(
        records,
        seed=int(config["training"]["screen_seed"]),
        repeat_cap=int(geometry["maximum_record_repeats_per_pass"]),
        effective_batch=int(geometry["effective_batch_size"]),
    )
    first_batch: list[int] | None = None
    pass_rows: list[dict[str, Any]] = []
    for pass_index in range(int(geometry["pass_count"])):
        sampler.set_pass(pass_index)
        batches = sampler.batches_for_pass()
        _require(len(batches) == int(geometry["updates_per_pass"]), "updates/pass changed")
        _require(all(len(batch) == 32 for batch in batches), "effective batch is not 32")
        _require(
            all(len({records[index].task for index in batch}) == 1 for batch in batches),
            "task-homogeneous sampler mixed tasks",
        )
        counts = Counter(index for batch in batches for index in batch)
        _require(max(counts.values()) <= 4, "record repeat cap exceeded")
        if first_batch is None:
            first_batch = batches[0]
        pass_rows.append(
            {
                "pass_number": pass_index + 1,
                "update_count": len(batches),
                "draw_count": sum(counts.values()),
                "maximum_record_repeat": max(counts.values()),
            }
        )
    _require(first_batch is not None and len(first_batch) == 32, "smoke batch is absent")

    device = require_cuda(physical_gpu_index)
    preflight = _load(Path(config["preflight_output"]))
    _require(preflight.get("status") == "XEDITCRITIC_V4_PREFLIGHT_PASS", "formal preflight is not PASS")
    required_free_bytes = math.ceil(
        (float(preflight["selected_peak_allocated_gib"]) + 2.0) * 1024**3
    )
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    _require(free_bytes >= required_free_bytes, "GPU5 free memory is below measured peak plus 2 GiB")

    row_by_id = {str(row["canonical_record_id"]): row for row in rows}
    selected_rows = [row_by_id[records[index].record_id] for index in first_batch]
    vocabs = build_preflight_vocabs_v4(rows)
    cache_payload = load_frozen_bottom_encoder_chunk_cache_v4(
        Path(config["bottom_six_cache"])
    )
    cache = FrozenBottomEncoderChunkCacheViewV4(
        cache_payload,
        set(str(value) for value in cache_payload["record_ids"]),
        validate_payload=False,
    )
    collator = XEditCriticCollatorV4(cache, minimum_physical_batch=4)
    examples = [preflight_example_v4(row, vocabs) for row in selected_rows]
    measurement = _measure_one_batch(
        config,
        vocabs,
        examples,
        collator,
        batch_size=32,
        device=device,
    )
    _require(measurement["passed_runtime"] is True, "CUDA smoke did not pass")
    _require(measurement["forward_precision"] == "BF16", "CUDA smoke is not BF16")
    _require(measurement["target_value_accessed"] is False, "CUDA smoke accessed a target")
    _require(measurement["validation_metric_read"] is False, "CUDA smoke read Validation metrics")
    _require(
        int(measurement["trainable_parameter_count"])
        == int(preflight["trainable_parameter_count"]),
        "CUDA smoke parameter count differs from formal preflight",
    )
    result = {
        "schema_version": "route_a_v3_route2_xeditcritic_v402_recovery_smoke.v1",
        "status": "XEDITCRITIC_V402_RECOVERY_TRAIN_ONLY_CUDA_SMOKE_PASS",
        "git_head": expected_head,
        "physical_gpu_index": physical_gpu_index,
        "train_record_count": len(records),
        "target_value_accessed": False,
        "validation_metric_read": False,
        "sampler_passes": pass_rows,
        "task_batch_allocations": sampler.task_batch_allocations,
        "cuda_measurement": measurement,
        "launch_free_memory_bytes": int(free_bytes),
        "required_free_memory_bytes": int(required_free_bytes),
        "cpu_fallback_used": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                _load(arguments.config),
                expected_head=arguments.expected_head,
                physical_gpu_index=arguments.physical_gpu_index,
                output=arguments.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
