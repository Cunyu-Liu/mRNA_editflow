#!/usr/bin/env python3
"""Benchmark Route 2 host-side batch construction without model updates."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.route_a_v3 import train_route2_delta_predictor_v1 as trainer


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def benchmark_profile(
    dataset: trainer.DeltaDataset,
    records: list[trainer.DeltaRecord],
    profile: dict[str, Any],
    *,
    seed: int,
    warmup_batches: int,
    measured_batches: int,
) -> dict[str, Any]:
    batch_size = int(profile["batch_size"])
    sampler = trainer.LengthBucketBatchSampler(records, batch_size, seed, True)
    options = trainer.data_loader_options(profile)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=trainer.collate,
        **options,
    )
    iterator = iter(loader)
    for _ in range(warmup_batches):
        try:
            next(iterator)
        except StopIteration:
            iterator = iter(loader)
            next(iterator)

    started = time.perf_counter()
    batch_count = 0
    record_count = 0
    padded_token_count = 0
    maximum_lengths = []
    while batch_count < measured_batches:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch_count += 1
        records_in_batch = int(batch["target"].shape[0])
        maximum_length = int(batch["source_tokens"].shape[1])
        record_count += records_in_batch
        padded_token_count += records_in_batch * maximum_length * 2
        maximum_lengths.append(maximum_length)
    elapsed = time.perf_counter() - started
    _require(elapsed > 0, "invalid benchmark duration")
    return {
        "profile_id": str(profile["profile_id"]),
        "batch_size": batch_size,
        "num_workers": options["num_workers"],
        "pin_memory": options["pin_memory"],
        "persistent_workers": options["persistent_workers"],
        "prefetch_factor": options.get("prefetch_factor"),
        "non_blocking_transfer": bool(profile.get("non_blocking_transfer", False)),
        "warmup_batches": warmup_batches,
        "measured_batches": batch_count,
        "measured_records": record_count,
        "elapsed_seconds": elapsed,
        "batches_per_second": batch_count / elapsed,
        "records_per_second": record_count / elapsed,
        "padded_source_candidate_tokens_per_second": padded_token_count / elapsed,
        "mean_batch_maximum_length": sum(maximum_lengths) / len(maximum_lengths),
        "maximum_batch_length": max(maximum_lengths),
        "scientific_metrics_computed": False,
        "model_parameter_updates": 0,
        "evaluation_pool_records_read": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    benchmark_config = json.loads(args.config.read_text(encoding="utf-8"))
    training_config = json.loads(
        Path(benchmark_config["training_config"]).read_text(encoding="utf-8")
    )

    manifest = trainer.load_manifest(Path(training_config["development_manifest"]))
    records = trainer.load_records(
        [Path(path) for path in training_config["canonical_paths"]], manifest
    )
    pretrained = trainer.FrozenPretrainedPairFeatures(
        Path(training_config["pretrained_feature_cache_path"]),
        {row.record_id for row in records},
    )
    records, included_studies, _excluded = trainer.select_study_subset(
        records, training_config.get("included_study_unit_ids")
    )
    records, included_regions, _region_excluded = trainer.select_region_subset(
        records, training_config.get("included_regions")
    )
    by_split, withheld = trainer.fixed_split_records(
        records, "HPO_VALIDATION_ONLY"
    )
    train_records = by_split["TRAIN"]
    target_scaler = trainer.fit_route2_target_scaler(
        train_records,
        mode=str(training_config.get("target_scaling_mode", trainer.TARGET_SCALING_NONE)),
        minimum_task_records=int(training_config.get("target_scale_minimum_task_records", 20)),
        floor=float(training_config.get("target_scale_floor", 1e-3)),
    )
    metadata_mode = str(training_config.get("metadata_mode", "FULL_CONTEXT"))
    if metadata_mode == "FULL_CONTEXT":
        vocabs = {
            field: trainer.build_vocab(train_records, field)
            for field in ("study", "assay", "context", "endpoint")
        }
    elif metadata_mode == "TRANSFERABLE_CONTEXT":
        vocabs = {
            "study": {"__UNK__": 0},
            **{
                field: trainer.build_vocab(train_records, field)
                for field in ("assay", "context", "endpoint")
            },
        }
    else:
        vocabs = {
            field: {"__UNK__": 0}
            for field in ("study", "assay", "context", "endpoint")
        }
    dataset = trainer.DeltaDataset(
        train_records,
        vocabs,
        metadata_mode=metadata_mode,
        weighting_mode=str(
            training_config.get(
                "training_weighting_mode", "SOURCE_CONTEXT_ENDPOINT_GROUP"
            )
        ),
        target_scaler=target_scaler,
        pretrained_features=pretrained,
    )
    profiles = benchmark_config["profiles"]
    _require(bool(profiles), "benchmark profiles are empty")
    results = [
        benchmark_profile(
            dataset,
            train_records,
            dict(profile),
            seed=int(training_config["seed"]),
            warmup_batches=int(benchmark_config["warmup_batches"]),
            measured_batches=int(benchmark_config["measured_batches"]),
        )
        for profile in profiles
    ]
    payload = {
        "schema_version": "route_a_v3_route2_dataloader_benchmark.v1",
        "status": "HOST_DATALOADER_BENCHMARK_COMPLETE_NO_MODEL_UPDATE",
        "training_config": str(benchmark_config["training_config"]),
        "included_study_unit_ids": included_studies,
        "included_regions": included_regions,
        "train_record_count": len(train_records),
        "validation_record_count": len(by_split["VALIDATION"]),
        "withheld_test_record_count": withheld,
        "pretrained_model_id": pretrained.model_id,
        "pretrained_feature_width": pretrained.width,
        "profiles": results,
        "scientific_metrics_computed": False,
        "model_parameter_updates": 0,
        "evaluation_pool_records_read": 0,
    }
    output = Path(benchmark_config["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    _require(not output.exists(), f"benchmark output already exists: {output}")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
