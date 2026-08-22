#!/usr/bin/env python3
"""Train one frozen XEditCritic V3 screen arm from Development projections."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_development_projection_v3 import load_projection_rows
from core.route2_edit_site_token_cache_v3 import load_edit_site_token_cache_v3
from core.route2_experiment_ledger import build_training_attempt_row, record_training_attempt
from core.route2_xeditcritic_ledger_v3 import (
    critic_v3_attempt_config,
    critic_v3_attempt_details,
    critic_v3_ledger_paths,
    critic_v3_seed_and_stage,
    require_critic_v3_confirmation_authorization,
    require_critic_v3_posttest_authorization,
)
from core.route2_xeditcritic_training_data_v3 import (
    PAD_TOKEN,
    RNA_TOKEN,
    SqrtTaskStudySourcePassSamplerV3,
    XEditCriticRecordV3,
    build_exact_source_task_candidate_bundle_permutation,
    build_vocabs,
    different_source_group_pairwise_logistic_loss,
    records_from_projection_rows,
)
from core.route2_xeditcritic_v3 import XEditCriticV3


class XEditCriticTrainingError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticTrainingError(message)


@dataclass(frozen=True)
class TaskRobustScalerV3:
    scales: Mapping[str, float]
    region_scales: Mapping[int, float]
    global_scale: float
    floor: float
    training_record_count: int

    def scale(self, task: str, region: int) -> float:
        if task in self.scales:
            return float(self.scales[task])
        if int(region) in self.region_scales:
            return float(self.region_scales[int(region)])
        return float(self.global_scale)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "route_a_v3_route2_xeditcritic_task_robust_scaler.v3",
            "fit_scope": "TRAIN_ONLY",
            "center_subtracted": False,
            "floor": self.floor,
            "training_record_count": self.training_record_count,
            "task_scales": dict(sorted(self.scales.items())),
            "region_scales": {
                str(key): value for key, value in sorted(self.region_scales.items())
            },
            "global_scale": self.global_scale,
        }


def fit_task_robust_scaler(
    records: Sequence[XEditCriticRecordV3], *, floor: float = 1e-3
) -> TaskRobustScalerV3:
    _require(bool(records) and floor > 0.0, "target scaler input is invalid")
    by_task: dict[str, list[float]] = {}
    by_region: dict[int, list[float]] = {}
    all_values = []
    for record in records:
        by_task.setdefault(record.task, []).append(record.target)
        by_region.setdefault(record.region, []).append(record.target)
        all_values.append(record.target)

    def robust_scale(values: Sequence[float]) -> float:
        array = np.asarray(values, dtype=np.float64)
        median = float(np.median(array))
        mad = 1.4826 * float(np.median(np.abs(array - median)))
        zero_anchored = float(np.median(np.abs(array)))
        return max(mad, zero_anchored, floor)

    scales = {task: robust_scale(values) for task, values in by_task.items()}
    region_scales = {
        region: robust_scale(values) for region, values in by_region.items()
    }
    return TaskRobustScalerV3(
        scales, region_scales, robust_scale(all_values), floor, len(records)
    )


class EditSiteCacheViewV3:
    def __init__(self, payload: Mapping[str, Any], expected_record_ids: set[str]) -> None:
        record_ids = [str(value) for value in payload["record_ids"]]
        _require(set(record_ids) == expected_record_ids, "edit-site cache does not exactly cover the projection")
        self.record_index = {record_id: index for index, record_id in enumerate(record_ids)}
        self.payload = payload
        self.width = int(payload["embedding_width"])

    def bundle(self, record_id: str) -> dict[str, torch.Tensor]:
        record_index = self.record_index[record_id]
        start = int(self.payload["record_edit_offsets"][record_index])
        end = int(self.payload["record_edit_offsets"][record_index + 1])
        source_feature_indices = self.payload["edit_source_feature_indices"][start:end]
        candidate_feature_indices = self.payload["edit_candidate_feature_indices"][start:end]
        source_sequence_index = int(self.payload["record_source_sequence_indices"][record_index])
        candidate_sequence_index = int(self.payload["record_candidate_sequence_indices"][record_index])
        return {
            "edit_positions": self.payload["edit_positions"][start:end],
            "source_site": self.payload["position_site_hidden"][source_feature_indices],
            "candidate_site": self.payload["position_site_hidden"][candidate_feature_indices],
            "source_window_mean": self.payload["position_window_mean"][source_feature_indices],
            "candidate_window_mean": self.payload["position_window_mean"][candidate_feature_indices],
            "source_window_max": self.payload["position_window_max"][source_feature_indices],
            "candidate_window_max": self.payload["position_window_max"][candidate_feature_indices],
            "source_global": self.payload["global_residuals"][source_sequence_index],
            "candidate_global": self.payload["global_residuals"][candidate_sequence_index],
        }


def study_source_group_weights(
    records: Sequence[XEditCriticRecordV3],
) -> dict[str, float]:
    """Equalize study then source-group contribution inside each task."""

    group_sizes = Counter(record.source_group for record in records)
    task_study_groups: dict[str, dict[str, set[str]]] = {}
    group_task_study = {}
    for record in records:
        task_study_groups.setdefault(record.task, {}).setdefault(record.study, set()).add(
            record.source_group
        )
        group_task_study[record.source_group] = (record.task, record.study)
    raw = {}
    for record in records:
        task, study = group_task_study[record.source_group]
        raw[record.record_id] = 1.0 / (
            len(task_study_groups[task])
            * len(task_study_groups[task][study])
            * group_sizes[record.source_group]
        )
    by_task_total = Counter()
    by_task_count = Counter()
    by_id = {record.record_id: record for record in records}
    for record_id, weight in raw.items():
        task = by_id[record_id].task
        by_task_total[task] += weight
        by_task_count[task] += 1
    return {
        record_id: weight * by_task_count[by_id[record_id].task] / by_task_total[by_id[record_id].task]
        for record_id, weight in raw.items()
    }


class XEditCriticDatasetV3(Dataset):
    def __init__(
        self,
        records: Sequence[XEditCriticRecordV3],
        *,
        all_records: Mapping[str, XEditCriticRecordV3],
        vocabs: Mapping[str, Mapping[str, int]],
        target_scaler: TaskRobustScalerV3,
        cache: EditSiteCacheViewV3 | None,
        candidate_bundle_overrides: Mapping[str, str] | None = None,
        neutral_studies: set[str] | None = None,
    ) -> None:
        self.records = list(records)
        self.all_records = all_records
        self.vocabs = vocabs
        self.target_scaler = target_scaler
        self.cache = cache
        self.overrides = dict(candidate_bundle_overrides or {})
        self.neutral_studies = set(neutral_studies or set())
        _require(set(self.overrides) <= {record.record_id for record in records}, "candidate permutation recipient is outside the dataset")
        self.weights = study_source_group_weights(records)
        sequences = {
            sequence
            for record in records
            for sequence in (record.source, record.candidate)
        }
        for donor_id in self.overrides.values():
            sequences.add(all_records[donor_id].candidate)
        self.tokens = {
            sequence: torch.tensor([RNA_TOKEN[base] for base in sequence], dtype=torch.long)
            for sequence in sequences
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        donor = self.all_records[self.overrides.get(record.record_id, record.record_id)]
        _require(record.source == donor.source and record.task == donor.task, "candidate donor left exact source/task stratum")
        scale = self.target_scaler.scale(record.task, record.region)
        result = {
            "record_id": record.record_id,
            "source_group": record.source_group,
            "task": record.task,
            "source": self.tokens[record.source],
            "candidate": self.tokens[donor.candidate],
            "edits": donor.edits,
            "target": record.target,
            "scaled_target": record.target / scale,
            "target_scale": scale,
            "sample_weight": self.weights[record.record_id],
            "study": 0 if record.study in self.neutral_studies else self.vocabs["study"].get(record.study, 0),
            "assay": self.vocabs["assay"].get(record.assay, 0),
            "context": self.vocabs["context"].get(record.context, 0),
            "quantity": self.vocabs["quantity"].get(record.quantity, 0),
            "measurement": self.vocabs["measurement"].get(record.measurement, 0),
            "numerator": self.vocabs["numerator"].get(record.numerator, 0),
            "denominator": self.vocabs["denominator"].get(record.denominator, 0),
            "region": record.region,
        }
        if self.cache is not None:
            result["feature_bundle"] = self.cache.bundle(donor.record_id)
        return result


class XEditCriticCollatorV3:
    def __init__(self, *, pretrained_width: int = 768) -> None:
        self.pretrained_width = int(pretrained_width)

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch_size = len(examples)
        maximum_length = max(len(example["source"]) for example in examples)
        maximum_edits = max(1, max(len(example["edits"]) for example in examples))
        source = torch.full((batch_size, maximum_length), PAD_TOKEN, dtype=torch.long)
        candidate = torch.full_like(source, PAD_TOKEN)
        padding_mask = torch.ones_like(source, dtype=torch.bool)
        edit_padding_mask = torch.ones((batch_size, maximum_edits), dtype=torch.bool)
        source_edit_base_ids = torch.full((batch_size, maximum_edits), PAD_TOKEN, dtype=torch.long)
        candidate_edit_base_ids = torch.full_like(source_edit_base_ids, PAD_TOKEN)
        normalized_positions = torch.zeros((batch_size, maximum_edits), dtype=torch.float32)
        edit_positions = torch.zeros((batch_size, maximum_edits), dtype=torch.long)
        local_names = ("site", "window_mean", "window_max")
        local = {
            f"{side}_{name}": torch.zeros(
                (batch_size, maximum_edits, self.pretrained_width), dtype=torch.float32
            )
            for side in ("source", "candidate")
            for name in local_names
        }
        global_features = {
            f"{side}_global": torch.zeros((batch_size, self.pretrained_width), dtype=torch.float32)
            for side in ("source", "candidate")
        }
        for batch_index, example in enumerate(examples):
            length = len(example["source"])
            source[batch_index, :length] = example["source"]
            candidate[batch_index, :length] = example["candidate"]
            padding_mask[batch_index, :length] = False
            edits = example["edits"]
            if edits:
                edit_padding_mask[batch_index, : len(edits)] = False
                denominator = max(1, length - 1)
                for edit_index, (position, source_base, candidate_base) in enumerate(edits):
                    source_edit_base_ids[batch_index, edit_index] = RNA_TOKEN[source_base]
                    candidate_edit_base_ids[batch_index, edit_index] = RNA_TOKEN[candidate_base]
                    edit_positions[batch_index, edit_index] = position
                    normalized_positions[batch_index, edit_index] = position / denominator
            bundle = example.get("feature_bundle")
            if bundle is not None:
                _require(bundle["edit_positions"].tolist() == [edit[0] for edit in edits], "cache edit positions differ from candidate bundle")
                for side in ("source", "candidate"):
                    for name in local_names:
                        local[f"{side}_{name}"][batch_index, : len(edits)] = bundle[f"{side}_{name}"].float()
                    global_features[f"{side}_global"][batch_index] = bundle[f"{side}_global"].float()
        result = {
            "record_ids": [example["record_id"] for example in examples],
            "source_groups": [example["source_group"] for example in examples],
            "task_ids": [example["task"] for example in examples],
            "source_tokens": source,
            "candidate_tokens": candidate,
            "padding_mask": padding_mask,
            "edit_padding_mask": edit_padding_mask,
            "source_edit_base_ids": source_edit_base_ids,
            "candidate_edit_base_ids": candidate_edit_base_ids,
            "normalized_edit_positions": normalized_positions,
            "edit_positions": edit_positions,
            "study_ids": torch.tensor([example["study"] for example in examples], dtype=torch.long),
            "assay_ids": torch.tensor([example["assay"] for example in examples], dtype=torch.long),
            "context_ids": torch.tensor([example["context"] for example in examples], dtype=torch.long),
            "quantity_ids": torch.tensor([example["quantity"] for example in examples], dtype=torch.long),
            "measurement_ids": torch.tensor([example["measurement"] for example in examples], dtype=torch.long),
            "numerator_ids": torch.tensor([example["numerator"] for example in examples], dtype=torch.long),
            "denominator_ids": torch.tensor([example["denominator"] for example in examples], dtype=torch.long),
            "region_ids": torch.tensor([example["region"] for example in examples], dtype=torch.long),
            "target": torch.tensor([example["target"] for example in examples], dtype=torch.float32),
            "scaled_target": torch.tensor([example["scaled_target"] for example in examples], dtype=torch.float32),
            "target_scale": torch.tensor([example["target_scale"] for example in examples], dtype=torch.float32),
            "sample_weight": torch.tensor([example["sample_weight"] for example in examples], dtype=torch.float32),
        }
        result.update(local)
        result.update(global_features)
        return result


def _move(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def validation_metrics(
    targets: Sequence[float],
    predictions: Sequence[float],
    scaled_targets: Sequence[float],
    scaled_predictions: Sequence[float],
    tasks: Sequence[str],
) -> dict[str, Any]:
    by_task: dict[str, list[int]] = {}
    for index, task in enumerate(tasks):
        by_task.setdefault(str(task), []).append(index)
    task_rows = {}
    spearmans = []
    standardized_maes = []
    target_array = np.asarray(targets)
    prediction_array = np.asarray(predictions)
    scaled_target_array = np.asarray(scaled_targets)
    scaled_prediction_array = np.asarray(scaled_predictions)
    for task, indices in sorted(by_task.items()):
        task_target = target_array[indices]
        task_prediction = prediction_array[indices]
        correlation = None
        if len(indices) >= 3 and np.std(task_target) > 0 and np.std(task_prediction) > 0:
            value = float(spearmanr(task_target, task_prediction).statistic)
            correlation = value if math.isfinite(value) else None
        standardized_mae = float(
            np.mean(np.abs(scaled_prediction_array[indices] - scaled_target_array[indices]))
        )
        if correlation is not None:
            spearmans.append(correlation)
        standardized_maes.append(standardized_mae)
        task_rows[task] = {
            "record_count": len(indices),
            "spearman": correlation,
            "standardized_mae": standardized_mae,
        }
    _require(len(spearmans) == len(by_task), "a Validation task has undefined Spearman")
    return {
        "record_count": len(targets),
        "task_count": len(by_task),
        "task_macro_spearman": float(np.mean(spearmans)),
        "task_macro_standardized_mae": float(np.mean(standardized_maes)),
        "positive_task_count": sum(value > 0 for value in spearmans),
        "prediction_std": float(np.std(prediction_array)),
        "prediction_min": float(np.min(prediction_array)),
        "prediction_max": float(np.max(prediction_array)),
        "tasks": task_rows,
    }


def evaluate(
    model: XEditCriticV3,
    loader: DataLoader,
    device: torch.device,
    *,
    prediction_output_path: Path | None = None,
) -> dict[str, Any]:
    targets: list[float] = []
    predictions: list[float] = []
    scaled_targets: list[float] = []
    scaled_predictions: list[float] = []
    tasks: list[str] = []
    record_ids: list[str] = []
    source_groups: list[str] = []
    model.eval()
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _move(raw_batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(batch)
            scaled_prediction = output["mean"].float()
            prediction = scaled_prediction * batch["target_scale"]
            targets.extend(batch["target"].float().cpu().tolist())
            predictions.extend(prediction.cpu().tolist())
            scaled_targets.extend(batch["scaled_target"].float().cpu().tolist())
            scaled_predictions.extend(scaled_prediction.cpu().tolist())
            tasks.extend(batch["task_ids"])
            record_ids.extend(batch["record_ids"])
            source_groups.extend(batch["source_groups"])
    metrics = validation_metrics(
        targets, predictions, scaled_targets, scaled_predictions, tasks
    )
    if prediction_output_path is not None:
        _require(not prediction_output_path.exists(), "Validation prediction artifact already exists")
        with prediction_output_path.open("w", encoding="utf-8") as handle:
            for values in zip(
                record_ids,
                source_groups,
                tasks,
                targets,
                predictions,
                scaled_targets,
                scaled_predictions,
            ):
                record_id, source_group, task, target, prediction, scaled_target, scaled_prediction = values
                handle.write(
                    json.dumps(
                        {
                            "record_id": record_id,
                            "source_group_id": source_group,
                            "task_id": task,
                            "target": target,
                            "prediction": prediction,
                            "scaled_target": scaled_target,
                            "scaled_prediction": scaled_prediction,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    return metrics


def require_cuda(physical_gpu_index: int) -> torch.device:
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden")
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    _require(0 <= physical_gpu_index < torch.cuda.device_count(), "physical GPU is unavailable")
    device = torch.device(f"cuda:{physical_gpu_index}")
    torch.cuda.set_device(device)
    name = torch.cuda.get_device_name(device)
    _require("A100" in name, "formal Critic V3 updates require an A100")
    return device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def critic_v3_stage_partitions(
    records: Sequence[XEditCriticRecordV3],
    *,
    run_stage: str,
    held_out_study: str | None = None,
) -> tuple[list[XEditCriticRecordV3], list[XEditCriticRecordV3], set[str]]:
    if run_stage in {"SCREEN", "CONFIRMATION"}:
        return (
            [record for record in records if record.split == "TRAIN"],
            [record for record in records if record.split == "VALIDATION"],
            set(),
        )
    if run_stage == "REFIT":
        return list(records), [], set()
    _require(run_stage == "LOSO" and bool(held_out_study), "Critic V3 stage partition differs")
    train = [record for record in records if record.study != held_out_study]
    validation = [record for record in records if record.study == held_out_study]
    _require(bool(train) and bool(validation), "LOSO fold has an empty train or held-out partition")
    return train, validation, {str(held_out_study)}


def run(
    config: Mapping[str, Any],
    *,
    arm: str,
    control_mode: str,
    candidate_bundle_permutation: bool,
    physical_gpu_index: int,
) -> dict[str, Any]:
    _require(arm in {"C0", "C1", "C2"}, "this cache trainer supports C0/C1/C2; C3 uses the online-LoRA runner")
    _require(not (control_mode != "NONE" and candidate_bundle_permutation), "candidate controls cannot be combined")
    if arm in {"C0", "C1"}:
        _require(control_mode == "NONE" and not candidate_bundle_permutation, "C0/C1 are fixed diagnostics")
    seed, run_stage = critic_v3_seed_and_stage(config)
    if run_stage == "CONFIRMATION":
        _require(
            control_mode == "NONE" and not candidate_bundle_permutation,
            "confirmation does not authorize candidate-information controls",
        )
        require_critic_v3_confirmation_authorization(config, arm=arm)
    elif run_stage in {"REFIT", "LOSO"}:
        _require(
            control_mode == "NONE" and not candidate_bundle_permutation,
            "post-TEST stages do not authorize candidate-information controls",
        )
        require_critic_v3_posttest_authorization(config, arm=arm)
    _set_seed(seed)
    device = require_cuda(physical_gpu_index)
    run_id = arm.lower()
    if control_mode != "NONE":
        run_id += "_" + control_mode.lower()
    if candidate_bundle_permutation:
        run_id += "_candidate_bundle_permutation"
    output_directory = Path(config["output_root"]) / run_id
    _require(not output_directory.exists(), f"terminal run directory already exists: {output_directory}")
    output_directory.mkdir(parents=True)
    started = time.time()
    attempt_config = critic_v3_attempt_config(
        config,
        run_id=run_id,
        arm=arm,
        control_mode=control_mode,
        candidate_bundle_permutation=candidate_bundle_permutation,
        physical_gpu_index=physical_gpu_index,
    )
    ledger_path, attempt_path = critic_v3_ledger_paths(config, output_directory)
    attempt_details = critic_v3_attempt_details(config)
    try:
        projection_rows = load_projection_rows(
            [Path(path) for path in config["projection_paths"]]
        )
        records = records_from_projection_rows(projection_rows)
        _require(len(records) == int(config["expected_record_count"]), "projection record count changed")
        train_records, validation_records, neutral_studies = critic_v3_stage_partitions(
            records,
            run_stage=run_stage,
            held_out_study=config.get("held_out_study"),
        )
        _require(len(train_records) == int(config["expected_train_count"]), "TRAIN count changed")
        _require(len(validation_records) == int(config["expected_validation_count"]), "VALIDATION count changed")
        record_by_id = {record.record_id: record for record in records}
        vocabs = build_vocabs(records)
        scaler = fit_task_robust_scaler(train_records, floor=float(config["target_scale_floor"]))
        cache = None
        if arm != "C0":
            cache = EditSiteCacheViewV3(
                load_edit_site_token_cache_v3(Path(config["edit_site_cache"])),
                set(record_by_id),
            )
        if candidate_bundle_permutation:
            overrides, permutation_summary = build_exact_source_task_candidate_bundle_permutation(
                train_records, seed=seed
            )
        else:
            overrides, permutation_summary = {}, {
                "complete_candidate_bundle_permuted": False,
                "recipient_count": 0,
                "eligible_task_count": 0,
            }
        train_dataset = XEditCriticDatasetV3(
            train_records,
            all_records=record_by_id,
            vocabs=vocabs,
            target_scaler=scaler,
            cache=cache,
            candidate_bundle_overrides=overrides,
        )
        validation_dataset = XEditCriticDatasetV3(
            validation_records,
            all_records=record_by_id,
            vocabs=vocabs,
            target_scaler=scaler,
            cache=cache,
            neutral_studies=neutral_studies,
        )
        collator = XEditCriticCollatorV3(pretrained_width=int(config["pretrained_width"]))
        sampler = SqrtTaskStudySourcePassSamplerV3(
            train_records,
            batch_size=int(config["batch_size"]),
            seed=seed,
            repeat_cap=int(config["maximum_record_repeats_per_pass"]),
        )
        validation_loader = (
            None
            if not validation_records
            else DataLoader(
                validation_dataset,
                batch_size=int(config["batch_size"]),
                shuffle=False,
                collate_fn=collator,
                num_workers=0,
                pin_memory=True,
            )
        )
        model = XEditCriticV3(
            arm=arm,
            control_mode=control_mode,
            study_count=len(vocabs["study"]),
            assay_count=len(vocabs["assay"]),
            context_count=len(vocabs["context"]),
            quantity_count=len(vocabs["quantity"]),
            measurement_count=len(vocabs["measurement"]),
            numerator_count=len(vocabs["numerator"]),
            denominator_count=len(vocabs["denominator"]),
            region_count=2,
            pretrained_width=int(config["pretrained_width"]),
            dropout=float(config["dropout"]),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["head_learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        attempt_details = critic_v3_attempt_details(
            config,
            trainable_parameter_count=model.trainable_parameter_count,
            train_record_count=len(train_records),
            validation_record_count=len(validation_records),
        )
        if arm in {"C1", "C2"}:
            attempt_details["frozen_pretrained_parameter_count"] = 113_389_056
            attempt_details["total_effective_parameter_count"] = (
                model.trainable_parameter_count + 113_389_056
            )
        record_training_attempt(
            ledger_path,
            attempt_path,
            build_training_attempt_row(
                attempt_config,
                output_directory,
                "RUNNING",
                repository_root=REPO_ROOT,
                details=attempt_details,
            ),
        )
        initial_parameter = next(model.parameters()).detach().clone()
        pass_rows = []
        update_count = 0
        for pass_index in range(int(config["passes"])):
            sampler.set_pass(pass_index)
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=sampler,
                collate_fn=collator,
                num_workers=0,
                pin_memory=True,
            )
            model.train()
            losses = []
            regression_losses = []
            ranking_losses = []
            for raw_batch in train_loader:
                batch = _move(raw_batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(batch)
                    per_record = F.huber_loss(
                        output["mean"],
                        batch["scaled_target"],
                        reduction="none",
                        delta=float(config["huber_delta"]),
                    )
                    regression = (per_record * batch["sample_weight"]).sum() / batch["sample_weight"].sum().clamp_min(1e-12)
                    ranking = None
                    if pass_index == int(config["passes"]) - 1:
                        ranking = different_source_group_pairwise_logistic_loss(
                            output["mean"],
                            batch["scaled_target"],
                            batch["source_groups"],
                            batch["task_ids"],
                        )
                    loss = regression if ranking is None else regression + float(config["ranking_loss_weight"]) * ranking
                _require(torch.isfinite(loss).item(), "training loss is nonfinite")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["gradient_clip_norm"])
                )
                _require(torch.isfinite(gradient_norm).item(), "gradient norm is nonfinite")
                optimizer.step()
                update_count += 1
                losses.append(float(loss.detach().cpu()))
                regression_losses.append(float(regression.detach().cpu()))
                if ranking is not None:
                    ranking_losses.append(float(ranking.detach().cpu()))
            metrics = (
                {
                    "status": "NOT_APPLICABLE_ALL_DEVELOPMENT_REFIT",
                    "task_count": 0,
                }
                if validation_loader is None
                else evaluate(
                    model,
                    validation_loader,
                    device,
                    prediction_output_path=(
                        output_directory / "final_validation_predictions.jsonl"
                        if pass_index == int(config["passes"]) - 1
                        else None
                    ),
                )
            )
            pass_row = {
                "pass": pass_index + 1,
                "update_count_cumulative": update_count,
                "mean_loss": float(np.mean(losses)),
                "mean_regression_loss": float(np.mean(regression_losses)),
                "mean_ranking_loss": None if not ranking_losses else float(np.mean(ranking_losses)),
                "validation": metrics,
            }
            pass_rows.append(pass_row)
            print(json.dumps({"event": "XEDITCRITIC_V3_PASS_COMPLETE", "run_id": run_id, **pass_row}, sort_keys=True), flush=True)

        final_metrics = pass_rows[-1]["validation"]
        parameter_changed = not torch.equal(
            initial_parameter, next(model.parameters()).detach()
        )
        _require(update_count > 0 and parameter_changed, "Critic V3 cache arm performed no learned update")
        checkpoint_path = output_directory / "final_pass_checkpoint.pt"
        torch.save(
            {
                "schema_version": "route_a_v3_route2_xeditcritic_v3_checkpoint.v1",
                "arm": arm,
                "control_mode": control_mode,
                "candidate_bundle_permutation": candidate_bundle_permutation,
                "seed": seed,
                "selected_pass": int(config["passes"]),
                "selection_policy": "FINAL_PASS_FIXED_NO_RANKING_PHASE_RESELECTION",
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "study_count": len(vocabs["study"]),
                    "assay_count": len(vocabs["assay"]),
                    "context_count": len(vocabs["context"]),
                    "quantity_count": len(vocabs["quantity"]),
                    "measurement_count": len(vocabs["measurement"]),
                    "numerator_count": len(vocabs["numerator"]),
                    "denominator_count": len(vocabs["denominator"]),
                    "pretrained_width": int(config["pretrained_width"]),
                },
                "vocabs": vocabs,
                "target_scaler": scaler.to_dict(),
                "validation_metrics": final_metrics,
            },
            checkpoint_path,
        )
        summary = {
            "schema_version": f"route_a_v3_route2_xeditcritic_v3_{run_stage.lower()}_run.v1",
            "status": f"TERMINAL_{run_stage}_ARM_COMPLETE",
            "run_id": run_id,
            "arm": arm,
            "control_mode": control_mode,
            "candidate_bundle_permutation": candidate_bundle_permutation,
            "candidate_permutation_summary": permutation_summary,
            "seed": seed,
            "physical_gpu_index": physical_gpu_index,
            "cuda_device_name": torch.cuda.get_device_name(device),
            "precision": "BF16",
            "trainable_parameter_count": model.trainable_parameter_count,
            "train_record_count": len(train_records),
            "validation_record_count": len(validation_records),
            "pass_count": len(pass_rows),
            "selected_pass": int(config["passes"]),
            "update_count": update_count,
            "parameter_changed": parameter_changed,
            "cuda_training_tensors_verified": True,
            "cpu_fallback_used": False,
            "selection_policy": "FINAL_PASS_FIXED_NO_RANKING_PHASE_RESELECTION",
            "sampler": {
                "policy": "SQRT_TASK_SIZE_TASK_HOMOGENEOUS_STUDY_SOURCE_GROUP_CYCLES",
                "repeat_cap": sampler.repeat_cap,
                "task_allocations_per_pass": dict(sorted(sampler.allocations.items())),
            },
            "target_scaler": scaler.to_dict(),
            "passes": pass_rows,
            "final_validation": final_metrics,
            "checkpoint_path": str(checkpoint_path),
            "validation_prediction_path": str(
                output_directory / "final_validation_predictions.jsonl"
            ) if validation_loader is not None else None,
            "training_scope": (
                "ALL_DEVELOPMENT"
                if run_stage == "REFIT"
                else "LEAVE_ONE_STUDY_OUT"
                if run_stage == "LOSO"
                else "FROZEN_TRAIN_VALIDATION"
            ),
            "held_out_study": config.get("held_out_study"),
            "held_out_study_scale_policy": (
                "UNKNOWN_STUDY_SCALE_FIXED_1" if run_stage == "LOSO" else None
            ),
            "elapsed_seconds": time.time() - started,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
            "development_test_outcomes_accessed": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        (output_directory / "run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        record_training_attempt(
            ledger_path,
            attempt_path,
            build_training_attempt_row(
                attempt_config,
                output_directory,
                "COMPLETED",
                repository_root=REPO_ROOT,
                details={
                    **attempt_details,
                    "optimizer_steps": update_count,
                    "selected_epoch": int(config["passes"]),
                    "validation_metrics": final_metrics,
                    "wall_time_seconds": summary["elapsed_seconds"],
                    "peak_vram_mb": summary["peak_vram_bytes"] / 1024**2,
                    "notes": f"terminal prospective Critic V3 {run_stage.lower()} arm; no TEST or Evaluation access",
                },
            ),
        )
        return summary
    except Exception as exc:
        failure = {
            "schema_version": f"route_a_v3_route2_xeditcritic_v3_{run_stage.lower()}_run_failure.v1",
            "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
            "arm": arm,
            "control_mode": control_mode,
            "candidate_bundle_permutation": candidate_bundle_permutation,
            "seed": seed,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": time.time() - started,
            "development_test_outcomes_accessed": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        (output_directory / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        record_training_attempt(
            ledger_path,
            attempt_path,
            build_training_attempt_row(
                attempt_config,
                output_directory,
                "FAILED",
                repository_root=REPO_ROOT,
                details={
                    **attempt_details,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "wall_time_seconds": failure["elapsed_seconds"],
                },
            ),
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("C0", "C1", "C2"))
    parser.add_argument(
        "--control-mode",
        default="NONE",
        choices=("NONE", "SOURCE_ONLY", "EDIT_METADATA_ONLY", "NO_CANDIDATE_SEQUENCE"),
    )
    parser.add_argument("--candidate-bundle-permutation", action="store_true")
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    print(
        json.dumps(
            run(
                config,
                arm=arguments.arm,
                control_mode=arguments.control_mode,
                candidate_bundle_permutation=arguments.candidate_bundle_permutation,
                physical_gpu_index=arguments.physical_gpu_index,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
