#!/usr/bin/env python3
"""Train one Route 2 Delta-predictor seed on the frozen Development split."""

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
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset, Sampler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_delta_predictor import (
    ROUTE2_DELTA_MODEL_KIND,
    ROUTE2_EDIT_CENTERED_MODEL_KIND,
    ROUTE2_EDIT_CENTERED_SOURCE_ONLY_KIND,
    ROUTE2_LEGACY_RNAFM_EDIT_CENTERED_MODEL_KIND,
    ROUTE2_PRETRAINED_EDIT_CENTERED_MODEL_KIND,
    Route2DeltaPredictor,
    Route2EditCenteredDeltaPredictor,
    Route2NeuralBaseline,
    Route2PretrainedEditCenteredDeltaPredictor,
)
from core.route2_gpu_failure_evidence import cuda_device_observation, write_gpu_failure_evidence
from core.route2_experiment_ledger import (
    build_training_attempt_row,
    record_training_attempt,
)
from core.route2_target_scaling import (
    TARGET_SCALING_NONE,
    TARGET_SCALING_TRAIN_TASK_ROBUST,
    Route2TargetScaler,
    fit_route2_target_scaler,
    task_key,
)


TOKEN = {"A": 0, "C": 1, "G": 2, "U": 3}
PAD = 4
REGION = {"5UTR": 0, "3UTR": 1}
TRAINING_UPDATE_STANDARD = "STANDARD"
TRAINING_UPDATE_TASK_GRADIENT_NORM_CALIBRATED = (
    "TRAIN_TASK_GRADIENT_NORM_CALIBRATED"
)
SHARED_EFFECT_EXCLUDED_PREFIXES = (
    "study.",
    "assay.",
    "context.",
    "endpoint.",
    "region.",
    "region_scale.",
    "region_shift.",
)
PRETRAINED_EDIT_CENTERED_MODEL_KINDS = {
    ROUTE2_PRETRAINED_EDIT_CENTERED_MODEL_KIND,
    ROUTE2_LEGACY_RNAFM_EDIT_CENTERED_MODEL_KIND,
}


class DeltaTrainingError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DeltaTrainingError(message)


def _normalize(value: Any) -> str:
    sequence = str(value).upper().replace("T", "U")
    _require(sequence and set(sequence) <= set(TOKEN), "sequence is outside RNA alphabet")
    return sequence


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            _require(row["pool_assignment"] == "DEVELOPMENT", "Evaluation record entered Delta manifest")
            result[str(row["canonical_record_id"])] = {
                "split": str(row["split"]),
                "study_unit_id": str(row["study_unit_id"]),
                "connected_source_component_id": str(row["connected_source_component_id"]),
            }
    _require(result, "Development manifest is empty")
    return result


@dataclass(frozen=True)
class DeltaRecord:
    record_id: str
    split: str
    source: str
    candidate: str
    target: float
    source_group: str
    study: str
    assay: str
    context: str
    endpoint: str
    region: int


class FrozenPretrainedPairFeatures:
    """Record-aligned frozen pair embeddings without sequence payloads."""

    def __init__(self, path: Path, expected_record_ids: set[str]):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _require(
            payload.get("schema_version") in {
                "route_a_v3_route2_frozen_pair_features.v1",
                "route_a_v3_route2_rnafm_pair_features.v1",
            },
            "unexpected pretrained feature-cache schema",
        )
        record_ids = [str(value) for value in payload["record_ids"]]
        _require(len(record_ids) == len(set(record_ids)), "pretrained feature ids are duplicated")
        _require(set(record_ids) == expected_record_ids, "pretrained features do not exactly cover Development")
        self.index = {record_id: index for index, record_id in enumerate(record_ids)}
        self.source = payload["source_embeddings"].float()
        self.candidate = payload["candidate_embeddings"].float()
        _require(self.source.ndim == 2 and self.source.shape == self.candidate.shape, "pretrained pair feature shape is invalid")
        _require(self.source.shape[0] == len(record_ids), "pretrained pair feature row count differs")
        _require(torch.isfinite(self.source).all().item() and torch.isfinite(self.candidate).all().item(), "pretrained pair feature is nonfinite")
        self.width = int(self.source.shape[1])
        self.pretrained_parameter_count = int(payload["pretrained_parameter_count"])
        self.model_id = str(payload["model_id"])

    def pair(self, record_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        index = self.index[record_id]
        return self.source[index], self.candidate[index]


def load_records(canonical_paths: Iterable[Path], manifest: Mapping[str, Mapping[str, str]]) -> list[DeltaRecord]:
    records = {}
    for path in canonical_paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                record_id = str(row["canonical_record_id"])
                if record_id not in manifest:
                    continue
                _require(row["pool_assignment"] == "DEVELOPMENT", "Evaluation record entered Delta loader")
                source, candidate = _normalize(row["source_sequence"]), _normalize(row["candidate_sequence"])
                _require(len(source) == len(candidate), f"length-changing row entered SUB predictor: {record_id}")
                target = row["direction_normalized_delta"]
                _require(isinstance(target, (int, float)) and not isinstance(target, bool) and math.isfinite(float(target)), f"invalid target: {record_id}")
                region = REGION.get(str(row["region"]).replace("′", "").replace("'", ""))
                _require(region is not None, f"unsupported region: {row['region']}")
                _require(record_id not in records, f"canonical record duplicated: {record_id}")
                study = str(row["study_unit_id"])
                context = str(row["biological_context_id"])
                endpoint = str(row["endpoint_id"])
                source_group = "::".join((study, str(row["source_id"]), context, endpoint))
                records[record_id] = DeltaRecord(
                    record_id, manifest[record_id]["split"], source, candidate, float(target), source_group,
                    study, str(row["assay_id"]), context, endpoint, region,
                )
    _require(set(records) == set(manifest), "canonical inputs do not exactly cover Development manifest")
    return [records[key] for key in sorted(records)]


def build_vocab(records: Iterable[DeltaRecord], field: str) -> dict[str, int]:
    return {"__UNK__": 0} | {
        value: index + 1 for index, value in enumerate(sorted({str(getattr(row, field)) for row in records}))
    }


class DeltaDataset(Dataset):
    def __init__(
        self,
        records: list[DeltaRecord],
        vocabs: Mapping[str, Mapping[str, int]],
        *,
        metadata_mode: str = "FULL_CONTEXT",
        weighting_mode: str = "SOURCE_CONTEXT_ENDPOINT_GROUP",
        target_scaler: Route2TargetScaler | None = None,
        candidate_overrides: Mapping[str, str] | None = None,
        pretrained_features: FrozenPretrainedPairFeatures | None = None,
    ):
        _require(
            metadata_mode in {"FULL_CONTEXT", "TRANSFERABLE_CONTEXT", "SEQUENCE_AND_REGION_ONLY"},
            "unknown metadata mode",
        )
        _require(
            weighting_mode in {
                "SOURCE_CONTEXT_ENDPOINT_GROUP",
                "STUDY_THEN_SOURCE_CONTEXT_ENDPOINT_GROUP",
                "TASK_THEN_SOURCE_CONTEXT_ENDPOINT_GROUP",
            },
            "unknown training weighting mode",
        )
        self.records = records
        self.vocabs = vocabs
        self.metadata_mode = metadata_mode
        self.target_scaler = target_scaler or fit_route2_target_scaler(
            records, mode=TARGET_SCALING_NONE
        )
        self.candidate_overrides = dict(candidate_overrides or {})
        self.pretrained_features = pretrained_features
        _require(
            set(self.candidate_overrides) <= {row.record_id for row in records},
            "candidate override is outside this dataset",
        )
        self.group_sizes = Counter(row.source_group for row in records)
        if weighting_mode == "SOURCE_CONTEXT_ENDPOINT_GROUP":
            raw_weights = {
                group: 1.0 / size for group, size in self.group_sizes.items()
            }
        elif weighting_mode == "STUDY_THEN_SOURCE_CONTEXT_ENDPOINT_GROUP":
            study_groups: dict[str, set[str]] = {}
            group_study = {}
            for row in records:
                study_groups.setdefault(row.study, set()).add(row.source_group)
                group_study[row.source_group] = row.study
            raw_weights = {
                group: 1.0 / (len(study_groups[group_study[group]]) * size)
                for group, size in self.group_sizes.items()
            }
        else:
            task_groups: dict[str, set[str]] = {}
            group_task = {}
            for row in records:
                task = task_key(row.endpoint, row.region)
                task_groups.setdefault(task, set()).add(row.source_group)
                group_task[row.source_group] = task
            raw_weights = {
                group: 1.0 / (len(task_groups[group_task[group]]) * size)
                for group, size in self.group_sizes.items()
            }
        raw_total = sum(raw_weights[row.source_group] for row in records)
        self.group_weights = {
            group: value * len(records) / raw_total for group, value in raw_weights.items()
        }

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        target_scale, target_scale_source = self.target_scaler.scale(row.endpoint, row.region)
        candidate = self.candidate_overrides.get(row.record_id, row.candidate)
        result = {
            "record_id": row.record_id,
            "source": [TOKEN[base] for base in row.source],
            "candidate": [TOKEN[base] for base in candidate],
            "target": row.target,
            "scaled_target": row.target / target_scale,
            "target_scale": target_scale,
            "target_scale_source": target_scale_source,
            "task_key": task_key(row.endpoint, row.region),
            "sample_weight": self.group_weights[row.source_group],
            "source_group": row.source_group,
            "study": 0 if self.metadata_mode != "FULL_CONTEXT" else self.vocabs["study"].get(row.study, 0),
            "assay": 0 if self.metadata_mode == "SEQUENCE_AND_REGION_ONLY" else self.vocabs["assay"].get(row.assay, 0),
            "context": 0 if self.metadata_mode == "SEQUENCE_AND_REGION_ONLY" else self.vocabs["context"].get(row.context, 0),
            "endpoint": 0 if self.metadata_mode == "SEQUENCE_AND_REGION_ONLY" else self.vocabs["endpoint"].get(row.endpoint, 0),
            "region": row.region,
        }
        if self.pretrained_features is not None:
            source_pretrained, candidate_pretrained = self.pretrained_features.pair(
                row.record_id
            )
            result["source_pretrained"] = source_pretrained
            result["candidate_pretrained"] = candidate_pretrained
        return result


def select_study_subset(
    records: list[DeltaRecord],
    included_study_unit_ids: list[str] | None,
) -> tuple[list[DeltaRecord], list[str], int]:
    available = sorted({row.study for row in records})
    if included_study_unit_ids is None:
        return records, available, 0
    included = sorted({str(value) for value in included_study_unit_ids})
    _require(included and len(included) == len(included_study_unit_ids), "included study list is empty or duplicated")
    _require(set(included) <= set(available), "included study is absent from Development")
    selected = [row for row in records if row.study in included]
    _require(selected, "included study subset is empty")
    return selected, included, len(records) - len(selected)


def select_region_subset(
    records: list[DeltaRecord],
    included_regions: list[str] | None,
) -> tuple[list[DeltaRecord], list[str], int]:
    if included_regions is None:
        return records, sorted(name for name, code in REGION.items() if any(row.region == code for row in records)), 0
    normalized = [str(value).replace("′", "").replace("'", "") for value in included_regions]
    _require(normalized and len(normalized) == len(set(normalized)), "included region list is empty or duplicated")
    _require(set(normalized) <= set(REGION), "included region is unsupported")
    codes = {REGION[name] for name in normalized}
    selected = [row for row in records if row.region in codes]
    _require(selected, "included region subset is empty")
    return selected, sorted(normalized), len(records) - len(selected)


def build_training_candidate_permutation(
    records: list[DeltaRecord], seed: int
) -> tuple[dict[str, str], dict[str, int]]:
    """Permute candidates within exact-source/task strata for a matched control."""

    groups: dict[tuple[str, str, int], list[DeltaRecord]] = {}
    for row in records:
        groups.setdefault((row.source, row.endpoint, row.region), []).append(row)
    overrides = {}
    permutable_records = 0
    changed_candidates = 0
    for group_number, key in enumerate(sorted(groups)):
        rows = sorted(groups[key], key=lambda row: row.record_id)
        if len(rows) < 2:
            continue
        order = list(range(len(rows)))
        random.Random(seed + group_number).shuffle(order)
        donors = order[1:] + order[:1]
        for recipient_index, donor_index in zip(order, donors):
            recipient = rows[recipient_index]
            candidate = rows[donor_index].candidate
            overrides[recipient.record_id] = candidate
            permutable_records += 1
            changed_candidates += candidate != recipient.candidate
    return overrides, {
        "permutation_stratum": "EXACT_SOURCE_SEQUENCE_ENDPOINT_REGION",
        "candidate_pool_membership_preserved": True,
        "edit_distance_multiset_preserved": True,
        "training_record_count": len(records),
        "permutable_record_count": permutable_records,
        "changed_candidate_sequence_count": changed_candidates,
        "singleton_or_unpermutable_record_count": len(records) - permutable_records,
    }


def fixed_split_records(
    records: list[DeltaRecord], result_stage: str
) -> tuple[dict[str, list[DeltaRecord]], int]:
    """Expose Development TEST outcomes only after the configuration is frozen."""
    _require(
        result_stage in {
            "HPO_VALIDATION_ONLY",
            "FROZEN_DEVELOPMENT_VALIDATION",
            "FROZEN_DEVELOPMENT_TEST",
            "FINAL_ALL_DEVELOPMENT_REFIT",
        },
        f"invalid result_stage for fixed split: {result_stage}",
    )
    complete = {
        split: [row for row in records if row.split == split]
        for split in ("TRAIN", "VALIDATION", "TEST")
    }
    _require(all(complete.values()), "Development split is incomplete")
    if result_stage in {"HPO_VALIDATION_ONLY", "FROZEN_DEVELOPMENT_VALIDATION"}:
        return {split: complete[split] for split in ("TRAIN", "VALIDATION")}, len(complete["TEST"])
    if result_stage == "FINAL_ALL_DEVELOPMENT_REFIT":
        return {"TRAIN": complete["TRAIN"] + complete["VALIDATION"] + complete["TEST"]}, 0
    return {"TRAIN": complete["TRAIN"] + complete["VALIDATION"], "TEST": complete["TEST"]}, 0


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Shuffle length-local batches without padding all records to 1,874 nt."""

    def __init__(self, records: list[DeltaRecord], batch_size: int, seed: int, shuffle: bool):
        self.records = records
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        ordered = sorted(range(len(self.records)), key=lambda index: len(self.records[index].source))
        batches = [ordered[index:index + self.batch_size] for index in range(0, len(ordered), self.batch_size)]
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
            for batch in batches:
                random.Random(self.seed + self.epoch * len(batches) + batch[0]).shuffle(batch)
        yield from batches

    def __len__(self):
        return math.ceil(len(self.records) / self.batch_size)


class SourceGroupBatchSampler(Sampler[list[int]]):
    """Keep each complete source-context-endpoint pool inside one ranking batch."""

    def __init__(self, records: list[DeltaRecord], batch_size: int, seed: int, shuffle: bool):
        self.records = records
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        groups: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            groups.setdefault(record.source_group, []).append(index)
        ordered_groups = sorted(groups.values(), key=lambda indices: max(len(records[index].source) for index in indices))
        self.batches: list[list[int]] = []
        current: list[int] = []
        for indices in ordered_groups:
            if current and len(current) + len(indices) > batch_size:
                self.batches.append(current)
                current = []
            if len(indices) > batch_size:
                self.batches.append(indices)
            else:
                current.extend(indices)
        if current:
            self.batches.append(current)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        batches = [list(batch) for batch in self.batches]
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
        yield from batches

    def __len__(self):
        return len(self.batches)


def evenly_spaced_batches(
    sampler: SourceGroupBatchSampler, maximum: int
) -> list[list[int]]:
    """Select deterministic coverage across a source-group sampler."""

    _require(maximum > 0, "maximum batch count must be positive")
    batches = list(sampler.batches)
    _require(bool(batches), "task sampler has no batches")
    if len(batches) <= maximum:
        return batches
    positions = (
        [
            round(index * (len(batches) - 1) / (maximum - 1))
            for index in range(maximum)
        ]
        if maximum > 1
        else [len(batches) // 2]
    )
    return [batches[index] for index in positions]


def collate(examples: list[dict[str, Any]]) -> dict[str, Any]:
    maximum = max(len(row["source"]) for row in examples)
    source = torch.full((len(examples), maximum), PAD, dtype=torch.long)
    candidate = torch.full_like(source, PAD)
    padding = torch.ones_like(source, dtype=torch.bool)
    for index, row in enumerate(examples):
        length = len(row["source"])
        source[index, :length] = torch.tensor(row["source"])
        candidate[index, :length] = torch.tensor(row["candidate"])
        padding[index, :length] = False
    result = {
        "record_ids": [row["record_id"] for row in examples],
        "source_groups": [row["source_group"] for row in examples],
        "task_keys": [row.get("task_key", "__UNSPECIFIED_TASK__") for row in examples],
        "target_scale_sources": [row.get("target_scale_source", "NONE") for row in examples],
        "source_tokens": source,
        "candidate_tokens": candidate,
        "padding_mask": padding,
        "study_ids": torch.tensor([row["study"] for row in examples]),
        "assay_ids": torch.tensor([row["assay"] for row in examples]),
        "context_ids": torch.tensor([row["context"] for row in examples]),
        "endpoint_ids": torch.tensor([row["endpoint"] for row in examples]),
        "region_ids": torch.tensor([row["region"] for row in examples]),
        "target": torch.tensor([row["target"] for row in examples], dtype=torch.float32),
        "scaled_target": torch.tensor([row.get("scaled_target", row["target"]) for row in examples], dtype=torch.float32),
        "target_scale": torch.tensor([row.get("target_scale", 1.0) for row in examples], dtype=torch.float32),
        "sample_weight": torch.tensor([row["sample_weight"] for row in examples], dtype=torch.float32),
    }
    if "source_pretrained" in examples[0]:
        _require(
            all("source_pretrained" in row and "candidate_pretrained" in row for row in examples),
            "pretrained features are partial within a batch",
        )
        result["source_pretrained"] = torch.stack(
            [row["source_pretrained"] for row in examples]
        )
        result["candidate_pretrained"] = torch.stack(
            [row["candidate_pretrained"] for row in examples]
        )
    return result


def require_cuda(device_text: str, physical_gpu_index: int) -> torch.device:
    _require(device_text.startswith("cuda"), "Delta parameter updates require CUDA; CPU fallback is forbidden")
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden for physical-device provenance")
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    _require(0 <= physical_gpu_index < torch.cuda.device_count(), "physical GPU index is unavailable")
    device = torch.device(device_text)
    _require(device.index == physical_gpu_index, "CUDA device index differs from declared physical GPU")
    torch.cuda.set_device(device)
    return device


def _forward(model, batch):
    arguments = (
        batch["source_tokens"], batch["candidate_tokens"], batch["padding_mask"],
        batch["study_ids"], batch["assay_ids"], batch["context_ids"],
        batch["endpoint_ids"], batch["region_ids"],
    )
    if "source_pretrained" in batch:
        return model(
            *arguments, batch["source_pretrained"], batch["candidate_pretrained"]
        )
    return model(*arguments)


def _move(batch, device):
    return {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in batch.items()}


def gaussian_nll(output, target, sample_weight=None):
    inverse_variance = torch.exp(-output["log_variance"])
    per_record = 0.5 * (inverse_variance * (output["mean"] - target) ** 2 + output["log_variance"])
    if sample_weight is None:
        return per_record.mean()
    return (per_record * sample_weight).mean()


def fixed_variance_gaussian_nll(output, target, sample_weight=None):
    fixed_output = {"mean": output["mean"], "log_variance": torch.zeros_like(target)}
    return gaussian_nll(fixed_output, target, sample_weight)


def huber_loss(output, target, sample_weight=None, delta: float = 1.0):
    per_record = torch.nn.functional.huber_loss(output["mean"], target, reduction="none", delta=delta)
    if sample_weight is None:
        return per_record.mean()
    return (per_record * sample_weight).mean()


def _training_target(batch: Mapping[str, Any]) -> torch.Tensor:
    return batch.get("scaled_target", batch["target"])


def ranking_group_loss(output, batch, loss_kind: str) -> torch.Tensor | None:
    predictions = output["mean"]
    targets = _training_target(batch)
    by_group: dict[str, list[int]] = {}
    for index, group in enumerate(batch["source_groups"]):
        by_group.setdefault(group, []).append(index)
    group_losses = []
    for indices in by_group.values():
        if len(indices) < 2:
            continue
        index = torch.tensor(indices, device=predictions.device)
        group_prediction = predictions[index]
        group_target = targets[index]
        if loss_kind == "pairwise":
            differences = group_target[:, None] - group_target[None, :]
            upper = torch.triu(torch.ones_like(differences, dtype=torch.bool), diagonal=1) & differences.ne(0)
            if upper.any():
                prediction_differences = group_prediction[:, None] - group_prediction[None, :]
                group_losses.append(torch.nn.functional.softplus(-differences.sign()[upper] * prediction_differences[upper]).mean())
        elif loss_kind == "listwise":
            target_distribution = torch.softmax(group_target, dim=0)
            group_losses.append(-(target_distribution * torch.log_softmax(group_prediction, dim=0)).sum())
        else:
            raise DeltaTrainingError(f"unknown ranking loss: {loss_kind}")
    return None if not group_losses else torch.stack(group_losses).mean()


def ranking_loss(output, batch, loss_kind: str):
    group_loss = ranking_group_loss(output, batch, loss_kind)
    return huber_loss(output, _training_target(batch), batch["sample_weight"]) if group_loss is None else group_loss


def multitask_loss(output, batch, rank_kind: str, rank_weight: float, huber_delta: float) -> torch.Tensor:
    _require(rank_weight > 0.0, "multitask ranking weight must be positive")
    regression = huber_loss(output, _training_target(batch), batch["sample_weight"], huber_delta)
    ranking = ranking_group_loss(output, batch, rank_kind)
    return regression if ranking is None else regression + rank_weight * ranking


def training_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    loss_kind: str,
    ranking_loss_weight: float,
    huber_delta: float,
) -> torch.Tensor:
    """Apply the configured objective without changing the standard path."""

    if loss_kind == "huber":
        return huber_loss(
            output,
            _training_target(batch),
            batch["sample_weight"],
            huber_delta,
        )
    if loss_kind == "fixed_variance_gaussian_nll":
        return fixed_variance_gaussian_nll(
            output, _training_target(batch), batch["sample_weight"]
        )
    if loss_kind == "learned_variance_gaussian_nll":
        return gaussian_nll(
            output, _training_target(batch), batch["sample_weight"]
        )
    if loss_kind.startswith("huber_plus_"):
        return multitask_loss(
            output,
            batch,
            loss_kind.removeprefix("huber_plus_"),
            ranking_loss_weight,
            huber_delta,
        )
    return ranking_loss(output, batch, loss_kind)


def shared_effect_parameters(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    """Select the shared effect representation, excluding categorical adapters."""

    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not any(name.startswith(prefix) for prefix in SHARED_EFFECT_EXCLUDED_PREFIXES)
    ]
    _require(bool(selected), "shared effect parameter set is empty")
    return selected


def gradient_vector(
    parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> torch.Tensor:
    values = []
    for _name, parameter in parameters:
        values.append(
            parameter.grad.reshape(-1)
            if parameter.grad is not None
            else torch.zeros_like(parameter).reshape(-1)
        )
    return torch.cat(values)


def task_gradient_norm_loss_multipliers(
    norms: Mapping[str, float],
) -> dict[str, float]:
    """Equalize frozen task-gradient norms around their geometric mean."""

    _require(len(norms) >= 2, "at least two TRAIN tasks are required for calibration")
    checked = {}
    for task, value in norms.items():
        number = float(value)
        _require(math.isfinite(number) and number > 0.0, f"invalid task gradient norm: {task}")
        checked[str(task)] = number
    reference = math.exp(sum(math.log(value) for value in checked.values()) / len(checked))
    return {task: reference / value for task, value in checked.items()}


def _select_batch(batch: Mapping[str, Any], indices: list[int]) -> dict[str, Any]:
    selected = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            index = torch.tensor(indices, device=value.device)
            selected[key] = value[index]
        elif isinstance(value, list):
            selected[key] = [value[index] for index in indices]
        else:
            selected[key] = value
    return selected


def _select_output(
    output: Mapping[str, torch.Tensor], indices: list[int]
) -> dict[str, torch.Tensor]:
    return {
        key: value[torch.tensor(indices, device=value.device)]
        for key, value in output.items()
    }


def task_gradient_scaled_training_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    multipliers: Mapping[str, float],
    loss_kind: str,
    ranking_loss_weight: float,
    huber_delta: float,
) -> torch.Tensor:
    """Average task losses after applying frozen TRAIN-only norm multipliers."""

    by_task: dict[str, list[int]] = {}
    for index, task in enumerate(batch["task_keys"]):
        by_task.setdefault(str(task), []).append(index)
    _require(bool(by_task), "training batch has no task keys")
    _require(set(by_task) <= set(multipliers), "training task lacks gradient calibration")
    scaled_losses = []
    for task, indices in sorted(by_task.items()):
        task_batch = _select_batch(batch, indices)
        task_output = _select_output(output, indices)
        scaled_losses.append(
            float(multipliers[task])
            * training_loss(
                task_output,
                task_batch,
                loss_kind,
                ranking_loss_weight,
                huber_delta,
            )
        )
    return torch.stack(scaled_losses).mean()


def calibrate_task_gradient_norms(
    *,
    model: torch.nn.Module,
    records: list[DeltaRecord],
    vocabs: Mapping[str, Mapping[str, int]],
    metadata_mode: str,
    weighting_mode: str,
    target_scaler: Route2TargetScaler,
    candidate_overrides: Mapping[str, str],
    loss_kind: str,
    ranking_loss_weight: float,
    huber_delta: float,
    batch_size: int,
    seed: int,
    maximum_batches_per_task: int,
    device: torch.device,
) -> dict[str, Any]:
    """Calibrate fixed task loss multipliers before any parameter update."""

    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    records_by_task: dict[str, list[DeltaRecord]] = {}
    for record in records:
        records_by_task.setdefault(task_key(record.endpoint, record.region), []).append(record)
    _require(len(records_by_task) >= 2, "gradient calibration requires multiple TRAIN tasks")
    parameters = shared_effect_parameters(model)
    norms = {}
    task_rows = {}
    cuda_losses_verified = True
    model.train()
    for task, task_records in sorted(records_by_task.items()):
        record_ids = {record.record_id for record in task_records}
        task_overrides = {
            record_id: candidate
            for record_id, candidate in candidate_overrides.items()
            if record_id in record_ids
        }
        dataset = DeltaDataset(
            task_records,
            vocabs,
            metadata_mode=metadata_mode,
            weighting_mode=weighting_mode,
            target_scaler=target_scaler,
            candidate_overrides=task_overrides,
        )
        sampler = SourceGroupBatchSampler(
            task_records, batch_size, seed, False
        )
        batches = evenly_spaced_batches(sampler, maximum_batches_per_task)
        loader = DataLoader(
            dataset,
            batch_sampler=batches,
            collate_fn=collate,
            num_workers=0,
        )
        model.zero_grad(set_to_none=True)
        loss_values = []
        for raw_batch in loader:
            batch = _move(raw_batch, device)
            output = _forward(model, batch)
            loss = training_loss(
                output,
                batch,
                loss_kind,
                ranking_loss_weight,
                huber_delta,
            )
            _require(loss.is_cuda and loss.device == device, "calibration loss left CUDA")
            (loss / len(batches)).backward()
            loss_values.append(float(loss.detach().cpu()))
        vector = gradient_vector(parameters).detach()
        _require(vector.is_cuda and vector.device == device, "calibration gradient left CUDA")
        norm = float(torch.linalg.vector_norm(vector))
        _require(math.isfinite(norm) and norm > 0.0, f"zero or invalid gradient: {task}")
        sampled_record_count = sum(len(batch) for batch in batches)
        norms[task] = norm
        task_rows[task] = {
            "training_record_count": len(task_records),
            "sampled_batch_count": len(batches),
            "sampled_record_count": sampled_record_count,
            "sampled_record_fraction": sampled_record_count / len(task_records),
            "mean_sampled_loss": sum(loss_values) / len(loss_values),
            "shared_gradient_norm": norm,
        }
    model.zero_grad(set_to_none=True)
    _require(
        all(
            torch.equal(initial_parameters[name], parameter.detach())
            for name, parameter in model.named_parameters()
        ),
        "gradient calibration changed a model parameter",
    )
    multipliers = task_gradient_norm_loss_multipliers(norms)
    for task, multiplier in multipliers.items():
        task_rows[task]["loss_multiplier"] = multiplier
    return {
        "mode": TRAINING_UPDATE_TASK_GRADIENT_NORM_CALIBRATED,
        "fit_scope": "TRAIN_ONLY_BEFORE_FIRST_OPTIMIZER_STEP",
        "maximum_batches_per_task": maximum_batches_per_task,
        "task_count": len(task_rows),
        "shared_parameter_count": sum(parameter.numel() for _, parameter in parameters),
        "task_diagnostics": task_rows,
        "loss_multipliers": multipliers,
        "cuda_losses_verified": cuda_losses_verified,
        "optimizer_steps": 0,
        "parameter_updates": 0,
    }


def metrics(
    targets: list[float],
    predictions: list[float],
    *,
    task_keys: list[str] | None = None,
    scaled_targets: list[float] | None = None,
    scaled_predictions: list[float] | None = None,
    predicted_standard_deviations: list[float] | None = None,
) -> dict[str, Any]:
    target = np.asarray(targets)
    prediction = np.asarray(predictions)
    correlation = None
    if len(target) >= 3 and np.std(target) > 0 and np.std(prediction) > 0:
        value = float(spearmanr(target, prediction).statistic)
        correlation = value if math.isfinite(value) else None
    result = {
        "record_count": len(targets),
        "mae": float(np.mean(np.abs(prediction - target))),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "spearman": correlation,
        "target_std": float(np.std(target)),
        "prediction_std": float(np.std(prediction)),
        "prediction_std_over_target_std": float(
            np.std(prediction) / max(np.std(target), 1e-12)
        ),
    }
    if predicted_standard_deviations is not None:
        scale = np.asarray(predicted_standard_deviations)
        _require(len(scale) == len(target), "predicted scales do not align")
        residual = np.abs(prediction - target)
        scale_correlation = None
        if len(scale) >= 3 and np.std(scale) > 0 and np.std(residual) > 0:
            value = float(spearmanr(scale, residual).statistic)
            scale_correlation = value if math.isfinite(value) else None
        result.update({
            "predicted_standard_deviation_mean": float(np.mean(scale)),
            "predicted_standard_deviation_median": float(np.median(scale)),
            "predicted_standard_deviation_p90": float(np.quantile(scale, 0.9)),
            "absolute_residual_scale_spearman": scale_correlation,
        })
    if task_keys is None:
        return result
    _require(len(task_keys) == len(targets), "task keys do not align with predictions")
    if scaled_targets is None or scaled_predictions is None:
        raise DeltaTrainingError("scaled prediction metrics are missing")
    by_task: dict[str, list[int]] = {}
    for index, key in enumerate(task_keys):
        by_task.setdefault(key, []).append(index)
    task_metrics = {}
    correlations = []
    scaled_maes = []
    scaled_target = np.asarray(scaled_targets)
    scaled_prediction = np.asarray(scaled_predictions)
    for key, indices in sorted(by_task.items()):
        task_target = target[indices]
        task_prediction = prediction[indices]
        task_correlation = None
        if len(indices) >= 3 and np.std(task_target) > 0 and np.std(task_prediction) > 0:
            value = float(spearmanr(task_target, task_prediction).statistic)
            task_correlation = value if math.isfinite(value) else None
        if task_correlation is not None:
            correlations.append(task_correlation)
        task_scaled_mae = float(np.mean(np.abs(scaled_prediction[indices] - scaled_target[indices])))
        scaled_maes.append(task_scaled_mae)
        task_metrics[key] = {
            "record_count": len(indices),
            "spearman": task_correlation,
            "mae": float(np.mean(np.abs(task_prediction - task_target))),
            "standardized_mae": task_scaled_mae,
        }
    result.update({
        "task_count": len(task_metrics),
        "defined_task_spearman_count": len(correlations),
        "task_macro_spearman": None if not correlations else float(np.mean(correlations)),
        "task_macro_standardized_mae": float(np.mean(scaled_maes)),
        "task_metrics": task_metrics,
    })
    return result


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    rows = []
    targets = []
    predictions = []
    task_keys = []
    scaled_targets = []
    scaled_predictions = []
    predicted_standard_deviations = []
    uncertainty_trained = bool(getattr(model, "learned_uncertainty", False))
    for raw in loader:
        batch = _move(raw, device)
        output = _forward(model, batch)
        standardized_means = output["mean"]
        raw_means = standardized_means * batch["target_scale"]
        raw_standard_deviation = (
            torch.exp(0.5 * output["log_variance"]) * batch["target_scale"]
            if uncertainty_trained
            else None
        )
        means = raw_means.cpu().tolist()
        standardized = standardized_means.cpu().tolist()
        standard_deviations = (
            raw_standard_deviation.cpu().tolist()
            if raw_standard_deviation is not None
            else [None] * len(means)
        )
        for record_id, mean, standardized_mean, target, scale, scale_source, predicted_std in zip(
            raw["record_ids"], means, standardized, raw["target"].tolist(),
            raw["target_scale"].tolist(), raw["target_scale_sources"], standard_deviations,
        ):
            rows.append({
                "canonical_record_id": record_id,
                "predicted_direction_normalized_delta": mean,
                "predicted_standardized_delta": standardized_mean,
                "target_scale": scale,
                "target_scale_source": scale_source,
                "predicted_variance": None if predicted_std is None else predicted_std ** 2,
                "prediction_uncertainty_status": (
                    "MODEL_ALEATORIC_DIAGNOSTIC_NOT_BIOLOGICAL_STANDARD_ERROR"
                    if predicted_std is not None
                    else "NOT_TRAINED_NO_UNIVERSAL_TRUE_STANDARD_ERROR"
                ),
            })
            targets.append(target)
            predictions.append(mean)
            if predicted_std is not None:
                predicted_standard_deviations.append(predicted_std)
        task_keys.extend(raw["task_keys"])
        scaled_targets.extend(raw["scaled_target"].tolist())
        scaled_predictions.extend(standardized)
    return rows, metrics(
        targets,
        predictions,
        task_keys=task_keys,
        scaled_targets=scaled_targets,
        scaled_predictions=scaled_predictions,
        predicted_standard_deviations=(
            predicted_standard_deviations if uncertainty_trained else None
        ),
    )


def train(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"output already exists: {output_dir}")
    device = require_cuda(str(config["device"]), int(config["physical_gpu_index"]))
    baseline_id = str(config["baseline_id"])
    _require(bool(baseline_id), "baseline identity is empty")
    cuda_provenance = cuda_device_observation(
        int(config["physical_gpu_index"]), require_physical_index_match=True
    )
    manifest = load_manifest(Path(config["development_manifest"]))
    records = load_records([Path(path) for path in config["canonical_paths"]], manifest)
    model_kind = str(config.get("model_kind", ROUTE2_DELTA_MODEL_KIND))
    pretrained_features = None
    if model_kind in PRETRAINED_EDIT_CENTERED_MODEL_KINDS:
        _require("pretrained_feature_cache_path" in config, "pretrained feature cache is required")
        pretrained_features = FrozenPretrainedPairFeatures(
            Path(config["pretrained_feature_cache_path"]),
            {row.record_id for row in records},
        )
    records, included_studies, excluded_record_count = select_study_subset(
        records, config.get("included_study_unit_ids")
    )
    records, included_regions, region_excluded_record_count = select_region_subset(
        records, config.get("included_regions")
    )
    metadata_mode = str(config.get("metadata_mode", "FULL_CONTEXT"))
    _require(
        metadata_mode in {"FULL_CONTEXT", "TRANSFERABLE_CONTEXT", "SEQUENCE_AND_REGION_ONLY"},
        "unknown metadata mode",
    )
    weighting_mode = str(config.get("training_weighting_mode", "SOURCE_CONTEXT_ENDPOINT_GROUP"))
    _require(
        weighting_mode in {
            "SOURCE_CONTEXT_ENDPOINT_GROUP",
            "STUDY_THEN_SOURCE_CONTEXT_ENDPOINT_GROUP",
            "TASK_THEN_SOURCE_CONTEXT_ENDPOINT_GROUP",
        },
        "unknown training weighting mode",
    )
    run_mode = str(config.get("run_mode", "FIXED_GROUPED_SPLIT"))
    _require(run_mode in {"FIXED_GROUPED_SPLIT", "LOSO_FROZEN_HYPERPARAMETERS"}, f"unknown run mode: {run_mode}")
    result_stage = str(config.get("result_stage", ""))
    loso_holdout = None
    excluded_bridge_count = 0
    development_test_record_count_withheld = 0
    if run_mode == "FIXED_GROUPED_SPLIT":
        by_split, development_test_record_count_withheld = fixed_split_records(records, result_stage)
    else:
        _require(
            result_stage == "LOSO_FROZEN_HYPERPARAMETERS",
            f"invalid result_stage for LOSO: {result_stage}",
        )
        loso_holdout = str(config["loso_holdout_study_unit_id"])
        holdout_components = {
            value["connected_source_component_id"]
            for value in manifest.values() if value["study_unit_id"] == loso_holdout
        }
        _require(holdout_components, f"LOSO study is absent: {loso_holdout}")
        train_ids = {
            record_id for record_id, value in manifest.items()
            if value["study_unit_id"] != loso_holdout
            and value["connected_source_component_id"] not in holdout_components
        }
        test_ids = {
            record_id for record_id, value in manifest.items()
            if value["study_unit_id"] == loso_holdout
        }
        excluded_bridge_count = sum(
            value["study_unit_id"] != loso_holdout
            and value["connected_source_component_id"] in holdout_components
            for value in manifest.values()
        )
        by_split = {
            "TRAIN": [row for row in records if row.record_id in train_ids],
            "VALIDATION": [],
            "TEST": [row for row in records if row.record_id in test_ids],
        }
        _require(by_split["TRAIN"] and by_split["TEST"], "LOSO train or holdout set is empty")
    target_scaler = fit_route2_target_scaler(
        by_split["TRAIN"],
        mode=str(config.get("target_scaling_mode", TARGET_SCALING_NONE)),
        minimum_task_records=int(config.get("target_scale_minimum_task_records", 20)),
        floor=float(config.get("target_scale_floor", 1e-3)),
    )
    candidate_control = str(config.get("candidate_control", "NONE"))
    _require(
        candidate_control in {"NONE", "WITHIN_EXACT_SOURCE_TASK_TRAIN_CANDIDATE_PERMUTATION"},
        "unknown candidate control",
    )
    if candidate_control == "WITHIN_EXACT_SOURCE_TASK_TRAIN_CANDIDATE_PERMUTATION":
        training_candidate_overrides, candidate_control_summary = build_training_candidate_permutation(
            by_split["TRAIN"], int(config["seed"])
        )
        _require(
            candidate_control_summary["changed_candidate_sequence_count"] > 0,
            "candidate permutation did not change any training candidate",
        )
    else:
        training_candidate_overrides = {}
        candidate_control_summary = {
            "permutation_stratum": "NONE",
            "candidate_pool_membership_preserved": True,
            "edit_distance_multiset_preserved": True,
            "training_record_count": len(by_split["TRAIN"]),
            "permutable_record_count": 0,
            "changed_candidate_sequence_count": 0,
            "singleton_or_unpermutable_record_count": len(by_split["TRAIN"]),
        }
    if metadata_mode == "FULL_CONTEXT":
        vocabs = {field: build_vocab(by_split["TRAIN"], field) for field in ("study", "assay", "context", "endpoint")}
    elif metadata_mode == "TRANSFERABLE_CONTEXT":
        vocabs = {
            "study": {"__UNK__": 0},
            **{field: build_vocab(by_split["TRAIN"], field) for field in ("assay", "context", "endpoint")},
        }
    else:
        vocabs = {field: {"__UNK__": 0} for field in ("study", "assay", "context", "endpoint")}
    datasets = {
        split: DeltaDataset(
            rows,
            vocabs,
            metadata_mode=metadata_mode,
            weighting_mode=weighting_mode,
            target_scaler=target_scaler,
            candidate_overrides=training_candidate_overrides if split == "TRAIN" else None,
            pretrained_features=pretrained_features,
        )
        for split, rows in by_split.items() if rows
    }
    loss_kind = str(config.get("loss_kind", "huber"))
    allowed_losses = {
        "huber",
        "fixed_variance_gaussian_nll",
        "learned_variance_gaussian_nll",
        "pairwise",
        "listwise",
        "huber_plus_pairwise",
        "huber_plus_listwise",
    }
    _require(loss_kind in allowed_losses, f"unknown or unauthorized loss_kind: {loss_kind}")
    ranking_loss_weight = float(config.get("ranking_loss_weight", 1.0))
    huber_delta = float(config.get("huber_delta", 1.0))
    training_update_mode = str(
        config.get("training_update_mode", TRAINING_UPDATE_STANDARD)
    )
    _require(
        training_update_mode in {
            TRAINING_UPDATE_STANDARD,
            TRAINING_UPDATE_TASK_GRADIENT_NORM_CALIBRATED,
        },
        "unknown training update mode",
    )
    sampler_class = SourceGroupBatchSampler if "pairwise" in loss_kind or "listwise" in loss_kind else LengthBucketBatchSampler
    samplers = {
        split: sampler_class(by_split[split], int(config["batch_size"]), int(config["seed"]), split == "TRAIN")
        for split in datasets
    }
    loaders = {
        split: DataLoader(datasets[split], batch_sampler=samplers[split], collate_fn=collate, num_workers=int(config.get("num_workers", 0)))
        for split in datasets
    }
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    shared_model_config = {
        "hidden_dim": int(config["hidden_dim"]), "depth": int(config["depth"]),
        "study_count": len(vocabs["study"]), "assay_count": len(vocabs["assay"]),
        "context_count": len(vocabs["context"]), "endpoint_count": len(vocabs["endpoint"]),
    }
    if model_kind == ROUTE2_DELTA_MODEL_KIND:
        checkpoint_model_config = {
            **shared_model_config,
            "study_specific_scale_calibration": bool(config.get("study_specific_scale_calibration", False)),
        }
        model = Route2DeltaPredictor(**checkpoint_model_config).to(device)
    elif model_kind in {ROUTE2_EDIT_CENTERED_MODEL_KIND, ROUTE2_EDIT_CENTERED_SOURCE_ONLY_KIND}:
        checkpoint_model_config = {
            **shared_model_config,
            "source_only_control": model_kind == ROUTE2_EDIT_CENTERED_SOURCE_ONLY_KIND,
        }
        model = Route2EditCenteredDeltaPredictor(**checkpoint_model_config).to(device)
    elif model_kind in PRETRAINED_EDIT_CENTERED_MODEL_KINDS:
        _require(pretrained_features is not None, "pretrained features were not loaded")
        checkpoint_model_config = {
            **shared_model_config,
            "pretrained_width": pretrained_features.width,
            "learned_uncertainty": loss_kind == "learned_variance_gaussian_nll",
        }
        model = Route2PretrainedEditCenteredDeltaPredictor(
            **checkpoint_model_config
        ).to(device)
    elif model_kind in Route2NeuralBaseline.MODES:
        checkpoint_model_config = {
            **shared_model_config,
            "mode": model_kind,
            "max_length": int(config.get("max_length", 2048)),
        }
        model = Route2NeuralBaseline(**checkpoint_model_config).to(device)
    else:
        raise DeltaTrainingError(f"unknown model_kind: {model_kind}")
    if loss_kind == "learned_variance_gaussian_nll":
        _require(
            model_kind in PRETRAINED_EDIT_CENTERED_MODEL_KINDS,
            "learned uncertainty is scoped to the matched pretrained critic comparison",
        )
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    expected_trainable = config.get("expected_trainable_parameter_count")
    if expected_trainable is not None:
        tolerance = float(config.get("parameter_count_relative_tolerance", 0.02))
        _require(
            abs(trainable_parameter_count - int(expected_trainable))
            <= tolerance * int(expected_trainable),
            "trainable parameter count differs from the configured max profile",
        )
    if training_update_mode == TRAINING_UPDATE_TASK_GRADIENT_NORM_CALIBRATED:
        _require(
            model_kind in {
                ROUTE2_EDIT_CENTERED_MODEL_KIND,
                ROUTE2_EDIT_CENTERED_SOURCE_ONLY_KIND,
            },
            "task-gradient calibration is scoped to edit-centered models",
        )
        _require(
            weighting_mode == "TASK_THEN_SOURCE_CONTEXT_ENDPOINT_GROUP",
            "task-gradient calibration requires task-balanced source-group weights",
        )
        _require(
            target_scaler.mode == TARGET_SCALING_TRAIN_TASK_ROBUST,
            "task-gradient calibration requires TRAIN task-robust targets",
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    initial_parameter = next(model.parameters()).detach().clone()
    history = []
    optimizer_steps = 0
    cuda_training_tensors_verified = False
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    serialized_config = json.dumps(dict(config), indent=2, sort_keys=True) + "\n"
    (output_dir / "training_config.json").write_text(serialized_config, encoding="utf-8")
    (output_dir / "config.yaml").write_text(serialized_config, encoding="utf-8")
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    log_path = output_dir / "train.log"
    started = time.time()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    attempt_details = {
        "started_at": started_at,
        "included_study_unit_ids": included_studies,
        "included_regions": included_regions,
        "record_counts": {split: len(rows) for split, rows in by_split.items()},
        "development_test_record_count_withheld": development_test_record_count_withheld,
        "evaluation_record_count": 0,
        "trainable_parameter_count": trainable_parameter_count,
        "frozen_pretrained_parameter_count": (
            pretrained_features.pretrained_parameter_count
            if pretrained_features is not None
            else 0
        ),
        "total_effective_parameter_count": trainable_parameter_count + (
            pretrained_features.pretrained_parameter_count
            if pretrained_features is not None
            else 0
        ),
        "pretrained_model_id": (
            pretrained_features.model_id if pretrained_features is not None else None
        ),
    }
    if config.get("experiment_ledger_path"):
        record_training_attempt(
            Path(config["experiment_ledger_path"]),
            output_dir / "training_attempt.json",
            build_training_attempt_row(
                config,
                output_dir,
                "RUNNING",
                repository_root=REPO_ROOT,
                details=attempt_details,
            ),
        )
    log_path.write_text(
        json.dumps({
            "event": "TRAINING_STARTED",
            "device": str(device),
            "physical_gpu_index": int(config["physical_gpu_index"]),
            "cuda_device_uuid": cuda_provenance["cuda_device_uuid"],
            "result_stage": result_stage,
            "run_mode": run_mode,
            "training_update_mode": training_update_mode,
        }, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    task_gradient_calibration = None
    task_gradient_loss_multipliers = None
    if training_update_mode == TRAINING_UPDATE_TASK_GRADIENT_NORM_CALIBRATED:
        task_gradient_calibration = calibrate_task_gradient_norms(
            model=model,
            records=by_split["TRAIN"],
            vocabs=vocabs,
            metadata_mode=metadata_mode,
            weighting_mode=weighting_mode,
            target_scaler=target_scaler,
            candidate_overrides=training_candidate_overrides,
            loss_kind=loss_kind,
            ranking_loss_weight=ranking_loss_weight,
            huber_delta=huber_delta,
            batch_size=int(config["batch_size"]),
            seed=int(config["seed"]),
            maximum_batches_per_task=int(
                config.get("task_gradient_calibration_max_batches_per_task", 16)
            ),
            device=device,
        )
        task_gradient_loss_multipliers = task_gradient_calibration[
            "loss_multipliers"
        ]
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "event": "TASK_GRADIENT_CALIBRATION_COMPLETED",
                "training_update_mode": training_update_mode,
                "task_gradient_calibration": task_gradient_calibration,
            }, sort_keys=True) + "\n")
    checkpoint_selection = str(config.get("checkpoint_selection", "FINAL_EPOCH"))
    _require(
        checkpoint_selection in {"FINAL_EPOCH", "BEST_VALIDATION"},
        "unknown checkpoint selection rule",
    )
    checkpoint_metric = str(config.get("checkpoint_metric", "GLOBAL_SPEARMAN_THEN_MAE"))
    _require(
        checkpoint_metric in {
            "GLOBAL_SPEARMAN_THEN_MAE",
            "TASK_MACRO_SPEARMAN_THEN_STANDARDIZED_MAE",
        },
        "unknown checkpoint metric",
    )
    _require(
        checkpoint_selection != "BEST_VALIDATION" or "VALIDATION" in loaders,
        "best-validation checkpoint selection requires a validation split",
    )

    def checkpoint_payload(epoch: int) -> dict[str, Any]:
        return {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_kind": model_kind,
            "baseline_id": baseline_id,
            "model_config": checkpoint_model_config,
            "vocabs": vocabs,
            "target_scaler": target_scaler.to_dict(),
            "completed_epoch": epoch,
            "candidate_control": candidate_control,
            "training_provenance": {
                "optimizer_steps": optimizer_steps,
                "parameter_changed": not torch.equal(initial_parameter, next(model.parameters()).detach()),
                "device": str(device),
                "physical_gpu_index": int(config["physical_gpu_index"]),
                "cpu_fallback_used": False,
                "cuda_training_tensors_verified": cuda_training_tensors_verified,
                "included_study_unit_ids": included_studies,
                "included_regions": included_regions,
                "metadata_mode": metadata_mode,
                "training_weighting_mode": weighting_mode,
                "training_update_mode": training_update_mode,
                "task_gradient_calibration": task_gradient_calibration,
                "target_scaling_mode": target_scaler.mode,
                "target_scaler_fit_scope": "TRAIN_ONLY",
                "candidate_control": candidate_control,
                "result_stage": result_stage,
                **cuda_provenance,
            },
        }

    best_rank: tuple[float, ...] | None = None
    for epoch in range(int(config["epochs"])):
        samplers["TRAIN"].set_epoch(epoch)
        model.train()
        losses = []
        for raw in loaders["TRAIN"]:
            batch = _move(raw, device)
            _require(
                batch["source_tokens"].device == device and batch["target"].device == device,
                "Delta training inputs left CUDA",
            )
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, batch)
            if training_update_mode == TRAINING_UPDATE_TASK_GRADIENT_NORM_CALIBRATED:
                _require(
                    task_gradient_loss_multipliers is not None,
                    "task gradient calibration is missing",
                )
                loss = task_gradient_scaled_training_loss(
                    output,
                    batch,
                    task_gradient_loss_multipliers,
                    loss_kind,
                    ranking_loss_weight,
                    huber_delta,
                )
            else:
                loss = training_loss(
                    output,
                    batch,
                    loss_kind,
                    ranking_loss_weight,
                    huber_delta,
                )
            _require(loss.is_cuda and loss.device == device, "Delta training loss left CUDA")
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            cuda_training_tensors_verified = True
            losses.append(float(loss.detach().cpu()))
        validation_metrics = None
        if "VALIDATION" in loaders:
            _validation_rows, validation_metrics = predict(model, loaders["VALIDATION"], device)
        epoch_row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation": validation_metrics}
        history.append(epoch_row)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_row, sort_keys=True) + "\n")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "EPOCH_COMPLETED", **epoch_row}, sort_keys=True) + "\n")
        torch.save(checkpoint_payload(epoch + 1), output_dir / "latest.pt")
        if validation_metrics is None:
            rank = (epoch_row["train_loss"],)
        elif checkpoint_metric == "TASK_MACRO_SPEARMAN_THEN_STANDARDIZED_MAE":
            task_spearman = validation_metrics["task_macro_spearman"]
            if task_spearman is None:
                rank = (1.0, validation_metrics["task_macro_standardized_mae"])
            else:
                rank = (
                    0.0,
                    -task_spearman,
                    validation_metrics["task_macro_standardized_mae"],
                )
        elif validation_metrics["spearman"] is None:
            rank = (1.0, validation_metrics["mae"])
        else:
            rank = (0.0, -validation_metrics["spearman"], validation_metrics["mae"])
        if best_rank is None or rank < best_rank:
            best_rank = rank
            torch.save(checkpoint_payload(epoch + 1), output_dir / "best.pt")
    changed = not torch.equal(initial_parameter, next(model.parameters()).detach())
    _require(optimizer_steps > 0 and changed, "no learned GPU parameter update occurred")
    selected_checkpoint_path = output_dir / (
        "best.pt" if checkpoint_selection == "BEST_VALIDATION" else "latest.pt"
    )
    selected_checkpoint = torch.load(
        selected_checkpoint_path, map_location=device, weights_only=False
    )
    selected_epoch = int(selected_checkpoint["completed_epoch"])
    model.load_state_dict(selected_checkpoint["model_state"])
    if "VALIDATION" in loaders:
        validation_rows, validation_metrics = predict(model, loaders["VALIDATION"], device)
    else:
        validation_rows, validation_metrics = [], None
    if "TEST" in loaders:
        test_rows, test_metrics = predict(model, loaders["TEST"], device)
    else:
        test_rows, test_metrics = [], None
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    summary = {
        "schema_version": "route_a_v3_route2_delta_predictor_training.v1",
        "status": "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
        "seed": int(config["seed"]),
        "baseline_id": baseline_id,
        "model_kind": model_kind,
        "study_specific_scale_calibration": bool(config.get("study_specific_scale_calibration", False)),
        "loss_kind": loss_kind,
        "ranking_loss_weight": float(config.get("ranking_loss_weight", 1.0)) if loss_kind.startswith("huber_plus_") else None,
        "run_mode": run_mode,
        "included_study_unit_ids": included_studies,
        "study_subset_excluded_record_count": excluded_record_count,
        "included_regions": included_regions,
        "region_subset_excluded_record_count": region_excluded_record_count,
        "metadata_mode": metadata_mode,
        "training_weighting_mode": weighting_mode,
        "training_update_mode": training_update_mode,
        "task_gradient_calibration": task_gradient_calibration,
        "target_scaler": target_scaler.to_dict(),
        "candidate_control": candidate_control,
        "candidate_control_summary": candidate_control_summary,
        "checkpoint_selection": checkpoint_selection,
        "checkpoint_metric": checkpoint_metric,
        "selected_checkpoint": selected_checkpoint_path.name,
        "selected_epoch": selected_epoch,
        "final_training_epoch": int(config["epochs"]),
        "result_stage": result_stage,
        "development_test_outcomes_evaluated": result_stage == "FROZEN_DEVELOPMENT_TEST",
        "development_test_record_count_withheld": development_test_record_count_withheld,
        "development_validation_folded_into_training": result_stage in {
            "FROZEN_DEVELOPMENT_TEST",
            "FINAL_ALL_DEVELOPMENT_REFIT",
        },
        "loso_holdout_study_unit_id": loso_holdout,
        "loso_excluded_connected_other_study_record_count": excluded_bridge_count,
        "physical_gpu_index": int(config["physical_gpu_index"]),
        "device": str(device),
        "cpu_fallback_used": False,
        "cuda_training_tensors_verified": cuda_training_tensors_verified,
        "parameter_count": parameter_count,
        "trainable_parameter_count": parameter_count,
        "frozen_pretrained_parameter_count": (
            pretrained_features.pretrained_parameter_count
            if pretrained_features is not None
            else 0
        ),
        "total_effective_parameter_count": parameter_count + (
            pretrained_features.pretrained_parameter_count
            if pretrained_features is not None
            else 0
        ),
        "pretrained_model_id": (
            pretrained_features.model_id if pretrained_features is not None else None
        ),
        "optimizer_steps": optimizer_steps,
        "parameter_changed": changed,
        "record_counts": {split: len(rows) for split, rows in by_split.items()},
        "maximum_sequence_length": max(len(row.source) for row in records),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "wall_time_seconds": time.time() - started,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        **cuda_provenance,
        "swap_antisymmetry_by_construction": model_kind in {
            ROUTE2_DELTA_MODEL_KIND,
            ROUTE2_EDIT_CENTERED_MODEL_KIND,
            *PRETRAINED_EDIT_CENTERED_MODEL_KINDS,
        },
        "identity_zero_by_construction": model_kind in {
            ROUTE2_DELTA_MODEL_KIND,
            ROUTE2_EDIT_CENTERED_MODEL_KIND,
            *PRETRAINED_EDIT_CENTERED_MODEL_KINDS,
        },
        "edit_centered_pooling": model_kind in {
            ROUTE2_EDIT_CENTERED_MODEL_KIND,
            ROUTE2_EDIT_CENTERED_SOURCE_ONLY_KIND,
            *PRETRAINED_EDIT_CENTERED_MODEL_KINDS,
        },
        "study_identity_used_by_effect_encoder": model_kind not in {
            ROUTE2_EDIT_CENTERED_MODEL_KIND,
            ROUTE2_EDIT_CENTERED_SOURCE_ONLY_KIND,
            *PRETRAINED_EDIT_CENTERED_MODEL_KINDS,
        } and metadata_mode == "FULL_CONTEXT",
        "uncertainty_head_used": bool(getattr(model, "learned_uncertainty", False)),
        "prediction_uncertainty_status": (
            "MODEL_ALEATORIC_DIAGNOSTIC_NOT_BIOLOGICAL_STANDARD_ERROR"
            if bool(getattr(model, "learned_uncertainty", False))
            else "NOT_TRAINED_NO_UNIVERSAL_TRUE_STANDARD_ERROR"
        ),
        "training_weight_unit": weighting_mode,
        "evaluation_outcomes_read": 0,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }
    selected_checkpoint = dict(selected_checkpoint)
    selected_provenance = dict(selected_checkpoint["training_provenance"])
    selected_provenance.update({
        "checkpoint_selection": checkpoint_selection,
        "checkpoint_metric": checkpoint_metric,
        "selected_epoch": selected_epoch,
        "total_run_optimizer_steps": optimizer_steps,
    })
    selected_checkpoint["training_provenance"] = selected_provenance
    selected_checkpoint["selection_provenance"] = {
        "checkpoint_selection": checkpoint_selection,
        "checkpoint_metric": checkpoint_metric,
        "selected_checkpoint": selected_checkpoint_path.name,
        "selected_epoch": selected_epoch,
        "final_training_epoch": int(config["epochs"]),
    }
    torch.save(selected_checkpoint, output_dir / "delta_predictor_checkpoint.pt")
    if validation_rows:
        (output_dir / "validation_predictions.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in validation_rows), encoding="utf-8"
        )
    if test_rows:
        (output_dir / "test_predictions.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in test_rows), encoding="utf-8"
        )
    serialized_summary = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (output_dir / "training_summary.json").write_text(serialized_summary, encoding="utf-8")
    (output_dir / "final_summary.json").write_text(serialized_summary, encoding="utf-8")
    if config.get("experiment_ledger_path"):
        record_training_attempt(
            Path(config["experiment_ledger_path"]),
            output_dir / "training_attempt.json",
            build_training_attempt_row(
                config,
                output_dir,
                "COMPLETED",
                repository_root=REPO_ROOT,
                details={**attempt_details, **summary},
            ),
        )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "event": "TRAINING_COMPLETED",
            "optimizer_steps": optimizer_steps,
            "wall_time_seconds": summary["wall_time_seconds"],
        }, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or Path(config["output_directory"])
    try:
        result = train(config, output_dir)
    except Exception as exc:
        write_gpu_failure_evidence(
            output_dir.with_name(output_dir.name + ".failed.json"), config, exc,
            entrypoint="train_route2_delta_predictor_v1",
            evaluation_outcomes_accessed=config.get("evaluation_outcomes_accessed", False),
        )
        if config.get("experiment_ledger_path"):
            record_training_attempt(
                Path(config["experiment_ledger_path"]),
                output_dir / "training_attempt.json",
                build_training_attempt_row(
                    config,
                    output_dir,
                    "FAILED",
                    repository_root=REPO_ROOT,
                    details={
                        "evaluation_record_count": 0,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                ),
            )
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
