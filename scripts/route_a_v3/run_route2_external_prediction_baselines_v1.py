#!/usr/bin/env python3
"""Run task-matched external 5'UTR MRL baselines on the frozen Route 2 split."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_gpu_failure_evidence import cuda_device_observation, write_gpu_failure_evidence


BASES = "ACGT"
TASK_STUDY = "GSE114002"
TASK_REGION = "5UTR"
TASK_ENDPOINT = "MEAN_RIBOSOME_LOAD"
SPLITS = ("TRAIN", "VALIDATION", "TEST")


class ExternalBaselineError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExternalBaselineError(message)


@dataclass(frozen=True)
class TaskRecord:
    record_id: str
    source_id: str
    source: str
    candidate: str
    target: float
    split: str


def _finite(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    result = float(value)
    _require(math.isfinite(result), f"{label} is not finite")
    return result


def splits_for_result_stage(result_stage: str) -> tuple[str, ...]:
    _require(
        result_stage in {"HPO_VALIDATION_ONLY", "FROZEN_DEVELOPMENT_TEST"},
        f"invalid result_stage: {result_stage}",
    )
    return ("TRAIN", "VALIDATION") if result_stage == "HPO_VALIDATION_ONLY" else SPLITS


def load_task_records(
    canonical_path: Path,
    manifest_path: Path,
    included_splits: tuple[str, ...] = SPLITS,
) -> tuple[list[TaskRecord], list[dict[str, Any]]]:
    manifest_rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    complete_manifest = {
        row["canonical_record_id"]: row for row in manifest_rows
        if row["study_unit_id"] == TASK_STUDY
    }
    _require(complete_manifest, "GSE114002 is absent from Development manifest")
    _require(all(row["split"] in SPLITS for row in complete_manifest.values()), "task manifest split changed")
    _require(set(included_splits) <= set(SPLITS), "included task splits are invalid")
    selected_manifest = {
        record_id: row for record_id, row in complete_manifest.items()
        if row["split"] in included_splits
    }
    records = []
    seen: set[str] = set()
    with canonical_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            record_id = str(row["canonical_record_id"])
            if record_id not in selected_manifest:
                continue
            _require(row["study_unit_id"] == TASK_STUDY, "task study changed")
            _require(row["region"] == TASK_REGION and row["endpoint_id"] == TASK_ENDPOINT, "task endpoint changed")
            _require(row["pool_assignment"] == "DEVELOPMENT", "external baseline read non-Development record")
            source, candidate = str(row["source_sequence"]).upper(), str(row["candidate_sequence"]).upper()
            _require(len(source) == len(candidate) == 50, "5'UTR external model requires exact 50-nt pairs")
            _require(not ((set(source) | set(candidate)) - set(BASES)), "task sequence alphabet changed")
            _require(record_id not in seen, "canonical record is duplicated")
            seen.add(record_id)
            records.append(TaskRecord(
                record_id=record_id,
                source_id=str(row["source_id"]),
                source=source,
                candidate=candidate,
                target=_finite(row["direction_normalized_delta"], "direction-normalized delta"),
                split=str(selected_manifest[record_id]["split"]),
            ))
    _require(seen == set(selected_manifest), "canonical records do not exactly cover task manifest")
    _require(all(any(record.split == split for record in records) for split in included_splits), "task split is incomplete")
    task_manifest = [row for row in manifest_rows if row["canonical_record_id"] in seen]
    return records, task_manifest


def one_hot(sequences: Iterable[str], device: torch.device) -> torch.Tensor:
    mapping = torch.full((256,), -1, dtype=torch.long)
    for index, base in enumerate(BASES):
        mapping[ord(base)] = index
    encoded = []
    for sequence in sequences:
        values = mapping[torch.tensor(list(sequence.encode("ascii")), dtype=torch.long)]
        _require(bool(torch.all(values >= 0)), "sequence contains an unsupported base")
        encoded.append(F.one_hot(values, num_classes=4).to(torch.float32))
    return torch.stack(encoded).to(device)


def _load_array(handle: h5py.File, path: str) -> torch.Tensor:
    _require(path in handle, f"weight dataset absent: {path}")
    return torch.from_numpy(np.asarray(handle[path], dtype=np.float32))


def _same_conv(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    kernel = int(weight.shape[-1])
    total = kernel - 1
    return F.conv1d(F.pad(x, (total // 2, total - total // 2)), weight, bias)


class Optimus5Prime(torch.nn.Module):
    """PyTorch inference port of the official Keras 2.1.3 50-nt MRL model."""

    def __init__(self, weight_path: Path):
        super().__init__()
        with h5py.File(weight_path, "r") as handle:
            def conv(name: str) -> tuple[torch.Tensor, torch.Tensor]:
                kernel = _load_array(handle, f"model_weights/{name}/{name}/kernel:0").permute(2, 1, 0)
                bias = _load_array(handle, f"model_weights/{name}/{name}/bias:0")
                return kernel, bias
            self.register_buffer("conv1_weight", conv("conv1d_1")[0]); self.register_buffer("conv1_bias", conv("conv1d_1")[1])
            self.register_buffer("conv2_weight", conv("conv1d_2")[0]); self.register_buffer("conv2_bias", conv("conv1d_2")[1])
            self.register_buffer("conv3_weight", conv("conv1d_3")[0]); self.register_buffer("conv3_bias", conv("conv1d_3")[1])
            self.register_buffer("dense1_weight", _load_array(handle, "model_weights/dense_1/dense_1/kernel:0").T)
            self.register_buffer("dense1_bias", _load_array(handle, "model_weights/dense_1/dense_1/bias:0"))
            self.register_buffer("dense2_weight", _load_array(handle, "model_weights/dense_2/dense_2/kernel:0").T)
            self.register_buffer("dense2_bias", _load_array(handle, "model_weights/dense_2/dense_2/bias:0"))
        _require(tuple(self.conv1_weight.shape) == (120, 4, 8), "Optimus conv1 geometry changed")
        _require(tuple(self.dense1_weight.shape) == (40, 6000), "Optimus dense geometry changed")

    def forward(self, sequence_one_hot: torch.Tensor) -> torch.Tensor:
        x = sequence_one_hot.transpose(1, 2)
        x = torch.relu(_same_conv(x, self.conv1_weight, self.conv1_bias))
        x = torch.relu(_same_conv(x, self.conv2_weight, self.conv2_bias))
        x = torch.relu(_same_conv(x, self.conv3_weight, self.conv3_bias))
        x = x.transpose(1, 2).reshape(x.shape[0], -1)
        x = torch.relu(F.linear(x, self.dense1_weight, self.dense1_bias))
        return F.linear(x, self.dense2_weight, self.dense2_bias).squeeze(-1)


class FramePool(torch.nn.Module):
    """PyTorch inference port of official Framepool_combined_residual.h5."""

    def __init__(self, weight_path: Path):
        super().__init__()
        with h5py.File(weight_path, "r") as handle:
            def conv(index: int) -> tuple[torch.Tensor, torch.Tensor]:
                prefix = f"model_weights/convolution_{index}/convolution_{index}_2"
                return _load_array(handle, prefix + "/kernel:0").permute(2, 1, 0), _load_array(handle, prefix + "/bias:0")
            for index in range(3):
                weight, bias = conv(index)
                self.register_buffer(f"conv{index}_weight", weight)
                self.register_buffer(f"conv{index}_bias", bias)
            self.register_buffer("dense_weight", _load_array(handle, "model_weights/fully_connected_0/fully_connected_0_2/kernel:0").T)
            self.register_buffer("dense_bias", _load_array(handle, "model_weights/fully_connected_0/fully_connected_0_2/bias:0"))
            self.register_buffer("output_weight", _load_array(handle, "model_weights/mrl_output_unscaled/mrl_output_unscaled_2/kernel:0").T)
            self.register_buffer("output_bias", _load_array(handle, "model_weights/mrl_output_unscaled/mrl_output_unscaled_2/bias:0"))
            self.register_buffer("scaling_weight", _load_array(handle, "model_weights/scaling_regression/scaling_regression_1/kernel:0"))
        _require(tuple(self.conv0_weight.shape) == (128, 4, 7), "FramePool conv0 geometry changed")
        _require(tuple(self.dense_weight.shape) == (64, 768), "FramePool dense geometry changed")
        _require(tuple(self.scaling_weight.shape) == (4, 1), "FramePool scaling geometry changed")

    def forward(self, sequence_one_hot: torch.Tensor) -> torch.Tensor:
        mask = sequence_one_hot.sum(dim=2).unsqueeze(1)
        x = sequence_one_hot.transpose(1, 2)
        x = torch.relu(_same_conv(x, self.conv0_weight, self.conv0_bias)) * mask
        shortcut = x
        x = torch.relu(_same_conv(x, self.conv1_weight, self.conv1_bias)) * mask + shortcut
        shortcut = x
        x = torch.relu(_same_conv(x, self.conv2_weight, self.conv2_bias)) * mask + shortcut
        x, mask = torch.flip(x, dims=(2,)), torch.flip(mask, dims=(2,))
        pooled = []
        frames = []
        for shift in range(3):
            frame, frame_mask = x[:, :, shift::3], mask[:, :, shift::3]
            frames.append((frame, frame_mask))
            pooled.append(frame.max(dim=2).values)
        for frame, frame_mask in frames:
            pooled.append(frame.sum(dim=2) / frame_mask.sum(dim=2).clamp_min(1.0))
        hidden = torch.relu(F.linear(torch.cat(pooled, dim=1), self.dense_weight, self.dense_bias))
        raw = F.linear(hidden, self.output_weight, self.output_bias)
        indicator = torch.zeros((raw.shape[0], 2), dtype=raw.dtype, device=raw.device)
        indicator[:, 1] = 1.0
        regression = torch.cat((raw * indicator, indicator), dim=1)
        return (regression @ self.scaling_weight).squeeze(-1)


def _predict_native(model: torch.nn.Module, records: list[TaskRecord], device: torch.device, batch_size: int) -> dict[str, float]:
    model.eval()
    result = {}
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            sources = one_hot((record.source for record in batch), device)
            candidates = one_hot((record.candidate for record in batch), device)
            values = (model(candidates) - model(sources)).detach().cpu().numpy()
            result.update({record.record_id: float(value) for record, value in zip(batch, values)})
    return result


def _source_group_weights(records: list[TaskRecord], device: torch.device) -> torch.Tensor:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.source_id] = counts.get(record.source_id, 0) + 1
    scale = len(records) / len(counts)
    return torch.tensor([scale / counts[record.source_id] for record in records], dtype=torch.float32, device=device)


def _multimolecule_rnafm_embeddings(
    sequences: list[str], model_path: Path, device: torch.device, batch_size: int,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, torch.Tensor], int]:
    try:
        from multimolecule.models.rnafm import RnaFmModel, RnaTokenizer
    except ImportError as exc:
        raise ExternalBaselineError("MultiMolecule RNA-FM conversion classes are unavailable") from exc
    tokenizer = RnaTokenizer.from_pretrained(model_path, local_files_only=True)
    model = RnaFmModel.from_pretrained(model_path, local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    _require(parameter_count > 90_000_000, "RNA-FM pretrained model geometry changed")
    result: dict[str, torch.Tensor] = {}
    unique = sorted(set(sequences))
    with torch.no_grad():
        for start in range(0, len(unique), batch_size):
            batch = unique[start:start + batch_size]
            tokens = tokenizer([sequence.replace("T", "U") for sequence in batch], padding=True, return_tensors="pt")
            tokens = {key: value.to(device) for key, value in tokens.items()}
            output = model(**tokens).last_hidden_state
            attention = tokens["attention_mask"].bool()
            special = torch.zeros_like(attention)
            special[:, 0] = True
            lengths = attention.sum(dim=1)
            special[torch.arange(len(batch), device=device), lengths - 1] = True
            keep = attention & ~special
            pooled = (output * keep.unsqueeze(-1)).sum(dim=1) / keep.sum(dim=1, keepdim=True).clamp_min(1)
            _require(
                pooled.is_cuda and pooled.device == device and torch.isfinite(pooled).all().item(),
                "MultiMolecule RNA-FM embedding left CUDA or became nonfinite",
            )
            for sequence, embedding in zip(batch, pooled):
                result[sequence] = embedding.detach()
            if progress is not None:
                progress({
                    "event": "RNAFM_EMBEDDING_BATCH_COMPLETED",
                    "completed_sequence_count": min(start + len(batch), len(unique)),
                    "total_sequence_count": len(unique),
                })
    del model
    torch.cuda.empty_cache()
    return result, parameter_count


def _train_multimolecule_rnafm_probe(
    records: list[TaskRecord], embeddings: Mapping[str, torch.Tensor], device: torch.device,
    seed: int, epochs: int, learning_rate: float, weight_decay: float, result_stage: str,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    by_split = {
        split: [record for record in records if record.split == split]
        for split in SPLITS if any(record.split == split for record in records)
    }
    train = by_split["TRAIN"]
    fit = train if result_stage == "HPO_VALIDATION_ONLY" else train + by_split["VALIDATION"]
    features = {
        split: torch.stack([embeddings[record.candidate] - embeddings[record.source] for record in rows]).to(device)
        for split, rows in by_split.items()
    }
    targets = {
        split: torch.tensor([record.target for record in rows], dtype=torch.float32, device=device)
        for split, rows in by_split.items()
    }
    fit_features = torch.stack([embeddings[record.candidate] - embeddings[record.source] for record in fit]).to(device)
    mean, std = fit_features.mean(dim=0), fit_features.std(dim=0).clamp_min(1e-6)
    features = {split: (value - mean) / std for split, value in features.items()}
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    head = torch.nn.Linear(features["TRAIN"].shape[1], 1).to(device)
    _require(next(head.parameters()).is_cuda and next(head.parameters()).device == device, "RNA-FM probe head left CUDA")
    initial = head.weight.detach().clone()
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    fit_feature_values = (fit_features - mean) / std
    fit_targets = torch.tensor([record.target for record in fit], dtype=torch.float32, device=device)
    weights = _source_group_weights(fit, device)
    validation_weights = _source_group_weights(by_split["VALIDATION"], device)
    best_state = None; best_validation = float("inf"); history = []
    for epoch in range(epochs):
        head.train(); optimizer.zero_grad(set_to_none=True)
        prediction = head(fit_feature_values).squeeze(1)
        loss = ((prediction - fit_targets) ** 2 * weights).mean()
        _require(loss.is_cuda and loss.device == device and torch.isfinite(loss).item(), "RNA-FM probe loss left CUDA or became nonfinite")
        loss.backward(); optimizer.step()
        head.eval()
        if result_stage == "HPO_VALIDATION_ONLY":
            with torch.no_grad():
                validation_error = head(features["VALIDATION"]).squeeze(1) - targets["VALIDATION"]
                validation = ((validation_error ** 2) * validation_weights).mean().item()
            if validation < best_validation:
                best_validation = validation
                best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
        else:
            validation = None
        if epoch == 0 or (epoch + 1) % 10 == 0:
            row = {
                "epoch": epoch + 1,
                "train_source_group_weighted_mse": float(loss.detach()),
                "validation_source_group_weighted_mse": validation,
            }
            history.append(row)
            if progress is not None:
                progress({"event": "RNAFM_PROBE_EPOCH_COMPLETED", **row})
    if result_stage == "FROZEN_DEVELOPMENT_TEST":
        best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
        best_validation = None
    _require(best_state is not None and not torch.equal(initial, head.weight.detach()), "MultiMolecule RNA-FM probe had no GPU parameter update")
    head.load_state_dict(best_state); head.eval()
    predictions = {}
    with torch.no_grad():
        for split, rows in by_split.items():
            values = head(features[split]).squeeze(1).cpu().numpy()
            predictions.update({record.record_id: float(value) for record, value in zip(rows, values)})
    artifact = {
        "feature_mean": mean.detach().cpu(),
        "feature_std": std.detach().cpu(),
        "probe_state": {key: value.detach().cpu() for key, value in best_state.items()},
    }
    return predictions, {
        "probe_parameter_count": sum(parameter.numel() for parameter in head.parameters()),
        "probe_optimizer_steps": epochs,
        "probe_parameter_changed": True,
        "best_validation_source_group_weighted_mse": best_validation,
        "development_validation_folded_into_probe_training": result_stage == "FROZEN_DEVELOPMENT_TEST",
        "history": history,
    }, artifact


def _train_multimolecule_rnafm_bottleneck_adapter(
    records: list[TaskRecord], embeddings: Mapping[str, torch.Tensor], device: torch.device,
    seed: int, epochs: int, learning_rate: float, weight_decay: float, bottleneck_dim: int,
    result_stage: str, progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    _require(bottleneck_dim > 0, "RNA-FM adapter bottleneck is not positive")
    by_split = {
        split: [record for record in records if record.split == split]
        for split in SPLITS if any(record.split == split for record in records)
    }
    train = by_split["TRAIN"]
    fit = train if result_stage == "HPO_VALIDATION_ONLY" else train + by_split["VALIDATION"]
    features = {
        split: torch.stack([embeddings[record.candidate] - embeddings[record.source] for record in rows]).to(device)
        for split, rows in by_split.items()
    }
    targets = {
        split: torch.tensor([record.target for record in rows], dtype=torch.float32, device=device)
        for split, rows in by_split.items()
    }
    fit_features = torch.stack([embeddings[record.candidate] - embeddings[record.source] for record in fit]).to(device)
    mean, std = fit_features.mean(dim=0), fit_features.std(dim=0).clamp_min(1e-6)
    features = {split: (value - mean) / std for split, value in features.items()}
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    head = torch.nn.Sequential(
        torch.nn.Linear(features["TRAIN"].shape[1], bottleneck_dim),
        torch.nn.GELU(),
        torch.nn.Linear(bottleneck_dim, 1),
    ).to(device)
    _require(next(head.parameters()).is_cuda and next(head.parameters()).device == device, "RNA-FM adapter left CUDA")
    initial = next(head.parameters()).detach().clone()
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    fit_feature_values = (fit_features - mean) / std
    fit_targets = torch.tensor([record.target for record in fit], dtype=torch.float32, device=device)
    weights = _source_group_weights(fit, device)
    validation_weights = _source_group_weights(by_split["VALIDATION"], device)
    best_state = None; best_validation = float("inf"); history = []
    for epoch in range(epochs):
        head.train(); optimizer.zero_grad(set_to_none=True)
        prediction = head(fit_feature_values).squeeze(1)
        loss = ((prediction - fit_targets) ** 2 * weights).mean()
        _require(loss.is_cuda and loss.device == device and torch.isfinite(loss).item(), "RNA-FM adapter loss left CUDA or became nonfinite")
        loss.backward(); optimizer.step()
        head.eval()
        if result_stage == "HPO_VALIDATION_ONLY":
            with torch.no_grad():
                validation_error = head(features["VALIDATION"]).squeeze(1) - targets["VALIDATION"]
                validation = ((validation_error ** 2) * validation_weights).mean().item()
            if validation < best_validation:
                best_validation = validation
                best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
        else:
            validation = None
        if epoch == 0 or (epoch + 1) % 10 == 0:
            row = {
                "epoch": epoch + 1,
                "train_source_group_weighted_mse": float(loss.detach()),
                "validation_source_group_weighted_mse": validation,
            }
            history.append(row)
            if progress is not None:
                progress({"event": "RNAFM_ADAPTER_EPOCH_COMPLETED", **row})
    if result_stage == "FROZEN_DEVELOPMENT_TEST":
        best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
        best_validation = None
    _require(best_state is not None and not torch.equal(initial, next(head.parameters()).detach()), "RNA-FM adapter had no GPU parameter update")
    head.load_state_dict(best_state); head.eval()
    predictions = {}
    with torch.no_grad():
        for split, rows in by_split.items():
            values = head(features[split]).squeeze(1).cpu().numpy()
            predictions.update({record.record_id: float(value) for record, value in zip(rows, values)})
    artifact = {
        "feature_mean": mean.detach().cpu(),
        "feature_std": std.detach().cpu(),
        "adapter_state": {key: value.detach().cpu() for key, value in best_state.items()},
        "bottleneck_dim": bottleneck_dim,
    }
    return predictions, {
        "adapter_parameter_count": sum(parameter.numel() for parameter in head.parameters()),
        "adapter_optimizer_steps": epochs,
        "adapter_parameter_changed": True,
        "best_validation_source_group_weighted_mse": best_validation,
        "development_validation_folded_into_adapter_training": result_stage == "FROZEN_DEVELOPMENT_TEST",
        "history": history,
    }, artifact


def _metrics(records: list[TaskRecord], predictions: Mapping[str, float], split: str) -> dict[str, Any]:
    rows = [record for record in records if record.split == split]
    observed = np.asarray([record.target for record in rows], dtype=float)
    predicted = np.asarray([predictions[record.record_id] for record in rows], dtype=float)
    correlation = (
        None
        if len(rows) < 3 or np.std(observed) == 0.0 or np.std(predicted) == 0.0
        else spearmanr(observed, predicted).statistic
    )
    return {
        "record_count": len(rows),
        "mae": float(np.mean(np.abs(observed - predicted))),
        "spearman": (
            float(correlation)
            if correlation is not None and math.isfinite(float(correlation))
            else None
        ),
    }


def _write_predictions(path: Path, baseline_id: str, records: list[TaskRecord], predictions: Mapping[str, float], split: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            if record.split == split:
                handle.write(json.dumps({
                    "canonical_record_id": record.record_id,
                    "baseline_id": baseline_id,
                    "predicted_direction_normalized_delta": predictions[record.record_id],
                }, sort_keys=True) + "\n")


def execute(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"output already exists: {output_dir}")
    _require(config["evaluation_outcomes_accessed"] is False, "external baseline accessed Evaluation")
    _require(str(config["device"]).startswith("cuda"), "external neural baseline requires CUDA")
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden for physical-device provenance")
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    _require(0 <= int(config["physical_gpu_index"]) < torch.cuda.device_count(), "physical GPU index is unavailable")
    device = torch.device(str(config["device"])); torch.cuda.set_device(device)
    _require(device.index == int(config["physical_gpu_index"]), "CUDA device index differs from declared physical GPU")
    cuda_provenance = cuda_device_observation(int(config["physical_gpu_index"]))
    result_stage = str(config.get("result_stage", ""))
    included_splits = splits_for_result_stage(result_stage)
    records, task_manifest = load_task_records(
        Path(config["canonical_path"]), Path(config["development_manifest_path"]), included_splits
    )
    output_splits = ("VALIDATION",) if result_stage == "HPO_VALIDATION_ONLY" else ("TEST",)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    serialized_config = json.dumps(dict(config), indent=2, sort_keys=True) + "\n"
    (output_dir / "run_config.json").write_text(serialized_config, encoding="utf-8")
    (output_dir / "config.yaml").write_text(serialized_config, encoding="utf-8")
    (output_dir / "task_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in task_manifest), encoding="utf-8"
    )
    log_path = output_dir / "train.log"
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    def progress(row: Mapping[str, Any]) -> None:
        serialized = json.dumps(dict(row), sort_keys=True) + "\n"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(serialized)
        if row.get("event") in {"RNAFM_PROBE_EPOCH_COMPLETED", "RNAFM_ADAPTER_EPOCH_COMPLETED"}:
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(serialized)

    progress({
        "event": "RUN_STARTED",
        "device": str(device),
        "physical_gpu_index": int(config["physical_gpu_index"]),
        "cuda_device_uuid": cuda_provenance["cuda_device_uuid"],
        "result_stage": result_stage,
    })
    started = time.time()
    summaries: dict[str, Any] = {}
    methods = (
        (
            "optimus5prime",
            "optimus5prime_official_d53df410",
            Optimus5Prime(Path(config["optimus5prime_weight_path"])),
        ),
        (
            "framepool",
            "framepool_official_c575f9cd",
            FramePool(Path(config["framepool_weight_path"])),
        ),
    )
    for baseline_id, provenance_id, model in methods:
        progress({"event": "NATIVE_BASELINE_STARTED", "baseline_id": baseline_id})
        model = model.to(device)
        predictions = _predict_native(model, records, device, int(config["native_batch_size"]))
        method_dir = output_dir / baseline_id
        method_dir.mkdir()
        for split in output_splits:
            _write_predictions(
                method_dir / f"{split.lower()}_predictions.jsonl",
                baseline_id,
                records,
                predictions,
                split,
            )
        summaries[baseline_id] = {
            "status": "COMMON_SOURCE_RELATIVE_TASK_COMPLETED",
            "pretrained_parameters_frozen": True,
            "pretrained_parameter_count": sum(buffer.numel() for buffer in model.buffers()),
            "prediction_definition": "f(candidate)-f(source)",
            "artifact_provenance_id": provenance_id,
            "pytorch_port_numeric_parity_status": "NOT_RUN_TENSORFLOW_UNAVAILABLE",
            "task_study_training_overlap": "KNOWN_GSE114002_STUDY_LEVEL_OVERLAP_DEVELOPMENT_ONLY",
            "independent_external_transfer_claim_allowed": False,
            **{split.lower(): _metrics(records, predictions, split) for split in output_splits},
        }
        progress({"event": "NATIVE_BASELINE_COMPLETED", "baseline_id": baseline_id})
        del model
    embeddings, rnafm_parameter_count = _multimolecule_rnafm_embeddings(
        [sequence for record in records for sequence in (record.source, record.candidate)],
        Path(config["rnafm_model_path"]),
        device,
        int(config["rnafm_batch_size"]),
        progress,
    )
    predictions, probe, probe_artifact = _train_multimolecule_rnafm_probe(
        records,
        embeddings,
        device,
        int(config["seed"]),
        int(config["rnafm_probe_epochs"]),
        float(config["rnafm_probe_learning_rate"]),
        float(config["rnafm_probe_weight_decay"]),
        result_stage,
        progress,
    )
    method_dir = output_dir / "multimolecule_rnafm_frozen_linear_probe"
    method_dir.mkdir()
    for split in output_splits:
        _write_predictions(
            method_dir / f"{split.lower()}_predictions.jsonl",
            "multimolecule_rnafm_frozen_linear_probe",
            records,
            predictions,
            split,
        )
    for name in ("rnafm_probe_state.pt", "latest.pt", "best.pt"):
        torch.save(probe_artifact, output_dir / name)
    summaries["multimolecule_rnafm_frozen_linear_probe"] = {
        "status": "COMMON_SOURCE_RELATIVE_TASK_COMPLETED",
        "pretrained_parameters_frozen": True,
        "pretrained_parameter_count": rnafm_parameter_count,
        "prediction_definition": "linear(candidate_embedding-source_embedding)",
        "artifact_identity": "multimolecule/rnafm unofficial conversion",
        "artifact_provenance_id": "multimolecule_rnafm_7d6e73ad",
        "official_original_checkpoint_used": False,
        "multimolecule_rnafm_token_hidden_state_used": True,
        "multimolecule_rnafm_random_pooler_used": False,
        "task_label_training_overlap": "NO_TASK_LABEL_TRAINING_IN_FOUNDATION_PRETRAINING",
        "exact_sequence_pretraining_overlap": "UNKNOWN_NOT_ASSERTED",
        "independent_external_transfer_claim_allowed": False,
        **{split.lower(): _metrics(records, predictions, split) for split in output_splits},
        **probe,
    }
    adapter_grid = list(config.get("rnafm_bottleneck_adapter_hpo_grid", []))
    if adapter_grid:
        _require(result_stage == "HPO_VALIDATION_ONLY", "RNA-FM adapter HPO grid is Development-validation only")
        _require(len(adapter_grid) == 4, "RNA-FM adapter requires the frozen four-trial HPO budget")
        trial_ids = [str(trial["trial_id"]) for trial in adapter_grid]
        _require(len(trial_ids) == len(set(trial_ids)), "RNA-FM adapter HPO trial is duplicated")
        adapter_root = output_dir / "multimolecule_rnafm_frozen_bottleneck_adapter"
        adapter_root.mkdir()
        trial_results = []
        prediction_by_trial = {}
        artifact_by_trial = {}
        for trial in adapter_grid:
            trial_id = str(trial["trial_id"])
            trial_predictions, adapter, adapter_artifact = _train_multimolecule_rnafm_bottleneck_adapter(
                records, embeddings, device, int(config["seed"]), int(config["rnafm_probe_epochs"]),
                float(trial["learning_rate"]), float(trial["weight_decay"]),
                int(config["rnafm_adapter_bottleneck_dim"]), result_stage, progress,
            )
            trial_dir = adapter_root / trial_id
            trial_dir.mkdir()
            _write_predictions(
                trial_dir / "validation_predictions.jsonl", trial_id, records, trial_predictions, "VALIDATION"
            )
            torch.save(adapter_artifact, trial_dir / "adapter_state.pt")
            validation_metrics = _metrics(records, trial_predictions, "VALIDATION")
            trial_results.append({
                "trial_id": trial_id,
                "learning_rate": float(trial["learning_rate"]),
                "weight_decay": float(trial["weight_decay"]),
                "validation": validation_metrics,
                **adapter,
            })
            prediction_by_trial[trial_id] = trial_predictions
            artifact_by_trial[trial_id] = adapter_artifact
        ranked = sorted(
            trial_results,
            key=lambda row: (
                row["validation"]["spearman"] is None,
                0.0 if row["validation"]["spearman"] is None else -row["validation"]["spearman"],
                row["validation"]["mae"], row["trial_id"],
            ),
        )
        selected = ranked[0]
        selected_id = selected["trial_id"]
        _write_predictions(
            adapter_root / "validation_predictions.jsonl",
            "multimolecule_rnafm_frozen_bottleneck_adapter",
            records, prediction_by_trial[selected_id], "VALIDATION",
        )
        torch.save(artifact_by_trial[selected_id], adapter_root / "selected_adapter_state.pt")
        summaries["multimolecule_rnafm_frozen_bottleneck_adapter"] = {
            "status": "COMMON_SOURCE_RELATIVE_TASK_COMPLETED_HPO_SELECTED",
            "pretrained_parameters_frozen": True,
            "pretrained_parameter_count": rnafm_parameter_count,
            "prediction_definition": "bottleneck_adapter(candidate_embedding-source_embedding)",
            "artifact_identity": "multimolecule/rnafm unofficial conversion",
            "artifact_provenance_id": "multimolecule_rnafm_7d6e73ad",
            "official_original_checkpoint_used": False,
            "selected_trial_id": selected_id,
            "selection_primary_metric": "DEVELOPMENT_VALIDATION_SPEARMAN",
            "hpo_trials": trial_results,
            "validation": selected["validation"],
            "adapter_parameter_count": selected["adapter_parameter_count"],
            "task_label_training_overlap": "NO_TASK_LABEL_TRAINING_IN_FOUNDATION_PRETRAINING",
            "exact_sequence_pretraining_overlap": "UNKNOWN_NOT_ASSERTED",
            "independent_external_transfer_claim_allowed": False,
        }
    summary = {
        "schema_version": "route_a_v3_route2_external_prediction_baselines.v1",
        "status": "TASK_MATCHED_EXTERNAL_COMMON_TASK_BASELINES_COMPLETED",
        "result_stage": result_stage,
        "development_test_outcomes_evaluated": result_stage == "FROZEN_DEVELOPMENT_TEST",
        "development_test_record_count_withheld": (
            0 if result_stage == "FROZEN_DEVELOPMENT_TEST"
            else sum(
                row["study_unit_id"] == TASK_STUDY and row["split"] == "TEST"
                for row in (
                    json.loads(line)
                    for line in Path(config["development_manifest_path"]).read_text(encoding="utf-8").splitlines()
                )
            )
        ),
        "task": {"study_unit_id": TASK_STUDY, "region": TASK_REGION, "endpoint_id": TASK_ENDPOINT},
        "record_counts": {split: sum(record.split == split for record in records) for split in SPLITS},
        "physical_gpu_index": int(config["physical_gpu_index"]),
        "device": str(device),
        "cpu_fallback_used": False,
        "cuda_training_tensors_verified": True,
        "baselines": summaries,
        "evaluation_outcomes_accessed": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "wall_time_seconds": time.time() - started,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        **cuda_provenance,
    }
    serialized_summary = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (output_dir / "summary.json").write_text(serialized_summary, encoding="utf-8")
    (output_dir / "final_summary.json").write_text(serialized_summary, encoding="utf-8")
    progress({"event": "RUN_COMPLETED", "wall_time_seconds": summary["wall_time_seconds"]})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or Path(config["output_directory"])
    try:
        result = execute(config, output_dir)
    except Exception as exc:
        write_gpu_failure_evidence(
            output_dir.with_name(output_dir.name + ".failed.json"), config, exc,
            entrypoint="run_route2_external_prediction_baselines_v1",
            evaluation_outcomes_accessed=config.get("evaluation_outcomes_accessed"),
        )
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
