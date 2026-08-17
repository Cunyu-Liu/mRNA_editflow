#!/usr/bin/env python3
"""Train the unguided Route 2 SUB+STOP base flow on Development data."""

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

import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_base_flow_model import Route2BaseFlowModel
from core.route2_experiment_ledger import (
    build_training_attempt_row,
    record_training_attempt,
)
from core.route2_gpu_failure_evidence import cuda_device_observation, write_gpu_failure_evidence


TOKEN = {"A": 0, "C": 1, "G": 2, "U": 3}
PAD = 4
REGION = {"5UTR": 0, "3UTR": 1}


class BaseFlowTrainingError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BaseFlowTrainingError(message)


def normalize_sequence(value: Any) -> str:
    sequence = str(value).upper().replace("T", "U")
    _require(sequence and set(sequence) <= set(TOKEN), "sequence is outside the RNA alphabet")
    return sequence


def load_manifest_ids(path: Path, split: str) -> set[str]:
    result = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            _require(row["pool_assignment"] == "DEVELOPMENT", "non-Development row entered base-flow manifest")
            if row["split"] == split:
                result.add(str(row["canonical_record_id"]))
    _require(result, f"manifest split is empty: {split}")
    return result


@dataclass(frozen=True)
class FlowRecord:
    record_id: str
    source: str
    candidate: str
    edits: tuple[tuple[int, int], ...]
    budget: int
    source_group: str
    region: int
    assay: str
    context: str


def assigned_budget(edit_count: int, allowed_budgets: tuple[int, ...]) -> int:
    for budget in allowed_budgets:
        if edit_count <= budget:
            return budget
    raise BaseFlowTrainingError(f"edit count {edit_count} exceeds the largest allowed budget")


def load_records(
    canonical_paths: Iterable[Path],
    selected_ids: set[str],
    *,
    allowed_budgets: tuple[int, ...],
) -> list[FlowRecord]:
    records: dict[str, FlowRecord] = {}
    skipped_over_budget = 0
    for path in canonical_paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                record_id = str(row["canonical_record_id"])
                if record_id not in selected_ids:
                    continue
                _require(row["pool_assignment"] == "DEVELOPMENT", "Evaluation row reached base-flow loader")
                source = normalize_sequence(row["source_sequence"])
                candidate = normalize_sequence(row["candidate_sequence"])
                _require(len(source) == len(candidate), f"length-changing record reached SUB flow: {record_id}")
                edits = tuple((index, TOKEN[right]) for index, (left, right) in enumerate(zip(source, candidate)) if left != right)
                try:
                    budget = assigned_budget(len(edits), allowed_budgets)
                except BaseFlowTrainingError:
                    skipped_over_budget += 1
                    continue
                region = REGION.get(str(row["region"]).replace("′", "").replace("'", ""))
                _require(region is not None, f"unsupported region: {row['region']}")
                _require(record_id not in records, f"canonical record duplicated: {record_id}")
                source_group = "::".join((
                    str(row["study_unit_id"]), str(row["source_id"]),
                    str(row["biological_context_id"]), str(row["endpoint_id"]),
                ))
                records[record_id] = FlowRecord(
                    record_id,
                    source,
                    candidate,
                    edits,
                    budget,
                    source_group,
                    region,
                    str(row["assay_id"]),
                    str(row["biological_context_id"]),
                )
    _require(set(records).issubset(selected_ids), "loader escaped the selected manifest ids")
    _require(len(records) + skipped_over_budget == len(selected_ids), "canonical inputs do not cover the manifest ids")
    _require(records, "no eligible base-flow records")
    return [records[key] for key in sorted(records)]


def build_vocab(records: Iterable[FlowRecord], field: str) -> dict[str, int]:
    values = sorted({str(getattr(record, field)) for record in records})
    return {value: index + 1 for index, value in enumerate(values)} | {"__UNK__": 0}


class FlowTrajectoryDataset(Dataset):
    """One deterministic random trajectory state per measured pair and epoch."""

    def __init__(self, records: list[FlowRecord], assay_vocab: Mapping[str, int], context_vocab: Mapping[str, int], seed: int):
        self.records = records
        self.assay_vocab = assay_vocab
        self.context_vocab = context_vocab
        self.seed = seed
        self.epoch = 0
        self.group_sizes = Counter(record.source_group for record in records)
        self.group_weight_scale = len(records) / len(self.group_sizes)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        rng = random.Random(self.seed + self.epoch * len(self.records) + index)
        ordered = list(record.edits)
        rng.shuffle(ordered)
        stop_available = len(ordered) < record.budget
        action_count = len(ordered) + int(stop_available)
        _require(action_count > 0, "trajectory has no trainable action")
        action_index = rng.randrange(action_count)
        prefix_length = min(action_index, len(ordered))
        current = list(record.source)
        for position, alt_token in ordered[:prefix_length]:
            current[position] = "ACGU"[alt_token]
        if action_index < len(ordered):
            position, alt_token = ordered[prefix_length]
            target_position = position
            target_alt = alt_token
            target_stop = False
        else:
            _require(stop_available, "STOP label requested after structural absorption")
            target_position = -1
            target_alt = -1
            target_stop = True
        return {
            "record_id": record.record_id,
            "source": [TOKEN[base] for base in record.source],
            "current": [TOKEN[base] for base in current],
            "remaining_budget": record.budget - prefix_length,
            "region": record.region,
            "assay": self.assay_vocab.get(record.assay, 0),
            "context": self.context_vocab.get(record.context, 0),
            "sample_weight": self.group_weight_scale / self.group_sizes[record.source_group],
            "target_position": target_position,
            "target_alt": target_alt,
            "target_stop": target_stop,
        }


def collate_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    maximum = max(len(example["source"]) for example in examples)
    batch = len(examples)
    source = torch.full((batch, maximum), PAD, dtype=torch.long)
    current = torch.full((batch, maximum), PAD, dtype=torch.long)
    padding = torch.ones((batch, maximum), dtype=torch.bool)
    target = torch.empty(batch, dtype=torch.long)
    for index, example in enumerate(examples):
        length = len(example["source"])
        source[index, :length] = torch.tensor(example["source"])
        current[index, :length] = torch.tensor(example["current"])
        padding[index, :length] = False
        target[index] = maximum * 4 if example["target_stop"] else example["target_position"] * 4 + example["target_alt"]
    return {
        "record_ids": [example["record_id"] for example in examples],
        "source_tokens": source,
        "current_tokens": current,
        "padding_mask": padding,
        "region_ids": torch.tensor([example["region"] for example in examples]),
        "assay_ids": torch.tensor([example["assay"] for example in examples]),
        "context_ids": torch.tensor([example["context"] for example in examples]),
        "remaining_budget": torch.tensor([example["remaining_budget"] for example in examples]),
        "target": target,
        "sample_weight": torch.tensor([example["sample_weight"] for example in examples], dtype=torch.float32),
    }


def require_cuda_device(device_text: str, physical_gpu_index: int) -> torch.device:
    _require(device_text.startswith("cuda"), "base-flow parameter updates require CUDA; CPU fallback is forbidden")
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden for physical-device provenance")
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    _require(0 <= physical_gpu_index < torch.cuda.device_count(), "physical GPU index is unavailable")
    device = torch.device(device_text)
    _require(device.index == physical_gpu_index, "CUDA device index differs from declared physical GPU")
    torch.cuda.set_device(device)
    return device


def _move(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in batch.items()}


def _loss(model: Route2BaseFlowModel, batch: Mapping[str, Any]) -> torch.Tensor:
    rates, legal = model.rates(
        batch["source_tokens"],
        batch["current_tokens"],
        batch["padding_mask"],
        batch["region_ids"],
        batch["assay_ids"],
        batch["context_ids"],
        batch["remaining_budget"],
    )
    target = batch["target"]
    _require(bool(legal.gather(1, target[:, None]).all().item()), "training target is not hard-legal")
    log_rates = torch.full_like(rates, -torch.inf)
    log_rates[legal] = torch.log(rates[legal])
    per_record = torch.nn.functional.cross_entropy(log_rates, target, reduction="none")
    return (per_record * batch["sample_weight"]).mean()


def train(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"output already exists: {output_dir}")
    device = require_cuda_device(str(config["device"]), int(config["physical_gpu_index"]))
    cuda_provenance = cuda_device_observation(int(config["physical_gpu_index"]), require_physical_index_match=True)
    train_ids = load_manifest_ids(Path(config["development_manifest"]), "TRAIN")
    validation_ids = load_manifest_ids(Path(config["development_manifest"]), "VALIDATION")
    canonical_paths = [Path(path) for path in config["canonical_paths"]]
    allowed_budgets = tuple(int(value) for value in config["allowed_edit_budgets"])
    _require(allowed_budgets == (1, 3, 5), "allowed edit budgets must remain 1/3/5")
    train_records = load_records(canonical_paths, train_ids, allowed_budgets=allowed_budgets)
    validation_records = load_records(canonical_paths, validation_ids, allowed_budgets=allowed_budgets)
    assay_vocab = build_vocab(train_records, "assay")
    context_vocab = build_vocab(train_records, "context")
    train_dataset = FlowTrajectoryDataset(train_records, assay_vocab, context_vocab, int(config["seed"]))
    validation_dataset = FlowTrajectoryDataset(validation_records, assay_vocab, context_vocab, int(config["seed"]) + 1)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=collate_examples,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=collate_examples,
    )
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    model = Route2BaseFlowModel(
        hidden_dim=int(config["hidden_dim"]),
        assay_count=len(assay_vocab),
        context_count=len(context_vocab),
    ).to(device)
    _require(next(model.parameters()).is_cuda, "model parameters are not on GPU")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    trainable_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    start = time.time()
    history = []
    optimizer_steps = 0
    cuda_training_tensors_verified = False
    initial_parameter = next(model.parameters()).detach().clone()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    serialized_config = json.dumps(dict(config), indent=2, sort_keys=True) + "\n"
    (output_dir / "training_config.json").write_text(serialized_config, encoding="utf-8")
    (output_dir / "config.yaml").write_text(serialized_config, encoding="utf-8")
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    log_path = output_dir / "train.log"
    log_path.write_text(json.dumps({
        "event": "TRAINING_STARTED",
        "device": str(device),
        "physical_gpu_index": int(config["physical_gpu_index"]),
        "cuda_device_uuid": cuda_provenance["cuda_device_uuid"],
        "guided_critic_used": False,
    }, sort_keys=True) + "\n", encoding="utf-8")
    attempt_details = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "included_study_unit_ids": sorted(
            {record.source_group.split("::", 1)[0] for record in train_records}
        ),
        "included_regions": sorted(
            {"5UTR" if record.region == 0 else "3UTR" for record in train_records}
        ),
        "record_counts": {
            "TRAIN": len(train_records),
            "VALIDATION": len(validation_records),
        },
        "development_test_record_count_withheld": 0,
        "evaluation_record_count": 0,
        "trainable_parameter_count": trainable_parameter_count,
        "frozen_pretrained_parameter_count": 0,
        "total_effective_parameter_count": trainable_parameter_count,
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

    def checkpoint_payload(epoch: int) -> dict[str, Any]:
        return {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": {
                "hidden_dim": int(config["hidden_dim"]),
                "assay_count": len(assay_vocab),
                "context_count": len(context_vocab),
            },
            "assay_vocab": assay_vocab,
            "context_vocab": context_vocab,
            "allowed_edit_budgets": list(allowed_budgets),
            "completed_epoch": epoch,
            "training_provenance": {
                "seed": int(config["seed"]),
                "optimizer_steps": optimizer_steps,
                "parameter_changed": not torch.equal(initial_parameter, next(model.parameters()).detach()),
                "torch_device": str(device),
                "physical_gpu_index": int(config["physical_gpu_index"]),
                "cpu_fallback_used": False,
                "cuda_training_tensors_verified": cuda_training_tensors_verified,
                **cuda_provenance,
            },
        }

    best_validation_nll: float | None = None
    best_epoch: int | None = None
    for epoch in range(int(config["epochs"])):
        train_dataset.set_epoch(epoch)
        validation_dataset.set_epoch(epoch)
        model.train()
        train_losses = []
        for raw_batch in train_loader:
            batch = _move(raw_batch, device)
            _require(
                batch["source_tokens"].device == device and batch["target"].device == device,
                "base-flow training inputs left CUDA",
            )
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, batch)
            _require(loss.is_cuda and loss.device == device, "base-flow training loss left CUDA")
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            cuda_training_tensors_verified = True
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for raw_batch in validation_loader:
                validation_losses.append(float(_loss(model, _move(raw_batch, device)).cpu()))
        epoch_row = {
            "epoch": epoch + 1,
            "train_nll": float(sum(train_losses) / len(train_losses)),
            "validation_nll": float(sum(validation_losses) / len(validation_losses)),
        }
        history.append(epoch_row)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_row, sort_keys=True) + "\n")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "EPOCH_COMPLETED", **epoch_row}, sort_keys=True) + "\n")
        torch.save(checkpoint_payload(epoch + 1), output_dir / "latest.pt")
        if best_validation_nll is None or epoch_row["validation_nll"] < best_validation_nll:
            best_validation_nll = epoch_row["validation_nll"]
            best_epoch = epoch + 1
            torch.save(checkpoint_payload(epoch + 1), output_dir / "best.pt")
    _require(best_epoch is not None, "no best validation epoch was selected")
    parameter_changed = not torch.equal(initial_parameter, next(model.parameters()).detach())
    _require(optimizer_steps > 0 and parameter_changed, "no learned parameter update occurred")
    elapsed = time.time() - start
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    torch.save(checkpoint_payload(int(config["epochs"])), output_dir / "base_flow_checkpoint.pt")
    summary = {
        "schema_version": "route_a_v3_route2_base_flow_g0_training.v1",
        "status": "LEARNED_BASE_FLOW_GPU_UPDATE_COMPLETE",
        "seed": int(config["seed"]),
        "physical_gpu_index": int(config["physical_gpu_index"]),
        "torch_device": str(device),
        "cpu_fallback_used": False,
        "cuda_training_tensors_verified": cuda_training_tensors_verified,
        "train_record_count": len(train_records),
        "validation_record_count": len(validation_records),
        "trainable_parameter_count": trainable_parameter_count,
        "over_budget_excluded_record_counts": {
            "TRAIN": len(train_ids) - len(train_records),
            "VALIDATION": len(validation_ids) - len(validation_records),
        },
        "optimizer_steps": optimizer_steps,
        "selected_epoch": best_epoch,
        "best_validation_nll": best_validation_nll,
        "parameter_changed": parameter_changed,
        "history": history,
        "wall_time_seconds": elapsed,
        "peak_vram_mb": peak_vram,
        **cuda_provenance,
        "evaluation_records_read": 0,
        "guided_critic_used": False,
        "biological_optimization_established": False,
    }
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
                details={
                    **attempt_details,
                    "optimizer_steps": optimizer_steps,
                    "selected_epoch": best_epoch,
                    "wall_time_seconds": elapsed,
                    "peak_vram_mb": peak_vram,
                    "notes": (
                        "unguided G0 engineering run; no critic guidance or "
                        "biological optimization claim"
                    ),
                },
            ),
        )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "event": "TRAINING_COMPLETED",
            "optimizer_steps": optimizer_steps,
            "wall_time_seconds": elapsed,
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
            entrypoint="train_route2_base_flow_g0_v1",
            evaluation_outcomes_accessed=False,
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
