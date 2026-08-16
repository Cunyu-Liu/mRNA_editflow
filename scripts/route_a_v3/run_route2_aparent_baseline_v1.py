#!/usr/bin/env python3
"""Run the official APARENT base model on the GSE269595 common task."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from core.route2_gpu_failure_evidence import cuda_device_observation, write_gpu_failure_evidence


BASES = "ACGT"
TASK_STUDY = "GSE269595"
TASK_REGION = "3UTR"
TASK_ENDPOINT = "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS"
TASK_SEQUENCE_LENGTH = 164
APARENT_SEQUENCE_LENGTH = 205
SPLITS = ("TRAIN", "VALIDATION", "TEST")


class AparentBaselineError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AparentBaselineError(message)


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
        str(row["canonical_record_id"]): row
        for row in manifest_rows if row["study_unit_id"] == TASK_STUDY
    }
    _require(complete_manifest, "GSE269595 is absent from Development manifest")
    _require(all(row["pool_assignment"] == "DEVELOPMENT" for row in complete_manifest.values()), "task left Development")
    _require(all(row["split"] in SPLITS for row in complete_manifest.values()), "task split changed")
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
            _require(record_id not in seen, "canonical record is duplicated")
            _require(row["study_unit_id"] == TASK_STUDY and row["pool_assignment"] == "DEVELOPMENT", "task identity changed")
            _require(row["region"] == TASK_REGION and row["endpoint_id"] == TASK_ENDPOINT, "task endpoint changed")
            source, candidate = str(row["source_sequence"]).upper(), str(row["candidate_sequence"]).upper()
            _require(len(source) == len(candidate) == TASK_SEQUENCE_LENGTH, "APARENT task requires exact 164-nt pairs")
            _require(not ((set(source) | set(candidate)) - set(BASES)), "task sequence alphabet changed")
            seen.add(record_id)
            records.append(TaskRecord(
                record_id=record_id,
                source_id=str(row["source_id"]),
                source=source,
                candidate=candidate,
                target=_finite(row["direction_normalized_delta"], "direction-normalized delta"),
                split=str(selected_manifest[record_id]["split"]),
            ))
    _require(seen == set(selected_manifest), "canonical records do not exactly cover GSE269595 manifest")
    _require(all(any(record.split == split for record in records) for split in included_splits), "task split is incomplete")
    return records, [row for row in manifest_rows if str(row["canonical_record_id"]) in seen]


def one_hot(sequences: Iterable[str], device: torch.device) -> torch.Tensor:
    mapping = torch.full((256,), -1, dtype=torch.long)
    for index, base in enumerate(BASES):
        mapping[ord(base)] = index
    encoded = []
    for sequence in sequences:
        _require(len(sequence) <= APARENT_SEQUENCE_LENGTH, "sequence exceeds APARENT 205-nt input")
        values = mapping[torch.tensor(list(sequence.encode("ascii")), dtype=torch.long)]
        _require(bool(torch.all(values >= 0)), "sequence contains an unsupported base")
        padded = torch.zeros((APARENT_SEQUENCE_LENGTH, 4), dtype=torch.float32)
        padded[:len(sequence)] = F.one_hot(values, num_classes=4).to(torch.float32)
        encoded.append(padded)
    result = torch.stack(encoded).unsqueeze(1).to(device)
    _require(result.is_cuda and result.device == device, "APARENT input silently left the declared GPU")
    return result


def _array(handle: h5py.File, path: str) -> torch.Tensor:
    _require(path in handle, f"APARENT weight dataset absent: {path}")
    return torch.from_numpy(np.asarray(handle[path], dtype=np.float32))


class AparentBase(torch.nn.Module):
    """PyTorch inference port of the official all-libraries APARENT base model."""

    def __init__(self, weight_path: Path):
        super().__init__()
        with h5py.File(weight_path, "r") as handle:
            prefix = "model_weights"
            self.register_buffer("conv1_weight", _array(handle, f"{prefix}/conv2d_1/conv2d_1/kernel:0").permute(3, 2, 0, 1))
            self.register_buffer("conv1_bias", _array(handle, f"{prefix}/conv2d_1/conv2d_1/bias:0"))
            self.register_buffer("conv2_weight", _array(handle, f"{prefix}/conv2d_2/conv2d_2/kernel:0").permute(3, 2, 0, 1))
            self.register_buffer("conv2_bias", _array(handle, f"{prefix}/conv2d_2/conv2d_2/bias:0"))
            for number in range(1, 5):
                self.register_buffer(
                    f"dense{number}_weight",
                    _array(handle, f"{prefix}/dense_{number}/dense_{number}/kernel:0").T,
                )
                self.register_buffer(
                    f"dense{number}_bias",
                    _array(handle, f"{prefix}/dense_{number}/dense_{number}/bias:0"),
                )
        _require(tuple(self.conv1_weight.shape) == (96, 1, 8, 4), "APARENT conv1 geometry changed")
        _require(tuple(self.conv2_weight.shape) == (128, 96, 6, 1), "APARENT conv2 geometry changed")
        _require(tuple(self.dense1_weight.shape) == (512, 12033), "APARENT dense1 geometry changed")
        _require(tuple(self.dense3_weight.shape) == (206, 269), "APARENT cut-head geometry changed")

    def forward(self, sequence_one_hot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(F.conv2d(sequence_one_hot, self.conv1_weight, self.conv1_bias))
        x = F.max_pool2d(x, kernel_size=(2, 1), stride=(2, 1))
        x = torch.relu(F.conv2d(x, self.conv2_weight, self.conv2_bias))
        # Keras Flatten consumes (position, width, channel), not channels-first order.
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)
        distal_flag = torch.ones((len(x), 1), dtype=x.dtype, device=x.device)
        x = torch.relu(F.linear(torch.cat((x, distal_flag), dim=1), self.dense1_weight, self.dense1_bias))
        x = torch.relu(F.linear(x, self.dense2_weight, self.dense2_bias))
        library_indicator = torch.zeros((len(x), 13), dtype=x.dtype, device=x.device)
        shared = torch.cat((x, library_indicator), dim=1)
        isoform_probability = torch.sigmoid(F.linear(shared, self.dense4_weight, self.dense4_bias)).squeeze(1)
        cut_probability = torch.softmax(F.linear(shared, self.dense3_weight, self.dense3_bias), dim=1)
        return isoform_probability, cut_probability


def proximal_log2_odds(cut_probability: torch.Tensor, cut_start: int, cut_end: int) -> torch.Tensor:
    _require(0 <= cut_start < cut_end <= 205, "APARENT cut window is invalid")
    probability = cut_probability[:, cut_start:cut_end].sum(dim=1)
    epsilon = torch.finfo(probability.dtype).eps
    probability = probability.clamp(min=epsilon, max=1.0 - epsilon)
    return torch.log2(probability) - torch.log2(1.0 - probability)


def predict(
    model: AparentBase,
    records: list[TaskRecord],
    device: torch.device,
    batch_size: int,
    cut_start: int,
    cut_end: int,
) -> dict[str, float]:
    model.eval()
    result: dict[str, float] = {}
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            source = one_hot((record.source for record in batch), device)
            candidate = one_hot((record.candidate for record in batch), device)
            _source_iso, source_cut = model(source)
            _candidate_iso, candidate_cut = model(candidate)
            values = (
                proximal_log2_odds(candidate_cut, cut_start, cut_end)
                - proximal_log2_odds(source_cut, cut_start, cut_end)
            )
            _require(values.is_cuda and values.device == device, "APARENT prediction silently fell back to CPU")
            result.update({record.record_id: float(value) for record, value in zip(batch, values.cpu())})
    return result


def _metrics(records: list[TaskRecord], predictions: Mapping[str, float], split: str) -> dict[str, Any]:
    rows = [record for record in records if record.split == split]
    observed = np.asarray([record.target for record in rows], dtype=float)
    predicted = np.asarray([predictions[record.record_id] for record in rows], dtype=float)
    correlation = None if len(rows) < 3 or np.std(observed) == 0 or np.std(predicted) == 0 else spearmanr(observed, predicted).statistic
    return {
        "record_count": len(rows),
        "mae": float(np.mean(np.abs(observed - predicted))),
        "spearman": float(correlation) if correlation is not None and math.isfinite(float(correlation)) else None,
    }


def _write_predictions(path: Path, records: list[TaskRecord], predictions: Mapping[str, float], split: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            if record.split == split:
                handle.write(json.dumps({
                    "canonical_record_id": record.record_id,
                    "baseline_id": "aparent_official_base_cut_window",
                    "predicted_direction_normalized_delta": predictions[record.record_id],
                }, sort_keys=True) + "\n")


def execute(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"output already exists: {output_dir}")
    _require(config["evaluation_outcomes_accessed"] is False, "APARENT baseline accessed Evaluation")
    _require(str(config["device"]).startswith("cuda:"), "APARENT inference requires an explicit CUDA device")
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden for physical-device provenance")
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    physical_gpu_index = int(config["physical_gpu_index"])
    _require(0 <= physical_gpu_index < torch.cuda.device_count(), "physical GPU index is unavailable")
    device = torch.device(str(config["device"]))
    _require(device.index == physical_gpu_index, "CUDA device differs from declared physical GPU")
    _require(config["official_git_revision"] == "69ad29791709b48689ff5d9e3a3daefc568de9ce", "APARENT revision changed")
    cut_start, cut_end = int(config["cut_start"]), int(config["cut_end"])
    _require((cut_start, cut_end) == (80, 105), "APARENT proximal cut window changed")
    torch.cuda.set_device(device)
    cuda_provenance = cuda_device_observation(physical_gpu_index)
    torch.cuda.reset_peak_memory_stats(device)
    result_stage = str(config.get("result_stage", ""))
    included_splits = splits_for_result_stage(result_stage)
    records, task_manifest = load_task_records(
        Path(config["canonical_path"]), Path(config["development_manifest_path"]), included_splits
    )
    output_splits = ("VALIDATION",) if result_stage == "HPO_VALIDATION_ONLY" else ("TEST",)
    model = AparentBase(Path(config["weight_path"])).to(device)
    started = time.time()
    predictions = predict(model, records, device, int(config["batch_size"]), cut_start, cut_end)
    torch.cuda.synchronize(device)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        (temporary / "task_manifest.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in task_manifest), encoding="utf-8"
        )
        for split in output_splits:
            _write_predictions(temporary / f"{split.lower()}_predictions.jsonl", records, predictions, split)
        summary = {
            "schema_version": "route_a_v3_route2_aparent_baseline.v1",
            "status": "APARENT_GSE269595_COMMON_TASK_COMPLETED",
            "result_stage": result_stage,
            "development_test_outcomes_evaluated": result_stage == "FROZEN_DEVELOPMENT_TEST",
            "baseline_id": "aparent_official_base_cut_window",
            "task": {"study_unit_id": TASK_STUDY, "region": TASK_REGION, "endpoint_id": TASK_ENDPOINT},
            "prediction_definition": "log2_odds(sum_cut_probability_80_105(candidate))-same(source)",
            "official_git_revision": config["official_git_revision"],
            "pretrained_parameters_frozen": True,
            "pretrained_parameter_count": sum(buffer.numel() for buffer in model.buffers()),
            "record_counts": {split: sum(record.split == split for record in records) for split in SPLITS},
            **{split.lower(): _metrics(records, predictions, split) for split in output_splits},
            "physical_gpu_index": physical_gpu_index,
            "device": str(device),
            "cpu_fallback_used": False,
            "evaluation_outcomes_accessed": False,
            "independent_external_transfer_claim_allowed": False,
            "wall_time_seconds": time.time() - started,
            "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
            **cuda_provenance,
            "scientific_claim_status": "NOT_ESTABLISHED",
        }
        (temporary / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "run_config.json").write_text(json.dumps(dict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.rename(temporary, output_dir)
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


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
            entrypoint="run_route2_aparent_baseline_v1",
            evaluation_outcomes_accessed=config.get("evaluation_outcomes_accessed"),
        )
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
