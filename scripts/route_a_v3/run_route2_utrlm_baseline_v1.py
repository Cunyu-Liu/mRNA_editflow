#!/usr/bin/env python3
"""Run the official frozen UTR-LM encoder with a GPU linear delta probe."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_gpu_failure_evidence import cuda_device_observation, write_gpu_failure_evidence


BASES = set("ACGT")
TASK_STUDY = "GSE114002"
TASK_REGION = "5UTR"
TASK_ENDPOINT = "MEAN_RIBOSOME_LOAD"
SPLITS = ("TRAIN", "VALIDATION", "TEST")
OFFICIAL_REVISION = "b77b589bf182eb9de6a1a5024fa09d44294d94fc"
OFFICIAL_CHECKPOINT_NAME = (
    "ESM2SISS_FS4.1_fiveSpeciesCao_6layers_16heads_128embedsize_4096batchToks_"
    "lr1e-05_supervisedweight1.0_structureweight1.0_MLMLossMin_epoch93.pkl"
)


class UtrLmBaselineError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise UtrLmBaselineError(message)


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
    complete = {
        str(row["canonical_record_id"]): row
        for row in manifest_rows if row["study_unit_id"] == TASK_STUDY
    }
    _require(complete, "GSE114002 is absent from Development manifest")
    _require(all(row["pool_assignment"] == "DEVELOPMENT" and row["split"] in SPLITS for row in complete.values()), "task split changed")
    _require(set(included_splits) <= set(SPLITS), "included task splits are invalid")
    selected = {
        record_id: row for record_id, row in complete.items()
        if row["split"] in included_splits
    }
    records = []
    seen: set[str] = set()
    with canonical_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            record_id = str(row["canonical_record_id"])
            if record_id not in selected:
                continue
            _require(record_id not in seen, "canonical record is duplicated")
            _require(row["pool_assignment"] == "DEVELOPMENT" and row["study_unit_id"] == TASK_STUDY, "task identity changed")
            _require(row["region"] == TASK_REGION and row["endpoint_id"] == TASK_ENDPOINT, "task endpoint changed")
            source, candidate = str(row["source_sequence"]).upper(), str(row["candidate_sequence"]).upper()
            _require(len(source) == len(candidate) == 50, "UTR-LM common task requires exact 50-nt pairs")
            _require(not ((set(source) | set(candidate)) - BASES), "task sequence alphabet changed")
            seen.add(record_id)
            records.append(TaskRecord(
                record_id=record_id,
                source_id=str(row["source_id"]),
                source=source,
                candidate=candidate,
                target=_finite(row["direction_normalized_delta"], "direction-normalized delta"),
                split=str(selected[record_id]["split"]),
            ))
    _require(seen == set(selected), "canonical records do not exactly cover GSE114002 manifest")
    _require(all(any(record.split == split for record in records) for split in included_splits), "task split is incomplete")
    return records, [row for row in manifest_rows if str(row["canonical_record_id"]) in seen]


def load_official_encoder(asset_root: Path, checkpoint_path: Path, device: torch.device):
    scripts_path = asset_root / "Scripts"
    _require((scripts_path / "esm/model/esm2_secondarystructure.py").is_file(), "official modified UTR-LM ESM source is absent")
    _require(checkpoint_path.name == OFFICIAL_CHECKPOINT_NAME and checkpoint_path.is_file(), "official UTR-LM checkpoint is absent or changed")
    sys.path.insert(0, str(scripts_path))
    try:
        from esm.data import Alphabet
        from esm.model.esm2_secondarystructure import ESM2
    except ImportError as exc:
        raise UtrLmBaselineError("official modified UTR-LM ESM package is not importable") from exc
    alphabet = Alphabet(mask_prob=0.0, standard_toks="AGCT")
    _require(alphabet.tok_to_idx == {
        "<pad>": 0, "<eos>": 1, "<unk>": 2, "A": 3, "G": 4,
        "C": 5, "T": 6, "<cls>": 7, "<mask>": 8, "<sep>": 9,
    }, "official UTR-LM vocabulary changed")
    model = ESM2(num_layers=6, embed_dim=128, attention_heads=16, alphabet=alphabet)
    raw_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _require(raw_state and all(str(key).startswith("module.") for key in raw_state), "UTR-LM checkpoint DDP key format changed")
    state = {str(key).removeprefix("module."): value for key, value in raw_state.items()}
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model = model.to(device).eval()
    _require(sum(parameter.numel() for parameter in model.parameters()) == 1_208_559, "UTR-LM parameter geometry changed")
    return model, alphabet


def encode_bos_embeddings(
    model,
    alphabet,
    sequences: list[str],
    device: torch.device,
    batch_size: int,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, torch.Tensor]:
    unique = sorted(set(sequences))
    converter = alphabet.get_batch_converter()
    result: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for start in range(0, len(unique), batch_size):
            batch = unique[start:start + batch_size]
            raw = [(str(index), sequence, sequence, []) for index, sequence in enumerate(batch)]
            converted = converter(raw)
            tokens = converted[3].to(device)
            output = model(
                tokens,
                repr_layers=[6],
                need_head_weights=False,
                return_contacts=False,
                return_representation=True,
            )
            bos = output["representations"][6][:, 0]
            _require(bos.is_cuda and bos.device == device and torch.isfinite(bos).all().item(), "UTR-LM embedding left GPU or became nonfinite")
            for sequence, embedding in zip(batch, bos):
                result[sequence] = embedding.detach()
            if progress is not None:
                progress({
                    "event": "EMBEDDING_BATCH_COMPLETED",
                    "completed_sequence_count": min(start + len(batch), len(unique)),
                    "total_sequence_count": len(unique),
                })
    return result


def _source_group_weights(records: list[TaskRecord], device: torch.device) -> torch.Tensor:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.source_id] = counts.get(record.source_id, 0) + 1
    scale = len(records) / len(counts)
    weights = torch.tensor([scale / counts[record.source_id] for record in records], dtype=torch.float32, device=device)
    _require(torch.isclose(weights.mean(), torch.tensor(1.0, device=device)).item(), "source-group weights do not normalize")
    return weights


def train_probe(
    records: list[TaskRecord],
    embeddings: Mapping[str, torch.Tensor],
    device: torch.device,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    result_stage: str = "HPO_VALIDATION_ONLY",
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, float], dict[str, Any], dict[str, torch.Tensor]]:
    _require(epochs > 0 and learning_rate > 0 and weight_decay >= 0, "UTR-LM probe budget is invalid")
    by_split = {
        split: [record for record in records if record.split == split]
        for split in SPLITS if any(record.split == split for record in records)
    }
    _require(result_stage in {"HPO_VALIDATION_ONLY", "FROZEN_DEVELOPMENT_TEST"}, "invalid probe result_stage")
    fit = by_split["TRAIN"] if result_stage == "HPO_VALIDATION_ONLY" else by_split["TRAIN"] + by_split["VALIDATION"]
    features = {
        split: torch.stack([embeddings[record.candidate] - embeddings[record.source] for record in rows]).to(device)
        for split, rows in by_split.items()
    }
    targets = {
        split: torch.tensor([record.target for record in rows], dtype=torch.float32, device=device)
        for split, rows in by_split.items()
    }
    raw_fit_features = torch.stack([embeddings[record.candidate] - embeddings[record.source] for record in fit]).to(device)
    mean = raw_fit_features.mean(dim=0)
    std = raw_fit_features.std(dim=0).clamp_min(1e-6)
    features = {split: (value - mean) / std for split, value in features.items()}
    fit_features = (raw_fit_features - mean) / std
    fit_targets = torch.tensor([record.target for record in fit], dtype=torch.float32, device=device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    head = torch.nn.Linear(128, 1).to(device)
    _require(next(head.parameters()).is_cuda and next(head.parameters()).device == device, "UTR-LM probe head left CUDA")
    initial = head.weight.detach().clone()
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_weights = _source_group_weights(fit, device)
    validation_weights = _source_group_weights(by_split["VALIDATION"], device)
    best_state = None
    best_validation = float("inf")
    history = []
    for epoch in range(epochs):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = head(fit_features).squeeze(1)
        loss = ((prediction - fit_targets) ** 2 * train_weights).mean()
        _require(loss.is_cuda and loss.device == device and torch.isfinite(loss).item(), "UTR-LM probe loss left CUDA or became nonfinite")
        loss.backward()
        optimizer.step()
        head.eval()
        if result_stage == "HPO_VALIDATION_ONLY":
            with torch.no_grad():
                validation_error = head(features["VALIDATION"]).squeeze(1) - targets["VALIDATION"]
                validation = float(((validation_error ** 2) * validation_weights).mean())
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
                progress({"event": "PROBE_EPOCH_COMPLETED", **row})
    if result_stage == "FROZEN_DEVELOPMENT_TEST":
        best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
        best_validation = None
    _require(best_state is not None and not torch.equal(initial, head.weight.detach()), "UTR-LM probe had no GPU parameter update")
    head.load_state_dict(best_state)
    head.eval()
    predictions: dict[str, float] = {}
    with torch.no_grad():
        for split, rows in by_split.items():
            values = head(features[split]).squeeze(1)
            _require(values.is_cuda and values.device == device, "UTR-LM probe prediction silently fell back to CPU")
            predictions.update({record.record_id: float(value) for record, value in zip(rows, values.cpu())})
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
                    "baseline_id": "utrlm_siss_frozen_bos_linear_probe",
                    "predicted_direction_normalized_delta": predictions[record.record_id],
                }, sort_keys=True) + "\n")


def execute(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"output already exists: {output_dir}")
    _require(config["evaluation_outcomes_accessed"] is False, "UTR-LM baseline accessed Evaluation")
    _require(config["official_git_revision"] == OFFICIAL_REVISION, "UTR-LM revision changed")
    _require(str(config["device"]).startswith("cuda:"), "UTR-LM probe requires an explicit CUDA device")
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden for physical-device provenance")
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    physical_gpu_index = int(config["physical_gpu_index"])
    _require(0 <= physical_gpu_index < torch.cuda.device_count(), "physical GPU index is unavailable")
    device = torch.device(str(config["device"]))
    _require(device.index == physical_gpu_index, "CUDA device differs from declared physical GPU")
    torch.cuda.set_device(device)
    cuda_provenance = cuda_device_observation(physical_gpu_index, require_physical_index_match=True)
    torch.cuda.reset_peak_memory_stats(device)
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
        if row.get("event") == "PROBE_EPOCH_COMPLETED":
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(serialized)

    progress({
        "event": "RUN_STARTED",
        "device": str(device),
        "physical_gpu_index": physical_gpu_index,
        "cuda_device_uuid": cuda_provenance["cuda_device_uuid"],
        "result_stage": result_stage,
    })
    asset_root = Path(config["official_asset_root"])
    checkpoint_path = Path(config["checkpoint_path"])
    model, alphabet = load_official_encoder(asset_root, checkpoint_path, device)
    progress({"event": "OFFICIAL_ENCODER_LOADED"})
    started = time.time()
    embeddings = encode_bos_embeddings(
        model,
        alphabet,
        [sequence for record in records for sequence in (record.source, record.candidate)],
        device,
        int(config["embedding_batch_size"]),
        progress,
    )
    predictions, probe, artifact = train_probe(
        records,
        embeddings,
        device,
        int(config["seed"]),
        int(config["probe_epochs"]),
        float(config["probe_learning_rate"]),
        float(config["probe_weight_decay"]),
        result_stage,
        progress,
    )
    torch.cuda.synchronize(device)
    for split in output_splits:
        _write_predictions(output_dir / f"{split.lower()}_predictions.jsonl", records, predictions, split)
    for name in ("utrlm_probe_state.pt", "latest.pt", "best.pt"):
        torch.save(artifact, output_dir / name)
    summary = {
            "schema_version": "route_a_v3_route2_utrlm_baseline.v1",
            "status": "UTRLM_GSE114002_COMMON_TASK_COMPLETED",
            "result_stage": result_stage,
            "development_test_outcomes_evaluated": result_stage == "FROZEN_DEVELOPMENT_TEST",
            "baseline_id": "utrlm_siss_frozen_bos_linear_probe",
            "task": {"study_unit_id": TASK_STUDY, "region": TASK_REGION, "endpoint_id": TASK_ENDPOINT},
            "prediction_definition": "linear(BOS(candidate)-BOS(source))",
            "official_git_revision": OFFICIAL_REVISION,
            "official_checkpoint_name": checkpoint_path.name,
            "pretraining_objectives": ["MLM", "MFE", "SECONDARY_STRUCTURE"],
            "encoder_layers": 6,
            "encoder_attention_heads": 16,
            "encoder_embedding_dimension": 128,
            "pretrained_parameters_frozen": True,
            "pretrained_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "record_counts": {split: sum(record.split == split for record in records) for split in SPLITS},
            **{split.lower(): _metrics(records, predictions, split) for split in output_splits},
            "task_label_training_overlap": "NO_GSE114002_LABELS_USED_IN_ENCODER_PRETRAINING",
            "exact_sequence_pretraining_overlap": "UNKNOWN_NOT_ASSERTED",
            "independent_external_transfer_claim_allowed": False,
            "physical_gpu_index": physical_gpu_index,
            "device": str(device),
            "cpu_fallback_used": False,
            "cuda_training_tensors_verified": True,
            "evaluation_outcomes_accessed": False,
            "wall_time_seconds": time.time() - started,
            "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
            **cuda_provenance,
            "scientific_claim_status": "NOT_ESTABLISHED",
            **probe,
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
            entrypoint="run_route2_utrlm_baseline_v1",
            evaluation_outcomes_accessed=config.get("evaluation_outcomes_accessed", False),
        )
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
