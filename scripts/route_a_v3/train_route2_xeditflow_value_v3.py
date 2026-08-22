#!/usr/bin/env python3
"""Distil one authorized XEditFlow V3 scalar potential on CUDA/BF16."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_experiment_ledger import build_training_attempt_row, record_training_attempt
from core.route2_gpu_failure_evidence import cuda_device_observation, write_gpu_failure_evidence
from core.route2_source_token_cache_v3 import SourceTokenCacheIndexV3, load_source_token_cache_v3
from core.route2_xeditflow_guidance_v3 import XEditValueV3
from core.route2_xeditflow_value_training_v3 import (
    BASE_FLOW_SEEDS_V3,
    VALUE_CHECKPOINT_SCHEMA_V3,
    VALUE_TARGET_SCHEMA_V3,
    collate_value_targets_v3,
    require_value_training_authorization_v3,
    value_distillation_loss_v3,
    value_target_records_v3,
)


class XEditFlowValueTrainerV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowValueTrainerV3Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _target_payload(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        return _json(path)
    value = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(value, dict), "value target artifact is not a mapping")
    return value


def validate_value_training_config_v3(
    config: Mapping[str, Any], target_payload: Mapping[str, Any]
) -> dict[str, int]:
    _require(config.get("schema_version") == "route_a_v3_route2_xeditflow_value_training_config.v1", "unexpected value training config schema")
    _require(target_payload.get("schema_version") == VALUE_TARGET_SCHEMA_V3, "unexpected value target schema")
    seed = int(config.get("base_flow_training_seed", -1))
    _require(seed in BASE_FLOW_SEEDS_V3, "undeclared value base-flow training seed")
    _require(int(target_payload.get("base_flow_training_seed", -1)) == seed, "value target and training seed differ")
    _require(float(config.get("kappa", -1)) == float(target_payload.get("kappa", -2)), "value target kappa differs")
    _require(float(config.get("temperature", -1)) == float(target_payload.get("temperature", -2)), "value target temperature differs")
    _require(int(config.get("passes", -1)) == 8, "value training passes changed")
    _require(int(config.get("batch_size", -1)) == 32, "value training batch size changed")
    _require(str(config.get("precision")) == "BF16", "value training precision changed")
    _require(float(config.get("learning_rate", -1)) == 3e-4, "value training learning rate changed")
    _require(float(config.get("weight_decay", -1)) == 1e-4, "value training weight decay changed")
    _require(float(config.get("gradient_clip_norm", -1)) == 1.0, "value gradient clipping changed")
    _require(float(config.get("dropout", -1)) == 0.10, "value dropout changed")
    _require(config.get("checkpoint_selection") == "FINAL_PASS_NO_EPOCH_RESELECTION", "value checkpoint rule changed")
    records = value_target_records_v3(target_payload)
    fields = {
        "assay_count": "assay_id",
        "context_count": "context_id",
        "quantity_count": "quantity_id",
        "measurement_count": "measurement_id",
        "numerator_count": "numerator_id",
        "denominator_count": "denominator_id",
    }
    sizes = {
        count_name: max(int(getattr(record, id_name)) for record in records) + 1
        for count_name, id_name in fields.items()
    }
    _require(min(sizes.values()) >= 1, "value endpoint vocabulary is empty")
    return sizes


def _move(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def train_value_v3(config: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"terminal value output already exists: {output_dir}")
    target_path = Path(config["value_target_path"])
    target_payload = _target_payload(target_path)
    sizes = validate_value_training_config_v3(config, target_payload)
    critic_readiness = _json(Path(config["critic_readiness_path"]))
    setflow_confirmation = _json(Path(config["setflow_confirmation_path"]))
    require_value_training_authorization_v3(critic_readiness, setflow_confirmation)
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden")
    physical_gpu = int(config["physical_gpu_index"])
    device = torch.device(str(config["device"]))
    _require(device == torch.device(f"cuda:{physical_gpu}"), "value device provenance changed")
    torch.cuda.set_device(device)
    _require(torch.cuda.is_bf16_supported(), "BF16 is unavailable on the selected GPU")
    cuda = cuda_device_observation(physical_gpu, require_physical_index_match=True)
    records = value_target_records_v3(target_payload)
    source_cache_payload = load_source_token_cache_v3(Path(config["source_token_cache_path"]))
    source_cache = SourceTokenCacheIndexV3(source_cache_payload)
    seed = int(config["base_flow_training_seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = XEditValueV3(**sizes, dropout=float(config["dropout"])).to(device)
    _require(next(model.parameters()).is_cuda, "value model parameters left CUDA")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        fused=True,
    )
    initial = next(model.parameters()).detach().clone()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    (output_dir / "training_config.json").write_text(
        json.dumps(dict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    ledger = Path(config["experiment_ledger_path"])
    run_config = {
        **dict(config),
        "run_id": f"xeditflow_v3_value_seed{seed}_k{config['kappa']}_t{config['temperature']}",
        "baseline_id": "xeditflow_v3_scalar_value",
        "attempt_purpose": "XEDITFLOW_V3_VALUE_DISTILLATION",
        "scientific_role": "SCALAR_SOFT_VALUE_POTENTIAL",
        "result_stage": "DEVELOPMENT_TRAIN",
        "model_kind": "XEDIT_VALUE_V3_6X384",
        "epochs": 8,
        "hidden_dim": 384,
        "depth": 6,
        "loss_kind": "HUBER_DELTA_1",
        "training_precision": "BF16",
        "optimizer_name": "AdamW",
        "optimizer_fused": True,
        "generation_action_space": "SUB+STOP",
    }
    attempt_details = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "train_state_count": len(records),
        "rollouts_per_state": 8,
        "critic_ensemble_size": 3,
        "trainable_parameter_count": model.trainable_parameter_count,
        "development_test_outcome_read_count": 0,
        "evaluation_outcome_read_count": 0,
    }
    record_training_attempt(
        ledger,
        output_dir / "training_attempt.json",
        build_training_attempt_row(run_config, output_dir, "RUNNING", repository_root=REPO_ROOT, details=attempt_details),
    )
    optimizer_steps = 0
    history = []
    started = time.time()
    model.train()
    for pass_index in range(8):
        generator = torch.Generator().manual_seed(seed + pass_index)
        order = torch.randperm(len(records), generator=generator).tolist()
        losses = []
        for start in range(0, len(order), int(config["batch_size"])):
            batch_records = [
                records[index]
                for index in order[start : start + int(config["batch_size"])]
            ]
            batch = _move(collate_value_targets_v3(batch_records, source_cache=source_cache), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = value_distillation_loss_v3(model, batch)
            _require(loss.is_cuda and bool(torch.isfinite(loss).item()), "value CUDA loss is invalid")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            _require(bool(torch.isfinite(gradient_norm).item()), "value gradient norm is nonfinite")
            optimizer.step()
            optimizer_steps += 1
            losses.append(float(loss.detach().float().cpu()))
        row = {
            "pass": pass_index + 1,
            "train_huber": math.fsum(losses) / len(losses),
            "optimizer_steps": optimizer_steps,
        }
        history.append(row)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps({"event": "XEDITFLOW_V3_VALUE_PASS_COMPLETE", **row}, sort_keys=True), flush=True)
    parameter_changed = not torch.equal(initial, next(model.parameters()).detach())
    _require(parameter_changed and optimizer_steps > 0, "value model did not receive a parameter update")
    checkpoint = {
        "schema_version": VALUE_CHECKPOINT_SCHEMA_V3,
        "model_state": model.state_dict(),
        "model_config": {**sizes, "dropout": float(config["dropout"]), "blocks": 6, "width": 384},
        "base_flow_training_seed": seed,
        "critic_seeds": list(target_payload["critic_seeds"]),
        "kappa": float(config["kappa"]),
        "temperature": float(config["temperature"]),
        "selected_pass": 8,
        "checkpoint_selection": "FINAL_PASS_NO_EPOCH_RESELECTION",
        "training_provenance": {
            "optimizer_steps": optimizer_steps,
            "parameter_changed": True,
            "cpu_fallback_used": False,
            "torch_device": str(device),
            "physical_gpu_index": physical_gpu,
            **cuda,
        },
    }
    torch.save(checkpoint, output_dir / "value_checkpoint.pt")
    result = {
        "schema_version": "route_a_v3_route2_xeditflow_value_training.v3",
        "status": "XEDITFLOW_V3_VALUE_TRAINING_COMPLETE",
        "base_flow_training_seed": seed,
        "kappa": float(config["kappa"]),
        "temperature": float(config["temperature"]),
        "state_count": len(records),
        "rollouts_per_state": 8,
        "critic_ensemble_size": 3,
        "completed_passes": 8,
        "selected_pass": 8,
        "optimizer_steps": optimizer_steps,
        "trainable_parameter_count": model.trainable_parameter_count,
        "parameter_changed": True,
        "final_train_huber": history[-1]["train_huber"],
        "history": history,
        "wall_time_seconds": time.time() - started,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "training_precision": "BF16",
        "cpu_fallback_used": False,
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    record_training_attempt(
        ledger,
        output_dir / "training_attempt.json",
        build_training_attempt_row(run_config, output_dir, "COMPLETED", repository_root=REPO_ROOT, details={**attempt_details, **result}),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = _json(args.config)
    try:
        result = train_value_v3(config, output_dir=args.output_dir)
    except Exception as exc:
        write_gpu_failure_evidence(
            args.output_dir.with_name(args.output_dir.name + ".failed.json"),
            config,
            exc,
            entrypoint="train_route2_xeditflow_value_v3",
            evaluation_outcomes_accessed=False,
        )
        raise
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
