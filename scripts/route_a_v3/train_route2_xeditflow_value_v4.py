#!/usr/bin/env python3
"""Distil one authorized mode-conditioned XEditFlow V4 scalar potential."""

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
from core.route2_xeditflow_guidance_v4 import XEditValueV4
from core.route2_xeditflow_value_training_v4 import (
    BASE_FLOW_SEEDS_V4,
    MODE_IDS_V4,
    VALUE_CHECKPOINT_SCHEMA_V4,
    VALUE_TARGET_SCHEMA_V4,
    collate_value_targets_v4,
    require_value_training_authorization_v4,
    value_distillation_loss_v4,
    value_target_records_v4,
)


class XEditFlowValueTrainerV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowValueTrainerV4Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _target_payload(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        return _json(path)
    value = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(value, dict), "V4 value target artifact is not a mapping")
    return value


def validate_value_training_config_v4(
    config: Mapping[str, Any], target_payload: Mapping[str, Any]
) -> dict[str, int]:
    _require(
        config.get("schema_version")
        == "route_a_v3_route2_xeditflow_value_training_config.v4",
        "unexpected V4 value training config schema",
    )
    _require(
        target_payload.get("schema_version") == VALUE_TARGET_SCHEMA_V4,
        "unexpected V4 value target schema",
    )
    seed = int(config.get("base_flow_training_seed", -1))
    _require(seed in BASE_FLOW_SEEDS_V4, "undeclared V4 value base-flow seed")
    _require(
        int(target_payload.get("base_flow_training_seed", -1)) == seed,
        "V4 value target and training seed differ",
    )
    _require(
        float(config.get("kappa", -1))
        == float(target_payload.get("kappa", -2)),
        "V4 value target kappa differs",
    )
    _require(
        float(config.get("temperature", -1))
        == float(target_payload.get("temperature", -2)),
        "V4 value target temperature differs",
    )
    _require(int(config.get("passes", -1)) == 8, "V4 value training passes changed")
    _require(int(config.get("batch_size", -1)) == 32, "V4 value training batch size changed")
    _require(str(config.get("precision")) == "BF16", "V4 value training precision changed")
    _require(
        float(config.get("learning_rate", -1)) == 3e-4,
        "V4 value training learning rate changed",
    )
    _require(
        float(config.get("weight_decay", -1)) == 1e-4,
        "V4 value training weight decay changed",
    )
    _require(
        float(config.get("gradient_clip_norm", -1)) == 1.0,
        "V4 value gradient clipping changed",
    )
    _require(float(config.get("dropout", -1)) == 0.10, "V4 value dropout changed")
    _require(
        config.get("checkpoint_selection")
        == "FINAL_PASS_8_NO_EPOCH_RESELECTION",
        "V4 value checkpoint rule changed",
    )
    _require(
        int(config.get("physical_gpu_index", -1)) in set(range(6)),
        "V4 value GPU is outside 0-5",
    )
    _require(
        str(config.get("device"))
        == f"cuda:{int(config.get('physical_gpu_index', -1))}",
        "V4 value device provenance changed",
    )
    _require(
        str(config.get("output_dir", "")).startswith(
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"
        ),
        "V4 value output left Route 2 /mnt",
    )
    _require(
        config.get("independent_evaluator_used") is False,
        "independent evaluator entered V4 value training config",
    )
    _require(
        config.get("development_test_outcomes_accessed_after_atomic_test") is False,
        "V4 value training config reopened Development TEST",
    )
    _require(
        config.get("new_final_evaluation_outcomes_accessed") is False,
        "V4 value training config accessed Evaluation outcomes",
    )
    records = value_target_records_v4(target_payload)
    _require(
        {record.trajectory_mode_id for record in records} == set(MODE_IDS_V4),
        "V4 value target artifact does not cover all eight modes",
    )
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
    _require(min(sizes.values()) >= 1, "V4 value endpoint vocabulary is empty")
    return sizes


def _move(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) for key, value in batch.items()
    }


def initialize_value_model_and_optimizer_v4(
    *,
    seed: int,
    sizes: Mapping[str, int],
    dropout: float,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
) -> tuple[XEditValueV4, torch.optim.Optimizer]:
    """Seed before constructing either the value model or its optimizer."""

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = XEditValueV4(**dict(sizes), dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=True,
    )
    return model, optimizer


def train_value_v4(
    config: Mapping[str, Any], *, output_dir: Path
) -> dict[str, Any]:
    _require(
        not output_dir.exists(),
        f"terminal V4 value output already exists: {output_dir}",
    )
    _require(
        output_dir == Path(str(config["output_dir"])),
        "V4 value output path differs from frozen config",
    )
    target_payload = _target_payload(Path(config["value_target_path"]))
    sizes = validate_value_training_config_v4(config, target_payload)
    critic_readiness = _json(Path(config["critic_readiness_path"]))
    setflow_confirmation = _json(Path(config["setflow_confirmation_path"]))
    require_value_training_authorization_v4(
        critic_readiness, setflow_confirmation
    )
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    _require(
        not os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES remapping is forbidden",
    )
    physical_gpu = int(config["physical_gpu_index"])
    device = torch.device(str(config["device"]))
    torch.cuda.set_device(device)
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    _require(bf16_supported, "BF16 is unavailable on selected GPU")
    cuda = cuda_device_observation(
        physical_gpu, require_physical_index_match=True
    )
    _require(
        "A100" in str(cuda.get("cuda_device_name", "")),
        "V4 value training requires an actual NVIDIA A100",
    )
    records = value_target_records_v4(target_payload)
    source_cache = SourceTokenCacheIndexV3(
        load_source_token_cache_v3(Path(config["source_token_cache_path"]))
    )
    seed = int(config["base_flow_training_seed"])
    model, optimizer = initialize_value_model_and_optimizer_v4(
        seed=seed,
        sizes=sizes,
        dropout=float(config["dropout"]),
        device=device,
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    _require(next(model.parameters()).is_cuda, "V4 value parameters left CUDA")
    initial = model.scalar_head[-1].weight.detach().clone()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    checkpoint_path = output_dir / "value_checkpoint.pt"
    execution_provenance = {
        "parameter_initialization_seed": seed,
        "parameter_initialization_seed_applied_before_model_construction": True,
        "cuda_available": True,
        "bf16_supported": bf16_supported,
        "training_precision": "BF16",
        "cpu_fallback_used": False,
        "torch_device": str(device),
        "physical_gpu_index": physical_gpu,
        "value_checkpoint_path": str(checkpoint_path),
        **cuda,
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(dict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    ledger = Path(config["experiment_ledger_path"])
    run_config = {
        **dict(config),
        "run_id": (
            f"xeditflow_v4_value_seed{seed}_k{config['kappa']}_"
            f"t{config['temperature']}"
        ),
        "baseline_id": "xeditflow_v4_mode_conditioned_scalar_value",
        "attempt_purpose": "XEDITFLOW_V4_VALUE_DISTILLATION",
        "scientific_role": "MODE_CONDITIONED_SCALAR_SOFT_VALUE_POTENTIAL",
        "result_stage": "DEVELOPMENT_TRAIN",
        "model_kind": "XEDIT_VALUE_V4_6X384_MODE8",
        "epochs": 8,
        "hidden_dim": 384,
        "depth": 6,
        "loss_kind": "HUBER_DELTA_1",
        "training_precision": "BF16",
        "optimizer_name": "AdamW",
        "optimizer_fused": True,
        "generation_action_space": "SUB+STOP",
        **execution_provenance,
    }
    attempt_details = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "train_state_mode_count": len(records),
        "rollouts_per_state_mode": 8,
        "critic_ensemble_size": 3,
        "trajectory_mode_count": 8,
        "trainable_parameter_count": model.trainable_parameter_count,
        **execution_provenance,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcome_read_count": 0,
    }
    record_training_attempt(
        ledger,
        output_dir / "training_attempt.json",
        build_training_attempt_row(
            run_config,
            output_dir,
            "RUNNING",
            repository_root=REPO_ROOT,
            details=attempt_details,
        ),
    )
    torch.cuda.reset_peak_memory_stats(device)
    optimizer_steps = 0
    history: list[dict[str, Any]] = []
    started = time.time()
    model.train()
    for pass_index in range(8):
        generator = torch.Generator().manual_seed(seed + pass_index)
        order = torch.randperm(len(records), generator=generator).tolist()
        losses: list[float] = []
        for start in range(0, len(order), int(config["batch_size"])):
            batch_records = [
                records[index]
                for index in order[start : start + int(config["batch_size"])]
            ]
            batch = _move(
                collate_value_targets_v4(
                    batch_records, source_cache=source_cache
                ),
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = value_distillation_loss_v4(model, batch)
            _require(
                loss.is_cuda and bool(torch.isfinite(loss).item()),
                "V4 value CUDA loss is invalid",
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip_norm"])
            )
            _require(
                bool(torch.isfinite(gradient_norm).item()),
                "V4 value gradient norm is nonfinite",
            )
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
        print(
            json.dumps(
                {"event": "XEDITFLOW_V4_VALUE_PASS_COMPLETE", **row},
                sort_keys=True,
            ),
            flush=True,
        )
    parameter_changed = not torch.equal(
        initial, model.scalar_head[-1].weight.detach()
    )
    _require(
        parameter_changed and optimizer_steps > 0,
        "V4 value model did not receive a parameter update",
    )
    checkpoint = {
        "schema_version": VALUE_CHECKPOINT_SCHEMA_V4,
        "model_state_dict": model.state_dict(),
        "model_config": {
            **sizes,
            "dropout": float(config["dropout"]),
            "blocks": 6,
            "width": 384,
            "heads": 8,
            "ffn_width": 1536,
            "trajectory_mode_count": 8,
        },
        "base_flow_training_seed": seed,
        "critic_seeds": list(target_payload["critic_seeds"]),
        "kappa": float(config["kappa"]),
        "temperature": float(config["temperature"]),
        "selected_pass": 8,
        "checkpoint_selection": "FINAL_PASS_8_NO_EPOCH_RESELECTION",
        "training_provenance": {
            **execution_provenance,
            "optimizer_steps": optimizer_steps,
            "parameter_changed": True,
        },
    }
    torch.save(checkpoint, checkpoint_path)
    result = {
        "schema_version": "route_a_v3_route2_xeditflow_value_training.v4",
        "status": "XEDITFLOW_V4_VALUE_TRAINING_COMPLETE",
        "base_flow_training_seed": seed,
        "kappa": float(config["kappa"]),
        "temperature": float(config["temperature"]),
        "state_mode_count": len(records),
        "rollouts_per_state_mode": 8,
        "critic_ensemble_size": 3,
        "trajectory_mode_count": 8,
        "completed_passes": 8,
        "selected_pass": 8,
        "optimizer_steps": optimizer_steps,
        "trainable_parameter_count": model.trainable_parameter_count,
        "parameter_changed": True,
        **execution_provenance,
        "final_train_huber": history[-1]["train_huber"],
        "history": history,
        "wall_time_seconds": time.time() - started,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "setflow_mode_is_fixed_trajectory_state": True,
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    record_training_attempt(
        ledger,
        output_dir / "training_attempt.json",
        build_training_attempt_row(
            run_config,
            output_dir,
            "COMPLETED",
            repository_root=REPO_ROOT,
            details={**attempt_details, **result},
        ),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    config = _json(arguments.config)
    try:
        result = train_value_v4(config, output_dir=arguments.output_dir)
    except Exception as exc:
        write_gpu_failure_evidence(
            arguments.output_dir.with_name(
                arguments.output_dir.name + ".failed.json"
            ),
            config,
            exc,
            entrypoint="train_route2_xeditflow_value_v4",
            evaluation_outcomes_accessed=False,
        )
        raise
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
