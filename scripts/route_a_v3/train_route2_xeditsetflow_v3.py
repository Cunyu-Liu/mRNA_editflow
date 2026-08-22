#!/usr/bin/env python3
"""Train one frozen XEditSetFlow V3 screen or confirmation arm."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_development_projection_v3 import load_projection_rows
from core.route2_experiment_ledger import build_training_attempt_row, record_training_attempt
from core.route2_gpu_failure_evidence import cuda_device_observation, write_gpu_failure_evidence
from core.route2_source_token_cache_v3 import SourceTokenCacheIndexV3, load_source_token_cache_v3
from core.route2_xeditsetflow_runtime_v3 import (
    ARM_CONFIGS_V3,
    build_setflow_arm_v3,
    early_stop_update_v3,
    setflow_batch_loss_v3,
)
from core.route2_xeditsetflow_gate_v3 import require_setflow_confirmation_authorization_v3
from core.route2_xeditsetflow_training_v3 import (
    BalancedTaskSourceGroupPassSamplerV3,
    SetMarginalStateDatasetV3,
    collate_setflow_states_v3,
    expanded_state_batches_v3,
    setflow_records_from_projection_rows,
    setflow_vocabs,
)


class SetFlowTrainingV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SetFlowTrainingV3Error(message)


def _move(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _states_for_batch(
    dataset: SetMarginalStateDatasetV3,
    state_indices: Sequence[tuple[int, int]],
    cache: SourceTokenCacheIndexV3,
) -> dict[str, Any]:
    return collate_setflow_states_v3(
        [dataset.state(record_index, state_slot) for record_index, state_slot in state_indices],
        source_cache=cache,
    )


@torch.no_grad()
def evaluate_common_validation_nll_v3(
    model: torch.nn.Module,
    arm: str,
    dataset: SetMarginalStateDatasetV3,
    *,
    source_cache: SourceTokenCacheIndexV3,
    batch_size: int,
    device: torch.device,
    bf16: bool,
) -> tuple[float, int]:
    model.eval()
    dataset.set_pass(0)
    all_states = [
        (record_index, state_slot)
        for record_index in range(len(dataset.records))
        for state_slot in range(dataset.states_per_record)
    ]
    weighted_loss = 0.0
    total_weight = 0.0
    active_state_count = 0
    for start in range(0, len(all_states), batch_size):
        raw = _states_for_batch(dataset, all_states[start : start + batch_size], source_cache)
        batch = _move(raw, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
            loss, active_weight = setflow_batch_loss_v3(model, arm, batch)
        value = float(loss.float().cpu())
        weight = float(active_weight.float().cpu())
        _require(math.isfinite(value) and math.isfinite(weight), "common Validation NLL is nonfinite")
        weighted_loss += value * weight
        total_weight += weight
        active_state_count += int((~batch["structural_budget_exhausted"]).sum().item())
    _require(total_weight > 0.0 and active_state_count > 0, "common Validation set is empty")
    return weighted_loss / total_weight, active_state_count


def train(config: Mapping[str, Any], *, arm: str, output_dir: Path) -> dict[str, Any]:
    _require(arm in ARM_CONFIGS_V3, "unknown SetFlow V3 training arm")
    _require(not output_dir.exists(), f"terminal SetFlow output already exists: {output_dir}")
    _require(int(config["batch_size"]) == 32, "SetFlow effective batch size changed")
    _require(int(config["states_per_record"]) == 2, "SetFlow states per record changed")
    _require(int(config["record_repeat_cap"]) == 4, "SetFlow record repeat cap changed")
    _require(int(config["maximum_passes"]) == 12 and int(config["early_stopping_patience"]) == 2, "SetFlow pass/early-stop rule changed")
    _require(int(config["seed"]) in {20260903, 20260904, 20260905, 20260906}, "undeclared SetFlow seed")
    require_setflow_confirmation_authorization_v3(config, arm=arm)
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden")
    device = torch.device(str(config["device"]))
    physical_gpu = int(config["physical_gpu_index"])
    _require(device == torch.device(f"cuda:{physical_gpu}"), "SetFlow device provenance changed")
    torch.cuda.set_device(device)
    _require(torch.cuda.is_bf16_supported(), "BF16 is unavailable on the selected GPU")
    cuda = cuda_device_observation(physical_gpu, require_physical_index_match=True)

    train_rows = load_projection_rows([Path(config["train_projection_path"])], allowed_splits=("TRAIN",))
    validation_rows = load_projection_rows([Path(config["validation_projection_path"])], allowed_splits=("VALIDATION",))
    train_records, train_eligibility = setflow_records_from_projection_rows(train_rows)
    validation_records, validation_eligibility = setflow_records_from_projection_rows(validation_rows)
    _require(len(train_records) == int(config["expected_train_record_count"]), "SetFlow TRAIN count changed")
    _require(len(validation_records) == int(config["expected_validation_record_count"]), "SetFlow VALIDATION count changed")
    vocabs = setflow_vocabs(train_records)
    cache_payload = load_source_token_cache_v3(Path(config["source_token_cache_path"]))
    source_cache = SourceTokenCacheIndexV3(cache_payload)
    _require(len(cache_payload["record_ids"]) == len(train_records) + len(validation_records), "source cache cohort changed")
    train_dataset = SetMarginalStateDatasetV3(
        train_records, vocabs, seed=int(config["seed"]), states_per_record=2
    )
    validation_dataset = SetMarginalStateDatasetV3(
        validation_records, vocabs, seed=int(config["common_validation_state_seed"]), states_per_record=2
    )
    sampler = BalancedTaskSourceGroupPassSamplerV3(
        train_records,
        record_batch_size=int(config["batch_size"]) // 2,
        seed=int(config["seed"]),
        repeat_cap=4,
    )
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    model, model_config = build_setflow_arm_v3(
        arm, vocabs=vocabs, dropout=float(config["dropout"])
    )
    model = model.to(device)
    _require(next(model.parameters()).is_cuda, "SetFlow model parameters left CUDA")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        fused=True,
    )
    initial_parameter = next(model.parameters()).detach().clone()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    (output_dir / "training_config.json").write_text(
        json.dumps({**dict(config), "arm": arm}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    ledger = Path(config["experiment_ledger_path"])
    attempt_details = {
        "started_at": started_at,
        "record_counts": {"TRAIN": len(train_records), "VALIDATION": len(validation_records)},
        "development_test_record_count_withheld": int(config["withheld_development_test_record_count"]),
        "evaluation_record_count": 0,
        "included_study_unit_ids": sorted({record.study for record in train_records}),
        "included_regions": ["5UTR", "3UTR"],
        "trainable_parameter_count": model_config["trainable_parameter_count"],
        "frozen_pretrained_parameter_count": int(cache_payload["pretrained_parameter_count"]),
        "total_effective_parameter_count": model_config["trainable_parameter_count"] + int(cache_payload["pretrained_parameter_count"]),
    }
    run_config = {
        **dict(config),
        "run_id": f"xeditsetflow_v3_{arm}_seed{config['seed']}",
        "baseline_id": f"xeditsetflow_v3_{arm}_seed{config['seed']}",
        "attempt_purpose": "XEDITSETFLOW_V3_SCREEN" if int(config["seed"]) == 20260903 else "XEDITSETFLOW_V3_CONFIRMATION",
        "scientific_role": ARM_CONFIGS_V3[arm]["model_kind"],
        "result_stage": "DEVELOPMENT_VALIDATION",
        "model_kind": ARM_CONFIGS_V3[arm]["model_kind"],
        "epochs": int(config["maximum_passes"]),
        "hidden_dim": model_config.get("model_width", model_config.get("hidden_dim")),
        "depth": model_config["depth"],
        "loss_kind": "SET_MARGINAL_LIKELIHOOD",
        "training_precision": "BF16",
        "optimizer_name": "AdamW",
        "optimizer_fused": True,
        "generation_action_space": "SUB+STOP",
    }
    record_training_attempt(
        ledger,
        output_dir / "training_attempt.json",
        build_training_attempt_row(run_config, output_dir, "RUNNING", repository_root=REPO_ROOT, details=attempt_details),
    )

    def checkpoint(pass_index: int, optimizer_steps: int) -> dict[str, Any]:
        return {
            "schema_version": "route_a_v3_route2_xeditsetflow_checkpoint.v3",
            "arm": arm,
            "model_config": model_config,
            "model_state": model.state_dict(),
            "vocabs": vocabs,
            "allowed_edit_budgets": [1, 3, 5],
            "completed_pass": pass_index,
            "source_token_cache_path": str(config["source_token_cache_path"]),
            "common_validation_state_seed": int(config["common_validation_state_seed"]),
            "training_provenance": {
                "seed": int(config["seed"]),
                "optimizer_steps": optimizer_steps,
                "parameter_changed": not torch.equal(initial_parameter, next(model.parameters()).detach()),
                "torch_device": str(device),
                "physical_gpu_index": physical_gpu,
                "cpu_fallback_used": False,
                "cuda_training_tensors_verified": optimizer_steps > 0,
                **cuda,
            },
        }

    history: list[dict[str, Any]] = []
    best: float | None = None
    best_pass: int | None = None
    stale = 0
    optimizer_steps = 0
    start = time.time()
    stopped_early = False
    for pass_index in range(int(config["maximum_passes"])):
        model.train()
        train_dataset.set_pass(pass_index)
        sampler.set_pass(pass_index)
        state_batches = expanded_state_batches_v3(
            sampler.batches_for_pass(),
            states_per_record=2,
            batch_size=int(config["batch_size"]),
        )
        weighted_loss = 0.0
        total_weight = 0.0
        for state_indices in state_batches:
            batch = _move(_states_for_batch(train_dataset, state_indices, source_cache), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, active_weight = setflow_batch_loss_v3(model, arm, batch)
            _require(loss.is_cuda and torch.isfinite(loss).item(), "SetFlow training loss is invalid")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            _require(torch.isfinite(gradient_norm).item(), "SetFlow gradient norm is nonfinite")
            optimizer.step()
            optimizer_steps += 1
            value = float(loss.detach().float().cpu())
            weight = float(active_weight.detach().float().cpu())
            weighted_loss += value * weight
            total_weight += weight
        validation_nll, validation_active = evaluate_common_validation_nll_v3(
            model,
            arm,
            validation_dataset,
            source_cache=source_cache,
            batch_size=int(config["batch_size"]),
            device=device,
            bf16=True,
        )
        improved, best, stale, should_stop = early_stop_update_v3(
            validation_nll,
            best=best,
            stale_passes=stale,
            patience=int(config["early_stopping_patience"]),
        )
        row = {
            "pass": pass_index + 1,
            "train_set_marginal_nll": weighted_loss / total_weight,
            "validation_common_set_marginal_nll": validation_nll,
            "validation_active_state_count": validation_active,
            "optimizer_steps": optimizer_steps,
            "improved": improved,
            "stale_passes": stale,
        }
        history.append(row)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        torch.save(checkpoint(pass_index + 1, optimizer_steps), output_dir / "latest.pt")
        if improved:
            best_pass = pass_index + 1
            torch.save(checkpoint(pass_index + 1, optimizer_steps), output_dir / "best.pt")
        print(json.dumps({"event": "XEDITSETFLOW_V3_PASS_COMPLETE", "arm": arm, **row}, sort_keys=True), flush=True)
        if should_stop:
            stopped_early = True
            break
    _require(best_pass is not None and best is not None, "SetFlow selected no checkpoint")
    parameter_changed = not torch.equal(initial_parameter, next(model.parameters()).detach())
    _require(optimizer_steps > 0 and parameter_changed, "SetFlow performed no parameter update")
    elapsed = time.time() - start
    summary = {
        "schema_version": "route_a_v3_route2_xeditsetflow_training_summary.v3",
        "status": "XEDITSETFLOW_V3_GPU_TRAINING_COMPLETE",
        "arm": arm,
        "selectable": bool(ARM_CONFIGS_V3[arm]["selectable"]),
        "seed": int(config["seed"]),
        "run_stage": str(
            config.get(
                "run_stage",
                "SCREEN" if int(config["seed"]) == 20260903 else "CONFIRMATION",
            )
        ),
        "train_record_count": len(train_records),
        "validation_record_count": len(validation_records),
        "over_budget_excluded_record_counts": {
            "TRAIN": train_eligibility["skipped_over_budget_count"],
            "VALIDATION": validation_eligibility["skipped_over_budget_count"],
        },
        "states_per_record_per_pass": 2,
        "effective_batch_size": int(config["batch_size"]),
        "completed_passes": len(history),
        "maximum_passes": int(config["maximum_passes"]),
        "stopped_early": stopped_early,
        "selected_pass": best_pass,
        "best_validation_common_set_marginal_nll": best,
        "optimizer_steps": optimizer_steps,
        "trainable_parameter_count": model_config["trainable_parameter_count"],
        "frozen_pretrained_parameter_count": int(cache_payload["pretrained_parameter_count"]),
        "parameter_changed": parameter_changed,
        "history": history,
        "wall_time_seconds": elapsed,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "physical_gpu_index": physical_gpu,
        "torch_device": str(device),
        "training_precision": "BF16",
        "cpu_fallback_used": False,
        "development_test_record_count_withheld": int(config["withheld_development_test_record_count"]),
        "development_test_outcomes_accessed": False,
        "evaluation_records_read": 0,
        "evaluation_outcomes_accessed": False,
        "critic_score_used": False,
        "independent_evaluator_used": False,
        "history_is_terminal": True,
        **cuda,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record_training_attempt(
        ledger,
        output_dir / "training_attempt.json",
        build_training_attempt_row(
            run_config,
            output_dir,
            "COMPLETED",
            repository_root=REPO_ROOT,
            details={
                **attempt_details,
                "optimizer_steps": optimizer_steps,
                "selected_epoch": best_pass,
                "wall_time_seconds": elapsed,
                "peak_vram_mb": summary["peak_vram_mb"],
                "notes": "unguided SetFlow V3; common Validation set-marginal NLL selection; no critic/evaluator/outcome guidance",
            },
        ),
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=tuple(ARM_CONFIGS_V3))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or Path(config["output_root"]) / args.arm
    try:
        result = train(config, arm=args.arm, output_dir=output_dir)
    except Exception as exc:
        write_gpu_failure_evidence(
            output_dir.with_name(output_dir.name + ".failed.json"),
            config,
            exc,
            entrypoint="train_route2_xeditsetflow_v3",
            evaluation_outcomes_accessed=False,
        )
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
