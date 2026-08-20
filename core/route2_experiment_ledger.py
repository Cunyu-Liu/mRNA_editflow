"""Maintain the Route 2 training-attempt table and per-run record."""

from __future__ import annotations

import csv
import fcntl
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


TRAINING_ATTEMPT_COLUMNS = (
    "attempt_id",
    "status",
    "attempt_purpose",
    "started_at",
    "updated_at",
    "completed_at",
    "code_commit",
    "baseline_id",
    "scientific_role",
    "result_stage",
    "run_mode",
    "model_kind",
    "pretrained_model_id",
    "pretrained_feature_cache_path",
    "development_manifest",
    "canonical_dataset_ids",
    "canonical_paths",
    "included_study_unit_ids",
    "included_regions",
    "train_record_count",
    "validation_record_count",
    "test_record_count",
    "withheld_test_record_count",
    "evaluation_record_count",
    "hidden_dim",
    "depth",
    "batch_size",
    "epochs",
    "learning_rate",
    "weight_decay",
    "loss_kind",
    "huber_delta",
    "metadata_mode",
    "training_weighting_mode",
    "training_sampling_mode",
    "loss_aggregation_mode",
    "target_scaling_mode",
    "candidate_control",
    "seed",
    "physical_gpu_index",
    "device",
    "optimizer_name",
    "optimizer_fused",
    "training_precision",
    "encoder_attention_backend",
    "pretrained_position_encoding",
    "critic_position_features",
    "generation_action_space",
    "generator_position_features",
    "algorithmic_time_feature",
    "num_workers",
    "pin_memory",
    "persistent_workers",
    "prefetch_factor",
    "non_blocking_transfer",
    "torch_compile",
    "trainable_parameter_count",
    "frozen_pretrained_parameter_count",
    "total_effective_parameter_count",
    "optimizer_steps",
    "selected_epoch",
    "validation_spearman",
    "validation_task_macro_spearman",
    "validation_mae",
    "validation_task_macro_standardized_mae",
    "test_spearman",
    "test_task_macro_spearman",
    "test_mae",
    "peak_vram_mb",
    "wall_time_seconds",
    "output_directory",
    "notes",
    "error_type",
    "error",
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _git_commit(repository_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _join(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, str):
        return values
    return ";".join(str(value) for value in values)


def canonical_dataset_ids(canonical_paths: Any) -> list[str]:
    result = []
    for raw_path in canonical_paths or []:
        path = Path(raw_path)
        parts = list(path.parts)
        if "canonical" in parts and parts.index("canonical") + 1 < len(parts):
            dataset_id = parts[parts.index("canonical") + 1]
        else:
            dataset_id = path.parent.parent.name or path.stem
        if dataset_id not in result:
            result.append(dataset_id)
    return result


def _metric(metrics: Any, key: str) -> Any:
    if not isinstance(metrics, Mapping):
        return ""
    value = metrics.get(key)
    return "" if value is None else value


def build_training_attempt_row(
    config: Mapping[str, Any],
    output_dir: Path,
    status: str,
    *,
    repository_root: Path,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one aggregate row; no sequence or per-record payload is included."""
    details = dict(details or {})
    now = _now()
    record_counts = details.get("record_counts") or {}
    validation_metrics = details.get("validation_metrics")
    test_metrics = details.get("test_metrics")
    baseline_id = str(
        config.get("baseline_id") or config.get("run_id") or output_dir.name
    )
    attempt_id = str(config.get("attempt_id") or f"{baseline_id}::{output_dir.name}")
    pretrained_count = details.get(
        "frozen_pretrained_parameter_count",
        config.get("expected_frozen_pretrained_parameter_count", ""),
    )
    trainable_count = details.get(
        "trainable_parameter_count",
        config.get("expected_trainable_parameter_count", ""),
    )
    total_count = details.get("total_effective_parameter_count", "")
    if total_count == "" and trainable_count != "" and pretrained_count != "":
        total_count = int(trainable_count) + int(pretrained_count)
    return {
        "attempt_id": attempt_id,
        "status": status,
        "attempt_purpose": config.get("attempt_purpose", ""),
        "started_at": details.get("started_at", now if status == "RUNNING" else ""),
        "updated_at": now,
        "completed_at": now if status in {"COMPLETED", "FAILED"} else "",
        "code_commit": details.get("code_commit", _git_commit(repository_root)),
        "baseline_id": baseline_id,
        "scientific_role": config.get("scientific_role", ""),
        "result_stage": config.get("result_stage", ""),
        "run_mode": config.get("run_mode", ""),
        "model_kind": config.get("model_kind", ""),
        "pretrained_model_id": details.get("pretrained_model_id", config.get("pretrained_model_id", "")),
        "pretrained_feature_cache_path": config.get("pretrained_feature_cache_path", ""),
        "development_manifest": config.get("development_manifest", ""),
        "canonical_dataset_ids": _join(canonical_dataset_ids(config.get("canonical_paths"))),
        "canonical_paths": _join(config.get("canonical_paths")),
        "included_study_unit_ids": _join(details.get("included_study_unit_ids", config.get("included_study_unit_ids"))),
        "included_regions": _join(details.get("included_regions", config.get("included_regions"))),
        "train_record_count": record_counts.get("TRAIN", ""),
        "validation_record_count": record_counts.get("VALIDATION", ""),
        "test_record_count": record_counts.get("TEST", ""),
        "withheld_test_record_count": details.get("development_test_record_count_withheld", ""),
        "evaluation_record_count": details.get("evaluation_record_count", 0),
        "hidden_dim": config.get("hidden_dim", ""),
        "depth": config.get("depth", ""),
        "batch_size": config.get("batch_size", ""),
        "epochs": config.get("epochs", ""),
        "learning_rate": config.get("learning_rate", ""),
        "weight_decay": config.get("weight_decay", ""),
        "loss_kind": config.get("loss_kind", ""),
        "huber_delta": config.get("huber_delta", ""),
        "metadata_mode": config.get("metadata_mode", ""),
        "training_weighting_mode": config.get("training_weighting_mode", ""),
        "training_sampling_mode": config.get("training_sampling_mode", ""),
        "loss_aggregation_mode": config.get("loss_aggregation_mode", ""),
        "target_scaling_mode": config.get("target_scaling_mode", ""),
        "candidate_control": config.get("candidate_control", ""),
        "seed": config.get("seed", ""),
        "physical_gpu_index": config.get("physical_gpu_index", ""),
        "device": config.get("device", ""),
        "optimizer_name": config.get("optimizer_name", "AdamW"),
        "optimizer_fused": config.get("optimizer_fused", False),
        "training_precision": config.get("training_precision", "FP32"),
        "encoder_attention_backend": config.get("encoder_attention_backend", ""),
        "pretrained_position_encoding": config.get("pretrained_position_encoding", ""),
        "critic_position_features": config.get("critic_position_features", ""),
        "generation_action_space": config.get("generation_action_space", ""),
        "generator_position_features": config.get("generator_position_features", ""),
        "algorithmic_time_feature": config.get("algorithmic_time_feature", ""),
        "num_workers": config.get("num_workers", 0),
        "pin_memory": config.get("pin_memory", False),
        "persistent_workers": config.get(
            "persistent_workers", int(config.get("num_workers", 0)) > 0
        ),
        "prefetch_factor": config.get("prefetch_factor", ""),
        "non_blocking_transfer": config.get("non_blocking_transfer", False),
        "torch_compile": config.get("torch_compile", False),
        "trainable_parameter_count": trainable_count,
        "frozen_pretrained_parameter_count": pretrained_count,
        "total_effective_parameter_count": total_count,
        "optimizer_steps": details.get("optimizer_steps", ""),
        "selected_epoch": details.get("selected_epoch", ""),
        "validation_spearman": _metric(validation_metrics, "spearman"),
        "validation_task_macro_spearman": _metric(validation_metrics, "task_macro_spearman"),
        "validation_mae": _metric(validation_metrics, "mae"),
        "validation_task_macro_standardized_mae": _metric(validation_metrics, "task_macro_standardized_mae"),
        "test_spearman": _metric(test_metrics, "spearman"),
        "test_task_macro_spearman": _metric(test_metrics, "task_macro_spearman"),
        "test_mae": _metric(test_metrics, "mae"),
        "peak_vram_mb": details.get("peak_vram_mb", ""),
        "wall_time_seconds": details.get("wall_time_seconds", ""),
        "output_directory": str(output_dir),
        "notes": details.get("notes", config.get("notes", "")),
        "error_type": details.get("error_type", ""),
        "error": details.get("error", ""),
    }


def record_training_attempt(
    ledger_path: Path,
    run_record_path: Path,
    row: Mapping[str, Any],
) -> None:
    """Upsert one attempt while preserving fields known from earlier states."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ledger_path.with_suffix(ledger_path.suffix + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing_rows: list[dict[str, str]] = []
        existing_columns: list[str] = []
        if ledger_path.exists():
            with ledger_path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                existing_rows = list(reader)
                existing_columns = list(reader.fieldnames or [])
        match = next(
            (item for item in existing_rows if item.get("attempt_id") == row["attempt_id"]),
            None,
        )
        merged = dict(match or {})
        for key in TRAINING_ATTEMPT_COLUMNS:
            if key in {"started_at", "code_commit"} and match and match.get(key):
                # A long run may finish after the shared worktree advances.  Its
                # start commit and start time describe the code that actually ran.
                merged[key] = match[key]
                continue
            value = row.get(key, "")
            if value not in {None, ""}:
                merged[key] = value
            elif key not in merged:
                merged[key] = ""
        if match is None:
            existing_rows.append(merged)
        else:
            existing_rows[existing_rows.index(match)] = merged
        output_columns = list(TRAINING_ATTEMPT_COLUMNS)
        output_columns.extend(
            column
            for column in existing_columns
            if column and column not in output_columns
        )
        temporary = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_columns)
            writer.writeheader()
            writer.writerows(
                {column: item.get(column, "") for column in output_columns}
                for item in existing_rows
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, ledger_path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    run_record_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_record = run_record_path.with_suffix(run_record_path.suffix + ".tmp")
    temporary_record.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_record, run_record_path)
