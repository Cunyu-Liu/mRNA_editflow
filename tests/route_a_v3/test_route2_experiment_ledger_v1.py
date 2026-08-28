from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "core/route2_experiment_ledger.py"
SYNC_PATH = ROOT / "scripts/route_a_v3/sync_route2_training_attempt_ledger_v1.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _config(tmp_path: Path) -> dict:
    return {
        "baseline_id": "attempt-a",
        "attempt_purpose": "LOSS_COMPARISON",
        "scientific_role": "DEVELOPMENT",
        "result_stage": "HPO_VALIDATION_ONLY",
        "run_mode": "FIXED_GROUPED_SPLIT",
        "model_kind": "critic",
        "canonical_paths": [
            "/mnt/project/canonical/GSE1/v1/canonical_records.jsonl",
            "/mnt/project/canonical/GSE2/v1/canonical_records.jsonl",
        ],
        "development_manifest": "/mnt/project/development_manifest.jsonl",
        "hidden_dim": 384,
        "depth": 10,
        "batch_size": 16,
        "epochs": 100,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "loss_kind": "huber",
        "huber_delta": 1.0,
        "metadata_mode": "FULL_CONTEXT",
        "training_weighting_mode": "STUDY_THEN_SOURCE_CONTEXT_ENDPOINT_GROUP",
        "training_sampling_mode": "COMPLETE_PASS_LENGTH_BUCKET",
        "loss_aggregation_mode": "RECORD_WEIGHTED",
        "target_scaling_mode": "TRAIN_TASK_ROBUST",
        "candidate_control": "NONE",
        "seed": 17,
        "parameter_initialization_seed": 17,
        "parameter_initialization_seed_applied_before_model_construction": True,
        "parameter_initialization_tensor_identity_scope": (
            "SHARED_V4_CONSTRUCTOR_WITHIN_IDENTICAL_ARCHITECTURE"
        ),
        "physical_gpu_index": 6,
        "device": "cuda:6",
        "cuda_available": True,
        "cuda_device_name": "NVIDIA A100-SXM4-80GB",
        "a100_device_verified": True,
        "bf16_supported": True,
        "cpu_fallback_used": False,
        "training_git_head": "a" * 40,
        "optimizer_name": "AdamW",
        "optimizer_fused": True,
        "training_precision": "FP32",
        "encoder_attention_backend": "OFFICIAL_PYTORCH_FALLBACK",
        "pretrained_position_encoding": "ALIBI_RELATIVE_BIAS",
        "critic_position_features": "NORMALIZED_ABSOLUTE_PLUS_EDIT_GATED",
        "generation_action_space": "SUB_PLUS_STOP",
        "generator_position_features": "NORMALIZED_ABSOLUTE_PLUS_EDIT_GATED",
        "algorithmic_time_feature": "CONSUMED_EDIT_BUDGET_FRACTION",
        "num_workers": 0,
        "pin_memory": False,
        "torch_compile": False,
        "expected_trainable_parameter_count": 9_000_000,
        "expected_frozen_pretrained_parameter_count": 113_000_000,
        "output_directory": str(tmp_path / "run"),
        "training_summary_path": str(tmp_path / "run" / "run_summary.json"),
        "checkpoint_path": str(tmp_path / "run" / "final_pass_8_checkpoint.pt"),
        "training_attempt_path": str(tmp_path / "run" / "training_attempt.json"),
    }


def test_training_attempt_upsert_preserves_start_and_adds_final_metrics(tmp_path: Path) -> None:
    ledger = _load(MODULE_PATH, "route2_experiment_ledger_test")
    config = _config(tmp_path)
    output_dir = Path(config["output_directory"])
    output_dir.mkdir()
    ledger_path = tmp_path / "attempts.csv"
    run_record = output_dir / "training_attempt.json"
    running = ledger.build_training_attempt_row(
        config,
        output_dir,
        "RUNNING",
        repository_root=ROOT,
        details={
            "started_at": "2026-08-17T10:00:00+08:00",
            "included_study_unit_ids": ["GSE1", "GSE2"],
            "included_regions": ["5UTR", "3UTR"],
            "record_counts": {"TRAIN": 90, "VALIDATION": 20},
            "development_test_record_count_withheld": 10,
        },
    )
    ledger.record_training_attempt(ledger_path, run_record, running)
    completed = ledger.build_training_attempt_row(
        config,
        output_dir,
        "COMPLETED",
        repository_root=ROOT,
        details={
            "optimizer_steps": 560,
            "selected_epoch": 88,
            "validation_metrics": {"spearman": 0.21, "mae": 0.7},
            "wall_time_seconds": 120.5,
            "peak_vram_mb": 4210.0,
        },
    )
    ledger.record_training_attempt(ledger_path, run_record, completed)

    with ledger_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["status"] == "COMPLETED"
    assert rows[0]["started_at"] == "2026-08-17T10:00:00+08:00"
    assert rows[0]["canonical_dataset_ids"] == "GSE1;GSE2"
    assert rows[0]["train_record_count"] == "90"
    assert rows[0]["optimizer_steps"] == "560"
    assert rows[0]["validation_spearman"] == "0.21"
    assert rows[0]["pretrained_position_encoding"] == "ALIBI_RELATIVE_BIAS"
    assert rows[0]["encoder_attention_backend"] == "OFFICIAL_PYTORCH_FALLBACK"
    assert rows[0]["optimizer_fused"] == "True"
    assert rows[0]["training_sampling_mode"] == "COMPLETE_PASS_LENGTH_BUCKET"
    assert rows[0]["loss_aggregation_mode"] == "RECORD_WEIGHTED"
    assert rows[0]["generation_action_space"] == "SUB_PLUS_STOP"
    assert rows[0]["algorithmic_time_feature"] == "CONSUMED_EDIT_BUDGET_FRACTION"
    assert rows[0]["parameter_initialization_seed"] == "17"
    assert (
        rows[0]["parameter_initialization_seed_applied_before_model_construction"]
        == "True"
    )
    assert rows[0]["cuda_device_name"] == "NVIDIA A100-SXM4-80GB"
    assert rows[0]["a100_device_verified"] == "True"
    assert rows[0]["bf16_supported"] == "True"
    assert rows[0]["cpu_fallback_used"] == "False"
    assert rows[0]["training_git_head"] == "a" * 40
    assert rows[0]["checkpoint_path"].endswith("final_pass_8_checkpoint.pt")
    assert json.loads(run_record.read_text())["status"] == "COMPLETED"


def test_long_run_preserves_start_commit_and_newer_ledger_columns(tmp_path: Path) -> None:
    ledger = _load(MODULE_PATH, "route2_experiment_ledger_version_skew_test")
    config = _config(tmp_path)
    output_dir = Path(config["output_directory"])
    output_dir.mkdir()
    ledger_path = tmp_path / "attempts.csv"
    run_record = output_dir / "training_attempt.json"

    running = ledger.build_training_attempt_row(
        config,
        output_dir,
        "RUNNING",
        repository_root=ROOT,
        details={
            "started_at": "2026-08-17T10:00:00+08:00",
            "code_commit": "commit-at-process-start",
        },
    )
    ledger.record_training_attempt(ledger_path, run_record, running)

    with ledger_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    columns.append("newer_training_field")
    rows[0]["newer_training_field"] = "preserve-me"
    with ledger_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    completed = ledger.build_training_attempt_row(
        config,
        output_dir,
        "COMPLETED",
        repository_root=ROOT,
        details={"code_commit": "worktree-advanced-after-start"},
    )
    ledger.record_training_attempt(ledger_path, run_record, completed)

    with ledger_path.open(newline="", encoding="utf-8-sig") as handle:
        final_rows = list(csv.DictReader(handle))
    assert final_rows[0]["code_commit"] == "commit-at-process-start"
    assert final_rows[0]["started_at"] == "2026-08-17T10:00:00+08:00"
    assert final_rows[0]["newer_training_field"] == "preserve-me"


def test_backfill_marks_existing_run_completed(tmp_path: Path) -> None:
    sync = _load(SYNC_PATH, "route2_experiment_ledger_sync_test")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = _config(tmp_path)
    (run_dir / "training_config.json").write_text(json.dumps(config))
    (run_dir / "training_summary.json").write_text(json.dumps({
        "record_counts": {"TRAIN": 90, "VALIDATION": 20},
        "optimizer_steps": 500,
        "selected_epoch": 75,
        "validation_metrics": {"task_macro_spearman": 0.3},
        "evaluation_record_count": 0,
    }))
    row = sync.sync_run(run_dir, tmp_path / "attempts.csv")
    assert row["status"] == "COMPLETED"
    assert row["optimizer_steps"] == 500
    assert row["validation_task_macro_spearman"] == 0.3


def test_backfill_reuses_existing_attempt_identity_when_config_lacks_it(
    tmp_path: Path,
) -> None:
    sync = _load(SYNC_PATH, "route2_experiment_ledger_sync_identity_test")
    run_dir = tmp_path / "development_v1"
    run_dir.mkdir()
    config = _config(tmp_path)
    config.pop("baseline_id")
    (run_dir / "training_config.json").write_text(json.dumps(config))
    (run_dir / "training_summary.json").write_text(
        json.dumps({"optimizer_steps": 10, "evaluation_record_count": 0})
    )
    (run_dir / "training_attempt.json").write_text(
        json.dumps(
            {
                "attempt_id": "base-flow-seed17::development_v1",
                "baseline_id": "base-flow-seed17",
                "status": "RUNNING",
            }
        )
    )

    row = sync.sync_run(run_dir, tmp_path / "attempts.csv")

    assert row["attempt_id"] == "base-flow-seed17::development_v1"
    assert row["baseline_id"] == "base-flow-seed17"
    with (tmp_path / "attempts.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["attempt_id"] == "base-flow-seed17::development_v1"


def test_training_attempt_uses_run_id_when_baseline_id_is_absent(tmp_path: Path) -> None:
    ledger = _load(MODULE_PATH, "route2_experiment_ledger_run_id_test")
    config = _config(tmp_path)
    del config["baseline_id"]
    config["run_id"] = "base-flow-g0"
    row = ledger.build_training_attempt_row(
        config,
        tmp_path / "run",
        "COMPLETED",
        repository_root=ROOT,
    )
    assert row["baseline_id"] == "base-flow-g0"
    assert row["attempt_id"] == "base-flow-g0::run"


def test_bulk_backfill_does_not_call_unfinished_history_running(tmp_path: Path) -> None:
    sync = _load(SYNC_PATH, "route2_experiment_ledger_incomplete_test")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = _config(tmp_path)
    manifest = tmp_path / "development_manifest.jsonl"
    manifest.write_text(json.dumps({
        "pool_assignment": "DEVELOPMENT",
        "study_unit_id": "GSE1",
        "split": "TRAIN",
    }) + "\n")
    config["development_manifest"] = str(manifest)
    (run_dir / "training_config.json").write_text(json.dumps(config))
    row = sync.sync_run(run_dir, tmp_path / "attempts.csv", active_run=False)
    assert row["status"] == "INCOMPLETE_NO_TERMINAL_RECORD"


def test_mrnabert_dataloader_benchmark_is_a_matched_worker_comparison() -> None:
    config = json.loads(
        (
            ROOT
            / "configs/route_a_v3_route2_mrnabert_dataloader_benchmark_gpu0_v1.json"
        ).read_text(encoding="utf-8")
    )
    profiles = config["profiles"]
    assert [profile["num_workers"] for profile in profiles] == [0, 4, 8]
    assert {profile["batch_size"] for profile in profiles} == {32}
    assert {profile["training_precision"] for profile in profiles} == {"BF16"}
    assert {profile["fused_adamw"] for profile in profiles} == {True}
    assert {profile["pin_memory"] for profile in profiles} == {True}
    assert {profile["non_blocking_transfer"] for profile in profiles} == {True}
    assert config["measured_steps"] >= 100
