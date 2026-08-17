from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/analyze_route2_mrnabert_huber_throughput_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("huber_throughput_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile(profile_id: str, records_per_second: float, workers: int = 0) -> dict:
    return {
        "profile_id": profile_id,
        "status": "PASS",
        "records_per_second": records_per_second,
        "num_workers": workers,
    }


def inputs() -> tuple[dict, dict, dict]:
    huber = {
        "status": "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
        "loss_kind": "huber",
        "result_stage": "HPO_VALIDATION_ONLY",
        "evaluation_outcomes_read": 0,
        "test_metrics": None,
        "final_training_epoch": 100,
        "wall_time_seconds": 40_000.0,
        "optimizer_steps": 560_000,
        "record_counts": {"TRAIN": 89_580, "VALIDATION": 18_293},
    }
    gpu = {
        "evaluation_pool_records_read": 0,
        "fp32_bf16_precision_comparison": {"precision_tolerance_pass": True},
        "profiles": [
            profile("B16_FP32_ADAMW", 140.0),
            profile("B16_BF16_FUSED_ADAMW", 240.0),
            profile("B32_BF16_FUSED_ADAMW", 340.0),
            profile("B64_BF16_FUSED_ADAMW", 205.0),
        ],
    }
    loader = {
        "evaluation_pool_records_read": 0,
        "profiles": [
            profile("B32_BF16_FUSED_WORKERS0", 340.0, 0),
            profile("B32_BF16_FUSED_WORKERS4", 330.0, 4),
            profile("B32_BF16_FUSED_WORKERS8", 310.0, 8),
        ],
    }
    return huber, gpu, loader


def test_recommends_measured_profile_without_changing_current_loss_cohort() -> None:
    module = load_module()
    result = module.analyze(*inputs())
    assert result["recommended_profile_for_a_new_fully_matched_cohort"] == {
        "batch_size": 32,
        "training_precision": "BF16",
        "optimizer_fused": True,
        "num_workers": 0,
        "pin_memory": True,
        "non_blocking_transfer": True,
    }
    assert "BATCH16_UNDERUTILIZED_GPU_FOR_NEW_MATCHED_RUNS" in result[
        "engineering_bottleneck_findings"
    ]
    assert "HOST_DATALOADER_WORKERS_NOT_A_MATERIAL_BOTTLENECK" in result[
        "engineering_bottleneck_findings"
    ]
    assert result["current_three_loss_comparison_configuration_changed"] is False
    assert result["development_test_opened"] is False
    assert result["evaluation_opened"] is False


def test_rejects_incomplete_training_or_nonmatched_worker_profiles() -> None:
    module = load_module()
    huber, gpu, loader = inputs()
    huber["status"] = "RUNNING"
    with pytest.raises(module.ThroughputAnalysisError, match="incomplete"):
        module.analyze(huber, gpu, loader)

    huber, gpu, loader = inputs()
    loader["profiles"].pop()
    with pytest.raises(module.ThroughputAnalysisError, match="workers 0/4/8"):
        module.analyze(huber, gpu, loader)
