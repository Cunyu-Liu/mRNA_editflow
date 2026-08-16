from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/assemble_route2_evaluation_strongest_prediction_predictions_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("assemble_route2_evaluation_strongest_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _selection() -> dict:
    return {
        "status": "DEVELOPMENT_VALIDATION_STRONGEST_BASELINES_SELECTED_BY_TASK",
        "evaluation_outcomes_accessed": False,
        "tasks": {"5UTR|KNOWN": {"strongest_baseline_id": "exact"}},
        "unseen_endpoint_fallbacks": {
            "policy": "EXACT_TASK_ELSE_COMPLETE_COVERAGE_REGION_ELSE_COMPLETE_COVERAGE_GLOBAL",
            "regions": {"3UTR": {"strongest_baseline_id": "region"}},
            "global": {"strongest_baseline_id": "global"},
        },
    }


def test_frozen_selection_resolves_exact_region_and_global_without_outcomes() -> None:
    module = _load()
    selection = _selection()
    assert module.baseline_for_task(selection, "5UTR|KNOWN") == ("exact", "EXACT_TASK")
    assert module.baseline_for_task(selection, "3UTR|UNSEEN") == ("region", "COMPLETE_COVERAGE_REGION")
    assert module.baseline_for_task(selection, "CDS|UNSEEN") == ("global", "COMPLETE_COVERAGE_GLOBAL")


def test_evaluation_loader_rejects_development_manifest(tmp_path: Path) -> None:
    module = _load()
    path = tmp_path / "manifest.jsonl"
    _jsonl(path, [{"canonical_record_id": "r", "pool_assignment": "DEVELOPMENT", "split": "TEST"}])
    with pytest.raises(module.EvaluationStrongestAssemblyError, match="Development row"):
        module.evaluation_manifest(path)


def test_prediction_summary_requires_cuda_zero_shot_and_no_metrics() -> None:
    module = _load()
    summary = {
        "status": "EVALUATION_ZERO_SHOT_PREDICTIONS_GENERATED",
        "baseline_id": "region",
        "evaluation_outcome_metrics_computed": False,
        "evaluation_outcomes_used_for_training_hpo_or_selection": 0,
        "cpu_fallback_used": False,
        "device": "cuda:6",
        "cuda_device_uuid": "GPU-real",
    }
    module.validate_prediction_summary(summary, "region")
    invalid = dict(summary, cpu_fallback_used=True)
    with pytest.raises(module.EvaluationStrongestAssemblyError, match="CPU fallback"):
        module.validate_prediction_summary(invalid, "region")
