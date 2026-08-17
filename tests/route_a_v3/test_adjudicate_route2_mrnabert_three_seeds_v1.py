from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/adjudicate_route2_mrnabert_three_seeds_v1.py"
PROTOCOL = ROOT / "configs/route_a_v3_route2_mrnabert_three_seed_gate_v1.json"


def _load():
    spec = importlib.util.spec_from_file_location("mrnabert_three_seed_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary(seed: int, offset: float = 0.0) -> dict:
    values = [0.20 + offset, 0.18, 0.30, 0.12, 0.10, 0.08, 0.16, 0.14, 0.11]
    return {
        "status": "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
        "result_stage": "FROZEN_DEVELOPMENT_VALIDATION",
        "run_mode": "FIXED_GROUPED_SPLIT",
        "model_kind": "delta_pretrained_mrnabert_edit_centered_antisymmetric",
        "candidate_control": "NONE",
        "loss_kind": "huber",
        "seed": seed,
        "trainable_parameter_count": 9_342_914,
        "final_training_epoch": 100,
        "evaluation_outcomes_read": 0,
        "development_test_outcomes_evaluated": False,
        "test_metrics": None,
        "validation_metrics": {
            "task_macro_spearman": sum(values) / len(values),
            "task_metrics": {f"task-{index}": {"spearman": value} for index, value in enumerate(values)},
        },
    }


def test_three_positive_seeds_allow_exactly_one_test() -> None:
    module = _load()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    summaries = [_summary(seed) for seed in (20260822, 20260823, 20260824)]
    result = module.adjudicate(protocol, summaries)
    assert result["supports_single_frozen_development_test"] is True
    assert result["single_frozen_test_seed"] == 20260823
    assert result["development_test_opened"] is False
    assert result["evaluation_opened"] is False


def test_one_nonpositive_seed_stops_before_test() -> None:
    module = _load()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    summaries = [_summary(seed) for seed in (20260822, 20260823, 20260824)]
    failed = summaries[1]
    values = [-0.2] * 9
    failed["validation_metrics"]["task_metrics"] = {
        f"task-{index}": {"spearman": value} for index, value in enumerate(values)
    }
    failed["validation_metrics"]["task_macro_spearman"] = -0.2
    result = module.adjudicate(protocol, summaries)
    assert result["supports_single_frozen_development_test"] is False
