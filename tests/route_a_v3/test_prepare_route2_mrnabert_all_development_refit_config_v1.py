import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts/route_a_v3/prepare_route2_mrnabert_all_development_refit_config_v1.py"
SPEC = importlib.util.spec_from_file_location("prepare_final_refit", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _config():
    return {
        "result_stage": "FROZEN_DEVELOPMENT_TEST",
        "model_kind": MODULE.PRIMARY_KIND,
        "loss_kind": "huber",
        "seed": 20260823,
        "candidate_control": "NONE",
        "evaluation_outcomes_accessed": False,
        "output_directory": "/old/test",
    }


def _summary():
    return {
        "status": "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
        "result_stage": "FROZEN_DEVELOPMENT_TEST",
        "development_test_outcomes_evaluated": True,
        "test_metrics": {"task_macro_spearman": 0.2},
        "evaluation_outcomes_read": 0,
        "model_kind": MODULE.PRIMARY_KIND,
        "loss_kind": "huber",
        "seed": 20260823,
        "candidate_control": "NONE",
        "checkpoint_selection": "FINAL_EPOCH",
    }


def test_builds_all_development_refit_without_selection():
    result = MODULE.build_config(
        _config(), _summary(), gpu=0, output_directory=Path("/new/all126165")
    )
    assert result["result_stage"] == "FINAL_ALL_DEVELOPMENT_REFIT"
    assert result["development_test_outcomes_accessed"] is True
    assert result["evaluation_outcomes_accessed"] is False
    assert result["checkpoint_selection"] == "FINAL_EPOCH"
    assert result["output_directory"] == "/new/all126165"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("development_test_outcomes_evaluated", False),
        ("test_metrics", None),
        ("evaluation_outcomes_read", 1),
        ("checkpoint_selection", "BEST_VALIDATION"),
    ],
)
def test_rejects_ineligible_frozen_test(field, value):
    summary = _summary()
    summary[field] = value
    with pytest.raises(MODULE.FinalRefitConfigError):
        MODULE.build_config(
            _config(), summary, gpu=0, output_directory=Path("/new/all126165")
        )
