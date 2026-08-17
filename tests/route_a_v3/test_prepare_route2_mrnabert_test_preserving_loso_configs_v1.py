import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts/route_a_v3/prepare_route2_mrnabert_test_preserving_loso_configs_v1.py"
SPEC = importlib.util.spec_from_file_location("prepare_mrnabert_loso", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _selected():
    return {
        "model_kind": MODULE.PRIMARY_KIND,
        "result_stage": "HPO_VALIDATION_ONLY",
        "run_mode": "FIXED_GROUPED_SPLIT",
        "candidate_control": "NONE",
        "development_test_outcomes_accessed": False,
        "evaluation_outcomes_accessed": False,
        "loss_kind": "huber",
        "output_directory": "/old",
    }


def _adjudication():
    return {
        "status": "THREE_FINAL_SEEDS_SUPPORT_ONE_FROZEN_DEVELOPMENT_TEST",
        "supports_single_frozen_development_test": True,
        "loss_kind": "huber",
    }


def test_builds_seven_studies_by_three_seeds_without_test():
    configs = MODULE.build_configs(
        _selected(), _adjudication(), run_root=Path("/runs/loso")
    )
    assert len(configs) == 21
    assert {row["loso_holdout_study_unit_id"] for row in configs} == set(MODULE.HOLDOUT_STUDIES)
    assert {row["seed"] for row in configs} == {20260822, 20260823, 20260824}
    assert {row["physical_gpu_index"] for row in configs} == {0, 3, 5}
    assert all(row["run_mode"] == "LOSO_DEVELOPMENT_TRAIN_VALIDATION_ONLY" for row in configs)
    assert all(row["checkpoint_selection"] == "FINAL_EPOCH" for row in configs)
    assert all(row["development_test_outcomes_accessed"] is False for row in configs)
    assert all(row["evaluation_outcomes_accessed"] is False for row in configs)


def test_rejects_nonpassing_seed_gate():
    adjudication = _adjudication()
    adjudication["supports_single_frozen_development_test"] = False
    with pytest.raises(MODULE.TestPreservingLosoConfigError):
        MODULE.build_configs(_selected(), adjudication, run_root=Path("/runs/loso"))
