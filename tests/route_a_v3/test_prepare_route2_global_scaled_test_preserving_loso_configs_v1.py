import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PATH = ROOT / "scripts/route_a_v3/prepare_route2_global_scaled_test_preserving_loso_configs_v1.py"
SPEC = importlib.util.spec_from_file_location("prepare_global_scaled_loso", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _base():
    return {
        "baseline_id": MODULE.BASELINE_ID,
        "model_kind": MODULE.MODEL_KIND,
        "result_stage": "HPO_VALIDATION_ONLY",
        "run_mode": "FIXED_GROUPED_SPLIT",
        "candidate_control": "NONE",
        "development_test_outcomes_accessed": False,
        "evaluation_outcomes_accessed": False,
    }


def test_builds_matched_21_fold_baseline_configs():
    configs = MODULE.build_configs(
        _base(),
        {"supports_single_frozen_development_test": True},
        run_root=Path("/runs/global"),
    )
    assert len(configs) == 21
    assert all(row["run_mode"] == "LOSO_DEVELOPMENT_TRAIN_VALIDATION_ONLY" for row in configs)
    assert all(row["development_test_outcomes_accessed"] is False for row in configs)
    assert all(row["evaluation_outcomes_accessed"] is False for row in configs)
    assert all(row["checkpoint_selection"] == "FINAL_EPOCH" for row in configs)


def test_rejects_baseline_substitution():
    base = _base()
    base["baseline_id"] = "other"
    with pytest.raises(MODULE.GlobalScaledLosoConfigError):
        MODULE.build_configs(
            base,
            {"supports_single_frozen_development_test": True},
            run_root=Path("/runs/global"),
        )
