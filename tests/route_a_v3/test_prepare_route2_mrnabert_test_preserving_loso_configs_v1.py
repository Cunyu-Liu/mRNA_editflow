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
    assert {row["physical_gpu_index"] for row in configs} == {0, 1, 2, 3, 4, 5}
    assert all(row["run_mode"] == "LOSO_DEVELOPMENT_TRAIN_VALIDATION_ONLY" for row in configs)
    assert all(row["checkpoint_selection"] == "FINAL_EPOCH" for row in configs)
    assert all(row["development_test_outcomes_accessed"] is False for row in configs)
    assert all(row["evaluation_outcomes_accessed"] is False for row in configs)


def test_rejects_nonpassing_seed_gate():
    adjudication = _adjudication()
    adjudication["supports_single_frozen_development_test"] = False
    with pytest.raises(MODULE.TestPreservingLosoConfigError):
        MODULE.build_configs(_selected(), adjudication, run_root=Path("/runs/loso"))


def test_model_and_baseline_loso_are_paired_across_all_six_gpus():
    baseline_path = (
        ROOT
        / "scripts/route_a_v3/prepare_route2_global_scaled_test_preserving_loso_configs_v1.py"
    )
    baseline_spec = importlib.util.spec_from_file_location(
        "prepare_matched_baseline_loso", baseline_path
    )
    assert baseline_spec is not None and baseline_spec.loader is not None
    baseline_module = importlib.util.module_from_spec(baseline_spec)
    baseline_spec.loader.exec_module(baseline_module)
    model_configs = MODULE.build_configs(
        _selected(), _adjudication(), run_root=Path("/runs/model")
    )
    baseline_configs = baseline_module.build_configs(
        {
            "baseline_id": baseline_module.BASELINE_ID,
            "model_kind": baseline_module.MODEL_KIND,
            "result_stage": "HPO_VALIDATION_ONLY",
            "run_mode": "FIXED_GROUPED_SPLIT",
            "candidate_control": "NONE",
            "development_test_outcomes_accessed": False,
            "evaluation_outcomes_accessed": False,
        },
        {"supports_single_frozen_development_test": True},
        run_root=Path("/runs/baseline"),
    )
    assignment = lambda row: (
        row["loso_holdout_study_unit_id"],
        row["seed"],
        row["physical_gpu_index"],
    )
    assert [assignment(row) for row in model_configs] == [
        assignment(row) for row in baseline_configs
    ]
    counts = {
        gpu: sum(row["physical_gpu_index"] == gpu for row in model_configs)
        for gpu in range(6)
    }
    assert sorted(counts.values()) == [3, 3, 3, 4, 4, 4]


def test_postselection_scheduler_runs_six_paired_gpu_workers():
    source = (
        ROOT / "scripts/route_a_v3/schedule_route2_mrnabert_postselection_controls_v1.sh"
    ).read_text(encoding="utf-8")
    assert "for gpu in 0 1 2 3 4 5" in source
    assert "run_loso_gpu_worker" in source
    assert "role=primary" in source
    assert "role=baseline" in source
    assert source.index("role=primary") < source.index("role=baseline")
    assert "all_paired_six_gpu_loso_runs_finished" in source
