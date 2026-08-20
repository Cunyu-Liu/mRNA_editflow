from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "scripts/route_a_v3/prepare_route2_mrnabert_critic_v2_test_preserving_loso_configs_v1.py"
)
REFIT_PROTOCOL = (
    ROOT
    / "configs/route_a_v3_route2_mrnabert_critic_v2_all_development_refit_protocol_v1.json"
)
LOSO_PROTOCOL = (
    ROOT
    / "configs/route_a_v3_route2_mrnabert_critic_v2_test_preserving_loso_protocol_v1.json"
)


def _load():
    spec = importlib.util.spec_from_file_location("critic_v2_loso_prepare_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _protocols() -> tuple[dict, dict]:
    return (
        json.loads(REFIT_PROTOCOL.read_text(encoding="utf-8")),
        json.loads(LOSO_PROTOCOL.read_text(encoding="utf-8")),
    )


def _refit_config(refit_protocol: dict) -> dict:
    seed = 20260823
    config = {
        key: value
        for key, value in refit_protocol["frozen_model_training_policy"].items()
        if key != "checkpoint_selection_before_test"
    }
    config.update(
        {
            "scientific_role": "CRITIC_V2_FINAL_ALL_DEVELOPMENT_REFIT",
            "result_stage": "FINAL_ALL_DEVELOPMENT_REFIT",
            "run_mode": "FIXED_GROUPED_SPLIT",
            "baseline_id": f"mrnabert_critic_v2_all126165_refit_seed{seed}",
            "seed": seed,
            "candidate_control": "NONE",
            "checkpoint_selection": "FINAL_EPOCH",
            "development_record_scope": "ALL_126165",
            "train_validation_test_folded_into_training": True,
            "refit_model_selection_performed": False,
            "test_metrics_used_for_refit_selection": False,
            "development_test_outcomes_accessed": True,
            "evaluation_outcomes_accessed": False,
            "all_development_refit_protocol_schema_version": refit_protocol[
                "schema_version"
            ],
            "output_directory": refit_protocol["run_directory"],
        }
    )
    return config


def _refit_summary(refit_config: dict) -> dict:
    direct_policy_keys = {
        "model_kind",
        "training_weighting_mode",
        "training_sampling_mode",
        "loss_aggregation_mode",
        "training_update_mode",
        "loss_kind",
        "huber_delta",
        "checkpoint_selection",
        "checkpoint_metric",
        "batch_size",
        "optimizer_name",
        "learning_rate",
        "weight_decay",
        "training_precision",
    }
    summary = {
        key: value for key, value in refit_config.items() if key in direct_policy_keys
    }
    summary.update(
        {
            "status": "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
            "result_stage": "FINAL_ALL_DEVELOPMENT_REFIT",
            "baseline_id": refit_config["baseline_id"],
            "seed": refit_config["seed"],
            "candidate_control": "NONE",
            "target_scaler": {"mode": "TRAIN_TASK_ROBUST"},
            "selected_epoch": 100,
            "final_training_epoch": 100,
            "development_validation_folded_into_training": True,
            "development_test_record_count_withheld": 0,
            "development_test_outcomes_evaluated": False,
            "test_metrics": None,
            "record_counts": {"TRAIN": 126165},
            "evaluation_outcomes_read": 0,
            "cuda_training_tensors_verified": True,
            "cpu_fallback_used": False,
            "parameter_changed": True,
            "optimizer_steps": 788600,
            "physical_gpu_index": 4,
        }
    )
    return summary


def _valid_inputs() -> tuple[dict, dict, dict, dict]:
    refit_protocol, loso_protocol = _protocols()
    refit_config = _refit_config(refit_protocol)
    return refit_config, _refit_summary(refit_config), refit_protocol, loso_protocol


def test_builds_exact_seven_study_three_seed_six_gpu_cohort() -> None:
    module = _load()
    configs = module.build_configs(*_valid_inputs())

    assert len(configs) == 21
    assert [
        (
            row["loso_holdout_study_unit_id"],
            row["seed"],
            row["physical_gpu_index"],
        )
        for row in configs
    ] == list(module.loso_assignments())
    assert {row["physical_gpu_index"] for row in configs} == set(range(6))
    assert all(
        row["result_stage"]
        == "LOSO_DEVELOPMENT_VALIDATION_ONLY_FROZEN_HYPERPARAMETERS"
        for row in configs
    )
    assert all(
        row["run_mode"] == "LOSO_DEVELOPMENT_TRAIN_VALIDATION_ONLY"
        for row in configs
    )
    assert all(row["checkpoint_selection"] == "FINAL_EPOCH" for row in configs)
    assert all(row["epochs"] == 100 for row in configs)
    assert all(row["development_test_outcomes_accessed"] is False for row in configs)
    assert all(row["test_metrics_used_for_loso_selection"] is False for row in configs)
    assert all(row["evaluation_outcomes_accessed"] is False for row in configs)
    assert all(
        row["development_record_scope"]
        == "TRAIN_VALIDATION_ONLY_TEST_WITHHELD"
        for row in configs
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("status", "RUNNING", "incomplete"),
        ("result_stage", "FROZEN_DEVELOPMENT_TEST", "not the all-Development refit"),
        ("record_counts", {"TRAIN": 126164}, "record count differs"),
        ("evaluation_outcomes_read", 1, "Evaluation entered"),
        ("selected_epoch", 99, "frozen final epoch"),
        ("parameter_changed", False, "no learned update"),
    ],
)
def test_rejects_nonterminal_contaminated_or_drifted_refit_summary(
    field: str, value: object, match: str
) -> None:
    module = _load()
    inputs = list(_valid_inputs())
    inputs[1][field] = value
    with pytest.raises(
        module.CriticV2TestPreservingLosoPreparationError, match=match
    ):
        module.build_configs(*inputs)


def test_rejects_refit_config_policy_drift() -> None:
    module = _load()
    inputs = list(_valid_inputs())
    inputs[0]["loss_aggregation_mode"] = "GLOBAL_MEAN"
    with pytest.raises(
        module.CriticV2TestPreservingLosoPreparationError,
        match="refit config frozen policy differs: loss_aggregation_mode",
    ):
        module.build_configs(*inputs)


def test_rejects_protocol_cohort_drift() -> None:
    module = _load()
    inputs = list(_valid_inputs())
    inputs[3]["required_seeds"] = [20260822, 20260823, 20260826]
    with pytest.raises(
        module.CriticV2TestPreservingLosoPreparationError,
        match="LOSO seed cohort differs",
    ):
        module.build_configs(*inputs)


@pytest.mark.parametrize("existing_target", ["config_root", "run"])
def test_write_configs_once_refuses_existing_targets(
    tmp_path: Path, existing_target: str
) -> None:
    module = _load()
    config_root = tmp_path / "runtime"
    run_directory = tmp_path / "runs" / "GSE200304" / "seed20260822_gpu0"
    configs = [
        {
            "baseline_id": "mrnabert_critic_v2_loso_gse200304_seed20260822",
            "output_directory": str(run_directory),
        }
    ]
    if existing_target == "config_root":
        config_root.mkdir()
        match = "config root already exists"
    else:
        run_directory.mkdir(parents=True)
        match = "run directory already exists"
    with pytest.raises(
        module.CriticV2TestPreservingLosoPreparationError, match=match
    ):
        module.write_configs_once(configs, config_root)


def test_write_configs_once_writes_all_without_creating_runs(tmp_path: Path) -> None:
    module = _load()
    configs = module.build_configs(*_valid_inputs())
    config_root = tmp_path / "runtime"
    rewritten = []
    for config in configs:
        row = dict(config)
        row["output_directory"] = str(
            tmp_path
            / "runs"
            / row["loso_holdout_study_unit_id"]
            / f"seed{row['seed']}_gpu{row['physical_gpu_index']}"
        )
        rewritten.append(row)
    paths = module.write_configs_once(rewritten, config_root)
    assert len(paths) == 21
    assert len(list(config_root.glob("*.json"))) == 21
    assert all(not Path(row["output_directory"]).exists() for row in rewritten)
