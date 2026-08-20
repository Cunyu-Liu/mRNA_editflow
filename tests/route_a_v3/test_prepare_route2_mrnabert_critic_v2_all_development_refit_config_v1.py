from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "scripts/route_a_v3/prepare_route2_mrnabert_critic_v2_all_development_refit_config_v1.py"
)
TEST_PROTOCOL = (
    ROOT
    / "configs/route_a_v3_route2_mrnabert_critic_v2_frozen_test_protocol_v1.json"
)
REFIT_PROTOCOL = (
    ROOT
    / "configs/route_a_v3_route2_mrnabert_critic_v2_all_development_refit_protocol_v1.json"
)


def _load():
    spec = importlib.util.spec_from_file_location("critic_v2_refit_prepare_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _protocols() -> tuple[dict, dict]:
    return (
        json.loads(TEST_PROTOCOL.read_text(encoding="utf-8")),
        json.loads(REFIT_PROTOCOL.read_text(encoding="utf-8")),
    )


def _test_config(test_protocol: dict) -> dict:
    seed = 20260823
    config = dict(test_protocol["frozen_training_policy"])
    config.update(
        {
            "scientific_role": "CRITIC_V2_SINGLE_FROZEN_DEVELOPMENT_TEST",
            "result_stage": "FROZEN_DEVELOPMENT_TEST",
            "run_mode": "FIXED_GROUPED_SPLIT",
            "baseline_id": f"mrnabert_critic_v2_single_frozen_test_seed{seed}",
            "seed": seed,
            "candidate_control": "NONE",
            "validation_checkpoint_selection_before_test": "BEST_VALIDATION",
            "checkpoint_selection": "FINAL_EPOCH",
            "epoch_count_source": "FROZEN_100_EPOCH_POLICY_BEFORE_TEST",
            "development_validation_folded_into_training": True,
            "development_test_outcomes_accessed": True,
            "evaluation_outcomes_accessed": False,
            "test_used_for_checkpoint_selection": False,
            "test_used_for_model_or_policy_selection": False,
            "frozen_test_protocol_schema_version": test_protocol["schema_version"],
            "output_directory": test_protocol["run_directory"],
        }
    )
    return config


def _test_summary(test_config: dict) -> dict:
    summary = {
        key: value
        for key, value in test_config.items()
        if key
        in {
            "model_kind",
            "training_weighting_mode",
            "training_sampling_mode",
            "loss_aggregation_mode",
            "training_update_mode",
            "loss_kind",
            "huber_delta",
            "checkpoint_metric",
            "batch_size",
            "optimizer_name",
            "learning_rate",
            "weight_decay",
            "training_precision",
        }
    }
    summary.update(
        {
            "status": "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
            "result_stage": "FROZEN_DEVELOPMENT_TEST",
            "baseline_id": test_config["baseline_id"],
            "seed": test_config["seed"],
            "candidate_control": "NONE",
            "target_scaler": {"mode": "TRAIN_TASK_ROBUST"},
            "checkpoint_selection": "FINAL_EPOCH",
            "selected_epoch": 100,
            "final_training_epoch": 100,
            "development_test_outcomes_evaluated": True,
            "development_test_record_count_withheld": 0,
            "development_validation_folded_into_training": True,
            "record_counts": {"TRAIN": 107873, "TEST": 18292},
            "test_metrics": {"task_macro_spearman": -0.9},
            "evaluation_outcomes_read": 0,
            "cuda_training_tensors_verified": True,
            "cpu_fallback_used": False,
            "parameter_changed": True,
            "optimizer_steps": 674200,
            "physical_gpu_index": 2,
        }
    )
    return summary


def _valid_inputs() -> tuple[dict, dict, dict, dict]:
    test_protocol, refit_protocol = _protocols()
    test_config = _test_config(test_protocol)
    return test_config, _test_summary(test_config), test_protocol, refit_protocol


def test_builds_fixed_all_development_refit_without_thresholding_test() -> None:
    module = _load()
    inputs = _valid_inputs()
    assert inputs[1]["test_metrics"]["task_macro_spearman"] < 0.0
    config = module.build_config(*inputs, gpu=5)

    assert config["scientific_role"] == "CRITIC_V2_FINAL_ALL_DEVELOPMENT_REFIT"
    assert config["result_stage"] == "FINAL_ALL_DEVELOPMENT_REFIT"
    assert config["development_record_scope"] == "ALL_126165"
    assert config["seed"] == 20260823
    assert config["physical_gpu_index"] == 5
    assert config["epochs"] == 100
    assert config["checkpoint_selection"] == "FINAL_EPOCH"
    assert config["train_validation_test_folded_into_training"] is True
    assert config["refit_model_selection_performed"] is False
    assert config["test_metrics_used_for_refit_selection"] is False
    assert config["development_test_outcomes_accessed"] is True
    assert config["evaluation_outcomes_accessed"] is False
    assert config["output_directory"] == inputs[3]["run_directory"]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("status", "RUNNING", "incomplete"),
        ("test_metrics", None, "metrics are missing"),
        ("evaluation_outcomes_read", 1, "Evaluation entered"),
        ("seed", 20260822, "summary seed differs"),
        ("baseline_id", "wrong", "summary identity differs"),
        ("selected_epoch", 99, "fixed final epoch"),
        ("parameter_changed", False, "no learned update"),
    ],
)
def test_rejects_incomplete_contaminated_or_drifted_test_summary(
    field: str, value: object, match: str
) -> None:
    module = _load()
    inputs = list(_valid_inputs())
    inputs[1][field] = value
    with pytest.raises(
        module.CriticV2AllDevelopmentRefitPreparationError, match=match
    ):
        module.build_config(*inputs, gpu=2)


def test_rejects_test_config_policy_drift() -> None:
    module = _load()
    inputs = list(_valid_inputs())
    inputs[0]["training_sampling_mode"] = "LENGTH_BUCKET"
    with pytest.raises(
        module.CriticV2AllDevelopmentRefitPreparationError,
        match="TEST config frozen policy differs: training_sampling_mode",
    ):
        module.build_config(*inputs, gpu=2)


def test_rejects_gpu_outside_zero_to_five() -> None:
    module = _load()
    with pytest.raises(
        module.CriticV2AllDevelopmentRefitPreparationError, match="GPU0-5"
    ):
        module.build_config(*_valid_inputs(), gpu=6)


@pytest.mark.parametrize("existing_target", ["config", "run"])
def test_write_config_once_refuses_existing_targets(
    tmp_path: Path, existing_target: str
) -> None:
    module = _load()
    config_path = tmp_path / "runtime" / "seed20260823.json"
    run_directory = tmp_path / "run" / "seed20260823"
    if existing_target == "config":
        config_path.parent.mkdir(parents=True)
        config_path.write_text("{}\n", encoding="utf-8")
        match = "runtime config already exists"
    else:
        run_directory.mkdir(parents=True)
        match = "run directory already exists"
    with pytest.raises(
        module.CriticV2AllDevelopmentRefitPreparationError, match=match
    ):
        module.write_config_once(
            {"result_stage": "FINAL_ALL_DEVELOPMENT_REFIT"},
            config_path,
            run_directory,
        )


def test_write_config_once_writes_without_creating_run(tmp_path: Path) -> None:
    module = _load()
    config_path = tmp_path / "runtime" / "seed20260823.json"
    run_directory = tmp_path / "run" / "seed20260823"
    config = {"result_stage": "FINAL_ALL_DEVELOPMENT_REFIT", "seed": 20260823}
    module.write_config_once(config, config_path, run_directory)
    assert json.loads(config_path.read_text(encoding="utf-8")) == config
    assert not run_directory.exists()
