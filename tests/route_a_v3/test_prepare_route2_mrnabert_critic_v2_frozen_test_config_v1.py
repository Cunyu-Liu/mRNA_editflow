from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "scripts/route_a_v3/prepare_route2_mrnabert_critic_v2_frozen_test_config_v1.py"
)
CONTROL_PROTOCOL = (
    ROOT / "configs/route_a_v3_route2_mrnabert_critic_v2_protocol_v1.json"
)
CONFIRMATION_PROTOCOL = (
    ROOT
    / "configs/route_a_v3_route2_mrnabert_critic_v2_three_seed_protocol_v1.json"
)
FROZEN_TEST_PROTOCOL = (
    ROOT
    / "configs/route_a_v3_route2_mrnabert_critic_v2_frozen_test_protocol_v1.json"
)


def _load():
    spec = importlib.util.spec_from_file_location("critic_v2_test_prepare_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _protocols() -> tuple[dict, dict, dict]:
    return tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (CONTROL_PROTOCOL, CONFIRMATION_PROTOCOL, FROZEN_TEST_PROTOCOL)
    )


def _baseline(protocol: dict) -> dict:
    frozen = protocol["strongest_same_information_baseline"]
    return {
        "baseline_id": frozen["baseline_id"],
        "task_macro_spearman": frozen["task_macro_spearman"],
        "task_macro_standardized_mae": frozen["task_macro_standardized_mae"],
    }


def _control_adjudication(control_protocol: dict) -> dict:
    return {
        "schema_version": "route_a_v3_route2_mrnabert_critic_v2_control_adjudication.v1",
        "status": "CRITIC_V2_CONTROLS_SUPPORT_THREE_FROZEN_SEEDS",
        "checks": {"full_beats_frozen_controls": True},
        "strongest_same_information_baseline": _baseline(control_protocol),
        "frozen_confirmation_seeds": [20260822, 20260823, 20260824],
        "supports_three_frozen_seeds": True,
        "development_test_opened": False,
        "evaluation_opened": False,
    }


def _confirmation_adjudication(confirmation_protocol: dict) -> dict:
    checks = {
        "control_adjudication_supports_three_frozen_seeds": True,
        "all_seed_metrics_finite": True,
        "all_seed_prediction_spreads_positive": True,
        "all_seed_task_macros_replay": True,
        "all_seed_spread_ratios_replay": True,
        "all_three_seed_margins_over_strongest_baseline_positive": True,
    }
    return {
        "schema_version": "route_a_v3_route2_mrnabert_critic_v2_three_seed_adjudication.v1",
        "status": "CRITIC_V2_THREE_SEEDS_SUPPORT_ONE_FROZEN_DEVELOPMENT_TEST",
        "checks": checks,
        "strongest_same_information_baseline": _baseline(confirmation_protocol),
        "seed_results": [
            {
                "seed": seed,
                "margin_over_strongest_same_information_baseline": 0.01,
                "nonfinite_metric_detected": False,
                "mean_collapse_detected": False,
            }
            for seed in (20260822, 20260823, 20260824)
        ],
        "supports_single_frozen_development_test": True,
        "development_test_opened": False,
        "evaluation_opened": False,
        "guided_generation_authorized": False,
    }


def _selected_confirmation(confirmation_protocol: dict) -> dict:
    seed = 20260823
    config = dict(confirmation_protocol["frozen_training_policy"])
    config.update(
        {
            "scientific_role": "CRITIC_V2_THREE_SEED_FROZEN_DEVELOPMENT_VALIDATION_CONFIRMATION",
            "result_stage": "FROZEN_DEVELOPMENT_VALIDATION",
            "run_mode": "FIXED_GROUPED_SPLIT",
            "baseline_id": f"mrnabert_critic_v2_full_confirmation_seed{seed}",
            "seed": seed,
            "candidate_control": "NONE",
            "development_test_outcomes_accessed": False,
            "evaluation_outcomes_accessed": False,
            "output_directory": str(
                Path(confirmation_protocol["run_root"]) / f"seed{seed}"
            ),
        }
    )
    return config


def _valid_inputs() -> tuple[dict, dict, dict, dict, dict, dict]:
    control, confirmation, frozen_test = _protocols()
    return (
        _selected_confirmation(confirmation),
        control,
        confirmation,
        frozen_test,
        _control_adjudication(control),
        _confirmation_adjudication(confirmation),
    )


def test_builds_exact_single_policy_matched_frozen_test_config() -> None:
    module = _load()
    inputs = _valid_inputs()
    config = module.build_config(*inputs, gpu=4)

    assert config["scientific_role"] == "CRITIC_V2_SINGLE_FROZEN_DEVELOPMENT_TEST"
    assert config["result_stage"] == "FROZEN_DEVELOPMENT_TEST"
    assert config["seed"] == 20260823
    assert config["physical_gpu_index"] == 4
    assert config["validation_checkpoint_selection_before_test"] == "BEST_VALIDATION"
    assert config["checkpoint_selection"] == "FINAL_EPOCH"
    assert config["epochs"] == 100
    assert config["epoch_count_source"] == "FROZEN_100_EPOCH_POLICY_BEFORE_TEST"
    assert config["development_validation_folded_into_training"] is True
    assert config["checkpoint_metric"] == "TASK_MACRO_SPEARMAN_THEN_STANDARDIZED_MAE"
    assert config["training_sampling_mode"] == "TASK_STUDY_SOURCE_GROUP_BALANCED_FIXED_DRAWS"
    assert config["loss_aggregation_mode"] == "TASK_MACRO_MEAN"
    assert config["development_test_outcomes_accessed"] is True
    assert config["evaluation_outcomes_accessed"] is False
    assert config["test_used_for_checkpoint_selection"] is False
    assert config["test_used_for_model_or_policy_selection"] is False
    assert config["output_directory"] == inputs[3]["run_directory"]


@pytest.mark.parametrize("gate", ["control", "confirmation"])
def test_rejects_either_failed_gate(gate: str) -> None:
    module = _load()
    selected, control, confirmation, frozen_test, control_adj, confirmation_adj = (
        _valid_inputs()
    )
    if gate == "control":
        control_adj["supports_three_frozen_seeds"] = False
        match = "control gate failed"
    else:
        confirmation_adj["supports_single_frozen_development_test"] = False
        match = "TEST gate failed"
    with pytest.raises(module.CriticV2FrozenTestPreparationError, match=match):
        module.build_config(
            selected,
            control,
            confirmation,
            frozen_test,
            control_adj,
            confirmation_adj,
            gpu=2,
        )


def test_rejects_nonpositive_confirmation_margin() -> None:
    module = _load()
    inputs = list(_valid_inputs())
    inputs[5]["seed_results"][1][
        "margin_over_strongest_same_information_baseline"
    ] = 0.0
    with pytest.raises(
        module.CriticV2FrozenTestPreparationError,
        match="does not beat the frozen baseline",
    ):
        module.build_config(*inputs, gpu=2)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("seed", "selected confirmation seed differs"),
        ("policy", "frozen policy differs: checkpoint_selection"),
        ("protected", "Development TEST entered frozen-TEST protocol"),
    ],
)
def test_rejects_seed_policy_or_protected_outcome_drift(
    mutation: str, match: str
) -> None:
    module = _load()
    inputs = list(_valid_inputs())
    if mutation == "seed":
        inputs[0]["seed"] = 20260822
    elif mutation == "policy":
        inputs[0]["checkpoint_selection"] = "FINAL_EPOCH"
    else:
        inputs[3]["development_test_outcomes_accessed"] = True
    with pytest.raises(module.CriticV2FrozenTestPreparationError, match=match):
        module.build_config(*inputs, gpu=2)


def test_rejects_gpu_outside_zero_to_five() -> None:
    module = _load()
    with pytest.raises(module.CriticV2FrozenTestPreparationError, match="GPU0-5"):
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
    with pytest.raises(module.CriticV2FrozenTestPreparationError, match=match):
        module.write_config_once(
            {"result_stage": "FROZEN_DEVELOPMENT_TEST"},
            config_path,
            run_directory,
        )


def test_write_config_once_writes_without_creating_run(tmp_path: Path) -> None:
    module = _load()
    config_path = tmp_path / "runtime" / "seed20260823.json"
    run_directory = tmp_path / "run" / "seed20260823"
    config = {"result_stage": "FROZEN_DEVELOPMENT_TEST", "seed": 20260823}
    module.write_config_once(deepcopy(config), config_path, run_directory)
    assert json.loads(config_path.read_text(encoding="utf-8")) == config
    assert not run_directory.exists()
