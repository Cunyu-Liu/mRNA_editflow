#!/usr/bin/env python3
"""Prepare the fixed Critic V2 all-Development refit after its one TEST."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


FROZEN_TEST_PROTOCOL_SCHEMA = (
    "route_a_v3_route2_mrnabert_critic_v2_frozen_test_protocol.v1"
)
REFIT_PROTOCOL_SCHEMA = (
    "route_a_v3_route2_mrnabert_critic_v2_all_development_refit_protocol.v1"
)
PRIMARY_KIND = "delta_pretrained_mrnabert_edit_centered_antisymmetric"
SINGLE_FROZEN_TEST_SEED = 20260823


class CriticV2AllDevelopmentRefitPreparationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CriticV2AllDevelopmentRefitPreparationError(message)


def _refit_policy_from_test_policy(test_policy: Mapping[str, Any]) -> dict[str, Any]:
    policy = dict(test_policy)
    policy["checkpoint_selection_before_test"] = policy.pop("checkpoint_selection")
    return policy


def _validate_test_config(
    config: Mapping[str, Any],
    test_protocol: Mapping[str, Any],
    refit_protocol: Mapping[str, Any],
) -> None:
    seed = int(refit_protocol["single_frozen_test_seed"])
    _require(
        config.get("scientific_role")
        == "CRITIC_V2_SINGLE_FROZEN_DEVELOPMENT_TEST",
        "input config is not the Critic V2 single frozen TEST",
    )
    _require(
        config.get("result_stage") == "FROZEN_DEVELOPMENT_TEST",
        "input config is not the frozen Development TEST stage",
    )
    _require(
        config.get("run_mode") == "FIXED_GROUPED_SPLIT",
        "frozen Development TEST split differs",
    )
    _require(config.get("model_kind") == PRIMARY_KIND, "TEST model kind differs")
    _require(config.get("candidate_control") == "NONE", "TEST config is a control")
    _require(
        config.get("baseline_id")
        == f"mrnabert_critic_v2_single_frozen_test_seed{seed}",
        "frozen Development TEST identity differs",
    )
    _require(int(config.get("seed", -1)) == seed, "frozen Development TEST seed differs")
    _require(
        config.get("output_directory") == str(test_protocol["run_directory"]),
        "frozen Development TEST run path differs",
    )
    _require(
        config.get("frozen_test_protocol_schema_version")
        == FROZEN_TEST_PROTOCOL_SCHEMA,
        "frozen Development TEST protocol binding differs",
    )
    _require(
        config.get("development_test_outcomes_accessed") is True,
        "frozen Development TEST config does not open TEST",
    )
    _require(
        config.get("evaluation_outcomes_accessed") is False,
        "Evaluation entered the frozen Development TEST config",
    )
    _require(
        config.get("test_used_for_checkpoint_selection") is False
        and config.get("test_used_for_model_or_policy_selection") is False,
        "frozen Development TEST was marked as a selection source",
    )
    _require(
        config.get("validation_checkpoint_selection_before_test")
        == "BEST_VALIDATION",
        "pre-TEST Validation checkpoint policy differs",
    )
    _require(
        config.get("checkpoint_selection") == "FINAL_EPOCH",
        "executable TEST checkpoint policy differs",
    )
    _require(
        config.get("epoch_count_source") == "FROZEN_100_EPOCH_POLICY_BEFORE_TEST",
        "TEST epoch source was not prospectively frozen",
    )
    _require(
        config.get("development_validation_folded_into_training") is True,
        "TEST config did not fold Development Validation into training",
    )
    for key, expected in test_protocol["frozen_training_policy"].items():
        if key == "checkpoint_selection":
            continue
        _require(config.get(key) == expected, f"TEST config frozen policy differs: {key}")


def _validate_test_summary(
    summary: Mapping[str, Any],
    test_config: Mapping[str, Any],
    test_protocol: Mapping[str, Any],
    refit_protocol: Mapping[str, Any],
) -> None:
    seed = int(refit_protocol["single_frozen_test_seed"])
    _require(
        summary.get("status") == "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
        "frozen Development TEST run is incomplete",
    )
    _require(
        summary.get("result_stage") == "FROZEN_DEVELOPMENT_TEST",
        "summary is not the frozen Development TEST result",
    )
    _require(
        summary.get("baseline_id") == test_config.get("baseline_id"),
        "frozen Development TEST summary identity differs",
    )
    _require(summary.get("model_kind") == PRIMARY_KIND, "TEST summary model kind differs")
    _require(int(summary.get("seed", -1)) == seed, "TEST summary seed differs")
    _require(summary.get("candidate_control") == "NONE", "TEST summary is a control")
    _require(
        summary.get("development_test_outcomes_evaluated") is True,
        "Development TEST was not evaluated",
    )
    _require(
        isinstance(summary.get("test_metrics"), Mapping),
        "frozen Development TEST metrics are missing",
    )
    _require(
        summary.get("evaluation_outcomes_read") == 0,
        "Evaluation entered the frozen Development TEST run",
    )
    _require(
        summary.get("development_validation_folded_into_training") is True,
        "Development Validation was not folded into TEST training",
    )
    _require(
        summary.get("development_test_record_count_withheld") == 0,
        "frozen Development TEST remained withheld",
    )
    record_counts = summary.get("record_counts")
    expected_train = int(refit_protocol["development_train_record_count"]) + int(
        refit_protocol["development_validation_record_count"]
    )
    _require(
        isinstance(record_counts, Mapping)
        and record_counts.get("TRAIN") == expected_train
        and record_counts.get("TEST")
        == int(refit_protocol["development_test_record_count"]),
        "frozen Development TEST record counts differ",
    )
    _require(
        summary.get("checkpoint_selection") == "FINAL_EPOCH",
        "TEST summary checkpoint policy differs",
    )
    _require(
        int(summary.get("selected_epoch", -1))
        == int(summary.get("final_training_epoch", -2))
        == int(refit_protocol["frozen_model_training_policy"]["epochs"]),
        "TEST summary did not use the prospectively fixed final epoch",
    )
    _require(
        summary.get("cuda_training_tensors_verified") is True,
        "TEST CUDA training tensors were not verified",
    )
    _require(summary.get("cpu_fallback_used") is False, "TEST used CPU fallback")
    _require(summary.get("parameter_changed") is True, "TEST has no learned update")
    _require(int(summary.get("optimizer_steps", 0)) > 0, "TEST has no optimizer steps")
    _require(
        0 <= int(summary.get("physical_gpu_index", -1)) <= 5,
        "TEST physical GPU is outside GPU0-5",
    )

    for key, expected in test_protocol["frozen_training_policy"].items():
        if key == "checkpoint_selection":
            observed = summary.get("checkpoint_selection")
            expected = "FINAL_EPOCH"
        elif key == "target_scaling_mode":
            observed = summary.get("target_scaler", {}).get("mode")
        elif key == "epochs":
            observed = summary.get("final_training_epoch")
        else:
            observed = summary.get(key)
        _require(observed == expected, f"TEST summary frozen policy differs: {key}")


def build_config(
    frozen_test_config: Mapping[str, Any],
    frozen_test_summary: Mapping[str, Any],
    frozen_test_protocol: Mapping[str, Any],
    refit_protocol: Mapping[str, Any],
    *,
    gpu: int,
) -> dict[str, Any]:
    """Construct the refit config without branching on TEST metric values."""

    _require(
        frozen_test_protocol.get("schema_version") == FROZEN_TEST_PROTOCOL_SCHEMA,
        "unexpected Critic V2 frozen-TEST protocol",
    )
    _require(
        frozen_test_protocol.get("status")
        == "FROZEN_BEFORE_CRITIC_V2_THREE_SEED_OUTCOMES",
        "Critic V2 frozen-TEST protocol was not prospectively frozen",
    )
    _require(
        frozen_test_protocol.get("development_test_outcomes_accessed") is False,
        "Development TEST entered the protocol before freeze",
    )
    _require(
        frozen_test_protocol.get("evaluation_outcomes_accessed") is False,
        "Evaluation entered the frozen-TEST protocol",
    )
    _require(
        frozen_test_protocol.get("guided_generation_authorized") is False,
        "guided generation entered the frozen-TEST protocol",
    )
    _require(
        refit_protocol.get("schema_version") == REFIT_PROTOCOL_SCHEMA,
        "unexpected Critic V2 all-Development refit protocol",
    )
    _require(
        refit_protocol.get("status")
        == "FROZEN_BEFORE_CRITIC_V2_THREE_SEED_OUTCOMES",
        "Critic V2 refit protocol was not prospectively frozen",
    )
    _require(
        refit_protocol.get("development_test_outcomes_accessed_at_protocol_freeze")
        is False,
        "Development TEST entered before refit protocol freeze",
    )
    _require(
        refit_protocol.get("evaluation_outcomes_accessed") is False,
        "Evaluation entered the refit protocol",
    )
    _require(
        refit_protocol.get("guided_generation_authorized") is False,
        "guided generation entered the refit protocol",
    )
    _require(
        refit_protocol.get("frozen_test_protocol")
        == "configs/route_a_v3_route2_mrnabert_critic_v2_frozen_test_protocol_v1.json",
        "refit protocol binds a different frozen-TEST protocol",
    )
    _require(
        int(refit_protocol["single_frozen_test_seed"])
        == int(frozen_test_protocol["single_frozen_test_seed"])
        == SINGLE_FROZEN_TEST_SEED,
        "single frozen TEST seed differs across protocols",
    )
    _require(
        int(refit_protocol["development_test_record_count"])
        == int(frozen_test_protocol["development_test_record_count"]),
        "Development TEST count differs across protocols",
    )
    _require(
        int(refit_protocol["development_record_count"])
        == sum(
            int(refit_protocol[key])
            for key in (
                "development_train_record_count",
                "development_validation_record_count",
                "development_test_record_count",
            )
        )
        == 126165,
        "all-Development record counts do not close",
    )
    expected_refit_policy = _refit_policy_from_test_policy(
        frozen_test_protocol["frozen_training_policy"]
    )
    _require(
        refit_protocol.get("frozen_model_training_policy")
        == expected_refit_policy,
        "refit model training policy differs from the frozen TEST protocol",
    )
    expected_execution_policy = {
        "result_stage": "FINAL_ALL_DEVELOPMENT_REFIT",
        "run_mode": "FIXED_GROUPED_SPLIT",
        "development_scope": "ALL_126165",
        "train_validation_test_folded_into_training": True,
        "checkpoint_selection": "FINAL_EPOCH",
        "epoch_count_source": "FROZEN_100_EPOCH_POLICY_BEFORE_TEST",
        "refit_model_selection_performed": False,
        "test_metrics_used_for_refit_selection": False,
    }
    _require(
        refit_protocol.get("refit_execution_policy") == expected_execution_policy,
        "all-Development refit execution policy differs",
    )

    _validate_test_config(frozen_test_config, frozen_test_protocol, refit_protocol)
    _validate_test_summary(
        frozen_test_summary,
        frozen_test_config,
        frozen_test_protocol,
        refit_protocol,
    )
    _require(
        isinstance(gpu, int) and not isinstance(gpu, bool) and 0 <= gpu <= 5,
        "all-Development refit must use physical GPU0-5",
    )

    seed = int(refit_protocol["single_frozen_test_seed"])
    config = dict(frozen_test_config)
    config.update(
        {
            "scientific_role": "CRITIC_V2_FINAL_ALL_DEVELOPMENT_REFIT",
            "result_stage": "FINAL_ALL_DEVELOPMENT_REFIT",
            "run_mode": "FIXED_GROUPED_SPLIT",
            "baseline_id": f"mrnabert_critic_v2_all126165_refit_seed{seed}",
            "attempt_purpose": "MRNABERT_CRITIC_V2_FINAL_ALL_126165_DEVELOPMENT_REFIT",
            "seed": seed,
            "device": f"cuda:{gpu}",
            "physical_gpu_index": gpu,
            "candidate_control": "NONE",
            "checkpoint_selection": "FINAL_EPOCH",
            "epoch_count_source": "FROZEN_100_EPOCH_POLICY_BEFORE_TEST",
            "development_record_scope": "ALL_126165",
            "train_validation_test_folded_into_training": True,
            "refit_model_selection_performed": False,
            "test_metrics_used_for_refit_selection": False,
            "development_test_outcomes_accessed": True,
            "evaluation_outcomes_accessed": False,
            "all_development_refit_protocol_schema_version": REFIT_PROTOCOL_SCHEMA,
            "output_directory": str(refit_protocol["run_directory"]),
            "notes": (
                "Fixed Critic V2 refit on all 126,165 Development records after the "
                "single legal TEST. Structure, loss, seed, 100-epoch budget and final-"
                "epoch rule were frozen before TEST outcomes; TEST metric values are "
                "not used for selection; Evaluation and guidance remain closed."
            ),
        }
    )
    return config


def write_config_once(
    config: Mapping[str, Any], output_config: Path, run_directory: Path
) -> None:
    _require(
        not output_config.exists(),
        f"all-Development refit runtime config already exists: {output_config}",
    )
    _require(
        not run_directory.exists(),
        f"all-Development refit run directory already exists: {run_directory}",
    )
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(
        json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-test-config", type=Path, required=True)
    parser.add_argument("--frozen-test-summary", type=Path, required=True)
    parser.add_argument("--frozen-test-protocol", type=Path, required=True)
    parser.add_argument("--refit-protocol", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()

    refit_protocol = json.loads(args.refit_protocol.read_text(encoding="utf-8"))
    config = build_config(
        json.loads(args.frozen_test_config.read_text(encoding="utf-8")),
        json.loads(args.frozen_test_summary.read_text(encoding="utf-8")),
        json.loads(args.frozen_test_protocol.read_text(encoding="utf-8")),
        refit_protocol,
        gpu=args.gpu,
    )
    output_config = Path(str(refit_protocol["runtime_config"]))
    run_directory = Path(str(refit_protocol["run_directory"]))
    _require(
        config["output_directory"] == str(run_directory),
        "prepared refit run path differs from the frozen protocol",
    )
    write_config_once(config, output_config, run_directory)
    print(
        json.dumps(
            {
                "status": "CRITIC_V2_ALL_DEVELOPMENT_REFIT_CONFIG_PREPARED_NOT_EXECUTED",
                "config": str(output_config),
                "run_directory": str(run_directory),
                "seed": config["seed"],
                "development_record_scope": "ALL_126165",
                "test_metric_values_used_for_selection": False,
                "refit_executed": False,
                "evaluation_opened": False,
                "guided_generation_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
