#!/usr/bin/env python3
"""Prepare the frozen 7-study x 3-seed Critic V2 LOSO cohort after refit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_loso_schedule import (  # noqa: E402
    FINAL_SEEDS,
    HOLDOUT_STUDIES,
    PHYSICAL_GPU_INDICES,
    loso_assignments,
)


REFIT_PROTOCOL_SCHEMA = (
    "route_a_v3_route2_mrnabert_critic_v2_all_development_refit_protocol.v1"
)
LOSO_PROTOCOL_SCHEMA = (
    "route_a_v3_route2_mrnabert_critic_v2_test_preserving_loso_protocol.v1"
)
PRIMARY_KIND = "delta_pretrained_mrnabert_edit_centered_antisymmetric"


class CriticV2TestPreservingLosoPreparationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CriticV2TestPreservingLosoPreparationError(message)


def _loso_policy_from_refit_policy(refit_policy: Mapping[str, Any]) -> dict[str, Any]:
    policy = dict(refit_policy)
    policy.pop("checkpoint_selection_before_test")
    policy["checkpoint_selection"] = "FINAL_EPOCH"
    return policy


def _validate_refit_config(
    config: Mapping[str, Any], refit_protocol: Mapping[str, Any]
) -> None:
    seed = int(refit_protocol["single_frozen_test_seed"])
    _require(
        config.get("scientific_role") == "CRITIC_V2_FINAL_ALL_DEVELOPMENT_REFIT",
        "input config is not the Critic V2 all-Development refit",
    )
    _require(
        config.get("result_stage") == "FINAL_ALL_DEVELOPMENT_REFIT",
        "input config is not the final refit stage",
    )
    _require(
        config.get("run_mode") == "FIXED_GROUPED_SPLIT",
        "all-Development refit split differs",
    )
    _require(config.get("model_kind") == PRIMARY_KIND, "refit model kind differs")
    _require(config.get("candidate_control") == "NONE", "refit config is a control")
    _require(
        config.get("baseline_id") == f"mrnabert_critic_v2_all126165_refit_seed{seed}",
        "all-Development refit identity differs",
    )
    _require(int(config.get("seed", -1)) == seed, "all-Development refit seed differs")
    _require(
        config.get("output_directory") == str(refit_protocol["run_directory"]),
        "all-Development refit run path differs",
    )
    _require(
        config.get("all_development_refit_protocol_schema_version")
        == REFIT_PROTOCOL_SCHEMA,
        "all-Development refit protocol binding differs",
    )
    _require(
        config.get("development_record_scope") == "ALL_126165"
        and config.get("train_validation_test_folded_into_training") is True,
        "all-Development refit scope differs",
    )
    _require(
        config.get("refit_model_selection_performed") is False
        and config.get("test_metrics_used_for_refit_selection") is False,
        "all-Development refit performed selection",
    )
    _require(
        config.get("development_test_outcomes_accessed") is True,
        "all-Development refit config did not include the prior TEST partition",
    )
    _require(
        config.get("evaluation_outcomes_accessed") is False,
        "Evaluation entered the all-Development refit config",
    )
    for key, expected in refit_protocol["frozen_model_training_policy"].items():
        if key == "checkpoint_selection_before_test":
            continue
        _require(config.get(key) == expected, f"refit config frozen policy differs: {key}")
    _require(
        config.get("checkpoint_selection") == "FINAL_EPOCH",
        "all-Development refit checkpoint policy differs",
    )


def _validate_refit_summary(
    summary: Mapping[str, Any],
    refit_config: Mapping[str, Any],
    refit_protocol: Mapping[str, Any],
    loso_protocol: Mapping[str, Any],
) -> None:
    _require(
        summary.get("status") == loso_protocol["required_refit_status"],
        "all-Development refit is incomplete",
    )
    _require(
        summary.get("result_stage") == loso_protocol["required_refit_stage"],
        "summary is not the all-Development refit",
    )
    _require(
        summary.get("baseline_id") == refit_config.get("baseline_id"),
        "all-Development refit summary identity differs",
    )
    _require(summary.get("model_kind") == PRIMARY_KIND, "refit summary model kind differs")
    _require(
        int(summary.get("seed", -1)) == int(refit_protocol["single_frozen_test_seed"]),
        "refit summary seed differs",
    )
    _require(summary.get("candidate_control") == "NONE", "refit summary is a control")
    _require(
        summary.get("development_validation_folded_into_training") is True,
        "refit summary did not fold all Development partitions into training",
    )
    _require(
        summary.get("development_test_record_count_withheld") == 0,
        "refit summary kept Development TEST withheld",
    )
    _require(
        summary.get("development_test_outcomes_evaluated") is False
        and "test_metrics" in summary
        and summary.get("test_metrics") is None,
        "refit evaluated a separate TEST or retained TEST metrics",
    )
    _require(
        summary.get("record_counts")
        == {"TRAIN": int(refit_protocol["development_record_count"])},
        "all-Development refit record count differs",
    )
    _require(
        summary.get("evaluation_outcomes_read") == 0,
        "Evaluation entered the all-Development refit",
    )
    _require(
        summary.get("checkpoint_selection") == "FINAL_EPOCH",
        "refit summary checkpoint policy differs",
    )
    _require(
        int(summary.get("selected_epoch", -1))
        == int(summary.get("final_training_epoch", -2))
        == int(refit_protocol["frozen_model_training_policy"]["epochs"]),
        "refit summary did not use the frozen final epoch",
    )
    _require(
        summary.get("cuda_training_tensors_verified") is True,
        "refit CUDA training tensors were not verified",
    )
    _require(summary.get("cpu_fallback_used") is False, "refit used CPU fallback")
    _require(summary.get("parameter_changed") is True, "refit has no learned update")
    _require(int(summary.get("optimizer_steps", 0)) > 0, "refit has no optimizer steps")
    _require(
        0 <= int(summary.get("physical_gpu_index", -1)) <= 5,
        "refit physical GPU is outside GPU0-5",
    )
    for key, expected in loso_protocol["frozen_loso_training_policy"].items():
        if key == "target_scaling_mode":
            observed = summary.get("target_scaler", {}).get("mode")
        elif key == "epochs":
            observed = summary.get("final_training_epoch")
        else:
            observed = summary.get(key)
        _require(observed == expected, f"refit summary frozen policy differs: {key}")


def _validate_protocols(
    refit_protocol: Mapping[str, Any], loso_protocol: Mapping[str, Any]
) -> None:
    _require(
        refit_protocol.get("schema_version") == REFIT_PROTOCOL_SCHEMA,
        "unexpected Critic V2 refit protocol",
    )
    _require(
        refit_protocol.get("status")
        == "FROZEN_BEFORE_CRITIC_V2_THREE_SEED_OUTCOMES",
        "Critic V2 refit protocol was not prospectively frozen",
    )
    _require(
        refit_protocol.get("evaluation_outcomes_accessed") is False,
        "Evaluation entered the refit protocol",
    )
    _require(
        refit_protocol.get("development_test_outcomes_accessed_at_protocol_freeze")
        is False,
        "Development TEST entered before refit protocol freeze",
    )
    _require(
        refit_protocol.get("guided_generation_authorized") is False,
        "guided generation entered the refit protocol",
    )
    _require(
        loso_protocol.get("schema_version") == LOSO_PROTOCOL_SCHEMA,
        "unexpected Critic V2 LOSO protocol",
    )
    _require(
        loso_protocol.get("status")
        == "FROZEN_BEFORE_CRITIC_V2_THREE_SEED_OUTCOMES",
        "Critic V2 LOSO protocol was not prospectively frozen",
    )
    _require(
        loso_protocol.get("all_development_refit_protocol")
        == "configs/route_a_v3_route2_mrnabert_critic_v2_all_development_refit_protocol_v1.json",
        "LOSO protocol binds a different refit protocol",
    )
    _require(
        loso_protocol.get("required_refit_status")
        == "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE"
        and loso_protocol.get("required_refit_stage")
        == "FINAL_ALL_DEVELOPMENT_REFIT",
        "LOSO required refit terminal state differs",
    )
    _require(
        tuple(loso_protocol["holdout_studies"]) == HOLDOUT_STUDIES,
        "LOSO holdout study cohort differs",
    )
    _require(
        tuple(int(seed) for seed in loso_protocol["required_seeds"]) == FINAL_SEEDS,
        "LOSO seed cohort differs",
    )
    _require(
        tuple(int(gpu) for gpu in loso_protocol["physical_gpu_indices"])
        == PHYSICAL_GPU_INDICES,
        "LOSO physical GPU cohort differs",
    )
    _require(
        int(loso_protocol["required_config_count"])
        == len(HOLDOUT_STUDIES) * len(FINAL_SEEDS)
        == 21,
        "LOSO config count differs",
    )
    _require(
        loso_protocol.get("assignment_policy")
        == "STUDY_MAJOR_SEED_MINOR_ROUND_ROBIN_GPU0_TO_GPU5",
        "LOSO GPU assignment policy differs",
    )
    expected_data_policy = {
        "input_partitions": ["TRAIN", "VALIDATION"],
        "development_test_partition_used": False,
        "holdout_unit": "STUDY_UNIT_ID",
        "cross_study_connected_source_components_excluded_from_training": True,
    }
    _require(
        loso_protocol.get("loso_data_policy") == expected_data_policy,
        "LOSO data policy differs",
    )
    expected_loso_policy = _loso_policy_from_refit_policy(
        refit_protocol["frozen_model_training_policy"]
    )
    _require(
        loso_protocol.get("frozen_loso_training_policy") == expected_loso_policy,
        "LOSO training policy differs from the frozen refit policy",
    )
    for key in (
        "development_test_outcomes_accessed_by_loso",
        "test_metrics_used_for_loso_selection",
        "evaluation_outcomes_accessed",
        "guided_generation_authorized",
    ):
        _require(loso_protocol.get(key) is False, f"protected boundary differs: {key}")
    _require(
        int(loso_protocol["development_test_record_count_withheld_per_run"])
        == int(refit_protocol["development_test_record_count"]),
        "LOSO withheld TEST count differs",
    )


def build_configs(
    refit_config: Mapping[str, Any],
    refit_summary: Mapping[str, Any],
    refit_protocol: Mapping[str, Any],
    loso_protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _validate_protocols(refit_protocol, loso_protocol)
    _validate_refit_config(refit_config, refit_protocol)
    _validate_refit_summary(
        refit_summary, refit_config, refit_protocol, loso_protocol
    )

    configs = []
    run_root = Path(str(loso_protocol["run_root"]))
    stage_specific_keys = {
        "all_development_refit_protocol_schema_version",
        "development_record_scope",
        "train_validation_test_folded_into_training",
        "refit_model_selection_performed",
        "test_metrics_used_for_refit_selection",
        "frozen_test_protocol_schema_version",
        "development_validation_folded_into_training",
        "validation_checkpoint_selection_before_test",
        "test_used_for_checkpoint_selection",
        "test_used_for_model_or_policy_selection",
    }
    for study, seed, gpu in loso_assignments():
        config = {
            key: value
            for key, value in refit_config.items()
            if key not in stage_specific_keys
        }
        config.update(loso_protocol["frozen_loso_training_policy"])
        config.update(
            {
                "scientific_role": "CRITIC_V2_TEST_PRESERVING_CROSS_STUDY_TRANSFER",
                "result_stage": "LOSO_DEVELOPMENT_VALIDATION_ONLY_FROZEN_HYPERPARAMETERS",
                "run_mode": "LOSO_DEVELOPMENT_TRAIN_VALIDATION_ONLY",
                "baseline_id": f"mrnabert_critic_v2_loso_{study.lower()}_seed{seed}",
                "attempt_purpose": "CRITIC_V2_THREE_SEED_TEST_PRESERVING_LOSO",
                "seed": seed,
                "device": f"cuda:{gpu}",
                "physical_gpu_index": gpu,
                "candidate_control": "NONE",
                "loso_holdout_study_unit_id": study,
                "development_record_scope": "TRAIN_VALIDATION_ONLY_TEST_WITHHELD",
                "development_test_outcomes_accessed": False,
                "development_test_outcomes_previously_accessed_for_single_frozen_test": True,
                "test_metrics_used_for_loso_selection": False,
                "all_development_refit_completed_before_loso": True,
                "evaluation_outcomes_accessed": False,
                "loso_protocol_schema_version": LOSO_PROTOCOL_SCHEMA,
                "output_directory": str(
                    run_root / study / f"seed{seed}_gpu{gpu}"
                ),
                "notes": (
                    "Critic V2 TEST-preserving LOSO after terminal all-Development "
                    "refit. This run uses original Development TRAIN/VALIDATION only; "
                    "the 18,292-row Development TEST and Evaluation remain unopened."
                ),
            }
        )
        configs.append(config)
    _require(len(configs) == 21, "LOSO config count differs after construction")
    _require(
        len({config["baseline_id"] for config in configs}) == 21,
        "LOSO config identity is duplicated",
    )
    return configs


def write_configs_once(
    configs: Sequence[Mapping[str, Any]], config_root: Path
) -> list[Path]:
    _require(
        not config_root.exists(),
        f"LOSO runtime config root already exists: {config_root}",
    )
    for config in configs:
        run_directory = Path(str(config["output_directory"]))
        _require(
            not run_directory.exists(),
            f"LOSO run directory already exists: {run_directory}",
        )
    config_root.mkdir(parents=True)
    paths = []
    for config in configs:
        path = config_root / f"{config['baseline_id']}.json"
        path.write_text(
            json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refit-config", type=Path, required=True)
    parser.add_argument("--refit-summary", type=Path, required=True)
    parser.add_argument("--refit-protocol", type=Path, required=True)
    parser.add_argument("--loso-protocol", type=Path, required=True)
    args = parser.parse_args()

    loso_protocol = json.loads(args.loso_protocol.read_text(encoding="utf-8"))
    configs = build_configs(
        json.loads(args.refit_config.read_text(encoding="utf-8")),
        json.loads(args.refit_summary.read_text(encoding="utf-8")),
        json.loads(args.refit_protocol.read_text(encoding="utf-8")),
        loso_protocol,
    )
    paths = write_configs_once(
        configs, Path(str(loso_protocol["runtime_config_root"]))
    )
    print(
        json.dumps(
            {
                "status": "CRITIC_V2_TEST_PRESERVING_LOSO_CONFIGS_PREPARED_NOT_EXECUTED",
                "config_count": len(paths),
                "holdout_studies": list(HOLDOUT_STUDIES),
                "seeds": list(FINAL_SEEDS),
                "physical_gpu_indices": list(PHYSICAL_GPU_INDICES),
                "development_test_opened_by_loso": False,
                "evaluation_opened": False,
                "loso_executed": False,
                "paths": [str(path) for path in paths],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
