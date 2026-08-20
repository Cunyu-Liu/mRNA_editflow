#!/usr/bin/env python3
"""Prepare strongest-baseline LOSO configs paired to Critic V2 primary folds."""

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


PRIMARY_LOSO_PROTOCOL_SCHEMA = (
    "route_a_v3_route2_mrnabert_critic_v2_test_preserving_loso_protocol.v1"
)
BASELINE_LOSO_PROTOCOL_SCHEMA = (
    "route_a_v3_route2_mrnabert_critic_v2_matched_baseline_loso_protocol.v1"
)
BASE_CONFIG_SCHEMA = "route_a_v3_route2_method_repair_screen.v1"
PRIMARY_KIND = "delta_pretrained_mrnabert_edit_centered_antisymmetric"
BASELINE_ID = "method_repair_global_scaled_seed20260821"
BASELINE_KIND = "delta_anchored_position_aware_antisymmetric"


class CriticV2MatchedBaselineLosoPreparationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CriticV2MatchedBaselineLosoPreparationError(message)


def _validate_protocols(
    primary_protocol: Mapping[str, Any], baseline_protocol: Mapping[str, Any]
) -> None:
    _require(
        primary_protocol.get("schema_version") == PRIMARY_LOSO_PROTOCOL_SCHEMA,
        "unexpected Critic V2 primary LOSO protocol",
    )
    _require(
        primary_protocol.get("status")
        == "FROZEN_BEFORE_CRITIC_V2_THREE_SEED_OUTCOMES",
        "Critic V2 primary LOSO protocol was not prospectively frozen",
    )
    _require(
        baseline_protocol.get("schema_version") == BASELINE_LOSO_PROTOCOL_SCHEMA,
        "unexpected Critic V2 matched-baseline LOSO protocol",
    )
    _require(
        baseline_protocol.get("status")
        == "FROZEN_BEFORE_CRITIC_V2_THREE_SEED_OUTCOMES",
        "matched-baseline LOSO protocol was not prospectively frozen",
    )
    _require(
        baseline_protocol.get("primary_loso_protocol")
        == "configs/route_a_v3_route2_mrnabert_critic_v2_test_preserving_loso_protocol_v1.json",
        "matched-baseline protocol binds a different primary LOSO protocol",
    )
    _require(
        baseline_protocol.get("base_config")
        == "configs/route_a_v3_route2_method_repair_global_scaled_seed20260821_gpu0_v1.json",
        "matched-baseline protocol binds a different base config",
    )
    for protocol, label in (
        (primary_protocol, "primary LOSO protocol"),
        (baseline_protocol, "matched-baseline LOSO protocol"),
    ):
        _require(
            tuple(protocol["holdout_studies"]) == HOLDOUT_STUDIES,
            f"holdout study cohort differs in {label}",
        )
        _require(
            tuple(int(seed) for seed in protocol["required_seeds"]) == FINAL_SEEDS,
            f"seed cohort differs in {label}",
        )
        _require(
            tuple(int(gpu) for gpu in protocol["physical_gpu_indices"])
            == PHYSICAL_GPU_INDICES,
            f"physical GPU cohort differs in {label}",
        )
        _require(
            int(protocol["required_config_count"]) == 21,
            f"config count differs in {label}",
        )
        _require(
            protocol.get("evaluation_outcomes_accessed") is False
            and protocol.get("guided_generation_authorized") is False,
            f"protected outcome boundary differs in {label}",
        )
    _require(
        primary_protocol.get("development_test_outcomes_accessed_by_loso") is False
        and primary_protocol.get("test_metrics_used_for_loso_selection") is False,
        "primary LOSO protocol opened or selected on TEST",
    )
    _require(
        baseline_protocol.get("development_test_outcomes_accessed_by_loso") is False
        and baseline_protocol.get("test_metrics_used_for_loso_selection") is False,
        "matched-baseline LOSO protocol opened or selected on TEST",
    )
    _require(
        baseline_protocol.get("pairing_policy")
        == "EXACT_STUDY_SEED_PHYSICAL_GPU_AND_TEST_PRESERVING_SPLIT",
        "matched-baseline pairing policy differs",
    )
    _require(
        int(baseline_protocol["development_test_record_count_withheld_per_run"])
        == int(primary_protocol["development_test_record_count_withheld_per_run"])
        == 18292,
        "matched LOSO withheld TEST count differs",
    )
    baseline = baseline_protocol["strongest_same_information_baseline"]
    _require(
        baseline.get("baseline_id") == BASELINE_ID
        and baseline.get("model_kind") == BASELINE_KIND,
        "strongest same-information baseline identity differs",
    )


def _base_value(base: Mapping[str, Any], key: str) -> Any:
    defaults = {
        "training_sampling_mode": "COMPLETE_PASS_LENGTH_BUCKET",
        "loss_aggregation_mode": "RECORD_WEIGHTED",
        "training_update_mode": "STANDARD",
        "optimizer_name": "AdamW",
        "training_precision": "FP32",
        "pin_memory": False,
        "non_blocking_transfer": False,
    }
    return base.get(key, defaults.get(key))


def _validate_base_config(
    base: Mapping[str, Any], baseline_protocol: Mapping[str, Any]
) -> None:
    _require(base.get("schema_version") == BASE_CONFIG_SCHEMA, "unexpected baseline base config")
    _require(base.get("scientific_role") == "FACTORIAL_GLOBAL_SCALED", "baseline role differs")
    _require(base.get("baseline_id") == BASELINE_ID, "strongest baseline was substituted")
    _require(base.get("model_kind") == BASELINE_KIND, "strongest baseline model kind differs")
    _require(base.get("result_stage") == "HPO_VALIDATION_ONLY", "baseline is not HPO-only")
    _require(base.get("run_mode") == "FIXED_GROUPED_SPLIT", "baseline split differs")
    _require(base.get("candidate_control") == "NONE", "baseline config is a control")
    _require(
        base.get("development_test_outcomes_accessed") is False,
        "Development TEST entered the strongest baseline config",
    )
    _require(
        base.get("evaluation_outcomes_accessed") is False,
        "Evaluation entered the strongest baseline config",
    )
    _require(
        base.get("checkpoint_selection") == "BEST_VALIDATION",
        "strongest baseline HPO checkpoint policy differs",
    )
    summary_path = Path(
        str(
            baseline_protocol["strongest_same_information_baseline"][
                "training_summary_path"
            ]
        )
    )
    _require(
        Path(str(base.get("output_directory"))) == summary_path.parent,
        "strongest baseline run provenance differs",
    )
    for key, expected in baseline_protocol["frozen_baseline_training_policy"].items():
        if key == "checkpoint_selection":
            continue
        _require(_base_value(base, key) == expected, f"baseline frozen policy differs: {key}")


def _validate_primary_configs(
    configs: Sequence[Mapping[str, Any]], primary_protocol: Mapping[str, Any]
) -> dict[tuple[str, int], Mapping[str, Any]]:
    _require(len(configs) == 21, "exactly 21 primary LOSO configs are required")
    by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    expected_assignments = {
        (study, seed): gpu for study, seed, gpu in loso_assignments()
    }
    run_root = Path(str(primary_protocol["run_root"]))
    for config in configs:
        study = str(config.get("loso_holdout_study_unit_id"))
        seed = int(config.get("seed", -1))
        key = (study, seed)
        _require(key not in by_key, f"primary LOSO fold is duplicated: {study}/{seed}")
        _require(key in expected_assignments, f"unexpected primary LOSO fold: {study}/{seed}")
        gpu = expected_assignments[key]
        _require(
            config.get("scientific_role")
            == "CRITIC_V2_TEST_PRESERVING_CROSS_STUDY_TRANSFER",
            f"primary LOSO role differs: {study}/{seed}",
        )
        _require(
            config.get("result_stage")
            == "LOSO_DEVELOPMENT_VALIDATION_ONLY_FROZEN_HYPERPARAMETERS"
            and config.get("run_mode")
            == "LOSO_DEVELOPMENT_TRAIN_VALIDATION_ONLY",
            f"primary LOSO stage/split differs: {study}/{seed}",
        )
        _require(config.get("model_kind") == PRIMARY_KIND, f"primary model kind differs: {study}/{seed}")
        _require(config.get("candidate_control") == "NONE", f"primary config is a control: {study}/{seed}")
        _require(
            config.get("baseline_id")
            == f"mrnabert_critic_v2_loso_{study.lower()}_seed{seed}",
            f"primary LOSO identity differs: {study}/{seed}",
        )
        _require(
            config.get("physical_gpu_index") == gpu
            and config.get("device") == f"cuda:{gpu}",
            f"primary LOSO GPU differs: {study}/{seed}",
        )
        _require(
            config.get("output_directory")
            == str(run_root / study / f"seed{seed}_gpu{gpu}"),
            f"primary LOSO run path differs: {study}/{seed}",
        )
        _require(
            config.get("loso_protocol_schema_version")
            == PRIMARY_LOSO_PROTOCOL_SCHEMA,
            f"primary LOSO protocol binding differs: {study}/{seed}",
        )
        _require(
            config.get("development_record_scope")
            == "TRAIN_VALIDATION_ONLY_TEST_WITHHELD"
            and config.get("development_test_outcomes_accessed") is False
            and config.get("test_metrics_used_for_loso_selection") is False,
            f"primary LOSO TEST boundary differs: {study}/{seed}",
        )
        _require(
            config.get("all_development_refit_completed_before_loso") is True,
            f"primary LOSO bypassed the refit gate: {study}/{seed}",
        )
        _require(
            config.get("evaluation_outcomes_accessed") is False,
            f"Evaluation entered primary LOSO: {study}/{seed}",
        )
        for policy_key, expected in primary_protocol[
            "frozen_loso_training_policy"
        ].items():
            _require(
                config.get(policy_key) == expected,
                f"primary LOSO policy differs ({policy_key}): {study}/{seed}",
            )
        by_key[key] = config
    _require(set(by_key) == set(expected_assignments), "primary LOSO fold set differs")
    return by_key


def build_configs(
    base: Mapping[str, Any],
    primary_configs: Sequence[Mapping[str, Any]],
    primary_protocol: Mapping[str, Any],
    baseline_protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _validate_protocols(primary_protocol, baseline_protocol)
    _validate_base_config(base, baseline_protocol)
    primary_by_key = _validate_primary_configs(primary_configs, primary_protocol)

    run_root = Path(str(baseline_protocol["run_root"]))
    configs = []
    for study, seed, gpu in loso_assignments():
        primary = primary_by_key[(study, seed)]
        config = dict(base)
        config.update(baseline_protocol["frozen_baseline_training_policy"])
        config.update(
            {
                "scientific_role": "CRITIC_V2_STRONGEST_BASELINE_TEST_PRESERVING_LOSO",
                "result_stage": "LOSO_DEVELOPMENT_VALIDATION_ONLY_FROZEN_HYPERPARAMETERS",
                "run_mode": "LOSO_DEVELOPMENT_TRAIN_VALIDATION_ONLY",
                "baseline_id": f"global_scaled_critic_v2_loso_{study.lower()}_seed{seed}",
                "attempt_purpose": "CRITIC_V2_MATCHED_STRONGEST_BASELINE_LOSO",
                "seed": seed,
                "device": f"cuda:{gpu}",
                "physical_gpu_index": gpu,
                "candidate_control": "NONE",
                "loso_holdout_study_unit_id": study,
                "development_record_scope": "TRAIN_VALIDATION_ONLY_TEST_WITHHELD",
                "development_test_outcomes_accessed": False,
                "test_metrics_used_for_loso_selection": False,
                "evaluation_outcomes_accessed": False,
                "primary_single_test_and_refit_preceded_matched_loso": True,
                "paired_primary_baseline_id": primary["baseline_id"],
                "paired_primary_output_directory": primary["output_directory"],
                "matched_baseline_loso_protocol_schema_version": BASELINE_LOSO_PROTOCOL_SCHEMA,
                "output_directory": str(
                    run_root / study / f"seed{seed}_gpu{gpu}"
                ),
                "notes": (
                    "Strongest same-information baseline LOSO paired to the exact "
                    "Critic V2 holdout study, seed, physical GPU and TEST-preserving "
                    "split. Native baseline capacity and 8-epoch FP32 budget are retained."
                ),
            }
        )
        configs.append(config)
    _require(len(configs) == 21, "matched-baseline LOSO config count differs")
    _require(
        len({config["baseline_id"] for config in configs}) == 21,
        "matched-baseline LOSO identity is duplicated",
    )
    return configs


def write_configs_once(
    configs: Sequence[Mapping[str, Any]], config_root: Path
) -> list[Path]:
    _require(
        not config_root.exists(),
        f"matched-baseline LOSO config root already exists: {config_root}",
    )
    for config in configs:
        run_directory = Path(str(config["output_directory"]))
        _require(
            not run_directory.exists(),
            f"matched-baseline LOSO run directory already exists: {run_directory}",
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


def _read_primary_configs(config_root: Path) -> list[dict[str, Any]]:
    _require(config_root.is_dir(), f"primary LOSO config root is absent: {config_root}")
    paths = sorted(config_root.glob("*.json"))
    _require(len(paths) == 21, "primary LOSO config root does not contain exactly 21 JSON files")
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--primary-config-root", type=Path, required=True)
    parser.add_argument("--primary-protocol", type=Path, required=True)
    parser.add_argument("--baseline-protocol", type=Path, required=True)
    args = parser.parse_args()

    baseline_protocol = json.loads(
        args.baseline_protocol.read_text(encoding="utf-8")
    )
    configs = build_configs(
        json.loads(args.base_config.read_text(encoding="utf-8")),
        _read_primary_configs(args.primary_config_root),
        json.loads(args.primary_protocol.read_text(encoding="utf-8")),
        baseline_protocol,
    )
    paths = write_configs_once(
        configs, Path(str(baseline_protocol["runtime_config_root"]))
    )
    print(
        json.dumps(
            {
                "status": "CRITIC_V2_MATCHED_BASELINE_LOSO_CONFIGS_PREPARED_NOT_EXECUTED",
                "config_count": len(paths),
                "pairing_policy": baseline_protocol["pairing_policy"],
                "development_test_opened": False,
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
