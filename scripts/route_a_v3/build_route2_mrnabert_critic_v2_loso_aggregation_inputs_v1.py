#!/usr/bin/env python3
"""Build three aligned Critic V2/matched-baseline LOSO aggregation inputs."""

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
    assigned_gpu,
)


PRIMARY_PROTOCOL_SCHEMA = (
    "route_a_v3_route2_mrnabert_critic_v2_test_preserving_loso_protocol.v1"
)
BASELINE_PROTOCOL_SCHEMA = (
    "route_a_v3_route2_mrnabert_critic_v2_matched_baseline_loso_protocol.v1"
)
AGGREGATION_PROTOCOL_SCHEMA = (
    "route_a_v3_route2_mrnabert_critic_v2_loso_aggregation_protocol.v1"
)
FROZEN_STATUS = "FROZEN_BEFORE_CRITIC_V2_THREE_SEED_OUTCOMES"
PRIMARY_KIND = "delta_pretrained_mrnabert_edit_centered_antisymmetric"
BASELINE_KIND = "delta_anchored_position_aware_antisymmetric"
ZERO_RECORD_STUDIES = ("GSE256185",)


class CriticV2LosoInputError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CriticV2LosoInputError(message)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{label} root is not an object: {path}")
    return value


def _validate_protocols(
    primary: Mapping[str, Any],
    baseline: Mapping[str, Any],
    aggregation: Mapping[str, Any],
) -> None:
    for value, schema, label in (
        (primary, PRIMARY_PROTOCOL_SCHEMA, "primary"),
        (baseline, BASELINE_PROTOCOL_SCHEMA, "baseline"),
        (aggregation, AGGREGATION_PROTOCOL_SCHEMA, "aggregation"),
    ):
        _require(value.get("schema_version") == schema, f"{label} protocol schema differs")
        _require(value.get("status") == FROZEN_STATUS, f"{label} protocol was not prospectively frozen")
        _require(value.get("evaluation_outcomes_accessed") is False, f"Evaluation entered {label} protocol")
        _require(value.get("guided_generation_authorized") is False, f"guidance entered {label} protocol")

    _require(
        aggregation.get("primary_loso_protocol")
        == "configs/route_a_v3_route2_mrnabert_critic_v2_test_preserving_loso_protocol_v1.json"
        and aggregation.get("matched_baseline_loso_protocol")
        == "configs/route_a_v3_route2_mrnabert_critic_v2_matched_baseline_loso_protocol_v1.json",
        "aggregation protocol bindings differ",
    )
    _require(
        tuple(int(seed) for seed in aggregation.get("required_seeds", ()))
        == FINAL_SEEDS
        and tuple(aggregation.get("required_loso_studies", ())) == HOLDOUT_STUDIES
        and tuple(aggregation.get("zero_record_development_studies", ()))
        == ZERO_RECORD_STUDIES,
        "aggregation seed or study inventory differs",
    )
    _require(
        tuple(int(seed) for seed in primary.get("required_seeds", ()))
        == tuple(int(seed) for seed in baseline.get("required_seeds", ()))
        == FINAL_SEEDS
        and tuple(primary.get("holdout_studies", ()))
        == tuple(baseline.get("holdout_studies", ()))
        == HOLDOUT_STUDIES,
        "LOSO protocols disagree on seeds or studies",
    )
    _require(
        aggregation.get("pairing_policy")
        == "EXACT_PRIMARY_BASELINE_STUDY_SEED_PHYSICAL_GPU_AND_OUTPUT_BINDING"
        and aggregation.get("metric_source")
        == "TERMINAL_TRAINING_SUMMARY_VALIDATION_METRICS"
        and aggregation.get("aggregation_input_schema")
        == "route_a_v3_route2_loso_aggregation_input.v1"
        and aggregation.get("aggregation_output_schema")
        == "route_a_v3_route2_loso_aggregation.v1"
        and aggregation.get("development_test_outcomes_accessed") is False,
        "aggregation metric or protected-outcome policy differs",
    )


def _index_configs(
    configs: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    primary: bool,
) -> dict[tuple[str, int], Mapping[str, Any]]:
    label = "primary" if primary else "baseline"
    _require(len(configs) == 21, f"exactly 21 {label} configs are required")
    expected_role = (
        "CRITIC_V2_TEST_PRESERVING_CROSS_STUDY_TRANSFER"
        if primary
        else "CRITIC_V2_STRONGEST_BASELINE_TEST_PRESERVING_LOSO"
    )
    expected_kind = PRIMARY_KIND if primary else BASELINE_KIND
    binding_key = (
        "loso_protocol_schema_version"
        if primary
        else "matched_baseline_loso_protocol_schema_version"
    )
    binding_schema = PRIMARY_PROTOCOL_SCHEMA if primary else BASELINE_PROTOCOL_SCHEMA
    run_root = Path(str(protocol["run_root"]))
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for config in configs:
        study = str(config.get("loso_holdout_study_unit_id"))
        seed = int(config.get("seed", -1))
        key = (study, seed)
        _require(key not in indexed, f"duplicate {label} fold: {study}/{seed}")
        _require(study in HOLDOUT_STUDIES and seed in FINAL_SEEDS, f"unexpected {label} fold: {study}/{seed}")
        gpu = assigned_gpu(study, seed)
        _require(
            config.get("scientific_role") == expected_role
            and config.get("model_kind") == expected_kind
            and config.get("candidate_control") == "NONE",
            f"{label} model identity differs: {study}/{seed}",
        )
        _require(
            config.get("run_mode") == "LOSO_DEVELOPMENT_TRAIN_VALIDATION_ONLY"
            and config.get("result_stage")
            == "LOSO_DEVELOPMENT_VALIDATION_ONLY_FROZEN_HYPERPARAMETERS"
            and config.get("development_record_scope")
            == "TRAIN_VALIDATION_ONLY_TEST_WITHHELD",
            f"{label} LOSO stage differs: {study}/{seed}",
        )
        _require(
            config.get("physical_gpu_index") == gpu
            and config.get("device") == f"cuda:{gpu}"
            and config.get("output_directory")
            == str(run_root / study / f"seed{seed}_gpu{gpu}"),
            f"{label} GPU or output binding differs: {study}/{seed}",
        )
        _require(
            config.get(binding_key) == binding_schema,
            f"{label} protocol binding differs: {study}/{seed}",
        )
        _require(
            config.get("development_test_outcomes_accessed") is False
            and config.get("test_metrics_used_for_loso_selection") is False
            and config.get("evaluation_outcomes_accessed") is False,
            f"protected outcome entered {label} config: {study}/{seed}",
        )
        indexed[key] = config
    _require(
        set(indexed) == {(study, seed) for study in HOLDOUT_STUDIES for seed in FINAL_SEEDS},
        f"{label} fold set differs",
    )
    return indexed


def _terminal_summary(
    config: Mapping[str, Any], *, model_kind: str, label: str
) -> dict[str, Any]:
    path = Path(str(config["output_directory"])) / "training_summary.json"
    summary = _read_json(path, f"{label} terminal summary")
    study = str(config["loso_holdout_study_unit_id"])
    seed = int(config["seed"])
    gpu = int(config["physical_gpu_index"])
    _require(
        summary.get("status") == "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE"
        and summary.get("model_kind") == model_kind
        and summary.get("candidate_control") == "NONE",
        f"{label} terminal model identity differs: {study}/{seed}",
    )
    _require(
        summary.get("run_mode") == "LOSO_DEVELOPMENT_TRAIN_VALIDATION_ONLY"
        and summary.get("result_stage")
        == "LOSO_DEVELOPMENT_VALIDATION_ONLY_FROZEN_HYPERPARAMETERS"
        and summary.get("loso_holdout_study_unit_id") == study
        and summary.get("seed") == seed,
        f"{label} terminal fold identity differs: {study}/{seed}",
    )
    _require(
        summary.get("physical_gpu_index") == gpu
        and summary.get("device") == f"cuda:{gpu}"
        and summary.get("cpu_fallback_used") is False
        and summary.get("optimizer_steps", 0) > 0
        and summary.get("parameter_changed") is True
        and summary.get("cuda_training_tensors_verified") is True,
        f"{label} terminal GPU update differs: {study}/{seed}",
    )
    _require(
        summary.get("loso_development_test_preserved") is True
        and summary.get("development_test_outcomes_evaluated") is False
        and summary.get("development_test_record_count_withheld") == 18292
        and summary.get("test_metrics") is None
        and summary.get("evaluation_outcomes_read") == 0,
        f"protected outcome entered {label} summary: {study}/{seed}",
    )
    metrics = summary.get("validation_metrics")
    _require(isinstance(metrics, Mapping), f"{label} validation metrics are absent: {study}/{seed}")
    return summary


def build_inputs(
    primary_configs: Sequence[Mapping[str, Any]],
    baseline_configs: Sequence[Mapping[str, Any]],
    primary_protocol: Mapping[str, Any],
    baseline_protocol: Mapping[str, Any],
    aggregation_protocol: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    primary_by_key, baseline_by_key = validate_config_pairs(
        primary_configs,
        baseline_configs,
        primary_protocol,
        baseline_protocol,
        aggregation_protocol,
    )

    outputs: dict[int, dict[str, Any]] = {}
    for seed in FINAL_SEEDS:
        model_results = []
        baseline_results = []
        for study in HOLDOUT_STUDIES:
            primary_config = primary_by_key[(study, seed)]
            baseline_config = baseline_by_key[(study, seed)]
            primary_summary = _terminal_summary(
                primary_config, model_kind=PRIMARY_KIND, label="primary"
            )
            baseline_summary = _terminal_summary(
                baseline_config, model_kind=BASELINE_KIND, label="baseline"
            )
            model_results.append(
                {
                    "study_unit_id": study,
                    "training_summary": primary_summary,
                    "evaluation": {
                        "split": f"LOSO::{study}",
                        "metrics": primary_summary["validation_metrics"],
                    },
                }
            )
            baseline_results.append(
                {
                    "study_unit_id": study,
                    "training_summary": baseline_summary,
                    "evaluation": {
                        "split": f"LOSO::{study}",
                        "metrics": baseline_summary["validation_metrics"],
                    },
                }
            )
        outputs[seed] = {
            "schema_version": "route_a_v3_route2_loso_aggregation_input.v1",
            "seed": seed,
            "development_inventory_studies": [
                *HOLDOUT_STUDIES,
                *ZERO_RECORD_STUDIES,
            ],
            "expected_loso_studies": list(HOLDOUT_STUDIES),
            "zero_record_development_studies": list(ZERO_RECORD_STUDIES),
            "model_results": model_results,
            "baseline_results": baseline_results,
            "primary_loso_protocol_schema_version": PRIMARY_PROTOCOL_SCHEMA,
            "matched_baseline_loso_protocol_schema_version": BASELINE_PROTOCOL_SCHEMA,
            "aggregation_protocol_schema_version": AGGREGATION_PROTOCOL_SCHEMA,
            "development_test_opened": False,
            "evaluation_opened": False,
        }
    return outputs


def validate_config_pairs(
    primary_configs: Sequence[Mapping[str, Any]],
    baseline_configs: Sequence[Mapping[str, Any]],
    primary_protocol: Mapping[str, Any],
    baseline_protocol: Mapping[str, Any],
    aggregation_protocol: Mapping[str, Any],
) -> tuple[
    dict[tuple[str, int], Mapping[str, Any]],
    dict[tuple[str, int], Mapping[str, Any]],
]:
    _validate_protocols(primary_protocol, baseline_protocol, aggregation_protocol)
    primary_by_key = _index_configs(primary_configs, primary_protocol, primary=True)
    baseline_by_key = _index_configs(baseline_configs, baseline_protocol, primary=False)
    for study in HOLDOUT_STUDIES:
        for seed in FINAL_SEEDS:
            primary_config = primary_by_key[(study, seed)]
            baseline_config = baseline_by_key[(study, seed)]
            _require(
                baseline_config.get("physical_gpu_index")
                == primary_config.get("physical_gpu_index")
                and baseline_config.get("paired_primary_baseline_id")
                == primary_config.get("baseline_id")
                and baseline_config.get("paired_primary_output_directory")
                == primary_config.get("output_directory"),
                f"primary/baseline pairing differs: {study}/{seed}",
            )
    return primary_by_key, baseline_by_key


def _read_config_root(root: Path, label: str) -> list[dict[str, Any]]:
    _require(root.is_dir(), f"{label} config root is absent: {root}")
    paths = sorted(root.glob("*.json"))
    _require(len(paths) == 21, f"{label} config root must contain exactly 21 JSON files")
    return [_read_json(path, f"{label} config") for path in paths]


def write_inputs_once(
    payloads: Mapping[int, Mapping[str, Any]],
    input_root: Path,
    aggregation_output_root: Path,
) -> list[Path]:
    _require(not input_root.exists(), f"LOSO input root already exists: {input_root}")
    _require(
        not aggregation_output_root.exists(),
        f"LOSO aggregation output root already exists: {aggregation_output_root}",
    )
    _require(set(payloads) == set(FINAL_SEEDS), "LOSO payload seed set differs")
    input_root.mkdir(parents=True)
    paths = []
    for seed in FINAL_SEEDS:
        path = input_root / f"critic_v2_test_preserving_loso_aggregation_input_seed{seed}.json"
        path.write_text(
            json.dumps(payloads[seed], indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-protocol", type=Path, required=True)
    parser.add_argument("--baseline-protocol", type=Path, required=True)
    parser.add_argument("--aggregation-protocol", type=Path, required=True)
    args = parser.parse_args()

    primary_protocol = _read_json(args.primary_protocol, "primary protocol")
    baseline_protocol = _read_json(args.baseline_protocol, "baseline protocol")
    aggregation_protocol = _read_json(
        args.aggregation_protocol, "aggregation protocol"
    )
    payloads = build_inputs(
        _read_config_root(
            Path(str(primary_protocol["runtime_config_root"])), "primary"
        ),
        _read_config_root(
            Path(str(baseline_protocol["runtime_config_root"])), "baseline"
        ),
        primary_protocol,
        baseline_protocol,
        aggregation_protocol,
    )
    paths = write_inputs_once(
        payloads,
        Path(str(aggregation_protocol["input_output_root"])),
        Path(str(aggregation_protocol["aggregation_output_root"])),
    )
    print(
        json.dumps(
            {
                "status": "CRITIC_V2_THREE_LOSO_AGGREGATION_INPUTS_BUILT_NOT_AGGREGATED",
                "paths": [str(path) for path in paths],
                "development_test_opened": False,
                "evaluation_opened": False,
                "aggregation_executed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
