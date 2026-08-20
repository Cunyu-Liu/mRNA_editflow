#!/usr/bin/env python3
"""Prepare the one prospectively frozen Critic V2 Development TEST config."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


CONTROL_PROTOCOL_SCHEMA = "route_a_v3_route2_mrnabert_critic_v2_protocol.v1"
CONFIRMATION_PROTOCOL_SCHEMA = (
    "route_a_v3_route2_mrnabert_critic_v2_three_seed_protocol.v1"
)
FROZEN_TEST_PROTOCOL_SCHEMA = (
    "route_a_v3_route2_mrnabert_critic_v2_frozen_test_protocol.v1"
)
CONTROL_ADJUDICATION_SCHEMA = (
    "route_a_v3_route2_mrnabert_critic_v2_control_adjudication.v1"
)
CONFIRMATION_ADJUDICATION_SCHEMA = (
    "route_a_v3_route2_mrnabert_critic_v2_three_seed_adjudication.v1"
)
PRIMARY_KIND = "delta_pretrained_mrnabert_edit_centered_antisymmetric"
SINGLE_FROZEN_TEST_SEED = 20260823
EXPECTED_CONFIRMATION_CHECKS = {
    "control_adjudication_supports_three_frozen_seeds",
    "all_seed_metrics_finite",
    "all_seed_prediction_spreads_positive",
    "all_seed_task_macros_replay",
    "all_seed_spread_ratios_replay",
    "all_three_seed_margins_over_strongest_baseline_positive",
}


class CriticV2FrozenTestPreparationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CriticV2FrozenTestPreparationError(message)


def _same_float(left: Any, right: Any) -> bool:
    try:
        return math.isclose(
            float(left),
            float(right),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    except (TypeError, ValueError):
        return False


def _validate_frozen_baseline(
    reference: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    label: str,
) -> None:
    _require(
        observed.get("baseline_id") == reference.get("baseline_id"),
        f"strongest baseline identity differs in {label}",
    )
    for key in ("task_macro_spearman", "task_macro_standardized_mae"):
        _require(
            _same_float(observed.get(key), reference.get(key)),
            f"strongest baseline {key} differs in {label}",
        )


def build_config(
    selected_confirmation: Mapping[str, Any],
    control_protocol: Mapping[str, Any],
    confirmation_protocol: Mapping[str, Any],
    frozen_test_protocol: Mapping[str, Any],
    control_adjudication: Mapping[str, Any],
    confirmation_adjudication: Mapping[str, Any],
    *,
    gpu: int,
) -> dict[str, Any]:
    """Validate both frozen gates and construct, but do not execute, TEST config."""

    _require(
        control_protocol.get("schema_version") == CONTROL_PROTOCOL_SCHEMA,
        "unexpected Critic V2 control protocol",
    )
    _require(
        control_protocol.get("status")
        == "FROZEN_BEFORE_CRITIC_V2_TRAINING_OUTCOMES",
        "Critic V2 control protocol was not frozen",
    )
    _require(
        confirmation_protocol.get("schema_version")
        == CONFIRMATION_PROTOCOL_SCHEMA,
        "unexpected Critic V2 confirmation protocol",
    )
    _require(
        confirmation_protocol.get("status")
        == "FROZEN_BEFORE_CRITIC_V2_THREE_SEED_OUTCOMES",
        "Critic V2 confirmation protocol was not prospectively frozen",
    )
    _require(
        frozen_test_protocol.get("schema_version") == FROZEN_TEST_PROTOCOL_SCHEMA,
        "unexpected Critic V2 frozen-TEST protocol",
    )
    _require(
        frozen_test_protocol.get("status")
        == "FROZEN_BEFORE_CRITIC_V2_THREE_SEED_OUTCOMES",
        "Critic V2 frozen-TEST protocol was not prospectively frozen",
    )
    for payload, label in (
        (control_protocol, "control protocol"),
        (confirmation_protocol, "confirmation protocol"),
        (frozen_test_protocol, "frozen-TEST protocol"),
    ):
        _require(
            payload.get("development_test_outcomes_accessed") is False,
            f"Development TEST entered {label}",
        )
        _require(
            payload.get("evaluation_outcomes_accessed") is False,
            f"Evaluation entered {label}",
        )
        _require(
            payload.get("guided_generation_authorized") is False,
            f"guided generation entered {label}",
        )

    frozen_policy = dict(frozen_test_protocol["frozen_training_policy"])
    _require(
        frozen_policy == control_protocol.get("frozen_training_policy"),
        "frozen-TEST policy differs from the control protocol",
    )
    _require(
        frozen_policy == confirmation_protocol.get("frozen_training_policy"),
        "frozen-TEST policy differs from the confirmation protocol",
    )
    _require(
        frozen_policy.get("model_kind") == PRIMARY_KIND,
        "frozen TEST is not the full Critic V2 model",
    )
    _require(
        frozen_policy.get("checkpoint_selection") == "BEST_VALIDATION",
        "frozen TEST does not preserve Validation-only checkpoint selection",
    )

    control_seeds = [int(seed) for seed in control_protocol["frozen_confirmation_seeds"]]
    confirmation_seeds = [int(seed) for seed in confirmation_protocol["required_seeds"]]
    test_protocol_seeds = [
        int(seed) for seed in frozen_test_protocol["required_confirmation_seeds"]
    ]
    _require(
        len(control_seeds) == 3
        and control_seeds == confirmation_seeds == test_protocol_seeds,
        "confirmation seed set differs across frozen protocols",
    )
    test_seed = int(frozen_test_protocol["single_frozen_test_seed"])
    _require(
        test_seed == SINGLE_FROZEN_TEST_SEED and test_seed in test_protocol_seeds,
        "single frozen TEST seed differs from the prospective protocol",
    )
    _require(
        int(frozen_test_protocol["development_test_record_count"])
        == int(control_protocol["development_test_record_count_withheld"]),
        "withheld Development TEST count differs",
    )
    expected_test_policy = {
        "result_stage": "FROZEN_DEVELOPMENT_TEST",
        "run_mode": "FIXED_GROUPED_SPLIT",
        "single_execution_only": True,
        "checkpoint_selected_by": "DEVELOPMENT_VALIDATION_ONLY",
        "test_used_for_checkpoint_selection": False,
        "test_used_for_model_or_policy_selection": False,
    }
    _require(
        frozen_test_protocol.get("test_policy") == expected_test_policy,
        "frozen TEST execution policy differs",
    )

    _require(
        control_adjudication.get("schema_version")
        == CONTROL_ADJUDICATION_SCHEMA,
        "unexpected Critic V2 control adjudication",
    )
    _require(
        control_adjudication.get("status")
        == "CRITIC_V2_CONTROLS_SUPPORT_THREE_FROZEN_SEEDS",
        "Critic V2 controls do not authorize confirmation seeds",
    )
    _require(
        control_adjudication.get("supports_three_frozen_seeds") is True,
        "Critic V2 control gate failed",
    )
    _require(
        control_adjudication.get("development_test_opened") is False,
        "Development TEST entered control adjudication",
    )
    _require(
        control_adjudication.get("evaluation_opened") is False,
        "Evaluation entered control adjudication",
    )
    control_checks = control_adjudication.get("checks")
    _require(
        isinstance(control_checks, Mapping)
        and bool(control_checks)
        and all(value is True for value in control_checks.values()),
        "Critic V2 control adjudication checks are not all PASS",
    )
    _require(
        [int(seed) for seed in control_adjudication["frozen_confirmation_seeds"]]
        == test_protocol_seeds,
        "control adjudication authorized a different seed set",
    )

    _require(
        confirmation_adjudication.get("schema_version")
        == CONFIRMATION_ADJUDICATION_SCHEMA,
        "unexpected Critic V2 confirmation adjudication",
    )
    _require(
        confirmation_adjudication.get("status")
        == "CRITIC_V2_THREE_SEEDS_SUPPORT_ONE_FROZEN_DEVELOPMENT_TEST",
        "Critic V2 confirmation seeds do not authorize Development TEST",
    )
    _require(
        confirmation_adjudication.get("supports_single_frozen_development_test")
        is True,
        "single frozen Development TEST gate failed",
    )
    _require(
        confirmation_adjudication.get("development_test_opened") is False,
        "Development TEST entered confirmation adjudication",
    )
    _require(
        confirmation_adjudication.get("evaluation_opened") is False,
        "Evaluation entered confirmation adjudication",
    )
    _require(
        confirmation_adjudication.get("guided_generation_authorized") is False,
        "guided generation entered confirmation adjudication",
    )
    confirmation_checks = confirmation_adjudication.get("checks")
    _require(
        isinstance(confirmation_checks, Mapping)
        and EXPECTED_CONFIRMATION_CHECKS <= set(confirmation_checks)
        and all(
            confirmation_checks.get(key) is True
            for key in EXPECTED_CONFIRMATION_CHECKS
        ),
        "Critic V2 confirmation adjudication checks are not all PASS",
    )
    seed_results = confirmation_adjudication.get("seed_results")
    _require(
        isinstance(seed_results, list) and len(seed_results) == 3,
        "Critic V2 confirmation adjudication does not contain exactly three seeds",
    )
    by_seed = {int(row.get("seed")): row for row in seed_results}
    _require(
        len(by_seed) == 3 and set(by_seed) == set(test_protocol_seeds),
        "confirmation adjudication seed set differs",
    )
    for seed in test_protocol_seeds:
        row = by_seed[seed]
        margin = row.get("margin_over_strongest_same_information_baseline")
        _require(
            isinstance(margin, (int, float))
            and not isinstance(margin, bool)
            and math.isfinite(float(margin))
            and float(margin) > 0.0,
            f"confirmation seed {seed} does not beat the frozen baseline",
        )
        _require(
            row.get("nonfinite_metric_detected") is False,
            f"confirmation seed {seed} has a nonfinite metric",
        )
        _require(
            row.get("mean_collapse_detected") is False,
            f"confirmation seed {seed} has mean collapse",
        )

    baseline = frozen_test_protocol["strongest_same_information_baseline"]
    for observed, label in (
        (control_protocol["strongest_same_information_baseline"], "control protocol"),
        (
            confirmation_protocol["strongest_same_information_baseline"],
            "confirmation protocol",
        ),
        (
            control_adjudication["strongest_same_information_baseline"],
            "control adjudication",
        ),
        (
            confirmation_adjudication["strongest_same_information_baseline"],
            "confirmation adjudication",
        ),
    ):
        _validate_frozen_baseline(baseline, observed, label=label)

    _require(
        selected_confirmation.get("scientific_role")
        == "CRITIC_V2_THREE_SEED_FROZEN_DEVELOPMENT_VALIDATION_CONFIRMATION",
        "selected config is not a Critic V2 confirmation config",
    )
    _require(
        selected_confirmation.get("result_stage")
        == "FROZEN_DEVELOPMENT_VALIDATION",
        "selected config is not Development Validation-only",
    )
    _require(
        selected_confirmation.get("run_mode") == "FIXED_GROUPED_SPLIT",
        "selected config split differs",
    )
    _require(
        selected_confirmation.get("model_kind") == PRIMARY_KIND,
        "selected config is not the full Critic V2 model",
    )
    _require(
        selected_confirmation.get("candidate_control") == "NONE",
        "selected config is a control",
    )
    _require(
        selected_confirmation.get("development_test_outcomes_accessed") is False,
        "selected config accessed Development TEST",
    )
    _require(
        selected_confirmation.get("evaluation_outcomes_accessed") is False,
        "selected config accessed Evaluation",
    )
    _require(
        int(selected_confirmation.get("seed", -1)) == test_seed,
        "selected confirmation seed differs from the frozen TEST seed",
    )
    _require(
        selected_confirmation.get("baseline_id")
        == f"mrnabert_critic_v2_full_confirmation_seed{test_seed}",
        "selected confirmation identity differs",
    )
    expected_confirmation_run = str(
        Path(str(confirmation_protocol["run_root"])) / f"seed{test_seed}"
    )
    _require(
        selected_confirmation.get("output_directory") == expected_confirmation_run,
        "selected confirmation run path differs",
    )
    for key, expected in frozen_policy.items():
        _require(
            selected_confirmation.get(key) == expected,
            f"selected confirmation frozen policy differs: {key}",
        )
    _require(
        isinstance(gpu, int) and not isinstance(gpu, bool) and 0 <= gpu <= 5,
        "frozen TEST must use physical GPU0-5",
    )

    config = dict(selected_confirmation)
    config.update(frozen_policy)
    config.update(
        {
            "scientific_role": "CRITIC_V2_SINGLE_FROZEN_DEVELOPMENT_TEST",
            "result_stage": "FROZEN_DEVELOPMENT_TEST",
            "run_mode": "FIXED_GROUPED_SPLIT",
            "baseline_id": f"mrnabert_critic_v2_single_frozen_test_seed{test_seed}",
            "attempt_purpose": "MRNABERT_CRITIC_V2_ONE_TIME_FROZEN_DEVELOPMENT_TEST",
            "seed": test_seed,
            "device": f"cuda:{gpu}",
            "physical_gpu_index": gpu,
            "candidate_control": "NONE",
            "development_test_outcomes_accessed": True,
            "evaluation_outcomes_accessed": False,
            "test_used_for_checkpoint_selection": False,
            "test_used_for_model_or_policy_selection": False,
            "frozen_test_protocol_schema_version": FROZEN_TEST_PROTOCOL_SCHEMA,
            "output_directory": str(frozen_test_protocol["run_directory"]),
            "notes": (
                "One prospectively frozen Critic V2 Development TEST execution after "
                "both V2 gates pass. Validation selects the checkpoint; TEST is not "
                "used for checkpoint, model or policy selection; Evaluation remains closed."
            ),
        }
    )
    return config


def write_config_once(
    config: Mapping[str, Any],
    output_config: Path,
    run_directory: Path,
) -> None:
    """Write the prepared config while preserving single-execution targets."""

    _require(
        not output_config.exists(),
        f"frozen TEST runtime config already exists: {output_config}",
    )
    _require(
        not run_directory.exists(),
        f"frozen TEST run directory already exists: {run_directory}",
    )
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(
        json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-confirmation-config", type=Path, required=True)
    parser.add_argument("--control-protocol", type=Path, required=True)
    parser.add_argument("--confirmation-protocol", type=Path, required=True)
    parser.add_argument("--frozen-test-protocol", type=Path, required=True)
    parser.add_argument("--control-adjudication", type=Path, required=True)
    parser.add_argument("--confirmation-adjudication", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()

    frozen_test_protocol = json.loads(
        args.frozen_test_protocol.read_text(encoding="utf-8")
    )
    config = build_config(
        json.loads(args.selected_confirmation_config.read_text(encoding="utf-8")),
        json.loads(args.control_protocol.read_text(encoding="utf-8")),
        json.loads(args.confirmation_protocol.read_text(encoding="utf-8")),
        frozen_test_protocol,
        json.loads(args.control_adjudication.read_text(encoding="utf-8")),
        json.loads(args.confirmation_adjudication.read_text(encoding="utf-8")),
        gpu=args.gpu,
    )
    output_config = Path(str(frozen_test_protocol["runtime_config"]))
    run_directory = Path(str(frozen_test_protocol["run_directory"]))
    _require(
        config["output_directory"] == str(run_directory),
        "prepared run path differs from the frozen protocol",
    )
    write_config_once(config, output_config, run_directory)
    print(
        json.dumps(
            {
                "status": "CRITIC_V2_SINGLE_FROZEN_TEST_CONFIG_PREPARED_NOT_EXECUTED",
                "config": str(output_config),
                "run_directory": str(run_directory),
                "seed": config["seed"],
                "development_test_will_open_if_executed": True,
                "development_test_opened": False,
                "evaluation_opened": False,
                "test_executed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
