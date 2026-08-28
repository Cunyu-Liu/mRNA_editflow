#!/usr/bin/env python3
"""Adjudicate only the technically complete three-seed S1 confirmation package."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping


WORKTREE = Path(__file__).resolve().parents[2]
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from core.route2_xeditsetflow_confirmation_s1 import (
    CHECKPOINT_PASSES,
    CONFIRMATION_PROTOCOL_SCHEMA,
    CONFIRMATION_PROTOCOL_STATUS,
    CONFIRMATION_RUN_ID,
    CONFIRMATION_RUNTIME_SCHEMA,
    CONFIRMATION_RUNTIME_STATUS,
    CONFIRMATION_SEEDS,
    OBJECTIVE_IDENTITY,
    OBJECTIVE_WEIGHT,
)
from core.route2_xeditsetflow_gate_s1 import adjudicate_setflow_confirmation_s1


class XEditSetFlowS1ConfirmationAdjudicationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditSetFlowS1ConfirmationAdjudicationError(message)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def protected_reads_zero(payload: Mapping[str, Any], *, label: str) -> None:
    require(
        int(payload.get("development_test_outcome_reads", -1)) == 0
        and int(payload.get("new_final_evaluation_outcome_reads", -1)) == 0,
        f"SetFlow S1 {label} reports a protected outcome read",
    )


def require_unique_summary_terminal(
    summary: Path, failure: Path, *, label: str
) -> None:
    require(
        summary.is_file()
        and not failure.exists()
        and not summary.with_suffix(summary.suffix + ".partial").exists()
        and not failure.with_suffix(failure.suffix + ".partial").exists(),
        f"SetFlow S1 {label} is not uniquely SUMMARY-terminal",
    )


def load_confirmation_configs_s1(
    manifest: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    require(
        manifest.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_s1_confirmation_config_manifest.v1"
        and manifest.get("status")
        == "THREE_S1_CONFIRMATION_CONFIGS_PREPARED_NOT_STARTED"
        and manifest.get("selected_model") == CONFIRMATION_RUN_ID
        and manifest.get("required_seeds") == list(CONFIRMATION_SEEDS)
        and int(manifest.get("training_job_count", -1)) == 3
        and int(manifest.get("single_mode_training_job_count", -1)) == 0
        and int(manifest.get("checkpoint_validation_job_count", -1)) == 12,
        "S1 confirmation config manifest identity changed",
    )
    runner_head = str(manifest.get("confirmation_runner_git_head", ""))
    require(
        re.fullmatch(r"[0-9a-f]{40}", runner_head) is not None,
        "S1 confirmation manifest runner HEAD is invalid",
    )
    protected_reads_zero(manifest, label="config manifest")
    paths = [Path(str(value)) for value in manifest.get("config_paths", [])]
    require(len(paths) == 3, "S1 confirmation config count changed")
    configs: dict[int, dict[str, Any]] = {}
    for path in paths:
        config = read_json(path)
        seed = int(config.get("training_seed", -1))
        require(
            config.get("schema_version") == CONFIRMATION_RUNTIME_SCHEMA
            and config.get("status") == CONFIRMATION_RUNTIME_STATUS
            and config.get("run_stage") == "CONFIRMATION"
            and config.get("selected_model") == CONFIRMATION_RUN_ID
            and config.get("confirmation_runner_git_head") == runner_head
            and seed in CONFIRMATION_SEEDS
            and seed not in configs,
            f"S1 confirmation config identity changed: {path}",
        )
        protected_reads_zero(config, label=f"seed {seed} config")
        configs[seed] = config
    require(
        tuple(configs) == CONFIRMATION_SEEDS,
        "S1 confirmation config seed cohort or order changed",
    )
    return configs


def collect_confirmation_summaries_s1(
    configs: Mapping[int, Mapping[str, Any]],
) -> dict[int, dict[int, dict[str, Any]]]:
    summaries: dict[int, dict[int, dict[str, Any]]] = {}
    for seed in CONFIRMATION_SEEDS:
        require(seed in configs, f"S1 confirmation config is absent: {seed}")
        config = configs[seed]
        training_directory = Path(str(config["output_root"])) / CONFIRMATION_RUN_ID
        training_summary_path = training_directory / "training_summary.json"
        training_failure_path = training_directory / "failure.json"
        require_unique_summary_terminal(
            training_summary_path,
            training_failure_path,
            label=f"seed {seed} training",
        )
        training = read_json(training_summary_path)
        require(
            training.get("schema_version")
            == "route_a_v3_route2_xeditsetflow_v4_s1_training_summary.v1"
            and training.get("status")
            == "TERMINAL_XEDITSETFLOW_V4_S1_TRAINING_COMPLETE_PENDING_VALIDATION"
            and training.get("run_stage") == "CONFIRMATION"
            and training.get("run_id") == CONFIRMATION_RUN_ID
            and training.get("selected_model") == CONFIRMATION_RUN_ID
            and int(training.get("seed", -1)) == seed
            and training.get("objective_identity") == OBJECTIVE_IDENTITY
            and float(
                training.get(
                    "cross_state_candidate_mode_responsibility_weight", -1.0
                )
            )
            == OBJECTIVE_WEIGHT
            and training.get("training_precision") == "BF16"
            and str(training.get("torch_device", "")).startswith("cuda:")
            and "A100" in str(training.get("device_name", ""))
            and training.get("cpu_fallback_used") is False,
            f"S1 seed {seed} training summary identity changed",
        )
        protected_reads_zero(training, label=f"seed {seed} training summary")
        expected_checkpoints = {
            str(checkpoint_pass): str(
                training_directory / f"pass_{checkpoint_pass}.pt"
            )
            for checkpoint_pass in CHECKPOINT_PASSES
        }
        require(
            training.get("saved_checkpoint_paths") == expected_checkpoints,
            f"S1 seed {seed} checkpoint paths changed",
        )
        summaries[seed] = {}
        for checkpoint_pass in CHECKPOINT_PASSES:
            output = (
                Path(str(config["validation_output_root"]))
                / CONFIRMATION_RUN_ID
                / f"pass_{checkpoint_pass}"
            )
            summary_path = output / "validation_summary.json"
            failure_path = output.with_name(output.name + ".failed.json")
            require_unique_summary_terminal(
                summary_path,
                failure_path,
                label=f"seed {seed} pass {checkpoint_pass} Validation",
            )
            summary = read_json(summary_path)
            require(
                summary.get("schema_version")
                == "route_a_v3_route2_xeditsetflow_v4_s1_checkpoint_validation.v1"
                and summary.get("status")
                == "TERMINAL_XEDITSETFLOW_V4_S1_CHECKPOINT_VALIDATION_COMPLETE"
                and summary.get("run_stage") == "CONFIRMATION"
                and summary.get("run_id") == CONFIRMATION_RUN_ID
                and summary.get("selected_model") == CONFIRMATION_RUN_ID
                and int(summary.get("seed", -1)) == seed
                and int(summary.get("checkpoint_pass", -1)) == checkpoint_pass
                and summary.get("objective_identity") == OBJECTIVE_IDENTITY
                and float(
                    summary.get(
                        "cross_state_candidate_mode_responsibility_weight", -1.0
                    )
                )
                == OBJECTIVE_WEIGHT
                and summary.get("checkpoint_path")
                == expected_checkpoints[str(checkpoint_pass)]
                and summary.get("training_summary_path")
                == str(training_summary_path)
                and summary.get("validation_summary_path") == str(summary_path)
                and summary.get("precision") == "BF16"
                and str(summary.get("torch_device", "")).startswith("cuda:")
                and "A100" in str(summary.get("device_name", ""))
                and summary.get("cpu_fallback_used") is False
                and int(summary.get("parameter_update_count", -1)) == 0,
                f"S1 seed {seed} pass {checkpoint_pass} Validation lineage changed",
            )
            protected_reads_zero(
                summary,
                label=f"seed {seed} pass {checkpoint_pass} Validation summary",
            )
            summaries[seed][checkpoint_pass] = summary
    require(
        sum(len(rows) for rows in summaries.values()) == 12,
        "S1 confirmation Validation package is not exactly twelve summaries",
    )
    return summaries


def adjudicate_complete_package_s1(
    protocol: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], Path]:
    require(
        protocol.get("schema_version") == CONFIRMATION_PROTOCOL_SCHEMA
        and protocol.get("status") == CONFIRMATION_PROTOCOL_STATUS
        and protocol.get("selected_model") == CONFIRMATION_RUN_ID
        and protocol.get("required_seeds") == list(CONFIRMATION_SEEDS)
        and protocol.get("additional_seed_authorized") is False,
        "S1 confirmation protocol identity changed",
    )
    protected_reads_zero(protocol, label="confirmation protocol")
    configs = load_confirmation_configs_s1(manifest)
    summaries = collect_confirmation_summaries_s1(configs)
    terminal_f2_path = Path(str(protocol["terminal_f2_validation_summary"]))
    require(
        terminal_f2_path.is_file(),
        f"terminal F2 reference is absent: {terminal_f2_path}",
    )
    terminal_f2 = read_json(terminal_f2_path)
    result = adjudicate_setflow_confirmation_s1(
        configs, summaries, terminal_f2
    )
    require(
        result.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_confirmation_gate.v1"
        and result.get("status")
        in {"XEDITSETFLOW_V4_G0_READY", "XEDITSETFLOW_V4_CONFIRMATION_NO_GO"}
        and result.get("required_seeds") == list(CONFIRMATION_SEEDS)
        and result.get("selected_model") == CONFIRMATION_RUN_ID
        and result.get("objective_identity") == OBJECTIVE_IDENTITY
        and float(
            result.get("cross_state_candidate_mode_responsibility_weight", -1.0)
        )
        == OBJECTIVE_WEIGHT
        and set(result.get("seed_results", {}))
        == {str(seed) for seed in CONFIRMATION_SEEDS},
        "S1 confirmation adjudication is not downstream-compatible V4 readiness",
    )
    protected_reads_zero(result, label="confirmation gate")
    runner_head = str(manifest["confirmation_runner_git_head"])
    output_template = protocol.get("runner_outputs", {}).get(
        "confirmation_gate_output_template"
    )
    require(
        isinstance(output_template, str) and output_template,
        "S1 confirmation gate output template is absent",
    )
    output = Path(output_template.format(runner_git_head=runner_head))
    require(
        all(
            Path(str(config.get("confirmation_gate_output"))) == output
            for config in configs.values()
        ),
        "S1 confirmation configs disagree on gate output",
    )
    return result, output


def write_terminal_gate_s1(path: Path, result: Mapping[str, Any]) -> None:
    require(not path.exists(), f"terminal S1 confirmation gate exists: {path}")
    failure = path.with_name(path.name + ".failed.json")
    partial = path.with_suffix(path.suffix + ".partial")
    require(
        not failure.exists()
        and not failure.with_suffix(failure.suffix + ".partial").exists()
        and not partial.exists(),
        "S1 confirmation gate failure or partial already exists",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(
        json.dumps(dict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--config-manifest", required=True, type=Path)
    arguments = parser.parse_args()
    protocol = read_json(arguments.protocol)
    manifest = read_json(arguments.config_manifest)
    result, output = adjudicate_complete_package_s1(protocol, manifest)
    write_terminal_gate_s1(output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
