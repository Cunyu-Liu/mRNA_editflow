#!/usr/bin/env python3
"""Atomically adjudicate the exact three-seed SetFlow V4 confirmation package."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditsetflow_gate_v4 import (
    adjudicate_setflow_confirmation_v4,
    confirmation_technical_failure_gate_v4,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"artifact is not an object: {path}")
    return payload


def load_confirmation_configs_v4(
    manifest: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    _require(
        manifest.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_confirmation_config_manifest.v1"
        and manifest.get("status")
        == "THREE_CONFIRMATION_CONFIGS_PREPARED_NOT_STARTED"
        and manifest.get("required_seeds") == [20260912, 20260913, 20260914]
        and manifest.get("selected_model") == "v4_full",
        "SetFlow V4 confirmation config manifest changed",
    )
    paths = [Path(value) for value in manifest.get("config_paths", [])]
    _require(len(paths) == 3, "SetFlow V4 confirmation config path count changed")
    configs = {int(config["training_seed"]): config for config in map(_read, paths)}
    _require(
        set(configs) == {20260912, 20260913, 20260914},
        "SetFlow V4 confirmation config seeds changed",
    )
    return configs


def collect_confirmation_terminal_artifacts_v4(
    configs: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[int, dict[int, dict[str, Any]]], list[dict[str, Any]]]:
    summaries: dict[int, dict[int, dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    for seed in (20260912, 20260913, 20260914):
        _require(seed in configs, f"SetFlow V4 confirmation config absent: {seed}")
        config = configs[seed]
        training_directory = Path(config["output_root"]) / "v4_full"
        training_summary = training_directory / "training_summary.json"
        training_failure = training_directory / "failure.json"
        _require(
            int(training_summary.exists()) + int(training_failure.exists()) == 1,
            f"SetFlow V4 confirmation training is not exactly terminal: {seed}",
        )
        if training_failure.exists():
            failures.append(
                {
                    "stage": "TRAINING",
                    "training_seed": seed,
                    **_read(training_failure),
                }
            )
            continue
        trained = _read(training_summary)
        _require(
            trained.get("status")
            == "TERMINAL_XEDITSETFLOW_V4_TRAINING_COMPLETE_PENDING_VALIDATION"
            and trained.get("run_stage") == "CONFIRMATION"
            and int(trained.get("seed", -1)) == seed,
            f"SetFlow V4 confirmation training summary identity changed: {seed}",
        )
        summaries[seed] = {}
        for checkpoint_pass in (4, 6, 8, 10):
            directory = (
                Path(config["validation_output_root"])
                / "v4_full"
                / f"pass_{checkpoint_pass}"
            )
            summary_path = directory / "validation_summary.json"
            failure_path = directory.with_name(directory.name + ".failed.json")
            _require(
                int(summary_path.exists()) + int(failure_path.exists()) == 1,
                f"SetFlow V4 confirmation validation is not exactly terminal: {seed}/pass{checkpoint_pass}",
            )
            if failure_path.exists():
                failures.append(
                    {
                        "stage": "CHECKPOINT_VALIDATION",
                        "training_seed": seed,
                        "checkpoint_pass": checkpoint_pass,
                        **_read(failure_path),
                    }
                )
            else:
                summaries[seed][checkpoint_pass] = _read(summary_path)
    return summaries, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--config-manifest", required=True, type=Path)
    arguments = parser.parse_args()
    protocol = _read(arguments.protocol)
    _require(
        protocol.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_confirmation_protocol.v1"
        and protocol.get("status")
        == "FROZEN_PROSPECTIVE_BEFORE_SCREEN_OR_CONFIRMATION_RESULT"
        and protocol.get("required_seeds") == [20260912, 20260913, 20260914]
        and protocol.get("additional_seed_authorized") is False,
        "SetFlow V4 confirmation protocol changed",
    )
    manifest = _read(arguments.config_manifest)
    configs = load_confirmation_configs_v4(manifest)
    output = Path(protocol["confirmation_gate_output"])
    _require(not output.exists(), f"terminal SetFlow V4 confirmation gate exists: {output}")
    summaries, failures = collect_confirmation_terminal_artifacts_v4(configs)
    if failures:
        result = confirmation_technical_failure_gate_v4(failures)
    else:
        terminal_f2 = _read(Path(protocol["terminal_f2_validation_summary"]))
        result = adjudicate_setflow_confirmation_v4(
            configs, summaries, terminal_f2
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
