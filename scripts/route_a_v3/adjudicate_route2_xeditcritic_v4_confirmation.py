#!/usr/bin/env python3
"""Atomically adjudicate the exact matched three-seed Critic V4 package."""

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

from core.route2_xeditcritic_gate_v4 import (
    adjudicate_critic_confirmation_v4,
    build_critic_confirmation_seed_payload_v4,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"artifact is not an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    _require(bool(rows), f"prediction artifact is empty: {path}")
    return rows


def load_critic_confirmation_configs_v4(
    manifest: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    _require(
        manifest.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_confirmation_config_manifest.v1"
        and manifest.get("status")
        == "THREE_MATCHED_CONFIRMATION_CONFIGS_PREPARED_NOT_STARTED"
        and manifest.get("required_seeds") == [20260908, 20260909, 20260910]
        and manifest.get("required_run_ids") == ["v4_full", "c0_v4"],
        "Critic V4 confirmation config manifest changed",
    )
    configs = {
        int(config["training_seed"]): config
        for config in map(_read, [Path(value) for value in manifest["config_paths"]])
    }
    _require(set(configs) == {20260908, 20260909, 20260910}, "Critic V4 confirmation configs changed")
    return configs


def collect_critic_confirmation_payloads_v4(
    configs: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    payloads: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for seed in (20260908, 20260909, 20260910):
        _require(seed in configs, f"Critic V4 confirmation config absent: {seed}")
        config = configs[seed]
        summaries = {}
        failed = False
        for run_id in ("v4_full", "c0_v4"):
            directory = Path(config["output_root"]) / run_id
            summary_path = directory / "run_summary.json"
            failure_path = directory / "failure.json"
            _require(
                int(summary_path.exists()) + int(failure_path.exists()) == 1,
                f"Critic V4 confirmation is not exactly terminal: {seed}/{run_id}",
            )
            if failure_path.exists():
                failures.append(
                    {"training_seed": seed, "run_id": run_id, **_read(failure_path)}
                )
                failed = True
            else:
                summaries[run_id] = _read(summary_path)
        if failed:
            continue
        payloads[seed] = build_critic_confirmation_seed_payload_v4(
            summaries["v4_full"],
            summaries["c0_v4"],
            _read_jsonl(Path(summaries["v4_full"]["validation_prediction_path"])),
            _read_jsonl(Path(summaries["c0_v4"]["validation_prediction_path"])),
            seed=seed,
            bootstrap_seed=int(config["bootstrap_seed"]),
        )
    return payloads, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--config-manifest", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    arguments = parser.parse_args()
    protocol = _read(arguments.protocol)
    _require(
        protocol.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_confirmation_protocol.v1"
        and protocol.get("status")
        == "FROZEN_PROSPECTIVE_BEFORE_SCREEN_OR_CONFIRMATION_RESULT",
        "Critic V4 confirmation protocol changed",
    )
    configs = load_critic_confirmation_configs_v4(_read(arguments.config_manifest))
    output = Path(protocol["confirmation_gate_output"])
    _require(not output.exists(), f"Critic V4 confirmation gate exists: {output}")
    payloads, failures = collect_critic_confirmation_payloads_v4(configs)
    if failures:
        _require(
            all(
                int(row.get("development_test_outcome_reads", -1)) == 0
                and int(row.get("new_final_evaluation_outcome_reads", -1)) == 0
                for row in failures
            ),
            "Critic V4 technical failure reports a protected read",
        )
        result = {
            "schema_version": "route_a_v3_route2_xeditcritic_v4_three_seed_gate.v1",
            "status": "XEDITCRITIC_V4_THREE_SEED_NO_GO",
            "reason": "ONE_OR_MORE_FROZEN_CONFIRMATION_RUNS_FAILED_TECHNICALLY",
            "technical_failures": failures,
            "development_test_authorized": False,
            "additional_seed_authorized": False,
            "guidance_authorized": False,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        }
    else:
        result = adjudicate_critic_confirmation_v4(
            next(iter(configs.values())),
            payloads,
            preflight=_read(arguments.preflight),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
