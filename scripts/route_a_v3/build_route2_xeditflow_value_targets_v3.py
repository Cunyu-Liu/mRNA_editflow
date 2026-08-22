#!/usr/bin/env python3
"""Assemble authorized K=8 TRAIN soft-value targets from frozen rollout scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditflow_value_training_v3 import assemble_value_targets_v3


class XEditFlowValueTargetBuilderV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowValueTargetBuilderV3Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                _require(isinstance(row, dict), f"JSONL row is not an object: {path}")
                rows.append(row)
    _require(bool(rows), f"JSONL input is empty: {path}")
    return rows


def build(config: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    _require(config.get("schema_version") == "route_a_v3_route2_xeditflow_value_target_build_config.v1", "unexpected value target build config schema")
    _require(not output_dir.exists(), f"terminal value target output already exists: {output_dir}")
    states = _jsonl(Path(config["train_state_path"]))
    rollouts = _jsonl(Path(config["frozen_rollout_score_path"]))
    payload = assemble_value_targets_v3(
        states,
        rollouts,
        critic_readiness=_json(Path(config["critic_readiness_path"])),
        setflow_confirmation=_json(Path(config["setflow_confirmation_path"])),
        base_flow_training_seed=int(config["base_flow_training_seed"]),
        kappa=float(config["kappa"]),
        temperature=float(config["temperature"]),
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    torch.save(payload, output_dir / "value_targets.pt")
    summary = {key: value for key, value in payload.items() if key != "records"}
    summary["value_target_path"] = str(output_dir / "value_targets.pt")
    summary["raw_outcome_values_persisted"] = False
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(_json(args.config), output_dir=args.output_dir)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
