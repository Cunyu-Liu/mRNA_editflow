#!/usr/bin/env python3
"""Atomically adjudicate the frozen three-seed Critic V3 confirmation cohort."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditcritic_gate_v3 import (
    adjudicate_critic_confirmation_v3,
    paired_source_group_task_macro_bootstrap_v3,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--screen-gate", type=Path, required=True)
    parser.add_argument("--runtime-config-root", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    screen_gate = json.loads(args.screen_gate.read_text(encoding="utf-8"))
    if screen_gate.get("status") != "XEDITCRITIC_V3_SCREEN_PASS":
        raise RuntimeError("Critic V3 screen does not authorize confirmation adjudication")
    selected = str(screen_gate["selected_arm"])
    seeds = [int(seed) for seed in protocol["required_seeds"]]
    payloads = {}
    bootstrap_artifacts = {}
    for seed in seeds:
        runtime = json.loads(
            (args.runtime_config_root / f"seed{seed}.json").read_text(encoding="utf-8")
        )
        if runtime.get("selected_arm") != selected or int(runtime.get("seed", -1)) != seed:
            raise RuntimeError(f"confirmation runtime identity differs: {seed}")
        root = Path(runtime["output_root"])
        candidate_root = root / selected.lower()
        baseline_root = root / "c0"
        candidate_summary = json.loads(
            (candidate_root / "run_summary.json").read_text(encoding="utf-8")
        )
        baseline_summary = json.loads(
            (baseline_root / "run_summary.json").read_text(encoding="utf-8")
        )
        bootstrap = paired_source_group_task_macro_bootstrap_v3(
            _read_jsonl(candidate_root / "final_validation_predictions.jsonl"),
            _read_jsonl(baseline_root / "final_validation_predictions.jsonl"),
            iterations=int(protocol["bootstrap_iterations"]),
            seed=int(protocol["bootstrap_seeds"][str(seed)]),
        )
        payloads[seed] = {
            "candidate_summary": candidate_summary,
            "baseline_summary": baseline_summary,
            "bootstrap": bootstrap,
        }
        bootstrap_artifacts[str(seed)] = bootstrap
    result = adjudicate_critic_confirmation_v3(payloads, selected_arm=selected)
    result["bootstrap_artifacts"] = bootstrap_artifacts
    output = Path(protocol["three_seed_gate_output"])
    if output.exists():
        raise RuntimeError(f"Critic V3 three-seed gate is already terminal: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
