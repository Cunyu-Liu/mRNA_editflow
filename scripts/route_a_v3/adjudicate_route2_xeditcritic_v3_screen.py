#!/usr/bin/env python3
"""Atomically adjudicate the frozen Critic V3 Development Validation screen."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditcritic_gate_v3 import adjudicate_critic_screen_v3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    output_path = Path(config["screen_gate_output"])
    if output_path.exists():
        raise RuntimeError(f"screen gate is already terminal: {output_path}")
    summaries = {
        run_id: json.loads((Path(config["output_root"]) / run_id / "run_summary.json").read_text(encoding="utf-8"))
        for run_id in config["required_screen_run_ids"]
    }
    result = adjudicate_critic_screen_v3(
        summaries, expected_seed=int(config["screen_seed"])
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, output_path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
