#!/usr/bin/env python3
"""Atomically adjudicate the exact three SetFlow V3 confirmation seeds."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditsetflow_gate_v3 import adjudicate_setflow_confirmation_v3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--screen-gate", type=Path, required=True)
    parser.add_argument("--runtime-config-root", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    screen_gate = json.loads(args.screen_gate.read_text(encoding="utf-8"))
    if screen_gate.get("status") != "XEDITSETFLOW_V3_SCREEN_PASS":
        raise RuntimeError("SetFlow V3 screen does not authorize confirmation adjudication")
    selected = str(screen_gate["selected_arm"])
    training = {}
    validation = {}
    for seed in [int(value) for value in protocol["required_seeds"]]:
        runtime = json.loads(
            (args.runtime_config_root / f"seed{seed}.json").read_text(encoding="utf-8")
        )
        if runtime.get("selected_arm") != selected or int(runtime.get("seed", -1)) != seed:
            raise RuntimeError(f"SetFlow confirmation runtime identity differs: {seed}")
        root = Path(runtime["output_root"])
        validation_root = Path(runtime["validation_output_root"])
        training[seed] = json.loads(
            (root / selected / "training_summary.json").read_text(encoding="utf-8")
        )
        validation[seed] = json.loads(
            (validation_root / selected / "validation_summary.json").read_text(encoding="utf-8")
        )
    result = adjudicate_setflow_confirmation_v3(
        training, validation, selected_arm=selected
    )
    output = Path(protocol["confirmation_gate_output"])
    if output.exists():
        raise RuntimeError(f"SetFlow V3 confirmation gate is already terminal: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
