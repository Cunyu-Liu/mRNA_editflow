#!/usr/bin/env python3
"""Adjudicate the frozen F2/F3 XEditSetFlow V3 screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditsetflow_gate_v3 import adjudicate_setflow_screen_v3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = Path(config["screen_gate_output_path"])
    if output.exists():
        raise RuntimeError(f"terminal SetFlow screen gate already exists: {output}")
    root = Path(config["output_root"])
    validation_root = Path(config["validation_output_root"])
    f0 = json.loads(Path(config["frozen_f0_common_nll_output_path"]).read_text(encoding="utf-8"))
    training = {
        arm: json.loads((root / arm / "training_summary.json").read_text(encoding="utf-8"))
        for arm in ("f1", "f2", "f3")
    }
    validation = {
        arm: json.loads((validation_root / arm / "validation_summary.json").read_text(encoding="utf-8"))
        for arm in ("f1", "f2", "f3")
    }
    result = adjudicate_setflow_screen_v3(f0, training, validation)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
