#!/usr/bin/env python3
"""Authorize V4 guidance only after both independent readiness gates pass."""

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

from core.route2_xeditflow_gate_v4 import authorize_xeditflow_guidance_v4


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"artifact is not an object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    arguments = parser.parse_args()
    protocol = _read(arguments.protocol)
    _require(
        protocol.get("schema_version")
        == "route_a_v3_route2_xeditflow_v4_guidance_protocol.v1",
        "unexpected V4 guidance protocol",
    )
    output = Path(protocol["authorization_output"])
    _require(not output.exists(), f"V4 guidance authorization exists: {output}")
    result = authorize_xeditflow_guidance_v4(
        _read(Path(protocol["critic_readiness_path"])),
        _read(Path(protocol["setflow_confirmation_path"])),
    )
    _require(
        result["guidance_authorized"] is True,
        "V4 guidance remains blocked by one or both frozen readiness gates",
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
