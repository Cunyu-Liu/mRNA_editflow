#!/usr/bin/env python3
"""Compose the four frozen Critic V4 predecessors into guidance readiness."""

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

from core.route2_xeditcritic_gate_v4 import adjudicate_critic_readiness_v4


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"artifact is not an object: {path}")
    return payload


def compose_readiness_v4(
    three_seed: Mapping[str, Any],
    posttest_receipt: Mapping[str, Any],
    refit: Mapping[str, Any],
    loso: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        posttest_receipt.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_posttest_authorization_receipt.v1"
        and posttest_receipt.get("development_test_metrics_in_receipt") is False,
        "Critic V4 outcome-free posttest receipt is absent",
    )
    _require(
        loso.get("status") == "XEDITCRITIC_V4_LOSO_TERMINAL"
        and isinstance(loso.get("loso_gate"), Mapping),
        "Critic V4 LOSO terminal gate is absent",
    )
    return adjudicate_critic_readiness_v4(
        three_seed,
        {
            "status": posttest_receipt.get("frozen_test_gate_status"),
            "all_development_refit_authorized": posttest_receipt.get(
                "all_development_refit_authorized"
            ),
        },
        refit,
        loso["loso_gate"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--three-seed-gate", required=True, type=Path)
    parser.add_argument("--posttest-authorization-receipt", required=True, type=Path)
    parser.add_argument("--refit-manifest", required=True, type=Path)
    parser.add_argument("--loso-adjudication", required=True, type=Path)
    arguments = parser.parse_args()
    protocol = _read(arguments.protocol)
    output = Path(protocol["readiness_output"])
    _require(not output.exists(), f"Critic V4 readiness output exists: {output}")
    result = compose_readiness_v4(
        _read(arguments.three_seed_gate),
        _read(arguments.posttest_authorization_receipt),
        _read(arguments.refit_manifest),
        _read(arguments.loso_adjudication),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
