#!/usr/bin/env python3
"""Compose all four frozen Critic V3 predecessors into guidance readiness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditcritic_gate_v3 import adjudicate_critic_readiness_v3


class CriticReadinessV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CriticReadinessV3Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def compose_readiness_v3(
    three_seed: dict[str, Any],
    atomic_test: dict[str, Any],
    refit: dict[str, Any],
    loso: dict[str, Any],
) -> dict[str, Any]:
    _require(
        atomic_test.get("status") == "ATOMIC_FROZEN_DEVELOPMENT_TEST_TERMINAL"
        and isinstance(atomic_test.get("frozen_test_gate"), dict),
        "atomic frozen TEST gate is absent",
    )
    _require(
        loso.get("status") == "XEDITCRITIC_V3_LOSO_TERMINAL"
        and isinstance(loso.get("loso_gate"), dict),
        "LOSO terminal gate is absent",
    )
    result = adjudicate_critic_readiness_v3(
        three_seed,
        atomic_test["frozen_test_gate"],
        refit,
        loso["loso_gate"],
    )
    result.update(
        {
            "development_test_access_event_count": 1,
            "general_test_projection_persisted": False,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--three-seed-gate", type=Path, required=True)
    parser.add_argument("--atomic-frozen-test", type=Path, required=True)
    parser.add_argument("--refit-manifest", type=Path, required=True)
    parser.add_argument("--loso-adjudication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"Critic readiness output exists: {args.output}")
    result = compose_readiness_v3(
        _json(args.three_seed_gate),
        _json(args.atomic_frozen_test),
        _json(args.refit_manifest),
        _json(args.loso_adjudication),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
