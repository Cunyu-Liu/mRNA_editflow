#!/usr/bin/env python3
"""Build the outcome-isolated TRAIN/VALIDATION projection for V3 models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_development_projection_v3 import (  # noqa: E402
    MODELING_SPLITS,
    DevelopmentProjectionError,
    build_development_projection,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DevelopmentProjectionError(message)


def run(config: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        config.get("schema_version")
        == "route_a_v3_route2_development_projection_config.v3",
        "unexpected projection config schema",
    )
    _require(config.get("included_splits") == list(MODELING_SPLITS), "projection splits must be TRAIN and VALIDATION")
    _require(config.get("development_test_outcomes_accessed") is False, "config declares TEST outcome access")
    _require(config.get("evaluation_outcomes_accessed") is False, "config declares Evaluation outcome access")
    return build_development_projection(
        manifest_path=Path(config["development_manifest"]),
        canonical_paths=[Path(path) for path in config["canonical_paths"]],
        endpoint_descriptor_path=Path(config["endpoint_descriptor_registry"]),
        output_directory=Path(config["output_directory"]),
        included_splits=tuple(config["included_splits"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(run(config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
