#!/usr/bin/env python3
"""Adopt the terminal V3 SetFlow source cache without rebuilding it."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_source_token_cache_v3 import (
    load_source_token_cache_v3,
    require_source_token_cache_identity_v3,
)
from scripts.route_a_v3.authorize_route2_xedit_v4_screen_stages import (
    require_cache_launch_authorization_v4,
)


class SetFlowSourceCacheAdoptionV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SetFlowSourceCacheAdoptionV4Error(message)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_legacy_summary(
    summary: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    _require(
        summary.get("schema_version")
        == "route_a_v3_route2_setflow_source_token_cache_summary.v3"
        and summary.get("status") == "XEDITSETFLOW_V3_SOURCE_TOKEN_CACHE_COMPLETE",
        "legacy SetFlow source cache summary is not terminal",
    )
    expected = {
        "projection_record_count": "expected_projection_record_count",
        "eligible_record_count": "expected_eligible_record_count",
        "unique_source_count": "expected_unique_source_count",
        "unique_source_token_count": "expected_token_count",
        "maximum_source_length": "expected_maximum_source_length",
        "embedding_width": "expected_embedding_width",
    }
    for summary_key, config_key in expected.items():
        _require(
            int(summary.get(summary_key, -1)) == int(config[config_key]),
            f"legacy SetFlow source cache summary changed: {summary_key}",
        )
    _require(
        str(summary.get("model_id")) == str(config["expected_model_id"]),
        "legacy SetFlow source cache mRNABERT revision changed",
    )
    _require(
        Path(str(summary.get("output_path", "")))
        == Path(str(config["legacy_cache_path"])),
        "legacy SetFlow source cache summary points to another payload",
    )
    _require(
        int(summary.get("raw_sequence_payload_written", -1)) == 0
        and int(summary.get("outcome_value_access_count", -1)) == 0
        and int(summary.get("development_test_record_count", -1)) == 0
        and summary.get("development_test_outcomes_accessed") is False
        and int(summary.get("evaluation_record_count", -1)) == 0
        and summary.get("evaluation_outcomes_accessed") is False,
        "legacy SetFlow source cache summary is not outcome isolated",
    )


def adopt(
    config: Mapping[str, Any], authorization: Mapping[str, Any]
) -> dict[str, Any]:
    current_head = _git_head()
    require_cache_launch_authorization_v4(
        "setflow", authorization, current_git_head=current_head
    )
    _require(
        config.get("legacy_artifact_policy") == "READ_ONLY_NO_REBUILD_NO_OVERWRITE",
        "SetFlow V3 terminal cache is not frozen read-only",
    )
    _require(
        int(config.get("encoder_forward_count", -1)) == 0
        and int(config.get("parameter_update_count", -1)) == 0,
        "SetFlow cache adoption may not encode or update parameters",
    )
    _require(
        int(config.get("development_test_outcome_reads", -1)) == 0
        and int(config.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "SetFlow cache adoption config reports a protected outcome read",
    )
    legacy_cache_path = Path(config["legacy_cache_path"])
    legacy_summary_path = Path(config["legacy_summary_path"])
    receipt_path = Path(config["adoption_receipt_path"])
    partial = receipt_path.with_suffix(receipt_path.suffix + ".partial")
    _require(legacy_cache_path.is_file(), "terminal V3 SetFlow source cache is absent")
    _require(legacy_summary_path.is_file(), "terminal V3 SetFlow source cache summary is absent")
    _require(not receipt_path.exists(), "SetFlow V4 cache adoption receipt already exists")
    _require(not partial.exists(), "partial SetFlow V4 cache adoption receipt exists")
    legacy_summary = _read(legacy_summary_path)
    _require_legacy_summary(legacy_summary, config)
    cache_payload = load_source_token_cache_v3(legacy_cache_path)
    identity = require_source_token_cache_identity_v3(
        cache_payload,
        expected_model_id=str(config["expected_model_id"]),
        expected_record_count=int(config["expected_eligible_record_count"]),
        expected_unique_source_count=int(config["expected_unique_source_count"]),
        expected_token_count=int(config["expected_token_count"]),
        expected_maximum_source_length=int(config["expected_maximum_source_length"]),
        expected_embedding_width=int(config["expected_embedding_width"]),
    )
    result = {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_source_cache_adoption_receipt.v1",
        "status": "XEDITSETFLOW_V4_SOURCE_CACHE_ADOPTED_READ_ONLY",
        "git_head": current_head,
        "cache_launch_authorization_status": authorization["status"],
        "legacy_cache_path": str(legacy_cache_path),
        "legacy_summary_path": str(legacy_summary_path),
        "legacy_summary_schema_version": legacy_summary["schema_version"],
        "legacy_summary_status": legacy_summary["status"],
        "legacy_artifact_policy": "READ_ONLY_NO_REBUILD_NO_OVERWRITE",
        "source_token_cache_identity": identity,
        "encoder_forward_count": 0,
        "parameter_update_count": 0,
        "cpu_fallback_used": False,
        "identity_validation_map_location": "CPU_READ_ONLY",
        "legacy_payload_modified": False,
        "legacy_summary_modified": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, receipt_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            adopt(_read(arguments.config), _read(arguments.authorization)),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
