#!/usr/bin/env python3
"""Time only the pre-V4 frozen strongest genetic baseline on the A100 cohort."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
SEARCH_SCRIPT = (
    REPO_ROOT / "scripts" / "route_a_v3" / "run_route2_search_generation_baselines_v1.py"
)


class XEditFlowStrongestTimingV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowStrongestTimingV4Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _write_new_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), f"failure evidence already exists: {path}")
    partial = path.with_suffix(path.suffix + ".partial")
    _require(not partial.exists(), f"partial failure evidence exists: {partial}")
    partial.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def bridge_strongest_timing_failure_v4(
    *,
    producer_failure_path: Path,
    declared_failure_path: Path,
    error: Exception,
) -> None:
    producer_evidence: dict[str, Any] | None = None
    producer_read_error: str | None = None
    if producer_failure_path.exists():
        try:
            producer_evidence = _json(producer_failure_path)
        except Exception as read_error:
            producer_read_error = (
                f"{type(read_error).__name__}: {read_error}"
            )
    payload: dict[str, Any] = {
        "schema_version": (
            "route_a_v3_route2_xeditflow_v4_final_job_failure.v1"
        ),
        "status": "TERMINAL_STRONGEST_TIMING_PRODUCER_FAILURE",
        "failure_stage": "STRONGEST_TIMING_CUDA_OR_EXECUTION",
        "job_key": "strongest_timing",
        "exception_type": type(error).__name__,
        "error": str(error),
        "producer_failure_path": str(producer_failure_path),
        "producer_failure_evidence_present": producer_evidence is not None,
        "job_process_started": True,
        "cpu_fallback_used": bool(
            producer_evidence.get("cpu_fallback_used", False)
        )
        if producer_evidence is not None
        else False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcome_reads": 0,
    }
    if producer_evidence is not None:
        payload["producer_failure_evidence"] = producer_evidence
    if producer_read_error is not None:
        payload["producer_failure_read_error"] = producer_read_error
    _write_new_atomic(declared_failure_path, payload)


def validate_strongest_timing_config_v4(
    config: Mapping[str, Any],
    strongest: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> None:
    _require(
        config.get("schema_version")
        == "route_a_v3_route2_xeditflow_strongest_timing_config.v4",
        "unexpected V4 strongest timing config schema",
    )
    _require(
        strongest.get("status")
        == "DEVELOPMENT_STRONGEST_GENERATION_BASELINE_FROZEN_INDEPENDENT_EVALUATOR_ONLY"
        and strongest.get("strongest_generation_baseline_id") == "genetic"
        and strongest.get("evaluation_outcomes_accessed") is False
        and int(strongest.get("forward_equivalent_budget_per_source", -1)) == 320,
        "V4 strongest timing baseline is not frozen",
    )
    _require(
        selection.get("selection_pool") == "DEVELOPMENT_MEASURED_NEIGHBORHOOD"
        and selection.get("evaluation_release_state") == "CLOSED",
        "V4 strongest timing selection boundary differs",
    )
    _require(
        config.get("method_id") == "genetic"
        and int(config.get("seed", -1)) == 20260816
        and int(config.get("critic_forward_budget_per_source", -1))
        == int(strongest.get("critic_forward_budget_per_source", -2))
        and str(config.get("guiding_checkpoint_path"))
        == str(strongest.get("guiding_checkpoint_path")),
        "V4 strongest timing method, seed, budget or checkpoint differs",
    )
    _require(
        (
            int(config.get("beam_width", -1)),
            int(config.get("genetic_population_size", -1)),
            int(config.get("oversample_factor", -1)),
            int(config.get("exhaustive_space_limit", -1)),
        )
        == (16, 32, 8, 4096),
        "V4 strongest timing search hyperparameters differ",
    )
    gpu = int(config.get("physical_gpu_index", -1))
    _require(
        gpu in range(6) and config.get("device") == f"cuda:{gpu}",
        "V4 strongest timing GPU provenance differs",
    )
    _require(
        str(config.get("output_dir", "")).startswith(
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"
        )
        and config.get("timing_only_no_baseline_reselection") is True
        and config.get("development_test_outcomes_accessed_after_atomic_test")
        is False
        and config.get("new_final_evaluation_outcomes_accessed") is False,
        "V4 strongest timing output or protected-input policy differs",
    )


def strongest_timing_command_v4(
    config: Mapping[str, Any], output_path: Path
) -> list[str]:
    return [
        sys.executable,
        str(SEARCH_SCRIPT),
        "--source-manifest",
        str(config["source_manifest_path"]),
        "--checkpoint",
        str(config["guiding_checkpoint_path"]),
        "--device",
        str(config["device"]),
        "--physical-gpu-index",
        str(config["physical_gpu_index"]),
        "--method",
        "genetic",
        "--max-critic-forwards",
        str(config["critic_forward_budget_per_source"]),
        "--beam-width",
        str(config["beam_width"]),
        "--genetic-population-size",
        str(config["genetic_population_size"]),
        "--oversample-factor",
        str(config["oversample_factor"]),
        "--exhaustive-space-limit",
        str(config["exhaustive_space_limit"]),
        "--seed",
        str(config["seed"]),
        "--output",
        str(output_path),
    ]


def run(
    config: Mapping[str, Any], *, failure_path: Path
) -> dict[str, Any]:
    strongest = _json(Path(str(config["strongest_generation_baseline_path"])))
    selection = _json(Path(str(config["baseline_selection_input_path"])))
    validate_strongest_timing_config_v4(config, strongest, selection)
    output_dir = Path(str(config["output_dir"]))
    _require(
        not output_dir.exists(), f"V4 strongest timing output exists: {output_dir}"
    )
    _require(
        not failure_path.exists(),
        f"V4 strongest timing declared failure exists: {failure_path}",
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    candidate_path = output_dir / "timed_genetic_candidates.private.jsonl"
    try:
        subprocess.run(
            strongest_timing_command_v4(config, candidate_path),
            cwd=REPO_ROOT,
            check=True,
        )
    except Exception as error:
        bridge_strongest_timing_failure_v4(
            producer_failure_path=candidate_path.with_suffix(
                candidate_path.suffix + ".failed.json"
            ),
            declared_failure_path=failure_path,
            error=error,
        )
        raise
    rows = [
        json.loads(line)
        for line in candidate_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_keys = list(dict.fromkeys(str(row.get("source_key")) for row in rows))
    _require(
        len(source_keys) == 891
        and all(row.get("method_id") == "genetic" for row in rows)
        and sum(
            float(row.get("source_equal_wall_time_seconds", 0.0)) > 0.0
            for row in rows
        )
        == 891,
        "V4 strongest timing terminal evidence is incomplete",
    )
    result = {
        "schema_version": "route_a_v3_route2_xeditflow_strongest_timing.v4",
        "status": "XEDITFLOW_V4_STRONGEST_BASELINE_A100_TIMING_COMPLETE",
        "method_id": "strongest_matched_baseline",
        "underlying_method_id": "genetic",
        "source_count": 891,
        "candidate_path": str(candidate_path),
        "frozen_baseline_reselected_for_v4": False,
        "timing_only_no_baseline_reselection": True,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--failure-path", required=True, type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(_json(arguments.config), failure_path=arguments.failure_path),
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
