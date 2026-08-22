#!/usr/bin/env python3
"""Re-execute only the frozen genetic search to obtain missing A100 timing evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
SEARCH_SCRIPT = REPO_ROOT / "scripts" / "route_a_v3" / "run_route2_search_generation_baselines_v1.py"


class XEditFlowStrongestTimingV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowStrongestTimingV3Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def validate_strongest_timing_config_v3(
    config: Mapping[str, Any],
    strongest: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> None:
    _require(
        config.get("schema_version")
        == "route_a_v3_route2_xeditflow_strongest_timing_config.v1",
        "unexpected strongest timing config schema",
    )
    _require(
        strongest.get("status")
        == "DEVELOPMENT_STRONGEST_GENERATION_BASELINE_FROZEN_INDEPENDENT_EVALUATOR_ONLY"
        and strongest.get("strongest_generation_baseline_id") == "genetic"
        and strongest.get("evaluation_outcomes_accessed") is False,
        "strongest timing baseline is not the frozen genetic method",
    )
    _require(
        selection.get("selection_pool") == "DEVELOPMENT_MEASURED_NEIGHBORHOOD"
        and selection.get("evaluation_release_state") == "CLOSED",
        "strongest timing selection boundary differs",
    )
    _require(
        config.get("method_id") == "genetic"
        and int(config.get("seed", -1)) == 20260816,
        "strongest timing method or decoder seed differs",
    )
    _require(
        int(config.get("critic_forward_budget_per_source", -1))
        == int(strongest.get("critic_forward_budget_per_source", -2))
        and int(strongest.get("forward_equivalent_budget_per_source", -1)) == 320,
        "strongest timing matched-compute budget differs",
    )
    _require(
        str(config.get("guiding_checkpoint_path"))
        == str(strongest.get("guiding_checkpoint_path")),
        "strongest timing guiding checkpoint differs",
    )
    _require(
        (
            int(config.get("beam_width", -1)),
            int(config.get("genetic_population_size", -1)),
            int(config.get("oversample_factor", -1)),
            int(config.get("exhaustive_space_limit", -1)),
        )
        == (16, 32, 8, 4096),
        "strongest timing frozen search hyperparameters differ",
    )
    gpu = int(config.get("physical_gpu_index", -1))
    _require(
        gpu in set(range(6)) and config.get("device") == f"cuda:{gpu}",
        "strongest timing GPU is outside physical GPU 0-5",
    )
    _require(
        str(config.get("output_dir", "")).startswith(
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"
        ),
        "strongest timing output left Route 2 /mnt",
    )
    _require(
        config.get("timing_only_no_baseline_reselection") is True
        and config.get("development_test_outcomes_accessed") is False
        and config.get("new_final_evaluation_outcomes_accessed") is False,
        "strongest timing run changed its evidence boundary",
    )


def strongest_timing_command_v3(config: Mapping[str, Any], output_path: Path) -> list[str]:
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


def run(config: Mapping[str, Any]) -> dict[str, Any]:
    strongest = _json(Path(str(config["strongest_generation_baseline_path"])))
    selection = _json(Path(str(config["baseline_selection_input_path"])))
    validate_strongest_timing_config_v3(config, strongest, selection)
    output_dir = Path(str(config["output_dir"]))
    _require(not output_dir.exists(), f"strongest timing output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    candidate_path = output_dir / "timed_genetic_candidates.private.jsonl"
    subprocess.run(
        strongest_timing_command_v3(config, candidate_path),
        cwd=REPO_ROOT,
        check=True,
    )
    rows = [
        json.loads(line)
        for line in candidate_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_keys = list(dict.fromkeys(str(row.get("source_key")) for row in rows))
    _require(len(source_keys) == 891, "strongest timing source count differs")
    _require(
        all(row.get("method_id") == "genetic" for row in rows),
        "strongest timing generated method differs",
    )
    _require(
        sum(float(row.get("source_equal_wall_time_seconds", 0.0)) > 0.0 for row in rows)
        == 891,
        "strongest timing per-source evidence is incomplete",
    )
    result = {
        "schema_version": "route_a_v3_route2_xeditflow_strongest_timing.v1",
        "status": "XEDITFLOW_V3_STRONGEST_BASELINE_A100_TIMING_COMPLETE",
        "method_id": "strongest_matched_baseline",
        "underlying_method_id": "genetic",
        "source_count": 891,
        "candidate_path": str(candidate_path),
        "frozen_baseline_reselected": False,
        "timing_only_no_baseline_reselection": True,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(_json(args.config)), sort_keys=True))


if __name__ == "__main__":
    main()
