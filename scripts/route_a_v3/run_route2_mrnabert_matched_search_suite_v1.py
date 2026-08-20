#!/usr/bin/env python3
"""Generate six search baselines under each guided source's exact critic budget."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SEARCH_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "route_a_v3"
    / "run_route2_search_generation_baselines_v1.py"
)
EXPECTED_METHODS = (
    "random_legal",
    "greedy",
    "beam",
    "genetic",
    "local_search",
    "generate_then_rerank",
)
MATCHED_CONFIG_SCHEMA = (
    "route_a_v3_route2_mrnabert_critic_v2_matched_search_protocol.v1"
)
READINESS_INPUT_SCHEMA = (
    "route_a_v3_route2_mrnabert_critic_v2_guidance_readiness_input.v1"
)
READINESS_ADJUDICATION_SCHEMA = (
    "route_a_v3_route2_mrnabert_critic_v2_guidance_readiness_adjudication.v1"
)
GUIDED_SUMMARY_SCHEMA = (
    "route_a_v3_route2_mrnabert_critic_v2_guided_xeditflow_development.v1"
)
ROUTE2_ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
EXPECTED_READINESS_INPUT = (
    ROUTE2_ROOT / "comparisons/mrnabert_critic_v2_guidance_readiness_input_v1.json"
)
EXPECTED_READINESS_ADJUDICATION = (
    ROUTE2_ROOT
    / "comparisons/mrnabert_critic_v2_guidance_readiness_adjudication_v1.json"
)
EXPECTED_CRITIC_CHECKPOINT = (
    ROUTE2_ROOT
    / "runs/mrnabert_critic_v2/all_development_refit_v1/seed20260823/delta_predictor_checkpoint.pt"
)
EXPECTED_GUIDED_ROOT = (
    ROUTE2_ROOT / "runs/mrnabert_critic_v2/guided_xeditflow_development_v1"
)


class MatchedSearchSuiteError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatchedSearchSuiteError(message)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    _require(path.is_file(), f"{label} is absent: {path}")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    _require(all(isinstance(row, dict) for row in rows), f"{label} contains a non-object row")
    return rows


def validate_config_boundary(config: Mapping[str, Any]) -> None:
    _require(
        config.get("schema_version") == MATCHED_CONFIG_SCHEMA
        and config.get("status") == "WAITING_FOR_CRITIC_V2_GUIDED_XEDITFLOW",
        "historical or unexpected matched-search config is not authorized",
    )
    _require(
        int(config.get("seed", -1)) == 20260825
        and Path(str(config.get("readiness_input_path")))
        == EXPECTED_READINESS_INPUT
        and Path(str(config.get("readiness_adjudication_path")))
        == EXPECTED_READINESS_ADJUDICATION
        and Path(str(config.get("critic_checkpoint_path")))
        == EXPECTED_CRITIC_CHECKPOINT
        and Path(str(config.get("guided_summary_path")))
        == EXPECTED_GUIDED_ROOT / "guided_summary.json"
        and Path(str(config.get("guided_compute_by_source_path")))
        == EXPECTED_GUIDED_ROOT / "guided_compute_by_source.jsonl",
        "Critic V2 matched-search artifact binding differs",
    )


def validate_inputs(
    config: Mapping[str, Any],
    readiness_input: Mapping[str, Any],
    readiness_adjudication: Mapping[str, Any],
    independent_evaluator_adjudication: Mapping[str, Any],
    guided_summary: Mapping[str, Any],
    compute_rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    validate_config_boundary(config)
    _require(
        config.get("schema_version") == MATCHED_CONFIG_SCHEMA,
        "matched-search schema differs",
    )
    _require(
        tuple(config.get("required_method_ids", ())) == EXPECTED_METHODS,
        "matched-search method set or order differs",
    )
    _require(
        config.get("critic_budget_rule")
        == "GUIDED_TOTAL_FORWARD_EQUIVALENTS_AS_SEARCH_CRITIC_CAP_PER_SOURCE",
        "matched-search critic budget rule differs",
    )
    _require(
        config.get("candidate_generation_only") is True
        and config.get("strongest_method_selection_in_this_suite") is False,
        "matched search incorrectly enables scientific method selection",
    )
    _require(
        config.get("evaluation_outcomes_accessed") is False,
        "Evaluation cannot enter Development search generation",
    )
    _require(
        independent_evaluator_adjudication.get("schema_version")
        == "route_a_v3_route2_independent_generation_evaluator_adjudication.v1"
        and independent_evaluator_adjudication.get("status") in {
            "INDEPENDENT_GENERATION_EVALUATOR_QUALIFIED",
            "INDEPENDENT_GENERATION_EVALUATOR_NO_GO",
        }
        and independent_evaluator_adjudication.get("development_test_outcomes_accessed")
        is False
        and independent_evaluator_adjudication.get("evaluation_outcomes_accessed")
        is False,
        "independent evaluator was not frozen before candidate generation",
    )
    _require(
        readiness_input.get("schema_version")
        == READINESS_INPUT_SCHEMA
        and readiness_input.get("guided_generation_executed") is False
        and readiness_input.get("evaluation_opened_by_readiness_builder") is False
        and readiness_adjudication.get("schema_version")
        == READINESS_ADJUDICATION_SCHEMA
        and readiness_adjudication.get("guided_unlocked") is True
        and readiness_adjudication.get("critic_status")
        == "CRITIC_READY_FOR_GUIDANCE"
        and readiness_adjudication.get("flow_status") == "FLOW_G0_READY"
        and readiness_adjudication.get("guided_generation_status")
        == "GUIDED_XEDITFLOW_DEVELOPMENT_ALLOWED"
        and readiness_adjudication.get("guided_generation_executed") is False
        and readiness_adjudication.get("evaluation_opened") is False,
        "critic and Flow readiness are not closed",
    )
    _require(
        Path(readiness_input["critic"]["refit_checkpoint"])
        == Path(str(config["critic_checkpoint_path"])),
        "matched search critic differs from guidance readiness",
    )
    _require(
        guided_summary.get("schema_version") == GUIDED_SUMMARY_SCHEMA
        and guided_summary.get("status") == "GUIDED_XEDITFLOW_DEVELOPMENT_COMPLETE"
        and guided_summary.get("matched_search_budget_rule")
        == config.get("critic_budget_rule")
        and guided_summary.get("evaluation_outcomes_read") == 0
        and guided_summary.get("generated_candidates_grant_canonical_credit") is False
        and guided_summary.get("biological_optimization_established") is False,
        "guided run did not publish the frozen matching budget rule",
    )
    _require(
        Path(str(guided_summary.get("per_source_compute_path")))
        == Path(str(config["guided_compute_by_source_path"])),
        "guided per-source compute path differs from matched-search input",
    )
    source_keys = [str(row["source_key"]) for row in source_rows]
    _require(
        bool(source_keys) and len(source_keys) == len(set(source_keys)),
        "source manifest is empty or contains duplicate keys",
    )
    budgets: dict[str, int] = {}
    for row in compute_rows:
        key = str(row["source_key"])
        value = row.get("matched_search_critic_forward_budget")
        _require(key not in budgets, f"guided source compute is duplicated: {key}")
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"guided source critic budget is invalid: {key}",
        )
        budgets[key] = value
    _require(
        set(budgets) == set(source_keys),
        "guided per-source compute does not exactly cover the source manifest",
    )
    return budgets


def build_commands(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    output_directory = Path(str(config["output_directory"]))
    commands = []
    for method in EXPECTED_METHODS:
        commands.append(
            {
                "method_id": method,
                "output_path": output_directory / f"{method}.private.jsonl",
                "command": [
                    sys.executable,
                    str(SEARCH_SCRIPT),
                    "--source-manifest",
                    str(config["source_manifest_path"]),
                    "--checkpoint",
                    str(config["critic_checkpoint_path"]),
                    "--mrnabert-model-path",
                    str(config["mrnabert_model_path"]),
                    "--reward-policy",
                    str(config["reward_policy_path"]),
                    "--attention-backend",
                    str(config["selected_attention_backend"]),
                    "--device",
                    str(config["device"]),
                    "--physical-gpu-index",
                    str(config["physical_gpu_index"]),
                    "--method",
                    method,
                    "--critic-budget-by-source",
                    str(config["guided_compute_by_source_path"]),
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
                    str(output_directory / f"{method}.private.jsonl"),
                ],
            }
        )
    return commands


def _method_aggregate(path: Path, method_id: str) -> dict[str, Any]:
    rows = _read_jsonl(path, f"{method_id} candidate output")
    _require(rows and {str(row["method_id"]) for row in rows} == {method_id}, f"method output identity differs: {method_id}")
    first_rows = [row for row in rows if int(row["critic_forwards"]) > 0]
    _require(first_rows, f"method made no critic calls: {method_id}")
    return {
        "method_id": method_id,
        "candidate_row_count": len(rows),
        "source_budget_cohort_count": len(first_rows),
        "critic_forward_budget_total": sum(int(row["critic_forward_budget"]) for row in first_rows),
        "critic_forward_count_total": sum(int(row["critic_forwards"]) for row in first_rows),
        "proposal_count_total": sum(int(row["proposal_count"]) for row in first_rows),
        "unique_candidate_count": len({(str(row["source_key"]), str(row["candidate_sequence"])) for row in rows}),
        "output_path": str(path),
    }


def execute(config_path: Path) -> dict[str, Any]:
    config = _read_json(config_path, "matched-search config")
    validate_config_boundary(config)
    output_directory = Path(str(config["output_directory"]))
    _require(not output_directory.exists(), f"matched-search output exists: {output_directory}")
    readiness_input = _read_json(Path(str(config["readiness_input_path"])), "readiness input")
    readiness_adjudication = _read_json(Path(str(config["readiness_adjudication_path"])), "readiness adjudication")
    independent_evaluator_adjudication = _read_json(
        Path(str(config["independent_evaluator_adjudication_path"])),
        "independent evaluator adjudication",
    )
    guided_summary = _read_json(Path(str(config["guided_summary_path"])), "guided summary")
    compute_rows = _read_jsonl(Path(str(config["guided_compute_by_source_path"])), "guided source compute")
    source_rows = _read_jsonl(Path(str(config["source_manifest_path"])), "source manifest")
    backend_adjudication = _read_json(
        Path(str(config["encoder_attention_backend_adjudication_path"])),
        "encoder attention backend adjudication",
    )
    _require(
        backend_adjudication.get("schema_version")
        == "route_a_v3_route2_mrnabert_sdpa_backend_adjudication.v1"
        and backend_adjudication.get("status") == "ONLINE_ENCODER_BACKEND_ADJUDICATED"
        and backend_adjudication.get("evaluation_opened") is False,
        "encoder attention backend adjudication is invalid",
    )
    execution_config = dict(config)
    execution_config["selected_attention_backend"] = str(
        backend_adjudication["selected_attention_backend"]
    )
    _require(
        execution_config["selected_attention_backend"]
        in {"OFFICIAL_PYTORCH_FALLBACK", "PYTORCH_SDPA_AUTO"},
        "encoder attention backend selection is unknown",
    )
    budgets = validate_inputs(
        config,
        readiness_input,
        readiness_adjudication,
        independent_evaluator_adjudication,
        guided_summary,
        compute_rows,
        source_rows,
    )
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir()
    log_directory = output_directory / "logs"
    log_directory.mkdir()
    started = time.time()
    method_summaries = []
    try:
        for spec in build_commands(execution_config):
            stdout_path = log_directory / f"{spec['method_id']}.stdout.log"
            stderr_path = log_directory / f"{spec['method_id']}.stderr.log"
            with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
                result = subprocess.run(
                    spec["command"],
                    cwd=REPO_ROOT,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                    check=False,
                )
            _require(result.returncode == 0, f"matched search failed: {spec['method_id']}")
            method_summaries.append(
                _method_aggregate(spec["output_path"], spec["method_id"])
            )
        summary = {
            "schema_version": "route_a_v3_route2_mrnabert_critic_v2_matched_search_suite.v1",
            "status": "MATCHED_MRNABERT_SEARCH_CANDIDATES_COMPLETE_NOT_SCIENTIFICALLY_SELECTED",
            "method_summaries": method_summaries,
            "method_count": len(method_summaries),
            "source_budget_cohort_count": len(budgets),
            "encoder_attention_backend": execution_config[
                "selected_attention_backend"
            ],
            "critic_budget_rule": config["critic_budget_rule"],
            "critic_budget_total_per_method": sum(budgets.values()),
            "critic_budget_minimum_per_source": min(budgets.values()),
            "critic_budget_maximum_per_source": max(budgets.values()),
            "candidate_generation_only": True,
            "strongest_method_selected": False,
            "independent_evaluator_status": independent_evaluator_adjudication["status"],
            "independent_evaluator_qualified": (
                independent_evaluator_adjudication["status"]
                == "INDEPENDENT_GENERATION_EVALUATOR_QUALIFIED"
            ),
            "evaluation_outcomes_read": 0,
            "generated_candidates_grant_canonical_credit": False,
            "wall_time_seconds": time.time() - started,
            "scientific_claim_status": "NOT_ESTABLISHED",
        }
    except Exception as exc:
        summary = {
            "schema_version": "route_a_v3_route2_mrnabert_critic_v2_matched_search_suite.v1",
            "status": "MATCHED_MRNABERT_SEARCH_CANDIDATE_GENERATION_FAILED",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "strongest_method_selected": False,
            "evaluation_outcomes_read": 0,
            "scientific_claim_status": "NOT_ESTABLISHED",
        }
        (output_directory / "failed_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
    (output_directory / "suite_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    serialized = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (output_directory / "suite_summary.json").write_text(serialized, encoding="utf-8")
    (output_directory / "final_summary.json").write_text(serialized, encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    execute(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
