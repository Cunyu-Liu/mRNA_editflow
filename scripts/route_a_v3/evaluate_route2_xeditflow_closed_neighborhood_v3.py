#!/usr/bin/env python3
"""Evaluate exact order-invariant XEditFlow V3 probabilities on Validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_closed_neighborhood_v3 import (
    closed_neighborhood_metrics_v1,
    exact_order_invariant_terminal_probability_v3,
    source_relative_substitutions_v3,
)
from core.route2_development_projection_v3 import load_projection_rows
from core.route2_gpu_failure_evidence import cuda_device_observation, write_gpu_failure_evidence
from core.route2_legal_xeditflow import FlowState, LegalAction
from core.route2_source_token_cache_v3 import SourceTokenCacheIndexV3, load_source_token_cache_v3
from core.route2_xeditflow_gate_v3 import authorize_xeditflow_guidance_v3
from core.route2_xeditflow_smc_runtime_v3 import (
    SetFlowValueProvidersV3,
    scalar_potential_rate_map_v3,
)
from core.route2_xeditsetflow_sampling_v3 import build_generation_metadata_v3
from scripts.route_a_v3.run_route2_base_flow_g0_validation_v1 import load_sources
from scripts.route_a_v3.run_route2_xeditflow_smc_v3 import load_value_checkpoint_v3
from scripts.route_a_v3.validate_route2_xeditsetflow_v3 import load_setflow_checkpoint_v3


class ClosedNeighborhoodRunnerV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ClosedNeighborhoodRunnerV3Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                _require(isinstance(row, dict), f"JSONL row is not an object: {path}")
                rows.append(row)
    _require(bool(rows), f"JSONL input is empty: {path}")
    return rows


def validate_closed_run_config_v3(config: Mapping[str, Any]) -> None:
    _require(config.get("schema_version") == "route_a_v3_route2_xeditflow_closed_neighborhood_config.v1", "unexpected closed benchmark config schema")
    _require(config.get("pool_assignment") == "DEVELOPMENT", "closed benchmark pool differs")
    _require(config.get("split") == "VALIDATION", "closed benchmark split differs")
    _require(int(config.get("maximum_enumerated_edits", -1)) == 5, "closed edit ceiling changed")
    _require(int(config.get("maximum_permutation_paths", -1)) == 120, "closed permutation ceiling changed")
    _require(config.get("enumeration") == "ALL_EDIT_PERMUTATIONS_EXACT_SUM", "closed enumeration changed")
    _require(config.get("analysis_unit") == "SOURCE", "closed analysis unit changed")
    _require(config.get("undefined_source_policy") == "EXCLUDE_NOT_ZERO_FILL", "closed undefined-source policy changed")
    _require(float(config.get("beta_max", -1)) in {0.5, 1.0, 2.0}, "closed beta is outside the frozen grid")


def run(config: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
    validate_closed_run_config_v3(config)
    _require(not output_dir.exists(), f"terminal closed benchmark output already exists: {output_dir}")
    authorization = authorize_xeditflow_guidance_v3(
        _json(Path(config["critic_readiness_path"])),
        _json(Path(config["setflow_confirmation_path"])),
    )
    _require(authorization["guidance_authorized"] is True, "closed guidance benchmark remains blocked before readiness")
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden")
    physical_gpu = int(config["physical_gpu_index"])
    device = torch.device(str(config["device"]))
    _require(device == torch.device(f"cuda:{physical_gpu}"), "closed benchmark device provenance changed")
    torch.cuda.set_device(device)
    _require(torch.cuda.is_bf16_supported(), "BF16 is unavailable for closed benchmark")
    cuda = cuda_device_observation(physical_gpu, require_physical_index_match=True)
    arm = str(config["setflow_arm"])
    setflow, checkpoint = load_setflow_checkpoint_v3(Path(config["setflow_checkpoint_path"]), arm, device)
    _require(int(checkpoint["training_provenance"]["seed"]) == int(config["base_flow_training_seed"]), "closed SetFlow seed differs")
    value = load_value_checkpoint_v3(Path(config["value_checkpoint_path"]), config=config, device=device)
    cache = SourceTokenCacheIndexV3(load_source_token_cache_v3(Path(config["source_token_cache_path"])))
    sources = load_sources(Path(config["source_eligibility_manifest"]))
    source_by_key = {str(row["source_key"]): row for row in sources}
    _require(len(source_by_key) == len(sources) == int(config["expected_source_count"]), "closed source cohort changed")
    projection = load_projection_rows([Path(config["validation_projection_path"])], allowed_splits=("VALIDATION",))
    metadata = build_generation_metadata_v3(sources, projection, checkpoint["vocabs"])
    metadata_by_key = {
        str(source["source_key"]): item
        for source, item in zip(sources, metadata, strict=True)
    }
    measured = _jsonl(Path(config["measured_neighborhood_path"]))
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in measured:
        source_key = str(row["source_key"])
        _require(source_key in source_by_key, "closed measured row has an unknown source")
        _require(row.get("pool_assignment") == "DEVELOPMENT", "closed benchmark entered a non-Development outcome")
        _require(row.get("split") == "VALIDATION", "closed benchmark entered a non-Validation measured row")
        source = str(source_by_key[source_key]["source_sequence"])
        candidate = str(row["candidate_sequence"])
        edits = source_relative_substitutions_v3(source, candidate)
        _require(len(edits) <= int(source_by_key[source_key]["edit_budget"]), "closed measured candidate exceeds source budget")
        by_source[source_key].append(row)
    _require(set(by_source) == set(source_by_key), "closed measured neighborhood does not cover the source cohort")
    scored_rows = []
    base_forward_calls = 0
    value_forward_calls = 0
    unique_state_count = 0
    started = time.time()
    for source_key in sorted(source_by_key):
        source = source_by_key[source_key]
        providers = SetFlowValueProvidersV3(
            setflow_model=setflow,
            setflow_arm=arm,
            value_model=value,
            metadata=metadata_by_key[source_key],
            source_cache=cache,
            device=device,
        )
        rate_cache: dict[FlowState, dict[LegalAction, float]] = {}

        def rate_function(state: FlowState, actions: Sequence[LegalAction]) -> Mapping[LegalAction, float]:
            nonlocal base_forward_calls, value_forward_calls
            if state not in rate_cache:
                rate_cache[state] = scalar_potential_rate_map_v3(
                    state,
                    actions,
                    providers.rates,
                    providers.values,
                    beta_max=float(config["beta_max"]),
                )
                base_forward_calls += 1
                value_forward_calls += 1
            _require(tuple(rate_cache[state]) == tuple(actions), "closed cached action order differs")
            return rate_cache[state]

        for row in by_source[source_key]:
            probability = exact_order_invariant_terminal_probability_v3(
                str(source["source_sequence"]),
                str(row["candidate_sequence"]),
                edit_budget=int(source["edit_budget"]),
                assay_id=str(source["assay_id"]),
                context_id=str(source["biological_context_id"]),
                rate_function=rate_function,
            )
            scored_rows.append(
                {
                    **row,
                    "terminal_probability": probability["terminal_probability"],
                    "permutation_path_count": probability["permutation_path_count"],
                    "terminal_causes": probability["terminal_causes"],
                    "all_edit_permutations_enumerated": True,
                }
            )
        unique_state_count += len(rate_cache)
    metrics = closed_neighborhood_metrics_v1(scored_rows)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    (output_dir / "closed_candidate_probabilities.private.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in scored_rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "route_a_v3_route2_xeditflow_closed_neighborhood.v3",
        "status": "XEDITFLOW_V3_CLOSED_NEIGHBORHOOD_COMPLETE",
        "method_id": str(config["method_id"]),
        "setflow_arm": arm,
        "base_flow_training_seed": int(config["base_flow_training_seed"]),
        "kappa": float(config["kappa"]),
        "temperature": float(config["temperature"]),
        "beta_max": float(config["beta_max"]),
        **{key: value for key, value in metrics.items() if key != "per_source"},
        "per_source": metrics["per_source"],
        "measured_candidate_count": len(scored_rows),
        "unique_scored_state_count": unique_state_count,
        "base_flow_scoring_forward_calls": base_forward_calls,
        "value_scoring_forward_calls": value_forward_calls,
        "enumeration_wall_time_seconds": time.time() - started,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "pool_assignment": "DEVELOPMENT",
        "split": "VALIDATION",
        "undefined_sources_are_not_filled_with_zero": True,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
        "cpu_fallback_used": False,
        **cuda,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = _json(args.config)
    try:
        result = run(config, output_dir=args.output_dir)
    except Exception as exc:
        write_gpu_failure_evidence(
            args.output_dir.with_name(args.output_dir.name + ".failed.json"),
            config,
            exc,
            entrypoint="evaluate_route2_xeditflow_closed_neighborhood_v3",
            evaluation_outcomes_accessed=False,
        )
        raise
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
