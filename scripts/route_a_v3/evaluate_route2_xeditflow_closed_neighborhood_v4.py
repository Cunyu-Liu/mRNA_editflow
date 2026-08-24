#!/usr/bin/env python3
"""Evaluate exact latent-mode-marginalized V4 probabilities on Validation."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

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
from core.route2_gpu_failure_evidence import (
    cuda_device_observation,
    write_gpu_failure_evidence,
)
from core.route2_legal_xeditflow import (
    FlowState,
    LegalAction,
    initial_state,
    legal_actions,
)
from core.route2_source_token_cache_v3 import (
    SourceTokenCacheIndexV3,
    load_source_token_cache_v3,
)
from core.route2_xeditflow_gate_v4 import authorize_xeditflow_guidance_v4
from core.route2_xeditflow_smc_runtime_v4 import (
    SetFlowModeRateProviderV4,
    SetFlowModeValueProvidersV4,
    scalar_potential_mode_rate_maps_v4,
)
from core.route2_xeditflow_guidance_v4 import SetFlowMixtureStateV4
from core.route2_xeditflow_value_training_v4 import (
    BASE_FLOW_SEEDS_V4,
    load_value_checkpoint_v4,
)
from core.route2_xeditsetflow_sampling_v3 import build_generation_metadata_v3
from core.route2_xeditsetflow_sampling_v4 import root_mode_priors_v4
from scripts.route_a_v3.generate_route2_xeditflow_value_rollouts_v4 import (
    _selected_checkpoint_pass_v4,
)
from scripts.route_a_v3.run_route2_base_flow_g0_validation_v1 import load_sources
from scripts.route_a_v3.validate_route2_xeditsetflow_v4_checkpoint import (
    load_checkpoint_v4,
)


class ClosedNeighborhoodRunnerV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ClosedNeighborhoodRunnerV4Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _require(
        bool(rows) and all(isinstance(row, dict) for row in rows),
        f"JSONL input is empty or invalid: {path}",
    )
    return rows


def validate_closed_run_config_v4(config: Mapping[str, Any]) -> None:
    _require(
        config.get("schema_version")
        == "route_a_v3_route2_xeditflow_closed_neighborhood_config.v4",
        "unexpected V4 closed benchmark config",
    )
    _require(
        config.get("pool_assignment") == "DEVELOPMENT"
        and config.get("split") == "VALIDATION",
        "V4 closed benchmark cohort differs",
    )
    _require(
        int(config.get("maximum_enumerated_edits", -1)) == 5
        and int(config.get("maximum_permutation_paths", -1)) == 120
        and config.get("enumeration") == "ALL_EDIT_PERMUTATIONS_EXACT_SUM"
        and config.get("analysis_unit") == "SOURCE"
        and config.get("undefined_source_policy") == "EXCLUDE_NOT_ZERO_FILL",
        "V4 closed benchmark enumeration policy differs",
    )
    _require(
        config.get("potential_kind") in {"SOFT_VALUE", "ZERO"}
        and config.get("latent_mode_policy")
        == "ROOT_PRIOR_WEIGHTED_SUM_OF_EIGHT_FIXED_MODE_TERMINAL_PROBABILITIES",
        "V4 closed benchmark potential or mode marginalization differs",
    )
    _require(
        int(config.get("base_flow_training_seed", -1)) in BASE_FLOW_SEEDS_V4
        and float(config.get("kappa", -1)) in {0.0, 0.5, 1.0}
        and float(config.get("temperature", -1)) in {0.5, 1.0}
        and float(config.get("beta_max", -1)) in {0.5, 1.0, 2.0}
        and bool(str(config.get("method_id", ""))),
        "V4 closed benchmark combination differs",
    )
    _require(
        int(config.get("expected_source_count", -1)) == 891,
        "V4 closed benchmark source count differs",
    )
    _require(
        int(config.get("root_prior_forward_batch_size", -1)) == 32
        and int(config.get("value_child_forward_batch_size", -1)) == 32,
        "V4 closed benchmark forward batch size differs",
    )
    gpu = int(config.get("physical_gpu_index", -1))
    _require(
        gpu in range(6) and str(config.get("device")) == f"cuda:{gpu}",
        "V4 closed benchmark GPU provenance differs",
    )
    _require(
        config.get("independent_evaluator_used") is False
        and config.get("development_test_outcomes_accessed_after_atomic_test")
        is False
        and config.get("new_final_evaluation_outcomes_accessed") is False,
        "V4 closed benchmark protected-input policy differs",
    )
    for field in (
        "source_token_cache_path",
        "source_eligibility_manifest",
        "validation_projection_path",
        "measured_neighborhood_path",
        "output_dir",
    ):
        _require(
            str(config.get(field, "")).startswith(
                "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"
            ),
            f"V4 closed benchmark {field} left Route 2 /mnt",
        )
    if config.get("potential_kind") == "SOFT_VALUE":
        _require(
            str(config.get("value_checkpoint_path", "")).startswith(
                "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"
            ),
            "V4 closed benchmark value checkpoint left Route 2 /mnt",
        )
    else:
        _require(
            "value_checkpoint_path" not in config,
            "V4 unguided closed benchmark received a value checkpoint",
        )


def mode_marginal_terminal_probability_v4(
    source_sequence: str,
    candidate_sequence: str,
    *,
    edit_budget: int,
    assay_id: str,
    context_id: str,
    mode_prior: Sequence[float],
    rate_maps: MutableMapping[FlowState, tuple[dict[LegalAction, float], ...]],
    rate_map_builder: Any,
) -> dict[str, Any]:
    prior = tuple(float(value) for value in mode_prior)
    _require(
        len(prior) == 8
        and all(math.isfinite(value) and value >= 0.0 for value in prior)
        and math.isclose(sum(prior), 1.0, rel_tol=0.0, abs_tol=1e-6),
        "V4 closed mode prior differs",
    )
    conditional = []
    path_counts = []
    for mode_id in range(8):

        def rate_function(
            state: FlowState, actions: Sequence[LegalAction]
        ) -> Mapping[LegalAction, float]:
            if state not in rate_maps:
                rate_maps[state] = rate_map_builder(state)
            selected = rate_maps[state][mode_id]
            _require(
                tuple(selected) == tuple(actions),
                "V4 closed cached action order differs",
            )
            return selected

        result = exact_order_invariant_terminal_probability_v3(
            source_sequence,
            candidate_sequence,
            edit_budget=edit_budget,
            assay_id=assay_id,
            context_id=context_id,
            rate_function=rate_function,
        )
        conditional.append(float(result["terminal_probability"]))
        path_counts.append(int(result["permutation_path_count"]))
    marginal = math.fsum(
        weight * probability
        for weight, probability in zip(prior, conditional, strict=True)
    )
    _require(
        math.isfinite(marginal) and 0.0 <= marginal <= 1.0,
        "V4 closed marginal terminal probability is invalid",
    )
    _require(len(set(path_counts)) == 1, "V4 closed mode path counts differ")
    return {
        "terminal_probability": marginal,
        "conditional_terminal_probability_by_mode": conditional,
        "root_mode_prior": list(prior),
        "permutation_path_count_per_mode": path_counts[0],
        "latent_mode_count": 8,
        "latent_mode_marginalized": True,
    }


def run(config: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
    validate_closed_run_config_v4(config)
    _require(output_dir == Path(str(config["output_dir"])), "V4 closed output differs")
    _require(not output_dir.exists(), f"terminal V4 closed output exists: {output_dir}")
    critic_readiness = _json(Path(config["critic_readiness_path"]))
    setflow_confirmation = _json(Path(config["setflow_confirmation_path"]))
    _require(
        authorize_xeditflow_guidance_v4(
            critic_readiness, setflow_confirmation
        )["guidance_authorized"]
        is True,
        "V4 closed benchmark remains blocked before joint readiness",
    )
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    _require(
        not os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES remapping is forbidden",
    )
    gpu = int(config["physical_gpu_index"])
    device = torch.device(str(config["device"]))
    torch.cuda.set_device(device)
    _require(torch.cuda.is_bf16_supported(), "BF16 is unavailable for V4 closed benchmark")
    cuda = cuda_device_observation(gpu, require_physical_index_match=True)
    seed = int(config["base_flow_training_seed"])
    selected_pass = _selected_checkpoint_pass_v4(
        setflow_confirmation, seed=seed
    )
    runtime = _json(Path(config["setflow_runtime_config_path"]))
    setflow, checkpoint, _training_summary = load_checkpoint_v4(
        runtime,
        run_id="v4_full",
        checkpoint_pass=selected_pass,
        device=device,
    )
    value = (
        load_value_checkpoint_v4(
            Path(config["value_checkpoint_path"]),
            base_flow_training_seed=seed,
            kappa=float(config["kappa"]),
            temperature=float(config["temperature"]),
            device=device,
        )
        if config["potential_kind"] == "SOFT_VALUE"
        else None
    )
    cache = SourceTokenCacheIndexV3(
        load_source_token_cache_v3(Path(config["source_token_cache_path"]))
    )
    sources = load_sources(Path(config["source_eligibility_manifest"]))
    _require(len(sources) == 891, "V4 closed source cohort changed")
    source_by_key = {str(row["source_key"]): row for row in sources}
    projection = load_projection_rows(
        [Path(config["validation_projection_path"])],
        allowed_splits=("VALIDATION",),
    )
    metadata = build_generation_metadata_v3(sources, projection, checkpoint["vocabs"])
    metadata_by_key = {
        str(source["source_key"]): item
        for source, item in zip(sources, metadata, strict=True)
    }
    roots = [
        initial_state(
            str(source["source_sequence"]),
            budget=int(source["edit_budget"]),
            assay_id=str(source["assay_id"]),
            context_id=str(source["biological_context_id"]),
        )
        for source in sources
    ]
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    mode_priors, prior_compute = root_mode_priors_v4(
        setflow,
        roots,
        metadata,
        source_cache=cache,
        device=device,
        forward_batch_size=int(config["root_prior_forward_batch_size"]),
    )
    measured = _jsonl(Path(config["measured_neighborhood_path"]))
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in measured:
        source_key = str(row["source_key"])
        _require(source_key in source_by_key, "V4 closed measured source differs")
        _require(
            row.get("pool_assignment") == "DEVELOPMENT"
            and row.get("split") == "VALIDATION",
            "V4 closed entered non-Development/Validation outcome",
        )
        edits = source_relative_substitutions_v3(
            str(source_by_key[source_key]["source_sequence"]),
            str(row["candidate_sequence"]),
        )
        _require(
            len(edits) <= int(source_by_key[source_key]["edit_budget"]),
            "V4 closed candidate exceeds source budget",
        )
        by_source[source_key].append(row)
    _require(set(by_source) == set(source_by_key), "V4 closed measured coverage differs")

    scored_rows = []
    exact_trunk_calls = 0
    exact_mode_calls = 0
    value_calls = 0
    unique_state_count = 0
    for source_index, source in enumerate(sources):
        source_key = str(source["source_key"])
        if value is None:
            providers = SetFlowModeRateProviderV4(
                setflow_model=setflow,
                metadata=metadata_by_key[source_key],
                source_cache=cache,
                device=device,
            )
        else:
            providers = SetFlowModeValueProvidersV4(
                setflow_model=setflow,
                value_model=value,
                metadata=metadata_by_key[source_key],
                source_cache=cache,
                device=device,
            )
        rate_cache: dict[FlowState, tuple[dict[LegalAction, float], ...]] = {}

        def build_rate_maps(
            state: FlowState,
        ) -> tuple[dict[LegalAction, float], ...]:
            nonlocal exact_trunk_calls, exact_mode_calls, value_calls
            if value is None:
                mode_states = [
                    SetFlowMixtureStateV4(state, mode_id)
                    for mode_id in range(8)
                ]
                rows = providers.rates(mode_states)
                actions = tuple(legal_actions(state))
                _require(
                    len(rows) == 8
                    and all(row.actions == actions for row in rows),
                    "V4 unguided closed mode-rate bundle differs",
                )
                result = tuple(
                    dict(zip(row.actions, row.rates, strict=True))
                    for row in rows
                )
                child_value_calls = 0
            else:
                result, child_value_calls = scalar_potential_mode_rate_maps_v4(
                    state,
                    providers.rates,
                    providers.values,
                    beta_max=float(config["beta_max"]),
                    value_forward_batch_size=int(
                        config["value_child_forward_batch_size"]
                    ),
                )
            exact_trunk_calls += 1
            exact_mode_calls += 8
            value_calls += child_value_calls
            return result

        for row in by_source[source_key]:
            probability = mode_marginal_terminal_probability_v4(
                str(source["source_sequence"]),
                str(row["candidate_sequence"]),
                edit_budget=int(source["edit_budget"]),
                assay_id=str(source["assay_id"]),
                context_id=str(source["biological_context_id"]),
                mode_prior=mode_priors[source_index],
                rate_maps=rate_cache,
                rate_map_builder=build_rate_maps,
            )
            scored_rows.append(
                {
                    **row,
                    **probability,
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
        "schema_version": "route_a_v3_route2_xeditflow_closed_neighborhood.v4",
        "status": "XEDITFLOW_V4_CLOSED_NEIGHBORHOOD_COMPLETE",
        "method_id": str(config["method_id"]),
        "potential_kind": str(config["potential_kind"]),
        "base_flow_training_seed": seed,
        "kappa": float(config["kappa"]),
        "temperature": float(config["temperature"]),
        "beta_max": float(config["beta_max"]),
        **{key: value for key, value in metrics.items() if key != "per_source"},
        "per_source": metrics["per_source"],
        "measured_candidate_count": len(scored_rows),
        "unique_scored_state_count": unique_state_count,
        "root_prior_trunk_forward_calls": int(
            prior_compute.trunk_forward_batch_count
        ),
        "root_prior_mode_forward_calls": int(
            prior_compute.mode_head_forward_state_count
        ),
        "exact_path_trunk_forward_calls": exact_trunk_calls,
        "exact_path_mode_head_forward_calls": exact_mode_calls,
        "total_trunk_forward_calls": int(
            prior_compute.trunk_forward_batch_count
        )
        + exact_trunk_calls,
        "total_mode_head_forward_calls": int(
            prior_compute.mode_head_forward_state_count
        )
        + exact_mode_calls,
        "value_scoring_forward_calls": value_calls,
        "critic_scoring_forward_calls_by_member": [0, 0, 0],
        "latent_mode_count": 8,
        "latent_mode_marginalized": True,
        "enumeration_wall_time_seconds": time.time() - started,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "pool_assignment": "DEVELOPMENT",
        "split": "VALIDATION",
        "undefined_sources_are_not_filled_with_zero": True,
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
        "cpu_fallback_used": False,
        **cuda,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    config = _json(arguments.config)
    try:
        result = run(config, output_dir=arguments.output_dir)
    except Exception as exc:
        write_gpu_failure_evidence(
            arguments.output_dir.with_name(arguments.output_dir.name + ".failed.json"),
            config,
            exc,
            entrypoint="evaluate_route2_xeditflow_closed_neighborhood_v4",
            evaluation_outcomes_accessed=False,
        )
        raise
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
