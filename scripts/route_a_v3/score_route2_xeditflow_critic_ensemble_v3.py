#!/usr/bin/env python3
"""Attach frozen three-member Critic V3 diagnostics to generated candidates."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_development_projection_v3 import load_projection_rows
from core.route2_gpu_failure_evidence import cuda_device_observation, write_gpu_failure_evidence
from core.route2_xeditflow_gate_v3 import authorize_xeditflow_guidance_v3
from core.route2_xeditflow_value_rollouts_v3 import attach_candidate_critic_rewards_v3
from core.route2_xeditflow_value_training_v3 import CRITIC_SEEDS_V3
from scripts.route_a_v3.generate_route2_xeditflow_value_rollouts_v3 import (
    _score_critic_member_v3,
)
from scripts.route_a_v3.run_route2_base_flow_g0_validation_v1 import load_sources


class XEditFlowCriticEnsembleScorerV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowCriticEnsembleScorerV3Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(bool(rows) and all(isinstance(row, dict) for row in rows), f"JSONL input is empty or invalid: {path}")
    return rows


def validate_critic_ensemble_score_config_v3(config: Mapping[str, Any]) -> None:
    _require(config.get("schema_version") == "route_a_v3_route2_xeditflow_critic_ensemble_score_config.v1", "unexpected Critic ensemble score config")
    _require(float(config.get("kappa", -1)) in {0.0, 0.5, 1.0}, "Critic ensemble kappa differs")
    _require(int(config.get("base_flow_training_seed", -1)) in {20260904, 20260905, 20260906}, "Critic ensemble base-flow seed differs")
    _require(int(config.get("critic_batch_size", -1)) == 256, "Critic ensemble batch size changed")
    _require(int(config.get("critic_online_microbatch_size", -1)) == 4, "Critic ensemble microbatch size changed")
    gpu = int(config.get("physical_gpu_index", -1))
    _require(gpu in set(range(6)) and config.get("device") == f"cuda:{gpu}", "Critic ensemble GPU provenance differs")


def _representatives_v3(
    sources: list[Mapping[str, Any]],
    projection: list[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    by_key = {}
    for row in projection:
        key = (
            str(row["source_sequence"]),
            str(row["endpoint_id"]),
            str(row["biological_context_id"]),
        )
        if key not in by_key or str(row["canonical_record_id"]) < str(by_key[key]["canonical_record_id"]):
            by_key[key] = row
    result = {}
    for source in sources:
        key = (
            str(source["source_sequence"]),
            str(source["endpoint_id"]),
            str(source["biological_context_id"]),
        )
        _require(key in by_key, "generated source lacks Validation endpoint metadata")
        result[str(source["source_key"])] = by_key[key]
    return result


def _critic_adapter_rows_v3(
    candidates: list[Mapping[str, Any]],
    sources: list[Mapping[str, Any]],
    representatives: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    source_by_key = {str(row["source_key"]): row for row in sources}
    result = []
    for candidate in candidates:
        source_key = str(candidate["source_key"])
        _require(source_key in source_by_key and source_key in representatives, "Critic candidate source differs")
        source_row = source_by_key[source_key]
        source = str(source_row["source_sequence"])
        sequence = str(candidate["candidate_sequence"])
        _require(len(source) == len(sequence), "Critic candidate length differs")
        edits = [
            {"position": index, "source_base": left, "candidate_base": right}
            for index, (left, right) in enumerate(zip(source, sequence))
            if left != right
        ]
        representative = representatives[source_key]
        region = str(source_row["region"]).replace("′", "").replace("'", "")
        _require(region in {"5UTR", "3UTR"}, "Critic candidate region differs")
        result.append(
            {
                "state_id": source_key,
                "rollout_index": int(candidate["generation_rank"]) - 1,
                "source_group_id": str(representative["source_group_id"]),
                "task_id": str(representative["task_id"]),
                "source_sequence": source,
                "candidate_sequence": sequence,
                "source_relative_edits": edits,
                "endpoint_descriptor": dict(representative["endpoint_descriptor"]),
                "assay_category": str(source_row["assay_id"]),
                "context_category": str(source_row["biological_context_id"]),
                "region_id": 0 if region == "5UTR" else 1,
            }
        )
    return result


def run(config: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
    validate_critic_ensemble_score_config_v3(config)
    _require(not output_dir.exists(), f"Critic ensemble score output exists: {output_dir}")
    readiness = _json(Path(config["critic_readiness_path"]))
    flow = _json(Path(config["setflow_confirmation_path"]))
    _require(authorize_xeditflow_guidance_v3(readiness, flow)["guidance_authorized"] is True, "Critic ensemble scoring remains blocked before readiness")
    refit = _json(Path(config["critic_refit_manifest_path"]))
    _require(refit.get("status") == "XEDITCRITIC_V3_ALL_DEVELOPMENT_REFIT_COMPLETE", "Critic ensemble refit manifest is incomplete")
    selected_arm = str(refit.get("selected_arm"))
    _require(selected_arm in {"C2", "C3"}, "Critic ensemble arm differs")
    checkpoints = {int(row["seed"]): Path(row["checkpoint_path"]) for row in refit["checkpoints"]}
    _require(tuple(sorted(checkpoints)) == CRITIC_SEEDS_V3, "Critic ensemble seed inventory differs")
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden")
    gpu = int(config["physical_gpu_index"])
    device = torch.device(str(config["device"]))
    torch.cuda.set_device(device)
    _require(torch.cuda.is_bf16_supported(), "BF16 is unavailable for Critic ensemble scoring")
    cuda = cuda_device_observation(gpu, require_physical_index_match=True)
    sources = load_sources(Path(config["source_eligibility_manifest"]))
    projection = load_projection_rows(
        [Path(config["validation_projection_path"])], allowed_splits=("VALIDATION",)
    )
    candidates = _jsonl(Path(config["candidate_path"]))
    _require({str(row["source_key"]) for row in candidates} == {str(row["source_key"]) for row in sources}, "Critic candidate source coverage differs")
    adapters = _critic_adapter_rows_v3(
        candidates, sources, _representatives_v3(sources, projection)
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    adapter_path = output_dir / "critic_adapter_candidates.private.jsonl"
    adapter_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in adapters),
        encoding="utf-8",
    )
    member_paths = {
        seed: output_dir / f"critic_member_seed{seed}.private.jsonl"
        for seed in CRITIC_SEEDS_V3
    }
    started = time.time()
    for seed in CRITIC_SEEDS_V3:
        count = _score_critic_member_v3(
            adapter_path,
            member_paths[seed],
            checkpoint_path=checkpoints[seed],
            selected_arm=selected_arm,
            seed=seed,
            model_path=Path(config["mrnabert_model_path"]),
            device=device,
            batch_size=256,
            microbatch_size=4,
        )
        _require(count == len(candidates), "Critic candidate member count differs")
    member_rows = {seed: _jsonl(member_paths[seed]) for seed in CRITIC_SEEDS_V3}
    scored = attach_candidate_critic_rewards_v3(
        candidates, member_rows, kappa=float(config["kappa"])
    )
    scored_path = output_dir / "critic_scored_candidates.private.jsonl"
    scored_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in scored),
        encoding="utf-8",
    )
    source_count = len(sources)
    _require(sum(int(row["critic_forwards"]) for row in scored) == source_count * 3, "Critic forward accounting differs")
    summary = {
        "schema_version": "route_a_v3_route2_xeditflow_critic_ensemble_scoring.v3",
        "status": "XEDITFLOW_V3_CRITIC_ENSEMBLE_SCORING_COMPLETE",
        "method_id": str(config["method_id"]),
        "base_flow_training_seed": int(config["base_flow_training_seed"]),
        "critic_arm": selected_arm,
        "critic_seeds": list(CRITIC_SEEDS_V3),
        "kappa": float(config["kappa"]),
        "source_count": source_count,
        "candidate_count": len(scored),
        "critic_forward_equivalents_per_source": 3,
        "scored_candidate_path": str(scored_path),
        "critic_self_score_used_for_candidate_selection": False,
        "study_neutral": True,
        "wall_time_seconds": time.time() - started,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "cpu_fallback_used": False,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
        **cuda,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
            entrypoint="score_route2_xeditflow_critic_ensemble_v3",
            evaluation_outcomes_accessed=False,
        )
        raise
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
