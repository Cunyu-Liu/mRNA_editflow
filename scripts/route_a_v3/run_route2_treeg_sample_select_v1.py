#!/usr/bin/env python3
"""TreeG-style sample-then-select diagnostic (SetFlow spec SubTask 4.3 / B2 contingency).

Context: B2 guided FAIL with all-task-uniform-gain attribution (journal
a8b8977c) -> guidance-form constraint. This diagnostic re-uses the base
sampler's existing candidate pool (the unguided 891x32 candidates from
guided_b2_20260903/b2_full_891/unguided/generated_candidates.private.jsonl),
re-scores every candidate with the frozen XEditCritic V5 critic, and per source
selects the top-K by critic value (destination-state sample-then-SELECT, TreeG
arXiv 2502.11420 spirit), then evaluates with the frozen Task-1 evaluator and
bootstraps Delta vs the unguided arm (source-group paired cluster, 2000 iters).

Registered scope (lightweight, selection-only): NO new generation. Primary
readout = hit@1 (selection quality); recovery@10 is secondary with the
shot-count caveat (top-1 selection yields 1 candidate/source vs the unguided
32). A full sample-then-select test with a larger sample pool (N>32) requires
new generation and is a registered follow-up, not run here.

Discipline: CUDA BF16 only (SystemExit if CUDA unavailable - no CPU fallback);
protected reads = 0; --smoke-sources caps the source cohort and marks the
output non-terminal (diagnostic only, never a scientific conclusion).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.route_a_v3.adjudicate_route2_guided_setflow_v5_b2_v1 import (  # noqa: E402
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    MEASURED_TOP_K,
    _measured_pool_by_source,
    _read_jsonl,
    per_source_arm_metrics,
    source_group_bootstrap,
)
from scripts.route_a_v3.evaluate_route2_generation_v1 import (  # noqa: E402
    load_source_manifest,
    validate_measured_pool,
)
from scripts.route_a_v3.run_route2_base_flow_g0_validation_v1 import load_sources  # noqa: E402
from core.route2_legal_xeditflow import (  # noqa: E402
    LegalAction,
    apply_action,
    initial_state,
)
from scripts.route_a_v3.route2_xeditcritic_v5_frozen_guidance_v1 import (  # noqa: E402
    FrozenXEditCriticV5,
)

REWARD_POLICY = (
    REPO_ROOT / "configs/route_a_v3_route2_mrnabert_guidance_reward_policy_v1.json"
)
DEFAULT_CRITIC_CHECKPOINT = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_v5/"
    "v5_screen_seed_20260907_runner_1113cd2c0dd9acb508f58782eecb40f458d2cab3/"
    "v5_full/final_pass_8_checkpoint.pt"
)
DEFAULT_MRNABERT_MODEL = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/"
    "mrnabert_a1eb7df25804d23f08646e1cb996b234d7208a40"
)
SCHEMA = "route_a_v3_route2_setflow_v5_treeg_sample_select.v1"


class TreeGSampleSelectError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TreeGSampleSelectError(message)


def select_top_k(
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    k: int,
) -> list[Mapping[str, Any]]:
    """Stable, tie-aware top-k by critic score (descending)."""
    order = sorted(range(len(rows)), key=lambda i: (-float(scores[i]), i))
    return [rows[i] for i in order[:k]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path,
                        help="unguided generated_candidates.private.jsonl (pool)")
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--critic-checkpoint", type=Path, default=DEFAULT_CRITIC_CHECKPOINT)
    parser.add_argument("--mrnabert-model", type=Path, default=DEFAULT_MRNABERT_MODEL)
    parser.add_argument("--select-k", type=int, default=1)
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--smoke-sources", type=int, default=0,
                        help=">0 caps source cohort (non-terminal smoke)")
    parser.add_argument("--encoder-attention-backend", default="xformers")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU required (no CPU fallback)")
    device = torch.device(f"cuda:{args.physical_gpu_index}")
    _require(args.select_k >= 1, "select-k must be >= 1")

    config = json.loads(args.config.read_text())
    manifest_path = Path(config["source_eligibility_manifest"])
    measured_path = Path(config["measured_neighborhood_path"])

    source_rows = _read_jsonl(manifest_path)
    source_by_key = {str(s["source_key"]): s for s in load_sources(manifest_path)}
    full_manifest = load_source_manifest(manifest_path)
    pool = _read_jsonl(args.candidates)
    covered = sorted({str(row["source_key"]) for row in pool})
    manifest = {key: full_manifest[key] for key in covered}
    if args.smoke_sources > 0:
        manifest = {key: manifest[key] for key in list(manifest)[: args.smoke_sources]}
    pool = [row for row in pool if str(row["source_key"]) in manifest]
    _require(bool(pool), "candidate pool is empty after cohort filter")
    source_group_by_key = {
        str(row["source_key"]): str(row["source_id"]) for row in source_rows
    }

    measured_rows = _read_jsonl(measured_path)
    validate_measured_pool(measured_rows, "DEVELOPMENT", "CLOSED")
    measured_rows = [r for r in measured_rows if str(r["source_key"]) in manifest]
    _require(bool(measured_rows), "smoke source subset has no measured neighborhood")

    transform = json.loads(REWARD_POLICY.read_text())
    minimum = float(transform["potential_transform"]["minimum"])
    maximum = float(transform["potential_transform"]["maximum"])
    critic = FrozenXEditCriticV5(
        Path(args.critic_checkpoint),
        Path(args.mrnabert_model),
        device,
        potential_minimum=minimum,
        potential_maximum=maximum,
    )

    pool_method = str(pool[0]["method_id"])
    # e.g. "unguided_xeditsetflow_v5_b_fix2_pass2_seed20260915"
    run_id = pool_method.split("_xeditsetflow_v5_")[1].split("_pass")[0]
    method_id = f"treeg_sample_select_v5_{run_id}_k{args.select_k}_seed{args.bootstrap_seed}"

    selected: list[Mapping[str, Any]] = []
    scoring = {"potential_query_count": 0, "scoring_batch_count": 0}
    for source_key in sorted(manifest):
        src = manifest[source_key]
        src_full = source_by_key[source_key]
        rows = [r for r in pool if str(r["source_key"]) == source_key]
        _require(bool(rows), f"pool has no candidates for source: {source_key}")
        source_seq = str(src_full["source_sequence"])
        region = str(src_full["region"]).replace("′", "").replace("'", "")
        states = []
        for r in rows:
            state = initial_state(
                source_seq,
                budget=int(src_full["edit_budget"]),
                assay_id=str(src_full["assay_id"]),
                context_id=str(src_full["biological_context_id"]),
            )
            for act_str in r["trajectory_actions"]:
                kind, *rest = str(act_str).split(":")
                if kind == "STOP":
                    state = apply_action(state, LegalAction("STOP"))
                else:
                    state = apply_action(state, LegalAction(kind, int(rest[0]), rest[1]))
                if state.terminal_cause is not None:
                    break
            states.append(state)
        scores = critic.potentials(
            states,
            endpoint_id=str(src_full["endpoint_id"]),
            region=region,
            source_row=src_full,
        )
        scoring["potential_query_count"] += len(scores)
        scoring["scoring_batch_count"] += 1
        for row in select_top_k(rows, scores, int(args.select_k)):
            out = dict(row)
            out["generation_score"] = float(scores[rows.index(row)])
            out["method_id"] = method_id
            out["treeg_critic_score"] = float(scores[rows.index(row)])
            selected.append(out)

    measured_pools = _measured_pool_by_source(measured_rows)
    treeg_metrics = per_source_arm_metrics(
        manifest, selected, measured_rows, measured_pools
    )
    unguided_metrics = per_source_arm_metrics(
        manifest, pool, measured_rows, measured_pools
    )

    group_keys: dict[str, list[str]] = defaultdict(list)
    for source_key in sorted(manifest):
        group_keys[str(source_group_by_key[source_key])].append(source_key)

    delta_recovery = source_group_bootstrap(
        group_keys, treeg_metrics, unguided_metrics, "candidate_recovery_rate",
        iterations=args.bootstrap_iterations, seed=args.bootstrap_seed,
    )
    delta_hit = source_group_bootstrap(
        group_keys, treeg_metrics, unguided_metrics, "hit_at_1",
        iterations=args.bootstrap_iterations, seed=args.bootstrap_seed,
    )

    def macro(metrics: Mapping[str, Mapping[str, float]], metric: str) -> float:
        return float(np.mean([row[metric] for row in metrics.values()]))

    report = {
        "schema_version": SCHEMA,
        "status": "SMOKE_NON_TERMINAL" if args.smoke_sources > 0 else "TERMINAL",
        "method_id": method_id,
        "select_k": args.select_k,
        "source_count": len(manifest),
        "treeg_candidate_count": len(selected),
        "pool_candidate_count": len(pool),
        "caveat": (
            "selection-only on existing unguided pool; recovery@10 shot count "
            "scales with select_k; primary readout is hit_at_1"
        ),
        "treeg": {
            "source_macro_candidate_recovery_rate": macro(treeg_metrics, "candidate_recovery_rate"),
            "source_macro_measured_top_k_recovery_at_k": macro(treeg_metrics, "measured_top_k_recovery_at_k"),
            "hit_at_1": macro(treeg_metrics, "hit_at_1"),
        },
        "unguided": {
            "source_macro_candidate_recovery_rate": macro(unguided_metrics, "candidate_recovery_rate"),
            "source_macro_measured_top_k_recovery_at_k": macro(unguided_metrics, "measured_top_k_recovery_at_k"),
            "hit_at_1": macro(unguided_metrics, "hit_at_1"),
        },
        "delta_recovery_vs_unguided": delta_recovery,
        "delta_hit_at_1_vs_unguided": delta_hit,
        "scoring": scoring,
        "critic_checkpoint_path": str(args.critic_checkpoint),
        "cpu_fallback_used": False,
        "precision": "BF16",
        "protected_reads": 0,
    }
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report, indent=1, sort_keys=True))
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
