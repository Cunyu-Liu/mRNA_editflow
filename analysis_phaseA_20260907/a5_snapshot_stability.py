#!/usr/bin/env python3
"""A5 - SetFlow V5 b_fix2 snapshot-stability drift table.

Report a per-pass { recovery, unique-candidate, NLL, hit@1, legality } drift
table across the THREE saved b_fix2 checkpoints (pass 2 / 4 / 6), with deltas
relative to the pass-2 baseline.  The pass-2 row reuses the project's OWN
frozen outcome-free 891x32 validation data (it is NOT re-generated).

Source of truth
---------------
* Generated candidates  : outcome_free_validation_generation/b_fix2/pass_<N>/
                          trajectories.private.jsonl   (frozen 891 x32)
* NLL / recovery / unique / legality : the sibling validation_summary.json
* hit@1                : recomputed via the REUSED metrics entry
                          scripts/route_a_v3/evaluate_route2_generation_v1.py
                          :: measured_neighborhood_metrics(k=1)  (tie-aware,
                          same semantics as the A1 metrics-audit hit@1).

Fallback generation
-------------------
If a pass's validation_summary.json is MISSING, the script shells out to the
project's own checkpoint-validation entry
(validate_route2_xeditsetflow_v5_checkpoint.py) to re-generate 891 x 32 into a
scratch output dir (needs one free CUDA/BF16 GPU; passed via
--physical-gpu-index) then aggregates.  With the existing frozen data present
the fallback is simply not exercised.

Discipline: protected writes = none on frozen artifacts; protected reads = 0
(only DEVELOPMENT VALIDATION structure + checkpoint weights + measured chain
identities & measured_direction_normalized_delta to resolve ordering).  No
TEST/EVALUATION outcome file is ever opened.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/"
    "route_a_v3_setflow_v5_base_fix_20260901"
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scripts.route_a_v3.evaluate_route2_generation_v1 import (  # noqa: E402
    _read_jsonl,
    load_source_manifest,
    measured_neighborhood_metrics,
    validate_measured_pool,
)

# ---------------------------------------------------------------------------
# Frozen project paths (same hard-coding convention as a3_critic_mixed_pool_probe)
# ---------------------------------------------------------------------------
CONFIG = REPO_ROOT / "configs/route_a_v3_route2_xeditsetflow_v5_screen_v1.json"
RUN_ID = "b_fix2"
FAMILY_ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
SCREEN_ROOT = (
    FAMILY_ROOT
    / "experiments/xeditsetflow_v5/screen_20260915"
)
VALIDATION_ROOT = SCREEN_ROOT / "outcome_free_validation_generation" / RUN_ID
MANIFEST = (
    FAMILY_ROOT
    / "generation_eligibility/development_validation_v1/source_eligibility.jsonl"
)
MEASURED = (
    FAMILY_ROOT
    / "generation_eligibility/development_validation_v1/"
    "measured_neighborhood.private.jsonl"
)
LAUNCH_AUTHORIZATION = (
    FAMILY_ROOT
    / "authorizations/xeditsetflow_v5/v5_screen_launch_authorization.json"
)
VALIDATE_ENTRY = REPO_ROOT / "scripts/route_a_v3/validate_route2_xeditsetflow_v5_checkpoint.py"

BASELINE_PASS = 2
# The three saved checkpoint snapshots (config training.saved_checkpoint_passes).
SAVED_PASSES = (2, 4, 6)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _compute_hit_at_1(candidate_rows, manifest, measured_rows) -> float:
    """Reuse measured_neighborhood_metrics(k=1) -> hit@1 (tie-aware)."""
    result = measured_neighborhood_metrics(
        manifest,
        candidate_rows,
        measured_rows,
        k=1,
        candidate_support_mode="OPEN_GENERATED_SUPPORT",
    )
    return float(result["source_macro_measured_top_k_recovery_at_k"])


def _rounds() -> dict:
    """Precompute measured pool + manifest once per process for hit@1."""
    manifest = load_source_manifest(MANIFEST)
    measured_rows = _read_jsonl(MEASURED)
    validate_measured_pool(measured_rows, "DEVELOPMENT", "CLOSED")
    return {"manifest": manifest, "measured": measured_rows}


def _aggregate_existing_pass(
    checkpoint_pass: int, shared: dict
) -> dict:
    """Build a per-pass row from the frozen validation_summary + trajectories."""
    val_dir = VALIDATION_ROOT / f"pass_{checkpoint_pass}"
    summary_path = val_dir / "validation_summary.json"
    summary = _read_json(summary_path)
    status = str(summary.get("status", ""))
    trajectory_path = val_dir / "trajectories.private.jsonl"
    if not trajectory_path.exists():
        raise FileNotFoundError(f"missing frozen trajectories: {trajectory_path}")
    candidates = _read_jsonl(trajectory_path)
    hit_at_1 = _compute_hit_at_1(candidates, shared["manifest"], shared["measured"])
    return {
        "checkpoint_pass": checkpoint_pass,
        "status": status,
        "g0_status": summary.get("g0_status"),
        "source_count": summary.get("source_count"),
        "candidate_count": summary.get("candidate_count"),
        "recovery": float(summary["source_macro_candidate_recovery_rate"]),
        "unique_candidate_rate": float(summary["source_macro_unique_candidate_rate"]),
        "nll": float(summary["common_validation_set_marginal_nll"]),
        "legality": float(summary["hard_legality_rate"]),
        "hit_at_1": hit_at_1,
        "on_top_k_recovery_at_10": float(
            summary["source_macro_measured_top_k_recovery_at_k"]
        ),
        "physical_gpu_index": summary.get("physical_gpu_index"),
        "validation_git_head": summary.get("validation_git_head"),
        "training_git_head": summary.get("training_git_head"),
        "precision": summary.get("precision"),
        "cpu_fallback_used": summary.get("cpu_fallback_used"),
        "data_source": "existing_frozen_validation_data",
    }


def _run_fallback_generation(
    checkpoint_pass: int,
    physical_gpu_index: int,
    scratch_root: Path,
    shared: dict,
) -> dict:
    """Regenerate a missing pass through the project validate entry (needs GPU)."""
    out_dir = scratch_root / f"pass_{checkpoint_pass}"
    if out_dir.exists():
        raise FileExistsError(f"A5 scratch output already exists: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(VALIDATE_ENTRY),
        "--config", str(CONFIG),
        "--run-id", RUN_ID,
        "--checkpoint-pass", str(checkpoint_pass),
        "--authorization", str(LAUNCH_AUTHORIZATION),
        "--physical-gpu-index", str(physical_gpu_index),
        "--output-dir", str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"A5 fallback generation failed (pass {checkpoint_pass}, rc={proc.returncode})"
        )
    summary_path = out_dir / "validation_summary.json"
    summary = _read_json(summary_path)
    candidates = _read_jsonl(out_dir / "trajectories.private.jsonl")
    hit_at_1 = _compute_hit_at_1(candidates, shared["manifest"], shared["measured"])
    row = _aggregate_existing_from_summary_and_trajectories(
        checkpoint_pass, summary, candidates, hit_at_1
    )
    row["data_source"] = "a5_fallback_regeneration"
    return row


def _aggregate_existing_from_summary_and_trajectories(
    checkpoint_pass: int, summary: dict, candidates, hit_at_1: float
) -> dict:
    return {
        "checkpoint_pass": checkpoint_pass,
        "status": summary.get("status"),
        "g0_status": summary.get("g0_status"),
        "source_count": summary.get("source_count"),
        "candidate_count": summary.get("candidate_count"),
        "recovery": float(summary["source_macro_candidate_recovery_rate"]),
        "unique_candidate_rate": float(summary["source_macro_unique_candidate_rate"]),
        "nll": float(summary["common_validation_set_marginal_nll"]),
        "legality": float(summary["hard_legality_rate"]),
        "hit_at_1": hit_at_1,
        "on_top_k_recovery_at_10": float(
            summary["source_macro_measured_top_k_recovery_at_k"]
        ),
        "physical_gpu_index": summary.get("physical_gpu_index"),
        "validation_git_head": summary.get("validation_git_head"),
        "training_git_head": summary.get("training_git_head"),
        "precision": summary.get("precision"),
        "cpu_fallback_used": summary.get("cpu_fallback_used"),
    }


def build_pass_row(
    checkpoint_pass: int,
    *,
    physical_gpu_index: int,
    scratch_root: Path,
    shared: dict,
    allow_regeneration: bool,
) -> dict:
    summary_path = VALIDATION_ROOT / f"pass_{checkpoint_pass}/validation_summary.json"
    if summary_path.exists():
        return _aggregate_existing_pass(checkpoint_pass, shared)
    if not allow_regeneration:
        return {
            "checkpoint_pass": checkpoint_pass,
            "status": "MISSING_AND_REGENERATION_NOT_ALLOWED",
            "recovery": None,
            "unique_candidate_rate": None,
            "nll": None,
            "legality": None,
            "hit_at_1": None,
            "data_source": "none",
        }
    return _run_fallback_generation(
        checkpoint_pass, physical_gpu_index, scratch_root, shared
    )


def _delta(a: dict, b: dict, key: str):
    if a.get(key) is None or b.get(key) is None:
        return None
    return round(float(a[key]) - float(b[key]), 6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--pass",
        dest="passes",
        type=int,
        nargs="+",
        default=None,
        help="checkpoint pass number(s); default = all saved passes (2,4,6)",
    )
    parser.add_argument(
        "--physical-gpu-index",
        type=int,
        default=0,
        help="GPU for any fallback regeneration (existing data path needs no GPU)",
    )
    parser.add_argument(
        "--allow-regeneration",
        action="store_true",
        help="if a pass has no frozen validation summary, regenerate it on GPU "
        "(default: false; report the pass as MISSING instead)",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "scratch_a5",
        help="scratch root for any fallback regeneration output",
    )
    args = parser.parse_args()

    passes = args.passes if args.passes else list(SAVED_PASSES)
    passes = sorted(set(passes))
    scratch_root = args.scratch_dir.resolve()

    shared = _rounds()
    rows = {}
    for p in passes:
        rows[str(p)] = build_pass_row(
            p,
            physical_gpu_index=args.physical_gpu_index,
            scratch_root=scratch_root,
            shared=shared,
            allow_regeneration=args.allow_regeneration,
        )

    baseline = rows.get(str(BASELINE_PASS))
    drift = {}
    if baseline is not None:
        for p in passes:
            if p == BASELINE_PASS:
                continue
            row = rows.get(str(p))
            if row is None:
                continue
            drift[str(p)] = {
                "recovery_delta_vs_pass2": _delta(row, baseline, "recovery"),
                "unique_candidate_rate_delta_vs_pass2": _delta(
                    row, baseline, "unique_candidate_rate"
                ),
                "nll_delta_vs_pass2": _delta(row, baseline, "nll"),
                "hit_at_1_delta_vs_pass2": _delta(row, baseline, "hit_at_1"),
                "legality_delta_vs_pass2": _delta(row, baseline, "legality"),
            }

    payload = {
        "schema_version": "route_a_v3_phaseA_a5_snapshot_stability.v1",
        "run_id": RUN_ID,
        "config_path": str(CONFIG),
        "baseline_pass": BASELINE_PASS,
        "saved_checkpoint_passes": [int(p) for p in SAVED_PASSES],
        "candidate_cap_per_source": 32,
        "aggregation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protected_reads": 0,
        "cpu_fallback_used": False,
        "passes": rows,
        "drift_vs_pass2_baseline": drift,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())