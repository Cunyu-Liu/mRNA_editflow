#!/usr/bin/env python3
"""Option 2 committee arm, step A: re-run A3 mixed-pool probe with per-source
detail dropped to disk (the original phaseA probe only persisted aggregates).

Identical pool construction, identical metrics, identical checkpoints as the
frozen reference rows (mixed_pool_probe_reference_v1.json). The ONLY changes:
  1. --per-source-out writes {source_key: {..., task, critic acc, base acc}}
  2. runs on a MIG slice (GPU 6/7 3g.20gb, currently idle)

Preregistration discipline: this script is committed BEFORE any new probe data
is generated (patch-first-then-run), so the dev/holdout split + routing
selection protocol (step B) can be registered against the committed code.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
)

# import the frozen probe implementation (pool construction + metrics) from the
# v8_stage1_prep worktree copy -- single source of truth for the calibre.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util as _ilu

_PROBE = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_v8_stage1_prep_20260904/"
    "scripts/route_a_v3/evaluate_route2_mixed_pool_probe_v1.py"
)
_spec = _ilu.spec_from_file_location("a3_probe_frozen_mod", _PROBE)
_mod = _ilu.module_from_spec(_spec)
sys.modules["a3_probe_frozen_mod"] = _mod
_spec.loader.exec_module(_mod)

_read_jsonl = _mod._read_jsonl
_norm = _mod._norm
source_metrics = _mod.source_metrics
CriticScorer = _mod.CriticScorer
StubScorer = _mod.StubScorer
MANIFEST = _mod.MANIFEST
MEASURED = _mod.MEASURED
UNGU = _mod.UNGU
END2TASK = _mod.END2TASK

SCHEMA = "route_a_v3_route2_option2_committee_probe_persource.v1"

CHECKPOINTS = {
    "full": (
        "v5",
        Path(
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_v5/"
            "v5_screen_seed_20260907_runner_1113cd2c0dd9acb508f58782eecb40f458d2cab3/"
            "v5_full/final_pass_8_checkpoint.pt"
        ),
    ),
    "v8_hbench9": (
        "v8",
        Path(
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/"
            "xeditcritic_route_a/v8_stage2_adapt_20260907/h_bench9/"
            "stage2_h_benchmark_full_epoch6.pt"
        ),
    ),
    "v8_smprau_in": (
        "v8",
        Path(
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/"
            "xeditcritic_route_a/v8_stage2_adapt_20260907/s_mprau_in/"
            "stage2_s_benchmark_full_epoch6.pt"
        ),
    ),
    "v8_smrlpolya": (
        "v8",
        Path(
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/"
            "xeditcritic_route_a/v8_stage1_joint_prefinetune_20260904/"
            "s_mrl-polya/stage1_s_epoch2.pt"
        ),
    ),
}

OUT_ROOT = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/"
    "analysis_option2_committee_20260909"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", required=True, choices=sorted(CHECKPOINTS))
    ap.add_argument("--physical-gpu-index", type=int, default=0)
    ap.add_argument("--dry-stub", action="store_true")
    ap.add_argument(
        "--allow-off-domain-collisions",
        action="store_true",
        help=(
            "s_mprau_in is known near-blind off-domain (per-task 0.008-0.03 "
            "probe acc, near-constant scores can collide bit-identically in "
            "BF16); allow ties and record them -- differs from the N5 whole-"
            "string-UNK bug (which produces CONSTANT scores for ALL candidates "
            "including on-domain) in that only off-domain sources collide"
        ),
    )
    args = ap.parse_args()

    kind, ckpt = CHECKPOINTS[args.which]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stem = f"persrc_probe_{args.which}"
    out_json = OUT_ROOT / f"{stem}.json"
    out_persrc = OUT_ROOT / f"{stem}.per_source.jsonl"

    manifest_rows = _read_jsonl(MANIFEST)
    manifest = {}
    for row in manifest_rows:
        sk = str(row["source_key"])
        manifest[sk] = {
            "source_sequence": _norm(row["source_sequence"]),
            "endpoint_id": str(row["endpoint_id"]),
            "region": str(row["region"]).replace("′", "").replace("'", ""),
            "assay_id": str(row["assay_id"]),
            "biological_context_id": str(row["biological_context_id"]),
            "_row": row,
        }

    measured = defaultdict(dict)
    for row in _read_jsonl(MEASURED):
        sk = str(row["source_key"])
        if row.get("pool_assignment") != "DEVELOPMENT":
            continue
        measured[sk][_norm(row["candidate_sequence"])] = float(
            row["measured_direction_normalized_delta"]
        )

    base_pool = defaultdict(dict)
    for row in _read_jsonl(UNGU):
        sk = str(row["source_key"])
        seq = _norm(row["candidate_sequence"])
        s = float(row["generation_score"])
        if seq not in base_pool[sk] or s > base_pool[sk][seq]:
            base_pool[sk][seq] = s

    sources = sorted(manifest)

    collision_counter = {"collision_groups": 0}
    if args.dry_stub:
        scorer = StubScorer()
        status = "DRY_SMOKE_STUB"
    else:
        if args.allow_off_domain_collisions:
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))
            import scripts.route_a_v3.route2_v8_frozen_guidance_v1 as _v8mod

            _orig_require = _v8mod._require

            def _tolerant_require(condition: bool, message: str) -> None:
                if not condition and "identical potentials" in message:
                    collision_counter["collision_groups"] += 1
                    return
                _orig_require(condition, message)

            _v8mod._require = _tolerant_require
        scorer = CriticScorer(
            ckpt,
            Path(
                "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/"
                "mrnabert_a1eb7df25804d23f08646e1cb996b234d7208a40"
            ),
            args.physical_gpu_index,
            kind=kind,
        )
        status = "TERMINAL"

    per_source: dict[str, Any] = {}
    for sk in sources:
        src = manifest[sk]
        source = src["source_sequence"]
        meas = measured.get(sk, {})
        if not meas:
            per_source[sk] = {"n_measured": 0, "skip": "no_measured"}
            continue
        base_seq_scores = dict(base_pool.get(sk, {}))
        mixed_seqs = list(dict.fromkeys([*base_seq_scores.keys(), *meas.keys()]))
        critic_scores = dict(
            zip(
                mixed_seqs,
                scorer.score(
                    source,
                    src["assay_id"],
                    src["biological_context_id"],
                    src["endpoint_id"],
                    src["region"],
                    src["_row"],
                    mixed_seqs,
                ),
                strict=True,
            )
        )
        if base_seq_scores:
            base_floor = min(base_seq_scores.values()) - 1.0
            base_scores = {
                seq: base_seq_scores.get(seq, base_floor) for seq in mixed_seqs
            }
        else:
            base_scores = {seq: -1.0 * (i + 1) for i, seq in enumerate(mixed_seqs)}
        c = source_metrics(critic_scores, meas)
        b = source_metrics(base_scores, meas)
        per_source[sk] = {
            "task": END2TASK.get(src["endpoint_id"], src["endpoint_id"]),
            "endpoint_id": src["endpoint_id"],
            "n_measured": len(meas),
            "pool_size": len(mixed_seqs),
            "critic_conditional_rank_acc@1": c["conditional_rank_acc@1"],
            "base_conditional_rank_acc@1": b["conditional_rank_acc@1"],
        }

    def macro(rows: list[dict[str, Any]], key: str) -> float:
        vals = [
            r[key]
            for r in rows
            if key in r
            and not (isinstance(r[key], float) and math.isnan(r[key]))
        ]
        return float(np.mean(vals)) if vals else float("nan")

    measured_rows = [r for r in per_source.values() if r.get("n_measured", 0) > 0]
    report = {
        "schema_version": SCHEMA,
        "status": status,
        "protected_reads": 0,
        "precision": "STUB" if args.dry_stub else "BF16",
        "device": "DRY_STUB_(no_model)" if args.dry_stub else f"cuda:{args.physical_gpu_index}",
        "critic_checkpoint": str(ckpt),
        "critic_kind": kind,
        "which": args.which,
        "pool_sources_with_measured": len(measured_rows),
        "source_count": len(sources),
        "overall": {
            "critic_conditional_rank_acc@1": macro(
                measured_rows, "critic_conditional_rank_acc@1"
            ),
            "base_conditional_rank_acc@1": macro(
                measured_rows, "base_conditional_rank_acc@1"
            ),
        },
        "per_source_out": str(out_persrc),
        "notes": [
            "identical pools/metrics/checkpoints as frozen reference rows "
            "(mixed_pool_probe_reference_v1.json); re-run solely to persist "
            "per-source records for the option-2 dev/holdout routing protocol",
        ],
    }
    if args.allow_off_domain_collisions and not args.dry_stub:
        scoring_calls = report.get("scorer", {}).get("source_scoring_calls", 0)
        # guard against the REAL whole-string-UNK fingerprint: near-constant
        # scores on EVERY group (including on-domain). Off-domain near-blind
        # ties for s_mprau_in collide on a minority of groups only.
        frac = (
            collision_counter["collision_groups"] / scoring_calls
            if scoring_calls
            else 1.0
        )
        report["collision_groups"] = collision_counter["collision_groups"]
        report["collision_group_fraction"] = frac
        if frac > 0.5:
            out_json.write_text(
                json.dumps(
                    {
                        "status": "REJECTED_WHOLE_POOL_CONSTANT",
                        "collision_group_fraction": frac,
                    },
                    indent=1,
                )
                + "\n"
            )
            print(
                "REJECTED: collision fraction "
                f"{frac:.3f} > 0.5 (whole-pool-constant fingerprint, N5)"
            )
            return 1
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with out_persrc.open("w", encoding="utf-8") as fh:
        for sk in sorted(per_source):
            fh.write(
                json.dumps({"source_key": sk, **per_source[sk]}, sort_keys=True) + "\n"
            )
    print(json.dumps({k: v for k, v in report.items() if k != "notes"}, indent=1))
    print("wrote", out_json, "and", out_persrc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
