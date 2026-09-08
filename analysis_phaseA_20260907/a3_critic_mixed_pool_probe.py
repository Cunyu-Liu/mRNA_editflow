#!/usr/bin/env python3
"""A3: critic mixed-pool discrimination probe (SetFlow V5 generator vs frozen XEditCritic V5).

H2 direct measurement: does the V5 critic, given a MIXED pool of a source's
unguided base-generated candidates PLUS its real measured candidates, rank the
TRUE-BEST measured candidate to the top-1 better than the base model's own
implicit ordering (generation_score)?

Pool per source (a.k.a. "mixed pool"):
    unguided 32 generated candidates (dedup, keep max generation_score)
    UNION all real measured candidates for the source (from DEVELOPMENT
    measured_neighborhood; measured seqs not present in the generated output are
    ADDED -- they are the "un-hit" measured candidates).
Dedupe to unique sequences. Pool size ~32 .. ~36 per source.

Scorers:
  - V5 critic  : FrozenXEditCriticV5.potentials(states, endpoint_id, region,
                 source_row), CUDA BF16 only (project convention; no CPU fallback).
  - base-self  : rank the SAME mixed pool by the B2 unguided generation_score;
                 measured ADDED sequences (not among the generator's outputs)
                 have no base score -> assigned min(gen_score)-1 so they sort to
                 the bottom (the base model never proposed them). This is the
                 honest baseline: base can only surface measured candidates its
                 own generator happened to emit.

Metrics per source (source-macro = unweighted mean over sources), k=1 tie-aware
(same semantics as A1 hit@1):
  - conditional_rank_acc@1 : among sources with >=1 measured in pool (here ALL,
        since measured are added), fraction where the true-best measured is
        ADMITTED into the top-1 tie block (inclusion prob 1/|block| per member).
        == the probability that a random member of the top-1 tie block is the
        true-best measured.
  - measured_best_avg_rank : mean over sources of the block-start rank (1-based)
        of the highest-ranked measured candidate (most favorable tie handling).

Support-32 subset: sources where the generated pool NATURALLY contains >=1
measured (A2: overall 24.2%). Reported for apples-to-apples critic vs base on
the same naturally-covered set.

--dry-stub mode: replaces the critic with a deterministic hash-based stub so the
full data/plumbing/metric/JSON pipeline can be validated WITHOUT any model
forward (status=DRY_SMOKE_STUB, NON-scientific; real BF16 run needs a free GPU).

Discipline: protected reads = 0 (only manifest structure + measured sequence
identities + measured_direction_normalized_delta to resolve true-best, per the
A1/A2 convention; NO TEST/EVALUATION outcome file is ever opened).
"""

from __future__ import annotations

import argparse
import hashlib
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

MANIFEST = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/generation_eligibility/"
    "development_validation_v1/source_eligibility.jsonl"
)
MEASURED = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/generation_eligibility/"
    "development_validation_v1/measured_neighborhood.private.jsonl"
)
UNGU = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditsetflow_v5/"
    "guided_b2_20260903/b2_full_891/unguided/generated_candidates.private.jsonl"
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
REWARD_POLICY = REPO_ROOT / "configs/route_a_v3_route2_mrnabert_guidance_reward_policy_v1.json"

END2TASK = {
    "MEAN_RIBOSOME_LOAD": "MRL",
    "MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE": "MPRAU",
    "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS": "polyA",
    "RNA_HALF_LIFE_MINUTES": "HL",
}

SCHEMA = "route_a_v3_phaseA_a3_critic_mixed_pool_probe.v1"


def _norm(seq: Any) -> str:
    return str(seq).upper().replace("T", "U")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class CriticScorer:
    """Real frozen XEditCritic V5 (CUDA BF16 only). Lazily imports deps."""

    def __init__(
        self, checkpoint: Path, mrnabert: Path, gpu_index: int, *, kind: str = "v5"
    ) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("critic requires CUDA; cannot CPU-fallback")
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from core.route2_legal_xeditflow import FlowState

        transform = json.loads(REWARD_POLICY.read_text())["potential_transform"]
        self._FlowState = FlowState
        if kind == "v8":
            from scripts.route_a_v3.route2_v8_frozen_guidance_v1 import FrozenV8Critic

            self.critic = FrozenV8Critic(
                checkpoint,
                mrnabert,
                torch.device(f"cuda:{gpu_index}"),
                potential_minimum=float(transform["minimum"]),
                potential_maximum=float(transform["maximum"]),
            )
            self.kind = "frozen_xeditcritic_v8"
        elif kind == "v5":
            from scripts.route_a_v3.route2_xeditcritic_v5_frozen_guidance_v1 import (
                FrozenXEditCriticV5,
            )

            self.critic = FrozenXEditCriticV5(
                checkpoint,
                mrnabert,
                torch.device(f"cuda:{gpu_index}"),
                potential_minimum=float(transform["minimum"]),
                potential_maximum=float(transform["maximum"]),
            )
            self.kind = "frozen_xeditcritic_v5"
        else:
            raise RuntimeError(f"unknown critic kind: {kind!r}")
        self.count = 0

    def score(
        self,
        source: str,
        assay: str,
        context: str,
        endpoint_id: str,
        region_key: str,
        source_row: Mapping[str, Any],
        sequences: Sequence[str],
    ) -> list[float]:
        FlowState = self._FlowState
        states = [
            FlowState(
                source_sequence=source,
                current_sequence=seq,
                source_relative_edits=(),
                remaining_budget=0,
                assay_id=assay,
                context_id=context,
            )
            for seq in sequences
        ]
        self.count += 1
        return list(
            self.critic.potentials(
                states,
                endpoint_id=endpoint_id,
                region=region_key,
                source_row=source_row,
            )
        )


class StubScorer:
    """Deterministic hash-based stub for --dry-stub (plumbing validation only)."""

    def __init__(self, seed: int = 20260907) -> None:
        self.seed = seed
        self.count = 0

    def _score(self, source: str, seq: str) -> float:
        digest = hashlib.sha256(f"{self.seed}:{source}:{seq}".encode()).digest()
        return (int.from_bytes(digest[:8], "big") % 10000) / 9999.0

    def score(
        self,
        source: str,
        assay: str,
        context: str,
        endpoint_id: str,
        region_key: str,
        source_row: Mapping[str, Any],
        sequences: Sequence[str],
    ) -> list[float]:
        self.count += 1
        return [self._score(source, seq) for seq in sequences]


def _desc_tie_blocks(scores: Mapping[str, float]) -> list[list[str]]:
    """Descending-order tie blocks of sequence labels by score."""
    items = sorted(scores.items(), key=lambda kv: (-float(kv[1]), kv[0]))
    blocks: list[list[str]] = []
    for seq, score in items:
        if blocks and abs(blocks[-1][1] - score) < 1e-12:
            blocks[-1][0].append(seq)
        else:
            blocks.append(([seq], score))  # type: ignore[list-item]
    return [block for block, _ in blocks]  # type: ignore[misc]


def source_metrics(
    scores: Mapping[str, float],
    true_val: Mapping[str, float],
) -> dict[str, float]:
    """Tie-aware k=1 metrics for one source ordering."""
    true_best_val = max(true_val.values()) if true_val else float("NaN")
    true_top = {
        seq for seq, val in true_val.items() if float(val) == float(true_best_val)
    }
    # conditional_rank_acc@1 : top-1 tie block admission of true-best measured
    if not scores:
        acc = 0.0
        best_rank = float("nan")
    else:
        top_score = max(scores.values())
        top_block = [seq for seq, s in scores.items() if float(s) == float(top_score)]
        acc = sum(
            1.0 / len(top_block) for seq in top_block if seq in true_top
        )
        # measured best rank = block-start rank of highest-ranked measured
        measured_scores = {
            seq: s for seq, s in scores.items() if seq in true_val
        }
        if not measured_scores:
            best_rank = float("nan")
        else:
            best_meas_score = max(measured_scores.values())
            best_rank = 1.0 + sum(
                1 for seq, s in scores.items() if float(s) > float(best_meas_score)
            )
    return {
        "conditional_rank_acc@1": float(acc),
        "measured_best_rank": float(best_rank),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--physical-gpu-index", type=int, default=0)
    ap.add_argument("--critic-checkpoint", type=Path, default=DEFAULT_CRITIC_CHECKPOINT)
    ap.add_argument("--mrnabert-model", type=Path, default=DEFAULT_MRNABERT_MODEL)
    ap.add_argument(
        "--critic-kind", choices=("v5", "v8"), default="v5",
        help="frozen critic family (v5 = XEditCritic V5; v8 = FrozenV8Critic)",
    )
    ap.add_argument("--dry-stub", action="store_true",
                    help="use deterministic hash stub (validates plumbing only; NON-scientific)")
    ap.add_argument("--cohort", type=int, default=0,
                    help=">0 caps number of sources (non-terminal smoke)")
    args = ap.parse_args()

    manifest_rows = _read_jsonl(MANIFEST)
    manifest = {}
    for row in manifest_rows:
        sk = str(row["source_key"])
        manifest[sk] = {
            "source_sequence": _norm(row["source_sequence"]),
            "endpoint_id": str(row["endpoint_id"]),
            "region": str(row["region"]).replace("′", "").replace("'", ""),
            "edit_budget": int(row["edit_budget"]),
            "assay_id": str(row["assay_id"]),
            "biological_context_id": str(row["biological_context_id"]),
            "measured_candidate_count": int(row["measured_candidate_count"]),
            "candidate_budget": int(row["candidate_budget"]),
            "_row": row,
        }

    # measured: sequence membership per source + true value (DEVELOPMENT label)
    measured = defaultdict(dict)
    for row in _read_jsonl(MEASURED):
        sk = str(row["source_key"])
        if row.get("pool_assignment") != "DEVELOPMENT":
            continue
        seq = _norm(row["candidate_sequence"])
        measured[sk][seq] = float(row["measured_direction_normalized_delta"])

    # unguided base pool: source -> {seq: max generation_score}
    base_pool = defaultdict(dict)
    for row in _read_jsonl(UNGU):
        sk = str(row["source_key"])
        seq = _norm(row["candidate_sequence"])
        s = float(row["generation_score"])
        if seq not in base_pool[sk] or s > base_pool[sk][seq]:
            base_pool[sk][seq] = s

    sources = sorted(manifest)
    if args.cohort > 0:
        sources = sources[: args.cohort]

    if args.dry_stub:
        scorer = StubScorer()
        status = "DRY_SMOKE_STUB"
    else:
        scorer = CriticScorer(args.critic_checkpoint, args.mrnabert_model,
                              args.physical_gpu_index, kind=args.critic_kind)
        status = "TERMINAL" if args.cohort == 0 else "SMOKE_NON_TERMINAL"

    per_source: dict[str, Any] = {}
    for sk in sources:
        src = manifest[sk]
        source = src["source_sequence"]
        meas = measured.get(sk, {})
        if not meas:
            # should not happen on VALIDATION; record and skip metric
            per_source[sk] = {"n_measured": 0, "skip": "no_measured"}
            continue
        base_seq_scores = dict(base_pool.get(sk, {}))
        # mixed pool: generated (with base score) + measured seqs (added)
        mixed_seqs = list(dict.fromkeys([*base_seq_scores.keys(), *meas.keys()]))
        # critic scores
        critic_scores = dict(zip(
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
        ))
        # base-self scores: measured-added get min(gen)-1 so they sort to bottom
        if base_seq_scores:
            base_floor = min(base_seq_scores.values()) - 1.0
            base_scores = {seq: base_seq_scores[seq] if seq in base_seq_scores
                           else base_floor for seq in mixed_seqs}
        else:
            base_scores = {seq: -1.0 * (i + 1) for i, seq in enumerate(mixed_seqs)}

        c = source_metrics(critic_scores, meas)
        b = source_metrics(base_scores, meas)
        nat_hit = bool(set(base_seq_scores) & set(meas))
        per_source[sk] = {
            "endpoint_id": src["endpoint_id"],
            "task": END2TASK.get(src["endpoint_id"], src["endpoint_id"]),
            "n_measured": len(meas),
            "pool_size": len(mixed_seqs),
            "generated_size": len(base_seq_scores),
            "generated_measured_hits": len(set(base_seq_scores) & set(meas)),
            "natural_hit_32": nat_hit,
            "true_best_is_added": (lambda best_val: not any(
                s in base_seq_scores for s, v in meas.items() if v == best_val
            ))(max(meas.values())),
            "critic_conditional_rank_acc@1": c["conditional_rank_acc@1"],
            "critic_measured_best_rank": c["measured_best_rank"],
            "base_conditional_rank_acc@1": b["conditional_rank_acc@1"],
            "base_measured_best_rank": b["measured_best_rank"],
        }

    def macro(rows: list[dict[str, Any]], key: str) -> float:
        vals = [r[key] for r in rows if key in r and not (
            isinstance(r[key], float) and math.isnan(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    def task_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_task[r["task"]].append(r)
        out = {}
        for task in ["MRL", "MPRAU", "HL", "polyA"]:
            rs = by_task.get(task, [])
            entries = [r for r in rs if r.get("n_measured", 0) > 0]
            if not entries:
                out[task] = {"source_count": len(rs), "support_measured": 0,
                             "note": "no measured candidates in pool (support=0)"}
                continue
            nat = [r for r in entries if r["natural_hit_32"]]
            out[task] = {
                "source_count": len(rs),
                "pool_sources_with_measured": len(entries),
                "natural_hit_32_count": len(nat),
                "natural_hit_32_frac": len(nat) / len(entries) if entries else 0.0,
                "critic_conditional_rank_acc@1": macro(entries, "critic_conditional_rank_acc@1"),
                "critic_measured_best_avg_rank": macro(entries, "critic_measured_best_rank"),
                "base_conditional_rank_acc@1": macro(entries, "base_conditional_rank_acc@1"),
                "base_measured_best_avg_rank": macro(entries, "base_measured_best_rank"),
                "critic_cond_acc_on_natural_hit": (
                    macro(nat, "critic_conditional_rank_acc@1") if nat else float("nan")
                ),
                "base_cond_acc_on_natural_hit": (
                    macro(nat, "base_conditional_rank_acc@1") if nat else float("nan")
                ),
            }
        return out

    measured_rows = [r for r in per_source.values() if r.get("n_measured", 0) > 0]
    nat_rows = [r for r in measured_rows if r["natural_hit_32"]]

    report = {
        "schema_version": SCHEMA,
        "status": status,
        "protected_reads": 0,
        "precision": "STUB" if args.dry_stub else "BF16",
        "device": "DRY_STUB_(no_model)" if args.dry_stub else f"cuda:{args.physical_gpu_index}",
        "critic_checkpoint": str(args.critic_checkpoint),
        "pool_sources_with_measured": len(measured_rows),
        "source_count": len(sources),
        "overall": {
            "critic_conditional_rank_acc@1": macro(measured_rows, "critic_conditional_rank_acc@1"),
            "critic_measured_best_avg_rank": macro(measured_rows, "critic_measured_best_rank"),
            "base_conditional_rank_acc@1": macro(measured_rows, "base_conditional_rank_acc@1"),
            "base_measured_best_avg_rank": macro(measured_rows, "base_measured_best_rank"),
        },
        "natural_hit_32_sources": {
            "count": len(nat_rows),
            "fraction_of_measured_sources": len(nat_rows) / len(measured_rows) if measured_rows else 0.0,
            "critic_conditional_rank_acc@1": macro(nat_rows, "critic_conditional_rank_acc@1"),
            "critic_measured_best_avg_rank": macro(nat_rows, "critic_measured_best_rank"),
            "base_conditional_rank_acc@1": macro(nat_rows, "base_conditional_rank_acc@1"),
            "base_measured_best_avg_rank": macro(nat_rows, "base_measured_best_rank"),
        },
        "per_task": task_table(measured_rows),
        "scorer": {
            "kind": "stub_hash" if args.dry_stub else scorer.kind,
            "source_scoring_calls": scorer.count,
        },
        "notes": [
            "mixed pool = unguided 32 generated (dedup,max genscore) UNION all measured; ",
            "measured added seqs receive base floor score (min-1) for the base-self baseline ",
            "(the base generator never proposed them); pools identical for critic and base.",
            "conditional_rank_acc@1 == tie-aware hit@1 at k=1 (top-1 tie block admission ",
            "probability of the true-best measured).",
            "measured_best_avg_rank == mean block-start rank (1-based) of the highest-ranked measured.",
            "on-manifold rho reference (A2/V5): polyA 0.8219 / MRL 0.1354 / MPRAU 0.1025 / HL ~0.",
            "n_measured distribution on VALIDATION: {2:317,3:481,4:93} (mean 2.75/source).",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())