#!/usr/bin/env python3
"""D15-2 polyA structure-directed sampling test (FULL PIPELINE, launched per user decision D18-A).

Arms (891-cohort polyA sources, 20 sources):
  arm1 oracle-structured: sample (n_edits, window-block) from measured structure
       profile, random bases in window, budget = measured n_edits (per-sample)
  arm2 local-window uniform: fixed budget 8, W=16 window, uniform positions/bases
Both arms scored by V5 critic (official frozen checkpoint); metrics:
  - graded support: pool contains candidate with same (n_edits, window-block)
    structure key as a measured candidate
  - graded sc-hit@1: among graded-supported sources, V5-critic top-1 ==
    argmax measured (within measured candidates of same structure key)
  - exact-match support (reference, expected ~0 per E8c)
Discipline: inference only, VALIDATION-only reporting, zero training, protected reads = 0.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
import torch

sys.path.insert(0, "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901")

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
PROJ = MNT / "projections/xedit_v3/development_train_validation_v1/validation.jsonl"
OUT = MNT / "experiments/analysis_d15_2_structured_sampling"
OUT.mkdir(parents=True, exist_ok=True)

V5_CKPT = MNT / "experiments/xeditcritic_v5/v5_screen_seed_20260907_runner_1113cd2c0dd9acb508f58782eecb40f458d2cab3/v5_full/final_pass_8_checkpoint.pt"
MRNABERT = MNT / "external_model_assets/mrnabert_a1eb7df25804d23f08646e1cb996b234d7208a40"
BASES = "ACGT"
N_SAMPLES = 32

class FakeState:
    def __init__(self, src, cand, assay, ctx):
        self.source_sequence = src
        self.current_sequence = cand
        self.assay_id = assay
        self.context_id = ctx

def main():
    from scripts.route_a_v3.route2_xeditcritic_v5_frozen_guidance_v1 import FrozenXEditCriticV5
    device = torch.device("cuda:3")
    critic = FrozenXEditCriticV5(V5_CKPT, MRNABERT, device)

    # B2 891 cohort polyA sources
    base = MNT / "experiments/xeditsetflow_v5/guided_b2_20260903/b2_full_891/guided/generated_candidates.private.jsonl"
    cohort_src_keys = set()
    with open(base) as f:
        for line in f:
            d = json.loads(line)
            if "POLYA" in d["source_key"]:
                cohort_src_keys.add(d["source_key"])

    polya = []
    with open(PROJ) as f:
        for line in f:
            d = json.loads(line)
            if d["task_id"] == "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS::region=1":
                polya.append(d)
    by_src = defaultdict(list)
    for r in polya:
        by_src[r["source_id"]].append(r)

    def struct_key(r):
        es = sorted(e.get("position", 0) for e in r.get("source_relative_edits", []))
        if not es:
            return None
        return (len(es), es[0] // 8, es[-1] // 8)

    rng = np.random.RandomState(20260912)

    def run_arm(arm_name, sampler):
        rows_out = []
        n_graded_sup = 0
        n_exact = 0
        n_eval_hit1 = 0
        n_correct_hit1 = 0
        for sid, rs in sorted(by_src.items()):
            s = rs[0]["source_sequence"].upper().replace("U", "T")
            L = len(s)
            keys = {struct_key(r) for r in rs if struct_key(r)}
            assay = rs[0]["assay_id"]
            ctx = rs[0]["biological_context_id"]
            src_row = dict(rs[0])
            src_row.setdefault("source_key", str(sid))
            # generate N_SAMPLES candidates
            cands = [sampler(s, L, rng) for _ in range(N_SAMPLES)]
            # graded support
            graded_keys = set()
            for c in cands:
                es = [p for p in range(min(len(s), len(c))) if s[p] != c[p]]
                if es:
                    gk = (len(es), es[0] // 8, es[-1] // 8)
                    graded_keys.add(gk)
            gs = bool(graded_keys & keys)
            exact = any(c in {r["candidate_sequence"] for r in rs} for c in cands)
            if gs:
                n_graded_sup += 1
            if exact:
                n_exact += 1
            # critic ranking among graded-supported: rank pool by critic, check top-1 among measured
            if gs:
                states = [FakeState(s.replace("T", "U"), c.replace("T", "U"), assay, ctx) for c in cands]
                try:
                    pots = critic.potentials(states, endpoint_id="PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS", region="3UTR", source_row=src_row)
                except Exception as e:
                    continue
                top = cands[int(np.argmax(pots))]
                # measured best within matched structure keys
                matched_measured = [r for r in rs if struct_key(r) in graded_keys]
                if matched_measured:
                    best_m = max(matched_measured, key=lambda r: r["direction_normalized_delta"])
                    n_eval_hit1 += 1
                    if top == best_m["candidate_sequence"]:
                        n_correct_hit1 += 1
        res = {
            "arm": arm_name,
            "n_sources": len(by_src),
            "graded_support": n_graded_sup,
            "graded_support_rate": n_graded_sup / len(by_src),
            "exact_match_support": n_exact,
            "graded_sc_hit1": (n_correct_hit1 / n_eval_hit1) if n_eval_hit1 else None,
            "n_eval_hit1": n_eval_hit1,
        }
        return res

    def sampler_oracle(s, L, rng):
        # structure from this source's own measured profile (oracle for structure, random bases)
        rs = by_src_of(s)
        keys = [struct_key(r) for r in rs if struct_key(r)]
        k = keys[rng.randint(len(keys))]
        n_e, blk_lo, blk_hi = k
        lo, hi = blk_lo * 8, min(L, (blk_hi + 1) * 8)
        if hi - lo < n_e:
            lo = max(0, hi - n_e)
        pos = rng.choice(np.arange(lo, hi), size=n_e, replace=False)
        cand = list(s)
        for p in pos:
            alts = [b for b in BASES if b != s[int(p)]]
            cand[int(p)] = alts[rng.randint(3)]
        return "".join(cand)

    def sampler_local8(s, L, rng):
        n_e = 8
        W = 16
        start = rng.randint(0, max(1, L - W))
        lo, hi = start, min(L, start + W)
        if hi - lo < n_e:
            lo = max(0, hi - n_e)
        pos = rng.choice(np.arange(lo, hi), size=n_e, replace=False)
        cand = list(s)
        for p in pos:
            alts = [b for b in BASES if b != s[int(p)]]
            cand[int(p)] = alts[rng.randint(3)]
        return "".join(cand)

    src_to_rows = {}
    def by_src_of(s):
        if s not in src_to_rows:
            for sid, rs in by_src.items():
                if rs[0]["source_sequence"].upper().replace("U", "T") == s:
                    src_to_rows[s] = rs
                    break
            else:
                src_to_rows[s] = []
        return src_to_rows[s]

    arm1 = run_arm("oracle_structured", sampler_oracle)
    print(json.dumps(arm1, indent=1))
    arm2 = run_arm("local8_W16", sampler_local8)
    print(json.dumps(arm2, indent=1))

    out = {"schema": "d15_2_full_pipeline.v1", "date": "2026-09-12",
           "protocol": "V5-critic ranking of structure-directed samples; graded support metric",
           "arm1_oracle_structured": arm1, "arm2_local8_W16": arm2,
           "reference_exact_match_E8c": 0.0}
    with open(OUT / "d15_2_full_pipeline.json", "w") as f:
        json.dump(out, f, indent=1)
    print("saved:", OUT / "d15_2_full_pipeline.json")

if __name__ == "__main__":
    main()
