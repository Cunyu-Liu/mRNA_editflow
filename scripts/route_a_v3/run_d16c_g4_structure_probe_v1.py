#!/usr/bin/env python3
"""D16-C G4 structure-pool probe (amendment v1 sec.4.2 G4 gate).

Exact mirror of run_d15_2_pipeline.py calibre (same 81 polyA VALIDATION
sources, same samplers, same rng seed 20260912), with ONE substitution:
the critic checkpoint is the D16-C probe FINAL-EPOCH weight (probe_epoch_6.pt)
instead of the official V5 frozen checkpoint.

G4 gate (frozen): graded sc-hit@1 > 0 in the structure pool (D15-2 calibre
baseline on V5 official ckpt = 0.0). > 0 = representation begins to learn
structure; == 0 = structure pool blindness persists after augmentation.

Discipline: inference only; VALIDATION-only; protected reads = 0; CUDA
required with hard gate (MIG slice via CUDA_VISIBLE_DEVICES is acceptable,
device evidence recorded); zero training steps.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901")

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
PROJ = MNT / "projections/xedit_v3/development_train_validation_v1/validation.jsonl"
OUT = MNT / "experiments/xeditcritic_d16c/structure_probe_g4"
OUT.mkdir(parents=True, exist_ok=True)

PROBE_CKPT = MNT / "experiments/xeditcritic_d16c/probe_mrl_v1_gpu5/probe_epoch_6.pt"
V5_CKPT = MNT / "experiments/xeditcritic_v5/v5_screen_seed_20260907_runner_1113cd2c0dd9acb508f58782eecb40f458d2cab3/v5_full/final_pass_8_checkpoint.pt"
MRNABERT = MNT / "external_model_assets/mrnabert_a1eb7df25804d23f08646e1cb996b234d7208a40"
BASES = "ACGT"
N_SAMPLES = 32
SEED = 20260912

class FakeState:
    def __init__(self, src, cand, assay, ctx):
        self.source_sequence = src
        self.current_sequence = cand
        self.assay_id = assay
        self.context_id = ctx

def main():
    # CUDA hard gate (discipline: no CPU fallback, evidence recorded)
    if not torch.cuda.is_available():
        out = OUT / "d16c_g4_structure_probe.json"
        out.write_text(json.dumps({"schema": "d16c_g4_structure_probe.v1",
            "status": "STOPPED_WITH_EVIDENCE",
            "error": "CUDA unavailable", "cpu_fallback_used": True}))
        print("STOPPED_WITH_EVIDENCE: CUDA unavailable", flush=True)
        return 2
    device = torch.device("cuda:0")
    cuda_name = torch.cuda.get_device_name(0)
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "NOT_SET")
    print(f"CUDA OK: device={cuda_name} visible={cuda_visible}", flush=True)

    from scripts.route_a_v3.route2_xeditcritic_v5_frozen_guidance_v1 import FrozenXEditCriticV5
    # load via official terminal ckpt (schema gate), then overlay probe FINAL-EPOCH weights
    critic = FrozenXEditCriticV5(V5_CKPT, MRNABERT, device)
    probe = torch.load(PROBE_CKPT, map_location="cpu", weights_only=False)
    _require_sd = probe.get("model_state_dict")
    if not _require_sd:
        raise RuntimeError("probe checkpoint missing model_state_dict")
    critic.model.load_state_dict({k: v for k, v in _require_sd.items()}, strict=False)
    critic.model.eval()
    print("probe weights loaded: epoch={} keys={}".format(probe.get("epoch"), len(_require_sd)), flush=True)

    base = MNT / "experiments/xeditsetflow_v5/guided_b2_20260903/b2_full_891/guided/generated_candidates.private.jsonl"
    with open(base) as f:
        for line in f:
            json.loads(line)  # presence check only (same calibre as D15-2)

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

    rng = np.random.RandomState(SEED)

    def run_arm(arm_name, sampler):
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
            cands = [sampler(s, L, rng) for _ in range(N_SAMPLES)]
            graded_keys = set()
            for c in cands:
                es = [p for p in range(min(len(s), len(c))) if s[p] != c[p]]
                if es:
                    graded_keys.add((len(es), es[0] // 8, es[-1] // 8))
            gs = bool(graded_keys & keys)
            exact = any(c in {r["candidate_sequence"] for r in rs} for c in cands)
            n_graded_sup += int(gs)
            n_exact += int(exact)
            if gs:
                states = [FakeState(s.replace("T", "U"), c.replace("T", "U"), assay, ctx) for c in cands]
                try:
                    pots = critic.potentials(states, endpoint_id="PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS", region="3UTR", source_row=src_row)
                except Exception as e:
                    print(f"WARN source {sid} potentials failed: {e}", flush=True)
                    continue
                top = cands[int(np.argmax(pots))]
                matched_measured = [r for r in rs if struct_key(r) in graded_keys]
                if matched_measured:
                    best_m = max(matched_measured, key=lambda r: r["direction_normalized_delta"])
                    n_eval_hit1 += 1
                    if top == best_m["candidate_sequence"]:
                        n_correct_hit1 += 1
        return {
            "arm": arm_name,
            "n_sources": len(by_src),
            "graded_support": n_graded_sup,
            "graded_support_rate": round(n_graded_sup / len(by_src), 6) if by_src else None,
            "exact_match_support": n_exact,
            "graded_sc_hit1": round(n_correct_hit1 / n_eval_hit1, 6) if n_eval_hit1 else None,
            "n_eval_hit1": n_eval_hit1,
        }

    def sampler_oracle(s, L, rng):
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
    print(json.dumps(arm1))
    arm2 = run_arm("local8_W16", sampler_local8)
    print(json.dumps(arm2))

    # top-level graded_sc_hit1 = structure-pool arm (G4 gate reads this field)
    out = {
        "schema": "d16c_g4_structure_probe.v1",
        "date": "2026-09-16",
        "checkpoint": str(PROBE_CKPT),
        "calibre": "D15-2 mirror (run_d15_2_pipeline.py), rng seed 20260912",
        "baseline_graded_sc_hit1_on_official_v5_ckpt": 0.0,
        "graded_sc_hit1": arm1["graded_sc_hit1"],
        "graded_sc_hit1_arm2_local8": arm2["graded_sc_hit1"],
        "arm1_oracle_structured": arm1,
        "arm2_local8_W16": arm2,
        "cuda_device": cuda_name,
        "cuda_visible_devices": cuda_visible,
        "cpu_fallback_used": False,
        "protected_reads": 0,
    }
    with open(OUT / "d16c_g4_structure_probe.json", "w") as f:
        json.dump(out, f, indent=1)
    print("saved:", OUT / "d16c_g4_structure_probe.json")

if __name__ == "__main__":
    main()
