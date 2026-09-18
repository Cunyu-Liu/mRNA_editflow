#!/usr/bin/env python3
"""ERK v2 G4 structure-pool probe (amendment ERK-G4, MRL task-domain clarification).

D15-2 calibre mirror with the task domain adapted to the ERK single-task MRL head:
same struct_key (n_edits, first_pos//8, last_pos//8), same oracle-structured /
local8_W16 samplers, same rng seed 20260912, 32 cand/source — but the pool is the
MRL VALIDATION sources and scoring is the ERK linear head.
G4 gate (frozen): graded_sc_hit1 > 0 on the structure pool.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
PROJ = MNT / "projections/xedit_v3/development_train_validation_v1/validation.jsonl"
CKPT = MNT / "experiments/xeditcritic_erk_v2/erk_train_seed2026091901/erk_epoch_6.pt"
OUT = MNT / "experiments/xeditcritic_erk_v2/structure_probe_g4"
OUT.mkdir(parents=True, exist_ok=True)

MRL_TASK = "MEAN_RIBOSOME_LOAD::region=0"
BASES = "ACGT"
N_SAMPLES = 32
SEED = 20260912


def edits_of(r):
    return [(e.get("position"), e.get("candidate_base")) for e in r.get("source_relative_edits", [])]


def main():
    if not torch.cuda.is_available():
        (OUT / "erk_g4_structure_probe.json").write_text(json.dumps(
            {"status": "STOPPED_WITH_EVIDENCE", "error": "CUDA unavailable", "cpu_fallback_used": True}))
        print("STOPPED_WITH_EVIDENCE: CUDA unavailable", flush=True)
        return 2
    torch.cuda.set_device(3)
    dev = torch.device("cuda:3")

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    assert ck["schema"] == "route_a_v3_erk_v2_train.v1" and ck["epoch"] == 6
    W = ck["model_state_dict"]["W"].to(dev, dtype=torch.float32)
    b = ck["model_state_dict"]["b"].to(dev, dtype=torch.float32)
    src_index = ck["source_index"]
    n_blocks = ck["n_blocks"]
    n_ctx = 64

    rows = []
    with open(PROJ) as f:
        for line in f:
            d = json.loads(line)
            if d["task_id"] == MRL_TASK:
                rows.append(d)
    by_src = defaultdict(list)
    for r in rows:
        by_src[r["source_id"]].append(r)

    def struct_key(r):
        es = sorted(e.get("position", 0) for e in r.get("source_relative_edits", []))
        if not es:
            return None
        return (len(es), es[0] // 8, es[-1] // 8)

    def feature_of(src, cand):
        s = src.upper().replace("U", "T")
        c = cand.upper().replace("U", "T")
        f = np.zeros((n_blocks, n_ctx), dtype=np.float32)
        for pos in range(min(len(s), len(c))):
            if s[pos] == c[pos]:
                continue
            alt = c[pos]
            if alt not in BASES:
                continue
            if pos >= n_blocks * 8:
                continue
            left = s[pos - 1] if pos > 0 and s[pos - 1] in BASES else "N"
            right = s[pos + 1] if pos + 1 < len(s) and s[pos + 1] in BASES else "N"
            if left == "N" and right == "N":
                continue
            if left == "N":
                left = right
            if right == "N":
                right = left
            ci = BASES.index(left) * 16 + BASES.index(right) * 4 + BASES.index(alt)
            f[pos // 8, ci] += 1.0
        return f.ravel()

    def score_delta(sid, src, cands):
        X = np.stack([feature_of(src, c) for c in cands])
        Xt = torch.from_numpy(X).to(dev)
        bi = torch.tensor(src_index.get(str(sid), 0), device=dev)
        with torch.inference_mode():
            return (Xt @ W + b[bi]).cpu().numpy()

    rng = np.random.RandomState(SEED)

    def sampler_oracle(s, L, rng, src_rows):
        keys = [struct_key(r) for r in src_rows if struct_key(r)]
        k = keys[rng.randint(len(keys))]
        n_e, blk_lo, blk_hi = k
        lo, hi = blk_lo * 8, min(L, (blk_hi + 1) * 8)
        if hi - lo < n_e:
            lo = max(0, hi - n_e)
        pos = rng.choice(np.arange(lo, hi), size=n_e, replace=False)
        cand = list(s)
        for p in pos:
            alts = [x for x in BASES if x != s[int(p)]]
            cand[int(p)] = alts[rng.randint(3)]
        return "".join(cand)

    def sampler_local8(s, L, rng):
        n_e = 8
        Wd = 16
        start = rng.randint(0, max(1, L - Wd))
        lo, hi = start, min(L, start + Wd)
        if hi - lo < n_e:
            lo = max(0, hi - n_e)
        pos = rng.choice(np.arange(lo, hi), size=n_e, replace=False)
        cand = list(s)
        for p in pos:
            alts = [x for x in BASES if x != s[int(p)]]
            cand[int(p)] = alts[rng.randint(3)]
        return "".join(cand)

    def run_arm(name, sampler):
        n_graded_sup = 0
        n_exact = 0
        n_eval = 0
        n_correct = 0
        for sid, rs in sorted(by_src.items()):
            s = rs[0]["source_sequence"].upper().replace("U", "T")
            L = len(s)
            keys = {struct_key(r) for r in rs if struct_key(r)}
            if not keys:
                continue
            cands = [sampler(s, L, rng, rs) if name == "oracle_structured" else sampler(s, L, rng)
                     for _ in range(N_SAMPLES)]
            graded_keys = set()
            for c in cands:
                es = [p for p in range(min(len(s), len(c))) if s[p] != c[p]]
                if es:
                    graded_keys.add((len(es), es[0] // 8, es[-1] // 8))
            gs = bool(graded_keys & keys)
            exact = any(c in {r["candidate_sequence"] for r in rs} for c in cands)
            n_graded_sup += int(gs)
            n_exact += int(exact)
            if not gs:
                continue
            deltas = score_delta(sid, s, cands)
            top = cands[int(np.argmax(deltas))]
            matched = [r for r in rs if struct_key(r) in graded_keys]
            if matched:
                best_m = max(matched, key=lambda r: r["direction_normalized_delta"])
                n_eval += 1
                if top == best_m["candidate_sequence"]:
                    n_correct += 1
        return {
            "arm": name,
            "n_sources": len(by_src),
            "graded_support": n_graded_sup,
            "graded_support_rate": round(n_graded_sup / len(by_src), 6) if by_src else None,
            "exact_match_support": n_exact,
            "graded_sc_hit1": round(n_correct / n_eval, 6) if n_eval else None,
            "n_eval_hit1": n_eval,
        }

    arm1 = run_arm("oracle_structured", lambda s, L, rng, rs: sampler_oracle(s, L, rng, rs))
    arm2 = run_arm("local8_W16", lambda s, L, rng, rs=None: sampler_local8(s, L, rng))
    print(json.dumps(arm1))
    print(json.dumps(arm2))

    out = {
        "schema": "erk_v2_g4_structure_probe.v1",
        "date": "2026-09-19",
        "checkpoint": str(CKPT),
        "task": "MRL",
        "calibre": "D15-2 mirror (struct_key n_edits x window-block 8), rng seed 20260912, task-domain adapted to MRL (ERK single-task head)",
        "baseline_reference": "D15-2 polyA on official V5 ckpt = 0.0 (D16-C G4 on probe ckpt = 0.0)",
        "graded_sc_hit1": arm1["graded_sc_hit1"],
        "graded_sc_hit1_arm2_local8": arm2["graded_sc_hit1"],
        "arm1_oracle_structured": arm1,
        "arm2_local8_W16": arm2,
        "cuda_device": "cuda:3",
        "cpu_fallback_used": False,
        "protected_reads": 0,
    }
    (OUT / "erk_g4_structure_probe.json").write_text(json.dumps(out, indent=1))
    print("saved:", OUT / "erk_g4_structure_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())