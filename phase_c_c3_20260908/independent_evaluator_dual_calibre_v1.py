#!/usr/bin/env python3
"""C2 full-891 independent-evaluator dual calibre (amendment 3.3, mandatory).

For each of the 891 sources, take the tie-aware top-1 candidate from the
guided (beta=0.25) pool and from the unguided baseline pool, score both with
the FROZEN independent evaluator:
  - MRL family (GSE114002, 50nt): frozen Optimus5Prime, delta = cand - src
  - polyA family (GSE269595, 164nt): frozen APARENT base, delta =
    proximal_log2_odds(cand, cut 80..105) - proximal_log2_odds(src, ...)
Report per-family: delta distributions (guided vs unguided top-1), paired
source-level bootstrap CI of (guided_delta - unguided_delta), and the
amendment 3.3 annotation rule:
  positive on recovery-family + NOT positive on independent delta
  -> "口径敏感的正结果" (calibre-sensitive positive) flagged, not auto-FAIL.

MPRAU/HL have no frozen independent evaluator (registered limitation) —
only MRL/polyA families are dual-calibre covered (672 of 891 sources).
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

V8 = Path("/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_v8_stage1_prep_20260904")
HARNESS = V8 / "scripts/route_a_v3/run_route2_external_prediction_baselines_v1.py"
APARENT_SCRIPT = V8 / "scripts/route_a_v3/run_route2_aparent_baseline_v1.py"

X = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditsetflow_v5")
GUIDED = X / "beta_full891_20260909/beta_0.25_full/guided/generated_candidates.private.jsonl"
UNGUIDED = X / "guided_b2_20260903/b2_full_891/unguided/generated_candidates.private.jsonl"
MANIFEST = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/generation_eligibility/"
                "development_validation_v1/source_eligibility.jsonl")
OUT = X / "beta_full891_20260909/independent_evaluator_dual_calibre.json"
BOOT_ITERS, BOOT_SEED = 2000, 20260816
OPTIMUS_W = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/optimus5prime/main_MRL_model.hdf5")
APARENT_W = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/aparent/saved_models/aparent_large_lessdropout_all_libs_no_sampleweights.h5")
CUT = (80, 105)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def norm(s):
    return str(s).upper().replace("T", "U")


def load_pool(path):
    gen = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        sk = r["source_key"]
        sq = norm(r["candidate_sequence"])
        s = float(r.get("generation_score") or 0.0)
        if sq not in gen[sk] or gen[sk][sq] < s:
            gen[sk][sq] = s
    return gen


def tie_aware_top1(seqs: dict[str, float]) -> str | None:
    if not seqs:
        return None
    ranked = sorted(seqs.items(), key=lambda kv: -kv[1])
    top = ranked[0][1]
    block = [sq for sq, s in ranked if s == top]
    return block[0] if len(block) == 1 else sorted(block)[0]  # deterministic under ties


def main() -> int:
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    device = torch.device(f"cuda:{gpu}")

    harness = _load("harness", HARNESS)
    apar = _load("apar", APARENT_SCRIPT)

    # source sequences by family
    src_seq = {}
    for line in MANIFEST.open():
        r = json.loads(line)
        src_seq[str(r["source_key"])] = str(r["source_sequence"]).upper()
    guided = load_pool(GUIDED)
    unguided = load_pool(UNGUIDED)
    shared = sorted(set(guided) & set(unguided) & set(src_seq))

    fam = {"mrl": [], "polya": []}
    for sk in shared:
        study = sk.split("::")[0]
        if study == "GSE114002":
            fam["mrl"].append(sk)
        elif study == "GSE269595":
            fam["polya"].append(sk)

    # ---- Optimus (MRL) ----
    # Optimus/APARENT one_hot expects DNA alphabet (A/C/G/T); pools are RNA (U)
    def to_dna(s: str) -> str:
        return s.replace("U", "T")

    opt = harness.Optimus5Prime(OPTIMUS_W).to(device).eval()
    mrl_rows = []
    with torch.no_grad():
        for start in range(0, len(fam["mrl"]), 64):
            batch = fam["mrl"][start:start + 64]
            seqs = []
            for sk in batch:
                seqs.append(to_dna(src_seq[sk]))
                seqs.append(to_dna(tie_aware_top1(guided[sk]) or src_seq[sk]))
                seqs.append(to_dna(tie_aware_top1(unguided[sk]) or src_seq[sk]))
            enc = harness.one_hot(seqs, device)
            vals = opt(enc).float().squeeze(-1).cpu().numpy()
            for i, sk in enumerate(batch):
                j = 3 * i
                mrl_rows.append({
                    "source_key": sk,
                    "guided_delta": float(vals[j + 1] - vals[j]),
                    "unguided_delta": float(vals[j + 2] - vals[j]),
                })

    # ---- APARENT (polyA) ----
    apa = apar.AparentBase(APARENT_W).to(device).eval()

    def apa_delta(seq: str) -> float:
        enc = apar.one_hot([seq], device)
        with torch.no_grad():
            _, cut = apa(enc)
        return float(apar.proximal_log2_odds(cut, CUT[0], CUT[1]))

    polya_rows = []
    for sk in fam["polya"]:
        s = to_dna(src_seq[sk])
        g = to_dna(tie_aware_top1(guided[sk]) or src_seq[sk])
        u = to_dna(tie_aware_top1(unguided[sk]) or src_seq[sk])
        base = apa_delta(s)
        polya_rows.append({
            "source_key": sk,
            "guided_delta": apa_delta(g) - base,
            "unguided_delta": apa_delta(u) - base,
        })

    # ---- paired bootstrap per family ----
    def family_report(rows):
        if not rows:
            return {"n": 0}
        deltas = [r["guided_delta"] - r["unguided_delta"] for r in rows]
        n = len(deltas)
        rng = random.Random(BOOT_SEED)
        boots = []
        for _ in range(BOOT_ITERS):
            idx = [rng.randrange(n) for _ in range(n)]
            boots.append(sum(deltas[i] for i in idx) / n)
        boots.sort()
        point = sum(deltas) / n
        ci = [boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]]
        gd = [r["guided_delta"] for r in rows]
        ud = [r["unguided_delta"] for r in rows]
        return {
            "n": n,
            "guided_top1_delta_mean": sum(gd) / n,
            "unguided_top1_delta_mean": sum(ud) / n,
            "delta_point": point,
            "delta_ci95": ci,
            "independent_positive": ci[0] > 0,
        }

    mrl_rep = family_report(mrl_rows)
    polya_rep = family_report(polya_rows)

    dual = mrl_rep.get("independent_positive") or polya_rep.get("independent_positive")
    payload = {
        "schema_version": "route_a_v3_phase_c_independent_evaluator_dual_calibre.v1",
        "amendment": "3.3 dual calibre (frozen Optimus MRL / frozen APARENT polyA; cut 80-105)",
        "coverage": {"mrl": mrl_rep.get("n", 0), "polya": polya_rep.get("n", 0),
                     "note": "MPRAU/HL have no frozen independent evaluator (registered limitation)"},
        "mrl": mrl_rep,
        "polya": polya_rep,
        "annotation_rule": "positive on recovery-family + NOT positive on independent delta -> calibre-sensitive positive (如实标注, not auto-FAIL)",
        "verdict": "INDEPENDENT_CONFIRMED" if dual else "CALIBRE_SENSITIVE",
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print("MRL:  ", json.dumps(mrl_rep))
    print("polyA:", json.dumps(polya_rep))
    print("verdict:", payload["verdict"])
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
