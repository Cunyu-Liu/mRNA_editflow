#!/usr/bin/env python3
"""COMB tier-2 third calibre: polyA APARENT report-only rows (prereg sec.3: n=20 report only).

For each polyA source (GSE269595, 20 in cohort): tie-aware top-1 candidate from each
D arm (genscore top-32 select-back) vs the A baseline top-1, scored with FROZEN
APARENT base (proximal_log2_odds cut 80..105) - the exact calibre of
independent_evaluator_dual_calibre_v1.py. Report-only (no gate) per prereg.
Also adds B/C reference pools so the 2x2 matrix polyA view is complete.
Run after the three D arms are terminal; this is the standalone polyA supplement
to comb_tier2_harvest.py (which covers sc/recovery/support + Optimus MRL).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

V8 = Path("/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_v8_stage1_prep_20260904")
HARNESS = V8 / "scripts/route_a_v3/run_route2_external_prediction_baselines_v1.py"
APARENT_SCRIPT = V8 / "scripts/route_a_v3/run_route2_aparent_baseline_v1.py"

X = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditsetflow_v5")
M = X / "comb_mechanism_20260911"
MANIFEST = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/generation_eligibility/development_validation_v1/source_eligibility.jsonl")
POOLS = {
    "A_baseline": X / "guided_b2_20260903/b2_full_891/unguided/generated_candidates.private.jsonl",
    "B_guide": X / "beta_full891_20260909/beta_0.25_full/guided/generated_candidates.private.jsonl",
    "C_explore": X / "pool256_unguided_20260908/b2_full_891_B256/unguided/generated_candidates.private.jsonl",
    "D_main": M / "D891_main/guided/generated_candidates.private.jsonl",
    "D_seed16": M / "D891_seed16/guided/generated_candidates.private.jsonl",
    "D_seed17": M / "D891_seed17/guided/generated_candidates.private.jsonl",
}
OUT = M / "comb_tier2_polya_aparent.json"
OPTIMUS_W = None
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
    if not Path(path).exists():
        return gen
    for line in open(path):
        r = json.loads(line)
        sk = r["source_key"]
        sq = norm(r["candidate_sequence"])
        s = float(r.get("generation_score") or 0.0)
        if sq not in gen[sk] or gen[sk][sq] < s:
            gen[sk][sq] = s
    return gen


def select32(gen):
    out = {}
    for sk, seqs in gen.items():
        ranked = sorted(seqs.items(), key=lambda kv: -kv[1])[:32]
        out[sk] = dict(ranked)
    return out


def tie_aware_top1(seqs):
    if not seqs:
        return None
    ranked = sorted(seqs.items(), key=lambda kv: -kv[1])
    top = ranked[0][1]
    block = [sq for sq, s in ranked if s == top]
    return block[0] if len(block) == 1 else sorted(block)[0]


def main() -> int:
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    device = torch.device(f"cuda:{gpu}")

    apar = _load("apar", APARENT_SCRIPT)

    src_seq = {}
    for line in MANIFEST.open():
        r = json.loads(line)
        src_seq[str(r["source_key"])] = str(r["source_sequence"]).upper()
    polya_sources = [sk for sk in src_seq if sk.split("::")[0] == "GSE269595"]
    print(f"polyA sources in manifest: {len(polya_sources)}", flush=True)

    pools = {}
    for tag, path in POOLS.items():
        if tag in ("B_guide", "A_baseline"):
            pools[tag] = load_pool(path)
        else:
            pools[tag] = select32(load_pool(path))

    apa = apar.AparentBase(APARENT_W).to(device).eval()

    def apa_delta(seq):
        enc = apar.one_hot([seq], device)
        with torch.no_grad():
            _, cut = apa(enc)
        return float(apar.proximal_log2_odds(cut, CUT[0], CUT[1]))

    rows = {}
    for tag, pool in pools.items():
        deltas = []
        for sk in polya_sources:
            if sk not in pool:
                continue
            top1 = tie_aware_top1(pool[sk])
            if top1 is None:
                continue
            s = src_seq[sk].replace("U", "T")
            g = top1.replace("U", "T")
            d = apa_delta(g) - apa_delta(s)
            deltas.append({"source_key": sk, "top1_delta": d})
        if deltas:
            mean = sum(x["top1_delta"] for x in deltas) / len(deltas)
            rows[tag] = {"n": len(deltas), "top1_delta_mean": mean, "report_only": True}
            print(tag, "->", rows[tag], flush=True)

    out = {
        "schema_version": "route_a_v3_comb_tier2_polya_aparent.v1",
        "caliber": "frozen APARENT base proximal_log2_odds (cut 80..105) on tie-aware top-1; report-only per prereg sec.3 (n=20 stratum)",
        "rows": rows,
        "note": "D arms use genscore top-32 select-back then top-1; A/B pools submitted as-is (32/256 trajectories resp.); C/D use select32.",
    }
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
