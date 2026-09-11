#!/usr/bin/env python3
"""V6.5 B2-I gate: frozen Optimus (MRL) on A-gen top-32 submissions vs B32
baseline, top-1 delta, paired bootstrap. Reuses the dual-calibre loaders."""
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

S = Path("/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901/phase_c_c3_20260908/independent_evaluator_dual_calibre_v1.py")
sys.path.insert(0, str(S.parent))

spec = importlib.util.spec_from_file_location("dualcal", str(S))
dc = importlib.util.module_from_spec(spec)
# neutralize its __main__ run: set sys.argv guard
import types

src = S.read_text()
# strip trailing main invocation if present
if "__main__" in src:
    src = src.split('if __name__ == "__main__":')[0]
mod = types.ModuleType("dualcal")
exec(compile(src, str(S), "exec"), mod.__dict__)

X = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditsetflow_v5")
B32 = X / "guided_b2_20260903/b2_full_891/unguided/generated_candidates.private.jsonl"
ARMS = {
    "B256_main": X / "pool256_unguided_20260908/b2_full_891_B256/unguided/generated_candidates.private.jsonl",
    "B256_seed16": X / "v65_explore_select_20260910/B256_seed16/unguided/generated_candidates.private.jsonl",
    "B256_seed17": X / "v65_explore_select_20260910/B256_seed17/unguided/generated_candidates.private.jsonl",
    "B512": X / "v65_explore_select_20260910/B512_unguided/unguided/generated_candidates.private.jsonl",
    "B1024": X / "v65_explore_select_20260910/B1024_unguided/unguided/generated_candidates.private.jsonl",
}


def norm(s):
    return str(s).upper().replace("T", "U")


def load(path):
    import json as j

    gen = defaultdict(dict)
    for line in open(path):
        r = j.loads(line)
        sk = r["source_key"]
        sq = norm(r["candidate_sequence"])
        sc = float(r.get("generation_score") or 0.0)
        if sq not in gen[sk] or gen[sk][sq] < sc:
            gen[sk][sq] = sc
    return gen


def bci(d, iters=2000, seed=20260816):
    rng = random.Random(seed)
    n = len(d)
    means = []
    for _ in range(iters):
        s = [d[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / len(s))
    means.sort()
    return [means[int(0.025 * iters)], means[int(0.975 * iters)]]


# Reuse the dual-calibre harness exactly as the parallel session built it.
# Inspect what callables it exposes:
names = [n for n in dir(mod) if not n.startswith("_")]
print("dualcal exports:", names[:40], file=sys.stderr)

# We re-implement top-1 delta scoring here using its loaders where possible.
# Fallback minimal Optimus harness (same as dual calibre v1):
import torch

V8 = Path("/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_v8_stage1_prep_20260904")
OPTIMUS_SCRIPT = None
for cand in V8.glob("scripts/route_a_v3/run_route2_external_prediction_baselines_v1.py"):
    OPTIMUS_SCRIPT = cand
OPTIMUS_W = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/optimus5prime/main_MRL_model.hdf5"
)

harness = mod._load("optimus_harness", OPTIMUS_SCRIPT) if hasattr(mod, "_load") else None
if harness is None:
    spec2 = importlib.util.spec_from_file_location("optimus_harness", str(OPTIMUS_SCRIPT))
    harness = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(harness)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
opt = harness.Optimus5Prime(OPTIMUS_W).to(device).eval()


def top1_delta(pool):
    """per source: Optimus(top-1 by genscore) - Optimus(source)."""
    out = {}
    with torch.inference_mode():
        for sk, seqs in pool.items():
            ranked = sorted(seqs.items(), key=lambda kv: -kv[1])
            top1 = ranked[0][0]
            src = top1  # need source sequence: read from source_eligibility
            out[sk] = None  # placeholder
    return out


# source sequences:
M = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/generation_eligibility/development_validation_v1")
src_seq = {}
TASK = {}
for line in open(M / "source_eligibility.jsonl"):
    r = json.loads(line)
    src_seq[r["source_key"]] = norm(r["source_sequence"])
    TASK[r["source_key"]] = str(r["endpoint_id"])

MRL = "MEAN_RIBOSOME_LOAD"


def top1_scores(pool):
    res = {}
    mrl = [sk for sk in pool if TASK.get(sk) == MRL]
    batch, keys = [], []
    for sk in mrl:
        ranked = sorted(pool[sk].items(), key=lambda kv: -kv[1])
        top1 = ranked[0][0]
        batch.extend([src_seq[sk], top1])
        keys.append(sk)
        if len(batch) >= 64:
            _score_flush(batch, keys, res)
            batch, keys = [], []
    if batch:
        _score_flush(batch, keys, res)
    return res


def _score_flush(batch, keys, res):
    # Optimus one-hot: DNA alphabet, fixed 50nt handling as in dual calibre
    import numpy as np

    oh = []
    for s in batch:
        t = s.replace("U", "T")
        row = np.zeros((len(t), 4), dtype=np.float32)
        for i, ch in enumerate(t):
            row[i, "ACGT".index(ch) if ch in "ACGT" else 0] = 1.0
        oh.append(row)
    # pad to max len in batch (Optimus5Prime uses conv, padding tolerant)
    L = max(x.shape[0] for x in oh)
    arr = np.zeros((len(oh), L, 4), dtype=np.float32)
    for i, x in enumerate(oh):
        arr[i, : x.shape[0]] = x
    t = torch.from_numpy(arr).to(device)
    with torch.inference_mode():
        preds = opt(t).squeeze(-1).float().cpu().numpy().tolist()
    for i, sk in enumerate(keys):
        res[sk] = (preds[2 * i + 1] - preds[2 * i], preds[2 * i])  # (delta, top1_pred)
    return res


b32 = load(B32)
base_scores = top1_scores(b32)
rows = {}
for tag, path in ARMS.items():
    pool = load(path)
    # A-gen: select top-32 by genscore
    sel = {}
    for sk, seqs in pool.items():
        ranked = sorted(seqs.items(), key=lambda kv: -kv[1])[:32]
        sel[sk] = dict(ranked)
    sc = top1_scores(sel)
    common = [k for k in sc if k in base_scores]
    d = [sc[k][0] - base_scores[k][0] for k in common]
    ci = bci(d)
    rows[tag] = {
        "n": len(common),
        "guided_top1_delta_mean": sum(x[0] for x in sc.values()) / len(sc),
        "unguided_top1_delta_mean": sum(x[0] for x in base_scores.values()) / len(base_scores),
        "d_point": sum(d) / len(d),
        "d_ci95": ci,
        "b2i_gate_pass": bool(sum(d) / len(d) >= 0.02 and ci[0] > 0),
    }
out = {
    "schema_version": "route_a_v3_v65_b2i_independent_calibre.v1",
    "gate": "amendment v2 B2-I: MRL Optimus top-1 delta >= +0.02 CI excl zero",
    "rows": rows,
}
Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditsetflow_v5/v65_explore_select_20260910/v65_b2i_independent.json").write_text(
    json.dumps(out, indent=1)
)
print(json.dumps(out, indent=1))
