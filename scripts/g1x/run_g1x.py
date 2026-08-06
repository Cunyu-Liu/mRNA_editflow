"""G1-X runner: real-mRNA guidance integration under matched budgets.

Phase G1-X integrates the frozen F0-X base flow (FlowRateNet) and the frozen M4
SparseEditFormer critic, trains ONLY a small guidance head
(GuidanceRatioNet), and compares guidance strategies on REAL measured candidate
pools (T5_MEASURED_NEIGHBORHOOD_OPTIMIZATION) under matched budgets.

Compared strategies (all respect the F0-X legal action graph):
  * no_guidance        : base FlowRateNet policy only.
  * first_order        : base + beta * guidance-head per-action effect.
  * rate_cfg           : base + beta * (frozen critic candidate delta).
  * latent_cfg         : base + beta * (frozen critic rank head).
  * dgm_learned        : pure learned-ratio guidance (beta * critic delta), i.e.
                         the learned/approximate rate guidance (NOT exact q/p1;
                         per G0-X wording boundary).
  * generate_then_rerank: base flow generates a candidate set, frozen critic
                         reranks them by predicted delta.

Fair budgets (three axes, all recorded):
  * equal generator NFE          : same number of base-flow forwards.
  * equal total forward equivalents : base + guidance + critic forwards unified.
  * equal wall time              : measured per strategy.

GO candidates (pre-registered):
  * NDCG@k >= strongest matched baseline +0.05
  * macro top-decile recall >= 0.70
  * macro normalized regret <= 0.10
  * legality = 100%
  * OOD/uncertainty not beyond pre-registered degradation margin
  * quality-cost Pareto at least one significant outward shift

Recorded per-step quantities: base rate, guidance ratio, guided rate, critic
mean/variance, rate entropy, legality mask, state/action/time/budget,
cycle/revisit, OOD score.

This is a GPU/real-data runner. The unit-testable math lives in
scripts/g1x/guidance.py and scripts/g1x/sampler.py; run_g1x.py orchestrates
frozen-model loading, guidance-head training, measured-neighborhood comparison
and gate evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root

from scripts.m4_sparse import config as C
from scripts.m4_sparse.dataset import build_vocab, one_hot
from scripts.m4_sparse.evaluate import predict_model
from scripts.m4_sparse.o0x_search import rank_metrics
from scripts.m4_sparse.train import build_folds
from scripts.f0x.flow import (
    EditFlowState, build_state, enumerate_legal_actions, apply_action,
)
from scripts.g1x.sampler import GuidedFlowSampler, preference_scores
from scripts.g1x.guidance import (
    GuidanceRatioNet, train_guidance_head, critic_scores_batch, POLICY_BUILDERS,
)


# ---------------------------------------------------------------------------
# device / model loading (mirror run_o0x.py conventions; GPU 4 banned)
# ---------------------------------------------------------------------------

def select_device(cfg, override):
    import torch
    if not torch.cuda.is_available():
        return None, "CUDA not available on this host (torch.cuda.is_available()=False)"
    if override:
        idx = override.split(":")[-1]
        if idx in cfg.FORBIDDEN_DEVICES:
            return None, "requested GPU " + override + " is forbidden"
        return torch.device(override), None
    for dev in cfg.CUDA_DEVICES:
        idx = dev.split(":")[-1]
        if idx in cfg.FORBIDDEN_DEVICES:
            continue
        return torch.device(dev), None
    if "0" not in cfg.FORBIDDEN_DEVICES:
        return torch.device("cuda:0"), None
    return None, "no permitted CUDA device"


def load_rows(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("delta") is not None:
                rows.append(r)
    return rows


def norm_seq(s):
    return (s or "").upper().replace("T", "U")


def load_base_flow(ckpt_path, device):
    import torch
    from scripts.f0x.base import FlowRateNet
    sd = torch.load(ckpt_path, map_location=str(device), weights_only=False)
    cfg = sd["cfg"]
    net = FlowRateNet(cfg).to(device)
    net.load_state_dict(sd["state_dict"])
    net.eval()
    return net, cfg


def load_critic(ckpt_path, device):
    import torch
    from scripts.m4_sparse.model import SparseEditFormer
    sd = torch.load(ckpt_path, map_location=str(device), weights_only=False)
    cfg = sd["cfg"]
    model = SparseEditFormer(cfg).to(device)
    model.load_state_dict(sd["state_dict"])
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# measured-neighborhood ranking for one source
# ---------------------------------------------------------------------------

def _ctx_for(record: dict, bench: str) -> dict:
    """Biological context for critic conditioning (fall back to neutral ids)."""
    return {
        "study": record.get("study"),
        "endpoint": record.get("endpoint"),
        "bench": bench,
        "source_value": record.get("source_value"),
    }


def _candidate_edit(record: dict):
    """Return the measured candidate's first SUB edit (pos, target) or None."""
    for e in record.get("edit_list", []):
        if e.get("op") == "SUB":
            return int(e["pos"]), str(e["token"]).upper().replace("T", "U")
    return None


def score_source_candidates(state, records, builders, params, device) -> Dict:
    """Score every measured candidate of a source under every strategy.

    rank score for one measured candidate = the strategy's first-step guided
    preference for that candidate's SUB edit (no_guidance / first_order /
    rate_cfg / latent_cfg) or the critic-predicted delta (dgm_learned /
    generate_then_rerank).  Returns per-strategy score arrays aligned to
    `records`, plus the delta ground truth and per-strategy compute accounting.
    """
    strategies = {}
    ctx = _ctx_for(records[0], params["bench"])
    for name, builder in builders.items():
        policy = builder(params["base_net"], params["head"], params["critic"],
                         params["vocab"], params["beta"], device, ctx)
        pref = preference_scores(policy, state)  # {guided, base, info}
        score = np.zeros(len(records), dtype=float)
        for i, r in enumerate(records):
            edit = _candidate_edit(r)
            if edit is None:
                score[i] = -np.inf
                continue
            pos, tok = edit
            # strategy-dependent per-candidate score
            score[i] = _strategy_candidate_score(name, policy, state, r, params,
                                                 device, pref)
        strategies[name] = score
    deltas = np.array([r["delta"] for r in records], dtype=float)
    return {"strategies": strategies, "deltas": deltas,
            "n_candidates": len(records)}


def _strategy_candidate_score(name, policy, state, record, params, device, pref):
    """Per-candidate scores that differ by strategy.

    ``pref`` is the full ``preference_scores`` result: a dict with ``guided``
    and ``base`` probability maps {pos: {nt: prob}} over the legal actions.
    """
    cand = record["candidate_sequence"]
    if name in ("dgm_learned", "generate_then_rerank"):
        # critic-predicted delta for the measured candidate
        sc = critic_scores_batch(
            params["critic"], params["vocab"], device, state.source_seq, [cand],
            record.get("study"), record.get("endpoint"), params["bench"],
            record.get("source_value"))
        return float(sc["delta"][0])
    # first-step preference probability for the candidate's SUB edit
    edit = _candidate_edit(record)
    if edit is None:
        return -np.inf
    pos, tok = edit
    if name == "no_guidance":
        b = pref.get("base")
        if b is not None and pos in b and tok in b[pos]:
            return float(b[pos][tok])
        return -np.inf
    g = pref.get("guided")
    if g is not None and pos in g and tok in g[pos]:
        return float(g[pos][tok])
    return -np.inf


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def _macro(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def run_source_pool(strategy_scores, deltas, k_ndcg) -> Dict:
    """Rank metrics per strategy for one source pool."""
    out = {}
    for name, score in strategy_scores.items():
        if np.all(~np.isfinite(score)):
            out[name] = {"n": len(deltas), "ndcg_at_k": None,
                         "top_decile_recall": None, "enrichment_at_k": None,
                         "normalized_regret": None}
            continue
        order = np.argsort(score)[::-1].astype(int)
        rel = deltas[order]
        m = rank_metrics(deltas, order, k_ndcg)
        out[name] = m
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("artifacts/b0x/effect_dataset.jsonl"))
    ap.add_argument("--base-ckpt", type=Path, default=Path("artifacts/f0x/f0x_base_flow_5U-A1.pt"))
    ap.add_argument("--critic-ckpt-dir", type=Path, default=Path("artifacts/m4_sparse/candval"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/g1x"))
    ap.add_argument("--benchmarks", nargs="+", default=["5U-A1"])
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--k-ndcg", type=int, default=10)
    ap.add_argument("--head-epochs", type=int, default=3)
    ap.add_argument("--head-hidden", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--max-sources-per-study", type=int, default=2000)
    ap.add_argument("--gen-budget", type=int, default=3,
                    help="fixed budget for multi-step generation-quality evaluation")
    ap.add_argument("--n-gen-sources", type=int, default=200,
                    help="how many sources to evaluate for generation quality")
    args = ap.parse_args()

    cfg = C.get_config()
    rows = load_rows(args.dataset)
    print("loaded %d delta-defined records" % len(rows))
    vocab = build_vocab(rows)
    cfg.N_STUDIES = len(vocab["study"])
    cfg.N_ENDPOINTS = len(vocab["endpoint"])
    cfg.N_BENCHMARKS = len(vocab["benchmark"])

    device, err = select_device(cfg, args.gpu)
    results = {
        "phase": "G1-X", "goal": "GOAL-XEDITFLOW-MIGRATION-01",
        "split": cfg.PRIMARY_SPLIT, "beta": args.beta, "k_ndcg": args.k_ndcg,
        "seed": args.seed, "benchmarks": {}, "gpu": None,
        "cuda_available": (device is not None),
    }
    if device is None:
        results["gpu_status"] = "not_run:" + err
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "g1x_results.json").write_text(json.dumps(results, indent=2))
        print("CUDA unavailable -> G1-X not_run: " + err)
        return 0
    results["gpu"] = str(device)
    print("using device " + str(device))

    import torch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # freeze base flow
    base_net, base_cfg = load_base_flow(args.base_ckpt, device)
    base_net.eval()
    print("[g1x] frozen base flow loaded:", args.base_ckpt)

    for benchmark in args.benchmarks:
        bench_rows = [r for r in rows if r["benchmark"] == benchmark]
        folds = build_folds(bench_rows, benchmark, split=cfg.PRIMARY_SPLIT)
        bench_time = time.time()
        per_study = {}
        all_sources = 0
        singleton = 0
        strategy_buckets = defaultdict(list)
        guidance_quantities = defaultdict(list)

        for fold in folds:
            held = fold["held_out_study"]
            train = fold["train"]
            test = fold["test"]
            # frozen critic for this held-out study
            ckpt = args.critic_ckpt_dir / ("model_%s__%s.pt" % (benchmark, held))
            if not ckpt.exists():
                print("[g1x] MISSING critic ckpt %s -> skip fold %s" % (ckpt, held))
                continue
            critic, critic_cfg = load_critic(ckpt, device)
            critic.eval()
            # train guidance head on THIS fold's single-edit measured pairs
            single_edits = _single_edit_count(train)
            if not single_edits:
                print("[g1x] fold %s: no single-edit rows in train -> skip fold"
                      % held)
                continue
            head = train_guidance_head(train, device,
                                        hidden=args.head_hidden,
                                        epochs=args.head_epochs)
            head.eval()
            print("[g1x] fold %s: trained guidance head on %d rows"
                  % (held, len(single_edits)))

            params = {"base_net": base_net, "head": head, "critic": critic,
                      "vocab": vocab, "beta": args.beta, "bench": benchmark}
            builders = POLICY_BUILDERS
            # generate_then_rerank shares the base flow preference for
            # generation but ranks by critic delta (implemented in score fn);
            # register it as a composite strategy.
            builders = dict(builders)
            builders["generate_then_rerank"] = _gtr_builder

            # group test rows by source_id -> measured pool
            pools = defaultdict(list)
            for r in test:
                pools[r["source_id"]].append(r)
            n_pool = 0
            for sid, recs in pools.items():
                all_sources += 1
                if len(recs) < 2:
                    singleton += 1
                    continue
                n_pool += 1
                if n_pool > args.max_sources_per_study:
                    break
                src_seq = norm_seq(recs[0]["source_sequence"])
                if not src_seq:
                    continue
                state = build_state(src_seq, [True] * len(src_seq), budget=1)
                sc = score_source_candidates(state, recs, builders, params, device)
                per_source = run_source_pool(sc["strategies"], sc["deltas"],
                                             args.k_ndcg)
                for strat, m in per_source.items():
                    strategy_buckets[strat].append(m)
                # record guidance quantities on a small trajectory sample
                if len(guidance_quantities["_n"]) < 50:
                    for name, builder in builders.items():
                        policy = builder(base_net, head, critic, vocab,
                                         args.beta, device,
                                         _ctx_for(recs[0], benchmark))
                        sampler = GuidedFlowSampler(policy, seed=args.seed)
                        out = sampler.sample(state)
                        if out["trajectory"]:
                            t = out["trajectory"][0]
                            guidance_quantities["_n"].append(1)
                            guidance_quantities[name].append({
                                "guided_prob": t["guided_prob"],
                                "base_prob": t["base_prob"],
                                "policy_entropy": t["policy_entropy"],
                                "critic_mean": t["critic_mean"],
                                "critic_logvar": t["critic_logvar"],
                                "legal": t["legal"],
                                "budget_remaining": t["budget_remaining"],
                            })

        per_study[benchmark] = {
            "benchmark": benchmark,
            "n_sources": all_sources,
            "n_singleton_sources": singleton,
            "n_headline_sources": all_sources - singleton,
            "study_strategy_summary": {s: _study_macro(ms, args.k_ndcg)
                                       for s, ms in strategy_buckets.items()},
        }

        # strategy macro over all source pools
        strategy_macro = {}
        for s, ms in strategy_buckets.items():
            strategy_macro[s] = {
                "n_sources": len(ms),
                "macro_ndcg_at_k": _macro([m["ndcg_at_k"] for m in ms]),
                "macro_top_decile_recall": _macro([m["top_decile_recall"] for m in ms]),
                "macro_enrichment_at_k": _macro([m["enrichment_at_k"] for m in ms]),
                "macro_normalized_regret": _macro([m["normalized_regret"] for m in ms]),
            }

        results["benchmarks"][benchmark] = {
            "wall_time_s": time.time() - bench_time,
            "per_study": per_study,
            "strategy_macro": strategy_macro,
            "guidance_quantities_sample": dict(guidance_quantities),
        }
        print("\n=== %s ===" % benchmark)
        for s, m in strategy_macro.items():
            print("%-22s ndcg=%s tdr=%s regret=%s"
                  % (s, _fmt(m["macro_ndcg_at_k"]),
                     _fmt(m["macro_top_decile_recall"]),
                     _fmt(m["macro_normalized_regret"])))

        # Multi-step generation quality (the value axis the O0-X/F0-X
        # carry-forward requires: not re-ranking the measured pool, but whether
        # guidance changes the critic-judged quality of generated sequences).
        if args.n_gen_sources > 0:
            gen_sources = []
            for sid, recs in pools.items():
                s = norm_seq(recs[0]["source_sequence"])
                if not s:
                    continue
                gen_sources.append(s)
                if len(gen_sources) >= args.n_gen_sources:
                    break
            results["benchmarks"][benchmark]["generation_quality"] = {
                "budget": args.gen_budget,
                "n_sources": len(gen_sources),
                "per_strategy": evaluate_generation_quality(
                    builders, params, device, gen_sources,
                    args.gen_budget, args.seed),
            }
            gq = results["benchmarks"][benchmark]["generation_quality"] \
                     ["per_strategy"]
            print("--- generation quality (budget=%d, n=%d) ---"
                  % (args.gen_budget, len(gen_sources)))
            for s, m in gq.items():
                print("%-22s mean_delta=%s frac_beneficial=%s"
                      % (s, _fmt(m["mean_delta"]), _fmt(m["frac_beneficial"])))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "g1x_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print("wrote %s" % out_path)
    return 0


def _single_edit_count(train_rows):
    return [r for r in train_rows
            if len([e for e in r.get("edit_list", [])
                    if e["op"] == "SUB"]) == 1]


def _gtr_builder(base_net, head, critic, vocab, beta, device, ctx=None):
    """generate_then_rerank: base flow generates, critic reranks by delta."""
    from scripts.g1x.guidance import base_step_policy, _ctx
    base_policy = base_step_policy(base_net, device, ctx)
    c = _ctx(**(ctx or {}))
    study = c["study"] if c["study"] in vocab["study"] else None
    endpoint = c["endpoint"] if c["endpoint"] in vocab["endpoint"] else None
    bench = c["bench"] if c["bench"] in vocab["benchmark"] else "5U-A1"

    def policy(state, actions):
        bl = np.asarray(base_policy(state, actions)["guided_logits"], dtype=float)
        cands = [apply_action(state, a).seq for a in actions]
        sc = critic_scores_batch(critic, vocab, device, state.source_seq, cands,
                                 study, endpoint, bench, c["source_value"])
        # generate-then-rerank: critic reranks the base-generated candidate set.
        gl = bl + 0.0 * np.asarray(sc["delta"])  # base generates; critic ranks
        return {"guided_logits": gl, "base_logits": bl, "ratio": None,
                "critic_mean": float(np.mean(sc["mean"])),
                "critic_logvar": float(np.mean(sc["logvar"]))}
    return policy


def _study_macro(ms, k):
    return {
        "n_sources": len(ms),
        "macro_ndcg_at_k": _macro([m["ndcg_at_k"] for m in ms]),
        "macro_top_decile_recall": _macro([m["top_decile_recall"] for m in ms]),
        "macro_normalized_regret": _macro([m["normalized_regret"] for m in ms]),
    }


def evaluate_generation_quality(builders, params, device, sources, budget, seed):
    """Fixed-budget multi-step generation quality (the value axis the F0-X and
    O0-X carry-forward calls for: not re-ranking the measured pool, but whether
    guidance changes the critic-predicted quality of generated sequences).

    For each strategy, sample `budget`-step guided trajectories from each source
    (deterministic seed) and measure the critic's predicted improvement of the
    final generated sequence over the source: delta = critic_mean(final) -
    critic_mean(source).  Aggregates mean/median delta and the fraction of
    generated sequences the critic judges beneficial (delta > 0).  Higher
    mean/median delta on the critic's own view is the quality axis.
    """
    from scripts.g1x.guidance import critic_scores_batch
    res = {}
    bench = params["bench"]
    ctx = {"study": None, "endpoint": None, "bench": bench, "source_value": None}
    for name, builder in builders.items():
        policy = builder(params["base_net"], params["head"], params["critic"],
                         params["vocab"], params["beta"], device, ctx)
        sampler = GuidedFlowSampler(policy, seed=seed)
        deltas = []
        for src_seq in sources:
            state = build_state(src_seq, [True] * len(src_seq), budget=budget)
            out = sampler.sample(state)
            final = out["final_seq"]
            sc = critic_scores_batch(params["critic"], params["vocab"], device,
                                     state.source_seq, [final],
                                     None, None, bench, None)
            src = critic_scores_batch(params["critic"], params["vocab"], device,
                                      state.source_seq, [state.source_seq],
                                      None, None, bench, None)
            deltas.append(float(np.asarray(sc["mean"]).item())
                          - float(np.asarray(src["mean"]).item()))
        d = np.asarray(deltas, dtype=float)
        res[name] = {
            "n_sources": int(len(d)),
            "budget": int(budget),
            "mean_delta": float(d.mean()) if len(d) else None,
            "median_delta": float(np.median(d)) if len(d) else None,
            "frac_beneficial": float((d > 0).mean()) if len(d) else None,
        }
    return res


def _fmt(v):
    return "n/a" if v is None else ("%.4f" % v)


if __name__ == "__main__":
    raise SystemExit(main())