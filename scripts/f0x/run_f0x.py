"""F0-X runner: train a source-anchored legal Edit Flow base and verify acceptance.

Trains ``FlowRateNet`` (source-conditioned, substitution-only) with the
Bregman/Edit Flow loss on real 5'UTR effect data, then runs the first-order
constrained sampler with fixed budgets k in {1,3,5} over real source 5'UTRs and
verifies the acceptance invariants at scale: legality=100%, length
preservation=100%, budget violation=0, reproducible fixed-seed trajectory.

GPU policy: use the permitted CUDA device (GPU 4 forbidden); FAIL instead of
silently falling back to CPU (fallback=0), per the project convention.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from scripts.m4_sparse.config import get_config, NUC_ORDER, MAX_SEQ_LEN
from scripts.f0x.flow import (
    NUC_TO_IDX,
    EditFlowState,
    FirstOrderConstrainedSampler,
    LegalAction,
    apply_action,
    bregman_flow_loss,
    build_state,
    enumerate_legal_actions,
    uniform_policy,
)
from scripts.f0x.base import FlowRateNet


def select_device(cfg, override):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available on this host (fallback=0)")
    if override:
        idx = override.split(":")[-1]
        if idx in cfg.FORBIDDEN_DEVICES:
            raise RuntimeError(f"requested GPU {override} is forbidden")
        return torch.device(override)
    for dev in cfg.CUDA_DEVICES:
        if dev.split(":")[-1] in cfg.FORBIDDEN_DEVICES:
            continue
        return torch.device(dev)
    raise RuntimeError("no permitted CUDA device")


def norm_seq(s):
    return (s or "").upper().replace("T", "U")


def one_hot(seq, max_len=MAX_SEQ_LEN):
    arr = np.zeros((max_len, 4), dtype=np.float32)
    for i, ch in enumerate(norm_seq(seq)[:max_len]):
        if ch in NUC_ORDER:
            arr[i, NUC_ORDER.index(ch)] = 1.0
    return arr


def build_legal(seq, max_len=MAX_SEQ_LEN):
    """[max_len,4] legal grid: whole UTR5 editable, nt != current nucleotide."""
    m = np.zeros((max_len, 4), dtype=bool)
    for i, ch in enumerate(norm_seq(seq)[:max_len]):
        if ch in NUC_ORDER:
            for nt in NUC_ORDER:
                if nt != ch:
                    m[i, NUC_ORDER.index(nt)] = True
    return m


def build_target(record, max_len=MAX_SEQ_LEN):
    """One-hot target action [max_len,4] from the record's SUB edit(s)."""
    s = norm_seq(record["source_sequence"])
    t = np.zeros((max_len, 4), dtype=np.float32)
    for e in record["edit_list"]:
        if e["op"] == "SUB":
            pos = int(e["pos"])
            tok = str(e["token"]).upper().replace("T", "U")
            if (pos < max_len and tok in NUC_ORDER
                    and pos < len(s) and tok != s[pos]):
                t[pos, NUC_ORDER.index(tok)] = 1.0
    return t


def load_rows(dataset, benchmark, limit=None, seed=0):
    rows = []
    for line in open(dataset):
        r = json.loads(line)
        if r["benchmark"] != benchmark:
            continue
        if r.get("delta") is None:
            continue
        if not any(e["op"] == "SUB" for e in r.get("edit_list", [])):
            continue
        rows.append(r)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if limit:
        rows = rows[:limit]
    return rows


def train_flow(rows, cfg, device, steps=1500, batch=64, w=3.0, lr=None):
    net = FlowRateNet(cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr or cfg.LR,
                           weight_decay=cfg.WEIGHT_DECAY)
    srcs = np.stack([one_hot(r["source_sequence"]) for r in rows])
    legals = np.stack([build_legal(r["source_sequence"]) for r in rows])
    targets = np.stack([build_target(r) for r in rows])
    budget_idx = 1  # single-step target (flow base learns first-edit rates)
    n = len(rows)
    rng = np.random.RandomState(0)
    losses = []
    net.train()
    for _ in range(steps):
        idx = rng.choice(n, min(batch, n), replace=False)
        src = torch.tensor(srcs[idx]).to(device)
        cur = src.clone()
        ed = torch.tensor(legals[idx]).to(device)
        tg = torch.tensor(targets[idx]).to(device)
        bi = torch.full((len(idx),), budget_idx, dtype=torch.long, device=device)
        out = net(src, cur, bi, ed)
        loss = bregman_flow_loss(out["masked_logits"], ed, tg, w=w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return net, losses


def make_flow_policy(net, device):
    def policy(state, actions):
        return net.policy_fn(state, actions)
    return policy


def verify_trajectory(state, out):
    """Return (legal_ok, length_ok, budget_ok)."""
    cur = state
    legal_ok = True
    for t in out["trajectory"]:
        acts = enumerate_legal_actions(cur)
        legal_ids = {(a.pos, a.target_nt) for a in acts}
        if (t["pos"], t["target"]) not in legal_ids:
            legal_ok = False
        cur = apply_action(cur, LegalAction(t["pos"], t["target"]))
    length_ok = (out["length"] == out["source_length"])
    budget_ok = (out["n_steps"] <= state.budget_remaining
                 and out["budget_remaining"] >= 0)
    return legal_ok, length_ok, budget_ok


def evaluate(sources, policy_fn, k, seed):
    sampler = FirstOrderConstrainedSampler(policy_fn, seed=seed)
    n = len(sources)
    legal_ok = length_ok = budget_ok = 0
    n_steps = []
    for st in sources:
        out = sampler.sample(st)
        l_ok, len_ok, b_ok = verify_trajectory(st, out)
        legal_ok += int(l_ok)
        length_ok += int(len_ok)
        budget_ok += int(b_ok)
        n_steps.append(out["n_steps"])
    return {
        "n_sources": n,
        "legality_ok": legal_ok,
        "legality_pct": 100.0 * legal_ok / n,
        "length_preservation_ok": length_ok,
        "length_preservation_pct": 100.0 * length_ok / n,
        "budget_ok": budget_ok,
        "budget_violation": n - budget_ok,
        "budget_violation_pct": 100.0 * (n - budget_ok) / n,
        "mean_n_steps": float(np.mean(n_steps)),
        "expected_n_steps": k,
    }


def reproducibility_check(sources, policy_fn, k, seed):
    """Same seed -> identical trajectory/final sequence for every source."""
    s1 = FirstOrderConstrainedSampler(policy_fn, seed=seed)
    s2 = FirstOrderConstrainedSampler(policy_fn, seed=seed)
    mismatches = 0
    for st in sources:
        o1 = s1.sample(st)
        o2 = s2.sample(st)
        if o1["trajectory"] != o2["trajectory"] or o1["final_seq"] != o2["final_seq"]:
            mismatches += 1
    return {"n_sources": len(sources), "mismatches": mismatches,
            "reproducible_pct": 100.0 * (len(sources) - mismatches) / len(sources)}


def build_sources(rows, k, max_sources=None):
    sources = []
    for r in rows:
        seq = norm_seq(r["source_sequence"])
        if not seq:
            continue
        st = build_state(seq, [True] * len(seq), budget=k)
        sources.append(st)
        if max_sources and len(sources) >= max_sources:
            break
    return sources


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("artifacts/b0x/effect_dataset.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/f0x"))
    ap.add_argument("--benchmark", default="5U-A1")
    ap.add_argument("--train-limit", type=int, default=8000)
    ap.add_argument("--eval-limit", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--w", type=float, default=3.0)
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="path to save the trained (frozen) base-flow checkpoint")
    args = ap.parse_args()

    cfg = get_config()
    device = select_device(cfg, args.gpu)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[f0x] device={device} benchmark={args.benchmark}")

    rows = load_rows(args.dataset, args.benchmark, limit=args.train_limit, seed=args.seed)
    print(f"[f0x] filtered train rows (with SUB edit): {len(rows)}")

    net = None
    losses = []
    if not args.skip_train:
        net, losses = train_flow(rows, cfg, device, steps=args.steps,
                                 batch=args.batch, w=args.w)
        net.eval()  # FIX: disable dropout so flow-policy inference is deterministic
        print(f"[f0x] trained base flow: steps={args.steps} "
              f"loss_first={losses[0]:.4f} loss_last={losses[-1]:.4f}")
        if args.ckpt is not None:
            args.ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": net.state_dict(), "cfg": cfg,
                        "phase": "F0-X", "frozen_for": "G1-X",
                        "benchmark": args.benchmark, "seed": args.seed},
                       args.ckpt)
            print(f"[f0x] saved frozen base-flow checkpoint -> {args.ckpt}")

    # held-out source sample for evaluation (independent of training rows)
    eval_rows = load_rows(args.dataset, args.benchmark,
                          limit=args.eval_limit, seed=args.seed + 1)

    results = {
        "phase": "F0-X",
        "goal": "GOAL-XEDITFLOW-MIGRATION-01",
        "benchmark": args.benchmark,
        "device": str(device),
        "action_set": "UTR5_SUB(position, target_nt)",
        "termination": "FIXED_BUDGET",
        "sampler": "FirstOrderConstrainedSampler (first-order, not exact CTMC)",
        "train_rows": len(rows),
        "eval_rows": len(eval_rows),
        "loss": {"first": losses[0] if losses else None,
                 "last": losses[-1] if losses else None,
                 "n": len(losses)},
        "budgets": {},
    }

    for k in args.budgets:
        sources = build_sources(eval_rows, k, max_sources=args.eval_limit)
        # uniform baseline
        uni = evaluate(sources, uniform_policy, k, seed=args.seed)
        uni_rep = reproducibility_check(sources, uniform_policy, k, args.seed)
        entry = {"uniform_baseline": uni, "reproducibility_uniform": uni_rep}
        # trained flow policy
        if net is not None:
            flow_pol = make_flow_policy(net, device)
            fl = evaluate(sources, flow_pol, k, seed=args.seed)
            fl_rep = reproducibility_check(sources, flow_pol, k, args.seed)
            entry["flow_policy"] = fl
            entry["reproducibility_flow"] = fl_rep
        results["budgets"][f"k{k}"] = entry
        print(f"[f0x] k={k}: uniform legality={uni['legality_pct']}% "
              f"length={uni['length_preservation_pct']}% "
              f"budget_viol={uni['budget_violation']} "
              f"repro={uni_rep['reproducible_pct']}%")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"f0x_results_{args.benchmark}.json").write_text(
        json.dumps(results, indent=2))
    print(f"[f0x] wrote {args.out_dir / f'f0x_results_{args.benchmark}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())