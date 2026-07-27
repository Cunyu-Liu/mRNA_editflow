#!/usr/bin/env python3
"""GPU GRPO-v2 correctness pilot on the legal mixed-resolution action graph."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from mrna_editflow.core.mixed_resolution_state import MixedResolutionState, apply_action
from mrna_editflow.data.nmi_benchmark_v2 import iter_role_records
from mrna_editflow.models.legal_action_policy import LegalActionPolicy
from mrna_editflow.rl.grpo_v2 import grpo_loss


def analytic_reward(source: MixedResolutionState, candidate: MixedResolutionState) -> float:
    # Deterministic optimization oracle for a correctness pilot, not a
    # biological label or a claim about measured protein output.
    score = 0.0
    for i, (a, b) in enumerate(zip(source.five_utr, candidate.five_utr)):
        score += ((ord(b) - ord(a)) % 7) * (1.0 + (i % 5) * 0.1)
    for i, (a, b) in enumerate(zip(source.cds, candidate.cds)):
        score += ((ord(b) - ord(a)) % 5) * 0.03 * (i + 1)
    return score


def load_state(root: Path) -> MixedResolutionState:
    row = next(iter_role_records(root / "manifests" / "train.json"))
    return MixedResolutionState(str(row["source_sequence"]), "AUGUAA", "", str(row.get("cargo_id", "")), str(row.get("cell_context", "")), str(row.get("source_id", "")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-root", default="data/nmi_benchmark_v2")
    ap.add_argument("--out", default="artifacts/phase4_policy/grpo_v2_pilot.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--hidden-dim", type=int, default=64)
    args = ap.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Phase 4 policy training must run on GPU")
    device = torch.device(args.device)
    state = load_state(Path(args.benchmark_root))
    policy = LegalActionPolicy(hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=3e-4)
    actions = None
    history = []
    for step in range(args.steps):
        logp, actions = policy.log_probs(state, actions)
        with torch.no_grad():
            rewards = torch.tensor([analytic_reward(state, apply_action(state, a)) for a in actions], device=device)
            old = logp.detach(); ref = logp.detach()
            groups = torch.zeros_like(rewards, dtype=torch.long)
        terms = grpo_loss(logp, old, ref, rewards, groups)
        opt.zero_grad(set_to_none=True); terms["loss"].backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        history.append({"step": step, "loss": float(terms["loss"].detach()), "policy_loss": float(terms["policy_loss"].detach()), "kl": float(terms["kl"].detach()), "entropy": float(terms["entropy"].detach()), "action_count": len(actions), "device": str(device)})
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": policy.state_dict(), "history": history, "device": str(device), "synthetic_oracle": True, "final_test_used": False}, out)
    report = out.with_suffix(".json"); report.write_text(json.dumps({"history": history, "checkpoint": str(out), "synthetic_oracle": True, "final_test_used": False}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checkpoint": str(out), "device": str(device), "steps": len(history)}, indent=2))


if __name__ == "__main__":
    main()
