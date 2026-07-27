#!/usr/bin/env python3
"""Equal-query smoke benchmark for policy/search decision makers."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable

from mrna_editflow.core.mixed_resolution_state import MixedAction, MixedResolutionState, apply_action, legal_actions
from mrna_editflow.data.nmi_benchmark_v2 import iter_role_records
from mrna_editflow.search.beam import beam_search
from mrna_editflow.search.exact_search import SearchResult, exact_search
from mrna_editflow.search.greedy import greedy_search
from mrna_editflow.search.mcts import mcts_search


def score_fn(source: MixedResolutionState) -> Callable[[MixedResolutionState], float]:
    def score(candidate: MixedResolutionState) -> float:
        value = 0.0
        for i, (a, b) in enumerate(zip(source.five_utr, candidate.five_utr)):
            if a != b:
                value += ((ord(b) - ord(a)) % 7) * (1.0 + (i % 5) * 0.1)
        for i, (a, b) in enumerate(zip(source.cds, candidate.cds)):
            if a != b:
                value += ((ord(b) - ord(a)) % 5) * 0.03 * (i + 1)
        return value
    return score


def random_search(state: MixedResolutionState, scorer, budget: int, seed: int = 0) -> SearchResult:
    rng = random.Random(seed); current = state; path = []; best = float(scorer(state)); calls = 1
    for _ in range(budget):
        acts = [a for a in legal_actions(current) if not a.is_stop()]
        if not acts: break
        action = rng.choice(acts); current = apply_action(current, action); path.append(action); calls += 1
        best = max(best, float(scorer(current)))
    return SearchResult(current, tuple(path), best, calls)


def state_from_row(row: dict) -> MixedResolutionState:
    return MixedResolutionState(str(row["source_sequence"]), "AUGUAA", "", str(row.get("cargo_id", "")), str(row.get("cell_context", "")), str(row.get("source_id", "")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-root", default="data/nmi_benchmark_v2")
    ap.add_argument("--out", default="artifacts/phase4_policy/benchmark_v2.json")
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--policy-checkpoint", default="")
    args = ap.parse_args()
    rows = list(iter_role_records(Path(args.benchmark_root) / "manifests" / "train.json"))[: args.limit]
    results = []
    for row in rows:
        state = state_from_row(row); scorer = score_fn(state)
        for budget in ([1] if args.smoke else [1, 3]):
            methods = {
                "random_legal": random_search(state, scorer, budget, seed=7),
                "exact_enumeration": exact_search(state, scorer, budget, max_nodes=128 if args.smoke else 2048),
                "greedy": greedy_search(state, scorer, budget),
                "beam": beam_search(state, scorer, budget, beam_width=4),
                "mcts": mcts_search(state, scorer, budget, simulations=8 if args.smoke else 32, seed=7),
            }
            if args.policy_checkpoint and budget == 1:
                import torch
                from mrna_editflow.models.legal_action_policy import LegalActionPolicy
                from mrna_editflow.search.exact_search import SearchResult
                payload = torch.load(args.policy_checkpoint, map_location="cpu", weights_only=False)
                policy = LegalActionPolicy(hidden_dim=64)
                policy.load_state_dict(payload["model"])
                policy.eval()
                logp, policy_actions = policy.log_probs(state)
                idx = int(logp.argmax())
                action = policy_actions[idx]
                candidate = apply_action(state, action)
                methods["grpo_v2_policy"] = SearchResult(candidate, (action,), float(scorer(candidate)), 0)
            for name, result in methods.items():
                training_calls = 0
                if name == "grpo_v2_policy" and args.policy_checkpoint:
                    training_calls = int(sum(int(h.get("action_count", 0)) for h in json.loads(Path(args.policy_checkpoint).with_suffix(".json").read_text()).get("history", [])))
                results.append({"source_id": state.transcript_id, "budget": budget, "method": name, "score": result.score, "query_calls": result.query_calls, "verification_calls": 1, "training_oracle_calls": training_calls, "synthetic_oracle": True})
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "phase4_policy_benchmark_v2", "results": results, "equal_query_accounting": True, "final_test_used": False, "biological_claim_eligible": False, "reason": "synthetic oracle smoke; replace with frozen independent oracle before headline use"}
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
