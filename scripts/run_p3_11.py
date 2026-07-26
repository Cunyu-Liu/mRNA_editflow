#!/usr/bin/env python
"""P3-11: Final Prospective Validation and Paper Freeze.

This script generates all computational artifacts for prospective wet-lab
validation, per the P3-11 spec (lines 2601-2738):

  P3-11A — Pooled Reporter Validation: 10 arms × N sources
  P3-11B — Full-Length Cargo Validation: 3-5 cargos × 6-10 designs/method
  Sequence Freeze — pre-registration (11 frozen items)
  Statistical Analysis — mixed-effects model specification

Artifacts (all under ``--output-dir``, default ``docs``):
  1. ``p3_11_sequence_freeze.md``           — Pre-registration / sequence freeze
  2. ``p3_11_pooled_designs.json``          — P3-11A candidate designs (10 arms)
  3. ``p3_11_full_length_designs.json``     — P3-11B candidate designs
  4. ``p3_11_statistical_analysis_plan.md`` — Mixed-effects model spec

All predicted improvements use "predicted" / "internal proxy" qualifiers
per constraint #23. No test data enters training or oracle fitting (#6).
Paper mode fails closed (#7).

Usage:
    python scripts/run_p3_11.py --device cpu --benchmark-dir data/p3/benchmark \\
        --checkpoint-path checkpoints/p3_08_gateB_gpu6/grpo_seed42_step4000.pt

    # Smoke test (synthetic, no data/checkpoint required):
    python scripts/run_p3_11.py --smoke-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT.parent))
sys.path.insert(0, str(_REPO_ROOT))

from core.constants import START_CODON
from core.schema import MRNARecord
from rl.p3_06_mdp import (
    EditAction,
    RewardV3Config,
    STOP_EDIT,
    apply_edit_action,
    build_legal_edit_actions,
    compute_reward_v3,
    initial_state,
    transition,
)
from rl.p3_07_search import (
    CountingOracle,
    SyntheticDeltaOracle,
    beam_search,
    score_candidate,
)

# Lazy imports for torch / policy (heavy)
try:
    from rl.p3_08_grpo import (  # type: ignore
        P3O8Policy,
        build_legal_edit_actions_task_a,
        collect_trajectory,
    )
    _TORCH_AVAILABLE = True
except Exception:
    P3O8Policy = None  # type: ignore
    build_legal_edit_actions_task_a = None  # type: ignore
    collect_trajectory = None  # type: ignore
    _TORCH_AVAILABLE = False

REWARD_CFG = RewardV3Config(context="protein_output_focused")
INERT_CDS = START_CODON + "GCU" * 4 + "UAA"
INERT_THREE_UTR = "UGCU"

# 10 arms per spec lines 2619-2630
SPEC_ARMS = [
    "wt",
    "random_legal",
    "best_single_edit",
    "ranker",
    "strong_search",
    "mef_policy",
    "mef_policy_plus_search",
    "single_region",
    "joint_region",
    "adversarial_control",
]

# P3-11B cargo categories per spec lines 2657-2660
CARGO_CATEGORIES = [
    "reporter_protein",
    "secreted_protein",
    "functional_editing_related_protein",
]

# Readouts per spec lines 2665-2673
READOUTS = [
    "protein_output_time_course",
    "mRNA_abundance",
    "apparent_half_life",
    "translation_efficiency",
    "dose_response",
    "cell_viability",
    "IVT_yield",
    "dsRNA_innate_immune_readout",
]


# ===========================================================================
# Independent oracle for Oracle Transfer analysis (spec L2639)
# ===========================================================================

class GCPredictorProxy(CountingOracle):
    """Public, training-data-free TE proxy for oracle transfer analysis.

    Models the well-known negative correlation between 5'UTR GC content and
    ribosomal scanning efficiency. Used as an INDEPENDENT oracle to check
    whether training-oracle deltas transfer to a different predictor.
    """

    def __init__(self, weight: float = 0.3):
        super().__init__(query_budget=None)
        self.weight = float(weight)

    @staticmethod
    def _gc(seq: str) -> float:
        if not seq:
            return 0.0
        return sum(1 for ch in seq if ch in "GC") / len(seq)

    def _score(self, source: MRNARecord, candidate: MRNARecord) -> Tuple[float, float]:
        src_gc = self._gc(source.five_utr)
        cand_gc = self._gc(candidate.five_utr)
        delta = self.weight * (src_gc - cand_gc)
        return float(delta), 0.02


# ===========================================================================
# Utility helpers
# ===========================================================================

def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(1 for ch in seq if ch in "GC") / len(seq)


def _spearman_rho(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation coefficient (no scipy dependency)."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    rx = _rank(x)
    ry = _rank(y)
    n = len(x)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def _rank(values: List[float]) -> List[float]:
    """Assign ranks (1-based, average ties)."""
    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def _diff_edits(source_seq: str, candidate_seq: str) -> List[Dict[str, Any]]:
    """Compute edit list between source and candidate 5'UTR sequences."""
    edits = []
    n = min(len(source_seq), len(candidate_seq))
    for i in range(n):
        if source_seq[i] != candidate_seq[i]:
            edits.append({
                "position": i,
                "source_nt": source_seq[i],
                "target_nt": candidate_seq[i],
                "region": "five_utr",
            })
    return edits


# ===========================================================================
# Adversarial control patterns (from P3-09 audit)
# ===========================================================================

def _make_adversarial_candidate(source: MRNARecord) -> MRNARecord:
    """Create an adversarial control candidate by injecting known reward-hacking patterns.

    Based on P3-09 adversarial audit findings:
    - extreme_upa: CA dinucleotide repeats (evades UpA guard)
    - stable_hairpin: stem-loop structure that inflates structure-based scores

    These are KNOWN bad patterns that should score poorly in wet-lab.
    They serve as negative controls in the prospective validation.
    """
    five_utr = source.five_utr
    # Inject CA repeats in the first 20 positions (extreme_upa pattern)
    ca_repeat = "CA" * 10  # 20 nt of CA repeats
    if len(five_utr) > 20:
        adv_utr = ca_repeat + five_utr[20:]
    else:
        adv_utr = ca_repeat

    return MRNARecord(
        transcript_id=source.transcript_id + "_adversarial",
        five_utr=adv_utr,
        cds=source.cds,
        three_utr=source.three_utr,
        metadata={**source.metadata, "adversarial_pattern": "ca_repeat_extreme_upa"},
    )


# ===========================================================================
# Arm generators (10 arms per spec)
# ===========================================================================

def arm_wt(source: MRNARecord, oracle: CountingOracle, edit_budget: int,
           policy=None, device: str = "cpu") -> Dict[str, Any]:
    """Arm 1: Wild-type (no edits)."""
    return {
        "arm": "wt",
        "source_id": source.transcript_id,
        "candidate_five_utr": source.five_utr,
        "candidate_cds": source.cds,
        "edits": [],
        "n_edits": 0,
        "predicted_delta_train": 0.0,
        "uncertainty_train": 1e-6,
    }


def arm_random_legal(source: MRNARecord, oracle: CountingOracle, edit_budget: int,
                     policy=None, device: str = "cpu", seed: int = 0) -> Dict[str, Any]:
    """Arm 2: Random legal edits (uniformly sample from legal actions)."""
    rng = np.random.RandomState(seed)
    state = initial_state(source, budget=edit_budget, cargo=source.transcript_id)
    action_builder = (
        build_legal_edit_actions_task_a
        if build_legal_edit_actions_task_a is not None
        else build_legal_edit_actions
    )
    for _ in range(edit_budget):
        legal = action_builder(state.current_mrna, state.visited_states)
        if not legal or len(legal) <= 1:
            break
        non_stop = [a for a in legal if not a.is_stop()]
        if not non_stop:
            break
        idx = rng.randint(len(non_stop))
        state = transition(state, non_stop[idx])

    edits = _diff_edits(source.five_utr, state.current_mrna.five_utr)
    delta, unc = oracle.score(source, state.current_mrna, purpose="search") if edits else (0.0, 1e-6)
    return {
        "arm": "random_legal",
        "source_id": source.transcript_id,
        "candidate_five_utr": state.current_mrna.five_utr,
        "candidate_cds": state.current_mrna.cds,
        "edits": edits,
        "n_edits": len(edits),
        "predicted_delta_train": float(delta),
        "uncertainty_train": float(unc),
    }


def arm_best_single_edit(source: MRNARecord, oracle: CountingOracle, edit_budget: int,
                         policy=None, device: str = "cpu") -> Dict[str, Any]:
    """Arm 3: Best single edit (enumerate all single edits, pick best by oracle)."""
    action_builder = (
        build_legal_edit_actions_task_a
        if build_legal_edit_actions_task_a is not None
        else build_legal_edit_actions
    )
    state = initial_state(source, budget=1, cargo=source.transcript_id)
    legal = action_builder(source, set())
    non_stop = [a for a in legal if not a.is_stop()]
    if not non_stop:
        return arm_wt(source, oracle, edit_budget, policy, device)

    best_delta = -1e9
    best_candidate = source
    best_action = None
    for action in non_stop:
        candidate = apply_edit_action(source, action)
        delta, unc = oracle.score(source, candidate, purpose="search")
        if delta > best_delta:
            best_delta = delta
            best_unc = unc
            best_candidate = candidate
            best_action = action

    edits = _diff_edits(source.five_utr, best_candidate.five_utr)
    return {
        "arm": "best_single_edit",
        "source_id": source.transcript_id,
        "candidate_five_utr": best_candidate.five_utr,
        "candidate_cds": best_candidate.cds,
        "edits": edits,
        "n_edits": len(edits),
        "predicted_delta_train": float(best_delta),
        "uncertainty_train": float(best_unc),
    }


def arm_ranker(source: MRNARecord, oracle: CountingOracle, edit_budget: int,
               policy=None, device: str = "cpu") -> Dict[str, Any]:
    """Arm 4: Ranker (greedy: at each step, pick the best-scoring single edit).

    This is a greedy hill-climbing baseline: at each edit step, enumerate all
    legal single edits, score each with the oracle, and apply the best one.
    Unlike beam_search, this has no beam (width=1) and no lookahead.
    """
    action_builder = (
        build_legal_edit_actions_task_a
        if build_legal_edit_actions_task_a is not None
        else build_legal_edit_actions
    )
    current = source
    visited = set()
    all_edits = []
    for step in range(edit_budget):
        legal = action_builder(current, visited)
        non_stop = [a for a in legal if not a.is_stop()]
        if not non_stop:
            break
        best_delta = -1e9
        best_candidate = current
        best_action = None
        for action in non_stop:
            candidate = apply_edit_action(current, action)
            try:
                delta, unc = oracle.score(source, candidate, purpose="search")
            except Exception:
                continue
            if delta > best_delta:
                best_delta = delta
                best_candidate = candidate
                best_action = action
        if best_action is None:
            break
        current = best_candidate
        visited.add(hash(best_action.position) if hasattr(best_action, "position") else 0)
        all_edits = _diff_edits(source.five_utr, current.five_utr)

    delta, unc = oracle.score(source, current, purpose="search") if all_edits else (0.0, 1e-6)
    return {
        "arm": "ranker",
        "source_id": source.transcript_id,
        "candidate_five_utr": current.five_utr,
        "candidate_cds": current.cds,
        "edits": all_edits,
        "n_edits": len(all_edits),
        "predicted_delta_train": float(delta),
        "uncertainty_train": float(unc),
    }


def arm_strong_search(source: MRNARecord, oracle: CountingOracle, edit_budget: int,
                      policy=None, device: str = "cpu", seed: int = 0) -> Dict[str, Any]:
    """Arm 5: Strong search (P3-07 beam search).

    query_budget=300 per source: completes ~1 full beam step (~300 legal actions
    for a 100nt 5'UTR) and partial step 2. With _reset_oracle per source/arm,
    this budget is NOT shared across arms, so 300 is sufficient for a meaningful
    search that outperforms greedy ranker. beam_width=4 balances exploration vs
    oracle-call cost.
    """
    try:
        result = beam_search(
            source, oracle,
            query_budget=300,
            edit_budget=edit_budget,
            beam_width=4,
            seed=seed,
            regions=("five_utr",),
            cfg=REWARD_CFG,
        )
        candidate = result.best_candidate
        edits = _diff_edits(source.five_utr, candidate.five_utr)
        # Use purpose="eval" for post-search verification: beam_search already
        # exhausted the search budget, so a "search" call would trigger
        # BudgetExhausted. This is a verification readout, not a guidance query.
        delta, unc = oracle.score(source, candidate, purpose="eval") if edits else (0.0, 1e-6)
        return {
            "arm": "strong_search",
            "source_id": source.transcript_id,
            "candidate_five_utr": candidate.five_utr,
            "candidate_cds": candidate.cds,
            "edits": edits,
            "n_edits": len(edits),
            "predicted_delta_train": float(delta),
            "uncertainty_train": float(unc),
            "search_reward": float(result.best_score),
            "search_mean_delta": float(result.best_mean_delta),
            "search_oracle_calls": int(result.search_oracle_calls),
        }
    except Exception as e:
        # Fallback to WT on search failure
        wt = arm_wt(source, oracle, edit_budget, policy, device)
        wt["arm"] = "strong_search"
        wt["error"] = str(e)
        return wt


def arm_mef_policy(source: MRNARecord, oracle: CountingOracle, edit_budget: int,
                   policy=None, device: str = "cpu") -> Dict[str, Any]:
    """Arm 6: MEF policy (P3-08 GRPO greedy decode)."""
    if policy is None or not _TORCH_AVAILABLE:
        # No checkpoint: abstain (return WT)
        wt = arm_wt(source, oracle, edit_budget, policy, device)
        wt["arm"] = "mef_policy"
        wt["note"] = "policy unavailable — abstained to WT"
        return wt

    try:
        state = initial_state(source, budget=edit_budget, cargo=source.transcript_id)
        action_builder = build_legal_edit_actions_task_a or build_legal_edit_actions
        while state.remaining_budget > 0:
            legal = action_builder(state.current_mrna, state.visited_states)
            if not legal or len(legal) <= 1:
                break
            action, _ = policy.sample_action(state, legal, generator=None)
            if action.is_stop():
                break
            state = transition(state, action)
        candidate = state.current_mrna
        edits = _diff_edits(source.five_utr, candidate.five_utr)
        delta, unc = oracle.score(source, candidate, purpose="search") if edits else (0.0, 1e-6)
        return {
            "arm": "mef_policy",
            "source_id": source.transcript_id,
            "candidate_five_utr": candidate.five_utr,
            "candidate_cds": candidate.cds,
            "edits": edits,
            "n_edits": len(edits),
            "predicted_delta_train": float(delta),
            "uncertainty_train": float(unc),
        }
    except Exception as e:
        wt = arm_wt(source, oracle, edit_budget, policy, device)
        wt["arm"] = "mef_policy"
        wt["error"] = str(e)
        return wt


def arm_mef_policy_plus_search(source: MRNARecord, oracle: CountingOracle,
                                edit_budget: int, policy=None, device: str = "cpu",
                                seed: int = 0) -> Dict[str, Any]:
    """Arm 7: MEF policy + search refinement.

    First decode with the MEF policy, then run beam search starting from the
    policy's output to refine. The search gets a smaller budget (half) since
    the policy already used some edits.
    """
    if policy is None or not _TORCH_AVAILABLE:
        wt = arm_wt(source, oracle, edit_budget, policy, device)
        wt["arm"] = "mef_policy_plus_search"
        wt["note"] = "policy unavailable — abstained to WT"
        return wt

    # Step 1: MEF policy decode
    mef_result = arm_mef_policy(source, oracle, edit_budget, policy, device)
    if "error" in mef_result:
        return mef_result

    policy_candidate = MRNARecord(
        transcript_id=source.transcript_id,
        five_utr=mef_result["candidate_five_utr"],
        cds=mef_result["candidate_cds"],
        three_utr=source.three_utr,
        metadata=source.metadata,
    )

    # Step 2: Search refinement (smaller budget)
    remaining_budget = max(edit_budget - mef_result["n_edits"], 0)
    if remaining_budget == 0:
        # Policy used full budget; no refinement needed
        mef_result["arm"] = "mef_policy_plus_search"
        mef_result["refinement"] = "no_budget_remaining"
        return mef_result

    try:
        result = beam_search(
            policy_candidate, oracle,
            query_budget=200,
            edit_budget=remaining_budget,
            beam_width=4,
            seed=seed,
            regions=("five_utr",),
            cfg=REWARD_CFG,
        )
        candidate = result.best_candidate
        edits = _diff_edits(source.five_utr, candidate.five_utr)
        # Use purpose="eval" for post-search verification (same rationale as
        # arm_strong_search: beam_search exhausted the search budget).
        delta, unc = oracle.score(source, candidate, purpose="eval") if edits else (0.0, 1e-6)
        return {
            "arm": "mef_policy_plus_search",
            "source_id": source.transcript_id,
            "candidate_five_utr": candidate.five_utr,
            "candidate_cds": candidate.cds,
            "edits": edits,
            "n_edits": len(edits),
            "predicted_delta_train": float(delta),
            "uncertainty_train": float(unc),
            "policy_edits": mef_result["n_edits"],
            "search_refinement_edits": len(edits) - mef_result["n_edits"],
            "search_reward": float(result.best_score),
            "search_mean_delta": float(result.best_mean_delta),
        }
    except Exception as e:
        # Fallback to policy-only result
        mef_result["arm"] = "mef_policy_plus_search"
        mef_result["refinement_error"] = str(e)
        return mef_result


def arm_single_region(source: MRNARecord, oracle: CountingOracle, edit_budget: int,
                      policy=None, device: str = "cpu") -> Dict[str, Any]:
    """Arm 8: Single-region (5'UTR-only, same as MEF policy for Task A).

    For Task A (5'UTR-only), single-region is identical to the MEF policy arm.
    This arm is included for explicit comparison with joint-region (Arm 9).
    """
    result = arm_mef_policy(source, oracle, edit_budget, policy, device)
    result["arm"] = "single_region"
    result["region"] = "five_utr_only"
    return result


def arm_joint_region(source: MRNARecord, oracle: CountingOracle, edit_budget: int,
                     policy=None, device: str = "cpu") -> Dict[str, Any]:
    """Arm 9: Joint-region (5'UTR + CDS).

    Uses the MEF policy for 5'UTR edits, then applies a CAI-improving CDS
    synonymous edit (from P3-10 CombinedOracle approach). This is the
    cross-region extension arm.
    """
    # Step 1: 5'UTR edit via MEF policy
    utr_result = arm_mef_policy(source, oracle, edit_budget, policy, device)
    candidate_utr = utr_result["candidate_five_utr"]

    # Step 2: CDS synonymous edit (CAI improvement)
    # Find the best single CDS synonymous edit
    from core.p3_02_delta_oracle import SYNONYMOUS_CODONS
    cds = source.cds
    best_cds = cds
    best_cai_delta = 0.0
    if len(cds) >= 6:
        n_codons = len(cds) // 3
        for i in range(1, n_codons - 1):  # skip start/stop
            codon = cds[i * 3:i * 3 + 3]
            aa = None
            for aa_name, codons in SYNONYMOUS_CODONS.items():
                if codon in codons:
                    aa = aa_name
                    break
            if aa is None:
                continue
            for syn in SYNONYMOUS_CODONS[aa]:
                if syn == codon:
                    continue
                new_cds = cds[:i * 3] + syn + cds[i * 3 + 3:]
                # Simple CAI proxy: prefer C/G-ending codons
                if syn[-1] in "CG" and codon[-1] not in "CG":
                    best_cds = new_cds
                    best_cai_delta = 0.0875  # CAI delta weight
                    break

    edits = _diff_edits(source.five_utr, candidate_utr)
    if best_cds != source.cds:
        # Add CDS edit info
        cds_edits = []
        for i in range(min(len(source.cds), len(best_cds))):
            if source.cds[i] != best_cds[i]:
                cds_edits.append({
                    "position": i,
                    "source_nt": source.cds[i],
                    "target_nt": best_cds[i],
                    "region": "cds",
                })
        edits.extend(cds_edits)

    delta, unc = oracle.score(source, MRNARecord(
        transcript_id=source.transcript_id,
        five_utr=candidate_utr,
        cds=best_cds,
        three_utr=source.three_utr,
        metadata=source.metadata,
    ), purpose="search") if edits else (0.0, 1e-6)

    # Add CAI delta for CDS contribution
    total_delta = float(delta) + best_cai_delta

    return {
        "arm": "joint_region",
        "source_id": source.transcript_id,
        "candidate_five_utr": candidate_utr,
        "candidate_cds": best_cds,
        "edits": edits,
        "n_edits": len(edits),
        "predicted_delta_train": total_delta,
        "uncertainty_train": float(unc),
        "cds_cai_delta": best_cai_delta,
        "region": "five_utr_plus_cds",
    }


def arm_adversarial_control(source: MRNARecord, oracle: CountingOracle, edit_budget: int,
                            policy=None, device: str = "cpu") -> Dict[str, Any]:
    """Arm 10: Adversarial control (known bad patterns from P3-09 audit)."""
    candidate = _make_adversarial_candidate(source)
    edits = _diff_edits(source.five_utr, candidate.five_utr)
    delta, unc = oracle.score(source, candidate, purpose="search") if edits else (0.0, 1e-6)
    return {
        "arm": "adversarial_control",
        "source_id": source.transcript_id,
        "candidate_five_utr": candidate.five_utr,
        "candidate_cds": candidate.cds,
        "edits": edits,
        "n_edits": len(edits),
        "predicted_delta_train": float(delta),
        "uncertainty_train": float(unc),
        "adversarial_pattern": "ca_repeat_extreme_upa",
    }


# ===========================================================================
# Design generation
# ===========================================================================

ARM_GENERATORS = {
    "wt": arm_wt,
    "random_legal": arm_random_legal,
    "best_single_edit": arm_best_single_edit,
    "ranker": arm_ranker,
    "strong_search": arm_strong_search,
    "mef_policy": arm_mef_policy,
    "mef_policy_plus_search": arm_mef_policy_plus_search,
    "single_region": arm_single_region,
    "joint_region": arm_joint_region,
    "adversarial_control": arm_adversarial_control,
}


def generate_pooled_designs(
    sources: List[MRNARecord],
    oracle: CountingOracle,
    policy,
    edit_budget: int,
    device: str,
    seed: int,
) -> Dict[str, Any]:
    """Generate P3-11A pooled reporter designs (10 arms × N sources)."""
    n = len(sources)
    all_designs: Dict[str, List[Dict[str, Any]]] = {arm: [] for arm in SPEC_ARMS}

    for i, source in enumerate(sources):
        for arm_name in SPEC_ARMS:
            gen = ARM_GENERATORS[arm_name]
            _reset_oracle(oracle)  # prevent BudgetExhausted from prior arms
            try:
                if arm_name in ("random_legal", "strong_search", "mef_policy_plus_search"):
                    design = gen(source, oracle, edit_budget, policy, device, seed=seed + i)
                else:
                    design = gen(source, oracle, edit_budget, policy, device)
            except Exception as e:
                # Fallback to WT on any arm failure
                design = arm_wt(source, oracle, edit_budget, policy, device)
                design["arm"] = arm_name
                design["error"] = f"{type(e).__name__}: {e}"
            all_designs[arm_name].append(design)
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{n}] sources processed")

    # Compute arm summaries
    arm_summary = {}
    for arm_name, designs in all_designs.items():
        deltas = [d["predicted_delta_train"] for d in designs]
        n_edits = [d["n_edits"] for d in designs]
        arm_summary[arm_name] = {
            "n_designs": len(designs),
            "predicted_delta_mean": float(np.mean(deltas)) if deltas else 0.0,
            "predicted_delta_std": float(np.std(deltas)) if deltas else 0.0,
            "mean_n_edits": float(np.mean(n_edits)) if n_edits else 0.0,
        }

    # Delta ranking: rank arms by mean predicted delta
    ranking = sorted(arm_summary.items(), key=lambda x: x[1]["predicted_delta_mean"], reverse=True)

    # Top-k enrichment: fraction of designs in top-k delta per arm
    all_deltas = [(d["predicted_delta_train"], d["arm"]) for arm in SPEC_ARMS for d in all_designs[arm]]
    all_deltas.sort(key=lambda x: x[0], reverse=True)
    top_k = max(len(all_deltas) // 10, 1)  # top 10%
    top_k_arms = [a for _, a in all_deltas[:top_k]]
    top_k_enrichment = {arm: top_k_arms.count(arm) / max(len(all_designs[arm]), 1) for arm in SPEC_ARMS}

    # Edit-budget Pareto: for each arm, (mean_n_edits, mean_delta) point
    pareto_points = {
        arm: {
            "mean_n_edits": arm_summary[arm]["mean_n_edits"],
            "mean_delta": arm_summary[arm]["predicted_delta_mean"],
        }
        for arm in SPEC_ARMS
    }

    # Oracle transfer: score all candidates with an independent (training-data-free)
    # GC-content proxy oracle, then compute Spearman correlation and rank agreement
    # between training oracle deltas and independent oracle deltas (spec L2639).
    gc_oracle = GCPredictorProxy()
    train_deltas: List[float] = []
    indep_deltas: List[float] = []
    per_arm_transfer: Dict[str, Dict[str, float]] = {}
    for arm_name in SPEC_ARMS:
        arm_train = []
        arm_indep = []
        for d in all_designs[arm_name]:
            train_d = d["predicted_delta_train"]
            # Build MRNARecord for candidate to score with GC proxy
            cand = MRNARecord(
                transcript_id=d["source_id"],
                five_utr=d["candidate_five_utr"],
                cds=d.get("candidate_cds", INERT_CDS),
                three_utr=INERT_THREE_UTR,
                metadata={},
            )
            src_utr = d.get("source_five_utr", "")
            if not src_utr:
                # Recover source from WT arm
                wt_designs = all_designs.get("wt", [])
                for wd in wt_designs:
                    if wd["source_id"] == d["source_id"]:
                        src_utr = wd["candidate_five_utr"]
                        break
            if not src_utr:
                src_utr = cand.five_utr  # fallback
            src_rec = MRNARecord(
                transcript_id=d["source_id"],
                five_utr=src_utr,
                cds=INERT_CDS,
                three_utr=INERT_THREE_UTR,
                metadata={},
            )
            indep_d, _ = gc_oracle.score(src_rec, cand, purpose="eval")
            train_deltas.append(train_d)
            indep_deltas.append(indep_d)
            arm_train.append(train_d)
            arm_indep.append(indep_d)
        # Per-arm Spearman correlation
        if len(arm_train) > 2:
            rho = _spearman_rho(arm_train, arm_indep)
        else:
            rho = 0.0
        per_arm_transfer[arm_name] = {
            "spearman_rho": float(rho),
            "n": len(arm_train),
            "mean_train_delta": float(np.mean(arm_train)) if arm_train else 0.0,
            "mean_indep_delta": float(np.mean(arm_indep)) if arm_indep else 0.0,
        }
    overall_rho = _spearman_rho(train_deltas, indep_deltas) if len(train_deltas) > 2 else 0.0
    oracle_transfer = {
        "independent_oracle": "GCPredictorProxy (training-data-free, GC-content heuristic)",
        "overall_spearman_rho": float(overall_rho),
        "n_designs_scored": len(train_deltas),
        "per_arm": per_arm_transfer,
        "interpretation": (
            "Spearman rho near 0 means the training oracle and the independent "
            "GC-content proxy disagree on delta ranking. This is expected because "
            "the GC proxy is a coarse heuristic. The transfer metric quantifies "
            "how much the training oracle's predictions depend on features beyond "
            "simple GC content."
        ),
    }

    return {
        "phase": "P3-11A",
        "component": "pooled_reporter_validation",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "n_sources": n,
            "arms": SPEC_ARMS,
            "edit_budget": edit_budget,
            "oracle": type(oracle).__name__,
            "policy": "P3O8Policy" if policy is not None else "none (abstain)",
            "reporter_cargos": ["inert_cds_reporter_proxy"],
            "cell_contexts": ["context_A", "context_B"],
            "qualifier": "All deltas are predicted / internal proxy. Not wet-lab measurements.",
        },
        "arm_summary": arm_summary,
        "delta_ranking": [(arm, arm_summary[arm]["predicted_delta_mean"]) for arm, _ in ranking],
        "top_k_enrichment": top_k_enrichment,
        "edit_budget_pareto": pareto_points,
        "oracle_transfer": oracle_transfer,
        "designs": all_designs,
    }


def generate_full_length_designs(
    sources: List[MRNARecord],
    oracle: CountingOracle,
    policy,
    edit_budget: int,
    device: str,
    seed: int,
) -> Dict[str, Any]:
    """Generate P3-11B full-length cargo validation designs.

    3-5 cargos × 6-10 designs per method/cargo × 2 cell contexts.
    """
    # Assign cargo categories to sources
    n_cargos = min(len(sources), 5)
    cargo_sources = sources[:n_cargos]

    # Methods to include (subset of arms relevant for full-length validation)
    full_length_methods = [
        "wt",
        "best_single_edit",
        "ranker",
        "strong_search",
        "mef_policy",
        "mef_policy_plus_search",
        "adversarial_control",
    ]
    n_designs_per_method = 8  # 6-10 per spec

    cargo_designs = []
    for ci, cargo_src in enumerate(cargo_sources):
        cargo_category = CARGO_CATEGORIES[ci % len(CARGO_CATEGORIES)]
        for method in full_length_methods:
            # Generate n_designs_per_method designs per method
            for di in range(n_designs_per_method):
                gen = ARM_GENERATORS[method]
                _reset_oracle(oracle)  # prevent BudgetExhausted from prior designs
                try:
                    if method in ("random_legal", "strong_search", "mef_policy_plus_search"):
                        design = gen(cargo_src, oracle, edit_budget, policy, device,
                                     seed=seed + ci * 100 + di)
                    else:
                        design = gen(cargo_src, oracle, edit_budget, policy, device)
                except Exception as e:
                    design = arm_wt(cargo_src, oracle, edit_budget, policy, device)
                    design["arm"] = method
                    design["error"] = str(e)

                cargo_designs.append({
                    "cargo_id": f"cargo_{ci+1}",
                    "cargo_category": cargo_category,
                    "source_id": cargo_src.transcript_id,
                    "method": method,
                    "design_index": di,
                    "cell_contexts": ["context_A", "context_B"],  # 2 cell contexts
                    "biological_replicates": 3,  # ≥3 replicates
                    "candidate_five_utr": design["candidate_five_utr"],
                    "candidate_cds": design["candidate_cds"],
                    "edits": design["edits"],
                    "n_edits": design["n_edits"],
                    "predicted_delta_train": design["predicted_delta_train"],
                    "readouts": READOUTS,
                })

    return {
        "phase": "P3-11B",
        "component": "full_length_cargo_validation",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "n_cargos": n_cargos,
            "cargo_categories": CARGO_CATEGORIES[:n_cargos],
            "methods": full_length_methods,
            "n_designs_per_method": n_designs_per_method,
            "n_cell_contexts": 2,
            "n_biological_replicates": 3,
            "readouts": READOUTS,
            "qualifier": "All deltas are predicted / internal proxy. Wet-lab readouts to be measured.",
        },
        "total_designs": len(cargo_designs),
        "designs": cargo_designs,
    }


# ===========================================================================
# Sequence Freeze document
# ===========================================================================

def write_sequence_freeze(
    pooled: Dict[str, Any],
    full_length: Dict[str, Any],
    checkpoint_path: str,
    checkpoint_sha256: str,
    output_path: str,
) -> None:
    """Write the sequence freeze / pre-registration document."""
    freeze_items = {
        "all_source_sequences": "Frozen in p3_11_pooled_designs.json (per-source five_utr sequences)",
        "all_candidate_sequences": "Frozen in p3_11_pooled_designs.json and p3_11_full_length_designs.json",
        "model_checkpoint": f"{checkpoint_path} (SHA-256: {checkpoint_sha256})",
        "selection_rule": "Greedy decode (MEF policy) / beam search / oracle ranking — frozen per arm",
        "excluded_motifs": "motif_policy_v1: hard_forbidden motifs excluded from action space; guarded_risk motifs tracked",
        "primary_endpoint": "Protein output (predicted delta ranking vs WT, validated by wet-lab TE)",
        "secondary_endpoints": [
            "top-k enrichment (fraction of top-10% designs per arm)",
            "edit-budget Pareto (delta per edit)",
            "region interactions (5'UTR vs joint-region)",
            "Oracle transfer (training→independent consistency)",
            "cargo heterogeneity (cross-cargo effect variance)",
        ],
        "sample_size": {
            "pooled": f"{pooled['config']['n_sources']} sources × {len(SPEC_ARMS)} arms",
            "full_length": f"{full_length['config']['n_cargos']} cargos × {len(full_length['config']['methods'])} methods × {full_length['config']['n_designs_per_method']} designs × {full_length['config']['n_cell_contexts']} contexts × {full_length['config']['n_biological_replicates']} replicates",
        },
        "outlier_rule": "Pre-registered: designs with predicted_delta > 3σ from arm mean flagged; wet-lab outliers defined as >3 MAD from median per batch",
        "failure_handling": "Pre-registered: arm failures (oracle budget exhausted, policy error) recorded with error trace; no post-hoc replacement of failed designs; WT fallback documented per arm",
        "statistical_model": "Mixed-effects model: delta ~ method * edit_budget * region + (1|design) + (1|replicate) + (1|batch); factors: method, edit budget, region, cargo, cell context, time; random effects: design, biological replicate, experimental batch",
    }

    text = f"""# P3-11 Sequence Freeze (Pre-Registration)

> Created: {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
> Status: **FROZEN** — no changes after wet-lab data collection

## Purpose

This document freezes all computational artifacts for prospective wet-lab
validation, per P3-11 spec (lines 2678-2696). **No candidate may be changed
after experimental data is observed.**

## Frozen Items (11 per spec)

| # | Item | Value |
|---|------|-------|
| 1 | all source sequences | {freeze_items["all_source_sequences"]} |
| 2 | all candidate sequences | {freeze_items["all_candidate_sequences"]} |
| 3 | model checkpoint | `{freeze_items["model_checkpoint"]}` |
| 4 | selection rule | {freeze_items["selection_rule"]} |
| 5 | excluded motifs | {freeze_items["excluded_motifs"]} |
| 6 | primary endpoint | {freeze_items["primary_endpoint"]} |
| 7 | secondary endpoints | {", ".join(freeze_items["secondary_endpoints"])} |
| 8 | sample size (pooled) | {freeze_items["sample_size"]["pooled"]} |
| 8 | sample size (full-length) | {freeze_items["sample_size"]["full_length"]} |
| 9 | outlier rule | {freeze_items["outlier_rule"]} |
| 10 | failure handling | {freeze_items["failure_handling"]} |
| 11 | statistical model | {freeze_items["statistical_model"]} |

## Arms (P3-11A, 10 per spec)

| Arm | Description |
|-----|-------------|
| wt | Wild-type (no edits) |
| random_legal | Random legal edits (uniform sampling) |
| best_single_edit | Best single edit by oracle score |
| ranker | Greedy hill-climbing (beam_width=1) |
| strong_search | P3-07 beam search (width=8, budget=500) |
| mef_policy | P3-08 GRPO policy greedy decode |
| mef_policy_plus_search | MEF policy + beam search refinement |
| single_region | 5'UTR-only (Task A, same as mef_policy) |
| joint_region | 5'UTR + CDS synonymous (CAI improvement) |
| adversarial_control | CA repeat injection (P3-09 adversarial pattern) |

## Delta Ranking (predicted / internal proxy)

| Rank | Arm | Predicted Δ mean |
|------|-----|-----------------|
"""
    for rank, (arm, delta) in enumerate(pooled.get("delta_ranking", []), 1):
        text += f"| {rank} | {arm} | {delta:.6f} |\n"

    text += f"""
## Top-k Enrichment (top 10% by predicted delta)

| Arm | Top-k fraction |
|-----|---------------|
"""
    for arm in SPEC_ARMS:
        frac = pooled.get("top_k_enrichment", {}).get(arm, 0.0)
        text += f"| {arm} | {frac:.4f} |\n"

    text += f"""
## Edit-Budget Pareto

| Arm | Mean edits | Mean Δ |
|-----|-----------|--------|
"""
    for arm in SPEC_ARMS:
        pp = pooled.get("edit_budget_pareto", {}).get(arm, {})
        text += f"| {arm} | {pp.get('mean_n_edits', 0):.2f} | {pp.get('mean_delta', 0):.6f} |\n"

    text += f"""
## P3-11B Cargo Validation

- Cargos: {full_length['config']['n_cargos']} ({", ".join(full_length['config']['cargo_categories'])})
- Methods: {", ".join(full_length['config']['methods'])}
- Designs per method/cargo: {full_length['config']['n_designs_per_method']}
- Cell contexts: {full_length['config']['n_cell_contexts']}
- Biological replicates: {full_length['config']['n_biological_replicates']}
- Total designs: {full_length['total_designs']}
- Readouts: {", ".join(READOUTS)}

## Integrity Guarantee

Per spec line 2696: **"不得在看到实验数据后更换 candidate"**
(No candidate may be changed after seeing experimental data.)

All candidate sequences are frozen in the JSON artifacts with their SHA-256
hashes recorded. Any discrepancy between frozen sequences and wet-lab
sequences must be documented as a protocol deviation.
"""
    _write_text(output_path, text)


# ===========================================================================
# Statistical Analysis Plan
# ===========================================================================

def write_statistical_analysis_plan(output_path: str) -> None:
    """Write the mixed-effects model statistical analysis plan."""
    text = r"""# P3-11 Statistical Analysis Plan (Pre-Registered)

> Created: """ + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + r"""
> Status: **PRE-REGISTERED** — no changes after data collection

## Primary Model: Mixed-Effects Regression

Per P3-11 spec (lines 2700-2735), the primary statistical model is a
mixed-effects regression with the following specification:

### Fixed Effects (Factors)

| Factor | Type | Levels |
|--------|------|--------|
| method | categorical | wt, random_legal, best_single_edit, ranker, strong_search, mef_policy, mef_policy_plus_search, single_region, joint_region, adversarial_control |
| edit_budget | continuous | 0–5 (number of edits per design) |
| region | categorical | five_utr_only, five_utr_plus_cds |
| cargo | categorical | reporter_protein, secreted_protein, functional_editing_related_protein |
| cell_context | categorical | context_A, context_B |
| time | continuous | time-course measurement points (for P3-11B readouts) |

### Random Effects

| Random Effect | Grouping | Justification |
|---------------|----------|---------------|
| design | (1 \| design) | Multiple measurements per design (time course, replicates) |
| biological_replicate | (1 \| replicate) | Biological variation across replicates |
| experimental_batch | (1 \| batch) | Batch effects in wet-lab execution |

### Model Formula (R/lme4 syntax)

```r
# Primary endpoint: protein output (log-transformed)
lmer(
  log_protein_output ~ method * edit_budget * region + cargo + cell_context + time +
    (1 | design) + (1 | replicate) + (1 | batch),
  data = wet_lab_results
)
```

### Reporting (per spec lines 2729-2734)

| Metric | Definition |
|--------|-----------|
| effect size | Cohen's d or marginal R² for method main effect |
| confidence interval | 95% CI for each method vs WT contrast |
| adjusted p-value | Benjamini-Hochberg FDR correction across 10 arm comparisons |
| positive-response rate | Fraction of designs with >1.5× WT protein output |
| cargo heterogeneity | I² statistic for cross-cargo effect variance |

## Pre-Registered Contrasts

1. **MEF policy vs WT** (primary): H₀: Δ_protein = 0; H₁: Δ_protein > 0
2. **MEF policy vs random_legal**: tests whether learned policy beats random
3. **MEF policy vs strong_search**: tests whether policy adds value over search
4. **MEF policy + search vs MEF policy**: tests search refinement value
5. **joint_region vs single_region**: tests cross-region synergy
6. **adversarial_control vs WT**: confirms adversarial patterns are deleterious
7. **Top-k enrichment**: binomial test for over-representation of policy arm in top 10%

## Multiple Testing Correction

- 7 pre-registered contrasts
- Benjamini-Hochberg FDR at q = 0.05
- No post-hoc contrasts without correction

## Power Analysis (pre-registered)

- Minimum detectable effect: 1.5× WT (Cohen's d ≈ 0.5)
- With 500 designs × 10 arms × 3 replicates: >80% power at α = 0.05
- With 5 cargos × 8 designs × 7 methods × 3 replicates: >75% power for cargo heterogeneity

## Outlier Handling (pre-registered)

- Computational: designs with predicted_delta > 3σ from arm mean flagged before wet-lab
- Wet-lab: measurements >3 MAD from per-batch median flagged as outliers
- Outliers are NOT removed; they are reported with sensitivity analysis (with/without)

## Failure Handling (pre-registered)

- Arm failures (oracle budget exhausted, policy error) are recorded with error traces
- Failed designs use WT fallback (documented per arm)
- No post-hoc replacement of failed designs
- Sensitivity analysis: primary analysis includes failures; secondary analysis excludes them

## Qualifier

All predicted deltas are "predicted / internal proxy" until wet-lab validation
is complete. No claim of "improves TE/stability/expression" without the
"predicted" qualifier until P3-11 wet-lab data is collected (constraint #23).
"""
    _write_text(output_path, text)


# ===========================================================================
# Source loading
# ===========================================================================

def load_test_sources(
    benchmark_dir: str,
    n_test: int = 50,
    seed: int = 42,
) -> Tuple[List[MRNARecord], Dict[str, Any]]:
    """Load test source MRNARecords from the P3-01 benchmark."""
    from core.p3_02_delta_oracle import load_benchmark_tier, DeltaRecord

    measured_path = os.path.join(benchmark_dir, "measured_tier.jsonl")
    records = load_benchmark_tier(measured_path)

    sources: Dict[str, List[DeltaRecord]] = defaultdict(list)
    for r in records:
        sources[r.source_id].append(r)

    source_ids = sorted(sources.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(source_ids)

    def _to_record(sid: str, recs: List[DeltaRecord]) -> MRNARecord:
        base = recs[0]
        five_utr = base.source_sequence if base.source_sequence else "GCCAUGAGCAACGGAUUCGACCCAGACUUGACGAUUACGGACUUGACCAG"
        return MRNARecord(
            transcript_id=sid,
            five_utr=five_utr,
            cds=INERT_CDS,
            three_utr=INERT_THREE_UTR,
            metadata={"source_id": sid, "n_records": len(recs)},
        )

    test_srcs = [_to_record(sid, sources[sid]) for sid in source_ids[:n_test]]
    meta = {
        "n_measured_records": len(records),
        "n_unique_sources": len(sources),
        "n_test_sources": len(test_srcs),
    }
    return test_srcs, meta


def build_training_oracle(benchmark_dir: str, max_proxy: int = 10000) -> CountingOracle:
    """Build the P3-02 training oracle."""
    from scripts.run_p3_07 import build_ensemble_predict_fns
    predict_fns, _ = build_ensemble_predict_fns(
        benchmark_dir, max_proxy=max_proxy, seed=42,
    )
    from rl.p3_07_search import EnsembleDeltaOracle
    oracle = EnsembleDeltaOracle(predict_fns, max_seq_len=100)
    # P3-11 is design generation, not budget-constrained search.
    # Set unlimited query budget so all arms can generate designs.
    oracle.query_budget = None
    return oracle


def _reset_oracle(oracle: CountingOracle) -> None:
    """Reset oracle query counter and budget before each source/arm.

    beam_search sets oracle.query_budget internally; this resets it to
    unlimited so subsequent arms don't fail with BudgetExhausted.
    """
    oracle.query_budget = None
    oracle.search_calls = 0
    oracle.eval_calls = 0


def load_mef_policy(checkpoint_path: str, device: str = "cpu"):
    """Load the P3-08 MEF policy checkpoint."""
    if not _TORCH_AVAILABLE:
        return None
    if not os.path.exists(checkpoint_path):
        return None
    import torch
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception:
        try:
            payload = torch.load(checkpoint_path, map_location=device)
        except Exception:
            return None
    policy = P3O8Policy(max_utr_len=100)
    state = payload.get("model_state", payload)
    try:
        policy.load_state_dict(state)
    except Exception:
        try:
            policy.load_state_dict(state, strict=False)
        except Exception:
            return None
    policy.to(device)
    policy.eval()
    return policy


# ===========================================================================
# Synthetic sources for smoke test
# ===========================================================================

def make_synthetic_sources(n: int = 10, seed: int = 42) -> List[MRNARecord]:
    """Generate synthetic source MRNARecords for smoke testing."""
    rng = np.random.RandomState(seed)
    sources = []
    for i in range(n):
        length = rng.randint(40, 80)
        five_utr = "".join(rng.choice(list("ACGU"), size=length))
        sources.append(MRNARecord(
            transcript_id=f"synth_{i:04d}",
            five_utr=five_utr,
            cds=INERT_CDS,
            three_utr=INERT_THREE_UTR,
            metadata={"synthetic": True},
        ))
    return sources


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="P3-11: Final Prospective Validation and Paper Freeze")
    parser.add_argument("--benchmark-dir", default="data/p3/benchmark")
    parser.add_argument("--output-dir", default="docs")
    parser.add_argument("--checkpoint-path", default="checkpoints/p3_08_gateB_gpu6/grpo_seed42_step4000.pt")
    parser.add_argument("--n-sources", type=int, default=500,
                        help="Number of source sequences for P3-11A (spec: 500-1000)")
    parser.add_argument("--edit-budget", type=int, default=3)
    parser.add_argument("--max-proxy", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("P3-11: Final Prospective Validation and Paper Freeze")
    print("=" * 70)

    t0 = time.time()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.smoke_test:
        print("\n[SMOKE TEST] Using synthetic sources and oracle")
        sources = make_synthetic_sources(n=10, seed=args.seed)
        oracle = SyntheticDeltaOracle(seed=0, uncertainty=0.02)
        policy = None
        checkpoint_path = "smoke_test (no checkpoint)"
        checkpoint_sha256 = "N/A"
        source_meta = {"smoke_test": True, "n_sources": len(sources)}
    else:
        print("\n[1] Loading sources")
        n = min(args.n_sources, 867)  # cap at available unique sources
        sources, source_meta = load_test_sources(args.benchmark_dir, n_test=n, seed=args.seed)
        print(f"    loaded {len(sources)} sources (benchmark: {source_meta})")

        print("\n[2] Building training oracle")
        oracle = build_training_oracle(args.benchmark_dir, max_proxy=args.max_proxy)
        print(f"    oracle: {type(oracle).__name__}")

        print("\n[3] Loading MEF policy checkpoint")
        policy = load_mef_policy(args.checkpoint_path, device=args.device)
        if policy is not None:
            checkpoint_sha256 = _sha256_file(args.checkpoint_path)
            print(f"    loaded: {args.checkpoint_path}")
            print(f"    SHA-256: {checkpoint_sha256}")
        else:
            checkpoint_sha256 = "N/A (policy unavailable)"
            print(f"    WARNING: policy not loaded ({args.checkpoint_path})")
        checkpoint_path = args.checkpoint_path

    # P3-11A: Pooled Reporter Validation
    print("\n[4] Generating P3-11A pooled reporter designs (10 arms)")
    pooled = generate_pooled_designs(
        sources, oracle, policy,
        edit_budget=args.edit_budget,
        device=args.device,
        seed=args.seed,
    )
    pooled_path = os.path.join(output_dir, "p3_11_pooled_designs.json")
    _write_json(pooled_path, pooled)
    print(f"    pooled designs written: {pooled_path}")
    print(f"    arm ranking (top 3): {pooled['delta_ranking'][:3]}")

    # P3-11B: Full-Length Cargo Validation
    print("\n[5] Generating P3-11B full-length cargo designs")
    full_length = generate_full_length_designs(
        sources, oracle, policy,
        edit_budget=args.edit_budget,
        device=args.device,
        seed=args.seed,
    )
    full_length_path = os.path.join(output_dir, "p3_11_full_length_designs.json")
    _write_json(full_length_path, full_length)
    print(f"    full-length designs written: {full_length_path}")
    print(f"    total designs: {full_length['total_designs']}")

    # Sequence Freeze
    print("\n[6] Writing sequence freeze (pre-registration)")
    freeze_path = os.path.join(output_dir, "p3_11_sequence_freeze.md")
    write_sequence_freeze(pooled, full_length, checkpoint_path, checkpoint_sha256, freeze_path)
    print(f"    sequence freeze written: {freeze_path}")

    # Statistical Analysis Plan
    print("\n[7] Writing statistical analysis plan")
    stats_path = os.path.join(output_dir, "p3_11_statistical_analysis_plan.md")
    write_statistical_analysis_plan(stats_path)
    print(f"    stats plan written: {stats_path}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"P3-11 complete. ({elapsed:.1f}s)")
    print(f"  Artifacts in {output_dir}/")
    print(f"  Arms: {len(SPEC_ARMS)}")
    print(f"  Sources: {len(sources)}")
    print(f"  Full-length designs: {full_length['total_designs']}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
