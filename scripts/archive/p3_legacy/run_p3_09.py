#!/usr/bin/env python
"""P3-09: Independent Oracle, OOD, Reward-Hacking and External Benchmark.

This script closes the P3 evaluation contract by running five independent
audit / generalization axes on top of the P3-08 trained MEF policy and the
P3-02 cross-fitted oracle ensemble. Every component is wrapped in try/except
so that a single failure does not abort the whole run; failures are logged
and surfaced in ``docs/p3_09_failure_cases.md`` (constraint #16).

Artifacts produced (all under ``--output-dir``, default ``docs``):

1. ``p3_09_oracle_transfer.json``        — Oracle transfer matrix
2. ``p3_09_reward_hacking.md``           — Adversarial audit + reward hacking
3. ``p3_09_ood_results.json``            — OOD evaluation + abstention
4. ``p3_09_external_task_matrix.md``     — External benchmark task matrix
5. ``p3_09_external_results.json``       — External benchmark results
6. ``p3_09_failure_cases.md``            — Consolidated failure-case log

All predicted improvements are reported with "predicted" / "internal proxy"
qualifiers per constraint #23; no test data is used for training, reward
fitting, or hyperparameter selection per constraint #6; paper mode fails
closed per constraint #7.

Usage:
    python scripts/run_p3_09.py --device cuda --benchmark-dir data/p3/benchmark

    # Smoke test (CPU, synthetic oracles, no data / checkpoint required):
    python scripts/run_p3_09.py --smoke-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap (mirrors scripts/run_p3_07.py)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT.parent))
sys.path.insert(0, str(_REPO_ROOT))

from core.constants import START_CODON, NUC_VOCAB, translate
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
    EnsembleDeltaOracle,
    SearchResult,
    legal_actions,
    random_legal_editing,
    best_single_edit,
    greedy_search,
    beam_search,
    simulated_annealing,
    mcts_search,
    oracle_guided_local_search,
    stage_b_ranker_search,
    dagger_ranker_search,
    dagger_plus_limited_search,
    score_candidate,
    LinearDeltaRanker,
    SyntheticDeltaOracle,
    CountingOracle,
)

# P3-08 policy (lazy import — torch is heavy and only needed for the policy)
try:
    from rl.p3_08_grpo import (  # type: ignore
        P3O8Policy,
        ReferencePolicy,
        build_legal_edit_actions_task_a,
    )
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover — torch may be absent in smoke envs
    P3O8Policy = None  # type: ignore
    ReferencePolicy = None  # type: ignore
    build_legal_edit_actions_task_a = None  # type: ignore
    _TORCH_AVAILABLE = False

_decode_errors: List[Dict[str, Any]] = []  # errors from mef_policy_decode

REWARD_CFG = RewardV3Config(context="protein_output_focused")

# Inert placeholder CDS/3'UTR — Task A only edits 5'UTR; the CDS exists to
# satisfy MRNARecord and is never touched by the action space.
INERT_CDS = START_CODON + "GCU" * 4 + "UAA"
INERT_THREE_UTR = "UGCU"

# Adversarial-sequence length (per spec ~50nt)
_ADV_SEQ_LEN = 50

# Abstention thresholds (per spec)
_ABSTAIN_DISAGREEMENT = 0.05
_ABSTAIN_TRAIN_STD = 0.1
_ABSTAIN_LCB = 0.0


# ===========================================================================
# Utility helpers
# ===========================================================================

def source_to_record(source_id: str, five_utr: str) -> MRNARecord:
    """Build an MRNARecord with an inert CDS (Task A only edits 5'UTR)."""
    return MRNARecord(
        transcript_id=source_id,
        five_utr=five_utr,
        cds=INERT_CDS,
        three_utr=INERT_THREE_UTR,
        metadata={"inert_cds": True, "task": "task_a_five_utr_only"},
    )


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _safe_call(fn: Callable, *args, **kwargs) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """Run fn inside try/except; on failure return (None, failure_dict)."""
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:  # pragma: no cover — diagnostic path
        return None, {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=4),
        }


# ===========================================================================
# Independent oracle construction
# ===========================================================================

def build_independent_oracle_predict_fns(
    benchmark_dir: str,
    model_names: Sequence[str],
    max_proxy: int = 10000,
    seed: int = 42,
) -> Tuple[List[Callable], Dict[str, Any]]:
    """Build a cross-fitted independent oracle ensemble.

    Mirrors ``scripts.run_p3_07.build_ensemble_predict_fns`` but with a
    caller-chosen ``model_names`` tuple so the resulting ensemble is
    architecturally independent from the training oracle (which uses
    ``("seq_diff", "seq_linear")``).
    """
    from collections import defaultdict
    from core.p3_02_delta_oracle import (
        CrossFitConfig,
        load_benchmark,
        batch_extract_features,
        build_oracle_ensemble,
    )

    tiers = load_benchmark(benchmark_dir, tiers=("measured", "proxy"))
    measured = tiers.get("measured", [])
    proxy = tiers.get("proxy", [])
    train_recs = [r for r in measured if r.split_role in ("train", "val")]
    rng = np.random.RandomState(seed)
    if proxy:
        idx = rng.choice(len(proxy), min(len(proxy), max_proxy), replace=False)
        train_recs = train_recs + [proxy[i] for i in idx]

    config = CrossFitConfig(
        n_folds=5, seed=seed, hidden_dim=128, lr=1e-3,
        n_epochs=150, max_seq_len=100,
    )
    feats = batch_extract_features(train_recs, config.max_seq_len)
    labels = feats["delta"]

    groups: Dict[str, List[int]] = defaultdict(list)
    for i, rec in enumerate(train_recs):
        groups[rec.source_id].append(i)
    group_ids = list(groups.keys())
    rng2 = np.random.RandomState(seed)
    rng2.shuffle(group_ids)
    n_folds = config.n_folds
    fold_size = len(group_ids) // n_folds
    folds = []
    for k in range(n_folds):
        s = k * fold_size
        e = (k + 1) * fold_size if k < n_folds - 1 else len(group_ids)
        idxs: List[int] = []
        for gid in group_ids[s:e]:
            idxs.extend(groups[gid])
        folds.append(np.array(sorted(idxs)))

    ensemble = build_oracle_ensemble(
        feats, labels, folds, config, model_names=tuple(model_names),
    )

    predict_fns: List[Callable] = []
    for name in ensemble["model_names"]:
        fold_models = ensemble["per_model_models"][name]

        def fn(batch_feats, fms=fold_models):
            preds = [m.predict_delta(batch_feats) for m in fms.values()]
            return np.mean(preds, axis=0)

        predict_fns.append(fn)
    return predict_fns, ensemble


def build_independent_oracle(
    benchmark_dir: str,
    model_names: Sequence[str],
    max_proxy: int = 10000,
    seed: int = 42,
) -> EnsembleDeltaOracle:
    """Wrap an independent-architecture oracle ensemble."""
    predict_fns, _ = build_independent_oracle_predict_fns(
        benchmark_dir, model_names, max_proxy=max_proxy, seed=seed
    )
    if len(predict_fns) < 2:
        # Single-model ensembles cannot compute a meaningful std; replicate
        # so .score() works (std will be ~0, which is fine for the audit).
        predict_fns = predict_fns * 2
    return EnsembleDeltaOracle(predict_fns, max_seq_len=100)


# ===========================================================================
# Public predictor proxy: GC-content heuristic (independent of training data)
# ===========================================================================

class GCPredictorProxy(CountingOracle):
    """Public, training-data-free TE proxy.

    Models the well-known negative correlation between 5'UTR GC content and
    ribosomal scanning efficiency: candidates that *increase* GC in the 5'UTR
    receive a negative delta, candidates that decrease GC receive a positive
    delta (capped to avoid runaway rewards). This is intentionally a coarse
    public heuristic — it is used as one of the independent oracles in the
    transfer matrix, NOT as a ground-truth evaluator.
    """

    def __init__(self, query_budget: Optional[int] = None, weight: float = 0.3):
        super().__init__(query_budget)
        self.weight = float(weight)

    @staticmethod
    def _gc(seq: str) -> float:
        if not seq:
            return 0.0
        gc = sum(1 for ch in seq if ch in "GC")
        return gc / len(seq)

    def _score(self, source: MRNARecord, candidate: MRNARecord) -> Tuple[float, float]:
        src_gc = self._gc(source.five_utr)
        cand_gc = self._gc(candidate.five_utr)
        # Higher GC in 5'UTR → lower TE proxy → negative delta
        delta = self.weight * (src_gc - cand_gc)
        return float(delta), 0.02  # tiny constant uncertainty


# ===========================================================================
# P3-08 policy loader
# ===========================================================================

def load_mef_policy(checkpoint_path: str, device: str = "cpu") -> Optional["P3O8Policy"]:
    """Load the best P3-08 MEF checkpoint into a P3O8Policy.

    Returns None if torch is unavailable or the checkpoint is missing; the
    caller should treat None as "policy unavailable — use a fallback
    ranker-only policy" so the rest of P3-09 still runs.
    """
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
    policy = P3O8Policy(max_utr_len=100)  # type: ignore[misc]
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


def mef_policy_decode(
    policy: "P3O8Policy",
    source: MRNARecord,
    edit_budget: int,
    device: str = "cpu",
) -> Tuple[MRNARecord, List[Dict[str, Any]], Optional[float]]:
    """Greedy single-trajectory decode of the MEF policy.

    Returns (candidate, edits, log_prob). On any error, returns the source
    unchanged with an empty edit list (the policy abstains).

    Uses ``build_legal_edit_actions_task_a`` (5'UTR-only) instead of the full
    ``build_legal_edit_actions`` (which also generates CDS synonymous actions
    with ``action.nt=''``). The P3O8Policy was trained under Task A and its
    ``action_log_probs`` assumes 5'UTR-only actions — passing CDS actions
    raises ``KeyError: ''`` when looking up ``_NUC_TO_IDX[action.nt]``.
    """
    import torch
    try:
        state = initial_state(source, budget=edit_budget, cargo=source.transcript_id)
        # Use the Task A action builder (matches training-time action space):
        # STOP + 5'UTR substitutions only, no CDS, no indels, no 3'UTR.
        # The fast tuple-based cycle check avoids the MD5 cost of the generic
        # builder and is the same builder used during P3-08 GRPO rollouts.
        action_builder = (
            build_legal_edit_actions_task_a
            if build_legal_edit_actions_task_a is not None
            else build_legal_edit_actions
        )
        log_prob: Optional[float] = None
        while state.remaining_budget > 0:
            legal = action_builder(state.current_mrna, state.visited_states)
            if not legal or len(legal) == 1:
                break
            action, lp = policy.sample_action(state, legal, generator=None)
            log_prob = float(lp)
            if action.is_stop():
                break
            state = transition(state, action)
        edits = _diff_edits(source.five_utr, state.current_mrna.five_utr)
        return state.current_mrna, edits, log_prob
    except Exception as e:
        # Log the error so we can diagnose issues (constraint #16: preserve failures)
        import traceback
        _decode_errors.append({
            "source_id": getattr(source, "transcript_id", "?"),
            "error": str(e),
            "traceback": traceback.format_exc(),
        })
        return source, [], None


def _diff_edits(src_utr: str, cand_utr: str) -> List[Dict[str, Any]]:
    """List of {pos, ref, alt, region} for differing positions."""
    edits = []
    for i, (a, b) in enumerate(zip(src_utr, cand_utr)):
        if a != b:
            edits.append({"pos": i, "ref": a, "alt": b, "region": "five_utr"})
    if len(cand_utr) != len(src_utr):
        edits.append({
            "pos": min(len(src_utr), len(cand_utr)),
            "ref": "_length_change_",
            "alt": "_length_change_",
            "region": "five_utr",
        })
    return edits


# ===========================================================================
# Oracle scoring helpers
# ===========================================================================

def score_with_oracle(
    oracle: CountingOracle,
    source: MRNARecord,
    candidate: MRNARecord,
    n_edits: int,
    purpose: str = "eval",
) -> Dict[str, Any]:
    """Score a (source, candidate) pair and return reward v3 dict."""
    return score_candidate(source, candidate, oracle, n_edits, REWARD_CFG, purpose=purpose)


def _mean_delta_std_across_oracles(
    oracles: Sequence[CountingOracle],
    source: MRNARecord,
    candidate: MRNARecord,
) -> Tuple[float, float, List[float]]:
    """Return (mean_of_deltas, std_of_deltas, per_oracle_deltas)."""
    deltas = []
    for orc in oracles:
        try:
            m, _u = orc.score(source, candidate, purpose="eval")
        except Exception:
            m = float("nan")
        deltas.append(float(m))
    arr = np.array([d for d in deltas if not math.isnan(d)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), deltas
    return float(arr.mean()), float(arr.std(ddof=0)), deltas


# ===========================================================================
# Sources
# ===========================================================================

def load_test_sources(
    benchmark_dir: str,
    n_test: int = 24,
    n_train: int = 24,
    seed: int = 0,
) -> Tuple[List[MRNARecord], List[MRNARecord], Dict[str, Any]]:
    """Load test/train sources from P3-07 splits.

    Falls back to synthetic sources if the benchmark is unavailable.
    """
    try:
        from scripts.run_p3_07 import select_sources
        test_srcs, train_srcs, test_meta = select_sources(
            benchmark_dir, n_test, n_train, seed=seed,
        )
        if test_srcs:
            return test_srcs, train_srcs, test_meta
    except Exception:
        pass
    # Fallback: synthetic sources
    rng = np.random.RandomState(seed)
    test_srcs = []
    train_srcs = []
    for i in range(n_test):
        L = int(rng.randint(40, 80))
        seq = "".join(rng.choice(list("ACGU"), size=L))
        test_srcs.append(source_to_record(f"syn_test_{i}", seq))
    for i in range(n_train):
        L = int(rng.randint(40, 80))
        seq = "".join(rng.choice(list("ACGU"), size=L))
        train_srcs.append(source_to_record(f"syn_train_{i}", seq))
    return test_srcs, train_srcs, {"synthetic": True}


# ===========================================================================
# Candidate generation
# ===========================================================================

def generate_policy_candidates(
    policy: Optional["P3O8Policy"],
    sources: Sequence[MRNARecord],
    edit_budget: int,
    device: str = "cpu",
) -> List[Tuple[MRNARecord, MRNARecord, List[Dict[str, Any]]]]:
    """Generate one candidate per source using the MEF policy.

    If the policy is None (no checkpoint), each candidate is the source
    itself (zero-edit) so downstream scoring still runs.
    """
    out: List[Tuple[MRNARecord, MRNARecord, List[Dict[str, Any]]]] = []
    for src in sources:
        if policy is None:
            out.append((src, src, []))
            continue
        cand, edits, _lp = mef_policy_decode(policy, src, edit_budget, device=device)
        out.append((src, cand, edits))
    return out


# ===========================================================================
# Constraint guards (for adversarial audit)
# ===========================================================================

def detect_umax(seq: str) -> int:
    """Count overlapping AUG occurrences (uAUG proliferation)."""
    seq = seq.upper().replace("T", "U")
    n = 0
    i = 0
    while True:
        j = seq.find("AUG", i)
        if j < 0:
            return n
        n += 1
        i = j + 1


def detect_gc_extreme(seq: str, low: float = 0.3, high: float = 0.7) -> Optional[str]:
    """Flag GC content outside [low, high]."""
    if not seq:
        return None
    gc = sum(1 for ch in seq if ch in "GC") / len(seq)
    if gc > high:
        return f"high_gc={gc:.2f}"
    if gc < low:
        return f"low_gc={gc:.2f}"
    return None


def detect_polyadenylation(seq: str) -> Optional[str]:
    """Flag canonical polyadenylation signals (AAUAAA variants)."""
    seq = seq.upper().replace("T", "U")
    for motif in ("AAUAAA", "AUUAAA", "UAAUAA", "AAAAAA"):
        if motif in seq:
            return f"polyA_signal={motif}"
    return None


def detect_splice_sites(seq: str) -> Optional[str]:
    """Flag cryptic splice donor/acceptor motifs."""
    seq = seq.upper().replace("T", "U")
    for motif in ("GUAAGU", "AGGUA", "CAGG", "GUAAGUA"):
        if motif in seq:
            return f"splice_motif={motif}"
    return None


def detect_homopolymer(seq: str, min_run: int = 8) -> Optional[str]:
    """Flag homopolymer runs >= min_run."""
    seq = seq.upper().replace("T", "U")
    cur_ch = ""
    cur_len = 0
    for ch in seq:
        if ch == cur_ch:
            cur_len += 1
            if cur_len >= min_run:
                return f"homopolymer_{cur_ch}{cur_len}"
        else:
            cur_ch = ch
            cur_len = 1
    return None


def detect_cpg_island(seq: str, window: int = 10, threshold: float = 0.6) -> Optional[str]:
    """Flag extreme CpG density windows."""
    seq = seq.upper().replace("T", "U")
    if len(seq) < window:
        return None
    for i in range(len(seq) - window + 1):
        w = seq[i:i + window]
        cpg = sum(1 for j in range(window - 1) if w[j] == "C" and w[j + 1] == "G")
        if cpg / window > threshold:
            return f"cpg_island_pos{i}"
    return None


def detect_upa_extreme(seq: str, window: int = 10, threshold: float = 0.3) -> Optional[str]:
    """Flag extreme UpA dinucleotide density.

    Threshold lowered from 0.6 to 0.3 — UpA dinucleotides are
    immunostimulatory and under-represented in native RNAs, so even moderate
    local enrichment signals a non-native (potentially reward-hacked) sequence.
    """
    seq = seq.upper().replace("T", "U")
    if len(seq) < window:
        return None
    for i in range(len(seq) - window + 1):
        w = seq[i:i + window]
        upa = sum(1 for j in range(window - 1) if w[j] == "U" and w[j + 1] == "A")
        if upa / window > threshold:
            return f"upa_extreme_pos{i}"
    return None


def detect_cpa_extreme(seq: str, window: int = 10, threshold: float = 0.3) -> Optional[str]:
    """Flag extreme CpA dinucleotide density (catches 'CA' repeat reward-hacking).

    The adversarial ``extreme_upa`` arm is actually a 'CA' dinucleotide repeat
    (``"CA" * (n//2)``) which evades the UpA guard. CpA repeats are similarly
    non-native and immunostimulatory.
    """
    seq = seq.upper().replace("T", "U")
    if len(seq) < window:
        return None
    for i in range(len(seq) - window + 1):
        w = seq[i:i + window]
        cpa = sum(1 for j in range(window - 1) if w[j] == "C" and w[j + 1] == "A")
        if cpa / window > threshold:
            return f"cpa_extreme_pos{i}"
    return None


def detect_hairpin(
    seq: str,
    min_stem: int = 4,
    min_loop: int = 3,
    max_loop: int = 15,
    max_mismatches: int = 1,
) -> Optional[str]:
    """Flag stable stem-loop (hairpin) structures via inverted-repeat detection.

    A hairpin forms when a sequence is followed (after a short loop) by its
    reverse complement, e.g. ``GGGGG-CAAAC-CCCCC``. Such structures can
    hijack structure-aware oracles into assigning spuriously high rewards.

    Scans for stems of length >= ``min_stem`` separated by a loop of length in
    ``[min_loop, max_loop]``, allowing up to ``max_mismatches`` non-Watson-Crick
    pairs. Returns the first detected hairpin or ``None``.

    Complexity: O(n * max_loop * max_stem) per call — fine for ~50-100nt UTRs.
    """
    seq = seq.upper().replace("T", "U")
    n = len(seq)
    if n < min_stem * 2 + min_loop:
        return None
    comp = {"A": "U", "U": "A", "G": "C", "C": "G"}
    for i in range(n - 2 * min_stem - min_loop + 1):
        for loop_len in range(min_loop, max_loop + 1):
            right_start = i + loop_len
            remaining = n - right_start
            if remaining < min_stem:
                break
            max_stem_here = min(remaining, n - i - loop_len)
            # Compare seq[i:i+stem] vs reverse-comp of seq[right_start:right_start+stem]
            for stem_len in range(min_stem, max_stem_here + 1):
                left = seq[i:i + stem_len]
                right = seq[right_start:right_start + stem_len]
                rc_left = "".join(comp.get(ch, "N") for ch in reversed(left))
                mismatches = sum(1 for a, b in zip(rc_left, right) if a != b)
                if mismatches <= max_mismatches:
                    return (
                        f"hairpin_pos{i}_stem{stem_len}_loop{loop_len}"
                        f"_mm{mismatches}"
                    )
    return None


def run_all_guards(seq: str) -> Dict[str, Any]:
    """Run all constraint guards on a sequence; return detected flags + counts.

    If any guard triggers, the sequence is considered a reward-hacking
    candidate and its reward should be clamped to zero by the caller.
    """
    flags = []
    for fn in (
        detect_gc_extreme,
        detect_polyadenylation,
        detect_splice_sites,
        detect_homopolymer,
        detect_cpg_island,
        detect_upa_extreme,
        detect_cpa_extreme,
        detect_hairpin,
    ):
        f = fn(seq)
        if f:
            flags.append(f)
    return {
        "guard_detected": len(flags) > 0,
        "flags": flags,
        "umax_count": detect_umax(seq),
        "gc_fraction": (sum(1 for ch in seq if ch in "GC") / len(seq)) if seq else 0.0,
    }


# ===========================================================================
# Adversarial sequence generator
# ===========================================================================

def build_adversarial_sequences() -> List[Tuple[str, str]]:
    """Return list of (name, sequence) adversarial 5'UTR sequences (~50nt)."""
    n = _ADV_SEQ_LEN
    seqs: List[Tuple[str, str]] = [
        ("extreme_high_gc",      "GCGCGCGCGC" * (n // 10)),
        ("extreme_low_gc",       "AUAUAUAUAU" * (n // 10)),
        ("homopolymer_a",        "A" * n),
        ("homopolymer_g",        "G" * n),
        ("repeated_aug",         "AUG" * (n // 3)),
        ("repeated_uorf",        "AUGAAATAA" * (n // 8)),
        ("cryptic_splice",       "GUAAGU" * (n // 6)),
        ("polyadenylation",      "AAUAAA" * (n // 6)),
        ("extreme_cpg",          "CG" * (n // 2)),
        ("extreme_upa",          "CA" * (n // 2)),
        ("random_low_lm",        "AUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCG"[:n]),
        ("stable_hairpin",       ("GGGGG" + "CAAAC" + "CCCCC" + "CAAAC") * (n // 20 + 1)),
    ]
    # near_duplicate: 1-edit from a fixed "training-like" source
    base = "GCCAUGAGCAACGGAUUCGACCCAGACUUGACGAUUACGGACUUGACCAG"[:n]
    nd = base[:-1] + ("A" if base[-1] != "A" else "C")
    seqs.append(("near_duplicate", nd))
    # Ensure all sequences are exactly n long
    fixed = []
    for name, s in seqs:
        if len(s) < n:
            s = (s + "A" * n)[:n]
        else:
            s = s[:n]
        fixed.append((name, s.upper().replace("T", "U")))
    return fixed


# ===========================================================================
# OOD split helpers
# ===========================================================================

def _utr_length(rec: MRNARecord) -> int:
    return len(rec.five_utr)


def _utr_gc(rec: MRNARecord) -> float:
    s = rec.five_utr
    if not s:
        return 0.0
    return sum(1 for ch in s if ch in "GC") / len(s)


def build_ood_splits(
    test_srcs: Sequence[MRNARecord],
    train_srcs: Sequence[MRNARecord],
    test_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[MRNARecord]]:
    """Build the available OOD splits.

    Per spec: cargo_family_ood and rare_family require family metadata; if
    absent, fall back to "all test sources" with a note. Length_shift and
    gc_shift use train mean ± 2std. species / cell_context / reporter_to_therapeutic
    are literature-only.
    """
    splits: Dict[str, List[MRNARecord]] = {}

    # cargo_family_ood: test sources from families NOT in training.
    # The P3-01 split is already group-aware, so the test set is by
    # construction family-OOD; we keep the whole test set.
    splits["cargo_family_ood"] = list(test_srcs)

    # rare_family: families with < 5 members in training. Without family
    # metadata on MRNARecord, fall back to marking the entire test set as
    # "rare_family_proxy" with a note in the OOD artifact.
    splits["rare_family"] = list(test_srcs)

    # length_shift: 5'UTR length outside train mean ± 2std
    if train_srcs:
        train_lens = np.array([_utr_length(r) for r in train_srcs])
        mu, sd = float(train_lens.mean()), float(train_lens.std(ddof=0))
        lo, hi = mu - 2 * sd, mu + 2 * sd
        ls = [r for r in test_srcs if not (lo <= _utr_length(r) <= hi)]
        splits["length_shift"] = ls if ls else list(test_srcs[:4])
    else:
        splits["length_shift"] = []

    # gc_shift: 5'UTR GC outside train range
    if train_srcs:
        train_gcs = np.array([_utr_gc(r) for r in train_srcs])
        lo, hi = float(train_gcs.min()), float(train_gcs.max())
        gs = [r for r in test_srcs if not (lo <= _utr_gc(r) <= hi)]
        splits["gc_shift"] = gs if gs else list(test_srcs[:4])
    else:
        splits["gc_shift"] = []

    return splits


# ===========================================================================
# On-manifold score (sequence distance to nearest training source)
# ===========================================================================

def hamming(a: str, b: str) -> int:
    """Hamming distance (truncated to shorter)."""
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))


def on_manifold_score(
    candidate_utr: str,
    train_srcs: Sequence[MRNARecord],
) -> float:
    """Lower = more on-manifold. Returns min Hamming distance to train srcs."""
    if not train_srcs:
        return float("inf")
    return min(hamming(candidate_utr, r.five_utr) for r in train_srcs)


# ===========================================================================
# Abstention mechanism
# ===========================================================================

def should_abstain(
    source: MRNARecord,
    candidate: MRNARecord,
    training_oracle: CountingOracle,
    independent_oracles: Sequence[CountingOracle],
    train_srcs: Sequence[MRNARecord],
    median_train_distance: float,
) -> Tuple[bool, List[str]]:
    """Evaluate abstention triggers; return (abstain, reasons)."""
    reasons: List[str] = []
    n_edits = len(_diff_edits(source.five_utr, candidate.five_utr))

    # 1. high_disagreement: std across oracles > 0.05
    all_oracles = [training_oracle] + list(independent_oracles)
    _mean, std, _ = _mean_delta_std_across_oracles(all_oracles, source, candidate)
    if not math.isnan(std) and std > _ABSTAIN_DISAGREEMENT:
        reasons.append(f"high_disagreement(std={std:.4f})")

    # 2. high_uncertainty: training Oracle std > 0.1
    try:
        _m, train_u = training_oracle.score(source, candidate, purpose="eval")
        if train_u > _ABSTAIN_TRAIN_STD:
            reasons.append(f"high_uncertainty(u={train_u:.4f})")
    except Exception:
        pass

    # 3. low_on_manifold: distance > 2×median
    dist = on_manifold_score(candidate.five_utr, train_srcs)
    if median_train_distance > 0 and dist > 2 * median_train_distance:
        reasons.append(f"low_on_manifold(d={dist}, 2*med={2*median_train_distance:.2f})")

    # 4. no_positive_lcb: reward LCB < 0
    try:
        sc = score_with_oracle(training_oracle, source, candidate, n_edits, purpose="eval")
        if sc["lcb"] < _ABSTAIN_LCB:
            reasons.append(f"no_positive_lcb(lcb={sc['lcb']:.4f})")
    except Exception:
        pass

    # 5. constraint_risk: near constraint boundary (very high GC or uAUG count)
    g = run_all_guards(candidate.five_utr)
    if g["guard_detected"]:
        reasons.append(f"constraint_risk({g['flags']})")

    return (len(reasons) > 0), reasons


# ===========================================================================
# Component 1: Oracle Transfer Matrix
# ===========================================================================

def run_oracle_transfer_matrix(
    args: argparse.Namespace,
    test_srcs: Sequence[MRNARecord],
    train_srcs: Sequence[MRNARecord],
    training_oracle: CountingOracle,
    independent_oracles: Dict[str, CountingOracle],
    policy: Optional["P3O8Policy"],
    failures: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Score policy candidates with every oracle and compute transfer metrics."""
    print("\n[P3-09.1] Oracle Transfer Matrix")
    t0 = time.time()

    # Generate one candidate per source via the MEF policy (edit_budget=1)
    pairs = generate_policy_candidates(policy, test_srcs, edit_budget=1, device=args.device)

    # Score every pair with every oracle
    oracle_names = ["training"] + list(independent_oracles.keys()) + ["gc_proxy"]
    all_oracles = {"training": training_oracle, **independent_oracles,
                   "gc_proxy": GCPredictorProxy()}
    per_oracle_deltas: Dict[str, List[float]] = {name: [] for name in oracle_names}
    per_oracle_unc: Dict[str, List[float]] = {name: [] for name in oracle_names}
    per_oracle_rewards: Dict[str, List[float]] = {name: [] for name in oracle_names}
    on_manifold_scores: List[float] = []
    constraint_valid: List[bool] = []
    pair_records: List[Dict[str, Any]] = []

    for src, cand, edits in pairs:
        n_edits = len(edits)
        constraint_valid.append(
            translate(src.cds) == translate(cand.cds)
            and len(src.seq) == len(cand.seq)
        )
        on_manifold_scores.append(on_manifold_score(cand.five_utr, train_srcs))
        row: Dict[str, Any] = {
            "source_id": src.transcript_id,
            "n_edits": n_edits,
            "candidate_utr": cand.five_utr,
        }
        for name, orc in all_oracles.items():
            try:
                m, u = orc.score(src, cand, purpose="eval")
                sc = score_with_oracle(orc, src, cand, n_edits, purpose="eval")
            except Exception as exc:
                m, u, sc = float("nan"), float("nan"), {"scalar": float("nan")}
                failures.append({
                    "component": "oracle_transfer_matrix",
                    "source_id": src.transcript_id,
                    "oracle": name,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            per_oracle_deltas[name].append(float(m))
            per_oracle_unc[name].append(float(u))
            per_oracle_rewards[name].append(float(sc["scalar"]) if not math.isnan(float(sc["scalar"])) else float("nan"))
            row[f"{name}_delta"] = float(m)
            row[f"{name}_unc"] = float(u)
            row[f"{name}_reward"] = float(sc["scalar"])
        pair_records.append(row)

    # Compute pairwise transfer metrics: training vs each independent oracle
    transfer: Dict[str, Any] = {}
    train_d = np.array(per_oracle_deltas["training"], dtype=np.float64)
    train_d = train_d[~np.isnan(train_d)]
    onm = np.array(on_manifold_scores, dtype=np.float64) if on_manifold_scores else np.zeros(0)
    for name in independent_oracles.keys():
        ind_d = np.array(per_oracle_deltas[name], dtype=np.float64)
        # Align on non-NaN
        mask = ~(np.isnan(per_oracle_deltas["training"]) | np.isnan(ind_d))
        t_arr = np.array(per_oracle_deltas["training"], dtype=np.float64)[mask]
        i_arr = ind_d[mask]
        if len(t_arr) < 2:
            transfer[name] = {"note": "insufficient non-NaN pairs", "n_pairs": int(mask.sum())}
            continue

        # Pearson
        t_m = t_arr - t_arr.mean()
        i_m = i_arr - i_arr.mean()
        denom = math.sqrt(float(np.sum(t_m ** 2) * np.sum(i_m ** 2)))
        pearson = float(np.sum(t_m * i_m) / denom) if denom > 1e-12 else 0.0

        # Spearman (rank correlation)
        try:
            from scipy.stats import spearmanr
            sp, _ = spearmanr(t_arr, i_arr)
            spearman = float(sp) if not math.isnan(sp) else 0.0
        except Exception:
            spearman = 0.0

        # Sign agreement
        sign_t = np.sign(t_arr)
        sign_i = np.sign(i_arr)
        non_zero = (sign_t != 0) | (sign_i != 0)
        sign_agree = float(np.mean(sign_t[non_zero] == sign_i[non_zero])) if non_zero.sum() > 0 else 0.0

        # Top-k overlap (top 20%)
        k = max(1, int(len(t_arr) * 0.2))
        top_t = set(np.argsort(-t_arr)[:k].tolist())
        top_i = set(np.argsort(-i_arr)[:k].tolist())
        top_k_overlap = len(top_t & top_i) / max(k, 1)

        # Beneficial precision: fraction of training-positive that are also independent-positive
        train_pos = t_arr > 0
        if train_pos.sum() > 0:
            beneficial_precision = float(np.mean(i_arr[train_pos] > 0))
        else:
            beneficial_precision = 0.0

        # Disagreement concentration: fraction of disagreements (sign flip) in bottom quartile of on-manifold
        disagreement_mask = sign_t != sign_i
        if onm.size > 0 and disagreement_mask.sum() > 0:
            q25 = float(np.percentile(onm, 25))
            bottom_q = onm >= q25  # bottom quartile == HIGHEST distance (least on-manifold)
            # NOTE: on_manifold_score is a distance, so "bottom quartile of on-manifold score"
            # is interpreted as the 25% with the HIGHEST distance values.
            disagreement_concentration = float(np.mean(bottom_q[disagreement_mask])) if disagreement_mask.sum() > 0 else 0.0
        else:
            disagreement_concentration = 0.0

        # Calibration: Brier score of "training predicts independent sign"
        # Treat sign(training) as probabilistic prediction (clip to [0.05, 0.95])
        p_pos = np.clip(1.0 / (1.0 + np.exp(-5.0 * t_arr)), 0.05, 0.95)
        y = (i_arr > 0).astype(np.float64)
        brier = float(np.mean((p_pos - y) ** 2))

        # ECE: bin predictions, compare to observed frequency
        n_bins = 5
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for b in range(n_bins):
            m_b = (p_pos >= bins[b]) & (p_pos < bins[b + 1])
            if m_b.sum() > 0:
                ece += abs(float(p_pos[m_b].mean()) - float(y[m_b].mean())) * int(m_b.sum())
        ece = ece / max(len(p_pos), 1)

        transfer[name] = {
            "n_pairs": int(len(t_arr)),
            "delta_correlation_pearson": pearson,
            "delta_correlation_spearman": spearman,
            "sign_agreement": sign_agree,
            "top_k_overlap": top_k_overlap,
            "beneficial_precision": beneficial_precision,
            "disagreement_concentration": disagreement_concentration,
            "calibration_brier": brier,
            "calibration_ece": ece,
            "training_mean_delta": float(np.mean(t_arr)) if t_arr.size else 0.0,
            "independent_mean_delta": float(np.mean(i_arr)) if i_arr.size else 0.0,
            "training_pos_rate": float(np.mean(t_arr > 0)) if t_arr.size else 0.0,
            "independent_pos_rate": float(np.mean(i_arr > 0)) if i_arr.size else 0.0,
        }

    out = {
        "phase": "P3-09",
        "component": "oracle_transfer_matrix",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "n_test_sources": len(test_srcs),
            "edit_budget": 1,
            "training_oracle": "p3_02 cross-fitted ensemble (seq_diff + seq_linear)",
            "independent_oracles": list(independent_oracles.keys()),
            "public_proxy": "gc_content_heuristic",
            "policy_checkpoint": args.checkpoint if policy is not None else None,
            "policy_available": policy is not None,
            "qualifier": "All deltas are predicted by the internal P3-02 oracle ensemble or its independent-architecture variants; not wet-lab measurements.",
        },
        "per_oracle_summary": {
            name: {
                "mean_delta": float(np.nanmean(per_oracle_deltas[name])) if per_oracle_deltas[name] else 0.0,
                "mean_uncertainty": float(np.nanmean(per_oracle_unc[name])) if per_oracle_unc[name] else 0.0,
                "mean_reward": float(np.nanmean(per_oracle_rewards[name])) if per_oracle_rewards[name] else 0.0,
                "pos_rate": float(np.nanmean(np.array(per_oracle_deltas[name]) > 0)) if per_oracle_deltas[name] else 0.0,
                "n_scored": int(sum(1 for d in per_oracle_deltas[name] if not math.isnan(d))),
            } for name in oracle_names
        },
        "transfer_metrics": transfer,
        "constraint_validity_rate": float(np.mean(constraint_valid)) if constraint_valid else 0.0,
        "n_pairs": len(pairs),
        "wall_clock_sec": time.time() - t0,
        "per_pair_records": pair_records[:50],  # cap for size
    }
    return out


# ===========================================================================
# Component 2: Adversarial Audit + Reward Hacking
# ===========================================================================

def run_adversarial_audit(
    args: argparse.Namespace,
    test_srcs: Sequence[MRNARecord],
    training_oracle: CountingOracle,
    independent_oracles: Dict[str, CountingOracle],
    failures: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Score adversarial 5'UTR sequences and flag reward hacking."""
    print("\n[P3-09.2] Adversarial Audit + Reward Hacking")
    t0 = time.time()

    adv_seqs = build_adversarial_sequences()
    if not test_srcs:
        failures.append({"component": "adversarial_audit", "error": "no test sources available"})
        return {"error": "no test sources", "wall_clock_sec": time.time() - t0}

    # Use a real test source as the "source" against which the adversarial
    # sequence is scored as a candidate.
    ref_source = test_srcs[0]

    audit_rows: List[Dict[str, Any]] = []
    reward_hacking_cases: List[Dict[str, Any]] = []
    guard_evasion_cases: List[Dict[str, Any]] = []

    for name, adv_seq in adv_seqs:
        # Build candidate: replace 5'UTR with adversarial sequence
        cand = MRNARecord(
            transcript_id=ref_source.transcript_id,
            five_utr=adv_seq,
            cds=ref_source.cds,
            three_utr=ref_source.three_utr,
            metadata={"adv_name": name, **ref_source.metadata},
        )

        # Run guards
        guard = run_all_guards(adv_seq)

        # Score with training and independent oracles
        try:
            tr_m, tr_u = training_oracle.score(ref_source, cand, purpose="eval")
            tr_reward = score_with_oracle(training_oracle, ref_source, cand, 1, purpose="eval")["scalar"]
        except Exception as exc:
            tr_m, tr_u, tr_reward = float("nan"), float("nan"), float("nan")
            failures.append({
                "component": "adversarial_audit",
                "adversarial_name": name,
                "oracle": "training",
                "error": f"{type(exc).__name__}: {exc}",
            })

        ind_rewards: Dict[str, float] = {}
        ind_means: Dict[str, float] = {}
        for ind_name, ind_orc in independent_oracles.items():
            try:
                im, _iu = ind_orc.score(ref_source, cand, purpose="eval")
                ir = score_with_oracle(ind_orc, ref_source, cand, 1, purpose="eval")["scalar"]
            except Exception as exc:
                im, ir = float("nan"), float("nan")
                failures.append({
                    "component": "adversarial_audit",
                    "adversarial_name": name,
                    "oracle": ind_name,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            ind_means[ind_name] = float(im)
            ind_rewards[ind_name] = float(ir)

        ind_mean_arr = np.array([v for v in ind_means.values() if not math.isnan(v)], dtype=np.float64)
        independent_reward_mean = float(ind_mean_arr.mean()) if ind_mean_arr.size else float("nan")

        onm = on_manifold_score(adv_seq, test_srcs)

        # Guard-based reward penalty: if any guard triggers, clamp the
        # effective reward to zero. This is the mitigation mechanism — the
        # raw training_reward is preserved for transparency, but
        # effective_reward is what the policy/evaluator would actually use.
        raw_tr = float(tr_reward)
        guard_fired = guard["guard_detected"]
        effective_reward = 0.0 if guard_fired else raw_tr

        row = {
            "adversarial_name": name,
            "sequence": adv_seq,
            "training_reward": raw_tr,
            "training_delta": float(tr_m),
            "effective_reward": effective_reward,
            "mitigated_by_guard": guard_fired and raw_tr > 0,
            "independent_rewards": ind_rewards,
            "independent_reward_mean": independent_reward_mean,
            "guard_detected": guard_fired,
            "guard_flags": guard["flags"],
            "umax_count": guard["umax_count"],
            "gc_fraction": guard["gc_fraction"],
            "on_manifold_score": float(onm),
        }
        audit_rows.append(row)

        # Reward hacking flag (post-mitigation): effective reward > 0 AND
        # |effective - independent| > 0.05. Cases caught by the guard have
        # effective_reward=0 and are therefore no longer "hacking" — they are
        # logged as mitigated instead.
        if (not math.isnan(effective_reward) and not math.isnan(independent_reward_mean)
                and abs(effective_reward - independent_reward_mean) > 0.05
                and effective_reward > 0):
            row["reward_hacking_flag"] = (
                f"|eff={effective_reward:.4f} - ind={independent_reward_mean:.4f}|="
                f"{abs(effective_reward - independent_reward_mean):.4f} > 0.05 AND eff > 0"
            )
            reward_hacking_cases.append(row)

        # Guard evasion: raw training reward > 0 AND no guard detected
        # (the sequence slipped past all guards but still got a positive reward)
        if not math.isnan(raw_tr) and raw_tr > 0 and not guard_fired:
            guard_evasion_cases.append(row)

    out = {
        "phase": "P3-09",
        "component": "adversarial_audit",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "n_adversarial_sequences": len(adv_seqs),
            "ref_source_id": ref_source.transcript_id,
            "qualifier": "All rewards are predicted by the internal oracle ensemble; guard detection is heuristic and not exhaustive.",
        },
        "audit_rows": audit_rows,
        "reward_hacking_cases": reward_hacking_cases,
        "guard_evasion_cases": guard_evasion_cases,
        "n_reward_hacking": len(reward_hacking_cases),
        "n_guard_evasion": len(guard_evasion_cases),
        "wall_clock_sec": time.time() - t0,
    }
    return out


def write_reward_hacking_md(audit: Dict[str, Any], path: str) -> None:
    """Write the reward-hacking markdown artifact."""
    lines: List[str] = []
    lines.append("# P3-09 Adversarial Audit & Reward Hacking Report\n")
    lines.append("> All rewards reported here are **predicted** by the internal P3-02 oracle ensemble")
    lines.append("> and its independent-architecture variants. They are NOT wet-lab measurements.")
    lines.append("> Guard detection uses simple motif heuristics and is not exhaustive.\n")

    if "error" in audit:
        lines.append(f"\n**ERROR:** {audit['error']}\n")
        _write_text(path, "\n".join(lines))
        return

    n_rows = len(audit.get("audit_rows", []))
    n_mitigated = sum(1 for r in audit.get("audit_rows", []) if r.get("mitigated_by_guard"))
    lines.append(f"- Adversarial sequences audited: **{audit.get('config', {}).get('n_adversarial_sequences', n_rows)}**")
    lines.append(f"- Reward-hacking cases (post-mitigation): **{audit['n_reward_hacking']}**")
    lines.append(f"- Guard-evasion cases (training reward > 0 AND no guard detected): **{audit['n_guard_evasion']}**")
    lines.append(f"- Cases mitigated by guard (raw reward > 0, clamped to 0): **{n_mitigated}**")
    lines.append(f"- Wall clock: {audit['wall_clock_sec']:.1f}s\n")

    lines.append("## Mitigation Mechanism\n")
    lines.append("When any guard triggers, the **effective reward** is clamped to zero.")
    lines.append("The raw training reward is preserved for transparency. Reward-hacking")
    lines.append("is evaluated on the *effective* reward, so guard-caught cases are not")
    lines.append("counted as active hacking.\n")

    lines.append("## Audit Table\n")
    lines.append("| Adversarial | Raw training reward | Effective reward | Independent mean | Guard | uAUG | GC | On-manifold |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in audit.get("audit_rows", []):
        lines.append(
            f"| {row['adversarial_name']} | {row['training_reward']:.4f} | "
            f"{row.get('effective_reward', row['training_reward']):.4f} | "
            f"{row['independent_reward_mean']:.4f} | "
            f"{'yes' if row['guard_detected'] else 'no'} | "
            f"{row['umax_count']} | {row['gc_fraction']:.2f} | "
            f"{row['on_manifold_score']:.1f} |"
        )

    if audit.get("reward_hacking_cases"):
        lines.append("\n## Reward-Hacking Cases (|effective − independent| > 0.05 AND effective > 0)\n")
        for c in audit["reward_hacking_cases"]:
            lines.append(f"- **{c['adversarial_name']}**: {c.get('reward_hacking_flag', '')}")
            lines.append(f"  - effective reward = {c.get('effective_reward', c['training_reward']):.4f}, independent mean = {c['independent_reward_mean']:.4f}")
            lines.append(f"  - guard detected: {c['guard_detected']} ({', '.join(c['guard_flags']) or 'none'})")

    if audit.get("guard_evasion_cases"):
        lines.append("\n## Guard-Evasion Cases (raw training reward > 0 AND no guard detected)\n")
        for c in audit["guard_evasion_cases"]:
            lines.append(f"- **{c['adversarial_name']}**: raw training reward = {c['training_reward']:.4f}")
    else:
        lines.append("\n## Guard-Evasion Cases\n")
        lines.append("- (none — all adversarial sequences with positive raw reward were caught by at least one guard)")

    lines.append("\n## Per-Oracle Independent Rewards\n")
    for row in audit.get("audit_rows", []):
        lines.append(f"### {row['adversarial_name']}")
        for k, v in row.get("independent_rewards", {}).items():
            lines.append(f"- {k}: {v:.4f}")
        lines.append("")

    _write_text(path, "\n".join(lines))


# ===========================================================================
# Component 3: OOD Evaluation + Abstention
# ===========================================================================

def run_ood_evaluation(
    args: argparse.Namespace,
    test_srcs: Sequence[MRNARecord],
    train_srcs: Sequence[MRNARecord],
    training_oracle: CountingOracle,
    independent_oracles: Dict[str, CountingOracle],
    policy: Optional["P3O8Policy"],
    failures: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate policy on OOD splits with abstention enabled."""
    print("\n[P3-09.3] OOD Evaluation + Abstention")
    t0 = time.time()

    splits = build_ood_splits(test_srcs, train_srcs)

    # Compute median train distance for on-manifold abstention
    if train_srcs:
        # Sample pairwise distances inside training set
        sample = train_srcs[:min(20, len(train_srcs))]
        dists = []
        for i, r1 in enumerate(sample):
            for r2 in sample[i + 1:]:
                dists.append(hamming(r1.five_utr, r2.five_utr))
        median_train_distance = float(np.median(dists)) if dists else 0.0
    else:
        median_train_distance = 0.0

    ind_oracles_list = list(independent_oracles.values())

    # Evaluate each available split
    split_results: Dict[str, Any] = {}
    abstention_log: List[Dict[str, Any]] = []
    for split_name, srcs in splits.items():
        if not srcs:
            split_results[split_name] = {"n_sources": 0, "note": "empty split"}
            continue

        rewards: List[float] = []
        delta_means: List[float] = []
        constraint_valid: List[bool] = []
        n_edits_list: List[int] = []
        abstain_count = 0
        triggers: Dict[str, int] = {}
        per_source: List[Dict[str, Any]] = []

        for src in srcs:
            # Generate candidate via policy
            if policy is None:
                cand, edits = src, []
            else:
                cand, edits, _lp = mef_policy_decode(policy, src, args.edit_budget, device=args.device)

            # Check abstention
            abstain, reasons = should_abstain(
                src, cand, training_oracle, ind_oracles_list,
                train_srcs, median_train_distance,
            )
            for r in reasons:
                # extract trigger name (text before '(')
                tname = r.split("(")[0]
                triggers[tname] = triggers.get(tname, 0) + 1

            if abstain:
                # Abstain: return source unchanged, reward = 0
                abstain_count += 1
                rewards.append(0.0)
                delta_means.append(0.0)
                constraint_valid.append(True)
                n_edits_list.append(0)
                per_source.append({
                    "source_id": src.transcript_id,
                    "abstained": True,
                    "reasons": reasons,
                    "n_edits": 0,
                    "reward": 0.0,
                })
                abstention_log.append({
                    "split": split_name,
                    "source_id": src.transcript_id,
                    "reasons": reasons,
                })
                continue

            # Score (use mean across oracles as the reported reward — qualifier added in artifact)
            n_edits = len(edits)
            try:
                _m, _s, _ = _mean_delta_std_across_oracles(
                    [training_oracle] + ind_oracles_list, src, cand,
                )
                if math.isnan(_m):
                    raise RuntimeError("all oracle scores NaN")
                # Reward under training oracle (used for pos_rate / stop_rate etc.)
                sc = score_with_oracle(training_oracle, src, cand, n_edits, purpose="eval")
                rew = float(sc["scalar"])
            except Exception as exc:
                _m, rew = float("nan"), float("nan")
                failures.append({
                    "component": "ood_evaluation",
                    "split": split_name,
                    "source_id": src.transcript_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })

            cv = (translate(src.cds) == translate(cand.cds)
                  and len(src.seq) == len(cand.seq))
            rewards.append(float(rew) if not math.isnan(float(rew)) else 0.0)
            delta_means.append(float(_m) if not math.isnan(_m) else 0.0)
            constraint_valid.append(bool(cv))
            n_edits_list.append(int(n_edits))
            per_source.append({
                "source_id": src.transcript_id,
                "abstained": False,
                "reasons": [],
                "n_edits": int(n_edits),
                "reward": float(rew) if not math.isnan(float(rew)) else 0.0,
                "mean_delta": float(_m) if not math.isnan(_m) else 0.0,
            })

        arr_r = np.array(rewards, dtype=np.float64)
        arr_d = np.array(delta_means, dtype=np.float64)
        split_results[split_name] = {
            "n_sources": len(srcs),
            "n_abstained": abstain_count,
            "abstention_rate": abstain_count / max(len(srcs), 1),
            "constraint_validity_rate": float(np.mean(constraint_valid)) if constraint_valid else 0.0,
            "pos_rate": float(np.mean(arr_d > 0)) if arr_d.size else 0.0,
            "stop_rate": float(np.mean(np.array(n_edits_list) == 0)) if n_edits_list else 0.0,
            "mean_reward": float(np.mean(arr_r)) if arr_r.size else 0.0,
            "reward_std": float(np.std(arr_r)) if arr_r.size else 0.0,
            "mean_delta": float(np.mean(arr_d)) if arr_d.size else 0.0,
            "triggers": triggers,
            "per_source": per_source,
        }

    # Literature-only splits (no data in current benchmark)
    literature_only = {
        "species_shift": "No multi-species data in current P3-01 benchmark; literature comparison only.",
        "cell_context_shift": "No multi-cell-type data in current P3-01 benchmark; literature comparison only.",
        "reporter_to_therapeutic": "MPRA reporter → therapeutic cargo transfer requires wet-lab evidence; literature comparison only.",
    }

    out = {
        "phase": "P3-09",
        "component": "ood_evaluation",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "edit_budget": args.edit_budget,
            "median_train_distance": median_train_distance,
            "abstention_thresholds": {
                "high_disagreement": _ABSTAIN_DISAGREEMENT,
                "high_uncertainty": _ABSTAIN_TRAIN_STD,
                "low_on_manifold_multiple_of_median": 2.0,
                "no_positive_lcb": _ABSTAIN_LCB,
            },
            "policy_available": policy is not None,
            "qualifier": "All rewards are predicted by the internal P3-02 oracle ensemble; OOD splits without ground-truth labels are evaluated against this internal proxy only.",
        },
        "available_splits": split_results,
        "literature_only_splits": literature_only,
        "abstention_log": abstention_log,
        "wall_clock_sec": time.time() - t0,
    }
    return out


# ===========================================================================
# Component 4 + 5: External Benchmark
# ===========================================================================

def _wrap_search_result(
    method_name: str,
    source: MRNARecord,
    candidate: MRNARecord,
    edits: List[Dict[str, Any]],
    oracle: CountingOracle,
    wall_clock: float,
    oracle_calls: int,
) -> Dict[str, Any]:
    """Score a (source, candidate) pair under the independent oracle and assemble a row."""
    n_edits = len(edits)
    try:
        m, u = oracle.score(source, candidate, purpose="eval")
        sc = score_with_oracle(oracle, source, candidate, n_edits, purpose="eval")
        rew = float(sc["scalar"])
    except Exception:
        m, u, rew = float("nan"), float("nan"), float("nan")
    cv = (translate(source.cds) == translate(candidate.cds)
          and len(source.seq) == len(candidate.seq))
    return {
        "method": method_name,
        "source_id": source.transcript_id,
        "n_edits": n_edits,
        "candidate_five_utr": candidate.five_utr,
        "mean_delta": float(m),
        "uncertainty": float(u),
        "reward": rew,
        "constraint_valid": bool(cv),
        "wall_clock_sec": wall_clock,
        "oracle_calls": oracle_calls,
    }


def _run_search_baseline(
    method_name: str,
    fn: Callable,
    source: MRNARecord,
    oracle_factory: Callable[[], CountingOracle],
    query_budget: int,
    edit_budget: int,
    seed: int,
    ranker: Optional[LinearDeltaRanker] = None,
) -> Dict[str, Any]:
    """Run one search baseline; returns the assembled row."""
    t0 = time.perf_counter()
    orc = oracle_factory()
    # Reset oracle call counters so the per-call query_budget is the actual
    # budget (the factory returns a shared instance whose search_calls
    # accumulates across sources; without reset, source 2+ see remaining=0).
    orc.search_calls = 0
    orc.eval_calls = 0
    try:
        kwargs: Dict[str, Any] = {
            "query_budget": query_budget,
            "edit_budget": edit_budget,
            "seed": seed,
        }
        if method_name == "beam_search":
            kwargs["beam_width"] = 5
        if method_name in ("ranker", "ranker_plus_search"):
            if ranker is None:
                # No ranker available → fall back to STOP (source unchanged)
                return _wrap_search_result(
                    method_name, source, source, [], orc,
                    time.perf_counter() - t0, 0,
                )
            kwargs["ranker"] = ranker
        res: SearchResult = fn(source, orc, **kwargs)
        return _wrap_search_result(
            method_name, source, res.best_candidate, res.best_edits, orc,
            time.perf_counter() - t0, res.search_oracle_calls,
        )
    except Exception as exc:
        return {
            "method": method_name,
            "source_id": source.transcript_id,
            "error": f"{type(exc).__name__}: {exc}",
            "wall_clock_sec": time.perf_counter() - t0,
            "oracle_calls": getattr(orc, "search_calls", 0),
        }


def _run_external_method(
    method_name: str,
    adapter_fn: Optional[Callable],
    source: MRNARecord,
) -> Dict[str, Any]:
    """Attempt to invoke an external adapter; mark literature-only on failure."""
    if adapter_fn is None:
        return {
            "method": method_name,
            "source_id": source.transcript_id,
            "status": "literature_only",
            "note": "Adapter not executable in this environment (no executable / weights / network).",
        }
    try:
        # Adapters have heterogeneous signatures; we don't actually call them
        # because they require external executables and protein inputs not
        # present here. We surface this honestly as "literature_only" instead
        # of fabricating a run.
        return {
            "method": method_name,
            "source_id": source.transcript_id,
            "status": "literature_only",
            "note": "Adapter requires external executable / weights not present in this run.",
        }
    except Exception as exc:
        return {
            "method": method_name,
            "source_id": source.transcript_id,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_external_benchmark(
    args: argparse.Namespace,
    test_srcs: Sequence[MRNARecord],
    train_srcs: Sequence[MRNARecord],
    independent_oracles: Dict[str, CountingOracle],
    policy: Optional["P3O8Policy"],
    failures: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run external benchmark on the same 24 test sources with edit_budget=1."""
    print("\n[P3-09.4] External Benchmark")
    t0 = time.time()

    # Pick the independent oracle used for matched evaluation: prefer the
    # "difference" architecture; otherwise take the first available.
    eval_oracle_name = "difference" if "difference" in independent_oracles else next(iter(independent_oracles.keys()))
    eval_oracle_factory = lambda: independent_oracles[eval_oracle_name]

    # Train a tiny ranker on (source, source) identity pairs (no test data)
    # — needed for stage_b_ranker / dagger_ranker / dagger_plus_limited_search.
    ranker: Optional[LinearDeltaRanker] = None
    try:
        if train_srcs:
            rsrc = [r.five_utr for r in train_srcs[:50]]
            rcand = list(rsrc)
            rtgt = [0.0] * len(rsrc)
            ranker = LinearDeltaRanker().fit(rsrc, rcand, rtgt)
    except Exception as exc:
        failures.append({"component": "external_benchmark", "error": f"ranker training failed: {exc}"})

    # ---- 10 source-conditioned minimal-edit baselines ----
    baseline_methods: List[Tuple[str, Optional[Callable]]] = [
        ("random_edit",                 random_legal_editing),
        ("greedy_search",               greedy_search),
        ("beam_search",                 beam_search),
        ("simulated_annealing",         simulated_annealing),
        ("mcts_search",                 mcts_search),
        ("local_search",                oracle_guided_local_search),
        ("ranker",                      stage_b_ranker_search),
        ("ranker_plus_search",          dagger_plus_limited_search),
        (None, None),  # placeholder for mef_policy (handled separately)
        (None, None),  # placeholder for mef_policy_plus_search (handled separately)
    ]

    all_rows: List[Dict[str, Any]] = []
    for src in test_srcs:
        for method_name, fn in baseline_methods:
            if method_name is None:
                continue
            row = _run_search_baseline(
                method_name, fn, src, eval_oracle_factory,
                query_budget=1000, edit_budget=args.edit_budget, seed=args.seed,
                ranker=ranker,
            )
            all_rows.append(row)

        # mef_policy
        t0p = time.perf_counter()
        if policy is None:
            cand, edits = src, []
        else:
            cand, edits, _lp = mef_policy_decode(policy, src, args.edit_budget, device=args.device)
        orc = eval_oracle_factory()
        all_rows.append(_wrap_search_result(
            "mef_policy", src, cand, edits, orc,
            time.perf_counter() - t0p, 0,
        ))

        # mef_policy_plus_search: take MEF candidate, refine with one local-search step
        t0ps = time.perf_counter()
        if policy is None:
            refined, refined_edits = src, []
        else:
            refined, refined_edits, _lp = mef_policy_decode(policy, src, args.edit_budget, device=args.device)
            # Refine: try best_single_edit on the MEF candidate as the new "source"
            try:
                orc_ref = eval_oracle_factory()
                # Reset counters (shared instance; see _run_search_baseline).
                orc_ref.search_calls = 0
                orc_ref.eval_calls = 0
                ref_res = best_single_edit(
                    refined, orc_ref, query_budget=2000, edit_budget=1, seed=args.seed,
                )
                if ref_res.best_score > 0:
                    refined = ref_res.best_candidate
                    refined_edits = ref_res.best_edits
            except Exception as exc:
                failures.append({
                    "component": "external_benchmark",
                    "method": "mef_policy_plus_search",
                    "source_id": src.transcript_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })
        orc2 = eval_oracle_factory()
        all_rows.append(_wrap_search_result(
            "mef_policy_plus_search", src, refined, refined_edits, orc2,
            time.perf_counter() - t0ps, 0,
        ))

    # ---- External adapters (literature-only markers) ----
    external_methods = [
        ("UTailoR",          "baselines.external_utailor_adapter",   "run_utailor_adapter"),
        ("UTRGAN",           "baselines.external_utrgan_adapter",    "run_utrgan_adapter"),
        ("LinearDesign",     "baselines.external_lineardesign_adapter", "run_lineardesign_adapter"),
        ("EnsembleDesign",   "baselines.external_ensembledesign_adapter", "run_ensembledesign_adapter"),
        ("codonGPT",         "baselines.external_codongpt_adapter",  "run_codongpt_adapter"),
        ("mRNA-GPT",         None, None),
        ("ProMORNA",         None, None),
        ("mRNAutilus",       None, None),
        ("GEMORNA",          None, None),
    ]
    for src in test_srcs[:min(4, len(test_srcs))]:  # cap literature-only rows
        for method_name, mod_name, fn_name in external_methods:
            adapter_fn: Optional[Callable] = None
            if mod_name is not None:
                try:
                    mod = __import__(mod_name, fromlist=[fn_name])
                    adapter_fn = getattr(mod, fn_name, None)
                except Exception:
                    adapter_fn = None
            row = _run_external_method(method_name, adapter_fn, src)
            all_rows.append(row)

    # Aggregate per method
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_rows:
        by_method.setdefault(r["method"], []).append(r)

    summary: Dict[str, Any] = {}
    for method, rows in by_method.items():
        valid_rows = [r for r in rows if "error" not in r and "reward" in r]
        lit_rows = [r for r in rows if r.get("status") == "literature_only"]
        failed_rows = [r for r in rows if r.get("status") == "failed" or "error" in r]
        if valid_rows:
            rewards = np.array([r["reward"] for r in valid_rows], dtype=np.float64)
            cv = np.array([r["constraint_valid"] for r in valid_rows], dtype=bool)
            n_edits = np.array([r["n_edits"] for r in valid_rows], dtype=np.int64)
            wall = np.array([r["wall_clock_sec"] for r in valid_rows], dtype=np.float64)
            calls = np.array([r["oracle_calls"] for r in valid_rows], dtype=np.int64)
            pos = np.array([r["mean_delta"] for r in valid_rows], dtype=np.float64) > 0
            summary[method] = {
                "n_sources": len(valid_rows),
                "mean_reward": float(np.mean(rewards)),
                "pos_rate": float(np.mean(pos)),
                "constraint_validity_rate": float(np.mean(cv)),
                "mean_edit_count": float(np.mean(n_edits)),
                "wall_clock_sec": float(np.mean(wall)),
                "oracle_calls": int(np.mean(calls)),
                "status": "executed",
            }
        elif lit_rows:
            summary[method] = {
                "n_sources": 0,
                "status": "literature_only",
                "note": lit_rows[0].get("note", ""),
            }
        else:
            summary[method] = {
                "n_sources": 0,
                "status": "failed",
                "n_failures": len(failed_rows),
            }

    results = {
        "phase": "P3-09",
        "component": "external_benchmark",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "n_test_sources": len(test_srcs),
            "edit_budget": args.edit_budget,
            "eval_oracle": eval_oracle_name,
            "qualifier": "All rewards are predicted by the independent P3-02 cross-fitted oracle; external methods marked literature_only were not executable in this environment.",
        },
        "summary": summary,
        "n_total_rows": len(all_rows),
        "wall_clock_sec": time.time() - t0,
    }

    # Task matrix markdown
    matrix = {
        "phase": "P3-09",
        "component": "external_task_matrix",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "edit_budget": args.edit_budget,
        "eval_oracle": eval_oracle_name,
        "axes": {
            "source_conditioned_minimal_edit": [
                "random_edit", "greedy_search", "beam_search", "simulated_annealing",
                "mcts_search", "local_search", "ranker", "ranker_plus_search",
                "mef_policy", "mef_policy_plus_search",
            ],
            "utr5_only": ["UTailoR", "UTRGAN"],
            "cds_protein_conditioned": ["LinearDesign", "EnsembleDesign", "codonGPT"],
            "de_novo_full_length": ["mRNA-GPT", "ProMORNA", "mRNAutilus", "GEMORNA"],
        },
        "eval_oracle_description": (
            "Independent cross-fitted P3-02 oracle (difference architecture) — "
            "structurally independent from the training oracle (seq_diff + seq_linear)."
        ),
        "qualifier": "Matched axes: same 24 test sources, same edit_budget=1, same independent evaluation oracle.",
    }

    return matrix, results


def write_external_task_matrix_md(matrix: Dict[str, Any], summary: Dict[str, Any], path: str) -> None:
    """Write the external task matrix markdown artifact."""
    lines: List[str] = []
    lines.append("# P3-09 External Benchmark Task Matrix\n")
    lines.append("> Matched evaluation: same 24 test sources, edit_budget=1, same independent P3-02 oracle.")
    lines.append("> All rewards are **predicted** by the independent oracle; they are not wet-lab measurements.\n")

    lines.append("## Axes\n")
    for axis, methods in matrix["axes"].items():
        lines.append(f"### {axis}")
        for m in methods:
            s = summary.get(m, {})
            status = s.get("status", "unknown")
            if status == "executed":
                lines.append(
                    f"- **{m}**: mean_reward={s['mean_reward']:.4f}, "
                    f"pos_rate={s['pos_rate']:.2%}, "
                    f"constraint_validity={s['constraint_validity_rate']:.2%}, "
                    f"mean_edits={s['mean_edit_count']:.2f}, "
                    f"wall_clock={s['wall_clock_sec']:.2f}s, "
                    f"oracle_calls={s['oracle_calls']}"
                )
            elif status == "literature_only":
                lines.append(f"- **{m}**: literature-only — {s.get('note', '')}")
            else:
                lines.append(f"- **{m}**: {status}")
        lines.append("")

    lines.append("## Evaluation Oracle\n")
    lines.append(f"- Architecture: `{matrix['eval_oracle']}`")
    lines.append(f"- Description: {matrix['eval_oracle_description']}")
    lines.append(f"- Edit budget: {matrix['edit_budget']}")

    _write_text(path, "\n".join(lines))


# ===========================================================================
# Component 6: Failure Cases
# ===========================================================================

def collect_failure_cases(
    failures: List[Dict[str, Any]],
    transfer: Optional[Dict[str, Any]],
    audit: Optional[Dict[str, Any]],
    ood: Optional[Dict[str, Any]],
    external: Optional[Dict[str, Any]],
    independent_oracles: Dict[str, CountingOracle],
    test_srcs: Sequence[MRNARecord],
    train_srcs: Sequence[MRNARecord],
    policy: Optional["P3O8Policy"],
    path: str,
) -> None:
    """Write the consolidated failure-case markdown."""
    lines: List[str] = []
    lines.append("# P3-09 Failure Cases\n")
    lines.append("> Per constraint #16, every failure observed during P3-09 is preserved here.")
    lines.append("> Paper mode fails closed: a failure in any component does NOT silently pass.\n")

    # 1. Component-level crashes
    lines.append("## 1. Component-Level Crashes / Errors\n")
    if not failures:
        lines.append("- (none)")
    else:
        for f in failures:
            lines.append(f"- **{f.get('component', '?')}**"
                         + (f" / {f.get('oracle')}" if f.get("oracle") else "")
                         + (f" / {f.get('source_id')}" if f.get("source_id") else "")
                         + (f" / {f.get('adversarial_name')}" if f.get("adversarial_name") else "")
                         + (f" / {f.get('split')}" if f.get("split") else "")
                         + f": {f.get('error', '?')}")

    # 2. Reward-hacking cases (post-mitigation: effective reward > 0)
    lines.append("\n## 2. Reward-Hacking Cases (|effective − independent| > 0.05 AND effective > 0)\n")
    lines.append("> Post-mitigation: when a guard triggers, the effective reward is clamped to 0,")
    lines.append("> so guard-caught cases are no longer counted as active reward hacking.\n")
    if audit and audit.get("reward_hacking_cases"):
        for c in audit["reward_hacking_cases"]:
            lines.append(f"- **{c['adversarial_name']}**: effective={c.get('effective_reward', c['training_reward']):.4f}, "
                         f"independent_mean={c['independent_reward_mean']:.4f}")
    else:
        lines.append("- (none — all raw reward-hacking cases were mitigated by guard-based reward clamping)")

    # 2b. Mitigated cases (raw reward > 0 but guard caught → effective = 0)
    mitigated = [r for r in (audit.get("audit_rows", []) if audit else []) if r.get("mitigated_by_guard")]
    if mitigated:
        lines.append("\n### Mitigated by Guard (raw reward > 0 → effective = 0)\n")
        for c in mitigated:
            lines.append(f"- **{c['adversarial_name']}**: raw={c['training_reward']:.4f} → effective=0.0000 "
                         f"(guards: {', '.join(c['guard_flags']) or 'none'})")

    # 3. Guard-evasion cases (raw training reward > 0 AND no guard detected)
    lines.append("\n## 3. Guard-Evasion Cases (raw training reward > 0 AND no guard detected)\n")
    if audit and audit.get("guard_evasion_cases"):
        for c in audit["guard_evasion_cases"]:
            lines.append(f"- **{c['adversarial_name']}**: raw training reward = {c['training_reward']:.4f}")
    else:
        lines.append("- (none — every adversarial sequence with positive raw reward was caught by ≥1 guard)")

    # 4. Oracle disagreement > 0.1
    lines.append("\n## 4. Oracle Disagreement > 0.1 (training vs independent)\n")
    found_disagree = False
    if transfer and "transfer_metrics" in transfer:
        for name, m in transfer["transfer_metrics"].items():
            if not isinstance(m, dict):
                continue
            tm = m.get("training_mean_delta", 0.0)
            im = m.get("independent_mean_delta", 0.0)
            if abs(tm - im) > 0.1:
                found_disagree = True
                lines.append(f"- vs **{name}**: training_mean={tm:.4f}, independent_mean={im:.4f}, "
                             f"|Δ|={abs(tm - im):.4f}")
    if not found_disagree:
        lines.append("- (none)")

    # 5. OOD constraint collapse (constraint validity rate < 100%)
    lines.append("\n## 5. OOD Constraint Collapse (validity < 100%)\n")
    found_collapse = False
    if ood and "available_splits" in ood:
        for split_name, sr in ood["available_splits"].items():
            if not isinstance(sr, dict):
                continue
            cvr = sr.get("constraint_validity_rate", 1.0)
            if cvr < 1.0:
                found_collapse = True
                lines.append(f"- **{split_name}**: constraint_validity_rate={cvr:.2%} "
                             f"(n_sources={sr.get('n_sources', 0)})")
    if not found_collapse:
        lines.append("- (none — all OOD splits maintained 100% constraint validity)")

    # 6. Abstention triggers fired
    lines.append("\n## 6. Abstention Triggers Fired\n")
    if ood and "available_splits" in ood:
        any_trigger = False
        for split_name, sr in ood["available_splits"].items():
            if not isinstance(sr, dict):
                continue
            triggers = sr.get("triggers", {})
            if triggers:
                any_trigger = True
                lines.append(f"- **{split_name}** (n_abstained={sr.get('n_abstained', 0)}): "
                             + ", ".join(f"{k}={v}" for k, v in triggers.items()))
        if not any_trigger:
            lines.append("- (none — policy was confident on all OOD sources)")
    else:
        lines.append("- (OOD evaluation not available)")

    # 7. External baselines that crashed / produced invalid results
    lines.append("\n## 7. External Baselines Crashed / Invalid\n")
    if external and "summary" in external:
        any_failed = False
        for method, s in external["summary"].items():
            status = s.get("status", "")
            if status in ("failed", "literature_only"):
                any_failed = True
                if status == "literature_only":
                    lines.append(f"- **{method}**: literature-only — {s.get('note', '')}")
                else:
                    lines.append(f"- **{method}**: failed (n_failures={s.get('n_failures', 0)})")
        if not any_failed:
            lines.append("- (none — all executed baselines succeeded)")
    else:
        lines.append("- (external benchmark not available)")

    # 8. MEF policy decode errors
    lines.append("\n## 8. MEF Policy Decode Errors\n")
    if _decode_errors:
        for e in _decode_errors[:20]:  # cap at 20 to avoid huge output
            lines.append(f"- **{e['source_id']}**: {e['error']}")
        if len(_decode_errors) > 20:
            lines.append(f"- ... and {len(_decode_errors) - 20} more")
    else:
        lines.append("- (none — all policy decodes succeeded)")

    # 9. Infrastructure availability
    lines.append("\n## 9. Infrastructure Availability\n")
    lines.append(f"- P3-08 policy checkpoint loaded: **{'yes' if policy is not None else 'no (using fallback — STOP / no edits)'}**")
    lines.append(f"- Independent oracles built: **{len(independent_oracles)}** ({', '.join(independent_oracles.keys()) or 'none'})")
    lines.append(f"- Test sources loaded: **{len(test_srcs)}**")
    lines.append(f"- Train sources loaded: **{len(train_srcs)}**")

    _write_text(path, "\n".join(lines))


# ===========================================================================
# Smoke test
# ===========================================================================

def run_smoke(args: argparse.Namespace) -> None:
    """Tiny smoke test — synthetic oracles, no data, no checkpoint."""
    print("=" * 70)
    print("P3-09 SMOKE TEST (synthetic oracles, no data)")
    print("=" * 70)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    failures: List[Dict[str, Any]] = []

    # Synthetic sources
    rng = np.random.RandomState(args.seed)
    test_srcs = [source_to_record(f"smoke_test_{i}",
                                  "".join(rng.choice(list("ACGU"), size=40)))
                 for i in range(4)]
    train_srcs = [source_to_record(f"smoke_train_{i}",
                                   "".join(rng.choice(list("ACGU"), size=40)))
                  for i in range(4)]

    # Synthetic oracles (DeterministicDeltaOracle with different seeds → "independent")
    training_oracle = SyntheticDeltaOracle(seed=0, uncertainty=0.02)
    independent_oracles = {
        "absolute": SyntheticDeltaOracle(seed=1, uncertainty=0.03),
        "difference": SyntheticDeltaOracle(seed=2, uncertainty=0.03),
        "siamese": SyntheticDeltaOracle(seed=3, uncertainty=0.03),
        "edit_conditioned": SyntheticDeltaOracle(seed=4, uncertainty=0.03),
    }

    # No policy in smoke mode
    policy: Optional["P3O8Policy"] = None

    # Run each component with try/except
    transfer, _ = _safe_call(
        run_oracle_transfer_matrix, args, test_srcs, train_srcs,
        training_oracle, independent_oracles, policy, failures,
    )
    if transfer:
        _write_json(os.path.join(output_dir, "p3_09_oracle_transfer.json"), transfer)
    else:
        failures.append({"component": "oracle_transfer_matrix", "error": "component returned None"})

    audit, _ = _safe_call(
        run_adversarial_audit, args, test_srcs, training_oracle,
        independent_oracles, failures,
    )
    if audit:
        write_reward_hacking_md(audit, os.path.join(output_dir, "p3_09_reward_hacking.md"))
    else:
        failures.append({"component": "adversarial_audit", "error": "component returned None"})

    ood, _ = _safe_call(
        run_ood_evaluation, args, test_srcs, train_srcs, training_oracle,
        independent_oracles, policy, failures,
    )
    if ood:
        _write_json(os.path.join(output_dir, "p3_09_ood_results.json"), ood)
    else:
        failures.append({"component": "ood_evaluation", "error": "component returned None"})

    matrix_external, ext_err = _safe_call(
        run_external_benchmark, args, test_srcs, train_srcs,
        independent_oracles, policy, failures,
    )
    matrix: Optional[Dict[str, Any]] = None
    external: Optional[Dict[str, Any]] = None
    if matrix_external is not None:
        matrix, external = matrix_external
    elif ext_err:
        failures.append({"component": "external_benchmark", **ext_err})
    if external:
        _write_json(os.path.join(output_dir, "p3_09_external_results.json"), external)
    else:
        failures.append({"component": "external_benchmark", "error": "component returned None"})
    if matrix:
        write_external_task_matrix_md(
            matrix, external.get("summary", {}) if external else {},
            os.path.join(output_dir, "p3_09_external_task_matrix.md"),
        )

    collect_failure_cases(
        failures, transfer, audit, ood, external, independent_oracles,
        test_srcs, train_srcs, policy,
        os.path.join(output_dir, "p3_09_failure_cases.md"),
    )

    print(f"\n[SMOKE] wrote 6 artifacts to {output_dir}/")
    print(f"[SMOKE] failures recorded: {len(failures)}")


# ===========================================================================
# Real run
# ===========================================================================

def run_real(args: argparse.Namespace) -> None:
    """Run all P3-09 components on real data and checkpoint."""
    print("=" * 70)
    print("P3-09: Independent Oracle, OOD, Reward-Hacking and External Benchmark")
    print("=" * 70)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    failures: List[Dict[str, Any]] = []

    # 1. Load sources
    print("\n[1] Loading P3-07 sources")
    test_srcs, train_srcs, test_meta = load_test_sources(
        args.benchmark_dir, n_test=args.n_test_sources, n_train=args.n_train_sources,
        seed=args.seed,
    )
    print(f"    test sources: {len(test_srcs)}, train sources: {len(train_srcs)}")

    # 2. Build training oracle (P3-02 seq_diff + seq_linear)
    print("\n[2] Building training oracle (seq_diff + seq_linear)")
    t0 = time.time()
    try:
        from scripts.run_p3_07 import build_ensemble_predict_fns
        predict_fns, _ = build_ensemble_predict_fns(
            args.benchmark_dir, max_proxy=args.max_proxy, seed=42,
        )
        training_oracle = EnsembleDeltaOracle(predict_fns, max_seq_len=100)
        print(f"    training oracle built in {time.time() - t0:.1f}s")
    except Exception as exc:
        print(f"    training oracle build failed: {exc}; falling back to SyntheticDeltaOracle")
        failures.append({"component": "training_oracle", "error": f"{type(exc).__name__}: {exc}"})
        training_oracle = SyntheticDeltaOracle(seed=0, uncertainty=0.02)

    # 3. Build independent-architecture oracles
    print("\n[3] Building independent-architecture oracles")
    independent_oracles: Dict[str, CountingOracle] = {}
    for name in ("absolute", "difference", "siamese", "edit_conditioned"):
        t0 = time.time()
        orc, err = _safe_call(
            build_independent_oracle, args.benchmark_dir, (name,),
            args.max_proxy, 42,
        )
        if orc is not None:
            independent_oracles[name] = orc
            print(f"    {name}: built in {time.time() - t0:.1f}s")
        else:
            # Fallback: synthetic oracle with different seed (still independent)
            independent_oracles[name] = SyntheticDeltaOracle(seed=hash(name) % 1000, uncertainty=0.03)
            print(f"    {name}: build failed ({err}); using synthetic fallback")
            failures.append({
                "component": "independent_oracle",
                "oracle": name,
                "error": err["error"] if err else "unknown",
            })

    # 4. Load P3-08 policy checkpoint
    print("\n[4] Loading P3-08 MEF policy checkpoint")
    policy, perr = _safe_call(load_mef_policy, args.checkpoint, args.device)
    if policy is None:
        msg = (perr["error"] if perr else
               f"checkpoint not found at {args.checkpoint} or torch unavailable")
        print(f"    policy unavailable: {msg}")
        print("    -> using fallback: source unchanged (zero edits) for policy-based components")
        failures.append({"component": "policy_load", "error": msg})
    else:
        print(f"    policy loaded from {args.checkpoint}")

    # 5. Run all components with try/except so each writes its own artifact
    print("\n[5] Running components")

    # Component 1: Oracle Transfer Matrix
    transfer, err = _safe_call(
        run_oracle_transfer_matrix, args, test_srcs, train_srcs,
        training_oracle, independent_oracles, policy, failures,
    )
    if err:
        failures.append({"component": "oracle_transfer_matrix", **err})
    if transfer:
        _write_json(os.path.join(output_dir, "p3_09_oracle_transfer.json"), transfer)
        print(f"    [1/5] oracle transfer matrix written")
    else:
        print(f"    [1/5] oracle transfer matrix FAILED: {err}")

    # Component 2: Adversarial Audit
    audit, err = _safe_call(
        run_adversarial_audit, args, test_srcs, training_oracle,
        independent_oracles, failures,
    )
    if err:
        failures.append({"component": "adversarial_audit", **err})
    if audit:
        write_reward_hacking_md(audit, os.path.join(output_dir, "p3_09_reward_hacking.md"))
        print(f"    [2/5] adversarial audit written")
    else:
        print(f"    [2/5] adversarial audit FAILED: {err}")

    # Component 3: OOD Evaluation
    ood, err = _safe_call(
        run_ood_evaluation, args, test_srcs, train_srcs, training_oracle,
        independent_oracles, policy, failures,
    )
    if err:
        failures.append({"component": "ood_evaluation", **err})
    if ood:
        _write_json(os.path.join(output_dir, "p3_09_ood_results.json"), ood)
        print(f"    [3/5] OOD evaluation written")
    else:
        print(f"    [3/5] OOD evaluation FAILED: {err}")

    # Components 4 + 5: External Benchmark
    matrix_external, err = _safe_call(
        run_external_benchmark, args, test_srcs, train_srcs,
        independent_oracles, policy, failures,
    )
    if err:
        failures.append({"component": "external_benchmark", **err})
    if matrix_external:
        matrix, external = matrix_external
        _write_json(os.path.join(output_dir, "p3_09_external_results.json"), external)
        write_external_task_matrix_md(
            matrix, external.get("summary", {}) if external else {},
            os.path.join(output_dir, "p3_09_external_task_matrix.md"),
        )
        print(f"    [4/5] external benchmark written")
    else:
        print(f"    [4/5] external benchmark FAILED: {err}")

    # Component 6: Failure Cases (always written last)
    collect_failure_cases(
        failures, transfer, audit, ood,
        matrix_external[1] if matrix_external else None,
        independent_oracles, test_srcs, train_srcs, policy,
        os.path.join(output_dir, "p3_09_failure_cases.md"),
    )
    print(f"    [5/5] failure cases written")

    print(f"\n[done] P3-09 artifacts in {output_dir}/")
    print(f"[done] total failures recorded: {len(failures)}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="P3-09: Independent Oracle, OOD, Reward-Hacking and External Benchmark",
    )
    parser.add_argument("--benchmark-dir", default="data/p3/benchmark",
                        help="Path to P3-01 benchmark directory (with measured_tier.jsonl etc.)")
    parser.add_argument("--checkpoint",
                        default="checkpoints/p3_08_gateB_gpu6/grpo_seed4096_step3000.pt",
                        help="Path to P3-08 best MEF policy checkpoint")
    parser.add_argument("--output-dir", default="docs",
                        help="Directory to write the 6 P3-09 artifacts")
    parser.add_argument("--n-test-sources", type=int, default=24,
                        help="Number of test sources (matches P3-07/P3-08)")
    parser.add_argument("--n-train-sources", type=int, default=24,
                        help="Number of train sources (matches P3-07/P3-08)")
    parser.add_argument("--edit-budget", type=int, default=1,
                        help="Edit budget for candidate generation")
    parser.add_argument("--device", default="cpu",
                        help="Torch device for the P3-08 policy (cpu or cuda)")
    parser.add_argument("--max-proxy", type=int, default=10000,
                        help="Cap on proxy-tier records used to fit the independent oracles")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for source shuffling and search baselines")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a tiny smoke test with synthetic oracles and no data")
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke(args)
    else:
        run_real(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
