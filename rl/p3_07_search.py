"""P3-07: Strong-search ceiling and RL necessity gate.

Answers, before any large-scale GRPO: does this task actually need RL?

Implements, on top of the P3-06 minimal-edit MDP action space:

Ten search baselines (all budget-respecting):
  1. random_legal_editing
  2. best_single_edit            (== exact one-edit optimum when budget allows)
  3. greedy_search
  4. stage_b_ranker_search       (learned ranker, 0 guidance oracle calls)
  5. beam_search
  6. simulated_annealing
  7. mcts_search
  8. oracle_guided_local_search
  9. dagger_ranker_search        (DAgger-trained ranker, 0 guidance calls)
 10. dagger_plus_limited_search  (DAgger ranker + oracle-verified beam)

Exact evaluation on tractable small states:
  - exact_one_edit_optimum
  - exact_two_edit_optimum
  - tiny_mdp_value_iteration     (finite-horizon optimal policy by DP)
  - regret: search optimality gap / greedy regret / ranker regret / policy regret

Algorithm semantics on a tiny enumerable state space:
  greedy intensity vs stochastic CTMC vs finite-horizon optimal policy vs
  exact terminal marginal vs beam search, compared with action KL, terminal
  KL, argmax agreement, expected return, constraint validity.

Query accounting: every oracle evaluation of a (source, candidate) pair goes
through CountingOracle and is counted. ``search`` calls are guidance queries
charged to the query budget; ``eval`` calls are the single final verification
of the proposed design (like a wet-lab validation readout) and are reported
separately. All baselines share the same scoring rule: Reward v3 with context
``protein_output_focused`` (LCB primary + edit-cost secondary), see
rl/p3_06_mdp.compute_reward_v3.
"""
from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.constants import NUC_VOCAB, translate
from core.schema import MRNARecord
from rl.p3_06_mdp import (
    EditAction,
    STOP_EDIT,
    apply_edit_action,
    build_legal_edit_actions,
    RewardV3Config,
    compute_reward_v3,
)


# ===========================================================================
# Region-restricted legal actions
# ===========================================================================

def legal_actions(
    record: MRNARecord,
    visited: Optional[set] = None,
    regions: Sequence[str] = ("five_utr",),
) -> List[EditAction]:
    """Legal P3-06 actions restricted to the given regions.

    The active primary task (p3_task_v2, task_a) allows only
    ``five_utr_substitution``; CDS synonymous actions belong to frozen
    fallback Task B and are excluded by default.
    """
    actions = build_legal_edit_actions(record, visited)
    allowed = []
    for a in actions:
        if a.is_stop():
            allowed.append(a)
        elif a.region() in regions:
            allowed.append(a)
    return allowed


def _seq_hash(record: MRNARecord) -> str:
    return hashlib.md5(record.seq.encode()).hexdigest()


# ===========================================================================
# Counting oracle
# ===========================================================================

class BudgetExhausted(RuntimeError):
    """Raised when a search tries to exceed its query budget."""


class CountingOracle:
    """Wraps a delta scorer; counts guidance (search) and verification calls.

    Subclasses implement ``_score(source, candidate) -> (mean_delta, unc)``.
    """

    def __init__(self, query_budget: Optional[int] = None):
        self.query_budget = query_budget
        self.search_calls = 0
        self.eval_calls = 0

    @property
    def remaining(self) -> float:
        if self.query_budget is None:
            return math.inf
        return self.query_budget - self.search_calls

    def score(
        self,
        source: MRNARecord,
        candidate: MRNARecord,
        *,
        purpose: str = "search",
    ) -> Tuple[float, float]:
        if purpose == "search":
            if self.query_budget is not None and self.search_calls >= self.query_budget:
                raise BudgetExhausted(
                    f"query budget {self.query_budget} exhausted"
                )
            self.search_calls += 1
        elif purpose == "eval":
            self.eval_calls += 1
        else:
            raise ValueError(f"unknown purpose {purpose!r}")
        return self._score(source, candidate)

    def _score(self, source: MRNARecord, candidate: MRNARecord) -> Tuple[float, float]:
        raise NotImplementedError


class SyntheticDeltaOracle(CountingOracle):
    """Deterministic analytic oracle for unit tests and tiny-MDP semantics.

    delta is a position-weighted function of 5'UTR base composition with
    interaction terms between positions, so multi-edit optima are NOT the
    sum of single-edit optima (makes search non-trivial). Uncertainty is a
    fixed small constant so LCB = mean - const.
    """

    def __init__(
        self,
        query_budget: Optional[int] = None,
        uncertainty: float = 0.05,
        seed: int = 0,
    ):
        super().__init__(query_budget)
        self._unc = float(uncertainty)
        rng = np.random.RandomState(seed)
        # Position weights (up to 64 positions) and pairwise interactions.
        self._pos_w = rng.randn(64, 4) * 0.15
        self._pair_w = rng.randn(16, 4, 4) * 0.08

    def _score(self, source: MRNARecord, candidate: MRNARecord) -> Tuple[float, float]:
        src, cand = source.five_utr, candidate.five_utr
        n = min(len(cand), 64)
        delta = 0.0
        nuc_idx = {ch: i for i, ch in enumerate(NUC_VOCAB)}
        for i in range(n):
            c = nuc_idx.get(cand[i], 0)
            s = nuc_idx.get(src[i] if i < len(src) else "A", 0)
            delta += self._pos_w[i, c] - self._pos_w[i, s]
        # Pairwise interaction among first 16 positions (epistasis).
        m = min(n, 16)
        for i in range(m):
            for j in range(i + 1, m):
                ci = nuc_idx.get(cand[i], 0)
                cj = nuc_idx.get(cand[j], 0)
                si = nuc_idx.get(src[i] if i < len(src) else "A", 0)
                sj = nuc_idx.get(src[j] if j < len(src) else "A", 0)
                delta += self._pair_w[i, ci, cj] - self._pair_w[i, si, sj]
        return float(delta), self._unc


class EnsembleDeltaOracle(CountingOracle):
    """Cross-fitted P3-02 ensemble as oracle (server-side real runs).

    Wraps a predictor object exposing ``predict_delta(features)`` and a
    feature extractor compatible with core.p3_02_delta_oracle.
    """

    def __init__(
        self,
        predict_fns: Sequence[Callable[[Dict[str, np.ndarray]], np.ndarray]],
        max_seq_len: int = 100,
        query_budget: Optional[int] = None,
    ):
        super().__init__(query_budget)
        if len(predict_fns) < 2:
            raise ValueError("ensemble needs >=2 predict fns for uncertainty")
        self._fns = list(predict_fns)
        self._max_seq_len = max_seq_len
        # Cache for source-bias centering: maps source 5'UTR → (mean, std) at 0-edit.
        # The raw ensemble predicts a non-zero delta for source→source because the
        # training label mean is negative. Centering subtracts this bias so that
        # delta(source→source) = 0 by construction, which is required for the
        # degenerate-reference guard and the relative-headroom decision to work.
        self._source_bias_cache: Dict[str, Tuple[float, float]] = {}

    def _raw_predict(self, source: MRNARecord, candidate: MRNARecord) -> np.ndarray:
        """Raw per-model predictions (no centering)."""
        from core.p3_02_delta_oracle import extract_features

        edits = _diff_edits(source.five_utr, candidate.five_utr)
        feats = extract_features(
            source.five_utr, candidate.five_utr, edits, self._max_seq_len
        )
        batch = {k: v[np.newaxis] for k, v in feats.items()}
        return np.array([float(fn(batch)[0]) for fn in self._fns])

    def _get_source_bias(self, source: MRNARecord) -> np.ndarray:
        """Per-model 0-edit prediction for this source (cached)."""
        key = source.five_utr
        if key not in self._source_bias_cache:
            self._source_bias_cache[key] = self._raw_predict(source, source)
        return self._source_bias_cache[key]

    def _score(self, source: MRNARecord, candidate: MRNARecord) -> Tuple[float, float]:
        preds = self._raw_predict(source, candidate)
        # Center: subtract each model's 0-edit prediction so delta(source→source)=0.
        bias = self._get_source_bias(source)
        centered = preds - bias
        mean = float(centered.mean())
        std = float(centered.std(ddof=0)) + 1e-6
        return mean, std


def _diff_edits(src: str, cand: str) -> List[Dict[str, Any]]:
    edits = []
    for i, (a, b) in enumerate(zip(src, cand)):
        if a != b:
            edits.append({"pos": i, "ref": a, "alt": b, "region": "five_utr"})
    return edits


# ===========================================================================
# Scoring helper (Reward v3, protein_output_focused context)
# ===========================================================================

def score_candidate(
    source: MRNARecord,
    candidate: MRNARecord,
    oracle: CountingOracle,
    n_edits: int,
    cfg: Optional[RewardV3Config] = None,
    *,
    purpose: str = "search",
) -> Dict[str, Any]:
    mean, unc = oracle.score(source, candidate, purpose=purpose)
    out = compute_reward_v3(
        source,
        candidate,
        predicted_deltas={"protein_output": mean},
        uncertainties={"protein_output": unc},
        n_edits=n_edits,
        config=cfg or RewardV3Config(context="protein_output_focused"),
    )
    out["mean_delta"] = mean
    out["uncertainty"] = unc
    return out


# ===========================================================================
# Search result
# ===========================================================================

@dataclass
class SearchResult:
    method: str
    source_id: str
    best_candidate: MRNARecord
    best_edits: List[Dict[str, Any]]
    best_score: float           # reward v3 scalar (LCB + edit cost)
    best_mean_delta: float
    best_uncertainty: float
    search_oracle_calls: int
    eval_oracle_calls: int
    wall_clock_sec: float
    query_budget: int
    edit_budget: int
    n_root_actions: int
    constraint_valid: bool
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "source_id": self.source_id,
            "best_candidate_five_utr": self.best_candidate.five_utr,
            "best_edits": self.best_edits,
            "n_edits": len(self.best_edits),
            "best_score": self.best_score,
            "best_mean_delta": self.best_mean_delta,
            "best_uncertainty": self.best_uncertainty,
            "search_oracle_calls": self.search_oracle_calls,
            "eval_oracle_calls": self.eval_oracle_calls,
            "wall_clock_sec": self.wall_clock_sec,
            "query_budget": self.query_budget,
            "edit_budget": self.edit_budget,
            "n_root_actions": self.n_root_actions,
            "constraint_valid": self.constraint_valid,
            "provenance": self.provenance,
        }


def _check_constraints(source: MRNARecord, candidate: MRNARecord) -> bool:
    """Hard-constraint check: protein identity + length invariance."""
    return (
        translate(source.cds) == translate(candidate.cds)
        and len(source.seq) == len(candidate.seq)
    )


def _finalize(
    method: str,
    source: MRNARecord,
    best: Tuple[MRNARecord, List[Dict[str, Any]]],
    oracle: CountingOracle,
    t0: float,
    query_budget: int,
    edit_budget: int,
    n_root_actions: int,
    cfg: RewardV3Config,
    provenance: Optional[Dict[str, Any]] = None,
) -> SearchResult:
    """Verify the proposed best candidate with one eval call and package."""
    cand, edits = best
    sc = score_candidate(source, cand, oracle, len(edits), cfg, purpose="eval")
    return SearchResult(
        method=method,
        source_id=source.transcript_id,
        best_candidate=cand,
        best_edits=edits,
        best_score=sc["scalar"],
        best_mean_delta=sc["mean_delta"],
        best_uncertainty=sc["uncertainty"],
        search_oracle_calls=oracle.search_calls,
        eval_oracle_calls=oracle.eval_calls,
        wall_clock_sec=time.perf_counter() - t0,
        query_budget=query_budget,
        edit_budget=edit_budget,
        n_root_actions=n_root_actions,
        constraint_valid=_check_constraints(source, cand),
        provenance=provenance or {},
    )


# ===========================================================================
# Baseline 1: random legal editing
# ===========================================================================

def random_legal_editing(
    source: MRNARecord,
    oracle: CountingOracle,
    *,
    query_budget: int,
    edit_budget: int,
    seed: int = 0,
    regions: Sequence[str] = ("five_utr",),
    cfg: Optional[RewardV3Config] = None,
) -> SearchResult:
    """Sample random legal edit walks; keep the oracle-best candidate."""
    t0 = time.perf_counter()
    cfg = cfg or RewardV3Config(context="protein_output_focused")
    rng = random.Random(seed)
    oracle.query_budget = query_budget

    n_root = len(legal_actions(source, regions=regions))
    best = (source, [])
    best_score = score_candidate(source, source, oracle, 0, cfg)["scalar"]

    while oracle.remaining > 0:
        n_edits = rng.randint(1, edit_budget)
        rec = source
        visited = {_seq_hash(source)}
        ok = True
        for _ in range(n_edits):
            acts = [a for a in legal_actions(rec, visited, regions) if not a.is_stop()]
            if not acts:
                ok = False
                break
            a = rng.choice(acts)
            rec = apply_edit_action(rec, a)
            visited.add(_seq_hash(rec))
        if not ok:
            continue
        edits = _diff_edits(source.five_utr, rec.five_utr)
        try:
            sc = score_candidate(source, rec, oracle, len(edits), cfg)
        except BudgetExhausted:
            break
        if sc["scalar"] > best_score:
            best_score = sc["scalar"]
            best = (rec, edits)

    return _finalize("random_legal_editing", source, best, oracle, t0,
                     query_budget, edit_budget, n_root, cfg)


# ===========================================================================
# Baseline 2: best single edit (== exact one-edit optimum if budget allows)
# ===========================================================================

def best_single_edit(
    source: MRNARecord,
    oracle: CountingOracle,
    *,
    query_budget: int,
    edit_budget: int = 1,
    seed: int = 0,
    regions: Sequence[str] = ("five_utr",),
    cfg: Optional[RewardV3Config] = None,
) -> SearchResult:
    """Evaluate single-edit candidates (all if budget allows, else random
    subset of size query_budget) and keep the best."""
    t0 = time.perf_counter()
    cfg = cfg or RewardV3Config(context="protein_output_focused")
    rng = random.Random(seed)
    oracle.query_budget = query_budget

    acts = [a for a in legal_actions(source, regions=regions) if not a.is_stop()]
    n_root = len(acts) + 1
    if len(acts) > query_budget:
        acts = rng.sample(acts, query_budget)

    best = (source, [])
    best_score = score_candidate(source, source, oracle, 0, cfg)["scalar"]
    for a in acts:
        rec = apply_edit_action(source, a)
        edits = _diff_edits(source.five_utr, rec.five_utr)
        try:
            sc = score_candidate(source, rec, oracle, 1, cfg)
        except BudgetExhausted:
            break
        if sc["scalar"] > best_score:
            best_score = sc["scalar"]
            best = (rec, edits)

    return _finalize("best_single_edit", source, best, oracle, t0,
                     query_budget, 1, n_root, cfg)


# ===========================================================================
# Baseline 3: greedy
# ===========================================================================

def greedy_search(
    source: MRNARecord,
    oracle: CountingOracle,
    *,
    query_budget: int,
    edit_budget: int,
    seed: int = 0,
    regions: Sequence[str] = ("five_utr",),
    cfg: Optional[RewardV3Config] = None,
) -> SearchResult:
    """Full-neighborhood greedy: each step evaluates all legal children and
    moves to the best while it improves on the current score."""
    t0 = time.perf_counter()
    cfg = cfg or RewardV3Config(context="protein_output_focused")
    oracle.query_budget = query_budget

    n_root = len(legal_actions(source, regions=regions))
    current = source
    edits: List[Dict[str, Any]] = []
    visited = {_seq_hash(source)}
    cur_score = score_candidate(source, source, oracle, 0, cfg)["scalar"]
    best = (source, [])
    best_score = cur_score

    for _step in range(edit_budget):
        acts = [a for a in legal_actions(current, visited, regions) if not a.is_stop()]
        step_best_rec = None
        step_best_score = cur_score
        for a in acts:
            rec = apply_edit_action(current, a)
            try:
                sc = score_candidate(source, rec, oracle, len(edits) + 1, cfg)
            except BudgetExhausted:
                break
            if sc["scalar"] > step_best_score:
                step_best_score = sc["scalar"]
                step_best_rec = rec
        if step_best_rec is None:
            break  # no improving child or budget exhausted
        current = step_best_rec
        visited.add(_seq_hash(current))
        edits = _diff_edits(source.five_utr, current.five_utr)
        cur_score = step_best_score
        if cur_score > best_score:
            best_score = cur_score
            best = (current, list(edits))
        if oracle.remaining <= 0:
            break

    return _finalize("greedy", source, best, oracle, t0,
                     query_budget, edit_budget, n_root, cfg)


# ===========================================================================
# Ranker model (Stage B ranker + DAgger ranker share this architecture)
# ===========================================================================

class LinearDeltaRanker:
    """Ridge-regression ranker over (source, candidate) feature pairs.

    Scores a candidate cheaply WITHOUT oracle calls. Trained on
    (source_seq, candidate_seq, target) triples; prediction target is the
    oracle reward scalar (LCB + edit cost) or raw delta.
    """

    def __init__(self, ridge: float = 1.0, max_seq_len: int = 100):
        self.ridge = float(ridge)
        self.max_seq_len = max_seq_len
        self._w: Optional[np.ndarray] = None

    @staticmethod
    def featurize(source_seq: str, candidate_seq: str, max_seq_len: int = 100) -> np.ndarray:
        from core.p3_02_delta_oracle import extract_features

        edits = _diff_edits(source_seq, candidate_seq)
        feats = extract_features(source_seq, candidate_seq, edits, max_seq_len)
        return np.concatenate([
            feats["source_feat"], feats["diff_feat"], feats["edit_feat"],
        ])

    def fit(
        self,
        source_seqs: Sequence[str],
        candidate_seqs: Sequence[str],
        targets: Sequence[float],
    ) -> "LinearDeltaRanker":
        X = np.stack([
            self.featurize(s, c, self.max_seq_len)
            for s, c in zip(source_seqs, candidate_seqs)
        ])
        y = np.asarray(targets, dtype=np.float64)
        n, d = X.shape
        A = X.T @ X + self.ridge * np.eye(d)
        b = X.T @ y
        self._w = np.linalg.solve(A, b)
        return self

    def score(self, source_seq: str, candidate_seq: str) -> float:
        if self._w is None:
            raise RuntimeError("ranker not fitted")
        x = self.featurize(source_seq, candidate_seq, self.max_seq_len)
        return float(x @ self._w)


def _ranker_decode(
    source: MRNARecord,
    ranker: LinearDeltaRanker,
    edit_budget: int,
    regions: Sequence[str],
) -> Tuple[MRNARecord, List[Dict[str, Any]]]:
    """Greedy decoding by ranker score only (0 oracle guidance calls)."""
    current = source
    visited = {_seq_hash(source)}
    cur_rank = ranker.score(source.five_utr, source.five_utr)
    for _step in range(edit_budget):
        acts = [a for a in legal_actions(current, visited, regions) if not a.is_stop()]
        best_rec = None
        best_rank = cur_rank
        for a in acts:
            rec = apply_edit_action(current, a)
            r = ranker.score(source.five_utr, rec.five_utr)
            if r > best_rank:
                best_rank = r
                best_rec = rec
        if best_rec is None:
            break
        current = best_rec
        visited.add(_seq_hash(current))
    edits = _diff_edits(source.five_utr, current.five_utr)
    return current, edits


# ===========================================================================
# Baseline 4: Stage B ranker (0 guidance oracle calls)
# ===========================================================================

def stage_b_ranker_search(
    source: MRNARecord,
    oracle: CountingOracle,
    *,
    ranker: LinearDeltaRanker,
    query_budget: int,
    edit_budget: int,
    seed: int = 0,
    regions: Sequence[str] = ("five_utr",),
    cfg: Optional[RewardV3Config] = None,
) -> SearchResult:
    """Ranker-guided greedy decode; oracle used only for final verification."""
    t0 = time.perf_counter()
    cfg = cfg or RewardV3Config(context="protein_output_focused")
    oracle.query_budget = query_budget  # untouched: 0 guidance calls

    n_root = len(legal_actions(source, regions=regions))
    cand, edits = _ranker_decode(source, ranker, edit_budget, regions)
    return _finalize("stage_b_ranker", source, (cand, edits), oracle, t0,
                     query_budget, edit_budget, n_root, cfg,
                     provenance={"ranker": "LinearDeltaRanker", "guidance_calls": 0})


# ===========================================================================
# Baseline 5: beam search
# ===========================================================================

def beam_search(
    source: MRNARecord,
    oracle: CountingOracle,
    *,
    query_budget: int,
    edit_budget: int,
    beam_width: int = 8,
    seed: int = 0,
    regions: Sequence[str] = ("five_utr",),
    cfg: Optional[RewardV3Config] = None,
) -> SearchResult:
    """Beam search over edit sequences, scored by oracle reward."""
    t0 = time.perf_counter()
    cfg = cfg or RewardV3Config(context="protein_output_focused")
    oracle.query_budget = query_budget

    n_root = len(legal_actions(source, regions=regions))
    src_score = score_candidate(source, source, oracle, 0, cfg)["scalar"]
    beam: List[Tuple[MRNARecord, List[Dict[str, Any]], float, set]] = [
        (source, [], src_score, {_seq_hash(source)})
    ]
    best = (source, [])
    best_score = src_score

    for _step in range(edit_budget):
        candidates: List[Tuple[float, MRNARecord, List[Dict[str, Any]], set]] = []
        exhausted = False
        for rec, edits, _s, visited in beam:
            if len(edits) >= edit_budget:
                candidates.append((_s, rec, edits, visited))
                continue
            acts = [a for a in legal_actions(rec, visited, regions) if not a.is_stop()]
            for a in acts:
                child = apply_edit_action(rec, a)
                child_edits = _diff_edits(source.five_utr, child.five_utr)
                try:
                    sc = score_candidate(source, child, oracle, len(child_edits), cfg)
                except BudgetExhausted:
                    exhausted = True
                    break
                candidates.append((sc["scalar"], child, child_edits, visited | {_seq_hash(child)}))
                if sc["scalar"] > best_score:
                    best_score = sc["scalar"]
                    best = (child, child_edits)
            if exhausted:
                break
        if not candidates:
            break
        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = [(rec, ed, sc, vis) for sc, rec, ed, vis in candidates[:beam_width]]
        if exhausted or oracle.remaining <= 0:
            break

    return _finalize("beam_search", source, best, oracle, t0,
                     query_budget, edit_budget, n_root, cfg,
                     provenance={"beam_width": beam_width})


# ===========================================================================
# Baseline 6: simulated annealing
# ===========================================================================

def simulated_annealing(
    source: MRNARecord,
    oracle: CountingOracle,
    *,
    query_budget: int,
    edit_budget: int,
    seed: int = 0,
    t_init: float = 0.5,
    cooling: float = 0.995,
    regions: Sequence[str] = ("five_utr",),
    cfg: Optional[RewardV3Config] = None,
) -> SearchResult:
    """Metropolis-Hastings over candidates with <= edit_budget edits.

    Neighbor move: if current has < edit_budget edits, apply a random legal
    action; if at budget, revert one random edit then apply a random action.
    """
    t0 = time.perf_counter()
    cfg = cfg or RewardV3Config(context="protein_output_focused")
    rng = random.Random(seed)
    oracle.query_budget = query_budget

    n_root = len(legal_actions(source, regions=regions))
    current = source
    cur_edits: List[Dict[str, Any]] = []
    cur_score = score_candidate(source, source, oracle, 0, cfg)["scalar"]
    best = (source, [])
    best_score = cur_score
    temp = t_init

    while oracle.remaining > 0:
        n_cur = len(cur_edits)
        if n_cur < edit_budget:
            acts = [a for a in legal_actions(current, regions=regions) if not a.is_stop()]
            if not acts:
                break
            proposal = apply_edit_action(current, rng.choice(acts))
        else:
            # revert to source then re-apply n_cur-1 random kept edits via random walk
            acts = [a for a in legal_actions(source, regions=regions) if not a.is_stop()]
            if not acts:
                break
            proposal = apply_edit_action(source, rng.choice(acts))
        prop_edits = _diff_edits(source.five_utr, proposal.five_utr)
        try:
            sc = score_candidate(source, proposal, oracle, len(prop_edits), cfg)
        except BudgetExhausted:
            break
        prop_score = sc["scalar"]
        d = prop_score - cur_score
        if d >= 0 or rng.random() < math.exp(d / max(temp, 1e-8)):
            current, cur_edits, cur_score = proposal, prop_edits, prop_score
        # Track global best by score, independent of acceptance.
        if prop_score > best_score:
            best_score = prop_score
            best = (proposal, prop_edits)
        temp *= cooling

    return _finalize("simulated_annealing", source, best, oracle, t0,
                     query_budget, edit_budget, n_root, cfg,
                     provenance={"t_init": t_init, "cooling": cooling})


# ===========================================================================
# Baseline 7: MCTS
# ===========================================================================

class _MCTSNode:
    __slots__ = ("record", "edits", "parent", "children", "visits", "value_sum", "untried")

    def __init__(self, record, edits, parent, untried):
        self.record = record
        self.edits = edits
        self.parent = parent
        self.children: List[_MCTSNode] = []
        self.visits = 0
        self.value_sum = 0.0
        self.untried = untried

    def ucb1(self, c: float = 1.414) -> float:
        if self.visits == 0:
            return math.inf
        return self.value_sum / self.visits + c * math.sqrt(
            math.log(max(self.parent.visits, 1)) / self.visits
        )


def mcts_search(
    source: MRNARecord,
    oracle: CountingOracle,
    *,
    query_budget: int,
    edit_budget: int,
    seed: int = 0,
    n_rollout_edits: Optional[int] = None,
    regions: Sequence[str] = ("five_utr",),
    cfg: Optional[RewardV3Config] = None,
) -> SearchResult:
    """UCT MCTS; each simulation's leaf evaluation costs 1 oracle call."""
    t0 = time.perf_counter()
    cfg = cfg or RewardV3Config(context="protein_output_focused")
    rng = random.Random(seed)
    oracle.query_budget = query_budget

    root_actions = [a for a in legal_actions(source, regions=regions) if not a.is_stop()]
    n_root = len(root_actions) + 1
    root = _MCTSNode(source, [], None, list(root_actions))

    best = (source, [])
    best_score = score_candidate(source, source, oracle, 0, cfg)["scalar"]

    def _rollout_value(rec: MRNARecord, n_done: int) -> Tuple[float, MRNARecord, List[Dict[str, Any]]]:
        cur = rec
        visited = {_seq_hash(rec)}
        steps = (n_rollout_edits or edit_budget) - n_done
        for _ in range(max(0, steps)):
            acts = [a for a in legal_actions(cur, visited, regions) if not a.is_stop()]
            if not acts:
                break
            cur = apply_edit_action(cur, rng.choice(acts))
            visited.add(_seq_hash(cur))
        edits = _diff_edits(source.five_utr, cur.five_utr)
        sc = score_candidate(source, cur, oracle, len(edits), cfg)
        return sc["scalar"], cur, edits

    while oracle.remaining > 0:
        # Selection
        node = root
        while node.untried == [] and node.children:
            node = max(node.children, key=lambda n: n.ucb1())
        # Expansion
        if node.untried and len(node.edits) < edit_budget:
            a = node.untried.pop(rng.randrange(len(node.untried)))
            child_rec = apply_edit_action(node.record, a)
            child_edits = _diff_edits(source.five_utr, child_rec.five_utr)
            visited_child = [x for x in legal_actions(child_rec, regions=regions) if not x.is_stop()]
            child = _MCTSNode(child_rec, child_edits, node, visited_child)
            node.children.append(child)
            node = child
        # Simulation
        try:
            value, term_rec, term_edits = _rollout_value(node.record, len(node.edits))
        except BudgetExhausted:
            break
        # Backprop
        n: Optional[_MCTSNode] = node
        while n is not None:
            n.visits += 1
            n.value_sum += value
            n = n.parent
        if value > best_score:
            best_score = value
            best = (term_rec, term_edits)

    return _finalize("mcts", source, best, oracle, t0,
                     query_budget, edit_budget, n_root, cfg)


# ===========================================================================
# Baseline 8: oracle-guided local search (sampled neighborhood)
# ===========================================================================

def oracle_guided_local_search(
    source: MRNARecord,
    oracle: CountingOracle,
    *,
    query_budget: int,
    edit_budget: int,
    neighborhood: int = 16,
    seed: int = 0,
    regions: Sequence[str] = ("five_utr",),
    cfg: Optional[RewardV3Config] = None,
) -> SearchResult:
    """First-improvement local search over a sampled neighborhood (cheaper
    per step than greedy's full enumeration); random perturbation on
    stagnation."""
    t0 = time.perf_counter()
    cfg = cfg or RewardV3Config(context="protein_output_focused")
    rng = random.Random(seed)
    oracle.query_budget = query_budget

    n_root = len(legal_actions(source, regions=regions))
    current = source
    visited = {_seq_hash(source)}
    cur_score = score_candidate(source, source, oracle, 0, cfg)["scalar"]
    best = (source, [])
    best_score = cur_score

    while oracle.remaining > 0 and len(_diff_edits(source.five_utr, current.five_utr)) < edit_budget:
        acts = [a for a in legal_actions(current, visited, regions) if not a.is_stop()]
        if not acts:
            break
        rng.shuffle(acts)
        improved = False
        for a in acts[:neighborhood]:
            child = apply_edit_action(current, a)
            edits = _diff_edits(source.five_utr, child.five_utr)
            try:
                sc = score_candidate(source, child, oracle, len(edits), cfg)
            except BudgetExhausted:
                improved = False
                break
            if sc["scalar"] > best_score:
                best_score = sc["scalar"]
                best = (child, edits)
            if sc["scalar"] > cur_score:
                current = child
                visited.add(_seq_hash(child))
                cur_score = sc["scalar"]
                improved = True
                break
        if not improved:
            break  # local optimum within sampled neighborhood

    return _finalize("oracle_guided_local_search", source, best, oracle, t0,
                     query_budget, edit_budget, n_root, cfg,
                     provenance={"neighborhood": neighborhood})


# ===========================================================================
# DAgger ranker training (shared by baselines 9 & 10)
# ===========================================================================

def train_dagger_ranker(
    train_sources: Sequence[MRNARecord],
    oracle: CountingOracle,
    *,
    n_rounds: int = 2,
    edits_per_round: int = 2,
    max_actions_per_state: int = 64,
    seed: int = 0,
    regions: Sequence[str] = ("five_utr",),
    cfg: Optional[RewardV3Config] = None,
    initial_pairs: Optional[Tuple[List[str], List[str], List[float]]] = None,
    training_query_budget: Optional[int] = None,
) -> Dict[str, Any]:
    """DAgger: roll out the ranker policy, label visited states' children
    with the oracle, aggregate, retrain.

    Oracle calls here are TRAINING queries (counted separately from the
    per-source inference search budget).

    Returns {"ranker": LinearDeltaRanker, "training_oracle_calls": int,
             "rounds": n_rounds, "n_pairs": int}.
    """
    cfg = cfg or RewardV3Config(context="protein_output_focused")
    rng = random.Random(seed)

    src_seqs: List[str] = []
    cand_seqs: List[str] = []
    targets: List[float] = []
    if initial_pairs is not None:
        s0, c0, t0_ = initial_pairs
        src_seqs.extend(s0)
        cand_seqs.extend(c0)
        targets.extend(t0_)

    ranker = LinearDeltaRanker()
    if src_seqs:
        ranker.fit(src_seqs, cand_seqs, targets)
    else:
        # Cold start: identity + small random perturbations labelled by oracle
        for src in train_sources:
            targets.append(0.0)
            src_seqs.append(src.five_utr)
            cand_seqs.append(src.five_utr)
        ranker.fit(src_seqs, cand_seqs, targets)

    calls_before = oracle.search_calls
    if training_query_budget is not None:
        oracle.query_budget = training_query_budget

    for _round in range(n_rounds):
        for src in train_sources:
            if oracle.remaining <= 0:
                break
            rec = src
            visited = {_seq_hash(src)}
            for _step in range(edits_per_round):
                acts = [a for a in legal_actions(rec, visited, regions) if not a.is_stop()]
                if not acts:
                    break
                # Policy rollout: follow ranker argmax
                scored = [(ranker.score(src.five_utr, apply_edit_action(rec, a).five_utr), a)
                          for a in acts]
                scored.sort(key=lambda x: x[0], reverse=True)
                policy_action = scored[0][1]
                # Expert labelling at visited state: oracle-score a capped
                # subset of children (random if too many).
                label_acts = acts
                if len(label_acts) > max_actions_per_state:
                    label_acts = rng.sample(label_acts, max_actions_per_state)
                for a in label_acts:
                    if oracle.remaining <= 0:
                        break
                    child = apply_edit_action(rec, a)
                    edits = _diff_edits(src.five_utr, child.five_utr)
                    try:
                        sc = score_candidate(src, child, oracle, len(edits), cfg)
                    except BudgetExhausted:
                        break
                    src_seqs.append(src.five_utr)
                    cand_seqs.append(child.five_utr)
                    targets.append(sc["scalar"])
                rec = apply_edit_action(rec, policy_action)
                visited.add(_seq_hash(rec))
        ranker.fit(src_seqs, cand_seqs, targets)

    calls = oracle.search_calls - calls_before
    return {
        "ranker": ranker,
        "training_oracle_calls": calls,
        "rounds": n_rounds,
        "n_pairs": len(targets),
    }


# ===========================================================================
# Baseline 9: DAgger ranker (0 guidance calls at inference)
# ===========================================================================

def dagger_ranker_search(
    source: MRNARecord,
    oracle: CountingOracle,
    *,
    ranker: LinearDeltaRanker,
    query_budget: int,
    edit_budget: int,
    seed: int = 0,
    regions: Sequence[str] = ("five_utr",),
    cfg: Optional[RewardV3Config] = None,
) -> SearchResult:
    t0 = time.perf_counter()
    cfg = cfg or RewardV3Config(context="protein_output_focused")
    oracle.query_budget = query_budget

    n_root = len(legal_actions(source, regions=regions))
    cand, edits = _ranker_decode(source, ranker, edit_budget, regions)
    return _finalize("dagger_ranker", source, (cand, edits), oracle, t0,
                     query_budget, edit_budget, n_root, cfg,
                     provenance={"ranker": "DAgger-LinearDeltaRanker", "guidance_calls": 0})


# ===========================================================================
# Baseline 10: DAgger ranker + limited search
# ===========================================================================

def dagger_plus_limited_search(
    source: MRNARecord,
    oracle: CountingOracle,
    *,
    ranker: LinearDeltaRanker,
    query_budget: int,
    edit_budget: int,
    top_m_actions: int = 4,
    beam_width: int = 4,
    seed: int = 0,
    regions: Sequence[str] = ("five_utr",),
    cfg: Optional[RewardV3Config] = None,
) -> SearchResult:
    """Ranker prunes each state's action set to top-m; oracle-scored beam
    search of width w runs within the pruned space under the query budget."""
    t0 = time.perf_counter()
    cfg = cfg or RewardV3Config(context="protein_output_focused")
    oracle.query_budget = query_budget

    n_root = len(legal_actions(source, regions=regions))
    src_score = score_candidate(source, source, oracle, 0, cfg)["scalar"]
    beam: List[Tuple[MRNARecord, float, set]] = [(source, src_score, {_seq_hash(source)})]
    best = (source, [])
    best_score = src_score

    for _step in range(edit_budget):
        cand_list: List[Tuple[float, MRNARecord, set]] = []
        exhausted = False
        for rec, _s, visited in beam:
            n_done = len(_diff_edits(source.five_utr, rec.five_utr))
            if n_done >= edit_budget:
                cand_list.append((_s, rec, visited))
                continue
            acts = [a for a in legal_actions(rec, visited, regions) if not a.is_stop()]
            if not acts:
                cand_list.append((_s, rec, visited))
                continue
            ranked = sorted(
                acts,
                key=lambda a: ranker.score(source.five_utr, apply_edit_action(rec, a).five_utr),
                reverse=True,
            )[:top_m_actions]
            for a in ranked:
                child = apply_edit_action(rec, a)
                edits = _diff_edits(source.five_utr, child.five_utr)
                try:
                    sc = score_candidate(source, child, oracle, len(edits), cfg)
                except BudgetExhausted:
                    exhausted = True
                    break
                cand_list.append((sc["scalar"], child, visited | {_seq_hash(child)}))
                if sc["scalar"] > best_score:
                    best_score = sc["scalar"]
                    best = (child, edits)
            if exhausted:
                break
        if not cand_list:
            break
        cand_list.sort(key=lambda x: x[0], reverse=True)
        beam = [(rec, sc, vis) for sc, rec, vis in cand_list[:beam_width]]
        if exhausted or oracle.remaining <= 0:
            break

    return _finalize("dagger_ranker_plus_limited_search", source, best, oracle, t0,
                     query_budget, edit_budget, n_root, cfg,
                     provenance={"top_m_actions": top_m_actions, "beam_width": beam_width})


# ===========================================================================
# Exact evaluation
# ===========================================================================

def exact_one_edit_optimum(
    source: MRNARecord,
    oracle: CountingOracle,
    *,
    regions: Sequence[str] = ("five_utr",),
    cfg: Optional[RewardV3Config] = None,
) -> Dict[str, Any]:
    """Exact optimum over all <=1-edit candidates (source included)."""
    cfg = cfg or RewardV3Config(context="protein_output_focused")
    oracle.query_budget = None  # exact evaluation is unbudgeted
    best_rec = source
    best_edits: List[Dict[str, Any]] = []
    source_sc = score_candidate(source, source, oracle, 0, cfg)
    best_sc = source_sc
    n = 1
    for a in legal_actions(source, regions=regions):
        if a.is_stop():
            continue
        child = apply_edit_action(source, a)
        edits = _diff_edits(source.five_utr, child.five_utr)
        sc = score_candidate(source, child, oracle, 1, cfg)
        n += 1
        if sc["scalar"] > best_sc["scalar"]:
            best_sc, best_rec, best_edits = sc, child, edits
    return {
        "optimum_score": best_sc["scalar"],
        "source_score": source_sc["scalar"],
        "improvement": best_sc["scalar"] - source_sc["scalar"],
        "optimum_mean_delta": best_sc["mean_delta"],
        "best_candidate": best_rec,
        "best_edits": best_edits,
        "n_evaluated": n,
        "search_oracle_calls": oracle.search_calls,
    }


def exact_two_edit_optimum(
    source: MRNARecord,
    oracle: CountingOracle,
    *,
    regions: Sequence[str] = ("five_utr",),
    cfg: Optional[RewardV3Config] = None,
    max_enumeration: int = 200000,
) -> Dict[str, Any]:
    """Exact optimum over all <=2-edit candidates. Raises if the reachable
    set exceeds ``max_enumeration`` (caller should restrict to small sources)."""
    cfg = cfg or RewardV3Config(context="protein_output_focused")
    oracle.query_budget = None
    one = exact_one_edit_optimum(source, oracle, regions=regions, cfg=cfg)
    best_sc_scalar = one["optimum_score"]
    best_rec = one["best_candidate"]
    best_edits = one["best_edits"]
    n = one["n_evaluated"]

    first_actions = [a for a in legal_actions(source, regions=regions) if not a.is_stop()]
    n_est = len(first_actions) ** 2
    if n_est > max_enumeration:
        raise ValueError(
            f"two-edit enumeration too large (~{n_est}); restrict source size"
        )
    for a1 in first_actions:
        r1 = apply_edit_action(source, a1)
        visited = {_seq_hash(source), _seq_hash(r1)}
        for a2 in legal_actions(r1, visited, regions):
            if a2.is_stop():
                continue
            r2 = apply_edit_action(r1, a2)
            edits = _diff_edits(source.five_utr, r2.five_utr)
            sc = score_candidate(source, r2, oracle, 2, cfg)
            n += 1
            if sc["scalar"] > best_sc_scalar:
                best_sc_scalar, best_rec, best_edits = sc["scalar"], r2, edits
    return {
        "optimum_score": best_sc_scalar,
        "best_candidate": best_rec,
        "best_edits": best_edits,
        "n_evaluated": n,
        "search_oracle_calls": oracle.search_calls,
    }


# ===========================================================================
# Tiny-MDP dynamic programming (finite-horizon optimal policy)
# ===========================================================================

def enumerate_states(
    source: MRNARecord,
    edit_budget: int,
    regions: Sequence[str] = ("five_utr",),
    max_states: int = 100000,
) -> Dict[str, MRNARecord]:
    """BFS-enumerate all states reachable within edit_budget edits."""
    states: Dict[str, MRNARecord] = {_seq_hash(source): source}
    frontier = [source]
    for _depth in range(edit_budget):
        nxt = []
        for rec in frontier:
            visited = {_seq_hash(rec)}
            for a in legal_actions(rec, visited, regions):
                if a.is_stop():
                    continue
                child = apply_edit_action(rec, a)
                h = _seq_hash(child)
                if h not in states:
                    states[h] = child
                    nxt.append(child)
                    if len(states) > max_states:
                        raise ValueError("state space too large for exact DP")
        frontier = nxt
    return states


def tiny_mdp_value_iteration(
    source: MRNARecord,
    oracle: CountingOracle,
    *,
    edit_budget: int,
    regions: Sequence[str] = ("five_utr",),
    cfg: Optional[RewardV3Config] = None,
    max_states: int = 100000,
) -> Dict[str, Any]:
    """Exact finite-horizon optimal value/policy by backward induction.

    State = candidate sequence; horizon = remaining edit steps. Revisits are
    allowed in this reference DP (horizon still bounds recursion), so the
    value is an UPPER BOUND on the no-revisit P3-06 MDP optimum and hence a
    valid ceiling for regret computation:

        V_0(s)   = R(s)
        V_h(s)   = max( R(s) [STOP], max_a V_{h-1}(child(s,a)) )
        optimal  = V_B(source)

    R(s) is the Reward-v3 scalar of the state's candidate.

    Returns V table (per remaining-horizon), optimal policy map
    (state_hash -> best child hash or None=STOP at each horizon), the optimal
    trajectory from source, and all enumerated states.
    """
    cfg = cfg or RewardV3Config(context="protein_output_focused")
    oracle.query_budget = None
    states = enumerate_states(source, edit_budget, regions, max_states)

    # Terminal values R(s) and children map
    R: Dict[str, float] = {}
    children: Dict[str, List[Tuple[str, EditAction]]] = {}
    for h, rec in states.items():
        edits = _diff_edits(source.five_utr, rec.five_utr)
        sc = score_candidate(source, rec, oracle, len(edits), cfg)
        R[h] = sc["scalar"]
        ch: List[Tuple[str, EditAction]] = []
        for a in legal_actions(rec, regions=regions):
            if a.is_stop():
                continue
            child = apply_edit_action(rec, a)
            chh = _seq_hash(child)
            if chh in states:
                ch.append((chh, a))
        children[h] = ch

    # Backward induction over remaining horizon
    V: List[Dict[str, float]] = [{h: R[h] for h in states}]  # V[0]
    pi: List[Dict[str, Optional[str]]] = [{h: None for h in states}]
    for rem in range(1, edit_budget + 1):
        v_prev = V[rem - 1]
        v_cur: Dict[str, float] = {}
        p_cur: Dict[str, Optional[str]] = {}
        for h in states:
            best_v = R[h]
            best_next: Optional[str] = None
            for chh, _a in children[h]:
                if v_prev[chh] > best_v:
                    best_v = v_prev[chh]
                    best_next = chh
            v_cur[h] = best_v
            p_cur[h] = best_next
        V.append(v_cur)
        pi.append(p_cur)

    # Unroll optimal trajectory from source with full budget
    traj: List[str] = []
    h = _seq_hash(source)
    rem = edit_budget
    traj.append(h)
    while rem > 0 and pi[rem][h] is not None:
        h = pi[rem][h]  # type: ignore[index]
        traj.append(h)
        rem -= 1
    opt_rec = states[traj[-1]]
    opt_edits = _diff_edits(source.five_utr, opt_rec.five_utr)

    return {
        "n_states": len(states),
        "V": V,                      # V[rem][state_hash]
        "R": R,
        "policy": pi,                # pi[rem][state_hash] -> child hash | None
        "children": children,
        "optimal_value": V[edit_budget][_seq_hash(source)],
        "optimal_trajectory": traj,
        "optimal_candidate": opt_rec,
        "optimal_edits": opt_edits,
        "search_oracle_calls": oracle.search_calls,
        "states": states,
    }


# ===========================================================================
# Regret computation
# ===========================================================================

def compute_regrets(
    results: Sequence[SearchResult],
    exact_optimum_score: float,
) -> Dict[str, float]:
    """Regret of each method = exact_optimum - method best score (>=0)."""
    out = {}
    for r in results:
        out[f"{r.method}_regret"] = float(exact_optimum_score - r.best_score)
    return out


# ===========================================================================
# Algorithm semantics on a tiny enumerable state space
# ===========================================================================
#
# Compared semantics (spec: do NOT presume greedy == true-flow marginal):
#   1. greedy_intensity          — deterministic argmax over CTMC intensities
#   2. stochastic_ctmc           — sample from normalized CTMC intensities
#   3. finite_horizon_optimal    — backward-induction optimal policy (DP)
#   4. exact_terminal_marginal   — exact terminal-state distribution induced
#                                  by the stochastic CTMC policy (forward DP)
#   5. beam_search               — deterministic width-w beam on R(child)
#
# Reported: action KL, terminal KL, argmax agreement, expected return,
# constraint validity.

def _softmax_dict(scores: Dict[str, float], beta: float) -> Dict[str, float]:
    if not scores:
        return {}
    m = max(scores.values())
    exps = {k: math.exp(beta * (v - m)) for k, v in scores.items()}
    z = sum(exps.values())
    return {k: v / z for k, v in exps.items()}


def _kl(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-12) -> float:
    """KL(p || q) with support smoothing."""
    keys = set(p) | set(q)
    return sum(
        p.get(k, eps) * math.log(max(p.get(k, eps), eps) / max(q.get(k, eps), eps))
        for k in keys
        if p.get(k, eps) > eps
    )


class CTMCIntensityModel:
    """CTMC intensity model over the tiny MDP.

    Intensity of edit action a at state s:  r(a|s) = exp(beta * (R(child) - R(s)))
    Intensity of STOP:                      r(stop|s) = exp(beta * 0) = 1
    (STOP keeps the current value; improvements are rate-amplified.)

    The normalized intensities define the stochastic CTMC policy; the argmax
    defines the greedy-intensity policy.
    """

    def __init__(self, beta: float = 4.0):
        self.beta = float(beta)

    def action_probs(
        self,
        state_hash: str,
        R: Dict[str, float],
        children: Dict[str, List[Tuple[str, EditAction]]],
    ) -> Dict[str, float]:
        """Normalized action distribution; key 'STOP' or child hash."""
        scores: Dict[str, float] = {"STOP": 0.0}
        for chh, _a in children.get(state_hash, []):
            scores[chh] = R[chh] - R[state_hash]
        return _softmax_dict(scores, self.beta)

    def greedy_action(
        self,
        state_hash: str,
        R: Dict[str, float],
        children: Dict[str, List[Tuple[str, EditAction]]],
    ) -> str:
        probs = self.action_probs(state_hash, R, children)
        return max(probs.items(), key=lambda kv: kv[1])[0]


def exact_terminal_distribution(
    source: MRNARecord,
    states: Dict[str, MRNARecord],
    children: Dict[str, List[Tuple[str, EditAction]]],
    policy_probs: Callable[[str], Dict[str, float]],
    edit_budget: int,
) -> Dict[str, float]:
    """Exact terminal-state distribution under a stochastic policy.

    Forward DP over (steps_taken, state). Trajectory ends when STOP is drawn
    or the edit budget is exhausted. Cycles are handled by horizon indexing.
    """
    src_h = _seq_hash(source)
    reach: List[Dict[str, float]] = [{src_h: 1.0}]
    term: Dict[str, float] = {}
    for step in range(edit_budget + 1):
        if step == edit_budget:
            for h, q in reach[step].items():
                term[h] = term.get(h, 0.0) + q
            break
        nxt: Dict[str, float] = {}
        for h, q in reach[step].items():
            probs = policy_probs(h)
            p_stop = probs.get("STOP", 0.0)
            if p_stop > 0:
                term[h] = term.get(h, 0.0) + q * p_stop
            for key, p in probs.items():
                if key == "STOP" or p == 0.0:
                    continue
                nxt[key] = nxt.get(key, 0.0) + q * p
        reach.append(nxt)
    # normalize (should already sum to 1)
    z = sum(term.values())
    if z > 0:
        term = {k: v / z for k, v in term.items()}
    return term


def deterministic_terminal_distribution(
    source: MRNARecord,
    children: Dict[str, List[Tuple[str, EditAction]]],
    greedy_action: Callable[[str], str],
    edit_budget: int,
) -> Dict[str, float]:
    """Terminal state (delta distribution) of a deterministic argmax policy."""
    h = _seq_hash(source)
    for _step in range(edit_budget):
        a = greedy_action(h)
        if a == "STOP":
            break
        h = a
    return {h: 1.0}


def expected_return(term_dist: Dict[str, float], R: Dict[str, float]) -> float:
    return sum(p * R[h] for h, p in term_dist.items())


def compare_algorithm_semantics(
    source: MRNARecord,
    oracle: CountingOracle,
    *,
    edit_budget: int = 2,
    beam_width: int = 2,
    beta: float = 4.0,
    regions: Sequence[str] = ("five_utr",),
    cfg: Optional[RewardV3Config] = None,
    max_states: int = 50000,
) -> Dict[str, Any]:
    """Run the five algorithm semantics on the tiny enumerable MDP and
    compute the comparison metrics.

    Returns a dict with per-semantics terminal distributions, expected
    returns, and pairwise action-KL / terminal-KL / argmax-agreement.
    """
    cfg = cfg or RewardV3Config(context="protein_output_focused")
    oracle.query_budget = None

    dp = tiny_mdp_value_iteration(
        source, oracle,
        edit_budget=edit_budget, regions=regions, cfg=cfg, max_states=max_states,
    )
    states: Dict[str, MRNARecord] = dp["states"]
    R: Dict[str, float] = dp["R"]
    children: Dict[str, List[Tuple[str, EditAction]]] = dp["children"]
    src_h = _seq_hash(source)

    ctmc = CTMCIntensityModel(beta=beta)

    # --- 1-2. greedy intensity & stochastic CTMC share the intensity model
    ctmc_probs = lambda h: ctmc.action_probs(h, R, children)  # noqa: E731
    term_ctmc = exact_terminal_distribution(source, states, children, ctmc_probs, edit_budget)
    term_greedy_int = deterministic_terminal_distribution(
        source, children, lambda h: ctmc.greedy_action(h, R, children), edit_budget
    )

    # --- 3. finite-horizon optimal (DP policy)
    policy = dp["policy"]

    def opt_action(h: str, rem: int) -> str:
        nxt = policy[rem][h]
        return "STOP" if nxt is None else nxt

    # deterministic optimal terminal
    h = src_h
    rem = edit_budget
    while rem > 0 and opt_action(h, rem) != "STOP":
        h = opt_action(h, rem)
        rem -= 1
    term_optimal = {h: 1.0}

    # --- 4. exact terminal marginal == term_ctmc (exact, not sampled)
    term_marginal = term_ctmc

    # --- 5. beam search (deterministic, ranked by R(child))
    beam = [src_h]
    best_beam_h = src_h
    for _step in range(edit_budget):
        cands: List[Tuple[float, str]] = []
        for hb in beam:
            for chh, _a in children.get(hb, []):
                cands.append((R[chh], chh))
        if not cands:
            break
        cands.sort(key=lambda x: x[0], reverse=True)
        beam = []
        seen = set()
        for _sc, chh in cands:
            if chh not in seen:
                beam.append(chh)
                seen.add(chh)
            if len(beam) >= beam_width:
                break
        if beam and R[beam[0]] > R[best_beam_h]:
            best_beam_h = beam[0]
    term_beam = {best_beam_h: 1.0}

    # --- action KL: mean over states reachable from source of
    #     KL(pi_optimal_delta || pi_ctmc) and KL(greedy_delta || pi_ctmc)
    action_kl_opt_vs_ctmc: Dict[str, float] = {}
    action_kl_greedy_vs_ctmc: Dict[str, float] = {}
    argmax_agree_opt_ctmc = 0
    argmax_agree_greedy_ctmc = 0
    n_states = 0
    for h in states:
        p_ctmc = ctmc_probs(h)
        if not p_ctmc:
            continue
        # optimal one-step (use remaining horizon = edit_budget as an
        # approximation at shallow states; exact would be horizon-indexed)
        opt_a = opt_action(h, edit_budget)
        greedy_a = ctmc.greedy_action(h, R, children)
        p_opt = {opt_a: 1.0}
        p_greedy = {greedy_a: 1.0}
        action_kl_opt_vs_ctmc[h] = _kl(p_opt, p_ctmc)
        action_kl_greedy_vs_ctmc[h] = _kl(p_greedy, p_ctmc)
        ctmc_argmax = max(p_ctmc.items(), key=lambda kv: kv[1])[0]
        argmax_agree_opt_ctmc += int(ctmc_argmax == opt_a)
        argmax_agree_greedy_ctmc += int(ctmc_argmax == greedy_a)
        n_states += 1

    mean_action_kl_opt = float(np.mean(list(action_kl_opt_vs_ctmc.values()))) if action_kl_opt_vs_ctmc else 0.0
    mean_action_kl_greedy = float(np.mean(list(action_kl_greedy_vs_ctmc.values()))) if action_kl_greedy_vs_ctmc else 0.0

    # --- terminal KL between semantics
    terminal_kls = {
        "greedy_intensity_vs_ctmc": _kl(term_greedy_int, term_ctmc),
        "optimal_vs_ctmc": _kl(term_optimal, term_ctmc),
        "beam_vs_ctmc": _kl(term_beam, term_ctmc),
        "greedy_intensity_vs_optimal": _kl(term_greedy_int, term_optimal),
        "beam_vs_optimal": _kl(term_beam, term_optimal),
        "marginal_vs_ctmc_sampled_exact": _kl(term_marginal, term_ctmc),  # == 0
    }

    # --- expected returns
    expected_returns = {
        "greedy_intensity": expected_return(term_greedy_int, R),
        "stochastic_ctmc": expected_return(term_ctmc, R),
        "finite_horizon_optimal": expected_return(term_optimal, R),
        "exact_terminal_marginal": expected_return(term_marginal, R),
        "beam_search": expected_return(term_beam, R),
    }

    # --- constraint validity: every enumerated candidate must satisfy
    #     protein identity + length invariance (by action-space construction)
    n_valid = sum(1 for rec in states.values() if _check_constraints(source, rec))
    constraint_validity = n_valid / max(len(states), 1)

    return {
        "edit_budget": edit_budget,
        "beta": beta,
        "beam_width": beam_width,
        "n_states": len(states),
        "R": R,
        "terminal_distributions": {
            "greedy_intensity": term_greedy_int,
            "stochastic_ctmc": term_ctmc,
            "finite_horizon_optimal": term_optimal,
            "exact_terminal_marginal": term_marginal,
            "beam_search": term_beam,
        },
        "expected_returns": expected_returns,
        "terminal_kl": terminal_kls,
        "action_kl": {
            "optimal_vs_ctmc_mean": mean_action_kl_opt,
            "greedy_vs_ctmc_mean": mean_action_kl_greedy,
        },
        "argmax_agreement": {
            "optimal_vs_ctmc": argmax_agree_opt_ctmc / max(n_states, 1),
            "greedy_vs_ctmc": argmax_agree_greedy_ctmc / max(n_states, 1),
        },
        "constraint_validity": constraint_validity,
        "oracle_calls": oracle.search_calls,
        "optimal_value_dp": dp["optimal_value"],
    }


# ===========================================================================
# Baseline registry
# ===========================================================================

BASELINE_METHODS = (
    "random_legal_editing",
    "best_single_edit",
    "greedy",
    "stage_b_ranker",
    "beam_search",
    "simulated_annealing",
    "mcts",
    "oracle_guided_local_search",
    "dagger_ranker",
    "dagger_ranker_plus_limited_search",
)


def run_all_baselines(
    source: MRNARecord,
    oracle_factory: Callable[[], CountingOracle],
    *,
    query_budget: int,
    edit_budget: int,
    seed: int = 0,
    stage_b_ranker: Optional[LinearDeltaRanker] = None,
    dagger_ranker: Optional[LinearDeltaRanker] = None,
    regions: Sequence[str] = ("five_utr",),
    cfg: Optional[RewardV3Config] = None,
) -> List[SearchResult]:
    """Run all 10 baselines with a fresh CountingOracle per method.

    ``oracle_factory`` must return a new oracle instance wrapping the SAME
    underlying scorer (query budgets are per-method; ranker training queries
    are separate).
    """
    results: List[SearchResult] = []

    results.append(random_legal_editing(
        source, oracle_factory(), query_budget=query_budget,
        edit_budget=edit_budget, seed=seed, regions=regions, cfg=cfg))
    results.append(best_single_edit(
        source, oracle_factory(), query_budget=query_budget,
        edit_budget=1, seed=seed, regions=regions, cfg=cfg))
    results.append(greedy_search(
        source, oracle_factory(), query_budget=query_budget,
        edit_budget=edit_budget, seed=seed, regions=regions, cfg=cfg))
    if stage_b_ranker is not None:
        results.append(stage_b_ranker_search(
            source, oracle_factory(), ranker=stage_b_ranker,
            query_budget=query_budget, edit_budget=edit_budget,
            seed=seed, regions=regions, cfg=cfg))
    results.append(beam_search(
        source, oracle_factory(), query_budget=query_budget,
        edit_budget=edit_budget, seed=seed, regions=regions, cfg=cfg))
    results.append(simulated_annealing(
        source, oracle_factory(), query_budget=query_budget,
        edit_budget=edit_budget, seed=seed, regions=regions, cfg=cfg))
    results.append(mcts_search(
        source, oracle_factory(), query_budget=query_budget,
        edit_budget=edit_budget, seed=seed, regions=regions, cfg=cfg))
    results.append(oracle_guided_local_search(
        source, oracle_factory(), query_budget=query_budget,
        edit_budget=edit_budget, seed=seed, regions=regions, cfg=cfg))
    if dagger_ranker is not None:
        results.append(dagger_ranker_search(
            source, oracle_factory(), ranker=dagger_ranker,
            query_budget=query_budget, edit_budget=edit_budget,
            seed=seed, regions=regions, cfg=cfg))
        results.append(dagger_plus_limited_search(
            source, oracle_factory(), ranker=dagger_ranker,
            query_budget=query_budget, edit_budget=edit_budget,
            seed=seed, regions=regions, cfg=cfg))
    return results
