"""P3-07 acceptance tests: strong-search ceiling and RL necessity gate.

Coverage:
1. All 10 baselines respect query budget (search calls <= budget).
2. All 10 baselines respect edit budget (n_edits <= edit_budget).
3. All proposed candidates satisfy hard constraints (protein identity +
   length invariance).
4. best_single_edit == exact one-edit optimum when budget allows.
5. tiny-MDP DP optimum >= every baseline (valid ceiling) and >= exact
   two-edit optimum is impossible (two-edit <= DP budget-2 upper bound).
6. Regrets are >= 0 for all methods.
7. Algorithm semantics: constraint validity == 1.0, KLs >= 0, argmax
   agreement in [0,1], optimal expected return >= stochastic CTMC, exact
   marginal == CTMC marginal (KL == 0).
8. Ranker + DAgger training run and count training oracle calls.
9. Oracle budget enforcement raises BudgetExhausted.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from core.constants import START_CODON, translate
from core.schema import MRNARecord
from rl.p3_06_mdp import RewardV3Config
from rl.p3_07_search import (
    BudgetExhausted,
    CountingOracle,
    SyntheticDeltaOracle,
    LinearDeltaRanker,
    SearchResult,
    _check_constraints,
    _seq_hash,
    random_legal_editing,
    best_single_edit,
    greedy_search,
    stage_b_ranker_search,
    beam_search,
    simulated_annealing,
    mcts_search,
    oracle_guided_local_search,
    dagger_ranker_search,
    dagger_plus_limited_search,
    train_dagger_ranker,
    exact_one_edit_optimum,
    exact_two_edit_optimum,
    enumerate_states,
    tiny_mdp_value_iteration,
    compute_regrets,
    compare_algorithm_semantics,
    run_all_baselines,
    legal_actions,
)

CFG = RewardV3Config(context="protein_output_focused")


def _make_record(utr: str = "ACGUACGU") -> MRNARecord:
    return MRNARecord(
        transcript_id="test_src",
        five_utr=utr,
        cds=START_CODON + "GCU" * 4 + "UAA",
        three_utr="UGCU",
    )


def _oracle(budget=None, seed=0) -> SyntheticDeltaOracle:
    return SyntheticDeltaOracle(query_budget=budget, uncertainty=0.01, seed=seed)


def _tiny_ranker(source: MRNARecord, seed: int = 0) -> LinearDeltaRanker:
    """Train a ranker on oracle-labelled single-edit children of source."""
    oracle = _oracle(budget=None, seed=seed)
    srcs, cands, tgts = [source.five_utr], [source.five_utr], [0.0]
    for a in legal_actions(source):
        if a.is_stop():
            continue
        from rl.p3_06_mdp import apply_edit_action
        child = apply_edit_action(source, a)
        mean, unc = oracle.score(source, child, purpose="eval")
        srcs.append(source.five_utr)
        cands.append(child.five_utr)
        tgts.append(mean - unc)
    return LinearDeltaRanker().fit(srcs, cands, tgts)


# ---------------------------------------------------------------------------
# 1-3. Baseline budget & constraint compliance
# ---------------------------------------------------------------------------

class TestBaselineCompliance:
    QUERY_BUDGET = 64
    EDIT_BUDGET = 3

    def _all_results(self) -> list:
        source = _make_record()
        ranker = _tiny_ranker(source)
        dag = train_dagger_ranker(
            [source], _oracle(budget=None, seed=1),
            n_rounds=1, edits_per_round=1, max_actions_per_state=12, seed=1,
        )
        results = run_all_baselines(
            source,
            lambda: _oracle(budget=None, seed=2),
            query_budget=self.QUERY_BUDGET,
            edit_budget=self.EDIT_BUDGET,
            seed=0,
            stage_b_ranker=ranker,
            dagger_ranker=dag["ranker"],
        )
        assert len(results) == 10, f"expected 10 baselines, got {len(results)}"
        return results

    def test_query_budget_respected(self):
        for r in self._all_results():
            assert r.search_oracle_calls <= self.QUERY_BUDGET, (
                f"{r.method}: {r.search_oracle_calls} > {self.QUERY_BUDGET}"
            )

    def test_edit_budget_respected(self):
        for r in self._all_results():
            assert len(r.best_edits) <= self.EDIT_BUDGET, (
                f"{r.method}: {len(r.best_edits)} edits > {self.EDIT_BUDGET}"
            )

    def test_hard_constraints(self):
        source = _make_record()
        for r in self._all_results():
            assert r.constraint_valid, f"{r.method}: constraint violated"
            assert translate(source.cds) == translate(r.best_candidate.cds)
            assert len(source.seq) == len(r.best_candidate.seq)

    def test_rankers_use_zero_guidance_calls(self):
        for r in self._all_results():
            if r.method in ("stage_b_ranker", "dagger_ranker"):
                assert r.search_oracle_calls == 0, (
                    f"{r.method} used {r.search_oracle_calls} guidance calls"
                )
                assert r.eval_oracle_calls == 1  # final verification only

    def test_result_schema(self):
        for r in self._all_results():
            d = r.to_dict()
            for key in (
                "method", "source_id", "best_edits", "best_score",
                "search_oracle_calls", "eval_oracle_calls", "wall_clock_sec",
                "query_budget", "edit_budget", "constraint_valid",
            ):
                assert key in d, f"missing key {key} in {r.method} result"


# ---------------------------------------------------------------------------
# 4. best_single_edit == exact one-edit optimum
# ---------------------------------------------------------------------------

class TestExactOneEdit:
    def test_best_single_edit_matches_exact(self):
        source = _make_record()
        n_actions = len([a for a in legal_actions(source) if not a.is_stop()])
        budget = n_actions + 1  # enough for full enumeration
        r = best_single_edit(source, _oracle(budget), query_budget=budget, cfg=CFG)
        exact = exact_one_edit_optimum(source, _oracle(None), cfg=CFG)
        assert abs(r.best_score - exact["optimum_score"]) < 1e-9
        assert exact["n_evaluated"] == n_actions + 1

    def test_best_single_edit_subsamples_when_tight(self):
        source = _make_record()
        budget = 8
        r = best_single_edit(source, _oracle(budget), query_budget=budget, cfg=CFG)
        assert r.search_oracle_calls <= budget
        assert len(r.best_edits) <= 1


# ---------------------------------------------------------------------------
# 5. tiny-MDP DP is a valid ceiling; two-edit exactness
# ---------------------------------------------------------------------------

class TestTinyMDPDP:
    def test_dp_ceiling_over_baselines(self):
        source = _make_record("ACGUAC")  # 6nt UTR: 18 actions, ~325 states
        budget = 2
        # DP and baselines MUST share the same underlying oracle weights
        # (same seed => identical SyntheticDeltaOracle parameters).
        dp = tiny_mdp_value_iteration(
            source, _oracle(None, seed=3), edit_budget=budget, cfg=CFG
        )
        opt = dp["optimal_value"]
        assert dp["n_states"] > 100
        results = run_all_baselines(
            source, lambda: _oracle(None, seed=3),
            query_budget=512, edit_budget=budget, seed=0,
            stage_b_ranker=_tiny_ranker(source, seed=3),
            dagger_ranker=_tiny_ranker(source, seed=3),
        )
        for r in results:
            assert r.best_score <= opt + 1e-9, (
                f"{r.method} beat DP optimum: {r.best_score} > {opt}"
            )

    def test_dp_v0_equals_terminal(self):
        source = _make_record("ACGUA")
        dp = tiny_mdp_value_iteration(source, _oracle(None), edit_budget=2, cfg=CFG)
        for h, v0 in dp["V"][0].items():
            assert v0 == dp["R"][h]

    def test_two_edit_optimum_le_dp_budget2(self):
        source = _make_record("ACGUA")  # 5nt: 15 actions, ~226 2-edit states
        two = exact_two_edit_optimum(source, _oracle(None), cfg=CFG)
        dp = tiny_mdp_value_iteration(source, _oracle(None), edit_budget=2, cfg=CFG)
        # DP allows revisits (upper bound); two-edit exact is a subset
        assert two["optimum_score"] <= dp["optimal_value"] + 1e-9

    def test_dp_optimal_trajectory_valid(self):
        source = _make_record("ACGUA")
        dp = tiny_mdp_value_iteration(source, _oracle(None), edit_budget=2, cfg=CFG)
        assert len(dp["optimal_edits"]) <= 2
        assert _check_constraints(source, dp["optimal_candidate"])

    def test_state_enumeration_size(self):
        source = _make_record("ACGUA")
        states = enumerate_states(source, 1)
        # 1 + 15 single edits
        assert len(states) == 16


# ---------------------------------------------------------------------------
# 6. Regrets
# ---------------------------------------------------------------------------

class TestRegrets:
    def test_regrets_nonnegative(self):
        source = _make_record("ACGUAC")
        dp = tiny_mdp_value_iteration(source, _oracle(None, seed=5), edit_budget=2, cfg=CFG)
        results = run_all_baselines(
            source, lambda: _oracle(None, seed=5),
            query_budget=256, edit_budget=2, seed=0,
            stage_b_ranker=_tiny_ranker(source, seed=5),
            dagger_ranker=_tiny_ranker(source, seed=5),
        )
        regrets = compute_regrets(results, dp["optimal_value"])
        for name, reg in regrets.items():
            assert reg >= -1e-9, f"{name}: negative regret {reg}"


# ---------------------------------------------------------------------------
# 7. Algorithm semantics
# ---------------------------------------------------------------------------

class TestAlgorithmSemantics:
    def test_semantics_metrics(self):
        source = _make_record("ACGUA")
        sem = compare_algorithm_semantics(
            source, _oracle(None), edit_budget=2, beam_width=2, beta=4.0, cfg=CFG
        )
        assert sem["constraint_validity"] == 1.0
        assert sem["n_states"] > 50

        # KLs non-negative
        for name, v in sem["terminal_kl"].items():
            assert v >= -1e-12, f"terminal_kl[{name}] = {v}"
        assert sem["action_kl"]["optimal_vs_ctmc_mean"] >= 0
        assert sem["action_kl"]["greedy_vs_ctmc_mean"] >= 0

        # Exact marginal IS the CTMC marginal: KL must be 0
        assert sem["terminal_kl"]["marginal_vs_ctmc_sampled_exact"] == pytest.approx(0.0)

        # Argmax agreement in [0, 1]
        for name, v in sem["argmax_agreement"].items():
            assert 0.0 <= v <= 1.0, f"argmax_agreement[{name}] = {v}"

        # Optimal expected return >= every other semantics' expected return
        opt_ret = sem["expected_returns"]["finite_horizon_optimal"]
        for name, v in sem["expected_returns"].items():
            assert v <= opt_ret + 1e-9, f"{name} expected return {v} > optimal {opt_ret}"

        # Optimal expected return matches DP optimal value
        assert opt_ret == pytest.approx(sem["optimal_value_dp"])

        # Terminal distributions are normalized
        for name, dist in sem["terminal_distributions"].items():
            assert sum(dist.values()) == pytest.approx(1.0), name

    def test_greedy_not_presumed_equal_to_marginal(self):
        """The spec forbids presuming greedy == true-flow marginal; verify the
        comparison actually distinguishes them (distributions can differ, and
        metrics are computed independently)."""
        source = _make_record("ACGUA")
        sem = compare_algorithm_semantics(
            source, _oracle(None), edit_budget=2, beta=1.0, cfg=CFG
        )
        g = sem["terminal_distributions"]["greedy_intensity"]
        m = sem["terminal_distributions"]["exact_terminal_marginal"]
        # greedy is a delta distribution; marginal is generally spread
        assert abs(sum(g.values()) - 1.0) < 1e-9
        assert max(g.values()) == pytest.approx(1.0)
        # both are valid distributions over reachable states
        assert all(v >= 0 for v in m.values())


# ---------------------------------------------------------------------------
# 8. DAgger training
# ---------------------------------------------------------------------------

class TestDaggerTraining:
    def test_dagger_counts_training_calls(self):
        source = _make_record("ACGUAC")
        oracle = _oracle(budget=None, seed=7)
        out = train_dagger_ranker(
            [source], oracle, n_rounds=2, edits_per_round=2,
            max_actions_per_state=8, seed=0,
        )
        assert out["training_oracle_calls"] > 0
        assert out["training_oracle_calls"] == oracle.search_calls
        assert out["n_pairs"] > 0
        # Ranker is usable
        r = dagger_ranker_search(
            source, _oracle(None), ranker=out["ranker"],
            query_budget=64, edit_budget=2, cfg=CFG,
        )
        assert r.constraint_valid

    def test_dagger_respects_training_budget(self):
        source = _make_record("ACGUAC")
        oracle = _oracle(budget=None, seed=8)
        cap = 10
        train_dagger_ranker(
            [source], oracle, n_rounds=3, edits_per_round=3,
            max_actions_per_state=8, seed=0, training_query_budget=cap,
        )
        assert oracle.search_calls <= cap


# ---------------------------------------------------------------------------
# 9. Budget enforcement
# ---------------------------------------------------------------------------

class TestBudgetEnforcement:
    def test_oracle_raises_when_exhausted(self):
        source = _make_record()
        oracle = _oracle(budget=2)
        oracle.score(source, source)
        oracle.score(source, source)
        with pytest.raises(BudgetExhausted):
            oracle.score(source, source)

    def test_eval_calls_not_charged_to_budget(self):
        source = _make_record()
        oracle = _oracle(budget=1)
        oracle.score(source, source, purpose="search")
        # eval calls bypass the search budget
        oracle.score(source, source, purpose="eval")
        assert oracle.search_calls == 1
        assert oracle.eval_calls == 1

    def test_greedy_stops_at_budget(self):
        source = _make_record()
        budget = 20
        r = greedy_search(
            source, _oracle(budget), query_budget=budget, edit_budget=5, cfg=CFG
        )
        assert r.search_oracle_calls <= budget


class TestDegenerateReferenceGuard:
    """make_decision must detect flat / STOP-dominated oracle landscapes."""

    @pytest.fixture
    def _make_decision(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_run_p3_07",
            Path(__file__).resolve().parent.parent / "scripts" / "run_p3_07.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.make_decision

    def test_degenerate_reference_emits_no_go(self, _make_decision):
        """When exact one-edit optimum is non-positive for majority of sources,
        the decision must be NO_GO_PREMISE_FAILURE, not a spurious Route A."""
        exact_one = {
            "src_a": {"optimum_score": -0.027, "optimum_mean_delta": -0.017},
            "src_b": {"optimum_score": -0.027, "optimum_mean_delta": -0.017},
            "src_c": {"optimum_score": -0.027, "optimum_mean_delta": -0.017},
        }
        grid = [
            {"method": "greedy", "query_budget": 128, "source_id": "src_a", "best_score": -0.05},
            {"method": "greedy", "query_budget": 2048, "source_id": "src_a", "best_score": -0.05},
        ]
        decision = _make_decision(grid, exact_one, {}, {}, [])
        assert decision["route"] == "NO_GO_PREMISE_FAILURE"
        assert decision["degenerate_reference"]["flag"] is True
        assert decision["degenerate_reference"]["frac_sources_positive_exact_one_edit"] == 0.0
        assert decision["normalized_reach"]["best_search_qb128"] is None

    def test_positive_reference_proceeds_to_route(self, _make_decision):
        """When exact one-edit optimum is positive, normal A/B/C routing applies."""
        exact_one = {
            "src_a": {"optimum_score": 0.03, "optimum_mean_delta": 0.03},
            "src_b": {"optimum_score": 0.03, "optimum_mean_delta": 0.03},
        }
        grid = [
            {"method": "greedy", "query_budget": 128, "source_id": "src_a", "best_score": 0.0288},
            {"method": "greedy", "query_budget": 128, "source_id": "src_b", "best_score": 0.0288},
        ]
        decision = _make_decision(grid, exact_one, {}, {}, [])
        assert decision["route"] in ("NO_RL_ROUTE_C", "RL_ROUTE_B", "RL_ROUTE_A")
        assert decision["degenerate_reference"]["flag"] is False
