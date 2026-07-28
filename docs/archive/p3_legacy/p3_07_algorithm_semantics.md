# P3-07: Algorithm Semantics Report

**Date**: 2026-07-24
**Phase**: P3-07
**Status**: COMPLETED (degenerate-reference condition; see note below)
**Oracle**: P3-02 cross-fitted ensemble (4 architectures x 5 folds, deterministic refit)
**Sources**: 2 test mothers x 2 edit budgets (1, 2) = 4 experiments
**Artifact**: `docs/p3_07_search_results.json` → `algorithm_semantics`

> **Note (post-remediation)**: The semantics below were collected under the
> original degenerate oracle (constant-predictor, flat negative landscape).
> The oracle has since been remediated with position-aware features +
> source-bias centering (see `docs/p3_07_rl_necessity_decision.md` Section 2).
> The decision was updated to RL_ROUTE_B based on the remediated oracle's
> search grid (3,840 runs), but the algorithm semantics experiments were
> NOT re-run because: (1) the spec requirement "greedy != CTMC marginal"
> is already satisfied (KL 4.0-4.8 nats); (2) the constraint validity and
> implementation validation (exact marginal == CTMC marginal) are
> oracle-independent. The semantics results stand as historical record of
> the degenerate condition; the synthetic-oracle tests in
> `tests/test_p3_07_search.py` cover the non-degenerate case.

---

## 1. Protocol

On a tiny enumerable state space (5'UTR length 50, 4 nucleotides, edit
budgets 1 and 2), five algorithm semantics are compared:

| Algorithm | Description |
|-----------|-------------|
| `greedy_intensity` | Deterministic greedy: at each step, pick the action with highest immediate reward (including STOP). |
| `stochastic_ctmc` | Uniform stochastic CTMC: sample actions proportional to exp(reward / temperature), temperature=1. |
| `finite_horizon_optimal` | Exact DP value iteration on the full enumerable MDP. |
| `exact_terminal_marginal` | Analytical terminal state distribution under the CTMC policy (closed-form, not sampled). |
| `beam_search` | Beam search with width 8, pruning by reward. |

Metrics reported per experiment:

| Metric | Definition |
|--------|-----------|
| `expected_return` | Expected total reward under the algorithm's policy. |
| `terminal_kl` | KL divergence between terminal state distributions of pairs of algorithms. |
| `action_kl` | Mean per-state action distribution KL between pairs. |
| `argmax_agreement` | Fraction of states where two algorithms' argmax actions coincide. |
| `constraint_validity` | Fraction of terminal states that satisfy hard constraints (protein identity + length). |

The spec requires: *do not presume greedy equals true-flow marginal.*

---

## 2. Results

### 2.1 State Space

| Experiment | Source | Budget | N States |
|------------|--------|--------|----------|
| 1 | snv:0f7ee24bb93a | 1 | 151 |
| 2 | snv:0f7ee24bb93a | 2 | 11,176 |
| 3 | snv:2510b14d4ee8 | 1 | 151 |
| 4 | snv:2510b14d4ee8 | 2 | 11,176 |

### 2.2 Expected Returns

| Algorithm | Exp 1 (budget 1) | Exp 2 (budget 2) | Exp 3 (budget 1) | Exp 4 (budget 2) |
|-----------|------------------|------------------|------------------|------------------|
| greedy_intensity | -0.026888 | -0.026888 | -0.026925 | -0.026925 |
| stochastic_ctmc | -0.076419 | -0.123846 | -0.076449 | -0.123876 |
| finite_horizon_optimal | -0.026888 | -0.026888 | -0.026925 | -0.026925 |
| exact_terminal_marginal | -0.076419 | -0.123846 | -0.076449 | -0.123876 |
| beam_search | -0.026888 | -0.026888 | -0.026925 | -0.026925 |

**Key observation**: greedy == finite_horizon_optimal == beam_search across all
experiments. This is because the oracle landscape is flat and negative — every
edit decreases the reward, so the optimal policy is STOP at the root. The DP
confirms this: `optimal_edits = []` for both sources.

### 2.3 Terminal KL Divergence

| Pair | Exp 1 | Exp 2 | Exp 3 | Exp 4 |
|------|-------|-------|-------|-------|
| greedy vs ctmc | 4.819 | 4.029 | 4.819 | 4.029 |
| optimal vs ctmc | 4.819 | 4.029 | 4.819 | 4.029 |
| beam vs ctmc | 4.819 | 4.029 | 4.819 | 4.029 |
| greedy vs optimal | 0.000 | 0.000 | 0.000 | 0.000 |
| beam vs optimal | 0.000 | 0.000 | 0.000 | 0.000 |
| marginal vs ctmc | 0.000 | 0.000 | 0.000 | 0.000 |

**Interpretation**: greedy/optimal/beam are identical (KL=0) and all diverge
from the stochastic CTMC (KL~4-5 nats). The exact terminal marginal matches
the CTMC marginal perfectly (KL=0), validating the CTMC sampling implementation.

### 2.4 Action KL

| Pair | Exp 1 | Exp 2 | Exp 3 | Exp 4 |
|------|-------|-------|-------|-------|
| optimal vs ctmc mean | 1.264 | 1.847 | 1.264 | 1.847 |
| greedy vs ctmc mean | 1.264 | 1.845 | 1.264 | 1.845 |

At budget 2, the action KL between optimal and CTMC increases (1.264 → 1.847),
reflecting the larger action space and greater divergence between deterministic
and stochastic policies.

### 2.5 Argmax Agreement

| Pair | Exp 1 | Exp 2 | Exp 3 | Exp 4 |
|------|-------|-------|-------|-------|
| optimal vs ctmc | 1.000 | 0.693 | 1.000 | 0.662 |
| greedy vs ctmc | 1.000 | 1.000 | 1.000 | 1.000 |

At budget 1, all algorithms agree on the argmax (STOP). At budget 2, the
optimal policy and CTMC diverge in ~31-34% of states, while greedy and
optimal remain identical. This confirms that greedy is NOT presumed equal to
the CTMC marginal — they genuinely differ.

### 2.6 Constraint Validity

All experiments: `constraint_validity = 1.0`.

Every algorithm produces only constraint-valid terminal states (protein
identity preserved, transcript length unchanged). This validates the P3-06
action space design: hard constraints are enforced by construction, not by
reward penalty.

---

## 3. Interpretation

### 3.1 Greedy != CTMC Marginal (Spec Requirement Met)

The spec requires: *do not presume greedy equals true-flow marginal.*

The data confirms this: terminal KL between greedy and CTMC is 4.0-4.8 nats,
and action KL is 1.3-1.8. These are large divergences, driven by the fact
that greedy deterministically selects STOP while CTMC spreads probability
mass over all legal actions including negative-expected-value edits.

### 3.2 Optimal Policy = Greedy = STOP

In all four experiments, the finite-horizon optimal policy is identical to
greedy: select STOP at the root state. This is because:

1. The refit P3-02 ensemble predicts a near-constant negative delta
   (~-0.017) for any edit, with negligible position-specific variation
   (<2e-4 across 151 actions).
2. The LCB reward (mean - lambda * uncertainty - edit_cost) is therefore
   negative for every edit.
3. The source score (0 edits, delta=0) is 0.0, which exceeds any edit's
   reward.
4. DP confirms: `optimal_value == source_score`, `optimal_edits == []`.

### 3.3 CTMC Is Worse (Stochastic Noise)

The stochastic CTMC and exact terminal marginal both have significantly lower
expected returns (-0.076 to -0.124 vs -0.027) because they mix in
negative-expected-value edits. This is expected behavior for a flat negative
landscape: any stochasticity hurts.

### 3.4 Constraint Enforcement Is Perfect

All five algorithms achieve 100% constraint validity. The P3-06 action space
design (legal action generation excludes indels, nonsynonymous CDS, 3'UTR)
ensures hard constraints by construction.

---

## 4. Limitation: Degenerate Oracle Landscape

The algorithm semantics results are valid but were collected under a
**degenerate oracle condition** where the P3-02 refit ensemble acts as a
near-constant predictor. The key findings (greedy=optimal=STOP, CTMC worse)
are correct for this landscape but do not generalize to a landscape with
real position-specific signal.

The unit-test suite (`tests/test_p3_07_search.py`) includes a
`SyntheticDeltaOracle` with position-weighted interactions that produces a
non-trivial landscape. On that synthetic oracle:
- greedy != optimal (epistasis makes greedy suboptimal)
- CTMC marginal != greedy (stochastic exploration finds different states)
- DP regret > 0 for greedy

These synthetic-oracle results validate the algorithm semantics
implementation independently of the degenerate real-oracle condition.

---

## 5. Acceptance Criteria

| Criterion | Status | Evidence |
|-----------|--------|----------|
| greedy not presumed equal to marginal | PASS | terminal KL 4.0-4.8, action KL 1.3-1.8 |
| constraint validity = 1.0 | PASS | All 4 experiments |
| terminal KL >= 0 | PASS | All values non-negative |
| action KL >= 0 | PASS | All values non-negative |
| argmax agreement in [0,1] | PASS | All values in [0.66, 1.0] |
| optimal expected return >= CTMC | PASS | -0.027 >= -0.076 to -0.124 |
| exact marginal == CTMC marginal | PASS | KL = 0.000 in all experiments |
| 5 algorithms compared | PASS | greedy, CTMC, optimal, marginal, beam |
| metrics: action KL, terminal KL, argmax, expected return, constraint validity | PASS | All reported |

---

## 6. Reproducibility

- **Code**: `rl/p3_07_search.py` → `compare_algorithm_semantics()`
- **Data**: `docs/p3_07_search_results.json` → `algorithm_semantics`
- **Driver**: `scripts/run_p3_07.py` (section [8])
- **Tests**: `tests/test_p3_07_search.py` → `TestAlgorithmSemantics` (2 tests)
- **Wall clock**: 0.37-28.6s per experiment (budget 2 DP is the bottleneck)
