# P3-07: RL Necessity Decision

**Date**: 2026-07-24
**Phase**: P3-07
**Decision**: NO_GO_PREMISE_FAILURE (H3 not established under current internal oracle)
**P3-08 Gate**: BLOCKED — do not enter Production GRPO

---

## 1. Decision Summary

The P3-07 strong-search ceiling experiment was executed in full:
- 10 baselines x 4 query budgets x 4 edit budgets x 24 test sources = 3,840 runs
- Exact one-edit optimum on 24 sources, exact two-edit on 6, tiny-MDP DP on 2
- Algorithm semantics on 2 sources x 2 budgets = 4 experiments

**Result**: The internal oracle (P3-02 refit ensemble) produces a flat
negative landscape where the LCB-optimal policy is STOP at the root. Every
edit decreases the reward, every search method converges to 0 edits, and
the exact one-edit optimum is -0.027 for 0/24 sources with positive headroom.

The original `make_decision` emitted `RL_ROUTE_A` (-100% reach), but this
was a **normalization artifact**: dividing by a negative reference scale
produces meaningless ratios. The corrected decision with the
degenerate-reference guard is:

```
NO_GO_PREMISE_FAILURE
```

**H3 (optimization headroom) is not established under the current internal
oracle.** The Route A/B/C decision is not applicable.

---

## 2. Root Cause Analysis

### 2.1 The Refit Ensemble Is a Constant Predictor

Diagnostic script (`/tmp/p3_07_diagnose.py`) confirmed:

| Metric | Value |
|--------|-------|
| Zero-edit raw mean delta | -0.0170 |
| Max single-edit raw delta (mean across 24 sources) | -0.0168 |
| Position-specific variation across 151 actions | < 2e-4 |
| Sources with any positive raw delta | 0/24 |
| Sources where LCB(best edit) > LCB(source) | 0/24 |

The ensemble output is essentially the training label mean regardless of
input sequence. The 20 global composition features (GC, k-mer frequencies,
dinucleotide counts) produce nearly identical values for any 50nt 5'UTR
variant, so the model cannot distinguish edit positions.

### 2.2 Label Mean Mechanism

The refit training mixture statistics:

| Tier | N | Mean Delta |
|------|---|------------|
| Measured train+val | 3,984 | +0.0766 |
| Proxy 10k subsample | 10,000 | -0.0622 |
| **Refit mixture** | **13,984** | **-0.0227** |
| Proxy 2k subsample (P3-02 original) | 2,000 | -0.0360 |

The P3-02 original run used 2k proxy records (mixture mean ~+0.008), producing
a small positive bias (+0.0295). The P3-07 driver uses 10k proxy records
(mixture mean -0.023), flipping the bias negative. The model, lacking
position-specific features, outputs this mixture mean for all inputs.

### 2.3 Corroboration from P3-02's Own Audit

P3-02's headroom analysis already flagged this:
> "The uniformity of deltas across sources, positions, and methods is suspicious.
> The oracle may have learned a constant positive bias rather than position-
> specific effects."
> "position_sensitive=False, GC-only risk=True, length-only risk=True"

P3-07 confirms this with the refit ensemble: the bias simply flipped sign
due to the different proxy subsample size, and the model has no mechanism
to produce position-specific predictions.

### 2.4 Not a Refit Fidelity Bug

The refit ensemble is NOT broken — it faithfully reproduces the P3-02
training pipeline (same config, seed, folds, architectures). The issue is
that the P3-02 model architecture itself (20 global composition features +
single-hidden-layer MLP) is structurally incapable of learning position-
specific edit effects. The refit just happens to expose this more starkly
because the training mixture mean shifted negative.

---

## 3. Search Ceiling Evidence

### 3.1 All Methods Converge to STOP

| Method | Mean Best Score | Max Best Score | N Edits = 0 |
|--------|----------------|----------------|-------------|
| random_legal_editing | -0.0269 | -0.0268 | 360/384 |
| best_single_edit | -0.0269 | -0.0268 | 360/384 |
| greedy | -0.0269 | -0.0268 | 360/384 |
| stage_b_ranker | -0.2642 | -0.0765 | 0/384 |
| beam_search | -0.0269 | -0.0268 | 360/384 |
| simulated_annealing | -0.0269 | -0.0268 | 360/384 |
| mcts | -0.0269 | -0.0268 | 360/384 |
| oracle_guided_local_search | -0.0269 | -0.0268 | 360/384 |
| dagger_ranker | -0.0269 | -0.0268 | 360/384 |
| dagger_ranker_plus_limited_search | -0.0269 | -0.0268 | 360/384 |

90% of runs (3456/3840) selected 0 edits (STOP). The remaining 384 runs with
edits > 0 are from `stage_b_ranker` (the only method that proposes edits
without oracle verification at proposal time), which also has the worst
scores.

### 3.2 Exact Evaluation Confirms

| Evaluation | Result |
|-----------|--------|
| Exact one-edit optimum (24 sources) | -0.0269 (0/24 positive) |
| Exact two-edit optimum (6 sources) | -0.0269 (= one-edit, STOP is optimal) |
| Tiny-MDP DP (2 sources, budget 2) | -0.0269, optimal_edits = [] |

The DP explicitly confirms: the finite-horizon optimal policy is STOP at
the root. No edit sequence of length <= 2 improves the reward.

### 3.3 Zero Regret Is Not a Search Success

All search methods achieve 0 regret vs DP, but this is trivially because
the optimal policy is STOP — and STOP is the default action every method
falls back to when no edit improves the score. Zero regret here means
"the search correctly identified that doing nothing is optimal," not
"search is powerful enough to find the optimum."

---

## 4. Algorithm Semantics Summary

See `docs/p3_07_algorithm_semantics.md` for full details. Key findings:

- greedy == finite_horizon_optimal == beam_search (all choose STOP)
- stochastic CTMC is worse (mixes in negative-value edits)
- terminal KL(greedy, CTMC) = 4.0-4.8 nats (greedy != marginal, spec met)
- constraint_validity = 1.0 (all methods, all experiments)
- exact_terminal_marginal == CTMC marginal (KL = 0, implementation validated)

---

## 5. Cost Report

### 5.1 Inference Cost (per new source)

| Method | Oracle Calls | Wall Clock (s) |
|--------|-------------|-----------------|
| random_legal_editing | <= query_budget | ~0.1-2.0 |
| best_single_edit | <= 151 | ~0.5 |
| greedy | <= 151 * edit_budget | ~0.5-5.0 |
| beam_search | <= 8 * 151 * edit_budget | ~1.0-10.0 |
| stage_b_ranker | 0 (guidance) + 1 (eval) | ~0.01 |
| dagger_ranker | 0 (guidance) + 1 (eval) | ~0.01 |
| dagger_ranker + limited search | <= 32 + 1 | ~0.1 |

### 5.2 Full Lifecycle Cost

| Component | Cost |
|-----------|------|
| Ensemble training oracle calls | 0 (trained on benchmark labels) |
| DAgger training oracle calls | 3,072 |
| Total experiment wall clock | 3,480s (~58 min, CPU-only) |
| GPU memory | 0 bytes (CPU-only numpy MLP) |
| Break-even designed cargos | N/A (no Route B) |

---

## 6. Corrected Decision Rule

The original `make_decision` in `scripts/run_p3_07.py` had a bug: it
normalized by `scale = max(abs(optimum_score), 1e-6)`, which when
`optimum_score` is negative, produces `best_score / abs(optimum_score)`
= negative/positive = negative, interpreted as "-100% reach." This
triggers Route A spuriously.

**Fix applied**: Added a `degenerate_reference` guard at the top of
`make_decision`. When the exact one-edit optimum is non-positive for
>= 50% of sources, the function returns `NO_GO_PREMISE_FAILURE` with
`normalized_reach` set to `null`, bypassing the A/B/C routing entirely.

Unit tests added:
- `test_degenerate_reference_emits_no_go`: verifies NO_GO when all
  references are negative
- `test_positive_reference_proceeds_to_route`: verifies normal A/B/C
  routing when references are positive

46/46 tests pass (44 original + 2 new).

---

## 7. Recommendation

### 7.1 Do NOT Enter P3-08

P3-08 (Production GRPO) is **blocked**. The P3-07 gate requires Route A
or Route B to proceed. The current result is NO_GO_PREMISE_FAILURE,
which is neither.

### 7.2 Root Cause Is Oracle, Not Search or RL

The search infrastructure is validated and correct:
- All 10 baselines respect query and edit budgets
- DP ceiling is correctly computed (11,176 states for budget 2)
- Algorithm semantics are correctly measured
- Constraint validity is 100%

The problem is that the P3-02 oracle has no position-specific signal. It
is a constant predictor whose output depends only on the training label
mean, not on the input sequence.

### 7.3 Required Remediation Before Re-running P3-07

1. **P3-02 feature upgrade**: Replace the 20 global composition features
   with position-aware features (one-hot encoding, k-mer at edit position,
   local sequence context). The current features cannot distinguish
   different edit positions on the same 50nt UTR.

2. **P3-05 real backbone**: Deploy a real neural backbone (1D CNN or
   transformer over one-hot sequence) that can learn position-specific
   effects. The single-hidden-layer MLP on global statistics is
   structurally incapable of the required discrimination.

3. **Proxy subsample consistency**: Fix the proxy subsample size (2k vs
   10k) to match P3-02's original configuration, or better, eliminate
   the dependence on label mean by using centered labels and
   position-aware features.

4. **Re-run P3-07** after oracle remediation with the same 10 baselines,
   budgets, and evaluation protocol. The search infrastructure, test
   suite, and driver script are ready and do not require modification.

### 7.4 Falsifiable Re-entry Condition

P3-07 may be re-run when:
- The oracle predicts positive raw mean delta for > 0/24 test sources
- The position sensitivity check (std of predictions across positions)
  exceeds 1e-3 (currently < 2e-4)
- The exact one-edit optimum is positive for >= 50% of sources

Until then, H3 (optimization headroom) remains unestablished under the
internal oracle, and the RL necessity question cannot be answered.

---

## 8. Claim Ladder Status

| Claim | Status | Justification |
|-------|--------|---------------|
| C0: Problem definition | ESTABLISHED | P3-00/01 complete |
| C1: Local edit predictability | WEAK | P3-02 oracle is constant predictor |
| C2: Internal optimization | NOT ESTABLISHED | P3-07 shows STOP is optimal |
| C3: Independent oracle transfer | NOT TESTED | Blocked by C2 |
| C4: Equal-budget optimization value | NOT TESTED | Blocked by C2 |
| C5: Prospective experimental improvement | NOT TESTED | Blocked by C2 |

H3 (optimization headroom) is the critical gate. Without it, C2-C5 cannot
be claimed, and the RL question is moot.

---

## 9. Unresolved Risks

1. **Oracle remediation may not suffice**: Even with position-aware
   features, the underlying benchmark data may not contain enough
   position-specific signal to learn meaningful edit effects. The
   measured tier has only 4,802 records across ~250 sources.

2. **Proxy data quality**: The 473K proxy records have a mean delta of
   -0.069, suggesting the proxy simulator may have a systematic negative
   bias. Mixing proxy data with measured data may dilute the positive
   signal from measured records.

3. **Test anchor validity**: The test anchor sequences lack AUG start
   codons in some cases, but since Task A only edits 5'UTR (not CDS),
   this should not affect the oracle's 5'UTR edit predictions. However,
   it may affect downstream protein-output prediction if the oracle
   implicitly uses CDS context.

---

## 10. Artifacts

| Artifact | Path | SHA-256 |
|----------|------|---------|
| Search results JSON | `docs/p3_07_search_results.json` | (to be computed at commit) |
| Algorithm semantics MD | `docs/p3_07_algorithm_semantics.md` | (this file) |
| RL necessity decision MD | `docs/p3_07_rl_necessity_decision.md` | (this file) |
| Search protocol MD | `docs/p3_07_search_protocol.md` | (existing) |
| Search code | `rl/p3_07_search.py` | (committed at 688d2fe) |
| Driver script | `scripts/run_p3_07.py` | (updated with guard) |
| Tests | `tests/test_p3_07_search.py` | (46/46 green) |
| Diagnostic script | `/tmp/p3_07_diagnose.py` | (server-only, not committed) |
