# P3-07 Search Protocol (pre-registered)

Phase: P3-07 — Strong-search ceiling and RL necessity gate.
Question answered: **does this task actually need RL before large-scale GRPO?**

This protocol is written before the experimental run and frozen by commit
`688d2fe` (code) + this document. All thresholds below are pre-registered;
the decision in `docs/p3_07_rl_necessity_decision.md` must reference them.

---

## 1. Task and action space

Active task: **task_a (5'UTR substitution only)**, per the P3-00A frozen
contract (`p3_contract_v2`, operational `allowed_actions =
five_utr_substitution`). The CDS and 3'UTR of each source record are
**inert placeholders** (`INERT_CDS = AUG + GCU*4 + UAA`, `INERT_THREE_UTR =
UGCU`) that exist only to satisfy the `MRNARecord` schema; the action space
never touches them. This is stated here so the placeholder CDS is never
mistaken for a real cargo.

Actions (from `rl/p3_06_mdp.py`):

- `STOP`
- `FIVE_UTR_SUB(position, nucleotide)` — single-nt substitution, no indels,
  no identity substitution, no revisit of a previously visited sequence
  (cycle-free by `visited_states`).

Hard constraints (enforced by the action space + rejection, never by linear
reward): protein identity 100%, transcript length 100% unchanged, no indels,
no 3'UTR edits, no single-nt CDS nonsynonymous intermediates.

## 2. Reward

`RewardV3Config(context="protein_output_focused")` (Reward v3, P3-06):

```
score(source, candidate) = LCB[Δprotein_output] − 0.05 · n_edits
LCB = ensemble_mean − λ · ensemble_uncertainty     (λ = 1.0)
```

Source-normalized by construction: score(source, source) = 0. Edit cost
0.05 per edit (`w_edit_cost = −0.05`). Uncertainty = ensemble std across
the 4 architectures.

## 3. Oracle (evaluation-time)

The P3-02 cross-fitted delta-oracle ensemble: 4 structurally different
architectures (absolute / difference / siamese / edit-conditioned) × 5
group-K-fold models (groups = source_id), trained on P3-01 benchmark
measured train/val + up to 10 000 proxy-tier records.

The published P3-02 checkpoints (`checkpoints/p3_delta_oracles/*.npz`) do
not serialize normalization statistics (`_y_mean`/`_y_std`/`_bias`), so the
driver **deterministically refits** the ensemble in memory with the frozen
P3-02 pipeline (seed 42, hidden 64, lr 1e-3, 100 epochs, max_seq_len 100).
This is a deterministic reproduction of the frozen training procedure, not
a re-experiment; fold assignment is identical because it uses the same
seeded group shuffle.

Every baseline queries the oracle through a `CountingOracle` wrapper that
enforces the query budget and raises `BudgetExhausted` when exceeded.
Training-time queries (DAgger) are accounted separately from
inference-time queries (per-source search).

## 4. Sources

- **Test mothers** (search evaluation): up to 24 `wild_type_anchor`
  records with `split_role == "test"` from `measured_tier.jsonl`, shuffled
  with seed 0.
- **Train mothers** (DAgger only): up to 24 `wild_type_anchor` records
  with `split_role == "train"`. No test mother is used for ranker or
  DAgger training.
- Stage B ranker training pairs: measured train/val + up to 20 000
  proxy-tier pairs (seed 42 subsample).

## 5. Baselines (10, all required)

| # | Method | Description |
|---|--------|-------------|
| 1 | `random_legal_editing` | Sample legal edit sequences uniformly; keep best scored. |
| 2 | `best_single_edit` | Evaluate all legal single edits; take the best (with budget left unused). |
| 3 | `greedy` | Iteratively take the best-scoring legal child until budget/STOP. |
| 4 | `stage_b_ranker` | Linear delta-ranker on benchmark pairs; propose top-ranked edits, verify by oracle. |
| 5 | `beam_search` | Width-8 beam over legal children ranked by oracle score. |
| 6 | `simulated_annealing` | Single-nt proposal chain, geometric cooling (×0.995/step); global best tracked by oracle score. |
| 7 | `mcts` | UCT (UCB1) over the edit tree; each rollout leaf evaluation costs 1 oracle call. |
| 8 | `oracle_guided_local_search` | First-improvement local search over a sampled neighborhood of 16 legal children per step (cheaper per step than greedy's full enumeration). |
| 9 | `dagger_ranker` | Ranker trained by 2 rounds of DAgger on train mothers (expert = oracle-scored best child). |
| 10 | `dagger_ranker_plus_limited_search` | DAgger ranker prunes the action space to a top-m shortlist; a width-4 beam runs in the pruned space under the query budget. |

Budgets (exact grid, no deviation):

```
query budgets Q ∈ {32, 128, 512, 2048}
edit budgets  E ∈ {1, 3, 5, 10}
```

Every (method × Q × E × test source) cell is run. A method may use fewer
queries than Q but never more (enforced by `BudgetExhausted`).

## 6. Exact evaluation

- **Exact one-edit optimum**: exhaustive enumeration of all legal single
  edits, on every test source.
- **Exact two-edit optimum**: exhaustive enumeration of all legal two-edit
  sequences, on the first 6 test sources.
- **Tiny-MDP dynamic programming**: full state-space enumeration + value
  iteration with edit budget 2, on the first 2 test sources. This gives the
  exact optimum of the budget-2 MDP, hence a valid ceiling for every
  budget-2-compatible baseline.

Reported per source: search optimality gap, greedy regret, ranker regret,
policy regret — all defined as `DP optimum − method best score ≥ 0`.

## 7. Algorithm semantics

On 2 test sources, edit budgets 1 and 2, compare 5 semantics of the same
policy/reward:

```
greedy intensity            (argmax policy)
stochastic CTMC             (softmax-sampled trajectories, β = 4)
finite-horizon optimal      (value iteration policy)
exact terminal marginal     (exact enumeration of the CTMC terminal law)
beam search                 (width 4)
```

Metrics: action KL, terminal KL, argmax agreement, expected return,
constraint validity (must be exactly 1.0 for all five). We do **not**
presuppose that greedy is equivalent to the true flow marginal.

## 8. Cost report

- Inference-time: oracle calls per newly designed source, wall-clock, GPU
  memory (expected 0 — the ensemble is CPU numpy MLPs).
- Life-cycle: DAgger training oracle calls, ensemble training queries
  (0 — trained on benchmark labels), inference calls, and the break-even
  number of designed cargos for Route B:

```
break_even_cargos = dagger_training_calls / (search_calls_per_cargo − policy_calls_per_cargo)
```

## 9. Decision rule (pre-registered)

Let `norm_score(method, Q)` = mean over test sources of
`best_score / |exact one-edit optimum|` (per-source achievable scale;
score(source) = 0 by construction).

- **RL_ROUTE_A (quality)**: `max_method norm_score(·, 2048) < 0.90` —
  even budget-2048 strong search leaves > 10% of the one-edit reference
  unreach­ed; search is insufficient within practical budgets.
- **NO_RL_ROUTE_C**: `max_method norm_score(·, 128) ≥ 0.95` (or
  `dagger_ranker_plus_limited_search` ≥ 0.95 without a cost asymmetry) —
  ranker + limited search is already near-optimal; RL has no quality or
  cost advantage. Stop scaling GRPO; the paper becomes
  local-delta prediction + minimal-edit benchmark + strong constrained
  search + prospective validation.
- **RL_ROUTE_B (amortization)**: otherwise — near-optimal quality is
  achievable but only at high per-source query cost, while
  `dagger_ranker_plus_limited_search` matches it at much lower amortized
  cost. Proceed to P3-08 with the claim "amortized constrained optimizer"
  and report the break-even deployment scale.

Gates: P3-08 executes only under RL_ROUTE_A or RL_ROUTE_B.

## 10. Reproducibility

```
git commit: 688d2fe (code) 
driver: scripts/run_p3_07.py --benchmark-dir data/p3/benchmark \
        --output-json docs/p3_07_search_results.json
seed: 0 (sources/DAgger), 42 (ensemble refit, ranker pairs)
hardware: CPU only (OMP_NUM_THREADS=8), GPU memory 0
```

All raw per-cell results are in `docs/p3_07_search_results.json`;
semantics numbers in that file are mirrored and discussed in
`docs/p3_07_algorithm_semantics.md`; the final gate call is in
`docs/p3_07_rl_necessity_decision.md`.
