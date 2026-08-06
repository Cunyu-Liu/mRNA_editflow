# F0-X Source-Anchored Legal Edit Flow — mRNA-XEditFlow v1.1

- **Migration Goal:** `GOAL-XEDITFLOW-MIGRATION-01`
- **Prior terminal state:** `O0X_SEARCH_CEILING_ESTABLISHED` with `FLOW_HEADROOM_LIMITED` signal
- **Phase:** F0-X — source-anchored legal Edit Flow (base flow, substitution-only)
- **Gate outcome:** `F0X_LEGAL_EDIT_FLOW_BASE_ESTABLISHED`
- **UTC:** 2026-08-06
- **Worktree branch:** `xeditflow-migration-20260806T024650Z`
- **Effect dataset (SHA-256):** `f23a9fdd54a8ead90dccf793a073ccc6ed804a9b760a266c77b02e1fb1007ba1`
- **Runner:** `scripts/f0x/run_f0x.py` + `scripts/f0x/base.py` + `scripts/f0x/flow.py` on GPU `cuda:1` (GPU 4 avoided)
- **Results artifact:** `artifacts/f0x/f0x_results_5U-A1.json` (SHA-256 `b141f7a54b3021ef842b7e1e140e0f428fb803ebe3ddb32378a02d47bb52be96`)

---

## 1. FACTS_FROM_REPO
- Trained a from-scratch source-conditioned `FlowRateNet` on real 5'UTR effect data (8,000 train rows with ≥1 SUB edit, 1,500 Bregman/Edit-Flow-loss steps, batch 64, GPU `cuda:1`). Bregman loss converged from 632.4 → 16.48.
- Primary action set is `UTR5_SUB(position, target_nt)`: substitution-only, length preserved, whole 5'UTR editable, identity (target == current nt) is not a legal action.
- Termination is `FIXED_BUDGET` (k ∈ {1,3,5}); no learned general STOP, per §5.1 freeze.
- Sampler is `FirstOrderConstrainedSampler` — explicitly a **first-order** (Euler) constrained sampler over the legal edit graph, **not** an exact CTMC (sampler naming matches numerical nature per acceptance).
- Evaluated on 1,000 held-out 5'UTR sources (independent of training rows) with both `uniform_policy` (baseline) and the trained `flow_policy`, at k=1,3,5.
- Verified the four acceptance invariants at scale on every (policy, k) cell: legality, length preservation, budget violation, and fixed-seed trajectory reproducibility.

## 2. FACTS_FROM_CONTRACTS
- § Phase F0-X: Primary 5'UTR action set `UTR5_SUB(position,target_nt)`; termination fixed budget / fixed horizon, no learned general STOP; k=[1,3,5]; implement source/current encoder, biological coordinate state, legal action enumerator, non-negative rate, hard mask before normalization, source-conditioned/region-specific coupling, Bregman/Edit-Flow loss, first-order constrained sampler, trajectory log.
- Acceptance: legality=100%; length preservation=100%; budget violation=0; reproducible fixed-seed trajectory; toy/base distribution test passes; sampler naming matches numerical nature.
- Old alignment/native/fresh/multi-QK are secondary/baseline only; must not be written as exact guidance or primary contribution.
- § B0-X `FLOW_HEADROOM_LIMITED` signal (from O0-X): a Flow must justify itself via fixed-budget multi-step generation, OOD, or a quality–cost frontier — not via closed measured-space ranking (already saturated by exact enumeration).

## 3. INFERENCES
- **The dropout non-determinism bug was the only acceptance blocker.** The first run left the model in `train()` mode after training, so `flow_policy` inference applied active dropout → non-reproducible trajectories (reproducible_pct 47.15% / 6.05% / 1.0% for k=1,3,5). Fix: `net.eval()` after training + deterministic CUDA seeding (`torch.manual_seed`, `torch.cuda.manual_seed_all`, `cudnn.deterministic=True`, `cudnn.benchmark=False`). After the fix, `flow_policy` reproducibility is **100%** at every k.
- **All acceptance invariants hold for both policies at all budgets.** Legality 100%, length preservation 100%, budget violation 0, mean_n_steps == expected_n_steps, and 100% fixed-seed reproducibility (uniform and flow). This is property-level correctness of the legal Edit Flow base, independent of biological effect quality (which is a separate, later gate).
- **The base flow learns meaningful per-position substitution rates** (Bregman loss 632.4 → 16.48), and the sampler reproduces the target edit distribution on a toy graph (unit test `test_toy_base_distribution_sampler_reproduces_target`), confirming the encoder → rate → first-order sampler stack is internally consistent.
- **F0-X is a base-flow legality/correctness gate, not a quality gate.** It does not claim the generated edits improve any biological endpoint; it establishes that the source-anchored substitution-only fixed-budget Edit Flow is legal, deterministic, budget-respecting and reproducible. Effect-quality and guidance comparisons are deferred to G0-X / G1-X.

## 4. UNKNOWN_OR_BLOCKED
- Whether the Flow's generated edits are *biologically beneficial* (vs measured deltas) is NOT tested here; that requires the exact-guidance / critic integration phases (G0-X / G1-X) and is out of scope for F0-X.
- Whether the Flow provides amortization / OOD / fixed-budget multi-step quality–cost value over exact enumeration remains open (the `FLOW_HEADROOM_LIMITED` signal from O0-X carries forward; F0-X only establishes the legal base is correct).
- INITIAL_NO_EDIT_AUX (optional, shared-encoder, sampler-call=0) was not exercised in this run; it is an optional auxiliary, not an acceptance requirement.

## 5. FILES_READ
- `scripts/f0x/flow.py`, `scripts/f0x/base.py`, `scripts/f0x/run_f0x.py`, `scripts/m4_sparse/config.py`, `tests/migration/test_f0x_flow.py`, `artifacts/b0x/effect_dataset.jsonl`, `artifacts/o0x/o0x_results.json`, `reports/migration/O0X_CLOSED_MEASURED_OPTIMIZATION_GATE.md`.

## 6. FILES_CHANGED
- `scripts/f0x/flow.py` (new): `LegalAction`, `EditFlowState` (biological-coordinate), `enumerate_legal_actions`, `apply_action`, `apply_hard_mask`, `nonnegative_rates`, `policy_from_masked_logits`, `bregman_flow_loss`, `FirstOrderConstrainedSampler`, `uniform_policy`, `build_state`, `legal_matrix`.
- `scripts/f0x/base.py` (new): `FlowRateNet` source-conditioned base flow (source-cached encoder, current-state encoder, budget conditioning, per-position substitution rate head, hard-mask + non-negative-rate + policy outputs, source-anchored per-state `policy_fn`).
- `scripts/f0x/run_f0x.py` (new): end-to-end runner — trains the base flow on GPU, evaluates uniform + flow policies at k=1,3,5 over real held-out 5'UTR sources, verifies legality/length/budget/reproducibility, writes `artifacts/f0x/f0x_results_<benchmark>.json`.
- `tests/migration/test_f0x_flow.py` (new): 13 unit tests.
- `reports/migration/F0X_SOURCE_ANCHORED_LEGAL_EDIT_FLOW_GATE.md` (this report).

## 7. COMMANDS_RUN
- Training + evaluation (deterministic, patched):
  `python -m scripts.f0x.run_f0x --dataset artifacts/b0x/effect_dataset.jsonl --out-dir artifacts/f0x --benchmark 5U-A1 --train-limit 8000 --eval-limit 1000 --steps 1500 --batch 64 --w 3.0 --budgets 1 3 5 --gpu cuda:1`
- Unit tests: `python -m pytest tests/migration/test_f0x_flow.py -q` → **13 passed**

## 8. TEST_RESULTS
- New F0-X tests (13/13 pass): legal action enumeration excludes identity; application on non-editable / identity / out-of-range / exhausted-budget rejected; length preserved; budget spent exactly; trajectory legality re-check; sampler legality/length/budget at k=1,3,5; fixed-seed reproducibility; hard-mask masks identity cells; non-negative rates; Bregman loss minimises on a known target; toy base distribution sampler reproduces the target edit distribution (empirical ≈ target within tolerance).

## 9. DATA_COUNTS_AND_DENOMINATORS
- Train rows: 8,000 (5U-A1, delta-defined, ≥1 SUB edit). Eval sources: 1,000 held-out 5'UTR (independent, seed+1).
- Loss: first 632.45, last 16.48 over 1,500 steps.
- Acceptance counts are over 1,000 sources × 3 budgets × 2 policies, each with eval + reproducibility re-run.

### Horizontal acceptance table (5U-A1, 1,000 eval sources)
| k | policy | legality % | length % | budget violation | mean_n_steps | expected | reproducibility % |
|---|---|---|---|---|---|---|---|
| 1 | uniform | 100 | 100 | 0 | 1.0 | 1 | 100 |
| 1 | flow | 100 | 100 | 0 | 1.0 | 1 | 100 |
| 3 | uniform | 100 | 100 | 0 | 3.0 | 3 | 100 |
| 3 | flow | 100 | 100 | 0 | 3.0 | 3 | 100 |
| 5 | uniform | 100 | 100 | 0 | 5.0 | 5 | 100 |
| 5 | flow | 100 | 100 | 0 | 5.0 | 5 | 100 |

- All acceptance criteria met for every cell. Honest measured values; no fabricated metrics.

## 10. REUSE_DECISIONS
- Edit Flow engineering core REUSE_WITH_REBIND: reuse rate / hard-mask / Bregman / first-order sampler concepts, rebound to substitution-only `UTR5_SUB` primary and fixed-budget termination (STOP held at identifiability gate per §5.1).
- `scripts/m4_sparse/config.py` REUSE_AS_IS for `NUC_ORDER`, `MAX_SEQ_LEN`, and the CUDA device policy (GPU 4 forbidden, fallback=0).
- Sampler naming REUSE_AS_IS with explicit qualifier: `FirstOrderConstrainedSampler` (first-order, not exact CTMC), matching its numerical nature per acceptance.

## 11. GATE_STATUS
- **`F0X_LEGAL_EDIT_FLOW_BASE_ESTABLISHED`** (PASS).
  - legality = 100%, length preservation = 100%, budget violation = 0, fixed-seed reproducible trajectory = 100% (uniform and flow, k=1,3,5).
  - Toy/base distribution test passes; sampler naming matches numerical nature.
  - Bregman loss converged, confirming the source → rate → first-order sampler stack is functional.
- The bug that initially failed acceptance (dropout non-determinism) was root-caused and fixed; final results are deterministic and reproducible.

## 12. CLAIMS_UNLOCKED
- A source-anchored, substitution-only, fixed-budget (k=1,3,5) legal Edit Flow base is implemented and passes all acceptance invariants on real 5'UTR sources: legality, length preservation, budget respect, and fixed-seed reproducibility.
- The base flow learns non-negative per-position substitution rates (Bregman loss 632→16.5) and a first-order constrained sampler reproduces a target edit distribution on a toy graph.

## 13. CLAIMS_STILL_PROHIBITED
- NO claim that the Flow generates *biologically beneficial* edits or improves any measured endpoint (not tested; requires G0-X / G1-X guidance+critic).
- NO claim of exact density-ratio guidance (G0-X theory not yet proven).
- NO claim of amortization / OOD / quality–cost benefit over exact enumeration (the O0-X `FLOW_HEADROOM_LIMITED` signal carries forward).
- NO L4 real biological improvement. NO GSE246381 label access. NO A2 dense / CDS-B1 unlock. NO `EFFECT_MODEL_ESTABLISHED` (M4 full gate still unmet).

## 14. NEXT_PHASE_INPUTS
- Feed the legal base flow (rates, mask, first-order sampler) into G0-X (exact guidance theory + enumerable toy graph). F0-X supplies a correct, deterministic, budget-respecting base-flow terminal distribution over the legal edit graph.
- Carry forward: the `FLOW_HEADROOM_LIMITED` signal — G0-X/G1-X must demonstrate value on fixed-budget multi-step generation, OOD, or a quality–cost frontier, not on re-ranking the measured pool.
- Carry forward: anchored SparseEditFormer checkpoints, effect dataset SHA-256, S4 folds, min-max normalized NDCG convention.

## 15. COMMIT_SHA
- `e394a6e`

## 16. MANIFEST_AND_HASHES
- `artifacts/b0x/effect_dataset.jsonl` SHA-256: `f23a9fdd54a8ead90dccf793a073ccc6ed804a9b760a266c77b02e1fb1007ba1`
- `artifacts/f0x/f0x_results_5U-A1.json` SHA-256: `b141f7a54b3021ef842b7e1e140e0f428fb803ebe3ddb32378a02d47bb52be96`
- Honest measured values; no fabricated metrics.