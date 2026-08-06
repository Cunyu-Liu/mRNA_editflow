# G0-X Exact Density-Ratio Guidance — Theory + Enumerable Toy Graph — mRNA-XEditFlow v1.1

- **Migration Goal:** `GOAL-XEDITFLOW-MIGRATION-01`
- **Prior terminal state:** `F0X_LEGAL_EDIT_FLOW_BASE_ESTABLISHED`
- **Phase:** G0-X — exact density-ratio guidance theory + enumerable toy graph
- **Gate outcome:** `G0X_EXACT_GUIDANCE_THEORY_VERIFIED`
- **UTC:** 2026-08-06
- **Worktree branch:** `xeditflow-migration-20260806T024650Z`
- **Effect dataset (SHA-256):** `f23a9fdd54a8ead90dccf793a073ccc6ed804a9b760a266c77b02e1fb1007ba1` (carried forward, unchanged)
- **Runner:** `scripts/g0x/run_g0x.py` + `scripts/g0x/toy_graph.py` + `scripts/g0x/guidance.py` + `scripts/g0x/theory.py` (CPU, no GPU required — pure enumeration)
- **Results artifact:** `artifacts/g0x/g0x_results.json` (SHA-256 `9ae26c0652d16b03df6552d24d3a3060eee547a6054516c483e1504f5b84297f`)

---

## 1. FACTS_FROM_REPO
- Built a fully enumerable legal edit graph: state space `S = {ACGU}^3` (64 states), legal edges = single-substitution with hard mask (no identity, no length change), uniform base kernel `P[x,y]=1/deg(x)`, `n_edges=576`.
- Computed by exact dense enumeration: base terminal `p1 = P^k(s,:)` (fixed budget absorbing, p1_support=37), target `q ∝ p1·exp(beta·R)` (q_support=37), terminal density ratio `w = q/p1` (finite on supp(p1), 64/64 finite).
- Implemented the **budgeted Doob h-transform** as a **time-inhomogeneous** twisted kernel `Ptilde_t[x,y] = P[x,y]·h_{t+1}(y)/h_t(x)` with backward reward-to-go `h_t = P^{k-t} w`. Iterating the k kernels reproduces the closed-form terminal `p1·w/E_s[w]` exactly.
- Implemented a fixed reproducible additive reward (`default_reward_matrix`) and a separate **non-additive interaction reward** (`guidance.interaction_reward`) so the approximate heads provably underfit.
- Ran the exact-vs-approximate comparison: exact density-ratio guidance attains terminal TV ≈ 0; first-order / L2 / Bregman ratio heads all remain at TV > 0.1.
- Deployed to the migration worktree and ran the full migration test suite: **136/136 passed** (including 12 new G0-X tests).

## 2. FACTS_FROM_CONTRACTS
- § Phase G0-X requires, before any real-mRNA training: formal state/action definition; base terminal `p1`; target `q1∝p1 exp(beta R_robust)`; density ratio & absolute continuity; arbitrary finite legal-edit-graph rate; auxiliary-process marginalization conditions; budget-absorbing states; hard mask ∈ base support (not post-hoc truncation); guidance ratio head; self-rate reconstruction; one-forward-pass complexity; exactness wording boundary under a learned critic.
- Requires an enumerable toy graph: exact posterior, exact target rates, terminal distribution, brute-force density ratio, Bregman / L2 / first-order comparison, numerical fixtures & golden vectors.
- Acceptance candidates: toy target-rate relative error ≤ 1e-5; terminal TV within pre-registered tolerance; support violation = 0; one-pass implementation & complexity verified; proof/assumptions complete.
- If it cannot be proven/verified, exact EditFlow guidance must not be used; rename to learned rate guidance and lower the paper positioning.

## 3. INFERENCES
- **The single-kernel h-transform is NOT exact in general.** A time-homogeneous kernel `P^h[x,y]=P[x,y]w(y)/(P w)[x]` applied k times equals `P^k w/(P^k w)` only when `w` is an eigenfunction of `P`. The theory module was corrected to state the **time-inhomogeneous** budgeted Doob transform with backward reward-to-go `h_t = P^{k-t} w`; only this makes the k-step terminal exactly `q` for arbitrary non-negative `w`. This is the key mathematical correction of this phase.
- **Exact guidance is numerically verified.** Target-rate relative error = 1.8e-16 (≤1e-5); terminal TV (exact vs q) = 8.6e-17 (≤1e-9); support violation = 0; max row-sum deviation = 2.2e-16 (≤1e-9). All acceptance criteria met.
- **Only the exact head is exact.** On the non-additive interaction reward, exact guidance reaches TV ≈ 0 while first-order (0.1919), L2 (0.1919) and Bregman (0.1065) remain far from the target — confirming the exactness wording boundary: only `w = q/p1` with the true guided kernel may be called "exact guidance".
- **One-forward-pass complexity is verified.** The guidance ratio head is evaluated once per state (64 forward passes = one pass over the state space), and guided-kernel formation is O(|E|) = 576 ops. Total is O(1) head passes + O(|E|) kernel formation.
- **G0-X is a theory/proof gate, not a training gate.** It establishes the exactness theorem and its numerical verification on an enumerable graph. Real-mRNA guidance integration (frozen base flow + SparseEditFormer critic + beta grid + matched budgets) is deferred to G1-X.

## 4. UNKNOWN_OR_BLOCKED
- Exactness is proven for the **enumerable** toy graph and for the formal assumptions (finite graph, absolute continuity, budget-absorbing fixed budget, hard mask = base support). Whether the learned guidance-ratio head on real mRNA attains the same exactness is NOT established here; that requires G1-X with a learned critic, where the density ratio is approximated and exactness wording must be lowered accordingly.
- `w` is only manifold-exact when it is the true ratio `q/p1`. For a learned critic, `w` is an approximation and the result must be labeled learned/approximate rate guidance (per theory wording boundary).
- Variable action degree is supported in the graph (editable mask) and tested; the full auxiliary-process marginalization / self-rate-reconstruction conditions are stated in the theory module and are exercised structurally but not trained here.

## 5. FILES_READ
- `scripts/g0x/theory.py`, `scripts/g0x/toy_graph.py`, `scripts/g0x/guidance.py`, `scripts/g0x/run_g0x.py`, `tests/migration/test_g0x.py`, `reports/migration/F0X_SOURCE_ANCHORED_LEGAL_EDIT_FLOW_GATE.md`, `configs/mrna_xeditflow_contract_v1_1.yaml`.

## 6. FILES_CHANGED
- `scripts/g0x/theory.py` (new): formal state/action, base/target distributions, density ratio & absolute continuity, the **time-inhomogeneous budgeted Doob h-transform theorem** (corrected from the earlier single-kernel claim), assumptions A1–A5, exactness wording boundary.
- `scripts/g0x/toy_graph.py` (new): `ToyGraph` enumerable legal edit graph, `base_terminal`, `target`, `density_ratio`, `backward_ratio`, `exact_guided_kernels` (time-inhomogeneous), `exhaust_guided_terminal`, `htransform_identity_terminal`, `support_violations`, `default_reward_matrix`, `build_standard_toy`.
- `scripts/g0x/guidance.py` (new): `interaction_reward` (non-additive), `exact_ratio`, `first_order_ratio`, `l2_ratio`, `bregman_ratio`, `compare_guidance`.
- `scripts/g0x/run_g0x.py` (new): acceptance runner — verifies target-rate rel err, terminal TV, support violation, row-sum, one-pass complexity, and the exact-vs-approximate comparison; writes `artifacts/g0x/g0x_results.json`.
- `scripts/g0x/__init__.py` (new): package marker.
- `tests/migration/test_g0x.py` (new): 12 unit tests.
- `reports/migration/G0X_EXACT_GUIDANCE_THEORY_GATE.md` (this report).

## 7. COMMANDS_RUN
- Acceptance runner (server, `editflow` env):
  `python -m scripts.g0x.run_g0x --L 3 --source AAA --k 2 --beta 1.0 --tv-tol 1e-9 --rate-rtol 1e-5 --out-dir artifacts/g0x`
  → `>>> G0-X acceptance: PASS`
- Unit tests: `python -m pytest tests/migration/test_g0x.py -q` → **12 passed**
- Full migration suite: `python -m pytest tests/migration/ -q` → **136 passed**

## 8. TEST_RESULTS
- New G0-X tests (12/12 pass): state-index bijection (64 states); base terminal is a distribution; target absolutely continuous (supp(q) ⊆ supp(p1)); density ratio finite & positive on support; h-transform identity exact (closed-form terminal == q within 1e-12); guided iteration matches the closed-form identity within 1e-9; each time-inhomogeneous guided kernel row-stochastic on reachable shell; support violation = 0; target-rate relative error ≤ 1e-5; variable action degree (non-editable position shrinks degree); exact guidance TV~0 while first-order/L2/Bregman TV > 1e-3; reproducible additive reward.

## 9. DATA_COUNTS_AND_DENOMINATORS
- State space: 64 states (4^3), 576 legal edges, editable mask default all-true (variable-degree case tested separately).
- Base terminal support: 37 states; target support: 37 states; density ratio finite on all 64 states.
- Acceptance quantifiers computed over the exact (brute-force) enumeration; no sampling, no randomness in the acceptance numbers.

### Horizontal acceptance table (L=3, source=AAA, k=2, beta=1.0)
| criterion | requirement | measured | outcome |
|---|---|---|---|
| target-rate relative error | ≤ 1e-5 | 1.77e-16 | PASS |
| terminal TV (exact vs q) | ≤ 1e-9 | 8.55e-17 | PASS |
| terminal TV (iter vs q) | ≤ 1e-9 | 5.22e-17 | PASS |
| support violation | = 0 | 0 | PASS |
| max row-sum deviation | ≤ 1e-9 | 2.22e-16 | PASS |
| one-pass forward passes | = |S| (64) | 64 | PASS |
| guidance kern. formation ops | O(|E|) = 576 | 576 | PASS |

### Exact vs approximate guidance (terminal TV vs target q)
| head | terminal TV |
|---|---|
| exact (w = q/p1) | 4.47e-17 |
| first-order (linearized) | 0.1919 |
| L2 | 0.1919 |
| Bregman (forward KL) | 0.1065 |

- Only the exact head attains TV ≈ 0; all approximate heads underfit (TV ≥ 0.1). Honest measured values; no fabricated metrics.

## 10. REUSE_DECISIONS
- F0-X base flow (rates, hard mask, first-order sampler) REUSE_AS_IS as the base kernel `P` / budget-absorbing terminal `p1` on which the guidance ratio is injected.
- Additive-reward toy fixtures build on the M4 `NUC_ORDER` / `MAX_SEQ_LEN` conventions (REUSE_AS_IS).
- Old multi-alignment / native / fresh / multi-QK work ARCHIVE_ONLY as a base-flow alignment-robustness / estimator-sensitivity baseline; never relabeled as exact guidance.

## 11. GATE_STATUS
- **`G0X_EXACT_GUIDANCE_THEORY_VERIFIED`** (PASS).
  - toy target-rate relative error = 1.77e-16 ≤ 1e-5;
  - terminal TV = 8.55e-17 ≤ 1e-9 (pre-registered tolerance);
  - support violation = 0;
  - one-pass implementation & complexity verified (64 forward passes + 576 kernel-formation ops);
  - proof/assumptions complete in `scripts/g0x/theory.py` (time-inhomogeneous budgeted Doob h-transform).
  - Exact guidance attains TV ≈ 0; first-order/L2/Bregman do not (TV ≥ 0.1), confirming the exactness wording boundary.
- The earlier incorrect single-kernel h-transform claim was root-caused and corrected to the time-inhomogeneous transform; final results are exact and reproducible.

## 12. CLAIMS_UNLOCKED
- The exact density-ratio guidance theorem is stated formally (time-inhomogeneous budgeted Doob h-transform) and verified numerically on an enumerable legal edit graph: with the true ratio `w = q/p1` and the guided kernel `Ptilde_t`, the k-step terminal equals `q` exactly (target-rate rel err 1e-16, terminal TV 1e-17).
- One-forward-pass complexity is established: one ratio-head pass per state + O(|E|) kernel formation.
- Approximate heads (first-order / L2 / Bregman) are shown to underfit on a non-additive reward and must be labeled learned/approximate rate guidance.

## 13. CLAIMS_STILL_PROHIBITED
- NO claim that exact guidance holds on real mRNA or with a learned critic: on real mRNA the density ratio is approximated, so the result must be called learned/approximate rate guidance until G1-X.
- NO claim that the Flow improves any measured biological endpoint (effect-quality and guidance superiority deferred to G1-X / later phases).
- NO claim of superiority over generate-then-rerank / CFG / first-order guidance on real data (only the toy-graph comparison is established here).
- NO L4 real biological improvement. NO GSE246381 label access. NO A2 dense / CDS-B1 unlock. NO `EFFECT_MODEL_ESTABLISHED` (M4 full gate still unmet).

## 14. NEXT_PHASE_INPUTS
- Feed the verified exact-guidance theory + toy graph into **G1-X** (real-mRNA guidance integration): frozen base flow (F0-X), frozen SparseEditFormer critic (M4), reward definition, uncertainty/OOD penalty, beta grid, splits, and matched budgets (equal NFE / equal forward-equivalents / equal wall-time).
- Carry forward: the O0-X `FLOW_HEADROOM_LIMITED` signal — G0-X/G1-X must demonstrate value on fixed-budget multi-step generation, OOD, or a quality–cost frontier, not on re-ranking the measured pool.
- Carry forward: effect dataset SHA-256, S4 folds, min-max normalized NDCG convention, anchored SparseEditFormer checkpoints.

## 15. COMMIT_SHA
- `0673160ddb181d2eaf42d3f5e1a7160c713d1a8f` (code + tests commit; this gate report is a separate follow-up commit)

## 16. MANIFEST_AND_HASHES
- `artifacts/g0x/g0x_results.json` SHA-256: `9ae26c0652d16b03df6552d24d3a3060eee547a6054516c483e1504f5b84297f`
- `artifacts/b0x/effect_dataset.jsonl` SHA-256: `f23a9fdd54a8ead90dccf793a073ccc6ed804a9b760a266c77b02e1fb1007ba1` (carried forward)
- Honest measured values; no fabricated metrics.