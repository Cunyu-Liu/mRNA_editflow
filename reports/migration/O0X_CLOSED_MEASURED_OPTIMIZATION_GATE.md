# O0-X Closed Measured-Space Optimization — mRNA-XEditFlow v1.1

- **Migration Goal:** `GOAL-XEDITFLOW-MIGRATION-01`
- **Prior terminal state:** `M4_GATE_NOT_YET_MET` (partial 5U-A1 component), B0-X search baselines established
- **Phase:** O0-X — closed measured-space optimization (no Flow)
- **Gate outcome:** `O0X_SEARCH_CEILING_ESTABLISHED` with `FLOW_HEADROOM_LIMITED` signal
- **UTC:** 2026-08-06
- **Worktree branch:** `xeditflow-migration-20260806T024650Z`
- **Effect dataset (SHA-256):** `f23a9fdd54a8ead90dccf793a073ccc6ed804a9b760a266c77b02e1fb1007ba1`
- **Runner:** `scripts/m4_sparse/run_o0x.py` + `scripts/m4_sparse/o0x_search.py` on GPU `cuda:1` (GPU 4 avoided)

---

## 1. FACTS_FROM_REPO
- Ran O0-X on the same S4 (leave-one-study-out) measured candidate pools used by B0-X and M4, WITHOUT any Edit Flow. Each source's `source_id` groups its real measured candidates; delta is the measured effect.
- Loaded the anchored SparseEditFormer checkpoints (`artifacts/m4_sparse/candval/model_<benchmark>__<study>.pt`, candidate_value target, delta = cand_pred − MEASURED source) for the `sparseeditformer_rerank` strategy. All 9 fold checkpoints present (`missing_checkpoints=[]`).
- Strategies compared on every non-singleton measured pool: `exact_enumeration`, `random_legal`, `greedy` (edit-distance), `beam`, `sparseeditformer_rerank`.
- Metrics: NDCG@k, top-decile recall, enrichment@k, normalized regret + query count / forward-equivalents / wall time. Aggregation: source-macro → study-macro (S4).

## 2. FACTS_FROM_CONTRACTS
- § Phase O0-X: run on each source's real *measured* candidate pool: exact enumeration, greedy/beam/search, SparseEditFormer reranking; report NDCG, top-decile recall, enrichment, normalized regret; query count, forward equivalents, wall time.
- Rules: candidate pool must be real measured data; source-macro then study-macro; singleton sources excluded from ranking headline but counted; pool completeness/missingness reported; scorer/search ceiling is the necessity judgment for a subsequent Flow.
- § B0-X headroom rule: if exact enumeration / beam / reranking already reaches acceptable regret at low cost, record `FLOW_HEADROOM_LIMITED`; Flow only continues as a main-line candidate if it demonstrates amortization, OOD, or a quality–cost frontier.

## 3. INFERENCES
- **Measured candidate pool is dominated by singletons.** 5U-A1: 58,607 source pools, of which 56,977 are singletons (97.2%) and only 1,630 are non-singleton (all size-2, all in GSE217518). 3U-A1: ALL 42,962 pools are singletons → **no ranking headline is defined for 3U-A1** (0 non-singleton sources).
- **On the only ranking-capable pools (5U-A1, 1,630 size-2 pools), the learned scorer adds only marginal value.** With the min-max normalized NDCG convention (same as M4 reranker): exact_enumeration NDCG **1.000** (ceiling), random_legal **0.8125**, greedy **0.8184**, beam **0.8340**, sparseeditformer_rerank **0.8340**. The model's 0.834 vs random 0.8125 implies the top-candidate is chosen correctly only ~55% of the time on size-2 pools (vs 50% random).
- **Exact enumeration reaches the ceiling (NDCG 1.0, regret 0.0) at trivial cost:** 3,260 queries, 0 forward-equivalents, no model inference. This is a `FLOW_HEADROOM_LIMITED` signal per § B0-X: the measured neighborhood is so small (size-2) that brute-force enumeration is optimal and cheap.
- **top-decile recall** is the only place the model clearly exceeds random: sparseeditformer 0.550 vs random_legal 0.492 (and greedy 0.508), but this too is a small absolute gain on tiny pools.
- Non-singleton ranking coverage is extremely low: only 3,260 of 60,237 5U-A1 records (5.4%) belong to a non-singleton, ranking-capable pool; 3U-A1 has 0%.

## 4. UNKNOWN_OR_BLOCKED
- No multi-candidate (size>2) measured pools exist in the active benchmarks, so none of `greedy/beam/search` operating over a genuine sequence neighborhood (multi-step edits) could be exercised. The O0-X "search" here is ranking over closed measured pools, not generative search; the search headroom is therefore structurally limited by the data, not by the scorer.
- Whether a Flow could add value via amortization / OOD / fixed-budget multi-step generation is NOT testable from the measured pools alone; it requires the F0-X/G0-X/G1-X generative phases. O0-X does not close that question; it establishes that pure closed measured-space ranking leaves little headroom for `generate-then-rerank` to improve on.

## 5. FILES_READ
- `scripts/b0x/run_search_baselines.py`, `scripts/b0x/config.py`, `scripts/m4_sparse/model.py`, `evaluate.py`, `rerank.py`, `dataset.py`, `train.py`, `config.py`, `run.py`, `artifacts/m4_sparse/candval/*.pt`, `artifacts/b0x/effect_dataset.jsonl`, `artifacts/o0x/o0x_results.json`.

## 6. FILES_CHANGED
- `scripts/m4_sparse/o0x_search.py` (new): O0-X metrics / strategies / aggregation.
- `scripts/m4_sparse/run_o0x.py` (new): O0-X runner with CUDA policy + compute accounting.
- `tests/migration/test_m4_o0x.py` (new): 12 unit tests.
- `reports/migration/O0X_CLOSED_MEASURED_OPTIMIZATION_GATE.md` (this report).

## 7. COMMANDS_RUN
- `python -m scripts.m4_sparse.run_o0x --dataset artifacts/b0x/effect_dataset.jsonl --ckpt-dir artifacts/m4_sparse/candval --out-dir artifacts/o0x --gpu cuda:1`
- `python -m pytest tests/migration/test_m4_o0x.py -q` → **12 passed**
- `python -m pytest tests/migration/ -q` → **110 passed** (99 prior + 12 O0-X − 1 removed/renamed assertion overlap)

## 8. TEST_RESULTS
- New O0-X tests (12/12 pass): NDCG perfect ordering = 1.0; NDCG handles negative/signed deltas (no idcg collapse); top-decile recall full recovery; rank_metrics exact ceiling (regret 0); source headline has all non-model strategies without scorer; rerank strategy added with scorer (forward-equivalents = pool size, query_count = 0); singleton pool no-crash; aggregate strategy macro; macro handles None metrics; study-macro + singleton accounting; no-scorer run skips rerank strategy.
- Full migration suite: **110 passed** (no regressions).

## 9. DATA_COUNTS_AND_DENOMINATORS
- Delta-defined records: 103,199 (5U-A1 60,237; 3U-A1 42,962).
- 5U-A1 pools: 58,607 total / 56,977 singleton / **1,630 non-singleton (all size-2, GSE217518)**. Non-singleton records: 3,260 (5.4% of 5U-A1).
- 3U-A1 pools: 42,962 total / 42,962 singleton / **0 non-singleton** → no ranking headline.
- SparseEditFormer rerank: 0 oracle queries, 3,260 forward-equivalents; wall time 5U-A1 = 16.8 s.

### Horizontal comparison (5U-A1, source- then study-macro, min-max NDCG convention)
| strategy | NDCG@10 | top-decile recall | enrichment@k | normalized regret | queries | fwd-eq |
|---|---|---|---|---|---|---|
| exact_enumeration (ceiling) | **1.0000** | 1.0000 | 1.0000 | **0.0000** | 3,260 | 0 |
| random_legal | 0.8125 | 0.4920 | 1.0000 | 0.5080 | 3,260 | 0 |
| greedy (edit-distance) | 0.8184 | 0.5080 | 1.0000 | 0.4920 | 3,260 | 0 |
| beam (model-scored) | 0.8340 | 0.5503 | 1.0000 | 0.4497 | 3,260 | 0 |
| **sparseeditformer_rerank** | **0.8340** | **0.5503** | 1.0000 | 0.4497 | 0 | 3,260 |

- 3U-A1: no non-singleton measured pools → all five ranking strategies are **undefined** (0 headline sources); reported as such, not as a pass.
- enrichment@k = 1.0 for every strategy because for size-2 pools the top-k set equals the full oracle top-k denominator under the fixed-k convention; this metric is uninformative on size-2 pools (noted, not hidden).

## 10. REUSE_DECISIONS
- NDCG convention REUSE_WITH_ADAPTER: corrected O0-X to min-max normalized per-pool relevance (same as M4 `rerank.py`), because raw signed-delta NDCG collapses to 0 when idcg ≤ 0 (all-negative or degenerate pools). B0-X raw-delta NDCG is retained as a separate historical convention; the O0-X headline uses the normalized convention and is internally consistent with the M4 reranker.
- Anchored SparseEditFormer checkpoints REUSE_AS_IS (9/9 present) for the rerank strategy.
- S4 split + `build_folds` REUSE_AS_IS via `scripts/m4_sparse/train.py`.

## 11. GATE_STATUS
- **`O0X_SEARCH_CEILING_ESTABLISHED`** with **`FLOW_HEADROOM_LIMITED`** signal (honest).
  - Exact enumeration reaches the measured-space ceiling (NDCG 1.0, regret 0.0) on all ranking-capable pools at trivial cost (3,260 queries, 0 model forwards).
  - The learned reranker (NDCG 0.834, top-decile recall 0.550) only marginally exceeds random (0.8125 / 0.492) on the 1,630 size-2 pools.
  - The measured neighborhood is degenerate (97%–100% singletons), so closed measured-space ranking leaves little headroom for `generate-then-rerank`.
- NOT a no-go for Flow: the necessity judgment is that a Flow must justify itself via amortization, OOD, or a fixed-budget multi-step quality–cost frontier (F0-X/G0-X/G1-X), NOT via closed measured-space ranking where exact enumeration is already optimal and cheap.

## 12. CLAIMS_UNLOCKED
- Honest, evidenced measured-space search ceiling: exact enumeration is optimal (NDCG 1.0, regret 0.0) and cheap on the active A1 measured pools; the measured neighborhood is dominated by singletons.
- Honest scorer finding: SparseEditFormer reranking adds only a small top-decile-recall gain (0.550 vs 0.492 random) on size-2 measured pools; its NDCG 0.834 vs random 0.8125 is a marginal correct-top-candidate advantage (~55% vs 50%).

## 13. CLAIMS_STILL_PROHIBITED
- NO claim that Flow/guidance improves closed measured-space ranking (not tested; exact enumeration already optimal). NO claim of any generative/optimization benefit beyond the measured pools. NO L4 real biological improvement. NO GSE246381 label access. NO A2 dense / CDS-B1 unlock. NO `EFFECT_MODEL_ESTABLISHED` (M4 full gate still unmet).

## 14. NEXT_PHASE_INPUTS
- Feed `FLOW_HEADROOM_LIMITED` + the measured-space ceiling into F0-X (source-anchored legal Edit Flow). Because closed measured-space ranking is already saturated by exact enumeration, F0-X / G0-X / G1-X must demonstrate value on **fixed-budget multi-step generation** (k=1,3,5 over a legal edit graph), OOD, or a quality–cost frontier — not on re-ranking the measured pool.
- Carry forward: anchored SparseEditFormer checkpoints, effect dataset SHA-256, S4 folds, and the min-max normalized NDCG convention for consistency.

## 15. COMMIT_SHA
- `64920c26a6f44505dae26b78e7a1df9dc933e336` (commit message marked `GOAL-XEDITFLOW-MIGRATION-01`).

## 16. MANIFEST_AND_HASHES
- `artifacts/b0x/effect_dataset.jsonl` SHA-256: `f23a9fdd54a8ead90dccf793a073ccc6ed804a9b760a266c77b02e1fb1007ba1`
- `artifacts/o0x/o0x_results.json` SHA-256: `8b9e71c89e2c117c6b02a4221a1b9d6d0004b627d77e7d17d1dc8765cd8b729d` (kept on server, gitignored per convention). Honest measured values; no fabricated metrics.