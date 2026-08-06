# B0-X Effect Baseline Ceiling — mRNA-XEditFlow v1.1

- **Migration Goal:** `GOAL-XEDITFLOW-MIGRATION-01`
- **Prior terminal state:** `DATA_BENCHMARK_READY_FOR_EFFECT_MODEL`
- **Phase:** B0-X effect-baseline ceiling
- **Gate outcome:** `B0X_EFFECT_BASELINE_CEILING_ESTABLISHED`
- **UTC:** 2026-08-06
- **Worktree branch:** `xeditflow-migration-20260806T024650Z`
- **Staging (measured):** `/mnt/cunyuliu/mrna_editflow_v3_1/d1_3u_rebuild_staging/ordinary`
- **Canonical (read-only):** `/home/cunyuliu/mrna_editflow_goal/mrna_editflow/data/d1_canonical_records.jsonl`

---

## 1. FACTS_FROM_REPO
- Authority `mrna_xeditflow_goal_v1_1` registry `xeditflow_benchmark_registry.yaml` (v2.0) marks two effect-model benchmarks ACTIVE: `EditBench-5U-A1-Natural` (GSE114002, GSE217518) and `EditBench-3U-A1-Variant` (ENCSR854RUF, GSE186455, GSE200304, GSE232571, GSE232572, GSE261709, GSE298114). GSE246381 is SEALED and excluded.
- B0-X runs on the accepted EFFECT_PRIMARY assets only, with matched records / S4 split / endpoint / budget, per §14 of the prior `DATA_REBUILD_GATE.md`.

## 2. FACTS_FROM_CONTRACTS
- Contract `GOAL-XEDITFLOW-MIGRATION-01` authorizes B0-X effect baseline ceiling: build effect dataset (delta derivation per asset), run effect baselines (mean, source_mean, feature/k-mer ridge, XGBoost, small CNN, small Transformer, siamese/full-pair encoders) and search/optimization baselines (random_legal, greedy, exact_enumeration), with honest data (no fabrication).
- Conflict freezes locked: no GSE246381 label access before frozen evaluator; no reverse leakage; no modification of old canonical data; no GPU 4.

## 3. INFERENCES
- The source-relative effect task under S4 (leave-one-study-out) transfer is **hard**: nearly every from-sequence supervised baseline (feature_ridge, kmer_ridge, xgboost, difference_features, and all four NN encoders) lands at or **below chance** (macro spearman ≈ 0.00 to −0.03) on both benchmarks.
- The only clearly positive effect baseline is **abs_candidate** (predict the candidate's measured value from candidate features, then subtract the known source anchor): 5U-A1 macro spearman **+0.227** (signacc 0.510), 3U-A1 **+0.113** (signacc 0.579, best). The effect signal is dominated by the candidate's intrinsic measured magnitude, not by the delta/edit structure.
- `mean` and `source_mean` yield **undefined Spearman** (constant predictions) — reported honestly as `n/a`. Under S4, `source_mean` collapses to the global mean because a held-out study's source IDs never appear in the training fold.
- Search headroom (measured-candidate-pool): on 5U-A1 the exact-enumeration ceiling is macro NDCG@10 **0.527** and top-decile recall **1.0**, vs random_legal 0.276 / 0.508 — i.e. a real, measurable ranking ceiling exists. 3U-A1 has **no search space** (all 42,962 sources are singletons), so no measured-neighborhood ranking is possible there.

## 4. UNKNOWN_OR_BLOCKED
- `EditBench-5U-A2-Dense` and `EditBench-CDS-B1-Synonymous` remain DORMANT (no qualified assets); not fabricated.
- GSE246381 remains sealed; no row-level labels accessed.
- 3U-A1 search/optimization task is **not applicable** (all singleton sources) — reported honestly, not forced.
- Causation of the near-chance transfer (dataset heterogeneity vs. insufficient signal) is not decomposed here; it is a B0-X observable, not a claim.

## 5. FILES_READ
- `docs/execution/xeditflow_benchmark_registry.yaml`, `xeditflow_asset_role_assignment.yaml`, `reports/migration/DATA_REBUILD_GATE.md`, `reports/migration/FINAL_MIGRATION_REPORT.md`, `M4_DATA_READINESS.json`, main-repo `data/d1_canonical_records.jsonl`, rebuild staging `utr_edit_pairs.jsonl` / `functional_observations.jsonl`, reconstructed pairs for GSE232571 / GSE261709 / GSE298114, raw GSE114002 designed library and GSE200304 count files.

## 6. FILES_CHANGED
- `scripts/b0x/config.py`, `scripts/b0x/build_effect_dataset.py`, `scripts/b0x/features.py`, `scripts/b0x/run_effect_baselines.py`, `scripts/b0x/run_search_baselines.py`, `scripts/b0x/__init__.py` (new package).
- `tests/migration/test_b0x_effect_dataset.py`, `tests/migration/test_b0x_baselines.py` (new).
- `reports/migration/B0X_EFFECT_BASELINE_GATE.md` (this report).
- Artifacts `artifacts/b0x/`: `effect_dataset.jsonl`, `effect_dataset_manifest.json`, `effect_baseline_results.json`, `search_baseline_results.json`.

## 7. COMMANDS_RUN
- `python -m scripts.b0x.build_effect_dataset --out-dir artifacts/b0x`
- `CUDA_VISIBLE_DEVICES=0 python -m scripts.b0x.run_effect_baselines --dataset artifacts/b0x/effect_dataset.jsonl --out-dir artifacts/b0x`
- `python -m scripts.b0x.run_search_baselines --dataset artifacts/b0x/effect_dataset.jsonl --out-dir artifacts/b0x`
- `pytest tests/migration/test_b0x_effect_dataset.py tests/migration/test_b0x_baselines.py -q`
- NN baselines trained on CUDA (GPU 0, `torch.cuda.is_available()=True`); GPU 4 avoided.

## 8. TEST_RESULTS
- `tests/migration/test_b0x_effect_dataset.py` + `tests/migration/test_b0x_baselines.py`: **18/18 PASS** (delta derivation per asset type, feature shapes, context metrics, NDCG/oracle metrics, mean/source_mean, abs-minus-abs regression, empty-token edit-feature regression).

## 9. DATA_COUNTS_AND_DENOMINATORS
- Effect dataset total records: **106,659**; delta-defined records used by baselines: **103,199** (5U-A1 60,237; 3U-A1 42,962). Records with `delta=None` (source_anchor_unavailable / missing) excluded: 3,460.
- Per-asset (total / delta-defined): GSE114002 55,184/55,043; GSE217518 7,128/5,194; ENCSR854RUF 11,969/11,934; GSE186455 649/621; GSE200304 6,885/6,120; GSE232571 14,387/14,101; GSE232572 9,343/9,072; GSE261709 749/749; GSE298114 365/365.
- 5U-A1 S4 folds: {held-out GSE114002: train 5,194 / test 55,043}, {held-out GSE217518: train 55,043 / test 5,194}; 3U-A1: 7 folds (train 28,861–42,597 / test 365–14,101).

### Horizontal baseline comparison (macro over S4 folds; spearman `n/a` = undefined for constant predictors)
| baseline | 5U-A1 spearman | 5U-A1 signacc | 3U-A1 spearman | 3U-A1 signacc |
|---|---|---|---|---|
| mean | n/a | 0.4900 | n/a | 0.4282 |
| source_mean | n/a | 0.4900 | n/a | 0.4282 |
| feature_ridge | −0.0167 | 0.4918 | 0.0033 | 0.4333 |
| kmer_ridge | −0.0282 | 0.4883 | 0.0110 | 0.5036 |
| xgboost | −0.0337 | 0.4877 | −0.0279 | 0.4953 |
| difference_features | 0.0016 | 0.4917 | 0.0021 | 0.4226 |
| **abs_candidate** | **0.2272** | **0.5100** | **0.1129** | **0.5789** |
| abs_candidate_minus_abs_source | −0.0235 | 0.4965 | −0.0226 | 0.4878 |
| small_cnn | −0.0029 | 0.4967 | 0.0068 | 0.4461 |
| small_transformer | −0.0186 | 0.4964 | −0.0009 | 0.4688 |
| siamese_encoder | −0.0106 | 0.5031 | −0.0076 | 0.5063 |
| full_pair_encoder | −0.0070 | 0.5071 | −0.0015 | 0.5188 |

### Search baseline ceiling (measured-candidate-pool ranking, macro)
| baseline | 5U-A1 macro NDCG@10 | 5U-A1 top-decile recall | 3U-A1 |
|---|---|---|---|
| random_legal | 0.276 | 0.508 | n/a (0 non-singleton sources) |
| greedy | 0.276 | 0.508 | n/a |
| **exact_enumeration** | **0.527** | **1.0** | n/a |

## 10. REUSE_DECISIONS
- D1 measured staging + read-only canonical: REUSE_AS_IS (B0-X reads only; no modification).
- Rebuilt reconstructed pairs (GSE232571/GSE261709/GSE298114): REUSE_AS_IS as the sequence-text source for those assets.
- Raw GSE114002 library / GSE200304 count: REUSE_AS_IS for source-anchor derivation.
- Feature families mirror `core/p3_02_delta_oracle.py`; kept local to `scripts/b0x` to avoid import coupling.

## 11. GATE_STATUS
- **PASS → `B0X_EFFECT_BASELINE_CEILING_ESTABLISHED`.** All 12 effect baselines and all applicable search baselines executed on the accepted measured assets with matched S4 split/endpoint/seed; every baseline is reported with its honest measured value (including undefined-spearman constant baselines and the no-search-space 3U-A1 case). No fabricated PASS.

## 12. CLAIMS_UNLOCKED
- An honest effect-model ceiling is now established: supervised from-sequence baselines are at chance under S4 transfer, while `abs_candidate` (candidate-intrinsic + source anchor) is the strongest effect predictor (5U-A1 0.227, 3U-A1 0.113 Spearman). A measured-space ranking ceiling (exact NDCG@10 0.527) exists on 5U-A1. These are inputs for the effect-model (M4 SparseEditFormer) and measured-neighborhood optimization phases.

## 13. CLAIMS_STILL_PROHIBITED
- No L4 real biological/therapeutic improvement claim. No claim that any baseline "improves" TE/stability/expression without the predicted/internal proxy qualifier. No claim of a learned effect model beating the ceiling yet (no trained effect model this phase). GSE246381 final labels not opened. CDS-B1 / 5U-A2 not auto-unlocked.

## 14. NEXT_PHASE_INPUTS
- Feed the B0-X ceiling into the effect-model phase (M4 SparseEditFormer) as the target to exceed on `EditBench-5U-A1-Natural` and `EditBench-3U-A1-Variant` (T5_SOURCE_RELATIVE_EFFECT, T5_SELECTIVE_EFFECT, T5_MEASURED_NEIGHBORHOOD_OPTIMIZATION, T5_FIXED_BUDGET_MULTI_STEP_OPTIMIZATION; 3U-A1: T3_EFFECT_TRANSFER, CROSS_REGION_TRANSFER). Then O0-X → F0-X → G0-X → G1-X → E0-X → X0-X.

## 15. COMMIT_SHA
- Worktree commit for this B0-X phase (see commit message; contract `GOAL-XEDITFLOW-MIGRATION-01`).

## 16. MANIFEST_AND_HASHES
- `artifacts/b0x/effect_dataset.jsonl` SHA-256: `f23a9fdd54a8ead90dccf793a073ccc6ed804a9b760a266c77b02e1fb1007ba1`.
- `effect_dataset_manifest.json`, `effect_baseline_results.json`, `search_baseline_results.json` committed alongside.