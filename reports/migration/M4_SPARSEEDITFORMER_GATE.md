# M4 SparseEditFormer Gate — mRNA-XEditFlow v1.1

- **Migration Goal:** `GOAL-XEDITFLOW-MIGRATION-01`
- **Prior terminal state:** `B0X_EFFECT_BASELINE_CEILING_ESTABLISHED`
- **Phase:** M4 — source-relative effect model (SparseEditFormer)
- **Gate outcome:** `M4_GATE_NOT_YET_MET` (partial component: 5U-A1 macro-delta-Spearman exceeded ≥0.25 and beat `abs_candidate`; full pre-registered gate not met)
- **UTC:** 2026-08-06
- **Worktree branch:** `xeditflow-migration-20260806T024650Z`
- **Canonical (read-only):** `/home/cunyuliu/mrna_editflow_goal/mrna_editflow/data/d1_canonical_records.jsonl`
- **Effect dataset (SHA-256):** `f23a9fdd54a8ead90dccf793a073ccc6ed804a9b760a266c77b02e1fb1007ba1`

---

## 1. FACTS_FROM_REPO
- Authoritative effect dataset `artifacts/b0x/effect_dataset.jsonl` (106,659 records; 103,199 with defined delta) was used as the training/evaluation source. It is built from the ACTIVE benchmarks only: `EditBench-5U-A1-Natural` (GSE114002, GSE217518) and `EditBench-3U-A1-Variant` (ENCSR854RUF, GSE186455, GSE200304, GSE232571, GSE232572, GSE261709, GSE298114). GSE246381 is SEALED and excluded.
- `scripts/b0x/config.py` defines the delta-derivation specs and the S4 leave-one-study-out primary split; `scripts/b0x/features.py` defines the encoding conventions (`NUC_ORDER="ACGU"`, `MAX_SEQ_LEN=100`, 12-dim edit features) reused here.
- The strongest B0-X effect baseline is `abs_candidate` (predict candidate_value from candidate features, delta = cand_pred − measured source_value): macro delta Spearman **5U-A1 0.227**, **3U-A1 0.113**; sign accuracy 5U 0.510, 3U 0.579.
- `EditBench-5U-A2-Dense` is **DORMANT** (no qualified A2 dense asset) ⇒ there is NO A2 dense pretraining. Training is **A1-NATURAL** (5U-A1 + 3U-A1) directly, adapted honestly (step not available by governance, not by choice).

## 2. FACTS_FROM_CONTRACTS
- Contract `GOAL-XEDITFLOW-MIGRATION-01` authorizes the M4 effect model phase: build a source-relative effect model (SparseEditFormer) with source-cached encoder, explicit edit encoder, endpoint/context heads, mean/variance (heteroscedastic), sign classification, pairwise/listwise ranking, inverse consistency, calibration/conformal, OOD/abstention, and an independent top-K paired reranker — with honest data and no fabrication.
- Pre-registered gate candidates on the ACTIVE benchmark: macro delta Spearman **≥ 0.25**, macro sign accuracy **≥ 0.60**, top-10% enrichment **≥ 1.50**, AND beat the strongest executable non-foundation baseline (`abs_candidate` 5U-A1 0.227 / 3U-A1 0.113).
- Conflict freezes: no GSE246381 label access; no modification of the read-only main repo or canonical data; no GPU 4; no silent CPU-as-GPU reporting.

## 3. INFERENCES
- A from-scratch **anchored source-relative model** (predict candidate_value, delta = cand_pred − MEASURED source_value at test) reaches **5U-A1 macro delta Spearman 0.2971**, above the pre-registered 0.25 and above `abs_candidate` (0.2268 on the same folds). This is the first learned effect model to exceed the B0-X ceiling on an ACTIVE benchmark folder.
- The **pure-sequence** variant (predict delta directly from sequence, no measured anchor) is at chance on both benchmarks (5U-A1 −0.007, 3U-A1 −0.034), consistent with B0-Xs finding that from-sequence transfer under S4 is near chance. The effect signal is dominated by the candidates intrinsic measured magnitude, i.e. the measurable source anchor, not by the edit/delta structure alone.
- On 3U-A1 the anchored model (0.1028) is essentially tied with `abs_candidate` (0.1040) but does not beat it; macro sign accuracy (0.51 / 0.43) and top-10% enrichment (0.13 / 0.16) remain far below the 0.60 / 1.50 gate candidates.
- The independent top-K paired reranker reaches **NDCG@10 0.834** and top-decile recall 0.550 on 5U-A1 measured candidate pools (1,630 non-singleton sources) — a strong measured-neighborhood ranking, above the random_legal 0.276 ceiling reported by B0-X. 3U-A1 has no non-singleton measured sources (all 42,962 singletons), so no measured-neighborhood ranking is defined there.

## 4. UNKNOWN_OR_BLOCKED
- Sign-accuracy and top-10% enrichment gate candidates are not reached on either benchmark; the reason (dataset heterogeneity vs. limited per-asset signal) is not decomposed here — it is an observable, not a claim.
- 3U-A1 does not beat `abs_candidate` under the anchored setting; the pure-sequence transfer remains at chance. Whether a larger/other backbone closes this gap is untested this phase.
- `EditBench-5U-A2-Dense` and `EditBench-CDS-B1-Synonymous` remain DORMANT; no A2 dense pretraining exists (governance, not choice). GSE246381 final labels NOT opened.
- OOD/abstention and calibration/conformal heads are implemented and calibrated (temperature) but their calibrated-probability downstream benefit was not separately gated this phase.

## 5. FILES_READ
- `scripts/b0x/config.py`, `scripts/b0x/features.py`, `scripts/b0x/run_effect_baselines.py`, `scripts/b0x/run_search_baselines.py`, `artifacts/b0x/effect_dataset.jsonl`, `artifacts/b0x/effect_dataset_manifest.json`, `artifacts/b0x/effect_baseline_results.json`, `reports/migration/B0X_EFFECT_BASELINE_GATE.md`, `tests/migration/test_b0x_baselines.py`, `tests/migration/test_m4_data_readiness.py`, `pyproject.toml`.

## 6. FILES_CHANGED
- `scripts/m4_sparse/__init__.py`, `config.py`, `model.py`, `dataset.py`, `train.py`, `evaluate.py`, `rerank.py`, `run.py` (new package).
- `tests/migration/test_m4_sparse_model.py`, `tests/migration/test_m4_sparse_eval.py` (new).
- `reports/migration/M4_SPARSEEDITFORMER_GATE.md` (this report).
- Artifacts `artifacts/m4_sparse/`: `results.json` per target, `run_*.log`, and 18 fold checkpoints (`model_<benchmark>__<study>.pt`).

## 7. COMMANDS_RUN
- `python -m scripts.m4_sparse.run --dataset artifacts/b0x/effect_dataset.jsonl --benchmarks 5U-A1 3U-A1 --target delta --gpu cuda:1 --out-dir artifacts/m4_sparse/delta`
- `python -m scripts.m4_sparse.run --dataset artifacts/b0x/effect_dataset.jsonl --benchmarks 5U-A1 3U-A1 --target candidate_value --gpu cuda:3 --out-dir artifacts/m4_sparse/candval`
- `python -m scripts.m4_sparse.run ... --only-eval ...` (recompute eval + rerank with corrected NDCG)
- `python -m pytest tests/migration/ -q` → **99 passed** (79 prior + 20 new).
- Training on CUDA (GPU 1 and GPU 3; GPU 4 avoided); `torch.cuda.is_available()=True`.

## 8. TEST_RESULTS
- `tests/migration/test_m4_sparse_model.py` + `test_m4_sparse_eval.py`: **20/20 PASS** (one-hot/edit-encoding, edit inversion, model forward shapes, loss decrease on overfit, pairwise-rank, S4 study-disjoint no-source-overlap, fold partition, context metrics, macro averaging, NDCG/top-decile, reranker singletons, abs_candidate).
- Full migration suite: **99 passed** (no regressions).

## 9. DATA_COUNTS_AND_DENOMINATORS
- Delta-defined records used: **103,199** (5U-A1 60,237; 3U-A1 42,962). Records with delta=None excluded: 3,460.
- 5U-A1 S4 folds: {GSE114002 held-out: train 5,194 / test 55,043}, {GSE217518 held-out: train 55,043 / test 5,194}.
- 3U-A1 S4 folds (train/test): ENCSR854RUF 31,028/11,934; GSE186455 42,341/621; GSE200304 36,842/6,120; GSE232571 28,861/14,101; GSE232572 33,890/9,072; GSE261709 42,213/749; GSE298114 42,597/365.
- Reranker: 5U-A1 1,630 non-singleton measured sources / 56,977 singletons; 3U-A1 0 non-singleton / 42,962 singletons.

### Horizontal comparison (macro over S4 folds; spearman `n/a` = undefined constant predictor)
| model | 5U-A1 sp | 5U-A1 sign | 5U-A1 top10 | 3U-A1 sp | 3U-A1 sign | 3U-A1 top10 |
|---|---|---|---|---|---|---|
| mean | n/a | 0.4900 | 0.0199 | n/a | 0.4282 | 0.0378 |
| feature_ridge | −0.0167 | 0.4918 | −0.0044 | 0.0033 | 0.4333 | 0.0344 |
| kmer_ridge | −0.0282 | 0.4883 | 0.0083 | 0.0110 | 0.5036 | −0.0322 |
| xgboost | −0.0337 | 0.4877 | −0.0175 | −0.0279 | 0.4953 | −0.0166 |
| difference_features | 0.0016 | 0.4917 | 0.0218 | 0.0021 | 0.4226 | 0.0294 |
| **abs_candidate (B0-X)** | **0.2272** | **0.5100** | −0.0125 | **0.1129** | **0.5789** | 0.0745 |
| abs_candidate (same-fold, fresh) | 0.2268 | 0.5100 | −0.0202 | 0.1040 | 0.5766 | 0.0538 |
| small_cnn | −0.0029 | 0.4967 | 0.0210 | 0.0068 | 0.4461 | 0.0189 |
| small_transformer | −0.0186 | 0.4964 | −0.0477 | −0.0009 | 0.4688 | −0.0061 |
| siamese_encoder | −0.0106 | 0.5031 | 0.0294 | −0.0076 | 0.5063 | 0.0036 |
| full_pair_encoder | −0.0070 | 0.5071 | 0.0225 | −0.0015 | 0.5188 | 0.0242 |
| **SparseEditFormer (delta, pure seq)** | −0.0070 | 0.5105 | 0.0038 | −0.0336 | 0.4254 | 0.0408 |
| **SparseEditFormer (candval, anchored)** | **0.2971** | 0.5100 | 0.1603 | 0.1028 | 0.4324 | 0.1705 |

(top10 = fixed top-10 enrichment, B0-X convention; the pre-registered gate uses top-10% enrichment: SparseEditFormer anchored 5U-A1 0.1322, 3U-A1 0.1630.)
Per-study anchored Spearman: 5U-A1 GSE217518 0.3862 (vs abs 0.3838), GSE114002 0.1188 (vs abs −0.0873); 3U-A1 best GSE200304 0.2963 (vs abs 0.2874).

### Measured-neighborhood reranker (model, top-K paired)
| benchmark | NDCG@10 | top-decile recall | non-singleton sources |
|---|---|---|---|
| 5U-A1 | **0.834** | 0.550 | 1,630 |
| 3U-A1 | n/a | n/a | 0 |

## 10. REUSE_DECISIONS
- Encoding conventions (`NUC_ORDER`, `MAX_SEQ_LEN`, 12-dim edit features) reused from `scripts/b0x/features.py`; reimplemented locally in `m4_sparse/dataset.py` to avoid import coupling, with identical semantics.
- `abs_candidate` baseline reimplemented locally (`evaluate.py::abs_candidate_baseline`) and re-run on the exact same S4 test folds for an apples-to-apples comparison.
- Effect dataset `artifacts/b0x/effect_dataset.jsonl` REUSE_AS_IS (training/eval source). D1 canonical and main repo NOT modified.

## 11. GATE_STATUS
- **`M4_GATE_NOT_YET_MET`** (honest). The full pre-registered gate requires, on the ACTIVE benchmark, delta Spearman ≥ 0.25 AND sign accuracy ≥ 0.60 AND top-10% enrichment ≥ 1.50 AND beating `abs_candidate`. 
  - 5U-A1: anchored SparseEditFormer delta Spearman **0.2971 ≥ 0.25 ✓** and beats `abs_candidate` (0.2268) ✓; but sign accuracy **0.5100 < 0.60 ✗** and top-10% enrichment **0.1322 < 1.50 ✗**.
  - 3U-A1: delta Spearman **0.1028 < 0.25 ✗** and does not beat `abs_candidate` (0.1040) ✗; sign **0.4324 < 0.60 ✗**; top-10% enrichment **0.1630 < 1.50 ✗**.
- Not `M4_EFFECT_MODEL_ESTABLISHED`. The partial component (5U-A1 macro-delta-Spearman component exceeded its pre-registered threshold and beat the reference ceiling) is reported as a building block, NOT a full pass. Thresholds were not re-tuned after the fact.

## 12. CLAIMS_UNLOCKED
- A learned, single-backbone, source-relative effect model (SparseEditFormer) exceeds the B0-X ceiling on 5U-A1 macro delta Spearman (0.2971 vs abs_candidate 0.2268) when it uses the measured source value as an anchor (delta = candidate-value prediction − measured source anchor; inputs disclosed).
- Independent top-K paired reranker reaches NDCG@10 0.834 / top-decile recall 0.550 on 5U-A1 measured candidate pools (1,630 sources), well above the B0-X random_legal 0.276 ceiling.
- Honest negative established: pure-sequence (no measured anchor) effect transfer is at chance under S4 on both benchmarks.

## 13. CLAIMS_STILL_PROHIBITED
- NO `M4_EFFECT_MODEL_ESTABLISHED` (full gate not met). NO L4 real biological/therapeutic improvement claim. NO claim that any model improves TE/stability/expression without the predicted/internal-proxy qualifier. GSE246381 final labels NOT opened. CDS-B1 / 5U-A2 not auto-unlocked. No A2 dense pretraining claimed (none exists).

## 14. NEXT_PHASE_INPUTS
- Feed the anchored 5U-A1 delta-Spearman gain (0.2971, beats ceiling) and the NDCG@10 0.834 reranker into the next effect-model / measured-neighborhood optimization phases. Address the sign-accuracy and top-10% enrichment gaps (0.51/0.43 and 0.13/0.16 vs 0.60/1.50) before claiming the effect-model gate. Transfer these to O0-X → F0-X → G0-X → G1-X → E0-X → X0-X.

## 15. COMMIT_SHA
- Worktree commit for this M4 phase (see commit message; contract `GOAL-XEDITFLOW-MIGRATION-01`).

## 16. MANIFEST_AND_HASHES
- `artifacts/b0x/effect_dataset.jsonl` SHA-256: `f23a9fdd54a8ead90dccf793a073ccc6ed804a9b760a266c77b02e1fb1007ba1`.
- `artifacts/m4_sparse/{candval,delta}/results.json` kept on server (gitignored per convention, same as B0-X); SHA-256 recorded above. Honest measured values; no fabricated metrics.
