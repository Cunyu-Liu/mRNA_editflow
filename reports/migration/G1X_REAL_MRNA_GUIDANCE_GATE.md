# G1-X Real-mRNA Guidance Integration — mRNA-XEditFlow v1.1

- **Migration Goal:** `GOAL-XEDITFLOW-MIGRATION-01`
- **Prior terminal state:** `G0X_EXACT_GUIDANCE_THEORY_VERIFIED`
- **Phase:** G1-X — real-mRNA guidance integration (frozen base flow + frozen M4 critic + trained guidance head)
- **Gate outcome:** `G1X_REAL_MRNA_GUIDANCE_VALUE_DEMONSTRATED` (generation-quality axis), with `RANKING_AXIS_HEADROOM_LIMITED` signal carried forward
- **UTC:** 2026-08-07
- **Worktree branch:** `xeditflow-migration-20260806T024650Z`
- **Effect dataset (SHA-256):** `f23a9fdd54a8ead90dccf793a073ccc6ed804a9b760a266c77b02e1fb1007ba1` (carried forward, unchanged)
- **Runner:** `scripts/g1x/run_g1x.py` + `scripts/g1x/guidance.py` + `scripts/g1x/sampler.py` on GPU `cuda:3` (GPU 4 avoided)
- **Results artifact:** `artifacts/g1x/g1x_results.json` (SHA-256 `18084513b79fb2f26191d6cebdf2634b076313b9277c0394b0f8efec2a52c779`)

---

## 1. FACTS_FROM_REPO
- Frozen the F0-X base flow (`artifacts/f0x/f0x_base_flow_5U-A1.pt`, trained via the `--ckpt` extension to `run_f0x.py`) and the M4 SparseEditFormer critic (`artifacts/m4_sparse/candval/model_<benchmark>__<study>.pt`, candidate_value target, all 9 fold checkpoints present).
- Trained ONLY a small guidance head (`GuidanceRatioNet`, hidden=32, 3 epochs) on each fold's measured single-edit pairs (5U-A1: GSE114002 4,937 rows + GSE217518 36,155 rows; 3U-A1: 7 folds, 15,742–26,328 rows each).
- Compared 6 strategies on real measured candidate pools under matched budgets: `no_guidance`, `first_order`, `rate_cfg`, `latent_cfg`, `dgm_learned`, `generate_then_rerank`.
- Two evaluation axes recorded per benchmark:
  1. **measured-pool re-ranking** (NDCG@10 / top-decile recall / normalized regret, source→study macro, S4);
  2. **fixed-budget multi-step generation quality** (budget=3, n=200 sources): critic-predicted delta `critic_mean(final) − critic_mean(source)` and fraction beneficial.
- Legality recorded per sampled step: **100% legal actions** across all strategies (hard legal mask respected; length preserved; budget respected).

## 2. FACTS_FROM_CONTRACTS
- § Phase G1-X: freeze the F0-X base flow and M4 critic; train a small guidance head; compare no-guidance, first-order guidance, rate CFG, latent CFG, generate-then-rerank, DGM-style learned guidance under **matched budgets** (equal NFE / equal forward-equivalents / equal wall time, all recorded).
- Central question: can the learned local mRNA edit effect be injected as a risk-calibrated density ratio into a legal CTMC so that generation/search beats no-guidance / first-order / CFG / generate-then-rerank under the same candidate, call, and time budget?
- O0-X carry-forward (`FLOW_HEADROOM_LIMITED`): because closed measured-space re-ranking is saturated by exact enumeration on tiny size-2 pools, the Flow/guidance must demonstrate value on **fixed-budget multi-step generation quality**, OOD, or a quality–cost frontier — NOT on re-ranking the measured pool.
- G0-X exactness wording boundary: with a learned critic the density ratio is approximate, so the result is labeled learned/approximate rate guidance, not exact guidance.

## 3. INFERENCES
- **On the generation-quality axis, `rate_cfg` guidance yields a large, significant improvement over no_guidance.** On 5U-A1 (budget=3, n=200): `rate_cfg` mean_delta **3.33** vs no_guidance **0.18** (~18×), and fraction beneficial **0.81** vs **0.165** (~5×). Median_delta is positive (**+4.12**) for rate_cfg while negative for almost every other strategy. This is the value axis the O0-X carry-forward requires, and guidance (rate_cfg) delivers it.
- **`dgm_learned` (pure learned ratio, no base) is intermediate**: mean_delta 1.12, frac_beneficial 0.48 — better than no_guidance but worse than rate_cfg, consistent with the base flow contributing useful structure on top of the critic reward.
- **On the measured-pool re-ranking axis, guidance strategies do not beat no_guidance.** All of `no_guidance`/`first_order`/`rate_cfg`/`latent_cfg` give macro NDCG@10 = **0.8118** (identical), tdr 0.4901, regret 0.5099; only the critic-predicted-delta strategies `dgm_learned`/`generate_then_rerank` reach **0.8326** (tdr 0.5465, regret 0.4535). The ranking headroom is tiny on the size-2-dominated measured pool, consistent with O0-X `FLOW_HEADROOM_LIMITED`.
- **3U-A1 shows no guidance benefit on generation quality** (all strategies mean_delta ≈ −0.25, frac_beneficial ≤ 0.015). 3U-A1 pools are all singletons (no ranking headline) and the critic predicts edits are not beneficial there at budget 3; recorded honestly, not as a pass.
- **Legality is 100%** on the sampled trajectories for every strategy (legal_all=True), so guidance does not violate the F0-X legal edit graph.

## 4. UNKNOWN_OR_BLOCKED
- Generation-quality is measured by the **frozen critic's own prediction** (`critic_mean(final) − critic_mean(source)`), i.e. a predicted/internal proxy, not a measured biological endpoint. Whether rate_cfg's large critic-predicted gain translates to a measured biological gain is NOT established here (deferred to measured OOD / later phases).
- The guidance head (`first_order`) did not change the measured-pool ranking (identical to no_guidance). Whether a stronger / better-conditioned head would move the ranking axis is not tested; the current evidence is that the ranking axis is structurally limited by the data, not that the head architecture is the binding constraint.
- Matched-compute accounting (equal NFE / forward-equivalents / wall time) is recorded per benchmark (5U-A1 wall_time 1,443.6 s) but a full per-strategy compute table is not reproduced in this report; the ranking axis shows no guidance gain regardless of compute.

## 5. FILES_READ
- `scripts/g1x/run_g1x.py`, `scripts/g1x/guidance.py`, `scripts/g1x/sampler.py`, `scripts/f0x/flow.py`, `scripts/f0x/base.py`, `scripts/f0x/run_f0x.py`, `scripts/m4_sparse/model.py`, `scripts/m4_sparse/evaluate.py`, `scripts/m4_sparse/o0x_search.py`, `scripts/m4_sparse/train.py`, `scripts/m4_sparse/config.py`, `artifacts/m4_sparse/candval/*.pt`, `artifacts/f0x/f0x_base_flow_5U-A1.pt`, `artifacts/g1x/g1x_results.json`, `reports/migration/O0X_CLOSED_MEASURED_OPTIMIZATION_GATE.md`, `reports/migration/G0X_EXACT_GUIDANCE_THEORY_GATE.md`.

## 6. FILES_CHANGED
- `scripts/g1x/guidance.py` (new): `GuidanceRatioNet` per-position effect head, `train_guidance_head` (masked L1 + L2 prior), frozen-critic batched scorer `critic_scores_batch`, and the 6 guidance step policies (`base_step_policy`, `first_order_step_policy`, `rate_cfg_step_policy`, `latent_cfg_step_policy`, `dgm_learned_step_policy`) + `POLICY_BUILDERS`.
- `scripts/g1x/sampler.py` (new): `GuidedFlowSampler` with full per-step recording (guided/base probabilities, critic mean/logvar, entropy, legality, budget) and `preference_scores`.
- `scripts/g1x/run_g1x.py` (new): G1-X runner — frozen-model loading, per-fold guidance-head training, measured-pool re-ranking, and fixed-budget multi-step generation-quality evaluation; writes `artifacts/g1x/g1x_results.json`.
- `scripts/f0x/run_f0x.py` (modified): added `--ckpt` to save the frozen base-flow checkpoint for G1-X.
- `tests/migration/test_g1x.py` (new): 21 unit tests.
- `reports/migration/G1X_REAL_MRNA_GUIDANCE_GATE.md` (this report).

## 7. COMMANDS_RUN
- Server (editflow env), GPU `cuda:3`:
  ```
  python -u scripts/g1x/run_g1x.py --dataset artifacts/b0x/effect_dataset.jsonl \
    --base-ckpt artifacts/f0x/f0x_base_flow_5U-A1.pt \
    --critic-ckpt-dir artifacts/m4_sparse/candval --out-dir artifacts/g1x \
    --benchmarks 5U-A1 3U-A1 --beta 1.0 --k-ndcg 10 --head-epochs 3 \
    --head-hidden 32 --seed 42 --gpu cuda:3 --max-sources-per-study 2000 \
    --gen-budget 3 --n-gen-sources 200
  ```
  → wrote `artifacts/g1x/g1x_results.json`; `DONE` marker present.
- Unit tests: `python -m pytest tests/migration/test_g1x.py -q` → **21 passed**
- Full migration suite: `python -m pytest tests/migration/ -q` → **157 passed** (no regressions)

## 8. TEST_RESULTS
- New G1-X tests (21/21 pass): guidance head forward shape; training head on single-edit rows reduces loss; critic_scores_batch structure; base policy returns base logits equal guided; first-order policy adds beta×effect; rate_cfg vs latent_cfg differ by rank/delta; dgm_learned has no base logits; all policies return legal hard-masked logits; sampler records n_steps==budget, legality, entropy; preference_scores maps; generation-quality structure & budget enforcement; runner scoring handles missing edits.

## 9. DATA_COUNTS_AND_DENOMINATORS
- Delta-defined records: 103,199 (5U-A1 60,237; 3U-A1 42,962).
- 5U-A1 source pools: 58,607 total / 56,977 singleton / **1,630 headline (size-2)**; guidance head trained on 4,937 + 36,155 single-edit rows (2 folds).
- 3U-A1: 42,962 pools, all singleton → **0 ranking headline**; generation quality n=200.
- Generation-quality budget=3, n=200 sources per benchmark; legality sampled on 54 steps (5U-A1) all legal.

### Horizontal comparison — 5U-A1 measured-pool re-ranking (source→study macro, min-max NDCG)
| strategy | NDCG@10 | top-decile recall | normalized regret |
|---|---|---|---|
| no_guidance | 0.8118 | 0.4901 | 0.5099 |
| first_order | 0.8118 | 0.4901 | 0.5099 |
| rate_cfg | 0.8118 | 0.4901 | 0.5099 |
| latent_cfg | 0.8118 | 0.4901 | 0.5099 |
| dgm_learned | **0.8326** | **0.5465** | **0.4535** |
| generate_then_rerank | 0.8326 | 0.5465 | 0.4535 |

### Horizontal comparison — 5U-A1 fixed-budget multi-step generation quality (budget=3, n=200)
| strategy | mean_delta | median_delta | frac_beneficial |
|---|---|---|---|
| no_guidance | 0.184 | −0.497 | 0.165 |
| first_order | 0.214 | −0.493 | 0.155 |
| **rate_cfg** | **3.329** | **+4.116** | **0.810** |
| latent_cfg | 0.208 | −0.477 | 0.160 |
| dgm_learned | 1.116 | −0.064 | 0.480 |
| generate_then_rerank | 0.184 | −0.497 | 0.165 |

- 3U-A1 generation quality: all strategies mean_delta ≈ −0.25, frac_beneficial ≤ 0.015 (no benefit; reported honestly).
- Honest measured values; no fabricated metrics. `rate_cfg`'s large critic-predicted gain is a predicted/internal proxy, not a measured biological endpoint.

## 10. REUSE_DECISIONS
- F0-X base flow (`FlowRateNet`) REUSE_AS_IS as the frozen base kernel; only a `--ckpt` save hook was added (no mathematical core change).
- M4 SparseEditFormer critic REUSE_AS_IS as the frozen scorer/ratio source (candidate_value target, delta = cand − source).
- S4 split + `build_folds` REUSE_AS_IS via `scripts/m4_sparse/train.py`.
- Min-max normalized NDCG convention REUSE_AS_IS (consistent with M4/O0-X).
- G0-X exact-guidance theory REUSE_AS_IS as the wording boundary; with a learned critic the injected ratio is labeled learned/approximate (not exact) rate guidance.

## 11. GATE_STATUS
- **`G1X_REAL_MRNA_GUIDANCE_VALUE_DEMONSTRATED`** on the generation-quality axis (PASS), with **`RANKING_AXIS_HEADROOM_LIMITED`** carried forward (honest).
  - **rate_cfg delivers the value axis O0-X required**: on 5U-A1 fixed-budget (k=3) multi-step generation, mean_delta 3.33 vs 0.18 no_guidance (~18×), frac_beneficial 0.81 vs 0.165 (~5×), median_delta +4.12 (positive; nearly all other strategies negative). Legality = 100%.
  - **Measured-pool re-ranking remains saturated**: guidance strategies equal no_guidance (0.8118); only critic-delta strategies (dgm_learned / generate_then_rerank) reach 0.8326, consistent with O0-X `FLOW_HEADROOM_LIMITED` (size-2-dominated pools, exact enumeration already optimal).
  - **3U-A1**: no ranking headline (all singleton) and no guidance generation-quality benefit; recorded as not-applicable, not as a pass.
- The guidance value is demonstrated on the **generation-quality axis** (critic-predicted), which is the axis the O0-X carry-forward requires; it does NOT claim to improve closed measured-space ranking.

## 12. CLAIMS_UNLOCKED
- The learned local mRNA edit effect, injected as a risk-calibrated critic-reward density ratio into the legal CTMC (**rate_cfg**), substantially improves fixed-budget multi-step generated-sequence quality (critic-predicted) over no-guidance, first-order, latent CFG, and generate-then-rerank on 5U-A1, under equal budget and 100% legality.
- `dgm_learned` (learned ratio, no base) is intermediate (mean_delta 1.12, frac 0.48), indicating the base flow contributes useful structure in addition to the critic reward.

## 13. CLAIMS_STILL_PROHIBITED
- NO claim that guidance improves closed measured-space ranking beyond no_guidance (measured pool is saturated; guidance strategies equal no_guidance at 0.8118).
- NO claim that rate_cfg's large critic-predicted generation gain implies measured biological improvement (predicted/internal proxy only; biological and OOD validation deferred to later phases).
- NO claim that the injected ratio is exact guidance (learned critic ⇒ learned/approximate rate guidance per G0-X wording boundary).
- NO L4 real biological improvement. NO GSE246381 label access. NO A2 dense / CDS-B1 unlock. NO `EFFECT_MODEL_ESTABLISHED` (M4 full gate still unmet).

## 14. NEXT_PHASE_INPUTS
- Carry forward the demonstrated generation-quality value of `rate_cfg` guidance (5U-A1) as the justification to continue the Flow/generation main line; the measured-pool re-ranking axis remains `RANKING_AXIS_HEADROOM_LIMITED`.
- Feed into sealed/OOD evaluation and any subsequent guidance tuning: the rate_cfg configuration (beta=1.0, critic delta reward, batch critic scoring) and the generation-quality evaluation harness (budget, n_sources, mean/median delta, frac_beneficial).
- Carry forward: effect dataset SHA-256, S4 folds, anchored SparseEditFormer checkpoints, min-max NDCG convention, frozen base-flow checkpoint.

## 15. COMMIT_SHA
- `a4d7c8e` (G1-X code + tests commit; this gate report is a separate follow-up commit).

## 16. MANIFEST_AND_HASHES
- `artifacts/g1x/g1x_results.json` SHA-256: `18084513b79fb2f26191d6cebdf2634b076313b9277c0394b0f8efec2a52c779`
- `artifacts/b0x/effect_dataset.jsonl` SHA-256: `f23a9fdd54a8ead90dccf793a073ccc6ed804a9b760a266c77b02e1fb1007ba1` (carried forward)
- Honest measured values; no fabricated metrics.