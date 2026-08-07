# E0-X Pre-registration & Ordinary Internal Test — mRNA-XEditFlow v1.1

- **Migration Goal:** `GOAL-XEDITFLOW-MIGRATION-01`
- **Prior terminal state:** `G1X_REAL_MRNA_GUIDANCE_VALUE_DEMONSTRATED` (+ `RANKING_AXIS_HEADROOM_LIMITED` carry-forward)
- **Phase:** E0-X — pre-registration, ordinary internal test, one-time sealed final (sealed final NOT executed; GSE246381 kept sealed)
- **Gate outcome:** `E0X_PREREG_FROZEN` + ordinary internal test completed under one-attempt frozen-key policy → **NO_GO (honest: sign_accuracy at class-prior ceiling)**; `SEALED_FINAL_NOT_EXECUTED` (GSE246381 preserved)
- **UTC:** 2026-08-07
- **Worktree branch:** `xeditflow-migration-20260806T024650Z`
- **Effect dataset (SHA-256):** `f23a9fdd54a8ead90dccf793a073ccc6ed804a9b760a266c77b02e1fb1007ba1` (carried forward, unchanged)
- **Pre-registration:** `configs/e0x_preregistration_v1.yaml` (id `E0X_PREREG_20260807`, status FROZEN)
- **Runner:** `scripts/e0x/run_e0x_final.py` (mode `internal`) + `scripts/e0x/prereg.py` + `scripts/e0x/sealed.py` on GPU `cuda:3` (GPU 4 avoided)
- **Internal-test result artifact:** `artifacts/e0x/e0x_internal_results.json`

---

## 1. FACTS_FROM_REPO
- Frozen before any sealed final access: canonical/data manifest (effect dataset SHA `f23a9fdd…`, 106,659 records / 103,199 delta-defined), split `S4` (leave-one-study-out), primary benchmarks `[5U-A1, 3U-A1]`, primary edit budgets `[1,3,5]`, model aliases + checkpoint hashes, reward/beta/calibration, primary metric family + Holm, seeds, evaluator command, GPU policy (fail-closed, GPU 4 forbidden), output schema (aggregate-only), and fallback route.
- M4 frozen critic checkpoints (9/9 candval) all confirm `TARGET="candidate_value"`, `ANCHOR_AT_TEST=True` (matches the pre-registration contract).
- The ordinary internal test evaluates **H1 effect transfer** on the non-sealed 5U-A1 folds (GSE114002 + GSE217518) with the frozen critic's anchored delta (`mean − measured source_value`), computes the H1 permutation p-value, applies the frozen Holm family, and enforces the full pre-registered effect gate.

## 2. FACTS_FROM_CONTRACTS
- § Phase E0-X: freeze the evaluation protocol before any final access; run an ordinary internal test under the **one-attempt frozen-key policy**; then exactly one sealed final on GSE246381 with `ACCESS_INTENT → compare-and-append reservation → exactly one terminal COMPLETION or ABORT`; abort/crash invalidates the v1 final and is not retryable; only the pre-registered aggregate is output (never row-level labels/IDs/order); all post-unblinding analysis marked `POST_UNBLINDING_EXPLORATORY`.
- Effect gate (`go_nogo.effect_gate`, frozen): `macro_delta_spearman_ge ≥ 0.25`, `macro_sign_accuracy_ge ≥ 0.60`, `top10_enrichment_ge ≥ 1.50`, `beat_strongest_nonfoundation_baseline` (vs abs_candidate on the same sealed folds).
- Hard constraints: `legality_ge = 1.00`, `length_preservation_ge = 1.00`, `budget_violation_le = 0.0`.

## 3. INFERENCES
- **The frozen critic's anchored delta shows a significant, positive monotone association with measured delta on held-out 5U-A1 studies:** `macro_delta_spearman = 0.297` (≥ 0.25), Holm-adjusted p = 0.0005 (significant), and it **beats the strongest non-foundation baseline** abs_candidate (`0.227`). Top-10% enrichment-over-random `= 9.92` (≥ 1.50). Legality hard constraint = 1.00 (by construction of the F0-X substitution-only legal flow).
- **`macro_sign_accuracy = 0.510 < 0.60` → the only failing effect-gate item.** A retrain experiment (dedicated sign head retargeted from `sign(candidate_value)` to `sign(delta)`) was run to test whether the failure is a training-target bug; it **did not improve** sign accuracy (GSE114002 0.4812, GSE217518 0.4988 — at or below the class prior ≈ 0.52). This confirms sign_accuracy is **capped by the class prior** on held-out studies, consistent with the B0-X finding that from-sequence supervised models are near chance at sign prediction under S4 transfer.
- **Therefore the ordinary internal test verdict is `NO_GO`** — driven by a single, scientifically-grounded metric (sign_accuracy) that cannot be legally raised without fabricating data or altering the frozen threshold. This is an honest negative, not a protocol/implementation failure.
- **GSE246381 is NOT consumed.** Because the internal test already fails the frozen effect gate (sign_accuracy at the prior ceiling), running the one-time sealed final would consume the single irreversible access for a predictable NO_GO. The sealed final is therefore **NOT executed** and GSE246381 remains sealed.

## 4. UNKNOWN_OR_BLOCKED
- Whether sign_accuracy ≥ 0.60 is scientifically attainable on this benchmark is blocked by the class prior; only a fundamentally stronger effect signal (new measured data / different endpoint) could change this. This is flagged, not resolved.
- The one-time GSE246381 sealed final is deliberately withheld pending a decision on the sign_accuracy threshold (a contract amendment would be required) or on proceeding with a known NO_GO.

## 5. FILES_READ
- `configs/e0x_preregistration_v1.yaml`, `scripts/e0x/prereg.py`, `scripts/e0x/sealed.py`, `scripts/e0x/run_e0x_final.py`, `scripts/m4_sparse/model.py`, `scripts/m4_sparse/evaluate.py`, `scripts/m4_sparse/train.py`, `scripts/m4_sparse/config.py`, `scripts/m4_sparse/dataset.py`, `scripts/m4_sparse/run.py`, `artifacts/m4_sparse/candval/*.pt`, `artifacts/b0x/effect_dataset.jsonl`, `artifacts/e0x/e0x_internal_results.json`, `reports/migration/G1X_REAL_MRNA_GUIDANCE_GATE.md`, `reports/migration/B0X_EFFECT_BASELINE_GATE.md`.

## 6. FILES_CHANGED
- `scripts/e0x/prereg.py` (new): E0-X pre-registration validation + Holm-Bonferroni.
- `scripts/e0x/sealed.py` (new): sealed access state machine, hash-chain + compare-and-append, aggregate-only output guard, verdict enforcing the **full** frozen effect gate (spearman + sign_accuracy + top10 enrichment + beat baseline).
- `scripts/e0x/run_e0x_final.py` (new): ordinary internal test + sealed-final orchestrator; enforces the frozen critic's `TARGET`/`ANCHOR_AT_TEST` from the checkpoint (fix: internal test now scores the anchored delta, not the raw candidate_value).
- `configs/e0x_preregistration_v1.yaml` (new, FROZEN).
- `tests/migration/test_e0x.py` (new): 34 unit tests (pre-registration validation, Holm, sealed lifecycle, aggregate-only guard, full-gate verdict).
- `reports/migration/E0X_PREREG_INTERNAL_GATE.md` (this report).

## 7. COMMANDS_RUN
- Server (editflow env), GPU `cuda:3`:
  ```
  python -m pytest tests/migration/test_e0x.py -q                     # 34 passed
  python -m pytest tests/migration/ -q                                # 191 passed
  python -m scripts.e0x.run_e0x_final --mode internal \
    --dataset artifacts/b0x/effect_dataset.jsonl \
    --prereg configs/e0x_preregistration_v1.yaml --gpu cuda:3
  ```
- Internal-test result written to `artifacts/e0x/e0x_internal_results.json` with verdict `NO_GO`.

## 8. TEST_RESULTS
- `tests/migration/test_e0x.py`: **34/34 PASS**.
- `tests/migration/` full suite: **191/191 PASS**.
- Internal test executed under the one-attempt frozen-key policy; output passes `assert_no_row_level` (no row-level labels/IDs/order leaked).

## 9. DATA_COUNTS_AND_DENOMINATORS
- H1 effect transfer evaluated on **5U-A1** non-sealed folds: n = 60,237 delta-defined records (GSE114002 55,043 + GSE217518 5,194), 3 metric contexts, study-macro.
- Effect dataset: 106,659 records / 103,199 delta-defined (denominator unchanged from B0-X).

## 10. REUSE_DECISIONS
- M4 frozen critic checkpoints: `REUSE_AS_IS` (hashes unchanged; anchored-delta evaluation contract now enforced correctly at runtime).
- B0-X effect dataset / S4 split: `REUSE_AS_IS`.
- Evaluator `assert_no_row_level` / sealed chain: newly built for E0-X (no legacy counterpart).

## 11. GATE_STATUS
- `E0X_PREREG_FROZEN` — pre-registration protocol frozen and validated.
- Ordinary internal test: **NO_GO** (sign_accuracy 0.510 < 0.60, class-prior-capped).
- `SEALED_FINAL_NOT_EXECUTED` — GSE246381 preserved; the one-time access is NOT consumed.

## 12. CLAIMS_UNLOCKED
- The frozen critic's anchored delta has a significant positive monotone association with measured delta on held-out 5U-A1 (spearman 0.297, Holm p 0.0005), beating abs_candidate.
- Top-10% enrichment-over-random (9.92) exceeds the frozen threshold.
- Legality hard constraint is met (1.00) by the F0-X legal flow construction.
- GSE246381 remains sealed and unconsumed.

## 13. CLAIMS_STILL_PROHIBITED
- Any claim that sign prediction meets the frozen 0.60 gate — sign_accuracy is at the class prior (~0.52) on held-out studies and cannot be legally raised with current data/modeling.
- Any claim of a sealed-final GO — the sealed final was not run.
- Any use of row-level GSE246381 labels (never accessed).

## 14. NEXT_PHASE_INPUTS
- The GSE246381 sealed final remains available for exactly one execution. Proceeding requires a decision on the sign_accuracy threshold (contract amendment) or an explicit instruction to consume the one-time access despite the known NO_GO.
- All E0-X code, tests, and the internal aggregate are committed and ready for that sealed final.

## 15. COMMIT_SHA
- See git log on branch `xeditflow-migration-20260806T024650Z` (E0-X code + tests + config + this report committed together).

## 16. MANIFEST_AND_HASHES
- Pre-registration: `configs/e0x_preregistration_v1.yaml` (id `E0X_PREREG_20260807`, FROZEN).
- Effect dataset SHA-256: `f23a9fdd54a8ead90dccf793a073ccc6ed804a9b760a266c77b02e1fb1007ba1`.
- Internal-test aggregate: `artifacts/e0x/e0x_internal_results.json` (verdict NO_GO).
- Critic checkpoints SHA-256: as frozen in `configs/e0x_preregistration_v1.yaml` (9/9, unchanged).
