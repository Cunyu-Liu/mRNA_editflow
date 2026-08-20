# Route 2 Critic V2 Matched-Baseline LOSO Implementation Plan

> **Execution policy:** Implement and verify configuration pairing only. Do not create real runtime configs or run baseline LOSO before the primary configs exist legally.

**Goal:** Add a V2-only preparer for 21 strongest-baseline LOSO folds exactly paired to primary Critic V2 LOSO.

**Architecture:** A prospective baseline protocol freezes comparator identity, native hyperparameters, cohort, pairing keys and fixed `/mnt` targets. The preparer validates 21 primary runtime configs and emits one baseline config per primary fold.

**Files:**

- Create `configs/route_a_v3_route2_mrnabert_critic_v2_matched_baseline_loso_protocol_v1.json`.
- Create `scripts/route_a_v3/prepare_route2_mrnabert_critic_v2_matched_baseline_loso_configs_v1.py`.
- Create `tests/route_a_v3/test_prepare_route2_mrnabert_critic_v2_matched_baseline_loso_configs_v1.py`.
- Update Route 2 execution and central-attempt records.

**Risk:** High. A missing fold or mismatched split/GPU would invalidate the LOSO comparison used by readiness.

**Implementation:** Validate prospective schemas/statuses, exact frozen baseline config and policy, exact seven-study/three-seed cohort, and every primary config's V2 identity, TEST-preserving split, seed, GPU and protected flags. Emit 21 native-baseline configs with identical pairing keys, fixed final epoch, TEST/Evaluation access false and cross-referenced primary identities. Refuse any existing target.

**Minimum verification:** Focused tests for exact pairing/cardinality, baseline substitution/policy drift, missing/duplicate or contaminated primary folds, protocol drift and overwrite refusal; include the shared LOSO pairing and aggregation provenance tests in the expanded suite.
