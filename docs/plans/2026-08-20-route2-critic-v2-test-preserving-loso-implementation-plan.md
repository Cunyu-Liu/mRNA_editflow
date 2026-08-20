# Route 2 Critic V2 TEST-Preserving LOSO Implementation Plan

> **Execution policy:** Implement and verify the config gate only. Do not create real runtime configs or run LOSO before the exact refit is terminal.

**Goal:** Add a V2-only preparer for the frozen 7-study × 3-seed mRNABERT TEST-preserving LOSO cohort.

**Architecture:** A prospective LOSO protocol fixes cohort, data boundary, policy, GPU assignment and `/mnt` targets. A standalone preparer validates the exact V2 refit protocol/config/summary and writes all 21 configs once.

**Files:**

- Create `configs/route_a_v3_route2_mrnabert_critic_v2_test_preserving_loso_protocol_v1.json`.
- Create `scripts/route_a_v3/prepare_route2_mrnabert_critic_v2_test_preserving_loso_configs_v1.py`.
- Create `tests/route_a_v3/test_prepare_route2_mrnabert_critic_v2_test_preserving_loso_configs_v1.py`.
- Update Route 2 execution and central-attempt records.

**Risk:** High. Incorrect ordering can bypass the single TEST/refit gates, and incorrect folds can leak the withheld TEST or held-out study.

**Implementation:** Require exact prospective schemas/statuses; exact terminal all-Development refit identity, 126,165-row scope, 100-epoch final checkpoint, CUDA update and zero Evaluation reads; exact shared seven-study/three-seed/GPU schedule; exact V2 training policy. Emit 21 non-overwriting `LOSO_DEVELOPMENT_VALIDATION_ONLY_FROZEN_HYPERPARAMETERS` configs using TRAIN/VALIDATION only and `FINAL_EPOCH`, with TEST/Evaluation access false.

**Minimum verification:** Focused tests for exact cardinality/mapping, valid refit ordering, nonterminal/contaminated refit rejection, policy drift, protocol cohort drift and existing-target refusal; include shared LOSO schedule and trainer split tests in the expanded suite.
