# Route 2 Critic V2 Guidance Readiness Implementation Plan

> **Execution policy:** Implement and verify the protocol, packet builder and adjudicator only. Do not read future terminal artifacts or authorize guidance now.

**Goal:** Add a V2-only readiness gate that binds the complete frozen critic/LOSO evidence chain and independent Flow G0 readiness.

**Architecture:** A prospective protocol fixes schemas, seeds, LOSO gate, dual-readiness rule and output paths. A builder assembles the already-authorized upstream evidence without deciding final LOSO/Flow readiness. A separate adjudicator recomputes all critic and Flow checks and writes PASS/NO-GO without running guidance.

**Files:**

- Create `configs/route_a_v3_route2_mrnabert_critic_v2_guidance_readiness_protocol_v1.json`.
- Create `scripts/route_a_v3/build_route2_mrnabert_critic_v2_guidance_readiness_input_v1.py`.
- Create `scripts/route_a_v3/adjudicate_route2_mrnabert_critic_v2_readiness_v1.py`.
- Create focused tests for both scripts.
- Update Route 2 execution and central-attempt records.

**Risk:** Critical. This gate is the only authorization for critic-guided generation.

**Implementation:** Validate prospective protocol bindings and exact terminal artifact schemas; exact seed set and seven-study LOSO completion; checkpoint presence; protected-outcome closure. Preserve negative but complete LOSO/Flow evidence for adjudication. Recompute V2 control/three-seed/TEST/refit/LOSO/reward/encoder/Evaluation checks and Flow G0 checks. Authorize guidance only when every critic and Flow check passes.

**Minimum verification:** Builder tests for exact schemas/seeds/checkpoints and one-write output; adjudicator tests for full PASS, each scientific dependency failure, TEST metric invariance, exact LOSO positivity, reward/encoder/Flow failure and Evaluation contamination; include legacy readiness and LOSO aggregation suites as compatibility checks.
