# Route 2 Critic V2 Development Generation Stage Implementation Plan

> **Execution policy:** Implement and verify only. Do not launch without a real V2 dual-readiness PASS.

**Goal:** Provide the executable GPU0-5 path from V2 readiness to ordered guided, matched-search and Development comparison stages.

**Architecture:** Reuse the three production validators and frozen config templates; validate readiness before writes; select one physical GPU by free memory; write one-time runtime configs; execute children sequentially with preserved logs.

**Files:**

- Add `scripts/route_a_v3/run_route2_mrnabert_critic_v2_development_generation_stage_v1.py`.
- Add focused GPU-selection, config-binding, overwrite and ordering tests.
- Update execution and central-attempt records.

**Risk:** High. This entrypoint eventually creates generated candidates and therefore must remain behind the complete readiness gate.

**Minimum verification:** Exact V2 schemas/paths/checkpoint/method identity; readiness validation precedes runtime writes; GPU0-5 and free-memory-only selection; one-write runtime root; strict child order; no Evaluation or canonical credit.
