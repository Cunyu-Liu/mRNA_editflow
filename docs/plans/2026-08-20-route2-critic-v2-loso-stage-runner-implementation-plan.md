# Route 2 Critic V2 LOSO Stage Runner Implementation Plan

> **Execution policy:** Implement and test only. Do not launch before TEST/refit and both LOSO config gates legitimately complete.

**Goal:** Provide the executable six-GPU path from 21+21 prepared V2 LOSO configs to three Development LOSO aggregates.

**Architecture:** Reuse the V2 aggregation builder's protocol/config-pair validator; construct six fixed GPU queues; run each queue sequentially and all queues concurrently; aggregate only after every process succeeds.

**Files:**

- Add `scripts/route_a_v3/run_route2_mrnabert_critic_v2_loso_stage_v1.py`.
- Add focused planning and ordering tests.
- Update execution and central-attempt records.

**Risk:** High. This runner eventually launches 42 GPU trainings, so its preflight and ordering must be exact before it is invoked.

**Minimum verification:** 21 unique pairs; six physical queues; exact config identity; primary then paired baseline on the same GPU; no existing output/log/input/result roots; aggregation only after worker completion; adjacent config-builder and aggregation regressions.
