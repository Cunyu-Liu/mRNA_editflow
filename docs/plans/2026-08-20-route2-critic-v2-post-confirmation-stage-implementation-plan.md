# Route 2 Critic V2 Post-confirmation Stage Implementation Plan

> **Execution policy:** Implement and verify first. Launch only the conditional watcher after the commit is pushed and the A100 worktree passes the same tests.

**Goal:** Close the executable gap from exact Critic V2 three-seed PASS through single TEST, refit, matched LOSO, readiness and gated Development generation.

**Architecture:** Reuse production builders/adjudicators for every gate and config, use one Python orchestrator for strict sequencing, use existing LOSO and Development generation runners for their stages, and use one 900-second shell watcher for the upstream terminal condition.

**Files:**

- Add `scripts/route_a_v3/run_route2_mrnabert_critic_v2_post_confirmation_stage_v1.py`.
- Add `scripts/route_a_v3/schedule_route2_mrnabert_critic_v2_post_confirmation_v1.sh`.
- Retire `scripts/route_a_v3/schedule_route2_mrnabert_postselection_controls_v1.sh` at entry.
- Add focused gate, ordering, target-refusal and watcher tests.
- Update execution and central-attempt records; do not add training rows before a trainer actually starts.

**Risk:** High. This is the only stage allowed to open the single frozen Development TEST. Therefore the production three-seed PASS must be validated before every write and the runner must refuse partial or duplicate targets.

**Minimum verification:** PASS-only TEST authorization; report-only TEST policy; exact TEST→refit→paired LOSO→readiness→conditional generation order; GPU0-5 free-memory-only selection; no Evaluation; no historical V1 execution route.
