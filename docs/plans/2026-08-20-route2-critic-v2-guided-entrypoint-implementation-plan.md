# Route 2 Critic V2 Guided Entrypoint Implementation Plan

> **Execution policy:** Freeze and verify the V2 entrypoint only. Do not build a real readiness packet or run generation.

**Goal:** Make the executable guided, matched-search and Development-comparison chain consume the exact Critic V2 readiness, final V2 refit checkpoint and V2 candidate artifacts.

**Architecture:** Retire all three old configs, add V2-specific configs, validate each boundary before any artifact read, and switch guided, matched-search and comparison contracts directly to V2 without a compatibility branch.

**Files:**

- Retire `configs/route_a_v3_route2_guided_xeditflow_development_gpu0_v1.json`.
- Add `configs/route_a_v3_route2_mrnabert_critic_v2_guided_xeditflow_development_gpu0_v1.json`.
- Update `scripts/route_a_v3/run_route2_guided_xeditflow_v1.py` to require V2 schemas and bindings.
- Retire the historical matched-search config and add a V2-specific replacement.
- Update `scripts/route_a_v3/run_route2_mrnabert_matched_search_suite_v1.py` to require V2 readiness and guided artifacts.
- Retire the historical comparison config and add a V2-specific replacement.
- Update `scripts/route_a_v3/run_route2_mrnabert_generation_comparison_suite_v1.py` to require the V2 guided method, candidates and checkpoint.
- Update focused tests for all three entrypoints.
- Update execution and central-attempt records.

**Risk:** Critical. This is the executable boundary between readiness and candidate generation.

**Minimum verification:** Reject all three old configs before artifact access; accept an exact synthetic V2 PASS; reject readiness status, online-encoder, Evaluation and checkpoint drift; bind the exact V2 guided/matched candidates; retain existing guidance, matching, comparison and accounting tests; run adjacent V2 readiness tests.
