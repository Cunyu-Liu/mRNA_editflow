# Route 2 Critic V2 All-Development Refit Implementation Plan

> **Execution policy:** Implement as one focused batch. Do not execute the real preparer or train a refit before the frozen upstream artifacts authorize it.

**Goal:** Add a V2-only, fail-closed preparer for the fixed all-126,165 Development refit after the single legal TEST.

**Architecture:** Freeze one refit protocol before outcomes. A standalone preparer validates the exact V2 TEST config and terminal summary against that protocol, then writes one non-overwriting refit config without inspecting TEST metric values or running it.

**Files:**

- Create `configs/route_a_v3_route2_mrnabert_critic_v2_all_development_refit_protocol_v1.json`.
- Create `scripts/route_a_v3/prepare_route2_mrnabert_critic_v2_all_development_refit_config_v1.py`.
- Create `tests/route_a_v3/test_prepare_route2_mrnabert_critic_v2_all_development_refit_config_v1.py`.
- Update the Route 2 training commands and central-attempt documentation.

**Risk:** High. This is the only configuration path that may train on all Development outcomes, but it still must not read Evaluation or perform model selection.

**Implementation:** Require the prospectively frozen TEST/refit schemas, exact seed and record counts, the full V2 policy, a completed CUDA TEST summary, TEST metrics present, no Evaluation read and matching TEST identity. Do not threshold TEST values. Emit one seed-20260823, 100-epoch, `FINAL_EPOCH`, all-Development config on GPU0-5 with fixed `/mnt` targets and explicit no-selection flags. Refuse an existing config or run.

**Minimum verification:** Focused tests for PASS, poor-but-valid TEST metrics, incomplete/missing TEST evidence, Evaluation contamination, identity/policy drift, GPU range and overwrite refusal; include the V2 TEST-gate and trainer split tests in the expanded check.
