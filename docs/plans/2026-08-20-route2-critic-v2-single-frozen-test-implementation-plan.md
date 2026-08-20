# Route 2 Critic V2 Single Frozen-TEST Implementation Plan

> **Execution policy:** Implement by functional batch under the global lightweight development and verification strategy. Do not apply task-by-task review or full-validation gates unless the batch risk warrants them.

**Goal:** Build a V2-only, fail-closed config preparer for the one frozen Development TEST that cannot be used before both Critic V2 gates pass.

**Architecture:** Add one standalone preparer beside the historical V1 entrypoint. It consumes both frozen V2 protocols, both terminal V2 adjudications and one exact confirmation config, replays the full training policy and emits one non-overwriting TEST config without running it.

**Tech Stack:** Python 3, JSON protocols/configs, pytest.

---

### Batch 1: V2 gate and config preparation

**Files:**
- Create: `configs/route_a_v3_route2_mrnabert_critic_v2_frozen_test_protocol_v1.json`
- Create: `scripts/route_a_v3/prepare_route2_mrnabert_critic_v2_frozen_test_config_v1.py`
- Create: `tests/route_a_v3/test_prepare_route2_mrnabert_critic_v2_frozen_test_config_v1.py`
- Modify: `docs/execution/route_a_v3_route2_mrnabert_training_commands_20260817.md`
- Modify: `docs/execution/route_a_v3_route2_training_attempt_table_20260817.md`

**Risk:** High — this is the only config path that may eventually open the withheld Development TEST.

**Implementation:** Freeze a separate TEST protocol before three-seed outcomes, including seed 20260823, exact Critic V2 policy and unique runtime/run targets. Validate the exact control/confirmation/TEST schemas and frozen states; require the control PASS and three-seed PASS statuses, exact seed set, three positive baseline margins, protected-outcome closure and identical frozen policy. Require the selected config to be the full Critic V2 confirmation run for seed 20260823. Emit one `FROZEN_DEVELOPMENT_TEST` config on GPU0-5 with the prospectively fixed 100-epoch `FINAL_EPOCH` execution rule required after TRAIN and VALIDATION are folded into training; retain `BEST_VALIDATION` as the pre-TEST selection provenance. Set TEST access explicitly true, Evaluation access false and a V2-specific output identity. Refuse existing output targets in the CLI.

**Minimum verification:** Focused pytest coverage for PASS, either gate failing, policy/seed drift, protected-outcome contamination, GPU range and CLI overwrite refusal; run the existing Critic V2 confirmation tests as the expanded compatibility check.

**Independent review:** No — the frozen requirements are explicit and the implementation is not authorized to run the real config; terminal execution remains a later gated task.
