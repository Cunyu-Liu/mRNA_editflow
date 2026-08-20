# Route 2 Critic V2 Task/Study-Balanced Implementation Plan

> **Execution policy:** Implement by functional batch under the global lightweight development and verification strategy. Do not apply task-by-task review or full-validation gates unless the batch risk warrants them.

**Goal:** Build and prospectively freeze the Development-only Critic V2 control screen that aligns mRNABERT critic optimization with task-macro selection.

**Architecture:** Extend the existing Delta trainer with hierarchical task/study/source-group sampling and task-macro Huber aggregation while preserving the default behavior of all prior runs. Add one parameter-matched source+edit-metadata control to the existing pretrained edit-centered model and a frozen four-arm control adjudication path.

**Tech Stack:** Python 3, PyTorch, NumPy/SciPy, JSON configs, pytest, Bash GPU scheduler.

---

### Batch 1: Balanced training primitive and provenance

**Files:**
- Modify: `scripts/route_a_v3/train_route2_delta_predictor_v1.py`
- Modify: `core/route2_experiment_ledger.py`
- Test: `tests/route_a_v3/test_route2_delta_predictor_v1.py`
- Test: `tests/route_a_v3/test_route2_experiment_ledger_v1.py`

**Risk:** Medium — sampling and loss aggregation can silently change the effective estimand or update budget.

**Implementation:** Add `TASK_THEN_STUDY_THEN_SOURCE_CONTEXT_ENDPOINT_GROUP` weights, a deterministic fixed-draw hierarchical TRAIN sampler with length-local batches, and `TASK_MACRO_MEAN` loss aggregation. Keep existing defaults unchanged. Persist both modes in checkpoints, terminal summaries, per-run records, and the central CSV.

**Minimum verification:** Focused sampler/loss/provenance tests plus the existing Delta trainer unit-test module.

**Independent review:** No — the frozen protocol and focused numerical tests directly cover the material risk.

### Batch 2: Independent source+edit-metadata control

**Files:**
- Modify: `core/route2_delta_predictor.py`
- Modify: `scripts/route_a_v3/train_route2_delta_predictor_v1.py`
- Test: `tests/route_a_v3/test_route2_delta_predictor_v1.py`

**Risk:** Medium — an incorrectly wired control could retain candidate-global information or collapse into source-only.

**Implementation:** Add a pretrained edit-metadata control model kind. Reuse the identical parameterized network, retain source plus raw edit identity/position/context, replace candidate pretrained/global context with the source, and omit the reverse candidate-background branch. Record the control identity in the checkpoint and summary.

**Minimum verification:** Assert exact parameter matching; anchor-only predictions must ignore changes to candidate pretrained features, respond to explicit edit changes, and not equal source-only by construction.

**Independent review:** No — focused behavioral tests cover the information boundary.

### Batch 3: Frozen Critic V2 screen, adjudication, and scheduler

**Files:**
- Create: `configs/route_a_v3_route2_mrnabert_critic_v2_protocol_v1.json`
- Create: `scripts/route_a_v3/prepare_route2_mrnabert_critic_v2_controls_v1.py`
- Create: `scripts/route_a_v3/adjudicate_route2_mrnabert_critic_v2_controls_v1.py`
- Create: `scripts/route_a_v3/schedule_route2_mrnabert_critic_v2_controls_v1.sh`
- Create: `tests/route_a_v3/test_prepare_route2_mrnabert_critic_v2_controls_v1.py`
- Create: `tests/route_a_v3/test_adjudicate_route2_mrnabert_critic_v2_controls_v1.py`

**Risk:** High — this gate decides whether three new final seeds may run.

**Implementation:** Freeze the existing 0.131714 strongest baseline, one screen seed, three confirmation seeds, exact data/budget/modes, four arm definitions, and predeclared control thresholds. Generate runtime configs without overwriting outputs. The scheduler selects only GPU0-5 by available memory and runs one copy per arm. The adjudicator validates CUDA completion, parameter/budget matching, protected-outcome closure, task breadth, prediction-scale diagnostics, and the control margins; it authorizes only the frozen three seeds after PASS.

The conditional confirmation scheduler is prepared prospectively but does not
create seed configs until the control adjudication exists and passes. It polls
that single terminal artifact every 15 minutes, chooses three sufficiently free
GPUs from GPU0-5, runs exactly seeds 20260822/20260823/20260824, and writes one
three-seed adjudication. It stops after reporting whether one frozen Development
TEST is authorized; it never opens TEST itself.

**Minimum verification:** Focused preparer/adjudicator tests, trainer tests, shell syntax check, and `git diff --check`. These checks detect config/gate drift that could authorize seeds from unmatched or protected-outcome runs; any failure blocks launch.

**Independent review:** No separate agent — the user did not request delegation; the high-risk gate receives expanded targeted validation instead.

### Batch 4: Project record, delivery, and launch

**Files:**
- Modify: `docs/execution/route_a_v3_route2_mrnabert_training_commands_20260817.md`
- Create: `audits/route_a_v3_route2_v332_freshness_and_critic_v2_freeze_v1.json`

**Risk:** Medium — stale status could cause duplicate terminal runs or premature TEST/guidance.

**Implementation:** Record evaluator qualification, matched-generation terminal/genetic strongest baseline, protected-outcome counts, Critic V2 frozen hypothesis, and exact artifact paths. Preserve the user's unrelated deletion. Commit and push only scoped files, fast-forward the clean A100 worktree, then launch exactly one control scheduler. The training entrypoint will upsert RUNNING/terminal rows in the central attempt table.

**Minimum verification:** Confirm the pushed commit, one scheduler/process tree, four distinct output paths, CUDA first-batch evidence, and central RUNNING rows. Do not inspect epoch outcomes until terminal or abnormal exit.

**Independent review:** No — launch acceptance is a bounded operational check, not a second scientific adjudication.
