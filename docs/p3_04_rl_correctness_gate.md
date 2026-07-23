# P3-04: RL Correctness Gate and Production-Path Freeze

**Date**: 2026-07-23
**Status**: PASS (scientific_validation_status=PENDING)
**Phase**: P3-04
**Spec**: `提示词/mrna的 rl 的后续优化的分阶段提示词.md#L1337-1496`

---

## 1. Overview

P3-04 fixes six correctness defects in the GRPO training pipeline and freezes
the unique production path. No 10-seed training was started. All changes are
additive — existing modules are not modified in-place; drop-in replacements
and wrappers are provided.

### Files Changed

| File | SHA-256 | Purpose |
|------|---------|---------|
| `rl/training_reward.py` | `680b3206a66bb93dc0f9aa94152575f3f9df0500de1969388889b859b8688a62` | Task 4: Training reward with risk adjustment, hard-constraint gating, STOP preference |
| `rl/p3_04_correctness.py` | `0e9e6e3d61ec7dea47936cde3fc828144ebace9d008717206f19acd3e7559a30` | Tasks 1, 2, 3, 5, 6: Multi-epoch GRPO, deterministic forward, complete MDP state, codon action, production path gate |
| `tests/test_p3_04_rl_correctness.py` | `b7f85fa53b9496f176f27a7ed087a82d2e0ee0bca8e47cc583015b0af81b9097` | 14 acceptance tests (29 test methods) |

### Test Results

```
29 passed in 3.15s
```

All 14 acceptance criteria pass. See `docs/p3_04_rl_correctness_matrix.json` for
the full test-by-test breakdown.

---

## 2. Task Summaries

### Task 1: Correct old/new ratio (`MultiEpochGRPOConfig`, `multi_epoch_grpo_update`)

**Bug found**: `train_grpo.py` performed a single update epoch with no
multi-epoch clipped objective. Old log-probs were not explicitly frozen
across epochs. The `risk_adjusted_reward` was computed but never entered
the advantage (audit-only dead code).

**Fix**: `MultiEpochGRPOConfig` adds `policy_epochs`, `trajectory_minibatch_size`,
`target_kl`, `max_kl`, `max_clip_fraction`, `gradient_clip`, `gradient_accumulation`,
and `lr`. `multi_epoch_grpo_update` implements:
- Snapshot rollout policy → sample trajectories → freeze `old_log_probs`
- Multiple update epochs with clipped surrogate objective
- `old_log_probs` never recomputed across epochs
- Epoch 1 ratio ≈ 1 (verified by `TestRatioBeforeUpdate`)
- Post-update ratio deviates from 1 (verified by `TestRatioAfterUpdate`)
- Clip fraction audited per epoch (`compute_ratio_stats`)

### Task 2: Deterministic policy forward (`DeterministicPolicy`)

**Bug found**: `rl/policy.py` `_model_forward(record, no_grad=True)` did not
enforce `model.eval()`. `train_grpo.py` called `model.eval()` for sampling
(L93) then `model.train()` for update (L97), causing dropout to corrupt
the PPO ratio.

**Fix**: `DeterministicPolicy` wrapper enforces `model.eval()` +
`backbone.eval()` during forward while keeping gradients flowing
(`requires_grad` untouched). Original train/eval mode is restored in
`finally` block.

**Verified**: Same params + same state + same mask → identical distribution
with grad and no-grad (`TestDeterministicForward`).

### Task 3: Complete MDP state (`CompletePolicyStep`)

**Bug found**: `PolicyStep` dataclass in `trajectory_sampler.py` stored only
7 fields. Critical fields like `record`, `source_id`, `current_sequence`,
`visited_sequence_hashes`, `task_id`, `editable_regions`, `preference`,
`legal_action_ids` were missing. The update step could not reconstruct
the exact mask from history.

**Fix**: `CompletePolicyStep` stores all 13 required fields:
`record`, `source_id`, `current_sequence`, `visited_sequence_hashes`,
`remaining_budget`, `task_id`, `editable_regions`, `preference`,
`legal_action_ids`, `action_mask_hash`, `old_log_prob`,
`reference_log_prob`, `reward_provenance`.

`build_complete_step()` constructs the step from trajectory data.
`recover_mask_from_history()` reconstructs the legal action mask from
stored `legal_action_ids` and `action_mask_hash`.

**Verified**: `TestCycleMaskRecovery` confirms all 13 fields present and
mask hash recoverable.

### Task 4: Training reward explicit (`build_training_reward`)

**Bug found**: `risk_adjusted_reward` in `trajectory_sampler.py` was
computed but never entered the advantage (audit-only dead code). The
uncertainty was double-subtracted (L172: `"uncertainty": -final["uncertainty"]`
in raw, then L175: `risk_adjusted = sum(raw.values()) - final["uncertainty"]`).

**Fix**: `build_training_reward()` in `rl/training_reward.py`:
- **Source-normalized**: scalar is always relative to `source_baseline`
- **Risk-adjusted**: uncertainty subtracted exactly once
- **Hard-failure gated**: any hard constraint failure → scalar = −1.0, STOP preferred
- **All-negative → STOP preferred**: if all soft objectives ≤ 0, scalar
  clamped to ≤ `stop_reward`
- **Auditable**: full provenance dict recorded

**Verified**: `TestAllNegativeStopPreferred` (3 tests) and
`TestSourceBaselineNormalization` (4 tests) confirm all guarantees.

### Task 5: Codon-level action (`CodonAction`)

**Bug found**: `rl/action_space.py` `Action(op, pos, nt)` is nucleotide-level.
A synonymous codon swap requiring >1 nt change costs multiple edit-budget
steps, creating nonsynonymous intermediate states.

**Fix**: `CodonAction(op="codon_sub", codon_pos, new_codon, old_codon)` is
an atomic codon-level substitution. `synonymous_codon_actions(cds)`
enumerates all legal synonymous codon substitutions. `apply_codon_action`
applies the substitution and verifies protein preservation.

**Verified**: `TestCodonActionProteinPreservation` (3 tests) confirms 100%
protein preservation and no single-nt intermediate states.

### Task 6: Production path freeze (`ProductionPathGate`)

**Bug found**: No separation between legacy pilot checkpoints, tiny-MDP
compatibility artifacts, and production constrained GRPO checkpoints.
Paper loader could accidentally load a legacy artifact.

**Fix**: `ProductionPathGate` provides:
- `is_legacy_checkpoint()`: checks for legacy markers (raw REINFORCE, no clip)
- `is_production_checkpoint()`: checks for production markers (clipped GRPO, MDP state, codon actions)
- `load_for_paper()`: rejects legacy artifacts with `ValueError`, accepts production only
- `classify_checkpoint()`: returns `"legacy"`, `"tiny_mdp"`, or `"production"`

**Verified**: `TestLegacyArtifactRejection` (3 tests) confirms paper loader
rejects legacy and accepts production.

---

## 3. Acceptance Test Summary

| # | Criterion | Test Class | Result |
|---|-----------|------------|--------|
| 1 | Update前 ratio≈1 | `TestRatioBeforeUpdate` | PASS |
| 2 | Update后 ratio≠1 | `TestRatioAfterUpdate` | PASS |
| 3 | Clip fraction test | `TestClipFraction` (3 tests) | PASS |
| 4 | Deterministic policy forward | `TestDeterministicForward` | PASS |
| 5 | Categorical KL ≥ −1e-7 | `TestCategoricalKL` (3 tests) | PASS |
| 6 | Legal action IDs consistent | `TestLegalActionConsistency` | PASS |
| 7 | Historical cycle mask recoverable | `TestCycleMaskRecovery` (2 tests) | PASS |
| 8 | All-negative → STOP | `TestAllNegativeStopPreferred` (3 tests) | PASS |
| 9 | Codon action 100% protein preservation | `TestCodonActionProteinPreservation` (3 tests) | PASS |
| 10 | Reference policy bitwise unchanged | `TestReferencePolicyUnchanged` | PASS |
| 11 | Smoke batch ≥ 4 sources | `TestSmokeBatchFourSources` (2 tests) | PASS |
| 12 | Checkpoint resume consistent | `TestCheckpointResume` | PASS |
| 13 | Legacy artifact rejected by paper loader | `TestLegacyArtifactRejection` (3 tests) | PASS |
| 14 | Source baseline normalization | `TestSourceBaselineNormalization` (4 tests) | PASS |

**Total**: 29 test methods, 29 passed, 0 failed.

---

## 4. Existing Bugs Identified (not modified in-place)

The following bugs were identified during exploration but are NOT modified
in-place (per project constraint: existing modules are not modified).
The new modules in `rl/p3_04_correctness.py` and `rl/training_reward.py`
provide drop-in replacements:

1. **`train_grpo.py` L93/L97**: `model.eval()` for sampling then `model.train()` for update → dropout corrupts PPO ratio
2. **`train_grpo.py`**: `risk_adjusted_reward` computed but never used for advantage (audit-only dead code)
3. **`rl/policy.py` `_model_forward`**: Does not enforce eval/train mode (root cause of dropout bug)
4. **`rl/trajectory_sampler.py` L172/L175**: Uncertainty double-subtracted
5. **`rl/trajectory_sampler.py`**: `risk_adjusted_reward` is audit-only
6. **`rl/action_space.py`**: Nucleotide-level action, no codon-level action
7. **`scripts/run_p2_05_grpo_pilot.py`**: Uses `GRPOREINFORCE` (raw REINFORCE, not clipped); `--kl-coef` is dead code; `max_steps` defaults to `limit` (1024) when omitted → memory blow-up; reference policy not frozen

---

## 5. Commands

```bash
# Run acceptance tests
cd /home/cunyuliu/mrna_editflow_goal/mrna_editflow
source /home/cunyuliu/miniconda3/etc/profile.d/conda.sh && conda activate editflow
python -m pytest tests/test_p3_04_rl_correctness.py -v --tb=short

# Verify SHA-256
sha256sum rl/training_reward.py rl/p3_04_correctness.py tests/test_p3_04_rl_correctness.py
```

---

## 6. Unresolved Risks

1. **Existing modules not modified in-place**: The production training script
   (`train_grpo.py`) still has the bugs listed in §4. The drop-in replacements
   must be wired into the production training loop in a future phase (P3-06/P3-08).
2. **No real model tested**: Acceptance tests use a `DummyModel` (length-agnostic
   per-position linear). Real `MRNAEditFormer` forward path is not tested here.
3. **No GPU smoke test**: All tests run on CPU. GPU memory and mixed-precision
   behavior is deferred to P3-08.
4. **Codon action not integrated into action space**: `CodonAction` is defined
   but not yet integrated into `build_legal_action_mask`. Integration is
   deferred to P3-06 (Minimal-Edit MDP and action space).

---

## 7. GO / NO-GO

**Status**: GO

All 14 acceptance criteria pass. The RL correctness gate is satisfied.
Production path is frozen via `ProductionPathGate`. The next phase (P3-05)
can proceed with the source-candidate delta architecture.

**Note**: `phase_status=PASS` but `scientific_validation_status=PENDING`.
P3-04 PASS means the RL correctness gate is satisfied, NOT that H1–H6
hypotheses are validated.
