# RL Stage 5: constrained multi-objective GRPO

## Policy definition and action masks

The policy is the existing CTMC intensity field normalized only over legal
`STOP`, `SUB`, `INS`, and `DEL` actions. `trajectory_sampler.py` applies the
region, CDS-synonym, frame/protein, task-operation, cycle, and edit-budget
masks before `softmax(log_score / temperature)`. Masked actions do not appear
in the support and therefore have probability exactly zero. T4 preserves start
and terminal stop codons and accepts only synonymous internal-CDS substitutions.

## Trajectory schema

Every sampled action records state, action, old behavior log-probability,
frozen-reference log-probability, action-mask SHA-256, remaining budget, and
termination reason. STOP may terminate before the edit budget; exhausted budget
forces a STOP-only support.

## Reward vector and advantages

The development Oracle is evaluated only after an on-policy trajectory is
sampled. Logged output includes raw reward components, risk-adjusted reward,
constraint status, Oracle agreement, edit distance, and the action trajectory.
Disagreement/uncertainty, novelty, edit cost, repeated motifs, extreme GC,
extreme length, and all-negative STOP preference are recorded. These remain
heuristic proxies, not experimental translation or stability results.

For each source's `G` trajectories, every objective is standardized separately.
Near-constant objectives receive zero advantage and increment the audit count.
Only then are advantages combined with the preference vector. The profile keeps
both independent and aggregate advantage vectors.

## Clipping, KL, and Edit-Flow replay

The update is a real clipped policy objective: each trajectory advantage is
broadcast to its sampled actions and uses `exp(new_log_prob-old_log_prob)` with
PPO-style clipping. This is not the Stage 3 teacher z-score scalarization.

The reference checkpoint is loaded separately, set `requires_grad=False`, and
never placed in the optimizer. Exact masked categorical `KL(current || ref)` is
logged. The adaptive controller raises its coefficient above target KL and
skips updates/reduces learning rate above `max_kl`.

`--flow-replay-ratio` / `--flow-loss-weight` are represented by
`flow_replay_ratio` / `flow_loss_weight` in `TrainGRPOConfig`; when selected,
the Stage A Edit-Flow loss is mixed into the minimized loss. Compare GRPO-only
(`beta_kl=beta_flow=0`), GRPO+KL, and GRPO+KL+flow replay as controlled runs.

## Failure handling and scientific limits

The trainer detects non-finite loss, clips gradients, supports deterministic
seed, reward/advantage clipping, checkpoint resume, gradient accumulation, and
CUDA autocast mixed precision (`--mixed-precision`). Development-mode output is local JSONL plus checkpoint. Paper
mode rejects functional GRPO conclusions because no real independent Oracle is
connected. No SOTA or biological-performance claim is made here.

## Verification

```bash
cd /home/cunyuliu/mrna_editflow_goal
PYTHONPATH=/home/cunyuliu/mrna_editflow_goal \
  /home/cunyuliu/miniconda3/envs/editflow/bin/python3 -m pytest -q \
  mrna_editflow/tests/test_stage5_constrained_grpo.py
```
