# RL Stage 4: offline DAgger state aggregation

## Scope

Stage 4 addresses the mismatch between source-only one-step teachers and the
intermediate states reached by multi-step model-guided decoding.  It remains
offline Oracle teacher generation and Bradley--Terry/vector-preference
distillation.  It is not online GRPO: there is no policy-ratio objective, no
on-policy gradient, no KL-regularised update, and no Oracle used to choose a
rollout action.

The implemented relabelling scope is the current audited Stage 3 T5 5'UTR
teacher. Other task grammars fail closed instead of being rolled out under one
grammar and relabelled under another; extending them requires a matching
task-aware vector teacher first.

## Data flow

```text
train source + current checkpoint
  -> constrained model-guided rollout (temperature/top-k/STOP/visited-state guard)
  -> saved trajectory with checkpoint SHA-256
  -> offline Oracle enumeration at each visited intermediate state
  -> reward-vector teacher rows + state-record JSONL
  -> replay buffer (original / rollout / hard-negative / STOP)
  -> existing offline proposal-ranker training
```

`rollout_model_guided_trajectory()` has no Oracle argument.  It uses the same
CTMC action score, task-region grammar, STOP selection, cycle rejection, and
reverse-edit rejection as `sample.model_guided_edit_record()`.  Oracle creation
is confined to `relabel_trajectory()` after the rollout is saved.

## Trajectory and potential contract

Each teacher row includes the source ID, full state sequence, step index,
checkpoint SHA-256, complete action history, candidate action, reward vector,
raw delta, STOP/termination status, and source/state/candidate properties.

The declared transition potential is the heuristic Oracle TE proxy:

```text
r_t = Phi(s_{t+1}) - Phi(s_t),  Phi(s) = TE_proxy(s)
```

It is a bookkeeping label, not an experimental translation measurement.  The
relabeler checks the telescoping identity within numerical tolerance before it
writes artifacts.  STOP receives the Stage 3 zero-delta teacher row and does
not consume edit budget.

## Replay and iteration artifacts

`ReplayMixConfig` defaults to 40% original, 40% rollout, 10% hard-negative,
and 10% STOP.  Sampling is exact (largest-remainder quota); it raises if a pool
cannot support the requested mixture rather than silently changing proportions.
Low-scoring rollout candidates form the default hard-negative pool, or an
explicit hard-negative JSONL can be supplied.

Each iteration is created as `iteration_NNN/` and refuses to overwrite an
existing directory.  It contains trajectories, visited state records, DAgger
teacher rows, replay buffer/manifest, mixed teacher JSONL, ranker profile and
checkpoint, and `iteration_manifest.json`.  The manifest records policy
version, buffer size, state diversity, mean trajectory reward, validation
regret, positive-edit precision, STOP rate, and cycle rejection rate.

Only `train_records` are passed to rollout/export.  The exporter rejects any
source in the validation ID set, and ranker validation continues to use the
unchanged validation records and teacher JSONL.

## Minimal command

```bash
cd /home/cunyuliu/mrna_editflow_goal
PYTHONPATH=/home/cunyuliu/mrna_editflow_goal \
  /home/cunyuliu/miniconda3/envs/editflow/bin/python3 -m mrna_editflow.train_dagger_ranker \
  --train-records-jsonl TRAIN.jsonl \
  --validation-records-jsonl VAL.jsonl \
  --original-train-teacher-jsonl TRAIN_TEACHER.jsonl \
  --validation-teacher-jsonl VAL_TEACHER.jsonl \
  --policy-checkpoint RANKER_OR_STAGE_A.pt \
  --output-root outputs/dagger --iteration 1
```

## Verification

```bash
PYTHONPATH=/home/cunyuliu/mrna_editflow_goal \
  /home/cunyuliu/miniconda3/envs/editflow/bin/python3 -m pytest -q \
  mrna_editflow/tests/test_stage4_dagger.py
```

The test covers model-only action selection, intermediate states, policy hash,
offline relabelling and potential telescoping, STOP retention, replay ratios,
non-overwriting iteration paths, and validation-partition exclusion.

On 2026-07-22, the Stage 1--4 focused regression selection completed with
`181 passed`; the only warning was PyTorch's existing nested-tensor prototype
warning.
