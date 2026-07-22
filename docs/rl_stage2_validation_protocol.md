# RL Stage 2: proposal-ranker validation protocol

## Scope

This protocol evaluates the offline Bradley-Terry proposal ranker. It does not
implement policy optimization, PPO, GRPO, a learned STOP head, or a new Oracle.
Teacher deltas remain heuristic Oracle-guided local-search labels and must not
be described as measured translation efficiency, half-life, or assay results.

## Data separation

Training requires four distinct inputs:

```text
--train-records-jsonl --train-teacher-jsonl
--val-records-jsonl   --val-teacher-jsonl
```

The process fails closed if train and validation share a transcript ID, an
exact full-sequence SHA-256, or a supplied `family_cluster`,
`family_cluster_id`, or `cluster_id`. In paper mode both record files are
verified against the immutable split contract's `train` and `val` roles, and
their teacher provenance must match the corresponding role. Validation rows
are only forwarded under `torch.no_grad()` and never contribute gradients.

## Metrics

For each transcript, candidates are ordered by model log action score. Teacher
delta means `teacher_score = Oracle(candidate) - Oracle(source)`.

- `mean_model_regret`, `median_model_regret`: teacher-best delta minus the
  model top-1 delta, averaged or medianed over records.
- `oracle_best_recall_at_k`: fraction of records whose teacher-best candidate
  is contained in model top-k (ties count as a hit).
- `ndcg_at_k`: DCG over non-negative teacher deltas, with gain
  `max(delta, 0) / log2(rank + 1)`, normalized by the ideal teacher ordering.
- `positive_edit_precision_at_k`: fraction of model top-k candidates with
  `teacher_delta > 0`.
- `mean_selected_teacher_delta`: mean teacher delta of model top-1.
- `stop_accuracy`: among records with no positive non-STOP teacher action,
  the fraction for which model top-1 is STOP. The denominator is recorded.

`candidate_cap=0` denotes the full supplied legal candidate pool and is the
only setting allowed to use the unqualified global-regret names above. Any
positive cap emits `restricted_*` metric names and `candidate_pool_scope` of
`restricted`; it cannot be used for global regret or the default checkpoint
metric.

## Checkpoint and early stopping

The default checkpoint selector is:

```text
--checkpoint-metric val_mean_model_regret --checkpoint-mode min
```

The supported alternatives are `val_oracle_best_recall_at_32` and
`val_ndcg_at_32` with `--checkpoint-mode max`. At each
`--validation-interval`, the profile records `train_loss`, validation metrics,
and `grad_norm`. Checkpoints are saved only when the validation metric improves
by `--minimum-improvement`; training loss never selects a checkpoint. Optional
`--early-stopping-patience` counts non-improving validation intervals only.

Each selected checkpoint records `best_validation_metric`,
`best_validation_step`, `validation_summary`,
`training_loss_at_best_step`, and `early_stopping_reason`.

## Paper-reporting boundary

Future papers may report these metrics only for a verified, transcript/family/
sequence-disjoint validation or held-out test role, with candidate-pool scope,
candidate cap, Oracle provenance, checkpoint-selection metric, and split
contract disclosed. They describe alignment to the stated heuristic teacher
over legal local proposals; they do not establish biological efficacy.

## Verification

```bash
cd /home/cunyuliu/mrna_editflow_goal/mrna_editflow
/home/cunyuliu/miniconda3/envs/pc_cng/bin/python -m pytest -q \
  tests/test_stage2_validation_protocol.py tests/test_training_sampling.py \
  tests/test_split_contract_enforcement.py
```

Baseline result after implementation: `6 passed` for the focused Stage 2
protocol tests, `34 passed` for ranker/sampling plus split-contract regression,
`92 passed` for `tests/test_eval.py`, and `16 passed` for the CDS/protein hard
constraint suites. These are correctness checks only; they make no performance
claim.
