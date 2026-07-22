# RL Stage 3: auditable vector rewards

## Scope

Stage 3 replaces fixed multi-objective scalarization as the primary teacher
contract with serializable reward vectors. It does not implement online GRPO,
does not change the Oracle definition, and does not turn heuristic Oracle
outputs into experimental translation, stability, or half-life measurements.

## Reward schema and hard constraints

Each component records `name`, `value`, `source_model`, `category`,
`independent`, `uncertainty`, and `valid`. Categories cover functional,
stability, structure, manufacturability, safety, edit cost, and hard
constraints. Protein identity, CDS frame, start/stop codons, and editable
region are hard action gates; preference combination rejects invalid gates and
never exchanges them against soft reward.

Teacher JSONL now retains `raw_absolute_level`, `raw_delta_from_source`,
`normalized_within_group`, uncertainty/agreement, validity, redundancy groups,
and a full `reward_vector`. STOP has zero raw deltas. Consequently a
within-pool normalized winner cannot beat STOP when every candidate's raw
delta is negative.

## Objective policy

- TE remains enabled.
- MRL is recorded as a derived, non-independent component and defaults to
  zero weight.
- GC is an interval-target penalty (`gc_constraint`), not an unconstrained
  reward for more GC.
- uAUG is a safety component.
- Accessibility remains a structural component and is labelled as sharing
  Oracle features with TE.
- CAI is invalid in UTR-only tasks, so it produces no objective-head gradient;
  it is enabled only for CDS-editing tasks.

## Redundancy diagnostics

`rl/reward_diagnostics.py` computes Pearson, Spearman, rank agreement,
constant-objective warnings, and redundancy groups. The exporter CLI writes:

```text
docs/reward_correlation_report.json
docs/reward_correlation_report.md
```

These reports must accompany teacher artifacts. Highly correlated objectives
must not be described as independent, and TE plus its deterministic MRL
transform must not receive duplicated default weighting.

## Preference-conditioned ranker

The ranker stores independent affine objective heads for `te`, `access`,
`gc_constraint`, `uaug`, `cai`, `edit_cost`, and `uncertainty` when those
components are valid in teacher rows. Objective-pair losses remain separate;
a Dirichlet-sampled preference supplies an additional temporary combined loss,
not a permanent single ordering. Checkpoints contain `objective_schema`,
`objective_head_state`, and `preference_training`.

Supported profiles are `balanced`, `translation_focused`,
`stability_focused`, `manufacturing_focused`, and `custom`. A preference is
combined only after hard-gate validation. `--uncertainty-penalty` applies
mean-minus-kappa-times-uncertainty; `--minimum-oracle-agreement` rejects lower
agreement teacher candidates.

For inference, `load_objective_head_from_checkpoint()` restores the objective
schema and independent head state; `score_objectives()` yields per-action
`Q_*`, and `combine_objective_scores()` applies a named or custom preference.

## Compatibility and verification

Legacy scalar teacher JSONL remains loadable: it is converted to a minimal
`RewardVector` without fabricated uncertainty or redundancy assertions.

```bash
cd /home/cunyuliu/mrna_editflow_goal
PYTHONPATH=/home/cunyuliu/mrna_editflow_goal \
  /home/cunyuliu/miniconda3/envs/editflow/bin/python3 -m pytest -q \
  mrna_editflow/tests/test_stage3_vector_reward.py \
  mrna_editflow/tests/test_baselines_ablation.py \
  mrna_editflow/tests/test_training_sampling.py
```

On 2026-07-22 the full Stage 3 regression selection (including the Stage 1
decoder, Stage 2 validation protocol, and evaluation suites) completed with
`174 passed`; the sole warning is PyTorch's existing nested-tensor prototype
warning.
