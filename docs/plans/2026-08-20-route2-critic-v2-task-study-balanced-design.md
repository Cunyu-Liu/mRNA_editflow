# Route 2 Critic V2 Task/Study-Balanced Design

## Status and evidence boundary

This design is frozen before any Critic V2 outcome is read. It uses only the
existing Development TRAIN/VALIDATION records and the already frozen strongest
same-information baseline (`task_macro_spearman=0.13171439492559175`).
Development TEST remains withheld, all external Evaluation outcomes remain
closed, and the existing terminal mRNABERT experiments will not be repeated.

## Hypothesis

The V1 mRNABERT critic is unstable across seeds because its optimization unit
does not match the task-macro selection estimand. Critic V2 keeps the scientific
input hypothesis and capacity fixed, and changes only the TRAIN sampling and
loss aggregation:

1. draw a fixed number of TRAIN examples through the hierarchy
   `task -> study -> source-context-endpoint group -> record`;
2. keep the total number of draws per epoch equal to the TRAIN record count;
3. preserve length-local batching for the existing memory budget;
4. compute Huber loss independently for each task represented in a batch, then
   average those task losses;
5. retain TRAIN-only task-robust target scaling and task-macro checkpoint
   selection.

This is a single prospective repair. It does not add capacity, change the
encoder, search a new loss, add seeds, or use TEST/Evaluation feedback.

## Model and information controls

The full model remains the 9.343M-parameter edit-centered critic over frozen
mRNABERT features. It retains local edit attention/max pooling, whole-sequence
source/candidate background, explicit source/candidate deltas, edit identity,
normalized position, region/context inputs, and exact antisymmetry.

The matched screen contains one full model and three parameter-matched controls:

- `candidate permutation`: candidate assignments are permuted only inside the
  existing exact-source/task support used by the frozen control policy;
- `source-only`: candidate tokens and candidate frozen features are replaced by
  the source, so no edit or candidate information remains;
- `source+edit metadata (anchor-only)`: the source sequence, context, edit
  identity and edit position remain visible, but candidate frozen/global
  mRNABERT representation is replaced by the source representation. The model
  evaluates only the source-anchored direction, so candidate sequence cannot
  enter through the reverse global-background path.

The anchor-only control is therefore not another name for source-only. It asks
whether explicit edits on a source anchor explain the full model without the
candidate's global pretrained representation.

## Frozen screen and gate

The existing strongest same-information baseline is reused rather than rerun.
The four Critic V2 arms use the same screen seed, data, Huber loss, 100 epochs,
optimizer, parameter count, sampling, loss aggregation, and checkpoint policy.
The control gate requires the full model to beat the frozen baseline, source-only
and anchor-only on task-macro Spearman with task breadth, and the permutation
control on the two predeclared tasks where the permutation changes enough TRAIN
candidates to be informative. All inputs must report complete CUDA training,
finite nine-task Validation metrics, no TEST metrics, and zero Evaluation reads.

Only a passing control gate may authorize the already frozen three confirmation
seeds `20260822`, `20260823`, and `20260824`. No fourth seed will be added. A
failing gate is terminal for this Critic V2 hypothesis and leaves TEST, refit,
LOSO, and guided XEditFlow closed.

## Failure handling and verification

Sampler tests will verify deterministic epoch replay, a fixed draw budget, and
near-exact task/study balance. Loss tests will verify equal task contribution
independently of row count. Model tests will verify that anchor-only is
parameter-matched, responds to edit metadata, ignores candidate pretrained
features, and differs from source-only. Gate tests will cover both PASS and
NO-GO without adding re-verification loops.
