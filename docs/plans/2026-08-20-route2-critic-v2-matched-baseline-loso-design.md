# Route 2 Critic V2 Matched-Baseline LOSO Design

## Boundary

This prospective gate prepares the strongest same-information baseline LOSO
cohort only after the 21 primary Critic V2 LOSO runtime configs exist. It reads
configuration metadata, not primary outcomes. It does not schedule training,
aggregate LOSO, adjudicate readiness, open Development TEST, or read Evaluation.

## What matched means

Each baseline fold is paired exactly to one primary fold by holdout study, seed,
physical GPU and TEST-preserving split. Both use the same original Development
TRAIN/VALIDATION fold construction and withhold all 18,292 Development TEST
records. Matching does not mean pretending the models have equal capacity or
training budget. The baseline retains the hyperparameters that made it the
frozen strongest same-information comparator: the anchored position-aware model,
transferable context, task→source weighting, TRAIN-task robust scaling,
pairwise-Huber objective, 8 epochs, batch 32 and FP32.

The held-out study cannot select a checkpoint, so each baseline fold uses the
prospectively fixed final epoch. The primary and baseline configs explicitly
cross-reference their paired identities and share the same GPU assignment.

## Gate

The V2-only preparer validates both prospective LOSO protocols, the exact frozen
global-scaled base config, and the complete set of 21 primary configs. It rejects
missing/duplicate folds, cohort or GPU drift, protected-outcome access, wrong
policy, and existing baseline config/run targets.

## Alternative rejected

The historical baseline preparer only checks an old three-seed boolean and
accepts arbitrary output roots. It cannot prove that the new Critic V2 TEST and
refit gates occurred or that baseline folds match the new primary configs.
Extending it would add an unnecessary two-schema branch, so V2 uses a dedicated
preparer.

## Verification

Synthetic tests verify exact 21-fold pairing, frozen baseline policy, missing or
substituted primary folds, protected-outcome closure, and one-time writes. No real
training summary or protected outcome is read.
