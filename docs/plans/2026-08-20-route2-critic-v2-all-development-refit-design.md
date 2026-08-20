# Route 2 Critic V2 All-Development Refit Design

## Boundary

This design is frozen before Critic V2 three-seed or TEST outcomes. It prepares
one all-Development refit config only after the prospectively frozen single TEST
has completed. It does not schedule the TEST or refit, inspect real metrics, or
authorize LOSO, readiness, guidance or Evaluation.

## Scientific rule

The V3.3.2 contract orders one frozen Development TEST followed by a fixed
structure/loss/policy refit on all 126,165 Development records. It explicitly
forbids using TEST to reselect structure, loss, seed, epoch or threshold. It does
not declare an additional TEST-metric threshold before refit. Therefore the gate
requires a complete, provenance-matched TEST summary and the presence of TEST
metrics, but never branches on their values.

The confirmation policy remains `BEST_VALIDATION`. The single TEST and final
refit both have no Validation loader because Development partitions are folded
into training, so their executable checkpoint policy is the prospectively fixed
100-epoch `FINAL_EPOCH`. This does not use TEST to choose an epoch.

## Gate and output

The dedicated V2 preparer binds the frozen TEST protocol, a separate prospective
refit protocol, the exact seed-20260823 TEST config and its terminal summary. It
requires matching model, loss, seed, baseline identity, full training policy,
record scope, CUDA execution, learned parameter update and zero Evaluation
reads. It rejects an incomplete TEST, any policy/provenance drift and existing
runtime/run targets.

The emitted config uses all 126,165 Development records, seed 20260823, the
frozen 100-epoch training budget, `FINAL_EPOCH`, GPU0-5 and a V2-specific run
identity. It records that TEST outcomes are part of the Development refit scope
but were not used for refit selection. It keeps Evaluation and guidance closed.

## Alternatives rejected

The historical V1 preparer accepts the old TEST identity and validates only a
small subset of policy fields. Extending it for two schemas would preserve an
unneeded compatibility path. Adding a TEST performance threshold would be a
post-outcome rule not stated by the contract. The V2 path is therefore separate
and outcome-value agnostic.

## Verification

Synthetic focused tests cover the valid refit, deliberately poor TEST metrics,
incomplete or contaminated summaries, identity/policy drift, GPU range and
single-write behavior. They never read a real TEST or Evaluation artifact.
