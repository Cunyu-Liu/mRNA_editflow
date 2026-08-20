# Route 2 Critic V2 LOSO Aggregation Input Design

## Boundary

This work freezes and implements the V2 aggregation-input path before any V2
LOSO outcome exists. It does not prepare runtime training configs, run LOSO,
build real inputs, aggregate real metrics or open TEST/Evaluation.

## Problem

The shared LOSO aggregator can validate terminal primary and matched-baseline
training provenance, but the existing three-seed input builder hard-codes the
historical V1 `seed..._huber_v1` and `seed..._global_scaled_v1` run names. The V2
preparers emit different run identities and directories. Without a V2 builder,
the 42 terminal V2 folds cannot become the three aggregate results required by
readiness.

## Decision

Freeze a V2 aggregation protocol that binds the primary and matched-baseline
LOSO protocols, exact three seeds, seven nonempty studies, GSE256185 as the
zero-record Development study, output roots and protected-outcome boundary.

The V2 builder reads exactly 21 JSON configs from each runtime-config root. It
indexes them by study and seed, validates the shared physical GPU and explicit
primary/baseline pairing fields, and reads each terminal summary from the
config's `output_directory`. Validation metrics are wrapped as `LOSO::<study>`
evaluations for the existing aggregator. The builder writes three inputs once;
the existing aggregator remains the single implementation of study alignment,
undefined-metric handling and macro improvement.

## Verification

Synthetic tests cover exact 21+21 pairing, successful three-input construction
accepted by the shared aggregator, pairing drift, nonterminal/protected summary
rejection, protocol drift and non-overwrite output. No real outcome is read.
