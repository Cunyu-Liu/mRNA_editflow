# Phase 2 reliable Local-Delta Oracle protocol

This protocol is the executable contract for `PairedDeltaFormer`.

## Scientific boundary

The only training records eligible for measured local-delta supervision are
records with `task_kind=local_delta`, `data_layer=C_source_matched_intervention`
and `local_delta_eligible=true`. Proxy records are used only in Stage A and
must retain `confidence=proxy` and their absolute-property provenance. They are
never mixed into the measured-only fine-tuning or calibration stage.

Training reads `train` and `val` only. Final roles are opened by the separate
evaluator only after a candidate freeze manifest is supplied and
`--allow-final-labels` is explicit.

## Fair comparison

`small`, `frozen_foundation` and `partial_foundation` share the same sequence
feature contract, Siamese/cross-attention adapter, context/edit encoders,
uncertainty/ranking head, train/val manifests, stage step budgets, seeds,
calibration protocol and one-configuration search budget. A foundation run is
scientifically eligible only when a real local checkpoint and SHA256 are
recorded. The adapter accepts either a local HuggingFace RNA model or the
repository's Stage-A mRNA-pretrained trunk, but the latter is explicitly
labelled `internal_stage_a_mrna_pretrained` and still requires a separate
pretraining-corpus leakage audit. The training runner and final evaluator
require that audit to attest zero exact eligible final-sequence substrings and
to match the checkpoint SHA256. The adapter stub is smoke-only.

The audit is intentionally not a family-level or semantic-independence proof;
those remain separate scientific requirements. An audit with any exact overlap
is fail-closed and cannot support a foundation scientific claim.

## Gate

`test_v2_untouched` requires Spearman >= 0.35, sign accuracy >= 0.68,
top-10% enrichment >= 1.75, beneficial precision >= 0.75 and ECE <= 0.10.
The independent assay requires Spearman >= 0.25, enrichment >= 1.40 and
beneficial precision >= 0.65. Empty or non-local-delta axes are blockers, not
zeros that can be reported as a scientific failure of the model.

## Forced route when the gate fails

The runner records error stratification and an active-learning acquisition
queue. The model contract already supports context-specific projections,
assay-specific embeddings/random-effect inputs and measured-only hierarchical
post-hoc calibration. The queue is an experimental acquisition request; it is
not a wet-lab result. Until source-matched measurements arrive, the scientific
claim remains blocked and the domain may only be narrowed in a preregistered
follow-up.
