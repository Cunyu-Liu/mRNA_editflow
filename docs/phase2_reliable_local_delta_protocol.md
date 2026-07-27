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
pretraining-corpus leakage audit. The checkpoint SHA256 and pretraining
corpus SHA256 are separate provenance objects: the former identifies the
loaded model, while the latter identifies the corpus scanned by the leakage
audit. The training runner and final evaluator verify both independently and
require zero exact eligible final-sequence substrings. The adapter stub is
smoke-only.

The audit is intentionally not a family-level or semantic-independence proof;
those remain separate scientific requirements. An audit with any exact overlap
is fail-closed and cannot support a foundation scientific claim.

`scripts/build_phase2_leakage_free_foundation_corpus.py` can construct a
prospective Stage-A corpus after excluding exact overlaps. This output is not
a checkpoint; a new Stage-A pretraining run and a fresh audit are still
required before `frozen_foundation` or `partial_foundation` can enter the
scientific comparison.

## Gate

`test_v2_untouched` requires Spearman >= 0.35, sign accuracy >= 0.68,
top-10% enrichment >= 1.75, beneficial precision >= 0.75 and ECE <= 0.10.
The independent assay requires Spearman >= 0.25, enrichment >= 1.40 and
beneficial precision >= 0.65. The registered independent axis is the 703-record
`GSE246381_mouse_Vglut_MPRA_combined_UMI` local-delta subset of `test_ood`,
which differs in assay, batch, cargo and cell context. The `test_assay` role is
not eligible local-delta ground truth and cannot be used for this gate. Empty
or non-local-delta axes are blockers, not zeros that can be reported as a
scientific failure of the model.

`scripts/freeze_phase2_candidate_manifest.py` requires a selection artifact
and an explicit pre-unblinding attestation. Generate the artifact with
`scripts/select_phase2_candidates.py`; its safe loader exposes only sequence,
edit and context inputs, records `labels_accessed=false`, and retains all
registered eligible candidates without a post-hoc score cutoff. The freeze
script verifies its role/alias and no-label candidate digest. Final evaluation
additionally recomputes the safe candidate digest, role-manifest SHA and
canonical records SHA; the boolean attestation alone cannot unlock labels.

`scripts/run_phase2_final_protocol.py` fixes the order operationally: it
chooses a checkpoint using validation-only metrics, creates both label-free
selection artifacts and freeze manifests, and only then invokes the evaluator
with `--allow-final-labels`. If either registered gate fails, invoke
`scripts/run_phase2_remediation.py` with both final reports; its
error-stratification report and source-matched active-learning queue are
mandatory evidence, not a claim of new measurements.

## Forced route when the gate fails

The runner records error stratification and an active-learning acquisition
queue. The model contract already supports context-specific projections,
assay-specific embeddings/random-effect inputs and measured-only hierarchical
post-hoc calibration. The queue is an experimental acquisition request; it is
not a wet-lab result. Until source-matched measurements arrive, the scientific
claim remains blocked and the domain may only be narrowed in a preregistered
follow-up.
