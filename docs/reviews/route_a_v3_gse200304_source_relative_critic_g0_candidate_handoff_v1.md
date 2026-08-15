# Route A V3 GSE200304 source-relative critic G0 candidate — handoff v1

Status: `DRAFT_FOR_REVIEW_NOT_ACTIVE_PROTOCOL`  
Authority: `NON_AUTHORITATIVE`  
Context: `V3-DEC-028`, runtime `A1-EVT-061`

This candidate prepares the static interface for the future exactly-one
`GSE200304_SOURCE_RELATIVE_CRITIC_G1` run. It validates the named-study
candidate-minus-source estimand, biological-source-group unit, missing/nonfinite
exclusion, antisymmetric pair mean, positive uncertainty scale and calibration
LCB formula. It also freezes a future single-split, single-seed, single-fit,
single-terminal-checkpoint, zero-refit execution shape.

The architecture plan uses full-length masked source/candidate visibility and an
antisymmetric shared-encoder pair difference. It does not reuse the legacy
100-position fixed-prefix implementation, because DEC028 requires every edit to
remain visible. This is a static shape contract only: no parameter tensor is
constructed here.

The command-line interface is validate-only. It reads the candidate config and
writes one JSON object to standard output. It has no model, optimizer, data
loader, CUDA path, checkpoint path, split publisher, training loop or runtime
artifact writer.

The baseline set, real membership, real component graph, real split assignments,
materialized rows, final critic gate bundle and independent implementation review
remain pending. Consequently this handoff is preparation evidence only; it is not
P0 PASS, G1 authority, critic PASS, A6 authority, A7, scientific qualification or
a change to `1/1/0/6547`.
