# GSE246381 data-status clarification

Date: 2026-08-03

This note corrects the earlier description that treated GSE246381 as a data-quality
or data-validity problem. The current committed exposure ledger was re-read from
the protected main project data and contains 1,184 GSE246381 rows, all classified as
paired `D_C` / `E2` records with `exposure_status=unexposed`,
`historically_exposed=false`, and labels allowed for new training and new
hyperparameter selection. The D1 audit reports dataset coverage and edit-script
verification as passed.

This is a description correction only; no raw data, labels, or ledger rows are
modified by this note. The active contract's separate historical-exposure/E4 field
is provenance/admission policy and must not be used as evidence that the data are
invalid. Until the contract owner explicitly amends that policy field, downstream
stages must record both the current data-ledger state and the contract-admission
state rather than silently collapsing them.
