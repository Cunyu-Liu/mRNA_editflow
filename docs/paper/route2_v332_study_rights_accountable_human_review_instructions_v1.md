# Route 2 V3.3.2 accountable-human study-rights review instructions

## Purpose and boundary

This decision register records accountable human judgment for the 14 reused
third-party studies. Machine-collected provider evidence is frozen in
`docs/paper/route2_v332_study_rights_accountable_human_review_packet_v1.csv`. It supports accession citation and an
analysis/publication route; it is not a study-specific licence and does not
authorize this project to redistribute source payloads.

Do not use Development TEST, new final Evaluation, sealed GSE246381,
E-MTAB-10902 outcomes, generated-candidate outcomes or guided XEditFlow output
for this review. Rights and non-outcome metadata are sufficient.

## Reviewer action

For each row, an accountable reviewer must identify themselves and the review
date; resolve the accession and non-outcome metadata; verify the data scope and
primary dataset citation; record the exact study-specific rights source and
terms; decide analysis/publication use; decide whether redistribution is
`NOT_AUTHORIZED`, `AUTHORIZED_EXACT_FILES` or `NOT_APPLICABLE`; check the
selected target-journal policy; approve the Data Availability wording; and add
an accountable sign-off.

Use `AUTHORIZED_EXACT_FILES` only when `authorized_exact_file_scope` names the
exact files covered by the reviewed authority. A general provider policy,
converter declaration or prior operational setting is not sufficient. Use
`HOLD` with a concrete reason when the evidence is unresolved. Keep public
payload release closed for every pending or held row.

## Validation and release boundary

After a human edits a copy of the CSV, validate it with this builder's
`--review-input` mode and write a new audit. Validation checks completeness and
internal consistency; it does not authenticate the reviewer or independently
adjudicate the legal conclusion. Even 14 completed rows do not automatically
authorize a public release: the exact release files, stable repository/version,
code licence, tracked legacy-payload policy and final project release decision
remain separate gates.

## Current manuscript wording boundary

Reused source accessions and aggregate evidence may be cited. Upstream study
payloads must not be redistributed by this project until accountable review and
the separate release decision authorize exact files. No availability-on-request
promise should be added without a durable responsible route and explicit access
conditions.
