# GOAL-V3-DATA-BENCH-01 — G7 Fresh Closure & Goal Terminal

- generated_at_utc: 2026-08-04T06:27:47.828944+00:00
- git_head: 7682bd30893f1c51c3a21aa0f2738699b0750da8
- g7_snapshot_id: g7_snapshot_20260803_001
- g7_run_id: g7_r_v1
- contract_sha256: 35dd4bf27a3c7d574ab777f5d858ad1b13dcb9273bdb4961e4c30a1a94bf8759
- terminal_status: **BLOCKED_WITH_EVIDENCE**
- done_generated: False
- gp0_status: LOCKED_NOT_AUTHORIZED
- resource_viability_status: NOT_VIABLE
- split_assignments: 856986
- fresh_assignment_summary: {"by_split": {"3utr_sequence_cluster_disjoint": 856986}, "five_utr_e_studies": 0, "five_utr_e_units": 0, "five_utr_f_units": 0, "five_utr_source_units": 0, "five_utr_study_units": 0, "source_or_study_disjoint_assignments": 0}

## Stage status

| Stage | Status | Evidence |
|-------|--------|----------|
| C3 | PASS (historical) | frozen contract/schemas/registries; marked STALE_INVALIDATED for G7 PASS |
| D0 | PASS (historical) | asset/license registry; marked STALE_INVALIDATED for G7 PASS |
| D1 | PASS (fresh re-run) | d1 validator exit=0 total_errors=0 |
| FM0-A | PASS (fresh re-run) | fm0 validator exit=0 total_errors=0 |
| B0 | PASS (reused B0_VALIDATOR.log + fresh light checks) | validator=PASS total_errors=0 light_errors=0 |
| Unit tests | PASS=True | pytest exit=0 |

## Benchmark partition root cause

The fresh D1 projection uses only provenance-bound grouping atoms and the fresh B0 builder materializes grouped assignments. Missing atoms remain INELIGIBLE_WITH_REASON; no sentinel group IDs are created. Assignment summary: {"by_split": {"3utr_sequence_cluster_disjoint": 856986}, "five_utr_e_studies": 0, "five_utr_e_units": 0, "five_utr_f_units": 0, "five_utr_source_units": 0, "five_utr_study_units": 0, "source_or_study_disjoint_assignments": 0}. DB_01 is closed only when a non-empty source/study-disjoint assignment is evidenced; resource viability remains an independent gate.

## Resource viability

- status: **NOT_VIABLE**
- denominators: {"ordinary_e_pairs": 88042, "ordinary_f_observations": 3322161, "restricted_e_pairs": 1184, "restricted_f_observations": 15392, "cluster_count": 9, "five_utr_e_pairs": 0, "five_utr_f_observations": 0, "split_assignments": 856986, "five_utr_e_studies": 0, "five_utr_source_units": 0, "five_utr_study_units": 0}
- reason: publication-grade candidate not asserted; failed gates: five_utr_independent_units_ge_500,five_utr_studies_ge_5,source_disjoint_partition_nonempty,study_disjoint_partition_nonempty,group_aware_ci_precision_pass,action_specific_strata_ge_100,no_single_study_or_library_over_70_percent

## GSE analytic/final counters

- forbidden_analytic_counts: {}
- all_forbidden_zero: True
- observed_nonanalytic_intents: {"restricted_d1_builder": 1184, "restricted_fm0a_aggregate_audit": 1, "G7_RESTRICTED_FINALIZER": 2}
- g7_closure_event_appended: True
- nonanalytic_machine_event_closed: {"RESTRICTED_BUILDER_PARSE": false, "AGGREGATE_QC_MACHINE": false, "FM_OVERLAP_AGGREGATE": false, "B0_ELIGIBILITY_SPLIT_BUILD": false, "G7_RESTRICTED_FINALIZER": true}
- access_chain_ok: True

## Blocker ledgers

- data_goal_set_equality: True
- model_rebind_set_equality: True
- intersection_empty: True

### data_goal_required_blocker_ids

- **DB_01_SPLIT_GROUPING_ATOMS_MISSING** -> OPEN_WITH_EVIDENCE
- **DB_02_GSE246381_ROW_ISOLATION** -> CLOSED_WITH_EVIDENCE
- **DB_03_DUAL_STORE_CONSERVATION** -> CLOSED_WITH_EVIDENCE
- **DB_04_ACCESS_CHAIN_INTEGRITY** -> CLOSED_WITH_EVIDENCE
- **DB_05_ANALYTIC_FINAL_COUNTERS_ZERO** -> CLOSED_WITH_EVIDENCE
- **DB_06_RESOURCE_VIABILITY_BINDING** -> CLOSED_WITH_EVIDENCE

### model_rebind_handoff_blocker_ids

- **MRB_01_GP0_PAIRED_COUNT_REBIND** -> OPEN
- **MRB_02_MODEL_TRAINING_NOT_AUTHORIZED** -> OPEN
- **MRB_03_SOURCE_BINDING_ORACLE** -> OPEN
- **MRB_04_METHOD_ATTRIBUTION_TESTS** -> OPEN

## Terminal determination

BLOCKED_WITH_EVIDENCE: the fresh data/benchmark result does not meet all terminal requirements; see blocker ledger and failed publication-grade gates. resource_viability_status=NOT_VIABLE. No DONE is generated.

## Next steps for the user

1. Review the fresh grouping-atom coverage and the failed publication-grade gates; acquire/rebuild only the missing provenance-bearing fields before a future B0/G7 rerun.
2. **Narrow the paper scope**: drop the split/anti-leakage benchmark objective and report only the data/engineering/closure transparency results (no model-rebind publication), accepting the LIMITED_DEVELOPMENT_ONLY grade.

## Handoff declarations

- `NO_GP0_TRAINING_PERFORMED`
- `ANALYTIC_FINAL_LABELS_ACCESSED=false`
- `GSE246381_PRIOR_ANALYTIC_USE=NONE_CONFIRMED_BY_OWNER`
- `GSE246381_LEGACY_PIPELINE_MATERIALIZATION=PRESENT`
- `NO_PROJECT_UNLABELED_PRETRAINING`
- `GP0_STATUS=LOCKED_NOT_AUTHORIZED`
