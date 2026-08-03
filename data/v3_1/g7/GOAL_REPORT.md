# GOAL-V3-DATA-BENCH-01 — G7 Fresh Closure & Goal Terminal

- generated_at_utc: 2026-08-03T19:33:37.116245+00:00
- git_head: e17ba9b36653a59baf03d4e2785407f0ed919e59
- g7_snapshot_id: g7_snapshot_20260803_001
- g7_run_id: g7_r_v1
- contract_sha256: 35dd4bf27a3c7d574ab777f5d858ad1b13dcb9273bdb4961e4c30a1a94bf8759
- terminal_status: **BLOCKED_WITH_EVIDENCE**
- done_generated: False
- gp0_status: LOCKED_NOT_AUTHORIZED
- resource_viability_status: LIMITED_DEVELOPMENT_ONLY
- split_assignments: 0

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

The benchmark cannot form a usable anti-leakage partition: **all split assignments = 0** because the D1 technical canonical lacks the grouping atoms required by the split contracts (GENE / SEQUENCE_CLUSTER / LIBRARY_LINEAGE / TILE_FAMILY / TRANSCRIPT / STUDY). Every task/split eligibility cell is INELIGIBLE_WITH_REASON, so no source/study-disjoint partition with assignments>0 can be formed. This is a data blocker (DB_01) that cannot be closed inside this Goal.

## Resource viability

- status: **LIMITED_DEVELOPMENT_ONLY**
- denominators: {"ordinary_e_pairs": 88042, "ordinary_f_observations": 3322161, "restricted_e_pairs": 1184, "restricted_f_observations": 15392, "cluster_count": 10, "five_utr_e_pairs": 67601, "five_utr_f_observations": 3021669, "split_assignments": 0}
- reason: split_assignments=0: no non-empty source/study-disjoint partition can be formed because the D1 canonical lacks the required grouping atoms (GENE/SEQUENCE_CLUSTER/LIBRARY_LINEAGE/TILE_FAMILY/TRANSCRIPT/STUDY); 3-UTR scope = EXPLORATORY_ONLY

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

BLOCKED_WITH_EVIDENCE: the data Goal is not fully closed (DB_01_SPLIT_GROUPING_ATOMS_MISSING is OPEN_WITH_EVIDENCE) and resource_viability_status=LIMITED_DEVELOPMENT_ONLY. No DONE is generated.

## Next steps for the user

1. **Extend data**: acquire/rebuild D1 data so the grouping atoms (GENE / SEQUENCE_CLUSTER / LIBRARY_LINEAGE / TILE_FAMILY / TRANSCRIPT / STUDY) are materialized, then re-run B0 eligibility/split/seal and G7. Only then can PUBLICATION_GRADE_CANDIDATE be reassessed.
2. **Narrow the paper scope**: drop the split/anti-leakage benchmark objective and report only the data/engineering/closure transparency results (no model-rebind publication), accepting the LIMITED_DEVELOPMENT_ONLY grade.

## Handoff declarations

- `NO_GP0_TRAINING_PERFORMED`
- `ANALYTIC_FINAL_LABELS_ACCESSED=false`
- `GSE246381_PRIOR_ANALYTIC_USE=NONE_CONFIRMED_BY_OWNER`
- `GSE246381_LEGACY_PIPELINE_MATERIALIZATION=PRESENT`
- `NO_PROJECT_UNLABELED_PRETRAINING`
- `GP0_STATUS=LOCKED_NOT_AUTHORIZED`
