# D1 + B0 UTR EditBench V2 Implementation Plan

> **Execution contract:** Follow
> `docs/contracts/mrna_latest_build_contract_v2.md` at SHA-256
> `c3dc5875868d847b8519fee40b14c43b65e4c5948dc5c3b98101ca61a5671dd5`.
> Do not enter B0 before the D1 snapshot is accepted and frozen.

**Goal:** Reconstruct auditable 5′UTR and 3′UTR source/candidate records, quantify
edit-path ambiguity and exposure, then freeze schemas, leakage-safe split
manifests, Track A/B/C roles, and a Data Card without opening new external final
labels or presenting engineering gates as scientific results.

**Stage:** `D1_B0_20260728T160012Z_8862125`

**Isolated branch/worktree:**

- branch: `d1-b0-utr-v2-20260729`
- base: `8862125c6a00fabaff97cbeb0b3160265261c585`
- worktree:
  `/mnt/cunyuliu/mrna_editflow_goal_worktrees/d1-b0-utr-v2-20260729`
- external stage data:
  `/mnt/cunyuliu/mrna_editflow_d1_b0/D1_B0_20260728T160012Z_8862125`

**Scientific boundary:** D1/B0 are data and benchmark gates. They do not train a
neural model and cannot establish Edit Flow efficacy, biological improvement,
SOTA, or prospective validity. Formal neural training remains GPU-only and is
out of scope for this stage.

## Task 1: Preserve preflight and stage authority

**Files**

- Create:
  `artifacts/stages/D1_B0_20260728T160012Z_8862125/preflight_manifest.json`
- Create:
  `artifacts/stages/D1_B0_20260728T160012Z_8862125/protected_state.json`
- Create:
  `artifacts/stages/D1_B0_20260728T160012Z_8862125/contract_goal.sha256`
- Create: `schemas/stage_manifest.schema.json`
- Create: `scripts/execution/validate_stage_manifest.py`
- Create: `tests/test_stage_manifest_v2.py`

**Acceptance**

- Record original and isolated Git state, dirty diff hash, protected PIDs, GPU,
  RAM, disk, input manifests, and ENCODE partial status.
- Record zero process termination and zero mutation of the original worktree.
- Keep ENCODE raw reconstruction observational-only while incomplete.
- Non-training stage manifests must not fabricate CUDA, checkpoint, or seed
  evidence.

## Task 2: Harden phase-gate evidence

**Files**

- Modify: `schemas/task_registry.schema.json`
- Modify: `scripts/execution/validate_registry.py`
- Modify: `tests/test_validate_registry.py`
- Modify: `docs/execution/task_registry_v2.yaml`
- Modify: `docs/decision_log.md`

**Acceptance**

- D1 tasks descend from `C0-05` and `D0-05`.
- B0 tasks cannot become `VERIFIED/FROZEN` before `D1-08:FROZEN`.
- Verified/frozen D1/B0 tasks require a real commit SHA, acceptance artifact,
  matching artifact hash, passing predicates, and no unresolved blockers.
- Smoke/fixture evidence cannot satisfy a D1/B0 phase gate.
- `allowed_parallel_tasks` is reciprocal; only explicitly non-conflicting work
  may overlap.

## Task 3: Implement canonical edit scripts

**Files**

- Create: `data/utr_benchmark_v2/__init__.py`
- Create: `data/utr_benchmark_v2/edit_script.py`
- Create: `tests/test_edit_script_v2.py`

**Acceptance**

- Support `SUB`, `INS`, `DEL`, and `STOP` with dynamic current-state
  coordinates.
- Reject no-op, invalid allele, out-of-range coordinate, invalid alphabet, and
  actions after `STOP`.
- Deterministically canonicalize a minimal script while separately quantifying
  equivalent minimal alignments in repeated sequence.
- Round-trip fixtures for 5′UTR and 3′UTR pass.

## Task 4: Freeze D1 dataset scope and provenance

**Files**

- Create: `data_registry/d1_dataset_scope_manifest.yaml`
- Create: `data/d1/pipelines/<dataset_id>/manifest.yaml`
- Create: `data/d1/pipelines/<dataset_id>/README.md`
- Create:
  `artifacts/stages/D1_B0_20260728T160012Z_8862125/D1/input_inventory.json`

**Dataset roles**

- `GSE114002`: primary 5′UTR source-paired measured subset plus separately
  labelled absolute/variable-length prior.
- `GSE200304`: primary 3′UTR source-paired measured substitution set;
  processed-label-only status retained.
- `GSE217518`: primary 5′UTR/3′UTR source-paired measured set from the
  publication's official code data; accept only uniquely paired Ref/Mut
  constructs after exact, region-specific primer removal, and reject all
  unpaired endpoints with stable reason codes.
- `GSE246381`: secondary 5′UTR sequence-pair asset, permanently E4
  retrospective; no training, selection, or untouched claim.
- `MPRAu_processed_ENCSR854RUF`: conditional primary 3′UTR/DEL asset only after
  frozen hg19/reference reconstruction reaches 100% round-trip.
- `GSE145046`: absolute dense landscape only until scaffold recovery.
- `GSE149487`: blocked canonical pairs until exact design/sequence mapping is
  recovered.
- `ENCSR854RUF_raw62`: observational-only; never intervention evidence.
- `GSE330741`, `GSE291719`: metadata-only; candidate/final labels remain
  unopened.
- `GSE173083`, `GSE207584`: excluded from current UTR-only phase.

**Acceptance**

- Every D0 capability row has exactly one D1 role, allowed use, blocked claim,
  license, exposure, and reason.
- Every included input has byte size and SHA-256.
- Incomplete ENCODE raw files cannot satisfy any D1 data-completeness
  predicate.

## Task 5: Build paper-clean, canonical, and rejected records

**Files**

- Create dataset-local contract entrypoints:
  `download.py`, `extract.py`, `paper_clean.py`, `canonical_clean.py`,
  `build_source_candidate.py`, `build_edit_scripts.py`,
  `reproduce_labels.py`, `audit_library_design.py`, `audit_exposure.py`
- Create: `scripts/data/build_d1_utr_benchmark.py`
- Create: `scripts/data/validate_d1_acceptance.py`
- Create: `tests/test_d1_utr_benchmark.py`

**Storage separation**

- Raw files remain immutable under `data/p0` or their existing historical raw
  path.
- Full canonical records and rejected rows are written under the immutable
  external stage data root.
- Label-free candidate records are a physically separate output.
- Label-bearing records are never required by candidate selection loaders.

**Acceptance**

- GSE114002 raw counts, mother groups, WT anchors, accepted records, and stable
  row-level rejection reasons reconcile exactly.
- GSE200304 construct/pair/label counts reconcile exactly; the reported 6,892
  versus available 6,885 pair discrepancy is explicit.
- GSE246381 RefSequence/AltSequence round-trip is exact and remains E4.
- GSE217518 Ref/Mut pairs are unique, fixed assay primers are removed by
  publication-grounded rules, all unpaired endpoints are rejected, and every
  retained 5′UTR/3′UTR pair round-trips exactly.
- MPRAu is either reconstructed against a frozen reference with 100%
  round-trip or remains explicitly blocked; no partial DEL claim.
- No source is invented for GSE145046/GSE149487.
- Provided-label-only and processed-only statuses are not written as raw label
  reproduction.

## Task 6: Produce and validate all D1 required artifacts

**Files**

- Create: `data/data_exposure_ledger.jsonl`
- Create: `data/library_ascertainment_report.json`
- Create: `data/edit_script_ambiguity_report.json`
- Create: `data/measured_action_coverage_report.json`
- Create: `reports/data_reproduction/summary.csv`
- Create: `data/d1/manifests/d1_canonical_snapshot.json`
- Create:
  `artifacts/stages/D1_B0_20260728T160012Z_8862125/D1/acceptance.json`

**Acceptance**

- All paper-eligible records have raw/processed provenance.
- `apply(edit_script, source) == candidate` is 100%.
- Path ambiguity is quantified without calling constructed paths observed.
- Source/candidate/endpoint mapping is reproducible.
- All rejected rows have stable reason codes.
- Raw/paper-clean/canonical outputs are distinct and hashed.
- Absolute sequence records are not interventions.
- Measured `INS=0` and observed trajectories `=0` remain explicit.

## Task 7: Freeze B0 schemas

**Files**

- Create: `schemas/utr_edit_record.schema.json`
- Create: `schemas/edit_script.schema.json`
- Create: `schemas/generation_task.schema.json`
- Create: `tests/test_utr_edit_schemas_v2.py`

**Acceptance**

- All frozen D1 records validate.
- Invalid region, endpoint collapse, fake observed path, invalid action,
  missing provenance, and label-bearing candidate-store records fail closed.

## Task 8: Generate deterministic split manifests

**Files**

- Create: `data/utr_benchmark_v2/split_graph.py`
- Create: `scripts/data/build_b0_splits.py`
- Create: `splits/5utr_source_disjoint.json`
- Create: `splits/5utr_study_disjoint.json`
- Create: `splits/3utr_source_disjoint.json`
- Create: `splits/3utr_study_disjoint.json`
- Create: `splits/cross_region_transfer.json`
- Create: `splits/split_registry_v2.yaml`
- Create: `tests/test_b0_split_manifests_v2.py`

**Acceptance**

- Assignment is label-independent and deterministic.
- Source/candidate/intermediate connected components are atomic.
- Source, study, sequence cluster, scaffold, gene, context, barcode, and
  library-batch groups are never silently ignored when missing or overlapping.
- Source-disjoint manifests make sequence-state components atomic and disclose
  every non-source overlap as predeclared explained or gate-failing
  unexplained overlap; they do not silently impose compound disjointness.
- Each study-disjoint file freezes label-independent leave-one-study-out folds.
  In every fold, the complete held-out study is final test, development labels
  from that outer-test study are unavailable to selection, and the remaining
  study is divided into source/state-disjoint train and validation components.
  Results are reported per fold and as a preregistered descriptive study macro;
  with two studies this is not a dataset-global hidden-final-label or robust
  study-level inferential claim.
- The cross-region file freezes all required strata rather than selecting a
  favourable estimand after seeing results:
  - GSE217518 5′→3′ within-study directional transfer;
  - GSE217518 3′→5′ within-study directional transfer;
  - GSE114002 5′→GSE200304 3′ cross-study joint-domain transfer.
  These strata are reported separately. The first two permit only predeclared
  study overlap; the third is limited to endpoint-agnostic generative metrics
  and explicitly confounds region with study/assay/context. None supports an
  isolated region-effect claim.
- If any required fold or stratum cannot be built without source/state/path
  leakage, the corresponding manifest is fail-closed and B0 remains
  `SAFE_PAUSED`; no random-pair or post-hoc subset substitute.

## Task 9: Audit all leakage axes

**Files**

- Create: `data/utr_benchmark_v2/leakage.py`
- Create: `scripts/data/audit_b0_leakage.py`
- Create: `evaluation/leakage_audit_v2.json`
- Create: `evaluation/foundation_pretraining_overlap_audit.json`
- Create: `tests/test_b0_leakage_v2.py`

**Acceptance**

- Exact source/candidate overlap, reverse edge/path leakage, and final endpoint
  as training intermediate are zero.
- Sequence-cluster, scaffold, gene, study, context, barcode/library batch
  status is explicit for every split.
- Every fold/stratum has an independent content hash and leakage report, while
  all five top-level manifests bind the same frozen canonical and structural
  record universe.
- A structural store with unchanged record IDs but changed sequence/provenance
  content fails before split assignment.
- Foundation overlap remains `UNKNOWN_PENDING_FM0`,
  `allowed_claim=NONE`, `requires_fm0_reaudit=true`; it is never described as
  unseen or no-overlap.
- Exposure ledger coverage is 100%.

## Task 10: Freeze Track A/B/C and Data Card

**Files**

- Create: `data/utr_benchmark_v2/track_loader.py`
- Create: `evaluation/tracks/closed_measured_pool.yaml`
- Create: `evaluation/tracks/heldout_generative.yaml`
- Create: `evaluation/tracks/open_legal_generation.yaml`
- Create: `evaluation/tracks/track_role_matrix.yaml`
- Create: `docs/data/UTR_EditBench_v2_Data_Card.md`
- Create: `scripts/data/validate_b0_acceptance.py`
- Create:
  `artifacts/stages/D1_B0_20260728T160012Z_8862125/B0/acceptance.json`
- Create: `tests/test_b0_tracks_and_acceptance_v2.py`

**Acceptance**

- Track A candidate IDs/hashes are frozen without final-label values.
- A privileged freeze proof verifies the real label-store bytes/hash and exact
  candidate-label ID bijection without exposing label contents to selection
  loaders.
- Exposure, all three tracks, the role matrix, Data Card, split manifests and
  leakage evidence bind the same frozen D1 universe.
- GSE246381 is retrospective only.
- Track B does not claim measured functional improvement.
- Track C is explicitly computational/predicted.
- Every record has one unambiguous track role.
- Data Card reports counts, biases, exposure, allowed claims, blocked claims,
  and unsupported capabilities.
- Exact B0 counters are:
  `unexplained_overlap=0`, `reverse_path_leakage=0`,
  `final_endpoint_as_train_intermediate=0`,
  `exposure_ledger_coverage=100%`, `track_role_ambiguity=0`.

## Task 11: Verify, freeze, commit, and publish

**Verification**

- Run focused D1 tests and production D1 build.
- Run D1 semantic validator; only then mark `D1-08:FROZEN`.
- Run B0 schema/split/leakage/track tests and production build.
- Run B0 semantic validator.
- Run the existing C0/D0 contract, registry, exposure, and execution
  regression tests.
- Run strict active-contract audit.
- Inspect exact changed paths and ensure no unrelated raw data or artifacts are
  staged.

**Publication**

- Commit only confirmed D1/B0 code and audit evidence.
- Push branch `d1-b0-utr-v2-20260729` to GitHub.
- If B0 is safe-paused, commit and push the truthful blocker evidence without
  marking B0 complete.
