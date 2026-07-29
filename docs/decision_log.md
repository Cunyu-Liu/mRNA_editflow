# Decision Log

Decision records are append-only. Core scientific changes require explicit
user approval and a new record.

## D-2026-07-28-UTR-V2

```yaml
decision_id: D-2026-07-28-UTR-V2
date: "2026-07-28"
old_text: >-
  public_intervention_contract_v1 made local-effect prediction the active
  scientific question, treated Flow as a later conditional value test,
  included active CDS/full-length work, and described GSE246381 as sealed.
new_text: >-
  utr_editflow_goal_v2 makes source-conditioned continuous-time Edit Flow the
  primary method, restricts the current scope to 5′UTR and 3′UTR, forbids new
  wet-lab work, and records GSE246381 as historically exposed retrospective
  external evidence.
reason: >-
  The user supplied mrna_latest_build_contract_v2.md as the only active
  scientific and engineering execution contract and explicitly required C0
  and D0 implementation without lowering its gates.
evidence:
  - docs/contracts/mrna_latest_build_contract_v2.md
  - artifacts/stages/C0_D0_20260728T120329Z_9f43133/preflight_manifest.json
affected_tasks: [C0-01, C0-02, C0-03, C0-04, C0-05, D0-01, D0-02, D0-03, D0-04, D0-05]
requires_rerun: >-
  Any future result that relied on public_intervention_contract_v1 cannot be
  relabelled as V2 evidence; it must be requalified or rerun under a frozen V2
  task/run manifest.
approved_by_user: false
authorization_basis: overall_D1_B0_execution_request
decision_owner: implementation_agent
user_approval_status: not_explicitly_obtained_for_this_dataset_admission_detail
requires_separate_user_scientific_fork: false
```

Historical files were moved unchanged into version-labelled archive
directories. No old result, log, or Git history was deleted or rewritten.

## D-2026-07-29-D1-B0-SCOPE

```yaml
decision_id: D-2026-07-29-D1-B0-SCOPE
date: "2026-07-29"
old_text: >-
  D0 retained GSE217518 as a candidate whose canonical Ref/Mut sequence
  mapping had not yet been recovered.
new_text: >-
  D1 admits the uniquely paired GSE217518 Ref/Mut constructs from the
  publication's official code data as source-candidate evidence for both
  five_utr and three_utr. All single-ended or ambiguous endpoint groups remain
  rejected. Canonicalization removes only publication-grounded, exactly
  matching assay primer sequence; raw oligos remain immutable and separately
  referenced.
reason: >-
  The official Figure4 SHdiNT_U3.csv and SHdiNT_U5.csv files contain full
  assayed sequences, endpoint role in seqName, and processed half-life fields.
  A label-independent identifier audit found 1,124 unique one-Ref/one-Mut
  three_utr groups and 1,756 unique one-Ref/one-Mut five_utr groups. The study
  publication specifies 115-bp variant-centered inserts and the assay primer
  sequences. Dataset admission was based on reproducible sequence pairing and
  provenance, not effect direction, magnitude, significance, or split outcome.
evidence:
  - data_registry/d1_dataset_scope_manifest.yaml
  - artifacts/stages/D1_B0_20260728T160012Z_8862125/D1/input_inventory.json
  - data/d1/pipelines/GSE217518/README.md
affected_tasks: [D1-04, D1-05, D1-06, D1-07, D1-08, B0-02, B0-03, B0-04, B0-05]
requires_rerun: >-
  Any change to pairing normalization, primer boundaries, or official source
  bytes invalidates the D1 snapshot and every downstream B0 split and leakage
  artifact.
core_scientific_question_changed: false
final_labels_used_for_dataset_admission: false
approved_by_user: true
```

The D1/B0 workload is data reconstruction and benchmark construction, not
formal neural training or neural GPU validation. No CPU fallback exception was
created; the GPU-only rule remains unchanged for all later formal neural work.

## D-2026-07-29-B0-SPLIT-ESTIMANDS

```yaml
decision_id: D-2026-07-29-B0-SPLIT-ESTIMANDS
date: "2026-07-29"
old_text: >-
  The first B0 implementation made every metadata axis jointly disjoint across
  train, validation and test, required three independent studies per region,
  and represented cross-region transfer as one undifferentiated partition.
new_text: >-
  Source/candidate/all-minimum-alignment state components are the mandatory
  split atoms. Study-disjoint evaluation is frozen as label-independent
  leave-one-study-out folds, with each outer-test study wholly excluded from
  that fold's training and selection. Cross-region evaluation freezes three
  separately reported required strata: GSE217518 five-to-three and
  three-to-five within-study directional transfer, plus GSE114002 five_utr to
  GSE200304 three_utr cross-study joint-domain transfer.
reason: >-
  Independent read-only audit showed that joint union over study, scaffold,
  library, context, gene, barcode and sequence groups imposed an undocumented
  compound-disjoint estimand and incorrectly blocked source-disjoint records.
  LOSO avoids selecting one favourable held-out study after label access.
  Keeping all cross-region strata avoids silently choosing between a
  within-study region-associated estimand and a cross-study domain-transfer
  estimand. Every non-source overlap remains explicit and can still fail the
  unexplained-overlap gate.
claim_boundary:
  - two-study LOSO is retrospective out-of-study cross-validation, not a globally sealed external test
  - n_study_2 macro summaries are descriptive, not robust study-level inference
  - within-study strata do not isolate region from assay or library
  - cross-study strata confound region with study, assay and context
  - cross-endpoint strata permit endpoint-agnostic generative metrics only
  - no stratum may be deleted after seeing results
evidence:
  - docs/plans/2026-07-29-d1-b0-utr-editbench-v2.md
  - data_registry/d1_dataset_scope_manifest.yaml
affected_tasks: [B0-02, B0-03, B0-04, B0-05]
requires_rerun: >-
  Any change to the required fold/stratum set, overlap policy, selection-access
  boundary, or record universe invalidates all B0 split, leakage, track and
  acceptance artifacts.
core_scientific_question_changed: false
headline_estimator_selected: false
approved_by_user: false
authorization_basis: overall_D1_B0_execution_request
decision_owner: implementation_agent
user_approval_status: pending_ratification
requires_separate_user_scientific_fork: true
```

## D-2026-07-29-GOVERNANCE-METADATA-CORRECTION-01

```yaml
decision_id: D-2026-07-29-GOVERNANCE-METADATA-CORRECTION-01
date: "2026-07-29"
record_type: append_only_metadata_correction
corrects:
  - decision_id: D-2026-07-29-D1-B0-SCOPE
    field_corrections:
      approved_by_user:
        recorded_value: true
        corrected_value: false
      authorization_basis:
        recorded_value: absent
        corrected_value: overall_D1_B0_execution_request
      decision_owner:
        recorded_value: absent
        corrected_value: implementation_agent
      user_approval_status:
        recorded_value: absent
        corrected_value: not_explicitly_obtained_for_this_dataset_admission_detail
      requires_separate_user_scientific_fork:
        recorded_value: absent
        corrected_value: false
  - decision_id: D-2026-07-29-B0-SPLIT-ESTIMANDS
    field_corrections:
      user_approval_status:
        recorded_value: pending_ratification
        corrected_value: >-
          not_required_for_contract_conforming_label_independent_implementation_freeze
      requires_separate_user_scientific_fork:
        recorded_value: true
        corrected_value: false
reason: >-
  The user authorized execution of D1 and B0 under the frozen contract but did
  not separately approve the detailed GSE217518 admission record. The B0
  implementation retains every contract-required split family and reporting
  stratum, uses no final labels for selection, and changes neither the core
  scientific question nor an acceptance gate. It is therefore an
  implementation freeze inside the existing authorization, not a pending
  scientific fork. This correction changes governance metadata only and does
  not rewrite either historical record.
scientific_content_changed: false
historical_records_modified: false
requires_rerun: false
boundary:
  - >-
    Any future removal or substitution of a required split family, axis,
    stratum, metric, label-access boundary, or acceptance gate requires a new
    explicit user-approved scientific decision.
  - >-
    This correction does not convert engineering or structural evidence into
    efficacy, SOTA, foundation-unseen, or final scientific evidence.
evidence:
  - docs/contracts/mrna_latest_build_contract_v2.md
  - docs/plans/2026-07-29-d1-b0-utr-editbench-v2.md
  - docs/audits/2026-07-29-d1-b0-independent-gate-review.md
affected_tasks: [D1-04, D1-05, D1-06, D1-07, D1-08, B0-02, B0-03, B0-04, B0-05]
approved_by_user: false
authorization_basis: overall_D1_B0_execution_request
decision_owner: implementation_agent
user_approval_status: not_explicitly_obtained_for_corrected_details
requires_separate_user_scientific_fork: false
```

## D-2026-07-29-GOVERNANCE-METADATA-CORRECTION-02

```yaml
decision_id: D-2026-07-29-GOVERNANCE-METADATA-CORRECTION-02
date: "2026-07-29"
record_type: append_only_metadata_correction
corrects:
  - decision_id: D-2026-07-28-UTR-V2
    field_corrections:
      approved_by_user:
        recorded_value: false
        corrected_value: true
      authorization_basis:
        recorded_value: overall_D1_B0_execution_request
        corrected_value: explicit_user_contract_adoption_request
      decision_owner:
        recorded_value: implementation_agent
        corrected_value: user
      user_approval_status:
        recorded_value: not_explicitly_obtained_for_this_dataset_admission_detail
        corrected_value: explicit_in_C0_D0_execution_request
reason: >-
  The D-2026-07-28-UTR-V2 record concerns adoption of the user-supplied V2
  contract as the sole active contract. The user explicitly ordered that
  adoption for C0 and D0. Dataset-admission metadata was later attached to this
  record by clerical error; that metadata belongs to the separate GSE217518
  implementation record corrected above.
scientific_content_changed: false
historical_records_modified: false
requires_rerun: false
evidence:
  - docs/contracts/mrna_latest_build_contract_v2.md
  - artifacts/stages/C0_D0_20260728T120329Z_9f43133/preflight_manifest.json
affected_tasks: [C0-01, C0-02, C0-03, C0-04, C0-05, D0-01, D0-02, D0-03, D0-04, D0-05]
approved_by_user: false
authorization_basis: append_only_correction_of_explicit_user_authorization_record
decision_owner: implementation_agent
user_approval_status: not_required_for_clerical_metadata_correction
requires_separate_user_scientific_fork: false
```

## D-2026-07-29-D1-FREEZE-CLOSURE

```yaml
decision_id: D-2026-07-29-D1-FREEZE-CLOSURE
date: "2026-07-29"
record_type: phase_gate_closure
decision: >-
  Freeze the D1 canonical structural-data snapshot after full-scope semantic
  acceptance, commit-bound code provenance, two retained failed freeze
  attempts, one successful exclusive freeze, independent exact recomputation,
  and canonical GitHub publication of the evidence commit.
reason: >-
  The final D1 acceptance passes every structural, provenance, rejection,
  exposure, report, global-store and audited-builder predicate. The canonical
  snapshot exactly recomputes from code commit
  5030431933b22d6fafb2a3c8a917552b0f416b72 and is the hash-matching blob in
  evidence commit a674912f4667bb0b88e244b2836599fef4bdba2c.
failure_lineage:
  - snapshot_freeze_001: FAILED_WITH_EVIDENCE_stage_identity_binding_missing
  - snapshot_freeze_002: FAILED_WITH_EVIDENCE_schema_enum_missing_formal_prelaunch_source
  - snapshot_freeze_003: COMMAND_COMPLETED
  - snapshot_validation_001: PASS_exact_live_recomputation
dataset_disposition:
  accepted: [GSE114002, GSE200304, GSE217518, GSE246381]
  blocked:
    - ENCSR854RUF_raw62
    - GSE145046
    - GSE149487
    - GSE173083
    - GSE207584
    - GSE291719
    - GSE330741
    - MPRAu_processed_ENCSR854RUF
claim_boundary:
  scientific_result_claimed: false
  model_efficacy_claimed: false
  biological_improvement_claimed: false
  sota_claimed: false
  prospective_validity_claimed: false
  gse246381_role: historically_exposed_retrospective_external_stress_test
  encode_role: blocked_observational_or_pretraining_candidate_not_intervention_labels
evidence:
  - artifacts/stages/D1_B0_20260728T160012Z_8862125/D1/acceptance.json
  - data/d1/manifests/d1_canonical_snapshot.json
  - docs/audits/2026-07-29-d1-b0-independent-gate-review.md
  - /mnt/cunyuliu/mrna_editflow_d1_b0/D1_B0_20260728T160012Z_8862125/attempts/D1_attempt_004_20260728T230404Z/audit/snapshot_freeze_001
  - /mnt/cunyuliu/mrna_editflow_d1_b0/D1_B0_20260728T160012Z_8862125/attempts/D1_attempt_004_20260728T230404Z/audit/snapshot_freeze_002
  - /mnt/cunyuliu/mrna_editflow_d1_b0/D1_B0_20260728T160012Z_8862125/attempts/D1_attempt_004_20260728T230404Z/audit/snapshot_freeze_003
  - /mnt/cunyuliu/mrna_editflow_d1_b0/D1_B0_20260728T160012Z_8862125/attempts/D1_attempt_004_20260728T230404Z/audit/snapshot_validation_001
affected_tasks: [D1-01, D1-02, D1-03, D1-04, D1-05, D1-06, D1-07, D1-08]
core_scientific_question_changed: false
gate_lowered: false
failed_evidence_deleted_or_relabelled: false
approved_by_user: false
authorization_basis: overall_D1_B0_execution_request
decision_owner: implementation_agent
user_approval_status: not_required_for_contract_conforming_D1_gate_closure
requires_separate_user_scientific_fork: false
```

## D-2026-07-29-B0-PATH-COMPLEXITY-STOP

```yaml
decision_id: D-2026-07-29-B0-PATH-COMPLEXITY-STOP
date: "2026-07-29"
record_type: failed_with_evidence_safe_pause
decision: >-
  Retain B0 attempt 001 as FAILED_WITH_EVIDENCE and safe-pause B0 without
  changing the all-shortest-path leakage scope or any acceptance gate.
reason: >-
  The formal attempt first stopped at the existing 5,000,000 primitive-action
  guard. A post-failure exact, regression-tested recomputation attributes the
  same witness to a 95,217-state closure above the unchanged 50,000-state
  guard. This witness and the immutable formal failure support SAFE_PAUSED.
  Broader one-off capacity numbers are context only because their replay
  command and independent log were not persisted.
attempt:
  id: B0_attempt_001_20260729T125546Z
  runtime_commit: 11c3fa2946c50691108293f537bee7836e0a54bb
  state: FAILED_WITH_EVIDENCE
  node: 01_canonical_validation
  status_sha256: 56132bf3c392421cf7965934a7cb230d647bde258b9c6962eb7aa915a0480d89
  failure_sha256: 2c8455c26a0566da17ba69697af7aa7020fd84ba8ff3f096e3c88101a1696ecf
stop_semantics:
  approximation_emitted: false
  state_or_record_dropped: false
  gate_lowered: false
  failed_evidence_deleted_or_relabelled: false
  b0_accepted: false
  b0_frozen: false
  scientific_result_claimed: false
recovery_boundary:
  - exact symbolic or streaming global connectivity with exhaustive parity evidence
  - or explicit approval and audit of every affected operational resource budget
  - either path first requires a persisted replayable capacity diagnostic
forbidden_recovery:
  - choose one traceback
  - sample paths
  - drop the witness
  - omit constructed intermediate states
  - weaken path or near-neighbor leakage
evidence:
  - docs/audits/2026-07-29-b0-attempt-001-path-complexity-stop.md
  - artifacts/stages/D1_B0_20260728T160012Z_8862125/B0/path_complexity_diagnosis.json
  - /mnt/cunyuliu/mrna_editflow_d1_b0/D1_B0_20260728T160012Z_8862125/attempts/B0_attempt_001_20260729T125546Z
affected_tasks: [B0-01, B0-02, B0-03, B0-04, B0-05]
core_scientific_question_changed: false
approved_by_user: false
authorization_basis: contract_stop_rule_and_failure_preservation
decision_owner: implementation_agent
user_approval_status: required_before_crossing_the_recorded_stop_boundary
requires_separate_user_stop_boundary_decision: true
requires_separate_user_scientific_fork: false
```
