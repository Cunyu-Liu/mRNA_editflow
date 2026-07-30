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
approved_by_user: true
```

Historical files were moved unchanged into version-labelled archive
directories. No old result, log, or Git history was deleted or rewritten.

## D-2026-07-31-B0-CAPACITY-NONBLOCKING

```yaml
decision_id: D-2026-07-31-B0-CAPACITY-NONBLOCKING
date: "2026-07-31"
old_text: >-
  B0 engineering execution treated exact EditFlow path/state capacity census
  as a blocking production gate. E1 stopped after four exact records exceeded
  50,000 reachable states and a fifth exceeded the bounded DP resource limit.
new_text: >-
  Under utr_editflow_goal_v2.2_b0_capacity_nonblocking, B0 capacity census is
  an optional diagnostic only. New B0 benchmark construction is accepted only
  against the formal B0 split, leakage, exposure, track-seal, artifact-binding
  and Data Card gates; it does not have a B0 capacity-validation failure state.
reason: >-
  The user explicitly authorized removal of all B0 capacity gates and ordered
  a fresh B0 benchmark rebuild. This alters engineering execution policy only;
  the scientific question, Edit Flow role, leakage gates, label seals and
  claim boundary are unchanged.
evidence:
  - docs/contracts/mrna_latest_build_contract_v2_2.md
  - /mnt/cunyuliu/mrna_editflow_b0_capacity/B0_capacity_20260730T043737Z_f8e30a4
historical_evidence_policy: >-
  Retain E1 capacity diagnostics without deletion or relabelling. They are not
  retroactively passed and do not prove or disprove model efficacy.
affected_tasks: [B0-REBUILD-20260731]
requires_rerun: >-
  A new B0 task/run manifest must bind the v2.2 contract hash. No previous B0
  capacity outcome can be used as the acceptance outcome of the new run.
approved_by_user: true
```

## D-2026-07-31-B0-FROZEN-REPLAY-SCOPE

```yaml
decision_id: D-2026-07-31-B0-FROZEN-REPLAY-SCOPE
date: "2026-07-31"
old_text: >-
  B0 split construction and leakage auditing recomputed exact all-order
  shortest-action state closures, which could recreate the removed capacity
  gate while overstating the path scope audited by a benchmark qualification.
new_text: >-
  Under utr_editflow_goal_v2.2_b0_frozen_d1_replay_scope, B0 deterministically
  replays every accepted D1 canonical edit-script prefix plus declared
  intermediates and endpoints. Zero path leakage remains a formal gate exactly
  within this recorded scope; no all-order or all-dynamic-path claim is made.
reason: >-
  The user explicitly removed every B0 capacity gate and ordered a fresh B0
  construction. The replacement preserves an executable, falsifiable leakage
  audit without reintroducing exhaustive path-state enumeration.
evidence:
  - docs/contracts/mrna_latest_build_contract_v2_2.md
  - configs/utr_editflow_contract_v2.yaml
historical_evidence_policy: >-
  Historical E1 capacity outputs remain immutable diagnostics and are neither
  deleted nor relabelled as a pass.
affected_tasks: [B0-REBUILD-20260731]
requires_rerun: >-
  The fresh B0 manifest, split manifests, leakage reports and Data Card must
  record the frozen D1 replay scope and new contract hash.
approved_by_user: true
```
