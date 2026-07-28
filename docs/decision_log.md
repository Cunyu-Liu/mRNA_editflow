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
