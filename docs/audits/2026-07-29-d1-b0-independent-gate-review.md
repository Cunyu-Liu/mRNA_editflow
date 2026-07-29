# D1/B0 Independent Gate Review

Date: 2026-07-29
Stage family: `D1_B0_20260728T160012Z_8862125`
Contract SHA-256:
`c3dc5875868d847b8519fee40b14c43b65e4c5948dc5c3b98101ca61a5671dd5`

## Decision

The first D1 production build must not be frozen solely on the pre-review
validators. Its immutable output is retained as attempt evidence. D1 and B0
remain locked until every blocking item below has a negative regression test
and full-scope production evidence.

This review changes neither the UTR-only scientific question nor the role of
Edit Flow. It tightens evidence binding and prevents false-positive gates.

## Blocking findings and required remediation

| ID | Severity | Finding | Required gate |
|---|---|---|---|
| GATE-01 | P0 | D1 required reports were not cryptographically compared with the build manifest or recomputed from frozen stores. | Every required path, byte count and SHA-256 must match the build manifest; semantic totals must be recomputed. |
| GATE-02 | P0 | Library ascertainment was represented by free text rather than the contract-required proposal, coverage, bias and shortcut audits. | All required fields and audits must be computed, or carry a stable reasoned `BLOCKED`/`NOT_APPLICABLE` state. |
| GATE-03 | P0 | B0 metadata overlap policy could relabel real sequence-cluster/scaffold/gene/context overlap as explained without a corresponding disjoint estimand. | Freeze independent label-free disjoint partitions and validate raw as well as explained overlap counts. |
| GATE-04 | P0 | Minimum-path closure omitted legal action-order permutations such as the `AC` intermediate for `AA -> CC`. | All states reachable by equivalent minimum scripts must be included, or the partition must fail closed. |
| GATE-05 | P0 | Track A label sealing checked candidate IDs but not finite measured values, endpoint identity, canonical binding or a real pre-label selection freeze. | Privileged verification must independently bind strict hidden-label rows, the canonical store, D1 acceptance and the selection-freeze artifact. |
| GATE-06 | P1 | D1 config bytes were frozen without validating exact scope, contract/scope hashes or label-free role selection. | Validate the complete config semantics and exact twelve-dataset D1 scope. |
| GATE-07 | P1 | Track B/C manifests could bypass intended open-world and allowed-action semantics. | Enforce exact per-track schemas and constrained edit feasibility, including `STOP`. |
| GATE-08 | P1 | Registry completion accepted formatted commit strings and arbitrary PASS text without validating Git objects or phase-specific acceptance. | Bind phase gates to parsed D1/B0 acceptance, committed artifact blobs and a published remote ref. |
| GATE-09 | P1 | GSE200304 aggregate pair accounting did not provide row-level lineage for both raw input tables. | Account for every construct and label row by fingerprint, normalized target or stable rejection/disposition reason. |
| GATE-10 | P1 | `paper_clean` integrity did not establish semantic correspondence with canonical/rejected/auxiliary outputs. | Recompute paper-clean schema, identity and count correspondence. |
| GATE-11 | P1 | Decision records overstated explicit user approval of implementation-level dataset and split details. | Record the actual authorization basis and decision owner; do not use a decision log to weaken a hard gate. |
| GATE-12 | P1 | The preflight manifest did not constitute a completion manifest. | Produce a separate completion manifest binding phase acceptances, artifacts, protection recheck, commits, publication and stop reason. |
| GATE-13 | P0 | Registry and completion validators accepted a minimal self-declared acceptance document rather than the complete D1/B0 semantic result. | Reject incomplete acceptance documents, require every nested production gate, and bind the exact committed bytes to the phase validator and artifact inventory. |
| GATE-14 | P0 | A locally forged `refs/remotes/*` ref could be presented as GitHub publication even when the repository had no remote. | Query the canonical remote with `git ls-remote`, fail closed on network/identity/ref mismatch, and bind the immutable release ref to the code, evidence, registry and completion blobs. |
| GATE-15 | P0 | B0 accepted supplied zero-leakage counts without recomputing leakage from the bound structural records and split manifests. | Recompute all five leakage reports in-process, bind the structural record bytes/content/universe, and require supplied and recomputed semantic evidence to agree exactly. |
| GATE-16 | P0 | B0 accepted an arbitrary twelve-row exposure ledger with any non-empty status, without binding it to the accepted D1 required artifact. | Bind path/bytes/SHA-256 to the D1 build manifest, recompute the D1 ledger payload and enforce the frozen exposure vocabulary and GSE246381 role. |
| GATE-17 | P1 | Track B/C validation bound identifiers but not the complete task payload, source-disjoint role, sequence, endpoint, action set or the frozen edit-budget protocol. | Reconstruct expected tasks from structural records and frozen roles, compare complete payloads, and require the predeclared budgets 1, 3 and 5 without narrowing Track B's future formal generative role. |
| GATE-18 | P1 | A Data Card containing only headings and empty prose could pass. | Bind structured counts, bias, exposure, allowed claims and unsupported capabilities to recomputed benchmark facts; treat Markdown as a rendered view, not the sole gate. |

## Scientific claim boundary

- A repaired D1 gate is a data-reconstruction qualification, not an Edit Flow
  efficacy result.
- A repaired B0 gate is a frozen benchmark/split/track qualification, not a
  measured improvement, SOTA, prospective or full-legal-action-space result.
- Foundation exposure remains `UNKNOWN_PENDING_FM0`,
  `allowed_claim=NONE`, and requires FM0 re-audit.
- The first failed or superseded attempt remains immutable and must never be
  deleted or relabelled as accepted evidence.

## Additional blocking findings from adversarial re-review

These findings were discovered after the first hardening pass. They remain
blocking until the implementation, negative regression and full-scope evidence
all agree; a unit-test repair alone is not a phase PASS.

| ID | Severity | Finding | Required gate |
|---|---|---|---|
| GATE-19 | P1 | A valid structured Data Card could have contradictory measured-improvement or SOTA prose appended and pass after its outer hash was resealed. | Independently render the one canonical Data Card from recomputed facts and require exact full-file bytes, including prose. |
| GATE-20 | P0 | A split could shrink its declared axis/role-pair firewall and expand its explained-overlap policy, then reseal the manifest and leakage report. | Rebuild the canonical per-partition axes, role pairs and overlap policy from split identity and compare the complete semantic objects. |
| GATE-21 | P0 | Valid benchmark records could be moved arbitrarily into `excluded` with invented reasons across all five manifests, shrinking the benchmark while preserving self-consistent hashes. | Rebuild all five complete canonical manifests from the bound structural records and compare eligibility, exclusions, reasons, roles, policies and universe exactly. |
| GATE-22 | P0 | The sealed on-disk role policy could permanently disable Track B's future formal evaluation while the separately derived claims and Data Card remained correct. | Compare the complete selection-bound role-policy object with the independently derived canonical policy. |
| GATE-23 | P0 | Track A's label seal could cite D1 acceptance/build hashes different from the D1 exposure-binding chain. | Require exact equality of both D1 hashes across the label seal, D1 acceptance, build manifest and exposure binding. |
| GATE-24 | P0 | D1 artifacts and their own manifests could be replaced and consistently resealed because acceptance did not bind the immutable audited-builder causal chain. | Verify exact audited argv, Git snapshot, non-neural execution boundary, zero exits, every audit-file hash, single-line stdout equality to the live build manifest and recursive live output hashes; publish the accepted evidence before B0 unlock. |
| GATE-25 | P0 | D1 semantic checks did not yet prove an exact candidate-store projection, independently recomputed ambiguity, complete paper/raw correspondence or the frozen GSE246381 split/exposure role. | Recompute these facts from the frozen stores and fail on any self-declared or role-shrinking substitute. |
| GATE-26 | P0 | The production D1 data root is external, while the completion validator assumed it was the repository acceptance directory; copying only the acceptance would not protect the external stores. | Keep the true external root, bind it through a compact committed root index and build/audit hashes, recheck live bytes at completion, and reject path escape, symlink or post-acceptance mutation. |

## D1 closure re-review

The findings above were remediation requirements, not a record that D1 had
already passed. After the hardening changes, the full production acceptance,
canonical snapshot freeze and independent exact recomputation were rerun
against the same immutable D1 production artifacts.

### Frozen evidence

- Code commit:
  `5030431933b22d6fafb2a3c8a917552b0f416b72`.
- Evidence commit:
  `a674912f4667bb0b88e244b2836599fef4bdba2c`.
- D1 acceptance:
  `artifacts/stages/D1_B0_20260728T160012Z_8862125/D1/acceptance.json`,
  SHA-256
  `f6da5ab89de1128ba157916ea7015e7bfb95d75c465e44ff5d68e65d6f824c05`.
- Canonical snapshot:
  `data/d1/manifests/d1_canonical_snapshot.json`, SHA-256
  `d9f82bf0fa249b533fc97993015ebe3f8a81016c855da6c9e89437d0204eebcd`.
- Published ref:
  `refs/heads/d1-b0-utr-v2-20260729` contained the evidence commit when
  queried from the canonical GitHub remote.

### Failure lineage retained

The first two freeze attempts remain immutable failed evidence:

- `snapshot_freeze_001`: `FAILED_WITH_EVIDENCE`; it exposed a missing
  `stage_id` binding in the initial snapshot implementation. Audit-manifest
  SHA-256:
  `8fa25a76f7153afbcf53f50bf1383faf0be41da198b17853dfabd3c9dc85e19a`.
- `snapshot_freeze_002`: `FAILED_WITH_EVIDENCE`; it exposed the missing
  `explicit_prelaunch_file_manifest` value in the closed JSON-schema enum.
  Audit-manifest SHA-256:
  `a62ff9556096957d07c50075559678382f312a4c05b93ae54fe3c6ac0a3a6d68`.

Neither failure produced or overwrote the canonical snapshot. The repaired
attempt `snapshot_freeze_003` completed with child and wrapper exit code zero,
empty stderr, an exact prelaunch Git fingerprint and no GPU activity because
the workload is non-neural. Its audit-manifest SHA-256 is
`2758f3c1b3f2e58fda364613689406d03aa29770ebc82621a7c0f6b8b452e836`.

The independent read-only `snapshot_validation_001` recomputed the snapshot
from the committed code and live bound artifacts. It returned
`status=PASS`, `errors=[]`, `scientific_result_claimed=false`, with empty
stderr. Its audit-manifest SHA-256 is
`0581a4336fd6e3de4e0f313e275e1d238463b57754fbb81c45b9646e2fa6be10`.

### Dataset disposition and claim boundary

The exact twelve-dataset scope is preserved. Four datasets are structurally
accepted: `GSE114002`, `GSE200304`, `GSE217518` and `GSE246381`. Eight remain
blocked, including both ENCSR854RUF representations. No blocked dataset was
promoted to obtain a positive result. `GSE246381` remains historically exposed
retrospective evidence and is not trainable or threshold-selecting evidence.

This closure qualifies D1 structural data only. It is not an Edit Flow
efficacy result, a prospective result, a SOTA result, or foundation-unseen
evidence. B0 remains a separate gate and may start only after the governed
registry validates `D1-08:FROZEN`.
