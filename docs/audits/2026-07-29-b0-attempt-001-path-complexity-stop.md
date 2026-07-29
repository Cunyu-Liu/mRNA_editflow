# B0 attempt 001 exact-path complexity stop

## Decision

`B0_attempt_001_20260729T125546Z` is `FAILED_WITH_EVIDENCE`.
B0 is `SAFE_PAUSED`; it is not accepted, frozen, or a scientific result.
D1 remains frozen and unchanged.

The failed attempt is retained at:

`/mnt/cunyuliu/mrna_editflow_d1_b0/D1_B0_20260728T160012Z_8862125/attempts/B0_attempt_001_20260729T125546Z`

No existing result was overwritten, no unrelated process was terminated, and
the ENCODE repair download was not monitored during this diagnosis.

## Immutable failure evidence

| Evidence | SHA-256 |
|---|---|
| `status.json` | `56132bf3c392421cf7965934a7cb230d647bde258b9c6962eb7aa915a0480d89` |
| `failure/failure.json` | `2c8455c26a0566da17ba69697af7aa7020fd84ba8ff3f096e3c88101a1696ecf` |
| `terminal.lock` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `audit/01_canonical_validation/audit_manifest.json` | `cd5a8e7737633491a094c76cdb986cccf189a0e02a959c61b4167fb2b67c1bc8` |
| `audit/01_canonical_validation/logs/stderr.log` | `15d3e46de8ce3927fd98bed3b14cb89a7d175eb9e0dab4aff4676fd1b94cad57` |

The terminal status is `FAILED_WITH_EVIDENCE`, exit code 1, at audited node
`01_canonical_validation`. Its direct driver reason is
`AUDITED_NODE_FAILED_01_canonical_validation`; the retained stderr identifies
the underlying stop rule as `STOP_RULE_B0_PATH_STATE_COMPLEXITY`. The original
driver ran from clean commit
`11c3fa2946c50691108293f537bee7836e0a54bb`.

Frozen inputs used for diagnosis:

| Input | SHA-256 |
|---|---|
| D1 canonical `records_with_labels.jsonl` | `68fca987bf9775df2ff74975fd60b7ce4574f774aa0f44e1c05988387d38bd44` |
| D1 `edit_script_ambiguity_report.json` | `6ffd36fd238a03b77ba1ac97a75a1fcc9997cb617fed107df825c56b19810e68` |

## Original trigger and post-failure exact attribution

The formal attempt used the pre-correction runtime implementation. It stopped
when that implementation exceeded its `max_neighbor_expansions=5,000,000`
primitive-action guard. It did not emit the 95,217-state diagnosis below and
did not reach the 50,000-state guard before stopping.

After the terminal failure bundle was sealed, a read-only exact diagnostic
recomputation used the optimized geodesic-action implementation without
changing the frozen inputs or any production gate. The resulting attribution
is a post-failure diagnostic, not an attempt-001 output and not B0 acceptance
evidence. The exact first-witness regression below supports the safe pause.
The broader capacity census is retained only as unpackaged context because its
one-off replay command and independent log were not persisted. Its
machine-readable record is:

`artifacts/stages/D1_B0_20260728T160012Z_8862125/B0/path_complexity_diagnosis.json`

The first witness is
`GSE217518:record:025e56d3b64660abb559dcbd`, canonical line 39,913:
5′UTR, 129 nt to 114 nt, exact edit distance 15, pure deletion, and 2,340
minimum character alignments.

Because the edit distance equals the length difference, every minimum path is
deletion-only. An exact subsequence-prefix/suffix enumeration, cross-checked
against exhaustive primitive-edit oracles on bounded small-sequence cases,
gives:

- 95,217 unique sequence states;
- 95,215 constructed intermediate states;
- 751,771 sequence-identity transitions;
- 3,934,510,691,993 state paths after coordinate-equivalent transition
  collapse;
- reachable-state digest
  `900076096ad75979a1b592b6d14fd7647dfe54c39b4cee80a053937de9411332`.

This exceeds the existing fail-closed `max_reachable_states=50,000`. No state
was truncated, sampled, or approximated.

## Post-failure frozen-universe capacity diagnosis

The one-off post-failure run reported 144 multi-edit records: 138 pure-indel
and six mixed-edit. It reported:

- maximum single-record closure: 95,423 states
  (`GSE217518:record:7f0dcbc493517a1775cda972`);
- three pure-indel records above 50,000 states;
- exact union of the 138 pure-indel closures: 350,127 states;
- split-eligible unique endpoints: 77,366;
- global near-neighbor state universe lower bound before the six mixed-edit
  closures: 427,217 states.

The six mixed-edit closures were not expanded. The run therefore interpreted
427,217 as a conservative global state-universe lower bound and the transition
and DP-cell figures as conservative lower bounds over the three largest
disjoint pure-indel closures.

These capacity numbers are not independently replayable from a persisted
command and log. They do not participate in the `SAFE_PAUSED` decision, do not
establish whether a simple resource-budget increase would be sufficient, and
cannot authorize any gate or budget change. Before recovery, they must be
recomputed by a persisted, reviewed diagnostic run.

## Narrow code correction retained

The associated code change does not raise any resource limit:

- B0 revalidation of the frozen D1 ambiguity report now computes exact
  minimum-alignment counts without expanding the stronger B0 state closure.
- General geodesic neighbors are enumerated from exact optimal-alignment
  edges.
- Pure insertion/deletion neighbors use exact subsequence certificates.
- The 50,000-state stop remains in force and the real witness is a regression
  test that must stop under the default, then reproduces the full 95,217-state
  closure only under an explicit diagnostic allowance.

## Additional release blockers found independently

These did not cause attempt 001 to fail, but they must also be repaired before
any future `B0:FROZEN` claim:

- the canonical completion code manifest binds an older code commit rather
  than the actual B0 runtime commit;
- completion v2 validates D1 and B0 separately but does not cross-bind the D1
  acceptance referenced by B0 to the D1 artifact in the completion release;
- completion v2 does not commit and validate the terminal B0 driver seal,
  runtime code/input manifests, checksum index, five split manifests, five
  leakage reports, three track manifests, and Data Card as one durable
  evidence snapshot;
- no deterministic production completion-manifest builder exists;
- the task registry names three aggregate B0 files that are neither required
  by the user contract nor produced by the current five-split/five-report
  driver, so the registry must be corrected or real aggregate indices must be
  added before B0-02/B0-03 can be verified.

## Recovery boundary

Recovery requires a separately reviewed exact design for symbolic or
streaming global connectivity, or explicit approval to revise all affected
operational resource budgets. It must preserve all-shortest-path leakage
semantics and pass exhaustive parity tests. A persisted replayable capacity
diagnostic is required before either recovery path can authorize a fresh
formal attempt. The completion/release blockers above must then be closed. It
is forbidden to unblock B0 by dropping the witness, selecting one traceback,
sampling paths, deleting intermediate states, or weakening any leakage gate.
