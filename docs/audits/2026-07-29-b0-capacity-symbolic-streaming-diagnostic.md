# B0 exact symbolic/streaming capacity diagnostic audit

## Decision and claim boundary

This work implements and audits a non-formal, label-free B0 path-capacity
diagnostic. It does not start, accept, freeze, or resume a formal B0 attempt.
It changes no scientific endpoint, split, leakage rule, production gate, or
budget. Smoke tests, the frozen witness, and structural capacity measurements
are engineering/capacity evidence only.

The authoritative B0 failure-index state remains
`SAFE_PAUSED_AFTER_FAILED_WITH_EVIDENCE` (shortened below to
`B0 SAFE_PAUSED`). A later budget decision requires explicit authorization
after the exact and lower-bound census results below are reviewed. The
historically exposed external stress test remains excluded from training,
tuning, threshold selection, and model selection.

## Frozen authorization and inputs

- Active contract:
  `docs/contracts/mrna_latest_build_contract_v2.md`,
  SHA-256
  `c3dc5875868d847b8519fee40b14c43b65e4c5948dc5c3b98101ca61a5671dd5`.
- Diagnostic config:
  `configs/b0_capacity_diagnostic_v1.json`,
  SHA-256
  `081c91b7629e48029e30d881c484602ad5e8e667416ede9f010b4d049d1eee67`.
- Authoritative worktree:
  `/mnt/cunyuliu/mrna_editflow_goal_worktrees/d1-b0-utr-v2-20260729`.
- Authoritative external Git directory:
  `/mnt/cunyuliu/mrna_editflow_d1_b0/D1_B0_20260728T160012Z_8862125/release/gitdir_code_hardening_v1.git`.
  The stale native `.git` was not used.
- D1 sealed label-free candidate store:
  `/mnt/cunyuliu/mrna_editflow_d1_b0/D1_B0_20260728T160012Z_8862125/attempts/D1_attempt_004_20260728T230404Z/artifacts/D1/candidate_store/candidates.jsonl`,
  44,151 rows, SHA-256
  `01c9fdaad7de013d2f6d5eeac7620ef8fd35bb3eb5693a9672975540ed701be1`.
- Split-eligible rows: 42,766. Deterministically selected multi-edit census
  rows: 144. Selected-ID digest:
  `21c70b4c2b5ab836c12db293a23a830aff32ecd4b69a5f6ed3d018eedcc82cbc`.
- The diagnostic selection and capacity algorithm never parse label values.
  The canonical label store is opened only as an opaque integrity hash by the
  pre-existing D1 snapshot validator.

The workload is a CPU-only, non-neural exact data benchmark. CUDA is neither
required nor used. Existing unrelated GPU processes were left untouched.
All large output, scratch, wheels, and runtime files live under
`/mnt/cunyuliu`; no large data or weights were written into Git or `/home`.

## Exact implementation

The prototype preserves all shortest-path sequence-identity semantics:

- unique sequence states are materialized layer by layer using bounded,
  deterministic external-memory sorting and merging;
- path multiplicities use unbounded integers;
- reachable nodes, unique transitions, shortest paths, primitive actions,
  state-DP cells, and state-set digests remain distinct counters;
- resource limits emit explicit lower-bound terminal rows or a safe pause and
  never substitute sampling, a single traceback, or an approximation;
- a census may start only from a completed, recursively verified exact frozen
  witness for the same contract, code, data, and config.

The replay contract binds the lexical Python launcher, resolved native
executable, exact `-I -B` flags, empty Python override environment, code
blobs, cwd, exact argv, `jsonschema`, `rfc3339-validator`, and frozen positive
and negative `date-time` semantics. Saved and live probes are compared using
canonical JSON hashes, and integer probe counts reject Python boolean aliases.

Relevant pushed commits on `d1-b0-utr-v2-20260729`:

- `d4e4e014ea4c5d34bf0f619d9c800eda5475ee2e` —
  `feat(b0): add auditable exact capacity diagnostic`;
- `58b60de3b31947486f3b1e24038e37bdf12ede09` —
  `fix(b0): preserve streaming capacity stops`;
- `4d01f19f60ba246956c2754069fac0ad752ca4d9` —
  `fix(b0): bind replay runtime semantics`.

The final code review found no P0 or P1 issue. It independently reproduced
and then verified closure of a boolean-versus-integer probe-equality bug.
Frozen `FROZEN_B0_LIMITS` are byte-equivalent to the parent commit.

## Verification

Final commit `4d01f19f60ba246956c2754069fac0ad752ca4d9`:

- focused local: 94/94 passed;
- focused remote in the sealed `/mnt` runtime: 94/94 passed;
- adjacent local B0 path, near-neighbor, split, leakage, driver, artifact, and
  acceptance suites: 282/282 passed;
- the same adjacent suites remotely: 282/282 passed;
- Black, `py_compile`, AST parsing, JSON validation, and `git diff --check`
  passed;
- the authoritative worktree was clean before each real diagnostic.

These test counts and the later 751.59 s external-validator timing are
session-observed audit evidence summarized here; they are not embedded raw
test logs inside a diagnostic bundle. The diagnostic bundles do contain their
own hash-bound command, stdout/stderr, events, metrics, process result, seal,
and `VERIFIED` evidence.

Two incomplete dependency-stage remote runs were retained in the audit trail:
the first stopped during collection because the new minimal runtime lacked
`PyYAML`; after fixing that exact dependency, the second produced 258 passes
and 24 same-root fixture failures because NumPy was absent. Exact versions
from the prior audited environment (`PyYAML==6.0.3`, `numpy==1.26.4`) were
installed from hashed wheels, followed by the clean 282/282 rerun. These were
test-runtime construction failures, not scientific or algorithmic passes.

## Same-server runtime

Authorizing runtime:

`/mnt/cunyuliu/mrna_editflow_b0_capacity_runtime/B0_capacity_runtime_20260729T220415Z_py310_minimal`

External seal:

`/mnt/cunyuliu/mrna_editflow_b0_capacity_runtime/runtime_seals/RUNTIME_SEAL_20260729T233713Z_py310_minimal`

The final freeze and wheelhouse each contain 17 fixed distributions/wheels.
Every wheel passed SHA-256 verification. The runtime tree was made
non-writable without removing execute/search bits, then bound by a sorted
per-file SHA-256 list plus path/type/mode/uid/gid/size/symlink metadata.
Pre-witness, post-witness, and post-replay trees were identical. The seal also
records the Python ELF, no-RPATH result, resolved dynamic libraries,
`/usr/bin/env`, `/bin/sh`, OS release, kernel, and glibc.

Seal identities:

- 2,703 regular files, file-list SHA-256
  `b1af0ac36891be25fff60fd9816851f6248dd899c1cd0e20672c019d7571494e`;
- 2,953 metadata rows, metadata SHA-256
  `42f63e1d14d859184bb70fa46273f389b1fdcf51991619a082de3e1f21aa57dc`;
- seal-member index SHA-256
  `49f77a4970868b50644718582f2ae91b0a6ad30bfb1215d80beff7a55382aa8d`;
- isolated runtime-identity SHA-256
  `e15a48d5d0513e52afcd4b762d915243a6564922ee425c48bbb26b0d3b9b6ea2`.

The runtime is authorizing only for same-server replay. Its standard library
comes from `/usr/lib/python3.10` and shared libraries from `/lib`; it is not a
container, cross-host, permanent, or bitwise-portable environment.
The full-tree runtime seal is a detached external operator seal referenced by
path and hashes in this audit; it is not recursively embedded as a bundle
member. Bundle-internal runtime identity independently binds and live-replays
the launcher, interpreter, package paths/versions, and date-time semantics.

Preserved construction attempts:

- `/mnt/cunyuliu/mrna_editflow_b0_capacity_runtime/B0_capacity_runtime_20260729T211534Z_py310`:
  redundant slow clone; the SSH session reset, leaving an incomplete preserved
  directory with no `conda-meta` files or executable Python;
- `/mnt/cunyuliu/mrna_editflow_b0_capacity_runtime/B0_capacity_runtime_20260729T220307Z_py310_minimal`:
  preserved failed `venv` attempt because system `ensurepip` was unavailable;
- the final `...T220415Z_py310_minimal` runtime was built with copied system
  Python plus fixed hashed wheels and contains no runtime-critical path under
  `/home/cunyuliu`.

## Preserved superseded and failed diagnostics

No previous bundle was overwritten, renamed, resumed, or deleted:

- `B0_capacity_20260729T191127Z_d4e4e01`: exact witness values, but its replay
  claim is superseded by the launcher audit;
- `B0_capacity_20260729T192247Z_d4e4e01`: safe failed census evidence,
  including the record-80 state-DP lower-bound history;
- `B0_capacity_20260729T204732Z_58b60de`: exact witness values, but
  non-authorizing because its saved replay launcher collapsed the `/mnt`
  virtual-environment semantics to a `/home` interpreter;
- `B0_capacity_20260729T210225Z_58b60de`: cancelled before root creation, so no
  artifact directory exists.

The launcher defect was not hidden: numerical witness evidence remains useful
as superseded development evidence, while its replay authorization is
explicitly rejected.

## Fresh witness and actual replay

Authorizing witness:

`/mnt/cunyuliu/mrna_editflow_b0_capacity/B0_capacity_20260729T235142Z_4d01f19`

It completed with `DONE`, `VERIFIED`, recursive bundle validation, exact
accounting 1/1, and an unchanged runtime seal. Independent validation
recomputed the exact row and global union.

Frozen witness result:

- record `GSE217518:record:025e56d3b64660abb559dcbd`, canonical line 39,913;
- source 129 nt, candidate 114 nt, edit distance 15;
- 2,340 minimum alignments and 3,445 evaluated DAG cells;
- 95,217 reachable states and 751,771 unique transitions;
- 3,934,510,691,993 shortest state paths;
- 1,205,477 primitive actions and 0 state-DP cells;
- state digest
  `900076096ad75979a1b592b6d14fd7647dfe54c39b4cee80a053937de9411332`;
- 92.05 s record compute, 61,730,816-byte peak RSS, and 107,737,552
  spill bytes.

This exact witness exceeds the unchanged frozen
`max_reachable_states=50,000`; therefore it is evidence that the current
production B0 gate does not admit the witness, not evidence that B0 passed.

Actual replay:

`/mnt/cunyuliu/mrna_editflow_b0_capacity/B0_capacity_20260730T000730Z_4d01f19`

The authorizing witness's generated `replay.sh` created this fresh root using
the sealed `/mnt` launcher, `env -i`, and `-I -B`. The replay bundle passed a
fresh full `_validate_bundle` and `_validate_verified_marker`, and the runtime
tree remained identical to the seal afterward.

## Full structural census

Run:

`/mnt/cunyuliu/mrna_editflow_b0_capacity/B0_capacity_20260730T001510Z_4d01f19`

Parent authorization:

`B0_capacity_20260729T235142Z_4d01f19`

The census completed with `DONE`, `VERIFIED`, committed return code 0, empty
captured stdout/stderr, and unchanged post-census runtime seal. It accounted
for all 144 scheduled records:

- 142 `EXACT_COMPLETED`;
- 2 `LOWER_BOUND_STOPPED`;
- 0 failed or unaccounted records;
- 138 pure-indel and 6 mixed-edit records.

The exact 142-record subset has:

| Quantity | Sum | Median | Nearest-rank p95 | Maximum |
|---|---:|---:|---:|---:|
| reachable states | 350,362 | 24 | 4,372 | 95,423 |
| unique transitions | 2,470,759 | 46.5 | 23,115 | 751,771 |
| shortest state paths | 5,070,220,604,294 | 28.5 | 11,069,135 | 3,934,510,691,993 |
| primitive actions | 3,905,519 | 67 | 35,270 | 1,205,477 |
| state-DP cells | 408,718 | 0 | 0 | 389,062 |

Three exact rows exceed the unchanged 50,000-state frozen gate. The maximum
exact row has 95,423 reachable states
(`GSE217518:record:7f0dcbc493517a1775cda972`); the frozen witness has
95,217.

The two lower-bound rows are:

1. ordinal 80, `GSE217518:record:867cbdefe7587abca464e2a5`,
   canonical line 41,400, 121-to-129 nt, edit distance 12, 245,202 minimum
   alignments. The exact closure crossed 100,000,001 state-DP cells versus the
   unchanged frozen 50,000,000 limit. Before stopping it proved at least
   32,912 states and 32,911 transitions. Runtime was 294.71 s and spill was
   163,297,928 bytes.
2. ordinal 122, `GSE217518:record:d36bb3bb896e7171f532da3b`,
   canonical line 42,283, 125-to-115 nt, edit distance 11, 19,380 minimum
   alignments. It crossed the same 100,000,001 state-DP diagnostic stop and
   proved at least 104,827 states and 104,826 transitions. Runtime was
   963.02 s and spill was 110,751,666 bytes.

Both rows retain `exact=null`, `state_set_complete=false`, the observed lower
bound and typed stop rule. No approximation was emitted. The program
continued after each row, which directly verifies the streaming-stop fix.

The global union contains 427,444 unique states with SHA-256
`8b4002b578b84a0a326a57715b3eabdf5e1afc84897bd7c61b6f41994ef4a77e`.
Its role is explicitly
`LOWER_BOUND_FROM_ENDPOINTS_AND_EXACT_COMPLETED_RECORDS`; it is not an exact
full-census universe because the two incomplete closures are omitted.
An independent audit repeated `wc`, SHA-256, sortedness, uniqueness, and the
bounded global-union verification over the 48,948,717-byte artifact.

Compute plus global merge reached terminal evidence after 7,270.11 s.
Independent exact-row/global-union verification and sealing completed at
8,005.72 s total (about 2 h 13 min). Post-seal peak RSS was 95,416,320 bytes,
the run root contained 776,428,838 bytes, and free disk remained
13,758,830,215,168 bytes.

The first external independent validator completed its remote process but its
SSH transport failed to deliver a terminal result. It is retained as
inconclusive, not PASS. After confirming that no child/scratch process
remained, the same read-only validation was replayed with keepalive:
`_validate_bundle` passed in 751.59 s and `_validate_verified_marker` passed.
It found no P0 or P1 issue.

Two non-blocking audit limitations are retained:

- exact rows are independently recomputed, while lower-row numeric bounds are
  not recomputed from scratch by the bundle verifier; their typed stops,
  field constraints, accounting, and no-approximation semantics are verified;
- ordinal 122's already observed 104,827-state lower bound also exceeds
  50,000, but `exceeded_limits` lists the dimension that actually triggered
  termination (`max_state_dp_cells`). This does not change the row's failed
  frozen-gate assessment.

Across all 144 rows, 139 pass the frozen per-row assessment and 5 fail
(three exact state-count failures plus the two state-DP stops).

## Scientific and operational conclusion

The exact symbolic/streaming prototype and replay contract pass their
engineering acceptance, and the census provides credible negative capacity
evidence. It does **not** establish a safe new production budget:

- `exact_capacity_complete=false`;
- `lower_bounds_present=true`;
- `usable_for_budget_decision=false`;
- `usable_for_b0_acceptance=false`;
- `budget_change_authorized=false`.

Therefore this audit recommends **no production budget or gate change** and no
fresh formal B0. The current frozen gate is demonstrably too small for at
least three exact records, while two other records already exceed
100,000,000 state-DP cells without completing. Simply raising the existing
cap would select an unsupported number and could move the failure to a later
near-neighbor or global-leakage stage.

The next separately authorized capacity step should be one of:

1. derive a more compressed exact closure/connectivity certificate for the
   two stopped records, retaining all-shortest-path semantics; or
2. predeclare and audit a higher **diagnostic-only** state-DP envelope with
   explicit wall, disk, and verification budgets, then replay both stopped
   records from fresh roots.

Until one path yields exact full-census evidence, B0 remains `SAFE_PAUSED`.
This is a capacity/engineering conclusion, not a final biological or model
performance claim.
