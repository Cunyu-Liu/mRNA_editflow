# ADR 0001: Isolated exact symbolic/streaming B0 capacity diagnostic

- **Status:** Accepted for a non-formal diagnostic prototype
- **Date:** 2026-07-29
- **Decision owner:** user authorization in the active Codex task
- **Scientific contract:** `docs/contracts/mrna_latest_build_contract_v2.md`
- **Contract SHA-256:** `c3dc5875868d847b8519fee40b14c43b65e4c5948dc5c3b98101ca61a5671dd5`

## Context

Formal `B0_attempt_001_20260729T125546Z` stopped with
`STOP_RULE_B0_PATH_STATE_COMPLEXITY`. The first exact witness has 95,217
sequence-identity states, exceeding the unchanged production guard of 50,000.
The previous broader capacity census was not packaged with a persisted command
and independent log, so its aggregate numbers cannot authorize a budget
change or a new formal B0 attempt.

The contract requires the union of all sequence states on every shortest
primitive dynamic edit path. Coordinate-distinct actions that produce the same
next sequence are one sequence edge. Path counts are over those unique
sequence edges. A single traceback, sampled paths, truncated states, or
coordinate-combination counts are not equivalent.

The user authorized only:

1. a replayable capacity diagnostic;
2. an exact symbolic/streaming prototype;
3. no scientific-criterion, gate, or production-budget change; and
4. no new formal B0 attempt until credible measurements are reviewed.

## Decision

Use a hybrid exact design, isolated from the production B0 entrypoints.

### Exact neighbor oracle

The prototype reuses the reviewed distance-reducing neighbor semantics in
`path_states.py`. A state is expanded only to unique sequences exactly one
primitive edit closer to the fixed candidate. Primitive actions are counted
separately before sequence-identity collapse.

At each state:

- pure insertion/deletion cases use the subsequence certificate;
- mixed cases use exact forward/backward banded Levenshtein costs;
- classification is dynamic, so a mixed record may later enter a pure-indel
  subproblem;
- a limit breach returns no partial exact result.

### External-memory layer traversal

The prototype processes one shortest-path distance layer at a time. It writes
parent contributions to bounded sorted chunks, merges them by full child
sequence, and sums Python arbitrary-precision path counts. The merged child
layer is the only input to the next layer.

Because every accepted edge reduces exact remaining distance by one, layers
are disjoint. Exact totals are:

- nodes: sum of unique sequence counts across all layers;
- edges: sum of unique `(parent_sequence, child_sequence)` edges;
- paths: sum of penultimate-layer path counts reaching the candidate;
- primitive actions and DP cells: the same scopes as the current oracle;
- state digest: SHA-256 over the globally lexicographically sorted complete
  state universe with the existing newline convention.

The prototype never silently substitutes an approximation. Disk, byte,
record, action, DP-cell, or wall-time limits produce typed stop evidence.

### Symbolic pure-indel certificate

For a pure deletion closure, the state language is exactly:

`{x | candidate is a subsequence of x and x is a subsequence of source}`.

For pure insertion, source and candidate exchange the short/long roles. A
deterministic subsequence automaton can count unique accepted sequence states
by layer before materialization. The first prototype may use this certificate
for exact preflight counts while using external-memory traversal for edges,
paths, and the sequence digest. It must not infer sequence-edge or path counts
from coordinate combinations.

### Replayable diagnostic bundle

Each run has a new, non-overwritable root under `/mnt/cunyuliu`, outside the
formal D1/B0 attempt tree. It binds:

- contract, clean code commit, frozen D1 snapshot, label-free candidate store,
  and input record-ID hashes;
- CLI arguments, runtime identity, environment, resource limits, and process
  return code;
- one row per selected record, with exact results or a typed uncomputed/stop
  reason;
- stdout, stderr, resource observations, checksums, status, terminal seal, and
  a replay command.

Every manifest states:

`formal=false`, `acceptance_evidence=false`, `b0_accepted=false`,
`b0_frozen=false`, `scientific_conclusion=false`, `gate_changed=false`, and
`new_formal_attempt=false`.

The diagnostic reads only the frozen label-free candidate store for structural
capacity. It must reproduce the 95,217-state witness before any broader
census. Exact totals and lower bounds are separate fields; missing mixed
closures can never be presented as an exact global total.

## Alternatives considered

### Raise the in-memory production budgets

Rejected for this authorization. It changes operational policy without a
credible replayable capacity estimate, repeats global work across split and
leakage processes, and may merely move the stop to near-neighbor materialization.

### SQLite-backed graph traversal

Viable but not selected for the first prototype. It reduces bespoke sorting
code but introduces transaction, collation, index, checkpoint, and database
version semantics into the proof surface.

### Fully contracted record automata

Potentially avoids sequence materialization, especially for global
near-neighbor connectivity. It is deferred because exact language-to-language
distance joins and compatibility with current sequence-level binding digests
require a larger proof and schema migration.

## Consequences

- Production `minimum_alignment_state_closure`,
  `global_near_neighbor_clusters`, formal drivers, and all existing default
  guards remain unchanged.
- Capacity remains output-sensitive; the prototype may stop for disk or time
  while preserving evidence.
- A successful witness replay validates the prototype’s exactness on that
  record, not B0 acceptance.
- A full structural census can inform a later budget decision but cannot
  independently establish leakage safety or scientific validity.
- Any parity, digest, count, hash-binding, resume, or completeness mismatch
  fails closed and blocks a fresh formal B0.

## Prototype acceptance

1. Exhaustive short-sequence parity covers substitution, insertion, deletion,
   mixed edits, repeated bases, and coordinate-equivalent action collapse.
2. Streaming output is invariant to chunk size and input order.
3. The frozen witness reproduces exactly 95,217 nodes, 751,771 edges,
   3,934,510,691,993 state paths, and state digest
   `900076096ad75979a1b592b6d14fd7647dfe54c39b4cee80a053937de9411332`.
4. Limit and corruption tests produce terminal failure evidence and no
   partial result advertised as exact.
5. The diagnostic bundle is replayable from a clean commit and passes schema,
   checksum, input-hash, row-count, and terminal-seal validation.

