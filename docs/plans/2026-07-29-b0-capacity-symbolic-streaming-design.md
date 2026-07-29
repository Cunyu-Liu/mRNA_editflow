# B0 capacity diagnostic and exact symbolic/streaming prototype design

## Authorization and boundary

This design implements the user-approved minimum recovery step after formal
`B0_attempt_001_20260729T125546Z` failed with exact path-state complexity.
It does not resume B0.

The following are immutable:

- the active scientific contract and its SHA-256;
- the frozen D1 snapshot, candidate/label ID bijection, and exposure boundary;
- all-shortest primitive dynamic edit execution-order semantics;
- sequence identity as state identity;
- coordinate-equivalent transition collapse;
- exact intermediate-state leakage semantics;
- the production 50,000-state guard and every other current B0 gate;
- GSE246381 as historically exposed retrospective evidence only.

The diagnostic is structural and label-free. It cannot claim B0 accepted,
frozen, leakage-safe, scientifically positive, or ready for training.

## Architecture

The system has three isolated layers.

### 1. Exact capacity census

`scripts/data/diagnose_b0_path_capacity.py` selects structural records from the
frozen label-free candidate store. Selection is deterministic and recorded.
For each record it first computes exact minimum-alignment statistics, classifies
the endpoint relation, and records cheap capacity features. It then applies
the explicitly requested diagnostic mode:

- `certificate`: exact symbolic node/layer count for supported pure-indel
  records;
- `stream`: exact external-memory closure traversal;
- `legacy-parity`: bounded comparison with the current in-memory enumerator.

Unsupported or resource-stopped records receive typed terminal row statuses.
They are never silently omitted. Summary fields distinguish exact totals,
lower bounds, and uncomputed counts.

The first mandatory record is the frozen 95,217-state witness. If its record
identity, endpoints, expected counts, or digest drift, the run stops before a
broader census.

### 2. Exact symbolic/streaming engine

`data/utr_benchmark_v2/symbolic_path_states.py` is a non-production prototype.
It imports the existing exact neighbor oracle rather than duplicating
scientific semantics.

Each layer input is a sorted TSV of:

`sequence<TAB>path_count`

For every parent, the engine:

1. obtains all unique distance-reducing child sequences;
2. records primitive-action and DP-cell counts;
3. emits one path contribution per unique sequence edge;
4. spills bounded chunks sorted by child sequence;
5. k-way merges chunks, sums exact path counts, and writes the next layer.

The engine maintains exact counters and resource observations. It writes a
completed layer atomically and records its hash before proceeding. Incomplete
temporary files are not valid checkpoints.

The final state digest requires global lexicographic order, not layer order.
The engine externally merges the completed sorted layer files and applies the
same newline-delimited SHA-256 definition as the legacy closure.

### 3. Immutable diagnostic bundle

The CLI creates a unique run root such as:

`/mnt/cunyuliu/mrna_editflow_b0_capacity/B0_capacity_<UTC>_<commit-prefix>`

It refuses an existing root. The bundle contains:

- `command.json` and `replay.sh`;
- `diagnostic_manifest.json`, `input_manifest.json`, `code_manifest.json`, and
  `runtime_manifest.json`;
- `records/results.jsonl`, `capacity_summary.json`, and low-frequency system
  metrics;
- `logs/stdout.log` and `logs/stderr.log`;
- `artifact_checksums.json`, `status.json`, `bundle_seal.json`, and a non-empty
  `terminal.lock`, followed by a post-validation `VERIFIED` marker;
- per-record prototype subdirectories and exact layer hashes where applicable.

`runtime_manifest.json` distinguishes the lexical Python invocation path from
its resolved executable target and binds both. It also records and live-replays
the prefix, base prefix, import path, `jsonschema` and `rfc3339-validator`
identities, frozen valid/invalid `date-time` semantics, and isolated entrypoint
`--help` probe. `replay.sh` uses a minimal fixed environment, disables
user-site and bytecode writes, probes dependencies before execution, and
retains the lexical launcher. Runtime-critical dependencies under
`/home/cunyuliu` are forbidden.

Every `EXACT_COMPLETED` row explicitly references
`record_workspaces/<ordinal>-<record-id-hash>/state_universe.tsv`. The file is
strictly sorted, its row count equals `reachable_node_count`, and its canonical
newline SHA-256 equals `reachable_states_sha256`. A lower-bound row has a null
reference and no completed file at that deterministic path; partial layer
files remain failure evidence.

The detached seal binds the manifest, complete pre-seal checksum index,
terminal marker, status, and process result. The terminal lock binds that seal.
The bundle verifier live-recomputes D1 snapshot trust and structural selection,
then compares all selection counts, record order, ID digests, endpoint values,
structural hashes, and terminal-row identities. It validates every exact
row's deterministic state file and replays a bounded external merge of the
eligible endpoints plus all exact state files. Count, byte length, newline
digest, and file SHA-256 must match the claimed global artifact. Self-consistent
saved manifests alone cannot pass. `VERIFIED` binds those terminal artifacts
only after validation. Captured stdout/stderr are OS-level,
hash-bound evidence and may be non-empty; warnings are preserved rather than
silently erased. A failed run preserves provenance plus typed failure evidence.

Global-universe roles are explicit: `EXACT` requires a completed fully
accounted all-exact run; a successful partial merge is
`LOWER_BOUND_FROM_ENDPOINTS_AND_EXACT_COMPLETED_RECORDS`; an endpoint-only
artifact is legal only with bound global-merge resource-pause evidence.
Verification uses the configured `max_open_chunks` and temporary scratch
outside the immutable run root, and removes that scratch after comparison.
The same declared scratch-byte, free-disk, RSS, and wall-time safety envelope
fails verification closed if the independent replay cannot finish safely.

Before global replay, every claimed exact row is independently recomputed from
its live source/candidate endpoints in fresh temporary scratch. The verifier
compares the full alignment statistics, nodes, transitions, state paths,
primitive actions, DP cells, state artifact, and frozen-gate assessment.
All row replays and global replay share one wall-time budget. It then rebuilds
the complete capacity summary and requires dictionary equality, so claim
boundaries and distribution fields cannot be altered by re-sealing.

A census also embeds hashes for the exact witness manifest, `VERIFIED` marker,
detached seal, terminal lock, and process result that authorized it. The
parent diagnostic ID alone is not an authorization proof.
The child verifier opens and validates that actual parent bundle recursively,
validates its `VERIFIED` marker, requires a completed parentless witness for
the same code/data/config, and rehashes every embedded parent reference.

## Exactness invariants

- Every emitted edge reduces exact remaining edit distance by one.
- One parent has at most one edge to a given child sequence, even if several
  dynamic coordinates produce it.
- Path multiplicity is propagated over unique sequence edges.
- A sequence cannot appear in two distance layers.
- Integer path counts are unbounded.
- State and record ordering is deterministic.
- Candidate rows must match the frozen D1 field allowlist and pass the existing
  recursive label-path detector.
- The exact D1 snapshot is live-recomputed first. Its validator may hash the
  canonical label store as opaque bytes, but neither selection nor capacity
  code parses or accesses any label value. Recompute explicitly binds the
  authoritative external Git directory and worktree so a stale native `.git`
  cannot change the result.
- A result is `exact=true` only when all layers complete and all hashes and
  counters validate.
- A summary may call a value a lower bound only when its included exact rows
  are enumerated and its omitted rows/reasons are explicit.

## Failure and recovery semantics

Resource limits are prototype guards, not scientific gates. Crossing one emits
`FAILED_WITH_EVIDENCE` or a typed per-record `STOPPED_WITH_EVIDENCE`; it does
not raise production budgets, weaken the record set, sample paths, or retry
with a less exact method.

The external-memory engine invokes a resource callback within record expansion,
merge, and digest loops. RSS, free disk, wall time, and total run bytes are
checked at that finer cadence, while metrics are persisted only at the
low-frequency heartbeat interval. Every merge forces a callback on its final
tail, even below the ordinary progress interval, and a second forced check runs
before terminal evidence is built. The guard scope ends at that preterminal
check; immutable sealing overhead is recorded in the post-seal metric. Any
retained `.partial` evidence is checksum-indexed.

A checkpoint may be resumed only when contract, code, input, CLI, algorithm,
and every completed-layer hash match. Otherwise resume fails closed. The first
implementation may deliberately disable resume while still using atomic layer
files; it must state `resume_supported=false` rather than imply crash recovery.

## Verification matrix

| Scope | Required evidence |
|---|---|
| Neighbor oracle | exhaustive A/C/G/U endpoints length 1–3 |
| Full closure | exact state/edge/path/action/digest parity on bounded cases |
| Repeats | coordinate actions counted separately, sequence edge collapsed |
| Streaming | chunk-size and input-order invariance |
| Symbolic certificate | pure insert/delete node and layer parity |
| Frozen witness | all four frozen exact quantities reproduced |
| Resource stops | no completed exact result, typed evidence retained |
| Bundle | schema, live D1 selection/order/endpoints, independent full replay of every exact row, deterministic state files, bounded global-union replay, regenerated full summary, recursive parent authorization, cross-accounting, complete SHA-256 index including partial evidence, exact absolute replay/cwd, detached seal, VERIFIED |
| Production isolation | existing B0 files/guards unchanged |

## Decision point after credible numbers

Only after the witness and structural census are independently auditable will
we compare:

- exact total states/edges and their distribution;
- peak memory, disk, wall time, DP cells, and output write volume;
- repeated global work in the current five split plus five leakage runs;
- projected near-neighbor candidate and exact-verification load.

That evidence will support a later choice among retaining external-memory
execution, authorizing an explicit audited resource envelope, or pursuing a
more compressed exact connectivity proof. It will not itself authorize any
change.
