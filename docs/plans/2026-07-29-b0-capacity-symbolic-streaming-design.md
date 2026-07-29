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

`/mnt/cunyuliu/mrna_editflow_b0_capacity/B0_CAPACITY_DIAG_<UTC>_<nonce>`

It refuses an existing root. The bundle contains:

- `command.json` and `replay.sh`;
- `run_manifest.json`, `input_manifest.json`, `code_manifest.json`, and
  `runtime_manifest.json`;
- `records.jsonl`, `summary.json`, and `resource_usage.json`;
- `logs/stdout.log` and `logs/stderr.log`;
- `checksums.sha256`, `status.json`, and `terminal.lock`;
- per-record prototype subdirectories and exact layer hashes where applicable.

The terminal seal is created only after all committed files validate. A failed
run preserves the same provenance bundle plus typed failure evidence.

## Exactness invariants

- Every emitted edge reduces exact remaining edit distance by one.
- One parent has at most one edge to a given child sequence, even if several
  dynamic coordinates produce it.
- Path multiplicity is propagated over unique sequence edges.
- A sequence cannot appear in two distance layers.
- Integer path counts are unbounded.
- State and record ordering is deterministic.
- The input candidate store is read without final-label values.
- A result is `exact=true` only when all layers complete and all hashes and
  counters validate.
- A summary may call a value a lower bound only when its included exact rows
  are enumerated and its omitted rows/reasons are explicit.

## Failure and recovery semantics

Resource limits are prototype guards, not scientific gates. Crossing one emits
`FAILED_WITH_EVIDENCE` or a typed per-record `STOPPED_WITH_EVIDENCE`; it does
not raise production budgets, weaken the record set, sample paths, or retry
with a less exact method.

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
| Bundle | schema, SHA-256, row count, replay command, terminal seal |
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

