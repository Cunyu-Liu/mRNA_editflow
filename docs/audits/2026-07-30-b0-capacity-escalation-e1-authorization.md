# B0 exact capacity diagnostic — E1 resource authorization

## Scope and claim boundary

On 2026-07-30, the project owner explicitly authorized completion of the
pending exact capacity process without treating the previously conservative
diagnostic budget as a cost ceiling.  This record implements that instruction
as the finite, reproducible **E1 diagnostic-only** resource envelope in
`configs/b0_capacity_diagnostic_escalation_e1_v1.json`.

E1 does not start, resume, accept, or claim a formal B0 attempt.  It does not
change the scientific question, selected 144-record structural census, input
data, any split, leakage rule, model, label-access boundary, frozen B0 gate,
or evaluation criterion.  It cannot turn a smoke result, a training result, or
a capacity measurement into a biological conclusion.

The unchanged production limits remain:

| Frozen B0 gate | Value |
| --- | ---: |
| reachable states | 50,000 |
| state-DP cells | 50,000,000 |
| DAG cells | 1,000,000 |
| neighbor expansions | 5,000,000 |

The previous exact census was correctly retained as a negative capacity
result: 142 exact rows, two `LOWER_BOUND_STOPPED` rows, and no unaccounted
record.  The two incomplete rows crossed the old diagnostic-only
100,000,000-state-DP safety stop.  This proves a lower bound, not an upper
bound.  E1 is therefore a fresh exact measurement, not a reinterpretation of
the old result.

## E1 resource envelope

| Diagnostic safety field | E1 value | Reason |
| --- | ---: | --- |
| maximum state-DP cells per record | 10,000,000,000 | permits an exact closure far beyond the measured 100,000,001-cell lower bounds |
| maximum DAG cells per record | 250,000,000 | prevents an unbounded DAG materialization while avoiding the old 1,000,000-cell diagnostic ceiling |
| maximum reachable states per record | 10,000,000 | keeps an independently auditable state-set ceiling above the former 150,000 limit |
| maximum neighbor expansions per record | 10,000,000,000 | retains an explicit exact-work guard |
| maximum spill per record | 2,000,000,000,000 bytes | allows external-memory exact closure without using `/home` |
| maximum total spill | 4,000,000,000,000 bytes | bounds the fresh witness plus census run |
| required free `/mnt` space | 8,000,000,000,000 bytes | reserves substantial shared storage for other work |
| maximum resident memory | 68,719,476,736 bytes (64 GiB) | unchanged conservative cap on a shared host |
| maximum wall time | 259,200 seconds (72 h) | finite upper bound for a full fresh census |
| heartbeat / external review cadence | no more frequent than every 300 s | matches the contract's low-frequency monitoring rule |

The 2026-07-30 read-only preflight observed 96 logical CPUs, approximately
303 GiB available RAM, and approximately 12.5 TiB free under `/mnt`.  E1 runs
one CPU-only, non-neural process at reduced scheduler priority.  It neither
requests CUDA nor touches the unrelated GPU jobs, and all run roots, logs,
temporary spill, runtime, and seals remain under `/mnt/cunyuliu`.

These are diagnostic safety ceilings, not proposed production budgets.  They
are intentionally recorded as finite values: a statement that cost is not the
decision constraint never authorizes silent unlimited resource use or removal
of a safety stop.

## Required sequence

1. Validate the new JSON config and all unchanged frozen limits.
2. Commit and push this config and authorization record before execution.
3. Run a fresh E1 frozen witness in a fresh output root using the sealed
   runtime and exact `-I -B` launcher.
4. Independently validate the witness bundle and `VERIFIED` marker.
5. Only then run the full 144-record E1 census from another fresh root.  No
   prior lower-bound root may be resumed or overwritten.
6. Independently validate the terminal bundle, manifest, checksums, and
   accounting.  Exact completion requires 144 accounted records, zero lower
   bounds, zero failed records, and no approximation.
7. Write and commit the E1 result audit.  A formal B0 decision remains a
   separate contract gate after exact evidence is available.

## Failure and continuation semantics

If E1 reaches a declared resource stop, integrity failure, invariant failure,
or host-capacity stop, it remains an auditable safe pause: partial evidence is
preserved and no result is relabeled as exact.  The owner has authorized
continued exact-capacity work.  Any E2 continuation must nevertheless create
a new versioned diagnostic config, preserve the E1 root, re-run a fresh
witness, and re-run the entire census; it may modify only diagnostic safety
limits after a new preflight.  It must not modify a frozen B0 gate, data,
selection, algorithmic semantics, or acceptance criterion.
