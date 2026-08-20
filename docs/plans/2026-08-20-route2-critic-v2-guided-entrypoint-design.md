# Route 2 Critic V2 Guided Entrypoint Design

## Boundary

This change freezes the executable Development guided-generation entrypoint
before Critic V2 readiness exists. It does not construct readiness evidence,
run generation, inspect TEST values or access Evaluation outcomes.

## Problem

The existing guided runner and config bind the historical V1 readiness input,
adjudication and refit checkpoint. The V2 readiness path deliberately uses a new
schema and artifact names. Leaving the old binding in place would either reject a
valid V2 PASS or allow a caller to present the obsolete V1 gate. The downstream
matched-search runner has the same obsolete readiness/refit binding and would
reject the V2 guided output even after a successful guided run.
The generation-comparison config also points to the historical guided method,
candidate directories and guiding checkpoint, so it would sever the chain after
matched search.

## Decision

Use a hard V2 cutover rather than a compatibility branch. Both existing runners
must reject their historical configs before reading any readiness artifact. New
V2 configs bind the V2 readiness input/adjudication, final all-Development refit
checkpoint, V2 guided summary/compute artifacts and V2-specific output
directories. The historical configs remain retired records and are not
executable through the runners. The Development comparison entrypoint likewise
accepts only the V2 guided method and exact V2 candidate/checkpoint paths.

The runner recomputes the minimal boundary needed at launch: exact V2 input and
adjudication schemas; critic/Flow ready statuses; guided-unlocked, not-executed
and Evaluation-closed fields; online encoder status; zero Evaluation use in the
reward policy; and exact critic/Flow checkpoint identity. It then retains the
existing frozen mean-potential, legal `SUB + STOP`, fixed-seed replay and compute
accounting implementation. Matched search retains the exact per-source total
forward-equivalent budget emitted by guided generation and remains candidate
generation only; it does not select a strongest method.
The final comparison retains the frozen independent-evaluator and paired-
bootstrap rules and continues to report Development evidence only.

## Verification

Focused tests require the exact V2 configs and packet, reject all retired V1
schemas and checkpoint drift, bind the V2 guided summary/compute consumer, and
the V2 candidate comparison, and retain all existing guidance-rate, replay,
backend, output-contract, comparison and compute-accounting tests. No real
artifact is read.
