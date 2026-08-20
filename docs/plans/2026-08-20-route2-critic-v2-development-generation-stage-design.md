# Route 2 Critic V2 Development Generation Stage Design

## Boundary

This entrypoint is implemented before Critic V2 readiness exists and is not
launched now. It operates only on Development candidate generation and frozen
Development comparison evidence. It cannot open final Evaluation or add
generated candidates to canonical records.

## Execution model

An explicit invocation first reads the V2 readiness input and adjudication and
passes them through the production guided validator. If dual readiness is not
unlocked, the stage stops before creating runtime configs or output directories.

The stage queries physical GPU0-5, selects the card with the most free memory
among those with at least 4096 MiB free, and writes three one-time runtime configs
that differ from the prospectively frozen templates only in `device` and
`physical_gpu_index`. GPU utilization is not a gate. Before each child stage it
waits at 900-second intervals for the selected card to regain the minimum free
memory.

The only order is guided XEditFlow, six-method matched search, then frozen
independent-evaluator Development comparison. A child failure preserves runtime
and log evidence and prevents later stages. The runner does not perform final
Evaluation or interpret Development comparison as biological success.

## Verification

Focused tests cover deterministic GPU selection, exact V2 runtime config
bindings, one-write behavior and guided-before-matched-before-comparison source
order. No GPU query or child process runs in tests.
