# Route 2 Critic V2 LOSO Stage Runner Design

## Boundary

This runner is implemented before its upstream gates pass and is not launched
now. It consumes only already-prepared V2 LOSO configs. It never prepares or
opens Development TEST, never accesses Evaluation and never authorizes guidance.

## Execution model

The frozen schedule assigns each `(study, seed)` pair to one physical GPU in
GPU0-5. One worker per GPU processes its assigned folds sequentially. Within a
fold, the primary Critic V2 run completes before its exact matched baseline on
the same GPU. A worker waits at 900-second intervals until its assigned GPU has
at least 4096 MiB free; GPU utilization is recorded nowhere and is not a gate.

The runner validates all 21+21 configs and pair bindings before creating logs or
starting training. It refuses any existing run, log, aggregation-input or result
root, so a terminal or partial cohort is not silently rerun. If any worker fails,
the evidence is preserved and aggregation does not run.

Only after all 42 runs succeed does the runner call the V2 aggregation-input
builder and then the shared aggregator for the three fixed seeds. The runner
stops after producing Development LOSO aggregates; readiness remains a separate
gate.

## Verification

Focused tests exercise exact six-GPU job planning, primary-before-baseline order,
V2 config filenames, existing-artifact refusal and source-level ordering of
training before aggregation. No subprocess or GPU is used in tests.
