# UTR EditFlow V2 Run State Machine

## Normal path

```text
REGISTERED
→ PREFLIGHT_PASSED
→ GPU_VERIFIED
→ STARTED
→ RUNNING
→ TRAINING_FINISHED
→ EVALUATED
→ VERIFIED
→ FROZEN
```

Exit code zero is not `VERIFIED`. Evaluation artifacts, checksums, contract
gates and manual audit must also pass.

## Exceptional paths

```text
RUNNING → SAFE_PAUSED → RESUMED_<N>
RUNNING → FAILED_WITH_EVIDENCE → SUPERSEDED_BY_<NEW_RUN_ID>
REGISTERED → WAITING_FOR_GPU
```

The same run may resume only when contract, code, data, split, config, seed,
optimizer semantics and foundation checkpoint are unchanged. Any semantic
change creates a new run ID and records `parent_run_id`.

## Fail-closed CUDA gate

A formal neural run must produce `logs/cuda_health.json` from its actual
model/input/optimizer step. All eight booleans in the execution contract must
be true. A synthetic launcher probe is a prerequisite but not a substitute.
Missing or false health evidence makes the run `FAILED_WITH_EVIDENCE`.

## Monitoring cadence

- one health check between minute 3 and 5;
- automatic system heartbeat no more often than every 5 minutes;
- semantic inspection no more often than every 30 minutes;
- long runs may use 60-minute semantic checks;
- no continuous `tail -f` or anxiety-driven refresh;
- exceptions are exit, non-finite state, resource danger, checkpoint failure,
  contract/hash drift, or another registered stop event.

The monitor is intentionally a single-pass command. Scheduling is external and
must respect the frozen cadence.

## Failure preservation

Before any retry:

1. write status and failure bundle;
2. preserve last healthy and best checkpoint;
3. preserve stdout/stderr, metrics, system metrics and events;
4. record exit code and stop reason;
5. diagnose;
6. create a new run ID when semantics change.

Only the current run's exact PID may be stopped after confirmed stall. No
wildcard process termination is permitted.
