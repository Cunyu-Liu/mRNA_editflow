# Route 2 Critic V2 Post-confirmation Stage Design

## Boundary

This entrypoint is frozen before the Critic V2 control or three-seed outcomes
exist. It is not launched during implementation. A separate 900-second watcher
may wait for the exact three-seed adjudication, but it must exit without creating
stage artifacts on NO-GO.

The stage is Development-only. It may execute the one prospectively authorized
Development TEST after a real three-seed PASS, but that TEST is report-only and
cannot alter structure, loss, seed, epoch, threshold or policy. Final Evaluation
remains closed throughout.

## Execution model

Before any runtime, log or output write, the runner uses the production frozen-
TEST builder to validate the control and three-seed adjudications and the selected
seed-20260823 confirmation config. It also validates the complete frozen V2
protocol set and refuses every already-started downstream target.

The only order is:

1. prepare and execute the single frozen Development TEST;
2. prepare and execute the all-126,165 Development refit without branching on
   TEST metric values;
3. prepare the 21 Critic V2 and 21 exactly matched baseline LOSO configs;
4. run the paired six-GPU LOSO stage and three fixed-seed aggregations;
5. build and adjudicate Critic/Flow dual readiness;
6. invoke the already frozen Development generation stage only when readiness
   returns `guided_unlocked=true`.

Single-run training dynamically selects the most-free physical GPU0-5 above
4096 MiB and waits at 900-second intervals using free memory only. The LOSO and
generation stages retain their own already frozen GPU0-5 scheduling policies.
Any child failure preserves evidence and prevents later stages.

The historical V1 postselection scheduler is retired with an immediate refusal;
it is not kept as a compatibility route.

## Verification

Focused tests cover deterministic GPU selection, production three-seed gate
validation before writes, all downstream targets, exact stage order, protected
outcome flags, the 900-second conditional watcher and immediate V1 retirement.
No GPU query, training subprocess or protected-outcome read runs in tests.
