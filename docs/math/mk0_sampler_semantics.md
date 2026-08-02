# MK0 sampler semantics and numerical contract

## Two samplers, two roles

### `paper_first_order_parallel`

This is the original-method-compatible numerical reference: a fixed time grid and first-order event-probability approximation, with all proposed events based on the pre-step state. It is explicitly recorded as:

```yaml
semantics: fixed_grid_parallel_first_order_approximation
exact_gillespie: false
role: numerical_reference_only
```

Collision application order is frozen as descending-index deletion, ascending-index substitution, then ascending gap/token insertion. This ordering is an implementation convention. Joint-event legality is checked before applying a parallel set. Post-sampling projection, culling or repair is not paper-compatible and is not allowed under this sampler name.

Its hard validity and convergence are reported separately. It does not carry the strict-budget UTR primary-generation gate.

### `constrained_single_event_first_order`

This is the MK0-v1 hard-constrained primary. For a substep from `t` to `t+h`, freeze the rate field at the substep start and compute stably:

\[
p_{event}=-expm1(-h\Lambda_{all}(S_t)).
\]

Only when `Lambda_all > 0` compute `u(a|S_t)/Lambda_all`. One event at most is selected, and it is applied at endpoint `t+h`; the remainder of that substep is not simulated. The precise name is:

```text
endpoint single-event frozen-rate approximation
```

It is a first-order approximation to a time-inhomogeneous process, not exact event-time simulation.

## Primary substep algorithm

1. Validate ACTIVE state and recompute dynamic current encoding, hard masks and all rates.
2. Fail numerically on NaN, Inf, negative rate/probability or damaged state.
3. If `h*Lambda_all > 0.05`, subdivide until the threshold is met and record the subdivision count.
4. If instantaneous total hazard is zero, emit `NO_EVENT`, advance to the substep endpoint and recompute. Do not divide and do not terminate merely because of that zero.
5. Otherwise draw the event Bernoulli using `-expm1(-h*Lambda_all)`.
6. On no event, advance time and recompute.
7. On an event, normalize only legal positive-rate actions, draw one action, apply it at the endpoint, decrement budget only for INS/SUB/DEL, validate the new state, then recompute.
8. STOP enters HALTED. No further state or rate update may edit the sequence.
9. At horizon, use the explicit termination reason. Use zero-remaining-integrated-hazard termination only after a separate no-event path integral verifies it.

Illegal actions are never sampled and silently discarded. No silent rate/probability clip is permitted.

## Strict budget and collision invariants

The primary sampler executes at most one event per substep, so budget and length constraints are checked on the exact pre-event state. Reversal and cycle edits still spend budget. Hard validity requires 100% valid nonnumerically-failed trajectories and zero budget violations.

The parallel reference can combine individually legal events into an illegal joint update. Its joint kernel therefore checks budget, length, anchor and index conflicts on the full set before application; a rejected set is logged rather than repaired.

## Replay record

Every substep records seed, start/end time, effective `h`, total hazard, candidate-action hash, event probability, event/action uniform draws, selected action, adaptive subdivisions, state hashes and rate-recomputation marker. Replay must reproduce every intermediate state hash, not only the final sequence. Corrupt or missing randomness is a replay failure.

## Frozen convergence study

The step sizes are `1/32`, `1/64`, `1/128` and `1/256`, with `1/128` the primary. Each uses the same preregistered seed design and 512 trajectories. The audit reports:

- total variation of action/termination summaries between the finest pair (threshold 0.03);
- mean edit-count difference (threshold 0.05);
- termination-fraction difference (threshold 0.03); and
- error against a numerical integrated-hazard tiny-state reference, required not to worsen with refinement.

These are numerical E0 checks, not functional-quality comparisons. If they fail, the failure bundle is retained and a new run ID is used after correction; the tolerances are not relaxed post hoc.
