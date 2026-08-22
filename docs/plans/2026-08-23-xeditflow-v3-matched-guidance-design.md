# XEditFlow V3 matched-guidance control design

## Scope and frozen meaning

This design implements only the six method roles already frozen in the V3
method-repair protocol.  It does not add a new arm, search a new
hyperparameter, consume Development TEST, or expose a final Evaluation
outcome.  Every learned method uses the same selected XEditSetFlow checkpoint,
source cohort, `SUB + STOP` legal action graph, edit budgets, candidate cap,
decoder streams, and GPU cohort.  The selected guidance-screen
`kappa/temperature/beta_max` is reused for every final seed without reselection.

The three critic-coupling controls must be behaviorally distinct after action
normalization.  `first_order_guidance` uses the discrete first-order expansion
of the frozen critic reward around the source: each substitution receives its
source-anchored single-edit coefficient
`R(source + edit) - R(source)`, which is then reused at later states and thus
cannot represent edit interactions.  `simple_rate_guidance` evaluates
`R(next) - R(current)` at every visited state and captures observed one-step
interactions, but it remains short-sighted.  `full_soft_value_smc` uses the
already frozen learned scalar value-to-go difference and is the only
non-myopic guidance method.  This is preferable to treating `R(next)` and
`R(next)-R(current)` as separate controls because the latter two differ only by
a state-wise constant and induce the same normalized action probabilities.

`generate_then_rerank` samples trajectories with the unguided SetFlow and
applies the same three-member frozen critic only after terminal deduplication;
the critic cannot alter a transition.  `unguided_setflow` never calls the
critic.  `strongest_matched_baseline` remains the prospectively frozen
Development-only classical/search winner and is not reselected per seed.

## Runtime and accounting

The matched controls use the same full-legal SetFlow proposal and particle
mechanics as the complete method.  A proposed transition is drawn from the
base rate distribution and then receives the appropriate scalar-potential
importance ratio.  This avoids enumerating roughly `3 × sequence length`
Critic evaluations at every state, which would violate the 320-forward ceiling,
while retaining support over every legal action.  The first-order potential
loads source-anchored single-edit coefficients lazily for edits actually
proposed and caches them.  The simple-rate potential evaluates the exact
current and proposed-child Critic rewards in one batch.  STOP has zero
potential difference because it preserves the sequence; structural budget
exhaustion remains distinct from explicit STOP.

Each runtime record keeps base-flow, value, and all three critic-member forward
counts separately under `MatchedComputeRecordV2`; no method may exceed 320
forward-equivalents or 32 returned candidates.  Fixed decoder streams are
replayed, but replay diagnostic work is reported in wall time rather than
credited as search budget.  Terminal candidates are deduplicated before
ranking.  The reranker may change only terminal order, never the generated
support or trajectories.  Full soft-value SMC retains its ESS<16 stratified
resampling and scalar-potential importance weights.

The final runner must emit the same candidate and compute schemas for all
learned methods, followed by the same closed-neighborhood, open-support, critic
diagnostic, and independent-evaluator stages.  Measured outcomes are used only
inside the frozen closed benchmark.  Independent evaluator scores never enter
generation or value gradients.  A critic-self-score increase without measured
NDCG/regret and independent-evaluator improvement remains reward exploitation,
not PASS.

## Failure behavior and tests

The runtime fails on an incomplete legal action bundle, non-positive or
non-finite legal base rate, missing source-anchored coefficient, non-finite
reward, candidate/compute overflow, replay mismatch, or protected-outcome
provenance.  It does not create fallback candidates or replace undefined
closed metrics with zero.

Focused tests cover: source-anchored first-order coefficients; the separation
between first-order and interaction-aware one-step guidance; hard-mask-before-
normalization; deterministic trajectory replay; STOP versus structural budget
exhaustion; per-member critic accounting; terminal deduplication; reranking
without support mutation; and the 32-candidate/320-forward ceilings.  The
existing exact small-graph scalar-potential tests continue to cover the full
soft-value method.
