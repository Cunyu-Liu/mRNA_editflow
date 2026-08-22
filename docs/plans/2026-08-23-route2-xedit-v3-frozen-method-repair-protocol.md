# Route 2 XEditCritic V3 + XEditSetFlow V3 frozen method-repair protocol

Status: `FROZEN_PROSPECTIVE_BEFORE_XEDIT_V3_TRAINING_OR_VALIDATION_OUTCOME_READ`

This protocol records the user-approved method repair after terminal Critic V2
and Base Flow V2.  The main Route A V3.3.2 contract remains the scientific
authority.  This document only adds stricter prospective performance gates and
new model definitions; it does not rewrite any terminal result.

## Outcome boundary

All new V3 training consumes the TRAIN/VALIDATION projection defined by
`DevelopmentProjectionV3`.  The projection builder resolves the frozen split
from the manifest before decoding a complete canonical row.  Development TEST
rows are identified but otherwise unparsed.  A general TEST projection is not
implemented; TEST remains available only to a future one-shot adjudicator after
the exact three-seed critic gate passes.  New final Evaluation outcomes remain
closed until predictor, generator, baselines, metrics and adaptation policy are
frozen.

The historical loader parsed canonical rows before split filtering.  Existing
metrics remain historical facts, and there is no evidence that TEST was used in
loss, checkpoint selection or reported metrics.  Nevertheless, those runs do
not satisfy the strongest claim that protected outcome fields were never
parsed.  New V3 evidence must use the projection boundary.

## Frozen model sequence

1. Build the outcome-isolated TRAIN/VALIDATION projection and endpoint
   descriptors.
2. Build edit-site token features without reading TEST outcomes.
3. Run the four-arm Critic V3 screen with seed `20260830`; only C2/C3 are
   selectable.
4. In parallel, run the four-arm SetFlow screen with seed `20260903`; F0 is a
   read-only terminal reference and only F2/F3 are selectable.
5. Run exactly the declared three confirmation seeds for each model only after
   its screen passes.
6. A critic three-seed PASS authorizes one frozen TEST, followed by
   all-Development refit and 7-study × 3-seed TEST-preserving LOSO.
7. Formal soft-value guidance remains blocked until both critic readiness and
   upgraded flow readiness are established.

The machine-readable authority for seeds, architectures, thresholds, selection
ties, guidance grid, SMC budget and stop conditions is
`configs/route_a_v3_route2_xedit_v3_method_repair_protocol_v1.json`.

## Publication standard

The historical `0.153287` task-macro Spearman is a weak internal reference, not
the new success criterion.  Critic confirmation requires every seed at least
`0.25`, the three-seed median at least `0.30`, every matched-baseline margin at
least `0.07`, the median margin at least `0.10`, at least 8/9 positive tasks,
at least 6/9 matched-baseline task wins, standardized MAE no greater than
`1.70`, and a positive paired-bootstrap lower confidence bound.

SetFlow screen requires at least 10% common set-NLL improvement over the frozen
F0 replay, source-macro measured-candidate recovery at least `0.25`, top-k
recovery at least `0.15`, unique-candidate rate at least `0.90`, and exact G0
legality/budget/replay/numerical correctness.

No threshold may be lowered and no seed, arm, guidance-grid point or baseline
may be added after outcomes are observed.  A failed frozen stage is terminal;
another method requires a new user discussion and prospective freeze.
