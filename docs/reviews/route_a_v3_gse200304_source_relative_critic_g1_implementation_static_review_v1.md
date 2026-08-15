# GSE200304 source-relative critic G1 implementation — static review v1

Verdict: `PASS_IMPLEMENTATION_BASELINE_EVALUATOR_ISOLATION_REVIEW_NOT_ACTIVE_NOT_RUN`

The implementation candidate contains a real full-length masked shared encoder,
antisymmetric forward/reverse pair construction, positive uncertainty scale,
fixed private row/split schemas, single-fit execution contract and fail-closed
gate bundle. The legacy fixed 100-position truncation and best-checkpoint behavior
are not reused.

The SS4 review gap is closed in this successor implementation.  All four frozen
TRAIN-only comparators are executable: source-group-equal global mean, directed
edit-type mean, GC/length ridge with alpha 1, and a deterministic signed
feature-hashed candidate-minus-source 15-mer count ridge with alpha 10.  The
4096-dimensional feature map replaces an otherwise impractical full 15-mer
design matrix and is frozen before the terminal run.  Baselines do not consume
CALIBRATION or TEST outcomes during fitting.

The primary critic metric and MAE are computed after reducing records to one
mean per biological source group.  Mean calibration and conformal scale are fit
on CALIBRATION only; coverage-risk thresholds are also frozen from CALIBRATION
before one terminal TEST evaluation.  Constant-rank critic predictions stop the
run rather than becoming a null metric.  No evaluator value is returned to the
optimizer, checkpoint selection, guide, threshold search or model selection.

The current config is inactive and all five future activation requirements are
unsatisfied. The operational entry checks that state before touching the asset
directory, importing PyTorch, constructing a model, probing CUDA or creating an
output.  A later config-only authority must bind the reviewed implementation
commit, exact SS3 rows, exact SS4 assignments, one Python/PyTorch/CUDA runtime,
one physical GPU and one output directory.  The production entry rechecks those
bindings before the first private-input read or CUDA/model construction.

The active-only path contains the fixed private schema reader, dynamic full-length
batching, one AdamW fit, group-weighted heteroscedastic loss, four frozen
baselines, calibration-only transformation/quantile/abstention thresholds, one
terminal test evaluation, one terminal checkpoint, private predictions, and an
aggregate calibration/LCB manifest candidate.  The LCB candidate explicitly
does not authorize A6.  The path remains unreachable while activation bindings
are null; materialized conformance and a runtime smoke must pass before the
separate exactly-one G1 authority is signed.

This review is sufficient only for metadata P0 implementation binding. It is not
materialization authority, G1 authority, critic PASS, training evidence, A6
learned-base/value authority, A7 or a scientific claim.
