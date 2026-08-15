# GSE200304 source-relative critic G1 implementation — static review v1

Verdict: `PASS_STATIC_IMPLEMENTATION_BINDING_ONLY_NOT_ACTIVE_NOT_RUN`

The implementation candidate contains a real full-length masked shared encoder,
antisymmetric forward/reverse pair construction, positive uncertainty scale,
fixed private row/split schemas, single-fit execution contract and fail-closed
gate bundle. The legacy fixed 100-position truncation and best-checkpoint behavior
are not reused.

The current config is inactive and all five future activation requirements are
unsatisfied. The operational entry checks that state before touching the asset
directory, importing PyTorch, constructing a model, probing CUDA or creating an
output. The static review therefore validates implementation presence and the
zero-I/O barrier only; it does not run a model or claim runtime correctness.

The active-only path contains the fixed private schema reader, dynamic full-length
batching, one AdamW fit, group-equalized heteroscedastic loss, calibration-only
quantile, terminal test metrics, terminal checkpoint and private predictions.
It is unreachable while the checked-in activation bindings are null. A later SS4
review must exercise that path on synthetic and materialized conformance inputs
before the separate G1 authority can make it active.

This review is sufficient only for metadata P0 implementation binding. It is not
materialization authority, G1 authority, critic PASS, training evidence, A6
learned-base/value authority, A7 or a scientific claim.
