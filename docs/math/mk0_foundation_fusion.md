# MK0 foundation encoder fusion and leakage firewall

## Representation contract

MK0 freezes two sequence representations:

\[
H_{src}=E_\psi(x_{src}),\qquad H_t=E_\psi(x_{current}).
\]

The source representation may be cached. The dynamic current representation is fully recomputed after every edit in the reference implementation. Incremental updating is disabled until a separate layer-by-layer equivalence test matches full re-encoding under frozen tolerances.

Foundation identity, revision, local snapshot hash, license, pretraining-exposure audit and length behavior are inherited from the passed FM0 artifact and rebound in `mk0_freeze_manifest.json`. A model name without those bindings is insufficient provenance.

## Required rate-field inputs

The trainable fusion/rate head receives real, nonconstant representations of:

- current tokens and insertion gaps;
- source-aligned features gathered only through `M_run`;
- source/current differences available identically at inference;
- 5′/3′ region adapter;
- assay, cell/context and endpoint;
- target direction/interval/quantile condition;
- external-time embedding;
- remaining budget; and
- inference-visible edit-history summary.

Both `H_src` and `H_t` must affect rates. A source cache alone with a stale or absent current encoding fails the contract.

## Prohibited inputs

The following never enter the encoder fusion, adapter, operation/token head, STOP head or sampler:

```text
Z_aux
x_target
target alignment
remaining target edits
features derived from any of the above
final evaluator scores, gradients or queries
```

Changing target/alignment while holding the full inference-visible state fixed may alter target weights used by the training loss; it must not alter the model rate vector. The paired leakage artifact records complete input hashes, output hashes, maximum discrepancy and tolerance.

## Mapping-aware gather

Current token positions gather a source feature only when their `M_run` origin is a surviving source token. Inserted tokens use an explicit inserted-origin representation plus neighboring logical-gap features; they are not assigned a fabricated target alignment. Deleted source tokens remain represented in the source cache but have null current coordinates. Protected flags derive from runtime source mapping only.

Gap representations are derived from the adjacent current states and stable gap IDs. Every indel rebuilds the gather map before the next rate evaluation.

## Optimization order and modes

The allowed progression is:

```text
frozen foundation -> adapter/LoRA -> partial unfreeze
```

MK0 validates the frozen-foundation case. A from-scratch small network is permitted only as a clearly labelled structural control. Paper-reference mode must not substitute placeholder foundation tensors; `paper-mode placeholder foundation = 0` is audited across all forward calls.

## GPU-only neural acceptance

All neural forward/backward and neural smoke validation run on CUDA with CPU fallback disabled. CPU is reserved for symbolic, exhaustive and non-neural tests. The GPU artifact records device index/UUID, framework/CUDA versions, input/parameter/output devices, peak allocated memory, AMP/TF32 settings and absence of fallback.

The tiny smoke constructs legal, nonzero oracle-rate cases that force-cover `INS`, `SUB`, `DEL` and `STOP`. For each type it runs foundation forward, dynamic fusion/rate forward, a finite loss, backward and a parameter update. Coverage is forced deliberately; an untrained random model is not expected to sample all four naturally.

The same smoke includes:

- fixed source with changed current sequence, requiring a changed dynamic current encoding/rate vector;
- identical inference state with replaced target/alignment, requiring unchanged rates;
- current full re-encoding after every forced edit; and
- no critic and no final-evaluator interface.

CUDA unavailability, a CPU neural tensor, nonzero CPU-fallback count or a silent device downgrade stops the run and preserves a failure bundle.

## Evidence boundary

Passing fusion smoke proves interface use, leakage isolation and GPU execution at E0. It does not prove that the foundation improves generation, that pretraining transfers, or that generated sequences improve function.
