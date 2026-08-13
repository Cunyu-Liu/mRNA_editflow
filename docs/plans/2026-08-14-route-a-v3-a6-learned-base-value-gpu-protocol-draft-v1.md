# Route A V3 A6 learned base/value GPU protocol — draft v1

## Status and decision boundary

| Field | Frozen value |
|---|---|
| Protocol ID | `ROUTE_A_V3_A6_LEARNED_BASE_VALUE_GPU_PROTOCOL_DRAFT_V1` |
| Document status | `DRAFT_FOR_REVIEW_NOT_ACTIVE_PROTOCOL` |
| Authority status | `NON_AUTHORITATIVE` |
| Implementation scope | protocol schema, static validator, and focused tests only |
| Runtime status | `NOT_RUN` |
| Evidence status | `NOT_RUN` |
| Scientific claim | `NOT_ESTABLISHED` |
| Formal `FLOW_BASE_LEGAL_CTMC` task | remains `NOT_RUN` |
| A6 phase | remains `IN_PROGRESS`; this draft does not assert PASS |
| L3 claim | remains `NOT_ESTABLISHED` |
| A7 | remains `NOT_RUN` and locked |
| Training / parameter updates | not allowed, not authorized, not run |
| GPU work | not allowed, not authorized, not run |
| Model or checkpoint selection | not allowed, not run |
| Private / sealed access | forbidden; reads remain zero |
| A1 counts / qualification / canonical state | unchanged |

The owner authorized design, semantic freezing, and implementation of a
non-executing protocol candidate. The owner explicitly did **not** authorize a
training run, an optimizer step, a CUDA process, model selection, A7 work,
private or sealed access, qualification, study credit, or canonical-state
changes. This draft therefore contains no trainer, no model checkpoint, no run
ID, no prediction, and no performance result.

The machine-readable companion is
`configs/route_a_v3_a6_learned_base_value_gpu_protocol_draft_v1.json`. A static
validator may read that file and report whether the draft is internally closed.
The validator has no model or data dependency and cannot activate or execute the
protocol.

## 1. Purpose

The protocol answers a prospective engineering and scientific-method question:

> If a later owner decision activates a formal A6 learned run, what exact
> source-anchored legal CTMC interface, ordinary-public data role, architecture,
> objective, independent exact reference, compute policy, and fail-closed gate
> will govern the one allowed base/value attempt?

It does not answer whether current data are qualified, whether the learned
network will pass, or whether learned guidance improves measured outcomes. A7,
not this draft, owns measured matched-compute guidance superiority.

## 2. Preconditions for any future activation

Every item below must exist under a later active authority before the first
parameter update. Missing any one item means
`STOP_BEFORE_DATA_MODEL_CUDA_AND_OUTPUT_IO`:

1. a new explicit owner authorization for execution and parameter updates;
2. registration of the reviewed protocol as active authority;
3. an A2-frozen, outcome-blind source-group/near-duplicate component split;
4. qualified ordinary-public development records with approved rights roles;
5. a frozen calibrated lower-confidence-bound terminal-score manifest;
6. a split, exposure, and rights manifest;
7. one explicitly assigned CUDA GPU UUID and an unused output run ID.

This draft cannot satisfy those preconditions by existing in Git. Static
validation, focused tests, a commit, or a hash do not constitute execution
authority.

## 3. Formal production interface

### 3.1 Extended state

The only supported primary state is

```text
(
  source_sequence,
  current_sequence,
  source_relative_edit_set,
  remaining_budget,
  assay_context,
  algorithmic_time,
  terminal_cause
)
```

The source and assay/context are immutable within a trajectory. The current
sequence must equal the source after applying exactly the recorded source-to-alt
edits. An edited position cannot be edited again or reverted to source. Each edit
decrements remaining budget once. Primary budgets are exactly 1, 3, and 5.

Algorithmic time is CTMC generation/control time, not physical mRNA time. This
protocol is time-homogeneous: neither base rates nor terminal tilt depend on
continuous time. General time-inhomogeneous exactness remains `NOT_RUN`.

### 3.2 Actions and terminal causes

The only actions are a source-base-to-alt-base edit at an unedited legal
position, or an explicit positive-rate `STOP`. Hard legality is evaluated before
any network or rate evaluation. A candidate with `k` source-relative edits is
assigned to the smallest primary budget greater than or equal to `k`: one edit
uses budget 1, two or three use budget 3, and four or five use budget 5.

The terminal causes remain distinct:

- `EXPLICIT_STOP`;
- `BUDGET_EXHAUSTED`;
- `NO_LEGAL_ACTION`;
- `NUMERICAL_FAILURE`.

Precedence is numerical failure, budget exhausted, no legal edit, then the
ordinary transient choice between legal edits and explicit STOP. Numerical
failure fails the run; it is not assigned stochastic terminal mass.

Event count means net source-relative edit count. The explicit STOP jump is not
an edit event.

### 3.3 Aliases, support, and rates

Raw actions mapping to the same **full next extended state** are aggregated before
normalization or model scoring. States with different terminal cause, context,
or time semantics are not aliases. Canonical transition provenance is retained
for audit, but alias multiplicity cannot create extra model degrees of freedom.

Every nonprohibited canonical transition receives a positive raw support floor
of `1e-8`. The base network emits one scalar logit for each canonical next state.
The total base exit hazard is fixed to 1.0:

\[
q_p(s,s')=
\frac{10^{-8}+\operatorname{softplus}(\ell_\theta(s,s'))}
{\sum_z[10^{-8}+\operatorname{softplus}(\ell_\theta(s,z))]}.
\]

Fixing the total hazard avoids pretending endpoint-only data identify a physical
rate scale. The scale is algorithmic and may not be interpreted biologically.

The learned value network emits a positive harmonic estimate
`h=1e-6+softplus(raw_value)` and `V=log(h)`. Guided off-diagonal rates are

\[
q_V(s,s')=q_p(s,s')\exp\{V(s')-V(s)\},
\]

and the diagonal is reconstructed as minus their sum. A free independent
action-ratio head is forbidden.

## 4. Ordinary-public data, split, exposure, and rights

### 4.1 Allowed role

Only records already qualified as ordinary-public and assigned by the future
A2 authority to the frozen development role are eligible. The allowed fields are
source and candidate sequence, the source-relative edit set, study and biological
source-group identity, assay, observable context, endpoint identity, calibrated
lower confidence bound, frozen split component, and rights/exposure roles.

Outer-test, confirmatory, sealed, private-canonical, rejected, excluded, and
unqualified-study records are forbidden. Row IDs and study IDs cannot be learned
lookup features. P values, significance calls, or outcome-dependent filters
cannot select membership, folds, architecture, checkpoints, or claims.

### 4.2 Split

The parent authority must be an A2-frozen, label-blind source-group and
near-duplicate component split. Components remain indivisible. Exact sequence,
near-duplicate, source, candidate, reverse-edge, and relevant study/context
leakage must all be zero.

Within the already-frozen development role, component-level assignment with salt
`ROUTE_A_V3_A6_LEARNED_BASE_VALUE_GPU_DRAFT_V1` creates:

- `A6_PARAMETER_TRAIN` — 80%;
- `A6_INDEPENDENT_EXACT_REFERENCE` — 20%.

The assignment is label-blind and cannot be retried after labels or results are
seen. Outer-test labels remain unopened.

### 4.3 Exposure

This is a scratch-only random-initialization design. The lists of pretrained
foundation checkpoints, pretrained weights, warm starts, external learned
embeddings, and external pretraining corpora are all exactly empty. Loading a
checkpoint before the first optimizer step is forbidden. A later crash resume is
allowed only from the latest scheduled checkpoint from the same run, with the
same protocol, data, split, optimizer state, and run ID.

Scratch-only is an explicit model-input route, not a waiver of exposure control.
Adding any external learned input requires a new protocol and a new exposure
audit.

### 4.4 Rights and public output

Each dataset must authorize qualification use and private processing/evaluation.
Unknown rights exclude that dataset and force recomputation of the eligible
universe. Public redistribution rights are not required because member-level
redistribution is forbidden: public outputs are aggregate only. No public member
ID, sequence, effect, prediction, replicate identity, or split assignment may be
emitted.

## 5. Frozen base and value architecture

### 5.1 Sequence representation

Each position uses 28 deterministic features:

- 5-way source one-hot;
- 5-way current one-hot;
- source-relative edit mask;
- valid mask;
- 16 sinusoidal position features.

The encoder is a masked 64-channel dilated residual Conv1D:

- stem kernel 5;
- four residual blocks, kernel 3, dilations 1, 2, 4, 8;
- GELU activation and position-wise layer normalization;
- no dropout in the sequence encoder;
- valid-mask mean and max pooling, producing 128 global features.

Observable context uses a randomly initialized 16-dimensional embedding with a
required `UNK_CONTEXT` token and context dropout 0.1. Study-ID embeddings are
forbidden. Remaining budget uses an 8-dimensional scratch embedding. Algorithmic
time contributes `t` and `log1p(t)`. The state vector width is 154.

### 5.2 Base-rate head

For an edit transition, the head receives the 64-dimensional local position
token, the 154-dimensional state vector, 5-way alt-base one-hot, normalized
position, and STOP indicator. STOP uses its distinct canonical action feature.
The 225-dimensional action vector passes through a `128 -> 64 -> 1` GELU MLP.
Illegal actions are never evaluated. Only canonical next states are scored.

### 5.3 Value head

The value network has the same encoder shape but separate randomly initialized
parameters. No parameter is shared with the base network. After base training,
the final base checkpoint is frozen. The 154-dimensional state vector passes
through a `128 -> 64 -> 1` GELU MLP, followed by the positive-h
parameterization above.

There is no foundation component and no architecture alternative in this
protocol. Changing widths, layers, activation, context handling, sharing, or
support semantics requires a new protocol version; it is not a permitted repair
inside one run.

## 6. Terminal tilt and objectives

### 6.1 Terminal tilt

The terminal score must be a previously frozen, calibrated lower confidence
bound from the allowed ordinary-development role. A bare critic mean is
forbidden. Within each training-only study/assay/context stratum, the score is
robustly standardized by median and MAD. A MAD below `1e-6` fails the protocol.

With `z_lcb` clipped to `[-4,4]` and fixed beta 1.0:

\[
w(a)=\exp(\operatorname{clip}(z_{LCB}(a),-4,4)).
\]

Thus terminal weights are strictly positive. The complete tilt manifest is
frozen before value updates. Any change requires a new protocol.

### 6.2 Base objective

Eligible observed candidates have at most five source-relative edits. For each
source/candidate pair, all target-edit orders and the final STOP are marginalized
exactly by subset dynamic programming. Legal non-target edits and premature STOP
remain competing probability mass; they are not deleted from denominators.

The base loss is exact observed-terminal negative log likelihood. Weighting is
equal study, then equal biological source group, then equal candidate within a
group. There is no significance weighting and no observed holding-time target.
The fixed unit exit hazard supplies algorithmic scale.

### 6.3 Value objective

The base parameters are frozen. Exact enumerable graphs use five editable
positions and budgets 1, 3, and 5. Measured edit positions are included, then any
remaining positions are filled using the frozen label-blind salt. Exact harmonic
`h` values are computed from the frozen base and terminal tilt. The value loss is
Huber loss, delta 0.1, on `log h`, weighted equally by graph and then transient
state. Joint base/value training is forbidden.

## 7. Independent exact reference and approximation gate

The independent reference implementation is CPU-only, does not import the
learner or Torch, and uses both backward dynamic programming and exhaustive path
enumeration. They must agree in terminal-distribution TV within `1e-12` before
the reference is valid.

The suite has 96 held-out graphs: 32 each for budgets 1, 3, and 5. A graph is one
frozen source-group/context with five label-blind editable positions; it is not
one candidate row. Within each budget, eligible frozen component IDs are ordered
by SHA-256 of the assignment salt plus component ID and the first 32 are used.
This digest is only a deterministic assignment function, not an evidence claim.
Fewer than 32 eligible graphs in any budget fails the protocol. Reference graphs
cannot update parameters or select a checkpoint, and only aggregate results may
be published.

For each canonical reference edge define

\[
e(s,s')=\left|
[V_\theta(s')-V_\theta(s)]
-[\log h^*(s')-\log h^*(s)]
\right|.
\]

Both gates are mandatory:

1. the maximum, over all 96 graphs, of the within-graph 95th percentile of
   `e(s,s')` is at most 0.10;
2. the maximum guided terminal-distribution TV over all graphs is at most 0.02.

No graph deletion, abstention, alternate checkpoint, seed, or threshold can
rescue a failure. Failure yields `FAIL_CURRENT_PROTOCOL_KEEP_A6_IN_PROGRESS`.

## 8. Single seed, optimizer, compute, and checkpoint policy

The only candidate policy is:

- seed `2026081401`; exactly one seed;
- AdamW, learning rate `3e-4`, betas `(0.9,0.999)`, epsilon `1e-8`, weight
  decay `1e-4`;
- constant learning rate, L2 gradient clip 1.0;
- float32, source-group batch size 32, no gradient accumulation;
- 40,000 base steps followed by 20,000 value steps;
- no early stopping, HPO, sweep, best-seed choice, or best-checkpoint choice.

Recovery checkpoints are written every 5,000 steps and are crash-resume only.
The only scientific checkpoint roles are base final step 40,000 and value final
step 20,000. A batch, precision, optimizer, schedule, data, or split change
requires a new run ID and new authority; it cannot overwrite or silently resume
the old run.

This single-seed run, if later authorized, can test A6 implementation and
approximation gates. It cannot make a five-seed or matched-compute scientific
claim.

## 9. CUDA fail-closed policy

Every parameter update must run on exactly one CUDA GPU whose UUID is frozen in
the later run authority. CPU parameter updates and silent CPU fallback are
forbidden. Before data, model, CUDA, or output mutation, the future runner must
confirm:

- CUDA is available;
- model and every train tensor are on CUDA;
- GPU UUID matches authority;
- PID, owner, memory, and utilization are recorded;
- no unrelated job is preempted;
- the output run ID is absent.

The run manifest records GPU UUID/model, driver, CUDA, PyTorch, device index,
and peak VRAM. Any failed check gives
`FAIL_CLOSED_ZERO_PARAMETER_UPDATE_ZERO_OUTPUT`.

No CUDA check is run while producing this draft because GPU work itself is not
authorized.

## 10. Mandatory later gates

An authorized run would have to pass all of the following without post-hoc
repair:

- time-homogeneous math scope and distinct `w`, exact `h`, and learned `V`;
- independent DP/path-enumeration agreement;
- unit-tilt base recovery with maximum rate relative error `<=1e-5`; this is an
  operator fixture using `w=1` and the exact constant `V=0`, not a requirement
  that a learned value network numerically outputs zero;
- hard legality 100%, support coverage 100%, and zero budget violations;
- source anchoring, no repeated edit, no revert, STOP competition, budget,
  no-action, numerical-failure, unequal-alias, and path-product fixtures;
- deterministic trajectory replay 100%, trajectory legality 100%, and zero
  trajectory budget violations;
- a frozen 20,000-trajectory statistical check using seed `2026081402`, with
  holding-time mean relative error at most 0.02 and sampled terminal-law TV at
  most 0.02;
- both learned-potential approximation thresholds in Section 7.

Failure remains failure. An earlier recovery checkpoint, another seed, altered
threshold, excluded graph, or CPU fallback cannot convert it to PASS.

## 11. Future artifacts, not outputs of this draft

If a later owner authorizes execution, the run must produce a frozen protocol
snapshot; input-role/rights, split/exposure, CUDA, and training manifests; final
base/value checkpoint manifests; recovery-role manifest; independent exact
reference report; aggregate base-recovery/legality/support, learned-potential,
trajectory, and compute reports; and a failure bundle when applicable.

Those artifacts do not exist now. This draft creates no runtime directory and no
event. Public reporting remains aggregate only and must attest zero private and
sealed access.

## 12. Final disposition

The completed candidate remains
`DRAFT_FOR_REVIEW_NOT_ACTIVE_PROTOCOL`. Its sole next action is independent
review followed by an explicit owner decision on whether to promote a successor
as active protocol. Even after review, **actual parameter updates require a
separate later authorization**. Until then, training, GPU work, model selection,
A6 PASS, L3 establishment, A7 unlock, qualification, credits, canonical changes,
and private/sealed access all remain prohibited and unchanged.
