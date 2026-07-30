# Scientific Question — UTR EditFlow V2.2

**Contract:** `utr_editflow_goal_v2`

**Question ID:** `RQ-UTR-EDITFLOW-V2`

**Goal SHA256:** `1ec94044e7f4626c3a0d9848e31a2f7122ed6d09ac1919a73cbda659c5d3993d`

**B0 amendments:** Capacity diagnostics are non-blocking for B0 under v2.2;
the active path-leakage audit deterministically replays each accepted D1
canonical edit script, its declared intermediates and endpoints. The historical
E1 failure evidence remains preserved and does not become a PASS or an efficacy
result. This qualification does not claim enumeration or clearance of all
alternative edit orders or dynamic paths.

## Primary question

Given an existing UTR source, region, assay/context, functional endpoint,
target condition, edit budget, and hard constraints, can a source-conditioned,
region-aware, grammar-constrained continuous-time mRNA-EditFlow learn
transferable distributions over legal edit trajectories and generate diverse,
sparse, variable-length, controllable 5′UTR and 3′UTR candidates?

The question is generative. Effect prediction, foundation models, benchmarks,
critics, and search are supporting systems. They cannot replace the Edit Flow
rate field or turn a candidate scorer into the primary method.

## Comparative question

Under matched training data, backbone, trainable parameters, optimization
steps, GPU budget, candidate count, oracle queries, action space, constraints,
edit budget, seeds, and model-selection budget, does mRNA-EditFlow produce a
better or non-dominated functional-control/validity/diversity/edit-cost/
efficiency Pareto frontier than:

- candidate-only and paired effect models;
- autoregressive full-sequence and edit-action models;
- masked iterative generation and discrete diffusion;
- generic/unconstrained Edit Flow;
- random legal, exhaustive, greedy, beam, best-of-N, simulated annealing,
  evolutionary, and local search.

## Formal model boundary

The target rate field is:

\[
\lambda_\theta(a \mid x_t, x_0, r, c, y^*, b, t)
\]

where `x_t` is the dynamic current UTR, `x_0` is the fixed source, `r` is the
5′/3′ region, `c` contains assay/context/endpoint, `y*` is the target, `b` is
remaining edit budget, `t` is continuous time, and `a` is INS/SUB/DEL/STOP.

The intermediate path is a latent algorithmic edit trajectory unless an
experiment directly observed it. Endpoint pairs do not identify a unique
biological path.

## Falsifiable hypotheses

### H1 — Edit-process modelling

The continuous-time source-conditioned rate field improves held-out
generative likelihood, transition reconstruction, candidate recovery, and
calibration over candidate-only, subtraction, Siamese, AR action, diffusion,
and generic Flow models under matched budgets.

### H2 — Architecture is not replaceable

Source conditioning, time, INS, DEL, SUB, STOP, dynamic variable-length state,
multi-step trajectories, region conditioning, legal masks, budget state, and
target/context conditioning each make a measurable contribution.

### H3 — Constructive hard validity

Invalid nucleotide, forbidden-position, anchor, budget, length, and identity
violations are exactly zero at every step and final sample. Rejection sampling
or post-hoc repair does not satisfy this hypothesis.

### H4 — Conditional controllability

Changing region, assay/context, endpoint, direction, or target strength changes
the generated distribution in a reproducible and interpretable way, while
permutation controls fail as expected.

### H5 — Generative advantage over search

At matched candidate/query/compute budgets, Edit Flow is not fully dominated
by strong search on measured recovery, independent-critic score, diversity,
edit cost, inference latency, and oracle-query efficiency.

### H6 — Cross-source and cross-study transfer

The learned edit process transfers under source-, gene-, study-, context-, and
exposure-aware holdouts. Random pair splits cannot be headline evidence.

### H7 — Foundation-model value

Foundation representations improve sample efficiency or transfer relative to
a reasonable from-scratch control. Frozen, adapter/LoRA, partial unfreeze, and
justified full fine-tune are compared fairly.

### H8 — Shared and region-specific UTR structure

5′UTR and 3′UTR share edit semantics and the main rate-field framework, while
endpoint heads, motif/anchor rules, length priors, context metadata, and region
adapters remain separate. A joint model must be compared with fully shared and
fully separate controls.

## Current scope and prohibited substitutions

Current scope is 5′UTR plus 3′UTR only. CDS, protein-conditioned codon flow,
full-transcript optimization, therapeutic efficacy, and new wet-lab
experiments require a new user-approved contract.

Results may not change the question into a predictor-only project. A failed
hypothesis is recorded as a negative result and followed by the preregistered
diagnostic or alternative evidence path without lowering a gate.

## Evidence boundary

- E0 engineering tests do not prove biological performance.
- E1 internal or proxy gains do not prove external transfer.
- E2/E3 require held-out measured labels with frozen selection.
- GSE246381 is E4, historically exposed retrospective evidence.
- E5 requires genuine pre-access freeze and an exposure audit.
- E6 is outside the current contract.

Open-support candidates can be described only as predicted, computational, or
proxy-supported. Hard validity is admissibility, not benefit.
