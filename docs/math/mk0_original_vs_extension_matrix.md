# MK0 original-method versus project-extension matrix

## Status and binding

This document is a provenance boundary, not a performance claim. It implements contract section 31A.1 for `math_kernel_v1` and is bound to contract SHA-256 `3a3a654ca5c10a988eca897bff40be2e0b45c841f744f7423fdfd60b298b5791`.

The paper record frozen for MK0 is **Edit Flows: Variable Length Discrete Flow Matching with Sequence-Level Edit Operations**, arXiv:2506.09018v3, NeurIPS 2025. As of the MK0 provenance snapshot, an official public implementation revision and hash were not verified. The required fields therefore remain exactly:

```yaml
official_implementation_public_status: NOT_VERIFIED_PUBLIC
official_implementation_revision: NOT_VERIFIED_PUBLIC
official_implementation_hash: NOT_VERIFIED_PUBLIC
```

`NOT_VERIFIED_PUBLIC` is an explicit provenance limitation. It must not be replaced by a guessed repository, a third-party implementation, or a hash from this project. The independent project implementation is bound separately by its own Git commit in `mk0_freeze_manifest.json`.

## Permanent boundary matrix

| Construct | Original Edit Flows | mRNA-EditFlow MK0 extension | Future only | Required wording/evidence |
|---|---:|---:|---:|---|
| Variable-length CTMC rate field | yes | reused and independently implemented | no | Time-inhomogeneous generator on the extended runtime state, with explicit external time. |
| Atomic `INS`, `SUB`, `DEL` | yes | reused with UTR legality masks | no | Original action family plus project-specific hard constraints. |
| Alignment-augmented probability path | yes | reused with a frozen canonical coupling and sensitivities | no | Alignment is a latent algorithmic construction, never an observed biological trajectory. |
| Bregman/rate-matching objective | yes | reused at unique-next-extended-state level | no | Report transition aggregation and brute-force oracle results. Do not call an action cross-entropy an Edit Flow objective. |
| Fixed-grid first-order sampler | yes | retained as `paper_first_order_parallel` reference | no | First-order approximation; `exact_gillespie: false`. |
| Fixed source anchoring and `M_run` | no | yes | no | Project extension needed for source preservation and protected anchors. |
| UTR region, grammar and protected-anchor masks | no | yes | no | Constructive hard legality before rate normalization. |
| Atomic edit budget | no | yes | no | Every executed INS/SUB/DEL consumes one unit, including reversals and cycles. |
| 5′/3′ region, assay/context/endpoint and target condition | no | yes | no | Inference-visible conditioning; no final-evaluator feedback. |
| Explicit structural `STOP` | no | yes | no | Post-completion positive dwell survival process with an absolute hazard scale. |
| Forced termination reasons | no | yes | no | Execution semantics outside learned STOP; never score them as correct STOP predictions. |
| Frozen foundation encoder fusion | no | yes | no | Encode both fixed source and dynamic current sequence; GPU-only neural validation. |
| Strict-budget single-event sampler | no | yes | no | `constrained_single_event_first_order`; frozen-rate endpoint approximation, not exact event-time simulation. |
| Functional critic guidance | no | no | FC0 or later | Must remain optional, logged and unable to change hard legality. |
| Final evaluator guidance or selection | no | no | prohibited | Final evaluator is isolated from generator, sampler and selector. |
| Exact normalized sequence likelihood | not established here | no | only after separate proof | Held-out flow objective is not to be labelled exact NLL. |
| Observed biological edit path | no | no | unsupported | Every constructed coupling records `path_is_observed: false`. |

## Frozen formula mapping

The original-method part covered by the independent MK0 implementation is limited to:

1. an `INS`/`SUB`/`DEL` variable-length rate field;
2. an alignment-augmented path with a monotone schedule;
3. a rate/Bregman objective; and
4. a fixed-step first-order sampling reference.

The following are versioned project constructs and must be named as such:

- `Y_t`, which contains source-current mapping, history, budget, conditions and ACTIVE/HALTED status;
- independent switch-clock product coupling as frozen for `mk0-v1`;
- UTR hard masks and protected anchors;
- the explicit survival STOP process;
- foundation fusion; and
- the strict-budget single-event sampler.

## Prohibited affirmative claims

Project code, configuration, artifacts and manuscript drafts must contain no unsupported affirmative statement that the project sampler is:

- Forbidden claims: “exact Gillespie” or “exact CTMC sampling”;
- an “observed biological edit trajectory”; or
- a proof of biological, functional or matched-budget superiority.

Negative statements, prohibition rules and related-work descriptions are allowed when their context is preserved. The text audit must distinguish those contexts from affirmative claims about this implementation.

## Freeze rule

MK0 may establish only E0 mathematical and engineering evidence. A PASS does not establish functional improvement, a valid final-label result, matched-budget superiority or paper success. Any change to the path, STOP construction, sampler semantics or primary loss creates a new kernel version and hash.
