# DEC028 A2 G0 synthetic evaluator Implementation Plan

> **Execution policy:** Implement by functional batch under the global lightweight development and verification strategy. Do not apply task-by-task review or full-validation gates unless the batch risk warrants them.

**Goal:** Forward-port a strictly synthetic, non-authoritative A2 evaluator candidate that tests only future evaluator semantics and cannot read a project row or establish an A2 result.

**Architecture:** The candidate has one JSON boundary contract and one Python validate-only entry point. It accepts only artificial SYN_ identifiers and synthetic endpoint metadata, forms connected components without outcomes, and computes candidate minus source only after finite endpoint and biological-SE validation. It has no real-evaluation, data-loading, guide, model, CUDA, checkpoint, or output-writing path.

**Tech Stack:** Python standard library, pytest; planning context from @brainstorming and @writing-plans.

---

### Batch 1: Frozen synthetic evaluator boundary

**Files:**

- Create: configs/route_a_v3_a2_g0_evaluator_candidate_v1.json
- Create: scripts/route_a_v3/a2_g0_synthetic_evaluator_candidate.py
- Test: tests/route_a_v3/test_a2_g0_synthetic_evaluator_candidate.py

**Risk:** Medium — an evaluator interface can later influence a scientific claim, so its present scope must reject project data and all real-evaluation paths.

**Implementation:** Freeze the non-authoritative status, forbidden operation set, synthetic identifier namespace, connected-component and outcome-blind split interface, candidate-minus-source direction, and fail-closed missing/nonfinite and biological-SE rules. The command exposes only validate-only output and reports zero runtime contacts.

**Minimum verification:** Run the validate-only command and its focused pytest module. These checks detect accidental I/O/model imports, wrong endpoint direction, leakage of an outcome into split planning, or coercion of missing/nonfinite values to zero.

**Independent review:** Yes — a separate reviewer must inspect the candidate and this plan before any future authority can treat the forward-port as SS1 evidence.

### Batch 2: DEC028 static-bundle binding

**Files:**

- Modify: scripts/route_a_v3/validate_dec028_static_bundle.py
- Test: tests/route_a_v3/test_dec028_static_bundle.py

**Risk:** Low — the static validator only checks that the new candidate remains visibly non-authoritative and cannot silently disappear from the DEC028 bundle.

**Implementation:** Bind the A2 candidate’s status, authority, activation, source-file presence, and zero runtime truth into the existing static-bundle validator. It does not activate DEC028, perform a runtime sync, or create an SS1 scientific result.

**Minimum verification:** Re-run the existing DEC028 static validator and its focused pytest module.

**Independent review:** No beyond Batch 1 — this is a mechanical binding of the already reviewed boundary.

### Batch 3: Aggregate-only static audit record

**Files:**

- Create: docs/reviews/2026-08-15-route-a-v3-dec028-a2-g0-independent-review.md
- Create: scripts/route_a_v3/record_dec028_ss1_a2_g0_static_validation.py
- Test: tests/route_a_v3/test_record_dec028_ss1_a2_g0_static_validation.py

**Risk:** Low — the recorder only preserves the already-permitted synthetic result and refuses to overwrite a pre-existing run directory.

**Implementation:** Write only aggregate validation, manifest, acceptance, and log files to a caller-supplied artifact directory. Record zero row, CUDA, model, checkpoint, and parameter-update contacts, and set the next phase to independent implementation review.

**Minimum verification:** Run the recorder against a temporary directory and assert its four output files, non-overwrite behavior, and NOT_ESTABLISHED acceptance boundary.

**Independent review:** No — this is an auditable rendering of the existing static result.

### Batch 4: Review and handoff

**Files:**

- Create: docs/reviews/2026-08-15-route-a-v3-dec028-a2-g0-independent-review.md

**Risk:** Medium — the review must distinguish a source-level static result from an A2 qualification, real evaluator result, or biological claim.

**Implementation:** Record the checkable static findings and explicitly leave the verdict pending an independent reviewer. Do not upgrade any evidence state or mark SS1 complete.

**Minimum verification:** Confirm the document cites the exact candidate paths and preserves the non-authoritative claim boundary.

**Independent review:** Required — this document is a handoff record, not a substitute for a distinct reviewer.
