# DEC028 A6 G0 current-HEAD forward-port Implementation Plan

> **Execution policy:** Implement by functional batch under the global lightweight development and verification strategy. Do not apply task-by-task review or full-validation gates unless the batch risk warrants them.

**Goal:** Bind the existing non-authoritative A6 G0 candidate to the DEC028 current head, preserve an aggregate-only zero-update audit record, and leave implementation review pending.

**Architecture:** The existing candidate, frozen configuration, and focused test are compared against their staging introduction; they have no source/config/test drift. A small recorder calls only its validate-only path and writes four aggregate audit files to a new caller-supplied directory. The recorder never creates a learner, CUDA contact, checkpoint, or future runtime artifact.

**Tech Stack:** Python standard library, pytest; planning context from @brainstorming and @writing-plans.

---

### Batch 1: Current-HEAD source comparison and binding

**Files:**

- Modify: scripts/route_a_v3/validate_dec028_static_bundle.py
- Create: docs/reviews/2026-08-15-route-a-v3-dec028-a6-g0-current-head-review.md

**Risk:** Medium — a reused candidate could silently drift from the staged semantics and then be misrepresented as an SS1 interface result.

**Implementation:** Bind the A6 candidate status, inactive authority, zero runtime truth, and current-head review handoff to the DEC028 static validator. The review records the no-drift source comparison and still leaves the independent verdict pending.

**Minimum verification:** Run the DEC028 static validator after the binding. It detects a removed candidate/review or any nonzero runtime truth.

**Independent review:** Required — the handoff is not a verdict from an independent reviewer.

### Batch 2: Aggregate-only static audit record

**Files:**

- Create: scripts/route_a_v3/record_dec028_ss1_a6_g0_static_validation.py
- Test: tests/route_a_v3/test_record_dec028_ss1_a6_g0_static_validation.py

**Risk:** Low — the recorder only serializes the candidate's allowed validate-only result and refuses to overwrite an existing run directory.

**Implementation:** Revalidate the candidate’s zero-update result, require zero audit values and no parameter tensor or future output creation, and write aggregate validation, manifest, acceptance, and log files. The acceptance stops at independent implementation review.

**Minimum verification:** Run the recorder test in a temporary directory, including the non-overwrite assertion; run the existing A6 candidate test and the DEC028 static validator.

**Independent review:** No beyond Batch 1 — the code is an auditable rendering of the existing static result.

### Batch 3: Claim-safe handoff

**Files:**

- Create: docs/reviews/2026-08-15-route-a-v3-dec028-a6-g0-current-head-review.md

**Risk:** Medium — a static A6 interface must not be reported as learned-model validation, L3 evidence, A7, or a scientific result.

**Implementation:** Preserve the exact static findings and the remaining independent-review requirement. Do not promote protocol status, run a GPU preflight, or update a qualification/credit/canonical state.

**Minimum verification:** Inspect the review document for the explicit pending verdict and claim boundary.

**Independent review:** Required — a distinct reviewer must execute the focused tests and inspect the source/config comparison.
