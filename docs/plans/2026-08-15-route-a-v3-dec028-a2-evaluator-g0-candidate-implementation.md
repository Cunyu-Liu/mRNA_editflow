# DEC028 A2 Evaluator G0 Candidate Implementation Plan

> **Execution policy:** Implement by functional batch under the global lightweight development and verification strategy. Do not apply task-by-task review or full-validation gates unless the batch risk warrants them.

**Goal:** Add a current-HEAD, synthetic-only A2 evaluator candidate that makes the future single-study evaluator contract machine-checkable without reading project rows, creating real split assignments, or authorizing learned execution.

**Architecture:** The candidate is a separate, non-authoritative config and standard-library Python module. It accepts only opaque synthetic component fixtures and aggregate schema declarations. It validates outcome-blind component-disjoint split *recipes*, candidate-minus-source and direction-normalized endpoint metadata, finite-positive biological-SE policy, missing/nonfinite rejection, and a same-information direct-baseline interface. Its public validation report has no project-data inputs and every real-evaluation, assignment, guide-feedback, model-selection, training, CUDA, or runtime-output request fails before its callback runs.

**Tech Stack:** Python standard library, JSON, dataclasses, `pytest`; no Torch, data loaders, CUDA, project data paths, or runtime writers.

---

### Batch 1: Synthetic evaluator candidate and fail-closed barriers

**Files:**

- Create: `configs/route_a_v3_dec028_a2_evaluator_g0_candidate_v1.json`
- Create: `scripts/route_a_v3/dec028_a2_evaluator_g0_candidate.py`
- Create: `tests/route_a_v3/test_dec028_a2_evaluator_g0_candidate.py`

**Risk:** High — a superficially harmless evaluator can otherwise create outcome leakage, real split assignments, guide-to-evaluator feedback, or a false A2/qualification claim.

**Implementation:** Freeze the non-authoritative DEC028 G0 boundary and the exact `1/1/0/6547`, `NOT_ESTABLISHED`, training/GPU/model-selection/A7/sealed locks. Implement pure validation for synthetic connected-component graphs, an outcome-blind recipe with `split_assignment_count=0`, a component-disjoint future split contract, candidate-minus-source/direction-normalized endpoint metadata, finite-positive biological SE policy, missing/nonfinite rejection, and a same-information baseline contract that rejects guide/checkpoint/model-selection fields. Emit only a JSON validation report to stdout. Guard any real split assignment, evaluator execution, guide feedback, model selection, CUDA, data-row access, or runtime output before the supplied callback executes.

**Minimum verification:** Run `PYTHONDONTWRITEBYTECODE=1 /home/cunyuliu/miniconda3/envs/editflow/bin/python -m pytest -q -p no:cacheprovider tests/route_a_v3/test_dec028_a2_evaluator_g0_candidate.py`. The fixture suite must cover valid synthetic graphs, cross-component/assignment/outcome leakage rejection, endpoint/SE/missingness rejection, baseline isolation, and pre-callback barriers.

**Independent review:** Yes — before this candidate can become part of an active evaluator or P0.9 input, a distinct reviewer must assess its current-HEAD binding and leakage semantics. Passing this batch is only `G0_PREPARATION_NOT_A2_PASS`.

### Batch 2: A6 current-HEAD reconciliation handoff

**Files:**

- Create: `docs/reviews/2026-08-15-route-a-v3-dec028-a6-g0-current-head-reconciliation.md`

**Risk:** Medium — the old non-authoritative candidate must not be silently treated as a DEC028 active implementation merely because its bytes are unchanged.

**Implementation:** Record the reviewed candidate commit, exact no-diff comparison against current HEAD, its focused test result, and the current DEC028 static authority anchors. State explicitly that this is reconciliation evidence, not a distinct independent review verdict, not an active binding, and not authorization for model/CUDA/data/checkpoint/runtime operations.

**Minimum verification:** Confirm the source diff against commit `8fde46ca7daa765fa3a8ad8ce24a3da82ce1a8d0` is empty and rerun `tests/route_a_v3/test_a6_learned_base_value_g0_candidate.py` with cache/bytecode writes disabled.

**Independent review:** No — this document requests and scopes the required distinct review; it cannot satisfy that review itself.
