# DEC028 A6 Nonlearned Exact-96 Candidate Implementation Plan

> **Execution policy:** Implement by functional batch under the global lightweight development and verification strategy. Do not apply task-by-task review or full-validation gates unless the batch risk warrants them.

**Goal:** Create a separate non-authoritative, synthetic CPU exact-reference candidate covering 96 source-anchored acyclic edit-DAG fixtures with budgets `{1,3,5}`, without reactivating the stale historical A6 publisher or claiming general time-inhomogeneous exactness.

**Architecture:** A standard-library module generates deterministic binary-alphabet synthetic DAG cases and builds canonical rates by aggregating two raw aliases per legal edit or STOP transition. It independently computes terminal jump-chain distributions by topologically ordered DAG dynamic programming and complete-path enumeration. It emits only in-memory aggregate reports. The config binds current DEC028 static bytes and freezes all data/CUDA/model/checkpoint/parameter-update/A6/L3/A7/sealed locks as false; its time scope explicitly remains `GENERAL_TIME_INHOMOGENEOUS_EXACTNESS_NOT_ESTABLISHED`.

**Tech Stack:** Python standard library, dataclasses, `pytest`; no Torch, data paths, device access, runtime publisher, or model code.

---

### Batch 1: Exact-96 synthetic reference candidate

**Files:**

- Create: `configs/route_a_v3_dec028_a6_nonlearned_exact96_g0_candidate_v1.json`
- Create: `scripts/route_a_v3/dec028_a6_nonlearned_exact96_g0_candidate.py`
- Create: `tests/route_a_v3/test_dec028_a6_nonlearned_exact96_g0_candidate.py`

**Risk:** High — a graph/reference implementation can accidentally count aliases twice, use illegal edits, hide a budget violation, or overstate an exactness scope.

**Implementation:** Generate exactly `4 source-lengths × 3 budgets × 8 deterministic rate/source variants = 96` synthetic cases. Validate source anchoring, acyclicity, hard legality before rate construction, positive support floor, STOP availability, raw-alias aggregation by full next extended state, budget accounting, and DP/enumeration terminal-distribution equality. Expose only a validate-only aggregate report. Keep general time-inhomogeneous exactness, learned base/value, A6 PASS, L3, A7, data, CUDA, model, checkpoint, and runtime state unestablished/locked.

**Minimum verification:** Run `PYTHONDONTWRITEBYTECODE=1 /home/cunyuliu/miniconda3/envs/editflow/bin/python -m pytest -q -p no:cacheprovider tests/route_a_v3/test_dec028_a6_nonlearned_exact96_g0_candidate.py`. The suite must cover all 96 cases, independent-reference equality, alias aggregation, budget/legality/STOP negative fixtures, status-boundary mutations, and static-only dependencies.

**Independent review:** Yes — before any current-head A6 evidence can be consumed by P0.9 or presented beyond `G0_NONLEARNED_SYNTHETIC_PREPARATION`, a distinct reviewer must assess the exact-reference semantics and the stated time boundary.
