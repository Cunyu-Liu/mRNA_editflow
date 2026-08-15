# DEC028 A6 Piecewise Time-Inhomogeneous Exact-Reference Candidate Implementation Plan

> **Execution policy:** Implement by functional batch under the global lightweight development and verification strategy. Do not apply task-by-task review or full-validation gates unless the batch risk warrants them.

**Goal:** Add a separate, non-authoritative synthetic CPU reference for a finite, piecewise-constant time-inhomogeneous CTMC on source-anchored edit DAGs, without implying contract-wide general time-inhomogeneous exactness or an A6/L3/scientific PASS.

**Architecture:** A standard-library module builds the same finite source-anchored, no-reedit DAG class used by the current G0 exact-96 candidate. During a finite continuous algorithmic-time schedule, the canonical edit and STOP rates change by segment. A Poisson-uniformization propagator computes the transient distribution with a recorded absolute tail bound. At the final finite boundary, a time-homogeneous tail is absorbed by a topological DP, whose root result is independently checked by complete terminal-path enumeration for every fixture. A separately implemented RK4 step-refinement calculation is a numerical cross-check, not a replacement for the certified reference.

**Tech Stack:** Python standard library, `dataclasses`, `math`, `pytest`; no Torch, project-data path, device/CUDA access, runtime publisher, model, checkpoint, or optimizer.

---

### Batch 1: Piecewise time-inhomogeneous synthetic CTMC candidate

**Files:**

- Create: `configs/route_a_v3_dec028_a6_piecewise_time_inhomogeneous_exact_g0_candidate_v1.json`
- Create: `scripts/route_a_v3/dec028_a6_piecewise_time_inhomogeneous_exact_g0_candidate.py`
- Create: `tests/route_a_v3/test_dec028_a6_piecewise_time_inhomogeneous_exact_g0_candidate.py`

**Risk:** High — a propagation mistake could conceal a lost probability mass, normalize raw aliases in the wrong order, reduce time dependence to a common scale factor, or claim an exactness scope wider than the finite schedule actually proves.

**Implementation:** Generate exactly `4 source lengths × 3 budgets × 8 variants = 96` deterministic binary-alphabet fixtures. Each structural state permits only source-relative one-time edits and a positive-rate STOP; every raw edit/STOP action has two aliases that aggregate by complete successor state before exit-rate normalization. Use three finite time segments with independently varying edit and STOP multipliers, followed by an explicit time-homogeneous tail. Propagate the nonterminal probability vector under each segment using finite uniformization terms until the retained Poisson tail is at most `1e-13`; carry the sum of segment bounds. Absorb the tail by topological DP and compare its root result with complete legal terminal-path enumeration. Independently integrate the finite schedule with RK4 step refinement and require agreement with the certified reference to `1e-10`. Emit only in-memory aggregate quantities and truthfully label the method as a piecewise-constant schedule reference, not a proof for arbitrary time-varying rates, physical kinetics, A6 PASS, L3, or A7.

**Minimum verification:** Run `PYTHONDONTWRITEBYTECODE=1 /home/cunyuliu/miniconda3/envs/editflow/bin/python -m pytest -q -p no:cacheprovider tests/route_a_v3/test_dec028_a6_piecewise_time_inhomogeneous_exact_g0_candidate.py`. The suite must run all 96 fixtures; prove the scheduled rates change relative action probabilities; bound uniformization truncation; compare terminal distributions with independently refined RK4; reject alias/order, source-anchoring, configuration, and forbidden-runtime drift; and AST-check that the module imports no model/device/data libraries.

**Independent review:** Yes — before this candidate is used as any current-head A6 evidence beyond `G0_NONLEARNED_SYNTHETIC_PREPARATION`, a reviewer distinct from this implementation must assess the continuous-time semantics, numerical error accounting, and stated scope boundary.
