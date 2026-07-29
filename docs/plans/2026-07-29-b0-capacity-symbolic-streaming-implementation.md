# B0 Capacity Diagnostic and Exact Symbolic/Streaming Prototype Implementation Plan

> **Execution contract:** Follow
> `docs/contracts/mrna_latest_build_contract_v2.md` at SHA-256
> `c3dc5875868d847b8519fee40b14c43b65e4c5948dc5c3b98101ca61a5671dd5`.
> This plan is non-formal and cannot resume, accept, or freeze B0.

**Goal:** Produce a replayable, label-free B0 path-capacity diagnostic and an
exact external-memory prototype that preserves all-shortest-path
sequence-identity semantics without changing any production gate or budget.

**Architecture:** Keep production B0 untouched. Add an isolated exact engine
that streams shortest-path layers through bounded sorted chunks, plus a
pure-indel symbolic certificate for exact preflight node/layer counts. Package
diagnostic executions as unique, immutable `/mnt` bundles bound to frozen
inputs and a clean code commit.

**Tech stack:** Python 3.10+, standard library, pytest, jsonschema, Git, SHA-256.

---

### Task 1: Freeze design and non-formal policy

**Files:**

- Create: `docs/adr/0001-b0-exact-symbolic-streaming-diagnostic.md`
- Create: `docs/plans/2026-07-29-b0-capacity-symbolic-streaming-design.md`
- Create:
  `docs/plans/2026-07-29-b0-capacity-symbolic-streaming-implementation.md`

**Step 1:** Record immutable scientific semantics, frozen inputs, and
non-claims.

**Step 2:** Record the external-memory algorithm, exactness invariants,
alternatives, stop rules, and later budget decision point.

**Step 3:** Confirm no production B0 gate or driver file is in the planned
change set.

### Task 2: Define the diagnostic artifact contract

**Files:**

- Create: `schemas/b0_capacity_diagnostic.schema.json`
- Create: `tests/test_b0_capacity_diagnostic.py`

**Step 1: Write failing schema tests**

Test that a valid non-formal summary passes and that any of the following
fails: `formal=true`, acceptance/freeze/scientific claims, gate change, missing
contract/input/code hashes, mixed exact/lower-bound fields, omitted row counts,
or an unsealed completed run.

**Step 2: Run the focused test and confirm failure**

Run:

`python -m pytest -q tests/test_b0_capacity_diagnostic.py`

Expected: FAIL because the schema and CLI do not exist.

**Step 3: Implement the minimal schema**

Define strict manifests for policy flags, provenance bindings, resource
limits, row status, exact/lower-bound separation, summary completeness, and
terminal status.

### Task 3: Implement exact symbolic pure-indel counts

**Files:**

- Create: `data/utr_benchmark_v2/symbolic_path_states.py`
- Create: `tests/test_b0_symbolic_path_states.py`

**Step 1: Write failing exhaustive parity tests**

Enumerate bounded A/C/G/U pure insertion/deletion endpoint pairs. Compare
symbolic node/layer counts with `minimum_alignment_state_closure`.

**Step 2: Confirm the focused test fails**

Run:

`python -m pytest -q tests/test_b0_symbolic_path_states.py -k symbolic`

**Step 3: Implement the minimal deterministic subsequence-language counter**

Return exact layer counts, total node count, endpoint hashes, algorithm ID, and
proof digest. Reject mixed edits and invalid RNA. Never infer edge/path counts.

**Step 4: Re-run the symbolic tests**

Expected: PASS.

### Task 4: Implement exact external-memory closure traversal

**Files:**

- Modify: `data/utr_benchmark_v2/symbolic_path_states.py`
- Modify: `tests/test_b0_symbolic_path_states.py`

**Step 1: Write failing parity and invariance tests**

Compare states digest, nodes, edges, paths, primitive actions, and DP cells
against the legacy enumerator on bounded substitution/indel/mixed cases.
Repeat with multiple chunk sizes and shuffled emission order.

**Step 2: Write failing resource-stop tests**

Force state, contribution, byte, DP-cell, and action limits. Require typed
stop evidence and absence of a completed exact result.

**Step 3: Implement chunk spill, k-way merge, atomic layer completion, global
state merge, and exact counters**

Use full sequence strings as sort keys. Use Python integers for counts. Do not
use probabilistic hashes for equality.

**Step 4: Re-run focused tests**

Expected: PASS with deterministic results.

### Task 5: Implement the replayable diagnostic CLI

**Files:**

- Create: `scripts/data/diagnose_b0_path_capacity.py`
- Modify: `tests/test_b0_capacity_diagnostic.py`

**Step 1: Write failing bundle tests**

Use a tiny label-free fixture. Verify unique-root refusal, candidate-store hash
binding, stable record selection, one output row per selected record, exact
versus uncomputed separation, logs, checksums, replay command, status, and
terminal seal.

**Step 2: Implement unique bundle creation**

Add explicit `--candidate-store`, `--d1-snapshot`, `--output-root`,
`--contract`, `--expected-code-commit`, mode, selection, and resource-limit
arguments. Refuse label-bearing structural rows and existing roots.

**Step 3: Implement mandatory witness precheck**

Bind the frozen record ID/endpoints and require its exact known node, edge,
path, and digest values before `--scope census` may continue.

**Step 4: Implement terminal validation and sealing**

Create checksums and `terminal.lock` only after the final status and summary
validate. On exception, retain provenance and failure evidence, return
non-zero, and never mark partial values exact.

**Step 5: Re-run focused tests**

Expected: PASS.

### Task 6: Verify isolation and regression safety

**Files:**

- Modify only if a test exposes a prototype bug.

**Step 1:** Run the new focused suites.

**Step 2:** Run existing B0 path, near-neighbor, split, leakage, driver, and
acceptance suites.

**Step 3:** Run strict contract and task-registry tests.

**Step 4:** Confirm production constants, formal scripts, frozen D1 files, and
existing artifacts are byte-identical to the preflight snapshot.

### Task 7: Commit and publish the reviewed prototype

**Step 1:** Inspect the authoritative worktree diff and exclude caches, data,
weights, logs, and unrelated files.

**Step 2:** Create a focused commit for design, schema, code, and tests.

**Step 3:** Push branch `d1-b0-utr-v2-20260729` and read back the remote hash.

**Step 4:** Require a clean committed code hash before the real diagnostic.

### Task 8: Run the non-formal witness and capacity diagnostic

**Artifacts:**

- Create a unique external root under
  `/mnt/cunyuliu/mrna_editflow_b0_capacity/`
- Do not write large run data into Git.

**Step 1:** Run the frozen witness replay with conservative explicit prototype
limits.

**Step 2:** Validate exact frozen counts/digest, bundle schema, hashes,
checksums, logs, and terminal seal.

**Step 3:** If the witness passes, run the deterministic structural census
requested by the diagnostic plan. Do not launch a formal B0 driver.

**Step 4:** Monitor only at the contract’s low-frequency cadence. While
waiting, audit manifests, row accounting, deterministic selection, and
production isolation.

**Step 5:** On a resource stop, preserve evidence and stop; do not raise the
limit within the same run unless a separately predeclared diagnostic envelope
already authorizes that exact replay.

### Task 9: Independently audit and report the budget decision evidence

**Files:**

- Create:
  `docs/audits/2026-07-29-b0-capacity-symbolic-streaming-diagnostic.md`
- Create a small Git-tracked diagnostic index/summary only if it contains no
  large data and binds the external bundle by SHA-256.

**Step 1:** Independently verify witness parity, exact/uncomputed accounting,
resource observations, and replay.

**Step 2:** State what is exact, what is a lower bound, and what remains
uncomputed.

**Step 3:** Compare measured needs with existing production budgets without
changing them.

**Step 4:** Commit and push the focused audit evidence.

**Stop condition:** End with B0 still `SAFE_PAUSED`. Present the credible
numbers and decision consequences to the user. A fresh formal B0 or any budget
change requires a subsequent explicit decision.

