# Public Intervention Contract R0 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reset the mRNA-EditFlow research contract around publicly measured mRNA intervention effects, with auditable legacy archiving and an execution registry.

**Architecture:** Preserve legacy files as unchanged Git renames under archive directories, then make one YAML contract authoritative for the scientific question, benchmark, datasets, sealed external evaluation, endpoint separation, splits, metrics, and claim boundaries. Validate the contract with focused pytest tests and validate all execution tasks against a JSON Schema-compatible registry validator.

**Tech Stack:** Git, YAML, JSON Schema, Python 3, pytest.

---

### Task 1: Archive the superseded P3/NMI contract

**Files:**
- Move: legacy P3 config files to `configs/archive/p3_legacy/`
- Move: legacy P3 document files to `docs/archive/p3_legacy/`
- Add: `configs/archive/p3_legacy/SUPERSEDED.md`
- Add: `docs/archive/p3_legacy/SUPERSEDED.md`
- Add: `scripts/contracts/audit_legacy_references.py`
- Test: `tests/test_audit_legacy_references.py`

**Step 1: Record the pre-archive commit and preserve file contents.**

Run: `git rev-parse HEAD`; verify archived files are Git-identical renames and historical results are untouched.

**Step 2: Remove active contract references.**

Update active README/container ignore references to point to `public_intervention_contract_v1` and archive paths. Keep historical archive content unchanged.

**Step 3: Run the strict audit.**

Run: `python scripts/contracts/audit_legacy_references.py --strict`

Expected: `active paper code references to legacy contract = 0`.

### Task 2: Freeze the public intervention contract

**Files:**
- Create: `configs/public_intervention_contract.yaml`
- Create: `docs/public_intervention_scientific_question.md`
- Create: `docs/public_intervention_claim_matrix.md`
- Create: `tests/test_public_intervention_contract.py`

**Step 1: Fix the scientific question and benchmark.**

Set `EditBench-5U-Natural` as the primary benchmark, retain separate dense, 3'UTR, and CDS-family sub-benchmarks, and identify Sample 2019, PLUMAGE, GSE145046, and the sealed GSE246381 external test.

**Step 2: Fix endpoint, split, metric, claim, and no-wet-lab boundaries.**

Keep MRL, TE, RNA abundance, half-life, and protein abundance separate; forbid random pair splits and sealed-label tuning; use macro-averaged per-study metrics; prohibit therapeutic/protein-output/wet-lab claims.

**Step 3: Run focused contract tests.**

Run: `python -m pytest tests/test_public_intervention_contract.py -q`

Expected: all focused contract tests pass.

### Task 3: Add the execution task registry

**Files:**
- Create: `docs/execution/task_registry.yaml`
- Create: `schemas/task_registry.schema.json`
- Create: `scripts/execution/validate_registry.py`
- Test: `tests/test_validate_registry.py`

**Step 1: Register R0 and downstream tasks.**

Each task records `task_id`, `status`, `dependencies`, `inputs`, `in_scope`, `out_of_scope`, `files`, `commands`, `outputs`, `acceptance`, `repair_loop`, `commit_sha`, and `report`.

**Step 2: Validate schema, fields, IDs, dependencies, and cycles.**

Run: `python scripts/execution/validate_registry.py`

Expected: `task registry VALID` with zero errors.

**Step 3: Run registry tests.**

Run: `python -m pytest tests/test_validate_registry.py -q`

Expected: all registry tests pass.

### Task 4: Final verification and handoff

Run the strict legacy audit, focused contract and registry tests in the configured `editflow` Python environment, then inspect `git diff --check`, Git rename detection, and the final worktree status. Do not reset or commit unrelated experiment artifacts, data, checkpoints, or training changes.
