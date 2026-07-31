# C0+D0 UTR EditFlow V2 Implementation Plan

> **For Codex:** REQUIRED SKILLS: use `Code`, `writing-plans`, and
> `nature-academic-search`; execute task-by-task with contract gates.

**Goal:** Align the active repository with `utr_editflow_goal_v2` and freeze
auditable hypothesis-driven data discovery without launching formal training.

**Architecture:** Work in an isolated sparse worktree based on the preflight
HEAD. Preserve old contracts as unchanged archives, make one V2 authority
chain active, enforce it with structured audits/tests, and treat D0 as
capability qualification rather than model evidence.

**Tech Stack:** Python 3.10+, PyYAML, pytest, JSON Schema documents, Git, CUDA
preflight via PyTorch/NVIDIA tooling.

**Execution status:** C0 and D0 acceptance gates are verified. The external
ENCODE reconstruction remains 61/62 and continues independently; the D0
inventory represents all 62 expected rows and remains fail-closed with
`complete=false`.

---

### Task 1: Protect current state

**Files:**

- Create: `artifacts/stages/C0_D0_20260728T120329Z_9f43133/preflight_manifest.json`
- Create: `artifacts/stages/C0_D0_20260728T120329Z_9f43133/contract_goal.sha256`
- Create: `docs/contracts/v2_contract_conflict_matrix.md`

**Steps:**

1. Record Goal hash, HEAD, branch, dirty diff hash, processes, GPU, RAM/disk,
   ENCODE status, and missing V2 outputs.
2. Create an isolated worktree and confirm it is clean.
3. Verify original training/download PIDs remain outside the isolated worktree.

**Test:** compare recorded hashes/status with the read-only preflight output.

### Task 2: Freeze the V2 authority chain

**Files:**

- Create: `configs/utr_editflow_execution_policy.yaml`
- Create: `docs/utr_editflow_scientific_question.md`
- Create: `docs/utr_editflow_claim_matrix.md`
- Create: `docs/decision_log.md`
- Modify: `README.md`
- Archive: V1 contract, V1 question/claim matrix, V1 task registry and plans

**Steps:**

1. Write failing contract assertions for ID/hash, Flow-primary, UTR-only,
   GSE246381 historical exposure, no wet lab, and GPU-only formal runs.
2. Add the machine-readable contract and documents.
3. Align README and archive V1 files without changing their contents.
4. Run contract tests.

**Test:** `pytest -q tests/test_single_contract.py`.

### Task 3: Add fail-closed contract and execution audits

**Files:**

- Create: `scripts/contracts/audit_single_contract.py`
- Create: `configs/execution_contract.yaml`
- Create: `schemas/run_manifest.schema.json`
- Create: `docs/execution/state_machine.md`
- Create: `scripts/execution/preflight.py`
- Create: `scripts/execution/launch_gpu_run.py`
- Create: `scripts/execution/monitor_run.py`
- Create: focused contract/execution tests

**Steps:**

1. Add a structured invariant audit plus active legacy-reference scan.
2. Define run IDs, absolute artifact layout, required manifest fields,
   fail-closed CUDA probe, low-frequency monitoring and stop rules.
3. Test CPU fallback rejection, NaN/Inf detection, manifest contents, and
   registry validity.

**Test:** `python scripts/contracts/audit_single_contract.py --strict` and
focused pytest suite.

### Task 4: Freeze D0 hypothesis and dataset capability

**Files:**

- Create: `docs/data/hypothesis_data_requirement_matrix.md`
- Create: `data_registry/dataset_capability_matrix.csv`
- Create: `docs/data/systematic_search_protocol.md`
- Create: `docs/data/systematic_search_results.md`

**Steps:**

1. Map H1–H8 to minimum/ideal supervision and forbidden claims.
2. Requalify every current candidate under V2.
3. Search official repositories and academic indexes for measured
   insertion/deletion, variable-length, multi-edit, and new external data.
4. Keep metadata-only discoveries separate from accessed labels.
5. Record negative search conclusions without weakening the core question.

**Test:** dataset matrix schema/coverage test and manual source audit.

### Task 5: Build non-invasive ENCODE inventory

**Files:**

- Create: `scripts/data/build_encode_inventory_v2.py`
- Create: `data_registry/encode_62_inventory_v2.csv`
- Create: `data_registry/encode_62_inventory_v2_summary.json`
- Create: `tests/test_build_encode_inventory_v2.py`

**Steps:**

1. Read the external reconstruction manifest only.
2. Write one row per expected accession or explicit missing row.
3. Count verified/failed/partial/missing entries and preserve checksums.
4. Keep role fixed as observational/pretraining candidate.

**Test:** fixture with verified, failed, and missing records; live summary must
not claim completion while any record is incomplete.

### Task 6: Acceptance, commit and publication

**Files:**

- Create: `artifacts/stages/C0_D0_20260728T120329Z_9f43133/acceptance.json`
- Update: `docs/execution/task_registry.yaml`

**Steps:**

1. Run contract, audit, registry, D0 and execution tests.
2. Run `git diff --check` and inspect the exact focused diff.
3. Mark C0 verified only if all C0 gates pass.
4. Mark D0 verified only if its discovery gates pass; keep ENCODE acquisition
   partial if checksum coverage is incomplete.
5. Commit only C0/D0 files and push the isolated branch to GitHub.

**Test:** acceptance JSON contains exact commands, exit codes, hashes and
honest phase states.
