# Phase D0 Data Discovery Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an auditable public mRNA intervention-data inventory and acquisition layer that follows `public_intervention_contract_v1`.

**Architecture:** Treat the frozen contract as the scientific gate: candidates retain explicit region, endpoint, evidence grade, pair type, WT/mutant availability, and sealed-test role. Keep catalog classification, candidate discovery, raw acquisition, checksum verification, and missing-data decisions as separate artifacts; never turn an unavailable or proxy dataset into a primary intervention result.

**Tech Stack:** Python 3, pandas, Parquet, YAML, JSON manifests, GEO/ENCODE/MaveDB APIs, curl, pytest.

---

### Task 1: Audit and stop stale jobs

**Files:**
- Inspect: GPU process table, project process table, logs, output manifests
- Record: D0 task-to-command mapping in the final report

**Step 1:** Map each GPU PID to command, cwd, parent tree, output, and current progress.

**Step 2:** Gracefully terminate only explicitly verified stale or out-of-scope project processes; preserve active P0 downloads and all partial/raw files.

**Step 3:** Recheck process table and download logs.

### Task 2: Rebuild the Excel inventory

**Files:**
- Input: `data/raw/codonflow_integrated_dataset_catalog_ranked.xlsx`
- Run: `scripts/data/import_excel_inventory.py`
- Outputs: `data_registry/excel_inventory.parquet`, `docs/data/excel_inventory_audit.md`
- Test: `tests/test_import_excel_inventory.py`

**Acceptance:** exactly 78 model rows and 14 resource rows are classified with no unexplained entries; the input SHA-256 is recorded.

### Task 3: Discover intervention datasets

**Files:**
- Run: `scripts/data/systematic_search.py`
- Outputs: `data_registry/intervention_candidates.yaml`, `docs/data/systematic_search_protocol.md`, `docs/data/systematic_search_results.md`, raw query artifacts
- Test: `tests/test_systematic_search.py`

**Acceptance:** every candidate has paper, accession, variant count, region, endpoint, WT/mutant/raw-count availability, license, evidence grade, sub-benchmark, and role. Primary benchmark candidates remain A1 true WT-mutant pairs; dense and synonymous-family data remain separate.

### Task 4: Acquire and verify P0 data

**Files:**
- Run: `scripts/data/download_geo.py`, `scripts/data/download_encode.py`, `scripts/data/download_mavedb.py`
- Verify: `scripts/data/verify_downloads.py`
- Outputs: external raw data root with per-accession manifests and `docs/data/download_verification.md`

**Acceptance:** successful files have provider URL/accession, byte size, SHA-256, and no HTML error page; failed or deferred files remain explicitly recorded.

### Task 5: Manage missing data and report boundaries

**Files:**
- Create/update: `data_registry/unavailable.yaml`
- Create/update: `docs/data/missing_dataset_acquisition.md`
- Update: `docs/execution/task_registry.yaml`

**Acceptance:** every unavailable dataset records searched locations, supplementary files, author code, archives, author-contact need, raw-read reconstruction possibility, current substitute, and next repair loop. No missing dataset is silently deleted or promoted to benchmark evidence.

### Task 6: Verify and commit

Run focused D0 tests, contract-aware provenance checks, download verification, registry validation, and `git diff --check`; commit only D0 artifacts/scripts/docs/tests. Do not include unrelated training outputs, checkpoints, or other users' work.
