# P0 Data Reconstruction v1 Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: use the executing-plans workflow task by task. This repository has no Git HEAD, so verification snapshots replace commits and worktrees.

**Goal:** Rebuild immutable, untruncated GENCODE v45 and RefSeq canonical mRNA records, derive explicitly capped model views, and issue verifiable family-disjoint and cross-source split manifests without changing existing training inputs or running processes.

**Architecture:** Add a versioned reconstruction pipeline beside the legacy public-corpus pipeline. It verifies raw transport integrity before parsing, writes full canonical records plus source metadata, derives model views with explicit lineage, creates deterministic cross-source family assignments from gene identity, exact transcript identity, and exact protein identity, and emits split manifests through the existing fail-closed split contract. Existing raw files, records, benchmarks, logs, checkpoints, and Stage A processes remain unchanged.

**Tech Stack:** Python 3.10 standard library, immutable dataclasses, JSON/JSONL, SHA-256, gzip, NumPy-free deterministic union-find and `unittest`.

---

## Design decision

Three approaches were considered. Overwriting the current public pipeline and records would silently change inputs used by active training and is therefore unsafe. Wrapping the current truncated records in richer manifests would preserve the core scientific defect. The selected approach creates a parallel frozen reconstruction namespace: canonical records never truncate UTR or CDS, while every lossy model view has a manifest binding it to its canonical source, selection map, caps, code digest, and exact SHA.

The RefSeq file currently under `data/raw/` is retained as historical evidence: its gzip stream is incomplete. Acquisition for this goal targets a fresh version directory, uses a temporary file and atomic promotion after gzip and SHA verification, and records official URL metadata. No existing data file is overwritten.

### Task 1: Safety baseline

**Files:**
- Create outside project: `/home/cunyuliu/mrna_editflow_goal/backups/p0-data-reconstruction-*/`

**Steps:**
1. Record the four protected process commands and current states.
2. Record configuration, existing raw, records, split-manifest, and handoff hashes.
3. Run the full CPU unit-test baseline.
4. Create a source-only snapshot excluding raw/processed data, benchmark, logs, checkpoints, caches, and external environments; record its SHA-256.

### Task 2: Reconstruction contracts and tests

**Files:**
- Create: `data/reconstruction.py`
- Create: `tests/test_data_reconstruction.py`
- Modify: `data/split_contract.py`

**Steps:**
1. Add failing tests for truncated gzip rejection, canonical no-truncation, stable source-prefixed identifiers, duplicate rejection, deterministic ordering, and invalid ORF attribution.
2. Add failing tests for derived-view lineage, selection-map coverage, caps, exact reproducibility, and canonical SHA mismatch.
3. Add failing tests for deterministic family assignments, source/gene/protein unions, cluster-disjoint roles, cross-source exclusions, and manifest tamper rejection.
4. Extend `build_split_manifest` to support a verified excluded role and relative-path serialization without weakening existing verification.
5. Run focused tests and then the existing data/split tests.

### Task 3: Canonical and derived build implementation

**Files:**
- Create: `data/reconstruction.py`
- Modify: `data/download_mrna.py`
- Modify: `data/__init__.py`

**Steps:**
1. Implement streaming raw-file integrity verification that requires a complete gzip stream and binds size/SHA.
2. Parse GENCODE and RefSeq into source metadata plus region-annotated records, retaining full UTR and valid CDS lengths.
3. Reject invalid alphabet/ORF records with stable reason counts; reject duplicate canonical identifiers rather than silently choosing one.
4. Write deterministic canonical records and metadata JSONL with a frozen reconstruction manifest.
5. Derive model-capped views without altering canonical files; write an index/identifier lineage map and verify every derived row against its canonical parent.

### Task 4: Cross-source family and split artifacts

**Files:**
- Create: `data/reconstruction.py`
- Create: `scripts/run_p0_data_reconstruction.sh`

**Steps:**
1. Build deterministic union-find families using normalized source gene identifiers, exact full RNA digests, and exact translated-protein digests.
2. Persist one cluster id per combined derived record plus a family-evidence table.
3. Emit GENCODE, RefSeq, combined-family, and GENCODE-to-RefSeq cross-source split directories.
4. Cover every records universe exactly with train/val/test plus a reasoned excluded role where required.
5. Generate exact-overlap and declared near-neighbour audit reports. Keep `paper_eligible=false` whenever an exhaustive near-neighbour or source-release requirement is not met; never turn a missing audit into a pass.
6. Load every manifest with `load_and_verify_split_manifest` after writing.

### Task 5: Real reconstruction run

**Files:**
- Create under: `data/reconstructed/p0_data_reconstruction_v1/`
- Create under: `benchmark/dev/p0_data_reconstruction_v1/`

**Steps:**
1. Copy the verified GENCODE raw object into the versioned namespace.
2. Freeze the complete RefSeq human RNA release (`human.1` through `human.15`), not the legacy single-file registry subset; verify official catalog length, every complete gzip stream and SHA, then atomically promote each partition.
3. Rebuild both canonical corpora and derived model views.
4. Verify the GENCODE derived view reproduces the legacy record semantics and explain any byte-level difference.
5. Generate family/split artifacts and record counts, hashes, exclusions, and blockers.
6. Do not overwrite or move any legacy raw, records, benchmark, log, or checkpoint.

### Task 6: Audit, verification, and handoff

**Files:**
- Create: `docs/p0_data_reconstruction_audit.json`
- Create: `docs/p0_data_reconstruction_audit.md`
- Create: `docs/p0_data_reconstruction_v1_handoff.md`

**Steps:**
1. Run focused reconstruction and split-contract tests.
2. Run the complete offline CPU `unittest` suite and report the exact count, failures, and skips.
3. Re-hash and independently reload all produced manifests.
4. Confirm canonical records exceed or equal every corresponding derived region length and that lineage is total and one-to-one.
5. Recheck all four protected Stage A PIDs read-only and record that no process-control action occurred.
6. List created/modified files, real artifact hashes, current blockers, recovery snapshot, and the recommended next goal.

## Definition of done

- Both raw sources pass transport integrity checks in the frozen namespace.
- Canonical records contain no length truncation; lossy views are explicitly named and fully traceable.
- GENCODE, RefSeq, combined-family, and cross-source manifests load through the immutable split contract.
- All family assignments and exclusions cover their record universes without cross-role cluster or exact-sequence overlap.
- Any uncompleted exhaustive near-neighbour audit remains an explicit block reason and keeps the affected manifest non-paper-eligible.
- Existing training inputs and historical artifacts have unchanged hashes.
- The full CPU test suite passes and the four Stage A jobs were never controlled or modified.
