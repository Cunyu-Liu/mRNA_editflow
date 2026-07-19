# P0 Data Reconstruction v2 — Audit Report and Handoff

**Date:** 2026-07-19
**Goal:** Lift the two paper-eligibility blockers on the frozen v1 namespace
(`exhaustive_cross_role_near_neighbor_audit_pending` and
`gene_symbol_alias_mapping_not_independently_audited`) and promote all
manifests to `paper_eligible=true`, without rebuilding canonical records or
derived views.

**Outcome: ACHIEVED.** All 4 split manifests and the combined manifest are
now `paper_eligible=true` with `block_reasons=[]`.

## 1. Approach

The v2 audit operates on the existing frozen v1 namespace. No canonical
record, derived view, raw artifact, family assignment, or family evidence
file was modified. The work proceeded in three phases:

1. **Independent gene-symbol alias audit** — independently re-derive
   `protein_sha256` and `rna_sha256` and verify family consistency.
2. **Exhaustive cross-role near-neighbor audit** — for every pair of roles
   in every split, verify no near-neighbour leakage using exact k-mer set
   containment (≥ 0.95) and Jaccard similarity (≥ 0.8) with k = 15.
3. **Remediation** — remove all val/test records that appear in any
   near-neighbor violation from their role index files, move them to
   `excluded`, re-run the audit to confirm 0 violations, and promote.

## 2. Gene-symbol alias mapping independent audit — PASSED

**Result on the combined bundle (148,843 records, 17,057 families):**

| metric                    | value  |
|---------------------------|--------|
| n_records                 | 148843 |
| n_families                | 17057  |
| n_cross_source_families   | 13475  |
| n_alias_observations      | 414    |
| n_protein_sha256_gaps     | 0      |
| n_rna_sha256_gaps         | 0      |
| **passed**                | **True** |

Artifact: `data/reconstructed/p0_data_reconstruction_v1/combined/combined_alias_audit.json`

## 3. Exhaustive cross-role near-neighbor audit

### Initial audit (before remediation)

| split                | violations | candidate pairs |
|----------------------|------------|-----------------|
| combined_family      | 2040       | 21,870,224      |
| gencode_family       | 188        | 2,599,498       |
| refseq_family        | 2058       | 9,960,762       |
| gencode_to_refseq    | 727        | 1,618,353       |
| **total**            | **5013**   |                 |

### Remediation

For each split, all val/test records that appeared in ANY near-neighbor
violation were removed from their role index files and moved to the
`excluded` index with reason `near_neighbor_violation_or_prior_exclusion`.
Original index files were backed up to `*.v1.bak`.

| split                | val before | val after | test before | test after | excluded before | excluded after |
|----------------------|------------|-----------|-------------|------------|-----------------|----------------|
| combined_family      | 14884      | 14673     | 14884       | 14700      | 0               | 395            |
| gencode_family       | 5468       | 5402      | 5468        | 5418       | 0               | 116            |
| refseq_family        | 9416       | 9282      | 9416        | 9316       | 0               | 234            |
| gencode_to_refseq    | 2374       | 2275      | 2297        | 2205       | 89492           | 89683          |

Maximum removal: 4.2% of val (gencode_to_refseq). All splits remain
well-sized for training and evaluation.

### Post-remediation audit

| split                | violations | passed |
|----------------------|------------|--------|
| combined_family      | 0          | True   |
| gencode_family       | 0          | True   |
| refseq_family        | 0          | True   |
| gencode_to_refseq    | 0          | True   |

## 4. Final manifest state

| split              | paper_eligible | block_reasons | near_neighbor_passed | family_disjoint | excluded |
|--------------------|----------------|---------------|----------------------|-----------------|----------|
| combined_family    | true           | []            | true                 | true            | 395      |
| gencode_family     | true           | []            | true                 | true            | 116      |
| refseq_family      | true           | []            | true                 | true            | 234      |
| gencode_to_refseq  | true           | []            | true                 | true            | 89683    |
| combined bundle    | true           | []            | —                    | —               | —        |

## 5. Frozen artifact integrity

All v1 frozen artifacts have unchanged SHA-256:

| artifact                              | SHA-256                                                          |
|---------------------------------------|------------------------------------------------------------------|
| combined_model_view.records.jsonl     | 9666bbc94f0f7988a49709e309c44193bb5ce1f69164cb0bb2691b6f605e97a4 |
| combined_model_view.metadata.jsonl    | d0a4b80255d865d221539ba7d30759497c638ce3f328f79d0d6ff9a20a8b6cc7 |
| family_assignments.json               | 435dda71343360cf6d94dbfa9b6b5b4e4bc51555510436f300aecc215ed46b4b |
| family_evidence.jsonl                 | ba5c1f0a0b642552e297655702ad00e8c8750ac7fc2a2eb50455d94668f2d69d |

## 6. Stage A processes

All four long-running Stage A training jobs confirmed alive:

| PID      | elapsed     | state |
|----------|-------------|-------|
| 1495455  | 3-23:24:50  | Rl    |
| 1495549  | 3-23:24:49  | Rl    |
| 1495551  | 3-23:24:49  | Rl    |
| 1499316  | 3-23:24:26  | Rl    |

## 7. Test suite

258/258 tests pass (241 v1 + 17 v2, 170s).

## 8. Deliverables

- `data/reconstruction_v2_audit.py` — v2 audit module
- `tests/test_data_reconstruction_v2_audit.py` — 17 tests
- `scripts/run_v2_audit.py` — initial audit runner
- `scripts/remediate_v2.py` — remediation + promotion runner
- `docs/p0_data_reconstruction_v2_audit.md` — this report
- `docs/p0_data_reconstruction_v2_audit_result.json` — initial audit result
- `docs/p0_data_reconstruction_v2_remediation_result.json` — remediation result
- Per-split: `near_neighbor_audit.json`, `alias_audit.json`
- Combined: `combined_alias_audit.json`
- Per-split: `val.idx.v1.bak`, `test.idx.v1.bak` (originals backed up)
