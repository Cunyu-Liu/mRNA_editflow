# FINAL Migration Report — mRNA-EditFlow v3.1 → mRNA-XEditFlow v1.1

- **Migration Goal:** `GOAL-XEDITFLOW-MIGRATION-01`
- **Terminal state:** `MIGRATION_READY_FOR_DATA_REBUILD`
- **UTC:** 2026-08-06
- **Migration worktree:** `/home/cunyuliu/mrna_editflow_goal/worktrees/xeditflow_migration_20260806T024650Z`
- **Migration branch:** `xeditflow-migration-20260806T024650Z`
- **New authoritative contract:** `mrna_xeditflow_goal_v1_1` (supersedes `utr_editflow_goal_v3.1_benchmark_first`)

---

## 1. FACTS_FROM_REPO
- Old contract `docs/contracts/utr_editflow_goal_v3_1.md` preserved read-only, marked `HISTORICAL_SUPERSEDED_BY_MRNA_XEDITFLOW_V1_1`.
- New authority documents created: contract md, executable config, supersession record, decision log, crosswalk, claim/task/split/task-split/asset/benchmark registries.
- v3.1 schemas, data registry and governance files preserved unmodified (governance retention is the core migration principle).

## 2. FACTS_FROM_CONTRACTS
- Migration prompt (`提示词/mrna 合同迁移.md`) mandates superseding only the top-level scientific line while retaining provenance/license/exposure/sealed/split/conservation/audit standards.
- Five conflict freezes locked: STOP (`HOLD_IDENTIFIABILITY_GATE`), unlabeled pretraining (`DISABLED_IN_PRIMARY_V1`), GSE246381 (`OPERATIONALLY_SEALED_RETROSPECTIVE_EXTERNAL`), edit budget `[1,3,5]/[10]`, indel (`substitution-only primary`).

## 3. INFERENCES
- A1/A2/B1/B2/C/D is an orthogonal axis over the retained E/F/AUX/REFERENCE, never a replacement.
- mRNA-EditBench v2 binds 4 sub-benchmarks to v4 task/split registries and accepted assets; CDS-B1 is DORMANT (no qualified data).
- 5'UTR and 3'UTR pools are independent endpoint heads (no cross-region mixing); GSE246381 is a sealed external final candidate.

## 4. UNKNOWN_OR_BLOCKED
- 19 PENDING_BLOCKED assets carry explicit reasons (REACQUIRE_OR_LABEL_JOIN), not silent drops.
- B3 model-engine blocker recorded for core/ef0 model/sampler/exact_sampler/bregman (rebind in F0-X).
- CDS-B1 requires rebuilt sequence/family/label before activation.

## 5. FILES_READ
- Migration prompt, old/new local contracts, v3_1 schemas/registry, M0-M3 reports, all xeditflow_* registries.

## 6. FILES_CHANGED
- See `artifacts/migration/FINAL_MIGRATION_MANIFEST.json` (all migration authority artifacts + SHA256).

## 7. COMMANDS_RUN
- `scripts/m3_build_benchmark_registry.py`, `pytest tests/migration/`, `scripts/m3_finalize_migration.py`.

## 8. TEST_RESULTS
- `tests/migration/` full suite: **47/47 PASS** (M1 15 + M2 21 + M3 11, editflow env pytest 9.1.1).

## 9. DATA_COUNTS_AND_DENOMINATORS
- Sub-benchmarks: 4 (ACTIVE: 3 — EditBench-5U-A1-Natural, EditBench-5U-A2-Dense, EditBench-3U-A1-Variant; DORMANT: EditBench-CDS-B1-Synonymous).
- Assets classified: 33 (ACCEPTED 14 / PENDING_BLOCKED 19).
- Crosswalk rows: 19.
- Blocker rebind rows: 23.

## 10. REUSE_DECISIONS
- Raw/provenance/hash: REUSE_AS_IS. Schemas/registry: REUSE_WITH_ADAPTER or versioned REBUILD. Governance/provenance/license/exposure/sealed/split/audit: RETAIN_EXACT (hash-bound). Model-engine: AUDIT_AND_REBIND_IN_F0_X.

## 11. GATE_STATUS
- **Migration authority layer PASS**: `MIGRATION_READY_FOR_DATA_REBUILD`. The contract supersession, schema/benchmark registries, asset roles and crosswalk are complete and verified. The data rebuild under the new schema/benchmark (qualified measured candidate pools) is the next gate; CDS-B1 stays DORMANT until qualified data is rebuilt. Effect-model execution (B0-X → M4 → O0-X → F0-X → G0-X → G1-X → E0-X → X0-X) is a separate tracked thread.

## 12. CLAIMS_UNLOCKED
- None. Migration supersesses the scientific line but unlocks no new experimental claim.

## 13. CLAIMS_STILL_PROHIBITED
- L4 (real biological/therapeutic improvement) PROHIBITED. No wet lab. CDS-B1 not auto-unlocked. Old PASS not auto-inherited.

## 14. NEXT_PHASE_INPUTS
- **Data rebuild gate**: rebuild qualified measured canonical pools under xedit_v1_1 schema/orthogonal axes and pass benchmark ingestion gates before claiming DATA_BENCHMARK_READY_FOR_EFFECT_MODEL.
- Then B0-X effect baseline ceiling on accepted EFFECT_PRIMARY assets; M4 SparseEditFormer; O0-X measured-space optimization; F0-X source-anchored legal Edit Flow; G0-X exact-guidance toy graph; G1-X integration; E0-X prereg + sealed final; X0-X 3'UTR/CDS transfer.

## 15. COMMIT_SHA
- M0 `caa30bb` · M1 `722935d`/`633f7e8` · M2 `0895ece`/`2641661` · M3 `187c95f`/`d09a621`.

## 16. MANIFEST_AND_HASHES
- `artifacts/migration/FINAL_MIGRATION_MANIFEST.json` and `FINAL_MIGRATION_SHA256SUMS` (see files).
