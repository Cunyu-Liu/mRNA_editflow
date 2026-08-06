# Data Rebuild Gate — mRNA-XEditFlow v1.1

- **Migration Goal:** `GOAL-XEDITFLOW-MIGRATION-01`
- **Prior terminal state:** `MIGRATION_READY_FOR_DATA_REBUILD`
- **Gate outcome:** `DATA_BENCHMARK_READY_FOR_EFFECT_MODEL`
- **UTC:** 2026-08-06
- **Worktree:** `xeditflow_migration_20260806T024650Z`
- **Rebuild canonical:** `/mnt/cunyuliu/mrna_editflow_v3_1/d1_3u_rebuild_staging/ordinary`

---

## 1. FACTS_FROM_REPO
- Migration authority (M0-M3) terminal `MIGRATION_READY_FOR_DATA_REBUILD`; all registries (benchmark/task/split/asset-role) committed on branch `xeditflow-migration-20260806T024650Z`.
- Benchmarks: `EditBench-5U-A1-Natural` (ACTIVE), `EditBench-3U-A1-Variant` (ACTIVE), `EditBench-5U-A2-Dense` (DORMANT), `EditBench-CDS-B1-Synonymous` (DORMANT).

## 2. FACTS_FROM_CONTRACTS
- New authority contract `mrna_xeditflow_goal_v1_1`; five conflict freezes locked (STOP, unlabeled pretraining, GSE246381, edit budget `[1,3,5]/[10]`, indel substitution-only).
- Data-rebuild gate requires rebuilt qualified measured canonical pools under the `xedit_v1_1` orthogonal axes and passing benchmark ingestion/readiness gates before claiming `DATA_BENCHMARK_READY_FOR_EFFECT_MODEL`.

## 3. INFERENCES
- 3U-A1 gap assets (GSE232571/GSE261709/GSE298114) were reconstructed from GEO/raw provenance and joined into D1 canonical, closing the 4/7 → 7/7 gap.
- Both ACTIVE effect-model benchmarks now carry qualified measured source/candidate pairs with independent endpoint heads and no cross-region mixing.

## 4. UNKNOWN_OR_BLOCKED
- `EditBench-5U-A2-Dense` stays DORMANT: no qualified 5'UTR A2 dense asset remains post region-correction (GSE330741 = non-variant in-vivo localization MPRA). Not fabricated.
- `EditBench-CDS-B1-Synonymous` stays DORMANT: GSE207584 is a PENDING_BLOCKED legacy CDS liability until sequence/family/label rebuild. Not auto-unlocked.
- GSE246381 remains sealed external final candidate; no row-level labels accessed before frozen evaluator.

## 5. FILES_READ
- `xeditflow_benchmark_registry.yaml`, `xeditflow_asset_role_assignment.yaml`, `M4_DATA_READINESS.json`, `build_v3_1_technical_canonical.py`, `validate_v3_1_technical_canonical.py`, migration final report/registries.

## 6. FILES_CHANGED
- `scripts/d1_3u_rebuild_finalize.py` (new), `tests/migration/test_d1_3u_rebuild_finalize.py` (new), `scripts/m4_data_readiness_audit.py` (D1_PAIRS → rebuild staging), `M4_DATA_READINESS.json` (rebuilt).
- Main repo: `d1_staging/scripts/d1/build_canonical_records.py` (+3 extractors), `reconstruct_gse232571_sequences.py`, `reconstruct_gse261709_sequences.py`, `reconstruct_gse298114_sequences.py`.
- Rebuild artifacts on `/mnt`: `D1_CANONICAL_MANIFEST.json`, `D1_SHA256SUMS`, all `<artifact>.jsonl`.

## 7. COMMANDS_RUN
- `build_v3_1_technical_canonical.py` (rebuild staging), `validate_v3_1_technical_canonical.py` (0 errors), `d1_3u_rebuild_finalize.py` (manifest+SHA), `m4_data_readiness_audit.py`, `pytest tests/migration/`.

## 8. TEST_RESULTS
- `tests/migration/`: **61/61 PASS** (M1 15 + M2 21 + M3 11 + M4 data-readiness + finalize).
- `D1_SHA256SUMS -c`: 14/14 OK.

## 9. DATA_COUNTS_AND_DENOMINATORS
- Combined canonical records: 1,166,777. `utr_edit_pairs`: 103,694. `functional_observations`: 3,836,226. `sequence_entities`: 1,259,060.
- `EditBench-5U-A1-Natural`: GSE114002=55184, GSE217518=3564 → **2/2 ready**.
- `EditBench-3U-A1-Variant`: ENCSR854RUF=11969, GSE186455=649, GSE200304=6885, GSE232571=14503, GSE232572=9343, GSE261709=749, GSE298114=400 → **7/7 ready**.
- `alignment_ok=true`; `d1_pairs_total` (scanned) = 103,694.

## 10. REUSE_DECISIONS
- Raw/provenance/hash: REUSE_AS_IS. Existing D1 rebuild pipeline: REUSE_WITH_ADAPTER (new extractors). Legacy `/mnt/.../d1/` canonical: RETAIN (immutable, not overwritten). New rebuild: new versioned namespace `d1_3u_rebuild_staging`.

## 11. GATE_STATUS
- **PASS → `DATA_BENCHMARK_READY_FOR_EFFECT_MODEL`.** Both ACTIVE effect-model benchmarks (5U-A1, 3U-A1) have qualified measured D1 pairs; alignment invariants hold. DORMANT benchmarks are honest non-PASS, not fabricated.

## 12. CLAIMS_UNLOCKED
- Effect-model execution on `EditBench-5U-A1-Natural` and `EditBench-3U-A1-Variant` (T5_SOURCE_RELATIVE_EFFECT, T5_SELECTIVE_EFFECT, T5_MEASURED_NEIGHBORHOOD_OPTIMIZATION, T5_FIXED_BUDGET_MULTI_STEP_OPTIMIZATION; 3U-A1: T3_EFFECT_TRANSFER, CROSS_REGION_TRANSFER).

## 13. CLAIMS_STILL_PROHIBITED
- L4 real biological/therapeutic improvement PROHIBITED. CDS-B1 / 5U-A2 not auto-unlocked. Old v3.1 PASS not auto-inherited. No wet lab. GSE246381 final labels not opened.

## 14. NEXT_PHASE_INPUTS
- **B0-X**: effect baseline ceiling on accepted EFFECT_PRIMARY assets (5U-A1, 3U-A1) with matched records/splits/endpoint/budget. Then M4 SparseEditFormer → O0-X → F0-X → G0-X → G1-X → E0-X → X0-X.

## 15. COMMIT_SHA
- Main repo: `2420dd8` (extractors + reconstruction). Worktree: `5924cd8` (M4 path), pending data-rebuild-gate commit.

## 16. MANIFEST_AND_HASHES
- `D1_CANONICAL_MANIFEST.json` + `D1_SHA256SUMS` (14 artifacts) in rebuild staging; `M4_DATA_READINESS.json` in `artifacts/migration/`.
