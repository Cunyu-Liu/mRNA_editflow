# X0-X CDS-B1 Rebuild Audit (GSE207584 iCodon) — mRNA-XEditFlow v1.1

- **Migration Goal:** `GOAL-XEDITFLOW-MIGRATION-01`
- **Prior terminal state:** `X0X_DEV_SCAFFOLDING_READY` (development-only; formal X0-X gate blocked on frozen sealed results)
- **Phase:** X0-X — 3'UTR & CDS transfer (**PURE DEVELOPMENT PREPARATION ONLY**; formal X0-X gate NOT triggered)
- **Gate outcome:** `X0X_CDS_B1_REBUILD_AUDIT_COMPLETE` (development-only; CDS-B1 stays **DORMANT_BLOCKED_ON_SEQUENCE**; no measured CDS B1 result, no sealed access, frozen 5' model untouched)
- **UTC:** 2026-08-07
- **Worktree branch:** `xeditflow-migration-20260806T024650Z`
- **New modules:** `scripts/x0x/cds_b1_rebuild_audit.py`, `tests/migration/test_cds_b1_rebuild.py` (11 tests)

---

## 1. FACTS_FROM_REPO
- `EditBench-CDS-B1-Synonymous` is **DORMANT** in `docs/execution/xeditflow_benchmark_registry.yaml`: no qualified B1 data accepted; GSE207584 is `PENDING_BLOCKED` legacy CDS liability (must rebuild sequence/family/label before entering B1).
- GSE207584 (Diez et al. 2022, Sci Rep 12:12126) is a synonymous-codon massive reporter library (iCodon zebrafish). Raw files are the perfect library CSV + reference FASTA under the v3.1 raw_view:
  - `.../raw_view/GSE207584/GSE207584_Zebrafish-library-perfect.csv.gz` (10,227 rows)
  - `.../raw_view/GSE207584/GSE207584_reference.fasta.gz` (1,395 sequences)
  - columns: `Name, Protein_id, Group, zf_library_2h_1..zf_library_8h_3` (3 timepoints x 3 replicates)
- Contract §16 (X0-X) CDS requires: state=synonymous codon; atomic codon substitution; protein identity/frame/start/stop hard invariants; family/listwise metric; protein-family split; external codon baselines; **GSE207584 only after sequence/family/label rebuild**.

## 2. FACTS_FROM_CONTRACTS
- §16 CDS: "GSE207584 只有完成 sequence/family/label 重建后才能进入 B1".
- §16: "不得用 secondary 结果反向调整 5′sealed 模型" — this audit must not touch/regress the frozen 5' model, and must not touch GSE246381 sealed labels.

## 3. INFERENCES
- The rebuild must be performed honestly: recover what IS recoverable (family structure + measured labels) and **prove with evidence** what is NOT recoverable (distinct per-variant synonymous sequences).
- Because the provided reference FASTA is keyed by construct *Name* and every group of a protein shares the same Name set, all codon-scheme groups collapse to the same underlying sequence set → distinct per-variant synonymous nucleotide sequences are **NOT recoverable** from the provided files. This is the sequence-recovery blocker that keeps CDS-B1 DORMANT.

## 4. UNKNOWN_OR_BLOCKED
- Distinct per-variant synonymous nucleotide sequences: **BLOCKED** (proved: 100/100 proteins share the same FASTA sequence set across their groups). Recovering them requires external codon-scheme tables or the original library's per-variant sequence files, which are not present.
- Formal X0-X transfer gate: **BLOCKED on frozen sealed results** (E0-X decision to preserve GSE246381). This audit does not unblock the formal gate.

## 5. FILES_READ
- GSE207584 perfect CSV + reference FASTA (server raw_view), `scripts/x0x/codon.py`, `docs/execution/xeditflow_benchmark_registry.yaml`, `reports/migration/X0X_DEV_SCAFFOLDING_GATE.md`.

## 6. FILES_CHANGED
- `scripts/x0x/cds_b1_rebuild_audit.py` (new): GSE207584 rebuild-audit — variant grouping (protein, group), timepoint/replicate label aggregation, sequence-recovery blocker proof, S7 protein-family-disjoint split, canonical emitters (group_registry / functional_observations / sequence_entities), SHA256SUMS + D1_CANONICAL_MANIFEST.
- `tests/migration/test_cds_b1_rebuild.py` (new): 11 unit tests (grouping, blocker proof, S7 split conservation, rankable flags, observation endpoints, sequence-anchor emission, aggregate means).
- `docs/execution/xeditflow_benchmark_registry.yaml` (updated): CDS-B1 `status_reason` + new `rebuild_audit` block recording the 2026-08-07 audit result.

## 7. COMMANDS_RUN
- Server (editflow env):
  ```
  python -m pytest tests/migration/test_x0x.py tests/migration/test_cds_b1_rebuild.py -q   # 33 passed
  python -m pytest tests/migration/ -q                                                       # 224 passed
  python -m scripts.x0x.cds_b1_rebuild_audit \
      --perfect <raw_view>/GSE207584_Zebrafish-library-perfect.csv.gz \
      --fasta  <raw_view>/GSE207584_reference.fasta.gz \
      --out-dir artifacts/x0x/cds_b1_rebuild_20260807 --seed 42 --split-train 0.70 --split-val 0.15
  sha256sum -c artifacts/x0x/cds_b1_rebuild_20260807/D1_SHA256SUMS   # all OK
  ```

## 8. TEST_RESULTS
- `tests/migration/test_cds_b1_rebuild.py`: **11/11 PASS**.
- `tests/migration/` full suite: **224/224 PASS** (no regression to prior 213).

## 9. DATA_COUNTS_AND_DENOMINATORS
- **Rebuilt (label + family only):** 100 proteins, 578 (protein, group) variants, **97 rankable families** (>=2 variants), 6,936 functional observations (aggregate + replicate endpoints).
- **Sequence:** 100/100 proteins BLOCKED for distinct per-variant synonymous sequences (`sequence_recovery=BLOCKED`). 1,395 FASTA sequences present but do not distinguish codon-scheme groups.
- **S7 split (seed 42, 0.70/0.15):** train=70, val=15, test=15 families (family-disjoint, conserved).
- Verdict: `REBUILD_PARTIAL_LABEL_FAMILY_ONLY`; `cds_b1_status=DORMANT_BLOCKED_ON_SEQUENCE`.

## 10. REUSE_DECISIONS
- `scripts/x0x/codon.py` (synonymous-codon machinery): `REUSE` — the rebuild audit reuses `translate`/`build_cds_state` to validate protein identity and emit the family-anchor CDS.
- GSE207584 legacy canonical (`v3.1_authority_rebind` run): `REBUILD` of family/label from raw CSV+FASTA with a fresh, honest blocker proof; no per-variant sequence fabricated.

## 11. GATE_STATUS
- `X0X_CDS_B1_REBUILD_AUDIT_COMPLETE` — development-only. CDS-B1 remains **DORMANT_BLOCKED_ON_SEQUENCE**. The formal X0-X transfer gate is NOT triggered (blocked on frozen sealed results). No measured CDS/3'UTR transfer claim is made.

## 12. CLAIMS_UNLOCKED
- GSE207584 family structure and measured expression labels (per protein x group, 3 timepoints x 3 replicates) are rebuilt as canonical artifacts with SHA-256 integrity (`artifacts/x0x/cds_b1_rebuild_20260807/`).
- The sequence-recovery blocker is **proved with evidence** (100/100 proteins share one FASTA sequence set per protein), so CDS-B1 stays honestly DORMANT rather than being force-unlocked without distinct per-variant sequences.

## 13. CLAIMS_STILL_PROHIBITED
- Any claim that a synonymous edit has a measured expression/stability effect (CDS-B1 remains DORMANT_BLOCKED_ON_SEQUENCE; requires distinct per-variant synonymous sequences + qualified B1 data).
- Any claim of a 5'→3'/CDS transfer result (formal X0-X gate not triggered).
- Any use of GSE246381 sealed labels (preserved; sealed final not executed).
- Any "improves TE/stability/expression" without a `predicted/internal proxy` qualifier (unchanged global constraint).

## 14. NEXT_PHASE_INPUTS
- CDS-B1 remains blocked on sequence recovery. To enter B1, GSE207584 would need external per-variant synonymous sequence tables (or the original library's per-variant files) to supplement the family/label canonical. 3'UTR adapter training on 3U-A1 and formal X0-X transfer remain gated on frozen sealed results.

## 15. COMMIT_SHA
- See git log on branch `xeditflow-migration-20260806T024650Z` (rebuild-audit code + tests + registry update + this report committed together).

## 16. MANIFEST_AND_HASHES
- Rebuild-audit artifacts SHA-256 (`artifacts/x0x/cds_b1_rebuild_20260807/D1_SHA256SUMS`):
  - `group_registry.jsonl` 82ad1fe3…
  - `functional_observations.jsonl` d30c4f89…
  - `sequence_entities.jsonl` 3668dc6d…
  - `protein_family_split.json` 6546c752…
  - `cds_b1_rebuild_audit.json` ce1f77f8…
- New code: `scripts/x0x/cds_b1_rebuild_audit.py`, `tests/migration/test_cds_b1_rebuild.py` (11 tests).
