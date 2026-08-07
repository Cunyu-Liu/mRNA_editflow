# X0-X Pure-Development Preparation — mRNA-XEditFlow v1.1

- **Migration Goal:** `GOAL-XEDITFLOW-MIGRATION-01`
- **Prior terminal state:** `E0X_PREREG_FROZEN` + `SEALED_FINAL_NOT_EXECUTED` (GSE246381 preserved)
- **Phase:** X0-X — 3'UTR & CDS transfer (**PURE DEVELOPMENT PREPARATION ONLY**; formal X0-X gate NOT triggered)
- **Gate outcome:** `X0X_DEV_SCAFFOLDING_READY` (development-only, no measured CDS B1 result, no sealed access, frozen 5' model untouched)
- **UTC:** 2026-08-07
- **Worktree branch:** `xeditflow-migration-20260806T024650Z`
- **New modules:** `scripts/x0x/codon.py`, `scripts/x0x/region.py`, `tests/migration/test_x0x.py` (22 tests)

---

## 1. FACTS_FROM_REPO
- X0-X (3'UTR & CDS transfer) has NO prior code in the worktree; this phase builds the development scaffolding only.
- Effect dataset `artifacts/b0x/effect_dataset.jsonl` (SHA-256 `f23a9fdd…`) contains ACTIVE `3U-A1` records: 42,962 delta-defined across 7 studies (ENCSR854RUF, GSE186455, GSE200304, GSE232571, GSE232572, GSE261709, GSE298114) with independent 3' endpoints (ep_Freq, ep_activity_alt_mean, ep_activity_HEK293_alt_mean, ep_log2fc, etc.).
- `EditBench-CDS-B1-Synonymous` is **DORMANT** in `docs/execution/xeditflow_benchmark_registry.yaml`: no qualified B1 data accepted; GSE207584 is `PENDING_BLOCKED` legacy CDS liability (must rebuild sequence/family/label before entering B1).
- M4 SparseEditFormer model (`scripts/m4_sparse/model.py`) currently uses a SINGLE shared mean/logvar/sign/rank head with study/endpoint/benchmark embeddings — i.e. it does NOT yet isolate 5' vs 3' endpoint heads. The X0-X 3'UTR adapter addresses exactly this.
- Contract §16 (X0-X) requires: 3'UTR independent endpoint heads; 3' mechanism adapter/coupling; study/context transfer; no 5' MRL / 3' stability mixing; CDS synonymous-codon state; atomic codon substitution; protein identity/frame/start/stop hard invariants; family/listwise metric; protein-family split; external codon baselines; GSE207584 only after sequence/family/label rebuild.

## 2. FACTS_FROM_CONTRACTS
- §16 (X0-X): "只有 5′primary 模型、threshold 和 sealed 结果冻结后执行" — the FORMAL X0-X gate is gated on frozen 5' primary model + threshold + sealed results, NONE of which are frozen yet (sealed final withheld).
- §16 CDS: state=synonymous codon; atomic codon substitution; protein identity/frame/start/stop hard invariant; family/listwise metric; protein-family split; external codon baselines; GSE207584 must complete sequence/family/label rebuild before entering B1.
- §16 3'UTR: independent endpoint heads; 3' mechanism adapter/coupling; study/context transfer; don't mix 5' MRL and 3' stability.
- §16: "不得用 secondary 结果反向调整 5′sealed 模型" — this scaffolding must not touch/regress the frozen 5' model.

## 3. INFERENCES
- Because the formal X0-X gate is blocked on frozen sealed results (withheld per E0-X decision), the correct forward move is **pure development preparation**: build the reusable, data-free design cores (codon-state machine + region adapter) with unit tests, without claiming any measured CDS/3'UTR transfer result. This is honest and does not fabricate a PASS.
- CDS-side: synonymous-codon machinery guarantees **protein identity by construction** (hard invariant), but does NOT by itself prove a measured expression/stability effect — that remains an empirical question once qualified B1 data exists. The code does not touch GSE207584.
- 3'UTR-side: a `RegionAdapter` with structurally independent 5'/3' mean/logvar/rank heads and a 3'-only endpoint embedding enforces the "no 5' MRL / 3' stability pooling" invariant at the architecture level (not via reward penalty).

## 4. UNKNOWN_OR_BLOCKED
- Measured CDS B1 effect data: **BLOCKED** — GSE207584 must complete sequence/family/label rebuild before it can be a B1 asset; no CDS B1 result is claimed.
- Formal X0-X transfer gate: **BLOCKED on frozen sealed results** (E0-X decision to preserve GSE246381). This scaffolding does not unblock the formal gate; it only prepares the development code.

## 5. FILES_READ
- `scripts/m4_sparse/model.py`, `scripts/m4_sparse/config.py`, `scripts/m4_sparse/dataset.py`, `scripts/m4_sparse/train.py`, `scripts/m4_sparse/run.py`, `scripts/b0x/config.py`, `docs/execution/xeditflow_benchmark_registry.yaml`, `docs/execution/xeditflow_split_registry.yaml`, `artifacts/b0x/effect_dataset.jsonl`, `reports/migration/E0X_PREREG_INTERNAL_GATE.md`, `reports/migration/G1X_REAL_MRNA_GUIDANCE_GATE.md`.

## 6. FILES_CHANGED
- `scripts/x0x/codon.py` (new): genetic code, `translate`, `synonymous_codons`, `build_synonymous_classes`, `CDSState`/`build_cds_state`, `CodonEdit`, `enumerate_synonymous_edits`, `apply_edit` (protein/frame/start/stop hard invariants), protein-family helpers (`protein_family_id`, `family_members`, `listwise_ndcg`, `macro_listwise_ndcg_by_family`).
- `scripts/x0x/region.py` (new): `RegionConfig`, `RegionAdapter` (independent 5'/3' effect heads + 3'-only endpoint embedding), `independent_endpoint_head_guard`, `build_region_config`.
- `tests/migration/test_x0x.py` (new): 22 unit tests (translate/invariants, synonymous edits, family listwise metric, region adapter head independence & routing).

## 7. COMMANDS_RUN
- Server (editflow env):
  ```
  python -m pytest tests/migration/test_x0x.py -q     # 22 passed
  python -m pytest tests/migration/ -q                # 213 passed (191 prior + 22 X0-X)
  ```

## 8. TEST_RESULTS
- `tests/migration/test_x0x.py`: **22/22 PASS**.
- `tests/migration/` full suite: **213/213 PASS** (no regression to prior 191).

## 9. DATA_COUNTS_AND_DENOMINATORS
- CDS: **no measured B1 data used** (DORMANT benchmark; GSE207584 PENDING_BLOCKED). Only pure codon-state machinery; no data denominator claimed.
- 3'UTR: effect dataset 3U-A1 = 42,962 delta-defined / 7 studies (observed, not re-derived here); the region adapter is development-only and not trained/validated on it in this phase.

## 10. REUSE_DECISIONS
- M4 SparseEditFormer backbone: `REUSE_WITH_ADAPTER` — shared source-cached encoder/cross-attention reused; 5'/3' effect heads isolated in a new `RegionAdapter` (frozen 5' critic not modified).
- Standard genetic code / codon invariants: `REBUILD` (pure, self-contained in `codon.py`; no legacy counterpart reused).
- `xeditflow_benchmark_registry.yaml` region/status classification: `REUSE_AS_IS` (3U-A1 ACTIVE, CDS-B1 DORMANT with PENDING_BLOCKED reason).

## 11. GATE_STATUS
- `X0X_DEV_SCAFFOLDING_READY` — development-only. The formal X0-X transfer gate is NOT triggered (blocked on frozen sealed results). No measured CDS/3'UTR transfer claim is made.

## 12. CLAIMS_UNLOCKED
- The synonymous-codon CDS state machine preserves protein identity/frame/start/stop by construction (verified by 22 unit tests).
- The 3'UTR region adapter structurally isolates 5' MRL and 3' stability endpoint heads (no pooling), satisfying the §16 architecture invariant.

## 13. CLAIMS_STILL_PROHIBITED
- Any claim that a synonymous edit has a measured expression/stability effect (requires qualified B1 data; GSE207584 not yet rebuilt).
- Any claim of a 5'→3'/CDS transfer result (formal X0-X gate not triggered).
- Any use of GSE246381 sealed labels (preserved; sealed final not executed).
- Any "improves TE/stability/expression" without a `predicted/internal proxy` qualifier (unchanged global constraint).

## 14. NEXT_PHASE_INPUTS
- The X0-X scaffolding is ready to be extended once the formal gate is unblocked: (a) CDS B1 requires GSE207584 sequence/family/label rebuild; (b) 3'UTR adapter can be trained on 3U-A1 once permitted; (c) sealed results still frozen-block the formal X0-X gate.

## 15. COMMIT_SHA
- See git log on branch `xeditflow-migration-20260806T024650Z` (X0-X code + tests + this report committed together).

## 16. MANIFEST_AND_HASHES
- Effect dataset SHA-256: `f23a9fdd54a8ead90dccf793a073ccc6ed804a9b760a266c77b02e1fb1007ba1` (unchanged).
- New X0-X modules + tests: `scripts/x0x/codon.py`, `scripts/x0x/region.py`, `tests/migration/test_x0x.py` (22 tests).
