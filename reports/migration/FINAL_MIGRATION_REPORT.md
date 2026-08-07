# FINAL Migration Report — mRNA-EditFlow v3.1 → mRNA-XEditFlow v1.1

- **Migration Goal:** `GOAL-XEDITFLOW-MIGRATION-01`
- **Terminal state:** `BLOCKED_WITH_EVIDENCE`
- **UTC:** 2026-08-07 (updated from 2026-08-06 M0-M3 layer)
- **Migration worktree:** `/home/cunyuliu/mrna_editflow_goal/worktrees/xeditflow_migration_20260806T024650Z`
- **Migration branch:** `xeditflow-migration-20260806T024650Z`
- **New authoritative contract:** `mrna_xeditflow_goal_v1_1` (supersedes `utr_editflow_goal_v3.1_benchmark_first`)

---

## 1. FACTS_FROM_REPO
- Old contract `docs/contracts/utr_editflow_goal_v3_1.md` preserved read-only, marked `HISTORICAL_SUPERSEDED_BY_MRNA_XEDITFLOW_V1_1`.
- New authority documents created: contract md, executable config, supersession record, decision log, crosswalk, claim/task/split/task-split/asset/benchmark registries.
- v3.1 schemas, data registry and governance files preserved unmodified (governance retention is the core migration principle).
- All effect-model/flow/transfer phases under the new contract have now been executed (see §9).

## 2. FACTS_FROM_CONTRACTS
- Migration prompt (`提示词/mrna 合同迁移.md`) mandates superseding only the top-level scientific line while retaining provenance/license/exposure/sealed/split/conservation/audit standards.
- Five conflict freezes locked: STOP (`HOLD_IDENTIFIABILITY_GATE`), unlabeled pretraining (`DISABLED_IN_PRIMARY_V1`), GSE246381 (`OPERATIONALLY_SEALED_RETROSPECTIVE_EXTERNAL`), edit budget `[1,3,5]/[10]`, indel (`substitution-only primary`).
- Allowed terminal states: `MIGRATION_READY_FOR_DATA_REBUILD` / `DATA_BENCHMARK_READY_FOR_EFFECT_MODEL` / `EFFECT_MODEL_GO` / `EFFECT_MODEL_NO_GO` / `FLOW_GUIDANCE_GO` / `FLOW_GUIDANCE_NO_GO_FALLBACK_TO_BENCHMARK_CRITIC` / `SEALED_FINAL_COMPLETE` / `BLOCKED_WITH_EVIDENCE`.

## 3. INFERENCES
- A1/A2/B1/B2/C/D is an orthogonal axis over the retained E/F/AUX/REFERENCE, never a replacement.
- mRNA-EditBench v2 binds 4 sub-benchmarks to v4 task/split registries and accepted assets; CDS-B1 is DORMANT.
- 5'UTR and 3'UTR pools are independent endpoint heads (no cross-region mixing); GSE246381 is a sealed external final candidate.

## 4. UNKNOWN_OR_BLOCKED
- **Sealed final (GSE246381):** NOT executed (one-time irreversible access preserved). E0-X ordinary internal test verdict was **NO_GO** because `macro_sign_accuracy 0.510 < 0.60` is capped by the class prior (~0.52) on from-sequence models under S4 transfer (a scientific ceiling, not an engineering bug). The sealed final is therefore expected to also be NO_GO under the frozen protocol.
- **Formal X0-X transfer gate:** BLOCKED on frozen sealed results (withheld per E0-X decision).
- **CDS-B1:** DORMANT_BLOCKED_ON_SEQUENCE — GSE207584 rebuild recovered label+family (100 proteins / 578 variants / 97 rankable families / 6,936 observations) but sequence recovery is BLOCKED (100/100 proteins share one FASTA sequence set; reference FASTA does not distinguish codon-scheme groups). Per-variant synonymous sequences not recoverable from provided files.

## 5. FILES_READ
- Migration prompt, old/new local contracts, v3_1 schemas/registry, M0-M3 reports, all xeditflow_* registries, per-phase gate reports (B0-X, M4, O0-X, F0-X, G0-X, G1-X, E0-X, X0-X, X0-X CDS-B1 rebuild audit).

## 6. FILES_CHANGED
- Migration authority layer (M0-M3): see `artifacts/migration/FINAL_MIGRATION_MANIFEST.json`.
- Effect-model/flow/transfer phases: per-phase commits (B0-X, M4, O0-X, F0-X, G0-X, G1-X, E0-X, X0-X scaffolding, X0-X CDS-B1 rebuild audit), each with code + tests + gate report.

## 7. COMMANDS_RUN
- Per-phase pytest runs on server editflow env; the migration suite is at **224/224 PASS** (includes X0-X 33 + CDS-B1 rebuild audit 11). SHA256 checks for emitted canonical artifacts pass (`sha256sum -c` 5/5 for the CDS-B1 rebuild audit).

## 8. TEST_RESULTS
- `tests/migration/` full suite: **224/224 PASS** (editflow env).
- X0-X: 33/33; CDS-B1 rebuild audit: 11/11.

## 9. DATA_COUNTS_AND_DENOMINATORS
- Data rebuild gate: `DATA_BENCHMARK_READY_FOR_EFFECT_MODEL` (3U-A1 candidate pool 7/7; D1 SHA256SUMS 14/14).
- B0-X: effect dataset 106,659 records / 103,199 delta (SHA-256 `f23a9fdd…`); 12 effect + 3 search baselines; S4 leave-one-study-out macro; gate `B0X_EFFECT_BASELINE_CEILING_ESTABLISHED`.
- M4: SparseEditFormer trained (A2 dense → A1 natural → calibration → freeze); prereg thresholds macro delta Spearman ≥0.25 / sign acc ≥0.60 / top-10% enrichment ≥1.50.
- O0-X: `O0X_SEARCH_CEILING_ESTABLISHED` / `FLOW_HEADROOM_LIMITED`.
- F0-X: source-anchored legal Edit Flow; legality=100%, length=100%, budget violation=0, reproducibility=100%; gate `F0X_LEGAL_EDIT_FLOW_BASE_ESTABLISHED`.
- G0-X: exact density-ratio guidance (time-inhomogeneous budgeted Doob h-transform); target-rate rel err 1.77e-16, terminal TV 8.55e-17; gate `G0X_EXACT_GUIDANCE_THEORY_VERIFIED`.
- G1-X: real-mRNA guidance integration on measured pools; generation-quality axis `rate_cfg` beats no-guidance (~18x mean_delta, ~5x frac_beneficial); ranking axis headroom limited; gate `G1X_REAL_MRNA_GUIDANCE_VALUE_DEMONSTRATED` + `RANKING_AXIS_HEADROOM_LIMITED`.
- E0-X: prereg frozen (`E0X_PREREG_20260807`); ordinary internal test NO_GO on sign_accuracy (0.510<0.60, class-prior-capped); `E0X_PREREG_FROZEN` + `SEALED_FINAL_NOT_EXECUTED`.
- X0-X: development scaffolding (`X0X_DEV_SCAFFOLDING_READY`); CDS-B1 rebuild audit (`X0X_CDS_B1_REBUILD_AUDIT_COMPLETE`; CDS-B1 DORMANT_BLOCKED_ON_SEQUENCE).

## 10. REUSE_DECISIONS
- Raw/provenance/hash: REUSE_AS_IS. Schemas/registry: REUSE_WITH_ADAPTER or versioned REBUILD. Governance/provenance/license/exposure/sealed/split/audit: RETAIN_EXACT (hash-bound). Model-engine: AUDIT_AND_REBIND_IN_F0_X. CDS codon machinery: REBUILD (pure, self-contained in scripts/x0x/codon.py).

## 11. GATE_STATUS
- Migration authority layer (M0-M3): PASS — `MIGRATION_READY_FOR_DATA_REBUILD`.
- Effect-model/flow/transfer execution (B0-X → M4 → O0-X → F0-X → G0-X → G1-X → E0-X → X0-X): all development phases executed and accepted with evidence.
- **Current terminal state: `BLOCKED_WITH_EVIDENCE`.** All achievable development and evaluation under the frozen protocol is complete. The final GO/NO-GO declaration and the formal X0-X transfer gate remain blocked on the one-time sealed-final (GSE246381) decision, and on CDS-B1 per-variant synonymous sequence recovery. No fabricated PASS is claimed.

## 12. CLAIMS_UNLOCKED
- None beyond documented per-phase development results. Migration supersedes the scientific line but unlocks no new experimental claim beyond what each accepted gate records (all with `predicted/internal proxy` qualifiers where relevant; CDS-B1 has no measured claim).

## 13. CLAIMS_STILL_PROHIBITED
- L4 (real biological/therapeutic improvement) PROHIBITED. No wet lab. CDS-B1 not auto-unlocked (DORMANT_BLOCKED_ON_SEQUENCE). Old PASS not auto-inherited. GSE246381 sealed labels not accessed.

## 14. NEXT_PHASE_INPUTS
- **Sealed-final decision (user):** consume the one-time GSE246381 access (freeze results → unblock formal X0-X gate, expected NO_GO on sign_accuracy under frozen protocol) OR preserve it and finalize the migration on the `EFFECT_MODEL fallback` terminal evaluation.
- CDS-B1 into B1 requires external per-variant synonymous sequence tables (or original per-variant files) to supplement the family/label canonical; otherwise remains DORMANT_BLOCKED_ON_SEQUENCE.
- 3'UTR adapter training on 3U-A1 and formal X0-X transfer remain gated on frozen sealed results.

## 15. COMMIT_SHA
- M0 `caa30bb` · M1 `722935d`/`633f7e8` · M2 `0895ece`/`2641661` · M3 `187c95f`/`d09a621` · B0-X `bbdb6a7` · M4 (separate) · O0-X (separate) · F0-X `e394a6e` · G0-X `0673160`/`8da2e9a` · G1-X `a4d7c8e`/`5552d60` · E0-X `841f2dc` · X0-X scaffolding `1706e71` · X0-X CDS-B1 rebuild audit `5f3b4fb`.

## 16. MANIFEST_AND_HASHES
- `artifacts/migration/FINAL_MIGRATION_MANIFEST.json` and `FINAL_MIGRATION_SHA256SUMS` (see files; updated terminal_state to BLOCKED_WITH_EVIDENCE).
