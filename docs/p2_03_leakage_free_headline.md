# P2-03 Leakage-Free Headline Eval — Pre-Registration & Status

**Status**: PRE-REGISTRATION COMPLETE; execution BLOCKED on multi-seed ranker training (see §4).
**Date**: 2026-07-20
**Split**: `benchmark/dev/p0_data_reconstruction_v1/combined_family/` (frozen, paper-eligible, leakage-free)
**Hard-constraint compliance**: ✓ v1 frozen namespace untouched; ✓ paper mode enforced; ✓ no running process killed; ✓ all claims use "predicted/internal proxy" qualifier until P2-01 multi-region oracle validated (BORDERLINE verdict, see `docs/cross_region_synergy_finding_v2.md`).

---

## 1. Pre-registered endpoints

Following the spec "预注册 1 primary + ≤2 secondary endpoints，其余 FDR/Holm 校正":

### 1.1 Primary endpoint (1)
- **`delta_oracle_te_vs_source`** (candidate TE − paired source TE, predicted/internal proxy)
  - Direction: higher is better
  - Rationale: the central claim of mRNA-EditFlow is improving predicted TE via legal edits. This is the single confirmatory test.
  - Statistical test: **family-cluster paired bootstrap CI** (10 000 resamples, cluster = gene family), 95 % CI lower bound > 0 to claim improvement; paired permutation test vs. each baseline with Holm-Bonferroni correction across the 8 non-primary contrasts.

### 1.2 Secondary endpoints (2)
- **`legal_fraction`** (T1: valid full mRNA fraction)
  - Direction: higher is better
  - Rationale: a baseline that improves TE but violates legality is unusable; this is the correctness gate.
  - Statistical test: family-cluster bootstrap CI, 95 % CI lower bound ≥ 0.95 to claim "legal".
- **`mean_edit_distance`** (T5: paired nt edit distance)
  - Direction: lower is better (within budget)
  - Rationale: edits should be minimal (regulatory-grade); this is the efficiency gate.
  - Statistical test: family-cluster bootstrap CI, 95 % CI upper bound ≤ edit_budget (3) to claim "within budget".

### 1.3 Exploratory endpoints (FDR-controlled)
All other `METRIC_SPECS` from `eval/run_multiseed_benchmark.py` (11 metrics total):
`mean_oracle_te`, `mean_oracle_mrl`, `kmer_js`, `codon_usage_kl`, `mean_novelty`, `exact_source_match_fraction`, `mean_protein_identity`, `within_budget_fraction`, `mean_abs_length_error`, `reading_frame_intact_fraction`, `delta_oracle_te_vs_source` (already primary).

These are reported with **Benjamini-Hochberg FDR q ≤ 0.05** across the 11-metric family. No confirmatory claims are made on exploratory endpoints.

### 1.4 Pre-registration fingerprint
```
primary:    delta_oracle_te_vs_source (higher better, family-cluster bootstrap 95% CI > 0)
secondary1: legal_fraction            (higher better, family-cluster bootstrap 95% CI >= 0.95)
secondary2: mean_edit_distance        (lower better,  family-cluster bootstrap 95% CI <= 3.0)
exploratory: BH-FDR q<=0.05 across 11 METRIC_SPECS
split:      benchmark/dev/p0_data_reconstruction_v1/combined_family/ (frozen, paper_eligible=true)
seeds:      >=3 training seeds × 10 decoder seeds (target 30 runs/baseline)
baselines:  te_only, scalar, pareto, grpo, hardneg_v2, random, legal, local-search, codon-lattice DP
```

---

## 2. Baseline inventory

### 2.1 Ranker-based baselines (5) — use `eval/run_multiseed_benchmark.py`

| baseline | checkpoint | head | training seeds available |
|----------|------------|------|--------------------------|
| `te_only` | `ckpts/proposal_ranker_t5_mo_te_only_head256/proposal_ranker_best.pt` | 256 | **1** (seed undefined in ckpt name) |
| `scalar` | `ckpts/proposal_ranker_t5_mo_scalar_head256/proposal_ranker_best.pt` | 256 | **1** |
| `pareto` | `ckpts/proposal_ranker_t5_mo_pareto_head256/proposal_ranker_best.pt` | 256 | **1** |
| `grpo` | `ckpts/proposal_ranker_t5_mo_grpo_head256/proposal_ranker_best.pt` | 256 | **1** |
| `hardneg_v2` | `ckpts/proposal_ranker_t5_cascade_hardneg_teacher_head256/proposal_ranker_best.pt` | 256 | **1** |

### 2.2 No-training baselines (4) — standalone or `--checkpoint None`

| baseline | implementation | training seeds | notes |
|----------|----------------|----------------|-------|
| `random` | `run_multiseed_benchmark.py --checkpoint None` | ≥3 trivially (random seed) | no ranker; random legal proposals |
| `legal` | `run_multiseed_benchmark.py --checkpoint None` (legal-only filter) | ≥3 trivially | no optimization; just legal candidates |
| `local-search` | `baselines/utr_local_search.py` | ≥3 (beam seed) | UTR-only predictor-guided beam search |
| `codon-lattice DP` | `baselines/codon_lattice_dp.py` | ≥3 (DP tie-break seed) | CDS-only synonymous codon DP |

### 2.3 Inference config (all baselines)
- `InferenceMode`: `calibrated_marginal` (default, per project convention; CTMC path + validation calibrated)
- `DecoderConfig`: edit_budget=3, proposal_top_k=64, cascade_recall_top_k=64, proposal_temperature=1.0, guidance_scale=0.0
- Oracle: P1-04 cross-fitted ensemble (multi-region: TE+MRL+stability+CAI+MFE) — pending P2-01 BORDERLINE upgrade to GO before full trust; until then all TE claims labeled "predicted/internal proxy".

---

## 3. Split contract & leakage audit

**Split**: `benchmark/dev/p0_data_reconstruction_v1/combined_family/`
- `split_manifest.json`: `paper_eligible=true`, `block_reasons=[]`
- `leakage_report.json`: `exact_match_count=0`, `family_disjoint=true`, `near_neighbor_threshold_passed=true`
- Records: 148 843 total → train=119 075, val=14 884, test=14 884
- Records path: `data/reconstructed/p0_data_reconstruction_v1/combined/combined_model_view.records.jsonl` (v1 frozen, SHA-256 `9666bbc9...`)
- `test.idx` SHA-256: (frozen, see manifest)

**Paper-mode enforcement** (per `eval/artifact_contract.py`):
- `--run-mode paper` requires `--split-manifest`, `--split-role test`, `--train-idx`, `--val-idx`, `--test-idx`
- Output must be under `benchmark/paper/`
- `prepare_scientific_records` requires non-None `split_contract`
- `validate_output_namespace` enforces `benchmark/paper/` prefix

---

## 4. Blocker: single-seed ranker checkpoints

**Problem**: The user constraint (hard) requires "≥3 training seeds × 10 decoder seeds" with "family-cluster bootstrap CI" and explicitly states "不再用 decoder seeds 替代 training seeds". The 5 ranker-based baselines (te_only, scalar, pareto, grpo, hardneg_v2) each have only **1 training seed** checkpoint available. The 4 no-training baselines (random, legal, local-search, codon-lattice DP) can trivially provide ≥3 seeds.

**Impact**:
- No-training baselines (4/9): can run the full 3×10 protocol immediately.
- Ranker baselines (5/9): can run 1×10 (preliminary) but **cannot make confirmatory claims** until 2 additional training seeds are produced per baseline.

**Unblock plan**:
1. Train 2 additional seeds for each of the 5 ranker baselines (10 training runs total).
   - Each ranker is a small head (~256 dim) on top of the frozen Stage A backbone; training is expected to be fast (~30 min – 2 h per seed on a single A100, depending on data slice).
   - Required: `scripts/train_proposal_ranker.py --mode {te_only,scalar,pareto,grpo,hardneg_v2} --seed {1,2}` (or equivalent).
2. Once 3-seed checkpoints exist, re-run the 3×10 protocol for all 5 ranker baselines.
3. Until then, ranker baseline results are labeled **"preliminary, 1 training seed, pending multi-seed"** and are NOT used for the primary-endpoint confirmatory claim.

---

## 5. Execution plan

### Phase A (now — no-training baselines, full 3×10 protocol)
- [ ] `random` — 3 seeds × 10 decoder seeds = 30 runs, paper mode, frozen test split
- [ ] `legal` — 3 seeds × 10 decoder seeds = 30 runs, paper mode, frozen test split
- [ ] `local-search` — 3 seeds × 10 decoder seeds = 30 runs (UTR-only)
- [ ] `codon-lattice DP` — 3 seeds × 10 decoder seeds = 30 runs (CDS-only)

### Phase B (now — ranker baselines, preliminary 1×10 protocol)
- [ ] `te_only` — 1 seed × 10 decoder seeds = 10 runs (preliminary)
- [ ] `scalar` — 1 seed × 10 decoder seeds = 10 runs (preliminary)
- [ ] `pareto` — 1 seed × 10 decoder seeds = 10 runs (preliminary)
- [ ] `grpo` — 1 seed × 10 decoder seeds = 10 runs (preliminary)
- [ ] `hardneg_v2` — 1 seed × 10 decoder seeds = 10 runs (preliminary)

### Phase C (follow-up — multi-seed ranker training)
- [ ] Train ranker seeds 1, 2 for each of the 5 modes (10 training runs)
- [ ] Re-run Phase B with 3×10 protocol
- [ ] Replace preliminary results with final 3×10 results
- [ ] Freeze `benchmark/paper/leakage_free_headline.json` (SHA-256 locked)

### GPU allocation
- GPU 0: P2-02 recovery run (ongoing, ~14 h remaining) — DO NOT USE
- GPU 4: calibrate PID 2544995 — DO NOT USE (hard constraint)
- GPU 5: available (25 GB free) → Phase A no-training baselines
- GPU 6: available (38 GB free) → Phase B ranker baselines
- GPU 7: AVOID (MIG issue, PyTorch sees only 4.75 GB)

---

## 6. Output artifacts (pending)

| artifact | path | status |
|----------|------|--------|
| Pre-registration doc | `docs/p2_03_leakage_free_headline.md` | this file (pre-registration complete) |
| Headline JSON | `benchmark/paper/leakage_free_headline.json` | pending (Phase A + B results) |
| Per-baseline summaries | `benchmark/paper/leakage_free_headline/<baseline>/multiseed_summary.json` | pending |
| Family-cluster bootstrap CI | `benchmark/paper/leakage_free_headline/family_cluster_ci.json` | pending |
| Paired permutation tests | `benchmark/paper/leakage_free_headline/paired_tests.json` | pending |
| FDR-controlled exploratory | `benchmark/paper/leakage_free_headline/exploratory_fdr.json` | pending |

---

## 7. Statistical analysis plan (SAP)

### 7.1 Family-cluster bootstrap CI
For each baseline and each endpoint:
1. Group test records by gene family (from `cluster_assignments.json`).
2. Resample families with replacement (10 000 iterations).
3. Compute the metric on the resampled test set.
4. Report the 2.5th and 97.5th percentiles as the 95 % CI.

### 7.2 Paired permutation test (primary endpoint)
For each baseline pair (e.g., te_only vs. scalar):
1. For each test record, compute `delta = metric_baseline_A - metric_baseline_B`.
2. Permute the sign of each delta 2 000 times (per-record paired permutation).
3. Report the two-sided p-value.
4. Apply Holm-Bonferroni correction across the 8 non-primary contrasts (8 pairwise tests vs. the reference baseline).

### 7.3 FDR control (exploratory endpoints)
For the 11 exploratory metrics across all 9 baselines:
1. Compute p-values for each metric × baseline combination.
2. Apply Benjamini-Hochberg procedure at q ≤ 0.05.
3. Report adjusted p-values and significance flags.

### 7.4 Reference baseline
The reference baseline for pairwise comparisons is **`te_only`** (single-TE reward, the simplest ranker). All other baselines are compared against it. This matches the existing `eval_multiobjective_ranker_ablation_head256.sh` protocol.

---

## 8. Current status

- **Pre-registration**: COMPLETE (this doc)
- **Phase A (no-training baselines)**: NOT STARTED — launching next
- **Phase B (ranker baselines, 1×10)**: NOT STARTED — launching after Phase A
- **Phase C (multi-seed ranker)**: BLOCKED — requires 10 training runs (follow-up)
- **P2-02 recovery**: ongoing (step ~1700/10000, val_loss decreasing, preliminary GO)

## 9. Update log
- 2026-07-20: pre-registration doc created; blocker identified (single-seed ranker checkpoints); Phase A/B launch planned.
