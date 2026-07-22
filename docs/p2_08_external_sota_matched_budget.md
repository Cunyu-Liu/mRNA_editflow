# P2-08: External SOTA Matched-Budget Head-to-Head — Status & Gap

**Status**: GAP DOCUMENTED, full head-to-head DEFERRED (low priority; primary
claim is NOT "beat external SOTA" — it's cross-region synergy + minimal-edit
+ RL innovation; existing audit sufficient for current paper draft).
**Date**: 2026-07-20

## Goal

Run each external SOTA method (LinearDesign, EnsembleDesign, codonGPT,
UTailoR, UTRGAN, Prot2RNA) with **matched computational budget** (same beam
size / iterations / samples) on the frozen combined_family test split, then
perform a 10-seed paired significance test vs MEF.

## What exists (P1 external SOTA audit)

Source: `docs/external_sota_real_run_audit.md`

| Model | Task family | Status | Outputs | Protocol fidelity | Sufficient |
|-------|-------------|--------|--------:|-------------------|:----------:|
| LinearDesign | cds_protein_conditioned | measured | 1024/1024 | unspecified | False |
| EnsembleDesign | cds_protein_conditioned | invalid | 1016/1024 | official_code_budgeted_beam100_iter3_runs1_vs_paper_default_beam200_iter30_runs20 | False |
| codonGPT | cds_protein_conditioned | measured | 1024/1024 | official_hf_pretrained_synonymous_masked_sampling | False |
| Prot2RNA | cds_protein_conditioned | missing | 0/1024 | None | None |
| UTailoR | utr5_only | measured | 315/1024 | official_public_code_and_weights_strict_25_100_nt_shared_public_subset_not_paper_dataset | False |
| UTRGAN | utr5_only | measured | 1024/1024 | official_code_paper_default_10000_steps | **True** |

Aggregate: 4/6 measured, 1 invalid, 1 missing. Only UTRGAN has sufficient
protocol fidelity. **"Ready for external SOTA claim: False"**.

## Gap: matched-budget protocol

The existing runs used each method's **default or paper-specified budget**,
not a **matched budget**. A matched-budget head-to-head requires:

1. **Define a common budget**: e.g., same total compute (FLOPs), same number
   of oracle calls, or same wall-clock. The project constraint is "predicted
   TE (internal proxy)" via Oracle #3, so the natural budget is **oracle
   calls** (each method gets the same number of Oracle #3 evaluations).
2. **Re-run each method under the matched budget**: LinearDesign with beam=
   matched value, UTRGAN with steps=matched value, codonGPT with samples=
   matched value, etc.
3. **Run on the frozen combined_family test split** (14884 sequences), not
   the head1024 slice.
4. **10-seed paired significance test** (family-cluster bootstrap CI) vs
   MEF on the frozen test split.
5. **Fix protocol fidelity gaps**: EnsembleDesign output coverage mismatch,
   Prot2RNA missing, UTailoR strict subset.

### Why this is deferred

1. **Primary claim is NOT "beat external SOTA"**: per
   `docs/next_steps_sota_roadmap.md` Section 0.1, MEF's differentiation is
   "minimal-edit + cross-region synergy + RL algorithm innovation", NOT
   de novo SOTA. The roadmap explicitly states "MEF 的目标不是在工程规模上
   追赶 GEMORNA/mRNA-GPT 的 de novo SOTA".
2. **The existing audit is honest**: the status JSONs record
   `paper_rl_protocol_reproduced: false` and protocol fidelity flags. This
   is sufficient for the current paper draft to report external baselines
   with appropriate caveats.
3. **The critical path is P2-02 → P2-05**: the Stage A recovery run (P2-02)
   and GRPO pilot (P2-05) are the algorithmic contributions. P2-08 is
   independent and lower priority.
4. **Substantial effort**: re-running 5 external methods under matched
   budget on 14884 test sequences × 10 seeds is a large compute and
   engineering task.

## Decision: DEFER

P2-08 is **low priority** (per user goal priority order: P2-08 after P2-07).
The critical path is P2-02 (recovery run) → P2-05 (GRPO pilot, blocked by
P2-02). P2-08 is independent and can be done later without affecting the
critical path.

**Rationale**: the project's primary claims (cross-region synergy,
minimal-edit auditability, GRPO pilot) do not require a matched-budget
external SOTA win. The existing external baseline runs (with protocol
fidelity caveats) are sufficient for the current paper draft. If reviewer
feedback requires matched-budget comparison, it can be added in a revision
cycle.

## Constraint compliance

| Constraint | Status |
|------------|--------|
| AiZynthFinder/DFT/SOTA in isolated venv | OK — existing external tools use isolated venvs (`external_tools/envs/codongpt/`, etc.). |
| 不擅自终止任何运行中进程 | OK — no processes touched. |
| 不删除/重命名 results/ 子目录 | OK — no directories touched. |
| 10-seed paired significance test | N/A — deferred. |
| "improves TE" 加 predicted/internal proxy 限定词 | OK — this doc uses "predicted TE (internal proxy)" where applicable. |

## Next steps (if/when prioritized)

1. Define matched-budget protocol (oracle-call budget as the common axis).
2. Re-run LinearDesign / UTRGAN / codonGPT / UTailoR under matched budget
   on the frozen combined_family test split.
3. Fix EnsembleDesign output coverage; obtain Prot2RNA checkpoint.
4. Run 10-seed paired family-cluster bootstrap CI vs MEF.
5. Update `docs/external_sota_real_run_audit.md` with matched-budget rows.
