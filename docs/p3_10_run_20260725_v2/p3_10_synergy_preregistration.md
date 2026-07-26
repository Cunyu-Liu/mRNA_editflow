# P3-10 Synergy Pre-Registration

> Pre-registered before running the analysis. Per constraint #17, no
> post-hoc model/seed/design changes based on test or wet-lab results.

## Analysis Plan

### P3-10A: 5'UTR–CDS Synergy

**Counterfactual arms** (200 sequences × 8 arms):

| Arm | Description | Scorer |
|---|---|---|
| WT | No edit (wild-type) | 5'UTR oracle |
| 5'UTR-only | Best single 5'UTR substitution (exhaustive search) | 5'UTR oracle |
| CDS-only | Best single CDS synonymous substitution (exhaustive search) | CAI delta |
| Joint | 5'UTR + CDS best edits combined | CombinedOracle (additive) |
| Matched random | 1 random 5'UTR + 1 random CDS edit | CombinedOracle |
| Shuffled joint | Same edit count, random positions | CombinedOracle |
| Additive reconstruction | Δ5'UTR + ΔCDS (no joint candidate scored) | sum of single-arm deltas |
| Joint policy | MEF policy decode (if checkpoint available) | policy + CombinedOracle |

**Scorers:**

- **5'UTR oracle**: P3-02 cross-fitted ensemble (seq_diff + seq_linear),
  trained on 5'UTR edit records only (4802 measured + 10K proxy).
- **CAI CDS scorer**: `ΔCDS = 0.5 × ΔCAI`, where CAI is the codon
  adaptation index (Sharp & Li 1987). Public, training-data-free,
  sequence-level heuristic. Non-zero for CDS synonymous edits by
  construction.
- **CombinedOracle**: `Δjoint = Δ5'UTR (oracle) + ΔCDS (CAI)`,
  quadrature uncertainty. Additive by construction.

**Synergy definition:**

```
synergy = Δjoint - Δ5'UTR - ΔCDS
```

**Statistical test:** OLS regression with interaction term:
```
delta = β0 + β1*has_5utr_edit + β2*has_cds_edit + β3*interaction + ε
```
where interaction = has_5utr_edit × has_cds_edit. β3 tests whether joint
editing differs from the additive sum of individual effects.

**Decision rule (pre-registered):**
- **GO**: β3 significant (|t| > 2), synergy > 0 on both training and
  independent oracle, multiple cargos, interpretable mechanism.
- **PARTIAL**: only computational oracle; OR effect small; OR single cargo.
- **NO-GO**: interaction unstable; OR explainable by single-region effect;
  OR experiment not supportive; OR 3'UTR reward hacking increases.

### P3-10B: 3'UTR Unlock Gate

**4 conditions** (all must pass to unlock):
1. Source-matched 3'UTR intervention labels exist in benchmark
2. 3'UTR delta Oracle passes independent test
3. Adversarial splicing/motif audit passes
4. At least one cargo has stable 3'UTR headroom

If any condition fails: `three_utr_status = locked`

### P3-10C: Full-Transcript MDP

Only unlocked if 3'UTR gate passes. Actions: STOP, 5'UTR_SUB,
CDS_SYNONYMOUS_SUB, 3'UTR_SUB. UTR indels remain locked.

### Mechanism Analysis

10 potential mediators assessed:
1. Start accessibility (GC proxy around AUG)
2. Kozak context (position -3 and +4)
3. uAUG/uORF count
4. Start-proximal codon structure (GC of first 5 codons)
5. Codon usage (CAI proxy)
6. Codon-pair context (rare pair fraction)
7. Global RNA structure (5'UTR GC content)
8. 3'UTR stability motifs (polyA signals)
9. RNA-binding protein motifs
10. Edit-order dependence (synergy estimate)

### Pre-registered Limitations

The P3-02 delta oracle was trained exclusively on 5'UTR edit records
(4802 measured + 10K proxy, all `edited_region = five_utr`). Feature
extraction operates on `five_utr` sequences only. Therefore:
- 5'UTR deltas come from a learned, position-aware oracle (P3-02).
- CDS deltas come from the CAI heuristic (sequence-level, public,
  training-data-free). Non-zero for CDS synonymous edits.
- Joint deltas use the CombinedOracle (additive: Δ5'UTR + ΔCDS), so
  predicted synergy ≈ 0 **by construction**. This is a known
  pre-registered limitation, not a post-hoc excuse.
- True biological super-additivity cannot be established without a
  learned joint oracle that processes 5'UTR and CDS features jointly
  (not additively). Such an oracle requires multi-region intervention
  data that is not currently available.
- 3'UTR editing remains locked (no intervention data).

### Pre-registered Verdict Logic

Given the additive CombinedOracle, predicted synergy ≈ 0 and β3 ≈ 0.
The pre-registered verdict is determined as follows:

- **GO criteria** (all PASS required): c1 independent consistency,
  c2 exceeds additive/random, c3 multiple cargos, c4 interpretable
  mechanism. All are PARTIAL (computational only, no wet-lab).
- **NO-GO criteria** (any PASS triggers NO-GO): n1 interaction unstable
  (FAIL — stable at zero), n2 single-region explains (FAIL — CDS
  contributes non-zero via CAI), n3 experiment unsupportive (N/A —
  no experiments, not PASS), n4 3'UTR reward hacking (FAIL — 3'UTR
  locked, no surface).
- **Verdict**: PARTIAL. The pre-registered PARTIAL criterion is met:
  "only computational oracle; effect small; single cargo". None of the
  NO-GO sub-criteria fire.

### Qualifiers

All predicted improvements use "predicted" / "internal proxy" qualifiers
per constraint #23. No test data enters training or oracle fitting (#6).
Paper mode fails closed (#7).
