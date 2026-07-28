# P3-10 Full-Transcript Decision

> Created: 2026-07-25T14:16:38Z

## Verdict: **PARTIAL**

PARTIAL for full-transcript extension. The CombinedOracle (5'UTR P3-02 oracle + CAI CDS scorer) produces non-zero deltas for both 5'UTR and CDS edits, so single-region (5'UTR-only) does NOT fully explain the joint prediction. The joint arm exceeds each single-region arm in magnitude. However, the combined oracle is additive by construction, so predicted synergy ≈ 0 and β3 ≈ 0; true biological super-additivity cannot be established without a learned joint oracle. 3'UTR editing remains locked (no intervention data), and no wet-lab joint editing experiments are available. This matches the pre-registered PARTIAL criterion: 'only computational oracle; or effect small; or single cargo'. The primary 5'UTR paper story is preserved per the P3-00A frozen contract: cross_region_synergy is a conditional extension, not a first-paper blocker.

## GO Criteria

- **c1_independent_consistency**: PARTIAL — 5'UTR deltas are direction-consistent on training and independent oracles (per P3-09 transfer matrix). CDS deltas use the CAI heuristic, which is a public sequence-level signal (not a learned oracle) and therefore has no separate independent oracle. The criterion is PARTIAL: computational direction-consistency holds for 5'UTR, CAI provides a sequence-grounded CDS signal, but no wet-lab joint experiment is available.
- **c2_exceeds_additive_random**: PARTIAL — Synergy mean = 0.000000, std = 0.000000; β3 = -0.000000, t = -0.00. The CombinedOracle is additive by construction, so synergy ≈ 0 in the predicted space. However, joint mean (0.2843) > 5'UTR-only mean (0.1968) and > CDS-only mean (0.0875), so joint editing does exceed each single-region arm in magnitude. The criterion is PARTIAL: joint exceeds single-region arms but does not exceed the additive reconstruction (by oracle design).
- **c3_multiple_cargos**: PARTIAL — The 5'UTR oracle transfers across cargos (per P3-09 transfer matrix), and the CAI scorer is cargo-agnostic by construction. However, the current benchmark provides only 5'UTR data on a single cargo panel; multi-cargo validation requires additional datasets. The criterion is PARTIAL: mechanistic transfer is expected but not yet empirically demonstrated across cargos.
- **c4_interpretable_mechanism**: PARTIAL — 7 of 10 mechanisms assessed with sequence heuristics (start accessibility, Kozak, uORF, codon usage, codon-pair, global GC, RBP motifs). Edit-order dependence requires a learned joint oracle (not available). 3'UTR stability motifs use an inert placeholder 3'UTR. The criterion is PARTIAL: mechanisms are interpretable but not fully assessable.

## NO-GO Criteria

- **n1_interaction_unstable**: FAIL — Interaction is stable at zero (synergy std small), not unstable. The additive CombinedOracle produces a deterministic, reproducible synergy estimate. NO-GO sub-criterion does NOT fire.
- **n2_single_region_explains**: FAIL — Single-region (5'UTR-only) does NOT fully explain the joint prediction. CDS-only mean = 0.087500 (CAI delta, non-zero), and joint mean = 0.284277 = 5'UTR + CDS by construction. Both regions contribute non-zero predicted deltas. NO-GO sub-criterion does NOT fire.
- **n3_experiment_unsupportive**: N/A — No wet-lab joint editing experiments are available. N/A is not PASS — the absence of disconfirming evidence is not evidence of NO-GO. NO-GO sub-criterion does NOT fire.
- **n4_3utr_reward_hacking**: FAIL — 3'UTR editing is locked (no 3'UTR intervention data, no 3'UTR MDP action). There is no reward-hacking surface. NO-GO sub-criterion does NOT fire.

## Synergy Verdict

- **computational_synergy**: PARTIAL — additive CombinedOracle (5'UTR oracle + CAI); predicted synergy ≈ 0 by construction; non-zero CDS signal via CAI
- **wet_lab_synergy**: NOT_EVALUATED — no joint editing experiments
- **three_utr_extension**: LOCKED — no 3'UTR intervention data
- **full_transcript_mdp**: LOCKED — requires 3'UTR unlock + learned joint oracle

## 3'UTR Status: **locked**


## Paper Implication

Per P3-00A frozen contract, the primary paper uses Task A (five_utr_minimal_substitution). Cross-region synergy and full-transcript editing are conditional extensions. The PARTIAL verdict is consistent with the contract: 'cross_region_synergy' is listed under 'not_required_for_first_paper'. The 5'UTR-only P3-08 GRPO policy remains the primary paper result; the CAI-based CDS scorer provides a defensible CDS signal for future work.


## Future Work

- Collect source-matched 3'UTR intervention labels to enable 3'UTR oracle
- Collect CDS synonymous edit measurements to enable learned joint 5'UTR×CDS oracle
- Build a learned joint oracle that processes 5'UTR and CDS features jointly (not additively)
- Re-run P3-10 once joint oracle and multi-region data are available
- Validate CAI-based CDS predictions against wet-lab CDS synonymous edit measurements