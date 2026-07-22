# P2-06: Innovation 2 Full-mRNA Validation — Status (Degraded)

**Status**: DEGRADED — P2-01 BORDERLINE verdict (d=+0.371, p<1e-29) downgrades
P2-06 to a **methodology contribution only**. Full biological claim of
"cross-region synergy via counterfactual RL" requires P2-01 GO (d>0.5, p<0.001),
which was not achieved.
**Date**: 2026-07-20

## Background

### Innovation 2 (counterfactual cross-region synergy RL)

`rl/synergy.py` implements `SynergyREINFORCE` — a REINFORCE variant that
shapes the reward via 4 counterfactual rollouts per edit step:

```
synergy_reward = Δ_joint − (Δ_5 + Δ_c + Δ_3)
```

where `Δ_joint` is the oracle gain from editing all three regions jointly,
and `Δ_5`/`Δ_c`/`Δ_3` are the gains from editing each region alone. The
lambda schedule (`LambdaSchedule`) anneals the synergy reward weight over
training.

### P2-01 verdict (cross-region synergy verification)

Source: `docs/cross_region_synergy_finding_v2.md` (FROZEN, SHA-256 locked)

| Metric | Value | Threshold | Result |
|--------|-------|-----------|--------|
| Cohen's d | +0.371 | > 0.5 (GO) | BORDERLINE (0.2, 0.5) |
| p-value | < 1e-29 | < 0.001 | PASS |
| Oracle | MultiRegionOracle v2 (CNN-50mer + CAI + stability + coupling) | independent | OK |

**Verdict: BORDERLINE** — cross-region synergy is statistically significant
but the effect size is moderate. The coupling terms in the oracle are
non-additive *by construction*, so the detected synergy could be partly an
artifact of the oracle design (see Limitations in the finding doc).

## P2-06 goal (original)

Validate Innovation 2 (`SynergyREINFORCE`) on **full-mRNA sequences** (not
the tiny MDP), using the MultiRegionOracle v2 as the reward signal, and
show that the synergy-shaped RL policy achieves higher `syn_sum` than:

1. Vanilla REINFORCE (no synergy shaping)
2. EMA-baseline REINFORCE
3. GRPO (P2-05, group-normalized)

with 10-seed paired significance test (family-cluster bootstrap CI).

## Degradation rationale

Because P2-01 is BORDERLINE (not GO):

1. **Biological claim weakened**: "cross-region synergy via counterfactual RL"
   cannot be claimed as a strong biological finding. The synergy signal
   exists (p<1e-29) but the effect size (d=0.371) is below the GO threshold
   (d>0.5).
2. **Methodology contribution still valid**: the *algorithm* (counterfactual
   rollout synergy shaping) is novel regardless of effect size. P2-06 can
   still demonstrate that the algorithm *runs* on full-mRNA sequences and
   *reduces variance* vs vanilla REINFORCE, even if the synergy gain is
   modest.
3. **Paper positioning**: per `docs/next_steps_sota_roadmap.md` Section 0.4,
   if multi-region oracle synergy is not significant (d>0.5), the project
   falls back to "壁垒 4 (RL algorithm innovation) + 壁垒 1 (regulatory-grade
   minimal-edit)" for Nature Methods / Nat Comm tier, dropping the
   "cross-region synergy" primary claim.

## What P2-06 would require (if prioritized)

1. **Fixed backbone checkpoint** from P2-02 (BLOCKED — P2-02 at step ~2160/10000).
2. **Full-mRNA Policy wrapper**: the existing `rl/policy.py::Policy` works
   on `MRNARecord` (5'UTR + CDS + 3'UTR), so no new wrapper is needed —
   but the `SynergyREINFORCE` trainer currently operates on `TinyMDP` and
   must be adapted to use the real `Policy` + `MultiRegionOracle`.
3. **Counterfactual rollout pipeline**: for each edit step, run 4 rollouts
   (joint, 5'UTR-only, CDS-only, 3'UTR-only) and compute `syn_sum`. This
   is 4× the compute of vanilla REINFORCE.
4. **10-seed paired significance test** vs vanilla REINFORCE / EMA-baseline /
   GRPO, with family-cluster bootstrap CI.
5. **Reward qualifier**: all results must use "predicted TE (internal proxy)"
   or "MultiRegionOracle v2 score" — NOT "TE" or "stability" unqualified,
   until P2-01 multi-region oracle is externally validated.

### Implementation estimate

- Full-mRNA SynergyREINFORCE trainer (~300 LOC): adapt `rl/synergy.py` to
  use real `Policy` + `MultiRegionOracle` instead of `TinyMDP`.
- Counterfactual rollout pipeline (~150 LOC): 4 rollouts per step, region
  masking.
- Training script with split contract (~200 LOC): `--train-idx/--val-idx/--test-idx`.
- Unit tests (~250 LOC): counterfactual rollout correctness, synergy reward
  computation, split contract.
- Total: ~900 LOC + tests.

## Decision: DEFER (degraded)

P2-06 is **degraded to methodology contribution** due to P2-01 BORDERLINE.
Full implementation is:

1. BLOCKED by P2-02 (needs fixed backbone checkpoint)
2. Lower ROI given BORDERLINE verdict (synergy claim weakened)
3. Substantial effort (~900 LOC + tests)

The existing `rl/synergy.py` (tiny-MDP validated, P1-12) + the P2-01
BORDERLINE finding document the methodology contribution sufficiently for
the current paper draft. Full-mRNA validation can be added in a revision
cycle if reviewer feedback requires it.

## Constraint compliance

| Constraint | Status |
|------------|--------|
| 不擅自终止任何运行中进程 | OK — no processes touched. |
| 不修改 v1 frozen namespace | OK — no changes under `data/reconstructed/p0_data_reconstruction_v1/`. |
| 所有新增训练接入 split contract | PENDING — training script not yet written. |
| "improves TE" 加 predicted/internal proxy 限定词 | OK — this doc uses "MultiRegionOracle v2 score" / "predicted TE (internal proxy)". |
| 10-seed paired significance test | N/A — deferred. |
| 所有新代码配套单元测试 | N/A — no new code (deferred). |

## Relationship to other tasks

- **P2-01**: BORDERLINE verdict is the root cause of P2-06 degradation.
- **P2-02**: blocks P2-06 (needs fixed backbone checkpoint).
- **P2-05**: GRPO pilot is the alternative RL algorithm validation; if GRPO
  shows strong gains, it partially compensates for the P2-06 degradation on
  the "RL algorithm innovation" axis (壁垒 4).
- **P2-09**: OOD robustness stress test would also use the synergy RL
  policy if P2-06 were complete; with P2-06 deferred, P2-09 will use the
  GRPO pilot policy (P2-05) instead.

## Next steps (if/when prioritized)

1. Wait for P2-02 to complete and lock the 10k checkpoint.
2. Implement full-mRNA `SynergyREINFORCE` trainer (adapt from `rl/synergy.py`).
3. Implement counterfactual rollout pipeline (4 rollouts per step).
4. Run 10-seed paired significance test vs vanilla REINFORCE / GRPO.
5. Report results with "MultiRegionOracle v2 score (internal proxy)"
   qualifier.
