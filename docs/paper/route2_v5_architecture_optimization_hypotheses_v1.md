# Route 2 V5 architecture-optimization hypotheses (v1, 2026-08-29)

**Status:** PREPARATION_ONLY — no V5 run authorized yet. V4.0.3 Critic controls
retry1 and SetFlow V4-S1 corrected screen are still running; this document
freezes the prospective hypotheses so the next family can be launched as soon
as they terminal, per the user's 2026-08-29 decision: both tracks optimized in
parallel, retraining only after the current runs finish.

## Evidence basis

1. V2/V3 history: full critic never beat the strongest baseline (0.1164 vs
   0.1317); edit-metadata control reached 0.1043; source-only MAE beat full.
2. V4 objective audit (2026-08-29): the effective objective ranks only
   different-source-group pairs
   (`different_source_group_pair_indices`), so the critic is never trained to
   discriminate candidates of the same source, while the evaluated
   task-macro Spearman pools all within-task pairs including same-source ones.
3. SetFlow V3: NLL improved with capacity (F2 2.068 → F3 2.050) while
   recovery/unique degraded (0.292→0.194 / 0.679→0.637): objective
   misalignment via mode concentration.
4. Literature: FlexFlow (arXiv 2606.10543) structured couplings + Dirichlet
   temperature; UTR-LM structure supervision; DDMut siamese antisymmetry;
   GLID2E / iterative distillation KL-clipped reward steering.

## Direction A — Critic within-source ranking objective (implemented)

New module `core/route2_xeditcritic_within_source_ranking_v5.py`:
`same_source_group_pair_indices` + `within_source_ranking_loss_v5`
(softplus ranking on same-source pairs, scale-matched to the V4 cross-source
term; optional target-gap weighting). Falsifiable hypothesis: adding this term
to the effective objective (at weight comparable to the existing pairwise
term) improves task-macro Spearman by forcing the candidate-content channel to
discriminate same-source candidates. Tests: 7/7 pass (CPU).

## Direction E — SetFlow inference-time temperature control (implemented)

New module `core/route2_xeditsetflow_temperature_control_v5.py`:
`temper_mode_prior_v5` (p^(1/T) renormalized), `scale_stop_rate_v5`
(multiplies only the STOP rate), `frozen_temperature_sweep_v5`
(5x5 grid, identity included). Zero training cost: applies to any terminal
checkpoint. Falsifiable hypothesis: the frozen sweep moves unique-candidate
rate and recovery in opposite monotone directions, identifying a
Pareto-better (temperature, stop_scale) than identity. Tests: 6/6 pass (CPU).

## Not yet implemented (decided, awaiting training windows)

- Direction B — Critic Δstructure features (ViennaRNA MFE/pairing
  differentials into the raw branch): needs feature cache build.
- Direction C — hard antisymmetric siamese readout: needs forward doubling.
- Direction D — SetFlow structured edit coupling (position×base empirical
  prior): needs retraining.

## Execution plan (approved by user 2026-08-29)

- Current V4.0.3 runs finish untouched (heartbeat monitors).
- After SetFlow S1 terminal: run the Direction E sweep on its checkpoints
  (inference only, GPU, outcome-free Validation cohort).
- After Critic controls retry1 + cross-root gate: launch the V5 Critic family
  with Direction A wired into the effective objective (and optionally B/C) in
  a new clean HEAD + independent roots; no gate/threshold changes without a
  new frozen protocol.
- Directions B/C/D follow the same prospective pattern.
