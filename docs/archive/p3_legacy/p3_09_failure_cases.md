# P3-09 Failure Cases

> Per constraint #16, every failure observed during P3-09 is preserved here.
> Paper mode fails closed: a failure in any component does NOT silently pass.

## 1. Component-Level Crashes / Errors

- (none)

## 2. Reward-Hacking Cases (|effective − independent| > 0.05 AND effective > 0)

> Post-mitigation: when a guard triggers, the effective reward is clamped to 0,
> so guard-caught cases are no longer counted as active reward hacking.

- (none — all raw reward-hacking cases were mitigated by guard-based reward clamping)

### Mitigated by Guard (raw reward > 0 → effective = 0)

- **homopolymer_a**: raw=0.0200 → effective=0.0000 (guards: low_gc=0.00, polyA_signal=AAAAAA, homopolymer_A8)
- **extreme_cpg**: raw=0.1388 → effective=0.0000 (guards: high_gc=1.00, hairpin_pos0_stem5_loop3_mm0)
- **extreme_upa**: raw=0.0837 → effective=0.0000 (guards: cpa_extreme_pos0)
- **stable_hairpin**: raw=0.0667 → effective=0.0000 (guards: hairpin_pos0_stem4_loop8_mm1)

## 3. Guard-Evasion Cases (raw training reward > 0 AND no guard detected)

- (none — every adversarial sequence with positive raw reward was caught by ≥1 guard)

## 4. Oracle Disagreement > 0.1 (training vs independent)

- (none)

## 5. OOD Constraint Collapse (validity < 100%)

- (none — all OOD splits maintained 100% constraint validity)

## 6. Abstention Triggers Fired

- **cargo_family_ood** (n_abstained=24): no_positive_lcb=11, constraint_risk=24
- **rare_family** (n_abstained=24): no_positive_lcb=11, constraint_risk=24
- **length_shift** (n_abstained=4): no_positive_lcb=2, constraint_risk=4
- **gc_shift** (n_abstained=2): constraint_risk=2

## 7. External Baselines Crashed / Invalid

- **UTailoR**: literature-only — Adapter requires external executable / weights not present in this run.
- **UTRGAN**: literature-only — Adapter requires external executable / weights not present in this run.
- **LinearDesign**: literature-only — Adapter requires external executable / weights not present in this run.
- **EnsembleDesign**: literature-only — Adapter requires external executable / weights not present in this run.
- **codonGPT**: literature-only — Adapter requires external executable / weights not present in this run.
- **mRNA-GPT**: literature-only — Adapter not executable in this environment (no executable / weights / network).
- **ProMORNA**: literature-only — Adapter not executable in this environment (no executable / weights / network).
- **mRNAutilus**: literature-only — Adapter not executable in this environment (no executable / weights / network).
- **GEMORNA**: literature-only — Adapter not executable in this environment (no executable / weights / network).

## 8. MEF Policy Decode Errors

- (none — all policy decodes succeeded)

## 9. Infrastructure Availability

- P3-08 policy checkpoint loaded: **yes**
- Independent oracles built: **4** (absolute, difference, siamese, edit_conditioned)
- Test sources loaded: **24**
- Train sources loaded: **24**