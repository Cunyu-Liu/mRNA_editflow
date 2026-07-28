# P3-09 Adversarial Audit & Reward Hacking Report

> All rewards reported here are **predicted** by the internal P3-02 oracle ensemble
> and its independent-architecture variants. They are NOT wet-lab measurements.
> Guard detection uses simple motif heuristics and is not exhaustive.

- Adversarial sequences audited: **13**
- Reward-hacking cases (post-mitigation): **0**
- Guard-evasion cases (training reward > 0 AND no guard detected): **0**
- Cases mitigated by guard (raw reward > 0, clamped to 0): **4**
- Wall clock: 0.6s

## Mitigation Mechanism

When any guard triggers, the **effective reward** is clamped to zero.
The raw training reward is preserved for transparency. Reward-hacking
is evaluated on the *effective* reward, so guard-caught cases are not
counted as active hacking.

## Audit Table

| Adversarial | Raw training reward | Effective reward | Independent mean | Guard | uAUG | GC | On-manifold |
|---|---|---|---|---|---|---|---|
| extreme_high_gc | -0.3107 | 0.0000 | 0.0005 | yes | 0 | 1.00 | 28.0 |
| extreme_low_gc | -0.4940 | 0.0000 | 0.0004 | yes | 0 | 0.00 | 35.0 |
| homopolymer_a | 0.0200 | 0.0000 | 0.0007 | yes | 0 | 0.00 | 34.0 |
| homopolymer_g | -1.8487 | 0.0000 | 0.0005 | yes | 0 | 1.00 | 23.0 |
| repeated_aug | -2.4545 | 0.0000 | 0.0003 | yes | 16 | 0.32 | 33.0 |
| repeated_uorf | -0.0330 | 0.0000 | 0.0005 | yes | 6 | 0.12 | 32.0 |
| cryptic_splice | -0.9308 | 0.0000 | 0.0005 | yes | 0 | 0.32 | 32.0 |
| polyadenylation | -1.4608 | 0.0000 | 0.0007 | yes | 0 | 0.00 | 34.0 |
| extreme_cpg | 0.1388 | 0.0000 | 0.0004 | yes | 0 | 1.00 | 27.0 |
| extreme_upa | 0.0837 | 0.0000 | 0.0004 | yes | 0 | 0.50 | 28.0 |
| random_low_lm | -0.0643 | 0.0000 | 0.0005 | yes | 0 | 0.48 | 31.0 |
| stable_hairpin | 0.0667 | 0.0000 | 0.0005 | yes | 0 | 0.70 | 29.0 |
| near_duplicate | -0.4707 | 0.0000 | 0.0001 | yes | 1 | 0.52 | 33.0 |

## Guard-Evasion Cases

- (none — all adversarial sequences with positive raw reward were caught by at least one guard)

## Per-Oracle Independent Rewards

### extreme_high_gc
- absolute: -0.0496
- difference: -0.0492
- siamese: -0.0496
- edit_conditioned: -0.0497

### extreme_low_gc
- absolute: -0.0495
- difference: -0.0495
- siamese: -0.0494
- edit_conditioned: -0.0498

### homopolymer_a
- absolute: -0.0493
- difference: -0.0490
- siamese: -0.0493
- edit_conditioned: -0.0497

### homopolymer_g
- absolute: -0.0496
- difference: -0.0492
- siamese: -0.0496
- edit_conditioned: -0.0497

### repeated_aug
- absolute: -0.0497
- difference: -0.0498
- siamese: -0.0497
- edit_conditioned: -0.0498

### repeated_uorf
- absolute: -0.0495
- difference: -0.0496
- siamese: -0.0494
- edit_conditioned: -0.0497

### cryptic_splice
- absolute: -0.0495
- difference: -0.0494
- siamese: -0.0495
- edit_conditioned: -0.0498

### polyadenylation
- absolute: -0.0493
- difference: -0.0492
- siamese: -0.0492
- edit_conditioned: -0.0497

### extreme_cpg
- absolute: -0.0497
- difference: -0.0492
- siamese: -0.0497
- edit_conditioned: -0.0497

### extreme_upa
- absolute: -0.0497
- difference: -0.0493
- siamese: -0.0496
- edit_conditioned: -0.0497

### random_low_lm
- absolute: -0.0494
- difference: -0.0494
- siamese: -0.0494
- edit_conditioned: -0.0497

### stable_hairpin
- absolute: -0.0496
- difference: -0.0493
- siamese: -0.0496
- edit_conditioned: -0.0497

### near_duplicate
- absolute: -0.0500
- difference: -0.0498
- siamese: -0.0500
- edit_conditioned: -0.0498
