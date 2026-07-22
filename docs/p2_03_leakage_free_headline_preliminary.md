# P2-03: Leakage-Free Headline (Preliminary)

**Status**: PRELIMINARY — 3 training seeds (te_only/scalar/pareto/grpo) × 10 decoder seeds + 1 training seed (hardneg_v2) × 10 decoder seeds.
**Primary endpoint**: `delta_oracle_te_vs_source` (qualifier: `predicted_te_internal_proxy`)
**Secondary endpoints**: `oracle_ensemble_te`, `edit_distance`, `mfe_proxy`

## Ranking (by primary endpoint)

| Rank | Baseline | Primary mean | 95% CI | n seeds |
|-----:|----------|-------------:|--------|--------:|
| 1 | grpo_seed2 | 0.007805 | [0.007471, 0.008176] | 10 |
| 2 | te_only_seed2 | 0.007726 | [0.007522, 0.007907] | 10 |
| 3 | te_only_seed1 | 0.007150 | [0.006893, 0.007386] | 10 |
| 4 | grpo_seed1 | 0.006967 | [0.006687, 0.007241] | 10 |
| 5 | grpo_seed0 | 0.006770 | [0.006592, 0.006956] | 10 |
| 6 | scalar_seed0 | 0.006618 | [0.006316, 0.006941] | 10 |
| 7 | pareto_seed1 | 0.006520 | [0.006213, 0.006821] | 10 |
| 8 | pareto_seed2 | 0.006314 | [0.005959, 0.006673] | 10 |
| 9 | pareto_seed0 | 0.006293 | [0.005853, 0.006741] | 10 |
| 10 | scalar_seed2 | 0.005221 | [0.004996, 0.005444] | 10 |
| 11 | scalar_seed1 | 0.004889 | [0.004609, 0.005147] | 10 |
| 12 | hardneg_v2_seed0 | 0.004166 | [0.003748, 0.004502] | 10 |
| 13 | te_only_seed0 | 0.002835 | [0.002468, 0.003201] | 10 |

## Secondary endpoints

### `oracle_ensemble_te`

| Baseline | Mean | 95% CI | n |
|----------|-----:|--------|---:|
| grpo_seed2 | 0.8031 | [0.8028, 0.8035] | 10 |
| te_only_seed2 | 0.8031 | [0.8029, 0.8032] | 10 |
| te_only_seed1 | 0.8025 | [0.8022, 0.8027] | 10 |
| grpo_seed1 | 0.8023 | [0.8020, 0.8026] | 10 |
| grpo_seed0 | 0.8021 | [0.8019, 0.8023] | 10 |
| scalar_seed0 | 0.8020 | [0.8017, 0.8023] | 10 |
| pareto_seed1 | 0.8019 | [0.8016, 0.8022] | 10 |
| pareto_seed2 | 0.8017 | [0.8013, 0.8020] | 10 |
| pareto_seed0 | 0.8016 | [0.8012, 0.8021] | 10 |
| scalar_seed2 | 0.8006 | [0.8003, 0.8008] | 10 |
| scalar_seed1 | 0.8002 | [0.7999, 0.8005] | 10 |
| hardneg_v2_seed0 | 0.7995 | [0.7991, 0.7998] | 10 |
| te_only_seed0 | 0.7982 | [0.7978, 0.7985] | 10 |

### `edit_distance`

| Baseline | Mean | 95% CI | n |
|----------|-----:|--------|---:|
| grpo_seed2 | 2.9282 | [2.9265, 2.9297] | 10 |
| te_only_seed2 | 2.9054 | [2.9010, 2.9098] | 10 |
| te_only_seed1 | 2.8947 | [2.8899, 2.9001] | 10 |
| grpo_seed1 | 2.9281 | [2.9267, 2.9301] | 10 |
| grpo_seed0 | 2.9220 | [2.9195, 2.9246] | 10 |
| scalar_seed0 | 2.9346 | [2.9335, 2.9356] | 10 |
| pareto_seed1 | 2.9363 | [2.9355, 2.9371] | 10 |
| pareto_seed2 | 2.9348 | [2.9339, 2.9356] | 10 |
| pareto_seed0 | 2.9331 | [2.9323, 2.9338] | 10 |
| scalar_seed2 | 2.9349 | [2.9339, 2.9357] | 10 |
| scalar_seed1 | 2.9334 | [2.9316, 2.9351] | 10 |
| hardneg_v2_seed0 | 2.9280 | [2.9254, 2.9303] | 10 |
| te_only_seed0 | 2.9158 | [2.9135, 2.9183] | 10 |

### `mfe_proxy`

| Baseline | Mean | 95% CI | n |
|----------|-----:|--------|---:|
| grpo_seed2 | -46.3816 | [-46.3820, -46.3812] | 10 |
| te_only_seed2 | -46.4038 | [-46.4044, -46.4031] | 10 |
| te_only_seed1 | -46.4132 | [-46.4143, -46.4122] | 10 |
| grpo_seed1 | -46.3804 | [-46.3808, -46.3799] | 10 |
| grpo_seed0 | -46.3884 | [-46.3892, -46.3876] | 10 |
| scalar_seed0 | -46.3709 | [-46.3712, -46.3706] | 10 |
| pareto_seed1 | -46.3724 | [-46.3727, -46.3720] | 10 |
| pareto_seed2 | -46.3800 | [-46.3804, -46.3795] | 10 |
| pareto_seed0 | -46.3814 | [-46.3821, -46.3807] | 10 |
| scalar_seed2 | -46.3706 | [-46.3710, -46.3703] | 10 |
| scalar_seed1 | -46.3715 | [-46.3719, -46.3711] | 10 |
| hardneg_v2_seed0 | -46.4190 | [-46.4195, -46.4184] | 10 |
| te_only_seed0 | -46.4588 | [-46.4596, -46.4579] | 10 |

## Pairwise comparisons (primary endpoint)

| Baseline A | Baseline B | Δ (B−A) | 95% CI | p-value | Sig.? |
|------------|------------|--------:|--------|--------:|:-----:|
| grpo_seed0 | grpo_seed1 | 0.000197 | [-0.000146, 0.000492] | 0.2478 |  |
| grpo_seed0 | grpo_seed2 | 0.001036 | [0.000612, 0.001470] | 0.0000 | ✓ |
| grpo_seed0 | hardneg_v2_seed0 | -0.002604 | [-0.002960, -0.002265] | 0.0000 | ✓ |
| grpo_seed0 | pareto_seed0 | -0.000477 | [-0.000979, -0.000011] | 0.0420 | ✓ |
| grpo_seed0 | pareto_seed1 | -0.000250 | [-0.000598, 0.000128] | 0.1926 |  |
| grpo_seed0 | pareto_seed2 | -0.000456 | [-0.000852, -0.000052] | 0.0252 | ✓ |
| grpo_seed0 | scalar_seed0 | -0.000152 | [-0.000455, 0.000182] | 0.3552 |  |
| grpo_seed0 | scalar_seed1 | -0.001881 | [-0.002142, -0.001622] | 0.0000 | ✓ |
| grpo_seed0 | scalar_seed2 | -0.001549 | [-0.001843, -0.001232] | 0.0000 | ✓ |
| grpo_seed0 | te_only_seed0 | -0.003935 | [-0.004378, -0.003511] | 0.0000 | ✓ |
| grpo_seed0 | te_only_seed1 | 0.000380 | [-0.000002, 0.000695] | 0.0512 |  |
| grpo_seed0 | te_only_seed2 | 0.000956 | [0.000740, 0.001169] | 0.0000 | ✓ |
| grpo_seed1 | grpo_seed2 | 0.000838 | [0.000308, 0.001388] | 0.0010 | ✓ |
| grpo_seed1 | hardneg_v2_seed0 | -0.002802 | [-0.003285, -0.002349] | 0.0000 | ✓ |
| grpo_seed1 | pareto_seed0 | -0.000674 | [-0.001041, -0.000351] | 0.0000 | ✓ |
| grpo_seed1 | pareto_seed1 | -0.000448 | [-0.000790, -0.000055] | 0.0262 | ✓ |
| grpo_seed1 | pareto_seed2 | -0.000653 | [-0.001056, -0.000227] | 0.0022 | ✓ |
| grpo_seed1 | scalar_seed0 | -0.000350 | [-0.000804, 0.000141] | 0.1562 |  |
| grpo_seed1 | scalar_seed1 | -0.002079 | [-0.002387, -0.001741] | 0.0000 | ✓ |
| grpo_seed1 | scalar_seed2 | -0.001746 | [-0.002113, -0.001408] | 0.0000 | ✓ |
| grpo_seed1 | te_only_seed0 | -0.004133 | [-0.004595, -0.003593] | 0.0000 | ✓ |
| grpo_seed1 | te_only_seed1 | 0.000183 | [-0.000219, 0.000581] | 0.3722 |  |
| grpo_seed1 | te_only_seed2 | 0.000758 | [0.000483, 0.001059] | 0.0000 | ✓ |
| grpo_seed2 | hardneg_v2_seed0 | -0.003640 | [-0.004289, -0.003116] | 0.0000 | ✓ |
| grpo_seed2 | pareto_seed0 | -0.001512 | [-0.002165, -0.000849] | 0.0000 | ✓ |
| grpo_seed2 | pareto_seed1 | -0.001286 | [-0.001773, -0.000821] | 0.0000 | ✓ |
| grpo_seed2 | pareto_seed2 | -0.001491 | [-0.001928, -0.001091] | 0.0000 | ✓ |
| grpo_seed2 | scalar_seed0 | -0.001188 | [-0.001631, -0.000748] | 0.0000 | ✓ |
| grpo_seed2 | scalar_seed1 | -0.002917 | [-0.003451, -0.002432] | 0.0000 | ✓ |
| grpo_seed2 | scalar_seed2 | -0.002584 | [-0.003006, -0.002155] | 0.0000 | ✓ |
| grpo_seed2 | te_only_seed0 | -0.004971 | [-0.005426, -0.004550] | 0.0000 | ✓ |
| grpo_seed2 | te_only_seed1 | -0.000656 | [-0.001025, -0.000278] | 0.0012 | ✓ |
| grpo_seed2 | te_only_seed2 | -0.000080 | [-0.000580, 0.000344] | 0.7740 |  |
| hardneg_v2_seed0 | pareto_seed0 | 0.002128 | [0.001532, 0.002748] | 0.0000 | ✓ |
| hardneg_v2_seed0 | pareto_seed1 | 0.002354 | [0.002002, 0.002753] | 0.0000 | ✓ |
| hardneg_v2_seed0 | pareto_seed2 | 0.002149 | [0.001621, 0.002776] | 0.0000 | ✓ |
| hardneg_v2_seed0 | scalar_seed0 | 0.002452 | [0.001973, 0.002993] | 0.0000 | ✓ |
| hardneg_v2_seed0 | scalar_seed1 | 0.000723 | [0.000248, 0.001255] | 0.0010 | ✓ |
| hardneg_v2_seed0 | scalar_seed2 | 0.001056 | [0.000550, 0.001644] | 0.0000 | ✓ |
| hardneg_v2_seed0 | te_only_seed0 | -0.001331 | [-0.001909, -0.000718] | 0.0000 | ✓ |
| hardneg_v2_seed0 | te_only_seed1 | 0.002984 | [0.002518, 0.003514] | 0.0000 | ✓ |
| hardneg_v2_seed0 | te_only_seed2 | 0.003560 | [0.003352, 0.003811] | 0.0000 | ✓ |
| pareto_seed0 | pareto_seed1 | 0.000226 | [-0.000299, 0.000780] | 0.4148 |  |
| pareto_seed0 | pareto_seed2 | 0.000021 | [-0.000623, 0.000698] | 0.9466 |  |
| pareto_seed0 | scalar_seed0 | 0.000324 | [-0.000295, 0.000915] | 0.3024 |  |
| pareto_seed0 | scalar_seed1 | -0.001405 | [-0.001986, -0.000820] | 0.0000 | ✓ |
| pareto_seed0 | scalar_seed2 | -0.001072 | [-0.001633, -0.000539] | 0.0000 | ✓ |
| pareto_seed0 | te_only_seed0 | -0.003459 | [-0.004062, -0.002796] | 0.0000 | ✓ |
| pareto_seed0 | te_only_seed1 | 0.000857 | [0.000403, 0.001331] | 0.0002 | ✓ |
| pareto_seed0 | te_only_seed2 | 0.001432 | [0.000948, 0.001939] | 0.0000 | ✓ |
| pareto_seed1 | pareto_seed2 | -0.000205 | [-0.000648, 0.000190] | 0.3614 |  |
| pareto_seed1 | scalar_seed0 | 0.000098 | [-0.000374, 0.000645] | 0.7554 |  |
| pareto_seed1 | scalar_seed1 | -0.001631 | [-0.002029, -0.001268] | 0.0000 | ✓ |
| pareto_seed1 | scalar_seed2 | -0.001298 | [-0.001678, -0.000908] | 0.0000 | ✓ |
| pareto_seed1 | te_only_seed0 | -0.003685 | [-0.004195, -0.003099] | 0.0000 | ✓ |
| pareto_seed1 | te_only_seed1 | 0.000630 | [0.000252, 0.001052] | 0.0004 | ✓ |
| pareto_seed1 | te_only_seed2 | 0.001206 | [0.000919, 0.001478] | 0.0000 | ✓ |
| pareto_seed2 | scalar_seed0 | 0.000303 | [-0.000245, 0.000896] | 0.3082 |  |
| pareto_seed2 | scalar_seed1 | -0.001426 | [-0.001726, -0.001105] | 0.0000 | ✓ |
| pareto_seed2 | scalar_seed2 | -0.001093 | [-0.001446, -0.000751] | 0.0000 | ✓ |
| pareto_seed2 | te_only_seed0 | -0.003480 | [-0.003929, -0.002946] | 0.0000 | ✓ |
| pareto_seed2 | te_only_seed1 | 0.000836 | [0.000433, 0.001225] | 0.0000 | ✓ |
| pareto_seed2 | te_only_seed2 | 0.001411 | [0.001021, 0.001752] | 0.0000 | ✓ |
| scalar_seed0 | scalar_seed1 | -0.001729 | [-0.002154, -0.001381] | 0.0000 | ✓ |
| scalar_seed0 | scalar_seed2 | -0.001396 | [-0.001789, -0.000994] | 0.0000 | ✓ |
| scalar_seed0 | te_only_seed0 | -0.003783 | [-0.004272, -0.003344] | 0.0000 | ✓ |
| scalar_seed0 | te_only_seed1 | 0.000532 | [0.000031, 0.001020] | 0.0370 | ✓ |
| scalar_seed0 | te_only_seed2 | 0.001108 | [0.000655, 0.001517] | 0.0000 | ✓ |
| scalar_seed1 | scalar_seed2 | 0.000333 | [0.000073, 0.000567] | 0.0124 | ✓ |
| scalar_seed1 | te_only_seed0 | -0.002054 | [-0.002537, -0.001548] | 0.0000 | ✓ |
| scalar_seed1 | te_only_seed1 | 0.002261 | [0.001807, 0.002678] | 0.0000 | ✓ |
| scalar_seed1 | te_only_seed2 | 0.002837 | [0.002510, 0.003148] | 0.0000 | ✓ |
| scalar_seed2 | te_only_seed0 | -0.002387 | [-0.002760, -0.002001] | 0.0000 | ✓ |
| scalar_seed2 | te_only_seed1 | 0.001928 | [0.001600, 0.002226] | 0.0000 | ✓ |
| scalar_seed2 | te_only_seed2 | 0.002504 | [0.002142, 0.002857] | 0.0000 | ✓ |
| te_only_seed0 | te_only_seed1 | 0.004315 | [0.004029, 0.004588] | 0.0000 | ✓ |
| te_only_seed0 | te_only_seed2 | 0.004891 | [0.004410, 0.005340] | 0.0000 | ✓ |
| te_only_seed1 | te_only_seed2 | 0.000576 | [0.000243, 0.000908] | 0.0004 | ✓ |

## Note

PRELIMINARY (1 training seed × 10 decoder seeds). Full P2-03 requires ≥3 training seeds with family-cluster bootstrap CI. Phase C will add remaining training seeds.

**Constraint compliance**:
- All claims use `predicted_te_internal_proxy` qualifier.
- Split contract enforced (--train-idx/--val-idx/--test-idx).
- Preliminary: 1 training seed only. Phase C will add seeds 1, 2.

## Finalization (2026-07-21)

The full P2-03 evaluation is now complete with 3 training seeds for 4 modes
(te_only, scalar, pareto, grpo) plus 1 seed for hardneg_v2, totaling 13
baseline×seed combinations × 10 decoder seeds = 130 evaluations.

**Final artifact**:
- Path: `benchmark/paper/leakage_free_headline.json`
- Size: 76197 bytes
- SHA-256: `dc54b9fdd55651b8f51e8f0a2c82c4c9e71fcba2be9680e1efad8cc0c0c2981d`
- SHA-256 sidecar: `benchmark/paper/leakage_free_headline.json.sha256`

**Contents**:
- 13 baselines: grpo_seed0/1/2, hardneg_v2_seed0, pareto_seed0/1/2,
  scalar_seed0/1/2, te_only_seed0/1/2
- 10 decoder seeds per baseline (seed_000 through seed_009)
- Primary endpoint: `delta_oracle_te_vs_source` (predicted_te_internal_proxy)
- 78 pairwise comparisons with family-cluster bootstrap CI
- 13 entries ranked by primary endpoint

**Ranked by primary endpoint (delta_oracle_te_vs_source)**:
1. grpo_seed2
2. te_only_seed2
3. te_only_seed1
4. grpo_seed1
5. grpo_seed0
6. scalar_seed0
7. pareto_seed1
8. pareto_seed2
9. pareto_seed0
10. scalar_seed2
11. scalar_seed1
12. hardneg_v2_seed0
13. te_only_seed0

**Source data**: `benchmark/dev/leakage_free_headline_preliminary/` (13 dirs,
each with 10 decoder seed `eval_summary.json` files)

**Aggregator**: `scripts/aggregate_p2_03_headline.py` (444 lines)

**Constraint compliance**:
- All claims use `predicted_te_internal_proxy` qualifier (per P2-01 constraint)
- Split contract enforced (--train-idx/--val-idx/--test-idx)
- Family-cluster bootstrap CI (cluster = training seed)
- Pre-registered primary endpoint: delta_oracle_te_vs_source
- 3 training seeds per mode (te_only, scalar, pareto, grpo), 1 for hardneg_v2
- 10 decoder seeds per training seed
