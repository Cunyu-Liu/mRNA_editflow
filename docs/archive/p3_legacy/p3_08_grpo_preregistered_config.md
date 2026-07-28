# P3-08: Pre-Registered GRPO Configuration

**Date**: 2026-07-24
**Phase**: P3-08 (Production GRPO / Amortized Policy Training)
**Pre-gate**: P3-07 RL_ROUTE_B (search amortization story, break-even 1.52 cargos)
**Status**: FROZEN — no modifications after Gate A launch

---

## 1. Training Configuration

### 1.1 Edit Budget Curriculum

| Stage | Edit Budget | Rollout Horizon | Region |
|-------|-------------|-----------------|--------|
| Stage 1 | 1 | 1 + STOP | 5'UTR-only (Task A) |
| Stage 2 | 3 | 3 + STOP | 5'UTR-only (Task A) |
| Stage 3 | 5 | 5 + STOP | 5'UTR-only (Task A) |
| Stage 4 | 10 | 10 + STOP | 5'UTR-only (Task A) |

**Note**: CDS-only and joint 5'UTR+CDS are conditional extensions (Task B/C)
locked per P3-00A frozen contract. Unlock requires H1+H2+H3 pass in BOTH
regions. Gate A/B uses Task A (5'UTR-only) only.

### 1.2 Multi-Source Batch

| Parameter | Value |
|-----------|-------|
| sources_per_batch | 8 |
| group_size_per_source | 4 |
| trajectories_per_update | 32 (8 × 4) |
| policy_epochs | 2 |
| trajectory_minibatches | enabled (4 minibatches) |

### 1.3 Optimizer

| Parameter | Value |
|-----------|-------|
| optimizer | AdamW |
| learning_rate | 3e-4 |
| weight_decay | 1e-4 |
| gradient_clip | 1.0 |
| gradient_accumulation | 1 |
| mixed_precision | disabled (fp32 for reproducibility) |
| warmup_steps | 50 (linear) |

---

## 2. Objective Function

```
L = L_clipped_GRPO
  + β_KL × KL(current || reference)
  + β_flow × L_EditFlow
  - β_entropy × H(π)
  + β_calib × L_calibration
```

| Coefficient | Value | Notes |
|-------------|-------|-------|
| β_KL (initial) | 0.05 | Adaptive KL controller, max_kl = 0.25 |
| β_flow | 0.0 | Flow replay disabled for Gate A (enable in Gate B if needed) |
| β_entropy | 0.01 | Encourage exploration |
| β_calib | 0.0 | Delta-calibration loss disabled for Gate A |
| clip_epsilon | 0.2 | PPO clip range |

### 2.1 Advantage Computation

- **Group normalization**: per-source, per-objective standardization
- **Risk-adjusted**: LCB (mean - λ × uncertainty) as primary objective
- **Source-normalized**: relative improvement (delta / source_scale)
- **Constant objective guard**: zero advantage if variance < 1e-8

### 2.2 Reward Configuration (RewardV3)

| Parameter | Value |
|-----------|-------|
| context | protein_output_focused |
| lambda_lcb | 1.0 |
| w_edit_cost | -0.05 |
| w_abundance | 0.0 (disabled) |
| w_half_life | 0.0 (disabled) |
| w_manifold | 0.0 (disabled) |
| w_manufacturability | 0.0 (disabled) |

### 2.3 Oracle

- **Oracle**: P3-02 EnsembleDeltaOracle (remediated)
- **Models**: seq_diff (MLP on one-hot diff) + seq_linear (ridge regression)
- **Centering**: source-bias subtraction (delta(source→source) = 0)
- **Uncertainty**: ensemble std (ddof=0) + 1e-6
- **Training oracle calls**: 0 (ensemble trained on benchmark labels)
- **Inference oracle calls**: 1 per trajectory (verification only)

---

## 3. Policy Architecture

### 3.1 Backbone: 1D CNN Sequence Encoder

```
Input: one-hot mRNA sequence (4 × L, L = 50 + 15 + 4 = 69 for Task A)
  ↓
Conv1d(4, 64, kernel=7, padding=3) + ReLU
  ↓
Conv1d(64, 128, kernel=5, padding=2) + ReLU
  ↓
AdaptiveMaxPool1d(1) → (128,)
  ↓
Linear(128, 64) + ReLU → seq_repr (64,)
```

### 3.2 Hierarchical Action Heads

| Level | Head | Input | Output |
|-------|------|-------|--------|
| 1 | STOP/EDIT | seq_repr + budget features | p_stop (scalar sigmoid) |
| 2 | Region | seq_repr | region_probs (2-way softmax, Task A masks CDS) |
| 3 | Position | seq_repr + region embedding | position_probs (max_utr_len-way softmax, masked) |
| 4 | Target | seq_repr + position embedding | target_probs (3-way softmax for 5'UTR, masked) |

### 3.3 Additional Features

- **Budget features**: [n_edits, remaining_budget, remaining_budget_frac]
- **Oracle features**: [current_predicted_delta, oracle_uncertainty]
- **Edit history**: none (Markovian assumption, history via current sequence)

### 3.4 Reference Policy

- **Reference**: frozen copy of initial policy (before any GRPO updates)
- **KL**: categorical KL between current and reference action distributions
- **Adaptive controller**: coefficient doubles if KL > max_kl, halves if KL < max_kl/2

---

## 4. Gate A: 3-Seed Production Pilot

### 4.1 Configuration

| Parameter | Value |
|-----------|-------|
| seeds | [42, 123, 456] |
| optimizer_updates | 1000 |
| edit_budget | 1 (Stage 1 only for pilot) |
| sources_per_batch | 8 |
| group_size_per_source | 4 |
| validation_interval | 100 updates |
| checkpoint_interval | 200 updates |

### 4.2 Pass Criteria (ALL must be met)

1. **No numerical collapse**: loss is finite for >= 99% of updates
2. **Hard constraints 100%**: all sampled trajectories preserve protein identity and length
3. **>= 2/3 seeds beat warm start**: validation LCB improvement > 0 for >= 2 seeds
4. **No reward hacking**: positive-improvement rate on validation >= 30% (not just STOP)
5. **STOP not collapsed**: p_stop at root < 0.9 (policy still explores edits)
6. **KL normal**: mean observed KL < 0.5 nats, no KL divergence explosions
7. **Clip fraction normal**: mean clip fraction in [0.05, 0.5]

### 4.3 Validation Protocol

- **Validation sources**: 24 test mothers from P3-07 (held-out, not in training)
- **Validation metric**: mean LCB improvement (policy vs source), averaged over 32 trajectories per source
- **Warm-start baseline**: initial policy (before GRPO) on same validation sources
- **Pre-registered**: validation sources and metric frozen before Gate A launch

---

## 5. Gate B: 10-Seed Paper Run

### 5.1 Configuration

| Parameter | Value |
|-----------|-------|
| seeds | [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144] |
| optimizer_updates | 5000 |
| edit_budget curriculum | 1 → 3 → 5 → 10 (1250 updates per stage) |
| sources_per_batch | 8 |
| group_size_per_source | 4 |
| validation_interval | 250 updates |
| checkpoint_interval | 500 updates |

### 5.2 Pass Criteria

1. All Gate A criteria met for >= 7/10 seeds
2. Mean validation LCB improvement > 0 across seeds
3. No seed cherry-picking: report all 10 seeds including failures
4. Best checkpoint selected by pre-registered criteria (Section 6)

---

## 6. Checkpoint Selection

### 6.1 Hard Requirement

```
hard_constraints = 100%
```

Any checkpoint with < 100% constraint validity is disqualified.

### 6.2 Selection Criteria (ranked by priority)

1. **Independent-oracle LCB improvement**: mean LCB on validation sources
2. **Positive-improvement rate**: fraction of validation trajectories with delta > 0
3. **Reward per edit**: mean reward / mean n_edits
4. **Pareto hypervolume**: (LCB, positive_rate, reward_per_edit) hypervolume
5. **STOP calibration**: |p_stop(root) - optimal_stop_rate| < 0.2
6. **OOD performance**: LCB improvement on proxy-tier sources
7. **KL constraint**: observed KL < 0.5 nats

### 6.3 Selection Rule

- Filter: hard_constraints = 100%
- Rank: Pareto hypervolume on (LCB, positive_rate, reward_per_edit)
- Tiebreak: lower KL, then fewer parameters
- **NOT** selected by: training return, test set performance

---

## 7. Data Splits

### 7.1 Training Sources

- 24 train mothers from P3-07 (benchmark measured tier, split_role=train)
- Each update samples 8 sources randomly from training pool

### 7.2 Validation Sources

- 24 test mothers from P3-07 (benchmark measured tier, split_role=test)
- Held-out: never used for training or checkpoint selection tuning

### 7.3 Oracle Data

- P3-02 benchmark: measured tier (3,984 train+val) + proxy tier (10k subsample)
- Ensemble: 5-fold cross-fitted, seed 42, models = (seq_diff, seq_linear)
- Centering: source-bias subtracted

---

## 8. Anti-Reward-Hacking Measures

1. **STOP collapse detection**: if p_stop(root) > 0.95 for > 50 consecutive updates, halt
2. **Identity edit detection**: if > 50% of edits are identity (no change), halt
3. **Oracle overfit detection**: if training LCB >> validation LCB (gap > 3x), halt
4. **Constraint monitoring**: if any trajectory violates protein identity, halt
5. **KL explosion**: if observed KL > 2.0 nats, halt

---

## 9. Reproducibility

| Item | Value |
|------|-------|
| Random seeds | Pre-registered (Section 4.1, 5.1) |
| Data splits | Frozen from P3-07 (same 24 test/train mothers) |
| Oracle | Frozen P3-02 ensemble (remediated, centering) |
| Config hash | Computed at launch, recorded in results JSON |
| Code commit | Recorded in results JSON |
| Hardware | NVIDIA A100 40GB (GPU 6 or 7, least utilized) |
| Software | PyTorch 2.5.1, CUDA 12.1, Python 3.10 |

---

## 10. Output Artifacts

| Artifact | Path | Content |
|----------|------|---------|
| Pre-registered config | `docs/p3_08_grpo_preregistered_config.md` | This file (frozen) |
| Training health | `docs/p3_08_grpo_training_health.md` | Loss/KL/entropy/clip curves, constraint validity, STOP rates |
| Results JSON | `docs/p3_08_grpo_results.json` | Per-seed metrics, checkpoint selection, validation scores |
| Failed runs | `docs/p3_08_failed_runs.md` | Any seeds that crashed or failed criteria (no cherry-picking) |
