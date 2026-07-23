# P3-06: Reward v3 Specification

**Date**: 2026-07-23
**Status**: DESIGN COMPLETE — implementation pending
**Phase**: P3-06
**Spec**: `提示词/mrna的 rl 的后续优化的分阶段提示词.md#L1791-1844`

---

## 1. Primary Reward

### Definition

```
R_primary = LCB[Δprotein_output]
```

where:

```
LCB = ensemble_mean - λ × ensemble_uncertainty
```

### Properties

| Property | Value |
|----------|-------|
| **Source-normalized** | Always relative to source: `Δ = predicted(candidate) - predicted(source)` |
| **Uncertainty-aware** | LCB penalizes high-uncertainty candidates |
| **Risk-adjusted** | Higher λ → more conservative (risk-averse); λ = 0 → pure mean |
| **Auditable** | Full provenance via `RewardComponent` |

### λ (Lambda) Selection

| λ | Behavior | Use Case |
|---|----------|----------|
| 0.0 | Pure mean (no risk adjustment) | Debugging only |
| 0.5 | Mild risk aversion | Default for exploration |
| 1.0 | Moderate risk aversion | Default for production |
| 2.0 | Strong risk aversion | Conservative design |

Default: **λ = 1.0** (moderate risk aversion).

---

## 2. Secondary Terms

Secondary terms are **additive** to the primary reward, each with its own weight:

```
R_total = R_primary + Σ w_i × R_secondary_i
```

| Term | Weight | Description | Source |
|------|--------|-------------|--------|
| `mrna_abundance_delta` | w_abundance | Predicted mRNA abundance change | P3-02 oracle |
| `half_life_delta` | w_half_life | Predicted half-life change | P3-02 oracle |
| `edit_cost` | w_edit_cost | Cost per edit (encourages fewer edits) | Fixed constant |
| `on_manifold_penalty` | w_manifold | Penalty for sequences off the natural manifold | Embedding distance |
| `manufacturability_penalty` | w_manufact | Penalty for manufacturing issues (GC extremes, homopolymers) | Rule-based |

### Default Weights (protein_output_focused context)

| Term | Weight |
|------|--------|
| R_primary (LCB protein output) | 1.0 |
| mrna_abundance_delta | 0.1 |
| half_life_delta | 0.1 |
| edit_cost | -0.05 (per edit) |
| on_manifold_penalty | 0.0 (disabled by default) |
| manufacturability_penalty | -0.1 (if violated) |

---

## 3. Hard Constraints

Hard constraints are **NOT** part of the linear reward. They are enforced by:

1. **Action space**: `apply_edit_action` guarantees protein identity and length invariance
2. **Rejection gate**: Any action that would violate a hard constraint is excluded from the legal action set
3. **STOP preference**: If all legal actions produce negative reward, STOP is preferred (from `build_training_reward`)

Hard constraints:

| Constraint | Enforced By | Verification |
|-----------|-------------|-------------|
| Protein identity | Action space (synonymous-only codon swap) | `translate(source.cds) == translate(candidate.cds)` |
| Transcript length | No indels in action space | `len(source.seq) == len(candidate.seq)` |
| Valid reading frame | Codon-level action (no frame shift) | CDS length is multiple of 3 |
| Preserved start codon | Action space (START_CODON excluded from synonymous set) | First codon is AUG |
| Preserved stop codon | Action space (STOP_CODONS excluded from synonymous set) | Last codon is stop |

---

## 4. Reward Vector Schema

Each action receives a `RewardVector` with full provenance:

```python
RewardVector(
    raw_absolute_level={
        "protein_output": 5.2,
        "mrna_abundance": 3.1,
        "half_life": 2.8,
    },
    raw_delta_from_source={
        "protein_output": 0.3,
        "mrna_abundance": 0.1,
        "half_life": 0.05,
    },
    normalized_within_group={
        "protein_output": 0.45,  # z-scored within group
    },
    components=[
        RewardComponent(
            name="protein_output",
            value=0.3,
            source_model="p3_02_delta_oracle",
            category="functional",
            independent=True,
            uncertainty=0.15,
            valid=True,
        ),
        RewardComponent(
            name="mrna_abundance",
            value=0.1,
            source_model="p3_02_delta_oracle",
            category="functional",
            independent=True,
            uncertainty=0.2,
            valid=True,
        ),
        RewardComponent(
            name="edit_cost",
            value=-0.05,
            source_model="fixed",
            category="edit_cost",
            independent=True,
            uncertainty=None,
            valid=True,
        ),
    ],
    validity={"protein_output": True, "mrna_abundance": True},
    redundancy_groups=["protein_output_group", "stability_group"],
)
```

---

## 5. Context Conditioning

Three pre-registered contexts (no continuous preference vector in the first paper):

### 5.1 `protein_output_focused`

```
R = LCB[Δprotein_output] - 0.05 × n_edits
```

Only protein output matters. All secondary terms except edit cost are zeroed.

### 5.2 `protein_output_with_stability_guard`

```
R = LCB[Δprotein_output] + 0.1 × Δhalf_life - 0.05 × n_edits
```

Protein output is primary, but half-life improvement is mildly rewarded.

### 5.3 `balanced_experimental_profile`

```
R = LCB[Δprotein_output] + 0.1 × Δmrna_abundance + 0.1 × Δhalf_life - 0.05 × n_edits - 0.1 × manufacturability_penalty
```

Balanced optimization across protein output, abundance, half-life, and manufacturability.

---

## 6. Implementation

```python
def compute_reward_v3(
    source: MRNARecord,
    candidate: MRNARecord,
    oracle: DeltaOracle,
    context: str = "protein_output_focused",
    lambda_lcb: float = 1.0,
    n_edits: int = 0,
) -> TrainingReward:
    """Compute reward v3 for a candidate edit.

    Returns TrainingReward with:
    - scalar: final reward for advantage computation
    - source_normalized_delta: always relative to source
    - risk_adjusted_delta: after LCB penalty
    - hard_constraint_gated: False (hard constraints enforced by action space)
    - stop_preferred: True if all-negative
    - raw_components: per-component deltas
    - reward_provenance: full audit trail
    """
    # 1. Predict deltas
    delta_pred = oracle.predict_delta(source, candidate)
    uncertainty = oracle.predict_uncertainty(source, candidate)

    # 2. Primary reward: LCB
    lcb = delta_pred["protein_output"] - lambda_lcb * uncertainty["protein_output"]

    # 3. Secondary terms (context-dependent)
    secondary = _compute_secondary(source, candidate, delta_pred, context, n_edits)

    # 4. Total
    scalar = lcb + secondary["total"]

    # 5. Build training reward
    return build_training_reward(
        raw_components={
            "protein_output": delta_pred["protein_output"],
            "mrna_abundance": delta_pred.get("mrna_abundance", 0.0),
            "half_life": delta_pred.get("half_life", 0.0),
            "edit_cost": -0.05 * n_edits,
        },
        uncertainty=uncertainty["protein_output"],
        hard_constraint_status=HardConstraintStatus(),  # Always pass (enforced by action space)
        source_baseline=0.0,  # Always relative to source
        stop_reward=0.0,
        uncertainty_penalty_multiplier=lambda_lcb,
    )
```

---

## 7. Acceptance Criteria Mapping

| Criterion | How Verified |
|-----------|-------------|
| Reward is source-normalized | `raw_delta_from_source` in `RewardVector`; `source_baseline=0.0` in `build_training_reward` |
| Uncertainty enters risk adjustment | `LCB = mean - λ × uncertainty` |
| Hard constraints not in linear reward | Hard constraints enforced by action space; `HardConstraintStatus` always passes |
| Action/reward provenance complete | `RewardComponent.source_model` + `reward_provenance` dict |
