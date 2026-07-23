# P3-06: Minimal-Edit MDP Specification

**Date**: 2026-07-23
**Status**: DESIGN COMPLETE — implementation pending
**Phase**: P3-06
**Spec**: `提示词/mrna的 rl 的后续优化的分阶段提示词.md#L1702-1883`

---

## 1. MDP State

The state `s_t` at step `t` contains:

| Field | Type | Description |
|-------|------|-------------|
| `source_mrna` | `MRNARecord` | The original wild-type source sequence (immutable) |
| `current_mrna` | `MRNARecord` | The current edited sequence (changes each step) |
| `edit_history` | `List[EditAction]` | Ordered list of all edits applied so far |
| `visited_states` | `Set[str]` | Sequence hashes of all visited states (for cycle detection) |
| `remaining_budget` | `int` | Remaining edit budget (starts at `k`, decrements each edit) |
| `cargo_identity` | `str` | Cargo identifier (e.g., "EGFP", "mCherry") |
| `cell_context` | `str` | Cell type / expression context |
| `oracle_uncertainty` | `float` | Current uncertainty from the delta oracle |
| `current_predicted_delta` | `float` | Current predicted delta vs source |

```python
@dataclass(frozen=True)
class MDPState:
    source_mrna: MRNARecord
    current_mrna: MRNARecord
    edit_history: List[EditAction]  # Immutable list
    visited_states: frozenset[str]  # Immutable set of sequence hashes
    remaining_budget: int
    cargo_identity: str
    cell_context: str = "default"
    oracle_uncertainty: float = 0.0
    current_predicted_delta: float = 0.0
```

---

## 2. Primary Actions

### Action Space

The primary task has exactly 3 action types (no indels):

| Action | Parameters | Description |
|--------|-----------|-------------|
| `STOP` | — | Terminate the trajectory |
| `FIVE_UTR_SUB` | `(position: int, nucleotide: str)` | Single-nt substitution in 5'UTR |
| `CDS_SYNONYMOUS_SUB` | `(codon_position: int, target_codon: str)` | Synonymous codon substitution in CDS |

### Action Constraints

1. **Protein identity 100%**: `translate(source.cds) == translate(candidate.cds)` — enforced by action space, not reward
2. **Transcript length 100% unchanged**: `len(source.seq) == len(candidate.seq)` — enforced by no indels
3. **No single-nt CDS nonsynonymous intermediate**: `CDS_SYNONYMOUS_SUB` is atomic — it replaces an entire codon, never a single nucleotide within a codon
4. **5'UTR substitution**: Any of A/C/G/U at any 5'UTR position (identity substitution excluded from legal set)
5. **CDS synonymous substitution**: Only codons from `SYNONYMOUS_CODONS[current_codon]` — start/stop codons excluded

### Action Application

```python
def apply_edit_action(record: MRNARecord, action: EditAction) -> MRNARecord:
    """Apply an edit action. Returns a new MRNARecord.

    Guarantees:
    - translate(record.cds) == translate(result.cds)  (protein identity)
    - len(record.seq) == len(result.seq)               (length invariant)
    - No single-nt CDS intermediate states              (atomic codon swap)
    """
    if action.is_stop():
        return record  # No change
    if action.op == "five_utr_sub":
        new_utr = record.five_utr[:action.pos] + action.nt + record.five_utr[action.pos+1:]
        return MRNARecord(record.transcript_id, new_utr, record.cds, record.three_utr)
    if action.op == "cds_synonymous_sub":
        # Replace codon at action.codon_pos with action.target_codon
        nt_start = action.codon_pos * 3
        new_cds = record.cds[:nt_start] + action.target_codon + record.cds[nt_start+3:]
        # Verify: protein preserved
        assert translate(record.cds) == translate(new_cds)
        return MRNARecord(record.transcript_id, record.five_utr, new_cds, record.three_utr)
    raise ValueError(f"Unknown action op: {action.op}")
```

---

## 3. Hierarchical Policy

The policy decomposes into 4 levels:

```
π(action | state) = π(STOP_or_EDIT | state)
                  × π(region | EDIT, state)
                  × π(position | region, state)
                  × π(target | position, state)
```

### Level 1: STOP or EDIT

Binary decision. Inputs:
- Global sequence representation (mean-pooled backbone embedding)
- Edit history summary (count, mean predicted delta)
- Remaining budget (normalized)
- Predicted improvement (best available LCB)
- Uncertainty

Output: `p_stop ∈ [0, 1]`

### Level 2: Region Selection

Given EDIT, choose region: `{5'UTR, CDS}`.

Inputs:
- Per-region representation (max-pooled backbone embedding per region)
- Per-region edit history (count per region)
- Remaining budget

Output: `p_region ∈ Δ^2` (3-dim probability vector, 3'UTR excluded in primary task)

### Level 3: Position Selection

Given region, choose position within that region.

Inputs:
- Per-position backbone embedding
- Per-position structure features
- Distance from start codon / region boundaries
- Visited state mask (avoid cycles)

Output: `p_position ∈ Δ^|region|` (probability over positions in the selected region)

### Level 4: Target Selection

Given position, choose target nucleotide (5'UTR) or target codon (CDS).

For 5'UTR: `p_target ∈ Δ^4` (A/C/G/U, identity excluded)
For CDS: `p_target ∈ Δ^|synonymous|` (legal synonymous codons for current codon)

Inputs:
- Current nucleotide/codon embedding
- Target nucleotide/codon embedding
- Predicted delta for each candidate
- Synonymous flag

### Full Log-Probability

```python
def log_pi(action: EditAction, state: MDPState) -> float:
    """Compute log π(action | state) via the hierarchical decomposition."""
    if action.is_stop():
        return log(p_stop)
    log_p = log(1 - p_stop)                        # Level 1: EDIT
    log_p += log(p_region[action.region])          # Level 2: region
    log_p += log(p_position[action.position])       # Level 3: position
    log_p += log(p_target[action.target])           # Level 4: target
    return log_p
```

---

## 4. Learnable STOP

### Input Features

| Feature | Source | Dimension |
|---------|--------|-----------|
| Global representation | Mean-pooled backbone embedding | `model_dim` |
| Edit count | `len(edit_history)` | 1 |
| Remaining budget (normalized) | `remaining_budget / total_budget` | 1 |
| Best available LCB | `max(LCB(candidates))` | 1 |
| Current uncertainty | `oracle_uncertainty` | 1 |
| Mean predicted delta | `mean(deltas in edit_history)` | 1 |
| Delta trend | `last_delta - first_delta` | 1 |
| **Total** | | `model_dim + 6` |

### STOP Variants (Ablation)

| Variant | Description | Parameters |
|---------|-------------|------------|
| `constant_stop` | Fixed `p_stop = 0.5` | 0 |
| `budget_aware_stop` | `p_stop = sigmoid(a + b * budget)` | 2 |
| `learned_stop` | `p_stop = sigmoid(MLP(features))` | `model_dim + 6` × hidden |

---

## 5. T7-Primary Trajectory

The T7-primary trajectory can select both 5'UTR and CDS edits in a single trajectory:

```
source
  → FIVE_UTR_SUB(pos=10, nt=G)     [5'UTR edit]
  → CDS_SYNONYMOUS_SUB(codon=5, target=GCC)  [CDS edit]
  → FIVE_UTR_SUB(pos=25, nt=A)     [5'UTR edit]
  → STOP
```

This is enabled by the hierarchical policy's Level 2 (region selection), which can switch regions across steps.

---

## 6. Acceptance Criteria Mapping

| Criterion | How Verified |
|-----------|-------------|
| Protein identity 100% | `apply_edit_action` asserts `translate(source.cds) == translate(candidate.cds)` |
| Transcript length 100% unchanged | No indels in action space; `apply_edit_action` preserves length |
| T7-primary trajectory can select 5'UTR/CDS | Hierarchical policy Level 2 allows region switching |
| Learned STOP trainable | STOP MLP with gradient flow |
| Reward is source-normalized | `build_training_reward` with `source_baseline` |
| Uncertainty enters risk adjustment | `LCB = mean - λ × uncertainty` in reward |
| Training/independent Oracle separation | `ProductionPathGate` + `RewardComponent.independent` flag |
| No single-nt CDS nonsynonymous intermediate | `CDS_SYNONYMOUS_SUB` is atomic |
| Action/reward provenance complete | `reward_provenance` dict + `RewardComponent.source_model` |

---

## 7. Implementation Plan

1. **`rl/p3_06_mdp.py`**: `MDPState`, `EditAction`, `apply_edit_action`, `build_legal_edit_actions`
2. **`rl/p3_06_policy.py`**: `HierarchicalPolicy` with 4-level decomposition
3. **`rl/p3_06_stop.py`**: `LearnableStop` with 3 variants (constant/budget-aware/learned)
4. **`rl/p3_06_reward.py`**: `RewardV3` with LCB primary + secondary terms
5. **`tests/test_p3_06_mdp.py`**: Acceptance tests
