# P3-05: Source-Candidate Delta Architecture

**Date**: 2026-07-23
**Status**: DESIGN COMPLETE — implementation pending
**Phase**: P3-05
**Spec**: `提示词/mrna的 rl 的后续优化的分阶段提示词.md#L1528-1612`

---

## 1. Motivation

The existing `MRNAEditFormer` is an **absolute sequence model** — it takes a single
sequence and predicts edit rates. It does not explicitly compare source vs. candidate.

For local-edit-effect prediction, the model must see:
- What the sequence looked like **before** editing (source)
- What it looks like **after** editing (candidate)
- **Which positions** changed
- **Which region** each change belongs to
- Whether the change is **synonymous** (protein-preserving)

This document specifies 6 delta-architecture variants to be compared.

---

## 2. Architecture Variants

### Variant A: Candidate-Only (baseline)

```
source → discarded
candidate → backbone → encoder → Δprediction
```

- **Input**: candidate sequence only
- **Pros**: Simplest, fewest parameters
- **Cons**: Cannot distinguish "good candidate" from "good source" — no reference point
- **Hypothesis**: Worst delta prediction because it lacks source context

### Variant B: Source-Only

```
candidate → discarded
source → backbone → encoder → Δprediction
```

- **Input**: source sequence only
- **Pros**: Knows the starting point
- **Cons**: Cannot see what changed — must infer edit effects from source alone
- **Hypothesis**: Poor delta prediction — cannot model edit-specific effects

### Variant C: Source + Candidate Concatenation

```
[source | candidate] → backbone → encoder → Δprediction
```

- **Input**: source and candidate concatenated with a separator token
- **Pros**: Backbone sees both sequences, can learn cross-position relationships
- **Cons**: Doubles sequence length (2L); quadratic attention cost (4× for dense); backbone may not generalize to concatenated format
- **Hypothesis**: Better than A/B but inefficient and may suffer from position-shift artifacts

### Variant D: Siamese Difference (recommended)

```
source → backbone → encoder_s → h_source
candidate → backbone → encoder_c → h_candidate
h_delta = h_candidate - h_source  (or learned gating)
h_delta → delta_head → Δprediction
```

- **Input**: source and candidate processed independently by shared backbone + encoder
- **Difference**: Element-wise subtraction (or learned multiplicative gating)
- **Pros**: Weight sharing (backbone sees single sequences); explicit delta representation; backbone pretraining transferable
- **Cons**: Cannot model position-level interactions between source and candidate
- **Hypothesis**: Strong baseline — explicitly models "what changed" at feature level

### Variant E: Edit-Token Conditioned

```
source → backbone → encoder → h_source
edit_tokens = [(pos, ref_nt, alt_nt, region, synonymous)] → edit_encoder → h_edit
h_combined = h_source + h_edit (broadcast to edited positions)
h_combined → delta_head → Δprediction
```

- **Input**: source sequence + sparse edit tokens (position, ref, alt, region, synonymous flag)
- **Pros**: Explicit edit representation; O(L + k) where k = number of edits (k ≪ L); naturally handles variable edit counts
- **Cons**: Requires explicit edit list; doesn't see full candidate sequence (may miss emergent effects)
- **Hypothesis**: Strong for few-edit scenarios (k ≤ 5); may miss long-range structural effects

### Variant F: Cross-Attention Source ↔ Candidate

```
source → backbone → encoder → h_source (queries)
candidate → backbone → encoder → h_candidate (keys/values)
cross_attn(h_source, h_candidate) → h_cross
h_cross → delta_head → Δprediction
```

- **Input**: source and candidate processed independently, then cross-attention
- **Pros**: Models position-level source↔candidate interactions; attention weights are interpretable (show which source positions attend to which candidate positions)
- **Cons**: O(L²) cross-attention cost; most parameters; may overfit on small datasets
- **Hypothesis**: Best performance but most expensive; attention weights provide interpretability

---

## 3. Recommended Primary Architecture: Variant D + E Hybrid

```
                    ┌──────────────────────────────────────┐
                    │         Frozen Backbone               │
                    │  (Orthrus 6-track, Mamba SSM)        │
source ────────────►│                                      │
                    │  per-token: h_source ∈ R^{L×512}     │
                    └──────────┬───────────────────────────┘
                               │
                    ┌──────────▼───────────────────────────┐
                    │    Source Encoder (trainable)         │
                    │  Linear(512, model_dim) + LayerNorm  │
                    │  + region FiLM + phase embedding      │
                    └──────────┬───────────────────────────┘
                               │  h_src ∈ R^{L×d}
                               │
candidate ───┬─────────────────┼──────────────────────────────
             │                 │
             ▼                 ▼
    ┌────────────────┐  ┌──────────────────┐
    │ Edit Encoder   │  │ Candidate Encode │
    │ (sparse tokens)│  │ Linear(512,d)    │
    │ pos_emb +      │  │ + region FiLM    │
    │ ref/alt emb +  │  └────────┬─────────┘
    │ region emb +   │           │ h_cand ∈ R^{L×d}
    │ synonymous emb │           │
    └───────┬────────┘           │
            │ h_edit ∈ R^{k×d}   │
            │                    │
            ▼                    ▼
    ┌─────────────────────────────────────────┐
    │     Difference Module                   │
    │  h_delta = h_cand - h_src               │
    │  h_delta[edited_pos] += h_edit          │
    │  h_delta → LayerNorm → dropout          │
    └─────────────────┬───────────────────────┘
                      │
    ┌─────────────────▼───────────────────────┐
    │     Delta Head                          │
    │  MLP(d → d//2 → 1) + uncertainty head   │
    │  Output: Δpredicted, σ_uncertainty      │
    └─────────────────────────────────────────┘
```

### Key Design Choices

1. **Siamese backbone (shared weights)**: source and candidate both pass through the same frozen Orthrus backbone, ensuring consistent feature space
2. **Explicit difference**: `h_delta = h_cand - h_src` — the model literally sees "what changed" at feature level
3. **Edit-token augmentation**: sparse edit tokens (position, ref, alt, region, synonymous) are added to the delta at edited positions — provides explicit edit metadata
4. **Uncertainty head**: separate output for prediction uncertainty (used by `build_training_reward` for risk adjustment)
5. **Region FiLM**: region and phase information injected via FiLM modulation (from existing `MRNAEditFormer`)

---

## 4. Phase-Aware CDS Encoding

### 4.1 Problem

The existing code uses `codon_phases()` which returns per-position phase (0, 1, 2).
But BOS tokens and 5'UTR length can shift the phase if not properly aligned.

### 4.2 Solution

```python
# In MRNARecord (core/schema.py):
def codon_phases(self) -> List[int]:
    """Return codon phase for each position.

    Phase is relative to CDS start (cds_start), NOT sequence start.
    5'UTR and 3'UTR positions get PHASE_NONE (-1).
    CDS positions get (pos - cds_start) % 3.
    """
    phases = []
    for i in range(len(self.seq)):
        if self.cds_start <= i < self.cds_end:
            phases.append((i - self.cds_start) % 3)
        else:
            phases.append(PHASE_NONE)
    return phases
```

### 4.3 Requirements (from spec)

- CDS codon boundary aligned to real `cds_start` ✓
- BOS and 5'UTR length do not change codon phase ✓ (phase is relative to cds_start)
- Codon representation maps only to CDS codon ✓ (UTR gets PHASE_NONE)
- UTR does not perform pseudo codon pooling ✓
- Synonymous candidate set from genetic code ✓ (uses `SYNONYMOUS_CODONS` from `core/constants.py`)

### 4.4 Codon Representation

For CDS positions, in addition to nucleotide embedding, a **codon embedding** is added:
```python
codon_id = (pos - cds_start) // 3  # which codon (0, 1, 2, ...)
codon_emb = nn.Embedding(max_codons, model_dim)(codon_id)
h = h + codon_emb  # additive
```

This lets the model know which codon each CDS position belongs to, enabling synonymous-aware representations.

---

## 5. Input Requirements

The model must explicitly see:

| Information | How Provided |
|-------------|-------------|
| Edit前是什么 (source) | Source sequence → backbone → h_source |
| Edit后是什么 (candidate) | Candidate sequence → backbone → h_candidate |
| 改了哪些位置 (which positions) | Edit tokens with position embedding |
| 属于哪个区域 (which region) | Region FiLM (5'UTR / CDS / 3'UTR) |
| 是否为同义密码子 (synonymous) | Synonymous flag in edit tokens |

---

## 6. Training Protocol

### 6.1 Loss Function

```python
L = L_delta + λ_uncertainty * L_uncertainty + λ_kl * L_kl

# L_delta: MSE or Huber loss on predicted vs true delta
# L_uncertainty: Gaussian NLL (−log p(y|x, σ))
# L_kl: KL divergence to reference policy (for RL compatibility)
```

### 6.2 Data

- **Source**: P3-01 local-edit benchmark (measured + proxy + literature + synthetic tiers)
- **Split**: mmseqs-based group-aware split (frozen in P3-01)
- **Augmentation**: None (edits are real, not augmented)

### 6.3 Ablation Plan

| Experiment | Variant | Backbone | Purpose |
|-----------|---------|----------|---------|
| D-orthrus | D (Siamese) | Orthrus 6-track | Primary |
| D-light | D (Siamese) | `none` (light) | Backbone contribution |
| D-rna-fm | D (Siamese) | RNA-FM | ncRNA control |
| D-random | D (Siamese) | Random Orthrus | Negative control |
| A-orthrus | A (Candidate-only) | Orthrus 6-track | Source context value |
| C-orthrus | C (Concat) | Orthrus 6-track | Concat vs Siamese |
| E-orthrus | E (Edit-token) | Orthrus 6-track | Sparse edit value |
| F-orthrus | F (Cross-attn) | Orthrus 6-track | Interaction value |

---

## 7. Acceptance Criteria

| Criterion | Threshold | Test |
|-----------|-----------|------|
| Source-aware > candidate-only | Spearman ρ(D) > ρ(A) by ≥ 0.05 | Paired bootstrap test |
| Local-delta metric improvement | MAE(D) < MAA(baseline) by ≥ 10% | P3-02 benchmark |
| Uncertainty calibration | ECE < 0.15 | Reliability diagram |
| 4096nt smoke train | Forward + backward completes | OOM check |
| No length-bucket collapse | Spearman ρ consistent across length buckets | Per-bucket evaluation |

---

## 8. Unresolved Risks

1. **Orthrus context limit (1024nt)**: Sequences > 1024nt require sliding-window aggregation. Strategy: mean-pool overlapping windows with cosine weighting.
2. **Backbone freeze vs. fine-tune**: Default is frozen. If frozen features are insufficient, consider LoRA fine-tuning of the last 4 Mamba blocks.
3. **Edit-token padding**: Variable edit counts (k) require padding or set-based pooling. Strategy: transformer-style positional padding with attention mask.
4. **Cross-attention cost (Variant F)**: O(L²) may be prohibitive for 4096nt. If selected, use sparse/local attention.
