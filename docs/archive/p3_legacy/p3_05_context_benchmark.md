# P3-05: Context Length and Structure Benchmark

**Date**: 2026-07-23
**Status**: DESIGN COMPLETE — pending implementation
**Phase**: P3-05
**Spec**: `提示词/mrna的 rl 的后续优化的分阶段提示词.md#L1630-1698`

---

## 1. Context Length Requirements

### 1.1 Primary Task Minimum

The primary task (source-conditioned minimal-edit) must support **4096 nt** sequences.

Typical mRNA lengths:
- 5'UTR: 50–200 nt
- CDS: 300–4000 nt (100–1300 codons)
- 3'UTR: 100–1000 nt
- Total: 500–5000 nt

4096 nt covers the vast majority of therapeutic mRNA constructs (e.g., EGFP ~850 nt, Cas9 ~4500 nt).

### 1.2 Extension to 8192 nt

Optional, determined by data length distribution. Not a forced innovation point. Required only if > 10% of P3-01 benchmark sequences exceed 4096 nt.

---

## 2. Long-Context Strategies

Five strategies are compared for handling sequences up to 4096 nt:

### Strategy 1: Dense Attention (baseline)

- **Cost**: O(L²) — 4096² = 16.7M attention entries per head per layer
- **Memory**: ~64MB per layer (float32, single head, L=4096)
- **Pros**: Full receptive field, no information loss
- **Cons**: Prohibitive for L > 4096; 4× cost vs 2048
- **Feasibility at 4096**: Marginal on single GPU with gradient checkpointing

### Strategy 2: Local Attention (sliding window)

- **Window**: w = 512 nt (covers ~170 codons or ~1 domain)
- **Cost**: O(L * w) — 4096 * 512 = 2.1M entries (8× cheaper than dense)
- **Pros**: Efficient; captures local context (codon environment, local structure)
- **Cons**: No long-range dependencies; misses global sequence effects
- **Feasibility**: Easy to implement; standard in Longformer/BigBird

### Strategy 3: Codon Compression

- **Mechanism**: Pool CDS positions into codon-level tokens (3:1 compression)
- **Effective length**: 5'UTR + CDS/3 + 3'UTR ≈ 200 + 1300 + 200 = 1700 tokens
- **Cost**: O(1700²) = 2.9M entries (5.8× cheaper than dense)
- **Pros**: Biologically motivated (codon is the functional unit); large reduction
- **Cons**: Loses nucleotide-resolution in CDS; may miss single-nt effects
- **Variant**: Hybrid — nucleotide resolution in edited region, codon-compressed elsewhere

### Strategy 4: Hierarchical Summaries

- **Level 0**: Nucleotide tokens (full resolution)
- **Level 1**: Regional summaries (5'UTR, CDS start, CDS middle, CDS end, 3'UTR) — 5 tokens
- **Level 2**: Global summary — 1 token
- **Mechanism**: Bottom-up pooling, top-down broadcast
- **Cost**: O(L²) for local + O(k²) for summary (k ≈ 5–10)
- **Pros**: Captures both local and global context; biologically interpretable
- **Cons**: Summary tokens may lose fine-grained information; requires careful pooling

### Strategy 5: SSM Hybrid (recommended)

- **Mechanism**: Mamba SSM blocks (O(L) cost) for long-range + local self-attention (O(L * w)) for edited regions
- **Cost**: O(L) + O(L * w) ≈ O(L * w) — same as local attention but with SSM's global receptive field
- **Pros**: Mamba's selective state-space naturally handles long sequences; attention focuses on edit neighborhoods
- **Cons**: Requires Mamba implementation (already available via Orthrus backbone)
- **Feasibility**: Orthrus uses Mamba natively — can reuse backbone's SSM blocks for the encoder

### 2.1 Comparison Table

| Strategy | Cost (L=4096) | Global Receptive | Local Precision | Implementation | Recommended |
|----------|---------------|-------------------|-----------------|----------------|-------------|
| Dense | O(L²) = 16.7M | ✓ | ✓ | Easy | Marginal |
| Local (w=512) | O(L*w) = 2.1M | ✗ | ✓ | Easy | Fallback |
| Codon compress | O((L/3)²) = 2.9M | ✓ (CDS) | ✗ (CDS) | Medium | For CDS-only |
| Hierarchical | O(L*w) + O(k²) | ✓ (summary) | ✓ (local) | Hard | If data warrants |
| **SSM Hybrid** | **O(L) + O(L*w)** | **✓ (SSM)** | **✓ (attn)** | **Medium** | **Primary** |

---

## 3. Structure and Local Context Inputs

### 3.1 Required Inputs (from spec)

| Input | Source | Granularity | Role |
|-------|--------|-------------|------|
| Start-region structure | ViennaRNA fold of first 80 nt | Per-position probability | Input feature |
| Local edit-window structure | ViennaRNA fold of ±50 nt around each edit | Per-position probability | Input feature |
| Full-sequence coarse structure | ViennaRNA fold (sliding window 300 nt, stride 100) | Regional summary | Auxiliary task |
| Codon context | Codon identity ± 2 codons around edit | Per-codon embedding | Input feature |
| Kozak/uAUG/uORF features | Sequence scan for Kozak motif, upstream AUGs, uORFs | Binary/positional flags | Input feature |

### 3.2 Structure as Input vs. Reward

**Design principle** (from spec): "结构特征应作为输入或辅助任务，不得未经验证直接作为主 reward。"

- **As input**: Structure probabilities (pairing probability, accessibility) are concatenated to backbone embeddings
- **As auxiliary task**: A separate head predicts MFE proxy and start-accessibility (existing `MRNAEditFormer` `aux` head with `use_aux_struct`)
- **NOT as reward**: Structure features do NOT enter the reward function until validated by P3-03 prospective assay

### 3.3 Feature Construction

```python
# For each position i in the sequence:
features[i] = concat(
    backbone_embedding[i],        # R^{512} from Orthrus
    region_embedding[region[i]],  # R^{d_region}
    phase_embedding[phase[i]],    # R^{d_phase} (CDS only)
    pairing_prob[i],              # R^{1} from ViennaRNA
    accessibility[i],             # R^{1} from ViennaRNA
    kozak_score[i],               # R^{1} (highest at Kozak motif positions)
    uaug_flag[i],                 # R^{1} (1 if upstream AUG)
    uorf_flag[i],                 # R^{1} (1 if in uORF)
)
# Total: 512 + d_region + d_phase + 5
```

---

## 4. Length-Bucket Collapse Check

### 4.1 Problem

Models may collapse on certain length buckets (e.g., perform well on short sequences but fail on long ones).

### 4.2 Buckets

| Bucket | Length Range | Expected % of P3-01 |
|--------|-------------|---------------------|
| Short | < 500 nt | ~15% |
| Medium | 500–1500 nt | ~40% |
| Long | 1500–3000 nt | ~30% |
| X-Long | 3000–4096 nt | ~10% |
| OOB | > 4096 nt | ~5% (excluded from primary) |

### 4.3 Collapse Detection

- Compute Spearman ρ per bucket
- **Collapse threshold**: if any bucket's ρ is < 50% of the best bucket's ρ
- **No-collapse criterion**: all buckets within 20% of each other

---

## 5. 4096 nt Smoke Test Protocol

### 5.1 Forward Pass Smoke Test

```python
# Construct a 4096 nt test sequence
record = MRNARecord(
    five_utr="A" * 100,
    cds=START_CODON + "GCU" * 1300 + "UAA",  # ~3900 nt CDS
    three_utr="U" * 90,
)
# Total: ~4096 nt

# Forward pass
backbone = load_orthrus_6track()
delta_model = DeltaArchitecture(backbone=backbone)
source = record
candidate = apply_codon_action(record, synonymous_codon_actions(record.cds)[0])

# Must complete without OOM on single GPU (12GB+)
predicted_delta, uncertainty = delta_model(source, candidate)
assert predicted_delta.shape == (1,)
assert uncertainty.shape == (1,)
```

### 5.2 Backward Pass Smoke Test

```python
loss = mse_loss(predicted_delta, torch.tensor([0.5]))
loss.backward()
# Must complete without OOM
# Gradients must flow to delta head (not backbone, which is frozen)
for name, param in delta_model.named_parameters():
    if "backbone" not in name:
        assert param.grad is not None, f"No gradient for {name}"
```

### 5.3 Multi-Length Smoke Test

Run forward+backward for sequences of length 512, 1024, 2048, 4096 nt.
All must complete without OOM on a single 12GB GPU.

---

## 6. Acceptance Criteria

| Criterion | Threshold | Method |
|-----------|-----------|--------|
| 4096 nt smoke train runnable | Forward + backward completes on 12GB GPU | §5 |
| No length-bucket collapse | All buckets within 20% ρ | §4.3 |
| Structure input improves delta | ρ(with structure) > ρ(without) by ≥ 0.02 | Ablation |
| SSM hybrid ≤ dense attention | ρ(SSM) ≥ ρ(dense) − 0.02 | Strategy comparison |
| Memory < 10GB at 4096 nt | Peak GPU memory < 10GB | Memory profiling |

---

## 7. Implementation Priority

1. **SSM Hybrid encoder** (Strategy 5) — primary, reuses Orthrus Mamba blocks
2. **Structure feature pipeline** — ViennaRNA integration for pairing/accessibility
3. **Codon context embedding** — ±2 codon window
4. **Kozak/uAUG/uORF scanner** — sequence-level feature extraction
5. **Length-bucket evaluation** — per-bucket Spearman ρ
6. **Local attention fallback** (Strategy 2) — if SSM hybrid fails on memory

---

## 8. Unresolved Risks

1. **ViennaRNA cost**: Folding 4096 nt sequences is O(L³) ≈ 70B operations. Use sliding window (300 nt, stride 100) for coarse structure; local fold (±50 nt) for edit-window structure.
2. **Orthrus 1024 nt limit**: Sequences > 1024 nt require sliding-window backbone inference. Overlap strategy: 512 nt windows with 256 nt overlap, cosine-weighted averaging.
3. **GPU memory**: Dense attention at 4096 nt requires ~4GB per layer (float32). With 8 layers, this is ~32GB — too much. SSM hybrid or gradient checkpointing is mandatory.
4. **Codon compression information loss**: Pooling CDS to codon level loses single-nt resolution. Mitigated by using nucleotide resolution in the edited region and codon compression only for non-edited CDS.
