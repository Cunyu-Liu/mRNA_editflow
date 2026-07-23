# P3-05: Real Backbone Selection

**Date**: 2026-07-23
**Status**: DESIGN COMPLETE — pending checkpoint download and integration
**Phase**: P3-05
**Spec**: `提示词/mrna的 rl 的后续优化的分阶段提示词.md#L1564-1588`

---

## 1. Selection Criteria

A real mRNA-native backbone must satisfy:

1. **mRNA-native pretraining**: trained on mRNA transcripts (not only ncRNA)
2. **Per-token embeddings**: returns nucleotide-resolution features (not only pooled)
3. **Publicly available checkpoint**: downloadable from HuggingFace or official repo
4. **Compatible license**: MIT / Apache-2.0 / CC-BY for academic use
5. **Reasonable context length**: ≥ 1024 nt (for full-length mRNA segments)
6. **CDS-aware (preferred)**: encodes coding sequence structure or phase information

---

## 2. Selected Backbone: Orthrus 6-track

### 2.1 Description

| Field | Value |
|-------|-------|
| **Model** | Orthrus (Mamba-based RNA foundation model) |
| **Version** | `orthrus-6-track` (Nature Methods publication) |
| **HuggingFace repo** | `antichronology/orthrus-6-track` |
| **Architecture** | Mamba SSM (selective state-space model) |
| **Parameters** | ~86M (6-track, embed_dim=512) |
| **Pretraining** | Contrastive learning on 45M RNA transcripts from 10 species + 400+ mammalian orthologs (Zoonomia) |
| **Tracks** | 6-track: nucleotide + splicing donor + splicing acceptor + CDS + 5'UTR + 3'UTR |
| **Context length** | 1024 nt (sliding window for longer sequences) |
| **License** | MIT |
| **Tokenizer** | Nucleotide-level (ACGU), one-hot encoded tracks |
| **Citation** | Fradkin et al., "Orthrus: toward evolutionary and functional RNA foundation models", Nature Methods, 2026. DOI: 10.1038/s41592-026-03064-3 |

### 2.2 Selection Rationale

- **mRNA-native**: pre-trained on mature mRNA transcripts, not just ncRNA
- **CDS-aware**: 6-track version includes CDS, 5'UTR, 3'UTR binary tracks — directly encodes region information
- **Per-token output**: `representation_unpooled()` returns `(B, L, 512)` — nucleotide-resolution features
- **Mamba SSM**: O(L) complexity (vs O(L²) for transformers), naturally handles long sequences
- **Contrastive pretraining**: structures latent space by functional/evolutionary similarity — transferable to edit-effect prediction
- **HuggingFace integration**: `AutoModel.from_pretrained(..., trust_remote_code=True)` — standardized loading
- **MIT license**: no commercial restriction

### 2.3 Loading Protocol

```python
from transformers import AutoModel
import torch

model = AutoModel.from_pretrained(
    "antichronology/orthrus-6-track",
    trust_remote_code=True,
)
model.eval()

# Per-token embeddings: (B, L, 512)
# x: (B, L, 6) one-hot encoded tracks
embeddings = model.representation_unpooled(x, channel_last=True)

# Pooled: (B, 512)
pooled = model.representation(x, lengths, channel_last=True)
```

### 2.4 Checkpoint Metadata (to be filled after download)

| Field | Value |
|-------|-------|
| checkpoint_source | `huggingface:antichronology/orthrus-6-track` |
| checkpoint_SHA256 | `PENDING_DOWNLOAD` |
| tokenizer | nucleotide-level, 6-track one-hot |
| version | Nature Methods publication (2026) |
| license | MIT |
| pretraining_corpus | 45M mature mRNA transcripts, 10 species + 400+ mammalian orthologs |
| pretraining_objective | Contrastive (isoform + ortholog similarity) |
| embed_dim | 512 |
| n_layers | 24 (Mamba blocks) |
| max_context | 1024 nt |

---

## 3. Control Backbones

### 3.1 Lightweight Task-Specific Encoder (positive control)

| Field | Value |
|-------|-------|
| **Name** | `none` (from-scratch light embedding) |
| **Source** | Existing `models/backbones.py` `FrozenBackbone(name="none")` |
| **Architecture** | Token embedding + region embedding + fixed sinusoidal position |
| **Parameters** | ~50K (embedding tables only) |
| **Purpose** | Validates that delta architecture improvements come from source-candidate modeling, not from backbone capacity |

### 3.2 ncRNA Foundation Encoder (cross-domain control)

| Field | Value |
|-------|-------|
| **Name** | RNA-FM |
| **HuggingFace repo** | `ml4bio/RNA-FM` |
| **Architecture** | Transformer (24 layers, 640 dim) |
| **Pretraining** | MLM on 23M ncRNA sequences |
| **License** | MIT |
| **Purpose** | Tests whether ncRNA-pretrained features transfer to mRNA edit-effect prediction (they should be weaker than mRNA-native) |

### 3.3 Random/Frozen Placeholder (negative control)

| Field | Value |
|-------|-------|
| **Name** | `random_orthrus` |
| **Architecture** | Same Mamba SSM as Orthrus 6-track, but randomly initialized (no pretraining) |
| **Parameters** | Same ~86M, frozen |
| **Purpose** | Negative control — if this performs as well as pretrained Orthrus, the backbone provides no useful inductive bias |
| **Paper mode** | **REJECTED** — `ProductionPathGate.load_for_paper()` raises `ValueError` for random-initialized backbones |

---

## 4. Backbone Comparison Protocol

### 4.1 Fairness Controls

All backbones are:
- **Frozen** during delta-model training (gradients flow only through delta heads)
- **Projected** to a common `model_dim` via a trainable linear layer
- **Evaluated** on the same P3-01 local-edit benchmark test split

### 4.2 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Spearman ρ (delta) | Rank correlation between predicted and true edit deltas |
| MAE (delta) | Mean absolute error of predicted delta |
| Beneficial-edit enrichment | Fraction of top-k predicted edits that are truly beneficial |
| Calibration ECE | Expected calibration error of uncertainty estimates |
| 4096nt smoke test | Forward pass completes without OOM on single GPU |

### 4.3 Expected Outcome

| Backbone | Expected Performance | Rationale |
|----------|---------------------|-----------|
| Orthrus 6-track | **Best** | mRNA-native + CDS-aware + contrastive pretraining |
| RNA-FM | Moderate | ncRNA pretraining, no CDS track, transformer O(L²) |
| `none` (light) | Moderate | No pretraining but trainable from scratch on task data |
| `random_orthrus` | Worst | Same architecture but no useful features |

---

## 5. Paper-Mode Gate

```python
# In ProductionPathGate (rl/p3_04_correctness.py):
# load_for_paper() checks:
# 1. backbone_name in MRNA_NATIVE_BACKBONES (not "none", not random)
# 2. checkpoint_sha256 != "RANDOM_INIT"
# 3. backbone frozen (requires_grad=False for all backbone params)
# 4. delta_head trained (requires_grad=True for delta head params)
```

Paper results may ONLY use Orthrus 6-track (or a future mRNA-native backbone that passes the same gate).

---

## 6. Integration with Existing Code

The existing `models/backbones.py` already has an ADAPTER-STUB for `orthrus` that:
- Documents how to load the real HF checkpoint from `weights_path`
- Falls back to a deterministic placeholder encoder when weights are unavailable

**P3-05 action**: Replace the stub with the real Orthrus adapter that calls `AutoModel.from_pretrained`. The adapter interface (`embed(token_ids, region_ids, padding_mask) -> Tensor[B, L, out_dim]`) remains unchanged.

### 6.1 Track Construction

Orthrus 6-track requires 6 binary tracks per position:
1. Nucleotide identity (ACGU → one-hot 4D)
2. Splice donor (binary)
3. Splice acceptor (binary)
4. CDS (binary: 1 if in CDS, 0 otherwise)
5. 5'UTR (binary)
6. 3'UTR (binary)

For mRNA-EditFlow, tracks 2-3 (splicing) are set to 0 (mature mRNA has no introns). Tracks 4-6 are derived from `MRNARecord.region_ids()`.

---

## 7. Unresolved Risks

1. **Download pending**: Orthrus 6-track checkpoint (~340MB) must be downloaded and SHA-256 verified before any training.
2. **GenomeKit dependency**: 6-track input construction may require `genomekit` package — needs separate venv installation per project constraint.
3. **Context length**: Orthrus supports 1024 nt; sequences > 1024 nt need sliding-window aggregation. The delta architecture must handle this.
4. **License compatibility**: MIT is compatible with the project's academic use case, but commercial use of derived models must be verified.
