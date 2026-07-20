# P1-11 Long-View Reconstruction Attrition Report

**Status**: Complete.
**Date**: 2026-07-19
**Author**: trae agent (autonomous execution)
**Script**: [scripts/run_long_view_reconstruction.py](file:///home/cunyuliu/mrna_editflow_goal/mrna_editflow/scripts/run_long_view_reconstruction.py)
**Top-level manifest SHA-256**: `5856cb2003130ce09c9a0babee0be4c682671ca1f8457325d4119f09efe23143`

---

## 1. Overview

This report documents the **P1-11 long-view reconstruction**: deriving a new model view from the v1 frozen canonical records with **4× larger caps** than v1's `model_capped_v1` view.

| View | max_5utr | max_cds | max_3utr | Source |
|---|---|---|---|---|
| v1 `model_capped_v1` | 128 | 1536 | 256 | P0 Data Reconstruction v1 |
| **P1 `long_view`** (this work) | **512** | **3072** | **1024** | v1 canonical (read-only) |

**Motivation**: The v1 `model_capped_v1` view truncates therapeutically relevant full-length mRNAs (e.g., 5'UTR p90=487 nt exceeds the 128 cap; CDS p90=2880 nt exceeds the 1536 cap; 3'UTR p90=3256 nt exceeds the 256 cap). The long-view caps preserve more of the natural sequence distribution, enabling the P1-13 counterfactual panel and future training to operate on realistic full-length mRNAs.

**Constraint honored**: The v1 frozen namespace (`data/reconstructed/p0_data_reconstruction_v1/`) is **read-only** — no files were modified. All long-view artifacts are written to `data/reconstructed/p1_long_view/`.

---

## 2. Truncation Policy

Matches v1 `model_capped_v1` (see [data/reconstruction.py](file:///home/cunyuliu/mrna_editflow_goal/mrna_editflow/data/reconstruction.py)):

| Region | Policy | Rationale |
|---|---|---|
| 5'UTR | Keep the **last** `max_5utr` nt | Closest to the start codon (most functionally relevant for translation initiation) |
| CDS | Keep the **first** `max_cds` nt | From the start codon (preserves coding frame and N-terminal protein) |
| 3'UTR | Keep the **first** `max_3utr` nt | After the stop codon (most regulatory elements are proximal) |

---

## 3. Attrition Statistics

### 3.1 GENCODE v45

| Metric | Value | % of total |
|---|---|---|
| Total canonical records | 80,290 | 100.0% |
| **Kept (all 3 regions non-empty)** | **76,208** | **94.9%** |
| Dropped: empty 5'UTR | 3,130 | 3.9% |
| Dropped: empty CDS | 0 | 0.0% |
| Dropped: empty 3'UTR | 952 | 1.2% |

**Truncation rates** (among kept records):

| Region | Truncated | % of kept |
|---|---|---|
| 5'UTR (> 512 nt) | 6,901 | 9.1% |
| CDS (> 3072 nt) | 6,693 | 8.8% |
| 3'UTR (> 1024 nt) | 31,383 | 41.2% |

**Pre-truncation length percentiles** (among kept records):

| Region | p50 | p90 | p99 |
|---|---|---|---|
| 5'UTR | 153 | 487 | 1,213 |
| CDS | 1,038 | 2,880 | 7,104 |
| 3'UTR | 741 | 3,256 | 7,624 |

**Observation**: The 3'UTR cap of 1024 nt truncates 41.2% of records, reflecting that human 3'UTRs are frequently long (median 741, p90 3256). This is a known biological property. The 5'UTR cap of 512 nt only truncates 9.1% (p90 = 487, just under the cap).

### 3.2 RefSeq Human RNA

| Metric | Value | % of total |
|---|---|---|
| Total canonical records | 197,627 | 100.0% |
| **Kept (all 3 regions non-empty)** | **195,669** | **99.0%** |
| Dropped: empty 5'UTR | 1,361 | 0.7% |
| Dropped: empty CDS | 0 | 0.0% |
| Dropped: empty 3'UTR | 597 | 0.3% |

**Truncation rates** (among kept records):

| Region | Truncated | % of kept |
|---|---|---|
| 5'UTR (> 512 nt) | 40,379 | 20.6% |
| CDS (> 3072 nt) | 38,564 | 19.7% |
| 3'UTR (> 1024 nt) | 104,849 | 53.6% |

**Pre-truncation length percentiles** (among kept records):

| Region | p50 | p90 | p99 |
|---|---|---|---|
| 5'UTR | 227 | 826 | 5,428 |
| CDS | 1,611 | 4,266 | 9,555 |
| 3'UTR | 1,167 | 4,392 | 10,183 |

**Observation**: RefSeq has longer 5'UTRs (p50=227 vs GENCODE 153) and longer CDS (p50=1611 vs 1038), likely due to different transcript selection criteria. The 3'UTR truncation rate (53.6%) is higher than GENCODE (41.2%), consistent with RefSeq's longer 3'UTR distribution.

### 3.3 Combined View

| Source | Kept records |
|---|---|
| GENCODE v45 | 76,208 |
| RefSeq Human RNA | 195,669 |
| **Combined** | **271,877** |

This is a **7.5× increase** over the v1 `model_capped_v1` combined view (which had ~36,204 records after the 128/256/1536 caps). The long-view reconstruction provides a substantially larger and more representative wild-type pool for the P1-13 counterfactual panel and future training.

---

## 4. Artifacts

### 4.1 Per-source artifacts

| File | SHA-256 | Description |
|---|---|---|
| `sources/gencode_v45/long_view.records.jsonl` | `13f3adec3353f7e78fc769a0ccfd76d523153945a4dd72ce862e14509cee6d9a` | 76,208 truncated records |
| `sources/gencode_v45/long_view.lineage.jsonl` | `030549e0a3a5230135f1ec64d4b388dce4d9b975d836e324f8f536c25aa34f9f` | Lineage (canonical → derived SHA-256 mapping) |
| `sources/gencode_v45/long_view_manifest.json` | (in top manifest) | Per-source manifest with stats |
| `sources/refseq_human_rna/long_view.records.jsonl` | (in top manifest) | 195,669 truncated records |
| `sources/refseq_human_rna/long_view.lineage.jsonl` | (in top manifest) | Lineage |
| `sources/refseq_human_rna/long_view_manifest.json` | (in top manifest) | Per-source manifest with stats |

### 4.2 Combined artifacts

| File | SHA-256 | Description |
|---|---|---|
| `combined/long_view.records.jsonl` | `1c4e1f183173fb5d4d8a6efd3c31437e3ff66b7f224bedcb7b6fa40aed288be4` | 271,877 merged records |
| `combined/long_view_attrition_report.json` | `a0a687904e047c4829592fa7a17b0d87806c5c84ef72fb5dfdaffca586f7069e` | Combined attrition report |
| `p1_long_view_manifest.json` | `5856cb2003130ce09c9a0babee0be4c682671ca1f8457325d4119f09efe23143` | Top-level manifest |

### 4.3 v1 frozen namespace (read-only, not modified)

| File | SHA-256 (recorded in P1 manifest) | Description |
|---|---|---|
| `p0_data_reconstruction_v1/sources/gencode_v45/canonical.records.jsonl` | `1c71880953119dc321972efcb4a9c82539d00f2f74b77b118ddde832eb31100d` | 80,290 canonical records |
| `p0_data_reconstruction_v1/sources/refseq_human_rna/canonical.records.jsonl` | `e620b458dbe4f480f45e6576c45e049da36fac43d06bc8b25594580d9484ee82` | 197,627 canonical records |

---

## 5. Reproducibility

### 5.1 Command

```bash
cd /home/cunyuliu/mrna_editflow_goal
PYTHONPATH=/home/cunyuliu/mrna_editflow_goal \
/home/cunyuliu/miniconda3/envs/editflow/bin/python \
  mrna_editflow/scripts/run_long_view_reconstruction.py
```

### 5.2 Environment

- Python: 3.x (conda env `editflow`)
- No GPU required (data processing only)
- Runtime: ~30 seconds for 277,917 input records

### 5.3 Determinism

- Input: v1 frozen canonical records (SHA-256 recorded)
- Caps: hardcoded in script (`LONG_VIEW_CAPS = {max_5utr: 512, max_cds: 3072, max_3utr: 1024}`)
- Truncation policy: deterministic (last-N for 5'UTR, first-N for CDS/3'UTR)
- Output: deterministic (same input → same output, verified by SHA-256)

---

## 6. Usage

### 6.1 For P1-13 Counterfactual Panel

The long-view records can be used as the wild-type pool for the P1-13 panel:

```python
from mrna_editflow.core.schema import MRNARecord
import json

records = []
with open("data/reconstructed/p1_long_view/combined/long_view.records.jsonl") as f:
    for line in f:
        d = json.loads(line)
        r = MRNARecord(
            transcript_id=d["transcript_id"],
            five_utr=d["five_utr"],
            cds=d["cds"],
            three_utr=d["three_utr"],
            species=d.get("species", "human"),
        )
        records.append(r)
# 271,877 records available for the 1000-wild-type panel.
```

### 6.2 For Future Training

The long-view records can be used as training data with the `--train-idx`/`--val-idx`/`--test-idx` split contract (P1-10). The longer caps preserve more sequence context, which should improve the model's ability to learn long-range dependencies.

### 6.3 For Family-Disjoint Splits

The v1 frozen family/cross-source split manifests (`data/reconstructed/p0_data_reconstruction_v1/combined/*_family_split.json`) are defined on the v1 canonical records. To apply them to the long-view records, use the lineage files to map `canonical_id` → `derived_transcript_id`. The family assignments are unchanged (long-view reconstruction does not re-cluster families).

---

## 7. Limitations

1. **3'UTR truncation is aggressive**: 41.2% (GENCODE) / 53.6% (RefSeq) of 3'UTRs are truncated at 1024 nt. If 3'UTR distal regulatory elements are important for the task, consider increasing the cap or using a different truncation policy (e.g., keep first + last N).
2. **No new family split**: The long-view records inherit the v1 family assignments. If the longer sequences reveal new family structure, a re-clustering may be needed (out of scope for P1-11).
3. **No deduplication**: The combined view may contain duplicate sequences across GENCODE and RefSeq (same transcript annotated by both sources). Deduplication is the responsibility of downstream consumers (e.g., via `records_content_digest` in the split contract).
4. **No length-based stratification**: The long-view records are not stratified by length. For training, consider length-balanced sampling to avoid bias toward short transcripts.

---

**End of report.**
