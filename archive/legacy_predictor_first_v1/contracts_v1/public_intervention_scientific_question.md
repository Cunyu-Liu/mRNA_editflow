# Scientific Question — public_intervention_contract_v1

**Contract:** `configs/public_intervention_contract.yaml` (`public_intervention_contract_v1`)
**Status:** FROZEN. Supersedes the entire P3/NMI legacy contract (`SUPERSEDED_LEGACY`, archived under `docs/archive/p3_legacy/`, `configs/archive/p3_legacy/`, `scripts/archive/p3_legacy/`).

---

## 1. Why the old contract was withdrawn

The legacy evidence chain was:

```text
local-delta prediction
-> constrained optimization
-> prospective candidate freeze
-> wet-lab protein-output validation
```

The final link is not executable. Therefore the legacy goals of prospective
protein-output improvement, multi-cargo wet-lab, protein-output AUC primary
endpoint, wet-lab unlocking of CDS/3'UTR/joint editing, and calling
model-designed candidates real beneficial mRNA are all withdrawn. The legacy
contract files are retained only for historical traceability in the archive
and must not be read by new training code, paper mode, or result-generation
code as a constraint source.

## 2. The single core scientific question

> **Can publicly measured mRNA variant intervention data be used to learn
> assay-specific functional increments of local sequence edits, such that the
> predictions transfer across sources, genes, studies, cellular contexts and
> mRNA regions?**

Decomposition:

- **Q1 — local effect predictability.** Given `source sequence + single or
  minimal edit + assay/context metadata`, predict the
  `candidate - source` functional effect.
- **Q2 — transfer across public studies.** Do regularities learned on earlier
  studies (Sample 2019, PLUMAGE) transfer to later studies (2025
  neurodevelopmental 5'UTR MPRA) without tuning on target-study labels?
- **Q3 — endpoint structure.** Do MRL, translation efficiency, RNA abundance,
  half-life and protein abundance share partial sequence regularities? These
  endpoints are studied separately and are **never merged into a unified
  "expression" label**.
- **Q4 — action representation.** Does a strictly legal sparse
  explicit-action representation (`source + edit action`) outperform plain
  sequence-difference and candidate-absolute representations?
- **Q5 — Flow value.** Does SparseEditFlow add value over the direct sparse
  scorer? Decided only by matched-compute experiments after the direct
  scorer exists; never预设.

## 3. What the paper is (and is not)

This is a **machine-learning methods + data benchmark + cross-experiment
generalization** study:

- resource: **mRNA-EditBench** (four independent sub-benchmarks:
  EditBench-5U-Natural, EditBench-5U-Dense, EditBench-3U-Variant,
  EditBench-CDS-Synonymous);
- model: **SparseEditFormer** (source encoded once + explicit biologically
  legal action + assay-aware effect prediction + top-K paired reranking +
  calibrated uncertainty);
- optional extension: **SparseEditFlow**, kept only if matched-compute
  experiments show added value.

The innovation is NOT "we concatenated mRNABERT, Orthrus and ESM-2". The
innovation is a public mRNA intervention-effect benchmark plus a
source-cached, action-sparse, assay-calibrated structured variant-effect
model.

## 4. Data foundation

Evidence Grade A WT–mutant pairs (primary task, `true_wt_mutant`):

| Dataset | Accession | Region | Endpoint(s) | Role |
|---|---|---|---|---|
| Sample 2019 natural 5'UTR variants | GSE114002 | 5'UTR | mean ribosome loading | train/dev |
| PLUMAGE 5'UTR mutation library | GSE149487 | 5'UTR | transcript abundance, translation efficiency | train/dev |
| 2025 neurodevelopmental 5'UTR MPRA | GSE246381 | 5'UTR | transcript abundance, 80S/monosome/polysome | **sealed external test** |

Dense synthetic landscape (Grade A2, pretraining): GSE145046
(`dense_synthetic_neighbor`, >1M measured 10-mer sequences).

3'UTR Grade A (cross-region): GSE217518, GSE200304 (prostate cancer 3'UTR
MPRA), MPRAu (ENCSR854RUF), MapUTR after validation.

CDS synonymous families (Grade B1, family ranking): iCodon, PERSIST-seq CDS
subset, GFP synonymous library.

**Sealed test rule.** GSE246381 per-variant labels must not be read by
training code before model and hyperparameters are frozen. The sealed
evaluation thresholds (delta Spearman >= 0.25, sign accuracy >= 0.60,
top-10% enrichment >= 1.5) are fixed in the contract and cannot be modified
after unsealing.

## 5. Endpoint discipline

- MRL is not protein output.
- TE is not protein output.
- Half-life is never merged with expression.
- 3'UTR and 5'UTR labels are never concatenated into one regression.
- Every conclusion retains its endpoint qualifier.

## 6. No-wet-lab statement

This project performs **no new wet-lab experiments**. No conclusion depends
on prospective experimental validation. Any statement about improved
translation, stability, abundance or expression is a *predicted* effect on a
measured public endpoint with explicit endpoint qualifiers, never a
demonstrated therapeutic benefit.
