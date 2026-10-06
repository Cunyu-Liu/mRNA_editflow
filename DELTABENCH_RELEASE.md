# DELTABENCH_RELEASE.md — DeltaBench benchmark release notes (2026-10-08)

> This document is the external-facing release guide for the DeltaBench benchmark
> (the resource behind the manuscript "DeltaBench: A Source-Relative Benchmark
> Reveals Why Absolute-Score Models Fail to Predict mRNA Edit Effects").
> For the manuscript itself, see `docs/paper/biorxiv_submission_v1/`.

## 1. What DeltaBench is

DeltaBench is a **source-relative benchmark for mRNA edit-effect prediction**.
Its evaluation unit is the paired record (source sequence, candidate sequence,
direction-normalized delta), and its headline quantity is the delta caliber
(`delta_hat = pred(candidate) - pred(source)`) rather than absolute endpoint
accuracy. Composition:

| Component | Content | Scale |
|---|---|---|
| Curated records | canonical paired records from 15 public MPRA studies | 126,165 (89,580 TRAIN / 18,293 VALIDATION / 18,292 TEST) |
| Evaluation cells | 9 existing VALIDATION tasks + 3 append-only rows (M1 / M6 / S1) | 13 cells (S1 carries two assay arms) |
| Model families | 5 newly ported + 9 archived reference families | 14 families; 65-cell frozen matrix (65/65 re-checked to floating-point identity) |
| Split discipline | near-duplicate-component manifest, pigeonhole leakage audit, protected TEST (18,292 rows; read exactly once per the frozen unblinding prereg) | — |

## 2. Why it was built (scarcity)

Source-relative delta evaluation requires measured libraries in which multiple
candidates share one measured source under the same assay — a property most
public MPRA deposits lack (random libraries without shared sources; single-mutant
scans without within-source density). DeltaBench's curation is, to our knowledge,
the largest source-relative collection of its kind for mRNA edit effects.

## 3. Repository layout (what to use)

| Path | Content |
|---|---|
| `scripts/route_a_v3/` | row construction (`build_{m1,m6,s1}_*_v1.py`), matrix execution/aggregation/re-check (eight files), analysis scripts, figure producers (`make_biorxiv_figures_v2.py` + `make_biorxiv_figures_v3_ext.py`) |
| `docs/paper/` | manuscript, supplementary, pre-registration documents (§ mapping in the paper's §10.4), port ledger, rights audit |
| `docs/paper/biorxiv_submission_v1/` | submission package (manuscript, figures, supplementary) |
| `experiments/` (on the compute server, `/mnt/.../route2/experiments/`) | frozen JSONs: matrix, density, first-order, bottomline, W-ladder, ERK, D16-C, 3-seed ensemble, figure pack + manifests |

Large model outputs and frozen artifacts live on the compute server under
`/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/` (not in the repo);
the repo carries the code and documents that produce them, each with SHA-tagged
producer scripts.

## 4. Reproduction entry points

1. Row construction: `build_{m1,m6,s1}_*_v1.py` (construction seeds archived).
2. Matrix: the eight execution / aggregation / re-check scripts under
   `scripts/route_a_v3/` (the re-check recomputes all 65 cells from
   `predictions.jsonl`; achieved max |Delta| = 0.0).
3. Analyses: e.g. `run_delta_vs_density_v2.py`.
4. Figures: `make_biorxiv_figures_v2.py` then `make_biorxiv_figures_v3_ext.py`
   (41 value-lock assertions, all passing at render time).

## 5. Data rights (release boundary)

Per-study rights review over all 15 studies (GEO / ENCODE / EMBL-EBI terms;
`docs/paper/data_rights_audit_v1.md`): 15/15 analysis-and-publication use
permitted; raw payload redistribution conservatively **not** authorized. The
release therefore ships seed-frozen converters and manifests (not raw payloads);
each study is regenerable from its public accession.

## 6. Discipline highlights (frozen, non-negotiable)

- protected TEST reads = 0 before the frozen unblinding prereg executed
  (receipt in the paper's Supplementary S4);
- pre-registered thresholds are never retro-edited (revisions go through amendments);
- FINAL-EPOCH / no peak-picking; FAIL verdicts are archived as-is;
- append-only rows: the frozen matrix is never rewritten, only extended.

## 7. Status

Preprint (bioRxiv) submission package complete (W2); manuscript v2.3 expression
revision + figure pack v2 extension (journal 155). Maintained on branch
`route-a-v3-w0-diagnosis-20260902` of this repository.
