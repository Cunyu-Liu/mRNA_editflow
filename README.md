# mRNA-EditFlow / DeltaBench

**DeltaBench: a source-relative benchmark for mRNA edit-effect prediction — and the
mechanism analysis of why absolute-score models fail at it.**

> **Current release (2026-10)**: the repository now hosts the **DeltaBench**
> benchmark and the accompanying preprint package
> (`docs/paper/biorxiv_submission_v1/`). The original mRNA-EditFlow generation
> research (source-conditioned, region-aware, grammar-constrained continuous-time
> Edit Flow) remains below as the project's research lineage and the source of the
> generation-line application chapter. Start here: [DELTABENCH_RELEASE.md](DELTABENCH_RELEASE.md).

## Quick start (DeltaBench)

- **What it is**: 126,165 canonical paired records (source / candidate /
  direction-normalized delta) curated from 15 public MPRA studies; 13 evaluation
  cells; a frozen-delta (zero-fine-tuning) protocol; a 14-family × 65-cell matrix,
  independently re-checked to floating-point identity.
- **Why it exists**: downstream edit pipelines consume the *ranking of differences*
  (Δ), not absolute activity; no public dataset measured that quantity before.
- **How to use it**: per-study seed-frozen converters rebuild every record from
  public accessions; the frozen-delta evaluator scores any model on identical
  records; see the reproduction entry points in
  [DELTABENCH_RELEASE.md](DELTABENCH_RELEASE.md) §4.
- **Headline results**: frozen-Δ decoupling 0.873 absolute vs 0.313 delta;
  in-domain strong / cross-library collapsed (0.814 vs 0.067); the two-factor
  regularity (supervision regime × data geometry); polyA carried to 91.3% of its
  ICC-0.90 label ceiling (TEST-confirmed).

## Repository map

| Path | Content |
|---|---|
| [DELTABENCH_RELEASE.md](DELTABENCH_RELEASE.md) | DeltaBench release guide (composition, layout, reproduction, rights, discipline) |
| `docs/paper/biorxiv_submission_v1/` | preprint package: manuscript, figures, supplementary |
| `docs/paper/` | pre-registrations, port ledger, rights audit, rigor audit |
| `scripts/route_a_v3/` | benchmark construction, evaluation, re-check, analysis, figure producers |
| `configs/`, `core/`, `train/`, `eval/`, `models/` | mRNA-EditFlow research lineage (see below) |

## Research lineage (mRNA-EditFlow)

The project began as a study of whether a source-conditioned, region-aware,
grammar-constrained continuous-time Edit Flow can learn transferable legal
edit-trajectory distributions and generate diverse, sparse, controllable
5′UTR / 3′UTR candidates. The frozen evidence from that line (terminal
`SCREEN_NO_GO` on XEditSetFlow V3; the SetFlow COMB mechanism matrix; the
positive polyA V5 result 0.8219, Holm p = 0.004) fed the pivot to the
benchmark-plus-mechanism deliverable that DeltaBench now is. The governing
contracts and discipline are preserved under `archive/` and `docs/`.

> **Research status notice (2026-08-01, lineage)**: all scientific hypotheses of
> the generation line are adjudicated against pre-registered gates; no wet-lab
> evidence is in scope. The Route 2 V3.3.2 evidence packet
> (`docs/paper/route2_v332_methods_results_draft_v1.md`) remains the internal
> record of that adjudication and is background to the current release.

## Citation & license

Code is released under the repository license; third-party model weights are
loaded from their official releases under each provider's terms (MIT /
Apache-2.0 / NVIDIA Open Model / AGPL-3.0 as registered in the port ledger) and
are never redistributed by this project. Raw study payloads are not
redistributed; see [DELTABENCH_RELEASE.md](DELTABENCH_RELEASE.md) §5.
