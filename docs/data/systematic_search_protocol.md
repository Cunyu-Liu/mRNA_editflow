# Systematic Search Protocol (D0-02)

Contract: `utr_editflow_contract_v2` (`configs/utr_editflow_contract_v2.yaml`)
Executor: `scripts/data/systematic_search.py`
Results: `docs/data/systematic_search_results.md`
Registry: `data_registry/intervention_candidates.yaml`

## Objective

Discover and verify all public datasets that contain measured mRNA variant
intervention effects (WT–mutant pairs, dense synthetic landscapes, synonymous
CDS families, modular full-length constructs) for the four mRNA-EditBench
sub-benchmarks, without using any placeholder or fabricated record.

## Searched sources

| source | access method | query |
|---|---|---|
| GEO | NCBI eutils `esearch db=gds, term={ACCN}[ACCN]` + `esummary` | one query per candidate accession |
| SRA | eutils `elink dbfrom=gds db=sra` | raw-read availability per GEO series |
| ENA | SRA mirror of GEO-linked runs | covered transitively via SRA links |
| ENCODE | REST `GET /publication-data/{accession}/?format=json` | ENCSR854RUF (MPRAu) |
| MaveDB | REST API v1 | URN-only lookups; no free-text search endpoint exists (verified 2026-07-28: `/api/v1/score-sets/` requires `urns`) |
| Zenodo | REST `GET /api/records?q={query}&size=3` | per-candidate supplementary mirror search |
| Figshare | REST `POST /v2/articles/search` | per-candidate supplementary mirror search |
| paper supplementary files | cited in publications | variant counts / endpoint definitions |
| official GitHub/Bitbucket | publication-referenced repos | code-side data pointers |

Search date: 2026-07-28. All raw API responses are stored under
`data_registry/search_artifacts/` for audit.

## Inclusion criteria

A dataset is admitted as a candidate only if all of the following hold:

1. measured functional endpoint(s) on mRNA sequence variants
   (MRL, TE, RNA abundance, half-life, protein abundance — kept separate);
2. WT/source and mutant/candidate sequences are both recoverable
   (Grade A), or the design forms a controlled dense/synonymous family
   (Grade B);
3. public accession (GEO/ENCODE/Zenodo/Figshare) is live-verified;
4. the record can carry the 10 required registry fields
   (paper, accession, variant count, region, endpoint, WT availability,
   mutant availability, raw count availability, license, evidence grade).

## Exclusion criteria

* absolute-property-only datasets with no recoverable source–candidate
  relationship (e.g., random 50-nt libraries without edit structure);
* records whose accession cannot be verified live;
* mixed "expression" labels that merge incompatible endpoints.

## Evidence grading

* `A1` — intentionally assayed WT–mutant pairs
* `A2` — controlled dense sequence landscape
* `B1` — same-protein synonymous design family
* `B2` — modular full-length construct family

## Priority-0 candidate list (frozen)

| candidate | accession | sub-benchmark | role |
|---|---|---|---|
| Sample 2019 natural 5'UTR variants | GSE114002 | EditBench-5U-Natural | primary benchmark |
| PLUMAGE 5'UTR mutation library | GSE149487 | EditBench-5U-Natural | primary benchmark |
| NDD 5'UTR mutation MPRA | GSE246381 | EditBench-5U-Natural | historically_exposed external stress test (E4) |
| dense synthetic 5'UTR library | GSE145046 | EditBench-5U-Dense | large-scale pretraining |
| 3'UTR stability variants | GSE217518 | EditBench-3U-Variant | cross-region benchmark |
| prostate 3'UTR MPRA | GSE200304 | EditBench-3U-Variant | cross-region benchmark |
| MPRAu | ENCSR854RUF | EditBench-3U-Variant | cross-region benchmark |
| iCodon synonymous library | GSE207584 | EditBench-CDS-Synonymous | codon benchmark |
| PERSIST-seq | GSE173083 | EditBench-CDS-Synonymous | full-length transfer |

## Historical-exposure rule (v2 §2)

Per-variant labels of GSE246381 must not be read by training code before the
model and hyperparameters are frozen (see contract `utr_editflow_contract_v2` §2 (historically_exposed, E4; labels forbidden for new training and new hyperparameter selection)).

## Repair loop

Any accession failing live verification triggers:
retry → alternate accession/mirror → archive lookup → raw-read reconstruction
→ author contact → documented unavailable (see D0-04). A failed download is
never silently dropped.
