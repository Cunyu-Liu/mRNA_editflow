# D0 systematic search protocol — UTR intervention evidence

Contract: `mrna_editflow_single_active_contract`

Search date: 2026-07-28

Mode: metadata and publication text only; no candidate-level labels from a
potential new final set may be accessed before freeze.

## Questions

1. Does a measured source-paired UTR insertion/deletion MPRA exist?
2. Does a variable-length UTR library retain exact source/candidate relations?
3. Does a combinatorial UTR study observe edit trajectories, rather than only
   endpoint constructs?
4. Are there multiple measured candidates per source with raw counts and
   replicates?
5. Is there a genuinely new external candidate whose exposure status can be
   frozen before label access?

## Sources and tiers

- T1 primary: PubMed/PMC or publisher paper, NCBI GEO/SRA, ENA, ENCODE,
  MaveDB, and official supplementary or code repositories.
- T2 discovery: Crossref, OpenAlex, bioRxiv, Zenodo and Figshare. A T2 record
  is not promoted without a T1 identity or provenance check.

Search families combined `5'UTR` or `3'UTR` with `insertion`, `deletion`,
`indel`, `variable length`, `combinatorial`, `multi-edit`, `MPRA`, `reporter`,
`massively parallel`, `raw counts`, `replicate`, `source variant`, and
`trajectory`. Accession-title matches and exact paper identities were checked
before deduplication.

## Eligibility and deduplication

- Include UTR sequence libraries only when region and endpoint are explicit.
- Record exact source and candidate recoverability as unknown unless the public
  schema establishes it.
- Treat changed multi-position endpoints as endpoints, not observed paths.
- Deduplicate by DOI, then normalized title, then accession.
- Keep negative results and source failures.
- Record historical exposure before assigning an evidence role.
- Do not choose a data version, preprocessing rule or checkpoint by looking at
  final labels.

## Decision rules

- `measured indel` requires a measured altered-length candidate relative to an
  explicit source. A random variable-length library alone is insufficient.
- `multi-edit coverage` does not mean `observed trajectory`; trajectory claims
  require intermediate observations or another explicit path measurement.
- A new accession remains `metadata-only candidate` until its source/candidate
  mapping, license, prior exposure and freeze timestamp pass D1/B0 audit.
- Failure to find an action type lowers only the corresponding evidence claim;
  it does not change the Edit Flow question or make Flow optional.

## Reproducibility record

The D0 search used official GEO pages for accessions, primary full text for
study design, and OpenAlex as a broad fallback. Search outputs are summarized
in `docs/data/systematic_search_results.md`; dataset-level qualification is
frozen in `data_registry/dataset_capability_matrix.csv`.
