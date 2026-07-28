# Systematic Search Results (D0-02)

- generated: 2026-07-28T03:57:08+00:00
- protocol: `docs/data/systematic_search_protocol.md`
- candidates yaml: `data_registry/intervention_candidates.yaml`
- sources queried: GEO, SRA, ENA, Zenodo, Figshare, ENCODE, MaveDB, paper supplementary files, official GitHub/Bitbucket

## Per-source query log

| source | method | result |
|---|---|---|
| GEO | eutils esearch/esummary `[ACCN]` | 8/8 series verified |
| SRA | eutils elink gds->sra | raw-read links recorded per series |
| ENCODE | REST `/publication-data/{acc}/` | 1/1 verified |
| ENA | SRA mirror of GEO-linked runs | covered via SRA links |
| Zenodo | REST `/api/records?q=` | supplementary mirror search, top-3 recorded |
| Figshare | REST `/v2/articles/search` | supplementary mirror search, top-3 recorded |
| MaveDB | API v1 is URN-only (no free-text); no UTR score set adopted at D0 | documented |
| paper supplementary | cited variant counts from publications | recorded in yaml |
| official GitHub/Bitbucket | referenced by protocol; no extra candidates adopted | documented |

## Candidates

| candidate_id | accession | region | evidence_grade | endpoint | variant_count | geo/encode status |
|---|---|---|---|---|---|---|
| editbench_5u_natural_sample2019 | GSE114002 | 5'UTR | A1 | mean_ribosome_loading | 3577 natural variants (of 280k random + 35,212 truncated human 5'UTR library) | verified |
| editbench_5u_natural_plumage | GSE149487 | 5'UTR | A1 | transcript_abundance;translation_efficiency | 545 somatic mutations / 914 synthetic full-length 5'UTR sequences (WT+mutant) | verified |
| editbench_5u_natural_ndd | GSE246381 | 5'UTR | A1 | transcript_abundance;80S_monosome_polysome | 997 NDD family 5'UTR mutations (6 biological replicates) | verified |
| editbench_5u_dense_gse145046 | GSE145046 | 5'UTR | A2 | ribosome_free_monosome_polysome;fluorescence;in_cell_half_life;in_vitro_half_life | >1,000,000 designed 10-nt randomized variants on fixed scaffold | verified |
| editbench_3u_gse217518 | GSE217518 | 3'UTR | A1 | decay_constant;half_life | 6555 disease-relevant UTR variants (WT+mutant allele) | verified |
| editbench_3u_gse200304 | GSE200304 | 3'UTR | A1 | translation_efficiency;steady_state_rna;mrna_stability | 6892 patient mutations (6892 WT/mutant 201-nt pairs) | verified |
| editbench_3u_mprau | ENCSR854RUF | 3'UTR | A1 | allele_specific_rna_abundance | 12173 3'UTR variants (6 cell lines) | verified |
| editbench_cds_icodon | GSE207584 | CDS | B1 | mrna_decay_2h_5h_8h | 1395 synthesized synonymous CDS (955 perfect; 100 proteins x 16 designs) | verified |
| editbench_cds_persistseq | GSE173083 | full_length | B2 | ribosome_load;in_cell_stability;in_solution_stability | 233 full-length mRNA constructs (24 CDS designs) | verified |

## Acceptance

| check | status | detail |
|---|---|---|
| all 10 required fields non-empty on every candidate | PASS | 9 candidates, 0 with schema errors |
| all accessions live-verified with title match | PASS | failed: none |
