# P3-10 Mechanism Analysis

> All mechanism features are computed from sequence heuristics.
> They are NOT wet-lab measurements.

> Qualifier: All mechanism features are computed from sequence heuristics, not wet-lab measurements.

## Key Finding

All 10 mechanisms can be computed as sequence-level features, but only 7 can be meaningfully assessed with the current 5'UTR-only data. 3'UTR stability motifs use an inert placeholder 3'UTR, and edit-order dependence requires a joint 5'UTR×CDS oracle that does not exist. The mechanism analysis is therefore PARTIAL.

## Mechanism Summary

| Mechanism | Mean | Std | Min | Max | N | Assessable |
|---|---|---|---|---|---|---|
| start_accessibility | 0.3679 | 0.1158 | 0.2174 | 0.7500 | 24 | assessed |
| kozak_context | 0.8333 | 0.3118 | 0.0000 | 1.0000 | 24 | assessed |
| uorf_count | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 24 | assessed |
| start_proximal_codon | 0.5333 | 0.0000 | 0.5333 | 0.5333 | 24 | assessed |
| codon_usage | 0.1667 | 0.0000 | 0.1667 | 0.1667 | 24 | assessed |
| codon_pair_context | 0.8000 | 0.0000 | 0.8000 | 0.8000 | 24 | assessed |
| global_structure_gc | 0.6208 | 0.1467 | 0.3000 | 0.8800 | 24 | assessed |
| three_utr_stability_motifs | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 24 | limited — 3'UTR is inert placeholder |
| rbp_motifs | 1.9583 | 1.6703 | 0.0000 | 7.0000 | 24 | assessed (5'UTR only) |
| edit_order_dependence | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 24 | not assessable — requires joint oracle |

## Edit-Order Dependence

- Synergy mean: 0.000000
- Synergy std: 0.000000
- Interpretation: Edit-order dependence cannot be established with the current 5'UTR-only oracle. The oracle processes 5'UTR features only, so CDS edits are invisible. True edit-order effects require a joint oracle that considers 5'UTR × CDS interactions.