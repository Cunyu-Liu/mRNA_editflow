# GSE114002 D1 pipeline

This audit-visible pipeline admits only source-anchored single-SNV records as
interventions. Random, variable-length, and truncated libraries remain
`absolute_property_only`; their deposited `rl` values are
`PROVIDED_LABEL_ONLY`, not raw-bin reproductions.

The dataset-local scripts expose the contract stages. Production execution is
atomic through `scripts/data/build_d1_utr_benchmark.py`, which refuses to
overwrite an existing snapshot.
