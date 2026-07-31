# Missing Dataset Acquisition Protocol (D0-04)

Contract: `mrna_editflow_single_active_contract`
Registry: `data_registry/unavailable.yaml`

## Policy

A download failure does **not** remove a dataset from the benchmark plan.
Every dataset that cannot be obtained after the initial download attempt must
be tracked in `data_registry/unavailable.yaml` and processed through the
following escalation ladder, in order:

```text
retry
→ alternate mirror
→ archive
→ raw reconstruction
→ author contact
→ documented unavailable
```

A dataset may only be marked `documented_unavailable` after every earlier
rung has been attempted and the attempt evidence has been recorded.

## Escalation ladder

### 1. retry

Re-run the download script (idempotent, checksum-verified) after fixing the
root cause (URL routing, rate limiting, expired token). Record: number of
retries, error class, fix applied.

### 2. alternate mirror

If the primary host fails, try, in order:

- NCBI GEO/SRA primary FTP (`ftp.ncbi.nlm.nih.gov`)
- ENA mirror (`ftp.sra.ebi.ac.uk`) for SRA accessions
- ENCODE mirror (`www.encodeproject.org` ↔ `encodedcc.org` S3)
- Publisher supplementary site (Nature/Cell/Science static content)
- Zenodo / Figshare / Dryad deposits referenced by the paper

Record every mirror URL attempted and its HTTP status.

### 3. archive

If no live mirror works, query:

- Internet Archive Wayback Machine (`web.archive.org`) for supplementary URLs
- SRA archive packages (`sra-pub-run-*`)
- GEO series `miniml` / `matrix` bundles as fallback when per-sample
  supplementary files are missing

### 4. raw reconstruction

If processed tables are unavailable but raw reads exist (SRA/ENA), mark the
dataset as `reconstructable_from_raw: true` and record the planned
reconstruction pipeline (fastq-dump → alignment/quant → endpoint table).
Reconstruction must preserve the original endpoint definition and is gated
behind a documented analysis protocol before use in any benchmark.

### 5. author contact

If no public route exists, record:

- corresponding author name + email (from the paper)
- date contacted / channel
- requested artifacts (processed table, WT/mutant pairing key, license)

### 6. documented unavailable

Only after rungs 1–5 are exhausted. The record must state the final blocker
(e.g. "paper states data available on request; no response after 2 contact
attempts") and the current substitute dataset used in its place.

## Required record fields (per missing dataset)

See `data_registry/unavailable.yaml` for the authoritative records. Each
entry must contain:

- `dataset_id` — registry identifier (accession or paper key)
- `paper` — citation
- `searched_locations` — every location checked, with URL and result
- `supplemental` — whether publisher supplemental was checked and what it contains
- `author_code` — author repository URL if any, and what it provides
- `archive` — archive locations checked (Wayback, SRA bundle, etc.)
- `needs_author_contact` — bool
- `reconstructable_from_raw` — bool (+ planned pipeline if true)
- `current_substitute` — dataset currently used in its place (or `none`)
- `status` — one of `retry_pending`, `mirror_pending`, `archive_pending`,
  `reconstruction_pending`, `author_contact_pending`, `documented_unavailable`
- `last_checked_utc` — ISO-8601 timestamp of the last acquisition attempt
