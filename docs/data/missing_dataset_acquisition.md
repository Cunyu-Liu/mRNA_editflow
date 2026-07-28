# Missing-dataset acquisition log (D0-04)

This document is part of `public_intervention_contract_v1`. A failed download is
an acquisition state, not permission to remove a dataset or alter an old result.
Candidates remain outside the primary benchmark until their raw files, labels,
checksums, and endpoint definitions pass the contract gates.

## Search and repair protocol

For each incomplete P0 accession the repair order is fixed:

1. retry the original provider with bounded retries;
2. use an alternate mirror or provider endpoint;
3. inspect archive copies while retaining the original accession;
4. assess reconstruction from raw reads (SRA/ENA) without assuming labels;
5. contact the authors if the required files or labels remain unavailable;
6. record the accession as documented unavailable when the preceding evidence is exhausted.

Search locations used in this run were GEO, SRA, ENA, Zenodo, Figshare,
ENCODE, paper supplementary files, and official GitHub/Bitbucket locations.
MaveDB was queried as a supplementary discovery route; no verified MaveDB URN
was adopted, so `download_mavedb.py --urn ...` was not run.

The first GEO attempts used a 600-second curl wall-clock limit. Large or slow
files were observed timing out and leaving an incomplete acquisition state. The
download helper was then repaired to accept
`MRNA_EDITFLOW_DOWNLOAD_TIMEOUT=3600`; the affected P0 accessions were
restarted with that setting. A `.part` file is evidence of an unfinished
transfer, never a checksum-verified raw file.

For ENCSR854RUF, live publication-data metadata lists 62 files totaling about
358 GB; the first fastq alone is 5.12 GiB. The PMC Supplementary Table 1
processed screen data and per-oligo read counts were downloaded separately and
are recorded in `data/p0/ENCSR854RUF/processed/processed_manifest.json`.
The 62 ENCODE raw files are still incomplete: cloud URLs, provider md5 values,
and `.part` evidence are retained, but an incomplete transfer is never
admitted as a raw file. Raw-read reconstruction remains an explicit repair path.
The later parallel-download `.part` files are retained at
`/mnt/cunyuliu/partial_evidence/ENCSR854RUF/`; because that volume is
separate from the repository home filesystem, they are not duplicated into the
Git-side evidence directory.

## Current acquisition states

| accession | state | evidence | permitted use now |
|---|---|---|---|
| GSE114002 | direct sample repair complete | 10 per-sample files downloaded; 1 RAW.tar duplicate skipped; incomplete archive retained in evidence storage | direct files pending final contract admission |
| GSE173083 | direct download complete | 6 files downloaded, 0 failed, 1 RAW.tar skipped; two stale `.part` files moved to `/mnt/cunyuliu/partial_evidence/GSE173083/`; final verifier pending | direct files pending contract admission |
| ENCSR854RUF | processed complete / raw incomplete | processed Supplementary Table 1 is 37,117,358 bytes and SHA-256 verified; 62 raw files remain incomplete and `.part` files are retained on the mounted acquisition volume | processed data for exploratory preprocessing; raw data not admitted |

The authoritative per-file evidence is stored under `data/p0/<accession>/` in
the corresponding `manifest.json` files. Processed MPRAu evidence is stored
under `data/p0/ENCSR854RUF/processed/processed_manifest.json`; it is explicitly
separate from the ENCODE raw-read manifest. Incomplete GSE114002 and ENCSR854RUF
transfers are preserved either at `data_registry/search_artifacts/partial_downloads/`
or at `/mnt/cunyuliu/partial_evidence/`, depending on filesystem placement.
Partial files are never treated as raw files.

## Scientific boundary

Even after acquisition, a dataset enters a benchmark only if WT/mutant or
dense-variant structure, endpoint identity, replicate handling, split rules,
and leakage checks satisfy the public intervention contract. Downloaded data
cannot by itself support a causal or wet-lab claim.
