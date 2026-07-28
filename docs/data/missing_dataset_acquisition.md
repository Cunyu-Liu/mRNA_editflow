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
358 GB; the first fastq alone is 5.12 GiB. The operational D0 cap is therefore
1 GiB. Files above the cap are recorded with provider md5 as deferred, while
raw-read reconstruction remains an explicit repair path.

## Current acquisition states

| accession | state | evidence | permitted use now |
|---|---|---|---|
| GSE114002 | direct sample repair in progress | 5 files previously downloaded; two missing samples are being resumed; RAW.tar partial moved to evidence storage | candidate metadata only |
| GSE173083 | direct download complete | 6 files downloaded, 0 failed, 1 RAW.tar skipped; final verifier pending | direct files pending contract admission |
| ENCSR854RUF | download in progress | 62 files / about 358 GB listed; 1 GiB operational cap; oversized fastq partial preserved separately | candidate metadata only |

The authoritative per-file evidence is stored under `data/p0/<accession>/` in
the corresponding `manifest.json` files. The incomplete GSE114002 archive is
preserved at `data_registry/search_artifacts/partial_downloads/` as acquisition
evidence, outside the raw-data root. `docs/data/download_verification.md` is
regenerated only after the active downloaders exit. Partial files are never
treated as raw files.

## Scientific boundary

Even after acquisition, a dataset enters a benchmark only if WT/mutant or
dense-variant structure, endpoint identity, replicate handling, split rules,
and leakage checks satisfy the public intervention contract. Downloaded data
cannot by itself support a causal or wet-lab claim.
