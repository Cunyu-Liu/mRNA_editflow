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

## Current acquisition states

| accession | state | evidence | permitted use now |
|---|---|---|---|
| GSE114002 | partial; retry required | 5 files downloaded, 5 timed out, RAW.tar listed but intentionally skipped | candidate metadata only |
| GSE173083 | direct download in progress | `.part` files retained; final manifest and checksum verification pending | candidate metadata only |
| ENCSR854RUF | queued | accession and publication-data metadata live-verified; acquisition not yet finalized | candidate metadata only |

The authoritative per-file evidence is stored under `data/p0/<accession>/` in
the corresponding `manifest.json` files. `docs/data/download_verification.md`
is regenerated only after the active downloader exits. Partial files are never
treated as raw files and are not deleted by this process.

## Scientific boundary

Even after acquisition, a dataset enters a benchmark only if WT/mutant or
dense-variant structure, endpoint identity, replicate handling, split rules,
and leakage checks satisfy the public intervention contract. Downloaded data
cannot by itself support a causal or wet-lab claim.
