# Download Verification (D0-03)

- root: `data/p0`
- manifests: 9
- complete files: 137
- files failed: 0
- files deferred: 0
- archives skipped: 5 duplicate `RAW.tar` entries
- processed MPRAu supplement: 37,117,358 bytes; SHA-256
  `a02e6bd45e4f57bc0cf877aee766f006699b40469568c82974d21ac4d0346145`
- ENCODE raw reads: 62/62 fastq.gz files downloaded to
  `data/p0/ENCSR854RUF/reconstructed/` (~357 GB), verified by
  provider_md5 + file size + presence check
- verdict: `COMPLETE`

| dataset | provider | complete | failed | deferred | skipped |
|---|---|---:|---:|---:|---:|
| ENCSR854RUF | ENCODE | 62 | 0 | 0 | 0 |
| GSE114002 | GEO | 10 | 0 | 0 | 1 |
| GSE145046 | GEO | 30 | 0 | 0 | 1 |
| GSE149487 | GEO | 18 | 0 | 0 | 1 |
| GSE173083 | GEO | 6 | 0 | 0 | 1 |
| GSE200304 | GEO | 2 | 0 | 0 | 1 |
| GSE207584 | GEO | 3 | 0 | 0 | 0 |
| GSE217518 | GEO | 4 | 0 | 0 | 0 |
| GSE246381 | GEO | 2 | 0 | 0 | 0 |

## Evidence boundary

The 75 complete GEO files have manifest byte sizes and SHA-256 values; the
earlier full verifier run reported all 75 as OK. The 10 interrupted transfers
(8 ENCODE and 2 GSE173083) were then moved without deletion to
`/mnt/cunyuliu/partial_evidence/`, and the raw data root was rechecked to
contain zero `.part` files.

The 62 ENCODE raw fastq.gz files were downloaded to
`data/p0/ENCSR854RUF/reconstructed/` between 2026-07-28 and 2026-07-30.
They are verified by ENCODE provider_md5, file size, and presence check.
SHA-256 is not recomputed (357 GB); provider_md5 serves as the integrity
credential. The MPRAu processed Supplementary Table 1 is a separate processed
input. No partial file is admitted as raw data or as a benchmark label source.
