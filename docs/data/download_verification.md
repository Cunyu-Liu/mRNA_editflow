# Download Verification (D0-03)

- root: `data/p0`
- manifests: 9
- complete files: 75
- files failed: 0 after partial-evidence relocation
- files deferred: 62 ENCODE raw files
- archives skipped: 5 duplicate `RAW.tar` entries
- processed MPRAu supplement: 37,117,358 bytes; SHA-256
  `a02e6bd45e4f57bc0cf877aee766f006699b40469568c82974d21ac4d0346145`
- verdict: `PARTIAL` — raw ENCSR854RUF remains incomplete, so D0-03 is not
  closed.

| dataset | provider | complete | failed | deferred | skipped |
|---|---|---:|---:|---:|---:|
| ENCSR854RUF | ENCODE | 0 | 0 | 62 | 0 |
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
contain zero `.part` files. The current verifier treats deferred files as
non-zero/partial rather than silently passing them.

The MPRAu processed Supplementary Table 1 is a separate processed input and
does not close the ENCODE raw-read gate. No partial file is admitted as raw
data or as a benchmark label source.
