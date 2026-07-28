# Download and ENCODE Inventory Verification (D0-03/D0-04)

- root: `data/p0`
- manifests: 9
- complete files: 75
- files failed: 0 after partial-evidence relocation
- ENCODE reconstruction snapshot: 61/62 `VERIFIED`
- ENCODE verified bytes: 366,043,904,843 / 378,589,831,611
- ENCODE inventory rows: 62 (61 `VERIFIED`, 1 `MISSING_MANIFEST_ROW`)
- ENCODE inventory `complete`: `false`
- archives skipped: 5 duplicate `RAW.tar` entries
- processed MPRAu supplement: 37,117,358 bytes; SHA-256
  `a02e6bd45e4f57bc0cf877aee766f006699b40469568c82974d21ac4d0346145`
- alternate raw-read provenance: 62/62 ENCODE files mapped to GEO/SRA; map
  retained at `/mnt/cunyuliu/partial_evidence/ENCSR854RUF_sra_reconstruction_map.json`
- verdict: `D0_INVENTORY_VERIFIED_ACQUISITION_PARTIAL` — the D0 inventory
  gate is closed without claiming that the independent raw acquisition is
  complete.

| dataset | provider | complete | failed | deferred | skipped |
|---|---|---:|---:|---:|---:|
| ENCSR854RUF | ENCODE/ENA reconstruction | 61 | 0 | 1 | 0 |
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
contain zero `.part` files. The verifier now also fails closed on unreadable
manifests and any residual `.part` file.

The MPRAu processed Supplementary Table 1 is a separate processed input and
does not close the ENCODE raw-read gate. No partial file is admitted as raw
data or as a benchmark label source.

The ENCODE inventory was generated at
`2026-07-28T15:40:11.742240+00:00` from the downloader's atomic reconstruction
manifest and the 62-row source manifest. It represents every expected
accession, preserves 61 downloader-computed SHA-256 values, and records the
unreported accession explicitly instead of synthesizing a checksum. The
inventory builder returned exit code 2 by design because `complete=false`.

No redundant 378,589,831,611-byte rehash or raw-data copy was performed while
the downloader was active. The raw acquisition remains an independent running
task and may later regenerate this inventory as 62/62. Even after completion,
these reads remain an observational/pretraining candidate with downstream
overlap `UNKNOWN_REQUIRES_D1_B0_AUDIT`; D0 does not promote them to
intervention evidence.
