# Download and ENCODE Inventory Verification (D0-03/D0-04)

- root: `data/p0`
- manifests: 9
- complete files: 75
- files failed: 0 after partial-evidence relocation
- ENCODE reconstruction snapshot: 62/62 `VERIFIED`
- ENCODE verified bytes: 378,589,831,611 / 378,589,831,611
- ENCODE inventory rows: 62 (62 `VERIFIED`)
- ENCODE inventory `complete`: `true`
- archives skipped: 5 duplicate `RAW.tar` entries
- processed MPRAu supplement: 37,117,358 bytes; SHA-256
  `a02e6bd45e4f57bc0cf877aee766f006699b40469568c82974d21ac4d0346145`
- alternate raw-read provenance: 62/62 ENCODE files mapped to GEO/SRA; map
  retained at `/mnt/cunyuliu/partial_evidence/ENCSR854RUF_sra_reconstruction_map.json`
- verdict: `D0_INVENTORY_RAW_ACQUISITION_COMPLETE` — raw acquisition is now
  complete, while the data remain observational/pretraining-only and are not
  intervention evidence.

| dataset | provider | complete | failed | deferred | skipped |
|---|---|---:|---:|---:|---:|
| ENCSR854RUF | ENCODE/ENA reconstruction | 62 | 0 | 0 | 0 |
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

The MPRAu processed Supplementary Table 1 remains a separate processed input.
The ENCODE raw-read acquisition closure does not turn either representation
into an intervention label source or a benchmark label source.

The historical 61/62 failure manifest is preserved unchanged. Its successor,
`/mnt/cunyuliu/partial_evidence/ENCSR854RUF/ENCSR854RUF_ena_reconstruction_manifest_20260730T142611Z_closure_v1.json`,
records the completed `ENCFF597AIT` reconstruction, the retained segmented
transfer ledger, full-stream gzip validation, one whole-file SHA-256
(`d0e403d31e3f8228becfcdc2c7bdd6ed507b906496334ee67611938fb00600fd`),
and the atomic `.part` to final-name promotion receipt. The inventory generated
at `2026-07-30T14:58:26.821522+00:00` contains 62 verified rows and exits zero.

No redundant 378,589,831,611-byte corpus rehash or raw-data copy was
performed. One whole-file SHA-256 was calculated only for the formerly missing
12,545,926,768-byte object after download completion. Even with all 62 files
verified, these reads remain an observational/pretraining candidate with
downstream overlap `UNKNOWN_REQUIRES_D1_B0_AUDIT`; D0 does not promote them to
intervention evidence.
