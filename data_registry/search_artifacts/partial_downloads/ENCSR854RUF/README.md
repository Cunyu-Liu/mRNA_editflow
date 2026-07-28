# ENCSR854RUF fastq partial evidence

- accession: `ENCSR854RUF`
- source: ENCODE publication-data / `ENCFF957TLC.fastq.gz`
- provider-listed size: `5124286754` bytes
- preserved partial size at capture: `7430144` bytes
- raw acquisition route: official ENCODE cloud URLs with provider md5 values
- status: incomplete; not a raw file and not admitted to a benchmark
- processed companion: `data/p0/ENCSR854RUF/processed/MPRAu_Supplementary_Table1.xlsx`
  is complete separately and has its own processed manifest; it does not close
  the raw-read acceptance gate
- alternate raw-read provenance: all 62 ENCODE files map to GEO/SRA in
  `/mnt/cunyuliu/partial_evidence/ENCSR854RUF_sra_reconstruction_map.json`
  (sha256 `6e9648cd956c2cd4bd09be576f3eb2ffacd5b6cc076e134b7d77c047348f107d`)
- the later parallel-download `.part` files are retained under
  `/mnt/cunyuliu/partial_evidence/ENCSR854RUF/` on the mounted acquisition volume
- next action: resume official cloud raw-file acquisition, or document raw-read
  reconstruction/archive/author contact after the scientific audit identifies
  that raw reads are required

The binary `.part` files are kept on the remote acquisition volume and are not
committed to Git. An earlier 7,430,144-byte capture is also retained in this
Git-side evidence directory for provenance.
