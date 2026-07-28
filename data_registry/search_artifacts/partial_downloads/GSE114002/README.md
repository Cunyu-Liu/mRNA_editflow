# GSE114002 archive partial evidence

- accession: `GSE114002`
- source: `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE114nnn/GSE114002/suppl/GSE114002_RAW.tar`
- provider-listed size: `431175680` bytes
- preserved partial size at capture: `28178928` bytes
- status: incomplete; not a raw file and not admitted to a benchmark
- reason retained: archive fallback was attempted after per-sample timeouts
- next action: retain the direct per-sample retry as the primary repair path;
  use this archive only if the direct files remain unavailable

The binary `.part` file is kept on the remote acquisition volume and is not
committed to Git. It is never used by `verify_downloads.py` as a valid payload.
