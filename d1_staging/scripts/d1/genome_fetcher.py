"""Genome fetcher module for D1 reconstruction scripts.

Provides sequence fetching from either:
1. A local FASTA file (via pyfaidx)
2. The Ensembl GRCh37 REST API (for remote/low-bandwidth scenarios)

The Ensembl API supports batch retrieval via POST, allowing efficient
fetching of many small regions without downloading the entire genome.
"""

import json
import subprocess
import sys
import time
import urllib.request
import urllib.parse
from typing import Dict, List, Optional, Tuple

from pathlib import Path


_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def normalize_seq(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().upper().replace("U", "T")
    return "".join(c for c in s if c in "ACGT")


def ensure_fa(genome_path: Path, twobittofa: Path) -> Path:
    """Convert 2bit to fa if needed. Returns path to .fa file."""
    if genome_path.suffix == ".2bit":
        fa_path = genome_path.with_suffix(".fa")
        if fa_path.exists() and fa_path.stat().st_size > 1e9:
            return fa_path
        print(f"  converting {genome_path} -> {fa_path} ...", file=sys.stderr)
        result = subprocess.run(
            [str(twobittofa), str(genome_path), str(fa_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"twoBitToFa failed: {result.stderr}")
        print(f"  done: {fa_path.stat().st_size / 1e9:.1f} GB", file=sys.stderr)
        return fa_path
    return genome_path


class FastaGenomeFetcher:
    """Fetches sequences from a local FASTA file using pyfaidx."""

    def __init__(self, fa_path: Path):
        import pyfaidx
        self.genome = pyfaidx.Fasta(str(fa_path))

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """Fetch + strand sequence (0-indexed, half-open [start, end))."""
        if not chrom.startswith("chr"):
            chrom = f"chr{chrom}"
        try:
            seq = str(self.genome[chrom][start:end]).upper()
        except KeyError:
            alt = chrom.replace("chr", "")
            seq = str(self.genome[alt][start:end]).upper()
        return seq

    def prefetch(self, regions: List[Tuple[str, int, int]]):
        """No-op for FASTA fetcher (random access is instant)."""
        pass


class EnsemblGenomeFetcher:
    """Fetches sequences from the Ensembl REST API.

    Uses batch POST requests for efficiency. Results are cached.
    Coordinate system: 0-indexed, half-open [start, end) internally,
    converted to 1-indexed inclusive for the API.

    Args:
        genome_build: "GRCh37" for hg19 or "GRCh38" for hg38.
    """

    BATCH_SIZE = 50
    MAX_RETRIES = 3
    RETRY_DELAY = 2

    def __init__(self, genome_build: str = "GRCh37"):
        self.genome_build = genome_build.upper()
        if self.genome_build in ("GRCH38", "HG38", "38"):
            self.BASE_URL = "https://rest.ensembl.org"
        else:
            self.BASE_URL = "https://grch37.rest.ensembl.org"
        print(f"  using Ensembl REST API ({self.genome_build}): {self.BASE_URL}", file=sys.stderr)
        self.cache: Dict[Tuple[str, int, int], str] = {}

    def _chrom_for_api(self, chrom: str) -> str:
        """Normalize chromosome name for Ensembl API (no 'chr' prefix)."""
        return chrom.replace("chr", "")

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """Fetch + strand sequence (0-indexed, half-open [start, end)).

        Returns empty string on failure.
        """
        key = (chrom, start, end)
        if key in self.cache:
            return self.cache[key]

        # Single region GET request
        api_chrom = self._chrom_for_api(chrom)
        # Convert: 0-indexed half-open -> 1-indexed inclusive
        ens_start = start + 1
        ens_end = end
        url = f"{self.BASE_URL}/sequence/region/human/{api_chrom}:{ens_start}-{ens_end}:1"
        try:
            req = urllib.request.Request(
                url,
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode("utf-8"))
                seq = data.get("seq", "").upper()
                self.cache[key] = seq
                return seq
        except Exception as e:
            print(f"  WARNING: fetch failed for {chrom}:{start}-{end}: {e}", file=sys.stderr)
            self.cache[key] = ""
            return ""

    def prefetch(self, regions: List[Tuple[str, int, int]]):
        """Batch fetch multiple regions via POST requests.

        Args:
            regions: list of (chrom, start, end) tuples (0-indexed, half-open)
        """
        # Deduplicate and filter uncached
        to_fetch = []
        seen = set()
        for chrom, start, end in regions:
            key = (chrom, start, end)
            if key not in self.cache and key not in seen:
                to_fetch.append(key)
                seen.add(key)

        if not to_fetch:
            return

        print(f"  prefetching {len(to_fetch)} unique regions from Ensembl API...", file=sys.stderr)
        n_batches = (len(to_fetch) + self.BATCH_SIZE - 1) // self.BATCH_SIZE

        for bi in range(n_batches):
            batch = to_fetch[bi * self.BATCH_SIZE : (bi + 1) * self.BATCH_SIZE]
            # Build region strings for POST body
            region_strs = []
            for chrom, start, end in batch:
                api_chrom = self._chrom_for_api(chrom)
                ens_start = start + 1
                ens_end = end
                region_strs.append(f"{api_chrom}:{ens_start}-{ens_end}:1")

            # POST request
            url = f"{self.BASE_URL}/sequence/region/human"
            body = json.dumps({"regions": region_strs}).encode("utf-8")

            success = False
            for attempt in range(self.MAX_RETRIES):
                try:
                    req = urllib.request.Request(
                        url,
                        data=body,
                        headers={
                            "Accept": "application/json",
                            "Content-Type": "application/json",
                        },
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=60) as r:
                        results = json.loads(r.read().decode("utf-8"))

                    # Map results back to regions
                    for i, result in enumerate(results):
                        if i < len(batch):
                            chrom, start, end = batch[i]
                            seq = result.get("seq", "").upper()
                            self.cache[(chrom, start, end)] = seq

                    success = True
                    break
                except Exception as e:
                    if attempt < self.MAX_RETRIES - 1:
                        print(f"  batch {bi+1}/{n_batches} attempt {attempt+1} failed: {e}",
                              file=sys.stderr)
                        time.sleep(self.RETRY_DELAY * (attempt + 1))
                    else:
                        print(f"  batch {bi+1}/{n_batches} FAILED: {e}", file=sys.stderr)
                        # Mark as empty to avoid refetching
                        for chrom, start, end in batch:
                            self.cache[(chrom, start, end)] = ""

            if (bi + 1) % 20 == 0 or bi == n_batches - 1:
                print(f"  fetched batch {bi+1}/{n_batches}", file=sys.stderr)

            # Rate limit: ~15 requests/s without API key
            if bi < n_batches - 1:
                time.sleep(0.1)

        cached_count = sum(1 for v in self.cache.values() if v)
        print(f"  cache: {cached_count}/{len(self.cache)} regions have sequences", file=sys.stderr)


def create_fetcher(genome_arg: str, twobittofa_path: str = "", genome_build: str = "GRCh37") -> object:
    """Factory: create the appropriate genome fetcher.

    Args:
        genome_arg: "ensembl" for API, or path to .fa/.2bit file
        twobittofa_path: path to twoBitToFa binary (for .2bit conversion)
        genome_build: "GRCh37" for hg19 or "GRCh38" for hg38 (API only)

    Returns:
        FastaGenomeFetcher or EnsemblGenomeFetcher
    """
    if genome_arg.lower() in ("ensembl", "api", "rest"):
        return EnsemblGenomeFetcher(genome_build=genome_build)

    genome_path = Path(genome_arg)
    if genome_path.suffix == ".2bit":
        fa_path = ensure_fa(genome_path, Path(twobittofa_path))
    else:
        fa_path = genome_path
    print(f"  using FASTA: {fa_path}", file=sys.stderr)
    return FastaGenomeFetcher(fa_path)
