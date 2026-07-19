"""Corpus acquisition for mRNA-EditFlow.

Two acquisition paths live here:

1. **Online** (kept fully offline in tests): a streaming downloader with
   SHA256 verification and a local checksum cache, plus a registry of the
   public human mRNA corpora (RefSeq / GENCODE). The streaming/SHA256/cache
   *pattern* is adapted from ``rug_ood.data.public_datasets``. URLs are stored
   as constants only; nothing is fetched at import time or during tests.

2. **Offline synthetic** (the default for CI / smoke tests):
   :func:`synthesize_corpus` deterministically generates biologically-plausible
   full-length mRNAs (valid ``AUG ... stop`` CDS built from random sense codons,
   free-length UTRs with occasional Kozak / upstream-AUG motifs). Records are
   organised into near-duplicate *families* (a founder plus silent-mutation
   variants) so the downstream family-disjoint splitter has real redundancy to
   cluster.

A light RefSeq-style parser (:func:`iter_fasta`, :func:`record_from_annotation`)
mirrors the extraction ideas in ``LucaOne`` without requiring Biopython: given a
transcript sequence and its CDS coordinates it emits a region-annotated
:class:`~mrna_editflow.core.schema.MRNARecord`.

Complexity: downloader is O(bytes) time / O(1) space; ``synthesize_corpus`` is
O(n * L) for total sequence length L.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import urllib.error
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from mrna_editflow.core.constants import (
    CODON_TABLE,
    START_CODON,
    STOP_CODONS,
    SYNONYMOUS_CODONS,
    is_valid_cds,
)
from mrna_editflow.core.schema import MRNARecord

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1024 * 1024  # 1 MiB streaming chunk
_CHECKSUM_CACHE_FILENAME = ".mef_mrna_checksums.json"

# Codons usable in the body of a CDS (exclude the three stops).
_SENSE_CODONS: Tuple[str, ...] = tuple(
    c for c, aa in sorted(CODON_TABLE.items()) if aa != "*"
)
# Amino acid -> synonymous codons, minus the stop pseudo-"aa", used for the
# silent-mutation variant generator.
_SENSE_SYNONYMS: Dict[str, List[str]] = {
    aa: codons for aa, codons in SYNONYMOUS_CODONS.items() if aa != "*"
}


# ---------------------------------------------------------------------------
# Streaming downloader + SHA256 verification + local checksum cache
# (pattern adapted from rug_ood.data.public_datasets)
# ---------------------------------------------------------------------------
def _open_url(url: str):
    """Open ``url`` with TLS verification, retrying via ``certifi`` if needed.

    Some macOS Python builds ship without a usable CA bundle; rather than
    disabling verification we fall back to ``certifi`` when it is installed.
    """
    import ssl
    import urllib.request

    request = urllib.request.Request(
        url, headers={"User-Agent": "mrna-editflow/0.1 (data-pipeline)"}
    )
    try:
        return urllib.request.urlopen(request, timeout=60)
    except ssl.SSLCertVerificationError as exc:
        try:
            import certifi  # type: ignore
        except Exception as certifi_exc:  # pragma: no cover - env dependent
            raise ssl.SSLCertVerificationError(
                f"{exc}. Missing trusted CA certificates. Install `certifi` "
                "or run the Python macOS certificate installer."
            ) from certifi_exc
        context = ssl.create_default_context(cafile=certifi.where())
        return urllib.request.urlopen(request, timeout=60, context=context)


def _compute_sha256(filepath: Path) -> str:
    """Streaming SHA256 of a file. Complexity: O(bytes) time, O(1) space."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _update_checksum_cache(cache_path: Path, filename: str, sha256: str) -> None:
    """Append/update a single file's checksum in the local cache JSON."""
    data: Dict[str, str] = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                data = loaded
        except (json.JSONDecodeError, OSError):
            data = {}
    data[filename] = sha256
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def load_cached_checksums(data_dir: str) -> Dict[str, str]:
    """Load ``{filename: sha256}`` recorded by a previous download session."""
    cache_path = Path(data_dir) / _CHECKSUM_CACHE_FILENAME
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    return {}


def _download_file(
    url: str,
    dest: Path,
    force: bool = False,
    expected_sha256: Optional[str] = None,
    checksum_cache_path: Optional[Path] = None,
) -> None:
    """Stream ``url`` to ``dest`` with SHA256 verification and caching.

    Downloads to a ``.part`` temp file, verifies (or records) its SHA256, then
    atomically renames into place. Raises ``RuntimeError`` on HTTP/URL errors
    and ``ValueError`` on a checksum mismatch. Complexity: O(bytes).
    """
    if dest.exists() and not force:
        if expected_sha256 is not None:
            actual = _compute_sha256(dest)
            if actual == expected_sha256:
                logger.info("File %s present with matching SHA256; skipping.", dest.name)
                return
            logger.warning(
                "Existing %s SHA256 mismatch (expected %s, got %s); re-downloading.",
                dest.name, expected_sha256, actual,
            )
        else:
            return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    logger.info("Downloading %s -> %s", url, dest)
    try:
        with _open_url(url) as response:
            with open(tmp, "wb") as handle:
                while True:
                    chunk = response.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
        actual = _compute_sha256(tmp)
        if expected_sha256 is not None:
            if actual != expected_sha256:
                tmp.unlink()
                raise ValueError(
                    f"SHA256 verification failed for {dest.name}: "
                    f"expected {expected_sha256}, got {actual}"
                )
            logger.info("SHA256 verified for %s", dest.name)
        else:
            logger.info("No expected checksum for %s; computed %s", dest.name, actual)
            if checksum_cache_path is not None:
                _update_checksum_cache(checksum_cache_path, dest.name, actual)
        tmp.replace(dest)
    except urllib.error.HTTPError as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"HTTP error downloading {url}: {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"URL error downloading {url}: {exc.reason}") from exc
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


# ---------------------------------------------------------------------------
# Public human mRNA corpus registry (URLs only; never fetched in tests)
# ---------------------------------------------------------------------------
MRNA_DATASETS: Dict[str, Dict[str, Any]] = {
    "refseq_human_rna": {
        "url": "https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/mRNA_Prot/",
        "files": ["human.1.rna.gbff.gz"],
        "description": (
            "RefSeq Homo sapiens curated RNA (GenBank flat file, gzipped). "
            "Contains per-transcript CDS feature annotation used to carve "
            "5'UTR / CDS / 3'UTR regions."
        ),
        "license": "US Government public domain (NCBI RefSeq)",
        # NCBI does not publish stable SHA256 sidecars; verified via local cache.
        "checksums": {"human.1.rna.gbff.gz": None},
    },
    "gencode_human_transcripts": {
        "url": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/",
        "files": ["gencode.v45.pc_transcripts.fa.gz"],
        "description": (
            "GENCODE v45 human protein-coding transcript sequences (FASTA, "
            "gzipped). CDS coordinates are encoded in the FASTA headers "
            "(CDS:start-end fields)."
        ),
        "license": "EMBL-EBI terms of use (open)",
        "checksums": {"gencode.v45.pc_transcripts.fa.gz": None},
    },
}


def _download_registry_entry(name: str, target_dir: Path, force: bool = False) -> Dict[str, str]:
    """Download every file of a registry entry, honouring the checksum cache."""
    entry = MRNA_DATASETS[name]
    base_url = entry["url"]
    checksums = entry.get("checksums", {})
    cached = load_cached_checksums(str(target_dir))
    cache_path = target_dir / _CHECKSUM_CACHE_FILENAME
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    for filename in entry["files"]:
        dest = target_dir / filename
        url = base_url + filename
        effective_sha = checksums.get(filename) or cached.get(filename)
        _download_file(url, dest, force=force, expected_sha256=effective_sha,
                       checksum_cache_path=cache_path)
        paths[filename] = str(dest)
    return paths


def download_dataset(name: str, target_dir: str, force: bool = False) -> Dict[str, str]:
    """Download a registered corpus into ``target_dir``. Never called in tests."""
    if name not in MRNA_DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Available: {list(MRNA_DATASETS)}")
    return _download_registry_entry(name, Path(target_dir), force=force)


def compute_and_cache_checksums(data_dir: str) -> Dict[str, str]:
    """Compute SHA256 for every present registry file and persist the cache."""
    data_path = Path(data_dir)
    result: Dict[str, str] = {}
    for entry in MRNA_DATASETS.values():
        for filename in entry["files"]:
            fp = data_path / filename
            if fp.exists():
                result[filename] = _compute_sha256(fp)
    cache_path = data_path / _CHECKSUM_CACHE_FILENAME
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    return result


def verify_dataset(name: str, data_dir: str) -> Dict[str, Any]:
    """Report which files exist and whether their SHA256 matches expectations."""
    if name not in MRNA_DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Available: {list(MRNA_DATASETS)}")
    entry = MRNA_DATASETS[name]
    data_path = Path(data_dir)
    hashes: Dict[str, Optional[str]] = {}
    missing: List[str] = []
    valid = True
    for filename in entry["files"]:
        fp = data_path / filename
        if not fp.exists():
            missing.append(filename)
            hashes[filename] = None
            valid = False
            continue
        actual = _compute_sha256(fp)
        hashes[filename] = actual
        expected = entry.get("checksums", {}).get(filename)
        if expected is not None and expected != actual:
            valid = False
    return {"valid": valid, "hashes": hashes, "missing_files": missing}


def get_dataset_info(name: str) -> Dict[str, Any]:
    if name not in MRNA_DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Available: {list(MRNA_DATASETS)}")
    entry = MRNA_DATASETS[name]
    return {"name": name, "url": entry["url"], "files": list(entry["files"]),
            "description": entry["description"], "license": entry["license"]}


def list_available_datasets() -> List[Dict[str, Any]]:
    return [get_dataset_info(name) for name in MRNA_DATASETS]


# ---------------------------------------------------------------------------
# Light RefSeq/GENCODE-style parsing (Biopython-free)
# ---------------------------------------------------------------------------
def iter_fasta(path: str) -> Iterator[Tuple[str, str]]:
    """Yield ``(header, sequence)`` pairs from a (optionally gzipped) FASTA.

    Complexity: O(file size) time, O(one record) space.
    """
    import gzip

    opener = gzip.open if str(path).endswith(".gz") else open
    header: Optional[str] = None
    chunks: List[str] = []
    with opener(path, "rt") as fh:  # type: ignore[call-overload]
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:]
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            yield header, "".join(chunks)


def record_from_annotation(
    transcript_id: str,
    sequence: str,
    cds_start: int,
    cds_end: int,
    species: str = "human",
) -> MRNARecord:
    """Carve a transcript into 5'UTR / CDS / 3'UTR by CDS coordinates.

    ``cds_start`` / ``cds_end`` are 0-based, half-open into ``sequence`` (the
    convention RefSeq/GENCODE headers translate to). Normalisation and
    validation are the cleaner's job; this only slices. Complexity: O(len).
    """
    return MRNARecord(
        transcript_id=transcript_id,
        five_utr=sequence[:cds_start],
        cds=sequence[cds_start:cds_end],
        three_utr=sequence[cds_end:],
        species=species,
    )


def parse_gencode_cds_range(header: str) -> Optional[Tuple[int, int]]:
    """Parse a GENCODE FASTA ``CDS:start-end`` field into Python coordinates.

    GENCODE transcript FASTA headers are pipe-delimited and protein-coding
    records normally include a one-based, closed interval such as
    ``CDS:76-1032``. The model schema uses zero-based, half-open slicing, so the
    conversion is:

    ``[start_1based, end_1based] -> [start_1based - 1, end_1based)``.

    The function returns ``None`` when no valid CDS interval is present, letting
    callers skip non-coding or malformed records without guessing. Complexity is
    ``O(len(header))``.
    """
    for field in str(header).split("|"):
        if not field.startswith("CDS:"):
            continue
        coords = field[4:]
        if "-" not in coords:
            return None
        start_text, end_text = coords.split("-", 1)
        try:
            start = int(start_text)
            end = int(end_text)
        except ValueError:
            return None
        if start < 1 or end < start:
            return None
        return start - 1, end
    return None


def transcript_id_from_gencode_header(header: str) -> str:
    """Return the stable transcript accession from a GENCODE FASTA header.

    The first pipe-delimited field is the transcript id, often versioned
    (``ENST...`` or ``ENST....1``). Keeping the version makes provenance exact
    for a specific release. Complexity is ``O(len(header))``.
    """
    text = str(header).strip()
    return text.split("|", 1)[0].split()[0]


def records_from_gencode_fasta(
    path: str,
    species: str = "human",
    limit: Optional[int] = None,
) -> List[MRNARecord]:
    """Load region-annotated records from a GENCODE transcript FASTA file.

    Only entries with a valid ``CDS:start-end`` header field and in-range CDS
    coordinates are emitted. Sequence normalization and ORF validity are handled
    by :mod:`mrna_editflow.data.clean_mrna`, so this parser remains a transparent
    extraction step. Complexity is ``O(total FASTA bytes)`` time and
    ``O(number_of_records)`` memory for the returned list.
    """
    out: List[MRNARecord] = []
    for header, sequence in iter_fasta(path):
        cds_range = parse_gencode_cds_range(header)
        if cds_range is None:
            continue
        cds_start, cds_end = cds_range
        if cds_end > len(sequence):
            continue
        out.append(
            record_from_annotation(
                transcript_id=transcript_id_from_gencode_header(header),
                sequence=sequence,
                cds_start=cds_start,
                cds_end=cds_end,
                species=species,
            )
        )
        if limit is not None and len(out) >= int(limit):
            break
    return out


def _open_text_maybe_gzip(path: str):
    """Open plain text or gzip-compressed text by suffix."""
    import gzip

    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def iter_genbank_records(path: str) -> Iterator[List[str]]:
    """Yield raw GenBank records from a plain or gzipped flat file.

    Records are delimited by ``//``. Keeping the parser line-oriented avoids a
    Biopython dependency and is sufficient for RefSeq RNA CDS feature extraction.
    Complexity is ``O(file bytes)`` time and ``O(one record)`` memory.
    """
    current: List[str] = []
    with _open_text_maybe_gzip(path) as fh:  # type: ignore[call-overload]
        for line in fh:
            text = line.rstrip("\n")
            current.append(text)
            if text.strip() == "//":
                yield current
                current = []
    if current:
        yield current


def parse_genbank_cds_location(location: str) -> Optional[Tuple[int, int]]:
    """Parse a conservative plus-strand GenBank CDS location.

    Supported forms are ``7..15`` and contiguous ``join(7..9,10..15)`` with
    optional ``<``/``>`` boundary markers. Complement, order, remote accessions
    and non-contiguous joins are skipped because they cannot be represented as a
    single contiguous CDS segment inside :class:`MRNARecord`.

    GenBank coordinates are one-based closed; the returned interval is zero-based
    half-open. Complexity is ``O(len(location))``.
    """
    text = "".join(str(location).strip().split())
    if not text or "complement" in text or "order" in text or ":" in text:
        return None
    if text.startswith("join(") and text.endswith(")"):
        parts = text[5:-1].split(",")
    else:
        parts = [text]
    intervals: List[Tuple[int, int]] = []
    for part in parts:
        clean = part.replace("<", "").replace(">", "")
        if ".." not in clean:
            return None
        start_text, end_text = clean.split("..", 1)
        if not start_text.isdigit() or not end_text.isdigit():
            return None
        start = int(start_text)
        end = int(end_text)
        if start < 1 or end < start:
            return None
        intervals.append((start - 1, end))
    intervals.sort()
    for prev, curr in zip(intervals, intervals[1:]):
        if prev[1] != curr[0]:
            return None
    return intervals[0][0], intervals[-1][1]


def _genbank_accession(lines: Sequence[str]) -> Optional[str]:
    version: Optional[str] = None
    accession: Optional[str] = None
    for line in lines:
        if line.startswith("VERSION"):
            parts = line.split()
            if len(parts) >= 2:
                version = parts[1]
        elif line.startswith("ACCESSION"):
            parts = line.split()
            if len(parts) >= 2:
                accession = parts[1]
        if version is not None:
            return version
    return accession


def _genbank_origin_sequence(lines: Sequence[str]) -> str:
    in_origin = False
    chunks: List[str] = []
    for line in lines:
        if line.startswith("ORIGIN"):
            in_origin = True
            continue
        if not in_origin:
            continue
        if line.strip() == "//":
            break
        chunks.append("".join(ch for ch in line if ch.isalpha()))
    return "".join(chunks)


def _genbank_cds_locations(lines: Sequence[str]) -> List[str]:
    locations: List[str] = []
    in_features = False
    collecting = False
    current: List[str] = []
    for line in lines:
        if line.startswith("FEATURES"):
            in_features = True
            continue
        if in_features and line.startswith("ORIGIN"):
            break
        if not in_features:
            continue
        key = line[5:21].strip() if len(line) >= 21 else ""
        body = line[21:].strip() if len(line) > 21 else ""
        if key:
            if collecting and current:
                locations.append("".join(current))
            collecting = key == "CDS"
            current = [body] if collecting else []
            continue
        if collecting:
            cont = line[21:].strip() if len(line) > 21 else ""
            if cont.startswith("/"):
                locations.append("".join(current))
                collecting = False
                current = []
            elif cont:
                current.append(cont)
    if collecting and current:
        locations.append("".join(current))
    return locations


def records_from_refseq_genbank(
    path: str,
    species: str = "human",
    limit: Optional[int] = None,
) -> List[MRNARecord]:
    """Load region-annotated records from a RefSeq RNA GenBank flat file.

    The parser is intentionally conservative and dependency-free. It emits one
    record per GenBank entry when there is exactly one representable plus-strand
    contiguous CDS feature with in-range coordinates. Ambiguous multi-CDS,
    complement or non-contiguous records are skipped for safety. Sequence
    normalization and ORF validity remain the cleaner's job. Complexity is
    ``O(total GenBank bytes)``.
    """
    out: List[MRNARecord] = []
    for lines in iter_genbank_records(path):
        transcript_id = _genbank_accession(lines)
        if not transcript_id:
            continue
        sequence = _genbank_origin_sequence(lines)
        locations = _genbank_cds_locations(lines)
        parsed = [loc for loc in (parse_genbank_cds_location(x) for x in locations) if loc]
        if len(parsed) != 1:
            continue
        cds_start, cds_end = parsed[0]
        if cds_end > len(sequence):
            continue
        out.append(
            record_from_annotation(
                transcript_id=transcript_id,
                sequence=sequence,
                cds_start=cds_start,
                cds_end=cds_end,
                species=species,
            )
        )
        if limit is not None and len(out) >= int(limit):
            break
    return out


def write_records_jsonl(records: Sequence[MRNARecord], path: str) -> None:
    """Write records as UTF-8 JSONL using the canonical schema.

    JSONL keeps large public corpora streamable and diffable while preserving
    every region string needed to reconstruct token, region and phase tracks.
    Complexity is ``O(sum(len(record.seq)))``.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def load_records_jsonl(path: str) -> List[MRNARecord]:
    """Load records written by :func:`write_records_jsonl`.

    Blank lines are ignored. Every non-blank row must be a JSON object matching
    :class:`~mrna_editflow.core.schema.MRNARecord.from_dict`. Complexity is
    ``O(file bytes)``.
    """
    records: List[MRNARecord] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_no} is not a JSON object")
            records.append(MRNARecord.from_dict(payload))
    return records


# ---------------------------------------------------------------------------
# Offline synthetic corpus generator
# ---------------------------------------------------------------------------
def _random_cds(rng: random.Random, n_sense: int) -> str:
    """Random valid CDS: ``AUG`` + ``n_sense`` sense codons + one stop codon."""
    body = "".join(rng.choice(_SENSE_CODONS) for _ in range(n_sense))
    return START_CODON + body + rng.choice(STOP_CODONS)


def _random_utr(rng: random.Random, length: int, uaug: bool, kozak: bool) -> str:
    """Random UTR of exactly ``length`` nt with optional regulatory motifs.

    ``uaug`` embeds an upstream ``AUG`` (uORF start) in the interior; ``kozak``
    makes the 3' end of the UTR the strong Kozak context ``GCCACC`` immediately
    preceding the CDS start. Both are clamped to keep the requested length.
    """
    if length <= 0:
        return ""
    seq = [rng.choice("ACGU") for _ in range(length)]
    if uaug and length >= 6:
        pos = rng.randint(0, length - 3)
        seq[pos:pos + 3] = list("AUG")
    if kozak and length >= 6:
        seq[length - 6:length] = list("GCCACC")
    return "".join(seq)


def _make_founder(rng: random.Random, family_id: int) -> MRNARecord:
    """Create one family founder: a valid, within-caps full-length mRNA."""
    five_len = rng.randint(8, 120)
    three_len = rng.randint(12, 240)
    n_sense = rng.randint(20, 260)
    five = _random_utr(rng, five_len, uaug=rng.random() < 0.2, kozak=rng.random() < 0.35)
    three = _random_utr(rng, three_len, uaug=False, kozak=False)
    cds = _random_cds(rng, n_sense)
    species = "human" if rng.random() < 0.85 else "mouse"
    return MRNARecord(
        transcript_id=f"SYN_{family_id:05d}_0",
        five_utr=five, cds=cds, three_utr=three, species=species,
    )


def _mutate_variant(rng: random.Random, founder: MRNARecord, family_id: int,
                    member: int) -> MRNARecord:
    """Near-duplicate of ``founder`` (>~90% identity) to form a redundancy family.

    CDS edits are strictly *silent* (synonymous codon swaps preserving the
    protein and frame); UTR edits are sparse point substitutions. Start and
    stop codons are never touched, so validity is preserved by construction.
    """
    codons = [founder.cds[i:i + 3] for i in range(0, len(founder.cds), 3)]
    for i in range(1, len(codons) - 1):  # skip AUG start & terminal stop
        if rng.random() < 0.06:
            aa = CODON_TABLE[codons[i]]
            synonyms = _SENSE_SYNONYMS.get(aa, [codons[i]])
            codons[i] = rng.choice(synonyms)
    cds = "".join(codons)

    def _point_mutate(utr: str) -> str:
        chars = list(utr)
        for j in range(len(chars)):
            if rng.random() < 0.03:
                chars[j] = rng.choice([c for c in "ACGU" if c != chars[j]])
        return "".join(chars)

    return MRNARecord(
        transcript_id=f"SYN_{family_id:05d}_{member}",
        five_utr=_point_mutate(founder.five_utr),
        cds=cds,
        three_utr=_point_mutate(founder.three_utr),
        species=founder.species,
    )


def synthesize_corpus(n: int, seed: int = 0) -> List[MRNARecord]:
    """Generate ``n`` biologically-plausible synthetic mRNAs, fully offline.

    Records are grouped into near-duplicate families (a founder plus 0-5
    silent-mutation variants), producing realistic redundancy for the
    family-disjoint splitter. Every returned record satisfies
    :func:`is_valid_cds` and respects nothing about length caps beyond the
    generator's own bounds (all fit the default caps). Deterministic in ``seed``.

    Complexity: O(n * L) for mean transcript length L.
    """
    if n <= 0:
        return []
    rng = random.Random(seed)
    records: List[MRNARecord] = []
    family_id = 0
    while len(records) < n:
        founder = _make_founder(rng, family_id)
        # Defensive: regenerate on the astronomically rare invalid draw.
        while not is_valid_cds(founder.cds):
            founder = _make_founder(rng, family_id)
        family_size = rng.randint(1, 6)
        for member in range(family_size):
            if len(records) >= n:
                break
            rec = founder if member == 0 else _mutate_variant(rng, founder, family_id, member)
            records.append(rec)
        family_id += 1
    return records[:n]


__all__ = [
    "MRNA_DATASETS",
    "download_dataset",
    "verify_dataset",
    "get_dataset_info",
    "list_available_datasets",
    "compute_and_cache_checksums",
    "load_cached_checksums",
    "iter_fasta",
    "record_from_annotation",
    "parse_gencode_cds_range",
    "transcript_id_from_gencode_header",
    "records_from_gencode_fasta",
    "iter_genbank_records",
    "parse_genbank_cds_location",
    "records_from_refseq_genbank",
    "write_records_jsonl",
    "load_records_jsonl",
    "synthesize_corpus",
]
