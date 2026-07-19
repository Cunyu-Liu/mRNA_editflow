"""Frozen, untruncated public-corpus reconstruction contracts.

The legacy public pipeline intentionally produces model-sized records.  This
module keeps a separate scientific source of truth: complete transport-verified
raw objects, full-length ORF-valid canonical records, and explicitly lossy
derived views with row-level lineage.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
import ssl
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from mrna_editflow.core.constants import START_CODON, translate
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.download_mrna import (
    _genbank_accession,
    _genbank_cds_locations,
    _genbank_origin_sequence,
    _open_url,
    iter_fasta,
    iter_genbank_records,
    parse_gencode_cds_range,
    parse_genbank_cds_location,
    record_from_annotation,
    transcript_id_from_gencode_header,
    write_records_jsonl,
)


class ReconstructionError(ValueError):
    """Base class for reconstruction contract failures."""


class RawIntegrityError(ReconstructionError):
    """A raw artifact is missing, truncated, or differs from its binding."""


class CanonicalRecordError(ReconstructionError):
    """Canonical identities or record/metadata alignment are invalid."""


class LineageError(ReconstructionError):
    """A derived view cannot be traced exactly to its canonical parent."""


class FamilyAssignmentError(ReconstructionError):
    """Family assignments do not cover the records universe."""


@dataclass(frozen=True)
class CanonicalInput:
    record: MRNARecord
    source: str
    source_accession: str
    source_record_index: int
    gene_id: str = ""
    gene_symbol: str = ""
    protein_id: str = ""
    source_file: str = ""

    @property
    def canonical_id(self) -> str:
        return f"{self.source}:{self.source_accession}"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(text: str) -> str:
    value = str(text).lower()
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise RawIntegrityError("expected_sha256 must be a 64-character SHA-256")
    return value


def verify_raw_artifact(
    path: str | Path,
    *,
    expected_sha256: Optional[str] = None,
    expected_size_bytes: Optional[int] = None,
) -> dict[str, object]:
    """Verify exact bytes and, for gzip files, consume the complete stream."""
    target = Path(path).resolve()
    if not target.is_file():
        raise RawIntegrityError(f"raw artifact does not exist: {target}")
    size = target.stat().st_size
    if expected_size_bytes is not None and size != int(expected_size_bytes):
        raise RawIntegrityError(
            f"raw artifact size mismatch: expected {expected_size_bytes}, got {size}"
        )
    digest = sha256_file(target)
    if expected_sha256 is not None and digest != _require_sha(expected_sha256):
        raise RawIntegrityError(
            f"raw artifact SHA mismatch: expected {expected_sha256}, got {digest}"
        )
    is_gzip = target.name.endswith(".gz") or target.name.endswith(".gz.part")
    uncompressed = size
    if is_gzip:
        uncompressed = 0
        try:
            with gzip.open(target, "rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    uncompressed += len(chunk)
        except (EOFError, OSError, gzip.BadGzipFile) as exc:
            raise RawIntegrityError(f"incomplete or invalid gzip stream: {target}") from exc
    return {
        "path": str(target),
        "size_bytes": size,
        "sha256": digest,
        "gzip_complete": True if is_gzip else None,
        "uncompressed_size_bytes": uncompressed,
    }


def _normalize(seq: str) -> str:
    return "".join(str(seq or "").upper().replace("T", "U").split())


def _classify_cds(cds: str) -> str:
    if not cds:
        return "empty_cds"
    if any(ch not in "ACGU" for ch in cds):
        return "illegal_chars"
    if len(cds) % 3:
        return "frame_not_multiple_of_3"
    if len(cds) < 6:
        return "cds_too_short"
    if cds[:3] != START_CODON:
        return "bad_start_codon"
    protein = translate(cds)
    if not protein.endswith("*"):
        return "no_terminal_stop"
    if "*" in protein[:-1]:
        return "internal_stop"
    return "ok"


def canonicalize_records(
    rows: Sequence[CanonicalInput],
) -> tuple[list[MRNARecord], list[dict[str, object]], dict[str, int]]:
    """Normalize and validate full records without any length cap or truncation."""
    reasons = (
        "empty_cds",
        "illegal_chars",
        "frame_not_multiple_of_3",
        "cds_too_short",
        "bad_start_codon",
        "no_terminal_stop",
        "internal_stop",
    )
    stats = {reason: 0 for reason in reasons}
    stats.update({"total": len(rows), "kept": 0, "truncated_5utr": 0, "truncated_3utr": 0})
    records: list[MRNARecord] = []
    metadata: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in rows:
        canonical_id = item.canonical_id
        if not item.source.strip() or not item.source_accession.strip():
            raise CanonicalRecordError("source and source_accession must be non-empty")
        if canonical_id in seen:
            raise CanonicalRecordError(f"duplicate canonical identity: {canonical_id}")
        seen.add(canonical_id)
        five = _normalize(item.record.five_utr)
        cds = _normalize(item.record.cds)
        three = _normalize(item.record.three_utr)
        if any(ch not in "ACGU" for ch in five + three):
            stats["illegal_chars"] += 1
            continue
        reason = _classify_cds(cds)
        if reason != "ok":
            stats[reason] += 1
            continue
        record = MRNARecord(
            transcript_id=item.source_accession,
            five_utr=five,
            cds=cds,
            three_utr=three,
            species=item.record.species or "human",
        )
        records.append(record)
        metadata.append(
            {
                "canonical_id": canonical_id,
                "source": item.source,
                "source_accession": item.source_accession,
                "source_record_index": int(item.source_record_index),
                "gene_id": str(item.gene_id or ""),
                "gene_symbol": str(item.gene_symbol or ""),
                "protein_id": str(item.protein_id or ""),
                "source_file": str(item.source_file or ""),
                "five_utr_len": len(five),
                "cds_len": len(cds),
                "three_utr_len": len(three),
                "sequence_sha256": hashlib.sha256(record.seq.encode("ascii")).hexdigest(),
                "protein_sha256": hashlib.sha256(translate(cds).encode("ascii")).hexdigest(),
            }
        )
        stats["kept"] += 1
    return records, metadata, stats


def _validate_alignment(
    records: Sequence[MRNARecord], metadata: Sequence[Mapping[str, object]]
) -> None:
    if len(records) != len(metadata):
        raise CanonicalRecordError("records and metadata counts differ")
    seen: set[str] = set()
    for index, (record, meta) in enumerate(zip(records, metadata)):
        cid = str(meta.get("canonical_id") or "")
        accession = str(meta.get("source_accession") or cid.split(":", 1)[-1])
        if not cid or cid in seen:
            raise CanonicalRecordError(f"invalid canonical_id at row {index}")
        if record.transcript_id not in (cid, accession):
            raise CanonicalRecordError(f"record/metadata identifier mismatch at row {index}")
        seen.add(cid)


def derive_model_view(
    canonical_records: Sequence[MRNARecord],
    metadata: Sequence[Mapping[str, object]],
    *,
    max_5utr: int,
    max_cds: int,
    max_3utr: int,
    qualified_ids: bool = True,
) -> tuple[list[MRNARecord], list[dict[str, object]], dict[str, int]]:
    """Apply declared model caps and produce total row-level parent lineage."""
    _validate_alignment(canonical_records, metadata)
    if min(int(max_5utr), int(max_cds), int(max_3utr)) < 0:
        raise LineageError("derived-view caps must be non-negative")
    view: list[MRNARecord] = []
    lineage: list[dict[str, object]] = []
    stats = {
        "total_canonical": len(canonical_records),
        "kept": 0,
        "dropped_cds_too_long": 0,
        "truncated_5utr": 0,
        "truncated_3utr": 0,
    }
    for index, (record, meta) in enumerate(zip(canonical_records, metadata)):
        if len(record.cds) > int(max_cds):
            stats["dropped_cds_too_long"] += 1
            continue
        trunc5 = len(record.five_utr) > int(max_5utr)
        trunc3 = len(record.three_utr) > int(max_3utr)
        five = record.five_utr[-int(max_5utr):] if trunc5 and max_5utr else ("" if trunc5 else record.five_utr)
        three = record.three_utr[: int(max_3utr)] if trunc3 else record.three_utr
        canonical_id = str(meta["canonical_id"])
        derived = MRNARecord(
            transcript_id=canonical_id if qualified_ids else record.transcript_id,
            five_utr=five,
            cds=record.cds,
            three_utr=three,
            species=record.species,
        )
        view.append(derived)
        lineage.append(
            {
                "derived_index": len(view) - 1,
                "canonical_index": index,
                "canonical_id": canonical_id,
                "derived_transcript_id": derived.transcript_id,
                "truncated_5utr": trunc5,
                "truncated_3utr": trunc3,
                "canonical_sequence_sha256": str(meta.get("sequence_sha256") or hashlib.sha256(record.seq.encode("ascii")).hexdigest()),
                "derived_sequence_sha256": hashlib.sha256(derived.seq.encode("ascii")).hexdigest(),
            }
        )
        stats["kept"] += 1
        stats["truncated_5utr"] += int(trunc5)
        stats["truncated_3utr"] += int(trunc3)
    return view, lineage, stats


def _normalized_gene_symbol(value: object) -> str:
    return re.sub(r"[^A-Z0-9_.-]", "", str(value or "").upper())


def build_family_assignments(
    records: Sequence[MRNARecord],
    metadata: Sequence[Mapping[str, object]],
) -> tuple[list[int], list[dict[str, object]]]:
    """Union exact genes, complete RNAs, and translated proteins deterministically."""
    _validate_alignment(records, metadata)
    n = len(records)
    parent = list(range(n))

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != root:
            parent[value], value = root, parent[value]
        return root

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    first: dict[tuple[str, str], int] = {}
    for index, (record, meta) in enumerate(zip(records, metadata)):
        gene = _normalized_gene_symbol(meta.get("gene_symbol"))
        keys = [
            ("rna_sha256", hashlib.sha256(record.seq.encode("ascii")).hexdigest()),
            ("protein_sha256", hashlib.sha256(translate(record.cds).encode("ascii")).hexdigest()),
        ]
        if gene:
            keys.append(("gene_symbol", gene))
        for key in keys:
            if key in first:
                union(index, first[key])
            else:
                first[key] = index

    groups: dict[int, list[int]] = {}
    for index in range(n):
        groups.setdefault(find(index), []).append(index)
    ordered = sorted(groups.values(), key=lambda members: min(members))
    assignments = [-1] * n
    evidence: list[dict[str, object]] = []
    for cluster_id, members in enumerate(ordered):
        for index in members:
            assignments[index] = cluster_id
        sources = sorted({str(metadata[index].get("source") or "") for index in members})
        genes = sorted(
            {value for index in members if (value := _normalized_gene_symbol(metadata[index].get("gene_symbol")))}
        )
        evidence.append(
            {
                "cluster_id": cluster_id,
                "count": len(members),
                "members": members,
                "sources": sources,
                "n_sources": len(sources),
                "gene_symbols": genes,
            }
        )
    if any(value < 0 for value in assignments):
        raise FamilyAssignmentError("family assignments did not cover every record")
    return assignments, evidence


def _source_name(meta: Mapping[str, object]) -> str:
    return str(meta.get("source") or "").lower()


def build_cross_source_roles(
    assignments: Sequence[int],
    metadata: Sequence[Mapping[str, object]],
    *,
    seed: int,
) -> tuple[dict[str, list[int]], list[int]]:
    """GENCODE train and non-overlapping RefSeq validation/test source holdout."""
    if len(assignments) != len(metadata):
        raise FamilyAssignmentError("assignments and metadata counts differ")
    train = [i for i, meta in enumerate(metadata) if "gencode" in _source_name(meta)]
    train_clusters = {int(assignments[i]) for i in train}
    excluded = [
        i
        for i, meta in enumerate(metadata)
        if "refseq" in _source_name(meta) and int(assignments[i]) in train_clusters
    ]
    excluded_set = set(excluded)
    remaining_clusters: dict[int, list[int]] = {}
    for i, meta in enumerate(metadata):
        if "refseq" not in _source_name(meta) or i in excluded_set:
            continue
        remaining_clusters.setdefault(int(assignments[i]), []).append(i)
    order = sorted(
        remaining_clusters,
        key=lambda cluster: hashlib.sha256(f"{int(seed)}:{cluster}".encode("ascii")).hexdigest(),
    )
    val: list[int] = []
    test: list[int] = []
    for position, cluster in enumerate(order):
        (val if position % 2 == 0 else test).extend(remaining_clusters[cluster])
    roles = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
    return roles, sorted(excluded)


def parse_gencode_inputs(
    path: str | Path, *, source: str = "gencode_v45", limit: Optional[int] = None
) -> tuple[list[CanonicalInput], dict[str, int]]:
    """Extract full GENCODE regions and source metadata in raw-file order."""
    rows: list[CanonicalInput] = []
    stats = {
        "total_entries": 0,
        "missing_cds_annotation": 0,
        "cds_out_of_range": 0,
        "emitted": 0,
    }
    for raw_index, (header, sequence) in enumerate(iter_fasta(str(path))):
        stats["total_entries"] += 1
        cds_range = parse_gencode_cds_range(header)
        if cds_range is None:
            stats["missing_cds_annotation"] += 1
            continue
        start, end = cds_range
        if end > len(sequence):
            stats["cds_out_of_range"] += 1
            continue
        fields = str(header).split("|")
        accession = transcript_id_from_gencode_header(header)
        row = CanonicalInput(
            record=record_from_annotation(accession, sequence, start, end),
            source=source,
            source_accession=accession,
            source_record_index=raw_index,
            gene_id=fields[1] if len(fields) > 1 else "",
            gene_symbol=fields[5] if len(fields) > 5 else "",
            protein_id="",
            source_file=Path(path).name,
        )
        rows.append(row)
        stats["emitted"] += 1
        if limit is not None and len(rows) >= int(limit):
            break
    return rows, stats


def _qualifier(lines: Sequence[str], name: str) -> str:
    pattern = re.compile(rf'/{re.escape(name)}="([^"]+)"')
    for line in lines:
        match = pattern.search(line)
        if match:
            return match.group(1)
    return ""


def _gene_id_qualifier(lines: Sequence[str]) -> str:
    pattern = re.compile(r'/db_xref="GeneID:([^"]+)"')
    for line in lines:
        match = pattern.search(line)
        if match:
            return match.group(1)
    return ""


def parse_refseq_inputs(
    path: str | Path, *, source: str = "refseq_human_rna", limit: Optional[int] = None
) -> tuple[list[CanonicalInput], dict[str, int]]:
    """Conservatively extract one contiguous plus-strand CDS per RefSeq entry."""
    rows: list[CanonicalInput] = []
    stats = {
        "total_entries": 0,
        "missing_accession": 0,
        "ambiguous_or_unsupported_cds": 0,
        "cds_out_of_range": 0,
        "emitted": 0,
    }
    for raw_index, lines in enumerate(iter_genbank_records(str(path))):
        stats["total_entries"] += 1
        accession = _genbank_accession(lines)
        if not accession:
            stats["missing_accession"] += 1
            continue
        parsed = [
            value
            for value in (
                parse_genbank_cds_location(location)
                for location in _genbank_cds_locations(lines)
            )
            if value is not None
        ]
        if len(parsed) != 1:
            stats["ambiguous_or_unsupported_cds"] += 1
            continue
        sequence = _genbank_origin_sequence(lines)
        start, end = parsed[0]
        if end > len(sequence):
            stats["cds_out_of_range"] += 1
            continue
        rows.append(
            CanonicalInput(
                record=record_from_annotation(accession, sequence, start, end),
                source=source,
                source_accession=accession,
                source_record_index=raw_index,
                gene_id=_gene_id_qualifier(lines),
                gene_symbol=_qualifier(lines, "gene"),
                protein_id=_qualifier(lines, "protein_id"),
                source_file=Path(path).name,
            )
        )
        stats["emitted"] += 1
        if limit is not None and len(rows) >= int(limit):
            break
    return rows, stats


def acquire_frozen_file(
    destination: str | Path,
    *,
    source_path: Optional[str | Path] = None,
    url: Optional[str] = None,
    expected_sha256: Optional[str] = None,
    expected_size_bytes: Optional[int] = None,
) -> dict[str, object]:
    """Copy or download into a new path and atomically promote after verification."""
    if (source_path is None) == (url is None):
        raise RawIntegrityError("provide exactly one of source_path or url")
    target = Path(destination).resolve()
    if target.exists():
        evidence = verify_raw_artifact(
            target,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
        )
        evidence.update({"acquisition": "preexisting_verified", "url": url})
        return evidence
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".part")
    if temp.exists():
        raise RawIntegrityError(f"stale acquisition temp file requires manual review: {temp}")
    try:
        if source_path is not None:
            with open(Path(source_path).resolve(), "rb") as src, open(temp, "xb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            acquisition = "local_copy"
        else:
            assert url is not None
            with _open_url(url) as response, open(temp, "xb") as dst:
                shutil.copyfileobj(response, dst, length=1024 * 1024)
            acquisition = "https_download"
        evidence = verify_raw_artifact(
            temp,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
        )
        os.replace(temp, target)
    except Exception:
        if temp.exists():
            temp.unlink()
        raise
    evidence["path"] = str(target)
    evidence.update({"acquisition": acquisition, "url": url})
    return evidence


def _open_range_url(url: str, start: int, end: int):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "mrna-editflow-data-reconstruction/1",
            "Range": f"bytes={start}-{end}",
        },
    )
    try:
        return urllib.request.urlopen(request, timeout=90)
    except ssl.SSLCertVerificationError:
        import certifi  # type: ignore

        context = ssl.create_default_context(cafile=certifi.where())
        return urllib.request.urlopen(request, timeout=90, context=context)


def download_http_ranges(
    destination: str | Path,
    *,
    url: str,
    expected_size_bytes: int,
    prefix_path: Optional[str | Path] = None,
    workers: int = 12,
    use_curl: bool = False,
) -> dict[str, object]:
    """Resume a verified prefix with parallel exact HTTP byte ranges.

    Range pieces are individually size checked, assembled in order, then the
    complete object must pass the gzip transport check before atomic promotion.
    Completed pieces are reusable after a transient failure.
    """
    target = Path(destination).resolve()
    if target.exists():
        raise RawIntegrityError(f"parallel download destination exists: {target}")
    total = int(expected_size_bytes)
    if total <= 0 or int(workers) <= 0:
        raise RawIntegrityError("expected size and workers must be positive")
    prefix = Path(prefix_path).resolve() if prefix_path is not None else None
    prefix_size = 0
    if prefix is not None:
        if not prefix.is_file():
            raise RawIntegrityError(f"resume prefix does not exist: {prefix}")
        prefix_size = prefix.stat().st_size
        if prefix_size >= total:
            raise RawIntegrityError("resume prefix is not shorter than expected object")
    parts_dir = target.with_name(target.name + ".range_parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    remaining = total - prefix_size
    chunk_size = max(1, (remaining + int(workers) - 1) // int(workers))
    ranges: list[tuple[int, int, Path]] = []
    cursor = prefix_size
    part_number = 0
    while cursor < total:
        end = min(total - 1, cursor + chunk_size - 1)
        ranges.append((cursor, end, parts_dir / f"part-{part_number:03d}-{cursor}-{end}.bin"))
        cursor = end + 1
        part_number += 1

    def fetch(spec: tuple[int, int, Path]) -> tuple[int, int, Path]:
        start, end, part = spec
        expected = end - start + 1
        if part.is_file() and part.stat().st_size == expected:
            return spec
        temp = part.with_name(part.name + ".part")
        if temp.exists() and temp.stat().st_size > expected:
            temp.unlink()
        if use_curl:
            stalled = 0
            while (temp.stat().st_size if temp.exists() else 0) < expected:
                have = temp.stat().st_size if temp.exists() else 0
                range_start = start + have
                segment = temp.with_name(temp.name + ".segment")
                if segment.exists():
                    segment.unlink()
                result = subprocess.run(
                    [
                        "curl", "-fsSL", "--connect-timeout", "30", "--max-time", "300",
                        "--range", f"{range_start}-{end}", "-o", str(segment), url,
                    ],
                    capture_output=True,
                    text=True,
                )
                gained = segment.stat().st_size if segment.exists() else 0
                if gained:
                    with open(temp, "ab") as dst, open(segment, "rb") as src:
                        shutil.copyfileobj(src, dst, length=1024 * 1024)
                    segment.unlink()
                    stalled = 0
                else:
                    stalled += 1
                if result.returncode != 0 and not gained and stalled < 20:
                    time.sleep(min(30, stalled * 2))
                if result.returncode != 0 and not gained and stalled >= 20:
                    raise RawIntegrityError(
                        f"curl range {range_start}-{end} made no progress: {result.stderr[-1000:]}"
                    )
                if temp.exists() and temp.stat().st_size > expected:
                    raise RawIntegrityError(f"curl range {start}-{end} exceeded expected size")
        else:
            if temp.exists():
                temp.unlink()
            with _open_range_url(url, start, end) as response, open(temp, "xb") as dst:
                content_range = str(response.headers.get("Content-Range") or "")
                if response.status != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
                    raise RawIntegrityError(
                        f"server did not honor exact range {start}-{end}: {response.status} {content_range}"
                    )
                shutil.copyfileobj(response, dst, length=1024 * 1024)
        if temp.stat().st_size != expected:
            raise RawIntegrityError(
                f"range {start}-{end} size mismatch: expected {expected}, got {temp.stat().st_size}"
            )
        os.replace(temp, part)
        return spec

    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        futures = [pool.submit(fetch, spec) for spec in ranges]
        for future in as_completed(futures):
            future.result()

    target.parent.mkdir(parents=True, exist_ok=True)
    assembled = target.with_name(target.name + ".part")
    if assembled.exists():
        raise RawIntegrityError(f"stale assembled temp file: {assembled}")
    with open(assembled, "xb") as dst:
        if prefix is not None:
            with open(prefix, "rb") as src:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
        for _start, _end, part in ranges:
            with open(part, "rb") as src:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
    evidence = verify_raw_artifact(assembled, expected_size_bytes=total)
    os.replace(assembled, target)
    evidence["path"] = str(target)
    evidence.update(
        {
            "acquisition": "https_parallel_range_resume" if prefix is not None else "https_parallel_ranges",
            "url": url,
            "resume_prefix_path": str(prefix) if prefix is not None else None,
            "resume_prefix_size_bytes": prefix_size,
            "workers": int(workers),
            "n_ranges": len(ranges),
            "range_transport": "curl" if use_curl else "urllib",
        }
    )
    return evidence


def download_refseq_release(
    destination_dir: str | Path,
    *,
    catalog: Mapping[str, Mapping[str, object]],
    concurrent_files: int = 3,
    workers_per_file: int = 4,
) -> dict[str, object]:
    """Freeze every catalogued RefSeq partition with bounded parallelism."""
    root = Path(destination_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if len(catalog) != 15:
        raise RawIntegrityError(f"RefSeq release catalog must contain 15 partitions, got {len(catalog)}")

    def acquire(item: tuple[str, Mapping[str, object]]) -> dict[str, object]:
        filename, metadata = item
        if not re.fullmatch(r"human\.\d+\.rna\.gbff\.gz", filename):
            raise RawIntegrityError(f"invalid RefSeq partition name: {filename}")
        size = int(metadata["size_bytes"])
        url = str(metadata["url"])
        target = root / filename
        if target.exists():
            result = verify_raw_artifact(target, expected_size_bytes=size)
            result.update({"acquisition": "preexisting_verified", "url": url})
        else:
            result = download_http_ranges(
                target,
                url=url,
                expected_size_bytes=size,
                workers=workers_per_file,
                use_curl=True,
            )
        result["official_last_modified"] = metadata.get("last_modified")
        return result

    artifacts: list[dict[str, object]] = []
    ordered_items = sorted(catalog.items(), key=lambda item: int(item[0].split(".")[1]))
    scheduled_items = sorted(
        ordered_items,
        key=lambda item: (
            (root / item[0]).exists(),
            int(item[0].split(".")[1]),
        ),
    )
    with ThreadPoolExecutor(max_workers=int(concurrent_files)) as pool:
        futures = {pool.submit(acquire, item): item[0] for item in scheduled_items}
        by_name: dict[str, dict[str, object]] = {}
        for future in as_completed(futures):
            by_name[futures[future]] = future.result()
    artifacts = [by_name[name] for name, _metadata in ordered_items]
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "refseq_human_rna_complete_release_catalog",
        "official_directory_url": "https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/mRNA_Prot/",
        "artifact_count": len(artifacts),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in artifacts),
        "artifacts": artifacts,
    }
    catalog_path = root / "release_catalog.json"
    if catalog_path.exists():
        raise RawIntegrityError(f"refusing to overwrite release catalog: {catalog_path}")
    with open(catalog_path, "x", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    payload["catalog_path"] = str(catalog_path)
    payload["catalog_sha256"] = sha256_file(catalog_path)
    return payload


def _canonical_json_sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_source_bundle(
    *,
    source: str,
    raw_path: str | Path | Sequence[str | Path],
    output_dir: str | Path,
    max_5utr: int = 128,
    max_cds: int = 1536,
    max_3utr: int = 256,
    limit: Optional[int] = None,
    acquisition_evidence: Optional[Mapping[str, object]] = None,
    trust_acquisition_evidence: bool = False,
) -> dict[str, object]:
    """Build one immutable canonical corpus and one model-capped derived view."""
    if isinstance(raw_path, (str, Path)):
        candidate = Path(raw_path)
        if candidate.is_dir():
            raw_paths = sorted(
                candidate.glob("human.*.rna.gbff.gz"),
                key=lambda path: int(path.name.split(".")[1]),
            )
        else:
            raw_paths = [candidate]
    else:
        raw_paths = [Path(value) for value in raw_path]
    if not raw_paths:
        raise RawIntegrityError("source bundle has no raw artifacts")
    if trust_acquisition_evidence:
        evidence_rows = list((acquisition_evidence or {}).get("artifacts", []))
        by_name = {Path(str(row.get("path"))).name: row for row in evidence_rows}
        if set(by_name) != {path.name for path in raw_paths}:
            raise RawIntegrityError("trusted acquisition evidence does not cover raw artifacts")
        raw_artifacts = []
        for path in raw_paths:
            row = dict(by_name[path.name])
            expected_size = int(row["size_bytes"])
            if path.stat().st_size != expected_size:
                raise RawIntegrityError(f"trusted raw size changed: {path}")
            actual_sha = sha256_file(path)
            if actual_sha != str(row["sha256"]):
                raise RawIntegrityError(f"trusted raw SHA changed: {path}")
            if row.get("gzip_complete") is not True:
                raise RawIntegrityError(f"trusted evidence lacks gzip completion: {path}")
            row["path"] = str(path.resolve())
            row["sha256"] = actual_sha
            raw_artifacts.append(row)
    else:
        raw_artifacts = [verify_raw_artifact(path) for path in raw_paths]
    inputs: list[CanonicalInput] = []
    parser_stats: dict[str, int] = {}
    raw_offset = 0
    remaining_limit = limit
    for path in raw_paths:
        if "gencode" in source.lower():
            parsed, file_stats = parse_gencode_inputs(path, source=source, limit=remaining_limit)
        elif "refseq" in source.lower():
            parsed, file_stats = parse_refseq_inputs(path, source=source, limit=remaining_limit)
        else:
            raise ReconstructionError(f"no reconstruction parser for source {source!r}")
        for item in parsed:
            inputs.append(
                CanonicalInput(
                    record=item.record,
                    source=item.source,
                    source_accession=item.source_accession,
                    source_record_index=raw_offset + item.source_record_index,
                    gene_id=item.gene_id,
                    gene_symbol=item.gene_symbol,
                    protein_id=item.protein_id,
                    source_file=item.source_file,
                )
            )
        raw_offset += int(file_stats.get("total_entries", 0))
        for key, value in file_stats.items():
            parser_stats[key] = parser_stats.get(key, 0) + int(value)
        if remaining_limit is not None:
            remaining_limit = max(0, int(remaining_limit) - len(parsed))
            if remaining_limit == 0:
                break
    canonical, metadata, canonical_stats = canonicalize_records(inputs)
    derived, lineage, derived_stats = derive_model_view(
        canonical,
        metadata,
        max_5utr=max_5utr,
        max_cds=max_cds,
        max_3utr=max_3utr,
        qualified_ids=False,
    )
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "canonical_records": root / "canonical.records.jsonl",
        "canonical_metadata": root / "canonical.metadata.jsonl",
        "model_view_records": root / "model_view.records.jsonl",
        "model_view_lineage": root / "model_view.lineage.jsonl",
        "manifest": root / "reconstruction_manifest.json",
    }
    for path in paths.values():
        if path.exists():
            raise ReconstructionError(f"refusing to overwrite frozen artifact: {path}")
    write_records_jsonl(canonical, str(paths["canonical_records"]))
    write_jsonl(metadata, paths["canonical_metadata"])
    write_records_jsonl(derived, str(paths["model_view_records"]))
    write_jsonl(lineage, paths["model_view_lineage"])
    registry = (
        "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/gencode.v45.pc_transcripts.fa.gz"
        if "gencode" in source.lower()
        else "https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/mRNA_Prot/"
    )
    complete_release = "refseq" not in source.lower() or len(raw_artifacts) == 15
    source_blockers = ["cross_source_family_split_not_yet_bound"]
    if not complete_release:
        source_blockers.append("refseq_release_partition_set_incomplete")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "p0_data_reconstruction_source_bundle",
        "source": source,
        "raw": {
            "artifacts": raw_artifacts,
            "artifact_count": len(raw_artifacts),
            "total_size_bytes": sum(int(row["size_bytes"]) for row in raw_artifacts),
            **(raw_artifacts[0] if len(raw_artifacts) == 1 else {}),
            "official_url": registry,
            "acquisition": dict(acquisition_evidence or {}),
            "release_binding_policy": "official URL plus frozen local size and SHA-256",
        },
        "parser_stats": parser_stats,
        "complete_release_partition_set": complete_release,
        "canonical": {
            "records_path": str(paths["canonical_records"]),
            "records_sha256": sha256_file(paths["canonical_records"]),
            "metadata_path": str(paths["canonical_metadata"]),
            "metadata_sha256": sha256_file(paths["canonical_metadata"]),
            "count": len(canonical),
            "stats": canonical_stats,
        },
        "derived_views": {
            "model_capped_v1": {
                "records_path": str(paths["model_view_records"]),
                "records_sha256": sha256_file(paths["model_view_records"]),
                "lineage_path": str(paths["model_view_lineage"]),
                "lineage_sha256": sha256_file(paths["model_view_lineage"]),
                "count": len(derived),
                "caps": {"max_5utr": max_5utr, "max_cds": max_cds, "max_3utr": max_3utr},
                "stats": derived_stats,
            }
        },
        "code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(__file__),
            "git_commit": None,
        },
        "paper_eligible": False,
        "block_reasons": source_blockers,
    }
    with open(paths["manifest"], "x", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    manifest["manifest_path"] = str(paths["manifest"])
    manifest["manifest_sha256"] = sha256_file(paths["manifest"])
    verify_source_bundle(paths["manifest"])
    return manifest


def verify_source_bundle(manifest_path: str | Path) -> dict[str, object]:
    """Re-hash a source bundle and verify canonical/derived lineage semantics."""
    path = Path(manifest_path).resolve()
    with open(path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("schema_version") != 1:
        raise ReconstructionError("unsupported reconstruction manifest schema")
    canonical = manifest.get("canonical", {})
    view = manifest.get("derived_views", {}).get("model_capped_v1", {})
    bindings = (
        (canonical, "records_path", "records_sha256"),
        (canonical, "metadata_path", "metadata_sha256"),
        (view, "records_path", "records_sha256"),
        (view, "lineage_path", "lineage_sha256"),
    )
    for section, path_key, sha_key in bindings:
        artifact = Path(str(section.get(path_key))).resolve()
        if not artifact.is_file() or sha256_file(artifact) != section.get(sha_key):
            raise ReconstructionError(f"bundle artifact binding failed: {path_key}")
    with open(canonical["metadata_path"], "r", encoding="utf-8") as fh:
        metadata = [json.loads(line) for line in fh if line.strip()]
    with open(view["lineage_path"], "r", encoding="utf-8") as fh:
        lineage = [json.loads(line) for line in fh if line.strip()]
    from mrna_editflow.data.download_mrna import load_records_jsonl

    canonical_records = load_records_jsonl(canonical["records_path"])
    derived_records = load_records_jsonl(view["records_path"])
    _validate_alignment(canonical_records, metadata)
    if len(derived_records) != len(lineage) or len(canonical_records) != int(canonical["count"]):
        raise LineageError("bundle counts or lineage coverage differ")
    for expected_index, (record, row) in enumerate(zip(derived_records, lineage)):
        if int(row.get("derived_index", -1)) != expected_index:
            raise LineageError("derived lineage order mismatch")
        parent_index = int(row.get("canonical_index", -1))
        if parent_index < 0 or parent_index >= len(canonical_records):
            raise LineageError("derived lineage parent out of range")
        parent = canonical_records[parent_index]
        if record.cds != parent.cds:
            raise LineageError("derived CDS changed from canonical parent")
        if not parent.five_utr.endswith(record.five_utr) or not parent.three_utr.startswith(record.three_utr):
            raise LineageError("derived UTR is not the declared CDS-proximal view")
        if hashlib.sha256(record.seq.encode("ascii")).hexdigest() != row.get("derived_sequence_sha256"):
            raise LineageError("derived lineage sequence digest mismatch")
    return manifest


def assign_family_roles(
    assignments: Sequence[int], *, seed: int, train_frac: float = 0.8, val_frac: float = 0.1
) -> dict[str, list[int]]:
    """Assign whole families to balanced train/validation/test roles."""
    if not 0 < train_frac < 1 or not 0 <= val_frac < 1 or train_frac + val_frac >= 1:
        raise FamilyAssignmentError("invalid split fractions")
    clusters: dict[int, list[int]] = {}
    for index, cluster in enumerate(assignments):
        clusters.setdefault(int(cluster), []).append(index)
    total = len(assignments)
    targets = {"train": total * train_frac, "val": total * val_frac, "test": total * (1 - train_frac - val_frac)}
    counts = {name: 0 for name in targets}
    roles = {name: [] for name in targets}
    order = sorted(
        clusters,
        key=lambda cluster: (
            -len(clusters[cluster]),
            hashlib.sha256(f"{int(seed)}:{cluster}".encode("ascii")).hexdigest(),
        ),
    )
    for cluster in order:
        role = max(targets, key=lambda name: targets[name] - counts[name])
        roles[role].extend(clusters[cluster])
        counts[role] += len(clusters[cluster])
    return {name: sorted(indices) for name, indices in roles.items()}


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _derived_rows_from_bundle(
    manifest: Mapping[str, object],
) -> tuple[list[MRNARecord], list[dict[str, object]]]:
    from mrna_editflow.data.download_mrna import load_records_jsonl

    canonical = manifest["canonical"]
    view = manifest["derived_views"]["model_capped_v1"]
    metadata = _read_jsonl(canonical["metadata_path"])
    lineage = _read_jsonl(view["lineage_path"])
    records = load_records_jsonl(view["records_path"])
    selected = [metadata[int(row["canonical_index"])] for row in lineage]
    if len(records) != len(selected):
        raise LineageError("derived bundle rows and selected metadata differ")
    qualified: list[MRNARecord] = []
    for record, meta in zip(records, selected):
        qualified.append(
            MRNARecord(
                transcript_id=str(meta["canonical_id"]),
                five_utr=record.five_utr,
                cds=record.cds,
                three_utr=record.three_utr,
                species=record.species,
            )
        )
    return qualified, selected


def _write_indices(path: Path, indices: Sequence[int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as fh:
        for index in indices:
            fh.write(f"{int(index)}\n")
    return str(path)


def _split_exact_overlap(records: Sequence[MRNARecord], roles: Mapping[str, Sequence[int]]) -> int:
    sets = {name: {records[i].seq for i in indices} for name, indices in roles.items()}
    return sum(
        len(sets[left] & sets[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    )


def _write_split_package(
    *,
    dataset_id: str,
    records_path: str | Path,
    records: Sequence[MRNARecord],
    assignments: Sequence[int],
    roles: Mapping[str, Sequence[int]],
    output_dir: str | Path,
    seed: int,
    algorithm: str,
    excluded: Optional[Sequence[int]] = None,
    excluded_reason: Optional[str] = None,
    additional_block_reasons: Sequence[str] = (),
) -> dict[str, object]:
    from mrna_editflow.data.split_contract import (
        build_split_manifest,
        load_and_verify_split_manifest,
        sha256_file as contract_sha256_file,
    )

    root = Path(output_dir).resolve()
    if root.exists():
        raise ReconstructionError(f"refusing to overwrite split package: {root}")
    root.mkdir(parents=True)
    role_paths = {
        role: _write_indices(root / f"{role}.idx", roles[role])
        for role in ("train", "val", "test")
    }
    cluster_path = root / "cluster_assignments.json"
    with open(cluster_path, "x", encoding="utf-8") as fh:
        json.dump([int(value) for value in assignments], fh, separators=(",", ":"))
    exact_matches = _split_exact_overlap(records, roles)
    role_clusters = {
        role: {int(assignments[index]) for index in roles[role]}
        for role in ("train", "val", "test")
    }
    cluster_disjoint = not (
        role_clusters["train"] & role_clusters["val"]
        or role_clusters["train"] & role_clusters["test"]
        or role_clusters["val"] & role_clusters["test"]
    )
    if exact_matches or not cluster_disjoint:
        raise FamilyAssignmentError("split package has exact or family overlap")
    leakage_path = root / "leakage_report.json"
    leakage = {
        "artifact_kind": "p0_data_reconstruction_split_audit",
        "dataset_id": dataset_id,
        "method": "exact_sequence_and_family_assignment_audit",
        "split": {"cluster_disjoint": True},
        "summary": {
            "n_records": len(records),
            "n_train": len(roles["train"]),
            "n_val": len(roles["val"]),
            "n_test": len(roles["test"]),
            "n_excluded": len(excluded or ()),
            "exact_match_count": 0,
            "near_neighbor_threshold_passed": False,
            "near_neighbor_exhaustive": False,
        },
        "scientific_warning": (
            "Exact RNA, exact translated-protein, gene-symbol family, and cluster "
            "disjointness are audited. Exhaustive approximate cross-role near-neighbour "
            "search is not completed by this artifact."
        ),
    }
    with open(leakage_path, "x", encoding="utf-8") as fh:
        json.dump(leakage, fh, indent=2, sort_keys=True)
    excluded_path = None
    if excluded is not None:
        excluded_path = _write_indices(root / "excluded.idx", excluded)
    blockers = [
        "exhaustive_cross_role_near_neighbor_audit_pending",
        "gene_symbol_alias_mapping_not_independently_audited",
    ]
    blockers = list(dict.fromkeys(blockers + list(additional_block_reasons)))
    manifest = build_split_manifest(
        dataset_id=dataset_id,
        records_path=str(Path(records_path).resolve()),
        role_idx_paths=role_paths,
        excluded_idx_path=excluded_path,
        excluded_reason=excluded_reason,
        leakage_report_path=str(leakage_path),
        algorithm=algorithm,
        seed=seed,
        family_threshold=1.0,
        family_disjoint=True,
        exact_cross_role_matches=0,
        near_neighbor_threshold_passed=False,
        cluster_assignment_path=str(cluster_path),
        paper_eligible=False,
        block_reasons=blockers,
    )
    manifest_path = root / "split_manifest.json"
    with open(manifest_path, "x", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    contract = load_and_verify_split_manifest(str(manifest_path))
    return {
        "dataset_id": dataset_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": contract_sha256_file(manifest_path),
        "records_count": contract.records_count,
        "role_counts": {role: contract.roles[role].count for role in ("train", "val", "test")},
        "excluded_count": contract.excluded.count if contract.excluded is not None else 0,
        "paper_eligible": contract.paper_eligible,
        "block_reasons": list(contract.block_reasons),
    }


def build_combined_reconstruction(
    *,
    gencode_manifest_path: str | Path,
    refseq_manifest_path: str | Path,
    output_dir: str | Path,
    split_root: str | Path,
    seed: int = 20260714,
) -> dict[str, object]:
    """Combine source views, freeze families, and emit four verified split contracts."""
    gencode_manifest = verify_source_bundle(gencode_manifest_path)
    refseq_manifest = verify_source_bundle(refseq_manifest_path)
    gencode_records, gencode_meta = _derived_rows_from_bundle(gencode_manifest)
    refseq_records, refseq_meta = _derived_rows_from_bundle(refseq_manifest)
    source_blockers = [
        reason
        for manifest in (gencode_manifest, refseq_manifest)
        for reason in manifest.get("block_reasons", [])
        if reason != "cross_source_family_split_not_yet_bound"
    ]
    combined_records = gencode_records + refseq_records
    combined_meta = gencode_meta + refseq_meta
    _validate_alignment(combined_records, combined_meta)
    assignments, family_evidence = build_family_assignments(combined_records, combined_meta)

    root = Path(output_dir).resolve()
    if root.exists():
        raise ReconstructionError(f"refusing to overwrite combined reconstruction: {root}")
    root.mkdir(parents=True)
    combined_records_path = root / "combined_model_view.records.jsonl"
    combined_metadata_path = root / "combined_model_view.metadata.jsonl"
    assignment_path = root / "family_assignments.json"
    family_path = root / "family_evidence.jsonl"
    write_records_jsonl(combined_records, str(combined_records_path))
    write_jsonl(combined_meta, combined_metadata_path)
    with open(assignment_path, "x", encoding="utf-8") as fh:
        json.dump(assignments, fh, separators=(",", ":"))
    write_jsonl(family_evidence, family_path)

    g_count = len(gencode_records)
    source_specs = {
        "gencode_family": list(range(g_count)),
        "refseq_family": list(range(g_count, len(combined_records))),
    }
    views: dict[str, dict[str, object]] = {}
    split_root_path = Path(split_root).resolve()
    for name, global_indices in source_specs.items():
        source_records = [combined_records[index] for index in global_indices]
        source_meta = [combined_meta[index] for index in global_indices]
        source_assignments = [assignments[index] for index in global_indices]
        records_path = root / f"{name}.records.jsonl"
        metadata_path = root / f"{name}.metadata.jsonl"
        write_records_jsonl(source_records, str(records_path))
        write_jsonl(source_meta, metadata_path)
        roles = assign_family_roles(source_assignments, seed=seed)
        views[name] = _write_split_package(
            dataset_id=f"p0_data_reconstruction_v1_{name}",
            records_path=records_path,
            records=source_records,
            assignments=source_assignments,
            roles=roles,
            output_dir=split_root_path / name,
            seed=seed,
            algorithm="exact_gene_rna_protein_family_union_v1",
            additional_block_reasons=source_blockers,
        )

    combined_roles = assign_family_roles(assignments, seed=seed)
    views["combined_family"] = _write_split_package(
        dataset_id="p0_data_reconstruction_v1_combined_family",
        records_path=combined_records_path,
        records=combined_records,
        assignments=assignments,
        roles=combined_roles,
        output_dir=split_root_path / "combined_family",
        seed=seed,
        algorithm="cross_source_exact_gene_rna_protein_family_union_v1",
        additional_block_reasons=source_blockers,
    )
    cross_roles, excluded = build_cross_source_roles(assignments, combined_meta, seed=seed)
    views["gencode_to_refseq"] = _write_split_package(
        dataset_id="p0_data_reconstruction_v1_gencode_to_refseq",
        records_path=combined_records_path,
        records=combined_records,
        assignments=assignments,
        roles=cross_roles,
        excluded=excluded,
        excluded_reason="refseq_family_overlaps_gencode_training_family",
        output_dir=split_root_path / "gencode_to_refseq",
        seed=seed,
        algorithm="gencode_train_refseq_holdout_exact_family_union_v1",
        additional_block_reasons=source_blockers,
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "p0_data_reconstruction_combined_bundle",
        "source_manifests": [
            {"path": str(Path(gencode_manifest_path).resolve()), "sha256": sha256_file(gencode_manifest_path)},
            {"path": str(Path(refseq_manifest_path).resolve()), "sha256": sha256_file(refseq_manifest_path)},
        ],
        "combined_records": {
            "path": str(combined_records_path),
            "sha256": sha256_file(combined_records_path),
            "count": len(combined_records),
            "gencode_count": len(gencode_records),
            "refseq_count": len(refseq_records),
        },
        "combined_metadata": {"path": str(combined_metadata_path), "sha256": sha256_file(combined_metadata_path)},
        "families": {
            "assignments_path": str(assignment_path),
            "assignments_sha256": sha256_file(assignment_path),
            "evidence_path": str(family_path),
            "evidence_sha256": sha256_file(family_path),
            "n_families": len(family_evidence),
            "n_cross_source_families": sum(int(row["n_sources"]) > 1 for row in family_evidence),
        },
        "split_manifests": views,
        "paper_eligible": False,
        "block_reasons": list(dict.fromkeys([
            "exhaustive_cross_role_near_neighbor_audit_pending",
            "gene_symbol_alias_mapping_not_independently_audited",
        ] + source_blockers)),
    }
    manifest_path = root / "combined_reconstruction_manifest.json"
    with open(manifest_path, "x", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    payload["manifest_path"] = str(manifest_path)
    payload["manifest_sha256"] = sha256_file(manifest_path)
    return payload


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gencode-source", required=True)
    refseq = parser.add_mutually_exclusive_group(required=True)
    refseq.add_argument("--refseq-source")
    refseq.add_argument("--refseq-source-dir")
    refseq.add_argument("--refseq-url")
    parser.add_argument("--frozen-root", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--gencode-sha256", default=None)
    parser.add_argument("--refseq-sha256", default=None)
    parser.add_argument("--refseq-size-bytes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-incomplete-refseq", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if not args.refseq_source_dir and not args.allow_incomplete_refseq:
        raise RawIntegrityError(
            "P0 Data Reconstruction requires the complete 15-part RefSeq release via --refseq-source-dir"
        )
    frozen = Path(args.frozen_root).resolve()
    if frozen.exists():
        raise ReconstructionError(f"frozen reconstruction root already exists: {frozen}")
    raw_root = frozen / "raw"
    gencode_raw = raw_root / "gencode.v45.pc_transcripts.fa.gz"
    refseq_raw = raw_root / "refseq"
    gencode_acquisition = acquire_frozen_file(
        gencode_raw,
        source_path=args.gencode_source,
        expected_sha256=args.gencode_sha256,
    )
    if args.refseq_source_dir:
        source_dir = Path(args.refseq_source_dir).resolve()
        catalog_path = source_dir / "release_catalog.json"
        if not catalog_path.is_file():
            raise RawIntegrityError("complete RefSeq source directory requires release_catalog.json")
        with open(catalog_path, "r", encoding="utf-8") as fh:
            release_catalog = json.load(fh)
        catalog_artifacts = {
            Path(str(row["path"])).name: row
            for row in release_catalog.get("artifacts", [])
        }
        source_files = sorted(
            source_dir.glob("human.*.rna.gbff.gz"),
            key=lambda path: int(path.name.split(".")[1]),
        )
        if len(source_files) != 15 or set(catalog_artifacts) != {path.name for path in source_files}:
            raise RawIntegrityError(
                "complete RefSeq release requires the same 15 partitions in files and catalog"
            )
        acquisitions = [
            acquire_frozen_file(
                refseq_raw / path.name,
                source_path=path,
                expected_sha256=str(catalog_artifacts[path.name]["sha256"]),
                expected_size_bytes=int(catalog_artifacts[path.name]["size_bytes"]),
            )
            for path in source_files
        ]
        refseq_acquisition: Mapping[str, object] = {
            "acquisition": "complete_15_partition_local_copy",
            "release_catalog_path": str(catalog_path),
            "release_catalog_sha256": sha256_file(catalog_path),
            "artifacts": acquisitions,
        }
    else:
        single_refseq = refseq_raw / "human.1.rna.gbff.gz"
        refseq_acquisition = acquire_frozen_file(
            single_refseq,
            source_path=args.refseq_source,
            url=args.refseq_url,
            expected_sha256=args.refseq_sha256,
            expected_size_bytes=args.refseq_size_bytes,
        )
    gencode = build_source_bundle(
        source="gencode_v45",
        raw_path=gencode_raw,
        output_dir=frozen / "sources" / "gencode_v45",
        limit=args.limit,
        acquisition_evidence=gencode_acquisition,
    )
    refseq = build_source_bundle(
        source="refseq_human_rna",
        raw_path=refseq_raw,
        output_dir=frozen / "sources" / "refseq_human_rna",
        limit=args.limit,
        acquisition_evidence=refseq_acquisition,
    )
    combined = build_combined_reconstruction(
        gencode_manifest_path=gencode["manifest_path"],
        refseq_manifest_path=refseq["manifest_path"],
        output_dir=frozen / "combined",
        split_root=args.split_root,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "frozen_root": str(frozen),
                "gencode_manifest": gencode["manifest_path"],
                "refseq_manifest": refseq["manifest_path"],
                "combined_manifest": combined["manifest_path"],
                "combined_records": combined["combined_records"],
                "split_manifests": combined["split_manifests"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def write_jsonl(rows: Iterable[Mapping[str, object]], path: str | Path) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
    return str(target)


__all__ = [
    "ReconstructionError",
    "RawIntegrityError",
    "CanonicalRecordError",
    "LineageError",
    "FamilyAssignmentError",
    "CanonicalInput",
    "sha256_file",
    "verify_raw_artifact",
    "canonicalize_records",
    "derive_model_view",
    "build_family_assignments",
    "build_cross_source_roles",
    "parse_gencode_inputs",
    "parse_refseq_inputs",
    "acquire_frozen_file",
    "download_http_ranges",
    "download_refseq_release",
    "build_source_bundle",
    "verify_source_bundle",
    "assign_family_roles",
    "build_combined_reconstruction",
    "write_jsonl",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
