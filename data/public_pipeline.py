"""Public-corpus build and audit pipeline for mRNA-EditFlow.

This module turns verifiable public mRNA resources into the canonical
``MRNARecord`` JSONL used by training. It deliberately keeps the build contract
dependency-light: downloading, parsing, completeness checks, validity checks,
normalisation and manifest writing use only the Python standard library plus the
project's own schema/cleaning code.

Mathematical contract
---------------------
For a raw transcript ``s`` with a GENCODE ``CDS:a-b`` header field or a RefSeq
GenBank ``CDS a..b`` feature, parsing converts the one-based closed interval to
Python coordinates ``[a-1,b)`` and emits:

``5UTR = s[:a-1]``, ``CDS = s[a-1:b]``, ``3UTR = s[b:]``.

Cleaning applies a deterministic map ``N(seq)=upper(seq).replace(T,U)`` and
keeps a record only if the CDS satisfies:

``CDS[:3] = AUG``, ``len(CDS) mod 3 = 0``, terminal codon is stop, and there is
no internal stop.

Complexity: parsing and cleaning are ``O(total transcript length)``; manifest
checksums are ``O(total input bytes)``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from mrna_editflow.core.config import DataConfig
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.clean_mrna import clean_corpus
from mrna_editflow.data.download_mrna import (
    MRNA_DATASETS,
    download_dataset,
    records_from_gencode_fasta,
    records_from_refseq_genbank,
    write_records_jsonl,
)


@dataclass(frozen=True)
class PublicCorpusBuildResult:
    """Returned handles for a public-corpus build.

    ``records`` are included for immediate in-process training, while
    ``records_path`` and ``manifest_path`` are the reproducible on-disk
    artifacts used by server runs. Access is ``O(1)`` except for the record list
    itself.
    """

    dataset_name: str
    records_path: str
    manifest_path: str
    n_raw: int
    n_clean: int
    cleaning_stats: Mapping[str, int]
    records: Sequence[MRNARecord]

    def to_dict(self) -> Dict[str, object]:
        """JSON-friendly summary without duplicating full sequences."""
        data = asdict(self)
        data.pop("records", None)
        return data


def _sha256_file(path: str) -> str:
    """Return a streaming SHA256 digest. Complexity: ``O(file bytes)``."""
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _length_stats(values: Sequence[int]) -> Dict[str, float]:
    """Small dependency-free length summary."""
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def _registry_files(dataset_name: str, data_dir: str) -> List[Dict[str, object]]:
    """Report registered files, local existence, size and SHA256 when present."""
    entry = MRNA_DATASETS[dataset_name]
    out: List[Dict[str, object]] = []
    for filename in entry["files"]:
        path = os.path.join(data_dir, filename)
        exists = os.path.exists(path)
        out.append(
            {
                "filename": filename,
                "path": path,
                "exists": exists,
                "size_bytes": os.path.getsize(path) if exists else 0,
                "sha256": _sha256_file(path) if exists else None,
                "public_url": entry["url"] + filename,
                "expected_sha256": entry.get("checksums", {}).get(filename),
            }
        )
    return out


def _preprocessing_contract(cfg: DataConfig) -> Dict[str, object]:
    """Document the exact checks and feature transformations used downstream."""
    return {
        "completeness_check": [
            "drop rows without a CDS interval in the public FASTA header or GenBank CDS feature",
            "drop rows whose CDS coordinates exceed transcript length",
            "require non-empty transcript_id and region strings after parsing",
        ],
        "validity_check": [
            "normalise DNA/RNA alphabet by uppercasing and mapping T to U",
            "reject any non-ACGU character after normalisation",
            "reject CDS without AUG start, terminal stop, frame length, or with internal stop",
            f"drop CDS longer than max_cds={cfg.max_cds}; truncate UTR caps only",
        ],
        "standardisation": {
            "sequence_map": "N(seq)=upper(strip_whitespace(seq)).replace('T','U')",
            "five_utr_cap": cfg.max_5utr,
            "cds_cap": cfg.max_cds,
            "three_utr_cap": cfg.max_3utr,
            "split_seed": cfg.seed,
        },
        "feature_engineering": [
            "token_ids: A/C/G/U -> 0/1/2/3 with PAD sentinel for batching",
            "region_ids: 5UTR/CDS/3UTR -> 0/1/2 aligned per nucleotide",
            "phase_ids: CDS codon phase i mod 3, PHASE_NONE outside CDS",
            "structural features: MFE proxy or ViennaRNA/RNAfold, start accessibility, pairing propensity",
            "length buckets for low-padding batches and family-disjoint split ids for leakage control",
        ],
        "math_and_complexity": {
            "parse": "O(total FASTA bytes)",
            "clean": "O(sum transcript lengths)",
            "feature_precompute": "fallback MFE O(min(L,cap)^2), pairing propensity O(L*window)",
        },
    }


def _records_summary(records: Sequence[MRNARecord]) -> Dict[str, object]:
    """Return corpus size and region-length distributions."""
    return {
        "n_records": len(records),
        "total_nt": int(sum(len(r.seq) for r in records)),
        "five_utr_len": _length_stats([len(r.five_utr) for r in records]),
        "cds_len": _length_stats([len(r.cds) for r in records]),
        "three_utr_len": _length_stats([len(r.three_utr) for r in records]),
        "species_counts": _counts([r.species for r in records]),
    }


def _counts(values: Sequence[str]) -> Dict[str, int]:
    """Frequency table with deterministic key order in the JSON manifest."""
    out: Dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return dict(sorted(out.items()))


def write_data_manifest(
    manifest_path: str,
    dataset_name: str,
    data_dir: str,
    records_path: str,
    raw_records: Sequence[MRNARecord],
    clean_records: Sequence[MRNARecord],
    cleaning_stats: Mapping[str, int],
    cfg: DataConfig,
) -> str:
    """Write a reproducibility manifest for one public-corpus build.

    The manifest is intentionally self-contained: a reviewer can verify public
    URLs, file hashes, attrition counts, normalisation rules and downstream
    feature definitions without reading code. Complexity is dominated by input
    file hashing, ``O(total input bytes)``.
    """
    entry = MRNA_DATASETS[dataset_name]
    payload: Dict[str, object] = {
        "schema_version": 1,
        "dataset": {
            "name": dataset_name,
            "description": entry["description"],
            "license": entry["license"],
            "registry_url": entry["url"],
            "files": _registry_files(dataset_name, data_dir),
        },
        "records_path": records_path,
        "records_sha256": _sha256_file(records_path) if os.path.exists(records_path) else None,
        "raw_summary": _records_summary(raw_records),
        "clean_summary": _records_summary(clean_records),
        "cleaning_drop_counts": dict(cleaning_stats),
        "preprocessing_contract": _preprocessing_contract(cfg),
    }
    target = Path(manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return str(target)


def build_public_corpus(
    dataset_name: str = "gencode_human_transcripts",
    data_dir: str = "./data/raw",
    out_dir: str = "./data/processed",
    cfg: Optional[DataConfig] = None,
    download: bool = False,
    force: bool = False,
    limit: Optional[int] = None,
) -> PublicCorpusBuildResult:
    """Build cleaned public mRNA records and a full provenance manifest.

    Supported public corpora:

    * ``gencode_human_transcripts``: transcript FASTA with ``CDS:start-end``
      header fields.
    * ``refseq_human_rna``: RefSeq RNA GenBank flat file with conservative
      plus-strand contiguous CDS feature parsing.
    """
    if cfg is None:
        cfg = DataConfig()
    if dataset_name not in MRNA_DATASETS:
        raise ValueError(f"unknown dataset {dataset_name!r}; available: {sorted(MRNA_DATASETS)}")
    data_path = Path(data_dir)
    out_path = Path(out_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)

    if download:
        download_dataset(dataset_name, str(data_path), force=force)

    filename = MRNA_DATASETS[dataset_name]["files"][0]
    raw_path = data_path / filename
    if not raw_path.exists():
        public_url = MRNA_DATASETS[dataset_name]["url"] + filename
        raise FileNotFoundError(
            f"missing {raw_path}; download from {public_url} or rerun with download=True"
        )

    if dataset_name == "gencode_human_transcripts":
        raw_records = records_from_gencode_fasta(str(raw_path), species="human", limit=limit)
    elif dataset_name == "refseq_human_rna":
        raw_records = records_from_refseq_genbank(str(raw_path), species="human", limit=limit)
    else:
        raise ValueError(f"no parser registered for dataset {dataset_name!r}")
    clean_records, cleaning_stats = clean_corpus(raw_records, cfg)
    records_path = out_path / f"{dataset_name}.records.jsonl"
    manifest_path = out_path / f"{dataset_name}.data_manifest.json"
    write_records_jsonl(clean_records, str(records_path))
    write_data_manifest(
        str(manifest_path),
        dataset_name=dataset_name,
        data_dir=str(data_path),
        records_path=str(records_path),
        raw_records=raw_records,
        clean_records=clean_records,
        cleaning_stats=cleaning_stats,
        cfg=cfg,
    )
    return PublicCorpusBuildResult(
        dataset_name=dataset_name,
        records_path=str(records_path),
        manifest_path=str(manifest_path),
        n_raw=len(raw_records),
        n_clean=len(clean_records),
        cleaning_stats=cleaning_stats,
        records=clean_records,
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build public mRNA corpus JSONL + manifest")
    parser.add_argument("--dataset", default="gencode_human_transcripts")
    parser.add_argument("--data-dir", default="./data/raw")
    parser.add_argument("--out-dir", default="./data/processed")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    cfg = DataConfig()
    if args.seed is not None:
        cfg.seed = int(args.seed)
    result = build_public_corpus(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        cfg=cfg,
        download=bool(args.download),
        force=bool(args.force),
        limit=args.limit,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())


__all__ = [
    "PublicCorpusBuildResult",
    "build_public_corpus",
    "write_data_manifest",
    "main",
]
