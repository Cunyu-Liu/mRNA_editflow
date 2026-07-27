#!/usr/bin/env python3
"""Derive species 5'UTR observational records from GENCODE FASTA headers."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Iterator, Tuple

UTR5_RE = re.compile(r"(?:^|\|)UTR5:(\d+)-(\d+)(?:\||$)")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_sequence(value: str) -> str:
    return value.strip().upper().replace("T", "U")


def records(path: Path, *, species: str) -> Iterator[Tuple[str, dict]]:
    header = None
    chunks = []
    with gzip.open(path, "rt") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield from one_record(header, "".join(chunks), species)
                header, chunks = line[1:], []
            else:
                chunks.append(line)
    if header is not None:
        yield from one_record(header, "".join(chunks), species)


def one_record(header: str, sequence: str, species: str) -> Iterator[Tuple[str, dict]]:
    match = UTR5_RE.search(header)
    if match is None:
        return
    start, end = int(match.group(1)), int(match.group(2))
    sequence = normalize_sequence(sequence)
    utr = sequence[start - 1:end]
    if len(utr) != end - start + 1 or not utr:
        return
    transcript_id = header.split("|", 1)[0]
    fields = header.split("|")
    record = {
        "record_id": f"gencode_{species}_vM36:{transcript_id}:5utr",
        "transcript_id": transcript_id,
        "species": species,
        "source": "GENCODE vM36",
        "source_header": header,
        "sequence": utr,
        "length": len(utr),
        "data_layer": "A_observational_pretraining",
        "label_semantics": "representation/observational_only",
        "label_policy": "never_local_delta_ground_truth",
        "gene_id": fields[1] if len(fields) > 1 else None,
        "gene_name": fields[4] if len(fields) > 4 else None,
    }
    yield record["record_id"], record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--species", default="mouse")
    args = ap.parse_args()
    input_path, output_path, manifest_path = map(Path, (args.input, args.output, args.manifest))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w") as out:
        for _, record in records(input_path, species=args.species):
            out.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    manifest = {
        "schema_version": "nmi_species_utr_asset_v1",
        "species": args.species,
        "source": "GENCODE vM36 protein-coding transcript FASTA",
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "record_count": count,
        "label_policy": "observational_auxiliary_only",
        "coordinate_policy": "FASTA header UTR5 1-based inclusive coordinates",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
