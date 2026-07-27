#!/usr/bin/env python3
"""Create an observational-only RNA structure feature asset from GENCODE."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import RNA


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()
    source = Path(args.input)
    output = Path(args.output)
    manifest_path = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    emitted = 0
    skipped = 0
    seen = set()
    with source.open() as handle, output.open("w") as out:
        for line in handle:
            if emitted >= args.limit:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            transcript_id = str(record.get("transcript_id", ""))
            utr = str(record.get("five_utr", "")).upper().replace("T", "U")[: args.max_len]
            if not transcript_id or not utr or transcript_id in seen:
                skipped += 1
                continue
            seen.add(transcript_id)
            structure, mfe = RNA.fold(utr)
            paired = sum(ch in "()[]{}<>" for ch in structure) / max(1, len(structure))
            feature = {
                "record_id": f"structure:gencode_v45:{transcript_id}",
                "source_id": f"gencode_v45:{transcript_id}",
                "transcript_id": transcript_id,
                "species": record.get("species", "human"),
                "region": "five_utr",
                "sequence_length": len(utr),
                "sequence_sha256": hashlib.sha256(utr.encode()).hexdigest(),
                "mfe": float(mfe),
                "paired_fraction": float(paired),
                "gc_fraction": (utr.count("G") + utr.count("C")) / max(1, len(utr)),
                "data_layer": "A_observational_pretraining",
                "task_kind": "observational_structure_auxiliary",
                "label_semantics": "representation_or_auxiliary_only; never local_delta_ground_truth",
                "source_asset": str(source),
            }
            out.write(json.dumps(feature, sort_keys=True, separators=(",", ":")) + "\n")
            emitted += 1
    manifest = {
        "schema_version": "nmi_rna_structure_asset_v1",
        "input_path": str(source),
        "input_sha256": sha256(source),
        "output_path": str(output),
        "output_sha256": sha256(output),
        "record_count": emitted,
        "skipped_count": skipped,
        "max_len": args.max_len,
        "engine": "ViennaRNA RNA.fold",
        "label_policy": "observational_auxiliary_only",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
