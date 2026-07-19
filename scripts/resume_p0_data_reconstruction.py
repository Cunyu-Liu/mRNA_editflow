"""Resume after raw acquisition without repeating already-proven gzip inflation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mrna_editflow.data.reconstruction import build_combined_reconstruction, build_source_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--staging-catalog", required=True)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    frozen = project / "data/reconstructed/p0_data_reconstruction_v1"
    raw_dir = frozen / "raw/refseq"
    with open(args.staging_catalog, "r", encoding="utf-8") as fh:
        catalog = json.load(fh)
    evidence = []
    for row in catalog["artifacts"]:
        item = dict(row)
        item["path"] = str(raw_dir / Path(str(row["path"])).name)
        evidence.append(item)
    refseq = build_source_bundle(
        source="refseq_human_rna",
        raw_path=raw_dir,
        output_dir=frozen / "sources/refseq_human_rna",
        acquisition_evidence={
            "acquisition": "resume_from_same_process_verified_frozen_raw",
            "release_catalog_path": str(Path(args.staging_catalog).resolve()),
            "artifacts": evidence,
        },
        trust_acquisition_evidence=True,
    )
    combined = build_combined_reconstruction(
        gencode_manifest_path=frozen / "sources/gencode_v45/reconstruction_manifest.json",
        refseq_manifest_path=refseq["manifest_path"],
        output_dir=frozen / "combined",
        split_root=project / "benchmark/dev/p0_data_reconstruction_v1",
        seed=args.seed,
    )
    print(json.dumps({"refseq": refseq["manifest_path"], "combined": combined["manifest_path"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
