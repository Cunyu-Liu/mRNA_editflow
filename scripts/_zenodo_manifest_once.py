"""One-off: verify locally-downloaded Zenodo zips and write manifest."""
from pathlib import Path

from data.download_mrna import _compute_sha256
from data.download_external_catalog import EXTERNAL_CATALOG, write_manifest

EXPECTED = {
    "3UTR.zip": "e69699f9d7dea8745b040012c12689d3560783dbd6fb2665dee10b3297a7f85c",
    "5UTR.zip": "22ffa20a734ff087ba7b1da22509eb01b4959961923bc34a3bbe90f6b9df62a2",
    "CDS.zip": "2c7ce9ac16f1bff0c1849701606b9f9b742827e813a384643b040ce11267f18a",
    "Spliceator.zip": "d97b1642b0350de860b2022f9648d6ccdd2bb7e7a71c0f9970e0e2b6113482ce",
    "full_length.zip": "4946aca5993b36bf6fff1ed6f91f489e6c7c8f45390f759617bdf7241610fccc",
    "protein.zip": "f35145bdc72fb4eeda7bf32f89181073f59555ded8e9e7a4903f9e88aa31de14",
    "te_ultra_full_length.zip": "d81cc5819002c29562b1c937574c1257c9b6f156777ea8cae4c921998f309d2d",
}


def main() -> None:
    root = Path("/mnt/cunyuliu/mrna_editflow_extdata/raw/mrnabert_downstream_zenodo")
    entry = EXTERNAL_CATALOG["mrnabert_downstream_zenodo"]
    records = []
    for f in entry["files"]:
        fp = root / f["filename"]
        sha = _compute_sha256(fp)
        assert sha == EXPECTED[f["filename"]], (f["filename"], sha)
        records.append({
            "filename": f["filename"],
            "url": f["url"],
            "sha256": sha,
            "byte_size": fp.stat().st_size,
        })
        print("VERIFIED", f["filename"], fp.stat().st_size)
    p = write_manifest(root, "mrnabert_downstream_zenodo", entry, records)
    print("MANIFEST", p)


if __name__ == "__main__":
    main()
