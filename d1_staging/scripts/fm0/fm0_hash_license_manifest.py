#!/usr/bin/env python
"""FM0-01: Hash + license manifest for the UTR-LM checkpoint.

Computes SHA-256 of every file in the HF snapshot directory and extracts the
license text. Writes a JSON manifest to data/fm0/hash_license_manifest.json.

This pins the EXACT bytes of the foundation checkpoint we use, so downstream
tasks can verify nothing changed (contract §1.7: foundation_strategy reuse_first).

Acceptance (FM0-01): hash/license manifest.

Usage:
    python scripts/fm0/fm0_hash_license_manifest.py \
        [--output data/fm0/hash_license_manifest.json]

Contract: utr_editflow_contract_v2 (FROZEN)
Task: FM0-01
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make fm0_common importable
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fm0_common import (  # noqa: E402
    CONFIG_PATH,
    DEFAULT_OUTPUT_DIR,
    ensure_offline_env,
    get_model_id,
    get_snapshot_dir,
    load_config,
    sha256_of_file,
    write_json,
)


EXPECTED_FILES = [
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "tokenizer_config.json",
    "vocab.txt",
    "license.md",
    "license-faq.md",
    "README.md",
]


def build_manifest() -> dict:
    cfg = load_config()
    snap = get_snapshot_dir()
    model_id = get_model_id()

    files = []
    for p in sorted(snap.iterdir()):
        if not p.is_file():
            continue
        st = p.stat()
        files.append({
            "filename": p.name,
            "size_bytes": st.st_size,
            "sha256": sha256_of_file(p),
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ"),
        })

    # Index by filename for convenience
    by_name = {f["filename"]: f for f in files}
    missing = [name for name in EXPECTED_FILES if name not in by_name]

    # License extraction (full text of license.md, truncated head of FAQ)
    license_text = ""
    license_faq_head = ""
    if "license.md" in by_name:
        with open(snap / "license.md", "r", encoding="utf-8", errors="replace") as f:
            license_text = f.read()
    if "license-faq.md" in by_name:
        with open(snap / "license-faq.md", "r", encoding="utf-8", errors="replace") as f:
            license_faq_head = f.read()[:4096]

    manifest = {
        "task_id": "FM0-01",
        "contract": "utr_editflow_contract_v2",
        "manifest_kind": "foundation_checkpoint_hash_license",
        "generated_at_utc": datetime.now(timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_id": model_id,
        "revision": cfg["model"]["revision"],
        "snapshot_dir": str(snap),
        "config_path": str(CONFIG_PATH),
        "expected_files": EXPECTED_FILES,
        "missing_files": missing,
        "files": files,
        "license": {
            "type": cfg["license"]["type"],
            "license_md_sha256": by_name.get("license.md", {}).get("sha256"),
            "license_md_size": by_name.get("license.md", {}).get("size_bytes"),
            "license_text": license_text,
            "license_faq_head": license_faq_head,
        },
        "architecture": {
            "model_type": cfg["model"]["model_type"],
            "architecture": cfg["model"]["architecture"],
            "num_hidden_layers": cfg["model"]["num_hidden_layers"],
            "hidden_size": cfg["model"]["hidden_size"],
            "num_attention_heads": cfg["model"]["num_attention_heads"],
            "intermediate_size": cfg["model"]["intermediate_size"],
            "vocab_size": cfg["model"]["vocab_size"],
            "max_position_embeddings": cfg["model"]["max_position_embeddings"],
        },
    }
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "hash_license_manifest.json"),
        help="Output JSON manifest path.",
    )
    args = ap.parse_args()

    ensure_offline_env()
    manifest = build_manifest()

    out = Path(args.output)
    write_json(out, manifest)

    # Console summary
    print(f"[FM0-01] Hash/license manifest -> {out}")
    print(f"  model_id: {manifest['model_id']}")
    print(f"  revision: {manifest['revision']}")
    print(f"  files: {len(manifest['files'])}")
    print(f"  missing: {manifest['missing_files']}")
    print(f"  license: {manifest['license']['type']}")
    for f in manifest["files"]:
        print(f"    {f['filename']:30s} {f['size_bytes']:>10d}  sha256={f['sha256'][:16]}...")

    if manifest["missing_files"]:
        print(
            "[FM0-01] WARNING: missing expected files: "
            + ", ".join(manifest["missing_files"]),
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
