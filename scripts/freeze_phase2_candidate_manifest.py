#!/usr/bin/env python3
"""Create a verifiable pre-unblinding candidate freeze manifest.

The command intentionally requires both a selection artifact and an explicit
pre-unblinding attestation. It records hashes and the selected candidate set;
it does not decide whether the attestation is scientifically truthful.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mrna_editflow.data.nmi_benchmark_v2 import manifest_sha256
from mrna_editflow.train.train_paired_delta import file_sha256
from scripts.evaluate_phase2_oracle import EXPECTED_ROLES, candidate_digest, load_final_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", default="data/nmi_benchmark_v2")
    parser.add_argument("--role", choices=["test_id", "test_ood"], required=True)
    parser.add_argument("--alias", choices=["test_v2_untouched", "independent_assay"], required=True)
    parser.add_argument("--selection-artifact", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--attest-before-unblinding", action="store_true")
    args = parser.parse_args()
    if EXPECTED_ROLES[args.alias] != args.role:
        raise SystemExit("role/alias combination is not registered for Phase 2")
    if not args.attest_before_unblinding:
        raise SystemExit("refusing to create freeze manifest without pre-unblinding attestation")
    selection_path = Path(args.selection_artifact)
    out = Path(args.out)
    if not selection_path.exists():
        raise SystemExit(f"selection artifact does not exist: {selection_path}")
    if out.exists():
        raise SystemExit(f"refusing to overwrite freeze manifest: {out}")
    root = Path(args.benchmark_root)
    role_manifest_path = root / "manifests" / f"{args.role}.json"
    role_manifest = json.loads(role_manifest_path.read_text())
    records_path = root / str(role_manifest["records_path"])
    rows = load_final_rows(root, args.role, args.alias)
    manifest = {
        "schema_version": "phase2_candidate_freeze_v1",
        "candidate_selection_frozen_before_unblinding": True,
        "role": args.role,
        "alias": args.alias,
        "selection_filter": "GSE246381_mouse_Vglut_MPRA_combined_UMI" if args.alias == "independent_assay" else None,
        "selection_artifact": str(selection_path.resolve()),
        "selection_artifact_sha256": file_sha256(str(selection_path)),
        "role_manifest_sha256": manifest_sha256(role_manifest_path),
        "records_sha256": file_sha256(str(records_path)),
        "candidate_digest": candidate_digest(rows),
        "eligible_local_delta_count": len(rows),
        "claim_policy": "freeze evidence only; final metrics require an independently trained checkpoint and explicit label access",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
