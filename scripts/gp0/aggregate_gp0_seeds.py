#!/usr/bin/env python3
"""Aggregate exactly five GP0 formal seed artifacts without lowering gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import CONTRACT_ID, CONTRACT_SHA256, artifact_checksums, sha256_file, write_json  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def aggregate(args: argparse.Namespace) -> int:
    if len(args.run_dir) != 5:
        raise RuntimeError("GP0 formal acceptance requires exactly five seed run directories")
    manifests: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    for path in args.run_dir:
        manifest_path = path / "run_manifest.json"
        evaluation_path = path / "evaluation.json"
        if not manifest_path.exists() or not evaluation_path.exists():
            raise RuntimeError(f"formal seed is missing manifest/evaluation: {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        manifests.append(manifest)
        evaluations.append(evaluation)
    seeds = [int(manifest.get("formal_seed_requirement", {}).get("current_seed", -1)) for manifest in manifests]
    if len(set(seeds)) != 5 or any(seed < 0 for seed in seeds):
        raise RuntimeError("formal GP0 seed manifests do not contain five unique non-negative seeds")
    if any(manifest.get("mode") != "formal" for manifest in manifests):
        raise RuntimeError("development evidence cannot be aggregated as formal GP0")
    if any(manifest.get("goal_contract", {}).get("sha256") != CONTRACT_SHA256 for manifest in manifests):
        raise RuntimeError("formal seed contract hash mismatch")
    binding_fields = (
        ("variant",),
        ("inputs", "data_manifest_sha256"),
        ("inputs", "split_manifest_sha256"),
        ("inputs", "foundation_checkpoint_sha256"),
        ("inputs", "exposure_ledger_version"),
    )
    def get_nested(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
        value: Any = payload
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        return value
    for field in binding_fields:
        values = {json.dumps(get_nested(manifest, field), sort_keys=True) for manifest in manifests}
        if len(values) != 1:
            raise RuntimeError(f"formal seed binding mismatch: {'.'.join(field)}")
    if any(evaluation.get("gates", {}).get("formal_scientific_acceptance") is True for evaluation in evaluations):
        raise RuntimeError("unexpected pre-asserted scientific acceptance in seed evaluation")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    seed_rows = []
    for manifest, evaluation, path in zip(manifests, evaluations, args.run_dir):
        seed_rows.append({
            "seed": int(manifest["formal_seed_requirement"]["current_seed"]),
            "run_id": manifest.get("run_id"),
            "run_dir": str(path),
            "manifest_sha256": sha256_file(path / "run_manifest.json"),
            "evaluation_sha256": sha256_file(path / "evaluation.json"),
            "evaluation_status": evaluation.get("status"),
            "gates": evaluation.get("gates", {}),
        })
    aggregate_payload = {
        "schema_version": "gp0_five_seed_aggregate_v1",
        "status": "FIVE_SEED_AGGREGATED_PENDING_FINALIZER",
        "goal_contract": {"id": CONTRACT_ID, "sha256": CONTRACT_SHA256},
        "required_seed_count": 5,
        "observed_seed_count": 5,
        "seeds": sorted(seed_rows, key=lambda row: row["seed"]),
        "gate_boundary": {
            "scientific_acceptance": False,
            "mode_collapse": "requires frozen contract threshold and finalizer review",
            "paper_eligibility": False,
            "done_marker_written": False,
        },
    }
    write_json(out_dir / "five_seed_aggregate.json", aggregate_payload)
    (out_dir / "artifact_checksums.sha256").write_text(artifact_checksums(out_dir), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return aggregate(_parse_args(argv))
    except Exception as error:
        print(f"FAILED_WITH_EVIDENCE: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
