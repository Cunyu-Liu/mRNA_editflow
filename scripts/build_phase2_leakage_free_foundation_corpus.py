#!/usr/bin/env python3
"""Build a Stage-A corpus excluding exact final-role sequence overlaps.

The resulting corpus is only a prospective pretraining input.  It does not
retroactively make an already-trained checkpoint leakage-free; a new Stage-A
checkpoint must be trained from this output and then audited again.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.audit_phase2_foundation_leakage import (
    file_snapshot,
    load_target_sequences,
    normalize_sequence,
    transcript_sequence,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_prefix_index(targets: dict[str, dict], min_length: int) -> tuple[dict[str, list[str]], int]:
    eligible = [sequence for sequence in targets if len(sequence) >= min_length]
    prefixes: dict[str, list[str]] = {}
    for sequence in eligible:
        prefixes.setdefault(sequence[:min_length], []).append(sequence)
    return prefixes, len(targets) - len(eligible)


def transcript_has_target(transcript: str, prefixes: dict[str, list[str]], min_length: int) -> bool:
    if len(transcript) < min_length:
        return False
    seen_prefixes = set()
    for offset in range(len(transcript) - min_length + 1):
        prefix = transcript[offset:offset + min_length]
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        if any(sequence in transcript for sequence in prefixes.get(prefix, [])):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", default="data/nmi_benchmark_v2")
    parser.add_argument("--foundation-corpus", default="data/processed/gencode_human_transcripts.records.jsonl")
    parser.add_argument("--roles", default="test_id,test_family,test_context,test_assay,test_ood")
    parser.add_argument("--min-sequence-length", type=int, default=15)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-manifest", required=True)
    args = parser.parse_args()
    if args.min_sequence_length < 1:
        raise SystemExit("min-sequence-length must be positive")
    foundation_path = Path(args.foundation_corpus)
    out_jsonl = Path(args.out_jsonl)
    out_manifest = Path(args.out_manifest)
    if out_jsonl.exists() or out_manifest.exists():
        raise SystemExit("refusing to overwrite an existing filtered corpus or manifest")
    roles = [role.strip() for role in args.roles.split(",") if role.strip()]
    targets, target_provenance = load_target_sequences(Path(args.benchmark_root), roles)
    prefixes, short_count = build_prefix_index(targets, args.min_sequence_length)
    if not targets or short_count:
        raise SystemExit(
            f"cannot build strict corpus: targets={len(targets)}, short_sequences={short_count}"
        )
    before = file_snapshot(foundation_path)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    kept = excluded = total = 0
    with foundation_path.open() as source, out_jsonl.open("w") as destination:
        for line in source:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            if transcript_has_target(transcript_sequence(row), prefixes, args.min_sequence_length):
                excluded += 1
                continue
            destination.write(line)
            kept += 1
    after = file_snapshot(foundation_path)
    if before != after:
        raise RuntimeError(f"foundation corpus changed during filtering: {before} -> {after}")
    manifest = {
        "schema_version": "phase2_leakage_free_foundation_corpus_v1",
        "status": "prospective_pretraining_input",
        "source_corpus": before,
        "source_corpus_sha256": sha256_file(foundation_path),
        "filtered_corpus": file_snapshot(out_jsonl),
        "filtered_corpus_sha256": sha256_file(out_jsonl),
        "benchmark_target_provenance": target_provenance,
        "roles": roles,
        "min_sequence_length": args.min_sequence_length,
        "total_records": total,
        "kept_records": kept,
        "excluded_records": excluded,
        "exact_overlap_after_filtering": "must be verified by audit_phase2_foundation_leakage.py",
        "checkpoint_status": "no_new_checkpoint_trained",
        "claim_policy": "filtered input is not a scientific foundation checkpoint until retraining and audit complete",
    }
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
