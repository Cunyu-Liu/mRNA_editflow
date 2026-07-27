#!/usr/bin/env python3
"""Audit exact sequence overlap between Stage-A corpus and final Phase 2 roles.

This is deliberately narrower than a semantic or family-level leakage proof:
it detects whether an eligible final source/candidate sequence occurs as an
exact substring of a Stage-A transcript.  Missing inputs, an empty corpus, or
any overlap are fail-closed unless ``--report-only`` is explicitly supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mrna_editflow.data.nmi_benchmark_v2 import load_manifest, manifest_sha256


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_snapshot(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def normalize_sequence(value: object) -> str:
    return str(value or "").upper().replace("T", "U")


def transcript_sequence(row: dict) -> str:
    parts = [row.get("five_utr"), row.get("cds"), row.get("three_utr")]
    joined = "".join(normalize_sequence(part) for part in parts if part)
    if joined:
        return joined
    for key in ("full_sequence", "sequence", "sequence_rna", "mrna"):
        candidate = normalize_sequence(row.get(key))
        if candidate:
            return candidate
    return ""


def load_target_sequences(root: Path, roles: list[str]) -> tuple[dict[str, dict], dict]:
    role_indexes: dict[str, set[str]] = {}
    records_path: Path | None = None
    manifest_digests: dict[str, str] = {}
    for role in roles:
        manifest_path = root / "manifests" / f"{role}.json"
        manifest = load_manifest(manifest_path, allow_final_labels=True)
        current_path = root / str(manifest["records_path"])
        if records_path is None:
            records_path = current_path
        elif current_path != records_path:
            raise RuntimeError("requested roles point to different records stores")
        index_path = root / str(manifest["index_path"])
        role_indexes[role] = {
            line.strip() for line in index_path.read_text().splitlines() if line.strip()
        }
        manifest_digests[role] = manifest_sha256(manifest_path)
    assert records_path is not None
    before = file_snapshot(records_path)
    wanted = set().union(*role_indexes.values())
    targets: dict[str, dict] = {}
    records_digest = hashlib.sha256()
    with records_path.open("rb") as handle:
        for raw_line in handle:
            records_digest.update(raw_line)
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            record_id = str(row.get("record_id"))
            if record_id not in wanted:
                continue
            if not (
                row.get("task_kind") == "local_delta"
                and row.get("data_layer") == "C_source_matched_intervention"
                and bool(row.get("local_delta_eligible"))
            ):
                continue
            for role, index in role_indexes.items():
                if record_id not in index:
                    continue
                for field in ("source_sequence", "candidate_sequence"):
                    sequence = normalize_sequence(row.get(field))
                    if sequence:
                        targets.setdefault(sequence, {
                            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                            "roles": set(), "fields": set(), "record_ids": [],
                        })["roles"].add(role)
                        targets[sequence]["fields"].add(field)
                        if len(targets[sequence]["record_ids"]) < 5:
                            targets[sequence]["record_ids"].append(record_id)
    after = file_snapshot(records_path)
    if before != after:
        raise RuntimeError(f"benchmark records changed during audit: {before} -> {after}")
    for item in targets.values():
        item["roles"] = sorted(item["roles"])
        item["fields"] = sorted(item["fields"])
    return targets, {
        "records_snapshot": after,
        "records_sha256": records_digest.hexdigest(),
        "manifest_sha256": manifest_digests,
        "role_record_ids": {role: len(index) for role, index in role_indexes.items()},
        "eligible_unique_sequence_count": len(targets),
    }


def load_foundation_transcripts(path: Path) -> tuple[list[tuple[str, str]], dict]:
    before = file_snapshot(path)
    transcripts: list[tuple[str, str]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sequence = transcript_sequence(row)
            if sequence:
                transcript_id = str(row.get("transcript_id", f"line:{line_number}"))
                transcripts.append((transcript_id, sequence))
    after = file_snapshot(path)
    if before != after:
        raise RuntimeError(f"foundation corpus changed during audit: {before} -> {after}")
    return transcripts, {
        "foundation_snapshot": after,
        "foundation_sha256": file_sha256(path),
        "foundation_record_count": sum(1 for _ in path.open()),
        "foundation_transcript_count": len(transcripts),
    }


def find_overlaps(targets: dict[str, dict], transcripts: list[tuple[str, str]], min_length: int) -> list[dict]:
    eligible = {sequence: meta for sequence, meta in targets.items() if len(sequence) >= min_length}
    prefixes: dict[str, list[str]] = {}
    prefix_length = min_length
    for sequence in eligible:
        prefixes.setdefault(sequence[:prefix_length], []).append(sequence)
    hits: dict[str, dict] = {}
    for transcript_id, transcript in transcripts:
        seen_prefixes = set()
        for offset in range(max(0, len(transcript) - prefix_length + 1)):
            prefix = transcript[offset:offset + prefix_length]
            if prefix in seen_prefixes:
                continue
            seen_prefixes.add(prefix)
            for sequence in prefixes.get(prefix, []):
                if sequence in transcript:
                    hit = hits.setdefault(sequence, {
                        "sequence_sha256": targets[sequence]["sequence_sha256"],
                        "sequence_length": len(sequence),
                        "roles": targets[sequence]["roles"],
                        "fields": targets[sequence]["fields"],
                        "record_ids": targets[sequence]["record_ids"],
                        "foundation_transcripts": [],
                    })
                    if len(hit["foundation_transcripts"]) < 5:
                        hit["foundation_transcripts"].append(transcript_id)
    return sorted(hits.values(), key=lambda item: item["sequence_sha256"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", default="data/nmi_benchmark_v2")
    parser.add_argument("--foundation-corpus", default="data/processed/gencode_human_transcripts.records.jsonl")
    parser.add_argument("--roles", default="test_id,test_family,test_context,test_assay,test_ood")
    parser.add_argument("--min-sequence-length", type=int, default=15)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    roles = [role.strip() for role in args.roles.split(",") if role.strip()]
    if not roles or args.min_sequence_length < 1:
        raise SystemExit("roles and min-sequence-length must be non-empty/positive")
    root = Path(args.benchmark_root)
    foundation_path = Path(args.foundation_corpus)
    targets, target_provenance = load_target_sequences(root, roles)
    transcripts, foundation_provenance = load_foundation_transcripts(foundation_path)
    overlaps = find_overlaps(targets, transcripts, args.min_sequence_length)
    short_count = sum(1 for sequence in targets if len(sequence) < args.min_sequence_length)
    status = "pass" if targets and transcripts and not overlaps else "blocked"
    report = {
        "schema_version": "phase2_foundation_leakage_audit_v1",
        "status": status,
        "audit_scope": "exact eligible final source/candidate sequence as an exact substring of Stage-A transcript",
        "semantic_family_overlap_proven": False,
        "roles": roles,
        "min_sequence_length": args.min_sequence_length,
        "target_provenance": target_provenance,
        "foundation_provenance": foundation_provenance,
        "eligible_sequence_count": len(targets),
        "short_sequences_not_substring_audited": short_count,
        "exact_overlap_count": len(overlaps),
        "overlaps": overlaps,
        "fail_closed": not args.report_only,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not targets or not transcripts:
        raise SystemExit("foundation leakage audit blocked: missing eligible targets or foundation transcripts")
    if overlaps and not args.report_only:
        raise SystemExit(f"foundation leakage audit failed: {len(overlaps)} exact overlaps")


if __name__ == "__main__":
    main()
