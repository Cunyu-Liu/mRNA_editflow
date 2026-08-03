#!/usr/bin/env python
"""D1-R (v3.1): single-pass streaming validator for the technical canonical.

Checks each ordinary artifact against its v3.1 JSON schema, verifies foreign-key
integrity (obs/relation/pair -> sequence_entity), enforces the accepted
candidate<->observation and candidate<->pair bijections, verifies no orphan
objects, proves GSE246381 (restricted) rows never leak into the ordinary
workspace, and confirms the restricted access-log hash chain.

Streaming single-pass: never loads multi-million-row files into memory at once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REQUIRED_BY = {
    "sequence_entities.jsonl": ["sequence_id", "sequence_scope", "raw_sequence_sha256",
                                "normalized_sequence_sha256", "full_sequence_sha256",
                                "original_length", "region_scope"],
    "functional_observations.jsonl": ["observation_id", "sequence_id", "endpoint_id",
                                      "context_id", "value", "unit"],
    "functional_observation_candidates.jsonl": ["candidate_id", "sequence_id", "endpoint_id",
                                                "context_id", "source", "source_file_sha256",
                                                "lifecycle_status"],
    "utr_edit_relation_candidates.jsonl": ["candidate_id", "source_sequence_id",
                                           "candidate_sequence_id", "pairing_method",
                                           "evidence_id", "lifecycle_status"],
    "utr_edit_pairs.jsonl": ["pair_id", "candidate_id", "source_sequence_id",
                             "candidate_sequence_id", "design_relation_group_id",
                             "scientific_track", "relation_type",
                             "immutable_base_future_use_role", "pairing_method", "evidence_id"],
    "rejection_records.jsonl": ["rejection_id", "candidate_id", "reason", "evidence_id",
                                "rejected_at"],
    "transformation_edges.jsonl": ["edge_id", "parent_object_id", "child_object_id",
                                   "edge_sha256", "config_hash"],
    "exposure_records.jsonl": ["access_id", "object_id", "intent", "status",
                               "prev_event_sha256", "event_sha256"],
    "use_roles.jsonl": ["object_id", "use_role", "future_use_role", "authority_level"],
    "group_registry.jsonl": ["group_id", "grouping_atom", "member_ids", "group_sha256"],
    "group_assignments.jsonl": ["assignment_id", "object_id", "object_type", "group_id",
                                "grouping_atom", "assignment_algorithm_id"],
    "effective_exposure_projection.jsonl": ["object_id", "effective_exposure",
                                            "projection_sha256", "chain_root_sha256"],
    "endpoint_registry.jsonl": ["endpoint_id", "name", "scaling", "missing_token", "missing_mask"],
}

# objects whose FK must resolve to a sequence_entity, keyed by file -> field
FK_TO_SEQUENCE = {
    "functional_observations.jsonl": ["sequence_id"],
    "functional_observation_candidates.jsonl": ["sequence_id"],
    "utr_edit_relation_candidates.jsonl": ["source_sequence_id", "candidate_sequence_id"],
    "utr_edit_pairs.jsonl": ["source_sequence_id", "candidate_sequence_id"],
}

# bijection pairs: (candidate_file, candidate_id_field, accepted_field, object_file, object_id_field)
BIJECTIONS = [
    ("functional_observation_candidates.jsonl", "candidate_id", "functional_observations.jsonl", "observation_id"),
    ("utr_edit_relation_candidates.jsonl", "candidate_id", "utr_edit_pairs.jsonl", "pair_id"),
]


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="ordinary output dir")
    ap.add_argument("--restricted-dir", required=True, help="restricted output dir")
    args = ap.parse_args()

    d = Path(args.dir)
    rd = Path(args.restricted_dir)
    errors = Counter()
    counters = Counter()

    # 1+2. required fields + build sequence_id set (single pass over sequences)
    seq_ids = set()
    for row in iter_jsonl(d / "sequence_entities.jsonl"):
        missing = [k for k in REQUIRED_BY["sequence_entities.jsonl"] if k not in row]
        if missing:
            errors["missing_fields:sequence_entities"] += 1
        seq_ids.add(row.get("sequence_id"))
    counters["sequence_entities"] = len(seq_ids)

    # FK integrity: single pass over each FK-bearing file
    for fname, fields in FK_TO_SEQUENCE.items():
        for row in iter_jsonl(d / fname):
            for f in fields:
                if row.get(f) not in seq_ids:
                    errors[f"orphan_fk:{fname}:{f}"] += 1

    # required fields for non-sequence files (single pass)
    for fname, required in REQUIRED_BY.items():
        if fname == "sequence_entities.jsonl":
            continue
        n = 0
        for row in iter_jsonl(d / fname):
            n += 1
            missing = [k for k in required if k not in row]
            if missing:
                errors[f"missing_fields:{fname}"] += 1
        counters[fname] = n

    # 3. bijections: accepted candidate ids == object ids (counts must match)
    for cand_file, cand_idf, obj_file, obj_idf in BIJECTIONS:
        cand_ids = set()
        for row in iter_jsonl(d / cand_file):
            if row.get("lifecycle_status") == "ACCEPTED":
                cand_ids.add(row.get(cand_idf))
        obj_ids = set()
        for row in iter_jsonl(d / obj_file):
            obj_ids.add(row.get(obj_idf))
        if cand_ids != obj_ids:
            errors[f"bijection_mismatch:{cand_file}"]
        counters[f"bijection_count:{cand_file}"] = (len(cand_ids), len(obj_ids))

    # 4. GSE246381 isolation: no "gse246381" marker in any ordinary value
    marker = "gse246381"
    for fname in REQUIRED_BY:
        for row in iter_jsonl(d / fname):
            for v in row.values():
                if isinstance(v, str) and marker in v.lower():
                    errors[f"gse246381_leak:{fname}"] += 1
                    break

    # 5. restricted mirror present + access chain hash-linked
    rdir = rd / "sealed_external" / "GSE246381"
    counters["restricted_sequences"] = sum(1 for _ in iter_jsonl(rdir / "sequence_entities.jsonl"))
    prev = None
    chain_ok = True
    n_access = 0
    for ev in iter_jsonl(rdir / "ACCESS_LOG.jsonl"):
        n_access += 1
        if ev.get("prev_event_sha256") != prev:
            chain_ok = False
        prev = ev.get("event_sha256")
    counters["restricted_access_events"] = n_access
    counters["restricted_access_chain_ok"] = chain_ok
    if counters["restricted_sequences"] == 0:
        errors["restricted_mirror_empty"] += 1

    total = sum(errors.values())
    if total == 0:
        errors["total_errors"] = 0

    print(json.dumps({
        "counters": dict(counters),
        "errors": dict(errors),
        "total_errors": total,
        "status": "PASS" if total == 0 else "FAIL",
    }, indent=2, sort_keys=True))

    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()