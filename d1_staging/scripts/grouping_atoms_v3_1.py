#!/usr/bin/env python3
"""Provenance-bound v3.1 grouping-atom projection.

The projection is intentionally conservative. It only emits an atom when the
source record contains an explicit field or an exact sequence/coordinate
derivation documented in ``PROJECTION_POLICY``. Missing atoms remain missing;
there are no sentinel groups and no record-id/row-number fallbacks.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict


GROUPING_ATOM_RULE_SHA256 = "bd8395ab0ec23d98d7c1b717e7fcb0bdd3df6d18002985624cd9eb41f8bd7983"
ASSIGNMENT_ALGORITHM_ID = "D1_PROVENANCE_ATOM_PROJECTION_V1"
GROUP_ID_ALGORITHM_ID = "SHA256_ATOM_VALUE_V1"

PROJECTION_POLICY = {
    "STUDY": "explicit accession",
    "LIBRARY_LINEAGE": "explicit metadata.library plus explicit metadata.source_file; source_file-only is a conservative source-asset lineage",
    "GENE": "first non-empty explicit field in metadata.gene, metadata.gene_symbol, metadata.genes",
    "TRANSCRIPT": "first non-empty explicit field in metadata.transcript_id, metadata.transcript_accession, metadata.transcript, metadata.enst, metadata.transcripts",
    "SEQUENCE_CLUSTER": "exact SHA256 of the normalized source/candidate sequence; no near-cluster inference",
    "TILE_FAMILY": "explicit tile field, or explicit genomic oligo/UTR interval coordinates",
    "BIOLOGICAL_PARENT": "explicit metadata.biological_parent_id, metadata.biological_parent, metadata.parent_id, or metadata.wt_id only",
    "SOURCE": "source_sequence_id for PAIR only",
    "PAIR": "pair_id for PAIR only",
    "CONTEXT": "explicit emitted context_id for OBSERVATION only",
}


def _clean(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in {"nan", "none", "null"}:
            return None
        return value
    if isinstance(value, (list, dict, int, float, bool)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    value = str(value).strip()
    return value or None


def _first_explicit(rec: dict, metadata: dict, keys):
    for key in keys:
        value = rec.get(key)
        if value is None:
            value = metadata.get(key)
        value = _clean(value)
        if value is not None:
            return key, value
    return None, None


def _put(out: OrderedDict, atom: str, value: str | None):
    value = _clean(value)
    if value is None:
        return
    out.setdefault(atom, [])
    if value not in out[atom]:
        out[atom].append(value)


def _coord_value(metadata: dict, keys, prefix: str):
    vals = []
    for key in keys:
        value = _clean(metadata.get(key))
        if value is None:
            return None
        vals.append((key, value))
    return prefix + json.dumps(dict(vals), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalized_sequence_sha256(sequence: str | None):
    sequence = _clean(sequence)
    if sequence is None:
        return None
    return hashlib.sha256(sequence.upper().encode("utf-8")).hexdigest()


def derive_grouping_atoms(
    record: dict,
    source_sequence_id: str | None,
    candidate_sequence_id: str | None,
    source_sequence: str | None,
    candidate_sequence: str | None,
    object_type: str,
    object_id: str,
    context_id: str | None = None,
) -> OrderedDict:
    """Return ``atom -> list of provenance-bound values`` for one object."""
    metadata = record.get("metadata") or {}
    accession = _clean(record.get("accession"))
    out: OrderedDict[str, list[str]] = OrderedDict()

    if accession is not None:
        _put(out, "STUDY", f"accession:{accession.lower()}")

    if object_type == "PAIR":
        _put(out, "PAIR", object_id)
        _put(out, "SOURCE", source_sequence_id)
        for seq in (source_sequence, candidate_sequence):
            digest = normalized_sequence_sha256(seq)
            if digest is not None:
                _put(out, "SEQUENCE_CLUSTER", f"normalized_sequence_sha256:{digest}")
        parent_key, parent_value = _first_explicit(
            record, metadata,
            ("biological_parent_id", "biological_parent", "parent_id", "wt_id"),
        )
        if parent_value is not None:
            _put(out, "BIOLOGICAL_PARENT", f"{parent_key}:{parent_value}")
    elif object_type == "OBSERVATION":
        digest = normalized_sequence_sha256(candidate_sequence or source_sequence)
        if digest is not None:
            _put(out, "SEQUENCE_CLUSTER", f"normalized_sequence_sha256:{digest}")
        _put(out, "CONTEXT", context_id)

    field, value = _first_explicit(
        record, metadata, ("gene", "gene_symbol", "genes"),
    )
    if value is not None:
        _put(out, "GENE", f"{field}:{value}")

    field, value = _first_explicit(
        record, metadata,
        ("transcript_id", "transcript_accession", "transcript", "enst", "transcripts"),
    )
    if value is not None:
        _put(out, "TRANSCRIPT", f"{field}:{value}")

    tile_key, tile_value = _first_explicit(
        record, metadata, ("tile_family", "tile_id", "tile"),
    )
    if tile_value is not None:
        _put(out, "TILE_FAMILY", f"{tile_key}:{tile_value}")
    else:
        # These are source-native coordinate systems, not coordinates inferred
        # from a record id. They are only used when every required field is
        # present in the source metadata.
        tile_value = _coord_value(metadata, ("chrom", "oligo_starts", "oligo_ends"), "oligo:")
        if tile_value is None:
            tile_value = _coord_value(metadata, ("chrom", "pos_start", "pos_end"), "utr_interval:")
        if tile_value is not None:
            _put(out, "TILE_FAMILY", tile_value)

    library = _clean(metadata.get("library"))
    source_file = _clean(metadata.get("source_file"))
    if library is not None or source_file is not None:
        basis = {
            "accession": accession.lower() if accession else None,
            "library": library,
            "source_file": source_file,
        }
        _put(out, "LIBRARY_LINEAGE", json.dumps(basis, ensure_ascii=False, sort_keys=True, separators=(",", ":")))

    return out


def group_id_for(atom: str, value: str) -> str:
    digest = hashlib.sha256(f"{atom}|{value}".encode("utf-8")).hexdigest()
    return f"grp_{atom.lower()}_{digest}"


def group_sha256(group_id: str, atom: str, member_ids: list[str]) -> str:
    payload = {
        "group_id": group_id,
        "grouping_atom": atom,
        "member_ids": sorted(set(member_ids)),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
