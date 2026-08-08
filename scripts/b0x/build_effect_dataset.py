"""B0-X effect dataset builder.

Loads the rebuilt D1 staging pairs + sequence entities + functional
observations and joins them with the read-only main-repo canonical records
(which carry the actual sequence text, edit scripts and candidate labels), then
derives the source-relative effect delta per asset per config.DELTA_SPECS.

No data is fabricated.  When a source anchor is genuinely not derivable, the
record is emitted with delta_source_status="source_anchor_unavailable" and a
null delta (excluded from delta-based ranking but present in the manifest).

Usage:
    python -m scripts.b0x.build_effect_dataset --out-dir artifacts/b0x
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from scripts.m4_sparse.ordinary_io import (  # noqa: E402
    assert_ordinary_path,
    assert_ordinary_rows,
)

CAND_SUFFIX = "__cand"
SRC_SUFFIX = "__src"
OBSERVATION_SCHEMA_VERSION = "lossless_functional_observations_v1"


def _study_of(seq_id: str) -> str:
    return seq_id.split("_")[0]


def load_pairs(pairs_path: Path) -> List[dict]:
    pairs_path = Path(pairs_path)
    assert_ordinary_path(pairs_path, purpose="pair input")
    pairs = []
    active = set(config.ACTIVE_STUDIES)
    for line in pairs_path.open(encoding="utf-8"):
        r = json.loads(line)
        assert_ordinary_rows([r], purpose="pair input")
        st = _study_of(r["candidate_sequence_id"])
        if st in active and r.get("scientific_track") == "E":
            pairs.append(r)
    return pairs


def load_canonical(records_path: Path, needed_record_ids: set) -> Dict[str, dict]:
    """Stream canonical records; keep only needed record_ids."""
    records_path = Path(records_path)
    assert_ordinary_path(records_path, purpose="canonical input")
    out: Dict[str, dict] = {}
    for line in records_path.open(encoding="utf-8"):
        r = json.loads(line)
        rid = r.get("record_id")
        if rid in needed_record_ids:
            assert_ordinary_rows([r], purpose="canonical input")
            out[rid] = r
    # Also load reconstructed pairs (rebuilt assets not in the pre-rebuild
    # main-repo canonical records file).
    for study, rp_path in config.RECONSTRUCTED_PAIRS.items():
        if not Path(rp_path).exists():
            continue
        assert_ordinary_path(Path(rp_path), purpose="reconstructed pair input")
        for line in Path(rp_path).open(encoding="utf-8"):
            r = json.loads(line)
            rid = r.get("record_id")
            if rid in needed_record_ids:
                assert_ordinary_rows(
                    [r], purpose="reconstructed pair input"
                )
                out[rid] = r
    return out


class ObservationConflictError(ValueError):
    """Raised when ordinary observation rows cannot be resolved losslessly."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _replicate_token(row: Mapping[str, Any]) -> str:
    value = row.get("replicate")
    if value is None:
        value = row.get("replicate_id")
    return _canonical_json(value)


def _observation_slot(row: Mapping[str, Any]) -> Tuple[str, ...]:
    """Identity at which more than one numeric value is a hard conflict.

    Observation ids are deliberately not part of this key: two rows with
    different ids but the same biological slot must not silently overwrite
    one another. Distinct, explicitly labelled replicates remain distinct.
    """
    return (
        str(row.get("sequence_id") or ""),
        str(row.get("endpoint_id") or ""),
        str(row.get("context_id") or ""),
        str(row.get("assay_context") or ""),
        _replicate_token(row),
        str(row.get("unit") or ""),
    )


def load_candidate_observations(
    obs_path: Path,
    cand_seq_ids: Dict[str, set],
) -> Dict[str, Dict[str, List[dict]]]:
    """Load candidate observations without endpoint-level reduction.

    Rows retain context, replicate, value, unit and provenance fields plus a
    stable row hash. Explicitly labelled replicates are preserved. Distinct
    values in the same biological slot are rejected rather than averaged or
    resolved by input order.
    """
    obs_path = Path(obs_path)
    assert_ordinary_path(obs_path, purpose="functional observation input")
    out: Dict[str, Dict[str, List[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    values_by_slot: Dict[Tuple[str, ...], set] = defaultdict(set)
    rows_by_observation_id: Dict[str, str] = {}
    with obs_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ObservationConflictError(
                    "functional observation row %d is not an object"
                    % line_number
                )
            assert_ordinary_rows(
                [row], purpose="functional observation input"
            )
            sid = row.get("sequence_id")
            if not isinstance(sid, str) or not sid:
                raise ObservationConflictError(
                    "functional observation row %d has no sequence_id"
                    % line_number
                )
            study = _study_of(sid)
            if study not in cand_seq_ids or sid not in cand_seq_ids[study]:
                continue
            endpoint = row.get("endpoint_id")
            if not isinstance(endpoint, str) or not endpoint:
                raise ObservationConflictError(
                    "functional observation row %d has no endpoint_id"
                    % line_number
                )
            try:
                value = float(row.get("value"))
            except (TypeError, ValueError):
                raise ObservationConflictError(
                    "functional observation row %d has a non-numeric value"
                    % line_number
                ) from None
            if not math.isfinite(value):
                raise ObservationConflictError(
                    "functional observation row %d has a non-finite value"
                    % line_number
                )
            preserved = dict(row)
            preserved["value"] = value
            preserved["input_row_sha256"] = _json_sha256(row)
            obs_id = _optional_text(row.get("observation_id"))
            if obs_id:
                previous = rows_by_observation_id.get(obs_id)
                current = preserved["input_row_sha256"]
                if previous is not None and previous != current:
                    raise ObservationConflictError(
                        "observation_id %s maps to conflicting rows" % obs_id
                    )
                rows_by_observation_id[obs_id] = current
            slot = _observation_slot(row)
            values_by_slot[slot].add(value)
            if len(values_by_slot[slot]) > 1:
                raise ObservationConflictError(
                    "multiple values for functional observation slot %s"
                    % (slot,)
                )
            out[sid][endpoint].append(preserved)
    result: Dict[str, Dict[str, List[dict]]] = {}
    for sid in sorted(out):
        result[sid] = {}
        for endpoint in sorted(out[sid]):
            result[sid][endpoint] = sorted(
                out[sid][endpoint], key=_canonical_json
            )
    return result


# Compatibility name retained for callers. Unlike the historical function it
# returns lossless observation lists, never endpoint -> scalar reduction.
load_candidate_values = load_candidate_observations


def _observation_join_key(
    row: Mapping[str, Any],
) -> Tuple[str, str, str, str]:
    return (
        str(row.get("context_id") or ""),
        str(row.get("assay_context") or ""),
        _replicate_token(row),
        str(row.get("unit") or ""),
    )


def _observation_groups(
    index: Mapping[str, Mapping[str, Any]], sid: str, endpoint: str,
) -> Dict[Tuple[str, str, str, str], dict]:
    raw = index.get(sid, {}).get(endpoint, [])
    if isinstance(raw, (int, float)):
        raw = [{
            "sequence_id": sid,
            "endpoint_id": endpoint,
            "value": float(raw),
        }]
    elif isinstance(raw, dict):
        raw = [raw]
    groups: Dict[Tuple[str, str, str, str], List[dict]] = defaultdict(list)
    for row in raw or []:
        groups[_observation_join_key(row)].append(dict(row))
    resolved = {}
    for key in sorted(groups):
        rows = sorted(groups[key], key=_canonical_json)
        values = {float(row["value"]) for row in rows}
        if len(values) != 1:
            raise ObservationConflictError(
                "unresolved multiple values for %s/%s at %s"
                % (sid, endpoint, key)
            )
        resolved[key] = {"value": next(iter(values)), "rows": rows}
    return resolved


def _normalize_sequence(value: Any) -> str:
    return str(value or "").strip().upper().replace("T", "U")


def _source_anchor_id(study: str, anchor_kind: str, raw_identity: Any) -> str:
    payload = {
        "study": study,
        "anchor_kind": anchor_kind,
        "raw_identity": raw_identity,
    }
    return "%s:%s" % (anchor_kind, _json_sha256(payload))


def _resolved_anchor(
    study: str,
    anchor_kind: str,
    raw_identity: Any,
    values: Sequence[float],
    evidence: Sequence[dict],
) -> dict:
    unique = sorted({float(value) for value in values})
    if not unique:
        status = "MISSING_MEASURED_VALUE"
        value = None
    elif len(unique) == 1:
        status = "MEASURED_SINGLE_VALUE"
        value = unique[0]
    else:
        status = "CONFLICTING_MEASURED_VALUES"
        value = None
    return {
        "source_anchor_identity": _source_anchor_id(
            study, anchor_kind, raw_identity
        ),
        "source_anchor_status": status,
        "value": value,
        "measured_values": unique,
        "source_anchor_evidence": sorted(
            [dict(item) for item in evidence], key=_canonical_json
        ),
    }


def _coerce_anchor(
    raw: Any, study: str, anchor_kind: str, raw_identity: Any,
) -> dict:
    if isinstance(raw, dict) and "source_anchor_status" in raw:
        return dict(raw)
    if raw is None:
        return _resolved_anchor(
            study, anchor_kind, raw_identity, [], []
        )
    return _resolved_anchor(
        study,
        anchor_kind,
        raw_identity,
        [float(raw)],
        [{"legacy_scalar_input": float(raw)}],
    )


def load_gse114002_source_anchors() -> Dict[Tuple[str, str], dict]:
    """Lossless WT rl anchor per (library, normalized mother sequence).

    Conflicting measured values are retained as evidence and made unusable;
    they are never averaged into an apparently unique source anchor.
    """
    import pandas as pd
    df = pd.read_csv(config.GSE114002_RAW_LIB, low_memory=False)
    anchor_values: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    evidence: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    raw_file_sha256 = sha256_file(Path(config.GSE114002_RAW_LIB))
    for row_index, (library, seq, rl) in enumerate(zip(
        df["library"].astype(str),
        df["utr"].astype(str),
        pd.to_numeric(df["rl"], errors="coerce"),
    )):
        if pd.notna(rl) and len(seq) > 10:
            key = (str(library).strip(), _normalize_sequence(seq))
            value = float(rl)
            anchor_values[key].append(value)
            evidence[key].append({
                "source_asset": str(config.GSE114002_RAW_LIB),
                "source_asset_sha256": raw_file_sha256,
                "raw_row_index": row_index,
                "library": key[0],
                "raw_sequence_sha256": hashlib.sha256(
                    key[1].encode("ascii")
                ).hexdigest(),
                "value": value,
                "endpoint": "rl",
            })
    return {
        key: _resolved_anchor(
            "GSE114002", "RAW_LIBRARY_RL", key,
            anchor_values[key], evidence[key],
        )
        for key in sorted(anchor_values)
    }


def load_gse200304_source_anchors() -> Dict[str, dict]:
    """WT Freq anchor per merged_id without barcode overwrite."""
    import pandas as pd
    cdf = pd.read_csv(config.GSE200304_COUNT, sep="\t", compression="infer")
    raw_count_sha256 = sha256_file(Path(config.GSE200304_COUNT))
    twist_sha256 = sha256_file(Path(config.GSE200304_TWIST))
    bc_freq: Dict[str, List[float]] = defaultdict(list)
    bc_evidence: Dict[str, List[dict]] = defaultdict(list)
    for row_index, (bc, fq) in enumerate(zip(
        cdf["Barcode"].astype(str),
        pd.to_numeric(cdf["Freq"], errors="coerce"),
    )):
        if pd.notna(fq):
            value = float(fq)
            bc_freq[bc].append(value)
            bc_evidence[bc].append({
                "source_asset": str(config.GSE200304_COUNT),
                "source_asset_sha256": raw_count_sha256,
                "raw_row_index": row_index,
                "barcode": bc,
                "value": value,
                "endpoint": "Freq",
            })
    # Map WT merged_id -> Freq. merged_id format "chr2:69461620_G-C_WT".
    tdf = pd.read_csv(config.GSE200304_TWIST, sep="\t", compression="infer")
    wt_freq: Dict[str, dict] = {}
    for mid in tdf["merged_id"].astype(str):
        if mid.endswith("_WT"):
            relation_evidence = list(bc_evidence.get(mid, [])) + [{
                "relation_source_asset": str(config.GSE200304_TWIST),
                "relation_source_asset_sha256": twist_sha256,
                "merged_id": mid,
                "relation": "WT_BARCODE_IDENTITY",
            }]
            wt_freq[mid] = _resolved_anchor(
                "GSE200304",
                "RAW_COUNT_FREQ",
                mid,
                bc_freq.get(mid, []),
                relation_evidence,
            )
    return wt_freq


def _attach_observation_provenance(
    record: dict, observations: Sequence[dict],
) -> None:
    rows = sorted([dict(row) for row in observations], key=_canonical_json)
    record["functional_observation_schema_version"] = (
        OBSERVATION_SCHEMA_VERSION
    )
    record["functional_observations"] = rows
    record["functional_observation_hashes"] = [
        row.get("input_row_sha256") or _json_sha256(row) for row in rows
    ]
    record["observation_ids"] = sorted({
        str(row["observation_id"]) for row in rows
        if row.get("observation_id") not in (None, "")
    })

    def unique_text(*names: str) -> Optional[str]:
        values = {
            str(row.get(name)).strip()
            for row in rows for name in names
            if row.get(name) not in (None, "")
        }
        return next(iter(values)) if len(values) == 1 else None

    # This is a raw assay arm used only as a crosswalk atom. It is explicitly
    # not a publication context_id.
    record["assay_arm"] = unique_text(
        "assay_arm", "context_id", "assay_context"
    )
    record["observation_context_id"] = unique_text("context_id")
    record["observation_assay_context"] = unique_text("assay_context")
    replicate_tokens = {_replicate_token(row) for row in rows}
    record["replicate"] = (
        json.loads(next(iter(replicate_tokens)))
        if len(replicate_tokens) == 1 else None
    )
    record["unit"] = unique_text("unit")


def _attach_source_anchor(record: dict, anchor: dict) -> None:
    record["source_anchor_identity"] = anchor.get(
        "source_anchor_identity"
    )
    record["source_anchor_status"] = anchor.get("source_anchor_status")
    record["source_anchor_evidence"] = anchor.get(
        "source_anchor_evidence", []
    )
    record["source_anchor_measured_values"] = anchor.get(
        "measured_values", []
    )


def derive_delta(study: str, spec: dict, record: dict,
                 cand_values: Mapping[str, Mapping[str, Any]],
                 src_anchor_gse114002: Mapping[str, Any],
                 src_anchor_gse200304: Mapping[str, Any],
                 cand_seq_id: str) -> List[dict]:
    """Return lossless delta records, one per resolvable observation slot."""
    cand_seq = record.get("candidate_sequence")
    src_seq = record.get("source_sequence")
    edit_list = record.get("edit_script", [])
    metadata = record.get("metadata", {}) or {}

    def base(source_anchor_rule: str):
        legacy_source_id = cand_seq_id[:-len(CAND_SUFFIX)] + SRC_SUFFIX
        return {
            "pair_id": None,
            "record_id": record.get("record_id"),
            "study": study,
            "benchmark": spec["benchmark"],
            "source_sequence": src_seq,
            "candidate_sequence": cand_seq,
            "edit_list": edit_list,
            "edit_count": len(edit_list),
            "source_id": legacy_source_id,
            "legacy_source_id": legacy_source_id,
            "source_id_status": (
                "LEGACY_MECHANICAL_NOT_PUBLICATION_GROUP"
            ),
            "candidate_id": cand_seq_id,
            "region": record.get("region"),
            "library": metadata.get("library"),
            "source_asset": metadata.get("source_file"),
            "source_anchor_rule": source_anchor_rule,
            "canonical_provenance": {
                "record_id": record.get("record_id"),
                "dataset": record.get("dataset"),
                "accession": record.get("accession"),
                "region": record.get("region"),
                "metadata": metadata,
                "canonical_record_sha256": _json_sha256(record),
            },
        }

    t = spec["type"]

    if t == "diff_cand_src":
        endpoint = spec["cand_endpoint"]
        groups = _observation_groups(cand_values, cand_seq_id, endpoint)
        source_anchor_rule = "diff_cand_src:%s" % spec["src_source"]
        if spec["src_source"] == "raw_library":
            raw_identity = (
                str(metadata.get("library") or "").strip(),
                _normalize_sequence(src_seq),
            )
            raw_anchor = src_anchor_gse114002.get(raw_identity)
            # Synthetic/backward-compatible scalar maps are accepted only at
            # this pure-function boundary. Production loaders use the strict
            # (library, normalized sequence) key above.
            if raw_anchor is None:
                raw_anchor = src_anchor_gse114002.get(src_seq)
            anchor = _coerce_anchor(
                raw_anchor, study, "RAW_LIBRARY_RL", raw_identity
            )
        elif spec["src_source"] == "raw_count":
            # GSE200304: source anchor is the _WT barcode of this pair.
            mid = metadata.get("merged_id")
            wt_id = None
            if mid:
                wt_id = (
                    str(mid)[:-len("_Mutant")] + "_WT"
                    if str(mid).endswith("_Mutant")
                    else str(mid) + "_WT"
                )
            anchor = _coerce_anchor(
                src_anchor_gse200304.get(wt_id) if wt_id else None,
                study,
                "RAW_COUNT_FREQ",
                wt_id,
            )
        else:
            raise ValueError(
                "unknown diff_cand_src source %s" % spec["src_source"]
            )
        slots = groups or {("", "", "null", ""): {
            "value": None, "rows": [],
        }}
        out = []
        for key in sorted(slots):
            observation = slots[key]
            cand_val = observation["value"]
            src_val = anchor.get("value")
            rec = base(source_anchor_rule)
            rec["endpoint"] = endpoint
            rec["candidate_value"] = cand_val
            rec["source_value"] = src_val
            rec["delta"] = (
                cand_val - src_val
                if cand_val is not None and src_val is not None else None
            )
            rec["delta_source_status"] = (
                "derived" if rec["delta"] is not None
                else "source_anchor_unavailable"
            )
            _attach_observation_provenance(rec, observation["rows"])
            _attach_source_anchor(rec, anchor)
            out.append(rec)
        return out

    if t == "diff_ref_alt":
        ref_endpoint = spec["ref_endpoint"]
        alt_endpoint = spec["alt_endpoint"]
        ref_groups = _observation_groups(
            cand_values, cand_seq_id, ref_endpoint
        )
        alt_groups = _observation_groups(
            cand_values, cand_seq_id, alt_endpoint
        )
        keys = sorted(set(ref_groups) | set(alt_groups)) or [
            ("", "", "null", "")
        ]
        out = []
        for key in keys:
            ref_group = ref_groups.get(key, {"value": None, "rows": []})
            alt_group = alt_groups.get(key, {"value": None, "rows": []})
            ref = ref_group["value"]
            alt = alt_group["value"]
            source_anchor_rule = "diff_ref_alt:%s" % ref_endpoint
            rec = base(source_anchor_rule)
            rec["endpoint"] = "%s/%s" % (ref_endpoint, alt_endpoint)
            rec["source_value"] = ref
            rec["candidate_value"] = alt
            rec["delta"] = (
                alt - ref if alt is not None and ref is not None else None
            )
            rec["delta_source_status"] = (
                "derived" if rec["delta"] is not None
                else "source_anchor_unavailable"
            )
            observations = ref_group["rows"] + alt_group["rows"]
            _attach_observation_provenance(rec, observations)
            anchor = _resolved_anchor(
                study,
                "FUNCTIONAL_REF_ENDPOINT",
                {
                    "source_sequence_sha256": hashlib.sha256(
                        _normalize_sequence(src_seq).encode("ascii")
                    ).hexdigest(),
                    "ref_endpoint": ref_endpoint,
                    "observation_join_key": key,
                },
                [] if ref is None else [ref],
                ref_group["rows"],
            )
            _attach_source_anchor(rec, anchor)
            out.append(rec)
        return out

    if t == "log2fc":
        endpoint = spec["endpoint"]
        groups = _observation_groups(cand_values, cand_seq_id, endpoint)
        slots = groups or {("", "", "null", ""): {
            "value": None, "rows": [],
        }}
        out = []
        for key in sorted(slots):
            observation = slots[key]
            value = observation["value"]
            rec = base("log2fc_direct:%s" % endpoint)
            rec["endpoint"] = endpoint
            rec["source_value"] = None
            rec["candidate_value"] = value
            rec["delta"] = value
            rec["delta_source_status"] = (
                "log2fc_direct" if value is not None
                else "source_anchor_unavailable"
            )
            _attach_observation_provenance(rec, observation["rows"])
            anchor = _resolved_anchor(
                study,
                "DIRECT_EFFECT_OBSERVATION",
                {
                    "source_sequence_sha256": hashlib.sha256(
                        _normalize_sequence(src_seq).encode("ascii")
                    ).hexdigest(),
                    "endpoint": endpoint,
                    "observation_join_key": key,
                },
                [] if value is None else [value],
                observation["rows"],
            )
            _attach_source_anchor(rec, anchor)
            out.append(rec)
        return out

    if t == "diff_wt_meta":
        out = []
        for ep_spec in spec["endpoints"]:
            endpoint = ep_spec["cand_endpoint"]
            groups = _observation_groups(cand_values, cand_seq_id, endpoint)
            slots = groups or {("", "", "null", ""): {
                "value": None, "rows": [],
            }}
            raw_wt = metadata.get(ep_spec["wt_meta"])
            wt_values = [] if raw_wt is None else [float(raw_wt)]
            anchor = _resolved_anchor(
                study,
                "CANONICAL_WT_METADATA",
                {
                    "source_sequence_sha256": hashlib.sha256(
                        _normalize_sequence(src_seq).encode("ascii")
                    ).hexdigest(),
                    "metadata_field": ep_spec["wt_meta"],
                },
                wt_values,
                [{
                    "canonical_record_sha256": _json_sha256(record),
                    "record_id": record.get("record_id"),
                    "metadata_field": ep_spec["wt_meta"],
                    "value": raw_wt,
                }] if raw_wt is not None else [],
            )
            for key in sorted(slots):
                observation = slots[key]
                cand_val = observation["value"]
                wt = anchor.get("value")
                rec = base("diff_wt_meta:%s" % ep_spec["wt_meta"])
                rec["endpoint"] = endpoint
                rec["source_value"] = wt
                rec["candidate_value"] = cand_val
                rec["delta"] = (
                    cand_val - wt
                    if cand_val is not None and wt is not None else None
                )
                rec["delta_source_status"] = (
                    "derived" if rec["delta"] is not None
                    else "source_anchor_unavailable"
                )
                _attach_observation_provenance(
                    rec, observation["rows"]
                )
                _attach_source_anchor(rec, anchor)
                out.append(rec)
        return out

    raise ValueError(f"unknown delta type {t}")


def build(out_dir: Path) -> dict:
    staging = Path(config.STAGING_DIR)
    pairs_path = staging / "utr_edit_pairs.jsonl"
    observations_path = staging / "functional_observations.jsonl"
    pairs = load_pairs(pairs_path)

    # Map pair -> candidate_seq_id / source_seq_id / study
    needed_record_ids = set()
    cand_seq_ids_by_study: Dict[str, set] = defaultdict(set)
    for p in pairs:
        cid = p["candidate_sequence_id"]
        rid = cid[:-len(CAND_SUFFIX)]
        needed_record_ids.add(rid)
        cand_seq_ids_by_study[_study_of(cid)].add(cid)

    canonical = load_canonical(Path(config.CANONICAL_RECORDS), needed_record_ids)

    cand_values = load_candidate_observations(
        observations_path, cand_seq_ids_by_study
    )

    src_114 = load_gse114002_source_anchors() if "GSE114002" in config.ACTIVE_STUDIES else {}
    src_200 = load_gse200304_source_anchors() if "GSE200304" in config.ACTIVE_STUDIES else {}

    records = []
    per_asset = defaultdict(lambda: defaultdict(int))
    for p in pairs:
        cid = p["candidate_sequence_id"]
        rid = cid[:-len(CAND_SUFFIX)]
        study = _study_of(cid)
        rec0 = canonical.get(rid)
        if rec0 is None:
            per_asset[study]["missing_canonical"] += 1
            continue
        spec = config.DELTA_SPECS[study]
        deltas = derive_delta(study, spec, rec0, cand_values,
                              src_114, src_200, cid)
        for d in deltas:
            d["pair_id"] = p["pair_id"]
            pair_provenance = dict(p)
            pair_provenance["evidence_sha256"] = _json_sha256(p)
            pair_provenance["relation_status"] = (
                p.get("relation_status") or "UNREVIEWED"
            )
            d["pair_provenance"] = pair_provenance
            records.append(d)
            per_asset[study]["total"] += 1
            per_asset[study][d["delta_source_status"]] += 1

    records = sorted(records, key=_canonical_json)
    assert_ordinary_rows(records, purpose="effect dataset output")
    out_dir = Path(out_dir)
    assert_ordinary_path(out_dir, purpose="effect dataset output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "effect_dataset.jsonl"
    man_path = out_dir / "effect_dataset_manifest.json"
    assert_ordinary_path(out_path, purpose="effect dataset output")
    assert_ordinary_path(man_path, purpose="effect dataset manifest output")
    if out_path.exists() or man_path.exists():
        raise FileExistsError(
            "effect builder refuses to overwrite existing outputs: %s, %s"
            % (out_path, man_path)
        )
    with out_path.open("x", encoding="utf-8") as f:
        for r in records:
            f.write(_canonical_json(r) + "\n")

    input_paths = {
        "pairs": pairs_path,
        "functional_observations": observations_path,
        "canonical": Path(config.CANONICAL_RECORDS),
    }
    if "GSE114002" in config.ACTIVE_STUDIES:
        input_paths["gse114002_raw_library"] = Path(
            config.GSE114002_RAW_LIB
        )
    if "GSE200304" in config.ACTIVE_STUDIES:
        input_paths["gse200304_raw_count"] = Path(config.GSE200304_COUNT)
        input_paths["gse200304_twist"] = Path(config.GSE200304_TWIST)
    for study, raw_path in sorted(config.RECONSTRUCTED_PAIRS.items()):
        path = Path(raw_path)
        if path.exists():
            input_paths["reconstructed_pairs_%s" % study] = path

    anchor_status_counts = defaultdict(int)
    for row in records:
        anchor_status_counts[str(row.get("source_anchor_status"))] += 1

    manifest = {
        "manifest_version": "b0x_effect_dataset_manifest_v2",
        "functional_observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "n_records": len(records),
        "n_pairs_scanned": len(pairs),
        "per_asset": {k: dict(v) for k, v in per_asset.items()},
        "source_anchor_status_counts": dict(
            sorted(anchor_status_counts.items())
        ),
        "legacy_source_lineage": {
            "status": "NOT_PUBLICATION_GROUPING",
            "n_unique_legacy_source_ids": len({
                row.get("legacy_source_id") for row in records
                if row.get("legacy_source_id")
            }),
            "n_unique_source_anchor_identities": len({
                row.get("source_anchor_identity") for row in records
                if row.get("source_anchor_identity")
            }),
            "required_next_step": (
                "R3 ordinary preparation must bind a frozen context and "
                "construct source_group_id_v2 before publication ranking."
            ),
        },
        "sha256": sha256_file(out_path),
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(input_paths.items())
        },
        "config": {
            "staging": str(staging),
            "canonical": config.CANONICAL_RECORDS,
            "split": config.PRIMARY_SPLIT,
            "seed": config.SEED,
        },
    }
    with man_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(
            manifest, indent=2, sort_keys=True, ensure_ascii=False
        ) + "\n")
    return manifest


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/b0x"))
    args = ap.parse_args()
    man = build(args.out_dir)
    print(json.dumps({k: (v if k != "per_asset" else v) for k, v in man.items() if k != "sha256"}, indent=2))
    print("sha256:", man["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
