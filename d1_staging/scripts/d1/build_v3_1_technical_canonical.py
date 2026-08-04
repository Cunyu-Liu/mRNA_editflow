#!/usr/bin/env python
"""D1-R (v3.1): Rebuild the technical canonical from the existing D1 canonical records.

Reads ``data/d1_canonical_records.jsonl`` (the intermediate canonical produced by
``build_canonical_records.py`` from raw P0 assets) and emits the v3.1 schema-compliant
technical canonical artifacts:

  Track E (relations):  utr_edit_relation_candidates + utr_edit_pairs
  Track F (observations): functional_observation_candidates + functional_observations
  sequence_entities (+ endpoint_registry)
  rejection_records, transformation_edges, exposure_records, use_roles,
  group_registry, group_assignments, effective_exposure_projection

GSE246381 rows are routed to the restricted sealed mirror (never written to the
ordinary workspace). All outputs are deterministically id-ed and hash-bound.

This is a data-engineering tool: it performs no training and no GPU work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, OrderedDict
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from grouping_atoms_v3_1 import (  # noqa: E402
    ASSIGNMENT_ALGORITHM_ID,
    GROUPING_ATOM_RULE_SHA256,
    PROJECTION_POLICY,
    derive_grouping_atoms,
    group_id_for,
    group_sha256,
)

REGION_SCOPE = {"5'UTR": "5UTR", "3'UTR": "3UTR", "5UTR": "5UTR", "3UTR": "3UTR"}
RESTRICTED_SEALED = {"GSE246381"}
STUDY_LABEL = {
    "GSE114002": "sample2019", "GSE145046": "gse145046", "GSE149487": "lim2021_5utr_mpra",
    "GSE173083": "lepplek2022_persistseq", "GSE186455": "gse186455", "GSE200304": "gse200304",
    "GSE207584": "gse207584", "GSE217518": "gse217518", "GSE232572": "gse232572_maputr",
    "GSE246381": "gse246381", "ENCSR854RUF": "encsr854ruf_mprau",
}
FUTURE_ROLE_ORDINARY = "AWAITING_B0_GLOBAL_DISPOSITION"
FUTURE_ROLE_SEALED = "SEALED_EXTERNAL_FINAL_CANDIDATE"
SEALED_OBJECT_PREFIX = "gse246381"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_str(s: str) -> str:
    return sha256_hex(s.encode("utf-8"))


def normalize_seq(s: str) -> str:
    return s.upper()


def region_scope(region: str) -> str:
    return REGION_SCOPE.get(region, "MULTI_REGION")


def sequence_scope(region: str) -> str:
    rs = region_scope(region)
    if rs in ("5UTR", "3UTR"):
        return rs
    return "MULTI_REGION"


def is_restricted(accession: str) -> bool:
    return accession in RESTRICTED_SEALED


def jl(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------------------
# endpoint registry
# ---------------------------------------------------------------------------

# Endpoint identification: each label key becomes an endpoint. Scaling is a
# conservative default (LOG for ratio/count-like continuous values, LINEAR for
# raw values). The registry is frozen per run and emitted once.
def _scaling_for(name: str) -> str:
    low = name.lower()
    if "log2" in low or "lfc" in low or "log_fold" in low:
        return "LOG"
    return "LINEAR"


def build_endpoint_registry(records_iter) -> "OrderedDict[str, dict]":
    """Return {endpoint_id: row} for every distinct label key across all records."""
    reg: "OrderedDict[str, dict]" = OrderedDict()
    for rec in records_iter:
        for key in (rec.get("labels") or {}).keys():
            eid = f"ep_{key}"
            if eid not in reg:
                reg[eid] = {
                    "endpoint_id": eid,
                    "name": key,
                    "scaling": _scaling_for(key),
                    "missing_token": False,
                    "missing_mask": False,
                }
    return reg


# ---------------------------------------------------------------------------
# sequence entity
# ---------------------------------------------------------------------------


def build_sequence_entity(seq: str, seq_id: str, region: str) -> dict:
    raw = seq
    norm = normalize_seq(seq)
    full = norm
    return {
        "sequence_id": seq_id,
        "sequence_scope": sequence_scope(region),
        "raw_sequence_sha256": sha256_str(raw),
        "normalized_sequence_sha256": sha256_str(norm),
        "full_sequence_sha256": sha256_str(full),
        "window_start": None,
        "window_end": None,
        "original_length": len(raw),
        "region_scope": region_scope(region),
        "scaffold": None,
        "editable_mask": None,
    }


def sequence_id_for(record_id: str, side: str) -> str:
    return f"{record_id}__{side}"


def _new_group_sink(base: Path):
    """Create an external-sort membership sink beside the output tree."""
    tmp_dir = Path(tempfile.mkdtemp(prefix=".grouping_atoms_", dir=str(base.parent)))
    return tmp_dir, open(tmp_dir / "memberships.tsv", "w", encoding="utf-8")


def _emit_group_registry(membership_path: Path, group_writer):
    """Materialize one complete GroupRegistry row per atom group.

    Memberships are externally sorted so the builder does not retain the
    multi-million-row object-to-group relation in RAM.
    """
    sorted_path = membership_path.with_name("memberships.sorted.tsv")
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    subprocess.run(
        [
            "sort", "-T", str(membership_path.parent), "-t", "\t",
            "-k1,1", "-k4,4", str(membership_path), "-o", str(sorted_path),
        ],
        check=True,
        env=env,
    )

    current_key = None
    current_members = []

    def flush():
        if current_key is None:
            return
        gid, atom = current_key
        members = current_members
        group_writer.write(jl({
            "group_id": gid,
            "grouping_atom": atom,
            "member_ids": members,
            "group_sha256": group_sha256(gid, atom, members),
        }) + "\n")

    with open(sorted_path, "r", encoding="utf-8") as fh:
        last_member = None
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                raise ValueError(f"malformed grouping membership row: {line!r}")
            gid, atom, _object_type, object_id = parts
            key = (gid, atom)
            if key != current_key:
                flush()
                current_key = key
                current_members = []
                last_member = None
            if object_id != last_member:
                current_members.append(object_id)
                last_member = object_id
    flush()
    sorted_path.unlink()


def _write_grouping_projection_audit(path: Path, counters: Counter):
    payload = {
        "artifact_id": "d1_grouping_atom_projection_v1",
        "schema_version": "3.1",
        "grouping_atom_projection_rule_sha256": GROUPING_ATOM_RULE_SHA256,
        "assignment_algorithm_id": ASSIGNMENT_ALGORITHM_ID,
        "group_id_algorithm_id": "SHA256_ATOM_VALUE_V1",
        "invented_atom_forbidden": True,
        "projection_policy": PROJECTION_POLICY,
        "counters": dict(sorted(counters.items())),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# main builder
# ---------------------------------------------------------------------------


def build(out_dir: Path, restricted_dir: Path, canonical_path: Path, config_hash: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    restricted_dir.mkdir(parents=True, exist_ok=True)

    # ---- first pass: enumerate label keys to build the endpoint registry ----
    with open(canonical_path, "r", encoding="utf-8") as fh:
        endpoint_reg = build_endpoint_registry((json.loads(l) for l in fh))

    # ---- second pass: emit artifacts ----
    counters = Counter()
    seq_log: dict[str, str] = {}  # seq_id -> canonical record_id (creator)
    seq_distinct: set[str] = set()
    ordinary_group_tmp, ordinary_membership_w = _new_group_sink(out_dir)
    restricted_group_tmp, restricted_membership_w = _new_group_sink(restricted_dir)
    grouping_audit = {"ordinary": Counter(), "restricted": Counter()}

    # output writers
    def open_writer(path: Path):
        f = open(path, "w", encoding="utf-8")
        return f

    # ordinary writers
    seq_w = open_writer(out_dir / "sequence_entities.jsonl")
    obs_cand_w = open_writer(out_dir / "functional_observation_candidates.jsonl")
    obs_w = open_writer(out_dir / "functional_observations.jsonl")
    rel_cand_w = open_writer(out_dir / "utr_edit_relation_candidates.jsonl")
    pair_w = open_writer(out_dir / "utr_edit_pairs.jsonl")
    use_w = open_writer(out_dir / "use_roles.jsonl")
    group_w = open_writer(out_dir / "group_registry.jsonl")
    group_assign_w = open_writer(out_dir / "group_assignments.jsonl")
    attrs_w = open_writer(out_dir / "object_attributes.jsonl")
    rej_w = open_writer(out_dir / "rejection_records.jsonl")
    trans_w = open_writer(out_dir / "transformation_edges.jsonl")
    exp_w = open_writer(out_dir / "exposure_records.jsonl")
    eff_w = open_writer(out_dir / "effective_exposure_projection.jsonl")
    # restricted writers (sealed mirror)
    r_dir = restricted_dir / "sealed_external" / "GSE246381"
    r_dir.mkdir(parents=True, exist_ok=True)
    r_seq_w = open_writer(r_dir / "sequence_entities.jsonl")
    r_obs_cand_w = open_writer(r_dir / "functional_observation_candidates.jsonl")
    r_obs_w = open_writer(r_dir / "functional_observations.jsonl")
    r_rel_cand_w = open_writer(r_dir / "utr_edit_relation_candidates.jsonl")
    r_pair_w = open_writer(r_dir / "utr_edit_pairs.jsonl")
    r_use_w = open_writer(r_dir / "use_roles.jsonl")
    r_group_w = open_writer(r_dir / "group_registry.jsonl")
    r_group_assign_w = open_writer(r_dir / "group_assignments.jsonl")
    r_attrs_w = open_writer(r_dir / "object_attributes.jsonl")
    r_rej_w = open_writer(r_dir / "rejection_records.jsonl")
    r_trans_w = open_writer(r_dir / "transformation_edges.jsonl")
    r_exp_w = open_writer(r_dir / "exposure_records.jsonl")
    r_eff_w = open_writer(r_dir / "effective_exposure_projection.jsonl")
    # restricted append-only access log (JCS chain)
    r_access_w = open_writer(r_dir / "ACCESS_LOG.jsonl")

    ordinary_grouping_writers = (group_w, group_assign_w, attrs_w, ordinary_membership_w)
    restricted_grouping_writers = (r_group_w, r_group_assign_w, r_attrs_w, restricted_membership_w)

    # endpoint registry emitted once (ordinary only; restricted recomputes from rows)
    ep_w = open_writer(out_dir / "endpoint_registry.jsonl")
    for row in endpoint_reg.values():
        ep_w.write(jl(row) + "\n")
    ep_w.close()
    r_ep_w = open_writer(r_dir / "endpoint_registry.jsonl")
    for row in endpoint_reg.values():
        if row["name"].startswith(("ep_", "hek_", "te_", "dna_", "log2fc_")):
            r_ep_w.write(jl(row) + "\n")
    r_ep_w.close()

    prev_access_sha = None
    access_seq = 0

    def access_event(intent, obj_id, status, reason=None):
        nonlocal prev_access_sha, access_seq
        payload = {
            "access_id": f"gse246381_access_{access_seq}",
            "object_id": obj_id,
            "intent": intent,
            "status": status,
            "prev_event_sha256": prev_access_sha,
        }
        if reason:
            payload["reason"] = reason
        clean = {k: v for k, v in payload.items() if k != "event_sha256"}
        ev = sha256_str(jl(clean))
        payload["event_sha256"] = ev
        prev_access_sha = ev
        access_seq += 1
        r_access_w.write(jl(payload) + "\n")
        return payload

    def emit_object_groupings(
        rec, object_id, object_type, scientific_track, region, source_id,
        candidate_id, source_sequence, candidate_sequence, context_id,
        local_attrs_w, local_group_assign_w, local_membership_w, audit,
    ):
        atoms = derive_grouping_atoms(
            rec, source_id, candidate_id, source_sequence, candidate_sequence,
            object_type, object_id, context_id=context_id,
        )
        accession = str(rec.get("accession") or "").lower()
        local_attrs_w.write(jl({
            "object_id": object_id,
            "object_type": object_type,
            "scientific_track": scientific_track,
            "region_scope": region_scope(region),
            "study": accession or None,
            "group_ids_by_atom": {
                atom: [group_id_for(atom, value) for value in values]
                for atom, values in atoms.items()
            },
        }) + "\n")
        audit[f"objects:{object_type}"] += 1
        for atom, values in atoms.items():
            audit[f"assigned_objects:{object_type}:{atom}"] += 1
            for value_index, value in enumerate(values):
                gid = group_id_for(atom, value)
                assignment = {
                    "assignment_id": f"asg_{object_id}_{atom.lower()}_{value_index}",
                    "object_id": object_id,
                    "object_type": object_type,
                    "group_id": gid,
                    "grouping_atom": atom,
                    "assignment_algorithm_id": ASSIGNMENT_ALGORITHM_ID,
                }
                local_group_assign_w.write(jl(assignment) + "\n")
                local_membership_w.write(
                    f"{gid}\t{atom}\t{object_type}\t{object_id}\n"
                )
                audit[f"assignments:{object_type}:{atom}"] += 1
        for atom in (
            "GENE", "TRANSCRIPT", "TILE_FAMILY", "SEQUENCE_CLUSTER",
            "LIBRARY_LINEAGE", "STUDY", "BIOLOGICAL_PARENT",
        ):
            if atom not in atoms:
                audit[f"missing_objects:{object_type}:{atom}"] += 1
        return atoms

    bias = 0
    with open(canonical_path, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            rec = json.loads(line)
            acc = rec.get("accession")
            restricted = is_restricted(acc)
            rid = rec.get("record_id")
            if not rid:
                counters["records_without_id"] += 1
                continue
            region = rec.get("region") or "5'UTR"
            labels = rec.get("labels") or {}
            source = rec.get("source_sequence")
            candidate = rec.get("candidate_sequence")
            has_pair = bool(source and candidate)

            local_seq_w = r_seq_w if restricted else seq_w
            local_obs_cand_w = r_obs_cand_w if restricted else obs_cand_w
            local_obs_w = r_obs_w if restricted else obs_w
            local_rel_cand_w = r_rel_cand_w if restricted else rel_cand_w
            local_pair_w = r_pair_w if restricted else pair_w
            local_use_w = r_use_w if restricted else use_w
            local_group_w = r_group_w if restricted else group_w
            local_group_assign_w = r_group_assign_w if restricted else group_assign_w
            local_attrs_w = r_attrs_w if restricted else attrs_w
            local_membership_w = restricted_membership_w if restricted else ordinary_membership_w
            local_grouping_audit = grouping_audit["restricted" if restricted else "ordinary"]
            local_rej_w = r_rej_w if restricted else rej_w
            local_trans_w = r_trans_w if restricted else trans_w
            local_exp_w = r_exp_w if restricted else exp_w
            local_eff_w = r_eff_w if restricted else eff_w

            # ---- sequence entities ----
            source_id = sequence_id_for(rid, "src")
            cand_id = sequence_id_for(rid, "cand")
            if source:
                local_seq_w.write(jl(build_sequence_entity(source, source_id, region)) + "\n")
                seq_distinct.add(source_id)
                seq_log[source_id] = rid
                counters["source_sequences"] += 1
            if candidate:
                local_seq_w.write(jl(build_sequence_entity(candidate, cand_id, region)) + "\n")
                seq_distinct.add(cand_id)
                seq_log[cand_id] = rid
                counters["candidate_sequences"] += 1

            # ---- Track E relation (source -> candidate) ----
            if has_pair:
                cand_id_e = f"{acc.lower()}_rel_{bias}"
                cand_id_f = f"{acc.lower()}_pair_{bias}"
                bias += 1
                design_group = f"{acc.lower()}_design_{rid}"
                role = FUTURE_ROLE_SEALED if restricted else FUTURE_ROLE_ORDINARY
                rel = {
                    "candidate_id": cand_id_e,
                    "source_sequence_id": source_id,
                    "candidate_sequence_id": cand_id,
                    "pairing_method": "native_reconstruction",
                    "evidence_id": f"evi_{rid}",
                    "lifecycle_status": "ACCEPTED",
                }
                pair = {
                    "pair_id": cand_id_f,
                    "candidate_id": cand_id_e,
                    "source_sequence_id": source_id,
                    "candidate_sequence_id": cand_id,
                    "design_relation_group_id": design_group,
                    "scientific_track": "E",
                    "relation_type": "SOURCE_CONDITIONED_EDIT",
                    "effect_evidence": None,
                    "landscape_role": None,
                    "immutable_base_future_use_role": role,
                    "pairing_method": "native_reconstruction",
                    "evidence_id": f"evi_{rid}",
                }
                local_rel_cand_w.write(jl(rel) + "\n")
                local_pair_w.write(jl(pair) + "\n")
                counters["pairs"] += 1
                emit_object_groupings(
                    rec, cand_id_f, "PAIR", "E", region, source_id, cand_id,
                    source, candidate, None, local_attrs_w, local_group_assign_w,
                    local_membership_w, local_grouping_audit,
                )
            else:
                counters["observational_rows"] += 1

            # ---- Track F observations (one per label endpoint) ----
            # no-edit/no-sequence rows (no source AND no candidate) cannot bind a
            # functional observation; they are quarantined per the contract.
            if not source and not candidate:
                counters["no_sequence_rows_rejected"] += 1
                local_rej_w.write(jl({
                    "rejection_id": f"rej_{rid}",
                    "candidate_id": f"ep_{rid}",
                    "reason": "NO_BINDABLE_SEQUENCE",
                    "evidence_id": f"evi_{rid}",
                    "rejected_at": "2026-08-03T00:00:00Z",
                }) + "\n")
                for key in labels:
                    counters["rejected_observations"] += 1
            for key, value in labels.items():
                if not source and not candidate:
                    continue
                if value is None:
                    counters["null_labels_skipped"] += 1
                    local_rej_w.write(jl({
                        "rejection_id": f"rej_{rid}_{key}",
                        "candidate_id": f"ep_{key}",
                        "reason": "NULL_VALUE",
                        "evidence_id": f"evi_{rid}",
                        "rejected_at": "2026-08-03T00:00:00Z",
                    }) + "\n")
                    continue
                try:
                    fval = float(value)
                except (TypeError, ValueError):
                    counters["nonnumeric_labels_skipped"] += 1
                    local_rej_w.write(jl({
                        "rejection_id": f"rej_{rid}_{key}",
                        "candidate_id": f"ep_{key}",
                        "reason": "NON_NUMERIC_VALUE",
                        "evidence_id": f"evi_{rid}",
                        "rejected_at": "2026-08-03T00:00:00Z",
                    }) + "\n")
                    continue
                eid = f"ep_{key}"
                obs_cand_id = f"{acc.lower()}_obsc_{bias}"
                obs_id = f"{acc.lower()}_obs_{bias}"
                bias += 1
                ctx_id = f"ctx_{str(acc).lower()}" if acc else "ctx_unknown"
                oc = {
                    "candidate_id": obs_cand_id,
                    "sequence_id": cand_id if candidate else source_id,
                    "endpoint_id": eid,
                    "context_id": ctx_id,
                    "source": "native_d1",
                    "source_file_sha256": config_hash,
                    "value": fval,
                    "lifecycle_status": "ACCEPTED",
                }
                obs = {
                    "observation_id": obs_id,
                    "sequence_id": cand_id if candidate else source_id,
                    "endpoint_id": eid,
                    "context_id": ctx_id,
                    "value": fval,
                    "unit": key,
                    "replicate": None,
                }
                local_obs_cand_w.write(jl(oc) + "\n")
                local_obs_w.write(jl(obs) + "\n")
                counters["observations"] += 1
                emit_object_groupings(
                    rec, obs_id, "OBSERVATION", "F", region, source_id, cand_id,
                    source, candidate, ctx_id, local_attrs_w, local_group_assign_w,
                    local_membership_w, local_grouping_audit,
                )

            # ---- use role (per object) ----
            role = FUTURE_ROLE_SEALED if restricted else FUTURE_ROLE_ORDINARY
            authority = "SEALED" if restricted else "ORDINARY"
            for obj_id in ([] if not source else [source_id]) + ([cand_id] if candidate else []):
                local_use_w.write(jl({
                    "object_id": obj_id,
                    "use_role": "D1_CANONICAL",
                    "future_use_role": role,
                    "authority_level": authority,
                }) + "\n")
                counters["use_roles"] += 1
                # immutable baseline exposure record (no analytic/final use at D1)
                local_exp_w.write(jl({
                    "access_id": f"exp_{obj_id}",
                    "object_id": obj_id,
                    "intent": "D1_CANONICAL_BUILD",
                    "status": "COMPLETION",
                    "prev_event_sha256": None,
                    "event_sha256": sha256_str(jl({"object_id": obj_id, "intent": "D1_CANONICAL_BUILD"})),
                }) + "\n")
                # effective exposure projection (AWAITING_B0; GSE246381 sealed)
                local_eff_w.write(jl({
                    "object_id": obj_id,
                    "effective_exposure": "SEALED_EXTERNAL_FINAL_CANDIDATE" if restricted else "AWAITING_B0_GLOBAL_DISPOSITION",
                    "projection_sha256": sha256_str(jl({"object_id": obj_id})),
                    "chain_root_sha256": None,
                }) + "\n")
                counters["exposure_records"] += 1

            # restricted access event (builder machine event only)
            if restricted:
                access_event("restricted_d1_builder", rid, "COMPLETION")

            counters["records"] += 1
            if idx % 200000 == 0 and idx:
                print(f"  ...{idx} records processed", flush=True)

    # transformation_edges.jsonl: no supersession at D1 (identity roots only); file left empty
    for w in (seq_w, obs_cand_w, obs_w, rel_cand_w, pair_w, use_w, group_assign_w,
              attrs_w, rej_w, trans_w, exp_w, eff_w,
              r_seq_w, r_obs_cand_w, r_obs_w, r_rel_cand_w, r_pair_w, r_use_w,
              r_group_assign_w, r_attrs_w, r_rej_w, r_trans_w, r_exp_w, r_eff_w):
        w.close()
    ordinary_membership_w.close()
    restricted_membership_w.close()

    # GroupRegistry is emitted only after all object assignments are known so
    # member_ids are complete and sorted. The temporary membership files are
    # outside the canonical output tree and are removed after materialization.
    _emit_group_registry(ordinary_group_tmp / "memberships.tsv", group_w)
    _emit_group_registry(restricted_group_tmp / "memberships.tsv", r_group_w)
    group_w.close()
    r_group_w.close()
    _write_grouping_projection_audit(out_dir / "GROUPING_ATOM_PROJECTION.json", grouping_audit["ordinary"])
    _write_grouping_projection_audit(r_dir / "GROUPING_ATOM_PROJECTION.json", grouping_audit["restricted"])
    shutil.rmtree(ordinary_group_tmp)
    shutil.rmtree(restricted_group_tmp)
    r_access_w.close()

    counters["distinct_sequences"] = len(seq_distinct)
    return counters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", required=True,
                    help="Path to d1_canonical_records.jsonl")
    ap.add_argument("--out", required=True, help="ordinary output dir")
    ap.add_argument("--restricted-out", required=True, help="restricted output dir")
    ap.add_argument("--config-hash", default="v3.1-D1",
                    help="config/adapter hash bound to every artifact")
    args = ap.parse_args()

    stats = build(Path(args.out), Path(args.restricted_out), Path(args.canonical), args.config_hash)
    print(json.dumps({k: v for k, v in stats.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
