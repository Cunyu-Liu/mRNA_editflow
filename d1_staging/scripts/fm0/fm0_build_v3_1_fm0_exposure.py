#!/usr/bin/env python
"""FM0-A (v3.1): per-checkpoint foundation exposure pre-check (no training).

Builds, from the D1 technical canonical and the frozen foundation config, the
FM0-A data-engineering closure:

  ordinary:
    foundation_candidates.json
    foundation_exposure_ledger.jsonl
    overlap_report.json
    GSE246381_FM0_AGGREGATE.json
    GSE246381_FM0_COMMITMENT.json
    versioned FM0 EffectiveExposureProjection (fm0_effective_exposure_projection.jsonl)

  restricted:
    <restricted>/sealed_external/GSE246381/FM0_ACCESS_LOG.jsonl (append)
    <restricted>/sealed_external/GSE246381/FM0_AGGREGATE.json

FM0-A only audits documented/external foundation overlap against the frozen
E/F/GSE clusters. It does NOT train, does NOT select training results, and
does NOT open any analytic/final evaluator.

Ledger unique key = object/cluster x checkpoint(ID/revision/weights_hash) x audit_run.

Candidate policy (contract §FM0-A):
  - allowed kinds: from_scratch_E, supervised_F_to_E, general_backbone (<=1),
    specialist (<=1 per region).
  - project-internal candidates (from_scratch_E, supervised_F_to_E) have no
    external pretraining corpus -> always overlap-clean on the overlap axis.
  - external backbone (UTR-LM) has a documented pretraining corpus; overlap is
    determined per cluster from documented corpus evidence (we do not possess
    the full pretraining corpus locally, so exact/near membership is recorded
    from the model's documented pretraining sources, marked DETECTED/CLEAN/
    UNKNOWN accordingly).

This is a data-engineering tool: it performs no training and no GPU work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (frozen, from config + D1 schema)
# ---------------------------------------------------------------------------

AUDIT_RUN = "audit_fm0a_v1"
CONFIG_HASH = "v3.1-FM0-A"

# Candidate policy
ALLOWED_KINDS = ("from_scratch_E", "supervised_F_to_E", "general_backbone", "specialist")
MAX_GENERAL_BACKBONES = 1
MAX_SPECIALISTS_PER_REGION = 1

# Frozen external foundation backbone (from configs/fm0_utrlm_config.yaml)
EXTERNAL_BACKBONE = {
    "candidate_id": "utrlm_general_backbone",
    "kind": "general_backbone",
    "is_external": True,
    "region": "5UTR",
    "model_id": "multimolecule/utrlm-mrl",
    "revision": "79e23de069449e659696b5210f833c28ddd0de50",
    "weights_sha256": "4646f79e76d970ed51aefad811777390ecd43a3e3e5ed6372780583d3be1541a",
    "license": "AGPL-3.0",
    "license_type": "agpl-3.0",
    "corpus_evidence": [
        "Ensembl 5'UTR (5 species: human, rat, mouse, chicken, zebrafish)",
        "Sample et al. 2019 (GSE114002) 5'UTR MPRA library",
        "Cao et al. 2021 (Nat Commun) 5'UTR MPRA library",
    ],
    "determination_method": "documented_corpus_evidence",
}

# Documented corpus -> accession overlap for the external backbone.
# UTR-LM saw GSE114002 (Sample2019) 5'UTR sequences during pretraining.
# DSL: {accession: {"overlap": "DETECTED"|"CLEAN", "type": str, "note": str}}
BACKBONE_OVERLAP = {
    "GSE114002": {"overlap": "DETECTED", "type": "sequence_prior_only",
                  "note": "UTR-LM documented pretraining corpus includes Sample2019 (GSE114002) 5'UTR sequences."},
    "GSE200304": {"overlap": "CLEAN", "type": "none",
                  "note": "3'UTR data; UTR-LM is 5'UTR-only pretraining."},
    "GSE217518": {"overlap": "CLEAN", "type": "none",
                  "note": "3'UTR data; UTR-LM is 5'UTR-only pretraining."},
    "GSE232572": {"overlap": "CLEAN", "type": "none",
                  "note": "3'UTR data; UTR-LM is 5'UTR-only pretraining."},
    "GSE186455": {"overlap": "CLEAN", "type": "none",
                  "note": "3'UTR data; UTR-LM is 5'UTR-only pretraining."},
    "GSE149487": {"overlap": "CLEAN", "type": "none",
                  "note": "5'UTR PLUMAGE; distinct from documented UTR-LM corpus."},
    "GSE145046": {"overlap": "CLEAN", "type": "none",
                  "note": "5'UTR; not in documented UTR-LM corpus."},
    "GSE173083": {"overlap": "CLEAN", "type": "none",
                  "note": "5'UTR (PersistSeq); not in documented UTR-LM corpus."},
    "ENCSR854RUF": {"overlap": "CLEAN", "type": "none",
                    "note": "5'UTR MPRA-U; not in documented UTR-LM corpus."},
    "GSE207584": {"overlap": "UNKNOWN", "type": "unknown",
                  "note": "No bindable sequences in D1 (label-only rows quarantined); overlap not determinable."},
    "GSE246381": {"overlap": "CLEAN", "type": "none",
                  "note": "Sealed external final candidate; not in documented UTR-LM corpus. Analyzed only as aggregate."},
}

# accessions used in ordinary (non-restricted) workspace
ORDINARY_ACCESSIONS = [
    "GSE114002", "GSE200304", "GSE217518", "GSE232572", "GSE186455",
    "GSE149487", "GSE145046", "GSE173083", "ENCSR854RUF", "GSE207584",
]
RESTRICTED_ACCESSIONS = ["GSE246381"]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_str(s: str) -> str:
    return sha256_hex(s.encode("utf-8"))


def jl(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def accession_of(seq_id: str) -> str:
    """Return the leading accession token from a sequence_id (e.g. GSE114002_...)."""
    head = seq_id.split("_")[0]
    # normalize case of the GSE and ENCSR prefixes
    return head.upper()


# ---------------------------------------------------------------------------
# cluster registry (from D1)
# ---------------------------------------------------------------------------


def build_clusters(seq_file: Path, pair_file: Path, obs_file: Path) -> "OrderedDict[str, dict]":
    """Build frozen E/F/GSE clusters from D1.

    Returns {accession: {region, e_pairs, f_observations}}. Region is derived
    from sequence_entities region_scope (first-seen per accession).
    """
    regions: "OrderedDict[str, str]" = OrderedDict()
    for row in iter_jsonl(seq_file):
        acc = accession_of(row.get("sequence_id", ""))
        if acc and acc not in regions:
            regions[acc] = row.get("region_scope", "UNKNOWN")

    e_counts: Counter = Counter()
    for row in iter_jsonl(pair_file):
        acc = accession_of(row.get("pair_id", ""))
        if acc:
            e_counts[acc] += 1

    f_counts: Counter = Counter()
    for row in iter_jsonl(obs_file):
        acc = accession_of(row.get("observation_id", ""))
        if acc:
            f_counts[acc] += 1

    clusters: "OrderedDict[str, dict]" = OrderedDict()
    for acc in regions:
        clusters[acc] = {
            "cluster_id": acc,
            "region": regions[acc],
            "e_pairs": e_counts.get(acc, 0),
            "f_observations": f_counts.get(acc, 0),
        }
    # include any accessions seen only in pairs/obs but not in sequences
    for acc in set(e_counts) | set(f_counts):
        if acc not in clusters:
            clusters[acc] = {
                "cluster_id": acc,
                "region": "UNKNOWN",
                "e_pairs": e_counts.get(acc, 0),
                "f_observations": f_counts.get(acc, 0),
            }
    return clusters


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------


def build_candidates(clusters: "OrderedDict[str, dict]") -> dict:
    """Build foundation_candidates.json subject to candidate policy."""
    backbone = dict(EXTERNAL_BACKBONE)

    # general backbone eligibility: overlap-clean across ALL clusters it claims
    backbone_overlap_hits = []
    for acc in clusters:
        o = BACKBONE_OVERLAP.get(acc, {"overlap": "UNKNOWN", "type": "unknown",
                                        "note": "no documented membership"})
        if o["overlap"] == "DETECTED":
            backbone_overlap_hits.append(acc)
    backbone["eligible"] = len(backbone_overlap_hits) == 0
    backbone["eligible_overlap_axis"] = len(backbone_overlap_hits) == 0
    backbone["license_clean"] = True  # AGPL-3.0 internal research use permitted
    backbone["overlap_detected_accessions"] = backbone_overlap_hits
    backbone["eligibility_reason"] = (
        "DETECTED sequence overlap with documented pretraining corpus for: "
        + (", ".join(sorted(backbone_overlap_hits)) if backbone_overlap_hits else "none")
        + ". Final alias must be overlap-clean; external backbone is excluded where DETECTED."
    )

    # project-internal candidates: no external corpus -> overlap-clean
    from_scratch = {
        "candidate_id": "from_scratch_E",
        "kind": "from_scratch_E",
        "is_external": False,
        "region": None,
        "model_id": None,
        "revision": None,
        "weights_sha256": None,
        "license": "PROJECT_INTERNAL",
        "license_type": "project_internal",
        "corpus_evidence": ["No external pretraining corpus; train from scratch on Track E."],
        "eligible": True,
        "eligible_overlap_axis": True,
        "license_clean": True,
        "overlap_detected_accessions": [],
        "eligibility_reason": "No external pretraining corpus -> overlap-clean; project-internal license.",
    }
    supervised = {
        "candidate_id": "supervised_F_to_E",
        "kind": "supervised_F_to_E",
        "is_external": False,
        "region": None,
        "model_id": None,
        "revision": None,
        "weights_sha256": None,
        "license": "PROJECT_INTERNAL",
        "license_type": "project_internal",
        "corpus_evidence": ["No external pretraining corpus; supervised transfer F->E."],
        "eligible": True,
        "eligible_overlap_axis": True,
        "license_clean": True,
        "overlap_detected_accessions": [],
        "eligibility_reason": "No external pretraining corpus -> overlap-clean; license project-internal. "
                             "F<->E lineage cleanliness verified in B0; if not clean, falls back to from_scratch_E.",
    }

    candidates = [from_scratch, supervised, backbone]

    # policy check
    n_general = sum(1 for c in candidates if c["kind"] == "general_backbone")
    n_spec = Counter(c["region"] for c in candidates if c["kind"] == "specialist")
    policy_ok = (
        n_general <= MAX_GENERAL_BACKBONES
        and all(v <= MAX_SPECIALISTS_PER_REGION for v in n_spec.values())
        and all(c["kind"] in ALLOWED_KINDS for c in candidates)
    )

    # final alias: overlap-clean + license-clean eligible set
    eligible_set = [c["candidate_id"] for c in candidates if c["eligible"] and c["license_clean"]]
    final_alias = {
        "alias": "from_scratch_E",
        "eligible_set": eligible_set,
        "reason": "Final alias points to the overlap-clean, license-clean eligible candidate set "
                  "(primary from_scratch_E; supervised_F_to_E if F<->E lineage clean).",
    }

    return {
        "phase": "FM0-A",
        "status": "GENERATED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_run": AUDIT_RUN,
        "config_hash": CONFIG_HASH,
        "candidate_policy": {
            "allowed_kinds": list(ALLOWED_KINDS),
            "max_general_backbones": MAX_GENERAL_BACKBONES,
            "max_specialists_per_region": MAX_SPECIALISTS_PER_REGION,
        },
        "policy_ok": policy_ok,
        "candidates": candidates,
        "final_alias": final_alias,
    }


# ---------------------------------------------------------------------------
# exposure ledger
# ---------------------------------------------------------------------------


def build_ledger(clusters: "OrderedDict[str, dict]") -> list:
    """Per (cluster x checkpoint x audit_run) exposure ledger rows.

    Only project-internal + external-backbone candidates are audited. GSE246381
    is recorded only as an aggregate-safe CLEAN row (no member rows leaked).
    """
    rows = []
    internal = ["from_scratch_E", "supervised_F_to_E"]
    for acc, cluster in clusters.items():
        for cid in internal:
            rows.append({
                "ledger_key": f"{acc}|{cid}|{AUDIT_RUN}",
                "cluster_id": acc,
                "cluster_region": cluster["region"],
                "cluster_e_pairs": cluster["e_pairs"],
                "cluster_f_observations": cluster["f_observations"],
                "checkpoint_id": cid,
                "model_id": None,
                "checkpoint_revision": None,
                "weights_sha256": None,
                "audit_run": AUDIT_RUN,
                "overlap_status": "CLEAN",
                "overlap_type": "none",
                "label_lineage_exposed": False,
                "evidence_grade": "E5",
                "determination_method": "project_internal_no_external_corpus",
                "evidence": "Project-internal candidate; no external pretraining corpus -> no overlap.",
                "eligible": True,
            })
        # external backbone
        o = BACKBONE_OVERLAP.get(acc, {"overlap": "UNKNOWN", "type": "unknown", "note": "no documented membership"})
        rows.append({
            "ledger_key": f"{acc}|{EXTERNAL_BACKBONE['candidate_id']}|{AUDIT_RUN}",
            "cluster_id": acc,
            "cluster_region": cluster["region"],
            "cluster_e_pairs": cluster["e_pairs"],
            "cluster_f_observations": cluster["f_observations"],
            "checkpoint_id": EXTERNAL_BACKBONE["candidate_id"],
            "model_id": EXTERNAL_BACKBONE["model_id"],
            "checkpoint_revision": EXTERNAL_BACKBONE["revision"],
            "weights_sha256": EXTERNAL_BACKBONE["weights_sha256"],
            "audit_run": AUDIT_RUN,
            "overlap_status": o["overlap"],
            "overlap_type": o["type"],
            "label_lineage_exposed": False,
            "evidence_grade": "E4" if o["overlap"] == "DETECTED" else ("E5" if o["overlap"] == "CLEAN" else "UNKNOWN"),
            "determination_method": EXTERNAL_BACKBONE["determination_method"],
            "evidence": o["note"],
            "eligible": o["overlap"] != "DETECTED",
        })
    return rows


# ---------------------------------------------------------------------------
# overlap report
# ---------------------------------------------------------------------------


def build_overlap_report(ledger: list) -> dict:
    detected = [r for r in ledger if r["overlap_status"] == "DETECTED"]
    unknown = [r for r in ledger if r["overlap_status"] == "UNKNOWN"]
    clean = [r for r in ledger if r["overlap_status"] == "CLEAN"]
    return {
        "phase": "FM0-A",
        "audit_run": AUDIT_RUN,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_ledger_rows": len(ledger),
        "counts": {"DETECTED": len(detected), "UNKNOWN": len(unknown), "CLEAN": len(clean)},
        "detected_rows": [{k: r[k] for k in ("ledger_key", "cluster_id", "checkpoint_id", "evidence_grade")}
                          for r in detected],
        "unknown_rows": [{k: r[k] for k in ("ledger_key", "cluster_id", "checkpoint_id", "evidence_grade")}
                         for r in unknown],
        "note": "DETECTED/UNKNOWN only excludes the corresponding checkpoint/claim for that cluster; "
                "it does not delete data globally.",
        "license_corpus_evidence": [
            {
                "checkpoint_id": EXTERNAL_BACKBONE["candidate_id"],
                "model_id": EXTERNAL_BACKBONE["model_id"],
                "revision": EXTERNAL_BACKBONE["revision"],
                "weights_sha256": EXTERNAL_BACKBONE["weights_sha256"],
                "license": EXTERNAL_BACKBONE["license"],
                "corpus_evidence": EXTERNAL_BACKBONE["corpus_evidence"],
            }
        ],
    }


# ---------------------------------------------------------------------------
# GSE246381 aggregate / commitment (ordinary) + restricted mirror
# ---------------------------------------------------------------------------


def build_gse246381_aggregate(ledger: list) -> dict:
    rows = [r for r in ledger if r["cluster_id"] == "GSE246381"]
    return {
        "phase": "FM0-A",
        "dataset": "GSE246381",
        "status": "SEALED_EXTERNAL_FINAL_CANDIDATE",
        "audit_run": AUDIT_RUN,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "aggregate_rows": rows,
        "note": "Aggregate/source-independent information only. No member-level sequence or label data is "
                "emitted to the ordinary workspace. Member rows remain in the restricted sealed mirror.",
        "member_data_emitted_to_ordinary": False,
        "analytic_or_final_counters": {"analytic": 0, "final": 0, "internal_test": 0},
    }


def build_gse246381_commitment(aggregate: dict, restricted_agg: dict) -> dict:
    return {
        "phase": "FM0-A",
        "dataset": "GSE246381",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_run": AUDIT_RUN,
        "aggregate_sha256": sha256_str(jl(aggregate)),
        "restricted_aggregate_sha256": sha256_str(jl(restricted_agg)),
        "commitment": "GSE246381 remains SEALED_EXTERNAL_FINAL_CANDIDATE. FM0-A overlap audited as "
                      "aggregate only; no member rows released to ordinary workspace.",
        "member_data_emitted_to_ordinary": False,
    }


def build_restricted_aggregate() -> dict:
    return {
        "phase": "FM0-A",
        "dataset": "GSE246381",
        "status": "SEALED_EXTERNAL_FINAL_CANDIDATE",
        "audit_run": AUDIT_RUN,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overlap_status": "CLEAN",
        "overlap_type": "none",
        "note": "Restricted aggregate file. Seed/source-only; no member rows written here.",
        "analytic_or_final_counters": {"analytic": 0, "final": 0, "internal_test": 0},
    }


# ---------------------------------------------------------------------------
# FM0 effective exposure projection (versioned)
# ---------------------------------------------------------------------------


def build_fm0_effective_exposure(seq_file: Path, out: Path) -> int:
    """Write a versioned FM0 EffectiveExposureProjection for every sequence entity.

    Effective exposure at FM0-A = AWAITING_B0_GLOBAL_DISPOSITION (unchanged from
    D1; FM0-A does not change disposition). Each row is hash-bound.
    """
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for row in iter_jsonl(seq_file):
            oid = row.get("sequence_id")
            if not oid:
                continue
            proj = {
                "object_id": oid,
                "effective_exposure": "AWAITING_B0_GLOBAL_DISPOSITION",
                "projection_sha256": sha256_str(oid),
                "chain_root_sha256": None,
            }
            fh.write(jl(proj) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-file", required=True, help="path to sequence_entities.jsonl")
    ap.add_argument("--pair-file", required=True, help="path to utr_edit_pairs.jsonl")
    ap.add_argument("--obs-file", required=True, help="path to functional_observations.jsonl")
    ap.add_argument("--out-dir", required=True, help="ordinary FM0 output dir")
    ap.add_argument("--restricted-dir", required=True, help="restricted GSE246381 run dir")
    args = ap.parse_args()

    seq_file = Path(args.seq_file)
    pair_file = Path(args.pair_file)
    obs_file = Path(args.obs_file)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rdir = Path(args.restricted_dir)
    rdir.mkdir(parents=True, exist_ok=True)

    clusters = build_clusters(seq_file, pair_file, obs_file)
    candidates = build_candidates(clusters)
    ledger = build_ledger(clusters)
    overlap_report = build_overlap_report(ledger)
    agg = build_gse246381_aggregate([r for r in ledger if r["cluster_id"] == "GSE246381"])
    r_agg = build_restricted_aggregate()
    commitment = build_gse246381_commitment(agg, r_agg)

    # ordinary outputs
    (out / "foundation_candidates.json").write_text(
        json.dumps(candidates, indent=2, sort_keys=True), encoding="utf-8")
    with open(out / "foundation_exposure_ledger.jsonl", "w", encoding="utf-8") as fh:
        for row in ledger:
            fh.write(jl(row) + "\n")
    (out / "overlap_report.json").write_text(
        json.dumps(overlap_report, indent=2, sort_keys=True), encoding="utf-8")
    (out / "GSE246381_FM0_AGGREGATE.json").write_text(
        json.dumps(agg, indent=2, sort_keys=True), encoding="utf-8")
    (out / "GSE246381_FM0_COMMITMENT.json").write_text(
        json.dumps(commitment, indent=2, sort_keys=True), encoding="utf-8")

    # versioned FM0 effective exposure projection
    n_proj = build_fm0_effective_exposure(
        seq_file, out / "fm0_effective_exposure_projection.jsonl")

    # restricted mirror: FM0 aggregate + append access log event
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "FM0_AGGREGATE.json").write_text(
        json.dumps(r_agg, indent=2, sort_keys=True), encoding="utf-8")
    # append-only access log (hash-chained)
    access_log = rdir / "ACCESS_LOG.jsonl"
    prev = None
    last_sha = None
    if access_log.exists():
        for line in access_log.open("r", encoding="utf-8"):
            if line.strip():
                ev = json.loads(line)
                last_sha = ev.get("event_sha256")
    event = {
        "access_id": f"gse246381_fm0a_{_next_access_seq(rdir)}",
        "object_id": "GSE246381_FM0_AUDIT",
        "intent": "restricted_fm0a_aggregate_audit",
        "status": "COMPLETION",
        "prev_event_sha256": last_sha,
    }
    clean = {k: v for k, v in event.items() if k != "event_sha256"}
    event["event_sha256"] = sha256_str(jl(clean))
    with open(access_log, "a", encoding="utf-8") as fh:
        fh.write(jl(event) + "\n")

    print(json.dumps({
        "clusters": {k: v for k, v in clusters.items()},
        "candidates": candidates["final_alias"],
        "ledger_rows": len(ledger),
        "overlap_counts": overlap_report["counts"],
        "projection_rows": n_proj,
        "policy_ok": candidates["policy_ok"],
        "status": "DONE",
    }, indent=2, sort_keys=True))


def _next_access_seq(rdir: Path) -> int:
    access_log = rdir / "ACCESS_LOG.jsonl"
    n = 0
    if access_log.exists():
        for line in access_log.open("r", encoding="utf-8"):
            if line.strip():
                n += 1
    return n


if __name__ == "__main__":
    main()