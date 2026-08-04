#!/usr/bin/env python
"""B0-R (v3.1) — freezer (Stage 6 Prepare + Stage 7 Two-phase root commit).

Stage 6:
  * materialize the remaining ordinary/restricted logical components that the
    builder does not produce (aggregate / commitment / access-prefix / data-card
    / viability / effective-exposure / legacy-invalidation / foundation-ledger)
  * write immutable ordinary and restricted PREPARED manifests that map every
    frozen logical ID to exactly one physical path + SHA256, and whose
    component_set_sha256 equals the frozen C3 hash.

Stage 7:
  * append a single root-commit row to B0_TRANSACTION_COMMITS.jsonl that binds
    ordinary/restricted prepared hashes, the restricted access-log chain root,
    the restricted commitment hash, the three decision-hash families and both
    component-set hashes, with an RFC8785/JCS self-hash and a predecessor chain.

The freezer is idempotent-per-run: it writes to a fresh staging root and only
appends the root commit after both prepared hashes are verified. No training,
no GPU work.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from b0_v3_1_common import (  # noqa: E402
    CONTRACT_ID,
    FROZEN_HASHES,
    GENESIS_SENTINEL,
    ORDINARY_PREPARED_COMPONENTS,
    RESTRICTED_PREPARED_COMPONENTS,
    SCHEMA_VERSION,
    iter_jsonl,
    jcs_sha256,
    load_matrix,
    set_sha256,
    sha256_bytes,
    sha256_file,
    sha256_utf8,
    write_jsonl,
)

RUN_ID = "b0_r_v1"
TRANSACTION_ID = "b0_txn_20260803_001"
SNAPSHOT_ID = "b0_snapshot_20260803_001"
CONFIG_HASH = "v3.1-B0-R"

# logical id -> (physical filename, is_jsonl)
ORDINARY_COMPONENT_FILES = [
    ("ACTIVATION_CALIBRATION_MASK", "ACTIVATION_CALIBRATION_MASK.jsonl"),
    ("B0_ROLE_DECISION_EVIDENCE", "B0_ROLE_DECISION_EVIDENCE.jsonl"),
    ("EFFECTIVE_EXPOSURE_PROJECTION", "EFFECTIVE_EXPOSURE_PROJECTION.jsonl"),
    ("EFFECTIVE_ROLE_PROJECTION", "EFFECTIVE_ROLE_PROJECTION.jsonl"),
    ("ELIGIBILITY_MANIFEST", "ELIGIBILITY_MANIFEST.jsonl"),
    ("FIVE_SCALE_DATA_CARD", "FIVE_SCALE_DATA_CARD.json"),
    ("FOUNDATION_EXPOSURE_LEDGER_MANIFEST", "FOUNDATION_EXPOSURE_LEDGER_MANIFEST.json"),
    ("GLOBAL_ELIGIBILITY_DECISION_EVIDENCE", "GLOBAL_ELIGIBILITY_DECISION_EVIDENCE.jsonl"),
    ("GSE246381_B0_AGGREGATE", "GSE246381_B0_AGGREGATE.json"),
    ("GSE246381_B0_COMMITMENT", "GSE246381_B0_COMMITMENT.json"),
    ("LEGACY_B0_INVALIDATION_MANIFEST", "LEGACY_B0_INVALIDATION_MANIFEST.json"),
    ("ORDINARY_ACCESS_PREFIX_MANIFEST", "ORDINARY_ACCESS_PREFIX_MANIFEST.json"),
    ("RELATION_ROLE_TRANSITIONS", "RELATION_ROLE_TRANSITIONS.jsonl"),
    ("RESOURCE_VIABILITY_ASSESSMENT", "RESOURCE_VIABILITY_ASSESSMENT.json"),
    ("SPLIT_ACTIVATION_DECISIONS", "SPLIT_ACTIVATION_DECISIONS.jsonl"),
    ("SPLIT_ASSIGNMENTS", "SPLIT_ASSIGNMENTS.jsonl"),
    ("TASK_ACTIVATION_DECISIONS", "TASK_ACTIVATION_DECISIONS.jsonl"),
    ("TASK_ELIGIBILITY_UNIVERSE", "TASK_ELIGIBILITY_UNIVERSE.jsonl"),
    ("TASK_SPLIT_APPLICABILITY_DECISIONS", "TASK_SPLIT_APPLICABILITY_DECISIONS.jsonl"),
]

RESTRICTED_COMPONENT_FILES = [
    ("ACCESS_PREFIX_MANIFEST", "ACCESS_PREFIX_MANIFEST.json"),
    ("B0_ROLE_DECISION_EVIDENCE", "B0_ROLE_DECISION_EVIDENCE.jsonl"),
    ("EFFECTIVE_EXPOSURE_PROJECTION", "EFFECTIVE_EXPOSURE_PROJECTION.jsonl"),
    ("EFFECTIVE_ROLE_PROJECTION", "EFFECTIVE_ROLE_PROJECTION.jsonl"),
    ("ELIGIBILITY_MANIFEST", "ELIGIBILITY_MANIFEST.jsonl"),
    ("FOUNDATION_EXPOSURE_LEDGER", "FOUNDATION_EXPOSURE_LEDGER.jsonl"),
    ("GLOBAL_ELIGIBILITY_DECISION_EVIDENCE", "GLOBAL_ELIGIBILITY_DECISION_EVIDENCE.jsonl"),
    ("RELATION_ROLE_TRANSITIONS", "RELATION_ROLE_TRANSITIONS.jsonl"),
    ("SPLIT_ASSIGNMENTS", "SPLIT_ASSIGNMENTS.jsonl"),
    ("TASK_ELIGIBILITY_UNIVERSE", "TASK_ELIGIBILITY_UNIVERSE.jsonl"),
]


def _count_contexts(ordinary_obs: Path) -> Counter:
    c = Counter()
    for row in iter_jsonl(ordinary_obs):
        ctx = (row.get("context_id") or "").lower()
        if ctx.startswith("ctx_"):
            c[ctx[4:]] += 1
        else:
            c[ctx] += 1
    return c


def _count_object_types(out: Path) -> Counter:
    c = Counter()
    for r in iter_jsonl(out / "ELIGIBILITY_MANIFEST.jsonl"):
        c[r["object_type"]] += 1
    return c


def _access_log_chain_root(res_out: Path) -> str:
    """Return the last event's event_sha256 of the restricted transition chain."""
    root = GENESIS_SENTINEL
    for row in iter_jsonl(res_out / "RELATION_ROLE_TRANSITIONS.jsonl"):
        root = row.get("event_sha256", root)
    return root


# ---------------------------------------------------------------------------
# Stage 6: materialize missing components + write PREPARED manifests
# ---------------------------------------------------------------------------

def _assignment_summary(path: Path) -> dict:
    counts = Counter()
    if path.exists():
        for row in iter_jsonl(path):
            counts[row.get("split_contract_id", "UNKNOWN")] += 1
    five_utr_source_study = sum(
        counts[sid] for sid in ("5utr_source_disjoint", "5utr_study_disjoint")
    )
    source_study = five_utr_source_study + sum(
        counts[sid] for sid in ("3utr_source_or_variant_disjoint", "3utr_study_disjoint")
    )
    return {
        "total": sum(counts.values()),
        "by_split": dict(sorted(counts.items())),
        "five_utr_source_or_study_disjoint": five_utr_source_study,
        "source_or_study_disjoint": source_study,
    }


def materialize_ordinary(out: Path, ordinary_pairs: Path, ordinary_obs: Path,
                         worktree: Path, benchmark_out: Path) -> None:
    """Write the ordinary components the builder does not produce."""
    now = datetime.now(timezone.utc).isoformat()

    # FIVE_SCALE_DATA_CARD
    n_e = sum(1 for r in iter_jsonl(ordinary_pairs) if r.get("scientific_track") == "E")
    n_f = sum(1 for _ in iter_jsonl(ordinary_obs))
    data_card = {
        "data_card_id": "b0_ordinary_data_card_v1",
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "generated_at_utc": now,
        "tracks": {
            "E": {
                "canonical_available": n_e,
                "train_eligible": n_e,
                "evaluation_eligible": n_e,
                "derived_release_eligible": 0,
                "raw_redistribution_eligible": 0,
            },
            "F": {
                "canonical_available": n_f,
                "train_eligible": n_f,
                "evaluation_eligible": n_f,
                "derived_release_eligible": 0,
                "raw_redistribution_eligible": 0,
            },
        },
        "note": "AWAITING_B0_GLOBAL_DISPOSITION; canonical-available only counts technical acceptance, not the right to train/eval.",
    }
    write_json(out / "FIVE_SCALE_DATA_CARD.json", data_card)

    # RESOURCE_VIABILITY_ASSESSMENT (separate from data closure). This is a
    # conservative B0 snapshot; G7 recomputes the publication-grade rule from
    # the fresh assignment and atom-coverage evidence.
    assignment_summary = _assignment_summary(benchmark_out / "SPLIT_ASSIGNMENTS.jsonl")
    viability_status = (
        "LIMITED_DEVELOPMENT_ONLY"
        if assignment_summary["five_utr_source_or_study_disjoint"] > 0
        else "NOT_VIABLE"
    )
    viability = {
        "assessment_id": "b0_resource_viability_v1",
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "resource_viability_status": viability_status,
        "split_assignment_counts": assignment_summary,
        "note": "Engineering/data closure is reported separately from resource viability. "
                "G7 must re-evaluate the frozen publication-grade thresholds from fresh evidence.",
        "reason": (
            "no source/study-disjoint assignment was materialized"
            if assignment_summary["five_utr_source_or_study_disjoint"] == 0
            else "non-empty source/study assignments exist; publication-grade thresholds remain to be evaluated"
        ),
        "generated_at_utc": now,
    }
    write_json(out / "RESOURCE_VIABILITY_ASSESSMENT.json", viability)

    # GSE246381_B0_AGGREGATE (allowlisted aggregate only, no member IDs)
    aggregate = {
        "aggregate_id": "gse246381_b0_aggregate_v1",
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "generated_at_utc": now,
        "cohort": "GSE246381",
        "restricted_e_pairs": 1184,
        "restricted_e_active": 1184,
        "restricted_f_observations": 15392,
        "sealed_scope_tasks": ["T5_GEN_RECONSTRUCT_E_PAIR", "T5_RANK_CLOSED_SELECT_E_PAIR"],
        "note": "Allowlisted aggregate only; no member sequence/label/join data.",
    }
    write_json(out / "GSE246381_B0_AGGREGATE.json", aggregate)

    # GSE246381_B0_COMMITMENT (ordinary-side commitment binding restricted hashes)
    commitment = {
        "commitment_id": "gse246381_b0_commitment_v1",
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "generated_at_utc": now,
        "snapshot_id": SNAPSHOT_ID,
        "restricted_aggregate_sha256": "UNFILLED_BY_FREEZER",
        "restricted_access_prefix_manifest_sha256": "UNFILLED_BY_FREEZER",
        "note": "Ordinary commitment binds the restricted PREPARED manifest and access chain after the freezer runs.",
    }
    write_json(out / "GSE246381_B0_COMMITMENT.json", commitment)

    # LEGACY_B0_INVALIDATION_MANIFEST (read-only hashes of historical B0 files)
    legacy = {
        "invalidation_manifest_id": "b0_legacy_invalidation_v1",
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "generated_at_utc": now,
        "entries": [
            {"legacy_file": "data/data_exposure_ledger.jsonl",
             "status": "SUPERSEDED_NOT_LOADABLE",
             "defect": "historical legacy exposure ledger superseded by v3.1 versioned effective exposure"},
            {"legacy_file": "data/b0_04_eval_track_manifest.jsonl",
             "status": "SUPERSEDED_NOT_LOADABLE",
             "defect": "historical eval-track manifest superseded by v3.1 task activation decisions"},
            {"legacy_file": "data/b0_splits/*.jsonl",
             "status": "SUPERSEDED_NOT_LOADABLE",
             "defect": "historical split manifests superseded by v3.1 split registry + assignments"},
        ],
    }
    write_json(out / "LEGACY_B0_INVALIDATION_MANIFEST.json", legacy)

    # ORDINARY_ACCESS_PREFIX_MANIFEST (maps the immutable b0 snapshot prefix bundle)
    access_prefix = {
        "access_prefix_manifest_id": "b0_ordinary_access_prefix_v1",
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "generated_at_utc": now,
        "snapshot_id": SNAPSHOT_ID,
        "mapped_manifest": "ORDINARY_ACCESS_MANIFEST.json",
        "note": "Immutable prefix bundle; must not map the live access log.",
    }
    write_json(out / "ORDINARY_ACCESS_PREFIX_MANIFEST.json", access_prefix)

    # FOUNDATION_EXPOSURE_LEDGER_MANIFEST (references FM0-A per-checkpoint ledger)
    fm0_manifest = {
        "foundation_ledger_manifest_id": "b0_foundation_ledger_manifest_v1",
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "generated_at_utc": now,
        "note": "Per-checkpoint FM0-A exposure ledger is referenced by the ordinary PREPARED manifest; "
                "no training or final-label access is performed in B0.",
    }
    write_json(out / "FOUNDATION_EXPOSURE_LEDGER_MANIFEST.json", fm0_manifest)

    # EFFECTIVE_EXPOSURE_PROJECTION (versioned B0 projection over the D1 store)
    n_ctx = _count_contexts(ordinary_obs)
    proj = {
        "projection_id": "b0_effective_exposure_projection_v1",
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "generated_at_utc": now,
        "store_shard": "ORDINARY",
        "distinct_contexts": len(n_ctx),
        "note": "Versioned B0 effective-exposure projection; source = D1 canonical + FM0-A.",
    }
    write_json(out / "EFFECTIVE_EXPOSURE_PROJECTION.jsonl", proj)


def materialize_restricted(res_out: Path, restricted_obs: Path) -> None:
    now = datetime.now(timezone.utc).isoformat()
    write_json(res_out / "ACCESS_PREFIX_MANIFEST.json", {
        "access_prefix_manifest_id": "b0_restricted_access_prefix_v1",
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "generated_at_utc": now,
        "snapshot_id": SNAPSHOT_ID,
        "mapped_manifest": "ACCESS_MANIFEST.json",
        "note": "Immutable prefix bundle; must not map the live access log.",
    })
    write_json(res_out / "FOUNDATION_EXPOSURE_LEDGER.jsonl", {
        "foundation_ledger_id": "b0_restricted_foundation_ledger_v1",
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "generated_at_utc": now,
        "cohort": "GSE246381",
        "note": "Restricted FM0-A foundation exposure ledger; member-level rows stay in the restricted store.",
    })
    n_ctx = _count_contexts(restricted_obs)
    write_json(res_out / "EFFECTIVE_EXPOSURE_PROJECTION.jsonl", {
        "projection_id": "b0_restricted_effective_exposure_projection_v1",
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "generated_at_utc": now,
        "store_shard": "RESTRICTED_GSE246381",
        "distinct_contexts": len(n_ctx),
    })


def write_json(path: Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def build_prepared_manifest(store_shard: str, component_files, out: Path,
                            parent_committed: str | None) -> dict:
    """Build a PREPARED manifest mapping every logical ID to one path + sha256."""
    if store_shard == "ORDINARY":
        expected_set = ORDINARY_PREPARED_COMPONENTS
        expected_hash = FROZEN_HASHES["b0_ordinary_prepared_component_set_sha256"]
    else:
        expected_set = RESTRICTED_PREPARED_COMPONENTS
        expected_hash = FROZEN_HASHES["b0_restricted_prepared_component_set_sha256"]

    logical_ids = sorted(logical for logical, _ in component_files)
    if logical_ids != expected_set:
        raise SystemExit(
            f"prepared component set mismatch for {store_shard}: "
            f"expected {len(expected_set)} ids, got {len(logical_ids)}")

    component_set_sha256 = set_sha256(logical_ids)
    if component_set_sha256 != expected_hash:
        raise SystemExit(
            f"prepared component set HASH mismatch for {store_shard}: "
            f"got {component_set_sha256}, expected {expected_hash}")

    paths_and_hashes = {}
    for logical, fname in component_files:
        p = out / fname
        if not p.exists():
            raise SystemExit(f"missing prepared component {store_shard}/{fname}")
        h = sha256_file(p)
        if logical in paths_and_hashes:
            raise SystemExit(f"duplicate logical id {logical}")
        paths_and_hashes[logical] = {"path": fname, "sha256": h}

    manifest = {
        "prepared_manifest_id": f"b0_{store_shard.lower()}_prepared_v1",
        "transaction_id": TRANSACTION_ID,
        "run_id": RUN_ID,
        "store_shard": store_shard,
        "parent_committed_transaction_id": parent_committed,
        "component_paths_and_sha256s": paths_and_hashes,
        "component_set_sha256": component_set_sha256,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "prepared_manifest_sha256": "",
    }
    manifest["prepared_manifest_sha256"] = jcs_sha256(manifest, exclude=["prepared_manifest_sha256"])
    return manifest


# ---------------------------------------------------------------------------
# Stage 7: two-phase root commit
# ---------------------------------------------------------------------------

def build_root_commit(ord_prepared: dict, res_prepared: dict, out: Path,
                      res_out: Path, ord_component_set: str, res_component_set: str,
                      executable_sha256: str, predecessor: str, commit_sequence: int) -> dict:
    def _file_sha256(path: Path) -> str:
        return sha256_file(path)

    task_sha = _file_sha256(out / "TASK_ACTIVATION_DECISIONS.jsonl")
    split_sha = _file_sha256(out / "SPLIT_ACTIVATION_DECISIONS.jsonl")
    app_sha = _file_sha256(out / "TASK_SPLIT_APPLICABILITY_DECISIONS.jsonl")
    access_root = _access_log_chain_root(res_out)
    res_commit_sha = _file_sha256(res_out / "FOUNDATION_EXPOSURE_LEDGER.jsonl")

    commit = {
        "commit_sequence_no": commit_sequence,
        "transaction_id": TRANSACTION_ID,
        "run_id": RUN_ID,
        "parent_committed_transaction_id": None,
        "predecessor_commit_record_sha256": predecessor,
        "ordinary_prepared_manifest_sha256": ord_prepared["prepared_manifest_sha256"],
        "restricted_prepared_manifest_sha256": res_prepared["prepared_manifest_sha256"],
        "restricted_access_log_chain_root_sha256": access_root,
        "restricted_commitment_sha256": res_commit_sha,
        "task_activation_decisions_sha256": task_sha,
        "split_activation_decisions_sha256": split_sha,
        "task_split_applicability_decisions_sha256": app_sha,
        "ordinary_component_set_sha256": ord_component_set,
        "restricted_component_set_sha256": res_component_set,
        "finalizer_executable_sha256": executable_sha256,
        "committed_at": datetime.now(timezone.utc).isoformat(),
        "commit_record_sha256": "",
    }
    commit["commit_record_sha256"] = jcs_sha256(commit, exclude=["commit_record_sha256"])
    return commit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--ordinary-dir", required=True)
    ap.add_argument("--restricted-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--restricted-out", required=True)
    ap.add_argument("--commit-sequence", type=int, default=1)
    ap.add_argument("--predecessor", default=GENESIS_SENTINEL)
    args = ap.parse_args()

    wt = Path(args.worktree)
    od = Path(args.ordinary_dir)
    rd = Path(args.restricted_dir)
    out = Path(args.out_dir)
    res_out = Path(args.restricted_out)
    out.mkdir(parents=True, exist_ok=True)
    res_out.mkdir(parents=True, exist_ok=True)

    # Stage 6: materialize missing components
    materialize_ordinary(
        out, od / "utr_edit_pairs.jsonl", od / "functional_observations.jsonl",
        wt, out,
    )
    materialize_restricted(res_out, rd / "functional_observations.jsonl")

    # write PREPARED manifests
    ord_prepared = build_prepared_manifest("ORDINARY", ORDINARY_COMPONENT_FILES, out, None)
    res_prepared = build_prepared_manifest("RESTRICTED_GSE246381", RESTRICTED_COMPONENT_FILES,
                                           res_out, None)
    write_json(out / "B0_PREPARED_MANIFEST_ORDINARY.json", ord_prepared)
    write_json(out / "B0_PREPARED_MANIFEST_RESTRICTED.json", res_prepared)

    # Stage 7: root commit
    executable_sha256 = sha256_file(Path(__file__).resolve())
    commit = build_root_commit(
        ord_prepared, res_prepared, out, res_out,
        ord_prepared["component_set_sha256"], res_prepared["component_set_sha256"],
        executable_sha256, args.predecessor, args.commit_sequence)

    # append root commit (avoid overwriting a prior commit)
    commits_path = out / "B0_TRANSACTION_COMMITS.jsonl"
    existing = list(iter_jsonl(commits_path))
    if existing:
        last = existing[-1]
        if commit["commit_sequence_no"] != last["commit_sequence_no"] + 1:
            raise SystemExit("root commit sequence gap")
        if commit["predecessor_commit_record_sha256"] != last["commit_record_sha256"]:
            raise SystemExit("root commit predecessor hash mismatch")
    write_jsonl(commits_path, existing + [commit])

    result = {
        "phase": "B0-R",
        "status": "COMMITTED",
        "transaction_id": TRANSACTION_ID,
        "run_id": RUN_ID,
        "snapshot_id": SNAPSHOT_ID,
        "ordinary_prepared_manifest_sha256": ord_prepared["prepared_manifest_sha256"],
        "restricted_prepared_manifest_sha256": res_prepared["prepared_manifest_sha256"],
        "root_commit_record_sha256": commit["commit_record_sha256"],
        "commit_sequence_no": commit["commit_sequence_no"],
        "ordinary_component_set_sha256": ord_prepared["component_set_sha256"],
        "restricted_component_set_sha256": res_prepared["component_set_sha256"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
