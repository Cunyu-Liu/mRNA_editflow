"""Small but structurally complete governance documents for validator tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.execution.acceptance_semantics import (
    B0_CLAIM_BOUNDARY,
    B0_FOUNDATION_STATE,
    B0_LEAKAGE_GATE_KEYS,
    B0_LEAKAGE_ZERO_COUNTS,
    B0_REQUIRED_PARTITION_IDS,
    D1_ACCEPTED_DATASETS,
    D1_BLOCKED_DATASETS,
    D1_REQUIRED_ARTIFACT_PATHS,
    D1_REQUIRED_SUPPORTED_DATASETS,
    D1_SCOPE_DATASETS,
)


def _d1_provenance_check(dataset_id: str) -> dict:
    is_accepted = dataset_id in D1_ACCEPTED_DATASETS
    raw_files = (
        [
            {
                "bytes": len(dataset_id),
                "defaults": {},
                "delimiter": None,
                "format": "tsv",
                "path": f"/production/inputs/{dataset_id}/input.tsv",
                "role": "sequence_and_provided_label_input",
                "sha256": hashlib.sha256(dataset_id.encode("utf-8")).hexdigest(),
                "sheet_name": None,
            }
        ]
        if is_accepted
        else []
    )
    return {
        "name": "production_input_provenance_complete",
        "passed": True,
        "detail": {
            "accepted_provenance_passed": is_accepted,
            "audit": {
                "complete": is_accepted,
                "download_manifest_complete": True,
                "license_complete": True,
                "raw_files": raw_files,
                "raw_files_complete": is_accepted,
            },
            "blocked_or_excluded": not is_accepted,
            "fixture_exemption": False,
            "integrity_failures": [],
            "metadata_only_provenance_passed": not is_accepted,
        },
    }


def valid_d1_acceptance(stage_root: Path) -> dict:
    stage_root.mkdir(parents=True, exist_ok=True)
    build_manifest_path = stage_root / "build_manifest.json"
    build_manifest_path.write_text(
        '{"schema_version":"d1_build_manifest_v2"}\n',
        encoding="utf-8",
    )
    return {
        "schema_version": "d1_acceptance_v2",
        "generated_at_utc": "2026-07-29T00:30:00+00:00",
        "stage_d1_root": str(stage_root.resolve()),
        "fixture_mode": False,
        "evidence_level": "production_reconstruction",
        "dataset_results": [
            {
                "dataset_id": dataset_id,
                "status": (
                    "accepted" if dataset_id in D1_ACCEPTED_DATASETS else "blocked"
                ),
                "passed": True,
                "paper_eligible": dataset_id in D1_ACCEPTED_DATASETS,
                "fixture_mode": False,
                "checks": [
                    {"name": "complete", "passed": True},
                    _d1_provenance_check(dataset_id),
                ],
                "counts": {"labels": 1 if dataset_id in D1_ACCEPTED_DATASETS else 0},
            }
            for dataset_id in sorted(D1_SCOPE_DATASETS)
        ],
        "required_supported_datasets": sorted(D1_REQUIRED_SUPPORTED_DATASETS),
        "missing_required_datasets": [],
        "expected_d1_scope_datasets": sorted(D1_SCOPE_DATASETS),
        "missing_d1_scope_datasets": [],
        "structural_validation_passed": True,
        "required_artifact_validation": {
            "passed": True,
            "semantic_checks": {
                "all_path_bytes_sha_bindings_match": True,
                "all_required_content_checks_pass": True,
            },
            "content_checks": {
                path: True for path in sorted(D1_REQUIRED_ARTIFACT_PATHS)
            },
            "binding_checks": {
                path: True for path in sorted(D1_REQUIRED_ARTIFACT_PATHS)
            },
            "artifacts": {
                path: {
                    "exists": True,
                    "bytes": index,
                    "sha256": f"{index:x}" * 64,
                }
                for index, path in enumerate(
                    sorted(D1_REQUIRED_ARTIFACT_PATHS), start=1
                )
            },
            "build_manifest": {
                "path": str(build_manifest_path.resolve()),
                "bytes": build_manifest_path.stat().st_size,
                "sha256": hashlib.sha256(build_manifest_path.read_bytes()).hexdigest(),
            },
        },
        "builder_audit_validation": {
            "passed": True,
            "checks": {"audited_builder_causal_chain_exact": True},
        },
        "global_store_validation": {
            "passed": True,
            "checks": {"global_store_integrity": True},
        },
        "config_binding_validation": {
            "passed": True,
            "checks": {"config_binding_exact": True},
        },
        "dataset_manifest_binding_validation": {
            "passed": True,
            "checks": {"dataset_manifest_bindings_exact": True},
        },
        "phase_gate_passed": True,
        "scientific_result_claimed": False,
        "note": "Structural D1 qualification only; not scientific efficacy.",
    }


assert D1_ACCEPTED_DATASETS == D1_REQUIRED_SUPPORTED_DATASETS
assert D1_ACCEPTED_DATASETS | D1_BLOCKED_DATASETS == D1_SCOPE_DATASETS
assert not D1_ACCEPTED_DATASETS & D1_BLOCKED_DATASETS


def _b0_acceptance_base() -> dict:
    eligible_track_ids = ["record-1"]
    eligible_track_ids_sha256 = hashlib.sha256(b"record-1\n").hexdigest()
    universe_binding = {
        "canonical_records_sha256": "a" * 64,
        "structural_records_sha256": "b" * 64,
        "record_ids_sha256": eligible_track_ids_sha256,
        "record_count": len(eligible_track_ids),
        "candidate_ids_sha256": "d" * 64,
        "candidate_count": 12,
        "task_ids_sha256": "e" * 64,
        "task_count": 12,
        "source_ids_sha256": "f" * 64,
        "source_count": 12,
    }
    required_artifacts = {
        name: {
            "exists": True,
            "schema_valid": True,
            "sha256": str(index) * 64,
            "bytes": index,
        }
        for index, name in enumerate(
            ("exposure_ledger", "track_role_matrix", "data_card", "claims"),
            start=1,
        )
    }
    return {
        "schema_version": "utr_b0_acceptance.v2",
        "b0_gate_passed": True,
        "failed_gates": [],
        "observed": {
            "leakage_report_count": 5,
            "exposure_ledger_coverage": 1.0,
            "exposure_identity_level": "dataset_id",
            "track_role_ambiguity_count": 0,
            "track_identity_universe_complete": True,
            "track_eligible_identity_universe_complete": True,
            "split_identities": [
                ["source_disjoint", "five_utr", None],
                ["study_disjoint", "five_utr", None],
                ["source_disjoint", "three_utr", None],
                ["study_disjoint", "three_utr", None],
                ["cross_region_transfer", "five_utr", "three_utr"],
            ],
            "split_universe": {
                "canonical_records_sha256": universe_binding[
                    "canonical_records_sha256"
                ],
                "structural_records_sha256": universe_binding[
                    "structural_records_sha256"
                ],
                "canonical_record_ids_sha256": "c" * 64,
                "canonical_record_count": 12,
                "structural_record_ids_sha256": "c" * 64,
                "structural_record_count": 12,
                "structural_content_sha256": "8" * 64,
            },
            "eligible_track_role_universe": eligible_track_ids,
            "eligible_track_role_universe_failures": [],
            "foundation_states": [
                {
                    "status": "UNKNOWN_PENDING_FM0",
                    "foundation_selected": False,
                    "allowed_claim": "NONE",
                    "re_audit_required": True,
                    "gate_passed": True,
                }
                for _ in range(5)
            ],
        },
        "allowed_claim": "NONE",
        "requires_fm0_reaudit": True,
        "re_audit_required_before_foundation_use": True,
        "claim_boundary": (
            "B0 structural split acceptance is not an efficacy or SOTA result"
        ),
        "exposure_ledger": {
            "coverage": 1.0,
            "gate_passed": True,
            "identity_level": "dataset_id",
            "missing_identities": [],
            "missing": [],
            "extra": [],
            "ledger_scope_gate_passed": True,
        },
        "track_role_audit": {
            "schema_version": "utr_track_role_audit.v2",
            "gate_passed": True,
            "track_count": 3,
            "track_role_ambiguity_count": 0,
            "identity_universe_complete": True,
            "eligible_identity_binding_checked": True,
            "eligible_identity_binding_complete": True,
            "issues": [],
            "universe_binding": universe_binding,
            "tracks": [
                {"track_id": "track-a", "track_type": "closed_measured_pool"},
                {"track_id": "track-b", "track_type": "heldout_generative"},
                {"track_id": "track-c", "track_type": "open_legal_generation"},
            ],
            "gse246381_role": (
                "historically_exposed_retrospective_external_stress_test"
            ),
        },
        "track_a_label_seal_audit": {
            "schema_version": "utr_track_a_label_seal_audit.v2",
            "track_id": "track-a",
            "gate_passed": True,
            "candidate_label_bijection": True,
            "record_label_bijection": True,
            "strict_hidden_label_schema_passed": True,
            "paired_finite_measured_labels": True,
            "canonical_identity_binding_passed": True,
            "d1_acceptance_binding_passed": True,
            "label_store_sha256": "1" * 64,
            "freeze_proof_sha256": "2" * 64,
            "selection_freeze_sha256": "3" * 64,
            "role_policy_sha256": "4" * 64,
            "hidden_label_schema_sha256": "5" * 64,
            "d1_acceptance_sha256": "6" * 64,
            "d1_build_manifest_sha256": "7" * 64,
            "candidate_ids_sha256": universe_binding["candidate_ids_sha256"],
            "canonical_records_sha256": universe_binding["canonical_records_sha256"],
            "structural_records_sha256": universe_binding["structural_records_sha256"],
            "record_ids_sha256": universe_binding["record_ids_sha256"],
        },
        "required_artifact_audit": {
            "schema_version": "utr_b0_required_artifact_audit.v2",
            "binding_manifest_path": "/fixture/artifact_bindings.json",
            "binding_manifest_sha256": "9" * 64,
            "gate_passed": True,
            "failures": [],
            "universe_binding": universe_binding,
            "artifacts": required_artifacts,
            "claims": {
                "allowed_claims_present": True,
                "unsupported_capabilities_present": True,
                "foundation_status": "UNKNOWN_PENDING_FM0",
                "allowed_claim": "NONE",
                "requires_fm0_reaudit": True,
                "gse246381_role": (
                    "historically_exposed_retrospective_external_stress_test"
                ),
            },
        },
    }


def valid_b0_acceptance() -> dict:
    """Return the exact 16-field production B0 acceptance document shape."""
    payload = _b0_acceptance_base()
    universe_binding = payload["track_role_audit"]["universe_binding"]
    split_universe = payload["observed"]["split_universe"]
    split_universe["structural_records_bytes"] = 12
    eligible_sha = universe_binding["record_ids_sha256"]
    payload["observed"]["eligible_track_role_universe"] = {
        "record_ids_sha256": eligible_sha,
        "record_count": universe_binding["record_count"],
        "excluded_record_ids_sha256": hashlib.sha256(b"record-2\n").hexdigest(),
        "excluded_record_count": 11,
    }
    payload["observed"]["foundation_states"] = [
        dict(B0_FOUNDATION_STATE) for _ in range(5)
    ]
    payload["claim_boundary"] = B0_CLAIM_BOUNDARY

    payload["exposure_ledger"] = {
        "coverage": 1.0,
        "covered": 4,
        "expected": 4,
        "identity_level": "dataset_id",
        "missing": [],
        "extra": sorted(D1_SCOPE_DATASETS)[4:],
        "required_ledger_identity_count": len(D1_SCOPE_DATASETS),
        "missing_from_required_ledger_scope": [],
        "outside_required_ledger_scope": [],
        "ledger_scope_gate_passed": True,
        "gate_passed": True,
    }

    track = payload["track_role_audit"]
    track.update(
        {
            "task_structural_binding_checked": True,
            "task_structural_binding_complete": True,
            "identity_universes": {},
            "data_card_counts": {},
        }
    )
    seal = payload["track_a_label_seal_audit"]
    seal.update(
        {
            "current_d1_chain_binding_passed": True,
            "role_policy_exact_binding_passed": True,
            "label_store_bytes": 1,
            "candidate_count": universe_binding["candidate_count"],
            "label_record_ids_sha256": eligible_sha,
            "label_record_count": universe_binding["record_count"],
        }
    )

    exposure_path = "/fixture/data/data_exposure_ledger.jsonl"
    required = payload["required_artifact_audit"]
    for index, (name, artifact) in enumerate(
        required["artifacts"].items(),
        start=1,
    ):
        artifact["path"] = (
            exposure_path if name == "exposure_ledger" else f"/fixture/{name}"
        )
        if name == "exposure_ledger":
            artifact["bytes"] = 1
            artifact["sha256"] = "1" * 64
    required["claims"]["schema_valid"] = True
    d1_binding = {
        "schema_version": "utr_b0_d1_exposure_binding.v2",
        "gate_passed": True,
        "failures": [],
        "d1_acceptance_path": "/fixture/D1/acceptance.json",
        "d1_acceptance_sha256": seal["d1_acceptance_sha256"],
        "d1_build_manifest_path": "/fixture/D1/build_manifest.json",
        "d1_build_manifest_sha256": seal["d1_build_manifest_sha256"],
        "exposure_ledger_path": exposure_path,
        "exposure_ledger_bytes": required["artifacts"]["exposure_ledger"]["bytes"],
        "exposure_ledger_sha256": required["artifacts"]["exposure_ledger"]["sha256"],
        "ledger_semantics_valid": True,
    }

    leakage_gates = {name: True for name in B0_LEAKAGE_GATE_KEYS}
    leakage_counts = {name: 0 for name in B0_LEAKAGE_ZERO_COUNTS.values()}
    leakage_counts.update(
        {
            "metadata_overlap_count": 0,
            "explained_metadata_overlap_count": 0,
            "unexplained_metadata_overlap_count": 0,
            "record_role_overlap_count": 0,
            "component_role_overlap_count": 0,
            "frozen_universe_issue_count": 0,
        }
    )
    reports = []
    for report_index, (
        identity,
        partition_ids,
    ) in enumerate(B0_REQUIRED_PARTITION_IDS.items(), start=1):
        split_kind, left_region, right_region = identity
        region = None if split_kind == "cross_region_transfer" else left_region
        source_region = left_region if split_kind == "cross_region_transfer" else None
        target_region = right_region if split_kind == "cross_region_transfer" else None
        partitions = []
        for partition_index, partition_id in enumerate(
            sorted(partition_ids),
            start=1,
        ):
            heldout_study = (
                partition_id.split(":", 1)[1]
                if split_kind == "study_disjoint"
                else None
            )
            partitions.append(
                {
                    "schema_version": ("utr_b0_partition_leakage_report.v2"),
                    "partition_id": partition_id,
                    "split_partition_sha256": (
                        f"{report_index * 100 + partition_index:064x}"
                    ),
                    "split_kind": split_kind,
                    "region": region,
                    "source_region": source_region,
                    "target_region": target_region,
                    "heldout_study": heldout_study,
                    "counts": dict(leakage_counts),
                    "acceptance_gates": dict(leakage_gates),
                    "gate_passed": True,
                    "foundation_pretraining_overlap": dict(B0_FOUNDATION_STATE),
                    "metadata_axis_status": {},
                    "required_axis_status": {},
                    "examples": {},
                }
            )
        reports.append(
            {
                "schema_version": "utr_b0_leakage_report.v2",
                "split_kind": split_kind,
                "region": region,
                "source_region": source_region,
                "target_region": target_region,
                "required_partition_ids": sorted(partition_ids),
                "partition_count": len(partitions),
                "partitions": partitions,
                "counts": dict(leakage_counts),
                "acceptance_gates": dict(leakage_gates),
                "gate_passed": True,
                "common_universe_binding": {
                    "full_record_count": split_universe["canonical_record_count"],
                    "full_record_ids_sha256": split_universe[
                        "canonical_record_ids_sha256"
                    ],
                    "full_record_universe_sha256": split_universe[
                        "structural_content_sha256"
                    ],
                },
                "foundation_pretraining_overlap": dict(B0_FOUNDATION_STATE),
                "structural_issues": [],
                "structural_records_path": ("/fixture/D1/structural_records.jsonl"),
                "structural_records_sha256": split_universe[
                    "structural_records_sha256"
                ],
                "structural_records_bytes": split_universe["structural_records_bytes"],
                "canonical_records_path": ("/fixture/D1/canonical_records.jsonl"),
                "canonical_records_sha256": split_universe["canonical_records_sha256"],
                "canonical_record_count": split_universe["canonical_record_count"],
                "canonical_record_ids_sha256": split_universe[
                    "canonical_record_ids_sha256"
                ],
                "structural_record_count": split_universe["structural_record_count"],
                "structural_record_ids_sha256": split_universe[
                    "structural_record_ids_sha256"
                ],
                "structural_content_sha256": split_universe[
                    "structural_content_sha256"
                ],
                "split_manifest_path": (f"/fixture/B0/split-{report_index}.json"),
                "split_manifest_sha256": f"{report_index + 20:064x}",
                "split_manifest_bytes": report_index,
                "foundation_exposure_path": None,
                "foundation_exposure_sha256": None,
                "recomputed_from_bound_structural_records": True,
                "canonical_manifest_exact_recomputation": True,
                "canonical_manifest_core_sha256": (f"{report_index + 30:064x}"),
                "auditor_binding": {
                    "schema_version": "utr_b0_leakage_auditor.v2",
                    "entrypoint_path": ("/fixture/scripts/data/audit_b0_leakage.py"),
                    "entrypoint_sha256": "a" * 64,
                    "canonical_auditor_path": (
                        "/fixture/data/utr_benchmark_v2/leakage.py"
                    ),
                    "canonical_auditor_sha256": "b" * 64,
                },
            }
        )

    payload.update(
        {
            "d1_exposure_ledger_binding": d1_binding,
            "supplied_leakage_reports_match_recomputation": True,
            "recomputed_leakage_reports": reports,
            "supplied_leakage_report_files": [
                {
                    "path": f"/fixture/B0/report-{index}.json",
                    "bytes": index,
                    "sha256": f"{index + 40:064x}",
                }
                for index in range(1, 6)
            ],
        }
    )
    return payload
