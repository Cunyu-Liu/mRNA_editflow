"""Small but structurally complete governance documents for validator tests."""
from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.execution.acceptance_semantics import (
    D1_REQUIRED_ARTIFACT_PATHS,
    D1_REQUIRED_SUPPORTED_DATASETS,
    D1_SCOPE_DATASETS,
)


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
                "status": "accepted",
                "passed": True,
                "paper_eligible": True,
                "fixture_mode": False,
                "checks": [{"name": "complete", "passed": True}],
                "counts": {"labels": 1},
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
                "sha256": hashlib.sha256(
                    build_manifest_path.read_bytes()
                ).hexdigest(),
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


def valid_b0_acceptance() -> dict:
    eligible_track_ids = ["record-1"]
    eligible_track_ids_sha256 = hashlib.sha256(
        b"record-1\n"
    ).hexdigest()
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
            "candidate_ids_sha256": universe_binding[
                "candidate_ids_sha256"
            ],
            "canonical_records_sha256": universe_binding[
                "canonical_records_sha256"
            ],
            "structural_records_sha256": universe_binding[
                "structural_records_sha256"
            ],
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
