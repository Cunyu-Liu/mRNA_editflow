from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from data.utr_benchmark_v2.d1_builder import ACTIVE_DATASET_POLICIES
from data.utr_benchmark_v2.d1_builder import D1_SCOPE_DATASETS
from data.utr_benchmark_v2.split_graph import METADATA_DIMENSIONS
from data.utr_benchmark_v2.split_graph import expected_partition_leakage_contract
from data.utr_benchmark_v2.track_loader import CandidateStoreLabelError
from data.utr_benchmark_v2.track_loader import TrackContractError
from data.utr_benchmark_v2.track_loader import audit_track_roles
from data.utr_benchmark_v2.track_loader import expected_generation_task
from data.utr_benchmark_v2.track_loader import load_track_manifest
from data.utr_benchmark_v2.track_loader import privileged_verify_track_a_label_seal
from scripts.data.validate_b0_acceptance import compute_exposure_coverage
from scripts.data.validate_b0_acceptance import render_canonical_data_card
from scripts.data.validate_b0_acceptance import validate_b0_acceptance
from scripts.data.validate_b0_acceptance import (
    validate_d1_exposure_ledger_binding,
)
from scripts.data.validate_b0_acceptance import validate_required_artifacts


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ids_sha256(values: list[str]) -> str:
    body = ("\n".join(sorted(values)) + "\n") if values else ""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _partition_sha256(payload: dict) -> str:
    frozen = {key: value for key, value in payload.items() if key != "partition_sha256"}
    return hashlib.sha256(
        json.dumps(
            frozen,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_track(
    root: Path,
    *,
    track_id: str,
    track_type: str,
    candidate_id: str,
    extra_candidate: dict | None = None,
    source_id: str | None = None,
    provenance_record_id: str | None = None,
    dataset_id: str | None = None,
) -> Path:
    candidate_path = root / f"{track_id}.jsonl"
    is_open_world = track_type == "open_legal_generation"
    actions = ["INS", "SUB", "DEL", "STOP"] if is_open_world else ["SUB", "STOP"]
    candidate = {
        "schema_version": "generation_task.v2",
        "task_id": f"task-{track_id}",
        "track_id": track_id,
        "region": "five_utr",
        "source_id": source_id or f"source-{track_id}",
        "source_sequence": "ACGU",
        "endpoint": "translation_efficiency",
        "candidate_id": None if is_open_world else candidate_id,
        "candidate_sequence": None if is_open_world else "AGGU",
        "legal_action_types": actions,
        "max_edits": 5 if is_open_world else 1,
        "constraints": {
            "source_conditioned": True,
            "sequence_alphabet": "RNA",
            "allowed_operations": actions,
            "min_length": 1 if is_open_world else 4,
            "max_length": 9 if is_open_world else 4,
        },
        "provenance": {
            "dataset_id": dataset_id or f"dataset-{track_id}",
            "study_id": f"study-{track_id}",
            "record_id": provenance_record_id or f"record-{track_id}",
        },
    }
    candidate.update(extra_candidate or {})
    candidate_path.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "utr_track.v2",
        "track_id": track_id,
        "track_type": track_type,
        "candidate_store": {
            "path": candidate_path.name,
            "sha256": _sha256(candidate_path),
            "bytes": candidate_path.stat().st_size,
        },
        "candidate_store_contains_labels": False,
        "selection_access": {"labels": False},
        "retrospective_external_stress_datasets": (
            ["GSE246381"] if dataset_id == "GSE246381" else []
        ),
        "universe_binding": {
            "canonical_records_sha256": "a" * 64,
            "structural_records_sha256": "b" * 64,
            "record_ids_sha256": _ids_sha256(
                [provenance_record_id or f"record-{track_id}"]
            ),
            "record_count": 1,
            "candidate_ids_sha256": _ids_sha256([candidate_id]),
            "candidate_count": 1,
            "task_ids_sha256": _ids_sha256([f"task-{track_id}"]),
            "task_count": 1,
            "source_ids_sha256": _ids_sha256([source_id or f"source-{track_id}"]),
            "source_count": 1,
        },
    }
    if is_open_world:
        manifest["evaluation_budget_protocol"] = {
            "required_budgets": [1, 3, 5],
            "task_representation": "single_maximum_budget",
            "maximum_budget": 5,
            "report_each_budget_separately": True,
            "silent_budget_reduction_forbidden": True,
        }
    if track_type == "closed_measured_pool":
        manifest["label_store"] = {
            "path": "labels.hidden.jsonl",
            "sha256": "0" * 64,
            "bytes": 0,
            "access": "FROZEN_FINAL_ONLY",
            "candidate_id_field": "candidate_id",
            "candidate_ids_sha256": _ids_sha256([candidate_id]),
            "candidate_count": 1,
            "freeze_proof": {
                "path": "labels.freeze.json",
                "sha256": "0" * 64,
            },
            "schema": {
                "path": "labels.schema.json",
                "sha256": "0" * 64,
                "bytes": 0,
            },
            "selection_freeze": {
                "path": "labels.selection.freeze.json",
                "sha256": "0" * 64,
                "bytes": 0,
            },
        }
    manifest_path = root / f"{track_id}.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8")
    return manifest_path


def _seal_track_universe(paths: list[Path]) -> dict:
    candidates = []
    for path in paths:
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        candidate_path = path.parent / manifest["candidate_store"]["path"]
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidates.append(candidate)
    binding = {
        "canonical_records_sha256": "a" * 64,
        "structural_records_sha256": "b" * 64,
        "record_ids_sha256": _ids_sha256(
            [item["provenance"]["record_id"] for item in candidates]
        ),
        "record_count": len(candidates),
        "candidate_ids_sha256": _ids_sha256(
            [
                item["candidate_id"]
                for item in candidates
                if item["candidate_id"] is not None
            ]
        ),
        "candidate_count": sum(item["candidate_id"] is not None for item in candidates),
        "task_ids_sha256": _ids_sha256([item["task_id"] for item in candidates]),
        "task_count": len(candidates),
        "source_ids_sha256": _ids_sha256(
            list({item["source_id"] for item in candidates})
        ),
        "source_count": len({item["source_id"] for item in candidates}),
    }
    for path in paths:
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        manifest["universe_binding"] = binding
        path.write_text(yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8")
    return binding


def _write_exact_track_bundle(
    root: Path,
) -> tuple[list[Path], list[dict], dict[str, str]]:
    records: list[dict] = []
    role_by_record: dict[str, str] = {}
    paths: list[Path] = []
    for index, track_type in enumerate(
        (
            "closed_measured_pool",
            "heldout_generative",
            "open_legal_generation",
        ),
        start=1,
    ):
        source = ("ACGU" * 4)[: 11 + index]
        candidate = source[:-1] + ("A" if source[-1] != "A" else "C")
        record = {
            "record_id": f"exact-r{index}",
            "candidate_id": f"d1-candidate-{index}",
            "dataset_id": f"dataset-{index}",
            "study_id": f"study-{index}",
            "region": "five_utr" if index < 3 else "three_utr",
            "source_id": f"source-{index}",
            "source_sequence": source,
            "candidate_sequence": candidate,
            "endpoint": f"endpoint-{index}",
            "edit_types": ["SUB"],
            "edit_count": 1,
            "edit_distance": 1,
            "pair_type": "true_wt_mutant",
        }
        task = expected_generation_task(record, track_type=track_type)
        candidate_path = root / f"{track_type}.jsonl"
        candidate_path.write_text(
            json.dumps(task, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": "utr_track.v2",
            "track_id": track_type,
            "track_type": track_type,
            "candidate_store": {
                "path": candidate_path.name,
                "sha256": _sha256(candidate_path),
                "bytes": candidate_path.stat().st_size,
            },
            "candidate_store_contains_labels": False,
            "selection_access": {"labels": False},
            "retrospective_external_stress_datasets": [],
            "universe_binding": {
                "canonical_records_sha256": "a" * 64,
                "structural_records_sha256": "b" * 64,
                "record_ids_sha256": "c" * 64,
                "record_count": 3,
                "candidate_ids_sha256": "d" * 64,
                "candidate_count": 2,
                "task_ids_sha256": "e" * 64,
                "task_count": 3,
                "source_ids_sha256": "f" * 64,
                "source_count": 3,
            },
        }
        if track_type == "closed_measured_pool":
            manifest["label_store"] = {
                "path": "labels.hidden.jsonl",
                "sha256": "0" * 64,
                "bytes": 1,
                "access": "FROZEN_FINAL_ONLY",
                "candidate_id_field": "candidate_id",
                "candidate_ids_sha256": _ids_sha256([str(task["candidate_id"])]),
                "candidate_count": 1,
                "freeze_proof": {
                    "path": "labels.freeze.json",
                    "sha256": "0" * 64,
                },
                "schema": {
                    "path": "labels.schema.json",
                    "sha256": "0" * 64,
                    "bytes": 1,
                },
                "selection_freeze": {
                    "path": "labels.selection.freeze.json",
                    "sha256": "0" * 64,
                    "bytes": 1,
                },
            }
        if track_type == "open_legal_generation":
            manifest["evaluation_budget_protocol"] = {
                "required_budgets": [1, 3, 5],
                "task_representation": "single_maximum_budget",
                "maximum_budget": 5,
                "report_each_budget_separately": True,
                "silent_budget_reduction_forbidden": True,
            }
        manifest_path = root / f"{track_type}.yaml"
        manifest_path.write_text(
            yaml.safe_dump(manifest, sort_keys=True),
            encoding="utf-8",
        )
        paths.append(manifest_path)
        records.append(record)
        role_by_record[record["record_id"]] = track_type
    _seal_track_universe(paths)
    return paths, records, role_by_record


def test_track_loader_verifies_label_free_candidate_store(tmp_path: Path) -> None:
    path = _write_track(
        tmp_path,
        track_id="track-a",
        track_type="closed_measured_pool",
        candidate_id="candidate-a",
    )
    loaded = load_track_manifest(path)
    assert loaded.track_type == "closed_measured_pool"
    assert loaded.candidate_ids == ("candidate-a",)


def test_track_audit_rebuilds_track_b_and_c_fields_from_structural_records(
    tmp_path: Path,
) -> None:
    paths, records, role_by_record = _write_exact_track_bundle(tmp_path)
    clean = audit_track_roles(
        [load_track_manifest(path) for path in paths],
        eligible_records=records,
        expected_role_by_record=role_by_record,
    )
    assert clean["task_structural_binding_complete"] is True
    assert clean["gate_passed"] is True

    track_b_path = next(path for path in paths if path.stem == "heldout_generative")
    manifest = yaml.safe_load(track_b_path.read_text(encoding="utf-8"))
    candidate_path = track_b_path.parent / manifest["candidate_store"]["path"]
    task = json.loads(candidate_path.read_text(encoding="utf-8"))
    task["endpoint"] = "tampered-endpoint"
    candidate_path.write_text(
        json.dumps(task, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["candidate_store"].update(
        {
            "sha256": _sha256(candidate_path),
            "bytes": candidate_path.stat().st_size,
        }
    )
    track_b_path.write_text(
        yaml.safe_dump(manifest, sort_keys=True),
        encoding="utf-8",
    )
    tampered = audit_track_roles(
        [load_track_manifest(path) for path in paths],
        eligible_records=records,
        expected_role_by_record=role_by_record,
    )
    assert tampered["task_structural_binding_complete"] is False
    assert any(
        issue["kind"] == "task_structural_binding_mismatch"
        for issue in tampered["issues"]
    )


def test_track_c_budget_protocol_cannot_silently_reduce_to_one(
    tmp_path: Path,
) -> None:
    paths, _, _ = _write_exact_track_bundle(tmp_path)
    track_c_path = next(path for path in paths if path.stem == "open_legal_generation")
    manifest = yaml.safe_load(track_c_path.read_text(encoding="utf-8"))
    manifest["evaluation_budget_protocol"]["required_budgets"] = [1]
    track_c_path.write_text(
        yaml.safe_dump(manifest, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(TrackContractError, match="budgets 1/3/5"):
        load_track_manifest(track_c_path)


def test_track_loader_rejects_physical_label_fields(tmp_path: Path) -> None:
    path = _write_track(
        tmp_path,
        track_id="track-a",
        track_type="closed_measured_pool",
        candidate_id="candidate-a",
        extra_candidate={"delta_normalized": 2.0},
    )
    with pytest.raises(CandidateStoreLabelError, match="delta_normalized"):
        load_track_manifest(path)


def test_track_role_overlap_is_reported_as_ambiguity(tmp_path: Path) -> None:
    paths = [
        _write_track(
            tmp_path,
            track_id="track-a",
            track_type="closed_measured_pool",
            candidate_id="same-candidate",
        ),
        _write_track(
            tmp_path,
            track_id="track-b",
            track_type="heldout_generative",
            candidate_id="same-candidate",
        ),
        _write_track(
            tmp_path,
            track_id="track-c",
            track_type="open_legal_generation",
            candidate_id="candidate-c",
        ),
    ]
    _seal_track_universe(paths)
    audit = audit_track_roles([load_track_manifest(path) for path in paths])
    assert audit["track_role_ambiguity_count"] >= 1
    assert any(issue["kind"] == "candidate_role_overlap" for issue in audit["issues"])
    assert audit["gate_passed"] is False


def test_track_roles_cannot_hide_record_or_source_overlap_behind_new_ids(
    tmp_path: Path,
) -> None:
    paths = [
        _write_track(
            tmp_path,
            track_id="track-a",
            track_type="closed_measured_pool",
            candidate_id="candidate-a",
            source_id="same-source",
            provenance_record_id="same-record",
        ),
        _write_track(
            tmp_path,
            track_id="track-b",
            track_type="heldout_generative",
            candidate_id="candidate-b",
            source_id="same-source",
            provenance_record_id="same-record",
        ),
        _write_track(
            tmp_path,
            track_id="track-c",
            track_type="open_legal_generation",
            candidate_id="candidate-c",
        ),
    ]
    _seal_track_universe(paths)
    audit = audit_track_roles([load_track_manifest(path) for path in paths])
    kinds = {issue["kind"] for issue in audit["issues"]}
    assert "record_role_overlap" in kinds
    assert "source_role_overlap" in kinds


def test_track_role_audit_binds_complete_canonical_record_and_source_universe(
    tmp_path: Path,
) -> None:
    paths = [
        _write_track(
            tmp_path,
            track_id="track-a",
            track_type="closed_measured_pool",
            candidate_id="candidate-a",
        ),
        _write_track(
            tmp_path,
            track_id="track-b",
            track_type="heldout_generative",
            candidate_id="candidate-b",
        ),
        _write_track(
            tmp_path,
            track_id="track-c",
            track_type="open_legal_generation",
            candidate_id="candidate-c",
        ),
    ]
    _seal_track_universe(paths)
    tracks = [load_track_manifest(path) for path in paths]
    canonical = [
        {
            "record_id": f"record-track-{suffix}",
            "source_id": f"source-track-{suffix}",
        }
        for suffix in ("a", "b", "c")
    ]
    complete = audit_track_roles(tracks, eligible_records=canonical)
    assert complete["eligible_identity_binding_complete"] is True
    assert complete["gate_passed"] is True

    incomplete = audit_track_roles(tracks, eligible_records=canonical[:-1])
    assert incomplete["eligible_identity_binding_complete"] is False
    assert any(
        issue["kind"] == "canonical_record_role_universe_mismatch"
        for issue in incomplete["issues"]
    )


def test_track_manifest_rejects_unsealed_top_level_fields(tmp_path: Path) -> None:
    path = _write_track(
        tmp_path,
        track_id="track-a",
        track_type="closed_measured_pool",
        candidate_id="candidate-a",
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["selection_score"] = 42
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(TrackContractError, match="sealed track schema"):
        load_track_manifest(path)


def _write_real_track_a_seal(manifest_path: Path) -> None:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    candidate_path = manifest_path.parent / manifest["candidate_store"]["path"]
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    canonical = {
        "record_id": candidate["provenance"]["record_id"],
        "candidate_id": "canonical-candidate-a",
        "dataset_id": candidate["provenance"]["dataset_id"],
        "study_id": candidate["provenance"]["study_id"],
        "assay_id": "assay-a",
        "context_id": "context-a",
        "endpoint": candidate["endpoint"],
        "pair_type": "true_wt_mutant",
        "source_id": candidate["source_id"],
        "source_sequence": candidate["source_sequence"],
        "candidate_sequence": candidate["candidate_sequence"],
        "source_value_raw": 1.0,
        "candidate_value_raw": 2.25,
        "delta_raw": 1.25,
        "delta_normalized": 1.25,
        "effect_standard_error": None,
        "replicate_count": 2,
        "label_provenance": {"status": "measured"},
    }
    d1_root = manifest_path.parent / "D1"
    canonical_path = d1_root / "canonical" / "records_with_labels.jsonl"
    structural_path = d1_root / "candidate_store" / "candidates.jsonl"
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    structural_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.write_text(json.dumps(canonical) + "\n", encoding="utf-8")
    structural_path.write_text(
        json.dumps({"record_id": canonical["record_id"]}) + "\n",
        encoding="utf-8",
    )
    canonical_sha = _sha256(canonical_path)
    structural_sha = _sha256(structural_path)
    record_ids_sha = _ids_sha256([canonical["record_id"]])
    manifest["universe_binding"]["canonical_records_sha256"] = canonical_sha
    manifest["universe_binding"]["structural_records_sha256"] = structural_sha
    build_manifest_path = d1_root / "build_manifest.json"
    build_manifest_path.write_text(
        json.dumps(
            {
                "global_stores": {
                    "canonical_label_store": {
                        "path": "canonical/records_with_labels.jsonl",
                        "sha256": canonical_sha,
                        "bytes": canonical_path.stat().st_size,
                        "records": 1,
                        "record_ids_sha256": record_ids_sha,
                    },
                    "sealed_label_free_candidate_store": {
                        "path": "candidate_store/candidates.jsonl",
                        "sha256": structural_sha,
                        "bytes": structural_path.stat().st_size,
                        "records": 1,
                        "record_ids_sha256": record_ids_sha,
                    },
                }
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    acceptance_path = manifest_path.parent / "d1_acceptance.json"
    acceptance_path.write_text(
        json.dumps(
            {
                "phase_gate_passed": True,
                "fixture_mode": False,
                "structural_validation_passed": True,
                "stage_d1_root": str(d1_root.resolve()),
                "global_store_validation": {"passed": True},
                "required_artifact_validation": {"passed": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    role_policy_path = manifest_path.parent / "track_role_policy.yaml"
    role_policy_path.write_text(
        yaml.safe_dump(
            {
                "selection_is_label_independent": True,
                "label_fields_read_for_selection": [],
                "atomic_components_split": False,
                "measured_eligibility": {
                    "effect_value_or_direction_used_for_selection": False
                },
                "track_evidence_counts": {
                    "closed_measured_pool": {
                        "measured_pair_type": 1,
                        "structural_unmeasured_pair_type": 0,
                    }
                },
                "track_b_future_formal_generative_evaluation_allowed": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    schema_source = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "track_a_hidden_label.schema.json"
    )
    schema_path = manifest_path.parent / "labels.schema.json"
    schema_path.write_bytes(schema_source.read_bytes())
    selection_path = manifest_path.parent / "labels.selection.freeze.json"
    selection = {
        "schema_version": "utr_track_a_prelabel_selection_freeze.v2",
        "track_id": manifest["track_id"],
        "candidate_store_path": str(candidate_path.resolve()),
        "candidate_store_sha256": _sha256(candidate_path),
        "candidate_store_bytes": candidate_path.stat().st_size,
        "selected_record_ids_sha256": record_ids_sha,
        "selected_record_count": 1,
        "selected_task_ids_sha256": _ids_sha256([candidate["task_id"]]),
        "selected_task_count": 1,
        "structural_records_sha256": structural_sha,
        "role_policy_path": str(role_policy_path.resolve()),
        "role_policy_sha256": _sha256(role_policy_path),
        "role_policy_bytes": role_policy_path.stat().st_size,
        "hidden_label_schema_path": str(schema_path.resolve()),
        "hidden_label_schema_sha256": _sha256(schema_path),
        "hidden_label_schema_bytes": schema_path.stat().st_size,
        "source_disjoint_partition_bindings": {
            "five_utr": {
                "split_manifest_sha256": "1" * 64,
                "partition_id": "source_disjoint:five_utr",
                "partition_sha256": "2" * 64,
            },
            "three_utr": {
                "split_manifest_sha256": "3" * 64,
                "partition_id": "source_disjoint:three_utr",
                "partition_sha256": "4" * 64,
            },
        },
        "d1_acceptance": {
            "path": str(acceptance_path.resolve()),
            "sha256": _sha256(acceptance_path),
            "bytes": acceptance_path.stat().st_size,
        },
        "d1_build_manifest": {
            "path": str(build_manifest_path.resolve()),
            "sha256": _sha256(build_manifest_path),
            "bytes": build_manifest_path.stat().st_size,
        },
        "canonical_records_declared": {
            "path": str(canonical_path.resolve()),
            "sha256": canonical_sha,
            "bytes": canonical_path.stat().st_size,
            "record_count": 1,
            "record_ids_sha256": record_ids_sha,
        },
        "structural_records": {
            "path": str(structural_path.resolve()),
            "sha256": structural_sha,
            "bytes": structural_path.stat().st_size,
            "record_count": 1,
            "record_ids_sha256": record_ids_sha,
            "structural_content_sha256": hashlib.sha256(
                structural_path.read_bytes()
            ).hexdigest(),
        },
        "canonical_label_store_opened": False,
        "selection_labels_hidden": True,
        "frozen_before_label_access": True,
    }
    selection_path.write_text(
        json.dumps(selection, sort_keys=True) + "\n", encoding="utf-8"
    )
    label_path = manifest_path.parent / "labels.hidden.jsonl"
    stable_canonical_sha = hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    label_path.write_text(
        json.dumps(
            {
                "schema_version": "utr_track_a_hidden_label.v2",
                "candidate_id": candidate["candidate_id"],
                "canonical_candidate_id": canonical["candidate_id"],
                "record_id": canonical["record_id"],
                "dataset_id": canonical["dataset_id"],
                "study_id": canonical["study_id"],
                "assay_id": canonical["assay_id"],
                "context_id": canonical["context_id"],
                "endpoint": canonical["endpoint"],
                "pair_type": canonical["pair_type"],
                "source_id": canonical["source_id"],
                "source_sequence_sha256": hashlib.sha256(
                    canonical["source_sequence"].encode()
                ).hexdigest(),
                "candidate_sequence_sha256": hashlib.sha256(
                    canonical["candidate_sequence"].encode()
                ).hexdigest(),
                "source_value_raw": canonical["source_value_raw"],
                "candidate_value_raw": canonical["candidate_value_raw"],
                "delta_raw": canonical["delta_raw"],
                "delta_normalized": canonical["delta_normalized"],
                "effect_standard_error": canonical["effect_standard_error"],
                "replicate_count": canonical["replicate_count"],
                "label_provenance": canonical["label_provenance"],
                "canonical_record_sha256": stable_canonical_sha,
                "measurement_evidence": ("paired_finite_measured_endpoints"),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_ids_sha = _ids_sha256([candidate["candidate_id"]])
    proof = {
        "schema_version": "utr_track_a_label_freeze_proof.v2",
        "track_id": manifest["track_id"],
        "candidate_store_sha256": _sha256(candidate_path),
        "candidate_store_bytes": candidate_path.stat().st_size,
        "label_store_sha256": _sha256(label_path),
        "label_store_bytes": label_path.stat().st_size,
        "candidate_ids_sha256": candidate_ids_sha,
        "candidate_count": 1,
        "label_candidate_ids_sha256": candidate_ids_sha,
        "label_count": 1,
        "label_record_ids_sha256": record_ids_sha,
        "label_record_count": 1,
        "selection_freeze_sha256": _sha256(selection_path),
        "selection_freeze_bytes": selection_path.stat().st_size,
        "role_policy_sha256": _sha256(role_policy_path),
        "hidden_label_schema_sha256": _sha256(schema_path),
        "canonical_records_sha256": manifest["universe_binding"][
            "canonical_records_sha256"
        ],
        "structural_records_sha256": manifest["universe_binding"][
            "structural_records_sha256"
        ],
        "record_ids_sha256": manifest["universe_binding"]["record_ids_sha256"],
        "frozen_before_label_access": True,
        "selection_labels_hidden": True,
    }
    proof_path = manifest_path.parent / "labels.freeze.json"
    proof_path.write_text(json.dumps(proof, sort_keys=True) + "\n", encoding="utf-8")
    manifest["label_store"].update(
        {
            "sha256": _sha256(label_path),
            "bytes": label_path.stat().st_size,
            "candidate_ids_sha256": candidate_ids_sha,
            "candidate_count": 1,
            "schema": {
                "path": schema_path.name,
                "sha256": _sha256(schema_path),
                "bytes": schema_path.stat().st_size,
            },
            "selection_freeze": {
                "path": selection_path.name,
                "sha256": _sha256(selection_path),
                "bytes": selection_path.stat().st_size,
            },
            "freeze_proof": {
                "path": proof_path.name,
                "sha256": _sha256(proof_path),
            },
        }
    )
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8")


def _privileged_verify_fixture(
    manifest_path: Path,
    *,
    expected_role_policy: dict | None = None,
):
    role_policy_path = manifest_path.parent / "track_role_policy.yaml"
    if expected_role_policy is None:
        expected_role_policy = (
            yaml.safe_load(role_policy_path.read_text(encoding="utf-8"))
            if role_policy_path.is_file()
            else {}
        )
    return privileged_verify_track_a_label_seal(
        manifest_path,
        expected_role_policy=expected_role_policy,
        expected_d1_acceptance_path=manifest_path.parent / "d1_acceptance.json",
    )


def test_selection_loader_never_opens_label_store_but_privileged_seal_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_track(
        tmp_path,
        track_id="track-a",
        track_type="closed_measured_pool",
        candidate_id="candidate-a",
    )
    _write_real_track_a_seal(path)
    original_open = Path.open

    def guarded_open(self: Path, *args, **kwargs):
        if self.name == "labels.hidden.jsonl":
            raise AssertionError("selection path opened final labels")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    loaded = load_track_manifest(path)
    assert loaded.label_store_access == "FROZEN_FINAL_ONLY"
    monkeypatch.undo()
    seal = _privileged_verify_fixture(path)
    assert seal["gate_passed"] is True
    assert seal["candidate_label_bijection"] is True


def test_privileged_track_a_seal_rejects_missing_tampered_or_nonbijective_labels(
    tmp_path: Path,
) -> None:
    path = _write_track(
        tmp_path,
        track_id="track-a",
        track_type="closed_measured_pool",
        candidate_id="candidate-a",
    )
    with pytest.raises(TrackContractError, match="does not exist"):
        _privileged_verify_fixture(path)

    _write_real_track_a_seal(path)
    label_path = tmp_path / "labels.hidden.jsonl"
    label_path.write_text(
        json.dumps({"candidate_id": "wrong-candidate", "measured_value": 4}) + "\n",
        encoding="utf-8",
    )
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    manifest["label_store"]["sha256"] = _sha256(label_path)
    manifest["label_store"]["bytes"] = label_path.stat().st_size
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises(
        TrackContractError, match="strict schema|freeze proof|bijection"
    ):
        _privileged_verify_fixture(path)


def test_legacy_two_field_label_cannot_pass_after_resealing_outer_hashes(
    tmp_path: Path,
) -> None:
    path = _write_track(
        tmp_path,
        track_id="track-a",
        track_type="closed_measured_pool",
        candidate_id="candidate-a",
    )
    _write_real_track_a_seal(path)
    label_path = tmp_path / "labels.hidden.jsonl"
    label_path.write_text(
        json.dumps(
            {
                "candidate_id": "candidate-a",
                "measured_value": 2.25,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    proof_path = tmp_path / manifest["label_store"]["freeze_proof"]["path"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["label_store_sha256"] = _sha256(label_path)
    proof["label_store_bytes"] = label_path.stat().st_size
    proof_path.write_text(json.dumps(proof, sort_keys=True) + "\n", encoding="utf-8")
    manifest["label_store"]["sha256"] = _sha256(label_path)
    manifest["label_store"]["bytes"] = label_path.stat().st_size
    manifest["label_store"]["freeze_proof"]["sha256"] = _sha256(proof_path)
    path.write_text(yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(TrackContractError, match="strict schema"):
        _privileged_verify_fixture(path)


def test_privileged_seal_rejects_fully_resealed_weakened_track_b_role_policy(
    tmp_path: Path,
) -> None:
    path = _write_track(
        tmp_path,
        track_id="track-a",
        track_type="closed_measured_pool",
        candidate_id="candidate-a",
    )
    _write_real_track_a_seal(path)
    role_policy_path = tmp_path / "track_role_policy.yaml"
    expected_role_policy = yaml.safe_load(role_policy_path.read_text(encoding="utf-8"))
    weakened_role_policy = deepcopy(expected_role_policy)
    weakened_role_policy["track_b_future_formal_generative_evaluation_allowed"] = False
    role_policy_path.write_text(
        yaml.safe_dump(weakened_role_policy, sort_keys=True),
        encoding="utf-8",
    )

    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    selection_path = tmp_path / manifest["label_store"]["selection_freeze"]["path"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["role_policy_sha256"] = _sha256(role_policy_path)
    selection["role_policy_bytes"] = role_policy_path.stat().st_size
    selection_path.write_text(
        json.dumps(selection, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    proof_path = tmp_path / manifest["label_store"]["freeze_proof"]["path"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["role_policy_sha256"] = _sha256(role_policy_path)
    proof["selection_freeze_sha256"] = _sha256(selection_path)
    proof["selection_freeze_bytes"] = selection_path.stat().st_size
    proof_path.write_text(
        json.dumps(proof, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["label_store"]["selection_freeze"]["sha256"] = _sha256(selection_path)
    manifest["label_store"]["selection_freeze"]["bytes"] = selection_path.stat().st_size
    manifest["label_store"]["freeze_proof"]["sha256"] = _sha256(proof_path)
    path.write_text(yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(
        TrackContractError,
        match="differs from independent structural recomputation",
    ):
        _privileged_verify_fixture(
            path,
            expected_role_policy=expected_role_policy,
        )


def test_track_loader_is_acgu_and_constraint_fail_closed(tmp_path: Path) -> None:
    ambiguous = _write_track(
        tmp_path,
        track_id="ambiguous",
        track_type="heldout_generative",
        candidate_id="candidate-n",
        extra_candidate={"candidate_sequence": "ANGU"},
    )
    with pytest.raises(TrackContractError, match="canonical RNA"):
        load_track_manifest(ambiguous)

    mismatch = _write_track(
        tmp_path,
        track_id="mismatch",
        track_type="heldout_generative",
        candidate_id="candidate-mismatch",
    )
    candidate_path = tmp_path / "mismatch.jsonl"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["constraints"]["allowed_operations"] = ["INS"]
    candidate_path.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    manifest = yaml.safe_load(mismatch.read_text(encoding="utf-8"))
    manifest["candidate_store"]["sha256"] = _sha256(candidate_path)
    manifest["candidate_store"]["bytes"] = candidate_path.stat().st_size
    mismatch.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises(TrackContractError, match="allowed_operations"):
        load_track_manifest(mismatch)

    over_budget = _write_track(
        tmp_path,
        track_id="over-budget",
        track_type="heldout_generative",
        candidate_id="candidate-over-budget",
        extra_candidate={"candidate_sequence": "AGGA"},
    )
    with pytest.raises(TrackContractError, match="exceeds max_edits"):
        load_track_manifest(over_budget)

    wrong_length = _write_track(
        tmp_path,
        track_id="wrong-length",
        track_type="heldout_generative",
        candidate_id="candidate-wrong-length",
        extra_candidate={"candidate_sequence": "ACGUA"},
    )
    with pytest.raises(TrackContractError, match="length violates"):
        load_track_manifest(wrong_length)


def test_edit_budget_uses_allowed_operations_not_plain_levenshtein(
    tmp_path: Path,
) -> None:
    path = _write_track(
        tmp_path,
        track_id="substitution-only-rotation",
        track_type="heldout_generative",
        candidate_id="candidate-rotation",
        extra_candidate={
            "candidate_sequence": "CGUA",
            "max_edits": 2,
        },
    )
    with pytest.raises(
        TrackContractError,
        match="exceeds max_edits under allowed_operations",
    ):
        load_track_manifest(path)


def test_track_types_enforce_exact_candidate_and_stop_semantics(
    tmp_path: Path,
) -> None:
    open_with_known_candidate = _write_track(
        tmp_path,
        track_id="open-with-known-candidate",
        track_type="open_legal_generation",
        candidate_id="candidate-open",
        extra_candidate={
            "candidate_id": "candidate-open",
            "candidate_sequence": "AGGU",
        },
    )
    with pytest.raises(TrackContractError, match="candidate null"):
        load_track_manifest(open_with_known_candidate)

    heldout_without_stop = _write_track(
        tmp_path,
        track_id="heldout-without-stop",
        track_type="heldout_generative",
        candidate_id="candidate-heldout",
    )
    candidate_path = tmp_path / "heldout-without-stop.jsonl"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["legal_action_types"] = ["SUB"]
    candidate["constraints"]["allowed_operations"] = ["SUB"]
    candidate_path.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    manifest = yaml.safe_load(heldout_without_stop.read_text(encoding="utf-8"))
    manifest["candidate_store"]["sha256"] = _sha256(candidate_path)
    manifest["candidate_store"]["bytes"] = candidate_path.stat().st_size
    heldout_without_stop.write_text(
        yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(TrackContractError, match="explicit STOP semantics"):
        load_track_manifest(heldout_without_stop)


def test_gse246381_is_rejected_from_normal_track_b_or_c(tmp_path: Path) -> None:
    for track_type in ("heldout_generative", "open_legal_generation"):
        path = _write_track(
            tmp_path,
            track_id=track_type,
            track_type=track_type,
            candidate_id=f"candidate-{track_type}",
            dataset_id="GSE246381",
        )
        with pytest.raises(TrackContractError, match="retrospective external stress"):
            load_track_manifest(path)


def _acceptance_bundle() -> tuple[list[dict], list[dict]]:
    clean = {
        "counts": {
            "unexplained_overlap_count": 0,
            "exact_source_leakage_count": 0,
            "exact_candidate_leakage_count": 0,
            "reverse_edge_leakage_count": 0,
            "path_leakage_count": 0,
            "near_neighbor_leakage_count": 0,
            "final_endpoint_as_train_intermediate_count": 0,
            "metadata_overlap_count": 0,
            "explained_metadata_overlap_count": 0,
            "unexplained_metadata_overlap_count": 0,
            "required_axis_overlap_count": 0,
        },
        "acceptance_gates": {
            "unexplained_overlap_zero": True,
            "exact_source_overlap_zero": True,
            "exact_candidate_overlap_zero": True,
            "reverse_edge_leakage_zero": True,
            "path_leakage_zero": True,
            "near_neighbor_leakage_zero": True,
            "final_endpoint_as_train_intermediate_zero": True,
            "required_axis_overlap_zero": True,
            "foundation_overlap_gate": True,
        },
        "metadata_axis_status": {
            axis: {
                "overlap_count": 0,
                "explained_overlap_count": 0,
                "unexplained_overlap_count": 0,
            }
            for axis in METADATA_DIMENSIONS
        },
        "foundation_pretraining_overlap": {
            "status": "UNKNOWN_PENDING_FM0",
            "foundation_selected": False,
            "allowed_claim": "NONE",
            "re_audit_required": True,
        },
    }
    identities = [
        ("source_disjoint", "five_utr", None),
        ("study_disjoint", "five_utr", None),
        ("source_disjoint", "three_utr", None),
        ("study_disjoint", "three_utr", None),
        ("cross_region_transfer", "five_utr", "three_utr"),
    ]
    manifests = []
    reports = []
    canonical_sha = "a" * 64
    structural_sha = "b" * 64
    full_record_ids = [f"r{index}" for index in range(1, 13)]
    role_record_ids = [f"r{index}" for index in range(1, 10)]
    record_ids_sha = _ids_sha256(full_record_ids)
    required_partitions = {
        ("source_disjoint", "five_utr", None): [
            "source_disjoint:five_utr",
            "sequence_cluster_disjoint:five_utr",
            "scaffold_disjoint:five_utr",
            "gene_disjoint:five_utr",
            "context_disjoint:five_utr",
            "barcode_batch_disjoint:five_utr",
            "library_batch_disjoint:five_utr",
        ],
        ("study_disjoint", "five_utr", None): [
            "loso:GSE114002",
            "loso:GSE217518",
        ],
        ("source_disjoint", "three_utr", None): [
            "source_disjoint:three_utr",
            "sequence_cluster_disjoint:three_utr",
            "scaffold_disjoint:three_utr",
            "gene_disjoint:three_utr",
            "context_disjoint:three_utr",
            "barcode_batch_disjoint:three_utr",
            "library_batch_disjoint:three_utr",
        ],
        ("study_disjoint", "three_utr", None): [
            "loso:GSE217518",
            "loso:GSE200304",
        ],
        ("cross_region_transfer", "five_utr", "three_utr"): [
            "within_study:GSE217518:five_to_three",
            "within_study:GSE217518:three_to_five",
            "cross_study:GSE114002_five_to_GSE200304_three",
        ],
    }
    for index, (split_kind, left_region, right_region) in enumerate(identities):
        manifest_sha = f"{index + 1:064x}"
        manifest = {
            "status": "READY",
            "split_kind": split_kind,
            "_artifact_sha256": manifest_sha,
            "canonical_records_sha256": canonical_sha,
            "structural_records_sha256": structural_sha,
            "structural_records_bytes": 1234,
            "canonical_record_ids_sha256": record_ids_sha,
            "canonical_record_count": 12,
            "structural_record_ids_sha256": record_ids_sha,
            "structural_record_count": 12,
            "structural_content_sha256": "3" * 64,
        }
        identity = (split_kind, left_region, right_region)
        if split_kind == "cross_region_transfer":
            eligible_ids = role_record_ids
        elif left_region == "five_utr":
            eligible_ids = role_record_ids[:4]
        else:
            eligible_ids = role_record_ids[4:]
        excluded_ids = sorted(set(full_record_ids) - set(eligible_ids))
        manifest.update(
            {
                "record_count": len(eligible_ids),
                "record_ids_sha256": _ids_sha256(eligible_ids),
                "eligible_records": [
                    {
                        "record_id": record_id,
                        "reason": "eligible_intervention_record",
                    }
                    for record_id in eligible_ids
                ],
                "excluded_record_count": len(excluded_ids),
                "excluded_records": [
                    {
                        "record_id": record_id,
                        "reason": "reasoned_fixture_exclusion",
                    }
                    for record_id in excluded_ids
                ],
            }
        )
        partitions = []
        partition_reports = []
        for partition_id in required_partitions[identity]:
            split_at = max(1, len(eligible_ids) - 2)
            expected_leakage_contract = expected_partition_leakage_contract(
                split_kind,
                region=left_region if split_kind != "cross_region_transfer" else None,
                partition_id=partition_id,
            )
            required_axes = expected_leakage_contract["required_disjoint_axes"]
            required_pairs = expected_leakage_contract["required_disjoint_role_pairs"]
            partition = {
                "partition_id": partition_id,
                "status": "READY",
                **deepcopy(expected_leakage_contract),
                "record_count": manifest["record_count"],
                "record_ids_sha256": manifest["record_ids_sha256"],
                "excluded_record_count": manifest["excluded_record_count"],
                "eligible_records": deepcopy(manifest["eligible_records"]),
                "excluded_records": deepcopy(manifest["excluded_records"]),
                "roles": {
                    "train": eligible_ids[:split_at],
                    "validation": eligible_ids[split_at : split_at + 1],
                    "test": eligible_ids[split_at + 1 :],
                },
            }
            partition["partition_sha256"] = _partition_sha256(partition)
            partitions.append(partition)
            partition_report = deepcopy(clean)
            partition_report.update(
                {
                    "partition_id": partition_id,
                    "split_partition_sha256": partition["partition_sha256"],
                    "required_axis_status": {
                        axis: {
                            "required_role_pairs": [
                                list(pair)
                                for pair in sorted(
                                    tuple(pair) for pair in required_pairs[axis]
                                )
                            ],
                            "raw_overlap_count": 0,
                            "gate_passed": True,
                        }
                        for axis in required_axes
                    },
                }
            )
            partition_reports.append(partition_report)
        manifest["required_partition_ids"] = required_partitions[identity]
        manifest["partitions"] = partitions
        if split_kind == "study_disjoint":
            manifest["folds"] = deepcopy(partitions)
        if split_kind == "cross_region_transfer":
            manifest["strata"] = deepcopy(partitions)
        report = deepcopy(clean)
        report.update(
            {
                "split_kind": split_kind,
                "split_manifest_sha256": manifest_sha,
                "canonical_records_sha256": canonical_sha,
                "structural_records_sha256": structural_sha,
                "structural_records_bytes": 1234,
                "canonical_record_ids_sha256": record_ids_sha,
                "canonical_record_count": 12,
                "structural_record_ids_sha256": record_ids_sha,
                "structural_record_count": 12,
                "structural_content_sha256": "3" * 64,
                "required_partition_ids": required_partitions[identity],
                "partitions": partition_reports,
            }
        )
        if split_kind == "cross_region_transfer":
            manifest["region"] = None
            manifest["source_region"] = left_region
            manifest["target_region"] = right_region
            report["region"] = None
            report["source_region"] = left_region
            report["target_region"] = right_region
        else:
            manifest["region"] = left_region
            report["region"] = left_region
            report["source_region"] = None
            report["target_region"] = None
        manifests.append(manifest)
        reports.append(report)
    return manifests, reports


def _clean_track_audit() -> dict:
    role_record_ids = [f"r{index}" for index in range(1, 10)]
    binding = {
        "canonical_records_sha256": "a" * 64,
        "structural_records_sha256": "b" * 64,
        "record_ids_sha256": _ids_sha256(role_record_ids),
        "record_count": 9,
        "candidate_ids_sha256": "d" * 64,
        "candidate_count": 6,
        "task_ids_sha256": "e" * 64,
        "task_count": 9,
        "source_ids_sha256": "f" * 64,
        "source_count": 9,
    }
    return {
        "track_role_ambiguity_count": 0,
        "gate_passed": True,
        "track_count": 3,
        "identity_universe_complete": True,
        "eligible_identity_binding_checked": True,
        "eligible_identity_binding_complete": True,
        "task_structural_binding_checked": True,
        "task_structural_binding_complete": True,
        "universe_binding": binding,
        "identity_universes": {
            "record": {
                "count": 9,
                "ids_sha256": _ids_sha256(role_record_ids),
            },
            "candidate": {
                "count": 6,
                "ids_sha256": "d" * 64,
            },
            "task": {
                "count": 9,
                "ids_sha256": "e" * 64,
            },
            "source": {
                "count": 9,
                "ids_sha256": "f" * 64,
            },
        },
        "data_card_counts": {
            "region_records": {"five_utr": 4, "three_utr": 5},
            "dataset_records": {"GSE114002": 9},
            "track_tasks": {
                "closed_measured_pool": 1,
                "heldout_generative": 2,
                "open_legal_generation": 6,
            },
            "track_evidence": {
                "closed_measured_pool": {
                    "measured_pair_type": 1,
                    "structural_unmeasured_pair_type": 0,
                },
                "heldout_generative": {
                    "measured_pair_type": 1,
                    "structural_unmeasured_pair_type": 1,
                },
                "open_legal_generation": {
                    "measured_pair_type": 6,
                    "structural_unmeasured_pair_type": 0,
                },
            },
        },
        "tracks": [
            {
                "track_id": "track-a",
                "track_type": "closed_measured_pool",
                "manifest_sha256": "1" * 64,
                "candidate_ids_sha256": "d" * 64,
                "candidate_count": 6,
                "label_store_sha256": "4" * 64,
                "label_store_bytes": 128,
                "label_freeze_proof_sha256": "5" * 64,
                "selection_freeze_sha256": "8" * 64,
                "label_schema_sha256": "9" * 64,
                "retrospective_external_stress_datasets": [],
            },
            {
                "track_id": "track-b",
                "track_type": "heldout_generative",
                "manifest_sha256": "2" * 64,
                "retrospective_external_stress_datasets": [],
            },
            {
                "track_id": "track-c",
                "track_type": "open_legal_generation",
                "manifest_sha256": "3" * 64,
                "retrospective_external_stress_datasets": [],
                "evaluation_budget_protocol": {
                    "required_budgets": [1, 3, 5],
                    "task_representation": "single_maximum_budget",
                    "maximum_budget": 5,
                    "report_each_budget_separately": True,
                    "silent_budget_reduction_forbidden": True,
                },
            },
        ],
        "gse246381_role": ("historically_exposed_retrospective_external_stress_test"),
    }


def _clean_role_policy(track_audit: dict) -> dict:
    return {
        "schema_version": "utr_b0_track_role_policy.v2",
        "record_counts": deepcopy(track_audit["data_card_counts"]["track_tasks"]),
        "track_evidence_counts": deepcopy(
            track_audit["data_card_counts"]["track_evidence"]
        ),
    }


def _clean_privileged_evidence() -> tuple[dict, dict]:
    return (
        {
            "schema_version": "utr_track_a_label_seal_audit.v2",
            "track_id": "track-a",
            "gate_passed": True,
            "candidate_label_bijection": True,
            "record_label_bijection": True,
            "strict_hidden_label_schema_passed": True,
            "paired_finite_measured_labels": True,
            "canonical_identity_binding_passed": True,
            "d1_acceptance_binding_passed": True,
            "current_d1_chain_binding_passed": True,
            "role_policy_exact_binding_passed": True,
            "label_store_sha256": "4" * 64,
            "label_store_bytes": 128,
            "freeze_proof_sha256": "5" * 64,
            "selection_freeze_sha256": "8" * 64,
            "role_policy_sha256": "a" * 64,
            "hidden_label_schema_sha256": "9" * 64,
            "d1_acceptance_sha256": "b" * 64,
            "d1_build_manifest_sha256": "c" * 64,
            "candidate_ids_sha256": "d" * 64,
            "candidate_count": 6,
            "canonical_records_sha256": "a" * 64,
            "structural_records_sha256": "b" * 64,
            "record_ids_sha256": _ids_sha256([f"r{index}" for index in range(1, 10)]),
        },
        {
            "schema_version": "utr_b0_required_artifact_audit.v2",
            "gate_passed": True,
            "universe_binding": _clean_track_audit()["universe_binding"],
            "artifacts": {
                name: {
                    "exists": True,
                    "bytes": 1,
                    "sha256": str(index) * 64,
                    "schema_valid": True,
                }
                for index, name in enumerate(
                    (
                        "exposure_ledger",
                        "track_role_matrix",
                        "data_card",
                        "claims",
                    ),
                    start=6,
                )
            },
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
    )


def _validate_fixture(
    manifests: list[dict],
    reports: list[dict],
    *,
    exposure_audit: dict | None = None,
    track_audit: dict | None = None,
    label_seal_audit: dict | None = None,
    required_artifact_audit: dict | None = None,
) -> dict:
    clean_label, clean_artifacts = _clean_privileged_evidence()
    return validate_b0_acceptance(
        leakage_reports=reports,
        exposure_audit=exposure_audit
        or {
            "coverage": 1.0,
            "gate_passed": True,
            "identity_level": "record_id",
        },
        track_audit=track_audit or _clean_track_audit(),
        split_manifests=manifests,
        track_a_label_seal_audit=label_seal_audit or clean_label,
        required_artifact_audit=required_artifact_audit or clean_artifacts,
        d1_exposure_ledger_binding={
            "schema_version": "utr_b0_d1_exposure_binding.v2",
            "gate_passed": True,
            "ledger_semantics_valid": True,
            "d1_acceptance_sha256": "b" * 64,
            "d1_build_manifest_sha256": "c" * 64,
        },
        supplied_leakage_reports_match_recomputation=True,
    )


def test_b0_acceptance_combines_exact_contract_gates() -> None:
    manifests, reports = _acceptance_bundle()
    result = _validate_fixture(manifests, reports)
    assert result["b0_gate_passed"] is True
    assert result["allowed_claim"] == "NONE"
    assert result["re_audit_required_before_foundation_use"] is True


def test_b0_acceptance_rejects_track_a_seal_from_a_different_d1_hash_chain() -> None:
    manifests, reports = _acceptance_bundle()
    label_seal, _ = _clean_privileged_evidence()
    label_seal["d1_acceptance_sha256"] = "e" * 64
    label_seal["d1_build_manifest_sha256"] = "f" * 64

    result = _validate_fixture(
        manifests,
        reports,
        label_seal_audit=label_seal,
    )

    assert result["b0_gate_passed"] is False
    assert "track_a_privileged_label_seal" in result["failed_gates"]


def test_b0_acceptance_rejects_resealed_weakened_overlap_policy() -> None:
    manifests, reports = _acceptance_bundle()
    manifest = next(
        item
        for item in manifests
        if item["split_kind"] == "source_disjoint" and item["region"] == "five_utr"
    )
    report = next(
        item
        for item in reports
        if item["split_kind"] == "source_disjoint" and item["region"] == "five_utr"
    )
    partition = next(
        item
        for item in manifest["partitions"]
        if item["partition_id"] == "scaffold_disjoint:five_utr"
    )
    partition_report = next(
        item
        for item in report["partitions"]
        if item["partition_id"] == partition["partition_id"]
    )
    weakened_pairs = [["train", "test"]]
    partition["required_disjoint_role_pairs"]["scaffold_group"] = weakened_pairs
    partition["overlap_policy"]["scaffold_group"]["allowed_role_pairs"] = [
        ["train", "validation"],
        ["validation", "test"],
    ]
    partition["partition_sha256"] = _partition_sha256(partition)
    partition_report["split_partition_sha256"] = partition["partition_sha256"]
    partition_report["required_axis_status"]["scaffold_group"][
        "required_role_pairs"
    ] = weakened_pairs
    manifest_without_artifact = {
        key: value for key, value in manifest.items() if key != "_artifact_sha256"
    }
    resealed_manifest_sha = hashlib.sha256(
        json.dumps(
            manifest_without_artifact,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest["_artifact_sha256"] = resealed_manifest_sha
    report["split_manifest_sha256"] = resealed_manifest_sha

    result = _validate_fixture(manifests, reports)

    assert result["b0_gate_passed"] is False
    assert "per_partition_frozen_leakage_contract" in result["failed_gates"]


def test_b0_acceptance_fails_on_any_proxy_for_contract_zero_gate() -> None:
    manifests, reports = _acceptance_bundle()
    reports[0]["counts"]["path_leakage_count"] = 1
    reports[0]["acceptance_gates"]["path_leakage_zero"] = False
    result = _validate_fixture(manifests, reports)
    assert result["b0_gate_passed"] is False
    assert "path_leakage_zero" in result["failed_gates"]


def test_b0_acceptance_rejects_subset_or_unbound_split_evidence() -> None:
    manifests, reports = _acceptance_bundle()
    subset = _validate_fixture(manifests[:-1], reports[:-1])
    assert subset["b0_gate_passed"] is False
    assert "exact_five_split_manifests_present" in subset["failed_gates"]
    assert "one_leakage_report_per_required_split" in subset["failed_gates"]

    reports[0]["split_manifest_sha256"] = "f" * 64
    unbound = _validate_fixture(manifests, reports)
    assert "leakage_report_manifest_binding" in unbound["failed_gates"]


def test_b0_acceptance_recomputes_foundation_clearance_evidence() -> None:
    manifests, reports = _acceptance_bundle()
    for report in reports:
        report["foundation_pretraining_overlap"] = {
            "status": "CLEARED_NO_OVERLAP",
            "foundation_selected": True,
            "allowed_claim": "FOUNDATION_OVERLAP_AUDITED",
            "re_audit_required": False,
        }
        report["acceptance_gates"]["foundation_overlap_gate"] = True
    result = _validate_fixture(manifests, reports)
    assert "foundation_state_must_remain_unknown_pending_fm0" in result["failed_gates"]
    assert result["allowed_claim"] == "NONE"
    assert result["requires_fm0_reaudit"] is True


def test_b0_acceptance_rejects_cross_split_universe_mismatch() -> None:
    manifests, reports = _acceptance_bundle()
    manifests[-1]["canonical_records_sha256"] = "9" * 64
    reports[-1]["canonical_records_sha256"] = "9" * 64
    result = _validate_fixture(manifests, reports)
    assert "five_splits_share_one_frozen_universe" in result["failed_gates"]


def test_b0_acceptance_requires_every_independently_bound_partition() -> None:
    manifests, reports = _acceptance_bundle()
    manifests[1]["partitions"][0]["roles"]["train"].append("tampered")
    reports[3]["partitions"][0]["counts"]["path_leakage_count"] = 1
    reports[3]["partitions"][0]["acceptance_gates"]["path_leakage_zero"] = False
    reports[0]["partitions"][0]["foundation_pretraining_overlap"] = {
        "foundation_selected": True,
        "status": "CLEARED_NO_OVERLAP",
        "checkpoint_sha256": "1" * 64,
        "corpus_manifest_sha256": "2" * 64,
        "audit_report_sha256": "3" * 64,
        "allowed_claim": "FOUNDATION_OVERLAP_AUDITED",
        "re_audit_required": False,
    }
    reports[-1]["partitions"].pop()
    result = _validate_fixture(manifests, reports)
    assert "split_partition_hash_binding" in result["failed_gates"]
    assert "per_partition_path_leakage_zero" in result["failed_gates"]
    assert "all_required_partitions_present" in result["failed_gates"]
    assert (
        "per_partition_foundation_state_unknown_pending_fm0" in result["failed_gates"]
    )


def test_b0_acceptance_keeps_reasoned_exclusions_out_of_all_roles() -> None:
    manifests, reports = _acceptance_bundle()
    manifests[0]["excluded_records"][0]["reason"] = ""
    missing_reason = _validate_fixture(manifests, reports)
    assert "eligible_and_excluded_record_accounting" in missing_reason["failed_gates"]

    manifests, reports = _acceptance_bundle()
    partition = manifests[0]["partitions"][0]
    excluded_id = partition["excluded_records"][0]["record_id"]
    partition["roles"]["train"].append(excluded_id)
    partition["partition_sha256"] = _partition_sha256(partition)
    reports[0]["partitions"][0]["split_partition_sha256"] = partition[
        "partition_sha256"
    ]
    assigned_exclusion = _validate_fixture(manifests, reports)
    assert (
        "excluded_record_has_track_or_split_role" in assigned_exclusion["failed_gates"]
    )


def test_b0_acceptance_rejects_minimal_tracks_without_complete_universe() -> None:
    manifests, reports = _acceptance_bundle()
    track_audit = _clean_track_audit()
    track_audit["identity_universe_complete"] = False
    track_audit["universe_binding"]["record_count"] = 3
    result = _validate_fixture(manifests, reports, track_audit=track_audit)
    assert "track_identity_universe_complete" in result["failed_gates"]
    assert "track_split_universe_binding" in result["failed_gates"]


def test_b0_acceptance_requires_privileged_label_and_required_artifact_audits() -> None:
    manifests, reports = _acceptance_bundle()
    clean_label, clean_artifacts = _clean_privileged_evidence()
    clean_label["gate_passed"] = False
    clean_artifacts["artifacts"]["data_card"]["exists"] = False
    clean_artifacts["gate_passed"] = False
    result = _validate_fixture(
        manifests,
        reports,
        label_seal_audit=clean_label,
        required_artifact_audit=clean_artifacts,
    )
    assert "track_a_privileged_label_seal" in result["failed_gates"]
    assert "required_artifacts_bound_and_valid" in result["failed_gates"]


def _write_required_artifact_bundle(root: Path, track_audit: dict) -> tuple[Path, Path]:
    exposure_path = root / "data_exposure_ledger.jsonl"
    exposure_path.write_text(
        "".join(
            json.dumps(
                {
                    "dataset_id": dataset_id,
                    "status": "accepted",
                    "historical_exposure": (
                        ACTIVE_DATASET_POLICIES["GSE246381"]["historical_exposure"]
                        if dataset_id == "GSE246381"
                        else "PREACCESS_FROZEN"
                    ),
                    "exposure_grade": (
                        ACTIVE_DATASET_POLICIES["GSE246381"]["exposure_grade"]
                        if dataset_id == "GSE246381"
                        else "E2"
                    ),
                    "allowed_uses": (
                        ACTIVE_DATASET_POLICIES["GSE246381"]["allowed_uses"]
                        if dataset_id == "GSE246381"
                        else ["benchmark"]
                    ),
                    "forbidden_uses": (
                        ACTIVE_DATASET_POLICIES["GSE246381"]["forbidden_uses"]
                        if dataset_id == "GSE246381"
                        else ["unblinded_selection"]
                    ),
                    "read_final_labels": False,
                    "provenance_complete": True,
                    "reason_code": None,
                }
            )
            + "\n"
            for dataset_id in sorted(D1_SCOPE_DATASETS)
        ),
        encoding="utf-8",
    )
    role_matrix_path = root / "track_role_matrix.yaml"
    role_matrix_path.write_text(
        yaml.safe_dump(track_audit, sort_keys=True), encoding="utf-8"
    )
    claims = {
        "schema_version": "utr_b0_claims.v2",
        "universe_binding": track_audit["universe_binding"],
        "foundation_status": "UNKNOWN_PENDING_FM0",
        "allowed_claim": "NONE",
        "requires_fm0_reaudit": True,
        "gse246381_role": ("historically_exposed_retrospective_external_stress_test"),
        "track_claims": {
            "closed_measured_pool": {
                "metric_name": "observed_pool_normalized_regret",
                "process_order_attestation_only": True,
                "forbidden_claims": ["full_legal_action_space_regret"],
            },
            "heldout_generative": {
                "b0_efficacy_conclusion_allowed": False,
                "future_formal_generative_evaluation_allowed": True,
                "evidence_scope": (
                    "B0 freezes heldout generative tasks without an efficacy "
                    "conclusion; later contract-gated formal generative "
                    "evaluation remains allowed"
                ),
                "forbidden_claims": ["measured_functional_improvement"],
            },
            "open_legal_generation": {
                "candidate_exposed": False,
                "required_evaluation_budgets": [1, 3, 5],
                "evidence_qualifiers": [
                    "predicted",
                    "computational",
                    "proxy-supported",
                ],
                "forbidden_claims": ["measured_improvement"],
            },
        },
        "allowed_claims": [
            "B0 structural benchmark, split, and track roles are frozen",
            (
                "Track B has no efficacy conclusion at B0; future formal "
                "generative evaluation under later contract gates is allowed"
            ),
        ],
        "unsupported_capabilities": [
            "No final efficacy, SOTA, or full legal action-space result",
            "No foundation-model exposure clearance before FM0 re-audit",
            "No measured improvement claim for open-world Track C",
        ],
    }
    claims_path = root / "claims.yaml"
    claims_path.write_text(yaml.safe_dump(claims, sort_keys=True), encoding="utf-8")
    binding = track_audit["universe_binding"]
    data_card_path = root / "Data_Card.md"
    data_card_path.write_text(
        render_canonical_data_card(
            universe_binding=binding,
            artifact_hashes={
                "exposure_ledger": _sha256(exposure_path),
                "track_role_matrix": _sha256(role_matrix_path),
                "claims": _sha256(claims_path),
            },
            track_audit=track_audit,
            role_policy=_clean_role_policy(track_audit),
        ),
        encoding="utf-8",
    )
    artifacts = {
        "exposure_ledger": (
            exposure_path,
            "d1_data_exposure_ledger.v2",
        ),
        "track_role_matrix": (
            role_matrix_path,
            "utr_track_role_audit.v2",
        ),
        "data_card": (data_card_path, "utr_editbench_data_card.v2"),
        "claims": (claims_path, "utr_b0_claims.v2"),
    }
    binding_manifest = {
        "schema_version": "utr_b0_artifact_bindings.v2",
        "universe_binding": binding,
        "artifacts": {
            name: {
                "path": path.name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "schema_version": schema_version,
            }
            for name, (path, schema_version) in artifacts.items()
        },
    }
    binding_path = root / "artifact_bindings.yaml"
    binding_path.write_text(
        yaml.safe_dump(binding_manifest, sort_keys=True), encoding="utf-8"
    )
    return binding_path, exposure_path


def test_required_artifact_validator_reads_and_binds_all_four_artifacts(
    tmp_path: Path,
) -> None:
    track_audit = _clean_track_audit()
    binding_path, exposure_path = _write_required_artifact_bundle(tmp_path, track_audit)
    audit = validate_required_artifacts(
        binding_path,
        track_audit=track_audit,
        exposure_ledger_path=exposure_path,
        role_policy=_clean_role_policy(track_audit),
    )
    assert audit["gate_passed"] is True
    assert set(audit["artifacts"]) == {
        "exposure_ledger",
        "track_role_matrix",
        "data_card",
        "claims",
    }
    assert all(
        item["exists"] and item["schema_valid"] for item in audit["artifacts"].values()
    )


def test_d1_ledger_binding_rejects_wrong_status_with_same_dataset_ids(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "data_exposure_ledger.jsonl"
    ledger_path.write_text(
        "".join(
            json.dumps(
                {
                    "dataset_id": dataset_id,
                    "status": "x",
                    "historical_exposure": (
                        ACTIVE_DATASET_POLICIES["GSE246381"]["historical_exposure"]
                        if dataset_id == "GSE246381"
                        else "PREACCESS_FROZEN"
                    ),
                    "exposure_grade": (
                        ACTIVE_DATASET_POLICIES["GSE246381"]["exposure_grade"]
                        if dataset_id == "GSE246381"
                        else "E2"
                    ),
                    "allowed_uses": (
                        ACTIVE_DATASET_POLICIES["GSE246381"]["allowed_uses"]
                        if dataset_id == "GSE246381"
                        else ["benchmark"]
                    ),
                    "forbidden_uses": (
                        ACTIVE_DATASET_POLICIES["GSE246381"]["forbidden_uses"]
                        if dataset_id == "GSE246381"
                        else ["unblinded_selection"]
                    ),
                    "read_final_labels": False,
                    "provenance_complete": True,
                    "reason_code": None,
                },
                sort_keys=True,
            )
            + "\n"
            for dataset_id in sorted(D1_SCOPE_DATASETS)
        ),
        encoding="utf-8",
    )
    ledger_ref = {
        "path": str(ledger_path.resolve()),
        "bytes": ledger_path.stat().st_size,
        "sha256": _sha256(ledger_path),
    }
    d1_root = tmp_path / "D1"
    d1_root.mkdir()
    build_path = d1_root / "build_manifest.json"
    build_path.write_text(
        json.dumps(
            {"required_artifacts": {"data/data_exposure_ledger.jsonl": ledger_ref}},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    build_ref = {
        "path": str(build_path.resolve()),
        "bytes": build_path.stat().st_size,
        "sha256": _sha256(build_path),
    }
    acceptance_path = tmp_path / "d1_acceptance.json"
    acceptance_path.write_text(
        json.dumps(
            {
                "phase_gate_passed": True,
                "fixture_mode": False,
                "structural_validation_passed": True,
                "stage_d1_root": str(d1_root.resolve()),
                "required_artifact_validation": {
                    "passed": True,
                    "build_manifest": build_ref,
                    "artifacts": {
                        "data/data_exposure_ledger.jsonl": {
                            **ledger_ref,
                            "exists": True,
                            "declared": ledger_ref,
                        }
                    },
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    audit = validate_d1_exposure_ledger_binding(
        acceptance_path,
        ledger_path,
    )
    assert audit["gate_passed"] is False
    assert "d1_exposure_ledger_semantics" in audit["failures"]


def test_required_artifact_validator_rejects_tamper_and_missing_artifacts(
    tmp_path: Path,
) -> None:
    track_audit = _clean_track_audit()
    binding_path, exposure_path = _write_required_artifact_bundle(tmp_path, track_audit)
    claims_path = tmp_path / "claims.yaml"
    claims_path.write_text(
        claims_path.read_text(encoding="utf-8") + "# tampered\n",
        encoding="utf-8",
    )
    tampered = validate_required_artifacts(
        binding_path,
        track_audit=track_audit,
        exposure_ledger_path=exposure_path,
        role_policy=_clean_role_policy(track_audit),
    )
    assert tampered["gate_passed"] is False
    assert "claims:hash_or_bytes_mismatch" in tampered["failures"]

    binding = yaml.safe_load(binding_path.read_text(encoding="utf-8"))
    binding["artifacts"]["data_card"]["path"] = "missing-data-card.md"
    binding_path.write_text(yaml.safe_dump(binding, sort_keys=True), encoding="utf-8")
    missing = validate_required_artifacts(
        binding_path,
        track_audit=track_audit,
        exposure_ledger_path=exposure_path,
        role_policy=_clean_role_policy(track_audit),
    )
    assert missing["gate_passed"] is False
    assert "data_card:missing" in missing["failures"]


def test_data_card_structured_facts_must_equal_recomputed_track_counts(
    tmp_path: Path,
) -> None:
    track_audit = _clean_track_audit()
    binding_path, exposure_path = _write_required_artifact_bundle(
        tmp_path,
        track_audit,
    )
    binding = yaml.safe_load(binding_path.read_text(encoding="utf-8"))
    data_card_path = tmp_path / binding["artifacts"]["data_card"]["path"]
    text = data_card_path.read_text(encoding="utf-8")
    front_text, body = text.removeprefix("---\n").split("---\n", 1)
    front = yaml.safe_load(front_text)
    front["structured_facts"]["counts"]["region_records"]["five_utr"] += 1
    data_card_path.write_text(
        "---\n" + yaml.safe_dump(front, sort_keys=True) + "---\n" + body,
        encoding="utf-8",
    )
    binding["artifacts"]["data_card"].update(
        {
            "sha256": _sha256(data_card_path),
            "bytes": data_card_path.stat().st_size,
        }
    )
    binding_path.write_text(
        yaml.safe_dump(binding, sort_keys=True),
        encoding="utf-8",
    )
    audit = validate_required_artifacts(
        binding_path,
        track_audit=track_audit,
        exposure_ledger_path=exposure_path,
        role_policy=_clean_role_policy(track_audit),
    )
    assert audit["gate_passed"] is False
    assert "data_card:schema_or_content" in audit["failures"]


def test_data_card_rejects_resealed_contradictory_scientific_prose(
    tmp_path: Path,
) -> None:
    track_audit = _clean_track_audit()
    binding_path, exposure_path = _write_required_artifact_bundle(
        tmp_path,
        track_audit,
    )
    binding = yaml.safe_load(binding_path.read_text(encoding="utf-8"))
    data_card_path = tmp_path / binding["artifacts"]["data_card"]["path"]
    data_card_path.write_text(
        data_card_path.read_text(encoding="utf-8")
        + "B0 proves measured improvement and state of the art efficacy.\n",
        encoding="utf-8",
    )
    binding["artifacts"]["data_card"].update(
        {
            "sha256": _sha256(data_card_path),
            "bytes": data_card_path.stat().st_size,
        }
    )
    binding_path.write_text(
        yaml.safe_dump(binding, sort_keys=True),
        encoding="utf-8",
    )

    audit = validate_required_artifacts(
        binding_path,
        track_audit=track_audit,
        exposure_ledger_path=exposure_path,
        role_policy=_clean_role_policy(track_audit),
    )

    assert audit["gate_passed"] is False
    assert "data_card:schema_or_content" in audit["failures"]


def test_claims_cannot_shrink_future_track_b_formal_evaluation(
    tmp_path: Path,
) -> None:
    track_audit = _clean_track_audit()
    binding_path, exposure_path = _write_required_artifact_bundle(
        tmp_path,
        track_audit,
    )
    binding = yaml.safe_load(binding_path.read_text(encoding="utf-8"))
    claims_path = tmp_path / binding["artifacts"]["claims"]["path"]
    claims = yaml.safe_load(claims_path.read_text(encoding="utf-8"))
    claims["track_claims"]["heldout_generative"][
        "future_formal_generative_evaluation_allowed"
    ] = False
    claims_path.write_text(
        yaml.safe_dump(claims, sort_keys=True),
        encoding="utf-8",
    )
    binding["artifacts"]["claims"].update(
        {
            "sha256": _sha256(claims_path),
            "bytes": claims_path.stat().st_size,
        }
    )
    binding_path.write_text(
        yaml.safe_dump(binding, sort_keys=True),
        encoding="utf-8",
    )
    audit = validate_required_artifacts(
        binding_path,
        track_audit=track_audit,
        exposure_ledger_path=exposure_path,
        role_policy=_clean_role_policy(track_audit),
    )
    assert audit["gate_passed"] is False
    assert "claims:schema_or_content" in audit["failures"]


def test_exposure_coverage_requires_an_explicit_disposition_for_every_dataset() -> None:
    records = [
        {"record_id": "r1", "dataset_id": "d1"},
        {"record_id": "r2", "dataset_id": "d2"},
    ]
    complete = compute_exposure_coverage(
        records,
        [
            {"dataset_id": "d1", "status": "PREACCESS_FROZEN"},
            {"dataset_id": "d2", "status": "HISTORICALLY_EXPOSED"},
        ],
        identity_level="dataset_id",
    )
    assert complete["coverage"] == 1.0
    assert complete["identity_level"] == "dataset_id"

    incomplete = compute_exposure_coverage(
        records,
        [{"dataset_id": "d1", "status": "PREACCESS_FROZEN"}],
        identity_level="dataset_id",
    )
    assert incomplete["coverage"] == 0.5
    assert incomplete["missing"] == ["d2"]

    with pytest.raises(ValueError, match="mixes or omits"):
        compute_exposure_coverage(
            records,
            [
                {"record_id": "r1", "status": "PREACCESS_FROZEN"},
                {"dataset_id": "d2", "status": "HISTORICALLY_EXPOSED"},
            ],
            identity_level="record_id",
        )
    with pytest.raises(ValueError, match="duplicate"):
        compute_exposure_coverage(
            records,
            [
                {"dataset_id": "d1", "status": "PREACCESS_FROZEN"},
                {"dataset_id": "d1", "status": "HISTORICALLY_EXPOSED"},
            ],
            identity_level="dataset_id",
        )
    blank = compute_exposure_coverage(
        records,
        [
            {"dataset_id": "d1", "status": ""},
            {"dataset_id": "d2", "status": None},
        ],
        identity_level="dataset_id",
    )
    assert blank["coverage"] == 0.0
    assert blank["gate_passed"] is False


def test_exposure_coverage_allows_only_an_exact_frozen_scope_superset() -> None:
    records = [
        {"record_id": "r1", "dataset_id": "d1"},
        {"record_id": "r2", "dataset_id": "d2"},
    ]
    complete = compute_exposure_coverage(
        records,
        [
            {"dataset_id": "d1", "status": "ELIGIBLE"},
            {"dataset_id": "d2", "status": "ELIGIBLE"},
            {
                "dataset_id": "blocked-dataset",
                "status": "BLOCKED_WITH_REASON",
            },
        ],
        identity_level="dataset_id",
        required_ledger_identities=["d1", "d2", "blocked-dataset"],
    )
    assert complete["coverage"] == 1.0
    assert complete["extra"] == ["blocked-dataset"]
    assert complete["ledger_scope_gate_passed"] is True
    assert complete["gate_passed"] is True

    missing_blocked = compute_exposure_coverage(
        records,
        [
            {"dataset_id": "d1", "status": "ELIGIBLE"},
            {"dataset_id": "d2", "status": "ELIGIBLE"},
        ],
        identity_level="dataset_id",
        required_ledger_identities=["d1", "d2", "blocked-dataset"],
    )
    assert missing_blocked["coverage"] == 1.0
    assert missing_blocked["ledger_scope_gate_passed"] is False
    assert missing_blocked["gate_passed"] is False

    out_of_scope = compute_exposure_coverage(
        records,
        [
            {"dataset_id": "d1", "status": "ELIGIBLE"},
            {"dataset_id": "d2", "status": "ELIGIBLE"},
            {"dataset_id": "unexpected", "status": "ELIGIBLE"},
        ],
        identity_level="dataset_id",
        required_ledger_identities=["d1", "d2"],
    )
    assert out_of_scope["outside_required_ledger_scope"] == ["unexpected"]
    assert out_of_scope["gate_passed"] is False
