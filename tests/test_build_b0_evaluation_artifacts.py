from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from data.utr_benchmark_v2.d1_builder import ACTIVE_DATASET_POLICIES
from data.utr_benchmark_v2.split_graph import record_ids_sha256
from data.utr_benchmark_v2.split_graph import record_universe_sha256
from data.utr_benchmark_v2.split_graph import build_split_manifest
from data.utr_benchmark_v2.split_graph import canonical_split_manifest_core
from data.utr_benchmark_v2.split_graph import partition_sha256
from scripts.data import build_b0_evaluation_artifacts as builder
from scripts.data.build_b0_splits import _bind_manifest


def _sequence(index: int, width: int = 16) -> str:
    alphabet = "ACGU"
    digest = hashlib.sha256(str(index).encode("utf-8")).digest()
    return "".join(alphabet[value & 3] for value in digest[:width])


def _structural_record(
    record_id: str,
    *,
    index: int,
    region: str = "five_utr",
    pair_type: str = "true_wt_mutant",
    source: str | None = None,
    candidate: str | None = None,
    dataset_id: str = "GSE114002",
) -> dict:
    source_sequence = source or _sequence(index)
    candidate_sequence = candidate or (
        source_sequence[:-1] + ("C" if source_sequence[-1] != "C" else "A")
    )
    return {
        "record_id": record_id,
        "candidate_id": f"canonical-candidate-{record_id}",
        "dataset_id": dataset_id,
        "study_id": dataset_id,
        "assay_id": f"assay-{record_id}",
        "context_id": f"context-{record_id}",
        "region": region,
        "source_id": f"source-{record_id}",
        "source_sequence": source_sequence,
        "candidate_sequence": candidate_sequence,
        "endpoint": "half_life",
        "edit_types": ["SUB"],
        "edit_count": 1,
        "edit_distance": 1,
        "edit_script": [
            {
                "op": "SUB",
                "pos": len(source_sequence) - 1,
                "ref": source_sequence[-1],
                "alt": candidate_sequence[-1],
            }
        ],
        "intermediate_sequences": [],
        "trajectory_observed": False,
        "pair_type": pair_type,
        "source_group": f"source-group-{record_id}",
        "study_group": dataset_id,
        "sequence_cluster": f"cluster-{record_id}",
        "scaffold_group": f"scaffold-{record_id}",
        "gene_group": f"gene-{record_id}",
        "context_group": f"context-{record_id}",
        "barcode_batch": f"barcode-{record_id}",
        "library_batch": f"library-{record_id}",
    }


def _source_manifest(
    region: str,
    *,
    train: list[str],
    validation: list[str],
    test: list[str],
) -> dict:
    required_ids = list(
        builder.expected_partition_ids("source_disjoint", region=region)
    )
    partitions = [
        {
            "partition_id": partition_id,
            "partition_sha256": hashlib.sha256(
                partition_id.encode("utf-8")
            ).hexdigest(),
            "status": "READY",
            "roles": {
                "train": train,
                "validation": validation,
                "test": test,
            },
        }
        for partition_id in required_ids
    ]
    return {
        "split_kind": "source_disjoint",
        "region": region,
        "_artifact_sha256": ("1" * 64 if region == "five_utr" else "2" * 64),
        "required_partition_ids": required_ids,
        "partitions": partitions,
    }


def test_source_manifest_selects_source_state_and_requires_every_axis() -> None:
    manifest = _source_manifest(
        "five_utr",
        train=["r1"],
        validation=["r2"],
        test=["r3"],
    )
    manifest["partitions"] = list(reversed(manifest["partitions"]))
    selected = builder._partition_for_source_manifest(manifest)
    assert selected["partition_id"] == "source_disjoint:five_utr"

    missing = json.loads(json.dumps(manifest))
    missing["partitions"].pop()
    with pytest.raises(builder.B0ArtifactBuildError, match="all seven frozen"):
        builder._partition_for_source_manifest(missing)

    wrong_id = json.loads(json.dumps(manifest))
    wrong_id["partitions"][0]["partition_id"] = "unknown_axis:five_utr"
    with pytest.raises(
        builder.B0ArtifactBuildError, match="differ from the frozen IDs"
    ):
        builder._partition_for_source_manifest(wrong_id)

    blocked = json.loads(json.dumps(manifest))
    blocked["partitions"][0]["status"] = "BLOCKED"
    with pytest.raises(
        builder.B0ArtifactBuildError, match="every frozen.*must be READY"
    ):
        builder._partition_for_source_manifest(blocked)


def test_label_free_component_policy_never_splits_mixed_test_component() -> None:
    measured_bridge = _structural_record("r1", index=1, source="AAAA", candidate="AAAC")
    unmeasured_bridge = _structural_record(
        "r2",
        index=2,
        source="AAAC",
        candidate="AAAG",
        pair_type="retrospective_constructed_neighbor",
    )
    measured_test = _structural_record("r3", index=30)
    validation = _structural_record("r4", index=40)
    train = _structural_record("r5", index=50)
    records = [
        measured_bridge,
        unmeasured_bridge,
        measured_test,
        validation,
        train,
    ]
    manifests = [
        _source_manifest(
            "five_utr",
            train=["r5"],
            validation=["r4"],
            test=["r1", "r2", "r3"],
        ),
        _source_manifest("three_utr", train=[], validation=[], test=[]),
    ]
    roles, policy = builder._derive_fixed_track_roles(
        manifests,
        {record["record_id"] for record in records},
        records,
    )

    assert roles["r1"] == roles["r2"] == "heldout_generative"
    assert roles["r3"] == "closed_measured_pool"
    assert roles["r4"] == "heldout_generative"
    assert roles["r5"] == "open_legal_generation"
    assert policy["atomic_components_split"] is False
    assert policy["label_fields_read_for_selection"] == []
    assert (
        policy["measured_eligibility"]["effect_value_or_direction_used_for_selection"]
        is False
    )
    assert policy["rerouted_test_component_count"] == 1
    assert policy["track_evidence_counts"]["closed_measured_pool"] == {
        "measured_pair_type": 1,
        "structural_unmeasured_pair_type": 0,
    }


def test_component_policy_fails_closed_when_track_a_would_be_empty() -> None:
    records = [
        _structural_record(
            "r1",
            index=1,
            pair_type="retrospective_constructed_neighbor",
        ),
        _structural_record("r2", index=2),
        _structural_record("r3", index=3),
    ]
    manifests = [
        _source_manifest("five_utr", train=["r3"], validation=["r2"], test=["r1"]),
        _source_manifest("three_utr", train=[], validation=[], test=[]),
    ]
    with pytest.raises(
        builder.B0ArtifactBuildError, match="all three tracks must be non-empty"
    ):
        builder._derive_fixed_track_roles(
            manifests,
            {record["record_id"] for record in records},
            records,
        )


def _late_binding(
    canonical_path: Path,
    structural_path: Path,
) -> dict:
    return {
        "passed": True,
        "canonical": {
            "path": str(canonical_path.resolve()),
            "sha256": builder._sha256(canonical_path),
        },
        "structural": {
            "path": str(structural_path.resolve()),
            "sha256": builder._sha256(structural_path),
        },
    }


def test_hidden_labels_require_finite_paired_arithmetic_and_identity_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structural = tmp_path / "structural.jsonl"
    structural.write_text("{}\n", encoding="utf-8")
    record = _structural_record("r1", index=1)
    record.update(
        {
            "source_value_raw": 1.25,
            "candidate_value_raw": 2.0,
            "delta_raw": 0.75,
            "delta_normalized": 0.5,
            "effect_standard_error": 0.1,
            "replicate_count": 3,
            "label_provenance": {"status": "measured"},
        }
    )
    canonical = tmp_path / "canonical.jsonl"
    canonical.write_text(json.dumps(record) + "\n", encoding="utf-8")
    task = builder._generation_task(record, track_type="closed_measured_pool")
    universe = {
        "canonical_records_sha256": builder._sha256(canonical),
        "canonical_record_count": 1,
        "canonical_record_ids_sha256": record_ids_sha256([record]),
        "structural_records_sha256": builder._sha256(structural),
    }
    monkeypatch.setattr(
        builder,
        "_validate_full_d1_binding",
        lambda *_: _late_binding(canonical, structural),
    )

    rows, _ = builder._hidden_label_rows(
        canonical,
        [task],
        d1_acceptance_path=tmp_path / "acceptance.json",
        split_universe=universe,
        structural_records_path=structural,
    )
    audit = builder._validate_hidden_label_rows(rows)
    assert audit["gate_passed"] is True
    assert rows[0]["record_id"] == "r1"
    assert rows[0]["canonical_candidate_id"] == record["candidate_id"]
    assert rows[0]["context_id"] == record["context_id"]
    assert rows[0]["measurement_evidence"] == ("paired_finite_measured_endpoints")

    record["delta_raw"] = 0.74
    canonical.write_text(json.dumps(record) + "\n", encoding="utf-8")
    universe["canonical_records_sha256"] = builder._sha256(canonical)
    with pytest.raises(builder.B0ArtifactBuildError, match="delta arithmetic mismatch"):
        builder._hidden_label_rows(
            canonical,
            [task],
            d1_acceptance_path=tmp_path / "acceptance.json",
            split_universe=universe,
            structural_records_path=structural,
        )


def _fixture_acceptance_bundle() -> tuple[list[dict], list[dict]]:
    path = Path(__file__).with_name("test_b0_tracks_and_acceptance_v2.py")
    spec = importlib.util.spec_from_file_location("_b0_fixture_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._acceptance_bundle()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_split_provenance_binding(root: Path) -> dict:
    acceptance_path = root / "split_input_d1_acceptance.json"
    ambiguity_path = root / "split_input_ambiguity_report.json"
    validation_path = root / "split_input_canonical_validation.json"
    _write_json(acceptance_path, {"phase_gate_passed": True})
    _write_json(ambiguity_path, {"count_scope": ["fixture"]})
    _write_json(validation_path, {"status": "PASS"})
    return {
        "d1_acceptance_path": str(acceptance_path.resolve()),
        "d1_acceptance_sha256": builder._sha256(acceptance_path),
        "d1_ambiguity_report_path": str(ambiguity_path.resolve()),
        "d1_ambiguity_report_sha256": builder._sha256(ambiguity_path),
        "ambiguity_count_scope": ["fixture"],
        "fresh_projection_comparison": {"passed": True},
        "canonical_validation_report_path": str(validation_path.resolve()),
        "canonical_validation_report_sha256": builder._sha256(validation_path),
    }


def _write_real_split_evidence(
    root: Path,
) -> tuple[list[Path], list[Path], Path, Path]:
    structural_rows = []
    for index in range(1, 13):
        region = "five_utr" if index <= 4 else "three_utr"
        if index <= 9:
            if index <= 2:
                dataset_id = "GSE114002"
            elif index <= 6:
                dataset_id = "GSE217518"
            else:
                dataset_id = "GSE200304"
            structural_rows.append(
                _structural_record(
                    f"r{index}",
                    index=index + 500,
                    region=region,
                    pair_type=(
                        "retrospective_constructed_neighbor"
                        if index == 9
                        else "true_wt_mutant"
                    ),
                    dataset_id=dataset_id,
                )
            )
        else:
            structural_rows.append(
                {
                    "record_id": f"r{index}",
                    "dataset_id": "GSE114002",
                    "region": region,
                }
            )
    structural_path = root / "structural.jsonl"
    canonical_path = root / "canonical.jsonl"
    structural_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in structural_rows),
        encoding="utf-8",
    )
    canonical_path.write_text(
        "".join(json.dumps(row) + "\n" for row in structural_rows),
        encoding="utf-8",
    )
    full_ids_sha = record_ids_sha256(structural_rows)
    binding = {
        **_write_split_provenance_binding(root),
        "canonical_records_path": str(canonical_path.resolve()),
        "canonical_records_sha256": builder._sha256(canonical_path),
        "canonical_record_count": len(structural_rows),
        "canonical_record_ids_sha256": full_ids_sha,
        "structural_records_path": str(structural_path.resolve()),
        "structural_records_sha256": builder._sha256(structural_path),
        "structural_records_bytes": structural_path.stat().st_size,
        "structural_record_count": len(structural_rows),
        "structural_record_ids_sha256": full_ids_sha,
        "structural_content_sha256": record_universe_sha256(structural_rows),
        "d1_phase_gate_passed": True,
    }
    split_paths: list[Path] = []
    report_paths: list[Path] = []
    for index, (split_kind, region) in enumerate(
        (
            ("source_disjoint", "five_utr"),
            ("study_disjoint", "five_utr"),
            ("source_disjoint", "three_utr"),
            ("study_disjoint", "three_utr"),
            ("cross_region_transfer", None),
        )
    ):
        manifest = build_split_manifest(
            structural_rows,
            split_kind=split_kind,
            region=region,
        )
        assert manifest["status"] == "READY"
        _bind_manifest(manifest, binding)
        manifest_path = root / "splits" / f"split-{index}.json"
        _write_json(manifest_path, manifest)
        report = builder.recompute_bound_leakage_report(
            structural_path,
            manifest_path,
        )
        assert report["gate_passed"] is True
        report_path = root / "reports" / f"report-{index}.json"
        _write_json(report_path, report)
        split_paths.append(manifest_path)
        report_paths.append(report_path)
    return split_paths, report_paths, structural_path, canonical_path


def test_split_evidence_recomputes_in_process_and_rejects_fake_zero_reports(
    tmp_path: Path,
) -> None:
    split_paths, report_paths, records_path, _ = _write_real_split_evidence(tmp_path)
    rows = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        if "source_sequence" not in row:
            continue
        row.update(
            {
                "scaffold_group": "shared-scaffold",
                "context_group": "shared-context",
                "barcode_batch": "shared-barcode",
                "library_batch": "shared-library",
            }
        )
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    # The supplied reports still self-declare zero leakage. The records are
    # nevertheless rejected because the builder recomputes from the bound file.
    assert all(
        json.loads(path.read_text(encoding="utf-8"))["counts"][
            "required_axis_overlap_count"
        ]
        == 0
        for path in report_paths
    )
    with pytest.raises(
        builder.B0ArtifactBuildError,
        match="bound leakage recomputation failed",
    ):
        builder._load_and_validate_split_evidence(
            split_paths,
            report_paths,
            structural_records_path=records_path,
        )


@pytest.mark.parametrize("mutation", ("reorder", "replace", "content"))
def test_split_evidence_rejects_record_store_substitution(
    tmp_path: Path,
    mutation: str,
) -> None:
    split_paths, report_paths, records_path, _ = _write_real_split_evidence(tmp_path)
    lines = records_path.read_text(encoding="utf-8").splitlines()
    if mutation == "reorder":
        lines = list(reversed(lines))
    else:
        rows = [json.loads(line) for line in lines]
        if mutation == "replace":
            rows[0]["record_id"] = "replacement-record"
        else:
            rows[0]["endpoint"] = "tampered-endpoint"
        lines = [json.dumps(row, sort_keys=True) for row in rows]
    records_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(
        builder.B0ArtifactBuildError,
        match="bound leakage recomputation failed",
    ):
        builder._load_and_validate_split_evidence(
            split_paths,
            report_paths,
            structural_records_path=records_path,
        )


def test_split_evidence_rejects_supplied_report_count_tampering(
    tmp_path: Path,
) -> None:
    split_paths, report_paths, records_path, _ = _write_real_split_evidence(tmp_path)
    report = json.loads(report_paths[0].read_text(encoding="utf-8"))
    report["counts"]["metadata_overlap_count"] = 0
    _write_json(tmp_path / "tampered-report.json", report)
    tampered_paths = [
        tmp_path / "tampered-report.json",
        *report_paths[1:],
    ]
    with pytest.raises(
        builder.B0ArtifactBuildError,
        match="supplied_leakage_report_differs_from_recomputation",
    ):
        builder._load_and_validate_split_evidence(
            split_paths,
            tampered_paths,
            structural_records_path=records_path,
        )


def test_canonical_manifest_recompute_rejects_resealed_invented_exclusion(
    tmp_path: Path,
) -> None:
    split_paths, report_paths, records_path, _ = _write_real_split_evidence(tmp_path)
    manifest_path = split_paths[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    structural_records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    record_by_id = {row["record_id"]: row for row in structural_records}
    moved = dict(manifest["eligible_records"][0])
    moved["reason"] = "invented_exclusion_to_shrink_benchmark"

    def shrink_accounting(payload: dict) -> None:
        payload["eligible_records"] = [
            row
            for row in payload["eligible_records"]
            if row["record_id"] != moved["record_id"]
        ]
        payload["excluded_records"] = sorted(
            [*payload["excluded_records"], moved],
            key=lambda row: row["record_id"],
        )
        remaining = [
            record_by_id[row["record_id"]] for row in payload["eligible_records"]
        ]
        payload["record_count"] = len(remaining)
        payload["record_ids_sha256"] = record_ids_sha256(remaining)
        payload["record_universe_sha256"] = record_universe_sha256(remaining)
        payload["eligible_record_accounting_sha256"] = builder._stable_sha256(
            payload["eligible_records"]
        )
        payload["excluded_record_count"] = len(payload["excluded_records"])
        payload["excluded_record_ids_sha256"] = record_ids_sha256(
            [{"record_id": row["record_id"]} for row in payload["excluded_records"]]
        )
        payload["excluded_record_accounting_sha256"] = builder._stable_sha256(
            payload["excluded_records"]
        )
        reason_counts: dict[str, int] = {}
        for row in payload["excluded_records"]:
            reason = row["reason"]
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        payload["exclusion_reason_counts"] = dict(sorted(reason_counts.items()))

    shrink_accounting(manifest)
    for partition in manifest["partitions"]:
        shrink_accounting(partition)
        for role_ids in partition["roles"].values():
            if moved["record_id"] in role_ids:
                role_ids.remove(moved["record_id"])
        partition["role_counts"] = {
            role: len(record_ids) for role, record_ids in partition["roles"].items()
        }
        partition["partition_sha256"] = partition_sha256(partition)
    manifest["partitions_sha256"] = builder._stable_sha256(
        [
            {
                "partition_id": partition["partition_id"],
                "partition_sha256": partition["partition_sha256"],
            }
            for partition in manifest["partitions"]
        ]
    )
    _write_json(manifest_path, manifest)

    report_path = report_paths[0]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["split_manifest_sha256"] = builder._sha256(manifest_path)
    report["split_manifest_bytes"] = manifest_path.stat().st_size
    report["canonical_manifest_core_sha256"] = builder._stable_sha256(
        canonical_split_manifest_core(manifest)
    )
    partition_sha_by_id = {
        partition["partition_id"]: partition["partition_sha256"]
        for partition in manifest["partitions"]
    }
    for partition_report in report["partitions"]:
        partition_report["split_partition_sha256"] = partition_sha_by_id[
            partition_report["partition_id"]
        ]
    _write_json(report_path, report)

    with pytest.raises(
        ValueError,
        match="split manifest differs from canonical structural recomputation",
    ):
        builder.recompute_bound_leakage_report(
            records_path,
            manifest_path,
        )


@pytest.mark.parametrize(
    ("manifest_index", "hash_field"),
    (
        (0, "partitions_sha256"),
        (1, "folds_sha256"),
        (4, "strata_sha256"),
    ),
)
def test_bound_recompute_rejects_tampered_aggregate_self_hashes(
    tmp_path: Path,
    manifest_index: int,
    hash_field: str,
) -> None:
    split_paths, _, records_path, _ = _write_real_split_evidence(tmp_path)
    manifest_path = split_paths[manifest_index]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[hash_field] = "f" * 64
    _write_json(manifest_path, manifest)

    with pytest.raises(
        ValueError,
        match="aggregate self-hash binding changed",
    ):
        builder.recompute_bound_leakage_report(records_path, manifest_path)


def test_bound_recompute_rejects_partition_d1_overlay_mismatch(
    tmp_path: Path,
) -> None:
    split_paths, _, records_path, _ = _write_real_split_evidence(tmp_path)
    manifest_path = split_paths[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    partition = manifest["partitions"][0]
    partition["d1_acceptance_sha256"] = "f" * 64
    partition["partition_sha256"] = partition_sha256(partition)
    manifest["partitions_sha256"] = builder._stable_sha256(
        [
            {
                "partition_id": item["partition_id"],
                "partition_sha256": item["partition_sha256"],
            }
            for item in manifest["partitions"]
        ]
    )
    _write_json(manifest_path, manifest)

    with pytest.raises(
        ValueError,
        match="partition D1 provenance overlay differs",
    ):
        builder.recompute_bound_leakage_report(records_path, manifest_path)


def test_bound_recompute_rejects_a_different_current_d1_acceptance(
    tmp_path: Path,
) -> None:
    split_paths, _, records_path, _ = _write_real_split_evidence(tmp_path)
    other_d1_acceptance = tmp_path / "other_d1_acceptance.json"
    _write_json(other_d1_acceptance, {"phase_gate_passed": True})

    with pytest.raises(
        ValueError,
        match="different D1 acceptance artifact",
    ):
        builder.recompute_bound_leakage_report(
            records_path,
            split_paths[0],
            expected_d1_acceptance_path=other_d1_acceptance,
        )


def test_full_builder_freezes_before_labels_and_emits_bound_three_tracks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structural_rows = []
    canonical_rows = []
    for index in range(1, 13):
        region = "five_utr" if index <= 4 else "three_utr"
        if index <= 9:
            if index <= 2:
                dataset_id = "GSE114002"
            elif index <= 6:
                dataset_id = "GSE217518"
            else:
                dataset_id = "GSE200304"
            pair_type = (
                "retrospective_constructed_neighbor" if index == 9 else "true_wt_mutant"
            )
            structural = _structural_record(
                f"r{index}",
                index=index + 100,
                region=region,
                pair_type=pair_type,
                dataset_id=dataset_id,
            )
        else:
            structural = {
                "record_id": f"r{index}",
                "dataset_id": "GSE114002",
                "region": region,
            }
        structural_rows.append(structural)
        canonical = dict(structural)
        if index <= 9:
            canonical.update(
                {
                    "source_value_raw": float(index),
                    "candidate_value_raw": float(index) + 1.0,
                    "delta_raw": 1.0,
                    "delta_normalized": 1.0,
                    "effect_standard_error": None,
                    "replicate_count": 2,
                    "label_provenance": {"status": "measured"},
                }
            )
        canonical_rows.append(canonical)

    d1_root = tmp_path / "D1"
    canonical_path = d1_root / "canonical" / "records_with_labels.jsonl"
    structural_path = d1_root / "candidate_store" / "candidates.jsonl"
    canonical_path.parent.mkdir(parents=True)
    structural_path.parent.mkdir(parents=True)
    canonical_path.write_text(
        "".join(json.dumps(row) + "\n" for row in canonical_rows),
        encoding="utf-8",
    )
    structural_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in structural_rows),
        encoding="utf-8",
    )
    canonical_sha = builder._sha256(canonical_path)
    structural_sha = builder._sha256(structural_path)
    full_ids_sha = record_ids_sha256(structural_rows)
    structural_content_sha = record_universe_sha256(structural_rows)

    binding = {
        **_write_split_provenance_binding(tmp_path),
        "canonical_records_path": str(canonical_path.resolve()),
        "canonical_records_sha256": canonical_sha,
        "canonical_record_count": len(canonical_rows),
        "canonical_record_ids_sha256": full_ids_sha,
        "structural_records_path": str(structural_path.resolve()),
        "structural_records_sha256": structural_sha,
        "structural_records_bytes": structural_path.stat().st_size,
        "structural_record_count": len(structural_rows),
        "structural_record_ids_sha256": full_ids_sha,
        "structural_content_sha256": structural_content_sha,
        "d1_phase_gate_passed": True,
    }
    split_specs = (
        ("source_disjoint", "five_utr"),
        ("study_disjoint", "five_utr"),
        ("source_disjoint", "three_utr"),
        ("study_disjoint", "three_utr"),
        ("cross_region_transfer", None),
    )
    manifests = [
        build_split_manifest(
            structural_rows,
            split_kind=split_kind,
            region=region,
        )
        for split_kind, region in split_specs
    ]
    assert all(manifest["status"] == "READY" for manifest in manifests)
    split_paths = []
    report_paths = []
    for index, manifest in enumerate(manifests):
        _bind_manifest(manifest, binding)
        manifest_path = tmp_path / "splits" / f"split-{index}.json"
        _write_json(manifest_path, manifest)
        report = builder.recompute_bound_leakage_report(
            structural_path,
            manifest_path,
        )
        report_path = tmp_path / "reports" / f"report-{index}.json"
        _write_json(report_path, report)
        split_paths.append(manifest_path)
        report_paths.append(report_path)

    exposure_path = tmp_path / "data_exposure_ledger.jsonl"
    exposure_path.write_text(
        "".join(
            json.dumps(
                {
                    "dataset_id": dataset_id,
                    "status": ("accepted" if dataset_id == "GSE114002" else "blocked"),
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
                    "reason_code": (
                        None
                        if dataset_id == "GSE114002"
                        else "BLOCKED_FIXTURE_ZERO_RECORD_SCOPE"
                    ),
                },
                sort_keys=True,
            )
            + "\n"
            for dataset_id in sorted(builder.D1_SCOPE_DATASETS)
        ),
        encoding="utf-8",
    )
    ledger_ref = {
        "path": str(exposure_path.resolve()),
        "bytes": exposure_path.stat().st_size,
        "sha256": builder._sha256(exposure_path),
    }
    build_manifest = {
        "global_stores": {
            "canonical_label_store": {
                "path": "canonical/records_with_labels.jsonl",
                "sha256": canonical_sha,
                "bytes": canonical_path.stat().st_size,
                "records": 12,
                "record_ids_sha256": full_ids_sha,
            },
            "sealed_label_free_candidate_store": {
                "path": "candidate_store/candidates.jsonl",
                "sha256": structural_sha,
                "bytes": structural_path.stat().st_size,
                "records": 12,
                "record_ids_sha256": full_ids_sha,
            },
        },
        "required_artifacts": {
            "data/data_exposure_ledger.jsonl": ledger_ref,
        },
    }
    build_manifest_path = d1_root / "build_manifest.json"
    _write_json(build_manifest_path, build_manifest)
    build_ref = {
        "path": str(build_manifest_path.resolve()),
        "bytes": build_manifest_path.stat().st_size,
        "sha256": builder._sha256(build_manifest_path),
    }
    acceptance = {
        "phase_gate_passed": True,
        "fixture_mode": False,
        "structural_validation_passed": True,
        "stage_d1_root": str(d1_root.resolve()),
        "global_store_validation": {"passed": True},
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
    }
    acceptance_path = tmp_path / "d1_acceptance.json"
    _write_json(acceptance_path, acceptance)
    current_d1_binding = {
        "d1_acceptance_path": str(acceptance_path.resolve()),
        "d1_acceptance_sha256": builder._sha256(acceptance_path),
    }
    for manifest_path, report_path in zip(split_paths, report_paths):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _bind_manifest(manifest, current_d1_binding)
        _write_json(manifest_path, manifest)
        _write_json(
            report_path,
            builder.recompute_bound_leakage_report(
                structural_path,
                manifest_path,
                expected_d1_acceptance_path=acceptance_path,
            ),
        )
    output_root = tmp_path / "B0"

    def late_binding(observed_canonical: Path, _observed_acceptance: Path) -> dict:
        freeze_path = (
            output_root
            / "evaluation"
            / "tracks"
            / "closed_measured_pool.selection.freeze.json"
        )
        assert freeze_path.is_file()
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        assert freeze["canonical_label_store_opened"] is False
        return {
            "passed": True,
            "canonical": {
                "path": str(observed_canonical.resolve()),
                "sha256": canonical_sha,
            },
            "structural": {
                "path": str(structural_path.resolve()),
                "sha256": structural_sha,
            },
            "d1_acceptance_sha256": builder._sha256(acceptance_path),
            "build_manifest_sha256": builder._sha256(d1_root / "build_manifest.json"),
        }

    monkeypatch.setattr(builder, "_validate_full_d1_binding", late_binding)
    result = builder.build_b0_evaluation_artifacts(
        d1_acceptance_path=acceptance_path,
        canonical_records_path=canonical_path,
        structural_records_path=structural_path,
        split_manifest_paths=split_paths,
        leakage_report_paths=report_paths,
        exposure_ledger_path=exposure_path,
        output_root=output_root,
    )

    assert result["status"] == "PASS"
    assert result["b0_gate_preview_passed"] is True
    policy_path = output_root / "evaluation" / "tracks" / "track_role_policy.yaml"
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    selection_freeze = json.loads(
        (
            output_root
            / "evaluation"
            / "tracks"
            / "closed_measured_pool.selection.freeze.json"
        ).read_text(encoding="utf-8")
    )
    assert selection_freeze["role_policy_sha256"] == builder._sha256(policy_path)
    # Under the frozen canonical replay scope, no alternate-order component reroutes r9.
    assert policy["rerouted_test_record_count"] == 0
    assert (
        policy["track_evidence_counts"]["closed_measured_pool"][
            "structural_unmeasured_pair_type"
        ]
        == 0
    )

    track_b_tasks = [
        json.loads(line)
        for line in (
            output_root / "evaluation" / "tracks" / "heldout_generative.tasks.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    # r9 is structural-only and no longer shares an all-order path component.
    assert "r9" not in {task["provenance"]["record_id"] for task in track_b_tasks}
    track_c_tasks = [
        json.loads(line)
        for line in (
            output_root / "evaluation" / "tracks" / "open_legal_generation.tasks.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(
        task["candidate_id"] is None and task["candidate_sequence"] is None
        for task in track_c_tasks
    )
    assert "r9" in {task["provenance"]["record_id"] for task in track_c_tasks}
    hidden_labels = [
        json.loads(line)
        for line in (
            output_root
            / "evaluation"
            / "tracks"
            / "closed_measured_pool.labels.hidden.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    track_a_tasks = [
        json.loads(line)
        for line in (
            output_root / "evaluation" / "tracks" / "closed_measured_pool.tasks.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["record_id"] for row in hidden_labels} == {
        task["provenance"]["record_id"] for task in track_a_tasks
    }
    assert builder._validate_hidden_label_rows(hidden_labels)["gate_passed"] is True
    data_card = (
        output_root / "docs" / "data" / "UTR_EditBench_v2_Data_Card.md"
    ).read_text(encoding="utf-8")
    assert "future contract-gated formal generative evaluation remains" in data_card
    assert "edit budgets 1, 3, and 5 separately" in data_card
    assert "source and legal actions only" in data_card
    with pytest.raises(FileExistsError):
        builder.build_b0_evaluation_artifacts(
            d1_acceptance_path=acceptance_path,
            canonical_records_path=canonical_path,
            structural_records_path=structural_path,
            split_manifest_paths=split_paths,
            leakage_report_paths=report_paths,
            exposure_ledger_path=exposure_path,
            output_root=output_root,
        )
