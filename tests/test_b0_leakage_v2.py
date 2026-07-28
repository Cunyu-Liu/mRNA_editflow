from __future__ import annotations

import copy

from data.utr_benchmark_v2.leakage import audit_cross_role_leakage
from data.utr_benchmark_v2.leakage import audit_split_manifest
from data.utr_benchmark_v2.split_graph import METADATA_DIMENSIONS
from data.utr_benchmark_v2.split_graph import ROLE_PAIRS
from data.utr_benchmark_v2.split_graph import build_atomic_components
from data.utr_benchmark_v2.split_graph import build_split_manifest
from data.utr_benchmark_v2.split_graph import global_near_neighbor_clusters
from data.utr_benchmark_v2.split_graph import partition_sha256
from data.utr_benchmark_v2.split_graph import record_ids_sha256
from data.utr_benchmark_v2.split_graph import record_structural_sha256
from data.utr_benchmark_v2.split_graph import record_universe_sha256


def _record(
    record_id: str,
    source: str,
    candidate: str,
    *,
    dataset_id: str | None = None,
    region: str = "five_utr",
) -> dict:
    suffix = record_id.replace("-", "_")
    dataset = dataset_id or f"dataset_{suffix}"
    return {
        "record_id": record_id,
        "candidate_id": f"candidate_{suffix}",
        "dataset_id": dataset,
        "study_id": dataset,
        "assay_id": f"assay_{dataset}",
        "region": region,
        "source_id": f"source_{suffix}",
        "source_sequence": source,
        "candidate_sequence": candidate,
        "edit_script": [],
        "intermediate_sequences": [],
        "trajectory_observed": False,
        "source_group": f"source_group_{suffix}",
        "study_group": dataset,
        "sequence_cluster": f"cluster_{suffix}",
        "scaffold_group": f"scaffold_{suffix}",
        "gene_group": f"gene_{suffix}",
        "context_group": f"context_{suffix}",
        "barcode_batch": f"barcode_{suffix}",
        "library_batch": f"library_{suffix}",
        "quality_flags": [],
    }


def _policy(*, forbid: str | None = None) -> dict:
    return {
        field: {
            "allowed_role_pairs": (
                [] if field == forbid else [list(pair) for pair in ROLE_PAIRS]
            ),
            "unlisted_role_pairs": "FORBIDDEN",
            "justification": f"test policy for {field}",
        }
        for field in METADATA_DIMENSIONS
    }


def _manifest(
    records: list[dict],
    train: list[str],
    validation: list[str],
    test: list[str],
    *,
    policy: dict | None = None,
) -> dict:
    near_neighbors = global_near_neighbor_clusters(records)
    occurrences = {
        record_id: role
        for role, ids in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        )
        for record_id in ids
    }
    components = build_atomic_components(records)
    component_roles = {}
    for component in components:
        roles = {
            occurrences[record_id]
            for record_id in component.record_ids
            if record_id in occurrences
        }
        if len(roles) == 1:
            component_roles[component.component_id] = next(iter(roles))
    record_index = {record["record_id"]: record for record in records}
    roles = {"train": train, "validation": validation, "test": test}
    eligible = [
        {
            "record_id": record["record_id"],
            "reason": "eligible_intervention_record",
            "structural_sha256": record_structural_sha256(record),
        }
        for record in sorted(records, key=lambda row: row["record_id"])
    ]
    partition = {
        "schema_version": "utr_split_manifest.v2",
        "status": "READY",
        "split_kind": "source_disjoint",
        "partition_id": "source_disjoint:five_utr",
        "region": "five_utr",
        "full_record_count": len(records),
        "full_record_ids_sha256": record_ids_sha256(records),
        "full_record_universe_sha256": record_universe_sha256(records),
        "record_count": len(records),
        "record_ids_sha256": record_ids_sha256(records),
        "record_universe_sha256": record_universe_sha256(records),
        "eligible_records": eligible,
        "eligible_record_accounting_sha256": __import__(
            "hashlib"
        ).sha256(
            __import__("json").dumps(
                eligible,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest(),
        "excluded_records": [],
        "excluded_record_count": 0,
        "excluded_record_ids_sha256": record_ids_sha256([]),
        "excluded_record_accounting_sha256": __import__(
            "hashlib"
        ).sha256(b"[]").hexdigest(),
        "exclusion_reason_counts": {},
        "component_count": len(components),
        "roles": roles,
        "component_roles": component_roles,
        "near_neighbor_binding": dict(near_neighbors.binding),
        "required_disjoint_axes": [
            "near_neighbor_cluster",
            "source_group",
            "scaffold_group",
        ],
        "required_disjoint_role_pairs": {
            axis: [list(pair) for pair in ROLE_PAIRS]
            for axis in (
                "near_neighbor_cluster",
                "source_group",
                "scaffold_group",
            )
        },
        "role_bindings": {
            role: {
                "record_count": len(ids),
                "record_ids_sha256": record_ids_sha256(
                    [record_index[record_id] for record_id in ids]
                ),
                "record_universe_sha256": record_universe_sha256(
                    [record_index[record_id] for record_id in ids]
                ),
            }
            for role, ids in roles.items()
        },
        "overlap_policy": policy or _policy(),
    }
    partition["partition_sha256"] = partition_sha256(partition)
    return partition


def test_clean_partition_has_zero_unexplained_structural_leakage() -> None:
    records = [
        _record("train-1", "AAAAAAAA", "AAAAAAAC"),
        _record("val-1", "CCCCCCCC", "CCCCCCCG"),
        _record("test-1", "GGGGGGGG", "GGGGGGGU"),
    ]
    report = audit_cross_role_leakage(
        records,
        _manifest(records, ["train-1"], ["val-1"], ["test-1"]),
        foundation_exposure=None,
    )
    assert report["gate_passed"] is True
    assert report["counts"]["unexplained_overlap_count"] == 0
    assert report["acceptance_gates"]["reverse_edge_leakage_zero"] is True
    assert report["acceptance_gates"]["path_leakage_zero"] is True
    foundation = report["foundation_pretraining_overlap"]
    assert foundation == {
        "status": "UNKNOWN_PENDING_FM0",
        "foundation_selected": False,
        "checkpoint_sha256": None,
        "corpus_manifest_sha256": None,
        "audit_report_sha256": None,
        "clearance_evidence_complete": False,
        "allowed_claim": "NONE",
        "re_audit_required": True,
        "gate_applicable": False,
        "gate_passed": True,
    }


def test_reverse_edge_and_all_minimum_path_overlap_are_detected() -> None:
    records = [
        _record("train-1", "AAAA", "CCCC"),
        _record("test-1", "CCCC", "AAAA"),
        _record("val-1", "GGGG", "GGGU"),
    ]
    report = audit_cross_role_leakage(
        records,
        _manifest(records, ["train-1"], ["val-1"], ["test-1"]),
    )
    assert report["counts"]["reverse_edge_leakage_count"] == 1
    assert report["counts"]["path_leakage_count"] > 0
    assert report["acceptance_gates"]["reverse_edge_leakage_zero"] is False


def test_final_endpoint_as_recomputed_train_dag_intermediate_is_detected() -> None:
    records = [
        # AC -> CA has CC as one reachable minimum-alignment intermediate.
        _record("train-1", "AC", "CA"),
        _record("val-1", "UU", "UA"),
        _record("test-1", "CG", "CC"),
    ]
    report = audit_cross_role_leakage(
        records,
        _manifest(records, ["train-1"], ["val-1"], ["test-1"]),
    )
    assert (
        report["counts"]["final_endpoint_as_train_intermediate_count"] == 1
    )
    assert report["acceptance_gates"][
        "final_endpoint_as_train_intermediate_zero"
    ] is False


def test_final_endpoint_from_noncanonical_action_order_is_detected() -> None:
    records = [
        # AA -> CC has two shortest SUB execution orders.  AC is reachable
        # only when the second position is edited first; the former
        # alignment-prefix implementation omitted it and falsely returned 0.
        _record("train-1", "AA", "CC"),
        _record("val-1", "UU", "UA"),
        _record("test-1", "GG", "AC"),
    ]
    report = audit_cross_role_leakage(
        records,
        _manifest(records, ["train-1"], ["val-1"], ["test-1"]),
    )
    assert (
        report["counts"]["final_endpoint_as_train_intermediate_count"] == 1
    )
    assert report["acceptance_gates"][
        "final_endpoint_as_train_intermediate_zero"
    ] is False


def test_metadata_overlap_must_be_predeclared_or_is_unexplained() -> None:
    records = [
        _record("train-1", "AAAA", "AAAC"),
        _record("val-1", "CCCC", "CCCG"),
        _record("test-1", "GGGG", "GGGU"),
    ]
    records[2]["study_group"] = records[0]["study_group"]
    explained = audit_cross_role_leakage(
        records,
        _manifest(records, ["train-1"], ["val-1"], ["test-1"]),
    )
    assert explained["counts"]["explained_metadata_overlap_count"] > 0
    assert explained["counts"]["unexplained_metadata_overlap_count"] == 0

    forbidden = audit_cross_role_leakage(
        records,
        _manifest(
            records,
            ["train-1"],
            ["val-1"],
            ["test-1"],
            policy=_policy(forbid="study_group"),
        ),
    )
    assert forbidden["counts"]["unexplained_metadata_overlap_count"] == 1
    assert forbidden["acceptance_gates"]["unexplained_overlap_zero"] is False
    assert forbidden["metadata_axis_status"]["study_group"]["status"] == "FAIL"


def test_required_axis_raw_overlap_cannot_be_relabelled_explained() -> None:
    records = [
        _record("train-1", "AAAAAAAA", "AAAAAAAC"),
        _record("val-1", "CCCCCCCC", "CCCCCCCG"),
        _record("test-1", "GGGGGGGG", "GGGGGGGU"),
    ]
    records[2]["gene_group"] = records[0]["gene_group"]
    manifest = _manifest(
        records,
        ["train-1"],
        ["val-1"],
        ["test-1"],
        # This is the historical false-PASS pattern: every overlap is
        # statically declared explained.
        policy=_policy(),
    )
    manifest["required_disjoint_axes"].append("gene_group")
    manifest["required_disjoint_role_pairs"]["gene_group"] = [
        list(pair) for pair in ROLE_PAIRS
    ]
    manifest["partition_sha256"] = partition_sha256(manifest)

    report = audit_cross_role_leakage(records, manifest)
    assert report["counts"]["unexplained_metadata_overlap_count"] == 0
    assert report["counts"]["required_axis_overlap_count"] == 1
    assert report["required_axis_status"]["gene_group"][
        "raw_overlap_count"
    ] == 1
    assert report["acceptance_gates"]["required_axis_overlap_zero"] is False
    assert report["gate_passed"] is False


def test_source_or_candidate_tamper_with_same_record_id_cannot_pass() -> None:
    records = [
        _record("train-1", "AAAA", "AAAC"),
        _record("val-1", "CCCC", "CCCG"),
        _record("test-1", "GGGG", "GGGU"),
    ]
    manifest = _manifest(
        records, ["train-1"], ["val-1"], ["test-1"]
    )
    for field, replacement in (
        ("source_sequence", "AAAU"),
        ("candidate_sequence", "AAAG"),
    ):
        tampered = copy.deepcopy(records)
        tampered[0][field] = replacement
        report = audit_cross_role_leakage(tampered, manifest)
        assert report["counts"]["frozen_universe_issue_count"] > 0
        assert report["counts"]["unexplained_overlap_count"] > 0
        assert report["gate_passed"] is False


def test_near_neighbor_algorithm_or_threshold_tamper_cannot_pass() -> None:
    records = [
        _record("train-1", "AAAAAAAA", "AAAAAAAC"),
        _record("val-1", "CCCCCCCC", "CCCCCCCG"),
        _record("test-1", "GGGGGGGG", "GGGGGGGU"),
    ]
    manifest = _manifest(
        records, ["train-1"], ["val-1"], ["test-1"]
    )
    manifest["near_neighbor_binding"]["edit_distance_threshold"] = 4
    manifest["partition_sha256"] = partition_sha256(manifest)

    report = audit_cross_role_leakage(records, manifest)
    assert report["counts"]["frozen_universe_issue_count"] > 0
    assert report["acceptance_gates"]["unexplained_overlap_zero"] is False
    assert report["gate_passed"] is False


def test_manifest_role_subset_cannot_pass_as_complete_universe() -> None:
    records = [
        _record("train-1", "AAAA", "AAAC"),
        _record("train-2", "AAAG", "AAAU"),
        _record("val-1", "CCCC", "CCCG"),
        _record("test-1", "GGGG", "GGGU"),
    ]
    manifest = _manifest(
        records,
        ["train-1", "train-2"],
        ["val-1"],
        ["test-1"],
    )
    manifest["roles"]["train"].remove("train-2")
    manifest["partition_sha256"] = partition_sha256(manifest)
    report = audit_cross_role_leakage(records, manifest)
    assert report["counts"]["frozen_universe_issue_count"] > 0
    assert report["acceptance_gates"]["unexplained_overlap_zero"] is False


def test_selected_foundation_unknown_and_unbound_clearance_both_block() -> None:
    records = [
        _record("train-1", "AAAA", "AAAC"),
        _record("val-1", "CCCC", "CCCG"),
        _record("test-1", "GGGG", "GGGU"),
    ]
    manifest = _manifest(
        records, ["train-1"], ["val-1"], ["test-1"]
    )
    unknown = audit_cross_role_leakage(
        records,
        manifest,
        foundation_exposure={
            "foundation_selected": True,
            "status": "UNKNOWN_PENDING_FM0",
        },
    )
    assert unknown["acceptance_gates"]["foundation_overlap_gate"] is False
    cleared = audit_cross_role_leakage(
        records,
        manifest,
        foundation_exposure={
            "foundation_selected": True,
            "status": "CLEARED_NO_OVERLAP",
        },
    )
    assert cleared["foundation_pretraining_overlap"]["status"] == (
        "INVALID_CLEARANCE_EVIDENCE"
    )
    assert cleared["foundation_pretraining_overlap"]["gate_passed"] is False


def test_top_level_audit_checks_every_loso_partition_and_common_universe() -> None:
    records = [
            _record(
                "g114-a",
                "AAAAAAAAAAAA",
                "AAAAAAAAAAAC",
                dataset_id="GSE114002",
            ),
            _record(
                "g114-b",
                "CCCCCCCCCCCC",
                "CCCCCCCCCCCG",
                dataset_id="GSE114002",
            ),
            _record(
                "g217-a",
                "GGGGGGGGGGGG",
                "GGGGGGGGGGGA",
                dataset_id="GSE217518",
            ),
            _record(
                "g217-b",
                "UUUUUUUUUUUU",
                "UUUUUUUUUUUC",
                dataset_id="GSE217518",
        ),
    ]
    manifest = build_split_manifest(
        records, region="five_utr", split_kind="study_disjoint"
    )
    report = audit_split_manifest(records, manifest)
    assert report["gate_passed"] is True
    assert set(report["required_partition_ids"]) == {
        "loso:GSE114002",
        "loso:GSE217518",
    }
    assert len(report["partitions"]) == 2
    assert all(
        partition["gate_passed"] for partition in report["partitions"]
    )
    assert {
        partition["partition_id"] for partition in report["partitions"]
    } == set(report["required_partition_ids"])
    assert all(
        partition["split_partition_sha256"]
        == next(
            item["partition_sha256"]
            for item in manifest["partitions"]
            if item["partition_id"] == partition["partition_id"]
        )
        for partition in report["partitions"]
    )


def test_top_level_audit_rejects_deleting_an_axis_partition_and_its_id() -> None:
    records = []
    for index in range(9):
        source, candidate = _unique_pair(index)
        records.append(
            _record(
                f"r-{index}",
                source,
                candidate,
                dataset_id=f"dataset-{index}",
            )
        )
    manifest = build_split_manifest(
        records, region="five_utr", split_kind="source_disjoint"
    )
    removed = manifest["partitions"].pop()
    manifest["required_partition_ids"].remove(removed["partition_id"])

    report = audit_split_manifest(records, manifest)
    assert report["gate_passed"] is False
    assert any(
        issue["kind"] == "frozen_required_partition_set_mismatch"
        for issue in report["structural_issues"]
    )


def _unique_pair(index: int) -> tuple[str, str]:
    alphabet = "ACGU"
    value = index
    digits = []
    for _ in range(6):
        digits.append(alphabet[value % 4])
        value //= 4
    prefix = "".join(base * 6 for base in reversed(digits))
    return prefix + "AAAAAA", prefix + "AAAAAC"


def test_exact_five_manifests_share_one_full_universe_and_all_partitions_audit() -> None:
    records: list[dict] = []
    specifications = (
        ("GSE114002", "five_utr"),
        ("GSE217518", "five_utr"),
        ("GSE217518", "three_utr"),
        ("GSE200304", "three_utr"),
    )
    index = 0
    for dataset_id, region in specifications:
        for _ in range(3):
            source, candidate = _unique_pair(index)
            records.append(
                _record(
                    f"r-{index}",
                    source,
                    candidate,
                    dataset_id=dataset_id,
                    region=region,
                )
            )
            index += 1
    manifests = [
        build_split_manifest(
            records,
            region="five_utr",
            split_kind="source_disjoint",
        ),
        build_split_manifest(
            records,
            region="five_utr",
            split_kind="study_disjoint",
        ),
        build_split_manifest(
            records,
            region="three_utr",
            split_kind="source_disjoint",
        ),
        build_split_manifest(
            records,
            region="three_utr",
            split_kind="study_disjoint",
        ),
        build_split_manifest(
            records,
            region=None,
            split_kind="cross_region_transfer",
        ),
    ]
    reports = [
        audit_split_manifest(records, manifest) for manifest in manifests
    ]
    assert all(manifest["status"] == "READY" for manifest in manifests)
    assert all(report["gate_passed"] is True for report in reports)
    assert manifests[3]["required_partition_ids"] == [
        "loso:GSE217518",
        "loso:GSE200304",
    ]
    assert manifests[4]["required_partition_ids"] == [
        "within_study:GSE217518:five_to_three",
        "within_study:GSE217518:three_to_five",
        "cross_study:GSE114002_five_to_GSE200304_three",
    ]
    assert {
        manifest["full_record_universe_sha256"] for manifest in manifests
    } == {record_universe_sha256(records)}
    assert {
        manifest["near_neighbor_binding"]["record_assignment_sha256"]
        for manifest in manifests
    } == {
        manifests[0]["near_neighbor_binding"][
            "record_assignment_sha256"
        ]
    }
    assert all(
        report["counts"]["unexplained_overlap_count"] == 0
        for report in reports
    )
