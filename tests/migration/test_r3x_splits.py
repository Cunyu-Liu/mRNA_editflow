"""Global relation-component split tests for R3 ordinary data."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.r3x.splits import (  # noqa: E402
    SplitHoldError,
    audit_component_disjointness,
    build_source_components,
    build_split_manifest,
    normalize_sequence,
    sequence_identity,
    verify_split_manifest_against_rows,
)


EVIDENCE_SHA = "e" * 64


def _atom(atom_id: str, *, member: str = "SOURCE", status: str = "FROZEN") -> dict:
    return {
        "atom_type": "SOURCE_ALIAS",
        "atom_id": atom_id,
        "member": member,
        "status": status,
        "evidence_sha256": EVIDENCE_SHA,
    }


def _row(
    row_id: str,
    source: str,
    candidate: str,
    *,
    ranking_pool: str | None = None,
    atoms: list[dict] | None = None,
    study: str = "study_a",
    endpoint: str = "endpoint_a",
    context: str = "context_a",
) -> dict:
    return {
        "r3_input_row_sha256": row_id,
        "normalized_source_sequence": source,
        "normalized_candidate_sequence": candidate,
        "source_sequence_sha256": sequence_identity(source),
        "candidate_sequence_sha256": sequence_identity(candidate),
        "source_group_id_v2": ranking_pool or f"ranking_{row_id}",
        "frozen_relation_atoms": atoms if atoms is not None else [_atom(f"alias_{row_id}")],
        "study": study,
        "endpoint": endpoint,
        "publication_context_id": context,
    }


def _independent_rows(n: int = 9) -> list[dict]:
    alphabet = "ACGU"
    rows = []
    for index in range(n):
        # Each pair gets sequences that cannot coincide with another pair.
        prefix = alphabet[index % len(alphabet)] * (3 + index)
        rows.append(_row(f"row_{index:02d}", prefix + "A", prefix + "C"))
    return rows


def test_sequence_identity_normalizes_case_whitespace_and_dna_t():
    assert normalize_sequence(" a c g t\n") == "ACGU"
    assert sequence_identity("acgt") == sequence_identity("ACGU")


def test_pair_edges_and_reused_sequence_identity_form_transitive_components():
    rows = [
        _row("r1", "AAAA", "CAAA"),
        _row("r2", "CAAA", "CCAA"),
        _row("r3", "GGGG", "UGGG"),
    ]
    graph = build_source_components(rows)
    assigned = graph["row_assignments"]
    assert assigned["r1"]["source_component_id"] == assigned["r2"]["source_component_id"]
    assert assigned["r3"]["source_component_id"] != assigned["r1"]["source_component_id"]
    assert graph["n_components"] == 2


def test_frozen_alias_atoms_bridge_components_through_transitive_closure():
    rows = [
        _row("r1", "AAAA", "CAAA", atoms=[_atom("shared", member="CANDIDATE")]),
        _row("r2", "GGGG", "UGGG", atoms=[_atom("shared", member="SOURCE")]),
        _row("r3", "CCCC", "ACCC", atoms=[_atom("other", member="SOURCE")]),
    ]
    graph = build_source_components(rows)
    assigned = graph["row_assignments"]
    assert assigned["r1"]["source_component_id"] == assigned["r2"]["source_component_id"]
    assert assigned["r3"]["source_component_id"] != assigned["r1"]["source_component_id"]


def test_logical_alias_identity_is_not_split_by_evidence_packaging():
    first = _atom("shared", member="SOURCE")
    second = _atom("shared", member="SOURCE")
    second["evidence_sha256"] = "f" * 64
    graph = build_source_components([
        _row("r1", "AAAA", "CAAA", atoms=[first]),
        _row("r2", "GGGG", "UGGG", atoms=[second]),
    ])
    assert graph["row_assignments"]["r1"]["source_component_id"] == graph["row_assignments"]["r2"]["source_component_id"]
    assert sorted(next(iter(graph["relation_atom_evidence"].values()))) == ["e" * 64, "f" * 64]


def test_source_group_v2_is_only_a_ranking_pool_and_never_a_split_edge():
    rows = [
        _row("r1", "AAAA", "CAAA", ranking_pool="same_pool", atoms=[_atom("a")]),
        _row("r2", "GGGG", "UGGG", ranking_pool="same_pool", atoms=[_atom("b")]),
    ]
    graph = build_source_components(rows)
    first = graph["row_assignments"]["r1"]
    second = graph["row_assignments"]["r2"]
    assert first["ranking_pool_id"] == second["ranking_pool_id"] == "same_pool"
    assert first["source_component_id"] != second["source_component_id"]
    assert graph["ranking_pool_policy"] == "SAME_CONTEXT_RANKING_ONLY_NOT_SPLIT_UNIT"


def test_all_frozen_grouping_atoms_are_edges_not_just_first_atom():
    rows = [
        _row("r1", "AAAA", "CAAA", atoms=[_atom("unique"), _atom("bridge")]),
        _row("r2", "GGGG", "UGGG", atoms=[_atom("bridge")]),
    ]
    graph = build_source_components(rows)
    assert graph["row_assignments"]["r1"]["source_component_id"] == graph[
        "row_assignments"
    ]["r2"]["source_component_id"]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda row: row.pop("frozen_relation_atoms"), "relation atoms"),
        (lambda row: row.__setitem__("frozen_relation_atoms", []), "relation atoms"),
        (
            lambda row: row.__setitem__(
                "frozen_relation_atoms", [_atom("x", status="PROPOSED")]
            ),
            "FROZEN",
        ),
        (lambda row: row.pop("source_group_id_v2"), "ranking pool"),
    ],
)
def test_missing_or_unfrozen_relation_evidence_holds(mutation, match):
    row = _row("r1", "AAAA", "CAAA")
    mutation(row)
    with pytest.raises(SplitHoldError, match=match):
        build_source_components([row])


def test_precomputed_sequence_hash_mismatch_holds():
    row = _row("r1", "AAAA", "CAAA")
    row["source_sequence_sha256"] = "0" * 64
    with pytest.raises(SplitHoldError) as exc:
        build_source_components([row])
    assert exc.value.code == "HOLD_SEQUENCE_IDENTITY_MISMATCH"


def test_component_graph_and_manifest_are_row_order_invariant():
    rows = _independent_rows()
    graph_forward = build_source_components(rows)
    graph_reverse = build_source_components(list(reversed(rows)))
    assert graph_forward == graph_reverse
    assert build_split_manifest(rows, seed=42) == build_split_manifest(
        list(reversed(rows)), seed=42
    )


def test_split_is_deterministic_and_seed_changes_component_assignment():
    rows = _independent_rows(12)
    first = build_split_manifest(rows, seed=42)
    repeat = build_split_manifest(copy.deepcopy(rows), seed=42)
    other = build_split_manifest(rows, seed=43)
    assert first == repeat
    assert first["row_assignments"] != other["row_assignments"]
    assert first["seed"] == 42
    assert set(first["partition_component_counts"]) == {"train", "development", "internal_test"}


def test_no_source_component_can_cross_partitions():
    rows = _independent_rows(12)
    manifest = build_split_manifest(rows, seed=11)
    audit = audit_component_disjointness(manifest)
    assert audit == {
        "status": "PASS_COMPONENT_DISJOINT",
        "overlap_component_ids": [],
    }
    component_partitions = {}
    for assignment in manifest["row_assignments"].values():
        component_partitions.setdefault(assignment["source_component_id"], set()).add(
            assignment["partition"]
        )
    assert all(len(partitions) == 1 for partitions in component_partitions.values())


def test_overlap_audit_fails_closed_for_tampered_manifest():
    manifest = build_split_manifest(_independent_rows(6), seed=11)
    row_ids = list(manifest["row_assignments"])
    manifest["row_assignments"][row_ids[1]]["source_component_id"] = manifest[
        "row_assignments"
    ][row_ids[0]]["source_component_id"]
    manifest["row_assignments"][row_ids[1]]["partition"] = "internal_test"
    manifest["row_assignments"][row_ids[0]]["partition"] = "train"
    with pytest.raises(SplitHoldError) as exc:
        audit_component_disjointness(manifest)
    assert exc.value.code == "HOLD_COMPONENT_OVERLAP"


def test_endpoint_nested_two_study_scope_is_explicitly_development_only():
    rows = [
        _row("a1", "AAAA", "CAAA", study="study_a", endpoint="ep_a"),
        _row("a2", "CCCC", "ACCC", study="study_a", endpoint="ep_a"),
        _row("a3", "GGGG", "UGGG", study="study_a", endpoint="ep_a"),
        _row("b1", "AAAAAA", "CAAAAA", study="study_b", endpoint="ep_b"),
        _row("b2", "CCCCCC", "ACCCCC", study="study_b", endpoint="ep_b"),
        _row("b3", "GGGGGG", "UGGGGG", study="study_b", endpoint="ep_b"),
    ]
    manifest = build_split_manifest(rows, seed=42)
    assert manifest["scientific_scope"] == "DEVELOPMENT_ONLY_TWO_STUDY_NESTED_ENDPOINTS"
    assert manifest["publication_cross_study_evidence"] is False


def test_single_study_never_asserts_cross_study_evidence():
    manifest = build_split_manifest(_independent_rows(9), seed=42)
    assert manifest["scientific_scope"] == "DEVELOPMENT_ONLY_SINGLE_STUDY"
    assert manifest["publication_cross_study_evidence"] is False


def test_rebuilt_split_rejects_forged_component_ids_and_stale_hash():
    rows = [
        _row("r1", "AAAA", "CAAA"),
        _row("r2", "CAAA", "CCAA"),
        _row("r3", "GGGG", "UGGG"),
    ]
    manifest = build_split_manifest(rows, seed=11)
    assert verify_split_manifest_against_rows(rows, manifest)["status"] == "PASS_REBUILT_COMPONENT_SPLIT"
    forged = copy.deepcopy(manifest)
    forged["row_assignments"]["r2"]["source_component_id"] = "fake_component"
    payload = copy.deepcopy(forged)
    payload.pop("manifest_sha256")
    from scripts.r3x.protocol import canonical_sha256
    forged["manifest_sha256"] = canonical_sha256(payload)
    with pytest.raises(SplitHoldError) as exc:
        verify_split_manifest_against_rows(rows, forged)
    assert exc.value.code == "HOLD_SPLIT_AUTHORITY_MISMATCH"
    stale = copy.deepcopy(manifest)
    stale["status"] = "FROZEN"
    with pytest.raises(SplitHoldError) as exc:
        verify_split_manifest_against_rows(rows, stale)
    assert exc.value.code == "HOLD_STALE_SPLIT_MANIFEST_HASH"


def test_rebuilt_split_rejects_disjoint_but_seed_inconsistent_repartition():
    rows = _independent_rows(12)
    manifest = build_split_manifest(rows, seed=11)
    forged = copy.deepcopy(manifest)
    by_partition: dict[str, str] = {}
    for assignment in forged["row_assignments"].values():
        by_partition.setdefault(assignment["partition"], assignment["source_component_id"])
    left = by_partition["train"]
    right = by_partition["internal_test"]
    for assignment in forged["row_assignments"].values():
        if assignment["source_component_id"] == left:
            assignment["partition"] = "internal_test"
        elif assignment["source_component_id"] == right:
            assignment["partition"] = "train"
    # Same-cell component swapping preserves every declared aggregate and
    # disjointness; only deterministic seed replay can detect this forgery.
    assert audit_component_disjointness(forged)["status"] == "PASS_COMPONENT_DISJOINT"
    payload = copy.deepcopy(forged)
    payload.pop("manifest_sha256")
    from scripts.r3x.protocol import canonical_sha256
    forged["manifest_sha256"] = canonical_sha256(payload)
    with pytest.raises(SplitHoldError) as exc:
        verify_split_manifest_against_rows(rows, forged)
    assert exc.value.code == "HOLD_SPLIT_REPLAY_MISMATCH"


def test_old_split_without_algorithm_or_support_replay_fields_fails_closed():
    rows = _independent_rows(9)
    manifest = build_split_manifest(rows, seed=42)
    for field in (
        "assignment_algorithm_id",
        "assignment_algorithm_parameters",
        "support_policy",
        "support_check",
    ):
        manifest.pop(field)
    payload = copy.deepcopy(manifest)
    payload.pop("manifest_sha256")
    from scripts.r3x.protocol import canonical_sha256
    manifest["manifest_sha256"] = canonical_sha256(payload)
    with pytest.raises(SplitHoldError) as exc:
        verify_split_manifest_against_rows(rows, manifest)
    assert exc.value.code in {
        "HOLD_UNREPLAYABLE_SPLIT_ALGORITHM",
        "HOLD_UNREPLAYABLE_SPLIT_SCHEMA",
    }


def test_rebuilt_split_rejects_self_hashed_tampered_component_counts():
    rows = _independent_rows(9)
    manifest = build_split_manifest(rows, seed=42)
    manifest["partition_component_counts"]["train"] += 1
    payload = copy.deepcopy(manifest)
    payload.pop("manifest_sha256")
    from scripts.r3x.protocol import canonical_sha256
    manifest["manifest_sha256"] = canonical_sha256(payload)
    with pytest.raises(SplitHoldError) as exc:
        verify_split_manifest_against_rows(rows, manifest)
    assert exc.value.code == "HOLD_SPLIT_REPLAY_MISMATCH"


def test_publication_manifest_rejects_empty_or_degenerate_cells():
    rows = [
        _row("r1", "AAAA", "CAAA"),
        _row("r2", "GGGG", "UGGG"),
    ]
    with pytest.raises(SplitHoldError) as exc:
        build_split_manifest(rows, seed=42, publication_claim=True)
    assert exc.value.code == "HOLD_MISSING_SPLIT_SUPPORT_POLICY"
    with pytest.raises(SplitHoldError) as exc:
        build_split_manifest(
            rows, seed=42, publication_claim=True,
            support_policy={"status": "FROZEN", "minimum_components_per_cell_partition": 1},
        )
    assert exc.value.code == "HOLD_DEGENERATE_PUBLICATION_CELL"
