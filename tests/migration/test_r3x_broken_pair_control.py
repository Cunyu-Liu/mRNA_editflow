from __future__ import annotations

import copy
import hashlib
import itertools
import json
import sys
from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from scripts.r3x.broken_pair_control import (
    BrokenPairControlError,
    ELIGIBLE_METHOD_IDS,
    FROZEN_SEEDS,
    INACTIVE_STATUS,
    REQUIRED_FUTURE_AUTHORITY_BINDINGS,
    build_broken_pair_candidate,
    build_candidate_policy,
    candidate_policy_sha256,
    derive_target_free_row_id,
    replay_broken_pair_manifest,
    replay_candidate_policy,
    validate_broken_pair_manifest,
    validate_candidate_policy,
    write_broken_pair_manifest_exclusive,
    write_candidate_policy_exclusive,
)


def _row(
    label: str,
    source: str,
    candidate: str,
    target: int | float,
    *,
    context: str = "ordinary_context_1",
    edit_count: int = 1,
) -> dict[str, Any]:
    if not label:
        raise ValueError("test label must be non-empty")
    condition = {
        "study": "ordinary_study_1",
        "estimand_id": "mean_delta",
        "endpoint": "mrna_abundance_delta",
        "publication_context_id": context,
        "region": "5UTR",
        "edit_count": edit_count,
    }
    return {
        "row_id": derive_target_free_row_id(
            source_sequence=source,
            candidate_sequence=candidate,
            condition=condition,
        ),
        "partition": "train",
        "source_sequence": source,
        "candidate_sequence": candidate,
        "target": target,
        "condition": condition,
    }


def _supported_rows() -> list[dict[str, Any]]:
    # Every original source/candidate pair is a one-substitution ordinary
    # train example.  Cross-row candidates are deliberately very different,
    # so the broken tensor pair need not replay the receiver's original edit.
    return [
        _row("a", "AACGAA", "AAUGAA", -1.25),
        _row("b", "UACCUU", "UAGCUU", 0.0),
        _row("c", "GGAAGG", "GGACGG", 2),
        _row("d", "CCAACC", "CCAGCC", 3.5),
    ]


def _build(
    rows: list[dict[str, Any]] | None = None,
    *,
    method_id: str = "FULL_PAIR_UNRESTRICTED_SIAMESE",
    seed: int = 23,
) -> dict[str, Any]:
    return build_broken_pair_candidate(
        _supported_rows() if rows is None else rows,
        method_id=method_id,
        seed=seed,
    )


def _target_key_paths(value: object, path: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key == "target":
                paths.append(child_path)
            paths.extend(_target_key_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_target_key_paths(child, f"{path}[{index}]"))
    return paths


@pytest.mark.parametrize("method_id", ELIGIBLE_METHOD_IDS)
def test_candidate_tensorizer_builds_only_for_preregistered_paired_methods(
    method_id: str,
) -> None:
    rows = _supported_rows()
    manifest = _build(rows, method_id=method_id)

    assert manifest["status"] == INACTIVE_STATUS == "PROPOSED_NOT_ACTIVE"
    assert manifest["candidate_execution_status"] == "CANDIDATE_TENSORS_BUILT"
    assert manifest["method_id"] == method_id
    assert manifest["partition_scope"] == "ORDINARY_TRAIN_PARTITION_ONLY"
    assert manifest["hold_reason"] is None
    assert manifest["can_satisfy_pre_test_freeze"] is False
    assert manifest["authority_binding_status"] == "NOT_BOUND_CANDIDATE_ONLY"
    assert manifest["required_future_authority_bindings"] == list(
        REQUIRED_FUTURE_AUTHORITY_BINDINGS
    )
    assert manifest["required_future_activation"] == (
        "INDEPENDENT_FROZEN_VERSIONED_BROKEN_PAIR_POLICY_AUTHORITY_"
        "AND_PROTOCOL_REVISION"
    )
    assert manifest["development_access"] is False
    assert manifest["internal_test_access"] is False
    assert manifest["sealed_access"] is False
    assert manifest["outcome_mutation"] is False
    assert manifest["original_edit_metadata_tensor_input"] is False

    originals = {row["row_id"]: row for row in rows}
    tensor_rows = manifest["tensor_rows"]
    assert [row["receiver_row_id"] for row in tensor_rows] == sorted(originals)
    assert {row["donor_row_id"] for row in tensor_rows} == set(originals)
    assert sorted(
        row["candidate_sequence"] for row in manifest["input_rows"]
    ) == sorted(row["tensor_input"]["candidate_sequence"] for row in tensor_rows)

    for tensor_row in tensor_rows:
        receiver = originals[tensor_row["receiver_row_id"]]
        donor = originals[tensor_row["donor_row_id"]]
        tensor_input = tensor_row["tensor_input"]
        assert tensor_row["receiver_row_id"] != tensor_row["donor_row_id"]
        assert tensor_input["row_id"] == receiver["row_id"]
        assert tensor_input["partition"] == "train"
        assert tensor_input["source_sequence"] == receiver["source_sequence"]
        assert tensor_input["candidate_sequence"] == donor["candidate_sequence"]
        assert tensor_input["candidate_sequence"] != receiver["candidate_sequence"]
        assert tensor_input["target"] == receiver["target"]
        assert tensor_input["condition"] == receiver["condition"]
        assert set(tensor_input) == {
            "row_id",
            "partition",
            "source_sequence",
            "candidate_sequence",
            "target",
            "condition",
        }
        assert "edit_list" not in tensor_input
        assert len(tensor_input["source_sequence"]) == len(
            tensor_input["candidate_sequence"]
        )
        assert set(tensor_input["source_sequence"]) <= set("ACGU")
        assert set(tensor_input["candidate_sequence"]) <= set("ACGU")
        assert (
            sum(
                left != right
                for left, right in zip(
                    tensor_input["source_sequence"],
                    tensor_input["candidate_sequence"],
                    strict=True,
                )
            )
            > receiver["condition"]["edit_count"]
        )
        assert tensor_row["audit"]["source_candidate_replay_required"] is False
        assert tensor_row["audit"]["edit_metadata_tensor_input"] is False


def test_matching_and_artifact_are_invariant_to_input_row_order() -> None:
    rows = _supported_rows()
    forward = _build(rows, seed=42)
    reverse = _build(list(reversed(rows)), seed=42)
    rotated = _build(rows[2:] + rows[:2], seed=42)

    assert forward == reverse == rotated
    assert [row["row_id"] for row in forward["input_rows"]] == sorted(
        row["row_id"] for row in rows
    )
    assert [row["receiver_row_id"] for row in forward["tensor_rows"]] == sorted(
        row["row_id"] for row in rows
    )


@pytest.mark.parametrize("seed", FROZEN_SEEDS)
def test_every_fixed_seed_is_deterministic_and_replayable(seed: int) -> None:
    first = _build(seed=seed)
    second = _build(seed=seed)
    assert first == second
    assert validate_broken_pair_manifest(first) == first


def test_receiver_target_is_row_level_only_and_never_summarized() -> None:
    rows = _supported_rows()
    manifest = _build(rows)
    original_targets = {row["row_id"]: row["target"] for row in rows}

    assert {
        row["row_id"]: row["target"] for row in manifest["input_rows"]
    } == original_targets
    assert {
        row["receiver_row_id"]: row["tensor_input"]["target"]
        for row in manifest["tensor_rows"]
    } == original_targets
    assert all(
        "target" not in key.lower() for summary in manifest["strata"] for key in summary
    )
    assert set(_target_key_paths(manifest)) == {
        *{
            f"$.input_rows[{index}].target"
            for index in range(len(manifest["input_rows"]))
        },
        *{
            f"$.tensor_rows[{index}].tensor_input.target"
            for index in range(len(manifest["tensor_rows"]))
        },
    }
    assert not any(
        token in key.lower()
        for key in manifest
        for token in (
            "target_mean",
            "target_std",
            "target_distribution",
            "target_summary",
        )
    )


def test_changing_only_targets_cannot_change_donor_assignment() -> None:
    rows = _supported_rows()
    changed_targets = copy.deepcopy(rows)
    for index, row in enumerate(changed_targets):
        row["target"] = 1000.0 + index

    original = _build(rows, seed=67)
    changed = _build(changed_targets, seed=67)
    original_pairs = [
        (row["receiver_row_id"], row["donor_row_id"]) for row in original["tensor_rows"]
    ]
    changed_pairs = [
        (row["receiver_row_id"], row["donor_row_id"]) for row in changed["tensor_rows"]
    ]
    assert original_pairs == changed_pairs
    assert original["permutation_sha256"] == changed["permutation_sha256"]
    assert original["tensor_rows_sha256"] != changed["tensor_rows_sha256"]


def test_arbitrary_or_outcome_bearing_row_id_relabel_is_rejected() -> None:
    row = _supported_rows()[0]
    row["row_id"] = hashlib.sha256(b"caller-selected-id").hexdigest()
    with pytest.raises(BrokenPairControlError) as exc_info:
        _build([row])
    assert exc_info.value.code == "HOLD_BROKEN_PAIR_ROW_IDENTITY"


class _PartitionOnlyProbe(Mapping[str, object]):
    """Mapping that fails if a caller touches anything except partition."""

    def __init__(self, partition: str):
        self.partition = partition
        self.accessed: list[str] = []

    def __getitem__(self, key: str) -> object:
        self.accessed.append(key)
        if key == "partition":
            return self.partition
        raise AssertionError(f"read prohibited field before partition reject: {key}")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("iterated row before partition reject")

    def __len__(self) -> int:
        raise AssertionError("measured row before partition reject")


class _SpoofTrainPartition:
    def __eq__(self, other: object) -> bool:
        return other == "train"


@pytest.mark.parametrize("partition", ["development", "internal_test", "sealed"])
def test_nontrain_partition_is_rejected_before_any_sequence_or_target_access(
    partition: str,
) -> None:
    probe = _PartitionOnlyProbe(partition)
    with pytest.raises(BrokenPairControlError) as exc_info:
        build_broken_pair_candidate(
            [probe],
            method_id="FULL_PAIR_UNRESTRICTED_SIAMESE",
            seed=23,
        )
    assert exc_info.value.code == "HOLD_BROKEN_PAIR_TRAIN_ONLY"
    assert probe.accessed == ["partition"]


def test_train_partition_must_be_an_exact_json_string() -> None:
    row = _supported_rows()[0]
    row["partition"] = _SpoofTrainPartition()
    with pytest.raises(BrokenPairControlError) as exc_info:
        _build([row])
    assert exc_info.value.code == "HOLD_BROKEN_PAIR_TRAIN_ONLY"


def test_protected_member_disguised_as_train_is_rejected() -> None:
    row = _supported_rows()[0]
    row["condition"]["study"] = "GSE246381"
    with pytest.raises(BrokenPairControlError) as exc_info:
        _build([row])
    assert exc_info.value.code == "HOLD_BROKEN_PAIR_ORDINARY_ONLY"


@pytest.mark.parametrize(
    "extra_field",
    [
        "edit_list",
        "delta",
        "label",
        "normalized_source_sequence",
        "normalized_candidate_sequence",
        "source_value",
        "candidate_value",
    ],
)
def test_input_schema_rejects_edit_metadata_aliases_and_extra_outcomes(
    extra_field: str,
) -> None:
    row = _supported_rows()[0]
    row[extra_field] = [] if extra_field == "edit_list" else "forbidden"
    with pytest.raises(BrokenPairControlError) as exc_info:
        _build([row])
    assert exc_info.value.code == "HOLD_BROKEN_PAIR_SCHEMA"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("source_sequence", "aacgaa", "HOLD_BROKEN_PAIR_SEQUENCE"),
        ("candidate_sequence", "AATGAA", "HOLD_BROKEN_PAIR_SEQUENCE"),
        ("candidate_sequence", "AAUGA", "HOLD_BROKEN_PAIR_SEQUENCE"),
        ("target", True, "HOLD_BROKEN_PAIR_TARGET"),
        ("target", float("nan"), "HOLD_BROKEN_PAIR_TARGET"),
        ("target", 10**10000, "HOLD_BROKEN_PAIR_TARGET"),
    ],
    ids=[
        "lowercase-source",
        "dna-candidate",
        "unequal-length",
        "boolean-target",
        "nan-target",
        "oversized-integer-target",
    ],
)
def test_invalid_sequence_or_target_fails_closed(
    field: str, value: object, code: str
) -> None:
    row = _supported_rows()[0]
    row[field] = value
    with pytest.raises(BrokenPairControlError) as exc_info:
        _build([row])
    assert exc_info.value.code == code


@pytest.mark.parametrize(
    "method_id",
    [
        "B_SPARSE_EDIT_SET_ADDITIVE",
        "B_SPARSE_EDIT_SET_INTERACTION",
        "B_SPARSE_EDIT_SET_INTERACTION_ONLY_IF_ACTION_SUPPORT_PUBLICATION_QUALIFIED",
        "SPARSE_EDIT_SET",
        "UNREGISTERED_SPARSE_EDIT_SET_MODEL",
    ],
)
def test_b_and_sparse_edit_set_families_are_hard_rejected(method_id: str) -> None:
    with pytest.raises(BrokenPairControlError) as exc_info:
        _build(method_id=method_id)
    assert exc_info.value.code == "HOLD_BROKEN_PAIR_B_SPARSE_EDIT_SET_PROHIBITED"


def test_unknown_method_and_unfrozen_seed_are_rejected() -> None:
    with pytest.raises(BrokenPairControlError) as method_exc:
        _build(method_id="UNREGISTERED_PAIRED_MODEL")
    assert method_exc.value.code == "HOLD_BROKEN_PAIR_METHOD"

    with pytest.raises(BrokenPairControlError) as seed_exc:
        _build(seed=19)
    assert seed_exc.value.code == "HOLD_BROKEN_PAIR_SEED"


def test_duplicate_candidate_support_without_perfect_matching_is_not_run() -> None:
    rows = [
        _row("dup-a", "AACGAA", "AAUGAA", -1.0),
        _row("dup-b", "AAGGAA", "AAUGAA", 0.0),
        _row("dup-c", "AAAGAA", "AAUGAA", 1.0),
        _row("unique", "UACCUU", "UAGCUU", 2.0),
    ]
    forward = _build(rows)
    reverse = _build(list(reversed(rows)))

    assert forward == reverse
    assert forward["status"] == "PROPOSED_NOT_ACTIVE"
    assert forward["candidate_execution_status"] == "NOT_RUN_WITH_REASON"
    assert forward["tensor_rows"] == []
    assert forward["tensor_rows_sha256"] == hashlib.sha256(b"[]").hexdigest()
    assert forward["output_candidate_marginal_sha256"] is None
    assert forward["permutation_sha256"] is None
    assert "PRE_TEST_FREEZE_HOLD" in forward["hold_reason"]
    assert "NOT_RUN_IS_NOT_PASS" in forward["hold_reason"]
    assert all(item["status"] == "NOT_RUN_WITH_REASON" for item in forward["strata"])
    assert all(
        item["output_candidate_marginal_sha256"] is None for item in forward["strata"]
    )
    assert all(item["permutation_sha256"] is None for item in forward["strata"])


def _hamming_sources(candidate: str, count: int, limit: int) -> list[str]:
    alternatives = {
        base: tuple(other for other in "ACGU" if other != base) for base in "ACGU"
    }
    output: list[str] = []
    for positions in itertools.combinations(range(len(candidate)), count):
        choices = [alternatives[candidate[position]] for position in positions]
        for replacements in itertools.product(*choices):
            sequence = list(candidate)
            for position, replacement in zip(positions, replacements, strict=True):
                sequence[position] = replacement
            output.append("".join(sequence))
            if len(output) == limit:
                return output
    raise AssertionError("test fixture lacks requested Hamming support")


def test_large_supported_stratum_uses_iterative_matching_not_python_recursion() -> None:
    first_candidate = "AAAAAAAA"
    second_candidate = "CCCCCCCC"
    rows = [
        _row(
            f"large-a-{index}",
            source,
            first_candidate,
            float(index),
            edit_count=3,
        )
        for index, source in enumerate(_hamming_sources(first_candidate, 3, 100))
    ]
    rows.extend(
        _row(
            f"large-c-{index}",
            source,
            second_candidate,
            float(index + 100),
            edit_count=3,
        )
        for index, source in enumerate(_hamming_sources(second_candidate, 3, 100))
    )

    prior_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(80)
        forward = _build(rows, seed=101)
        reverse = _build(list(reversed(rows)), seed=101)
    finally:
        sys.setrecursionlimit(prior_limit)

    assert forward == reverse
    assert forward["candidate_execution_status"] == "CANDIDATE_TENSORS_BUILT"
    assert len(forward["tensor_rows"]) == 200


def test_any_failed_stratum_holds_the_whole_control_without_partial_rows() -> None:
    rows = _supported_rows()[:2]
    rows.append(
        _row(
            "singleton-other-cell",
            "GGUUGG",
            "GGAUGG",
            4.0,
            context="ordinary_context_2",
        )
    )
    manifest = _build(rows)

    assert manifest["n_strata"] == 2
    assert manifest["candidate_execution_status"] == "NOT_RUN_WITH_REASON"
    assert manifest["tensor_rows"] == []
    assert all(item["status"] == "NOT_RUN_WITH_REASON" for item in manifest["strata"])
    assert any("atomic control held" in item["reason"] for item in manifest["strata"])
    assert all(
        item["output_candidate_marginal_sha256"] is None for item in manifest["strata"]
    )
    assert all(item["permutation_sha256"] is None for item in manifest["strata"])


def test_policy_is_exact_hash_bound_and_inactive() -> None:
    policy = build_candidate_policy()
    assert validate_candidate_policy(policy) == policy
    assert policy["status"] == "PROPOSED_NOT_ACTIVE"
    assert policy["activation_status"] == "NO_VERSIONED_POLICY_AUTHORITY_ACTIVE"
    assert policy["pre_test_freeze_effect"] == (
        "CANNOT_SATISFY_OR_ADVANCE_PRE_TEST_FREEZE"
    )
    assert policy["fixed_seeds"] == list(FROZEN_SEEDS)
    assert policy["eligible_method_ids"] == list(ELIGIBLE_METHOD_IDS)
    assert len(candidate_policy_sha256(policy)) == 64

    mutated = copy.deepcopy(policy)
    mutated["fixed_seeds"].append(999)
    with pytest.raises(BrokenPairControlError) as exc_info:
        validate_candidate_policy(mutated)
    assert exc_info.value.code == "HOLD_BROKEN_PAIR_POLICY_SCHEMA"


@pytest.mark.parametrize(
    "mutation",
    ["tensor_target", "policy_hash", "row_order", "extra_manifest_key"],
)
def test_manifest_tampering_cannot_replay(mutation: str) -> None:
    manifest = _build()
    tampered = copy.deepcopy(manifest)
    if mutation == "tensor_target":
        tampered["tensor_rows"][0]["tensor_input"]["target"] = 999.0
    elif mutation == "policy_hash":
        tampered["policy_sha256"] = "0" * 64
    elif mutation == "row_order":
        tampered["input_rows"] = list(reversed(tampered["input_rows"]))
    else:
        tampered["target_summary"] = {"mean": 999.0}

    with pytest.raises(BrokenPairControlError):
        validate_broken_pair_manifest(tampered)


def test_exclusive_policy_and_manifest_writers_replay_exact_canonical_bytes(
    tmp_path,
) -> None:
    policy = build_candidate_policy()
    policy_path = tmp_path / "broken_pair_policy.json"
    policy_sha = write_candidate_policy_exclusive(policy_path, policy)
    assert policy_sha == hashlib.sha256(policy_path.read_bytes()).hexdigest()
    assert replay_candidate_policy(policy_path, policy_sha) == policy
    with pytest.raises(BrokenPairControlError) as policy_overwrite:
        write_candidate_policy_exclusive(policy_path, policy)
    assert policy_overwrite.value.code == "HOLD_BROKEN_PAIR_ARTIFACT_IO"

    manifest = _build()
    manifest_path = tmp_path / "broken_pair_manifest.json"
    manifest_sha = write_broken_pair_manifest_exclusive(manifest_path, manifest)
    assert manifest_sha == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert replay_broken_pair_manifest(manifest_path, manifest_sha) == manifest
    with pytest.raises(BrokenPairControlError) as manifest_overwrite:
        write_broken_pair_manifest_exclusive(manifest_path, manifest)
    assert manifest_overwrite.value.code == "HOLD_BROKEN_PAIR_ARTIFACT_IO"

    with pytest.raises(BrokenPairControlError) as wrong_hash:
        replay_broken_pair_manifest(manifest_path, "0" * 64)
    assert wrong_hash.value.code == "HOLD_BROKEN_PAIR_ARTIFACT_IO"


def test_replay_rejects_noncanonical_manifest_bytes(tmp_path) -> None:
    manifest = _build()
    path = tmp_path / "noncanonical_broken_pair_manifest.json"
    path.write_bytes(json.dumps(manifest, sort_keys=False).encode("utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(BrokenPairControlError) as exc_info:
        replay_broken_pair_manifest(path, digest)
    assert exc_info.value.code == "HOLD_BROKEN_PAIR_NONCANONICAL"
