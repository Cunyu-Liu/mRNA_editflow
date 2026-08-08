from __future__ import annotations

import copy
import hashlib
import json

import pytest

from scripts.r3x.controls import (
    pairing_shuffle,
    permute_context_labels,
    position_shuffle_within_source,
    source_context_shuffle_within_action,
    target_token_shuffle_within_cell,
)
from scripts.r3x.features import validated_substitutions


def make_row(record_id: str, source: str, pos: int, alt: str, *, context: str = "c1") -> dict:
    ref = source[pos]
    candidate = source[:pos] + alt + source[pos + 1 :]
    return {
        "r3_input_row_sha256": hashlib.sha256(record_id.encode("utf-8")).hexdigest(),
        "record_id": record_id,
        "partition": "train",
        "study": "s",
        "estimand_id": "estimand",
        "endpoint_id": "e",
        "context_id_v2": context,
        "context_id": context,
        "publication_context_id": context,
        "region": "5UTR",
        "source_sequence": source,
        "normalized_source_sequence": source,
        "candidate_sequence": candidate,
        "normalized_candidate_sequence": candidate,
        "edit_list": [{"op": "SUB", "pos": pos, "ref": ref, "alt": alt}],
        "actual_edit_count": 1,
        "source_component_id": f"comp_{record_id}",
        "delta": float(len(record_id)),
    }


def test_pairing_shuffle_fails_closed_instead_of_emitting_unreplayable_rows() -> None:
    rows = [
        make_row("a", "AACGAA", 2, "U"),
        make_row("b", "AACCAA", 2, "A"),
        make_row("c", "AAGCAA", 2, "U"),
    ]
    one = pairing_shuffle(rows, seed=23)
    two = pairing_shuffle(list(reversed(rows)), seed=23)
    assert one.status == "NOT_RUN_WITH_REASON"
    assert one.rows == two.rows
    assert one.rows == []
    assert one.manifest["input_sha256"] == two.manifest["input_sha256"]
    assert "replay-required" in one.reason
    assert "PRE_TEST_FREEZE_HOLD" in one.manifest["reason"]
    assert "NOT_RUN_IS_NOT_PASS" in one.manifest["reason"]
    assert (
        "PREREGISTER_AND_FREEZE_A_BROKEN_PAIR_TENSORIZER_AND_PROTOCOL_REVISION_THEN_RERUN"
        in one.manifest["reason"]
    )


def test_context_permutation_never_changes_sequences_or_targets() -> None:
    rows = [
        make_row("a", "AACGAA", 2, "U", context="c1"),
        make_row("b", "AACCAA", 2, "A", context="c2"),
    ]
    out = permute_context_labels(rows, seed=11)
    assert out.status == "RUN"
    original = {r["record_id"]: r for r in rows}
    for row in out.rows:
        before = original[row["record_id"]]
        assert row["source_sequence"] == before["source_sequence"]
        assert row["candidate_sequence"] == before["candidate_sequence"]
        assert row["delta"] == before["delta"]
        assert row["context_id_v2"] != before["context_id_v2"]
        assert row["publication_context_id"] == row["context_id_v2"] == row["context_id"]


def test_source_context_shuffle_preserves_action_and_exact_replay() -> None:
    rows = [
        make_row("a", "AACGAA", 2, "U"),
        make_row("b", "UACCUU", 2, "U"),
    ]
    out = source_context_shuffle_within_action(rows, seed=42)
    assert out.status == "RUN"
    original_source = {r["record_id"]: r["source_sequence"] for r in rows}
    for row in out.rows:
        assert row["source_sequence"] != original_source[row["record_id"]]
        assert validated_substitutions(row) == [(2, "C", "U")]
        assert row["control_donor_record_id"] != row["record_id"]
        assert row["normalized_source_sequence"] == row["source_sequence"]
        assert row["normalized_candidate_sequence"] == row["candidate_sequence"]


def test_position_shuffle_preserves_ref_alt_and_replay() -> None:
    rows = [make_row("a", "AACCAA", 1, "G")]
    out = position_shuffle_within_source(rows, seed=67)
    assert out.status == "RUN"
    edits = validated_substitutions(out.rows[0])
    assert edits[0][0] != 1
    assert edits[0][1:] == ("A", "G")
    assert out.rows[0]["normalized_candidate_sequence"] == out.rows[0]["candidate_sequence"]


def test_target_shuffle_preserves_positions_and_replay() -> None:
    rows = [
        make_row("a", "AACGAA", 2, "U"),
        make_row("b", "AACCAA", 2, "A"),
    ]
    out = target_token_shuffle_within_cell(rows, seed=101)
    assert out.status == "RUN"
    before = {r["record_id"]: validated_substitutions(r)[0] for r in rows}
    for row in out.rows:
        edit = validated_substitutions(row)[0]
        assert edit[:2] == before[row["record_id"]][:2]
        assert edit[2] != before[row["record_id"]][2]
        assert row["normalized_candidate_sequence"] == row["candidate_sequence"]


def test_insufficient_support_is_explicit_not_silent_identity() -> None:
    singleton = [make_row("a", "AACGAA", 2, "U")]
    out = pairing_shuffle(singleton, seed=23)
    assert out.status == "NOT_RUN_WITH_REASON"
    assert out.rows == []
    assert "PRE_TEST_FREEZE_HOLD" in out.manifest["reason"]


def test_negative_control_rejects_non_train_input_instead_of_self_attesting() -> None:
    row = make_row("a", "AACGAA", 2, "U")
    row["partition"] = "internal_test"
    row["source_sequence"] = "not-a-sequence"
    row["normalized_source_sequence"] = "also-not-a-sequence"
    with pytest.raises(ValueError, match="train partition only"):
        pairing_shuffle([row], seed=23)


def test_negative_control_rejects_already_perturbed_input() -> None:
    row = make_row("a", "AACGAA", 2, "U")
    row["control_name"] = "FORGED_PRIOR_CONTROL"
    with pytest.raises(ValueError, match="unperturbed train rows"):
        position_shuffle_within_source([row], seed=23)


@pytest.mark.parametrize(
    ("left", "right", "changed"),
    [
        ("source_sequence", "normalized_source_sequence", "UUUUUU"),
        ("candidate_sequence", "normalized_candidate_sequence", "UUUUUU"),
        ("publication_context_id", "context_id_v2", "different"),
    ],
)
def test_conflicting_model_visible_aliases_are_strictly_rejected(
    left: str, right: str, changed: str
) -> None:
    row = make_row("a", "AACGAA", 2, "U")
    assert row[left] == row[right]
    row[right] = changed
    with pytest.raises(ValueError, match="conflicting negative-control"):
        pairing_shuffle([row], seed=23)


def test_normalized_publication_only_rows_are_canonicalized_for_outputs() -> None:
    rows = [
        make_row("a", "AACGAA", 2, "U", context="c1"),
        make_row("b", "AACCAA", 2, "A", context="c2"),
    ]
    for row in rows:
        del row["source_sequence"]
        del row["candidate_sequence"]
        del row["context_id_v2"]
        del row["context_id"]
    out = permute_context_labels(rows, seed=11)
    assert out.status == "RUN"
    for row in out.rows:
        assert row["normalized_source_sequence"] == row["source_sequence"]
        assert row["normalized_candidate_sequence"] == row["candidate_sequence"]
        assert row["publication_context_id"] == row["context_id_v2"] == row["context_id"]


def test_run_controls_preserve_every_target_identity_and_train_field() -> None:
    fixtures_and_transforms = [
        (
            [
                make_row("ca", "AACGAA", 2, "U", context="c1"),
                make_row("cb", "AACCAA", 2, "A", context="c2"),
            ],
            permute_context_labels,
        ),
        (
            [make_row("sa", "AACGAA", 2, "U"), make_row("sb", "UACCUU", 2, "U")],
            source_context_shuffle_within_action,
        ),
        ([make_row("pa", "AACCAA", 1, "G")], position_shuffle_within_source),
        (
            [make_row("ta", "AACGAA", 2, "U"), make_row("tb", "AACCAA", 2, "A")],
            target_token_shuffle_within_cell,
        ),
    ]
    for rows, transform in fixtures_and_transforms:
        for index, row in enumerate(rows):
            row.update(
                {
                    "delta": float(index) - 0.5,
                    "target": float(index) + 1.0,
                    "label": f"label-{index}",
                    "source_value": 2.0 + index,
                    "candidate_value": 3.0 + index,
                    "custom_outcome": {"replicate": index, "value": 4.0 + index},
                }
            )
        before = {row["r3_input_row_sha256"]: copy.deepcopy(row) for row in rows}
        out = transform(rows, seed=19)
        assert out.status == "RUN"
        assert [row["r3_input_row_sha256"] for row in out.rows] == sorted(before)
        for row in out.rows:
            original = before[row["r3_input_row_sha256"]]
            assert row["record_id"] == original["record_id"]
            assert row["partition"] == "train"
            for field in (
                "delta", "target", "label", "source_value", "candidate_value", "custom_outcome"
            ):
                assert row[field] == original[field]


def test_run_manifest_binds_canonical_order_and_exact_output_bytes() -> None:
    rows = [
        make_row("a", "AACGAA", 2, "U", context="c1"),
        make_row("b", "AACCAA", 2, "A", context="c2"),
    ]
    first = permute_context_labels(rows, seed=11)
    reverse = permute_context_labels(list(reversed(rows)), seed=11)
    assert first.rows == reverse.rows
    assert first.manifest["input_sha256"] == reverse.manifest["input_sha256"]
    assert first.manifest["output_sha256"] == reverse.manifest["output_sha256"]
    encoded = json.dumps(
        first.rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == first.manifest["output_sha256"]
    ordered_ids = [row["r3_input_row_sha256"] for row in first.rows]
    assert ordered_ids == sorted(ordered_ids)
    ordered_ids_sha = hashlib.sha256(
        json.dumps(ordered_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert ordered_ids_sha == first.manifest["ordered_input_row_ids_sha256"]
