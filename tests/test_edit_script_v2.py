from __future__ import annotations

import pytest

from data.utr_benchmark_v2.edit_script import (
    EditAction,
    EditScriptError,
    analyze_edit_script_ambiguity,
    apply_edit_script,
    canonical_edit_script,
    canonicalize_edit_script,
)


def test_dynamic_coordinates_apply_sub_ins_del_and_stop() -> None:
    actions = [
        EditAction("SUB", 1, ref="C", alt="U"),
        EditAction("INS", 2, alt="C"),
        EditAction("DEL", 3, ref="G"),
        EditAction("STOP"),
    ]
    assert apply_edit_script("ACGU", actions) == "AUCU"


def test_actions_after_stop_fail_closed() -> None:
    with pytest.raises(EditScriptError, match="after STOP"):
        apply_edit_script(
            "ACGU",
            [
                EditAction("STOP"),
                EditAction("SUB", 0, ref="A", alt="G"),
            ],
        )


def test_cycle_back_to_a_previous_state_fails_closed() -> None:
    with pytest.raises(EditScriptError, match="cycle"):
        apply_edit_script(
            "ACGU",
            [
                EditAction("SUB", 0, ref="A", alt="G"),
                EditAction("SUB", 0, ref="G", alt="A"),
            ],
        )


@pytest.mark.parametrize(
    ("action", "message"),
    [
        (EditAction("SUB", 0, ref="A", alt="A"), "no-op"),
        (EditAction("SUB", 0, ref="C", alt="G"), "reference allele"),
        (EditAction("SUB", 4, ref="A", alt="G"), "out of bounds"),
        (EditAction("INS", 0, alt=""), "non-empty"),
        (EditAction("INS", 5, alt="A"), "out of bounds"),
        (EditAction("INS", 0, ref="A", alt="G"), "empty reference"),
        (EditAction("DEL", 0, ref=""), "non-empty"),
        (EditAction("DEL", 0, ref="C"), "reference allele"),
        (EditAction("DEL", 3, ref="UA"), "out of bounds"),
        (EditAction("STOP", 0), "position"),
        (EditAction("SUB", 0, ref="A", alt="T"), "RNA alphabet"),
    ],
)
def test_invalid_actions_fail_closed(action: EditAction, message: str) -> None:
    with pytest.raises(EditScriptError, match=message):
        apply_edit_script("ACGU", [action])


def test_canonical_script_is_deterministic_minimal_and_roundtrips() -> None:
    first = canonical_edit_script("ACGU", "AGCUU")
    second = canonical_edit_script("ACGU", "AGCUU")
    assert first == second
    assert apply_edit_script("ACGU", first) == "AGCUU"

    result = canonicalize_edit_script("ACGU", "AGCUU")
    assert apply_edit_script("ACGU", result) == "AGCUU"
    assert result["actions"] == [action.to_dict() for action in first]
    assert result["minimal_edit_count"] == 2
    assert result["canonical_action_count"] == 2
    assert result["coordinate_system"] == "0_based_dynamic_state"


def test_adjacent_indels_are_canonicalized_as_one_dynamic_event() -> None:
    deletion = canonicalize_edit_script("AACCGU", "AGU")
    assert deletion["minimal_edit_count"] == 3
    assert deletion["canonical_action_count"] == 1
    assert apply_edit_script("AACCGU", deletion) == "AGU"

    insertion = canonicalize_edit_script("ACGU", "AAGCGU")
    assert insertion["minimal_edit_count"] == 2
    assert insertion["canonical_action_count"] == 1
    assert apply_edit_script("ACGU", insertion) == "AAGCGU"


def test_repeated_sequence_indel_ambiguity_is_counted_and_classified() -> None:
    result = canonicalize_edit_script("AAAA", "AAA")
    assert result["minimal_edit_count"] == 1
    assert result["equivalent_minimal_script_count"] == 4
    assert result["path_ambiguity"] is True
    assert result["ambiguity_category"] == "repeat_indel_coordinate"
    # Match-first tie-breaking gives a stable rightmost deletion anchor.
    assert result["actions"] == [
        {"op": "DEL", "pos": 3, "ref": "A", "alt": ""},
    ]
    assert apply_edit_script("AAAA", result) == "AAA"
    assert analyze_edit_script_ambiguity("AAAA", "AAA") == result


def test_nonrepeated_indel_has_unique_minimal_alignment() -> None:
    result = canonicalize_edit_script("ACGU", "ACG")
    assert result["equivalent_minimal_script_count"] == 1
    assert result["path_ambiguity"] is False
    assert result["ambiguity_category"] == "unique"
    assert apply_edit_script("ACGU", result) == "ACG"


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("acgu", "ACGU"),
        ("ACGT", "ACGU"),
        ("ACGN", "ACGU"),
    ],
)
def test_noncanonical_sequence_alphabet_fails_closed(
    source: str, candidate: str
) -> None:
    with pytest.raises(EditScriptError):
        canonicalize_edit_script(source, candidate)
