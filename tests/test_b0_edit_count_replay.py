from __future__ import annotations

import copy

import pytest

from data.utr_benchmark_v2.edit_script import canonicalize_edit_script
from data.utr_benchmark_v2.split_graph import SplitGraphError
from data.utr_benchmark_v2.split_graph import frozen_edit_script_state_closure


def test_frozen_replay_uses_d1_edit_count_not_character_distance() -> None:
    canonical = canonicalize_edit_script("AAAAACCCC", "AAAAA")
    record = {
        "source_sequence": "AAAAACCCC",
        "candidate_sequence": "AAAAA",
        "edit_script": canonical["actions"],
        "edit_count": canonical["canonical_action_count"],
        "edit_distance": canonical["minimal_edit_count"],
        "intermediate_sequences": [],
        "trajectory_observed": False,
    }
    assert record["edit_count"] == 1
    assert record["edit_distance"] == 4

    closure = frozen_edit_script_state_closure(record)
    assert closure.reachable_transition_count == record["edit_count"]

    malformed = copy.deepcopy(record)
    malformed["edit_count"] = 2
    with pytest.raises(
        SplitGraphError,
        match="edit_count does not equal canonical edit-script length",
    ):
        frozen_edit_script_state_closure(malformed)
