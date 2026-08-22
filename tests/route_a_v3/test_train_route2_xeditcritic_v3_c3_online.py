from __future__ import annotations

import torch

from scripts.route_a_v3.train_route2_xeditcritic_v3_c3_online import (
    microbatch_indices,
    select_batch_rows,
)


def test_effective_batch_is_partitioned_without_drop_or_repeat() -> None:
    chunks = microbatch_indices(32, 3)
    assert [index for chunk in chunks for index in chunk] == list(range(32))
    assert max(map(len, chunks)) == 3


def test_batch_selection_keeps_tensor_and_metadata_alignment() -> None:
    batch = {
        "values": torch.tensor([[0], [1], [2], [3]]),
        "record_ids": ["a", "b", "c", "d"],
        "constant": "x",
    }
    selected = select_batch_rows(batch, [3, 1])
    assert selected["values"].flatten().tolist() == [3, 1]
    assert selected["record_ids"] == ["d", "b"]
    assert selected["constant"] == "x"
