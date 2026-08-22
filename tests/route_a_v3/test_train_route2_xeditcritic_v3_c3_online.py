from __future__ import annotations

import torch

from scripts.route_a_v3.train_route2_xeditcritic_v3_c3_online import (
    microbatch_indices,
    select_batch_rows,
    singleton_online_pair_loss_sum_v3,
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


def test_ranking_pair_uses_two_separate_singleton_encoder_forwards() -> None:
    class RecordingEncoder:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def forward_cache_anchored(self, batch):
            self.batch_sizes.append(len(batch["record_ids"]))
            return batch

    class IdentityModel:
        def __call__(self, batch):
            return {"mean": batch["prediction"]}

    raw_batch = {
        "record_ids": ["left", "right"],
        "prediction": torch.tensor([0.25, -0.50], requires_grad=True),
        "scaled_target": torch.tensor([1.0, -1.0]),
    }
    encoder = RecordingEncoder()
    loss = singleton_online_pair_loss_sum_v3(
        IdentityModel(),
        encoder,
        raw_batch,
        [(0, 1)],
        torch.device("cpu"),
    )
    expected = torch.nn.functional.softplus(torch.tensor(-0.75))
    assert torch.allclose(loss, expected)
    assert encoder.batch_sizes == [1, 1]
    loss.backward()
    assert raw_batch["prediction"].grad is not None
