from __future__ import annotations

from collections import Counter

import pytest
import torch

from core.route2_xeditcritic_training_data_v3 import (
    SqrtTaskStudySourcePassSamplerV3,
    XEditCriticRecordV3,
    build_exact_source_task_candidate_bundle_permutation,
    capped_sqrt_task_allocations,
    different_source_group_pairwise_logistic_loss,
    records_from_projection_rows,
)


def _record(index: int, task: str, study: str, group: str, source: str = "AAAA") -> XEditCriticRecordV3:
    candidate = "ACAA" if index % 2 else "AGAA"
    return XEditCriticRecordV3(
        record_id=f"r{index}", split="TRAIN", source=source, candidate=candidate,
        edits=((1, "A", candidate[1]),), target=float(index), task=task,
        study=study, source_group=group, assay="a", context="c", region=0,
        quantity="q", measurement="m", numerator="n", denominator="d",
    )


def test_projection_conversion_rejects_edit_bundle_drift() -> None:
    row = {
        "canonical_record_id": "r", "split": "TRAIN", "source_sequence": "AAAA",
        "candidate_sequence": "ACAA", "source_relative_edits": [
            {"position": 2, "source_base": "A", "candidate_base": "C"}
        ],
        "direction_normalized_delta": 1.0, "task_id": "t", "study_unit_id": "s",
        "source_group_id": "g", "assay_id": "a", "biological_context_id": "c",
        "region_id": 0, "endpoint_descriptor": {
            "quantity_family": "q", "measurement_form": "m",
            "numerator_family": None, "denominator_family": None,
        },
    }
    with pytest.raises(Exception, match="edit bundle"):
        records_from_projection_rows([row])


def test_sqrt_allocation_redistributes_after_small_task_hits_repeat_cap() -> None:
    allocation = capped_sqrt_task_allocations(
        {"small": 2, "medium": 8, "large": 90}, draw_count=100, repeat_cap=4
    )
    assert sum(allocation.values()) == 100
    assert allocation["small"] == 8
    assert allocation["medium"] <= 32
    assert allocation["large"] <= 360
    assert allocation["medium"] / 8 > allocation["large"] / 90


def test_pass_sampler_is_task_homogeneous_and_never_repeats_a_row_over_four_times() -> None:
    records = [
        _record(index, "small" if index < 2 else "large", f"s{index % 2}", f"g{index % 5}")
        for index in range(30)
    ]
    sampler = SqrtTaskStudySourcePassSamplerV3(records, batch_size=4, seed=17)
    batches = sampler.batches_for_pass()
    counts = Counter(index for batch in batches for index in batch)
    assert sum(counts.values()) == len(records)
    assert max(counts.values()) <= 4
    assert all(len({records[index].task for index in batch}) == 1 for batch in batches)


def test_complete_candidate_bundle_permutation_stays_inside_exact_source_task() -> None:
    records = [_record(index, "t", "s", f"g{index}") for index in range(4)]
    overrides, summary = build_exact_source_task_candidate_bundle_permutation(records, seed=9)
    by_id = {record.record_id: record for record in records}
    for recipient_id, donor_id in overrides.items():
        assert by_id[recipient_id].source == by_id[donor_id].source
        assert by_id[recipient_id].task == by_id[donor_id].task
        assert recipient_id != donor_id
    assert summary["complete_candidate_bundle_permuted"] is True


def test_pairwise_loss_uses_only_different_source_groups() -> None:
    prediction = torch.tensor([0.0, 100.0, 1.0], requires_grad=True)
    target = torch.tensor([0.0, 1.0, 2.0])
    # The extreme same-group pair 0/1 must not enter; only pairs with index 2 do.
    loss = different_source_group_pairwise_logistic_loss(
        prediction, target, ["g", "g", "h"], ["t", "t", "t"]
    )
    assert loss is not None
    expected = torch.stack(
        (
            torch.nn.functional.softplus(-(prediction[2] - prediction[0])),
            torch.nn.functional.softplus(-(prediction[2] - prediction[1])),
        )
    ).mean()
    assert torch.allclose(loss, expected)


def test_pairwise_loss_rejects_mixed_task_batch() -> None:
    with pytest.raises(Exception, match="not task homogeneous"):
        different_source_group_pairwise_logistic_loss(
            torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]), ["a", "b"], ["x", "y"]
        )
