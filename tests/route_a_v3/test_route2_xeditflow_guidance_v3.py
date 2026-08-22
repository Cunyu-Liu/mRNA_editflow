from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from core.route2_xeditflow_guidance_v3 import (
    MatchedComputeRecordV2,
    XEditValueV3,
    beta_schedule_v3,
    deduplicate_terminal_candidates_v3,
    effective_sample_size_v3,
    potential_guided_rates_v3,
    soft_value_target_v3,
    stratified_resample_v3,
    uncertainty_penalized_reward_v3,
)


def test_reward_temperature_and_beta_are_frozen_and_numerically_stable() -> None:
    predictions = torch.tensor([[1.0, 2.0, 3.0]])
    reward = uncertainty_penalized_reward_v3(predictions, kappa=0.5)
    assert reward.item() == pytest.approx(2.0 - 0.5 * np.std([1.0, 2.0, 3.0]))
    rollouts = torch.tensor([[1000.0] * 8, [0.0, 1.0] * 4])
    target = soft_value_target_v3(rollouts, temperature=0.5)
    assert torch.isfinite(target).all()
    assert target[0].item() == pytest.approx(1000.0)
    assert beta_schedule_v3(torch.tensor([0.0, 1.0]), beta_max=2.0).tolist() == pytest.approx([0.5, 2.0])


def test_guided_rates_use_only_one_scalar_potential_difference() -> None:
    base = torch.tensor([[2.0, 3.0, 0.0]])
    legal = torch.tensor([[True, True, False]])
    current = torch.tensor([1.0])
    children = torch.tensor([[2.0, 0.5, float("nan")]])
    guided = potential_guided_rates_v3(
        base, legal, current, children, progress=torch.tensor([1.0]), beta_max=1.0
    )
    assert guided[0, 0].item() == pytest.approx(2.0 * math.exp(1.0))
    assert guided[0, 1].item() == pytest.approx(3.0 * math.exp(-0.5))
    assert guided[0, 2].item() == 0.0


def _value_batch():
    return {
        "source_tokens": torch.tensor([[0, 1, 4], [2, 3, 0]]),
        "current_tokens": torch.tensor([[0, 2, 4], [2, 3, 1]]),
        "padding_mask": torch.tensor([[False, False, True], [False, False, False]]),
        "source_pretrained_tokens": torch.zeros(2, 3, 768),
        "remaining_budget": torch.tensor([2, 0]),
        "quantity_ids": torch.tensor([0, 1]),
        "measurement_ids": torch.tensor([0, 1]),
        "numerator_ids": torch.tensor([0, 1]),
        "denominator_ids": torch.tensor([0, 1]),
        "assay_ids": torch.tensor([0, 1]),
        "context_ids": torch.tensor([0, 1]),
        "region_ids": torch.tensor([0, 1]),
    }


def test_value_network_is_six_block_width384_scalar_only() -> None:
    model = XEditValueV3(
        assay_count=2, context_count=2, quantity_count=2, measurement_count=2,
        numerator_count=2, denominator_count=2,
    )
    output = model(_value_batch())
    assert output.shape == (2,)
    assert torch.isfinite(output).all()
    assert len(model.blocks) == 6
    assert model.scalar_head[-1].out_features == 1
    assert not hasattr(model, "action_ratio_head")


def test_smc_ess_and_stratified_resampling_are_replayable() -> None:
    uniform = [0.0] * 32
    assert effective_sample_size_v3(uniform) == pytest.approx(32.0)
    unchanged = stratified_resample_v3(uniform, seed=7)
    assert unchanged["resampled"] is False
    collapsed = [0.0] + [-100.0] * 31
    assert effective_sample_size_v3(collapsed) < 16.0
    first = stratified_resample_v3(collapsed, seed=7)
    second = stratified_resample_v3(collapsed, seed=7)
    assert first == second
    assert first["resampled"] is True
    assert first["ancestor_indices"] == [0] * 32
    assert first["log_weights_after"] == pytest.approx([-math.log(32.0)] * 32)


def test_terminal_dedup_merges_particle_mass_and_caps_candidates() -> None:
    particles = [
        {"candidate_sequence": "A", "log_weight": math.log(0.2), "id": 1},
        {"candidate_sequence": "A", "log_weight": math.log(0.3), "id": 2},
    ] + [
        {"candidate_sequence": f"X{index}", "log_weight": math.log(0.01), "id": index}
        for index in range(40)
    ]
    deduplicated = deduplicate_terminal_candidates_v3(particles)
    assert len(deduplicated) == 32
    assert deduplicated[0]["candidate_sequence"] == "A"
    assert deduplicated[0]["particle_multiplicity"] == 2
    assert math.exp(deduplicated[0]["merged_log_weight"]) == pytest.approx(0.5)


def test_matched_compute_counts_each_critic_member_and_hard_ceiling() -> None:
    record = MatchedComputeRecordV2("source")
    record.base_flow_forwards = 100
    record.value_forwards = 100
    for member in range(3):
        record.add_critic_forwards(member, 40)
    record.candidate_count = 32
    payload = record.to_dict()
    assert payload["critic_forwards_by_member"] == [40, 40, 40]
    assert payload["total_forward_equivalents"] == 320
    record.add_critic_forwards(0)
    with pytest.raises(Exception, match="ceiling"):
        record.to_dict()
