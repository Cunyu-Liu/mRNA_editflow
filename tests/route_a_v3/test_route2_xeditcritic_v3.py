from __future__ import annotations

import torch

from core.route2_xeditcritic_v3 import StudyLogScaleCalibrationV3, XEditCriticV3


def _model(*, arm: str = "C2", control_mode: str = "NONE") -> XEditCriticV3:
    torch.manual_seed(7)
    return XEditCriticV3(
        arm=arm,
        control_mode=control_mode,
        study_count=3,
        assay_count=3,
        context_count=4,
        quantity_count=5,
        measurement_count=6,
        numerator_count=4,
        denominator_count=4,
        pretrained_width=8,
        condition_width=16,
        raw_hidden_dim=17,
        raw_depth=2,
        model_width=32,
        transformer_depth=2,
        transformer_heads=4,
        transformer_ffn_width=64,
        dropout=0.0,
    ).eval()


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
    source_tokens = torch.tensor([[0, 1, 2, 3, 0], [1, 1, 2, 2, 4]])
    candidate_tokens = torch.tensor([[0, 2, 2, 3, 0], [1, 1, 2, 2, 4]])
    padding_mask = source_tokens.eq(4)
    edit_padding_mask = torch.tensor([[False, True, True], [True, True, True]])
    source_site = torch.randn(2, 3, 8)
    candidate_site = source_site.clone()
    candidate_site[0, 0] += 0.3
    source_window_mean = torch.randn(2, 3, 8)
    candidate_window_mean = source_window_mean.clone()
    candidate_window_mean[0, 0] -= 0.2
    source_window_max = torch.randn(2, 3, 8)
    candidate_window_max = source_window_max.clone()
    candidate_window_max[0, 0] += 0.1
    source_global = torch.randn(2, 8)
    candidate_global = source_global.clone()
    candidate_global[0] += 0.4
    return {
        "source_tokens": source_tokens,
        "candidate_tokens": candidate_tokens,
        "padding_mask": padding_mask,
        "source_site": source_site,
        "candidate_site": candidate_site,
        "source_window_mean": source_window_mean,
        "candidate_window_mean": candidate_window_mean,
        "source_window_max": source_window_max,
        "candidate_window_max": candidate_window_max,
        "source_global": source_global,
        "candidate_global": candidate_global,
        "source_edit_base_ids": torch.tensor([[1, 4, 4], [4, 4, 4]]),
        "candidate_edit_base_ids": torch.tensor([[2, 4, 4], [4, 4, 4]]),
        "normalized_edit_positions": torch.tensor([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "edit_padding_mask": edit_padding_mask,
        "study_ids": torch.tensor([0, 2]),
        "assay_ids": torch.tensor([1, 2]),
        "context_ids": torch.tensor([2, 3]),
        "quantity_ids": torch.tensor([1, 4]),
        "measurement_ids": torch.tensor([2, 5]),
        "numerator_ids": torch.tensor([1, 3]),
        "denominator_ids": torch.tensor([2, 3]),
        "region_ids": torch.tensor([0, 1]),
    }


def _swap(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    swapped = dict(batch)
    for suffix in (
        "tokens",
        "site",
        "window_mean",
        "window_max",
        "global",
        "edit_base_ids",
    ):
        swapped[f"source_{suffix}"] = batch[f"candidate_{suffix}"]
        swapped[f"candidate_{suffix}"] = batch[f"source_{suffix}"]
    return swapped


def test_primary_model_is_strictly_antisymmetric_and_identity_is_zero() -> None:
    model = _model()
    batch = _batch()
    prediction = model(batch)["mean"]
    reverse = model(_swap(batch))["mean"]
    assert torch.equal(prediction, -reverse)
    assert prediction[1].item() == 0.0


def test_identity_is_strictly_zero_during_training_with_configured_dropout() -> None:
    model = _model().train()
    model.paired_dropout_probability = 0.1
    batch = _batch()
    for _ in range(3):
        assert model(batch)["mean"][1].item() == 0.0


def test_unknown_study_scale_is_exactly_one_without_an_intercept() -> None:
    calibration = StudyLogScaleCalibrationV3(4)
    with torch.no_grad():
        calibration.known_log_scale.copy_(torch.tensor([0.2, -0.4, 0.7]))
    scale = calibration.scale(torch.tensor([0, 1, 2, 3]))
    assert scale[0].item() == 1.0
    assert torch.allclose(scale[1:], torch.exp(torch.tensor([0.2, -0.4, 0.7])))
    assert not any("intercept" in name for name, _ in calibration.named_parameters())


def test_study_identity_only_appears_in_nuisance_calibration() -> None:
    model = _model()
    study_parameters = [name for name, _ in model.named_parameters() if "study" in name]
    assert study_parameters == ["study_calibration.known_log_scale"]


def test_all_frozen_screen_arms_have_finite_outputs() -> None:
    batch = _batch()
    for arm in ("C0", "C1", "C2", "C3"):
        output = _model(arm=arm)(batch)
        assert output["mean"].shape == (2,)
        assert torch.isfinite(output["mean"]).all()


def test_candidate_information_controls_execute_with_matched_full_geometry() -> None:
    batch = _batch()
    full_count = _model(control_mode="NONE").trainable_parameter_count
    for control in ("SOURCE_ONLY", "EDIT_METADATA_ONLY", "NO_CANDIDATE_SEQUENCE"):
        model = _model(control_mode=control)
        assert model.trainable_parameter_count == full_count
        assert torch.isfinite(model(batch)["mean"]).all()


def test_default_c2_capacity_is_in_the_frozen_range() -> None:
    model = XEditCriticV3(
        arm="C2",
        study_count=8,
        assay_count=16,
        context_count=64,
        quantity_count=7,
        measurement_count=7,
        numerator_count=7,
        denominator_count=7,
    )
    assert 27_000_000 <= model.trainable_parameter_count <= 30_000_000
