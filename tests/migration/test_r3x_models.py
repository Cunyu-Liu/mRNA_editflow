from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from scripts.r3x.checkpoint import (
    CheckpointProtocolError,
    build_checkpoint,
    load_checkpoint_exact,
    save_checkpoint_exclusive,
    validate_checkpoint,
)
from scripts.r3x.models import (
    BoundedLocalConditionUnknownControl,
    BoundedLocalGlobalExactPotentialCritic,
    BoundedLocalParameterMatchedUnrestrictedCritic,
    BoundedLocalWrongPositionControl,
    ConditionAwareParameterMatchedUnrestrictedSiamese,
    EditAwareFullPairUnrestrictedCritic,
    ExactPotentialCritic,
    FullPairUnrestrictedSiamese,
    ModelConfig,
    ParameterMatchedUnrestrictedSiamese,
    R2StyleNoExplicitStudyCrossAttentionCritic,
    ReplayDerivedSparseEditSetAdditiveCritic,
    ReplayDerivedSparseEditSetConditionUnknownControl,
    ReplayDerivedSparseEditSetPositionAgnosticControl,
    ReplayDerivedSparseEditSetWrongPositionControl,
    SiteAdditiveLocalGlobalExactPotentialCritic,
    SiteAdditiveParameterMatchedUnrestrictedCritic,
    SparseEditSetCritic,
    count_parameters,
    derive_replay_edit_tensors,
    derive_wrong_local_feature_positions,
    require_parameter_match,
)


def config() -> ModelConfig:
    return ModelConfig(
        vocab_size=5,
        d_model=16,
        n_heads=4,
        n_layers=1,
        ff_dim=32,
        max_len=16,
        n_endpoints=3,
        n_contexts=4,
        max_edits=5,
        pair_rank=2,
    )


def sequences():
    source = torch.tensor([[1, 2, 3, 4, 1, 0], [4, 3, 2, 1, 0, 0]])
    candidate = source.clone()
    candidate[0, 2] = 4
    candidate[1, 1] = 1
    source_mask = source != 0
    candidate_mask = candidate != 0
    endpoint = torch.tensor([1, 2])
    context = torch.tensor([2, 3])
    return source, candidate, source_mask, candidate_mask, endpoint, context


class _TokenValueEncoder(torch.nn.Module):
    """Deterministic test encoder exposing token-local and mean-pooled signals."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        hidden = torch.zeros(
            (*tokens.shape, self.d_model), dtype=torch.float32, device=tokens.device
        )
        hidden[..., 0] = tokens.to(torch.float32) * mask.to(torch.float32)
        pooled = hidden.sum(dim=1) / mask.sum(dim=1, keepdim=True).to(hidden.dtype)
        return hidden, pooled


def test_exact_potential_identity_and_swap_are_algebraic() -> None:
    torch.manual_seed(11)
    model = ExactPotentialCritic(config()).eval()
    source, candidate, sm, cm, endpoint, context = sequences()
    identity = model(source, source, sm, sm, endpoint, context)
    forward = model(source, candidate, sm, cm, endpoint, context)
    reverse = model(candidate, source, cm, sm, endpoint, context)
    assert torch.equal(identity["mean"], torch.zeros_like(identity["mean"]))
    assert torch.equal(forward["mean"], -reverse["mean"])
    assert torch.equal(forward["logvar"], reverse["logvar"])
    assert torch.equal(forward["sign"], forward["mean"])
    assert torch.equal(forward["rank"], forward["mean"])


def test_exact_potential_has_no_study_embedding_or_study_state_key() -> None:
    model = ExactPotentialCritic(config())
    assert not hasattr(model, "study_embedding")
    assert all("study" not in key.lower() for key in model.state_dict())


def test_a1_site_additive_potential_preserves_exact_algebra() -> None:
    torch.manual_seed(17)
    model = SiteAdditiveLocalGlobalExactPotentialCritic(config()).eval()
    source, candidate, sm, cm, endpoint, context = sequences()
    identity = model(source, source, sm, sm, endpoint, context)
    forward = model(source, candidate, sm, cm, endpoint, context)
    reverse = model(candidate, source, cm, sm, endpoint, context)
    assert torch.equal(identity["mean"], torch.zeros_like(identity["mean"]))
    assert torch.equal(forward["mean"], -reverse["mean"])
    assert torch.equal(forward["logvar"], reverse["logvar"])
    assert forward["local_site_branch"] is True
    assert forward["site_normalization"] == "SUM_OVER_SQRT_VISIBLE_LENGTH"
    assert all("study" not in key.lower() for key in model.state_dict())


def test_a1_site_branch_reduces_single_edit_length_dilution_from_l_to_sqrt_l() -> None:
    model = SiteAdditiveLocalGlobalExactPotentialCritic(config()).eval()
    model.encoder = _TokenValueEncoder(model.cfg.d_model)
    with torch.no_grad():
        for parameter in model.potential_head.parameters():
            parameter.zero_()
        first = model.site_potential_head[0]
        second = model.site_potential_head[2]
        first.weight.zero_()
        first.bias.zero_()
        first.weight[0, 0] = 1.0
        second.weight.zero_()
        second.bias.zero_()
        second.weight[0, 0] = 1.0

    def effect(length: int) -> torch.Tensor:
        source = torch.ones((1, length), dtype=torch.long)
        candidate = source.clone()
        candidate[0, 1] = 2
        mask = torch.ones_like(source, dtype=torch.bool)
        return model(
            source,
            candidate,
            mask,
            mask,
            torch.tensor([1]),
            torch.tensor([1]),
        )["mean"]

    short = effect(4)
    long = effect(16)
    assert short.abs().item() > 0.0
    assert (long / short).item() == pytest.approx(0.5, rel=1e-6)


def test_a1_matched_control_is_parameter_matched_and_condition_aware() -> None:
    torch.manual_seed(19)
    exact = SiteAdditiveLocalGlobalExactPotentialCritic(config()).eval()
    control = SiteAdditiveParameterMatchedUnrestrictedCritic(config()).eval()
    assert abs(count_parameters(control) - count_parameters(exact)) / count_parameters(exact) <= 0.05
    source, _, sm, _, endpoint, context = sequences()
    first = control(source, source, sm, sm, endpoint, context)
    changed = control(
        source,
        source,
        sm,
        sm,
        endpoint,
        (context + 1) % control.cfg.n_contexts,
    )
    assert not torch.equal(first["mean"], torch.zeros_like(first["mean"]))
    assert not torch.allclose(first["mean"], changed["mean"], atol=1e-8, rtol=0)
    assert first["condition_aware_nonintegrable_residual"] is True


def test_bounded_local_a1_has_exact_finite_receptive_field_support() -> None:
    torch.manual_seed(20)
    model = BoundedLocalGlobalExactPotentialCritic(config()).eval()
    length = 16
    source = torch.ones((1, length), dtype=torch.long)
    candidate = source.clone()
    edited_positions = (2, 8, 13)
    for position in edited_positions:
        candidate[0, position] = 2
    mask = torch.ones_like(source, dtype=torch.bool)
    condition = model.condition(torch.tensor([1]), torch.tensor([1]))
    source_energy = model._bounded_local_site_energies(source, mask, condition)
    candidate_energy = model._bounded_local_site_energies(candidate, mask, condition)
    changed_support = candidate_energy.ne(source_energy).squeeze(0)
    expected_support = torch.zeros(length, dtype=torch.bool)
    radius = model.local_receptive_field_radius
    for position in edited_positions:
        expected_support[
            max(0, position - radius) : min(length, position + radius + 1)
        ] = True
    assert torch.any(changed_support)
    assert not torch.any(changed_support & ~expected_support)
    assert int(changed_support.sum()) <= len(edited_positions) * (2 * radius + 1)


def test_bounded_local_a1_far_shared_background_cannot_change_local_edit_delta() -> None:
    torch.manual_seed(21)
    model = BoundedLocalGlobalExactPotentialCritic(config()).eval()
    length = 16
    source = torch.ones((1, length), dtype=torch.long)
    candidate = source.clone()
    candidate[0, 3] = 2
    mask = torch.ones_like(source, dtype=torch.bool)
    condition = model.condition(torch.tensor([1]), torch.tensor([1]))

    def local_delta(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_energy = model._bounded_local_site_energies(left, mask, condition)
        right_energy = model._bounded_local_site_energies(right, mask, condition)
        return (right_energy - left_energy).sum(dim=1) / length**0.5

    baseline = local_delta(source, candidate)
    background_source = source.clone()
    background_candidate = candidate.clone()
    background_source[0, 13] = 4
    background_candidate[0, 13] = 4
    shifted = local_delta(background_source, background_candidate)
    assert torch.equal(baseline, shifted)

    # The full critic deliberately retains a global Transformer potential.  A
    # finite receptive field is a property of the local branch only.
    base_full = model(
        source,
        candidate,
        mask,
        mask,
        torch.tensor([1]),
        torch.tensor([1]),
    )["mean"]
    shifted_full = model(
        background_source,
        background_candidate,
        mask,
        mask,
        torch.tensor([1]),
        torch.tensor([1]),
    )["mean"]
    assert not torch.allclose(base_full, shifted_full, atol=1e-8, rtol=0)


def test_bounded_local_a1_length_normalization_is_not_extension_invariance() -> None:
    torch.manual_seed(211)
    model = BoundedLocalGlobalExactPotentialCritic(config()).eval()
    condition = model.condition(torch.tensor([1]), torch.tensor([1]))

    def local_delta(length: int) -> torch.Tensor:
        source = torch.ones((1, length), dtype=torch.long)
        candidate = source.clone()
        candidate[0, 2] = 2
        mask = torch.ones_like(source, dtype=torch.bool)
        source_energy = model._bounded_local_site_energies(source, mask, condition)
        candidate_energy = model._bounded_local_site_energies(candidate, mask, condition)
        return (candidate_energy - source_energy).sum(dim=1) / length**0.5

    short = local_delta(8)
    long = local_delta(16)
    assert short.abs().item() > 0.0
    assert (long / short).item() == pytest.approx(2**-0.5, rel=1e-5)


def test_bounded_local_exact_cycle_closes_but_unrestricted_control_does_not() -> None:
    torch.manual_seed(212)
    exact = BoundedLocalGlobalExactPotentialCritic(config()).eval()
    control = BoundedLocalParameterMatchedUnrestrictedCritic(config()).eval()
    a = torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.long)
    b = a.clone()
    b[0, 1] = 2
    c = b.clone()
    c[0, 4] = 3
    mask = torch.ones_like(a, dtype=torch.bool)
    endpoint = torch.tensor([1])
    context = torch.tensor([1])

    def cycle(model) -> torch.Tensor:
        ab = model(a, b, mask, mask, endpoint, context)["mean"]
        bc = model(b, c, mask, mask, endpoint, context)["mean"]
        ca = model(c, a, mask, mask, endpoint, context)["mean"]
        return ab + bc + ca

    assert torch.allclose(cycle(exact), torch.zeros(1), atol=1e-7, rtol=0)
    assert cycle(control).abs().item() > 1e-7


def test_bounded_local_a1_preserves_exact_algebra_padding_and_parameter_match() -> None:
    torch.manual_seed(22)
    exact = BoundedLocalGlobalExactPotentialCritic(config()).eval()
    control = BoundedLocalParameterMatchedUnrestrictedCritic(config()).eval()
    matched = require_parameter_match(exact, control)
    assert matched["matched"] is True
    source, candidate, sm, cm, endpoint, context = sequences()
    identity = exact(source, source, sm, sm, endpoint, context)
    forward = exact(source, candidate, sm, cm, endpoint, context)
    reverse = exact(candidate, source, cm, sm, endpoint, context)
    assert torch.equal(identity["mean"], torch.zeros_like(identity["mean"]))
    assert torch.equal(forward["mean"], -reverse["mean"])
    assert torch.equal(forward["logvar"], reverse["logvar"])
    assert forward["bounded_local_branch_support"] is True
    assert (
        forward["overall_effect_support"]
        == "GLOBAL_PLUS_FINITE_RF_LOCAL_NOT_STRICTLY_BOUNDED"
    )
    assert forward["whole_mean_bounded_support"] is False
    assert forward["uncertainty_bounded_support"] is False
    assert forward["global_branch_present"] is True
    assert forward["local_receptive_field_overlap_possible_through_distance"] == 4
    assert forward["strict_local_interaction_exclusion_minimum_distance"] == 5
    assert forward["length_extension_invariant"] is False
    assert exact.local_depthwise.kernel_size == (5,)
    assert exact.local_depthwise.dilation == (1,)
    assert exact.local_depthwise.stride == (1,)
    assert exact.local_depthwise.groups == exact.cfg.d_model
    assert exact.local_depthwise.bias is None
    assert exact.local_pointwise.kernel_size == (1,)
    assert forward["local_receptive_field_radius"] == 2
    assert (
        forward["local_uncertainty_summary"]
        == "SWAP_SYMMETRIC_ABSOLUTE_SITE_ENERGY_DELTA"
    )
    assert exact.local_pointwise.bias is None
    assert exact.local_site_head[-1].bias is None

    padded_source = torch.nn.functional.pad(source, (0, 3), value=0)
    padded_candidate = torch.nn.functional.pad(candidate, (0, 3), value=0)
    padded_sm = torch.nn.functional.pad(sm, (0, 3), value=False)
    padded_cm = torch.nn.functional.pad(cm, (0, 3), value=False)
    padded = exact(
        padded_source, padded_candidate, padded_sm, padded_cm, endpoint, context
    )
    assert torch.allclose(forward["mean"], padded["mean"], atol=1e-6, rtol=0)

    source_identity = control(source, source, sm, sm, endpoint, context)
    assert not torch.equal(
        source_identity["mean"], torch.zeros_like(source_identity["mean"])
    )
    assert source_identity["condition_aware_nonintegrable_residual"] is True


def test_bounded_local_a1_rejects_non_substitution_tensor_support() -> None:
    model = BoundedLocalGlobalExactPotentialCritic(config()).eval()
    source = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    candidate = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    source_mask = source != 0
    candidate_mask = candidate != 0
    with pytest.raises(ValueError, match="exact equal mask"):
        model(
            source,
            candidate,
            source_mask,
            candidate_mask,
            torch.tensor([1]),
            torch.tensor([1]),
        )

    noncontiguous = torch.tensor([[True, False, True, False]])
    noncontiguous_tokens = torch.tensor([[1, 0, 3, 0]], dtype=torch.long)
    with pytest.raises(ValueError, match="right-contiguous"):
        model(
            noncontiguous_tokens,
            noncontiguous_tokens,
            noncontiguous,
            noncontiguous,
            torch.tensor([1]),
            torch.tensor([1]),
        )


def test_parameter_matched_control_is_within_five_percent_and_unrestricted() -> None:
    torch.manual_seed(23)
    exact = ExactPotentialCritic(config()).eval()
    control = ParameterMatchedUnrestrictedSiamese(config()).eval()
    exact_n = count_parameters(exact)
    control_n = count_parameters(control)
    assert abs(control_n - exact_n) / exact_n <= 0.05
    source, _, sm, _, endpoint, context = sequences()
    out = control(source, source, sm, sm, endpoint, context)
    assert not torch.equal(out["mean"], torch.zeros_like(out["mean"]))
    assert control.architecture_id != exact.architecture_id


def test_condition_aware_a0_control_is_parameter_matched_and_condition_dependent() -> None:
    torch.manual_seed(23)
    exact = ExactPotentialCritic(config()).eval()
    control = ConditionAwareParameterMatchedUnrestrictedSiamese(config()).eval()
    assert abs(count_parameters(control) - count_parameters(exact)) / count_parameters(exact) <= 0.05
    source, _, sm, _, endpoint, context = sequences()
    first = control(source, source, sm, sm, endpoint, context)
    changed = control(
        source,
        source,
        sm,
        sm,
        endpoint,
        (context + 1) % control.cfg.n_contexts,
    )
    assert not torch.allclose(first["mean"], changed["mean"], atol=1e-8, rtol=0)
    assert first["condition_aware_nonintegrable_residual"] is True


def test_full_pair_strong_baseline_is_parameter_matched_and_order_sensitive() -> None:
    torch.manual_seed(29)
    exact = ExactPotentialCritic(config()).eval()
    baseline = FullPairUnrestrictedSiamese(config()).eval()
    assert abs(count_parameters(baseline) - count_parameters(exact)) / count_parameters(exact) <= 0.05
    source, candidate, sm, cm, endpoint, context = sequences()
    identity = baseline(source, source, sm, sm, endpoint, context)
    forward = baseline(source, candidate, sm, cm, endpoint, context)
    reverse = baseline(candidate, source, cm, sm, endpoint, context)
    assert not torch.equal(identity["mean"], torch.zeros_like(identity["mean"]))
    assert not torch.allclose(forward["mean"], reverse["mean"], atol=1e-8, rtol=0)
    assert not torch.allclose(forward["mean"], -reverse["mean"], atol=1e-8, rtol=0)
    assert forward["potential_constraint"] is False
    assert baseline.information_set == exact.information_set


def test_edit_aware_full_pair_is_parameter_matched_and_padding_invariant() -> None:
    torch.manual_seed(29)
    exact = ExactPotentialCritic(config()).eval()
    baseline = EditAwareFullPairUnrestrictedCritic(config()).eval()
    assert abs(count_parameters(baseline) - count_parameters(exact)) / count_parameters(exact) <= 0.05
    source, candidate, sm, cm, endpoint, context = sequences()
    forward = baseline(source, candidate, sm, cm, endpoint, context)
    assert forward["changed_position_token_pooling"] is True
    assert forward["token_level_cross_attention"] is False
    assert baseline.information_set == exact.information_set

    padded_source = torch.nn.functional.pad(source, (0, 3), value=0)
    padded_candidate = torch.nn.functional.pad(candidate, (0, 3), value=0)
    padded_sm = torch.nn.functional.pad(sm, (0, 3), value=False)
    padded_cm = torch.nn.functional.pad(cm, (0, 3), value=False)
    padded = baseline(
        padded_source, padded_candidate, padded_sm, padded_cm, endpoint, context
    )
    assert torch.allclose(forward["mean"], padded["mean"], atol=1e-6, rtol=0)


def test_edit_aware_full_pair_retains_changed_token_summary_across_lengths() -> None:
    model = EditAwareFullPairUnrestrictedCritic(config()).eval()
    model.encoder = _TokenValueEncoder(model.cfg.d_model)
    captured = []
    handle = model.pair_head[0].register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )
    try:
        for length in (4, 16):
            source = torch.ones((1, length), dtype=torch.long)
            candidate = source.clone()
            candidate[0, 1] = 2
            mask = torch.ones_like(source, dtype=torch.bool)
            model(
                source,
                candidate,
                mask,
                mask,
                torch.tensor([1]),
                torch.tensor([1]),
            )
    finally:
        handle.remove()
    d = model.cfg.d_model
    # Blocks 5:8 are changed-source, changed-candidate, and their difference.
    assert torch.equal(captured[0][:, 5 * d : 8 * d], captured[1][:, 5 * d : 8 * d])
    # The global source/candidate blocks still differ with length, proving the
    # equality above comes from the edit-aware branch rather than the mean pool.
    assert not torch.equal(captured[0][:, : 2 * d], captured[1][:, : 2 * d])


def test_edit_aware_full_pair_rejects_unaligned_or_out_of_budget_pairs() -> None:
    model = EditAwareFullPairUnrestrictedCritic(config()).eval()
    source, candidate, sm, cm, endpoint, context = sequences()
    with pytest.raises(ValueError, match="aligned equal-length"):
        model(source[:, :-1], candidate, sm[:, :-1], cm, endpoint, context)
    two_edit_candidate = candidate.clone()
    two_edit_candidate[0, 0] = 2
    with pytest.raises(ValueError, match="changed-token count"):
        model(source, two_edit_candidate, sm, cm, endpoint, context)


def test_r2_style_adaptation_asserts_only_no_explicit_study_input() -> None:
    torch.manual_seed(31)
    model = R2StyleNoExplicitStudyCrossAttentionCritic(config()).eval()
    assert all("study" not in key.lower() for key in model.state_dict())
    assert not any("study" in name.lower() for name, _ in model.named_modules())
    source, candidate, sm, cm, endpoint, context = sequences()
    output = model(source, candidate, sm, cm, endpoint, context)
    assert output["explicit_study_id_input_or_embedding"] is False
    assert output["semantic_study_proxy_exclusion_established"] is False
    assert output["historical_r2_single_variable_ablation"] is False
    assert torch.isfinite(output["mean"]).all()
    assert torch.isfinite(output["logvar"]).all()

    padded_source = torch.nn.functional.pad(source, (0, 3), value=0)
    padded_candidate = torch.nn.functional.pad(candidate, (0, 3), value=0)
    padded_sm = torch.nn.functional.pad(sm, (0, 3), value=False)
    padded_cm = torch.nn.functional.pad(cm, (0, 3), value=False)
    replay = model(
        padded_source, padded_candidate, padded_sm, padded_cm, endpoint, context
    )
    assert torch.allclose(output["mean"], replay["mean"], atol=1e-6, rtol=0)


def test_r2_style_adaptation_fails_closed_on_unaligned_primary_route() -> None:
    model = R2StyleNoExplicitStudyCrossAttentionCritic(config()).eval()
    source, candidate, sm, cm, endpoint, context = sequences()
    with pytest.raises(ValueError, match="aligned equal-length"):
        model(source[:, :-1], candidate, sm[:, :-1], cm, endpoint, context)
    unequal_mask = cm.clone()
    unequal_mask[0, -1] = True
    candidate_with_masked_nonzero = candidate.clone()
    candidate_with_masked_nonzero[0, -1] = 1
    with pytest.raises(ValueError, match="aligned equal-length"):
        model(source, candidate_with_masked_nonzero, sm, unequal_mask, endpoint, context)


def test_model_config_rejects_parameter_mismatch_domain_below_eight() -> None:
    with pytest.raises(ValueError, match="d_model >= 8"):
        ModelConfig(
            vocab_size=5, d_model=4, n_heads=1, n_layers=1, ff_dim=8,
            max_len=16, n_endpoints=2, n_contexts=2, max_edits=5, pair_rank=1,
        )


def test_parameter_match_is_checked_for_the_actual_frozen_configuration() -> None:
    cfg = config()
    matched = require_parameter_match(
        ExactPotentialCritic(cfg), EditAwareFullPairUnrestrictedCritic(cfg)
    )
    assert matched["matched"] is True
    assert matched["gap_fraction"] <= 0.05

    narrow = ModelConfig(
        vocab_size=5,
        d_model=8,
        n_heads=1,
        n_layers=1,
        ff_dim=16,
        max_len=1,
        n_endpoints=2,
        n_contexts=2,
        max_edits=5,
        pair_rank=1,
    )
    with pytest.raises(ValueError, match="actual frozen configuration"):
        require_parameter_match(
            ExactPotentialCritic(narrow),
            EditAwareFullPairUnrestrictedCritic(narrow),
        )

    oversized_rank = ModelConfig(
        vocab_size=5,
        d_model=16,
        n_heads=4,
        n_layers=1,
        ff_dim=32,
        max_len=16,
        n_endpoints=3,
        n_contexts=4,
        max_edits=5,
        pair_rank=128,
    )
    with pytest.raises(ValueError, match="actual frozen configuration"):
        require_parameter_match(
            SiteAdditiveLocalGlobalExactPotentialCritic(oversized_rank),
            SiteAdditiveParameterMatchedUnrestrictedCritic(oversized_rank),
        )
    with pytest.raises(ValueError, match="actual frozen configuration"):
        require_parameter_match(
            BoundedLocalGlobalExactPotentialCritic(oversized_rank),
            BoundedLocalParameterMatchedUnrestrictedCritic(oversized_rank),
        )


def edit_inputs():
    source = torch.tensor([[1, 2, 3, 4, 1, 0], [4, 3, 2, 1, 0, 0]])
    source_mask = source != 0
    positions = torch.tensor([[1, 3, 4, 0, 0], [1, 0, 0, 0, 0]])
    ref = torch.tensor([[2, 4, 1, 0, 0], [3, 0, 0, 0, 0]])
    alt = torch.tensor([[3, 1, 2, 0, 0], [1, 0, 0, 0, 0]])
    edit_mask = torch.tensor([[True, True, True, False, False], [True, False, False, False, False]])
    candidate = source.clone()
    for row in range(2):
        for col in range(5):
            if edit_mask[row, col]:
                candidate[row, positions[row, col]] = alt[row, col]
    endpoint = torch.tensor([1, 2])
    context = torch.tensor([2, 3])
    return source, candidate, source_mask, positions, ref, alt, edit_mask, endpoint, context


def test_sparse_edit_additive_is_order_and_padding_invariant() -> None:
    torch.manual_seed(42)
    model = SparseEditSetCritic(config(), interaction=False).eval()
    source, candidate, mask, pos, ref, alt, edit_mask, endpoint, context = edit_inputs()
    first = model(source, mask, pos, ref, alt, edit_mask, endpoint, context, candidate_tokens=candidate)
    permutation = torch.tensor([2, 0, 1, 4, 3])
    second = model(
        source,
        mask,
        pos[:, permutation],
        ref[:, permutation],
        alt[:, permutation],
        edit_mask[:, permutation],
        endpoint,
        context,
        candidate_tokens=candidate,
    )
    assert torch.allclose(first["mean"], second["mean"], atol=1e-6, rtol=0)
    assert torch.allclose(first["logvar"], second["logvar"], atol=1e-6, rtol=0)


def test_sparse_edit_candidate_is_replay_validation_not_free_branch() -> None:
    model = SparseEditSetCritic(config(), interaction=False).eval()
    source, candidate, mask, pos, ref, alt, edit_mask, endpoint, context = edit_inputs()
    good = model(source, mask, pos, ref, alt, edit_mask, endpoint, context, candidate_tokens=candidate)
    with pytest.raises(ValueError, match="replay"):
        model(source, mask, pos, ref, alt, edit_mask, endpoint, context,
              candidate_tokens=source)
    assert good["candidate_dense_branch_used"] is False


def test_sparse_edit_rejects_out_of_contract_budget_and_ref_mismatch() -> None:
    model = SparseEditSetCritic(config(), interaction=False).eval()
    source, candidate, mask, pos, ref, alt, edit_mask, endpoint, context = edit_inputs()
    bad_mask = edit_mask.clone()
    bad_mask[1, 1] = True  # two edits is outside {1,3,5}
    with pytest.raises(ValueError, match="budget"):
        model(source, mask, pos, ref, alt, bad_mask, endpoint, context)
    bad_ref = ref.clone()
    bad_ref[0, 0] = 4
    with pytest.raises(ValueError, match="source token"):
        model(source, mask, pos, bad_ref, alt, edit_mask, endpoint, context)


def _compatible_wrong_local_pair():
    source = torch.tensor([[1, 2, 3, 4] * 5], dtype=torch.long)
    candidate = source.clone()
    candidate[0, [0, 1, 2, 3, 4]] = torch.tensor([2, 3, 4, 1, 2])
    mask = torch.ones_like(source, dtype=torch.bool)
    return source, candidate, mask, torch.tensor([1]), torch.tensor([2])


def test_replay_derived_b_additive_matches_explicit_sparse_action_kernel() -> None:
    torch.manual_seed(420)
    cfg = ModelConfig(**{**config().__dict__, "max_len": 20})
    derived = ReplayDerivedSparseEditSetAdditiveCritic(cfg).eval()
    explicit = SparseEditSetCritic(cfg, interaction=False).eval()
    explicit.load_state_dict(derived.state_dict(), strict=True)
    source, candidate, mask, endpoint, context = _compatible_wrong_local_pair()
    positions, ref, alt, edit_mask = derive_replay_edit_tensors(
        source, candidate, mask, mask, max_edits=cfg.max_edits
    )
    derived_output = derived(
        source, candidate, mask, mask, endpoint, context
    )
    explicit_output = explicit(
        source,
        mask,
        positions,
        ref,
        alt,
        edit_mask,
        endpoint,
        context,
        candidate_tokens=candidate,
    )
    assert torch.equal(derived_output["mean"], explicit_output["mean"])
    assert torch.equal(derived_output["logvar"], explicit_output["logvar"])
    assert derived_output["candidate_dense_branch_used"] is False
    assert derived_output["interaction_branch_present"] is False


def test_wrong_local_controls_relocate_features_without_changing_action_replay() -> None:
    cfg = ModelConfig(**{**config().__dict__, "max_len": 20})
    source, candidate, mask, endpoint, context = _compatible_wrong_local_pair()
    positions, ref, _alt, edit_mask = derive_replay_edit_tensors(
        source, candidate, mask, mask, max_edits=cfg.max_edits
    )
    relocated = derive_wrong_local_feature_positions(
        source, mask, positions, ref, edit_mask
    )
    assert not torch.equal(relocated[edit_mask], positions[edit_mask])
    assert torch.equal(
        source.gather(1, relocated)[edit_mask], ref[edit_mask]
    )

    a_control = BoundedLocalWrongPositionControl(cfg).eval()
    a_output = a_control(source, candidate, mask, mask, endpoint, context)
    assert a_output["wrong_local_scope"] == "LOCAL_CANDIDATE_BRANCH_ONLY"
    assert a_output["actual_action_replay_changed"] is False
    assert a_output["potential_constraint"] is False

    b_control = ReplayDerivedSparseEditSetWrongPositionControl(cfg).eval()
    b_output = b_control(source, candidate, mask, mask, endpoint, context)
    assert b_output["wrong_local_position_control"] is True
    assert b_output["actual_action_replay_changed"] is False
    assert (
        b_output["feature_position_policy"]
        == "COMPATIBLE_UNEDITED_WRONG_SOURCE_POSITION"
    )


def test_wrong_local_control_has_explicit_hold_not_identity_fallback() -> None:
    source = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    candidate = source.clone()
    candidate[0, 0] = 2
    mask = torch.ones_like(source, dtype=torch.bool)
    positions, ref, _alt, edit_mask = derive_replay_edit_tensors(
        source, candidate, mask, mask, max_edits=5
    )
    with pytest.raises(
        ValueError, match="HOLD_WRONG_LOCAL_INSUFFICIENT_COMPATIBLE_POSITION_SUPPORT"
    ):
        derive_wrong_local_feature_positions(
            source, mask, positions, ref, edit_mask
        )


@pytest.mark.parametrize(
    "control_class",
    [
        BoundedLocalConditionUnknownControl,
        ReplayDerivedSparseEditSetConditionUnknownControl,
    ],
)
def test_condition_unk_only_controls_cannot_consume_observed_condition_ids(
    control_class,
) -> None:
    cfg = ModelConfig(**{**config().__dict__, "max_len": 20})
    torch.manual_seed(421)
    model = control_class(cfg).eval()
    source, candidate, mask, endpoint, context = _compatible_wrong_local_pair()
    first = model(source, candidate, mask, mask, endpoint, context)
    second = model(
        source,
        candidate,
        mask,
        mask,
        torch.tensor([2]),
        torch.tensor([3]),
    )
    assert torch.equal(first["mean"], second["mean"])
    assert torch.equal(first["logvar"], second["logvar"])
    assert first["condition_unk_only"] is True
    assert first["observed_endpoint_or_context_consumed"] is False


def test_b_position_agnostic_control_is_exactly_parameter_matched() -> None:
    cfg = ModelConfig(**{**config().__dict__, "max_len": 20})
    candidate = ReplayDerivedSparseEditSetAdditiveCritic(cfg)
    control = ReplayDerivedSparseEditSetPositionAgnosticControl(cfg).eval()
    assert count_parameters(candidate) == count_parameters(control)
    assert require_parameter_match(candidate, control)["gap_fraction"] == 0.0
    source, edited, mask, endpoint, context = _compatible_wrong_local_pair()
    output = control(source, edited, mask, mask, endpoint, context)
    assert output["position_agnostic_matched_control"] is True
    assert (
        output["position_coordinate_policy"]
        == "REF_ALT_PSEUDOCOORDINATE_NO_ACTUAL_POSITION"
    )
    assert (
        output["source_hidden_feature_policy"]
        == "GLOBAL_POOLED_SOURCE_REPEATED_PER_EDIT_NO_POSITION_GATHER"
    )
    assert output["actual_edit_position_consumed_by_prediction_branch"] is False
    assert (
        output["actual_edit_position_usage"]
        == "EXACT_ACTION_AND_CANDIDATE_REPLAY_VALIDATION_ONLY"
    )


def test_b_position_agnostic_control_is_exactly_invariant_to_actual_edit_locus() -> None:
    cfg = ModelConfig(**{**config().__dict__, "max_len": 8})
    torch.manual_seed(422)
    control = ReplayDerivedSparseEditSetPositionAgnosticControl(cfg).eval()
    source = torch.tensor([[1, 2, 1, 3, 4, 2, 3, 4]], dtype=torch.long)
    edited_left = source.clone()
    edited_right = source.clone()
    edited_left[0, 0] = 2
    edited_right[0, 2] = 2
    mask = torch.ones_like(source, dtype=torch.bool)
    endpoint = torch.tensor([1])
    context = torch.tensor([2])

    left = control(source, edited_left, mask, mask, endpoint, context)
    right = control(source, edited_right, mask, mask, endpoint, context)

    assert torch.equal(left["mean"], right["mean"])
    assert torch.equal(left["logvar"], right["logvar"])
    assert left["feature_position_policy"] == (
        "ACTUAL_EDIT_POSITION_EXCLUDED_FROM_PREDICTION"
    )
    assert left["global_pooled_source_repeated_per_edit"] is True


def checkpoint_bindings() -> dict:
    return {
        "run_start_sha256": "c" * 64,
        "run_plan_sha256": "d" * 64,
        "activation_manifest_sha256": "e" * 64,
        "estimand_binding_sha256": "f" * 64,
        "method_id": "A_EXACT_POTENTIAL_DIFFERENCE",
        "training_variant_id": "PRIMARY",
        "training_config_sha256": "1" * 64,
        "condition_vocab": {
            "endpoint": {"__UNK__": 0, "endpoint_a": 1, "endpoint_b": 2},
            "context": {
                "__UNK__": 0, "context_a": 1, "context_b": 2,
                "context_c": 3,
            },
        },
        "information_set_fingerprint": "2" * 64,
        "selected_epoch": 3,
        "checkpoint_selection_sha256": "3" * 64,
        "partition_row_set_sha256": {
            "train": "4" * 64,
            "development": "5" * 64,
            "internal_test": "6" * 64,
        },
    }


def test_checkpoint_binds_architecture_and_all_authority_hashes() -> None:
    model = ExactPotentialCritic(config())
    hashes = {
        "route_authorization_sha256": "1" * 64,
        "ready_dataset_sha256": "2" * 64,
        "inclusion_manifest_sha256": "3" * 64,
        "context_source_crosswalk_sha256": "4" * 64,
        "endpoint_estimand_registry_sha256": "5" * 64,
        "source_component_split_manifest_sha256": "6" * 64,
        "metric_zero_policy_registry_sha256": "7" * 64,
        "model_optimizer_seed_config_sha256": "8" * 64,
        "analysis_code_tree_manifest_sha256": "9" * 64,
        "method_protocol_sha256": "a" * 64,
        "paired_strongest_baseline_comparison_policy_sha256": "b" * 64,
    }
    payload = build_checkpoint(
        model,
        config(),
        authority_hashes=hashes,
        information_set="PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        seed=11,
        scientific_scope="DEVELOPMENT_ONLY_TWO_STUDY_NESTED_ENDPOINTS",
        **checkpoint_bindings(),
    )
    validated = validate_checkpoint(payload, expected_architecture_id=model.architecture_id,
                                    expected_authority_hashes=hashes)
    assert validated["architecture_id"] == model.architecture_id
    tampered = copy.deepcopy(payload)
    tampered["authority_hashes"]["ready_dataset_sha256"] = "a" * 64
    with pytest.raises(CheckpointProtocolError, match="ready_dataset"):
        validate_checkpoint(tampered, expected_architecture_id=model.architecture_id,
                            expected_authority_hashes=hashes)
    unexpected = copy.deepcopy(payload)
    unexpected["extra"] = "not allowed"
    with pytest.raises(CheckpointProtocolError, match="strict schema"):
        validate_checkpoint(
            unexpected, expected_architecture_id=model.architecture_id,
            expected_authority_hashes=hashes,
        )
    stale_vocab = copy.deepcopy(payload)
    stale_vocab["condition_vocab"]["endpoint"] = {
        "__UNK__": 0, "different_endpoint": 1, "endpoint_b": 2,
    }
    with pytest.raises(CheckpointProtocolError, match="vocab digest is stale"):
        validate_checkpoint(
            stale_vocab, expected_architecture_id=model.architecture_id,
            expected_authority_hashes=hashes,
        )


def test_checkpoint_loader_consumes_the_verified_bytes_safely(tmp_path: Path) -> None:
    model = ExactPotentialCritic(config())
    hashes = {
        name: format(index + 1, "x") * 64
        for index, name in enumerate((
            "route_authorization_sha256", "ready_dataset_sha256",
            "inclusion_manifest_sha256", "context_source_crosswalk_sha256",
            "endpoint_estimand_registry_sha256", "source_component_split_manifest_sha256",
            "metric_zero_policy_registry_sha256", "model_optimizer_seed_config_sha256",
            "analysis_code_tree_manifest_sha256", "method_protocol_sha256",
            "paired_strongest_baseline_comparison_policy_sha256",
        ))
    }
    payload = build_checkpoint(
        model, config(), authority_hashes=hashes,
        information_set="PAIRED_SEQUENCE_NO_MEASURED_ANCHOR", seed=11,
        scientific_scope="DEVELOPMENT_ONLY_SINGLE_STUDY",
        **checkpoint_bindings(),
    )
    path = tmp_path / "checkpoint.pt"
    digest = save_checkpoint_exclusive(path, payload)
    loaded = load_checkpoint_exact(path, digest)
    state_shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    validate_checkpoint(
        loaded, expected_architecture_id=model.architecture_id,
        expected_authority_hashes=hashes, expected_model_config=config(),
        expected_condition_vocab=checkpoint_bindings()["condition_vocab"],
        expected_run_start_sha256="c" * 64,
        expected_run_plan_sha256="d" * 64,
        expected_activation_manifest_sha256="e" * 64,
        expected_training_config_sha256="1" * 64,
        expected_checkpoint_selection_sha256="3" * 64,
        expected_seed=11, expected_scientific_scope="DEVELOPMENT_ONLY_SINGLE_STUDY",
        expected_method_id="A_EXACT_POTENTIAL_DIFFERENCE",
        expected_training_variant_id="PRIMARY",
        expected_estimand_binding_sha256="f" * 64,
        expected_information_set_fingerprint="2" * 64,
        expected_partition_row_set_sha256=checkpoint_bindings()["partition_row_set_sha256"],
        expected_state_shapes=state_shapes,
    )
    nonfinite = copy.deepcopy(loaded)
    first_key = next(iter(nonfinite["state_dict"]))
    if nonfinite["state_dict"][first_key].is_floating_point():
        nonfinite["state_dict"][first_key].view(-1)[0] = float("nan")
        with pytest.raises(CheckpointProtocolError, match="non-finite"):
            validate_checkpoint(
                nonfinite,
                expected_architecture_id=model.architecture_id,
                expected_authority_hashes=hashes,
            )
