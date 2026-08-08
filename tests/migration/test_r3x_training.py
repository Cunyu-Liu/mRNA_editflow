from __future__ import annotations

import pytest
import torch

from scripts.r3x.train import (
    TrainingProtocolError,
    apply_condition_unknown_dropout,
    balanced_epoch_indices,
    effect_loss,
    fixed_train_scale_sign_candidate_loss,
    frozen_train_sign_scale,
    grouped_pairwise_rank_loss,
    hard_sign,
    normal_sign_probability,
    select_checkpoint,
)


def test_probit_probability_and_hard_sign_are_coherent() -> None:
    mu = torch.tensor([-1.0, 0.0, 1.0])
    logvar = torch.zeros(3)
    prob = normal_sign_probability(mu, logvar, temperature=1.0)
    assert torch.all(prob[1:] >= prob[:-1])
    assert prob[1].item() == pytest.approx(0.5)
    assert hard_sign(mu).tolist() == [-1.0, 1.0, 1.0]
    assert hard_sign(mu).tolist() == hard_sign(mu, temperature=99.0).tolist()


def test_sign_bce_cannot_game_variance_gradient() -> None:
    mu = torch.tensor([0.2, -0.2], requires_grad=True)
    logvar = torch.tensor([0.0, 0.0], requires_grad=True)
    out = effect_loss(
        {"mean": mu, "logvar": logvar},
        torch.tensor([1.0, -1.0]),
        huber_weight=0.0,
        nll_weight=0.0,
        sign_weight=1.0,
    )
    out["loss"].backward()
    assert mu.grad is not None and torch.any(mu.grad != 0)
    assert logvar.grad is None or torch.all(logvar.grad == 0)


def test_effect_loss_keeps_hard_sign_on_mean() -> None:
    outputs = {
        "mean": torch.tensor([0.5, -0.5], requires_grad=True),
        "logvar": torch.tensor([0.0, 0.0], requires_grad=True),
    }
    out = effect_loss(outputs, torch.tensor([1.0, -1.0]))
    assert torch.isfinite(out["loss"])
    assert out["sign_decision"].tolist() == [1.0, -1.0]
    assert out["sign_decision_rule"] == "RAW_PREDICTED_DELTA_MEAN_GE_ZERO"


def test_fixed_train_scale_sign_candidate_is_train_bound_and_keeps_hard_sign() -> None:
    train_target = torch.tensor([-4.0, -1.0, 0.0, 2.0, 8.0])
    scale = frozen_train_sign_scale(train_target)
    assert scale == pytest.approx(2.0)
    mean = torch.tensor([-100.0, 100.0], requires_grad=True)
    logvar = torch.zeros(2, requires_grad=True)
    out = fixed_train_scale_sign_candidate_loss(
        {"mean": mean, "logvar": logvar},
        torch.tensor([1.0, -1.0]),
        frozen_train_scale=scale,
        huber_weight=0.0,
        nll_weight=0.0,
        sign_weight=1.0,
    )
    out["loss"].backward()
    assert torch.all(torch.abs(mean.grad) > 0.0)
    assert out["sign_decision"].tolist() == [-1.0, 1.0]
    assert out["sign_accuracy_gate"] == 0.60
    assert out["candidate_status"] == "PROPOSED_NOT_ACTIVE_DEVELOPMENT_ONLY"
    assert out["sign_link"] == "FIXED_TRAIN_MEDIAN_ABS_NONZERO_DELTA_LOGIT"


def test_fixed_train_scale_candidate_rejects_unfrozen_or_zero_scale() -> None:
    with pytest.raises(TrainingProtocolError, match="non-zero train effects"):
        frozen_train_sign_scale(torch.zeros(3))
    with pytest.raises(TrainingProtocolError, match="frozen_train_scale"):
        fixed_train_scale_sign_candidate_loss(
            {"mean": torch.zeros(1), "logvar": torch.zeros(1)},
            torch.ones(1),
            frozen_train_scale=0.0,
        )


def test_rank_loss_is_on_mean_and_fails_closed_without_policy() -> None:
    mu = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
    y = torch.tensor([0.0, 2.0, 1.0])
    groups = ["g", "g", "other"]
    with pytest.raises(TrainingProtocolError, match="PUBLICATION_QUALIFIED"):
        grouped_pairwise_rank_loss(mu, y, groups, enabled=True, policy_status="DEV_AVAILABLE")
    loss, count = grouped_pairwise_rank_loss(
        mu, y, groups, enabled=True, policy_status="PUBLICATION_QUALIFIED"
    )
    assert count == 1
    loss.backward()
    assert mu.grad is not None and torch.any(mu.grad[:2] != 0)


def test_balanced_epoch_indices_equalize_cells_and_components() -> None:
    rows = []
    for i in range(20):
        rows.append({"source_component_id": "dominant", "study": "s1", "endpoint": "e1", "publication_context_id": "c", "edit_count": 1})
    rows.append({"source_component_id": "rare", "study": "s1", "endpoint": "e1", "publication_context_id": "c", "edit_count": 1})
    rows.append({"source_component_id": "other_cell", "study": "s2", "endpoint": "e2", "publication_context_id": "d", "edit_count": 1})
    a = balanced_epoch_indices(rows, seed=23)
    b = balanced_epoch_indices(rows, seed=23)
    assert a == b
    selected = [rows[i] for i in a]
    by_cell = {}
    for r in selected:
        cell = (r["study"], r["endpoint"], r["publication_context_id"], r["edit_count"])
        by_cell[cell] = by_cell.get(cell, 0) + 1
    assert len(set(by_cell.values())) == 1
    first = [r["source_component_id"] for r in selected if r["study"] == "s1"]
    assert abs(first.count("dominant") - first.count("rare")) <= 1
    assert len(a) <= len(rows) + len(by_cell)


def test_condition_unknown_dropout_is_deterministic_and_trains_unk_rows() -> None:
    endpoint = torch.tensor([1, 1, 2, 2], dtype=torch.long)
    context = torch.tensor([1, 2, 1, 2], dtype=torch.long)
    first = apply_condition_unknown_dropout(endpoint, context, rate=0.15, seed=23)
    second = apply_condition_unknown_dropout(endpoint, context, rate=0.15, seed=23)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert first[2] == second[2]
    assert first[2]["n_endpoint_masked"] >= 1
    assert first[2]["n_context_masked"] >= 1
    assert torch.any(first[0] == 0)
    assert torch.any(first[1] == 0)


def test_checkpoint_selection_is_lexicographic_and_not_seed_cherry_picking() -> None:
    candidates = [
        {"epoch": 1, "macro_sign_accuracy": 0.59, "macro_delta_spearman": 0.9, "macro_gaussian_nll": 0.1},
        {"epoch": 2, "macro_sign_accuracy": 0.61, "macro_delta_spearman": 0.3, "macro_gaussian_nll": 0.5},
        {"epoch": 3, "macro_sign_accuracy": 0.60, "macro_delta_spearman": 0.4, "macro_gaussian_nll": 0.7},
        {"epoch": 4, "macro_sign_accuracy": 0.62, "macro_delta_spearman": 0.4, "macro_gaussian_nll": 0.6},
        {"epoch": 5, "macro_sign_accuracy": 0.70, "macro_delta_spearman": 0.4, "macro_gaussian_nll": 0.5},
    ]
    out = select_checkpoint(candidates, sign_gate=0.60)
    assert out["selected_epoch"] == 5
    assert out["status"] == "DEV_SIGN_GATE_MET"
    assert out["selection_rule"] == "SIGN_GATE_THEN_SPEARMAN_THEN_NLL_THEN_EARLIEST_EPOCH"


def test_checkpoint_selection_below_gate_reports_hold() -> None:
    out = select_checkpoint([
        {"epoch": 1, "macro_sign_accuracy": 0.51, "macro_delta_spearman": 0.2, "macro_gaussian_nll": 1.0},
        {"epoch": 2, "macro_sign_accuracy": 0.55, "macro_delta_spearman": 0.1, "macro_gaussian_nll": 0.9},
    ])
    assert out["selected_epoch"] == 2
    assert out["status"] == "HOLD_DEV_SIGN_GATE_NOT_MET"
